"""
EB-adjust the three per-mode NB SPF outputs and emit per-intersection
risk fields for the bike, ped, and vehicle-only modes, plus a composite
all-mode KSI estimate.

For each mode the pipeline is:
  1. NB crash-count prediction (from the mode's pkl) →
  2. AASHTO-style Empirical Bayes shrinkage on the mode crash count
     (w = 1 / (1 + α·μ); eb = w·μ + (1−w)·N) →
  3. Mode-KSI prior: μ_KSI = μ_crashes × citywide_mode_KSI_share →
  4. Direct EB on the mode-KSI count: posterior Gamma(k + N_KSI, k/μ_KSI + 1),
     yielding mean + 90% credible interval per-intersection.

The composite "all-mode KSI" is the sum of the three mode-KSI posteriors —
a slight overstatement because some crashes are flagged as both bike AND
ped (2 of 71 KSI events citywide, 2.8% inflation). This is the Phase-2
trade-off; resolving requires per-crash mode resolution that the snap
pipeline doesn't currently surface.

Inputs
------
data/intermediate/intersection_predictions.parquet  — per-mode NB predictions
data/intermediate/intersection_features.parquet     — per-intersection features
data/model/nb_v3_bike.pkl, nb_v3_ped.pkl, nb_v3_vehicle.pkl — fitted NB results

Output
------
data/intermediate/intersection_scores.parquet
    Per-intersection: per-mode EB crash counts and KSI estimates with 90% CIs,
    composite all-mode KSI, per-mode contributors and recommendations, plus
    backwards-compat bike-headline aliases the current frontend consumes.
"""

import json
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from scipy import stats

from pipeline.columns import SEVERITY_COLUMNS
from pipeline.contributors import compute_for_row_as_dicts
from pipeline.treatments import (
    PredictionInterval,
    Treatment,
    load_treatments,
    rank_as_dicts,
)

ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_PATH = ROOT / "data" / "intermediate" / "intersection_predictions.parquet"
FEATURES_PATH    = ROOT / "data" / "intermediate" / "intersection_features.parquet"
OUT_PATH         = ROOT / "data" / "intermediate" / "intersection_scores.parquet"
MODEL_DIR        = ROOT / "data" / "model"

YEARS_OBSERVED = 6
CREDIBLE_LEVEL = 0.90
TIER_CUTS      = [(90, "very_high"), (70, "high"), (40, "moderate"), (20, "low")]


class AlphaUnavailableError(RuntimeError):
    """NB dispersion alpha could not be recovered from the saved model."""


@dataclass(frozen=True)
class FittedModel:
    alpha:  float
    params: dict


@dataclass(frozen=True)
class ModeSpec:
    label:           str   # "bike" | "ped" | "vehicle"
    crash_actual:    str   # column with observed crash count
    crash_predicted: str   # column with NB-predicted crash total
    ksi_actual:      str   # column with observed mode-KSI count


MODES: tuple[ModeSpec, ...] = (
    ModeSpec("bike",    "bike_actual",    "bike_expected_total",    "bike_ksi_total"),
    ModeSpec("ped",     "ped_actual",     "ped_expected_total",     "ped_ksi_total"),
    ModeSpec("vehicle", "vehicle_actual", "vehicle_expected_total", "vehicle_only_ksi_total"),
)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_predictions() -> pd.DataFrame:
    if not PREDICTIONS_PATH.exists():
        sys.exit(
            f"[ERROR] {PREDICTIONS_PATH} not found.\n"
            "Run:  python -m pipeline.fit_risk_model"
        )
    return pd.read_parquet(PREDICTIONS_PATH)


def load_features() -> pd.DataFrame:
    if not FEATURES_PATH.exists():
        sys.exit(
            f"[ERROR] {FEATURES_PATH} not found.\n"
            "Run:  python -m pipeline.assemble_features"
        )
    return pd.read_parquet(FEATURES_PATH)


def load_fitted_models() -> dict[str, FittedModel]:
    """Returns {mode_label: FittedModel} for bike, ped, vehicle."""
    return {mode.label: _load_mode_model(mode.label) for mode in MODES}


def _load_mode_model(label: str) -> FittedModel:
    path = MODEL_DIR / f"nb_v3_{label}.pkl"
    if not path.exists():
        raise AlphaUnavailableError(
            f"{path} not found. Run: python -m pipeline.fit_risk_model"
        )
    with open(path, "rb") as f:
        result = pickle.load(f)

    alpha = _extract_alpha(result)
    if alpha is None or alpha <= 0:
        raise AlphaUnavailableError(
            f"{path}: saved NB model does not expose a positive 'alpha' attribute. "
            "Re-fit with the current statsmodels version."
        )
    params = {name: float(value) for name, value in result.params.items()}
    return FittedModel(alpha=alpha, params=params)


def _extract_alpha(result) -> Optional[float]:
    if hasattr(result, "params") and "alpha" in getattr(result.params, "index", []):
        return float(result.params["alpha"])
    if hasattr(result, "alpha"):
        return float(result.alpha)
    return None


# ---------------------------------------------------------------------------
# EB on a mode's crash count
# ---------------------------------------------------------------------------

def compute_eb_estimate(
    predicted: pd.Series, observed: pd.Series, alpha: float
) -> pd.Series:
    """AASHTO HSM Part C Empirical Bayes shrinkage on a count.

        w  = 1 / (1 + α·predicted)
        eb = w·predicted + (1−w)·observed

    Pure math primitive — no DataFrame/column-name coupling.
    """
    weight = 1.0 / (1.0 + alpha * predicted)
    return weight * predicted + (1.0 - weight) * observed


def compute_mode_crash_eb(
    predictions: pd.DataFrame, mode: ModeSpec, alpha: float
) -> pd.DataFrame:
    """Apply EB shrinkage to one mode's crash count."""
    predicted = predictions[mode.crash_predicted].astype(float)
    observed  = predictions[mode.crash_actual].astype(float)

    if predicted.isna().any() or (predicted < 0).any():
        sys.exit(
            f"[ERROR] {mode.label}: predicted column has null or negative values. "
            "Re-run: python -m pipeline.fit_risk_model"
        )

    eb_count = compute_eb_estimate(predicted, observed, alpha)

    return pd.DataFrame({
        "intersection_id":           predictions["intersection_id"].values,
        f"{mode.label}_eb_count":    eb_count.values,
        f"{mode.label}_eb_per_year": (eb_count / YEARS_OBSERVED).values,
    })


# ---------------------------------------------------------------------------
# Mode-KSI: prior + Poisson-Gamma EB with 90% CI
# ---------------------------------------------------------------------------

def citywide_mode_ksi_share(predictions: pd.DataFrame, mode: ModeSpec) -> float:
    total_crashes = float(predictions[mode.crash_actual].sum())
    total_ksi     = float(predictions[mode.ksi_actual].sum())
    if total_crashes <= 0:
        return 0.0
    return total_ksi / total_crashes


def compute_mode_ksi_eb(
    predictions: pd.DataFrame,
    mode: ModeSpec,
    alpha: float,
    level: float = CREDIBLE_LEVEL,
) -> pd.DataFrame:
    """Direct EB on mode-KSI counts. Prior = NB crash prediction × citywide share."""
    tail       = (1 - level) / 2
    k          = 1.0 / alpha
    city_share = citywide_mode_ksi_share(predictions, mode)

    mu_ksi = (predictions[mode.crash_predicted].astype(float) * city_share).clip(lower=1e-9)
    n_ksi  = predictions[mode.ksi_actual].astype(float)

    shape = k + n_ksi
    scale = mu_ksi / (k + mu_ksi)
    posterior_mean = shape * scale
    ci_low  = pd.Series(stats.gamma.ppf(tail,       shape, scale=scale), index=mu_ksi.index)
    ci_high = pd.Series(stats.gamma.ppf(1.0 - tail, shape, scale=scale), index=mu_ksi.index)

    return pd.DataFrame({
        "intersection_id":             predictions["intersection_id"].values,
        f"{mode.label}_ksi_eb_count":  posterior_mean.values,
        f"{mode.label}_ksi_per_year":  (posterior_mean / YEARS_OBSERVED).values,
        f"{mode.label}_ksi_ci_low":    (ci_low  / YEARS_OBSERVED).values,
        f"{mode.label}_ksi_ci_high":   (ci_high / YEARS_OBSERVED).values,
    })


# ---------------------------------------------------------------------------
# Composite all-mode KSI
# ---------------------------------------------------------------------------

def compute_composite_all_mode_ksi(scores: pd.DataFrame) -> pd.DataFrame:
    """Sum of per-mode KSI rates with pessimistic CI combination.

    Slight overstatement at the citywide level (~2.8%) because a few crashes
    are flagged as BOTH bike AND ped — they contribute to both bike_ksi and
    ped_ksi components. Documented limitation of the Phase-2 decomposition.
    """
    all_per_year = sum(scores[f"{m.label}_ksi_per_year"] for m in MODES)
    all_ci_low   = sum(scores[f"{m.label}_ksi_ci_low"]   for m in MODES)
    all_ci_high  = sum(scores[f"{m.label}_ksi_ci_high"]  for m in MODES)
    return pd.DataFrame({
        "intersection_id":   scores["intersection_id"].values,
        "all_ksi_per_year":  all_per_year.values,
        "all_ksi_ci_low":    all_ci_low.values,
        "all_ksi_ci_high":   all_ci_high.values,
    })


# ---------------------------------------------------------------------------
# Per-mode contributors
# ---------------------------------------------------------------------------

def compute_mode_contributors_column(
    features: pd.DataFrame, params: dict
) -> pd.Series:
    encoded = features.apply(
        lambda row: json.dumps(compute_for_row_as_dicts(row, params, top_n=3)),
        axis=1,
    )
    return encoded


# ---------------------------------------------------------------------------
# Recommendations (bike-only for now; ped/vehicle empty until task 36)
# ---------------------------------------------------------------------------

def compute_treatment_recommendations(
    scores: pd.DataFrame, features: pd.DataFrame, treatments: list[Treatment], mode: str
) -> pd.Series:
    features_by_id = features.set_index("intersection_id").to_dict(orient="index")

    def _per_row(row: pd.Series) -> str:
        feat = features_by_id.get(row["intersection_id"], {})
        prediction = PredictionInterval(
            mean    = float(row[f"{mode}_ksi_per_year"]),
            ci_low  = float(row[f"{mode}_ksi_ci_low"]),
            ci_high = float(row[f"{mode}_ksi_ci_high"]),
        )
        return json.dumps(rank_as_dicts(feat, prediction, treatments, mode, top_n=3))

    return scores.apply(_per_row, axis=1)


# ---------------------------------------------------------------------------
# Rank / tier
# ---------------------------------------------------------------------------

def _percentile_rank(series: pd.Series) -> pd.Series:
    return series.rank(pct=True, method="average", na_option="keep") * 100


def _assign_tier(score: float) -> str:
    for cut, label in TIER_CUTS:
        if score >= cut:
            return label
    return "very_low"


def attach_rank_and_tier(scores: pd.DataFrame) -> pd.DataFrame:
    """Rank by bike_ksi_per_year for now (Task 37 switches to all_ksi_per_year)."""
    scores = scores.copy()
    scores["expected_percentile"] = _percentile_rank(scores["bike_ksi_per_year"])
    scores["all_mode_percentile"] = _percentile_rank(scores["all_ksi_per_year"])
    scores["bike_percentile"]     = scores["expected_percentile"]
    scores["ped_percentile"]      = _percentile_rank(scores["ped_ksi_per_year"])
    scores["vehicle_percentile"]  = _percentile_rank(scores["vehicle_ksi_per_year"])
    scores["risk_score"]          = scores["expected_percentile"]
    scores["risk_rank"]           = scores["risk_score"].rank(method="dense", ascending=False).astype(int)
    scores["risk_tier"]           = scores["risk_score"].apply(_assign_tier)
    return scores


# ---------------------------------------------------------------------------
# Scoring composition
# ---------------------------------------------------------------------------

def compute_scores(
    predictions: pd.DataFrame,
    features:    pd.DataFrame,
    fitted:      dict[str, FittedModel],
) -> pd.DataFrame:
    """Build per-mode + composite scoring frame for all modelled arterials."""
    severity_carry = [c for c in SEVERITY_COLUMNS if c in predictions.columns]
    scores = predictions[["intersection_id"] + severity_carry].copy()

    for mode in MODES:
        crash_eb = compute_mode_crash_eb(predictions, mode, fitted[mode.label].alpha)
        ksi_eb   = compute_mode_ksi_eb(predictions, mode, fitted[mode.label].alpha)
        scores = scores.merge(crash_eb, on="intersection_id", how="left")
        scores = scores.merge(ksi_eb,   on="intersection_id", how="left")

    composite = compute_composite_all_mode_ksi(scores)
    scores = scores.merge(composite, on="intersection_id", how="left")

    scores = attach_rank_and_tier(scores)
    scores = _attach_per_mode_contributors(scores, features, fitted)
    scores = _attach_treatment_recommendations(scores, features)
    scores = _attach_backwards_compat_aliases(scores)

    return _project_output_columns(scores, severity_carry)


def _attach_per_mode_contributors(
    scores: pd.DataFrame, features: pd.DataFrame, fitted: dict[str, FittedModel]
) -> pd.DataFrame:
    feat_with_id = features[["intersection_id"]].copy()
    for mode in MODES:
        feat_with_id[f"{mode.label}_top_contributors"] = compute_mode_contributors_column(
            features, fitted[mode.label].params
        ).values
    return scores.merge(feat_with_id, on="intersection_id", how="left")


def _attach_treatment_recommendations(
    scores: pd.DataFrame, features: pd.DataFrame
) -> pd.DataFrame:
    treatments = load_treatments()
    scores = scores.copy()
    scores["bike_recommended_treatments"]    = compute_treatment_recommendations(scores, features, treatments, "bike")
    scores["ped_recommended_treatments"]     = compute_treatment_recommendations(scores, features, treatments, "ped")
    scores["vehicle_recommended_treatments"] = compute_treatment_recommendations(scores, features, treatments, "vehicle")
    return scores


def _attach_backwards_compat_aliases(scores: pd.DataFrame) -> pd.DataFrame:
    """Keep the bike-headline field names the current frontend consumes."""
    scores = scores.copy()
    scores["expected_bike_ksi_per_year"] = scores["bike_ksi_per_year"]
    scores["expected_bike_ksi_ci_low"]   = scores["bike_ksi_ci_low"]
    scores["expected_bike_ksi_ci_high"]  = scores["bike_ksi_ci_high"]
    scores["top_contributors"]           = scores["bike_top_contributors"]
    scores["recommended_treatments"]     = scores["bike_recommended_treatments"]
    return scores


def _project_output_columns(scores: pd.DataFrame, severity_carry: list[str]) -> pd.DataFrame:
    per_mode_cols = [
        f"{m.label}_{suffix}"
        for m in MODES
        for suffix in (
            "eb_count", "eb_per_year",
            "ksi_eb_count", "ksi_per_year", "ksi_ci_low", "ksi_ci_high",
            "top_contributors", "recommended_treatments", "percentile",
        )
    ]
    return scores[[
        "intersection_id",
        # backwards-compat bike headline aliases (current frontend)
        "expected_bike_ksi_per_year",
        "expected_bike_ksi_ci_low",
        "expected_bike_ksi_ci_high",
        "top_contributors",
        "recommended_treatments",
        # composite all-mode KSI (Task 37 will surface as headline)
        "all_ksi_per_year", "all_ksi_ci_low", "all_ksi_ci_high",
        # rank + tier
        "risk_score", "risk_rank", "risk_tier",
        "expected_percentile", "all_mode_percentile",
    ] + per_mode_cols + severity_carry]


def write_output(scores: pd.DataFrame) -> None:
    scores = scores.copy()
    scores["model_version"] = "nb_v3_per_mode"
    scores["scored_at"]     = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(OUT_PATH, index=False)


# ---------------------------------------------------------------------------
# Diagnostic printers
# ---------------------------------------------------------------------------

def print_share_per_mode(predictions: pd.DataFrame) -> None:
    print("\n--- Citywide KSI share per mode ---")
    for mode in MODES:
        share = citywide_mode_ksi_share(predictions, mode)
        n_crash = int(predictions[mode.crash_actual].sum())
        n_ksi   = int(predictions[mode.ksi_actual].sum())
        print(f"  {mode.label:8s}  {n_ksi}/{n_crash} = {share*100:.2f}% of {mode.label} crashes are KSI")


def print_top_sites(scores: pd.DataFrame, n: int = 10) -> None:
    cols = [
        "intersection_id",
        "all_ksi_per_year", "all_ksi_ci_low", "all_ksi_ci_high",
        "bike_ksi_per_year", "ped_ksi_per_year", "vehicle_ksi_per_year",
    ]
    print(f"\n--- Top {n} intersections by all-mode KSI / year ---")
    print(
        scores.nlargest(n, "all_ksi_per_year")[cols]
        .round({c: 4 for c in cols if c != "intersection_id"})
        .to_string(index=False)
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def main() -> None:
    predictions = load_predictions()
    features    = load_features()
    fitted      = load_fitted_models()

    print(f"Loaded predictions for {len(predictions)} sites.")
    for mode in MODES:
        m = fitted[mode.label]
        print(f"  {mode.label:8s}  alpha={m.alpha:.4f}  n_params={len(m.params)}")

    print_share_per_mode(predictions)

    scores = compute_scores(predictions, features, fitted)
    print(f"\nComputed scores: {len(scores)} rows × {len(scores.columns)} columns")

    print_top_sites(scores)

    print(
        "\n[NOTE] risk_score / risk_rank / risk_tier are still based on bike-KSI "
        "percentile to preserve the current frontend; Task 37 switches them to "
        "all_ksi_per_year (composite Vision Zero metric) when the mode-selector "
        "UI lands. all_mode_percentile is already computed and available."
    )

    write_output(scores)
    print(f"\nWrote {len(scores)} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
