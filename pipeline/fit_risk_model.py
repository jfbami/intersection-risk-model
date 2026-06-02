"""
Fit Negative Binomial SPFs (Safety Performance Functions) for three crash modes
at Capitol Hill arterial intersections.

Follows AASHTO Highway Safety Manual Chapter 12 methodology: NB2 family,
log link, log(AADT) as the volume predictor, offset = log(years_observed)
for the 6-year exposure window. Three independent fits — one each for
bike, pedestrian, and motor-vehicle-only crash counts — share the same
predictors so coefficients can be compared across modes.

Mode-specific KSI rates are derived downstream in `score_risk.py` via
empirical severity shares (citywide KSI events / mode total) — direct
KSI-target fits are infeasible at our event counts (bike-KSI = 16,
ped-KSI = 32, vehicle-only-KSI = 23; all below the ~10 events / parameter
threshold for stable NB MLE).

Scope
-----
Arterial intersections only (arterial_class >= 1) with positive AADT. Local
streets are excluded per HSM Chapter 12 facility-type stratification.

Inputs
------
data/intermediate/intersection_features.parquet
data/intermediate/crashes_by_intersection.parquet

Outputs
-------
data/intermediate/intersection_predictions.parquet
    One row per modelled arterial intersection with mode-suffixed columns:
        {mode}_expected_total, {mode}_expected_per_year,
        {mode}_actual, {mode}_residual           for mode in {bike, ped, vehicle}
    Plus severity carry-through and fitted_at timestamp.

data/model/nb_v3_{mode}.pkl                       for mode in {bike, ped, vehicle}
    Three serialized statsmodels NegativeBinomialResults objects.
"""

import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from statsmodels.tools.sm_exceptions import ConvergenceWarning

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from pipeline.columns import SEVERITY_COLUMNS
from pipeline.feature_encoding import (
    LEG_CATEGORY_COLUMN,
    LEG_CATEGORY_TERM,
    leg_category,
)

ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH    = ROOT / "data" / "intermediate" / "intersection_features.parquet"
CRASHES_PATH     = ROOT / "data" / "intermediate" / "crashes_by_intersection.parquet"
PREDICTIONS_PATH = ROOT / "data" / "intermediate" / "intersection_predictions.parquet"
MODEL_DIR        = ROOT / "data" / "model"

EXPECTED_RAW_ROWS = 651
CALIBRATION_THRESHOLD_PCT = 15.0

# Shared predictors across all three mode models. Identical formula keeps
# coefficients comparable; the volume proxy varies per fit. Leg count is a
# top-coded categorical (see feature_encoding) rather than a continuous slope,
# to avoid extrapolating rare 5+/6-leg geometries to implausible risk.
SHARED_PREDICTORS = (
    f"is_signalized + {LEG_CATEGORY_TERM} + max_speed_limit"
    " + bike_facility + C(arterial_class)"
)


@dataclass(frozen=True)
class ModeSpec:
    label:        str   # short tag used in column suffixes and filenames
    target:       str   # column name in the joined dataframe
    display_name: str   # human-readable for diagnostics
    predictors:   str   # predictor formula string


MODES: tuple[ModeSpec, ...] = (
    ModeSpec("bike",    "bike_total",         "Bicycle",            f"{SHARED_PREDICTORS} + log_bike_centrality"),
    ModeSpec("ped",     "ped_total",          "Pedestrian",         f"{SHARED_PREDICTORS} + log_aadt"),
    ModeSpec("vehicle", "vehicle_only_total", "Motor vehicle only", f"{SHARED_PREDICTORS} + log_aadt"),
)


# ---------------------------------------------------------------------------
# Load + join
# ---------------------------------------------------------------------------

def load_and_join() -> pd.DataFrame:
    """Load features and crash counts; inner-join on intersection_id."""
    missing = []
    if not FEATURES_PATH.exists():
        missing.append(f"  {FEATURES_PATH}  ->  run: python -m pipeline.assemble_features")
    if not CRASHES_PATH.exists():
        missing.append(f"  {CRASHES_PATH}  ->  run: python -m pipeline.snap_crashes")
    if missing:
        sys.exit("[ERROR] Missing required inputs:\n" + "\n".join(missing))

    features = pd.read_parquet(FEATURES_PATH)
    crashes  = pd.read_parquet(CRASHES_PATH)
    df = features.merge(crashes, on="intersection_id", how="inner")

    if len(df) != EXPECTED_RAW_ROWS:
        _exit_with_join_diagnostic(features, crashes, df)
    return df


def _exit_with_join_diagnostic(features: pd.DataFrame, crashes: pd.DataFrame, df: pd.DataFrame) -> None:
    feat_ids  = set(features["intersection_id"])
    crash_ids = set(crashes["intersection_id"])
    msg = f"[ERROR] Inner join produced {len(df)} rows, expected {EXPECTED_RAW_ROWS}.\n"
    only_feat  = feat_ids - crash_ids
    only_crash = crash_ids - feat_ids
    if only_feat:
        msg += f"  In features but not crashes ({len(only_feat)}): {sorted(only_feat)[:5]} ...\n"
    if only_crash:
        msg += f"  In crashes but not features ({len(only_crash)}): {sorted(only_crash)[:5]} ...\n"
    sys.exit(msg)


# ---------------------------------------------------------------------------
# Prepare
# ---------------------------------------------------------------------------

def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply scope filter, derive log_aadt + offset + vehicle_only_total."""
    df = df.copy()
    n_raw = len(df)

    df, scope_stats = _restrict_to_modellable_arterials(df)
    df["log_aadt"]            = np.log(df["max_aadt"])
    if "bike_centrality" in df.columns:
        df["log_bike_centrality"] = np.log(df["bike_centrality"])
    df["offset"]              = np.log(df["years_observed"])
    df[LEG_CATEGORY_COLUMN]   = df["num_legs"].map(leg_category)

    bad_obs = df[df["years_observed"] != 6][["intersection_id", "years_observed"]]

    _assert_no_unexpected_nan(df)
    _assert_vehicle_only_present(df)

    n_speed_nan = int(df["max_speed_limit"].isna().sum())
    median_speed = None
    if n_speed_nan:
        median_speed = df["max_speed_limit"].median()
        df["max_speed_limit"] = df["max_speed_limit"].fillna(median_speed)

    return df, {
        "n_raw":             n_raw,
        "n_after_scope":     len(df),
        "n_dropped_local":   scope_stats["n_dropped_local"],
        "n_dropped_no_aadt": scope_stats["n_dropped_no_aadt"],
        "n_speed_nan":       n_speed_nan,
        "median_speed":      median_speed,
        "bad_obs":           bad_obs,
    }


def _restrict_to_modellable_arterials(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    is_local        = df["arterial_class"] < 1
    has_usable_aadt = df["max_aadt"].notna() & (df["max_aadt"] > 0)
    has_centrality  = df["bike_centrality"].notna() if "bike_centrality" in df.columns else True

    n_dropped_local   = int(is_local.sum())
    n_dropped_no_aadt = int((~is_local & ~has_usable_aadt).sum())

    keep = ~is_local & has_usable_aadt & has_centrality
    return df[keep].copy(), {
        "n_dropped_local":   n_dropped_local,
        "n_dropped_no_aadt": n_dropped_no_aadt,
    }


def _assert_no_unexpected_nan(df: pd.DataFrame) -> None:
    must_be_clean = ["is_signalized", "num_legs", "is_arterial", "arterial_class", "bike_facility"]
    dirty = {
        col: df.loc[df[col].isna(), "intersection_id"].tolist()
        for col in must_be_clean if df[col].isna().any()
    }
    if not dirty:
        return
    msg = (
        "[ERROR] Unexpected NaN in features that were clean after assemble_features.py.\n"
        "This indicates a regression upstream — fix assemble_features.py and re-run.\n"
    )
    for col, ids in dirty.items():
        msg += f"  {col}: {ids}\n"
    sys.exit(msg)


def _assert_vehicle_only_present(df: pd.DataFrame) -> None:
    if "vehicle_only_total" not in df.columns:
        sys.exit(
            "[ERROR] vehicle_only_total column missing from features+crashes join. "
            "Re-run: python -m pipeline.snap_crashes"
        )


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def fit_for_mode(df: pd.DataFrame, mode: ModeSpec):
    """Fit NB2 against this mode's target column. Returns fitted result."""
    formula = f"{mode.target} ~ {mode.predictors}"
    model   = smf.negativebinomial(formula, data=df, offset=df["offset"].values)

    result, warns = _fit_with_warning_capture(model)
    if _is_converged(result, warns):
        return result

    result, warns = _fit_with_warning_capture(model, method="bfgs", maxiter=200)
    if _is_converged(result, warns):
        return result

    print(result.summary())
    print(f"\nmle_retvals: {result.mle_retvals}")
    if warns:
        print(f"ConvergenceWarnings: {[str(w.message) for w in warns]}")
    sys.exit(f"[ERROR] {mode.label} model did not converge after BFGS retry.")


def _fit_with_warning_capture(model, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = model.fit(disp=False, **kwargs)
    conv_warns = [w for w in caught if issubclass(w.category, ConvergenceWarning)]
    return res, conv_warns


def _is_converged(res, conv_warns: list) -> bool:
    return (not conv_warns) and res.mle_retvals.get("converged", True)


# ---------------------------------------------------------------------------
# Validate per mode
# ---------------------------------------------------------------------------

def validate_for_mode(result, df: pd.DataFrame, mode: ModeSpec) -> pd.DataFrame:
    """Return a 4-column predictions frame keyed by intersection_id for this mode."""
    expected_total = result.predict(df, offset=df["offset"].values)

    _assert_predictions_nonnegative(expected_total, mode)
    _assert_calibration_within_threshold(expected_total, df[mode.target], mode)

    return pd.DataFrame({
        "intersection_id":             df["intersection_id"].values,
        f"{mode.label}_expected_total":    expected_total.values,
        f"{mode.label}_expected_per_year": (expected_total / df["years_observed"]).values,
        f"{mode.label}_actual":            df[mode.target].astype(int).values,
        f"{mode.label}_residual":          (df[mode.target] - expected_total).values,
    })


def _assert_predictions_nonnegative(expected_total: pd.Series, mode: ModeSpec) -> None:
    n_negative = int((expected_total < 0).sum())
    if not n_negative:
        return
    sys.exit(
        f"[ERROR] {mode.label}: {n_negative} expected_total values are negative — "
        "prediction is not on the response scale."
    )


def _assert_calibration_within_threshold(
    expected_total: pd.Series, actual: pd.Series, mode: ModeSpec
) -> None:
    sum_pred   = float(expected_total.sum())
    sum_actual = int(actual.sum())
    if sum_actual == 0:
        return
    pct_diff = abs(sum_pred - sum_actual) / sum_actual * 100
    if pct_diff <= CALIBRATION_THRESHOLD_PCT:
        return
    sys.exit(
        f"[ERROR] {mode.label} calibration failed: sum(predicted)={sum_pred:.1f} vs "
        f"sum(actual)={sum_actual} ({pct_diff:.1f}% gap > {CALIBRATION_THRESHOLD_PCT}% threshold)."
    )


# ---------------------------------------------------------------------------
# Combined output
# ---------------------------------------------------------------------------

def combine_predictions(df: pd.DataFrame, per_mode: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge severity carry-through with each mode's predictions into one frame."""
    severity_cols = [c for c in SEVERITY_COLUMNS if c in df.columns]
    out = df[["intersection_id", "years_observed"] + severity_cols].copy()

    for predictions in per_mode.values():
        out = out.merge(predictions, on="intersection_id", how="left")

    return out


def write_outputs(
    fitted: dict[str, "smf.OLS"],  # NegativeBinomialResults isn't easily typed
    predictions: pd.DataFrame,
) -> None:
    predictions = predictions.copy()
    predictions["fitted_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(PREDICTIONS_PATH, index=False)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for label, result in fitted.items():
        result.save(str(MODEL_DIR / f"nb_v3_{label}.pkl"))


# ---------------------------------------------------------------------------
# Diagnostic printers
# ---------------------------------------------------------------------------

def print_preparation_diagnostics(stats: dict) -> None:
    print(
        f"Scope filter:\n"
        f"  raw rows                   {stats['n_raw']}\n"
        f"  dropped: non-arterial      {stats['n_dropped_local']}\n"
        f"  dropped: missing AADT      {stats['n_dropped_no_aadt']}\n"
        f"  rows used for fit          {stats['n_after_scope']}"
    )
    bad_obs = stats["bad_obs"]
    if len(bad_obs):
        print(f"\n[WARN] {len(bad_obs)} rows have years_observed != 6:")
        print(bad_obs.to_string(index=False))
    else:
        print("\nyears_observed: all rows == 6.  Good.")

    if stats["n_speed_nan"]:
        print(
            f"[WARN] Filled {stats['n_speed_nan']} NaN in max_speed_limit with median "
            f"({stats['median_speed']}). Unexpected — check assemble_features.py."
        )
    else:
        print("max_speed_limit: fully populated (0 NaN). Good.")


def extract_alpha(result) -> Optional[float]:
    if "alpha" in result.params.index:
        return float(result.params["alpha"])
    if hasattr(result, "alpha"):
        return float(result.alpha)
    return None


def print_mode_summary(mode: ModeSpec, result, predictions: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print(f"  {mode.display_name} model  (target: {mode.target})")
    print("=" * 70)
    print(result.summary())

    alpha = extract_alpha(result)
    if alpha is not None:
        verdict = "overdispersed (NB correct)" if alpha > 0.05 else "near-Poisson"
        print(f"\nalpha = {alpha:.4f}   ({verdict})")

    pred_col   = f"{mode.label}_expected_total"
    actual_col = f"{mode.label}_actual"
    sum_pred   = float(predictions[pred_col].sum())
    sum_actual = int(predictions[actual_col].sum())
    pct_diff   = 100.0 * (sum_pred - sum_actual) / sum_actual if sum_actual else 0.0
    mae        = float(predictions[f"{mode.label}_residual"].abs().mean())
    print(f"Calibration: sum_pred={sum_pred:.1f} vs sum_actual={sum_actual} ({pct_diff:+.1f}%)")
    print(f"MAE: {mae:.2f} {mode.label} crashes per intersection (2018–2023)")


def print_cross_mode_coefficients(fitted: dict) -> None:
    """Side-by-side view of how each predictor moves expected crashes per mode."""
    print("\n" + "=" * 70)
    print("  Cross-mode coefficient comparison  (beta; positive => more crashes)")
    print("=" * 70)
    rows: list[dict] = []
    all_terms: set[str] = set()
    for label, result in fitted.items():
        for term in result.params.index:
            all_terms.add(term)
    for term in sorted(all_terms):
        row = {"term": term}
        for label, result in fitted.items():
            row[label] = result.params.get(term, float("nan"))
        rows.append(row)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def main() -> None:
    df = load_and_join()
    print(f"Joined dataset: {len(df)} rows (expect {EXPECTED_RAW_ROWS}).\n")

    df, prep_stats = prepare(df)
    print_preparation_diagnostics(prep_stats)

    fitted:        dict[str, object]       = {}
    per_mode_pred: dict[str, pd.DataFrame] = {}

    for mode in MODES:
        print(f"\nFitting {mode.display_name} model "
              f"(target: {mode.target}, n_events: {int(df[mode.target].sum())})...")
        result = fit_for_mode(df, mode)
        per_mode_pred[mode.label] = validate_for_mode(result, df, mode)
        fitted[mode.label]        = result
        print(f"  Convergence: OK")

    predictions = combine_predictions(df, per_mode_pred)

    for mode in MODES:
        print_mode_summary(mode, fitted[mode.label], predictions)

    print_cross_mode_coefficients(fitted)

    write_outputs(fitted, predictions)
    print(f"\nWrote predictions -> {PREDICTIONS_PATH}")
    for label in fitted:
        print(f"Saved model        -> {MODEL_DIR / f'nb_v3_{label}.pkl'}")


if __name__ == "__main__":
    main()
