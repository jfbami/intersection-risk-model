"""In-sample diagnostics for the three per-mode NB SPFs.

Run after `python -m pipeline.fit_risk_model` to inspect:

  - Coefficient precision per mode (z-stats, p-values, 90% CIs)
  - Variance Inflation Factors — flags multicollinearity > 5
  - Pseudo R², AIC, log-likelihood, LL ratio test
  - Calibration sum_pred vs sum_actual
  - Mean absolute error, root mean squared error, Spearman rank correlation
  - Top positive and negative residuals (edge-case sites)
  - Observed-vs-predicted zero-count comparison (zero-inflation check)
  - Cross-mode residual correlation — proxy for unobserved site heterogeneity

Out-of-sample cross-validation is intentionally out of scope here; it's a
Phase 8 capability that requires refactoring the fit logic to be callable
against arbitrary row subsets.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import patsy
import statsmodels.formula.api as smf
from scipy import stats as scipy_stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

from pipeline.fit_risk_model import (
    DESIGN_PREDICTORS,
    MODES,
    ModeSpec,
    load_and_join,
    prepare,
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR        = ROOT / "data" / "model"
PREDICTIONS_PATH = ROOT / "data" / "intermediate" / "intersection_predictions.parquet"

CI_LEVEL = 0.10  # ⇒ 90% intervals
VIF_FLAG = 5.0


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_fitted_results() -> dict[str, object]:
    """Returns {mode_label: NegativeBinomialResults}."""
    return {m.label: _load_pkl(MODEL_DIR / f"nb_v3_{m.label}.pkl") for m in MODES}


def _load_pkl(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Re-run pipeline.fit_risk_model.")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_modelling_frame() -> pd.DataFrame:
    """Reconstruct the same dataframe the fit ran on (post-scope-filter)."""
    df, _ = prepare(load_and_join())
    return df


# ---------------------------------------------------------------------------
# Coefficient table
# ---------------------------------------------------------------------------

def print_coefficient_table(mode: ModeSpec, result) -> None:
    print(f"\n--- {mode.display_name} model: coefficients ({CI_LEVEL*100:.0f}% CIs) ---")
    ci = result.conf_int(alpha=CI_LEVEL)
    ci.columns = ["ci_low", "ci_high"]
    table = pd.DataFrame({
        "beta":   result.params.round(3),
        "se":     result.bse.round(3),
        "z":      result.tvalues.round(2),
        "p":      result.pvalues.round(3),
        "ci_low":  ci["ci_low"].round(3),
        "ci_high": ci["ci_high"].round(3),
    })
    print(table.to_string())


# ---------------------------------------------------------------------------
# VIF
# ---------------------------------------------------------------------------

def print_vif_table(df: pd.DataFrame, mode: ModeSpec) -> None:
    X = _design_matrix(mode, df)
    vifs = []
    for i, col in enumerate(X.columns):
        if col == "Intercept":
            continue
        vif = variance_inflation_factor(X.values, i)
        flag = "⚠ HIGH" if vif > VIF_FLAG else ""
        vifs.append({"predictor": col, "vif": round(vif, 2), "flag": flag})
    print(f"\n--- {mode.display_name} VIF (>{VIF_FLAG} flags multicollinearity) ---")
    print(pd.DataFrame(vifs).to_string(index=False))


def _design_matrix(mode: ModeSpec, df: pd.DataFrame) -> pd.DataFrame:
    formula = f"{mode.target} ~ {DESIGN_PREDICTORS}"
    _y, X = patsy.dmatrices(formula, data=df, return_type="dataframe")
    return X


# ---------------------------------------------------------------------------
# Fit-quality metrics
# ---------------------------------------------------------------------------

def print_fit_quality(mode: ModeSpec, result, predictions: pd.DataFrame) -> None:
    actual    = predictions[f"{mode.label}_actual"]
    predicted = predictions[f"{mode.label}_expected_total"]
    residual  = predictions[f"{mode.label}_residual"]

    sum_pred   = float(predicted.sum())
    sum_actual = int(actual.sum())
    pct_diff   = 100.0 * (sum_pred - sum_actual) / sum_actual if sum_actual else 0.0
    mae        = float(residual.abs().mean())
    rmse       = float(np.sqrt((residual ** 2).mean()))
    spearman, _ = scipy_stats.spearmanr(actual, predicted)

    print(f"\n--- {mode.display_name} fit quality ---")
    print(f"  Events observed:               {sum_actual}")
    print(f"  Events predicted (sum):        {sum_pred:.1f}  ({pct_diff:+.1f}%)")
    print(f"  Pseudo R²:                     {result.prsquared:.4f}")
    print(f"  Log-likelihood:                {result.llf:.1f}")
    print(f"  AIC:                           {result.aic:.1f}")
    print(f"  LL-ratio p-value (vs null):    {result.llr_pvalue:.2e}")
    print(f"  MAE per site:                  {mae:.3f}")
    print(f"  RMSE per site:                 {rmse:.3f}")
    print(f"  Spearman ρ (actual vs pred):   {spearman:+.3f}")


# ---------------------------------------------------------------------------
# Top residuals (edge cases)
# ---------------------------------------------------------------------------

def print_top_residuals(mode: ModeSpec, predictions: pd.DataFrame, n: int = 5) -> None:
    cols = [
        "intersection_id",
        f"{mode.label}_actual",
        f"{mode.label}_expected_total",
        f"{mode.label}_residual",
    ]
    table = predictions[cols].copy()
    table.columns = ["intersection_id", "actual", "expected", "residual"]

    print(f"\n--- {mode.display_name}: top {n} actual >> predicted (under-predicted) ---")
    print(table.nlargest(n, "residual").round(2).to_string(index=False))
    print(f"\n--- {mode.display_name}: top {n} predicted >> actual (over-predicted) ---")
    print(table.nsmallest(n, "residual").round(2).to_string(index=False))


# ---------------------------------------------------------------------------
# Zero-prediction (zero-inflation check)
# ---------------------------------------------------------------------------

def print_zero_prediction_check(mode: ModeSpec, result, df: pd.DataFrame) -> None:
    """Observed vs NB-implied probability of an exact-zero count."""
    actual = df[mode.target]
    n_observed_zero = int((actual == 0).sum())

    mu_per_site = result.predict(df, offset=df["offset"].values)
    alpha = result.params.get("alpha", 0.0)
    n_expected_zero = _nb_zero_probability_sum(mu_per_site, alpha)

    print(f"\n--- {mode.display_name}: zero-count check ---")
    print(f"  Observed zero-{mode.label} sites:        {n_observed_zero} of {len(df)}")
    print(f"  NB-predicted P(0) summed across sites:  {n_expected_zero:.1f}")
    diff = n_observed_zero - n_expected_zero
    diagnosis = (
        "Observed excess zeros — consider zero-inflated NB"
        if diff > 0.10 * len(df)
        else "NB handles the zero count adequately"
    )
    print(f"  Verdict: {diagnosis}")


def _nb_zero_probability_sum(mu: pd.Series, alpha: float) -> float:
    if alpha <= 0:
        return float(np.exp(-mu).sum())
    n = 1.0 / alpha
    p_zero_per_site = (n / (n + mu)) ** n
    return float(p_zero_per_site.sum())


# ---------------------------------------------------------------------------
# Cross-mode residual correlation
# ---------------------------------------------------------------------------

def print_cross_mode_residual_correlation(predictions: pd.DataFrame) -> None:
    """If residuals correlate across modes at the same site, there's an
    unobserved site-level factor (lighting, sight distance, etc.) that
    affects all modes — a hint that random-effects / spatial structure
    would tighten the fits."""
    residual_frame = predictions[[f"{m.label}_residual" for m in MODES]].copy()
    residual_frame.columns = [m.label for m in MODES]
    corr = residual_frame.corr(method="pearson").round(3)
    print("\n--- Cross-mode residual correlation ---")
    print("(Strong off-diagonal correlation ⇒ unobserved site heterogeneity)")
    print(corr.to_string())


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def main() -> None:
    df          = load_modelling_frame()
    fitted      = load_fitted_results()
    predictions = pd.read_parquet(PREDICTIONS_PATH)

    print(f"\nEvaluating three fitted models against {len(df)} arterial sites.")

    for mode in MODES:
        print("\n" + "=" * 72)
        print(f"  {mode.display_name} model".center(72))
        print("=" * 72)
        result = fitted[mode.label]
        print_coefficient_table(mode, result)
        print_vif_table(df, mode)
        print_fit_quality(mode, result, predictions)
        print_top_residuals(mode, predictions)
        print_zero_prediction_check(mode, result, df)

    print("\n" + "=" * 72)
    print("  Cross-mode diagnostics".center(72))
    print("=" * 72)
    print_cross_mode_residual_correlation(predictions)


if __name__ == "__main__":
    main()
