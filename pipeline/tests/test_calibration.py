"""Calibration check: NB predictive intervals should cover at the nominal rate.

This integration test reads the live parquet outputs from data/intermediate/
and the fitted pkls from data/model/. Skipped if the pipeline has not run.
"""

from pathlib import Path

import pandas as pd
import pytest
from scipy.stats import nbinom

from pipeline.score_risk import MODES, load_fitted_models

ROOT = Path(__file__).resolve().parent.parent.parent
PREDICTIONS_PATH = ROOT / "data" / "intermediate" / "intersection_predictions.parquet"

COVERAGE_LOWER_BOUND = 85.0   # 90% nominal, allow slack for discreteness


@pytest.fixture(scope="module")
def predictions() -> pd.DataFrame:
    if not PREDICTIONS_PATH.exists():
        pytest.skip("predictions parquet not built; run `python -m pipeline.fit_risk_model`")
    return pd.read_parquet(PREDICTIONS_PATH)


@pytest.fixture(scope="module")
def fitted() -> dict:
    return load_fitted_models()


def _nb_predictive_coverage(mu: pd.Series, n_obs: pd.Series, alpha: float) -> float:
    n_nb = 1.0 / alpha
    p    = 1.0 / (1.0 + alpha * mu.clip(lower=1e-9))
    lo   = nbinom.ppf(0.05, n_nb, p)
    hi   = nbinom.ppf(0.95, n_nb, p)
    return float(((n_obs >= lo) & (n_obs <= hi)).mean() * 100)


@pytest.mark.parametrize("mode", MODES, ids=lambda m: m.label)
def test_mode_crash_count_nb_predictive_interval_covers_at_nominal_rate(predictions, fitted, mode):
    alpha    = fitted[mode.label].alpha
    coverage = _nb_predictive_coverage(
        predictions[mode.crash_predicted],
        predictions[mode.crash_actual],
        alpha,
    )
    assert coverage >= COVERAGE_LOWER_BOUND, (
        f"{mode.label} 90% NB predictive coverage was {coverage:.1f}% "
        f"(below {COVERAGE_LOWER_BOUND}% threshold). Model may be miscalibrated."
    )


@pytest.mark.parametrize("mode", MODES, ids=lambda m: m.label)
def test_mode_ksi_proxy_nb_predictive_interval_covers_at_nominal_rate(predictions, fitted, mode):
    """The mode-KSI proxy (NB prediction × citywide share) should also be
    calibrated; borrowing the mode's α to the KSI-rate process is the
    Phase-2 simplification."""
    alpha       = fitted[mode.label].alpha
    crash_total = predictions[mode.crash_actual].sum()
    ksi_total   = predictions[mode.ksi_actual].sum()
    if crash_total == 0 or ksi_total == 0:
        pytest.skip(f"no {mode.label} crashes or KSI events in this dataset")
    city_share = ksi_total / crash_total
    mu_ksi     = predictions[mode.crash_predicted] * city_share

    coverage = _nb_predictive_coverage(mu_ksi, predictions[mode.ksi_actual], alpha)

    assert coverage >= COVERAGE_LOWER_BOUND, (
        f"{mode.label}-KSI proxy 90% predictive coverage was {coverage:.1f}% "
        f"(below {COVERAGE_LOWER_BOUND}% threshold)."
    )
