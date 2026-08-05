"""Shared helpers for experiments E2 (NB vs Poisson), E6 (exposure offset)
and E7 (NB vs ZINB).

Read-only with respect to pipeline/ and data/: it imports helper functions
from pipeline.fit_risk_model but never calls main() and never writes to
data/.

Conventions used by all three experiments
-----------------------------------------
* In-sample fits use the *production* formula API path
  (smf.negativebinomial / smf.poisson with offset=df["offset"].values).
* Cross-validation uses patsy design matrices built ONCE on the full frame,
  then row-subset per fold, and the statsmodels *array* API. This keeps the
  column set identical across folds (categorical levels legs_cat==5 (n=13)
  and arterial_class==5 (n=19) are rare enough that a fold-local patsy build
  could silently change the design). The only information that crosses the
  fold boundary is the *set of columns*, not any response value.
* AIC/BIC are recomputed by hand as -2*LL + 2k / -2*LL + k*log(n) with
  k = len(result.params), because statsmodels' own .bic for discrete models
  does not always count the NB dispersion parameter. Both are reported.
"""

from __future__ import annotations

import json
import platform
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import scipy
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats as sps
from statsmodels.tools.sm_exceptions import ConvergenceWarning

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "experiments" / "results"

CV_SPLITS = 5
CV_SEED = 0

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def get_modelling_frame():
    """The exact 346-row arterial frame the production fit uses."""
    from pipeline.fit_risk_model import load_and_join, prepare

    df, stats = prepare(load_and_join())
    return df.reset_index(drop=True), stats


def modes():
    from pipeline.fit_risk_model import MODES

    return MODES


def environment() -> dict:
    try:
        import sklearn

        skl = sklearn.__version__
    except ImportError:
        skl = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "statsmodels": statsmodels.__version__,
        "patsy": patsy.__version__,
        "sklearn": skl,
    }


# ---------------------------------------------------------------------------
# Fold generation
# ---------------------------------------------------------------------------


def kfold_indices(n: int, n_splits: int = CV_SPLITS, seed: int = CV_SEED):
    """5-fold indices. Uses sklearn.KFold(shuffle=True, random_state=seed)
    when sklearn is importable (it is, 1.6.1); otherwise an equivalent numpy
    implementation. Returns (list_of_(train_idx, test_idx), backend_name)."""
    try:
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(kf.split(np.arange(n))), "sklearn.model_selection.KFold"
    except ImportError:  # pragma: no cover - sklearn is present here
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        folds = np.array_split(perm, n_splits)
        out = []
        for i in range(n_splits):
            test = folds[i]
            train = np.concatenate([folds[j] for j in range(n_splits) if j != i])
            out.append((np.sort(train), np.sort(test)))
        return out, "numpy fallback"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def mae_rmse(y, mu) -> tuple[float, float]:
    err = np.asarray(y, dtype=float) - np.asarray(mu, dtype=float)
    return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err ** 2)))


def mean_sd(vals) -> dict:
    a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
    if a.size == 0:
        return {"mean": None, "sd": None, "n": 0}
    return {
        "mean": float(a.mean()),
        "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "n": int(a.size),
    }


def poisson_interval_coverage(y, mu, level: float = 0.90) -> dict:
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    mu = np.clip(np.asarray(mu, dtype=float), 1e-9, None)
    lo = sps.poisson.ppf(lo_q, mu)
    hi = sps.poisson.ppf(hi_q, mu)
    y = np.asarray(y, dtype=float)
    return {
        "coverage_pct": float(np.mean((y >= lo) & (y <= hi)) * 100),
        "mean_width": float(np.mean(hi - lo)),
        "median_width": float(np.median(hi - lo)),
    }


def nb_interval_coverage(y, mu, alpha: float, level: float = 0.90) -> dict:
    """Matches pipeline/tests/test_calibration.py: n = 1/alpha,
    p = 1/(1 + alpha*mu)."""
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    mu = np.clip(np.asarray(mu, dtype=float), 1e-9, None)
    n_nb = 1.0 / alpha
    p = 1.0 / (1.0 + alpha * mu)
    lo = sps.nbinom.ppf(lo_q, n_nb, p)
    hi = sps.nbinom.ppf(hi_q, n_nb, p)
    y = np.asarray(y, dtype=float)
    return {
        "coverage_pct": float(np.mean((y >= lo) & (y <= hi)) * 100),
        "mean_width": float(np.mean(hi - lo)),
        "median_width": float(np.median(hi - lo)),
    }


# ---------------------------------------------------------------------------
# Design matrices
# ---------------------------------------------------------------------------


def design(formula: str, df: pd.DataFrame):
    """patsy design built once on the full frame. Returns (y, X, design_info)."""
    y, X = patsy.dmatrices(formula, data=df, return_type="dataframe")
    return np.asarray(y).ravel(), X, X.design_info


# ---------------------------------------------------------------------------
# Fit wrappers (array API) with warning capture
# ---------------------------------------------------------------------------


def _fit_capture(model, **kw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = model.fit(disp=False, **kw)
    conv = [w for w in caught if issubclass(w.category, ConvergenceWarning)]
    return res, [str(w.message) for w in conv]


def converged_flag(res, conv_warns) -> bool:
    return (not conv_warns) and bool(res.mle_retvals.get("converged", True))


def fit_summary(res, conv_warns, label: str, n: int) -> dict:
    k = int(len(res.params))
    llf = float(res.llf)
    out = {
        "spec": label,
        "n": int(n),
        "k_params_incl_dispersion": k,
        "loglik": llf,
        "aic_manual": float(-2 * llf + 2 * k),
        "bic_manual": float(-2 * llf + np.log(n) * k),
        "aic_statsmodels": float(res.aic),
        "bic_statsmodels": float(res.bic),
        "converged": converged_flag(res, conv_warns),
        "mle_converged_flag": bool(res.mle_retvals.get("converged", True)),
        "convergence_warnings": conv_warns,
        "optimizer_iterations": int(res.mle_retvals.get("iterations", -1))
        if isinstance(res.mle_retvals.get("iterations", -1), (int, np.integer))
        else None,
    }
    return out


# ---------------------------------------------------------------------------
# JSON dump helper
# ---------------------------------------------------------------------------


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if np.isfinite(v) else str(v)
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else str(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Series):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    return obj


def dump_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2)
    print(f"\n[wrote] {path}")
