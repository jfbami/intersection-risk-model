"""Sanity: confirm the robust design-matrix fitter reproduces the production
formula fit on the three production specs. Read-only."""
import warnings

import numpy as np
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

from experiments.ab.e1_e3_e5_common import get_df, design, robust_nb, predict_new  # noqa: E402
from pipeline.fit_risk_model import MODES  # noqa: E402

df = get_df()
for mode in MODES:
    f = f"{mode.target} ~ {mode.predictors}"
    # production path: newton, then bfgs on failure (fit_risk_model.fit_for_mode)
    m = smf.negativebinomial(f, data=df, offset=df["offset"].values)
    r1 = m.fit(disp=False)
    if not r1.mle_retvals.get("converged", True):
        r1 = m.fit(disp=False, method="bfgs", maxiter=200)
    p1 = r1.predict(df, offset=df["offset"].values).values

    y, X = design(df, mode.target, mode.predictors)
    fit = robust_nb(y, X, df["offset"].values)
    p2 = predict_new(X.design_info, fit["beta"], df, df["offset"].values)

    print(f"{mode.label:8s} prod llf={r1.llf:.6f} aic={r1.aic:.4f} conv={r1.mle_retvals.get('converged')}")
    print(f"         robust llf={fit['llf']:.6f} aic={fit['aic']:.4f} conv={fit['converged']} "
          f"opt={fit['best_method']}")
    print(f"         max|pred diff|={np.max(np.abs(p1 - p2)):.3e}   "
          f"max|beta diff|={np.max(np.abs(np.asarray(r1.params)[:len(fit['beta'])] - fit['beta'])):.3e}")
    print(f"         attempts: " + ", ".join(
        f"{a['method']}(conv={a.get('converged')},llf="
        f"{a['llf']:.3f})" if a.get('llf') is not None else f"{a['method']}(ERR)"
        for a in fit["attempts"]))
