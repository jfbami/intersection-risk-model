"""Shared harness for experiments E1, E3, E5.

READ-ONLY with respect to pipeline/ and data/. Imports helper functions from
pipeline.fit_risk_model but never calls main() and never writes model artifacts.

Why this does not just call `smf.negativebinomial(...).fit(disp=False)`
---------------------------------------------------------------------
Verified empirically (experiments/ab/e1_raw_aadt_diagnostic.py): NB2 MLE on this
data is badly conditioned when a predictor is on a raw 1e3..4e4 scale. The
production newton->bfgs ladder returns a DIVERGED optimum for
`ped_total ~ SHARED + max_aadt` (llf -404.25) while the true optimum is
llf -347.27, reachable via Nelder-Mead or by rescaling AADT. Column-scaling the
design matrix is an exact reparameterisation (identical model, identical llf/AIC,
coefficients trivially unscaled afterwards) and fixes it. We therefore:
  1. column-scale the design matrix by its (training) SD,
  2. try several optimizers,
  3. keep the CONVERGED solution with the highest log-likelihood.
This is applied identically to every spec so comparisons stay fair, and the
naive production-style result is recorded alongside for contrast.
"""
from __future__ import annotations

import traceback
import warnings

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats as sps
from sklearn.model_selection import KFold, StratifiedKFold
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from pipeline.fit_risk_model import load_and_join, prepare, MODES, SHARED_PREDICTORS  # noqa: F401

N_SPLITS = 5
SEED = 0

# optimizer ladder, cheapest first
FULL_METHODS = [
    ("newton", {}),
    ("bfgs", {"method": "bfgs", "maxiter": 200}),
    ("lbfgs", {"method": "lbfgs", "maxiter": 5000}),
    ("nm", {"method": "nm", "maxiter": 20000, "maxfun": 20000}),
]
CV_METHODS = FULL_METHODS[:3]  # nm reserved as fallback inside CV


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def get_df() -> pd.DataFrame:
    df, _stats = prepare(load_and_join())
    df = df.copy()
    df["sqrt_aadt"] = np.sqrt(df["max_aadt"])
    return df


def strata_for(y: np.ndarray) -> np.ndarray:
    """Count strata for StratifiedKFold: 0, 1, 2-3, 4+ events."""
    y = np.asarray(y)
    s = np.full(len(y), 3)
    s[y == 0] = 0
    s[y == 1] = 1
    s[(y >= 2) & (y <= 3)] = 2
    return s


def design(df, target, predictors):
    y, X = patsy.dmatrices(f"{target} ~ {predictors}", data=df, return_type="dataframe")
    return np.asarray(y).ravel(), X


# ---------------------------------------------------------------------------
# Robust NB2 MLE
# ---------------------------------------------------------------------------

def _scales(X: pd.DataFrame) -> np.ndarray:
    s = X.std(ddof=0).to_numpy(dtype=float)
    s[~np.isfinite(s)] = 1.0
    s[s == 0] = 1.0
    return s


def robust_nb(y, X: pd.DataFrame, offset, methods=FULL_METHODS, need_se=True) -> dict:
    """Fit NB2 on a column-scaled design; return best CONVERGED solution.

    Returns dict with beta/bse on the ORIGINAL column scale, plus a per-method
    audit trail. Never raises.
    """
    s = _scales(X)
    Xs = (X.to_numpy(dtype=float) / s)
    k = X.shape[1]
    attempts = []
    for name, kw in methods:
        rec = {"method": name}
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                r = sm.NegativeBinomial(y, Xs, offset=offset).fit(disp=False, **kw)
            cw = [w for w in caught if issubclass(w.category, ConvergenceWarning)]
            llf = float(r.llf)
            beta_s = np.asarray(r.params, dtype=float)
            try:
                bse_s = np.asarray(r.bse, dtype=float)
            except Exception:
                bse_s = np.full_like(beta_s, np.nan)
            rec.update({
                "converged": bool(r.mle_retvals.get("converged", True)) and not cw,
                "flagged_by_statsmodels": bool(cw) or not r.mle_retvals.get("converged", True),
                "llf": llf if np.isfinite(llf) else None,
                "se_finite": bool(np.all(np.isfinite(bse_s))),
                "_beta_s": beta_s, "_bse_s": bse_s,
            })
        except Exception as e:
            rec.update({"converged": False, "llf": None, "se_finite": False,
                        "error": repr(e)})
        attempts.append(rec)

    def ok(a, want_se):
        return (a.get("converged") and a.get("llf") is not None
                and (a.get("se_finite") or not want_se))

    pool = [a for a in attempts if ok(a, need_se)] or [a for a in attempts if ok(a, False)]
    if not pool:
        pool = [a for a in attempts if a.get("llf") is not None]
    if not pool:
        return {"ok": False, "attempts": [_clean(a) for a in attempts]}

    best = max(pool, key=lambda a: a["llf"])
    beta = best["_beta_s"].copy()
    bse = best["_bse_s"].copy()
    beta[:k] /= s
    bse[:k] /= s
    n = len(y)
    npar = len(beta)
    llf = best["llf"]
    return {
        "ok": True,
        "converged": bool(best["converged"]),
        "best_method": best["method"],
        "llf": llf,
        "aic": float(-2 * llf + 2 * npar),
        "bic": float(-2 * llf + npar * np.log(n)),
        "n_params": int(npar),
        "n_obs": int(n),
        "beta": beta[:k],
        "bse": bse[:k],
        "alpha": float(beta[k]) if npar > k else float("nan"),
        "columns": list(X.columns),
        "attempts": [_clean(a) for a in attempts],
        "any_method_diverged": bool(
            any(a.get("llf") is not None and a["llf"] < llf - 1.0 for a in attempts)),
    }


def _clean(a):
    return {k: v for k, v in a.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# In-sample
# ---------------------------------------------------------------------------

def fit_spec(df: pd.DataFrame, target: str, predictors: str) -> dict:
    """Robust in-sample NB2 fit + the naive production-style fit for contrast."""
    out = {"formula": f"{target} ~ {predictors}", "target": target, "predictors": predictors}
    try:
        y, X = design(df, target, predictors)
        offset = df["offset"].values
        fit = robust_nb(y, X, offset)
        if not fit["ok"]:
            out.update({"ok": False, "converged": False, "attempts": fit["attempts"]})
            return out
        cols = fit["columns"]
        eta = X.to_numpy(dtype=float) @ fit["beta"] + offset
        pred = np.exp(eta)
        with np.errstate(all="ignore"):
            z = fit["beta"] / fit["bse"]
            pvals = 2 * sps.norm.sf(np.abs(z))
        out.update({
            "ok": True,
            "converged": fit["converged"],
            "best_method": fit["best_method"],
            "optimizer_attempts": fit["attempts"],
            "any_method_diverged": fit["any_method_diverged"],
            "llf": fit["llf"], "aic": fit["aic"], "bic": fit["bic"],
            "n_params": fit["n_params"], "n_obs": fit["n_obs"],
            "alpha": fit["alpha"],
            "params": {c: float(b) for c, b in zip(cols, fit["beta"])},
            "bse": {c: float(b) for c, b in zip(cols, fit["bse"])},
            "pvalues": {c: float(p) for c, p in zip(cols, pvals)},
            "insample_mae": float(np.mean(np.abs(y - pred))),
            "insample_rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
            "sum_pred": float(pred.sum()), "sum_actual": float(y.sum()),
            "_beta": fit["beta"], "_design_info": X.design_info,
        })
        out["naive_production_fit"] = _naive_fit(df, target, predictors)
    except Exception:
        out.update({"ok": False, "converged": False, "traceback": traceback.format_exc()})
    return out


def _naive_fit(df, target, predictors) -> dict:
    """Exactly what pipeline.fit_risk_model.fit_for_mode would do: newton, then
    bfgs(200) on failure. Recorded so we can show where it lands vs the robust fit."""
    rec = {}
    try:
        model = smf.negativebinomial(f"{target} ~ {predictors}", data=df,
                                     offset=df["offset"].values)
        with warnings.catch_warnings(record=True) as c1:
            warnings.simplefilter("always")
            r = model.fit(disp=False)
        cw1 = [w for w in c1 if issubclass(w.category, ConvergenceWarning)]
        used = "newton"
        if cw1 or not r.mle_retvals.get("converged", True):
            with warnings.catch_warnings(record=True) as c2:
                warnings.simplefilter("always")
                r = model.fit(disp=False, method="bfgs", maxiter=200)
            cw1 = [w for w in c2 if issubclass(w.category, ConvergenceWarning)]
            used = "bfgs(200) retry"
        conv = bool(r.mle_retvals.get("converged", True)) and not cw1
        rec = {"ok": True, "method_used": used, "converged": conv,
               "llf": float(r.llf), "aic": float(r.aic), "bic": float(r.bic),
               "would_pipeline_sys_exit": (not conv)}
    except Exception:
        rec = {"ok": False, "traceback": traceback.format_exc()}
    return rec


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cv_spec(df: pd.DataFrame, target: str, predictors: str,
            stratified: bool = True, seed: int = SEED) -> dict:
    """5-fold CV, refit per training split with the robust fitter.

    The patsy design is built once on the full frame so the categorical level set
    is identical across folds; only the coefficients are estimated per fold.
    Columns constant within a training fold are dropped for that fold and flagged.
    """
    y, X = design(df, target, predictors)
    offset = df["offset"].values
    n = len(y)

    if stratified:
        splits = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                                 random_state=seed).split(np.zeros(n), strata_for(y))
    else:
        splits = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed).split(np.zeros(n))

    oof = np.full(n, np.nan)
    fold_mae, fold_rmse, fold_conv, degenerate, errors, methods = [], [], [], [], [], []

    for k, (tr, te) in enumerate(splits):
        Xtr = X.iloc[tr]
        keep = [c for c in Xtr.columns if c == "Intercept" or Xtr[c].nunique() > 1]
        if len(keep) != len(Xtr.columns):
            degenerate.append({"fold": k,
                               "dropped": [c for c in Xtr.columns if c not in keep]})
        fit = robust_nb(y[tr], Xtr[keep], offset[tr], methods=CV_METHODS, need_se=False)
        if fit["ok"] and not fit["converged"]:
            fit2 = robust_nb(y[tr], Xtr[keep], offset[tr], methods=FULL_METHODS,
                             need_se=False)
            if fit2["ok"] and (fit2["converged"] or fit2["llf"] > fit["llf"]):
                fit = fit2
        if not fit["ok"]:
            fold_conv.append(False)
            errors.append({"fold": k, "attempts": fit.get("attempts")})
            continue
        methods.append(fit["best_method"])
        fold_conv.append(bool(fit["converged"]))
        eta = X.iloc[te][keep].to_numpy(dtype=float) @ fit["beta"] + offset[te]
        p = np.exp(np.clip(eta, -50, 50))
        oof[te] = p
        fold_mae.append(float(np.mean(np.abs(y[te] - p))))
        fold_rmse.append(float(np.sqrt(np.mean((y[te] - p) ** 2))))

    res = {"stratified": stratified, "seed": seed, "n_splits": N_SPLITS,
           "fold_mae": fold_mae, "fold_rmse": fold_rmse,
           "fold_methods": methods,
           "all_folds_converged": bool(fold_conv) and all(fold_conv),
           "n_folds_converged": int(sum(fold_conv)),
           "n_folds_completed": len(fold_mae),
           "degenerate_columns": degenerate, "fold_errors": errors}
    if fold_mae:
        res.update({
            "cv_mae_mean": float(np.mean(fold_mae)), "cv_mae_sd": float(np.std(fold_mae, ddof=1)),
            "cv_rmse_mean": float(np.mean(fold_rmse)), "cv_rmse_sd": float(np.std(fold_rmse, ddof=1)),
        })
    ok = ~np.isnan(oof)
    if ok.sum() > 2:
        rho, pv = sps.spearmanr(oof[ok], y[ok])
        res.update({
            "pooled_oof_mae": float(np.mean(np.abs(y[ok] - oof[ok]))),
            "pooled_oof_rmse": float(np.sqrt(np.mean((y[ok] - oof[ok]) ** 2))),
            "pooled_oof_spearman": float(rho), "pooled_oof_spearman_p": float(pv),
            "n_oof_rows": int(ok.sum()),
        })
        res["_oof"] = oof
    return res


def repeated_cv(df, target, predictors, seeds=range(10), stratified=True) -> dict:
    """Repeat 5-fold across seeds -> seed-to-seed noise floor on the OOS metric."""
    maes, rmses, rhos = [], [], []
    nonconv = 0
    for s in seeds:
        r = cv_spec(df, target, predictors, stratified=stratified, seed=s)
        if not r.get("all_folds_converged", False):
            nonconv += 1
        if "pooled_oof_mae" in r:
            maes.append(r["pooled_oof_mae"])
            rmses.append(r["pooled_oof_rmse"])
            rhos.append(r["pooled_oof_spearman"])
    if not maes:
        return {"n_seeds_ok": 0, "n_seeds_with_nonconverged_fold": nonconv}
    return {
        "n_seeds_ok": len(maes), "seeds": list(seeds),
        "n_seeds_with_nonconverged_fold": nonconv,
        "oof_mae_mean": float(np.mean(maes)), "oof_mae_sd": float(np.std(maes, ddof=1)),
        "oof_rmse_mean": float(np.mean(rmses)), "oof_rmse_sd": float(np.std(rmses, ddof=1)),
        "oof_spearman_mean": float(np.mean(rhos)), "oof_spearman_sd": float(np.std(rhos, ddof=1)),
        "oof_mae_per_seed": maes,
    }


def evaluate(df, target, predictors, label, seeds=range(10)) -> dict:
    rec = {"label": label, "predictors": predictors}
    ins = fit_spec(df, target, predictors)
    rec["_beta"] = ins.pop("_beta", None)
    rec["_design_info"] = ins.pop("_design_info", None)
    rec["insample"] = ins
    rec["cv_stratified"] = cv_spec(df, target, predictors, stratified=True, seed=SEED)
    rec["cv_kfold"] = cv_spec(df, target, predictors, stratified=False, seed=SEED)
    rec["repeated_cv"] = repeated_cv(df, target, predictors, seeds=seeds)
    return rec


def predict_new(design_info, beta, newdf, offset):
    """Predict expected counts for new rows using a stored patsy design_info."""
    Xn = patsy.build_design_matrices([design_info], newdf, return_type="dataframe")[0]
    return np.exp(Xn.to_numpy(dtype=float) @ np.asarray(beta) + np.asarray(offset))


def print_row(spec, rec):
    ins, cvs = rec["insample"], rec["cv_stratified"]
    if not ins.get("ok"):
        print(f"  {spec:14s} FIT FAILED")
        return
    print(f"  {spec:14s} conv={ins['converged']!s:5s} opt={ins['best_method']:7s} "
          f"AIC={ins['aic']:9.2f} BIC={ins['bic']:9.2f} LL={ins['llf']:9.2f} | "
          f"cvMAE={cvs.get('cv_mae_mean', float('nan')):.4f}"
          f"+-{cvs.get('cv_mae_sd', float('nan')):.4f} "
          f"cvRMSE={cvs.get('cv_rmse_mean', float('nan')):.4f}"
          f"+-{cvs.get('cv_rmse_sd', float('nan')):.4f} "
          f"oofMAE={cvs.get('pooled_oof_mae', float('nan')):.4f} "
          f"oofRMSE={cvs.get('pooled_oof_rmse', float('nan')):.4f} "
          f"rho={cvs.get('pooled_oof_spearman', float('nan')):.3f}")
    rc = rec["repeated_cv"]
    if rc.get("n_seeds_ok"):
        print(f"                 repeated-CV(10 seeds) oofMAE={rc['oof_mae_mean']:.4f}"
              f"+-{rc['oof_mae_sd']:.4f}  oofRMSE={rc['oof_rmse_mean']:.4f}"
              f"+-{rc['oof_rmse_sd']:.4f}  rho={rc['oof_spearman_mean']:.3f}"
              f"+-{rc['oof_spearman_sd']:.3f}")


def strip_results(obj):
    if isinstance(obj, dict):
        return {str(k): strip_results(v) for k, v in obj.items()
                if not (isinstance(k, str) and k.startswith("_"))}
    if isinstance(obj, (list, tuple)):
        return [strip_results(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [float(x) for x in obj.ravel()]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj
