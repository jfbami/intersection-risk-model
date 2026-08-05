"""
E4 — Pooled all-crash NB + share scaling (the v2 approach) vs three per-mode
NB models (the current v3 approach).

Read-only with respect to pipeline/ and data/. Imports helper functions from
pipeline.fit_risk_model but never calls main() and never writes to data/.

Outputs
-------
experiments/results/e4_results.json
experiments/results/e4_pooled_vs_permode.md   (written by e4_report.py)
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import optimize
from scipy.stats import nbinom, spearmanr
from sklearn.model_selection import KFold
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from pipeline.fit_risk_model import (
    MODES,
    SHARED_PREDICTORS,
    load_and_join,
    prepare,
)

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT / "experiments" / "results"
JSON_PATH = RESULTS_DIR / "e4_results.json"

N_SPLITS = 5
RANDOM_STATE = 0
N_BOOT = 2000
BOOT_SEED = 0

# Arm 1 target/formula: identical predictor set to the v3 fits, pooled onto the
# all-crash target with log_aadt as the volume term. This isolates the
# pooled-vs-per-mode decision while holding the predictor encoding fixed.
POOLED_FORMULA = f"total_crashes ~ {SHARED_PREDICTORS} + log_aadt"

# Historical v2 used num_legs as a continuous slope rather than the top-coded
# categorical. Fit as a sensitivity check only.
POOLED_FORMULA_V2_LEGS = (
    "total_crashes ~ is_signalized + num_legs + max_speed_limit"
    " + bike_facility + C(arterial_class) + log_aadt"
)

MODE_LABELS = ("bike", "ped", "vehicle")
MODE_TARGET = {m.label: m.target for m in MODES}
MODE_FORMULA = {m.label: f"{m.target} ~ {m.predictors}" for m in MODES}
MODE_KSI = {
    "bike": "bike_ksi_total",
    "ped": "ped_ksi_total",
    "vehicle": "vehicle_only_ksi_total",
}


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_nb(formula: str, data: pd.DataFrame, tag: str) -> dict:
    """Fit NB2 exactly as production does, with the same BFGS retry ladder.

    Returns a dict with the result object plus a full convergence record.
    Never exits; a non-converged fit is reported, not swallowed.
    """
    model = smf.negativebinomial(formula, data=data, offset=data["offset"].values)

    record = {"tag": tag, "formula": formula, "n_rows": int(len(data))}

    res, warns, err = _fit_capture(model)
    attempts = [_attempt_record("newton (statsmodels default)", res, warns, err)]
    if res is not None and _converged(res, warns):
        record.update(
            method="default", converged=True, attempts=attempts,
            retry_used=False, error=None,
            converged_under_production_ladder=True,
            rescued_by_out_of_production_method=False,
        )
        return {"result": res, "record": record}

    res2, warns2, err2 = _fit_capture(model, method="bfgs", maxiter=200)
    attempts.append(_attempt_record("bfgs maxiter=200", res2, warns2, err2))
    if res2 is not None and _converged(res2, warns2):
        record.update(
            method="bfgs", converged=True, attempts=attempts,
            retry_used=True, error=None,
            converged_under_production_ladder=True,
            rescued_by_out_of_production_method=False,
        )
        return {"result": res2, "record": record}

    # Production's ladder (newton -> bfgs) is now exhausted: `fit_for_mode`
    # would sys.exit here. We go one step further ONLY as a diagnostic, to
    # record whether the failure is an optimiser artefact or a genuinely
    # unidentified fit. The extra attempt is flagged as out-of-production.
    res3, warns3, err3 = _fit_capture(model, method="nm", maxiter=10000, maxfun=10000)
    attempts.append(_attempt_record("nm maxiter=10000 (BEYOND production ladder)", res3, warns3, err3))
    rescued = res3 is not None and _converged(res3, warns3)

    final = res3 if rescued else (res2 if res2 is not None else res)
    record.update(
        method="nm(diagnostic)" if rescued else ("bfgs" if res2 is not None else "default"),
        converged=False,                      # production would have FAILED here
        converged_under_production_ladder=False,
        rescued_by_out_of_production_method=bool(rescued),
        attempts=attempts, retry_used=True,
        error=str(err3 or err2 or err) if (err3 or err2 or err) else None,
    )
    print(f"[NON-CONVERGENCE under production ladder] {tag} "
          f"(rescued by out-of-production nm: {rescued})", file=sys.stderr)
    return {"result": final, "record": record, "production_would_exit": True}


def _fit_capture(model, **kwargs):
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = model.fit(disp=False, **kwargs)
        conv = [w for w in caught if issubclass(w.category, ConvergenceWarning)]
        return res, conv, None
    except Exception as exc:  # noqa: BLE001 - we want the traceback text recorded
        import traceback
        return None, [], traceback.format_exc()


def _converged(res, conv_warns) -> bool:
    return (not conv_warns) and bool(res.mle_retvals.get("converged", True))


def _attempt_record(name, res, warns, err) -> dict:
    if res is None:
        return {"method": name, "raised": True, "traceback": err}
    return {
        "method": name,
        "raised": False,
        "mle_converged": bool(res.mle_retvals.get("converged", True)),
        "convergence_warnings": [str(w.message) for w in warns],
        "iterations": int(res.mle_retvals.get("iterations", -1)),
    }


def alpha_of(res) -> float:
    if "alpha" in res.params.index:
        return float(res.params["alpha"])
    return float(getattr(res, "alpha"))


def n_params(res) -> int:
    """Number of estimated parameters including alpha."""
    return int(len(res.params))


def predict(res, data: pd.DataFrame) -> np.ndarray:
    return np.asarray(res.predict(data, offset=data["offset"].values), dtype=float)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def nb_logscore(y: np.ndarray, mu: np.ndarray, alpha: float) -> float:
    """Mean NB2 log predictive density. Higher is better. Proper scoring rule,
    evaluated on the SAME target for both arms so it is directly comparable."""
    mu = np.clip(mu, 1e-9, None)
    n = 1.0 / alpha
    p = 1.0 / (1.0 + alpha * mu)
    return float(np.mean(nbinom.logpmf(y, n, p)))


def fit_alpha_given_mu(y: np.ndarray, mu: np.ndarray) -> float:
    """MLE of the NB2 dispersion holding mu fixed. Used to give Arm 1 its best
    possible dispersion rather than forcing it to borrow the pooled alpha."""
    mu = np.clip(mu, 1e-9, None)

    def nll(log_alpha: float) -> float:
        a = float(np.exp(log_alpha))
        n = 1.0 / a
        p = 1.0 / (1.0 + a * mu)
        return -float(np.sum(nbinom.logpmf(y, n, p)))

    out = optimize.minimize_scalar(nll, bounds=(np.log(1e-4), np.log(50.0)), method="bounded")
    return float(np.exp(out.x))


def nb_coverage(y: np.ndarray, mu: np.ndarray, alpha: float, level: float = 0.90) -> float:
    """90% NB predictive interval coverage, identical estimator to
    pipeline/tests/test_calibration.py:33."""
    tail = (1 - level) / 2
    mu = np.clip(mu, 1e-9, None)
    n = 1.0 / alpha
    p = 1.0 / (1.0 + alpha * mu)
    lo = nbinom.ppf(tail, n, p)
    hi = nbinom.ppf(1 - tail, n, p)
    return float(np.mean((y >= lo) & (y <= hi)) * 100.0)


def core_metrics(y: np.ndarray, mu: np.ndarray) -> dict:
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    err = mu - y
    sa = float(y.sum())
    rho = spearmanr(mu, y).statistic if len(np.unique(mu)) > 1 and len(np.unique(y)) > 1 else float("nan")
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "spearman": float(rho),
        "sum_pred": float(mu.sum()),
        "sum_actual": sa,
        "calibration_pct": float((mu.sum() - sa) / sa * 100.0) if sa > 0 else float("nan"),
    }


def topk_event_capture(y: np.ndarray, mu: np.ndarray, k: int) -> dict:
    """Of all observed events, how many sit in the top-k sites by predicted risk?
    This is the decision-relevant metric for a prioritisation tool and is far more
    interpretable than Spearman when the event count is tiny."""
    order = np.argsort(-np.asarray(mu, dtype=float), kind="stable")
    top = order[:k]
    total = float(np.sum(y))
    captured = float(np.sum(np.asarray(y, dtype=float)[top]))
    return {
        "k": k,
        "events_captured": captured,
        "events_total": total,
        "pct_captured": float(captured / total * 100.0) if total > 0 else float("nan"),
        "pct_captured_if_random": float(k / len(y) * 100.0),
    }


def agg(fold_metrics: list[dict], keys: list[str]) -> dict:
    out = {}
    for k in keys:
        vals = np.array([fm[k] for fm in fold_metrics], dtype=float)
        finite = vals[np.isfinite(vals)]
        out[k] = {
            "mean": float(finite.mean()) if finite.size else float("nan"),
            "sd": float(finite.std(ddof=1)) if finite.size > 1 else float("nan"),
            "per_fold": [None if not np.isfinite(v) else float(v) for v in vals],
            "n_finite_folds": int(finite.size),
        }
    return out


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_spearman_diff(y, mu_a, mu_b, n_boot=N_BOOT, seed=BOOT_SEED) -> dict:
    """Paired site-level bootstrap. Resamples the 346 sites with replacement and
    recomputes both arms' Spearman on the SAME resample, so the difference is
    paired. Predictions are held fixed (no refit inside the bootstrap) — this
    captures sampling noise in the evaluation set, not parameter uncertainty."""
    y = np.asarray(y, dtype=float)
    mu_a = np.asarray(mu_a, dtype=float)
    mu_b = np.asarray(mu_b, dtype=float)
    n = len(y)
    rng = np.random.default_rng(seed)

    ra, rb, rd = [], [], []
    n_degenerate = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            n_degenerate += 1
            continue
        sa = spearmanr(mu_a[idx], yy).statistic
        sb = spearmanr(mu_b[idx], yy).statistic
        if not (np.isfinite(sa) and np.isfinite(sb)):
            n_degenerate += 1
            continue
        ra.append(sa); rb.append(sb); rd.append(sb - sa)

    def ci(v):
        v = np.asarray(v, dtype=float)
        if v.size == 0:
            return {"point_mean": None, "ci_lo": None, "ci_hi": None}
        return {
            "point_mean": float(v.mean()),
            "ci_lo": float(np.percentile(v, 2.5)),
            "ci_hi": float(np.percentile(v, 97.5)),
        }

    d = np.asarray(rd, dtype=float)
    return {
        "n_boot_requested": n_boot,
        "n_boot_used": len(rd),
        "n_degenerate_resamples": n_degenerate,
        "arm1_pooled": ci(ra),
        "arm2_permode": ci(rb),
        "diff_arm2_minus_arm1": ci(d),
        "p_diff_gt_0": float(np.mean(d > 0)) if d.size else None,
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out: dict = {
        "experiment": "E4 pooled all-crash + share scaling vs three per-mode NB models",
        "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "env": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }

    df, prep_stats = prepare(load_and_join())
    df = df.reset_index(drop=True)
    n = len(df)

    # -- event counts & the 1720 vs 1730 double-count -----------------------
    counts = {c: int(df[c].sum()) for c in [
        "total_crashes", "bike_total", "ped_total", "vehicle_only_total",
        "bike_ksi_total", "ped_ksi_total", "vehicle_only_ksi_total", "ksi_total",
    ]}
    counts["sum_of_three_mode_targets"] = (
        counts["bike_total"] + counts["ped_total"] + counts["vehicle_only_total"]
    )
    counts["mode_sum_minus_total"] = counts["sum_of_three_mode_targets"] - counts["total_crashes"]
    counts["sum_of_three_mode_ksi"] = (
        counts["bike_ksi_total"] + counts["ped_ksi_total"] + counts["vehicle_only_ksi_total"]
    )
    counts["mode_ksi_sum_minus_ksi_total"] = counts["sum_of_three_mode_ksi"] - counts["ksi_total"]
    out["n_sites"] = n
    out["counts"] = counts
    out["formulas"] = {
        "pooled_arm1": POOLED_FORMULA,
        "pooled_v2_legs_sensitivity": POOLED_FORMULA_V2_LEGS,
        **{f"permode_{k}": v for k, v in MODE_FORMULA.items()},
    }
    print(f"n={n} sites; counts={counts}")

    # -----------------------------------------------------------------------
    # Full-data (in-sample) fits
    # -----------------------------------------------------------------------
    print("\n=== Full-data fits ===")
    pooled = fit_nb(POOLED_FORMULA, df, "pooled_total_crashes_full")
    pooled_res = pooled["result"]

    permode = {}
    for lab in MODE_LABELS:
        permode[lab] = fit_nb(MODE_FORMULA[lab], df, f"permode_{lab}_full")

    pooled_v2legs = fit_nb(POOLED_FORMULA_V2_LEGS, df, "pooled_v2_legs_sensitivity_full")

    def fit_summary(f) -> dict:
        r = f["result"]
        return {
            **f["record"],
            "llf": float(r.llf),
            "aic": float(r.aic),
            "bic": float(r.bic),
            "alpha": alpha_of(r),
            "n_params_incl_alpha": n_params(r),
            "df_model": int(r.df_model),
        }

    out["fits"] = {
        "pooled": fit_summary(pooled),
        **{f"permode_{k}": fit_summary(v) for k, v in permode.items()},
        "pooled_v2_legs_sensitivity": fit_summary(pooled_v2legs),
    }
    for k, v in out["fits"].items():
        print(f"  {k:32s} converged={v['converged']} method={v['method']} "
              f"llf={v['llf']:.2f} aic={v['aic']:.2f} alpha={v['alpha']:.4f} k={v['n_params_incl_alpha']}")

    # Sanity: statsmodels' discrete-NB `fittedvalues` is the LINEAR PREDICTOR
    # excluding the offset, so predict(df, offset=...) must equal
    # exp(fittedvalues + offset) and equal exp(X@beta + offset) exactly.
    _p = predict(pooled_res, df)
    _X = pooled_res.model.exog
    _beta = pooled_res.params.values[: _X.shape[1]]
    chk_fv = float(np.max(np.abs(_p - np.exp(np.asarray(pooled_res.fittedvalues) + df["offset"].values))))
    chk_manual = float(np.max(np.abs(_p - np.exp(_X @ _beta + df["offset"].values))))
    out["sanity"] = {
        "max_abs_predict_minus_exp_linpred_plus_offset": chk_fv,
        "max_abs_predict_minus_manual_Xbeta_offset": chk_manual,
        "note": "fittedvalues on a statsmodels discrete NB is the linear predictor "
                "WITHOUT the offset; predict(df, offset=...) is on the response scale.",
    }
    print(f"  sanity |predict - exp(linpred+offset)| max = {chk_fv:.3e}; manual = {chk_manual:.3e}")
    assert chk_fv < 1e-8 and chk_manual < 1e-8, "prediction scale sanity check failed"

    mu_pooled_full = predict(pooled_res, df)
    mu_mode_full = {lab: predict(permode[lab]["result"], df) for lab in MODE_LABELS}
    alpha_pooled_full = alpha_of(pooled_res)
    alpha_mode_full = {lab: alpha_of(permode[lab]["result"]) for lab in MODE_LABELS}

    # -----------------------------------------------------------------------
    # PART A — in-sample
    # -----------------------------------------------------------------------
    print("\n=== Part A in-sample ===")
    partA_is = {}
    for lab in MODE_LABELS:
        y = df[MODE_TARGET[lab]].to_numpy(float)
        share = float(y.sum() / df["total_crashes"].sum())
        mu1 = mu_pooled_full * share
        mu2 = mu_mode_full[lab]

        a1_borrow = alpha_pooled_full
        a1_refit = fit_alpha_given_mu(y, mu1)
        a2 = alpha_mode_full[lab]

        m1 = core_metrics(y, mu1)
        m1.update(
            share_used=share,
            alpha_borrowed_pooled=a1_borrow,
            alpha_refit_given_mu=a1_refit,
            nb_logscore_alpha_borrowed=nb_logscore(y, mu1, a1_borrow),
            nb_logscore_alpha_refit=nb_logscore(y, mu1, a1_refit),
            coverage90_alpha_borrowed=nb_coverage(y, mu1, a1_borrow),
            coverage90_alpha_refit=nb_coverage(y, mu1, a1_refit),
        )
        m2 = core_metrics(y, mu2)
        m2.update(
            alpha=a2,
            nb_logscore=nb_logscore(y, mu2, a2),
            coverage90=nb_coverage(y, mu2, a2),
        )
        partA_is[lab] = {"arm1_pooled_scaled": m1, "arm2_permode": m2}
        print(f"  {lab:8s} MAE  pooled={m1['mae']:.4f}  permode={m2['mae']:.4f}   "
              f"RMSE {m1['rmse']:.4f}/{m2['rmse']:.4f}  rho {m1['spearman']:+.4f}/{m2['spearman']:+.4f}")
    out["part_A_in_sample"] = partA_is

    # -----------------------------------------------------------------------
    # Cross-validation (shared across Parts A and B)
    # -----------------------------------------------------------------------
    print(f"\n=== {N_SPLITS}-fold CV (KFold shuffle=True random_state={RANDOM_STATE}) ===")
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    folds = list(kf.split(df))

    oof_pooled = np.full(n, np.nan)
    oof_mode = {lab: np.full(n, np.nan) for lab in MODE_LABELS}
    oof_share = {lab: np.full(n, np.nan) for lab in MODE_LABELS}          # train-only crash share
    oof_ksi_share_pooled = np.full(n, np.nan)                              # bike_ksi / total_crashes (train)
    oof_ksi_share_bike = np.full(n, np.nan)                                # bike_ksi / bike_total (train)
    oof_alpha_pooled = np.full(n, np.nan)
    oof_alpha_mode = {lab: np.full(n, np.nan) for lab in MODE_LABELS}
    oof_pooled_v2 = np.full(n, np.nan)          # arm 1b: true historical v2 spec
    oof_alpha_pooled_v2 = np.full(n, np.nan)

    cv_fit_records = []
    fold_rows = []

    for fold_i, (tr, te) in enumerate(folds):
        trd = df.iloc[tr].copy()
        ted = df.iloc[te].copy()

        fp = fit_nb(POOLED_FORMULA, trd, f"pooled_fold{fold_i}")
        cv_fit_records.append(fp["record"])
        mu_p_te = predict(fp["result"], ted)
        oof_pooled[te] = mu_p_te
        a_p = alpha_of(fp["result"])
        oof_alpha_pooled[te] = a_p

        # Arm 1b: the TRUE historical v2 spec (num_legs continuous), exactly
        # reconstructed (see provenance_check_all_crash_target).
        fv2 = fit_nb(POOLED_FORMULA_V2_LEGS, trd, f"pooled_v2legs_fold{fold_i}")
        cv_fit_records.append(fv2["record"])
        oof_pooled_v2[te] = predict(fv2["result"], ted)
        oof_alpha_pooled_v2[te] = alpha_of(fv2["result"])

        fm = {}
        for lab in MODE_LABELS:
            f = fit_nb(MODE_FORMULA[lab], trd, f"permode_{lab}_fold{fold_i}")
            cv_fit_records.append(f["record"])
            fm[lab] = f
            oof_mode[lab][te] = predict(f["result"], ted)
            oof_alpha_mode[lab][te] = alpha_of(f["result"])
            # TRAIN-ONLY share — never uses held-out rows, so no leakage.
            oof_share[lab][te] = float(trd[MODE_TARGET[lab]].sum() / trd["total_crashes"].sum())

        oof_ksi_share_pooled[te] = float(trd["bike_ksi_total"].sum() / trd["total_crashes"].sum())
        oof_ksi_share_bike[te] = float(trd["bike_ksi_total"].sum() / trd["bike_total"].sum())

        row = {
            "fold": fold_i, "n_train": int(len(tr)), "n_test": int(len(te)),
            "pooled_converged": fp["record"]["converged"],
            "pooled_alpha": a_p,
            "train_total_crashes": int(trd["total_crashes"].sum()),
            "test_bike_ksi": int(ted["bike_ksi_total"].sum()),
            "train_bike_ksi_share_of_total": float(oof_ksi_share_pooled[te][0]),
            "train_bike_ksi_share_of_bike": float(oof_ksi_share_bike[te][0]),
        }
        for lab in MODE_LABELS:
            row[f"{lab}_converged"] = fm[lab]["record"]["converged"]
            row[f"{lab}_alpha"] = alpha_of(fm[lab]["result"])
            row[f"train_share_{lab}"] = float(oof_share[lab][te][0])
        fold_rows.append(row)
        print(f"  fold {fold_i}: ntr={len(tr)} nte={len(te)} "
              f"conv pooled={row['pooled_converged']} "
              + " ".join(f"{l}={row[f'{l}_converged']}" for l in MODE_LABELS))

    assert not np.isnan(oof_pooled).any(), "OOF pooled predictions incomplete"
    out["cv"] = {
        "n_splits": N_SPLITS, "shuffle": True, "random_state": RANDOM_STATE,
        "splitter": "sklearn.model_selection.KFold",
        "fold_summary": fold_rows,
        "all_fits_converged": all(r["converged"] for r in cv_fit_records),
        "n_cv_fits": len(cv_fit_records),
        "non_converged_fits": [r for r in cv_fit_records if not r["converged"]],
    }

    # -----------------------------------------------------------------------
    # PART A — out-of-sample
    # -----------------------------------------------------------------------
    print("\n=== Part A out-of-sample ===")
    partA_cv = {}
    metric_keys = ["mae", "rmse", "spearman", "calibration_pct"]
    for lab in MODE_LABELS:
        y = df[MODE_TARGET[lab]].to_numpy(float)
        mu1_oof = oof_pooled * oof_share[lab]
        mu2_oof = oof_mode[lab]

        per_fold_1, per_fold_2 = [], []
        for _, te in folds:
            m1 = core_metrics(y[te], mu1_oof[te])
            m1["nb_logscore"] = nb_logscore(y[te], mu1_oof[te], float(oof_alpha_pooled[te][0]))
            m2 = core_metrics(y[te], mu2_oof[te])
            m2["nb_logscore"] = nb_logscore(y[te], mu2_oof[te], float(oof_alpha_mode[lab][te][0]))
            per_fold_1.append(m1); per_fold_2.append(m2)

        keys = metric_keys + ["nb_logscore"]
        pooled_oof_1 = core_metrics(y, mu1_oof)
        pooled_oof_1["nb_logscore_alpha_borrowed"] = float(np.mean([
            nb_logscore(y[te], mu1_oof[te], float(oof_alpha_pooled[te][0])) * len(te) for _, te in folds
        ]) * len(folds) / n)
        pooled_oof_1["alpha_refit_on_oof"] = fit_alpha_given_mu(y, mu1_oof)
        pooled_oof_1["nb_logscore_alpha_refit_oof"] = nb_logscore(y, mu1_oof, pooled_oof_1["alpha_refit_on_oof"])
        pooled_oof_2 = core_metrics(y, mu2_oof)
        pooled_oof_2["nb_logscore"] = float(np.mean([
            nb_logscore(y[te], mu2_oof[te], float(oof_alpha_mode[lab][te][0])) * len(te) for _, te in folds
        ]) * len(folds) / n)

        boot = bootstrap_spearman_diff(y, mu1_oof, mu2_oof)

        partA_cv[lab] = {
            "arm1_pooled_scaled": {"fold_agg": agg(per_fold_1, keys), "pooled_oof": pooled_oof_1},
            "arm2_permode": {"fold_agg": agg(per_fold_2, keys), "pooled_oof": pooled_oof_2},
            "delta_pooled_oof": {
                "mae_arm2_minus_arm1": pooled_oof_2["mae"] - pooled_oof_1["mae"],
                "rmse_arm2_minus_arm1": pooled_oof_2["rmse"] - pooled_oof_1["rmse"],
                "spearman_arm2_minus_arm1": pooled_oof_2["spearman"] - pooled_oof_1["spearman"],
                "mae_pct_change": (pooled_oof_2["mae"] - pooled_oof_1["mae"]) / pooled_oof_1["mae"] * 100.0,
                "rmse_pct_change": (pooled_oof_2["rmse"] - pooled_oof_1["rmse"]) / pooled_oof_1["rmse"] * 100.0,
            },
            "bootstrap_spearman": boot,
        }
        d = partA_cv[lab]["delta_pooled_oof"]
        print(f"  {lab:8s} OOF MAE pooled={pooled_oof_1['mae']:.4f} permode={pooled_oof_2['mae']:.4f} "
              f"({d['mae_pct_change']:+.2f}%)  RMSE {pooled_oof_1['rmse']:.4f}/{pooled_oof_2['rmse']:.4f} "
              f"({d['rmse_pct_change']:+.2f}%)  rho {pooled_oof_1['spearman']:+.4f}/{pooled_oof_2['spearman']:+.4f}")
    out["part_A_cv"] = partA_cv

    # -----------------------------------------------------------------------
    # PART B — bike KSI
    # -----------------------------------------------------------------------
    print("\n=== Part B: bike KSI ===")
    y_ksi = df["bike_ksi_total"].to_numpy(float)

    share_v2_full = float(df["bike_ksi_total"].sum() / df["total_crashes"].sum())
    share_v3_full = float(df["bike_ksi_total"].sum() / df["bike_total"].sum())

    mu_ksi1_is = mu_pooled_full * share_v2_full
    mu_ksi2_is = mu_mode_full["bike"] * share_v3_full

    partB_is = {
        "share_v2_bike_ksi_over_total_crashes": share_v2_full,
        "share_v3_bike_ksi_over_bike_total": share_v3_full,
        "arm1_v2_route": {
            **core_metrics(y_ksi, mu_ksi1_is),
            "alpha_used": alpha_pooled_full,
            "coverage90": nb_coverage(y_ksi, mu_ksi1_is, alpha_pooled_full),
            "nb_logscore": nb_logscore(y_ksi, mu_ksi1_is, alpha_pooled_full),
        },
        "arm2_v3_route": {
            **core_metrics(y_ksi, mu_ksi2_is),
            "alpha_used": alpha_mode_full["bike"],
            "coverage90": nb_coverage(y_ksi, mu_ksi2_is, alpha_mode_full["bike"]),
            "nb_logscore": nb_logscore(y_ksi, mu_ksi2_is, alpha_mode_full["bike"]),
        },
        "bootstrap_spearman": bootstrap_spearman_diff(y_ksi, mu_ksi1_is, mu_ksi2_is),
        "topk_capture_arm1": {k: topk_event_capture(y_ksi, mu_ksi1_is, k) for k in (10, 20, 50, 100)},
        "topk_capture_arm2": {k: topk_event_capture(y_ksi, mu_ksi2_is, k) for k in (10, 20, 50, 100)},
    }
    print(f"  in-sample MAE {partB_is['arm1_v2_route']['mae']:.4f} vs {partB_is['arm2_v3_route']['mae']:.4f}"
          f"  rho {partB_is['arm1_v2_route']['spearman']:+.4f} vs {partB_is['arm2_v3_route']['spearman']:+.4f}")
    print(f"  in-sample bootstrap diff rho: {partB_is['bootstrap_spearman']['diff_arm2_minus_arm1']}")

    # out-of-sample, train-only shares
    mu_ksi1_oof = oof_pooled * oof_ksi_share_pooled
    mu_ksi2_oof = oof_mode["bike"] * oof_ksi_share_bike

    per_fold_1, per_fold_2 = [], []
    for _, te in folds:
        a_p = float(oof_alpha_pooled[te][0])
        a_b = float(oof_alpha_mode["bike"][te][0])
        m1 = core_metrics(y_ksi[te], mu_ksi1_oof[te])
        m1["coverage90"] = nb_coverage(y_ksi[te], mu_ksi1_oof[te], a_p)
        m1["nb_logscore"] = nb_logscore(y_ksi[te], mu_ksi1_oof[te], a_p)
        m2 = core_metrics(y_ksi[te], mu_ksi2_oof[te])
        m2["coverage90"] = nb_coverage(y_ksi[te], mu_ksi2_oof[te], a_b)
        m2["nb_logscore"] = nb_logscore(y_ksi[te], mu_ksi2_oof[te], a_b)
        per_fold_1.append(m1); per_fold_2.append(m2)

    keysB = ["mae", "rmse", "spearman", "calibration_pct", "coverage90", "nb_logscore"]

    def _wavg(field, per_fold):
        return float(sum(fm[field] * len(te) for fm, (_, te) in zip(per_fold, folds)) / n)

    oofB1 = core_metrics(y_ksi, mu_ksi1_oof)
    oofB1["coverage90"] = _wavg("coverage90", per_fold_1)
    oofB1["nb_logscore"] = _wavg("nb_logscore", per_fold_1)
    oofB2 = core_metrics(y_ksi, mu_ksi2_oof)
    oofB2["coverage90"] = _wavg("coverage90", per_fold_2)
    oofB2["nb_logscore"] = _wavg("nb_logscore", per_fold_2)

    bootB = bootstrap_spearman_diff(y_ksi, mu_ksi1_oof, mu_ksi2_oof)

    out["part_B_in_sample"] = partB_is
    out["part_B_cv"] = {
        "arm1_v2_route": {"fold_agg": agg(per_fold_1, keysB), "pooled_oof": oofB1},
        "arm2_v3_route": {"fold_agg": agg(per_fold_2, keysB), "pooled_oof": oofB2},
        "delta_pooled_oof": {
            "mae_arm2_minus_arm1": oofB2["mae"] - oofB1["mae"],
            "rmse_arm2_minus_arm1": oofB2["rmse"] - oofB1["rmse"],
            "spearman_arm2_minus_arm1": oofB2["spearman"] - oofB1["spearman"],
        },
        "bootstrap_spearman": bootB,
        "topk_capture_arm1": {k: topk_event_capture(y_ksi, mu_ksi1_oof, k) for k in (10, 20, 50, 100)},
        "topk_capture_arm2": {k: topk_event_capture(y_ksi, mu_ksi2_oof, k) for k in (10, 20, 50, 100)},
        "n_bike_ksi_events": int(y_ksi.sum()),
        "n_sites_with_any_bike_ksi": int((y_ksi > 0).sum()),
    }
    print(f"  OOF  MAE {oofB1['mae']:.4f} vs {oofB2['mae']:.4f}; rho {oofB1['spearman']:+.4f} vs {oofB2['spearman']:+.4f}")
    print(f"  OOF bootstrap diff rho: {bootB['diff_arm2_minus_arm1']}")

    # -----------------------------------------------------------------------
    # ARM 1b — the TRUE historical v2 spec, out-of-sample
    # Arm 1 above upgrades v2's leg encoding to v3's legs_cat, which isolates
    # pooling-vs-per-mode but also strengthens v2. This block runs the exactly
    # reconstructed historical v2 so the "was the rewrite justified" question
    # is answered against what actually shipped.
    # -----------------------------------------------------------------------
    print("\n=== Arm 1b: true historical v2 spec (num_legs continuous), OOF ===")
    arm1b = {"spec": POOLED_FORMULA_V2_LEGS}
    for lab in MODE_LABELS:
        y = df[MODE_TARGET[lab]].to_numpy(float)
        mu1b = oof_pooled_v2 * oof_share[lab]
        m = core_metrics(y, mu1b)
        m["nb_logscore"] = float(sum(
            nb_logscore(y[te], mu1b[te], float(oof_alpha_pooled_v2[te][0])) * len(te)
            for _, te in folds) / n)
        m["boot_vs_arm2_permode"] = bootstrap_spearman_diff(y, mu1b, oof_mode[lab])
        arm1b[lab] = m
        a2 = partA_cv[lab]["arm2_permode"]["pooled_oof"]
        print(f"  {lab:8s} arm1b MAE={m['mae']:.4f} RMSE={m['rmse']:.4f} rho={m['spearman']:+.4f}"
              f"   | arm2 MAE={a2['mae']:.4f} RMSE={a2['rmse']:.4f} rho={a2['spearman']:+.4f}")

    mu_ksi1b_oof = oof_pooled_v2 * oof_ksi_share_pooled
    mb = core_metrics(y_ksi, mu_ksi1b_oof)
    mb["coverage90"] = float(sum(
        nb_coverage(y_ksi[te], mu_ksi1b_oof[te], float(oof_alpha_pooled_v2[te][0])) * len(te)
        for _, te in folds) / n)
    mb["boot_vs_arm2_permode"] = bootstrap_spearman_diff(y_ksi, mu_ksi1b_oof, mu_ksi2_oof)
    arm1b["bike_ksi"] = mb
    out["arm_1b_true_v2_oof"] = arm1b
    print(f"  bike_ksi arm1b MAE={mb['mae']:.5f} rho={mb['spearman']:+.4f}  | "
          f"arm2 MAE={oofB2['mae']:.5f} rho={oofB2['spearman']:+.4f}")
    print(f"  bike_ksi boot diff (arm2 - arm1b): {mb['boot_vs_arm2_permode']['diff_arm2_minus_arm1']}")

    # -----------------------------------------------------------------------
    # PART C — complexity
    # -----------------------------------------------------------------------
    print("\n=== Part C: complexity ===")
    k_pooled = n_params(pooled_res)
    k_mode = {lab: n_params(permode[lab]["result"]) for lab in MODE_LABELS}
    out["part_C"] = {
        "arm1_n_models": 1,
        "arm1_n_params_incl_alpha": k_pooled,
        "arm2_n_models": 3,
        "arm2_n_params_per_model": k_mode,
        "arm2_n_params_total": int(sum(k_mode.values())),
        "arm1_aic": float(pooled_res.aic),
        "arm1_bic": float(pooled_res.bic),
        "arm1_llf": float(pooled_res.llf),
        "arm2_aic_sum_NOT_COMPARABLE": float(sum(permode[l]["result"].aic for l in MODE_LABELS)),
        "arm2_bic_sum_NOT_COMPARABLE": float(sum(permode[l]["result"].bic for l in MODE_LABELS)),
        "arm2_llf_sum_NOT_COMPARABLE": float(sum(permode[l]["result"].llf for l in MODE_LABELS)),
        "comparability_note": (
            "Summing AIC/BIC across the three per-mode fits and comparing to the "
            "pooled fit's AIC is NOT defensible: the likelihoods are over different "
            "random variables (total_crashes, 1720 events, vs the three mode targets, "
            "1730 events including double-counted bike+ped crashes). Information "
            "criteria are only comparable across models of the SAME data. The numbers "
            "are recorded for completeness and must not be used as a verdict."
        ),
        "legitimate_same_target_comparison": {
            lab: {
                "arm1_nb_logscore_alpha_borrowed": partA_is[lab]["arm1_pooled_scaled"]["nb_logscore_alpha_borrowed"],
                "arm1_nb_logscore_alpha_refit": partA_is[lab]["arm1_pooled_scaled"]["nb_logscore_alpha_refit"],
                "arm2_nb_logscore": partA_is[lab]["arm2_permode"]["nb_logscore"],
                "arm1_effective_free_params": 1,   # only the share (alpha borrowed)
                "arm2_free_params": k_mode[lab],
            } for lab in MODE_LABELS
        },
    }
    print(f"  arm1 k={k_pooled}; arm2 k={k_mode} total={sum(k_mode.values())}")

    # -----------------------------------------------------------------------
    # PART D — ranking disagreement
    # -----------------------------------------------------------------------
    print("\n=== Part D: ranking ===")
    rank1 = pd.Series(mu_ksi1_is).rank(ascending=False, method="average")
    rank2 = pd.Series(mu_ksi2_is).rank(ascending=False, method="average")
    rho_rank = float(spearmanr(mu_ksi1_is, mu_ksi2_is).statistic)

    # EB-posterior ranking, mirroring score_risk.compute_mode_ksi_eb exactly
    def eb_posterior_mean(mu_ksi, alpha, n_obs):
        k = 1.0 / alpha
        mu_ksi = np.clip(mu_ksi, 1e-9, None)
        shape = k + n_obs
        scale = mu_ksi / (k + mu_ksi)
        return shape * scale

    eb1 = eb_posterior_mean(mu_ksi1_is, alpha_pooled_full, y_ksi)
    eb2 = eb_posterior_mean(mu_ksi2_is, alpha_mode_full["bike"], y_ksi)
    eb_rank1 = pd.Series(eb1).rank(ascending=False, method="average")
    eb_rank2 = pd.Series(eb2).rank(ascending=False, method="average")
    rho_eb = float(spearmanr(eb1, eb2).statistic)

    shift = (rank2 - rank1).abs()
    eb_shift = (eb_rank2 - eb_rank1).abs()

    tbl = pd.DataFrame({
        "intersection_id": df["intersection_id"].values,
        "bike_total": df["bike_total"].values,
        "bike_ksi_total": df["bike_ksi_total"].values,
        "total_crashes": df["total_crashes"].values,
        "mu_ksi_arm1": mu_ksi1_is,
        "mu_ksi_arm2": mu_ksi2_is,
        "rank_arm1": rank1.values,
        "rank_arm2": rank2.values,
        "rank_shift": shift.values,
        "eb_rank_arm1": eb_rank1.values,
        "eb_rank_arm2": eb_rank2.values,
        "eb_rank_shift": eb_shift.values,
    })
    top_shift = tbl.nlargest(10, "rank_shift").sort_values("rank_shift", ascending=False)
    top_eb_shift = tbl.nlargest(10, "eb_rank_shift").sort_values("eb_rank_shift", ascending=False)

    def topk_overlap(a, b, k):
        ia = set(np.argsort(-np.asarray(a))[:k])
        ib = set(np.argsort(-np.asarray(b))[:k])
        return len(ia & ib)

    out["part_D"] = {
        "spearman_arm1_vs_arm2_prior_ranking": rho_rank,
        "spearman_arm1_vs_arm2_eb_ranking": rho_eb,
        "kendall_note": "prior ranking is a monotone function of the underlying crash prediction; the KSI share is a positive constant and does not affect ranks",
        "rank_shift_stats": {
            "mean": float(shift.mean()), "median": float(shift.median()),
            "p90": float(shift.quantile(0.9)), "max": float(shift.max()),
        },
        "eb_rank_shift_stats": {
            "mean": float(eb_shift.mean()), "median": float(eb_shift.median()),
            "p90": float(eb_shift.quantile(0.9)), "max": float(eb_shift.max()),
        },
        "top10_overlap_prior": topk_overlap(mu_ksi1_is, mu_ksi2_is, 10),
        "top20_overlap_prior": topk_overlap(mu_ksi1_is, mu_ksi2_is, 20),
        "top50_overlap_prior": topk_overlap(mu_ksi1_is, mu_ksi2_is, 50),
        "top10_overlap_eb": topk_overlap(eb1, eb2, 10),
        "top20_overlap_eb": topk_overlap(eb1, eb2, 20),
        "top50_overlap_eb": topk_overlap(eb1, eb2, 50),
        "largest_10_rank_shifts_prior": top_shift.to_dict(orient="records"),
        "largest_10_rank_shifts_eb": top_eb_shift.to_dict(orient="records"),
        "top10_sites_arm1": tbl.nsmallest(10, "rank_arm1")[
            ["intersection_id", "mu_ksi_arm1", "mu_ksi_arm2", "rank_arm1", "rank_arm2",
             "bike_total", "bike_ksi_total"]].to_dict(orient="records"),
        "top10_sites_arm2": tbl.nsmallest(10, "rank_arm2")[
            ["intersection_id", "mu_ksi_arm1", "mu_ksi_arm2", "rank_arm1", "rank_arm2",
             "bike_total", "bike_ksi_total"]].to_dict(orient="records"),
        "kendall_tau_prior": float(pd.Series(mu_ksi1_is).corr(pd.Series(mu_ksi2_is), method="kendall")),
        "kendall_tau_eb": float(pd.Series(eb1).corr(pd.Series(eb2), method="kendall")),
    }

    # Decision-relevant head of the ranking: a tool that prioritises the worst
    # ~50 intersections only cares about disagreement there.
    head = tbl[(tbl["rank_arm1"] <= 50) | (tbl["rank_arm2"] <= 50)]
    head20 = tbl[(tbl["rank_arm1"] <= 20) | (tbl["rank_arm2"] <= 20)]
    out["part_D"]["head_of_ranking"] = {
        "n_sites_in_union_of_top50": int(len(head)),
        "rank_shift_within_top50_union": {
            "mean": float(head["rank_shift"].mean()),
            "median": float(head["rank_shift"].median()),
            "max": float(head["rank_shift"].max()),
        },
        "n_sites_in_union_of_top20": int(len(head20)),
        "rank_shift_within_top20_union": {
            "mean": float(head20["rank_shift"].mean()),
            "median": float(head20["rank_shift"].median()),
            "max": float(head20["rank_shift"].max()),
        },
        "top20_union_table": head20.sort_values("rank_arm1")[
            ["intersection_id", "bike_total", "bike_ksi_total", "total_crashes",
             "mu_ksi_arm1", "mu_ksi_arm2", "rank_arm1", "rank_arm2", "rank_shift"]
        ].to_dict(orient="records"),
        "note": "Large absolute rank shifts concentrate in the flat middle of the "
                "distribution where predicted KSI differences are numerically tiny; "
                "the head of the ranking is where a shift would change decisions.",
    }
    print(f"  top50-union rank shift mean={head['rank_shift'].mean():.1f} "
          f"median={head['rank_shift'].median():.1f}; top20-union n={len(head20)} "
          f"mean shift={head20['rank_shift'].mean():.1f}")
    print(f"  prior-rank rho={rho_rank:.4f}  EB-rank rho={rho_eb:.4f}  "
          f"top20 overlap prior={out['part_D']['top20_overlap_prior']}/20")

    # sensitivity: historical v2 leg encoding
    mu_v2legs = predict(pooled_v2legs["result"], df)
    y_tot = df["total_crashes"].to_numpy(float)
    out["provenance_check_all_crash_target"] = {
        "note": "README.md:162 records v2 as sum_pred 1772.7 vs 1720 (+3.1%), MAE 3.47. "
                "Comparing both pooled specs on the all-crash target identifies which "
                "one corresponds to the historical nb_v2_arterial_aadt.",
        "arm1_spec_legs_cat": core_metrics(y_tot, mu_pooled_full),
        "v2_spec_num_legs_continuous": core_metrics(y_tot, mu_v2legs),
        "readme_recorded_v2": {"sum_pred": 1772.7, "sum_actual": 1720,
                               "calibration_pct": 3.1, "mae": 3.47},
    }
    out["sensitivity_v2_leg_encoding"] = {
        "note": "Historical v2 used num_legs as a continuous slope. Arm 1 above uses the "
                "v3 top-coded legs_cat so the A/B isolates pooled-vs-per-mode. This block "
                "checks the leg encoding does not drive the result.",
        "spearman_mu_v2legs_vs_mu_arm1": float(spearmanr(mu_v2legs, mu_pooled_full).statistic),
        "bike": core_metrics(df["bike_total"].to_numpy(float),
                             mu_v2legs * float(df["bike_total"].sum() / df["total_crashes"].sum())),
        "bike_ksi": core_metrics(y_ksi, mu_v2legs * share_v2_full),
    }

    # -----------------------------------------------------------------------
    # Repeated CV across 10 fold seeds — the Part A/B effects are ~1% of MAE,
    # so a single random_state=0 split cannot establish sign stability.
    # -----------------------------------------------------------------------
    print("\n=== Repeated CV across 10 fold seeds ===")
    rep = {lab: {"mae1": [], "mae2": [], "rmse1": [], "rmse2": [], "rho1": [], "rho2": []}
           for lab in MODE_LABELS}
    rep_ksi = {"mae1": [], "mae2": [], "rmse1": [], "rmse2": [], "rho1": [], "rho2": []}
    rep_failures = []                     # every fit that failed production's ladder
    seed_clean = {"pooled": [], **{lab: [] for lab in MODE_LABELS}}

    for seed in range(10):
        o_pool, o_mode = np.full(n, np.nan), {l: np.full(n, np.nan) for l in MODE_LABELS}
        o_sh = {l: np.full(n, np.nan) for l in MODE_LABELS}
        o_kp, o_kb = np.full(n, np.nan), np.full(n, np.nan)
        ok = {"pooled": True, **{lab: True for lab in MODE_LABELS}}

        for fold_j, (tr, te) in enumerate(
            KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed).split(df)
        ):
            trd, ted = df.iloc[tr], df.iloc[te]
            fp = fit_nb(POOLED_FORMULA, trd, f"pooled_seed{seed}_fold{fold_j}")
            if not fp["record"]["converged"]:
                rep_failures.append(fp["record"]); ok["pooled"] = False
            o_pool[te] = predict(fp["result"], ted)
            for lab in MODE_LABELS:
                f_ = fit_nb(MODE_FORMULA[lab], trd, f"permode_{lab}_seed{seed}_fold{fold_j}")
                if not f_["record"]["converged"]:
                    rep_failures.append(f_["record"]); ok[lab] = False
                o_mode[lab][te] = predict(f_["result"], ted)
                o_sh[lab][te] = float(trd[MODE_TARGET[lab]].sum() / trd["total_crashes"].sum())
            o_kp[te] = float(trd["bike_ksi_total"].sum() / trd["total_crashes"].sum())
            o_kb[te] = float(trd["bike_ksi_total"].sum() / trd["bike_total"].sum())

        for key in ok:
            seed_clean[key].append(bool(ok[key]))

        for lab in MODE_LABELS:
            yy = df[MODE_TARGET[lab]].to_numpy(float)
            m1 = core_metrics(yy, o_pool * o_sh[lab])
            m2 = core_metrics(yy, o_mode[lab])
            rep[lab]["mae1"].append(m1["mae"]); rep[lab]["mae2"].append(m2["mae"])
            rep[lab]["rmse1"].append(m1["rmse"]); rep[lab]["rmse2"].append(m2["rmse"])
            rep[lab]["rho1"].append(m1["spearman"]); rep[lab]["rho2"].append(m2["spearman"])

        k1 = core_metrics(y_ksi, o_pool * o_kp)
        k2 = core_metrics(y_ksi, o_mode["bike"] * o_kb)
        rep_ksi["mae1"].append(k1["mae"]); rep_ksi["mae2"].append(k2["mae"])
        rep_ksi["rmse1"].append(k1["rmse"]); rep_ksi["rmse2"].append(k2["rmse"])
        rep_ksi["rho1"].append(k1["spearman"]); rep_ksi["rho2"].append(k2["spearman"])
        flag = "" if (ok["pooled"] and ok["bike"]) else "   <-- NON-CONVERGED FIT IN THIS SEED"
        print(f"  seed {seed}: bike MAE {rep['bike']['mae1'][-1]:.4f}/{rep['bike']['mae2'][-1]:.4f}  "
              f"ksi rho {k1['spearman']:+.4f}/{k2['spearman']:+.4f}{flag}")

    def _rep_block(d: dict, keep: list[bool]) -> dict:
        """Aggregate across seeds. `keep` marks seeds where BOTH arms converged
        under the production ladder; contaminated seeds are excluded from the
        aggregate but retained per-seed and counted."""
        keep_arr = np.array(keep, dtype=bool)
        o = {"n_seeds_total": int(len(keep)), "n_seeds_clean": int(keep_arr.sum()),
             "seed_converged_flags": [bool(v) for v in keep]}
        for metric in ("mae", "rmse", "rho"):
            a1_all = np.array(d[f"{metric}1"], float); a2_all = np.array(d[f"{metric}2"], float)
            a1, a2 = a1_all[keep_arr], a2_all[keep_arr]
            diff = a2 - a1
            better = np.sum(diff < 0) if metric != "rho" else np.sum(diff > 0)
            o[metric] = {
                "arm1_mean": float(a1.mean()), "arm1_sd": float(a1.std(ddof=1)),
                "arm2_mean": float(a2.mean()), "arm2_sd": float(a2.std(ddof=1)),
                "diff_mean": float(diff.mean()), "diff_sd": float(diff.std(ddof=1)),
                "diff_min": float(diff.min()), "diff_max": float(diff.max()),
                "n_seeds_arm2_better": int(better),
                "n_seeds_used": int(len(diff)),
                "arm1_per_seed_all": [float(v) for v in a1_all],
                "arm2_per_seed_all": [float(v) for v in a2_all],
            }
        return o

    keep_mode = {lab: [p and m for p, m in zip(seed_clean["pooled"], seed_clean[lab])]
                 for lab in MODE_LABELS}

    out["repeated_cv_10_seeds"] = {
        "n_seeds": 10, "seeds": list(range(10)), "n_splits": N_SPLITS,
        "n_fits": 10 * N_SPLITS * 4,
        "all_converged": len(rep_failures) == 0,
        "n_failed_fits": len(rep_failures),
        "failed_fits": rep_failures,
        "seed_convergence_flags": seed_clean,
        "part_A": {lab: _rep_block(rep[lab], keep_mode[lab]) for lab in MODE_LABELS},
        "part_B_bike_ksi": _rep_block(rep_ksi, keep_mode["bike"]),
        "note": "'n_seeds_arm2_better' counts seeds where per-mode wins (lower "
                "MAE/RMSE, higher rho). Aggregates EXCLUDE seeds containing a fit "
                "that failed production's newton->bfgs ladder; per-seed values for "
                "all 10 seeds are kept in *_per_seed_all.",
    }
    for lab in MODE_LABELS:
        b = out["repeated_cv_10_seeds"]["part_A"][lab]
        print(f"  {lab:8s} [{b['n_seeds_clean']}/10 clean seeds] MAE arm2-arm1 "
              f"mean={b['mae']['diff_mean']:+.5f} sd={b['mae']['diff_sd']:.5f} "
              f"arm2 wins {b['mae']['n_seeds_arm2_better']}/{b['mae']['n_seeds_used']}")
    bk = out["repeated_cv_10_seeds"]["part_B_bike_ksi"]
    print(f"  bikeKSI [{bk['n_seeds_clean']}/10 clean] rho arm2-arm1 mean={bk['rho']['diff_mean']:+.4f} "
          f"sd={bk['rho']['diff_sd']:.4f} arm2 wins {bk['rho']['n_seeds_arm2_better']}/{bk['rho']['n_seeds_used']}")
    print(f"  TOTAL FITS FAILING PRODUCTION LADDER: {len(rep_failures)} of {10*N_SPLITS*4}")

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nWrote {JSON_PATH}")


if __name__ == "__main__":
    main()
