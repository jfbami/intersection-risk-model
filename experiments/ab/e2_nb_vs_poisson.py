"""E2 - Negative Binomial (NB2) vs Poisson, head to head.

WHY THIS EXPERIMENT EXISTS
--------------------------
pipeline/fit_risk_model.py:363 declares the NB family correct purely because
the estimated dispersion alpha exceeds 0.05:

    verdict = "overdispersed (NB correct)" if alpha > 0.05 else "near-Poisson"

No Poisson model was ever fit, so there is no side-by-side likelihood, no
information criterion comparison, no formal test of H0: alpha = 0, and no
out-of-sample number anywhere in the repo. This script turns that assertion
into a measurement.

WHAT IT RUNS
------------
For each of the three modes (bike / ped / vehicle), the SAME formula and the
SAME offset are fit under:
  (a) NB2      smf.negativebinomial(...)  -- the production spec
  (b) Poisson  smf.poisson(...)

Reported: LL, AIC, BIC, convergence, alpha (NB), 5-fold CV MAE/RMSE, the
boundary-corrected likelihood-ratio test of alpha = 0, and 90% predictive
interval coverage for both families.

Run:
  cd C:/Users/jfbaa/project-cycle-group && \
    PYTHONPATH=C:/Users/jfbaa/project-cycle-group \
    python experiments/ab/e2_nb_vs_poisson.py
"""

from __future__ import annotations

import sys
import traceback

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats as sps

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

from e2_e6_e7_common import (  # noqa: E402
    RESULTS_DIR,
    _fit_capture,
    design,
    dump_json,
    environment,
    fit_summary,
    get_modelling_frame,
    kfold_indices,
    mae_rmse,
    mean_sd,
    modes,
    nb_interval_coverage,
    poisson_interval_coverage,
)


# ---------------------------------------------------------------------------
# Array-API fitters with an explicit retry ladder (every attempt is recorded)
# ---------------------------------------------------------------------------


def fit_poisson_array(y, X, offset):
    model = sm.Poisson(y, X, offset=offset)
    attempts = []
    for kw, tag in [
        ({}, "default(newton,maxiter=35)"),
        ({"method": "bfgs", "maxiter": 500}, "bfgs,maxiter=500"),
    ]:
        try:
            res, warns = _fit_capture(model, **kw)
            ok = (not warns) and bool(res.mle_retvals.get("converged", True))
            attempts.append({"attempt": tag, "converged": ok, "warnings": warns})
            if ok:
                return res, attempts
        except Exception as exc:  # pragma: no cover
            attempts.append({"attempt": tag, "converged": False, "error": repr(exc)})
    return None, attempts


def fit_formula_ladder(model, family: str):
    """Mirror of pipeline.fit_risk_model.fit_for_mode: try the statsmodels
    default first, then method='bfgs', maxiter=200 (the exact production
    retry). Two further rungs are added so a failure is a *measured* failure
    rather than an artefact of a short optimiser budget. Every attempt is
    recorded."""
    if family == "nb":
        ladder = [
            ({}, "default(bfgs,maxiter=35)"),
            ({"method": "bfgs", "maxiter": 200}, "bfgs,maxiter=200 [production retry]"),
            ({"method": "bfgs", "maxiter": 500}, "bfgs,maxiter=500"),
            ({"method": "nm", "maxiter": 5000}, "nelder-mead,maxiter=5000"),
        ]
    else:
        ladder = [
            ({}, "default(newton,maxiter=35)"),
            ({"method": "bfgs", "maxiter": 500}, "bfgs,maxiter=500"),
        ]
    attempts = []
    for kw, tag in ladder:
        try:
            res, warns = _fit_capture(model, **kw)
            ok = (not warns) and bool(res.mle_retvals.get("converged", True))
            attempts.append(
                {"attempt": tag, "converged": ok, "loglik": float(res.llf), "warnings": warns}
            )
            if ok:
                return res, warns, attempts, tag
        except Exception as exc:  # pragma: no cover
            attempts.append({"attempt": tag, "converged": False, "error": repr(exc)})
    return None, None, attempts, None


def fit_nb_array(y, X, offset, start_params=None):
    model = sm.NegativeBinomial(y, X, offset=offset)
    attempts = []
    ladder = [
        ({}, "default(bfgs,maxiter=35)"),
        ({"method": "bfgs", "maxiter": 500}, "bfgs,maxiter=500"),
        ({"method": "nm", "maxiter": 3000}, "nelder-mead,maxiter=3000"),
    ]
    if start_params is not None:
        ladder.insert(
            1,
            (
                {"start_params": start_params, "method": "bfgs", "maxiter": 500},
                "bfgs,maxiter=500,start=poisson",
            ),
        )
    for kw, tag in ladder:
        try:
            res, warns = _fit_capture(model, **kw)
            ok = (not warns) and bool(res.mle_retvals.get("converged", True))
            attempts.append({"attempt": tag, "converged": ok, "warnings": warns})
            if ok:
                return res, attempts
        except Exception as exc:  # pragma: no cover
            attempts.append({"attempt": tag, "converged": False, "error": repr(exc)})
    return None, attempts


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def cross_validate(y, X, offset, family: str) -> dict:
    n = len(y)
    folds, backend = kfold_indices(n)
    Xv = np.asarray(X, dtype=float)

    fold_rows = []
    oof_mu = np.full(n, np.nan)
    oof_alpha = np.full(n, np.nan)

    for i, (tr, te) in enumerate(folds):
        if family == "poisson":
            res, attempts = fit_poisson_array(y[tr], Xv[tr], offset[tr])
        else:
            pois, _ = fit_poisson_array(y[tr], Xv[tr], offset[tr])
            sp = None
            if pois is not None:
                sp = np.append(np.asarray(pois.params, dtype=float), 1.0)
            res, attempts = fit_nb_array(y[tr], Xv[tr], offset[tr], start_params=sp)

        if res is None:
            fold_rows.append(
                {
                    "fold": i,
                    "n_train": int(len(tr)),
                    "n_test": int(len(te)),
                    "converged": False,
                    "attempts": attempts,
                    "mae": None,
                    "rmse": None,
                    "alpha": None,
                }
            )
            continue

        beta = np.asarray(res.params, dtype=float)
        if family == "nb":
            alpha = float(beta[-1])
            beta = beta[:-1]
        else:
            alpha = None
        mu = np.exp(Xv[te] @ beta + offset[te])
        mae, rmse = mae_rmse(y[te], mu)
        oof_mu[te] = mu
        if alpha is not None:
            oof_alpha[te] = alpha
        fold_rows.append(
            {
                "fold": i,
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "converged": True,
                "attempts": attempts,
                "mae": mae,
                "rmse": rmse,
                "alpha": alpha,
            }
        )

    pooled = {"mae": None, "rmse": None}
    if np.isfinite(oof_mu).all():
        pm, pr = mae_rmse(y, oof_mu)
        pooled = {"mae": pm, "rmse": pr}

    # out-of-fold predictive coverage
    oof_cov = None
    if np.isfinite(oof_mu).all():
        if family == "poisson":
            oof_cov = poisson_interval_coverage(y, oof_mu)
        else:
            lo = sps.nbinom.ppf(
                0.05, 1.0 / oof_alpha, 1.0 / (1.0 + oof_alpha * np.clip(oof_mu, 1e-9, None))
            )
            hi = sps.nbinom.ppf(
                0.95, 1.0 / oof_alpha, 1.0 / (1.0 + oof_alpha * np.clip(oof_mu, 1e-9, None))
            )
            oof_cov = {
                "coverage_pct": float(np.mean((y >= lo) & (y <= hi)) * 100),
                "mean_width": float(np.mean(hi - lo)),
                "median_width": float(np.median(hi - lo)),
            }

    return {
        "backend": backend,
        "n_splits": len(folds),
        "seed": 0,
        "folds": fold_rows,
        "mae_across_folds": mean_sd([f["mae"] for f in fold_rows]),
        "rmse_across_folds": mean_sd([f["rmse"] for f in fold_rows]),
        "pooled_out_of_fold": pooled,
        "out_of_fold_coverage_90": oof_cov,
        "all_folds_converged": all(f["converged"] for f in fold_rows),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    df, prep = get_modelling_frame()
    print(f"Modelling frame: {len(df)} rows")
    print(f"years_observed unique values: {df['years_observed'].unique().tolist()}")

    out = {
        "experiment": "E2 - NB2 vs Poisson",
        "environment": environment(),
        "n_rows": int(len(df)),
        "modes": {},
    }

    for mode in modes():
        label = mode.label
        formula = f"{mode.target} ~ {mode.predictors}"
        y_col = df[mode.target].astype(float)
        print("\n" + "=" * 78)
        print(f"  {mode.display_name}  ({mode.target})")
        print("=" * 78)
        print(f"formula: {formula}")

        rec: dict = {
            "target": mode.target,
            "formula": formula,
            "offset": "log(years_observed) == log(6) == 1.791759 for every row",
            "descriptives": {
                "n_events": int(y_col.sum()),
                "mean": float(y_col.mean()),
                "variance": float(y_col.var(ddof=1)),
                "variance_over_mean": float(y_col.var(ddof=1) / y_col.mean()),
                "max": float(y_col.max()),
                "n_zero_sites": int((y_col == 0).sum()),
            },
        }
        d = rec["descriptives"]
        print(
            f"events={d['n_events']}  mean={d['mean']:.4f}  var={d['variance']:.4f}  "
            f"var/mean={d['variance_over_mean']:.3f}  zeros={d['n_zero_sites']}"
        )

        # ---------------- production formula fits ----------------
        nb_res, nb_warns, nb_attempts, nb_tag = fit_formula_ladder(
            smf.negativebinomial(formula, data=df, offset=df["offset"].values), "nb"
        )
        po_res, po_warns, po_attempts, po_tag = fit_formula_ladder(
            smf.poisson(formula, data=df, offset=df["offset"].values), "poisson"
        )
        if nb_res is None or po_res is None:
            raise RuntimeError(
                f"{label}: a full-data fit failed on every rung. "
                f"NB attempts={nb_attempts} Poisson attempts={po_attempts}"
            )

        n = len(df)
        rec["nb"] = fit_summary(nb_res, nb_warns, "NB2 (production)", n)
        rec["poisson"] = fit_summary(po_res, po_warns, "Poisson", n)
        rec["nb"]["fit_attempts"] = nb_attempts
        rec["nb"]["converged_on"] = nb_tag
        rec["poisson"]["fit_attempts"] = po_attempts
        rec["poisson"]["converged_on"] = po_tag
        print(f"NB2 converged on rung: {nb_tag}   Poisson converged on rung: {po_tag}")

        alpha = float(nb_res.params["alpha"])
        a_se = float(nb_res.bse["alpha"])
        a_ci = nb_res.conf_int().loc["alpha"].tolist()
        rec["nb"]["alpha"] = alpha
        rec["nb"]["alpha_se"] = a_se
        rec["nb"]["alpha_z"] = float(nb_res.tvalues["alpha"])
        rec["nb"]["alpha_wald_p_two_sided"] = float(nb_res.pvalues["alpha"])
        rec["nb"]["alpha_ci95"] = [float(a_ci[0]), float(a_ci[1])]

        print(
            f"NB2     LL={rec['nb']['loglik']:.3f}  AIC={rec['nb']['aic_manual']:.2f}  "
            f"BIC={rec['nb']['bic_manual']:.2f}  k={rec['nb']['k_params_incl_dispersion']}  "
            f"converged={rec['nb']['converged']}  alpha={alpha:.4f} (SE {a_se:.4f})"
        )
        print(
            f"Poisson LL={rec['poisson']['loglik']:.3f}  AIC={rec['poisson']['aic_manual']:.2f}  "
            f"BIC={rec['poisson']['bic_manual']:.2f}  k={rec['poisson']['k_params_incl_dispersion']}  "
            f"converged={rec['poisson']['converged']}"
        )

        # ---------------- LR test of H0: alpha = 0 ----------------
        lr = float(2.0 * (nb_res.llf - po_res.llf))
        rec["lr_test_alpha_zero"] = {
            "LR_statistic": lr,
            "naive_chi2_1df_p": float(sps.chi2.sf(lr, 1)),
            "boundary_corrected_p": float(0.5 * sps.chi2.sf(lr, 1)),
            "note": (
                "alpha=0 lies on the boundary of the parameter space, so the null "
                "distribution of LR is the 50:50 mixture 0.5*chi2(0) + 0.5*chi2(1); "
                "the naive chi2(1) p-value is therefore exactly twice the correct one "
                "(i.e. conservative - it understates the evidence against Poisson)."
            ),
        }
        t = rec["lr_test_alpha_zero"]
        print(
            f"LR(alpha=0) = {lr:.3f}   naive chi2(1) p = {t['naive_chi2_1df_p']:.3e}   "
            f"boundary-corrected p = {t['boundary_corrected_p']:.3e}"
        )

        # ---------------- design matrix + array-API sanity check ----------------
        yv, X, _ = design(formula, df)
        Xv = np.asarray(X, dtype=float)
        offv = df["offset"].values.astype(float)

        mu_nb_formula = np.asarray(nb_res.predict(df, offset=offv), dtype=float)
        mu_nb_manual = np.exp(Xv @ np.asarray(nb_res.params, dtype=float)[:-1] + offv)
        mu_po_formula = np.asarray(po_res.predict(df, offset=offv), dtype=float)
        mu_po_manual = np.exp(Xv @ np.asarray(po_res.params, dtype=float) + offv)
        rec["sanity"] = {
            "design_columns": list(X.columns),
            "design_rank": int(np.linalg.matrix_rank(Xv)),
            "design_ncols": int(Xv.shape[1]),
            "max_abs_diff_nb_predict_vs_manual": float(
                np.max(np.abs(mu_nb_formula - mu_nb_manual))
            ),
            "max_abs_diff_poisson_predict_vs_manual": float(
                np.max(np.abs(mu_po_formula - mu_po_manual))
            ),
        }
        print(
            f"design: {rec['sanity']['design_ncols']} cols, rank "
            f"{rec['sanity']['design_rank']}   predict-vs-manual max|diff| NB="
            f"{rec['sanity']['max_abs_diff_nb_predict_vs_manual']:.2e} "
            f"Pois={rec['sanity']['max_abs_diff_poisson_predict_vs_manual']:.2e}"
        )

        # ---------------- in-sample 90% predictive coverage ----------------
        rec["nb"]["in_sample_coverage_90"] = nb_interval_coverage(yv, mu_nb_formula, alpha)
        rec["poisson"]["in_sample_coverage_90"] = poisson_interval_coverage(yv, mu_po_formula)
        cn, cp = rec["nb"]["in_sample_coverage_90"], rec["poisson"]["in_sample_coverage_90"]
        rec["coverage_comparison"] = {
            "nb_minus_poisson_coverage_pts": cn["coverage_pct"] - cp["coverage_pct"],
            "nb_over_poisson_mean_width_ratio": (
                cn["mean_width"] / cp["mean_width"] if cp["mean_width"] > 0 else None
            ),
            "n_sites_outside_poisson_interval": int(
                round((100 - cp["coverage_pct"]) / 100 * len(df))
            ),
            "n_sites_outside_nb_interval": int(
                round((100 - cn["coverage_pct"]) / 100 * len(df))
            ),
        }
        print(
            f"90% PI coverage  NB={cn['coverage_pct']:.1f}% (mean width {cn['mean_width']:.2f})   "
            f"Poisson={cp['coverage_pct']:.1f}% (mean width {cp['mean_width']:.2f})"
        )

        # ---------------- in-sample point-fit metrics ----------------
        m_nb = mae_rmse(yv, mu_nb_formula)
        m_po = mae_rmse(yv, mu_po_formula)
        rec["nb"]["in_sample_mae"], rec["nb"]["in_sample_rmse"] = m_nb
        rec["poisson"]["in_sample_mae"], rec["poisson"]["in_sample_rmse"] = m_po

        # ---------------- cross-validation ----------------
        print("running 5-fold CV ...")
        rec["nb"]["cv"] = cross_validate(yv, X, offv, "nb")
        rec["poisson"]["cv"] = cross_validate(yv, X, offv, "poisson")
        for fam in ("nb", "poisson"):
            cv = rec[fam]["cv"]
            print(
                f"  CV {fam:<7} MAE {cv['mae_across_folds']['mean']:.4f} "
                f"+/- {cv['mae_across_folds']['sd']:.4f}   "
                f"RMSE {cv['rmse_across_folds']['mean']:.4f} "
                f"+/- {cv['rmse_across_folds']['sd']:.4f}   "
                f"pooled MAE {cv['pooled_out_of_fold']['mae']:.4f} / RMSE "
                f"{cv['pooled_out_of_fold']['rmse']:.4f}   "
                f"all folds converged: {cv['all_folds_converged']}"
            )

        # paired per-fold difference (NB - Poisson) to judge signal vs noise
        d_mae = [
            a["mae"] - b["mae"]
            for a, b in zip(rec["nb"]["cv"]["folds"], rec["poisson"]["cv"]["folds"])
            if a["mae"] is not None and b["mae"] is not None
        ]
        d_rmse = [
            a["rmse"] - b["rmse"]
            for a, b in zip(rec["nb"]["cv"]["folds"], rec["poisson"]["cv"]["folds"])
            if a["rmse"] is not None and b["rmse"] is not None
        ]
        rec["cv_paired_difference_nb_minus_poisson"] = {
            "mae_delta": mean_sd(d_mae),
            "rmse_delta": mean_sd(d_rmse),
            "mae_delta_per_fold": d_mae,
            "rmse_delta_per_fold": d_rmse,
        }
        if len(d_mae) > 1:
            tt = sps.ttest_1samp(d_mae, 0.0)
            rec["cv_paired_difference_nb_minus_poisson"]["mae_paired_t_p"] = float(tt.pvalue)
        print(
            "  paired per-fold MAE delta (NB - Poisson): "
            f"{mean_sd(d_mae)['mean']:+.5f} +/- {mean_sd(d_mae)['sd']:.5f}"
        )

        out["modes"][label] = rec

    dump_json(out, RESULTS_DIR / "e2_results.json")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
