"""E7 - Negative Binomial vs Zero-Inflated Negative Binomial.

WHY THIS EXPERIMENT EXISTS
--------------------------
pipeline/evaluate_models.py contains a zero-inflation check
(print_zero_prediction_check, line ~164) that compares observed zero-count
sites against the NB-implied expected number of zeros. It has never run.
This script re-runs that check standalone and then goes further: it actually
fits the zero-inflated alternative, which the repo never has. Bike crashes
are sparse (169 events over 346 sites, 256 of them zero), so excess zeros
are plausible and untested.

APPROACH / API CHOICE
---------------------
ZINB is fit with the statsmodels ARRAY API
(statsmodels.discrete.count_model.ZeroInflatedNegativeBinomialP) on a patsy
design matrix built from the production formula, rather than the formula API.
Reasons: the inflation design is then explicit (exog_infl = a single constant
column, i.e. a constant excess-zero probability), the same design matrix is
reused for the NB comparator and for every CV fold so the two families are
compared on identical columns, and warm starts can be handed in as plain
arrays. offset = log(years_observed) exactly as in production.

These models are fragile. Every optimiser rung that is tried is recorded, and
the converged rung with the highest log-likelihood is the one reported. If no
rung converges, that is reported as a failure - no number is forced.

Run:
  cd C:/Users/jfbaa/project-cycle-group && \
    PYTHONPATH=C:/Users/jfbaa/project-cycle-group \
    python experiments/ab/e7_nb_vs_zinb.py
"""

from __future__ import annotations

import sys
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sps
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP

sys.path.insert(0, str(Path(__file__).resolve().parent))

from e2_e6_e7_common import (  # noqa: E402
    RESULTS_DIR,
    _fit_capture,
    design,
    dump_json,
    environment,
    get_modelling_frame,
    kfold_indices,
    mae_rmse,
    mean_sd,
    modes,
)


# ---------------------------------------------------------------------------
# Fitters
# ---------------------------------------------------------------------------


def fit_nb_best(y, X, offset):
    """NB2, best converged rung by log-likelihood."""
    y = np.asarray(y, float)
    Xv = np.asarray(X, float)
    model = sm.NegativeBinomial(y, Xv, offset=offset)
    poisson_start = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pres = sm.Poisson(y, Xv, offset=offset).fit(disp=False)
        mu = np.exp(Xv @ np.asarray(pres.params, float) + offset)
        a0 = float(
            np.clip(np.mean(((y - mu) ** 2 - mu) / np.clip(mu ** 2, 1e-9, None)), 1e-3, 5)
        )
        poisson_start = np.append(np.asarray(pres.params, float), a0)
    except Exception:
        pass

    ladder = [
        ({}, "default(bfgs,maxiter=35)"),
        ({"method": "bfgs", "maxiter": 200}, "bfgs,maxiter=200 [production retry]"),
        ({"method": "bfgs", "maxiter": 1000}, "bfgs,maxiter=1000"),
        ({"method": "nm", "maxiter": 8000}, "nelder-mead,maxiter=8000"),
    ]
    if poisson_start is not None:
        ladder.append(
            (
                {"start_params": poisson_start, "method": "bfgs", "maxiter": 1000},
                "bfgs,maxiter=1000,start=poisson+MoM_alpha",
            )
        )
    return _run_ladder(model, ladder)


def fit_zinb_best(y, X, offset, nb_params=None):
    """ZINB-P (p=2, i.e. NB2 main part) with a constant-only logit inflation
    equation. Best converged rung by log-likelihood."""
    y = np.asarray(y, float)
    Xv = np.asarray(X, float)
    model = ZeroInflatedNegativeBinomialP(
        y, Xv, exog_infl=None, offset=offset, inflation="logit", p=2
    )
    ladder = [
        ({}, "default(bfgs,maxiter=35)"),
        ({"method": "bfgs", "maxiter": 500}, "bfgs,maxiter=500 [task-specified retry]"),
        ({"method": "bfgs", "maxiter": 2000}, "bfgs,maxiter=2000"),
        ({"method": "nm", "maxiter": 20000}, "nelder-mead,maxiter=20000"),
    ]
    if nb_params is not None:
        # param order for ZINB-P: [inflation params, main exog params, alpha]
        for g0 in (-3.0, -1.0):
            ws = np.concatenate([[g0], np.asarray(nb_params, float)])
            ladder.append(
                (
                    {"start_params": ws, "method": "bfgs", "maxiter": 2000},
                    f"bfgs,maxiter=2000,start=NB+logit_intercept({g0:.0f})",
                )
            )
            ladder.append(
                (
                    {"start_params": ws, "method": "nm", "maxiter": 20000},
                    f"nelder-mead,start=NB+logit_intercept({g0:.0f})",
                )
            )
    return _run_ladder(model, ladder)


def _run_ladder(model, ladder):
    attempts, best = [], None
    for kw, tag in ladder:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                res, warns = _fit_capture(model, **kw)
            ok = (not warns) and bool(res.mle_retvals.get("converged", True))
            llf = float(res.llf)
            attempts.append(
                {"attempt": tag, "converged": ok, "loglik": llf, "warnings": warns}
            )
            if ok and np.isfinite(llf) and (best is None or llf > best[1]):
                best = (res, llf, tag)
        except Exception as exc:
            attempts.append({"attempt": tag, "converged": False, "error": repr(exc)})
    if best is None:
        return None, attempts, None
    return best[0], attempts, best[2] + " [selected: highest LL among converged rungs]"


# ---------------------------------------------------------------------------
# Prediction helpers (computed explicitly, cross-checked against statsmodels)
# ---------------------------------------------------------------------------


def nb_pieces(res, X, offset):
    p = np.asarray(res.params, float)
    beta, alpha = p[:-1], float(p[-1])
    mu = np.exp(np.asarray(X, float) @ beta + offset)
    return mu, alpha


def nb_p_zero(mu, alpha):
    n = 1.0 / alpha
    return (n / (n + mu)) ** n


def zinb_pieces(res, X, offset):
    """Returns (pi, mu_main, alpha). Param order: [gamma (1), beta (k), alpha]."""
    p = np.asarray(res.params, float)
    gamma = p[0]
    beta = p[1:-1]
    alpha = float(p[-1])
    pi = 1.0 / (1.0 + np.exp(-gamma))  # constant-only logit inflation
    mu = np.exp(np.asarray(X, float) @ beta + offset)
    return float(pi), mu, alpha


def zinb_p_zero(pi, mu, alpha):
    return pi + (1.0 - pi) * nb_p_zero(mu, alpha)


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def cv_family(y, X, offset, family: str) -> dict:
    y = np.asarray(y, float)
    Xv = np.asarray(X, float)
    n = len(y)
    folds, backend = kfold_indices(n)
    maes, rmses = [], []
    oof = np.full(n, np.nan)
    fold_rows = []
    for i, (tr, te) in enumerate(folds):
        nb, _, _ = fit_nb_best(y[tr], Xv[tr], offset[tr])
        if family == "nb":
            res = nb
        else:
            res, _, _ = fit_zinb_best(
                y[tr],
                Xv[tr],
                offset[tr],
                nb_params=np.asarray(nb.params, float) if nb is not None else None,
            )
        if res is None:
            fold_rows.append({"fold": i, "converged": False})
            continue
        if family == "nb":
            mu, _ = nb_pieces(res, Xv[te], offset[te])
            pred = mu
        else:
            pi, mu, _ = zinb_pieces(res, Xv[te], offset[te])
            pred = (1.0 - pi) * mu  # marginal mean of a ZI model
        oof[te] = pred
        a, b = mae_rmse(y[te], pred)
        maes.append(a)
        rmses.append(b)
        fold_rows.append({"fold": i, "converged": True, "mae": a, "rmse": b})
    pooled = mae_rmse(y, oof) if np.isfinite(oof).all() else (None, None)
    return {
        "backend": backend,
        "folds": fold_rows,
        "n_folds_failed": int(sum(1 for f in fold_rows if not f["converged"])),
        "mae_across_folds": mean_sd(maes),
        "rmse_across_folds": mean_sd(rmses),
        "pooled_out_of_fold_mae": pooled[0],
        "pooled_out_of_fold_rmse": pooled[1],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    df, _ = get_modelling_frame()
    out = {
        "experiment": "E7 - NB vs Zero-Inflated NB",
        "environment": environment(),
        "n_rows": int(len(df)),
        "api_note": (
            "ZINB fit via statsmodels ARRAY API "
            "(ZeroInflatedNegativeBinomialP, p=2, inflation='logit', "
            "exog_infl = constant only) on a patsy design matrix built from the "
            "production formula; offset = log(years_observed)."
        ),
        "modes": {},
    }
    off = df["offset"].values.astype(float)
    n = len(df)

    for mode in modes():
        formula = f"{mode.target} ~ {mode.predictors}"
        print("\n" + "=" * 78)
        print(f"  {mode.display_name}  ({mode.target})")
        print("=" * 78)
        y, X, _ = design(formula, df)
        Xv = np.asarray(X, float)
        rec = {"target": mode.target, "formula": formula, "n_events": int(y.sum())}

        # ---------------- NB ----------------
        nb, nb_att, nb_tag = fit_nb_best(y, Xv, off)
        if nb is None:
            rec["nb"] = {"converged": False, "fit_attempts": nb_att}
            print("  NB FAILED on every rung; skipping mode.")
            out["modes"][mode.label] = rec
            continue
        mu_nb, alpha = nb_pieces(nb, Xv, off)
        k_nb = len(nb.params)
        pz_nb = nb_p_zero(mu_nb, alpha)
        obs_zero = int((y == 0).sum())

        rec["observed_zero_sites"] = obs_zero
        rec["n_sites"] = n
        rec["nb"] = {
            "converged": True,
            "converged_on": nb_tag,
            "fit_attempts": nb_att,
            "loglik": float(nb.llf),
            "k_params": int(k_nb),
            "aic_manual": float(-2 * nb.llf + 2 * k_nb),
            "bic_manual": float(-2 * nb.llf + np.log(n) * k_nb),
            "aic_statsmodels": float(nb.aic),
            "bic_statsmodels": float(nb.bic),
            "alpha": alpha,
            "expected_zero_sites": float(pz_nb.sum()),
            "zero_gap_observed_minus_expected": float(obs_zero - pz_nb.sum()),
            "in_sample_mae": mae_rmse(y, mu_nb)[0],
            "in_sample_rmse": mae_rmse(y, mu_nb)[1],
        }
        # Is the observed-vs-expected zero gap outside sampling noise?
        # Parametric bootstrap: simulate 4000 datasets from the fitted NB and
        # count zeros. This turns "gap = -0.45" into a calibrated statement.
        rng = np.random.default_rng(0)
        n_sim = 4000
        nn = 1.0 / alpha
        pp = 1.0 / (1.0 + alpha * mu_nb)
        sim_zeros = np.array(
            [int((rng.negative_binomial(nn, pp) == 0).sum()) for _ in range(n_sim)]
        )
        lo, hi = np.percentile(sim_zeros, [5, 95])
        rec["zero_count_parametric_bootstrap"] = {
            "n_sim": n_sim,
            "rng": "numpy.random.default_rng(0)",
            "observed_zero_sites": obs_zero,
            "simulated_zero_mean": float(sim_zeros.mean()),
            "simulated_zero_sd": float(sim_zeros.std(ddof=1)),
            "simulated_zero_p05": float(lo),
            "simulated_zero_p95": float(hi),
            "observed_inside_90pct_band": bool(lo <= obs_zero <= hi),
            "p_value_observed_ge_simulated": float(np.mean(sim_zeros >= obs_zero)),
        }
        bs = rec["zero_count_parametric_bootstrap"]
        print(
            f"  observed zero sites: {obs_zero}/{n}   "
            f"NB-implied expected zeros: {pz_nb.sum():.2f}   "
            f"gap: {obs_zero - pz_nb.sum():+.2f}"
        )
        print(
            f"  parametric bootstrap under fitted NB ({n_sim} sims): zeros "
            f"{bs['simulated_zero_mean']:.1f} +/- {bs['simulated_zero_sd']:.1f}, "
            f"90% band [{lo:.0f}, {hi:.0f}]  -> observed {obs_zero} inside band: "
            f"{bs['observed_inside_90pct_band']}"
        )
        print(
            f"  NB   LL={nb.llf:.3f} AIC={rec['nb']['aic_manual']:.2f} "
            f"BIC={rec['nb']['bic_manual']:.2f} k={k_nb} alpha={alpha:.4f}"
            f"\n       converged on: {nb_tag}"
        )

        # ---------------- ZINB ----------------
        zi, zi_att, zi_tag = fit_zinb_best(y, Xv, off, nb_params=np.asarray(nb.params, float))
        if zi is None:
            rec["zinb"] = {
                "converged": False,
                "fit_attempts": zi_att,
                "note": "ZINB did not converge on ANY rung - reported as a failure, "
                "no number forced.",
            }
            print("  ZINB DID NOT CONVERGE ON ANY RUNG. Attempts:")
            for a in zi_att:
                print(f"    - {a}")
            out["modes"][mode.label] = rec
            continue

        pi, mu_zi, alpha_zi = zinb_pieces(zi, Xv, off)
        k_zi = len(zi.params)
        pz_zi = zinb_p_zero(pi, mu_zi, alpha_zi)
        pred_zi = (1.0 - pi) * mu_zi

        # cross-check our explicit formulas against statsmodels' own predict
        xcheck = {}
        try:
            sm_mean = np.asarray(zi.predict(which="mean"), float)
            xcheck["max_abs_diff_mean_vs_statsmodels"] = float(
                np.max(np.abs(sm_mean - pred_zi))
            )
        except Exception as exc:
            xcheck["mean_predict_error"] = repr(exc)
        try:
            sm_prob = np.asarray(zi.predict(which="prob"), float)
            xcheck["max_abs_diff_p_zero_vs_statsmodels"] = float(
                np.max(np.abs(sm_prob[:, 0] - pz_zi))
            )
        except Exception as exc:
            xcheck["prob_predict_error"] = repr(exc)

        rec["zinb"] = {
            "converged": True,
            "converged_on": zi_tag,
            "fit_attempts": zi_att,
            "loglik": float(zi.llf),
            "k_params": int(k_zi),
            "aic_manual": float(-2 * zi.llf + 2 * k_zi),
            "bic_manual": float(-2 * zi.llf + np.log(n) * k_zi),
            "aic_statsmodels": float(zi.aic),
            "bic_statsmodels": float(zi.bic),
            "alpha": alpha_zi,
            "inflation_logit_intercept": float(np.asarray(zi.params, float)[0]),
            "inflation_logit_intercept_se": float(np.asarray(zi.bse, float)[0]),
            "inflation_probability_pi": pi,
            "expected_zero_sites": float(pz_zi.sum()),
            "zero_gap_observed_minus_expected": float(obs_zero - pz_zi.sum()),
            "in_sample_mae": mae_rmse(y, pred_zi)[0],
            "in_sample_rmse": mae_rmse(y, pred_zi)[1],
            "statsmodels_cross_check": xcheck,
        }
        z = rec["zinb"]
        print(
            f"  ZINB LL={zi.llf:.3f} AIC={z['aic_manual']:.2f} BIC={z['bic_manual']:.2f} "
            f"k={k_zi} alpha={alpha_zi:.4f} pi={pi:.6e}"
            f"\n       logit intercept={z['inflation_logit_intercept']:.4f} "
            f"(SE {z['inflation_logit_intercept_se']:.4f})"
            f"\n       converged on: {zi_tag}"
            f"\n       expected zeros: {z['expected_zero_sites']:.2f} "
            f"(gap {z['zero_gap_observed_minus_expected']:+.2f})"
            f"\n       cross-check vs statsmodels predict: {xcheck}"
        )

        rec["comparison"] = {
            "delta_loglik_zinb_minus_nb": float(zi.llf - nb.llf),
            "delta_aic_zinb_minus_nb": float(z["aic_manual"] - rec["nb"]["aic_manual"]),
            "delta_bic_zinb_minus_nb": float(z["bic_manual"] - rec["nb"]["bic_manual"]),
            "zero_count_observed": obs_zero,
            "zero_count_expected_nb": float(pz_nb.sum()),
            "zero_count_expected_zinb": float(pz_zi.sum()),
        }

        # ---------------- Vuong ----------------
        try:
            ll_nb_i = np.asarray(nb.model.loglikeobs(np.asarray(nb.params, float)), float)
            ll_zi_i = np.asarray(zi.model.loglikeobs(np.asarray(zi.params, float)), float)
            chk = {
                "sum_ll_nb_obs_vs_llf": float(ll_nb_i.sum() - nb.llf),
                "sum_ll_zinb_obs_vs_llf": float(ll_zi_i.sum() - zi.llf),
            }
            m = ll_zi_i - ll_nb_i
            sd_m = float(np.std(m, ddof=1))
            max_abs_m = float(np.max(np.abs(m)))
            chk["max_abs_per_obs_ll_difference"] = max_abs_m
            chk["sd_per_obs_ll_difference"] = sd_m
            # If the two per-observation log-likelihood vectors agree to
            # numerical noise, the ZINB has collapsed onto the NB and every
            # Vuong variant is 0/0. The parameter-count-corrected variants are
            # then a constant divided by ~0 and explode to meaningless values,
            # so they are suppressed rather than reported.
            # Substantive criterion first: if the estimated inflation
            # probability has gone to the NB boundary (pi ~ 0) the ZINB IS the
            # NB, and any residual per-observation difference is optimiser
            # tolerance, not signal.
            degenerate = (pi < 1e-6) or (max_abs_m < 1e-4)
            chk["zinb_inflation_probability_pi"] = pi
            if degenerate:
                rec["vuong"] = {
                    "computed": True,
                    "degenerate": True,
                    "statistic_raw": float(
                        np.sqrt(len(m)) * np.mean(m) / sd_m
                    ) if sd_m > 0 else 0.0,
                    "p_two_sided": float(
                        2 * sps.norm.sf(abs(np.sqrt(len(m)) * np.mean(m) / sd_m))
                    ) if sd_m > 0 else 1.0,
                    "statistic_aic_corrected": None,
                    "statistic_bic_corrected": None,
                    "suppressed_reason": (
                        f"The fitted inflation probability collapsed to pi = {pi:.3e} "
                        "(the NB boundary), and the fitted ZINB reproduces the NB "
                        f"per-observation log-likelihoods to max|difference| = {max_abs_m:.3e}. The "
                        "parameter-count-corrected Vuong variants are then a fixed "
                        "constant divided by a numerical-noise standard deviation and "
                        "take arbitrary huge values; they carry no information and are "
                        "suppressed. The raw statistic is ~0, i.e. the two models are "
                        "numerically the same model."
                    ),
                    "loglikeobs_check": chk,
                    "mean_per_obs_ll_difference": float(np.mean(m)),
                }
                print(
                    f"  Vuong: DEGENERATE - pi collapsed to {pi:.3e} (NB boundary); "
                    f"ZINB reproduces NB per-obs LL to max|diff|={max_abs_m:.3e}; "
                    f"raw V~0, corrected variants suppressed as meaningless."
                )
                print(f"  loglikeobs reconciliation vs .llf: {chk}")
            elif sd_m <= 0 or not np.isfinite(sd_m):
                rec["vuong"] = {
                    "computed": False,
                    "reason": "per-observation log-likelihood difference has zero/non-finite "
                    "variance, so the Vuong statistic is undefined",
                    "loglikeobs_check": chk,
                    "mean_diff": float(np.mean(m)),
                }
            else:
                V = float(np.sqrt(len(m)) * np.mean(m) / sd_m)
                # Akaike/Schwarz-corrected variants (Vuong 1989 corrections for
                # the difference in the number of free parameters)
                dk = k_zi - k_nb
                m_aic = m - dk / len(m)
                m_bic = m - (dk * np.log(len(m)) / 2.0) / len(m)
                rec["vuong"] = {
                    "computed": True,
                    "statistic_raw": V,
                    "p_two_sided": float(2 * sps.norm.sf(abs(V))),
                    "p_one_sided_favouring_zinb": float(sps.norm.sf(V)),
                    "statistic_aic_corrected": float(
                        np.sqrt(len(m)) * np.mean(m_aic) / np.std(m_aic, ddof=1)
                    ),
                    "statistic_bic_corrected": float(
                        np.sqrt(len(m)) * np.mean(m_bic) / np.std(m_bic, ddof=1)
                    ),
                    "mean_per_obs_ll_difference": float(np.mean(m)),
                    "loglikeobs_check": chk,
                    "interpretation_rule": (
                        "V > +1.96 favours ZINB, V < -1.96 favours NB, "
                        "|V| <= 1.96 is indistinguishable"
                    ),
                    "VALIDITY_CAVEAT": (
                        "The statistic below is computed correctly from the "
                        "per-observation log-likelihoods, and the loglikeobs sums "
                        "reconcile with .llf to ~1e-9. Its INTERPRETATION is contested: "
                        "Vuong (1989) assumes strictly non-nested models, whereas NB is "
                        "the limit of ZINB as the inflation probability goes to 0, so the "
                        "null distribution is not exactly standard normal here "
                        "(cf. Wilson 2015). Treat it as suggestive only; the verdict in "
                        "this report rests on AIC/BIC, the observed-vs-expected zero "
                        "counts, and out-of-sample CV."
                    ),
                }
                print(
                    f"  Vuong V={V:+.4f} (two-sided p={2 * sps.norm.sf(abs(V)):.4f}); "
                    f"AIC-corrected {rec['vuong']['statistic_aic_corrected']:+.4f}, "
                    f"BIC-corrected {rec['vuong']['statistic_bic_corrected']:+.4f}"
                )
                print(f"  loglikeobs reconciliation vs .llf: {chk}")
        except Exception:
            rec["vuong"] = {"computed": False, "traceback": traceback.format_exc()}
            print("  Vuong computation raised:\n" + traceback.format_exc())

        # ---------------- CV ----------------
        print("  running 5-fold CV (NB and ZINB) ...")
        rec["nb"]["cv"] = cv_family(y, Xv, off, "nb")
        rec["zinb"]["cv"] = cv_family(y, Xv, off, "zinb")
        for fam in ("nb", "zinb"):
            cv = rec[fam]["cv"]
            if cv["mae_across_folds"]["mean"] is None:
                print(f"    CV {fam}: all folds failed")
                continue
            print(
                f"    CV {fam:<5} MAE {cv['mae_across_folds']['mean']:.4f} "
                f"+/- {cv['mae_across_folds']['sd']:.4f}   "
                f"RMSE {cv['rmse_across_folds']['mean']:.4f} "
                f"+/- {cv['rmse_across_folds']['sd']:.4f}   "
                f"pooled MAE {cv['pooled_out_of_fold_mae']:.4f} / RMSE "
                f"{cv['pooled_out_of_fold_rmse']:.4f}   folds failed: {cv['n_folds_failed']}"
            )

        out["modes"][mode.label] = rec

    dump_json(out, RESULTS_DIR / "e7_results.json")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
