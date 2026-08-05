"""E6 - exposure as offset=log(years_observed) vs as a free covariate.

WHY THIS EXPERIMENT EXISTS
--------------------------
An audit flagged the offset-vs-covariate question as *untestable* on the real
data, because years_observed == 6 for all 346 rows, so a free exposure
covariate is perfectly collinear with the intercept. This script:

  Part A - DEMONSTRATES that non-identifiability instead of asserting it
           (design-matrix rank vs column count, plus whatever statsmodels
           actually does when asked to fit the degenerate spec).

  Part B - actually TESTS the offset constraint on a dataset where exposure
           genuinely varies, built by sub-sampling years from
           data/intermediate/crashes_by_intersection_year.parquet.

           *** Part B is a SYNTHETIC EXPOSURE DESIGN. *** It tests the
           modelling assumption (is the log-exposure coefficient 1?) under a
           construction where the true coefficient IS 1 by design. It does
           NOT and cannot change what is knowable from the real, uniform
           6-year observation window.

Run:
  cd C:/Users/jfbaa/project-cycle-group && \
    PYTHONPATH=C:/Users/jfbaa/project-cycle-group \
    python experiments/ab/e6_exposure_offset_vs_covariate.py
"""

from __future__ import annotations

import sys
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning

sys.path.insert(0, str(Path(__file__).resolve().parent))

from e2_e6_e7_common import (  # noqa: E402
    ROOT,
    RESULTS_DIR,
    _fit_capture,
    dump_json,
    environment,
    get_modelling_frame,
    kfold_indices,
    mae_rmse,
    mean_sd,
    modes,
)

YEAR_GRID = ROOT / "data" / "intermediate" / "crashes_by_intersection_year.parquet"
SITE_CRASHES = ROOT / "data" / "intermediate" / "crashes_by_intersection.parquet"


# ---------------------------------------------------------------------------
# Shared NB fitter (array API, explicit retry ladder)
# ---------------------------------------------------------------------------


def fit_nb(y, X, offset, warm_start=None):
    """Fit NB2 by running EVERY rung of the ladder and keeping the converged
    rung with the highest log-likelihood.

    Early-returning on the first rung that merely reports converged=True is
    not safe here: on this data the statsmodels default (bfgs, maxiter=35)
    can report success at a point where alpha has collapsed to 0, giving a
    log-likelihood well below the true maximum. Because spec (a) is nested
    inside spec (b), such a local optimum would have produced the impossible
    result LL(b) < LL(a). Selecting on log-likelihood removes that artefact.
    """
    y = np.asarray(y, float)
    Xv = np.asarray(X, float)
    model = sm.NegativeBinomial(y, Xv, offset=offset)

    # a Poisson-based warm start (beta from Poisson, alpha from a
    # method-of-moments style guess)
    poisson_start = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pres = sm.Poisson(y, Xv, offset=offset).fit(disp=False)
        mu = np.exp(Xv @ np.asarray(pres.params, float) + offset)
        a0 = float(np.clip(np.mean(((y - mu) ** 2 - mu) / np.clip(mu ** 2, 1e-9, None)), 1e-3, 5))
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
    if warm_start is not None:
        ladder.append(
            (
                {"start_params": np.asarray(warm_start, float), "method": "bfgs", "maxiter": 1000},
                "bfgs,maxiter=1000,start=nested_spec_a",
            )
        )

    attempts, best = [], None
    for kw, tag in ladder:
        try:
            res, warns = _fit_capture(model, **kw)
            ok = (not warns) and bool(res.mle_retvals.get("converged", True))
            llf = float(res.llf)
            attempts.append(
                {
                    "attempt": tag,
                    "converged": ok,
                    "loglik": llf,
                    "alpha": float(np.asarray(res.params, float)[-1]),
                    "warnings": warns,
                }
            )
            if ok and np.isfinite(llf) and (best is None or llf > best[1]):
                best = (res, llf, tag)
        except Exception as exc:
            attempts.append({"attempt": tag, "converged": False, "error": repr(exc)})

    if best is None:
        return None, attempts, None
    return best[0], attempts, best[2] + " [selected: highest LL among converged rungs]"


def nb_cv(y, X, offset):
    y = np.asarray(y, float)
    Xv = np.asarray(X, float)
    n = len(y)
    folds, backend = kfold_indices(n)
    maes, rmses, oof = [], [], np.full(n, np.nan)
    nfail = 0
    for tr, te in folds:
        res, _, _ = fit_nb(y[tr], Xv[tr], offset[tr])
        if res is None:
            nfail += 1
            continue
        beta = np.asarray(res.params, float)[:-1]
        mu = np.exp(Xv[te] @ beta + offset[te])
        oof[te] = mu
        a, b = mae_rmse(y[te], mu)
        maes.append(a)
        rmses.append(b)
    pooled = mae_rmse(y, oof) if np.isfinite(oof).all() else (None, None)
    return {
        "backend": backend,
        "n_folds_failed": nfail,
        "mae_across_folds": mean_sd(maes),
        "rmse_across_folds": mean_sd(rmses),
        "pooled_out_of_fold_mae": pooled[0],
        "pooled_out_of_fold_rmse": pooled[1],
    }


def summarise(res, tag, n, extra_term=None, X=None):
    k = len(res.params)
    llf = float(res.llf)
    out = {
        "converged_on": tag,
        "n": int(n),
        "k_params_incl_dispersion": k,
        "loglik": llf,
        "aic_manual": float(-2 * llf + 2 * k),
        "bic_manual": float(-2 * llf + np.log(n) * k),
        "aic_statsmodels": float(res.aic),
        "alpha": float(np.asarray(res.params, float)[-1]),
    }
    if extra_term is not None and X is not None:
        j = list(X.columns).index(extra_term)
        params = np.asarray(res.params, float)
        bse = np.asarray(res.bse, float)
        out["coef_" + extra_term] = float(params[j])
        out["se_" + extra_term] = float(bse[j])
        out["ci95_" + extra_term] = [
            float(params[j] - 1.959963985 * bse[j]),
            float(params[j] + 1.959963985 * bse[j]),
        ]
    return out


# ===========================================================================
# PART A - non-identifiability on the real data
# ===========================================================================


def part_a(df, out):
    print("\n" + "=" * 78)
    print("  PART A - is exposure-as-covariate identifiable on the real 346 rows?")
    print("=" * 78)

    uniq = sorted(df["years_observed"].unique().tolist())
    print(f"df['years_observed'].unique() -> {uniq}")
    a = {
        "years_observed_unique": uniq,
        "n_rows": int(len(df)),
        "offset_unique": sorted(df["offset"].unique().tolist()),
        "specs": {},
    }

    for mode in modes():
        base = mode.predictors
        for term_name, term in [
            ("years_observed", "years_observed"),
            ("log_years_observed", "np.log(years_observed)"),
        ]:
            formula = f"{mode.target} ~ {term} + {base}"
            key = f"{mode.label}::{term_name}"
            entry = {"formula": formula, "offset": "NONE (deliberately omitted)"}

            # --- design matrix rank ---
            exp_col = None
            X = None
            try:
                yv, X = patsy.dmatrices(formula, data=df, return_type="dataframe")
                Xv = np.asarray(X, float)
                rank = int(np.linalg.matrix_rank(Xv))
                entry["design_ncols"] = int(Xv.shape[1])
                entry["design_rank"] = rank
                entry["rank_deficient"] = rank < Xv.shape[1]
                entry["design_columns"] = list(X.columns)
                exp_col = [c for c in X.columns if "years_observed" in c][0]
                entry["exposure_column"] = exp_col
                # the exact linear dependency with the intercept
                icept = np.asarray(X["Intercept"], float)
                col = np.asarray(X[exp_col], float)
                entry["exposure_column_unique_values"] = np.unique(col).tolist()
                entry["exposure_col_over_intercept_unique"] = np.unique(
                    np.round(col / icept, 12)
                ).tolist()
                entry["condition_number"] = float(np.linalg.cond(Xv))
                entry["smallest_singular_value"] = float(
                    np.linalg.svd(Xv, compute_uv=False)[-1]
                )
            except Exception:
                entry["design_error"] = traceback.format_exc()

            # --- what does statsmodels actually do? (fit isolated from diagnostics) ---
            res = None
            try:
                model = smf.negativebinomial(formula, data=df, offset=None)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    res = model.fit(disp=False, method="bfgs", maxiter=200)
                entry["fit_outcome"] = (
                    "returned a result object - NO exception, NO dropped term"
                )
                entry["fit_warnings"] = [str(w.message) for w in caught]
                entry["n_params_returned"] = int(len(res.params))
                entry["term_dropped"] = exp_col not in res.params.index
                entry["params"] = {str(k): float(v) for k, v in res.params.items()}
                entry["loglik"] = float(res.llf)
                entry["mle_converged"] = bool(res.mle_retvals.get("converged", True))
            except Exception:
                entry["fit_outcome"] = "the fit call itself raised"
                entry["fit_traceback"] = traceback.format_exc()

            if res is not None:
                try:
                    bse = np.asarray(res.bse, float)
                    entry["bse"] = {str(k): float(v) for k, v in res.bse.items()}
                    entry["n_nan_or_inf_std_errors"] = int((~np.isfinite(bse)).sum())
                    entry["n_params"] = int(len(bse))
                    entry["std_errors_outcome"] = (
                        "all std errors NaN/inf"
                        if not np.isfinite(bse).any()
                        else (
                            "SOME std errors NaN/inf"
                            if not np.isfinite(bse).all()
                            else "all std errors finite (statsmodels reports usable-looking "
                            "SEs for a rank-deficient design)"
                        )
                    )
                except Exception:
                    entry["std_errors_outcome"] = "res.bse raised"
                    entry["bse_traceback"] = traceback.format_exc()
                try:
                    cp = np.asarray(res.cov_params(), float)
                    entry["cov_params_outcome"] = (
                        "all NaN/inf" if not np.isfinite(cp).any() else "finite"
                    )
                except Exception:
                    entry["cov_params_outcome"] = "res.cov_params() raised"
                    entry["cov_params_traceback"] = traceback.format_exc()

                # --- the decisive demonstration: perturb the start values ---
                # Under non-identifiability the likelihood has a flat ridge:
                # the optimiser lands anywhere along it, so (Intercept, exposure)
                # move while the log-likelihood and the SUM that the model can
                # actually see stay put.
                ridge = []
                for delta in (-2.0, -0.5, 0.0, 0.5, 2.0):
                    try:
                        sp = np.zeros(len(res.params))
                        sp[:] = np.asarray(res.params, float)
                        j = list(res.params.index).index(exp_col)
                        ji = list(res.params.index).index("Intercept")
                        xval = float(np.unique(np.asarray(X[exp_col], float))[0])
                        sp[j] += delta
                        sp[ji] -= delta * xval  # move ALONG the ridge
                        with warnings.catch_warnings(record=True):
                            warnings.simplefilter("always")
                            r2 = smf.negativebinomial(
                                formula, data=df, offset=None
                            ).fit(disp=False, method="bfgs", maxiter=200, start_params=sp)
                        ridge.append(
                            {
                                "start_shift": delta,
                                "intercept": float(r2.params["Intercept"]),
                                "exposure_coef": float(r2.params[exp_col]),
                                "intercept_plus_x_times_coef": float(
                                    r2.params["Intercept"] + xval * r2.params[exp_col]
                                ),
                                "loglik": float(r2.llf),
                            }
                        )
                    except Exception as exc:
                        ridge.append({"start_shift": delta, "error": repr(exc)})
                entry["likelihood_ridge_probe"] = ridge
                fin = [r for r in ridge if "loglik" in r]
                if len(fin) > 1:
                    entry["ridge_summary"] = {
                        "loglik_spread": float(
                            max(r["loglik"] for r in fin) - min(r["loglik"] for r in fin)
                        ),
                        "exposure_coef_spread": float(
                            max(r["exposure_coef"] for r in fin)
                            - min(r["exposure_coef"] for r in fin)
                        ),
                        "intercept_spread": float(
                            max(r["intercept"] for r in fin) - min(r["intercept"] for r in fin)
                        ),
                        "identified_combination_spread": float(
                            max(r["intercept_plus_x_times_coef"] for r in fin)
                            - min(r["intercept_plus_x_times_coef"] for r in fin)
                        ),
                    }

            a["specs"][key] = entry
            cond = entry.get("condition_number")
            print(
                f"\n[{key}]"
                f"\n  formula: {formula}   (offset deliberately omitted)"
                f"\n  design: {entry.get('design_ncols')} cols, rank {entry.get('design_rank')}"
                f"  -> rank deficient: {entry.get('rank_deficient')}"
                f"\n  exposure column '{entry.get('exposure_column')}' unique values: "
                f"{entry.get('exposure_column_unique_values')}"
                f"   (= constant multiple of the Intercept column)"
                f"\n  smallest singular value: {entry.get('smallest_singular_value'):.3e}"
                f"   condition number: {cond:.3e}"
                f"\n  fit outcome: {entry.get('fit_outcome')}"
                f"\n  term dropped: {entry.get('term_dropped')}"
                f"   std errors: {entry.get('std_errors_outcome')}"
                f"   ({entry.get('n_nan_or_inf_std_errors')}/{entry.get('n_params')} non-finite)"
                f"\n  cov_params(): {entry.get('cov_params_outcome')}"
                f"\n  mle 'converged' flag: {entry.get('mle_converged')}"
                f"   LL: {entry.get('loglik')}"
            )
            if "ridge_summary" in entry:
                rs = entry["ridge_summary"]
                print(
                    f"  RIDGE PROBE (restart along the flat direction):"
                    f"\n    log-likelihood spread ............ {rs['loglik_spread']:.3e}"
                    f"\n    exposure coefficient spread ..... {rs['exposure_coef_spread']:.4f}"
                    f"\n    intercept spread ................ {rs['intercept_spread']:.4f}"
                    f"\n    (Intercept + x*coef) spread ..... "
                    f"{rs['identified_combination_spread']:.3e}   <- the only identified quantity"
                )
            for tb_key in ("fit_traceback", "bse_traceback", "cov_params_traceback"):
                if tb_key in entry:
                    print(f"  --- {tb_key} ---")
                    print("  " + entry[tb_key].replace("\n", "\n  "))

    # a genuinely-identifiable control: offset spec on the same rows
    ctrl = {}
    for mode in modes():
        f = f"{mode.target} ~ {mode.predictors}"
        yv, X = patsy.dmatrices(f, data=df, return_type="dataframe")
        Xv = np.asarray(X, float)
        ctrl[mode.label] = {
            "formula": f,
            "offset": "log(years_observed)",
            "design_ncols": int(Xv.shape[1]),
            "design_rank": int(np.linalg.matrix_rank(Xv)),
            "condition_number": float(np.linalg.cond(Xv)),
        }
    a["control_production_spec"] = ctrl
    print("\ncontrol (production offset spec) design rank vs ncols:")
    for k, v in ctrl.items():
        print(f"  {k:8s} ncols={v['design_ncols']} rank={v['design_rank']}")

    out["part_a_non_identifiability"] = a
    return a


# ===========================================================================
# PART B - synthetic variable-exposure dataset
# ===========================================================================

CONSTRUCTION_DOC = """
SYNTHETIC VARIABLE-EXPOSURE CONSTRUCTION (rng = numpy.random.default_rng(SEED))

Input: data/intermediate/crashes_by_intersection_year.parquet
       (3906 rows = 651 intersections x 6 years 2018-2023; columns
        intersection_id, year, crash_count). NOTE: this grid carries only the
        TOTAL crash count per intersection-year - it has NO mode breakdown.
       data/intermediate/crashes_by_intersection.parquet supplies the 6-year
       per-mode totals (bike_total, ped_total, vehicle_only_total).

Verified facts used by the construction:
  * every site has exactly 6 year-rows;
  * per-site sum(crash_count) == total_crashes exactly (651/651 sites);
  * each mode total <= total_crashes at every site;
  * every site with total_crashes == 0 has all three mode totals == 0.

Step 1 (exposure draw). For each of the 346 modelled intersections, in a
  fixed intersection_id-sorted order:
      k        = rng.integers(2, 7)                 -> 2..6 inclusive
      S (years)= rng.choice(the 6 years, size=k, replace=False)
  Set years_observed := k for that site.

Step 2 (crash-year allocation). The year grid has no mode split, so each
  mode's 6-year total is allocated to years by drawing iid year labels from
  that site's OWN empirical year distribution of total crashes,
      p_y = crash_count_y / total_crashes    (only sites with total > 0 ever
                                              have a non-zero mode total)
  i.e. for mode m with 6-year count c_m,  years_m = rng.choice(years, size=c_m,
  replace=True, p=p_y).

Step 3 (thinning). The synthetic observed count is the number of allocated
  crash-years of mode m that fall inside S:
      y_m := count of allocated mode-m years that are members of S.

Because S is drawn independently of the crash years, E[y_m | site] =
  mu_m * k / 6, so the TRUE log-exposure coefficient in this design is
  exactly 1.0. That is precisely what makes it a usable test bed: spec (b)
  should recover 1.0 if the offset constraint is right.

SUPPLEMENTARY (no allocation needed): the same year-subset S is also applied
  to the site's REAL per-year total crash counts, giving a total-crash target
  in which the crash-to-year assignment is REAL data and only the observation
  window is synthetic.

Features (log_aadt, log_bike_centrality, legs_cat, ...) are taken unchanged
from the 346-row production frame and joined on intersection_id.
"""


def build_variable_exposure(df, seed: int):
    grid = pd.read_parquet(YEAR_GRID)
    site = pd.read_parquet(SITE_CRASHES).set_index("intersection_id")

    years = np.array(sorted(grid["year"].unique()))
    wide = (
        grid.pivot(index="intersection_id", columns="year", values="crash_count")
        .reindex(columns=years)
        .fillna(0)
        .astype(int)
    )

    ids = sorted(df["intersection_id"].tolist())
    rng = np.random.default_rng(seed)

    mode_cols = {m.label: m.target for m in modes()}
    rows = []
    for sid in ids:
        k = int(rng.integers(2, 7))
        chosen = rng.choice(years, size=k, replace=False)
        chosen_set = set(chosen.tolist())

        counts = wide.loc[sid].to_numpy()
        tot = int(counts.sum())
        p = counts / tot if tot > 0 else None

        rec = {
            "intersection_id": sid,
            "years_observed": k,
            "chosen_years": sorted(chosen_set),
            # supplementary: real crash-years, synthetic window only
            "total_real_window": int(
                sum(int(counts[i]) for i, yy in enumerate(years) if yy in chosen_set)
            ),
            "total_full6": tot,
        }
        for label, col in mode_cols.items():
            c = int(site.loc[sid, col])
            if c == 0:
                rec[col] = 0
            else:
                alloc = rng.choice(years, size=c, replace=True, p=p)
                rec[col] = int(np.isin(alloc, list(chosen_set)).sum())
            rec[col + "_full6"] = c
        rows.append(rec)

    ve = pd.DataFrame(rows)
    feat = df.drop(columns=[c for c in ve.columns if c != "intersection_id" and c in df.columns])
    out = feat.merge(ve, on="intersection_id", how="inner")
    out["offset"] = np.log(out["years_observed"])
    out["log_years_observed"] = np.log(out["years_observed"])
    return out.reset_index(drop=True)


def part_b(df, out, primary_seed: int = 0):
    print("\n" + "=" * 78)
    print("  PART B - offset vs free covariate on SYNTHETIC variable exposure")
    print("=" * 78)
    print(CONSTRUCTION_DOC)

    ve = build_variable_exposure(df, primary_seed)
    print(f"variable-exposure frame: {len(ve)} rows")
    print("years_observed distribution:")
    print(ve["years_observed"].value_counts().sort_index().to_string())

    b = {
        "seed": primary_seed,
        "construction": CONSTRUCTION_DOC.strip(),
        "is_synthetic_exposure": True,
        "n_rows": int(len(ve)),
        "years_observed_distribution": {
            int(k): int(v) for k, v in ve["years_observed"].value_counts().sort_index().items()
        },
        "target_totals": {},
        "modes": {},
    }
    for m in modes():
        b["target_totals"][m.target] = {
            "synthetic_variable_exposure": int(ve[m.target].sum()),
            "real_full_6_year": int(ve[m.target + "_full6"].sum()),
            "n_zero_sites": int((ve[m.target] == 0).sum()),
        }
    b["target_totals"]["total_crashes(real years, synthetic window)"] = {
        "synthetic_variable_exposure": int(ve["total_real_window"].sum()),
        "real_full_6_year": int(ve["total_full6"].sum()),
        "n_zero_sites": int((ve["total_real_window"] == 0).sum()),
    }
    print("\ntarget totals after thinning:")
    for k, v in b["target_totals"].items():
        print(f"  {k:52s} {v['synthetic_variable_exposure']:5d}  (full-6yr {v['real_full_6_year']})")

    targets = [(m.label, m.target, m.predictors) for m in modes()]
    targets.append(
        (
            "total_supplementary",
            "total_real_window",
            modes()[2].predictors,  # same predictors as the vehicle model (uses log_aadt)
        )
    )

    n = len(ve)
    zeros = np.zeros(n)
    for label, target, preds in targets:
        print("\n" + "-" * 78)
        print(f"  {label}   target={target}")
        print("-" * 78)
        rec = {"target": target, "predictors": preds, "specs": {}}

        # --- (a) offset, coefficient constrained to exactly 1 ---
        f_a = f"{target} ~ {preds}"
        y_a, X_a = patsy.dmatrices(f_a, data=ve, return_type="dataframe")
        off = ve["offset"].values.astype(float)
        res_a, att_a, tag_a = fit_nb(np.asarray(y_a).ravel(), X_a, off)

        # --- (b) free covariate, NO offset ---
        f_b = f"{target} ~ log_years_observed + {preds}"
        y_b, X_b = patsy.dmatrices(f_b, data=ve, return_type="dataframe")

        # Warm starts built from the NESTED spec (a) solution. Spec (a) is
        # exactly spec (b) with the log-exposure coefficient pinned to 1
        # (and spec (c) with it pinned to 0), so these points are guaranteed
        # to be feasible and to have LL == LL(a); any converged optimum the
        # ladder keeps must therefore be at least as good.
        def warm_from_a(pin_value):
            if res_a is None:
                return None
            pa = np.asarray(res_a.params, float)
            cols_a = list(X_a.columns)
            vec = np.zeros(len(X_b.columns) + 1)
            for i, c in enumerate(X_b.columns):
                if c == "log_years_observed":
                    vec[i] = pin_value
                elif c in cols_a:
                    vec[i] = pa[cols_a.index(c)]
            vec[-1] = pa[-1]  # alpha
            return vec

        res_b, att_b, tag_b = fit_nb(
            np.asarray(y_b).ravel(), X_b, zeros, warm_start=warm_from_a(1.0)
        )

        # --- (c) offset PLUS free term = estimated deviation from 1 ---
        res_c, att_c, tag_c = fit_nb(
            np.asarray(y_b).ravel(), X_b, off, warm_start=warm_from_a(0.0)
        )

        yv = np.asarray(y_a).ravel()
        for key, (res, att, tag, X, offs, form, extra) in {
            "a_offset_constrained": (res_a, att_a, tag_a, X_a, off, f_a, None),
            "b_free_covariate_no_offset": (
                res_b, att_b, tag_b, X_b, zeros, f_b, "log_years_observed",
            ),
            "c_offset_plus_free_deviation": (
                res_c, att_c, tag_c, X_b, off, f_b, "log_years_observed",
            ),
        }.items():
            if res is None:
                rec["specs"][key] = {
                    "formula": form,
                    "converged": False,
                    "fit_attempts": att,
                }
                print(f"  [{key}] FIT FAILED on all rungs: {att}")
                continue
            s = summarise(res, tag, n, extra_term=extra, X=X)
            s["formula"] = form
            s["offset_used"] = "log(years_observed)" if offs is off else "none"
            s["converged"] = True
            s["fit_attempts"] = att
            s["cv"] = nb_cv(yv, X, offs)
            rec["specs"][key] = s
            msg = (
                f"  [{key}] LL={s['loglik']:.3f} AIC={s['aic_manual']:.2f} "
                f"alpha={s['alpha']:.4f} conv_on={tag} "
                f"CV MAE={s['cv']['mae_across_folds']['mean']:.4f}"
                f"+/-{s['cv']['mae_across_folds']['sd']:.4f}"
            )
            if extra:
                ci = s["ci95_log_years_observed"]
                msg += (
                    f"\n        coef(log exposure)={s['coef_log_years_observed']:+.4f} "
                    f"SE={s['se_log_years_observed']:.4f} "
                    f"95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]"
                )
            print(msg)

        sb = rec["specs"].get("b_free_covariate_no_offset", {})
        sc = rec["specs"].get("c_offset_plus_free_deviation", {})
        if sb.get("converged") and sc.get("converged"):
            ci_b = sb["ci95_log_years_observed"]
            ci_c = sc["ci95_log_years_observed"]
            rec["key_question"] = {
                "spec_b_coef": sb["coef_log_years_observed"],
                "spec_b_ci95": ci_b,
                "spec_b_ci_contains_1": bool(ci_b[0] <= 1.0 <= ci_b[1]),
                "spec_b_z_vs_1": float(
                    (sb["coef_log_years_observed"] - 1.0) / sb["se_log_years_observed"]
                ),
                "spec_c_deviation_coef": sc["coef_log_years_observed"],
                "spec_c_ci95": ci_c,
                "spec_c_ci_contains_0": bool(ci_c[0] <= 0.0 <= ci_c[1]),
                "reparam_check_b_minus_c_should_be_1": float(
                    sb["coef_log_years_observed"] - sc["coef_log_years_observed"]
                ),
                "reparam_check_loglik_diff_should_be_0": float(
                    sb["loglik"] - sc["loglik"]
                ),
            }
            sa = rec["specs"].get("a_offset_constrained", {})
            if sa.get("converged"):
                dll = sb["loglik"] - sa["loglik"]
                rec["key_question"]["nested_check_LL_b_minus_LL_a"] = float(dll)
                rec["key_question"]["nested_check_ok_LL_b_ge_LL_a"] = bool(dll >= -1e-6)
                lr = 2.0 * dll
                rec["key_question"]["LR_offset_constraint"] = float(lr)
                rec["key_question"]["LR_offset_constraint_p_chi2_1df"] = float(
                    __import__("scipy.stats", fromlist=["chi2"]).chi2.sf(max(lr, 0.0), 1)
                )
            kq = rec["key_question"]
            print(
                f"  KEY: 95% CI for log-exposure coef contains 1.0? "
                f"{kq['spec_b_ci_contains_1']}   (z vs 1 = {kq['spec_b_z_vs_1']:+.3f})"
            )
            print(
                f"  reparam check: coef_b - coef_c = "
                f"{kq['reparam_check_b_minus_c_should_be_1']:.6f} (should be 1.0); "
                f"LL_b - LL_c = {kq['reparam_check_loglik_diff_should_be_0']:.2e} (should be 0)"
            )
            if "nested_check_LL_b_minus_LL_a" in kq:
                print(
                    f"  nested check: LL_b - LL_a = {kq['nested_check_LL_b_minus_LL_a']:+.4f} "
                    f"(must be >= 0): {kq['nested_check_ok_LL_b_ge_LL_a']}   "
                    f"LR vs offset constraint = {kq['LR_offset_constraint']:.3f}, "
                    f"chi2(1) p = {kq['LR_offset_constraint_p_chi2_1df']:.4f}"
                )
        b["modes"][label] = rec

    # ---------------- multi-seed sensitivity ----------------
    print("\n" + "-" * 78)
    print("  seed sensitivity for spec (b): does the CI cover 1.0 across seeds 0-9?")
    print("-" * 78)
    sweep = {}
    for label, target, preds in targets:
        coefs, contains, zs = [], [], []
        per_seed = []
        for seed in range(10):
            ve_s = build_variable_exposure(df, seed)
            f_b = f"{target} ~ log_years_observed + {preds}"
            y_s, X_s = patsy.dmatrices(f_b, data=ve_s, return_type="dataframe")
            r, _, _ = fit_nb(np.asarray(y_s).ravel(), X_s, np.zeros(len(ve_s)))
            if r is None:
                per_seed.append({"seed": seed, "converged": False})
                continue
            j = list(X_s.columns).index("log_years_observed")
            c = float(np.asarray(r.params, float)[j])
            se = float(np.asarray(r.bse, float)[j])
            lo, hi = c - 1.959963985 * se, c + 1.959963985 * se
            coefs.append(c)
            zs.append((c - 1.0) / se)
            contains.append(bool(lo <= 1.0 <= hi))
            per_seed.append(
                {
                    "seed": seed,
                    "converged": True,
                    "coef": c,
                    "se": se,
                    "ci95": [lo, hi],
                    "ci_contains_1": bool(lo <= 1.0 <= hi),
                }
            )
        sweep[label] = {
            "per_seed": per_seed,
            "coef_summary": mean_sd(coefs),
            "n_seeds_ci_contains_1": int(sum(contains)),
            "n_seeds_converged": len(coefs),
            "mean_z_vs_1": mean_sd(zs)["mean"],
        }
        print(
            f"  {label:22s} coef mean {mean_sd(coefs)['mean']:+.4f} "
            f"+/- {mean_sd(coefs)['sd']:.4f}   "
            f"CI covers 1.0 in {sum(contains)}/{len(coefs)} seeds   "
            f"mean z vs 1 = {mean_sd(zs)['mean']:+.3f}"
        )
    b["seed_sensitivity_spec_b"] = sweep

    out["part_b_synthetic_variable_exposure"] = b
    return b


def main() -> None:
    df, _ = get_modelling_frame()
    out = {
        "experiment": "E6 - exposure offset vs free covariate",
        "environment": environment(),
        "n_rows": int(len(df)),
    }
    part_a(df, out)
    part_b(df, out)
    dump_json(out, RESULTS_DIR / "e6_results.json")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
