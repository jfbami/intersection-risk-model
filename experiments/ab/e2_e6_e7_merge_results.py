"""Merge the E2/E6/E7 result JSONs into one machine-readable file and print
the exact figures quoted in experiments/results/e2_e6_e7_family.md.

Run:
  cd C:/Users/jfbaa/project-cycle-group && \
    PYTHONPATH=C:/Users/jfbaa/project-cycle-group \
    python experiments/ab/e2_e6_e7_merge_results.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2_e6_e7_common import RESULTS_DIR, dump_json, environment  # noqa: E402


def load(name):
    with open(RESULTS_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    e2, e6, e7 = load("e2_results.json"), load("e6_results.json"), load("e7_results.json")

    merged = {
        "generated_by": "experiments/ab/e2_e6_e7_merge_results.py",
        "environment": environment(),
        "commands_run": [
            "cd C:/Users/jfbaa/project-cycle-group && PYTHONPATH=C:/Users/jfbaa/project-cycle-group python experiments/ab/e2_nb_vs_poisson.py",
            "cd C:/Users/jfbaa/project-cycle-group && PYTHONPATH=C:/Users/jfbaa/project-cycle-group python experiments/ab/e6_exposure_offset_vs_covariate.py",
            "cd C:/Users/jfbaa/project-cycle-group && PYTHONPATH=C:/Users/jfbaa/project-cycle-group python experiments/ab/e7_nb_vs_zinb.py",
            "cd C:/Users/jfbaa/project-cycle-group && PYTHONPATH=C:/Users/jfbaa/project-cycle-group python experiments/ab/e2_e6_e7_merge_results.py",
        ],
        "E2": e2,
        "E6": e6,
        "E7": e7,
        "e7_premise_verification": {
            "claim": (
                "pipeline/evaluate_models.py is broken - it imports a name that "
                "does not exist - so its zero-inflation check has never run."
            ),
            "verdict": "CONFIRMED at HEAD, and a second independent breakage found behind it",
            "evidence_head": {
                "bad_import": "DESIGN_PREDICTORS from pipeline.fit_risk_model",
                "occurrences_of_DESIGN_PREDICTORS_in_fit_risk_model_py": 0,
                "fit_risk_model_py_modified": False,
                "error_loading_head_version": (
                    "ImportError: cannot import name 'DESIGN_PREDICTORS' from "
                    "'pipeline.fit_risk_model'"
                ),
            },
            "concurrency_note": (
                "Two other agents were working in this repo concurrently. During this "
                "session one of them modified pipeline/evaluate_models.py (removing the "
                "DESIGN_PREDICTORS import and rewriting _design_matrix to use "
                "mode.predictors). That edit was NOT made by these experiments - nothing "
                "under pipeline/ or data/ was written here. Consequently HEAD and the "
                "working tree fail differently."
            ),
            "evidence_working_tree": {
                "command": "python -m pipeline.evaluate_models",
                "error": (
                    "TypeError: StringDtype.__init__() takes from 1 to 2 positional "
                    "arguments but 3 were given"
                ),
                "location": "pipeline/evaluate_models.py:60 in _load_pkl, pickle.load(f)",
                "cause": (
                    "data/model/nb_v3_*.pkl were pickled under a different pandas version "
                    "than the installed 2.2.3"
                ),
                "also_disables": "pipeline/tests/test_calibration.py (loads the same pkls)",
            },
            "conclusion": (
                "print_zero_prediction_check (evaluate_models.py:164) is unreachable both "
                "at HEAD and after the concurrent import fix; the zero-inflation check has "
                "never produced a number. Fixing only the import would not have made it run."
            ),
        },
    }

    # --- derived: seed-sweep nominal SE vs realised spread (E6) ---
    derived = {"e6_seed_sweep_se_vs_spread": {}}
    sweep = e6["part_b_synthetic_variable_exposure"]["seed_sensitivity_spec_b"]
    for lab, v in sweep.items():
        ses = [s["se"] for s in v["per_seed"] if s.get("converged")]
        cfs = [s["coef"] for s in v["per_seed"] if s.get("converged")]
        derived["e6_seed_sweep_se_vs_spread"][lab] = {
            "mean_nominal_se": float(np.mean(ses)),
            "between_seed_sd_of_coef": float(np.std(cfs, ddof=1)),
            "mean_coef": float(np.mean(cfs)),
            "min_coef": float(np.min(cfs)),
            "max_coef": float(np.max(cfs)),
            "n_seeds_ci_contains_1": v["n_seeds_ci_contains_1"],
            "n_seeds": len(cfs),
        }
    merged["derived"] = derived

    dump_json(merged, RESULTS_DIR / "e2_e6_e7_results.json")

    # ------------------------------------------------------------------
    print("\n================ E2 table ================")
    hdr = (
        f"{'mode':8s} {'mean':>7s} {'var':>8s} {'v/m':>6s} "
        f"{'LL_NB':>10s} {'LL_Pois':>10s} {'AIC_NB':>9s} {'AIC_Pois':>9s} "
        f"{'BIC_NB':>9s} {'BIC_Pois':>9s} {'alpha':>7s} {'LR':>9s} {'p_bnd':>10s}"
    )
    print(hdr)
    for lab, m in e2["modes"].items():
        d, nb, po, t = m["descriptives"], m["nb"], m["poisson"], m["lr_test_alpha_zero"]
        print(
            f"{lab:8s} {d['mean']:7.4f} {d['variance']:8.4f} {d['variance_over_mean']:6.3f} "
            f"{nb['loglik']:10.3f} {po['loglik']:10.3f} {nb['aic_manual']:9.2f} "
            f"{po['aic_manual']:9.2f} {nb['bic_manual']:9.2f} {po['bic_manual']:9.2f} "
            f"{nb['alpha']:7.4f} {t['LR_statistic']:9.3f} {t['boundary_corrected_p']:10.2e}"
        )

    print("\n---- E2 coverage + CV ----")
    for lab, m in e2["modes"].items():
        cn, cp = m["nb"]["in_sample_coverage_90"], m["poisson"]["in_sample_coverage_90"]
        cvn, cvp = m["nb"]["cv"], m["poisson"]["cv"]
        print(
            f"{lab:8s} cov NB {cn['coverage_pct']:5.1f}% (w {cn['mean_width']:6.2f}) | "
            f"Pois {cp['coverage_pct']:5.1f}% (w {cp['mean_width']:5.2f}) | "
            f"width ratio {m['coverage_comparison']['nb_over_poisson_mean_width_ratio']:.3f} | "
            f"OOF cov NB {m['nb']['cv']['out_of_fold_coverage_90']['coverage_pct']:5.1f}% "
            f"Pois {m['poisson']['cv']['out_of_fold_coverage_90']['coverage_pct']:5.1f}%"
        )
        print(
            f"{'':8s} CV MAE NB {cvn['mae_across_folds']['mean']:.4f}"
            f"+/-{cvn['mae_across_folds']['sd']:.4f} "
            f"Pois {cvp['mae_across_folds']['mean']:.4f}+/-{cvp['mae_across_folds']['sd']:.4f} | "
            f"CV RMSE NB {cvn['rmse_across_folds']['mean']:.4f}"
            f"+/-{cvn['rmse_across_folds']['sd']:.4f} "
            f"Pois {cvp['rmse_across_folds']['mean']:.4f}+/-{cvp['rmse_across_folds']['sd']:.4f}"
        )
        print(
            f"{'':8s} pooled OOF MAE NB {cvn['pooled_out_of_fold']['mae']:.4f} "
            f"Pois {cvp['pooled_out_of_fold']['mae']:.4f} | RMSE NB "
            f"{cvn['pooled_out_of_fold']['rmse']:.4f} Pois {cvp['pooled_out_of_fold']['rmse']:.4f}"
            f" | paired MAE delta {m['cv_paired_difference_nb_minus_poisson']['mae_delta']['mean']:+.5f}"
            f" +/- {m['cv_paired_difference_nb_minus_poisson']['mae_delta']['sd']:.5f}"
            f" (t p={m['cv_paired_difference_nb_minus_poisson'].get('mae_paired_t_p', float('nan')):.3f})"
        )

    print("\n================ E6 Part A ================")
    for key, s in e6["part_a_non_identifiability"]["specs"].items():
        rs = s.get("ridge_summary", {})
        print(
            f"{key:32s} ncols={s.get('design_ncols')} rank={s.get('design_rank')} "
            f"cond={s.get('condition_number'):.2e} | SEs: {s.get('std_errors_outcome')} | "
            f"dropped={s.get('term_dropped')} | LL spread on ridge "
            f"{rs.get('loglik_spread', float('nan')):.1e}, coef spread "
            f"{rs.get('exposure_coef_spread', float('nan')):.3f}, identified combo spread "
            f"{rs.get('identified_combination_spread', float('nan')):.1e}"
        )

    print("\n================ E6 Part B (seed 0) ================")
    for lab, m in e6["part_b_synthetic_variable_exposure"]["modes"].items():
        print(f"-- {lab} ({m['target']})")
        for k, s in m["specs"].items():
            if not s.get("converged"):
                print(f"   {k:30s} FAILED")
                continue
            extra = ""
            if "coef_log_years_observed" in s:
                ci = s["ci95_log_years_observed"]
                extra = (
                    f" coef={s['coef_log_years_observed']:+.4f} "
                    f"CI[{ci[0]:+.4f},{ci[1]:+.4f}]"
                )
            print(
                f"   {k:30s} LL={s['loglik']:9.3f} AIC={s['aic_manual']:8.2f} "
                f"CVMAE={s['cv']['mae_across_folds']['mean']:.4f}"
                f"+/-{s['cv']['mae_across_folds']['sd']:.4f}{extra}"
            )
        kq = m.get("key_question", {})
        if kq:
            print(
                f"   CI contains 1.0: {kq['spec_b_ci_contains_1']}  z vs 1 = "
                f"{kq['spec_b_z_vs_1']:+.3f}  LR vs offset = {kq['LR_offset_constraint']:.3f} "
                f"(p={kq['LR_offset_constraint_p_chi2_1df']:.4f})"
            )

    print("\n---- E6 seed sweep ----")
    for lab, v in derived["e6_seed_sweep_se_vs_spread"].items():
        print(
            f"{lab:22s} coef {v['mean_coef']:+.4f} (range {v['min_coef']:+.3f}..{v['max_coef']:+.3f}) "
            f"between-seed SD {v['between_seed_sd_of_coef']:.4f} vs mean nominal SE "
            f"{v['mean_nominal_se']:.4f} | CI covers 1.0 in "
            f"{v['n_seeds_ci_contains_1']}/{v['n_seeds']}"
        )

    print("\n================ E7 ================")
    for lab, m in e7["modes"].items():
        nb, zi = m["nb"], m["zinb"]
        bs = m["zero_count_parametric_bootstrap"]
        print(
            f"{lab:8s} obs zeros {m['observed_zero_sites']}/{m['n_sites']} | NB exp "
            f"{nb['expected_zero_sites']:.2f} (gap {nb['zero_gap_observed_minus_expected']:+.2f}) | "
            f"bootstrap 90% band [{bs['simulated_zero_p05']:.0f},{bs['simulated_zero_p95']:.0f}] "
            f"inside={bs['observed_inside_90pct_band']}"
        )
        print(
            f"{'':8s} NB LL {nb['loglik']:.3f} AIC {nb['aic_manual']:.2f} BIC {nb['bic_manual']:.2f} "
            f"| ZINB LL {zi['loglik']:.3f} AIC {zi['aic_manual']:.2f} BIC {zi['bic_manual']:.2f} "
            f"| dLL {m['comparison']['delta_loglik_zinb_minus_nb']:+.2e} "
            f"dAIC {m['comparison']['delta_aic_zinb_minus_nb']:+.3f} "
            f"dBIC {m['comparison']['delta_bic_zinb_minus_nb']:+.3f}"
        )
        print(
            f"{'':8s} pi_hat {zi['inflation_probability_pi']:.3e} logit int "
            f"{zi['inflation_logit_intercept']:.3f} (SE {zi['inflation_logit_intercept_se']}) "
            f"| ZINB exp zeros {zi['expected_zero_sites']:.2f} "
            f"| converged_on {zi['converged_on']}"
        )
        print(
            f"{'':8s} CV MAE NB {nb['cv']['mae_across_folds']['mean']:.4f}"
            f"+/-{nb['cv']['mae_across_folds']['sd']:.4f} ZINB "
            f"{zi['cv']['mae_across_folds']['mean']:.4f}+/-{zi['cv']['mae_across_folds']['sd']:.4f}"
            f" | CV RMSE NB {nb['cv']['rmse_across_folds']['mean']:.4f} ZINB "
            f"{zi['cv']['rmse_across_folds']['mean']:.4f}"
            f" | pooled MAE NB {nb['cv']['pooled_out_of_fold_mae']:.4f} ZINB "
            f"{zi['cv']['pooled_out_of_fold_mae']:.4f}"
            f" | folds failed NB {nb['cv']['n_folds_failed']} ZINB {zi['cv']['n_folds_failed']}"
        )
        vg = m["vuong"]
        print(f"{'':8s} Vuong: degenerate={vg.get('degenerate')} raw V={vg.get('statistic_raw')}")


if __name__ == "__main__":
    main()
