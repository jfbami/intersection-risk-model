"""E5 - Leg-count encoding: top-coded categorical vs continuous vs full categorical.

Also verifies the uncited claim in pipeline/feature_encoding.py:8-16 that a
continuous per-leg slope "reads a 6-leg intersection as roughly +280% over a
4-leg one, with a credible interval spanning +80% to +700%", and that
"2-to-4-leg sites are 97% of the data" with "no six-leg site actually
supporting it".

Run:
  cd C:/Users/jfbaa/project-cycle-group && \
  PYTHONPATH=C:/Users/jfbaa/project-cycle-group python experiments/ab/e5_leg_encoding.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from scipy import stats as sps

warnings.filterwarnings("ignore")

from experiments.ab.e1_e3_e5_common import (  # noqa: E402
    MODES, SHARED_PREDICTORS, evaluate, get_df, print_row, strip_results,
)
from pipeline.feature_encoding import LEG_CATEGORY_TERM  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

BASE_NO_LEGS = SHARED_PREDICTORS.replace(f"{LEG_CATEGORY_TERM} + ", "").strip()
LEG_SPECS = {
    "a_topcoded_cat": LEG_CATEGORY_TERM,
    "b_continuous":   "num_legs",
    "c_full_cat":     "C(num_legs, Treatment(reference=4))",
}
DOCSTRING_CLAIM = {"point_pct": 280.0, "ci_lo_pct": 80.0, "ci_hi_pct": 700.0,
                   "share_2_to_4_legs_pct": 97.0, "no_six_leg_site": True}


def six_vs_four(beta, se):
    """exp(2*beta) as a % change, with Wald 90% and 95% intervals."""
    out = {"beta_num_legs": float(beta), "se": float(se),
           "point_multiplier": float(np.exp(2 * beta)),
           "point_pct_change": float(100 * (np.exp(2 * beta) - 1))}
    for lvl, z in (("90", 1.6448536), ("95", 1.9599640)):
        lo, hi = np.exp(2 * (beta - z * se)), np.exp(2 * (beta + z * se))
        out[f"ci{lvl}_multiplier"] = [float(lo), float(hi)]
        out[f"ci{lvl}_pct_change"] = [float(100 * (lo - 1)), float(100 * (hi - 1))]
    return out


def main():
    df = get_df()
    vc = df["num_legs"].value_counts().sort_index()
    n = len(df)
    n_2_4 = int(vc.reindex([2, 3, 4]).fillna(0).sum())

    results = {
        "experiment": "E5",
        "question": "Leg-count encoding: top-coded categorical vs continuous vs full categorical",
        "reason": ("pipeline/feature_encoding.py:8-16 justifies top-coding with a specific "
                   "uncited claim: a continuous per-leg slope 'reads a 6-leg intersection as "
                   "roughly +280% over a 4-leg one, with a credible interval spanning +80% to "
                   "+700%' and 'no six-leg site actually supporting it'. No saved run backs this."),
        "n_rows": n,
        "base_predictors_without_leg_term": BASE_NO_LEGS,
        "leg_specs": LEG_SPECS,
        "num_legs_distribution": {
            "counts": {int(k): int(v) for k, v in vc.items()},
            "pct": {int(k): float(100 * v / n) for k, v in vc.items()},
            "n_2_to_4_legs": n_2_4,
            "pct_2_to_4_legs": float(100 * n_2_4 / n),
            "n_six_leg_sites": int(vc.get(6, 0)),
            "n_five_leg_sites": int(vc.get(5, 0)),
        },
        "docstring_claim": DOCSTRING_CLAIM,
        "modes": {},
    }
    print("E5: num_legs distribution over the 346-row fit set")
    for k, v in vc.items():
        print(f"    {k} legs: {v:4d}  ({100*v/n:5.2f}%)")
    print(f"    2-4 legs: {n_2_4}/{n} = {100*n_2_4/n:.2f}%  "
          f"(docstring says 97%)")
    print(f"    six-leg sites present: {int(vc.get(6, 0))}  "
          f"(docstring says 'no six-leg site actually supporting it')")

    # crash counts at 5/6-leg sites, for the 'insufficient events' question
    ev = {}
    for mode in MODES:
        ev[mode.label] = {str(int(k)): int(df.loc[df["num_legs"] == k, mode.target].sum())
                          for k in vc.index}
    results["events_by_num_legs"] = ev
    print("\n  Events by num_legs:")
    for lab, d in ev.items():
        print(f"    {lab:8s} " + "  ".join(f"{k}legs={v}" for k, v in d.items()))

    for mode in MODES:
        vol = mode.predictors.replace(SHARED_PREDICTORS, "").strip()
        vol = vol.lstrip("+ ").strip()
        print(f"\n=== {mode.label} ({mode.target}, {int(df[mode.target].sum())} events, "
              f"volume term: {vol}) ===")
        mode_rec = {"volume_term": vol, "specs": {}}
        for spec, legterm in LEG_SPECS.items():
            pred = f"{BASE_NO_LEGS} + {legterm} + {vol}"
            rec = evaluate(df, mode.target, pred, spec)
            rec.pop("_beta", None)
            rec.pop("_design_info", None)
            ins = rec["insample"]
            if spec == "b_continuous" and ins.get("ok"):
                rec["six_vs_four_leg"] = six_vs_four(ins["params"]["num_legs"],
                                                     ins["bse"]["num_legs"])
                rec["num_legs_term"] = {"beta": ins["params"]["num_legs"],
                                        "se": ins["bse"]["num_legs"],
                                        "p": ins["pvalues"]["num_legs"],
                                        "exp_beta_per_leg": float(np.exp(ins["params"]["num_legs"]))}
            if spec == "c_full_cat" and ins.get("ok"):
                rec["full_cat_terms"] = {
                    k: {"beta": ins["params"][k], "se": ins["bse"][k], "p": ins["pvalues"][k],
                        "exp_beta": float(np.exp(ins["params"][k]))}
                    for k in ins["params"] if k.startswith("C(num_legs")}
            mode_rec["specs"][spec] = rec
            print_row(spec, rec)
            cv = rec["cv_stratified"]
            if cv.get("degenerate_columns"):
                print(f"                 !! folds with a level absent from training: "
                      f"{cv['degenerate_columns']}")
            if not cv.get("all_folds_converged"):
                print(f"                 !! only {cv.get('n_folds_converged')}/5 folds "
                      f"converged (seed 0)")
            rc = rec["repeated_cv"]
            if rc.get("n_seeds_with_nonconverged_fold"):
                print(f"                 !! {rc['n_seeds_with_nonconverged_fold']}/10 seeds "
                      f"had >=1 non-converged fold")
            if spec == "b_continuous" and "num_legs_term" in rec:
                t = rec["num_legs_term"]
                s = rec["six_vs_four_leg"]
                print(f"                 num_legs: beta={t['beta']:.6g} se={t['se']:.6g} "
                      f"p={t['p']:.4g}  exp(beta)={t['exp_beta_per_leg']:.4f} per leg")
                print(f"                 6-leg vs 4-leg = exp(2*beta) = "
                      f"{s['point_multiplier']:.4f}x = {s['point_pct_change']:+.1f}%")
                print(f"                   90% CI: {s['ci90_pct_change'][0]:+.1f}% to "
                      f"{s['ci90_pct_change'][1]:+.1f}%   "
                      f"({s['ci90_multiplier'][0]:.3f}x-{s['ci90_multiplier'][1]:.3f}x)")
                print(f"                   95% CI: {s['ci95_pct_change'][0]:+.1f}% to "
                      f"{s['ci95_pct_change'][1]:+.1f}%   "
                      f"({s['ci95_multiplier'][0]:.3f}x-{s['ci95_multiplier'][1]:.3f}x)")
            if spec == "c_full_cat" and "full_cat_terms" in rec:
                for k, v in rec["full_cat_terms"].items():
                    lvl = k.split("T.")[-1].rstrip("]")
                    print(f"                 {lvl}-leg vs 4-leg: beta={v['beta']:.4g} "
                          f"se={v['se']:.4g} p={v['p']:.4g} exp(beta)={v['exp_beta']:.4f}")
        # OOS ranking within this mode
        rk = sorted(((s, mode_rec["specs"][s]["repeated_cv"]["oof_mae_mean"],
                      mode_rec["specs"][s]["repeated_cv"]["oof_mae_sd"])
                     for s in LEG_SPECS
                     if "oof_mae_mean" in mode_rec["specs"][s]["repeated_cv"]),
                    key=lambda t: t[1])
        mode_rec["oos_ranking"] = [{"spec": s, "oof_mae_mean": m, "oof_mae_sd": sd}
                                   for s, m, sd in rk]
        print("  OOS ranking (repeated 10-seed pooled OOF MAE): " +
              "  ".join(f"{s}={m:.4f}+-{sd:.4f}" for s, m, sd in rk))
        results["modes"][mode.label] = mode_rec

    # ---- verdict on the docstring claim ----
    verdicts = {}
    for lab, mr in results["modes"].items():
        s = mr["specs"]["b_continuous"].get("six_vs_four_leg")
        if not s:
            verdicts[lab] = {"verdict": "FIT FAILED"}
            continue
        pt = s["point_pct_change"]
        claim = DOCSTRING_CLAIM
        # is the claimed point inside our 95% interval, and ours inside theirs?
        pt_in_our_ci95 = s["ci95_pct_change"][0] <= claim["point_pct"] <= s["ci95_pct_change"][1]
        ratio = (1 + pt / 100) / (1 + claim["point_pct"] / 100)
        if abs(pt - claim["point_pct"]) <= 0.25 * claim["point_pct"]:
            v = "APPROXIMATELY RIGHT"
        elif pt_in_our_ci95:
            v = "WRONG as a point estimate, but not excluded by our 95% CI"
        else:
            v = "WRONG"
        verdicts[lab] = {
            "measured_point_pct": pt,
            "measured_ci90_pct": s["ci90_pct_change"],
            "measured_ci95_pct": s["ci95_pct_change"],
            "claimed_point_pct": claim["point_pct"],
            "claimed_ci_pct": [claim["ci_lo_pct"], claim["ci_hi_pct"]],
            "claimed_point_inside_measured_ci95": bool(pt_in_our_ci95),
            "measured_over_claimed_multiplier_ratio": float(ratio),
            "verdict": v,
        }
    # ---- fold sparsity: can spec (c) even see a 6-leg site in every training split? ----
    from sklearn.model_selection import StratifiedKFold  # noqa: E402
    from experiments.ab.e1_e3_e5_common import strata_for  # noqa: E402
    sparsity = {}
    for mode in MODES:
        y = df[mode.target].to_numpy()
        miss = {lvl: 0 for lvl in sorted(vc.index)}
        for seed in range(10):
            for tr, _te in StratifiedKFold(n_splits=5, shuffle=True,
                                           random_state=seed).split(np.zeros(len(df)),
                                                                    strata_for(y)):
                present = set(df["num_legs"].to_numpy()[tr])
                for lvl in miss:
                    if lvl not in present:
                        miss[lvl] += 1
        sparsity[mode.label] = {"n_train_splits_checked": 50,
                                "splits_missing_level": {int(k): int(v) for k, v in miss.items()}}
    results["fold_level_sparsity_spec_c"] = sparsity
    print("\n  Spec (c) fold sparsity: training splits (of 50 = 10 seeds x 5 folds) "
          "missing each num_legs level")
    for lab, d in sparsity.items():
        print(f"    {lab:8s} " + "  ".join(f"{k}legs={v}"
                                           for k, v in d["splits_missing_level"].items()))

    results["docstring_verdict_by_mode"] = verdicts
    results["docstring_share_verdict"] = {
        "claimed_pct": 97.0, "measured_pct": results["num_legs_distribution"]["pct_2_to_4_legs"],
        "verdict": ("APPROXIMATELY RIGHT"
                    if abs(results["num_legs_distribution"]["pct_2_to_4_legs"] - 97.0) <= 1.5
                    else "WRONG"),
    }
    results["docstring_six_leg_verdict"] = {
        "claim": "no six-leg site actually supporting it",
        "measured_n_six_leg_sites": results["num_legs_distribution"]["n_six_leg_sites"],
        "verdict": ("WRONG - six-leg sites exist in the fit set"
                    if results["num_legs_distribution"]["n_six_leg_sites"] > 0
                    else "CONFIRMED"),
    }
    print("\n=== Docstring claim verdicts (6-leg vs 4-leg, continuous slope) ===")
    print(f"  claimed: +280% (CI +80% to +700%)")
    for lab, v in verdicts.items():
        print(f"  {lab:8s} measured {v['measured_point_pct']:+.1f}%  "
              f"95%CI [{v['measured_ci95_pct'][0]:+.1f}%, {v['measured_ci95_pct'][1]:+.1f}%]  "
              f"90%CI [{v['measured_ci90_pct'][0]:+.1f}%, {v['measured_ci90_pct'][1]:+.1f}%]  "
              f"-> {v['verdict']}")
    print(f"  2-4 leg share: measured "
          f"{results['docstring_share_verdict']['measured_pct']:.2f}% vs claimed 97% -> "
          f"{results['docstring_share_verdict']['verdict']}")
    print(f"  six-leg sites: {results['docstring_six_leg_verdict']['measured_n_six_leg_sites']} "
          f"-> {results['docstring_six_leg_verdict']['verdict']}")

    (OUT / "e5_results.json").write_text(
        json.dumps(strip_results(results), indent=2), encoding="utf-8")
    print(f"\nWrote {OUT / 'e5_results.json'}")


if __name__ == "__main__":
    main()
