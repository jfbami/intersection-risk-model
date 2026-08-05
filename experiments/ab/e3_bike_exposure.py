"""E3 - Bike volume/exposure predictor: centrality vs AADT vs both vs neither.

Target: bike_total (169 events over 346 arterial sites).

Run:
  cd C:/Users/jfbaa/project-cycle-group && \
  PYTHONPATH=C:/Users/jfbaa/project-cycle-group python experiments/ab/e3_bike_exposure.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from scipy import stats as sps

warnings.filterwarnings("ignore")

from statsmodels.stats.outliers_influence import variance_inflation_factor  # noqa: E402

from experiments.ab.e1_e3_e5_common import (  # noqa: E402
    SHARED_PREDICTORS, design, evaluate, get_df, print_row, strip_results,
)

OUT = Path(__file__).resolve().parent.parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

TARGET = "bike_total"
SPECS = {
    "a_centrality":  f"{SHARED_PREDICTORS} + log_bike_centrality",
    "b_aadt":        f"{SHARED_PREDICTORS} + log_aadt",
    "c_both":        f"{SHARED_PREDICTORS} + log_bike_centrality + log_aadt",
    "d_neither":     SHARED_PREDICTORS,
}
EXPOSURE_TERMS = {
    "a_centrality": ["log_bike_centrality"],
    "b_aadt": ["log_aadt"],
    "c_both": ["log_bike_centrality", "log_aadt"],
    "d_neither": [],
}


def main():
    df = get_df()
    x, z = df["log_bike_centrality"].values, df["log_aadt"].values
    pr, pp = sps.pearsonr(x, z)
    sr, sp = sps.spearmanr(x, z)
    prr, ppr = sps.pearsonr(df["bike_centrality"].values, df["max_aadt"].values)

    results = {
        "experiment": "E3",
        "question": "Does log_bike_centrality earn its place as the bike model's exposure term?",
        "reason": ("The bike model silently uses log_bike_centrality where ped/vehicle use "
                   "log_aadt. No commit message, docstring or note justifies the swap, and in "
                   "the production fit the centrality coefficient is not significant "
                   "(beta=0.141, SE=0.157, p=0.369)."),
        "target": TARGET,
        "n_rows": int(len(df)),
        "n_events": int(df[TARGET].sum()),
        "n_zero_sites": int((df[TARGET] == 0).sum()),
        "specs": SPECS,
        "correlation_centrality_vs_aadt": {
            "pearson_r_logs": float(pr), "pearson_p_logs": float(pp),
            "spearman_rho_logs": float(sr), "spearman_p_logs": float(sp),
            "pearson_r_raw": float(prr), "pearson_p_raw": float(ppr),
            "r_squared_logs": float(pr ** 2),
        },
        "specs_results": {},
    }
    print(f"E3: target={TARGET}  n={len(df)}  events={int(df[TARGET].sum())}  "
          f"zero-count sites={int((df[TARGET] == 0).sum())}")
    print(f"  corr(log_bike_centrality, log_aadt): pearson r={pr:.4f} (p={pp:.3g}, "
          f"r2={pr**2:.4f}), spearman rho={sr:.4f} (p={sp:.3g})")
    print(f"  corr(bike_centrality, max_aadt) raw: pearson r={prr:.4f} (p={ppr:.3g})\n")

    for spec, pred in SPECS.items():
        rec = evaluate(df, TARGET, pred, spec)
        rec.pop("_beta", None)
        rec.pop("_design_info", None)
        ins = rec["insample"]
        if ins.get("ok"):
            rec["exposure_terms"] = {
                t: {"beta": ins["params"][t], "se": ins["bse"][t], "p": ins["pvalues"][t],
                    "exp_beta": float(np.exp(ins["params"][t])),
                    "ci95_lo": ins["params"][t] - 1.959964 * ins["bse"][t],
                    "ci95_hi": ins["params"][t] + 1.959964 * ins["bse"][t]}
                for t in EXPOSURE_TERMS[spec]
            }
        results["specs_results"][spec] = rec
        print_row(spec, rec)
        for t, v in rec.get("exposure_terms", {}).items():
            print(f"                 {t}: beta={v['beta']:.6g} se={v['se']:.6g} "
                  f"p={v['p']:.4g}  exp(beta)={v['exp_beta']:.4f}  "
                  f"95%CI[{v['ci95_lo']:.4g}, {v['ci95_hi']:.4g}]")

    # ---- VIF for spec (c) ----
    _y, Xc = design(df, TARGET, SPECS["c_both"])
    Xn = Xc.to_numpy(dtype=float)
    vif = {c: float(variance_inflation_factor(Xn, i)) for i, c in enumerate(Xc.columns)}
    results["vif_spec_c"] = vif
    print("\n  VIF, spec (c) design matrix (intercept included, so its VIF is not "
          "interpretable):")
    for c, v in sorted(vif.items(), key=lambda kv: -kv[1]):
        print(f"    {v:9.3f}  {c}")

    # ---- likelihood-ratio tests vs the nested null (d) ----
    lld = results["specs_results"]["d_neither"]["insample"]
    lrts = {}
    for spec in ("a_centrality", "b_aadt", "c_both"):
        ins = results["specs_results"][spec]["insample"]
        if ins.get("ok") and lld.get("ok"):
            stat = 2 * (ins["llf"] - lld["llf"])
            dfree = ins["n_params"] - lld["n_params"]
            lrts[spec] = {"vs": "d_neither", "lr_stat": float(stat), "df": int(dfree),
                          "p": float(sps.chi2.sf(max(stat, 0.0), dfree))}
    results["lrt_vs_d_neither"] = lrts
    print("\n  Likelihood-ratio tests against spec (d) 'no exposure term':")
    for spec, v in lrts.items():
        print(f"    {spec:14s} LR={v['lr_stat']:7.4f}  df={v['df']}  p={v['p']:.4f}")

    # ---- OOS ranking ----
    rank = sorted(
        ((s, results["specs_results"][s]["repeated_cv"]["oof_mae_mean"],
          results["specs_results"][s]["repeated_cv"]["oof_mae_sd"],
          results["specs_results"][s]["cv_stratified"]["pooled_oof_mae"],
          results["specs_results"][s]["cv_stratified"]["cv_mae_sd"])
         for s in SPECS if "oof_mae_mean" in results["specs_results"][s]["repeated_cv"]),
        key=lambda t: t[1])
    results["oos_ranking_by_repeated_oof_mae"] = [
        {"spec": s, "oof_mae_mean": m, "oof_mae_sd_across_seeds": sd,
         "oof_mae_seed0": s0, "fold_sd_seed0": fsd} for s, m, sd, s0, fsd in rank]
    print("\n  OOS ranking (repeated stratified 5-fold, 10 seeds, pooled OOF MAE):")
    for s, m, sd, s0, fsd in rank:
        print(f"    {s:14s} {m:.4f} +- {sd:.4f} (seed-to-seed)   "
              f"[seed0 pooled {s0:.4f}, fold SD {fsd:.4f}]")
    best, worst = rank[0], rank[-1]
    results["oos_margin_best_vs_worst"] = float(worst[1] - best[1])
    results["oos_margin_best_vs_production_a"] = float(
        next(m for s, m, _, _, _ in rank if s == "a_centrality") - best[1])
    print(f"\n  margin best({best[0]}) vs worst({worst[0]}) = {worst[1] - best[1]:.4f} MAE")

    (OUT / "e3_results.json").write_text(
        json.dumps(strip_results(results), indent=2), encoding="utf-8")
    print(f"\nWrote {OUT / 'e3_results.json'}")


if __name__ == "__main__":
    main()
