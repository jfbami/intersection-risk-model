"""E1 - Volume functional form: log(AADT) vs raw AADT vs sqrt(AADT) vs none.

Modes: ped, vehicle (the two that use AADT).
Also directly tests the README:125-140 claim that raw AADT under a log link
yields "astronomical predictions" at a 50,000-AADT site.

Run:
  cd C:/Users/jfbaa/project-cycle-group && \
  PYTHONPATH=C:/Users/jfbaa/project-cycle-group python experiments/ab/e1_volume_form.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from experiments.ab.e1_e3_e5_common import (  # noqa: E402
    SHARED_PREDICTORS, evaluate, get_df, predict_new, print_row, strip_results,
)

OUT = Path(__file__).resolve().parent.parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

SPECS = {
    "a_log_aadt":  f"{SHARED_PREDICTORS} + log_aadt",
    "b_raw_aadt":  f"{SHARED_PREDICTORS} + max_aadt",
    "c_sqrt_aadt": f"{SHARED_PREDICTORS} + sqrt_aadt",
    "d_no_volume": SHARED_PREDICTORS,
}
VOLUME_TERM = {"a_log_aadt": "log_aadt", "b_raw_aadt": "max_aadt",
               "c_sqrt_aadt": "sqrt_aadt", "d_no_volume": None}
TARGETS = {"ped": "ped_total", "vehicle": "vehicle_only_total"}


def reference_row(df: pd.DataFrame) -> dict:
    """Other predictors at dataset mode (categorical/binary) or median (speed)."""
    return {
        "is_signalized":   int(df["is_signalized"].mode().iloc[0]),
        "legs_cat":        int(df["legs_cat"].mode().iloc[0]),
        "num_legs":        int(df["num_legs"].mode().iloc[0]),
        "max_speed_limit": float(df["max_speed_limit"].median()),
        "bike_facility":   int(df["bike_facility"].mode().iloc[0]),
        "arterial_class":  int(df["arterial_class"].mode().iloc[0]),
        "offset":          float(np.log(6.0)),
    }


def main():
    df = get_df()
    aadt_max = float(df["max_aadt"].max())
    results = {
        "experiment": "E1",
        "question": "Volume functional form for the ped and vehicle NB2 SPFs",
        "reason": ("README.md:125-140 argues for log(AADT) on functional-form theory alone: "
                   "raw AADT under a log link implies mu ~ exp(beta*AADT), which the README "
                   "calls 'not physical' and claims gives 'astronomical predictions' at a "
                   "50,000-AADT site. No alternative was ever fit."),
        "n_rows": int(len(df)),
        "shared_predictors": SHARED_PREDICTORS,
        "specs": SPECS,
        "aadt_distribution": {
            "min": float(df["max_aadt"].min()), "max": aadt_max,
            "mean": float(df["max_aadt"].mean()), "median": float(df["max_aadt"].median()),
            "sd": float(df["max_aadt"].std()),
            **{f"q{int(q*100):02d}": float(df["max_aadt"].quantile(q))
               for q in (0.05, 0.25, 0.75, 0.95, 0.99)},
            **{f"n_above_{t}": int((df["max_aadt"] > t).sum())
               for t in (20000, 25000, 30000, 40000, 50000)},
        },
        "reference_row": reference_row(df),
        "modes": {},
    }
    print(f"E1: n={len(df)}  max_aadt in [{df['max_aadt'].min():.0f}, {aadt_max:.0f}]  "
          f"n>30k={(df['max_aadt'] > 30000).sum()}  n>50k={(df['max_aadt'] > 50000).sum()}")

    for mode, target in TARGETS.items():
        print(f"\n=== {mode} ({target}, {int(df[target].sum())} events) ===")
        mode_rec, betas, dinfos = {}, {}, {}
        for spec, pred in SPECS.items():
            rec = evaluate(df, target, pred, spec)
            betas[spec] = rec.pop("_beta")
            dinfos[spec] = rec.pop("_design_info")
            ins = rec["insample"]
            vt = VOLUME_TERM[spec]
            if vt and ins.get("ok"):
                b, se, p = ins["params"][vt], ins["bse"][vt], ins["pvalues"][vt]
                rec["volume_term"] = {"name": vt, "beta": b, "se": se, "p": p,
                                      "ci95_lo": b - 1.959964 * se, "ci95_hi": b + 1.959964 * se}
            mode_rec[spec] = rec
            print_row(spec, rec)
            if vt and ins.get("ok"):
                v = rec["volume_term"]
                print(f"                 {vt}: beta={v['beta']:.6g} se={v['se']:.6g} "
                      f"p={v['p']:.4g}  95%CI[{v['ci95_lo']:.6g}, {v['ci95_hi']:.6g}]")
            npf = ins.get("naive_production_fit", {})
            if npf.get("ok") and abs(npf["llf"] - ins["llf"]) > 0.01:
                print(f"                 !! naive production-style fit lands at llf="
                      f"{npf['llf']:.4f} / AIC={npf['aic']:.2f} (conv={npf['converged']}) "
                      f"vs robust llf={ins['llf']:.4f} / AIC={ins['aic']:.2f}")

        # ---- README extrapolation claim ----
        ref = reference_row(df)
        grid = [float(df["max_aadt"].median()), float(df["max_aadt"].quantile(0.95)),
                aadt_max, 50000.0, 100000.0]
        rows = []
        for aadt in grid:
            nd = pd.DataFrame([{**ref, "max_aadt": float(aadt)}])
            nd["log_aadt"] = np.log(nd["max_aadt"])
            nd["sqrt_aadt"] = np.sqrt(nd["max_aadt"])
            r = {"max_aadt": float(aadt)}
            for spec in SPECS:
                if betas[spec] is None:
                    r[spec] = None
                    continue
                r[spec] = float(predict_new(dinfos[spec], betas[spec], nd,
                                            nd["offset"].values)[0])
            r["ratio_raw_over_log"] = (r["b_raw_aadt"] / r["a_log_aadt"]
                                       if r.get("a_log_aadt") else None)
            rows.append(r)
        mode_rec["_extrapolation"] = {
            "reference_row": ref, "aadt_grid": grid,
            "note": ("expected crash count over the full 6-year window; other predictors "
                     "held at dataset mode (binary/categorical) or median (speed)"),
            "predicted_counts_6yr": rows,
            "observed_max_aadt": aadt_max,
            "aadt_50k_is_extrapolation_beyond_observed_max": True,
            "pct_beyond_observed_max_at_50k": float(100 * (50000 - aadt_max) / aadt_max),
        }
        print("\n  README claim test - expected crashes over 6 years "
              "(others at mode/median):")
        print(f"    {'AADT':>10s} " + " ".join(f"{s:>13s}" for s in SPECS) + "   raw/log")
        for r in rows:
            cells = " ".join(f"{r[s]:13.4f}" if r[s] is not None and abs(r[s]) < 1e7
                             else f"{r[s]:13.4g}" for s in SPECS)
            rr = r["ratio_raw_over_log"]
            tag = "  <- observed max" if abs(r["max_aadt"] - aadt_max) < 1 else ""
            tag = "  <- README's 50k" if abs(r["max_aadt"] - 50000) < 1 else tag
            print(f"    {r['max_aadt']:10.0f} {cells}   {rr:7.2f}x{tag}")
        results["modes"][mode] = mode_rec

    (OUT / "e1_results.json").write_text(
        json.dumps(strip_results(results), indent=2), encoding="utf-8")
    print(f"\nWrote {OUT / 'e1_results.json'}")


if __name__ == "__main__":
    main()
