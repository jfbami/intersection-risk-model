"""E1 diagnostic: why does ped ~ SHARED + max_aadt fail to converge, and is the
failure intrinsic to the model or just numeric conditioning of a raw predictor
spanning 1e3..4e4? Tests several optimizers and a pure rescaling (AADT/10000),
which is a reparameterisation that cannot change the fitted model."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

from experiments.ab.e1_e3_e5_common import SHARED_PREDICTORS, get_df  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results"
df = get_df()
df["aadt_10k"] = df["max_aadt"] / 10000.0

CASES = [
    ("ped_total", f"{SHARED_PREDICTORS} + max_aadt", "raw max_aadt"),
    ("ped_total", f"{SHARED_PREDICTORS} + aadt_10k", "max_aadt/10000 (rescaled)"),
    ("vehicle_only_total", f"{SHARED_PREDICTORS} + max_aadt", "raw max_aadt"),
    ("vehicle_only_total", f"{SHARED_PREDICTORS} + aadt_10k", "max_aadt/10000 (rescaled)"),
]
METHODS = [
    ("newton (default)", {}),
    ("bfgs maxiter=200", {"method": "bfgs", "maxiter": 200}),
    ("bfgs maxiter=5000", {"method": "bfgs", "maxiter": 5000}),
    ("nm maxiter=10000", {"method": "nm", "maxiter": 10000}),
    ("lbfgs maxiter=5000", {"method": "lbfgs", "maxiter": 5000}),
]

out = []
for target, pred, desc in CASES:
    print(f"\n=== {target} ~ ... + {desc} ===")
    for mname, kw in METHODS:
        rec = {"target": target, "spec": desc, "method": mname}
        try:
            r = smf.negativebinomial(f"{target} ~ {pred}", data=df,
                                     offset=df["offset"].values).fit(disp=False, **kw)
            vt = "max_aadt" if "max_aadt" in pred else "aadt_10k"
            b = float(r.params[vt])
            se = float(r.bse[vt])
            # express on the per-unit-AADT scale for comparability
            b_per_aadt = b if vt == "max_aadt" else b / 10000.0
            rec.update({"converged": bool(r.mle_retvals.get("converged", True)),
                        "llf": float(r.llf), "aic": float(r.aic),
                        "beta_raw_scale": b, "se_raw_scale": se,
                        "beta_per_unit_aadt": b_per_aadt,
                        "pvalue": float(r.pvalues[vt]),
                        "alpha": float(r.params.get("alpha", np.nan))})
            print(f"  {mname:20s} conv={rec['converged']!s:5s} llf={r.llf:10.4f} "
                  f"aic={r.aic:9.2f} beta/AADT={b_per_aadt:.6e} se={se:.6g} p={rec['pvalue']:.4g}")
        except Exception as e:
            rec.update({"converged": False, "error": repr(e)})
            print(f"  {mname:20s} EXCEPTION {e!r}")
        out.append(rec)

# best achievable llf for the raw-AADT ped spec, whatever the route
ped = [r for r in out if r["target"] == "ped_total" and "llf" in r]
if ped:
    best = max(ped, key=lambda r: r["llf"])
    print(f"\nBest ped raw/rescaled-AADT llf found: {best['llf']:.4f} "
          f"via {best['method']} on {best['spec']} (converged={best['converged']})")
(OUT / "e1_raw_aadt_diagnostic.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(f"\nWrote {OUT / 'e1_raw_aadt_diagnostic.json'}")
