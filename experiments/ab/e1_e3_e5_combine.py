"""Merge e1/e3/e5 result JSONs into one machine-readable file and print the
plain-KFold secondary CV table (the stratified table is printed by each script)."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "results"
combined = {"generated_by": "experiments/ab/{e1_volume_form,e3_bike_exposure,e5_leg_encoding}.py"}
for name in ("e1", "e3", "e5"):
    p = OUT / f"{name}_results.json"
    combined[name.upper()] = json.loads(p.read_text(encoding="utf-8"))
combined["E1_raw_aadt_optimizer_diagnostic"] = json.loads(
    (OUT / "e1_raw_aadt_diagnostic.json").read_text(encoding="utf-8"))
(OUT / "e1_e3_e5_results.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
print(f"Wrote {OUT / 'e1_e3_e5_results.json'}")


def row(tag, r):
    k, s = r["cv_kfold"], r["cv_stratified"]
    print(f"  {tag:34s} KFold: MAE {k['cv_mae_mean']:.4f}+-{k['cv_mae_sd']:.4f} "
          f"RMSE {k['cv_rmse_mean']:.4f}+-{k['cv_rmse_sd']:.4f} "
          f"oofMAE {k['pooled_oof_mae']:.4f} oofRMSE {k['pooled_oof_rmse']:.4f} "
          f"rho {k['pooled_oof_spearman']:.3f} | Strat oofMAE {s['pooled_oof_mae']:.4f}")


print("\nPlain KFold(n_splits=5, shuffle=True, random_state=0) - secondary CV")
print("\nE1")
for mode, mr in combined["E1"]["modes"].items():
    for spec, r in mr.items():
        if spec.startswith("_"):
            continue
        row(f"{mode}/{spec}", r)
print("\nE3")
for spec, r in combined["E3"]["specs_results"].items():
    row(f"bike/{spec}", r)
print("\nE5")
for mode, mr in combined["E5"]["modes"].items():
    for spec, r in mr["specs"].items():
        row(f"{mode}/{spec}", r)
