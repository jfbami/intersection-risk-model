"""Counterfactual prediction for the intersection risk model.

Wraps the fitted Negative Binomial result with a clean API: given an
intersection's current feature row and a set of hypothetical overrides,
return the model's expected crashes per year under both configurations
and the delta.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "data" / "model" / "nb_v2_arterial_aadt.pkl"
YEARS_OBSERVED = 6

MODEL_FEATURES = (
    "is_signalized",
    "num_legs",
    "max_speed_limit",
    "bike_facility",
    "arterial_class",
)


@lru_cache(maxsize=1)
def _load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Run: python pipeline/fit_risk_model.py"
        )
    return sm.load(str(MODEL_PATH))


def _row_to_frame(row: Mapping, overrides: Mapping | None = None) -> pd.DataFrame:
    overrides = overrides or {}
    df = pd.DataFrame([{
        "is_signalized":   int(overrides.get("is_signalized",   row["is_signalized"])),
        "num_legs":        int(overrides.get("num_legs",        row["num_legs"])),
        "max_speed_limit": float(overrides.get("max_speed_limit", row["max_speed_limit"])),
        "bike_facility":   int(overrides.get("bike_facility",   row["bike_facility"])),
        "arterial_class":  int(overrides.get("arterial_class",  row["arterial_class"])),
    }])
    df["offset"] = np.log(YEARS_OBSERVED)
    return df


def predict_per_year(row: Mapping, overrides: Mapping | None = None) -> float:
    """Expected crashes per year at the given (possibly hypothetical) features."""
    result = _load_model()
    df = _row_to_frame(row, overrides)
    pred_total = float(result.predict(df, offset=df["offset"].values).iloc[0])
    return pred_total / YEARS_OBSERVED


def compare(row: Mapping, overrides: Mapping) -> dict:
    """Return current vs. hypothetical predictions and the delta."""
    current = predict_per_year(row, overrides=None)
    hypothetical = predict_per_year(row, overrides=overrides)
    delta = hypothetical - current
    pct = (delta / current * 100) if current > 0 else float("nan")
    return {
        "current_per_year":      current,
        "hypothetical_per_year": hypothetical,
        "delta":                 delta,
        "pct_change":            pct,
    }
