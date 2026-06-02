"""Derive the top model-driven contributing factors for an intersection.

The fitted Negative Binomial gives a log-linear rate:
    log(μ) = β·x + log(years)

Each feature contributes a multiplicative factor to the rate, expressed
relative to a "reference" intersection (unsignalized, 4-leg, 25 mph,
no bike facility, local-access arterial class 0). This module computes
those factors and returns them sorted by absolute log-magnitude.

Output shape per intersection: list of {label, pct_change} dicts, e.g.
    [
        {"label": "Minor arterial", "pct_change": 220.0},
        {"label": "Unsignalized",   "pct_change": 0.0},
        ...
    ]

A positive pct_change indicates the feature raises expected crashes
relative to the reference; negative indicates it lowers them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from pipeline.feature_encoding import (
    REFERENCE_NUM_LEGS,
    leg_category,
    leg_category_param,
    leg_label,
)

ARTERIAL_LABELS = {
    1: "Principal arterial",
    2: "Minor arterial",
    3: "Collector arterial",
    4: "Other arterial",
    5: "Other arterial (5)",
}

REFERENCE_SPEED_MPH = 25
# Median AWDT across the modelled arterial set (~8,000 vehicles/day on
# Capitol Hill). Comparable to a typical Seattle minor arterial — a
# convenient anchor for relative-volume framing.
REFERENCE_AADT      = 8000


@dataclass(frozen=True)
class Contributor:
    label: str
    pct_change: float

    def to_dict(self) -> dict:
        return {"label": self.label, "pct_change": round(self.pct_change, 1)}


def compute_for_row(features: Mapping, params: Mapping[str, float]) -> list[Contributor]:
    """Return contributors sorted by |log(factor)| descending."""
    raw: list[Contributor] = []
    raw.extend(_signal_contributor(features, params))
    raw.extend(_bike_facility_contributor(features, params))
    raw.extend(_arterial_class_contributor(features, params))
    raw.append(_speed_contributor(features, params))
    raw.append(_num_legs_contributor(features, params))
    raw.extend(_aadt_contributor(features, params))
    raw.sort(key=lambda c: -abs(math.log(1.0 + c.pct_change / 100.0)))
    return raw


def compute_for_row_as_dicts(
    features: Mapping,
    params: Mapping[str, float],
    top_n: int = 3,
) -> list[dict]:
    return [c.to_dict() for c in compute_for_row(features, params)[:top_n]]


# ---------------------------------------------------------------------------
# Per-feature contributions
# ---------------------------------------------------------------------------

def _signal_contributor(features: Mapping, params: Mapping[str, float]) -> list[Contributor]:
    if "is_signalized" not in params:
        return []
    coef = params["is_signalized"]
    if int(features.get("is_signalized", 0)) == 1:
        return [Contributor("Signalized", 100 * (math.exp(coef) - 1))]
    return [Contributor("Unsignalized", 0.0)]


def _bike_facility_contributor(features: Mapping, params: Mapping[str, float]) -> list[Contributor]:
    if "bike_facility" not in params:
        return []
    coef = params["bike_facility"]
    if int(features.get("bike_facility", 0)) == 1:
        return [Contributor("Bike facility present", 100 * (math.exp(coef) - 1))]
    return [Contributor("No bike facility", 0.0)]


def _arterial_class_contributor(
    features: Mapping, params: Mapping[str, float]
) -> list[Contributor]:
    cls = int(features.get("arterial_class", 0))
    if cls == 0:
        return [Contributor("Local / non-arterial", 0.0)]
    dummy = f"C(arterial_class)[T.{cls}]"
    if dummy not in params:
        return []
    coef = params[dummy]
    label = ARTERIAL_LABELS.get(cls, f"Arterial class {cls}")
    return [Contributor(label, 100 * (math.exp(coef) - 1))]


def _speed_contributor(features: Mapping, params: Mapping[str, float]) -> Contributor:
    coef = params.get("max_speed_limit", 0.0)
    speed = float(features.get("max_speed_limit", REFERENCE_SPEED_MPH))
    delta = speed - REFERENCE_SPEED_MPH
    pct = 100 * (math.exp(coef * delta) - 1)
    return Contributor(f"Speed limit {int(speed)} mph", pct)


def _num_legs_contributor(features: Mapping, params: Mapping[str, float]) -> Contributor:
    legs = int(features.get("num_legs", REFERENCE_NUM_LEGS))
    label = leg_label(legs)
    if leg_category(legs) == REFERENCE_NUM_LEGS:
        return Contributor(label, 0.0)
    coef = params.get(leg_category_param(legs), 0.0)
    return Contributor(label, 100 * (math.exp(coef) - 1))


def _aadt_contributor(features: Mapping, params: Mapping[str, float]) -> list[Contributor]:
    """Exposure factor: relative-AADT contribution vs. the modelled median."""
    if "log_aadt" not in params:
        return []
    aadt = features.get("max_aadt")
    if not _is_positive_finite(aadt):
        return []
    coef = params["log_aadt"]
    delta = math.log(float(aadt)) - math.log(REFERENCE_AADT)
    pct = 100 * (math.exp(coef * delta) - 1)
    return [Contributor(f"AADT {int(aadt):,} veh/day", pct)]


def _is_positive_finite(value) -> bool:
    if value is None:
        return False
    try:
        f = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f) and f > 0
