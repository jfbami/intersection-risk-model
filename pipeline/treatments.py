"""Phase-5 prescriptive analysis: rank treatments per intersection using CMFs.

Loads a curated library of Crash Modification Factors (data/cmf_library.json),
filters to treatments applicable at each intersection by simple precondition
checks, and applies the HSM Part C counterfactual formula:

    μ_post = μ_pre × CMF
    crashes_prevented_per_year = expected_bike_ksi_per_year × (1 - CMF)

Uncertainty propagates through both the prediction credible interval (from
score_risk) and the CMF interval (from the library):

    prevented_low  = pred_ci_low  × (1 - cmf_ci_high)
    prevented_high = pred_ci_high × (1 - cmf_ci_low)

Preconditions in the library are simple feature comparisons. Three suffix
conventions are recognised:

    "key"             equality:   features[key] == value
    "key_at_least"    threshold:  features[key] >= value
    "key_at_most"     threshold:  features[key] <= value

A treatment is applicable to an intersection only when every precondition
key matches. Empty preconditions = universally applicable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parent.parent
CMF_LIBRARY_PATH = ROOT / "data" / "cmf_library.json"

AT_LEAST_SUFFIX = "_at_least"
AT_MOST_SUFFIX  = "_at_most"


class CmfLibraryError(RuntimeError):
    """Raised when the CMF library is missing or malformed."""


@dataclass(frozen=True)
class Treatment:
    id: str
    name: str
    description: str
    cmf: float
    cmf_ci_low: float
    cmf_ci_high: float
    applies_to: str
    preconditions: Mapping[str, Any]
    source: str
    notes: str


@dataclass(frozen=True)
class PredictionInterval:
    """Per-intersection prediction with credible interval (per year)."""
    mean:    float
    ci_low:  float
    ci_high: float


@dataclass(frozen=True)
class TreatmentRanking:
    id: str
    name: str
    prevented_per_year_mean:    float
    prevented_per_year_ci_low:  float
    prevented_per_year_ci_high: float
    cmf: float

    def to_dict(self) -> dict:
        return {
            "id":                         self.id,
            "name":                       self.name,
            "prevented_per_year_mean":    round(self.prevented_per_year_mean, 4),
            "prevented_per_year_ci_low":  round(self.prevented_per_year_ci_low, 4),
            "prevented_per_year_ci_high": round(self.prevented_per_year_ci_high, 4),
            "cmf":                        round(self.cmf, 3),
        }


# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------

def load_treatments(path: Path = CMF_LIBRARY_PATH) -> list[Treatment]:
    """Load and validate the curated CMF library."""
    if not path.exists():
        raise CmfLibraryError(f"CMF library not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    raw_treatments = payload.get("treatments")
    if not isinstance(raw_treatments, list) or not raw_treatments:
        raise CmfLibraryError(f"{path}: 'treatments' must be a non-empty list")
    return [_build_treatment(entry, path) for entry in raw_treatments]


def _build_treatment(entry: dict, path: Path) -> Treatment:
    required = (
        "id", "name", "description", "cmf", "cmf_ci_low", "cmf_ci_high",
        "applies_to", "preconditions", "source", "notes",
    )
    missing = [k for k in required if k not in entry]
    if missing:
        raise CmfLibraryError(f"{path}: treatment {entry.get('id', '?')!r} missing fields: {missing}")
    return Treatment(
        id=entry["id"],
        name=entry["name"],
        description=entry["description"],
        cmf=float(entry["cmf"]),
        cmf_ci_low=float(entry["cmf_ci_low"]),
        cmf_ci_high=float(entry["cmf_ci_high"]),
        applies_to=entry["applies_to"],
        preconditions=dict(entry["preconditions"]),
        source=entry["source"],
        notes=entry["notes"],
    )


# ---------------------------------------------------------------------------
# Applicability
# ---------------------------------------------------------------------------

def is_applicable(features: Mapping, treatment: Treatment, mode: str) -> bool:
    """True when every precondition matches and applies_to matches the mode."""
    if treatment.applies_to != mode:
        return False
    return all(
        _precondition_holds(features, key, value)
        for key, value in treatment.preconditions.items()
    )


def _precondition_holds(features: Mapping, raw_key: str, value: Any) -> bool:
    if raw_key.endswith(AT_LEAST_SUFFIX):
        feature_key = raw_key[: -len(AT_LEAST_SUFFIX)]
        actual = features.get(feature_key)
        return actual is not None and actual >= value
    if raw_key.endswith(AT_MOST_SUFFIX):
        feature_key = raw_key[: -len(AT_MOST_SUFFIX)]
        actual = features.get(feature_key)
        return actual is not None and actual <= value
    return features.get(raw_key) == value


def applicable_treatments(
    features: Mapping, treatments: list[Treatment], mode: str
) -> list[Treatment]:
    return [t for t in treatments if is_applicable(features, t, mode)]


# ---------------------------------------------------------------------------
# Effect estimation
# ---------------------------------------------------------------------------

def compute_prevention(
    prediction: PredictionInterval, treatment: Treatment
) -> TreatmentRanking:
    """Apply HSM Part C counterfactual: prevented = prediction × (1 - CMF).

    Pessimistic interval combination — the prevention CI is widest when the
    prediction CI and the CMF CI both work against the estimate.
    """
    mean    = prediction.mean    * (1.0 - treatment.cmf)
    ci_low  = prediction.ci_low  * (1.0 - treatment.cmf_ci_high)
    ci_high = prediction.ci_high * (1.0 - treatment.cmf_ci_low)
    return TreatmentRanking(
        id=treatment.id,
        name=treatment.name,
        prevented_per_year_mean=mean,
        prevented_per_year_ci_low=ci_low,
        prevented_per_year_ci_high=ci_high,
        cmf=treatment.cmf,
    )


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def rank_recommended_treatments(
    features: Mapping,
    prediction: PredictionInterval,
    treatments: list[Treatment],
    mode: str,
    top_n: int = 3,
) -> list[TreatmentRanking]:
    """Top-N applicable treatments ranked by expected prevented per year."""
    candidates = [
        compute_prevention(prediction, t)
        for t in applicable_treatments(features, treatments, mode)
    ]
    candidates.sort(key=lambda r: r.prevented_per_year_mean, reverse=True)
    return candidates[:top_n]


def rank_as_dicts(
    features: Mapping,
    prediction: PredictionInterval,
    treatments: list[Treatment],
    mode: str,
    top_n: int = 3,
) -> list[dict]:
    return [r.to_dict() for r in rank_recommended_treatments(features, prediction, treatments, mode, top_n)]
