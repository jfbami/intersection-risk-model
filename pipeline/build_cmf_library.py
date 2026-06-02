"""Build data/cmf_library.json from the FHWA CMF Clearinghouse CSV export.

One-shot ingester — re-run when the Clearinghouse refreshes its database.

Inputs
------
data/raw/cmf_clearinghouse_<YYYY-MM-DD>.csv
    Full Clearinghouse database export (~9,800 CMFs, 183 columns).
data/cmf_curation_notes.json
    Domain commentary per treatment id (kept separate so reviewers can edit
    prose without touching Python source).

Output
------
data/cmf_library.json
    Curated library of CMFs relevant to bike-KSI at urban arterial intersections.

Methodology
-----------
1. Filter the full Clearinghouse to: approved studies, bike-involved crash
   types, intersection-related, non-rural.
2. For each curated cmid, aggregate qualifying studies:
     - all studies report an SE → variance-weighted average
     - some / none report an SE → simple mean with across-study variance for CI
     - single study → that study's value (with its SE if present, else point estimate)
3. For anti-treatments (cmid measures an action we want to PROHIBIT),
   invert the AMF before emitting so downstream consumers can compute
   prevention the standard way: `prevented = prediction × (1 − cmf)`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLEARINGHOUSE_CSV     = ROOT / "data" / "raw" / "cmf_clearinghouse_2025-11-10.csv"
CURATION_NOTES_PATH   = ROOT / "data" / "cmf_curation_notes.json"
OUTPUT_JSON           = ROOT / "data" / "cmf_library.json"
SOURCE_DATE           = "2025-11-10"

Z_90 = 1.6449  # 90% two-sided normal quantile

DIRECTION_HELPFUL          = "helpful"
DIRECTION_ANTI_INDICATION  = "anti_indication"
DIRECTION_INVERTED         = "anti_treatment_inverted_to_prevention"


class CmfLibraryError(RuntimeError):
    """Raised when the Clearinghouse data or curation notes are malformed."""


# ---------------------------------------------------------------------------
# Curated mapping
# ---------------------------------------------------------------------------
# Pairs each Clearinghouse cmid we care about with our internal id, the
# recommended-action name we show to users, applicability preconditions in
# our feature space, and a flag marking treatments whose recommended action
# is the INVERSE of the studied action (e.g. cmid 114 measures "permit RTOR";
# the recommended action is "prohibit RTOR"). Long-form evidence commentary
# lives in data/cmf_curation_notes.json keyed by `id`.

@dataclass(frozen=True)
class CuratedEntry:
    id: str
    cmid: int
    recommended_name: str
    description: str
    preconditions: dict
    studied_action_is_inverse: bool
    applies_to: str


BIKE_CURATED: tuple[CuratedEntry, ...] = (
    CuratedEntry(
        id="cycle_track_at_intersection",
        cmid=1080,
        recommended_name="Install cycle track / protected bike lane at intersection",
        description="Physically separated cycle track (raised, bollard-, or curb-protected) extending through the intersection.",
        preconditions={"bike_facility": 0},
        studied_action_is_inverse=False,
        applies_to="bike",
    ),
    CuratedEntry(
        id="offset_cycle_track_with_priority",
        cmid=1037,
        recommended_name="Cycle track set back 2–5m from the main road with cyclist priority",
        description="Cycle track offset from the carriageway, with priority for cyclists at intersection crossings.",
        preconditions={"bike_facility": 0},
        studied_action_is_inverse=False,
        applies_to="bike",
    ),
    CuratedEntry(
        id="raised_bike_crossing",
        cmid=1042,
        recommended_name="Raised bicycle crossing / vehicle speed-reducing measure at side road",
        description="Raised crossing surface or other speed-reducing geometry where vehicles enter/leave the side road.",
        preconditions={},
        studied_action_is_inverse=False,
        applies_to="bike",
    ),
    CuratedEntry(
        id="install_bike_lane_generic",
        cmid=543,
        recommended_name="Install conventional (painted) bike lane",
        description="Standard on-street bike lane separated by paint only.",
        preconditions={"bike_facility": 0},
        studied_action_is_inverse=False,
        applies_to="bike",
    ),
    CuratedEntry(
        id="prohibit_right_turn_on_red",
        cmid=114,
        recommended_name="Prohibit right-turn-on-red",
        description="Sign and enforce 'No turn on red' at the intersection.",
        preconditions={"is_signalized": 1},
        studied_action_is_inverse=True,
        applies_to="bike",
    ),
    CuratedEntry(
        id="convert_yield_to_signalized",
        cmid=1421,
        recommended_name="Convert yield/stop control to signalized",
        description="Install a full signal at a previously yield- or stop-controlled intersection.",
        preconditions={"is_signalized": 0},
        studied_action_is_inverse=False,
        applies_to="bike",
    ),
    CuratedEntry(
        id="bike_lane_at_signalized_intersection",
        cmid=872,
        recommended_name="Install bike lane through a signalized intersection",
        description="Continue a painted bike lane through the conflict zone at a signalized intersection.",
        preconditions={"bike_facility": 0, "is_signalized": 1},
        studied_action_is_inverse=False,
        applies_to="bike",
    ),
    CuratedEntry(
        id="convert_to_roundabout_single_lane",
        cmid=1283,
        recommended_name="Convert intersection to single-lane roundabout (NOT recommended for bike-KSI)",
        description="Replace a conventional intersection with a single-lane modern roundabout.",
        preconditions={},
        studied_action_is_inverse=False,
        applies_to="bike",
    ),
)

PED_CURATED: tuple[CuratedEntry, ...] = (
    CuratedEntry(
        id="lpi_signal_timing",
        cmid=328,
        recommended_name="Modify signal phasing (implement a leading pedestrian interval)",
        description="Gives pedestrians a 3-7 second head start when entering the crosswalk.",
        preconditions={"is_signalized": 1},
        studied_action_is_inverse=False,
        applies_to="ped",
    ),
)

VEHICLE_CURATED: tuple[CuratedEntry, ...] = (
    CuratedEntry(
        id="road_diet",
        cmid=151,
        recommended_name="Road diet (Convert 4-lane undivided road to 2-lanes plus turning lane)",
        description="Convert 4-lane undivided to 3-lane (2 through lanes + center turn lane).",
        preconditions={},
        studied_action_is_inverse=False,
        applies_to="vehicle",
    ),
    CuratedEntry(
        id="prohibit_right_turn_on_red_vehicle",
        cmid=114,
        recommended_name="Prohibit right-turn-on-red",
        description="Sign and enforce 'No turn on red' at the intersection.",
        preconditions={"is_signalized": 1},
        studied_action_is_inverse=True,
        applies_to="vehicle",
    ),
)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_clearinghouse() -> pd.DataFrame:
    if not CLEARINGHOUSE_CSV.exists():
        raise FileNotFoundError(f"Clearinghouse CSV not found at {CLEARINGHOUSE_CSV}")
    return pd.read_csv(CLEARINGHOUSE_CSV, low_memory=False, encoding="utf-8", encoding_errors="replace")


def load_curation_notes(path: Path = CURATION_NOTES_PATH) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Curation notes not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    notes = payload.get("by_id")
    if not isinstance(notes, dict):
        raise CmfLibraryError(f"{path}: 'by_id' must be a dict mapping id -> note text")
    return notes


def filter_intersection_universe(df: pd.DataFrame, crash_type_regex: str) -> pd.DataFrame:
    is_mode     = df["crashType"].astype(str).str.contains(crash_type_regex, case=False, na=False, regex=True)
    is_isect    = df["intersectionRelated"].astype(str).str.lower().eq("yes")
    is_approved = df["approved"].astype(str).str.lower().eq("yes")
    not_rural   = ~df["areaType"].astype(str).str.lower().str.strip().eq("rural")
    return df.loc[is_mode & is_isect & is_approved & not_rural].copy()


# ---------------------------------------------------------------------------
# Per-study extraction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StudySample:
    amf: float
    se:  Optional[float]
    se_method: str          # "adjusted" | "unadjusted" | "none"
    qual_rating: Optional[int]
    citation:    str


def find_studies_for_cmid(universe: pd.DataFrame, cmid: int) -> list[StudySample]:
    sub = universe[universe["cmid"] == cmid]
    samples: list[StudySample] = []
    for _, row in sub.iterrows():
        amf = pd.to_numeric(row.get("accModFactor"), errors="coerce")
        if pd.isna(amf) or amf <= 0:
            continue
        se, method = extract_standard_error(row)
        samples.append(StudySample(
            amf=float(amf),
            se=se,
            se_method=method,
            qual_rating=parse_int_or_none(row.get("qualRating")),
            citation=str(row.get("citation") or "").strip() or f"Clearinghouse cmid {cmid}",
        ))
    return samples


def extract_standard_error(row: pd.Series) -> tuple[Optional[float], str]:
    adj = pd.to_numeric(row.get("adjStanErrorAmf"), errors="coerce")
    if pd.notna(adj) and adj > 0:
        return float(adj), "adjusted"
    unadj = pd.to_numeric(row.get("unAdjStanErrorAmf"), errors="coerce")
    if pd.notna(unadj) and unadj > 0:
        return float(unadj), "unadjusted"
    return None, "none"


def parse_int_or_none(raw) -> Optional[int]:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AggregatedAmf:
    cmf:         float
    cmf_ci_low:  float
    cmf_ci_high: float
    se:          float
    se_method:   str
    n_studies:   int


def aggregate_studies(samples: list[StudySample]) -> AggregatedAmf:
    if not samples:
        raise ValueError("Cannot aggregate zero studies")
    if len(samples) == 1:
        return aggregate_single_study(samples[0])
    if all(s.se is not None for s in samples):
        return aggregate_via_variance_weighting(samples)
    return aggregate_across_studies(samples)


def aggregate_single_study(sample: StudySample) -> AggregatedAmf:
    if sample.se is None:
        return AggregatedAmf(
            cmf=sample.amf, cmf_ci_low=sample.amf, cmf_ci_high=sample.amf,
            se=0.0, se_method="none", n_studies=1,
        )
    return AggregatedAmf(
        cmf         = sample.amf,
        cmf_ci_low  = max(0.0, sample.amf - Z_90 * sample.se),
        cmf_ci_high = sample.amf + Z_90 * sample.se,
        se          = sample.se,
        se_method   = sample.se_method,
        n_studies   = 1,
    )


def aggregate_via_variance_weighting(samples: list[StudySample]) -> AggregatedAmf:
    weights = [1.0 / (s.se ** 2) for s in samples]
    total_w = sum(weights)
    mean    = sum(w * s.amf for w, s in zip(weights, samples)) / total_w
    se      = math.sqrt(1.0 / total_w)
    method  = "adjusted" if all(s.se_method == "adjusted" for s in samples) else "unadjusted"
    return AggregatedAmf(
        cmf         = mean,
        cmf_ci_low  = max(0.0, mean - Z_90 * se),
        cmf_ci_high = mean + Z_90 * se,
        se          = se,
        se_method   = method,
        n_studies   = len(samples),
    )


def aggregate_across_studies(samples: list[StudySample]) -> AggregatedAmf:
    amfs = [s.amf for s in samples]
    mean = sum(amfs) / len(amfs)
    variance = sum((x - mean) ** 2 for x in amfs) / (len(amfs) - 1)
    se = math.sqrt(variance / len(amfs))
    return AggregatedAmf(
        cmf         = mean,
        cmf_ci_low  = max(0.0, mean - Z_90 * se),
        cmf_ci_high = mean + Z_90 * se,
        se          = se,
        se_method   = "across_study",
        n_studies   = len(samples),
    )


# ---------------------------------------------------------------------------
# Recommendation resolution
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResolvedRecommendation:
    cmf:         float
    cmf_ci_low:  float
    cmf_ci_high: float
    direction:   str


def resolve_recommendation(entry: CuratedEntry, agg: AggregatedAmf) -> ResolvedRecommendation:
    if entry.studied_action_is_inverse:
        return ResolvedRecommendation(
            cmf         = 1.0 / agg.cmf,
            cmf_ci_low  = 1.0 / agg.cmf_ci_high,  # inversion swaps the bounds
            cmf_ci_high = 1.0 / agg.cmf_ci_low,
            direction   = DIRECTION_INVERTED,
        )
    direction = DIRECTION_HELPFUL if agg.cmf < 1.0 else DIRECTION_ANTI_INDICATION
    return ResolvedRecommendation(
        cmf         = agg.cmf,
        cmf_ci_low  = agg.cmf_ci_low,
        cmf_ci_high = agg.cmf_ci_high,
        direction   = direction,
    )


def assemble_library_entry(
    entry: CuratedEntry,
    resolved: ResolvedRecommendation,
    agg: AggregatedAmf,
    samples: list[StudySample],
    notes: dict[str, str],
) -> dict:
    return {
        "id":                       entry.id,
        "name":                     entry.recommended_name,
        "description":              entry.description,
        "cmf":                      round(resolved.cmf, 4),
        "cmf_ci_low":               round(resolved.cmf_ci_low, 4),
        "cmf_ci_high":              round(resolved.cmf_ci_high, 4),
        "applies_to":               entry.applies_to,
        "preconditions":            dict(entry.preconditions),
        "direction":                resolved.direction,
        "cmf_clearinghouse_id":     entry.cmid,
        "studied_amf":              round(agg.cmf, 4),
        "studied_amf_ci_low":       round(agg.cmf_ci_low, 4),
        "studied_amf_ci_high":      round(agg.cmf_ci_high, 4),
        "se_method":                agg.se_method,
        "n_studies":                agg.n_studies,
        "studies":                  [study_to_dict(s) for s in samples],
        "source":                   f"FHWA CMF Clearinghouse (cmid {entry.cmid}, export {SOURCE_DATE})",
        "notes":                    notes.get(entry.id, ""),
    }


def study_to_dict(sample: StudySample) -> dict:
    return {
        "citation":    sample.citation,
        "amf":         round(sample.amf, 4),
        "se":          round(sample.se, 4) if sample.se is not None else None,
        "se_method":   sample.se_method,
        "qual_rating": sample.qual_rating,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_library() -> dict:
    df_raw = load_clearinghouse()
    notes    = load_curation_notes()

    treatments: list[dict] = []
    
    mode_filters = {
        "bike": "Bike|Bicycle|Pedalcyclist",
        "ped": "Pedestrian|Ped",
        "vehicle": "Vehicle|All"
    }
    
    all_curated = BIKE_CURATED + PED_CURATED + VEHICLE_CURATED

    for entry in all_curated:
        universe = filter_intersection_universe(df_raw, mode_filters[entry.applies_to])
        samples = find_studies_for_cmid(universe, entry.cmid)
        if not samples:
            print(f"  [SKIP] cmid={entry.cmid} ({entry.id}): no qualifying studies in filtered universe")
            continue
        agg       = aggregate_studies(samples)
        resolved  = resolve_recommendation(entry, agg)
        treatments.append(assemble_library_entry(entry, resolved, agg, samples, notes))
        print(
            f"  [OK] {entry.id:42s} cmid={entry.cmid:5d}  studies={agg.n_studies:2d}  "
            f"cmf={resolved.cmf:.3f} (CI {resolved.cmf_ci_low:.3f}–{resolved.cmf_ci_high:.3f})  "
            f"se={agg.se_method}  direction={resolved.direction}"
        )

    return {
        "schema_version":      2,
        "source":              "FHWA CMF Clearinghouse",
        "source_export_date":  SOURCE_DATE,
        "generated_at":        date.today().isoformat(),
        "filter": {
            "crash_type":           "Filtered per mode (bike/ped/vehicle)",
            "intersection_related": True,
            "approved":             True,
            "exclude_rural":        True,
        },
        "methodology": {
            "se_fallback":      "adjStanErrorAmf → unAdjStanErrorAmf → across-study variance",
            "aggregation":      "Variance-weighted average ONLY when every study in the group reports an SE. When SE coverage is uneven, falls back to a simple mean across all studies with across-study sample variance for the CI — avoids over-weighting whichever paper happened to report SEs.",
            "ci_level":         0.90,
            "ci_z_score":       Z_90,
            "anti_indications": "Treatments with studied AMF > 1 are kept and labeled — the studied direction is NOT a recommendation; see each entry's `direction` and `notes`.",
        },
        "treatments": treatments,
    }


def main() -> None:
    library = build_library()
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(library, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(library['treatments'])} treatments → {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
