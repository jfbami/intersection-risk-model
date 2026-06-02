"""
FastAPI bridge: reads pipeline parquet output → serves GeoJSON for the Next.js frontend.

GET /api/intersections      →  GeoJSON FeatureCollection (651 intersection points)
GET /api/bike-facilities    →  GeoJSON FeatureCollection (bike lane lines)
GET /health                 →  {"status": "ok"}

Run the pipeline first, then start this server:
    python -m pipeline.build_intersections
    python -m pipeline.snap_crashes
    python -m pipeline.assemble_features
    python -m pipeline.fit_risk_model
    python -m pipeline.score_risk
    uvicorn api_server:app --port 8000 --reload
"""

import json
from pathlib import Path
from typing import Literal

import geopandas as gpd
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "intermediate"
RAW  = ROOT / "data" / "raw"

ARTERIAL_LABELS = {
    0: "Local / Non-arterial",
    1: "Principal Arterial",
    2: "Minor Arterial",
    3: "Collector Arterial",
    4: "Other Arterial",
}

BIKE_FACILITY_PRESENT = "Bike lane"
BIKE_FACILITY_ABSENT  = "None"

YEARS_OBSERVED_2018_2023 = 6

TierLabel = Literal["very_low", "low", "moderate", "high", "very_high"]


class Contributor(BaseModel):
    label: str
    pct_change: float


class RecommendedTreatment(BaseModel):
    """One CMF-based design recommendation for an intersection.

    `prevented_per_year_mean` is the expected reduction in bike-KSI per year
    if this treatment were installed; the CI is the credible interval
    propagated through both the model prediction and the CMF uncertainty.
    """
    id: str
    name: str
    prevented_per_year_mean:    float
    prevented_per_year_ci_low:  float
    prevented_per_year_ci_high: float
    cmf: float


class IntersectionProperties(BaseModel):
    """Allow-list of fields served per intersection. Adding a field here serves it;
    omitting one drops it. No comment-driven drops."""

    intersection_id: str
    name: str
    # ── Phase 1 headline metric ──────────────────────────────────
    expected_bike_ksi_per_year: float
    expected_bike_ksi_ci_low:   float
    expected_bike_ksi_ci_high:  float
    top_contributors: list[Contributor]
    # ── Phase 5 prescriptive recommendations ─────────────────────
    recommended_treatments: list[RecommendedTreatment]
    # ── Secondary percentile ranking ─────────────────────────────
    risk_score: float
    risk_rank:  int
    risk_tier:  TierLabel
    # ── Underlying model + observed counts ───────────────────────
    predicted_crashes: float
    eb_predicted: float
    observed_crashes: float
    years_observed: int
    # ── Vision Zero severity sub-counts ──────────────────────────
    injury_total: int
    ksi_total: int
    fatal_total: int
    ped_total: int
    bike_total: int
    bike_ksi_total: int
    # ── Infrastructure features ──────────────────────────────────
    is_signalized: int
    num_legs: int
    max_speed_limit: float
    bike_facility: str
    arterial_class: str


ALLOWED_PROPERTIES: tuple[str, ...] = tuple(IntersectionProperties.model_fields.keys())


app = FastAPI(title="Capitol Hill Vision Zero API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Loaders (one parquet each, single responsibility)
# ---------------------------------------------------------------------------

def load_intersection_geometry() -> gpd.GeoDataFrame:
    gdf = gpd.read_parquet(DATA / "intersections.parquet")
    if "geom" in gdf.columns and "geometry" not in gdf.columns:
        gdf = gdf.rename_geometry("geometry")
    return gdf.to_crs("EPSG:4326")


def load_scores() -> pd.DataFrame:
    scores = pd.read_parquet(DATA / "intersection_scores.parquet")
    eb_count_by_mode = ["bike_eb_count", "ped_eb_count", "vehicle_eb_count"]
    scores["eb_predicted"] = scores[eb_count_by_mode].sum(axis=1)
    return scores


def load_features() -> pd.DataFrame:
    return pd.read_parquet(DATA / "intersection_features.parquet")


def load_predictions() -> pd.DataFrame:
    predictions = pd.read_parquet(DATA / "intersection_predictions.parquet")
    actual_by_mode   = ["bike_actual", "ped_actual", "vehicle_actual"]
    expected_by_mode = ["bike_expected_total", "ped_expected_total", "vehicle_expected_total"]
    return pd.DataFrame({
        "intersection_id":   predictions["intersection_id"],
        "observed_crashes":  predictions[actual_by_mode].sum(axis=1),
        "predicted_crashes": predictions[expected_by_mode].sum(axis=1),
    })


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def merge_intersection_data(
    intersections: gpd.GeoDataFrame,
    scores: pd.DataFrame,
    features: pd.DataFrame,
    predictions: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """Inner-join on scores so we serve only intersections the model actually fit.

    The NB SPF is scoped to arterials with positive AADT (HSM Ch. 12 facility-
    type stratification). Local-access intersections are not in `scores` and
    must not be served — surfacing a zero-filled prediction would be a lie.
    """
    return (
        intersections
        .merge(scores,                                            on="intersection_id", how="inner")
        .merge(features.drop(columns=["num_legs"], errors="ignore"), on="intersection_id", how="left")
        .merge(predictions,                                       on="intersection_id", how="left")
    )


def add_display_fields(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    df = df.copy()
    df["name"] = df.apply(
        lambda r: f"{r.geometry.y:.4f}°N, {abs(r.geometry.x):.4f}°W", axis=1
    )
    df["years_observed"] = YEARS_OBSERVED_2018_2023
    df["bike_facility"]  = _label_bike_facility(df.get("bike_facility"))
    df["arterial_class"] = _label_arterial_class(df.get("arterial_class"))
    df["top_contributors"]       = _parse_json_list_column(df.get("top_contributors"))
    df["recommended_treatments"] = _parse_json_list_column(df.get("recommended_treatments"))
    return df


def _parse_json_list_column(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series([], dtype=object)
    return series.fillna("[]").apply(json.loads)


def _label_bike_facility(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(BIKE_FACILITY_ABSENT, index=[])
    return series.apply(
        lambda v: BIKE_FACILITY_ABSENT if pd.isna(v) or int(v) == 0 else BIKE_FACILITY_PRESENT
    )


def _label_arterial_class(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series("Local / Non-arterial", index=[])
    return (
        series.fillna(0).astype(int).map(ARTERIAL_LABELS).fillna("Other Arterial")
    )


def fill_property_defaults(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    numeric_defaults = {
        "expected_bike_ksi_per_year": 0.0,
        "expected_bike_ksi_ci_low":   0.0,
        "expected_bike_ksi_ci_high":  0.0,
        "risk_score":         0.0,
        "risk_rank":          0,
        "predicted_crashes":  0.0,
        "eb_predicted":       0.0,
        "observed_crashes":   0.0,
        "injury_total":       0,
        "ksi_total":          0,
        "fatal_total":        0,
        "ped_total":          0,
        "bike_total":         0,
        "bike_ksi_total":     0,
        "max_speed_limit":    0,
        "num_legs":           4,
        "is_signalized":      0,
    }
    df = df.copy()
    for col, default in numeric_defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(default)
    df["risk_tier"]     = df["risk_tier"].fillna("very_low")
    df["bike_facility"] = df["bike_facility"].fillna(BIKE_FACILITY_ABSENT)
    return df


def project_to_allowed_properties(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    keep = [c for c in ALLOWED_PROPERTIES if c in df.columns] + ["geometry"]
    return df[keep]


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

def build_intersections_geojson() -> dict:
    merged = merge_intersection_data(
        load_intersection_geometry(),
        load_scores(),
        load_features(),
        load_predictions(),
    )
    merged = add_display_fields(merged)
    merged = fill_property_defaults(merged)
    merged = project_to_allowed_properties(merged)
    return json.loads(merged.to_json())


def build_bike_facilities_geojson() -> dict:
    path = RAW / "bike_facilities.geojson"
    if not path.exists():
        raise FileNotFoundError(path)
    gdf = gpd.read_file(path).to_crs("EPSG:4326")
    gdf = gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()
    keep = [c for c in ("BIKEFACILITY", "FACILITYTYPE", "STREETNAME", "geometry") if c in gdf.columns]
    return json.loads(gdf[keep].to_json())


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/intersections")
def get_intersections() -> dict:
    try:
        return build_intersections_geojson()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Pipeline output not found: {exc}. "
                "Run the full pipeline (build_intersections → snap_crashes → "
                "assemble_features → fit_risk_model → score_risk) then restart."
            ),
        )


@app.get("/api/bike-facilities")
def get_bike_facilities() -> dict:
    try:
        return build_bike_facilities_geojson()
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="bike_facilities.geojson not found. Run: python seattle_arcgis.py",
        )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
