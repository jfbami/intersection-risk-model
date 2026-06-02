"""
Assemble the feature matrix for the Seattle intersection risk model.

Joins infrastructure data to the 651 Capitol Hill intersections to produce
one row per intersection with all predictors. NaNs are left in place — the
model script handles imputation.

Inputs
------
data/intermediate/intersections.parquet   — built by build_intersections.py
data/raw/streets.geojson                  — downloaded by seattle_arcgis.py
data/raw/traffic_signals.geojson          — downloaded by seattle_arcgis.py
data/raw/bike_facilities.geojson          — downloaded by seattle_arcgis.py
data/raw/aadt.geojson                     — optional; NaN column if absent

Output
------
data/intermediate/intersection_features.parquet
"""

import sys
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INTERSECTIONS_PATH = ROOT / "data" / "intermediate" / "intersections.parquet"
STREETS_PATH       = ROOT / "data" / "raw" / "streets.geojson"
SIGNALS_PATH       = ROOT / "data" / "raw" / "traffic_signals.geojson"
BIKE_PATH          = ROOT / "data" / "raw" / "bike_facilities.geojson"
AADT_PATH          = ROOT / "data" / "raw" / "aadt.geojson"
BIKE_EXPOSURE_PATH = ROOT / "data" / "intermediate" / "bike_exposure.parquet"
OUT_PATH           = ROOT / "data" / "intermediate" / "intersection_features.parquet"

UTM = "EPSG:32610"

SIGNAL_SNAP_M = 25.0
BIKE_SNAP_M   = 15.0
AADT_SNAP_M   = 30.0


# ---------------------------------------------------------------------------
# Geometry helper (mirrors snap_crashes.py)
# ---------------------------------------------------------------------------

def _normalize_geometry(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if "geometry" not in gdf.columns and "geom" in gdf.columns:
        gdf = gdf.rename_geometry("geometry")
    return gdf.set_geometry("geometry")


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_inputs() -> tuple[
    gpd.GeoDataFrame,
    gpd.GeoDataFrame,
    gpd.GeoDataFrame,
    gpd.GeoDataFrame,
    Optional[gpd.GeoDataFrame],
]:
    """Returns (intersections, streets, signals, bike, aadt). aadt is None if absent."""
    _exit_if_required_missing()

    intersections = _load_layer(INTERSECTIONS_PATH)
    streets       = _load_layer(STREETS_PATH)
    signals       = _load_layer(SIGNALS_PATH)
    bike          = _load_layer(BIKE_PATH)
    aadt          = _load_layer(AADT_PATH) if AADT_PATH.exists() else None
    return intersections, streets, signals, bike, aadt


def _exit_if_required_missing() -> None:
    required = (
        (INTERSECTIONS_PATH, "python -m pipeline.build_intersections"),
        (STREETS_PATH,       "python seattle_arcgis.py"),
        (SIGNALS_PATH,       "python seattle_arcgis.py"),
        (BIKE_PATH,          "python seattle_arcgis.py"),
    )
    missing = [f"  {path}  →  run: {cmd}" for path, cmd in required if not path.exists()]
    if missing:
        sys.exit("[ERROR] Missing required inputs:\n" + "\n".join(missing))


def _load_layer(path: Path) -> gpd.GeoDataFrame:
    if path.suffix == ".parquet":
        return _normalize_geometry(gpd.read_parquet(path)).to_crs(UTM)
    return _normalize_geometry(gpd.read_file(path)).to_crs(UTM)


# ---------------------------------------------------------------------------
# Schema inspection
# ---------------------------------------------------------------------------

def inspect_schema(
    streets: gpd.GeoDataFrame,
    signals: gpd.GeoDataFrame,
    bike: gpd.GeoDataFrame,
    aadt: Optional[gpd.GeoDataFrame],
) -> tuple[str, str, Optional[str], Optional[str]]:
    """Print column names + key distributions; return (compkey_col, artclass_col, aadt_count_col, aadt_year_col)."""
    _print_layer_summary(streets, "streets")

    if "COMPKEY" not in streets.columns:
        raise RuntimeError("COMPKEY column missing from streets.geojson. Re-download via seattle_arcgis.py.")
    compkey_col = "COMPKEY"

    _print_streets_distributions(streets)
    _print_layer_summary(signals, "traffic_signals")
    _print_layer_summary(bike, "bike_facilities")

    artclass_col = _find_artclass_column(streets)
    aadt_count_col, aadt_year_col = _inspect_aadt_layer(aadt)
    return compkey_col, artclass_col, aadt_count_col, aadt_year_col


def _print_layer_summary(gdf: gpd.GeoDataFrame, label: str) -> None:
    non_geom = gdf.drop(columns=gdf.geometry.name)
    print(f"\n=== {label} schema ===")
    print(non_geom.dtypes.to_string())
    print(f"\n--- {label}: first 3 rows ---")
    print(non_geom.head(3).to_string())


def _print_streets_distributions(streets: gpd.GeoDataFrame) -> None:
    for field in ("SPEEDLIMIT", "ARTCLASS", "ARTDESCRIPT"):
        if field in streets.columns:
            print(f"\n--- streets '{field}' distribution ---")
            print(streets[field].value_counts(dropna=False).sort_index().to_string())
        else:
            print(f"\n[WARN] streets: '{field}' column not found.")


def _find_artclass_column(streets: gpd.GeoDataFrame) -> str:
    candidates = [c for c in streets.columns if "artclass" in c.lower()]
    if not candidates:
        raise RuntimeError(
            "No ARTCLASS-like column found in streets.geojson. "
            "Inspect schema and update artclass_col manually."
        )
    return candidates[0]


# Seattle's GIS layer publishes AWDT (Annual Weekday Daily Traffic), not AADT.
# AWDT is the standard local proxy — same annual factoring, weekday-only basis.
# AWDT_ROUND is the published-rounded variant; ADT is over the study period only.
# Preference order is used below before falling back to a generic "AADT" match
# so that other cities' data still work.
SEATTLE_VOLUME_PREFERENCE: tuple[str, ...] = ("AWDT", "AWDT_ROUND", "ADT")


def _inspect_aadt_layer(aadt: Optional[gpd.GeoDataFrame]) -> tuple[Optional[str], Optional[str]]:
    if aadt is None:
        return None, None
    _print_layer_summary(aadt, "aadt")

    aadt_count_col = _find_volume_column(aadt)
    aadt_year_col  = _find_year_column(aadt)

    if aadt_count_col is None:
        print("[WARN] AADT: no count field found — aadt feature will be all-NaN.")
    else:
        print(f"\n[INFO] Volume count field: '{aadt_count_col}' "
              f"(Seattle AWDT is used as the AADT proxy)")
    if aadt_year_col is None:
        print("[WARN] AADT: no year field found — will use all rows (no year filter).")
    else:
        print(f"[INFO] AADT year field:  '{aadt_year_col}'")
        print(f"       year range: {aadt[aadt_year_col].min()} – {aadt[aadt_year_col].max()}")
    return aadt_count_col, aadt_year_col


def _find_volume_column(aadt: gpd.GeoDataFrame) -> Optional[str]:
    for preferred in SEATTLE_VOLUME_PREFERENCE:
        if preferred in aadt.columns:
            return preferred
    generic = [
        c for c in aadt.columns
        if c.upper() in ("AADT", "COUNTAADT", "AADT_COUNT", "COUNT") or "aadt" in c.lower()
    ]
    return generic[0] if generic else None


def _find_year_column(aadt: gpd.GeoDataFrame) -> Optional[str]:
    candidates = [c for c in aadt.columns if "year" in c.lower() or c.upper() == "YEAR"]
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# Feature functions
# ---------------------------------------------------------------------------

def add_signal_feature(
    intersections: gpd.GeoDataFrame,
    signals: gpd.GeoDataFrame,
) -> pd.Series:
    """is_signalized (0/1): a signal point within SIGNAL_SNAP_M of the intersection."""
    joined = gpd.sjoin_nearest(
        intersections[["intersection_id", "geometry"]],
        signals[["geometry"]],
        how="left",
        max_distance=SIGNAL_SNAP_M,
        distance_col="_sig_dist",
    )
    signalized = joined.groupby("intersection_id")["_sig_dist"].min().notna()
    return signalized.astype(int).rename("is_signalized")


def add_speed_feature(
    intersections: gpd.GeoDataFrame,
    streets: gpd.GeoDataFrame,
    compkey_col: str,
) -> pd.Series:
    """max_speed_limit: max SPEEDLIMIT across the intersection's connected streets."""
    if "SPEEDLIMIT" not in streets.columns:
        return pd.Series(float("nan"), index=intersections["intersection_id"], name="max_speed_limit")

    speed_lookup = (
        streets[[compkey_col, "SPEEDLIMIT"]]
        .dropna(subset=["SPEEDLIMIT"])
        .set_index(compkey_col)["SPEEDLIMIT"]
        .to_dict()
    )

    def _max_speed(compkeys: list) -> float:
        speeds = [speed_lookup[k] for k in compkeys if k in speed_lookup]
        return max(speeds) if speeds else float("nan")

    return (
        intersections.set_index("intersection_id")["connected_street_ids"]
        .apply(_max_speed)
        .rename("max_speed_limit")
    )


def add_arterial_feature(
    intersections: gpd.GeoDataFrame,
    streets: gpd.GeoDataFrame,
    compkey_col: str,
    artclass_col: str,
) -> pd.DataFrame:
    """is_arterial (0/1) and arterial_class (int) from max ARTCLASS on connected streets."""
    class_lookup = (
        streets[[compkey_col, artclass_col]]
        .dropna(subset=[artclass_col])
        .set_index(compkey_col)[artclass_col]
        .to_dict()
    )

    def _max_class(compkeys: list) -> int:
        classes = [class_lookup[k] for k in compkeys if k in class_lookup]
        return int(max(classes)) if classes else 0

    arterial_class = (
        intersections.set_index("intersection_id")["connected_street_ids"]
        .apply(_max_class)
        .rename("arterial_class")
    )
    is_arterial = (arterial_class >= 1).astype(int).rename("is_arterial")
    return pd.DataFrame({"is_arterial": is_arterial, "arterial_class": arterial_class})


def add_bike_feature(
    intersections: gpd.GeoDataFrame,
    bike: gpd.GeoDataFrame,
) -> pd.Series:
    """bike_facility (0/1): any bike facility within BIKE_SNAP_M of the intersection."""
    joined = gpd.sjoin_nearest(
        intersections[["intersection_id", "geometry"]],
        bike[["geometry"]],
        how="left",
        max_distance=BIKE_SNAP_M,
        distance_col="_bike_dist",
    )
    has_bike = joined.groupby("intersection_id")["_bike_dist"].min().notna()
    return has_bike.astype(int).rename("bike_facility")


def add_aadt_feature(
    intersections: gpd.GeoDataFrame,
    aadt: Optional[gpd.GeoDataFrame],
    aadt_count_col: Optional[str],
    aadt_year_col: Optional[str],
) -> pd.Series:
    """max_aadt (float): max AADT among segments within AADT_SNAP_M, latest year per segment."""
    nan_series = pd.Series(float("nan"), index=intersections["intersection_id"], name="max_aadt")
    if aadt is None or aadt_count_col is None:
        return nan_series

    aadt_clean = _prepare_aadt_layer(aadt, aadt_count_col, aadt_year_col)
    if aadt_clean.empty:
        return nan_series

    joined = gpd.sjoin_nearest(
        intersections[["intersection_id", "geometry"]],
        aadt_clean[["geometry", aadt_count_col]],
        how="left",
        max_distance=AADT_SNAP_M,
        distance_col="_aadt_dist",
    )
    return joined.groupby("intersection_id")[aadt_count_col].max().rename("max_aadt")


def _prepare_aadt_layer(
    aadt: gpd.GeoDataFrame,
    aadt_count_col: str,
    aadt_year_col: Optional[str],
) -> gpd.GeoDataFrame:
    if aadt_year_col is not None:
        aadt = (
            aadt.sort_values(aadt_year_col, ascending=False)
            .drop_duplicates(subset=[aadt.geometry.name])
        )
    df = aadt[[aadt_count_col, "geometry"]].dropna(subset=[aadt_count_col]).copy()
    df[aadt_count_col] = pd.to_numeric(df[aadt_count_col], errors="coerce")
    return df.dropna(subset=[aadt_count_col])


def add_bike_exposure_feature(intersections: gpd.GeoDataFrame) -> pd.Series:
    nan_series = pd.Series(float("nan"), index=intersections["intersection_id"], name="bike_centrality")
    if not BIKE_EXPOSURE_PATH.exists():
        print(f"[WARN] {BIKE_EXPOSURE_PATH.name} not found. bike_centrality will be NaN.")
        return nan_series
    exposure = pd.read_parquet(BIKE_EXPOSURE_PATH)
    merged = intersections[["intersection_id"]].merge(exposure, on="intersection_id", how="left")
    return merged.set_index("intersection_id")["bike_centrality"]


# ---------------------------------------------------------------------------
# Feature table assembly
# ---------------------------------------------------------------------------

def build_feature_table(
    intersections: gpd.GeoDataFrame,
    streets: gpd.GeoDataFrame,
    signals: gpd.GeoDataFrame,
    bike: gpd.GeoDataFrame,
    aadt: Optional[gpd.GeoDataFrame],
    compkey_col: str,
    artclass_col: str,
    aadt_count_col: Optional[str],
    aadt_year_col: Optional[str],
) -> pd.DataFrame:
    """Build the 651-row feature DataFrame ready to write to parquet."""
    features = intersections[["intersection_id", "num_legs"]].copy()

    feature_columns = [
        add_signal_feature(intersections, signals),
        add_speed_feature(intersections, streets, compkey_col),
        add_arterial_feature(intersections, streets, compkey_col, artclass_col),
        add_bike_feature(intersections, bike),
        add_aadt_feature(intersections, aadt, aadt_count_col, aadt_year_col),
        add_bike_exposure_feature(intersections),
    ]
    for fc in feature_columns:
        features = features.merge(fc.reset_index(), on="intersection_id", how="left")

    return _cast_binary_columns(features)


def _cast_binary_columns(features: pd.DataFrame) -> pd.DataFrame:
    for col in ("is_signalized", "is_arterial", "bike_facility"):
        features[col] = features[col].fillna(0).astype(int)
    features["arterial_class"] = features["arterial_class"].fillna(0).astype(int)
    return features


# ---------------------------------------------------------------------------
# Coverage report
# ---------------------------------------------------------------------------

def print_coverage_report(features: pd.DataFrame) -> None:
    print("=== Feature coverage report ===")
    print(f"{'Column':<22} {'non-null':>8} {'null':>6}   summary")
    print("-" * 60)
    for col in ("num_legs", "is_signalized", "max_speed_limit",
                "is_arterial", "arterial_class", "bike_facility", "max_aadt", "bike_centrality"):
        _print_coverage_row(features, col)
    _print_coverage_highlights(features)
    print("\narterial_class value_counts:")
    print(features["arterial_class"].value_counts().sort_index().to_string())
    print("\nFirst 5 rows:")
    print(features.head().to_string(index=False))


def _print_coverage_row(features: pd.DataFrame, col: str) -> None:
    s = features[col]
    n_nn = int(s.notna().sum())
    n_na = int(s.isna().sum())
    summary = _column_summary(features, col)
    print(f"  {col:<20} {n_nn:>8} {n_na:>6}   {summary}")


def _column_summary(features: pd.DataFrame, col: str) -> str:
    s = features[col]
    if col in ("is_signalized", "is_arterial", "bike_facility"):
        return f"0={int((s==0).sum())}  1={int((s==1).sum())}"
    if col == "arterial_class":
        return f"max={int(s.max())}  vc below"
    if int(s.notna().sum()):
        return f"min={s.min():.0f}  med={s.median():.0f}  max={s.max():.0f}"
    return "all NaN"


def _print_coverage_highlights(features: pd.DataFrame) -> None:
    n = len(features)
    print(f"\nis_signalized == 1: {int((features['is_signalized']==1).sum())} of {n}  "
          "(expect ~10–25% if signal join is working)")
    print(f"non-null max_speed_limit: {int(features['max_speed_limit'].notna().sum())} of {n}  "
          "(expect near 651; gaps = broken COMPKEY join)")
    print(f"non-null max_aadt:        {int(features['max_aadt'].notna().sum())} of {n}  "
          "(AADT is sparse; 20–50% is acceptable)")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def main() -> None:
    intersections, streets, signals, bike, aadt = load_inputs()
    if aadt is None:
        print(f"[WARN] {AADT_PATH} not found — max_aadt will be all-NaN.")

    print("\n" + "=" * 60)
    compkey_col, artclass_col, aadt_count_col, aadt_year_col = inspect_schema(
        streets, signals, bike, aadt
    )
    print("=" * 60 + "\n")

    features = build_feature_table(
        intersections, streets, signals, bike, aadt,
        compkey_col, artclass_col, aadt_count_col, aadt_year_col,
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUT_PATH, index=False)
    print(f"Wrote {len(features)} rows → {OUT_PATH}\n")

    print_coverage_report(features)


if __name__ == "__main__":
    main()
