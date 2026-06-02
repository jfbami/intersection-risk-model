"""
Snap SDOT collision records to Capitol Hill intersections and produce the
target variable for the Negative Binomial crash-frequency model.

Inputs
------
data/intermediate/intersections.parquet  — built by build_intersections.py
data/raw/collisions.geojson              — downloaded by seattle_arcgis.py

Outputs
-------
data/intermediate/crashes_by_intersection_year.parquet
    One row per (intersection_id, year) for every intersection × 2018-2023.
    Zero-crash rows are included — they are required training data.

data/intermediate/crashes_by_intersection.parquet
    One row per intersection with total_crashes, years_observed, and severity
    sub-counts (injury_total, ksi_total, fatal_total, ped_total, bike_total).
"""

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INTERSECTIONS_PATH = ROOT / "data" / "intermediate" / "intersections.parquet"
COLLISIONS_PATH    = ROOT / "data" / "raw" / "collisions.geojson"
OUT_DIR            = ROOT / "data" / "intermediate"

UTM   = "EPSG:32610"

YEAR_MIN = 2018
YEAR_MAX = 2023
SNAP_DISTANCE_M = 25.0

# Exact-match allowlist; substring matching would wrongly keep
# "At Intersection (but not related to intersection)".
JUNCTION_TYPES_AT_INTERSECTION: frozenset[str] = frozenset({
    "At Intersection (intersection related)",
    "Mid-Block (but intersection related)",
})

# SDOT changed crash-modality encoding around 2018:
#   PEDCOUNT / PEDCYLCOUNT — structured counts, populated pre-2018 only.
#   SDOT_COLDESC — free text, populated across the whole range. Post-2018
#       ped/bike crashes are encoded only here with the keywords below.
PED_KEYWORD  = "PEDESTRIAN"
BIKE_KEYWORD = "PEDALCYCLIST"

SEVERITY_RAW_COLS = ("MAXSEVERITYCODE", "PEDCOUNT", "PEDCYLCOUNT", "SDOT_COLDESC")


# ---------------------------------------------------------------------------
# Mode inference (extracted for unit-test coverage)
# ---------------------------------------------------------------------------

def infer_modes_from_description(text: str) -> set[str]:
    """Modes encoded in an SDOT_COLDESC string. Returns subset of {"ped", "bike"}."""
    if not text:
        return set()
    upper = text.upper()
    modes: set[str] = set()
    if PED_KEYWORD in upper:
        modes.add("ped")
    if BIKE_KEYWORD in upper:
        modes.add("bike")
    return modes


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def _normalize_geometry(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if "geometry" not in gdf.columns and "geom" in gdf.columns:
        gdf = gdf.rename_geometry("geometry")
    return gdf.set_geometry("geometry")


def load_inputs() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load intersections and collisions; normalize geometry; reproject to UTM."""
    missing = []
    if not INTERSECTIONS_PATH.exists():
        missing.append(f"  {INTERSECTIONS_PATH}  →  run: python -m pipeline.build_intersections")
    if not COLLISIONS_PATH.exists():
        missing.append(f"  {COLLISIONS_PATH}  →  run: python seattle_arcgis.py")
    if missing:
        sys.exit("[ERROR] Missing required inputs:\n" + "\n".join(missing))

    intersections = _normalize_geometry(gpd.read_parquet(INTERSECTIONS_PATH)).to_crs(UTM)
    collisions    = _normalize_geometry(gpd.read_file(COLLISIONS_PATH)).to_crs(UTM)
    return intersections, collisions


# ---------------------------------------------------------------------------
# Schema inspection
# ---------------------------------------------------------------------------

def inspect_collisions_schema(collisions: gpd.GeoDataFrame) -> tuple[str, str]:
    """Print column info and return (junction_col, date_col)."""
    non_geom = collisions.drop(columns=collisions.geometry.name)
    print("=== Collisions schema ===")
    print(non_geom.dtypes.to_string())
    print("\n--- First 3 rows ---")
    print(non_geom.head(3).to_string())

    junction_col = _find_junction_column(collisions)
    print(f"\n--- Junction field: '{junction_col}' unique values ---")
    print(collisions[junction_col].value_counts(dropna=False).to_string())

    _print_severity_field_if_present(collisions)

    date_col = _find_date_column(collisions)
    print(f"\n[INFO] Using date column: '{date_col}', junction column: '{junction_col}'")
    return junction_col, date_col


def _find_junction_column(collisions: gpd.GeoDataFrame) -> str:
    candidates = [c for c in collisions.columns if "junction" in c.lower()]
    if not candidates:
        raise RuntimeError(
            "No junction-type column found in collisions. "
            "Cannot filter intersection-related crashes — stopping."
        )
    return candidates[0]


def _print_severity_field_if_present(collisions: gpd.GeoDataFrame) -> None:
    candidates = [c for c in collisions.columns if "severity" in c.lower()]
    if not candidates:
        print("\n[INFO] No severity field found (not needed for MVP).")
        return
    severity_col = candidates[0]
    print(f"\n--- Severity field: '{severity_col}' unique values ---")
    print(collisions[severity_col].value_counts(dropna=False).to_string())


def _find_date_column(collisions: gpd.GeoDataFrame) -> str:
    if "INCDTTM" in collisions.columns:
        return "INCDTTM"
    if "INCDATE" in collisions.columns:
        return "INCDATE"
    other = [c for c in collisions.columns if "date" in c.lower()]
    if not other:
        raise RuntimeError("No date column found in collisions. Cannot filter by year — stopping.")
    return other[0]


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------

def filter_crashes(
    collisions: gpd.GeoDataFrame,
    junction_col: str,
    date_col: str,
) -> tuple[gpd.GeoDataFrame, dict]:
    """Pipeline: junction-null guard → parse dates → allowlist → date-range."""
    df = collisions.copy()
    _assert_junction_field_populated(df, junction_col)

    parsed = _parse_dates(df, date_col)
    df["_year"] = parsed.dt.year

    _log_dropped_junction_values(df, junction_col)
    df = df[df[junction_col].isin(JUNCTION_TYPES_AT_INTERSECTION)].copy()
    n_after_junction = len(df)

    df = df[df["_year"].between(YEAR_MIN, YEAR_MAX)].copy()
    n_after_date = len(df)

    valid_years = parsed.dt.year.dropna()
    stats = {
        "nat_count":        int(parsed.isna().sum()),
        "year_min_obs":     int(valid_years.min()) if len(valid_years) else None,
        "year_max_obs":     int(valid_years.max()) if len(valid_years) else None,
        "n_after_junction": n_after_junction,
        "n_after_date":     n_after_date,
    }
    return df, stats


def _assert_junction_field_populated(df: gpd.GeoDataFrame, junction_col: str) -> None:
    if df[junction_col].isna().all():
        sys.exit(
            f"[ERROR] Junction field '{junction_col}' is entirely null in the raw "
            "collisions layer. Cannot identify intersection crashes — stopping."
        )


def _parse_dates(df: gpd.GeoDataFrame, date_col: str) -> pd.Series:
    """INCDTTM is a date string; INCDATE is int64 milliseconds since epoch."""
    if date_col == "INCDATE":
        return pd.to_datetime(df[date_col], unit="ms", errors="coerce")
    return pd.to_datetime(df[date_col], errors="coerce")


def _log_dropped_junction_values(df: gpd.GeoDataFrame, junction_col: str) -> None:
    """Surface junction values being filtered out so silent-data-drift is visible."""
    dropped = df.loc[~df[junction_col].isin(JUNCTION_TYPES_AT_INTERSECTION), junction_col]
    if dropped.empty:
        return
    counts = dropped.value_counts(dropna=False)
    print(f"\n--- Junction values being dropped ({len(dropped)} rows) ---")
    print(counts.to_string())


# ---------------------------------------------------------------------------
# Snap
# ---------------------------------------------------------------------------

def snap_to_intersections(
    collisions: gpd.GeoDataFrame,
    intersections: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Spatial-join each crash to the nearest intersection within SNAP_DISTANCE_M."""
    _assert_utm_geometry(collisions, "collisions")
    _assert_utm_geometry(intersections, "intersections")

    keep = ["geometry", "_year"] + [c for c in SEVERITY_RAW_COLS if c in collisions.columns]
    return gpd.sjoin_nearest(
        collisions[keep],
        intersections[["intersection_id", "geometry"]],
        how="left",
        max_distance=SNAP_DISTANCE_M,
        distance_col="_snap_dist",
    )


def _assert_utm_geometry(gdf: gpd.GeoDataFrame, label: str) -> None:
    crs = gdf.crs.to_epsg() if gdf.crs else None
    print(f"  CRS check — {label}: EPSG:{crs}")
    if crs != 32610:
        sys.exit(
            f"[ERROR] {label} CRS is EPSG:{crs}, expected EPSG:32610. "
            "Snap aborted — distances would be meaningless."
        )
    if gdf.geometry.is_empty.all():
        sys.exit(f"[ERROR] {label} active geometry column is entirely empty.")


# ---------------------------------------------------------------------------
# Build output tables
# ---------------------------------------------------------------------------

def build_target_tables(
    snapped: gpd.GeoDataFrame,
    intersections: gpd.GeoDataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(crashes_by_year_grid, crashes_by_intersection_with_severity)."""
    matched = snapped.dropna(subset=["intersection_id"]).copy()
    matched["year"] = matched["_year"].astype(int)

    crashes_by_year = _build_crashes_by_year_grid(matched, intersections)
    crashes_by_intersection = _build_per_intersection_summary(crashes_by_year, matched)
    return crashes_by_year, crashes_by_intersection


def _build_crashes_by_year_grid(
    matched: pd.DataFrame,
    intersections: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Every intersection × every year, zero-filled."""
    counts = (
        matched.groupby(["intersection_id", "year"])
        .size()
        .reset_index(name="crash_count")
    )
    all_years = list(range(YEAR_MIN, YEAR_MAX + 1))
    grid = pd.MultiIndex.from_product(
        [intersections["intersection_id"].unique(), all_years],
        names=["intersection_id", "year"],
    ).to_frame(index=False)
    grid = grid.merge(counts, on=["intersection_id", "year"], how="left")
    grid["crash_count"] = grid["crash_count"].fillna(0).astype(int)
    return grid


def _build_per_intersection_summary(
    crashes_by_year: pd.DataFrame,
    matched: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        crashes_by_year.groupby("intersection_id")["crash_count"]
        .sum()
        .reset_index(name="total_crashes")
    )
    summary["years_observed"] = YEAR_MAX - YEAR_MIN + 1

    severity = _aggregate_severity_per_intersection(matched)
    summary = summary.merge(severity, on="intersection_id", how="left")
    severity_cols = [
        "injury_total", "ksi_total", "fatal_total",
        "ped_total", "bike_total", "vehicle_only_total",
        "bike_ksi_total", "ped_ksi_total", "vehicle_only_ksi_total",
    ]
    for col in severity_cols:
        summary[col] = summary[col].fillna(0).astype(int)
    return summary


def _aggregate_severity_per_intersection(matched: pd.DataFrame) -> pd.DataFrame:
    severity     = _severity_flags(matched)
    modes        = _mode_flags(matched)
    bike_ksi     = (modes["bike_total"] & severity["ksi_total"]).astype(int).rename("bike_ksi_total")
    ped_ksi      = (modes["ped_total"]  & severity["ksi_total"]).astype(int).rename("ped_ksi_total")
    # vehicle_only = neither a bike nor a ped was involved. Computed at the
    # per-crash level so bike-ped overlap crashes (rare but real) are
    # correctly excluded from the vehicle-only count.
    vehicle_only     = ((~modes["bike_total"].astype(bool))
                        & (~modes["ped_total"].astype(bool))).astype(int).rename("vehicle_only_total")
    vehicle_only_ksi = (vehicle_only & severity["ksi_total"]).astype(int).rename("vehicle_only_ksi_total")
    flags = pd.concat(
        [pd.DataFrame({"intersection_id": matched["intersection_id"]}),
         severity, modes, bike_ksi.to_frame(), ped_ksi.to_frame(),
         vehicle_only.to_frame(), vehicle_only_ksi.to_frame()],
        axis=1,
    )
    cols = [
        "injury_total", "ksi_total", "fatal_total",
        "ped_total", "bike_total", "vehicle_only_total",
        "bike_ksi_total", "ped_ksi_total", "vehicle_only_ksi_total",
    ]
    return flags.groupby("intersection_id")[cols].sum().reset_index()


def _severity_flags(matched: pd.DataFrame) -> pd.DataFrame:
    """Per-crash flags derived from MAXSEVERITYCODE (1=PDO 2=Injury 3=Serious 4=Fatal)."""
    if "MAXSEVERITYCODE" not in matched.columns:
        zero = pd.Series(0, index=matched.index)
        return pd.DataFrame({"injury_total": zero, "ksi_total": zero, "fatal_total": zero})
    sev = pd.to_numeric(matched["MAXSEVERITYCODE"], errors="coerce").fillna(0)
    return pd.DataFrame({
        "injury_total": (sev >= 2).astype(int),
        "ksi_total":    (sev >= 3).astype(int),
        "fatal_total":  (sev == 4).astype(int),
    })


def _mode_flags(matched: pd.DataFrame) -> pd.DataFrame:
    """Per-crash ped/bike flags: union of structured counts and SDOT_COLDESC keywords."""
    desc = (matched.get("SDOT_COLDESC", pd.Series("", index=matched.index))
            .fillna("").str.upper())
    ped_count  = (matched["PEDCOUNT"].fillna(0) > 0) \
        if "PEDCOUNT" in matched.columns else pd.Series(False, index=matched.index)
    bike_count = (matched["PEDCYLCOUNT"].fillna(0) > 0) \
        if "PEDCYLCOUNT" in matched.columns else pd.Series(False, index=matched.index)
    ped_desc  = desc.str.contains(PED_KEYWORD,  regex=False, na=False)
    bike_desc = desc.str.contains(BIKE_KEYWORD, regex=False, na=False)
    return pd.DataFrame({
        "ped_total":  (ped_count  | ped_desc).astype(int),
        "bike_total": (bike_count | bike_desc).astype(int),
    })


# ---------------------------------------------------------------------------
# Output + diagnostics
# ---------------------------------------------------------------------------

def write_outputs(crashes_by_year: pd.DataFrame, crashes_by_intersection: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    crashes_by_year.to_parquet(OUT_DIR / "crashes_by_intersection_year.parquet", index=False)
    crashes_by_intersection.to_parquet(OUT_DIR / "crashes_by_intersection.parquet", index=False)
    print(f"\nWrote crashes_by_intersection_year.parquet  ({len(crashes_by_year)} rows)")
    print(f"Wrote crashes_by_intersection.parquet       ({len(crashes_by_intersection)} rows)")


def print_filter_diagnostics(stats: dict) -> None:
    print(f"\nDate parsing: {stats['nat_count']} rows with unparseable date (NaT)")
    print(f"Parsed year range: {stats['year_min_obs']} – {stats['year_max_obs']}  (expect ~2003–2025)")
    print(f"Crashes after junction allowlist filter:    {stats['n_after_junction']}")
    print(f"Crashes after date filter ({YEAR_MIN}–{YEAR_MAX}):      {stats['n_after_date']}")


def print_snap_diagnostics(snapped: gpd.GeoDataFrame) -> None:
    n_dropped = int(snapped["intersection_id"].isna().sum())
    n_snapped = int(snapped["intersection_id"].notna().sum())
    print(f"\nCrashes dropped (>{SNAP_DISTANCE_M} m from any intersection): {n_dropped}")
    print(f"Crashes snapped to an intersection:                          {n_snapped}")


def print_sanity_check(crashes_by_intersection: pd.DataFrame) -> None:
    print("\n=== Sanity check: crashes_by_intersection ===")
    print(f"Total intersections:          {len(crashes_by_intersection)}  (expect 651)")
    print(f"Intersections with 0 crashes: {int((crashes_by_intersection['total_crashes'] == 0).sum())}")
    print(f"Mean total_crashes:           {crashes_by_intersection['total_crashes'].mean():.2f}")
    print(f"Max  total_crashes:           {crashes_by_intersection['total_crashes'].max()}")
    print("\ntotal_crashes value_counts:")
    print(crashes_by_intersection["total_crashes"].value_counts().sort_index().to_string())
    print("\nTop 10 intersections by total_crashes:")
    print(crashes_by_intersection.nlargest(10, "total_crashes").to_string(index=False))


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def main() -> None:
    intersections, collisions = load_inputs()
    print(f"Raw collisions loaded: {len(collisions)}")

    print("\n" + "=" * 60)
    junction_col, date_col = inspect_collisions_schema(collisions)
    print("=" * 60 + "\n")

    print("--- Junction type distribution BEFORE filter ---")
    print(collisions[junction_col].value_counts(dropna=False).to_string())

    filtered, stats = filter_crashes(collisions, junction_col, date_col)
    print_filter_diagnostics(stats)
    print("\n--- Junction type distribution AFTER filter ---")
    print(filtered[junction_col].value_counts(dropna=False).to_string())

    snapped = snap_to_intersections(filtered, intersections)
    print_snap_diagnostics(snapped)

    crashes_by_year, crashes_by_intersection = build_target_tables(snapped, intersections)
    write_outputs(crashes_by_year, crashes_by_intersection)
    print_sanity_check(crashes_by_intersection)


if __name__ == "__main__":
    main()
