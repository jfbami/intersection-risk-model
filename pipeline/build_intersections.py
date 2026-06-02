"""
Build the intersection layer for the Seattle traffic risk-modeling pipeline.

Reads street centerlines from data/raw/streets.geojson, extracts segment
endpoints, clusters nearby endpoints (≤10 m) into single intersection nodes
using a KDTree + union-find approach, and writes the result to
data/intermediate/intersections.parquet.

Run after: python pipeline/seattle_arcgis.py
Run before: any downstream feature-join script.
"""

import hashlib
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import Point

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
STREETS_PATH = ROOT / "data" / "raw" / "streets.geojson"
OUT_PATH = ROOT / "data" / "intermediate" / "intersections.parquet"

# Capitol Hill bounding box (EPSG:4326)
BBOX_4326 = {"west": -122.3320, "east": -122.3000, "south": 47.6080, "north": 47.6360}

UTM = "EPSG:32610"
WGS84 = "EPSG:4326"
CLUSTER_TOLERANCE_M = 10.0


# ---------------------------------------------------------------------------
# Step 1 – load
# ---------------------------------------------------------------------------

def load_streets() -> gpd.GeoDataFrame:
    """Load streets.geojson; reproject to UTM; filter to Capitol Hill bbox."""
    if not STREETS_PATH.exists():
        sys.exit(
            f"[ERROR] {STREETS_PATH} not found.\n"
            "Run:  python pipeline/seattle_arcgis.py\n"
            "to download the streets layer first."
        )

    gdf = gpd.read_file(STREETS_PATH)

    # Reproject first so the spatial filter is in the same CRS as all later
    # geometry ops (distances in metres via UTM).
    gdf = gdf.to_crs(UTM)

    # Transform the bbox corners to UTM for the spatial filter.
    bbox_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(
            [BBOX_4326["west"], BBOX_4326["east"]],
            [BBOX_4326["south"], BBOX_4326["north"]],
        ),
        crs=WGS84,
    ).to_crs(UTM)
    min_x, min_y = bbox_gdf.geometry.iloc[0].x, bbox_gdf.geometry.iloc[0].y
    max_x, max_y = bbox_gdf.geometry.iloc[1].x, bbox_gdf.geometry.iloc[1].y

    gdf = gdf.cx[min_x:max_x, min_y:max_y].copy()
    return gdf


# ---------------------------------------------------------------------------
# Step 2 – extract endpoints
# ---------------------------------------------------------------------------

def extract_endpoints(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Return a DataFrame with one row per LineString endpoint.

    Columns: x, y, street_id
    """
    # COMPKEY is Seattle's permanent street identifier and survives dataset
    # refreshes. OBJECTID is ArcGIS-internal and must NOT be used here.
    if "COMPKEY" not in gdf.columns:
        raise KeyError(
            "COMPKEY column not found in streets layer. "
            "Re-download streets via pipeline/seattle_arcgis.py and confirm "
            "the Seattle_Streets_1 layer includes the COMPKEY field."
        )

    rows: list[dict] = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        coords = list(geom.coords)
        for pt in (coords[0], coords[-1]):
            rows.append({"x": pt[0], "y": pt[1], "street_id": int(row["COMPKEY"])})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 3 – cluster endpoints
# ---------------------------------------------------------------------------

def cluster_endpoints(endpoints: pd.DataFrame) -> np.ndarray:
    """
    Cluster endpoints within CLUSTER_TOLERANCE_M of one another.

    Returns an integer label array (one label per endpoint row) using
    union-find on all pairs reported by cKDTree.query_pairs.
    """
    coords = endpoints[["x", "y"]].to_numpy()
    tree = cKDTree(coords)
    pairs = tree.query_pairs(CLUSTER_TOLERANCE_M)  # set of (i, j) tuples

    # Union-find
    parent = list(range(len(coords)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]  # path compression
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        parent[find(i)] = find(j)

    for i, j in pairs:
        union(i, j)

    # Normalise labels to contiguous integers
    root_map: dict[int, int] = {}
    labels = np.empty(len(coords), dtype=int)
    for idx in range(len(coords)):
        root = find(idx)
        if root not in root_map:
            root_map[root] = len(root_map)
        labels[idx] = root_map[root]

    return labels


# ---------------------------------------------------------------------------
# Step 4 – build intersections GeoDataFrame
# ---------------------------------------------------------------------------

def build_intersections(
    endpoints: pd.DataFrame,
    labels: np.ndarray,
) -> gpd.GeoDataFrame:
    """
    Aggregate clustered endpoints into one row per intersection.

    Columns: intersection_id, geom (UTM Point), num_legs,
             connected_street_ids.
    """
    endpoints = endpoints.copy()
    endpoints["cluster"] = labels

    rows: list[dict] = []
    for cluster_id, grp in endpoints.groupby("cluster"):
        cx = grp["x"].mean()
        cy = grp["y"].mean()
        street_ids = sorted(grp["street_id"].unique().tolist())
        num_legs = len(street_ids)
        iid = hashlib.md5(f"{round(cx)}_{round(cy)}".encode()).hexdigest()[:12]
        rows.append(
            {
                "intersection_id": iid,
                "geom": Point(cx, cy),
                "num_legs": num_legs,
                "connected_street_ids": street_ids,
            }
        )

    gdf = gpd.GeoDataFrame(rows, geometry="geom", crs=UTM)
    return gdf


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Load & filter
    streets = load_streets()
    print(f"Street segments after Capitol Hill filter: {len(streets)}")

    # 2. Extract endpoints
    endpoints = extract_endpoints(streets)
    print(f"Raw endpoints extracted:                   {len(endpoints)}")

    # 3. Cluster
    labels = cluster_endpoints(endpoints)

    # 4. Build intersections (in UTM)
    intersections_utm = build_intersections(endpoints, labels)
    print(f"Intersections after clustering:            {len(intersections_utm)}")

    # 5. Filter degenerate cases
    degenerate_mask = (intersections_utm["num_legs"] == 1) | (intersections_utm["num_legs"] > 8)
    high_degree = intersections_utm[intersections_utm["num_legs"] > 8]
    if len(high_degree):
        print(f"[WARN] Excluding {len(high_degree)} intersection(s) with num_legs > 8 (likely artifacts):")
        print(high_degree[["intersection_id", "num_legs"]].to_string(index=False))

    n_excluded = degenerate_mask.sum()
    intersections = intersections_utm[~degenerate_mask].copy()
    print(f"Excluded as degenerate (num_legs==1 or >8): {n_excluded}")

    # 6. Reproject to WGS-84 for storage consistency
    intersections = intersections.to_crs(WGS84)

    # 7. Write output
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    intersections.to_parquet(OUT_PATH)
    print(f"Wrote {len(intersections)} intersections → {OUT_PATH}")

    # Sanity-check summary
    print("\nnum_legs distribution:")
    print(intersections["num_legs"].value_counts().sort_index().to_string())
    all_compkeys = sorted({k for ids in intersections["connected_street_ids"] for k in ids})
    print(f"\nCOMPKEY range: min={all_compkeys[0]}, max={all_compkeys[-1]}")
    print("\nFirst 5 rows:")
    print(intersections[["intersection_id", "num_legs", "connected_street_ids"]].head().to_string(index=False))
