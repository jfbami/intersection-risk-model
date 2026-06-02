"""
Builds a bicycle exposure proxy using OpenStreetMap network centrality.

Downloads the bike network for the study area, calculates betweenness centrality,
and maps each intersection to the nearest OSM node's centrality score.
"""

import sys
import warnings
from pathlib import Path

import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
from shapely.geometry import box

# Suppress osmnx/pandas FutureWarnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
INTERSECTIONS_PATH = ROOT / "data" / "intermediate" / "intersections.parquet"
OUT_PATH = ROOT / "data" / "intermediate" / "bike_exposure.parquet"


def _load_intersections(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run build_intersections.py first.")
    return gpd.read_parquet(path)


def _get_buffered_bounding_box(gdf: gpd.GeoDataFrame, pad_degrees: float = 0.005) -> box:
    """Returns a padded bounding box for the given GeoDataFrame."""
    gdf_4326 = gdf.to_crs("EPSG:4326")
    minx, miny, maxx, maxy = gdf_4326.total_bounds
    return box(minx - pad_degrees, miny - pad_degrees, maxx + pad_degrees, maxy + pad_degrees)


def _download_bike_network(bounds: box) -> nx.MultiDiGraph:
    print("Downloading OSM bike network for bounding box...")
    graph = ox.graph_from_polygon(bounds, network_type='bike', simplify=True)
    print(f"Downloaded graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges.")
    return graph


def _compute_node_centrality(graph: nx.MultiDiGraph, target_crs) -> gpd.GeoDataFrame:
    print("Calculating betweenness centrality...")
    centrality = nx.betweenness_centrality(graph, weight='length')
    nx.set_node_attributes(graph, centrality, 'centrality')
    
    nodes_gdf = ox.graph_to_gdfs(graph, edges=False)
    return nodes_gdf.to_crs(target_crs)


def _map_centrality_to_intersections(intersections: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame) -> pd.DataFrame:
    print("Matching intersections to nearest OSM node...")
    joined = gpd.sjoin_nearest(
        intersections[["intersection_id", "geometry"]],
        nodes[["centrality", "geometry"]],
        how="left",
        distance_col="_node_dist"
    )
    joined = joined.drop_duplicates(subset="intersection_id")
    
    min_nonzero = joined.loc[joined["centrality"] > 0, "centrality"].min()
    if pd.isna(min_nonzero):
        min_nonzero = 1e-6
        
    joined["bike_centrality"] = joined["centrality"].clip(lower=min_nonzero)
    return joined[["intersection_id", "bike_centrality"]].copy()


def _save_exposure(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"\nWrote {len(df)} rows \u2192 {path}")
    print("\nSummary of bike_centrality:")
    print(df["bike_centrality"].describe().to_string())


def main():
    intersections = _load_intersections(INTERSECTIONS_PATH)
    if "geometry" not in intersections.columns and "geom" in intersections.columns:
        intersections = intersections.rename_geometry("geometry")
        
    bounds_polygon = _get_buffered_bounding_box(intersections)
    bike_graph = _download_bike_network(bounds_polygon)
    centrality_nodes = _compute_node_centrality(bike_graph, target_crs=intersections.crs)
    
    exposure_df = _map_centrality_to_intersections(intersections, centrality_nodes)
    _save_exposure(exposure_df, OUT_PATH)

if __name__ == "__main__":
    main()
