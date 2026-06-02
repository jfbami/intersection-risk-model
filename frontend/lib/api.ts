import type { FeatureCollection } from 'geojson';

const BASE = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`API ${res.status}: ${path}`);
  return res.json() as Promise<T>;
}

/**
 * GeoJSON FeatureCollection of Points — one per intersection.
 *
 * Property schema is the api_server.IntersectionProperties Pydantic model.
 * Headline: expected_bike_ksi_per_year (with ci_low / ci_high) and top_contributors.
 * Secondary: risk_score / risk_rank / risk_tier (percentile rank).
 */
export const fetchIntersections = () => get<FeatureCollection>('/api/intersections');

/** GeoJSON FeatureCollection of LineStrings — bike facility network. */
export const fetchBikeFacilities = () => get<FeatureCollection>('/api/bike-facilities');
