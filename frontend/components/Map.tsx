'use client';

import { useState, useCallback, useMemo, useRef, useEffect } from 'react';
import ReactMap, {
  Layer, Source, NavigationControl, ScaleControl, Popup,
  type MapRef, type MapLayerMouseEvent,
} from 'react-map-gl';
import type { FeatureCollection } from 'geojson';
import {
  type RiskTier,
  type SelectedIntersection,
  type Contributor,
  type RecommendedTreatment,
} from '@/lib/types';
import { fetchBikeFacilities } from '@/lib/api';
import 'mapbox-gl/dist/mapbox-gl.css';

interface Props {
  intersections: FeatureCollection | null;
  activeTiers: RiskTier[];
  onTierToggle: (tier: RiskTier) => void;
  onIntersectionClick: (i: SelectedIntersection | null) => void;
}

// Capitol Hill, Seattle
const INITIAL_VIEW = { longitude: -122.3149, latitude: 47.6188, zoom: 14.5 };

const TIER_LABEL: Record<RiskTier, string> = {
  very_high: 'Very High',
  high:      'High',
  moderate:  'Moderate',
  low:       'Low',
  very_low:  'Very Low',
};

const TIER_HEX: Record<RiskTier, string> = {
  very_high: '#ef4444',
  high:      '#f97316',
  moderate:  '#eab308',
  low:       '#84cc16',
  very_low:  '#22c55e',
};

interface HoverInfo {
  intersectionId: string;
  lng: number;
  lat: number;
  name: string;
  tier: RiskTier;
  speedLimit: number;
}

function HoverTooltip({ name, tier, speedLimit }: Omit<HoverInfo, 'intersectionId' | 'lng' | 'lat'>) {
  const color = TIER_HEX[tier];
  const speedLabel = speedLimit > 0 ? `${speedLimit} mph` : '—';
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6, minWidth: 160 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <span style={{ width: 8, height: 8, borderRadius: '50%', background: color, boxShadow: `0 0 6px ${color}` }} />
        <span className="text-xs uppercase font-semibold" style={{ color, letterSpacing: '0.08em' }}>
          {TIER_LABEL[tier]}
        </span>
      </div>
      {name && <div className="text-xs text-muted tabular-nums">{name}</div>}
      <div style={{ height: 1, background: 'rgba(255,255,255,0.08)', margin: '2px 0' }} />
      <div className="text-xs" style={{ display: 'flex', justifyContent: 'space-between', gap: 12 }}>
        <span className="text-muted">Speed limit</span>
        <span className="tabular-nums">{speedLabel}</span>
      </div>
    </div>
  );
}

function parseJsonList<T>(raw: unknown): T[] {
  if (Array.isArray(raw)) return raw as T[];
  if (typeof raw === 'string' && raw.length) {
    try { return JSON.parse(raw) as T[]; } catch { return []; }
  }
  return [];
}

function medianOf(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2
    ? sorted[mid]
    : (sorted[mid - 1] + sorted[mid]) / 2;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const TIER_COLOR: any = [
  'match', ['get', 'risk_tier'],
  'very_high', '#ef4444',
  'high',      '#f97316',
  'moderate',  '#eab308',
  'low',       '#84cc16',
  'very_low',  '#22c55e',
  '#6b7280',
];

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const CIRCLE_RADIUS: any = [
  'interpolate', ['linear'], ['get', 'risk_score'],
  0, 5, 50, 9, 100, 15,
];

export default function TrafficMap({ intersections, activeTiers, onTierToggle: _onTierToggle, onIntersectionClick }: Props) {
  const mapRef = useRef<MapRef>(null);
  const [cursor, setCursor] = useState<'grab' | 'pointer'>('grab');
  const [bikeLines, setBikeLines] = useState<FeatureCollection | null>(null);
  const [hovered, setHovered] = useState<HoverInfo | null>(null);

  // Load bike facility lines once
  useEffect(() => {
    fetchBikeFacilities()
      .then(setBikeLines)
      .catch(() => {}); // non-critical — map works without it
  }, []);

  // Citywide median expected_bike_ksi_per_year — anchor for "vs. median" framing.
  const citywideMedianBikeKsi = useMemo(() => {
    if (!intersections) return 0;
    return medianOf(
      intersections.features
        .map((f) => Number(f.properties?.expected_bike_ksi_per_year ?? 0))
        .filter((v) => Number.isFinite(v)),
    );
  }, [intersections]);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const tierFilter: any = activeTiers.length < 5
    ? ['in', ['get', 'risk_tier'], ['literal', activeTiers]]
    : ['has', 'risk_tier'];

  const handleClick = useCallback((e: MapLayerMouseEvent) => {
    const feature = e.features?.[0];
    if (!feature || feature.layer?.id !== 'intersection-circles') {
      onIntersectionClick(null);
      return;
    }
    const p = feature.properties ?? {};
    onIntersectionClick({
      intersectionId:         String(p.intersection_id ?? ''),
      name:                   String(p.name               ?? 'Unknown'),
      expectedBikeKsiPerYear: Number(p.expected_bike_ksi_per_year ?? 0),
      expectedBikeKsiCiLow:   Number(p.expected_bike_ksi_ci_low   ?? 0),
      expectedBikeKsiCiHigh:  Number(p.expected_bike_ksi_ci_high  ?? 0),
      citywideMedianBikeKsi,
      totalArterials:         intersections?.features.length ?? 0,
      topContributors:        parseJsonList<Contributor>(p.top_contributors),
      recommendedTreatments:  parseJsonList<RecommendedTreatment>(p.recommended_treatments),
      riskScore:              Number(p.risk_score          ?? 0),
      riskRank:               Number(p.risk_rank           ?? 0),
      riskTier:               String(p.risk_tier           ?? 'very_low') as RiskTier,
      predictedCrashes:       Number(p.predicted_crashes   ?? 0),
      ebPredicted:            Number(p.eb_predicted        ?? 0),
      observedCrashes:        Number(p.observed_crashes    ?? 0),
      yearsObserved:          Number(p.years_observed      ?? 6),
      injuryTotal:            Number(p.injury_total        ?? 0),
      ksiTotal:               Number(p.ksi_total           ?? 0),
      fatalTotal:             Number(p.fatal_total         ?? 0),
      pedTotal:               Number(p.ped_total           ?? 0),
      bikeTotal:              Number(p.bike_total          ?? 0),
      bikeKsiTotal:           Number(p.bike_ksi_total      ?? 0),
      isSignalized:           Boolean(p.is_signalized),
      numLegs:                Number(p.num_legs            ?? 4),
      maxSpeedLimit:          Number(p.max_speed_limit     ?? 0),
      bikeFacility:           String(p.bike_facility       ?? 'None'),
      arterialClass:          String(p.arterial_class      ?? ''),
      coordinates:            { lat: e.lngLat.lat, lng: e.lngLat.lng },
    });
  }, [onIntersectionClick, citywideMedianBikeKsi, intersections]);

  const handleMouseMove = useCallback((e: MapLayerMouseEvent) => {
    const feature = e.features?.[0];
    const onCircle = feature?.layer?.id === 'intersection-circles';
    setCursor(onCircle ? 'pointer' : 'grab');

    if (!onCircle || !feature) {
      setHovered(prev => (prev === null ? prev : null));
      return;
    }

    const p = feature.properties ?? {};
    const id = String(p.intersection_id ?? '');
    const [lng, lat] = (feature.geometry as { coordinates: [number, number] }).coordinates;
    setHovered(prev =>
      prev?.intersectionId === id ? prev : {
        intersectionId: id,
        lng, lat,
        name:       String(p.name ?? ''),
        tier:       String(p.risk_tier ?? 'very_low') as RiskTier,
        speedLimit: Number(p.max_speed_limit ?? 0),
      }
    );
  }, []);

  const handleMouseLeave = useCallback(() => {
    setCursor('grab');
    setHovered(null);
  }, []);

  const token = process.env.NEXT_PUBLIC_MAPBOX_TOKEN;
  if (!token) {
    return (
      <div className="layout-container" style={{ alignItems: 'center', justifyContent: 'center' }}>
        <p style={{ color: 'var(--accent-danger)', backgroundColor: 'var(--bg-panel)', padding: '8px 16px', borderRadius: '4px', border: '1px solid var(--accent-danger)' }}>
          Missing <code style={{ fontFamily: 'monospace' }}>NEXT_PUBLIC_MAPBOX_TOKEN</code> in <code style={{ fontFamily: 'monospace' }}>.env.local</code>
        </p>
      </div>
    );
  }

  return (
    <ReactMap
      ref={mapRef}
      mapboxAccessToken={token}
      initialViewState={INITIAL_VIEW}
      style={{ width: '100%', height: '100%' }}
      mapStyle="mapbox://styles/mapbox/dark-v11"
      cursor={cursor}
      interactiveLayerIds={['intersection-circles']}
      onClick={handleClick}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
    >
      <NavigationControl position="top-right" />
      <ScaleControl position="bottom-right" unit="imperial" />

      {hovered && (
        <Popup
          longitude={hovered.lng}
          latitude={hovered.lat}
          closeButton={false}
          closeOnClick={false}
          anchor="bottom"
          offset={18}
          className="risk-popup"
        >
          <HoverTooltip
            name={hovered.name}
            tier={hovered.tier}
            speedLimit={hovered.speedLimit}
          />
        </Popup>
      )}

      {/* ── Bike facility lines (real lane geometry) ─────────────────────── */}
      {bikeLines && (
        <Source id="bike-lines" type="geojson" data={bikeLines}>
          {/* Glow / casing */}
          <Layer
            id="bike-lines-glow"
            type="line"
            paint={{
              'line-color': '#06b6d4',
              'line-width': 4,
              'line-opacity': 0.15,
              'line-blur': 3,
            }}
          />
          {/* Main line */}
          <Layer
            id="bike-lines-main"
            type="line"
            paint={{
              'line-color': '#22d3ee',
              'line-width': 1.5,
              'line-opacity': 0.75,
              'line-dasharray': [2, 1.5],
            }}
          />
        </Source>
      )}

      {/* ── Intersection risk circles ─────────────────────────────────────── */}
      {intersections && (
        <Source id="intersections" type="geojson" data={intersections}>
          {/* Outer glow — simulates radial gradient */}
          <Layer
            id="intersection-glow"
            type="circle"
            filter={tierFilter}
            paint={{
              'circle-color': TIER_COLOR,
              'circle-radius': ['*', CIRCLE_RADIUS, 2.2] as any,
              'circle-opacity': 0.12,
              'circle-blur': 1,
              'circle-stroke-width': 0,
            }}
          />
          {/* Mid glow */}
          <Layer
            id="intersection-mid"
            type="circle"
            filter={tierFilter}
            paint={{
              'circle-color': TIER_COLOR,
              'circle-radius': ['*', CIRCLE_RADIUS, 1.45] as any,
              'circle-opacity': 0.22,
              'circle-blur': 0.6,
              'circle-stroke-width': 0,
            }}
          />
          {/* Core circle */}
          <Layer
            id="intersection-circles"
            type="circle"
            filter={tierFilter}
            paint={{
              'circle-color': TIER_COLOR,
              'circle-radius': CIRCLE_RADIUS,
              'circle-opacity': 0.92,
              'circle-blur': 0.08,
              'circle-stroke-width': 0,
            }}
          />
          {/* Bright centre highlight */}
          <Layer
            id="intersection-highlight"
            type="circle"
            filter={tierFilter}
            paint={{
              'circle-color': '#ffffff',
              'circle-radius': ['*', CIRCLE_RADIUS, 0.32] as any,
              'circle-opacity': 0.18,
              'circle-blur': 0.4,
              'circle-stroke-width': 0,
            }}
          />
        </Source>
      )}
    </ReactMap>
  );
}
