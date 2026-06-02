'use client';

import { useState, useEffect, useMemo } from 'react';
import dynamic from 'next/dynamic';
import type { FeatureCollection } from 'geojson';
import IntersectionReport from '@/components/IntersectionReport';
import LeftPanel from '@/components/LeftPanel';
import { fetchIntersections } from '@/lib/api';
import { ALL_TIERS, type RiskTier, type SelectedIntersection, type ScorecardStats } from '@/lib/types';

const TrafficMap = dynamic(() => import('@/components/Map'), { ssr: false });

export default function HomePage() {
  const [intersections, setIntersections] = useState<FeatureCollection | null>(null);
  const [selected, setSelected] = useState<SelectedIntersection | null>(null);
  const [activeTiers, setActiveTiers] = useState<RiskTier[]>([...ALL_TIERS]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchIntersections()
      .then(setIntersections)
      .catch(() => setError('Cannot reach the API server.'))
      .finally(() => setLoading(false));
  }, []);

  const toggleTier = (tier: RiskTier) =>
    setActiveTiers(prev =>
      prev.includes(tier) ? prev.filter(t => t !== tier) : [...prev, tier]
    );

  const scorecard = useMemo<ScorecardStats | null>(() => {
    if (!intersections) return null;
    const features = intersections.features.filter(f =>
      activeTiers.includes(f.properties?.risk_tier as RiskTier)
    );
    const sum = (key: string) => features.reduce((acc, f) => acc + (Number(f.properties?.[key]) || 0), 0);
    return {
      count:    features.length,
      crashes:  sum('observed_crashes'),
      injuries: sum('injury_total'),
      ksi:      sum('ksi_total'),
      fatal:    sum('fatal_total'),
      ped:      sum('ped_total'),
      bike:     sum('bike_total'),
    };
  }, [intersections, activeTiers]);

  if (loading) {
    return (
      <div className="layout-container" style={{ alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '12px' }}>
          <div className="spinner" />
          <p className="text-muted text-sm">Loading intersection data…</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="layout-container" style={{ alignItems: 'center', justifyContent: 'center' }}>
        <div className="glass-panel" style={{ padding: '32px', maxWidth: '480px', margin: '16px' }}>
          <h3 style={{ color: 'var(--accent-danger)', marginBottom: '8px', fontSize: '1.2rem' }}>Backend not connected</h3>
          <p className="text-muted text-sm" style={{ marginBottom: '16px' }}>{error}</p>
          <pre style={{ backgroundColor: 'rgba(0,0,0,0.5)', padding: '12px', borderRadius: '8px', color: 'var(--accent-cyan)', fontSize: '0.8rem', overflowX: 'auto' }}>
            python -m uvicorn api_server:app --port 8000 --reload
          </pre>
        </div>
      </div>
    );
  }

  return (
    <main className="layout-container">
      <div className="map-container">
        <TrafficMap
          intersections={intersections}
          activeTiers={activeTiers}
          onTierToggle={toggleTier}
          onIntersectionClick={setSelected}
        />
      </div>

      {/* Floating Left Panel */}
      <div style={{ position: 'absolute', top: '24px', left: '24px', zIndex: 10 }}>
        <LeftPanel
          stats={scorecard}
          totalIntersections={intersections?.features.length ?? 0}
          activeTiers={activeTiers}
          onTierToggle={toggleTier}
        />
      </div>

      {/* Full-page intersection report */}
      {selected && (
        <IntersectionReport intersection={selected} onClose={() => setSelected(null)} />
      )}
    </main>
  );
}
