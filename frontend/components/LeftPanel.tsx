'use client';

import { TIER_META, ALL_TIERS, type RiskTier, type ScorecardStats } from '@/lib/types';

interface Props {
  stats: ScorecardStats | null;
  totalIntersections: number;
  activeTiers: RiskTier[];
  onTierToggle: (tier: RiskTier) => void;
}

function StatRow({ label, value, warn }: { label: string; value: number; warn?: boolean }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '4px 0' }}>
      <span className="text-xs text-muted">{label}</span>
      <span className="text-sm font-semibold" style={{ color: warn && value > 0 ? 'var(--accent-danger)' : 'var(--text-primary)', fontVariantNumeric: 'tabular-nums' }}>
        {value}
      </span>
    </div>
  );
}

export default function LeftPanel({ stats, totalIntersections, activeTiers, onTierToggle }: Props) {
  return (
    <aside className="glass-panel animate-fade-in" style={{ width: '300px', display: 'flex', flexDirection: 'column', maxHeight: 'calc(100vh - 48px)' }}>

      {/* Header */}
      <div className="panel-header">
        <p className="text-xs uppercase font-semibold text-muted">
          Capitol Hill · Seattle
        </p>
        <p className="text-base font-semibold" style={{ margin: '4px 0' }}>Vision Zero Risk Map</p>
        <p className="text-xs text-muted">2018 – 2023</p>
      </div>

      {/* Scorecard */}
      {stats && (
        <div className="panel-header" style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
          <p className="text-xs uppercase font-semibold text-muted" style={{ marginBottom: '8px' }}>
            {stats.count} / {totalIntersections} intersections
          </p>
          <StatRow label="Crashes"  value={stats.crashes}  />
          <StatRow label="Injuries" value={stats.injuries} />
          <StatRow label="KSI"      value={stats.ksi}      warn />
          <StatRow label="Fatal"    value={stats.fatal}    warn />
          <StatRow label="Ped"      value={stats.ped}      />
          <StatRow label="Bike"     value={stats.bike}     />
        </div>
      )}

      {/* Risk Tier Filter */}
      <div className="panel-header">
        <p className="text-xs uppercase font-semibold text-muted" style={{ marginBottom: '12px' }}>Risk Tier</p>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
          {ALL_TIERS.map((tier) => {
            const { label, color, range } = TIER_META[tier];
            const active = activeTiers.includes(tier);
            return (
              <label key={tier} className="interactive-row" style={{ opacity: active ? 1 : 0.6 }}>
                <input
                  type="checkbox"
                  checked={active}
                  onChange={() => onTierToggle(tier)}
                  style={{ accentColor: color, cursor: 'pointer', width: '14px', height: '14px' }}
                />
                <div style={{ width: '10px', height: '10px', borderRadius: '50%', backgroundColor: color }} />
                <div style={{ display: 'flex', flexDirection: 'column' }}>
                  <p className="text-sm">{label}</p>
                  <p className="text-xs text-muted">{range}</p>
                </div>
              </label>
            );
          })}
        </div>
      </div>

      {/* Legend */}
      <div className="panel-content">
        <p className="text-xs uppercase font-semibold text-muted" style={{ marginBottom: '12px' }}>Legend</p>
        <p className="text-xs text-muted" style={{ marginBottom: '8px' }}>Circle size = risk score</p>
        <div style={{ display: 'flex', alignItems: 'flex-end', gap: '8px', marginBottom: '16px' }}>
          {[8, 12, 18].map((r) => (
            <div key={r} style={{ width: r, height: r, backgroundColor: 'var(--bg-panel-hover)', borderRadius: '50%' }} />
          ))}
          <span className="text-xs text-muted" style={{ marginLeft: '4px' }}>higher →</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{ width: '16px', height: '2px', backgroundColor: 'var(--accent-cyan)' }} />
          <span className="text-xs text-muted">Bike lane</span>
        </div>
      </div>

    </aside>
  );
}
