export function MetricCard({
  label,
  value,
  sub,
  warn,
}: {
  label: string;
  value: string;
  sub?: string;
  warn?: boolean;
}) {
  return (
    <div style={{ backgroundColor: 'var(--bg-panel-hover)', borderRadius: '8px', padding: '12px', border: '1px solid var(--border-light)' }}>
      <p className="text-xs uppercase text-muted" style={{ marginBottom: '4px' }}>{label}</p>
      <p className="font-semibold text-lg" style={{ color: warn ? 'var(--accent-danger)' : 'var(--text-primary)' }}>
        {value}
      </p>
      {sub && <p className="text-xs text-muted" style={{ marginTop: '4px' }}>{sub}</p>}
    </div>
  );
}

export function FeatureBadge({ label, active }: { label: string; active: boolean }) {
  return (
    <span
      style={{
        fontSize: '0.65rem',
        padding: '2px 8px',
        borderRadius: '999px',
        fontWeight: 500,
        backgroundColor: active ? 'rgba(34, 211, 238, 0.15)' : 'rgba(255, 255, 255, 0.05)',
        color: active ? 'var(--accent-cyan)' : 'var(--text-secondary)',
        border: `1px solid ${active ? 'rgba(34, 211, 238, 0.3)' : 'var(--border-light)'}`
      }}
    >
      {label}
    </span>
  );
}

export function bikeFacilityLabel(facility: string): string {
  const lower = facility.toLowerCase();
  if (lower.includes('protected')) return '🟢 Protected bike lane';
  if (lower.includes('bike lane') || lower.includes('bike_lane')) return '🟡 Bike lane';
  if (lower.includes('shared')) return '🔵 Shared lane';
  return '⚫ No bike facility';
}
