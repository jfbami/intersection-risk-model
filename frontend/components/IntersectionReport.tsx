'use client';

import { useEffect, useMemo } from 'react';
import {
  PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis,
  LabelList, Tooltip, ResponsiveContainer,
} from 'recharts';
import { CREDIBLE_LEVEL_PCT, TIER_META, type SelectedIntersection } from '@/lib/types';
import { MetricCard, FeatureBadge, bikeFacilityLabel } from './MetricPrimitives';

interface Props {
  intersection: SelectedIntersection;
  onClose: () => void;
}

const CHART_BIKE = '#22d3ee';
const CHART_PED = '#a78bfa';
const CHART_VEHICLE = '#64748b';
const CHART_MEDIAN = '#64748b';
const AXIS_LABEL = '#9ca3af';

const tooltipStyle = {
  contentStyle: {
    background: 'rgba(22, 24, 29, 0.96)',
    border: '1px solid rgba(255, 255, 255, 0.12)',
    borderRadius: 8,
    fontSize: 12,
  },
  itemStyle: { color: '#f1f3f5' },
  labelStyle: { color: '#9ca3af' },
} as const;

function probabilityWithin(ratePerYear: number, years: number): number {
  return 1 - Math.exp(-ratePerYear * years);
}

function pct(fraction: number): string {
  return `${Math.round(fraction * 100)}%`;
}

function useEscapeToClose(onClose: () => void) {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose]);
}

interface Headline {
  lambdaPerYear: number;
  returnPeriodYears: number | null;
  prob10: number;
  prob10Low: number;
  prob10High: number;
  multipleOfMedian: number | null;
  rankPct: number;
}

function deriveHeadline(i: SelectedIntersection): Headline {
  const lambdaPerYear = i.expectedBikeKsiPerYear;
  return {
    lambdaPerYear,
    returnPeriodYears: lambdaPerYear > 0 ? 1 / lambdaPerYear : null,
    prob10: probabilityWithin(lambdaPerYear, 10),
    prob10Low: probabilityWithin(i.expectedBikeKsiCiLow, 10),
    prob10High: probabilityWithin(i.expectedBikeKsiCiHigh, 10),
    multipleOfMedian: i.citywideMedianBikeKsi > 0 ? lambdaPerYear / i.citywideMedianBikeKsi : null,
    rankPct: i.totalArterials > 0 ? (i.riskRank / i.totalArterials) * 100 : 0,
  };
}

function Card({
  title,
  span,
  children,
}: {
  title?: string;
  span: number;
  children: React.ReactNode;
}) {
  return (
    <section className={`report-card span-${span}`}>
      {title && <p className="card-title">{title}</p>}
      {children}
    </section>
  );
}

function Footnote({ children }: { children: React.ReactNode }) {
  return (
    <p className="text-xs text-muted" style={{ marginTop: 'auto', paddingTop: '12px', lineHeight: 1.45 }}>
      {children}
    </p>
  );
}

function ReportNav({ intersection, onClose }: Props) {
  const { label, color } = TIER_META[intersection.riskTier];
  return (
    <nav className="report-nav">
      <button className="back-button" onClick={onClose} aria-label="Back to map">
        <span style={{ fontSize: '1.1rem', lineHeight: 1 }}>←</span> Back to Map
      </button>
      <div style={{ flexGrow: 1, overflow: 'hidden' }}>
        <p className="text-xs uppercase text-muted" style={{ fontFamily: 'monospace' }}>
          Intersection {intersection.intersectionId.slice(0, 8)}
        </p>
        <h1 className="text-xl font-semibold" style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
          {intersection.name}
        </h1>
      </div>
      <span className="tier-pill" style={{ background: `${color}1f`, color, border: `1px solid ${color}59` }}>
        <span style={{ width: 8, height: 8, borderRadius: '50%', background: color }} />
        {label} risk
      </span>
    </nav>
  );
}

function HeroCard({ i, h }: { i: SelectedIntersection; h: Headline }) {
  const { color } = TIER_META[i.riskTier];
  return (
    <Card title="Expected bike-KSI risk" span={4}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', marginBottom: '20px' }}>
        <span className="text-5xl font-bold" style={{ color }}>
          {h.returnPeriodYears !== null ? `~1` : '—'}
        </span>
        {h.returnPeriodYears !== null && (
          <span className="text-xl text-muted">
            every {Math.round(h.returnPeriodYears)} years
          </span>
        )}
      </div>

      <div style={{ paddingTop: '16px', borderTop: '1px solid var(--border-light)' }}>
        <p className="text-2xl font-bold tabular-nums">{pct(h.prob10)}</p>
        <p className="text-sm text-muted">chance of a bike-KSI in the next 10 years</p>
        <p className="text-xs text-muted" style={{ marginTop: '4px' }}>
          {CREDIBLE_LEVEL_PCT}% CI: {pct(h.prob10Low)} – {pct(h.prob10High)}
        </p>
      </div>

      <div style={{ marginTop: '16px', paddingTop: '16px', borderTop: '1px solid var(--border-light)' }}>
        <p className="text-sm">
          Rank <span className="font-semibold">{i.riskRank}</span> of {i.totalArterials}{' '}
          <span style={{ color }}>(top {Math.round(h.rankPct)}%)</span>
        </p>
        {h.multipleOfMedian !== null && h.multipleOfMedian >= 1.5 && (
          <p className="text-sm text-muted" style={{ marginTop: '4px' }}>
            ~<span className="font-semibold" style={{ color: 'var(--text-primary)' }}>{h.multipleOfMedian.toFixed(1)}×</span> the median Capitol Hill arterial
          </p>
        )}
        <p className="text-xs text-muted" style={{ marginTop: '8px' }}>
          Underlying rate {h.lambdaPerYear.toFixed(3)} /yr · observed {i.bikeKsiTotal} over {i.yearsObserved} yr
        </p>
      </div>

      <Footnote>
        Phase-1 proxy: empirical-Bayes on bike-KSI counts using the all-crash NB
        prediction × citywide bike-KSI share as the prior. Directional until the
        bike-specific model lands.
      </Footnote>
    </Card>
  );
}

function RiskComparisonCard({ i }: { i: SelectedIntersection }) {
  const { color } = TIER_META[i.riskTier];
  const data = [
    { name: 'This site', value: i.expectedBikeKsiPerYear, fill: color },
    { name: 'CH median', value: i.citywideMedianBikeKsi, fill: CHART_MEDIAN },
  ];
  const maxValue = Math.max(i.expectedBikeKsiPerYear, i.citywideMedianBikeKsi, 0.001);

  return (
    <Card title="Risk vs. Capitol Hill median" span={4}>
      <div style={{ height: 150, width: '100%' }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ top: 4, right: 48, bottom: 0, left: 0 }}>
            <XAxis type="number" hide domain={[0, maxValue * 1.3]} />
            <YAxis
              type="category"
              dataKey="name"
              width={78}
              tickLine={false}
              axisLine={false}
              tick={{ fill: AXIS_LABEL, fontSize: 12 }}
            />
            <Tooltip
              {...tooltipStyle}
              cursor={{ fill: 'rgba(255,255,255,0.04)' }}
              formatter={(value) => [`${Number(value).toFixed(3)} /yr`, 'bike-KSI']}
            />
            <Bar dataKey="value" radius={[0, 6, 6, 0]} barSize={26} isAnimationActive={false}>
              {data.map((d) => (
                <Cell key={d.name} fill={d.fill} />
              ))}
              <LabelList
                dataKey="value"
                position="right"
                formatter={(value) => Number(value).toFixed(3)}
                style={{ fill: '#f1f3f5', fontSize: 12, fontWeight: 600 }}
              />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <Footnote>Expected bike-KSI per year vs. the median modelled arterial intersection.</Footnote>
    </Card>
  );
}

function CrashDistributionCard({ i }: { i: SelectedIntersection }) {
  const vehicleOnly = Math.max(0, i.observedCrashes - i.pedTotal - i.bikeTotal);
  const slices = [
    { name: 'Bike', value: i.bikeTotal, color: CHART_BIKE },
    { name: 'Pedestrian', value: i.pedTotal, color: CHART_PED },
    { name: 'Vehicle only', value: vehicleOnly, color: CHART_VEHICLE },
  ].filter((s) => s.value > 0);
  const total = slices.reduce((sum, s) => sum + s.value, 0);

  return (
    <Card title="Crash distribution" span={4}>
      {total === 0 ? (
        <div style={{ height: 150, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <p className="text-sm text-muted">No recorded crashes ({i.yearsObserved} yr)</p>
        </div>
      ) : (
        <>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div style={{ position: 'relative', width: 150, height: 150, flexShrink: 0 }}>
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={slices}
                    dataKey="value"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    innerRadius={48}
                    outerRadius={70}
                    paddingAngle={2}
                    stroke="none"
                    isAnimationActive={false}
                  >
                    {slices.map((s) => (
                      <Cell key={s.name} fill={s.color} />
                    ))}
                  </Pie>
                  <Tooltip {...tooltipStyle} />
                </PieChart>
              </ResponsiveContainer>
              <div
                style={{
                  position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column',
                  alignItems: 'center', justifyContent: 'center', pointerEvents: 'none',
                }}
              >
                <span className="text-2xl font-bold tabular-nums">{total}</span>
                <span className="text-xs text-muted">crashes</span>
              </div>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', flexGrow: 1 }}>
              {slices.map((s) => (
                <div key={s.name} style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span style={{ width: 10, height: 10, borderRadius: 3, background: s.color, flexShrink: 0 }} />
                  <span className="text-sm" style={{ flexGrow: 1 }}>{s.name}</span>
                  <span className="text-sm font-semibold tabular-nums">{s.value}</span>
                </div>
              ))}
            </div>
          </div>
          <Footnote>Historical crashes by most-vulnerable party involved ({i.yearsObserved} yr).</Footnote>
        </>
      )}
    </Card>
  );
}

function TreatmentsCard({ i }: { i: SelectedIntersection }) {
  return (
    <Card title="Recommended treatments" span={12}>
      {i.recommendedTreatments.length === 0 ? (
        <p className="text-sm text-muted">No starter-library treatment matches this configuration.</p>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          {i.recommendedTreatments.map((t) => (
            <div
              key={t.id}
              style={{
                background: 'var(--bg-panel-hover)', borderRadius: '10px',
                border: '1px solid var(--border-light)', padding: '12px',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '8px' }}>
                <span className="text-sm font-medium">{t.name}</span>
                <span className="text-base font-bold" style={{ color: 'var(--accent-success)', whiteSpace: 'nowrap' }}>
                  −{t.prevented_per_year_mean.toFixed(3)}/yr
                </span>
              </div>
              <p className="text-xs text-muted" style={{ marginTop: '6px' }}>
                90% CI: −{t.prevented_per_year_ci_low.toFixed(3)} … −{t.prevented_per_year_ci_high.toFixed(3)}
                {'  ·  '}CMF {t.cmf.toFixed(2)}
              </p>
            </div>
          ))}
        </div>
      )}
      <Footnote>
        Expected bike-KSI/yr prevented per FHWA CMF Clearinghouse values × the
        site&apos;s predicted rate. Verify CMFs before publication.
      </Footnote>
    </Card>
  );
}

function CrashCountsCard({ i }: { i: SelectedIntersection }) {
  return (
    <Card title="Crash counts (NB + EB)" span={6}>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '10px' }}>
        <MetricCard label="NB Predicted" value={i.predictedCrashes.toFixed(1)} sub="per 6 yr" />
        <MetricCard label="EB Adjusted" value={i.ebPredicted.toFixed(1)} sub="shrunk" />
        <MetricCard label="Observed" value={String(i.observedCrashes)} sub={`${i.yearsObserved} yr`} />
      </div>
      <Footnote>
        All-mode counts. EB = w·predicted + (1−w)·observed, w = 1/(1+α·predicted).
      </Footnote>
    </Card>
  );
}

function SeverityCard({ i }: { i: SelectedIntersection }) {
  return (
    <Card title="Vision Zero severity" span={6}>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '10px' }}>
        <MetricCard label="Injuries" value={String(i.injuryTotal)} />
        <MetricCard label="KSI" value={String(i.ksiTotal)} warn={i.ksiTotal > 0} sub="Killed / Serious" />
        <MetricCard label="Fatal" value={String(i.fatalTotal)} warn={i.fatalTotal > 0} />
        <MetricCard label="Ped" value={String(i.pedTotal)} sub="involved" />
        <MetricCard label="Bike" value={String(i.bikeTotal)} sub="involved" />
        <MetricCard label="Bike-KSI" value={String(i.bikeKsiTotal)} warn={i.bikeKsiTotal > 0} />
      </div>
    </Card>
  );
}

function InfrastructureCard({ i }: { i: SelectedIntersection }) {
  return (
    <Card title="Infrastructure & location" span={12}>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '10px', marginBottom: '12px' }}>
        <MetricCard label="Speed Limit" value={i.maxSpeedLimit ? `${i.maxSpeedLimit} mph` : 'N/A'} />
        <MetricCard label="Legs" value={String(i.numLegs)} sub="approaches" />
        <MetricCard label="Signalized" value={i.isSignalized ? 'Yes' : 'No'} />
        <MetricCard label="Class" value={i.arterialClass || 'N/A'} />
      </div>
      <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', alignItems: 'center' }}>
        <FeatureBadge label="Signalized" active={i.isSignalized} />
        <FeatureBadge label={i.arterialClass} active />
        <span className="text-sm" style={{ marginLeft: '4px' }}>{bikeFacilityLabel(i.bikeFacility)}</span>
        <span className="text-xs text-muted" style={{ marginLeft: 'auto', fontFamily: 'monospace' }}>
          {i.coordinates.lat.toFixed(5)}, {i.coordinates.lng.toFixed(5)}
        </span>
      </div>
    </Card>
  );
}

export default function IntersectionReport({ intersection, onClose }: Props) {
  useEscapeToClose(onClose);
  const headline = useMemo(() => deriveHeadline(intersection), [intersection]);
  const { color } = TIER_META[intersection.riskTier];

  return (
    <div className="report-overlay" style={{ ['--report-glow' as string]: `${color}22` } as React.CSSProperties}>
      <ReportNav intersection={intersection} onClose={onClose} />
      <div className="report-grid">
        <HeroCard i={intersection} h={headline} />
        <RiskComparisonCard i={intersection} />
        <CrashDistributionCard i={intersection} />
        <TreatmentsCard i={intersection} />
        <CrashCountsCard i={intersection} />
        <SeverityCard i={intersection} />
        <InfrastructureCard i={intersection} />
      </div>
    </div>
  );
}
