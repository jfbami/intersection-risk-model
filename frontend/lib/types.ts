export type RiskTier = 'very_high' | 'high' | 'moderate' | 'low' | 'very_low';

export const ALL_TIERS: RiskTier[] = ['very_high', 'high', 'moderate', 'low', 'very_low'];

export const TIER_META: Record<RiskTier, { label: string; color: string; range: string }> = {
  very_high: { label: 'Very High', color: '#ef4444', range: '≥ 90th pct' },
  high:      { label: 'High',      color: '#f97316', range: '70–89th'    },
  moderate:  { label: 'Moderate',  color: '#eab308', range: '40–69th'    },
  low:       { label: 'Low',       color: '#84cc16', range: '20–39th'    },
  very_low:  { label: 'Very Low',  color: '#22c55e', range: '< 20th'     },
};

/** Display % the credible interval covers — must match score_risk.CREDIBLE_LEVEL. */
export const CREDIBLE_LEVEL_PCT = 90;

/** A single model-driven driver of an intersection's predicted risk. */
export interface Contributor {
  label: string;
  pct_change: number;
}

/** A CMF-based design recommendation. Phase-5 prescriptive output. */
export interface RecommendedTreatment {
  id: string;
  name: string;
  prevented_per_year_mean:    number;
  prevented_per_year_ci_low:  number;
  prevented_per_year_ci_high: number;
  cmf: number;
}

/** One selected intersection — normalised from the GeoJSON feature properties. */
export interface SelectedIntersection {
  intersectionId: string;     // hex hash from build_intersections.py; not numeric
  name: string;
  // ── Phase 1 headline ─────────────────────────────────────
  expectedBikeKsiPerYear: number;
  expectedBikeKsiCiLow:   number;
  expectedBikeKsiCiHigh:  number;
  citywideMedianBikeKsi:  number;  // anchor for relative-comparison framing
  totalArterials:         number;  // modelled-site count, for "X of N" rank framing
  topContributors: Contributor[];
  recommendedTreatments: RecommendedTreatment[];
  // ── Secondary percentile ─────────────────────────────────
  riskScore: number;
  riskRank:  number;
  riskTier:  RiskTier;
  // ── Underlying model + observed counts ───────────────────
  predictedCrashes: number;
  ebPredicted: number;
  observedCrashes: number;
  yearsObserved: number;
  // ── Vision Zero severity ─────────────────────────────────
  injuryTotal: number;
  ksiTotal: number;
  fatalTotal: number;
  pedTotal: number;
  bikeTotal: number;
  bikeKsiTotal: number;
  // ── Infrastructure features ──────────────────────────────
  isSignalized: boolean;
  numLegs: number;
  maxSpeedLimit: number;
  bikeFacility: string;
  arterialClass: string;
  coordinates: { lat: number; lng: number };
}

export interface LayerVisibility {
  circles: boolean;
  bikeFacility: boolean;
}

export interface ScorecardStats {
  count: number;
  crashes: number;
  injuries: number;
  ksi: number;
  fatal: number;
  ped: number;
  bike: number;
}
