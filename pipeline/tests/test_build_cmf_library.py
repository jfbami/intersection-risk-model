"""Unit tests for build_cmf_library.

Locks in the methodology decisions made in Phase 5:
  - Aggregation dispatch: variance-weighting ONLY when every study has an SE
  - Inverse-treatment math: 1/AMF with CI bounds swapped
  - Direction labelling: helpful / anti_indication / inverted
  - Standard-error fallback: adjusted → unadjusted → none
"""

import pandas as pd
import pytest

from pipeline.build_cmf_library import (
    CuratedEntry,
    DIRECTION_ANTI_INDICATION,
    DIRECTION_HELPFUL,
    DIRECTION_INVERTED,
    StudySample,
    aggregate_across_studies,
    aggregate_single_study,
    aggregate_studies,
    aggregate_via_variance_weighting,
    extract_standard_error,
    resolve_recommendation,
)


def _study(amf: float, se: float | None = None, method: str = "adjusted") -> StudySample:
    return StudySample(
        amf=amf,
        se=se,
        se_method=method if se is not None else "none",
        qual_rating=3,
        citation="fixture",
    )


# ---------------------------------------------------------------------------
# Dispatch rule — the critical methodology decision
# ---------------------------------------------------------------------------

def test_dispatch_uses_variance_weighting_when_every_study_has_se():
    samples = [_study(0.5, se=0.1), _study(0.7, se=0.1)]

    agg = aggregate_studies(samples)

    expected = aggregate_via_variance_weighting(samples)
    assert agg.se_method == expected.se_method
    assert agg.cmf == pytest.approx(expected.cmf)


def test_dispatch_falls_back_to_across_study_when_any_se_is_missing():
    # The exact case the methodology guards against: one SE-bearing study
    # would dominate variance-weighting and drop the other three.
    samples = [
        _study(0.27),
        _study(1.43, se=0.36),
        _study(0.16),
        _study(0.35),
    ]

    agg = aggregate_studies(samples)

    assert agg.se_method == "across_study"
    assert agg.cmf == pytest.approx(sum(s.amf for s in samples) / 4)


def test_dispatch_with_single_study_returns_that_study():
    samples = [_study(0.6, se=0.2)]

    agg = aggregate_studies(samples)

    assert agg.n_studies == 1
    assert agg.cmf == pytest.approx(0.6)


def test_dispatch_raises_when_no_studies():
    with pytest.raises(ValueError):
        aggregate_studies([])


# ---------------------------------------------------------------------------
# Aggregation primitives
# ---------------------------------------------------------------------------

def test_variance_weighting_recovers_known_inverse_variance_mean():
    # Two studies with identical SE → simple mean; weighted CI tighter than either.
    samples = [_study(0.4, se=0.1), _study(0.6, se=0.1)]

    agg = aggregate_via_variance_weighting(samples)

    assert agg.cmf == pytest.approx(0.5)
    # Each study's individual 90% CI half-width = 1.6449 * 0.1 = 0.164
    # Combined SE = sqrt(1 / (2 / 0.01)) = sqrt(0.005) ≈ 0.0707  → half-width ≈ 0.116
    assert agg.se < 0.1  # combined SE strictly smaller than either input SE


def test_across_study_variance_uses_sample_variance_for_ci_width():
    samples = [_study(0.2), _study(0.6), _study(1.0)]

    agg = aggregate_across_studies(samples)

    assert agg.cmf == pytest.approx(0.6)
    assert agg.se_method == "across_study"
    assert agg.cmf_ci_low < agg.cmf < agg.cmf_ci_high


def test_single_study_with_no_se_has_zero_width_interval():
    only = _study(0.7)  # no SE

    agg = aggregate_single_study(only)

    assert agg.cmf == 0.7
    assert agg.cmf_ci_low == 0.7
    assert agg.cmf_ci_high == 0.7
    assert agg.se_method == "none"


# ---------------------------------------------------------------------------
# Recommendation resolution + direction
# ---------------------------------------------------------------------------

def _entry(is_inverse: bool = False) -> CuratedEntry:
    return CuratedEntry(
        id="t",
        cmid=999,
        recommended_name="Test",
        description="",
        preconditions={},
        studied_action_is_inverse=is_inverse,
    )


def test_helpful_direction_when_studied_amf_below_one():
    agg = aggregate_across_studies([_study(0.4), _study(0.5)])

    resolved = resolve_recommendation(_entry(), agg)

    assert resolved.direction == DIRECTION_HELPFUL
    assert resolved.cmf == pytest.approx(agg.cmf)


def test_anti_indication_direction_when_studied_amf_above_one():
    agg = aggregate_across_studies([_study(1.4), _study(1.6)])

    resolved = resolve_recommendation(_entry(), agg)

    assert resolved.direction == DIRECTION_ANTI_INDICATION


def test_inverse_treatment_inverts_amf_and_swaps_ci_bounds():
    # Studied "permit X" with AMF 2.0 (CI 1.6–2.4) becomes
    # "prohibit X" with recommended CMF 0.5 (CI 1/2.4 .. 1/1.6).
    samples = [_study(2.0, se=0.243), _study(2.0, se=0.243)]
    agg = aggregate_via_variance_weighting(samples)

    resolved = resolve_recommendation(_entry(is_inverse=True), agg)

    assert resolved.direction == DIRECTION_INVERTED
    assert resolved.cmf == pytest.approx(1.0 / agg.cmf)
    # Inversion swaps bounds: low end of recommended = 1/high end of studied
    assert resolved.cmf_ci_low  == pytest.approx(1.0 / agg.cmf_ci_high)
    assert resolved.cmf_ci_high == pytest.approx(1.0 / agg.cmf_ci_low)


# ---------------------------------------------------------------------------
# Standard-error extraction
# ---------------------------------------------------------------------------

def test_extract_se_prefers_adjusted_over_unadjusted():
    row = pd.Series({"adjStanErrorAmf": 0.10, "unAdjStanErrorAmf": 0.30})

    se, method = extract_standard_error(row)

    assert se == 0.10
    assert method == "adjusted"


def test_extract_se_falls_back_to_unadjusted_when_adjusted_missing():
    row = pd.Series({"adjStanErrorAmf": None, "unAdjStanErrorAmf": 0.30})

    se, method = extract_standard_error(row)

    assert se == 0.30
    assert method == "unadjusted"


def test_extract_se_returns_none_when_neither_present():
    row = pd.Series({"adjStanErrorAmf": None, "unAdjStanErrorAmf": None})

    se, method = extract_standard_error(row)

    assert se is None
    assert method == "none"


def test_extract_se_treats_zero_se_as_missing():
    row = pd.Series({"adjStanErrorAmf": 0.0, "unAdjStanErrorAmf": 0.30})

    se, method = extract_standard_error(row)

    assert se == 0.30
    assert method == "unadjusted"
