"""Unit tests for the EB math and tier-assignment logic in score_risk.py."""

import pandas as pd
import pytest

from pipeline.score_risk import (
    _assign_tier,
    _percentile_rank,
    compute_eb_estimate,
)


def test_eb_pulls_predicted_halfway_when_weight_equals_one_half():
    # w = 1 / (1 + 0.5 * 2) = 0.5  →  eb = 0.5*2 + 0.5*4 = 3.0
    predicted = pd.Series([2.0])
    observed  = pd.Series([4.0])

    eb = compute_eb_estimate(predicted, observed, alpha=0.5)

    assert eb.iloc[0] == pytest.approx(3.0)


def test_eb_returns_predicted_when_predicted_is_zero():
    # w = 1 / (1 + alpha * 0) = 1  →  eb = predicted = 0
    predicted = pd.Series([0.0])
    observed  = pd.Series([5.0])

    eb = compute_eb_estimate(predicted, observed, alpha=0.5)

    assert eb.iloc[0] == pytest.approx(0.0)


def test_eb_converges_to_one_over_alpha_plus_observed_at_large_predicted():
    # As predicted → ∞ with α fixed:
    #   w·predicted = predicted / (1 + α·predicted) → 1/α
    #   (1−w)·observed → observed
    # So eb → 1/α + observed (NOT simply observed — the NB scale parameter
    # contributes a residual term).
    predicted = pd.Series([1e6])
    observed  = pd.Series([5.0])
    alpha     = 0.5

    eb = compute_eb_estimate(predicted, observed, alpha=alpha)

    assert eb.iloc[0] == pytest.approx(1.0 / alpha + observed.iloc[0], abs=1e-3)


def test_percentile_rank_spans_zero_to_one_hundred():
    series = pd.Series([1.0, 2.0, 3.0, 4.0])

    ranks = _percentile_rank(series).tolist()

    assert min(ranks) > 0
    assert max(ranks) == pytest.approx(100.0)
    assert ranks == sorted(ranks)  # monotonic in input


def test_assign_tier_boundary_values():
    assert _assign_tier(90.0) == "very_high"
    assert _assign_tier(89.9) == "high"
    assert _assign_tier(70.0) == "high"
    assert _assign_tier(69.9) == "moderate"
    assert _assign_tier(40.0) == "moderate"
    assert _assign_tier(20.0) == "low"
    assert _assign_tier(19.9) == "very_low"
    assert _assign_tier(0.0)  == "very_low"
