"""Unit tests for pipeline.treatments — CMF library, applicability, ranking."""

import pytest

from pipeline.treatments import (
    PredictionInterval,
    Treatment,
    applicable_treatments,
    compute_prevention,
    is_applicable,
    load_treatments,
    rank_recommended_treatments,
)


def _make_treatment(**overrides) -> Treatment:
    base = dict(
        id="t1",
        name="Test treatment",
        description="",
        cmf=0.5,
        cmf_ci_low=0.4,
        cmf_ci_high=0.6,
        applies_to="bike_ksi",
        preconditions={},
        source="",
        notes="",
    )
    base.update(overrides)
    return Treatment(**base)


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------

def test_curated_library_loads_and_has_entries():
    treatments = load_treatments()
    assert len(treatments) > 0
    for t in treatments:
        assert 0.0 <= t.cmf <= 2.0
        assert t.cmf_ci_low <= t.cmf <= t.cmf_ci_high


# ---------------------------------------------------------------------------
# Precondition matching
# ---------------------------------------------------------------------------

def test_empty_preconditions_means_universally_applicable():
    t = _make_treatment(preconditions={})

    assert is_applicable({}, t)
    assert is_applicable({"anything": 99}, t)


def test_equality_precondition_matches_exact_value():
    t = _make_treatment(preconditions={"is_signalized": 1})

    assert is_applicable({"is_signalized": 1}, t)
    assert not is_applicable({"is_signalized": 0}, t)


def test_at_least_suffix_enforces_lower_bound():
    t = _make_treatment(preconditions={"max_speed_limit_at_least": 30})

    assert is_applicable({"max_speed_limit": 35}, t)
    assert is_applicable({"max_speed_limit": 30}, t)
    assert not is_applicable({"max_speed_limit": 25}, t)


def test_at_most_suffix_enforces_upper_bound():
    t = _make_treatment(preconditions={"arterial_class_at_most": 2})

    assert is_applicable({"arterial_class": 1}, t)
    assert is_applicable({"arterial_class": 2}, t)
    assert not is_applicable({"arterial_class": 3}, t)


def test_missing_feature_fails_threshold_check():
    t_at_least = _make_treatment(preconditions={"max_speed_limit_at_least": 30})
    t_at_most  = _make_treatment(preconditions={"arterial_class_at_most": 2})

    assert not is_applicable({}, t_at_least)
    assert not is_applicable({}, t_at_most)


# ---------------------------------------------------------------------------
# Effect estimation
# ---------------------------------------------------------------------------

def test_compute_prevention_returns_prediction_times_one_minus_cmf():
    pred = PredictionInterval(mean=0.10, ci_low=0.05, ci_high=0.20)
    t    = _make_treatment(cmf=0.40, cmf_ci_low=0.30, cmf_ci_high=0.50)

    result = compute_prevention(pred, t)

    assert result.prevented_per_year_mean == pytest.approx(0.10 * (1 - 0.40))


def test_ci_propagation_uses_pessimistic_corners():
    pred = PredictionInterval(mean=0.10, ci_low=0.05, ci_high=0.20)
    t    = _make_treatment(cmf=0.40, cmf_ci_low=0.30, cmf_ci_high=0.50)

    result = compute_prevention(pred, t)

    # Low corner: low prediction × high-CMF (smallest 1-CMF) = least prevention
    assert result.prevented_per_year_ci_low  == pytest.approx(0.05 * (1 - 0.50))
    # High corner: high prediction × low-CMF (largest 1-CMF) = most prevention
    assert result.prevented_per_year_ci_high == pytest.approx(0.20 * (1 - 0.30))


# ---------------------------------------------------------------------------
# Applicable filter + ranking
# ---------------------------------------------------------------------------

def test_applicable_treatments_filters_by_preconditions():
    treatments = [
        _make_treatment(id="signal_only",       preconditions={"is_signalized": 1}),
        _make_treatment(id="needs_speed_30",    preconditions={"max_speed_limit_at_least": 30}),
        _make_treatment(id="universal"),
    ]
    features = {"is_signalized": 0, "max_speed_limit": 25}

    result_ids = {t.id for t in applicable_treatments(features, treatments)}

    assert result_ids == {"universal"}


def test_ranking_sorts_by_expected_prevented_descending():
    pred = PredictionInterval(mean=1.0, ci_low=0.5, ci_high=2.0)
    treatments = [
        _make_treatment(id="weak",   cmf=0.90, cmf_ci_low=0.85, cmf_ci_high=0.95),
        _make_treatment(id="strong", cmf=0.40, cmf_ci_low=0.30, cmf_ci_high=0.50),
        _make_treatment(id="medium", cmf=0.70, cmf_ci_low=0.60, cmf_ci_high=0.80),
    ]

    ranked = rank_recommended_treatments({}, pred, treatments, top_n=3)

    assert [r.id for r in ranked] == ["strong", "medium", "weak"]


def test_ranking_truncates_to_top_n():
    pred = PredictionInterval(mean=1.0, ci_low=0.5, ci_high=2.0)
    treatments = [_make_treatment(id=f"t{i}", cmf=0.5) for i in range(5)]

    ranked = rank_recommended_treatments({}, pred, treatments, top_n=2)

    assert len(ranked) == 2
