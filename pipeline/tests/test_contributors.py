"""Unit tests for the top-contributors computation."""

import math

from pipeline.contributors import compute_for_row, compute_for_row_as_dicts
from pipeline.feature_encoding import leg_category_param


PARAMS_FIXTURE = {
    "Intercept":                  -10.0,
    "is_signalized":                0.5,
    leg_category_param(2):         -0.5,
    leg_category_param(3):         -0.3,
    leg_category_param(5):          0.2,   # the top-coded "5+" category
    "max_speed_limit":              0.05,
    "bike_facility":               -0.3,
    "C(arterial_class)[T.1]":       1.0,
    "C(arterial_class)[T.2]":       2.0,
}


def _leg_contributor(features):
    result = compute_for_row(features, PARAMS_FIXTURE)
    return next(c for c in result if "leg" in c.label.lower())


def test_local_access_unsignalized_has_baseline_contributions():
    features = {
        "is_signalized":   0,
        "num_legs":        4,
        "max_speed_limit": 25,
        "bike_facility":   0,
        "arterial_class":  0,
    }

    result = compute_for_row(features, PARAMS_FIXTURE)

    for c in result:
        assert math.isclose(c.pct_change, 0.0, abs_tol=1e-6)


def test_signalized_minor_arterial_surfaces_both_drivers():
    features = {
        "is_signalized":   1,
        "num_legs":        4,
        "max_speed_limit": 25,
        "bike_facility":   0,
        "arterial_class":  2,
    }

    result = compute_for_row(features, PARAMS_FIXTURE)
    labels = [c.label for c in result]

    assert "Signalized" in labels
    assert "Minor arterial" in labels


def test_top_n_is_sorted_by_magnitude():
    features = {
        "is_signalized":   1,
        "num_legs":        4,
        "max_speed_limit": 25,
        "bike_facility":   0,
        "arterial_class":  2,
    }

    top3 = compute_for_row_as_dicts(features, PARAMS_FIXTURE, top_n=3)

    # Magnitudes should be non-increasing.
    magnitudes = [abs(c["pct_change"]) for c in top3]
    assert magnitudes == sorted(magnitudes, reverse=True)
    # Highest-magnitude is arterial class 2 (coef 2.0, exp-1 ≈ 638%).
    assert top3[0]["label"] == "Minor arterial"


def test_higher_speed_limit_increases_risk_when_speed_coef_positive():
    features = {
        "is_signalized":   0,
        "num_legs":        4,
        "max_speed_limit": 35,
        "bike_facility":   0,
        "arterial_class":  0,
    }

    result = compute_for_row(features, PARAMS_FIXTURE)
    speed = next(c for c in result if "mph" in c.label)

    assert speed.pct_change > 0
    assert "35 mph" in speed.label


def test_aadt_contributor_skipped_when_max_aadt_is_nan():
    features = {
        "is_signalized":   0,
        "num_legs":        4,
        "max_speed_limit": 25,
        "bike_facility":   0,
        "arterial_class":  0,
        "max_aadt":        float("nan"),
    }
    params_with_aadt = {**PARAMS_FIXTURE, "log_aadt": 0.3}

    labels = [c.label for c in compute_for_row(features, params_with_aadt)]

    assert not any("AADT" in lbl for lbl in labels)


def test_aadt_contributor_surfaces_when_aadt_present_and_log_aadt_in_params():
    features = {
        "is_signalized":   0,
        "num_legs":        4,
        "max_speed_limit": 25,
        "bike_facility":   0,
        "arterial_class":  0,
        "max_aadt":        24000,
    }
    params_with_aadt = {**PARAMS_FIXTURE, "log_aadt": 0.3}

    result = compute_for_row(features, params_with_aadt)
    aadt_contrib = next(c for c in result if "AADT" in c.label)

    assert aadt_contrib.pct_change > 0   # 24k > 8k reference
    assert "24,000" in aadt_contrib.label


def test_four_leg_reference_has_no_leg_effect():
    leg = _leg_contributor({
        "is_signalized": 0, "num_legs": 4, "max_speed_limit": 25,
        "bike_facility": 0, "arterial_class": 0,
    })

    assert leg.label == "4-leg intersection"
    assert math.isclose(leg.pct_change, 0.0, abs_tol=1e-6)


def test_three_leg_uses_its_own_category_coefficient():
    leg = _leg_contributor({
        "is_signalized": 0, "num_legs": 3, "max_speed_limit": 25,
        "bike_facility": 0, "arterial_class": 0,
    })

    assert leg.label == "3-leg intersection"
    assert math.isclose(leg.pct_change, 100 * (math.exp(-0.3) - 1), abs_tol=1e-6)


def test_six_leg_collapses_into_top_coded_five_plus_category():
    leg = _leg_contributor({
        "is_signalized": 0, "num_legs": 6, "max_speed_limit": 25,
        "bike_facility": 0, "arterial_class": 0,
    })

    assert leg.label == "5+ legs"
    assert math.isclose(leg.pct_change, 100 * (math.exp(0.2) - 1), abs_tol=1e-6)


def test_five_and_six_leg_share_one_effect():
    base = {
        "is_signalized": 0, "max_speed_limit": 25,
        "bike_facility": 0, "arterial_class": 0,
    }

    five = _leg_contributor({**base, "num_legs": 5})
    six  = _leg_contributor({**base, "num_legs": 6})

    assert five.label == six.label == "5+ legs"
    assert math.isclose(five.pct_change, six.pct_change, abs_tol=1e-9)
