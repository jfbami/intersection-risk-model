"""Unit tests for assemble_features helpers that don't require GeoDataFrame fixtures."""

import pandas as pd

from pipeline.assemble_features import (
    _cast_binary_columns,
    _column_summary,
    _find_volume_column,
    _find_year_column,
)


def test_cast_binary_columns_fills_nan_and_casts_to_int():
    raw = pd.DataFrame({
        "is_signalized":  [1.0, 0.0, None],
        "is_arterial":    [None, 1.0, 0.0],
        "bike_facility":  [1.0, None, 0.0],
        "arterial_class": [2.0, None, 3.0],
    })

    cast = _cast_binary_columns(raw)

    assert cast["is_signalized"].tolist()  == [1, 0, 0]
    assert cast["is_arterial"].tolist()    == [0, 1, 0]
    assert cast["bike_facility"].tolist()  == [1, 0, 0]
    assert cast["arterial_class"].tolist() == [2, 0, 3]
    for col in ("is_signalized", "is_arterial", "bike_facility", "arterial_class"):
        assert cast[col].dtype.kind == "i"


def test_column_summary_for_binary_column_counts_zeros_and_ones():
    df = pd.DataFrame({"is_signalized": [1, 0, 1, 0, 0]})

    summary = _column_summary(df, "is_signalized")

    assert "0=3" in summary
    assert "1=2" in summary


def test_column_summary_for_continuous_column_reports_min_med_max():
    df = pd.DataFrame({"max_speed_limit": [25.0, 30.0, 35.0, 40.0]})

    summary = _column_summary(df, "max_speed_limit")

    assert "min=25" in summary
    assert "max=40" in summary


def test_column_summary_for_all_nan_column_reports_all_nan():
    df = pd.DataFrame({"max_aadt": [float("nan"), float("nan")]})

    summary = _column_summary(df, "max_aadt")

    assert summary == "all NaN"


def test_volume_column_detector_prefers_seattle_awdt():
    seattle_layer = pd.DataFrame(columns=["year", "AMPK", "PMPK", "AWDT", "ADT", "AWDT_ROUND"])

    assert _find_volume_column(seattle_layer) == "AWDT"


def test_volume_column_detector_falls_back_to_awdt_round_then_adt():
    awdt_round_only = pd.DataFrame(columns=["year", "AWDT_ROUND", "ADT"])
    adt_only        = pd.DataFrame(columns=["year", "ADT"])

    assert _find_volume_column(awdt_round_only) == "AWDT_ROUND"
    assert _find_volume_column(adt_only)        == "ADT"


def test_volume_column_detector_matches_generic_aadt_names_for_other_cities():
    other_city = pd.DataFrame(columns=["year", "AADT", "geometry"])

    assert _find_volume_column(other_city) == "AADT"


def test_volume_column_detector_returns_none_when_no_match():
    nothing_useful = pd.DataFrame(columns=["FID", "year", "OBJECTID"])

    assert _find_volume_column(nothing_useful) is None


def test_year_column_detector_finds_lowercase_year():
    assert _find_year_column(pd.DataFrame(columns=["year", "AWDT"])) == "year"


def test_year_column_detector_returns_none_when_absent():
    assert _find_year_column(pd.DataFrame(columns=["AWDT", "FID"])) is None
