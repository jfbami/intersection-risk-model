"""Unit tests for snap_crashes.infer_modes_from_description."""

from pipeline.snap_crashes import infer_modes_from_description


def test_empty_string_yields_no_modes():
    assert infer_modes_from_description("") == set()


def test_pedestrian_keyword_yields_ped_mode():
    assert infer_modes_from_description("MOTOR VEHICLE STRUCK PEDESTRIAN") == {"ped"}


def test_pedalcyclist_keyword_yields_bike_mode():
    assert infer_modes_from_description("MOTOR VEHICLE STRUCK PEDALCYCLIST") == {"bike"}


def test_both_keywords_yield_both_modes():
    assert infer_modes_from_description("PEDESTRIAN AND PEDALCYCLIST INVOLVED") == {"ped", "bike"}


def test_keyword_match_is_case_insensitive():
    assert infer_modes_from_description("Vehicle vs pedestrian") == {"ped"}


def test_unrelated_text_yields_no_modes():
    assert infer_modes_from_description("REAR-END COLLISION ON ARTERIAL") == set()
