"""SUMMARY.md reports the Temporal-YOLOv8 arm as its own Holm family, and labels it truthfully.

The family is declared before any of its scorecards exist. If these comparisons were folded
into the main table, the larger family would change the Holm-adjusted p of every row already
published there -- a result moving because a different question was asked beside it.
"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def ms():
    return pytest.importorskip("tools.make_summary")


def test_prior_art_scorecards_parse(ms):
    assert ms.parse_name("tyolov8_ardmav-e70-s1") == ("ardmav", "tyolov8", "e70", 1)
    assert ms.parse_name("tyolov8_nps-e70-s0") == ("nps", "tyolov8", "e70", 0)


def test_existing_names_still_parse_exactly_as_before(ms):
    assert ms.parse_name("temporal_ardmav-e100-s2") == ("ardmav", "temporal", "e100", 2)
    assert ms.parse_name("singleframe_nps-s0") == ("nps", "singleframe", "e30", 0)
    assert ms.parse_name("yolomg_nps_seed1") == ("nps", "yolomg", "e30", 1)
    assert ms.parse_name("dt12_nps-s0") is None


def test_the_prior_art_comparison_is_its_own_holm_family(ms):
    assert not set(ms.PRIOR_ART_TESTS) & set(ms.TESTS)
    assert all("tyolov8" not in (t[0], t[2]) for t in ms.TESTS)
    assert any(t[0] == "temporal" and t[2] == "tyolov8" for t in ms.PRIOR_ART_TESTS)


def test_budget_labels_are_true(ms):
    assert ms._budget_label("tyolov8", "e70") == "70 ep"
    assert ms._budget_label("yolomg", "e30") == "100 ep"
    assert ms._budget_label("temporal", "e100") == "100 ep"
    assert ms._budget_label("temporal", "e30") == "30 ep"


def test_the_60fps_subset_is_declared_and_descriptive(ms):
    """Named before any Temporal-YOLOv8 scorecard existed; printed as effect sizes only."""
    assert ms.NPS_60FPS_TEST == ("Clip_049", "Clip_050")
    assert ms.without_60fps_lines({}, [0, 1, 2]) == []
