"""SUMMARY.md prints the GT each arm is scored on; it no longer asserts they are the same.

The file used to open by saying every arm was scored on "the same ... labels". On ARD-MAV
the YOLOMG scorecards carry 28,138 instances against our 28,160 -- two fewer in 11 of 15
sequences. The helper that replaced the assertion is pinned here with synthetic rows, so a
regression shows up in seconds rather than hours into a resampling job.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest


def _rows(n_by_arm):
    rows = {}
    for (arm, budget), n in n_by_arm.items():
        for seed in (0, 1, 2):
            rows[(arm, budget, seed)] = {
                "seqs": [SimpleNamespace(n_gt=n // 2), SimpleNamespace(n_gt=n - n // 2)],
                "ap": 0.5}
    return rows


def test_differing_gt_is_printed_per_arm_not_asserted():
    ms = pytest.importorskip("tools.make_summary")
    out = ms.gt_scored_lines(_rows({("temporal", "e100"): 28160, ("yolomg", "e30"): 28138}))
    assert "28,160" in out[0] and "28,138" in out[0]
    assert "printed rather than asserted" in out[0]


def test_identical_gt_is_stated_once():
    ms = pytest.importorskip("tools.make_summary")
    out = ms.gt_scored_lines(_rows({("temporal", "e100"): 100, ("yolomg", "e30"): 100}))
    assert out[0] == "Every arm is scored on the same 100 GT instances."


def test_the_opening_sentence_no_longer_claims_identical_labels():
    ms = pytest.importorskip("tools.make_summary")
    import inspect
    src = inspect.getsource(ms)
    assert "labels, the same evaluator" not in src
