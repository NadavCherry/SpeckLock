"""The Holm correction this repository's own rule requires, and what it changes.

docs/research/INFRA.md section 6.1 says: Holm across the whole table, and a difference is
called only when the CI excludes zero AND the Holm-adjusted p < 0.05. For months no tool
applied it. These tests pin the helper every paired table now goes through, pin both
tools to it, and pin the committed NPS table's result so the finding cannot quietly
regress.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from dronedet.stats import apply_holm, holm, holm_adjust

REPO = Path(__file__).resolve().parents[2]


def test_holm_adjust_returns_values_in_the_order_given():
    ps = [0.04, 0.01, 0.30]
    by_name = {n: a for n, _, a in holm([(str(i), p) for i, p in enumerate(ps)])}
    adj = holm_adjust(ps)
    assert adj == [by_name["0"], by_name["1"], by_name["2"]]
    assert adj[1] == pytest.approx(0.03)   # smallest of three, times 3
    assert adj[0] == pytest.approx(0.08)   # next, times 2
    assert adj[2] == pytest.approx(0.30)


def test_a_family_of_one_is_unchanged():
    (r,) = apply_holm([{"lo": 0.01, "hi": 0.2, "p_perm": 0.03}])
    assert r["p_perm_holm"] == pytest.approx(0.03)
    assert r["significant"] and r["significant_raw"]


def test_a_ci_through_zero_is_never_significant_however_small_the_p():
    (r,) = apply_holm([{"lo": -0.01, "hi": 0.2, "p_perm": 0.0001}])
    assert not r["significant"] and not r["significant_raw"]


def test_three_marginal_results_fall_together():
    rows = apply_holm([{"lo": 0.01, "hi": 0.1, "p_perm": p} for p in (0.02, 0.03, 0.04)])
    assert all(r["significant_raw"] for r in rows)
    assert not any(r["significant"] for r in rows)   # all three adjust to 0.06


# The committed work/reports/SUMMARY.md NPS table, seed-matched: (name, lo, hi, p_perm).
# Three rows print "worse" uncorrected.
NPS_TABLE = [
    ("temporal-single 30ep s0", -0.109, -0.031, 0.0175),
    ("temporal-single 30ep s1", -0.057, 0.103, 0.4863),
    ("temporal-single 30ep s2", -0.052, 0.065, 0.7791),
    ("temporal-single 100ep s0", -0.050, 0.039, 0.9350),
    ("temporal-single 100ep s1", -0.129, 0.117, 0.5917),
    ("temporal-single 100ep s2", -0.200, -0.057, 0.0205),
    ("temporal30-yolomg s0", -0.127, 0.032, 0.6842),
    ("temporal30-yolomg s1", -0.107, 0.025, 0.3533),
    ("temporal30-yolomg s2", -0.128, -0.029, 0.2534),
    ("temporal100-yolomg s0", -0.147, 0.061, 0.7396),
    ("temporal100-yolomg s1", -0.115, 0.094, 0.8491),
    ("temporal100-yolomg s2", -0.195, -0.053, 0.0100),
]


def test_the_three_nps_worse_verdicts_do_not_survive_the_rule():
    rows = apply_holm([{"name": n, "lo": lo, "hi": hi, "p_perm": p}
                       for n, lo, hi, p in NPS_TABLE])
    assert {r["name"] for r in rows if r["significant_raw"]} == {
        "temporal-single 30ep s0", "temporal-single 100ep s2", "temporal100-yolomg s2"}
    assert not any(r["significant"] for r in rows)


@pytest.mark.parametrize("tool", ["tools/make_summary.py", "tools/size_curve.py"])
def test_every_paired_table_goes_through_apply_holm(tool):
    """A source check, stated as one: both tools that print paired tables must call the
    helper. The failure it guards against is the one that happened -- a rule written
    down and never wired in."""
    assert "apply_holm(" in (REPO / tool).read_text(encoding="utf-8")


def test_make_summary_prints_the_corrected_verdict_and_names_what_it_removed():
    ms = pytest.importorskip("tools.make_summary")
    table = [(f"cmp | {s}", f"cmp, seed {s}",
              {"observed": -0.1, "lo": -0.2, "hi": -0.05, "p": 0.001, "p_perm": p})
             for s, p in enumerate((0.01, 0.03, 0.5))]
    out = ms.holm_table(table)
    assert out[0].endswith("| 0.0100 | 0.0300 | **worse** |")       # 0.01 x 3 survives
    assert out[1].endswith("| 0.0300 | 0.0600 | no difference |")   # 0.03 x 2 does not
    assert "removed 1 verdict(s): cmp, seed 1 (uncorrected: worse)" in "\n".join(out)


def test_size_curve_renders_holm_and_names_the_removed_marks():
    sc = pytest.importorskip("tools.size_curve")
    rows = [{"bin": "<8 px", "seed": s, "d_ap": 0.1, "lo": 0.01, "hi": 0.2, "p_perm": p,
             "significant": p < 0.05, "favours": "ours"}
            for s, p in enumerate((0.01, 0.03, 0.5))]
    md, won = sc.holm_paired_table(rows)
    assert won == {"<8 px": [True, False, False]}
    assert "removed the mark from: <8 px seed 1." in "\n".join(md)
