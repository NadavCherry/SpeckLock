"""tools/report_diff.py: a regenerated report may add columns; it may not move a shared cell.

Its predecessor compared every numeral in order and failed a correct regeneration the moment
a footnote said "best-F1". This one must pass additions and still catch a single moved digit.
"""
from __future__ import annotations

from tools.report_diff import compare

OLD = """Intro prose.

| arm | AP | recall |
|---|---|---|
| single | **0.159** | 0.199 |
| temporal | **0.895** | 0.840 |

More prose.
"""


def test_an_identical_report_passes():
    assert compare(OLD, OLD) == []


def test_an_added_column_passes():
    new = (OLD.replace("| arm | AP | recall |", "| arm | AP | recall | recall, all‡ |")
              .replace("|---|---|---|", "|---|---|---|---|")
              .replace("| 0.199 |", "| 0.199 | 0.570 |")
              .replace("| 0.840 |", "| 0.840 | 0.911 |"))
    assert compare(OLD, new) == []


def test_a_footnote_mark_on_a_header_is_not_a_rename():
    assert compare(OLD, OLD.replace("| recall |", "| recall† |")) == []


def test_added_prose_passes():
    assert compare(OLD, OLD + "\nA new paragraph with 12 numbers, 0.5 and 3.\n") == []


def test_one_moved_digit_fails_and_names_row_and_column():
    problems = compare(OLD, OLD.replace("0.840", "0.841"))
    assert len(problems) == 1
    assert "temporal" in problems[0] and "'recall'" in problems[0]


def test_a_dropped_column_fails():
    new = (OLD.replace("| arm | AP | recall |", "| arm | AP |")
              .replace("|---|---|---|", "|---|---|")
              .replace(" 0.199 |", "").replace(" 0.840 |", ""))
    assert any("column 'recall' is gone" in p for p in compare(OLD, new))


def test_a_dropped_row_fails():
    new = OLD.replace("| temporal | **0.895** | 0.840 |\n", "")
    assert any("2 rows -> 1" in p for p in compare(OLD, new))


# The contract always said rows could be ADDED; the implementation compared by position, so a
# fourth arm in a local report or a new prior-art table in SUMMARY.md failed a correct report.

TWO = OLD + """
| comparison | seed | d AP |
|---|---|---|
| ours - them | 0 | +0.010 |
| ours - them | 1 | +0.020 |
"""


def test_an_added_row_passes_wherever_it_is_inserted():
    row = "| prior art | **0.700** | 0.600 |\n"
    appended = OLD.replace("| temporal | **0.895** | 0.840 |\n",
                           "| temporal | **0.895** | 0.840 |\n" + row)
    inserted = OLD.replace("| temporal |", row + "| temporal |")
    assert compare(OLD, appended) == []
    assert compare(OLD, inserted) == []


def test_an_inserted_table_does_not_misalign_the_tables_after_it():
    extra = "| comparison | seed | d AP |\n|---|---|---|\n| ours - prior art | 0 | +0.300 |\n\n"
    new = TWO.replace("More prose.\n", "More prose.\n\n" + extra)
    assert compare(TWO, new) == []
    moved = new.replace("| ours - them | 1 | +0.020 |", "| ours - them | 1 | +0.021 |")
    problems = compare(TWO, moved)
    assert len(problems) == 1
    assert "ours - them" in problems[0] and "'d AP'" in problems[0]


def test_repeated_first_cells_are_matched_by_occurrence():
    new = TWO.replace("| ours - them | 1 | +0.020 |",
                      "| ours - them | 1 | +0.020 |\n| ours - them | 2 | +0.030 |")
    assert compare(TWO, new) == []
    swapped = TWO.replace("| 0 | +0.010 |", "| 0 | +0.020 |")
    assert len(compare(TWO, swapped)) == 1


def test_a_dropped_table_fails():
    new = TWO.split("| comparison |")[0]
    assert any("('comparison | seed | d AP') is gone" in p for p in compare(TWO, new))
