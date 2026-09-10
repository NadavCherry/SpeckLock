#!/usr/bin/env python3
"""Compare the tables of two versions of a generated markdown report, cell by cell.

WHY THIS EXISTS
---------------
A generator's job checks its output against the committed report before that output may
replace it. The first version of that check compared every numeral in order, which is right
for a prose-only regeneration and wrong the moment a column is added on purpose: it fails a
correct report. This is the general form. A regeneration may ADD tables, columns, rows or
prose. It may not CHANGE a cell both versions share, nor DROP a table, row or column -- and
when it does, the table, the row and the column are named.

HOW TABLES AND ROWS ARE MATCHED
-------------------------------
Not by position. The previous version promised that rows could be added and then compared
tables and rows by index, so one added arm failed a correct report, and one inserted table
would have misaligned every table after it. Now:

  tables  in document order. An old table is matched to the first later new table that has
          the same first column heading, at least half of its column headings, and at least
          one of its rows -- the last so that an inserted table with the same headings (a
          second "comparison | seed | ..." table) cannot take an old table's place.
  rows    by first cell, and by occurrence where a first cell repeats (the k-th
          "ours - YOLOMG" row is the k-th such row in both), so a row inserted anywhere is an
          addition, not a shift.

    python tools/report_diff.py OLD.md NEW.md      # exit 0 only if nothing shared moved
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

MARKS = "†‡*"                      # footnote marks on a header are not a rename
SEP = re.compile(r":?-{3,}:?")


def tables(text: str) -> list[tuple[list[str], list[list[str]]]]:
    """Every markdown table as (header cells, [row cells, ...]), in document order."""
    out, cur = [], None
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("|") and s.endswith("|") and len(s) > 1:
            cells = [c.strip() for c in s[1:-1].split("|")]
            if cur is None:
                cur = (cells, [])
            elif not all(SEP.fullmatch(c) for c in cells):
                cur[1].append(cells)
        elif cur is not None:
            out.append(cur)
            cur = None
    if cur is not None:
        out.append(cur)
    return out


def _norm(h: str) -> str:
    return h.strip().strip(MARKS).strip().lower()


def _keys(rows: list[list[str]]) -> list[tuple[str, int]]:
    """(first cell, occurrence) for every row: an identity that survives inserted rows."""
    seen: dict[str, int] = {}
    out = []
    for r in rows:
        k = r[0] if r else ""
        out.append((k, seen.get(k, 0)))
        seen[k] = seen.get(k, 0) + 1
    return out


def _corresponds(old, new) -> bool:
    (ho, ro), (hn, rn) = old, new
    if not ho or not hn or _norm(ho[0]) != _norm(hn[0]):
        return False
    have = {_norm(h) for h in hn}
    if 2 * sum(_norm(h) in have for h in ho) < len(ho):
        return False
    return not ro or bool(set(_keys(ro)) & set(_keys(rn)))


def compare(old: str, new: str) -> list[str]:
    """Human-readable problems; empty means every shared cell is unchanged."""
    problems = []
    to, tn = tables(old), tables(new)
    start = 0
    for t, (ho, ro) in enumerate(to, 1):
        j = next((k for k in range(start, len(tn)) if _corresponds((ho, ro), tn[k])), None)
        if j is None:
            problems.append(f"table {t} ('{' | '.join(ho)}') is gone")
            continue
        start = j + 1
        hn, rn = tn[j]
        where: dict[str, int] = {}
        for c, h in enumerate(hn):
            where.setdefault(_norm(h), c)
        cols = []
        for i, h in enumerate(ho):
            c = where.get(_norm(h))
            if c is None:
                problems.append(f"table {t}: column '{h}' is gone")
            else:
                cols.append((i, c, h))
        by_key = dict(zip(_keys(rn), rn))
        gone = [k for k in _keys(ro) if k not in by_key]
        if gone:
            problems.append(f"table {t}: {len(ro)} rows -> {len(rn)}; gone: "
                            + ", ".join(k if n == 0 else f"{k} (#{n + 1})" for k, n in gone))
        for r, (key, a) in enumerate(zip(_keys(ro), ro), 1):
            b = by_key.get(key)
            if b is None:
                continue
            for i, c, h in cols:
                va = a[i] if i < len(a) else ""
                vb = b[c] if c < len(b) else ""
                if va != vb:
                    problems.append(f"table {t}, row {r} ({a[0] if a else '?'}), "
                                    f"column '{h}': {va!r} -> {vb!r}")
    return problems


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print(__doc__)
        return 2
    old, new = (Path(p).read_text(encoding="utf-8") for p in argv)
    problems = compare(old, new)
    if problems:
        print(f"FAIL: {len(problems)} shared cell(s) changed")
        for p in problems[:20]:
            print("  " + p)
        return 1
    to, tn = tables(old), tables(new)
    print(f"VERIFIED: every shared cell unchanged ({len(to)} -> {len(tn)} tables, "
          f"{sum(len(r) for _, r in to)} -> {sum(len(r) for _, r in tn)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
