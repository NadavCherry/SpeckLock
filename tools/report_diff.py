#!/usr/bin/env python3
"""Compare the tables of two versions of a generated markdown report, cell by cell.

WHY THIS EXISTS
---------------
A generator's job checks its output against the committed report before that output may
replace it. The first version of that check compared every numeral in order, which is right
for a prose-only regeneration and wrong the moment a column is added on purpose: it fails a
correct report. This is the general form. A regeneration may ADD columns, rows or prose. It
may not CHANGE a cell in a column both versions share -- and when it does, the table, the row
and the column are named.

    python tools/report_diff.py OLD.md NEW.md      # exit 0 only if no shared cell moved
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


def compare(old: str, new: str) -> list[str]:
    """Human-readable problems; empty means every shared cell is unchanged."""
    problems = []
    to, tn = tables(old), tables(new)
    if len(to) != len(tn):
        problems.append(f"table count changed: {len(to)} -> {len(tn)}")
    for t, ((ho, ro), (hn, rn)) in enumerate(zip(to, tn), 1):
        where = {}
        for j, h in enumerate(hn):
            where.setdefault(_norm(h), j)
        cols = []
        for i, h in enumerate(ho):
            j = where.get(_norm(h))
            if j is None:
                problems.append(f"table {t}: column '{h}' is gone")
            else:
                cols.append((i, j, h))
        if len(ro) != len(rn):
            problems.append(f"table {t}: {len(ro)} rows -> {len(rn)}")
        for r, (a, b) in enumerate(zip(ro, rn), 1):
            for i, j, h in cols:
                va = a[i] if i < len(a) else ""
                vb = b[j] if j < len(b) else ""
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
    tn = tables(new)
    print(f"VERIFIED: every shared cell unchanged ({len(tn)} tables, "
          f"{sum(len(r) for _, r in tn)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
