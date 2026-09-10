#!/usr/bin/env python3
"""The campaign's result tables: ours vs the competitor, per dataset, with statistics.

THREE KINDS OF CLAIM, KEPT APART
--------------------------------
1. OURS vs OURS (temporal vs single-frame). One variable -- the input representation --
   everything else identical. A paired test over sequences is meaningful, and this is the
   only comparison in the project that isolates the mechanism.
2. OURS vs THE COMPETITOR (YOLOMG, trained by us). Same videos, splits, labels, evaluator
   and confidence floor; different architectures, each under its own published recipe. A
   paired test is meaningful, and the difference is between two SYSTEMS, not two loss terms.
3. OURS vs PUBLISHED SCALARS. One number from someone else's run, with no distribution
   behind it, so NO p-value is possible and none is printed. Listed to place our result,
   never subtracted from it.

Pairing is seed-matched throughout -- ours-s0 against theirs-s0. Pooling across mismatched
seeds folds seed variance into the effect and inflates it.

    PYTHONPATH=. python tools/make_summary.py --out work/reports/SUMMARY.md
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from benchmarks.fast_bootstrap import (paired_bootstrap_pooled_ap,  # noqa: E402
                                       paired_permutation_pooled_ap)
from dronedet.stats import apply_holm  # noqa: E402
from benchmarks.catalog import ARD_CONDITIONS  # noqa: E402
from benchmarks.scorecard import pooled_ap  # noqa: E402
from dronedet.console import use_utf8_stdio  # noqa: E402
from tools.check_scorecard import load_sequences  # noqa: E402

#: Every scorecard this campaign wrote matches one of these shapes. Anything else is
#: reported as unparsed at the bottom of the page rather than silently dropped -- a table
#: that quietly omits a run is worse than one that admits it could not read it.
_PAT = re.compile(
    r"^(?P<arm>singleframe|temporal|yolomg|tyolov8)[_-](?P<ds>ardmav|nps)"
    r"(?:_seed(?P<s2>\d+)|(?:-(?P<budget>e100|e70))?-s(?P<s1>\d+))$")


def parse_name(name: str):
    """'temporal_ardmav-e100-s1' -> ('ardmav', 'temporal', 'e100', 1)."""
    m = _PAT.match(name)
    if not m:
        return None
    g = m.groupdict()
    seed = int(g["s1"] if g["s1"] is not None else g["s2"])
    return g["ds"], g["arm"], (g["budget"] or "e30"), seed


def _read_scorecard(p: Path) -> dict:
    """Scorecards ship gzipped (623.7 MB raw -> 29.1 MB), so read either form."""
    if p.suffix == ".gz":
        import gzip
        return json.loads(gzip.decompress(p.read_bytes()).decode("utf-8"))
    return json.loads(p.read_text(encoding="utf-8"))


def load_all(scorecard_dir: Path):
    out, unparsed = {}, []
    # Prefer ONE file per scorecard. A working directory can hold both forms -- the
    # cluster's does, 42 .json beside 42 .json.gz -- and globbing both then loads every
    # card twice: harmless for the results (the second load overwrites the same key with
    # identical data) but it doubles the work and made the skipped-card footnote read
    # "24 unparsed" where a fresh clone, which ships only the .gz, sees 12.
    by_stem: dict[str, Path] = {}
    for p in sorted([*scorecard_dir.glob("*.json"), *scorecard_dir.glob("*.json.gz")]):
        stem = p.name[:-3] if p.name.endswith(".gz") else p.name
        by_stem.setdefault(stem, p)          # .json wins only if it sorted first; either
                                             # form parses to the same object, verified
                                             # over all 42 pairs.
    for p in sorted(by_stem.values()):
        # Path('a.json.gz').stem is 'a.json', which the name pattern rejects -- strip the
        # compression suffix before parsing, or every gzipped card reads as unparseable.
        stem = p.stem[:-5] if p.stem.endswith(".json") else p.stem
        key = parse_name(stem)
        if key is None:
            unparsed.append(stem)
            continue
        payload = _read_scorecard(p)
        seqs = load_sequences(payload)
        out[key] = {"seqs": seqs, "ap": pooled_ap(seqs),
                    "split": payload.get("split", "?"),
                    "protocol": payload.get("protocol_key", "?")}
    return out, unparsed


def paired(a_seqs, b_seqs, n_resamples: int, seed: int = 0):
    """Paired test over the sequences the two arms actually share."""
    ia = {s.sequence: s for s in a_seqs}
    ib = {s.sequence: s for s in b_seqs}
    common = sorted(set(ia) & set(ib))
    if len(common) < 2:
        return None
    ua = [ia[k] for k in common]
    ub = [ib[k] for k in common]
    r = paired_bootstrap_pooled_ap(ua, ub, n_resamples=n_resamples, seed=seed)
    r["p_perm"] = paired_permutation_pooled_ap(ua, ub, n_resamples=n_resamples, seed=seed)
    r["n_seq"] = len(common)
    # Both tests must agree. The bootstrap alone will call a small-N difference
    # significant; requiring the permutation test too makes a thin benchmark report
    # "inconclusive" rather than manufacture confidence it has not earned.
    r["significant"] = (r["lo"] > 0 or r["hi"] < 0) and r["p_perm"] < 0.05
    return r


def verdict_of(r) -> str:
    """The printed verdict for one row, from its (Holm-corrected) ``significant``."""
    if not r["significant"]:
        return "no difference"
    return "**better**" if r["observed"] > 0 else "**worse**"


def holm_table(table) -> list[str]:
    """One paired-test table, re-verdicted under Holm across the whole table.

    ``table`` is ``[(cells, name, r), ...]``: the leading markdown cells, a readable name
    for the note under the table, and the result of `paired`. The family is the table --
    docs/research/INFRA.md section 6.1 required that for months before any tool applied
    it, and until it was applied this file printed three NPS "worse" verdicts that do not
    survive it. The uncorrected p keeps its own column and every verdict the correction
    removes is named, so the change is visible rather than silent.
    """
    apply_holm([r for _, _, r in table])
    out, changed = [], []
    for cells, name, r in table:
        out.append(f"| {cells} | {r['observed']:+.3f} | [{r['lo']:+.3f}, {r['hi']:+.3f}] | "
                   f"{r['p']:.4f} | {r['p_perm']:.4f} | {r['p_perm_holm']:.4f} | "
                   f"{verdict_of(r)} |")
        if r["significant_raw"] and not r["significant"]:
            changed.append(f"{name} (uncorrected: {'better' if r['observed'] > 0 else 'worse'})")
    out.append("")
    if table:
        head = f"Holm over the {len(table)} rows of this table"
        out += [f"{head} removed {len(changed)} verdict(s): {'; '.join(changed)}." if changed
                else f"{head} changed no verdict.", ""]
    return out


def gt_scored_lines(rows) -> list[str]:
    """One line per dataset saying how many GT instances each arm is actually scored on.

    Printed rather than asserted. This file used to open by saying every arm was scored on
    "the same ... labels"; on ARD-MAV the YOLOMG scorecards carry 22 fewer instances than
    ours (28,138 against 28,160 -- two per sequence in 11 of the 15 sequences). Immaterial to
    the AP, and it still made the first sentence of the file untrue.
    """
    parts, every = [], set()
    for arm, budget, label in ROWS:
        totals = sorted({sum(q.n_gt for q in v["seqs"])
                         for (a2, b2, _s), v in rows.items() if a2 == arm and b2 == budget})
        if totals:
            shown = _budget_label(arm, budget)
            parts.append(f"{label} ({shown}) {' / '.join(f'{t:,}' for t in totals)}")
            every.update(totals)
    if not every:
        return []
    if len(every) == 1:
        return [f"Every arm is scored on the same {every.pop():,} GT instances.", ""]
    return ["GT instances scored, per arm: " + "; ".join(parts)
            + ". They differ, so they are printed rather than asserted.", ""]


def _budget_label(arm: str, budget: str) -> str:
    """YOLOMG always ran its published 100 epochs; our arms carry the budget in their names."""
    if arm == "yolomg" or budget == "e100":
        return "100 ep"
    return {"e70": "70 ep"}.get(budget, "30 ep")


def fmt_seeds(vals):
    if not vals:
        return "--"
    xs = [vals[s] for s in sorted(vals)]
    mean = st.fmean(xs)
    spread = f" +/- {st.stdev(xs):.3f}" if len(xs) > 1 else ""
    return f"**{mean:.3f}**{spread}  ({', '.join(f'{v:.3f}' for v in xs)})"


ROWS = (("temporal", "e30", "**ours** temporal"),
        ("temporal", "e100", "**ours** temporal"),
        ("singleframe", "e30", "ours single-frame (control)"),
        ("singleframe", "e100", "ours single-frame (control)"),
        ("yolomg", "e30", "**YOLOMG** (competitor)"),
        ("tyolov8", "e70", "Temporal-YOLOv8 (prior art, reimplemented)"))

TESTS = (("temporal", "e30", "singleframe", "e30",
          "ours temporal - ours single-frame (30 ep)"),
         ("temporal", "e100", "singleframe", "e100",
          "ours temporal - ours single-frame (100 ep)"),
         ("temporal", "e30", "yolomg", "e30", "ours temporal 30 ep - YOLOMG"),
         ("temporal", "e100", "yolomg", "e30", "ours temporal 100 ep - YOLOMG"))

#: A SEPARATE Holm family, declared before any of its scorecards existed. Folding it into
#: TESTS would enlarge that table's family and quietly change verdicts it had already
#: published. One question only: a Temporal-YOLOv8-vs-single-frame row was considered and
#: left out -- it differs in network, head, epochs AND window, so it could not attribute any
#: difference to the window, and it would have made this family's correction harsher.
PRIOR_ART_TESTS = (("temporal", "e100", "tyolov8", "e70",
                    "ours temporal 100 ep - Temporal-YOLOv8 70 ep"),)


def main() -> int:
    use_utf8_stdio()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scorecards", type=Path, default=REPO / "work/scorecards")
    ap.add_argument("--reports", type=Path, default=REPO / "work/reports")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--n-resamples", type=int, default=10000)
    a = ap.parse_args()

    cards, unparsed = load_all(a.scorecards)
    if not cards:
        raise SystemExit(f"no parseable scorecards under {a.scorecards}")

    by_ds = defaultdict(dict)
    for (ds, arm, budget, seed), v in cards.items():
        by_ds[ds][(arm, budget, seed)] = v

    L = ["# SpeckLock -- results", "",
         "Every number below was produced by this repository: the same videos, splits and "
         "ground-truth files, the same evaluator, and the same confidence floor (0.001) for every "
         "arm -- though not always the same scored instances, which are printed per dataset. "
         "The competitor is **YOLOMG** (arXiv:2503.07115), trained by us under its own "
         "published recipe -- 100 epochs at 1280 px against our 30 at 640, i.e. roughly "
         "twice our gradient steps.", "",
         ### STAMP THE SETTINGS INTO THE ARTIFACT.
         ###
         ### Every point estimate here is a deterministic function of the scorecards, but
         ### the confidence intervals and p-values are Monte-Carlo and depend on
         ### --n-resamples. An earlier version of this file recorded neither, so
         ### regenerating it produced identical effect sizes and identical verdicts beside
         ### CIs and p-values that differed in the third decimal -- and there was no way to
         ### tell whether that was a settings difference or a real change, which is exactly
         ### the ambiguity a report is supposed to remove.
         f"Generated by `tools/make_summary.py` with `--n-resamples {a.n_resamples}` over "
         f"`{a.scorecards}`. Point estimates are deterministic; the CIs and p-values are "
         f"Monte-Carlo, so reproducing them bit-for-bit needs that same resample count -- "
         f"which is why it is recorded here rather than left to the argparse default.",
         "",
         "Verdicts are **Holm-corrected across each table** -- the family is the whole "
         "table (docs/research/INFRA.md section 6.1). A difference is called only when the "
         "95% CI excludes zero and the Holm-adjusted permutation p is below 0.05; the "
         "uncorrected p is printed beside it, and any verdict the correction removes is "
         "named under its table.",
         ""]

    for ds in sorted(by_ds):
        rows = by_ds[ds]
        sample = next(iter(rows.values()))
        L += [f"## {ds}", "",
              f"Protocol `{sample['protocol']}`, split `{sample['split']}`. Seed-matched "
              f"paired bootstrap **and** permutation over sequences; significant only when "
              f"both agree, after a Holm correction across the table.", "",
              "| arm | budget | AP mean (per seed) |", "|---|---|---|"]

        for arm, budget, label in ROWS:
            sd = {s: v["ap"] for (a2, b2, s), v in rows.items()
                  if a2 == arm and b2 == budget}
            if sd:
                shown = _budget_label(arm, budget)
                L.append(f"| {label} | {shown} | {fmt_seeds(sd)} |")
        L.append("")
        L += gt_scored_lines(rows)

        L += ["### Paired tests, seed-matched", "",
              "| comparison | seed | d AP | 95% CI | p boot | p perm | p perm, Holm | verdict |",
              "|---|---|---|---|---|---|---|---|"]
        seeds = sorted({s for (_a, _b, s) in rows})
        table = []
        for a_arm, a_bud, b_arm, b_bud, title in TESTS:
            for seed in seeds:
                ka, kb = (a_arm, a_bud, seed), (b_arm, b_bud, seed)
                if ka not in rows or kb not in rows:
                    continue
                r = paired(rows[ka]["seqs"], rows[kb]["seqs"], a.n_resamples, seed)
                if r is not None:
                    table.append((f"{title} | {seed}", f"{title}, seed {seed}", r))
        L += holm_table(table)

        prior = []
        for a_arm, a_bud, b_arm, b_bud, title in PRIOR_ART_TESTS:
            for seed in seeds:
                ka, kb = (a_arm, a_bud, seed), (b_arm, b_bud, seed)
                if ka not in rows or kb not in rows:
                    continue
                r = paired(rows[ka]["seqs"], rows[kb]["seqs"], a.n_resamples, seed)
                if r is not None:
                    prior.append((f"{title} | {seed}", f"{title}, seed {seed}", r))
        if prior:
            L += ["### Against the closest prior art: Temporal-YOLOv8, reimplemented", "",
                  "van Leeuwen et al., Sensors 2024: three grayscale frames at t-15, t, t+15, no "
                  "stabilisation, YOLOv8m, 70 epochs, Adam. configs/experiments/prior_art.py lists "
                  "what is reproduced and what is not. Its window reads half a second of the "
                  "future and ours does not. This table is its own Holm family, declared before "
                  "its scorecards existed, so adding it changed no verdict above.", "",
                  "| comparison | seed | d AP | 95% CI | p boot | p perm | p perm, Holm | verdict |",
                  "|---|---|---|---|---|---|---|---|"]
            L += holm_table(prior)


        # --- ARD-MAV only: GLAD's own condition grouping. The overall AP hides the
        # result that matters most to this project: the three conditions differ in target
        # size, and the two systems order DIFFERENTLY on them. Small is where an 11.8 px
        # median drops to a few pixels -- the stated problem -- and it is the condition
        # where the temporal arm leads both the competitor and GLAD's published 0.580.
        if ds == "ardmav":
            L += ["### By condition (GLAD's grouping: 5 sequences each)", "",
                  "| arm | ordinary | complex | small |", "|---|---|---|---|",
                  "| GLAD (published, for placement only) | 0.910 | 0.810 | 0.580 |"]
            for arm, budget, label in ROWS:
                per_cond = {}
                for cond in ("ordinary", "complex", "small"):
                    vals = []
                    for (a2, b2, s), v in sorted(rows.items()):
                        if a2 != arm or b2 != budget:
                            continue
                        sub = [q for q in v["seqs"]
                               if ARD_CONDITIONS.get(q.sequence, ()) == (cond,)]
                        if sub:
                            vals.append(pooled_ap(sub))
                    if vals:
                        per_cond[cond] = st.fmean(vals)
                if per_cond:
                    shown = _budget_label(arm, budget)
                    L.append(f"| {label} ({shown}) | "
                             + " | ".join(f"{per_cond.get(c, float('nan')):.3f}"
                                          for c in ("ordinary", "complex", "small")) + " |")
            L.append("")

            L += ["#### Paired test on the SMALL condition, ours (100 ep) vs YOLOMG", "",
                  "> Five sequences admit only 2^5 = 32 sign patterns, so the permutation "
                  "p cannot go below 1/33 = 0.0303 however large the effect. Printed so "
                  "the floor is not mistaken for strength of evidence.", "",
                  "| seed | d AP | 95% CI | p boot | p perm | p perm, Holm | verdict |",
                  "|---|---|---|---|---|---|---|"]
            small = []
            for seed in sorted({s for (_a, _b, s) in rows}):
                ka, kb = ("temporal", "e100", seed), ("yolomg", "e30", seed)
                if ka not in rows or kb not in rows:
                    continue
                sa = sorted([q for q in rows[ka]["seqs"]
                             if ARD_CONDITIONS.get(q.sequence, ()) == ("small",)],
                            key=lambda q: q.sequence)
                sb = sorted([q for q in rows[kb]["seqs"]
                             if ARD_CONDITIONS.get(q.sequence, ()) == ("small",)],
                            key=lambda q: q.sequence)
                r = paired(sa, sb, a.n_resamples, seed)
                if r is not None:
                    small.append((f"{seed}", f"seed {seed}", r))
            L += holm_table(small)

    local = sorted(a.reports.glob("local*_seed*.md")) if a.reports.exists() else []
    if local:
        L += ["## The project's own videos", "",
              "One held-out flight, so the interval there is a **moving-block bootstrap "
              "over 30-frame blocks WITHIN one sequence**: stability across that flight's "
              "segments, not generalisation to another flight. Per-seed tables:", ""]
        L += [f"- `{p.name}`" for p in local]
        L.append("")

    if unparsed:
        L += [f"_Unparsed scorecards ({len(unparsed)}): {', '.join(unparsed[:8])}_", ""]

    out = "\n".join(L) + "\n"
    print(out)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(out, encoding="utf-8")
        print(f"-> {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
