#!/usr/bin/env python3
"""A/B of `trackclass.classify_tracks(allow_motion=...)`, measured through final/run_final.py.

WHY THIS EXISTS
---------------
`allow_motion=True` -- the default, and what `final/run_final.py` and
`tools/tracks_to_dets.py --classify` both use -- counts every detection whose label contains
'motion' (score >= 0.35) as drone evidence. In PC-MAX's detections a bare 'motion' label is
what `dronedet/methods/hybrid2.py` emits when the 640 verifier looked at a motion proposal
and found neither a drone nor a bird, i.e. it is the verifier's *non*-confirmation. EDGE-RT's
single net emits no 'motion' labels at all. This tool measures what the flag does to the
numbers an operator sees, per profile and video, instead of arguing about it.

HOW IT MEASURES
---------------
`final.run_final.main()` is executed **unmodified**. Only two of its steps are substituted:
step 1 (detection) copies a given detection JSON, and step 2 (tracker) copies a given track
JSON -- or, with `--retrack`, runs today's tracker on the detections. Steps 3-5
(classification, track-integrated detections, alarm summary) are run_final's own code. The
False arm flips the *default* of `classify_tracks` / `classify_files`, which is the only way
to reach step 4's `tools/tracks_to_dets.py` subprocess without editing run_final; that script
is therefore run in-process. Arms:

    on            allow_motion=True,  LONG_TRACK as shipped      (today's run_final)
    off           allow_motion=False, LONG_TRACK as shipped
    off_nobypass  allow_motion=False, no LONG_TRACK promotion    (the rule that reproduces
                                                                  every committed round-3
                                                                  tracked-*.json file)

Per arm it reports: target recovery (track level, `tools/track_level_birds.py`; and frame
level from run_final's `tracked_dets.json`), alarm latency (the classifier's decision on the
track prefix, and run_final's own `alarms.txt` line), clutter and bird tracks raised, tracks
promoted by LONG_TRACK alone, and the number of raised tracks `alarms.txt` itself calls
"not confirmed".

    # the committed round-3 artifacts (default cases)
    python tools/allow_motion_ab.py --out work/reports/allow_motion/committed
    # same detections, today's tracker
    python tools/allow_motion_ab.py --retrack --out work/reports/allow_motion/retrack
    # a run_final.py output directory (dets.json + tracks.json), e.g. the shipped package
    python tools/allow_motion_ab.py --case pc-max 10_06 out/dets.json out/tracks.json \\
        --out work/reports/allow_motion/shipped
    # which rule produced the committed round-3 tracked files
    python tools/allow_motion_ab.py --provenance --out work/reports/allow_motion/provenance
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import dronedet.track as TRK  # noqa: E402
import dronedet.trackclass as TC  # noqa: E402
import final.run_final as RF  # noqa: E402
from dronedet.console import use_utf8_stdio  # noqa: E402
from dronedet.detections import DetectionSet  # noqa: E402
from dronedet.evaluate import evaluate, report  # noqa: E402
from dronedet.gt import GroundTruth  # noqa: E402

DEFAULT_CASES = [
    ("pc-max", "07_05", "work/det3/0705/pc-max.json", "work/tracks3/0705/pc-max-all.json"),
    ("pc-max", "10_06", "work/det3/1006/pc-max.json", "work/tracks3/1006/pc-max-all.json"),
    ("edge-rt", "07_05", "realtime/work/out3/0705/rt-c-full1280/dets.json",
     "realtime/work/out3/0705/rt-c-full1280/tracks-all.json"),
    ("edge-rt", "10_06", "realtime/work/out3/1006/rt-c-full1280/dets.json",
     "realtime/work/out3/1006/rt-c-full1280/tracks-all.json"),
]
# the round-3 tracked files the default cases back (causal, smoothed); compared value-for-value
PUBLISHED = {
    ("pc-max", "07_05"): ("work/det3/0705/tracked-pcmax.json", "work/det3/0705/tracked-pcmax-smooth.json"),
    ("pc-max", "10_06"): ("work/det3/1006/tracked-pcmax.json", "work/det3/1006/tracked-pcmax-smooth.json"),
    ("edge-rt", "07_05"): ("realtime/work/out3/0705/rt-c-full1280/tracked-rt-c-full1280.json",
                           "realtime/work/out3/0705/rt-c-full1280/tracked-smooth.json"),
    ("edge-rt", "10_06"): ("realtime/work/out3/1006/rt-c-full1280/tracked-rt-c-full1280.json", None),
}
GT = {"07_05": "work/gt_user.json", "10_06": "realtime/work/gt_1006_v2.json"}
VAL = {"07_05": (342, 571)}          # the round-3 07_05 validation split
LONG_TRACK0 = TC.LONG_TRACK
ARMS = [("on", True, LONG_TRACK0), ("off", False, LONG_TRACK0), ("off_nobypass", False, None)]

_real_run = subprocess.run


def _run_tracks_to_dets(argv: list[str]) -> None:
    old = sys.argv
    sys.argv = ["tracks_to_dets.py", *argv]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            runpy.run_path(str(REPO / "tools/tracks_to_dets.py"), run_name="__main__")
    finally:
        sys.argv = old


def _inproc_run(cmd, *a, **kw):
    if len(cmd) > 1 and str(cmd[1]).replace("\\", "/").endswith("tools/tracks_to_dets.py"):
        _run_tracks_to_dets([str(c) for c in cmd[2:]])
        return subprocess.CompletedProcess(cmd, 0)
    return _real_run(cmd, *a, **kw)


@contextlib.contextmanager
def arm_patches(dets_src: Path, tracks_src: Path, retrack: bool, motion: bool,
                long_track: int | None):
    real_tracker, real_edge, real_pcmax = TRK.run_tracker_file, RF.run_edge, RF.run_pcmax

    def fake_detect(_video, out):
        shutil.copyfile(dets_src, Path(out) / "dets.json")

    def tracker(video, dets, out, **kw):
        if retrack:
            with contextlib.redirect_stdout(io.StringIO()):
                return real_tracker(video, dets, out, **kw)
        shutil.copyfile(tracks_src, out)

    RF.run_edge = RF.run_pcmax = fake_detect
    TRK.run_tracker_file = tracker
    subprocess.run = _inproc_run
    TC.classify_tracks.__defaults__ = (motion,)
    TC.classify_files.__defaults__ = (motion,)
    TC.LONG_TRACK = long_track if long_track is not None else 10 ** 9
    try:
        yield
    finally:
        RF.run_edge, RF.run_pcmax, TRK.run_tracker_file = real_edge, real_pcmax, real_tracker
        subprocess.run = _real_run
        TC.classify_tracks.__defaults__ = (True,)
        TC.classify_files.__defaults__ = (True,)
        TC.LONG_TRACK = LONG_TRACK0


def run_final_steps(profile: str, video: str, out: Path) -> str:
    old = sys.argv
    sys.argv = ["run_final.py", "--video", str(REPO / f"data/videos/{video}.mp4"),
                "--profile", profile, "--out", str(out), "--no-video"]
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            RF.main()
    finally:
        sys.argv = old
    # run_final writes only the coast-smoothed (offline) file; add the causal one, same classifier
    _run_tracks_to_dets(["--tracks", str(out / "tracks.json"), "--src", str(out / "dets.json"),
                         "--out", str(out / "tracked_dets_causal.json"), "--classify"])
    return buf.getvalue()


def track_ids(path: Path) -> list[int]:
    d = json.loads(path.read_text(encoding="utf-8"))
    return sorted({int(m.group(1)) for v in d["frames"].values() for x in v
                   if (m := re.match(r"track(\d+):", x[5]))})


def score(gt: GroundTruth, path: Path, fr=None) -> dict:
    """The round-3 scorer (`dronedet.evaluate`, tau 12) -- what every published tracked row
    used -- plus the threshold-free view: every emitted tracked box counts."""
    ds = DetectionSet.load(str(path))
    r = report(gt, ds, 12.0, frame_range=fr)
    ev = evaluate(gt, ds, 12.0, objects={"far"}, frame_range=fr)
    tp = sum(x[1] for x in ev["records"])
    fp = len(ev["records"]) - tp
    return {"ap_far": round(r.ap_far, 4), "best_f1": round(r.best_f1, 4),
            "recall_any": round(tp / max(ev["n_gt"], 1), 4), "fp_boxes_any": fp,
            "n_gt": ev["n_gt"]}


def prefix_decision(tr: dict, dets: dict, motion: bool, long_track: int | None):
    """First frame at which classify_tracks' rule holds on the track *so far* -- the frame an
    online classifier would raise it. Mirrors classify_tracks line for line."""
    det_frames = {int(f): v for f, v in dets["frames"].items()}
    nc = nm = nt = 0
    for f in sorted(int(k) for k in tr["frames"]):
        v = tr["frames"][str(f)]
        if v[4] != "tracked":
            continue
        nt += 1
        best, bd = None, TC.MATCH_DIST
        for d in det_frames.get(f, []):
            dist = math.hypot((d[0] + d[2]) / 2 - v[0], (d[1] + d[3]) / 2 - v[1])
            if dist < bd:
                bd, best = dist, d
        if best is not None:
            nm += 1
            if ((best[5].startswith("drone") and best[4] >= TC.DRONE_SCORE)
                    or (motion and "motion" in best[5] and best[4] >= 0.35)):
                nc += 1
        if nc / max(nm, 1) >= TC.CONF_FRAC and nc >= TC.N_CONF:
            return f, "confirmed"
        if long_track is not None and nt >= long_track:
            return f, "sustained"
    return None, None


def birds(gt_path: str, out: Path, motion: bool) -> dict:
    argv = ["track_level_birds.py", "--gt", str(REPO / gt_path), "--tracks", str(out / "tracks.json"),
            "--dets", str(out / "dets.json"), "--out", str(out / "birds.md")]
    if not motion:
        argv.append("--no-motion")
    old = sys.argv
    sys.argv = argv
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            runpy.run_path(str(REPO / "tools/track_level_birds.py"), run_name="__main__")
    except SystemExit:
        pass
    finally:
        sys.argv = old
    return json.loads((out / "birds.json").read_text(encoding="utf-8"))


def frames_equal(a: Path, b: Path) -> bool:
    fa = json.loads(a.read_text(encoding="utf-8"))["frames"]
    fb = json.loads(b.read_text(encoding="utf-8"))["frames"]
    return fa == fb


def measure(profile, video, dets_src, tracks_src, retrack, published, tmp: Path) -> list[dict]:
    gt = GroundTruth.load(REPO / GT[video])
    excl = set(gt.meta.get("exclude_frames", []))
    gt_first = min(f for f in gt.objects["far"].frames if f not in excl)
    rows = []
    for arm, motion, long_track in ARMS:
        out = tmp / f"{profile}_{video}_{arm}"
        out.mkdir(parents=True)
        with arm_patches(dets_src, tracks_src, retrack, motion, long_track):
            run_final_steps(profile, video, out)
            b = birds(GT[video], out, motion)
            dets = json.loads((out / "dets.json").read_text(encoding="utf-8"))
            raw = json.loads((out / "tracks.json").read_text(encoding="utf-8"))
            kept = json.loads((out / "tracks_drone.json").read_text(encoding="utf-8"))
            kinds = {r["id"]: r for r in b["tracks"]}
            online = {t["id"]: prefix_decision(t, dets, motion, long_track) for t in raw["tracks"]}
        kept_ids = sorted(t["id"] for t in kept["tracks"])
        alarms = (out / "alarms.txt").read_text(encoding="utf-8").strip().splitlines()
        alarm_line = {int(m.group(1)): ln for ln in alarms if (m := re.match(r"track (\d+) ", ln))}
        raised = []
        for tid in kept_ids:
            r, (f, clause) = kinds[tid], online[tid]
            start = min(int(k) for k in next(t for t in raw["tracks"] if t["id"] == tid)["frames"])
            raised.append({"id": tid, "cls": r["pred"], "gt_kind": r["gt_kind"],
                           "n_frames": r["n_frames"], "n_tracked": r["n_tracked"],
                           "conf_frac": r["conf_frac"], "n_conf": r["n_conf"],
                           "sustained_only": r["sustained_only"], "start": start,
                           "prefix_decision_frame": f, "prefix_decision_clause": clause,
                           "alarms_txt": alarm_line.get(tid, "")})
        target = [x for x in raised if x["gt_kind"] == "target"]
        t_alarm = min((x["prefix_decision_frame"] for x in target
                       if x["prefix_decision_frame"] is not None), default=None)
        rf_latency = [int(m.group(1)) for x in target
                      if (m := re.search(r"latency (\d+) frames", x["alarms_txt"]))]
        online_fa = sorted((f, tid) for tid, (f, _c) in online.items()
                           if f is not None and kinds[tid]["gt_kind"] in ("nothing", "ambiguous"))
        sc = {"causal": score(gt, out / "tracked_dets_causal.json"),
              "smoothed": score(gt, out / "tracked_dets.json")}
        if video in VAL:
            sc["causal_val"] = score(gt, out / "tracked_dets_causal.json", VAL[video])
            sc["smoothed_val"] = score(gt, out / "tracked_dets.json", VAL[video])
        op, cm = b["operating_point"], b["confusion_matrix"]
        row = {
            "profile": profile, "video": video, "arm": arm, "allow_motion": motion,
            "long_track": long_track, "tracker": "rerun" if retrack else "committed",
            "n_confirmed_tracks": len(raw["tracks"]), "kept_track_ids": kept_ids,
            "target_objects_recovered": op["track_recall_over_objects"],
            "clutter_tracks_raised": op["clutter_false_alarms"],
            "bird_tracks_raised": op["bird_false_alarms"],
            "bird_tracks_formed": sum(cm[p]["bird"] for p in cm),
            "promoted_by_long_track_only": b["long_track_bypass"]["promoted_by_sustained_only"],
            "raised_but_alarms_txt_not_confirmed": sum(1 for x in raised
                                                       if x["alarms_txt"].endswith("not confirmed")),
            "gt_first_target_frame": gt_first,
            "target_alarm_frame": t_alarm,
            "target_alarm_latency_from_gt_first": None if t_alarm is None else t_alarm - gt_first,
            "run_final_alarm_latency_from_track_start": rf_latency,
            "online_clutter_alarms": [{"frame": f, "id": tid} for f, tid in online_fa],
            "scores": sc, "raised": raised,
            "confusion_matrix": cm,
        }
        if published:
            row["matches_published_tracked_files"] = {
                "causal": frames_equal(out / "tracked_dets_causal.json", REPO / published[0]),
                "smoothed": (frames_equal(out / "tracked_dets.json", REPO / published[1])
                             if published[1] else None)}
        rows.append(row)
    return rows


# every committed tracked-*.json whose (tracks, dets) inputs are also committed
PROVENANCE_FILES = [
    ("work/tracks3/0705/pc-max-all.json", "work/det3/0705/pc-max.json", "work/det3/0705/tracked-pcmax.json"),
    ("work/tracks3/1006/pc-max-all.json", "work/det3/1006/pc-max.json", "work/det3/1006/tracked-pcmax.json"),
    ("work/tracks3/0705/pc-max-ms15.json", "work/det3/0705/pc-max.json", "work/det3/0705/tracked-pcmax-ms15.json"),
    ("work/tracks3/0705/moe3-v3-all.json", "work/det3/0705/moe3-v3.json", "work/det3/0705/tracked-moe3v3.json"),
    ("work/tracks3/1006/moe3-v3-all.json", "work/det3/1006/moe3-v3.json", "work/det3/1006/tracked-moe3v3.json"),
    ("work/tracks3/0705/moe3-v3-cpu-all.json", "work/det3/0705/moe3-v3-cpu.json",
     "work/det3/0705/tracked-moe3v3-cpu.json"),
    ("work/tracks2/moe3-stacked-all.json", "work/det2/moe3-stacked.json", "work/det2/tracked-moe3-cls.json"),
    ("realtime/work/out3/0705/rt-c-full1280/tracks-all.json", "realtime/work/out3/0705/rt-c-full1280/dets.json",
     "realtime/work/out3/0705/rt-c-full1280/tracked-rt-c-full1280.json"),
    ("realtime/work/out3/1006/rt-c-full1280/tracks-all.json", "realtime/work/out3/1006/rt-c-full1280/dets.json",
     "realtime/work/out3/1006/rt-c-full1280/tracked-rt-c-full1280.json"),
    ("realtime/work/out3/0705/rt-d-full640/tracks-all.json", "realtime/work/out3/0705/rt-d-full640/dets.json",
     "realtime/work/out3/0705/rt-d-full640/tracked-rt-d-full640.json"),
    ("realtime/work/out3/1006/rt-d-full640/tracks-all.json", "realtime/work/out3/1006/rt-d-full640/dets.json",
     "realtime/work/out3/1006/rt-d-full640/tracked-rt-d-full640.json"),
]


def provenance() -> str:
    """Which classifier rule produced the committed tracked files? Each candidate rule is
    applied to the committed (tracks, dets) and its kept-track set compared with the track ids
    present in the committed tracked file."""
    import itertools

    def load(p):
        return json.loads((REPO / p).read_text(encoding="utf-8"))

    def evidence(tracks, dets, drone_score, motion):
        det_frames = {int(f): v for f, v in dets["frames"].items()}
        out = {}
        for tr in tracks["tracks"]:
            nc = nm = nt = 0
            for f, v in tr["frames"].items():
                if v[4] != "tracked":
                    continue
                nt += 1
                best, bd = None, TC.MATCH_DIST
                for d in det_frames.get(int(f), []):
                    dist = math.hypot((d[0] + d[2]) / 2 - v[0], (d[1] + d[3]) / 2 - v[1])
                    if dist < bd:
                        bd, best = dist, d
                if best is None:
                    continue
                nm += 1
                if ((best[5].startswith("drone") and best[4] >= drone_score)
                        or (motion and "motion" in best[5] and best[4] >= 0.35)):
                    nc += 1
            out[tr["id"]] = (nc / max(nm, 1), nc, nt)
        return out

    rules = list(itertools.product([0.35, 0.5], [0.5, 0.7], [True, False], [120, None]))
    hits = {r: [] for r in rules}
    for t, d, o in PROVENANCE_FILES:
        T, D = load(t), load(d)
        want = track_ids(REPO / o)
        cache = {}
        for r in rules:
            score_thr, frac, motion, lt = r
            ev = cache.setdefault((score_thr, motion), evidence(T, D, score_thr, motion))
            keep = sorted(i for i, (cf, nc, nt) in ev.items()
                          if (cf >= frac and nc >= TC.N_CONF) or (lt is not None and nt >= lt))
            hits[r].append(keep == want)
    L = ["# Which classifier rule produced the committed tracked files?", "",
         f"{len(PROVENANCE_FILES)} committed `tracked-*.json` files whose track and detection "
         "inputs are also committed. For each candidate rule (N_CONF = 8 throughout), the "
         "kept-track set is compared with the track ids present in the committed file.", "",
         "| drone score | conf_frac | motion clause | LONG_TRACK | files reproduced |",
         "|---|---|---|---|---|"]
    for r in sorted(rules, key=lambda r: -sum(hits[r])):
        L.append(f"| {r[0]} | {r[1]} | {'yes' if r[2] else 'no'} | {r[3] or 'none'} | "
                 f"{sum(hits[r])} / {len(PROVENANCE_FILES)} |")
    L += ["", "Files: " + ", ".join(f"`{o}`" for _t, _d, o in PROVENANCE_FILES)]
    return "\n".join(L) + "\n"


def to_markdown(rows: list[dict], title: str) -> str:
    L = [f"# allow_motion A/B -- {title}", "",
         "Generated by `tools/allow_motion_ab.py`; run_final.py steps 2-5, round-3 scorer "
         "(tau 12). Latency = frames from the first verified GT frame to the frame the "
         "classifier rule first holds on the target track's prefix; `alarms.txt` latency is "
         "run_final's own line (from track start).", "",
         "| profile | video | arm | tracks raised | target recovered | clutter tracks | bird tracks "
         "(formed) | LONG_TRACK-only | `not confirmed` in alarms.txt | target alarm latency "
         "(alarms.txt) | causal AP / F1 | smoothed AP / F1 | FP boxes, any score | "
         "matches published file |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        c, s = r["scores"]["causal"], r["scores"]["smoothed"]
        pub = r.get("matches_published_tracked_files")
        pub_s = "-" if pub is None else ("yes" if pub["causal"] and pub["smoothed"] in (True, None)
                                         else "no")
        L.append(
            f"| {r['profile']} | {r['video']} | {r['arm']} | {len(r['kept_track_ids'])} | "
            f"{r['target_objects_recovered']:.0%} | {r['clutter_tracks_raised']} | "
            f"{r['bird_tracks_raised']} ({r['bird_tracks_formed']}) | "
            f"{r['promoted_by_long_track_only']} | {r['raised_but_alarms_txt_not_confirmed']} | "
            f"{r['target_alarm_latency_from_gt_first']} "
            f"({', '.join(map(str, r['run_final_alarm_latency_from_track_start'])) or '-'}) | "
            f"{c['ap_far']:.3f} / {c['best_f1']:.3f} | {s['ap_far']:.3f} / {s['best_f1']:.3f} | "
            f"{s['fp_boxes_any']} | {pub_s} |")
    return "\n".join(L) + "\n"


def main() -> int:
    use_utf8_stdio()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", nargs=4, action="append",
                    metavar=("PROFILE", "VIDEO", "DETS", "TRACKS"),
                    help="profile (pc-max|edge-rt), video (07_05|10_06), detection JSON, track "
                         "JSON; repeatable. Default: the committed round-3 artifacts.")
    ap.add_argument("--retrack", action="store_true",
                    help="run today's tracker on the detections instead of copying TRACKS")
    ap.add_argument("--provenance", action="store_true",
                    help="instead: find which classifier rule reproduces the committed "
                         "tracked-*.json files (writes <out>.md)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", required=True, type=Path, help="writes <out>.json and <out>.md")
    a = ap.parse_args()

    if a.provenance:
        md = provenance()
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.with_suffix(".md").write_text(md, encoding="utf-8")
        print(md)
        return 0

    cases = [tuple(c) for c in a.case] if a.case else DEFAULT_CASES
    rows = []
    with tempfile.TemporaryDirectory(prefix="allow_motion_ab_") as td:
        for profile, video, dets, tracks in cases:
            pub = PUBLISHED.get((profile, video)) if not a.case and not a.retrack else None
            rows += measure(profile, video, Path(dets).resolve(), Path(tracks).resolve(),
                            a.retrack, pub, Path(td))
    def where(p: str) -> str:            # repo-relative, or just the run-dir/name of an outside file
        q = Path(p).resolve()
        try:
            return q.relative_to(REPO).as_posix()
        except ValueError:
            return f"{q.parent.name}/{q.name}"

    sources = [{"profile": p, "video": v, "dets": where(d), "tracks": where(t)}
               for p, v, d, t in cases]
    payload = {"sources": sources, "retrack": a.retrack,
               "constants": {"CONF_FRAC": TC.CONF_FRAC, "N_CONF": TC.N_CONF,
                             "LONG_TRACK": LONG_TRACK0, "DRONE_SCORE": TC.DRONE_SCORE},
               "rows": rows}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md = to_markdown(rows, a.title or ("re-tracked" if a.retrack else "committed tracks"))
    a.out.with_suffix(".md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
