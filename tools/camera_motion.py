#!/usr/bin/env python3
"""How much does the camera move? Measured the same way on every dataset a claim is made about.

WHY THIS EXISTS
---------------
The README described SpeckLock as a method for "a moving camera". The two videos behind its
headline numbers turned out to be a static rig to sub-pixel precision (`10_06` drifts 0.76 px
in x and 1.07 px in y across the whole clip), while ARD-MAV and NPS -- the benchmarks where the
temporal stack does NOT separate from a single frame -- had never been measured at all. A claim
about camera motion should rest on a measurement of camera motion, taken identically on every
dataset it is made about. This is that measurement.

WHAT IS MEASURED
----------------
Global image translation between CONSECUTIVE frames, by phase correlation on the grayscale
frame (`cv2.phaseCorrelate` with a Hanning window) -- the estimator `dronedet.stabilize` uses in
'translation' mode, but frame-to-frame rather than against frame 0, so an accumulated drift
cannot make one hard frame look like a jump. Per video and pooled per dataset:

  step_px       |(dx, dy)| between frame t-1 and t: camera speed in pixels per frame
  aperture_px   |sum of the steps over the last A frames|: how far the background moves
                across a temporal stack's window. A = 12 for SpeckLock's t-12/t-6/t;
                A = 30 for Temporal-YOLOv8's t-15/t/t+15.
  response      phase-correlation peak response, and the fraction of steps below the
                0.35 that Stabilizer itself uses to decide a registration is trustworthy

It is a LOWER BOUND on camera motion: translation only, so rotation, zoom and parallax are not
in it, and a large moving object can pull the estimate toward its own motion. The response
columns are there so a reader can see how much of each number to trust.

    PYTHONPATH=. python tools/camera_motion.py --dataset ardmav \\
        --gt-dir work/ext_datasets/gt/ardmav --video-root data/external/ard_mav/ARD-MAV/videos
    PYTHONPATH=. python tools/camera_motion.py --dataset local \\
        --pair data/videos/10_06.mp4 realtime/work/gt_1006_v2.json \\
        --pair data/videos/07_05.mp4 work/gt_user.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

APERTURES = (12, 30)   # t-12/t-6/t spans 12 frames; Temporal-YOLOv8's t-15/t/t+15 spans 30
LOW_RESPONSE = 0.35    # dronedet.stabilize.Stabilizer's own threshold for a trusted registration
VIDEO_EXTS = (".mp4", ".avi", ".MP4", ".mov")


def measure_frames(frames):
    """Frame-to-frame translation over an iterable of BGR or gray frames.

    Returns ``(steps, response, shape)``: an (N-1, 2) array of per-step (dx, dy) in pixels,
    the matching phase-correlation responses, and the (h, w) of the frames.
    """
    import cv2

    steps, resp = [], []
    prev = window = shape = None
    for f in frames:
        g = (cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f).astype(np.float32)
        if prev is None:
            prev, shape = g, g.shape
            window = cv2.createHanningWindow(g.shape[::-1], cv2.CV_32F)
            continue
        (dx, dy), r = cv2.phaseCorrelate(prev, g, window)
        steps.append((dx, dy))
        resp.append(r)
        prev = g
    return (np.asarray(steps, dtype=np.float64).reshape(-1, 2),
            np.asarray(resp, dtype=np.float64), shape)


def _q(a, q):
    return float(np.quantile(a, q)) if len(a) else float("nan")


def _dist(a) -> dict:
    return {"median": _q(a, 0.5), "p95": _q(a, 0.95),
            "max": float(np.max(a)) if len(a) else float("nan")}


def aperture_displacements(steps, a: int) -> np.ndarray:
    """|net background displacement| over every window of `a` consecutive steps."""
    if len(steps) < a:
        return np.zeros(0)
    cum = np.vstack([[0.0, 0.0], np.cumsum(steps, axis=0)])
    d = cum[a:] - cum[:-a]
    return np.hypot(d[:, 0], d[:, 1])


def stats(steps, resp, shape, apertures=APERTURES) -> dict:
    mag = np.hypot(steps[:, 0], steps[:, 1]) if len(steps) else np.zeros(0)
    return {
        "n_frames": int(len(steps) + (1 if shape else 0)),
        "resolution": [int(shape[1]), int(shape[0])] if shape else None,
        "step_px": _dist(mag),
        "aperture_px": {str(a): _dist(aperture_displacements(steps, a)) for a in apertures},
        "response": {"median": _q(resp, 0.5),
                     "frac_low": float((resp < LOW_RESPONSE).mean()) if len(resp) else float("nan")},
    }


def pooled(arrays, apertures=APERTURES) -> dict:
    """Every step and every window of every video, pooled -- weighted by frames, not videos."""
    mags = [np.hypot(s[:, 0], s[:, 1]) for s, _ in arrays if len(s)]
    resp = [r for _, r in arrays if len(r)]
    allm = np.concatenate(mags) if mags else np.zeros(0)
    allr = np.concatenate(resp) if resp else np.zeros(0)
    return {
        "n_steps": int(len(allm)),
        "step_px": _dist(allm),
        "aperture_px": {str(a): _dist(np.concatenate([aperture_displacements(s, a)
                                                       for s, _ in arrays] or [np.zeros(0)]))
                        for a in apertures},
        "response": {"median": _q(allr, 0.5),
                     "frac_low": float((allr < LOW_RESPONSE).mean()) if len(allr) else float("nan")},
    }


def target_sizes(gt_path: Path) -> list[float]:
    """sqrt(w*h) of every non-ignored labelled box: the size the motion is compared against."""
    from dronedet.gt import GroundTruth

    gt = GroundTruth.load(gt_path)
    return [math.sqrt(b[2] * b[3]) for o in gt.objects.values() if not o.ignore
            for b in o.frames.values()]


def resolve_items(a) -> list[tuple[Path, Path]]:
    items = [(Path(v), Path(g)) for v, g in a.pair]
    if a.gt_dir:
        for g in sorted(a.gt_dir.glob("*.json")):
            v = next((a.video_root / (g.stem + e) for e in VIDEO_EXTS
                      if (a.video_root / (g.stem + e)).is_file()), None)
            if v is None:
                # Never skip silently: a dataset measured on the videos that happened to be
                # present is a different dataset from the one the claim is about.
                raise SystemExit(f"no video for {g.stem} under {a.video_root}")
            items.append((v, g))
    return items


def render_md(r: dict) -> str:
    A = [str(a) for a in r["apertures"]]
    L = [f"# Camera motion -- {r['dataset']}", "",
         "Frame-to-frame global translation by phase correlation (a lower bound on camera "
         "motion: translation only). *step* = pixels per frame; *window A* = net background "
         f"displacement across A frames; *low-resp.* = share of steps with response < {LOW_RESPONSE}.",
         "",
         "| video | frames | resolution | step median | step p95 | step max | "
         + " | ".join(f"window {a} median | window {a} p95" for a in A)
         + " | response median | low-resp. |",
         "|" + "---|" * (6 + 2 * len(A) + 2)]

    def row(name, n, res, d):
        return (f"| {name} | {n} | {res} | {d['step_px']['median']:.3f} | {d['step_px']['p95']:.3f} | "
                f"{d['step_px']['max']:.3f} | "
                + " | ".join(f"{d['aperture_px'][a]['median']:.3f} | {d['aperture_px'][a]['p95']:.3f}"
                             for a in A)
                + f" | {d['response']['median']:.3f} | {100 * d['response']['frac_low']:.1f} % |")

    for name, d in r["videos"].items():
        res = "x".join(str(x) for x in d["resolution"]) if d["resolution"] else "?"
        L.append(row(name, d["n_frames"], res, d))
    L.append(row("**pooled**", r["pooled"]["n_steps"] + len(r["videos"]), "", r["pooled"]))
    t = r["target_px_median"]
    w = r["pooled"]["aperture_px"][A[0]]["median"]
    L += ["", f"Median labelled target: **{t:.1f} px**. Across a {A[0]}-frame window the "
          f"background moves a median **{w:.2f} px** -- {w / t:.2f} of the target's size.", ""]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--gt-dir", type=Path, help="one GT json per video; names the videos")
    ap.add_argument("--video-root", type=Path)
    ap.add_argument("--pair", nargs=2, action="append", default=[], metavar=("VIDEO", "GT"))
    ap.add_argument("--apertures", type=int, nargs="+", default=list(APERTURES))
    ap.add_argument("--limit-frames", type=int, default=0, help="first N frames only (smoke)")
    ap.add_argument("--out", type=Path, default=REPO / "work/reports/camera_motion")
    a = ap.parse_args()

    from dronedet.video import frames as vframes

    items = resolve_items(a)
    if not items:
        raise SystemExit("nothing to measure: give --gt-dir/--video-root or --pair")
    per, arrays, sizes = {}, [], []
    for v, g in items:
        it = (f for _, f in vframes(str(v), stop=a.limit_frames or None))
        steps, resp, shape = measure_frames(it)
        per[v.stem] = stats(steps, resp, shape, a.apertures)
        arrays.append((steps, resp))
        sizes += target_sizes(g)
        d = per[v.stem]
        print(f"  {v.stem}: {d['n_frames']} frames, step median {d['step_px']['median']:.3f} px, "
              f"window-{a.apertures[0]} median {d['aperture_px'][str(a.apertures[0])]['median']:.3f} px",
              flush=True)
    r = {"dataset": a.dataset, "apertures": a.apertures, "low_response": LOW_RESPONSE,
         "method": "cv2.phaseCorrelate, consecutive grayscale frames, Hanning window, full resolution",
         "target_px_median": float(np.median(sizes)) if sizes else float("nan"),
         "pooled": pooled(arrays, a.apertures), "videos": per}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{a.dataset}.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
    (a.out / f"{a.dataset}.md").write_text(render_md(r), encoding="utf-8")
    print(render_md(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
