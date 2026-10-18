#!/usr/bin/env python3
"""Weakly-labelled crops from single-class clips (Halmstad: one object class per clip, class in the
filename) - more birds and more airframes for the fixed-camera classifier, without the dataset's
MATLAB labels (which this repo cannot decode; see benchmarks/adapters/halmstad.py).

How the weak label is made: the clip is run through the live ``MotionStage``; only frames where the
camera is still are used. Proposals are linked into tracks; a track that has flown (moved) along a
smooth path (small residual of a quadratic-in-time fit) is a flier, and a flier in a clip named
BIRD is a bird. In DRONE clips only the single longest smooth track is taken as the drone (a clip
may also contain a passing bird or clutter; the longest smooth mover is the filmed target).
AIRPLANE/HELICOPTER fliers become "other" (they are not drones, and a bird label would be wrong).
Everything else the motion stage proposes is "other" (subsampled), plus static-blob probes.

    python tools/fixedcam_harvest_weak.py --videos DIR/V_BIRD_0*.mp4 ... --out crops.npz

Output matches ``tools/fixedcam_harvest.py``.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CLASS_OF = {"DRONE": 0, "BIRD": 1, "AIRPLANE": 2, "HELICOPTER": 2}


def link(props_per_frame, gate=25.0, max_gap=3):
    """Greedy nearest-neighbour linking. props_per_frame: list of lists of (x, y, size)."""
    tracks, active = [], []
    for f, props in enumerate(props_per_frame):
        used = set()
        for tr in active:
            lf, lx, ly, _ = tr[-1]
            best, bd = None, gate * (f - lf)
            for j, (x, y, s) in enumerate(props):
                d = math.hypot(x - lx, y - ly)
                if j not in used and d < bd:
                    best, bd = j, d
            if best is not None:
                used.add(best)
                tr.append((f,) + tuple(props[best]))
        for j, p in enumerate(props):
            if j not in used:
                active.append([(f,) + tuple(p)])
        still = []
        for tr in active:
            (tracks if f - tr[-1][0] > max_gap else still).append(tr)
        active = still
    return tracks + active


def smooth_flier(tr, min_n=10, min_move=8.0):
    if len(tr) < min_n:
        return False
    a = np.array(tr, float)
    move = float(np.hypot(a[:, 1] - a[0, 1], a[:, 2] - a[0, 2]).max())
    if move < min_move:
        return False
    t = a[:, 0] - a[:, 0].mean()
    res = [a[:, k] - np.polyval(np.polyfit(t, a[:, k], 2), t) for k in (1, 2)]
    rough = float(np.sqrt(np.mean(res[0] ** 2 + res[1] ** 2)))
    return rough <= max(2.0, 0.3 * float(np.median(a[:, 3])))


def main(argv=None) -> None:
    import cv2
    import torch

    from realtime.fixedcam.motion import MotionStage
    from realtime.fixedcam.net import OFFSETS, crop_size, raw_crops

    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-other", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(a.seed)
    raws, S, lab, off, src = [], [], [], [], []
    stats = {0: 0, 1: 0, 2: 0}
    for vpath in a.videos:
        m = re.search(r"_(DRONE|BIRD|AIRPLANE|HELICOPTER)_", Path(vpath).name.upper())
        if not m:
            continue
        cls = CLASS_OF[m.group(1)]
        # pass 1: proposals on still frames (keep the frames for pass 2 - clips are ~300 x 640x512)
        cap = cv2.VideoCapture(str(vpath))
        ms, props_pf, frames = None, [], []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            gt = torch.from_numpy(g).cuda()
            if ms is None:
                ms = MotionStage(*g.shape)
            props, _ = ms.step(gt)
            props_pf.append([] if ms.moving else [(p.x, p.y, p.size) for p in props])
            frames.append(g)
        tracks = link(props_pf)
        fliers = [t for t in tracks if smooth_flier(t)]
        if cls == 0 and fliers:
            fliers = [max(fliers, key=len)]
        on_flier = {(p[0], round(p[1], 1), round(p[2], 1)) for t in fliers for p in t}
        # pass 2: crops (same temporal channels as the live pipeline)
        per_frame = {}
        for t in fliers:
            for (f, x, y, s) in t:
                per_frame.setdefault(f, []).append((x, y, s, cls, True))
        for f, props in enumerate(props_pf):
            for (x, y, s) in props:
                if (f, round(x, 1), round(y, 1)) not in on_flier and rng.random() < a.keep_other:
                    per_frame.setdefault(f, []).append((x, y, s, 2, False))
        H, W = frames[0].shape
        for f, items in per_frame.items():
            if f < max(OFFSETS):
                continue
            centers, labels, targets, sources = [], [], [], []
            for (x, y, s, c, is_flier) in items:
                cs = crop_size(s)
                centers.append((x, y, cs)); labels.append(c); targets.append((0.0, 0.0)); sources.append(0)
                if is_flier:   # plus a jittered probe crop, as the tracker would request
                    jx, jy = rng.uniform(-0.25, 0.25, 2) * cs
                    centers.append((x + jx, y + jy, cs)); labels.append(c)
                    targets.append((-jx / (cs / 2), -jy / (cs / 2))); sources.append(1)
            fr3 = [torch.from_numpy(frames[f - k]).cuda().half() for k in OFFSETS]
            r = raw_crops(fr3, centers).clamp(0, 255).round().byte().cpu().numpy()
            raws.append(r)
            S.extend(c[2] for c in centers)
            lab.extend(labels)
            off.extend(targets)
            src.extend(sources)
            for c in labels:
                stats[c] += 1
        print(f"{Path(vpath).name}: {len(fliers)} flier track(s) as class {cls}; cum. {stats}", flush=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, raw=np.concatenate(raws), S=np.array(S, np.float32), label=np.array(lab, np.int8),
                        off=np.clip(np.array(off, np.float32), -1, 1).reshape(-1, 2),
                        frame=np.zeros(len(lab), np.int32), source=np.array(src, np.int8))
    print(f"{out.name}: {len(lab)} crops | drone {stats[0]} | bird {stats[1]} | other {stats[2]}")


if __name__ == "__main__":
    main()
