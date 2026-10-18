#!/usr/bin/env python3
"""Harvest fixed-camera-style classifier crops from a MOVING-camera labelled video (ARD-MAV).

The fixed-camera pipeline classifies temporal crops (t-6, t-3, t) whose background is static. A
moving-camera clip gives the same kind of crop once the camera motion is removed *per crop*: the
crop centre at t is mapped into t-3 and t-6 through the inter-frame homographies (LK on a 512 px
frame + RANSAC), so the static world lines up and only the flier moves - exactly what a still
camera would record. That lets a public corpus of many different small drones teach the
classifier what "a drone" looks like, instead of the few airframes one site's own footage contains.

    python tools/fixedcam_harvest_moving.py --dataset ardmav --root data/external/ard_mav/ARD-MAV \
        --seqs phantom02 ... --out crops.npz [--every 3] [--max-per-seq 180]

Positives are jittered crops at the labelled drone (like a tracker probe); negatives are the
frame's strongest compact static blobs (roof corners, branches) and random background patches.
Output matches ``tools/fixedcam_harvest.py`` (label 0 drone, 2 other; no birds here).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def homography(prev, cur, lk_w=512):
    """Map CURRENT native px -> PREVIOUS native px (None if it cannot be estimated)."""
    import cv2

    H, W = cur.shape
    s = lk_w / W
    a = cv2.resize(cur, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    b = cv2.resize(prev, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    pts = cv2.goodFeaturesToTrack(a, 400, 0.01, 7)
    if pts is None or len(pts) < 15:
        return None
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(a, b, pts, None, winSize=(21, 21), maxLevel=4)
    ok = st.ravel() == 1
    if ok.sum() < 15:
        return None
    Hm, _ = cv2.findHomography(pts[ok], nxt[ok], cv2.RANSAC, 1.5)
    if Hm is None:
        return None
    S, Si = np.diag([s, s, 1.0]), np.diag([1 / s, 1 / s, 1.0])
    return Si @ Hm @ S


def apply(Hm, x, y):
    v = Hm @ np.array([x, y, 1.0])
    return v[0] / v[2], v[1] / v[2]


def main(argv=None) -> None:
    import cv2
    import torch
    import torch.nn.functional as F
    from torchvision.ops import roi_align

    from realtime.fixedcam.net import CROP, crop_size

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ardmav")
    ap.add_argument("--root", required=True)
    ap.add_argument("--seqs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--max-per-seq", type=int, default=180)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(a.seed)
    from benchmarks.adapters import ADAPTERS

    ad = ADAPTERS[a.dataset](a.root)
    raws, S, lab, off, src = [], [], [], [], []
    for seq in a.seqs:
        boxes = ad.boxes(seq)
        video = ad.image_source(seq).video
        cap = cv2.VideoCapture(str(video))
        ring, Hs = [], []          # last 7 grey frames; Hs[i] maps frame i -> frame i-1
        f, n_pos = -1, 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            f += 1
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            Hm = homography(ring[-1], g) if ring else np.eye(3)
            ring.append(g)
            Hs.append(Hm)
            if len(ring) > 7:
                ring.pop(0)
                Hs.pop(0)
            if len(ring) < 7 or f % a.every or f not in boxes or any(h is None for h in Hs[-6:]):
                continue
            if n_pos >= a.max_per_seq:
                break
            Hh, Wh = g.shape
            # maps from the current frame to t-3 and t-6
            # Hs[i] maps frame i -> i-1, so current -> t-3 applies Hs[-1] first
            m3 = Hs[-3] @ Hs[-2] @ Hs[-1]
            m6 = Hs[-6] @ Hs[-5] @ Hs[-4] @ m3
            centers, labels = [], []
            for b in boxes[f]:
                cx, cy = (b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2
                s = crop_size(max(b.x2 - b.x1, b.y2 - b.y1))
                jx, jy = rng.uniform(-0.25, 0.25, 2) * s
                centers.append((cx + jx, cy + jy, s, (-jx / (s / 2), -jy / (s / 2))))
                labels.append(0)
                n_pos += 1
            # negatives: strongest compact static blobs + one random patch, away from the drone
            gf = torch.from_numpy(g).cuda().float()[None, None]   # GPU: 9/31 px pools on 1080p
            closing = -F.max_pool2d(-F.max_pool2d(gf, 9, 1, 4), 9, 1, 4)
            opening = F.max_pool2d(-F.max_pool2d(-gf, 9, 1, 4), 9, 1, 4)
            con = torch.maximum(closing - gf, gf - opening)[0, 0]
            peak = (con == F.max_pool2d(con[None, None], 31, 1, 15)[0, 0]) & (con > 14)
            ys, xs = torch.nonzero(peak, as_tuple=True)
            order = torch.argsort(con[ys, xs], descending=True)[:40]
            cand = list(zip(xs[order].cpu().tolist(), ys[order].cpu().tolist()))
            cand += [(rng.uniform(40, Wh - 40), rng.uniform(40, Hh - 40)) for _ in range(3)]
            picked = 0
            for x, y in [cand[i] for i in rng.permutation(len(cand))]:
                if any(abs(x - (bb.x1 + bb.x2) / 2) < 40 and abs(y - (bb.y1 + bb.y2) / 2) < 40
                       for bb in boxes[f]) or not (20 < x < Wh - 20 and 20 < y < Hh - 20):
                    continue
                centers.append((x, y, crop_size(rng.uniform(3, 14)), (0.0, 0.0)))
                labels.append(2)
                picked += 1
                if picked >= 3:
                    break
            # per-crop motion compensation: each moment is cropped where the crop centre was then
            frames = {0: ring[-1], 3: ring[-4], 6: ring[-7]}
            maps = {0: np.eye(3), 3: m3, 6: m6}
            chans = []
            for k in (6, 3, 0):
                img = torch.from_numpy(frames[k]).float()[None, None]
                bxs = []
                for x, y, s, _ in centers:
                    px, py = apply(maps[k], x, y)
                    bxs.append([0, px - s / 2, py - s / 2, px + s / 2, py + s / 2])
                chans.append(roi_align(img, torch.tensor(bxs, dtype=torch.float32), (CROP, CROP),
                                       1.0, 2, True))
            r = torch.cat(chans, 1).clamp(0, 255).round().byte().numpy()
            raws.append(r)
            S.extend(c[2] for c in centers)
            off.extend([list(c[3]) for c in centers])
            lab.extend(labels)
            src.extend([1 if l_ == 0 else 3 for l_ in labels])
        cap.release()
        print(f"{seq}: {n_pos} drone crops", flush=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, raw=np.concatenate(raws), S=np.array(S, np.float32),
                        label=np.array(lab, np.int8), off=np.clip(np.array(off, np.float32), -1, 1),
                        frame=np.zeros(len(lab), np.int32), source=np.array(src, np.int8))
    lab = np.array(lab)
    print(f"{out.name}: {len(lab)} crops | drone {int((lab == 0).sum())} | other {int((lab == 2).sum())}")


if __name__ == "__main__":
    main()
