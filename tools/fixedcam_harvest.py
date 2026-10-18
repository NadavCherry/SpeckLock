#!/usr/bin/env python3
"""Harvest classifier training crops from one labelled clip, exactly as the live pipeline sees them.

The clip is run through the same causal ``MotionStage`` the real-time pipeline uses; every
proposal becomes a crop labelled from the GT (drone / bird / other), and "probe" crops are cut at
the GT drone and bird positions with jitter - the crops the tracker requests for a live track,
which is how a hovering drone (no motion) is kept. A few random background probes teach the net
that an empty patch is "other", and "static probes" at the frame's highest-contrast compact static
blobs (pole tops, roof corners, bushes - the things a probe can get stuck on) teach it that a
static drone-sized blob is not a drone unless it looks like one.

    python tools/fixedcam_harvest.py --video V.mp4 --gt G.json --out crops.npz [--osd '[[x0,y0,x1,y1]]']

GT: ``dronedet.gt`` format. Objects named drone*/far are the drone, bird* are birds, ``near``
(a landed drone) and ``unsure*`` are skipped, anything else is an "other" mover (a hard negative:
every proposal on it is kept, and it is probed like a bird). Output .npz: raw uint8 crops
[N, 3, 32, 32] (t-6, t-3, t), crop side S, label (0 drone, 1 bird, 2 other), centre-offset target
(units of S/2), frame, source (0 proposal, 1 probe, 2 background probe, 3 static probe).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None) -> None:
    import cv2
    import torch
    import torch.nn.functional as F

    from dronedet.gt import GroundTruth
    from realtime.fixedcam.motion import MotionStage
    from realtime.fixedcam.net import OFFSETS, crop_size, raw_crops

    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--osd", default="[]", help="JSON list of normalised [x0,y0,x1,y1] rects")
    ap.add_argument("--keep-other", type=float, default=0.35)
    ap.add_argument("--bird-probe-every", type=int, default=2)
    ap.add_argument("--static-per-frame", type=int, default=2)
    ap.add_argument("--no-mover-negatives", action="store_true",
                    help="skip proposals on labelled non-drone movers instead of using them as hard negatives")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(a.seed)

    gt = GroundTruth.load(a.gt)
    excl = set(int(f) for f in gt.meta.get("exclude_frames", []))
    drones = [o for o in gt.objects.values() if o.name.startswith(("drone", "far"))]
    birds = [o for o in gt.objects.values() if o.name.startswith("bird")]
    # 'near' (a landed drone) and 'unsure*' (unclassifiable) are skipped; every other labelled
    # mover (cars, insects, flicker) is the hardest negative there is: kept, and probed
    skip = [o for o in gt.objects.values() if o.name.startswith(("near", "unsure", "ignore"))]
    others = [o for o in gt.objects.values() if o not in drones and o not in birds and o not in skip]
    if a.no_mover_negatives:
        skip, others = skip + others, []

    cap = cv2.VideoCapture(a.video)
    ms = None
    static_pts = []
    raws, S, lab, off, frm, src = [], [], [], [], [], []
    f = -1
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        f += 1
        g = torch.from_numpy(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)).cuda()
        if ms is None:
            ms = MotionStage(g.shape[0], g.shape[1], osd=json.loads(a.osd))
            H, W = g.shape
        props, _ = ms.step(g)
        if f in excl:
            continue
        dboxes = [o.frames[f] for o in drones if f in o.frames]
        bboxes = [o.frames[f] for o in birds if f in o.frames]
        sboxes = [o.frames[f] for o in skip if f in o.frames]
        oboxes = [o.frames[f] for o in others if f in o.frames]
        centers, labels, targets, sources = [], [], [], []

        def near(x, y, boxes):
            best = None
            for b in boxes:
                d = math.hypot(x - b[0], y - b[1])
                r = max(8.0, 0.75 * max(b[2], b[3]))
                if d <= r and (best is None or d < best[0]):
                    best = (d, b)
            return best

        for p in props:
            s = crop_size(p.size)
            hit = near(p.x, p.y, dboxes)
            if hit:
                b = hit[1]
                centers.append((p.x, p.y, s)); labels.append(0)
                targets.append(((b[0] - p.x) / (s / 2), (b[1] - p.y) / (s / 2))); sources.append(0)
                continue
            hit = near(p.x, p.y, bboxes)
            if hit:
                b = hit[1]
                centers.append((p.x, p.y, s)); labels.append(1)
                targets.append(((b[0] - p.x) / (s / 2), (b[1] - p.y) / (s / 2))); sources.append(0)
                continue
            if near(p.x, p.y, sboxes):
                continue
            if near(p.x, p.y, oboxes):   # a labelled non-drone mover: always kept
                centers.append((p.x, p.y, s)); labels.append(2); targets.append((0.0, 0.0)); sources.append(0)
                continue
            if rng.random() < a.keep_other:
                centers.append((p.x, p.y, s)); labels.append(2); targets.append((0.0, 0.0)); sources.append(0)
        # probes at the GT objects, jittered like a tracker prediction would be
        for b in dboxes:
            s = crop_size(max(b[2], b[3]))
            jx, jy = rng.uniform(-0.25, 0.25, 2) * s
            centers.append((b[0] + jx, b[1] + jy, s)); labels.append(0)
            targets.append((-jx / (s / 2), -jy / (s / 2))); sources.append(1)
        if f % a.bird_probe_every == 0:
            for b in bboxes:
                s = crop_size(max(b[2], b[3]))
                jx, jy = rng.uniform(-0.25, 0.25, 2) * s
                centers.append((b[0] + jx, b[1] + jy, s)); labels.append(1)
                targets.append((-jx / (s / 2), -jy / (s / 2))); sources.append(1)
        # static probes: the frame's strongest compact static blobs (refreshed every 25 frames)
        if f % 25 == 0 or f == 0:
            gf = g.float()[None, None]
            closing = -F.max_pool2d(-F.max_pool2d(gf, 9, 1, 4), 9, 1, 4)
            opening = F.max_pool2d(-F.max_pool2d(-gf, 9, 1, 4), 9, 1, 4)
            con = torch.maximum(closing - gf, gf - opening)[0, 0]
            peak = (con == F.max_pool2d(con[None, None], 31, 1, 15)[0, 0]) & (con > 14)
            ys, xs = torch.nonzero(peak, as_tuple=True)
            v = con[ys, xs]
            order = torch.argsort(v, descending=True)[:60]
            static_pts = list(zip(xs[order].tolist(), ys[order].tolist()))
        if static_pts:
            for i in rng.permutation(len(static_pts))[:a.static_per_frame * 3]:
                x, y = static_pts[i]
                if near(x, y, dboxes + bboxes + sboxes + oboxes) or not (20 < x < W - 20 and 20 < y < H - 20):
                    continue
                sz = crop_size(rng.uniform(3, 12))
                jx, jy = rng.uniform(-0.2, 0.2, 2) * sz
                centers.append((x + jx, y + jy, sz)); labels.append(2); targets.append((0.0, 0.0)); sources.append(3)
                if sum(1 for q in sources if q == 3) >= a.static_per_frame:
                    break
        if f % a.bird_probe_every == 0:
            for b in oboxes:          # probes on labelled non-drone movers, like a tracker would ask
                s = crop_size(max(b[2], b[3]))
                jx, jy = rng.uniform(-0.25, 0.25, 2) * s
                centers.append((b[0] + jx, b[1] + jy, s)); labels.append(2); targets.append((0.0, 0.0)); sources.append(1)
        # one background probe: somewhere with no labelled object nearby
        for _ in range(4):
            x, y = rng.uniform(40, W - 40), rng.uniform(40, H - 40)
            if not near(x, y, dboxes + bboxes + sboxes + oboxes) and not any(
                    math.hypot(x - q.x, y - q.y) < 30 for q in props):
                s = crop_size(rng.uniform(3, 20))
                centers.append((x, y, s)); labels.append(2); targets.append((0.0, 0.0)); sources.append(2)
                break
        if not centers:
            continue
        frames = [ms.recent(k) for k in OFFSETS]
        r = raw_crops(frames, centers).clamp(0, 255).round().byte().cpu().numpy()
        raws.append(r)
        S.extend(c[2] for c in centers)
        lab.extend(labels)
        off.extend(np.clip(np.array(targets, float), -1, 1).tolist())
        frm.extend([f] * len(centers))
        src.extend(sources)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, raw=np.concatenate(raws) if raws else np.zeros((0, 3, 32, 32), np.uint8),
                        S=np.array(S, np.float32), label=np.array(lab, np.int8),
                        off=np.array(off, np.float32).reshape(-1, 2), frame=np.array(frm, np.int32),
                        source=np.array(src, np.int8))
    lab = np.array(lab)
    src = np.array(src)
    print(f"{out.name}: {len(lab)} crops | drone {int((lab == 0).sum())} "
          f"(proposals {int(((lab == 0) & (src == 0)).sum())}) | bird {int((lab == 1).sum())} "
          f"(proposals {int(((lab == 1) & (src == 0)).sum())}) | other {int((lab == 2).sum())}")


if __name__ == "__main__":
    main()
