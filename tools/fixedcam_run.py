#!/usr/bin/env python3
"""Run the fixed-camera pipeline on a video: outputs, per-stage timing, and an annotated video.

    python tools/fixedcam_run.py --video V.mp4 --engine clf.engine --out DIR [--osd JSON]
                                 [--weights model.pt] [--render] [--gt G.json]

Writes DIR/points.json (``dronedet.detections`` format: label drone / bird / unknown, box = the
tracked size around the reported centre - score it with ``python -m dronedet centerscore``),
DIR/timing.json (per-stage mean ms over all frames, warm-up included; wall-clock median/p95 and FPS over
frames 50+), DIR/frames.json (per-frame points with track ids and the coasting flag, read by
tools/render_track_video.py --frames) and, with --render,
DIR/annotated.mp4 drawn in a second pass so drawing never counts toward the timing.

Render convention: drone red (box + crosshair on the reported centre + confidence), bird cyan,
unconfirmed tracks grey, coasting (not observed this frame) drawn dim, GT (with --gt) green.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

COL = {"drone": (40, 40, 255), "bird": (255, 220, 0), "unknown": (170, 170, 170), "gt": (60, 220, 60)}


def draw(frame, pts, cands, gt_pts, idx, fps_txt):
    import cv2

    H, W = frame.shape[:2]
    th = max(1, W // 960)
    for (gx, gy) in gt_pts:
        cv2.circle(frame, (int(gx), int(gy)), 14 * th, COL["gt"], th)
    for (x, y, tid) in cands:
        cv2.circle(frame, (int(x), int(y)), 6 * th, COL["unknown"], th)
    for (x, y, label, conf, tid, observed, size) in pts:
        c = COL.get(label, COL["unknown"])
        if not observed:
            c = tuple(int(v * 0.5) for v in c)
        h = max(8, int(size * 0.8) + 6)
        x0, y0 = int(x - h), int(y - h)
        cv2.rectangle(frame, (x0, y0), (int(x + h), int(y + h)), c, 2 * th)
        if label == "drone":
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                cv2.line(frame, (int(x + dx * 3), int(y + dy * 3)), (int(x + dx * (h + 8)), int(y + dy * (h + 8))), c, th)
        txt = f"{'DRONE' if label == 'drone' else 'bird'} {conf:.2f}"
        cv2.putText(frame, txt, (x0, max(12, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * th, c, th + 1)
    cv2.putText(frame, f"f{idx}  {fps_txt}", (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * th, (255, 255, 255), th + 1)
    return frame


def main(argv=None) -> None:
    import cv2
    import numpy as np
    import torch

    from dronedet.detections import Detection, DetectionSet
    from realtime.fixedcam.pipeline import FixedCamPipeline
    from realtime.fixedcam.tracker import TrackerConfig
    from realtime.fixedcam.trt import EnsembleClassifier, TorchClassifier, TRTClassifier

    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--engine", nargs="+", help="TensorRT engine(s) of the crop classifier; several = averaged ensemble")
    ap.add_argument("--weights", help="PyTorch model.pt (instead of --engine)")
    ap.add_argument("--osd", default="[]")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--gt", help="overlay GT drone centres (green) in the render")
    ap.add_argument("--name", default="fixedcam")
    ap.add_argument("--tracker-kw", default="{}", help="JSON overrides for TrackerConfig")
    ap.add_argument("--motion-kw", default="{}", help="JSON overrides for MotionStage")
    ap.add_argument("--appearance-every", type=int, default=0,
                    help="add appearance (no-motion) proposals every N frames (hover re-acquisition); 0 = off")
    ap.add_argument("--v1-compat", action="store_true",
                    help="reproduce the first frozen configuration: no path-roughness rule, no probe clamp")
    ap.add_argument("--extra-dets", help="another detector's per-frame detections (dronedet.detections "
                    "JSON, same video): each becomes a candidate that is cropped, classified and tracked")
    ap.add_argument("--extra-weight", type=float, default=0.0,
                    help="weight of that detector's drone/bird label in the class estimate (0 = locations only)")
    ap.add_argument("--extra-min-score", default='{"drone": 0.05, "bird": 0.1, "": 0.0}',
                    help="JSON label-prefix -> minimum score for an extra detection to be used (a label "
                         "matching no prefix is dropped; drop motion-only ones by leaving out \"\")")
    ap.add_argument("--protect-tracks", action="store_true",
                    help="the motion stage may not mute the cells of a live drone track")
    ap.add_argument("--dump-candidates", action="store_true",
                    help="write DIR/candidates.npz: every cropped candidate per frame with the classifier's "
                         "raw opinion and its source (0 motion/appearance, 1 external, 2 probe) - for "
                         "finding which stage loses a target")
    ap.add_argument("--dump-track-features", action="store_true",
                    help="write DIR/track_features.npz: every observed track's trackfeat.FEATURES per "
                         "frame (the rows a track classifier is trained on)")
    ap.add_argument("--extra-bird-veto", type=float, default=None,
                    help="ignore the extra detector's 'drone' where the crop classifier gives p(bird) >= this")
    ap.add_argument("--extra-strict-births", action="store_true",
                    help="an extra detection off every motion proposal may update a track but start one "
                         "only like an appearance proposal (p(drone) >= the tracker's appearance_birth)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if a.engine:
        clf = (TRTClassifier(a.engine[0]) if len(a.engine) == 1 else
               EnsembleClassifier(TRTClassifier(e) for e in a.engine))
    else:
        from realtime.fixedcam.net import TinyNet

        ck = torch.load(a.weights, map_location="cpu")
        net = TinyNet(ck.get("width", 32))
        net.load_state_dict(ck["state_dict"])
        clf = TorchClassifier(net)

    extra_by_frame: dict[int, list] = {}
    if a.extra_dets:
        from realtime.fixedcam.tracker import det_evidence

        floors = json.loads(a.extra_min_score)
        for f, dets in DetectionSet.load(a.extra_dets).frames.items():
            keep = []
            for d in dets:
                pre = max((k for k in floors if d.label.startswith(k)), key=len, default=None)
                if pre is None or d.score < floors[pre]:
                    continue
                keep.append((d.cx, d.cy, max(d.w, d.h, 2.0), det_evidence(d.label, d.score)))
            extra_by_frame[int(f)] = keep

    cap = cv2.VideoCapture(a.video)
    pipe = None
    dump: list = []
    cand_dump: list = []
    ds = DetectionSet(video=Path(a.video).name, method=a.name)
    per_frame = []
    wall = []
    while True:
        t0 = time.perf_counter()
        ok, fr = cap.read()
        t1 = time.perf_counter()
        if not ok:
            break
        if pipe is None:
            tkw = json.loads(a.tracker_kw)
            if a.v1_compat:
                tkw.update(smooth_commit=1e9, smooth_keep=1e9)
            pipe = FixedCamPipeline(clf, fr.shape[0], fr.shape[1], osd=json.loads(a.osd),
                                    tracker_cfg=TrackerConfig(**tkw), motion_kw=json.loads(a.motion_kw),
                                    probe_clamp=not a.v1_compat,
                                    appearance_every=a.appearance_every,
                                    extra_weight=a.extra_weight,
                                    extra_strict_births=a.extra_strict_births,
                                    extra_bird_veto=a.extra_bird_veto,
                                    protect_tracks=a.protect_tracks)
        pipe.debug = a.dump_candidates
        r = pipe.process(fr, decode_s=t1 - t0, extra=extra_by_frame.get(pipe.idx + 1))
        wall.append(time.perf_counter() - t0)
        if a.dump_candidates:
            cand_dump.extend((r.index, *c) for c in pipe.last_candidates)
        if a.dump_track_features:
            for (tid, x, y, label, fv) in pipe.tracker.feature_rows():
                dump.append((r.index, tid, x, y, label, fv))
        sizes = {t.id: t.size for t in pipe.tracker.tracks}
        dets = []
        rec_pts = []
        for (x, y, label, conf, tid, observed) in r.points:
            s = max(4.0, sizes.get(tid, 8.0))
            dets.append(Detection(x - s / 2, y - s / 2, x + s / 2, y + s / 2, conf, label))
            rec_pts.append([round(x, 2), round(y, 2), label, round(conf, 4), tid, bool(observed), round(s, 1)])
        for (x, y, tid) in r.candidates:
            dets.append(Detection(x - 4, y - 4, x + 4, y + 4, 0.0, "unknown"))
        ds.add(r.index, dets)
        per_frame.append({"f": r.index, "pts": rec_pts,
                          "cands": [[round(x, 1), round(y, 1), t] for x, y, t in r.candidates],
                          "props": r.n_props, "probes": r.n_probes, "moving": r.moving})
    n = len(per_frame)
    if pipe is None:
        raise SystemExit(f"no frames decoded from {a.video}")
    # per-stage means over ALL frames (warm-up included); the wall_ms_* fields below skip the first 50
    steady = pipe.timer
    timing = {"frames": n, "stage_ms_mean_all_frames": steady.report(),
              "motion_breakdown_ms": steady.per_frame(pipe.motion.prof, n),
              "wall_ms_median": round(1000 * float(np.median(wall[50:] or wall)), 3),
              "wall_ms_p95": round(1000 * float(np.percentile(wall[50:] or wall, 95)), 3),
              "fps_median": round(1.0 / float(np.median(wall[50:] or wall)), 2),
              "gpu": torch.cuda.get_device_name(0), "backend": "tensorrt" if a.engine else "torch"}
    ds.meta = {"timing": timing, "n_frames": n, "tracker_kw": json.loads(a.tracker_kw),
               "motion_kw": json.loads(a.motion_kw)}
    ds.save(out / "points.json")
    (out / "frames.json").write_text(json.dumps(per_frame), encoding="utf-8")
    (out / "timing.json").write_text(json.dumps(timing, indent=1), encoding="utf-8")
    print(json.dumps(timing))
    if a.dump_candidates:
        np.savez_compressed(out / "candidates.npz", rows=np.array(cand_dump, float).reshape(-1, 8),
                            columns=np.array(["frame", "x", "y", "side", "p_drone", "p_bird", "p_other", "source"]))
    if a.dump_track_features:
        from realtime.fixedcam.trackfeat import FEATURES

        np.savez_compressed(out / "track_features.npz",
                            frame=np.array([d[0] for d in dump], int), tid=np.array([d[1] for d in dump], int),
                            xy=np.array([[d[2], d[3]] for d in dump], float).reshape(-1, 2),
                            label=np.array([d[4] for d in dump]),
                            X=np.array([d[5] for d in dump], float).reshape(-1, len(FEATURES)),
                            features=np.array(FEATURES))

    if a.render:
        gtc = {}
        if a.gt:
            from dronedet.gt import GroundTruth

            g = GroundTruth.load(a.gt)
            for o in g.objects.values():
                if o.name.startswith(("drone", "far")):
                    for f, b in o.frames.items():
                        gtc.setdefault(f, []).append((b[0], b[1]))
        cap = cv2.VideoCapture(a.video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        vw = None
        fps_txt = f"{timing['fps_median']:.0f} fps ({timing['gpu'].replace('NVIDIA GeForce ', '')}, {timing['backend']})"
        for rec in per_frame:
            ok, fr = cap.read()
            if not ok:
                break
            if vw is None:
                vw = cv2.VideoWriter(str(out / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                     (fr.shape[1], fr.shape[0]))
            vw.write(draw(fr, rec["pts"], rec["cands"], gtc.get(rec["f"], []), rec["f"], fps_txt))
        if vw is not None:
            vw.release()
        print("rendered", out / "annotated.mp4")


if __name__ == "__main__":
    main()
