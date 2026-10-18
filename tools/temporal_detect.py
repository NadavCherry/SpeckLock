#!/usr/bin/env python3
"""Run a temporal-stack YOLO over a video, causally, at native resolution -> detections JSON.

    python tools/temporal_detect.py --weights W.pt --video V.mp4 --out D.json [--dt 5] [--conf 0.05]
        [--upscale bicubic|fsrcnn --scale 2]

Per frame: translation stabilisation, a (2*dt+1)-frame ring of greys, and the three moments
(t-2dt, t-dt, t) aligned to frame t (``make_dataset_external._stack_aligned_to_now``), i.e.
exactly the stack the training tiles were cut from. The model sees the whole frame at its native
size (imgsz = frame width, rectangular padding), so a model trained on native-resolution tiles
meets every target at the pixel size it was trained on. Output: ``dronedet.detections`` JSON with
the model's own class names as labels; ``meta.fps_end_to_end`` is decode to detections.

``--upscale`` puts an upscaler (``dronedet.superres``) at the very front: every decoded frame is
enlarged ``--scale`` times before stacking (its stabilising translation is measured on the native
frame and scaled), the model runs at the enlarged size, and the boxes are mapped back to
original-frame pixels.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None) -> None:
    import cv2
    import numpy as np
    from ultralytics import YOLO

    from dronedet.detections import Detection, DetectionSet
    from dronedet.stabilize import Stabilizer
    from tools.make_dataset_external import _stack_aligned_to_now

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dt", type=int, default=5)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--half", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--upscale", choices=["none", "bicubic", "fsrcnn", "area"], default="none",
                    help="resample every frame first by --scale (bicubic/fsrcnn to enlarge, area to shrink)")
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--osd", default="[]", help="JSON list of normalised [x0,y0,x1,y1] burned-in text "
                    "rects: inside them all three moments are set to the current frame, so a ticking "
                    "clock leaves no temporal trail")
    a = ap.parse_args(argv)
    osd = json.loads(a.osd)

    model = YOLO(a.weights)
    up = None
    if a.upscale != "none":
        from dronedet.superres import Upscaler

        up = Upscaler(a.upscale, a.scale)
    k = a.scale if up else 1.0
    stab = Stabilizer("translation")
    buf: deque = deque(maxlen=2 * a.dt + 1)
    ds = DetectionSet(video=Path(a.video).name, method=f"temporal-{Path(a.weights).stem}")
    cap = cv2.VideoCapture(a.video)
    idx, t0 = 0, time.perf_counter()
    names = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        m = stab.update(frame)          # stabilise at native size; an upscaled frame shifts k times as far
        if up:
            frame = up(frame)
        buf.append((cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), float(m[0, 2]) * k, float(m[1, 2]) * k))
        stack = np.dstack(_stack_aligned_to_now(buf, a.dt))
        H, W = stack.shape[:2]
        for x0, y0, x1, y1 in osd:
            ys, xs = slice(int(y0 * H), int(y1 * H) + 1), slice(int(x0 * W), int(x1 * W) + 1)
            stack[ys, xs, 0] = stack[ys, xs, 2]
            stack[ys, xs, 1] = stack[ys, xs, 2]
        r = model.predict(stack, imgsz=frame.shape[1], conf=a.conf, half=a.half, verbose=False, device=0)[0]
        names = r.names
        dets = []
        for b in r.boxes:
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            dets.append(Detection(x1 / k, y1 / k, x2 / k, y2 / k, float(b.conf[0]), names[int(b.cls[0])]))
        ds.add(idx, dets)
        idx += 1
    elapsed = time.perf_counter() - t0
    ds.meta = {"n_frames": idx, "fps_end_to_end": round(idx / max(elapsed, 1e-9), 2), "dt": a.dt,
               "conf": a.conf, "weights": str(a.weights), "classes": names,
               "upscale": a.upscale, "scale": k}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    ds.save(a.out)
    print(f"{idx} frames, {sum(len(v) for v in ds.frames.values())} detections, "
          f"{ds.meta['fps_end_to_end']} fps -> {a.out}")


if __name__ == "__main__":
    main()
