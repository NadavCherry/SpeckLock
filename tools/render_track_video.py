#!/usr/bin/env python3
"""Render what a system reports - boxes at their own size, the drone's trajectory, track ids -
optionally side by side with a second system, like the baseline-vs-pipeline showcase.

    python tools/render_track_video.py --video V.mp4 --points P.json --title "ours" --out O.mp4
        [--frames F.json] [--fps-label "123 FPS"] [--gt G.json]
        [--compare P2.json --compare-title "baseline" --compare-fps-label "25 FPS"]

--points is a ``dronedet.detections`` file whose labels are drone / bird (anything else is not
drawn). Trajectories come from track ids when --frames (a fixed-cam ``frames.json``) is given;
otherwise drone outputs are linked frame to frame by nearest neighbour, so every system is drawn
the same way. Colours: drone red (box + crosshair + trail), bird cyan, labelled truth green.
Each panel gets a magnified inset around its drone. Rendering happens after the fact, from the
JSON, so it never counts toward any timing.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RED, CYAN, GREEN, WHITE = (40, 40, 255), (255, 220, 0), (60, 220, 60), (255, 255, 255)


def link_points(per_frame: dict[int, list[tuple[float, float]]], gate: float = 80.0,
                max_gap: int = 10) -> dict[int, list[tuple[int, float, float]]]:
    """Give per-frame drone points track ids by greedy nearest-neighbour linking.

    A point continues the nearest live track whose last point is within ``gate`` px (the gate
    grows with the gap); a track unseen for more than ``max_gap`` frames ends. Returns
    {frame: [(track_id, x, y)]}.
    """
    live: dict[int, tuple[int, float, float]] = {}   # id -> (last frame, x, y)
    out: dict[int, list[tuple[int, float, float]]] = {}
    nxt = 0
    for f in sorted(per_frame):
        for tid in [t for t, (lf, _, _) in live.items() if f - lf > max_gap]:
            del live[tid]
        pairs = []
        for i, (x, y) in enumerate(per_frame[f]):
            for tid, (lf, lx, ly) in live.items():
                d = math.hypot(x - lx, y - ly)
                if d <= gate * (1 + 0.5 * (f - lf - 1)):
                    pairs.append((d, i, tid))
        used_i, used_t, res = set(), set(), []
        for d, i, tid in sorted(pairs):
            if i in used_i or tid in used_t:
                continue
            used_i.add(i)
            used_t.add(tid)
            res.append((tid, *per_frame[f][i]))
        for i, (x, y) in enumerate(per_frame[f]):
            if i not in used_i:
                res.append((nxt, x, y))
                nxt += 1
        for tid, x, y in res:
            live[tid] = (f, x, y)
        out[f] = res
    return out


def load_system(points: str, frames_json: str | None, dim_below: float = 0.0):
    """-> {frame: [dict(x, y, w, h, label, conf, tid, observed)]}; a drone output whose confidence is
    below ``dim_below`` is drawn dimmed, like a coasting track (``observed`` False)."""
    import json

    from dronedet.detections import DetectionSet

    ds = DetectionSet.load(points)
    per: dict[int, list[dict]] = {}
    for f, dets in ds.frames.items():
        for d in dets:
            lab = d.label.split(":")[0]
            if not (lab.startswith("drone") or lab.startswith("bird") or d.label.startswith("track")):
                continue
            label = "bird" if lab.startswith("bird") else "drone"
            per.setdefault(int(f), []).append(dict(x=d.cx, y=d.cy, w=d.w, h=d.h, label=label,
                                                   conf=d.score, tid=None,
                                                   observed=not (label == "drone" and d.score < dim_below)))
    if frames_json:   # fixed-cam: real track ids and the coasting flag
        recs = json.loads(Path(frames_json).read_text(encoding="utf-8"))
        for rec in recs:
            outs = per.get(int(rec["f"]), [])
            for (x, y, label, conf, tid, observed, size) in rec["pts"]:
                best = min(outs, key=lambda o: math.hypot(o["x"] - x, o["y"] - y), default=None)
                if best is not None and math.hypot(best["x"] - x, best["y"] - y) < 1.0:
                    best["tid"], best["observed"] = tid, bool(observed)
    else:
        drones = {f: [(o["x"], o["y"]) for o in v if o["label"] == "drone"] for f, v in per.items()}
        for f, linked in link_points(drones).items():
            ds_ = [o for o in per[f] if o["label"] == "drone"]
            for (tid, x, y) in linked:
                for o in ds_:
                    if o["tid"] is None and o["x"] == x and o["y"] == y:
                        o["tid"] = tid
                        break
    return per


def load_gt(path: str | None):
    if not path:
        return {}
    from dronedet.gt import GroundTruth

    g = GroundTruth.load(path)
    out: dict[int, list] = {}
    for o in g.objects.values():
        if o.name.startswith(("drone", "far")):
            for f, b in o.frames.items():
                out.setdefault(int(f), []).append(b)
    return out


class Panel:
    def __init__(self, per, title, fps_label, gt, trail, scale):
        self.per, self.title, self.fps_label, self.gt = per, title, fps_label, gt
        self.trail, self.scale = trail, scale
        self.hist: dict[int, list[tuple[int, float, float]]] = {}
        self.focus = None

    def render(self, frame, idx):
        import cv2
        import numpy as np

        img = frame.copy()
        H, W = img.shape[:2]
        th = max(1, round(H / 720))
        for b in self.gt.get(idx, []):
            cx, cy, w, h = b[:4]
            pad = 4 * th
            cv2.rectangle(img, (int(cx - w / 2 - pad), int(cy - h / 2 - pad)),
                          (int(cx + w / 2 + pad), int(cy + h / 2 + pad)), GREEN, 1, cv2.LINE_AA)
        outs = self.per.get(idx, [])
        for o in outs:
            if o["label"] == "drone" and o["tid"] is not None:
                self.hist.setdefault(o["tid"], []).append((idx, o["x"], o["y"]))
        # trajectories: the last `trail` frames of every drone track still alive
        for tid, pts in self.hist.items():
            pts[:] = [p for p in pts if idx - p[0] <= self.trail]
            for (f0, x0, y0), (f1, x1, y1) in zip(pts, pts[1:]):
                if f1 - f0 > 12:
                    continue
                a = 1.0 - (idx - f1) / self.trail
                col = tuple(int(c * (0.35 + 0.65 * a)) for c in RED)
                cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), col, max(1, th), cv2.LINE_AA)
        best = None
        for o in outs:
            col = RED if o["label"] == "drone" else CYAN
            if not o["observed"]:
                col = tuple(int(c * 0.55) for c in col)
            w, h = max(o["w"], 6.0), max(o["h"], 6.0)
            pad = 3 * th
            x1, y1 = int(o["x"] - w / 2 - pad), int(o["y"] - h / 2 - pad)
            x2, y2 = int(o["x"] + w / 2 + pad), int(o["y"] + h / 2 + pad)
            cv2.rectangle(img, (x1, y1), (x2, y2), col, th + (o["label"] == "drone"), cv2.LINE_AA)
            if o["label"] == "drone":
                arm = 10 * th
                for (dx, dy) in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    sx = x2 if dx > 0 else x1 if dx < 0 else int(o["x"])
                    sy = y2 if dy > 0 else y1 if dy < 0 else int(o["y"])
                    cv2.line(img, (sx, sy), (sx + dx * arm, sy + dy * arm), col, th, cv2.LINE_AA)
                if best is None or o["conf"] > best["conf"]:
                    best = o
            tag = ("DRONE" if o["label"] == "drone" else "bird") + f" {o['conf']:.2f}"
            if o["tid"] is not None and o["label"] == "drone":
                tag += f"  #{o['tid']}"
            cv2.putText(img, tag, (x1, max(14 * th, y1 - 5 * th)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45 * th, col, th, cv2.LINE_AA)
        if best is not None:
            self.focus = (best["x"], best["y"], max(best["w"], best["h"]))
        # downscale the panel, then draw crisp text and the inset at display size
        out = cv2.resize(img, (int(W * self.scale), int(H * self.scale)), interpolation=cv2.INTER_AREA)
        oh, ow = out.shape[:2]
        if self.focus is not None:
            # magnified inset cut from the CLEAN full-resolution frame, so the overlay never hides
            # the drone; thin outlines are drawn at inset scale instead
            fx, fy, fs = self.focus
            half = int(max(24, min(160, 2.2 * fs)))
            x0 = int(np.clip(fx - half, 0, W - 2 * half))
            y0 = int(np.clip(fy - half, 0, H - 2 * half))
            side = int(min(0.35 * oh, max(200, 0.24 * oh)))
            k = side / (2 * half)
            z = cv2.resize(frame[y0:y0 + 2 * half, x0:x0 + 2 * half], (side, side),
                           interpolation=cv2.INTER_NEAREST)
            for b in self.gt.get(idx, []):
                cx, cy, w, h = b[:4]
                cv2.rectangle(z, (int((cx - w / 2 - x0) * k) - 2, int((cy - h / 2 - y0) * k) - 2),
                              (int((cx + w / 2 - x0) * k) + 2, int((cy + h / 2 - y0) * k) + 2), GREEN, 1)
            for o in outs:
                col = RED if o["label"] == "drone" else CYAN
                w, h = max(o["w"], 4.0), max(o["h"], 4.0)
                cv2.rectangle(z, (int((o["x"] - w / 2 - x0) * k) - 3, int((o["y"] - h / 2 - y0) * k) - 3),
                              (int((o["x"] + w / 2 - x0) * k) + 3, int((o["y"] + h / 2 - y0) * k) + 3), col, 1)
            cv2.rectangle(z, (0, 0), (side - 1, side - 1), WHITE, 2)
            cv2.putText(z, f"x{k:.0f}", (6, side - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)
            out[44:44 + side, ow - 10 - side:ow - 10] = z
        cv2.rectangle(out, (0, 0), (ow, 34), (20, 20, 20), -1)
        cv2.putText(out, self.title, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)
        if self.fps_label:
            (tw, _), _ = cv2.getTextSize(self.fps_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.putText(out, self.fps_label, (ow - tw - 10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (120, 230, 255), 1, cv2.LINE_AA)
        return out


def main(argv=None) -> None:
    import cv2
    import numpy as np

    from dronedet.video import frames, probe

    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--points", required=True)
    ap.add_argument("--frames", help="fixed-cam frames.json: real track ids and coasting")
    ap.add_argument("--title", default="system")
    ap.add_argument("--fps-label", default="")
    ap.add_argument("--compare", help="a second system's points, drawn in a left panel")
    ap.add_argument("--compare-frames")
    ap.add_argument("--compare-title", default="baseline")
    ap.add_argument("--compare-fps-label", default="")
    ap.add_argument("--gt", help="labelled truth (drawn green) - omit for a clean output")
    ap.add_argument("--panel-width", type=int, default=0, help="panel width in px (default: native, "
                    "or 1280 per panel side by side)")
    ap.add_argument("--trail", type=int, default=75, help="trajectory length in frames")
    ap.add_argument("--dim-below", type=float, default=0.0,
                    help="draw drone outputs with confidence below this dimmed (e.g. unconfirmed)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    info = probe(a.video)
    pw = a.panel_width or (min(1280, info.width) if a.compare else info.width)
    scale = pw / info.width
    gt = load_gt(a.gt)
    panels = []
    if a.compare:
        panels.append(Panel(load_system(a.compare, a.compare_frames, a.dim_below), a.compare_title,
                            a.compare_fps_label, gt, a.trail, scale))
    panels.append(Panel(load_system(a.points, a.frames, a.dim_below), a.title, a.fps_label, gt, a.trail, scale))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    vw = None
    legend = ("red = drone (box, crosshair, trajectory)" + (f"; dim = below {a.dim_below:g}" if a.dim_below else "")
              + "   cyan = bird" + ("   green = labelled truth" if gt else ""))
    for idx, frame in frames(a.video):
        tiles = [p.render(frame, idx) for p in panels]
        if len(tiles) > 1:
            sep = np.full((tiles[0].shape[0], 4, 3), 255, np.uint8)
            canvas = np.hstack([tiles[0], sep, tiles[1]])
        else:
            canvas = tiles[0]
        ch, cw = canvas.shape[:2]
        bar = np.full((28, cw, 3), 20, np.uint8)
        cv2.putText(bar, f"frame {idx}   t={idx / (info.fps or 25):.1f}s", (10, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(legend, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(bar, legend, (cw - tw - 10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
        canvas = np.vstack([canvas, bar])
        if vw is None:
            vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), info.fps or 25.0,
                                 (canvas.shape[1], canvas.shape[0]))
        vw.write(canvas)
    if vw is not None:
        vw.release()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
