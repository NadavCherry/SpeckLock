#!/usr/bin/env python3
"""Temporal-stack YOLO tiles from any video + ``dronedet.gt`` file, drone and bird classes.

    python tools/make_dataset_gt_temporal.py --video V.mp4 --gt G.json --out DIR --prefix clipA
        [--dt 5] [--stride 2] [--tile 640] [--min-side 12] [--val-every 10] [--scale 1.0]

Each selected frame becomes stabilised 3-moment stacks (t-2dt, t-dt, t), aligned to frame t, cut
into native-resolution tiles: one centred (with jitter) on every labelled object plus one random
object-free tile (``make_dataset_external.extract_yolo_tiled_temporal``, the builder of the
shipped temporal models). Classes come from GT object names: ``drone*``/``far`` -> 0 (drone),
``bird*`` -> 1 (bird); anything else is not labelled. Frames in ``meta.exclude_frames`` are
skipped. Every ``val_every``-th selected frame goes to ``val/`` (monitoring only) and the rest to
``train/``. Call it once per clip; a data.yaml can then list several clips' directories.
``--scale`` < 1 resizes every frame (area interpolation) before stabilisation and stacking, and the
boxes with it, so a detector that will run on downscaled frames is trained on exactly that input.

dt: the shipped models use dt 6 at 30 fps (0.2 s); at 25 fps the same interval is 5 frames.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CLASSES = ("drone", "bird")


def boxes_by_frame(gt_path: str | Path, avoid: tuple[str, ...] = ()) -> tuple[dict[int, list[tuple]], set[int]]:
    """GT -> {frame: [(x1, y1, x2, y2, class_id)]}, and the excluded frames. Objects whose name
    starts with a prefix in ``avoid`` get class -1: kept out of object-free tiles, never labelled."""
    g = json.loads(Path(gt_path).read_text(encoding="utf-8"))
    out: dict[int, list[tuple]] = {}
    for name, obj in g.get("objects", {}).items():
        cls = (-1 if avoid and name.startswith(tuple(avoid)) else
               0 if name.startswith(("drone", "far")) else 1 if name.startswith("bird") else None)
        if cls is None:
            continue
        for f, (cx, cy, w, h, *_) in obj.get("frames", {}).items():
            out.setdefault(int(f), []).append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, cls))
    return out, {int(f) for f in g.get("meta", {}).get("exclude_frames", [])}


def main(argv=None) -> None:
    from tools.make_dataset_external import extract_yolo_tiled_temporal

    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--dt", type=int, default=5)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--min-side", type=float, default=12.0)
    ap.add_argument("--val-every", type=int, default=10)
    ap.add_argument("--n-frames", type=int, default=None, help="default: GT meta n_frames")
    ap.add_argument("--scale", type=float, default=1.0, help="resize frames (and boxes) by this factor first")
    ap.add_argument("--avoid", default="", help="comma-separated object-name prefixes to keep out of "
                    "object-free tiles without labelling them (e.g. near = a landed drone)")
    a = ap.parse_args(argv)

    boxes, excluded = boxes_by_frame(a.gt, tuple(p for p in a.avoid.split(",") if p))
    if a.scale != 1.0:
        boxes = {f: [(x1 * a.scale, y1 * a.scale, x2 * a.scale, y2 * a.scale, c) for x1, y1, x2, y2, c in bb]
                 for f, bb in boxes.items()}
    n = a.n_frames or json.loads(Path(a.gt).read_text(encoding="utf-8"))["meta"]["n_frames"]
    picked = [f for f in range(2 * a.dt, n, a.stride) if f not in excluded]   # full trails only
    val = set(picked[::a.val_every])
    out = Path(a.out)
    rep = {}
    for split, ids in (("train", [f for f in picked if f not in val]), ("val", sorted(val))):
        ni, nb = extract_yolo_tiled_temporal(a.video, boxes, ids, out / split / "images", out / split / "labels",
                                             a.prefix, a.min_side, tile=a.tile, dt=a.dt, scale=a.scale)
        rep[split] = {"frames": len(ids), "tiles": ni, "boxes": nb}
    (out / f"{a.prefix}_build.json").write_text(json.dumps({**rep, "args": vars(a)}, indent=1), encoding="utf-8")
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
