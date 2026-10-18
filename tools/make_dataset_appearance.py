#!/usr/bin/env python3
"""Single-frame appearance crops of small airborne objects (drone / bird / other) from several public corpora.

An *appearance* crop is one native frame, centred on the object and scale-normalised by the object's size: the
stored crop is ``STORE`` x ``STORE`` grey pixels covering a square of ``CONTEXT`` x size native pixels (border
replicated). A classifier then reads the central ``WINDOW`` x ``WINDOW`` (2.5 x size) resized to ``NET`` x ``NET``
(`net_window`), so a 12 px and a 120 px object fill the same share of the input and only the detail differs.
Unlike the temporal stacks of `tools/fixedcam_harvest.py`, nothing here needs consecutive frames, so stills
(UAV_SMID) and weakly labelled clips (Halmstad) can be mixed with video ground truth.

Sources (one sub-command each; every one writes the same npz layout, see `save`):

    halmstad   Halmstad drone/bird/airplane/helicopter clips, weak labels: the strongest compact blob against a
               31-px median background, the longest run of consistent detections (the dataset's MATLAB labels are
               not readable here; see benchmarks/adapters/halmstad.py). DRONE -> drone, BIRD -> bird, else other.
    ardmav     ARD-MAV VOC boxes (1-based XML index = decoded frame + 1) plus one random background patch per
               sampled frame as 'other'.
    uav_smid   UAV_SMID stills (VOC XML), single-object images, each object re-rendered at small sizes (area
               downscale of the whole image + JPEG) before cropping. `uav-select` writes the image list first.
    gt         any dronedet GT json (center format) with a name -> class map, e.g. 'far=drone,bird*=bird'.
    points     any video + an npz of points (frame, cx, cy, size, label[, key]) -> crops (generic hook for callers
               that pick their own points, e.g. candidate rows of a tracker).

    python tools/make_dataset_appearance.py halmstad --videos DIR/V_*.mp4 --out halmstad.npz
    python tools/make_dataset_appearance.py ardmav --root data/external/ard_mav/ARD-MAV --seqs phantom02 ... --out a.npz
    python tools/make_dataset_appearance.py uav-select --root data/external/uav_smid --out sel.json
    python tools/make_dataset_appearance.py uav_smid --root data/external/uav_smid --select sel.json --out u.npz
    python tools/make_dataset_appearance.py gt --video V.mp4 --gt G.json --map far=drone,bird*=bird --out g.npz

Imports are CI-safe: numpy at module level, OpenCV only inside the functions that decode or resize.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

CLASSES = ("drone", "bird", "other")
LABEL = {c: i for i, c in enumerate(CLASSES)}
STORE = 64          # stored crop side (px)
CONTEXT = 3.2       # stored crop covers CONTEXT x object size (native px)
WINDOW = 50         # classifier window inside the stored crop: 50 / 64 * 3.2 = 2.5 x size
NET = 48            # classifier input side
MIN_SIDE = 8.0      # never cut a region smaller than this (native px)
SIZE_CLAMP = (4.0, 64.0)


# ------------------------------------------------------------------ crop geometry
def region_side(size: float, context: float = CONTEXT) -> float:
    """Native side of the stored region for an object of `size` px."""
    return max(MIN_SIDE, context * max(1.0, float(size)))


def crop_patch(gray: np.ndarray, cx: float, cy: float, size: float, store: int = STORE,
               context: float = CONTEXT) -> np.ndarray:
    """store x store uint8 crop of a square of side context x size centred at (cx, cy); border replicated;
    area interpolation when shrinking, bilinear when enlarging."""
    import cv2

    side = int(math.ceil(region_side(size, context)))
    patch = cv2.getRectSubPix(gray, (side, side), (float(cx), float(cy)))
    interp = cv2.INTER_AREA if side > store else cv2.INTER_LINEAR
    return cv2.resize(patch, (store, store), interpolation=interp)


def net_window(crop: np.ndarray, side: float = WINDOW, dx: float = 0.0, dy: float = 0.0, out: int = NET) -> np.ndarray:
    """The classifier's view of a stored crop: a side x side window centred at the crop centre + (dx, dy),
    resized to out x out (float32, 0..255)."""
    import cv2

    c = (crop.shape[1] / 2.0 + dx, crop.shape[0] / 2.0 + dy)
    s = int(round(side))
    w = cv2.getRectSubPix(crop.astype(np.float32), (s, s), c)
    interp = cv2.INTER_AREA if s > out else cv2.INTER_LINEAR
    return cv2.resize(w, (out, out), interpolation=interp)


def scale_plane_value(size_native: float) -> float:
    """Constant value of the scale input plane: 0 at 16 px, +-1 per factor 4, clamped to SIZE_CLAMP."""
    s = float(np.clip(size_native, *SIZE_CLAMP))
    return (math.log(s) - math.log(16.0)) / math.log(4.0)


def standardise(win: np.ndarray) -> np.ndarray:
    """Per-crop standardisation of a 0..255 window (robust to haze and exposure)."""
    x = win.astype(np.float32) / 255.0
    return (x - x.mean()) / (x.std() + 0.02)


# ------------------------------------------------------------------ containers
class Builder:
    """Accumulates crops and their metadata; `save` writes one npz."""

    def __init__(self) -> None:
        self.crops, self.label, self.size, self.frame, self.src, self.group, self.split, self.key = ([] for _ in range(8))

    def add(self, crop, label, size, frame, src, group, split=0, key=-1) -> None:
        self.crops.append(np.asarray(crop, np.uint8))
        self.label.append(int(label))
        self.size.append(float(size))
        self.frame.append(int(frame))
        self.src.append(str(src))
        self.group.append(str(group))
        self.split.append(int(split))
        self.key.append(int(key))

    def __len__(self) -> int:
        return len(self.label)

    def arrays(self) -> dict:
        n = len(self.label)
        return dict(crops=np.stack(self.crops) if n else np.zeros((0, STORE, STORE), np.uint8),
                    label=np.array(self.label, np.int8), size=np.array(self.size, np.float32),
                    frame=np.array(self.frame, np.int32), src=np.array(self.src), group=np.array(self.group),
                    split=np.array(self.split, np.int8), key=np.array(self.key, np.int64))

    def save(self, path) -> dict:
        a = self.arrays()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **a)
        return {c: int((a["label"] == i).sum()) for i, c in enumerate(CLASSES)}


def iter_gray(video):
    """Sequential decode (never seek): yields (index, grey frame)."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        yield i, cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        i += 1
    cap.release()


def crops_from_video(video, points, builder: Builder, src: str, group: str) -> int:
    """points: iterable of (frame, cx, cy, size, label, split, key) in native px. One sequential decode."""
    by_f: dict[int, list] = {}
    for p in points:
        by_f.setdefault(int(p[0]), []).append(p)
    if not by_f:
        return 0
    last, n = max(by_f), 0
    for i, g in iter_gray(video):
        for (f, cx, cy, size, lab, split, key) in by_f.get(i, []):
            builder.add(crop_patch(g, cx, cy, size), lab, size, f, src, group, split, key)
            n += 1
        if i >= last:
            break
    return n


# ------------------------------------------------------------------ Halmstad (weak labels)
def salient_blob(g: np.ndarray, margin: int = 16):
    """Strongest compact blob against a 31-px median background: (score, cx, cy, size) or None."""
    import cv2

    bg = cv2.medianBlur(g, 31).astype(np.float32)
    D = np.abs(g.astype(np.float32) - bg)
    D[:margin] = 0
    D[-margin:] = 0
    D[:, :margin] = 0
    D[:, -margin:] = 0
    thr = max(15.0, float(np.percentile(D, 99.9)))
    n, _, st, cen = cv2.connectedComponentsWithStats((D > thr).astype(np.uint8))
    best = None
    for i in range(1, n):
        if 4 <= st[i, 4] <= 3000:
            sc = float(D[st[i, 1]:st[i, 1] + st[i, 3], st[i, 0]:st[i, 0] + st[i, 2]].sum())
            if best is None or sc > best[0]:
                best = (sc, float(cen[i][0]), float(cen[i][1]), float(max(st[i, 2], st[i, 3])))
    return best


def longest_run(dets, max_step: float = 30.0, min_len: int = 16):
    """dets: per frame (score, cx, cy, size) or None -> the longest run [(f, cx, cy, size)] of consecutive frames
    whose detections move < max_step px per frame; [] if shorter than min_len."""
    runs, cur = [], []
    for f, d in enumerate(dets):
        if d and cur and math.hypot(d[1] - cur[-1][1], d[2] - cur[-1][2]) < max_step:
            cur.append((f, d[1], d[2], d[3]))
        else:
            if len(cur) >= min_len:
                runs.append(cur)
            cur = [(f, d[1], d[2], d[3])] if d else []
    if len(cur) >= min_len:
        runs.append(cur)
    return max(runs, key=len) if runs else []


HALMSTAD_CLASS = {"DRONE": "drone", "BIRD": "bird", "AIRPLANE": "other", "HELICOPTER": "other"}


def halmstad(videos, builder: Builder, every: int = 2, val_mod: int = 5, min_len: int = 16) -> dict:
    stats = {}
    for v in sorted(videos):
        m = re.search(r"V_(DRONE|BIRD|AIRPLANE|HELICOPTER)_(\d+)", Path(v).name.upper())
        if not m:
            continue
        cls, num = HALMSTAD_CLASS[m.group(1)], int(m.group(2))
        frames = [g for _, g in iter_gray(v)]
        run = longest_run([salient_blob(g) for g in frames], min_len=min_len)
        split = 1 if num % val_mod == 0 else 0
        for (f, cx, cy, size) in run[::every]:
            builder.add(crop_patch(frames[f], cx, cy, size), LABEL[cls], size, f, "halmstad", Path(v).stem, split)
        stats[Path(v).stem] = (cls, len(run), float(np.median([r[3] for r in run])) if run else None)
    return stats


# ------------------------------------------------------------------ VOC helpers (ARD-MAV, UAV_SMID)
def voc_boxes(xml_path):
    """[(name, x1, y1, x2, y2)] of a Pascal VOC file; [] for a file without objects."""
    root = ET.parse(xml_path).getroot()
    out = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        if bb is None:
            continue
        nm = obj.find("name")
        name = (nm.text or "").strip().lower() if nm is not None else ""
        x1, y1, x2, y2 = (float(bb.find(t).text) for t in ("xmin", "ymin", "xmax", "ymax"))
        out.append((name, min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
    return out


def ardmav(root, seqs, builder: Builder, val_seqs=(), every: int = 6, seed: int = 0) -> dict:
    """Drone boxes of every `every`-th annotated frame + one background patch per sampled frame."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    stats = {}
    for seq in seqs:
        boxes = {}
        for xf in sorted((root / "Annotations" / seq).glob(f"{seq}_*.xml")):
            try:
                idx = int(xf.stem.split("_")[-1]) - 1          # 1-based XML index -> decoded frame
                boxes[idx] = voc_boxes(xf)
            except (ValueError, ET.ParseError):
                continue
        want = {f: b for f, b in boxes.items() if f % every == 0 and b}
        split = 1 if seq in set(val_seqs) else 0
        n = 0
        if not want:
            continue
        last = max(want)
        for i, g in iter_gray(root / "videos" / f"{seq}.mp4"):
            if i in want:
                H, W = g.shape
                sizes = []
                for (_, x1, y1, x2, y2) in want[i]:
                    s = max(x2 - x1, y2 - y1)
                    sizes.append(s)
                    builder.add(crop_patch(g, (x1 + x2) / 2, (y1 + y2) / 2, s), LABEL["drone"], s, i, "ardmav", seq, split)
                    n += 1
                s = float(sizes[0]) * float(rng.uniform(0.8, 2.0))
                for _ in range(20):                               # a background patch away from every box
                    cx, cy = rng.uniform(2 * s, W - 2 * s), rng.uniform(2 * s, H - 2 * s)
                    if all(math.hypot(cx - (b[1] + b[3]) / 2, cy - (b[2] + b[4]) / 2) >= 3 * max(s, b[3] - b[1], b[4] - b[2])
                           for b in want[i]):
                        builder.add(crop_patch(g, cx, cy, s), LABEL["other"], s, i, "ardmav_bg", seq, split)
                        break
            if i >= last:
                break
        stats[seq] = n
    return stats


UAV_CLASS = {"drone": "drone", "uav": "drone", "bird": "bird", "aeroplane": "other", "airplane": "other",
             "helicopter": "other"}


def uav_select(root, caps=None, seed: int = 0, ann_tar=None) -> list:
    """Single-object UAV_SMID images of the drone / bird / aeroplane+helicopter classes, per split up to caps
    {(split, class): n}, first n after a seeded shuffle of the sorted annotation stems. Returns
    [{split, stem, image, label, box}] (paths relative to root). ann_tar: read the XMLs from a tar of
    '<split>/annotations/*.xml' instead of the directory (one file open instead of thousands)."""
    import io
    import tarfile

    caps = caps or {("train", "drone"): 1000, ("train", "bird"): 1000, ("train", "other"): 400,
                    ("val", "drone"): 250, ("val", "bird"): 250, ("val", "other"): 100}
    root = Path(root)
    data = root / "dataset" if (root / "dataset").is_dir() else root
    tar_xml = {}
    if ann_tar is not None:
        with tarfile.open(ann_tar) as tf:
            for m in tf.getmembers():
                if m.isfile() and m.name.endswith(".xml"):
                    tar_xml[m.name.replace("\\", "/").lstrip("./")] = tf.extractfile(m).read()
    out = []
    for split in sorted({k[0] for k in caps}):
        if ann_tar is not None:
            stems = sorted(Path(k).stem for k in tar_xml if k.startswith(f"{split}/annotations/"))
            src = lambda st, sp=split: io.BytesIO(tar_xml[f"{sp}/annotations/{st}.xml"])  # noqa: E731
        else:
            stems = sorted(p.stem for p in (data / split / "annotations").glob("*.xml"))
            src = lambda st, sp=split: data / sp / "annotations" / f"{st}.xml"  # noqa: E731
        imgs = {}
        img_dir = data / split / "images"
        for nm in (sorted(p.name for p in img_dir.iterdir()) if img_dir.is_dir() else []):
            imgs.setdefault(Path(nm).stem, nm)
        order = np.random.default_rng(seed).permutation(len(stems))
        taken = {}
        for j in order:
            b = voc_boxes(src(stems[j]))
            if len(b) != 1 or b[0][0] not in UAV_CLASS or stems[j] not in imgs:
                continue
            cls = UAV_CLASS[b[0][0]]
            if taken.get(cls, 0) >= caps.get((split, cls), 0):
                continue
            taken[cls] = taken.get(cls, 0) + 1
            img = data / split / "images" / imgs[stems[j]]
            out.append({"split": split, "stem": stems[j], "image": str(img.relative_to(root)).replace("\\", "/"),
                        "label": cls, "box": list(b[0][1:])})
    return out


def render_small(img_gray: np.ndarray, box, target: float, rng, jpeg=(60, 90)):
    """Area-downscale the whole image so the object's long side becomes `target` px, JPEG it, and return
    (stored crop, size) at the object's new centre."""
    import cv2

    x1, y1, x2, y2 = box
    k = float(target) / max(1.0, max(x2 - x1, y2 - y1))
    k = min(1.0, k)
    small = cv2.resize(img_gray, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1.0 else img_gray
    ok, enc = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(jpeg[0], jpeg[1] + 1))])
    small = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    cx, cy, s = (x1 + x2) / 2 * k, (y1 + y2) / 2 * k, max(x2 - x1, y2 - y1) * k
    return crop_patch(small, cx, cy, s), s


def uav_smid(root, selection, builder: Builder, sizes=(16.0, 64.0), n_render: int = 2, seed: int = 0) -> dict:
    import cv2

    rng = np.random.default_rng(seed)
    root = Path(root)
    stats = {}
    for it in selection:
        g = cv2.imread(str(root / it["image"]), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        for _ in range(n_render):
            t = float(np.exp(rng.uniform(np.log(sizes[0]), np.log(sizes[1]))))
            crop, s = render_small(g, it["box"], t, rng)
            builder.add(crop, LABEL[it["label"]], s, 0, "uav_smid", it["stem"], 1 if it["split"] == "val" else 0)
            stats[(it["split"], it["label"])] = stats.get((it["split"], it["label"]), 0) + 1
    return {f"{k[0]}:{k[1]}": v for k, v in stats.items()}


# ------------------------------------------------------------------ dronedet GT
def parse_map(spec: str) -> list:
    """'far=drone,bird*=bird' -> [(glob, class)]."""
    out = []
    for part in spec.split(","):
        k, v = (s.strip() for s in part.split("="))
        if v not in LABEL:
            raise ValueError(f"unknown class {v!r} in --map (expected one of {CLASSES})")
        out.append((k, v))
    return out


def gt_points(gt_json, name_map, every=None) -> list:
    """(frame, cx, cy, size, label) for every box of every GT object whose name matches the map; `every` =
    {class: k} keeps every k-th frame of that class."""
    g = json.loads(Path(gt_json).read_text(encoding="utf-8"))
    every = every or {}
    excl = set(int(f) for f in g.get("meta", {}).get("exclude_frames", []) or [])
    out = []
    for name, o in g["objects"].items():
        cls = next((c for pat, c in name_map if fnmatch.fnmatch(name, pat)), None)
        if cls is None:
            continue
        for f, b in o["frames"].items():
            f = int(f)
            if f in excl or f % int(every.get(cls, 1)) != 0:
                continue
            out.append((f, float(b[0]), float(b[1]), float(max(b[2], b[3])), LABEL[cls]))
    return sorted(out)


# ------------------------------------------------------------------ CLI
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("halmstad")
    h.add_argument("--videos", nargs="+", required=True)
    h.add_argument("--every", type=int, default=2)
    h.add_argument("--val-mod", type=int, default=5)
    a = sub.add_parser("ardmav")
    a.add_argument("--root", required=True)
    a.add_argument("--seqs", nargs="+", required=True)
    a.add_argument("--val-seqs", nargs="*", default=[])
    a.add_argument("--every", type=int, default=6)
    us = sub.add_parser("uav-select")
    us.add_argument("--root", required=True)
    us.add_argument("--ann-tar", default=None)
    u = sub.add_parser("uav_smid")
    u.add_argument("--root", required=True)
    u.add_argument("--select", required=True)
    g = sub.add_parser("gt")
    g.add_argument("--video", required=True)
    g.add_argument("--gt", required=True)
    g.add_argument("--map", required=True)
    g.add_argument("--every", default="", help="e.g. drone=2")
    g.add_argument("--src", default="gt")
    p = sub.add_parser("points")
    p.add_argument("--video", required=True)
    p.add_argument("--points", required=True, help="npz with frame, cx, cy, size, label [, key, split]")
    p.add_argument("--src", default="points")
    for q in (h, a, us, u, g, p):
        q.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    if args.cmd == "uav-select":
        sel = uav_select(args.root, ann_tar=args.ann_tar)
        Path(args.out).write_text(json.dumps(sel), encoding="utf-8")
        print(f"{len(sel)} images selected", flush=True)
        return
    b = Builder()
    if args.cmd == "halmstad":
        stats = halmstad(args.videos, b, args.every, args.val_mod)
    elif args.cmd == "ardmav":
        stats = ardmav(args.root, args.seqs, b, args.val_seqs, args.every)
    elif args.cmd == "uav_smid":
        stats = uav_smid(args.root, json.loads(Path(args.select).read_text(encoding="utf-8")), b)
    elif args.cmd == "gt":
        every = {k: int(v) for k, v in (x.split("=") for x in args.every.split(",") if x)}
        pts = [(f, x, y, s, lab, 0, -1) for (f, x, y, s, lab) in gt_points(args.gt, parse_map(args.map), every)]
        stats = {"crops": crops_from_video(args.video, pts, b, args.src, Path(args.video).stem)}
    else:
        z = np.load(args.points)
        n = len(z["frame"])
        key = z["key"] if "key" in z else np.full(n, -1)
        split = z["split"] if "split" in z else np.zeros(n, int)
        pts = list(zip(z["frame"], z["cx"], z["cy"], z["size"], z["label"], split, key))
        stats = {"crops": crops_from_video(args.video, pts, b, args.src, Path(args.video).stem)}
    counts = b.save(args.out)
    print(json.dumps({"out": args.out, "counts": counts, "n": len(b)}), flush=True)
    print(json.dumps(stats, default=str)[:4000], flush=True)


if __name__ == "__main__":
    main()
