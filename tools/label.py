"""Browser-based frame-by-frame bounding-box labeler for tiny drones.

Zero extra dependencies: a stdlib HTTP server (``http.server``) serves a
single-page canvas app; frames are decoded once, sequentially, JPEG-encoded, and held in memory
for instant random access.

    .venv/bin/python tools/label.py                      # data/videos/07_05.mp4 -> work/labels.json
    .venv/bin/python tools/label.py --from-gt work/gt.json   # seed boxes from existing GT
    .venv/bin/python tools/label.py --export-gt work/gt.json # write labels back as GT and exit
    .venv/bin/python tools/label.py --export-yolo DIR    # write labels as a YOLO dataset and exit

A side panel sets what the next click places: a label from the class list and a box size
(presets or any W x H, in image pixels). A click drops a box of that size centred on the
click - the YOLO convention, where a box is its centre plus its width and height - and a drag
still draws a free box. The panel also edits the selected box and shows its YOLO line.

Labels live in ``work/labels.json`` as per-frame corner boxes::

    {"video": "07_05.mp4", "width": 1280, "height": 720, "n_frames": 591,
     "classes": ["far", "near", "bird"], "exclude_frames": [..],
     "frames": {"0": [{"x1":.., "y1":.., "x2":.., "y2":.., "label":"far"}], ...}}

The UI autosaves; boxes are in original-image pixel coordinates, matching
``dronedet.gt`` / ``dronedet.detections``, and stay inside the image. A frame with no boxes
means "nothing visible"; a frame nobody can judge is marked instead (``exclude_frames``).
GT export groups boxes by label into GT objects (``objects[label].frames[t] = [cx, cy, w, h]``),
carrying over ``meta`` (e.g. stabilization shifts) from an existing GT file when present.
YOLO export writes one ``id cx cy w h`` line per box, normalised to [0, 1], and a
``classes.txt`` whose line i is class id i (``export_yolo``).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import sys
import threading
import webbrowser
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dronedet.video import frames, probe

DEFAULT_CLASSES = ["far", "near", "bird"]

# Repo root, so relative paths (the video, work/*.json) resolve the same way
# whether the tool is launched from the root or from an IDE with a different cwd.
ROOT = Path(__file__).resolve().parents[1]


def _resolve(p: str | Path | None) -> Path | None:
    """Absolute paths are honored; a relative path is used as-is if it already
    exists in the cwd, otherwise anchored to the repo root."""
    if p is None:
        return None
    pp = Path(p)
    if pp.is_absolute() or pp.exists():
        return pp
    return ROOT / pp


def _show(p: Path) -> str:
    """A path as the user typed it: relative to the repo root when inside it."""
    try:
        return Path(p).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


# ---------------------------------------------------------------------------
# Shared server state (guarded by STATE["lock"])
# ---------------------------------------------------------------------------
STATE: dict = {
    "jpegs": [],          # list[bytes] one JPEG per readable frame
    "info": {},           # {video, n, width, height}
    "labels": {},         # {"video","width","height","n_frames","classes","exclude_frames","frames"}
    "labels_path": None,  # Path
    "video_path": None,   # str, for the YOLO image export
    "defaults": {},       # {"gt_out": Path, "yolo_out": Path, "prefix": str, "yolo_classes": list|None}
    # One id per run. Every frame URL and every write carries it: frame URLs are otherwise the
    # same for every video ("/api/frame/12.jpg"), so a browser that cached a frame from the
    # previous run would show it in this one, and a tab left open from the previous run would
    # save its boxes into this run's store.
    "session": "",
    "lock": threading.Lock(),
}


# ---------------------------------------------------------------------------
# Frame loading / label persistence
# ---------------------------------------------------------------------------
def load_frames(video: str, quality: int = 95) -> list[bytes]:
    """Decode the whole video sequentially into per-frame JPEG bytes."""
    info = probe(video)
    print(f"loading {video} ({info.width}x{info.height})...", flush=True)
    jpegs: list[bytes] = []
    enc = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    for idx, frame in frames(video):
        ok, buf = cv2.imencode(".jpg", frame, enc)
        if not ok:
            raise RuntimeError(f"failed to JPEG-encode frame {idx}")
        jpegs.append(buf.tobytes())
        if idx % 100 == 0:
            print(f"  decoded {idx} frames", flush=True)
    print(f"loaded {len(jpegs)} frames", flush=True)
    return jpegs


_NUMBERED = re.compile(r"^(.*?)(?:[#_]\d+)?$")


def base_label(name: str) -> str:
    """The class of a label or GT object name: ``bird#2`` and ``bird_3`` -> ``bird``.

    GT export numbers same-class boxes within a frame (``bird#2``) and hand-built GT numbers
    its tracks (``bird_3``). Neither number is a class, and a YOLO class id must not depend
    on it. ``unsure_drone`` carries no number and stays whole.
    """
    m = _NUMBERED.match(name)
    return m.group(1) if m and m.group(1) else name


def gt_to_labels(gt_path: str | Path) -> tuple[dict[str, list[dict]], list[str], list[int]]:
    """Convert a GT file's objects into per-frame corner boxes.

    Returns (frames_dict, class_names, exclude_frames). GT stores (cx, cy, w, h); we emit
    (x1, y1, x2, y2) tagged with the object's class (``base_label`` of its name).
    """
    raw = json.loads(Path(gt_path).read_text(encoding="utf-8"))
    per_frame: dict[str, list[dict]] = {}
    classes: list[str] = []
    for name, obj in raw.get("objects", {}).items():
        cls = base_label(name)
        if cls not in classes:
            classes.append(cls)
        for f, box in obj.get("frames", {}).items():
            cx, cy, w, h = box[:4]
            per_frame.setdefault(str(int(f)), []).append({
                "x1": round(cx - w / 2, 2), "y1": round(cy - h / 2, 2),
                "x2": round(cx + w / 2, 2), "y2": round(cy + h / 2, 2),
                "label": cls,
            })
    exclude = sorted(int(f) for f in raw.get("meta", {}).get("exclude_frames", []))
    return per_frame, classes, exclude


def labels_to_gt(labels: dict, gt_path: str | Path) -> None:
    """Write per-frame boxes back into GT object format.

    Boxes are grouped by class; a class carrying multiple boxes in the same
    frame is split into ``label#2``, ``label#3`` slots (stable by position).
    ``meta`` from an existing GT at the same path is preserved (shifts etc.); the store's
    ``exclude_frames`` replace the old ones, and the store's size and length fill in
    ``width``/``height``/``n_frames`` when the GT has none, so the scorer can read it alone.
    """
    gt_path = Path(gt_path)
    meta: dict = {}
    prev_ignore: dict[str, bool] = {}
    if gt_path.exists():
        old = json.loads(gt_path.read_text(encoding="utf-8"))
        meta = old.get("meta", {})
        for n, o in old.get("objects", {}).items():
            b = base_label(n)
            prev_ignore[b] = prev_ignore.get(b, False) or bool(o.get("ignore", False))
    for k in ("width", "height", "n_frames"):
        if k in labels and k not in meta:
            meta[k] = labels[k]
    if "exclude_frames" in labels:
        meta["exclude_frames"] = sorted(int(f) for f in labels["exclude_frames"])

    objects: dict[str, dict[str, list]] = {}
    for f, boxes in labels.get("frames", {}).items():
        counts: dict[str, int] = {}
        for b in boxes:
            base = base_label(b.get("label", "obj"))
            counts[base] = counts.get(base, 0) + 1
            name = base if counts[base] == 1 else f"{base}#{counts[base]}"
            cx = (b["x1"] + b["x2"]) / 2.0
            cy = (b["y1"] + b["y2"]) / 2.0
            w = abs(b["x2"] - b["x1"])
            h = abs(b["y2"] - b["y1"])
            objects.setdefault(name, {})[str(int(f))] = [
                round(cx, 2), round(cy, 2), round(w, 2), round(h, 2)]

    payload = {
        "video": labels.get("video", ""),
        "meta": meta,
        "objects": {
            name: {"ignore": prev_ignore.get(base_label(name), False),
                   "frames": dict(sorted(fr.items(), key=lambda kv: int(kv[0])))}
            for name, fr in objects.items()
        },
    }
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    gt_path.write_text(json.dumps(payload), encoding="utf-8")
    print(f"exported {len(objects)} objects to {gt_path}")


# ---------------------------------------------------------------------------
# YOLO export
# ---------------------------------------------------------------------------
def yolo_line(box: dict, cls_id: int, width: int, height: int) -> str | None:
    """One corner box in pixels -> ``"id cx cy w h"``: the YOLO (Darknet / ultralytics)
    label line, each coordinate normalised to [0, 1] by the image size, six decimals.

    The box is clipped to the image first. YOLO coordinates may not leave [0, 1], and
    ultralytics drops an image whose label does as corrupt. A box with no area inside the
    image gives None.
    """
    x1, x2 = sorted((float(box["x1"]), float(box["x2"])))
    y1, y2 = sorted((float(box["y1"]), float(box["y2"])))
    x1, x2 = min(max(x1, 0.0), width), min(max(x2, 0.0), width)
    y1, y2 = min(max(y1, 0.0), height), min(max(y2, 0.0), height)
    if x2 <= x1 or y2 <= y1:
        return None
    return (f"{cls_id} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
            f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}")


def _stem(prefix: str, f: int) -> str:
    return f"{prefix}_{f:06d}"


def _drop_stale(folder: Path, prefix: str, ext: str, keep: set[str]) -> int:
    """Remove this prefix's files from an earlier export that the new one does not write."""
    pat = re.compile(rf"^{re.escape(prefix)}_\d{{6}}\.{ext}$")
    stale = [p for p in folder.glob(f"*.{ext}") if pat.match(p.name) and p.name not in keep]
    for p in stale:
        p.unlink()
    return len(stale)


def export_yolo(labels: dict, out_dir: str | Path, classes: list[str] | None = None,
                prefix: str = "frame", labeled_only: bool = False,
                video: str | Path | None = None) -> dict:
    """Write a label store as a YOLO dataset and return a summary.

    ``labels/<prefix>_<frame:06d>.txt`` holds one ``yolo_line`` per box, and ``classes.txt``
    the class list (line i is class id i). With ``video``, ``images/<prefix>_<frame:06d>.png``
    holds the frame itself, decoded sequentially and written losslessly: a JPEG would
    re-compress the few pixels a tiny drone has.

    - Class ids are positions in ``classes`` (default: the store's list). Every clip of one
      dataset must use the same list, so pass it explicitly when stores differ.
    - A box whose class is not in ``classes`` is skipped and counted, never renumbered.
    - Frames in the store's ``exclude_frames`` get no file. YOLO has no "can't judge", and an
      empty file would teach "nothing here".
    - Every other frame gets a file, an empty one when it has no boxes (YOLO's background
      image; in this store an unboxed frame means "nothing visible"), unless
      ``labeled_only``.
    - This prefix's files from an earlier export that this one does not write are removed,
      so the folder always matches the store.
    """
    out = Path(out_dir)
    classes = list(labels.get("classes", []) if classes is None else classes)
    if len(set(classes)) != len(classes) or not all(classes):
        raise ValueError(f"class names must be unique and non-empty: {classes}")
    ids = {c: i for i, c in enumerate(classes)}
    width, height = int(labels["width"]), int(labels["height"])
    boxes = {int(f): b for f, b in labels.get("frames", {}).items()}
    excluded = {int(f) for f in labels.get("exclude_frames", [])}
    warning = None
    if labeled_only:
        targets = sorted(f for f, b in boxes.items() if b and f not in excluded)
    else:
        n = labels.get("n_frames")
        if n is None:
            n = max(boxes, default=-1) + 1
            warning = ("the store records no n_frames (open it in the labeler once): "
                       f"wrote frames 0..{n - 1} only")
        targets = [f for f in range(int(n)) if f not in excluded]

    lbl_dir = out / "labels"
    lbl_dir.mkdir(parents=True, exist_ok=True)
    stale = _drop_stale(lbl_dir, prefix, "txt", {_stem(prefix, f) + ".txt" for f in targets})
    n_boxes = n_empty = n_outside = 0
    skipped: Counter = Counter()
    for f in targets:
        lines = []
        for b in boxes.get(f, []):
            cls = base_label(str(b.get("label", "")))
            if cls not in ids:
                skipped[cls] += 1
                continue
            line = yolo_line(b, ids[cls], width, height)
            if line is None:
                n_outside += 1
                continue
            lines.append(line)
        n_boxes += len(lines)
        n_empty += not lines
        (lbl_dir / f"{_stem(prefix, f)}.txt").write_text(
            "".join(ln + "\n" for ln in lines), encoding="utf-8", newline="\n")
    (out / "classes.txt").write_text("".join(c + "\n" for c in classes),
                                     encoding="utf-8", newline="\n")

    n_images = 0
    if video is not None:
        img_dir = out / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        wanted = set(targets)
        stale += _drop_stale(img_dir, prefix, "png", {_stem(prefix, f) + ".png" for f in targets})
        last = max(wanted, default=-1)
        for idx, frame in frames(str(video)):
            if idx > last:
                break
            if idx not in wanted:
                continue
            if frame.shape[:2] != (height, width):
                raise ValueError(f"frame {idx} of {video} is {frame.shape[1]}x{frame.shape[0]}, "
                                 f"but the labels are for {width}x{height}")
            ok, buf = cv2.imencode(".png", frame)
            if not ok:
                raise OSError(f"failed to PNG-encode frame {idx}")
            (img_dir / f"{_stem(prefix, idx)}.png").write_bytes(buf.tobytes())
            n_images += 1
        if n_images != len(wanted):
            warning = f"the video decoded {n_images} of the {len(wanted)} frames to write"

    return {"dir": _show(out), "classes": classes, "frames": len(targets),
            "empty_frames": n_empty, "boxes": n_boxes, "images": n_images,
            "excluded": len(excluded), "skipped_class": dict(skipped),
            "outside": n_outside, "stale_removed": stale, "warning": warning}


def default_exports(labels_path: Path) -> tuple[Path, Path, str]:
    """Default GT file, YOLO folder and YOLO file prefix for a label store.

    They sit beside the store, so a store under a private folder exports into that folder;
    only the default store keeps its old GT target, ``work/gt_labeled.json``.
    """
    prefix = labels_path.stem
    if labels_path.resolve() == (ROOT / "work" / "labels.json").resolve():
        gt = ROOT / "work" / "gt_labeled.json"
    else:
        gt = labels_path.with_name(f"{prefix}_gt.json")
    return gt, labels_path.with_name(f"{prefix}_yolo"), prefix


def save_labels() -> None:
    """Atomically persist STATE['labels'] to disk (caller holds the lock)."""
    path: Path = STATE["labels_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(STATE["labels"]), encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str, cache: bool = False):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # only frame URLs, which carry this run's session id, may be cached
        self.send_header("Cache-Control", "private, max-age=3600" if cache else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _current(self) -> bool:
        """Does the request come from a page of this run? Otherwise answer 409: the tab is
        left over from an earlier run (another video) and must not read or write this one."""
        s = parse_qs(urlparse(self.path).query).get("s", [""])[0]
        if s and s == STATE["session"]:
            return True
        self._json(409, {"ok": False, "error": "stale tab: this labeler run has ended; close the tab"})
        return False

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            return
        if path == "/api/info":
            with STATE["lock"]:
                d = STATE["defaults"]
                self._json(200, {
                    "session": STATE["session"],
                    "info": STATE["info"],
                    "classes": STATE["labels"].get("classes", DEFAULT_CLASSES),
                    "labels": STATE["labels"].get("frames", {}),
                    "exclude": STATE["labels"].get("exclude_frames", []),
                    "defaults": {"gt_out": _show(d["gt_out"]), "yolo_out": _show(d["yolo_out"])},
                })
            return
        if path.startswith("/api/frame/"):
            if not self._current():
                return
            try:
                i = int(path.rsplit("/", 1)[1].split(".")[0])
                if i < 0:
                    raise IndexError  # -1 must 404, not wrap to the last frame
                body = STATE["jpegs"][i]
            except (ValueError, IndexError):
                self._send(404, b"no such frame", "text/plain")
                return
            self._send(200, body, "image/jpeg", cache=True)
            return
        self._send(404, b"not found", "text/plain")

    def _read_body(self) -> bytes:
        """Read exactly Content-Length bytes (always, so the next request on a
        keep-alive connection stays framed); tolerate a bad/absent length."""
        try:
            n = int(self.headers.get("Content-Length", 0))
        except (TypeError, ValueError):
            n = 0
        return self.rfile.read(n) if n > 0 else b""

    def _read_json(self, default):
        """Drain + parse the body; returns (obj, ok). The body is always
        consumed before we reply, even on a parse error, to avoid keep-alive
        request desync."""
        raw = self._read_body()
        if not raw:
            return default, True
        try:
            return json.loads(raw), True
        except (ValueError, TypeError):
            return default, False

    def _save_frame(self, i: int, boxes) -> None:
        with STATE["lock"]:
            fr = STATE["labels"].setdefault("frames", {})
            if boxes:
                fr[str(i)] = boxes
            else:
                fr.pop(str(i), None)
            save_labels()

    def _labels_index(self, path: str):
        try:
            return int(path.rsplit("/", 1)[1]), True
        except ValueError:
            return -1, False

    def do_PUT(self) -> None:
        path = urlparse(self.path).path
        boxes, ok = self._read_json([])          # drain body first
        if not path.startswith("/api/labels/"):
            self._send(404, b"not found", "text/plain")
            return
        if not self._current():
            return
        i, ok_i = self._labels_index(path)
        if not ok_i:
            self._send(400, b"bad frame index", "text/plain")
            return
        if not ok:
            self._send(400, b"bad json", "text/plain")
            return
        self._save_frame(i, boxes)
        self._json(200, {"ok": True})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        body, ok = self._read_json(None)         # drain body first, whatever the answer
        if path.startswith("/api/") and not self._current():
            return
        # sendBeacon (fired on tab close) can only POST -> accept the save here
        if path.startswith("/api/labels/"):
            i, ok_i = self._labels_index(path)
            if ok and ok_i:
                self._save_frame(i, body or [])
            self._json(200, {"ok": ok and ok_i})
            return
        if path == "/api/classes":
            classes = body if body is not None else []
            with STATE["lock"]:
                old = STATE["labels"].get("classes", [])
                # append-only: a class's YOLO id is its position, so no class may move
                ok = (ok and isinstance(classes, list) and all(isinstance(c, str) and c for c in classes)
                      and len(set(classes)) == len(classes) and classes[:len(old)] == old)
                if ok:
                    STATE["labels"]["classes"] = classes
                    save_labels()
            self._json(200 if ok else 400, {"ok": ok})
            return
        body = body if isinstance(body, dict) else {}
        if path == "/api/exclude":
            try:
                i, on = int(body["frame"]), bool(body["on"])
            except (KeyError, TypeError, ValueError):
                self._send(400, b"bad exclude request", "text/plain")
                return
            with STATE["lock"]:
                ex = set(STATE["labels"].get("exclude_frames", []))
                if on:
                    ex.add(i)
                else:
                    ex.discard(i)
                STATE["labels"]["exclude_frames"] = sorted(ex)
                save_labels()
            self._json(200, {"ok": True})
            return
        if path == "/api/export_gt":
            out = _resolve(body.get("path") or STATE["defaults"]["gt_out"])
            with STATE["lock"]:
                labels_to_gt(STATE["labels"], out)
            self._json(200, {"ok": True, "path": _show(out)})
            return
        if path == "/api/export_yolo":
            d = STATE["defaults"]
            out = _resolve(body.get("dir") or d["yolo_out"])
            with STATE["lock"]:   # export a snapshot, outside the lock: images take a while
                snap = json.loads(json.dumps(STATE["labels"]))
            try:
                rep = export_yolo(snap, out, d.get("yolo_classes"), d["prefix"],
                                  video=STATE["video_path"] if body.get("images") else None)
            except (ValueError, OSError) as e:
                self._json(200, {"ok": False, "error": str(e)})
                return
            print(f"YOLO export: {json.dumps(rep)}", flush=True)
            self._json(200, {"ok": True, **rep})
            return
        self._send(404, b"not found", "text/plain")


# ---------------------------------------------------------------------------
# Single-page app
# ---------------------------------------------------------------------------
PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>drone labeler</title>
<style>
  :root { --bg:#14161a; --panel:#1d2027; --line:#2c313c; --fg:#e6e9ef;
          --mut:#8a92a6; --accent:#4c9aff; --ok:#3ecf8e; --warn:#ffb020; }
  * { box-sizing: border-box; }
  html,body { margin:0; height:100%; background:var(--bg); color:var(--fg);
              font:13px/1.4 ui-sans-serif,system-ui,Segoe UI,Roboto,sans-serif;
              overflow:hidden; -webkit-user-select:none; user-select:none; }
  #app { display:grid; grid-template-rows:auto 1fr auto; height:100%; }
  header { display:flex; align-items:center; gap:10px; padding:6px 10px;
           background:var(--panel); border-bottom:1px solid var(--line); flex-wrap:wrap; }
  header .grp { display:flex; align-items:center; gap:6px; }
  header .sep { width:1px; height:22px; background:var(--line); margin:0 2px; }
  button { background:#272b34; color:var(--fg); border:1px solid var(--line);
           border-radius:6px; padding:5px 9px; cursor:pointer; font-size:12px; }
  button:hover { background:#30353f; }
  button.on { background:var(--accent); border-color:var(--accent); color:#04101f; font-weight:600; }
  button.warn.on { background:var(--warn); border-color:var(--warn); }
  input[type=number], input[type=text], select { background:#0e1014; color:var(--fg);
           border:1px solid var(--line); border-radius:6px; padding:4px 6px; font-size:12px; }
  .title { font-weight:700; letter-spacing:.3px; }
  .mut { color:var(--mut); }
  #main { display:flex; min-height:0; }
  #stage { position:relative; overflow:hidden; background:#0a0b0d; flex:1; min-width:0; }
  #view { position:absolute; inset:0; width:100%; height:100%; display:block; cursor:crosshair; outline:none; }
  #loupe { position:absolute; right:12px; top:12px; width:200px; height:200px;
           border:1px solid var(--line); border-radius:8px; background:#000;
           box-shadow:0 6px 24px rgba(0,0,0,.5); pointer-events:none; opacity:0; transition:opacity .12s; }
  #loupe.show { opacity:1; }
  #exclbanner { position:absolute; left:50%; top:10px; transform:translateX(-50%); pointer-events:none;
                background:rgba(255,176,32,.92); color:#1a1200; font-weight:600; border-radius:8px;
                padding:5px 12px; font-size:12px; }
  #panel { width:264px; flex:none; overflow-y:auto; background:var(--panel);
           border-left:1px solid var(--line); padding:10px 12px; display:flex;
           flex-direction:column; gap:16px; }
  #panel h3 { margin:0 0 6px; font-size:11px; text-transform:uppercase; letter-spacing:.6px;
              color:var(--mut); font-weight:600; display:flex; justify-content:space-between; }
  #panel h3 b { color:var(--fg); text-transform:none; letter-spacing:0; }
  .note { color:var(--mut); font-size:11px; margin-top:6px; line-height:1.5; }
  .row { display:flex; align-items:center; gap:6px; margin-top:6px; }
  .swatch { width:12px; height:12px; border-radius:3px; flex:none; }
  .cls { display:flex; align-items:center; gap:8px; padding:4px 6px; border-radius:6px;
         cursor:pointer; border:1px solid transparent; }
  .cls:hover { background:#232730; }
  .cls.active { border-color:var(--accent); background:#232a36; }
  .cls .id { margin-left:auto; color:var(--mut); font-size:11px; font-variant-numeric:tabular-nums; }
  .chips { display:flex; flex-wrap:wrap; gap:4px; }
  .chips button { min-width:38px; padding:4px 6px; font-variant-numeric:tabular-nums; }
  .num { width:66px; }
  .grid4 { display:grid; grid-template-columns:auto 1fr auto 1fr; gap:6px 6px; align-items:center; margin-top:6px; }
  .grid4 input { width:100%; }
  code.yolo { display:block; margin-top:4px; background:#0e1014; border:1px solid var(--line);
              border-radius:6px; padding:5px 7px; font:11px/1.5 ui-monospace,Consolas,monospace;
              -webkit-user-select:text; user-select:text; word-break:break-all; }
  .keys { color:var(--mut); font-size:11px; line-height:1.9; }
  .keys b { color:var(--fg); font-weight:600; }
  footer { display:flex; align-items:center; gap:12px; padding:6px 10px; background:var(--panel);
           border-top:1px solid var(--line); }
  #scrub { flex:1; }
  #timeline { position:relative; height:14px; flex:2; background:#0e1014;
              border:1px solid var(--line); border-radius:4px; overflow:hidden; cursor:pointer; }
  .tick { position:absolute; top:0; bottom:0; width:2px; background:var(--ok); opacity:.55; }
  .tick.ex { background:var(--warn); opacity:.85; }
  #cursorline { position:absolute; top:-2px; bottom:-2px; width:2px; background:var(--accent); }
  .pill { padding:2px 8px; border-radius:999px; background:#0e1014; border:1px solid var(--line);
          font-variant-numeric:tabular-nums; }
  #saved.dirty { color:var(--warn); }
  #saved.ok { color:var(--ok); }
  kbd { background:#0e1014; border:1px solid var(--line); border-bottom-width:2px;
        border-radius:4px; padding:0 4px; font-size:10px; color:var(--fg); }
  [hidden] { display:none !important; }
</style>
</head>
<body>
<div id="app">
  <header>
    <span class="title">🛰 drone labeler</span>
    <span class="sep"></span>
    <div class="grp">
      <button id="prev" title="prev frame (,)">◀</button>
      <input id="frameno" type="number" min="0" value="0" style="width:72px" title="go to frame">
      <span class="mut" id="framecnt">/ 0</span>
      <button id="next" title="next frame (.)">▶</button>
    </div>
    <span class="sep"></span>
    <div class="grp">
      <button id="copyprev" title="copy boxes from previous frame (c)">⧉ copy prev</button>
      <button id="carry" title="carry boxes onto next empty frame">carry ▶</button>
      <button id="clear" title="delete all boxes on this frame">✕ clear</button>
      <button id="exclude" class="warn" title="nobody can judge this frame: leave it out of scoring and of the YOLO export (x)">⊘ can't judge</button>
    </div>
    <span class="sep"></span>
    <div class="grp">
      <button id="zoomout" title="zoom out (-)">−</button>
      <span class="pill" id="zoomlbl">100%</span>
      <button id="zoomin" title="zoom in (+)">+</button>
      <button id="fit" title="fit (f)">fit</button>
      <button id="one" title="1:1 (0)">1:1</button>
    </div>
    <span class="sep"></span>
    <div class="grp">
      <button id="exportgt" title="export to GT json">⬇ GT</button>
      <button id="exportyolo" title="export YOLO labels (one txt per frame + classes.txt)">⬇ YOLO</button>
      <span id="saved" class="pill ok">saved</span>
    </div>
  </header>

  <div id="main">
    <div id="stage">
      <canvas id="view" tabindex="0"></canvas>
      <canvas id="loupe" width="200" height="200"></canvas>
      <div id="exclbanner" hidden>can't judge: left out of scoring and YOLO export · <kbd>x</kbd> undoes</div>
    </div>

    <aside id="panel">
      <section>
        <h3>Next box · label <b id="nextlbl"></b></h3>
        <div id="classlist"></div>
        <div class="row">
          <input id="newcls" type="text" placeholder="new class" style="flex:1;min-width:0">
          <button id="addcls">add</button>
        </div>
        <div class="note">The YOLO id is the class's position in this list. New classes go to the
          end, so no id ever changes.</div>
      </section>

      <section>
        <h3>Next box · size <b id="sizelbl"></b></h3>
        <div id="sizes" class="chips"></div>
        <div class="row">
          <span class="mut">W</span><input id="bw" class="num" type="number" min="0.5" step="1">
          <button id="sqlock" title="keep the box square">■ square</button>
          <span class="mut">H</span><input id="bh" class="num" type="number" min="0.5" step="1">
        </div>
        <div class="note">Image pixels. A click places this box centred on the cursor (the dashed
          preview); a drag draws a free box.</div>
      </section>

      <section>
        <h3>Selected box</h3>
        <div id="selnone" class="mut">none: click a box to select it</div>
        <div id="selbody" hidden>
          <div class="row"><span class="mut">label</span><select id="sellabel" style="flex:1;min-width:0"></select></div>
          <div class="grid4">
            <span class="mut">cx</span><input id="scx" type="number" step="0.5">
            <span class="mut">cy</span><input id="scy" type="number" step="0.5">
            <span class="mut">w</span><input id="sw" type="number" min="0.5" step="1">
            <span class="mut">h</span><input id="sh" type="number" min="0.5" step="1">
          </div>
          <div class="row"><span class="mut">YOLO line (id cx cy w h, normalised)</span></div>
          <code class="yolo" id="yololine"></code>
          <div class="row">
            <button id="selsize" title="resize the selected box to the next-box size, same centre (s)">↺ next size</button>
            <button id="seldel" title="delete (Del)">✕ delete</button>
          </div>
        </div>
      </section>

      <section class="keys">
        <h3>Keys</h3>
        <b>click</b> place box · <b>drag</b> free box · <b>Shift+click</b> place on top of a box<br>
        <b>drag a box</b> move · <b>handles</b> resize · <b>space/middle-drag</b> pan · <b>wheel</b> zoom<br>
        <kbd>1</kbd>–<kbd>9</kbd> next label · <kbd>[</kbd> <kbd>]</kbd> next size<br>
        <kbd>Shift</kbd>+<kbd>1</kbd>–<kbd>9</kbd> relabel selected · <kbd>s</kbd> selected → next size<br>
        <kbd>,</kbd> <kbd>.</kbd> frame · <kbd>c</kbd> copy prev · <kbd>x</kbd> can't judge<br>
        <kbd>Del</kbd> delete · arrows nudge (Shift ×5, Alt ×¼) · <kbd>f</kbd> fit · <kbd>Esc</kbd> deselect
      </section>
    </aside>
  </div>

  <footer>
    <input id="scrub" type="range" min="0" max="0" value="0">
    <div id="timeline"><div id="cursorline" style="left:0"></div></div>
    <span class="pill" id="posinfo">–</span>
    <span class="pill" id="boxinfo">0 boxes</span>
  </footer>
</div>

<script>
"use strict";
const $ = s => document.querySelector(s);
const view = $("#view"), vctx = view.getContext("2d");
const loupe = $("#loupe"), lctx = loupe.getContext("2d");

const PALETTE = ["#4c9aff","#3ecf8e","#ffb020","#ff6b6b","#c084fc",
                 "#22d3ee","#f472b6","#a3e635","#fb923c","#94a3b8"];
const SIZE_PRESETS = [4,6,8,12,16,24,32,48,64,96,128,192,256];
const PREFS_KEY = "specklock-labeler-next";

let INFO = {n:0, width:1280, height:720, video:""};
let DEFAULTS = {gt_out:"work/gt_labeled.json", yolo_out:""};
let SESSION = "";          // this run's id: every frame URL and write carries it
let STALE = false;         // the run this tab belongs to has ended
let CLASSES = [];
let LABELS = {};            // {frameIndex: [ {x1,y1,x2,y2,label} ]}
let EXCLUDED = new Set();   // frames nobody can judge
let cur = 0;               // current frame index
let boxes = [];            // boxes of the current frame (working copy)
let activeClass = 0;       // label of the next placed box
let nextW = 8, nextH = 8;  // size of the next placed box (image px)
let square = true;
let selected = -1;         // index into boxes
let carry = false;
let ghost = null;          // [ix, iy]: where a click would place the next box

// view transform: screen = image*scale + (ox,oy)
let scale = 1, ox = 0, oy = 0;
let img = new Image();
const imgCache = new Map();  // idx -> Image

// ---- helpers ----
function baseLabel(name){ const m=/^(.*?)(?:[#_]\d+)?$/.exec(name||""); return (m && m[1]) || name || ""; }
function classColor(name){ const i = CLASSES.indexOf(baseLabel(name)); return PALETTE[(i<0?0:i)%PALETTE.length]; }
function s2i(sx, sy){ return [(sx-ox)/scale, (sy-oy)/scale]; }
function i2s(ix, iy){ return [ix*scale+ox, iy*scale+oy]; }
function clampFrame(i){ return Math.max(0, Math.min(INFO.n-1, i|0)); }
function clamp(v, a, b){ return Math.max(a, Math.min(b, v)); }
function norm(b){ return {x1:Math.min(b.x1,b.x2), y1:Math.min(b.y1,b.y2),
                          x2:Math.max(b.x1,b.x2), y2:Math.max(b.y1,b.y2), label:b.label}; }
// YOLO boxes may not leave the image: a box over the edge keeps only its inside part
function clip(b){ const n=norm(b);
  return {x1:clamp(n.x1,0,INFO.width), y1:clamp(n.y1,0,INFO.height),
          x2:clamp(n.x2,0,INFO.width), y2:clamp(n.y2,0,INFO.height), label:n.label}; }
// a moved box keeps its size and slides back inside (clipped only if larger than the image)
function keepInside(b){ const n=norm(b);
  const dx = n.x1<0 ? -n.x1 : (n.x2>INFO.width ? INFO.width-n.x2 : 0);
  const dy = n.y1<0 ? -n.y1 : (n.y2>INFO.height ? INFO.height-n.y2 : 0);
  return clip({x1:n.x1+dx, y1:n.y1+dy, x2:n.x2+dx, y2:n.y2+dy, label:n.label}); }
function boxAt(cx, cy, w, h, label){ return clip({x1:cx-w/2, y1:cy-h/2, x2:cx+w/2, y2:cy+h/2, label}); }
function yoloLine(raw){
  const b=clip(raw), w=b.x2-b.x1, h=b.y2-b.y1, id=CLASSES.indexOf(baseLabel(b.label));
  if(!(w>0 && h>0)) return "(outside the image: not exported)";
  const f=v=>v.toFixed(6);
  return `${id<0?"?":id} ${f((b.x1+b.x2)/2/INFO.width)} ${f((b.y1+b.y2)/2/INFO.height)} ${f(w/INFO.width)} ${f(h/INFO.height)}`
         + (id<0 ? "   (class not in the list: not exported)" : "");
}
function fmtPx(v){ return String(Math.round(v*100)/100); }
function api(path){ return path + (path.includes("?") ? "&" : "?") + "s=" + SESSION; }
// a tab left open from an earlier run (another video, or a restart) must neither show that
// run's frames nor save into it: the server answers it 409, and the tab stops
function staleTab(){
  if(STALE) return; STALE=true;
  setSaved("⚠ old tab: close it","dirty"); document.title="⚠ old · "+document.title;
  alert("This tab belongs to a labeler run that has ended (another video, or a restart).\n"
        + "Nothing more is saved from it. Close it and use the tab of the run you started.");
}
function checkStale(){ fetch("/api/info").then(r=>r.json()).then(j=>{ if(j.session!==SESSION) staleTab(); }).catch(()=>{}); }
async function post(path, obj){
  const r = await fetch(api(path), {method:"POST", body:JSON.stringify(obj)});
  if(r.status===409){ staleTab(); throw new Error("old tab"); }
  return r;
}

function resize(){
  const r = $("#stage").getBoundingClientRect();
  view.width = Math.round(r.width); view.height = Math.round(r.height);
  draw();
}

function fitView(){
  scale = Math.min(view.width/INFO.width, view.height/INFO.height);
  ox = (view.width - INFO.width*scale)/2;
  oy = (view.height - INFO.height*scale)/2;
  draw(); syncUI();
}

function zoomAt(sx, sy, factor){
  const [ix, iy] = s2i(sx, sy);
  scale = Math.max(0.05, Math.min(80, scale*factor));
  ox = sx - ix*scale; oy = sy - iy*scale;
  draw(); $("#zoomlbl").textContent=Math.round(scale*100)+"%";
}

// ---- next-box settings (remembered in this browser) ----
function loadPrefs(){
  try{
    const p = JSON.parse(localStorage.getItem(PREFS_KEY) || "{}");
    if(p.w>0) nextW=p.w; if(p.h>0) nextH=p.h;
    if(typeof p.square==="boolean") square=p.square;
    const i = CLASSES.indexOf(p.cls); if(i>=0) activeClass=i;
  }catch(_){}
}
function savePrefs(){
  try{ localStorage.setItem(PREFS_KEY, JSON.stringify({w:nextW, h:nextH, square, cls:CLASSES[activeClass]})); }catch(_){}
}
function setActiveClass(i){ activeClass=i; savePrefs(); renderClasses(); draw(); }
function setNextSize(w, h){
  nextW=Math.max(0.5, Math.round((+w||nextW)*100)/100);
  nextH=Math.max(0.5, Math.round((+h||nextH)*100)/100);
  savePrefs(); renderSizes(); draw();
}
function stepSize(d){
  const p = d>0 ? (SIZE_PRESETS.find(s=>s>nextW) ?? SIZE_PRESETS[SIZE_PRESETS.length-1])
                : ([...SIZE_PRESETS].reverse().find(s=>s<nextW) ?? SIZE_PRESETS[0]);
  setNextSize(p, square ? p : p*nextH/nextW);
}

// ---- selected-box edits ----
function relabelSelected(name){
  if(selected<0) return;
  boxes[selected].label=name; draw(); markDirty(); syncUI();
}
function resizeSelected(w, h){
  if(selected<0) return;
  const b=norm(boxes[selected]);
  boxes[selected]=boxAt((b.x1+b.x2)/2, (b.y1+b.y2)/2, w, h, b.label);
  draw(); markDirty(); syncUI();
}
function editSelected(){
  if(selected<0) return;
  const cx=+$("#scx").value, cy=+$("#scy").value, w=+$("#sw").value, h=+$("#sh").value;
  if([cx,cy,w,h].every(Number.isFinite) && w>0 && h>0){
    boxes[selected]=boxAt(cx, cy, w, h, boxes[selected].label); draw(); markDirty();
  }
  syncUI();
}
function deleteSelected(){
  if(selected<0) return;
  boxes.splice(selected,1); selected=-1; draw(); markDirty(); syncUI();
}

// ---- frame loading & prefetch ----
function frameImg(i){
  if(imgCache.has(i)) return imgCache.get(i);
  const im = new Image();
  im.onerror = checkStale;
  im.src = api(`/api/frame/${i}.jpg`);
  imgCache.set(i, im);
  if(imgCache.size > 60){ const k = imgCache.keys().next().value; imgCache.delete(k); }
  return im;
}
function prefetch(i){ for(let d=-2; d<=4; d++){ const j=i+d; if(j>=0&&j<INFO.n) frameImg(j); } }

function loadFrame(i, {commit=true}={}){
  if(commit) flush();
  cur = clampFrame(i);
  const wasEmpty = !(LABELS[cur] && LABELS[cur].length);
  boxes = (LABELS[cur] || []).map(b => ({...b}));
  if(carry && wasEmpty && cur>0 && LABELS[cur-1]){
    boxes = LABELS[cur-1].map(b => ({...b}));
    markDirty();
  }
  selected = -1;
  img = frameImg(cur);
  if(img.complete) draw(); else img.onload = draw;
  prefetch(cur);
  syncUI();
}

// ---- drawing ----
function draw(){
  vctx.setTransform(1,0,0,1,0,0);
  vctx.clearRect(0,0,view.width,view.height);
  vctx.fillStyle = "#0a0b0d"; vctx.fillRect(0,0,view.width,view.height);
  vctx.imageSmoothingEnabled = false;
  if(img && img.complete && img.naturalWidth){
    vctx.drawImage(img, ox, oy, INFO.width*scale, INFO.height*scale);
  }
  if(EXCLUDED.has(cur)){
    vctx.fillStyle = "rgba(255,176,32,.12)"; vctx.fillRect(ox, oy, INFO.width*scale, INFO.height*scale);
  }
  // image border
  vctx.strokeStyle = "#2c313c"; vctx.lineWidth = 1;
  vctx.strokeRect(ox+.5, oy+.5, INFO.width*scale, INFO.height*scale);

  for(let k=0;k<boxes.length;k++){
    const b = norm(boxes[k]);
    const [x1,y1] = i2s(b.x1,b.y1), [x2,y2] = i2s(b.x2,b.y2);
    const col = classColor(b.label);
    vctx.lineWidth = k===selected ? 2 : 1.4;
    vctx.strokeStyle = col;
    vctx.strokeRect(x1, y1, x2-x1, y2-y1);
    // crosshair through center for pinpoint tiny boxes
    if(k===selected){
      const cx=(x1+x2)/2, cy=(y1+y2)/2;
      vctx.strokeStyle = col+"80"; vctx.lineWidth=1;
      vctx.beginPath(); vctx.moveTo(cx-8,cy); vctx.lineTo(cx+8,cy);
      vctx.moveTo(cx,cy-8); vctx.lineTo(cx,cy+8); vctx.stroke();
      for(const [hx,hy] of handlePts(b)){
        vctx.fillStyle = "#0a0b0d"; vctx.fillRect(hx-4,hy-4,8,8);
        vctx.strokeStyle = col; vctx.lineWidth=1.5; vctx.strokeRect(hx-4,hy-4,8,8);
      }
    }
    // label tag
    const tag = b.label;
    vctx.font = "11px ui-sans-serif";
    const tw = vctx.measureText(tag).width + 8;
    const ty = y1 > 14 ? y1-14 : y2+2;
    vctx.fillStyle = col; vctx.fillRect(x1, ty, tw, 13);
    vctx.fillStyle = "#04101f"; vctx.fillText(tag, x1+4, ty+10);
  }
  // where a click would put the next box
  if(ghost && !drag){
    const g = boxAt(ghost[0], ghost[1], nextW, nextH, "");
    const [x1,y1] = i2s(g.x1,g.y1), [x2,y2] = i2s(g.x2,g.y2);
    vctx.setLineDash([4,3]); vctx.lineWidth = 1;
    vctx.strokeStyle = PALETTE[activeClass%PALETTE.length];
    vctx.strokeRect(x1, y1, x2-x1, y2-y1); vctx.setLineDash([]);
  }
}

function handlePts(b){  // 8 handles in screen space
  const [x1,y1]=i2s(b.x1,b.y1), [x2,y2]=i2s(b.x2,b.y2);
  const mx=(x1+x2)/2, my=(y1+y2)/2;
  return [[x1,y1,"nw"],[mx,y1,"n"],[x2,y1,"ne"],[x2,my,"e"],
          [x2,y2,"se"],[mx,y2,"s"],[x1,y2,"sw"],[x1,my,"w"]];
}

// ---- hit testing ----
function hitHandle(sx, sy){
  if(selected<0) return null;
  for(const [hx,hy,tag] of handlePts(norm(boxes[selected])))
    if(Math.abs(sx-hx)<=6 && Math.abs(sy-hy)<=6) return tag;
  return null;
}
function hitBox(sx, sy){
  const [ix,iy]=s2i(sx,sy); const tol=6/scale;
  for(let k=boxes.length-1;k>=0;k--){
    const b=norm(boxes[k]);
    if(ix>=b.x1-tol && ix<=b.x2+tol && iy>=b.y1-tol && iy<=b.y2+tol) return k;
  }
  return -1;
}
function insideImage(ix, iy){ return ix>=0 && iy>=0 && ix<=INFO.width && iy<=INFO.height; }

// ---- loupe (magnifier under cursor) ----
function drawLoupe(sx, sy){
  if(!img || !img.complete || !img.naturalWidth){ loupe.classList.remove("show"); return; }
  const [ix,iy]=s2i(sx,sy);
  if(!insideImage(ix, iy)){ loupe.classList.remove("show"); return; }
  loupe.classList.add("show");
  // dock in the corner farthest from the cursor so it never occludes the
  // target (which enters from the top-right) or sits under the pointer.
  loupe.style.left = sx < view.width/2 ? "auto" : "12px";
  loupe.style.right = sx < view.width/2 ? "12px" : "auto";
  loupe.style.top = sy < view.height/2 ? "auto" : "12px";
  loupe.style.bottom = sy < view.height/2 ? "12px" : "auto";
  // 8x for tiny boxes; less when the next box would not fit
  const mag=clamp(loupe.width/Math.max(25, 2.2*Math.max(nextW, nextH)), 1, 8), half=loupe.width/(2*mag);
  lctx.imageSmoothingEnabled=false;
  lctx.clearRect(0,0,loupe.width,loupe.height);
  lctx.drawImage(img, ix-half, iy-half, half*2, half*2, 0,0,loupe.width,loupe.height);
  // draw boxes in loupe space
  const L = (x,y)=>[(x-(ix-half))*mag, (y-(iy-half))*mag];
  for(const raw of boxes){ const b=norm(raw); const col=classColor(b.label);
    const [lx1,ly1]=L(b.x1,b.y1),[lx2,ly2]=L(b.x2,b.y2);
    lctx.strokeStyle=col; lctx.lineWidth=1.5; lctx.strokeRect(lx1,ly1,lx2-lx1,ly2-ly1); }
  if(ghost && !drag){
    const g=boxAt(ghost[0], ghost[1], nextW, nextH, ""); const [gx1,gy1]=L(g.x1,g.y1), [gx2,gy2]=L(g.x2,g.y2);
    lctx.setLineDash([4,3]); lctx.strokeStyle=PALETTE[activeClass%PALETTE.length]; lctx.lineWidth=1.5;
    lctx.strokeRect(gx1,gy1,gx2-gx1,gy2-gy1); lctx.setLineDash([]);
  }
  // crosshair
  lctx.strokeStyle="#ffffff90"; lctx.lineWidth=1;
  lctx.beginPath(); lctx.moveTo(loupe.width/2,0); lctx.lineTo(loupe.width/2,loupe.height);
  lctx.moveTo(0,loupe.height/2); lctx.lineTo(loupe.width,loupe.height/2); lctx.stroke();
}

// ---- pointer interaction ----
let drag = null;  // {mode, sx0, sy0, ...}
view.addEventListener("mousedown", e=>{
  view.focus();
  const sx=e.offsetX, sy=e.offsetY;
  const panning = e.button===1 || spaceDown;
  if(panning){ drag={mode:"pan", sx0:sx, sy0:sy, ox0:ox, oy0:oy}; e.preventDefault(); return; }
  if(e.button!==0) return;
  if(!e.shiftKey){   // Shift+click always places a new box, even on top of another
    const h = hitHandle(sx, sy);
    if(h){ drag={mode:"resize", handle:h, k:selected, box0:{...norm(boxes[selected])}}; return; }
    const k = hitBox(sx, sy);
    if(k>=0){
      selected=k;
      const [ix,iy]=s2i(sx,sy);
      drag={mode:"move", k, ix0:ix, iy0:iy, box0:{...norm(boxes[k])}};
      draw(); syncUI(); return;
    }
  }
  // new box: a click places the next-box size and label, a drag draws a free box
  const [ix,iy]=s2i(sx,sy);
  if(!insideImage(ix, iy)) return;
  if(!CLASSES.length){ alert("add a class first"); return; }
  boxes.push({x1:ix,y1:iy,x2:ix,y2:iy,label:CLASSES[activeClass]});
  selected=boxes.length-1;
  drag={mode:"draw", k:selected, sx0:sx, sy0:sy, ix0:ix, iy0:iy};
  draw(); syncUI();
});

window.addEventListener("mousemove", e=>{
  const rect=view.getBoundingClientRect();
  const sx=e.clientX-rect.left, sy=e.clientY-rect.top;
  const over = sx>=0&&sy>=0&&sx<=rect.width&&sy<=rect.height && e.target===view;
  let g = null;
  if(over){
    const [ix,iy]=s2i(sx,sy);
    $("#posinfo").textContent = `x ${ix.toFixed(1)}  y ${iy.toFixed(1)}`;
    if(!drag && !spaceDown && insideImage(ix,iy) && (e.shiftKey || (hitBox(sx,sy)<0 && !hitHandle(sx,sy)))) g=[ix,iy];
  }
  const moved = (g===null) !== (ghost===null) || (g && ghost && (g[0]!==ghost[0] || g[1]!==ghost[1]));
  ghost = g;
  if(over && (!drag || drag.mode!=="pan")) drawLoupe(sx,sy);
  if(!drag){ if(moved) draw(); return; }
  if(drag.mode==="pan"){ ox=drag.ox0+(sx-drag.sx0); oy=drag.oy0+(sy-drag.sy0); draw(); return; }
  const [ix,iy]=s2i(sx,sy);
  if(drag.mode==="draw"){ const b=boxes[drag.k]; b.x2=ix; b.y2=iy; draw(); markDirty(); }
  else if(drag.mode==="move"){
    const b=boxes[drag.k], o=drag.box0, dx=ix-drag.ix0, dy=iy-drag.iy0;
    b.x1=o.x1+dx; b.y1=o.y1+dy; b.x2=o.x2+dx; b.y2=o.y2+dy; draw(); markDirty(); renderSelected();
  } else if(drag.mode==="resize"){
    // non-dragged edges come from the fixed anchor box0, so dragging a handle
    // past the opposite edge pivots the box instead of collapsing it.
    const b=boxes[drag.k], o=drag.box0, h=drag.handle;
    b.x1 = h.includes("w") ? ix : o.x1;
    b.x2 = h.includes("e") ? ix : o.x2;
    b.y1 = h.includes("n") ? iy : o.y1;
    b.y2 = h.includes("s") ? iy : o.y2;
    draw(); markDirty(); renderSelected();
  }
});
view.addEventListener("mouseleave", ()=>{ if(ghost){ ghost=null; draw(); } loupe.classList.remove("show"); });

window.addEventListener("mouseup", e=>{
  if(drag && drag.mode==="draw"){
    const r=view.getBoundingClientRect();
    const dist=Math.hypot(e.clientX-r.left-drag.sx0, e.clientY-r.top-drag.sy0);
    const label=boxes[drag.k].label;
    // under 4 screen px of travel is a click: the next-box size, centred where it landed
    boxes[drag.k] = dist<4 ? boxAt(drag.ix0, drag.iy0, nextW, nextH, label) : clip(boxes[drag.k]);
    const b=boxes[drag.k];
    if(!(b.x2-b.x1>0 && b.y2-b.y1>0)){ boxes.splice(drag.k,1); selected=-1; }
    draw(); markDirty();
  } else if(drag && drag.mode==="move"){
    boxes[drag.k]=keepInside(boxes[drag.k]); draw(); markDirty();
  } else if(drag && drag.mode==="resize"){
    boxes[drag.k]=clip(boxes[drag.k]); draw(); markDirty();
  }
  if(drag){ drag=null; syncUI(); }
});

view.addEventListener("wheel", e=>{
  e.preventDefault();
  zoomAt(e.offsetX, e.offsetY, e.deltaY<0 ? 1.15 : 1/1.15);
}, {passive:false});

// ---- keyboard ----
// e.code names the physical key, so the shortcuts also work in a Hebrew (or any) layout
let spaceDown=false;
window.addEventListener("keydown", e=>{
  const tag=e.target.tagName;
  if(tag==="INPUT" || tag==="SELECT" || tag==="TEXTAREA") return;
  if(e.ctrlKey || e.metaKey) return;   // leave browser shortcuts alone
  const c=e.code, k=e.key;
  if(c==="Space"){ spaceDown=true; view.style.cursor="grab"; e.preventDefault(); return; }
  if(c==="Comma"||c==="KeyA"){ loadFrame(cur-1); e.preventDefault(); return; }
  if(c==="Period"||c==="KeyD"){ loadFrame(cur+1); e.preventDefault(); return; }
  if(k==="PageDown"){ loadFrame(cur-10); e.preventDefault(); return; }
  if(k==="PageUp"){ loadFrame(cur+10); e.preventDefault(); return; }
  if(c==="KeyC"){ copyPrev(); return; }
  if(c==="KeyF"){ fitView(); return; }
  if(c==="KeyX"){ toggleExclude(); return; }
  if(c==="KeyS"){ resizeSelected(nextW, nextH); return; }
  if(c==="BracketLeft"){ stepSize(-1); e.preventDefault(); return; }
  if(c==="BracketRight"){ stepSize(1); e.preventDefault(); return; }
  if(c==="Digit0"||c==="Numpad0"){ zoomAt(view.width/2,view.height/2, 1/scale); return; }
  if(c==="Equal"||c==="NumpadAdd"){ zoomAt(view.width/2,view.height/2,1.2); return; }
  if(c==="Minus"||c==="NumpadSubtract"){ zoomAt(view.width/2,view.height/2,1/1.2); return; }
  if(k==="Escape"){ selected=-1; draw(); syncUI(); return; }
  if(k==="Delete"||k==="Backspace"){ deleteSelected(); e.preventDefault(); return; }
  const dm=/^(?:Digit|Numpad)([1-9])$/.exec(c);
  if(dm){
    const idx=parseInt(dm[1])-1;
    if(idx<CLASSES.length){ if(e.shiftKey) relabelSelected(CLASSES[idx]); else setActiveClass(idx); }
    e.preventDefault(); return;
  }
  // arrows: nudge selected box, else navigate
  if(k.startsWith("Arrow")){
    e.preventDefault();
    if(selected>=0){
      const step = e.shiftKey?5:(e.altKey?0.25:1);
      const b=boxes[selected];
      if(k==="ArrowLeft"){ b.x1-=step; b.x2-=step; }
      if(k==="ArrowRight"){ b.x1+=step; b.x2+=step; }
      if(k==="ArrowUp"){ b.y1-=step; b.y2-=step; }
      if(k==="ArrowDown"){ b.y1+=step; b.y2+=step; }
      boxes[selected]=keepInside(b);
      draw(); markDirty(); renderSelected();
    } else {
      if(k==="ArrowLeft") loadFrame(cur-1);
      if(k==="ArrowRight") loadFrame(cur+1);
    }
  }
});
window.addEventListener("keyup", e=>{ if(e.code==="Space"){ spaceDown=false; view.style.cursor="crosshair"; } });

// ---- actions ----
function copyPrev(){
  if(cur<=0) return;
  const prev = LABELS[cur-1] || [];
  boxes = boxes.concat(prev.map(b=>({...b})));
  draw(); markDirty(); syncUI();
}
function toggleExclude(){
  const f=cur, on=!EXCLUDED.has(f);
  if(on) EXCLUDED.add(f); else EXCLUDED.delete(f);
  draw(); renderTimeline(); syncUI();
  post("/api/exclude", {frame:f, on})
    .then(r=>{ if(!r.ok) throw new Error("http "+r.status); })
    .catch(()=>{ if(on) EXCLUDED.delete(f); else EXCLUDED.add(f);
                 draw(); renderTimeline(); syncUI(); if(!STALE) alert("could not save the can't-judge mark"); });
}
$("#copyprev").onclick=copyPrev;
$("#clear").onclick=()=>{ boxes=[]; selected=-1; draw(); markDirty(); syncUI(); };
$("#exclude").onclick=()=>{ toggleExclude(); view.focus(); };
$("#prev").onclick=()=>loadFrame(cur-1);
$("#next").onclick=()=>loadFrame(cur+1);
$("#fit").onclick=fitView;
$("#one").onclick=()=>zoomAt(view.width/2,view.height/2,1/scale);
$("#zoomin").onclick=()=>zoomAt(view.width/2,view.height/2,1.2);
$("#zoomout").onclick=()=>zoomAt(view.width/2,view.height/2,1/1.2);
$("#carry").onclick=()=>{ carry=!carry; $("#carry").classList.toggle("on",carry); };
// blur after committing so global keyboard shortcuts resume (inputs swallow them)
$("#frameno").onchange=e=>{ loadFrame(parseInt(e.target.value)||0); e.target.blur(); };
$("#scrub").oninput=e=>loadFrame(parseInt(e.target.value));
$("#scrub").onchange=e=>e.target.blur();
$("#timeline").onclick=e=>{ const r=e.currentTarget.getBoundingClientRect();
  loadFrame(Math.round((e.clientX-r.left)/r.width*(INFO.n-1))); };

// panel controls
function blurOnEnter(el){ el.addEventListener("keydown", e=>{ if(e.key==="Enter"||e.key==="Escape"){ e.target.blur(); view.focus(); } }); }
$("#bw").onchange=e=>{ const w=+e.target.value; if(w>0) setNextSize(w, square ? w : nextH); else renderSizes(); };
$("#bh").onchange=e=>{ const h=+e.target.value; if(h>0) setNextSize(square ? h : nextW, h); else renderSizes(); };
$("#sqlock").onclick=()=>{ square=!square; if(square) setNextSize(nextW, nextW); else { savePrefs(); renderSizes(); } view.focus(); };
$("#sellabel").onchange=e=>{ relabelSelected(e.target.value); view.focus(); };
for(const id of ["#scx","#scy","#sw","#sh"]) $(id).onchange=editSelected;
for(const id of ["#bw","#bh","#scx","#scy","#sw","#sh","#newcls"]) blurOnEnter($(id));
$("#selsize").onclick=()=>{ resizeSelected(nextW, nextH); view.focus(); };
$("#seldel").onclick=()=>{ deleteSelected(); view.focus(); };
function addClass(){
  const v=$("#newcls").value.trim();
  if(v && !CLASSES.includes(v)){
    CLASSES.push(v); activeClass=CLASSES.length-1; $("#newcls").value="";
    post("/api/classes", CLASSES)
      .then(r=>{ if(!r.ok) throw new Error(); }).catch(()=>{ if(!STALE) alert("could not save the class list"); });
    savePrefs(); syncUI();
  }
  view.focus();
}
$("#addcls").onclick=addClass;
$("#newcls").addEventListener("keydown", e=>{ if(e.key==="Enter"){ e.preventDefault(); addClass(); } });

async function drain(){                  // push every pending edit before an export reads the store
  await flush();
  for(let i=0; (pending.size||flushing) && i<100; i++) await new Promise(r=>setTimeout(r,50));
}
$("#exportgt").onclick=async ()=>{
  const path = prompt("export GT to:", DEFAULTS.gt_out);
  if(!path) return;
  await drain();
  const r = await post("/api/export_gt", {path});
  const j = await r.json();
  alert(j.ok ? `exported to ${j.path}` : "export failed");
  view.focus();
};
$("#exportyolo").onclick=async ()=>{
  const dir = prompt("export YOLO labels to folder:", DEFAULTS.yolo_out);
  if(!dir) return;
  const images = confirm("Also write every frame as a lossless PNG into images/?\n\n"
    + "OK = labels + images (a few GB for a 1080p clip; takes a minute or two)\nCancel = labels only");
  await drain();
  setSaved("exporting…","dirty");
  try{
    const r = await post("/api/export_yolo", {dir, images});
    const j = await r.json();
    if(!j.ok) throw new Error(j.error || "export failed");
    const sk = Object.entries(j.skipped_class||{}).map(([c,n])=>`${c} ×${n}`).join(", ");
    alert(`YOLO export → ${j.dir}\n\n`
      + `${j.frames} label files (${j.empty_frames} empty = no object), ${j.boxes} boxes, ${j.images} images\n`
      + `class ids: ${j.classes.map((c,i)=>i+" = "+c).join(",  ")}`
      + (j.excluded ? `\n${j.excluded} can't-judge frames left out` : "")
      + (sk ? `\nskipped, class not in the list: ${sk}` : "")
      + (j.outside ? `\n${j.outside} boxes outside the image skipped` : "")
      + (j.warning ? `\n\n⚠ ${j.warning}` : ""));
  } catch(err){ alert("YOLO export failed: " + err.message); }
  saveNow();
  view.focus();
};

// ---- persistence ----------------------------------------------------------
// Working edits live in `boxes`; committing writes them into LABELS[cur] and
// marks that frame `pending`. saveNow() drains `pending` one frame at a time,
// retrying on failure -- so a lost/failed PUT (or a frame we navigated away
// from) is never dropped. Each pending frame is keyed by its own index, so a
// retry always re-sends the RIGHT frame's boxes, not the current view's.
let dirty=false, saveTimer=null, flushing=false;
const pending=new Set();
function setSaved(txt, cls){ const el=$("#saved"); el.textContent=txt; el.className="pill "+cls; }
function cleanBoxes(){
  return boxes.map(b=>{ const n=clip(b);
    return {x1:+n.x1.toFixed(2),y1:+n.y1.toFixed(2),x2:+n.x2.toFixed(2),y2:+n.y2.toFixed(2),label:n.label}; })
    .filter(b => (b.x2-b.x1)>0 && (b.y2-b.y1)>0 && ((b.x2-b.x1)>=0.5 || (b.y2-b.y1)>=0.5));
}
function commitLocal(){
  const clean=cleanBoxes();
  if(clean.length) LABELS[cur]=clean; else delete LABELS[cur];
  pending.add(cur); dirty=false;
}
function markDirty(){
  dirty=true; setSaved("unsaved","dirty");
  clearTimeout(saveTimer); saveTimer=setTimeout(flush, 500);
}
function flush(){                        // commit current frame, then push all pending
  clearTimeout(saveTimer);
  if(dirty) commitLocal();
  return saveNow();
}
function saveNow(){
  if(flushing || STALE) return Promise.resolve();
  if(pending.size===0){ if(!dirty) setSaved("saved","ok"); return Promise.resolve(); }
  flushing=true; setSaved("saving…","dirty");
  const f=pending.values().next().value;
  const payload=LABELS[f]||[];
  return fetch(api(`/api/labels/${f}`), {method:"PUT", body:JSON.stringify(payload)})
    .then(r=>{ if(r.status===409) staleTab(); if(!r.ok) throw new Error("http "+r.status); pending.delete(f); })
    .then(()=>{ flushing=false; renderTimeline();
                if(pending.size||dirty) return saveNow(); setSaved("saved","ok"); })
    .catch(()=>{ flushing=false; if(STALE) return; setSaved("save error — retrying","dirty");
                 clearTimeout(saveTimer); saveTimer=setTimeout(saveNow, 1500); });
}
window.addEventListener("beforeunload", ()=>{
  if(STALE) return;
  if(dirty) commitLocal();
  // sendBeacon can only POST; the backend accepts POST on /api/labels/ too.
  for(const f of pending) navigator.sendBeacon(api(`/api/labels/${f}`), JSON.stringify(LABELS[f]||[]));
});

// ---- UI sync ----
function renderClasses(){
  const list=$("#classlist"); list.textContent="";
  CLASSES.forEach((c,i)=>{
    const row=document.createElement("div");
    row.className="cls"+(i===activeClass?" active":"");
    row.title=`the next click places a "${c}" box`;
    const sw=document.createElement("span"); sw.className="swatch"; sw.style.background=PALETTE[i%PALETTE.length];
    const nm=document.createElement("span"); nm.textContent=c;
    const id=document.createElement("span"); id.className="id"; id.textContent=`id ${i}`;
    row.append(sw, nm, id);
    if(i<9){ const kb=document.createElement("kbd"); kb.textContent=String(i+1); row.append(kb); }
    row.onclick=()=>{ setActiveClass(i); view.focus(); };
    list.append(row);
  });
  $("#nextlbl").textContent = CLASSES[activeClass] || "";
}
function renderSizes(){
  const chips=$("#sizes"); chips.textContent="";
  for(const p of SIZE_PRESETS){
    const b=document.createElement("button"); b.textContent=p;
    if(nextW===p && nextH===p) b.className="on";
    b.onclick=()=>{ setNextSize(p, square ? p : p*nextH/nextW); view.focus(); };
    chips.append(b);
  }
  if(document.activeElement!==$("#bw")) $("#bw").value=fmtPx(nextW);
  if(document.activeElement!==$("#bh")) $("#bh").value=fmtPx(nextH);
  $("#sqlock").classList.toggle("on", square);
  $("#sizelbl").textContent=`${fmtPx(nextW)} × ${fmtPx(nextH)} px`;
}
function renderSelected(){
  const has = selected>=0 && selected<boxes.length;
  $("#selnone").hidden=has; $("#selbody").hidden=!has;
  if(!has) return;
  const b=clip(boxes[selected]), cls=baseLabel(b.label);
  const sel=$("#sellabel");
  if(document.activeElement!==sel){
    sel.textContent="";
    const names = CLASSES.includes(cls) ? CLASSES : CLASSES.concat([b.label]);
    for(const c of names){
      const o=document.createElement("option"); o.value=c; const i=CLASSES.indexOf(c);
      o.textContent = i>=0 ? `${c}  (id ${i})` : `${c}  (not in the list)`; sel.append(o);
    }
    sel.value = CLASSES.includes(cls) ? cls : b.label;
  }
  const put=(id,v)=>{ const el=$(id); if(document.activeElement!==el) el.value=fmtPx(v); };
  put("#scx",(b.x1+b.x2)/2); put("#scy",(b.y1+b.y2)/2); put("#sw",b.x2-b.x1); put("#sh",b.y2-b.y1);
  $("#yololine").textContent=yoloLine(b);
}
function renderTimeline(){
  const tl=$("#timeline");
  tl.querySelectorAll(".tick").forEach(t=>t.remove());
  const W=tl.clientWidth||1;
  const step=Math.max(1, Math.floor(INFO.n/Math.min(INFO.n, W)));
  for(let i=0;i<INFO.n;i+=step){
    let any=false, ex=false;
    for(let j=i;j<Math.min(INFO.n,i+step);j++){
      if(EXCLUDED.has(j)) ex=true; else if(LABELS[j]&&LABELS[j].length) any=true;
    }
    if(any||ex){ const t=document.createElement("div"); t.className="tick"+(ex?" ex":"");
      t.style.left=(i/(INFO.n-1)*100)+"%"; tl.appendChild(t); }
  }
}
function syncUI(){
  $("#frameno").value=cur; $("#framecnt").textContent=`/ ${INFO.n-1}`;
  $("#scrub").value=cur; $("#cursorline").style.left=(cur/(INFO.n-1)*100)+"%";
  $("#zoomlbl").textContent=Math.round(scale*100)+"%";
  $("#boxinfo").textContent=`${boxes.length} box${boxes.length===1?"":"es"}`+(selected>=0?" · 1 sel":"");
  const ex=EXCLUDED.has(cur);
  $("#exclude").classList.toggle("on", ex); $("#exclbanner").hidden=!ex;
  renderClasses(); renderSizes(); renderSelected();
}

// ---- boot ----
async function boot(){
  const r = await fetch("/api/info"); const j = await r.json();
  SESSION = j.session; INFO = j.info; CLASSES = j.classes.slice(); LABELS = {};
  for(const [k,v] of Object.entries(j.labels)) LABELS[parseInt(k)] = v;
  EXCLUDED = new Set((j.exclude||[]).map(Number));
  DEFAULTS = {...DEFAULTS, ...(j.defaults||{})};
  loadPrefs();
  $("#frameno").max=INFO.n-1; $("#scrub").max=INFO.n-1;
  $("#framecnt").textContent=`/ ${INFO.n-1}`;
  document.title = `label · ${INFO.video}`;
  resize(); fitView();
  loadFrame(0, {commit:false});
  renderTimeline();
  window.addEventListener("resize", ()=>{ resize(); renderTimeline(); });
}
boot();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run(video: str, labels_path: str, from_gt: str | None,
        classes: list[str] | None, host: str, port: int, no_open: bool,
        gt_out: str | None = None, yolo_classes: list[str] | None = None,
        yolo_prefix: str | None = None) -> None:
    video = str(_resolve(video))
    from_gt = str(_resolve(from_gt)) if from_gt else None
    if not Path(video).exists():
        raise SystemExit(f"video not found: {video}\n"
                         f"(pass --video with a path relative to {ROOT} or an absolute path)")
    jpegs = load_frames(video)
    info = probe(video)

    lpath = _resolve(labels_path)
    if lpath.exists():
        labels = json.loads(lpath.read_text(encoding="utf-8"))
        stored = Path(str(labels.get("video", ""))).name
        if stored and stored != Path(video).name:     # an existing store is never reused for another video
            raise SystemExit(f"{lpath} labels {stored!r}, not {Path(video).name!r}: pass --labels for this video")
        labels.setdefault("frames", {})
        labels.setdefault("classes", classes or DEFAULT_CLASSES)
        print(f"loaded {len(labels['frames'])} labeled frames from {lpath}")
    else:
        seed_frames: dict[str, list[dict]] = {}
        seed_classes: list[str] = []
        seed_exclude: list[int] = []
        if from_gt:
            seed_frames, seed_classes, seed_exclude = gt_to_labels(from_gt)
            print(f"seeded {len(seed_frames)} frames from {from_gt}")
        cls = list(classes or seed_classes or DEFAULT_CLASSES)
        cls += [c for c in seed_classes if c not in cls]   # every seeded box's class is listed
        labels = {
            "video": Path(video).name,
            "width": info.width,
            "height": info.height,
            "classes": cls,
            "exclude_frames": seed_exclude,
            "frames": seed_frames,
        }
    labels.setdefault("width", info.width)
    labels.setdefault("height", info.height)
    labels["n_frames"] = len(jpegs)   # the sequential decode count, not the container's claim

    gt_def, yolo_def, prefix_def = default_exports(lpath)
    STATE["jpegs"] = jpegs
    STATE["info"] = {"video": Path(video).name, "n": len(jpegs),
                     "width": info.width, "height": info.height}
    STATE["labels"] = labels
    STATE["labels_path"] = lpath
    STATE["video_path"] = video
    STATE["session"] = secrets.token_hex(8)
    STATE["defaults"] = {"gt_out": _resolve(gt_out) if gt_out else gt_def, "yolo_out": yolo_def,
                         "prefix": yolo_prefix or prefix_def, "yolo_classes": yolo_classes}
    with STATE["lock"]:
        save_labels()

    # bind, retrying a few ports if busy
    server = None
    for p in range(port, port + 20):
        try:
            server = ThreadingHTTPServer((host, p), Handler)
            port = p
            break
        except OSError:
            continue
    if server is None:
        raise SystemExit(f"no free port in {port}..{port+20}")

    url = f"http://{host}:{port}/"
    print(f"\n  labeler ready -> {url}")
    print(f"  labels: {lpath}   ({len(jpegs)} frames)")
    print(f"  exports: GT -> {STATE['defaults']['gt_out']}   YOLO -> {yolo_def}\n  Ctrl-C to stop\n", flush=True)
    if not no_open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping.")
        server.shutdown()


def _csv(s: str | None) -> list[str] | None:
    return [c.strip() for c in s.split(",") if c.strip()] if s else None


def main() -> None:
    ap = argparse.ArgumentParser(description="frame-by-frame bbox labeler")
    ap.add_argument("--video", default="data/videos/07_05.mp4")
    ap.add_argument("--labels", default="work/labels.json",
                    help="per-frame label store (read + written)")
    ap.add_argument("--from-gt", help="seed an empty label store from a GT json")
    ap.add_argument("--classes", help="comma-separated class names (their order sets the YOLO ids)")
    ap.add_argument("--export-gt", metavar="PATH",
                    help="convert the label store to GT format and exit")
    ap.add_argument("--gt-out", metavar="PATH",
                    help="where the UI's GT export writes by default (default: <store>_gt.json "
                         "beside the store; work/gt_labeled.json for the default store)")
    ap.add_argument("--export-yolo", metavar="DIR",
                    help="write the label store as a YOLO dataset (labels/*.txt + classes.txt) and exit")
    ap.add_argument("--yolo-classes",
                    help="comma-separated classes whose order sets the YOLO ids (default: the store's "
                         "list); boxes of any other class are skipped. Use one list for every clip")
    ap.add_argument("--yolo-prefix", help="YOLO file-name prefix (default: the store's file name)")
    ap.add_argument("--yolo-images", action="store_true",
                    help="with --export-yolo: also write each frame as a lossless PNG (decodes --video)")
    ap.add_argument("--yolo-labeled-only", action="store_true",
                    help="with --export-yolo: only frames with boxes (default: every frame not marked "
                         "can't-judge; a frame without boxes becomes an empty file = no object)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-open", action="store_true", help="don't open a browser")
    a = ap.parse_args()

    if a.export_gt:
        labels = json.loads(_resolve(a.labels).read_text(encoding="utf-8"))
        labels_to_gt(labels, _resolve(a.export_gt))
        return

    if a.export_yolo:
        lpath = _resolve(a.labels)
        labels = json.loads(lpath.read_text(encoding="utf-8"))
        video = None
        if a.yolo_images:
            video = str(_resolve(a.video))
            if Path(video).name != Path(str(labels.get("video", ""))).name:
                raise SystemExit(f"--video is {Path(video).name!r}, but {lpath} labels "
                                 f"{labels.get('video')!r}: pass the store's own video")
        rep = export_yolo(labels, _resolve(a.export_yolo), _csv(a.yolo_classes),
                          a.yolo_prefix or lpath.stem, a.yolo_labeled_only, video)
        print(json.dumps(rep, indent=1))
        return

    run(a.video, a.labels, a.from_gt, _csv(a.classes), a.host, a.port, a.no_open,
        gt_out=a.gt_out, yolo_classes=_csv(a.yolo_classes), yolo_prefix=a.yolo_prefix)


if __name__ == "__main__":
    main()
