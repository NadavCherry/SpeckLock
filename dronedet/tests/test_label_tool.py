"""tools/label.py's YOLO export and GT round trip.

The labeler stores corner boxes in pixels; YOLO wants ``id cx cy w h`` normalised to
[0, 1], inside the image, with ids that are positions in one class list. Pinned here:

1. ``yolo_line`` normalises by the image size, and clips a box over the edge to its inside
   part (ultralytics drops an image whose label leaves [0, 1]).
2. ``export_yolo`` writes one file per frame (empty = no object), none for a can't-judge
   frame, skips and counts a class that is not in the list rather than renumbering it, and
   reads back through the repo's own YOLO reader to the same pixels.
3. Track / slot numbers (``bird_3``, ``bird#2``) are not classes.
4. GT export carries the store's can't-judge frames and size into ``meta``.

Nothing here decodes a real video or binds a port: frames are stubbed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.adapters.base import parse_yolo_label, read_class_names  # noqa: E402
from tools import label as L  # noqa: E402

W, H = 1920, 1080


def _store(**kw):
    s = {"video": "clip.mp4", "width": W, "height": H, "n_frames": 5,
         "classes": ["drone", "bird"], "exclude_frames": [], "frames": {}}
    s.update(kw)
    return s


def test_yolo_line_normalises_centre_and_size():
    line = L.yolo_line({"x1": 100, "y1": 50, "x2": 110, "y2": 58, "label": "drone"}, 0, W, H)
    cid, cx, cy, w, h = line.split()
    assert cid == "0"
    assert float(cx) == pytest.approx(105 / W, abs=1e-6)
    assert float(cy) == pytest.approx(54 / H, abs=1e-6)
    assert float(w) == pytest.approx(10 / W, abs=1e-6)
    assert float(h) == pytest.approx(8 / H, abs=1e-6)
    assert all(len(v.split(".")[1]) == 6 for v in (cx, cy, w, h))


def test_yolo_line_accepts_reversed_corners():
    a = L.yolo_line({"x1": 110, "y1": 58, "x2": 100, "y2": 50}, 1, W, H)
    b = L.yolo_line({"x1": 100, "y1": 50, "x2": 110, "y2": 58}, 1, W, H)
    assert a == b


def test_a_box_over_the_edge_keeps_only_its_inside_part():
    # an 8 px box centred 2 px from the left edge: 6 px of it are inside
    line = L.yolo_line({"x1": -2, "y1": 100, "x2": 6, "y2": 108}, 0, W, H)
    _, cx, _, w, _ = (float(v) for v in line.split())
    assert w == pytest.approx(6 / W, abs=1e-6) and cx == pytest.approx(3 / W, abs=1e-6)
    far = L.yolo_line({"x1": W - 4, "y1": H - 4, "x2": W + 4, "y2": H + 4}, 0, W, H)
    assert all(0.0 <= float(v) <= 1.0 for v in far.split()[1:])


def test_a_box_wholly_outside_the_image_has_no_line():
    assert L.yolo_line({"x1": -20, "y1": 5, "x2": -10, "y2": 15}, 0, W, H) is None
    assert L.yolo_line({"x1": 5, "y1": 5, "x2": 5, "y2": 15}, 0, W, H) is None   # zero width


@pytest.mark.parametrize("name,cls", [("bird#2", "bird"), ("bird_3", "bird"), ("drone", "drone"),
                                      ("unsure_drone", "unsure_drone"), ("far", "far"),
                                      ("other_12", "other")])
def test_track_and_slot_numbers_are_not_classes(name, cls):
    assert L.base_label(name) == cls


def test_export_writes_one_file_per_frame_and_reads_back_to_the_same_pixels(tmp_path):
    drone = {"x1": 100.0, "y1": 50.0, "x2": 110.0, "y2": 58.0, "label": "drone"}
    bird = {"x1": 300.0, "y1": 200.0, "x2": 316.0, "y2": 212.0, "label": "bird_3"}
    rep = L.export_yolo(_store(frames={"1": [drone, bird], "3": [drone]}), tmp_path, prefix="c1")
    files = sorted(p.name for p in (tmp_path / "labels").iterdir())
    assert files == [f"c1_{f:06d}.txt" for f in range(5)]
    assert rep["frames"] == 5 and rep["boxes"] == 3 and rep["empty_frames"] == 3
    names = read_class_names(tmp_path / "classes.txt")
    assert names == {0: "drone", 1: "bird"}
    text = (tmp_path / "labels" / "c1_000001.txt").read_text(encoding="utf-8")
    assert "\r" not in text
    got = parse_yolo_label(text, W, H, names)
    assert [b.cls for b in got] == ["drone", "bird"]
    assert (got[1].x1, got[1].y1, got[1].x2, got[1].y2) == pytest.approx((300, 200, 316, 212), abs=1e-3)
    assert (tmp_path / "labels" / "c1_000000.txt").read_text(encoding="utf-8") == ""


def test_a_cant_judge_frame_gets_no_file_at_all(tmp_path):
    rep = L.export_yolo(_store(exclude_frames=[2]), tmp_path, prefix="c")
    assert not (tmp_path / "labels" / "c_000002.txt").exists()
    assert rep["frames"] == 4 and rep["excluded"] == 1


def test_a_class_outside_the_list_is_skipped_and_counted_never_renumbered(tmp_path):
    store = _store(frames={"0": [{"x1": 1, "y1": 1, "x2": 9, "y2": 9, "label": "unsure"},
                                 {"x1": 20, "y1": 20, "x2": 28, "y2": 28, "label": "bird"}]})
    rep = L.export_yolo(store, tmp_path, classes=["drone", "bird"], prefix="c")
    assert rep["skipped_class"] == {"unsure": 1}
    lines = (tmp_path / "labels" / "c_000000.txt").read_text(encoding="utf-8").split("\n")
    assert [ln.split()[0] for ln in lines if ln] == ["1"]


def test_the_class_list_sets_the_ids(tmp_path):
    store = _store(frames={"0": [{"x1": 1, "y1": 1, "x2": 9, "y2": 9, "label": "drone"}]})
    L.export_yolo(store, tmp_path, classes=["bird", "drone"], prefix="c")
    assert (tmp_path / "labels" / "c_000000.txt").read_text(encoding="utf-8").startswith("1 ")
    with pytest.raises(ValueError):
        L.export_yolo(store, tmp_path, classes=["drone", "drone"], prefix="c")


def test_labeled_only_and_stale_files_from_an_earlier_export(tmp_path):
    box = {"x1": 1, "y1": 1, "x2": 9, "y2": 9, "label": "drone"}
    L.export_yolo(_store(frames={"0": [box], "4": [box]}), tmp_path, prefix="c")
    (tmp_path / "labels" / "other_000001.txt").write_text("", encoding="utf-8")   # not ours
    rep = L.export_yolo(_store(frames={"4": [box]}), tmp_path, prefix="c", labeled_only=True)
    names = sorted(p.name for p in (tmp_path / "labels").iterdir())
    assert names == ["c_000004.txt", "other_000001.txt"] and rep["stale_removed"] == 4


def test_images_are_written_losslessly_with_the_label_stems(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    imgs = [rng.integers(0, 255, (H // 60, W // 60, 3), dtype=np.uint8) for _ in range(3)]
    monkeypatch.setattr(L, "frames", lambda path: iter(enumerate(imgs)))
    store = _store(width=W // 60, height=H // 60, n_frames=3, exclude_frames=[1])
    rep = L.export_yolo(store, tmp_path, prefix="c", video="clip.mp4")
    assert rep["images"] == 2
    import cv2
    back = cv2.imread(str(tmp_path / "images" / "c_000002.png"))
    assert np.array_equal(back, imgs[2])
    assert not (tmp_path / "images" / "c_000001.png").exists()


def test_images_of_another_size_are_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "frames", lambda path: iter([(0, np.zeros((10, 10, 3), np.uint8))]))
    with pytest.raises(ValueError, match="10x10"):
        L.export_yolo(_store(n_frames=1), tmp_path, prefix="c", video="clip.mp4")


def test_gt_export_carries_cant_judge_frames_and_size_and_reads_back(tmp_path):
    box = {"x1": 100, "y1": 50, "x2": 110, "y2": 58, "label": "bird"}
    store = _store(exclude_frames=[3], frames={"1": [box, dict(box, x1=200, x2=210)]})
    gt = tmp_path / "gt.json"
    L.labels_to_gt(store, gt)
    raw = json.loads(gt.read_text(encoding="utf-8"))
    assert raw["meta"] == {"width": W, "height": H, "n_frames": 5, "exclude_frames": [3]}
    assert sorted(raw["objects"]) == ["bird", "bird#2"]
    frames_, classes, exclude = L.gt_to_labels(gt)
    assert classes == ["bird"] and exclude == [3]
    assert [b["label"] for b in frames_["1"]] == ["bird", "bird"]


def test_gt_export_keeps_existing_meta_and_ignore_flags_by_class(tmp_path):
    gt = tmp_path / "gt.json"
    gt.write_text(json.dumps({"video": "clip.mp4", "meta": {"shifts": [[0, 0]], "height": 720},
                              "objects": {"bird_1": {"ignore": True, "frames": {}}}}), encoding="utf-8")
    box = {"x1": 1, "y1": 1, "x2": 9, "y2": 9, "label": "bird"}
    L.labels_to_gt(_store(frames={"0": [box]}), gt)
    raw = json.loads(gt.read_text(encoding="utf-8"))
    assert raw["meta"]["shifts"] == [[0, 0]] and raw["meta"]["height"] == 720
    assert raw["objects"]["bird"]["ignore"] is True


def _request(method, path, body=b""):
    """Drive the handler without a socket: raw request in, (status, JSON reply) out."""
    import io
    h = L.Handler.__new__(L.Handler)
    h.path, h.command, h.request_version = path, method, "HTTP/1.1"
    h.requestline, h.client_address, h.close_connection = f"{method} {path}", ("127.0.0.1", 0), False
    h.headers = {"Content-Length": str(len(body))}
    h.rfile, h.wfile = io.BytesIO(body), io.BytesIO()
    getattr(h, f"do_{method}")()
    head, _, payload = h.wfile.getvalue().partition(b"\r\n\r\n")
    return int(head.split()[1]), payload


def test_a_tab_from_an_earlier_run_can_neither_see_frames_nor_save(tmp_path, monkeypatch):
    # frame URLs used to be the same for every video, so the browser showed a cached frame of
    # the previous clip, and a tab left open could write its boxes into the next clip's store
    store = tmp_path / "s01.json"
    monkeypatch.setitem(L.STATE, "session", "run2")
    monkeypatch.setitem(L.STATE, "labels", _store())
    monkeypatch.setitem(L.STATE, "labels_path", store)
    monkeypatch.setitem(L.STATE, "jpegs", [b"jpeg0"])
    monkeypatch.setitem(L.STATE, "defaults", {"gt_out": tmp_path / "gt.json", "yolo_out": tmp_path / "y",
                                              "prefix": "s01", "yolo_classes": None})
    box = json.dumps([{"x1": 1, "y1": 1, "x2": 9, "y2": 9, "label": "drone"}]).encode()

    assert _request("GET", "/api/frame/0.jpg?s=run1")[0] == 409
    assert _request("GET", "/api/frame/0.jpg")[0] == 409
    assert _request("GET", "/api/frame/0.jpg?s=run2") == (200, b"jpeg0")
    for method in ("PUT", "POST"):
        assert _request(method, "/api/labels/0?s=run1", box)[0] == 409
    assert not store.exists() and L.STATE["labels"]["frames"] == {}
    assert _request("POST", "/api/exclude?s=run1", b'{"frame": 2, "on": true}')[0] == 409
    assert _request("PUT", "/api/labels/0?s=run2", box)[0] == 200
    assert json.loads(store.read_text(encoding="utf-8"))["frames"]["0"][0]["label"] == "drone"
    info = json.loads(_request("GET", "/api/info")[1])
    assert info["session"] == "run2"


def test_default_exports_sit_beside_a_store(tmp_path):
    gt, yolo, prefix = L.default_exports(tmp_path / "s01.json")
    assert gt == tmp_path / "s01_gt.json" and yolo == tmp_path / "s01_yolo" and prefix == "s01"
    gt, _, _ = L.default_exports(L.ROOT / "work" / "labels.json")
    assert gt == L.ROOT / "work" / "gt_labeled.json"
