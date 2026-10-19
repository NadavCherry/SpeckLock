"""`allow_motion` is opt-in: a 'motion' label is not drone evidence unless the caller asks.

In PC-MAX's detections a bare 'motion' label is a proposal the verifier looked at and did
not confirm (`dronedet/methods/hybrid2.py`). When it counted by default, final/run_final.py
raised 4-15 clutter tracks per video and gained the drone nothing
(docs/reports/allow-motion.md). These tests pin the default where callers reach it: the
library function, its file wrapper, and `tools/tracks_to_dets.py --classify`, which
run_final calls as its own step.
"""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

from dronedet import trackclass as TC

REPO = Path(__file__).resolve().parents[2]
N = 20   # tracked frames: past N_CONF, well short of LONG_TRACK, so only evidence decides


def _payloads(label: str, score: float = 0.6) -> tuple[dict, dict]:
    """One track sitting exactly on one detection per frame, every detection labelled `label`."""
    assert TC.N_CONF < N < TC.LONG_TRACK
    frames = {str(f): [100.0 + f, 50.0, 6.0, 6.0, "tracked"] for f in range(N)}
    tracks = {"video": "x.mp4", "tracks": [{"id": 1, "score": 0.6, "n": N, "frames": frames}]}
    dets = {"video": "x.mp4", "method": "fixture", "meta": {"n_frames": N},
            "frames": {str(f): [[97.0 + f, 47.0, 103.0 + f, 53.0, score, label]]
                       for f in range(N)}}
    return tracks, dets


def test_motion_label_is_not_evidence_by_default():
    tracks, dets = _payloads("motion")
    assert TC.classify_tracks(tracks, dets)[1]["cls"] == "other"
    assert TC.classify_tracks(tracks, dets, allow_motion=True)[1]["cls"] == "drone"


def test_appearance_evidence_does_not_depend_on_the_flag():
    for label in ("drone", "drone+motion"):
        tracks, dets = _payloads(label)
        assert TC.classify_tracks(tracks, dets)[1]["cls"] == "drone", label
        assert TC.classify_tracks(tracks, dets, allow_motion=True)[1]["cls"] == "drone", label


def test_classify_files_defaults_off(tmp_path):
    tracks, dets = _payloads("motion")
    (tmp_path / "t.json").write_text(json.dumps(tracks), encoding="utf-8")
    (tmp_path / "d.json").write_text(json.dumps(dets), encoding="utf-8")
    assert TC.classify_files(tmp_path / "t.json", tmp_path / "d.json")[1]["cls"] == "other"
    assert TC.classify_files(tmp_path / "t.json", tmp_path / "d.json",
                             allow_motion=True)[1]["cls"] == "drone"


def _tracked_boxes(tmp_path, monkeypatch, *extra: str) -> int:
    tracks, dets = _payloads("motion")
    (tmp_path / "t.json").write_text(json.dumps(tracks), encoding="utf-8")
    (tmp_path / "d.json").write_text(json.dumps(dets), encoding="utf-8")
    out = tmp_path / f"out{len(extra)}.json"
    monkeypatch.setattr(sys, "argv", ["tracks_to_dets.py", "--tracks", str(tmp_path / "t.json"),
                                      "--src", str(tmp_path / "d.json"), "--out", str(out),
                                      "--classify", *extra])
    runpy.run_path(str(REPO / "tools/tracks_to_dets.py"), run_name="__main__")
    return sum(len(v) for v in json.loads(out.read_text(encoding="utf-8"))["frames"].values())


def test_tracks_to_dets_classify_needs_the_opt_in(tmp_path, monkeypatch):
    assert _tracked_boxes(tmp_path, monkeypatch) == 0
    assert _tracked_boxes(tmp_path, monkeypatch, "--allow-motion") == N
