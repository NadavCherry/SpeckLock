"""tools/render_track_video.py: linking per-frame drone points into trajectories."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.render_track_video import link_points  # noqa: E402


def test_two_crossing_paths_keep_their_ids():
    # A moves right along y=100, B moves left along y=300: nearest-neighbour keeps them apart
    per = {f: [(100.0 + 10 * f, 100.0), (500.0 - 10 * f, 300.0)] for f in range(20)}
    out = link_points(per)
    ids_a = {tid for f in out for tid, x, y in out[f] if y == 100.0}
    ids_b = {tid for f in out for tid, x, y in out[f] if y == 300.0}
    assert len(ids_a) == 1 and len(ids_b) == 1 and ids_a != ids_b


def test_a_short_gap_continues_the_track_and_a_long_one_starts_a_new_one():
    per = {f: [(100.0 + 5 * f, 100.0)] for f in list(range(10)) + list(range(14, 20))}
    per.update({f: [(100.0 + 5 * f, 100.0)] for f in range(40, 45)})
    out = link_points(per, max_gap=10)
    assert len({tid for f in range(20) for tid, _, _ in out.get(f, [])}) == 1
    assert out[40][0][0] != out[0][0][0]


def test_a_jump_beyond_the_gate_is_a_new_track():
    per = {0: [(100.0, 100.0)], 1: [(400.0, 100.0)]}
    out = link_points(per, gate=80.0)
    assert out[0][0][0] != out[1][0][0]


def test_dim_below_marks_low_confidence_drones_only(tmp_path):
    from dronedet.detections import Detection, DetectionSet
    from tools.render_track_video import load_system

    ds = DetectionSet(video="v.mp4", method="m")
    ds.add(0, [Detection(0, 0, 10, 10, 0.3, "drone"), Detection(50, 50, 60, 60, 0.9, "drone"),
               Detection(90, 90, 96, 96, 0.1, "bird")])
    p = tmp_path / "points.json"
    ds.save(p)
    per = load_system(str(p), None, dim_below=0.5)
    by_conf = {round(o["conf"], 1): o["observed"] for o in per[0]}
    assert by_conf == {0.3: False, 0.9: True, 0.1: True}
    assert all(o["observed"] for o in load_system(str(p), None)[0])
