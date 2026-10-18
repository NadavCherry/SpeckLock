"""The class-aware center scorecard (dronedet.centerscore)."""

from __future__ import annotations

import math

import pytest

from dronedet.centerscore import pool, score_clip
from dronedet.detections import Detection, DetectionSet
from dronedet.gt import GroundTruth, GTObject


def _pt(x, y, label, s=0.9, r=3.0):
    return Detection(x - r, y - r, x + r, y + r, s, label)


def _gt(n=10, drone=(100.0, 100.0), bird=None, exclude=()):
    gt = GroundTruth(video="synthetic.mp4", meta={"exclude_frames": list(exclude)})
    gt.objects["drone"] = GTObject("drone", frames={f: (drone[0] + f, drone[1], 8, 8) for f in range(n)})
    if bird is not None:
        gt.objects["bird_1"] = GTObject("bird_1", ignore=True,
                                        frames={f: (bird[0], bird[1] + f, 6, 6) for f in range(n)})
    return gt


def _preds(fn, n=10):
    ds = DetectionSet(video="synthetic.mp4", method="t")
    for f in range(n):
        ds.frames[f] = fn(f)
    return ds


def test_perfect_centering_scores_one():
    s = score_clip(_gt(), _preds(lambda f: [_pt(100 + f, 100, "drone")]), 10, 1080).summary()
    assert s["center_score"] == pytest.approx(1.0)
    assert s["hit_rate@tau"] == 1.0 and s["miss_rate"] == 0.0
    assert s["lock_latency_frames"] == 0 and s["other_false_alarms"] == 0


def test_score_is_linear_in_error_and_zero_beyond_tau_max():
    half = score_clip(_gt(), _preds(lambda f: [_pt(110 + f, 100, "drone")]), 10, 1080).summary()
    assert half["center_score"] == pytest.approx(0.5)          # CLE 10 of tau_max 20
    far = score_clip(_gt(), _preds(lambda f: [_pt(130 + f, 100, "drone")]), 10, 1080).summary()
    assert far["center_score"] == 0.0 and far["hit_rate@tau"] == 0.0
    assert far["other_false_alarms"] == 10                      # 30 px off the drone: an alarm


def test_frames_without_any_output_count_as_misses():
    # outputs exist only on even frames: the file does not decide which frames are scored
    s = score_clip(_gt(), _preds(lambda f: [_pt(100 + f, 100, "drone")] if f % 2 == 0 else []),
                   10, 1080).summary()
    assert s["center_score"] == pytest.approx(0.5)
    assert s["hit_rate@tau"] == pytest.approx(0.5)


def test_a_bird_label_on_the_drone_is_a_miss_not_a_hit():
    s = score_clip(_gt(), _preds(lambda f: [_pt(100 + f, 100, "bird")]), 10, 1080).summary()
    assert s["center_score"] == 0.0 and s["bird_as_drone"] == 0


def test_drone_label_on_a_bird_is_counted_and_bird_label_on_a_bird_is_recall():
    gt = _gt(bird=(500.0, 300.0))
    bad = score_clip(gt, _preds(lambda f: [_pt(100 + f, 100, "drone"), _pt(500, 300 + f, "drone")]),
                     10, 1080).summary()
    assert bad["bird_as_drone"] == 10 and bad["bird_frames_with_alarm"] == 1.0
    assert bad["other_false_alarms"] == 0 and bad["center_score"] == pytest.approx(1.0)
    good = score_clip(gt, _preds(lambda f: [_pt(100 + f, 100, "drone"), _pt(500, 300 + f, "bird")]),
                      10, 1080).summary()
    assert good["bird_as_drone"] == 0 and good["bird_recall"] == 1.0


def test_distances_are_1080p_equivalent():
    # a 1440p clip: 13.33 native px == 10 1080p-px -> score 0.5
    gt = _gt()
    s = score_clip(gt, _preds(lambda f: [_pt(100 + f + 40 / 3, 100, "drone")]), 10, 1440).summary()
    assert s["center_score"] == pytest.approx(0.5, abs=1e-6)


def test_excluded_frames_are_not_scored():
    s = score_clip(_gt(exclude=[0, 1, 2, 3, 4]),
                   _preds(lambda f: [] if f < 5 else [_pt(100 + f, 100, "drone")]), 10, 1080).summary()
    assert s["frames"] == 5 and s["center_score"] == pytest.approx(1.0)


def test_lock_latency_and_losses():
    # lock from frame 3, lost on frames 5..9 (5 misses in a row = one loss)
    s = score_clip(_gt(n=12), _preds(lambda f: [_pt(100 + f, 100, "drone")] if 3 <= f < 5 or f >= 10 else [],
                                     n=12), 12, 1080).summary()
    assert s["lock_latency_frames"] == 3 and s["lock_losses"] == 1


def test_duplicates_on_the_drone_are_not_alarms():
    s = score_clip(_gt(), _preds(lambda f: [_pt(100 + f, 100, "drone"), _pt(103 + f, 100, "drone")]),
                   10, 1080).summary()
    assert s["other_false_alarms"] == 0 and s["duplicates"] == 10


def test_pool_is_frame_weighted():
    a = score_clip(_gt(), _preds(lambda f: [_pt(100 + f, 100, "drone")]), 10, 1080, name="a")
    b = score_clip(_gt(n=30), _preds(lambda f: [], n=30), 30, 1080, name="b")
    p = pool([a, b]).summary()
    assert p["drone_frames"] == 40 and p["center_score"] == pytest.approx(10 / 40)
    assert not math.isnan(p["center_score"])


def test_a_bird_label_on_the_drone_is_counted_as_drone_as_bird():
    s = score_clip(_gt(), _preds(lambda f: [_pt(100 + f, 100, "bird")] if f < 4 else
                                 [_pt(100 + f, 100, "drone")]), 10, 1080).summary()
    assert s["drone_as_bird_rate"] == pytest.approx(0.4) and s["hit_rate@tau"] == pytest.approx(0.6)


def test_precision_counts_every_drone_output_off_the_drone():
    gt = _gt(bird=(500.0, 300.0))
    # per frame: one on the drone, one duplicate on it, one on the bird, one on nothing
    s = score_clip(gt, _preds(lambda f: [_pt(100 + f, 100, "drone"), _pt(102 + f, 100, "drone"),
                                         _pt(500, 300 + f, "drone"), _pt(900, 900, "drone")]),
                   10, 1080).summary()
    assert s["precision"] == pytest.approx(0.25)
    assert score_clip(gt, _preds(lambda f: []), 10, 1080).summary()["precision"] is None


def test_box_quality_on_hits():
    # GT 8x8; output 16x16 on the same centre: IoU 0.25, size ratio 2
    s = score_clip(_gt(), _preds(lambda f: [_pt(100 + f, 100, "drone", r=8.0)]), 10, 1080).summary()
    assert s["mean_iou_hits"] == pytest.approx(0.25) and s["median_size_ratio_hits"] == pytest.approx(2.0)
    assert s["iou50_rate"] == 0.0
    exact = score_clip(_gt(), _preds(lambda f: [_pt(100 + f, 100, "drone", r=4.0)]), 10, 1080).summary()
    assert exact["mean_iou_hits"] == pytest.approx(1.0) and exact["iou50_rate"] == 1.0


def test_average_precision_ranks_outputs_by_score():
    from dronedet.centerscore import average_precision

    gt = _gt()
    # every frame: a true output at 0.9; frames 0-4 also carry a false one ranked above it
    preds = _preds(lambda f: [_pt(100 + f, 100, "drone", s=0.9, r=4.0)]
                   + ([_pt(600, 600, "drone", s=0.95, r=4.0)] if f < 5 else []))
    assert average_precision([(gt, preds, 10, 1080)], "iou") == pytest.approx(10 / 15, abs=1e-4)
    clean = _preds(lambda f: [_pt(100 + f, 100, "drone", s=0.9, r=4.0),
                              _pt(600, 600, "drone", s=0.1, r=4.0)])
    assert average_precision([(gt, clean, 10, 1080)], "center") == pytest.approx(1.0)
    assert average_precision([(GroundTruth(video="x", meta={}), clean, 10, 1080)]) is None


def test_outputs_on_an_unsure_object_are_neither_credited_nor_charged():
    gt = _gt()
    gt.objects["unsure_1"] = GTObject("unsure_1", ignore=True, frames={f: (700.0, 400.0, 6, 6) for f in range(10)})
    s = score_clip(gt, _preds(lambda f: [_pt(100 + f, 100, "drone"), _pt(700, 400, "drone")]), 10, 1080).summary()
    assert s["other_false_alarms"] == 0 and s["ignored_outputs"] == 10 and s["center_score"] == pytest.approx(1.0)
