"""Temporal-YOLOv8's centred window: t-dt, t, t+dt, oldest first, every frame exactly once.

The window is non-causal, so the generator runs dt frames behind its input and must flush the
tail. Getting either end wrong would silently train (or score) the prior-art arm on a different
representation from the one the paper describes, and the comparison would be against a straw man.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
M = pytest.importorskip("tools.make_dataset_external")


def _frames(n, h=8, w=8):
    """Frame i is uniformly 10*i, so each channel of a stack names its source frame."""
    return [np.full((h, w, 3), 10 * i, dtype=np.uint8) for i in range(n)]


def _ids(stack):
    return [int(round(float(stack[..., c].mean()) / 10)) for c in range(3)]


def test_every_frame_gets_exactly_one_stack_in_order():
    out = list(M.centred_stacks_from(_frames(20), dt=3))
    assert [t for t, _ in out] == list(range(20))


def test_the_window_is_t_minus_dt_t_t_plus_dt_oldest_first():
    out = dict(M.centred_stacks_from(_frames(20), dt=3))
    assert _ids(out[10]) == [7, 10, 13]


def test_both_ends_clamp_to_the_video():
    out = dict(M.centred_stacks_from(_frames(20), dt=3))
    assert _ids(out[0]) == [0, 0, 3]
    assert _ids(out[1]) == [0, 1, 4]
    assert _ids(out[18]) == [15, 18, 19]
    assert _ids(out[19]) == [16, 19, 19]


def test_a_video_shorter_than_the_window():
    out = dict(M.centred_stacks_from(_frames(2), dt=3))
    assert sorted(out) == [0, 1]
    assert _ids(out[0]) == [0, 0, 1] and _ids(out[1]) == [0, 1, 1]


def test_stop_yields_only_earlier_frames_each_with_its_real_future_tap():
    out = dict(M.centred_stacks_from(_frames(20), dt=3, stop=5))
    assert sorted(out) == [0, 1, 2, 3, 4]
    assert _ids(out[4]) == [1, 4, 7]           # read ahead, not clamped


def test_with_stabilisation_off_the_taps_are_the_raw_frames():
    fr = _frames(10)
    fr[5][2, 2] = 255
    out = dict(M.centred_stacks_from(fr, dt=2))
    assert out[5][2, 2, 1] == 255              # middle channel = frame 5, never resampled
    assert out[3][2, 2, 2] == 255              # frame 5 is frame 3's future tap


def test_centred_builds_never_share_a_directory_with_the_shipped_one():
    assert M._variant("causal", "translation") == ""
    assert M._temporal_name("ardmav_yolo_temporal", True, 6) == "ardmav_yolo_temporal"
    assert (M._temporal_name("ardmav_yolo_temporal", True, 15, M._variant("centred", "off"))
            == "ardmav_yolo_temporal_dt15_centred_staboff")


def test_inference_builds_the_centred_window_with_the_builders_generator():
    it = pytest.importorskip("tools.infer_tiled")
    assert "mde.centred_stacks" in inspect.getsource(it.frame_source)


def test_the_prior_art_arm_is_registered_as_the_paper_specifies():
    from configs.experiments import get
    for name in ("tyolov8_ardmav", "tyolov8_nps"):
        c = get(name)
        assert (c.model_cfg, c.weights, c.epochs) == ("yolov8m.yaml", "yolov8m.pt", 70)
        assert (c.optimizer, c.lr0, c.stack_dt, c.nwd) == ("Adam", 0.001, 15, False)
        assert c.expected_strides == (8, 16, 32)
        assert "centred_staboff" in c.data
        assert c.vram_ref.measured is False


def test_the_optimizer_and_learning_rate_reach_the_trainer(tmp_path):
    from configs.experiments import get
    tr = pytest.importorskip("tools.train")
    kw = tr.ultralytics_kwargs(get("tyolov8_nps"), 0, tmp_path / "tyolov8_nps-s0")
    assert kw["optimizer"] == "Adam" and kw["lr0"] == 0.001 and kw["epochs"] == 70
    assert "optimizer" not in tr.ultralytics_kwargs(get("temporal_nps"), 0, tmp_path / "t-s0")
