"""tools/camera_motion.py must recover a known camera motion, and report none when there is none.

The README's statements about camera motion now rest on this tool, so the tool is tested
against motion that is known exactly: a textured scene cropped from a moving window.
"""
from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
cm = pytest.importorskip("tools.camera_motion")


def _texture(h=440, w=560, seed=0):
    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur(rng.integers(0, 255, (h, w), dtype=np.uint8), (0, 0), 2)


def test_a_static_camera_measures_no_motion():
    t = _texture()
    frames = [t[100:340, 100:420].copy() for _ in range(20)]
    steps, resp, shape = cm.measure_frames(frames)
    s = cm.stats(steps, resp, shape, (12,))
    assert s["n_frames"] == 20
    assert s["step_px"]["max"] < 0.05
    assert s["aperture_px"]["12"]["max"] < 0.5


def test_a_known_pan_is_recovered_per_step_and_across_the_window():
    t = _texture()
    frames = [t[100:340, 100 + 2 * i: 420 + 2 * i].copy() for i in range(40)]  # 2 px/frame in x
    steps, resp, shape = cm.measure_frames(frames)
    s = cm.stats(steps, resp, shape, (12, 30))
    assert s["step_px"]["median"] == pytest.approx(2.0, abs=0.1)
    assert s["aperture_px"]["12"]["median"] == pytest.approx(24.0, abs=1.0)
    assert s["aperture_px"]["30"]["median"] == pytest.approx(60.0, abs=2.0)
    assert s["response"]["frac_low"] == 0.0


def test_pooling_weights_by_frames_not_by_videos():
    still = np.zeros((99, 2))
    pan = np.tile([[3.0, 4.0]], (1, 1))            # one 5 px step
    p = cm.pooled([(still, np.ones(99)), (pan, np.ones(1))], (12,))
    assert p["n_steps"] == 100
    assert p["step_px"]["median"] == 0.0            # 99 of 100 steps are still
    assert p["step_px"]["max"] == pytest.approx(5.0)
