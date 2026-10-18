"""dronedet/superres.py: the PyTorch FSRCNN port matches OpenCV's dnn_superres, and the upscaler's shapes."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dronedet.superres import FSRCNN_PB, Upscaler  # noqa: E402


def _scene(h=48, w=64, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 140, np.uint8)
    img[10:20, 12:30] = (40, 60, 70)
    img[30:34, 40:44] = 20
    return np.clip(img.astype(int) + rng.integers(-6, 7, img.shape), 0, 255).astype(np.uint8)


def test_bicubic_shape():
    out = Upscaler("bicubic", 2)(_scene())
    assert out.shape == (96, 128, 3)


def test_unknown_method_rejected():
    with pytest.raises(ValueError):
        Upscaler("magic", 2)


@pytest.mark.skipif(not FSRCNN_PB.exists(), reason="FSRCNN weights not present")
def test_fsrcnn_port_matches_opencv():
    torch = pytest.importorskip("torch")
    import cv2

    if not hasattr(cv2, "dnn_superres"):
        pytest.skip("opencv-contrib dnn_superres not available")
    img = _scene()
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(str(FSRCNN_PB))
    sr.setModel("fsrcnn", 4)
    ref = cv2.cvtColor(sr.upsample(img), cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(float)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = cv2.cvtColor(Upscaler("fsrcnn", 4, device=dev)(img), cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(float)
    assert out.shape == ref.shape
    assert np.abs(out - ref)[8:-8, 8:-8].mean() < 1.5


def test_area_downscale_shape():
    out = Upscaler("area", 0.5)(_scene())
    assert out.shape == (24, 32, 3)
