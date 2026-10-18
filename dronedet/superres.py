"""Whole-frame upscaling for a "super-resolution first" front end.

``Upscaler(method, scale)`` turns a BGR frame into a ``scale``x larger BGR frame:

- ``bicubic``: ``cv2.resize`` (the control: more pixels, no new information);
- ``area``: ``cv2.resize`` with area interpolation, for a scale < 1 (a detector run on downscaled frames);
- ``fsrcnn``: the learned FSRCNN x4 network (``work/models/FSRCNN_x4.pb``, the OpenCV dnn_superres
  model) ported to PyTorch so it runs on the GPU, applied to the luma channel (chroma bicubic),
  then area-resampled from x4 to the requested scale.

The port reads the network's weights through ``cv2.dnn`` (conv kernels, biases, PReLU slopes) and
is checked against OpenCV's own ``dnn_superres`` output in ``dronedet/tests/test_superres.py``.
torch and the weights are loaded lazily, so importing this module needs numpy and OpenCV only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

FSRCNN_PB = Path(__file__).resolve().parents[1] / "work" / "models" / "FSRCNN_x4.pb"
# The graph's output stage adds a constant that cv2.dnn does not expose as a blob; fitted against
# OpenCV's dnn_superres output: 0.4667 on every sub-pixel channel (std 0.001), independent of the input.
OUT_SHIFT = 0.4667


def fsrcnn_weights(pb: str | Path = FSRCNN_PB) -> list[dict]:
    """[{'w': (out, in, k, k), 'b': (out,) or None, 'alpha': (out,) or None}] for conv1..conv8."""
    import cv2

    net = cv2.dnn.readNetFromTensorflow(str(pb))
    names = list(net.getLayerNames())
    layers = []
    for i, n in enumerate(names):
        if not n.startswith("conv"):
            continue
        w = net.getLayer(n).blobs[0]
        b = alpha = None
        # the graph after each conv: add (bias, Scale) -> Relu, Abs, sub, mul (slope, Scale), Power, add
        for m in names[i + 1:i + 8]:
            if m.startswith("conv") or m == "DepthToSpace":
                break
            lay = net.getLayer(m)
            if lay.type == "Scale" and lay.blobs:
                if b is None and m.startswith("add"):
                    b = lay.blobs[0].reshape(-1)
                elif m.startswith("mul"):
                    alpha = lay.blobs[0].reshape(-1)
        layers.append({"w": w, "b": b, "alpha": alpha})
    return layers


class _FSRCNN:
    def __init__(self, device="cuda", half=True):
        import torch

        self.torch, self.device = torch, device
        self.dtype = torch.float16 if (half and device != "cpu") else torch.float32
        self.layers = []
        for L in fsrcnn_weights():
            w = torch.tensor(L["w"], dtype=self.dtype, device=device)
            b = torch.tensor(L["b"], dtype=self.dtype, device=device) if L["b"] is not None else None
            a = torch.tensor(L["alpha"], dtype=self.dtype, device=device) if L["alpha"] is not None else None
            self.layers.append((w, b, a))

    def luma_x4(self, y: np.ndarray) -> np.ndarray:
        """uint8 HxW luma -> float32 4Hx4W luma in [0, 1]."""
        torch = self.torch
        F = torch.nn.functional
        x = torch.tensor(y, dtype=self.dtype, device=self.device)[None, None] / 255.0
        with torch.no_grad():
            for w, b, a in self.layers:
                x = F.conv2d(x, w, b, padding=w.shape[-1] // 2)
                if a is not None:
                    x = F.prelu(x, a)
            x = F.pixel_shuffle(x, 4) + OUT_SHIFT
        return x[0, 0].float().clamp(0, 1).cpu().numpy()


class Upscaler:
    def __init__(self, method: str = "bicubic", scale: float = 2.0, device: str = "cuda"):
        if method not in ("bicubic", "fsrcnn", "area"):
            raise ValueError(f"unknown upscaler {method!r}")
        self.method, self.scale = method, float(scale)
        self.net = _FSRCNN(device) if method == "fsrcnn" else None

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        import cv2

        h, w = bgr.shape[:2]
        size = (int(round(w * self.scale)), int(round(h * self.scale)))
        if self.method in ("bicubic", "area"):
            return cv2.resize(bgr, size, interpolation=cv2.INTER_CUBIC if self.method == "bicubic" else cv2.INTER_AREA)
        ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        y4 = (self.net.luma_x4(ycc[:, :, 0]) * 255.0 + 0.5).astype(np.uint8)
        y = cv2.resize(y4, size, interpolation=cv2.INTER_AREA) if size != (4 * w, 4 * h) else y4
        crcb = cv2.resize(ycc[:, :, 1:], size, interpolation=cv2.INTER_CUBIC)
        return cv2.cvtColor(np.dstack([y, crcb]), cv2.COLOR_YCrCb2BGR)
