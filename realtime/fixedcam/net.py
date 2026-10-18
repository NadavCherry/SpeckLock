"""Crop extraction and the tiny drone / bird / other classifier.

Input per candidate: a native-resolution square of side S = clamp(3 * size, 24, 96) px centred on
the candidate, sampled at three moments (t-6, t-3, t) - the crop-level version of the temporal
stack, so wing beats and motion are visible - resized to 32x32, plus a fourth constant plane
log2(S / 32) so the net knows the true scale. Each crop is normalised by its own median and spread
(contrast invariance: a dark bird on sky vs. against terrain).

Outputs: 3 logits (drone, bird, other) and the drone's centre offset (dx, dy) in units of S/2
relative to the crop centre - the head that keeps the reported point on the drone's centre.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

CROP = 32
CLASSES = ("drone", "bird", "other")
OFFSETS = (6, 3, 0)


def crop_size(size: float) -> float:
    return float(min(96.0, max(24.0, 3.0 * size)))


def raw_crops(frames: list[torch.Tensor], centers: list[tuple[float, float, float]]) -> torch.Tensor:
    """frames: [g(t-6), g(t-3), g(t)] fp16/fp32 [H, W] on the device (current view).
    centers: (x, y, S). Returns float32 [N, 3, 32, 32] in grey levels (unnormalised)."""
    if not centers:
        return torch.zeros((0, 3, CROP, CROP), device=frames[0].device)
    boxes = torch.tensor([[0, x - s / 2, y - s / 2, x + s / 2, y + s / 2] for x, y, s in centers],
                         dtype=frames[0].dtype, device=frames[0].device)
    # one roi_align per moment on a [1, 1, H, W] view: no full-frame copy or conversion
    chans = [roi_align(f[None, None], boxes, output_size=(CROP, CROP), spatial_scale=1.0,
                       sampling_ratio=2, aligned=True) for f in frames]
    return torch.cat(chans, dim=1).float()


def normalize(raw: torch.Tensor, sizes: torch.Tensor) -> torch.Tensor:
    """raw [N, 3, 32, 32] grey levels, sizes [N] crop side S -> network input [N, 4, 32, 32]."""
    if raw.shape[0] == 0:
        return torch.zeros((0, 4, CROP, CROP), device=raw.device)
    ref = raw[:, 2:3]
    med = ref.flatten(1).median(dim=1).values.view(-1, 1, 1, 1)
    spread = (ref - med).abs().flatten(1).mean(dim=1).view(-1, 1, 1, 1)
    x = (raw - med) / (spread * 2.0 + 4.0)
    plane = torch.log2(sizes.float() / CROP).view(-1, 1, 1, 1).expand(-1, 1, CROP, CROP)
    return torch.cat([x, plane.to(x.device)], dim=1)


def extract(frames: list[torch.Tensor], centers: list[tuple[float, float, float]]) -> torch.Tensor:
    raw = raw_crops(frames, centers)
    sizes = torch.tensor([s for _, _, s in centers], device=raw.device)
    return normalize(raw, sizes)


APPEARANCE = ("centre_signed", "centre_contrast", "blob_area", "change_3", "change_6")


def appearance_stats(x: torch.Tensor) -> torch.Tensor:
    """Cheap per-crop appearance measurements from the network input x [N, 4, 32, 32]
    (channel 2 = the current moment, normalised to the crop's own median and spread).

    centre_signed    mean of the central 8x8 (sign: darker or brighter than the background)
    centre_contrast  mean |.| of the central 8x8 minus that of the area outside the central 16x16
    blob_area        share of the central 16x16 whose |.| exceeds 1.5 (the target's extent)
    change_3/_6      mean |current - moment t-3 / t-6| over the central 8x8 (how fast it changes)

    Tracked over time they carry what one crop cannot: a flapping wing makes contrast and area
    oscillate at 2-10 Hz, compression flicker blinks, a drone body stays steady.
    """
    if x.shape[0] == 0:
        return torch.zeros((0, len(APPEARANCE)), device=x.device)
    cur, mid, old = x[:, 2], x[:, 1], x[:, 0]
    c, inner = slice(12, 20), slice(8, 24)
    centre = cur[:, c, c]
    outer = (cur.abs().sum((1, 2)) - cur[:, inner, inner].abs().sum((1, 2))) / (CROP * CROP - 256)
    return torch.stack([centre.mean((1, 2)),
                        centre.abs().mean((1, 2)) - outer,
                        (cur[:, inner, inner].abs() > 1.5).float().mean((1, 2)),
                        (centre - mid[:, c, c]).abs().mean((1, 2)),
                        (centre - old[:, c, c]).abs().mean((1, 2))], dim=1)


def _block(cin, cout, stride):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout),
                         nn.ReLU(inplace=True))


class TinyNet(nn.Module):
    """~0.44 M parameters; 32x32x4 -> 3 logits + 2 offsets."""

    def __init__(self, width: int = 32):
        super().__init__()
        w = width
        self.body = nn.Sequential(
            _block(4, w, 1), _block(w, w, 1),
            _block(w, 2 * w, 2), _block(2 * w, 2 * w, 1),          # 16x16
            _block(2 * w, 4 * w, 2), _block(4 * w, 4 * w, 1),      # 8x8
            _block(4 * w, 4 * w, 2),                               # 4x4
        )
        self.cls = nn.Linear(4 * w * 2, 3)
        self.reg = nn.Linear(4 * w * 2, 2)

    def forward(self, x):
        f = self.body(x)
        f = torch.cat([F.adaptive_avg_pool2d(f, 1), F.adaptive_max_pool2d(f, 1)], 1).flatten(1)
        return self.cls(f), torch.tanh(self.reg(f))
