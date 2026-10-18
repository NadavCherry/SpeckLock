"""Causal GPU motion stage for fixed / PTZ cameras.

Per frame (native resolution, float16/32 on the GPU):

    bg     = median(g[t-8], g[t-16], g[t-32])        lagged median: a mover is gone from 2 of 3
                                                       lags once it has moved > size/16 px/frame
    d      = g - bg
    sigma  = EMA(|d|)                                  per-pixel noise: swaying grass, shimmering
                                                       cloud edges and compression noise raise it
    z      = |d| / (sigma + c0 + k * pulse)            pulse = how far this frame's median |d| (sparse
                                                       grid) exceeds its running level: the keyframe
                                                       "pulse" of cameras with a fixed keyframe
                                                       interval (GOP), which lifts |d| everywhere at once
    mask   = z > zthr & |d| > dthr
    act    = EMA(mask) on a 4x4-cell grid              a cell that fires often is clutter, not a
                                                       flier: suppressed while act > act_max

Proposals are connected components of the surviving cells (downloaded as a small cell-grid map),
ranked by summed z and capped at ``max_props``. Static OSD rectangles are masked out.

Camera motion: a 256-px phase correlation every frame (cheap). Only when it reports motion is a
homography estimated (LK on a 512-px frame); the lag ring, sigma and activity are then warped
into the current view, and ``moving`` is exposed so the tracker can carry its tracks through it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F

LAGS = (8, 16, 32)


@dataclass
class Proposal:
    x: float
    y: float
    size: float     # native px (cell-quantised)
    score: float    # summed z over the component


class CameraGate:
    """Phase correlation on a small frame; homography (LK) only while the camera moves."""

    def __init__(self, small_w: int = 256, lk_w: int = 512, shift_thr: float = 0.35):
        self.small_w, self.lk_w, self.shift_thr = small_w, lk_w, shift_thr
        self.prev_small = None
        self.prev_lk = None
        self.win = None

    def update(self, gray_gpu: torch.Tensor) -> tuple[np.ndarray, float]:
        """Returns (M, native_shift): M maps CURRENT native px -> PREVIOUS native px."""
        H, W = gray_gpu.shape
        sh = max(16, int(round(H * self.small_w / W)))
        lh = int(round(H * self.lk_w / W))
        g = gray_gpu[None, None].float()
        small = F.interpolate(g, size=(sh, self.small_w), mode="area")[0, 0].cpu().numpy()
        lk = F.interpolate(g, size=(lh, self.lk_w), mode="area")[0, 0]      # stays on the GPU
        M, shift = np.eye(3), 0.0
        if self.prev_small is not None:
            if self.win is None:
                self.win = cv2.createHanningWindow((self.small_w, sh), cv2.CV_32F)
            (dx, dy), resp = cv2.phaseCorrelate(small, self.prev_small, self.win)
            shift = float(np.hypot(dx, dy)) * W / self.small_w
            if shift > self.shift_thr * W / self.small_w or resp < 0.2:
                M = self._homography(lk, self.prev_lk, W)
                if M is None:
                    M = np.array([[1, 0, dx * W / self.small_w], [0, 1, dy * W / self.small_w],
                                  [0, 0, 1]], float)
        self.prev_small = small
        self.prev_lk = lk
        return M, shift

    def _homography(self, lk_cur, lk_prev, W):
        if lk_prev is None:
            return None
        cur = lk_cur.clamp(0, 255).byte().cpu().numpy()
        prev = lk_prev.clamp(0, 255).byte().cpu().numpy()
        pts = cv2.goodFeaturesToTrack(cur, 300, 0.01, 8)
        if pts is None or len(pts) < 20:
            return None
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(cur, prev, pts, None, winSize=(21, 21), maxLevel=4)
        ok = st.ravel() == 1
        if ok.sum() < 20:
            return None
        Hm, _ = cv2.findHomography(pts[ok], nxt[ok], cv2.RANSAC, 1.5)
        if Hm is None:
            return None
        s = self.lk_w / W
        S, Si = np.diag([s, s, 1.0]), np.diag([1 / s, 1 / s, 1.0])
        return Si @ Hm @ S


def _warp(img: torch.Tensor, M: np.ndarray, fill: float) -> torch.Tensor:
    """Resample img (in the PREVIOUS view) into the current view; M: current px -> previous px."""
    H, W = img.shape[-2:]
    dev = img.device
    ys, xs = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32),
                            torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")
    Mt = torch.tensor(M, device=dev, dtype=torch.float32)
    den = Mt[2, 0] * xs + Mt[2, 1] * ys + Mt[2, 2]
    sx = (Mt[0, 0] * xs + Mt[0, 1] * ys + Mt[0, 2]) / den
    sy = (Mt[1, 0] * xs + Mt[1, 1] * ys + Mt[1, 2]) / den
    grid = torch.stack([sx / (W - 1) * 2 - 1, sy / (H - 1) * 2 - 1], -1)[None]
    src = img.float()
    squeeze = src.dim() == 2
    if squeeze:
        src = src[None, None]
    elif src.dim() == 3:
        src = src[None]
    out = F.grid_sample(src, grid, mode="bilinear", padding_mode="border", align_corners=True)
    inside = ((sx >= 0) & (sx <= W - 1) & (sy >= 0) & (sy <= H - 1))
    out = torch.where(inside, out, torch.full_like(out, fill))
    out = out[0, 0] if squeeze else out[0]
    return out.to(img.dtype)


class MotionStage:
    def __init__(self, height: int, width: int, device: str = "cuda", cell: int = 4,
                 zthr: float = 5.0, dthr: float = 9.0, c0: float = 1.5, k_global: float = 3.0,
                 sigma_alpha: float = 0.03, act_alpha: float = 0.02, act_max: float = 0.08,
                 max_props: int = 24, osd: list | None = None, keep: int = 7,
                 short_lags: bool = False, restart_after_move: bool = False, settle: int = 12):
        # short_lags: propose from the 5th frame on, with the lags scaled down to what the ring
        #   holds (they grow to LAGS), instead of staying blind until the ring is half full;
        # restart_after_move: after a camera move, drop the ring rather than warping blurred pan
        #   frames into the background, and resume after `settle` still frames.
        self.short_lags, self.restart_after_move, self.settle = short_lags, restart_after_move, settle
        self.H, self.W, self.dev, self.cell = height, width, device, cell
        self.zthr, self.dthr, self.c0, self.kg = zthr, dthr, c0, k_global
        self.sa, self.aa, self.amax, self.max_props = sigma_alpha, act_alpha, act_max, max_props
        self.ring: list = []                          # (fp16 frame, map current->its view)
        self.nring = max(LAGS) + 1
        self.keep = keep                              # recent frames kept for crops
        self.sigma = torch.full((height, width), 4.0, device=device, dtype=torch.float16)
        hc, wc = height // cell, width // cell
        self.act = torch.zeros((hc, wc), device=device)
        self.osd = torch.zeros((hc, wc), dtype=torch.bool, device=device)
        for r in (osd or []):
            x0, y0, x1, y1 = r
            self.osd[int(y0 * height) // cell:int(np.ceil(y1 * height / cell)) + 1,
                     int(x0 * width) // cell:int(np.ceil(x1 * width / cell)) + 1] = True
        self.gate = CameraGate()
        self.gn_level = None
        self.profile = False
        self.prof: dict[str, float] = {}
        self.moving = False
        self.still_frames = 0
        self.t = -1

    def recent(self, back: int) -> torch.Tensor:
        """Frame t-back (clamped to the oldest available), resampled into the current view."""
        img, A = self.ring[max(0, len(self.ring) - 1 - back)]
        if np.allclose(A, np.eye(3), atol=1e-4):
            return img
        return _warp(img, A, 0)

    def _tick(self, key, t0):
        """Sub-stage timing (only when profiling: it synchronises the GPU)."""
        if not self.profile:
            return 0.0
        torch.cuda.synchronize()
        t = time.perf_counter()
        self.prof[key] = self.prof.get(key, 0.0) + 1000.0 * (t - t0)
        return t

    def step(self, gray: torch.Tensor, protect: list | None = None) -> tuple[list[Proposal], np.ndarray]:
        """gray: uint8 [H, W] on the device. Returns (proposals, M current->previous).
        protect: (x, y, r) regions, e.g. live drone tracks, whose cells the activity map may not
        mute - a slow drone fires in the same cells frame after frame, like clutter does."""
        self.t += 1
        t0 = time.perf_counter() if self.profile else 0.0
        M, shift = self.gate.update(gray)
        t0 = self._tick("camera gate (cpu phase corr.)", t0)
        if not np.allclose(M, np.eye(3), atol=1e-3):
            self.moving = True
            self.still_frames = 0
            # lazy: each stored frame keeps a map current-view -> its own view
            self.ring = [] if self.restart_after_move else [(img, A @ M) for img, A in self.ring]
            self.sigma = _warp(self.sigma, M, 8.0)
            up = F.interpolate(self.act[None, None], scale_factor=self.cell, mode="nearest")[0, 0]
            self.act = F.interpolate(_warp(up, M, 0.0)[None, None], size=self.act.shape,
                                     mode="area")[0, 0]
            t0 = self._tick("camera move warps", t0)
        else:
            self.still_frames += 1
            if self.still_frames > self.settle:
                self.moving = False
        g = gray.half()                               # one conversion; the ring is fp16
        self.ring.append((g, np.eye(3)))
        if len(self.ring) > self.nring:
            self.ring.pop(0)
        n = len(self.ring)
        if self.short_lags:
            if n < 5 or self.moving:
                return [], M
            lags = tuple(min(L, max(1, (n - 1) * L // max(LAGS))) for L in LAGS)
        else:
            if n <= max(LAGS) // 2 or self.moving:
                return [], M
            lags = LAGS
        a, b, c = (self.recent(L) for L in lags)
        bg = torch.maximum(torch.minimum(a, b), torch.minimum(torch.maximum(a, b), c))
        d = g - bg
        ad = d.abs()
        self.sigma.mul_(1 - self.sa).add_(self.sa * ad)
        # keyframe pulse, kept on the device (no mid-stage synchronisation)
        gn = ad[::16, ::16].float().median()
        if self.gn_level is None:
            self.gn_level = gn.clone()
        self.gn_level = 0.95 * self.gn_level + 0.05 * torch.minimum(gn, 2 * self.gn_level)
        pulse = (gn - self.gn_level).clamp(min=0).half()
        z = ad / (self.sigma + self.c0 + self.kg * pulse)
        m = (z > self.zthr) & (ad > self.dthr)
        zc = F.max_pool2d((z * m).float()[None, None], self.cell, self.cell)[0, 0]
        cellm = zc > 0
        self.act.mul_(1 - self.aa).add_(self.aa * cellm.float())
        quiet = self.act <= self.amax
        if protect:
            cs = self.cell
            for (px, py, r) in protect:
                y0, y1 = max(0, int((py - r) // cs)), int((py + r) // cs) + 1
                x0, x1 = max(0, int((px - r) // cs)), int((px + r) // cs) + 1
                quiet[y0:y1, x0:x1] = True
        keep = cellm & quiet & ~self.osd
        zc = torch.where(keep, zc, torch.zeros_like(zc))
        t0 = self._tick("background+z (gpu, full res)", t0)
        zc_np = zc.cpu().numpy()
        t0 = self._tick("download cell map", t0)
        props = self._components(zc_np)
        self._tick("components (cpu)", t0)
        return props, M

    def _components(self, zc: np.ndarray) -> list[Proposal]:
        m = (zc > 0).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if n <= 1:
            return []
        ys, xs = np.nonzero(m)
        lv = lab[ys, xs]
        w = zc[ys, xs].astype(np.float64)
        sw = np.bincount(lv, weights=w, minlength=n)
        sx = np.bincount(lv, weights=w * xs, minlength=n)
        sy = np.bincount(lv, weights=w * ys, minlength=n)
        out = []
        c = self.cell
        for i in range(1, n):
            if sw[i] <= 0:
                continue
            bw, bh = st[i][2], st[i][3]
            if max(bw, bh) * c > 160:
                continue
            out.append(Proposal((sx[i] / sw[i] + 0.5) * c, (sy[i] / sw[i] + 0.5) * c,
                                float(max(bw, bh) * c), float(sw[i])))
        out.sort(key=lambda p: -p.score)
        return out[:self.max_props]

    def appearance_peaks(self, k: int = 8, min_contrast: float = 14.0) -> list[Proposal]:
        """The k strongest compact blobs (local contrast: black-hat or top-hat, 9 px) in the current
        frame - proposals that do not need motion, so a hovering drone can be (re)acquired. The
        tracker still demands flight evidence before any of them becomes a drone."""
        if not self.ring:
            return []
        g = self.ring[-1][0].float()[None, None]
        closing = -F.max_pool2d(-F.max_pool2d(g, 9, 1, 4), 9, 1, 4)
        opening = F.max_pool2d(-F.max_pool2d(-g, 9, 1, 4), 9, 1, 4)
        con = torch.maximum(closing - g, g - opening)[0, 0]
        cell = F.max_pool2d(con[None, None], self.cell, self.cell)[0, 0]
        cell = torch.where(self.osd, torch.zeros_like(cell), cell)
        peak = (cell == F.max_pool2d(cell[None, None], 9, 1, 4)[0, 0]) & (cell > min_contrast)
        ys, xs = torch.nonzero(peak, as_tuple=True)
        if len(ys) == 0:
            return []
        v = cell[ys, xs]
        order = torch.argsort(v, descending=True)[:k]
        c = self.cell
        return [Proposal((float(x) + 0.5) * c, (float(y) + 0.5) * c, 6.0, float(val))
                for x, y, val in zip(xs[order].tolist(), ys[order].tolist(), v[order].tolist())]
