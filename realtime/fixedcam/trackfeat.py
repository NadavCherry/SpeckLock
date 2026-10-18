"""Track-level features and a small track classifier: appearance over time + kinematics.

Position alone cannot separate a hovering drone from clutter jittering in place, nor a bird from
a drone at a few pixels. What separates them is how the target *looks over time*: a wing beat
makes contrast and blob area oscillate at 2-10 Hz (resolved at 25 fps), compression flicker
blinks, a drone body stays steady, and the crop classifier's own opinion is more telling as a
distribution over a second than as one frame's argmax.

Each observation carries a vector (see ``OBS_FIELDS``): the crop classifier's raw class
probabilities, the ``net.APPEARANCE`` measurements, and where the observation came from.
``track_features`` summarises the last ``window`` of them, plus kinematics, into ``FEATURES``.
``TrackClassifier`` scores that vector with a model stored as plain JSON (logistic regression or
a one-hidden-layer MLP), so the runtime needs numpy only.

numpy only: unit-tested in CI.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

OBS_FIELDS = ("p_drone", "p_bird", "p_other", "centre_signed", "centre_contrast", "blob_area",
              "change_3", "change_6", "source")   # source: 0 motion/appearance, 1 external, 2 probe
SRC_EXTERNAL, SRC_PROBE = 1.0, 2.0

FEATURES = ("p_drone_mean", "p_bird_mean", "p_other_mean", "p_drone_std", "p_bird_std",
            "sign_consistency", "contrast_mean", "contrast_cv", "area_mean", "area_cv",
            "change3_mean", "change6_mean", "flicker_contrast", "flicker_area",
            "external_share", "probe_share", "speed", "turn", "straightness", "speed_long",
            "log_obs", "observed_share", "log_size")


def band_power(series: np.ndarray, fps: float = 25.0, lo: float = 2.0, hi: float = 10.0) -> float:
    """Share of the (detrended) signal's spectral power between lo and hi Hz; 0 if too short."""
    s = np.asarray(series, float)
    if len(s) < 12:
        return 0.0
    t = np.arange(len(s))
    s = s - np.polyval(np.polyfit(t, s, 1), t)
    p = np.abs(np.fft.rfft(s)) ** 2
    f = np.fft.rfftfreq(len(s), 1.0 / fps)
    tot = p[1:].sum()
    return float(p[(f >= lo) & (f <= hi)].sum() / tot) if tot > 1e-12 else 0.0


def _cv(v: np.ndarray) -> float:
    m = float(np.mean(np.abs(v)))
    return float(np.std(v) / m) if m > 1e-6 else 0.0


def track_features(appear, hist, n_obs: int, age: int, size: float, px_scale: float = 1.0,
                   window: int = 32, kin_window: int = 15) -> np.ndarray:
    """appear: sequence of (age, obs_vector) - the track's observations, oldest first;
    hist: sequence of (age, x, y) in native px. Returns a vector ordered as ``FEATURES``."""
    A = np.array([v for _, v in list(appear)[-window:]], float).reshape(-1, len(OBS_FIELDS))
    if len(A) == 0:
        A = np.zeros((1, len(OBS_FIELDS)))
    p_d, p_b, p_o = A[:, 0], A[:, 1], A[:, 2]
    sgn = np.sign(A[:, 3])
    med = np.sign(np.median(A[:, 3])) or 1.0
    H = np.array(list(hist), float).reshape(-1, 3)

    def kin(h):
        if len(h) < 3:
            return math.pi, 0.0, 0.0
        v = np.diff(h[:, 1:], axis=0)
        dt = np.maximum(1.0, np.diff(h[:, 0]))
        sp = np.hypot(v[:, 0], v[:, 1]) / dt
        ang = np.arctan2(v[:, 1], v[:, 0])
        turn = np.abs(np.angle(np.exp(1j * np.diff(ang))))
        path = float(np.hypot(v[:, 0], v[:, 1]).sum())
        net = float(np.hypot(*(h[-1, 1:] - h[0, 1:])))
        return float(np.median(turn)) if len(turn) else math.pi, float(np.median(sp)), net / path if path > 1e-6 else 0.0

    turn, speed, straight = kin(H[-kin_window:])
    _, speed_long, _ = kin(H[-4 * kin_window:])
    return np.array([
        p_d.mean(), p_b.mean(), p_o.mean(), p_d.std(), p_b.std(),
        float(np.mean(sgn == med)), A[:, 4].mean(), _cv(A[:, 4]), A[:, 5].mean(), _cv(A[:, 5]),
        A[:, 6].mean(), A[:, 7].mean(), band_power(A[:, 4]), band_power(A[:, 5]),
        float(np.mean(A[:, 8] == SRC_EXTERNAL)), float(np.mean(A[:, 8] == SRC_PROBE)),
        speed * px_scale, turn, straight, speed_long * px_scale,
        math.log1p(n_obs), n_obs / max(1, age + 1), math.log(max(1.0, size * px_scale)),
    ], dtype=float)


def _band_power_rows(S: np.ndarray, fps: float = 25.0, lo: float = 2.0, hi: float = 10.0) -> np.ndarray:
    """band_power for a stack of equal-length series S (n, L), the linear detrend in closed form."""
    n, L = S.shape
    if L < 12:
        return np.zeros(n)
    t = np.arange(L, dtype=float)
    tc = t - t.mean()
    slope = (S - S.mean(1, keepdims=True)) @ tc / (tc @ tc)
    icpt = S.mean(1) - slope * t.mean()
    R = S - (slope[:, None] * t[None, :] + icpt[:, None])
    p = np.abs(np.fft.rfft(R, axis=1)) ** 2
    f = np.fft.rfftfreq(L, 1.0 / fps)
    tot = p[:, 1:].sum(1)
    band = p[:, (f >= lo) & (f <= hi)].sum(1)
    return np.where(tot > 1e-12, band / np.where(tot > 1e-12, tot, 1.0), 0.0)


def _masked_median(V: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """np.median of V[i][valid[i]] per row (nan where a row has no valid entry): invalid entries sorted last."""
    S = np.sort(np.where(valid, V, np.inf), axis=1)
    k = valid.sum(1)
    r = np.arange(len(V))
    lo = S[r, np.maximum(k - 1, 0) // 2]
    hi = S[r, np.minimum(k // 2, V.shape[1] - 1)]          # odd k: hi == lo
    with np.errstate(invalid="ignore"):
        return np.where(k > 0, (lo + hi) / 2, np.nan)


def _kin_padded(H: np.ndarray, lh: np.ndarray, L: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """track_features' kin() over the last min(lh, L) rows of right-aligned histories H (n, LH, 3)."""
    n, LH, _ = H.shape
    W = H[:, LH - L:]
    m = np.minimum(lh, L)                                 # valid rows, right-aligned
    v = np.diff(W[:, :, 1:], axis=1)
    dt = np.maximum(1.0, np.diff(W[:, :, 0], axis=1))
    step = np.hypot(v[:, :, 0], v[:, :, 1])
    vs = np.arange(L - 1)[None, :] >= (L - m)[:, None]    # step j valid when rows j and j+1 are
    sp = np.where(vs, step / dt, np.nan)
    ang = np.arctan2(v[:, :, 1], v[:, :, 0])
    turn = np.abs(np.angle(np.exp(1j * np.diff(ang, axis=1))))
    vt = np.arange(L - 2)[None, :] >= (L - m)[:, None]   # turn j needs steps j and j+1
    turn = np.where(vt, turn, np.nan)
    ok = m >= 3
    tm = np.where(ok & vt.any(1), _masked_median(turn, vt), math.pi)
    sm = np.where(ok, _masked_median(sp, vs), 0.0)
    path = np.where(vs, step, 0.0).sum(1)
    first = np.clip(L - m, 0, L - 1)
    net = np.hypot(W[:, -1, 1] - W[np.arange(n), first, 1], W[:, -1, 2] - W[np.arange(n), first, 2])
    st = np.where(path > 1e-6, net / np.where(path > 1e-6, path, 1.0), 0.0)
    return np.where(ok, tm, math.pi), np.where(ok, sm, 0.0), np.where(ok, st, 0.0)


def track_features_padded(A: np.ndarray, la: np.ndarray, H: np.ndarray, lh: np.ndarray, n_obs, age, size,
                          px_scale: float = 1.0, kin_window: int = 15) -> np.ndarray:
    """``track_features`` for many tracks at once, from padded windows:
    A (n, W, len(OBS_FIELDS)) - each track's last <= W appearance vectors, oldest first, LEFT-aligned, la valid rows
    (>= 1; a track with none passes one zero row, as track_features does);
    H (n, LH, 3) - its last <= LH (age, x, y) rows, RIGHT-aligned, lh valid (LH >= 4 * kin_window);
    n_obs, age, size: per track. Equal to track_features row by row up to floating-point summation order (~1e-15);
    the band powers are grouped by window length. Returns (n, len(FEATURES))."""
    n = len(la)
    if n == 0:
        return np.zeros((0, len(FEATURES)))
    la = np.asarray(la)
    lh = np.asarray(lh)
    W = A.shape[1]
    M = np.arange(W)[None, :] < la[:, None]
    cnt = la.astype(float)

    def mean(V):
        return np.where(M, V, 0.0).sum(1) / cnt

    def std(V):
        mu = mean(V)
        return np.sqrt(np.where(M, (V - mu[:, None]) ** 2, 0.0).sum(1) / cnt)

    def cv(V):
        m = mean(np.abs(V))
        return np.where(m > 1e-6, std(V) / np.where(m > 1e-6, m, 1.0), 0.0)
    med = np.sign(_masked_median(A[:, :, 3], M))
    med[med == 0] = 1.0
    sgn = np.sign(A[:, :, 3])
    band4, band5 = np.zeros(n), np.zeros(n)
    for L in np.unique(la[la >= 12]):
        idx = np.nonzero(la == L)[0]
        band4[idx] = _band_power_rows(A[idx, :L, 4])
        band5[idx] = _band_power_rows(A[idx, :L, 5])
    turn, speed, straight = _kin_padded(H, lh, kin_window)
    _, speed_long, _ = _kin_padded(H, lh, 4 * kin_window)
    n_obs = np.asarray(n_obs, float)
    age = np.asarray(age, float)
    size = np.asarray(size, float)
    return np.stack([
        mean(A[:, :, 0]), mean(A[:, :, 1]), mean(A[:, :, 2]), std(A[:, :, 0]), std(A[:, :, 1]),
        mean((sgn == med[:, None]).astype(float)), mean(A[:, :, 4]), cv(A[:, :, 4]), mean(A[:, :, 5]), cv(A[:, :, 5]),
        mean(A[:, :, 6]), mean(A[:, :, 7]), band4, band5,
        mean((A[:, :, 8] == SRC_EXTERNAL).astype(float)), mean((A[:, :, 8] == SRC_PROBE).astype(float)),
        speed * px_scale, turn, straight, speed_long * px_scale,
        np.log1p(n_obs), n_obs / np.maximum(1, age + 1), np.log(np.maximum(1.0, size * px_scale))], axis=1)


class TrackClassifier:
    """p(drone) for a track from ``FEATURES``; the model is JSON:
    {"features": [...], "mean": [...], "std": [...], and either "w", "b" (logistic regression) or
    "W1", "b1", "w2", "b2" (one hidden ReLU layer)}."""

    def __init__(self, model: dict | str | Path):
        m = model if isinstance(model, dict) else json.loads(Path(model).read_text(encoding="utf-8"))
        if list(m["features"]) != list(FEATURES):
            raise ValueError("track classifier was trained on a different feature list")
        self.mean = np.asarray(m["mean"], float)
        self.std = np.asarray(m["std"], float)
        self.m = m

    def __call__(self, f: np.ndarray) -> float:
        z = (np.asarray(f, float) - self.mean) / self.std
        if "W1" in self.m:
            h = np.maximum(0.0, np.asarray(self.m["W1"]) @ z + np.asarray(self.m["b1"]))
            s = float(np.asarray(self.m["w2"]) @ h + self.m["b2"])
        else:
            s = float(np.asarray(self.m["w"]) @ z + self.m["b"])
        return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, s))))
