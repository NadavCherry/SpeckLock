"""The exact-output fast tracker (realtime.fixedcam.fasttrack): numpy ring state, one vectorised camera move,
vectorised probe targets, lazy roughness (LOOSE) and batched feature rows must reproduce the stock Tracker.

Torch-free (numpy + scipy), like the tracker it mirrors.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from realtime.fixedcam.fasttrack import (FastTracker, RingPool, apply_homography, match_extras,
                                         match_extras_reference)
from realtime.fixedcam.trackfeat import OBS_FIELDS, track_features, track_features_padded
from realtime.fixedcam.tracker import Obs, Tracker, TrackerConfig, _apply

LOOSE = {"forget": 0.97, "min_obs": 2, "commit": 0.6, "min_motion_obs": 1, "min_move": 0.0, "hover_move": 0.0,
         "smooth_commit": 1e9, "smooth_keep": 1e9, "new_max_other": 0.97, "probe_accept_other": 0.9,
         "coast_report": 5, "max_miss": 10}


def _crop_size(size):
    return float(min(96.0, max(24.0, 3.0 * size)))


def _homography(rng):
    return np.eye(3) + rng.normal(0, 1, (3, 3)) * np.array([[2e-3, 2e-3, 6.0], [2e-3, 2e-3, 6.0], [2e-6, 2e-6, 1e-4]])


def test_apply_homography_is_bitwise_the_per_point_apply():
    rng = np.random.default_rng(0)
    for _ in range(40):
        Minv = np.linalg.inv(_homography(rng))
        P = rng.uniform(-50, 2600, (300, 2))
        ref = np.array([_apply(Minv, x, y) for x, y in P])
        assert np.array_equal(apply_homography(Minv, P), ref)
    assert apply_homography(np.eye(3), np.zeros((0, 2))).shape == (0, 2)


def _stream(cfg, n_frames=260, seed=1, move_every=37, feats=True):
    """Drive a stock Tracker and a FastTracker with the same calls, the way FixedCamPipeline does (predict(M),
    probe_targets(proposals), update(motion + probe observations)); assert they agree after every frame."""
    rng = np.random.default_rng(seed)
    lazy = cfg.get("smooth_commit", 0.25) >= 1e8 and cfg.get("smooth_keep", 0.6) >= 1e8
    a, b = Tracker(TrackerConfig(**cfg)), FastTracker(TrackerConfig(**cfg), lazy_roughness=lazy, cap=8)
    for tr in (a, b):
        tr.cfg.px_scale = 0.75
    movers = []
    n_rows = 0
    for f in range(n_frames):
        while len(movers) < 14:
            kind = rng.choice(3, p=[0.4, 0.3, 0.3])
            movers.append({"x": rng.uniform(50, 2500), "y": rng.uniform(50, 1400), "v": rng.normal(0, 2.5, 2),
                           "kind": int(kind), "life": int(rng.integers(5, 200)), "size": rng.uniform(3, 30)})
        movers = [m for m in movers if m["life"] > 0]
        M = None
        if f and f % move_every in (0, 1, 2):
            M = _homography(rng)
        elif rng.random() < 0.3:
            M = np.eye(3) + rng.normal(0, 1e-5, (3, 3))          # 'still' (within allclose 1e-3)
        a.predict(M)
        b.predict(M)
        obs = []
        for m in movers:
            m["life"] -= 1
            m["x"] += m["v"][0] + rng.normal(0, 0.4)
            m["y"] += m["v"][1] + rng.normal(0, 0.4)
            if rng.random() < 0.15:
                continue                                          # missed this frame
            p = rng.dirichlet(np.ones(3) * 0.6)
            p[m["kind"]] += 1.5
            p = p / p.sum()
            feat = np.r_[p, rng.normal(0, 3, 5), rng.integers(0, 2)] if feats else None
            obs.append(Obs(m["x"], m["y"], m["size"], p, feat=feat))
        props = [(o.x, o.y) for o in obs]
        pa, pb = a.probe_targets(props), b.probe_targets(props)
        assert [t.id for t in pa] == [t.id for t in pb]
        for t in pa:                                              # probe crops at the predicted positions
            p = rng.dirichlet(np.ones(3))
            feat = np.r_[p, rng.normal(0, 3, 5), 2.0] if feats else None
            obs.append(Obs(t.x + rng.normal(0, 0.5), t.y + rng.normal(0, 0.5), t.size, p, probe_of=t.id, feat=feat))
        a.update(obs)
        b.update(obs)
        assert a.report() == b.report()
        assert a.candidates() == b.candidates()
        assert [t.id for t in a.tracks] == [t.id for t in b.tracks]
        for s, t in zip(a.tracks, b.tracks):
            assert (s.x, s.y, s.vx, s.vy, s.size, s.label, s.n_obs, s.misses, s.age) == \
                   (t.x, t.y, t.vx, t.vy, t.size, t.label, t.n_obs, t.misses, t.age)
            assert np.array_equal(s.logp, t.logp)
            assert np.array_equal(np.array(list(s.hist), float).reshape(-1, 3), t.hist.array())
            assert list(s.hist) == list(t.hist)
            assert s.moved(75) == t.moved(75) and s.heading_stats(15) == t.heading_stats(15)
        ra, rb = a.feature_rows(), b.feature_rows()
        assert [r[:4] for r in ra] == [r[:4] for r in rb]
        if ra:
            assert np.allclose(np.array([r[4] for r in ra]), np.array([r[4] for r in rb]), rtol=0, atol=1e-12)
        n_rows += len(ra)
    return a, b, n_rows


def test_fast_tracker_equals_tracker_loose_with_camera_moves():
    a, b, n_rows = _stream(LOOSE)
    assert n_rows > 500 and b.pool.cap > 8                       # the pool grew; rows were produced
    assert {t.label for t in a.tracks} - {"unknown"}              # some tracks were labelled


def test_fast_tracker_equals_tracker_strict_config_polyfit_path():
    cfg = {"commit": 0.9, "min_obs": 4, "forget": 0.97, "min_motion_obs": 4, "flight_turn": 0.8, "heading_gate": 1.2}
    _stream(cfg, n_frames=200, seed=3)


def test_lazy_roughness_is_refused_outside_the_loose_config():
    with pytest.raises(AssertionError):
        FastTracker(TrackerConfig())
    FastTracker(TrackerConfig(), lazy_roughness=False)


def test_ring_view_is_the_deque():
    tr = FastTracker(TrackerConfig(**LOOSE), cap=2)._new_track(1.0, 2.0, 5.0)
    ref_h, ref_a = deque(maxlen=160), deque(maxlen=32)
    rng = np.random.default_rng(5)
    for k in range(400):
        item = (k, float(rng.normal()), float(rng.normal()))
        tr.hist.append(item)
        ref_h.append(item)
        v = rng.normal(size=len(OBS_FIELDS))
        tr.appear.append((k, v))
        ref_a.append((k, v))
        if k in (0, 7, 159, 160, 161, 399):
            assert len(tr.hist) == len(ref_h) and list(tr.hist) == list(ref_h) and tr.hist[-1] == ref_h[-1]
            assert [x[0] for x in tr.appear] == [x[0] for x in ref_a]
            assert all(np.array_equal(x[1], y[1]) for x, y in zip(tr.appear, ref_a))
    pool = RingPool(cap=1)
    assert pool.alloc() == 0 and pool.alloc() == 1 and pool.cap == 2


def test_ring_pool_slots_are_reused_clean():
    ft = FastTracker(TrackerConfig(**LOOSE), cap=2)
    t1 = ft._new_track(0.0, 0.0, 4.0)
    t1.hist.append((0, 5.0, 6.0))
    t1.appear.append((0, np.ones(len(OBS_FIELDS))))
    ft._drop(t1)
    t2 = ft._new_track(1.0, 1.0, 4.0)
    assert t2.slot == t1.slot and len(t2.hist) == 0 and not t2.appear
    assert not ft.pool.hist[t2.slot].any() and not ft.pool.app[t2.slot].any()
    t3, t4 = ft._new_track(0.0, 0.0, 4.0), ft._new_track(0.0, 0.0, 4.0)
    assert ft.pool.cap == 4 and len({t2.slot, t3.slot, t4.slot}) == 3


def test_track_features_padded_matches_track_features():
    rng = np.random.default_rng(2)
    items = []
    for i in range(120):
        la = [0, 1, 2, 5, 11, 12, 13, 31, 32, 40, 64][i % 11]
        lh = [1, 2, 3, 4, 14, 15, 16, 59, 60, 61, 160][(i * 7) % 11]
        ap = [(k, np.r_[rng.dirichlet(np.ones(3)), rng.normal(0, 3, 5), rng.integers(0, 3)]) for k in range(la)]
        x0, y0 = rng.uniform(100, 1800, 2)
        hi = [(k, x0 + 0.8 * k + rng.normal(0, 1), y0 + rng.normal(0, 1)) for k in range(lh)]
        items.append((ap, hi, int(rng.integers(1, 300)), int(rng.integers(1, 400)), float(rng.uniform(2, 40))))
    ref = np.array([track_features(ap, hi, n, a, s, 0.75) for ap, hi, n, a, s in items])
    n = len(items)
    A, H = np.zeros((n, 32, len(OBS_FIELDS))), np.zeros((n, 60, 3))
    la, lh = np.ones(n, int), np.zeros(n, int)
    for i, (ap, hi, *_rest) in enumerate(items):
        w = ap[-32:]
        if w:
            A[i, :len(w)] = [v for _, v in w]
            la[i] = len(w)
        h = hi[-60:]
        H[i, 60 - len(h):] = h
        lh[i] = len(h)
    got = track_features_padded(A, la, H, lh, [it[2] for it in items], [it[3] for it in items], [it[4] for it in items], 0.75)
    assert np.allclose(got, ref, rtol=0, atol=1e-12)
    assert track_features_padded(A[:0], la[:0], H[:0], lh[:0], [], [], [], 0.75).shape == (0, ref.shape[1])


def test_match_extras_equals_the_pipeline_loop():
    rng = np.random.default_rng(4)
    for trial in range(60):
        props = [(float(x), float(y)) for x, y in rng.uniform(0, 400, (int(rng.integers(0, 12)), 2))]
        extras = []
        for _ in range(int(rng.integers(0, 15))):
            size = float(rng.uniform(1, 40))
            if props and rng.random() < 0.4:                  # exactly on the match radius of a proposal
                px, py = props[int(rng.integers(len(props)))]
                r = max(6.0, 0.5 * _crop_size(size))
                ang = rng.uniform(0, 2 * np.pi)
                x, y = px + r * np.cos(ang), py + r * np.sin(ang)
            else:
                x, y = rng.uniform(0, 400, 2)
            ev = None if rng.random() < 0.3 else rng.dirichlet(np.ones(3))
            extras.append((float(x), float(y), size, ev))
        got, ref = match_extras(props, extras, _crop_size), match_extras_reference(props, extras, _crop_size)
        assert got[0] == ref[0] and got[2] == ref[2]
        assert got[1].keys() == ref[1].keys() and all(got[1][k] is ref[1][k] for k in ref[1])
