"""The evidence-accumulating tracker of the fixed-camera pipeline (realtime.fixedcam.tracker).

Torch-free by design: this module is what decides "drone" vs "bird", so it must stay testable in CI.
"""

from __future__ import annotations

import numpy as np
import pytest

from realtime.fixedcam.tracker import Obs, Tracker, TrackerConfig

DRONEY = np.array([0.8, 0.15, 0.05])
BIRDY = np.array([0.15, 0.8, 0.05])


def _run(tr, frames):
    """frames: list of lists of Obs; returns the report after each frame."""
    out = []
    for obs in frames:
        tr.predict(None)
        tr.update(obs)
        out.append(tr.report())
    return out


def test_a_drone_is_labelled_only_after_enough_evidence():
    tr = Tracker()
    reps = _run(tr, [[Obs(100 + 2 * f, 50, 8, DRONEY)] for f in range(12)])
    first = next(i for i, r in enumerate(reps) if r)
    assert first >= TrackerConfig().min_obs - 1
    x, y, label, conf, tid, observed = reps[-1][0]
    assert label == "drone" and conf > 0.9 and observed
    assert abs(x - 122) < 1e-6          # the reported point is the latest observation


def test_a_long_lived_bird_never_becomes_a_drone():
    # the old offline rule labelled any track >= 120 frames as a drone
    tr = Tracker()
    reps = _run(tr, [[Obs(300 + f, 200 + 0.5 * f, 6, BIRDY)] for f in range(200)])
    labels = {p[2] for r in reps for p in r}
    assert labels == {"bird"}


def test_ambiguous_frames_do_not_flip_a_confident_track():
    tr = Tracker()
    frames = [[Obs(100 + f, 50, 8, DRONEY)] for f in range(20)]
    frames += [[Obs(120 + f, 50, 8, np.array([0.45, 0.5, 0.05]))] for f in range(3)]
    reps = _run(tr, frames)
    assert reps[-1][0][2] == "drone"


def test_sustained_contrary_evidence_relabels():
    tr = Tracker()
    frames = [[Obs(100 + f, 50, 8, DRONEY)] for f in range(10)]
    frames += [[Obs(110 + f, 50, 8, np.array([0.01, 0.98, 0.01]))] for f in range(60)]
    reps = _run(tr, frames)
    assert reps[-1][0][2] == "bird"


def test_probe_keeps_a_hovering_drone_and_coasting_is_bounded():
    tr = Tracker()
    _run(tr, [[Obs(100 + f, 50, 8, DRONEY)] for f in range(10)])
    tid = tr.tracks[0].id
    # hovering: no motion proposals, only a probe at the predicted position
    reps = _run(tr, [[Obs(109, 50, 8, DRONEY, probe_of=tid)] for _ in range(30)])
    assert all(r and r[0][2] == "drone" and r[0][5] for r in reps)
    # then nothing at all: reported while coasting a few frames, then silent
    reps = _run(tr, [[] for _ in range(10)])
    n_rep = sum(1 for r in reps if r)
    assert 1 <= n_rep <= TrackerConfig().coast_report


def test_other_observations_do_not_start_tracks():
    tr = Tracker()
    _run(tr, [[Obs(10 * f, 10, 5, np.array([0.05, 0.05, 0.9]))] for f in range(10)])
    assert tr.tracks == []


def test_camera_motion_carries_tracks():
    tr = Tracker()
    _run(tr, [[Obs(100.0, 100.0, 8, DRONEY)] for _ in range(6)])
    M = np.array([[1, 0, 30.0], [0, 1, 0], [0, 0, 1]])   # current px -> previous px: scene moved -30
    tr.predict(M)
    assert abs(tr.tracks[0].x - 70.0) < 1e-6


def test_a_static_drone_looking_speck_never_becomes_a_drone():
    # compression flicker on a roof corner: drone-like crops, but it has never been seen flying
    tr = Tracker()
    reps = _run(tr, [[Obs(500.0, 300.0, 5, DRONEY)] for _ in range(60)])
    assert not any(p[2] == "drone" for r in reps for p in r)


def test_a_drone_that_stops_moving_entirely_is_demoted():
    tr = Tracker()
    _run(tr, [[Obs(100 + 2 * f, 50, 8, DRONEY)] for f in range(10)])
    tid = tr.tracks[0].id
    reps = _run(tr, [[Obs(118.0, 50.0, 8, DRONEY, probe_of=tid)] for _ in range(120)])
    assert reps[0] and reps[0][0][2] == "drone"      # a hover right after flight is kept...
    assert not reps[-1]                               # ...but 3 s of zero movement is clutter


def test_detector_evidence_from_labels():
    from realtime.fixedcam.tracker import det_evidence

    assert np.allclose(det_evidence("drone+motion", 0.9), [0.9, 0.05, 0.05])
    assert np.allclose(det_evidence("bird", 0.6), [0.2, 0.6, 0.2])
    assert np.allclose(det_evidence("drone", 1.0), [0.95, 0.025, 0.025])   # never a veto
    assert det_evidence("motion", 0.5) is None


def test_fusion_is_a_weighted_product_of_experts():
    from realtime.fixedcam.tracker import fuse

    p = np.array([0.4, 0.4, 0.2])
    ev = np.array([0.8, 0.1, 0.1])
    assert np.allclose(fuse(p, ev, 0.0), p) and np.allclose(fuse(p, None, 1.0), p)
    q = fuse(p, ev, 1.0)
    assert q.sum() == pytest.approx(1.0) and q[0] > 0.7 and q.argmax() == 0


def test_an_external_drone_opinion_never_overrules_a_bird_the_classifier_sees():
    from realtime.fixedcam.tracker import fuse

    birdy = np.array([0.3, 0.6, 0.1])
    drone_ev, bird_ev = np.array([0.9, 0.05, 0.05]), np.array([0.05, 0.9, 0.05])
    assert fuse(birdy, drone_ev, 1.0).argmax() == 0                        # without the veto it flips
    assert np.allclose(fuse(birdy, drone_ev, 1.0, bird_veto=0.4), birdy)   # with it, the bird stays
    assert fuse(birdy, bird_ev, 1.0, bird_veto=0.4)[1] > birdy[1]          # bird evidence still counts
    unsure = np.array([0.5, 0.2, 0.3])                                     # classifier sees no bird
    assert fuse(unsure, drone_ev, 1.0, bird_veto=0.4)[0] > 0.5


def _jitter_then(n_jitter, extra, seed=0):
    rng = np.random.default_rng(seed)
    pts = [(400 + rng.normal(0, 1.2), 300 + rng.normal(0, 1.2)) for _ in range(n_jitter)]
    return [[Obs(float(x), float(y), 6, DRONEY)] for x, y in pts] + extra


def test_heading_gate_holds_back_a_drone_looking_point_that_jitters_in_place():
    # drone-looking crops that wander a few px at random: no heading, so never reported as a drone
    frames = _jitter_then(0, [[Obs(400 + 3 * np.sin(f), 300 + 3 * np.cos(2.3 * f), 6, DRONEY)] for f in range(80)])
    loose = _run(Tracker(TrackerConfig(smooth_commit=1e9, smooth_keep=1e9, hover_move=0)), frames)
    gated = Tracker(TrackerConfig(smooth_commit=1e9, smooth_keep=1e9, hover_move=0, heading_gate=1.0))
    reps = _run(gated, frames)
    assert any(p[2] == "drone" for r in loose for p in r)            # the old rules let it through
    assert not any(p[2] == "drone" for r in reps for p in r)         # the heading gate does not
    assert gated.candidates()                                          # it is still drawn, grey


def test_heading_gate_passes_a_straight_flier():
    tr = Tracker(TrackerConfig(heading_gate=1.0))
    reps = _run(tr, [[Obs(100 + 3 * f, 50 + 1 * f, 8, DRONEY)] for f in range(40)])
    assert reps[-1] and reps[-1][0][2] == "drone"


def test_a_drone_that_has_flown_may_hover_under_the_flight_rule():
    cfg = TrackerConfig(flight_turn=0.8, flight_speed=1.0, hover_move=0)
    fly = [[Obs(100 + 3 * f, 50, 8, DRONEY)] for f in range(30)]
    rng = np.random.default_rng(1)
    hover = [[Obs(187 + rng.normal(0, 0.8), 50 + rng.normal(0, 0.8), 8, DRONEY)] for _ in range(40)]
    tr = Tracker(cfg)
    reps = _run(tr, fly + hover)
    assert tr.tracks[0].flown and reps[-1] and reps[-1][0][2] == "drone"
    never_flew = Tracker(TrackerConfig(flight_turn=0.8, flight_speed=1.0, hover_move=0))
    reps = _run(never_flew, hover + hover)
    assert not any(p[2] == "drone" for r in reps for p in r)


def test_a_track_hopping_between_clutter_blobs_is_not_a_drone():
    # drone-looking crops, lots of ground covered - but a random walk, not a flight path
    rng = np.random.default_rng(0)
    pts = np.cumsum(rng.normal(0, 9, size=(80, 2)), axis=0) + 400
    tr = Tracker()
    reps = _run(tr, [[Obs(float(x), float(y), 6, DRONEY)] for x, y in pts])
    assert not any(p[2] == "drone" for r in reps for p in r)
