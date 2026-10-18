"""Appearance-over-time track features and the JSON track classifier (realtime.fixedcam.trackfeat).

numpy only, like the tracker: CI imports it without torch."""

from __future__ import annotations

import json

import numpy as np
import pytest

from realtime.fixedcam.trackfeat import FEATURES, OBS_FIELDS, TrackClassifier, band_power, track_features
from realtime.fixedcam.tracker import Obs, Tracker, TrackerConfig


def test_band_power_finds_a_wing_beat_and_ignores_a_steady_target():
    t = np.arange(32)
    flap = 1.0 + 0.5 * np.sin(2 * np.pi * 5.0 * t / 25.0)          # 5 Hz at 25 fps
    steady = 1.0 + 0.01 * t                                          # a slow drift, removed as trend
    assert band_power(flap) > 0.8
    assert band_power(steady) < 0.2
    assert band_power(flap[:8]) == 0.0                               # too short to say


def _appear(n, contrast):
    return [(a, np.array([0.8, 0.1, 0.1, -1.0, c, 0.1, 0.2, 0.3, 0.0])) for a, c in zip(range(n), contrast)]


def test_track_features_shape_and_meaning():
    n = 32
    hist = [(a, 100.0 + 3 * a, 50.0) for a in range(n)]
    f = track_features(_appear(n, np.ones(n)), hist, n_obs=n, age=n - 1, size=8.0)
    assert f.shape == (len(FEATURES),)
    d = dict(zip(FEATURES, f))
    assert d["p_drone_mean"] == pytest.approx(0.8) and d["sign_consistency"] == 1.0
    assert d["speed"] == pytest.approx(3.0) and d["turn"] == pytest.approx(0.0) and d["straightness"] == pytest.approx(1.0)
    flapping = dict(zip(FEATURES, track_features(
        _appear(n, 1 + 0.5 * np.sin(2 * np.pi * 5 * np.arange(n) / 25)), hist, n, n - 1, 8.0)))
    assert flapping["flicker_contrast"] > 0.8 > d["flicker_contrast"]
    assert len(OBS_FIELDS) == 9


def _model(tmp_path, bias):
    m = {"features": list(FEATURES), "mean": [0.0] * len(FEATURES), "std": [1.0] * len(FEATURES),
         "w": [0.0] * len(FEATURES), "b": bias}
    p = tmp_path / "clf.json"
    p.write_text(json.dumps(m), encoding="utf-8")
    return str(p)


def test_classifier_json_logistic_and_mlp(tmp_path):
    assert TrackClassifier(_model(tmp_path, 2.0))(np.zeros(len(FEATURES))) == pytest.approx(1 / (1 + np.exp(-2)))
    mlp = {"features": list(FEATURES), "mean": [0.0] * len(FEATURES), "std": [1.0] * len(FEATURES),
           "W1": [[1.0] + [0.0] * (len(FEATURES) - 1)], "b1": [0.0], "w2": [1.0], "b2": 0.0}
    x = np.zeros(len(FEATURES)); x[0] = 3.0
    assert TrackClassifier(mlp)(x) == pytest.approx(1 / (1 + np.exp(-3)))
    with pytest.raises(ValueError):
        TrackClassifier({**mlp, "features": ["other"]})


def test_a_track_the_classifier_rejects_is_shown_grey_not_as_a_drone(tmp_path):
    droney = np.array([0.8, 0.15, 0.05])
    frames = [[Obs(100 + 3 * f, 50, 8, droney, feat=np.array([0.8, 0.15, 0.05, -1, 1, 0.1, 0.2, 0.2, 0]))]
              for f in range(30)]
    for bias, reported in ((-5.0, False), (5.0, True)):
        tr = Tracker(TrackerConfig(track_clf=_model(tmp_path, bias)))
        reps = []
        for obs in frames:
            tr.predict(None)
            tr.update(obs)
            reps.append(tr.report())
        assert any(p[2] == "drone" for r in reps for p in r) is reported
        if not reported:
            assert tr.candidates()
