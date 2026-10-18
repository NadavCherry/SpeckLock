"""tools/make_dataset_appearance.py: crop geometry, weak-label run finding and GT point selection."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools import make_dataset_appearance as mda  # noqa: E402


def _square(h=200, w=300, cx=120, cy=80, s=10):
    g = np.full((h, w), 100, np.uint8)
    g[cy - s // 2:cy + s // 2, cx - s // 2:cx + s // 2] = 220
    return g


def test_crop_is_centred_and_scale_normalised():
    for s in (6, 10, 30):
        g = _square(cx=150, cy=100, s=s)
        c = mda.crop_patch(g, 150 - 0.5, 100 - 0.5, s)
        assert c.shape == (mda.STORE, mda.STORE) and c.dtype == np.uint8
        bright = np.argwhere(c > 160)
        centre = bright.mean(0)
        assert np.all(np.abs(centre - (mda.STORE - 1) / 2) <= 1.0)
        span = bright[:, 1].max() - bright[:, 1].min() + 1
        assert abs(span - mda.STORE / mda.CONTEXT) <= 3      # the object fills 1 / CONTEXT of the stored crop


def test_crop_at_the_border_replicates_instead_of_failing():
    g = _square(cx=4, cy=4, s=6)
    c = mda.crop_patch(g, 3.5, 3.5, 6)
    assert c.shape == (mda.STORE, mda.STORE)


def test_net_window_is_the_central_2p5_size_view():
    c = np.zeros((mda.STORE, mda.STORE), np.uint8)
    c[16:48, 16:48] = 200
    w = mda.net_window(c)
    assert w.shape == (mda.NET, mda.NET) and w.dtype == np.float32
    assert w[mda.NET // 2, mda.NET // 2] > 150 and w[0, 0] < 50
    assert abs(mda.WINDOW / mda.STORE * mda.CONTEXT - 2.5) < 1e-9


def test_scale_plane_is_zero_at_16_and_clamped():
    assert mda.scale_plane_value(16) == 0.0
    assert abs(mda.scale_plane_value(64) - 1.0) < 1e-9
    assert mda.scale_plane_value(500) == mda.scale_plane_value(64)
    assert mda.scale_plane_value(1) == mda.scale_plane_value(4)


def test_longest_run_breaks_on_jumps_and_gaps():
    d = [(1, 10 + f, 10, 5) for f in range(20)] + [None] + [(1, 200 + f, 10, 5) for f in range(30)]
    run = mda.longest_run(d, min_len=16)
    assert len(run) == 30 and run[0][1] == 200
    assert mda.longest_run(d[:10], min_len=16) == []


def test_salient_blob_finds_the_dark_speck():
    g = np.full((256, 320), 140, np.uint8)              # the speck is < 0.1 % of the pixels (p99.9 threshold)
    g[60:66, 90:96] = 40
    sc, cx, cy, s = mda.salient_blob(g)
    assert abs(cx - 92.5) < 1 and abs(cy - 62.5) < 1 and s == 6


def test_gt_points_maps_names_and_subsamples(tmp_path):
    gt = {"video": "v.mp4", "meta": {"exclude_frames": [4]},
          "objects": {"far": {"ignore": False, "frames": {str(f): [10.0, 20.0, 8.0, 6.0] for f in range(6)}},
                      "bird#2": {"ignore": True, "frames": {"1": [5.0, 5.0, 6.0, 6.0]}},
                      "near": {"ignore": True, "frames": {"0": [1.0, 1.0, 50.0, 50.0]}}}}
    p = tmp_path / "g.json"
    p.write_text(json.dumps(gt), encoding="utf-8")
    pts = mda.gt_points(p, mda.parse_map("far=drone,bird*=bird"), {"drone": 2})
    assert [(f, lab) for f, _, _, _, lab in pts] == [(0, 0), (1, 1), (2, 0)]   # frame 4 is excluded, near unmapped
    assert pts[0][3] == 8.0


def test_builder_round_trip(tmp_path):
    b = mda.Builder()
    b.add(np.zeros((mda.STORE, mda.STORE)), 1, 12.5, 7, "src", "grp", 1, 42)
    counts = b.save(tmp_path / "x.npz")
    z = np.load(tmp_path / "x.npz")
    assert counts == {"drone": 0, "bird": 1, "other": 0}
    assert z["crops"].shape == (1, mda.STORE, mda.STORE) and int(z["key"][0]) == 42 and str(z["group"][0]) == "grp"
