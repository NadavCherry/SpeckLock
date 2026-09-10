"""Both temporal builders hand their window and stabiliser to the tile extractor.

A builder that names its output directory _centred_staboff but calls the extractor with the
defaults would train the prior-art arm on causal, stabilised stacks under a centred label --
and nothing downstream could tell. The first centred build logged "taps t-30, t-15, t" from a
hardcoded print while its tiles were centred; that line is why the pass-through, and the
line, are pinned here rather than trusted.
"""
from __future__ import annotations

import pytest

pytest.importorskip("cv2")
M = pytest.importorskip("tools.make_dataset_external")


@pytest.fixture
def calls(tmp_path, monkeypatch):
    seen: list[dict] = []

    def fake_extractor(video, boxes, chosen, img_dir, lbl_dir, prefix, min_side, **kw):
        seen.append(kw)
        return 0, 0

    monkeypatch.setattr(M, "extract_yolo_tiled_temporal", fake_extractor)
    monkeypatch.setattr(M, "OUT_ROOT", tmp_path)
    monkeypatch.setattr(M, "write_data_yaml", lambda root, build=None: None)
    monkeypatch.setattr(M, "_nps_video", lambda clip: tmp_path / f"{clip}.mov")
    monkeypatch.setattr(M, "parse_nps_dogfight", lambda clip: {0: [(0, 0, 4, 4)]})
    monkeypatch.setattr(M, "_ard_all", lambda: ["phantom02"])
    monkeypatch.setattr(M, "parse_ardmav", lambda vid: {0: [(0, 0, 4, 4)]})
    return seen


def test_nps_builder_passes_the_centred_window(calls):
    M.build_nps_tiled(1, 1, 0.0, temporal=True, dt=15, taps="centred", stab_mode="off")
    assert len(calls) == len(M.NPS_TRAIN) + len(M.NPS_VAL)
    assert all((kw["dt"], kw["taps"], kw["stab_mode"]) == (15, "centred", "off") for kw in calls)


def test_ardmav_builder_passes_the_centred_window(calls, capsys):
    M.build_ardmav_temporal_tiled(1, 1, 0.0, dt=15, taps="centred", stab_mode="off")
    assert len(calls) == 1 + len(M.ARD_VAL_IDS)
    assert all((kw["dt"], kw["taps"], kw["stab_mode"]) == (15, "centred", "off") for kw in calls)
    assert "taps t-15, t, t+15" in capsys.readouterr().out


def test_the_shipped_defaults_are_unchanged(calls, capsys):
    M.build_nps_tiled(1, 1, 0.0, temporal=True)
    M.build_ardmav_temporal_tiled(1, 1, 0.0)
    assert all((kw["dt"], kw["taps"], kw["stab_mode"]) == (M.TEMPORAL_DT, "causal", "translation")
               for kw in calls)
    dt = M.TEMPORAL_DT
    assert f"taps t-{2 * dt}, t-{dt}, t" in capsys.readouterr().out
