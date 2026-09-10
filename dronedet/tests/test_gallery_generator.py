"""The gallery page is generated, so the generator -- not the page -- is the source of truth.

An earlier round corrected docs/gallery.html by hand and left tools/make_gallery.py alone. Running
the generator the README tells readers to run then put three retracted claims back on the page:
"unseen test video" for a development video, "AP / F1 = 1.000 . zero false positives", and
"~74 fps end to end", a figure that did not reproduce. These tests hold the two together.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_the_committed_gallery_is_exactly_what_the_generator_renders():
    G = pytest.importorskip("tools.make_gallery")
    manifest = json.loads((ROOT / "docs/media/showcase.json").read_text(encoding="utf-8"))
    assert G.render(manifest) == (ROOT / "docs/gallery.html").read_text(encoding="utf-8")


def test_no_retracted_claim_survives_in_the_generator():
    src = (ROOT / "tools/make_gallery.py").read_text(encoding="utf-8")
    body = src.split('dict(mp4="media/10_06_baseline_vs_pcmax_vs_edgert.mp4"', 1)[1]
    for retracted in ("unseen test video", "AP / F1 = 1.000", "zero false positives",
                      "~74 fps", "3&#8211;14 px", "nothing below the treeline"):
        assert retracted not in body, retracted
