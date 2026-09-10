"""Temporal-YOLOv8's local arm: paired with the shipped local arms wherever the paper is silent,
with the paper wherever it is not, and pointed at the directory its builder actually writes.

A config whose `data` names a directory the builder never produces fails at train time; one
that names a SHIPPED directory trains the arm on the wrong representation and fails nowhere.
"""
from __future__ import annotations

import pytest

from configs.experiments.local_video import TEMPORAL_LOCAL_AB
from configs.experiments.prior_art import TYOLOV8_DT, TYOLOV8_LOCAL_AB, TYOLOV8_NPS


def test_the_local_arm_reads_the_directory_its_builder_writes():
    M = pytest.importorskip("tools.make_dataset_external")
    name = M._temporal_name("local_yolo_temporal", True, TYOLOV8_DT, M._variant("centred", "off"))
    assert TYOLOV8_LOCAL_AB.data == f"work/ext_datasets/{name}/data.yaml"
    assert name != "local_yolo_temporal"


def test_matched_to_the_paired_local_arms_where_the_paper_is_silent():
    t, ours = TYOLOV8_LOCAL_AB, TEMPORAL_LOCAL_AB
    fields = ("epochs", "imgsz", "batch", "tile_px", "min_side", "protocol_key", "datasets",
              "classes", "patience", "aug")
    assert {f: getattr(t, f) for f in fields} == {f: getattr(ours, f) for f in fields}


def test_the_paper_where_it_is_not():
    t, bench = TYOLOV8_LOCAL_AB, TYOLOV8_NPS
    fields = ("model_cfg", "optimizer", "lr0", "stack_dt", "temporal_stack", "nwd",
              "expected_strides")
    assert {f: getattr(t, f) for f in fields} == {f: getattr(bench, f) for f in fields}
    for flag in ("--task local-temporal", f"--dt {TYOLOV8_DT}", "--taps centred", "--stab off",
                 "--stride-train 1", "--min-side 0"):
        assert flag in t.build_command


def test_registered_but_kept_out_of_the_benchmark_group():
    """Registered, so tools/train.py can run it. Kept out of GROUPS["prior_art"]: a group run
    cannot pass the per-seed NPS initialisation this arm is defined by, and would silently
    start it from COCO weights -- a different arm under the same name."""
    from configs.experiments import EXPERIMENTS, GROUPS

    assert EXPERIMENTS["tyolov8_local_ab"] is TYOLOV8_LOCAL_AB
    assert "tyolov8_local_ab" not in GROUPS["prior_art"]
