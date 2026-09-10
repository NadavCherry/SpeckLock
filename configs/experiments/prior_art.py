"""Temporal-YOLOv8, reimplemented from its paper as a paired comparison arm.

van Leeuwen, Fokkinga, Huizinga, Baan and Heslinga (TNO), "Toward Versatile Small Object
Detection with Temporal-YOLOv8", Sensors 24(22):7387, 2024.

It is the closest published relative of this project's core idea -- three moments of video in
the three input channels of a stock YOLOv8 -- and it was published two years earlier. Neither
code nor weights are released and its Nano-VID dataset is proprietary, so this is a
reimplementation of what the paper specifies, trained on this repository's splits and scored by
its evaluator: the way YOLOMG is compared, and for the same reason. A number measured on a
private dataset cannot be placed beside one measured here.

REPRODUCED, because the paper specifies it
  * Three grayscale frames in the RGB channels, "around 15 frames ... sampled before and after
    the current frame" at 30 fps: taps t-15, t, t+15, one fixed frame offset on every clip as
    the paired SpeckLock arm uses. ARD-MAV runs at 29.77-29.97 fps, NPS's training and
    validation clips at 28.0-29.97 and the local videos at 30, so 15 frames is 0.50-0.54 s --
    the paper's "around" half second (cluster/tyolov8_build.sbatch prints every clip's rate).
    The window is NON-CAUSAL: it reads about half a second of the future, which a deployed
    interceptor cannot.
  * No camera-motion compensation. Its cameras are stationary, and the paper itself expects the
    method to do worse from a moving platform.
  * YOLOv8m from COCO-pretrained weights, with the standard three-scale head (no P2).
  * 70 epochs with the "default Adam optimizer": Adam at its default learning rate, 1e-3.

MATCHED TO THE PAIRED SPECKLOCK ARM -- because the paper does not specify it, or this pipeline
cannot reproduce it -- and held equal so it is not a second variable
  * 640 px tiles, the same stride, the same labels (min_side 0: true extents).
  * Augmentation NO_PHOTOMETRIC_AUG. The paper's CLAHE (p = 0.1), 10 % scale jitter and
    "balanced mosaicking" of 420 / 750 / 1920 px crops are NOT reproduced.
  * Batch 8 and patience 25; the paper states neither.
  * Scoring by this repository's protocol for each dataset. The paper's own metric (IoU >= 0.01,
    several detections per object accepted) is used for neither arm.
"""

from __future__ import annotations

from .base import NO_PHOTOMETRIC_AUG, ExperimentConfig, VramReference

#: The paper's half second at 30 fps. Not SpeckLock's TEMPORAL_DT.
TYOLOV8_DT = 15

#: NOT measured. YOLOv8m at 640 px, batch 8, stock three-scale head -- sized above
#: yolov8s-p2's measured 3.54 GiB at batch 8 (a smaller network, but with a costly stride-4
#: head). tools/train.py records torch.cuda.max_memory_allocated() into RESULT.json, so the
#: first run replaces this with a fact.
YOLOV8M_ESTIMATE = VramReference(
    gib=5.0, batch=8, imgsz=640, measured=False,
    source="ESTIMATE, yolov8m @ 640 px batch 8, never run in this repo; replace from RESULT.json")

# The shipped temporal arms' flags verbatim (configs/experiments/ardmav.py _TEMPORAL_BUILD),
# plus the window and the stabiliser -- explicit, so a changed default cannot drift the pair.
_BUILD = ("PYTHONPATH=. python tools/make_dataset_external.py --task {task} --tile 640 "
          "--stride-train 4 --stride-val 10 --min-side 0 "
          f"--dt {TYOLOV8_DT} --taps centred --stab off")

_TYOLOV8 = dict(
    model_cfg="yolov8m.yaml",
    weights="yolov8m.pt",
    imgsz=640, batch=8, epochs=70, seed=0,
    optimizer="Adam", lr0=0.001,
    tile_px=640, min_side=0.0,
    temporal_stack=True, stack_dt=TYOLOV8_DT,
    nwd=False,
    aug=NO_PHOTOMETRIC_AUG,
    expected_strides=(8, 16, 32),
    vram_ref=YOLOV8M_ESTIMATE,
    tags=("prior-art", "temporal-yolov8"),
)

TYOLOV8_ARDMAV = ExperimentConfig(
    name="tyolov8_ardmav",
    datasets=("ardmav",),
    data="work/ext_datasets/ardmav_yolo_temporal_dt15_centred_staboff/data.yaml",
    protocol_key="ardmav-official",
    build_command=_BUILD.format(task="ardmav-temporal-tiled"),
    notes="Temporal-YOLOv8, reimplemented (see the module docstring), paired against "
          "temporal_ardmav at 100 epochs on ARD-MAV's official 15-video test split.",
    **_TYOLOV8,
)

TYOLOV8_NPS = ExperimentConfig(
    name="tyolov8_nps",
    datasets=("nps",),
    data="work/ext_datasets/nps_yolo_temporal_dt15_centred_staboff/data.yaml",
    protocol_key="nps-official",
    build_command=_BUILD.format(task="nps-temporal-tiled"),
    notes="Temporal-YOLOv8, reimplemented (see the module docstring), paired against "
          "temporal_nps at 100 epochs on the video-disjoint NPS test clips.",
    **_TYOLOV8,
)
