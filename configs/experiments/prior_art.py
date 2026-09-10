"""Temporal-YOLOv8, reimplemented from its paper as a paired comparison arm.

van Leeuwen, Fokkinga, Huizinga, Baan and Heslinga (TNO), "Toward Versatile Small Object
Detection with Temporal-YOLOv8", Sensors 24(22):7387, 2024, doi:10.3390/s24227387.

It is the closest published relative of this project's core idea -- grayscale frames from three
moments in the three input channels of a stock YOLOv8 -- and it was published two years earlier.
The paper links no code or weights, and every number it reports is measured on Nano-VID, its
in-house dataset; two public datasets, VIRAT-Ground and VisDrone-VID, appear only as extra
training data. So this is a reimplementation, trained on this repository's splits and scored by
its evaluator -- the way YOLOMG is compared, and for the same reason: a number measured on
another dataset cannot be placed beside one measured here.

Every statement below about the paper was checked against its full text (Europe PMC,
PMC11598073) on 2026-09-10, and the quotations are the paper's own.

SPECIFIED BY THE PAPER, AND REPRODUCED
  * The window: "Assuming a 30-frames-per-second (FPS) source video, around 15 frames are
    sampled before and after the current frame" -- taps t-15, t, t+15, grayscale, in the three
    channels, one fixed frame offset on every clip, as the paired SpeckLock arm uses. Every clip
    either arm trains on runs at 28.0-29.97 fps (ARD-MAV 29.77-29.97, NPS 28.0-29.97; the local
    videos 30), so there 15 frames is 0.50-0.54 s. The window is NON-CAUSAL: it reads about
    half a second of the future, which a deployed interceptor cannot.
  * No camera-motion compensation. Its videos are "only stationary-recorded", and the paper
    expects that from moving platforms "the current setup for temporal YOLO will probably not
    perform as well".
  * "Pretrained weights from the m variant of the YOLOv8 model": here Ultralytics' released
    yolov8m.pt, the public YOLOv8m weights (trained on COCO), with the standard 3-scale head.
  * 70 epochs under Ultralytics' learning-rate schedule.

AN INTERPRETATION, NOT A SPECIFICATION
  * The learning rate. The paper uses "the default Adam optimizer" and cites Kingma and Ba, whose
    published default step size is 1e-3, but prints no value. Adam at lr0 = 1e-3 is that default,
    and is the value Ultralytics' own configuration documents for Adam ("SGD=1e-2,
    Adam/AdamW=1e-3"); Ultralytics' generic lr0 of 0.01 is its SGD value.

NOT REPRODUCED, AND WHY
  * Its 15-pixel box enlargement ("bounding boxes with a width or height below 15 pixels are
    scaled" to at least 15), which its own ablation credits with 0.166 mAP -- under its IoU >= 0.01
    metric. Here both arms are scored at IoU >= 0.5 against true extents, and a 15 px box centred
    on a target narrower than about 10.6 px cannot reach 0.5 (a 6 px target: 36/225 = 0.16).
    Enlarged labels would make ARD-MAV's <8 px and 8-10 px bins unwinnable by construction, so
    both arms train on true extents (min_side 0).
  * Its "balanced mosaicking" -- crops of 420, 750 and 1920 px, downscaling floored at 10x10 px
    objects, a box blur before each downscale -- and its CLAHE (p = 0.1) and 10 % scale jitter.
    Both arms use the paired arm's augmentation: photometric augmentation off, standard mosaic
    on 640 px tiles at native resolution.
  * Its metric: IoU >= 0.01, with several detections inside one annotation all counted correct.
    Both arms are scored by this repository's protocol for each dataset (IoU >= 0.5 on the
    benchmarks, centre distance on the local videos).

MATCHED TO THE PAIRED SPECKLOCK ARM, because the paper does not specify it
  * 640 px tiles, the same stride, splits and labels, batch 8, patience 25.

Two of the ten NPS TEST clips run at 59.9 fps (named in the build log and in
tools/make_summary.py). On them 15 frames is 0.25 s, as SpeckLock's 6 frames is 0.10 s: both
arms meet half the window they were trained on. Declared before any scorecard of this arm
existed: they are scored like every other clip, and SUMMARY.md also prints the NPS difference
with those two clips left out -- descriptively, without a verdict, so the Holm family stays the
one question it was declared as.
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


# ------------------------------------------------------------ the project's own 8 px task
#: Temporal-YOLOv8 on 07_05 -> 10_06, the task on which every other arm is also fine-tuned
#: (cluster/local_finetune.sbatch): initialised from ITS OWN NPS-trained checkpoint -- the job
#: passes --weights work/runs_tyolov8/tyolov8_nps-s<seed>/weights/best.pt -- exactly as ours
#: and YOLOMG start from theirs, so no arm borrows another's prior. The 60 epochs, tiling and
#: labels are the paired local arms' (configs/experiments/local_video.py); network, window and
#: stabiliser (none) are the paper's, and the optimizer is the benchmark arms' Adam at 1e-3 --
#: the interpretation the module docstring states, not a value the paper prints.
TYOLOV8_LOCAL_AB = ExperimentConfig(
    name="tyolov8_local_ab",
    datasets=("local:07_05",),
    data="work/ext_datasets/local_yolo_temporal_dt15_centred_staboff/data.yaml",
    protocol_key="specklock-centre",
    build_command=("PYTHONPATH=. python tools/make_dataset_external.py --task local-temporal "
                   "--tile 640 --stride-train 1 --stride-val 4 --min-side 0 "
                   f"--dt {TYOLOV8_DT} --taps centred --stab off"),
    notes="Temporal-YOLOv8 on the project's own task, fine-tuned from tyolov8_nps-s<seed>, "
          "paired against temporal_local_ab and yolomg_local_ft on the held-out 10_06 flight.",
    **dict(_TYOLOV8, epochs=60, tags=("prior-art", "temporal-yolov8", "local")),
)
