# The closest prior art: Temporal-YOLOv8, reimplemented and compared

## Why this exists

Temporal-YOLOv8 (van Leeuwen, Fokkinga, Huizinga, Baan and Heslinga, TNO; *Sensors*
24(22):7387, 2024, [doi:10.3390/s24227387](https://doi.org/10.3390/s24227387)) puts
grayscale frames from three moments into the three input channels of a stock YOLOv8 --
the same basic move as this project, published two years earlier. It is the comparison a
reader is entitled to ask for first: is anything here more than that idea, applied twice?

It is prior art, not a leaderboard entry. Every number the paper reports is measured on
Nano-VID, its in-house dataset (VIRAT-Ground and VisDrone-VID appear only as extra
training data), and the paper links no code or weights. So it is reimplemented from the
paper, trained on this repository's splits, and scored by this repository's evaluator --
the way YOLOMG is compared, and for the same reason: a number measured on another dataset
cannot be placed beside one measured here.

Every claim below about the paper's own text was checked against its full text (Europe
PMC, PMC11598073); quotations are its own. `configs/experiments/prior_art.py` is the
executable specification this report is checked against.

## How the two methods differ

| | SpeckLock (paired arm) | Temporal-YOLOv8, as reimplemented |
|---|---|---|
| window | t-12, t-6, t -- **causal** | t-15, t, t+15 -- **reads about 0.5 s of the future** |
| camera motion | ego-stabilised (global translation, phase correlation) | none -- the paper's videos are "only stationary-recorded" |
| network | YOLOv8s + stride-4 P2 head, NWD loss | YOLOv8m, standard three-scale head |
| training | 100 epochs, Ultralytics' default optimiser | 70 epochs, Adam at 1e-3 |

The non-causal window is an advantage Temporal-YOLOv8 carries into every number below and
could not carry into deployment: a detector that must decide at frame t does not have
frame t+15. Stabilisation runs the other way: the paper itself expects that from moving
platforms "the current setup for temporal YOLO will probably not perform as well".

## Specified, interpreted, not reproduced

**Specified by the paper and reproduced.** The window -- "Assuming a 30-frames-per-second
(FPS) source video, around 15 frames are sampled before and after the current frame"; no
camera-motion compensation; "pretrained weights from the m variant of the YOLOv8 model"
(Ultralytics' released `yolov8m.pt`); 70 epochs under Ultralytics' learning-rate schedule.

**An interpretation.** The learning rate: the paper uses "the default Adam optimizer",
citing Kingma and Ba, whose default step size is 1e-3, but prints no value. Adam at 1e-3
is that default, and is also the value Ultralytics documents for Adam.

**Not reproduced, and why.**

- *The 15-pixel box enlargement* ("bounding boxes with a width or height below 15 pixels
  are scaled" to at least 15), which the paper's own ablation credits with 0.166 mAP --
  under its IoU >= 0.01 metric. Here both arms are scored at IoU >= 0.5 against true
  extents, and a 15 px box centred on a target narrower than about 10.6 px cannot reach
  0.5: a 6 px target gives 36/225 = 0.16. Enlarged labels would make ARD-MAV's two
  smallest size bins unwinnable by construction, so both arms train on true extents.
- *Balanced mosaicking* -- crops of 420, 750 and 1920 px, downscaling floored at 10x10 px
  objects, a box blur before each downscale -- and CLAHE (p = 0.1) and 10 % scale jitter.
  Both arms use the paired arm's augmentation: no photometric augmentation, standard
  mosaic on 640 px tiles at native resolution.
- *Its metric* -- IoU >= 0.01, with several detections inside one annotation all counted
  correct. Both arms are scored by this repository's protocol for each dataset.

**Matched to the paired arm, because the paper does not specify it.** 640 px tiles,
stride, splits, labels, batch 8, patience 25.

Each of the not-reproduced items could move Temporal-YOLOv8's numbers, in either
direction. What this comparison measures is *Temporal-YOLOv8's representation and
network, trained and scored under this project's protocol* -- not the paper's full
recipe.

## Frame rates, and what "15 frames" means

Every clip either arm trains on runs at 28.0-29.97 fps (ARD-MAV 29.97; NPS 28.0-29.97;
one ARD-MAV test video runs at 29.77), so there 15 frames is 0.50-0.54 s -- the paper's
"around" half second ([frame rates](../../work/reports/prior_art/frame_rates.md)). Two
NPS test clips, Clip_049 and Clip_050, run at 59.9 fps: on them 15 frames is 0.25 s, as
SpeckLock's 6 frames is 0.10 s -- both arms meet half the window they were trained on.
Both are scored on every clip with fixed frame offsets; the table below also gives the
NPS comparison without those two clips, as effect sizes only.

## Results

### ARD-MAV, official 15-video test split

| arm | budget | AP mean (3 seeds) |
|---|---|---|
| **ours** temporal | 100 ep | **0.809** +/- 0.005  (0.806, 0.816, 0.807) |
| Temporal-YOLOv8, reimplemented | 70 ep | **0.807** +/- 0.001  (0.807, 0.806, 0.808) |
| YOLOMG (competitor) | 100 ep | 0.834 |

Essentially tied on point estimates. Paired, seed-matched, Holm over this table (its own
family, declared before any of these scorecards existed --
`tools/make_summary.py:PRIOR_ART_TESTS`):

| comparison | seed | d AP | 95% CI | p perm | p perm, Holm | verdict |
|---|---|---|---|---|---|---|
| ours temporal 100 ep - Temporal-YOLOv8 70 ep | 0 | -0.001 | [-0.052, +0.062] | 0.9695 | 1.0000 | no difference |
| ours temporal 100 ep - Temporal-YOLOv8 70 ep | 1 | +0.010 | [-0.037, +0.072] | 0.7866 | 1.0000 | no difference |
| ours temporal 100 ep - Temporal-YOLOv8 70 ep | 2 | -0.001 | [-0.045, +0.058] | 0.9660 | 1.0000 | no difference |

No difference on any seed, before or after Holm. `d AP` is within +/-0.010 of zero on
every seed -- the two arms are not just statistically indistinguishable here, they are
numerically close.

### NPS-Drones, video-disjoint test

| arm | budget | AP mean (3 seeds) |
|---|---|---|
| **ours** temporal | 100 ep | **0.487** +/- 0.055  (0.482, 0.544, 0.435) |
| Temporal-YOLOv8, reimplemented | 70 ep | **0.531** +/- 0.012  (0.543, 0.532, 0.519) |
| YOLOMG (competitor) | 100 ep | 0.527 |

Temporal-YOLOv8 leads on the point estimate (0.531 vs 0.487) and is
markedly more stable across seeds (+/-0.012 against ours' +/-0.055: one of
our three seeds runs unusually high, one unusually low). Paired, seed-matched, Holm over
this table:

| comparison | seed | d AP | 95% CI | p boot | p perm | p perm, Holm | verdict |
|---|---|---|---|---|---|---|---|
| ours temporal 100 ep - Temporal-YOLOv8 70 ep | 0 | -0.061 | [-0.106, -0.015] | 0.0070 | 0.5662 | 1.0000 | no difference |
| ours temporal 100 ep - Temporal-YOLOv8 70 ep | 1 | +0.012 | [-0.057, +0.061] | 0.6800 | 0.8866 | 1.0000 | no difference |
| ours temporal 100 ep - Temporal-YOLOv8 70 ep | 2 | -0.084 | [-0.150, -0.034] | 0.0000 | 0.2129 | 0.6387 | no difference |

Two of three seeds show ours numerically lower, with the bootstrap interval alone
excluding zero on both -- but the permutation test does not agree on either (p = 0.57,
0.21), and under this project's rule a difference is only called when both tests agree.
Neither survives Holm. **No difference, on the two-test rule this project holds every
other comparison to.**

#### Without the two 59.9 fps test clips -- descriptive, no verdict

Declared before any Temporal-YOLOv8 scorecard existed
(`tools/make_summary.py:NPS_60FPS_TEST`); no test is run on this subset, so the Holm
family above stays the one question it was declared as.

| comparison | seed | sequences | d AP |
|---|---|---|---|
| 0 | 8 | -0.078 |
| 1 | 8 | -0.024 |
| 2 | 8 | -0.127 |

The direction does not flip and, if anything, the gap widens slightly without the two odd
clips -- there is no sign that Temporal-YOLOv8's NPS lead depends on them.

### By target size

Each dataset is its own Holm family, added after the overall result above was known, on
the `mission` bins fixed long before ([ARD-MAV](../../work/reports/size_curve/vs_tyolov8/ardmav_mission.md)
· [NPS](../../work/reports/size_curve/vs_tyolov8/nps_mission.md)). **†** marks a bin with
at least one significant seed after Holm.

**ARD-MAV**

| bin | n | ours | Temporal-YOLOv8 | d AP |
|---|---|---|---|---|
| <8 px | 5677 | 0.503 ± 0.011 | 0.258 ± 0.001 | +0.245 |
| 8-10 px | 5055 | 0.602 ± 0.007 | 0.514 ± 0.019 | +0.088 |
| 10-16 px | 7529 | 0.732 ± 0.013 | 0.827 ± 0.005 | -0.096 |
| 16-25 px | 4731 | 0.729 ± 0.029 | 0.888 ± 0.003 | -0.159 |
| >25 px | 5168 | 0.739 ± 0.023 | 0.938 ± 0.002 | -0.199 **†** |

**NPS** (no target under 10 px in this dataset; `*` = fewer than 50 GT instances, treat as
underpowered)

| bin | n | ours | Temporal-YOLOv8 | d AP |
|---|---|---|---|---|
| 10-16 px | 6700 | 0.394 ± 0.052 | 0.405 ± 0.018 | -0.010 |
| 16-25 px | 2396 | 0.265 ± 0.055 | 0.380 ± 0.015 | -0.115 |
| >25 px* | 28 | 0.005 ± 0.001 | 0.077 ± 0.101 | -0.072 |

Every bin below 25 px, on both benchmarks, shows no difference under the two-test rule,
before or after Holm -- the same picture as the overall result. **Above 25 px on ARD-MAV,
Temporal-YOLOv8 leads, significantly on 2 of 3 seeds after Holm:**

| seed | d AP | 95% CI | p perm | p perm, Holm | verdict |
|---|---|---|---|---|---|
| 0 | -0.221 | [-0.494, -0.089] | 0.0010 | 0.0150 | **significant, favours Temporal-YOLOv8** |
| 1 | -0.178 | [-0.424, -0.065] | 0.0010 | 0.0150 | **significant, favours Temporal-YOLOv8** |
| 2 | -0.199 | [-0.491, -0.068] | 0.0050 | 0.0649 | not significant |

This is the one place in the whole comparison where a difference survives correction, and
it favours Temporal-YOLOv8 -- the same size regime where YOLOMG also leads our arm (README
§4, §6). Nothing run here says why large targets favour the standard three-scale head both
competitors use over this project's stride-4 P2 head; it is named, not explained.

### The project's own 8 px task

Temporal-YOLOv8 fine-tuned from its own NPS checkpoint, as every other arm here is from
its own -- no arm borrows another's prior. Scored on the held-out `10_06` flight two
ways: against the single frame, in the same four-arm format as the shipped local reports
([seed 0](../../work/reports/local_ft_tyolov8_10_06_seed0.md) ·
[1](../../work/reports/local_ft_tyolov8_10_06_seed1.md) ·
[2](../../work/reports/local_ft_tyolov8_10_06_seed2.md)); and directly against our
temporal arm, with Temporal-YOLOv8 as the baseline
([seed 0](../../work/reports/local_ft_ours_vs_tyolov8_10_06_seed0.md) ·
[1](../../work/reports/local_ft_ours_vs_tyolov8_10_06_seed1.md) ·
[2](../../work/reports/local_ft_ours_vs_tyolov8_10_06_seed2.md)). One flight, so the
interval is a moving-block bootstrap within it (stability across the flight's segments,
not generalisation to another flight):

| seed | Temporal-YOLOv8 AP | ours AP | d AP | 95% CI | p | verdict |
|---|---|---|---|---|---|---|
| 0 | 0.698 | 0.834 | +0.137 | [+0.010, +0.296] | 0.0138 | better |
| 1 | 0.881 | 0.914 | +0.032 | [-0.022, +0.102] | 0.2818 | no difference |
| 2 | 0.848 | 0.773 | -0.075 | [-0.206, +0.060] | 0.2596 | no difference |

Ours leads on 1 of 3 seeds significantly (seed 0); the other two show no difference, one
favouring each side. Temporal-YOLOv8 also raises **zero** false alarms on this video's two
labelled distractors, on every seed -- the same as our temporal arm, and unlike YOLOMG,
which raised 2, 5 and 13 across the three seeds on the same task.

## What this does and does not establish

**Overall, no difference from Temporal-YOLOv8 has been detected on either public
benchmark**, on the two-test-plus-Holm rule this project applies to every other
comparison. On ARD-MAV the two arms are numerically close as well as statistically
indistinguishable overall. On NPS, Temporal-YOLOv8's point estimate leads and its
seed-to-seed variance is smaller than ours', but that lead does not clear the significance
bar either. On the project's own 8 px task, ours leads on one of three seeds and ties on
the other two.

**The one exception:** on ARD-MAV targets above 25 px, Temporal-YOLOv8 significantly
outperforms our temporal arm on 2 of its 3 seeds after Holm. That is the same size regime
where YOLOMG also beats this project (README §4). Below 25 px, on both benchmarks, nothing
separates the two arms.

This reimplementation carries a real, structural advantage this project's own arm does
not: a non-causal window reading half a second of the future. That it still does not
separate from a causal arm on the benchmarks, and loses outright on the one seed that
reaches significance on the project's own task, is the honest headline -- not a win
being talked down. What is *not* established is any claim that either representation
is better than the other; what the evidence supports is that a decade-old idea done two
ways, one of them handicapped by causality, land in the same place.
