# Accuracy vs target size -- nps

Bins on sqrt(area) in pixels (mission). Evaluator: rule=iou, IoU=0.5, conf>=0.001, pooled over 10 sequences. Every arm is scored by this one evaluator; the arms differ only in which detections they read.

`n` is GT instances in the bin. `+-` is the sample standard deviation over training seeds. `CI` is a 95 % bootstrap interval over SEQUENCES.

| bin | n | ours | tyolov8 |
|---|---|---|---|
| 10-16 px | 6700 | 0.394 ± 0.052 | 0.405 ± 0.018 |
| 16-25 px | 2396 | 0.265 ± 0.055 | 0.380 ± 0.015 |
| >25 px * | 28 | 0.005 ± 0.001 | 0.077 ± 0.101 |

`*` = fewer than 50 GT instances; treat as underpowered.

## Crossover: ours minus tyolov8

| bin | n | delta | leader |
|---|---|---|---|
| 10-16 px | 6700 | -0.010 | tyolov8 |
| 16-25 px | 2396 | -0.115 | tyolov8 |
| >25 px * | 28 | -0.072 | tyolov8 |

Bins excluded from the crossover test: `>25 px` — fewer than 50 GT instances, or a delta below 0.001.

**No crossover on nps.** tyolov8 leads in every bin with enough GT to judge (`10-16 px`, `16-25 px`), so there is no size regime that separates the methods here.

`nps` contains **no ground truth at all** in `<8 px`, `8-10 px`. Nothing about those sizes can be concluded from this dataset.

## Is the difference real? Paired, seed-matched, over sequences

Paired bootstrap **and** permutation over the 10 shared sequences; a bin is called significant only when both agree after a Holm correction across this table, matching `tools/make_summary.py`. Seeds are matched pairwise.

| bin | seed | d AP | 95% CI | p perm | p perm, Holm | verdict |
|---|---|---|---|---|---|---|
| 10-16 px | 0 | -0.031 | [-0.076, +0.015] | 0.7153 | 1.0000 | no difference |
| 10-16 px | 1 | +0.037 | [-0.042, +0.093] | 0.6104 | 1.0000 | no difference |
| 10-16 px | 2 | -0.037 | [-0.091, -0.001] | 0.4785 | 1.0000 | no difference |
| 16-25 px | 0 | -0.124 | [-0.195, -0.027] | 0.2807 | 1.0000 | no difference |
| 16-25 px | 1 | -0.045 | [-0.121, +0.056] | 0.6823 | 1.0000 | no difference |
| 16-25 px | 2 | -0.176 | [-0.289, -0.062] | 0.0490 | 0.3427 | no difference |
| >25 px | 0 | -0.017 | [-0.110, +0.001] | 0.0370 | 0.2957 | no difference |
| >25 px | 1 | -0.010 | [-0.065, +0.000] | 0.3327 | 1.0000 | no difference |
| >25 px | 2 | -0.189 | [-0.593, +0.000] | 0.0020 | 0.0180 | no difference |

Holm over the 9 rows of this table removed the mark from: 16-25 px seed 2.

**No bin where ours wins significantly on every seed.**

