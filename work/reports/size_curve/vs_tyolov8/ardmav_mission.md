# Accuracy vs target size -- ardmav

Bins on sqrt(area) in pixels (mission). Evaluator: rule=iou, IoU=0.5, conf>=0.001, pooled over 15 sequences. Every arm is scored by this one evaluator; the arms differ only in which detections they read.

`n` is GT instances in the bin. `+-` is the sample standard deviation over training seeds. `CI` is a 95 % bootstrap interval over SEQUENCES.

| bin | n | ours | tyolov8 |
|---|---|---|---|
| <8 px | 5677 | 0.503 ± 0.011 | 0.258 ± 0.001 |
| 8-10 px | 5055 | 0.602 ± 0.007 | 0.514 ± 0.019 |
| 10-16 px | 7529 | 0.732 ± 0.013 | 0.827 ± 0.005 |
| 16-25 px | 4731 | 0.729 ± 0.029 | 0.888 ± 0.003 |
| >25 px | 5168 | 0.739 ± 0.023 | 0.938 ± 0.002 |

`*` = fewer than 50 GT instances; treat as underpowered.

## Crossover: ours minus tyolov8

| bin | n | delta | leader |
|---|---|---|---|
| <8 px | 5677 | +0.245 | ours |
| 8-10 px | 5055 | +0.088 | ours |
| 10-16 px | 7529 | -0.096 | tyolov8 |
| 16-25 px | 4731 | -0.159 | tyolov8 |
| >25 px | 5168 | -0.199 | tyolov8 |

**Crossover between `8-10 px` and `10-16 px`.** ours leads on the smaller side (+0.088) and trails on the larger (-0.096).

## Is the difference real? Paired, seed-matched, over sequences

Paired bootstrap **and** permutation over the 15 shared sequences; a bin is called significant only when both agree after a Holm correction across this table, matching `tools/make_summary.py`. Seeds are matched pairwise.

| bin | seed | d AP | 95% CI | p perm | p perm, Holm | verdict |
|---|---|---|---|---|---|---|
| <8 px | 0 | +0.246 | [+0.109, +0.385] | 0.1499 | 1.0000 | no difference |
| <8 px | 1 | +0.255 | [+0.115, +0.406] | 0.1049 | 1.0000 | no difference |
| <8 px | 2 | +0.234 | [+0.086, +0.373] | 0.1249 | 1.0000 | no difference |
| 8-10 px | 0 | +0.064 | [-0.067, +0.202] | 0.6923 | 1.0000 | no difference |
| 8-10 px | 1 | +0.107 | [-0.018, +0.234] | 0.4266 | 1.0000 | no difference |
| 8-10 px | 2 | +0.094 | [-0.042, +0.192] | 0.5335 | 1.0000 | no difference |
| 10-16 px | 0 | -0.114 | [-0.197, -0.006] | 0.0959 | 1.0000 | no difference |
| 10-16 px | 1 | -0.085 | [-0.165, +0.007] | 0.1968 | 1.0000 | no difference |
| 10-16 px | 2 | -0.089 | [-0.175, +0.001] | 0.2957 | 1.0000 | no difference |
| 16-25 px | 0 | -0.183 | [-0.398, -0.050] | 0.0519 | 0.6234 | no difference |
| 16-25 px | 1 | -0.131 | [-0.329, -0.022] | 0.1558 | 1.0000 | no difference |
| 16-25 px | 2 | -0.164 | [-0.363, -0.036] | 0.1479 | 1.0000 | no difference |
| >25 px | 0 | -0.221 | [-0.494, -0.089] | 0.0010 | 0.0150 | **significant** |
| >25 px | 1 | -0.178 | [-0.424, -0.065] | 0.0010 | 0.0150 | **significant** |
| >25 px | 2 | -0.199 | [-0.491, -0.068] | 0.0050 | 0.0649 | no difference |

Holm over the 15 rows of this table removed the mark from: >25 px seed 2.

**No bin where ours wins significantly on every seed.**

