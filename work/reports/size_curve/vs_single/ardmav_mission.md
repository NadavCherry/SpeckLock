# Accuracy vs target size -- ardmav

Bins on sqrt(area) in pixels (mission). Evaluator: rule=iou, IoU=0.5, conf>=0.001, pooled over 15 sequences. Every arm is scored by this one evaluator; the arms differ only in which detections they read.

`n` is GT instances in the bin. `+-` is the sample standard deviation over training seeds. `CI` is a 95 % bootstrap interval over SEQUENCES.

| bin | n | ours | ours-single |
|---|---|---|---|
| <8 px | 5677 | 0.503 ± 0.011 | 0.378 ± 0.031 |
| 8-10 px | 5055 | 0.602 ± 0.007 | 0.493 ± 0.029 |
| 10-16 px | 7529 | 0.732 ± 0.013 | 0.728 ± 0.033 |
| 16-25 px | 4731 | 0.729 ± 0.029 | 0.758 ± 0.039 |
| >25 px | 5168 | 0.739 ± 0.023 | 0.771 ± 0.054 |

`*` = fewer than 50 GT instances; treat as underpowered.

## Crossover: ours minus ours-single

| bin | n | delta | leader |
|---|---|---|---|
| <8 px | 5677 | +0.124 | ours |
| 8-10 px | 5055 | +0.109 | ours |
| 10-16 px | 7529 | +0.004 | ours |
| 16-25 px | 4731 | -0.029 | ours-single |
| >25 px | 5168 | -0.033 | ours-single |

**Crossover between `10-16 px` and `16-25 px`.** ours leads on the smaller side (+0.004) and trails on the larger (-0.029).

## Is the difference real? Paired, seed-matched, over sequences

Paired bootstrap **and** permutation over the 15 shared sequences; a bin is called significant only when both agree after a Holm correction across this table, matching `tools/make_summary.py`. Seeds are matched pairwise.

| bin | seed | d AP | 95% CI | p perm | p perm, Holm | verdict |
|---|---|---|---|---|---|---|
| <8 px | 0 | +0.158 | [+0.033, +0.317] | 0.0629 | 0.9441 | no difference |
| <8 px | 1 | +0.103 | [+0.017, +0.233] | 0.0819 | 1.0000 | no difference |
| <8 px | 2 | +0.111 | [-0.005, +0.287] | 0.1988 | 1.0000 | no difference |
| 8-10 px | 0 | +0.124 | [-0.050, +0.247] | 0.3357 | 1.0000 | no difference |
| 8-10 px | 1 | +0.083 | [-0.097, +0.222] | 0.4066 | 1.0000 | no difference |
| 8-10 px | 2 | +0.119 | [-0.075, +0.265] | 0.3526 | 1.0000 | no difference |
| 10-16 px | 0 | +0.020 | [-0.032, +0.103] | 0.5734 | 1.0000 | no difference |
| 10-16 px | 1 | -0.020 | [-0.086, +0.047] | 0.5485 | 1.0000 | no difference |
| 10-16 px | 2 | +0.011 | [-0.079, +0.075] | 0.7982 | 1.0000 | no difference |
| 16-25 px | 0 | -0.009 | [-0.114, +0.084] | 0.8631 | 1.0000 | no difference |
| 16-25 px | 1 | -0.026 | [-0.127, +0.056] | 0.5465 | 1.0000 | no difference |
| 16-25 px | 2 | -0.052 | [-0.169, +0.029] | 0.4156 | 1.0000 | no difference |
| >25 px | 0 | +0.007 | [-0.055, +0.077] | 0.8631 | 1.0000 | no difference |
| >25 px | 1 | -0.050 | [-0.158, +0.038] | 0.3207 | 1.0000 | no difference |
| >25 px | 2 | -0.056 | [-0.138, +0.021] | 0.3606 | 1.0000 | no difference |

Holm over the 15 rows of this table changed no verdict.

**No bin where ours wins significantly on every seed.**

