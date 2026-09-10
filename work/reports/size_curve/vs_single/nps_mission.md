# Accuracy vs target size -- nps

Bins on sqrt(area) in pixels (mission). Evaluator: rule=iou, IoU=0.5, conf>=0.001, pooled over 10 sequences. Every arm is scored by this one evaluator; the arms differ only in which detections they read.

`n` is GT instances in the bin. `+-` is the sample standard deviation over training seeds. `CI` is a 95 % bootstrap interval over SEQUENCES.

| bin | n | ours | ours-single |
|---|---|---|---|
| 10-16 px | 6700 | 0.394 ± 0.052 | 0.407 ± 0.008 |
| 16-25 px | 2396 | 0.265 ± 0.055 | 0.306 ± 0.060 |
| >25 px * | 28 | 0.005 ± 0.001 | 0.007 ± 0.002 |

`*` = fewer than 50 GT instances; treat as underpowered.

## Crossover: ours minus ours-single

| bin | n | delta | leader |
|---|---|---|---|
| 10-16 px | 6700 | -0.012 | ours-single |
| 16-25 px | 2396 | -0.041 | ours-single |
| >25 px * | 28 | -0.002 | ours-single |

Bins excluded from the crossover test: `>25 px` — fewer than 50 GT instances, or a delta below 0.001.

**No crossover on nps.** ours-single leads in every bin with enough GT to judge (`10-16 px`, `16-25 px`), so there is no size regime that separates the methods here.

`nps` contains **no ground truth at all** in `<8 px`, `8-10 px`. Nothing about those sizes can be concluded from this dataset.

## Is the difference real? Paired, seed-matched, over sequences

Paired bootstrap **and** permutation over the 10 shared sequences; a bin is called significant only when both agree after a Holm correction across this table, matching `tools/make_summary.py`. Seeds are matched pairwise.

| bin | seed | d AP | 95% CI | p perm | p perm, Holm | verdict |
|---|---|---|---|---|---|---|
| 10-16 px | 0 | -0.021 | [-0.059, +0.038] | 0.6024 | 1.0000 | no difference |
| 10-16 px | 1 | +0.052 | [-0.121, +0.130] | 0.4376 | 1.0000 | no difference |
| 10-16 px | 2 | -0.068 | [-0.160, -0.029] | 0.0839 | 0.5874 | no difference |
| 16-25 px | 0 | +0.021 | [-0.053, +0.074] | 0.6913 | 1.0000 | no difference |
| 16-25 px | 1 | +0.012 | [-0.183, +0.099] | 0.8851 | 1.0000 | no difference |
| 16-25 px | 2 | -0.157 | [-0.323, -0.079] | 0.0180 | 0.1439 | no difference |
| >25 px | 0 | -0.000 | [-0.002, +0.003] | 0.9830 | 1.0000 | no difference |
| >25 px | 1 | -0.001 | [-0.025, +0.002] | 0.7562 | 1.0000 | no difference |
| >25 px | 2 | -0.005 | [-0.040, +0.000] | 0.0020 | 0.0180 | no difference |

Holm over the 9 rows of this table removed the mark from: 16-25 px seed 2.

**No bin where ours wins significantly on every seed.**

