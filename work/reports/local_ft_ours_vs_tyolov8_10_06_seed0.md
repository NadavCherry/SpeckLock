# 10_06 — one held-out flight

Protocol `specklock-centre` (centre distance, tau=12 px), 250 labelled instances over 250 frames, resampled as **9 blocks of 30 frames**.

> **This interval is WITHIN ONE SEQUENCE.** It measures whether a difference holds across the segments of this single flight. It is *not* evidence that the difference generalises to another flight — two videos cannot support that claim, and resampling one of them harder does not change what was measured.

| method | AP | Δ vs baseline | 95% CI on Δ | p | verdict |
|---|---|---|---|---|---|
| **tyolov8** (baseline) | 0.698 | — | — | — | — |
| temporal | 0.834 | +0.137 | [+0.010, +0.296] | 0.0138 | **better** |

### False alarms on labelled distractors

_No detection landed on a labelled distractor in any arm. In this GT the birds are `ignore=True`, so a hit on one is recorded here rather than silently counted as a false positive against the background._
