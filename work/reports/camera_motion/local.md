# Camera motion -- local

Frame-to-frame global translation by phase correlation (a lower bound on camera motion: translation only). *step* = pixels per frame; *window A* = net background displacement across A frames; *low-resp.* = share of steps with response < 0.35.

| video | frames | resolution | step median | step p95 | step max | window 12 median | window 12 p95 | window 30 median | window 30 p95 | response median | low-resp. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10_06 | 361 | 1280x720 | 0.020 | 0.138 | 0.311 | 0.142 | 0.346 | 0.331 | 0.499 | 0.910 | 0.0 % |
| 07_05 | 571 | 1280x720 | 0.023 | 0.136 | 1.520 | 0.157 | 0.670 | 0.313 | 1.371 | 0.875 | 0.0 % |
| **pooled** | 932 |  | 0.021 | 0.136 | 1.520 | 0.146 | 0.532 | 0.327 | 0.898 | 0.877 | 0.0 % |

Median labelled target: **8.0 px**. Across a 12-frame window the background moves a median **0.15 px** -- 0.02 of the target's size.

