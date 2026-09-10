# Camera motion -- nps

Frame-to-frame global translation by phase correlation (a lower bound on camera motion: translation only). *step* = pixels per frame; *window A* = net background displacement across A frames; *low-resp.* = share of steps with response < 0.35.

| video | frames | resolution | step median | step p95 | step max | window 12 median | window 12 p95 | window 30 median | window 30 p95 | response median | low-resp. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Clip_41 | 1529 | 1920x1080 | 4.375 | 14.443 | 25.519 | 43.206 | 167.504 | 98.158 | 424.399 | 0.760 | 11.6 % |
| Clip_42 | 1660 | 1920x1080 | 7.195 | 15.510 | 21.619 | 78.148 | 176.123 | 180.475 | 410.825 | 0.731 | 8.0 % |
| Clip_43 | 803 | 1280x960 | 3.857 | 13.114 | 32.244 | 42.481 | 131.509 | 95.638 | 247.015 | 0.861 | 9.2 % |
| Clip_44 | 859 | 1280x960 | 7.982 | 14.830 | 59.268 | 94.660 | 159.580 | 221.992 | 358.287 | 0.803 | 9.3 % |
| Clip_45 | 936 | 1280x960 | 6.842 | 12.918 | 19.085 | 78.396 | 143.544 | 192.111 | 319.303 | 0.788 | 8.3 % |
| Clip_46 | 1131 | 1280x960 | 5.097 | 12.853 | 63.083 | 51.968 | 147.193 | 113.545 | 332.086 | 0.820 | 7.6 % |
| Clip_47 | 936 | 1920x1080 | 7.122 | 21.373 | 27.912 | 78.404 | 248.214 | 178.022 | 499.252 | 0.809 | 11.2 % |
| Clip_48 | 902 | 1920x1080 | 12.140 | 26.808 | 36.223 | 143.275 | 310.051 | 298.629 | 751.524 | 0.812 | 6.1 % |
| Clip_49 | 1800 | 1920x1080 | 3.864 | 8.755 | 13.229 | 45.592 | 99.870 | 106.371 | 232.507 | 0.853 | 3.4 % |
| Clip_50 | 1799 | 1920x1080 | 3.308 | 8.139 | 10.814 | 39.323 | 95.783 | 93.418 | 227.740 | 0.857 | 0.0 % |
| **pooled** | 12355 |  | 5.381 | 15.623 | 63.083 | 59.551 | 174.774 | 136.336 | 409.452 | 0.821 | 6.9 % |

Median labelled target: **14.8 px**. Across a 12-frame window the background moves a median **59.55 px** -- 4.01 of the target's size.

