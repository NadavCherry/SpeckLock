# Which classifier rule produced the committed tracked files?

11 committed `tracked-*.json` files whose track and detection inputs are also committed. For each candidate rule (N_CONF = 8 throughout), the kept-track set is compared with the track ids present in the committed file.

| drone score | conf_frac | motion clause | LONG_TRACK | files reproduced |
|---|---|---|---|---|
| 0.35 | 0.7 | no | none | 11 / 11 |
| 0.5 | 0.5 | no | none | 11 / 11 |
| 0.35 | 0.5 | no | none | 10 / 11 |
| 0.35 | 0.7 | no | 120 | 9 / 11 |
| 0.5 | 0.5 | no | 120 | 9 / 11 |
| 0.5 | 0.7 | no | 120 | 9 / 11 |
| 0.5 | 0.7 | no | none | 9 / 11 |
| 0.35 | 0.5 | no | 120 | 8 / 11 |
| 0.35 | 0.7 | yes | 120 | 4 / 11 |
| 0.35 | 0.7 | yes | none | 4 / 11 |
| 0.5 | 0.5 | yes | 120 | 4 / 11 |
| 0.5 | 0.5 | yes | none | 4 / 11 |
| 0.5 | 0.7 | yes | 120 | 4 / 11 |
| 0.35 | 0.5 | yes | 120 | 3 / 11 |
| 0.35 | 0.5 | yes | none | 3 / 11 |
| 0.5 | 0.7 | yes | none | 2 / 11 |

Files: `work/det3/0705/tracked-pcmax.json`, `work/det3/1006/tracked-pcmax.json`, `work/det3/0705/tracked-pcmax-ms15.json`, `work/det3/0705/tracked-moe3v3.json`, `work/det3/1006/tracked-moe3v3.json`, `work/det3/0705/tracked-moe3v3-cpu.json`, `work/det2/tracked-moe3-cls.json`, `realtime/work/out3/0705/rt-c-full1280/tracked-rt-c-full1280.json`, `realtime/work/out3/1006/rt-c-full1280/tracked-rt-c-full1280.json`, `realtime/work/out3/0705/rt-d-full640/tracked-rt-d-full640.json`, `realtime/work/out3/1006/rt-d-full640/tracked-rt-d-full640.json`
