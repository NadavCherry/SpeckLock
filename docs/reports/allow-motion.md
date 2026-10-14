# `allow_motion`: what the motion clause costs, measured end to end

**Status: measurement and recommendation only.** No behaviour and no published number is
changed by this report. The decision belongs to the owner; §6 sets it out.

## 1. The question

`dronedet.trackclass.classify_tracks(..., allow_motion=True)` counts any matched detection
whose label contains `motion` (score ≥ 0.35) as drone evidence, alongside the verifier's
`drone*` confirmations. `True` is the default, and it is what `final/run_final.py` (step 3)
and `tools/tracks_to_dets.py --classify` (run_final's step 4) both use. `tools/run_max.py` is
the only caller that decides: it passes `allow_motion=near_static`, from a camera-motion
pre-scan.

What the label means depends on the detector that produced it:

- **PC-MAX** fuses `fullS` with `moe3-stacked`, i.e. `dronedet/methods/hybrid2.py`. That method
  labels a motion proposal bare `motion` when the 640 verifier looked at it and found **neither a
  drone nor a bird** ([`hybrid2.py:127-133`](../../dronedet/methods/hybrid2.py)). Under
  `allow_motion=True`, the verifier's *non*-confirmation therefore counts as confirmation. Bare
  `motion` is 73 % of PC-MAX's detections at score ≥ 0.35 on 07_05 (4148 / 5685) and 76 % on 10_06
  (1400 / 1837).
- **EDGE-RT** (one yolov8n-p2) emits `drone` and `bird` only. It has no `motion` labels at all.
- **run_max's near-static path** (`mc_hybrid.py`) also labels unverified motion `motion`, but it
  scales that score by a per-frame clutter factor that suppresses it in busy frames. Round 6
  added the clause for that path: its combined multi-dataset appearance model scores only
  AP 0.291 on the black drone in 10_06 by appearance alone
  ([round 6](round6-max-pipeline.md)).

## 2. How it was measured

[`tools/allow_motion_ab.py`](../../tools/allow_motion_ab.py) runs `final.run_final.main()`
**unmodified** for steps 3–5: classification, track-integrated detections and the alarm
summary. Only step 1 (detection) is substituted, by copying a detection JSON, and step 2
(tracker) by copying a track JSON or, with `--retrack`, running today's tracker. Three arms:

| arm | `allow_motion` | `LONG_TRACK` promotion |
|---|---|---|
| `on` | True | 120 tracked frames (as shipped): **today's run_final** |
| `off` | False | 120 (as shipped) |
| `off_nobypass` | False | none |

Three sources of detections and tracks:

1. **Committed round-3 artifacts.** These are the split-trained v3 models whose outputs back
   every round-3 tracked number: PC-MAX `work/det3/*/pc-max.json` + `work/tracks3/*/pc-max-all.json`,
   and EDGE-RT `realtime/work/out3/*/rt-c-full1280/`.
2. **The same detections re-tracked by today's tracker.** The tracker no longer reproduces the
   committed track files: on 07_05 PC-MAX it confirms 25 tracks where the committed file has 20.
   This source checks that the conclusion does not depend on which tracks one starts from.
3. **The shipped `final/` package**, run end to end on this machine with `final/run_final.py`.
   §4 covers it.

Definitions:

- **Target recovered** is track-level object recall from `tools/track_level_birds.py`: the
  drone is recovered if some raised track sits on it.
- **Clutter tracks** and **bird tracks** are tracks raised as `drone`/`near` whose frames sit
  mostly on nothing, or mostly on a labelled bird. Bird tracks are reported as raised (formed).
  Tracks on the landed drone (`near`) are scored neither way, as in that tool.
- **Alarm latency** counts frames from the first verified GT frame of the drone to the first
  frame at which the classifier rule holds on the target track's prefix. In brackets is
  run_final's own `alarms.txt` latency, counted from track start, whose rule is the 8th `drone*`
  detection at ≥ 0.5.
- **AP / F1** come from the round-3 scorer (`dronedet.evaluate`, τ = 12 px), the scorer behind
  every published tracked row. *FP boxes* counts every emitted tracked box off the target at
  any score, which is what an operator is shown. AP hides these because it is score-weighted.

## 3. Results — committed round-3 artifacts

Full video, both videos. 07_05 columns also give the round-3 validation split (frames 342–570).

| profile | video | arm | tracks raised | target recovered | **clutter tracks** | bird tracks (formed) | LONG_TRACK-only | alarm latency (alarms.txt) | causal AP / F1 | smoothed AP / F1 | FP boxes (smoothed) | 07_05-val causal AP / F1 | = published file |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PC-MAX | 07_05 | `on` | 14 | 1 / 1 | **11** | 0 (3) | 0 | 7 (7) | 0.778 / 0.745 | 0.779 / 0.745 | 2163 | 0.580 / 0.529 | no |
| PC-MAX | 07_05 | `off` | 4 | 1 / 1 | **2** | 0 (3) | 2 | 7 (7) | 0.848 / 0.750 | 0.849 / 0.752 | 427 | 0.600 / 0.537 | no |
| PC-MAX | 07_05 | `off_nobypass` | 2 | 1 / 1 | **0** | 0 (3) | 0 | 7 (7) | 0.998 / 0.998 | 1.000 / 1.000 | 0 | 0.995 / 0.995 | **yes** |
| PC-MAX | 10_06 | `on` | 5 | 1 / 1 | **4** | — | 0 | 7 (7) | 1.000 / 1.000 | 1.000 / 1.000 | 709 | | no |
| PC-MAX | 10_06 | `off` | 1 | 1 / 1 | **0** | — | 0 | 7 (7) | 1.000 / 1.000 | 1.000 / 1.000 | 0 | | **yes** |
| PC-MAX | 10_06 | `off_nobypass` | 1 | 1 / 1 | **0** | — | 0 | 7 (7) | 1.000 / 1.000 | 1.000 / 1.000 | 0 | | **yes** |
| EDGE-RT | 07_05 | all three | 1 | 1 / 1 | 0 | 0 (0) | 0 | 9 (7) | 0.995 / 0.996 | 0.996 / 0.998 | 0 | 0.995 / 0.995 | yes |
| EDGE-RT | 10_06 | all three | 1 | 1 / 1 | 0 | — | 0 | 8 (8) | 1.000 / 1.000 | 1.000 / 1.000 | 0 | | yes |

*= published file* means run_final's `tracked_dets.json`, plus the causal variant, is
value-for-value identical to the committed round-3 `tracked-pcmax.json` /
`tracked-pcmax-smooth.json` (EDGE-RT: `tracked-rt-c-full1280.json` / `tracked-smooth.json`).
On 07_05, 2 of the raised tracks in `on` and 1 in `off` sit on the landed drone and are not scored.
Full rows: [`work/reports/allow_motion/committed.md`](../../work/reports/allow_motion/committed.md)
(and `.json`).

**Re-tracked by today's tracker** ([`retrack.md`](../../work/reports/allow_motion/retrack.md)):
the same picture, slightly larger. PC-MAX 07_05 gives 15 / 2 / 0 clutter tracks across the three
arms (5 bird tracks formed, none raised), and 10_06 gives 3 / 0 / 0. Target recovery and alarm
latency are unchanged in every arm, and EDGE-RT is identical in every arm.

## 4. Results — the shipped `final/` package

`final/run_final.py --profile {pc-max,edge-rt} --no-video` was run unmodified on both videos
on an RTX 4080 Laptop GPU. Its `dets.json` / `tracks.json` were then fed through the same three
arms. These are the weights in `final/`, retrained on all of 07_05, so **07_05 is their
training video** and only 10_06 is held out. The detections come from this GPU. The round-3 §6
run used another card, so the two need not match box for box.

| profile | video | arm | tracks raised | target recovered | **clutter tracks** | **bird tracks** (formed) | LONG_TRACK-only | `not confirmed` in alarms.txt | alarm latency (alarms.txt) | AP / F1 (causal = smoothed) | FP boxes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PC-MAX | 07_05 | `on` | 13 | 1 / 1 | **9** | **1** (4) | 1 | 10 of 13 | 7 (8) | 0.928 / 0.861 | 1631 |
| PC-MAX | 07_05 | `off` | 3 | 1 / 1 | **0** | **1** (4) | 1 | 1 of 3 | 7 (8) | 1.000 / 1.000 | 84 |
| PC-MAX | 07_05 | `off_nobypass` | 2 | 1 / 1 | **0** | **0** (4) | 0 | 0 | 7 (8) | 1.000 / 1.000 | 0 |
| PC-MAX | 10_06 | `on` | 2 | 1 / 1 | **1** | — | 0 | 1 of 2 | 7 (7) | 1.000 / 1.000 | 131 |
| PC-MAX | 10_06 | `off`, `off_nobypass` | 1 | 1 / 1 | **0** | — | 0 | 0 | 7 (7) | 1.000 / 1.000 | 0 |
| EDGE-RT | 07_05 | all three | 1 | 1 / 1 | 0 | 0 (0) | 0 | 0 | 7 (7) | 1.000 / 1.000 | 0 |
| EDGE-RT | 10_06 | all three | 1 | 1 / 1 | 0 | — | 0 | 0 | 8 (10) | 1.000 / 1.000 | 0 |

Full rows: [`work/reports/allow_motion/shipped.md`](../../work/reports/allow_motion/shipped.md)
(and `.json`).

- **On the held-out 10_06, the shipped default raises one clutter track** (track 77, 131 frames).
  It passes at `conf_frac` 0.714, just over 0.70, only because motion detections count, and its
  own `alarms.txt` line says "not confirmed". With `off` it is gone, and the output is the single
  drone track that round 3 §6 describes.
- **On 07_05 the shipped package raises a labelled bird as a drone, in both `on` and `off`.**
  Track 12 sits on a GT `bird*` object for 268 frames, about 9 s. It has **one** confirmed
  detection in 130 tracked frames (`conf_frac` 0.008), and the `LONG_TRACK` clause promotes it
  at frame 279, when it reaches 120 tracked frames. Only `off_nobypass` rejects it. This is the
  failure [`track-level-birds.md`](track-level-birds.md) warned the bypass could produce and
  found "harmless by luck of the data". With the shipped weights the data are no longer lucky.
  No other bird track is raised in any of the 36 runs here (3 sources × 4 cases × 3 arms).
- EDGE-RT again does not move: one drone track per video, alarm latency 7 and 10 frames
  (`alarms.txt`), matching round 3 §6.

## 5. What the numbers say

**EDGE-RT: the flag is a no-op.** Its detector emits no `motion` labels, so every metric is
identical in every arm, on every source. Whatever is decided costs EDGE-RT nothing.

**PC-MAX: motion evidence buys the drone nothing measurable, and it raises the clutter.**
Going from `on` to `off` changes none of the following, on either video and from any of the three
sources:

- target recovery: 1 / 1 everywhere;
- alarm latency: 7 frames from the first GT frame, the same frame in both arms;
- bird outcome: identical in both arms.

PC-MAX's verifier already confirms the drone without motion evidence. Its target track's
`conf_frac` moves from 0.997 to 0.994 on 07_05 and from 0.963 to 0.941 on 10_06 (shipped:
1.000 → 1.000 and 0.943 → 0.937). What `on` adds is clutter:

| source | 07_05 clutter tracks, `on` → `off` | 10_06 clutter tracks, `on` → `off` |
|---|---|---|
| committed round-3 artifacts | 11 → 2 | 4 → 0 |
| re-tracked | 15 → 2 | 3 → 0 |
| shipped package | 9 → 0 | 1 → 0 |

The committed artifacts also lose 2163 → 427 and 709 → 0 false tracked boxes. On 10_06, AP stays
1.000 in every arm because the clutter scores below the best-F1 threshold. That is why AP never
showed this.

**run_final contradicts itself under `on`.** Its alarm line (`alarms.txt`) only "confirms" a
track with eight `drone*` detections at ≥ 0.5. Of the tracks run_final raises and paints,
`alarms.txt` calls 10 of 14 "not confirmed" on 07_05 and 4 of 5 on 10_06 (shipped: 10 of 13
and 1 of 2). Under `off` the count is 0 everywhere except the shipped bird track 12, which
LONG_TRACK raises.

**Gating on camera motion, as run_max does, would change nothing here.** run_max's own pre-scan
(`prescan_ego`) measures **0.030 px/frame on 07_05 and 0.072 px/frame on 10_06**, far below its
1.5 px/frame threshold. Both videos are near-static, so the gate selects `allow_motion=True`,
which is today's default, with its 11 and 4 clutter tracks (shipped: 9 and 1). The clutter on these videos is not
moving-camera parallax. It is foliage and texture on a static camera, and the verifier has
already rejected it. Whether `motion` is evidence is decided by the **detector** that wrote the
label, not by the camera.

**`off` alone does not restore the published numbers on 07_05.** Two clutter tracks remain,
636 and 645, about 8 s each. Without motion evidence their whole-track `conf_frac` is 0.47 and
0.42, so they pass **only through the `LONG_TRACK` bypass**. That is the one promotion with no
appearance requirement. [`track-level-birds.md`](track-level-birds.md) reports that the bypass
"never fired", and that holds only under `on`, where motion evidence had already confirmed these
two tracks. Every committed round-3 tracked file comes from a rule with **neither** clause.
[`provenance.md`](../../work/reports/allow_motion/provenance.md) applies 16 candidate rules to
the 11 committed tracked files whose inputs are also committed. Only appearance-only rules with
no bypass reproduce all 11. Today's rule (0.35 / 0.70 / motion / 120) reproduces 4, all edge.
The committed files cannot tell 0.35 / 0.70 from 0.50 / 0.50: both reproduce all 11.

**One caveat on "online".** 636 and 645 do satisfy the rule on their *prefix*, by appearance
alone, at frames 351 and 344, before their `conf_frac` decays. An online classifier that
latches would raise them in every arm, `off_nobypass` included. The zero in `off_nobypass` is a
whole-track result, and the round-3 figures share that property.

## 6. Recommendation

**Change the default to `allow_motion=False`. Do not gate run_final on camera motion.**

1. Off is the safe default for every detector this repository ships. On EDGE-RT it is a no-op.
   On PC-MAX it removes 9 of 11 clutter tracks on 07_05 and 4 of 4 on 10_06; with the shipped
   weights it removes 9 of 9 and 1 of 1. None of this costs recovery, latency or bird rejection,
   as far as measured. It is the rule round 3 measured and published, and with it the 10_06
   PC-MAX output equals the published file exactly.
2. The one caller that wants motion evidence, run_max's mc-hybrid path, already passes it
   explicitly, so flipping the default does not touch it. A default that turns a verifier
   *rejection* into confirmation is the wrong way round for a system whose measured failure
   mode is false alarms.
3. A camera-motion gate in run_final would pick `True` on both videos and fix nothing (§5).
   Camera motion is the wrong key here; the label semantics are the right one, and only the
   caller that knows its detector can decide. So keep the decision explicit at the call site,
   as run_max does, rather than inferring it from the camera.

What the change would involve (not made here):

- `dronedet/trackclass.py`: default `False` in `classify_tracks` and `classify_files`, and the
  docstring updated.
- `tools/tracks_to_dets.py`: an `--allow-motion` opt-in, so motion is not reachable only
  through a default.
- `final/run_final.py`: no change is needed if the default flips. If the default stays, run_final
  must pass `False` in **both** step 3 and step 4. Those steps classify independently, and fixing
  one would leave `tracks_drone.json` / `annotated.mp4` disagreeing with `tracked_dets.json`.
- `tools/track_level_birds.py`: its CLI defaults to motion on (`--no-motion` opts out). Match it
  to whatever run_final ships, so the bird report measures the shipped behaviour.
- Published numbers: [`track-level-birds.md`](track-level-birds.md) describes today's default
  (11 / 4 clutter tracks) and should be re-run after the decision.

**A second decision, independent of the first: `LONG_TRACK` in run_final.** With motion off, it
is the only thing that still separates the committed 07_05 PC-MAX output from its published
numbers: 2 clutter tracks, smoothed AP 0.849 against 1.000. With the shipped weights it is the
clause that **raises a labelled bird** on 07_05 (track 12, one confirmed detection in 130),
whatever `allow_motion` is set to. Round 6 introduced it for run_max, to keep low-confidence
drones on NPS / ARD-MAV under the combined model. In run_final, on every source here, it
promoted two clutter tracks and one bird, and never the drone. The options are:

- (a) keep it, and republish the 07_05 PC-MAX tracked numbers;
- (b) make it a run_max-only parameter, which restores the published round-3 outputs exactly;
- (c) require a minimum appearance fraction for it.

Its effect on run_max's external datasets is **not measured here**. That is where removing it
could cost recall, so (b) is the lowest-risk choice for run_final and still a decision for the
owner.

## 7. Reproducing

```bash
# committed round-3 artifacts (no GPU, seconds)
python tools/allow_motion_ab.py --out work/reports/allow_motion/committed
# same detections, today's tracker (CPU, ~1 min)
python tools/allow_motion_ab.py --retrack --out work/reports/allow_motion/retrack
# which classifier rule produced the committed tracked files
python tools/allow_motion_ab.py --provenance --out work/reports/allow_motion/provenance
# the shipped package (GPU, ~7 min here): run_final once per profile and video, then the A/B
python final/run_final.py --video data/videos/07_05.mp4 --profile pc-max --out run/pc-max_07_05 --no-video
#   ... likewise pc-max_10_06, edge-rt_07_05, edge-rt_10_06
python tools/allow_motion_ab.py \
    --case pc-max 07_05 run/pc-max_07_05/dets.json run/pc-max_07_05/tracks.json \
    --case pc-max 10_06 run/pc-max_10_06/dets.json run/pc-max_10_06/tracks.json \
    --case edge-rt 07_05 run/edge-rt_07_05/dets.json run/edge-rt_07_05/tracks.json \
    --case edge-rt 10_06 run/edge-rt_10_06/dets.json run/edge-rt_10_06/tracks.json \
    --out work/reports/allow_motion/shipped
```

The shipped package's own detections and tracks are not committed. They are GPU output and
need not match box for box on another card. The committed `shipped.json` records every row
derived from them.
