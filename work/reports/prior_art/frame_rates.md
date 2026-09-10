# Frame rates of every video the Temporal-YOLOv8 comparison reads

Temporal-YOLOv8 samples "around 15 frames ... before and after the current frame", "assuming a
30-frames-per-second (FPS) source video" (van Leeuwen et al., *Sensors* 24(22):7387, 2024). Fifteen
frames are half a second only at about 30 fps, so every clip's rate is recorded here, where
`configs/experiments/prior_art.py` and `tools/make_summary.py` can point to it.

## Benchmark videos

Printed by the dataset build's frame-rate check (`cluster/tyolov8_build.sbatch`, job 21175379,
`cv2.CAP_PROP_FPS` on every video), verbatim:

```
  ardmav: 60 videos (29.77 fps x1, 29.97 fps x59); 15 frames = 0.501 .. 0.504 s
    clips the builder reads: 45, fps 29.97 .. 29.97
  nps: 50 videos (28.0 fps x1, 28.89 fps x1, 29.0 fps x8, 29.58 fps x1, 29.66 fps x1, 29.69 fps x1, 29.77 fps x1, 29.8 fps x4, 29.9 fps x1, 29.95 fps x1, 29.96 fps x3, 29.97 fps x25, 59.9 fps x2); 15 frames = 0.250 .. 0.536 s
    outside 29.5-30.5 fps: Clip_014 29.0, Clip_015 29.0, Clip_017 28.0, Clip_018 29.0, Clip_028 29.0, Clip_032 29.0, Clip_033 29.0, Clip_037 29.0, Clip_043 28.89 (TEST), Clip_044 29.0 (TEST), Clip_049 59.9 (TEST), Clip_050 59.9 (TEST)
    clips the builder reads: 40, fps 28.0 .. 29.97
```

"Clips the builder reads" are the training and validation clips; ARD-MAV's one 29.77 fps video is
a test video.

## The project's own videos

Read from the container with `cv2.CAP_PROP_FPS`: `data/videos/07_05.mp4` 30.0 fps and
`data/videos/10_06.mp4` 30.0 fps, both 1280×720.

## What follows

- Every clip either arm trains on runs at 28.0–29.97 fps, so there 15 frames is 0.50–0.54 s: the
  paper's "around" half second.
- Two NPS test clips, Clip_049 and Clip_050, run at 59.9 fps. On them 15 frames is 0.25 s, as
  SpeckLock's 6 frames is 0.10 s: both arms meet half the window they were trained on. Both are
  scored like every other clip, and `tools/make_summary.py` also prints the NPS difference with
  the two left out, as effect sizes only — declared before any Temporal-YOLOv8 scorecard existed.
