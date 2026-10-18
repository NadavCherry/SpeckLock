# `realtime.fixedcam` — drone vs bird on fixed / PTZ cameras, for a Jetson Orin Nano

The edge pipeline for fixed or PTZ cameras that are still most of the time and re-point (pan/tilt/
zoom) now and then (designed for 1080p–1440p at 25 fps), with a drone anywhere from a few px to
hundreds of px and birds in the same sky. The task is to **stay on the drone's centre** and **never announce a bird as a drone**, at
≥ 24 FPS through TensorRT on an Orin Nano.

## Why not the full-frame temporal detector

`final/edge_rt` (EDGE-RT) runs a temporal YOLOv8n-P2 on a 1280×1280 frame every frame. On an Orin
Nano that projects to 10–15 FPS, and the model saw one airframe and one flock of birds. A still
camera makes motion cheap and strong, so this pipeline spends its network budget on a few small
crops instead of the whole frame:

| stage | where | what |
|---|---|---|
| `motion.MotionStage` | GPU, native resolution | lagged-median background (t−8/16/32), per-pixel noise (EMA of \|d\|), keyframe-pulse term, activity map that silences vegetation and cloud edges, OSD mask, camera gate (phase correlation; a homography only while the head moves) |
| `net.extract` | GPU | (t−6, t−3, t) grey crops of side clamp(3×size, 24, 96) px around every proposal **and** every live track's prediction, 32×32 + a log-scale plane (a target wider than 96 px is seen in part) |
| `net.TinyNet` | TensorRT FP16, static batch 32 | drone / bird / other + the drone's centre offset (≈0.44 M parameters) |
| `tracker.Tracker` | CPU, numpy + scipy | Hungarian association, per-track forgetting sum of class log-probabilities |
| `fasttrack.FastTracker` | CPU, numpy + scipy | the same tracker with the same outputs at a fraction of the Python cost (`FixedCamPipeline(fast=True)`): per-track numpy ring state, one vectorised camera move per frame, vectorised probe targets and external-candidate matching, batched feature rows (`trackfeat.track_features_padded`); the path roughness is skipped only when both roughness gates are off (`smooth_commit` and `smooth_keep` ≥ 1e8), where it cannot change a decision (asserted) |

## Tracker invariants (each one guards against a failure mode)

- **Evidence, not length.** A track is labelled only when its class posterior clears `commit`. The
  old `trackclass` rule made any track ≥ 120 frames a drone; a long-lived bird walked through it.
- **Watched flying** (the pursuit ring's rule): no drone label until the track has made ≥ 4 motion
  observations and moved; a drone-labelled track that stops moving *entirely* for 3 s is dropped.
  Probe crops of a roof corner look drone-like forever; a hovering drone still drifts.
- **Smooth path.** The last 20 observations must fit a quadratic in time with a small residual. A
  track that hops between clutter blobs covers ground, but not smoothly.
- **Probes nudge, never drag.** A probe (crop at the predicted position — how a hovering drone,
  which makes no motion, stays locked) may move its track only max(3, 0.3×size) px.
- **The reported point is the observation** (proposal centre + regressed offset), not the filter
  state; the filter only predicts and gates.

## Tools

| tool | purpose |
|---|---|
| `tools/fixedcam_harvest.py` | crops from a labelled clip, exactly as the live pipeline sees them (proposals labelled from GT, probes at the drone and birds, static-clutter and background negatives, labelled non-drone movers as hard negatives) |
| `tools/fixedcam_harvest_moving.py` | the same from a moving-camera labelled corpus (ARD-MAV), with per-crop homography compensation so the crops look like a still camera's |
| `tools/fixedcam_harvest_weak.py` | weak labels from single-class clips (Halmstad: fliers in a BIRD clip are birds) |
| `tools/fixedcam_train.py` | class-balanced training; flips, inversion, noise, blur, "freeze" |
| `tools/fixedcam_run.py` | the streaming loop: `points.json` (score it with `python -m dronedet centerscore`), `frames.json` (per-frame track ids and coasting), `timing.json` (per-stage means over all frames; wall-clock median and FPS over frames 50+), and with `--render` an annotated video (drone red, bird cyan, unconfirmed grey, GT green) |
| `tools/render_track_video.py` | renders any system's `points.json` after the fact (optionally `--frames frames.json` for real track ids), side by side with a second system if given |
| `tools/fixedcam_project_orin.py` | projects a measured timing onto an Orin Nano, stage by stage, with sourced factors (labelled PROJECTED) |

No trained classifier ships with this repository. Train one with `tools/fixedcam_train.py`, which
writes `DIR/model.pt`; `tools/fixedcam_run.py --weights DIR/model.pt` runs it in PyTorch. For
TensorRT, load that model into `realtime.fixedcam.net.TinyNet`, export it with
`realtime.fixedcam.trt.export_onnx`, and build the engine on the target with `trt.build_engine` (or
`trtexec --onnx=clf.onnx --fp16 --saveEngine=clf.engine`); engines are tied to the GPU and the
TensorRT version.

## Orin projection method

Each measured stage is scaled by the factor matching what bounds it (see the tool's docstring for
sources): small TensorRT networks by a YOLO26n anchor (Ultralytics' published TensorRT FP16 time on an Orin
Nano Super against the same model measured on an RTX 4080 Laptop: 2.32×); full-frame GPU arithmetic by memory bandwidth (4.2× Super, 6.4× original clocks); CPU
work by single-core benchmarks (3.5–3.9×); decode moves to NVDEC. Nothing here has been measured on
a Jetson.
