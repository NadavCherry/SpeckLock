#!/usr/bin/env python3
"""Project a measured fixed-camera pipeline timing onto a Jetson Orin Nano - labelled PROJECTED.

    python tools/fixedcam_project_orin.py DIR/timing.json [...]

Nothing here is a measurement on a Jetson. Each measured stage is scaled by the factor that
matches what bounds it, with every factor sourced:

* GPU compute, small networks (classifier, crop kernels): 2.32x. Anchor: YOLO26n TensorRT FP16 at
  640 px - 4.57 ms on an Orin Nano Super (Ultralytics Jetson guide, https://docs.ultralytics.com/guides/nvidia-jetson/, JetPack 6.1,
  ``model.benchmark``, inference only) vs 1.97 ms median measured here with the same model,
  format and size (RTX 4080 Laptop, TensorRT 10.3, ultralytics 8.4.84).
* GPU memory-bound full-frame arithmetic (lagged median, noise and z maps): bandwidth ratio,
  RTX 4080 Laptop 432 GB/s vs Orin Nano 8 GB 102 GB/s (Super) / 68 GB/s (original clocks):
  4.24x / 6.35x.
* CPU (phase correlation, connected components, tracker, Python glue): Geekbench 5 single-core,
  i9-13980HX ~2,120 vs Orin Nano 538 (magazinmehatronika review, original 1.5 GHz clocks) = 3.94x;
  Super's 1.7 GHz -> ~3.5x.
* Decode: the Orin decodes H.264 on NVDEC (hardware, off the CPU/GPU critical path) and its luma
  plane *is* the grey frame, so the laptop's CPU decode and grey conversion are replaced by a
  0.5 ms copy allowance.

Non-Super clocks: GPU compute x1.63 (625 vs 1020 MHz) on top of the Super factors.
"""

from __future__ import annotations

import json
import sys

GPU_SMALL = {"super": 2.32, "orig": 2.32 * 1.63}
GPU_BW = {"super": 432 / 102, "orig": 432 / 68}
CPU = {"super": 3.5, "orig": 3.94}

# which bound each measured (sub)stage belongs to
MOTION_MAP = {
    "camera gate (cpu phase corr.)": "cpu",
    "camera move warps": "bw",
    "background+z (gpu, full res)": "bw",
    "download cell map": "zero",      # unified memory on the Orin
    "components (cpu)": "cpu",
}
STAGE_MAP = {"crops": "gpu", "classifier": "gpu", "tracker": "cpu"}


def project(t: dict, mode: str) -> dict:
    st = t["stage_ms_mean_all_frames"]
    mb = t.get("motion_breakdown_ms", {})
    f = {"cpu": CPU[mode], "gpu": GPU_SMALL[mode], "bw": GPU_BW[mode], "zero": 0.0}
    out = {"decode+grey (NVDEC luma copy)": 0.5}
    motion_known = 0.0
    for k, v in mb.items():
        out[f"motion: {k}"] = v * f[MOTION_MAP.get(k, "cpu")]
        motion_known += v
    rest = max(0.0, st.get("motion+components", 0.0) - motion_known)
    if rest:
        out["motion: unattributed (as cpu)"] = rest * f["cpu"]
    for k, kind in STAGE_MAP.items():
        out[k] = st.get(k, 0.0) * f[kind]
    out = {k: round(v, 2) for k, v in out.items()}
    out["TOTAL_projected_ms"] = round(sum(out.values()), 2)
    out["projected_fps"] = round(1000.0 / out["TOTAL_projected_ms"], 1)
    return out


def main(argv=None) -> None:
    paths = (argv or sys.argv[1:])
    for p in paths:
        t = json.load(open(p, encoding="utf-8"))
        res = {"source": p, "measured_on": t.get("gpu"), "measured_total_ms": t["stage_ms_mean_all_frames"]["TOTAL"],
               "PROJECTED_orin_nano_super": project(t, "super"),
               "PROJECTED_orin_nano_original_clocks": project(t, "orig")}
        print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
