"""Fixed / PTZ-camera drone-vs-bird pipeline for edge hardware (Jetson Orin Nano class).

Why a new pipeline: the fixed / PTZ cameras this targets are *still* most of the time (a PTZ
head re-points now and then; designed for 1080p-1440p at 25 fps), and the task is to hold the
drone's CENTER while never calling a bird a drone. A still camera makes motion cheap and strong,
so the expensive part of the old edge path - a full-frame network at 1280 px - is replaced by:

    motion   -- GPU lagged-median background at native resolution, per-pixel noise and
                activity maps (wind-blown vegetation and cloud edges suppress themselves),
                a camera-motion gate that estimates a homography only while the head moves
    net      -- native-resolution temporal crops (t-6, t-3, t) around each proposal and around
                every live track's predicted position (a hovering drone makes no motion), and
                a tiny CNN on 32x32 crops: drone / bird / other + the drone's center offset;
                exported to ONNX and run as a static-batch TensorRT FP16 engine
    tracker  -- numpy multi-target tracker that accumulates class evidence per track and
                labels a track drone or bird only when the evidence says so (no length bypass)

Conventions: all positions are native-frame pixels (x right, y down). Only ``tracker``,
``fasttrack`` (its exact-output fast path) and ``trackfeat`` are torch-free (they are unit-tested
in CI); ``motion``, ``net`` (also torchvision), ``pipeline`` and ``trt`` import torch at module
level, so import them only where torch exists (``trt.build_engine`` also needs tensorrt).
"""

__all__ = ["tracker"]
