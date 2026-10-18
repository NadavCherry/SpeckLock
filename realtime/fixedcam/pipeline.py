"""The streaming loop: one causal pass, per-stage timing, JSON-ready outputs.

    decode -> upload+grey -> motion (GPU) -> components (CPU) -> probes -> crops (GPU)
           -> classifier (TensorRT) -> tracker (CPU) -> report

Timings are measured with CUDA synchronisation around each stage, so they are stage costs, not
queueing artefacts; the total is what one un-pipelined loop costs per frame.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

from .motion import MotionStage, Proposal
from .net import OFFSETS, appearance_stats, crop_size, extract
from .tracker import Obs, Tracker, TrackerConfig, fuse


@dataclass
class FrameResult:
    index: int
    points: list            # (x, y, label, confidence, track_id, observed)
    candidates: list        # (x, y, track_id) unlabelled live tracks
    n_props: int
    n_probes: int
    moving: bool


@dataclass
class Timer:
    ms: dict = field(default_factory=dict)
    n: int = 0

    def add(self, k, dt):
        self.ms[k] = self.ms.get(k, 0.0) + 1000.0 * dt

    def report(self):
        rep = {k: round(v / max(1, self.n), 3) for k, v in self.ms.items()}
        rep["TOTAL"] = round(sum(rep.values()), 3)
        return rep

    @staticmethod
    def per_frame(d: dict, n: int) -> dict:
        return {k: round(v / max(1, n), 3) for k, v in d.items()}


class FixedCamPipeline:
    def __init__(self, classifier, height: int, width: int, osd: list | None = None,
                 tracker_cfg: TrackerConfig | None = None, motion_kw: dict | None = None,
                 sync_timing: bool = True, probe_clamp: bool = True, appearance_every: int = 0,
                 extra_weight: float = 0.0, extra_strict_births: bool = False,
                 extra_bird_veto: float | None = None, protect_tracks: bool = False, fast: bool = False):
        # fast: fasttrack.FastTracker (numpy ring state, one vectorised camera move, vectorised probe targets, lazy
        # roughness under the LOOSE config, batched feature rows) and the vectorised extra matching - same outputs
        self.fast = fast
        self.clf = classifier
        # protect_tracks: the motion stage may not mute the cells of a live drone track (a slow
        # drone fires in the same cells for many frames, like clutter)
        self.protect_tracks = protect_tracks
        # candidates from an external detector (process(..., extra=...)) are cropped and classified
        # like motion proposals; their class evidence enters as p * evidence**extra_weight. With
        # extra_strict_births a candidate that lands on no motion proposal may update a track but
        # start one only like an appearance proposal (p(drone) >= appearance_birth)
        self.extra_weight = extra_weight
        self.extra_strict_births = extra_strict_births
        self.extra_bird_veto = extra_bird_veto   # see tracker.fuse
        self.debug = False       # keep every candidate of the last frame in self.last_candidates
        self.last_candidates: list = []
        self.motion = MotionStage(height, width, osd=osd, **(motion_kw or {}))
        self.motion.profile = sync_timing
        if fast:
            from .fasttrack import FastTracker
            c = tracker_cfg or TrackerConfig()
            self.tracker = FastTracker(c, lazy_roughness=c.smooth_commit >= 1e8 and c.smooth_keep >= 1e8)
        else:
            self.tracker = Tracker(tracker_cfg)
        self.tracker.cfg.px_scale = 1080.0 / height   # speed thresholds are 1080p-equivalent
        self.timer = Timer()
        self.sync = sync_timing
        self.probe_clamp = probe_clamp
        self.appearance_every = appearance_every   # 0 = off
        self.idx = -1
        # TensorRT wants a non-default stream (the default one forces extra synchronisation)
        self.stream = torch.cuda.Stream()

    def process(self, frame_bgr: np.ndarray, decode_s: float = 0.0, extra: list | None = None) -> FrameResult:
        """``extra``: this frame's external detections as (x, y, size, evidence | None), native px."""
        with torch.cuda.stream(self.stream):
            return self._process(frame_bgr, decode_s, extra or [])

    def _t(self):
        if self.sync:
            torch.cuda.synchronize()
        return time.perf_counter()

    def _process(self, frame_bgr: np.ndarray, decode_s: float = 0.0, extra: list | None = None) -> FrameResult:
        self.idx += 1
        tm = self.timer
        tm.n += 1
        tm.add("decode", decode_s)
        t0 = self._t()
        gray = torch.from_numpy(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)).cuda(non_blocking=True)
        t1 = self._t()
        tm.add("grey+upload", t1 - t0)
        protect = ([(t.x, t.y, max(8.0, 2.0 * t.size)) for t in self.tracker.tracks if t.label == "drone"]
                   if self.protect_tracks else None)
        props, M = self.motion.step(gray, protect=protect)
        t2 = self._t()
        tm.add("motion+components", t2 - t1)
        self.tracker.predict(M)
        n_motion_props = len(props)
        if (self.appearance_every and not self.motion.moving
                and self.idx % self.appearance_every == 0):
            taken = [(p.x, p.y) for p in props] + [(t.x, t.y) for t in self.tracker.tracks]
            for q in self.motion.appearance_peaks():
                if all((q.x - x) ** 2 + (q.y - y) ** 2 > 400 for x, y in taken):
                    props.append(q)
        n_app_end = len(props)
        # external detections: a candidate on an existing proposal lends it its class evidence;
        # anywhere else it becomes a proposal of its own (a detector hit counts like motion)
        evidence: dict[int, np.ndarray] = {}
        extra_only: set[int] = set()
        if self.fast:
            from .fasttrack import match_extras
            new, evidence, extra_only = match_extras([(p.x, p.y) for p in props], extra or [], crop_size)
            props.extend(Proposal(x, y, s, 0.0) for x, y, s in new)
        for (x, y, size, ev) in ([] if self.fast else (extra or [])):
            r = max(6.0, 0.5 * crop_size(size))
            j = next((k for k, p in enumerate(props) if (p.x - x) ** 2 + (p.y - y) ** 2 <= r * r), None)
            if j is None:
                props.append(Proposal(float(x), float(y), float(size), 0.0))
                j = len(props) - 1
                extra_only.add(j)
            if ev is not None and (j not in evidence or ev[0] > evidence[j][0]):
                evidence[j] = ev
        probes = self.tracker.probe_targets([(p.x, p.y) for p in props])
        centers = [(p.x, p.y, crop_size(p.size)) for p in props]
        centers += [(t.x, t.y, crop_size(t.size)) for t in probes]
        frames = [self.motion.recent(k) for k in OFFSETS]
        x = extract(frames, centers)
        t3 = self._t()
        tm.add("crops", t3 - t2)
        probs, offs = self.clf(x)
        stats = appearance_stats(x)
        t4 = self._t()
        tm.add("classifier", t4 - t3)
        probs = probs.cpu().numpy()
        offs = offs.cpu().numpy()
        stats = stats.cpu().numpy()
        obs = []
        np_ = len(props)
        for i, (cx, cy, S) in enumerate(centers):
            p = probs[i]
            # what this observation looked like, for the track's appearance-over-time record:
            # the classifier's own opinion (before any fusion), the crop measurements, the source
            src = 2.0 if i >= np_ else (1.0 if i in extra_only else 0.0)
            feat = np.concatenate([p, stats[i], [src]])
            if i in evidence:
                p = fuse(p, evidence[i], self.extra_weight, self.extra_bird_veto)
            ox, oy = (offs[i] * S / 2.0) if p[2] < 0.5 else (0.0, 0.0)
            size = props[i].size if i < np_ else probes[i - np_].size
            if i >= np_ and self.probe_clamp:
                # a probe may nudge its track, not drag it: clamp the regressed move
                lim = max(3.0, 0.3 * size)
                mag = float(np.hypot(ox, oy))
                if mag > lim:
                    ox, oy = ox * lim / mag, oy * lim / mag
            obs.append(Obs(cx + float(ox), cy + float(oy), size, p,
                           None if i < np_ else probes[i - np_].id,
                           appearance=(n_motion_props <= i < n_app_end
                                       or (self.extra_strict_births and i in extra_only)),
                           feat=feat))
        if self.debug:   # (crop x, crop y, crop side, p_drone, p_bird, p_other, source) per candidate
            self.last_candidates = [(cx, cy, S, *probs[i], 2.0 if i >= np_ else (1.0 if i in extra_only else 0.0))
                                    for i, (cx, cy, S) in enumerate(centers)]
        self.tracker.update(obs)
        pts = self.tracker.report()
        cands = self.tracker.candidates()
        t5 = time.perf_counter()
        tm.add("tracker", t5 - t4)
        return FrameResult(self.idx, pts, cands, len(props), len(probes), self.motion.moving)
