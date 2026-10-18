"""Class-aware center scorecard: "stay on the drone's center, never call a bird a drone".

The legacy scorers (``evaluate.py``, ``metrics.py``) rank detections for AP; they ignore the
detection label and only visit frames present in the detection file. This module scores what a
tracking system *reports*, frame by frame, over every frame of the clip:

* **Center score** (headline): for each frame where a drone is visible, CLE = distance from the
  GT center to the nearest output labeled drone (infinite if there is none);
  score = mean(max(0, 1 - CLE / tau_max)). It equals the area under the precision curve
  P(CLE <= tau) for tau in [0, tau_max], so it rewards centering and charges every miss.
* **Bird-as-drone**: drone-labeled outputs that land on a GT bird (center distance within
  ``max(tau, 0.5*sqrt(area))`` of it and not on the drone). Reported per minute and as the share of
  bird-visible frames that carry one.
* **Other false alarms**: drone-labeled outputs on neither the drone nor a bird, per minute.
* **Drone-as-bird**: the share of drone-visible frames in which an output labeled bird lies on
  the drone - the system saw the drone and called it a bird.
* **Precision**: the share of drone-labeled outputs that are on the drone (one per drone per
  frame; duplicates, bird alarms and other alarms are the rest; outputs on ignore regions are
  left out).
* Box quality, on the hit frames: IoU of the nearest drone output's box with the GT box, the
  share of drone frames with a drone output at IoU >= 0.5, and the size ratio
  sqrt(output area / GT area). ``average_precision`` gives AP over the drone outputs' scores,
  matched by IoU or by center distance.
* **Ignore regions**: GT objects named ``unsure*`` / ``ignore*`` (e.g. an object the labelers could
  not classify): a drone-labeled output on one is neither credited nor charged.
* Diagnostics: hit rate at ``tau`` px, median/mean CLE over hits, frames to the first correct lock,
  lock losses, bird recall (bird-labeled outputs on GT birds).

All distances are in *1080p-equivalent pixels* (native px x 1080 / frame height), so clips of
different resolution pool fairly. GT uses :mod:`dronedet.gt` (objects named ``drone*`` - or the
legacy ``far`` - for the target, ``bird*`` for birds, anything else is an "other" mover).
Outputs use :mod:`dronedet.detections` (label ``drone*`` / ``bird*``; the box center is the point).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from .detections import DetectionSet
from .gt import GroundTruth

REF_HEIGHT = 1080.0


@dataclass
class ClipScore:
    name: str
    n_frames: int              # frames scored (excluded frames removed)
    drone_frames: int          # frames with a visible drone
    bird_frames: int           # frames with at least one visible bird
    center_sum: float          # sum over drone frames of max(0, 1 - CLE/tau_max)
    hits: int                  # drone frames with CLE <= tau
    hit_cle: list = field(default_factory=list)     # CLE of the hit frames
    bird_alarms: int = 0       # drone-labeled outputs on a bird (count of outputs)
    bird_alarm_frames: int = 0  # bird-visible frames with >= 1 such output
    other_alarms: int = 0      # drone-labeled outputs on nothing / on an "other" mover
    ignored: int = 0           # drone-labeled outputs on an unsure/ignore object (not scored)
    duplicates: int = 0        # extra drone-labeled outputs on the drone itself
    bird_boxes: int = 0        # GT bird instances (object-frames)
    bird_recalled: int = 0     # of which a bird-labeled output lies on it
    first_visible: int | None = None
    first_lock: int | None = None
    lock_losses: int = 0
    fps: float = 25.0
    drone_as_bird_frames: int = 0  # drone frames with a bird-labeled output on the drone
    true_outputs: int = 0      # drone-labeled outputs on the drone (at most one per drone)
    iou50_frames: int = 0      # drone frames with a drone output at IoU >= 0.5
    hit_iou: list = field(default_factory=list)     # IoU of the nearest output, hit frames
    hit_size_ratio: list = field(default_factory=list)  # sqrt(area ratio) output/GT, hit frames

    # ---- derived -------------------------------------------------------------------------
    @property
    def minutes(self) -> float:
        return self.n_frames / self.fps / 60.0

    def summary(self) -> dict:
        d = self.drone_frames
        med = sorted(self.hit_cle)[len(self.hit_cle) // 2] if self.hit_cle else None
        return {
            "clip": self.name,
            "frames": self.n_frames,
            "drone_frames": d,
            "center_score": round(self.center_sum / d, 4) if d else None,
            "hit_rate@tau": round(self.hits / d, 4) if d else None,
            "miss_rate": round(1 - self.hits / d, 4) if d else None,
            "median_cle_hits": round(med, 2) if med is not None else None,
            "mean_cle_hits": round(sum(self.hit_cle) / len(self.hit_cle), 2) if self.hit_cle else None,
            "bird_frames": self.bird_frames,
            "bird_as_drone": self.bird_alarms,
            "bird_as_drone_per_min": round(self.bird_alarms / self.minutes, 3) if self.minutes else None,
            "bird_frames_with_alarm": round(self.bird_alarm_frames / self.bird_frames, 4)
            if self.bird_frames else None,
            "other_false_alarms": self.other_alarms,
            "other_fa_per_min": round(self.other_alarms / self.minutes, 3) if self.minutes else None,
            "duplicates": self.duplicates,
            "ignored_outputs": self.ignored,
            "bird_recall": round(self.bird_recalled / self.bird_boxes, 4) if self.bird_boxes else None,
            "first_visible": self.first_visible,
            "lock_latency_frames": (self.first_lock - self.first_visible)
            if self.first_lock is not None and self.first_visible is not None else None,
            "lock_losses": self.lock_losses,
            "drone_as_bird_rate": round(self.drone_as_bird_frames / d, 4) if d else None,
            "precision": round(self.true_outputs / scored, 4) if (scored := self.true_outputs
                         + self.duplicates + self.bird_alarms + self.other_alarms) else None,
            "mean_iou_hits": round(sum(self.hit_iou) / len(self.hit_iou), 4) if self.hit_iou else None,
            "iou50_rate": round(self.iou50_frames / d, 4) if d else None,
            "median_size_ratio_hits": round(_median(self.hit_size_ratio), 3) if self.hit_size_ratio else None,
        }


def _median(v: list) -> float:
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def iou(a, b) -> float:
    """IoU of two corner boxes (x1, y1, x2, y2)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _corners(box) -> tuple[float, float, float, float]:
    cx, cy, w, h = box[:4]
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def _is(name: str, prefixes) -> bool:
    return any(name.startswith(p) for p in prefixes)


def _radius(box, scale: float, tau: float) -> float:
    w, h = box[2] * scale, box[3] * scale
    return max(tau, 0.5 * math.sqrt(max(0.0, w * h)))


def score_clip(gt: GroundTruth, preds: DetectionSet, n_frames: int, height: float,
               name: str = "", tau: float = 12.0, tau_max: float = 20.0, fps: float = 25.0,
               drone_prefixes=("drone", "far"), bird_prefixes=("bird",),
               ignore_prefixes=("unsure", "ignore"),
               min_loss_run: int = 5) -> ClipScore:
    """Score one clip. ``n_frames`` is the decoded length: frames with no output still count."""
    scale = REF_HEIGHT / float(height)
    excluded = set(int(f) for f in gt.meta.get("exclude_frames", []))
    drones = [o for o in gt.objects.values() if _is(o.name, drone_prefixes)]
    birds = [o for o in gt.objects.values() if _is(o.name, bird_prefixes)]
    unsure = [o for o in gt.objects.values() if _is(o.name, ignore_prefixes)]
    cs = ClipScore(name=name or gt.video, n_frames=0, drone_frames=0, bird_frames=0,
                   center_sum=0.0, hits=0, fps=fps)
    hit_state, miss_run = None, 0
    for f in range(n_frames):
        if f in excluded:
            continue
        cs.n_frames += 1
        outs = preds.frames.get(f, [])
        ddets = [d for d in outs if d.label.startswith("drone")]
        dpts = [(d.cx * scale, d.cy * scale) for d in ddets]
        bpts = [(d.cx * scale, d.cy * scale) for d in outs if d.label.startswith("bird")]
        dboxes = [o.frames[f] for o in drones if f in o.frames]
        bboxes = [o.frames[f] for o in birds if f in o.frames]
        uboxes = [o.frames[f] for o in unsure if f in o.frames]
        # --- center score on the drone(s)
        on_drone = [False] * len(dpts)
        if dboxes:
            cs.drone_frames += 1
            if cs.first_visible is None:
                cs.first_visible = f
            frame_hit = frame_as_bird = frame_iou50 = False
            for box in dboxes:
                gx, gy = box[0] * scale, box[1] * scale
                dists = [math.hypot(px - gx, py - gy) for px, py in dpts]
                cle = min(dists) if dists else math.inf
                r = _radius(box, scale, tau)
                for i, dd in enumerate(dists):
                    if dd <= r:
                        on_drone[i] = True
                cs.true_outputs += int(any(dd <= r for dd in dists))
                cs.center_sum += max(0.0, 1.0 - cle / tau_max) / len(dboxes)
                gbox = _corners(box)
                if any(iou((d.x1, d.y1, d.x2, d.y2), gbox) >= 0.5 for d in ddets):
                    frame_iou50 = True
                if cle <= tau:
                    frame_hit = True
                    cs.hit_cle.append(cle)
                    near = ddets[dists.index(cle)]
                    cs.hit_iou.append(iou((near.x1, near.y1, near.x2, near.y2), gbox))
                    if box[2] > 0 and box[3] > 0:
                        cs.hit_size_ratio.append(math.sqrt(max(0.0, near.w * near.h) / (box[2] * box[3])))
                if any(math.hypot(px - gx, py - gy) <= r for px, py in bpts):
                    frame_as_bird = True
            cs.drone_as_bird_frames += int(frame_as_bird)
            cs.iou50_frames += int(frame_iou50)
            if frame_hit:
                cs.hits += 1
                if cs.first_lock is None:
                    cs.first_lock = f
                hit_state, miss_run = True, 0
            else:
                if hit_state is True:
                    miss_run += 1
                    if miss_run == min_loss_run:
                        cs.lock_losses += 1
        # --- every drone-labeled output that is not on the drone is an alarm
        n_on = sum(on_drone)
        cs.duplicates += max(0, n_on - len(dboxes)) if dboxes else 0
        frame_bird_alarm = False
        for i, (px, py) in enumerate(dpts):
            if on_drone[i]:
                continue
            if any(math.hypot(px - b[0] * scale, py - b[1] * scale) <= _radius(b, scale, tau)
                   for b in bboxes):
                cs.bird_alarms += 1
                frame_bird_alarm = True
            elif any(math.hypot(px - b[0] * scale, py - b[1] * scale) <= _radius(b, scale, tau)
                     for b in uboxes):
                cs.ignored += 1
            else:
                cs.other_alarms += 1
        if bboxes:
            cs.bird_frames += 1
            cs.bird_alarm_frames += int(frame_bird_alarm)
            for b in bboxes:
                cs.bird_boxes += 1
                if any(math.hypot(px - b[0] * scale, py - b[1] * scale) <= _radius(b, scale, tau)
                       for px, py in bpts):
                    cs.bird_recalled += 1
    return cs


def pool(scores: list[ClipScore], name: str = "pooled") -> ClipScore:
    """Frame-weighted pooling (sums of counts; latency/lock are not poolable and are dropped)."""
    out = ClipScore(name=name, n_frames=0, drone_frames=0, bird_frames=0, center_sum=0.0,
                    hits=0, fps=scores[0].fps if scores else 25.0)
    for s in scores:
        for k in ("n_frames", "drone_frames", "bird_frames", "center_sum", "hits", "bird_alarms",
                  "bird_alarm_frames", "other_alarms", "ignored", "duplicates", "bird_boxes", "bird_recalled",
                  "lock_losses", "drone_as_bird_frames", "true_outputs", "iou50_frames"):
            setattr(out, k, getattr(out, k) + getattr(s, k))
        out.hit_cle.extend(s.hit_cle)
        out.hit_iou.extend(s.hit_iou)
        out.hit_size_ratio.extend(s.hit_size_ratio)
    return out


def average_precision(clips: list[tuple[GroundTruth, DetectionSet, int, float]], match: str = "iou",
                      thr: float = 0.5, tau: float = 12.0,
                      drone_prefixes=("drone", "far")) -> float | None:
    """AP of the drone-labeled outputs over one or more clips (all-point interpolation).

    ``clips`` holds (gt, preds, n_frames, height). Outputs are ranked by score; each takes the
    best unmatched GT drone in its frame, by IoU >= ``thr`` (``match="iou"``) or by center
    distance <= max(tau, 0.5*sqrt(area)) in 1080p-equivalent px (``match="center"``).
    Excluded frames are skipped. None when no drone is visible.
    """
    cands, n_gt, gts = [], 0, {}
    for c, (gt, preds, n_frames, height) in enumerate(clips):
        scale = REF_HEIGHT / float(height)
        excluded = set(int(f) for f in gt.meta.get("exclude_frames", []))
        drones = [o for o in gt.objects.values() if _is(o.name, drone_prefixes)]
        for f in range(n_frames):
            if f in excluded:
                continue
            boxes = [o.frames[f] for o in drones if f in o.frames]
            n_gt += len(boxes)
            gts[(c, f)] = (boxes, scale, [False] * len(boxes))
            for d in preds.frames.get(f, []):
                if d.label.startswith("drone"):
                    cands.append((d.score, c, f, d))
    if n_gt == 0:
        return None
    cands.sort(key=lambda t: -t[0])
    tp = []
    for _, c, f, d in cands:
        boxes, scale, used = gts[(c, f)]
        best, best_j = -1.0, -1
        for j, b in enumerate(boxes):
            if used[j]:
                continue
            if match == "iou":
                q = iou((d.x1, d.y1, d.x2, d.y2), _corners(b))
                ok = q >= thr
            else:
                dist = math.hypot((d.cx - b[0]) * scale, (d.cy - b[1]) * scale)
                q, ok = -dist, dist <= _radius(b, scale, tau)
            if ok and q > best:
                best, best_j = q, j
        if best_j >= 0:
            used[best_j] = True
        tp.append(best_j >= 0)
    # precision envelope over recall steps
    ap, n_tp, prec = 0.0, 0, []
    for k, t in enumerate(tp, 1):
        n_tp += t
        prec.append((n_tp / k, n_tp / n_gt, t))
    best_after = 0.0
    for p, r, t in reversed(prec):
        best_after = max(best_after, p)
        if t:
            ap += best_after / n_gt
    return round(ap, 4)


def as_dict(cs: ClipScore) -> dict:
    d = asdict(cs)
    d.pop("hit_cle")
    return d
