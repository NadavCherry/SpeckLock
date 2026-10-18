"""Evidence-accumulating multi-target tracker (numpy + scipy only; unit-tested in CI).

Why evidence, not a rule: a 6 px drone and a 6 px bird are often indistinguishable in one frame,
but a classifier that is right 70 % of the time per frame is right almost always over 15 frames.
Each track keeps a forgetting sum of per-frame class log-probabilities (drone / bird / other); its
label is the posterior's argmax once the posterior clears ``commit`` - and nothing else (the old
``trackclass`` let any track longer than 120 frames become a drone with no evidence; a long-lived
bird walked through that door).

Per frame the caller passes observations (classified proposals and classified *probes* - crops
at the predicted position of a live track, which is how a hovering drone that makes no motion
stays locked) and the camera map M (current px -> previous px). Positions are native pixels.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

DRONE, BIRD, OTHER = 0, 1, 2
LABELS = ("drone", "bird", "other")


@dataclass
class Obs:
    x: float
    y: float
    size: float
    p: np.ndarray               # (3,) class probabilities
    probe_of: int | None = None  # track id if this is a probe crop
    appearance: bool = False     # an appearance proposal (no motion): may only start a track
                                 # when it looks strongly like a drone
    feat: np.ndarray | None = None  # per-observation vector, trackfeat.OBS_FIELDS (optional)


@dataclass
class Track:
    id: int
    x: float
    y: float
    size: float
    vx: float = 0.0
    vy: float = 0.0
    logp: np.ndarray = field(default_factory=lambda: np.zeros(3))
    n_obs: int = 0
    misses: int = 0
    label: str = "unknown"
    observed: bool = False       # observed this frame
    age: int = 0
    n_motion: int = 0            # observations that came from a motion proposal (not a probe)
    hist: deque = field(default_factory=lambda: deque(maxlen=160))  # (age, x, y) of observations
    flown: bool = False          # has shown a stretch of directed flight (see TrackerConfig.flight_turn)
    appear: deque = field(default_factory=lambda: deque(maxlen=64))  # (age, Obs.feat) over time

    def heading_stats(self, n: int = 15) -> tuple[float, float]:
        """(median absolute turn in radians, median speed in native px/frame) over the last n
        observations; (pi, 0) when there are fewer than 3."""
        h = list(self.hist)[-n:]
        if len(h) < 3:
            return math.pi, 0.0
        a = np.array(h, float)
        v = np.diff(a[:, 1:], axis=0)
        dt = np.maximum(1.0, np.diff(a[:, 0]))
        speed = np.hypot(v[:, 0], v[:, 1]) / dt
        ang = np.arctan2(v[:, 1], v[:, 0])
        turn = np.abs(np.angle(np.exp(1j * np.diff(ang))))
        return float(np.median(turn)), float(np.median(speed))

    def roughness(self, n: int = 20) -> float | None:
        """RMS residual of a quadratic-in-time fit to the last n observations (None if too few).
        A flier traces a smooth path (lines, loops, hover jitter); a track hopping between
        clutter blobs does not - its residual is large even when it covers a lot of ground."""
        h = list(self.hist)[-n:]
        if len(h) < 8:
            return None
        t = np.array([q[0] for q in h], float)
        t -= t.mean()
        res = []
        for k in (1, 2):
            v = np.array([q[k] for q in h], float)
            coef = np.polyfit(t, v, 2)
            res.append(v - np.polyval(coef, t))
        return float(np.sqrt(np.mean(np.square(res[0]) + np.square(res[1]))))

    def moved(self, window: int | None = None) -> float:
        """Largest distance between the latest observation and any observation within `window`."""
        if len(self.hist) < 2:
            return 0.0
        a, x, y = self.hist[-1]
        return max(math.hypot(x - hx, y - hy) for ha, hx, hy in self.hist
                   if window is None or a - ha <= window)

    def posterior(self) -> np.ndarray:
        e = self.logp - self.logp.max()
        p = np.exp(e)
        return p / p.sum()


@dataclass
class TrackerConfig:
    forget: float = 0.96        # evidence half-life ~17 frames
    p_floor: float = 0.02
    temper: float = 1.0         # each frame adds temper * log p: < 1 keeps one confident frame from
                                # cancelling several (the crops of one target are far from independent)
    probe_accept_other: float = 0.5  # a probe updates its track only while its p(other) is below this;
                                     # raise it to hold a confirmed drone that hovers (no trail -> "other")
    min_obs: int = 4
    commit: float = 0.90        # posterior needed to label a track
    flip: float = 0.98          # posterior another class needs to relabel a labelled track
    revert: float = 0.30        # a labelled track whose own class falls below this is unlabelled
    new_max_other: float = 0.7  # a motion obs starts a track only if p(other) < this
    appearance_birth: float = 0.8  # an appearance proposal starts a track only if p(drone) >= this
    beta: float = 0.3
    max_miss_drone: int = 30
    max_miss: int = 8
    coast_report: int = 3       # report a drone for this many unobserved frames (predicted)
    probe_min_drone: float = 0.35
    # "nothing is a drone until it has been watched flying" (the pursuit ring's rule): a static
    # clutter point that a few compression flickers once made drone-like must not stay a drone
    # because probe crops of it keep looking drone-like
    min_motion_obs: int = 4      # motion (not probe) observations before any label
    min_move: float = 4.0        # px moved over its life before a drone label
    hover_window: int = 75       # a drone must keep moving at least...
    hover_move: float = 3.0      # ...this many px within this window (a hovering drone drifts)
    smooth_commit: float = 0.25  # drone label needs path roughness <= max(1.5, this * size)
    smooth_keep: float = 0.6     # a drone whose roughness exceeds max(3, this * size) is dropped
    # heading: a flier holds its direction from step to step; clutter jittering in place, or a
    # track chained across candidates, turns at random. Both gates act on reporting only (a
    # gated drone track is shown as an unconfirmed candidate) and are off by default.
    heading_gate: float | None = None   # report a drone only while the median turn over the last
                                        # heading_window observations is <= this (radians)
    flight_turn: float | None = None    # report a drone only once it has flown: some window with
    flight_speed: float = 1.0           # median turn <= flight_turn and median speed >= flight_speed
    heading_window: int = 15            # (1080p-equivalent px per frame); it may hover afterwards
    px_scale: float = 1.0               # native px -> 1080p-equivalent px (set by the pipeline)
    # appearance over time: a learned track classifier (trackfeat.TrackClassifier JSON) must give
    # a drone track p >= track_clf_min before it is reported; off by default
    track_clf: str | None = None
    track_clf_min: float = 0.5


def det_evidence(label: str, score: float) -> np.ndarray | None:
    """An external detector's opinion as (drone, bird, other) class evidence.

    ``drone*`` with confidence s -> (s, (1-s)/2, (1-s)/2); ``bird*`` -> ((1-s)/2, s, (1-s)/2); any other
    label (e.g. a motion-only candidate) carries no class opinion -> None. s is clipped to
    [0.05, 0.95] so no single detection can veto a class.
    """
    s = float(min(0.95, max(0.05, score)))
    rest = (1.0 - s) / 2.0
    if label.startswith("drone"):
        return np.array([s, rest, rest])
    if label.startswith("bird"):
        return np.array([rest, s, rest])
    return None


def fuse(p: np.ndarray, evidence: np.ndarray | None, weight: float,
         bird_veto: float | None = None) -> np.ndarray:
    """Product of experts: p * evidence**weight, renormalised (weight 0 or no evidence -> p).

    ``bird_veto``: when the evidence says drone but the crop classifier gives the crop
    p(bird) >= bird_veto, the evidence is ignored - an external "drone" never overrules a bird the
    classifier sees (external detectors call birds drones far more often than the reverse).
    """
    if evidence is None or weight <= 0:
        return p
    if bird_veto is not None and int(np.argmax(evidence)) == DRONE and p[BIRD] >= bird_veto:
        return p
    q = np.asarray(p, float) * np.power(evidence, weight)
    return q / q.sum()


def _apply(M: np.ndarray, x: float, y: float) -> tuple[float, float]:
    v = M @ np.array([x, y, 1.0])
    return float(v[0] / v[2]), float(v[1] / v[2])


class Tracker:
    def __init__(self, cfg: TrackerConfig | None = None):
        self.cfg = cfg or TrackerConfig()
        self.tracks: list[Track] = []
        self.next_id = 0
        self.clf = None
        if self.cfg.track_clf:
            from .trackfeat import TrackClassifier

            self.clf = TrackClassifier(self.cfg.track_clf)

    def features(self, tr: Track) -> np.ndarray:
        from .trackfeat import track_features

        return track_features(tr.appear, tr.hist, tr.n_obs, tr.age, tr.size, self.cfg.px_scale)

    def feature_rows(self) -> list[tuple[int, float, float, str, np.ndarray]]:
        """(id, x, y, label, features) of every observed track with enough evidence - the rows a
        track classifier is trained on."""
        return [(t.id, t.x, t.y, t.label, self.features(t)) for t in self.tracks
                if t.observed and t.n_obs >= self.cfg.min_obs and t.appear]

    # ---- helpers the pipeline uses -----------------------------------------------------------
    def gate(self, tr: Track) -> float:
        speed = math.hypot(tr.vx, tr.vy)
        return max(14.0, 3.0 * speed + tr.size) + 6.0 * min(tr.misses, 5)

    def probe_targets(self, proposals_xy: list[tuple[float, float]]) -> list[Track]:
        """Live tracks worth a probe crop this frame: promising and with no proposal near their
        predicted position (so the probe is the only way to observe them)."""
        out = []
        for tr in self.tracks:
            post = tr.posterior() if tr.n_obs else np.array([1 / 3] * 3)
            if tr.label != "drone" and post[DRONE] < self.cfg.probe_min_drone:
                continue
            g = 0.6 * self.gate(tr)
            if any(math.hypot(px - tr.x, py - tr.y) < g for px, py in proposals_xy):
                continue
            out.append(tr)
        return out

    def predict(self, M: np.ndarray | None) -> None:
        """Advance every track one frame; carry it through a camera move (M: current -> previous)."""
        Minv = None
        if M is not None and not np.allclose(M, np.eye(3), atol=1e-3):
            Minv = np.linalg.inv(M)
        for tr in self.tracks:
            if Minv is not None:
                nx, ny = _apply(Minv, tr.x, tr.y)
                ex, ey = _apply(Minv, tr.x + tr.vx, tr.y + tr.vy)
                tr.x, tr.y, tr.vx, tr.vy = nx, ny, ex - nx, ey - ny
                # keep the observation history in the current view (movement is world movement)
                tr.hist = deque(((a,) + _apply(Minv, hx, hy) for a, hx, hy in tr.hist), maxlen=160)
            tr.x += tr.vx
            tr.y += tr.vy
            tr.observed = False
            tr.age += 1

    # ---- the per-frame update ----------------------------------------------------------------
    def update(self, obs: list[Obs]) -> None:
        c = self.cfg
        motion = [o for o in obs if o.probe_of is None]
        probes = {o.probe_of: o for o in obs if o.probe_of is not None}
        # 1. Hungarian on motion observations
        matched_t, matched_o = set(), set()
        if self.tracks and motion:
            T = np.array([[t.x, t.y] for t in self.tracks])
            G = np.array([self.gate(t) for t in self.tracks])
            O = np.array([[o.x, o.y] for o in motion])
            D = np.hypot(T[:, None, 0] - O[None, :, 0], T[:, None, 1] - O[None, :, 1])
            cost = np.where(D <= G[:, None], D / G[:, None], 1e6)
            ri, ci = linear_sum_assignment(cost)
            for i, j in zip(ri, ci):
                if cost[i, j] < 1e5:
                    self._observe(self.tracks[i], motion[j])
                    matched_t.add(i)
                    matched_o.add(j)
        # 2. probes update their own track (only if the crop still looks like the target)
        for i, tr in enumerate(self.tracks):
            if i in matched_t or tr.id not in probes:
                continue
            o = probes[tr.id]
            limit = c.probe_accept_other if tr.label == "drone" else 0.5
            if o.p[OTHER] < limit and math.hypot(o.x - tr.x, o.y - tr.y) <= self.gate(tr):
                self._observe(tr, o)
                matched_t.add(i)
        # 3. misses
        keep = []
        for i, tr in enumerate(self.tracks):
            if i not in matched_t:
                tr.misses += 1
            limit = c.max_miss_drone if tr.label == "drone" else c.max_miss
            if tr.misses <= limit:
                keep.append(tr)
            else:
                self._drop(tr)
        self.tracks = keep
        # 4. births from unmatched motion observations
        for j, o in enumerate(motion):
            if j in matched_o or o.p[OTHER] >= c.new_max_other:
                continue
            if o.appearance and o.p[DRONE] < c.appearance_birth:
                continue
            if any(math.hypot(o.x - t.x, o.y - t.y) < 0.5 * self.gate(t) for t in self.tracks):
                continue
            tr = self._new_track(o.x, o.y, o.size)
            self._observe(tr, o, birth=True)
            self.tracks.append(tr)

    def _new_track(self, x: float, y: float, size: float) -> Track:
        """A new track with the next id (a subclass may keep its state elsewhere: fasttrack.FastTracker)."""
        tr = Track(self.next_id, x, y, size)
        self.next_id += 1
        return tr

    def _drop(self, tr: Track) -> None:
        """Called once for every track that update() ends (no-op here; fasttrack.FastTracker frees its ring slot)."""

    def _observe(self, tr: Track, o: Obs, birth: bool = False) -> None:
        c = self.cfg
        if not birth:
            # the reported point is the observation itself (the centre head is accurate); the
            # innovation only corrects the velocity that drives prediction and gating
            tr.vx += c.beta * (o.x - tr.x)
            tr.vy += c.beta * (o.y - tr.y)
            tr.x, tr.y = o.x, o.y
        tr.size = 0.7 * tr.size + 0.3 * o.size if tr.n_obs else o.size
        p = np.clip(np.asarray(o.p, float), c.p_floor, 1.0)
        tr.logp = c.forget * tr.logp + c.temper * np.log(p / p.sum())
        tr.n_obs += 1
        tr.n_motion += int(o.probe_of is None)
        tr.misses = 0
        tr.observed = True
        tr.hist.append((tr.age, tr.x, tr.y))
        if o.feat is not None:
            tr.appear.append((tr.age, o.feat))
        if c.flight_turn is not None and not tr.flown and len(tr.hist) >= c.heading_window:
            turn, speed = tr.heading_stats(c.heading_window)
            tr.flown = turn <= c.flight_turn and speed * c.px_scale >= c.flight_speed
        self._relabel(tr)

    def _gated(self, tr: Track) -> bool:
        """A drone-labelled track the heading rules keep from being reported."""
        c = self.cfg
        if tr.label != "drone":
            return False
        if c.flight_turn is not None and not tr.flown:
            return True
        if c.heading_gate is not None and tr.heading_stats(c.heading_window)[0] > c.heading_gate:
            return True
        return self.clf is not None and self.clf(self.features(tr)) < c.track_clf_min

    def _relabel(self, tr: Track) -> None:
        c = self.cfg
        if tr.n_obs < c.min_obs or tr.n_motion < c.min_motion_obs:
            return
        post = tr.posterior()
        k = int(post.argmax())
        if tr.label == "drone" and tr.age >= c.hover_window and tr.moved(c.hover_window) < c.hover_move:
            tr.label = "unknown"   # not moving at all for 3 s: static clutter, not a hovering drone
            return
        rough = tr.roughness()
        if tr.label == "drone" and rough is not None and rough > max(3.0, c.smooth_keep * tr.size):
            tr.label = "unknown"   # hopping between blobs, not flying
            return
        if tr.label == "unknown":
            smooth_ok = rough is not None and rough <= max(1.5, c.smooth_commit * tr.size)
            if post[k] >= c.commit and (k != 0 or (tr.moved(c.hover_window) >= c.min_move and smooth_ok)):
                tr.label = LABELS[k]
        elif LABELS[k] != tr.label and post[k] >= c.flip:
            tr.label = LABELS[k]
        elif post[LABELS.index(tr.label)] < c.revert:
            tr.label = "unknown"   # the evidence that earned the label has decayed away

    # ---- output -----------------------------------------------------------------------------
    def report(self) -> list[tuple[float, float, str, float, int, bool]]:
        """(x, y, label, confidence, track_id, observed) for tracks worth drawing/scoring."""
        out = []
        for tr in self.tracks:
            if tr.label == "unknown" or tr.label == "other":
                continue
            if not tr.observed and tr.misses > self.cfg.coast_report:
                continue
            if self._gated(tr):
                continue
            post = tr.posterior()
            out.append((tr.x, tr.y, tr.label, float(post[LABELS.index(tr.label)]), tr.id, tr.observed))
        return out

    def candidates(self) -> list[tuple[float, float, int]]:
        """Unlabelled live tracks, and drone tracks the heading rules hold back (drawn grey)."""
        return [(t.x, t.y, t.id) for t in self.tracks
                if t.observed and (t.label == "unknown" or self._gated(t))]
