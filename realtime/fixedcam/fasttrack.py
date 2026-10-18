"""Exact-output fast path of the fixed-camera tracker (numpy + scipy only; unit-tested in CI).

``FastTracker`` is ``tracker.Tracker`` with the same outputs - the same tracks, ids, labels and positions, and
feature rows equal up to floating-point summation order (~1e-15) - at a fraction of the per-frame Python cost:

- per-track numpy ring state: every live track's observation history (age, x, y; 160 deep, as the deque) and
  appearance record (age + ``trackfeat.OBS_FIELDS``; the last 32, all that ``track_features`` reads) live in one
  ``RingPool``, one slot per track. ``Track.hist`` / ``Track.appear`` become ``RingView`` objects with the deque
  interface the stock code uses (append, len, iteration, indexing);
- one vectorised camera move per frame: positions, velocities and every history entry of every track go through
  the inverse homography in one batched call. It is the stock per-point ``_apply`` bit for bit: numpy runs the same
  BLAS gemv per point inside one C loop (``apply_homography``; a plain 3x3 @ 3xN product or an elementwise
  formula would differ in the last bit, because the BLAS kernel fuses multiply-adds);
- vectorised probe targets (posteriors, gates and the track x proposal distances as arrays; a distance within
  1e-9 of its gate is re-decided with math.hypot, so every decision is the stock one);
- lazy path roughness for the LOOSE configuration (smooth_commit = smooth_keep >= 1e8): there the roughness only
  matters as None / not None, so its two polyfits are skipped. Any other configuration computes the stock polyfit
  from the ring (``lazy_roughness=False``; asserted when True);
- ``feature_rows()`` for all rows at once from the rings (``trackfeat.track_features_padded``): no deque -> array
  conversion.

``match_extras`` is FixedCamPipeline's external-candidate matching, vectorised with the same first-match order and
an exact re-check of borderline distances.
"""

from __future__ import annotations

import math

import numpy as np

from .trackfeat import OBS_FIELDS, track_features, track_features_padded
from .tracker import Track, Tracker, TrackerConfig

HIST_LEN = 160          # Track.hist's deque length
APPEAR_LEN = 32         # track_features' window (Track.appear keeps 64, only the last 32 are ever read)
KIN_ROWS = 60           # 4 * track_features' kin_window: the history rows the features read
_EPS = 1e-9             # relative band within which a vectorised comparison is re-decided exactly


def apply_homography(Minv: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """tracker._apply for many points: (n, 2) -> (n, 2), bit-identical to the per-point call (one BLAS gemv per
    point, as ``Minv @ np.array([x, y, 1.0])`` runs, inside numpy's stacked-matmul loop)."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    if len(xy) == 0:
        return np.zeros((0, 2))
    P = np.empty((len(xy), 3, 1))
    P[:, 0, 0] = xy[:, 0]
    P[:, 1, 0] = xy[:, 1]
    P[:, 2, 0] = 1.0
    V = np.matmul(np.ascontiguousarray(Minv, dtype=float), P)[:, :, 0]
    return np.stack([V[:, 0] / V[:, 2], V[:, 1] / V[:, 2]], axis=1)


class RingPool:
    """Ring buffers of all live tracks, one slot each: hist (cap, HIST_LEN, 3) = (age, x, y) and
    app (cap, APPEAR_LEN, 1 + len(OBS_FIELDS)) = (age, obs vector). Counts and write positions live on the tracks
    (Python ints); a freed slot is zeroed when it is handed out again, so unused entries always read 0."""

    def __init__(self, cap: int = 256, hist_len: int = HIST_LEN, app_len: int = APPEAR_LEN):
        self.HL, self.AL, self.NF = hist_len, app_len, len(OBS_FIELDS)
        self.hist = np.zeros((cap, hist_len, 3))
        self.app = np.zeros((cap, app_len, 1 + self.NF))
        self.free = list(range(cap - 1, -1, -1))

    @property
    def cap(self) -> int:
        return len(self.hist)

    def alloc(self) -> int:
        if not self.free:
            n = self.cap
            self.hist = np.concatenate([self.hist, np.zeros_like(self.hist)])
            self.app = np.concatenate([self.app, np.zeros_like(self.app)])
            self.free = list(range(2 * n - 1, n - 1, -1))
        s = self.free.pop()
        self.hist[s] = 0.0
        self.app[s] = 0.0
        return s

    def release(self, s: int) -> None:
        self.free.append(s)

    # ---- windows for the features: the stack_windows layout, straight from the rings
    def windows(self, tracks, window: int = APPEAR_LEN, kin_rows: int = KIN_ROWS):
        """-> A (n, window, NF) left-aligned with la valid rows (>= 1), H (n, kin_rows, 3) right-aligned with lh."""
        sl = np.fromiter((t.slot for t in tracks), np.int64, len(tracks))
        an = np.fromiter((t.an for t in tracks), np.int64, len(tracks))
        ap = np.fromiter((t.ap for t in tracks), np.int64, len(tracks))
        hn = np.fromiter((t.hn for t in tracks), np.int64, len(tracks))
        hp = np.fromiter((t.hp for t in tracks), np.int64, len(tracks))
        la = np.minimum(an, window)
        ia = (ap[:, None] - la[:, None] + np.arange(window)[None, :]) % self.AL     # oldest valid first
        A = self.app[sl[:, None], ia, 1:]
        A[np.arange(window)[None, :] >= la[:, None]] = 0.0
        la = np.maximum(la, 1)                                                     # empty -> one zero row
        lh = np.minimum(hn, kin_rows)
        ih = (hp[:, None] - kin_rows + np.arange(kin_rows)[None, :]) % self.HL     # newest last
        H = self.hist[sl[:, None], ih]
        H[np.arange(kin_rows)[None, :] < (kin_rows - lh)[:, None]] = 0.0
        return A, la, H, lh

    # ---- one camera move for every history entry
    def transform(self, Minv: np.ndarray, tracks) -> None:
        if not tracks:
            return
        sl = np.fromiter((t.slot for t in tracks), np.int64, len(tracks))
        hn = np.fromiter((t.hn for t in tracks), np.int64, len(tracks))
        hp = np.fromiter((t.hp for t in tracks), np.int64, len(tracks))
        N = int(hn.sum())
        if N == 0:
            return
        start = np.repeat(np.cumsum(hn) - hn, hn)
        k = np.arange(N) - start
        si = np.repeat(sl, hn)
        ri = (np.repeat(hp - hn, hn) + k) % self.HL
        self.hist[si, ri, 1:] = apply_homography(Minv, self.hist[si, ri, 1:])


class RingView:
    """The deque interface Tracker uses on Track.hist / Track.appear, over a track's ring in a RingPool.
    kind 'hist': items (age, x, y); kind 'appear': items (age, obs vector)."""

    __slots__ = ("tr", "kind")

    def __init__(self, tr: "RingTrack", kind: str):
        self.tr, self.kind = tr, kind

    def _rows(self) -> np.ndarray:
        t, p = self.tr, self.tr.pool
        if self.kind == "hist":
            n, pos, L, buf = t.hn, t.hp, p.HL, p.hist
        else:
            n, pos, L, buf = t.an, t.ap, p.AL, p.app
        idx = (pos - n + np.arange(n)) % L
        return buf[t.slot, idx]

    def append(self, item) -> None:
        t, p = self.tr, self.tr.pool
        if self.kind == "hist":
            p.hist[t.slot, t.hp] = item
            t.hp = (t.hp + 1) % p.HL
            if t.hn < p.HL:
                t.hn += 1
        else:
            row = p.app[t.slot, t.ap]
            row[0] = item[0]
            row[1:] = item[1]
            t.ap = (t.ap + 1) % p.AL
            if t.an < p.AL:
                t.an += 1

    def __len__(self) -> int:
        return self.tr.hn if self.kind == "hist" else self.tr.an

    def __bool__(self) -> bool:
        return len(self) > 0

    def __iter__(self):
        R = self._rows()
        if self.kind == "hist":
            return iter([(int(a), x, y) for a, x, y in R.tolist()])
        return iter([(int(r[0]), np.array(r[1:])) for r in R])

    def __getitem__(self, i):
        return list(self)[i]

    def array(self, n: int | None = None) -> np.ndarray:
        """The last n rows (all when None), oldest first: (k, 3) for hist, (k, 1 + NF) for appear (a copy)."""
        R = self._rows()
        return R if n is None else R[max(0, len(R) - n):]


class RingTrack(Track):
    """A Track whose hist / appear live in a RingPool (FastTracker makes them)."""

    def __init__(self, id: int, x: float, y: float, size: float, pool: RingPool, lazy: bool):
        super().__init__(id, x, y, size)
        self.pool, self.lazy = pool, lazy
        self.slot = pool.alloc()
        self.hn = self.hp = self.an = self.ap = 0
        self.hist = RingView(self, "hist")
        self.appear = RingView(self, "appear")

    def heading_stats(self, n: int = 15) -> tuple[float, float]:
        if self.hn < 3 or n < 3:                       # (and list[-n:] edge cases) the stock code on the view
            return Track.heading_stats(self, n)
        a = self.hist.array(n)
        v = np.diff(a[:, 1:], axis=0)
        dt = np.maximum(1.0, np.diff(a[:, 0]))
        speed = np.hypot(v[:, 0], v[:, 1]) / dt
        ang = np.arctan2(v[:, 1], v[:, 0])
        turn = np.abs(np.angle(np.exp(1j * np.diff(ang))))
        return float(np.median(turn)), float(np.median(speed))

    def roughness(self, n: int = 20) -> float | None:
        if n <= 0:                                     # list[-n:] edge cases: the stock code on the view
            r = Track.roughness(self, n)
            return (None if r is None else 0.0) if self.lazy else r
        if self.lazy:                    # LOOSE: only None / not None matters (FastTracker asserts the config)
            return None if min(self.hn, n) < 8 else 0.0
        if min(self.hn, n) < 8:
            return None
        h = self.hist.array(n)
        t = np.array(h[:, 0], float)
        t -= t.mean()
        res = []
        for k in (1, 2):
            v = np.array(h[:, k], float)
            coef = np.polyfit(t, v, 2)
            res.append(v - np.polyval(coef, t))
        return float(np.sqrt(np.mean(np.square(res[0]) + np.square(res[1]))))

    def moved(self, window: int | None = None) -> float:
        if self.hn < 2:
            return 0.0
        R = self.hist.array()
        a, x, y = R[-1].tolist()
        if window is not None:
            R = R[a - R[:, 0] <= window]
        return max(math.hypot(x - hx, y - hy) for _, hx, hy in R.tolist())


class FastTracker(Tracker):
    """Tracker with numpy ring state, one vectorised camera move per frame, vectorised probe targets, lazy roughness
    (LOOSE only) and batched feature rows. Same outputs as Tracker (see the module docstring)."""

    def __init__(self, cfg: TrackerConfig | None = None, lazy_roughness: bool = True, cap: int = 256):
        super().__init__(cfg)
        c = self.cfg
        if lazy_roughness:
            assert c.smooth_commit >= 1e8 and c.smooth_keep >= 1e8, "lazy roughness is exact only for the LOOSE config"
        self.lazy = lazy_roughness
        self.pool = RingPool(cap)

    def _new_track(self, x: float, y: float, size: float) -> Track:
        tr = RingTrack(self.next_id, x, y, size, self.pool, self.lazy)
        self.next_id += 1
        return tr

    def _drop(self, tr: Track) -> None:
        self.pool.release(tr.slot)

    def predict(self, M: np.ndarray | None) -> None:
        Minv = None
        if M is not None and not np.allclose(M, np.eye(3), atol=1e-3):
            Minv = np.linalg.inv(M)
        ts = self.tracks
        if Minv is not None and ts:
            n = len(ts)
            P = np.array([(t.x, t.y) for t in ts] + [(t.x + t.vx, t.y + t.vy) for t in ts], float)
            V = apply_homography(Minv, P).tolist()
            for i, tr in enumerate(ts):
                nx, ny = V[i]
                ex, ey = V[n + i]
                tr.x, tr.y, tr.vx, tr.vy = nx, ny, ex - nx, ey - ny
            self.pool.transform(Minv, ts)
        for tr in ts:
            tr.x += tr.vx
            tr.y += tr.vy
            tr.observed = False
            tr.age += 1

    def probe_targets(self, proposals_xy: list[tuple[float, float]]) -> list[Track]:
        ts = self.tracks
        if not ts:
            return []
        L = np.array([t.logp for t in ts], float)
        P = np.exp(L - L.max(1, keepdims=True))
        P = P / P.sum(1, keepdims=True)
        nobs = np.fromiter((t.n_obs for t in ts), np.int64, len(ts))
        pd = np.where(nobs > 0, P[:, 0], 1.0 / 3.0)
        drone = np.fromiter((t.label == "drone" for t in ts), bool, len(ts))
        pmin = self.cfg.probe_min_drone
        low = pd < pmin
        for i in np.nonzero((nobs > 0) & (np.abs(pd - pmin) <= _EPS * max(abs(pmin), 1e-300)))[0]:
            low[i] = ts[i].posterior()[0] < pmin      # borderline: the stock per-track posterior decides
        ok = drone | ~low
        G = np.array([0.6 * self.gate(t) for t in ts])
        if proposals_xy:
            Q = np.asarray(proposals_xy, float).reshape(-1, 2)
            T = np.array([(t.x, t.y) for t in ts], float)
            D = np.hypot(T[:, None, 0] - Q[None, :, 0], T[:, None, 1] - Q[None, :, 1])
            close = D < G[:, None]
            amb = np.abs(D - G[:, None]) <= _EPS * G[:, None]
            for i, j in zip(*np.nonzero(amb)):          # re-decide borderline pairs exactly as Tracker does
                close[i, j] = math.hypot(Q[j, 0] - ts[i].x, Q[j, 1] - ts[i].y) < G[i]
            near = close.any(1)
        else:
            near = np.zeros(len(ts), bool)
        return [t for t, o, nr in zip(ts, ok, near) if o and not nr]

    # ---- features straight from the rings
    def features(self, tr: Track) -> np.ndarray:
        ap = [(0, r[1:]) for r in tr.appear.array()]
        return track_features(ap, tr.hist.array(), tr.n_obs, tr.age, tr.size, self.cfg.px_scale)

    def feature_rows(self) -> list[tuple[int, float, float, str, np.ndarray]]:
        live = [t for t in self.tracks if t.observed and t.n_obs >= self.cfg.min_obs and t.an > 0]
        if not live:
            return []
        X = self.feature_matrix(live)
        return [(t.id, t.x, t.y, t.label, X[i]) for i, t in enumerate(live)]

    def feature_matrix(self, live) -> np.ndarray:
        A, la, H, lh = self.pool.windows(live)
        return track_features_padded(A, la, H, lh, [t.n_obs for t in live], [t.age for t in live],
                                     [t.size for t in live], self.cfg.px_scale)


def match_extras(props_xy, extras, crop_size):
    """FixedCamPipeline's external-candidate matching. props_xy: the frame's proposals (x, y) so far, in order;
    extras: [(x, y, size, evidence | None, ...)]. Each extra lends its evidence to the FIRST proposal within
    r = max(6, crop_size(size) / 2) px - proposals appended for earlier extras included - or becomes a proposal of its
    own. Returns (new proposals [(x, y, size)] appended in order, {proposal index: evidence}, set of extra-only
    proposal indices). Distances are vectorised; a squared distance within 1e-9 of r^2 is re-decided with the
    pipeline's own expression."""
    n0 = len(props_xy)
    evidence: dict = {}
    extra_only: set = set()
    new: list = []
    if not extras:
        return new, evidence, extra_only
    EX = np.array([(e[0], e[1]) for e in extras], float)
    R = [max(6.0, 0.5 * crop_size(e[2])) for e in extras]
    first = np.full(len(extras), -1)
    if n0:
        PX = np.asarray(props_xy, float).reshape(-1, 2)
        R2 = np.array(R) ** 2
        D2 = (PX[None, :, 0] - EX[:, None, 0]) ** 2 + (PX[None, :, 1] - EX[:, None, 1]) ** 2
        hit = D2 <= R2[:, None]
        amb = np.abs(D2 - R2[:, None]) <= _EPS * R2[:, None]
        for q, k in zip(*np.nonzero(amb)):
            px, py = props_xy[k]
            x, y = extras[q][0], extras[q][1]
            hit[q, k] = (px - x) ** 2 + (py - y) ** 2 <= R[q] * R[q]
        first = np.where(hit.any(1), hit.argmax(1), -1)
    for q, e in enumerate(extras):
        x, y, size, ev = e[0], e[1], e[2], e[3]
        r = R[q]
        j = int(first[q]) if first[q] >= 0 else next(
            (n0 + k for k, (px, py, _) in enumerate(new) if (px - x) ** 2 + (py - y) ** 2 <= r * r), None)
        if j is None:
            new.append((float(x), float(y), float(size)))
            j = n0 + len(new) - 1
            extra_only.add(j)
        if ev is not None and (j not in evidence or ev[0] > evidence[j][0]):
            evidence[j] = ev
    return new, evidence, extra_only


def match_extras_reference(props_xy, extras, crop_size):
    """The pipeline's loop verbatim (for tests): same return values as match_extras."""
    props = [(float(x), float(y)) for x, y in props_xy]
    n0 = len(props)
    evidence: dict = {}
    extra_only: set = set()
    new: list = []
    for e in extras:
        x, y, size, ev = e[0], e[1], e[2], e[3]
        r = max(6.0, 0.5 * crop_size(size))
        j = next((k for k, (px, py) in enumerate(props) if (px - x) ** 2 + (py - y) ** 2 <= r * r), None)
        if j is None:
            props.append((float(x), float(y)))
            new.append((float(x), float(y), float(size)))
            j = len(props) - 1
            extra_only.add(j)
        if ev is not None and (j not in evidence or ev[0] > evidence[j][0]):
            evidence[j] = ev
    return new, evidence, extra_only
