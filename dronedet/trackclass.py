"""Track-level classification: which confirmed tracks are drones?

Per-frame appearance at 3-14 px is weak (a bird and a drone genuinely
converge at 4 px), but aggregated over a track's lifetime the 2-class
verifier's opinion separates cleanly. Measured on 07_05 (moe3) + 10_06:

    track                             conf_frac   n_conf
    far drone (07_05)                    0.99       432
    drone (10_06)                        0.94       203
    landed/near drone (07_05)            1.00       571   (big boxes)
    all foliage-clutter tracks        0.00-0.02      0-1
    brief unlabeled sky object           1.00         4   <- n_conf kills it

Rule: a track is a DRONE iff at least CONF_FRAC (0.70) of its matched detections
are verifier-confirmed drone detections (label 'drone*', score >= DRONE_SCORE = 0.35)
AND it has at least N_CONF (8) such detections (sustained evidence -- a
4-detection flash is an anecdote, not a track). Tracks whose detections
are mostly large boxes are the landed/near drone ('near'); everything
else is 'other' (foliage, birds, unknown movers).

...OR the track is `sustained`: LONG_TRACK = 120 tracked frames promotes it with **no
appearance requirement at all**. That clause is not a detail -- it is the only path to
'drone' that never consults the verifier, and `dronedet/track.py` says in its own comment
that the kinematic filter does not exclude birds ("Birds pass too -- bird vs drone is an
appearance/classifier problem, not a kinematic one"). A bird that flies straight for four
seconds is what it admits. `tools/track_level_birds.py` reports how many tracks it
promoted and how many of those were labelled birds; do not remove that measurement.

This docstring previously said "score >= 0.5" while the code used 0.35, and did not
mention LONG_TRACK at all -- so the module contradicted itself and the round-3 report
repeated the wrong number. The constants below are the authority; if you change one,
change this paragraph in the same commit.

`allow_motion` defaulted to True until October 2026. Under that default every detection
labelled 'motion' counted as drone evidence, including the PC-MAX proposals the verifier
had looked at and rejected. Measured through final/run_final.py, it gained the drone
nothing and raised 11 and 4 clutter tracks on 07_05 and 10_06
(docs/reports/allow-motion.md), so it is now off unless a caller opts in. Neither
`allow_motion` nor LONG_TRACK existed when the round-3 tracked files were produced; only a
rule with neither clause reproduces them (work/reports/allow_motion/provenance.md).
LONG_TRACK is unchanged, pending its own decision.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

CONF_FRAC = 0.70   # fraction of tracked dets that must be drone-confirmed
N_CONF = 8         # minimum confirmed detections (sustained evidence)
LONG_TRACK = 120   # a track sustained this long by the (directedness-vetted) tracker is a
#                    real object even when the detector's confidence is too low to confirm it
#                    (a tiny drone on a hard dataset) -- lets recall survive weak appearance
DRONE_SCORE = 0.35  # appearance confidence for 'drone' evidence (combined multi-dataset
#                     model scores lower on hard datasets than the single-video model)
BIG_W = 60.0       # box width that says "near/landed drone", not the far target
MATCH_DIST = 8.0   # track-position -> detection association radius


def classify_tracks(tracks: dict, dets: dict, allow_motion: bool = False) -> dict[int, dict]:
    """tracks/dets: parsed JSON payloads (tracker output + the detection
    set it consumed). Returns {track_id: {cls, conf_frac, n_conf, ...}}.

    allow_motion: also count 'motion'-labelled detections (score >= 0.35) as drone
    evidence. Off by default, because what the label means depends on the detector. In
    PC-MAX's detections (dronedet/methods/hybrid2.py) a bare 'motion' is a proposal the
    verifier did NOT confirm, and EDGE-RT emits none. Opt in only for a detector whose
    'motion' label is evidence in its own right. tools/run_max.py does so for its
    near-static mc-hybrid path, where a weak appearance model needs colour-blind motion
    and the motion scores are already clutter-gated."""
    det_frames = {int(f): v for f, v in dets["frames"].items()}
    out = {}
    for tr in tracks["tracks"]:
        fs = {int(f): v for f, v in tr["frames"].items()}
        tracked = [(f, v) for f, v in fs.items() if v[4] == "tracked"]
        n_conf = 0
        n_big = 0
        n_matched = 0
        for f, v in tracked:
            best, bd = None, MATCH_DIST
            for d in det_frames.get(f, []):
                cx, cy = (d[0] + d[2]) / 2, (d[1] + d[3]) / 2
                dist = math.hypot(cx - v[0], cy - v[1])
                if dist < bd:
                    bd, best = dist, d
            if best is None:
                continue
            n_matched += 1
            label, score, w = best[5], best[4], best[2] - best[0]
            # drone evidence = appearance-confirmed, OR (opt-in) motion-confirmed. The
            # motion clause lets a colour-blind ego-motion detector vouch for an
            # appearance-poor target (a tiny black drone the verifier can't confirm).
            # The tracker's directedness/re-acq filters do NOT make that safe on their
            # own: on PC-MAX they let 4-15 static-camera clutter tracks through per video.
            appearance = label.startswith("drone") and score >= DRONE_SCORE
            motion = allow_motion and ("motion" in label) and score >= 0.35
            if appearance or motion:
                n_conf += 1
            if w >= BIG_W:
                n_big += 1
        conf_frac = n_conf / max(n_matched, 1)
        big_frac = n_big / max(n_matched, 1)
        confirmed = conf_frac >= CONF_FRAC and n_conf >= N_CONF
        sustained = len(tracked) >= LONG_TRACK        # long directed track = real object
        if confirmed or sustained:
            cls = "near" if big_frac >= 0.5 else "drone"
        else:
            cls = "other"
        out[tr["id"]] = {
            "cls": cls, "conf_frac": round(conf_frac, 3), "n_conf": n_conf,
            "big_frac": round(big_frac, 3), "n_tracked": len(tracked),
            "n": len(fs),
        }
    return out


def classify_files(tracks_path: str | Path, dets_path: str | Path,
                   allow_motion: bool = False) -> dict[int, dict]:
    return classify_tracks(json.loads(Path(tracks_path).read_text(encoding="utf-8")),
                           json.loads(Path(dets_path).read_text(encoding="utf-8")), allow_motion=allow_motion)


def mean_ego_motion(dets: dict) -> float:
    """Mean frame-to-frame camera translation (px) from stored 2x3 transforms."""
    import numpy as np
    tf = dets.get("meta", {}).get("transforms")
    if not tf:
        return 0.0
    items = sorted(((int(k), np.asarray(v, float).reshape(2, 3)) for k, v in tf.items()))
    shifts = []
    for (i0, m0), (i1, m1) in zip(items, items[1:]):
        A = np.linalg.inv(np.vstack([m1, [0, 0, 1]])) @ np.vstack([m0, [0, 0, 1]])
        shifts.append(float(np.hypot(A[0, 2], A[1, 2])))
    return float(np.median(shifts)) if shifts else 0.0
