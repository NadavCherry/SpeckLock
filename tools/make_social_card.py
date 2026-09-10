#!/usr/bin/env python3
"""The 1280x640 social preview card.

Two places want one image that says what this is: GitHub's repository social
preview (Settings -> General -> Social preview) and the ``og:image`` a link
unfurls to on LinkedIn or Slack. Both crop toward the centre and both render it
small, so this is deliberately four numbers and a sentence rather than a
diagram -- the end-to-end figure is `make_arch_figure_system.py` and does not
survive being 400 px wide in a feed.

    .venv/bin/python tools/make_social_card.py
        -> docs/media/social_card.png / .svg

The provenance line is not decoration. It is the first thing a sceptical reader
looks for and the last thing a headline number usually says.
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_arch_figures_final import BLUE, CARD, EDGE, GREEN, SUB, TXT, Fig  # noqa: E402

AMBER = "#d29922"
W, H = 1280, 640

# Every number here is a row of README section 12, with the provenance colour it carries
# there. This card used to lead with "1.000 ... unseen real video" and "24 / 24" -- both
# retracted by the README, and both still rendered here as the og:image a shared link
# shows, because the text was corrected and the picture was never regenerated.
STATS = [
    ("0.159 &#8594; 0.895", "AP: 1 frame vs 3 moments", "near-static video &#183; paste confound", GREEN),
    ("0.809", "AP, ARD-MAV", "official test split &#183; 3 seeds", GREEN),
    ("54 / 62", "closed-loop intercepts", "sim &#183; seeker's own detections", AMBER),
    ("58.9 fps", "1280 px, TensorRT FP16", "RTX 4090 &#183; AP in the same pass", GREEN),
]


def main() -> int:
    F = Fig(W, H)

    F.svg.append(f'<text x="64" y="118" font-size="58" font-weight="700" fill="{TXT}">'
                 f'See the drone, then hit it</text>')
    # No "from a moving camera": the headline detection number is from a camera that drifts
    # ~1 px over the whole clip, and on the two moving-camera benchmarks the temporal stack
    # does not separate from a single frame (README section 6).
    F.svg.append(f'<text x="64" y="166" font-size="23" fill="{SUB}">'
                 f'Finding a drone <tspan font-weight="700" fill="{TXT}">4&#8211;15 pixels</tspan> '
                 f'across in 720p video &#8212;</text>')
    F.svg.append(f'<text x="64" y="200" font-size="23" fill="{SUB}">'
                 f'then flying into it, with nothing but that camera.</text>')

    # four numbers, evenly spaced
    x0, gap, cw, ch = 64, 24, (W - 128 - 3 * 24) / 4, 190
    for i, (v, k, n, col) in enumerate(STATS):
        x = x0 + i * (cw + gap)
        y = 250
        F.svg.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" rx="14" '
                     f'fill="{CARD}" stroke="{EDGE}" stroke-width="2"/>')
        F.svg.append(f'<rect x="{x}" y="{y}" width="{cw}" height="5" rx="2.5" fill="{col}"/>')
        vs = 42 if len(html.unescape(v)) <= 8 else 32  # "0.159 -> 0.895" overruns a card at 42
        F.svg.append(f'<text x="{x + cw / 2}" y="{y + 76}" text-anchor="middle" font-size="{vs}" '
                     f'font-weight="700" fill="{TXT}">{v}</text>')
        F.svg.append(f'<text x="{x + cw / 2}" y="{y + 112}" text-anchor="middle" font-size="17" '
                     f'font-weight="600" fill="{TXT}" opacity="0.9">{k}</text>')
        F.svg.append(f'<text x="{x + cw / 2}" y="{y + 140}" text-anchor="middle" font-size="14.5" '
                     f'fill="{SUB}">{n}</text>')

    # the provenance line -- the point of the two colours above
    y = 500
    F.svg.append(f'<circle cx="74" cy="{y - 5}" r="7" fill="{GREEN}"/>')
    F.svg.append(f'<text x="92" y="{y}" font-size="18" fill="{TXT}" opacity="0.9">'
                 f'detection measured on real hand-labelled video</text>')
    F.svg.append(f'<circle cx="622" cy="{y - 5}" r="7" fill="{AMBER}"/>')
    F.svg.append(f'<text x="640" y="{y}" font-size="18" fill="{TXT}" opacity="0.9">'
                 f'interception measured closed-loop in Isaac Sim &#8212; no flight test</text>')

    F.svg.append(f'<rect x="64" y="{y + 34}" width="{W - 128}" height="1.5" fill="{EDGE}"/>')
    F.svg.append(f'<text x="64" y="{y + 84}" font-size="20" font-weight="700" fill="{BLUE}">'
                 f'github.com/NadavCherry/SpeckLock</text>')
    F.svg.append(f'<text x="{W - 64}" y="{y + 84}" text-anchor="end" font-size="18" fill="{SUB}">'
                 f'Nadav Cherry</text>')

    F.write("social_card")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
