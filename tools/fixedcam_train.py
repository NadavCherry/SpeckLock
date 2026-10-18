#!/usr/bin/env python3
"""Train the fixed-camera crop classifier (drone / bird / other + centre offset).

    python tools/fixedcam_train.py --train a.npz b.npz ... [--val c.npz ...] --out DIR [--epochs 30]

Inputs are ``tools/fixedcam_harvest.py`` outputs. Classes are balanced by sampling weights (the
clips are dominated by "other"); the offset head is trained on drone and bird crops only.
Augmentation happens on the GPU: horizontal flips (offset mirrored), optional grey-level inversion
(a drone is dark on a lit wall and light on dark trees), grey-level noise, occasional blur, and
"freeze" - replacing the two past channels with the present one - so a hovering drone,
whose crops show no change over time, is still a drone. Writes DIR/model.pt and DIR/train.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load(paths):
    raw, S, lab, off = [], [], [], []
    for p in paths:
        d = np.load(p)
        raw.append(d["raw"]); S.append(d["S"]); lab.append(d["label"]); off.append(d["off"])
    return (np.concatenate(raw), np.concatenate(S), np.concatenate(lab).astype(np.int64),
            np.concatenate(off))


def main(argv=None) -> None:
    import torch
    import torch.nn.functional as F

    from realtime.fixedcam.net import TinyNet, normalize

    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--invert-p", type=float, default=0.0,
                    help="probability of inverting a crop's grey levels (polarity follows the background)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    raw, S, lab, off = load(a.train)
    dev = "cuda"
    Xr = torch.from_numpy(raw).to(dev)
    St = torch.from_numpy(S).to(dev)
    Y = torch.from_numpy(lab).to(dev)
    O = torch.from_numpy(off).to(dev)
    counts = np.bincount(lab, minlength=3)
    w = (1.0 / np.maximum(counts, 1))[lab]
    w = torch.from_numpy(w / w.sum()).to(dev)
    val = load(a.val) if a.val else None

    model = TinyNet(a.width).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps_per_epoch = max(1, len(lab) // a.batch)
    total = a.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.15)

    def augment(xr, s, o):
        xr = xr.float()
        n = xr.shape[0]
        flip = torch.rand(n, device=dev) < 0.5
        xr = torch.where(flip.view(-1, 1, 1, 1), xr.flip(-1), xr)
        o = o.clone()
        o[flip, 0] = -o[flip, 0]
        inv = torch.rand(n, device=dev) < a.invert_p
        xr = torch.where(inv.view(-1, 1, 1, 1), 255.0 - xr, xr)
        frz = torch.rand(n, device=dev) < 0.08
        xr[frz, 0] = xr[frz, 2]
        xr[frz, 1] = xr[frz, 2]
        xr = xr + torch.randn_like(xr) * torch.rand(n, 1, 1, 1, device=dev) * 3.0
        blur = torch.rand(n, device=dev) < 0.15
        if blur.any():
            k = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], device=dev, dtype=torch.float32) / 16
            xb = F.conv2d(xr[blur].reshape(-1, 1, 32, 32), k.view(1, 1, 3, 3), padding=1)
            xr[blur] = xb.reshape(-1, 3, 32, 32)
        s = s * torch.exp(torch.randn(n, device=dev) * 0.1)
        return normalize(xr.clamp(0, 255), s), o

    log = []
    t0 = time.time()
    step = 0
    for ep in range(a.epochs):
        model.train()
        tot = {"loss": 0.0, "cls": 0.0, "reg": 0.0}
        for _ in range(steps_per_epoch):
            idx = torch.multinomial(w, a.batch, replacement=True)
            x, o = augment(Xr[idx], St[idx], O[idx])
            y = Y[idx]
            logits, pred = model(x)
            lc = F.cross_entropy(logits, y, label_smoothing=0.05)
            m = y < 2
            lr_ = F.smooth_l1_loss(pred[m], o[m], beta=0.05) if m.any() else logits.sum() * 0
            loss = lc + 2.0 * lr_
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            tot["loss"] += loss.item(); tot["cls"] += lc.item(); tot["reg"] += lr_.item()
        rec = {"epoch": ep, **{k: round(v / steps_per_epoch, 4) for k, v in tot.items()}}
        if val is not None and (ep % 5 == 4 or ep == a.epochs - 1):
            rec.update(evaluate(model, val, dev))
        log.append(rec)
        print(json.dumps(rec), flush=True)
    torch.save({"state_dict": model.state_dict(), "width": a.width, "train": a.train,
                "counts": counts.tolist(), "epochs": a.epochs}, out / "model.pt")
    (out / "train.json").write_text(json.dumps({"args": vars(a), "counts": counts.tolist(),
                                                "log": log, "seconds": round(time.time() - t0, 1)},
                                               indent=1), encoding="utf-8")
    print(f"saved {out / 'model.pt'} ({time.time() - t0:.0f}s)")


def evaluate(model, val, dev):
    import torch

    from realtime.fixedcam.net import normalize

    raw, S, lab, off = val
    model.eval()
    P, R = [], []
    with torch.no_grad():
        for i in range(0, len(lab), 2048):
            x = normalize(torch.from_numpy(raw[i:i + 2048]).to(dev).float(),
                          torch.from_numpy(S[i:i + 2048]).to(dev))
            lg, rg = model(x)
            P.append(lg.argmax(1).cpu().numpy())
            R.append(rg.cpu().numpy())
    P = np.concatenate(P)
    R = np.concatenate(R)
    conf = np.zeros((3, 3), int)
    for t, p in zip(lab, P):
        conf[t, p] += 1
    m = lab == 0
    err = np.hypot(*((R[m] - off[m]) * (S[m, None] / 2)).T) if m.any() else np.array([np.nan])
    return {"val_confusion(rows=true d,b,o)": conf.tolist(),
            "val_drone_recall": round(conf[0, 0] / max(1, conf[0].sum()), 4),
            "val_bird_as_drone": round(conf[1, 0] / max(1, conf[1].sum()), 4),
            "val_other_as_drone": round(conf[2, 0] / max(1, conf[2].sum()), 4),
            "val_drone_center_err_px_median": round(float(np.median(err)), 3)}


if __name__ == "__main__":
    main()
