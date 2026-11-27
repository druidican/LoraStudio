# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""sdloss.py - the loss of a run: the trainer's lines read for epoch and loss (_note_loss, called by the run's reader),
the loss per epoch analysed (loss_analysis: the save point to try first, rising at the end, still falling) and shown
(loss_html).
"""
from __future__ import annotations
__version__ = "0"  # LoRA Studio version, see version.py

import html
import math
import re

import sdgui as S


# --------------------------------------------------------------------------- loss indicator
EPOCH_RE = re.compile(r"\s*epoch\s+(\d+)\s*/\s*(\d+)\s*$")
LOSS_RE = re.compile(r"\|\s*(\d+)/\d+\s*\[.*?avr_loss=([0-9.eE+\-]+|nan|inf)\s*\]")


def _note_loss(run: S.Run, prog: str):
    """Record one (step, avr_loss, epoch) sample from a tqdm line. Called with run.lock held."""
    m = LOSS_RE.search(prog)
    if not m:
        return
    step = int(m.group(1)) + run.step_offset
    try:
        val = float(m.group(2))
    except ValueError:
        return
    if not math.isfinite(val) and run.nan_at is None:
        run.nan_at = {"step": step, "epoch": run.epoch, "stage": run.stage_no}
    if run.loss and run.loss[-1][0] == step:
        return
    run.loss.append((step, val, run.epoch))
    if len(run.loss) > 40000:
        del run.loss[1:len(run.loss):2]          # thin out very long runs


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def loss_analysis(samples):
    """Plain-language reading of the avr_loss history. Returns dict with epochs, best/worst, verdict.
    avr_loss is only a smoothed training signal: it does not measure likeness. The verdicts compare the
    run with itself (its own best so far), not with a fixed number."""
    import math
    bad = [x for x in samples if not math.isfinite(x[1])]
    good = [x for x in samples if math.isfinite(x[1])]
    out = {"n": len(good), "epochs": [], "best": None, "worst": None, "level": "info", "msgs": []}
    if bad:
        out["level"] = "bad"
        out["msgs"].append("avr_loss is NaN/inf: training has diverged. Stop it, halve the learning rate and restart "
                           "(or resume from the last good state).")
    if len(good) < 8:
        if not bad:
            out["msgs"].append("Collecting data... (needs a few dozen steps)")
        return out
    by_ep = {}
    for step, v, ep in good:
        by_ep.setdefault(ep, []).append(v)
    rows = []
    for ep in sorted(by_ep):
        vs = by_ep[ep]
        rows.append({"epoch": ep, "mean": _mean(vs), "end": vs[-1], "n": len(vs)})
    out["epochs"] = rows
    # only epochs with enough samples count for best / worst (the running epoch is judged by its samples so far)
    full = [r for r in rows if r["n"] >= 5]
    if full:
        out["best"] = min(full, key=lambda r: r["mean"])
        out["worst"] = max(full, key=lambda r: r["mean"])
    vals = [v for _, v, _ in good]
    k = max(5, len(vals) // 20)
    cur = _mean(vals[-k:])
    sm = [_mean(vals[max(0, i - k + 1):i + 1]) for i in range(len(vals))]
    warm = max(k, len(vals) // 6)                 # ignore the noisy start when looking for the best point
    best_i = min(range(warm, len(sm)), key=lambda i: sm[i]) if len(sm) > warm else len(sm) - 1
    best_sm = sm[best_i]
    out["best_pt"] = (good[best_i][0], best_sm)
    out["worst_pt"] = max(((good[i][0], sm[i]) for i in range(warm, len(sm))), key=lambda t: t[1], default=None)
    msgs = out["msgs"]
    level = out["level"]
    rise = (cur - best_sm) / best_sm if best_sm > 0 else 0
    last20 = vals[-20:]
    med = sorted(last20)[len(last20) // 2]
    if len(good) > 40 and vals[-1] > 1.5 * med and med > 0:
        msgs.append(f"Spike: avr_loss jumped to {vals[-1]:.4f} (typical {med:.4f}). One spike is usually a bad batch; "
                    "repeated spikes mean the learning rate is too high.")
        level = max(level, "warn", key=["info", "ok", "warn", "bad"].index)
    if len(good) > 3 * warm and rise > 0.30:
        msgs.append(f"**Poor-likeness zone:** avr_loss {cur:.4f} is {rise:.0%} above its best ({best_sm:.4f} at step {good[best_i][0]}). "
                    "The model is moving away from the data: the learning rate may be too high, or the run is overfitting/unstable. "
                    "Compare the sample images; the checkpoint from near the best step is the safer pick.")
        level = max(level, "bad", key=["info", "ok", "warn", "bad"].index)
    elif len(good) > 3 * warm and rise > 0.12:
        msgs.append(f"Drifting up: avr_loss {cur:.4f} is {rise:.0%} above its best ({best_sm:.4f} at step {good[best_i][0]}). "
                    "Watch the next epochs; if it keeps rising, prefer the earlier checkpoint.")
        level = max(level, "warn", key=["info", "ok", "warn", "bad"].index)
    tail = vals[-max(20, len(vals) // 4):]
    if len(good) > 4 * warm and max(tail) > 0 and (max(tail) - min(tail)) / _mean(tail) < 0.04 and rise <= 0.12:
        msgs.append("Flat: avr_loss has stopped moving. The model has learned most of what it can from this data; "
                    "extra epochs mostly add overfitting risk.")
        level = max(level, "ok", key=["info", "ok", "warn", "bad"].index)
    if len(full) >= 3:
        a, b, c = full[-3]["mean"], full[-2]["mean"], full[-1]["mean"]
        if c > b > a and out["best"] is not full[-1]:
            msgs.append("The last 3 epoch averages are rising; compare the sample images from the best epoch with the latest.")
            level = max(level, "warn", key=["info", "ok", "warn", "bad"].index)
    if not msgs:
        first = vals[: max(5, len(vals) // 10)]
        drop = (_mean(first) - cur) / _mean(first) if _mean(first) else 0
        msgs.append(f"Healthy: avr_loss is {cur:.4f}, {drop:.0%} below the start and near its best." if drop > 0.02
                    else f"avr_loss is {cur:.4f}, steady. Diffusion loss falls only slowly; judge likeness from the sample images.")
        level = "ok"
    out["level"] = level
    return out


def _why_best_worst(an):
    rows = an["epochs"]
    if not an["best"] or not an["worst"] or len(rows) < 2:
        return "", ""
    b, w, f = an["best"], an["worst"], rows[0]
    last = rows[-1]["epoch"]
    why_b = f"Lowest epoch average ({b['mean']:.4f}), {100 * (f['mean'] - b['mean']) / f['mean']:.0f}% below epoch {f['epoch']}."
    if b["epoch"] == last:
        why_b += " Still improving: nothing later to compare with yet."
    else:
        why_b += " Later epochs are not lower, so extra training since then did not reduce the loss."
    if w["epoch"] == rows[0]["epoch"]:
        why_w = "Highest average, as expected: the model has not learned the data yet (early epoch)."
    elif w["epoch"] == last:
        why_w = "The newest epoch is the highest: loss is rising. Possible instability or overfitting; check the samples."
    else:
        why_w = "Highest average although later epochs are lower. Likely an unstable stretch (learning-rate spike or a hard batch)."
    return why_b, why_w


def loss_html() -> str:
    with S.RUN.lock:
        samples = list(S.RUN.loss)
    if S.RUN.proc is None or not samples:
        return ("<div class='lossbox'><span class='lossmuted'>Loss indicator: waiting for avr_loss values "
                "(they appear once training prints its progress bar).</span></div>")
    an = loss_analysis(samples)
    col = {"ok": "var(--ls-ok)", "info": "var(--ls-info)", "warn": "var(--ls-warn)", "bad": "var(--ls-bad)"}[an["level"]]
    icon = {"ok": "🟢", "info": "⚪", "warn": "🟠", "bad": "🔴"}[an["level"]]
    msgs = "".join(f"<div>{html.escape(m).replace('**', '')}</div>" for m in an["msgs"])
    why_b, why_w = _why_best_worst(an)

    def cell(title, r, why, c):
        if not r:
            return f"<div class='losscell'><b>{title}</b><br><span class='lossmuted'>needs at least 2 epochs</span></div>"
        return (f"<div class='losscell' style='border-left:4px solid {c}'><b>{title}</b>: epoch {r['epoch']}, "
                f"avg avr_loss <b>{r['mean']:.4f}</b><br><span class='lossmuted'>{html.escape(why)}</span></div>")
    cells = cell("Best epoch", an["best"], why_b, "var(--ls-ok)") + cell("Worst epoch", an["worst"], why_w, "var(--ls-bad)")

    # chart
    W, H, P = 600, 120, 8
    pts = [(s, v) for s, v, _ in samples if v == v and abs(v) != float("inf")]
    svg = ""
    if len(pts) >= 2:
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        x0, x1 = min(xs), max(xs); y0, y1 = min(ys), max(ys)
        if x1 == x0: x1 = x0 + 1
        if y1 == y0: y1 = y0 + 1e-6
        X = lambda x: P + (W - 2 * P) * (x - x0) / (x1 - x0)
        Y = lambda y: H - P - (H - 2 * P) * (y - y0) / (y1 - y0)
        step = max(1, len(pts) // 300)
        d = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in pts[::step])
        svg = (f"<svg viewBox='0 0 {W} {H}' width='100%' height='{H}' preserveAspectRatio='none' class='losschart'>"
               f"<polyline fill='none' stroke='var(--ls-line)' stroke-width='1.6' points='{d}'/>")
        for key, c in (("best_pt", "var(--ls-ok)"), ("worst_pt", "var(--ls-bad)")):
            pt = an.get(key)
            if pt:
                svg += (f"<line x1='{X(pt[0]):.1f}' x2='{X(pt[0]):.1f}' y1='{P}' y2='{H - P}' stroke='{c}' stroke-width='2' "
                        f"stroke-dasharray='4 3' vector-effect='non-scaling-stroke'/>")
        svg += "</svg>"
        svg += (f"<div class='lossmuted' style='display:flex;justify-content:space-between'><span>step {x0}</span>"
                f"<span>avr_loss {y0:.4f} to {y1:.4f}; <span style='color:var(--ls-ok)'>┆</span> best point, "
                f"<span style='color:var(--ls-bad)'>┆</span> worst point (after warm-up)</span><span>step {x1}</span></div>")
    ep = ""
    if an["epochs"]:
        best_e = an["best"]["epoch"] if an["best"] else None
        ep = "<div class='lossmuted' style='margin-top:6px'>" + " &nbsp;|&nbsp; ".join(
            (f"<b>ep {r['epoch']}: {r['mean']:.4f}</b>" if r["epoch"] == best_e else f"ep {r['epoch']}: {r['mean']:.4f}")
            for r in an["epochs"]) + "</div>"
    note = ("<div class='lossmuted' style='margin-top:6px'>avr_loss is a smoothed training signal, not a likeness score. "
            "It is compared with this run's own best, so the verdict is a hint. Always confirm with the sample images.</div>")
    return (f"<div class='lossbox' style='border-left:6px solid {col}'><div style='font-weight:600;margin-bottom:4px'>"
            f"{icon} Loss indicator</div>{msgs}<div class='losscells'>{cells}</div>{svg}{ep}{note}</div>")
