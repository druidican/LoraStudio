# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""One timer for the whole LoRA Studio page.

Every gr.Timer tick is a round trip that makes the browser re-check the whole (very big) page, so five timers ticking
every one or two seconds kept the browser busy all the time and every click had to wait its turn. The pages now
register what they want refreshed here, and lora_studio.py makes ONE timer that refreshes all of it in one go.
Standalone (sdgui.py on its own) nothing is registered and the parts keep their own timers.
"""
from __future__ import annotations
__version__ = "1.251026g"  # LoRA Studio version, see version.py

import gradio as gr

TICKS: list = []
ACTIVE = {"on": False}
INTERVAL = 2.0          # while something runs (training, the batch fix, a download)
IDLE = 6.0              # otherwise: each tick costs the browser about half a second on this big page
BUSY: list = []         # functions that say whether something is running


def timer(interval: float, fn, inputs, outputs, **kw):
    """Like gr.Timer(interval).tick(fn, inputs, outputs): in LoRA Studio it joins the one shared timer."""
    ins = list(inputs) if isinstance(inputs, (list, tuple)) else ([inputs] if inputs is not None else [])
    outs = list(outputs) if isinstance(outputs, (list, tuple)) else [outputs]
    if ACTIVE["on"]:
        TICKS.append((fn, ins, outs))
        return None
    return gr.Timer(interval).tick(fn, inputs, outputs, **kw)


def build():
    """Make the one timer (call once, inside the Blocks, after every part has registered)."""
    import os
    only = os.environ.get("LS_TICK_ONLY")       # diagnostics: keep only the parts whose function name has this text
    if only:
        TICKS[:] = [t for t in TICKS if only in getattr(t[0], "__name__", "")]
    if not TICKS:
        return None
    ins = [c for _, i, _ in TICKS for c in i]
    outs = [c for _, _, o in TICKS for c in o]
    sent = gr.State({})                       # per browser tab: what each output last showed

    def tick_all(*vals):
        *vals, last = vals
        last = dict(last or {})
        res, k = [], 0
        for fn, i, o in TICKS:
            args = vals[k:k + len(i)]
            k += len(i)
            try:
                r = fn(*args)
            except Exception:                            # noqa: BLE001 - one part failing must not stop the others
                r = [gr.update()] * len(o)
            if len(o) == 1:
                r = [r[0] if isinstance(r, (list, tuple)) and len(r) == 1 else r]
            res += list(r)
        out = []                               # only what changed goes to the browser (each change redraws a part)
        for n, v in enumerate(res):
            key = hash(repr(v))
            if last.get(n) == key:
                out.append(gr.update())
            else:
                last[n] = key
                out.append(v)
        busy = False
        for f in BUSY:
            try:
                busy = busy or bool(f())
            except Exception:                            # noqa: BLE001
                pass
        want = INTERVAL if busy else IDLE
        tmr = gr.update() if last.get("interval") == want else gr.Timer(value=want)
        last["interval"] = want
        return out + [last, tmr]
    t = gr.Timer(IDLE)
    t.tick(tick_all, ins + [sent], outs + [sent, t], show_progress="hidden")
    return t
