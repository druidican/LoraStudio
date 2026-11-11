# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""One timer for the whole LoRA Studio page.

Every gr.Timer tick is a round trip that makes the browser re-check the whole (very big) page, so five timers ticking
every one or two seconds kept the browser busy all the time and every click had to wait its turn. The pages now
register what they want refreshed here, and lora_studio.py makes ONE timer that refreshes all of it in one go.
A tick answers with one small hidden value per part; only a part with something new changes its value, and that
value's change event redraws just that part (an idle tick used to cost the browser 0.3-0.6 s, every part being an
output of the one event).
Standalone (sdgui.py on its own) nothing is registered and the parts keep their own timers.
"""
from __future__ import annotations
__version__ = "1.111126d"  # LoRA Studio version, see version.py

import os

import gradio as gr

TICKS: list = []
ACTIVE = {"on": False}
INTERVAL = 2.0          # while something runs (training, the batch fix, a download)
IDLE = 10.0             # otherwise (the status strip and the queue refresh without it: livepoll.py)
DEBUG = bool(os.environ.get("LS_TICK_DEBUG"))
BUSY: list = []         # functions that say whether something is running


SLEEP = {"on": False}       # the timer may sleep while idle (only when the page wakes it: livepoll.py)


def is_busy() -> bool:
    for f in BUSY:
        try:
            if f():
                return True
        except Exception:                                # noqa: BLE001
            pass
    return False


def _empty(v) -> bool:
    return isinstance(v, dict) and set(v) <= {"__type__"}


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
    only = os.environ.get("LS_TICK_ONLY")       # diagnostics: keep only the parts whose function name has this text
    if only:
        TICKS[:] = [t for t in TICKS if only in getattr(t[0], "__name__", "")]
    if not TICKS:
        return None
    ins = [c for _, i, _ in TICKS for c in i]
    sent = gr.State({})                       # per browser tab: what each output last showed
    sigs = [gr.Textbox(visible=False) for _ in TICKS]   # one per part: changes only when that part has news
    PENDING: dict = {}                        # (browser tab, part) -> the updates its redraw sends

    def tick_all(*vals):
        *vals, last = vals
        last = dict(last or {})
        tab = last.setdefault("tab", f"{id(last)}")
        sig_out, k = [], 0
        for p, (fn, i, o) in enumerate(TICKS):
            args = vals[k:k + len(i)]
            k += len(i)
            try:
                r = fn(*args)
            except Exception:                            # noqa: BLE001 - one part failing must not stop the others
                r = [gr.update()] * len(o)
            if len(o) == 1:
                r = [r[0] if isinstance(r, (list, tuple)) and len(r) == 1 else r]
            out, changed = [], False           # only what changed goes to the browser (each change redraws a part)
            for n, v in enumerate(r):
                key = hash(repr(v))
                if _empty(v) or last.get(f"{p}.{n}") == key:     # gr.update() alone changes nothing: no redraw
                    out.append(gr.update())
                else:
                    last[f"{p}.{n}"] = key
                    out.append(v)
                    changed = True
                    if DEBUG:
                        print(f"[tick] {getattr(fn, '__name__', p)} output {n}: {repr(v)[:100]}", flush=True)
            if changed:
                if len(PENDING) > 2000:
                    PENDING.clear()
                PENDING[(tab, p)] = out
                last[f"{p}.n"] = last.get(f"{p}.n", 0) + 1
                sig_out.append(f"{tab}:{last[f'{p}.n']}")
            else:
                sig_out.append(gr.update())    # nothing new for this part: no redraw at all
        busy = is_busy()
        if not busy and all(isinstance(x, dict) for x in sig_out) and SLEEP["on"]:
            want = "off"                       # nothing runs and nothing changed: the timer sleeps; the page's own
            tmr = gr.Timer(active=False)       # check (livepoll.py) wakes it when a run or download starts
        else:
            want = INTERVAL if busy else IDLE
            tmr = gr.Timer(value=want, active=True)
        tmr = gr.update() if last.get("interval") == want else tmr
        last["interval"] = want
        return sig_out + [last, tmr]

    def redrawer(p, n_out):
        def redraw(s):
            out = PENDING.pop(((s or ":").rsplit(":", 1)[0], p), None) or [gr.update()] * n_out
            return out[0] if n_out == 1 else out          # Gradio wants a single value for one output
        return redraw
    t = gr.Timer(IDLE)
    t.tick(tick_all, ins + [sent], sigs + [sent, t], show_progress="hidden")
    wake = gr.Button("wake", elem_id="ls_wake", elem_classes=["ls-hidden"])
    wake.click(lambda last: (gr.Timer(value=INTERVAL, active=True), {**(last or {}), "interval": INTERVAL}),
               sent, [t, sent], show_progress="hidden") \
        .then(tick_all, ins + [sent], sigs + [sent, t], show_progress="hidden")    # and show it at once
    for p, (_, _, o) in enumerate(TICKS):
        sigs[p].change(redrawer(p, len(o)), sigs[p], o, show_progress="hidden")
    return t
