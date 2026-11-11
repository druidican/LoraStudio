# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""All the page's start-up loaders as ONE event.

Every page part fills itself when the page opens (app.load(...).then(...)). That was about 40 Gradio events, and each
event costs the browser 0.3-0.5 s on LoRA Studio's big page, so opening LoRA Studio took 15-20 s of a busy browser.
While the page is built, app.load is caught here; afterwards one load event runs every step in the same order. A
later step reads what an earlier one set (a chosen dataset, a filled list), as it did with .then. Steps that cannot
be joined (with their own js, or that stream with yield) stay separate events.
"""
from __future__ import annotations
__version__ = "1.111126d"  # LoRA Studio version, see version.py

import inspect
import os
import time

import gradio as gr

STEPS: list = []              # [(fn, inputs, outputs)] in the order the page registered them
DEBUG = bool(os.environ.get("LS_LOAD_DEBUG"))   # print how long each start-up step takes


def _lst(x):
    return list(x) if isinstance(x, (list, tuple)) else ([] if x is None else [x])


class _Chain:
    """What app.load(...) returns while the page is built: .then / .success add the next step."""

    def __init__(self, real_load):
        self.real_load = real_load

    def then(self, fn=None, inputs=None, outputs=None, **kw):
        return add(self.real_load, fn, inputs, outputs, **kw)

    success = then


def add(real_load, fn=None, inputs=None, outputs=None, **kw):
    joinable = fn is not None and not kw.get("js") and not inspect.isgeneratorfunction(fn) and \
        not inspect.iscoroutinefunction(fn)
    if not joinable:
        return real_load(fn, inputs, outputs, **kw)
    STEPS.append((fn, _lst(inputs), _lst(outputs)))
    return _Chain(real_load)


def catch(app):
    """Call right after `with gr.Blocks() as app:` starts."""
    real = app.load
    app.load = lambda fn=None, inputs=None, outputs=None, **kw: add(real, fn, inputs, outputs, **kw)
    app._ls_real_load = real


def _value(v):
    if isinstance(v, dict) and v.get("__type__") == "update":
        return v["value"] if "value" in v else _NOVAL
    return v


_NOVAL = object()


def _merge(old, new):
    if isinstance(new, dict) and new.get("__type__") == "update":
        if isinstance(old, dict) and old.get("__type__") == "update":
            return {**old, **new}
        if old is not _NOVAL and "value" not in new:
            return {**new, "value": old} if not isinstance(old, dict) else new
        return new
    return new


def build(app):
    """Make the one start-up event (inside the Blocks, after every part has been built)."""
    real = getattr(app, "_ls_real_load", app.load)
    app.load = real
    if not STEPS:
        return
    ins = []
    for _, i, _ in STEPS:
        for c in i:
            if all(c is not x for x in ins):
                ins.append(c)
    outs = []
    for _, _, o in STEPS:
        for c in o:
            if all(c is not x for x in outs):
                outs.append(c)
    steps = list(STEPS)

    def run_all(*vals):
        cur = {id(c): v for c, v in zip(ins, vals)}          # what each component holds now
        res = {}                                             # id -> the update sent at the end
        times = []
        for fn, i, o in steps:
            t0 = time.perf_counter()
            try:
                r = fn(*[cur.get(id(c)) for c in i])
                if DEBUG:
                    times.append((time.perf_counter() - t0, getattr(fn, "__qualname__", str(fn))))
            except Exception as e:                           # noqa: BLE001 - one part failing must not stop the others
                print(f"[start-up] {getattr(fn, '__qualname__', fn)}: {e!r}")
                continue
            if len(o) == 1:
                r = [r]
            elif not isinstance(r, (list, tuple)):
                continue
            for c, v in zip(o, r):
                res[id(c)] = _merge(res.get(id(c), _NOVAL), v)
                val = _value(v)
                if val is not _NOVAL:
                    cur[id(c)] = val
        if DEBUG:
            print(f"[start-up] {sum(t for t, _ in times):.2f} s in {len(times)} steps; slowest:", flush=True)
            for t, name in sorted(times, reverse=True)[:12]:
                print(f"   {t:6.3f} s  {name}", flush=True)
        return [res.get(id(c), gr.update()) for c in outs]

    real(run_all, ins, outs, show_progress="hidden")
    STEPS.clear()
