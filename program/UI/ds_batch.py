# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Datasets › Improvements: Fix the ticked datasets (dsflow.batch_start), its progress, Look at the fixed ones, and
What Fix does (the batch settings, dsflow.batch_settings)."""
from __future__ import annotations
__version__ = "1.131126c"  # LoRA Studio version, see version.py

import json

from nicegui import ui

import datasets_ui as V
import dsflow as F
import settings_ui as SU
from widgets import note

TICKS = "document.querySelectorAll('.ds-imptick input')"


def tools(page):
    """The buttons in page.i_tools and the progress in page.i_prog (a 2 s timer while a batch runs or just ended)."""
    seen: list[str] = []

    async def start():
        v = await ui.run_javascript(f"JSON.stringify(Array.from({TICKS}).filter(function(e){{ return e.checked; }})"
                                    ".map(function(e){ return e.dataset.ds; }))")
        page.i_msg.content = F.batch_start(json.loads(v or "[]"))
        tick()

    def check_next():
        done = [r["ds"] for r in F.BATCH["results"] if r.get("ok")]
        todo = [d for d in done if d not in seen]
        if not todo:
            page.i_msg.content = "All fixed datasets are looked at." if done else "Fix some datasets first."
            return
        seen.append(todo[0])
        page.i_msg.content = f"Opened {todo[0]} at step 5; {len(todo) - 1} more after it."
        page.go(view="flow", mode="edit", ds=todo[0], step=5)

    with page.i_tools:
        ui.button("Tick all", on_click=lambda: ui.run_javascript(f"{TICKS}.forEach(function(i){{ i.checked = true; }})")) \
            .props("flat dense no-caps")
        ui.button("None", on_click=lambda: ui.run_javascript(f"{TICKS}.forEach(function(i){{ i.checked = false; }})")) \
            .props("flat dense no-caps")
        ui.button("Fix the ticked datasets", on_click=start).props("unelevated dense no-caps")
        ui.button("Stop", on_click=lambda: setattr(page.i_msg, "content", F.batch_stop())) \
            .props("outline dense no-caps color=negative")
        ui.button("Look at the fixed ones", on_click=check_next).props("flat dense no-caps")
    last = {"h": None}

    def tick():
        h = V.batch_html()
        if h != last["h"]:
            last["h"] = h
            page.i_prog.content = h
    ui.timer(2.0, tick)


def settings(page):
    """What Fix does: the choices, saved in batch_settings.json."""
    import features as FT
    import sd_script as SC
    b = F.batch_settings()
    c = {}
    note("Saved in batch_settings.json (in the backup too). Everything moved can be put back (_reserve/, _originals/).")
    ui.label("Pictures to keep (step 5 Select; the rest go to _reserve/)").classes("font-bold")
    c["keep_mode"] = ui.radio({v: k for k, v in SU.KEEP_CHOICES}).props("dense")
    c["max_pics"] = ui.number("Number of pictures to keep in the dataset folder", min=5, max=1000, precision=0) \
        .props("outlined dense").classes("w-80")
    c["select_mode"] = ui.radio({"safe": "Safe: the cleanest pictures first",
                                 "varied": "Varied: more close-ups, angles, outfits and details"}).props("dense")
    ui.label("Safety: move nothing in step 3 when more than this % of the pictures look not usable").classes("ls-note")
    c["safety_pct"] = ui.slider(min=5, max=100, step=5).props("label-always").classes("w-96 mt-8")
    ui.label("LoRA type and purpose (step 2; only for a dataset that has none yet)").classes("font-bold mt-2")
    with ui.row().classes("gap-3"):
        c["default_type"] = ui.select(dict((v, l) for l, v in SC.type_choices()), label="LoRA type") \
            .props("outlined dense").classes("w-72")
        c["default_purpose"] = ui.select(dict((v, l) for l, v in F.PURPOSES), label="Purpose").props("outlined dense").classes("w-56")
    ui.label("Watermarks and size (step 6)").classes("font-bold mt-2")
    c["watermarks"] = ui.radio({v: k for k, v in SU.WM_CHOICES}).props("dense")
    c["upscale"] = ui.checkbox("Upscale pictures that are a little too small (picture tools, Real-ESRGAN)")
    ui.label("Captions (step 7) and the end").classes("font-bold mt-2")
    ui.label(FT.label("Captions", "ollama")).classes("ls-note")
    c["captions"] = ui.radio({v: k for k, v in SU.CAP_CHOICES}).props("dense")
    if not FT.have("ollama"):
        c["captions"].disable()
    ui.label("Consistency check").classes("ls-note")
    c["consistency"] = ui.radio({v: k for k, v in SU.CONS_CHOICES}).props("dense")
    c["cons_each"] = ui.checkbox("Answer every trait with “This picture only” (each caption keeps its own; untick to "
                                 "walk through them yourself later in step 7)")
    c["cons_each"].bind_visibility_from(c["consistency"], "value", backward=lambda v: v == "check")
    c["launch_script"] = ui.checkbox("Write a launch script with the recommended settings at the end (step 9)")
    msg = ui.markdown("")
    summ = ui.markdown("").classes("ls-note")

    def load(b):
        for k, el in c.items():
            if k in b:
                el.value = b[k]
        summ.content = "**Now:** " + F.batch_summary(b)

    def save():
        vals = {k: el.value for k, el in c.items()}
        vals["max_pics"] = int(vals["max_pics"] or F.BATCH_MAX)
        load(F.batch_settings_save(vals))
        msg.content = "✔ Saved. The next *Fix the ticked datasets* uses these."
        page.i_what.content = V.batch_what()

    def reset():
        load(F.batch_settings_save(dict(F.BATCH_DEFAULTS)))
        msg.content = "✔ Back to the defaults."
        page.i_what.content = V.batch_what()
    with ui.row().classes("gap-2"):
        ui.button("Save", on_click=save).props("unelevated dense no-caps")
        ui.button("Back to the defaults", on_click=reset).props("outline dense no-caps")
    load(b)
