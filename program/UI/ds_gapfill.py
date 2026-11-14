# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Step 5's card "Fill the gaps with your LoRA" (logic in Data/gapfill.py): what is missing or too much, how many
pictures to make of each, making them through ComfyUI, and picking the good ones."""
from __future__ import annotations
__version__ = "1.141126b"  # LoRA Studio version, see version.py

from nicegui import run, ui

import dsflow as F
import gapfill as GF
from widgets import note, pic_url


def card(ds, after_add):
    """after_add(): async, redraws step 5's picture list."""
    a = GF.analyse(ds)
    with ui.card().props("flat bordered").classes("ls-card w-full gap-2"):
        ui.label("Fill the gaps with your LoRA").classes("font-semibold")
        if not a["few"] and not a["many"]:
            note("The detailed scan found no gaps: every framing, angle, pose and expression is there often enough.")
        lo_lbl = note("Looking for this dataset's LoRA …")
        counts = {}
        if a["few"]:
            note("Too few: new pictures are made with this dataset's LoRA through your ComfyUI test-picture workflow. "
                 "How many of each:")
            with ui.row().classes("w-full gap-x-6 gap-y-1"):
                for dim, value, n in a["few"]:
                    with ui.row().classes("items-center gap-2 no-wrap"):
                        counts[(dim, value)] = ui.number(value=2, min=0, max=8, precision=0) \
                            .props("outlined dense").classes("w-16")
                        ui.label(f"{GF.LABEL.get(dim, dim)} · {value} (has {n})")
        msg = ui.markdown("")
        if a["many"]:
            note("Too many of one kind: the pictures made above outweigh it; for a framing you can also set some of "
                 "them aside.")
            for dim, value, n, tot in a["many"]:
                with ui.row().classes("items-center gap-3"):
                    ui.label(f"{GF.LABEL.get(dim, dim)} · {value}: {n} of {tot} pictures")
                    if dim == "framing":
                        async def aside(d=dim, v=value):
                            msg.content = await run.io_bound(GF.set_aside, ds, d, v)
                            await after_add()
                        ui.button("Set some aside", on_click=aside).props("outline dense no-caps")
        with ui.row().classes("items-center gap-2") as make_row:
            make_btn = ui.button("Make the pictures").props("unelevated dense no-caps color=primary")
            ui.button("Stop", on_click=F.STOP.set).props("flat dense no-caps color=negative")
        make_row.set_visibility(bool(a["few"]))
        grid = ui.row().classes("w-full gap-2")
        with ui.row().classes("items-center gap-2") as pick_row:
            ui.button("Add the ticked to the dataset", on_click=lambda: add()).props("unelevated dense no-caps color=teal")
            ui.button("Throw them all away", on_click=lambda: discard()).props("outline dense no-caps color=negative")
        ticks = {}

        def big(f, m):
            with ui.dialog() as d, ui.card().classes("items-center"):
                ui.image(pic_url(GF._dir(ds) / f, 1024)).classes("w-[640px] max-w-full").props("fit=contain")
                ui.label(m["prompt"]).classes("ls-note text-xs")
                ui.button("Close", on_click=d.close).props("flat no-caps")
            d.open()

        def draw_made():
            st = GF.state(ds)
            grid.clear()
            ticks.clear()
            with grid:
                for f, m in st["made"].items():
                    with ui.column().classes("gap-0 items-center w-40"):
                        ui.image(pic_url(GF._dir(ds) / f, 320)).classes("w-40 h-40 rounded cursor-pointer") \
                            .props("fit=contain").on("click", lambda f=f, m=m: big(f, m))
                        ticks[f] = ui.checkbox(f"{m['value']}").classes("text-xs")
            pick_row.set_visibility(bool(st["made"]))

        async def make():
            jobs = [(d, v, int(c.value or 0)) for (d, v), c in counts.items() if int(c.value or 0) > 0]
            if not jobs:
                msg.content = "Set at least one number above 0."
                return
            F.STOP.clear()
            make_btn.disable()
            live = {"line": "⏳ Starting …"}
            t = ui.timer(0.5, lambda: setattr(msg, "content", live["line"]))
            try:
                text = await run.io_bound(GF.make, ds, jobs, lambda line: live.__setitem__("line", line))
            finally:
                t.deactivate()
                make_btn.enable()
            msg.content = text
            draw_made()
        make_btn.on_click(make)

        async def add():
            files = [f for f, cb in ticks.items() if cb.value]
            if not files:
                msg.content = "Tick the pictures to add first (or throw them all away)."
                return
            msg.content = await run.io_bound(GF.add, ds, files)
            draw_made()
            await after_add()

        async def discard():
            msg.content = await run.io_bound(GF.discard, ds)
            draw_made()

        async def first():
            lo = await run.io_bound(GF.lora_for, ds)
            lo_lbl.text = (f"Made with {lo['label']}, the newest version of this dataset's LoRA." if lo else
                           "No LoRA of this dataset can make pictures yet: it needs a trained version and a workflow "
                           "for its model family in Settings › Test pictures.")
            if not lo:
                make_btn.disable()
            draw_made()
        ui.timer(0, first, once=True)
