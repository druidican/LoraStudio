# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Step 5's cards (logic in Data/gapfill.py): "Your pictures made with this LoRA" (every purpose: tick which ones
Ollama scans, then which ones join the dataset) and "Fill the gaps with your LoRA" (after the detailed scan: what is
missing or too much, looking through your own pictures first, making the rest through ComfyUI, picking the good ones)."""
from __future__ import annotations
__version__ = "1.201126f"  # LoRA Studio version, see version.py

from pathlib import Path

from nicegui import run, ui

import dsflow as F
import gapfill as GF
from widgets import note, pic_url, stream


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
            note("Too few: first your own ComfyUI pictures made with this dataset's LoRA alone are looked through "
                 "(Ollama says what each one shows); what they cannot fill is made new with the LoRA through your "
                 "ComfyUI test-picture workflow. How many of each:")
            with ui.row().classes("w-full gap-x-6 gap-y-1"):
                for dim, value, n in a["few"]:
                    with ui.row().classes("items-center gap-2 no-wrap"):
                        counts[(dim, value)] = ui.number(value=2, min=0, max=8, precision=0) \
                            .props("outlined dense").classes("w-16")
                        ui.label(f"{GF.LABEL.get(dim, dim)} · {value} (has {n})")
        msg = ui.markdown("")
        warn_lbl = note("")
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
            make_btn = ui.button("Fill the gaps").props("unelevated dense no-caps color=primary")
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
                ui.label(m["prompt"] if not m.get("found") else
                         f"Your own picture {Path(m['found']).name}, made with {m['lora']} at strength "
                         f"{m.get('strength', 1.0):g}").classes("ls-note text-xs")
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
                        ticks[f] = ui.checkbox(f"{m['value']}" + (" · yours" if m.get("found") else "")) \
                            .classes("text-xs")
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
                text = await run.io_bound(GF.fill, ds, jobs, lambda line: live.__setitem__("line", line))
            finally:
                t.deactivate()
                make_btn.enable()
            msg.content = text
            await first()
        make_btn.on_click(make)

        async def add():
            files = [f for f, cb in ticks.items() if cb.value]
            if not files:
                msg.content = "Tick the pictures to add first (or throw them all away)."
                return
            msg.content = await run.io_bound(GF.add, ds, files)
            warn_lbl.text = await run.io_bound(GF.share_note, ds)
            draw_made()
            await after_add()

        async def discard():
            msg.content = await run.io_bound(GF.discard, ds)
            draw_made()

        async def first():
            warn_lbl.text = await run.io_bound(GF.share_note, ds)
            lo = await run.io_bound(GF.lora_for, ds)
            n = len(await run.io_bound(GF.candidates, ds))
            lo_lbl.text = ((f"{n} picture(s) of your own made with this LoRA alone, not offered yet. " if n else
                            "No pictures of your own made with this LoRA alone (scan your ComfyUI picture folders on "
                            "the LoRA page). ")
                           + (f"New ones are made with {lo['label']}, the newest version of this dataset's LoRA." if lo
                              else "New ones cannot be made yet: that needs a trained version and a workflow for its "
                                   "model family in Settings › Test pictures."))
            if not lo and not n:
                make_btn.disable()
            draw_made()
        ui.timer(0, first, once=True)


# --------------------------------------------------------------------------- your pictures with the LoRA
PAGE = 48


def own_card(ds, after_add):
    """after_add(): async, redraws step 5's picture list."""
    import html as H
    box = {"rows": [], "shown": PAGE}
    with ui.expansion("Your pictures made with this LoRA: add the good ones to the dataset").classes("w-full ls-card") \
            as exp:
        note("Your ComfyUI pictures made with this dataset's LoRA alone (a picture with other LoRAs too is left out: "
             "their look would be learned as well), newest version first. Tick the ones that look right and press Scan "
             "the ticked: Ollama says whether each is usable and what it shows. Then tick the ones to add: they are "
             "copied into the repeat folder, pinned 📌 always keep. A few that show something the dataset lacks help; "
             "many teach the LoRA its own mistakes. (Found through the LoRA page's Pictures made with it.)")
        head = ui.label("").classes("font-semibold")
        grid = ui.html("", sanitize=False).classes("w-full").props(f'id="own_{id(box)}"')
        more = ui.button("Show more", on_click=lambda: (box.__setitem__("shown", box["shown"] + PAGE), draw())) \
            .props("flat dense no-caps")
        with ui.row().classes("items-center gap-2"):
            ui.button("Tick all shown", on_click=lambda: tick("true")).props("flat dense no-caps")
            ui.button("Untick all", on_click=lambda: tick("false")).props("flat dense no-caps")
            scan_btn = ui.button("Scan the ticked").props("unelevated dense no-caps color=primary")
            ui.button("Stop", on_click=F.STOP.set).props("flat dense no-caps color=negative")
            ui.button("Add the ticked to the dataset", on_click=lambda: add()) \
                .props("unelevated dense no-caps color=teal")
        msg = ui.markdown("")
        warn = note("")
    gid = f"own_{id(box)}"

    def tick(on: str):
        ui.run_javascript(f"document.querySelectorAll('#{gid} .own-tick').forEach(i => i.checked = {on});")

    async def ticked() -> list[str]:
        idx = await ui.run_javascript(
            f"Array.from(document.querySelectorAll('#{gid} .own-tick:checked')).map(e => +e.dataset.i)")
        rows = box["rows"]
        return [rows[i]["path"] for i in (idx or []) if 0 <= i < len(rows)]

    def draw(keep: set | None = None):
        rows = box["rows"]
        cells = []
        for i, r in enumerate(rows[:box["shown"]]):
            w = r["what"]
            on = keep is not None and r["path"] in keep and not (w or {}).get("bad")
            state = (' <span class="ls-chip ok">in the dataset</span>' if r["added"] else "")
            cells.append(
                f'<label class="own-pic{" own-bad" if (w or {}).get("bad") else ""}">'
                f'<img loading="lazy" src="{pic_url(r["path"], 240)}" title="{H.escape(Path(r["path"]).name)}">'
                f'<span><input type="checkbox" class="own-tick" data-i="{i}"{" checked" if on else ""}> '
                f'{H.escape(r["lora"])} · {r["strength"]:g}{state}</span>'
                f'<small>{H.escape(GF.what_text(w)) or "not scanned"}</small></label>')
        grid.content = ('<div class="own-grid">' + "".join(cells) + "</div>") if cells else (
            '<p class="ls-note">No pictures of yours made with this LoRA alone yet. Scan your ComfyUI picture folders '
            "on the LoRA page (Pictures made with it).</p>")
        n, scanned = len(rows), sum(1 for r in rows if r["what"])
        head.text = f"{n} picture(s), {scanned} scanned, {sum(1 for r in rows if r['added'])} in the dataset"
        more.set_visibility(n > box["shown"])

    async def load(keep: set | None = None):
        box["rows"] = await run.io_bound(GF.own_pictures, ds)
        draw(keep)
        warn.text = await run.io_bound(GF.share_note, ds)

    async def scan():
        paths = await ticked()
        if not paths:
            msg.content = "Tick the pictures to scan first."
            return
        F.STOP.clear()
        scan_btn.disable()
        try:
            msg.content = await stream(GF.own_scan(ds, paths), lambda line: setattr(msg, "content", line))
        finally:
            scan_btn.enable()
        await load(set(paths))                           # the usable ones stay ticked
    scan_btn.on_click(scan)

    async def add():
        paths = await ticked()
        if not paths:
            msg.content = "Tick the pictures to add first."
            return
        msg.content = await run.io_bound(GF.own_add, ds, paths)
        await load()
        await after_add()

    exp.on_value_change(lambda e: e.value and load())
