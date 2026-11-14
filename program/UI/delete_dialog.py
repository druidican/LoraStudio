# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""The Delete dialog of the Collection (Browse › Select › Delete…, and the LoRA page): per LoRA, tick the versions
and the parts to remove, and whether the dataset folder goes too. Everything goes to the Trash (Gallery/delete.py)."""
from __future__ import annotations
__version__ = "1.141126b"  # LoRA Studio version, see version.py

from pathlib import Path

from nicegui import run, ui

import delete as DEL
import lora_library as LL
from widgets import note


async def open_dialog(gids: list[str], after=None):
    """after(message) is called once something was moved to the Trash."""
    if not gids:
        ui.notify("Nothing selected: tick Select, click the cards you want, then press Delete again.", type="warning")
        return
    ui.notify("Looking up what belongs to it …")
    plan = await run.io_bound(DEL.plan, gids)
    if not plan:
        ui.notify("No LoRA found for that selection.", type="warning")
        return
    human = LL.HK.human
    picks = {e["gid"]: {"versions": {v["key"]: True for v in e["versions"]},
                        "parts": {k: True for k, _ in DEL.PARTS}, "dataset": False} for e in plan}

    first = {}                                           # the dataset tick is shown once per dataset
    for e in plan:
        if e["dataset"]:
            first.setdefault(e["dataset"], e["gid"])

    def ds_out(e) -> bool:
        return bool(e["dataset"]) and picks[first[e["dataset"]]]["dataset"]

    def total() -> int:
        n = sum(e["dataset_size"] for e in plan if e["dataset"] and first[e["dataset"]] == e["gid"] and ds_out(e))
        for e in plan:
            p = picks[e["gid"]]
            for v in e["versions"]:
                if p["versions"][v["key"]]:
                    for k, on in p["parts"].items():
                        if on and not (ds_out(e) and all(Path(f).is_relative_to(e["dataset"]) for f in v["parts"][k])):
                            n += v["sizes"][k]
        return n

    with ui.dialog().props("persistent") as d, ui.card().classes("w-[860px] max-w-full ls-page"):
        ui.label("Delete LoRAs" if len(plan) > 1 else f"Delete {plan[0]['title']}").classes("text-lg font-semibold")
        note("Tick what goes. Everything is moved to the Trash, so it can be put back from the file manager until the "
             "Trash is emptied.")
        size_lbl = ui.label("").classes("font-semibold")

        def upd():
            size_lbl.text = f"Moves {human(total())} to the Trash."
        with ui.scroll_area().classes("w-full h-[min(60vh,640px)]"):
            for e in plan:
                p = picks[e["gid"]]
                with ui.card().classes("w-full ls-card gap-1"):
                    ui.label(e["title"]).classes("font-semibold")
                    with ui.row().classes("items-center gap-2"):
                        ui.label("Versions:").classes("ls-note")
                        for v in e["versions"]:
                            vs = sum(v["sizes"].values())
                            ui.checkbox(f"{v['label']} ({human(vs)})", value=True,
                                        on_change=lambda ev, p=p, k=v["key"]: (p["versions"].__setitem__(k, ev.value), upd()))
                    for k, label in DEL.PARTS:
                        n = sum(len(v["parts"][k]) for v in e["versions"])
                        sz = sum(v["sizes"][k] for v in e["versions"])
                        cb = ui.checkbox(f"{label} · {n} item(s), {human(sz)}", value=n > 0,
                                         on_change=lambda ev, p=p, k=k: (p["parts"].__setitem__(k, ev.value), upd()))
                        if not n:
                            p["parts"][k] = False
                            cb.disable()
                    if e["dataset"] and first[e["dataset"]] != e["gid"]:
                        note(f"Dataset folder {e['dataset_name']}: see the first LoRA of it above.")
                    elif e["dataset"]:
                        ui.checkbox(f"The dataset folder too: {e['dataset_name']} (pictures, captions, everything in it), "
                                    f"{human(e['dataset_size'])}", value=False,
                                    on_change=lambda ev, p=p: (p.__setitem__("dataset", ev.value), upd())) \
                            .classes("text-red-700 dark:text-red-300")
                        if e["others"]:
                            note("⚠ The dataset is also used by: " + ", ".join(e["others"]) + ". Deleting the folder "
                                 "removes their pictures too.")
                    with ui.expansion("Show the files").classes("w-full"):
                        for v in e["versions"]:
                            files = [f for k, _ in DEL.PARTS for f in v["parts"][k]]
                            ui.label(f"{v['label']}: {len(files)} item(s)").classes("font-semibold text-sm")
                            ui.label("\n".join(files) or "nothing").classes("ls-mono text-xs whitespace-pre-wrap")
        ok = ui.checkbox("I want these moved to the Trash")
        msg = ui.markdown("")

        async def go():
            if not ok.value:
                msg.content = "Tick *I want these moved to the Trash* first."
                return
            sel = {g: {"versions": [k for k, on in p["versions"].items() if on],
                       "parts": [k for k, on in p["parts"].items() if on], "dataset": p["dataset"]}
                   for g, p in picks.items()}
            btn.disable()
            msg.content = "Moving to the Trash …"
            text = await run.io_bound(DEL.run, plan, sel)
            await run.io_bound(LL.LIB.scan)
            d.close()
            if after:
                r = after(text)
                if hasattr(r, "__await__"):
                    await r
        with ui.row().classes("w-full justify-end gap-2"):
            ui.button("Cancel", on_click=d.close).props("flat no-caps")
            btn = ui.button("Move to the Trash", on_click=go).props("unelevated no-caps color=negative")
        upd()
    d.open()
