# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Find on this computer (Settings › Find files, and the setup page): the folders to look through, the search, and
the suggestions with ticks. The search and the saving are in Settings/scanner.py."""
from __future__ import annotations
__version__ = "1.191126c"  # LoRA Studio version, see version.py

import os
import threading
from pathlib import Path

from nicegui import run, ui

import scanner as SC
from widgets import note

LABELS = {"sd": "sd-scripts folder", "mu": "musubi-tuner folder", "datasets": "Dataset folder",
          "loras": "LoRA folder (your finished LoRAs)", "models": "Model folder", "pictures": "ComfyUI picture folder",
          "lm_examples": "LoRA Manager's example pictures folder"}


def _unset(v: str) -> bool:
    """Nothing usable set: empty, the preset's /PATH/TO, or a path that is not there."""
    v = (v or "").strip()
    if not v or "/PATH/TO" in v:
        return True
    return not all(Path(os.path.expanduser(x)).exists() for x in v.splitlines() if x.strip())


def build(with_files: bool = True, on_use=None, only=None):
    """with_files: also the model files per LoRA type (Settings). on_use(folders: dict): instead of saving (the setup
    page fills its own fields). only: the folder settings to offer."""
    stop = threading.Event()
    keys = only or list(LABELS)
    with ui.column().classes("w-full gap-2"):
        note("Looks through the folders you list here, and their subfolders (at most 8 deep), for the trainers, "
             "ComfyUI, your LoRAs, datasets and model files, and suggests what to use. Nothing changes until you "
             "tick and press Use. Every model file's header is read once and remembered, so a second search is quick.")
        rt = ui.textarea("Folders to search, one per line", value="\n".join(SC.roots()),
                         placeholder="/mnt/Backup\n~/AI").props("outlined dense rows=3").classes("w-full")
        with ui.row().classes("items-center gap-2"):
            go = ui.button("Search").props("unelevated dense no-caps color=primary")
            ui.button("Stop", on_click=stop.set).props("flat dense no-caps color=negative")
        prog = ui.label("").classes("ls-note")
        box = ui.column().classes("w-full gap-1")
        msg = ui.markdown("")
        picks = {"folders": {}, "files": {}}

        async def search():
            rs = SC.save_roots(rt.value)
            if not rs:
                prog.text = "Write at least one folder to search."
                return
            stop.clear()
            go.disable()
            box.clear()
            msg.content = ""
            live = {"t": "Looking …"}
            t = ui.timer(0.5, lambda: setattr(prog, "text", live["t"]))
            try:
                res = await run.io_bound(SC.scan, rs, lambda x: live.__setitem__("t", x), stop)
                mp = await run.io_bound(SC.model_picks, res["models"]) if with_files else {}
                cur = await run.io_bound(SC.current) if on_use is None else {"files": {}}
            finally:
                t.deactivate()
                go.enable()
            prog.text = (f"Looked through {res['seen']} folders, {len(res['models'])} model files"
                         + (" (stopped)" if stop.is_set() else "") + ".")
            draw(res, mp, cur)
        go.on_click(search)

        def draw(res, mp, cur):
            picks["folders"].clear()
            picks["files"].clear()
            with box:
                ui.label("Folders").classes("font-semibold")
                any_f = False
                for k in keys:
                    cands = res["folders"].get(k) or []
                    if not cands:
                        continue
                    any_f = True
                    now = cur.get(k) or ""
                    with ui.row().classes("w-full items-center gap-2 no-wrap"):
                        cb = ui.checkbox(value=_unset(now))
                        sel = ui.select({p: f"{p}  ·  {why}" for p, why in cands}, value=cands[0][0],
                                        label=LABELS[k]).props("outlined dense").classes("grow")
                        ui.label("now: " + (now.replace("\n", ", ") or "not set")).classes("ls-note w-72 truncate")
                    picks["folders"][k] = (cb, sel)
                if not any_f:
                    note("No trainers, ComfyUI, LoRAs, datasets or model folders found in these folders.")
                comfy = res["folders"].get("comfy") or []
                if comfy:
                    note("ComfyUI found in: " + ", ".join(p for p, _ in comfy))
                if mp:
                    ui.label("Model files per LoRA type").classes("font-semibold mt-2")
                    note("Ticked where nothing usable is set now. The best match is chosen; the list has the others. "
                         "SDXL, Illustrious, Pony and NoobAI files look the same inside, so their base models are "
                         "matched by the file name; a file whose name does not say which it is stays unticked.")
                    import sdgui as S
                    _, presets = S.merged_presets()
                    for tkey, slots in mp.items():
                        have = cur["files"].get(tkey) or {}
                        lab = (presets.get(tkey) or {}).get("label", tkey)
                        todo = sum(1 for slot in slots if _unset(have.get(slot, "")))
                        with ui.expansion(f"{lab}" + (f"  ·  {todo} to set" if todo else ""), value=bool(todo)) \
                                .classes("w-full"):
                            for slot, cands in slots.items():
                                now = have.get(slot, "")
                                with ui.row().classes("w-full items-center gap-2 no-wrap"):
                                    cb = ui.checkbox(value=_unset(now) and cands[0][2])
                                    sel = ui.select({p: f"{Path(p).name}  ·  {d}" for p, d, _ in cands}, value=cands[0][0],
                                                    label=f"--{slot}").props("outlined dense").classes("grow")
                                    ui.label("now: " + (Path(now).name if now else "not set")).classes("ls-note w-56 truncate")
                                picks["files"].setdefault(tkey, {})[slot] = (cb, sel)
                ui.button("Use the ticked", on_click=use).props("unelevated no-caps color=primary")

        async def use():
            folders = {k: sel.value for k, (cb, sel) in picks["folders"].items() if cb.value and sel.value}
            files = {t: {s: sel.value for s, (cb, sel) in sl.items() if cb.value and sel.value}
                     for t, sl in picks["files"].items()}
            files = {t: v for t, v in files.items() if v}
            if on_use is not None:
                on_use(folders)
                msg.content = "✔ Filled in above; nothing is saved until you finish the setup."
                return
            msg.content = await run.io_bound(SC.apply, folders, files)
