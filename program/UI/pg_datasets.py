# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Datasets: the dataset list (+ New dataset) and one page per dataset with four parts: Prepare (steps
1-8), Train (9), Results (its runs) and LoRA (its LoRA files).

The page keeps one state (self.s, the same keys as before: view, mode, ds, step, …). A change of state redraws the bar,
the step strip and the open step only: the open step's controls are built anew in self.box, which is quick in NiceGUI.
Pictures, grids and lists are drawn as HTML by datasets_ui's renderers; their clicks come back as ls_ds_box events
(route). What each step does to the dataset folder is in dsflow.py.
"""
from __future__ import annotations
__version__ = "1.201126e"  # LoRA Studio version, see version.py

import os
from pathlib import Path

from nicegui import run, ui

import datasets_ui as V
import dsflow as F
import sdgui as S
from widgets import note, pic_url

NAMES = {2: "Setup", 3: "Usable scan", 4: "Sort", 5: "Select", 6: "Watermarks and size", 7: "Captions", 8: "Review",
         9: "Train", 10: "Results", 11: "LoRA"}


class Datasets:
    def __init__(self, shell):
        self.shell = shell
        self.s = {"view": "edit", "mode": "edit", "ds": "", "step": 2}
        self.steps = {}                                  # step number -> method that builds it

    # ------------------------------------------------------------------ state
    def go(self, **kw):
        s = dict(self.s)
        s.pop("open_run", None)                          # a run is opened once (Results), not on every later visit
        if kw.get("step") and kw["step"] <= 8:
            kw.setdefault("prep", kw["step"])            # Prepare reopens at the step you were at
            kw.setdefault("prep_ds", kw.get("ds", s.get("ds")))
        s.update(kw)
        self.s = s
        if s.get("view") == "flow" and s.get("ds"):
            import home
            home.remember(s["ds"], s.get("step") or 0)
        self.draw()

    def on_show(self, **kw):
        if "find" in kw:
            self.m_find.value = kw["find"]
            return self.go(view="edit")
        if kw.get("run"):
            self.route("run|" + kw["run"])
        elif kw.get("ds"):
            part = kw.get("part")
            step = kw.get("step") or (V.PART_STEP.get(part) if part else None) or F.next_step(kw["ds"])
            self.go(view="flow", mode="edit", ds=kw["ds"], step=int(step))
        elif kw.get("new"):
            self.go(view="new", src="")
        elif kw.get("view"):
            self.go(view=kw["view"])
        else:
            self.draw()

    def route(self, v: str):
        """Clicks inside the HTML parts (lsSend('ds_box', …))."""
        act, _, arg = (v or "").partition("|")
        s = self.s
        if act == "edit":                                 # a dataset tile: open its page
            return self.open_ds(arg)
        if act in ("open", "step") and s.get("ds"):
            try:
                return self.go(view="flow", step=int(arg))
            except ValueError:
                return
        if act == "part" and s.get("ds"):
            n = ((s.get("prep") if s.get("prep_ds") == s["ds"] and s.get("prep") else min(F.next_step(s["ds"]), 8))
                 if arg == "prepare" else V.PART_STEP.get(arg, 9))
            return self.go(view="flow", step=n)
        if act == "dspart":
            ds, _, n = arg.rpartition("|")
            if ds in S.list_lora_dirs():
                self.go(view="flow", mode="edit", ds=ds, step=int(n or 11))
            return
        if act == "run":
            import training_ui as TU
            have = S.list_lora_dirs()
            ds = TU.dataset_of_log(arg)
            ds = ds if ds in have else (s.get("ds") or (have[0] if have else ""))
            if ds:
                self.go(view="flow", mode="edit", ds=ds, step=10, open_run=arg)
            return
        if act == "folder":
            self.s["src"] = arg
            self.n_src = arg
            self.n_name.value = F.safe_name(arg.replace("_batch", "").replace("_raw", ""))
            self.n_list.content = V.new_list_html(arg)
            self.draw_tree()
            return
        if act == "pin" and s.get("ds"):
            state, _, path = arg.partition("|")
            F.set_pin(s["ds"], path, state)
            return
        handler = getattr(self, "on_" + act, None)       # clicks a step handles itself (pick, fix, rmode …)
        if handler:
            handler(arg)

    # ------------------------------------------------------------------ drawing
    def draw(self):
        s = self.s
        view = s.get("view", "edit")
        self.p_new.set_visibility(view == "new")
        self.p_edit.set_visibility(view == "edit")
        self.p_flow.set_visibility(view == "flow")
        self.draw_menu()
        if view == "new":
            self.n_list.content = V.new_list_html(s.get("src", ""))
            self.draw_tree()
        elif view == "edit":
            self.draw_list()
        elif view == "flow":
            self.bar.content = V.bar_html(s)
            self.box.clear()
            with self.box:
                build = self.steps.get(s.get("step"))
                if build:
                    build()
                else:
                    note(f"Step {s.get('step')} ({NAMES.get(s.get('step'), '')}) comes in the next part of the move.")

    def open_ds(self, ds: str):
        """A dataset's page, at the step you were at (or its next step)."""
        if ds not in S.list_lora_dirs():
            return
        s = self.s
        step = s.get("prep") if s.get("prep_ds") == ds and s.get("prep") else min(F.next_step(ds), 8)
        self.go(view="flow", mode="edit", ds=ds, step=int(step))

    def draw_list(self):
        rows = F.overview(S.list_lora_dirs())
        find = (self.m_find.value or "").strip().lower()
        only_new = self.m_new.value
        out = []
        for r in rows:
            if find and find not in r["ds"].lower():
                continue
            if only_new and self.lora_count(r["ds"]):
                continue
            out.append(f'<button class="ds-erow" onclick="{V.send("edit|" + r["ds"])}"><b>{V.esc(r["ds"])}</b>'
                       f'<span class="ls-note">{r["n"]} pictures · next: {r["next"]} {V.esc(r["label"])}</span></button>')
        empty = "No datasets match." if (find or only_new) else "No datasets."
        self.e_list.content = '<div class="ds-egrid">' + ("".join(out) or f'<p class="ls-empty">{empty}</p>') + "</div>"

    def build_step1(self):
        """Step 1 Folder: the dataset folder (missing folders are made), adding pictures, and every step's state."""
        ds = self.s.get("ds")
        try:
            made = F.ensure_folders(ds)                   # step 1's folders, made when missing
        except OSError as e:
            made = [f"could not: {e}"]
        with ui.card().props("flat bordered").classes("ls-card w-full gap-2"):
            ui.label("The dataset folder").classes("text-lg font-semibold")
            ui.html(f'<span class="ls-chip ok">{V.esc(F.folder_text(ds))}</span>'
                    + (f' <span class="ls-note">made now: {V.esc(", ".join(made))}</span>' if made else ""),
                    sanitize=False)
        with ui.expansion("Add pictures from a folder").classes("w-full ls-card"):
            note("Pictures that are good enough and not already in the dataset are copied into its repeat "
                 "folder (and into _originals/). Then sort and select again.")
            with ui.row().classes("w-full items-center gap-2"):
                self.e_add_dir = ui.input("Folder", placeholder="~/Pictures/new batch") \
                    .props("outlined dense").classes("grow")
                self.e_add_sub = ui.checkbox("Include subfolders", value=True)
                ui.button("Add pictures", on_click=self.add_pics).props("outline dense no-caps")
            self.e_add_msg = ui.markdown("")
        sts = F.step_states(ds)
        nxt = F.next_step(ds)
        rows = []
        for n, key, label in F.STEPS[1:]:
            st, text = sts.get(key, ("todo", ""))
            chip = {"done": "ok", "warn": "warn", "optional": ""}.get(st, "")
            rows.append(f'<div class="ds-srow{" ds-srow-next" if n == nxt else ""}"><span class="ds-snum">{n}</span>'
                        f'<b>{V.esc(label)}</b><span class="ls-chip {chip}">{V.esc(text)}</span><span class="ds-sp"></span>'
                        f'<button class="tr-btn tr-small" onclick="{V.send(f"open|{n}")}">'
                        f'{"Next: open" if n == nxt else "Open"}</button></div>')
        ui.html('<div class="ds-slist">' + "".join(rows) + "</div>", sanitize=False).classes("w-full")

    # ------------------------------------------------------------------ the left menu
    STEP_ICONS = {1: "folder", 2: "tune", 3: "fact_check", 4: "sort", 5: "checklist", 6: "crop", 7: "subtitles",
                  8: "rate_review"}

    def build_menu(self):
        import sidemenu as SM
        menu = self.shell.menu
        with menu.section("datasets"):
            self.m_find = menu.search("Search datasets", lambda: self.search_changed())
            self.m_all = SM.Item("view_list", "All datasets", on_click=lambda: self.go(view="edit"))
            self.m_newds = SM.Item("add", "New dataset", on_click=lambda: self.go(view="new", src=""))
            with ui.column().classes("w-full gap-0") as self.m_filters:
                SM.header("Show")
                with ui.column().classes("ls-mfilters gap-0 q-mini-drawer-hide"):
                    self.m_new = ui.checkbox("Only not trained yet", on_change=lambda: self.draw_list()).props("dense")
            with ui.column().classes("w-full gap-0") as self.m_ds:
                self.m_card = ui.html("", sanitize=False).classes("ls-mcard")
                SM.header("Prepare")
                self.m_steps = {}
                for n, key, label in F.STEPS[:8]:
                    self.m_steps[n] = SM.Item(self.STEP_ICONS.get(n, "circle"), f"{n} {label}", sub=True,
                                              on_click=lambda n=n: self.go(view="flow", step=n))
                SM.header("Next")
                self.m_train = SM.Item("play_circle", "Train", on_click=lambda: self.go(view="flow", step=9))
                self.m_res = SM.Item("bar_chart", "Results", on_click=lambda: self.go(view="flow", step=10))
                self.m_lora = SM.Item("collections", "LoRA", on_click=lambda: self.go(view="flow", step=11))

    def search_changed(self):
        if self.s.get("view") != "edit":
            self.go(view="edit")
        else:
            self.draw_list()

    def lora_count(self, ds: str) -> int:
        """Finished LoRA files in the dataset's model/ folder (no save points), looked at every 30 s at most."""
        import re
        import time
        hit = self._lora_n.get(ds)
        if hit and time.time() - hit[0] < 30:
            return hit[1]
        n = 0
        try:
            d = S.lora_path(ds) / "model"
            n = sum(1 for f in d.glob("*.safetensors") if not re.search(r"-\d{6}$", f.stem)) if d.is_dir() else 0
        except OSError:
            pass
        self._lora_n[ds] = (time.time(), n)
        return n

    def draw_menu(self):
        s = self.s
        view, ds, step = s.get("view", "edit"), s.get("ds"), s.get("step") or 2
        self.m_all.active(view == "edit")
        self.m_newds.active(view == "new")
        self.m_all.set_right(str(len(S.list_lora_dirs())))
        self.m_filters.set_visibility(view == "edit")
        flow = view == "flow" and bool(ds)
        self.m_ds.set_visibility(flow)
        if not flow:
            return
        import features as FT
        m = F.meta(ds)
        labels = F.type_labels()
        kind = " + ".join(labels.get(t, t) for t in m["types"]) or "type not set"
        self.m_card.content = (f"{V.esc(ds)}<small>{V.esc(kind)} · {V.esc(F.PURPOSE_LABEL.get(m['purpose'], m['purpose']))}"
                               f" · {len(F.images(ds))} pictures</small>")
        states = F.step_states(ds)
        for n, key, label in F.STEPS[:8]:
            st = states.get(key, ("todo", ""))[0] if n > 1 else "done"
            it = self.m_steps[n]
            it.active(step == n)
            off = key == "tags" and not FT.have("ollama") and any(x["style"] != "tags" for x in F.caption_sets(ds))
            if st == "done" and step != n:
                it.set_icon("check_circle", "ok")
                it.set_right("✓")
            elif st == "warn":
                it.set_icon("error", "warn")
                it.set_right("!")
            else:
                it.set_icon(self.STEP_ICONS.get(n, "circle"), "off" if off else "")
                it.set_right("")
        self.m_train.active(step == 9)
        self.m_res.active(step == 10)
        self.m_lora.active(step == 11)
        nl = self.lora_count(ds)
        self.m_lora.set_right(f"{nl} file{'s' if nl != 1 else ''}" if nl else "")

    def draw_tree(self):
        self.n_tree.content = V.new_tree(self.n_name.value, self.n_rep.value, self.s.get("src", ""), self.n_out.value)

    # ------------------------------------------------------------------ list and new
    def create(self):
        ds, msg = F.create(self.s.get("src", ""), self.n_name.value, self.n_rep.value, self.n_out.value)
        self.n_msg.content = msg
        if ds:
            self.go(view="flow", mode="new", ds=ds, step=2, src="")

    async def add_pics(self):
        ds = self.s.get("ds")
        folder = os.path.expanduser((self.e_add_dir.value or "").strip())
        if not ds or not Path(folder).is_dir():
            self.e_add_msg.content = f"{folder or 'That'} is not a folder." if ds else "Pick a dataset first."
            return
        self.e_add_msg.content = "Looking at the pictures …"
        self.e_add_msg.content = await run.io_bound(F.add_pictures, ds, folder, self.e_add_sub.value)
        self.draw_menu()

    # ------------------------------------------------------------------ build
    def build(self):
        self._lora_n: dict = {}
        self.build_menu()
        with ui.column().classes("ls-page gap-3"):
            self.build_new()
            self.build_list()
            with ui.column().classes("w-full gap-2") as self.p_flow:
                self.bar = ui.html("", sanitize=False).classes("w-full")
                self.box = ui.column().classes("w-full gap-3")
        import ds_steps
        ds_steps.attach(self)
        self.steps[1] = self.build_step1
        ui.add_head_html("<script>" + V.GRID_JS + "</script>")
        ui.on("ls_ds_box", lambda e: self.route(str(e.args)))
        ui.on("ls_rs_box", lambda e: getattr(self, "on_rs", lambda v: None)(str(e.args)))
        ui.on("ls_wm_tool", lambda e: getattr(self, "on_wm_tool", lambda v: None)(str(e.args)))
        self.draw()

    def build_new(self):
        with ui.column().classes("w-full gap-2") as self.p_new:
            ui.label("New dataset").classes("ls-h")
            with ui.row().classes("w-full no-wrap items-start gap-4"):
                with ui.card().props("flat bordered").classes("ls-card grow basis-3/5 gap-2"):
                    ui.label("Pick a folder in your dataset folder").classes("text-lg font-semibold")
                    note("A folder with only pictures becomes a new dataset. Folders that already are datasets are in "
                         "the dataset list.")
                    self.n_list = ui.html("", sanitize=False).classes("w-full")
                    self.n_out = ui.input("Or a folder somewhere else (the pictures are copied in)",
                                          placeholder="~/Pictures/…", on_change=lambda: self.draw_tree()) \
                        .props("outlined dense").classes("w-full")
                with ui.card().props("flat bordered").classes("ls-card grow basis-2/5 gap-2"):
                    ui.label("What happens").classes("text-lg font-semibold")
                    with ui.row().classes("w-full no-wrap gap-2"):
                        self.n_name = ui.input("Dataset name", on_change=lambda: self.draw_tree()) \
                            .props("outlined dense").classes("grow")
                        self.n_rep = ui.number("Repeats", value=10, min=1, precision=0,
                                               on_change=lambda: self.draw_tree()).props("outlined dense").classes("w-28")
                    self.n_tree = ui.html("", sanitize=False)
                    ui.button("Create and continue →", on_click=self.create).props("unelevated no-caps color=primary")
                    self.n_msg = ui.markdown("")

    def build_list(self):
        with ui.column().classes("w-full gap-2") as self.p_edit:
            self.e_list = ui.html("", sanitize=False).classes("w-full")


def build(shell):
    d = Datasets(shell)
    d.build()
    return d

