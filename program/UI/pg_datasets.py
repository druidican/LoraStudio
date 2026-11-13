# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Datasets: the dataset list (+ New dataset), Improvements, and one page per dataset with four parts: Prepare (steps
1-8), Train (9), Results (its runs) and LoRA (its LoRA files).

The page keeps one state (self.s, the same keys as before: view, mode, ds, step, …). A change of state redraws the bar,
the step strip and the open step only: the open step's controls are built anew in self.box, which is quick in NiceGUI.
Pictures, grids and lists are drawn as HTML by datasets_ui's renderers; their clicks come back as ls_ds_box events
(route). What each step does to the dataset folder is in dsflow.py.
"""
from __future__ import annotations
__version__ = "1.131126b"  # LoRA Studio version, see version.py

import os
from pathlib import Path

from nicegui import run, ui

import datasets_ui as V
import dsflow as F
import sdgui as S
from widgets import note

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
        if act == "edit":
            return self.go(view="edit", ds=arg, mode="edit")
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
        self.p_imp.set_visibility(view == "imp")
        self.p_flow.set_visibility(view == "flow")
        self.b_edit.classes(**({"add": "ls-nav-on"} if view in ("edit", "new") else {"remove": "ls-nav-on"}))
        self.b_imp.classes(**({"add": "ls-nav-on"} if view == "imp" else {"remove": "ls-nav-on"}))
        if view == "new":
            self.n_list.content = V.new_list_html(s.get("src", ""))
            self.draw_tree()
        elif view == "edit":
            self.draw_list()
        elif view == "imp":
            self.draw_imp()
        elif view == "flow":
            self.bar.content = V.bar_html(s)
            self.stepper.content = V.stepper_html(s) if (s.get("step") or 2) <= 8 else ""
            self.box.clear()
            with self.box:
                build = self.steps.get(s.get("step"))
                if build:
                    build()
                else:
                    note(f"Step {s.get('step')} ({NAMES.get(s.get('step'), '')}) comes in the next part of the move.")

    def draw_list(self):
        rows = F.overview(S.list_lora_dirs())
        find = (self.e_find.value or "").strip().lower()
        out = []
        for r in rows:
            if find and find not in r["ds"].lower():
                continue
            on = " ds-frow-on" if r["ds"] == self.s.get("ds") else ""
            out.append(f'<button class="ds-erow{on}" onclick="{V.send("edit|" + r["ds"])}"><b>{V.esc(r["ds"])}</b>'
                       f'<span class="ls-note">{r["n"]} · next: {r["next"]} {V.esc(r["label"])}</span></button>')
        self.e_list.content = '<div class="ds-elist">' + ("".join(out) or '<p class="ls-empty">No datasets.</p>') + "</div>"
        ds = self.s.get("ds")
        self.e_open.set_visibility(bool(ds))
        if not ds:
            self.e_head.content, self.e_steps.content = '<p class="ls-empty">Pick a dataset on the left.</p>', ""
            return
        m = F.meta(ds)
        labels = F.type_labels()
        self.e_head.content = (f'<div class="ds-ehead"><b>{V.esc(ds)}</b> · '
                               f'{V.esc(" + ".join(labels.get(t, t) for t in m["types"]) or "type not set")} · '
                               f'{V.esc(F.PURPOSE_LABEL.get(m["purpose"], m["purpose"]))} · {len(F.images(ds))} pictures</div>')
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
        self.e_steps.content = '<div class="ds-slist">' + "".join(rows) + "</div>"

    def draw_tree(self):
        self.n_tree.content = V.new_tree(self.n_name.value, self.n_rep.value, self.s.get("src", ""), self.n_out.value)

    def draw_imp(self):
        dss = S.list_lora_dirs()
        cards = []
        for ds in dss:
            try:
                items = F.improvements(ds)
            except Exception as e:                       # noqa: BLE001
                items = [("error", str(e))]
            if not items:
                continue
            body = "".join(f'<div class="ds-improw"><span class="ls-chip warn">{V.esc(t)}</span><span>{V.esc(x)}</span></div>'
                           for t, x in items)
            cards.append(f'<div class="ls-card ds-impcard"><div class="ds-imph"><label class="ds-tick ds-imptick">'
                         f'<input type="checkbox" data-ds="{V.esc(ds)}"><b>{V.esc(ds)}</b></label>'
                         f'<button class="tr-btn tr-small" onclick="{V.send("edit|" + ds)}">Open</button></div>{body}</div>')
        self.i_head.content = f'<div class="rs-big">{len(cards)} of {len(dss)} datasets could be better</div>'
        self.i_html.content = "".join(cards) or '<p class="ls-empty">Nothing to improve that is known.</p>'
        self.i_what.content = V.batch_what()

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
        self.e_add_msg.content = await run.io_bound(add_pictures, ds, folder, self.e_add_sub.value)
        self.draw_list()

    # ------------------------------------------------------------------ build
    def build(self):
        with ui.column().classes("ls-page gap-3"):
            with ui.row().classes("gap-2"):
                self.b_edit = ui.button("Datasets", on_click=lambda: self.go(view="edit")).props("flat no-caps").classes("ls-nav")
                self.b_imp = ui.button("Improvements", on_click=lambda: self.go(view="imp")).props("flat no-caps").classes("ls-nav")
            self.build_new()
            self.build_list()
            self.build_imp()
            with ui.column().classes("w-full gap-2") as self.p_flow:
                self.bar = ui.html("", sanitize=False).classes("w-full")
                self.stepper = ui.html("", sanitize=False).classes("w-full")
                self.box = ui.column().classes("w-full gap-3")
        import ds_steps
        ds_steps.attach(self)
        ui.add_head_html("<script>" + V.GRID_JS + "</script>")
        ui.on("ls_ds_box", lambda e: self.route(str(e.args)))
        self.draw()

    def build_new(self):
        with ui.column().classes("w-full gap-2") as self.p_new:
            ui.button("← All datasets", on_click=lambda: self.go(view="edit")).props("flat dense no-caps")
            ui.html(V.stepper_html({"ds": "", "step": 1}), sanitize=False).classes("w-full")
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
        with ui.row().classes("w-full no-wrap items-start gap-6") as self.p_edit:
            with ui.column().classes("w-72 flex-none gap-2"):
                ui.button("+ New dataset", on_click=lambda: self.go(view="new", src="")) \
                    .props("unelevated no-caps color=primary")
                self.e_find = ui.input("Find a dataset", on_change=lambda: self.draw_list()) \
                    .props("outlined dense clearable").classes("w-full")
                self.e_list = ui.html("", sanitize=False).classes("w-full")
            with ui.column().classes("grow min-w-0 gap-2"):
                self.e_head = ui.html("", sanitize=False)
                with ui.column().classes("w-full gap-2") as self.e_open:
                    with ui.expansion("Add pictures from a folder").classes("w-full ls-card"):
                        note("Pictures that are good enough and not already in the dataset are copied into its repeat "
                             "folder (and into _originals/). Then sort and select again.")
                        with ui.row().classes("w-full items-center gap-2"):
                            self.e_add_dir = ui.input("Folder", placeholder="~/Pictures/new batch") \
                                .props("outlined dense").classes("grow")
                            self.e_add_sub = ui.checkbox("Include subfolders", value=True)
                            ui.button("Add pictures", on_click=self.add_pics).props("outline dense no-caps")
                        self.e_add_msg = ui.markdown("")
                    self.e_steps = ui.html("", sanitize=False).classes("w-full")

    def build_imp(self):
        with ui.column().classes("w-full gap-2") as self.p_imp:
            with ui.row().classes("w-full items-center"):
                self.i_head = ui.html("", sanitize=False).classes("grow")
                ui.button("Refresh", on_click=self.draw_imp).props("flat dense no-caps")
            note("From what is known already; nothing is scanned on this page.")
            with ui.card().props("flat bordered").classes("ls-card w-full gap-2"):
                self.i_what = ui.html("", sanitize=False)
                self.i_tools = ui.row().classes("items-center gap-2")   # Fix the ticked: ds_steps.attach
                self.i_msg = ui.markdown("")
                self.i_prog = ui.html("", sanitize=False).classes("w-full")
            self.i_html = ui.html("", sanitize=False).classes("w-full")


def add_pictures(ds, folder, sub) -> str:
    """Copy the good pictures of a folder into the dataset's repeat folder (and _originals/)."""
    import picker
    rows = S.dataset_stats(ds)["rows"]
    if not rows:
        return "This dataset has no repeat folder yet."
    res = picker.scan(folder, bool(sub), F.images(ds), F.min_side(F.meta(ds)["types"]), F.meta(ds)["purpose"],
                      "", "", False, lambda p: None, lambda p, d: None)
    pick = [e["path"] for e in res["pics"] if e["pick"]]
    copied, errs = picker.copy_pictures(pick, rows[0]["folder"])
    F.ensure_originals(ds, copied)
    skipped = len(res["pics"]) - len(pick)
    return (f"✔ {len(copied)} pictures copied into {rows[0]['folder'].name}/" + (f"; {skipped} left out (too small, "
            "blurry or already in the dataset)" if skipped else "") + ". Sort and select again (steps 4 and 5).")


def build(shell):
    d = Datasets(shell)
    d.build()
    return d

