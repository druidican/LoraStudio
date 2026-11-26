# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""The dataset page's Review and export part: the test pictures of a run, one row per save point; you pick the row
to keep, check what goes to LoRA Manager and export it (lmexport.py). The status strip and Home › Just trained open
it after a run's test pictures are made."""
from __future__ import annotations
__version__ = "1.271126"  # LoRA Studio version, see version.py


from nicegui import run, ui

import lmexport as LX
from widgets import note, pic_url


class Review:
    def __init__(self, page):
        self.page = page
        self.ds = page.s["ds"]
        self.v: dict = {}
        self.pick = ""
        self.preview = ""
        self.job = None

    # ------------------------------------------------------------------ data
    def load(self, log: str):
        self.v = LX.view(log) or {}
        self.pick = self.v.get("pick", "")
        row = self.row(self.pick)
        self.preview = next((p for p in (row or {}).get("pics", []) if p), "")

    def row(self, label):
        return next((r for r in self.v.get("rows", []) if r["label"] == label), None)

    # ------------------------------------------------------------------ drawing
    def draw(self):
        import comfytest as CT
        v = self.v
        self.rows_box.clear()
        self.result.set_visibility(False)
        self.main.set_visibility(bool(v))
        if not v:
            return
        self.title.text = f"Review and export: {v['out']}"
        self.sub.text = (f"trained {v['started']}" + (f" · test pictures made {v['at']}" if v.get("at") else ""))
        job = CT.JOBS.get(v["log"]) or {}
        self.job = job.get("state")
        self.busy.text = ("⏳ Making test pictures in ComfyUI … " + (job.get("text") or "")
                          + (f" ({job['done']} of {job['total']})" if job.get("total") else "")) \
            if job.get("state") == "running" else ("Waiting for ComfyUI: the test pictures are made as soon as it "
                                                   "answers." if v["log"] in CT.PENDING else "")
        with self.rows_box:
            for r in v["rows"]:
                on = r["label"] == self.pick
                with ui.row().classes("rv-row w-full items-center gap-3 no-wrap" + (" on" if on else "")) as box:
                    with ui.column().classes("w-36 flex-none gap-0"):
                        ui.label(("◉ " if on else "○ ") + r["label"]).classes("font-semibold text-base")
                        if r["loss"] is not None:
                            note(f"loss {r['loss']:.3f}")
                        if r["best"]:
                            ui.label("★ try first").classes("ls-chip ok")
                        if v.get("kept") == r["label"]:
                            ui.label("✔ exported").classes("ls-chip")
                        if not r["ok"]:
                            note("file gone")
                    with ui.row().classes("gap-2 no-wrap overflow-x-auto"):
                        for p in r["pics"]:
                            if p:
                                img = ui.image(pic_url(p, 300)).classes(
                                    "rv-pic w-24 h-24 rounded flex-none cursor-pointer" + (" on" if p == self.preview else "")) \
                                    .props("fit=cover")
                                img.on("click", lambda _, lab=r["label"], p=p: self.choose(lab, p))
                                img.on("dblclick", lambda _, p=p: self.big(p))
                            else:
                                ui.element("div").classes("w-24 h-24 rounded flex-none bg-gray-300/30")
                if r["ok"]:
                    box.on("click", lambda _, lab=r["label"]: self.choose(lab, None))
                else:
                    box.classes("opacity-50")
            if not v["rows"]:
                note("No test pictures for this run yet.")
        self.draw_form()

    def draw_form(self):
        v = self.v
        r = self.row(self.pick) or {}
        n = len([p for p in r.get("pics", []) if p])
        self.f_name.text = f"{v['out']}.safetensors"
        self.f_ex.text = f"the whole row of {self.pick or '—'}: {n} picture{'s' if n != 1 else ''}"
        self.f_prev.set_source(pic_url(self.preview, 200) if self.preview else "")
        self.f_prev.set_visibility(bool(self.preview))
        self.f_lm.text = ("" if v.get("lm_root") else "LoRA Manager's example picture folder is not known: set it in "
                          "Collection › Maintenance; the LoRA and its data are exported anyway.")
        size = v.get("cleanup_size") or 0
        self.f_clean.text = (f"clean up the other save points and states ({size / 1024 ** 3:.1f} GB)" if size >= 1e8
                             else f"clean up the other save points and states ({size / 1024 ** 2:.0f} MB)")
        self.f_done.text = f"mark {v['type']} {v['out']} Completed"
        self.export_b.set_enabled(bool(r.get("ok")))

    def choose(self, label, pic):
        r = self.row(label)
        if not r or not r["ok"]:
            return
        if label != self.pick:
            self.pick = label
            self.preview = pic or next((p for p in r["pics"] if p), "")
        elif pic:
            self.preview = pic
        self.draw()

    def big(self, path):
        with ui.dialog() as d, ui.card().classes("p-2 max-w-[95vw]"):
            ui.image(pic_url(path, 1600)).classes("max-h-[85vh] w-[80vw]").props("fit=contain")
        d.open()

    def open_run(self, log):
        if log:
            self.load(log)
        self.f_trig.value = self.v.get("trigger", "")
        self.f_str.value = self.v.get("strength", 1.0)
        self.f_dest.value = self.v.get("dest", "")
        self.f_base.text = self.v.get("base", "")
        self.f_train.text = self.v.get("training", "")
        self.f_rate.value = self.v.get("rating", 0)
        self.f_note.value = ""
        self.f_donebox.value = True
        self.f_cleanbox.value = False
        self.draw()

    # ------------------------------------------------------------------ actions
    def make_more(self):
        import comfytest as CT
        if not self.v:
            return
        CT.WHICH[self.v["log"]] = "around"
        ui.notify(CT.make_in_background(self.v["log"]))
        self.draw()

    def tick(self):
        import comfytest as CT
        if not self.v or self.main.is_deleted:
            return
        st = (CT.JOBS.get(self.v["log"]) or {}).get("state")
        if st == "running":                              # new pictures come in: keep your pick
            self.v = LX.view(self.v["log"]) or self.v
            self.draw()
        elif st != self.job:                             # they are done
            self.load(self.v["log"])
            self.draw()

    async def export(self):
        from widgets import confirm
        v = self.v
        if not v or not self.pick:
            return
        if self.f_cleanbox.value and not await confirm(
                f"Export {self.pick} and move the other save points and resume states of this run to the Trash?",
                "Export and clean up"):
            return
        form = {"dest": self.f_dest.value, "trigger": self.f_trig.value, "strength": self.f_str.value,
                "preview": self.preview, "note": self.f_note.value, "rating": self.f_rate.value,
                "done": self.f_donebox.value, "cleanup": self.f_cleanbox.value}
        self.export_b.disable()
        self.emsg.text = "⏳ Exporting …"
        steps = await run.io_bound(LX.export, v["log"], self.pick, form)
        self.emsg.text = ""
        self.show_result(steps)

    def show_result(self, steps):
        self.main.set_visibility(False)
        self.result.set_visibility(True)
        self.r_title.text = f"Exported: {self.v['out']}" if steps and steps[-1][0] == "✔" else "Export: see below"
        self.r_list.clear()
        colour = {"✔": "text-teal-700", "!": "text-amber-700", "❌": "text-red-700"}
        with self.r_list:
            for state, title, detail in steps:
                with ui.row().classes("w-full items-start gap-3 no-wrap"):
                    ui.label(state).classes(f"w-5 flex-none font-bold {colour.get(state, '')}")
                    with ui.column().classes("gap-0 min-w-0"):
                        ui.label(title)
                        if detail:
                            note(detail).classes("break-words")
        r = self.row(self.pick) or {}
        self.r_prev.set_source(pic_url(self.preview, 300) if self.preview else "")
        self.r_name.text = self.v["out"]
        self.r_meta.text = (f"{self.v.get('base', '')} · trigger: {self.f_trig.value} · strength {self.f_str.value}")
        self.r_thumbs.clear()
        with self.r_thumbs:
            for p in [x for x in r.get("pics", []) if x][:8]:
                ui.image(pic_url(p, 120)).classes("w-11 h-11 rounded").props("fit=cover")

    def back(self):
        self.open_run(self.v.get("log", ""))

    # ------------------------------------------------------------------ build
    def build(self):
        runs = LX.runs_of(self.ds)
        want = self.page.s.get("review")
        log = want if want and any(r["log"] == want for r in runs) else (runs[0]["log"] if runs else "")
        if want and not log:
            log = want                                   # its pictures are still being made
        with ui.row().classes("w-full items-baseline gap-3"):
            self.title = ui.label("Review and export").classes("text-lg font-semibold")
            self.sub = note("")
            ui.space()
            if len(runs) > 1:
                ui.select({r["log"]: f"{r['name']} · {r['started']}" for r in runs}, value=log, label="Run",
                          on_change=lambda e: self.open_run(e.value)).props("outlined dense").classes("w-80")
        if not log:
            note("No run of this dataset has test pictures yet. They are made by themselves after a run (Settings › "
                 "Test pictures, ComfyUI running); Results › a run › Review and export makes them for older runs.")
        self.busy = note("")
        with ui.row().classes("w-full no-wrap items-start gap-4") as self.main:
            with ui.card().props("flat bordered").classes("ls-card grow min-w-0 gap-1"):
                ui.label("1  Pick the version").classes("font-semibold")
                note("Every save point with the test pictures LoRA Studio made (the same prompts and seeds in each "
                     "row). Click a row to keep it, a picture for the preview; double-click a picture to see it big.")
                self.rows_box = ui.column().classes("w-full gap-1")
                ui.button("Make pictures of the save points around it", on_click=self.make_more) \
                    .props("flat dense no-caps")
            with ui.card().props("flat bordered").classes("ls-card w-[27rem] flex-none gap-1"):
                ui.label("2  What goes to LoRA Manager").classes("font-semibold")
                with ui.grid(columns="9rem 1fr").classes("w-full items-center gap-x-3 gap-y-1"):
                    note("File name")
                    self.f_name = ui.label("")
                    note("Copy to")
                    self.f_dest = ui.input().props("outlined dense").classes("w-full")
                    note("Base model")
                    self.f_base = ui.label("")
                    note("Trigger words")
                    self.f_trig = ui.input().props("outlined dense").classes("w-full")
                    note("Strength")
                    self.f_str = ui.number(value=1.0, step=0.05, min=0, max=3, precision=2).props("outlined dense") \
                        .classes("w-28")
                    note("Preview")
                    with ui.row().classes("items-center gap-2 no-wrap"):
                        self.f_prev = ui.image("").classes("rv-pic on w-14 h-14 rounded cursor-pointer").props("fit=cover")
                        self.f_prev.on("click", lambda: self.preview and self.big(self.preview))
                        note("click a picture in the row to choose")
                    note("Example pictures")
                    self.f_ex = ui.label("")
                    note("Training")
                    self.f_train = note("")
                    note("Notes")
                    self.f_note = ui.input(placeholder="your note …").props("outlined dense").classes("w-full")
                    note("Your rating")
                    self.f_rate = ui.toggle({0: "–", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5"}, value=0) \
                        .props("dense unelevated")
                    note("After export")
                    with ui.column().classes("gap-0"):
                        self.f_done = self.f_donebox = ui.checkbox("", value=True).props("dense")
                        self.f_clean = self.f_cleanbox = ui.checkbox("", value=False).props("dense")
                self.f_lm = note("")
                with ui.row().classes("items-center gap-2 mt-2"):
                    self.export_b = ui.button("Select and export to LoRA Manager", on_click=self.export) \
                        .props("unelevated no-caps color=primary")
                    ui.button("Not now", on_click=lambda: self.page.go(step=10)).props("flat dense no-caps")
                self.emsg = note("")
        with ui.row().classes("w-full no-wrap items-start gap-4") as self.result:
            with ui.card().props("flat bordered").classes("ls-card grow min-w-0 gap-2"):
                self.r_title = ui.label("").classes("text-lg font-semibold")
                self.r_list = ui.column().classes("w-full gap-1")
                with ui.row().classes("gap-2"):
                    ui.button("Open in the Collection", on_click=lambda: self.page.go(step=11)) \
                        .props("unelevated dense no-caps color=primary")
                    ui.button("Back to the review", on_click=self.back).props("outline dense no-caps")
            with ui.card().props("flat bordered").classes("ls-card w-[27rem] flex-none gap-1"):
                ui.label("In LoRA Manager").classes("font-semibold")
                with ui.row().classes("gap-3 no-wrap"):
                    self.r_prev = ui.image("").classes("w-28 h-36 rounded flex-none").props("fit=cover")
                    with ui.column().classes("gap-1 min-w-0"):
                        self.r_name = ui.label("").classes("font-semibold")
                        self.r_meta = note("")
                        self.r_thumbs = ui.row().classes("gap-1")
        self.result.set_visibility(False)
        self.main.set_visibility(False)
        if log:
            self.load(log)
            self.open_run("")
        ui.timer(3.0, self.tick)


def step12(page):
    Review(page).build()
