# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""The dataset page's Results part (step 10): its runs, one run (advice, save points, test pictures, clean up,
rating, report, TensorBoard) and comparing runs. The data and the HTML parts are training_ui's; clicks inside them
come back as ls_rs_box events (open, copy, cmpuse, best)."""
from __future__ import annotations
__version__ = "1.131126b"  # LoRA Studio version, see version.py

from nicegui import ui

import features as FT
import sdgui as sg
import training_ui as T
from widgets import note

PLOT = {"grid": {"left": 50, "right": 20, "top": 20, "bottom": 30}, "tooltip": {"trigger": "axis"},
        "legend": {"show": False}, "xAxis": {"type": "value", "name": "epoch", "minInterval": 1},
        "yAxis": {"type": "value", "scale": True}, "series": []}


def _series(points) -> list:
    """echarts lines: points is [(epoch, loss)] (one run) or {run name: [(epoch, loss)]} (compare)."""
    if not points:
        return []
    if isinstance(points, list):
        return [{"type": "line", "name": "loss", "data": [[float(e), float(v)] for e, v in points], "color": "#E8A33D"}]
    return [{"type": "line", "name": str(k), "data": [[float(e), float(v)] for e, v in pts]} for k, pts in points.items()]


class Results:
    def __init__(self, page):
        self.page = page
        self.ds = page.s["ds"]
        self.st: dict = {}
        self.logs: list = []

    # ------------------------------------------------------------------ the list
    def draw_list(self):
        self.lst.content = T.results_html(self.f_ds.value, self.f_res.value, self.f_notc.value)

    def show(self, view: str):
        self.list_col.set_visibility(view == "list")
        self.one.set_visibility(view == "one")
        self.cmp_col.set_visibility(view == "cmp")

    # ------------------------------------------------------------------ one run
    def open(self, log: str):
        h, t, p, s, c, th, tl, d, st = T.open_run(log)
        self.st = st
        self.head.content, self.tryf.content, self.sp.content, self.clean_md.content = h, t, s, c
        self.this.content, self.tail.value, self.dest.value = th, tl, d
        self.plot.options["series"] = _series(p)
        self.plot.update()
        self.cmsg.content = self.clean_msg.content = self.again_msg.content = ""
        logdir = ""
        if st.get("script"):
            try:
                logdir = str(sg.script_paths(st["script"])[1] or "")
            except Exception:                            # noqa: BLE001
                logdir = ""
        self.tb_dir.value = logdir
        info = T.script_info(st["script"]) if st.get("script") and __import__("os").path.isfile(st["script"]) else None
        self.rate.value = T.rating_of(info)
        self.cmp_html.content = T.compare_html(st)
        self.show("one")

    def reopen(self):
        if self.st:
            h, t, p, s, c, th, tl, d, _ = T.open_run(self.st["log"])
            self.sp.content, self.clean_md.content, self.tryf.content = s, c, t
            self.plot.options["series"] = _series(p)
            self.plot.update()

    def click(self, v: str):
        act, _, arg = (v or "").partition("|")
        if act == "open":
            self.open(arg)
        elif act == "copy":
            self.cmsg.content = T.copy_file(self.st, arg, self.dest.value, self.over.value)
            self.reopen()
        elif act == "cmpuse":
            import comfytest as CT
            self.cmsg.content = CT.use(self.st.get("log"), arg, self.dest.value, self.st, self.over.value)
            self.cmp_html.content = T.compare_html(self.st)
        elif act == "best":
            msg = T.toggle_best(arg)
            self.again_msg.content = self.cmp_bmsg.content = msg
            if self.logs:
                self.cmp_head.content = T.compare_head_html(self.logs)

    def tick(self):
        import comfytest as CT
        if self.one.visible and self.st.get("log") and (CT.JOBS.get(self.st["log"]) or {}).get("state") == "running":
            self.cmp_html.content = T.compare_html(self.st)
        if self.cmp_col.visible and self.logs and any(
                (CT.JOBS.get(ln) or {}).get("state") == "running" or ln in CT.PENDING for ln in self.logs):
            self.cmp_pics.content = T.compare_pics_html(self.logs, self.which.value)

    # ------------------------------------------------------------------ comparing
    async def compare(self):
        logs = await ui.run_javascript("Array.from(document.querySelectorAll('#rs_list .rs-tick:checked'))"
                                       ".map(e => e.dataset.log)")
        logs = [x for x in (logs or []) if x][:T.CMP_MAX]
        if len(logs) < 2:
            self.msg0.content = "Tick two to four runs in the list first."
            return
        self.logs = logs
        self.cmp_head.content = T.compare_head_html(logs)
        self.cmp_plot.options["series"] = _series(T.compare_plot(logs))
        self.cmp_plot.options["legend"]["show"] = True
        self.cmp_plot.update()
        self.cmp_set.content = T.compare_settings_html(logs, self.cmp_all.value)
        self.cmp_pics.content = T.compare_pics_html(logs, self.which.value)
        self.msg0.content = ""
        self.show("cmp")

    def report(self):
        import report
        path, text = report.make((self.st or {}).get("log", ""), self.rep_note.value or "")
        ui.clipboard.write(text)
        self.rep_msg.content = (f"✔ Report made (userdata/reports/{path.name}, token and home folder removed) and "
                                "copied: paste it into a chat with Claude, or drop the file there.")

    def new_version(self):
        ds = T.dataset_of_log(self.st.get("log", "")) if self.st else ""
        self.page.go(view="flow", mode="edit", ds=ds if ds in sg.list_lora_dirs() else self.ds, step=9)

    # ------------------------------------------------------------------ build
    def build(self):
        ui.label("Results: the runs of this dataset").classes("text-lg font-semibold")
        with ui.column().classes("w-full gap-2") as self.list_col:
            with ui.row().classes("w-full items-end gap-3"):
                ch = ["All"] + sg.list_lora_dirs()
                self.f_ds = ui.select(ch, value=self.ds if self.ds in ch else "All", label="Runs of",
                                      on_change=lambda: self.draw_list()).props("outlined dense").classes("w-56")
                self.f_res = ui.toggle(["All", "Done", "Failed", "Stopped"], value="All",
                                       on_change=lambda: self.draw_list()).props("dense no-caps unelevated")
                self.f_notc = ui.checkbox("only not copied yet", on_change=lambda: self.draw_list())
                ui.button("Refresh", on_click=self.draw_list).props("flat dense no-caps")
                ui.button("Compare the ticked", on_click=self.compare).props("unelevated dense no-caps color=primary")
            self.msg0 = ui.markdown("")
            self.lst = ui.html("", sanitize=False).classes("w-full").props('id="rs_list"')
        with ui.column().classes("w-full gap-3") as self.cmp_col:
            ui.button("← All runs", on_click=lambda: (self.draw_list(), self.show("list"))).props("flat dense no-caps")
            ui.label("Compare runs").classes("font-semibold")
            self.cmp_head = ui.html("", sanitize=False).classes("w-full")
            self.cmp_bmsg = ui.markdown("")
            with ui.row().classes("w-full no-wrap items-start gap-4"):
                with ui.card().props("flat bordered").classes("ls-card grow basis-3/7 gap-1"):
                    ui.label("Loss per epoch").classes("font-semibold")
                    self.cmp_plot = ui.echart(dict(PLOT, legend={"show": True})).classes("w-full h-64")
                    note("Lower is not always better for image LoRAs; the shape matters more: a run that is still "
                         "falling could train longer, one that rises at the end trained too long.")
                with ui.card().props("flat bordered").classes("ls-card grow basis-4/7 gap-1"):
                    ui.label("Settings difference").classes("font-semibold")
                    self.cmp_set = ui.html("", sanitize=False).classes("w-full")
                    self.cmp_all = ui.checkbox("show all settings, also the ones that are the same",
                                               on_change=lambda: setattr(self.cmp_set, "content", T.compare_settings_html(
                                                   self.logs, self.cmp_all.value)))
            with ui.card().props("flat bordered").classes("ls-card w-full gap-1"):
                ui.label("Test pictures side by side").classes("font-semibold")
                with ui.row().classes("items-center gap-2"):
                    self.which = ui.toggle(["Try first", "Final"], value="Try first", on_change=lambda: setattr(
                        self.cmp_pics, "content", T.compare_pics_html(self.logs, self.which.value))) \
                        .props("dense no-caps unelevated")

                    def same():
                        import comfytest as CT
                        self.cmp_bmsg.content = CT.share_prompts(self.logs or [])
                        self.cmp_pics.content = T.compare_pics_html(self.logs, self.which.value)
                    ui.button("Make test pictures with the same prompts", on_click=same).props("outline dense no-caps")
                    ui.button("Refresh", on_click=lambda: setattr(self.cmp_pics, "content", T.compare_pics_html(
                        self.logs, self.which.value))).props("flat dense no-caps")
                self.cmp_pics = ui.html("", sanitize=False).classes("w-full")
        with ui.column().classes("w-full gap-3") as self.one:
            ui.button("← All runs", on_click=lambda: (self.draw_list(), self.show("list"))).props("flat dense no-caps")
            self.head = ui.html("", sanitize=False).classes("w-full")
            with ui.row().classes("w-full no-wrap items-start gap-4"):
                with ui.card().props("flat bordered").classes("ls-card grow basis-2/5 gap-2"):
                    self.tryf = ui.html("", sanitize=False)
                    with ui.row().classes("gap-2"):
                        ui.button("Copy the one to try first", on_click=lambda: (setattr(self.cmsg, "content", T.copy_best(
                            self.st, self.dest.value, self.over.value)), self.reopen())).props("unelevated dense no-caps color=primary")
                        ui.button("Copy the final", on_click=lambda: (setattr(self.cmsg, "content", T.copy_final(
                            self.st, self.dest.value, self.over.value)), self.reopen())).props("outline dense no-caps")
                    self.dest = ui.input("Copy to").props("outlined dense").classes("w-full")
                    note("The folder where older versions of this LoRA are; an epoch file is copied with a name like "
                         "mycharZITV1-e8.")
                    self.over = ui.checkbox("Replace a different file with the same name (the old one goes to the Trash)")
                    self.cmsg = ui.markdown("")
                with ui.card().props("flat bordered").classes("ls-card grow basis-3/5 gap-1"):
                    ui.label("Loss per epoch").classes("font-semibold")
                    self.plot = ui.echart(dict(PLOT)).classes("w-full h-60")
            ui.label("Save points").classes("font-semibold")
            self.sp = ui.html("", sanitize=False).classes("w-full")
            with ui.card().props("flat bordered").classes("ls-card w-full gap-1"):
                ui.label("Test pictures (ComfyUI)").classes("font-semibold")

                def make():
                    import comfytest as CT
                    if self.st.get("log"):
                        self.cmsg.content = CT.make_in_background(self.st["log"])
                        self.cmp_html.content = T.compare_html(self.st)
                with ui.row().classes("gap-2"):
                    ui.button("Make test pictures", on_click=make).props("unelevated dense no-caps color=primary")
                    ui.button("Refresh", on_click=lambda: setattr(self.cmp_html, "content", T.compare_html(self.st))) \
                        .props("flat dense no-caps")
                self.cmp_html = ui.html("", sanitize=False).classes("w-full")
            with ui.card().props("flat bordered").classes("ls-card w-full gap-1"):
                ui.label("Clean up this run").classes("font-semibold")
                self.clean_md = ui.markdown("")
                self.clean_msg = ui.markdown("")

                async def clean():
                    from widgets import confirm
                    if await confirm("Move this run's leftovers to the Trash?", "Move to the Trash"):
                        self.clean_msg.content = T.cleanup(self.st, True)
                        self.reopen()
                ui.button("Move to the Trash", on_click=clean).props("unelevated dense no-caps color=negative")
            with ui.card().props("flat bordered").classes("ls-card w-full gap-2"):
                ui.label("This run").classes("font-semibold")
                self.this = ui.html("", sanitize=False)
                with ui.row().classes("items-center gap-2"):
                    ui.button("Train a new version…", on_click=self.new_version).props("outline dense no-caps")
                    if FT.have("sd_scripts") or FT.have("musubi"):
                        ui.button("Run it again", on_click=lambda: setattr(self.again_msg, "content", T.run_again(
                            self.st))).props("outline dense no-caps")
                    ui.button("★ Best version: mark / unmark", on_click=lambda: setattr(
                        self.again_msg, "content", T.toggle_best((self.st or {}).get("log", "")))).props("outline dense no-caps")
                    ui.label("Your rating").classes("ls-note")
                    self.rate = ui.toggle({0: "–", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5"}, value=0,
                                          on_change=lambda e: setattr(self.again_msg, "content", T.set_rating(
                                              self.st, e.value)) if self.st else None).props("dense unelevated")
                    ui.button("Report for Claude", on_click=self.report).props("outline dense no-caps")
                self.rep_note = ui.input("What do you see? (optional, goes into the report)",
                                         placeholder="for example: it stopped at epoch 3 / the pictures look burnt") \
                    .props("outlined dense").classes("w-full")
                self.rep_msg = ui.markdown("")
                note("A new version opens Train of this dataset. 4-5 good, 1-2 weak: Insights learns from your rating.")
                self.again_msg = ui.markdown("")
                with ui.expansion("TensorBoard").classes("w-full"):
                    with ui.row().classes("w-full items-end gap-2"):
                        self.tb_dir = ui.input("Log folder").props("outlined dense").classes("grow")
                        tb_port = ui.number("Port", value=6006, precision=0).props("outlined dense").classes("w-28")
                    tb_md = ui.markdown("TensorBoard: not running")
                    with ui.row().classes("gap-2"):
                        ui.button("Start TensorBoard", on_click=lambda: setattr(tb_md, "content", sg.tb_start(
                            self.tb_dir.value, tb_port.value))).props("outline dense no-caps")
                        ui.button("Stop TensorBoard", on_click=lambda: setattr(tb_md, "content", sg.tb_stop_ui())) \
                            .props("flat dense no-caps")
            with ui.expansion("End of the run log", value=True).classes("w-full ls-card"):
                self.tail = ui.textarea().props("readonly outlined rows=14 input-style='font-family:monospace;font-size:12px'") \
                    .classes("w-full")
        self.page.on_rs = self.click
        ui.timer(4.0, self.tick)
        self.draw_list()
        log = self.page.s.get("open_run")
        if log:
            self.open(log)
        else:
            self.show("list")


def step10(page):
    Results(page).build()
