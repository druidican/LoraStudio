# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Queue: the launch scripts, the queue and the run that is training now.

The lists are drawn by training_ui (scripts_html, queue_html) and sdgui (stage, progress, loss); their buttons send
ls_tr_box events (lsSend) that tr_action handles. Two timers keep the queue and the live run current; each only
changes what changed.
"""
from __future__ import annotations
__version__ = "1.201126e"  # LoRA Studio version, see version.py

import json

from nicegui import ui

import features as FT
import sdgui as sg
import sdloss
import sdqueue
import training_ui as T
from widgets import note


class Queue:
    def __init__(self, shell):
        self.shell = shell
        self.last = {}
        self.log_version = -1

    def set_html(self, el, key: str, html_: str):
        if self.last.get(key) != html_:
            self.last[key] = html_
            el.content = html_

    def draw_lists(self):
        self.set_html(self.scripts, "scripts", T.scripts_html(self.show_done.value))
        self.set_html(self.queue, "queue", T.queue_html())

    def action(self, v: str):
        s, q, msg = T.tr_action(v, self.show_done.value)
        self.set_html(self.scripts, "scripts", s)
        self.set_html(self.queue, "queue", q)
        self.msg.text = msg

    async def add_ticked(self):
        keys = await ui.run_javascript(
            "Array.from(document.querySelectorAll('#tr_scripts .tr-tick:checked')).map(e => e.dataset.k)")
        s, q, msg = T.add_many(json.dumps(keys or []), self.show_done.value)
        self.set_html(self.scripts, "scripts", s)
        self.set_html(self.queue, "queue", q)
        self.msg.text = msg

    # ------------------------------------------------------------------ the left menu
    def build_menu(self):
        import sidemenu as SM

        def jump(a):
            return lambda: ui.run_javascript(f"document.getElementById('{a}')?.scrollIntoView({{behavior:'smooth',block:'start'}})")
        with self.shell.menu.section("queue"):
            SM.header("Queue")
            self.m = {"running": SM.Item("play_arrow", "Running", on_click=jump("q_live")),
                      "queued": SM.Item("hourglass_empty", "Waiting", on_click=jump("tr_queue")),
                      "done": SM.Item("check_circle", "Done", on_click=jump("tr_queue")),
                      "failed": SM.Item("error", "Failed", on_click=jump("tr_queue"))}
            SM.header("Add")
            SM.Item("description", "Launch scripts", on_click=jump("tr_scripts"))
            SM.header("Controls")
            self.m_start = SM.Item("play_circle", "Start the queue",
                                   on_click=lambda: (T.start_queue(sdqueue.QUEUE.get("stop_on_fail", True)), self.draw_lists()))
            self.m_after = SM.Item("stop_circle", "Stop after this run", on_click=sdqueue.queue_stop_after_current)

    def draw_menu(self):
        items = sdqueue.QUEUE["items"]
        n = {k: sum(1 for i in items if i.get("state") == k) for k in ("queued", "done")}
        n["failed"] = sum(1 for i in items if i.get("state") in ("failed", "stopped"))
        n["running"] = 1 if sg.RUN.running() else 0
        for k, it in self.m.items():
            it.set_right(str(n[k]))
        self.m["running"].active(bool(n["running"]))
        self.m["failed"].set_icon("error", "warn" if n["failed"] else "")
        self.m["done"].set_icon("check_circle", "ok" if n["done"] else "")
        self.m_start.show(bool(n["queued"]) and not sdqueue.QUEUE.get("running") and not n["running"])
        self.m_after.show(bool(sdqueue.QUEUE.get("running")) and not sdqueue.QUEUE.get("stop_flag"))

    def tick(self):
        self.draw_menu()
        self.set_html(self.queue, "queue", T.queue_html())
        import comfytest as CT
        now = dict(CT.NOW)
        self.pics.set_visibility(bool(now))
        if now:
            self.pics_lbl.text = "🖼 " + CT.now_text()
            self.pics_bar.value = (now["done"] / now["total"]) if now.get("total") else None
            w = sum(1 for it in sdqueue.QUEUE["items"] if it.get("state") == "queued")
            self.pics_note.text = (f"The next run ({w} waiting) starts when the pictures are done and ComfyUI's memory is "
                                   "freed." if w else "The queue is done after these pictures.")
        run = sg.RUN
        self.live.set_visibility(bool(run.name))
        if not run.name:
            return
        self.set_html(self.head, "head", T.live_head())
        self.set_html(self.stage, "stage", sg.stage_html())
        self.set_html(self.prog, "prog", sg.progress_html())
        self.set_html(self.loss, "loss", sdloss.loss_html())
        self.set_html(self.sp, "sp", T.savepoints_html())
        if run.version != self.log_version:
            self.log_version = run.version
            self.log.value = sg.log_text()
            ui.run_javascript(f"var t = document.querySelector('#c{self.log.id} textarea'); "
                              "if (t) t.scrollTop = t.scrollHeight;")

    def on_show(self, **kw):
        self.draw_lists()

    def build(self):
        self.build_menu()
        trains = FT.have("sd_scripts") or FT.have("musubi")
        with ui.column().classes("ls-page gap-3"):
            if not trains:
                note("Neither sd-scripts nor musubi-tuner is installed here, so the queue and Start are hidden. The "
                     "launch scripts below are made on the dataset page's Train part as usual: copy the dataset folder "
                     "to the computer with the trainer and run them there, or install a trainer and tick it in "
                     "Settings › Start and network.")
            with ui.row().classes("w-full no-wrap items-start gap-6"):
                with ui.column().classes("grow basis-0 min-w-0 gap-2"):
                    ui.label("Launch scripts").classes("text-lg font-semibold")
                    with ui.row().classes("items-center gap-2"):
                        self.show_done = ui.checkbox("show finished", on_change=lambda: self.draw_lists())
                        ui.button("Refresh", on_click=self.draw_lists).props("flat dense no-caps")
                    if trains:
                        with ui.row().classes("items-center gap-2"):
                            ui.button("Tick the ones not run", on_click=lambda: ui.run_javascript(
                                "document.querySelectorAll('#tr_scripts .tr-tick:not([disabled])')"
                                ".forEach(i => { i.checked = i.dataset.notrun === '1'; })")).props("flat dense no-caps")
                            ui.button("Untick all", on_click=lambda: ui.run_javascript(
                                "document.querySelectorAll('#tr_scripts .tr-tick').forEach(i => { i.checked = false; })")
                                      ).props("flat dense no-caps")
                            ui.button("Add the ticked to the queue", on_click=self.add_ticked) \
                                .props("unelevated dense no-caps color=primary")
                    self.scripts = ui.html("", sanitize=False).props('id="tr_scripts"').classes("w-full")
                with ui.column().classes("grow basis-0 min-w-0 gap-2") as qcol:
                    ui.label("Queue").classes("text-lg font-semibold")
                    self.queue = ui.html("", sanitize=False).props('id="tr_queue"').classes("w-full")
                    with ui.row().classes("items-center gap-2"):
                        fail = ui.checkbox("Stop the queue when a run fails", value=sdqueue.QUEUE.get("stop_on_fail", True),
                                           on_change=lambda e: sdqueue.queue_failstop(e.value))
                        ui.button("Clear finished", on_click=lambda: (T.clear_queue(True), self.draw_lists())) \
                            .props("flat dense no-caps")
                        ui.button("Clear the queue", on_click=lambda: (T.clear_queue(False), self.draw_lists())) \
                            .props("flat dense no-caps")
                        ui.button("Start queue", on_click=lambda: (T.start_queue(fail.value), self.draw_lists())) \
                            .props("unelevated dense no-caps color=primary")
                    note("Starting the queue stops Ollama to free video memory. It is not started again by itself.")
                    self.msg = note("")
                qcol.set_visibility(trains)
            with ui.card().props("flat bordered").classes("ls-card w-full gap-1") as self.pics:
                self.pics_lbl = ui.label("").classes("font-semibold")
                self.pics_bar = ui.linear_progress(value=None, show_value=False).props("rounded color=amber-8")
                self.pics_note = ui.label("").classes("ls-note")
            self.pics.set_visibility(False)
            with ui.card().props('flat bordered id="q_live"').classes("ls-card w-full gap-2") as self.live:
                with ui.row().classes("w-full items-center gap-3"):
                    self.head = ui.html("", sanitize=False).classes("grow")
                    ui.button("Stop after this run", on_click=sdqueue.queue_stop_after_current).props("outline dense no-caps")
                    ui.button("Stop now", on_click=sg.stop_run).props("unelevated dense no-caps color=negative")
                self.stage = ui.html("", sanitize=False).classes("w-full").props('id="stage_box"')
                self.prog = ui.html("", sanitize=False).classes("w-full")
                with ui.row().classes("w-full no-wrap items-start gap-4"):
                    self.loss = ui.html("", sanitize=False).classes("grow basis-3/5 min-w-0").props('id="loss_box"')
                    self.sp = ui.html("", sanitize=False).classes("grow basis-2/5 min-w-0")
                with ui.expansion("Run log").classes("w-full"):
                    self.log = ui.textarea().props("readonly outlined input-style='font-family:monospace;font-size:12px' "
                                                   "rows=18").classes("w-full")
        ui.on("ls_tr_box", lambda e: self.action(str(e.args)))
        self.draw_lists()
        self.tick()
        ui.timer(1.0, self.tick)


def build(shell):
    q = Queue(shell)
    q.build()
    return q
