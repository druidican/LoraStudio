# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Home: what is training, what was just trained (with its test pictures), where you left off, what needs a look
and the last results. The data comes from Other/home.py and Training/comfytest.py; nothing is scanned here.

Just trained: the runs of the last finished queue. One LoRA: its test pictures with the save point to try first and
the final file; several LoRAs: the final file of each, side by side (comfytest.after_queue decides which are made).
"""
from __future__ import annotations
__version__ = "1.131126c"  # LoRA Studio version, see version.py

import time

from nicegui import ui

import home as H
from widgets import note, pic_url


class Home:
    def __init__(self, shell):
        self.shell = shell
        self.batch_sig = None

    # ------------------------------------------------------------------ Training now
    def training(self):
        import sdgui as S
        waiting = sum(1 for i in S.QUEUE["items"] if i["state"] == "queued")
        if S.RUN.running():
            return f"● {S.RUN.name or 'training'}", S.progress_text(), f"{waiting} more waiting in the queue"
        return "Nothing is training", (f"{waiting} waiting in the queue." if waiting else "The queue is empty."), ""

    def tick(self):
        a, b, c = self.training()
        if (a, b, c) != (self.t_big.text, self.t_prog.text, self.t_more.text):
            self.t_big.text, self.t_prog.text, self.t_more.text = a, b, c
        sig = self.batch_signature()
        if sig != self.batch_sig:
            self.draw_batch()

    # ------------------------------------------------------------------ Just trained
    def batch_signature(self):
        import comfytest as CT
        b = CT.last_batch()
        logs = b.get("logs") or []
        busy = tuple((ln, (CT.JOBS.get(ln) or {}).get("state"), (CT.JOBS.get(ln) or {}).get("done"), ln in CT.PENDING)
                     for ln in logs)
        return (b.get("at"), busy)

    def draw_batch(self):
        import comfytest as CT
        self.batch_sig = self.batch_signature()
        b = CT.last_batch()
        self.batch.clear()
        if not b:
            self.batch_card.set_visibility(False)
            return
        self.batch_card.set_visibility(True)
        runs = CT.batch_view(b["logs"])
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(b["at"]))
        with self.batch:
            note(f"Finished {when}: {len(runs)} LoRA{'s' if len(runs) != 1 else ''}. "
                 + ("Test pictures: the save point to try first and the final file." if len(runs) == 1 else
                    "Test pictures: the final file of each, side by side."))
            for r in runs:
                with ui.column().classes("w-full gap-1"):
                    with ui.row().classes("items-center gap-3 w-full"):
                        ui.label(f"{r['dataset']} · {r['type']} {r['version']}").classes("font-semibold")
                        if r["state"]:
                            note({"making": "⏳ making the test pictures …",
                                  "waiting": "⏳ waiting for ComfyUI"}.get(r["state"], r["state"]))
                        ui.space()
                        ui.button("Its results →", on_click=lambda _, ln=r["log"]: self.shell.go("datasets", run=ln)) \
                            .props("flat dense no-caps")
                        if not r["labels"] and not r["state"]:
                            ui.button("Make test pictures",
                                      on_click=lambda _, ln=r["log"], n=len(runs): self.make(ln, n)) \
                                .props("outline dense no-caps")
                    for lab in r["labels"]:
                        with ui.row().classes("items-center gap-2 no-wrap w-full overflow-x-auto"):
                            ui.label(lab).classes("ls-note w-20 flex-none")
                            for path in r["pics"].get(lab) or []:
                                if path:
                                    ui.image(pic_url(path, 300)).classes("w-28 h-28 rounded cursor-pointer flex-none") \
                                        .props("fit=cover").on("click", lambda _, p=path: self.big(p))
                                else:
                                    ui.element("div").classes("w-28 h-28 rounded flex-none bg-gray-300/30")

    def make(self, log: str, n_runs: int):
        import comfytest as CT
        CT.WHICH[log] = "best+final" if n_runs == 1 else "final"
        ui.notify(CT.make_in_background(log))
        self.draw_batch()

    def big(self, path):
        with ui.dialog() as d, ui.card().classes("p-2 max-w-[95vw]"):
            ui.image(pic_url(path, 1600)).classes("max-h-[85vh] w-[80vw]").props("fit=contain")
        d.open()

    # ------------------------------------------------------------------ the other cards
    def draw_static(self):
        self.cont.clear()
        with self.cont:
            items = H.continue_items()
            for it in items:
                ui.link(f"{it['ds']} · {it['label']}").classes("ls-row") \
                    .on("click", lambda _, it=it: self.shell.go("datasets", ds=it["ds"], step=it["step"]))
            if not items:
                note("The datasets you open appear here.")
        self.look.clear()
        with self.look:
            items = H.look_items()
            for it in items[:6]:
                with ui.row().classes("ls-row items-center gap-2 no-wrap cursor-pointer") \
                        .on("click", lambda _, it=it: self.shell.go(it["go"][0], **it["go"][1])):
                    ui.element("span").classes(f"ls-dot {'problem' if it['level'] == 'bad' else 'warning'}")
                    ui.label(it["title"]).classes("font-semibold")
                    ui.label(it["detail"]).classes("ls-note truncate")
            if len(items) > 6:
                note(f"{len(items) - 6} more: Settings › Health check, Datasets › Improvements, Collection › Status.")
            if not items:
                note("Nothing needs a look.")
        self.results.clear()
        with self.results:
            rows = H.last_runs()
            for r in rows:
                label = f"{r['info']['dataset']} · {r['info']['type']}" if r.get("info") else r["stem"]
                with ui.row().classes("ls-row items-center gap-2 no-wrap cursor-pointer") \
                        .on("click", lambda _, ln=r["log"]: self.shell.go("datasets", run=ln)):
                    ui.element("span").classes(f"ls-dot {'ok' if r['result'] == 'done' else 'problem'}")
                    ui.label(label).classes("font-semibold")
                    ui.label(f"{r['result']} · {r['started'][5:]}").classes("ls-note")
            if not rows:
                note("No finished runs yet.")

    def on_show(self, **kw):
        self.draw_static()
        self.draw_batch()

    def build(self):
        with ui.column().classes("ls-page gap-4"):
            with ui.row().classes("items-center gap-3"):
                ui.label("Home").classes("ls-h")
                ui.button("Refresh", on_click=self.on_show).props("flat dense no-caps")
            with ui.card().props("flat bordered").classes("ls-card w-full gap-2") as self.batch_card:
                ui.label("Just trained").classes("text-lg font-semibold")
                self.batch = ui.column().classes("w-full gap-3")
            with ui.element("div").classes("w-full grid gap-4 grid-cols-1 lg:grid-cols-2"):
                with ui.card().props("flat bordered").classes("ls-card gap-1"):
                    ui.label("Training now").classes("text-lg font-semibold")
                    self.t_big = ui.label("").classes("text-base font-semibold")
                    self.t_prog = note("")
                    self.t_more = note("")
                    ui.button("Open the queue", on_click=lambda: self.shell.go("queue")).props("outline dense no-caps")
                with ui.card().props("flat bordered").classes("ls-card gap-1"):
                    ui.label("Continue where you left off").classes("text-lg font-semibold")
                    self.cont = ui.column().classes("gap-0 w-full")
                    ui.button("+ New dataset", on_click=lambda: self.shell.go("datasets", new=True)) \
                        .props("outline dense no-caps")
                with ui.card().props("flat bordered").classes("ls-card gap-1"):
                    ui.label("Needs a look").classes("text-lg font-semibold")
                    self.look = ui.column().classes("gap-0 w-full")
                with ui.card().props("flat bordered").classes("ls-card gap-1"):
                    ui.label("Last results").classes("text-lg font-semibold")
                    self.results = ui.column().classes("gap-0 w-full")
        self.on_show()
        self.tick()
        ui.timer(2.5, self.tick)


def build(shell):
    h = Home(shell)
    h.build()
    return h
