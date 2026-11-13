# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""The frame around every page: the header (logo, tabs, ?, dark/light), the status strip and the tab panels.

Each page is a module program/UI/pg_<key>.py with build(shell) -> an object (or None) that may have on_show(**kw): called when
the page is opened with shell.go(key, **kw), e.g. shell.go("datasets", ds="pip", part="results").
The status strip is redrawn by a timer only when its text changed.
"""
from __future__ import annotations
__version__ = "1.131126b"  # LoRA Studio version, see version.py

import importlib
from pathlib import Path

from nicegui import app, ui

import lspaths
import version

PAGES = [("home", "Home"), ("datasets", "Datasets"), ("queue", "Queue"), ("collection", "Collection"),
         ("settings", "Settings")]
CSS = (Path(__file__).with_name("theme.css")).read_text(encoding="utf-8")


class Shell:
    def __init__(self):
        self.tabs = None
        self.pages: dict[str, object] = {}
        self.dark = None
        self._going = False

    def go(self, key: str, **kw):
        """Open a page (and tell it what to show)."""
        self._going = True
        self.tabs.value = key
        self._going = False
        page = self.pages.get(key)
        if page is not None and hasattr(page, "on_show"):
            page.on_show(**kw)

    def changed(self, e):
        """A tab clicked: the page refreshes what it shows (pages opened with go() are told by go)."""
        if self._going:
            return
        page = self.pages.get(e.value)
        if page is not None and hasattr(page, "on_show"):
            page.on_show()


def _strip(shell: Shell):
    import statusbar
    box = ui.element("div").classes("ls-strip ls-st-idle")
    with box:
        run = ui.label("").classes("ls-run")
        machine = ui.html("", sanitize=False).classes("ls-machine")
    last = {"run": None, "machine": None, "state": None, "log": ""}

    def tick():
        line, state = statusbar.run_html()
        import html as _h
        line = _h.unescape(line)
        if line != last["run"]:
            run.text = last["run"] = line
        if state != last["state"]:
            box.classes(remove=f"ls-st-{last['state']}", add=f"ls-st-{state}")
            last["state"] = state
        ev = statusbar.DONE_EVENTS[-1] if statusbar.DONE_EVENTS and state in ("done", "fail") else None
        last["log"] = ev.get("log", "") if ev else ""
        m = statusbar.machine_html() + statusbar.ollama_html()
        if m != last["machine"]:
            machine.content = last["machine"] = m

    def click():
        shell.go("home" if last["log"] else "queue")
    run.on("click", click)
    run.tooltip("Open the queue, or Home › Just trained after a run")
    tick()
    ui.timer(1.5, tick)


def build_page():
    """The whole program for one browser tab."""
    import lora_library as LL
    import studio_css
    ui.add_head_html(f"<style>{LL.CSS}</style>")          # the cards, tiles and tables drawn as HTML
    ui.add_head_html(f"<style>{studio_css.CSS}</style>")  # the shared page parts (+ your userdata/custom.css)
    ui.add_css(CSS)
    # the older page parts (lora_library's cards) colour themselves with body.dark; NiceGUI sets body--dark
    ui.add_body_html("<script>(function(){var b=document.body;var f=function(){b.classList.toggle('dark',"
                     "b.classList.contains('body--dark'));};new MutationObserver(f).observe(b,{attributes:true,"
                     "attributeFilter:['class']});f();})();</script>")
    shell = Shell()
    shell.dark = ui.dark_mode(app.storage.user.get("dark", False))
    with ui.header(elevated=False).classes("ls-header column gap-0 p-0"):
        with ui.row().classes("w-full items-center gap-3 px-4 pt-2 no-wrap"):
            logo = lspaths.PROG / "logo" / "lorastudio_logo_80.png"
            if logo.is_file():
                ui.image(lspaths.file_url(logo)).classes("ls-logo")
            ui.label("LoRA Studio").classes("ls-title")
            ui.label(version.VERSION).classes("ls-ver")
            shell.tabs = ui.tabs(on_change=shell.changed).props("dense indicator-color=amber-6 active-color=amber-6 no-caps") \
                .classes("ml-4")
            with shell.tabs:
                for key, label in PAGES:
                    ui.tab(key, label)
            ui.space()

            def flip():
                shell.dark.toggle()
                app.storage.user["dark"] = shell.dark.value
            ui.button(icon="dark_mode", on_click=flip).props("flat round color=white").tooltip("Dark / light")
            ui.button("?", on_click=lambda: help_dialog(shell)).props("flat round color=white") \
                .tooltip("The manual for this page")
        _strip(shell)
    with ui.tab_panels(shell.tabs, value=app.storage.user.get("start", "home")) \
            .props("animated=false keep-alive").classes("w-full bg-transparent"):
        for key, label in PAGES:
            with ui.tab_panel(key).classes("p-0"):
                try:
                    mod = importlib.import_module("pg_" + key)
                except ModuleNotFoundError:
                    with ui.column().classes("ls-page"):
                        ui.label(f"{label}: not moved to the new program yet.").classes("ls-note")
                    continue
                shell.pages[key] = mod.build(shell)
    return shell


def help_dialog(shell: Shell):
    """The manual chapter of the current page (the full Manual page comes later)."""
    import re
    chap = {"home": "01_welcome", "datasets": "03_datasets", "queue": "04_training", "collection": "06_collection",
            "settings": "02_settings"}.get(shell.tabs.value, "01_welcome")
    f = next((lspaths.PROG / "manual").glob(chap + "*.md"), None)
    text = f.read_text(encoding="utf-8") if f else "No manual chapter found."
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)          # pictures come with the new manual
    with ui.dialog() as d, ui.card().classes("w-[900px] max-w-[95vw]"):
        ui.markdown(text).classes("w-full")
        ui.button("Close", on_click=d.close).props("flat no-caps")
    d.open()
