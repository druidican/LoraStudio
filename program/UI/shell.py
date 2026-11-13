# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""The frame around every page: the header (logo, tabs, ?, dark/light), the status strip and the tab panels.

Each page is a module program/UI/pg_<key>.py with build(shell) -> an object (or None) that may have on_show(**kw): called when
the page is opened with shell.go(key, **kw), e.g. shell.go("datasets", ds="pip", part="results").
The status strip is redrawn by a timer only when its text changed.
"""
from __future__ import annotations
__version__ = "1.131126c"  # LoRA Studio version, see version.py

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
    import setup_page
    import studio_css
    ui.add_head_html(f"<style>{LL.CSS}</style>")          # the cards, tiles and tables drawn as HTML
    import sdgui
    ui.add_head_html(f"<style>{sdgui.CSS}</style>")       # the queue, run and results parts
    ui.add_head_html(f"<style>{studio_css.CSS}</style>")  # the shared page parts (+ your userdata/custom.css)
    ui.add_css(CSS)
    # the older page parts (lora_library's cards) colour themselves with body.dark; NiceGUI sets body--dark
    ui.add_body_html("<script>(function(){var b=document.body;var f=function(){b.classList.toggle('dark',"
                     "b.classList.contains('body--dark'));};new MutationObserver(f).observe(b,{attributes:true,"
                     "attributeFilter:['class']});f();})();</script>")
    ui.add_head_html(BRIDGE_JS)
    shell = Shell()
    setup_page.notice_gate()                              # shown once, before anything else
    ui.on("ls_nav", lambda e: shell.go("settings", group=e.args))
    ui.on("ls_manual", lambda e: getattr(shell, "manual_show", lambda k: None)(str(e.args)))
    ui.on("ls_lora", lambda e: shell.go("collection", lora=str(e.args)))
    import notify
    notify.SHOW = lambda msg, kind: ui.notify(msg, type="warning" if kind == "warning" else "info")
    ui.on("ls_go", lambda e: shell.go("datasets", run=e.args[4:]) if str(e.args).startswith("run|")
          else shell.go(str(e.args)))
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
    with ui.tab_panels(shell.tabs, value=_start_page()) \
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


BRIDGE_JS = """
<script>
// Links inside the HTML parts the program draws (cards, tables, lists) send events to the page's Python side:
// lsSend('<box>', value) -> event ls_<box>; lsSetNav('<settings group>'); lsGoQueue(); lsGoRun('<log>').
window.lsManual = function(k){ emitEvent('ls_manual', k); };
window.lsSend = function(box, v){ emitEvent('ls_' + box, v); };
window.lsSetNav = function(k){ emitEvent('ls_nav', k); };
window.lsGoQueue = function(){ emitEvent('ls_go', 'queue'); };
window.lsGoRun = function(log){ emitEvent('ls_go', 'run|' + log); };
window.lsOpenLora = function(gid){ emitEvent('ls_lora', 'gid:' + gid); };
</script>
"""


def _start_page() -> str:
    import launchopts
    k = launchopts.load().get("start_page") or "home"
    return k if k in dict(PAGES) else "home"


def help_dialog(shell: Shell):
    """The manual, opened at the chapter of the current page; chapters on the left, pictures from manual/img/."""
    import manual
    key = {"home": "welcome", "datasets": "datasets", "queue": "training", "collection": "collection",
           "settings": "settings"}.get(shell.tabs.value, "welcome")
    with ui.dialog().props("maximized") as d, ui.card().classes("w-full ls-page"):
        with ui.row().classes("w-full items-center"):
            ui.label("Manual").classes("text-xl font-bold grow")
            ui.button("Close", on_click=d.close).props("flat no-caps")
        with ui.row().classes("w-full no-wrap items-start gap-6"):
            nav = ui.html("", sanitize=False).classes("w-64 flex-none")
            body = ui.scroll_area().classes("grow h-[calc(100vh-110px)]")
            with body:
                text = ui.html("", sanitize=False).classes("w-full")

    def show(k):
        b, n = manual.page(k)
        text.content, nav.content = b, n
        body.scroll_to(percent=0)
    shell.manual_show = show                             # chapter links: the ls_manual event (build_page)
    show(key)
    d.open()

