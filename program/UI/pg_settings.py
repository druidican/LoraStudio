# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Settings: the groups in the left menu (Health check first, with a search over their names) and the open group's
panel. A panel is built when it is opened (set_panels.py), so the page starts quickly; the health check and the counts
in the menu come from settings_ui.checks()."""
from __future__ import annotations
__version__ = "1.201126e"  # LoRA Studio version, see version.py

from nicegui import ui

import settings_ui as SU

ICONS = {"health": "health_and_safety", "start": "lan", "folders": "folder", "find": "travel_explore",
         "ai": "smart_toy", "trainers": "memory", "models": "inventory_2", "speed": "speed", "comfy": "image",
         "look": "palette", "backup": "backup", "updates": "system_update", "version": "info"}


class Settings:
    def __init__(self, shell):
        self.shell = shell
        self.key = "health"
        self.btns = {}

    def open(self, key: str):
        import set_panels as P
        self.key = key if key in SU.KEYS else "health"
        self.draw_menu()
        self.body.clear()
        with self.body:
            ui.label(SU.nav_label(self.key)).classes("ls-h")
            build = getattr(P, "panel_" + self.key, None)
            if build is None:
                ui.label("This group comes in the next part of the move.").classes("ls-note")
            else:
                try:
                    build(self)
                except Exception as e:                   # noqa: BLE001
                    ui.label(f"This panel could not be built: {e!r}").classes("ls-note")

    def on_show(self, **kw):
        self.open(kw.get("group") or self.key)

    # ------------------------------------------------------------------ the left menu
    def build_menu(self):
        import sidemenu as SM
        with self.shell.menu.section("settings"):
            self.m_find = self.shell.menu.search("Find a setting group", lambda e: self.filter_menu(e.value or ""))
            for grp, items in SU.NAV:
                h = SM.header(grp.capitalize()) if grp else None
                for key, label in items:
                    self.btns[key] = SM.Item(ICONS.get(key, "settings"), label, on_click=lambda k=key: self.open(k))
                    self.heads[key] = h

    def filter_menu(self, text: str):
        t = text.strip().lower()
        shown = set()
        for k, it in self.btns.items():
            on = not t or t in SU.nav_label(k).lower() or t in k
            it.show(on)
            if on:
                shown.add(id(self.heads[k]))
        for h in {id(h): h for h in self.heads.values() if h is not None}.values():
            h.set_visibility(id(h) in shown)

    def draw_menu(self):
        counts = SU.nav_counts()
        for k, it in self.btns.items():
            n = counts.get(k, 0)
            lab = SU.nav_label(k)
            it.set_label(lab)
            it.set_right(str(n) if n else "")
            it.set_icon(ICONS.get(k, "settings"), "off" if "missing" in lab else ("warn" if n else ""))
            it.active(k == self.key)

    def build(self):
        self.heads = {}
        self.build_menu()
        with ui.column().classes("ls-page w-full gap-3") as self.body:
            pass
        ui.on("ls_ai_box", lambda e: getattr(self, "on_ai", lambda v: None)(str(e.args)))
        self.open("health")


def build(shell):
    s = Settings(shell)
    s.build()
    return s
