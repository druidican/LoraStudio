# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Settings: a grouped list on the left (Health check first) and the open group's panel on the right. A panel is
built when it is opened (set_panels.py), so the page starts quickly; the health check and the counts in the list
come from settings_ui.checks()."""
from __future__ import annotations
__version__ = "1.131126b"  # LoRA Studio version, see version.py

from nicegui import ui

import settings_ui as SU


class Settings:
    def __init__(self, shell):
        self.shell = shell
        self.key = "health"
        self.btns = {}

    def open(self, key: str):
        import set_panels as P
        self.key = key if key in SU.KEYS else "health"
        counts = SU.nav_counts()
        for k, b in self.btns.items():
            n = counts.get(k, 0)
            b.text = SU.nav_label(k) + (f"  ·  {n}" if n else "")
            b.classes(**({"add": "ls-set-on"} if k == self.key else {"remove": "ls-set-on"}))
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

    def build(self):
        with ui.row().classes("ls-page w-full no-wrap items-start gap-6"):
            with ui.column().classes("w-60 flex-none gap-0"):
                for grp, items in SU.NAV:
                    if grp:
                        ui.label(grp).classes("ls-note tracking-widest mt-3 mb-1 text-xs")
                    for key, label in items:
                        self.btns[key] = ui.button(SU.nav_label(key), on_click=lambda k=key: self.open(k)) \
                            .props("flat no-caps align=left").classes("w-full ls-set-item")
            self.body = ui.column().classes("grow min-w-0 gap-3")
        self.open("health")


def build(shell):
    s = Settings(shell)
    s.build()
    return s
