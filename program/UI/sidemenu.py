# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""The left menu: one drawer for the whole program whose contents follow the top tab (each page fills its own section:
with shell.menu.section("datasets"): …). It folds to a strip of icons (the button at the bottom; remembered per browser
login in app.storage.user["menu_mini"]); labels, headers and search fields carry Quasar's q-mini-drawer-hide."""
from __future__ import annotations
__version__ = "1.201126e"  # LoRA Studio version, see version.py

from nicegui import app, ui


class Item:
    """One menu line: icon, label, an optional note on the right (a count, a ✓)."""

    def __init__(self, icon: str, label: str, on_click=None, right: str = "", sub: bool = False, tip: str = ""):
        with ui.element("div").classes("ls-mi" + (" ls-mi-sub" if sub else "")) \
                .props(f'title="{(tip or label).replace(chr(34), "")}"') as self.el:
            self.icon = ui.icon(icon).classes("ls-mi-ic")
            self.label = ui.label(label).classes("ls-mi-lab q-mini-drawer-hide")
            self.right = ui.label(right).classes("ls-mi-r q-mini-drawer-hide")
        if on_click:
            self.el.on("click", lambda _e: on_click())
        self._on = False
        self._state = ""

    def active(self, on: bool):
        if on != self._on:
            self._on = on
            self.el.classes(**({"add": "ls-mi-on"} if on else {"remove": "ls-mi-on"}))

    def set_right(self, text: str):
        if self.right.text != text:
            self.right.text = text

    def set_icon(self, name: str, state: str = ""):
        """state: '' / 'ok' / 'warn' / 'off' (greyed: the part it needs is not installed)."""
        if self.icon.name != name:
            self.icon.name = name
        if state != self._state:
            if self._state:
                self.el.classes(remove=f"ls-mi-{self._state}")
            if state:
                self.el.classes(add=f"ls-mi-{state}")
            self._state = state

    def set_label(self, text: str):
        if self.label.text != text:
            self.label.text = text

    def show(self, on: bool):
        self.el.set_visibility(on)


def header(text: str):
    return ui.label(text).classes("ls-mh q-mini-drawer-hide")


class SideMenu:
    def __init__(self, shell):
        self.shell = shell
        self.sections: dict[str, ui.element] = {}
        mini = bool(app.storage.user.get("menu_mini", False))
        self.drawer = ui.left_drawer(value=True, fixed=True, bordered=False) \
            .props("width=248 mini-width=60 behavior=desktop show-if-above" + (" mini" if mini else "")) \
            .classes("ls-side p-0")
        self.mini = mini
        with self.drawer:
            self.box = ui.column().classes("w-full gap-0 ls-side-box")
            with ui.element("div").classes("ls-fold") as fold:
                self.fold_icon = ui.icon("keyboard_double_arrow_right" if mini else "keyboard_double_arrow_left")
                ui.label("Fold in").classes("q-mini-drawer-hide")
            fold.on("click", lambda _e: self.toggle())
            fold.tooltip("Fold the menu in / out")

    def section(self, key: str):
        """The container of one page's menu (made once; shown while that page is open)."""
        if key not in self.sections:
            with self.box:
                self.sections[key] = ui.column().classes("w-full gap-0")
            self.sections[key].set_visibility(False)
        return self.sections[key]

    def show(self, key: str):
        for k, sec in self.sections.items():
            sec.set_visibility(k == key)

    def toggle(self, mini: bool | None = None):
        self.mini = (not self.mini) if mini is None else mini
        self.drawer.props(add="mini") if self.mini else self.drawer.props(remove="mini")
        self.fold_icon.name = "keyboard_double_arrow_right" if self.mini else "keyboard_double_arrow_left"
        try:
            app.storage.user["menu_mini"] = self.mini
        except Exception:                                # noqa: BLE001
            pass

    def search(self, placeholder: str, on_change):
        """A search field (folded: a search icon that unfolds the menu)."""
        inp = ui.input(placeholder=placeholder, on_change=on_change) \
            .props("dense outlined clearable debounce=250").classes("ls-msearch q-mini-drawer-hide")
        with inp.add_slot("prepend"):
            ui.icon("search").classes("text-lg")
        b = ui.element("div").classes("ls-mi q-mini-drawer-only")
        with b:
            ui.icon("search").classes("ls-mi-ic")
        b.on("click", lambda _e: self.toggle(False))
        return inp
