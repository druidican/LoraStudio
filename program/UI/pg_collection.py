# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Collection: Browse (Pictures or Status), the LoRA page, Insights and Maintenance.

The tiles and cards are drawn by lora_library (render_gallery, render_library) as one piece of HTML: a filter change
replaces that one element, which is what makes this page quick. Clicks on them call small JavaScript helpers (COL_JS)
that send an event back here (emitEvent): a tile opens its dataset page, a type label or a Status card that LoRA's page.
Select mode is kept in the browser (picked cards get the class ll-picked) and read when a button needs it.
"""
from __future__ import annotations
__version__ = "1.131126b"  # LoRA Studio version, see version.py

from nicegui import run, ui

import lora_library as LL
from widgets import note

LIB = LL.LIB

COL_JS = """
<script>
window.lsSelecting = function(){ return document.body.classList.contains('ls-selecting'); };
window.lsSelCount = function(){
  var n = document.querySelectorAll('.ll-card.ll-picked[data-gid]').length;
  document.querySelectorAll('.ls-selcount').forEach(function(e){ e.textContent = n + ' selected'; });
};
window.lsPick = function(card){ card.classList.toggle('ll-picked'); lsSelCount(); };
window.llTile = function(card){
  if (lsSelecting()) { lsPick(card); return; }
  emitEvent('ll_tile', card.dataset.dsn || '');
};
window.llOpenCard = function(gid, card){
  if (card && lsSelecting() && card.dataset.gid) { lsPick(card); return; }
  emitEvent('ll_open', 'gid:' + gid);
};
window.loraPick = function(key, card){
  if (card && lsSelecting() && card.dataset.gid) { lsPick(card); return; }
  emitEvent('ll_open', key);
};
window.llToggle = function(gid, btn){
  if (lsSelecting() && btn.dataset.gid) { lsPick(btn); return; }
  var root = (btn && btn.closest('.ll-wrap')) || document;
  var target = root.querySelector('.ll-vers[data-gid="' + gid + '"]');
  var wasOpen = target && target.classList.contains('ll-show');
  root.querySelectorAll('.ll-vers.ll-show').forEach(function(e){ e.classList.remove('ll-show'); });
  root.querySelectorAll('.ll-card.ll-open').forEach(function(e){ e.classList.remove('ll-open'); });
  if (target && !wasOpen) { target.classList.add('ll-show'); if (btn) btn.classList.add('ll-open'); }
};
</script>
"""

STATUS_MAP = {"OK": "OK", "Warnings": "Warnings", "Problems": "Problems"}


class Collection:
    def __init__(self, shell):
        self.shell = shell
        self.f = {"search": "", "base": "All", "sort": "Newest first", "only_preview": False, "favorites": False,
                  "look": "Pictures", "status": ["Warnings", "Problems"], "group": "Character"}
        self.tabs = None
        self.view = None
        self.lora = None

    # ------------------------------------------------------------------ Browse
    def browse_html(self) -> str:
        f = self.f
        if f["look"] == "Pictures":
            return LL.render_gallery(LIB, f["search"], f["base"], f["sort"], f["only_preview"], f["favorites"])
        bases = LIB.base_names() + ["Unknown"] if f["base"] in ("", "All") else [f["base"]]
        return LL.render_library(LIB, f["search"], bases, f["status"], f["group"])

    def redraw(self):
        self.view.content = self.browse_html()
        self.status_box.set_visibility(self.f["look"] == "Status")
        ui.run_javascript("lsSelCount()")

    def set(self, **kw):
        self.f.update(kw)
        self.redraw()

    async def rescan(self):
        self.msg.text = "Reading your dataset folders …"
        await run.io_bound(LIB.scan)
        self.base_dd.options = ["All"] + LIB.base_names()
        self.base_dd.update()
        self.msg.text = f"{sum(1 for l in LIB.loras.values() if l.dataset is not None)} LoRA files read."
        self.redraw()

    async def mark(self, done: bool):
        gids = await ui.run_javascript(
            "[...document.querySelectorAll('.ll-card.ll-picked[data-gid]')].map(e => e.dataset.gid)")
        self.msg.text = LL.mark_completed(gids or [], done).replace("**", "")
        self.redraw()

    def select_mode(self, on: bool):
        ui.run_javascript(f"document.body.classList.toggle('ls-selecting', {str(bool(on)).lower()});"
                          + ("" if on else "document.querySelectorAll('.ll-picked').forEach(e => e.classList.remove('ll-picked'));")
                          + "lsSelCount();")
        self.sel_tools.set_visibility(on)

    def build_browse(self):
        with ui.row().classes("w-full no-wrap items-start gap-6"):
            with ui.column().classes("w-60 gap-3 flex-none"):
                ui.input("Search", placeholder="Name, character, trigger word",
                         on_change=lambda e: self.set(search=e.value or "")).props("outlined dense clearable") \
                    .classes("w-full")
                self.base_dd = ui.select(["All"] + LIB.base_names(), value="All", label="Base model",
                                         on_change=lambda e: self.set(base=e.value)).props("outlined dense") \
                    .classes("w-full")
                ui.radio(["Newest first", "Name"], value="Newest first",
                         on_change=lambda e: self.set(sort=e.value)).props("dense")
                ui.checkbox("Only with preview picture", on_change=lambda e: self.set(only_preview=e.value))
                ui.checkbox("Favorites only", on_change=lambda e: self.set(favorites=e.value))
                with ui.column().classes("gap-1") as self.status_box:
                    ui.label("Status").classes("ls-note")
                    for s in LL.STATUS_CHOICES:
                        ui.checkbox(s, value=s in self.f["status"],
                                    on_change=lambda e, s=s: self.set(status=[x for x in LL.STATUS_CHOICES
                                                                              if (x == s and e.value) or
                                                                              (x != s and x in self.f["status"])]))
                    ui.label("Group by").classes("ls-note mt-2")
                    ui.radio(["Character", "Base model"], value="Character",
                             on_change=lambda e: self.set(group=e.value)).props("dense")
                self.status_box.set_visibility(False)
                ui.button("Rescan folders", on_click=self.rescan).props("unelevated no-caps color=primary")
            with ui.column().classes("grow gap-2 min-w-0"):
                with ui.row().classes("items-center gap-4 w-full"):
                    ui.toggle(["Pictures", "Status"], value="Pictures",
                              on_change=lambda e: self.set(look=e.value)).props("no-caps unelevated")
                    ui.checkbox("Select", on_change=lambda e: self.select_mode(e.value))
                    with ui.row().classes("items-center gap-2") as self.sel_tools:
                        ui.button("Select all shown", on_click=lambda: ui.run_javascript(
                            "document.querySelectorAll('.ll-card[data-gid]').forEach(e => e.classList.add('ll-picked')); lsSelCount();")
                                  ).props("flat dense no-caps")
                        ui.button("Clear", on_click=lambda: ui.run_javascript(
                            "document.querySelectorAll('.ll-picked').forEach(e => e.classList.remove('ll-picked')); lsSelCount();")
                                  ).props("flat dense no-caps")
                        ui.label("0 selected").classes("ls-selcount ls-note")
                        ui.button("✓ Mark Completed", on_click=lambda: self.mark(True)).props("unelevated dense no-caps color=teal")
                        ui.button("Mark not completed", on_click=lambda: self.mark(False)).props("flat dense no-caps")
                    self.sel_tools.set_visibility(False)
                self.msg = note("")
                self.view = ui.html(self.browse_html(), sanitize=False).classes("w-full")
                note("Click a tile: its dataset page. Click a type label on a tile, or a card in Status: that LoRA's page. "
                     "A green label = Completed.")

    # ------------------------------------------------------------------ events
    def open_tile(self, ds_name: str):
        if "datasets" in self.shell.pages:
            self.shell.go("datasets", ds=ds_name, part="lora")
        else:                                            # until the dataset page is moved: its newest LoRA's page
            key = ds_name.lower()
            g = next((g for g, m in LIB.groups.items() if m and m[-1].dataset is not None
                      and m[-1].dataset.name.lower() == key), None)
            if g:
                self.open_lora("gid:" + g)

    def open_lora(self, key: str):
        self.tabs.value = "lora"
        if self.lora is not None:
            self.lora.show(key)

    def on_show(self, **kw):
        if kw.get("lora"):
            self.open_lora(kw["lora"])


def build(shell):
    ui.add_head_html(COL_JS + f"<style>{LL.CSS}</style>")
    c = Collection(shell)
    ui.on("ll_tile", lambda e: c.open_tile(e.args))
    ui.on("ll_open", lambda e: c.open_lora(e.args))
    with ui.column().classes("ls-page gap-2"):
        c.tabs = ui.tabs().props("dense no-caps align=left active-color=amber-8 indicator-color=amber-8") \
            .classes("ls-sub-tabs")
        with c.tabs:
            ui.tab("browse", "Browse")
            ui.tab("lora", "LoRA page")
            ui.tab("insights", "Insights")
            ui.tab("maint", "Maintenance")
        with ui.tab_panels(c.tabs, value="browse").props("animated=false").classes("w-full bg-transparent"):
            with ui.tab_panel("browse").classes("p-0 pt-2"):
                c.build_browse()
            with ui.tab_panel("lora").classes("p-0 pt-2"):
                try:
                    import lora_page
                    c.lora = lora_page.build(shell, c)
                except ModuleNotFoundError:
                    note("The LoRA page comes in the next step of the move.")
            with ui.tab_panel("insights").classes("p-0 pt-2"):
                note("Insights comes in a later step of the move.")
            with ui.tab_panel("maint").classes("p-0 pt-2"):
                note("Maintenance comes in a later step of the move.")
    return c
