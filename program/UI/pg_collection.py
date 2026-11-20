# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Collection: Browse (Pictures or Status), the LoRA page, Insights and Maintenance, chosen in the left menu (which
also holds the search and the filters of Browse, and the parts of an open LoRA page).

The tiles and cards are drawn by lora_library (render_gallery, render_library) as one piece of HTML: a filter change
replaces that one element, which is what makes this page quick. Clicks on them call small JavaScript helpers (COL_JS)
that send an event back here (emitEvent): a tile opens its dataset page, a type label or a Status card that LoRA's page.
Select mode is kept in the browser (picked cards get the class ll-picked) and read when a button needs it.
"""
from __future__ import annotations
__version__ = "1.201126e"  # LoRA Studio version, see version.py

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
        self.draw_menu()
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

    async def delete(self):
        import delete_dialog as DD
        gids = await ui.run_javascript(
            "[...document.querySelectorAll('.ll-card.ll-picked[data-gid]')].map(e => e.dataset.gid)")
        await DD.open_dialog(list(gids or []), after=self.deleted)

    def deleted(self, text: str):
        self.msg.text = text.split("\n")[0]
        ui.notify(text, multi_line=True, close_button=True, timeout=0)
        self.redraw()

    def select_mode(self, on: bool):
        ui.run_javascript(f"document.body.classList.toggle('ls-selecting', {str(bool(on)).lower()});"
                          + ("" if on else "document.querySelectorAll('.ll-picked').forEach(e => e.classList.remove('ll-picked'));")
                          + "lsSelCount();")
        self.sel_tools.set_visibility(on)

    def build_browse(self):
        with ui.row().classes("items-center gap-4 w-full"):
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
                ui.button("Delete…", on_click=self.delete).props("outline dense no-caps color=negative")
            self.sel_tools.set_visibility(False)
        self.msg = note("")
        self.view = ui.html(self.browse_html(), sanitize=False).classes("w-full")
        note("Click a tile: its dataset page. Click a type label on a tile, or a card in Status: that LoRA's page. "
             "A green label = Completed.")

    # ------------------------------------------------------------------ the left menu
    def build_menu(self):
        import sidemenu as SM
        with self.shell.menu.section("collection"):
            self.m_find = self.shell.menu.search("Name, character, trigger word",
                                                 lambda e: self.set(search=e.value or ""))
            SM.header("Collection")
            self.m = {"Pictures": SM.Item("photo_library", "Pictures", on_click=lambda: self.go_browse("Pictures")),
                      "Status": SM.Item("fact_check", "Status", on_click=lambda: self.go_browse("Status")),
                      "insights": SM.Item("insights", "Insights", on_click=lambda: self.go_tab("insights")),
                      "maint": SM.Item("build", "Maintenance", on_click=lambda: self.go_tab("maint"))}
            with ui.column().classes("w-full gap-0") as self.m_lora:
                SM.header("LoRA")
                self.m_card = ui.html("", sanitize=False).classes("ls-mcard q-mini-drawer-hide")
                self.m_lp = {k: SM.Item(i, t, sub=True, on_click=lambda k=k: self.go_lora_part(k))
                             for k, i, t in (("over", "dashboard", "Overview"), ("pics", "image", "Pictures"),
                                             ("train", "model_training", "Training"), ("edit", "edit", "Edit"))}
            self.m_lora.set_visibility(False)
            with ui.column().classes("w-full gap-2 ls-mfilters q-mini-drawer-hide") as self.m_filters:
                SM.header("Filter")
                self.base_dd = ui.select(["All"] + LIB.base_names(), value="All", label="Base model",
                                         on_change=lambda e: self.set(base=e.value)).props("outlined dense") \
                    .classes("w-full")
                ui.radio(["Newest first", "Name"], value="Newest first",
                         on_change=lambda e: self.set(sort=e.value)).props("dense")
                ui.checkbox("Only with preview picture", on_change=lambda e: self.set(only_preview=e.value)).props("dense")
                ui.checkbox("Favorites only", on_change=lambda e: self.set(favorites=e.value)).props("dense")
                with ui.column().classes("gap-1") as self.status_box:
                    ui.label("Status").classes("ls-note")
                    for s in LL.STATUS_CHOICES:
                        ui.checkbox(s, value=s in self.f["status"],
                                    on_change=lambda e, s=s: self.set(status=[x for x in LL.STATUS_CHOICES
                                                                              if (x == s and e.value) or
                                                                              (x != s and x in self.f["status"])])).props("dense")
                    ui.label("Group by").classes("ls-note mt-2")
                    ui.radio(["Character", "Base model"], value="Character",
                             on_change=lambda e: self.set(group=e.value)).props("dense")
                self.status_box.set_visibility(False)
                ui.button("Rescan folders", on_click=self.rescan).props("unelevated dense no-caps color=primary")

    def draw_menu(self):
        if not getattr(self, "m", None) or self.tabs is None:
            return
        tab = self.tabs.value
        for k, it in self.m.items():
            it.active(tab == k or (tab == "browse" and k == self.f["look"]))
        self.m_filters.set_visibility(tab == "browse")
        lp = self.lora
        has = bool(lp is not None and getattr(lp, "key", ""))
        self.m_lora.set_visibility(has)
        if has:
            import html as H
            self.m_card.content = (f'{H.escape(lp.title.text)}<small>{H.escape(lp.subtitle.text)}</small>')
            for k, it in self.m_lp.items():
                it.active(tab == "lora" and lp.tabs.value == k)

    def go_browse(self, look: str):
        self.tabs.value = "browse"
        if self.f["look"] != look:
            self.set(look=look)
        self.draw_menu()

    def go_tab(self, key: str):
        self.tabs.value = key
        self.draw_menu()

    def go_lora_part(self, part: str):
        self.tabs.value = "lora"
        self.lora.tabs.value = part
        self.draw_menu()

    async def insights(self, force: bool):
        if getattr(self, "cmp_draw", None):              # Compare settings: read the comparisons again
            await self.cmp_draw()
        if self.ins_html.content and not force:
            return
        import insights
        self.ins_html.content = '<p class="ls-note">Reading …</p>'
        data = await run.io_bound(insights.compute, LIB, force=force)
        self.ins_html.content = insights.page_html(data, LL.esc)

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
        self.draw_menu()

    def on_show(self, **kw):
        async def lm_check():                            # LoRA Manager changes (a preview you set there) come in
            try:
                n = await run.io_bound(LL.quick_sync, LIB)
            except Exception:                            # noqa: BLE001
                return
            if n:
                self.redraw()
                ui.notify(f"{n} LoRA card(s) updated from LoRA Manager.")
        ui.timer(0.2, lm_check, once=True)
        if kw.get("look"):
            self.go_browse(kw["look"])
        if kw.get("lora"):
            self.open_lora(kw["lora"])


def build(shell):
    ui.add_head_html(COL_JS)
    c = Collection(shell)
    ui.on("ll_tile", lambda e: c.open_tile(e.args))
    ui.on("ll_open", lambda e: c.open_lora(e.args))
    c.build_menu()

    def tab_changed(e):
        c.draw_menu()
        if e.value == "insights":
            return c.insights(False)
    with ui.column().classes("ls-page gap-2"):
        with ui.tab_panels(value="browse", on_change=tab_changed).props("animated=false") \
                .classes("w-full bg-transparent") as c.tabs:
            with ui.tab_panel("browse").classes("p-0 pt-2"):
                c.build_browse()
            with ui.tab_panel("lora").classes("p-0 pt-2"):
                try:
                    import lora_page
                    c.lora = lora_page.build(shell, c)
                except ModuleNotFoundError:
                    note("The LoRA page comes in the next step of the move.")
            with ui.tab_panel("insights").classes("p-0 pt-2"):
                build_insights(c)
            with ui.tab_panel("maint").classes("p-0 pt-2"):
                build_maintenance(c)
    c.draw_menu()
    return c


def build_insights(c: Collection):
    ui.label("Insights").classes("ls-h")
    note("What your own datasets, LoRAs and runs say, read on this computer: the datasets behind your good and weak "
         "LoRAs, and what your graphics card managed per LoRA type. Rate runs in Results (or on a LoRA page) to teach "
         "it. With few LoRAs these are hints, not rules: each says how many it comes from.")
    ui.button("Read again", on_click=lambda: c.insights(True)).props("outline dense no-caps")
    import cmp_ui                                        # Compare settings: what won, per LoRA type
    c.cmp_draw = cmp_ui.best_card()
    c.ins_html = ui.html("", sanitize=False).classes("w-full")


def build_maintenance(c: Collection):
    import maint
    with ui.row().classes("w-full no-wrap items-start gap-4"):
        with ui.card().props("flat bordered").classes("ls-card grow basis-0 gap-2"):
            ui.label("IN · SYNC").classes("ls-note tracking-widest")
            ui.label("Personal sync from your LoRA folder").classes("text-lg font-semibold")
            note(f"Copies the LoRA Manager files of your own LoRAs (.metadata.json, preview and example pictures) from "
                 f"the LoRA folder into each dataset's {LL.ARCHIVE_DIR}/ folder, where the Collection reads them. A LoRA "
                 "belongs to a dataset when it has the same name and size as a model file in the dataset, or the same "
                 "SHA256 hash (renamed copies). Downloads are left alone. The .metadata.json files are synced both ways: "
                 "what only one side has is copied to the other, and where both differ the newer file wins. A .bak is "
                 "made before a file is changed.")
            with ui.expansion("LoRA Manager's example pictures folder").classes("w-full"):
                roots = LL.lm_example_roots(LIB)
                lm = ui.input("Example pictures folder (empty = read from LoRA Manager's settings)",
                              value=LIB.data.get("lm_examples_dir") or "",
                              placeholder=roots[0] if roots else "the 'Example images' folder set in LoRA Manager") \
                    .props("outlined dense").classes("w-full")
                lm_msg = note("Found now: " + (roots[0] if roots else "nothing (set the folder here)") + ".")
                ui.button("Save folder", on_click=lambda: setattr(lm_msg, "text", maint.set_example_folder(lm.value))) \
                    .props("outline dense no-caps")
            sync_msg = ui.markdown("")

            async def do_sync():
                sync_msg.content = "Syncing …"
                sync_msg.content = await run.io_bound(maint.sync)
                c.redraw()
            ui.button("Sync now", on_click=do_sync).props("unelevated no-caps color=primary")
            ui.label("TRAINING CARDS").classes("ls-note tracking-widest mt-4")
            ui.label("Scan all LoRAs").classes("text-lg font-semibold")
            note(f"Reads every LoRA of your datasets once: its launch script, the LoRA file's training data, the repeat "
                 f"folders and the run log, into the LoRA page's Training part (saved in the dataset's {LL.ARCHIVE_DIR}/ "
                 "copy of the LoRA Manager file). A scanned version is read again only when its file changes.")
            with ui.row().classes("items-center gap-3"):
                force = ui.checkbox("Read the scanned ones again too")
                tc_msg = ui.markdown("")

                async def do_cards():
                    tc_msg.content = "Reading …"
                    tc_msg.content = await run.io_bound(maint.training_cards, force.value)
                ui.button("Scan all LoRAs", on_click=do_cards).props("unelevated no-caps color=primary")
        with ui.card().props("flat bordered").classes("ls-card grow basis-0 gap-2"):
            ui.label("OUT · CLEAN UP").classes("ls-note tracking-widest")
            ui.label("Free disk space").classes("text-lg font-semibold")
            note("Everything goes to the desktop Trash; restore it from the file manager. One run at a time: Results "
                 "on the dataset page.")
            with ui.expansion("Leftovers of all finished LoRAs", value=True).classes("w-full"):
                note("Epoch files, saved states (-state folders), .tmp.safetensors, .npz latent caches and musubi-tuner's "
                     "cache folders. Only for LoRAs that have a final file; a dataset with a training run in progress is "
                     "skipped.")
                lo = {"paths": []}
                lo_md = ui.markdown("")

                async def find_lo():
                    lo_md.content = "Looking …"
                    lo_md.content, lo["paths"] = await run.io_bound(maint.find_leftovers)

                async def trash_lo():
                    if not lo["paths"]:
                        lo_md.content = "Press Find leftovers first."
                        return
                    from widgets import confirm
                    if await confirm(f"Move {len(lo['paths'])} item(s) to the Trash?", "Move to Trash"):
                        lo_md.content = await run.io_bound(maint.trash_leftovers, lo["paths"])
                        lo["paths"] = []
                with ui.row().classes("gap-2"):
                    ui.button("Find leftovers", on_click=find_lo).props("outline dense no-caps")
                    ui.button("Move to Trash", on_click=trash_lo).props("unelevated dense no-caps color=negative")
            with ui.expansion("Orphaned LoRA Manager files").classes("w-full"):
                note("A .metadata.json / .civitai.info with its preview and example images, whose .safetensors file no "
                     "longer exists (deleted or renamed LoRAs).")
                or_md = ui.markdown("")
                or_pick = ui.select({}, multiple=True, label="Orphaned meta files").props("outlined dense use-chips") \
                    .classes("w-full")

                async def find_or():
                    md, choices = await run.io_bound(maint.find_orphans)
                    or_md.content = md
                    or_pick.options = {v: label for label, v in choices}
                    or_pick.value = [v for _, v in choices]
                    or_pick.update()

                async def trash_or():
                    from widgets import confirm
                    if or_pick.value and await confirm(f"Move {len(or_pick.value)} ticked group(s) to the Trash?",
                                                       "Move to Trash"):
                        or_md.content = await run.io_bound(maint.trash_orphans, or_pick.value)
                        await find_or()
                with ui.row().classes("gap-2"):
                    ui.button("Scan", on_click=find_or).props("outline dense no-caps")
                    ui.button("Move ticked to Trash", on_click=trash_or).props("unelevated dense no-caps color=negative")
