# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Collection › LoRA page: one LoRA (all its versions) in four parts: Overview, Pictures (made with it), Training and
Edit. The data and the actions are in Gallery/lorapage.py; this file only draws them."""
from __future__ import annotations
__version__ = "1.131126c"  # LoRA Studio version, see version.py

from nicegui import run, ui

import lora_library as LL
import lorapage as LP
from widgets import note, pic_url

LIB = LL.LIB


class LoraPage:
    def __init__(self, shell, collection):
        self.shell, self.col = shell, collection
        self.key = ""
        self.gid = ""
        self.d: dict | None = None
        self.pic_page = 1

    # ------------------------------------------------------------------ open
    def show(self, key: str):
        key = LP.resolve(key)
        if not key:
            return
        self.key = key
        self.d = d = LP.details(key)
        self.gid = d["gid"]
        self.empty.set_visibility(False)
        self.body.set_visibility(True)
        self.title.text, self.subtitle.text = d["title"], d["subtitle"]
        self.types.options = {g: b for b, g in d["types"]}
        self.types.value = d["gid"]
        self.types.set_visibility(len(d["types"]) > 1)
        self.types.update()
        self.ver.options = {k: label for label, k in d["versions"]}
        self.ver.value = key
        self.ver.set_visibility(len(d["versions"]) > 1)
        self.ver.update()
        self.fill_overview()
        self.fill_training()
        self.fill_edit()
        self.fill_pictures()

    def fill_overview(self):
        d = self.d
        self.preview.set_source(pic_url(d["preview"], 900) if d["preview"] else "")
        self.preview.set_visibility(bool(d["preview"]))
        self.no_preview.set_visibility(not d["preview"])
        self.preview.classes(add="ls-blur" if d["blur"] else "", remove="" if d["blur"] else "ls-blur")
        self.examples.clear()
        with self.examples:
            for p in d["examples"]:
                ui.image(pic_url(p, 240)).classes("w-24 h-24 rounded cursor-pointer") \
                    .on("click", lambda _, p=p: self.big(p))
        self.examples_box.set_visibility(bool(d["examples"]))
        self.done.set_text(d["done"]["label"])
        self.done.set_value(d["done"]["value"])
        self.done.set_enabled(d["done"]["can"])
        self.done_msg.text = ""
        self.head.content = d["head"]
        self.accept_msg.text = ""
        dest, n = LP.copy_info(self.key)
        self.cp_dest.value, self.cp_note.text, self.cp_msg.text = dest, n, ""

    def fill_training(self):
        d = self.d
        self.settings.content = d["settings"]
        self.loss.options["series"][0]["data"] = [[s, v] for s, v in d["loss"]]
        self.loss.update()
        self.loss_box.set_visibility(bool(d["loss"]))
        self.versions.content = d["versions_html"]

    def fill_edit(self):
        e = self.d["edit"]
        for k, w in self.fields.items():
            w.value = e.get(k)
        self.e_msg.text = self.d["filled"]

    def fill_pictures(self, page: int = 1):
        folder, n = LP.picture_folder_note(self.gid)
        self.pf.value, self.pf_note.text = folder, n
        self.pver.options = {v: label for label, v in LL.picture_versions(self.gid)}
        self.pver.value = ""
        self.pver.update()
        ch, linked = LP.old_names(self.gid)
        self.old.options = {n: label for label, n in ch}
        self.old.value = linked
        self.old.update()
        self.draw_pictures(page)

    def draw_pictures(self, page: int):
        html_, self.pic_page, pages = LL.render_pictures(self.gid, self.pver.value or "", page)
        self.pgrid.content = html_
        self.ppages.text = f"page {self.pic_page} of {pages}" if pages > 1 else ""
        self.pprev.set_enabled(self.pic_page > 1)
        self.pnext.set_enabled(self.pic_page < pages)

    # ------------------------------------------------------------------ actions
    def big(self, path):
        with ui.dialog() as dlg, ui.card().classes("p-2 max-w-[95vw]"):
            ui.image(pic_url(path, 1600)).classes("max-h-[85vh] w-auto").props("fit=contain")
        dlg.open()

    def open_picture(self, path: str):
        img, info, prompt = LL.picture_details(path, self.gid)
        self.pv_img.set_source(pic_url(path, 1200) if img is not None else "")
        self.pv_img.set_visibility(img is not None)
        self.pv_img.classes(add="ls-blur" if self.d and self.d["blur"] else "")
        self.pv_info.content = info
        self.pv_prompt.value = prompt

    async def scan(self):
        self.pf_note.text = "Scanning … (only new and changed pictures are read)"
        msg, folder = await run.io_bound(LP.scan_pictures, self.gid, self.pf.value)
        self.fill_pictures()
        self.pf.value, self.pf_note.text = folder, msg

    async def scan_all(self):
        self.pf_note.text = "Scanning all picture folders …"
        msg = await run.io_bound(LP.scan_all_pictures)
        self.fill_pictures()
        self.pf_note.text = msg

    def toggle_done(self, value):
        if self.d and value != self.d["done"]["value"]:
            self.done_msg.text = LP.set_done(self.key, value)
            self.d["done"]["value"] = value
            self.col.redraw()

    def save(self):
        e = {k: w.value for k, w in self.fields.items()}
        self.e_msg.text = LP.save_edits(self.key, e)
        self.col.redraw()

    async def read_again(self):
        await run.io_bound(LP.read_training_again, self.key)
        self.show(self.key)

    def new_version(self):
        lora = LIB.loras.get(self.key)
        if lora and lora.dataset is not None:
            self.shell.go("datasets", ds=lora.dataset.name, part="train")

    # ------------------------------------------------------------------ build
    def build(self):
        with ui.row().classes("w-full items-center gap-3"):
            ui.button("← Browse", on_click=lambda: setattr(self.col.tabs, "value", "browse")).props("flat dense no-caps")
            with ui.column().classes("gap-0 grow"):
                self.title = ui.label("").classes("text-lg font-semibold")
                self.subtitle = ui.label("").classes("ls-note")
            self.types = ui.toggle({}, on_change=lambda e: e.value and e.value != self.gid and self.show("gid:" + e.value)) \
                .props("no-caps unelevated dense")
            self.ver = ui.select({}, label="Version", on_change=lambda e: e.value and e.value != self.key and self.show(e.value)) \
                .props("outlined dense").classes("w-72")
        self.empty = note("No LoRA open. Click a type label on a tile in Browse, or a card in Status.")
        with ui.column().classes("w-full gap-2") as self.body:
            self.tabs = ui.tabs().props("dense no-caps align=left").classes("ls-sub-tabs")
            with self.tabs:
                ui.tab("over", "Overview")
                ui.tab("pics", "Pictures")
                ui.tab("train", "Training")
                ui.tab("edit", "Edit")
            with ui.tab_panels(self.tabs, value="over").props("animated=false").classes("w-full bg-transparent"):
                with ui.tab_panel("over").classes("p-0 pt-2"):
                    self.build_overview()
                with ui.tab_panel("pics").classes("p-0 pt-2"):
                    self.build_pictures()
                with ui.tab_panel("train").classes("p-0 pt-2"):
                    self.build_training()
                with ui.tab_panel("edit").classes("p-0 pt-2"):
                    self.build_edit()
        self.body.set_visibility(False)

    def build_overview(self):
        with ui.row().classes("w-full no-wrap items-start gap-6"):
            with ui.column().classes("w-[420px] flex-none gap-2"):
                self.preview = ui.image("").classes("w-full rounded-lg cursor-pointer") \
                    .on("click", lambda: self.d and self.d["preview"] and self.big(self.d["preview"]))
                self.no_preview = note("No preview picture yet (LoRA Manager sets one).")
                with ui.column().classes("gap-1") as self.examples_box:
                    note("Example pictures from LoRA Manager")
                    self.examples = ui.row().classes("gap-1")
            with ui.column().classes("grow gap-3 min-w-0"):
                self.done = ui.checkbox("Completed", on_change=lambda e: self.toggle_done(e.value))
                self.done_msg = note("")
                self.head = ui.html("", sanitize=False)
                with ui.row().classes("items-center gap-3"):
                    ui.button("Accept the warnings",
                              on_click=lambda: (setattr(self.accept_msg, "text", LP.accept_warnings(self.key)),
                                                self.show(self.key), self.col.redraw())).props("outline dense no-caps")
                    ui.button("Pictures made with it", on_click=self.pictures_now).props("outline dense no-caps")
                self.accept_msg = note("")
                with ui.expansion("Copy this LoRA to your LoRA folder").classes("w-full ls-card"):
                    self.cp_dest = ui.input("Copy to folder").props("outlined dense").classes("w-full")
                    self.cp_note = note("")
                    self.cp_over = ui.checkbox("Replace a different file with the same name (the old one goes to the Trash)")
                    ui.button("Copy to LoRA folder",
                              on_click=lambda: (setattr(self.cp_msg, "text",
                                                        LP.copy_to(self.key, self.cp_dest.value, self.cp_over.value)),
                                                self.cp_over.set_value(False))).props("unelevated no-caps color=primary")
                    self.cp_msg = note("")

    async def pictures_now(self):
        self.tabs.value = "pics"
        if self.pf.value:
            await self.scan()

    def build_pictures(self):
        note("Every picture you made with this LoRA, all its versions together. Found through the .json ComfyUI saves "
             "next to each picture (or the PNG's own metadata). A scan only reads new and changed pictures.")
        with ui.row().classes("w-full items-end gap-3"):
            self.pver = ui.select({"": "All versions"}, value="", label="Version",
                                  on_change=lambda e: self.draw_pictures(1)).props("outlined dense").classes("w-64")
            self.pf = ui.input("Picture folder for this LoRA type",
                               placeholder="Where ComfyUI saves the pictures of this LoRA type") \
                .props("outlined dense").classes("grow")
            ui.button("Scan for new pictures", on_click=self.scan).props("unelevated no-caps color=primary")
            ui.button("Scan all picture folders", on_click=self.scan_all).props("outline no-caps")
            ui.button("Stop", on_click=lambda: (LL.PG.stop(), setattr(self.pf_note, "text", "Stopping … (what was read is kept)"))) \
                .props("flat no-caps color=negative")
        self.pf_note = note("")
        with ui.expansion("Old names of this LoRA (pictures made before a rename)").classes("w-full ls-card"):
            note("Names found in your pictures that no LoRA file has now, but that look like this LoRA. Tick the ones "
                 "that are this LoRA under an older name; their pictures join as old name.")
            self.old = ui.select({}, multiple=True, label="Old names").props("outlined dense use-chips").classes("w-full")
            with ui.row().classes("w-full items-end gap-3"):
                typed = ui.input("Or type a name", placeholder="e.g. mycharILLUSTRIOUS").props("outlined dense").classes("grow")
                old_msg = note("")

                def save_old():
                    old_msg.text = LP.save_old_names(self.gid, self.old.value, typed.value)
                    typed.value = ""
                    self.fill_pictures()
                ui.button("Save old names", on_click=save_old).props("unelevated no-caps color=primary")
        with ui.row().classes("w-full no-wrap items-start gap-4"):
            with ui.column().classes("grow min-w-0 gap-2"):
                with ui.row().classes("items-center gap-2"):
                    self.pprev = ui.button("◀ Newer", on_click=lambda: self.draw_pictures(self.pic_page - 1)) \
                        .props("flat dense no-caps")
                    self.ppages = ui.label("").classes("ls-note")
                    self.pnext = ui.button("Older ▶", on_click=lambda: self.draw_pictures(self.pic_page + 1)) \
                        .props("flat dense no-caps")
                self.pgrid = ui.html("", sanitize=False).classes("w-full")
            with ui.column().classes("w-[460px] flex-none gap-2"):
                self.pv_img = ui.image("").classes("w-full rounded-lg")
                self.pv_img.set_visibility(False)
                self.pv_info = ui.markdown("")
                self.pv_prompt = ui.textarea("Prompt").props("outlined readonly autogrow").classes("w-full")

    def build_training(self):
        self.settings = ui.html("", sanitize=False).classes("w-full")
        with ui.row().classes("items-center gap-3"):
            ui.button("Read the training again", on_click=self.read_again).props("outline dense no-caps")
            note("after you changed the launch script or the log")
        with ui.column().classes("w-full gap-1") as self.loss_box:
            ui.label("Loss").classes("font-semibold")
            self.loss = ui.echart({
                "grid": {"left": 50, "right": 20, "top": 10, "bottom": 30},
                "tooltip": {"trigger": "axis"},
                "xAxis": {"type": "value", "name": "step"}, "yAxis": {"type": "value", "scale": True},
                "series": [{"type": "line", "data": [], "showSymbol": False, "lineStyle": {"width": 1.5},
                            "color": "#E8A33D"}]}).classes("w-full h-64")
        ui.label("All versions").classes("font-semibold mt-2")
        self.versions = ui.html("", sanitize=False).classes("w-full")
        with ui.row().classes("items-center gap-3"):
            ui.button("Train a new version…", on_click=self.new_version).props("outline dense no-caps")
            note("opens Train on the dataset page")

    def build_edit(self):
        f = self.fields = {}
        with ui.grid(columns=2).classes("w-full gap-3"):
            f["name"] = ui.input("Display name").props("outlined dense")
            f["character"] = ui.input("Character (groups it in Browse)").props("outlined dense")
        with ui.row().classes("w-full items-end gap-3"):
            f["version"] = ui.input("Version (shown in ComfyUI's LoRA Manager)").props("outlined dense").classes("w-72")
            f["clip_skip"] = ui.number("Clip skip", min=1, precision=0).props("outlined dense").classes("w-32")
            with ui.column().classes("gap-0"):
                note("Your rating (4-5 good, 1-2 weak: Collection › Insights compares the datasets behind them)")
                f["rating"] = ui.toggle({0: "–", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5"}, value=0).props("dense unelevated")
        f["trigger"] = ui.textarea("Trigger words (one per line)").props("outlined autogrow").classes("w-full")
        f["description"] = ui.textarea("Description").props("outlined autogrow").classes("w-full")
        f["notes"] = ui.textarea("Notes").props("outlined autogrow").classes("w-full")
        with ui.row().classes("w-full items-end gap-3"):
            f["strength"] = ui.number("Strength", step=0.05).props("outlined dense").classes("w-32")
            f["clip"] = ui.number("CLIP strength", step=0.05).props("outlined dense").classes("w-36")
            f["nsfw"] = ui.select({0: "Unknown (0)", **{v: k for k, v in LL.NSFW_LEVELS.items()}}, value=0,
                                  label="Preview NSFW level").props("outlined dense").classes("w-48")
            f["favorite"] = ui.checkbox("Favorite")
        with ui.row().classes("w-full items-end gap-3"):
            f["flag"] = ui.select(["none", "warning", "problem"], value="none", label="Flag") \
                .props("outlined dense").classes("w-40")
            f["reason"] = ui.input("Flag reason").props("outlined dense").classes("grow")
        with ui.row().classes("items-center gap-3"):
            ui.button("Save", on_click=self.save).props("unelevated no-caps color=primary")
            note(f"Saved in the dataset's {LL.ARCHIVE_DIR}/ folder, a .bak is kept. Training settings (the Training "
                 "part) come from the LoRA file, its launch script and its log and cannot be edited.")
        self.e_msg = note("")


def build(shell, collection):
    ui.add_head_html(PIC_JS)
    p = LoraPage(shell, collection)
    p.build()
    ui.on("ll_pic", lambda e: p.open_picture(e.args))
    return p


PIC_JS = """
<script>
window.llPicOpen = function(path, tile){
  document.querySelectorAll('.ll-ptile.ll-psel').forEach(function(e){ e.classList.remove('ll-psel'); });
  if (tile) tile.classList.add('ll-psel');
  emitEvent('ll_pic', path);
};
</script>
"""
