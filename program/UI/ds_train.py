# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""The dataset page's Train part (step 9): the launch script of a LoRA type, its plan and settings, the script text,
saving and the queue. The script logic is sd_script.py (cs_* return what changed as a dict); the plan and the
settings table are datasets_ui (calc, apply_settings, SETTINGS)."""
from __future__ import annotations
__version__ = "1.131126c"  # LoRA Studio version, see version.py

from pathlib import Path

from nicegui import run, ui

import datasets_ui as V
import dsflow as F
import features as FT
import sd_settings as SSET
import sdgui as S
from widgets import note

PRIORITIES = {"balanced": "Balanced", "quality": "Best quality", "fast": "Fastest"}


class Train:
    def __init__(self, page):
        self.page = page
        self.ds = page.s["ds"]
        m = F.meta(self.ds)
        self.types = m["types"]
        t = page.s.get("tkey")
        self.tkey = t if t in self.types else (self.types[0] if self.types else "")
        self.script, self.text, self.plan = S.NEW, "", {}
        self.newname, self.prompts_path, self.prompts_text = "", "", ""
        self.purpose = m["purpose"]
        S.set_choice(self.purpose, S.CHOICE.get("priority"))
        self.busy = False                                # while the code fills the fields (no write-back)

    # ------------------------------------------------------------------ the editor follows a result dict
    def apply(self, r: dict):
        self.busy = True
        try:
            if "script_choices" in r:
                self.w_script.options = {v: lab for lab, v in r["script_choices"]}
            if "script" in r:
                self.script = r["script"]
                self.w_script.value = self.script
            self.w_script.update()
            if "type" in r and r["type"] in self.types:
                self.tkey = r["type"]
            if "text" in r:
                self.text = r["text"] or ""
                self.w_text.value = self.text
            if "fields" in r:
                for w, v in zip(self.fields, r["fields"]):
                    w.value = v or ""
                self.model_vis()
            if "explain" in r:
                self.w_explain.content = r["explain"] or ""
            for k in ("prompts_path", "prompts_text"):
                if k in r:
                    setattr(self, k, r[k] or "")
            if "newname" in r:
                self.w_newname.value = r["newname"] or ""
            if "can_save_over" in r:
                self.w_save.set_visibility(bool(r["can_save_over"]))
            if "status" in r:
                self.w_status.content = r["status"] or ""
            if "plan" in r:
                self.plan = r["plan"] or {}
            self.draw_plan()
        finally:
            self.busy = False

    def args(self):
        return (self.w_steps.value or 0, self.w_userep.value, self.w_hf.value, False)

    def load(self):
        self.apply(S.cs_ds_change(self.ds, self.tkey, *self.args()))
        self.draw_repeat()

    def draw_plan(self):
        a, b, vals = V.calc(self.ds, self.tkey, self.text)
        self.w_calc.content, self.w_eff.content = a, b
        self.busy = True
        try:
            for (w, info), (cur, inf) in zip(self.sets, vals):
                w.value = cur
                info.text = inf
        finally:
            self.busy = False

    def model_vis(self):
        try:
            used = {k for k, _ in S.type_slots(self.tkey)} if self.tkey else set()
        except Exception:                                # noqa: BLE001
            used = set()
        names = [n for n, _ in S.FORM_GROUPS[3][1]]
        for n, w in zip(names, self.fields[-len(names):]):
            w.set_visibility((not used) or n in used or bool((w.value or "").strip()))

    def draw_repeat(self):
        r = S.rp_refresh(self.ds, self.tkey, self.w_steps.value or 0)
        self.rp_folder.options = {v: lab for lab, v in r["choices"]}
        self.rp_folder.value = r["folder"]
        self.rp_folder.update()
        self.rp_rep.value, self.rp_name.value, self.rp_info.content = r["repeats"], r["newname"], r["info"]

    # ------------------------------------------------------------------ events
    def set_type(self, t):
        if t == self.tkey:
            return
        self.tkey = t
        self.page.s["tkey"] = t
        self.apply(S.cs_retype(self.ds, self.script, t, self.text, *self.args()))
        self.draw_caps()
        self.draw_download()
        self.draw_repeat()

    def set_script(self, v):
        if self.busy or not v or v == self.script:
            return
        self.apply(S.cs_script_change(self.ds, v, self.tkey, *self.args()))

    def set_choice(self):
        S.set_choice(self.purpose, self.w_prio.value)
        self.apply(S.cs_retype(self.ds, self.script, self.tkey, self.text, *self.args()))

    def text_changed(self, v):
        if self.busy:
            return
        self.text = v or ""
        r = S.cs_read(self.ds, self.tkey, self.script, self.text, self.plan)
        r.pop("text", None)
        self.apply(r)

    def field_left(self):
        if self.busy:
            return
        r = S.cs_apply_quiet(self.ds, self.tkey, self.script, self.text, self.plan, *[w.value for w in self.fields])
        if r:
            self.apply(r)

    def setting_left(self):
        if self.busy:
            return
        new, msg = V.apply_settings(self.ds, self.tkey, self.text, *[w.value for w, _ in self.sets])
        self.w_amsg.content = msg
        if new != self.text:
            self.apply({"text": new, "fields": S.read_form(new)})

    def draw_caps(self):
        sets = F.train_sets(self.ds, self.tkey) if self.tkey else []
        self.w_caps_row.set_visibility(len(sets) > 1)
        if len(sets) > 1:
            self.w_caps.options = {x["style"]: f"{'Tags' if x['style'] == 'tags' else 'Sentences'} (IMG{x['ext']})"
                                   for x in sets}
            self.w_caps.value = F.train_style(self.ds, self.tkey)
            self.w_caps.update()

    def pick_caps(self, style):
        if self.busy or not style:
            return
        F.set_train_style(self.ds, self.tkey, style)
        import sd_script as SS
        if self.text.strip():
            self.apply({"text": SS.set_caption_ext(self.text, F.caption_ext(self.ds, self.tkey))})

    def draw_download(self):
        try:
            import hfdownload as HD
            miss = HD.missing(self.tkey) if self.tkey else []
        except Exception:                                # noqa: BLE001
            miss = []
        self.w_dl.set_visibility(bool(miss))
        if miss:
            k = HD.known(self.tkey)
            rows = ", ".join(f"{SSET.slot_label(f)} ({Path(k[f][1]).name})" for f in miss)
            tok = "" if SSET.read_hf_token() else " A Hugging Face token (Settings › Hugging Face token) is needed for gated models."
            self.w_dl_note.text = (f"Missing on this computer: {rows}. They can be downloaded from Hugging Face into "
                                   f"your models folder.{tok}")

    async def save(self, how: str):
        mk = self.w_mk.value
        if how == "over":
            msg = S.cs_save_over(self.ds, self.script, self.text, self.tkey, self.plan, mk)
            self.w_status.content = self.w_msg.content = msg
        else:
            r = S.cs_save_new(self.ds, self.w_newname.value, self.text, self.prompts_path, self.prompts_text,
                              self.tkey, self.script, self.plan, mk)
            self.apply(r)
            msg = r.get("status", "")
            self.w_msg.content = msg
        self.draw_repeat()
        self.page.bar.content = V.bar_html(self.page.s)
        return msg

    async def save_and_queue(self):
        can_over = self.script not in (None, "", S.NEW) and Path(self.script).is_file()
        msg = await self.save("over" if can_over else "new")
        if not str(msg or "").lstrip().startswith(("✔", "**Saved")):
            return
        sc = F.launch_scripts(self.ds)
        cur = Path(self.script) if self.script and self.script != S.NEW and Path(self.script).is_file() else None
        p = cur or (max(sc, key=lambda x: x.stat().st_mtime) if sc else None)
        if p is None:
            self.w_msg.content = "No launch script in this dataset yet: save one first."
            return
        with S.QLOCK:
            if any(i["path"] == str(p) and i["state"] in ("queued", "running") for i in S.QUEUE["items"]):
                self.w_msg.content = f"{p.name} is already in the queue."
                return
            S.QUEUE["items"].append({"path": str(p), "resume": "", "state": "queued", "note": ""})
        out = f"✔ {p.name} added to the queue."
        if self.w_auto.value and not S.QUEUE.get("running") and not S.RUN.running():
            S.queue_start(S.QUEUE.get("stop_on_fail", True))
            if S.QUEUE.get("running"):
                out += " Training starts now: the status strip at the top follows it."
        else:
            waiting = sum(1 for i in S.QUEUE["items"] if i["state"] == "queued")
            out += f" {waiting} waiting; the Queue tab starts the queue."
        self.w_msg.content = out
        self.w_queue.content = V.queue_card(self.ds)

    async def check(self):
        self.w_check.content = "Asking your trainer for its options …"
        self.w_check.content = await run.io_bound(S.cs_check, self.text)

    def autostart(self, v):
        import launchopts
        lo = launchopts.load()
        lo["queue_autostart"] = bool(v)
        launchopts.save(lo)

    # ------------------------------------------------------------------ build
    def build(self):
        ui.label("Train: the launch script").classes("text-lg font-semibold")
        if not self.types:
            note("Set the LoRA type first (step 2, Setup).")
            return
        labels = F.type_labels()
        ui.toggle({t: FT.label(labels.get(t, t), FT.trainer_for(t)) for t in self.types}, value=self.tkey,
                  on_change=lambda e: self.set_type(e.value)).props("no-caps unelevated")
        with ui.row().classes("items-center gap-3") as self.w_caps_row:
            ui.label("Caption set for this launch script").classes("font-semibold")
            self.w_caps = ui.radio({}, on_change=lambda e: self.pick_caps(e.value)).props("inline dense")
        with ui.card().props("flat bordered").classes("ls-card w-full") as self.w_dl:
            with ui.row().classes("w-full items-center gap-3"):
                self.w_dl_note = ui.label("").classes("grow")
                dl_msg = ui.label("").classes("ls-note")

                def download():
                    import hfdownload
                    dl_msg.text = hfdownload.start_background([self.tkey])
                ui.button("⬇ Download from Hugging Face", on_click=download).props("unelevated dense no-caps color=primary")
        with ui.row().classes("w-full items-end gap-3"):
            self.w_script = ui.select({S.NEW: "+ New launch script"}, value=S.NEW, label="Launch script",
                                      on_change=lambda e: self.set_script(e.value)).props("outlined dense").classes("w-96")
            self.w_prio = ui.select(PRIORITIES, value=S.CHOICE.get("priority") or "balanced", label="Priority",
                                    on_change=lambda: self.set_choice()).props("outlined dense").classes("w-48")
        self.w_status = ui.markdown("")
        with ui.card().props("flat bordered").classes("ls-card w-full gap-1"):
            ui.label("Your plan").classes("font-semibold")
            with ui.row().classes("w-full no-wrap items-start gap-4"):
                self.w_calc = ui.html("", sanitize=False).classes("grow basis-0")
                self.w_eff = ui.html("", sanitize=False).classes("grow basis-0")
        with ui.card().props("flat bordered").classes("ls-card w-full gap-1"):
            note("Settings · your value, with the recommendation and what it means below it. A change is written into "
                 "the script text when you leave the field; Save keeps it.")
            self.sets = []
            with ui.grid(columns=4).classes("w-full gap-3"):
                for flag, label, why in V.SETTINGS:
                    with ui.column().classes("gap-0"):
                        w = ui.input(label).props("outlined dense").classes("w-full")
                        info = ui.label(why).classes("ls-note text-xs")
                        w.on("blur", lambda: self.setting_left())
                        w.on("keydown.enter", lambda: self.setting_left())
                        self.sets.append((w, info))
            self.w_amsg = ui.markdown("")
        with ui.expansion("More settings: optimizer, scheduler, saving, model files, advanced, repeat folder") \
                .classes("w-full ls-card"):
            with ui.row().classes("w-full items-center gap-3"):
                self.w_steps = ui.number("Target total steps (0 = recommended for the type)", value=0, precision=0) \
                    .props("outlined dense").classes("w-72")
                self.w_userep = ui.checkbox("Plan with the recommended repeat count", value=True)
                self.w_hf = ui.checkbox("Fill HF_TOKEN from the token file", value=True)
            self.w_steps.on("keydown.enter", lambda: self.apply(S.cs_retype(self.ds, self.script, self.tkey, self.text,
                                                                            *self.args())))
            ui.markdown(S.recs_status()).classes("ls-note")
            self.fields = []
            for gname, flds in S.FORM_GROUPS:
                with ui.expansion(gname + (" (only the files this LoRA type uses)" if gname == "Model files" else "")) \
                        .classes("w-full"):
                    with ui.grid(columns=2 if gname == "Model files" else 4).classes("w-full gap-2"):
                        for name, label in flds:
                            w = ui.input(label).props(f'outlined dense hint="--{name}"').classes("w-full")
                            w.set_visibility(name not in S.STUDIO_TABLE)
                            w.on("blur", lambda: self.field_left())
                            self.fields.append(w)
            with ui.row().classes("gap-2"):
                ui.button("Use model files from Settings", on_click=lambda: self.apply(S.cs_use_settings(
                    self.ds, self.tkey, self.script, self.text, self.plan))).props("outline dense no-caps")
                ui.button("Fill HF_TOKEN in the script text", on_click=lambda: self.apply(S.cs_fill_hf(
                    self.text, self.plan))).props("outline dense no-caps")
            self.w_mk = ui.checkbox("Create missing folders when saving: the repeat folder (<N>_<name>), model and log. "
                                    "Images lying directly in the dataset folder are moved into a new repeat folder.",
                                    value=True)
            with ui.expansion("Repeat folder").classes("w-full"):
                self.rp_info = ui.markdown("")
                with ui.row().classes("w-full items-end gap-2"):
                    self.rp_folder = ui.select({}, label="Image folder").props("outlined dense").classes("grow")
                    self.rp_rep = ui.number("Repeats", value=0, precision=0).props("outlined dense").classes("w-28")
                    self.rp_name = ui.input("New folder name").props("outlined dense").classes("w-56")
                for w in (self.rp_folder, self.rp_rep):
                    w.on_value_change(lambda: setattr(self.rp_name, "value", SSET.rp_new_name(
                        self.ds, self.rp_folder.value, self.rp_rep.value)))
                rp_ok = ui.checkbox("Allow moving images that sit directly in the dataset folder into a new repeat folder")
                rp_msg = ui.markdown("")

                def rename():
                    msg, done = S.rp_rename(self.ds, self.rp_folder.value, self.rp_name.value, rp_ok.value)
                    rp_msg.content = msg
                    if done:
                        rp_ok.value = False
                        self.draw_repeat()

                def epochs():
                    self.apply(S.cs_set_epochs(self.ds, self.tkey, self.script, self.text, self.plan,
                                               self.w_steps.value or 0))
                with ui.row().classes("gap-2"):
                    ui.button("Rename / create repeat folder", on_click=rename).props("unelevated dense no-caps color=primary")
                    ui.button("Set epochs to hit the target steps", on_click=epochs).props("outline dense no-caps")
                    ui.button("Create missing folders now (repeat folder, model, log)", on_click=lambda: setattr(
                        rp_msg, "content", S.rp_make(self.ds, self.tkey, self.text, self.plan))).props("outline dense no-caps")
        with ui.expansion("Why these settings: notes and checks before training").classes("w-full ls-card"):
            self.w_explain = ui.markdown("")
        with ui.expansion("The script text").classes("w-full ls-card"):
            self.w_text = ui.codemirror("", language="Shell").classes("w-full h-[520px]")
            self.w_text.on_value_change(lambda e: self.text_changed(e.value))
        with ui.row().classes("w-full items-end gap-2"):
            ui.button("Fill with recommended settings", on_click=lambda: (
                self.apply(S.cs_suggest(self.ds, self.script, self.tkey, *self.args())), self.draw_repeat())) \
                .props("outline dense no-caps")
            ui.button("Check flags against my trainer", on_click=self.check).props("outline dense no-caps")
            self.w_save = ui.button("Save", on_click=lambda: self.save("over")).props("unelevated dense no-caps color=primary")
            self.w_newname = ui.input("Save as").props("outlined dense").classes("w-64")
            ui.button("Save as new version", on_click=lambda: self.save("new")).props("unelevated dense no-caps color=primary")
            q = ui.button("Save and add to the queue →", on_click=self.save_and_queue) \
                .props("unelevated dense no-caps color=amber-8")
            q.set_visibility(bool(self.tkey) and FT.can_train(self.tkey))
            import launchopts
            self.w_auto = ui.checkbox("Start the queue when nothing is training",
                                      value=launchopts.load()["queue_autostart"],
                                      on_change=lambda e: self.autostart(e.value))
        self.w_check = ui.markdown("")
        self.w_msg = ui.markdown("")
        with ui.card().props("flat bordered").classes("ls-card w-full gap-1"):
            ui.label("This dataset in the queue").classes("font-semibold")
            self.w_queue = ui.html(V.queue_card(self.ds), sanitize=False).classes("w-full")
        self.draw_caps()
        self.draw_download()
        self.load()


def step9(page):
    Train(page).build()
