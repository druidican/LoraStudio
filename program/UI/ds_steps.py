# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""The dataset page's steps, one builder per step (attach() registers them on the Datasets page). Each builder draws
the open step's controls into the page's box from the dataset as it is now; what a step does to the folder is in
dsflow.py, the HTML parts (grids, lists) come from datasets_ui."""
from __future__ import annotations
__version__ = "1.131126b"  # LoRA Studio version, see version.py

import re
import time

from nicegui import run, ui

import datacheck as DC
import datasets_ui as V
import dsflow as F
import sdgui as S
from widgets import note


def attach(page):
    for n, fn in ((2, step2), (3, step3)):
        page.steps[n] = (lambda fn=fn: fn(page))


def head(title: str, sub: str = ""):
    ui.label(title).classes("text-lg font-semibold")
    if sub:
        note(sub)


# --------------------------------------------------------------------------- 2 Setup
def save_setup(ds, types, purpose, wm, trig, words, worn, cls, caps) -> str:
    """Save step 2's choices; returns '' when it may go on, else what is missing."""
    m = F.meta(ds)
    if types:
        m["types"] = list(types)
    m.update(purpose=purpose or "character", wm_removed=bool(wm), trigger=(trig or ds).strip(),
             concept_words=(words or "").strip())
    m["caption_choice"] = caps or "auto"
    if DC.base_purpose(purpose or "character") == "character":
        m["class_word"] = re.sub(r"^(an?|the)\s+", "", (cls or "").strip(), flags=re.I)
    if purpose == "worn":
        m["worn_kind"] = worn or "jewelry"
    if not types:
        F.meta_save(ds, m)
        return "Pick at least one LoRA type, then press Continue again."
    if purpose in F.CONCEPT_KINDS and purpose != "worn" and not (words or "").strip():
        F.meta_save(ds, m)
        return ("Write the words for it in the field under Purpose (for example katana, sword), then press Continue "
                "again: the captions leave these words out, so the LoRA learns them.")
    m["steps"]["setup"] = {"at": time.strftime("%Y-%m-%d %H:%M")}
    F.meta_save(ds, m)
    return ""


def step2(page):
    ds = page.s["ds"]
    m = F.meta(ds)
    types = dict((v, lab) for lab, v in S.type_choices())
    head("Set up the dataset")
    with ui.row().classes("w-full no-wrap items-start gap-4"):
        with ui.card().props("flat bordered").classes("ls-card grow basis-3/5 gap-2"):
            t = ui.select(types, value=[x for x in m["types"] if x in types], multiple=True,
                          label="LoRA types (one or more)").props("outlined dense use-chips").classes("w-full")
            note("Several types: sorting uses the strictest size, step 7 writes one caption set per caption style, "
                 "step 9 one launch script per type.")
            caps = ui.radio(dict((v, lab) for lab, v in F.CAPTION_CHOICES), value=m.get("caption_choice") or "auto") \
                .props("inline dense")
            caps_l = note("Captions for Anima, FLUX and Chroma: these read both tags and sentences. Both: step 7 writes "
                          "a tag set and a sentence set, and step 9 lets you pick which one a launch script trains on.")
            ui.label("Purpose").classes("ls-note mt-2")
            pu = ui.radio(dict((v, lab) for lab, v in F.PURPOSES), value=m["purpose"]).props("inline dense")
            words = ui.input(F.CONCEPT_ASK.get(m["purpose"], F.CONCEPT_ASK["object"]),
                             value=m.get("concept_words") or "").props("outlined dense").classes("w-full")
            worn = ui.radio(dict((v, lab) for lab, v in F.WORN_KINDS), value=m.get("worn_kind") or "jewelry") \
                .props("inline dense")
            wm = ui.checkbox("Watermarks are already removed (unticked: the detailed sort also looks for them and step 6 "
                             "offers a crop)", value=m["wm_removed"])
            trig = ui.input("Trigger word (first word of every caption)", value=m["trigger"] or ds) \
                .props("outlined dense").classes("w-full")
            cls = ui.input("Class word (sentence captions)", value=m.get("class_word", "person"), placeholder="person") \
                .props("outlined dense").classes("w-full")
            cls_l = note("For a character: what the trigger word is (person, woman, man, robot, cat …). Sentence "
                         "captions begin \"trigger, a class word, …\". Tag captions are not changed.")
            msg = ui.markdown("")

            def shown():
                p = pu.value or "character"
                words.set_visibility(p in F.CONCEPT_KINDS)
                words.props(f'label="{F.CONCEPT_ASK.get(p, F.CONCEPT_ASK["object"])}"')
                worn.set_visibility(p == "worn")
                ch = DC.base_purpose(p) == "character"
                cls.set_visibility(ch)
                cls_l.set_visibility(ch)
                both = any(F.can_both(x) for x in t.value or [])
                caps.set_visibility(both)
                caps_l.set_visibility(both)
                card_html, missing = V.setup_card(t.value or [], p)
                ready.content = card_html
                dl.set_visibility(missing)

            def go_on():
                why = save_setup(ds, t.value or [], pu.value, wm.value, trig.value, words.value, worn.value, cls.value,
                                 caps.value)
                if why:
                    msg.content = why
                else:
                    page.go(step=3, tkey=(t.value or [""])[0])
            ui.button("Continue →", on_click=go_on).props("unelevated no-caps color=primary")
        with ui.column().classes("grow basis-2/5 gap-2"):
            ready = ui.html("", sanitize=False).classes("w-full")
            dlmsg = ui.markdown("")

            def download():
                import hfdownload
                dlmsg.content = hfdownload.start_background(t.value or [])
            dl = ui.button("⬇ Download the missing model files", on_click=download).props("outline dense no-caps")
            imgs = F.images(ds)
            strip = "".join(f'<img src="{V.thumb(p, 110)}" alt="">' for p in imgs[:24])
            ui.html(f'<div class="ls-card"><b>All {len(imgs)} pictures</b><div class="ds-strip">{strip}</div>'
                    + (f'<div class="ls-note">… {len(imgs) - 24} more</div>' if len(imgs) > 24 else "") + "</div>",
                    sanitize=False).classes("w-full")
    for w in (t, pu):
        w.on_value_change(lambda: shown())
    shown()


# --------------------------------------------------------------------------- 3 Usable scan
def step3(page):
    ds = page.s["ds"]
    head("Scan for usable pictures (optional)")
    n_extra = len(F.extra_pictures(ds))
    extra = ui.checkbox((f"Also look at the {n_extra} picture{'s' if n_extra != 1 else ''} outside the repeat folder "
                         "(_reserve/, _extra/ and other folders)") if n_extra else
                        "Also look at pictures outside the repeat folder (there are none)",
                        value=F.include_extra(ds))
    extra.set_enabled(bool(n_extra) or F.include_extra(ds))
    note("Off: only the repeat folder (<N>_<dataset name>) is the dataset. On: the scan, the sort and the selection also "
         "see those pictures, and the ones you keep in step 5 join the repeat folder. It switches off again when a "
         "launch script is saved.")
    md = ui.html("", sanitize=False).classes("w-full")
    with ui.row().classes("items-center gap-2"):
        ui.button("Skip", on_click=lambda: page.go(step=4)).props("flat dense no-caps")
        scan_b = ui.button("Scan the pictures").props("unelevated dense no-caps color=primary")
        show = ui.toggle(["Not usable first", "All"], value="Not usable first").props("dense no-caps unelevated")
    g = ui.html("", sanitize=False).classes("w-full")
    with ui.row().classes("items-center gap-2"):
        move_b = ui.button("Move not usable to _reserve").props("outline dense no-caps")
        ui.button("Continue →", on_click=lambda: page.go(step=4)).props("unelevated dense no-caps color=primary")
    msg = ui.markdown("")

    def draw(sc=None):
        sc = sc or F._load(ds, "scan.json")
        if not sc:
            md.content = (f"<p><b>{len(F.candidates(ds))} pictures.</b> Size, sharpness, compression and broken files; "
                          "no AI, about a second per 10 pictures. Skip it when you know the pictures are fine.</p>")
            g.content = ""
            return
        md.content, g.content = V.s3_render(ds, sc, show.value)

    async def scan():
        md.content = "<p class='ls-note'>Scanning …</p>"
        sc = await run.io_bound(F.usable_scan, ds, F.meta(ds)["types"])
        draw(sc)

    def move():
        sc = F._load(ds, "scan.json")
        if not sc:
            msg.content = "Scan first."
            return
        main = {str(p) for p in F.images(ds)}
        bad = [p for p, e in sc["pics"].items() if e["bad"] and p in main and __import__("os").path.exists(p)]
        if not bad:
            msg.content = "Nothing to move (only pictures in the repeat folder are moved)."
            return
        n, err = F.move_to_reserve(ds, bad)
        msg.content = f"✔ {n} moved to _reserve/." + (f" {'; '.join(err[:3])}" if err else "")
        draw()

    def set_extra(e):
        F.set_include_extra(ds, e.value)
        draw()
        if e.value and F._load(ds, "scan.json"):
            msg.content = "Scan again to check the pictures from outside the repeat folder."
    scan_b.on_click(scan)
    move_b.on_click(move)
    show.on_value_change(lambda: draw())
    extra.on_value_change(set_extra)
    draw()
