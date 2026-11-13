# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""The dataset page's steps, one builder per step (attach() registers them on the Datasets page). Each builder draws
the open step's controls into the page's box from the dataset as it is now; what a step does to the folder is in
dsflow.py, the HTML parts (grids, lists) come from datasets_ui."""
from __future__ import annotations
__version__ = "1.131126c"  # LoRA Studio version, see version.py

import json
import re
import time
from pathlib import Path

from nicegui import run, ui

import datacheck as DC
import datasets_ui as V
import detailscan as DSC
import dsflow as F
import features as FT
import sdgui as S
from widgets import CropPanel, PaintPanel, note, pic_url, stream


def attach(page):
    import ds_results
    import ds_train
    for n, fn in ((2, step2), (3, step3), (4, step4), (5, step5), (6, step6), (7, step7), (8, step8),
                  (9, ds_train.step9), (10, ds_results.step10), (11, step11)):
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


# --------------------------------------------------------------------------- 4 Sort
def step4(page):
    ds = page.s["ds"]
    m = F.meta(ds)
    ol = FT.have("ollama")
    head("Sort with Ollama" if ol else "Sort",
         "Pass 1 is quick (no AI): too blurry, too small for your LoRA types, duplicates. Pass 2 looks at each picture "
         "with the Ollama model: one clear subject, framing, outfit, place, quality, watermarks and how much it shows of "
         "the subject. Pictures that hide the subject or show almost nothing of it are left out; a part with a feature "
         "(tail, horns, a tattoo) is kept as a detail shot. Step 5 lets you overrule any of it with pins.")
    if not ol:
        note("Pass 2 · detailed (Ollama missing): only the quick pass runs, so framing, outfits, places and watermarks "
             "are not known (step 5 then picks by quality only).")
    if ol and DC.base_purpose(m["purpose"]) == "character":
        ui.checkbox("Detailed scan (characters): after the sort, look at the dataset as a whole (framings, angles, "
                    "poses, expressions, outfits, places and light) and suggest pictures to add, upscale, sharpen or crop "
                    "in step 5. About twice the time of the sort.", value=bool(m.get("detailed_scan")),
                    on_change=lambda e: DSC.set_enabled(ds, e.value))
    status = ui.label("").classes("ls-note")
    box = ui.html(V.s4_html(ds), sanitize=False).classes("w-full")

    async def sort():
        mm = F.meta(ds)
        if not mm["types"]:
            status.text = "Set the LoRA type first (step 2, Setup)."
            return
        url, model = S.oc_settings()

        def lines():
            for line in F.sort_run(ds, mm["types"], mm["purpose"], mm["wm_removed"], url, model):
                if line == "done":
                    break
                yield f"⏳ {line}"
            if DSC.enabled(ds) and (F.sort_result(ds) or {}).get("ai") and not F.STOP.is_set():
                for line in DSC.run(ds, url, model):
                    if line == "done":
                        r = DSC.result(ds) or {}
                        ni, ng = len(r.get("items") or []), len(r.get("gaps") or [])
                        yield f"✔ Detailed scan: {ni} suggestions, {ng} gaps. See step 5."
                        return
                    yield f"⏳ {line}"
            yield ""
        await stream(lines(), lambda line: setattr(status, "text", line))
        box.content = V.s4_html(ds)
        page.bar.content, page.stepper.content = V.bar_html(page.s), V.stepper_html(page.s)
    with ui.row().classes("items-center gap-2"):
        ui.button("Sort" if ol else "Sort (quick pass only)", on_click=sort).props("unelevated dense no-caps color=primary")
        ui.button("Stop", on_click=F.STOP.set).props("flat dense no-caps color=negative")
    ui.button("Continue to Select →", on_click=lambda: page.go(step=5)).props("unelevated no-caps color=primary")


# --------------------------------------------------------------------------- 5 Select
def step5(page):
    ds = page.s["ds"]
    head("Select the best pictures",
         "📌 Always keeps a picture whatever the rules say (a unique detail the sort missed); ⛔ Never leaves one out "
         "(useless content the sort kept). Your pins are saved at once, stay after a new sort and count in the automatic "
         "improvement too.")
    n, rng, why = F.suggest_size(ds)
    mode0 = F.meta(ds).get("select_mode") or F.batch_settings().get("select_mode") or "safe"
    with ui.row().classes("w-full no-wrap items-start gap-4"):
        with ui.card().props("flat bordered").classes("ls-card grow basis-0 gap-2"):
            ui.html(V.s5_suggest_html(ds, n, rng, why), sanitize=False)
            with ui.row().classes("items-center gap-2"):
                keep = ui.number("Keep", value=n, min=1, precision=0).props("outlined dense").classes("w-28")
                ui.button("Use the suggestion", on_click=lambda: (keep.set_value(F.suggest_size(ds)[0]), draw())) \
                    .props("outline dense no-caps")
            mode = ui.radio({"safe": "Safe: the cleanest pictures first",
                             "varied": "Varied: more close-ups, angles, outfits and details"}, value=mode0)
        with ui.card().props("flat bordered").classes("ls-card grow basis-0"):
            mix = ui.html("", sanitize=False)
    r = DSC.result(ds)
    if r and DSC.enabled(ds):
        with ui.card().props("flat bordered").classes("ls-card w-full gap-2"):
            dhtml = ui.html(V.detail_html(ds, r), sanitize=False)
            dpick = ui.select({i["id"]: i["label"] for i in r["items"]}, multiple=True,
                              label="Tick what to do (nothing happens until you press the button)") \
                .props("outlined dense use-chips").classes("w-full")
            dmsg = ui.label("").classes("ls-note")

            async def do_ticked():
                ids = list(dpick.value or [])
                await stream(DSC.apply(ds, ids), lambda line: setattr(dmsg, "text", line))
                rr = DSC.result(ds) or {}
                dhtml.content = V.detail_html(ds, rr)
                dpick.options = {i["id"]: i["label"] for i in rr.get("items") or []}
                dpick.value = []
                dpick.update()
                draw()
            with ui.row().classes("items-center gap-2"):
                ui.button("Tick all suggestions", on_click=lambda: dpick.set_value(list(dpick.options))) \
                    .props("flat dense no-caps")
                ui.button("Do the ticked", on_click=do_ticked).props("unelevated dense no-caps color=primary")
                ui.button("Stop", on_click=F.STOP.set).props("flat dense no-caps color=negative")
    with ui.row().classes("items-center gap-2"):
        ui.button("Tick all", on_click=lambda: ui.run_javascript("dsTickAll(true)")).props("flat dense no-caps")
        ui.button("Untick all", on_click=lambda: ui.run_javascript("dsTickAll(false)")).props("flat dense no-caps")
    g = ui.html("", sanitize=False).classes("w-full").props('id="ds_s5"')
    msg = ui.markdown("")

    def draw():
        mix.content, g.content = V.s5_grid(ds, keep.value)

    def set_mode(e):
        mm = F.meta(ds)
        mm["select_mode"] = e.value
        F.meta_save(ds, mm)
        draw()

    async def confirm():
        sel = await ui.run_javascript("Array.from(document.querySelectorAll('#ds_s5 .ds-cell input:checked'))"
                                      ".map(e => e.closest('.ds-cell').dataset.p)")
        if not sel:
            msg.content = "Nothing is ticked."
            return
        msg.content = F.apply_select(ds, sel)
        page.go(step=6)

    def back():
        msg.content = f"✔ {F.back_from_reserve(ds)[0]} pictures put back."
        draw()
    mode.on_value_change(set_mode)
    keep.on("keydown.enter", lambda: draw())
    keep.on("blur", lambda: draw())
    with ui.row().classes("items-center gap-2"):
        ui.button("Confirm selection →", on_click=confirm).props("unelevated no-caps color=primary")
        ui.button("Move all back from _reserve", on_click=back).props("outline no-caps")
    draw()


# --------------------------------------------------------------------------- 6 Watermarks and size
def step6(page):
    import sd_prepare as SP
    ds = page.s["ds"]
    head("Watermarks and size")
    saved = (F._load(ds, "resize.json", {}) or {}).get("max_side")
    with ui.expansion("Watermarks", value=True).classes("w-full ls-card"):
        note("Pictures where the sort found marks added on top of the picture (watermarks, site names, logo stamps, "
             "signatures, date stamps): remove them (picture tools), crop them off, set the picture aside, or keep it "
             "(it may be learned). Text that belongs to the scene (signs, shirts) is left alone.")
        wm_sum = ui.markdown("")
        wm_html = ui.html("", sanitize=False).classes("w-full").props('id="wm_html"')
        wm_msg = ui.label("").classes("ls-note")
        with ui.card().props("flat bordered").classes("ls-card w-full gap-2") as tool:
            tool_title = ui.label("").classes("font-semibold")
            with ui.column().classes("w-full") as paint_col:
                note("Paint over the mark with the brush (a little over its edges is fine). The picture tools fill in "
                     "what is under it; the rest of the picture stays as it is.")
                paint = PaintPanel()
            with ui.column().classes("w-full max-w-2xl") as crop_col:
                note("Move the edges until the mark is outside; the shaded part is cut off.")
                tool_rp = {"rp": ""}

                def wm_crop_change(l, t, r, b, aspect):
                    return SP.wm_crop_show(ds, tool_rp["rp"], l * 100, r * 100, t * 100, b * 100, aspect)
                wm_crop = CropPanel(wm_crop_change)
            with ui.row().classes("items-center gap-2"):
                paint_go = ui.button("Remove what I painted").props("unelevated dense no-caps color=primary")
                crop_go = ui.button("Use this crop").props("unelevated dense no-caps color=primary")
                ui.button("Close", on_click=lambda: tool.set_visibility(False)).props("flat dense no-caps")
            tool_msg = ui.label("").classes("ls-note")
        tool.set_visibility(False)

    def wm_draw():
        wm_sum.content, wm_html.content = SP.wm_render(ds)

    def open_tool(v):
        act, _, rp = v.partition("|")
        if act not in ("paint", "crop") or not rp:
            return
        tool_rp["rp"] = rp
        src, box = SP.wm_tool_info(ds, rp)
        tool.set_visibility(True)
        tool_title.text = ("Paint over the mark: " if act == "paint" else "Crop it yourself: ") + rp
        paint_col.set_visibility(act == "paint")
        crop_col.set_visibility(act == "crop")
        paint_go.set_visibility(act == "paint")
        crop_go.set_visibility(act == "crop")
        tool_msg.text = ""
        if act == "paint":
            paint.show(pic_url(src, 1600))
        else:
            wm_crop.set(box, F.crop_preview(str(src), box))

    async def do_paint():
        a = await paint.mask()
        tool_msg.text = "⏳ The picture tools fill in what you painted …"
        msg = await run.io_bound(SP.wm_paint_mask, ds, tool_rp["rp"], a)
        wm_msg.text = msg
        if msg.startswith("✔"):
            tool.set_visibility(False)
        wm_draw()

    def do_crop():
        l, t, r, b = wm_crop.box()
        msg = SP.wm_crop_apply(ds, tool_rp["rp"], l * 100, r * 100, t * 100, b * 100, wm_crop.aspect.value)[0]
        wm_msg.text = msg
        if msg.startswith("✔"):
            tool.set_visibility(False)
        wm_draw()
    paint_go.on_click(do_paint)
    crop_go.on_click(do_crop)

    async def wm_apply(only_pending: bool):
        choices = "" if only_pending else json.dumps(await ui.run_javascript(
            "var o = {}; document.querySelectorAll('#wm_html .wm-row').forEach(function(r){ "
            "var s = r.querySelector('.wm-opt.sel'); if (s) o[r.dataset.p] = s.dataset.c; }); return o;") or {})
        await stream(SP.wm_apply_live(ds, choices, only_pending=only_pending), lambda line: setattr(wm_msg, "text", line))
        wm_draw()
        draw()
    page.on_wm_tool = open_tool

    sm = ui.html("", sanitize=False).classes("w-full")
    with ui.row().classes("items-center gap-2"):
        ms = ui.number("Longest side at most (px)", value=int(saved or F.default_max_side(F.meta(ds)["types"])),
                       min=256, precision=0).props("outlined dense").classes("w-56")
        ui.button("Put the originals back", on_click=lambda: (setattr(msg, "content", F.restore_originals(ds)), draw())) \
            .props("outline dense no-caps")
    with ui.row().classes("w-full no-wrap items-start gap-4"):
        with ui.column().classes("grow basis-3/5 min-w-0 gap-1"):
            note("Size, per picture: what will happen to each picture; change it in the Do column, or click a picture "
                 "to crop it yourself.")
            table = ui.html("", sanitize=False).classes("w-full")
        with ui.card().props("flat bordered").classes("ls-card grow basis-2/5 gap-2") as crop_card:
            cname = ui.label("").classes("font-semibold")
            cur = {"rel": ""}

            def crop_change(l, t, r, b, aspect):
                p = F.root(ds) / cur["rel"]
                from PIL import Image, ImageOps
                with Image.open(p) as im:
                    w, h = ImageOps.exif_transpose(im).size
                bx = F.fit_box(w, h, l, t, r, b, aspect)
                return F.crop_preview(str(p), bx), f"result {int((bx[2] - bx[0]) * w)}×{int((bx[3] - bx[1]) * h)}"
            crop = CropPanel(crop_change)

            def person():
                bx = F.person_box(str(F.root(ds) / cur["rel"]), crop.aspect.value, run=True)
                if not bx:
                    crop.note.text = ("No person found (or the picture tools are not installed: Settings › Trainers "
                                      "and GPU).")
                    return
                crop.set(bx, F.crop_preview(str(F.root(ds) / cur["rel"]), bx), "suggested: around the person")

            def use():
                p = F.root(ds) / cur["rel"]
                from PIL import Image, ImageOps
                with Image.open(p) as im:
                    w, h = ImageOps.exif_transpose(im).size
                over = F.resize_overrides(ds)
                over[cur["rel"]] = {"mode": "crop", "box": F.fit_box(w, h, *crop.box(), crop.aspect.value)}
                F.resize_save_over(ds, over, int(ms.value or 1536))
                crop_card.set_visibility(False)
                draw()
            with ui.row().classes("gap-2"):
                ui.button("Reset", on_click=lambda: (crop.aspect.set_value("free"), crop.set([0, 0, 1, 1]),
                                                     crop.changed())).props("flat dense no-caps")
                ui.button("Around the person", on_click=person).props("outline dense no-caps")
                ui.button("Use this crop", on_click=use).props("unelevated dense no-caps color=primary")
        crop_card.set_visibility(False)
    msg = ui.markdown("")

    def draw():
        a, b, label = V.s6_html(ds, int(ms.value or 1536))
        sm.content, table.content = a, b
        go_b.text = label

    def pick(path):
        rel = str(Path(path).relative_to(F.root(ds)))
        cur["rel"] = rel
        o = F.resize_overrides(ds).get(rel) or {}
        auto = None if o.get("box") else F.person_box(str(F.root(ds) / rel))
        box = o.get("box") or auto or [0, 0, 1, 1]
        crop_card.set_visibility(True)
        cname.text = "Crop " + Path(rel).name
        crop.set(box, F.crop_preview(str(F.root(ds) / rel), box), "suggested: around the person" if auto else "")

    def rmode(arg):
        rel, _, mode = arg.rpartition("|")
        over = F.resize_overrides(ds)
        if mode in ("scale", "none", "upscale"):
            over.pop(rel, None)
        elif mode == "keep":
            over[rel] = {"mode": mode}
        elif mode == "crop":
            return pick(str(F.root(ds) / rel))
        F.resize_save_over(ds, over, int(ms.value or 1536))
        draw()

    async def resize():
        r = await stream(F.apply_resize_steps(ds, int(ms.value or 1536)), lambda line: setattr(msg, "content", line))
        msg.content = r or ""
        page.go(step=7)
    page.on_pick = pick
    page.on_rmode = rmode
    ms.on("keydown.enter", lambda: draw())
    ms.on("blur", lambda: draw())
    with ui.row().classes("items-center gap-2"):
        ui.button("Recommended for the undecided", on_click=lambda: wm_apply(True)).props("outline dense no-caps")
        ui.button("Apply the watermark choices", on_click=lambda: wm_apply(False)).props("outline dense no-caps")
        ui.button("Watermarks are done", on_click=lambda: (setattr(msg, "content", F.wm_done(ds)),
                                                           setattr(page.bar, "content", V.bar_html(page.s)))) \
            .props("outline dense no-caps")
        go_b = ui.button("Resize →", on_click=resize).props("unelevated no-caps color=primary")
    wm_draw()
    draw()


# --------------------------------------------------------------------------- 7 Captions
LEAVE_LABEL = {"character": "Never describe (always part of the person)",
               "style": "Never describe (the style itself: the trigger word learns it)",
               "object": "Never describe (the object itself: the trigger word learns it)",
               "worn": "Never describe (the item's own look: the trigger word learns it)",
               "action": "Never describe (the pose or action itself: the trigger word learns it)",
               "concept": "Never describe (the effect itself: the trigger word learns it)"}

CC_PAGE_JS = """
<script>
window.ccOpen = function(btn){
  var d = btn.closest('.cc-odd'); if (!d) return;
  document.querySelectorAll('#cc_html .cc-odd.cc-sel').forEach(function(e){ e.classList.remove('cc-sel'); });
  d.classList.add('cc-sel');
  emitEvent('cc_open', d.dataset.p);
};
</script>
"""


def leave_default(ds) -> str:
    m = F.meta(ds)
    if m.get("leave_out") is not None:
        return m["leave_out"]
    CAP = S.CAP
    if CAP is None:
        return ""
    words = m.get("concept_words") or ""
    if m["purpose"] in F.CONCEPT_KINDS and words:
        return words
    return CAP.STYLE_LEAVE_OUT if m["purpose"] == "style" else CAP.DEFAULT_LEAVE_OUT


def step7(page):
    import sd_prepare as SP
    ds = page.s["ds"]
    m = F.meta(ds)
    purpose = m["purpose"]
    if S.CAP is not None:
        S.CAP.PURPOSE.update(value=purpose, words=m.get("concept_words") or "", **{"class": m.get("class_word", "person")})
    tkey = page.s.get("tkey") if page.s.get("tkey") in m["types"] else (m["types"] or [""])[0]
    url0, model0 = S.oc_settings()
    head("Tag every picture",
         "Tags (Illustrious, Pony, SDXL anime, Anima) are written by the WD tagger (run by sd-scripts; the first time it "
         "downloads about 1.2 GB). Only general tags: no character names, so the trigger word stands for them; the Never "
         "describe list and the consistency check apply as usual. Sentences (FLUX, Qwen-Image, Z-Image, …) are written by "
         "Ollama.")
    if not FT.have("ollama"):
        note("Ollama is not installed: tag sets work, sentence captions do not. Write those yourself in 8 Review (click "
             f"a picture), or put .txt files next to the pictures. Ollama: {FT.ollama_guide()}, then Settings › Start "
             "and network.")
    ui.html(V.s7_rec_html(purpose), sanitize=False).classes("w-full")
    sets_html = ui.html(V.s7_sets_html(ds), sanitize=False).classes("w-full")
    with ui.expansion("Caption settings: trigger word, never describe, length, detail, try on 3 pictures").classes("w-full ls-card"):
        note("The Ollama model (Settings › AI helpers) looks at every picture and writes a caption that starts with the "
             "trigger word. For a person, captions describe what changes (clothes, pose, place, light, framing) and leave "
             "out what is always that person, so the trigger word learns it. Try on 3 pictures writes nothing.")
        with ui.row().classes("w-full no-wrap gap-2"):
            trig = ui.input("Trigger word (first word of every caption)", value=m["trigger"] or ds) \
                .props("outlined dense").classes("w-64")
            leave = ui.input(LEAVE_LABEL.get(purpose, LEAVE_LABEL["character"]), value=leave_default(ds)) \
                .props("outlined dense").classes("grow")
        rec_md, style0, length0, detail0 = SP.oc_recommend(tkey, purpose)
        with ui.row().classes("w-full items-end gap-4"):
            style = ui.toggle({"sentences": "Sentences", "tags": "Tags"}, value=style0).props("dense no-caps unelevated")
            with ui.column().classes("gap-0 grow"):
                ui.label("Length (1 very short … 5 very long)").classes("ls-note")
                length = ui.slider(min=1, max=5, step=1, value=length0).props("label markers")
            with ui.column().classes("gap-0 grow"):
                ui.label("Detail (1 basic … 4 everything)").classes("ls-note")
                detail = ui.slider(min=1, max=4, step=1, value=detail0).props("label markers")
        rec = ui.markdown(rec_md)
        ex = ui.html("", sanitize=False).classes("w-full").props('id="oc_ex"')

        def example():
            ex.content = SP.oc_example(style.value, length.value, detail.value, trig.value, tkey)

        def recommended():
            md, s_, l_, d_ = SP.oc_recommend(tkey, purpose)
            rec.content = md
            style.value, length.value, detail.value = s_, l_, d_
            example()
        with ui.row().classes("items-center gap-2"):
            hints = ui.checkbox("Give the old tags to the model as hints (they can be wrong)")
            ui.button("Use recommended", on_click=recommended).props("flat dense no-caps")
            prev_b = ui.button("Try on 3 pictures").props("outline dense no-caps")
            ui.button("Undo last change", on_click=lambda: setattr(out, "content", SP.rv_undo())).props("flat dense no-caps")
        for w in (style, length, detail, trig):
            w.on_value_change(lambda: example())
        example()
    out = ui.markdown("")

    def keep_leave():
        mm = F.meta(ds)
        mm["leave_out"] = leave.value
        F.meta_save(ds, mm)

    async def preview():
        keep_leave()
        await stream(SP.oc_preview(ds, url0, model0, trig.value, leave.value, hints.value, style.value, length.value,
                                   detail.value), lambda line: setattr(out, "content", line))
    prev_b.on_click(preview)

    async def write(only_missing=False):
        keep_leave()
        sets = F.caption_sets(ds)
        msg = ""

        def lines():
            nonlocal msg
            for st_ in sets:
                last = ""
                for line in F.write_set(ds, url0, model0, trig.value or ds, leave.value, st_["style"], length.value,
                                        detail.value, st_["ext"], only_missing=only_missing):
                    last = line
                    yield (msg + "\n\n" if msg else "") + line
                msg += ("\n\n" if msg else "") + last
            yield msg
        await stream(lines(), lambda line: setattr(out, "content", line))
        sets_html.content = V.s7_sets_html(ds)
        if is_char or worn:
            mv_draw()
        if worn:
            kind_msg.content = F.worn_balance(ds)
    with ui.row().classes("items-center gap-2"):
        ui.button("Write the captions", on_click=lambda: write(False)).props("unelevated dense no-caps color=primary")
        ui.button("Only pictures without a caption", on_click=lambda: write(True)).props("outline dense no-caps")
        ui.button("Stop", on_click=lambda: (F.STOP.set(), S.OC_STOP.update(flag=True))).props("flat dense no-caps color=negative")

    worn = purpose == "worn"
    is_char = DC.base_purpose(purpose) == "character"
    # ---- worn item: kinds and places
    with ui.card().props("flat bordered").classes("ls-card w-full gap-2") as kind_card:
        ui.label("Kinds and where they are worn").classes("font-semibold")
        note("One line per kind of item you want to ask for in a prompt, with the other words the tagger uses for it "
             "after = (for example hair clip = hairclip, hair ornament), and the same for where they are worn (on index "
             "finger, in hair). Make the words the same turns the other words into the kind's name in every caption and "
             "puts the trigger word, the kinds and the places first. It also counts the pictures per kind.")
        w = F.worn_setup(ds)
        with ui.row().classes("w-full no-wrap gap-2"):
            kinds = ui.textarea("Kinds (one per line: name = other words)", value=w["kinds"],
                                placeholder="ring = rings\nhair clip = hairclip, hair ornament").props("outlined").classes("grow")
            places = ui.textarea("Where they are worn (one per line: name = other words)", value=w["places"],
                                 placeholder="on index finger\nin hair = hair clip in hair").props("outlined").classes("grow")
        kind_msg = ui.markdown(F.worn_balance(ds) if worn else "")

        def kind_suggest():
            k, pl = F.suggest_worn(ds, ".txt", trig.value or ds)
            if not (k or pl):
                kind_msg.content = "No kinds found in the captions yet: write the tag captions first."
                return
            merge = lambda cur, new: "\n".join(dict.fromkeys([x for x in (cur or "").splitlines() if x.strip()]
                                                             + [x for x in new.splitlines() if x.strip()]))
            kinds.value, places.value = merge(kinds.value, k), merge(places.value, pl)
            kind_msg.content = ("Found in the captions: check the lines, remove what you don't want to ask for, add the "
                                "rest, then Make the words the same.")

        def kind_save():
            F.set_worn_setup(ds, kinds.value, places.value)
            kind_msg.content = "✔ Saved. " + F.worn_balance(ds)

        def kind_go():
            F.set_worn_setup(ds, kinds.value, places.value)
            kind_msg.content = F.same_words(ds, ".txt", trig.value or ds)
            mv_draw()
        with ui.row().classes("gap-2"):
            ui.button("Suggest from the tags", on_click=kind_suggest).props("outline dense no-caps")
            ui.button("Save", on_click=kind_save).props("flat dense no-caps")
            ui.button("Make the words the same", on_click=kind_go).props("unelevated dense no-caps color=primary")
    kind_card.set_visibility(worn)
    # ---- tags -> trigger words
    with ui.card().props("flat bordered").classes("ls-card w-full gap-2") as mv_card:
        ui.label("Item tags → trigger words" if worn else "Character tags → trigger words").classes("font-semibold")
        note(("For a worn item the WD tagger writes every tag, also the item's own look (metal, stones, pattern, "
              "material …). Tick the ones that are the item's look: Move takes them out of the captions (so the trigger "
              "word learns them) and puts them in the Item description below. A tag with a kind in it (gold ring) leaves "
              "the kind (ring).") if worn else
             ("For a character the WD tagger writes every tag, also the character's own (hair, eyes, freckles …). Tick "
              "the ones that belong to the character: Move takes them out of the captions (so the trigger word learns "
              "them) and puts them in the Character description below. Ticked already: the traits in most pictures. "
              "The captions with all tags are kept, so Put them back undoes it."))
        mv_tags = ui.select({}, multiple=True, label="Tags of the caption set (pictures that have them)") \
            .props("outlined dense use-chips").classes("w-full")
        mv_msg = ui.markdown("")

        def mv_draw(msg=""):
            rows, n = F.tag_counts(ds, ".txt", trig.value or ds)
            if not n:
                mv_tags.options, mv_tags.value = {}, []
                mv_tags.update()
                mv_msg.content = msg or "No captions yet: write the tag captions above first."
                return
            mm = F.meta(ds)
            kk, pp = F.worn_names(ds)
            mv_tags.options = {t: f"{t} ({c})" for t, c in rows[:120]}
            mv_tags.value = F.default_ticks(rows, n, mm["purpose"], mm.get("worn_kind"), kk, pp)
            mv_tags.update()
            moved = F.moved_tags(ds)
            mv_msg.content = msg or (f"{n} captions, {len(rows)} different tags."
                                     + (f" Moved already: {', '.join(moved)}." if moved else ""))

        def mv_go():
            msg = F.move_tags(ds, mv_tags.value or [], ".txt", trig.value or ds)
            mv_draw(msg)
            desc.value = F.description(ds)

        def mv_back():
            msg = F.put_back(ds, ".txt")
            mv_draw(msg)
            desc.value = F.description(ds)
        with ui.row().classes("gap-2"):
            ui.button("Read the tags", on_click=lambda: mv_draw()).props("outline dense no-caps")
            ui.button("Move the ticked tags", on_click=mv_go).props("unelevated dense no-caps color=primary")
            ui.button("Put them back", on_click=mv_back).props("flat dense no-caps")
    mv_card.set_visibility(is_char or worn)
    # ---- description next to the trigger word
    with ui.card().props("flat bordered").classes("ls-card w-full gap-2") as desc_card:
        ui.label(("Item" if worn else "Character") + " description, next to the trigger word").classes("font-semibold")
        note("The traits are not in the captions (moved above, or never written), so the trigger word learns them. Name "
             "them here too: they are added to the LoRA's trigger words in its LoRA Manager data, so ComfyUI offers them "
             "next to the trigger word. The captions do not change. Suggest takes the traits the WD tagger saw in most "
             "pictures (never ethnicity, skin or age).")
        desc = ui.input("Item description" if worn else "Character description", value=F.description(ds),
                        placeholder="silver, leaf design, small green stones" if worn else
                        "round glasses, red scarf, silver earrings").props("outlined dense").classes("w-full")
        desc_msg = ui.markdown("")

        def desc_save():
            d = F.set_description(ds, desc.value)
            desc.value = d
            desc_msg.content = ((f"✔ Saved. The LoRAs of this dataset get {d} next to their trigger word when they are "
                                 "scanned (Collection › Maintenance › Sync now passes it on to LoRA Manager).") if d
                                else "✔ No description: only the trigger word.")

        def desc_suggest():
            sug = F.suggest_description(ds)
            if not sug:
                desc_msg.content = ("Nothing to suggest yet: write the tag captions with the WD tagger first (tags only; "
                                    "for sentence captions type the description yourself).")
                return
            words = [w_.strip() for w_ in (desc.value or "").split(",") if w_.strip()]
            for w_ in sug.split(", "):
                if w_.lower() not in {x.lower() for x in words}:
                    words.append(w_)
            desc.value = ", ".join(words)
            desc_msg.content = "Suggested from the tags: check it, add build or height yourself, then Save."
        with ui.row().classes("gap-2"):
            ui.button("Save", on_click=desc_save).props("unelevated dense no-caps color=primary")
            ui.button("Suggest from the tags", on_click=desc_suggest).props("outline dense no-caps")
    desc_card.set_visibility(is_char or worn)
    # ---- consistency
    with ui.card().props("flat bordered").classes("ls-card w-full gap-2"):
        ui.label("Consistency, checked while tagging").classes("font-semibold")
        note("The traits of the character itself that the captions mention (hair, eyes, build, skin, horns, tattoos, a "
             "signature object): the same in most pictures is taken out of the captions so the trigger word learns it; "
             "words that mean the same become one word; odd pictures that say something else are shown so you can look. "
             "Clothes, places, poses, expressions and light stay in each caption.")
        cc_sum = ui.markdown("")
        with ui.row().classes("w-full no-wrap items-start gap-4"):
            cc_html = ui.html("", sanitize=False).classes("grow basis-3/5 min-w-0").props('id="cc_html"')
            with ui.column().classes("grow basis-2/5 gap-1"):
                cc_img = ui.image("").classes("w-full rounded")
                cc_img.set_visibility(False)
                cc_cap = ui.markdown("Click a picture that says something else to look at it with its caption.")
        cc_msg = ui.markdown("")
        cc_ok = ui.checkbox("Replace the locked choices")
        cc_ok.set_visibility(False)

        def cc_draw():
            cc_sum.content, cc_html.content = SP.cc_render(ds)

        async def cc_check():
            cc_msg.content = "Reading the captions …"
            r = await run.io_bound(SP.cc_find, ds, url0, model0, trig.value, None)
            cc_sum.content, cc_html.content = r
            cc_msg.content = ""

        async def cc_apply():
            res = await ui.run_javascript("(" + SP.CC_JS + ")('', '', '', '', '', false)")
            choices = res[1] if isinstance(res, list) and len(res) > 1 else "{}"
            msg, upd = await run.io_bound(SP.cc_apply, ds, choices, url0, model0, trig.value, cc_ok.value, None)
            cc_msg.content = msg
            cc_ok.set_visibility(bool(upd))
            if not upd:
                cc_ok.value = False
            cc_draw()
            sets_html.content = V.s7_sets_html(ds)
            lv = SP.cc_leave_out(ds, leave.value)
            if lv:
                leave.value = lv
                keep_leave()

        def cc_open(path):
            img, cap = SP.cc_open(path)
            if isinstance(img, str):
                cc_img.set_source(pic_url(img, 900))
                cc_img.set_visibility(True)
                cc_cap.content = cap

        def mirror():
            sets = F.caption_sets(ds)
            if len(sets) < 2:
                cc_msg.content = "There is only one caption set."
                return
            cc_msg.content = "\n\n".join(F.mirror_kept(ds, x["ext"], url0, model0, trig.value) for x in sets[1:])
        page.on_cc_open = cc_open
        with ui.row().classes("items-center gap-2"):
            ui.button("Check the captions", on_click=cc_check).props("outline dense no-caps")
            ui.button("Stop", on_click=SP.cc_stop).props("flat dense no-caps color=negative")
            ui.button("Apply choices", on_click=cc_apply).props("unelevated dense no-caps color=primary")
            if len(F.caption_sets(ds)) > 1:
                ui.button("Apply to the other caption set too", on_click=mirror).props("outline dense no-caps")
        cc_draw()
    ui.button("Continue to Review →", on_click=lambda: page.go(step=8)).props("unelevated no-caps color=primary")
    if is_char or worn:
        mv_draw()


# --------------------------------------------------------------------------- 8 Review
def step8(page):
    ds = page.s["ds"]
    head("Review")
    sets = F.caption_sets(ds)
    with ui.row().classes("w-full no-wrap items-start gap-4"):
        with ui.card().props("flat bordered").classes("ls-card w-72 flex-none gap-1"):
            ui.label("The folder").classes("font-semibold")
            tree = ui.html("", sanitize=False)
            ui.label("Checks").classes("font-semibold")
            checks = ui.html("", sanitize=False)
        with ui.column().classes("grow min-w-0 gap-2"):
            show = ui.toggle(["Problems first", "All"], value="Problems first").props("dense no-caps unelevated")
            g = ui.html("", sanitize=False).classes("w-full")
        with ui.card().props("flat bordered").classes("ls-card w-96 flex-none gap-2"):
            img = ui.image("").classes("w-full rounded")
            img.set_visibility(False)
            name = ui.label("Click a picture to see and edit its caption.").classes("ls-note")
            cap1 = ui.textarea("Caption").props("outlined autogrow").classes("w-full")
            cap2 = ui.textarea(f"Caption IMG{sets[1]['ext']}" if len(sets) > 1 else "Second caption set") \
                .props("outlined autogrow").classes("w-full")
            cap2.set_visibility(len(sets) > 1)
            msg = ui.markdown("")
            cur = {"p": ""}

            def save():
                if not cur["p"]:
                    msg.content = "Click a picture first."
                    return
                p = Path(cur["p"])
                F.set_file(p, sets[0]["ext"]).write_text((cap1.value or "").strip() + "\n", encoding="utf-8")
                if len(sets) > 1 and (cap2.value or "").strip():
                    F.set_file(p, sets[1]["ext"]).write_text(cap2.value.strip() + "\n", encoding="utf-8")
                msg.content = f"✔ Saved the caption of {p.name}."
            ui.button("Save caption changes", on_click=save).props("outline dense no-caps")

    def draw():
        tree.content, checks.content, g.content = V.s8_render(ds, show.value)

    def pick(path):
        p = Path(path)
        cur["p"] = str(p)
        img.set_source(pic_url(p, 900))
        img.set_visibility(True)
        name.text = p.name
        cap1.value = F.read_set(p, sets[0]["ext"]) or ""
        cap2.value = (F.read_set(p, sets[1]["ext"]) or "") if len(sets) > 1 else ""
        msg.content = ""

    def fix(arg):
        action, _, path = arg.partition("|")
        msg.content = F.fix(ds, action, path)
        draw()

    def done():
        F.mark_step(ds, "review")
        page.go(step=9)
    page.on_pick, page.on_fix = pick, fix
    show.on_value_change(lambda: draw())
    ui.button("Continue to Train →", on_click=done).props("unelevated no-caps color=primary")
    draw()


# --------------------------------------------------------------------------- 11 LoRA
def step11(page):
    head("LoRA: its LoRA files in the Collection")
    ui.html(V.lora_html(page.s["ds"]), sanitize=False).classes("w-full")
