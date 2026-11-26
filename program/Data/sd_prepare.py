# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Preparing a dataset: captions with Ollama, the picture check and watermarks (Datasets › 6 and 7).

Part of sdgui.py (split by topic): the names of sdgui are reached as S.<name>; sdgui imports
the names of this file back, so callers use sdgui.<name> as before."""
from __future__ import annotations
__version__ = "1.261126b"  # LoRA Studio version, see version.py

import hashlib
import html
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import sdgui as S


# --------------------------------------------------------------------------- review tab
def record_change(desc, files):
    """files: {Path: old_text_or_None}. Kept in memory for Undo and dumped to gui_logs/ as a safety copy."""
    S.LAST_CHANGE["desc"] = desc
    S.LAST_CHANGE["files"] = dict(files)
    try:
        S.LOG_DIR.mkdir(exist_ok=True)
        dump = S.LOG_DIR / f"caption_backup_{datetime.now():%Y%m%d_%H%M%S}.json"
        dump.write_text(json.dumps({"desc": desc, "files": {str(k): v for k, v in files.items()}}))
    except OSError:
        pass


# ---- Recaption with Ollama (Create Data Launch Script tab): natural-language captions from a local vision model
OC_STOP = {"flag": False}


def oc_settings():
    g = S.read_settings().get("general", {})
    return (g.get("OLLAMA_URL") or (S.CAP.DEFAULT_URL if S.CAP else "http://127.0.0.1:11434")), (g.get("OLLAMA_MODEL") or "")


def oc_save_settings(url, model):
    data = S.read_settings()
    sec = data.setdefault("general", {})
    if sec.get("OLLAMA_URL") == url and sec.get("OLLAMA_MODEL") == model:
        return
    sec["OLLAMA_URL"], sec["OLLAMA_MODEL"] = url, model
    try:
        S.write_settings(data)
    except OSError:
        pass


# Caption style per model family: which text encoder reads the captions, and what it was trained on
OC_FAMILY = {
    "sdxl": ("tags", "CLIP-L + CLIP-G", "trained on booru-style tags; reads only the first 75 tokens (about 20 tags)"),
    "sd15": ("tags", "CLIP-L", "trained on tags and short phrases; reads only the first 75 tokens (about 20 tags)"),
    "anima": ("tags", "Qwen3 0.6B", "trained on danbooru tags and on sentences: both work, tags are the usual choice"),
    "flux": ("sentences", "T5-XXL (+ CLIP-L)", "T5 understands sentences and reads up to 512 tokens"),
    "chroma": ("sentences", "T5-XXL", "T5 understands sentences and reads up to 512 tokens"),
    "sd3": ("sentences", "CLIP-L, CLIP-G + T5-XXL", "T5 carries most of the meaning: sentences"),
    "lumina": ("sentences", "Gemma-2", "a language model: sentences"),
    "hunyuan": ("sentences", "Qwen2.5-VL + ByT5", "a language model: sentences"),
    "zimage": ("sentences", "Qwen3 4B", "a language model: sentences, not booru tags"),
    "zimagebase": ("sentences", "Qwen3 4B", "a language model: sentences, not booru tags"),
    "qwenimage": ("sentences", "Qwen2.5-VL 7B", "a language model trained on long descriptions: sentences"),
    "flux2": ("sentences", "Qwen3", "a language model: sentences"),
}


def _oc_family(tkey):
    fam = ""
    try:
        recs = S.load_recs()
        fam = (recs.FAMILY.get(tkey) or recs.FAMILY.get(S._type_base(tkey)) or "") if recs else ""
    except Exception:
        pass
    return fam or tkey or ""


def oc_recommend(tkey, purpose):
    """(markdown, style, length, detail) recommended for this LoRA type and purpose."""
    fam = _oc_family(tkey)
    style, enc, why = OC_FAMILY.get(fam, ("sentences", "its text encoder", "no table entry for this type: sentences are the safe choice"))
    _, presets = S.merged_presets()
    label = ((presets or {}).get(tkey) or {}).get("label", tkey or "this type")
    length, detail = 3, 3
    if purpose == "style":
        detail, length = 4, (3 if style == "tags" else 4)
    note = ""
    if purpose == "style":
        note = ("  \nFor a **style** LoRA the captions describe everything **in** the picture (the people with their "
                "hair, eyes and clothes, the place, the light) and never the style itself: the art style, medium and "
                "technique are in *Never describe*, so the trigger word learns them.")
    elif purpose in ("object", "worn", "action", "concept"):
        what = {"object": "the object", "worn": "the worn item's own look", "action": "the pose or action",
                "concept": "the effect"}[purpose]
        note = (f"  \nFor this LoRA the captions describe everything except {what} itself: the trigger word stands "
                "for it (its words are in *Never describe*).")
    md = (f"**Recommended for {label}: {'Tags' if style == 'tags' else 'Sentences'}**, length "
          f"{S.CAP.LENGTH_LABELS[length] if S.CAP else length}, detail {S.CAP.DETAIL_LABELS[detail] if S.CAP else detail}. "
          f"It reads captions through **{enc}**: {why}.{note}")
    return md, style, length, detail


def oc_example(style, length, detail, trigger, tkey):
    """The invented example caption at these settings, with its size and (for CLIP types) the 75-token limit."""
    if S.CAP is None:
        return ""
    style, length, detail = style or "sentences", int(length or 3), int(detail or 3)
    t = (trigger or "").strip() or "ohwx"
    ex = S.CAP.example(style, length, detail, t)
    n_words, n_tags, tok = len(ex.split()), ex.count(",") + 1, S.CAP.clip_tokens(ex)
    limit = (S.CAP.TAG_COUNT[length], "tags") if style == "tags" else (S.CAP.SENT_WORDS[length], "words")
    size = f"{n_tags} tags" if style == "tags" else f"{n_words} words"
    fam = _oc_family(tkey)
    clip = OC_FAMILY.get(fam, ("", "", ""))[1].startswith("CLIP")
    pct = min(100, round(tok / 75 * 100)) if clip else None
    bar = ""
    if clip:
        cls = "ok" if tok <= 75 else "bad"
        bar = (f"<div class='ocbar'><div class='ocfill {cls}' style='width:{pct}%'></div></div>"
               f"<div class='ocnote {cls}'>about {tok} of the 75 CLIP tokens"
               + ("" if tok <= 75 else ": everything after token 75 is cut off in training (sd-scripts reads one "
                  "75-token chunk unless --max_token_length 150/225 is set)") + "</div>")
    aspects = ", ".join(S.CAP.aspects(detail))
    short = (f"<div class='ocnote'>At {S.CAP.DETAIL_LABELS[detail]} detail this example runs out at {size}: a longer "
             "length adds nothing more here (a real photo with more in it can fill more).</div>"
             if (n_words if style != "tags" else n_tags) < limit[0] * 0.8 and length >= 3 else "")
    return (f"<div class='ocex'><div class='ochead'>Example (an invented photo, not one of yours): "
            f"<b>{S.CAP.LENGTH_LABELS[length]}</b> = up to {limit[0]} {limit[1]} · "
            f"<b>{S.CAP.DETAIL_LABELS[detail]}</b> = {html.escape(aspects)}</div>"
            f"<div class='octext'>{html.escape(ex)}</div><div class='ocnote'>{size}</div>{bar}{short}</div>")


def oc_images(ds):
    """Images of the dataset's repeat folders (the same images the trainers use)."""
    stats = S.with_virtual_row(S.dataset_stats(ds))
    return [im for r in stats["rows"] for im in S.list_images(r["folder"])]


def _oc_check(ds, url, model, trigger):
    if S.CAP is None:
        return "captioner.py is missing next to sdgui.py."
    if not ds:
        return "Pick a LoRA folder first."
    if not model:
        return "Pick a vision model first (press **Find models**)."
    if not (trigger or "").strip() or "," in trigger:
        return "Enter the trigger word (one word, no commas)."
    if S.RUN.running():
        return ("A training run is active. Captioning would share the GPU's 16 GB with it: wait until it has finished "
                "(or press Stop now in the Queue).")
    return ""


def _oc_hint(old: str) -> str:
    return ", ".join(t for t in S.split_tags(old or "")[1:40]) if old else ""


def oc_preview(ds, url, model, trigger, leave_out, hints, style="sentences", length=3, detail=3, n=3):
    """Caption a few images without writing anything."""
    bad = _oc_check(ds, url, model, trigger)
    if bad:
        yield bad
        return
    url = (url or "").strip() or S.CAP.DEFAULT_URL
    oc_save_settings(url, model)
    imgs = oc_images(ds)
    if not imgs:
        yield "No images in this dataset's repeat folders."
        return
    step = max(1, len(imgs) // n)
    pick = imgs[::step][:n]
    out = [f"### Preview: {len(pick)} of {len(imgs)} images, nothing written\n"
           "The first image also loads the model into VRAM, so it takes longer.\n"]
    yield "\n".join(out) + f"\n⏳ {pick[0].name} ..."
    for i, im in enumerate(pick, 1):
        old = S.read_caption(im)
        t0 = time.time()
        try:
            r = S.CAP.caption(url, model, im, trigger.strip(), leave_out, _oc_hint(old) if hints else "",
                              style=style or "sentences", length=int(length or 3), detail=int(detail or 3))
        except S.CAP.OllamaError as e:
            out.append(f"❌ {e}")
            yield "\n".join(out)
            return
        warn = f"  \n⚠️ still mentions: {', '.join(r['leaks'])}" if r["leaks"] else ""
        if r.get("dropped"):
            warn += f"  \n(left out tags about: {', '.join(r['dropped'])})"
        out.append(f"**{im.parent.name}/{im.name}** ({time.time() - t0:.0f} s)  \n"
                   f"old: `{(old or '(none)')[:220]}`  \nnew: **{html.escape(r['caption'])}**{warn}\n")
        yield "\n".join(out) + (f"\n⏳ {pick[i].name} ..." if i < len(pick) else "")
    out.append("Happy with these? Press **Recaption all images**. Not happy: change the leave-out list or try another model.")
    yield "\n".join(out)


DC_CACHE_NAME = "datacheck_cache.json"


def _dc_cache_path() -> Path:
    return (S.CONFIG_DIR or S.LOG_DIR) / DC_CACHE_NAME


def _dc_cache_load() -> dict:
    try:
        return json.loads(_dc_cache_path().read_text(encoding="utf-8"))
    except Exception:                       # noqa: BLE001
        return {}


def _dc_cache_save(c: dict):
    try:
        _dc_cache_path().write_text(json.dumps(c), encoding="utf-8")
    except OSError:
        pass


def _dc_min_side(tkey) -> int:
    _, presets = S.merged_presets()
    side = int(((presets or {}).get(tkey) or {}).get("min_side") or 0)
    if not side:
        try:
            recs = S.load_recs()
            side = int(recs.MIN_SIDE.get(recs.FAMILY.get(tkey) or recs.FAMILY.get(S._type_base(tkey)) or "", 0)) if recs else 0
        except Exception:                   # noqa: BLE001
            side = 0
    return side


# ---- Watermarks, stamps and site names: review the crop options and choose per picture
WM_LABEL = {"A": "Crop A", "B": "Crop B", "R": "Removed", "P": "Painted over by you", "C": "Your crop",
            "aside": "Set aside", "keep": "Keep as is"}


def _pictools_ready() -> bool:
    """Something can fill in marks: the ComfyUI inpaint workflow (while ComfyUI runs) or the picture tools."""
    try:
        return S.DC.can_fill()
    except Exception:                       # noqa: BLE001
        return False


def _filler() -> str:
    try:
        import comfyinpaint as CI
        return "your ComfyUI inpaint workflow" if CI.usable() else "the picture tools"
    except Exception:                                    # noqa: BLE001
        return "the picture tools"


def _wm_paintable(e) -> bool:
    if not e.get("marks"):
        return False                        # a frame / bars only: nothing to paint over, crop it off
    W, H = e.get("W") or 1, e.get("H") or 1
    area = sum((m["box"][2] - m["box"][0]) * (m["box"][3] - m["box"][1]) for m in e.get("marks") or [])
    return area <= 0.2 * W * H


def _wm_recommended(e) -> str:
    if e.get("border") and e.get("options"):
        return "A"                          # a frame or bars: cut them off (the crop also leaves the marks out)
    if _pictools_ready() and _wm_paintable(e):
        return "R"                          # remove the mark, keep the whole picture
    return "A" if e.get("options") else "aside"


def wm_render(ds):
    """(summary markdown, review HTML) from <dataset>/_set_aside/marks.json."""
    if S.DC is None or not ds:
        return "", ""
    root = S.lora_path(ds)
    marks = S.DC.reconcile(root, S.DC.marks_load(root))
    rows = [(rp, e) for rp, e in sorted(marks.items()) if e.get("marks") or e.get("border")]
    if not rows:
        return "", ""
    pending = sum(1 for _, e in rows if not e.get("choice"))
    done = Counter(e.get("choice") for _, e in rows if e.get("choice"))
    md = (f"**{len(rows)} picture{'s' if len(rows) != 1 else ''} with watermarks, logos, site names or a frame.** Red = what "
          "was found, yellow = the picture inside a frame or bars, green = the subject that must stay. Click an option per picture, then press **Apply choices** "
          "(originals are kept in `_set_aside/originals/`, so you can change your mind later). "
          + (f"**{pending} not decided yet** (the recommended option is selected). " if pending else "")
          + (("Applied: " + ", ".join(f"{WM_LABEL.get(k, k)} {v}" for k, v in done.items()) + ".") if done else ""))
    out = []
    for rp, e in rows:
        here = root / rp
        src = S.DC.original_of(root, rp) if e.get("choice") in ("A", "B", "R", "P", "C") else here
        if e.get("choice") == "aside":
            ap = Path(e.get("aside_path") or "")
            src = root / ap if ap and not ap.is_absolute() else ap
        if not src or not Path(src).exists():
            continue
        cur = e.get("choice") or ""
        sel = cur or _wm_recommended(e)
        kinds = "; ".join(f"{m['kind']}" + (f" “{html.escape(m['text'])}”" if m.get("text") else "")
                          + (" <i>(detector)</i>" if m.get("by") == "detector" else "") for m in e.get("marks") or [])
        if e.get("border"):
            kinds = "; ".join(x for x in (kinds, html.escape(e["border"].get("what") or "frame")) if x)
        opts = []
        for o in e.get("options") or []:
            opts.append((o["id"], f"<img src='{S.DC.crop_preview(src, o['box'])}'>", f"<b>Crop {o['id']}</b>"
                         f"<span>{html.escape(o['what'])}<br>{o['size']} px, keeps {o['keep']}%</span>"))
        if not e.get("options"):
            opts.append(("", "<div class='wm-big'>✂︎</div>", "<b>No crop possible</b><span>the marks are too close to the subject</span>"))
        if _pictools_ready() and _wm_paintable(e):
            opts.insert(0, ("R", "<div class='wm-big'>✎</div>", "<b>Remove</b><span>fill in the marks (" + _filler() + "), "
                                                               "keep the whole picture; the mark is saved in "
                                                               "_originals/_watermarks/</span>"))
        opts.append(("aside", "<div class='wm-big'>⇥</div>", "<b>Set aside</b><span>not used for training (with its caption)</span>"))
        opts.append(("keep", "<div class='wm-big'>✓</div>", "<b>Keep as is</b><span>the mark may be learned</span>"))
        cells = []
        for cid, pic, txt in opts:
            if not cid:
                cells.append(f"<div class='wm-opt' style='cursor:default;opacity:.6'>{pic}{txt}</div>")
                continue
            now = " <span class='wm-now'>✓ applied</span>" if cid == cur else ""
            cells.append(f"<button class='wm-opt{' sel' if cid == sel else ''}' data-c='{cid}' onclick=\"this.parentNode."
                         f"querySelectorAll('.wm-opt').forEach(function(x){{x.classList.remove('sel')}});"
                         f"this.classList.add('sel')\">{pic}{txt}{now}</button>")
        q = rp.replace("\\", "\\\\").replace("'", "\\'")
        tools = ((f"<button class='wm-tool' onclick=\"lsSend('wm_tool', 'paint|{html.escape(q)}')\">✎ Paint over it "
                  f"yourself</button>" if _pictools_ready() else "")
                 + f"<button class='wm-tool' onclick=\"lsSend('wm_tool', 'crop|{html.escape(q)}')\">✂ Crop it yourself</button>")
        after = ""
        if cur in ("A", "B", "R", "P", "C") and here.exists():
            after = (f"<div class='wm-after'><span>now · {html.escape(WM_LABEL.get(cur, cur))}</span>"
                     f"<img src='{S.DC.thumb_uri(here, 200)}'></div>")
        out.append(f"<div class='wm-row' data-p='{html.escape(rp)}'><div class='wm-src'>"
                   f"<img src='{S.DC.marked_preview(src, e)}'><div class='wm-name'>{html.escape(rp)}</div>"
                   f"<div class='wm-marks'>{kinds}</div><div class='wm-tools'>{tools}</div>{after}</div>"
                   f"<div class='wm-opts'>{''.join(cells)}</div></div>")
    return md, "".join(out)


def _wm_source(root, rp, e):
    """The untouched picture of a watermark row (the original when a choice is applied)."""
    if e.get("choice") in ("A", "B", "R", "P", "C"):
        o = S.DC.original_of(root, rp)
        if o.exists():
            return o
    if e.get("choice") == "aside":
        ap = Path(e.get("aside_path") or "")
        ap = root / ap if ap and not ap.is_absolute() else ap
        if ap and Path(ap).exists():
            return Path(ap)
    return root / rp


def _wm_box(ds, rp, l, r, t, b, aspect):
    root = S.lora_path(ds)
    e = S.DC.marks_load(root).get(rp) or {}
    src = _wm_source(root, rp, e)
    import datacheck as DC
    W, H = DC.shown_size(src)
    import dsflow as F
    fr = F.fit_box(W, H, l / 100, t / 100, r / 100, b / 100, aspect)
    return src, W, H, fr, [round(fr[0] * W), round(fr[1] * H), round(fr[2] * W), round(fr[3] * H)]


def wm_crop_show(ds, rp, l, r, t, b, aspect):
    if not ds or not rp:
        return None, ""
    import dsflow as F
    src, W, H, fr, px = _wm_box(ds, rp, l, r, t, b, aspect)
    e = S.DC.marks_load(S.lora_path(ds)).get(rp) or {}
    inside = [m for m in e.get("marks") or []
              if not (m["box"][2] <= px[0] or m["box"][0] >= px[2] or m["box"][3] <= px[1] or m["box"][1] >= px[3])]
    note = f"result {px[2] - px[0]}×{px[3] - px[1]}" + (" · ⚠ a found mark is still inside the box" if inside else "")
    return F.crop_preview(str(src), fr), note


def wm_crop_apply(ds, rp, l, r, t, b, aspect):
    """(message, done)."""
    if not ds or not rp:
        return "Pick a picture first.", False
    root = S.lora_path(ds)
    marks = S.DC.marks_load(root)
    e = marks.get(rp)
    if e is None:
        return "This picture is no longer in the watermark list.", False
    _, W, H, fr, px = _wm_box(ds, rp, l, r, t, b, aspect)
    if (px[2] - px[0]) < 64 or (px[3] - px[1]) < 64:
        return "That crop is too small.", False
    err = S.DC._back_to_original(root, rp, e) or S.DC.apply_choice(root, rp, e, "C", box=px)
    S.DC.marks_save(root, marks)
    if err:
        return f"❌ {err}", False
    return f"✔ `{rp}` cropped to {px[2] - px[0]}×{px[3] - px[1]}. The original is in `_set_aside/originals/`.", True


def wm_tool_info(ds, rp) -> tuple:
    """(picture to show, crop box as fractions) for the paint and crop tools of one picture."""
    root = S.lora_path(ds)
    e = S.DC.marks_load(root).get(rp) or {}
    src = _wm_source(root, rp, e)
    import datacheck as DC
    W, H = DC.shown_size(src)
    box = e.get("custom_box") or next((o["box"] for o in e.get("options") or []), None) or [0, 0, W, H]
    return src, [box[0] / W, box[1] / H, box[2] / W, box[3] / H]


def wm_paint_mask(ds, rp, a) -> str:
    """Fill in the painted part (a: the mask, an 'L' picture, white = painted): ComfyUI or the picture tools."""
    if S.RUN.running():
        return "A training run is using the GPU: paint over it when the run has finished."
    if a is None or not a.getbbox():
        return "Paint over the mark first."
    root = S.lora_path(ds)
    marks = S.DC.marks_load(root)
    e = marks.get(rp)
    if e is None:
        return "This picture is no longer in the watermark list."
    mdir = S.CACHE_DIR / "wm_masks"
    mdir.mkdir(parents=True, exist_ok=True)
    mp = mdir / (hashlib.md5(f"{root}|{rp}".encode()).hexdigest() + ".png")
    a.save(mp)
    err = S.DC._back_to_original(root, rp, e) or S.DC.apply_choice(root, rp, e, "P", mask=mp)
    S.DC.marks_save(root, marks)
    if err:
        return f"❌ {err}"
    return (f"✔ The painted part of {rp} is filled in (see now in its row). The original is in "
            "_set_aside/originals/, the cut-out in _originals/_watermarks/.")


def wm_apply(ds, choices_json, only_pending=False):
    """Apply the options clicked in the review (or the recommended ones). Returns a message."""
    g = wm_apply_steps(ds, choices_json, only_pending)
    while True:
        try:
            next(g)
        except StopIteration as e:
            return e.value


def wm_apply_live(ds, choices_json, only_pending=False):
    """For the buttons: progress lines while the pictures are changed, then the message."""
    g = wm_apply_steps(ds, choices_json, only_pending)
    while True:
        try:
            yield next(g)
        except StopIteration as e:
            yield e.value
            return


def wm_apply_steps(ds, choices_json, only_pending=False):
    """Generator behind wm_apply: yields a progress line before each picture; returns the message."""
    if S.DC is None or not ds:
        return "Pick a LoRA folder first."
    root = S.lora_path(ds)
    if S.RUN.running() and S.script_home(S.RUN.script or "/") == root:
        return "A training run of this dataset is active: change the pictures when it has finished."
    marks = S.DC.reconcile(root, S.DC.marks_load(root))
    try:
        want = json.loads(choices_json or "{}")
    except ValueError:
        want = {}
    count, errs = Counter(), []
    for rp, e in marks.items():
        if not e.get("marks") and not e.get("border"):
            continue
        if only_pending:
            if e.get("choice"):
                continue
            c = _wm_recommended(e)
        else:
            c = want.get(rp)
            if not c:
                continue
        if c == (e.get("choice") or ""):
            continue
        yield f"⏳ {WM_LABEL.get(c, c)}: {rp}" + (f" ({_filler()} fills in the marks)" if c == "R" else "")
        err = S.DC.apply_choice(root, rp, e, c)
        if err and c == "R" and only_pending:          # removal did not work: the next best choice
            c = "A" if e.get("options") else "aside"
            err = S.DC.apply_choice(root, rp, e, c)
        if err:
            errs.append(err)
        else:
            count[c] += 1
    S.DC.marks_save(root, marks)
    if not count and not errs:
        return "Nothing changed."
    return ("✔ " + ", ".join(f"{WM_LABEL[k]}: {v}" for k, v in count.items()) + ". The step plan above is updated."
            + ("\n\nProblems: " + "; ".join(errs[:5]) if errs else ""))


def rv_undo():
    files = S.LAST_CHANGE["files"]
    if not files:
        return "Nothing to undo."
    n = 0
    for p, old in files.items():
        try:
            if old is None:
                p.unlink(missing_ok=True)
            else:
                p.write_text(old, encoding="utf-8")
            n += 1
        except OSError:
            pass
    desc = S.LAST_CHANGE["desc"]
    S.LAST_CHANGE["files"] = {}
    return f"Undid '{desc}': restored {n} file(s)."
