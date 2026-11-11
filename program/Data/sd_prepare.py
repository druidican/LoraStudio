# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Preparing a dataset: captions with Ollama, the picture check, watermarks, the automatic preparation and the
consistency check (Datasets › 6 and 7).

Part of sdgui.py (split by topic): the names of sdgui are reached as S.<name>; sdgui imports
the names of this file back, so callers use sdgui.<name> as before."""
from __future__ import annotations
__version__ = "1.111126e"  # LoRA Studio version, see version.py

import copy
import hashlib
import html
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
import gradio as gr

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


def oc_apply_recommendation(tkey, purpose):
    md, style, length, detail = oc_recommend(tkey, purpose)
    return md, gr.update(value=style), gr.update(value=length), gr.update(value=detail)


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
                "(or press Stop now in Training › Run).")
    return ""


def _oc_hint(old: str) -> str:
    return ", ".join(t for t in S.split_tags(old or "")[1:40]) if old else ""


def _locked_leave_out(ds, leave_out):
    """The leave-out list plus the traits locked as kept in the LoRA for this dataset (traits.txt)."""
    if S.CC is None or not ds:
        return leave_out
    try:
        kept = S.CC.kept_traits(S.lora_path(ds))
    except Exception:                       # noqa: BLE001
        return leave_out
    have = {t.strip().lower() for t in (leave_out or "").split(",")}
    add = [t for t in kept if t.lower() not in have]
    return ", ".join([x for x in [leave_out] if x] + add)


def oc_preview(ds, url, model, trigger, leave_out, hints, style="sentences", length=3, detail=3, n=3):
    """Caption a few images without writing anything."""
    leave_out = _locked_leave_out(ds, leave_out)
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
        if S.CC is not None:
            r["caption"] = S.CC.lock_caption(S.lora_path(ds), im, r["caption"])
        out.append(f"**{im.parent.name}/{im.name}** ({time.time() - t0:.0f} s)  \n"
                   f"old: `{(old or '(none)')[:220]}`  \nnew: **{html.escape(r['caption'])}**{warn}\n")
        yield "\n".join(out) + (f"\n⏳ {pick[i].name} ..." if i < len(pick) else "")
    out.append("Happy with these? Press **Recaption all images**. Not happy: change the leave-out list or try another model.")
    yield "\n".join(out)


def oc_run(ds, url, model, trigger, leave_out, hints, unload_after, style="sentences", length=3, detail=3,
           only_missing=False):
    """Rewrite every caption of the dataset. Old captions: Undo button + safety copy in gui_logs/."""
    leave_out = _locked_leave_out(ds, leave_out)
    bad = _oc_check(ds, url, model, trigger)
    if bad:
        yield bad
        return
    url = (url or "").strip() or S.CAP.DEFAULT_URL
    oc_save_settings(url, model)
    imgs = oc_images(ds)
    if only_missing:
        imgs = [im for im in imgs if not S.read_caption(im)]
    if not imgs:
        yield "No images in this dataset's repeat folders." if not only_missing else "Every picture already has a caption."
        return
    olds = {im: S.read_caption(im) for im in imgs}
    record_change(f"recaption {ds} with {model}",
                  {S.caption_file(im): (S.caption_file(im).read_text(encoding="utf-8", errors="ignore")
                                      if S.caption_file(im).is_file() else None) for im in imgs})
    OC_STOP["flag"] = False
    done, leaky, failed, t_start = 0, [], [], time.time()
    for i, im in enumerate(imgs, 1):
        if OC_STOP["flag"]:
            break
        el = time.time() - t_start
        eta = f", about {S.fmt(el / max(done, 1) * (len(imgs) - done))} left" if done else ""
        yield f"⏳ {i}/{len(imgs)}: {im.parent.name}/{im.name}{eta}"
        try:
            r = S.CAP.caption(url, model, im, trigger.strip(), leave_out, _oc_hint(olds[im]) if hints else "",
                          style=style or "sentences", length=int(length or 3), detail=int(detail or 3))
        except S.CAP.OllamaError as e:
            failed.append(f"{im.name}: {e}")
            if "not reachable" in str(e):
                break
            continue
        cap = S.CC.lock_caption(S.lora_path(ds), im, r["caption"]) if S.CC is not None else r["caption"]
        S.caption_file(im).write_text(cap + "\n", encoding="utf-8")
        done += 1
        if r["leaks"]:
            leaky.append(f"{im.parent.name}/{im.name}: {', '.join(r['leaks'])}")
    if unload_after:
        S.CAP.unload(url, model)
    L = [f"{'■ Stopped' if OC_STOP['flag'] else '✔ Done'}: {done} of {len(imgs)} captions rewritten in {S.fmt(time.time() - t_start)}"
         + (" (model unloaded from VRAM)." if unload_after else "."),
         "**Undo last change** restores all the old captions (a safety copy is in gui_logs/)."]
    if leaky:
        L.append(f"\n⚠️ {len(leaky)} caption(s) still mention a left-out trait; check and edit them by hand:")
        L += [f"- {x}" for x in leaky[:30]]
    if failed:
        L.append(f"\n❌ {len(failed)} image(s) failed:")
        L += [f"- {x}" for x in failed[:10]]
    yield "\n".join(L)


# ---- Check dataset pictures (datacheck.py): technical checks, Ollama content pass, balance; set aside / back
DC_STOP = {"flag": False}
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


def _dc_report(results, purpose, notes, elapsed, ran):
    """(summary markdown, HTML table, checkbox update)."""
    rows = []
    for p, r in results.items():
        sev = "bad" if any(s == S.DC.BAD for s, _ in r["issues"]) else ("warn" if r["issues"] else "ok")
        rows.append((p, sev, r))
    bad = [x for x in rows if x[1] == "bad"]
    warn = [x for x in rows if x[1] == "warn"]
    L = [f"**{len(rows)} pictures checked** ({', '.join(ran)}; {S.fmt(elapsed)}): **{len(bad)} not recommended**, "
         f"{len(warn)} with a warning, {len(rows) - len(bad) - len(warn)} fine. Purpose: **{purpose}**."]
    if notes:
        L.append("\n**Dataset balance**")
        L += [f"- {n}" for n in notes]
    if bad:
        L.append("\nThe *not recommended* pictures are ticked below. Check them, untick what you want to keep, "
                 "and press **Move ticked to _set_aside** (captions move along; no trainer reads that folder).")
    cells = []
    for p, sev, r in sorted(rows, key=lambda x: (x[1] != "bad", x[1] != "warn", x[0].name)):
        if sev == "ok":
            continue
        uri = S.DC.thumb_uri(p)
        issues = "".join(f"<li class='dc{s}'>{html.escape(t)}</li>" for s, t in r["issues"])
        cells.append(f"<tr class='dcrow-{sev}'><td><img src='{uri}' loading='lazy'></td>"
                     f"<td><b>{html.escape(p.parent.name)}/{html.escape(p.name)}</b>"
                     f"<span class='dctag dctag-{sev}'>{'not recommended' if sev == 'bad' else 'warning'}</span>"
                     f"<ul>{issues}</ul></td></tr>")
    table = ("<table class='dctable'>" + "".join(cells) + "</table>") if cells else "<p>No problems found.</p>"
    choices = [(f"{p.parent.name}/{p.name} — {'; '.join(t for _, t in r['issues'])[:140]}", str(p))
               for p, sev, r in sorted(rows, key=lambda x: (x[1] != "bad", x[0].name)) if sev != "ok"]
    return "\n".join(L), table, gr.update(choices=choices, value=[str(p) for p, sev, _ in bad])


def dc_run(ds, tkey, purpose, checks, url, model):
    """Generator: (summary, table html, checkbox update)."""
    none = gr.update()
    if S.DC is None:
        yield "datacheck.py is missing next to sdgui.py.", "", none
        return
    if not ds:
        yield "Pick a LoRA folder first.", "", none
        return
    checks = checks or []
    imgs = oc_images(ds)
    if not imgs:
        yield "No pictures in this dataset's repeat folders.", "", none
        return
    purpose = purpose or "character"
    DC_STOP["flag"] = False
    t0, ran = time.time(), []
    results = {p: {"issues": [], "content": None} for p in imgs}
    if "technical" in checks:
        side = _dc_min_side(tkey)
        yield f"⏳ Technical checks on {len(imgs)} pictures ...", "", none
        tech = S.DC.technical(imgs, side)
        for p, r in tech.items():
            results[p]["issues"] += r["issues"]
        ran.append("technical" + (f", {side} px" if side else ""))
    use_ai = "content" in checks or "balance" in checks
    if use_ai:
        model = model or oc_settings()[1]
        url = (url or "").strip() or oc_settings()[0]
        if S.CAP is None or not model:
            results_note = "Ollama pass skipped: pick a vision model in *Recaption with Ollama* above (Find models)."
            yield results_note, "", none
            use_ai = False
        elif S.RUN.running():
            yield "Ollama pass skipped: a training run is active (it would share the GPU).", "", none
            use_ai = False
    root = S.lora_path(ds)
    marks = S.DC.reconcile(root, S.DC.marks_load(root)) if use_ai else {}
    crop_min = _dc_min_side(tkey) or 512
    n_marks = 0
    if use_ai:
        cache, done = _dc_cache_load(), 0
        for i, p in enumerate(imgs, 1):
            if DC_STOP["flag"]:
                break
            st = p.stat()
            key = f"{p}|{st.st_size}|{int(st.st_mtime)}|{model}"
            d = cache.get(key)
            if d is None:
                eta = f", about {S.fmt((time.time() - t0) / max(done, 1) * (len(imgs) - i + 1))} left" if done else ""
                yield f"⏳ Ollama {i}/{len(imgs)}: {p.parent.name}/{p.name}{eta}", "", none
                try:
                    d = S.DC.content(url, model, p)
                except Exception as e:      # noqa: BLE001
                    _dc_cache_save(cache)
                    if marks:
                        S.DC.marks_save(root, marks)          # keep what was found so far
                    yield f"❌ Ollama: {e}", "", none
                    return
                cache[key] = d
                done += 1
                if done % 5 == 0:
                    _dc_cache_save(cache)
            results[p]["content"] = d
            if "content" in checks:
                issues = S.DC.content_issues(d, purpose)
                if S.DC._true(d.get("text_or_watermark")):
                    rp = S.DC.rel(root, p)
                    e = marks.get(rp)
                    sig = f"{st.st_size}|{int(st.st_mtime)}"
                    if not (e and e.get("choice")) and (not e or e.get("sig") != sig):
                        yield f"⏳ Ollama {i}/{len(imgs)}: where are the watermarks in {p.parent.name}/{p.name}?", "", none
                        try:
                            f = S.DC.find_marks(url, model, p)
                            e = {**f, "sig": sig, "choice": "",
                                 "options": S.DC.crop_options(f["W"], f["H"], f["marks"], f["subject"], crop_min)}
                            marks[rp] = e
                        except Exception as ex:           # noqa: BLE001
                            e = None
                            print(f"[check] watermark search failed for {p.name}: {ex!r}")
                    if e is not None:
                        issues = [x for x in issues if "watermark" not in x[1]]
                        if e.get("marks"):
                            n_marks += 1
                            kinds = ", ".join(sorted({m["kind"] for m in e["marks"]}))
                            if e.get("options"):
                                issues.append((S.DC.WARN, f"{kinds} on the picture: can be cropped off (see Watermarks below)"))
                            else:
                                issues.append((S.DC.BAD if purpose == "style" else S.DC.WARN,
                                               f"{kinds} over the picture: no crop keeps the subject (see Watermarks below)"))
                results[p]["issues"] += issues
        _dc_cache_save(cache)
        if marks:
            S.DC.marks_save(root, marks)
        ran.append(f"content with {model}" + (" (stopped)" if DC_STOP["flag"] else ""))
        if n_marks:
            ran.append(f"{n_marks} with watermarks or stamps")
    notes = S.DC.balance(results, purpose) if "balance" in checks and use_ai else []
    yield _dc_report(results, purpose, notes, time.time() - t0, ran or ["nothing"])


def dc_stop():
    DC_STOP["flag"] = True
    return "Stopping after the current picture..."


def dc_set_aside(ds, picked):
    if S.DC is None or not ds:
        return "Pick a LoRA folder first."
    if not picked:
        return "Nothing ticked."
    moved, err = wm_set_aside(S.lora_path(ds), [Path(p) for p in picked])
    left = len(S.DC.set_aside_list(S.lora_path(ds)))
    return (f"✔ {moved} picture(s) and their captions moved to `{S.lora_path(ds) / S.DC.SET_ASIDE}` ({left} there now). "
            "The step plan above is updated. **Move all back** restores them."
            + ("\n\nCould not move: " + "; ".join(err[:5]) if err else ""))


def wm_set_aside(root: Path, pics: list[Path]):
    """Set aside; a picture with a watermark entry goes through it (its uncropped original is what is set aside)."""
    marks = S.DC.reconcile(root, S.DC.marks_load(root))
    plain, moved, err = [], 0, []
    for p in pics:
        rp = S.DC.rel(root, p)
        if rp in marks:
            e2 = S.DC.apply_choice(root, rp, marks[rp], "aside")
            if e2:
                err.append(e2)
            else:
                moved += 1
        else:
            plain.append(p)
    if marks:
        S.DC.marks_save(root, marks)
    m2, e3 = S.DC.set_aside(root, plain)
    return moved + m2, err + e3


# ---- Watermarks, stamps and site names: review the crop options and choose per picture
WM_LABEL = {"A": "Crop A", "B": "Crop B", "R": "Removed", "P": "Painted over by you", "C": "Your crop",
            "aside": "Set aside", "keep": "Keep as is"}


def _pictools_ready() -> bool:
    try:
        import pictools
        return pictools.ready()
    except Exception:                       # noqa: BLE001
        return False


def _wm_paintable(e) -> bool:
    W, H = e.get("W") or 1, e.get("H") or 1
    area = sum((m["box"][2] - m["box"][0]) * (m["box"][3] - m["box"][1]) for m in e.get("marks") or [])
    return area <= 0.2 * W * H


def _wm_recommended(e) -> str:
    if _pictools_ready() and _wm_paintable(e):
        return "R"                          # remove the mark, keep the whole picture
    return "A" if e.get("options") else "aside"


def wm_render(ds):
    """(summary markdown, review HTML) from <dataset>/_set_aside/marks.json."""
    if S.DC is None or not ds:
        return "", ""
    root = S.lora_path(ds)
    marks = S.DC.reconcile(root, S.DC.marks_load(root))
    rows = [(rp, e) for rp, e in sorted(marks.items()) if e.get("marks")]
    if not rows:
        return "", ""
    pending = sum(1 for _, e in rows if not e.get("choice"))
    done = Counter(e.get("choice") for _, e in rows if e.get("choice"))
    md = (f"**{len(rows)} picture{'s' if len(rows) != 1 else ''} with watermarks, stamps or site names.** Red = what was "
          "found, green = the subject that must stay. Click an option per picture, then press **Apply choices** "
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
        kinds = "; ".join(f"{m['kind']}" + (f" “{html.escape(m['text'])}”" if m.get("text") else "") for m in e["marks"])
        opts = []
        for o in e.get("options") or []:
            opts.append((o["id"], f"<img src='{S.DC.crop_preview(src, o['box'])}'>", f"<b>Crop {o['id']}</b>"
                         f"<span>{html.escape(o['what'])}<br>{o['size']} px, keeps {o['keep']}%</span>"))
        if not e.get("options"):
            opts.append(("", "<div class='wm-big'>✂︎</div>", "<b>No crop possible</b><span>the marks are too close to the subject</span>"))
        if _pictools_ready() and _wm_paintable(e):
            opts.insert(0, ("R", "<div class='wm-big'>✎</div>", "<b>Remove</b><span>paint over the marks, keep the whole "
                                                               "picture; the mark is saved in _originals/_watermarks/</span>"))
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


def wm_tool_open(ds, v):
    """Open the paint or crop panel for one picture (lsSend('wm_tool', 'paint|rp') from the review)."""
    keep = gr.update()
    act, _, rest = (v or "").partition("|")
    rp = rest.rsplit("|", 1)[0] if "|" in rest else rest
    if not ds or act not in ("paint", "crop") or not rp:
        return [keep] * 15
    root = S.lora_path(ds)
    marks = S.DC.marks_load(root)
    e = marks.get(rp) or {}
    src = _wm_source(root, rp, e)
    from PIL import Image, ImageOps
    try:
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((1600, 1600))
            im.load()
    except Exception as ex:                     # noqa: BLE001
        return [keep] * 13 + [f"Could not open the picture: {ex}", keep]
    title = f"**{'Paint over the mark' if act == 'paint' else 'Crop it yourself'}:** `{rp}`"
    if act == "paint":
        return [gr.update(visible=True), title, gr.update(visible=True), gr.update(visible=False),
                {"background": im, "layers": [], "composite": im}, keep, keep, keep, keep, keep,
                gr.update(visible=True), gr.update(visible=False), rp, "", ""]
    with Image.open(src) as o:
        W, H = ImageOps.exif_transpose(o).size
    box = e.get("custom_box") or next((o["box"] for o in e.get("options") or []), None) or [0, 0, W, H]
    fr = [box[0] / W, box[1] / H, box[2] / W, box[3] / H]
    import dsflow as F
    return [gr.update(visible=True), title, gr.update(visible=False), gr.update(visible=True), keep,
            F.crop_preview(str(src), fr), round(fr[0] * 100), round(fr[2] * 100), round(fr[1] * 100), round(fr[3] * 100),
            gr.update(visible=False), gr.update(visible=True), rp, "", ""]


def _wm_box(ds, rp, l, r, t, b, aspect):
    root = S.lora_path(ds)
    e = S.DC.marks_load(root).get(rp) or {}
    src = _wm_source(root, rp, e)
    from PIL import Image, ImageOps
    with Image.open(src) as im:
        W, H = ImageOps.exif_transpose(im).size
    import dsflow as F
    fr = F.fit_box(W, H, l / 100, t / 100, r / 100, b / 100, aspect)
    return src, W, H, fr, [round(fr[0] * W), round(fr[1] * H), round(fr[2] * W), round(fr[3] * H)]


def wm_crop_show(ds, rp, l, r, t, b, aspect):
    if not ds or not rp:
        return gr.update(), ""
    import dsflow as F
    src, W, H, fr, px = _wm_box(ds, rp, l, r, t, b, aspect)
    e = S.DC.marks_load(S.lora_path(ds)).get(rp) or {}
    inside = [m for m in e.get("marks") or []
              if not (m["box"][2] <= px[0] or m["box"][0] >= px[2] or m["box"][3] <= px[1] or m["box"][1] >= px[3])]
    note = f"result {px[2] - px[0]}×{px[3] - px[1]}" + (" · ⚠ a found mark is still inside the box" if inside else "")
    return F.crop_preview(str(src), fr), note


def wm_crop_apply(ds, rp, l, r, t, b, aspect):
    if not ds or not rp:
        return "Pick a picture first.", gr.update()
    root = S.lora_path(ds)
    marks = S.DC.marks_load(root)
    e = marks.get(rp)
    if e is None:
        return "This picture is no longer in the watermark list.", gr.update()
    _, W, H, fr, px = _wm_box(ds, rp, l, r, t, b, aspect)
    if (px[2] - px[0]) < 64 or (px[3] - px[1]) < 64:
        return "That crop is too small.", gr.update()
    err = S.DC._back_to_original(root, rp, e) or S.DC.apply_choice(root, rp, e, "C", box=px)
    S.DC.marks_save(root, marks)
    if err:
        return f"❌ {err}", gr.update()
    return f"✔ `{rp}` cropped to {px[2] - px[0]}×{px[3] - px[1]}. The original is in `_set_aside/originals/`.", gr.update(visible=False)


def wm_paint_apply(ds, rp, ed):
    if not ds or not rp:
        return "Pick a picture first.", gr.update()
    if S.RUN.running():
        return "A training run is using the GPU: paint over it when the run has finished.", gr.update()
    layers = (ed or {}).get("layers") or []
    if not layers:
        return "Paint over the mark first.", gr.update()
    from PIL import ImageChops
    a = None
    for lay in layers:
        if lay is None:
            continue
        al = lay.convert("RGBA").getchannel("A")
        a = al if a is None else ImageChops.lighter(a, al)
    if a is None or not a.getbbox():
        return "Paint over the mark first.", gr.update()
    root = S.lora_path(ds)
    marks = S.DC.marks_load(root)
    e = marks.get(rp)
    if e is None:
        return "This picture is no longer in the watermark list.", gr.update()
    mdir = S.CACHE_DIR / "wm_masks"
    mdir.mkdir(parents=True, exist_ok=True)
    mp = mdir / (hashlib.md5(f"{root}|{rp}".encode()).hexdigest() + ".png")
    a.save(mp)
    err = S.DC._back_to_original(root, rp, e) or S.DC.apply_choice(root, rp, e, "P", mask=mp)
    S.DC.marks_save(root, marks)
    if err:
        return f"❌ {err}", gr.update()
    return (f"✔ The painted part of `{rp}` is filled in (see *now* in its row). The original is in "
            "`_set_aside/originals/`, the cut-out in `_originals/_watermarks/`.", gr.update(visible=False))


def wm_apply(ds, choices_json, only_pending=False, prefer="remove"):
    """Apply the options clicked in the review (or the recommended ones). Returns a message."""
    g = wm_apply_steps(ds, choices_json, only_pending, prefer)
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


def wm_apply_steps(ds, choices_json, only_pending=False, prefer="remove"):
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
        if not e.get("marks"):
            continue
        if only_pending:
            if e.get("choice"):
                continue
            c = _wm_recommended(e)
            if prefer == "crop":                        # Settings › Automatic improvement
                c = "A" if e.get("options") else "aside"
            elif prefer == "aside":
                c = "aside"
        else:
            c = want.get(rp)
            if not c:
                continue
        if c == (e.get("choice") or ""):
            continue
        yield f"⏳ {WM_LABEL.get(c, c)}: {rp}" + (" (the picture tools paint over the marks)" if c == "R" else "")
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


def dc_back(ds):
    if S.DC is None or not ds:
        return "Pick a LoRA folder first."
    n = len(S.DC.set_aside_list(S.lora_path(ds)))
    if not n:
        return "No set-aside pictures in this dataset."
    moved, err = S.DC.move_back(S.lora_path(ds))
    return f"✔ {moved} picture(s) moved back to their repeat folders." + ("\n\n" + "; ".join(err[:5]) if err else "")


def oc_stop():
    OC_STOP["flag"] = True
    return "Stopping after the current image..."


def rv_undo():
    files = S.LAST_CHANGE["files"]
    if not files:
        gr.Info("Nothing to undo.")
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



CC_CORE = ("hair colour", "hair length", "eye colour", "skin tone", "build", "horns", "ears", "tail", "wings", "fur")


def _cc_thumb(p: str, side: int = 150) -> str:
    return S.DC.thumb_uri(Path(p), side) if S.DC is not None else ""


def _cc_recommend(g, total) -> str:
    if g["trait"] in CC_CORE or any(w in g["trait"] for w in ("hair", "eye", "skin", "build", "horn")):
        return "lora"
    return "lora" if total and g["mentioned"] >= 0.3 * total else "each"


def _cc_explain(g, total, rec) -> tuple[str, dict]:
    """(why this choice is recommended, {choice: what it means for the LoRA}) for one trait."""
    trait = g["trait"]
    vals = g["values"]
    main = " / ".join(vals[0]["words"]) if vals else trait
    n_main = len(vals[0]["pics"]) if vals else 0
    men = max(1, g["mentioned"])
    core = trait in CC_CORE or any(w in trait for w in ("hair", "eye", "skin", "build", "horn"))
    if len(vals) < 2:
        why = (f"Every caption that mentions it says <b>{html.escape(main)}</b> ({men} of {total}): it is always the same, "
               "so it is part of the character.")
    elif rec.startswith("lora") and n_main >= 0.7 * men:
        share = round(100 * n_main / men)
        why = (f"<b>{html.escape(main)}</b> in {n_main} of the {men} captions that mention it ({share}%)"
               + (f"; {html.escape(trait)} is something the character always has" if core else "")
               + ": most likely part of the character, and the others are mistakes or odd pictures.")
    elif rec.startswith("lora"):
        split = " / ".join(f"{html.escape(' / '.join(v['words']))} {len(v['pics'])}" for v in vals[:4])
        why = (f"Split ({split}), but {html.escape(trait)} is something a character usually always has: one of them is "
               "probably wrong. Look at the pictures below before you choose; pick <i>This picture only</i> if the "
               "character really changes it.")
    else:
        split = ", ".join(f"{html.escape(' / '.join(v['words']))} {len(v['pics'])}" for v in vals[:4])
        why = (f"Split ({split}): it changes between the pictures (an outfit, a wig, the light), so it is probably not "
               "part of the character itself.")
    means = {
        "lora": (f"The trigger word always gives <b>{html.escape(main)}</b>, without writing it in a prompt. Asking for "
                 f"another {html.escape(trait)} in a prompt works against the LoRA."),
        "each": (f"{html.escape(trait.capitalize())} is not fixed in the LoRA: you write it in your prompt, and each of "
                 "these values works."),
        "skip": ("The captions stay as they are: the LoRA learns this trait half-way (sometimes in the trigger word, "
                 "sometimes in the words)."),
    }
    return why, means


def cc_state(ds):
    """(state, text) of step 5 for pp_status."""
    if S.CC is None or not ds:
        return "", ""
    res = S.CC.load_result(ds)
    if not res:
        return "to do", "not checked yet"
    imgs = oc_images(ds)
    now = S.CC.captions_key({str(p): S.read_caption(p) for p in imgs if S.read_caption(p)})
    if res.get("applied") == now:
        return "done", f"applied {res.get('applied_at', '')}"
    if res.get("captions") != now:
        return "to do", "captions changed since the last check"
    n = sum(1 for g in res["groups"] if g["kind"] == "contradiction")
    return ("decide" if res["groups"] else "done"), (f"{n} contradiction{'s' if n != 1 else ''}, {len(res['groups'])} trait{'s' if len(res['groups']) != 1 else ''} to decide"
                                                     if res["groups"] else "no character traits in the captions")


def cc_render(ds):
    """(summary markdown, review HTML) of the last consistency check of a dataset."""
    if S.CC is None or not ds:
        return "", ""
    res = S.CC.load_result(ds)
    if not res:
        return ("Press **Check the captions** after the captions are written. Only traits of the character itself "
                "are listed (hair, eyes, build, skin, horns, tattoos, ...), not clothes, places or poses."), ""
    total = res["total"]
    contra = [g for g in res["groups"] if g["kind"] == "contradiction"]
    md = (f"Checked **{total} captions** ({res['at']}, {res['seconds']} s): **{len(contra)} trait"
          f"{'s' if len(contra) != 1 else ''} with contradictions**, {len(res['groups']) - len(contra)} the same "
          "wherever they are mentioned. For each trait pick **Keep in the LoRA** (removed from the captions so the trigger "
          "word learns it) or **This picture only** (each caption keeps its own; words that mean the same are merged into "
          "the word you pick). Pictures that say something else are left as they are until you look at them (click) and "
          "choose **Fix**. Then press **Apply choices** (Undo in step 4 restores the captions).")
    if res.get("lexicon"):
        md += ("  \n⚠ Ollama could not be reached" + (f" ({html.escape(res['error'][:120])})" if res.get("error") else "")
               + ": only the common traits were found with the built-in word list.")
    if res.get("applied") and res.get("applied") == res.get("captions_after"):
        md += f"  \n✓ Applied {res.get('applied_at', '')}."
    imgs = oc_images(ds)
    now = S.CC.captions_key({str(p): S.read_caption(p) for p in imgs if S.read_caption(p)})
    if now not in (res.get("captions"), res.get("applied")):
        md += "  \n⚠ The captions changed since this check: press **Check the captions** again."
    if not res["groups"]:
        return md, "<p class='cc-none'>No traits of the character were found in the captions. Nothing to decide.</p>"
    root = S.lora_path(ds)
    res = S.CC.with_locks(res, root)
    nlock = sum(1 for g in res["groups"] if g.get("lock"))
    if nlock and S.CC.LOCKS_ON:
        md += (f"  \n🔒 **{nlock} trait{'s' if nlock != 1 else ''} locked** in `{S.CC.LOCK_NAME}` (in the dataset folder): "
               "they come first, here and when new captions are written in step 4.")
    out = []
    for g in res["groups"]:
        gid, lk = g["id"], g.get("lock")
        rec = _cc_recommend(g, total)
        sel = ("lora:0" if lk["mode"] == "keep" else "each") if lk else ("lora:0" if rec == "lora" else rec)
        if lk and lk["mode"] == "keep" and not g.get("lock_found"):
            sel = "skip"                    # the locked value is not in these captions: nothing is chosen for you
        exc = set(g.get("exceptions") or [])
        vals = []
        for i, v in enumerate(g["values"]):
            words = v["words"]
            if len(words) > 1:
                locked_w = next((w for w, o in ((lk or {}).get("use") or {}).items() if w in words), "")
                opts = "".join(f"<option value='{html.escape(w)}'{' selected' if w == locked_w else ''}>"
                               f"{html.escape(w)} ({v['count'][w]})</option>" for w in words)
                word = (f"<select class='cc-word' data-g='{gid}' data-i='{i}' title='The word to use for all of these "
                        f"(This picture only)'>{opts}</select>")
            else:
                word = f"<b>{html.escape(words[0])}</b>"
            vals.append(f"<span class='cc-val{' cc-main' if i == 0 else ''}'>{word} <small>{len(v['pics'])} "
                        f"picture{'s' if len(v['pics']) != 1 else ''}</small></span>")
        kind = (f"<span class='cc-kind cc-contra'>{len(g['values'])} different</span>" if g["kind"] == "contradiction"
                else "<span class='cc-kind cc-same'>the same</span>")
        if lk:
            kind += ("<span class='cc-kind cc-lock' title='Locked in traits.txt'>🔒 locked: "
                     + (f"kept in the LoRA ({html.escape(lk.get('value', ''))})" if lk["mode"] == "keep" else "this picture only")
                     + "</span>")
        why, means = _cc_explain(g, total, sel)
        radios = []
        for i, v in enumerate(g["values"][:4]):
            lab = f"Keep in the LoRA: <b>{html.escape(' / '.join(v['words']))}</b>"
            if i == 0:
                lab += " <small>(removed from the captions)</small>"
            if i == 0:
                lab += f"<span class='cc-means'>{means['lora']}</span>"
            radios.append((f"lora:{i}", lab))
        radios += [("each", "This picture only <small>(each caption keeps its own)</small>"
                            f"<span class='cc-means'>{means['each']}</span>"),
                   ("skip", f"Leave as it is<span class='cc-means'>{means['skip']}</span>")]
        rad = []
        for val, lab in radios:
            tag = (" <i>locked</i>" if lk and val == sel else (" <i>recommended</i>" if not lk and val == sel else ""))
            rad.append(f"<label><input type='radio' name='cc-m-{gid}' value='{val}'{' checked' if val == sel else ''} "
                       f"onchange='window.ccMode && ccMode(this)'> {lab}{tag}</label>")
        main_i = int(sel.split(":")[1]) if sel.startswith("lora:") else -1
        odd = []
        for vi, v in enumerate(g["values"]):
            if len(g["values"]) < 2:
                break
            for p, phrase in v["pics"].items():
                if p in exc or any(p in w["pics"] for j, w in enumerate(g["values"]) if j != vi and j < vi):
                    continue
                cap = html.escape(phrase or " / ".join(v["words"]))
                nm = f"cc-o-{gid}-{abs(hash(p)) % 10**9}"
                hide = " style='display:none'" if (main_i < 0 or vi == main_i) else ""
                odd.append(f"<div class='cc-odd' data-p='{html.escape(p)}' data-v='{vi}'{hide}>"
                           f"<button class='cc-look' title='Look at it' onclick=\"window.ccOpen && ccOpen(this)\">"
                           f"<img src='{_cc_thumb(p)}'></button><div class='cc-oddtxt'><span class='cc-name'>"
                           f"{html.escape(Path(p).name)}</span><span>says <b>{cap}</b></span>"
                           f"<label><input type='radio' name='{nm}' value='fix'> Fix <small>(remove it too)</small></label>"
                           f"<label><input type='radio' name='{nm}' value='own'> This picture only "
                           f"<small>(locked as an exception)</small></label></div></div>")
        hidden = "" if main_i >= 0 else " style='display:none'"
        oddhtml = (f"<div class='cc-oddhead'{hidden}>Pictures that say something "
                   f"else: look at each (click), then choose. Undecided ones are left as they are.</div>"
                   f"<div class='cc-odds'>{''.join(odd)}</div>" if odd else "")
        if exc:
            oddhtml += (f"<div class='cc-exc'>🔒 {len(exc)} locked exception{'s' if len(exc) != 1 else ''} keep their own "
                        f"value: {html.escape(', '.join(Path(p).name for p in sorted(exc)[:8]))}</div>")
        out.append(f"<div class='cc-group' data-g='{gid}'><div class='cc-head'><span class='cc-trait'>"
                   f"{html.escape(g['trait'].capitalize())}</span>{kind}<span class='cc-cov'>in {g['mentioned']} of "
                   f"{total} captions</span></div><div class='cc-vals'>{''.join(vals)}</div>"
                   f"<div class='cc-why'><b>Why this is recommended:</b> {why}</div>"
                   f"<div class='cc-modes'>{''.join(rad)}</div>{oddhtml}</div>")
    return md, "".join(out)


CC_JS = ("(ds, x, u, m, t, ok) => { var o = {}; document.querySelectorAll('#cc_html .cc-group').forEach(function(g){ "
         "var id = g.dataset.g; var m = g.querySelector('input[name=\"cc-m-' + id + '\"]:checked'); "
         "var e = {mode: m ? m.value : 'skip', fix: [], own: [], words: {}}; "
         "g.querySelectorAll('.cc-odd').forEach(function(d){ if (d.style.display === 'none') return; "
         "var r = d.querySelector('input:checked'); if (r && r.value === 'fix') e.fix.push(d.dataset.p); "
         "if (r && r.value === 'own') e.own.push(d.dataset.p); }); "
         "g.querySelectorAll('select.cc-word').forEach(function(s){ e.words[s.dataset.i] = s.value; }); o[id] = e; }); "
         "return [ds, JSON.stringify(o), u, m, t, ok]; }")


CC_OPEN_JS = """
if (!window.ccMode) {
  window.ccMode = function(r){
    var g = r.closest('.cc-group'); if (!g) return;
    var m = /^lora:(\\d+)$/.exec(r.value), k = m ? m[1] : null;
    g.querySelectorAll('.cc-odd').forEach(function(d){ d.style.display = (k !== null && d.dataset.v !== k) ? '' : 'none'; });
    var h = g.querySelector('.cc-oddhead'); if (h) h.style.display = k !== null ? '' : 'none';
  };
}
if (!window.ccOpen) {
  window.ccOpen = function(btn){
    var d = btn.closest('.cc-odd'); if (!d) return;
    document.querySelectorAll('#cc_html .cc-odd.cc-sel').forEach(function(e){ e.classList.remove('cc-sel'); });
    d.classList.add('cc-sel');
    var el = document.querySelector('#cc_box textarea, #cc_box input'); if (!el) return;
    var setter = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(el), 'value').set;
    setter.call(el, d.dataset.p + '|' + Date.now());
    el.dispatchEvent(new Event('input', {bubbles:true}));
  };
}
"""


def cc_find(ds, url, model, trigger, progress=gr.Progress()):
    if S.CC is None:
        return "The consistency check is not available (consistency.py is missing).", ""
    if not ds:
        return "Pick a dataset first.", ""
    imgs = oc_images(ds)
    if not any(S.read_caption(p) for p in imgs):
        return "This dataset has no captions yet: write them first (step 4).", ""
    if not (url and model):
        url, model = oc_settings()
    S.CC.scan(ds, imgs, S.read_caption, url, model, trigger or ds, progress)
    return cc_render(ds)


def cc_apply(ds, choices_json, url, model, trigger, confirm=False, progress=gr.Progress()):
    """(message, confirm checkbox update). The choices are locked in <dataset>/traits.txt first; a choice that clashes
    with a lock is only written after the confirmation is ticked."""
    hide = gr.update(visible=False, value=False)
    if S.CC is None or not ds:
        return "Pick a dataset first.", hide
    res = S.CC.load_result(ds)
    if not res:
        return "Press **Check the captions** first.", hide
    try:
        choices = json.loads(choices_json or "{}")
    except json.JSONDecodeError:
        return "Could not read the choices on the page.", hide
    root = S.lora_path(ds)
    base = res
    res = S.CC.with_locks(copy.deepcopy(base), root)
    locks, clashes = S.CC.lock_changes(res, choices, root)
    if clashes and not confirm:
        return ("**These choices clash with locked traits** in `" + S.CC.LOCK_NAME + "`:\n\n"
                + "\n".join(f"- {c}" for c in clashes)
                + "\n\nTick **Replace the locked choices** below and press **Apply choices** again, or change the "
                  "choices back."), gr.update(visible=True, value=False)
    if not (url and model):
        url, model = oc_settings()
    try:
        S.CC.locks_save(root, locks)
    except OSError as e:
        return f"Could not write `{S.CC.lock_file(root)}`: {e}", hide
    backup, changed, kept, skipped = S.CC.apply(res, choices, S.read_caption, S.caption_file, url, model, trigger or ds, progress)
    if backup:
        record_change(f"consistency of {ds}", backup)
    imgs = oc_images(ds)
    base["applied"] = S.CC.captions_key({str(p): S.read_caption(p) for p in imgs if S.read_caption(p)})
    base["applied_at"] = time.strftime("%Y-%m-%d %H:%M")
    base["captions_after"] = base["applied"]
    base["kept"] = kept
    try:
        S.CC.result_path(ds).write_text(json.dumps(base), encoding="utf-8")
    except OSError:
        pass
    msg = (f"✔ Changed **{len(changed)} caption{'s' if len(changed) != 1 else ''}**" + (" (Undo restores them)." if changed else "."))
    if kept:
        msg += " Taken out of the captions (the trigger word learns them): " + ", ".join(kept) + "."
    if skipped:
        msg += (f" ⚠ {len(skipped)} sentence caption(s) were left unchanged: removing the trait would break the sentence "
                "and Ollama could not rewrite it (" + ", ".join(Path(p).name for p in skipped[:5]) + ").")
    for p, old, new in changed[:4]:
        msg += f"\n\n`{Path(p).name}`  \n~~{html.escape(old[:160])}~~  \n{html.escape(new[:160])}"
    return msg, hide


def cc_leave_out(ds, leave):
    """Kept-in-LoRA traits are added to the 'Never describe' list of the captioner, so new captions skip them too."""
    res = S.CC.load_result(ds) if (S.CC and ds) else None
    if not res or not res.get("applied"):
        return gr.update()
    kept = res.get("kept") or []
    have = [t.strip().lower() for t in (leave or "").split(",") if t.strip()]
    add = [t for t in kept if t.lower() not in have]
    return gr.update(value=(leave + ", " if leave else "") + ", ".join(add)) if add else gr.update()


def cc_open(v):
    p = (v or "").rsplit("|", 1)[0]
    if not p or not Path(p).is_file():
        return gr.update(), ""
    cap = S.read_caption(Path(p)) or ""
    return p, f"**{Path(p).name}**  \n{html.escape(cap)}"


def cc_stop():
    if S.CC is not None:
        S.CC.STOP.set()
    return "Stopping after the current caption ... (what was read so far is remembered)"
