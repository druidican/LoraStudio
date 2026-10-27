# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Datasets: New dataset | Edit existing | Improvements, and the nine steps they share.

New dataset starts at 1 Folder; Edit existing lists every dataset with its next step and opens the same steps from
where the dataset is. The steps reuse sdgui's parts where they exist: the watermark choices (6), the caption writer
and the consistency check (7), the repeat folder and the launch script page (9). What each step does to the folder
is in dsflow.py.
"""
from __future__ import annotations
__version__ = "1.261026z"  # LoRA Studio version, see version.py

import html
import json
import os
import re
import sys
import time
from pathlib import Path

import gradio as gr

import datacheck as DC
import dsflow as F
import features as FT

FLOW = list(range(2, 10))
_THUMBS: dict = {}


def sg():
    return sys.modules["sdgui"]


def esc(s) -> str:
    return html.escape(str(s if s is not None else ""))


def thumb(p, side=150) -> str:
    try:
        st = Path(p).stat()
    except OSError:
        return ""
    k = (str(p), st.st_mtime, side)
    if k not in _THUMBS:
        if len(_THUMBS) > 4000:
            _THUMBS.clear()
        _THUMBS[k] = DC.thumb_uri(Path(p), side)
    return _THUMBS[k]


def q(s: str) -> str:
    """A value for lsSend('ds_box', '...') inside an onclick attribute."""
    return esc(str(s).replace("\\", "\\\\").replace("'", "\\'"))


def send(v: str) -> str:
    return f"lsSend('ds_box', '{q(v)}')"


# --------------------------------------------------------------------------- navigation bits
def bar_html(st) -> str:
    ds = st.get("ds")
    if not ds:
        return ""
    m = F.meta(ds)
    labels = F.type_labels()
    chips = []
    if m["types"]:
        chips.append(f'<span class="ls-chip dark">{esc(" + ".join(labels.get(t, t) for t in m["types"]))}</span>')
        chips.append(f'<span class="ls-chip">{esc(F.PURPOSE_LABEL.get(m["purpose"], m["purpose"]))}</span>')
        found, pending = sg()._marks_counts(ds)
        if m.get("wm_done"):
            chips.append('<span class="ls-chip ok">watermarks: done</span>')
        elif not m["wm_removed"] and (pending or not found):
            chips.append('<span class="ls-chip warn">watermarks: ' + (f'{pending} not decided' if pending else
                                                                      'not checked yet') + '</span>')
    else:
        chips.append('<span class="ls-chip">LoRA type: not set yet</span>')
    n = len(F.images(ds))
    return (f'<div class="ds-bar"><span class="ds-bar-l">Dataset</span><b>{esc(ds)}</b>{"".join(chips)}'
            f'<span class="ds-bar-n">{n} pictures</span></div>')


def stepper_html(st) -> str:
    ds, cur = st.get("ds"), st.get("step", 2)
    states = F.step_states(ds) if ds else {}
    out = []
    for n, key, label in F.STEPS:
        s = states.get(key, ("todo", ""))[0]
        cls = "ds-st-now" if n == cur else {"done": "ds-st-done", "warn": "ds-st-warn"}.get(s, "")
        mark = "✓" if s == "done" and n != cur else str(n)
        click = f' onclick="{send(f"step|{n}")}"' if n >= 2 else ""
        miss = " ds-missing" if key == "tags" and not FT.have("ollama") and ds and any(
            x["style"] != "tags" for x in F.caption_sets(ds)) else ""
        lab = FT.label(label, "ollama") if miss else label
        out.append(f'<button class="ds-step {cls}{miss}"{click}{" disabled" if n == 1 else ""}><span class="ds-num">{mark}</span>'
                   f'<span>{esc(lab)}</span></button>')
    return '<nav class="ds-stepper" aria-label="Steps">' + "".join(out) + "</nav>"


def nav_updates(st):
    view = st.get("view", "new")
    flow = view == "flow"
    mode = st.get("mode", "new")
    active = mode if flow else view
    outs = [gr.update(visible=view == "new"), gr.update(visible=view == "edit"), gr.update(visible=view == "imp"),
            gr.update(visible=flow)]
    for k in ("new", "edit", "imp"):
        outs.append(gr.update(elem_classes=["ds-nav"] + (["ds-nav-on"] if active == k else [])))
    for n in FLOW:
        outs.append(gr.update(visible=flow and st.get("step") == n))
    outs.append(bar_html(st) if flow else "")
    outs.append(stepper_html(st) if flow else "")
    return outs


# --------------------------------------------------------------------------- 1 Folder (New dataset)
def new_list_html(chosen="") -> str:
    fs = F.candidate_folders()
    if F.datasets_dir() is None:
        return '<p class="ls-empty">The dataset folder is not set: Settings › Folders.</p>'
    rows = []
    for f in fs:
        if f["kind"] == "dataset":
            rows.append(f'<div class="ds-frow"><span class="ds-fname">{esc(f["name"])}/</span><span class="ls-chip">dataset</span>'
                        f'<button class="tr-btn tr-small" onclick="{send("edit|" + f["name"])}">Edit existing</button></div>')
        elif f["kind"] == "pictures":
            on = " ds-frow-on" if f["name"] == chosen else ""
            rows.append(f'<button class="ds-frow ds-frow-pick{on}" onclick="{send("folder|" + f["name"])}">'
                        f'<span class="ds-fname">{esc(f["name"])}/</span><span class="ls-chip ok">only pictures · {f["n"]}</span></button>')
    if not rows:
        rows.append('<p class="ls-empty">No folders yet. Put a folder of pictures in your dataset folder, or use a folder '
                    'somewhere else below.</p>')
    return '<div class="ds-flist">' + "".join(rows) + "</div>"


def new_tree(name, rep, src, outside) -> str:
    name = F.safe_name(name) or "name"
    try:
        rep = max(1, int(rep or 10))
    except (TypeError, ValueError):
        rep = 10
    n = 0
    if outside and Path(os.path.expanduser(outside)).is_dir():
        n = len(F._pictures_in(Path(os.path.expanduser(outside)), True))
    elif src and F.datasets_dir():
        n = next((f["n"] for f in F.candidate_folders() if f["name"] == src), 0)
    return (f'<div class="ds-tree"><div><b>{esc(name)}/</b></div><div>&nbsp;&nbsp;{rep}_{esc(name)}/ <span>the {n} pictures</span></div>'
            f'<div>&nbsp;&nbsp;{F.ORIGINALS}/ <span>untouched copies</span></div><div>&nbsp;&nbsp;model/ &nbsp;log/</div></div>'
            f'<p class="ls-note">Every later step changes pictures only in {rep}_{esc(name)}/; {F.ORIGINALS}/ lets you undo.'
            + (" The pictures are copied in; the folder outside stays as it is." if outside else "") + "</p>")


# --------------------------------------------------------------------------- picture grids
def grid(rows, ticks=False, click=True, elem="g") -> str:
    """rows: [{path, label, why, keep, bad}]"""
    cells = []
    for r in rows:
        p = r["path"]
        cls = "ds-cell" + ("" if r.get("keep", True) else " ds-off") + (" ds-bad" if r.get("bad") else "")
        tick = (f'<label class="ds-tick"><input type="checkbox" {"checked" if r.get("keep") else ""} '
                f'onchange="dsCount(this)"><span>{esc(r.get("label") or Path(p).name)}</span></label>' if ticks
                else f'<div class="ds-name">{esc(r.get("label") or Path(p).name)}</div>')
        img = f'<img src="{thumb(p)}" loading="lazy" alt="">'
        img = f'<button class="ds-img" onclick="{send("pick|" + p)}">{img}</button>' if click else img
        pin = ""
        if "pin" in r:                                   # step 5: your own decision, kept through new sorts
            pv = r["pin"] or ""
            pin = ('<div class="ds-pins">' + "".join(
                f'<button class="ds-pin{" ds-pin-on" if pv == k else ""}" data-k="{k}" title="{t}" '
                f'onclick="dsPin(this)">{lab}</button>'
                for k, lab, t in (("keep", "📌 Always", "Always keep this picture: it wins over every rule and stays "
                                                       "pinned after a new sort. Click again to unpin."),
                                  ("never", "⛔ Never", "Never use this picture, whatever the sort says. Click again "
                                                        "to unpin."))) + "</div>")
        cells.append(f'<div class="{cls}" data-p="{esc(p)}">{img}{tick}{pin}<div class="ds-why">{esc(r.get("why", ""))}</div></div>')
    return f'<div class="ds-grid" id="ds-grid-{elem}">' + "".join(cells) + "</div>"


GRID_JS = """
window.dsCount = function(el){
  var g = el.closest('.ds-grid'); if (!g) return;
  var c = el.closest('.ds-cell'); if (c) c.classList.toggle('ds-off', !el.checked);
  var n = g.querySelectorAll('input:checked').length, t = g.querySelectorAll('input').length;
  document.querySelectorAll('.ds-count').forEach(function(e){ e.textContent = n + ' ticked · ' + (t - n) + ' go to _reserve/'; });
};
window.dsPin = function(b){
  var c = b.closest('.ds-cell'); if (!c) return;
  var on = !b.classList.contains('ds-pin-on');
  c.querySelectorAll('.ds-pin').forEach(function(x){ x.classList.remove('ds-pin-on'); });
  var k = on ? b.dataset.k : '';
  if (on) b.classList.add('ds-pin-on');
  var i = c.querySelector('input[type=checkbox]');
  if (i && k) { i.checked = (k === 'keep'); dsCount(i); }
  var w = c.querySelector('.ds-why');
  if (w) w.textContent = k === 'keep' ? 'always keep (your pin)' : (k === 'never' ? 'never use (your pin)' : 'unpinned: press Use the suggestion to sort it in again');
  lsSend('ds_box', 'pin|' + (k || 'none') + '|' + c.dataset.p);
};
window.dsTickAll = function(on){
  document.querySelectorAll('#ds_s5 .ds-grid input').forEach(function(i){ i.checked = on; dsCount(i); });
};
"""
TICKED_JS = ("(x) => [JSON.stringify(Array.from(document.querySelectorAll('#ds_s5 .ds-cell input:checked'))"
             ".map(function(e){ return e.closest('.ds-cell').dataset.p; }))]")


# --------------------------------------------------------------------------- 2 Usable scan
def s2_extra_update(ds):
    """The 'also look outside the repeat folder' checkbox: its label says how many pictures that is."""
    n = len(F.extra_pictures(ds)) if ds else 0
    return gr.update(value=F.include_extra(ds), interactive=bool(n) or F.include_extra(ds),
                     label=(f"Also look at the {n} picture{'s' if n != 1 else ''} outside the repeat folder "
                            "(_reserve/, _extra/ and other folders)") if n else
                     "Also look at pictures outside the repeat folder (there are none)")


def s2_load(st):
    ds = st.get("ds")
    if st.get("view") != "flow" or st.get("step") != 2 or not ds:
        return gr.update(), gr.update(), gr.update(), gr.update()
    sc = F._load(ds, "scan.json")
    n = len(F.candidates(ds))
    if not sc:
        return (f"**{n} pictures.** Size, sharpness, compression and broken files; no AI, about a second per 10 "
                "pictures. Skip it when you know the pictures are fine.", "", gr.update(value="Not usable first"),
                s2_extra_update(ds))
    return s2_render(ds, sc, "Not usable first") + (gr.update(), s2_extra_update(ds))


def _where(ds, p) -> str:
    """'from _reserve/' for a picture outside the repeat folder, '' for one in it."""
    try:
        rel = Path(p).relative_to(F.root(ds))
    except ValueError:
        return ""
    rows = [x["folder"] for x in F.sg().dataset_stats(ds)["rows"]]
    if any(Path(p).parent == f for f in rows):
        return ""
    return f"from {rel.parts[0]}/ · " if len(rel.parts) > 1 else "from the dataset folder · "


def s2_render(ds, sc, show):
    have = {str(p) for p in F.candidates(ds)}
    pics = {p: e for p, e in sc["pics"].items() if p in have}
    c = {}
    for e in pics.values():
        c[e["cat"]] = c.get(e["cat"], 0) + 1
    chips = (f'<span class="ls-chip ok">{c.get("usable", 0)} usable</span>' +
             "".join(f'<span class="ls-chip bad">{n} {esc(k)}</span>' for k, n in c.items() if k != "usable"))
    md = f'<div class="ds-chips">{chips}<span class="ls-note">scanned {esc(sc["at"])}, smallest side wanted {sc["min_side"] or "—"} px</span></div>'
    rows = [{"path": p, "why": _where(ds, p) + (e["why"] if e["bad"] else f'{e["w"]}×{e["h"]}'), "keep": not e["bad"],
             "bad": e["bad"]}
            for p, e in sorted(pics.items(), key=lambda kv: (not kv[1]["bad"], Path(kv[0]).name))
            if show == "All" or e["bad"]]
    if not rows:
        return md, '<p class="ls-empty">Every picture is usable.</p>'
    return md, grid(rows, click=False, elem="s2")


# --------------------------------------------------------------------------- 4 Sort
def s4_html(ds) -> str:
    r = F.sort_result(ds)
    if not r:
        return '<p class="ls-empty">Not sorted yet. Press <b>Sort</b>.</p>'
    pics = r["pics"]
    cats = {}
    for p, e in pics.items():
        cats.setdefault(e["cat"], []).append(e)
    out_why = {}
    for e in cats.get("out", []):
        k = "too blurry" if "blurry" in e["why"] else "resolution too low" if "small" in e["why"] else "broken"
        out_why[k] = out_why.get(k, 0) + 1
    labels = F.type_labels()
    tname = " + ".join(labels.get(t, t) for t in r.get("types") or []) or "the LoRA type"
    p1 = (f'<div class="ds-pass"><div class="ds-pass-h"><b>Pass 1 · quick</b><span class="ls-chip ok">done</span></div>'
          f'<div>{r["pass1_out"]} out' + (": " + ", ".join(f"{n} {k}" for k, n in out_why.items()) if out_why else "")
          + f' (for {esc(tname)}). {len(pics) - r["pass1_out"]} go on.</div>'
          + ('<div class="ls-note">Quality rated and look-alikes found by the picture tools.</div>' if r.get("tools") else
             f'<div class="ls-note">{esc(r.get("tools_note") or "The picture tools are not installed: quality is judged by sharpness and size only (Settings › Trainers and GPU).")}</div>')
          + '</div>')
    if r.get("ai"):
        fr = r.get("framing") or {}
        p2body = (f'<div class="ds-base ds-base-{r["baseline"]}">baseline: {esc(r["baseline"])}</div>'
                  f'<div>{r["good"]} good pictures. Framing: ' + " · ".join(f"{n} {esc(k)}" for k, n in sorted(fr.items(), key=lambda kv: -kv[1]))
                  + f'. {r["outfits"]} outfits, {r["places"]} places.</div>')
        if r.get("notes"):
            p2body += "<ul>" + "".join(f"<li>{esc(n)}</li>" for n in r["notes"]) + "</ul>"
    else:
        p2body = f'<div class="ls-note">Skipped: {esc(r.get("note") or "no Ollama model")}. Sort again when Ollama runs.</div>'
    p2 = (f'<div class="ds-pass"><div class="ds-pass-h"><b>Pass 2 · detailed (Ollama)</b>'
          f'<span class="ls-chip {"ok" if r.get("ai") else ""}">{"done" if r.get("ai") else "skipped"}</span></div>'
          f'<div class="ls-note">Subject, framing, outfit, place, quality, watermarks. Remembered for next time.</div>{p2body}</div>')
    groups = []
    if r["groups"]:
        n = sum(1 for e in pics.values() if e["cat"] == "alike")
        groups.append(("Too alike", f'{len(r["groups"])} groups · {n} pictures', "the best of each group is kept",
                       " · ".join(f'{esc(g["label"])} ({g["n"]})' for g in r["groups"][:4])))
    if cats.get("dup"):
        groups.append(("Duplicates", f'{len(cats["dup"])}', "the larger, sharper one is kept", ""))
    if cats.get("detail"):
        parts = {}
        for e in cats["detail"]:
            parts[e.get("part") or "detail"] = parts.get(e.get("part") or "detail", 0) + 1
        groups.append(("Detail shots", f'{len(cats["detail"])} kept',
                       'only part of the character: not "no clear subject", they teach details the full pictures show too small',
                       " · ".join(f"{esc(k)} ({n})" for k, n in parts.items())))
    if cats.get("poor"):
        reasons = {}
        for e in cats["poor"]:
            k = e["why"].split(";")[0].split(":")[0][:40]
            reasons[k] = reasons.get(k, 0) + 1
        groups.append(("Not good enough", f'{len(cats["poor"])}', " · ".join(f"{esc(k)}" for k in reasons), ""))
    wm = sum(1 for e in pics.values() if e.get("wm"))
    if wm:
        groups.append(("Watermarks", f"{wm}", "a crop is offered in step 6", ""))
    g_html = "".join(f'<div class="ds-grp"><b>{a}</b><span class="ls-chip">{b}</span><span class="ls-note">{c}</span>'
                     + (f'<div class="ds-grp-x">{d}</div>' if d else "") + "</div>" for a, b, c, d in groups)
    return f'<div class="ds-sort">{p1}{p2}{g_html}<p class="ls-note">Sorted {esc(r["at"])} in {r["seconds"]} s.</p></div>'


# --------------------------------------------------------------------------- 5 Select
def s5_suggest_html(ds, n, rng, why) -> str:
    hint = ""
    try:                                                 # what the datasets of your good LoRAs of this type had
        import insights as IN
        t = (F.meta(ds)["types"] or [""])[0]
        hint = IN.dataset_hint(t) if t else ""
    except Exception:                                    # noqa: BLE001
        pass
    return (f'<div class="ds-sug"><span class="rs-label">SUGGESTED SIZE</span><div class="rs-big">{n} pictures '
            f'<span class="ls-note">({rng} works)</span></div><div class="ls-note">{esc(why)}</div>'
            + (f'<div class="ls-note" style="margin-top:6px">★ {esc(hint)}</div>' if hint else "") + '</div>')


def s5_grid(ds, keep_n):
    rows = F.rank(ds, int(keep_n or 0))
    k = sum(1 for r in rows if r["keep"])
    mix = (f'<div class="ds-mix"><span class="rs-label">MIX OF THE SELECTION</span><div>{esc(F.mix_text(rows))}</div>'
           '<div class="ls-note">' + ("Picked for variety: what each picture adds (framing, outfit, place, a detail) "
                                       "counts as much as its quality." if F.meta(ds).get("select_mode") == "varied" else
                                       "Picked by quality first, then what each picture adds (framing, outfit, place).")
           + '</div></div>')
    cells = [{"path": r["path"], "why": _where(ds, r["path"]) + r.get("reason", ""), "keep": r["keep"],
              "bad": r["cat"] in ("out", "poor") and not r.get("pin"), "pin": r.get("pin", "")}
             for r in sorted(rows, key=lambda r: (not r["keep"], r["cat"] in ("out", "poor", "dup"), Path(r["path"]).name))]
    outside = {str(x) for x in F.extra_pictures(ds)} if F.include_extra(ds) else set()
    n_in = sum(1 for r in rows if r["keep"] and str(r["path"]) in outside)
    n_drop = sum(1 for r in rows if not r["keep"] and str(r["path"]) not in outside)
    head = (f'<div class="ds-count">{k} ticked · {n_drop} go to _reserve/'
            + (f' · {n_in} from outside join the repeat folder' if n_in else "") + '</div>')
    return mix, head + grid(cells, ticks=True, click=False, elem="s5")


# --------------------------------------------------------------------------- 6 Resize
MODE_LABEL = {"scale": "Scale down", "none": "Unchanged", "keep": "Keep the original size", "crop": "Crop",
              "upscale": "Upscale (Real-ESRGAN)"}


def s6_html(ds, max_side):
    rows = F.resize_plan(ds, int(max_side or 1536))
    big = [r for r in rows if r["mode"] == "scale"]
    crop = [r for r in rows if r["mode"] == "crop"]
    up = [r for r in rows if r["mode"] == "upscale"]
    labels = F.type_labels()
    m = F.meta(ds)
    up_txt = (f' {len(up)} are a little small and are upscaled by the picture tools (Real-ESRGAN, gently: only up to '
              f'the size your LoRA type trains at).' if up else "")
    sm = (f'<div class="ds-sum"><b>Keep the shape, scale down.</b> {len(big)} of {len(rows)} are bigger than needed and '
          f'are scaled down; {len(crop)} cropped by you; the rest stay as they are.{up_txt} Nothing is padded or squared: the '
          f'trainer groups pictures by shape ({esc(" + ".join(labels.get(t, t) for t in m["types"]) or "your LoRA type")}).</div>')
    tr, rest = [], []
    for r in sorted(rows, key=lambda r: (not r["by_you"], r["mode"] == "none", Path(r["path"]).name)):
        sugg = r.get("sugg", "none")
        opts = "".join(f'<option value="{k}"{" selected" if k == r["mode"] else ""}>{MODE_LABEL[k]}'
                       f'{" (suggested)" if k == sugg else ""}</option>' for k in (sugg, "keep", "crop"))
        res = (f'{r["new"][0]}×{r["new"][1]}' if r["mode"] in ("scale", "crop", "upscale") else "as it is")
        (tr if (r["mode"] != "none" or r["by_you"]) else rest).append(f'<tr><td><button class="ds-img ds-small" onclick="{send("pick|" + r["path"])}"><img src="{thumb(r["path"], 64)}" alt=""></button></td>'
                  f'<td class="rs-mono">{esc(Path(r["path"]).name)}</td><td>{r["w"]}×{r["h"]}</td>'
                  f'<td><select onchange="lsSend(\'ds_box\', \'rmode|{q(r["rel"])}|\' + this.value)">{opts}</select></td>'
                  f'<td>{res}</td><td>{"<span class=ls-chip>changed by you</span>" if r["by_you"] else ""}</td></tr>')
    head = ('<thead><tr><th></th><th>Picture</th><th>Now</th><th>Do</th><th>Result</th><th></th></tr></thead>')
    table = ""
    if tr:
        table = f'<div class="rs-wrap"><table class="rs-table">{head}<tbody>{"".join(tr)}</tbody></table></div>'
    else:
        table = '<p class="ls-note">Every picture can stay as it is.</p>'
    if rest:
        table += (f'<details class="ds-fold"{" open" if not tr and len(rest) <= 12 else ""}><summary>{len(rest)} '
                  f'picture{"s" if len(rest) != 1 else ""} stay as they are: show them (to crop or scale one of them '
                  f'anyway)</summary><div class="rs-wrap"><table class="rs-table">{head}<tbody>{"".join(rest)}'
                  '</tbody></table></div></details>')
    n_do = len(big) + len(crop) + len(up)
    label = (f"Resize {len(big)}, crop {len(crop)}" + (f", upscale {len(up)}" if up else "") + " →") if n_do else "Continue to Tags →"
    return sm, table, gr.update(value=label)


# --------------------------------------------------------------------------- 7 Tags
def s7_sets_html(ds) -> str:
    sets = F.caption_sets(ds)
    cards = []
    for s in sets:
        c, t = F.set_count(ds, s["ext"])
        lab = "Tags" if s["style"] == "tags" else "Sentences"
        cards.append(f'<div class="ds-set"><div><b>{lab}</b> <span class="rs-mono">IMG{esc(s["ext"])}</span>'
                     f'<span class="ls-chip {"ok" if c == t and t else ""}">{c}/{t}</span></div>'
                     f'<div class="ls-note">for {esc(", ".join(s["types"]) or "your LoRA type")}</div></div>')
    note = ("Two LoRA types with different caption styles, so two caption sets side by side. Each launch script reads "
            "its own (--caption_extension)." if len(sets) > 1 else "One caption set for your LoRA type.")
    return f'<div class="ls-note">{note}</div><div class="ds-sets">' + "".join(cards) + "</div>"


REC_TEXT = {
    "character": ("Character", "The captions describe what <b>changes</b> from picture to picture: clothes, pose, place, light, "
                  "framing. <i>Never describe</i> holds what is always the person (face, hair and eye colour, body), so the "
                  "trigger word learns it. Consistency below takes out traits that are the same in most pictures."),
    "style": ("Style", "The captions describe <b>everything in</b> the picture: who is shown with their hair, eyes and "
              "clothes, animals and objects, the place, the light and the composition. They never name the style "
              "itself (the art style, the medium, the technique, the artist): those words are in <i>Never describe</i>, "
              "so the trigger word learns the style. Detail is set to Detailed or Everything. The character consistency "
              "check is not used for a style."),
    "object": ("Object", "The captions name the object with the trigger word and describe <b>everything around it</b>: "
               "who holds, wears or uses it, the clothes, the pose, the place, the light. They never describe the object "
               "itself (shape, colours, material, details): its words are in <i>Never describe</i>, so the trigger word "
               "learns it. The character consistency check is not used."),
    "action": ("Action / pose", "The captions describe <b>the person and the scene</b>: hair, clothes, expression, what "
               "they hold, the place, the light, the camera angle. They never describe the pose or action itself (how the "
               "body, arms, legs and hands are placed): the trigger word learns it. The pictures should show many "
               "different people and places. The character consistency check is not used."),
    "concept": ("Concept / effect", "The captions describe everything in the picture except the effect itself: its "
                "words are in <i>Never describe</i>, so the trigger word learns it. The character consistency check is "
                "not used."),
}


def s7_rec_html(purpose) -> str:
    t, body = REC_TEXT.get(purpose, REC_TEXT["character"])
    return f'<div class="ls-card ds-rec"><span class="rs-label">RECOMMENDED FOR A {t.upper()} LORA</span><div>{body}</div></div>'


def leave_update(purpose, cur, words=""):
    """'Never describe' for this purpose: changed only when it still holds another purpose's default."""
    CAP = sg().CAP
    if CAP is None:
        return gr.update()
    defaults = {"character": CAP.DEFAULT_LEAVE_OUT, "style": CAP.STYLE_LEAVE_OUT, "object": words, "action": words,
                "concept": words}
    labels = {"character": "Never describe (always part of the person)",
              "style": "Never describe (the style itself: the trigger word learns it)",
              "object": "Never describe (the object itself: the trigger word learns it)",
              "action": "Never describe (the pose or action itself: the trigger word learns it)",
              "concept": "Never describe (the effect itself: the trigger word learns it)"}
    cur = (cur or "").strip()
    want = defaults.get(purpose, CAP.DEFAULT_LEAVE_OUT)
    if purpose in F.CONCEPT_KINDS and words and cur != words:
        return gr.update(value=words, label=labels.get(purpose))      # the words you gave in step 3
    if cur in ("",) + tuple(v for k, v in defaults.items() if k != purpose):
        return gr.update(value=want, label=labels.get(purpose))
    return gr.update(label=labels.get(purpose))


def batch_what() -> str:
    import html as _h
    t = F.batch_summary()
    t = _h.escape(t).replace("`_reserve/`", "<code>_reserve/</code>")
    return ('<div class="ls-note"><b>Fix the ticked datasets automatically</b> (less is better): ' + t +
            ' <a href="#" onclick="lsSetNav(\'batch\'); return false;">Change this in Settings › Automatic improvement →</a></div>')


def batch_html() -> str:
    B = F.BATCH
    if not B["running"] and not B["results"]:
        return ""
    out = []
    if B["running"]:
        n_done = len(B["results"])
        total = n_done + len(B["queue"]) + (1 if B["current"] else 0)
        out.append(f'<div class="ds-bline"><span class="ls-chip warn">running</span> <b>{esc(B["current"])}</b> '
                   f'({n_done + 1} of {total}) · {esc(B["step"])}</div>')
    for r in reversed(B["results"]):
        chip = '<span class="ls-chip ok">done</span>' if r["ok"] else '<span class="ls-chip bad">stopped / error</span>'
        out.append(f'<div class="ds-bres"><div>{chip} <b>{esc(r["ds"])}</b> <span class="ls-note">{r["secs"] // 60} min</span></div>'
                   + "".join(f'<div class="ls-note">· {esc(l)}</div>' for l in r["lines"]) + "</div>")
    if not B["running"] and B["results"]:
        out.insert(0, f'<div class="ds-bline"><b>Finished {len(B["results"])} dataset(s).</b> Press Refresh to see what is '
                      'left to improve.</div>')
    return '<div class="ds-batch">' + "".join(out) + "</div>"


# --------------------------------------------------------------------------- 8 Review
def s8_render(ds, show="Problems first"):
    rv = F.review(ds)
    tree = '<div class="ds-tree">' + "".join(f'<div>{esc(a)} <span>{esc(b)}</span></div>' for a, b in rv["tree"]) + "</div>"
    ck = []
    for state, text, action, path in rv["checks"]:
        mark = {"ok": "✓", "warn": "!", "bad": "✕"}[state]
        btn = ""
        if action == "delete":
            btn = f'<button class="tr-btn tr-small" onclick="{send("fix|delete|" + path)}">Delete</button>'
        elif action == "movein":
            btn = f'<button class="tr-btn tr-small" onclick="{send("fix|movein|" + path)}">Move in</button>'
        ck.append(f'<div class="hc-row"><span class="hc-dot hc-{state}">{mark}</span><span class="hc-t">{esc(text)}</span>{btn}</div>')
    checks = '<div class="hc-card">' + "".join(ck) + "</div>"
    probs = rv["problems"]
    imgs = rv["images"]
    order = sorted(imgs, key=lambda p: (p not in probs, Path(p).name)) if show == "Problems first" else sorted(imgs, key=lambda p: Path(p).name)
    cells = [{"path": p, "why": "; ".join(probs.get(p, [])), "keep": True, "bad": p in probs} for p in order]
    return tree, checks, grid(cells, click=True, elem="s8")


# --------------------------------------------------------------------------- 9 Launch script: the calculation
SETTINGS = [("max_train_epochs", "Epochs", "More passes over the same pictures. Too many steps and the LoRA starts to copy the pictures instead of learning what they share."),
            ("repeats", "Repeats", "Repeats × epochs = views per picture. More repeats with fewer epochs gives the same steps but fewer save points to choose from."),
            ("train_batch_size", "Batch size", "2 halves the steps per epoch and smooths learning, but needs more video memory and a slightly higher learning rate."),
            ("learning_rate", "Learning rate", "Higher learns faster but can burn in (harsh colours, NaN loss). Lower is safer but may need more epochs."),
            ("network_dim", "Network dim", "Higher holds more small detail (marks, outfit patterns) but makes a bigger file and copies the pictures sooner."),
            ("network_alpha", "Network alpha", "Scales the learning of the LoRA weights; half the dim is the usual choice."),
            ("resolution", "Resolution", "Higher gives sharper detail, with much more video memory and time."),
            ("save_every_n_epochs", "Save every N epochs", "More save points to compare; Results says which one to try first.")]


def _sec_per_step(tkey) -> float:
    """From the last finished run of this LoRA type: seconds per step (0 when unknown)."""
    try:
        import training_ui as T
        for r in T.runs(60):
            if r["result"] != "done" or not r["info"] or r["info"]["tkey"] != tkey:
                continue
            f = sg().LOG_DIR / r["log"]
            tail = f.read_text(encoding="utf-8", errors="replace")[-20000:]
            m = re.findall(r"(\d+)/(\d+) \[", tail)
            if m:
                total = int(m[-1][1])
                secs = T._secs(r["dur"])
                if total and secs:
                    return secs / total
    except Exception:                                    # noqa: BLE001
        pass
    return 0.0


def calc(ds, tkey, text):
    """(calculation html, effects html, values for the settings inputs)."""
    S = sg()
    if not ds or not tkey or not text:
        return '<p class="ls-empty">Pick the LoRA type above.</p>', "", [gr.update()] * len(SETTINGS)
    base = S.lora_path(ds)
    stats = S.dataset_stats(ds)
    pics = stats["total"]
    reps = [r["repeats"] for r in stats["rows"]] or [1]
    rep = reps[0]
    arg = lambda n: S.script_arg(text, n, base)
    try:
        ep = int(float(arg("max_train_epochs") or 0))
    except ValueError:
        ep = 0
    try:
        bs = int(float(arg("train_batch_size") or 1))
    except ValueError:
        bs = 1
    try:
        ga = int(float(arg("gradient_accumulation_steps") or 1))
    except ValueError:
        ga = 1
    try:
        plan = S.make_plan(ds, tkey)
        rec = plan["rec"]
        tg = plan["cfg"]["targets"]
    except Exception:                                    # noqa: BLE001
        plan, rec, tg = None, None, None
    if rec and not arg("train_batch_size"):
        bs, ga = rec["bs"], rec["ga"]
    samples = sum(r["n"] * r["repeats"] for r in stats["rows"]) or pics * rep
    spe = max(1, -(-samples // max(1, bs * ga)))
    total = spe * ep
    views = ep * rep
    labels = F.type_labels()
    rows = [("Pictures", f"{pics}"), ("× Repeats", " / ".join(str(r) for r in reps)), ("= Samples per epoch", f"{samples}"),
            ("÷ Batch × accumulation", f"{bs} × {ga}"), ("= Steps per epoch", f"{spe}"),
            ("× Epochs", f"{ep}" + (f" (recommended {rec['epochs']})" if rec else "")), ("= Total steps", f"{total:,}"),
            ("Views per picture", f"{views}")]
    calc_html = ('<table class="ds-calc">' + "".join(f"<tr><td>{a}</td><td><b>{b}</b></td></tr>" for a, b in rows) + "</table>")
    if tg:
        calc_html += (f'<p class="ls-note">Recommended for {esc(labels.get(tkey, tkey))}: {tg[0]:,}–{tg[-1]:,} steps. Steps per '
                      "epoch can be a few more with shape buckets.</p>")
    if rec and rec.get("want_repeats") and rec["want_repeats"] != rep:
        calc_html += (f'<p class="ls-note"><b>Recommended repeats: {rec["want_repeats"]}</b> (the folder says {rep}). Change '
                      "Repeats below, or use the Repeat folder fold-out.</p>")
    eff = []
    sps, gnote = _sec_per_step(tkey), ""
    try:                                                 # your own runs of this type: per batch size, and memory
        import insights as IN
        g = next((x for x in IN.compute()["gpu"] if x["type"] == tkey), None)
        if g:
            same = [c["sps"] for c in g["batches"] if c["sps"] and c["batch"] == bs]
            if same:
                sps = sorted(same)[len(same) // 2]
            if g["oom_at"] and bs >= g["oom_at"]:
                gnote = (f"batch {g['oom_at']} ran out of video memory before"
                         + (f"; batch {g['safe']} finished" if g["safe"] else "") + " (Collection › Insights)")
            elif g["safe"] and bs <= g["safe"]:
                gnote = f"batch {bs} has finished before on this computer"
    except Exception:                                    # noqa: BLE001
        pass
    if gnote:
        eff.append(("Your graphics card", "⚠ too big?" if "ran out" in gnote else "fits", gnote))
    if sps and total:
        mins = sps * total / 60
        line = f"about {mins:.0f} min"
        if rec:
            d = sps * (total - rec["total_steps"]) / 60
            line += f" ({'+' if d >= 0 else '−'}{abs(d):.0f} min against the recommendation)"
        eff.append(("Training time", line, f"from your earlier runs of this type ({sps:.2f} s per step)"))
    else:
        eff.append(("Training time", "unknown", "known after the first finished run of this type"))
    purpose = F.meta(ds)["purpose"]
    if tg:
        lo, hi = tg[0], tg[-1]
        strong = {"character": "Likeness / consistency", "style": "Style strength", "object": "How exact the object is",
                  "action": "How exact the pose is", "concept": "Effect strength"}.get(purpose, "Likeness / consistency")
        flex = {"style": "Flexibility (other subjects, prompts that ask for another look)",
                "object": "Flexibility (other people, places and uses)",
                "action": "Flexibility (other people, clothes and places)",
                "concept": "Flexibility (other subjects and settings)"}.get(purpose, "Flexibility (new poses, outfits)")
        if total > hi:
            eff.append((strong, "stronger", ""))
            eff.append((flex, "less", f"{total:,} steps is above the range: " + (
                "the style shows strongly, but it can take over the subject and repeat things from the pictures."
                if purpose == "style" else "the face sticks better, but poses and outfits can get stiff."
                if purpose == "character" else "it sticks, but it can bring along people and places from the pictures.")
                + " Save points let Results advise an earlier one."))
        elif total < lo:
            eff.append((strong, "weaker", ""))
            eff.append((flex, "more", f"{total:,} steps is below the range: " + (
                "the style may come out faint." if purpose == "style" else "the LoRA may not hold the character well."
                if purpose == "character" else "it may not come through.")))
        else:
            eff.append(("Strength and flexibility", "balanced", f"{total:,} steps is inside the range."))
    eff_html = ('<div class="ds-eff">' + "".join(f'<div class="ds-effrow"><span>{esc(a)}</span><b>{esc(b)}</b>'
                                                 f'<span class="ls-note">{esc(c)}</span></div>' for a, b, c in eff) + "</div>")
    recd = {}
    if plan and rec:
        cf = S.form_dict(" ".join(plan["cfg"].get("flags", [])))
        recd = {"max_train_epochs": rec["epochs"], "repeats": rec.get("want_repeats") or (rec["repeats"] or [""])[0],
                "train_batch_size": rec["bs"], "learning_rate": plan["cfg"].get("lr"),
                "network_dim": cf.get("network_dim"), "network_alpha": cf.get("network_alpha"),
                "resolution": cf.get("resolution"), "save_every_n_epochs": plan["cfg"].get("save_every", 1)}
    vals = []
    for flag, _, why in SETTINGS:
        cur = str(rep) if flag == "repeats" else (arg(flag) or "")
        r_ = recd.get(flag)
        if r_ in (None, ""):
            info = why
        else:
            same = _same(cur, r_) or cur == ""
            info = (f"Recommended: {r_}" + ("  ·  not set in your script (the trainer's default is used)" if cur == ""
                                             else "" if same else "  ·  yours is different")
                    + (" (the folder is renamed when you change it here or press Fill with recommended)" if flag == "repeats" and not same else "")
                    + f".  {why}")
        vals.append(gr.update(value=cur, info=info))
    return calc_html, eff_html, vals


def _same(a, b) -> bool:
    try:
        return abs(float(str(a).split(",")[0]) - float(str(b).split(",")[0])) < 1e-12
    except ValueError:
        return str(a).strip() == str(b).strip()


def apply_settings(ds, tkey, text, *vals):
    S = sg()
    if not ds or not text:
        return gr.update(), "Open a launch script first."
    new = text
    msgs = []
    for (flag, label, _), v in zip(SETTINGS, vals):
        v = str(v or "").strip()
        if not v:
            continue
        if flag == "repeats":
            if v.isdigit():
                r_ = F.rename_repeats(ds, int(v))
                if r_:
                    msgs.append(r_)
            continue
        new, changed, ok = S.set_arg(new, flag, v)
        if not ok:
            msgs.append(f"{label}: no place for --{flag} in this script")
    return new, (("✔ Written into the script text. " if new != text else "") + "; ".join(msgs)
                 + (" Press **Save** to keep it." if new != text else "") if new != text or msgs else "")


# --------------------------------------------------------------------------- the page
def build(app, show):
    S = sg()
    UI = S.UI
    _types = S.type_choices()
    st = gr.State({"view": "new", "mode": "new", "ds": "", "step": 2})
    box = gr.Textbox(elem_id="ds_box", show_label=False, container=False)
    navbox = gr.Textbox(visible=False)
    sync = gr.Textbox(visible=False)
    with gr.Row(elem_id="ds_navrow"):
        b_new = gr.Button("New dataset", elem_classes=["ds-nav", "ds-nav-on"], size="sm", scale=0, min_width=150)
        b_edit = gr.Button("Edit existing", elem_classes=["ds-nav"], size="sm", scale=0, min_width=150)
        b_imp = gr.Button("Improvements", elem_classes=["ds-nav"], size="sm", scale=0, min_width=150)

    # ---- New dataset: 1 Folder
    with gr.Column(visible=True) as p_new:
        gr.HTML(stepper_html({"ds": "", "step": 1}))
        with gr.Row(equal_height=False):
            with gr.Column(scale=3, elem_classes=["ls-card"]):
                gr.Markdown("### Pick a folder in your dataset folder")
                gr.Markdown("A folder with only pictures becomes a new dataset. Folders that already are datasets open in "
                            "**Edit existing**.", elem_classes=["ls-note"])
                n_list = gr.HTML(elem_id="ds_newlist")
                n_out = gr.Textbox(label="Or a folder somewhere else (the pictures are copied in)",
                                   placeholder="~/Pictures/…")
            with gr.Column(scale=2, elem_classes=["ls-card"]):
                gr.Markdown("### What happens")
                with gr.Row():
                    n_name = gr.Textbox(label="Dataset name", scale=3)
                    n_rep = gr.Number(label="Repeats", value=10, precision=0, minimum=1, scale=1)
                n_tree = gr.HTML(new_tree("", 10, "", ""))
                n_go = gr.Button("Create and continue →", variant="primary")
                n_msg = gr.Markdown()
        n_src = gr.State("")

    # ---- Edit existing
    with gr.Column(visible=False) as p_edit:
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=260, elem_classes=["ls-filters"]):
                e_find = gr.Textbox(label="Find a dataset", placeholder="name")
                e_list = gr.HTML(elem_id="ds_editlist")
            with gr.Column(scale=4):
                e_head = gr.HTML()
                with gr.Accordion("Add pictures from a folder", open=False) as e_add_acc:
                    gr.Markdown("Pictures that are good enough and not already in the dataset are copied into its repeat "
                                "folder (and into _originals/). Then sort and select again.", elem_classes=["ls-note"])
                    with gr.Row():
                        e_add_dir = gr.Textbox(label="Folder", placeholder="~/Pictures/new batch", scale=4)
                        e_add_sub = gr.Checkbox(label="Include subfolders", value=True, scale=1)
                        e_add_btn = gr.Button("Add pictures", size="sm", scale=0, min_width=140)
                    e_add_msg = gr.Markdown()
                e_steps = gr.HTML(elem_id="ds_editsteps")

    # ---- Improvements
    with gr.Column(visible=False) as p_imp:
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                i_head = gr.HTML()
            i_ref = gr.Button("Refresh", size="sm", scale=0, min_width=100)
        gr.Markdown("From what is known already; nothing is scanned on this page.", elem_classes=["ls-note"])
        with gr.Column(elem_classes=["ls-card"]):
            i_what = gr.HTML(batch_what())
            with gr.Row(elem_classes=["ls-bar"]):
                i_all = gr.Button("Tick all", size="sm", scale=0, min_width=100)
                i_none = gr.Button("Untick all", size="sm", scale=0, min_width=110)
                i_go = gr.Button("Fix the ticked datasets", variant="primary", size="sm", scale=0, min_width=220)
                i_stop = gr.Button("Stop", variant="stop", size="sm", scale=0, min_width=90)
            i_sel = gr.Textbox(visible=False)
            i_msg = gr.Markdown()
            i_prog = gr.HTML(elem_id="ds_batch")
        i_html = gr.HTML(elem_id="ds_imp")

    # ---- the steps 2..9
    with gr.Column(visible=False) as p_flow:
        f_bar = gr.HTML(elem_id="ds_barbox")
        f_step = gr.HTML(elem_id="ds_stepbox")
        steps = {}
        with gr.Column(visible=False, elem_id="ds_s2") as steps[2]:
            gr.Markdown("### Scan for usable pictures (optional)")
            s2_extra = gr.Checkbox(label="Also look at pictures outside the repeat folder", value=False,
                                   info="Off: only the repeat folder (<N>_<dataset name>) is the dataset; everything "
                                        "else is left out of the counts, checks and Improvements. On: the scan, the "
                                        "sort and the selection also see those pictures, and the ones you keep in step 5 "
                                        "join the repeat folder. It switches off again when a launch script is saved.")
            s2_md = gr.Markdown()
            with gr.Row(elem_classes=["ls-bar"]):
                s2_skip = gr.Button("Skip", size="sm", scale=0, min_width=100)
                s2_scan = gr.Button("Scan the pictures", variant="primary", size="sm", scale=0, min_width=170)
                s2_show = gr.Radio(["Not usable first", "All"], value="Not usable first", show_label=False,
                                   container=False, scale=0, min_width=260, elem_classes=["ds-seg"])
            s2_grid = gr.HTML()
            with gr.Row(elem_classes=["ls-bar"]):
                s2_move = gr.Button("Move not usable to _reserve", size="sm", scale=0, min_width=230)
                s2_next = gr.Button("Continue →", variant="primary", size="sm", scale=0, min_width=140)
            s2_msg = gr.Markdown()
        with gr.Column(visible=False, elem_id="ds_s3") as steps[3]:
            gr.Markdown("### Set up the dataset")
            with gr.Row(equal_height=False):
                with gr.Column(scale=3, elem_classes=["ls-card"]):
                    s3_types = gr.CheckboxGroup(choices=_types, label="LoRA types (one or more)")
                    gr.Markdown("Several types: sorting uses the strictest size, step 7 writes one caption set per caption "
                                "style, step 9 one launch script per type.", elem_classes=["ls-note"])
                    s3_purpose = gr.Radio(F.PURPOSES, value="character", label="Purpose")
                    s3_words = gr.Textbox(label=F.CONCEPT_ASK["object"], visible=False,
                                          info="These words go into Never describe in step 7: the captions name the "
                                               "trigger word instead, so it learns what they mean.")
                    s3_wm = gr.Checkbox(label="Watermarks are already removed",
                                        info="Unticked: the detailed sort also looks for watermarks, stamps and site names "
                                             "and offers a crop in step 6.")
                    s3_trig = gr.Textbox(label="Trigger word (first word of every caption)")
                    with gr.Row(elem_classes=["ls-bar", "ds-bottom"]):
                        s3_next = gr.Button("Continue →", variant="primary", size="sm", scale=0, min_width=160)
                    s3_msg = gr.Markdown()
                with gr.Column(scale=2):
                    s3_pics = gr.HTML()
        with gr.Column(visible=False, elem_id="ds_s4") as steps[4]:
            gr.Markdown("### Sort with Ollama" if FT.have("ollama") else "### Sort")
            gr.Markdown("Pass 1 is quick (no AI): too blurry, too small for your LoRA types, duplicates. Pass 2 looks at "
                        "each picture with the Ollama model: one clear subject, framing, outfit, place, quality, "
                        "watermarks and how much it shows of the subject. Pictures that hide the subject or show "
                        "almost nothing of it are left out; a part with a feature (tail, horns, a tattoo) is kept as a "
                        "detail shot, plain hands or feet are not. Step 5 lets you overrule any of it with pins.",
                        elem_classes=["ls-note"])
            if not FT.have("ollama"):
                gr.Markdown("**Pass 2 · detailed (Ollama missing)**: only the quick pass runs, so framing, outfits, places "
                            "and watermarks are not known (step 5 then picks by quality only).",
                            elem_classes=["ls-missing-note"])
            with gr.Row(elem_classes=["ls-bar"]):
                s4_go = gr.Button("Sort" if FT.have("ollama") else "Sort (quick pass only)", variant="primary",
                                  size="sm", scale=0, min_width=120 if FT.have("ollama") else 200)
                s4_stop = gr.Button("Stop", variant="stop", size="sm", scale=0, min_width=90)
            s4_status = gr.Markdown()
            s4_box = gr.HTML()
            with gr.Row(elem_classes=["ls-bar", "ds-bottom"]):
                s4_next = gr.Button("Continue to Select →", variant="primary", size="sm", scale=0, min_width=220)
        with gr.Column(visible=False, elem_id="ds_s5") as steps[5]:
            gr.Markdown("### Select the best pictures")
            gr.Markdown("**📌 Always** keeps a picture whatever the rules say (a unique detail the sort missed); "
                        "**⛔ Never** leaves one out (useless content the sort kept). Your pins are saved at once, stay "
                        "after a new sort and count in the automatic improvement too.", elem_classes=["ls-note"])
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, elem_classes=["ls-card"]):
                    s5_sug = gr.HTML()
                    with gr.Row():
                        s5_keep = gr.Number(label="Keep", value=40, precision=0, minimum=1, scale=1)
                        s5_use = gr.Button("Use the suggestion", size="sm", scale=0, min_width=170)
                    s5_mode = gr.Radio([("Safe: the cleanest pictures first", "safe"),
                                        ("Varied: more close-ups, angles, outfits and details", "varied")],
                                       value="safe", label="How to pick", elem_id="ds_s5mode")
                with gr.Column(scale=1, elem_classes=["ls-card"]):
                    s5_mix = gr.HTML()
            with gr.Row(elem_classes=["ls-bar"]):
                s5_all = gr.Button("Tick all", size="sm", scale=0, min_width=100)
                s5_none = gr.Button("Untick all", size="sm", scale=0, min_width=110)
            s5_gridh = gr.HTML()
            s5_sel = gr.Textbox(visible=False)
            with gr.Row(elem_classes=["ls-bar", "ds-bottom"]):
                s5_go = gr.Button("Confirm selection →", variant="primary", size="sm", scale=0, min_width=200)
                s5_back = gr.Button("Move all back from _reserve", size="sm", scale=0, min_width=230)
            s5_msg = gr.Markdown()
        with gr.Column(visible=False, elem_id="ds_s6") as steps[6]:
            gr.Markdown("### Watermarks and size")
            with gr.Accordion("Watermarks", open=True):
                show("p_wm")
            s6_sum = gr.HTML()
            with gr.Row(elem_classes=["ls-bar"]):
                s6_max = gr.Number(label="Longest side at most (px)", value=1536, precision=0, minimum=256, scale=0,
                                   min_width=220)
                s6_restore = gr.Button("Put the originals back", size="sm", scale=0, min_width=200)
            with gr.Row(equal_height=False):
                with gr.Column(scale=3):
                    gr.Markdown("**Size, per picture:** what will happen to each picture; change it in the *Do* "
                                "column, or click a picture to crop it yourself.", elem_classes=["ls-note"])
                    s6_table = gr.HTML(elem_id="ds_s6table")
                with gr.Column(scale=2, elem_classes=["ls-card"], visible=False) as s6_crop:
                    s6_cname = gr.Markdown()
                    s6_img = gr.Image(type="pil", interactive=False, height=380, show_label=False)
                    s6_aspect = gr.Radio(["free", "3:4", "1:1", "4:3"], value="free", label="Shape")
                    with gr.Row():
                        s6_l = gr.Slider(0, 100, value=0, step=1, label="Left %")
                        s6_r = gr.Slider(0, 100, value=100, step=1, label="Right %")
                    with gr.Row():
                        s6_t = gr.Slider(0, 100, value=0, step=1, label="Top %")
                        s6_b = gr.Slider(0, 100, value=100, step=1, label="Bottom %")
                    s6_res = gr.Markdown()
                    with gr.Row():
                        s6_reset = gr.Button("Reset", size="sm")
                        s6_person = gr.Button("Around the person", size="sm")
                        s6_use = gr.Button("Use this crop", variant="primary", size="sm")
                    s6_rel = gr.State("")
            with gr.Row(elem_classes=["ls-bar"], elem_id="s6_bottom"):
                s6_wm_reco = gr.Button("Recommended for the undecided", size="sm", scale=0, min_width=250)
                s6_wm_apply = gr.Button("Apply the watermark choices", size="sm", scale=0, min_width=240)
                s6_wm_done = gr.Button("Watermarks are done", size="sm", scale=0, min_width=190)
                s6_go = gr.Button("Resize →", variant="primary", size="sm", scale=0, min_width=220)
            s6_msg = gr.Markdown()
        with gr.Column(visible=False, elem_id="ds_s7") as steps[7]:
            gr.Markdown("### Tag every picture")
            gr.Markdown("**Tags** (Illustrious, Pony, SDXL anime, Anima) are written by the **WD tagger** "
                        "(SmilingWolf/wd-eva02-large-tagger-v3, run by sd-scripts; the first time it downloads about "
                        "1.2 GB). Only general tags: no character names, so the trigger word stands for them; the "
                        "*Never describe* list and the consistency check apply as usual. **Sentences** (FLUX, Qwen-Image, "
                        "Z-Image, …) are written by Ollama.", elem_classes=["ls-note"])
            if not FT.have("ollama"):
                gr.Markdown("Ollama is not installed: tag sets work, sentence captions do not. Write those yourself in "
                            "**8 Review** (click a picture), or put `.txt` files next to the pictures. "
                            f"Ollama: [{FT.ollama_guide()}]({FT.ollama_guide()}), then Settings › Start and network.",
                            elem_classes=["ls-missing-note"])
            s7_rec = gr.HTML(elem_id="ds_s7rec")
            s7_sets = gr.HTML()
            with gr.Row(elem_classes=["ls-bar"]):
                s7_both = gr.Button("Write the captions", variant="primary", size="sm", scale=0, min_width=190)
                s7_missing = gr.Button("Only pictures without a caption", size="sm", scale=0, min_width=240)
                s7_stop = gr.Button("Stop", variant="stop", size="sm", scale=0, min_width=90)
            s7_out = gr.Markdown()
            with gr.Accordion("Caption settings: length, detail, never describe, try on 3 pictures", open=False):
                show("p_caption")
            with gr.Column(elem_classes=["ls-card"]) as s7_cons:
                gr.Markdown("#### Consistency, checked while tagging")
                show("p_cons")
                with gr.Row(elem_classes=["ls-bar"]) as s7_mirror_row:
                    s7_mirror = gr.Button("Apply to the other caption set too", size="sm", scale=0, min_width=260)
                    gr.Markdown("What was taken out of IMG.txt is taken out of the second set as well.",
                                elem_classes=["ls-note"])
                s7_mmsg = gr.Markdown()
            with gr.Row(elem_classes=["ls-bar", "ds-bottom"]):
                s7_next = gr.Button("Continue to Review →", variant="primary", size="sm", scale=0, min_width=220)
        with gr.Column(visible=False, elem_id="ds_s8") as steps[8]:
            gr.Markdown("### Review")
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, elem_classes=["ls-card"]):
                    gr.Markdown("**The folder**")
                    s8_tree = gr.HTML()
                    gr.Markdown("**Checks**")
                    s8_checks = gr.HTML()
                with gr.Column(scale=3):
                    s8_show = gr.Radio(["Problems first", "All"], value="Problems first", show_label=False,
                                       container=False, elem_classes=["ds-seg"])
                    s8_grid = gr.HTML()
                with gr.Column(scale=2, elem_classes=["ls-card"]):
                    s8_img = gr.Image(type="filepath", interactive=False, height=360, show_label=False)
                    s8_name = gr.Markdown("*Click a picture to see and edit its caption.*")
                    s8_cap1 = gr.Textbox(label="Caption", lines=4)
                    s8_cap2 = gr.Textbox(label="Second caption set", lines=4, visible=False)
                    s8_save = gr.Button("Save caption changes", size="sm")
                    s8_msg = gr.Markdown()
                    s8_path = gr.State("")
            with gr.Row(elem_classes=["ls-bar", "ds-bottom"]):
                s8_next = gr.Button("Continue to Launch script →", variant="primary", size="sm", scale=0, min_width=260)
        with gr.Column(visible=False, elem_id="ds_s9") as steps[9]:
            gr.Markdown("### Launch script")
            s9_type = gr.Radio(choices=[], label="LoRA type", elem_id="ds_s9type")
            with gr.Row(elem_classes=["ls-card"], visible=False, equal_height=True) as s9_dl_row:
                s9_dl_note = gr.Markdown()
                s9_dl_btn = gr.Button("⬇ Download from Hugging Face", variant="primary", size="sm",
                                      scale=0, min_width=260)
                s9_dl_stop = gr.Button("Stop", variant="stop", size="sm", scale=0, min_width=80)
            s9_dl_md = gr.Markdown()
            show("s9_head")
            with gr.Column(elem_classes=["ls-card"], elem_id="ds_s9plan"):
                gr.Markdown("**Your plan**")
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        s9_calc = gr.HTML()
                    with gr.Column(scale=1):
                        s9_eff = gr.HTML()
            with gr.Column(elem_classes=["ls-card"], elem_id="ds_s9set"):
                gr.Markdown("**Settings** · your value, with the recommendation and what it means below it. A change is "
                            "written into the script text when you leave the field; **Save** keeps it.",
                            elem_classes=["ls-note"])
                s9_vals = []
                with gr.Row():
                    for flag, label, why in SETTINGS[:4]:
                        s9_vals.append(gr.Textbox(label=label, info=why, scale=1))
                with gr.Row():
                    for flag, label, why in SETTINGS[4:]:
                        s9_vals.append(gr.Textbox(label=label, info=why, scale=1))
                s9_amsg = gr.Markdown()
            with gr.Accordion("More settings: optimizer, scheduler, saving, model files, advanced, repeat folder",
                              open=False):
                show("s9_fields")
                show("s9_opts")
                with gr.Accordion("Repeat folder", open=False):
                    show("p_repeat")
            with gr.Accordion("Why these settings: notes and checks before training", open=False):
                show("s9_why")
            with gr.Accordion("The script text", open=False):
                show("s9_text")
            show("s9_save")
            show("script")                       # the (now empty) holder of the parts above


    # ------------------------------------------------------------------ navigation
    nav_outs = [p_new, p_edit, p_imp, p_flow, b_new, b_edit, b_imp] + [steps[n] for n in FLOW] + [f_bar, f_step]

    def go(cur, **kw):
        s = dict(cur or {})
        s.update(kw)
        return s

    def load_new(s):
        if s.get("view") != "new":
            return gr.update()
        return new_list_html(s.get("src", ""))

    def edit_list(s, find=""):
        rows = F.overview(S.list_lora_dirs())
        find = (find or "").strip().lower()
        out = []
        for r in rows:
            if find and find not in r["ds"].lower():
                continue
            on = " ds-frow-on" if r["ds"] == s.get("ds") else ""
            out.append(f'<button class="ds-erow{on}" onclick="{send("edit|" + r["ds"])}"><b>{esc(r["ds"])}</b>'
                       f'<span class="ls-note">{r["n"]} · next: {r["next"]} {esc(r["label"])}</span></button>')
        return '<div class="ds-elist">' + ("".join(out) or '<p class="ls-empty">No datasets.</p>') + "</div>"

    def edit_steps(ds):
        if not ds:
            return '<p class="ls-empty">Pick a dataset on the left.</p>', ""
        m = F.meta(ds)
        labels = F.type_labels()
        head = (f'<div class="ds-ehead"><b>{esc(ds)}</b> · {esc(" + ".join(labels.get(t, t) for t in m["types"]) or "type not set")}'
                f' · {esc(F.PURPOSE_LABEL.get(m["purpose"], m["purpose"]))} · {len(F.images(ds))} pictures</div>')
        sts = F.step_states(ds)
        nxt = F.next_step(ds)
        rows = []
        for n, key, label in F.STEPS[1:]:
            s, text = sts.get(key, ("todo", ""))
            chip = {"done": "ok", "warn": "warn", "optional": ""}.get(s, "")
            miss = key == "tags" and not FT.have("ollama") and any(x["style"] != "tags" for x in F.caption_sets(ds))
            rows.append(f'<div class="ds-srow{" ds-srow-next" if n == nxt else ""}{" ds-missing" if miss else ""}"><span class="ds-snum">{n}</span>'
                        f'<b>{esc(FT.label(label, "ollama") if miss else label)}</b>'
                        f'<span class="ls-chip {chip}">{esc(text)}</span><span class="ds-sp"></span>'
                        f'<button class="tr-btn tr-small" onclick="{send(f"open|{n}")}">{"Next: open" if n == nxt else "Open"}</button></div>')
        return head, '<div class="ds-slist">' + "".join(rows) + "</div>"

    def load_edit(s, find):
        if s.get("view") != "edit":
            return gr.update(), gr.update(), gr.update()
        h, steps_ = edit_steps(s.get("ds"))
        return edit_list(s, find), h, steps_

    def load_imp(s):
        if s.get("view") != "imp":
            return gr.update(), gr.update()
        dss = S.list_lora_dirs()
        cards = []
        for ds in dss:
            try:
                items = F.improvements(ds)
            except Exception as e:                       # noqa: BLE001
                items = [("error", str(e))]
            if not items:
                continue
            body = "".join(f'<div class="ds-improw"><span class="ls-chip warn">{esc(t)}</span><span>{esc(x)}</span></div>'
                           for t, x in items)
            cards.append(f'<div class="ls-card ds-impcard"><div class="ds-imph"><label class="ds-tick ds-imptick">'
                         f'<input type="checkbox" data-ds="{esc(ds)}"><b>{esc(ds)}</b></label>'
                         f'<button class="tr-btn tr-small" onclick="{send("edit|" + ds)}">Open in Edit existing</button></div>{body}</div>')
        head = f'<div class="rs-big">{len(cards)} of {len(dss)} datasets could be better</div>'
        return head, "".join(cards) or '<p class="ls-empty">Nothing to improve that is known.</p>'

    def sync_key(s):
        if s.get("view") == "flow" and s.get("step") in (6, 7, 9) and s.get("ds"):
            m = F.meta(s["ds"])
            t = (s.get("tkey") if s.get("tkey") in m["types"] else (m["types"][0] if m["types"] else ""))
            return f"{s['ds']}|{t}|{m['purpose']}"
        return gr.update()

    def set_cs(key):
        ds, t, pu = (key.split("|") + ["", "", ""])[:3]
        S.set_choice(pu or "character", S.CHOICE.get("priority"))     # recommendations and captions follow the purpose
        ch = S.list_lora_dirs()
        dsu = gr.update(choices=ch, value=ds if ds in ch else None)
        tu = gr.update(choices=S.type_choices(), value=t) if t else gr.update()   # also types added since the start
        return dsu, dsu, tu, tu, gr.update(value=pu or "character"), gr.update(value=pu or "character")

    def after_nav(ev):
        """One round trip redraws the whole page for the state: the nav, the step that is open and its contents."""
        return ev.then(refresh_all, [st, e_find, s8_show, UI["cs_text"], UI["oc"]["leave"]], all_outs,
                       show_progress="hidden")

    # ---- step loaders
    def s3_load(s):
        if s.get("view") != "flow" or s.get("step") != 3 or not s.get("ds"):
            return [gr.update()] * 6
        m = F.meta(s["ds"])
        imgs = F.images(s["ds"])
        strip = "".join(f'<img src="{thumb(p, 110)}" alt="">' for p in imgs[:24])
        pics = (f'<div class="ls-card"><b>All {len(imgs)} pictures</b><div class="ds-strip">{strip}</div>'
                + (f'<div class="ls-note">… {len(imgs) - 24} more</div>' if len(imgs) > 24 else "") + "</div>")
        return (m["types"], m["purpose"], m["wm_removed"], m["trigger"] or s["ds"], pics,
                gr.update(value=m.get("concept_words") or "", visible=m["purpose"] in F.CONCEPT_KINDS,
                          label=F.CONCEPT_ASK.get(m["purpose"], F.CONCEPT_ASK["object"])))

    def s4_load(s):
        if s.get("view") != "flow" or s.get("step") != 4 or not s.get("ds"):
            return gr.update(), gr.update()
        return s4_html(s["ds"]), ""

    def s5_load(s):
        if s.get("view") != "flow" or s.get("step") != 5 or not s.get("ds"):
            return [gr.update()] * 5
        n, rng, why = F.suggest_size(s["ds"])
        mix, g = s5_grid(s["ds"], n)
        mode = F.meta(s["ds"]).get("select_mode") or F.batch_settings().get("select_mode") or "safe"
        return s5_suggest_html(s["ds"], n, rng, why), n, mix, g, mode

    def s6_load(s):
        if s.get("view") != "flow" or s.get("step") != 6 or not s.get("ds"):
            return [gr.update()] * 5
        saved = (F._load(s["ds"], "resize.json", {}) or {}).get("max_side")
        ms = int(saved or F.default_max_side(F.meta(s["ds"])["types"]))
        sm, tb, btn = s6_html(s["ds"], ms)
        return ms, sm, tb, btn, gr.update(visible=False)

    def s7_load(s, leave):
        if s.get("view") != "flow" or s.get("step") != 7 or not s.get("ds"):
            return [gr.update()] * 5
        m = F.meta(s["ds"])
        purpose = m["purpose"]
        if S.CAP is not None:
            S.CAP.PURPOSE["value"] = purpose
            S.CAP.PURPOSE["words"] = m.get("concept_words") or ""
        return (s7_sets_html(s["ds"]), gr.update(visible=len(F.caption_sets(s["ds"])) > 1), s7_rec_html(purpose),
                gr.update(visible=purpose == "character"), leave_update(purpose, leave, m.get("concept_words") or ""))

    def s8_load(s, show_):
        if s.get("view") != "flow" or s.get("step") != 8 or not s.get("ds"):
            return [gr.update()] * 4
        t, c, g = s8_render(s["ds"], show_)
        sets = F.caption_sets(s["ds"])
        return t, c, g, gr.update(visible=len(sets) > 1, label=f"Caption IMG{sets[1]['ext']}" if len(sets) > 1 else "")

    def s9_load(s):
        if s.get("view") != "flow" or s.get("step") != 9 or not s.get("ds"):
            return gr.update(), gr.update(), gr.update(), gr.update()
        m = F.meta(s["ds"])
        labels = F.type_labels()
        ch = [(FT.label(labels.get(t, t), FT.trainer_for(t)), t) for t in m["types"]]
        cur = s.get("tkey") if s.get("tkey") in m["types"] else (m["types"][0] if m["types"] else None)
        return (gr.update(choices=ch, value=cur, visible=len(ch) > 0), s9_queue_vis(s)) + s9_dl_vis(s)

    def s9_dl_vis(s):
        """The download card when model files of the type are missing and Hugging Face has them (hfdownload.py)."""
        m = F.meta(s["ds"]) if s.get("ds") else {"types": []}
        cur = s.get("tkey") if s.get("tkey") in m["types"] else (m["types"][0] if m["types"] else "")
        try:
            import hfdownload as HD
            import sd_settings as SS
            miss = HD.missing(cur) if cur else []
        except Exception:                                  # noqa: BLE001
            miss = []
        if not miss:
            return gr.update(visible=False), gr.update(value="")
        k = HD.known(cur)
        rows = ", ".join(f"**{SS.slot_label(f)}** (`{Path(k[f][1]).name}`)" for f in miss)
        tok = "" if SS.read_hf_token() else " A Hugging Face token (Settings › Hugging Face token) is needed for gated models."
        return (gr.update(visible=True), gr.update(value=f"Missing on this computer: {rows}. They can be downloaded from "
                                                         f"Hugging Face into your models folder.{tok}"))

    def s9_dl_fn(s):
        import hfdownload as HD
        m = F.meta(s["ds"]) if s.get("ds") else {"types": []}
        cur = s.get("tkey") if s.get("tkey") in m["types"] else (m["types"][0] if m["types"] else "")
        if not cur:
            yield "Pick the LoRA type first."
            return
        yield from HD.fetch_missing(cur)

    def s9_queue_vis(s):
        """'Add to the queue' only when the trainer of the type is installed (the script is made either way)."""
        m = F.meta(s["ds"]) if s.get("ds") else {"types": []}
        cur = s.get("tkey") if s.get("tkey") in m["types"] else (m["types"][0] if m["types"] else "")
        return gr.update(visible=bool(cur) and FT.can_train(cur))

    all_outs = (nav_outs + [n_list] + [e_list, e_head, e_steps] + [i_head, i_html, i_what] + [s2_md, s2_grid, s2_show, s2_extra]
                + [s3_types, s3_purpose, s3_wm, s3_trig, s3_pics, s3_words] + [s4_box, s4_status] + [s5_sug, s5_keep, s5_mix, s5_gridh, s5_mode]
                + [s6_max, s6_sum, s6_table, s6_go, s6_crop] + [s7_sets, s7_mirror_row, s7_rec, s7_cons, UI["oc"]["leave"]] + [s8_tree, s8_checks, s8_grid, s8_cap2]
                + [s9_type, UI["cs_queue_btn"], s9_dl_row, s9_dl_note, sync, s9_calc, s9_eff] + s9_vals)

    def refresh_all(s, find, show8, text, leave):
        out = list(nav_updates(s))
        out.append(load_new(s))
        out += list(load_edit(s, find))
        out += list(load_imp(s))
        out.append(batch_what() if s.get("view") == "imp" else gr.update())
        out += list(s2_load(s))
        out += list(s3_load(s))
        out += list(s4_load(s))
        out += list(s5_load(s))
        out += list(s6_load(s))
        out += list(s7_load(s, leave))
        out += list(s8_load(s, show8))
        out += list(s9_load(s))
        out.append(sync_key(s))
        out += list(s9_calc_fn(s, text))
        return out

    # ---- the router for clicks inside the HTML parts
    def route(v, s):
        """Clicks that change the page go through navbox (the whole page is redrawn); the others stay here."""
        act, _, rest = (v or "").partition("|")
        arg = rest.rsplit("|", 1)[0] if "|" in rest else rest
        s = dict(s or {})
        now = str(time.time())
        keep = gr.update()
        if act == "edit":
            return go(s, view="edit", ds=arg, mode="edit"), now, keep, keep, keep
        if act in ("open", "step") and s.get("ds"):
            try:
                n = int(arg)
            except ValueError:
                return s, keep, keep, keep, keep
            return go(s, view="flow", step=n), now, keep, keep, keep
        if act == "folder":
            return go(s, src=arg), keep, keep, keep, v
        if act in ("pick", "rmode") and s.get("step") == 6:      # only the part that the click is for runs
            return s, keep, v, keep, keep
        if act in ("pick", "fix") and s.get("step") == 8:
            return s, keep, keep, v, keep
        if act == "pin" and s.get("ds"):                 # step 5: saved at once; the cell already shows it
            state, _, path = arg.partition("|")
            F.set_pin(s["ds"], path, state)
            return s, keep, keep, keep, keep
        return s, keep, keep, keep, keep

    box6 = gr.Textbox(visible=False)
    box8 = gr.Textbox(visible=False)
    box1 = gr.Textbox(visible=False)
    box.change(route, [box, st], [st, navbox, box6, box8, box1], show_progress="hidden")
    navbox.change(refresh_all, [st, e_find, s8_show, UI["cs_text"], UI["oc"]["leave"]], all_outs, show_progress="hidden")
    for b, view in ((b_new, "new"), (b_edit, "edit"), (b_imp, "imp")):
        after_nav(b.click(lambda s, v=view: go(s, view=v, mode=v if v != "imp" else s.get("mode", "new")), st, st,
                          show_progress="hidden"))
    after_nav(app.load(lambda s: s, st, st))
    e_find.change(lambda s, f: edit_list(s, f), [st, e_find], e_list, show_progress="hidden")
    i_ref.click(lambda s: list(load_imp(s)) + [batch_what()], st, [i_head, i_html, i_what])
    i_all.click(None, None, None, js="() => document.querySelectorAll('#ds_imp .ds-imptick input').forEach(function(i){ i.checked = true; })")
    i_none.click(None, None, None, js="() => document.querySelectorAll('#ds_imp .ds-imptick input').forEach(function(i){ i.checked = false; })")
    i_go.click(lambda v: F.batch_start(json.loads(v or "[]")), i_sel, i_msg,
               js="(x) => [JSON.stringify(Array.from(document.querySelectorAll('#ds_imp .ds-imptick input:checked'))"
                  ".map(function(e){ return e.dataset.ds; }))]")
    i_stop.click(lambda: F.batch_stop(), None, i_msg, queue=False)
    i_last = gr.State("")

    def batch_tick(prev):
        h = batch_html()
        return (gr.update(), prev) if h == prev else (h, h)
    import ticker
    ticker.timer(2.0, batch_tick, i_last, [i_prog, i_last], show_progress="hidden")

    # sdgui's step parts follow the dataset and type of the open step
    ds_ev = sync.change(set_cs, sync, [UI["cs_ds"], UI["pp_ds"], UI["cs_type"], UI["pp_type"], UI["cs_purpose"],
                                       UI["pp_purpose"]], show_progress="hidden")
    def s9_calc_fn(s, text):
        ds = s.get("ds")
        if not ds or s.get("step") != 9:
            return [gr.update()] * (2 + len(s9_vals))
        t = s.get("tkey") if s.get("tkey") in F.meta(ds)["types"] else (F.meta(ds)["types"] or [""])[0]
        a, b, vals = calc(ds, t, text)
        return [a, b] + list(vals)
    UI["ds_chain"](ds_ev).then(lambda s: F.meta(s["ds"])["trigger"] if s.get("ds") else gr.update(), st,
                               UI["oc"]["trigger"]).then(s9_calc_fn, [st, UI["cs_text"]], [s9_calc, s9_eff] + s9_vals,
                                                          concurrency_id="s9calc", concurrency_limit=1)

    # ---- 1 Folder
    def pick_src(s):
        src = s.get("src", "")
        return src, (F.safe_name(src.replace("_batch", "").replace("_raw", "")) if src else gr.update())
    box1.change(lambda s, v: pick_src(s) + (new_list_html(s.get("src", "")),) if (v or "").startswith("folder|")
                else (gr.update(), gr.update(), gr.update()), [st, box1], [n_src, n_name, n_list], show_progress="hidden") \
        .then(new_tree, [n_name, n_rep, n_src, n_out], n_tree)
    for c in (n_name, n_rep, n_out):
        c.change(new_tree, [n_name, n_rep, n_src, n_out], n_tree, show_progress="hidden")

    def create(s, src, name, rep, outside):
        ds, msg = F.create(src, name, rep, outside)
        if not ds:
            return s, msg
        return go(s, view="flow", mode="new", ds=ds, step=2, src=""), msg
    after_nav(n_go.click(create, [st, n_src, n_name, n_rep, n_out], [st, n_msg]))

    # ---- Edit existing: add pictures from a folder
    def add_pics(s, folder, sub):
        ds = s.get("ds")
        if not ds:
            return "Pick a dataset first."
        folder = os.path.expanduser((folder or "").strip())
        if not Path(folder).is_dir():
            return f"`{folder}` is not a folder."
        import picker
        rows = S.dataset_stats(ds)["rows"]
        if not rows:
            return "This dataset has no repeat folder yet."
        res = picker.scan(folder, bool(sub), F.images(ds), F.min_side(F.meta(ds)["types"]), F.meta(ds)["purpose"],
                          "", "", False, lambda p: None, lambda p, d: None)
        pick = [e["path"] for e in res["pics"] if e["pick"]]
        copied, errs = picker.copy_pictures(pick, rows[0]["folder"])
        F.ensure_originals(ds, copied)
        skipped = len(res["pics"]) - len(pick)
        return (f"✔ {len(copied)} pictures copied into `{rows[0]['folder'].name}/`" + (f"; {skipped} left out (too small, "
                "blurry or already in the dataset)" if skipped else "") + ". Sort and select again (steps 4 and 5).")
    after_nav(e_add_btn.click(add_pics, [st, e_add_dir, e_add_sub], e_add_msg).then(lambda s: s, st, st))

    # ---- 2 Usable scan
    def s2_do(s):
        ds = s.get("ds")
        sc = F.usable_scan(ds, F.meta(ds)["types"])
        md, g = s2_render(ds, sc, "Not usable first")
        return md, g
    s2_scan.click(s2_do, st, [s2_md, s2_grid])

    def s2_extra_fn(s, on):
        ds = s.get("ds")
        if not ds:
            return gr.update(), gr.update(), gr.update()
        F.set_include_extra(ds, on)
        sc = F._load(ds, "scan.json")
        if sc:
            md, g = s2_render(ds, sc, "Not usable first")
        else:
            md, g = (f"**{len(F.candidates(ds))} pictures.** Press **Scan the pictures**.", "")
        note = (" Scan again to check the pictures from outside the repeat folder." if on and sc else "")
        return md + (f'<div class="ls-note">{note}</div>' if note else ""), g, s
    s2_extra.input(s2_extra_fn, [st, s2_extra], [s2_md, s2_grid, st])
    s2_show.change(lambda s, v: s2_render(s["ds"], F._load(s["ds"], "scan.json"), v) if s.get("ds") and F._load(s["ds"], "scan.json")
                   else (gr.update(), gr.update()), [st, s2_show], [s2_md, s2_grid])

    def s2_move_fn(s):
        sc = F._load(s["ds"], "scan.json")
        if not sc:
            return "Scan first."
        main = {str(p) for p in F.images(s["ds"])}          # pictures outside the repeat folder stay where they are
        bad = [p for p, e in sc["pics"].items() if e["bad"] and Path(p).exists() and p in main]
        if not bad:
            return "Nothing to move (only pictures in the repeat folder are moved)."
        n, err = F.move_to_reserve(s["ds"], bad)
        return f"✔ {n} moved to `_reserve/`." + (f" {'; '.join(err[:3])}" if err else "")
    s2_move.click(s2_move_fn, st, s2_msg).then(s2_do, st, [s2_md, s2_grid])
    for b in (s2_next, s2_skip):
        after_nav(b.click(lambda s: go(s, step=3), st, st))

    # ---- 3 Setup
    def s3_save(s, types, purpose, wm, trig, words):
        m = F.meta(s["ds"])
        # what you picked is kept even when something is still missing: the page is drawn again after Continue,
        # and without this it went back to the saved purpose (and hid the words field), so step 3 just repeated
        if types:
            m["types"] = list(types)
        m.update(purpose=purpose or "character", wm_removed=bool(wm), trigger=(trig or s["ds"]).strip(),
                 concept_words=(words or "").strip())
        if not types:
            F.meta_save(s["ds"], m)
            return s, "**Pick at least one LoRA type** (the list at the top), then press Continue again."
        if purpose in F.CONCEPT_KINDS and not (words or "").strip():
            F.meta_save(s["ds"], m)
            return s, ("**Write the words for it** in the field under Purpose (for example *katana, sword*), then "
                       "press Continue again: the captions leave these words out, so the LoRA learns them.")
        m["steps"]["setup"] = {"at": time.strftime("%Y-%m-%d %H:%M")}
        F.meta_save(s["ds"], m)
        return go(s, step=4, tkey=types[0]), ""
    after_nav(s3_next.click(s3_save, [st, s3_types, s3_purpose, s3_wm, s3_trig, s3_words], [st, s3_msg]))
    s3_purpose.change(lambda p: gr.update(visible=p in F.CONCEPT_KINDS, label=F.CONCEPT_ASK.get(p, F.CONCEPT_ASK["object"])),
                      s3_purpose, s3_words, show_progress="hidden")

    # ---- 4 Sort
    def s4_run(s):
        ds = s.get("ds")
        m = F.meta(ds)
        if not m["types"]:
            yield "Set the LoRA type first (step 3).", gr.update()
            return
        url, model = S.oc_settings()
        for line in F.sort_run(ds, m["types"], m["purpose"], m["wm_removed"], url, model):
            if line == "done":
                break
            yield f"⏳ {line}", gr.update()
        yield "", s4_html(ds)
    s4_go.click(s4_run, st, [s4_status, s4_box]).then(lambda s: (bar_html(s), stepper_html(s)), st, [f_bar, f_step])
    s4_stop.click(lambda: F.STOP.set(), None, None, queue=False)
    after_nav(s4_next.click(lambda s: go(s, step=5), st, st))

    # ---- 5 Select
    s5_use.click(lambda s: F.suggest_size(s["ds"])[0], st, s5_keep).then(lambda s, k: s5_grid(s["ds"], k), [st, s5_keep],
                                                                         [s5_mix, s5_gridh])
    s5_keep.submit(lambda s, k: s5_grid(s["ds"], k), [st, s5_keep], [s5_mix, s5_gridh])

    def s5_mode_set(s, mode, k):
        m = F.meta(s["ds"])
        m["select_mode"] = mode
        F.meta_save(s["ds"], m)
        return s5_grid(s["ds"], k)
    s5_mode.input(s5_mode_set, [st, s5_mode, s5_keep], [s5_mix, s5_gridh])
    s5_all.click(None, None, None, js="() => dsTickAll(true)")
    s5_none.click(None, None, None, js="() => dsTickAll(false)")

    def s5_apply(s, sel):
        try:
            keep = json.loads(sel or "[]")
        except ValueError:
            keep = []
        if not keep:
            return s, "Nothing is ticked."
        return go(s, step=6), F.apply_select(s["ds"], keep)
    after_nav(s5_go.click(s5_apply, [st, s5_sel], [st, s5_msg], js=TICKED_JS.replace("(x)", "(s, x)").replace("=> [", "=> [s, ")))
    after_nav(s5_back.click(lambda s: (s, "✔ {} pictures put back.".format(F.back_from_reserve(s["ds"])[0])), st, [st, s5_msg]))

    # ---- 6 Resize
    def s6_refresh(s, ms):
        return s6_html(s["ds"], ms)
    s6_max.submit(s6_refresh, [st, s6_max], [s6_sum, s6_table, s6_go])

    def crop_open(s, v, ms):
        act, _, rest = (v or "").partition("|")
        if s.get("step") != 6:
            return [gr.update()] * 9
        ds = s["ds"]
        r = F.root(ds)
        if act == "pick":
            path = rest.rsplit("|", 1)[0]
            rel = str(Path(path).relative_to(r))
        elif act == "rmode":
            rel, _, mode = rest.rsplit("|", 1)[0].rpartition("|")
            over = F.resize_overrides(ds)
            if mode in ("keep", "scale", "none", "upscale"):
                if mode in ("scale", "none", "upscale"):        # the suggestion again
                    over.pop(rel, None)
                else:
                    over[rel] = {"mode": mode}
                F.resize_save_over(ds, over, ms)
                return [gr.update(visible=False)] + [gr.update()] * 8
        else:
            return [gr.update()] * 9
        o = F.resize_overrides(ds).get(rel) or {}
        auto = None if o.get("box") else F.person_box(str(r / rel))
        box = o.get("box") or auto or [0, 0, 1, 1]
        img = F.crop_preview(str(r / rel), box)
        return (gr.update(visible=True), f"**Crop {Path(rel).name}**", img, rel, round(box[0] * 100), round(box[2] * 100),
                round(box[1] * 100), round(box[3] * 100), "suggested: around the person" if auto else "")
    crop_outs = [s6_crop, s6_cname, s6_img, s6_rel, s6_l, s6_r, s6_t, s6_b, s6_res]
    box6.change(crop_open, [st, box6, s6_max], crop_outs, show_progress="hidden").then(
        lambda s, v, ms: s6_html(s["ds"], ms) if s.get("step") == 6 and (v or "").startswith("rmode|") else
        (gr.update(), gr.update(), gr.update()), [st, box6, s6_max], [s6_sum, s6_table, s6_go], show_progress="hidden")

    def crop_box(s, rel, l, r_, t, b, aspect):
        if not rel:
            return gr.update(), ""
        p = F.root(s["ds"]) / rel
        try:
            from PIL import Image, ImageOps
            with Image.open(p) as im:
                w, h = ImageOps.exif_transpose(im).size
        except Exception:                                # noqa: BLE001
            return gr.update(), ""
        bx = F.fit_box(w, h, l / 100, t / 100, r_ / 100, b / 100, aspect)
        cw, ch = int((bx[2] - bx[0]) * w), int((bx[3] - bx[1]) * h)
        return F.crop_preview(str(p), bx), f"result {cw}×{ch}"
    crop_in = [st, s6_rel, s6_l, s6_r, s6_t, s6_b, s6_aspect]
    for c in (s6_l, s6_r, s6_t, s6_b):
        c.release(crop_box, crop_in, [s6_img, s6_res])
    s6_aspect.change(crop_box, crop_in, [s6_img, s6_res])
    def crop_person(s, rel, aspect):
        if not rel:
            return [gr.update()] * 6
        p = F.root(s["ds"]) / rel
        bx = F.person_box(str(p), aspect, run=True)
        if not bx:
            return [gr.update()] * 4 + [gr.update(), "No person found (or the picture tools are not installed: "
                                                    "Settings › Trainers and GPU)."]
        return (round(bx[0] * 100), round(bx[2] * 100), round(bx[1] * 100), round(bx[3] * 100),
                F.crop_preview(str(p), bx), "suggested: around the person")
    s6_person.click(crop_person, [st, s6_rel, s6_aspect], [s6_l, s6_r, s6_t, s6_b, s6_img, s6_res])
    s6_reset.click(lambda: (0, 100, 0, 100, "free"), None, [s6_l, s6_r, s6_t, s6_b, s6_aspect]).then(
        crop_box, [st, s6_rel, s6_l, s6_r, s6_t, s6_b, s6_aspect], [s6_img, s6_res])

    def crop_use(s, rel, l, r_, t, b, aspect, ms):
        p = F.root(s["ds"]) / rel
        from PIL import Image, ImageOps
        with Image.open(p) as im:
            w, h = ImageOps.exif_transpose(im).size
        over = F.resize_overrides(s["ds"])
        over[rel] = {"mode": "crop", "box": F.fit_box(w, h, l / 100, t / 100, r_ / 100, b / 100, aspect)}
        F.resize_save_over(s["ds"], over, ms)
        return (gr.update(visible=False),) + s6_html(s["ds"], ms)
    s6_use.click(crop_use, [st, s6_rel, s6_l, s6_r, s6_t, s6_b, s6_aspect, s6_max], [s6_crop, s6_sum, s6_table, s6_go])
    after_nav(s6_restore.click(lambda s: (s, F.restore_originals(s["ds"])), st, [st, s6_msg]))
    s6_wm_apply.click(None, None, None, js="() => { var b = document.getElementById('wm_apply_btn'); if (b) b.click(); }")
    after_nav(s6_wm_done.click(lambda s: (s, F.wm_done(s.get("ds"))), st, [st, s6_msg]))
    s6_wm_reco.click(None, None, None, js="() => { var b = document.getElementById('wm_reco_btn'); if (b) b.click(); }")
    def s6_go_fn(s, ms):
        g = F.apply_resize_steps(s["ds"], int(ms or 1536))
        while True:
            try:
                line = next(g)
            except StopIteration as e:
                yield go(s, step=7), e.value
                return
            yield s, line
    after_nav(s6_go.click(s6_go_fn, [st, s6_max], [st, s6_msg]))

    # ---- 7 Tags: the first set with sdgui's caption writer settings, the others with dsflow
    oc = UI["oc"]

    def write_all(s, url, model, trig, leave, length, detail, only_missing=False):
        ds = s["ds"]
        if S.CAP is not None:
            S.CAP.PURPOSE["value"] = F.meta(ds)["purpose"]
            S.CAP.PURPOSE["words"] = F.meta(ds).get("concept_words") or ""
        sets = F.caption_sets(ds)
        url, model = (url or S.oc_settings()[0]), (model or S.oc_settings()[1])
        msg = ""
        for i, st_ in enumerate(sets):
            last = ""
            for line in F.write_set(ds, url, model, trig or ds, leave, st_["style"], length, detail, st_["ext"],
                                    only_missing=only_missing):
                last = line
                yield (msg + "\n\n" if msg else "") + line, gr.update()
            msg += ("\n\n" if msg else "") + last
        yield msg, s7_sets_html(ds)
    s7_both.click(write_all, [st, oc["url"], oc["model"], oc["trigger"], oc["leave"], oc["length"], oc["detail"]],
                  [s7_out, s7_sets])
    s7_missing.click(lambda *a: (yield from write_all(*a, only_missing=True)),
                     [st, oc["url"], oc["model"], oc["trigger"], oc["leave"], oc["length"], oc["detail"]], [s7_out, s7_sets])
    s7_stop.click(lambda: (F.STOP.set(), S.OC_STOP.update(flag=True)), None, None, queue=False)

    def mirror(s, url, model, trig):
        sets = F.caption_sets(s["ds"])
        if len(sets) < 2:
            return "There is only one caption set."
        url, model = (url or S.oc_settings()[0]), (model or S.oc_settings()[1])
        return "\n\n".join(F.mirror_kept(s["ds"], x["ext"], url, model, trig) for x in sets[1:])
    s7_mirror.click(mirror, [st, oc["url"], oc["model"], oc["trigger"]], s7_mmsg)
    UI["cc_apply_btn"].click(lambda s: s7_sets_html(s["ds"]) if s.get("ds") else gr.update(), st, s7_sets)
    after_nav(s7_next.click(lambda s: go(s, step=8), st, st))

    # ---- 8 Review
    s8_show.change(lambda s, v: s8_render(s["ds"], v), [st, s8_show], [s8_tree, s8_checks, s8_grid])

    def s8_click(s, v):
        act, _, rest = (v or "").partition("|")
        if s.get("step") != 8:
            return [gr.update()] * 6
        arg = rest.rsplit("|", 1)[0]
        ds = s["ds"]
        if act == "pick":
            p = Path(arg)
            sets = F.caption_sets(ds)
            c1 = F.read_set(p, sets[0]["ext"]) or ""
            c2 = F.read_set(p, sets[1]["ext"]) or "" if len(sets) > 1 else ""
            return str(p), f"**{p.name}**", c1, c2, str(p), ""
        if act == "fix":
            action, _, path = arg.partition("|")
            return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), F.fix(ds, action, path)
        return [gr.update()] * 6
    box8.change(s8_click, [st, box8], [s8_img, s8_name, s8_cap1, s8_cap2, s8_path, s8_msg], show_progress="hidden").then(
        lambda s, v, sh: s8_render(s["ds"], sh) if s.get("step") == 8 and (v or "").startswith("fix|") else
        (gr.update(), gr.update(), gr.update()), [st, box8, s8_show], [s8_tree, s8_checks, s8_grid], show_progress="hidden")

    def s8_save_fn(s, path, c1, c2):
        if not path:
            return "Click a picture first."
        p = Path(path)
        sets = F.caption_sets(s["ds"])
        F.set_file(p, sets[0]["ext"]).write_text((c1 or "").strip() + "\n", encoding="utf-8")
        if len(sets) > 1 and (c2 or "").strip():
            F.set_file(p, sets[1]["ext"]).write_text(c2.strip() + "\n", encoding="utf-8")
        return f"✔ Saved the caption of {p.name}."
    s8_save.click(s8_save_fn, [st, s8_path, s8_cap1, s8_cap2], s8_msg)

    def s8_done(s):
        F.mark_step(s["ds"], "review")
        return go(s, step=9)
    after_nav(s8_next.click(s8_done, st, st))

    # ---- 9 Launch script
    def s9_pick(s, t):
        return go(s, tkey=t)
    fn, ins, outs = UI["cs_retype"]
    s9_type.input(s9_pick, [st, s9_type], st).then(sync_key, st, sync).then(s9_queue_vis, st, UI["cs_queue_btn"]) \
        .then(s9_dl_vis, st, [s9_dl_row, s9_dl_note]).then(lambda: "", None, s9_dl_md)
    _mfn, _mins, _mouts = UI["cs_models"]
    s9_dl_btn.click(s9_dl_fn, st, s9_dl_md, concurrency_id="hfdl", concurrency_limit=1) \
        .then(s9_dl_vis, st, [s9_dl_row, s9_dl_note]).then(_mfn, _mins, _mouts)
    s9_dl_stop.click(lambda: __import__("hfdownload").STOP.set(), None, None, queue=False)
    # one at a time, in the order they were asked for: a slower one for the type before must not land last
    UI["cs_text"].change(s9_calc_fn, [st, UI["cs_text"]], [s9_calc, s9_eff] + s9_vals, show_progress="hidden",
                         concurrency_id="s9calc", concurrency_limit=1)
    def s9_apply_fn(s, text, *v):
        new, msg = apply_settings(s.get("ds"), s.get("tkey"), text, *v)
        return new, msg
    for comp in s9_vals:
        for ev in (comp.blur, comp.submit):
            ev(s9_apply_fn, [st, UI["cs_text"]] + s9_vals, [UI["cs_text"], s9_amsg], show_progress="hidden")

    def s9_queue_fn(s, script):
        sc = F.launch_scripts(s.get("ds"))
        if not sc:
            return "No launch script in this dataset yet: save one first."
        cur = Path(script) if script and script != S.NEW and Path(script).is_file() else None
        p = cur or max(sc, key=lambda x: x.stat().st_mtime)
        with S.QLOCK:
            if any(i["path"] == str(p) and i["state"] in ("queued", "running") for i in S.QUEUE["items"]):
                return f"`{p.name}` is already in the queue."
            S.QUEUE["items"].append({"path": str(p), "resume": "", "state": "queued", "note": ""})
        return f"✔ `{p.name}` added to the queue: Training › Run." + (
            " (Save first to queue the changes you made.)" if cur is None else "")
    UI["cs_queue_btn"].click(s9_queue_fn, [st, UI["cs_script"]], UI["cs_save_msg"])

    return {"state": st, "open_flow": lambda: None, "after_nav": after_nav, "go": go}
