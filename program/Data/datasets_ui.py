# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Datasets: the dataset list (with + New dataset) | Improvements, and the nine steps.

+ New dataset starts at 1 Folder; the list shows every dataset with its next step and opens the steps from where the
dataset is. Improvements fixes ticked datasets (its choices are folded on the page) and opens them one by one to check. The steps reuse sdgui's parts where they exist: the watermark choices (6), the caption writer
and the consistency check (7), the repeat folder and the launch script page (9). What each step does to the folder
is in dsflow.py.
"""
from __future__ import annotations
__version__ = "1.141126"  # LoRA Studio version, see version.py

import html
import os
import re
import sys
from pathlib import Path


import datacheck as DC
import detailscan as DSC
import dsflow as F
import features as FT

FLOW = list(range(2, 12))      # 2-8 Prepare, 9 Train, 10 Results, 11 LoRA (one page per dataset)
PARTS = [("prepare", "Prepare", "steps 1-8"), ("train", "Train", "launch script and queue"),
         ("results", "Results", "its runs"), ("lora", "LoRA", "in the Collection")]
PART_STEP = {"train": 9, "results": 10, "lora": 11}
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
            f'<span class="ds-bar-n">{n} pictures</span></div>' + parts_html(st))


def part_of(step) -> str:
    return {9: "train", 10: "results", 11: "lora"}.get(step, "prepare")


def parts_html(st) -> str:
    """The dataset page's four parts: Prepare (steps 1-8), Train (9), Results (its runs), LoRA (its LoRA files)."""
    cur = part_of(st.get("step"))
    out = []
    for key, label, sub in PARTS:
        on = " ds-part-on" if key == cur else ""
        out.append(f'<button class="ds-part{on}" onclick="{send("part|" + key)}"><b>{esc(label)}</b>'
                   f'<span>{esc(sub)}</span></button>')
    return '<nav class="ds-parts" aria-label="Parts of the dataset page">' + "".join(out) + "</nav>"


def stepper_html(st) -> str:
    ds, cur = st.get("ds"), st.get("step", 2)
    states = F.step_states(ds) if ds else {}
    out = []
    for n, key, label in F.STEPS[:8]:                       # 9 Train is a part of its own (parts_html)
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


# --------------------------------------------------------------------------- 1 Folder (New dataset)
def new_list_html(chosen="") -> str:
    fs = F.candidate_folders()
    if F.datasets_dir() is None:
        return '<p class="ls-empty">The dataset folder is not set: Settings › Folders.</p>'
    rows = []
    for f in fs:
        if f["kind"] == "dataset":
            rows.append(f'<div class="ds-frow"><span class="ds-fname">{esc(f["name"])}/</span><span class="ls-chip">dataset</span>'
                        f'<button class="tr-btn tr-small" onclick="{send("edit|" + f["name"])}">Open</button></div>')
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


# --------------------------------------------------------------------------- 2 Setup: is everything there?


def setup_card(types, purpose="character") -> tuple[str, bool]:
    """(card html, model files missing): the model files of the ticked LoRA types and the trainer, and whether
    Ollama has a picture model (steps 4 and 7). Missing files can download in the background while you go on."""
    import hfdownload as HD
    rows, missing = [], False
    labels = F.type_labels()
    for t in types or []:
        lab = esc(labels.get(t, t))
        if not FT.can_train(t):
            rows.append(f'<li class="r-bad">✕ {lab}: its trainer is not installed (Settings › Start and network)</li>')
            continue
        try:
            miss = HD.missing(t)
        except Exception:                                  # noqa: BLE001
            miss = []
        if miss:
            missing = True
            import sd_settings as SS
            rows.append(f'<li class="r-warn">⬇ {lab}: missing {esc(", ".join(SS.slot_label(f) for f in miss))}; it '
                        f'can download while you work on the next steps</li>')
        else:
            rows.append(f'<li class="r-ok">✓ {lab}: model files are here</li>')
    if not types:
        rows.append('<li>Tick a LoRA type to see whether its model files are here.</li>')
    if not FT.have("ollama"):
        rows.append('<li class="r-warn">Ollama is not installed: the sort uses only pass 1, and sentence captions '
                    'need it</li>')
    else:
        try:
            import ollamactl
            vis = [m["name"] for m in ollamactl.installed() if m.get("vision")]
            pick = ollamactl.settings()[1]
            rows.append(f'<li class="r-ok">✓ Ollama: {esc(pick if pick in vis else vis[0])} reads pictures</li>' if vis else
                        '<li class="r-warn">Ollama has no model that reads pictures: Settings › AI helpers › Get '
                        'another model</li>')
        except Exception:                                  # noqa: BLE001
            rows.append('<li class="r-warn">Ollama does not answer: start it before the sort (step 4)</li>')
    if HD.BG["running"]:
        missing = False                                    # the download is already on its way
    return (f'<div class="ls-card"><b>Ready for these LoRA types?</b><ul class="ds-ready">{"".join(rows)}</ul></div>',
            missing)


# --------------------------------------------------------------------------- 3 Usable scan


def _repeat_folders(ds) -> list:
    """The repeat folders of a dataset (read once per page, not once per picture)."""
    return [Path(x["folder"]) for x in F.sg().dataset_stats(ds)["rows"]]


def _where(ds, p, reps=None) -> str:
    """'from _reserve/' for a picture outside the repeat folder, '' for one in it."""
    try:
        rel = Path(p).relative_to(F.root(ds))
    except ValueError:
        return ""
    if any(Path(p).parent == f for f in (reps if reps is not None else _repeat_folders(ds))):
        return ""
    return f"from {rel.parts[0]}/ · " if len(rel.parts) > 1 else "from the dataset folder · "


def s3_render(ds, sc, show):
    have = {str(p) for p in F.candidates(ds)}
    pics = {p: e for p, e in sc["pics"].items() if p in have}
    c = {}
    for e in pics.values():
        c[e["cat"]] = c.get(e["cat"], 0) + 1
    chips = (f'<span class="ls-chip ok">{c.get("usable", 0)} usable</span>' +
             "".join(f'<span class="ls-chip bad">{n} {esc(k)}</span>' for k, n in c.items() if k != "usable"))
    md = f'<div class="ds-chips">{chips}<span class="ls-note">scanned {esc(sc["at"])}, smallest side wanted {sc["min_side"] or "—"} px</span></div>'
    reps = _repeat_folders(ds)
    rows = [{"path": p, "why": _where(ds, p, reps) + (e["why"] if e["bad"] else f'{e["w"]}×{e["h"]}'), "keep": not e["bad"],
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
    bal = ""
    if F.meta(ds).get("purpose") == "worn":              # a worn item: how the kinds are spread (from the captions)
        bal = re.sub(r"\*\*", "", F.worn_balance(ds)) or ("Worn item: name the kinds in step 7 (Kinds and where they "
                                                           "are worn) to see how many pictures each kind has.")
    return (f'<div class="ds-sug"><span class="rs-label">SUGGESTED SIZE</span><div class="rs-big">{n} pictures '
            f'<span class="ls-note">({rng} works)</span></div><div class="ls-note">{esc(why)}</div>'
            + (f'<div class="ls-note" style="margin-top:6px">★ {esc(hint)}</div>' if hint else "")
            + (f'<div class="ls-note" style="margin-top:6px">{esc(bal)}</div>' if bal else "") + '</div>')


def s5_grid(ds, keep_n):
    rows = F.rank(ds, int(keep_n or 0))
    k = sum(1 for r in rows if r["keep"])
    mix = (f'<div class="ds-mix"><span class="rs-label">MIX OF THE SELECTION</span><div>{esc(F.mix_text(rows))}</div>'
           '<div class="ls-note">' + ("Picked for variety: what each picture adds (framing, outfit, place, a detail) "
                                       "counts as much as its quality." if F.meta(ds).get("select_mode") == "varied" else
                                       "Picked by quality first, then what each picture adds (framing, outfit, place).")
           + '</div></div>')
    reps = _repeat_folders(ds)
    cells = [{"path": r["path"], "why": _where(ds, r["path"], reps) + r.get("reason", ""), "keep": r["keep"],
              "bad": r["cat"] in ("out", "poor") and not r.get("pin"), "pin": r.get("pin", "")}
             for r in sorted(rows, key=lambda r: (not r["keep"], r["cat"] in ("out", "poor", "dup"), Path(r["path"]).name))]
    outside = {str(x) for x in F.extra_pictures(ds)} if F.include_extra(ds) else set()
    n_in = sum(1 for r in rows if r["keep"] and str(r["path"]) in outside)
    n_drop = sum(1 for r in rows if not r["keep"] and str(r["path"]) not in outside)
    head = (f'<div class="ds-count">{k} ticked · {n_drop} go to _reserve/'
            + (f' · {n_in} from outside join the repeat folder' if n_in else "") + '</div>')
    return mix, head + grid(cells, ticks=True, click=False, elem="s5")


KIND_ICON = {"add": "➕", "upscale": "⤢", "sharpen": "✦", "crop": "✂"}


def detail_html(ds, r: dict) -> str:
    """Step 5: the detailed scan's gaps and its suggestions, each with a thumbnail and the reason."""
    gaps = "".join(f"<li>{esc(g)}</li>" for g in r.get("gaps") or []) or "<li>No gaps: every framing, angle, pose and "\
        "expression it looks for is there at least twice.</li>"
    counts = " · ".join(f"<b>{esc(DSC.DIMS[d][0])}</b> " + ", ".join(f"{esc(v)} {n}" for v, n in
                                                                      sorted(c.items(), key=lambda t: -t[1])[:6])
                        for d, c in (r.get("counts") or {}).items() if c)
    rows = []
    for it in r.get("items") or []:
        rows.append(f'<div class="dsc-row"><img src="{thumb(it["path"], 96)}" loading="lazy">'
                    f'<div><b>{KIND_ICON.get(it["kind"], "")} {esc(it["label"])}</b><br>'
                    f'<span class="ls-note">{esc(it["why"])}</span></div></div>')
    import comfyupscale as U
    why = U.why_not()
    up_note = ("" if not why else f'<div class="ls-missing-note">Upscaling and sharpening: {esc(why)}. Upscales then '
               "use LoRA Studio's own picture tools when they are installed; sharpening waits for ComfyUI.</div>")
    return (f'<div class="rs-label">DETAILED SCAN · {esc(r.get("at", ""))} · {r.get("kept", 0)} pictures kept by the sort'
            f'</div><div><b>Gaps</b><ul>{gaps}</ul></div><div class="ls-note">{counts}</div>{up_note}'
            f'<style>.dsc-row{{display:flex;gap:10px;align-items:center;margin:6px 0}}.dsc-row img{{width:72px;'
            f'height:72px;object-fit:cover;border-radius:6px}}</style>'
            + ("".join(rows) or "<p>No suggestions.</p>"))


# --------------------------------------------------------------------------- 6 Watermarks and size
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
    return sm, table, label


# --------------------------------------------------------------------------- 7 Captions
def s7_sets_html(ds) -> str:
    sets = F.caption_sets(ds)
    cards = []
    for s in sets:
        c, t = F.set_count(ds, s["ext"])
        lab = "Tags" if s["style"] == "tags" else "Sentences"
        cards.append(f'<div class="ds-set"><div><b>{lab}</b> <span class="rs-mono">IMG{esc(s["ext"])}</span>'
                     f'<span class="ls-chip {"ok" if c == t and t else ""}">{c}/{t}</span></div>'
                     f'<div class="ls-note">for {esc(", ".join(s["types"]) or "your LoRA type")}</div></div>')
    note = ("Two caption sets side by side: one with tags, one with sentences. Each launch script reads its own "
            "(--caption_extension); a type that can read both (Anima, FLUX, Chroma) trains on the one you pick in step 9."
            if len(sets) > 1 else "One caption set for your LoRA type.")
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
    "worn": ("Worn item", "The captions start with the trigger word, then <b>which kind</b> of item it is and <b>where it "
             "is worn</b> (ring, on index finger; hair clip, in hair), then <b>the person in full</b>: hair, face, the "
             "other clothes, the pose, the place, the light. They never describe the item's own look (design, pattern, "
             "material, decorations): the trigger word learns it. With tags, <b>Item tags → trigger words</b> below "
             "moves those tags out, and <b>Kinds and where they are worn</b> keeps the kind and place words the same in "
             "every caption, so you can ask for each kind in a prompt."),
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


def batch_what() -> str:
    import html as _h
    t = F.batch_summary()
    t = _h.escape(t).replace("`_reserve/`", "<code>_reserve/</code>")
    return ('<div class="ls-note"><b>Fix the ticked datasets automatically</b> (less is better): ' + t +
            ' Change it under <i>What Fix does</i> below.</div>')


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


# --------------------------------------------------------------------------- 9 Train: the calculation
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
        return '<p class="ls-empty">Pick the LoRA type above.</p>', "", []
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
    try:                                                 # what your good LoRAs of this type were trained with
        import insights as IN
        sh, eh = IN.settings_hint(tkey), IN.epochs_hint(tkey, ep)
        if sh:
            eff.append(("Your good LoRAs", "for comparison", sh[:1].upper() + sh[1:] + " (Collection › Insights)."))
        if eh:
            eff.append(("Epochs", "maybe fewer", eh[:1].upper() + eh[1:] + "."))
    except Exception:                                    # noqa: BLE001
        pass
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
                  "worn": "How exact the item is",
                  "action": "How exact the pose is", "concept": "Effect strength"}.get(purpose, "Likeness / consistency")
        flex = {"style": "Flexibility (other subjects, prompts that ask for another look)",
                "object": "Flexibility (other people, places and uses)",
                "worn": "Flexibility (other people, outfits and places)",
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
        vals.append((cur, info))
    return calc_html, eff_html, vals


def _same(a, b) -> bool:
    try:
        return abs(float(str(a).split(",")[0]) - float(str(b).split(",")[0])) < 1e-12
    except ValueError:
        return str(a).strip() == str(b).strip()


def apply_settings(ds, tkey, text, *vals):
    S = sg()
    if not ds or not text:
        return text, "Open a launch script first."
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


# --------------------------------------------------------------------------- 9 Train's queue card, 11 LoRA
def queue_card(ds) -> str:
    """The queue items and the last run of this dataset (Train part)."""
    S = sg()
    home = F.root(ds)
    mine = [i for i in S.QUEUE["items"] if Path(i["path"]).parent == home]
    rows = [f'<div class="ds-qrow"><span class="ls-chip{" ok" if i["state"] == "done" else " warn" if i["state"] == "running" else ""}">'
            f'{esc(i["state"])}</span><b>{esc(Path(i["path"]).name)}</b><span class="ls-note">{esc(i.get("note") or "")}</span></div>'
            for i in mine[-8:]]
    body = "".join(rows) or '<p class="ls-note">Nothing of this dataset is in the queue.</p>'
    try:
        import training_ui as TU
        last = [r for r in TU.runs(40) if r["result"] != "running" and r["info"] and r["info"]["dataset"] == ds][:3]
    except Exception:                                    # noqa: BLE001
        last = []
    if last:
        body += '<p class="ls-note">Its last runs:</p>' + "".join(
            f'<div class="ds-qrow"><span class="ls-chip{" ok" if r["result"] == "done" else " bad"}">{esc(r["result"])}'
            f'</span><a href="#" onclick="window.lsGoRun && lsGoRun(\'{q(r["log"])}\'); return false;">'
            f'<b>{esc(r["info"]["type"])} {esc(r["info"]["version"])}</b></a><span class="ls-note">{esc(r["started"])}</span></div>'
            for r in last)
    return (body + '<div class="ds-qrow"><button class="tr-btn tr-small" onclick="window.lsGoQueue && lsGoQueue()">'
            'Open the queue</button><button class="tr-btn tr-small" onclick="' + send("part|results") + '">'
            'Its results →</button></div>')


def lora_html(ds) -> str:
    """The LoRA part: this dataset's LoRA files as the Collection knows them, one card per LoRA (all its versions)."""
    import lora_library as LL
    groups = [m for m in LL.LIB.groups.values() if m and m[-1].dataset is not None and m[-1].dataset.name == ds]
    if not groups:
        return ('<p class="ls-empty">No LoRA of this dataset in your LoRA folders yet. Train makes one; the Collection '
                'finds it after a run (or Collection › Browse › Rescan folders).</p>')
    order = {b: i for i, b in enumerate(LL.LIB.base_names())}
    cards = []
    for m in sorted(groups, key=lambda m: (order.get(m[-1].base, 99), m[-1].base)):
        n = LL.LIB.newest(m[-1].gid) or m[-1]
        src = n if n.preview else next((x for x in reversed(m) if x.preview), None)
        uri = LL.thumb_data_uri(src.preview, 240) if src else ""
        pic = f'<img src="{uri}" alt="">' if uri else '<span class="ds-lph">No preview yet</span>'
        status, issues = LL.group_status(m)
        chip = {"ok": "ok", "warning": "warn", "problem": "bad"}.get(status, "")
        best = LL.best_member(m)
        badges = (" · ★ best: " + esc(best.vlabel) if best else "") + (" · ✓ Completed" if LL.is_completed(n) else "")
        iss = "".join(f'<div class="ls-note">{esc(x)}</div>' for _, x in issues[:3])
        cards.append(f'<div class="ls-card ds-lcard">{pic}<div class="ds-lbody"><b>{esc(LL.group_title(m))}</b>'
                     f'<span>{esc(n.base)} · {len(m)} version{"s" if len(m) != 1 else ""} '
                     f'({esc(", ".join(x.vlabel for x in m))}){badges}</span>'
                     f'<span><span class="ls-chip {chip}">{esc(status)}</span></span>{iss}'
                     f'<span><button class="tr-btn tr-small" onclick="lsOpenLora(\'{q(n.gid)}\')">Open the LoRA page'
                     f'</button></span></div></div>')
    return ("".join(cards) + '<p class="ls-note">The LoRA page (Collection) has its pictures, the training of each '
            'version and the LoRA Manager data. Rate runs and mark the best version in Results.</p>')


# --------------------------------------------------------------------------- the page
