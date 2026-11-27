# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""libcards.py - the Collection drawn as HTML: a LoRA's card and its version chip, the Carddex picture wall
(render_gallery, one tile per dataset), the Library list, a LoRA's details and files check, the versions table, and
the CSS of all of them (added to the page by shell.py). The data comes from lora_library.py (LIB).
"""
from __future__ import annotations
__version__ = "1.271126b"  # LoRA Studio version, see version.py

import math
import re
import time
from pathlib import Path

import lora_library as LL


def sparkline(points: list, w=268, h=44) -> str:
    if len(points) < 2:
        return ""
    step = max(1, len(points) // 80)
    vals = points[::step]
    good = [v for _, v in vals if not (math.isnan(v) or math.isinf(v))]
    if not good:
        return ""
    lo, hi = min(good), max(good)
    rng = (hi - lo) or 1
    x0, x1 = vals[0][0], vals[-1][0]
    xr = (x1 - x0) or 1
    pts, nan_x = [], None
    for s, v in vals:
        x = (s - x0) / xr * w
        if math.isnan(v) or math.isinf(v):
            nan_x = nan_x if nan_x is not None else x
            continue
        y = 4 + (1 - (v - lo) / rng) * (h - 8)
        pts.append(f"{x:.1f},{y:.1f}")
    marker = (f'<line x1="{nan_x:.1f}" y1="2" x2="{nan_x:.1f}" y2="{h}" stroke="#E8A33D" '
              f'stroke-width="2" stroke-dasharray="3 3"/>') if nan_x is not None else ""
    return (f'<svg class="ll-spark" viewBox="0 0 {w} {h}" preserveAspectRatio="none" role="img" '
            f'aria-label="Loss curve"><polyline points="{" ".join(pts)}" fill="none" '
            f'stroke="#4FB3A9" stroke-width="2"/>{marker}</svg>')


def settings_rows(lora: LL.Lora) -> list[tuple[str, str]]:
    m = lora.meta
    if not any(k.startswith("ss_") for k in m):
        return []
    lr = m.get("ss_unet_lr") if m.get("ss_unet_lr") not in (None, "", "None") else m.get("ss_learning_rate")
    opt = (m.get("ss_optimizer") or "").split(".")[-1].split("(")[0]
    batch = m.get("ss_total_batch_size") or m.get("ss_batch_size_per_device") or ""
    rows = [
        ("Base checkpoint", Path(m.get("ss_sd_model_name", "")).name),
        ("Network", f"dim {m.get('ss_network_dim', '?')} / alpha {m.get('ss_network_alpha', '?')}"),
        ("Optimizer", f"{opt} · {LL.fmt_lr(lr)}".strip(" ·")),
        ("Scheduler", m.get("ss_lr_scheduler", "")),
        ("Epochs", f"{m.get('ss_epoch', '?')} / {m.get('ss_num_epochs', '?')}"),
        ("Steps", f"{m.get('ss_steps', '?')} / {m.get('ss_max_train_steps', '?')}"),
        ("Images", m.get("ss_num_train_images", "")),
        ("Batch size", batch),
        ("Resolution", m.get("ss_resolution", "")),
        ("Noise offset", m.get("ss_noise_offset", "")),
        ("Min SNR gamma", m.get("ss_min_snr_gamma", "")),
        ("Precision", m.get("ss_mixed_precision", "")),
        ("Seed", m.get("ss_seed", "")),
    ]
    for k, label in (("ss_training_started_at", "Started"), ("ss_training_finished_at", "Finished")):
        try:
            rows.append((label, time.strftime("%Y-%m-%d %H:%M", time.localtime(float(m[k])))))
        except (KeyError, ValueError, TypeError):
            pass
    return [(a, str(b)) for a, b in rows if str(b) not in ("", "None", "? / ?")]


BEST_BADGE = ('<span class="ll-done ll-best" title="Marked as the best version in Results (on the dataset page): shown first">'
              '&#9733; Best</span>')
DONE_BADGE = '<span class="ll-done" title="Completed: its launch scripts are hidden in the Queue">&#10003; Completed</span>'


def picture_source(members: list):
    """The newest version that has a preview image (None if no version has one)."""
    return next((m for m in reversed(members) if m.preview), None)


def version_chip(members: list) -> str:
    """'V1–V3 · 3 versions ▾': click it to fold out the older versions (the card itself shows the newest)."""
    if len(members) < 2:
        return ""
    return (f'<span class="ll-vchip ll-vchip-btn" title="Show all versions" '
            f'onclick="event.stopPropagation(); llToggle(\'{members[-1].gid}\', this.closest(\'.ll-card\'))">'
            f'{LL.esc(members[0].vlabel)}&ndash;{LL.esc(members[-1].vlabel)} &middot; {len(members)} versions &#9662;</span>')


def render_card(lib: LL.Library, lora: LL.Lora, members: list | None = None, onclick: str = "", extra: str = "",
                selectable: bool = False) -> str:
    """One Library card. With `members` (a version group, oldest first) the card shows the newest
    version, the version range, and the worst status / all issues of the group. `onclick` replaces the
    default (open Details); `extra` is HTML added at the bottom of the card."""
    members = members or [lora]
    st, issues_all = LL.group_status(members)
    vchip = version_chip(members)
    pic = lora if lora.preview else (picture_source(members) or lora)
    if LL.is_completed(lora):
        vchip = DONE_BADGE + vchip
    if LL.best_member(members) is lora and len(members) > 1:
        vchip = BEST_BADGE + vchip
    if pic is not lora:
        vchip += f'<span class="ll-note ll-muted">picture from {LL.esc(pic.vlabel)}</span>'
    blur = " ll-blur" if lib.blur(pic) else ""
    status = f'<span class="ll-dot ll-dot-{st}" title="{LL.esc(st)}"></span>'
    trig = lora.trigger_words[0].split(",")[0] if lora.trigger_words else ""
    fav = '<span class="ll-fav" title="Favorite">&#9733;</span>' if (lora.sidecar or {}).get("favorite") else ""
    badge = f'<span class="ll-badge">{LL.esc(lora.base)}</span>'
    where = ""
    click = onclick or f'onclick="loraPick(\'{lora.key}\', this)"'
    if selectable:
        click += f' data-gid="{lora.gid}"'
        if lora.dataset is not None:
            click += f' data-ds="{LL.esc(lora.dataset.name.lower())}"'
    sel = '<span class="ll-selmark" aria-hidden="true"></span>' if selectable else ""
    uri = LL.thumb_data_uri(pic.preview) if pic.preview else ""
    if uri:
        second = f"trigger: {LL.esc(trig)}" if trig else (f"{len(lora.epoch_files)} epoch files" if lora.epoch_files else "")
        return (
            f'<button class="ll-card ll-card-img ll-st-{st}" {click} title="{LL.esc(lora.path)}">{sel}'
            f'<span class="ll-img{blur}"><img src="{uri}" alt="" loading="lazy"></span>'
            f'<span class="ll-bar"><span class="ll-row"><span class="ll-name">{fav}{LL.esc(LL.group_title(members))}</span>'
            f'{badge}</span><span class="ll-row ll-mono"><span class="ll-trunc">{status}{LL.esc(lora.stem)}</span>'
            f'<span class="ll-trunc">{second}</span></span>{vchip}{where}{extra}</span></button>'
        )
    rows = settings_rows(lora)
    pick = {"Network", "Optimizer", "Epochs", "Base checkpoint"}
    dl = "".join(f"<span class='ll-dt'>{LL.esc(a)}</span><span class='ll-dd'>{LL.esc(b)}</span>"
                 for a, b in rows if a in pick)
    if not dl:
        dl = "<span class='ll-dt'>Metadata</span><span class='ll-dd'>no training data in file</span>"
    if trig:
        dl += f"<span class='ll-dt'>Trigger</span><span class='ll-dd'>{LL.esc(trig)}</span>"
    tags = "".join(f"<span class='ll-tag'>{LL.esc(t)}</span>" for t, _ in lora.tags[:5])
    issues = "".join(f"<span class='ll-issue ll-issue-{l}'>{LL.esc(t)}</span>" for l, t in issues_all[:2])
    return (
        f'<button class="ll-card ll-card-data ll-st-{st}" {click} title="{LL.esc(lora.path)}">{sel}'
        f'<span class="ll-row"><span class="ll-name ll-big">{fav}{LL.esc(LL.group_title(members))}</span>{badge}</span>'
        f'<span class="ll-mono ll-sub">{status}{LL.esc(lora.stem)}</span>{vchip}'
        f'<span class="ll-dl">{dl}</span>'
        f'<span class="ll-tags">{tags}</span>{issues}'
        f'{sparkline(lora.loss)}{where}{extra}'
        f'<span class="ll-foot">No preview image</span></button>'
    )


def render_missing(base: str, ds: LL.Dataset | None) -> str:
    extra = f"Dataset ready · {ds.images} images" if ds else "No dataset found"
    return (f'<div class="ll-card ll-missing"><span class="ll-miss-title">No {LL.esc(base)} version</span>'
            f'<span>{LL.esc(extra)}</span></div>')


def render_dataset_only(ds: LL.Dataset) -> str:
    dirs = ", ".join(p.name for p in ds.repeat_dirs[:3]) or ds.root.name
    cap = f"{ds.captions}/{ds.images} captioned"
    return (f'<div class="ll-card ll-untrained"><span class="ll-name ll-big">Dataset only</span>'
            f'<span class="ll-mono ll-sub">{LL.esc(dirs)}</span><span>{ds.images} images · {cap}</span>'
            f'<span class="ll-warn-text">No trained LoRA found</span></div>')


BASE_ABBR = {"illustrious": "ILL", "noobai": "NOOB", "noobaivpred": "NOOBVP", "pony": "PO", "anima": "ANI", "flux": "F1", "chroma": "CHR", "zimage": "ZI", "zimagebase": "ZIB", "qwenimage": "QWI", "flux2": "F2", "sdxl": "XL",
             "sd15": "15", "sd3": "SD3", "lumina": "LUM", "hunyuan": "HY", "hunyuanimage": "HY", "unknown": "?"}


def base_abbr(base: str) -> str:
    """Short label for a LoRA type in the Carddex: Illustrious -> ILL, Pony -> PO, Flux -> F1, Anima -> ANI."""
    k = re.sub(r"[^a-z0-9]", "", (base or "").lower())
    return BASE_ABBR.get(k) or (base or "?")[:3].upper()


PURPOSES = [("character", "Character"), ("style", "Art style"), ("object", "Object"), ("worn", "Worn item"),
            ("action", "Action / pose"),
            ("concept", "Concept / effect")]
PURPOSE_COLORS = {"character": "#3B82C4", "style": "#C2410C", "object": "#15803D", "worn": "#BE185D", "action": "#7C3AED",
                  "concept": "#B45309"}


def purpose_colors(lib) -> dict:
    """{'on': bool, purpose: colour}: Settings › Look and base models › Colour the cards by purpose."""
    d = dict(lib.data.get("purpose_colors") or {})
    out = {"on": bool(d.get("on"))}
    for k, _ in PURPOSES:
        out[k] = d.get(k) or PURPOSE_COLORS[k]
    return out


def _purpose_of(ds_name: str) -> str:
    try:
        import dsflow
        return dsflow.meta(ds_name).get("purpose") or "character"
    except Exception:                                    # noqa: BLE001
        return "character"


def render_gallery(lib: LL.Library, search: str, base: str, sort: str, only_preview: bool, favorites: bool) -> str:
    """Picture wall of your own LoRAs: one tile per dataset (mychar), with the newest picture and a label for
    every LoRA type it has (ANI, ILL, ...). Click a tile to see that dataset in the Library, folded out."""
    q = (search or "").strip().lower()
    personal = [l for l in lib.loras.values() if l.dataset is not None]

    def mtime(l):
        try:
            return l.path.stat().st_mtime
        except OSError:
            return 0

    def matches(l):
        if base and base != "All" and l.base != base:
            return False
        if favorites and not (l.sidecar or {}).get("favorite"):
            return False
        if q and q not in " ".join([l.stem, l.display_name, l.character, l.base, l.group_name, l.dataset.name,
                                    " ".join(l.trigger_words)]).lower():
            return False
        return True

    by_ds: dict[str, list] = {}
    for l in personal:
        by_ds.setdefault(l.dataset.name.lower(), []).append(l)
    order = {b: i for i, b in enumerate(lib.base_names())}
    tiles = []
    for key, members in by_ds.items():
        hits = sorted((l for l in members if matches(l)), key=mtime, reverse=True)
        if not hits:
            continue
        shown = lib.newest(hits[0].gid)                  # the version marked best, else the newest
        top = shown if shown in hits else hits[0]
        pic_src = top if top.preview else next((l for l in hits if l.preview), None)
        if only_preview and not pic_src:
            continue
        tiles.append((key, members, top, pic_src, mtime(hits[0])))

    if sort == "Name" or sort == "Character":
        tiles.sort(key=lambda t: t[2].dataset.name.lower())
    else:
        tiles.sort(key=lambda t: t[4], reverse=True)

    pc = purpose_colors(lib)

    def tile(key, members, newest, src) -> str:
        ds = newest.dataset
        col = ""
        if pc["on"]:                                     # Settings › Look: a colour per purpose
            pu = _purpose_of(ds.name)
            c = pc.get(pu) or PURPOSE_COLORS.get(pu, "")
            lab = dict(PURPOSES).get(pu, pu)
            col = (f' style="--pc:{LL.esc(c)}"', f'<span class="ll-pchip" title="Purpose">{LL.esc(lab)}</span>')
        nex = sum(len(l.examples) for l in members if l is newest)
        exn = (f'<span class="ll-exn" title="Example pictures linked in LoRA Manager">{nex} example'
               f'{"s" if nex != 1 else ""}</span>') if nex else ""
        uri = LL.thumb_data_uri(src.preview, 320) if src else ""
        blur = " ll-blur" if uri and lib.blur(src) else ""
        if uri:
            pic = f'<span class="ll-gimg{blur}"><img src="{uri}" alt="" loading="lazy"></span>'
        else:
            pic = (f'<span class="ll-gimg ll-gph"><span class="ll-gph-letter">{LL.esc(ds.name[:1].upper())}</span>'
                   f'<span>No preview yet</span></span>')
        # one label per LoRA type, in the order of the base models in Settings; ✓ = Completed
        types: dict[str, list] = {}
        for l in members:
            types.setdefault(l.base, []).append(l)
        chips = []
        for b in sorted(types, key=lambda b: (order.get(b, 99), b)):
            ls = types[b]
            done = LL.is_completed(ls[0])
            n_groups = len({l.gid for l in ls})
            tip = f"{b}: {len(ls)} file{'s' if len(ls) != 1 else ''}" + (" · Completed" if done else "")
            cls = "ll-tchip" + (" ll-tchip-done" if done else "") + (" ll-tchip-cur" if b == newest.base else "")
            gid_t = max(ls, key=mtime).gid
            chips.append(f'<span class="{cls}" title="{LL.esc(tip)} · click to open this one" data-open="{gid_t}" '
                         f'onclick="event.stopPropagation(); llOpenCard(this.dataset.open, this.closest(\'.ll-gtile\'), this.closest(\'.ll-gtile\').dataset.ds)">'
                         f'{"&#10003;" if done else ""}'
                         f'{LL.esc(base_abbr(b))}{f"<small>&times;{n_groups}</small>" if n_groups > 1 else ""}</span>')
        worst = max((l.status for l in members), key=lambda x: LL.STATUS_ORDER[x])
        dot = "" if worst == "ok" else f'<span class="ll-dot ll-dot-{worst}" title="{LL.esc(worst)}"></span>'
        fav = '<span class="ll-fav">&#9733;</span>' if any((l.sidecar or {}).get("favorite") for l in members) else ""
        note = f"newest: {newest.stem}" + ("" if src is None or src is newest else f" · picture from {src.stem}")
        return (f'<button class="ll-card ll-gtile{" ll-pcol" if col else ""}" '
                f'onclick="llTile(this)"{col[0] if col else ""} '
                f'data-gid="ds:{LL.esc(key)}" data-ds="{LL.esc(key)}" data-dsn="{LL.esc(ds.name)}" data-open="{newest.gid}" '
                f'title="{LL.esc(note)}">'
                f'<span class="ll-selmark" aria-hidden="true"></span>{pic}'
                f'<span class="ll-gcap ll-gcap2"><span class="ll-name">{fav}{dot}{LL.esc(ds.name)}</span>'
                f'<span class="ll-tchips">{col[1] if col else ""}{"".join(chips)}{exn}</span></span></button>')

    parts = [tile(*t[:4]) for t in tiles]
    n_files = len(personal)
    if not parts:
        msg = ("No personal LoRAs found yet. Check the dataset folders in Settings." if not n_files
               else "Nothing matches these filters.")
        return f'<div class="ll-wrap"><p class="ll-muted">{msg}</p></div>'
    head = (f'<div class="ll-muted" style="margin:2px 0 10px">{len(by_ds)} datasets with LoRAs ({n_files} files) '
            f'&middot; newest picture shown (or the version marked ★ best), labels = LoRA types (&#10003; = Completed) &middot; '
            f'click a tile to open its newest LoRA, or a label to open that type</div>')
    return f'<div class="ll-wrap">{head}<div class="ll-ggrid">{"".join(parts)}</div></div>'


def epochs_line(l: LL.Lora) -> str:
    if not l.epoch_files:
        return ""
    eps = ", ".join(str(e) for e, _ in l.epoch_files)
    return f'<span class="ll-mono ll-sub">epoch files: {LL.esc(eps)}{"" if l.final else " (no final file)"}</span>'


def library_entry(lib: LL.Library, l: LL.Lora) -> str:
    """A Library card for a version group, showing only the newest version; click opens its Details.
    With several versions, the 'V1–V3 ▾' chip folds out a row with every version (newest first)."""
    members = lib.group_of(l)
    if len(members) < 2:
        return render_card(lib, l, members, extra=epochs_line(l), selectable=True)
    head = render_card(lib, l, members, extra=epochs_line(l), selectable=True)   # click = Details of the newest
    versions = "".join(
        render_card(lib, m, [m], extra=f'<span class="ll-vlabel">{LL.esc(m.vlabel)}'
                    f'{" &middot; newest" if m is members[-1] else ""}'
                    f'{" &middot; &#9733; best" if m is LL.best_member(members) else ""}</span>{epochs_line(m)}')
        for m in reversed(members))
    fold = (f'<div class="ll-vers" data-gid="{l.gid}"><div class="ll-vers-head"><b>{LL.esc(LL.group_title(members))}</b>'
            f'<span class="ll-muted">{len(members)} versions, newest first &middot; click a version to open its details</span></div>'
            f'<div class="ll-grid">{versions}</div></div>')
    return head + fold


def render_library(lib: LL.Library, search: str, bases: list, statuses: list, group: str) -> str:
    search = (search or "").strip().lower()
    bases = set(bases or [])
    statuses = set(statuses or [])
    status_map = {"OK": "ok", "Warnings": "warning", "Problems": "problem"}
    want_status = {status_map[s] for s in statuses if s in status_map}
    show_untrained = "Untrained datasets" in statuses
    show_missing = "Missing bases" in statuses
    show_foreign = False          # only your own LoRAs (the LoRA folder is not scanned)
    offered = {x.base for x in lib.loras.values() if x.dataset is not None} | {"Unknown"}

    def match(l: LL.Lora) -> bool:
        if l.dataset is None and not show_foreign:
            return False      # not made by you: no dataset in your dataset folders
        if bases and l.base in offered and l.base not in bases:
            return False      # bases that are not offered as a filter (only used by downloads) always pass
        if LL.group_status(lib.group_of(l))[0] not in want_status:
            return False
        if search:
            hay = " ".join([l.stem, l.display_name, l.character, l.base, " ".join(l.trigger_words),
                            " ".join(t for t, _ in l.tags)]).lower()
            return search in hay
        return True

    # one card per version group: a group is shown when any of its versions matches, as its newest version
    shown = {}
    for l in lib.loras.values():
        if match(l):
            shown[l.gid] = lib.newest(l.gid) or l
    loras = list(shown.values())
    order = {b: i for i, b in enumerate(lib.base_names())}
    sections: dict[str, list[str]] = {}
    trained = {l.character.lower() for l in lib.loras.values()}

    if group == "Base model":
        for l in sorted(loras, key=lambda l: (order.get(l.base, 99), l.character.lower(), l.stem.lower())):
            sections.setdefault(l.base, []).append(library_entry(lib, l))
    else:
        by_char: dict[str, list[LL.Lora]] = {}
        for l in loras:
            by_char.setdefault(l.character.lower(), []).append(l)
        names = {k: v[0].character for k, v in by_char.items()}
        if show_untrained or show_missing:
            for k, ds in lib.datasets.items():
                if not search or search in ds.name.lower():
                    names.setdefault(k, ds.name)
        for k in sorted(names):
            cards = [library_entry(lib, l) for l in sorted(
                by_char.get(k, []), key=lambda l: (order.get(l.base, 99), l.stem.lower()))]
            ds = lib.datasets.get(k)
            if k not in trained:
                if show_untrained and ds:
                    cards.append(render_dataset_only(ds))
            elif show_missing and ds:
                have = {l.base for l in lib.loras.values() if l.character.lower() == k}
                for b in lib.settings["expected_bases"]:
                    if b not in have and (not bases or b in bases):
                        cards.append(render_missing(b, ds))
            if cards:
                sections[names[k]] = cards

    gst = [LL.group_status(m)[0] for m in lib.groups.values() if m and (m[-1].dataset or show_foreign)]
    n_prob, n_warn = gst.count("problem"), gst.count("warning")
    head = (f'<div class="ll-stats"><span><b>{len(loras)}</b> LoRAs shown</span>'
            f'<span><b>{len(lib.datasets)}</b> datasets</span>'
            f'<span class="ll-c-warn"><b>{n_warn}</b> warnings</span>'
            f'<span class="ll-c-prob"><b>{n_prob}</b> problems</span>'
            f'<span class="ll-muted">scanned {LL.esc(lib.scanned_at)}</span></div>')
    if not sections:
        empty = ("No LoRAs found. Check the folders in Settings › Folders."
                 if not lib.loras else "Nothing matches these filters.")
        return f'<div class="ll-wrap">{head}<p class="ll-muted">{empty}</p></div>'
    body = "".join(f'<h3 class="ll-h">{LL.esc(name)}</h3><div class="ll-grid">{"".join(cards)}</div>'
                   for name, cards in sections.items())
    return f'<div class="ll-wrap">{head}{body}</div>'


def _card_script(lora) -> str:
    """The launch script that made this version (from its training card), as a file name."""
    import trainingcard as TC
    sp = TC.stored(lora.sidecar).get("_script")
    return Path(sp).name if sp else ""


def files_check_html(lib: LL.Library, lora: LL.Lora) -> str:
    ds = lora.dataset
    has_meta = any(k.startswith("ss_") for k in lora.meta)
    logging_in_script = any(lg for _, lg in lora.scripts)
    rows = [
        ("Training metadata", has_meta, "in .safetensors" if has_meta else "none in file"),
        ("Dataset folder", bool(ds), ds.root.name if ds else "not found"),
        ("Captions (.txt)", bool(ds) and ds.captions == ds.images,
         f"{ds.captions}/{ds.images}" if ds else "-"),
        ("Model file", lora.dataset_path is not None,
         lora.dataset_path.parent.name + "/" + lora.dataset_path.name if lora.dataset_path else "not found"),
        ("Launch script", bool(lora.scripts) or bool(_card_script(lora)),
         _card_script(lora) or ", ".join(p.name for p, _ in lora.scripts[:2]) or "not found"),
        ("TensorBoard log", bool(lora.loss),
         Path(lora.log_run).name if lora.log_run else
         ("script has --logging_dir, no log matched" if logging_in_script else "not found - add --logging_dir")),
        ("LoRA Manager metadata", lora.sidecar is not None,
         f"{LL.ARCHIVE_DIR}/{lora.stem}.metadata.json" if lora.sidecar else "missing - run Personal sync"),
        ("Preview image", lora.preview is not None, lora.preview.name if lora.preview else "missing - run Personal sync"),
    ]
    li = "".join(f"<li><span>{LL.esc(a)}</span><span class='{'ll-ok' if ok else 'll-c-warn'}'>{LL.esc(t)}</span></li>"
                 for a, ok, t in rows)
    return f"<ul class='ll-check'>{li}</ul>"


def details_html(lib: LL.Library, lora: LL.Lora) -> tuple[str, str]:
    issues = "".join(f"<div class='ll-issue ll-issue-{l}'>{LL.esc(t)}</div>" for l, t in lora.issues)
    if getattr(lora, "accepted", None):
        issues += (f"<div class='ll-muted' style='margin-top:6px'>Accepted by you: {LL.esc('; '.join(lora.accepted))} "
                   "(no mark in the Collection)</div>")
    epochs = ""
    if lora.epoch_files:
        epochs = ("<div class='ll-muted' style='margin-top:8px'>Epoch files: "
                  + ", ".join(str(e) for e, _ in lora.epoch_files) + "</div>")
    used = ""
    try:
        import insights
        u = insights.usage_of(lora)
        if u and u["pictures"]:
            used = (f"<div class='ll-muted' style='margin-top:6px'>Used in <b>{u['pictures']}</b> of your pictures"
                    + (f", usually at strength <b>{u['strength']:.2f}</b>" if u.get("strength") is not None else "")
                    + (f", last on {LL.esc(u['last'])}" if u.get("last") else "") + "</div>")
    except Exception:                            # noqa: BLE001
        pass
    head = (f"<div class='ll-dhead'><div class='ll-muted ll-upper'>{LL.esc(lora.base)} · {LL.esc(lora.character)}</div>"
            f"<h2>{LL.esc(lora.display_name)}</h2><div class='ll-mono ll-muted'>{LL.esc(lora.path)}</div>"
            f"{issues}{epochs}{used}</div>")
    import trainingcard as TC
    card = TC.stored(lora.sidecar)
    if card:                                             # what you set (Edit, or synced from LoRA Manager) is shown
        if lora.trigger_words:
            card["trigger"] = ", ".join(str(w) for w in lora.trigger_words)
        vname = ((lora.sidecar or {}).get("civitai") or {}).get("name")
        if vname:
            card["version"] = vname
        cs = TC.usage_tips(lora.sidecar).get("clip_skip")
        if cs not in (None, "") and not card.get("clip_skip"):
            card["clip_skip"] = f"{cs} (usage tip)"
        settings = "<h4>Training</h4>" + TC.card_html(card, LL.esc)
    else:
        rows = settings_rows(lora)                       # a LoRA without a dataset: what its file says
        if rows:
            dl = "".join(f"<dt>{LL.esc(a)}</dt><dd>{LL.esc(b)}</dd>" for a, b in rows)
            settings = f"<h4>Training settings <span class='ll-muted'>(from .safetensors)</span></h4><dl class='ll-ddl'>{dl}</dl>"
        else:
            settings = "<h4>Training settings</h4><p class='ll-muted'>No training data found for this LoRA.</p>"
    tags = "".join(f"<span class='ll-tag'>{LL.esc(t)} {p}%</span>" for t, p in lora.tags)
    if tags:
        settings += f"<h4>Caption tags</h4><div class='ll-tags'>{tags}</div>"
    settings += f"<h4>Files check</h4>{files_check_html(lib, lora)}"
    return head, settings


CSS = """
/* Normal (light) palette; the dark one below is used in dark mode */
:root { --ll-card:#FDFDFB; --ll-line:#C9CDD3; --ll-line2:#BCC2CA; --ll-text:#1C1F24; --ll-muted:#5A626C;
        --ll-dim:#7D858F; --ll-ok:#21857C; --ll-warn:#A86A0C; --ll-prob:#C0392B; --ll-chip:#EDEFF2;
        --ll-chip-text:#3A4048; --ll-imgbg:#E6E9ED; --ll-dash:#B5BBC3; --ll-warn-bg:#FBF1DF; --ll-warn-fg:#7A4E05;
        --ll-prob-bg:#FBE6E3; --ll-prob-fg:#8E2A1F; --ll-focus:#E8A33D88; }
.dark { --ll-card:#1B1E23; --ll-line:#2A2E36; --ll-line2:#343944; --ll-text:#E6E8EB; --ll-muted:#9AA1AB;
        --ll-dim:#6B7280; --ll-ok:#4FB3A9; --ll-warn:#E8A33D; --ll-prob:#F07C6A; --ll-chip:#262A31;
        --ll-chip-text:#C3C8CF; --ll-imgbg:#0E0F12; --ll-dash:#3A404B; --ll-warn-bg:#2B2417; --ll-warn-fg:#F4C06F;
        --ll-prob-bg:#2E1D1B; --ll-prob-fg:#F2A497; --ll-focus:#E8A33D55; }
.ll-pgrid { display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr)); gap:10px; }
.ll-ptile { all:unset; cursor:pointer; display:flex; flex-direction:column; border-radius:10px; overflow:hidden;
  background:var(--block-background-fill); border:1px solid var(--border-color-primary); }
.ll-ptile:hover { border-color:var(--color-accent); }
.ll-ptile.ll-psel { outline:3px solid var(--color-accent); outline-offset:-3px; }
.ll-pimg { display:block; aspect-ratio:3/4; overflow:hidden; background:var(--background-fill-secondary); }
.ll-pimg img { width:100%; height:100%; object-fit:cover; display:block; }
.ll-pcap { display:flex; justify-content:space-between; gap:6px; padding:5px 8px; font-size:12px; }
.ll-pcap span { opacity:.65; }
.ll-wrap { color:var(--ll-text); }
.ll-stats { display:flex; flex-wrap:wrap; gap:18px; font-size:14px; color:var(--ll-muted); margin:4px 0 8px; }
.ll-stats b { color:var(--ll-text); font-family:'JetBrains Mono',monospace; }
.ll-c-warn, .ll-c-warn b { color:var(--ll-warn) !important; }
.ll-c-prob, .ll-c-prob b { color:var(--ll-prob) !important; }
.ll-ok { color:var(--ll-ok); }
.ll-muted { color:var(--ll-muted); }
.ll-upper { text-transform:uppercase; letter-spacing:.06em; font-size:12px; }
.ll-h { margin:22px 0 10px; font-size:15px; font-weight:600; color:var(--ll-muted); }
.ll-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(240px,1fr)); gap:16px; }
.ll-card { all:unset; box-sizing:border-box; display:flex; flex-direction:column; background:var(--ll-card);
           border:1px solid var(--ll-line); border-radius:10px; overflow:hidden; cursor:pointer; color:var(--ll-text);
           font-family:inherit; text-align:left; min-width:0; }
.ll-card:hover, .ll-card:focus-visible { border-color:var(--ll-line2); outline:2px solid var(--ll-focus); }
.ll-st-problem { border-color:var(--ll-prob); }
.ll-img { display:block; aspect-ratio:3/4; overflow:hidden; background:var(--ll-imgbg); }
.ll-img img { width:100%; height:100%; object-fit:cover; display:block; transition:filter .2s; }
.ll-blur img { filter:blur(22px) brightness(.8); }
.ll-blur:hover img { filter:none; }
.ll-bar { display:flex; flex-direction:column; gap:5px; padding:9px 12px 10px; }
.ll-row { display:flex; justify-content:space-between; align-items:center; gap:8px; min-width:0; }
.ll-name { font-weight:600; font-size:14px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; min-width:0; }
.ll-big { font-size:16px; }
.ll-mono { font-family:'JetBrains Mono',monospace; font-size:11px; color:var(--ll-muted); }
.ll-trunc { white-space:nowrap; overflow:hidden; text-overflow:ellipsis; min-width:0; }
.ll-gtile.ll-pcol { border:2px solid var(--pc) !important; }
.ll-gtile.ll-pcol .ll-tchip-cur { background:var(--pc) !important; color:#fff !important; border-color:var(--pc) !important; }
.ll-pchip { font-size:10px; padding:1px 7px; border-radius:99px; background:var(--pc); color:#fff; font-weight:600; }
.ll-exn { font-size:10px; color:var(--ll-muted); margin-left:4px; }
.ll-badge { flex:none; font-family:'JetBrains Mono',monospace; font-size:11px; padding:2px 8px; border-radius:99px;
            background:var(--ll-chip); color:var(--ll-chip-text); }
.ll-note { font-size:11px; color:var(--ll-warn); }
.ll-dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; vertical-align:middle; }
.ll-dot-ok { background:var(--ll-ok); } .ll-dot-warning { background:var(--ll-warn); } .ll-dot-problem { background:var(--ll-prob); }
.ll-fav { color:var(--ll-warn); margin-right:5px; }
.ll-card-data { padding:14px 16px; gap:10px; min-height:300px; }
.ll-sub { margin-top:-4px; }
.ll-dl { display:grid; grid-template-columns:auto 1fr; gap:5px 12px; font-size:12px; }
.ll-dt { color:var(--ll-muted); } .ll-dd { font-family:'JetBrains Mono',monospace; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.ll-tags { display:flex; flex-wrap:wrap; gap:6px; }
.ll-tag { font-family:'JetBrains Mono',monospace; font-size:11px; padding:3px 8px; background:var(--ll-chip);
          color:var(--ll-chip-text); border-radius:99px; }
.ll-spark { width:100%; height:44px; margin-top:auto; }
.ll-foot { font-size:11px; color:var(--ll-dim); border-top:1px solid var(--ll-line); padding-top:8px; }
.ll-issue { font-size:12px; padding:6px 10px; border-radius:6px; margin-top:6px; display:block; }
.ll-issue-warning { background:var(--ll-warn-bg); color:var(--ll-warn-fg); }
.ll-issue-problem { background:var(--ll-prob-bg); color:var(--ll-prob-fg); }
.ll-missing { cursor:default; background:transparent; border:1px dashed var(--ll-dash); align-items:center; justify-content:center;
              gap:6px; padding:24px; min-height:200px; color:var(--ll-dim); font-size:13px; text-align:center; }
.ll-miss-title { font-size:15px; color:var(--ll-muted); }
.ll-untrained { cursor:default; border:1px dashed var(--ll-warn); padding:16px; gap:8px; font-size:13px; min-height:140px; }
.ll-warn-text { color:var(--ll-warn); }
.ll-dhead h2 { margin:4px 0 4px; font-size:26px; }
.ll-ddl { display:grid; grid-template-columns:auto 1fr; gap:6px 16px; margin:0; font-size:13px; }
.ll-ddl dt { color:var(--ll-muted); } .ll-ddl dd { margin:0; font-family:'JetBrains Mono',monospace; }
.ll-tgrid { display:grid; grid-template-columns:repeat(auto-fit,minmax(420px,1fr)); gap:10px; }
.ll-tcard { border:1px solid var(--ll-line); border-radius:10px; padding:10px 12px; background:var(--ll-card); }
.ll-tcard .rs-label { font-size:11px; letter-spacing:.06em; color:var(--ll-muted); margin-bottom:6px; }
.ll-tcard .ll-ddl { grid-template-columns:minmax(120px,max-content) 1fr; }
.ll-tcard .ll-ddl dd { overflow-wrap:anywhere; font-size:12.5px; }
.ll-check { list-style:none; padding:0; margin:0; display:flex; flex-direction:column; gap:6px; font-size:13px; }
.ll-check li { display:flex; justify-content:space-between; gap:12px; }
h4 { margin:18px 0 8px; font-size:14px; }
.ll-ggrid { display:grid; grid-template-columns:repeat(auto-fill,minmax(190px,1fr)); gap:14px; }
.ll-gimg { display:flex; aspect-ratio:3/4; overflow:hidden; background:var(--ll-imgbg); }
.ll-gimg img { width:100%; height:100%; object-fit:cover; display:block; transition:filter .2s; }
.ll-gph { flex-direction:column; align-items:center; justify-content:center; gap:6px; color:var(--ll-dim); font-size:12px; }
.ll-gph-letter { font-size:40px; font-weight:600; color:var(--ll-dash); }
.ll-gcap { display:flex; justify-content:space-between; align-items:center; gap:6px; padding:8px 10px; }
.ll-gcap .ll-name { font-size:13px; }
.ll-gcap .ll-badge { font-size:10px; padding:2px 7px; }
.ll-ggroup { grid-column:1 / -1; border:1px solid var(--ll-line2); border-radius:12px; padding:10px 12px 12px;
             display:flex; flex-direction:column; gap:10px; }
.ll-ghead { display:flex; align-items:center; gap:12px; flex-wrap:wrap; }
.ll-ghead .ll-name { font-size:15px; }
.ll-grid { grid-auto-flow:dense; }
.ll-vers { display:none; grid-column:1 / -1; border:1px solid var(--ll-line2); border-radius:12px;
           padding:12px; flex-direction:column; gap:10px; background:var(--ll-imgbg); }
.ll-vers.ll-show { display:flex; }
.ll-vers-head { display:flex; gap:12px; align-items:baseline; flex-wrap:wrap; font-size:14px; }
.ll-card.ll-open { outline:2px solid var(--ll-warn); }
.ll-done { align-self:flex-start; font-size:11px; font-weight:600; padding:2px 8px; border-radius:99px;
           background:var(--ll-ok); color:#0E1412; margin-right:6px; }
.ll-done-tile { padding:1px 6px; }
.ll-card { position:relative; }
.ll-selmark { display:none; position:absolute; top:8px; left:8px; z-index:3; width:22px; height:22px; border-radius:6px;
              border:2px solid #fff; background:rgba(0,0,0,.45); box-shadow:0 1px 3px rgba(0,0,0,.5); }
body.ls-selecting .ll-selmark { display:block; }
body.ls-selecting .ll-card-data[data-gid] { padding-top:38px; }
.ll-card.ll-picked { outline:3px solid var(--ll-ok) !important; }
.ll-card.ll-picked .ll-selmark { background:var(--ll-ok); border-color:var(--ll-ok); }
.ll-card.ll-picked .ll-selmark::after { content:"\\2713"; color:#0E1412; font-weight:700; font-size:15px;
              position:absolute; left:3px; top:-1px; }
.ll-selbar { align-items:center !important; }
.ll-selcount { font-family:'JetBrains Mono',monospace; font-size:13px; color:var(--ll-muted); }
.ll-gcap2 { flex-direction:column; align-items:flex-start; gap:6px; }
.ll-tchips { display:flex; flex-wrap:wrap; gap:4px; }
.ll-tchip { font-family:'JetBrains Mono',monospace; font-size:10px; font-weight:600; padding:2px 6px; border-radius:6px;
            border:1px solid var(--ll-line2); color:var(--ll-muted); }
.ll-tchip { cursor:pointer; } .ll-tchip:hover { border-color:var(--ll-warn); }
.ll-tchip small { font-size:9px; margin-left:2px; opacity:.8; }
.ll-tchip-cur { color:var(--ll-text, inherit); border-color:var(--ll-muted); }
.ll-tchip-done { background:var(--ll-ok); border-color:var(--ll-ok); color:#0E1412; }
.ll-card.ll-jump { outline:3px solid var(--ll-warn) !important; }
.ll-vchip-btn { cursor:pointer; } .ll-vchip-btn:hover { border-color:var(--ll-warn); color:var(--ll-warn); }
.ll-vlabel { font-family:'JetBrains Mono',monospace; font-size:12px; font-weight:600; color:var(--ll-warn); }
.ll-gver { flex:none; font-family:'JetBrains Mono',monospace; font-size:10px; padding:2px 7px; border-radius:99px;
           border:1px solid var(--ll-line2); color:var(--ll-muted); margin-right:4px; }
.ll-vchip { align-self:flex-start; font-family:'JetBrains Mono',monospace; font-size:11px; padding:2px 8px;
            border-radius:99px; border:1px solid var(--ll-line2); color:var(--ll-muted); }

/* ---- the status strip and the shared page parts ---- */
.ls-strip { display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:10px 18px;
            padding:8px 14px; border-radius:10px; background:#262A31; color:#E6E8EB; font-size:13px; }
.ls-run { cursor:pointer; font-weight:600; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; flex:1 1 320px; }
.ls-st-run .ls-run { color:#F4C06F; } .ls-st-done .ls-run { color:#7FD3C9; } .ls-st-fail .ls-run { color:#F2A497; }
.ls-st-idle .ls-run { color:#C3C8CF; font-weight:500; }
.ls-machine { display:flex; flex-wrap:wrap; gap:6px 14px; align-items:center; }
.ls-meter { display:inline-flex; flex-direction:column; gap:3px; min-width:96px; }
.ls-mlab { font-family:'JetBrains Mono',monospace; font-size:11px; color:#C3C8CF; white-space:nowrap; }
.ls-mbar { display:block; height:4px; border-radius:2px; background:#3A404B; overflow:hidden; }
.ls-mbar > span { display:block; height:100%; background:#21857C; }
.ls-hot .ls-mbar > span { background:#E8A33D; } .ls-hot .ls-mlab { color:#F4C06F; }
.ls-mnote { font-size:11px; color:#9AA1AB; }
.ls-oll { font-family:'JetBrains Mono',monospace; font-size:12px; white-space:nowrap; align-self:center; }
.ls-oll-on { color:#7FDDD3; font-weight:600; } .ls-oll-off { color:#B8BEC7; }
.ls-empty, .ls-empty p { color:var(--ll-muted); font-style:italic; }
.ls-filters { border-right:1px solid var(--ll-line); padding-right:10px; }
.ls-actions { align-items:center !important; }
.pp-box { display:flex; flex-direction:column; gap:6px; padding:10px 12px; border:1px solid var(--ll-line); border-radius:10px;
          background:var(--ll-card); }
.pp-row { display:grid; grid-template-columns:170px 110px 1fr; gap:10px; align-items:center; font-size:14px; }
.pp-name { font-weight:600; }
.pp-chip { justify-self:start; font-family:'JetBrains Mono',monospace; font-size:11px; font-weight:600; padding:2px 9px; border-radius:99px; }
.pp-ok { background:#21857C; color:#fff; } .pp-todo { background:#FBE9C8; color:#8F560B; }
.dark .pp-todo { background:#3A2E1A; color:#F4C06F; }
.pp-text { color:var(--ll-muted); font-size:13px; }
.pp-next { margin:6px 0 0; font-weight:600; color:var(--ll-ok); }
.pp-empty { margin:0; color:var(--ll-muted); }
@media (max-width: 700px) { .pp-row { grid-template-columns:1fr; gap:2px; } .ls-filters { border-right:0; } }
"""







STATUS_CHOICES = ["OK", "Warnings", "Problems"]


def version_choices(lora) -> list:
    members = LL.LIB.group_of(lora)
    newest, best = members[-1], LL.best_member(members)
    return [(f"{m.vlabel}{' (newest)' if m is newest and len(members) > 1 else ''}"
             f"{' ★ best' if m is best else ''}  ·  {m.stem}", m.key)
            for m in reversed(members)]


def versions_html(lora) -> str:
    """All versions of a LoRA (one row each): trained, pictures, steps, final loss, completed."""
    rows = []
    for l in sorted(LL.LIB.group_of(lora), key=lambda x: -(x.version or 0)):
        m = l.meta or {}
        when = ""
        try:
            t = float(m.get("ss_training_finished_at") or 0) or (l.path.stat().st_mtime if l.path else 0)
            when = time.strftime("%Y-%m-%d", time.localtime(t)) if t else ""
        except (OSError, ValueError, TypeError):
            pass
        import trainingcard as TC
        card = TC.stored(l.sidecar)                      # the training card, when the version was scanned
        when = (card.get("trained_on") or "")[:10] or when
        pics = card.get("pictures") or m.get("ss_num_train_images") or ""
        steps = card.get("steps") or m.get("ss_max_train_steps") or m.get("ss_steps") or ""
        loss = f"{l.loss[-1][1]:.3f}" if l.loss else (card.get("final_loss") or "")
        done = "✓ yes" if LL.is_completed(l) else "no"
        rows.append(f"<tr><td><b>{LL.esc(l.vlabel)}</b></td><td>{LL.esc(when)}</td><td>{LL.esc(pics)}</td><td>{LL.esc(steps)}</td>"
                    f"<td>{LL.esc(loss)}</td><td>{done}</td></tr>")
    return ('<table class="rs-table"><thead><tr><th>Version</th><th>Trained</th><th>Pictures</th><th>Steps</th>'
            '<th>Final loss</th><th>Completed</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table>")
