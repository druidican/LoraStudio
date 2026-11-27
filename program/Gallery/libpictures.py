# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""libpictures.py - your pictures made with a LoRA: the picture folder per LoRA type, which pictures belong to which LoRA
and save point (by the LoRA names in their prompts, also old names you linked: picture_aliases), and the gallery of a
LoRA's pictures with the details of one (render_pictures, picture_details). The picture index is picgallery.py.
"""
from __future__ import annotations
__version__ = "1.271126b"  # LoRA Studio version, see version.py

import re
import time
from pathlib import Path

import libcards
import lora_library as LL
import picgallery as PG


PIC_PAGE = 120


def _simple(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def picture_folder(base: str) -> tuple[str, str]:
    """(folder, how it was found) for a LoRA type's pictures; folder "" = not found, ask."""
    s = LL.LIB.settings
    for t, d in (s.get("picture_dirs") or {}).items():
        if t.lower() == base.lower():
            return d, f"the folder you set for {base}"
    root = s.get("pictures_dir") or ""
    if not root:
        return "", "No ComfyUI picture folder is set (Settings > Folders)."
    if not Path(root).is_dir():
        return "", f"The ComfyUI picture folder `{root}` does not exist."
    kws = next((k for n, k in s["bases"] if n == base), [])
    wanted = {_simple(base), _simple(libcards.base_abbr(base))} | {_simple(k) for k in kws}
    wanted.discard("")
    level = [Path(root)]
    for depth in (1, 2):
        nxt = []
        for d in level:
            try:
                subs = sorted((c for c in d.iterdir() if c.is_dir() and not c.name.startswith(".")), key=lambda c: c.name)
            except OSError:
                continue
            for c in subs:
                if _simple(c.name) in wanted:
                    return str(c), f"`{c.name}` in your ComfyUI picture folder"
            nxt += subs
        level = nxt[:400]
    return "", f"No `{base}` folder was found in `{root}`."


OLD = "old"          # marks a picture linked through an old LoRA name


def picture_aliases(gid: str) -> list[str]:
    """Old LoRA names (from before a rename) linked to this LoRA by you, saved in library_data.json."""
    return list(((LL.LIB.data.get("picture_aliases") or {}).get(gid)) or [])


def set_picture_aliases(gid: str, names: list[str]):
    al = LL.LIB.data.setdefault("picture_aliases", {})
    names = sorted({PG.norm(n) for n in names if str(n).strip()})
    if names:
        al[gid] = names
    else:
        al.pop(gid, None)
    LL.LIB.save_data()


def _file_hashes(l) -> set:
    """sha256 of the LoRA's files, only where already known (LoRA Manager data or the hash cache):
    no hashing of big files just for the Gallery."""
    out = set()
    sha = str((l.sidecar or {}).get("sha256") or "").lower()
    if len(sha) >= 10:
        out.add(sha)
    for p in [l.lora_path, l.dataset_path] + [f for _, f in (l.epoch_files or [])]:
        hit = LL.LIB.hash_cache.get(str(p)) if p else None
        if hit and hit.get("sha"):
            out.add(hit["sha"].lower())
    return out


def group_lora_names(members: list, gid: str = "") -> tuple[dict, set]:
    """name (as PG.norm makes it) -> (Lora, epoch | None | OLD) for every file of the group and the old
    names you linked; plus the known sha256 hashes."""
    names, hashes = {}, set()
    for l in members:
        for p in (l.lora_path, l.dataset_path):
            if p:
                names[PG.norm(p.name)] = (l, None)
        names.setdefault(l.stem.lower(), (l, None))
        for ep, f in l.epoch_files or []:
            names.setdefault(PG.norm(f.name), (l, ep))
        hashes |= _file_hashes(l)
    if members:
        for a in picture_aliases(gid or members[-1].gid):
            names.setdefault(a, (members[0], OLD))
    return names, hashes


def known_lora_names() -> set:
    """Every name that already belongs to some LoRA (files, epochs, linked old names)."""
    out = set()
    for gid, members in LL.LIB.groups.items():
        out |= set(group_lora_names(members, gid)[0])
    return out


def alias_suggestions(gid: str) -> list[tuple[str, str]]:
    """Checkbox choices: names in your pictures that no LoRA file has, but that look like this LoRA
    (contain the dataset / character name), plus the old names already linked. [(label, name)]"""
    members = LL.LIB.groups.get(gid) or []
    if not members:
        return []
    n = members[-1]
    keys = {_simple(x) for x in (n.dataset.name if n.dataset else "", n.character, n.group_name) if x}
    keys = {k for k in keys if len(k) >= 3}
    linked = set(picture_aliases(gid))
    taken = known_lora_names() - linked
    counts = PG.name_counts()
    out = []
    for name, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        if name in taken:
            continue
        if name in linked or any(k in _simple(name) for k in keys):
            out.append((f"{name}  ({c} picture{'s' if c != 1 else ''})", name))
    for a in sorted(linked - set(counts)):
        out.append((f"{a}  (no pictures yet)", a))
    return out


def group_pictures(gid: str) -> list[dict]:
    members = LL.LIB.groups.get(gid) or []
    if not members:
        return []
    names, hashes = group_lora_names(members, gid)
    out = []
    for path, e, hit in PG.pictures_for(set(names), hashes):
        l, ep = names.get(hit.lstrip("#"), (None, None))
        if l is None:      # matched by hash only: the newest member with that hash
            l = next((m for m in members if str((m.sidecar or {}).get("sha256") or "").lower()[:10]
                      in {h[:10] for h in (e.get("hashes") or {})}), members[-1])
        out.append({"path": path, "e": e, "lora": l, "epoch": ep, "name": hit.lstrip("#"),
                    "strength": (e.get("loras") or {}).get(hit.lstrip("#"))})
    return out


def picture_versions(gid: str, pics: list | None = None) -> list:
    pics = group_pictures(gid) if pics is None else pics
    count: dict[str, int] = {}
    for p in pics:
        k = OLD if p["epoch"] == OLD else p["lora"].key
        count[k] = count.get(k, 0) + 1
    members = LL.LIB.groups.get(gid) or []
    out = [("All versions", "")] + [(f"{m.vlabel} · {m.stem} ({count[m.key]})", m.key)
                                    for m in reversed(members) if m.key in count]
    if OLD in count:
        out.append((f"Old names ({count[OLD]})", OLD))
    return out


def _when(t) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(float(t)))
    except (TypeError, ValueError):
        return ""


def render_pictures(gid: str, version: str, page) -> tuple[str, int, int]:
    """Gallery grid HTML for one page; returns (html, page, pages)."""
    members = LL.LIB.groups.get(gid) or []
    if not members:
        return '<div class="ll-wrap"><p class="ll-muted">Pick a LoRA above.</p></div>', 1, 1
    pics = group_pictures(gid)
    total_all = len(pics)
    if version == OLD:
        pics = [p for p in pics if p["epoch"] == OLD]
    elif version:
        pics = [p for p in pics if p["lora"].key == version and p["epoch"] != OLD]
    pages = max(1, -(-len(pics) // PIC_PAGE))
    page = min(max(1, int(page or 1)), pages)
    shown = pics[(page - 1) * PIC_PAGE: page * PIC_PAGE]
    if not shown:
        msg = ("No pictures found for this LoRA yet. Press <b>Scan for new pictures</b> to look in its picture folder."
               if not total_all else "No pictures of this version.")
        return f'<div class="ll-wrap"><p class="ll-muted">{msg}</p></div>', page, pages
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=6) as ex:        # first time only: thumbnails are kept in .thumbs
        uris = list(ex.map(lambda p: LL.thumb_data_uri(Path(p["path"]), 320), shown))
    blur = " ll-blur" if LL.LIB.blur(members[-1]) else ""
    tiles = []
    for p, uri in zip(shown, uris):
        l = p["lora"]
        st = p["strength"]
        if p["epoch"] == OLD:
            cap = "old name"
        else:
            cap = l.vlabel + (f" · ep {p['epoch']}" if p["epoch"] is not None else "")
        cap += f" · {st:g}" if isinstance(st, (int, float)) else ""
        img = f'<img src="{uri}" alt="" loading="lazy">' if uri else '<span class="ll-muted">no preview</span>'
        tip = Path(p["path"]).name + (f" · made with {p['name']}" if p["epoch"] == OLD else "")
        tiles.append(f'<button class="ll-ptile" data-p="{LL.esc(p["path"])}" onclick="llPicOpen(this.dataset.p, this)" '
                     f'title="{LL.esc(tip)}"><span class="ll-pimg{blur}">{img}</span>'
                     f'<span class="ll-pcap"><b>{LL.esc(cap)}</b><span>{LL.esc(_when(p["e"].get("t")))}</span></span></button>')
    head = (f'<div class="ll-muted" style="margin:2px 0 10px">{len(pics)} picture{"s" if len(pics) != 1 else ""}'
            f'{f" of {total_all}" if version else ""} made with <b>{LL.esc(LL.group_title(members))}</b> '
            f'({members[-1].base}, {len(members)} version{"s" if len(members) != 1 else ""}) &middot; newest first'
            f'{f" &middot; page {page} of {pages}" if pages > 1 else ""} &middot; click a picture to see it with its prompt</div>')
    return f'<div class="ll-wrap">{head}<div class="ll-pgrid">{"".join(tiles)}</div></div>', page, pages


def picture_details(path: str, gid: str):
    """(image, info markdown, prompt) for the viewer next to the grid."""
    e = PG.entry(path)
    if not e or not Path(path).exists():
        return None, "That picture is no longer there: press *Scan for new pictures*.", ""
    members = LL.LIB.groups.get(gid) or []
    names, _ = group_lora_names(members, gid) if members else ({}, set())
    lines = [f"**{Path(path).name}**  ·  {_when(e.get('t'))}", f"`{Path(path).parent}`", ""]
    used = []
    for n, st in (e.get("loras") or {}).items():
        mine = names.get(n)
        if mine and mine[1] == OLD:
            label = f"**{n}** (old name of this LoRA)"
        elif mine:
            label = f"**{mine[0].stem}** ({mine[0].vlabel}{f', epoch {mine[1]}' if mine[1] is not None else ''})"
        else:
            label = n
        used.append(f"{label}" + (f" at {st:g}" if isinstance(st, (int, float)) else ""))
    if used:
        lines.append("LoRAs: " + ", ".join(used))
    if e.get("ckpt"):
        lines.append(f"Model: `{e['ckpt']}`")
    if e.get("seed") is not None:
        lines.append(f"Seed: `{e['seed']}`")
    lines.append(f"Read from: {'the .json next to it' if e.get('src') == 'json' else 'the PNG metadata'}")
    return LL.load_image(Path(path), 2048), "  \n".join(lines), e.get("prompt") or ""
