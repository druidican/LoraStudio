# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""A test: the Collection's Browse page made with NiceGUI instead of Gradio.

Start it with ./start.sh in this folder (it makes its own venv the first time). It opens http://127.0.0.1:7870.

It only READS: the scan LoRA Studio saved (userdata/library_cache.pkl), your LoRA Manager data and the Completed
marks. It writes nothing, not even thumbnails (they are made in memory). LoRA Studio itself is not changed and can
run at the same time.
"""
from __future__ import annotations

import hashlib
import io
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if not (ROOT / "program" / "lspaths.py").is_file():
    sys.exit("Put the try-nicegui folder inside your LoRA Studio folder (next to lora_studio.py) and start it again.")
sys.path.insert(0, str(ROOT / "program"))
import lspaths  # noqa: E402
lspaths.bootstrap()

from fastapi import Response  # noqa: E402
from nicegui import app, ui  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

import lora_library as LL  # noqa: E402

LIB = LL.LIB
LOADED = LIB.load_cache()            # the saved scan only: never a new scan (that would write the cache)

# --------------------------------------------------------------------------- thumbnails, in memory
PICS: dict[str, Path] = {}           # token -> picture
THUMBS: dict[str, bytes] = {}


def pic_url(p: Path | None) -> str:
    if not p:
        return ""
    tok = hashlib.md5(str(p).encode()).hexdigest()[:16]
    PICS[tok] = p
    return f"/thumb/{tok}"


@app.get("/thumb/{tok}")
def thumb(tok: str):
    if tok not in THUMBS:
        p = PICS.get(tok)
        if not p:
            return Response(status_code=404)
        try:
            with Image.open(p) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail((320, 640))
                b = io.BytesIO()
                im.save(b, "JPEG", quality=85)
                THUMBS[tok] = b.getvalue()
        except Exception:                                # noqa: BLE001
            return Response(status_code=404)
    return Response(THUMBS[tok], media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})


# --------------------------------------------------------------------------- the same choices as Browse
def mtime(l) -> float:
    try:
        return l.path.stat().st_mtime
    except OSError:
        return 0


def tiles(search: str, base: str, sort: str, only_preview: bool, favorites: bool) -> list[dict]:
    """One tile per dataset, exactly as Collection › Browse picks them (LL.render_gallery)."""
    q = (search or "").strip().lower()
    order = {b: i for i, b in enumerate(LIB.base_names())}
    by_ds: dict[str, list] = {}
    for l in LIB.loras.values():
        if l.dataset is not None:
            by_ds.setdefault(l.dataset.name.lower(), []).append(l)

    def matches(l):
        if base and base != "All" and l.base != base:
            return False
        if favorites and not (l.sidecar or {}).get("favorite"):
            return False
        return not q or q in " ".join([l.stem, l.display_name, l.character, l.base, l.group_name, l.dataset.name,
                                       " ".join(l.trigger_words)]).lower()

    pc = LL.purpose_colors(LIB)
    out = []
    for key, members in by_ds.items():
        hits = sorted((l for l in members if matches(l)), key=mtime, reverse=True)
        if not hits:
            continue
        shown = LIB.newest(hits[0].gid)
        top = shown if shown in hits else hits[0]
        src = top if top.preview else next((l for l in hits if l.preview), None)
        if only_preview and not src:
            continue
        types: dict[str, list] = {}
        for l in members:
            types.setdefault(l.base, []).append(l)
        chips = [{"label": LL.base_abbr(b), "done": LL.is_completed(types[b][0]), "cur": b == top.base, "base": b}
                 for b in sorted(types, key=lambda b: (order.get(b, 99), b))]
        purpose = LL._purpose_of(top.dataset.name) if pc["on"] else ""
        out.append({
            "name": top.dataset.name, "members": members, "top": top, "t": mtime(hits[0]),
            "pic": pic_url(src.preview) if src else "", "blur": bool(src and LIB.blur(src)),
            "chips": chips, "fav": any((l.sidecar or {}).get("favorite") for l in members),
            "status": max((l.status for l in members), key=lambda x: LL.STATUS_ORDER[x]),
            "colour": pc.get(purpose, "") if purpose else "", "purpose": dict(LL.PURPOSES).get(purpose, ""),
        })
    if sort == "Name":
        out.sort(key=lambda t: t["name"].lower())
    else:
        out.sort(key=lambda t: t["t"], reverse=True)
    return out


# --------------------------------------------------------------------------- the page
CSS = """
body { background:#F6F4F0; }
.body--dark { background:#16181C; }
.tile { width:200px; cursor:pointer; border-radius:12px; overflow:hidden; background:white; border:1px solid #E3E0DA;
        transition: transform .12s, box-shadow .12s; }
.body--dark .tile { background:#22252B; border-color:#33373F; }
.tile:hover { transform: translateY(-2px); box-shadow: 0 6px 18px rgba(0,0,0,.12); border-color:#E8A33D; }
.tile img { width:200px; height:260px; object-fit:cover; display:block; }
.tile .ph { width:200px; height:260px; display:flex; flex-direction:column; align-items:center; justify-content:center;
            background:#ECEAE5; color:#8A8F98; font-size:12px; }
.body--dark .tile .ph { background:#2C3036; }
.tile .ph b { font-size:56px; font-weight:600; color:#C7C2B8; }
.blur img { filter: blur(14px); }
.chip { font-size:11px; padding:1px 7px; border-radius:999px; border:1px solid #D5D1C9; color:#5C6370; }
.chip.cur { border-color:#E8A33D; }
.chip.done { background:#21857C; border-color:#21857C; color:white; }
.dot { width:8px; height:8px; border-radius:50%; display:inline-block; }
.dot.warning { background:#D9902B; } .dot.problem { background:#C0392B; }
"""


@ui.page("/")
def page():
    ui.add_css(CSS)
    dark = ui.dark_mode()
    state = {"search": "", "base": "All", "sort": "Newest first", "only_preview": False, "favorites": False}

    with ui.header().classes("items-center bg-[#1C1F24] text-white"):
        ui.label("LoRA Studio").classes("text-xl font-semibold")
        ui.label("Collection › Browse · NiceGUI test").classes("text-sm opacity-70")
        ui.space()
        ui.button("Dark / Normal", on_click=dark.toggle).props("flat color=white no-caps")

    if not LOADED:
        ui.label("No saved scan found. Start LoRA Studio once (and open the Collection), then start this test again.") \
            .classes("m-8 text-lg")
        return

    with ui.row().classes("w-full items-end gap-4 px-6 pt-4"):
        ui.input("Search", placeholder="name, character or trigger word",
                 on_change=lambda e: (state.update(search=e.value), grid.refresh())) \
            .props("clearable outlined dense").classes("w-80")
        ui.select(["All"] + LIB.base_names(), value="All", label="Base model",
                  on_change=lambda e: (state.update(base=e.value), grid.refresh())).props("outlined dense").classes("w-48")
        ui.select(["Newest first", "Name"], value="Newest first", label="Sort",
                  on_change=lambda e: (state.update(sort=e.value), grid.refresh())).props("outlined dense").classes("w-40")
        ui.switch("Only with preview picture",
                  on_change=lambda e: (state.update(only_preview=e.value), grid.refresh()))
        ui.switch("Favorites only", on_change=lambda e: (state.update(favorites=e.value), grid.refresh()))

    def details(t: dict):
        """In LoRA Studio a tile opens the dataset page; here a short look at its LoRAs."""
        groups: dict[str, list] = {}
        for l in t["members"]:
            groups.setdefault(l.gid, []).append(l)
        with ui.dialog() as dlg, ui.card().classes("min-w-[520px] max-w-[760px]"):
            ui.label(t["name"]).classes("text-xl font-semibold")
            if t["purpose"]:
                ui.label(t["purpose"]).classes("text-sm opacity-70")
            for gid, _ in groups.items():
                m = LIB.groups.get(gid) or groups[gid]
                n = LIB.newest(gid) or m[-1]
                st, issues = LL.group_status(m)
                best = LL.best_member(m)
                with ui.card().props("flat bordered").classes("w-full"):
                    with ui.row().classes("items-center gap-2"):
                        ui.label(LL.group_title(m)).classes("font-semibold")
                        ui.badge(st, color={"ok": "teal", "warning": "orange", "problem": "red"}.get(st, "grey"))
                        if best:
                            ui.badge(f"★ best {best.vlabel}", color="amber")
                        if LL.is_completed(n):
                            ui.badge("✓ Completed", color="teal")
                    ui.label(f"{n.base} · {len(m)} version{'s' if len(m) != 1 else ''}: "
                             + ", ".join(x.vlabel for x in m)).classes("text-sm")
                    if n.trigger_words:
                        ui.label("Trigger: " + ", ".join(n.trigger_words)).classes("text-sm opacity-80")
                    for _, text in issues[:3]:
                        ui.label(text).classes("text-xs opacity-70")
            ui.label("In LoRA Studio a click on a tile opens the dataset page (its LoRA part).") \
                .classes("text-xs opacity-60")
            ui.button("Close", on_click=dlg.close).props("flat no-caps")
        dlg.open()

    @ui.refreshable
    def grid():
        t0 = time.perf_counter()
        ts = tiles(**state)
        n_files = sum(1 for l in LIB.loras.values() if l.dataset is not None)
        info = ui.label().classes("px-6 text-sm opacity-70")
        with ui.row().classes("w-full gap-4 px-6 pb-10"):
            for t in ts:
                style = f"border:2px solid {t['colour']}" if t["colour"] else ""
                with ui.element("div").classes("tile" + (" blur" if t["blur"] else "")).style(style) \
                        .on("click", lambda _, t=t: details(t)):
                    if t["pic"]:
                        ui.html(f'<img src="{t["pic"]}" loading="lazy" alt="">', sanitize=False)
                    else:
                        ui.html(f'<div class="ph"><b>{t["name"][:1].upper()}</b>No preview yet</div>', sanitize=False)
                    with ui.column().classes("p-2 gap-1"):
                        with ui.row().classes("items-center gap-1 no-wrap"):
                            if t["fav"]:
                                ui.label("★").classes("text-amber-500")
                            if t["status"] != "ok":
                                ui.html(f'<span class="dot {t["status"]}"></span>', sanitize=False)
                            ui.label(t["name"]).classes("font-semibold truncate")
                        with ui.row().classes("gap-1"):
                            for c in t["chips"]:
                                ui.label(("✓" if c["done"] else "") + c["label"]) \
                                    .classes("chip" + (" done" if c["done"] else "") + (" cur" if c["cur"] else "")) \
                                    .tooltip(c["base"] + (" · Completed" if c["done"] else ""))
        info.text = (f"{len(ts)} of {len({l.dataset.name.lower() for l in LIB.loras.values() if l.dataset})} datasets "
                     f"({n_files} LoRA files) · built in {1000 * (time.perf_counter() - t0):.0f} ms")

    grid()


ui.run(host="127.0.0.1", port=7870, title="LoRA Studio · NiceGUI test", reload=False, show=True,
       show_welcome_message=True)
