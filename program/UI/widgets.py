# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Shared parts of the NiceGUI pages: picture links, chips, cards, a confirm dialog.

Pictures are never put into the page itself: pic_url() gives a short link (/pic/<token>) that the program answers with
the Collection's cached thumbnail (lora_library.thumb_file, made when the browser first asks for it). Other files
(full pictures, the manual's pictures, reports) are linked with lspaths.file_url and served from the allowed folders
only (allowed_roots).
"""
from __future__ import annotations
__version__ = "1.131126b"  # LoRA Studio version, see version.py

import hashlib
from pathlib import Path

from fastapi import Response
from fastapi.responses import FileResponse
from nicegui import app, ui

import lspaths

_PICS: dict[str, Path] = {}


def _token(p: Path) -> str:
    tok = hashlib.md5(str(p).encode()).hexdigest()[:20]
    _PICS[tok] = p
    return tok


def pic_url(p: Path | str | None, width: int = 360) -> str:
    """A link to a thumbnail of the picture (made once, cached in userdata/.thumbs)."""
    if not p:
        return ""
    return f"/pic/{_token(Path(p))}?w={int(width)}"


@app.get("/pic/{tok}")
def _pic(tok: str, w: int = 360):
    import lora_library as LL
    p = _PICS.get(tok)
    tp = LL.thumb_file(p, max(64, min(w, 1600))) if p else None
    if not tp:
        return Response(status_code=404)
    return FileResponse(tp, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})


_ROOTS = {"t": 0.0, "roots": []}


def allowed_roots() -> list[Path]:
    """The folders whose files a page may link (lspaths.file_url): the program's own (thumbnails, manual, logo,
    reports), your datasets, LoRA folders, sd-scripts and LoRA Manager's example pictures."""
    import time
    if time.time() - _ROOTS["t"] < 30:
        return _ROOTS["roots"]
    import os
    import lora_library as LL
    s = LL.LIB.settings
    r = [LL.THUMB_DIR, lspaths.PROG / "manual", lspaths.PROG / "logo", lspaths.USER / "reports"]
    r += [Path(os.path.expanduser(d)) for d in list(s.get("dataset_dirs") or []) + list(s.get("lora_dirs") or [])
          + list(s.get("script_dirs") or [])]
    if s.get("sd_scripts_dir"):
        r.append(Path(os.path.expanduser(s["sd_scripts_dir"])))
    try:
        r += [Path(x) for x in LL.lm_example_roots(LL.LIB)]
    except Exception:                                    # noqa: BLE001
        pass
    _ROOTS["roots"] = [x.resolve() for x in r if str(x)]
    _ROOTS["t"] = time.time()
    return _ROOTS["roots"]


@app.get(lspaths.FILE_ROUTE + "{path:path}")
def _lsfile(path: str):
    p = Path("/" + path.lstrip("/")) if not path.startswith("/") else Path(path)
    try:
        rp = p.resolve()
    except OSError:
        return Response(status_code=404)
    if not rp.is_file() or not any(rp.is_relative_to(r) for r in allowed_roots()):
        return Response(status_code=404)
    return FileResponse(rp, headers={"Cache-Control": "max-age=86400"})


# --------------------------------------------------------------------------- small parts
def chip(text: str, kind: str = "", tip: str = ""):
    """A small rounded label: kind '' | ok | warn | bad | cur."""
    c = ui.label(text).classes("ls-chip" + (f" {kind}" if kind else ""))
    if tip:
        c.tooltip(tip)
    return c


def heading(text: str, sub: str = ""):
    with ui.column().classes("gap-0"):
        ui.label(text).classes("ls-h")
        if sub:
            ui.label(sub).classes("ls-note")


def note(text: str):
    return ui.label(text).classes("ls-note")


def card():
    return ui.card().props("flat bordered").classes("ls-card")


async def confirm(question: str, yes: str = "Yes", no: str = "Cancel") -> bool:
    with ui.dialog() as d, ui.card():
        ui.label(question)
        with ui.row().classes("justify-end w-full"):
            ui.button(no, on_click=lambda: d.submit(False)).props("flat no-caps")
            ui.button(yes, on_click=lambda: d.submit(True)).props("unelevated no-caps color=primary")
    return bool(await d)
