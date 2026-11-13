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


async def stream(gen, show):
    """Run a generator of progress lines in a thread; show(line) for each line (on the page). Returns the
    generator's return value (StopIteration.value)."""
    from nicegui import run as _run
    it = iter(gen)
    done = object()

    def step():
        try:
            return next(it)
        except StopIteration as e:
            return (done, e.value)
    while True:
        r = await _run.io_bound(step)
        if isinstance(r, tuple) and len(r) == 2 and r[0] is done:
            return r[1]
        show(r)


class CropPanel:
    """A picture with four edge sliders (percent), a shape and a note: on_change(l, t, r, b, aspect) -> (picture, note)
    is called when an edge is let go; values are fractions 0-1."""

    def __init__(self, on_change):
        self.on_change = on_change
        self.img = ui.image("").classes("w-full rounded").props("fit=contain")
        self.aspect = ui.toggle(["free", "3:4", "1:1", "4:3"], value="free",
                                on_change=lambda: self.changed()).props("dense no-caps unelevated")
        with ui.grid(columns=2).classes("w-full gap-x-4"):
            self.l = self._slider("Left %", 0)
            self.r = self._slider("Right %", 100)
            self.t = self._slider("Top %", 0)
            self.b = self._slider("Bottom %", 100)
        self.note = ui.label("").classes("ls-note")

    def _slider(self, label, v):
        with ui.column().classes("gap-0 w-full"):
            ui.label(label).classes("ls-note")
            s = ui.slider(min=0, max=100, step=1, value=v).props("label")
            s.on("change", lambda: self.changed())
        return s

    def set(self, box, picture=None, note=""):
        self.l.value, self.t.value, self.r.value, self.b.value = (round(box[0] * 100), round(box[1] * 100),
                                                                  round(box[2] * 100), round(box[3] * 100))
        if picture is not None:
            self.img.set_source(picture)
        self.note.text = note

    def box(self):
        return self.l.value / 100, self.t.value / 100, self.r.value / 100, self.b.value / 100

    def changed(self):
        pic, n = self.on_change(*self.box(), self.aspect.value)
        if pic is not None:
            self.img.set_source(pic)
        self.note.text = n or ""


PAINT_JS = """
<script>
window.lsPaintInit = function(id){
  var box = document.getElementById(id); if (!box) return;
  var img = box.querySelector('img'), cv = box.querySelector('canvas');
  var setup = function(){ cv.width = img.naturalWidth; cv.height = img.naturalHeight; };
  if (img.complete) setup(); else img.onload = setup;
  var ctx = cv.getContext('2d'), down = false, size = 28;
  var at = function(e){ var r = cv.getBoundingClientRect(); return [(e.clientX - r.left) * cv.width / r.width,
                                                                     (e.clientY - r.top) * cv.height / r.height]; };
  var dot = function(e){ var p = at(e); var s = size * cv.width / cv.getBoundingClientRect().width;
                         ctx.fillStyle = 'rgba(255,59,48,0.75)'; ctx.beginPath(); ctx.arc(p[0], p[1], s / 2, 0, 7); ctx.fill(); };
  cv.onpointerdown = function(e){ down = true; cv.setPointerCapture(e.pointerId); dot(e); };
  cv.onpointermove = function(e){ if (down) dot(e); };
  cv.onpointerup = function(){ down = false; };
  box.lsClear = function(){ ctx.clearRect(0, 0, cv.width, cv.height); };
  box.lsSize = function(v){ size = v; };
};
window.lsPaintMask = function(id){ var cv = document.querySelector('#' + id + ' canvas'); return cv ? cv.toDataURL('image/png') : ''; };
</script>
"""


class PaintPanel:
    """A picture to paint over with a red brush; mask() gives the painted part (an 'L' picture, white = painted)."""

    def __init__(self):
        import uuid
        self.id = "paint" + uuid.uuid4().hex[:8]
        ui.add_head_html(PAINT_JS)
        self.html = ui.html("", sanitize=False).classes("w-full")
        with ui.row().classes("items-center gap-3"):
            ui.label("Brush").classes("ls-note")
            ui.slider(min=6, max=120, value=28, on_change=lambda e: ui.run_javascript(
                f"var b = document.getElementById('{self.id}'); if (b && b.lsSize) b.lsSize({int(e.value)});")) \
                .classes("w-48")
            ui.button("Clear", on_click=lambda: ui.run_javascript(
                f"var b = document.getElementById('{self.id}'); if (b && b.lsClear) b.lsClear();")).props("flat dense no-caps")

    def show(self, url: str):
        self.html.content = (f'<div id="{self.id}" style="position:relative;display:inline-block;max-width:100%">'
                             f'<img src="{url}" style="max-width:100%;display:block">'
                             '<canvas style="position:absolute;inset:0;width:100%;height:100%;cursor:crosshair;'
                             'touch-action:none"></canvas></div>')
        ui.timer(0.3, lambda: ui.run_javascript(f"lsPaintInit('{self.id}')"), once=True)

    async def mask(self):
        import base64
        import io
        from PIL import Image
        url = await ui.run_javascript(f"lsPaintMask('{self.id}')")
        if not url or "," not in url:
            return None
        im = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGBA")
        return im.getchannel("A")
