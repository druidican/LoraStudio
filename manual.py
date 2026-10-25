# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""The Manual tab: a guide with text and pictures, read from manual/NN_name.md (pictures in manual/img/).

Each chapter is one Markdown file; its first "# " line is the title in the list on the left. Pictures are written
as ![what it shows](img/name.jpg) and are shown from the program folder. Links to another chapter:
[text](#chapter-name) where chapter-name is the file name without the number (04_training.md -> #training).
"""
from __future__ import annotations
__version__ = "1.251026g"  # LoRA Studio version, see version.py

import html
import re
from pathlib import Path

import gradio as gr

HERE = Path(__file__).resolve().parent
DIR = HERE / "manual"
FILE_RE = re.compile(r"^(\d+)_([\w-]+)\.md$")


def chapters() -> list[dict]:
    out = []
    for f in sorted(DIR.glob("*.md")):
        m = FILE_RE.match(f.name)
        if not m:
            continue
        text = f.read_text(encoding="utf-8")
        t = re.search(r"^#\s+(.+)$", text, re.M)
        out.append({"key": m.group(2), "file": f, "title": t.group(1).strip() if t else m.group(2)})
    return out


def _render(text: str) -> str:
    """Markdown → HTML, with the picture paths pointing at the program folder and chapter links that switch chapters."""
    from markdown_it import MarkdownIt
    md = MarkdownIt("commonmark", {"html": False}).enable("table")
    out = md.render(text)
    base = "/gradio_api/file=" + html.escape(str(DIR).replace("\\", "/"))
    out = re.sub(r'src="(img/[^"]+)"', lambda m: f'src="{base}/{m.group(1)}" loading="lazy"', out)
    out = re.sub(r'<a href="#([\w-]+)">', lambda m: f'<a href="#" onclick="lsManual(\'{m.group(1)}\'); return false;">', out)
    out = re.sub(r'<a href="(https?://[^"]+)">', r'<a href="\1" target="_blank" rel="noopener">', out)
    return out


def page(key: str) -> tuple[str, str]:
    """(html of the chapter, html of the list on the left)."""
    ch = chapters()
    if not ch:
        return ('<div class="mn-body"><p class="ls-empty">The manual is not installed yet. An update from a LoRA Studio '
                'older than 1.251026d copies only the program files: install the same LoraStudio.zip once more in '
                'Settings › Updates (it then adds the manual and the logo), or unpack its <code>manual/</code> folder '
                'next to the program.</p></div>', "")
    cur = next((c for c in ch if c["key"] == key), ch[0])
    i = ch.index(cur)
    nav = "".join(f'<button class="mn-item{" mn-on" if c is cur else ""}" onclick="lsManual(\'{c["key"]}\')">'
                  f'<span class="mn-n">{n + 1}</span>{html.escape(c["title"])}</button>' for n, c in enumerate(ch))
    body = _render(cur["file"].read_text(encoding="utf-8"))
    btn = lambda c, lab: f'<button class="tr-btn" onclick="lsManual(\'{c["key"]}\')">{html.escape(lab)}</button>'
    foot = ('<div class="mn-foot">' + (btn(ch[i - 1], "← " + ch[i - 1]["title"]) if i > 0 else "<span></span>")
            + (btn(ch[i + 1], ch[i + 1]["title"] + " →") if i + 1 < len(ch) else "") + "</div>")
    return f'<div class="mn-body">{body}{foot}</div>', f'<div class="mn-nav">{nav}</div>'


JS = """
window.lsManual = function(key){
  var el = document.querySelector('#mn_box textarea, #mn_box input'); if (!el) return;
  var setter = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(el), 'value').set;
  setter.call(el, key + '|' + Date.now()); el.dispatchEvent(new Event('input', {bubbles:true}));
  var top = document.getElementById('mn_body'); if (top) top.scrollIntoView({behavior:'smooth', block:'start'});
};
"""


def build(app):
    with gr.Row(equal_height=False, elem_id="mn_wrap"):
        with gr.Column(scale=1, min_width=230):
            nav = gr.HTML(elem_id="mn_navbox")
        with gr.Column(scale=4, elem_id="mn_body"):
            body = gr.HTML()
    box = gr.Textbox(elem_id="mn_box", show_label=False, container=False, elem_classes=["ls-hidden"])

    def show(v):
        return page((v or "").split("|")[0])
    box.change(show, box, [body, nav], show_progress="hidden")
    return {"load": (lambda: page(""), None, [body, nav])}     # loaded when the tab is opened, not with the page
