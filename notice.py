# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""The notice about LoRA Studio and the programs it uses: the very first page, confirmed once.

LoRA Studio is a GUI for sd-scripts and musubi-tuner, which belong to kohya-ss and their contributors. The notice says
so and has to be confirmed once: in install.sh, and on the first page in the browser (the setup page or LoRA Studio
itself). The answer is kept in notice_accepted.json next to the program; a new NOTICE_VERSION asks again. The same
text is in the NOTICE file and in Settings › Version › Licences and credits.
"""
from __future__ import annotations
__version__ = "1.251026n"  # LoRA Studio version, see version.py

import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILE = HERE / "notice_accepted.json"
NOTICE_VERSION = 1

TITLE = "LoRA Studio and the programs it uses"
TEXT = """\
LoRA Studio is a graphical front end, © 2026 L. G. H. Druidican, free software under the GNU General Public License
version 3. It comes with no warranty.

LoRA Studio trains with **sd-scripts** and **musubi-tuner**. They are made by **kohya-ss** and their contributors and
are licensed under the **Apache License 2.0** (some parts under the licences named in their own README). They are
separate programs: LoRA Studio does not include, own or license them, and L. G. H. Druidican has no rights in them.
LoRA Studio only starts them, and the installer only downloads them from their own repositories:

- sd-scripts: https://github.com/kohya-ss/sd-scripts
- musubi-tuner: https://github.com/kohya-ss/musubi-tuner

The same goes for PyTorch, ROCm, CUDA, Gradio, Ollama and ComfyUI, and for every model you train with or download:
each keeps its own owner and licence, and you use them under their terms.

The LoRAs you make, and the pictures you train them on, are yours and your responsibility."""


def accepted() -> bool:
    try:
        return int(json.loads(FILE.read_text(encoding="utf-8")).get("version", 0)) >= NOTICE_VERSION
    except Exception:                                    # noqa: BLE001
        return False


def accept(where: str = "page") -> None:
    FILE.write_text(json.dumps({"version": NOTICE_VERSION, "accepted": time.strftime("%Y-%m-%d %H:%M"),
                                "where": where}, indent=1), encoding="utf-8")


def plain() -> str:
    """The text without Markdown, for the terminal and the NOTICE file."""
    return TITLE + "\n\n" + TEXT.replace("**", "")


CSS = """
#ls_notice { position:fixed; inset:0; z-index:1000; overflow:auto; padding:32px 16px;
  background:var(--body-background-fill, #f6f4f0); }
#ls_notice > * { max-width:760px; margin-left:auto !important; margin-right:auto !important; }
#ls_notice .ls-notice-card { border:1px solid var(--border-color-primary, #ddd); border-radius:14px; padding:20px 24px;
  background:var(--block-background-fill, #fff); }
#ls_notice img.ls-notice-logo { display:block; width:min(420px, 100%); margin:0 auto 18px; border-radius:10px; }
"""


def build_gate(app, logo_url: str = ""):
    """The notice as a page over everything else, inside a gr.Blocks. Hidden when it was confirmed already.
    Returns the gate column (None when already confirmed)."""
    import gradio as gr
    if accepted():
        return None
    with gr.Column(elem_id="ls_notice") as gate:
        gr.HTML(f"<style>{CSS}</style>" + (f'<img class="ls-notice-logo" alt="" src="{logo_url}">' if logo_url else ""))
        with gr.Column(elem_classes=["ls-notice-card"]):
            gr.Markdown(f"## {TITLE}\n\n{TEXT}\n\nThis page is shown once. You can read it again in Settings › Version › "
                        "Licences and credits, and in the NOTICE file in the program folder.")
            ok = gr.Checkbox(label="I have read this and agree", value=False)
            go = gr.Button("Continue to LoRA Studio", variant="primary", interactive=False)
    ok.change(lambda v: gr.update(interactive=bool(v)), ok, go)

    def confirm(v):
        if not v:
            return gr.update()
        accept("page")
        return gr.update(visible=False)
    go.click(confirm, ok, gate)
    # the page is built once at start: a later visit (or another browser) after the confirmation hides it at load
    app.load(lambda: gr.update(visible=not accepted()), None, gate, show_progress="hidden")
    return gate


def logo_data_url() -> str:
    import base64
    f = HERE / "logo" / "lorastudio_login.jpg"
    try:
        return "data:image/jpeg;base64," + base64.b64encode(f.read_bytes()).decode()
    except OSError:
        return ""
