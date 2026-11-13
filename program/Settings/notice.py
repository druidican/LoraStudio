# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""The notice about LoRA Studio and the programs it uses: the very first page, confirmed once.

LoRA Studio is a GUI for sd-scripts and musubi-tuner, which belong to kohya-ss and their contributors. The notice says
so and has to be confirmed once: in install.sh, and on the first page in the browser (the setup page or LoRA Studio
itself). The answer is kept in notice_accepted.json next to the program; a new NOTICE_VERSION asks again. The same
text is in the NOTICE file and in Settings › Version › Licences and credits.
"""
from __future__ import annotations
__version__ = "1.011126"  # LoRA Studio version, see version.py

import json
import time

import lspaths as _LP  # noqa: E402  (program/lspaths.py: ROOT, PROG, USER)
HERE = _LP.USER
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

The same goes for PyTorch, ROCm, CUDA, NiceGUI, Ollama and ComfyUI, and for every model you train with or download:
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


def logo_data_url() -> str:
    import base64
    f = _LP.PROG / "logo" / "lorastudio_login.jpg"
    try:
        return "data:image/jpeg;base64," + base64.b64encode(f.read_bytes()).decode()
    except OSError:
        return ""
