# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""The look of the LoRA Studio pages: read from lora_studio.css (and custom.css, when you make one) next to the
program. Added after sdgui.CSS and lora_library.CSS by lora_studio.py, so it uses the same palette (--ll-*).

  lora_studio.css   the program's styles (an update replaces it)
  custom.css        your own changes: loaded last, never replaced by an update
"""
__version__ = "1.251026o"  # LoRA Studio version, see version.py
from pathlib import Path

import lspaths as _LP  # noqa: E402  (program/lspaths.py: ROOT, PROG, USER)
APP_DIR = Path(__file__).resolve().parent
FILE = APP_DIR / "lora_studio.css"
CUSTOM = _LP.USER / "custom.css"                # yours: in userdata/, never replaced


def _read(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except OSError:
        return ""


CSS = _read(FILE) + ("\n/* ---- custom.css ---- */\n" + _read(CUSTOM) if CUSTOM.is_file() else "")
