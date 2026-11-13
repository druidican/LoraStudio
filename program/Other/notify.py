# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Short messages from the program's logic to whoever shows them (the page sets SHOW; without a page they are
printed). kind: info | warning."""
from __future__ import annotations
__version__ = "1.131126c"  # LoRA Studio version, see version.py

SHOW = None


def info(msg: str):
    _say(msg, "info")


def warning(msg: str):
    _say(msg, "warning")


def _say(msg, kind):
    if SHOW is not None:
        try:
            SHOW(msg, kind)
            return
        except Exception:                                # noqa: BLE001
            pass
    print(f"[{kind}] {msg}")
