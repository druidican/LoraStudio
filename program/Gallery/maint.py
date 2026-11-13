# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Collection › Maintenance, without page code: Personal sync, LoRA Manager's example folder, the training cards of
all LoRAs, and freeing disk space (training leftovers, orphaned LoRA Manager files). Everything removed goes to the
desktop Trash."""
from __future__ import annotations
__version__ = "1.131126c"  # LoRA Studio version, see version.py

import os
from pathlib import Path

import lora_library as LL

LIB = LL.LIB


def sync() -> str:
    msg = LL.personal_sync(LIB, None)
    return msg


def set_example_folder(d: str) -> str:
    d = (d or "").strip()
    if d and not Path(os.path.expanduser(d)).is_dir():
        return f"❌ Not a folder: {d}"
    LIB.data["lm_examples_dir"] = d
    LIB.save_data()
    LL._LM_ROOTS.clear()
    r = LL.lm_example_roots(LIB)
    return (f"✔ Saved. Example pictures are read from {r[0]}. Press Sync now." if r
            else "✔ Saved (read from LoRA Manager's settings).")


def training_cards(force: bool) -> str:
    import trainingcard as TC
    return TC.scan_all(LIB, bool(force), None)


def find_leftovers() -> tuple[str, list[str]]:
    return LL.leftovers_md(LIB)


def trash_leftovers(paths: list[str]) -> str:
    if not paths:
        return "Press Find leftovers first."
    HK = LL.HK
    busy = LL.busy_datasets()
    todo = [p for p in paths if not any(Path(p).is_relative_to(b) for b in busy)]
    size = sum(HK.size_of(Path(p)) for p in todo)
    moved, err = HK.trash(todo)
    msg = f"✔ Moved {moved} item(s), {HK.human(size)}, to the Trash."
    if len(todo) < len(paths):
        msg += f" {len(paths) - len(todo)} item(s) skipped: a training run started in that dataset."
    if err:
        msg += "\n\nCould not move: " + "; ".join(err[:5])
    return msg


def find_orphans() -> tuple[str, list[tuple[str, str]]]:
    md, choices, _ = LL.orphans_list(LIB)
    return md, choices


def trash_orphans(picked: list[str]) -> str:
    files = [f for group in (picked or []) for f in group.split("\x1f") if f]
    if not files:
        return "Nothing ticked."
    moved, err = LL.HK.trash(files)
    return f"✔ Moved {moved} file(s) / folder(s) to the Trash." + ("\n\nCould not move: " + "; ".join(err[:5]) if err else "")
