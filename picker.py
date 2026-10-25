# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Datasets > Add pictures: scan a folder (and its subfolders) and pick the pictures that add to a dataset.

A picture is picked when it
  * is good enough: big enough for the LoRA type, sharp, readable, not a collage or screenshot (the quick checks
    without AI run first),
  * is not a near-copy: of near-identical shots only the best one is kept, and nothing that is already in the dataset,
  * shows one clear subject (the Ollama content check, only on the pictures the quick checks left),
  * adds something: a framing, outfit or place the dataset (and the pictures picked before it) has few of. A picture
    that only repeats what is there many times already is not picked, but shown with that reason.
Everything is shown with a reason; you tick or untick before anything is copied.
"""
from __future__ import annotations
__version__ = "1.251026g"  # LoRA Studio version, see version.py

import json
import os
import re
import shutil
import threading
import time
from collections import Counter
from pathlib import Path

import datacheck as DC

APP_DIR = Path(__file__).resolve().parent
LAST_FILE = APP_DIR / "gui_cache" / "picker_last.json"
STOP = threading.Event()
SKIP_DIRS = {DC.SET_ASIDE, "_reserve", "_originals", ".lsflow", "LoraManager", "model", "log", "logs", ".git", "__pycache__", ".thumbnails"}
SAME_LIMIT = 3          # a framing + outfit + place combination seen this often already adds little


def list_pictures(folder, recursive=True) -> list:
    root = Path(os.path.expanduser(folder or "")).resolve()
    if not root.is_dir():
        return []
    out = []
    if recursive:
        for d, dirs, files in os.walk(root):
            dirs[:] = sorted(x for x in dirs if not x.startswith(".") and x not in SKIP_DIRS)
            out += [Path(d) / f for f in sorted(files) if f.lower().endswith(DC.IMG_EXT)]
    else:
        out = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in DC.IMG_EXT)
    return out


def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip().lower())


def _combo(a: dict) -> tuple:
    return (_norm(a.get("framing")) or "other", _norm(a.get("outfit")), _norm(a.get("place")))


def scan(folder, recursive, existing: list, min_side: int, purpose: str, url: str, model: str, use_ai: bool,
         cache_get, cache_put, progress=None) -> dict:
    """cache_get(path) -> content answer or None; cache_put(path, answer). progress(text) for status lines.
    Returns {"pics": [{path, w, h, pick, why, adds, warn, sev}], ...} sorted best first."""
    STOP.clear()
    say = progress or (lambda *_: None)
    t0 = time.time()
    found = list_pictures(folder, recursive)
    existing = [Path(p) for p in existing]
    res = {"folder": str(folder), "recursive": bool(recursive), "at": time.strftime("%Y-%m-%d %H:%M"), "pics": [],
           "found": len(found), "ai": False, "error": ""}
    if not found:
        return res
    exist_set = {p.resolve() for p in existing}
    found = [p for p in found if p.resolve() not in exist_set]          # the dataset folder itself was picked
    say(f"Quick checks on {len(found)} pictures (and {len(existing)} already in the dataset) ...")
    tech = DC.technical(found + existing, min_side)
    info = {}
    ex_hash = [(p, tech[p]["hash"]) for p in existing if "hash" in tech.get(p, {})]
    for p in found:
        r = tech[p]
        e = {"path": str(p), "w": r.get("w", 0), "h": r.get("h", 0), "pick": True, "why": [], "adds": [], "warn": [],
             "score": (r.get("w", 0) * r.get("h", 0)) ** 0.5 + 4 * min(r.get("sharp") or 0, 400)}
        for sev, text in r["issues"]:
            if text.startswith("near-duplicate of"):
                other = text.split("near-duplicate of ", 1)[1].split(" (")[0]
                if not any(other.endswith(x.parent.name + "/" + x.name) for x in existing):
                    e["pick"] = False
                    e["why"].append(f"near-copy of {other} (that one is better)")
                continue
            if sev == DC.BAD:
                e["pick"] = False
                e["why"].append(text)
            else:
                e["warn"].append(text)
        if "hash" in r:
            twin = next((x for x, h in ex_hash if bin(r["hash"] ^ h).count("1") <= 5), None)
            if twin is not None:
                e["pick"] = False
                e["why"].insert(0, f"already in the dataset (like {twin.parent.name}/{twin.name})")
        info[p] = e
    # Ollama content check, only on the pictures the quick checks left (and the dataset, for the gaps)
    answers = {}
    if use_ai and url and model:
        todo = [p for p in found if info[p]["pick"]] + existing
        for i, p in enumerate(todo, 1):
            if STOP.is_set():
                res["error"] = "stopped"
                break
            d = cache_get(p)
            if d is None:
                el = time.time() - t0
                say(f"Ollama {i}/{len(todo)}: {p.parent.name}/{p.name}"
                    + (f" (about {int(el / max(i - 1, 1) * (len(todo) - i + 1) / 60) + 1} min left)" if i > 3 else ""))
                try:
                    d = DC.content(url, model, p)
                except Exception as ex:                     # noqa: BLE001
                    res["error"] = f"Ollama: {ex}"
                    break
                cache_put(p, d)
            answers[p] = d
        res["ai"] = bool(answers)
        for p in found:
            e, d = info[p], answers.get(p)
            if not e["pick"] or d is None:
                continue
            for sev, text in DC.content_issues(d, purpose):
                if sev == DC.BAD:
                    e["pick"] = False
                    e["why"].append(text)
                else:
                    e["warn"].append(text)
            e["framing"], e["outfit"], e["place"] = _combo(d)
    # gaps: what each picture adds to the dataset plus the pictures picked before it (best first)
    if res["ai"]:
        have = [_combo(answers[p]) for p in existing if p in answers]
        fr = Counter(c[0] for c in have)
        outfits = Counter(c[1] for c in have if c[1])
        places = Counter(c[2] for c in have if c[2])
        combos = Counter(have)
        cands = sorted((info[p] for p in found if info[p]["pick"] and "framing" in info[p]), key=lambda e: -e["score"])
        for e in cands:
            n = max(1, sum(fr.values()))
            c = (e["framing"], e["outfit"], e["place"])
            if fr[c[0]] / n < 0.2 or fr[c[0]] < 2:
                e["adds"].append(f"{c[0]} ({fr[c[0]]} so far)")
            if c[1] and outfits[c[1]] == 0:
                e["adds"].append(f"new outfit: {c[1]}")
            if c[2] and places[c[2]] == 0:
                e["adds"].append(f"new place: {c[2]}")
            if not e["adds"] and combos[c] >= SAME_LIMIT:
                e["pick"] = False
                e["why"].append(f"adds little: {combos[c]} pictures already have {c[0]}, {c[1] or 'same outfit'}, "
                                f"{c[2] or 'same place'}")
                continue
            fr[c[0]] += 1
            outfits[c[1]] += 1
            places[c[2]] += 1
            combos[c] += 1
    pics = sorted(info.values(), key=lambda e: (not e["pick"], -len(e["adds"]), -e["score"]))
    for e in pics:
        e.pop("score", None)
    res["pics"] = pics
    res["seconds"] = round(time.time() - t0, 1)
    try:
        LAST_FILE.parent.mkdir(exist_ok=True)
        LAST_FILE.write_text(json.dumps(res), encoding="utf-8")
    except OSError:
        pass
    return res


def safe_name(name: str) -> str:
    n = re.sub(r"[^\w.-]+", "_", (name or "").strip()).strip("._")
    return n


def _free(dest: Path) -> Path:
    if not dest.exists():
        return dest
    i = 2
    while True:
        c = dest.with_name(f"{dest.stem}_{i}{dest.suffix}")
        if not c.exists():
            return c
        i += 1


def copy_pictures(paths: list, target: Path, with_captions: bool = False) -> tuple:
    """Copy pictures into target (made when missing); a name already there gets _2, _3 ... Returns (copied, errors)."""
    target.mkdir(parents=True, exist_ok=True)
    copied, errors = [], []
    for p in paths:
        src = Path(p)
        try:
            dest = _free(target / src.name)
            shutil.copy2(src, dest)
            copied.append(dest)
            if with_captions:
                for c in DC.caption_of(src):
                    shutil.copy2(c, dest.with_suffix(c.suffix))
        except OSError as e:
            errors.append(f"{src.name}: {e}")
    return copied, errors
