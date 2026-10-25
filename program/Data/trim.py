# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Datasets > Trim: keep the best pictures of a big dataset (up to a number you set), move the rest to _reserve/.

1. You pick three guide pictures: a full body, a face close-up and one more. The Ollama model lists the details of
   the character it sees in them (hair, eyes, build, horns, a signature item ...), plus the traits locked in
   traits.txt. The guides are a guideline, not "the best picture": each picture is checked against the list.
2. Quick checks without AI: broken, too small, blurry, near-copies (only the best of near-identical shots stays).
3. For every picture that is left, the model answers per detail: matches / different / not visible. A picture
   passes when about 8 of 10 VISIBLE details match (the share can be set); a detail that cannot be seen does not count
   against it.
4. Pictures are ranked by quality (size, sharpness, no content problems) and by how many details they confirm, then
   picked best first while keeping a framing mix (close-ups, upper body, some full body) and skipping pictures that
   only repeat what is picked already (same framing, outfit and place). It stops at the number you set, or earlier
   when nothing new is left; good pictures that did not fit are shown unticked as "over the limit".
5. You review, then the unticked pictures move with their captions to <dataset>/_reserve/<repeat folder>/
   (sd-scripts ignores that folder). "Move all back" undoes it.
"""
from __future__ import annotations
__version__ = "1.251026o"  # LoRA Studio version, see version.py

import hashlib
import json
import re
import shutil
import threading
import time
from collections import Counter
from pathlib import Path

import datacheck as DC

import lspaths as _LP  # noqa: E402  (program/lspaths.py: ROOT, PROG, USER)
APP_DIR = _LP.USER
CACHE_FILE = APP_DIR / "gui_cache" / "trim_cache.json"
RESERVE = "_reserve"
MOVED_FILE = "moved.json"
STOP = threading.Event()
SAME_LIMIT = 3
QUOTA = {"close-up": 0.35, "upper body": 0.35, "full body": 0.2, "other": 0.1}   # character datasets
GUIDE_SLOTS = ("Full body", "Face close-up", "Random")

GUIDE_PROMPT = """This picture shows a character that a LoRA is trained on. List the details of the CHARACTER ITSELF
that you can see: hair colour, length and style, eye colour, skin, build, body or species features (horns, ears, tail,
wings, fur, freckles, scars, tattoos, piercings), and a signature item the character clearly always has.
Do NOT list clothes that could change, pose, expression, place, light or camera.
Short phrases of 2-4 words, e.g. "long white hair", "blue eyes", "curved ram horns".
Answer only JSON: {"details": ["...", "..."]}"""

CHECK_PROMPT = """The character in this picture should have these details:
{lines}
For every detail answer "yes" (visible and it matches), "no" (visible but clearly different) or "hidden" (cannot be
seen in this picture, e.g. covered, turned away, out of frame, too small).
Answer only JSON: {{"answers": [{{"n": 1, "a": "yes"}}, ...]}}"""


def _key(*parts) -> str:
    return hashlib.sha1("\x1f".join(str(p) for p in parts).encode()).hexdigest()[:20]


def _cache_load() -> dict:
    try:
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except Exception:                                       # noqa: BLE001
        return {}


def _cache_save(c: dict):
    try:
        CACHE_FILE.parent.mkdir(exist_ok=True)
        tmp = CACHE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(c), encoding="utf-8")
        tmp.replace(CACHE_FILE)
    except OSError:
        pass


def _sig(p: Path) -> str:
    st = p.stat()
    return f"{p}|{st.st_size}|{int(st.st_mtime)}"


def _ask_image(url, model, prompt, img, num_predict=400):
    import captioner as CAP
    r = CAP._req(url, "/api/chat", {
        "model": model, "stream": False, "format": "json", "keep_alive": "5m",
        "options": {"temperature": 0.0, "num_predict": num_predict},
        "messages": [{"role": "user", "content": prompt, "images": [CAP.image_b64(img, 896)]}],
    }, timeout=600)
    txt = re.sub(r"<think>.*?</think>", "", ((r.get("message") or {}).get("content") or r.get("response") or ""), flags=re.S)
    txt = re.sub(r"^```(json)?|```$", "", txt.strip()).strip()
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", txt, re.S)
        try:
            return json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            return {}


def _norm_detail(d: str) -> str:
    return re.sub(r"\s+", " ", str(d or "").strip().lower().strip(".,;:'\""))


def guide_details(url, model, guides: list, locked: list, cache: dict) -> list:
    """The detail list: what the model sees in the guide pictures (a detail named in any guide), plus the locked
    traits (they come first). At most 12."""
    out = []
    for d in locked:
        d = _norm_detail(d)
        if d and d not in out:
            out.append(d)
    counts = Counter()
    for g in guides:
        g = Path(g)
        k = "g:" + _key(_sig(g), model)
        if k not in cache:
            ans = _ask_image(url, model, GUIDE_PROMPT, g)
            cache[k] = [_norm_detail(x) for x in (ans.get("details") or []) if isinstance(x, str)] if isinstance(ans, dict) else []
        counts.update(dict.fromkeys(cache[k], 1))
    for d, _ in counts.most_common():
        if d and d not in out and not any(d in o or o in d for o in out):
            out.append(d)
    return out[:12]


def check_details(url, model, img: Path, details: list, cache: dict) -> dict:
    """{detail: "yes" | "no" | "hidden"}"""
    k = "c:" + _key(_sig(img), model, *details)
    if k not in cache:
        lines = "\n".join(f"{i}. {d}" for i, d in enumerate(details, 1))
        ans = _ask_image(url, model, CHECK_PROMPT.format(lines=lines), img, num_predict=60 + 20 * len(details))
        res = {}
        for a in (ans.get("answers") or []) if isinstance(ans, dict) else []:
            try:
                n = int(a.get("n")) - 1
            except (TypeError, ValueError, AttributeError):
                continue
            v = str(a.get("a") or "").strip().lower()
            if 0 <= n < len(details):
                res[details[n]] = "yes" if v.startswith("y") else ("no" if v.startswith("n") and v != "not visible" else "hidden")
        cache[k] = res
    return cache[k]


def _framing(d: dict) -> str:
    f = re.sub(r"\s+", " ", str((d or {}).get("framing") or "").strip().lower())
    return f if f in QUOTA else "other"


def scan(images: list, guides: list, target: int, match: float, purpose: str, min_side: int, url: str, model: str,
         locked: list, content_get, content_put, say=None) -> dict:
    """images: the dataset's pictures. Returns {"details", "pics": [{path, keep, why, good, warn, framing, ...}]}."""
    STOP.clear()
    say = say or (lambda *_: None)
    t0 = time.time()
    images = [Path(p) for p in images]
    guides = [Path(g) for g in guides if g]
    gset = {g.resolve() for g in guides}
    res = {"at": time.strftime("%Y-%m-%d %H:%M"), "target": target, "match": match, "details": [], "pics": [],
           "error": "", "total": len(images)}
    say(f"Quick checks on {len(images)} pictures ...")
    tech = DC.technical(images, min_side)
    info = {}
    for p in images:
        r = tech.get(p, {"issues": []})
        e = {"path": str(p), "w": r.get("w", 0), "h": r.get("h", 0), "keep": True, "out": False, "why": [], "good": [],
             "warn": [], "guide": p.resolve() in gset,
             "q": (r.get("w", 0) * r.get("h", 0)) ** 0.5 + 3 * min(r.get("sharp") or 0, 500)}
        for sev, text in r["issues"]:
            if text.startswith("near-duplicate of") or sev == DC.BAD:
                e["out"] = True
                e["why"].append(text.replace("near-duplicate of", "near-copy of"))
            else:
                e["warn"].append(text)
                e["q"] *= 0.85
        if e["guide"]:
            e["out"], e["why"] = False, []
            e["good"].append("guide picture")
        info[p] = e
    cache = _cache_load()
    try:
        say("Reading the details of the character in the guide pictures ...")
        details = guide_details(url, model, guides, locked, cache) if (url and model and guides) else list(locked)
    except Exception as ex:                                 # noqa: BLE001
        details, res["error"] = list(locked), f"Ollama: {ex}"
    res["details"] = details
    todo = [p for p in images if not info[p]["out"]]
    answers = {}
    if url and model and not res["error"]:
        for i, p in enumerate(todo, 1):
            if STOP.is_set():
                res["error"] = "stopped"
                break
            el = time.time() - t0
            say(f"Ollama {i}/{len(todo)}: {p.parent.name}/{p.name}"
                + (f" (about {int(el / max(i - 1, 1) * (len(todo) - i + 1) / 60) + 1} min left)" if i > 3 else ""))
            try:
                d = content_get(p)
                if d is None:
                    d = DC.content(url, model, p)
                    content_put(p, d)
                answers[p] = d
                det = check_details(url, model, p, details, cache) if details else {}
            except Exception as ex:                         # noqa: BLE001
                res["error"] = f"Ollama: {ex}"
                break
            e = info[p]
            e["framing"] = _framing(d)
            e["combo"] = (e["framing"], str(d.get("outfit") or "").strip().lower(), str(d.get("place") or "").strip().lower())
            for sev, text in DC.content_issues(d, purpose):
                if sev == DC.BAD and not e["guide"]:
                    e["out"] = True
                    e["why"].append(text)
                else:
                    e["warn"].append(text)
                    e["q"] *= 0.85
            yes = [k for k, v in det.items() if v == "yes"]
            no = [k for k, v in det.items() if v == "no"]
            e["yes"], e["no"] = len(yes), len(no)
            seen = len(yes) + len(no)
            if seen >= 2 and len(yes) / seen < match and not e["guide"]:
                e["out"] = True
                e["why"].append(f"details differ: {', '.join(no[:3])} ({len(yes)} of {seen} visible details match)")
            elif seen:
                e["good"].append(f"confirms {len(yes)} of {seen} visible details")
            else:
                e["warn"].append("few of the details are visible")
            if i % 10 == 0:
                _cache_save(cache)
        _cache_save(cache)
    res["ai"] = bool(answers)
    # rank: quality (within this dataset) and confirmed details
    cand = [info[p] for p in images if not info[p]["out"]]
    qs = sorted(e["q"] for e in cand)
    pct = lambda q: (qs.index(q) + 1) / len(qs) if qs else 0
    nd = max(1, len(details))
    for e in cand:
        e["rank"] = 0.5 * pct(e["q"]) + 0.5 * (e.get("yes", 0) / nd) + (10 if e["guide"] else 0)
    cand.sort(key=lambda e: -e["rank"])
    # pick: best first, a framing mix (characters), no repeats of the same framing + outfit + place
    keep, held, fr, combos = [], [], Counter(), Counter()
    quota = {k: max(1, round(v * target)) for k, v in QUOTA.items()} if purpose == "character" and res["ai"] else None
    for e in cand:
        if len(keep) >= target and not e["guide"]:
            held.append(e)
            continue
        c = e.get("combo")
        if c and combos[c] >= SAME_LIMIT and not e["guide"]:
            e["keep"] = False
            e["why"].append(f"adds little: {combos[c]} kept pictures already have this framing, outfit and place")
            continue
        if quota and not e["guide"] and fr[e["framing"]] >= quota[e["framing"]]:
            held.append(e)
            continue
        keep.append(e)
        fr[e.get("framing", "other")] += 1
        if c:
            combos[c] += 1
    for e in held:                                          # fill up from the framings that were full
        c = e.get("combo")
        if len(keep) < target and not (c and combos[c] >= SAME_LIMIT):
            keep.append(e)
            fr[e.get("framing", "other")] += 1
            if c:
                combos[c] += 1
        else:
            e["keep"] = False
            e["why"].append("good, but over the limit" if len(keep) >= target else
                            "adds little: same framing, outfit and place as kept pictures")
    for e in keep:
        e["keep"] = True
        if e.get("framing"):
            e["good"].append(e["framing"])
    for e in info.values():
        if e["out"]:
            e["keep"] = False
    res["framing"] = dict(fr)
    order = {id(e): i for i, e in enumerate(keep)}
    pics = sorted(info.values(), key=lambda e: (not e["keep"], order.get(id(e), 0), -e.get("rank", 0)))
    for e in pics:
        for k in ("q", "rank", "combo", "out"):
            e.pop(k, None)
    res["pics"] = pics
    res["seconds"] = round(time.time() - t0, 1)
    return res


# --------------------------------------------------------------------------- moving to and from _reserve
def move_to_reserve(root: Path, paths: list) -> tuple:
    """Move pictures (with their captions) to <root>/_reserve/<repeat folder>/. Returns (moved, errors)."""
    root = Path(root)
    res_dir = root / RESERVE
    log_file = res_dir / MOVED_FILE
    try:
        log = json.loads(log_file.read_text(encoding="utf-8"))
    except Exception:                                       # noqa: BLE001
        log = {}
    moved, errors = 0, []
    for p in paths:
        src = Path(p)
        try:
            rel = src.parent.resolve().relative_to(root.resolve())
        except ValueError:
            errors.append(f"{src.name}: not in this dataset")
            continue
        dest_dir = res_dir / rel
        dest_dir.mkdir(parents=True, exist_ok=True)
        try:
            for f in [src] + DC.caption_of(src):
                dest = dest_dir / f.name
                shutil.move(str(f), str(dest))
                log[str(dest.relative_to(root))] = str(f.relative_to(root))
            moved += 1
        except OSError as e:
            errors.append(f"{src.name}: {e}")
    res_dir.mkdir(exist_ok=True)
    log_file.write_text(json.dumps(log, indent=1), encoding="utf-8")
    return moved, errors


def reserve_list(root: Path) -> list:
    d = Path(root) / RESERVE
    return sorted(p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in DC.IMG_EXT) if d.is_dir() else []


def move_back(root: Path) -> tuple:
    """Every picture in _reserve back to where it came from (or its repeat folder of the same name)."""
    root = Path(root)
    res_dir = root / RESERVE
    try:
        log = json.loads((res_dir / MOVED_FILE).read_text(encoding="utf-8"))
    except Exception:                                       # noqa: BLE001
        log = {}
    n, errors = 0, []
    for f in sorted(p for p in res_dir.rglob("*") if p.is_file() and p.name != MOVED_FILE) if res_dir.is_dir() else []:
        rel = str(f.relative_to(root))
        dest = root / log.get(rel, str(f.relative_to(res_dir)))
        if dest.exists():
            dest = dest.with_name(f"{dest.stem}_back{dest.suffix}")
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), str(dest))
            n += f.suffix.lower() in DC.IMG_EXT
            log.pop(rel, None)
        except OSError as e:
            errors.append(f"{f.name}: {e}")
    if res_dir.is_dir():
        (res_dir / MOVED_FILE).write_text(json.dumps(log, indent=1), encoding="utf-8")
    return n, errors
