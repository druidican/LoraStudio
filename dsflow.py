# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""The Datasets flow: what each of the nine steps does to a dataset folder, and what it remembers.

  1 Folder        a folder with only pictures becomes a dataset: <N>_<name>/ with the pictures, _originals/ with
                  untouched copies, model/ and log/
  2 Usable scan   size, sharpness, compression, broken files (no AI); not usable ones can go to _reserve/
  3 Setup         LoRA types (one or more), purpose, "watermarks already removed"
  4 Sort          pass 1 quick (technical), pass 2 detailed (Ollama: subject, framing, outfit, place, quality,
                  watermarks); groups of pictures that are too alike, duplicates, detail shots (kept), not good enough
  5 Select        a suggested size, picked by quality first and then by what each picture adds; the rest -> _reserve/
  6 Resize        keep the shape, scale down to a longest side; per picture: keep or crop; originals in _originals/
  7 Tags          one caption set per caption style (IMG.txt, IMG.sentences.txt), consistency inside
  8 Review        the folder, the pictures and their captions, and checks
  9 Launch script one per LoRA type (sdgui's launch script page)

State lives in <dataset>/.lsflow/: dataset.json (setup and step times), scan.json, sort.json, select.json,
resize.json. Nothing in there is needed by the trainers.
"""
from __future__ import annotations
__version__ = "1.251026g"  # LoRA Studio version, see version.py

import json
import os
import re
import shutil
import sys
import threading
import time
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

import datacheck as DC

FLOW_DIR = ".lsflow"
ORIGINALS = "_originals"
RESERVE = "_reserve"
STOP = threading.Event()
SAME_LIMIT = 3
STEPS = [(1, "folder", "Folder"), (2, "scan", "Usable scan"), (3, "setup", "Setup"), (4, "sort", "Sort"),
         (5, "select", "Select"), (6, "resize", "Resize"), (7, "tags", "Tags"), (8, "review", "Review"),
         (9, "script", "Launch script")]
PURPOSES = [("Character / person", "character"), ("Art style", "style"), ("Object / item", "object"),
            ("Action / pose", "action"), ("Concept / effect", "concept")]
PURPOSE_LABEL = {"character": "Character", "style": "Art style", "object": "Object", "action": "Action / pose",
                 "concept": "Concept / effect"}
CONCEPT_KINDS = ("object", "action", "concept")
CONCEPT_ASK = {"object": "The object, in a few words (e.g. katana, sword)",
               "action": "The action or pose, in a few words (e.g. kneeling, one knee down)",
               "concept": "The concept or effect, in a few words (e.g. glowing aura, wet clothes)"}


def sg():
    return sys.modules["sdgui"]


def root(ds) -> Path:
    return sg().lora_path(ds)


def _flow(ds) -> Path:
    d = root(ds) / FLOW_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load(ds, name, default=None):
    try:
        return json.loads((root(ds) / FLOW_DIR / name).read_text(encoding="utf-8"))
    except Exception:                                    # noqa: BLE001
        return default


def _save(ds, name, data):
    p = _flow(ds) / name
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    tmp.replace(p)


def meta(ds) -> dict:
    m = _load(ds, "dataset.json", None)
    if m is None:                                        # a dataset from before: its launch scripts tell the types
        m = {}
        try:
            ts = [sg().script_type(str(p)) for p in sorted(root(ds).glob("*.sh"))] if ds else []
            m["types"] = list(dict.fromkeys(t for t in ts if t))
        except Exception:                                # noqa: BLE001
            m["types"] = []
    m.setdefault("types", [])
    m.setdefault("purpose", "character")
    m.setdefault("wm_removed", False)
    m.setdefault("trigger", ds or "")
    m.setdefault("steps", {})
    return m


def meta_save(ds, m: dict):
    _save(ds, "dataset.json", m)


def mark_step(ds, key, info=None):
    m = meta(ds)
    m["steps"][key] = {"at": time.strftime("%Y-%m-%d %H:%M"), **(info or {})}
    meta_save(ds, m)


def images(ds) -> list[Path]:
    """The training pictures: the dataset's own repeat folder <N>_<dataset name> only."""
    return sg().oc_images(ds) if ds else []


# Pictures outside the repeat folder (_reserve/, _extra/, other repeat folders, other folders, the dataset folder
# itself) are left out of everything (counts, checks, Improvements) unless the usable scan (step 2) is told to look
# at them: then steps 2, 4 and 5 treat them as candidates, the ones kept in step 5 join the repeat folder, and the
# setting switches itself off again when a launch script is saved.
NOT_CANDIDATES = {ORIGINALS, FLOW_DIR, "model", "models", "log", "logs", "loramanager", "_old_runs", "output",
                  "outputs", "sample", "samples", "compare"}


def include_extra(ds) -> bool:
    return bool(ds) and bool(meta(ds).get("include_extra"))


def set_include_extra(ds, on: bool):
    m = meta(ds)
    if bool(m.get("include_extra")) != bool(on):
        m["include_extra"] = bool(on)
        meta_save(ds, m)


def extra_pictures(ds) -> list[Path]:
    """Pictures of the dataset folder that are not in its repeat folder (originals and set-aside ones excluded)."""
    if not ds:
        return []
    r = root(ds)
    main = {str(p) for p in images(ds)}
    skip = {s.lower() for s in NOT_CANDIDATES | {DC.SET_ASIDE}}
    out = []
    for dirpath, dirnames, filenames in os.walk(r):
        rel = Path(dirpath).relative_to(r)
        dirnames[:] = sorted(d for d in dirnames if d.lower() not in skip and not d.startswith("."))
        for f in sorted(filenames):
            p = Path(dirpath) / f
            if p.suffix.lower() in DC.IMG_EXT and str(p) not in main and not (rel.parts and rel.parts[0].lower() in skip):
                out.append(p)
    return out


def candidates(ds) -> list[Path]:
    """The pictures steps 2, 4 and 5 look at: the repeat folder, plus the others when step 2 says so."""
    imgs = images(ds)
    return imgs + extra_pictures(ds) if include_extra(ds) else imgs


def join_repeat_folder(ds, paths) -> tuple[int, list]:
    """Move pictures from outside the repeat folder into it (with their captions); a name that is taken gets _2, _3."""
    rows = [x for x in sg().dataset_stats(ds)["rows"] if not x.get("virtual")]
    if not rows:
        return 0, ["there is no repeat folder yet"]
    dest = rows[0]["folder"]
    moved, errors = 0, []
    for p in paths:
        p = Path(p)
        if not p.is_file() or p.parent == dest:
            continue
        stem, n = p.stem, 1
        while (dest / f"{stem}{p.suffix}").exists():
            n += 1
            stem = f"{p.stem}_{n}"
        try:
            for f in [p] + DC.caption_of(p):
                tail = f.name[len(p.stem):]                 # .png, .txt, .sentences.txt
                shutil.move(str(f), str(dest / (stem + tail)))
            moved += 1
        except OSError as e:
            errors.append(f"{p.name}: {e}")
    return moved, errors


def script_saved(ds) -> str:
    """A launch script was saved: other repeat folders go to _extra/ (sd-scripts would train every <N>_name folder
    in the dataset folder), and the step-2 setting to look outside the repeat folder switches off. Returns a note."""
    if not ds:
        return ""
    S = sg()
    notes = []
    st = S.dataset_stats(ds)
    r = root(ds)
    for f, _ in st.get("others") or []:
        if f.parent != r:
            continue
        dest = r / S.EXTRA_DIR / f.name
        if dest.exists():
            notes.append(f"`{f.name}/` is not part of this dataset but `{S.EXTRA_DIR}/{f.name}` exists: move it yourself")
            continue
        try:
            dest.parent.mkdir(exist_ok=True)
            f.rename(dest)
            notes.append(f"`{f.name}/` moved to `{S.EXTRA_DIR}/` (not part of this dataset, so not trained)")
        except OSError as e:
            notes.append(f"could not move `{f.name}/`: {e}")
    if include_extra(ds):
        set_include_extra(ds, False)
        notes.append("pictures outside the repeat folder are left out again (step 2)")
    return "; ".join(notes)


def _sig(ps) -> str:
    return str(len(ps)) + ":" + str(sum(int(p.stat().st_mtime) % 100000 for p in ps if p.exists()))


def type_labels() -> dict:
    out = {}
    for l, v in sg().type_choices():
        out[v] = re.sub(r"\s*[\(\[].*?[\)\]]", "", l).strip()
    return out


def min_side(types) -> int:
    return max([sg()._dc_min_side(t) for t in (types or [])] + [0])


def tools_ok() -> bool:
    """The picture tools (quality, look-alikes, person boxes, watermark removal, upscaling) can be used now."""
    try:
        import pictools
        return pictools.ready() and not sg().RUN.running()
    except Exception:                                    # noqa: BLE001
        return False


UPSCALE_FLOOR = 0.4          # with the picture tools, a short side down to 40 % of the type's size is upscaled in step 6


def soften_small(issues, w, h, side, tools):
    """With the picture tools, 'too small' between 40 % and 60 % of the type's size is a warning: step 6 upscales it."""
    if not (tools and side and w and h and min(w, h) >= side * UPSCALE_FLOOR):
        return issues
    return [(DC.WARN, f"small: {w}x{h}, upscaled with Real-ESRGAN in step 6") if s == DC.BAD and t.startswith("too small")
            else (s, t) for s, t in issues]


def default_max_side(types) -> int:
    ms = min_side(types)
    base = 512 if 0 < ms < 600 else 1024
    return int(base * 1.5)


# --------------------------------------------------------------------------- 1 Folder
def datasets_dir() -> Path | None:
    d = sg().images_dir()
    return d if d and d.is_dir() else None


def candidate_folders() -> list[dict]:
    """Folders in the dataset folder: {'name', 'kind': 'pictures' | 'dataset' | 'empty', 'n'}."""
    d = datasets_dir()
    out = []
    if not d:
        return out
    for f in sorted(d.iterdir(), key=lambda p: p.name.lower()):
        if not f.is_dir() or f.name.startswith((".", "_")):
            continue
        reps = [x for x in f.iterdir() if x.is_dir() and sg().REPEAT_RE.match(x.name)] if f.is_dir() else []
        if reps:
            n = sum(len(sg().list_images(r)) for r in reps)
            out.append({"name": f.name, "kind": "dataset", "n": n})
        else:
            n = len([x for x in f.iterdir() if x.is_file() and x.suffix.lower() in DC.IMG_EXT])
            out.append({"name": f.name, "kind": "pictures" if n else "empty", "n": n})
    return out


def safe_name(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", (name or "").strip()).strip("._")


def _pictures_in(folder: Path, recursive: bool) -> list[Path]:
    it = folder.rglob("*") if recursive else folder.iterdir()
    skip = {RESERVE, ORIGINALS, FLOW_DIR, DC.SET_ASIDE, "model", "log", "LoraManager"}
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in DC.IMG_EXT
                  and not (set(p.relative_to(folder).parts[:-1]) & skip))


def create(source: str, name: str, repeats: int, outside: str = "") -> tuple[str, str]:
    """Make a dataset. source = a folder in the dataset folder with only pictures (moved in place), or outside =
    any folder (its pictures are copied). Returns (dataset name or '', message)."""
    d = datasets_dir()
    if not d:
        return "", "The dataset folder is not set or does not exist (Settings › Folders)."
    name = safe_name(name)
    if not name:
        return "", "Give the dataset a name."
    repeats = max(1, int(repeats or 10))
    target = d / name
    if outside:
        src = Path(os.path.expanduser(outside.strip()))
        if not src.is_dir():
            return "", f"The folder `{outside}` does not exist."
        pics = _pictures_in(src, True)
        if not pics:
            return "", f"No pictures in `{src}`."
        if target.exists() and any(target.iterdir()):
            return "", f"`{name}` already exists in the dataset folder: pick another name."
        move = False
    else:
        src = d / (source or "")
        if not source or not src.is_dir():
            return "", "Pick a folder with pictures."
        pics = _pictures_in(src, False)
        if not pics:
            return "", f"`{source}` has no pictures."
        if src.name != name:
            if target.exists():
                return "", f"`{name}` already exists in the dataset folder: pick another name."
            src.rename(target)
            pics = [target / p.name for p in pics]
        move = True
    rep = target / f"{repeats}_{name}"
    orig = target / ORIGINALS / rep.name
    rep.mkdir(parents=True, exist_ok=True)
    orig.mkdir(parents=True, exist_ok=True)
    for sub in ("model", "log"):
        (target / sub).mkdir(exist_ok=True)
    n = 0
    for p in pics:
        dest = rep / p.name
        k = 1
        while dest.exists():
            dest = rep / f"{p.stem}_{k}{p.suffix}"
            k += 1
        if move:
            shutil.move(str(p), str(dest))
            for c in DC.caption_of(p):
                shutil.move(str(c), str(rep / (dest.stem + c.name[len(p.stem):])))
        else:
            shutil.copy2(p, dest)
            for c in DC.caption_of(p):
                shutil.copy2(c, rep / (dest.stem + c.name[len(p.stem):]))
        shutil.copy2(dest, orig / dest.name)
        n += 1
    m = meta(name)
    m["created"] = time.strftime("%Y-%m-%d %H:%M")
    m["trigger"] = name
    m["steps"]["folder"] = {"at": m["created"]}
    meta_save(name, m)
    return name, f"✔ Dataset **{name}** made: {n} pictures in `{rep.name}/`, untouched copies in `{ORIGINALS}/`."


def ensure_originals(ds, paths) -> int:
    """Copy pictures to _originals/ (same repeat folder) when no copy is there yet."""
    r = root(ds)
    n = 0
    for p in paths:
        p = Path(p)
        try:
            rel = p.relative_to(r)
        except ValueError:
            continue
        dest = r / ORIGINALS / rel
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
            n += 1
    return n


# --------------------------------------------------------------------------- watermarks
def main_person(persons, W, H):
    """The box of the main person from the picture tools: the most certain, biggest one; None when unsure."""
    ps = [p for p in persons or [] if len(p) >= 5 and p[4] >= 0.7]
    if not ps or not W or not H:
        return None
    best = max(ps, key=lambda p: p[4] * (p[2] - p[0]) * (p[3] - p[1]))
    if (best[2] - best[0]) * (best[3] - best[1]) < 0.06 * W * H:
        return None
    return [int(v) for v in best[:4]]


def wm_done(ds) -> str:
    """You are happy with the watermarks of this dataset (also when some could not be removed): the ones not decided
    stay as they are, and the dataset no longer asks about watermarks."""
    if not ds:
        return ""
    r = root(ds)
    marks = DC.reconcile(r, DC.marks_load(r))
    n = 0
    for e in marks.values():
        if e.get("marks") and not e.get("choice"):
            e["choice"] = "keep"
            n += 1
    DC.marks_save(r, marks)
    m = meta(ds)
    m["wm_done"] = True
    meta_save(ds, m)
    return ("✔ Watermarks marked as done" + (f": {n} undecided picture{'s' if n != 1 else ''} kept as they are." if n
                                              else ".") + " Change any picture in the list above at any time.")


# --------------------------------------------------------------------------- the repeat folder
def rename_repeats(ds, want: int) -> str:
    """Give the dataset's one repeat folder the repeat count `want` (10_bob -> 20_bob), with its copies in _originals/,
    _reserve/ and _set_aside/originals/, and the paths LoRA Studio remembers for it. Returns what was done ('' if
    nothing), or a line starting with '⚠'."""
    S = sg()
    r = root(ds)
    rows = [x for x in S.dataset_stats(ds)["rows"] if not x.get("virtual")]
    if len(rows) != 1 or not want or int(want) < 1:
        return ""
    f = rows[0]["folder"]
    if rows[0]["repeats"] == int(want):
        return ""
    if S.RUN.running() and S.script_home(S.RUN.script or "/") == r:
        return "⚠ A training run of this dataset is active: the repeat folder was not renamed."
    new_name = S.REPEAT_RE.sub(f"{int(want)}_", f.name, count=1)
    dest = f.with_name(new_name)
    if dest.exists():
        return f"⚠ `{new_name}` already exists: the repeat folder was not renamed."
    rel_old = str(f.relative_to(r))
    rel_new = str(dest.relative_to(r))
    f.rename(dest)
    done = [f"repeat folder `{f.name}` → `{new_name}`"]
    for side in (ORIGINALS, RESERVE, f"{DC.SET_ASIDE}/originals"):
        o = r / side / rel_old
        if o.is_dir() and not (r / side / rel_new).exists():
            o.rename(r / side / rel_new)
            done.append(f"`{side}/{f.name}` too")
    # the paths remembered for the pictures (sort, selection, resize, watermarks)
    a_old, a_new = str(f) + "/", str(dest) + "/"
    files = list((r / FLOW_DIR).glob("*.json")) + [r / DC.SET_ASIDE / DC.MARKS_FILE]
    for jf in files:
        try:
            t = jf.read_text(encoding="utf-8")
        except OSError:
            continue
        t2 = t.replace(json.dumps(a_old)[1:-1], json.dumps(a_new)[1:-1]).replace(f'"{rel_old}/', f'"{rel_new}/')
        if t2 != t:
            jf.write_text(t2, encoding="utf-8")
    return "; ".join(done)


# --------------------------------------------------------------------------- 2 Usable scan
CAT_WORDS = (("broken", "cannot be read"), ("small", "too small"), ("blurry", "blurry"))


def usable_scan(ds, types, upscale: bool = True) -> dict:
    imgs = candidates(ds)
    side = min_side(types)
    tech = DC.technical(imgs, side)
    tools = tools_ok()
    up = tools and upscale
    pics = {}
    counts = Counter()
    for p, r in tech.items():
        r["issues"] = soften_small(r["issues"], r.get("w"), r.get("h"), side, up)
        bad = [t for s, t in r["issues"] if s == DC.BAD]
        cat = "usable"
        if bad:
            cat = next((c for c, w in CAT_WORDS if any(w in b for b in bad)), "other")
        counts[cat] += 1
        pics[str(p)] = {"w": r.get("w"), "h": r.get("h"), "cat": cat,
                        "why": "; ".join(t for _, t in r["issues"]), "bad": bool(bad)}
    res = {"at": time.strftime("%Y-%m-%d %H:%M"), "min_side": side, "pics": pics, "counts": dict(counts),
           "sig": _sig(imgs)}
    _save(ds, "scan.json", res)
    mark_step(ds, "scan", {"not_usable": len(pics) - counts["usable"]})
    return res


def move_to_reserve(ds, paths) -> tuple[int, list]:
    import trim
    return trim.move_to_reserve(root(ds), [Path(p) for p in paths])


def back_from_reserve(ds) -> tuple[int, list]:
    import trim
    return trim.move_back(root(ds))


def reserve_count(ds) -> int:
    import trim
    return len(trim.reserve_list(root(ds))) if ds else 0


# --------------------------------------------------------------------------- 4 Sort
def _content_cached(p: Path, model: str, cache: dict):
    st = p.stat()
    return cache.get(f"{p}|{st.st_size}|{int(st.st_mtime)}|{model}"), f"{p}|{st.st_size}|{int(st.st_mtime)}|{model}"


def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip().lower())


def sort_run(ds, types, purpose, wm_removed, url, model, upscale: bool = True):
    """Generator: yields progress lines; the result is saved to sort.json (read it with sort_result)."""
    S = sg()
    STOP.clear()
    imgs = candidates(ds)
    if not imgs:
        yield "No pictures in this dataset's repeat folders."
        return
    side = min_side(types)
    t0 = time.time()
    yield f"Pass 1 · quick: size, sharpness and duplicates of {len(imgs)} pictures …"
    tech = DC.technical(imgs, side)
    tools = tools_ok()
    pics = {}
    for p, r in tech.items():
        r["issues"] = soften_small(r["issues"], r.get("w"), r.get("h"), side, tools and upscale)
        bad = [t for s, t in r["issues"] if s == DC.BAD]
        dup = next((t for s, t in r["issues"] if "near-duplicate" in t), "")
        pics[str(p)] = {"w": r.get("w") or 0, "h": r.get("h") or 0, "sharp": r.get("sharp") or 0,
                        "cat": "out" if bad else ("dup" if dup else "good"), "why": "; ".join(bad) or dup,
                        "warn": [t for s, t in r["issues"] if s == DC.WARN and "near-duplicate" not in t]}
    pass1_out = [p for p, e in pics.items() if e["cat"] == "out"]
    rest = [Path(p) for p, e in pics.items() if e["cat"] != "out"]
    # the picture tools (sd-scripts venv): a quality score, photo or drawing, person boxes, and look-alikes
    tools_note = ""
    if tools:
        import pictools
        A = {}
        try:
            for i in range(0, len(rest), 16):
                if STOP.is_set():
                    break
                yield f"Pass 1 · rating quality and finding look-alikes: {min(i + 16, len(rest))} of {len(rest)}"
                A.update(pictools.analyze(rest[i:i + 16]))
        except Exception as e:                           # noqa: BLE001
            tools, tools_note = False, f"picture tools: {e}"
        for p, a in A.items():
            if p in pics:
                pics[p].update(aes=a["aes"], kind=a["kind"], persons=a.get("persons") or [])
        good = {p: A[p] for p in A if p in pics and pics[p]["cat"] == "good"}
        for a, b, sim in sorted(pictools.lookalikes(good, 0.95) if tools else [], key=lambda t: -t[2]):
            ea, eb = pics[a], pics[b]
            if ea["cat"] != "good" or eb["cat"] != "good":
                continue
            worse, better = (a, b) if (ea.get("aes", 0), ea["sharp"]) < (eb.get("aes", 0), eb["sharp"]) else (b, a)
            pics[worse]["cat"] = "dup"
            pics[worse]["why"] = f"shows the same as {Path(better).name} ({sim:.0%} alike); that one rates higher"
    url = (url or "").strip() or S.oc_settings()[0]
    model = model or S.oc_settings()[1]
    import features as FT
    ai = FT.have("ollama") and bool(model) and S.CAP is not None and not S.RUN.running()
    note = "" if ai else ("Ollama is not installed: quick pass only" if not FT.have("ollama") else
                          "no Ollama model chosen (Settings › AI helpers)" if not model else
                          "a training run is active" if S.RUN.running() else "Ollama is not available")
    root_ = root(ds)
    marks = DC.reconcile(root_, DC.marks_load(root_)) if ai and not wm_removed else {}
    crop_min = side or 512
    done = 0
    if ai:
        cache = S._dc_cache_load()
        for i, p in enumerate(rest, 1):
            if STOP.is_set():
                note = "stopped"
                break
            d, key = _content_cached(p, model, cache)
            if d is None:
                eta = f", about {S.fmt((time.time() - t0) / max(done, 1) * (len(rest) - i + 1))} left" if done else ""
                yield f"Pass 2 · detailed (Ollama): {i} of {len(rest)}{eta}"
                try:
                    d = DC.content(url, model, p)
                except Exception as e:                   # noqa: BLE001
                    S._dc_cache_save(cache)
                    note = f"Ollama stopped answering: {e}"
                    break
                cache[key] = d
                done += 1
                if done % 5 == 0:
                    S._dc_cache_save(cache)
            e = pics[str(p)]
            e["framing"] = _norm(d.get("framing")) or "other"
            e["outfit"], e["place"] = _norm(d.get("outfit")), _norm(d.get("place"))
            e["part"] = _norm(d.get("part_only"))
            if e["part"] in ("none", "no", "n/a", "''"):
                e["part"] = ""
            issues = DC.content_issues(d, purpose)
            if e["part"] and purpose != "style":         # a detail shot is not "no clear subject"
                issues = [x for x in issues if not any(w in x[1] for w in ("no person", "no clear main subject",
                                                                          "face not visible", "face partly"))]
                if e["cat"] == "good":
                    e["cat"], e["why"] = "detail", f"only the {e['part']}: teaches a detail the full pictures show too small"
            bad = [t for s, t in issues if s == DC.BAD and "watermark" not in t]
            if bad and e["cat"] in ("good", "detail"):
                e["cat"], e["why"] = "poor", "; ".join(bad)
            e["warn"] += [t for s, t in issues if s == DC.WARN and "watermark" not in t]
            if DC._true(d.get("text_or_watermark")) and not wm_removed:
                e["wm"] = True
                rp = DC.rel(root_, p)
                ent = marks.get(rp)
                st = p.stat()
                sig = f"{st.st_size}|{int(st.st_mtime)}"
                # a picture you already decided on is left alone (its file changed because of your choice)
                if not (ent and ent.get("choice")) and (not ent or ent.get("sig") != sig):
                    yield f"Pass 2 · where are the watermarks in {p.name}?"
                    try:
                        f = DC.find_marks(url, model, p)
                        person = main_person(e.get("persons"), f["W"], f["H"])
                        if person:                      # the person finder is more exact than the vision model
                            f["subject"] = person
                        marks[rp] = {**f, "sig": sig, "choice": "",
                                     "options": DC.crop_options(f["W"], f["H"], f["marks"], f["subject"], crop_min)}
                    except Exception:                   # noqa: BLE001
                        pass
        S._dc_cache_save(cache)
        if marks:
            DC.marks_save(root_, marks)
    # quality: sharpness rank in this dataset, size against what the type needs, minus warnings
    order = sorted(pics.values(), key=lambda e: e["sharp"])
    for i, e in enumerate(order):
        sharp = i / max(1, len(order) - 1)
        size_ = min(1.0, min(e["w"], e["h"]) / max(side or 1024, 1))
        if "aes" in e:                                   # the quality score of the picture tools leads
            aes = max(0.0, min(1.0, (e["aes"] - 3.0) / 4.0))
            e["q"] = round(0.5 * aes + 0.3 * sharp + 0.2 * size_ - 0.08 * len(e["warn"]), 3)
        else:
            e["q"] = round(0.6 * sharp + 0.4 * size_ - 0.08 * len(e["warn"]), 3)
    # too alike: the same framing, outfit and place more than a few times (the best ones are kept)
    combos: dict = {}
    for p, e in pics.items():
        if e["cat"] in ("good", "detail") and e.get("framing"):
            combos.setdefault((e["framing"], e["outfit"], e["place"]), []).append(p)
    groups = []
    for c, ps in combos.items():
        if len(ps) > SAME_LIMIT:
            ps.sort(key=lambda x: -pics[x]["q"])
            for x in ps[2:]:
                if pics[x]["cat"] == "good":
                    pics[x]["cat"] = "alike"
                    pics[x]["why"] = f"one of {len(ps)} alike: {' · '.join(v for v in c if v)}"
            groups.append({"label": " · ".join(v for v in c if v), "n": len(ps)})
    groups.sort(key=lambda g: -g["n"])
    usable = [e for e in pics.values() if e["cat"] in ("good", "detail", "alike")]
    fr = Counter(e.get("framing", "") for e in usable if e.get("framing"))
    outfits = len({e["outfit"] for e in usable if e.get("outfit")})
    places = len({e["place"] for e in usable if e.get("place")})
    good_n = sum(1 for e in usable if e["cat"] in ("good", "detail"))
    need = {"character": 15, "style": 20, "concept": 10}.get(DC.base_purpose(purpose), 15)
    if good_n >= need and (purpose != "character" or fr.get("full body", 0) or not ai):
        baseline = "good"
    elif good_n >= need:
        baseline = "thin"
    else:
        baseline = "weak"
    notes = DC.balance({Path(p): {"content": {"framing": e.get("framing"), "outfit": e.get("outfit"),
                                              "place": e.get("place"), "people": 1}}
                        for p, e in pics.items() if e.get("framing") and e["cat"] != "out"}, purpose) if ai else []
    res = {"at": time.strftime("%Y-%m-%d %H:%M"), "purpose": purpose, "types": list(types or []), "ai": ai,
           "note": note, "pics": pics, "groups": groups, "framing": dict(fr), "outfits": outfits, "places": places,
           "baseline": baseline, "notes": notes, "pass1_out": len(pass1_out), "sig": _sig(candidates(ds)),
           "seconds": round(time.time() - t0, 1), "good": good_n, "tools": tools, "tools_note": tools_note}
    _save(ds, "sort.json", res)
    mark_step(ds, "sort", {"baseline": baseline})
    yield "done"


def sort_result(ds) -> dict | None:
    return _load(ds, "sort.json")


# --------------------------------------------------------------------------- 5 Select
QUOTA = {"close-up": 0.35, "upper body": 0.35, "full body": 0.2}


def suggest_size(ds, res=None) -> tuple[int, str, str]:
    """(number, range text, why)."""
    res = res or sort_result(ds) or {}
    m = meta(ds)
    purpose = m["purpose"]
    cands = [e for e in (res.get("pics") or {}).values() if e["cat"] in ("good", "detail", "alike")]
    o, pl = res.get("outfits", 0), res.get("places", 0)
    if purpose == "style":
        n, rng = max(40, min(100, 40 + 2 * pl)), "40–100"
    elif purpose in CONCEPT_KINDS:
        n, rng = max(20, min(40, 15 + 2 * o + pl)), "20–40"
    else:
        n, rng = max(25, min(50, 20 + 2 * o + pl)), "30–50"
    labels = type_labels()
    tname = " + ".join(labels.get(t, t) for t in m["types"]) or "this"
    why = ((f"{tname}, art style, {pl} different places and scenes: a style needs variety (people, objects, "
            f"landscapes, light) more than a big number; about {n} pictures.") if purpose == "style" and (o or pl) else
           (f"{tname}, {PURPOSE_LABEL.get(purpose, purpose).lower()}, {o} outfits and {pl} places: more than about "
            f"{n} pictures mostly adds repeats of the same look.") if o or pl else
           f"Without the detailed sort there is no outfit or place information; {n} is the usual size.")
    if cands and len(cands) < n:
        n = len(cands)
        why += f" There are only {len(cands)} usable pictures, so all of them."
    return n, rng, why


def rank(ds, target: int, mode: str | None = None) -> list[dict]:
    """Every picture with keep True/False and a reason. mode "safe": quality first, then what each picture adds;
    "varied": what each picture adds counts as much as quality (more close-ups, angles, outfits, places, details)."""
    mode = mode or meta(ds).get("select_mode") or "safe"
    if mode == "varied":
        return _rank_varied(ds, target)
    res = sort_result(ds) or {}
    pics = res.get("pics") or {}
    have = {str(p) for p in candidates(ds)}
    rows = []
    for p in sorted(have):
        e = pics.get(p)
        if e is None:
            rows.append({"path": p, "cat": "new", "q": 0.5, "framing": "", "outfit": "", "place": ""})
        else:
            rows.append({"path": p, **e})
    cands = [r for r in rows if r["cat"] in ("good", "detail", "alike", "new")]
    cands.sort(key=lambda r: (r["cat"] == "alike", -r["q"]))
    purpose = meta(ds)["purpose"]
    keep, fr_n, combo_n = [], Counter(), Counter()
    quota = {k: max(1, round(v * target)) for k, v in QUOTA.items()} if purpose == "character" else {}
    detail_max = max(1, round(0.12 * target))
    for r in cands:
        if len(keep) >= target:
            break
        f = r.get("framing") or "other"
        c = (f, r.get("outfit"), r.get("place"))
        if r["cat"] == "detail" and sum(1 for k in keep if k["cat"] == "detail") >= detail_max:
            continue
        if f in quota and fr_n[f] >= quota[f] and f != "full body":
            continue
        if r.get("framing") and combo_n[c] >= SAME_LIMIT - 1:
            continue
        keep.append(r)
        fr_n[f] += 1
        combo_n[c] += 1
    for r in cands:                                      # fill up by quality when the quotas left room
        if len(keep) >= target:
            break
        if r not in keep:
            keep.append(r)
    kept = {id(r) for r in keep}
    for r in rows:
        r["keep"] = id(r) in kept
        f = r.get("framing") or ""
        if r["keep"]:
            if r["cat"] == "detail":
                r["reason"] = r.get("why") or "detail shot"
            elif r["cat"] == "new":
                r["reason"] = "added after the sort"
            else:
                bits = [f] if f else []
                if r.get("outfit"):
                    bits.append(r["outfit"])
                r["reason"] = "kept: " + (" · ".join(bits) if bits else f"quality {r['q']:.2f}")
        else:
            if r["cat"] in ("out", "dup", "poor"):
                r["reason"] = r.get("why") or r["cat"]
            elif r["cat"] == "alike":
                r["reason"] = r.get("why") or "too alike"
            else:
                r["reason"] = "lower quality, or adds the same framing, outfit and place again"
    return rows


def mix_text(rows) -> str:
    k = [r for r in rows if r["keep"]]
    fr = Counter((r.get("framing") or "unknown") for r in k if r["cat"] != "detail")
    det = sum(1 for r in k if r["cat"] == "detail")
    parts = [f"{n} {f}" for f, n in fr.most_common()]
    if det:
        parts.append(f"{det} detail shot{'s' if det != 1 else ''} (about {100 * det // max(1, len(k))} %)")
    return " · ".join(parts)


def apply_select(ds, keep_paths) -> str:
    keep = {str(Path(p)) for p in keep_paths}
    drop = [p for p in images(ds) if str(p) not in keep]
    n, err = move_to_reserve(ds, drop) if drop else (0, [])
    joined = [p for p in extra_pictures(ds) if str(p) in keep] if include_extra(ds) else []
    j, jerr = join_repeat_folder(ds, joined) if joined else (0, [])
    err += jerr
    kept = [str(p) for p in images(ds)]
    mark_step(ds, "select", {"kept": len(kept), "reserve": reserve_count(ds)})
    _save(ds, "select.json", {"at": time.strftime("%Y-%m-%d %H:%M"), "kept": sorted(kept)})
    return (f"✔ {len(kept)} pictures in the repeat folder; {n} moved to `{RESERVE}/` (not deleted, with their captions)"
            + (f"; {j} from outside it joined" if j else "") + "."
            + (f" Could not move: {'; '.join(err[:3])}" if err else ""))


# --------------------------------------------------------------------------- 6 Resize
def resize_overrides(ds) -> dict:
    return (_load(ds, "resize.json", {}) or {}).get("over", {})


def resize_save_over(ds, over: dict, max_side: int):
    _save(ds, "resize.json", {"over": over, "max_side": int(max_side)})


def resize_plan(ds, max_side: int, over: dict | None = None, upscale: bool = True) -> list[dict]:
    over = over if over is not None else resize_overrides(ds)
    rows = []
    r = root(ds)
    side = min_side(meta(ds)["types"])
    tools = tools_ok() and upscale
    for p in images(ds):
        try:
            with Image.open(p) as im:
                w, h = ImageOps.exif_transpose(im).size
        except Exception:                                # noqa: BLE001
            continue
        rel = str(p.relative_to(r))
        o = over.get(rel) or {}
        sugg = suggest_mode(w, h, max_side, side, tools)
        mode = o.get("mode") or sugg
        if mode == "upscale" and not tools:
            mode = "none"
        cw, ch = w, h
        if mode == "crop" and o.get("box"):
            l, t, rr, b = o["box"]
            cw, ch = max(1, int((rr - l) * w)), max(1, int((b - t) * h))
        if mode == "upscale":
            f = min(max(1.0, side / max(1, min(cw, ch))), max_side / max(cw, ch), 4.0)
        else:
            f = min(1.0, max_side / max(cw, ch)) if mode in ("scale", "crop") else 1.0
        rows.append({"path": str(p), "rel": rel, "w": w, "h": h, "mode": mode, "by_you": rel in over, "sugg": sugg,
                     "new": (round(cw * f), round(ch * f)), "box": o.get("box")})
    return rows


def suggest_mode(w, h, max_side, side, tools) -> str:
    if max(w, h) > max_side:
        return "scale"
    if tools and side and side * UPSCALE_FLOOR <= min(w, h) < side:
        return "upscale"
    return "none"


def person_box(path: str, aspect: str = "free", run: bool = False) -> list[float] | None:
    """A crop around the person(s) the picture tools found; run=False only uses what they already know."""
    try:
        import pictools
        p = Path(path)
        if run:
            if not tools_ok():
                return None
            e = pictools.analyze([p]).get(str(p))
        else:
            e = pictools.cached(p)
        if not e:
            return None
        with Image.open(p) as im:
            w, h = ImageOps.exif_transpose(im).size
        a = {"1:1": 1.0, "3:4": 0.75, "4:3": 4 / 3}.get(aspect)
        return pictools.smart_box(e, w, h, a)
    except Exception:                                    # noqa: BLE001
        return None


def crop_preview(path: str, box, side: int = 640) -> Image.Image | None:
    try:
        im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    except Exception:                                    # noqa: BLE001
        return None
    im.thumbnail((side, side))
    if box:
        w, h = im.size
        l, t, r, b = box
        shade = Image.new("RGBA", im.size, (0, 0, 0, 120))
        d = ImageDraw.Draw(shade)
        d.rectangle([int(l * w), int(t * h), int(r * w), int(b * h)], fill=(0, 0, 0, 0))
        im = Image.alpha_composite(im.convert("RGBA"), shade).convert("RGB")
        ImageDraw.Draw(im).rectangle([int(l * w), int(t * h), int(r * w) - 1, int(b * h) - 1], outline=(232, 163, 61),
                                     width=3)
    return im


def fit_box(w, h, l, t, r, b, aspect: str):
    """Make a box (fractions) follow an aspect ratio ('free', '1:1', '3:4', '4:3'), shrinking around its centre."""
    l, t, r, b = [max(0.0, min(1.0, float(x))) for x in (l, t, r, b)]
    if r - l < 0.05:
        r = min(1.0, l + 0.05)
    if b - t < 0.05:
        b = min(1.0, t + 0.05)
    if aspect in ("1:1", "3:4", "4:3"):
        aw, ah = (int(x) for x in aspect.split(":"))
        bw, bh = (r - l) * w, (b - t) * h
        want = aw / ah
        if bw / bh > want:
            nbw = bh * want
            cx = (l + r) / 2 * w
            l, r = (cx - nbw / 2) / w, (cx + nbw / 2) / w
        else:
            nbh = bw / want
            cy = (t + b) / 2 * h
            t, b = (cy - nbh / 2) / h, (cy + nbh / 2) / h
    return [round(max(0.0, l), 4), round(max(0.0, t), 4), round(min(1.0, r), 4), round(min(1.0, b), 4)]


def apply_resize(ds, max_side: int, upscale: bool = True) -> str:
    g = apply_resize_steps(ds, max_side, upscale)
    while True:
        try:
            next(g)
        except StopIteration as e:
            return e.value


def apply_resize_steps(ds, max_side: int, upscale: bool = True):
    """Generator: yields a progress line per picture; its return value is the message (see apply_resize)."""
    over = resize_overrides(ds)
    rows = resize_plan(ds, max_side, over, upscale)
    todo = [x for x in rows if x["mode"] in ("scale", "crop", "upscale")
            and (x["new"] != (x["w"], x["h"]) or x["mode"] == "crop")]
    ensure_originals(ds, [x["path"] for x in todo])
    n_s = n_c = n_u = 0
    errs = []
    t0 = time.time()
    for i, x in enumerate(todo):
        p = Path(x["path"])
        left = ""
        if i:
            left = f" · about {int((time.time() - t0) / i * (len(todo) - i))} s left"
        yield (f"⏳ {'Upscaling' if x['mode'] == 'upscale' else 'Cropping' if x['mode'] == 'crop' else 'Scaling'} "
               f"{p.name} ({i + 1} of {len(todo)}){left}")
        if x["mode"] == "upscale":
            try:
                import pictools
                pictools.upscale(p, p, min(x["new"]), "photo")
                n_u += 1
            except Exception as e:                       # noqa: BLE001
                errs.append(f"{p.name}: {e}")
            continue
        try:
            with Image.open(p) as im0:
                info = im0.info
                fmt = im0.format
                im = ImageOps.exif_transpose(im0)
                im.load()
            if x["mode"] == "crop" and x.get("box"):
                l, t, r, b = x["box"]
                w, h = im.size
                im = im.crop((int(l * w), int(t * h), int(r * w), int(b * h)))
                n_c += 1
            if max(im.size) > max_side:
                f = max_side / max(im.size)
                im = im.resize((max(1, round(im.size[0] * f)), max(1, round(im.size[1] * f))), Image.LANCZOS)
                n_s += 1 if x["mode"] == "scale" else 0
            kw = {}
            if fmt == "JPEG":
                kw = {"quality": 95, "subsampling": 0}
                if im.mode not in ("RGB", "L"):
                    im = im.convert("RGB")
            elif fmt == "PNG" and info.get("icc_profile"):
                kw = {"icc_profile": info["icc_profile"]}
            im.save(p, format=fmt or None, **kw)
        except Exception as e:                           # noqa: BLE001
            errs.append(f"{p.name}: {e}")
    over = {k: v for k, v in over.items() if v.get("mode") == "keep"}  # done crops are part of the picture now
    resize_save_over(ds, over, max_side)
    mark_step(ds, "resize", {"scaled": n_s, "cropped": n_c, "upscaled": n_u, "max_side": int(max_side)})
    return (f"✔ {n_s} picture{'s' if n_s != 1 else ''} scaled down, {n_c} cropped"
            + (f", {n_u} upscaled" if n_u else "") + ". The untouched pictures are in "
            f"`{ORIGINALS}/`." + (f" Could not change: {'; '.join(errs[:3])}" if errs else ""))


def restore_originals(ds) -> str:
    r = root(ds)
    n = 0
    for p in images(ds):
        o = r / ORIGINALS / p.relative_to(r)
        if o.is_file():
            shutil.copy2(o, p)
            n += 1
    m = meta(ds)
    m["steps"].pop("resize", None)
    meta_save(ds, m)
    return f"✔ {n} pictures put back from `{ORIGINALS}/`."


# --------------------------------------------------------------------------- 7 Tags
def style_of(tkey) -> str:
    S = sg()
    fam = S._oc_family(tkey)
    return S.OC_FAMILY.get(fam, ("sentences",))[0]


def caption_sets(ds) -> list[dict]:
    """[{'style', 'ext', 'types': [labels], 'tkeys'}]: the first LoRA type's style is IMG.txt."""
    m = meta(ds)
    labels = type_labels()
    out = []
    for t in m["types"] or []:
        st = style_of(t)
        e = next((x for x in out if x["style"] == st), None)
        if e is None:
            e = {"style": st, "ext": ".txt" if not out else f".{st}.txt", "types": [], "tkeys": []}
            out.append(e)
        e["types"].append(labels.get(t, t))
        e["tkeys"].append(t)
    if not out:
        out.append({"style": "sentences", "ext": ".txt", "types": [], "tkeys": []})
    return out


def set_file(img: Path, ext: str) -> Path:
    if ext == ".txt":
        return sg().caption_file(img)
    return img.with_name(img.stem + ext)


def read_set(img: Path, ext: str):
    f = set_file(img, ext)
    if not f.is_file():
        return None
    return f.read_text(encoding="utf-8", errors="ignore").strip() or None


def set_count(ds, ext) -> tuple[int, int]:
    imgs = images(ds)
    return sum(1 for p in imgs if read_set(p, ext)), len(imgs)


def write_set(ds, url, model, trigger, leave_out, style, length, detail, ext, only_missing=False):
    """Generator like sdgui.oc_run, for any caption set (IMG.sentences.txt ...)."""
    S = sg()
    bad = S._oc_check(ds, url, model, trigger)
    if bad:
        yield bad
        return
    url = (url or "").strip() or S.CAP.DEFAULT_URL
    imgs = images(ds)
    if only_missing:
        imgs = [p for p in imgs if not read_set(p, ext)]
    if not imgs:
        yield "Every picture already has a caption in this set."
        return
    S.record_change(f"captions {ext} of {ds}", {set_file(p, ext): (set_file(p, ext).read_text(encoding="utf-8", errors="ignore")
                                                                 if set_file(p, ext).is_file() else None) for p in imgs})
    STOP.clear()
    done, failed, t0 = 0, [], time.time()
    for i, p in enumerate(imgs, 1):
        if STOP.is_set():
            break
        eta = f", about {S.fmt((time.time() - t0) / max(done, 1) * (len(imgs) - done))} left" if done else ""
        yield f"⏳ {style} {i}/{len(imgs)}: {p.name}{eta}"
        try:
            r = S.CAP.caption(url, model, p, (trigger or ds).strip(), leave_out or "", style=style,
                              length=int(length or 3), detail=int(detail or 3))
        except S.CAP.OllamaError as e:
            failed.append(f"{p.name}: {e}")
            if "not reachable" in str(e):
                break
            continue
        set_file(p, ext).write_text(r["caption"] + "\n", encoding="utf-8")
        done += 1
    msg = f"{'■ Stopped' if STOP.is_set() else '✔ Done'}: {done} of {len(imgs)} captions written to `IMG{ext}`."
    if failed:
        msg += f" {len(failed)} failed: " + "; ".join(failed[:3])
    yield msg


def mirror_kept(ds, ext, url, model, trigger) -> str:
    """Take the traits the consistency check took out of IMG.txt out of another caption set too."""
    import consistency as CC
    base = CC.load_result(ds) or {}
    kept = set(base.get("kept") or [])
    if not kept:
        return "Nothing was taken out of the first set, so the other set stays as it is."
    imgs = [p for p in images(ds) if read_set(p, ext)]
    if not imgs:
        return "The other set has no captions yet."
    res = CC.scan(f"{ds}#{ext}", imgs, lambda p: read_set(p, ext), url, model, trigger or ds)
    choices = {g["id"]: {"mode": "lora"} for g in res["groups"] if g["trait"] in kept}
    if not choices:
        return "The other set does not mention those traits."
    backup, changed, _, skipped = CC.apply(res, choices, lambda p: read_set(p, ext), lambda p: set_file(p, ext),
                                           url, model, trigger or ds)
    if backup:
        sg().record_change(f"consistency of {ds} {ext}", backup)
    return (f"✔ Also taken out of `IMG{ext}`: {', '.join(sorted(kept))} ({len(changed)} captions changed)."
            + (f" {len(skipped)} sentence(s) left as they were." if skipped else ""))


# --------------------------------------------------------------------------- 8 Review
def review(ds) -> dict:
    r = root(ds)
    S = sg()
    m = meta(ds)
    sets = caption_sets(ds)
    imgs = images(ds)
    trig = (m.get("trigger") or ds).strip().lower()
    checks, problems = [], {}
    for st in sets:
        miss = [p for p in imgs if not read_set(p, st["ext"])]
        if miss:
            checks.append(("bad", f"{len(miss)} picture{'s' if len(miss) != 1 else ''} without a caption in IMG{st['ext']}",
                           "", ""))
            for p in miss:
                problems.setdefault(str(p), []).append(f"no caption ({st['ext']})")
        else:
            checks.append(("ok", f"{len(imgs)} pictures, each with a caption in IMG{st['ext']}", "", ""))
        notrig = [p for p in imgs if read_set(p, st["ext"]) and not read_set(p, st["ext"]).lower().startswith(trig)]
        if notrig:
            checks.append(("warn", f"The trigger word '{trig}' is not first in {len(notrig)} caption(s) of IMG{st['ext']}", "", ""))
            for p in notrig:
                problems.setdefault(str(p), []).append("trigger word not first")
        elif imgs:
            checks.append(("ok", f"Trigger word first in all captions of IMG{st['ext']}", "", ""))
        longs = []
        for p in imgs:
            c = read_set(p, st["ext"]) or ""
            n = len(c.split(",")) if st["style"] == "tags" else len(c.split())
            if c and n > (40 if st["style"] == "tags" else 160):
                longs.append(p)
        if longs:
            checks.append(("warn", f"{len(longs)} very long caption(s) in IMG{st['ext']}", "", ""))
            for p in longs:
                problems.setdefault(str(p), []).append("very long caption")
    # caption files without a picture, and pictures outside the repeat folders
    stats = S.dataset_stats(ds)
    exts = [s["ext"] for s in sets]
    for row in stats["rows"]:
        folder = row["folder"]
        stems = {p.stem for p in S.list_images(folder)}
        for f in sorted(folder.glob("*.txt")) + sorted(folder.glob("*.caption")):
            stem = f.name.split(".")[0] if f.name.count(".") > 1 else f.stem
            if stem not in stems:
                checks.append(("warn", f"{f.name} has no picture", "delete", str(f)))
    outside = extra_pictures(ds)
    if outside:
        checks.append(("ok", f"{len(outside)} picture{'s' if len(outside) != 1 else ''} outside the repeat folder "
                              "(_reserve/ and other folders) are not part of the dataset and not trained", "", ""))
    tree = [(f"{ds}/", "")]
    for row in stats["rows"]:
        caps = sum(1 for p in S.list_images(row["folder"]) if read_set(p, exts[0]))
        tree.append((f"  {row['folder'].name}/", f"{row['n']} pictures + {caps} captions"))
    for sub in (ORIGINALS, RESERVE, DC.SET_ASIDE):
        d = r / sub
        if d.is_dir():
            n = sum(1 for p in d.rglob("*") if p.is_file() and p.suffix.lower() in DC.IMG_EXT)
            tree.append((f"  {sub}/", f"{n} pictures"))
    for sub in ("model", "log"):
        if (r / sub).is_dir():
            tree.append((f"  {sub}/", ""))
    for f in sorted(r.glob("*.sh")):
        tree.append((f"  {f.name}", "launch script"))
    return {"checks": checks, "problems": problems, "tree": tree, "sets": sets, "images": [str(p) for p in imgs]}


def fix(ds, action, path) -> str:
    p = Path(path)
    S = sg()
    if action == "delete":
        if S.HK is not None:
            n, err = S.HK.trash([p])
            return f"✔ {p.name} moved to the Trash." if n else f"Could not move {p.name}: {'; '.join(err)}"
        return "housekeeping.py is missing."
    if action == "movein":
        rows = S.dataset_stats(ds)["rows"]
        if not rows:
            return "There is no repeat folder yet."
        dest = rows[0]["folder"]
        for f in [p] + DC.caption_of(p):
            shutil.move(str(f), str(dest / f.name))
        return f"✔ {p.name} moved into {dest.name}/."
    return ""


# --------------------------------------------------------------------------- the state of every step
def launch_scripts(ds) -> list[Path]:
    return sorted(root(ds).glob("*.sh")) if ds else []


def step_states(ds) -> dict:
    """{key: (state, text)} with state done | todo | optional | warn."""
    out = {}
    if not ds:
        return out
    m = meta(ds)
    imgs = images(ds)
    n = len(imgs)
    st = m["steps"]
    labels = type_labels()
    out["folder"] = ("done", f"{n} pictures")
    sc = _load(ds, "scan.json")
    out["scan"] = (("done", f"done {sc['at']} · {sum(1 for e in sc['pics'].values() if e['bad'])} not usable")
                   if sc else ("optional", "not done (optional)"))
    if m["types"]:
        out["setup"] = ("done", " + ".join(labels.get(t, t) for t in m["types"]) + f" · {PURPOSE_LABEL.get(m['purpose'], m['purpose'])}"
                        + (" · watermarks removed" if m["wm_removed"] else ""))
    else:
        out["setup"] = ("todo", "LoRA type and purpose not set yet")
    so = sort_result(ds)
    if so:
        changed = any(str(p) not in so["pics"] for p in imgs)
        out["sort"] = ("warn" if changed else "done", f"done {so['at']} · baseline {so['baseline']}"
                       + (" · new pictures since" if changed else ""))
    else:
        out["sort"] = ("todo", "not sorted yet")
    sel = st.get("select")
    res_n = reserve_count(ds)
    out["select"] = (("done", f"{n} kept · {res_n} in {RESERVE}/ (can be swapped back in)") if sel else
                     ("todo", f"{n} pictures, none set aside yet" + (f" · {res_n} in {RESERVE}/" if res_n else "")))
    rz = st.get("resize")
    out["resize"] = (("done", f"done · originals in {ORIGINALS}/") if rz else ("todo", "not resized yet"))
    sets = caption_sets(ds)
    miss = []
    for s in sets:
        c, t = set_count(ds, s["ext"])
        if c < t:
            miss.append(f"{t - c} without a caption in IMG{s['ext']}")
    if not n:
        out["tags"] = ("todo", "no pictures")
    elif miss:
        have_any = any(set_count(ds, s["ext"])[0] for s in sets)
        out["tags"] = ("warn" if have_any else "todo", "; ".join(miss))
    else:
        out["tags"] = ("done", f"{n} captions" + (f" × {len(sets)} sets" if len(sets) > 1 else ""))
    rv = st.get("review")
    out["review"] = (("done", f"done {rv['at']}") if rv else ("todo", "after the tags"))
    scripts = launch_scripts(ds)
    out["script"] = (("done", ", ".join(p.name for p in scripts[:3]) + (" …" if len(scripts) > 3 else "")) if scripts
                     else ("todo", "none yet"))
    return out


def next_step(ds) -> int:
    s = step_states(ds)
    for num, key, _ in STEPS:
        if key in ("folder", "scan"):
            continue
        if s.get(key, ("todo",))[0] in ("todo", "warn"):
            return num
    return 9


_SIG_CACHE: dict = {}


def ds_sig(ds) -> tuple:
    """A cheap fingerprint of a dataset: changes when pictures, captions, steps, watermarks or the consistency check
    change (folder and file times only, no picture is opened). The lists above the steps are cached by it."""
    r = root(ds)
    parts = []
    try:
        parts.append(r.stat().st_mtime)
        for c in os.scandir(r):
            if c.is_dir():
                parts.append((c.name, c.stat().st_mtime))
                if c.name not in (FLOW_DIR, "model", "log", "LoraManager"):
                    try:
                        for cc in os.scandir(c.path):          # repeat folders one level down
                            if cc.is_dir():
                                parts.append((cc.name, cc.stat().st_mtime))
                    except OSError:
                        pass
            elif c.name.endswith(".sh"):
                parts.append((c.name, c.stat().st_mtime))
        fd = r / FLOW_DIR
        if fd.is_dir():
            parts += [(f.name, f.stat().st_mtime) for f in fd.glob("*.json")]
        mk = r / DC.SET_ASIDE / DC.MARKS_FILE
        if mk.is_file():
            parts.append(("marks", mk.stat().st_mtime))
        try:
            import consistency as CC
            cf = CC.result_path(ds)
            if cf.is_file():
                parts.append(("cc", cf.stat().st_mtime))
        except Exception:                                # noqa: BLE001
            pass
    except OSError:
        return (time.time(),)
    return tuple(parts)


def _cached(kind, ds, fn):
    sig = ds_sig(ds)
    hit = _SIG_CACHE.get((kind, ds))
    if hit and hit[0] == sig:
        return hit[1]
    val = fn()
    _SIG_CACHE[(kind, ds)] = (sig, val)
    return val


def overview(ds_list) -> list[dict]:
    out = []
    labels = dict((a, c) for a, b, c in STEPS)

    def one(ds):
        try:
            n = len(images(ds))
            nxt = next_step(ds)
        except Exception:                                # noqa: BLE001
            n, nxt = 0, 2
        return {"ds": ds, "n": n, "next": nxt, "label": labels[nxt]}
    for ds in ds_list:
        out.append(_cached("overview", ds, lambda ds=ds: one(ds)))
    return out


# --------------------------------------------------------------------------- Improvements
def improvements(ds) -> list[tuple[str, str]]:
    """What could be better, from what is known already (nothing is scanned here). Cached per dataset."""
    return _cached("improvements", ds, lambda: _improvements(ds))


def _improvements(ds) -> list[tuple[str, str]]:
    out = []
    m = meta(ds)
    imgs = images(ds)
    n = len(imgs)
    if not n:
        return [("no pictures", "The repeat folders hold no pictures.")]
    so = sort_result(ds)
    if not so:
        if not sg()._checked_count(imgs):
            out.append(("not checked", "The pictures were never sorted: sizes, duplicates and framings are unknown."))
    else:
        alike = sum(1 for e in so["pics"].values() if e["cat"] == "alike")
        if m["purpose"] == "character" and n > 60:
            sug, _, _ = suggest_size(ds, so)
            out.append(("too many pictures", f"{n} pictures" + (f", {alike} of them alike" if alike else "")
                        + f". About {sug} would train faster with the same result."))
        if m["purpose"] == "character" and so.get("ai") and not so["framing"].get("full body"):
            out.append(("framing", "No full-body pictures: the LoRA will struggle with whole outfits and poses. Add 4–6."))
        for note in so.get("notes") or []:
            if "full-body" not in note:
                out.append(("balance", note))
    if m["purpose"] == "character" and 0 < n < 15:
        out.append(("few pictures", f"Only {n} pictures: 20 or more give a steadier likeness."))
    for s in caption_sets(ds):
        c, t = set_count(ds, s["ext"])
        if 0 < c < t:
            out.append(("captions", f"{t - c} pictures have no caption in IMG{s['ext']} (added after tagging)."))
        elif not c:
            out.append(("captions", "No captions yet." if s["ext"] == ".txt" else f"No captions in IMG{s['ext']} yet."))
    try:
        import consistency as CC
        res = CC.load_result(ds)
        if res and not res.get("applied"):
            nc = sum(1 for g in res["groups"] if g["kind"] == "contradiction")
            if nc:
                out.append(("captions", f"{nc} trait{'s' if nc != 1 else ''} named in different ways in the captions "
                                        "(consistency not decided yet)."))
    except Exception:                                    # noqa: BLE001
        pass
    found, pending = sg()._marks_counts(ds)
    if pending:
        out.append(("watermarks", f"{pending} picture{'s' if pending != 1 else ''} with a watermark not decided yet."))
    if not m["types"]:
        out.append(("setup", "LoRA type and purpose not set: the sort and the size suggestion need them."))
    if not launch_scripts(ds):
        out.append(("launch script", "No launch script yet."))
    return out


# --------------------------------------------------------------------------- Improvements: fix many datasets at once
BATCH = {"running": False, "stop": False, "queue": [], "current": "", "step": "", "results": [], "started": 0.0}
BATCH_MAX = 50                         # "less is better": at most this many pictures per dataset
DEFAULT_TYPE = "anima"
BATCH_FILE = Path(__file__).resolve().parent / "batch_settings.json"     # Settings › Automatic improvement
BATCH_DEFAULTS = {
    "keep_mode": "suggested",      # suggested: the suggestion for the type, at most max_pics · fixed: max_pics
    "select_mode": "safe",         # safe: cleanest pictures first · varied: more close-ups, angles, outfits, details
    "max_pics": BATCH_MAX,
    "default_type": DEFAULT_TYPE,  # only for a dataset without a LoRA type
    "default_purpose": "character",
    "watermarks": "remove",        # remove (crop if that fails) · crop · aside · leave
    "upscale": True,               # upscale pictures a little too small (picture tools)
    "captions": "missing",         # missing: only pictures without a caption · all: write every caption again
    "consistency": "ignore",       # ignore · check: run the consistency check
    "cons_each": True,             # with check: answer every trait with "This picture only" (else you decide in step 7)
    "safety_pct": 50,              # move nothing when more than this % of the pictures look not usable
    "launch_script": False,        # write a launch script with the recommended settings at the end
}


def batch_settings() -> dict:
    try:
        d = json.loads(BATCH_FILE.read_text(encoding="utf-8"))
    except Exception:                                    # noqa: BLE001
        d = {}
    out = dict(BATCH_DEFAULTS)
    out.update({k: v for k, v in d.items() if k in BATCH_DEFAULTS})
    out["max_pics"] = max(5, min(1000, int(out["max_pics"] or BATCH_MAX)))
    out["safety_pct"] = max(5, min(100, int(out["safety_pct"] or 50)))
    return out


def batch_settings_save(d: dict) -> dict:
    cur = batch_settings()
    cur.update({k: v for k, v in d.items() if k in BATCH_DEFAULTS})
    BATCH_FILE.write_text(json.dumps(cur, indent=1), encoding="utf-8")
    return batch_settings()


def batch_summary(b: dict | None = None) -> str:
    """The automatic improvement in one line per step, from the settings."""
    b = b or batch_settings()
    lab = type_labels()
    keep = (f"the suggestion for the type, at most {b['max_pics']} pictures" if b["keep_mode"] == "suggested"
            else f"{b['max_pics']} pictures (fewer if there are not that many)")
    keep += ", picked for variety" if b.get("select_mode") == "varied" else ", the cleanest first"
    wm = {"remove": "removed (cropped if that fails)", "crop": "cropped off", "aside": "set aside",
          "leave": "left as they are"}[b["watermarks"]]
    return (f"2 Usable scan: not usable pictures to `_reserve/` (nothing moves when more than {b['safety_pct']} % look "
            f"not usable) · 3 Setup: the dataset's own LoRA type and purpose, else "
            f"{lab.get(b['default_type'], b['default_type'])} + {PURPOSE_LABEL.get(b['default_purpose'], b['default_purpose'])}"
            f" · 4 Sort · 5 Select: {keep}, the rest to `_reserve/` · 6 Resize"
            + (", small pictures upscaled" if b["upscale"] else "") + f", watermarks {wm} · 7 Tags: "
            + ("captions only for pictures without one" if b["captions"] == "missing" else "every caption written again")
            + (("; the consistency check runs and every trait is answered with “This picture only”" if b["cons_each"]
                else "; the consistency check runs and its findings wait for you in step 7") if b["consistency"] == "check"
               else "; the consistency check is left out")
            + (" · 9 a launch script with the recommended settings." if b["launch_script"]
               else ". Review and Launch script are skipped."))


def _batch_one(ds, url, model, say) -> list[str]:
    """Steps 2-7 with the defaults. Returns what was done, one line per step."""
    S = sg()
    B = batch_settings()
    done = []
    m = meta(ds)
    if not m["types"]:                                   # 3 Setup: keep what the dataset has, else the default
        if B["default_type"] in type_labels():
            m["types"] = [B["default_type"]]
        if not (m.get("steps") or {}).get("setup"):
            m["purpose"] = B["default_purpose"]
    m["steps"]["setup"] = {"at": time.strftime("%Y-%m-%d %H:%M"), "auto": True}
    meta_save(ds, m)
    if not m["types"]:
        return [f"skipped: no LoRA type known and {B['default_type']} is not among the LoRA types"]
    labels = type_labels()
    done.append("setup: " + " + ".join(labels.get(t, t) for t in m["types"]) + f" · {m['purpose']}")
    say("2 Usable scan")                                 # 2 Usable scan, not usable -> _reserve
    sc = usable_scan(ds, m["types"], upscale=B["upscale"])
    bad = [p for p, e in sc["pics"].items() if e["bad"] and Path(p).exists()]
    if bad and len(bad) > len(sc["pics"]) * B["safety_pct"] / 100:   # safety: never empty a dataset without you looking
        done.append(f"usable scan: {len(bad)} of {len(sc['pics'])} not usable — too many to move without you looking; "
                    "left in place (check step 2 by hand)")
        bad = []
    elif bad:
        move_to_reserve(ds, bad)
        done.append(f"usable scan: {len(bad)} not usable to {RESERVE}/")
    else:
        done.append("usable scan: all usable")
    if BATCH["stop"]:
        return done + ["stopped"]
    for line in sort_run(ds, m["types"], m["purpose"], m["wm_removed"], url, model, upscale=B["upscale"]):   # 4 Sort
        if BATCH["stop"]:
            STOP.set()
        say(f"4 Sort · {line}" if line != "done" else "4 Sort done")
    so = sort_result(ds) or {}
    done.append(f"sort: baseline {so.get('baseline', '?')}" + ("" if so.get("ai") else f" (no Ollama: {so.get('note', '')})"))
    if BATCH["stop"]:
        return done + ["stopped"]
    say("5 Select")                                      # 5 Select: the suggestion, at most BATCH_MAX
    n, _, _ = suggest_size(ds, so)
    n = min(n, B["max_pics"]) if B["keep_mode"] == "suggested" else B["max_pics"]
    rows = rank(ds, n, meta(ds).get("select_mode") or B["select_mode"])
    keep = [r["path"] for r in rows if r["keep"]]
    if keep:
        msg = apply_select(ds, keep)
        done.append(f"select: {len(keep)} kept" + (f", {len(rows) - len(keep)} to {RESERVE}/" if len(rows) > len(keep) else ""))
    else:
        done.append("select: nothing to keep, skipped (" + msg_none(rows) + ")")
    if BATCH["stop"]:
        return done + ["stopped"]
    say("6 Resize")                                      # 6 Watermarks (recommended choice) and resize
    if not m["wm_removed"] and B["watermarks"] != "leave":
        w = S.wm_apply(ds, "", only_pending=True, prefer=B["watermarks"])
        if w and w != "Nothing changed.":
            done.append("watermarks: " + re.sub(r"\s*The step plan above is updated\.", "", w.lstrip("✔ ").split("\n")[0]))
    ms = default_max_side(m["types"])
    r = apply_resize(ds, ms, upscale=B["upscale"])
    done.append("resize: " + r.lstrip("✔ ").split(" The untouched")[0])
    if BATCH["stop"]:
        return done + ["stopped"]
    import features as FT
    if not FT.have("ollama"):
        done.append("captions: not written (Ollama missing)")
    for st in (caption_sets(ds) if FT.have("ollama") else []):   # 7 Tags: only pictures without a caption
        tkey = st["tkeys"][0] if st["tkeys"] else m["types"][0]
        _, style, length, detail = S.oc_recommend(tkey, m["purpose"])
        style = st["style"]
        CAP = S.CAP
        if CAP is not None:
            CAP.PURPOSE["value"] = m["purpose"]
            CAP.PURPOSE["words"] = m.get("concept_words") or ""
        leave = {"character": CAP.DEFAULT_LEAVE_OUT if CAP else "", "style": CAP.STYLE_LEAVE_OUT if CAP else ""}.get(
            m["purpose"], m.get("concept_words") or "")
        last = ""
        for line in write_set(ds, url, model, m.get("trigger") or ds, leave, style, length, detail, st["ext"],
                              only_missing=B["captions"] == "missing"):
            last = line
            if BATCH["stop"]:
                STOP.set()
            say(f"7 Tags · {line}")
        c, t = set_count(ds, st["ext"])
        done.append(f"captions IMG{st['ext']}: {c}/{t}" + ("" if c == t else f" ({last[:80]})"))
    m = meta(ds)
    m["steps"]["tags"] = {"at": time.strftime("%Y-%m-%d %H:%M"), "auto": True}
    meta_save(ds, m)
    if B["consistency"] == "check" and m["purpose"] == "character" and not BATCH["stop"]:  # characters only
        say("7 Tags · consistency check")
        try:
            msg = S.cc_find(ds, url, model, m.get("trigger") or ds, progress=lambda *a, **k: None)
            import consistency as CC
            res = CC.load_result(ds) or {}
            groups = res.get("groups", [])
            nc = sum(1 for g in groups if g["kind"] == "contradiction")
            if res and groups and B["cons_each"]:        # "This picture only" for every trait: captions keep their own
                S.cc_apply(ds, json.dumps({g["id"]: {"mode": "each", "fix": [], "own": [], "words": {}} for g in groups}),
                           url, model, m.get("trigger") or ds, confirm=True, progress=lambda *a, **k: None)
                done.append(f"consistency: {len(groups)} trait{'s' if len(groups) != 1 else ''} answered with "
                            "“This picture only”")
            else:
                done.append(f"consistency: {nc} trait{'s' if nc != 1 else ''} to decide in step 7" if nc
                            else "consistency: nothing to decide" if res
                            else f"consistency: {str(msg[0] if isinstance(msg, tuple) else msg)[:80]}")
        except Exception as e:                           # noqa: BLE001
            done.append(f"consistency: not run ({e})")
    if B["launch_script"] and not BATCH["stop"]:
        say("9 Launch script")
        for t in m["types"]:
            note = S.auto_script(ds, t)
            done.append("launch script: " + (re.sub(r"[*`]", "", note).strip()[:140] or "not written"))
    return done


def _rank_varied(ds, target: int) -> list[dict]:
    res = sort_result(ds) or {}
    pics = res.get("pics") or {}
    rows = []
    for p in sorted(str(x) for x in candidates(ds)):
        e = pics.get(p)
        rows.append({"path": p, "cat": "new", "q": 0.5, "framing": "", "outfit": "", "place": ""} if e is None
                    else {"path": p, **e})
    cands = [r for r in rows if r["cat"] in ("good", "detail", "alike", "new")]
    embs = {}
    try:
        import pictools
        for r in cands:
            a = pictools.cached(r["path"])
            if a and a.get("emb"):
                embs[r["path"]] = pictools.emb(a)
    except Exception:                                    # noqa: BLE001
        embs = {}
    detail_max = max(1, round(0.2 * target))
    keep, seen = [], {"framing": Counter(), "outfit": Counter(), "place": Counter()}
    while len(keep) < target and cands:
        best, best_s = None, -9.0
        for r in cands:
            if r["cat"] == "detail" and sum(1 for k in keep if k["cat"] == "detail") >= detail_max:
                continue
            s_ = 0.45 * r["q"] - (0.15 if r["cat"] == "alike" else 0.0)
            for k, w in (("framing", 0.25), ("outfit", 0.15), ("place", 0.15)):
                v = r.get(k) or ""
                if v:
                    s_ += w / (1 + seen[k][v])               # something not in the selection yet counts most
            if r["cat"] == "detail":
                s_ += 0.1
            if embs.get(r["path"]) and keep:
                sims = [sum(a * b for a, b in zip(embs[r["path"]], embs[k["path"]])) for k in keep if embs.get(k["path"])]
                if sims:
                    s_ -= 0.5 * max(0.0, max(sims) - 0.8)     # looks much like one already kept
            if s_ > best_s:
                best, best_s = r, s_
        if best is None:
            break
        keep.append(best)
        cands.remove(best)
        for k in ("framing", "outfit", "place"):
            if best.get(k):
                seen[k][best[k]] += 1
    kept = {id(r) for r in keep}
    for r in rows:
        r["keep"] = id(r) in kept
        if r["keep"]:
            bits = [b for b in (r.get("framing"), r.get("outfit"), r.get("place")) if b]
            r["reason"] = ("detail shot" if r["cat"] == "detail" else "kept for variety: " + " · ".join(bits)
                           if bits else f"quality {r['q']:.2f}")
        else:
            r["reason"] = r.get("why") or ("not needed: others show the same" if r["cat"] in ("good", "alike", "new")
                                           else r["cat"])
    return rows


def msg_none(rows) -> str:
    return f"{len(rows)} pictures, none usable" if rows else "no pictures"


def batch_start(ds_list) -> str:
    S = sg()
    if BATCH["running"]:
        return "A batch is already running."
    if S.RUN.running():
        return "A training run is active: it uses the GPU that Ollama needs. Start the batch when it has finished."
    ds_list = [d for d in ds_list if d]
    if not ds_list:
        return "Tick one or more datasets first."
    url, model = S.oc_settings()
    BATCH.update(running=True, stop=False, queue=list(ds_list), current="", step="", results=[], started=time.time())

    def work():
        try:
            while BATCH["queue"] and not BATCH["stop"]:
                ds = BATCH["queue"].pop(0)
                BATCH["current"], BATCH["step"] = ds, "starting"
                t0 = time.time()

                def say(s, ds=ds):
                    BATCH["step"] = s
                try:
                    lines = _batch_one(ds, url, model, say)
                    ok = not any(l == "stopped" for l in lines)
                except Exception as e:                   # noqa: BLE001
                    lines, ok = [f"error: {e!r}"], False
                BATCH["results"].append({"ds": ds, "ok": ok, "lines": lines, "secs": round(time.time() - t0)})
        finally:
            BATCH.update(running=False, current="", step="")
            STOP.clear()
    threading.Thread(target=work, daemon=True, name="dataset-batch").start()
    return f"Started: {len(ds_list)} dataset{'s' if len(ds_list) != 1 else ''}."


def batch_stop() -> str:
    if not BATCH["running"]:
        return "No batch is running."
    BATCH["stop"] = True
    STOP.set()
    return "Stopping after the current picture …"
