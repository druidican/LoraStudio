# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""The Datasets flow: what each of the nine steps does to a dataset folder, and what it remembers.

  1 Folder        a folder with only pictures becomes a dataset: <N>_<name>/ with the pictures, _originals/ with
                  untouched copies, model/ and log/
  2 Setup         LoRA types (one or more), purpose, trigger and class word, "watermarks already removed"; shows
                  whether the types' model files and an Ollama picture model are there
  3 Usable scan   size (for the LoRA types), sharpness, compression, broken files (no AI); not usable -> _reserve/
  4 Sort          pass 1 quick (technical), pass 2 detailed (Ollama: subject, framing, outfit, place, quality,
                  watermarks); groups of pictures that are too alike, duplicates, detail shots (kept), not good enough
  5 Select        a suggested size, picked by quality first and then by what each picture adds; the rest -> _reserve/
  6 Watermarks and size  keep the shape, scale down to a longest side; per picture: keep or crop; originals in _originals/
  7 Captions      one caption set per caption style (IMG.txt, IMG.sentences.txt)
  8 Review        the folder, the pictures and their captions, and checks
  9 Train         one launch script per LoRA type (sdgui's launch script page), then into the queue

Here: the dataset's meta and pins, step 1 and 8, and the state of every step. Steps 3-5 are in dssort.py, step 6 in
dsresize.py, step 7 in dscaptions.py (worn items: dsworn.py); the dataset folders and files in dsfiles.py.

State lives in <dataset>/.lsflow/: dataset.json (setup and step times), scan.json, sort.json, select.json,
resize.json. Nothing in there is needed by the trainers.
"""
from __future__ import annotations
__version__ = "1.271126b"  # LoRA Studio version, see version.py

import json
import os
import re
import shutil
import sys
import threading
import time
from pathlib import Path

import datacheck as DC
import dssort
import dscaptions

FLOW_DIR = ".lsflow"
ORIGINALS = "_originals"
RESERVE = "_reserve"
STOP = threading.Event()
SAME_LIMIT = 3
STEPS = [(1, "folder", "Folder"), (2, "setup", "Setup"), (3, "scan", "Usable scan"), (4, "sort", "Sort"),
         (5, "select", "Select"), (6, "resize", "Watermarks and size"), (7, "tags", "Captions"), (8, "review", "Review"),
         (9, "script", "Train")]
PURPOSES = [("Character / person", "character"), ("Art style", "style"), ("Object / item", "object"),
            ("Worn item (clothes, equipment, jewelry)", "worn"), ("Action / pose", "action"),
            ("Concept / effect", "concept")]
PURPOSE_LABEL = {"character": "Character", "style": "Art style", "object": "Object", "worn": "Worn item",
                 "action": "Action / pose", "concept": "Concept / effect"}
CONCEPT_KINDS = ("object", "worn", "action", "concept")
WORN_KINDS = [("Clothes", "clothes"), ("Equipment (armour, bags, tools worn)", "equipment"),
              ("Jewelry and small items worn", "jewelry")]
CONCEPT_ASK = {"object": "The object, in a few words (e.g. katana, sword)",
               "worn": "The item's own look, in a few words (e.g. silver leaf design; red plaid wool)",
               "action": "The action or pose, in a few words (e.g. kneeling, one knee down)",
               "concept": "The concept or effect, in a few words (e.g. glowing aura, wet clothes)"}


def sg():
    return sys.modules["sdgui"]


def root(ds) -> Path:
    import dsfiles
    return dsfiles.lora_path(ds)


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


# ---- pins: your own decision for a picture (step 5), kept through new sorts and winning over every rule
PIN_LABEL = {"keep": "always keep (your pin)", "never": "never use (your pin)"}


def _pin_key(p) -> str:
    return Path(str(p)).stem.lower()                     # by name: survives moves, a renamed repeat folder, a new format


def pins(ds) -> dict:
    """{picture name: "keep" | "never"}"""
    return dict(meta(ds).get("pins") or {}) if ds else {}


def set_pin(ds, path, state: str) -> str:
    m = meta(ds)
    p = dict(m.get("pins") or {})
    if state in PIN_LABEL:
        p[_pin_key(path)] = state
    else:
        p.pop(_pin_key(path), None)
    m["pins"] = p
    meta_save(ds, m)
    return state if state in PIN_LABEL else ""


def pin_of(pn: dict, path) -> str:
    return pn.get(_pin_key(path), "")


def _apply_pins(ds, rows: list, cands: list) -> tuple[list, list]:
    """(pinned 'keep' rows, the candidates without any pinned row). A 'never' picture is never a candidate;
    an 'always keep' one is taken first, whatever its category (even a picture the sort left out)."""
    pn = pins(ds)
    if not pn:
        return [], cands
    for r in rows:
        r["pin"] = pin_of(pn, r["path"])
    first = [r for r in rows if r["pin"] == "keep"]
    return first, [r for r in cands if not r.get("pin")]


def _pin_reasons(rows):
    for r in rows:
        if r.get("pin"):
            r["keep"] = r["pin"] == "keep"
            r["reason"] = PIN_LABEL[r["pin"]]


def mark_step(ds, key, info=None):
    m = meta(ds)
    m["steps"][key] = {"at": time.strftime("%Y-%m-%d %H:%M"), **(info or {})}
    meta_save(ds, m)


def images(ds) -> list[Path]:
    """The training pictures: the dataset's own repeat folder <N>_<dataset name> only."""
    import sd_prepare
    return sd_prepare.oc_images(ds) if ds else []


# Pictures outside the repeat folder (_reserve/, _extra/, other repeat folders, other folders, the dataset folder
# itself) are left out of everything (counts, checks) unless the usable scan (step 3) is told to look
# at them: then steps 3, 4 and 5 treat them as candidates, the ones kept in step 5 join the repeat folder, and the
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
    """The pictures steps 3, 4 and 5 look at: the repeat folder, plus the others when step 3 says so."""
    imgs = images(ds)
    return imgs + extra_pictures(ds) if include_extra(ds) else imgs


def join_repeat_folder(ds, paths) -> tuple[int, list]:
    """Move pictures from outside the repeat folder into it (with their captions); a name that is taken gets _2, _3."""
    import sd_script
    rows = [x for x in sd_script.dataset_stats(ds)["rows"] if not x.get("virtual")]
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
    import sd_script
    if not ds:
        return ""
    notes = []
    st = sd_script.dataset_stats(ds)
    r = root(ds)
    for f, _ in st.get("others") or []:
        if f.parent != r:
            continue
        dest = r / sd_script.EXTRA_DIR / f.name
        if dest.exists():
            notes.append(f"`{f.name}/` is not part of this dataset but `{sd_script.EXTRA_DIR}/{f.name}` exists: move it yourself")
            continue
        try:
            dest.parent.mkdir(exist_ok=True)
            f.rename(dest)
            notes.append(f"`{f.name}/` moved to `{sd_script.EXTRA_DIR}/` (not part of this dataset, so not trained)")
        except OSError as e:
            notes.append(f"could not move `{f.name}/`: {e}")
    if include_extra(ds):
        set_include_extra(ds, False)
        notes.append("pictures outside the repeat folder are left out again (step 3)")
    return "; ".join(notes)


def _sig(ps) -> str:
    return str(len(ps)) + ":" + str(sum(int(p.stat().st_mtime) % 100000 for p in ps if p.exists()))


def type_labels() -> dict:
    import sd_script
    out = {}
    for l, v in sd_script.type_choices():
        out[v] = re.sub(r"\s*[\(\[].*?[\)\]]", "", l).strip()
    return out


def min_side(types) -> int:
    import sd_prepare
    return max([sd_prepare._dc_min_side(t) for t in (types or [])] + [0])


def tools_ok() -> bool:
    """The picture tools (quality, look-alikes, person boxes, watermark removal, upscaling) can be used now."""
    try:
        import pictools
        return pictools.ready() and not sg().RUN.running()
    except Exception:                                    # noqa: BLE001
        return False


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
    import dsfiles
    d = datasets_dir()
    out = []
    if not d:
        return out
    for f in sorted(d.iterdir(), key=lambda p: p.name.lower()):
        if not f.is_dir() or f.name.startswith((".", "_")):
            continue
        reps = [x for x in f.iterdir() if x.is_dir() and sg().REPEAT_RE.match(x.name)] if f.is_dir() else []
        if reps:
            n = sum(len(dsfiles.list_images(r)) for r in reps)
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


def add_pictures(ds: str, folder: str, sub: bool) -> str:
    """Datasets › Add pictures from a folder: copy its pictures (and their captions) into the repeat folder and
    _originals/. Pictures that are already in the dataset (the same file) are left out; too small or blurry ones
    are found by the usable scan (step 3) as usual."""
    import sd_script
    import hashlib
    src = Path(os.path.expanduser((folder or "").strip()))
    if not src.is_dir():
        return f"{folder or 'That'} is not a folder."
    rows = sd_script.dataset_stats(ds)["rows"]
    if not rows:
        return "This dataset has no repeat folder yet."
    rep = Path(rows[0]["folder"])

    def digest(p: Path) -> str:
        try:
            return hashlib.sha1(p.read_bytes()).hexdigest()
        except OSError:
            return ""
    have = {digest(p) for p in images(ds)}
    pics = _pictures_in(src, bool(sub))
    copied, same = [], 0
    for p in pics:
        if digest(p) in have:
            same += 1
            continue
        dest, k = rep / p.name, 1
        while dest.exists():
            dest, k = rep / f"{p.stem}_{k}{p.suffix}", k + 1
        shutil.copy2(p, dest)
        for c in DC.caption_of(p):
            shutil.copy2(c, rep / (dest.stem + c.name[len(p.stem):]))
        copied.append(dest)
    ensure_originals(ds, copied)
    if not pics:
        return f"No pictures in {src}."
    if not copied:
        return f"Nothing new: all {same} pictures are in the dataset already."
    return (f"✔ {len(copied)} pictures copied into {rep.name}/" + (f"; {same} were in the dataset already" if same else "")
            + ". Run the usable scan, sort and select again (steps 3-5).")


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


def ensure_folders(ds) -> list[str]:
    """Step 1's folders for a dataset that already exists: model/, log/ and _originals/<repeat folder>/ with an
    untouched copy of every picture are made when they are missing (a dataset made by hand, or folders deleted). A
    folder with only pictures (opened from the dataset list) first gets its repeat folder 10_<name>/, the pictures and
    their captions moved in. Returns what was made."""
    import dsfiles
    import sd_script
    r = root(ds)
    if not r.is_dir():
        return []
    rows = sd_script.dataset_stats(ds)["rows"]
    made = []
    if not rows:                                       # a folder with only pictures: make it a dataset where it is
        pics = _pictures_in(r, False)
        if not pics:
            return []
        rep = r / f"10_{ds}"
        rep.mkdir(exist_ok=True)
        for p in pics:
            shutil.move(str(p), str(rep / p.name))
            for c in DC.caption_of(p):
                shutil.move(str(c), str(rep / (p.stem + c.name[len(p.stem):])))
        made.append(f"{rep.name}/ ({len(pics)} pictures moved in)")
        m = meta(ds)
        if not m.get("created"):
            m["created"] = time.strftime("%Y-%m-%d %H:%M")
            m["trigger"] = m.get("trigger") or ds
            m["steps"]["folder"] = {"at": m["created"]}
            meta_save(ds, m)
        rows = sd_script.dataset_stats(ds)["rows"]
    for sub in ("model", "log"):
        if not (r / sub).is_dir():
            (r / sub).mkdir(exist_ok=True)
            made.append(f"{sub}/")
    for row in rows:
        rep = Path(row["folder"])
        orig = r / ORIGINALS / rep.name
        if not orig.is_dir():
            orig.mkdir(parents=True, exist_ok=True)
            n = ensure_originals(ds, dsfiles.list_images(rep))
            made.append(f"{ORIGINALS}/{rep.name}/ ({n} copies)")
    return made


def folder_text(ds) -> str:
    """The folders of step 1, for the step list: '10_name · _originals · model · log'."""
    import sd_script
    r = root(ds)
    reps = [Path(x["folder"]).name for x in sd_script.dataset_stats(ds)["rows"]]
    have = [n for n in (ORIGINALS, "model", "log") if (r / n).is_dir()]
    return " · ".join(reps + have) or "no repeat folder"


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


# --------------------------------------------------------------------------- the repeat folder
def rename_repeats(ds, want: int) -> str:
    """Give the dataset's one repeat folder the repeat count `want` (10_bob -> 20_bob), with its copies in _originals/,
    _reserve/ and _set_aside/originals/, and the paths LoRA Studio remembers for it. Returns what was done ('' if
    nothing), or a line starting with '⚠'."""
    import sd_script
    S = sg()
    r = root(ds)
    rows = [x for x in sd_script.dataset_stats(ds)["rows"] if not x.get("virtual")]
    if len(rows) != 1 or not want or int(want) < 1:
        return ""
    f = rows[0]["folder"]
    if rows[0]["repeats"] == int(want):
        return ""
    if S.RUN.running() and sd_script.script_home(S.RUN.script or "/") == r:
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


# --------------------------------------------------------------------------- 8 Review
def review(ds) -> dict:
    import dsfiles
    import sd_script
    r = root(ds)
    m = meta(ds)
    sets = dscaptions.caption_sets(ds)
    imgs = images(ds)
    trig = (m.get("trigger") or ds).strip().lower()
    checks, problems = [], {}
    for st in sets:
        miss = [p for p in imgs if not dscaptions.read_set(p, st["ext"])]
        if miss:
            checks.append(("bad", f"{len(miss)} picture{'s' if len(miss) != 1 else ''} without a caption in IMG{st['ext']}",
                           "", ""))
            for p in miss:
                problems.setdefault(str(p), []).append(f"no caption ({st['ext']})")
        else:
            checks.append(("ok", f"{len(imgs)} pictures, each with a caption in IMG{st['ext']}", "", ""))
        notrig = [p for p in imgs if dscaptions.read_set(p, st["ext"]) and not dscaptions.read_set(p, st["ext"]).lower().startswith(trig)]
        if notrig:
            checks.append(("warn", f"The trigger word '{trig}' is not first in {len(notrig)} caption(s) of IMG{st['ext']}", "", ""))
            for p in notrig:
                problems.setdefault(str(p), []).append("trigger word not first")
        elif imgs:
            checks.append(("ok", f"Trigger word first in all captions of IMG{st['ext']}", "", ""))
        longs = []
        for p in imgs:
            c = dscaptions.read_set(p, st["ext"]) or ""
            n = len(c.split(",")) if st["style"] == "tags" else len(c.split())
            if c and n > (40 if st["style"] == "tags" else 160):
                longs.append(p)
        if longs:
            checks.append(("warn", f"{len(longs)} very long caption(s) in IMG{st['ext']}", "", ""))
            for p in longs:
                problems.setdefault(str(p), []).append("very long caption")
    # caption files without a picture, and pictures outside the repeat folders
    stats = sd_script.dataset_stats(ds)
    exts = [s["ext"] for s in sets]
    for row in stats["rows"]:
        folder = row["folder"]
        stems = {p.stem for p in dsfiles.list_images(folder)}
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
        caps = sum(1 for p in dsfiles.list_images(row["folder"]) if dscaptions.read_set(p, exts[0]))
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
    import sd_script
    p = Path(path)
    S = sg()
    if action == "delete":
        if S.HK is not None:
            n, err = S.HK.trash([p])
            return f"✔ {p.name} moved to the Trash." if n else f"Could not move {p.name}: {'; '.join(err)}"
        return "housekeeping.py is missing."
    if action == "movein":
        rows = sd_script.dataset_stats(ds)["rows"]
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
    so = dssort.sort_result(ds)
    if so:
        find = dssort.sort_lookup(so.get("pics"))              # only NEW pictures ask for a new sort, removed ones do not
        changed = any(find(p) is None for p in imgs)
        out["sort"] = ("warn" if changed else "done", f"done {so['at']} · baseline {so['baseline']}"
                       + (" · new pictures since" if changed else ""))
    else:
        out["sort"] = ("todo", "not sorted yet")
    sel = st.get("select")
    res_n = dssort.reserve_count(ds)
    out["select"] = (("done", f"{n} kept · {res_n} in {RESERVE}/ (can be swapped back in)") if sel else
                     ("todo", f"{n} pictures, none set aside yet" + (f" · {res_n} in {RESERVE}/" if res_n else "")))
    rz = st.get("resize")
    out["resize"] = (("done", f"done · originals in {ORIGINALS}/") if rz else ("todo", "not resized yet"))
    sets = dscaptions.caption_sets(ds)
    miss = []
    for s in sets:
        c, t = dscaptions.set_count(ds, s["ext"])
        if c < t:
            miss.append(f"{t - c} without a caption in IMG{s['ext']}")
    if not n:
        out["tags"] = ("todo", "no pictures")
    elif miss:
        have_any = any(dscaptions.set_count(ds, s["ext"])[0] for s in sets)
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
    """A cheap fingerprint of a dataset: changes when pictures, captions, steps or watermarks change (folder and file times only, no picture is opened). The lists above the steps are cached by it."""
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
            gf = fd / "gapfill" / "made.json"
            if gf.is_file():
                parts.append(("gapfill", gf.stat().st_mtime))
        for d in r.iterdir():                            # captions rewritten in place change no folder time
            if d.is_dir() and re.match(r"^\d+_", d.name):
                parts.append(("caps", d.name, max((c.stat().st_mtime for c in d.glob("*.txt")), default=0)))
        mk = r / DC.SET_ASIDE / DC.MARKS_FILE
        if mk.is_file():
            parts.append(("marks", mk.stat().st_mtime))
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


