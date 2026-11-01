#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""
LoRA Library
============
A standalone Gradio app that catalogues your LoRAs and their datasets.

Expected dataset layout (one folder per dataset inside a "dataset folder"):

    <dataset folder>/
      Name/
        10_name/        images + .txt captions (repeat folder, any number of them)
        log/            TensorBoard logs (sd-scripts --logging_dir)
        model/          trained .safetensors (these are "your LoRAs")
        *.sh            launch scripts
        LoraManager/    LoRA Manager data (.metadata.json, preview, example images), filled by Personal sync

Only the dataset folders are scanned. The LoRA folder is read only by the Personal sync.

Tabs
  * Carddex       - one tile per dataset, newest picture, a label per LoRA type
  * Gallery       - the pictures you made with a LoRA (all versions), linked through ComfyUI metadata (picgallery.py)
  * Library       - card grid grouped by character or base model (newest version per dataset + type)
  * Details       - training settings, loss curve, editing
  * Personal sync - copy .metadata.json, preview and example images from the LoRA folder
                    into each dataset's LoraManager/ folder
  * Settings      - folders, base-model keywords and NSFW blur

Training settings always come from the .safetensors metadata (read-only).
Edits go to the LoRA Manager sidecar in the dataset (LoraManager/<name>.metadata.json),
never into the .safetensors file or the LoRA folder.

Run:  python lora_library.py            (opens on http://127.0.0.1:7870)
"""

from __future__ import annotations
__version__ = "1.011126"  # LoRA Studio version, see version.py

import contextlib
import copy
import hashlib
import html
import inspect
import json
import math
import os
import pickle
import re
import shutil
import struct
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import gradio as gr

try:                                   # shared "Completed" marks (completed.py next to this file)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import completed as COMPLETED
except Exception:
    COMPLETED = None
import pandas as pd
try:                                   # copy to LoRA folder, Trash, leftovers, orphaned meta files
    import housekeeping as HK
except Exception:                      # noqa: BLE001
    HK = None
from PIL import Image, ImageOps
import picgallery as PG                # Gallery tab: which pictures were made with which LoRA

# --------------------------------------------------------------------------- #
# Files and constants
# --------------------------------------------------------------------------- #

import lspaths as _LP  # noqa: E402  (program/lspaths.py: ROOT, PROG, USER)
APP_DIR = _LP.USER                  # your settings, data and caches (userdata/)
SETTINGS_FILE = APP_DIR / "library_settings.txt"  # (version 1 used settings.txt, migrated once)
FOLDERS_BLOCK = LOOK_BLOCK = None                 # settings panels, see build_library_ui()
ON_SETTINGS_SAVED: list = []                      # callbacks(settings) run after saving, used by lora_studio.py
DATA_FILE = APP_DIR / "library_data.json"        # character overrides + flags
HASH_CACHE_FILE = APP_DIR / "hash_cache.json"    # sha256 cache
CACHE_FILE = APP_DIR / "library_cache.pkl"       # last scan, loaded at startup
CACHE_VERSION = 6
SCAN_LOCK = threading.RLock()   # one scan at a time
THUMB_DIR = APP_DIR / ".thumbs"

ARCHIVE_DIR = "LoraManager"                      # subfolder inside each dataset
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
SKIP_DIRS = {"venv", ".venv", ".git", "node_modules", "__pycache__", "site-packages",
             ".thumbs", "sample", "samples", ARCHIVE_DIR, "_set_aside"}
LOG_DIR_NAMES = ("log", "logs")
EPOCH_RE = re.compile(r"^(.*)-(\d{6})$")
REPEAT_RE = re.compile(r"^(\d+)_(.+)$")
NSFW_LEVELS = {"PG (1)": 1, "PG-13 (2)": 2, "R (4)": 4, "X (8)": 8, "XXX (16)": 16}
STATUS_ORDER = {"ok": 0, "warning": 1, "problem": 2}

# --------------------------------------------------------------------------- #
# Settings (settings.txt, INI style with indented multi-line lists)
# --------------------------------------------------------------------------- #

def _parse_settings_text(text: str) -> dict:
    sections: dict[str, dict[str, list[str]]] = {}
    section = key = None
    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", ";")):
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1].strip().lower()
            sections.setdefault(section, {})
            key = None
            continue
        if section is None:
            continue
        if raw[:1] in (" ", "\t") and key is not None:
            sections[section][key].append(stripped)
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            key = k.strip()
            sections[section][key] = [v.strip()] if v.strip() else []
    return sections


def _paths(values: list[str]) -> list[str]:
    out = []
    for v in values:
        for part in v.split(";"):
            part = part.strip()
            if part:
                out.append(str(Path(os.path.expanduser(part))))
    return out


def _yes(values: list[str], default: bool) -> bool:
    if not values:
        return default
    return values[0].strip().lower() in ("yes", "true", "1", "on")


DEFAULT_BASES = [
    ("Illustrious", "illustrious, noobai, nova"),
    ("Pony", "pony"),
    ("Anima", "anima"),
    ("Chroma", "chroma"),        # before Flux: Chroma LoRAs are trained with networks.lora_flux
    ("Qwen-Image", "qwen_image, qwenimage, qwen-image, qwi"),   # before Flux and the generic names
    ("Flux.2", "flux2, flux.2, flux_2, klein, f2k"),             # before Flux: "flux2" also contains "flux"
    ("Flux", "flux"),
    ("Z-Image", "z_image, zimage, z-image"),
    ("SDXL", "sdxl, juggernaut, stable-diffusion-xl"),
]
STYLES = ("dark", "normal")


def tilde(path: str) -> str:
    """/home/<user>/x -> ~/x, so saved folders follow whoever is logged in."""
    path = str(path or "").strip()
    home = str(Path.home())
    if path == home:
        return "~"
    if path.startswith(home + os.sep):
        return "~" + path[len(home):]
    return path


def render_settings(v: dict) -> str:
    """The settings file text. Used for the default file and for saving from the Settings tab."""
    def block(paths):
        return "".join(f"\n    {tilde(p)}" for p in paths)

    bases = "\n".join(f"{n} = {k if isinstance(k, str) else ', '.join(k)}" for n, k in v["bases"])
    pictures = "\n".join(f"{t} = {tilde(d)}" for t, d in sorted((v.get("picture_dirs") or {}).items()))
    return f"""\
# LoRA Studio / LoRA Library settings
# Edit here or on the Personal Data > Settings tab. Folder lists: one path per line, indented.
# Lines starting with # are ignored.

[folders]
# Your sd-scripts folder (with venv/ and the *_train_network.py files)
sd_scripts_dir = {tilde(v["sd_scripts_dir"])}

# Where ALL your datasets are. Each dataset is a folder with repeat folders (10_name),
# log/, model/, launch scripts and LoraManager/. The FIRST one is where the Training
# tabs look for datasets and create new ones.
dataset_dirs ={block(v["dataset_dirs"])}

# Where ALL your LoRAs are, together with the LoRA Manager files ComfyUI makes
# (.metadata.json, preview and example images). Only read by Personal sync.
lora_dirs ={block(v["lora_dirs"])}

# Extra folders with launch scripts (.sh) for Training › Run, besides the dataset folders
script_dirs ={block(v["script_dirs"])}

# Folder the Monitor tab shows sample images from (empty = <sd-scripts>/output)
sample_dir = {tilde(v["sample_dir"])}

# Default folder for training model files (base model, text encoder, VAE) when a LoRA type
# has no folder of its own yet (Settings > Training)
models_dir = {tilde(v.get("models_dir", ""))}

# Where ComfyUI saves your pictures (its output folder). The Gallery tab looks in here for a folder per
# LoRA type (Anima, Illustrious, Pony, ...) and links each picture to the LoRAs in its .json metadata.
pictures_dir = {tilde(v.get("pictures_dir", ""))}

# Where Settings > Backup saves backups of these settings and LoRA Studio's databases
backup_dir = {tilde(v.get("backup_dir", ""))}

[pictures]
# Picture folder for a LoRA type that is not <pictures_dir>/<type>. Filled in by the Gallery tab
# when you enter a folder there. LoRA type = folder
{pictures}

[bases]
# Base model name = keywords looked for in the safetensors metadata (checked top to bottom)
{bases}

[display]
# Look: dark or normal
style = {v["style"]}
# Bases each character is expected to have (shown as "missing" cards). Comma separated.
expected_bases = {", ".join(v["expected_bases"])}
# Blur preview images at or above an NSFW level (1, 2, 4, 8, 16)
nsfw_blur = {"yes" if v["nsfw_blur"] else "no"}
nsfw_level = {v["nsfw_level"]}
# Also blur previews with unknown level (0) when the model is marked NSFW
blur_unknown_nsfw = {"yes" if v["blur_unknown_nsfw"] else "no"}
"""


DATA_HOME = Path.home() / "LoraStudio-data"


def guess_pictures_dir() -> str:
    """ComfyUI's output folder, when it is in one of the usual places."""
    home = Path.home()
    for c in (home / "ComfyUI" / "output", _LP.ROOT.parent / "ComfyUI" / "output", home / "comfy" / "ComfyUI" / "output",
              home / "AI" / "ComfyUI" / "output"):
        if c.is_dir():
            return str(c)
    return ""


def default_settings_values() -> dict:
    return {
        "sd_scripts_dir": str(_LP.ROOT / "sd-scripts" if (_LP.ROOT / "sd-scripts").is_dir()   # install.sh layout
                              else Path.home() / "sd-scripts"),
        "dataset_dirs": [str(DATA_HOME / "datasets")],
        "lora_dirs": [str(DATA_HOME / "loras")],
        "script_dirs": [],
        "sample_dir": "",
        "models_dir": str(DATA_HOME / "models"),
        "pictures_dir": guess_pictures_dir(),
        "picture_dirs": {},
        "backup_dir": str(Path.home() / "LoraStudio-backups"),
        "bases": DEFAULT_BASES,
        "style": "dark",
        "expected_bases": [b for b, _ in DEFAULT_BASES],
        "nsfw_blur": False,
        "nsfw_level": 8,
        "blur_unknown_nsfw": True,
    }


def _migrate_old_settings():
    """Version 1 kept its settings in settings.txt next to the app. Copy the folders over once
    (only when that file really is ours: it has a [folders] section)."""
    old = APP_DIR / "settings.txt"
    if SETTINGS_FILE.exists() or not old.exists():
        return
    try:
        text = old.read_text(encoding="utf-8")
    except OSError:
        return
    if "[folders]" not in text:
        return
    sec = _parse_settings_text(text)
    v = default_settings_values()
    f = sec.get("folders", {})
    for k in ("lora_dirs", "dataset_dirs"):
        if _paths(f.get(k, [])):
            v[k] = _paths(f.get(k, []))
    if sec.get("bases"):
        v["bases"] = [(n, ", ".join(vals)) for n, vals in sec["bases"].items()]
    d = sec.get("display", {})
    if d.get("expected_bases"):
        v["expected_bases"] = [b.strip() for b in ",".join(d["expected_bases"]).split(",") if b.strip()]
    v["nsfw_blur"] = _yes(d.get("nsfw_blur", []), False)
    v["nsfw_level"] = _int((d.get("nsfw_level") or ["8"])[0]) or 8
    v["blur_unknown_nsfw"] = _yes(d.get("blur_unknown_nsfw", []), True)
    SETTINGS_FILE.write_text(render_settings(v), encoding="utf-8")


def load_settings() -> dict:
    _migrate_old_settings()
    if not SETTINGS_FILE.exists():
        SETTINGS_FILE.write_text(render_settings(default_settings_values()), encoding="utf-8")
    sec = _parse_settings_text(SETTINGS_FILE.read_text(encoding="utf-8"))
    dflt = default_settings_values()
    folders = sec.get("folders", {})
    bases = []
    for name, vals in sec.get("bases", {}).items():
        kws = [k.strip().lower() for k in ",".join(vals).split(",") if k.strip()]
        bases.append((name, kws))
    if not bases:
        bases = [(n, [k.strip() for k in kw.split(",") if k.strip()]) for n, kw in DEFAULT_BASES]
    elif not any("chroma" in kws for _, kws in bases) and not any(n.lower() == "chroma" for n, _ in bases):
        # settings from before Chroma existed: add it in front of Flux (saved with the next Settings save)
        at = next((i for i, (n, kws) in enumerate(bases) if "flux" in kws or n.lower() == "flux"), len(bases))
        bases.insert(at, ("Chroma", ["chroma"]))
    for new_name, kw in (("Qwen-Image", ["qwen_image", "qwenimage", "qwen-image", "qwi"]),
                         ("Flux.2", ["flux2", "flux.2", "flux_2", "klein", "f2k"])):
        # settings from before this type existed: add it in front of Flux (saved with the next Settings save)
        if bases and not any(n.lower() == new_name.lower() for n, _ in bases):
            at = next((i for i, (n, kws) in enumerate(bases) if "flux" in kws or n.lower() == "flux"), len(bases))
            bases.insert(at, (new_name, kw))
    display = sec.get("display", {})
    expected = [b.strip() for b in ",".join(display.get("expected_bases", [])).split(",") if b.strip()]
    style = ((display.get("style") or ["dark"])[0]).strip().lower()
    sd_dir = (_paths(folders.get("sd_scripts_dir", [])) or [dflt["sd_scripts_dir"]])[0]
    return {
        "sd_scripts_dir": sd_dir,
        "lora_dirs": _paths(folders.get("lora_dirs", [])),
        "dataset_dirs": _paths(folders.get("dataset_dirs", [])) or [str(Path(sd_dir) / "images")],
        "script_dirs": _paths(folders.get("script_dirs", [])),
        "sample_dir": (_paths(folders.get("sample_dir", [])) or [""])[0],
        "models_dir": (_paths(folders.get("models_dir", [])) or [""])[0],
        "pictures_dir": ((_paths(folders["pictures_dir"]) or [""])[0] if "pictures_dir" in folders
                         else guess_pictures_dir()),
        "picture_dirs": {t: _paths(v)[0] for t, v in sec.get("pictures", {}).items() if _paths(v)},
        "backup_dir": (_paths(folders.get("backup_dir", [])) or [str(Path.home() / "LoraStudio-backups")])[0],
        "bases": bases,
        "expected_bases": expected or [b for b, _ in bases],
        "style": style if style in STYLES else "dark",
        "nsfw_blur": _yes(display.get("nsfw_blur", []), False),
        "nsfw_level": _int((display.get("nsfw_level") or ["8"])[0]) or 8,
        "blur_unknown_nsfw": _yes(display.get("blur_unknown_nsfw", []), True),
    }


def write_settings(sd_dir, dataset_dirs, lora_dirs, script_dirs, sample_dir, models_dir, bases_text, expected,
                   style, nsfw_blur, nsfw_level, blur_unknown, pictures_dir=None):
    def lines(text):
        return [l.strip() for l in (text or "").splitlines() if l.strip()]

    bases = []
    for l in (bases_text or "").splitlines():
        if "=" in l and not l.strip().startswith("#"):
            k, val = l.split("=", 1)
            bases.append((k.strip(), val.strip()))
    v = {
        "sd_scripts_dir": (sd_dir or "").strip() or default_settings_values()["sd_scripts_dir"],
        "dataset_dirs": lines(dataset_dirs), "lora_dirs": lines(lora_dirs),
        "script_dirs": lines(script_dirs), "sample_dir": (sample_dir or "").strip(),
        "models_dir": (models_dir or "").strip(),
        "bases": bases or DEFAULT_BASES,
        "style": "normal" if str(style).lower().startswith("normal") else "dark",
        "expected_bases": list(expected or []),
        "nsfw_blur": bool(nsfw_blur), "nsfw_level": NSFW_LEVELS.get(nsfw_level, 8),
        "blur_unknown_nsfw": bool(blur_unknown),
    }
    now = load_settings()                       # kept: set elsewhere (Gallery tab)
    v["pictures_dir"] = now["pictures_dir"] if pictures_dir is None else (pictures_dir or "").strip()
    v["picture_dirs"] = now.get("picture_dirs") or {}
    v["backup_dir"] = now.get("backup_dir") or ""
    _write_values(v)


def set_backup_dir(folder: str):
    """Remember where Settings > Backup saves (from the Backup tab)."""
    v = load_settings()
    v["backup_dir"] = folder
    _write_values(v)


def set_picture_dir(base: str, folder: str):
    """Remember the picture folder of one LoRA type (from the Gallery tab)."""
    v = load_settings()
    dirs = {k: d for k, d in (v.get("picture_dirs") or {}).items() if k.lower() != base.lower()}
    if folder:
        dirs[base] = folder
    v["picture_dirs"] = dirs
    _write_values(v)


def _write_values(v: dict):
    if SETTINGS_FILE.exists():
        shutil.copy2(SETTINGS_FILE, SETTINGS_FILE.with_suffix(".txt.bak"))
    SETTINGS_FILE.write_text(render_settings(v), encoding="utf-8")
    for cb in ON_SETTINGS_SAVED:
        try:
            cb(load_settings())
        except Exception as e:
            print(f"[LoRA Library] settings callback failed: {e!r}")


# --------------------------------------------------------------------------- #
# Small JSON stores
# --------------------------------------------------------------------------- #

def _load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _save_json(path: Path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def file_sha256(path: Path, cache: dict) -> str:
    st = path.stat()
    key = str(path)
    hit = cache.get(key)
    if hit and hit.get("size") == st.st_size and hit.get("mtime") == st.st_mtime:
        return hit["sha"]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    sha = h.hexdigest()
    cache[key] = {"size": st.st_size, "mtime": st.st_mtime, "sha": sha}
    return sha


# --------------------------------------------------------------------------- #
# Readers: safetensors header, TensorBoard events, images
# --------------------------------------------------------------------------- #

def read_safetensors_meta(path: Path) -> dict:
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            if n <= 0 or n > 200_000_000:
                return {}
            header = json.loads(f.read(n))
        return header.get("__metadata__") or {}
    except Exception:
        return {}


def _json_field(meta: dict, key: str, default):
    v = meta.get(key)
    if not v:
        return default
    try:
        return json.loads(v)
    except Exception:
        return default


def _int(v) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return 0


def _varint(b: bytes, i: int):
    r = s = 0
    while True:
        c = b[i]
        i += 1
        r |= (c & 0x7F) << s
        if not c & 0x80:
            return r, i
        s += 7


def _pb_fields(b: bytes):
    i, n = 0, len(b)
    while i < n:
        key, i = _varint(b, i)
        f, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(b, i)
        elif wt == 1:
            v, i = b[i:i + 8], i + 8
        elif wt == 2:
            ln, i = _varint(b, i)
            v, i = b[i:i + ln], i + ln
        elif wt == 5:
            v, i = b[i:i + 4], i + 4
        else:
            return
        yield f, wt, v


def read_tb_scalars(path: Path) -> dict[str, list[tuple[int, float]]]:
    """Minimal TensorBoard event reader (no tensorboard/protobuf needed)."""
    out: dict[str, list[tuple[int, float]]] = {}
    try:
        data = path.read_bytes()
    except Exception:
        return out
    i = 0
    while i + 12 <= len(data):
        (ln,) = struct.unpack_from("<Q", data, i)
        i += 12
        rec = data[i:i + ln]
        i += ln + 4
        try:
            step, summary = 0, None
            for f, wt, v in _pb_fields(rec):
                if f == 2 and wt == 0:
                    step = v
                elif f == 5 and wt == 2:
                    summary = v
            if summary is None:
                continue
            for f, wt, v in _pb_fields(summary):
                if f != 1 or wt != 2:
                    continue
                tag = val = None
                for f2, wt2, v2 in _pb_fields(v):
                    if f2 == 1 and wt2 == 2:
                        tag = v2.decode("utf-8", "replace")
                    elif f2 == 2 and wt2 == 5:
                        val = struct.unpack("<f", v2)[0]
                    elif f2 == 8 and wt2 == 2:  # TensorProto
                        for f3, wt3, v3 in _pb_fields(v2):
                            if f3 == 5 and wt3 == 5:
                                val = struct.unpack("<f", v3)[0]
                            elif f3 == 5 and wt3 == 2 and len(v3) >= 4:
                                val = struct.unpack("<f", v3[:4])[0]
                            elif f3 == 6 and wt3 == 1:
                                val = struct.unpack("<d", v3)[0]
                            elif f3 == 4 and wt3 == 2 and len(v3) >= 4:
                                val = struct.unpack("<f", v3[:4])[0]
                if tag is not None and val is not None:
                    out.setdefault(tag, []).append((int(step), float(val)))
        except Exception:
            continue
    return out


def thumb_file(path: Path, width: int = 360) -> Path | None:
    """A small JPEG copy of a picture in .thumbs (made once; a changed picture gets a new one)."""
    try:
        st = path.stat()
        key = hashlib.md5(f"{path}|{st.st_mtime}|{st.st_size}|{width}".encode()).hexdigest()
        THUMB_DIR.mkdir(exist_ok=True)
        tp = THUMB_DIR / f"{key}.jpg"
        if not tp.exists():
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail((width, width * 2))
                tmp = tp.with_suffix(f".{threading.get_ident()}.tmp")
                im.save(tmp, "JPEG", quality=85)
                tmp.replace(tp)
        return tp
    except Exception:
        return None


def thumb_data_uri(path: Path, width: int = 360) -> str:
    """Link to the thumbnail (served by Gradio from .thumbs). The browser caches it, so pages stay small:
    pictures used to be embedded in the page itself and sent again on every page load and filter change."""
    tp = thumb_file(path, width)
    if not tp:
        return ""
    from urllib.parse import quote
    return "/gradio_api/file=" + quote(str(tp))


def warm_thumbnails(lib) -> None:
    """Make the Gallery and Library thumbnails in the background after loading, so the first page is quick."""
    t = time.time()
    n = 0
    for l in list(lib.loras.values()):
        if l.preview:
            for w in (360, 320):
                if thumb_file(l.preview, w):
                    n += 1
    print(f"[LoRA Library] {n} thumbnails ready ({time.time() - t:.1f} s)")


def load_image(path: Path, max_side: int = 1024):
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((max_side, max_side))
            return im.copy()
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

@dataclass
class Dataset:
    name: str
    root: Path
    repeat_dirs: list = field(default_factory=list)
    images: int = 0
    captions: int = 0
    log_runs: list = field(default_factory=list)      # [{path, events, ts, prefix}]
    scripts: list = field(default_factory=list)       # [{path, text, char, logging}]
    model_groups: dict = field(default_factory=dict)  # (dir, stem) -> [(epoch|None, Path)]

    @property
    def archive_dir(self) -> Path:
        return self.root / ARCHIVE_DIR


@dataclass
class Lora:
    key: str
    stem: str
    lora_path: Path | None           # copy in the LoRA folder (LoRA Manager lives here)
    dataset_path: Path | None        # copy in the dataset's model folder
    meta: dict
    epoch_files: list                # [(epoch, Path)]
    final: bool
    dataset: Dataset | None = None
    sidecar: dict | None = None
    sidecar_path: Path | None = None
    sidecar_from_archive: bool = False
    preview: Path | None = None
    examples: list = field(default_factory=list)    # [Path] LoRA Manager example images
    base: str = "Unknown"
    character: str = ""
    output_name: str = ""
    tags: list = field(default_factory=list)        # [(tag, pct)]
    loss: list = field(default_factory=list)        # [(step, value)]
    loss_tag: str = ""
    log_run: str = ""
    scripts: list = field(default_factory=list)     # [(Path, has_logging_dir)]
    samples: list = field(default_factory=list)     # [Path]
    issues: list = field(default_factory=list)      # [(level, text)]
    accepted: list = field(default_factory=list)    # warnings accepted by you: not shown, not marked
    status: str = "ok"
    gid: str = ""                                    # version group (mycharAnimaV1, V2, ... share one)
    group_name: str = ""                             # name without the version: mycharAnima
    version: int | None = None                       # 2 for mycharAnimaV2, None when the name has none

    @property
    def vlabel(self) -> str:
        return f"V{self.version}" if self.version is not None else "no version"

    @property
    def path(self) -> Path:
        return self.lora_path or self.dataset_path

    @property
    def display_name(self) -> str:
        if self.sidecar and self.sidecar.get("model_name"):
            return self.sidecar["model_name"]
        return self.stem

    @property
    def trigger_words(self) -> list:
        if self.sidecar:
            return list((self.sidecar.get("civitai") or {}).get("trainedWords") or [])
        return []

    def nsfw_level(self) -> tuple[int, bool]:
        sc = self.sidecar or {}
        civ = sc.get("civitai") or {}
        flagged = bool(civ.get("is_nsfw") or (civ.get("model") or {}).get("nsfw"))
        try:
            return int(sc.get("preview_nsfw_level") or 0), flagged
        except (TypeError, ValueError):
            return 0, flagged


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

_WATCH: dict | None = None   # folder -> mtime, filled while scanning


def walk(root, max_depth: int | None = None, skip=SKIP_DIRS):
    root_p = Path(root)
    if not root_p.is_dir():
        return
    base_depth = len(root_p.parts)
    for dp, dns, fns in os.walk(root_p):
        if _WATCH is not None:          # remember the folder's time, to notice added/deleted files later
            try:
                _WATCH[dp] = os.stat(dp).st_mtime_ns
            except OSError:
                pass
        dns[:] = [d for d in dns if d not in skip and not d.endswith("-state") and not d.startswith(".")]
        if max_depth is not None and len(Path(dp).parts) - base_depth >= max_depth:
            dns[:] = []
        yield Path(dp), dns, fns


def group_model_files(paths: list[Path]) -> dict:
    groups: dict[tuple, list] = {}
    for p in paths:
        m = EPOCH_RE.match(p.stem)
        if m:
            groups.setdefault((str(p.parent), m.group(1)), []).append((int(m.group(2)), p))
        else:
            groups.setdefault((str(p.parent), p.stem), []).append((None, p))
    return groups


def main_file(files: list) -> tuple[Path, bool, list]:
    final = next((p for e, p in files if e is None), None)
    epochs = sorted((e, p) for e, p in files if e is not None)
    return (final or epochs[-1][1]), final is not None, epochs


def strip_base_suffix(name: str, bases: list[str]) -> str:
    tokens = ["sdxl", "ill", "illustrious", "ani", "anima", "pony", "flux", "zit", "zimage", "qwi", "f2k",
              "lora", "xl"] + [b.lower().replace("-", "") for b in bases]
    s = name
    changed = True
    while changed and s:
        changed = False
        s2 = re.sub(r"[_\-\s]*v\d+(\.\d+)?$", "", s, flags=re.I)
        for t in sorted(set(tokens), key=len, reverse=True):
            if len(s2) > len(t) + 1 and s2.lower().endswith(t):
                s2 = s2[: -len(t)]
                break
        s2 = s2.rstrip("_- ")
        if s2 != s and s2:
            s, changed = s2, True
    return s or name


def detect_base(meta: dict, sidecar: dict | None, path: Path, bases: list) -> str:
    sources = [
        " ".join(str(meta.get(k, "")) for k in
                 ("modelspec.architecture", "ss_base_model_version", "ss_sd_model_name",
                  "ss_sd_model_hash_name", "ss_network_module")).lower(),
        str((sidecar or {}).get("base_model", "")).lower(),
        str(path.parent).lower(),
    ]
    for text in sources:
        if not text.strip():
            continue
        for name, kws in bases:
            if any(k and k in text for k in kws):
                return name
    if sidecar and sidecar.get("base_model"):
        return str(sidecar["base_model"])
    return "Unknown"


PREVIEW_EXT = (".webp", ".png", ".jpg", ".jpeg", ".gif", ".mp4", ".webm")


def find_preview(folder: Path, stem: str, sidecar: dict | None) -> Path | None:
    """The preview: the one LoRA Manager's card links (preview_url, the picture you chose there) first, else
    <stem>.preview.<ext> / <stem>.<ext> next to the file."""
    if sidecar and sidecar.get("preview_url"):
        p = Path(str(sidecar["preview_url"]))
        try:
            if p.is_absolute() and p.is_file() and p.suffix.lower() in IMG_EXT:
                return p
        except OSError:
            pass
    for ext in PREVIEW_EXT:
        for name in (stem + ext, stem + ".preview" + ext):
            p = folder / name
            if p.exists() and p.suffix.lower() in IMG_EXT:
                return p
    if sidecar and sidecar.get("preview_url"):
        p = Path(sidecar["preview_url"])
        if p.parent == folder and p.exists() and p.suffix.lower() in IMG_EXT:
            return p
    return None


_LISTINGS: dict[str, tuple[list, list]] = {}   # folder -> (sorted lower-case names, names); emptied per scan


def folder_files(folder: Path) -> tuple[list, list]:
    """Files in a folder, listed once per scan (a LoRA folder with thousands of files used to be listed
    again for every LoRA in it)."""
    key = str(folder)
    hit = _LISTINGS.get(key)
    if hit is None:
        names = []
        try:
            with os.scandir(folder) as it:
                for e in it:
                    try:
                        if e.is_file():
                            names.append(e.name)
                    except OSError:
                        pass
        except OSError:
            pass
        names.sort(key=str.lower)
        hit = _LISTINGS[key] = ([n.lower() for n in names], names)
    return hit


def files_starting_with(folder: Path, prefix: str) -> list[str]:
    """File names in `folder` that start with `prefix` (case-insensitive), found by binary search."""
    import bisect
    low, names = folder_files(folder)
    pre = prefix.lower()
    i = bisect.bisect_left(low, pre)
    out = []
    while i < len(low) and low[i].startswith(pre):
        out.append(names[i])
        i += 1
    return out


_LM_ROOTS: dict = {}


def lm_example_roots(lib) -> list[str]:
    """LoRA Manager's example picture folder(s) (its setting example_images_path, one sub-folder per sha256, or per
    library and then sha256), read from LoRA Manager's settings.json."""
    if "v" in _LM_ROOTS:
        return _LM_ROOTS["v"]
    own = str((lib.data.get("lm_examples_dir") or "")).strip()       # set by you (Maintenance › Sync)
    cands = [Path.home() / ".config" / "ComfyUI-LoRA-Manager" / "settings.json"]
    for r in lib.settings.get("lora_dirs") or []:
        p = Path(os.path.expanduser(r)).resolve()
        for up in [p] + list(p.parents)[:4]:
            cands.append(up / "custom_nodes" / "ComfyUI-Lora-Manager" / "settings.json")
            cands.append(up / "custom_nodes" / "comfyui-lora-manager" / "settings.json")
    out = []
    if own and Path(os.path.expanduser(own)).is_dir():
        r = Path(os.path.expanduser(own))
        out += [str(r)] + [str(d) for d in r.iterdir() if d.is_dir() and not re.fullmatch(r"[0-9a-f]{64}", d.name)]
    for c in dict.fromkeys(cands):
        try:
            data = json.loads(c.read_text(encoding="utf-8")) if c.is_file() else {}
        except (OSError, ValueError):
            continue
        root = (data or {}).get("example_images_path") if isinstance(data, dict) else ""
        if root and Path(root).is_dir():
            out.append(str(root))
            out += [str(d) for d in Path(root).iterdir() if d.is_dir() and not re.fullmatch(r"[0-9a-f]{64}", d.name)]
    _LM_ROOTS["v"] = list(dict.fromkeys(out))
    return _LM_ROOTS["v"]


def find_examples(folder: Path, stem: str, sha: str, preview: Path | None, roots: list[str]) -> list[Path]:
    """LoRA Manager example images: <stem>.<anything>.<img> next to the LoRA, or a
    sub-folder named after the LoRA's sha256 or name."""
    found: list[Path] = []
    skip = (stem.lower(), stem.lower() + ".preview")
    for name in files_starting_with(folder, stem + "."):
        p = folder / name
        if p.suffix.lower() in IMG_EXT and p != preview and p.stem.lower() not in skip:
            found.append(p)
    cands = []
    for name in filter(None, [sha, stem]):
        cands += [folder / name, folder / "example_images" / name]
        cands += [Path(r) / name for r in roots] + [Path(r) / "example_images" / name for r in roots]
    for d in cands:
        if d.is_dir():
            found += sorted(p for p in d.rglob("*") if p.suffix.lower() in IMG_EXT)
    return list(dict.fromkeys(found))


VERSION_RE = re.compile(r"(?<=[A-Za-z_\-\s])[vV](\d{1,3})(?=$|[_\-\sA-Z])")


def split_version(name: str) -> tuple[str, int | None]:
    """mycharAnimaV2 -> ("mycharAnima", 2); otherV2ILL -> ("otherILL", 2); thirdILL -> ("thirdILL", None).
    The last V<number> in the name counts; it must follow a letter/_/- and end the name or be followed
    by _, -, space or a capital letter (so words like "vivid" or "rev2design" are left alone)."""
    m = None
    for m in VERSION_RE.finditer(name):
        pass
    if not m:
        return name, None
    base = (name[:m.start()] + name[m.end():]).strip("_- ")
    base = re.sub(r"[_\-\s]{2,}", "_", base)
    return (base or name), int(m.group(1))


def repeat_name(dirname: str) -> str:
    m = REPEAT_RE.match(dirname)
    return m.group(2) if m else dirname


def fmt_lr(v) -> str:
    try:
        f = float(v)
        return f"{f:.0e}".replace("e-0", "e-") if f < 0.01 else f"{f:g}"
    except (TypeError, ValueError):
        return str(v or "")


def esc(s) -> str:
    return html.escape(str(s), quote=True)


# --------------------------------------------------------------------------- #
# Library
# --------------------------------------------------------------------------- #

class Library:
    def __init__(self):
        self.settings = load_settings()
        self.loras: dict[str, Lora] = {}
        self.datasets: dict[str, Dataset] = {}
        self.hash_cache = _load_json(HASH_CACHE_FILE, {})
        self.data = _load_json(DATA_FILE, {"loras": {}})
        self.scanned_at = ""
        self.scan_finished = 0.0
        self.watch: dict = {}          # folder -> mtime at the last scan
        self.groups: dict = {}
        self._tb_cache: dict[str, tuple[float, dict]] = {}

    # ---- helpers ------------------------------------------------------- #

    def overrides(self, key: str) -> dict:
        return self.data.setdefault("loras", {}).setdefault(key, {})

    def save_data(self):
        _save_json(DATA_FILE, self.data)

    def save_hash_cache(self):
        _save_json(HASH_CACHE_FILE, self.hash_cache)

    def base_names(self) -> list[str]:
        return [b for b, _ in self.settings["bases"]]

    def blur(self, lora: Lora) -> bool:
        s = self.settings
        if not s["nsfw_blur"]:
            return False
        level, flagged = lora.nsfw_level()
        if level >= s["nsfw_level"]:
            return True
        return level == 0 and flagged and s["blur_unknown_nsfw"]

    def dataset_by_name(self, name: str) -> Dataset | None:
        return self.datasets.get((name or "").lower())

    # ---- scanning ------------------------------------------------------ #

    def scan(self, progress=None):
        """Read all folders. Only one scan runs at a time: a second caller (another browser tab, a page
        reload during a scan) waits for the running scan and then uses its result."""
        started = time.time()
        with SCAN_LOCK:
            if self.scan_finished > started:      # someone else scanned while we waited
                return
            global _WATCH
            _LISTINGS.clear()
            _WATCH = {}
            try:
                for root in self.settings_now()["dataset_dirs"]:
                    try:
                        _WATCH[str(Path(root))] = os.stat(root).st_mtime_ns
                    except OSError:
                        _WATCH[str(Path(root))] = 0
                self._scan(progress)
                self.watch = _WATCH
            finally:
                _LISTINGS.clear()
                _WATCH = None
            self.save_cache()
            self.scan_finished = time.time()
            print(f"[LoRA Library] Scanned {len(self.loras)} LoRAs and {len(self.datasets)} datasets "
                  f"in {self.scan_finished - started:.1f} s")

    @staticmethod
    def settings_now() -> dict:
        return load_settings()

    def folders_changed(self) -> str:
        """The first folder in which files were added, deleted or renamed since the last scan ('' = none).
        Only folder times are checked, so this is quick even for thousands of files."""
        if not self.watch:
            return ""
        for d, m in self.watch.items():
            try:
                if os.stat(d).st_mtime_ns != m:
                    return d
            except OSError:
                return d
        return ""

    def ensure_loaded(self, progress=None):
        """Startup / page load: use what is in memory, else the saved scan, else scan the folders.
        When files were added or deleted since the last scan, the folders are scanned again."""
        with SCAN_LOCK:
            loaded = bool(self.loras or self.datasets) or self.load_cache()
            changed = self.folders_changed() if loaded else ""
            if loaded and not changed:
                return
        if changed:
            print(f"[LoRA Library] Files were added or removed in {changed}: scanning again.")
        self.scan(progress)

    def _scan(self, progress=None):
        self.settings = load_settings()
        self.data = _load_json(DATA_FILE, {"loras": {}})
        self.datasets = self._scan_datasets()
        self.loras = {}

        # Your LoRAs are the trained models inside the datasets (model/ folders). The LoRA folder is only read by
        # the Personal sync, which copies its LoRA Manager files into each dataset's LoraManager/ folder.
        groups = [(ds, files) for ds in self.datasets.values() for files in ds.model_groups.values()]
        total = max(1, len(groups))
        for n, (ds, files) in enumerate(groups):
            main, final, epochs = main_file(files)
            if progress is not None:
                progress((n + 1) / total, desc=f"Reading {main.stem}")
            lora = Lora(key=hashlib.md5(str(main).encode()).hexdigest()[:12], stem=main.stem,
                        lora_path=None, dataset_path=main, meta=read_safetensors_meta(main),
                        epoch_files=epochs, final=final, dataset=ds)
            self._safe_complete(lora)
            self.loras[lora.key] = lora

        self.save_hash_cache()
        self._build_groups()
        self.scanned_at = time.strftime("%Y-%m-%d %H:%M")

    # ---- version groups ------------------------------------------------ #

    def _build_groups(self):
        """All LoRAs of one LoRA type in one dataset form one group (mychar · Anima: mycharAnimaV1, mycharAnimaV2, ...),
        oldest first, so the newest is the last one. Downloads (no dataset) are grouped by name instead.
        The version number comes from the file name (mycharAnimaV2), else from the training metadata's output name."""
        self.groups: dict[str, list[Lora]] = {}
        for l in self.loras.values():
            base, ver = split_version(EPOCH_RE.sub(r"\1", l.stem))
            if ver is None and l.output_name:
                obase, ver = split_version(l.output_name)
                if ver is not None:
                    base = obase
            l.group_name, l.version = base, ver
            if l.dataset is not None:
                key = f"{l.dataset.name.lower()}|type:{COMPLETED.type_key(l.base) if COMPLETED else l.base.lower()}"
            else:
                key = f"{l.path.parent}|{base.lower()}"
            l.gid = hashlib.md5(key.encode()).hexdigest()[:12]
            self.groups.setdefault(l.gid, []).append(l)

        def started(l):
            try:
                return float(l.meta.get("ss_training_started_at") or 0) or l.path.stat().st_mtime
            except (OSError, TypeError, ValueError):
                return 0.0

        for members in self.groups.values():
            members.sort(key=lambda l: (-1 if l.version is None else l.version, started(l), l.stem.lower()))
    def group_of(self, lora: Lora) -> list:
        return self.groups.get(lora.gid) or [lora]

    def newest(self, gid: str):
        """The version a card and the LoRA page show: the one marked best (Training › Results), else the newest."""
        members = self.groups.get(gid)
        if not members:
            return None
        return best_member(members) or members[-1]


    # ---- cache --------------------------------------------------------- #

    @staticmethod
    def _settings_hash() -> str:
        """Only the settings that change what a scan finds (folders and base models). Style, NSFW blur,
        the sample folder etc. can change without throwing the saved scan away."""
        try:
            s = load_settings()
            used = {"datasets": s["dataset_dirs"], "bases": s["bases"]}
            return hashlib.md5(json.dumps(used, sort_keys=True, default=str).encode()).hexdigest()
        except Exception:
            return ""

    def save_cache(self):
        try:
            tmp = CACHE_FILE.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                pickle.dump({"version": CACHE_VERSION, "settings": self._settings_hash(),
                             "scanned_at": self.scanned_at, "loras": self.loras,
                             "datasets": self.datasets, "watch": self.watch}, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(CACHE_FILE)
        except Exception as e:
            import traceback
            print(f"[LoRA Library] Could not save the scan to {CACHE_FILE}: {e!r}\n"
                  "The next start will scan the folders again. Details:")
            traceback.print_exc()

    def load_cache(self) -> bool:
        """Load the last scan. Returns False if there is none or the settings changed."""
        if not CACHE_FILE.exists():
            print("[LoRA Library] No saved scan yet: scanning the folders (later starts use the saved scan).")
            return False
        try:
            with open(CACHE_FILE, "rb") as f:
                c = pickle.load(f)
            if c.get("version") != CACHE_VERSION:
                print("[LoRA Library] The saved scan is from an older version: scanning again.")
                return False
            if c.get("settings") != self._settings_hash():
                print("[LoRA Library] Folders or base models changed since the saved scan: scanning again.")
                return False
            self.settings = load_settings()
            self.data = _load_json(DATA_FILE, {"loras": {}})
            self.loras, self.datasets = c["loras"], c["datasets"]
            self.watch = c.get("watch") or {}
            self.scanned_at = c["scanned_at"] + " (cached)"
            self._build_groups()
            print(f"[LoRA Library] Loaded the saved scan from {c['scanned_at']} ({len(self.loras)} LoRAs). "
                  "Press Rescan folders to read the folders again.")
            return True
        except Exception as e:
            print(f"[LoRA Library] Could not read the saved scan ({e!r}): scanning again.")
            return False


    def _scan_datasets(self) -> dict[str, Dataset]:
        out: dict[str, Dataset] = {}
        for root in self.settings["dataset_dirs"]:
            root_p = Path(root)
            if not root_p.is_dir():
                continue
            for d in sorted(p for p in root_p.iterdir() if p.is_dir() and not p.name.startswith(".")):
                ds = self._read_dataset(d)
                if ds and ds.name.lower() not in out:
                    out[ds.name.lower()] = ds
        return out

    def _read_dataset(self, d: Path) -> Dataset | None:
        ds = Dataset(name=d.name, root=d)
        models = []
        for sub, dns, fns in walk(d):
            imgs = [f for f in fns if Path(f).suffix.lower() in IMG_EXT]
            if imgs and (REPEAT_RE.match(sub.name) or sub == d):
                if sub != d or not ds.repeat_dirs:
                    ds.repeat_dirs.append(sub)
                    ds.images += len(imgs)
                    names = set(fns)
                    ds.captions += sum(1 for f in imgs if (os.path.splitext(f)[0] + ".txt") in names)
            for f in fns:
                if f.endswith(".safetensors"):
                    models.append(sub / f)
        # loose images in the dataset root only count when there are no repeat folders
        if len(ds.repeat_dirs) > 1 and d in ds.repeat_dirs:
            ds.repeat_dirs.remove(d)
            ds.images = ds.captions = 0
            for sub in ds.repeat_dirs:
                imgs = [p for p in sub.iterdir() if p.suffix.lower() in IMG_EXT]
                ds.images += len(imgs)
                ds.captions += sum(1 for p in imgs if p.with_suffix(".txt").exists())
        # only the dataset's own repeat folder <N>_<dataset name> is the dataset (else the ones directly in it)
        own = [s for s in ds.repeat_dirs if s.parent == d and REPEAT_RE.match(s.name)
               and s.name[REPEAT_RE.match(s.name).end():].lower() == d.name.lower()]
        keep = own or [s for s in ds.repeat_dirs if s.parent == d]
        if keep and keep != ds.repeat_dirs:
            ds.repeat_dirs = keep
            ds.images = ds.captions = 0
            for sub in keep:
                imgs = [p for p in sub.iterdir() if p.suffix.lower() in IMG_EXT]
                ds.images += len(imgs)
                ds.captions += sum(1 for p in imgs if p.with_suffix(".txt").exists())
        ds.model_groups = group_model_files(models)
        # logs
        for ln in LOG_DIR_NAMES:
            ld = d / ln
            if not ld.is_dir():
                continue
            runs = [p for p in ld.iterdir() if p.is_dir()]
            if list(ld.glob("events.out.tfevents.*")):
                runs.append(ld)
            for run in sorted(runs):
                events = list(run.rglob("events.out.tfevents.*"))
                if not events:
                    continue
                m = re.search(r"(\d{14})$", run.name)
                ts = None
                if m:
                    try:
                        ts = time.mktime(time.strptime(m.group(1), "%Y%m%d%H%M%S"))
                    except ValueError:
                        ts = None
                prefix = run.name[: m.start()] if m else ("" if run == ld else run.name)
                ds.log_runs.append({"path": run, "events": events, "ts": ts, "prefix": prefix.lower()})
        # launch scripts
        for p in sorted(d.glob("*.sh")):
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            m = re.search(r"CHARACTER_NAME\s*=\s*[\"']?([\w.-]+)", text)
            ds.scripts.append({"path": p, "text": text, "char": m.group(1).lower() if m else "",
                               "logging": "--logging_dir" in text})
        if not (ds.repeat_dirs or ds.model_groups or ds.scripts):
            return None
        return ds

    def _safe_complete(self, lora: Lora):
        """Fill in a LoRA's details; one odd file must never stop the whole scan."""
        try:
            self._complete(lora)
        except Exception as e:
            print(f"[LoRA Library] Could not read everything for {lora.path}: {e!r}")
            if not lora.character:
                lora.character = lora.dataset.name if lora.dataset else lora.stem
            lora.issues = [("warning", f"Could not read all data ({type(e).__name__}: {e})")]
            lora.status = "warning"

    def _complete(self, lora: Lora):
        meta = lora.meta
        lora.output_name = meta.get("ss_output_name") or EPOCH_RE.sub(r"\1", lora.stem)

        # Dataset by training metadata if not found via the model file
        if lora.dataset is None:
            for n in self._meta_dataset_names(meta):
                ds = self.dataset_by_name(n)
                if ds:
                    lora.dataset = ds
                    break
        ov = self.overrides(lora.key)
        if ov.get("character") and self.dataset_by_name(ov["character"]) and lora.dataset is None:
            lora.dataset = self.dataset_by_name(ov["character"])

        # LoRA Manager data: the dataset's LoraManager/ copy (filled by the Personal sync, edited in Details)
        sc = None
        if lora.dataset:
            arch = lora.dataset.archive_dir
            lora.sidecar_path = arch / (lora.stem + ".metadata.json")
            if lora.sidecar_path.exists():
                sc = _load_json(lora.sidecar_path, None)
                lora.sidecar_from_archive = True
            if arch.is_dir():
                lora.preview = find_preview(arch, lora.stem, None)
                ex_dir = arch / f"{lora.stem}_examples"       # copies made by older versions of the sync
                if ex_dir.is_dir():
                    lora.examples = sorted(p for p in ex_dir.rglob("*") if p.suffix.lower() in IMG_EXT)
        if isinstance(sc, dict):                          # LoRA Manager's example pictures, linked by the sync
            linked = [Path(x) for x in ((sc.get("lorastudio") or {}).get("examples") or [])]
            seen = {p.name for p in lora.examples}
            lora.examples += [p for p in linked if p.suffix.lower() in IMG_EXT and p.name not in seen and p.is_file()]
        lora.sidecar = sc if isinstance(sc, dict) else None

        lora.base = detect_base(meta, lora.sidecar, lora.path, self.settings["bases"])

        # Tags (sd-scripts groups them per folder; some trainers store a flat {tag: count})
        img_total = 0
        dsd = _json_field(meta, "ss_dataset_dirs", {})
        for info in (dsd.values() if isinstance(dsd, dict) else []):
            img_total += _int(info.get("img_count") if isinstance(info, dict) else 0)
        tag_freq: dict[str, int] = {}

        def add_one(t, c):
            n = _int(c)
            if n and isinstance(t, str) and t.strip():
                tag_freq[t.strip()] = tag_freq.get(t.strip(), 0) + n

        def add_tags(d):
            if not isinstance(d, dict):
                return
            for k, v in d.items():
                if isinstance(v, dict):
                    for t, c in v.items():
                        add_one(t, c)
                else:
                    add_one(k, v)

        add_tags(_json_field(meta, "ss_tag_frequency", {}))
        dss = _json_field(meta, "ss_datasets", [])
        for ds in (dss if isinstance(dss, list) else []):
            if not isinstance(ds, dict):
                continue
            for sub in ds.get("subsets", []) or []:
                if isinstance(sub, dict):
                    img_total += _int(sub.get("img_count"))
            add_tags(ds.get("tag_frequency"))
        if not img_total:
            img_total = _int(meta.get("ss_num_train_images"))
        top = sorted(tag_freq.items(), key=lambda kv: -kv[1])[:25]
        lora.tags = [(t, round(100 * c / img_total) if img_total else c) for t, c in top]

        # Character
        if ov.get("character"):
            lora.character = ov["character"]
        elif lora.dataset:
            lora.character = lora.dataset.name
        else:
            lora.character = strip_base_suffix(lora.output_name, self.base_names())

        # Logs, scripts, samples from the dataset
        if lora.dataset:
            run = self._match_run(lora, lora.dataset.log_runs)
            if run:
                lora.log_run = str(run["path"])
                scalars: dict[str, list] = {}
                for ev in run["events"]:
                    for tag, pts in self._tb(ev).items():
                        scalars.setdefault(tag, []).extend(pts)
                for pref in ("loss/average", "loss/current", "loss/epoch"):
                    if pref in scalars:
                        lora.loss_tag = pref
                        break
                else:
                    lora.loss_tag = next((t for t in scalars if t.startswith("loss")), "")
                if lora.loss_tag:
                    lora.loss = sorted(scalars[lora.loss_tag])
            lora.scripts = self._match_scripts(lora, lora.dataset.scripts)
        self._evaluate(lora)

    def _meta_dataset_names(self, meta: dict) -> list[str]:
        names = []
        dsd = _json_field(meta, "ss_dataset_dirs", {})
        for dname in (dsd if isinstance(dsd, dict) else {}):
            names.append(repeat_name(Path(dname).name))
        dss = _json_field(meta, "ss_datasets", [])
        for ds in (dss if isinstance(dss, list) else []):
            if not isinstance(ds, dict):
                continue
            for sub in ds.get("subsets", []) or []:
                if isinstance(sub, dict) and sub.get("image_dir"):
                    p = Path(sub["image_dir"])
                    names += [repeat_name(p.name), p.parent.name]
        return list(dict.fromkeys(n for n in names if n))

    def _tb(self, path: Path) -> dict:
        try:
            mt = path.stat().st_mtime
        except OSError:
            return {}
        hit = self._tb_cache.get(str(path))
        if hit and hit[0] == mt:
            return hit[1]
        data = read_tb_scalars(path)
        self._tb_cache[str(path)] = (mt, data)
        return data

    def _match_run(self, lora: Lora, runs: list[dict]):
        if not runs:
            return None
        name = lora.output_name.lower()
        try:
            started = float(lora.meta.get("ss_training_started_at") or 0)
        except ValueError:
            started = 0
        best, best_d = None, None
        for r in runs:
            named = bool(r["prefix"]) and name.startswith(r["prefix"].rstrip("_-"))
            if r["ts"] and started:
                dist = abs(r["ts"] - started)
                if not named and dist > 600:
                    continue
            elif named:
                dist = 1e12 - (r["ts"] or 0)
            else:
                continue
            if best is None or dist < best_d:
                best, best_d = r, dist
        return best

    def _match_scripts(self, lora: Lora, scripts: list[dict]) -> list:
        name = lora.output_name
        trainer = {"Anima": "anima_train", "Flux": "flux_train"}.get(lora.base, "")
        literal, by_char, same_trainer = [], [], []
        for s in scripts:
            if trainer and trainer not in s["text"]:
                continue
            if not trainer and ("anima_train" in s["text"] or "flux_train" in s["text"]):
                continue
            same_trainer.append((s["path"], s["logging"]))
            if re.search(rf"(?<![\w]){re.escape(name)}(?![\w])", s["text"]):
                literal.append((s["path"], s["logging"]))
            elif s["char"] and name.lower().startswith(s["char"]):
                by_char.append((s["path"], s["logging"]))
        return literal or by_char or same_trainer

    def _evaluate(self, lora: Lora):
        issues = []
        ov = self.overrides(lora.key)
        if ov.get("flag") in ("warning", "problem"):
            issues.append((ov["flag"], ov.get("flag_reason") or "Flagged by you"))
        bad = [(s, v) for s, v in lora.loss if math.isnan(v) or math.isinf(v)]
        if bad:
            issues.append(("problem", f"Loss went NaN at step {bad[0][0]}"))
        if not lora.final and lora.epoch_files:
            issues.append(("warning", f"No final file - last saved epoch is {lora.epoch_files[-1][0]}"))
        try:
            ep, n_ep = int(lora.meta.get("ss_epoch") or 0), int(lora.meta.get("ss_num_epochs") or 0)
            if lora.final and ep and n_ep and ep < n_ep:
                issues.append(("warning", f"Run stopped at epoch {ep} of {n_ep}"))
        except ValueError:
            pass
        ds = lora.dataset
        if ds and ds.captions < ds.images:
            issues.append(("warning", f"{ds.images - ds.captions} dataset images have no caption"))
        accepted = set(ov.get("accepted") or [])        # warnings you looked at and accepted (LoRA page › Accept)
        lora.accepted = [t for lvl, t in issues if lvl == "warning" and t in accepted]
        issues = [(lvl, t) for lvl, t in issues if not (lvl == "warning" and t in accepted)]
        lora.issues = issues
        lora.status = max((lvl for lvl, _ in issues), key=lambda l: STATUS_ORDER[l], default="ok")


# --------------------------------------------------------------------------- #
# HTML rendering
# --------------------------------------------------------------------------- #

def sparkline(points: list, w=268, h=44) -> str:
    if len(points) < 2:
        return ""
    step = max(1, len(points) // 80)
    vals = points[::step]
    good = [v for _, v in vals if not (math.isnan(v) or math.isinf(v))]
    if not good:
        return ""
    lo, hi = min(good), max(good)
    rng = (hi - lo) or 1
    x0, x1 = vals[0][0], vals[-1][0]
    xr = (x1 - x0) or 1
    pts, nan_x = [], None
    for s, v in vals:
        x = (s - x0) / xr * w
        if math.isnan(v) or math.isinf(v):
            nan_x = nan_x if nan_x is not None else x
            continue
        y = 4 + (1 - (v - lo) / rng) * (h - 8)
        pts.append(f"{x:.1f},{y:.1f}")
    marker = (f'<line x1="{nan_x:.1f}" y1="2" x2="{nan_x:.1f}" y2="{h}" stroke="#E8A33D" '
              f'stroke-width="2" stroke-dasharray="3 3"/>') if nan_x is not None else ""
    return (f'<svg class="ll-spark" viewBox="0 0 {w} {h}" preserveAspectRatio="none" role="img" '
            f'aria-label="Loss curve"><polyline points="{" ".join(pts)}" fill="none" '
            f'stroke="#4FB3A9" stroke-width="2"/>{marker}</svg>')


def suggest_edit(lora: Lora) -> dict:
    """What LoRA Studio knows about a LoRA it trained, for the empty fields of LoRA page › Edit: the trigger word (the
    dataset's), notes with the training (from the LoRA file, its launch script and its run log) and a strength."""
    m = lora.meta or {}
    out = {}
    ds = lora.dataset.name if lora.dataset else ""
    trig = ""
    try:
        import dsflow
        if ds:
            trig = (dsflow.meta(ds).get("trigger") or "").strip()
    except Exception:                                    # noqa: BLE001
        pass
    if not trig and lora.tags:
        trig = str(lora.tags[0][0])
    # the launch script fills in what the LoRA file does not say (musubi-tuner files carry little metadata)
    sargs = {}
    try:
        sg_ = sys.modules.get("sdgui")
        if sg_ is not None and lora.scripts:
            sp = Path(lora.scripts[0][0])
            stext = sp.read_text(encoding="utf-8", errors="ignore")
            home = sg_.script_home(sp)
            for k in ("max_train_epochs", "learning_rate", "network_dim", "network_alpha", "resolution",
                      "pretrained_model_name_or_path", "dit", "max_train_steps"):
                v = sg_.script_arg(stext, k, home)
                if v:
                    sargs[k] = v
            if not trig and lora.dataset:
                trig = lora.dataset.name
    except Exception:                                    # noqa: BLE001
        pass
    if not trig and lora.dataset:
        trig = lora.dataset.name
    if trig:
        out["trigger"] = trig
    if lora.dataset and not (lora.character or "").strip():
        out["character"] = lora.dataset.name
    m = dict(m)
    for mk, sk in (("ss_num_epochs", "max_train_epochs"), ("ss_learning_rate", "learning_rate"),
                   ("ss_network_dim", "network_dim"), ("ss_network_alpha", "network_alpha"),
                   ("ss_resolution", "resolution"), ("ss_max_train_steps", "max_train_steps")):
        if not m.get(mk) and sargs.get(sk):
            m[mk] = sargs[sk]
    if not m.get("ss_sd_model_name") and (sargs.get("pretrained_model_name_or_path") or sargs.get("dit")):
        m["ss_sd_model_name"] = sargs.get("pretrained_model_name_or_path") or sargs.get("dit")
    lr = m.get("ss_unet_lr") if m.get("ss_unet_lr") not in (None, "", "None") else m.get("ss_learning_rate")
    bits = []
    when = ""
    try:
        when = time.strftime("%Y-%m-%d", time.localtime(float(m["ss_training_started_at"])))
    except (KeyError, ValueError, TypeError):
        pass
    head = "Trained" + (f" on {when}" if when else "") + (f" from the dataset {ds}" if ds else "")
    if m.get("ss_num_train_images"):
        head += f" ({m['ss_num_train_images']} pictures with repeats)"
    bits.append(head)
    facts = [f"{m['ss_num_epochs']} epochs" if m.get("ss_num_epochs") else "",
             f"{m['ss_max_train_steps']} steps" if m.get("ss_max_train_steps") else "",
             f"learning rate {fmt_lr(lr)}" if lr not in (None, "", "None") else "",
             f"dim {m['ss_network_dim']} / alpha {m['ss_network_alpha']}" if m.get("ss_network_dim") else "",
             f"resolution {m['ss_resolution']}" if m.get("ss_resolution") else "",
             f"base model {Path(str(m['ss_sd_model_name'])).name}" if m.get("ss_sd_model_name") else ""]
    if any(facts):
        bits.append(", ".join(f for f in facts if f))
    if lora.loss:
        bits.append(f"final loss {lora.loss[-1][1]:.4f}")
    if lora.scripts:
        bits.append(f"launch script {Path(lora.scripts[0][0]).name}")
    if lora.log_run:
        bits.append(f"log {Path(lora.log_run).name}")
    if len(bits) > 1 or m:
        out["notes"] = ". ".join(b[:1].upper() + b[1:] for b in bits) + "."
    out["strength"] = 1.0
    out["clip_strength"] = 1.0
    return out


def settings_rows(lora: Lora) -> list[tuple[str, str]]:
    m = lora.meta
    if not any(k.startswith("ss_") for k in m):
        return []
    lr = m.get("ss_unet_lr") if m.get("ss_unet_lr") not in (None, "", "None") else m.get("ss_learning_rate")
    opt = (m.get("ss_optimizer") or "").split(".")[-1].split("(")[0]
    batch = m.get("ss_total_batch_size") or m.get("ss_batch_size_per_device") or ""
    rows = [
        ("Base checkpoint", Path(m.get("ss_sd_model_name", "")).name),
        ("Network", f"dim {m.get('ss_network_dim', '?')} / alpha {m.get('ss_network_alpha', '?')}"),
        ("Optimizer", f"{opt} · {fmt_lr(lr)}".strip(" ·")),
        ("Scheduler", m.get("ss_lr_scheduler", "")),
        ("Epochs", f"{m.get('ss_epoch', '?')} / {m.get('ss_num_epochs', '?')}"),
        ("Steps", f"{m.get('ss_steps', '?')} / {m.get('ss_max_train_steps', '?')}"),
        ("Images", m.get("ss_num_train_images", "")),
        ("Batch size", batch),
        ("Resolution", m.get("ss_resolution", "")),
        ("Noise offset", m.get("ss_noise_offset", "")),
        ("Min SNR gamma", m.get("ss_min_snr_gamma", "")),
        ("Precision", m.get("ss_mixed_precision", "")),
        ("Seed", m.get("ss_seed", "")),
    ]
    for k, label in (("ss_training_started_at", "Started"), ("ss_training_finished_at", "Finished")):
        try:
            rows.append((label, time.strftime("%Y-%m-%d %H:%M", time.localtime(float(m[k])))))
        except (KeyError, ValueError, TypeError):
            pass
    return [(a, str(b)) for a, b in rows if str(b) not in ("", "None", "? / ?")]


def group_status(members: list) -> tuple[str, list]:
    """Worst status of a version group and all its issues, each marked with its version."""
    if len(members) == 1:
        return members[0].status, members[0].issues
    st = max((m.status for m in members), key=lambda x: STATUS_ORDER[x])
    issues = [(lvl, f"{m.vlabel}: {t}") for m in reversed(members) for lvl, t in m.issues]
    return st, issues


def group_title(members: list) -> str:
    """Name for a version group: a LoRA Manager name from the newest version that has one, else the
    name without the version (mycharAnima)."""
    for m in reversed(members):
        if m.sidecar and m.sidecar.get("model_name"):
            return m.sidecar["model_name"]
    return members[-1].group_name if len(members) > 1 else members[-1].display_name


def done_key(l: Lora) -> tuple[str, str]:
    """(dataset, LoRA type) used for the Completed mark, ('', '') when the LoRA has no dataset."""
    if l.dataset is None or COMPLETED is None:
        return "", ""
    return l.dataset.name, COMPLETED.type_key(l.base)


def is_completed(l: Lora) -> bool:
    ds, t = done_key(l)
    return bool(ds) and COMPLETED.is_done(ds, t, version=COMPLETED.version_of(l.stem))


def best_member(members: list):
    """The version of a group marked as the best one, or None."""
    if not members or COMPLETED is None:
        return None
    ds, tk = done_key(members[-1])
    if not ds:
        return None
    v = COMPLETED.best_version(ds, tk)
    return next((m for m in members if v is not None and m.version == v), None)



BEST_BADGE = ('<span class="ll-done ll-best" title="Marked as the best version in Training › Results: shown first">'
              '&#9733; Best</span>')
DONE_BADGE = '<span class="ll-done" title="Completed: its launch scripts are hidden in Training › Run">&#10003; Completed</span>'


def picture_source(members: list):
    """The newest version that has a preview image (None if no version has one)."""
    return next((m for m in reversed(members) if m.preview), None)


def version_chip(members: list) -> str:
    """'V1–V3 · 3 versions ▾': click it to fold out the older versions (the card itself shows the newest)."""
    if len(members) < 2:
        return ""
    return (f'<span class="ll-vchip ll-vchip-btn" title="Show all versions" '
            f'onclick="event.stopPropagation(); llToggle(\'{members[-1].gid}\', this.closest(\'.ll-card\'))">'
            f'{esc(members[0].vlabel)}&ndash;{esc(members[-1].vlabel)} &middot; {len(members)} versions &#9662;</span>')


def render_card(lib: Library, lora: Lora, members: list | None = None, onclick: str = "", extra: str = "",
                selectable: bool = False) -> str:
    """One Library card. With `members` (a version group, oldest first) the card shows the newest
    version, the version range, and the worst status / all issues of the group. `onclick` replaces the
    default (open Details); `extra` is HTML added at the bottom of the card."""
    members = members or [lora]
    st, issues_all = group_status(members)
    vchip = version_chip(members)
    pic = lora if lora.preview else (picture_source(members) or lora)
    if is_completed(lora):
        vchip = DONE_BADGE + vchip
    if best_member(members) is lora and len(members) > 1:
        vchip = BEST_BADGE + vchip
    if pic is not lora:
        vchip += f'<span class="ll-note ll-muted">picture from {esc(pic.vlabel)}</span>'
    blur = " ll-blur" if lib.blur(pic) else ""
    status = f'<span class="ll-dot ll-dot-{st}" title="{esc(st)}"></span>'
    trig = lora.trigger_words[0].split(",")[0] if lora.trigger_words else ""
    fav = '<span class="ll-fav" title="Favorite">&#9733;</span>' if (lora.sidecar or {}).get("favorite") else ""
    badge = f'<span class="ll-badge">{esc(lora.base)}</span>'
    where = ""
    click = onclick or f'onclick="loraPick(\'{lora.key}\', this)"'
    if selectable:
        click += f' data-gid="{lora.gid}"'
        if lora.dataset is not None:
            click += f' data-ds="{esc(lora.dataset.name.lower())}"'
    sel = '<span class="ll-selmark" aria-hidden="true"></span>' if selectable else ""
    uri = thumb_data_uri(pic.preview) if pic.preview else ""
    if uri:
        second = f"trigger: {esc(trig)}" if trig else (f"{len(lora.epoch_files)} epoch files" if lora.epoch_files else "")
        return (
            f'<button class="ll-card ll-card-img ll-st-{st}" {click} title="{esc(lora.path)}">{sel}'
            f'<span class="ll-img{blur}"><img src="{uri}" alt="" loading="lazy"></span>'
            f'<span class="ll-bar"><span class="ll-row"><span class="ll-name">{fav}{esc(group_title(members))}</span>'
            f'{badge}</span><span class="ll-row ll-mono"><span class="ll-trunc">{status}{esc(lora.stem)}</span>'
            f'<span class="ll-trunc">{second}</span></span>{vchip}{where}{extra}</span></button>'
        )
    rows = settings_rows(lora)
    pick = {"Network", "Optimizer", "Epochs", "Base checkpoint"}
    dl = "".join(f"<span class='ll-dt'>{esc(a)}</span><span class='ll-dd'>{esc(b)}</span>"
                 for a, b in rows if a in pick)
    if not dl:
        dl = "<span class='ll-dt'>Metadata</span><span class='ll-dd'>no training data in file</span>"
    if trig:
        dl += f"<span class='ll-dt'>Trigger</span><span class='ll-dd'>{esc(trig)}</span>"
    tags = "".join(f"<span class='ll-tag'>{esc(t)}</span>" for t, _ in lora.tags[:5])
    issues = "".join(f"<span class='ll-issue ll-issue-{l}'>{esc(t)}</span>" for l, t in issues_all[:2])
    return (
        f'<button class="ll-card ll-card-data ll-st-{st}" {click} title="{esc(lora.path)}">{sel}'
        f'<span class="ll-row"><span class="ll-name ll-big">{fav}{esc(group_title(members))}</span>{badge}</span>'
        f'<span class="ll-mono ll-sub">{status}{esc(lora.stem)}</span>{vchip}'
        f'<span class="ll-dl">{dl}</span>'
        f'<span class="ll-tags">{tags}</span>{issues}'
        f'{sparkline(lora.loss)}{where}{extra}'
        f'<span class="ll-foot">No preview image</span></button>'
    )


def render_missing(base: str, ds: Dataset | None) -> str:
    extra = f"Dataset ready · {ds.images} images" if ds else "No dataset found"
    return (f'<div class="ll-card ll-missing"><span class="ll-miss-title">No {esc(base)} version</span>'
            f'<span>{esc(extra)}</span></div>')


def render_dataset_only(ds: Dataset) -> str:
    dirs = ", ".join(p.name for p in ds.repeat_dirs[:3]) or ds.root.name
    cap = f"{ds.captions}/{ds.images} captioned"
    return (f'<div class="ll-card ll-untrained"><span class="ll-name ll-big">Dataset only</span>'
            f'<span class="ll-mono ll-sub">{esc(dirs)}</span><span>{ds.images} images · {cap}</span>'
            f'<span class="ll-warn-text">No trained LoRA found</span></div>')


BASE_ABBR = {"illustrious": "ILL", "pony": "PO", "anima": "ANI", "flux": "F1", "chroma": "CHR", "zimage": "ZI", "qwenimage": "QWI", "flux2": "F2", "sdxl": "XL",
             "sd15": "15", "sd3": "SD3", "lumina": "LUM", "hunyuan": "HY", "hunyuanimage": "HY", "unknown": "?"}


def base_abbr(base: str) -> str:
    """Short label for a LoRA type in the Carddex: Illustrious -> ILL, Pony -> PO, Flux -> F1, Anima -> ANI."""
    k = re.sub(r"[^a-z0-9]", "", (base or "").lower())
    return BASE_ABBR.get(k) or (base or "?")[:3].upper()


PURPOSES = [("character", "Character"), ("style", "Art style"), ("object", "Object"), ("action", "Action / pose"),
            ("concept", "Concept / effect")]
PURPOSE_COLORS = {"character": "#3B82C4", "style": "#C2410C", "object": "#15803D", "action": "#7C3AED",
                  "concept": "#B45309"}


def purpose_colors(lib) -> dict:
    """{'on': bool, purpose: colour}: Settings › Look and base models › Colour the cards by purpose."""
    d = dict(lib.data.get("purpose_colors") or {})
    out = {"on": bool(d.get("on"))}
    for k, _ in PURPOSES:
        out[k] = d.get(k) or PURPOSE_COLORS[k]
    return out


def _purpose_of(ds_name: str) -> str:
    try:
        import dsflow
        return dsflow.meta(ds_name).get("purpose") or "character"
    except Exception:                                    # noqa: BLE001
        return "character"


def render_gallery(lib: Library, search: str, base: str, sort: str, only_preview: bool, favorites: bool) -> str:
    """Picture wall of your own LoRAs: one tile per dataset (mychar), with the newest picture and a label for
    every LoRA type it has (ANI, ILL, ...). Click a tile to see that dataset in the Library, folded out."""
    q = (search or "").strip().lower()
    personal = [l for l in lib.loras.values() if l.dataset is not None]

    def mtime(l):
        try:
            return l.path.stat().st_mtime
        except OSError:
            return 0

    def matches(l):
        if base and base != "All" and l.base != base:
            return False
        if favorites and not (l.sidecar or {}).get("favorite"):
            return False
        if q and q not in " ".join([l.stem, l.display_name, l.character, l.base, l.group_name, l.dataset.name,
                                    " ".join(l.trigger_words)]).lower():
            return False
        return True

    by_ds: dict[str, list] = {}
    for l in personal:
        by_ds.setdefault(l.dataset.name.lower(), []).append(l)
    order = {b: i for i, b in enumerate(lib.base_names())}
    tiles = []
    for key, members in by_ds.items():
        hits = sorted((l for l in members if matches(l)), key=mtime, reverse=True)
        if not hits:
            continue
        shown = lib.newest(hits[0].gid)                  # the version marked best, else the newest
        top = shown if shown in hits else hits[0]
        pic_src = top if top.preview else next((l for l in hits if l.preview), None)
        if only_preview and not pic_src:
            continue
        tiles.append((key, members, top, pic_src, mtime(hits[0])))

    if sort == "Name" or sort == "Character":
        tiles.sort(key=lambda t: t[2].dataset.name.lower())
    else:
        tiles.sort(key=lambda t: t[4], reverse=True)

    pc = purpose_colors(lib)

    def tile(key, members, newest, src) -> str:
        ds = newest.dataset
        col = ""
        if pc["on"]:                                     # Settings › Look: a colour per purpose
            pu = _purpose_of(ds.name)
            c = pc.get(pu) or PURPOSE_COLORS.get(pu, "")
            lab = dict(PURPOSES).get(pu, pu)
            col = (f' style="--pc:{esc(c)}"', f'<span class="ll-pchip" title="Purpose">{esc(lab)}</span>')
        nex = sum(len(l.examples) for l in members if l is newest)
        exn = (f'<span class="ll-exn" title="Example pictures linked in LoRA Manager">{nex} example'
               f'{"s" if nex != 1 else ""}</span>') if nex else ""
        uri = thumb_data_uri(src.preview, 320) if src else ""
        blur = " ll-blur" if uri and lib.blur(src) else ""
        if uri:
            pic = f'<span class="ll-gimg{blur}"><img src="{uri}" alt="" loading="lazy"></span>'
        else:
            pic = (f'<span class="ll-gimg ll-gph"><span class="ll-gph-letter">{esc(ds.name[:1].upper())}</span>'
                   f'<span>No preview yet</span></span>')
        # one label per LoRA type, in the order of the base models in Settings; ✓ = Completed
        types: dict[str, list] = {}
        for l in members:
            types.setdefault(l.base, []).append(l)
        chips = []
        for b in sorted(types, key=lambda b: (order.get(b, 99), b)):
            ls = types[b]
            done = is_completed(ls[0])
            n_groups = len({l.gid for l in ls})
            tip = f"{b}: {len(ls)} file{'s' if len(ls) != 1 else ''}" + (" · Completed" if done else "")
            cls = "ll-tchip" + (" ll-tchip-done" if done else "") + (" ll-tchip-cur" if b == newest.base else "")
            gid_t = max(ls, key=mtime).gid
            chips.append(f'<span class="{cls}" title="{esc(tip)} · click to open this one" data-open="{gid_t}" '
                         f'onclick="event.stopPropagation(); llOpenCard(this.dataset.open, this.closest(\'.ll-gtile\'), this.closest(\'.ll-gtile\').dataset.ds)">'
                         f'{"&#10003;" if done else ""}'
                         f'{esc(base_abbr(b))}{f"<small>&times;{n_groups}</small>" if n_groups > 1 else ""}</span>')
        worst = max((l.status for l in members), key=lambda x: STATUS_ORDER[x])
        dot = "" if worst == "ok" else f'<span class="ll-dot ll-dot-{worst}" title="{esc(worst)}"></span>'
        fav = '<span class="ll-fav">&#9733;</span>' if any((l.sidecar or {}).get("favorite") for l in members) else ""
        note = f"newest: {newest.stem}" + ("" if src is None or src is newest else f" · picture from {src.stem}")
        return (f'<button class="ll-card ll-gtile{" ll-pcol" if col else ""}" '
                f'onclick="llOpenCard(this.dataset.open, this, this.dataset.ds)"{col[0] if col else ""} '
                f'data-gid="ds:{esc(key)}" data-ds="{esc(key)}" data-open="{newest.gid}" title="{esc(note)}">'
                f'<span class="ll-selmark" aria-hidden="true"></span>{pic}'
                f'<span class="ll-gcap ll-gcap2"><span class="ll-name">{fav}{dot}{esc(ds.name)}</span>'
                f'<span class="ll-tchips">{col[1] if col else ""}{"".join(chips)}{exn}</span></span></button>')

    parts = [tile(*t[:4]) for t in tiles]
    n_files = len(personal)
    if not parts:
        msg = ("No personal LoRAs found yet. Check the dataset folders in Settings." if not n_files
               else "Nothing matches these filters.")
        return f'<div class="ll-wrap"><p class="ll-muted">{msg}</p></div>'
    head = (f'<div class="ll-muted" style="margin:2px 0 10px">{len(by_ds)} datasets with LoRAs ({n_files} files) '
            f'&middot; newest picture shown (or the version marked ★ best), labels = LoRA types (&#10003; = Completed) &middot; '
            f'click a tile to open its newest LoRA, or a label to open that type</div>')
    return f'<div class="ll-wrap">{head}<div class="ll-ggrid">{"".join(parts)}</div></div>'


# --------------------------------------------------------------------------- #
# Gallery tab: pictures you made with a LoRA (all its versions together)
# --------------------------------------------------------------------------- #

PIC_PAGE = 120


def _simple(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def picture_folder(base: str) -> tuple[str, str]:
    """(folder, how it was found) for a LoRA type's pictures; folder "" = not found, ask."""
    s = LIB.settings
    for t, d in (s.get("picture_dirs") or {}).items():
        if t.lower() == base.lower():
            return d, f"the folder you set for {base}"
    root = s.get("pictures_dir") or ""
    if not root:
        return "", "No ComfyUI picture folder is set (Settings > Folders)."
    if not Path(root).is_dir():
        return "", f"The ComfyUI picture folder `{root}` does not exist."
    kws = next((k for n, k in s["bases"] if n == base), [])
    wanted = {_simple(base), _simple(base_abbr(base))} | {_simple(k) for k in kws}
    wanted.discard("")
    level = [Path(root)]
    for depth in (1, 2):
        nxt = []
        for d in level:
            try:
                subs = sorted((c for c in d.iterdir() if c.is_dir() and not c.name.startswith(".")), key=lambda c: c.name)
            except OSError:
                continue
            for c in subs:
                if _simple(c.name) in wanted:
                    return str(c), f"`{c.name}` in your ComfyUI picture folder"
            nxt += subs
        level = nxt[:400]
    return "", f"No `{base}` folder was found in `{root}`."


OLD = "old"          # marks a picture linked through an old LoRA name


def picture_aliases(gid: str) -> list[str]:
    """Old LoRA names (from before a rename) linked to this LoRA by you, saved in library_data.json."""
    return list(((LIB.data.get("picture_aliases") or {}).get(gid)) or [])


def set_picture_aliases(gid: str, names: list[str]):
    al = LIB.data.setdefault("picture_aliases", {})
    names = sorted({PG.norm(n) for n in names if str(n).strip()})
    if names:
        al[gid] = names
    else:
        al.pop(gid, None)
    LIB.save_data()


def _file_hashes(l) -> set:
    """sha256 of the LoRA's files, only where already known (LoRA Manager data or the hash cache):
    no hashing of big files just for the Gallery."""
    out = set()
    sha = str((l.sidecar or {}).get("sha256") or "").lower()
    if len(sha) >= 10:
        out.add(sha)
    for p in [l.lora_path, l.dataset_path] + [f for _, f in (l.epoch_files or [])]:
        hit = LIB.hash_cache.get(str(p)) if p else None
        if hit and hit.get("sha"):
            out.add(hit["sha"].lower())
    return out


def group_lora_names(members: list, gid: str = "") -> tuple[dict, set]:
    """name (as PG.norm makes it) -> (Lora, epoch | None | OLD) for every file of the group and the old
    names you linked; plus the known sha256 hashes."""
    names, hashes = {}, set()
    for l in members:
        for p in (l.lora_path, l.dataset_path):
            if p:
                names[PG.norm(p.name)] = (l, None)
        names.setdefault(l.stem.lower(), (l, None))
        for ep, f in l.epoch_files or []:
            names.setdefault(PG.norm(f.name), (l, ep))
        hashes |= _file_hashes(l)
    if members:
        for a in picture_aliases(gid or members[-1].gid):
            names.setdefault(a, (members[0], OLD))
    return names, hashes


def known_lora_names() -> set:
    """Every name that already belongs to some LoRA (files, epochs, linked old names)."""
    out = set()
    for gid, members in LIB.groups.items():
        out |= set(group_lora_names(members, gid)[0])
    return out


def alias_suggestions(gid: str) -> list[tuple[str, str]]:
    """Checkbox choices: names in your pictures that no LoRA file has, but that look like this LoRA
    (contain the dataset / character name), plus the old names already linked. [(label, name)]"""
    members = LIB.groups.get(gid) or []
    if not members:
        return []
    n = members[-1]
    keys = {_simple(x) for x in (n.dataset.name if n.dataset else "", n.character, n.group_name) if x}
    keys = {k for k in keys if len(k) >= 3}
    linked = set(picture_aliases(gid))
    taken = known_lora_names() - linked
    counts = PG.name_counts()
    out = []
    for name, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        if name in taken:
            continue
        if name in linked or any(k in _simple(name) for k in keys):
            out.append((f"{name}  ({c} picture{'s' if c != 1 else ''})", name))
    for a in sorted(linked - set(counts)):
        out.append((f"{a}  (no pictures yet)", a))
    return out


def group_pictures(gid: str) -> list[dict]:
    members = LIB.groups.get(gid) or []
    if not members:
        return []
    names, hashes = group_lora_names(members, gid)
    out = []
    for path, e, hit in PG.pictures_for(set(names), hashes):
        l, ep = names.get(hit.lstrip("#"), (None, None))
        if l is None:      # matched by hash only: the newest member with that hash
            l = next((m for m in members if str((m.sidecar or {}).get("sha256") or "").lower()[:10]
                      in {h[:10] for h in (e.get("hashes") or {})}), members[-1])
        out.append({"path": path, "e": e, "lora": l, "epoch": ep, "name": hit.lstrip("#"),
                    "strength": (e.get("loras") or {}).get(hit.lstrip("#"))})
    return out


def picture_versions(gid: str, pics: list | None = None) -> list:
    pics = group_pictures(gid) if pics is None else pics
    count: dict[str, int] = {}
    for p in pics:
        k = OLD if p["epoch"] == OLD else p["lora"].key
        count[k] = count.get(k, 0) + 1
    members = LIB.groups.get(gid) or []
    out = [("All versions", "")] + [(f"{m.vlabel} · {m.stem} ({count[m.key]})", m.key)
                                    for m in reversed(members) if m.key in count]
    if OLD in count:
        out.append((f"Old names ({count[OLD]})", OLD))
    return out


def _when(t) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(float(t)))
    except (TypeError, ValueError):
        return ""


def render_pictures(gid: str, version: str, page) -> tuple[str, int, int]:
    """Gallery grid HTML for one page; returns (html, page, pages)."""
    members = LIB.groups.get(gid) or []
    if not members:
        return '<div class="ll-wrap"><p class="ll-muted">Pick a LoRA above.</p></div>', 1, 1
    pics = group_pictures(gid)
    total_all = len(pics)
    if version == OLD:
        pics = [p for p in pics if p["epoch"] == OLD]
    elif version:
        pics = [p for p in pics if p["lora"].key == version and p["epoch"] != OLD]
    pages = max(1, -(-len(pics) // PIC_PAGE))
    page = min(max(1, int(page or 1)), pages)
    shown = pics[(page - 1) * PIC_PAGE: page * PIC_PAGE]
    if not shown:
        msg = ("No pictures found for this LoRA yet. Press <b>Scan for new pictures</b> to look in its picture folder."
               if not total_all else "No pictures of this version.")
        return f'<div class="ll-wrap"><p class="ll-muted">{msg}</p></div>', page, pages
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=6) as ex:        # first time only: thumbnails are kept in .thumbs
        uris = list(ex.map(lambda p: thumb_data_uri(Path(p["path"]), 320), shown))
    blur = " ll-blur" if LIB.blur(members[-1]) else ""
    tiles = []
    for p, uri in zip(shown, uris):
        l = p["lora"]
        st = p["strength"]
        if p["epoch"] == OLD:
            cap = "old name"
        else:
            cap = l.vlabel + (f" · ep {p['epoch']}" if p["epoch"] is not None else "")
        cap += f" · {st:g}" if isinstance(st, (int, float)) else ""
        img = f'<img src="{uri}" alt="" loading="lazy">' if uri else '<span class="ll-muted">no preview</span>'
        tip = Path(p["path"]).name + (f" · made with {p['name']}" if p["epoch"] == OLD else "")
        tiles.append(f'<button class="ll-ptile" data-p="{esc(p["path"])}" onclick="llPicOpen(this.dataset.p, this)" '
                     f'title="{esc(tip)}"><span class="ll-pimg{blur}">{img}</span>'
                     f'<span class="ll-pcap"><b>{esc(cap)}</b><span>{esc(_when(p["e"].get("t")))}</span></span></button>')
    head = (f'<div class="ll-muted" style="margin:2px 0 10px">{len(pics)} picture{"s" if len(pics) != 1 else ""}'
            f'{f" of {total_all}" if version else ""} made with <b>{esc(group_title(members))}</b> '
            f'({members[-1].base}, {len(members)} version{"s" if len(members) != 1 else ""}) &middot; newest first'
            f'{f" &middot; page {page} of {pages}" if pages > 1 else ""} &middot; click a picture to see it with its prompt</div>')
    return f'<div class="ll-wrap">{head}<div class="ll-pgrid">{"".join(tiles)}</div></div>', page, pages


def picture_details(path: str, gid: str):
    """(image, info markdown, prompt) for the viewer next to the grid."""
    e = PG.entry(path)
    if not e or not Path(path).exists():
        return None, "That picture is no longer there: press *Scan for new pictures*.", ""
    members = LIB.groups.get(gid) or []
    names, _ = group_lora_names(members, gid) if members else ({}, set())
    lines = [f"**{Path(path).name}**  ·  {_when(e.get('t'))}", f"`{Path(path).parent}`", ""]
    used = []
    for n, st in (e.get("loras") or {}).items():
        mine = names.get(n)
        if mine and mine[1] == OLD:
            label = f"**{n}** (old name of this LoRA)"
        elif mine:
            label = f"**{mine[0].stem}** ({mine[0].vlabel}{f', epoch {mine[1]}' if mine[1] is not None else ''})"
        else:
            label = n
        used.append(f"{label}" + (f" at {st:g}" if isinstance(st, (int, float)) else ""))
    if used:
        lines.append("LoRAs: " + ", ".join(used))
    if e.get("ckpt"):
        lines.append(f"Model: `{e['ckpt']}`")
    if e.get("seed") is not None:
        lines.append(f"Seed: `{e['seed']}`")
    lines.append(f"Read from: {'the .json next to it' if e.get('src') == 'json' else 'the PNG metadata'}")
    return load_image(Path(path), 2048), "  \n".join(lines), e.get("prompt") or ""


def epochs_line(l: Lora) -> str:
    if not l.epoch_files:
        return ""
    eps = ", ".join(str(e) for e, _ in l.epoch_files)
    return f'<span class="ll-mono ll-sub">epoch files: {esc(eps)}{"" if l.final else " (no final file)"}</span>'


def library_entry(lib: Library, l: Lora) -> str:
    """A Library card for a version group, showing only the newest version; click opens its Details.
    With several versions, the 'V1–V3 ▾' chip folds out a row with every version (newest first)."""
    members = lib.group_of(l)
    if len(members) < 2:
        return render_card(lib, l, members, extra=epochs_line(l), selectable=True)
    head = render_card(lib, l, members, extra=epochs_line(l), selectable=True)   # click = Details of the newest
    versions = "".join(
        render_card(lib, m, [m], extra=f'<span class="ll-vlabel">{esc(m.vlabel)}'
                    f'{" &middot; newest" if m is members[-1] else ""}'
                    f'{" &middot; &#9733; best" if m is best_member(members) else ""}</span>{epochs_line(m)}')
        for m in reversed(members))
    fold = (f'<div class="ll-vers" data-gid="{l.gid}"><div class="ll-vers-head"><b>{esc(group_title(members))}</b>'
            f'<span class="ll-muted">{len(members)} versions, newest first &middot; click a version to open its details</span></div>'
            f'<div class="ll-grid">{versions}</div></div>')
    return head + fold


def render_library(lib: Library, search: str, bases: list, statuses: list, group: str) -> str:
    search = (search or "").strip().lower()
    bases = set(bases or [])
    statuses = set(statuses or [])
    status_map = {"OK": "ok", "Warnings": "warning", "Problems": "problem"}
    want_status = {status_map[s] for s in statuses if s in status_map}
    show_untrained = "Untrained datasets" in statuses
    show_missing = "Missing bases" in statuses
    show_foreign = False          # only your own LoRAs (the LoRA folder is not scanned)
    offered = {x.base for x in lib.loras.values() if x.dataset is not None} | {"Unknown"}

    def match(l: Lora) -> bool:
        if l.dataset is None and not show_foreign:
            return False      # not made by you: no dataset in your dataset folders
        if bases and l.base in offered and l.base not in bases:
            return False      # bases that are not offered as a filter (only used by downloads) always pass
        if group_status(lib.group_of(l))[0] not in want_status:
            return False
        if search:
            hay = " ".join([l.stem, l.display_name, l.character, l.base, " ".join(l.trigger_words),
                            " ".join(t for t, _ in l.tags)]).lower()
            return search in hay
        return True

    # one card per version group: a group is shown when any of its versions matches, as its newest version
    shown = {}
    for l in lib.loras.values():
        if match(l):
            shown[l.gid] = lib.newest(l.gid) or l
    loras = list(shown.values())
    order = {b: i for i, b in enumerate(lib.base_names())}
    sections: dict[str, list[str]] = {}
    trained = {l.character.lower() for l in lib.loras.values()}

    if group == "Base model":
        for l in sorted(loras, key=lambda l: (order.get(l.base, 99), l.character.lower(), l.stem.lower())):
            sections.setdefault(l.base, []).append(library_entry(lib, l))
    else:
        by_char: dict[str, list[Lora]] = {}
        for l in loras:
            by_char.setdefault(l.character.lower(), []).append(l)
        names = {k: v[0].character for k, v in by_char.items()}
        if show_untrained or show_missing:
            for k, ds in lib.datasets.items():
                if not search or search in ds.name.lower():
                    names.setdefault(k, ds.name)
        for k in sorted(names):
            cards = [library_entry(lib, l) for l in sorted(
                by_char.get(k, []), key=lambda l: (order.get(l.base, 99), l.stem.lower()))]
            ds = lib.datasets.get(k)
            if k not in trained:
                if show_untrained and ds:
                    cards.append(render_dataset_only(ds))
            elif show_missing and ds:
                have = {l.base for l in lib.loras.values() if l.character.lower() == k}
                for b in lib.settings["expected_bases"]:
                    if b not in have and (not bases or b in bases):
                        cards.append(render_missing(b, ds))
            if cards:
                sections[names[k]] = cards

    gst = [group_status(m)[0] for m in lib.groups.values() if m and (m[-1].dataset or show_foreign)]
    n_prob, n_warn = gst.count("problem"), gst.count("warning")
    head = (f'<div class="ll-stats"><span><b>{len(loras)}</b> LoRAs shown</span>'
            f'<span><b>{len(lib.datasets)}</b> datasets</span>'
            f'<span class="ll-c-warn"><b>{n_warn}</b> warnings</span>'
            f'<span class="ll-c-prob"><b>{n_prob}</b> problems</span>'
            f'<span class="ll-muted">scanned {esc(lib.scanned_at)}</span></div>')
    if not sections:
        empty = ("No LoRAs found. Check the folders in Settings › Folders."
                 if not lib.loras else "Nothing matches these filters.")
        return f'<div class="ll-wrap">{head}<p class="ll-muted">{empty}</p></div>'
    body = "".join(f'<h3 class="ll-h">{esc(name)}</h3><div class="ll-grid">{"".join(cards)}</div>'
                   for name, cards in sections.items())
    return f'<div class="ll-wrap">{head}{body}</div>'


# --------------------------------------------------------------------------- #
# Details
# --------------------------------------------------------------------------- #

def _card_script(lora) -> str:
    """The launch script that made this version (from its training card), as a file name."""
    import trainingcard as TC
    sp = TC.stored(lora.sidecar).get("_script")
    return Path(sp).name if sp else ""


def files_check_html(lib: Library, lora: Lora) -> str:
    ds = lora.dataset
    has_meta = any(k.startswith("ss_") for k in lora.meta)
    logging_in_script = any(lg for _, lg in lora.scripts)
    rows = [
        ("Training metadata", has_meta, "in .safetensors" if has_meta else "none in file"),
        ("Dataset folder", bool(ds), ds.root.name if ds else "not found"),
        ("Captions (.txt)", bool(ds) and ds.captions == ds.images,
         f"{ds.captions}/{ds.images}" if ds else "-"),
        ("Model file", lora.dataset_path is not None,
         lora.dataset_path.parent.name + "/" + lora.dataset_path.name if lora.dataset_path else "not found"),
        ("Launch script", bool(lora.scripts) or bool(_card_script(lora)),
         _card_script(lora) or ", ".join(p.name for p, _ in lora.scripts[:2]) or "not found"),
        ("TensorBoard log", bool(lora.loss),
         Path(lora.log_run).name if lora.log_run else
         ("script has --logging_dir, no log matched" if logging_in_script else "not found - add --logging_dir")),
        ("LoRA Manager metadata", lora.sidecar is not None,
         f"{ARCHIVE_DIR}/{lora.stem}.metadata.json" if lora.sidecar else "missing - run Personal sync"),
        ("Preview image", lora.preview is not None, lora.preview.name if lora.preview else "missing - run Personal sync"),
    ]
    li = "".join(f"<li><span>{esc(a)}</span><span class='{'ll-ok' if ok else 'll-c-warn'}'>{esc(t)}</span></li>"
                 for a, ok, t in rows)
    return f"<ul class='ll-check'>{li}</ul>"


def details_html(lib: Library, lora: Lora) -> tuple[str, str]:
    issues = "".join(f"<div class='ll-issue ll-issue-{l}'>{esc(t)}</div>" for l, t in lora.issues)
    if getattr(lora, "accepted", None):
        issues += (f"<div class='ll-muted' style='margin-top:6px'>Accepted by you: {esc('; '.join(lora.accepted))} "
                   "(no mark in the Collection)</div>")
    epochs = ""
    if lora.epoch_files:
        epochs = ("<div class='ll-muted' style='margin-top:8px'>Epoch files: "
                  + ", ".join(str(e) for e, _ in lora.epoch_files) + "</div>")
    used = ""
    try:
        import insights
        u = insights.usage_of(lora)
        if u and u["pictures"]:
            used = (f"<div class='ll-muted' style='margin-top:6px'>Used in <b>{u['pictures']}</b> of your pictures"
                    + (f", usually at strength <b>{u['strength']:.2f}</b>" if u.get("strength") is not None else "")
                    + (f", last on {esc(u['last'])}" if u.get("last") else "") + "</div>")
    except Exception:                            # noqa: BLE001
        pass
    head = (f"<div class='ll-dhead'><div class='ll-muted ll-upper'>{esc(lora.base)} · {esc(lora.character)}</div>"
            f"<h2>{esc(lora.display_name)}</h2><div class='ll-mono ll-muted'>{esc(lora.path)}</div>"
            f"{issues}{epochs}{used}</div>")
    import trainingcard as TC
    card = TC.stored(lora.sidecar)
    if card:                                             # what you set (Edit, or synced from LoRA Manager) is shown
        if lora.trigger_words:
            card["trigger"] = ", ".join(str(w) for w in lora.trigger_words)
        vname = ((lora.sidecar or {}).get("civitai") or {}).get("name")
        if vname:
            card["version"] = vname
        cs = TC.usage_tips(lora.sidecar).get("clip_skip")
        if cs not in (None, "") and not card.get("clip_skip"):
            card["clip_skip"] = f"{cs} (usage tip)"
        settings = "<h4>Training</h4>" + TC.card_html(card, esc)
    else:
        rows = settings_rows(lora)                       # a LoRA without a dataset: what its file says
        if rows:
            dl = "".join(f"<dt>{esc(a)}</dt><dd>{esc(b)}</dd>" for a, b in rows)
            settings = f"<h4>Training settings <span class='ll-muted'>(from .safetensors)</span></h4><dl class='ll-ddl'>{dl}</dl>"
        else:
            settings = "<h4>Training settings</h4><p class='ll-muted'>No training data found for this LoRA.</p>"
    tags = "".join(f"<span class='ll-tag'>{esc(t)} {p}%</span>" for t, p in lora.tags)
    if tags:
        settings += f"<h4>Caption tags</h4><div class='ll-tags'>{tags}</div>"
    settings += f"<h4>Files check</h4>{files_check_html(lib, lora)}"
    return head, settings


def loss_frame(lora: Lora) -> pd.DataFrame:
    pts = [(s, v) for s, v in lora.loss if not (math.isnan(v) or math.isinf(v))]
    if len(pts) > 2000:
        pts = pts[:: len(pts) // 2000 + 1]
    return pd.DataFrame(pts, columns=["step", "loss"])


def usage_tips(lora: Lora) -> dict:
    raw = (lora.sidecar or {}).get("usage_tips") or "{}"
    try:
        tips = json.loads(raw) if isinstance(raw, str) else dict(raw)
        return tips if isinstance(tips, dict) else {}
    except Exception:
        return {}


# --------------------------------------------------------------------------- #
# Editing (writes the LoRA Manager sidecar)
# --------------------------------------------------------------------------- #

def new_sidecar(lib: Library, lora: Lora) -> dict:
    st = lora.path.stat()
    return {
        "file_name": lora.stem,
        "model_name": lora.stem,
        "file_path": str(lora.path),
        "size": st.st_size,
        "modified": st.st_mtime,
        "sha256": file_sha256(lora.path, lib.hash_cache),
        "base_model": lora.base,
        "preview_url": str(lora.preview) if lora.preview else "",
        "preview_nsfw_level": 0,
        "notes": "",
        "from_civitai": False,
        "civitai": {"trainedWords": []},
        "tags": [],
        "modelDescription": "",
        "favorite": False,
        "exclude": False,
        "db_checked": False,
        "skip_metadata_refresh": False,
        "last_checked_at": 0,
        "hash_status": "completed",
        "usage_tips": "{}",
    }


def write_sidecar(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.copy2(path, path.with_name(path.name + ".bak"))
    _save_json(path, data)


# --------------------------------------------------------------------------- #
# Personal sync: LoRA folder -> <dataset>/LoraManager/   (the only time the LoRA folder is read)
# --------------------------------------------------------------------------- #

def _renamed(name: str, old_stem: str, new_stem: str) -> str:
    """'MyChar_Illustrious.preview.png' -> 'mycharILLV1.preview.png' (files are stored under the dataset file's name)."""
    if name.lower().startswith(old_stem.lower()):
        return new_stem + name[len(old_stem):]
    return f"{new_stem}.{name}"


def _sync_one(src: Path, dst: Path, stats: dict) -> None:
    """Copy when missing, or when the LoRA folder's file is newer (an edit made in the dataset copy is kept)."""
    try:
        s = src.stat()
        if dst.exists():
            d = dst.stat()
            if s.st_mtime <= d.st_mtime + 1:
                stats["same"] += 1
                return
            if dst.suffix == ".json":
                shutil.copy2(dst, dst.with_name(dst.name + ".bak"))
            stats["updated"] += 1
        else:
            stats["new"] += 1
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    except OSError as e:
        stats["errors"].append(f"{src.name}: {e}")


# fields that describe where a file lies: each side keeps its own
_META_OWN = {"file_name", "file_path", "folder", "preview_url", "modified"}


def _empty(v) -> bool:
    """Missing for the merge: None, "", "{}", [], {}, 0 (booleans are values: False is kept)."""
    if isinstance(v, bool):
        return False
    if v is None or v == 0 or v == "" or v == [] or v == {}:
        return True
    return isinstance(v, str) and v.strip() in ("", "{}", "[]")


def merge_meta(a: dict, b: dict, a_wins: bool) -> dict:
    """Both .metadata.json files together: a field that is missing or empty on one side is taken from the other;
    when both have a different value, the newer file (a_wins) decides. Nested parts (civitai …) merge the same way."""
    out = {}
    for k in list(a) + [k for k in b if k not in a]:
        if k not in b:
            out[k] = a[k]
        elif k not in a:
            out[k] = b[k]
        elif isinstance(a[k], dict) and isinstance(b[k], dict):
            out[k] = merge_meta(a[k], b[k], a_wins)
        elif _empty(a[k]):
            out[k] = b[k]
        elif _empty(b[k]):
            out[k] = a[k]
        else:
            out[k] = a[k] if a_wins else b[k]
    return out


def _tips(sc: dict) -> dict:
    raw = (sc or {}).get("usage_tips") or "{}"
    try:
        t = json.loads(raw) if isinstance(raw, str) else dict(raw)
        return t if isinstance(t, dict) else {}
    except (ValueError, TypeError):
        return {}


def sync_rules(lm: dict, ls: dict, m: dict, lm_newer: bool = False) -> dict:
    """What LoRA Studio knows best wins over the merge: the version name (from the training card or your Edit), and
    the trigger words of the dataset when neither side has any. Usage tips (strength, clip skip, …) are merged field
    by field; no clip skip anywhere means the default, 1."""
    m = dict(m)
    civ = dict(m.get("civitai") or {})
    ls_civ = (ls.get("civitai") or {}) if isinstance(ls.get("civitai"), dict) else {}
    card = ((ls.get("lorastudio") or {}).get("training") or {}) if isinstance(ls.get("lorastudio"), dict) else {}
    ver = ls_civ.get("name") or card.get("version")
    if ver:
        civ["name"] = ver
    if not civ.get("trainedWords") and card.get("trigger"):
        civ["trainedWords"] = [w.strip() for w in str(card["trigger"]).split(",") if w.strip()]
    if civ or "civitai" in m:
        m["civitai"] = civ
    t_lm, t_ls = _tips(lm), _tips(ls)
    tips = merge_meta(t_lm, t_ls, lm_newer)            # tips: fill what one side lacks; the newer file on a clash
    try:
        cs = int(float(tips.get("clip_skip")))
    except (TypeError, ValueError):
        cs = 0
    tips["clip_skip"] = cs if cs >= 1 else int(float(card.get("clip_skip") or 1))
    m["usage_tips"] = json.dumps(tips)
    return m


def _sync_meta(src: Path, dst: Path, stats: dict) -> None:
    """Sync both ways: data only one .metadata.json has is copied into the other (the LoRA folder's file and the
    dataset copy); for fields both have, the newer file wins. Location fields (file path, folder, preview) stay as
    they are on each side. A .bak is made before a file is changed; files that already agree are not touched."""
    if not dst.exists():
        _sync_one(src, dst, stats)
        return
    a, b = _load_json(src, None), _load_json(dst, None)
    if not isinstance(a, dict) or not isinstance(b, dict):
        _sync_one(src, dst, stats)                       # one of them is not readable JSON: the old rule
        return
    try:
        src_newer = src.stat().st_mtime > dst.stat().st_mtime + 1
    except OSError as e:
        stats["errors"].append(f"{src.name}: {e}")
        return
    m = sync_rules(a, b, merge_meta(a, b, src_newer), src_newer)
    changed = False
    for path, cur in ((src, a), (dst, b)):
        new = dict(m)
        for k in _META_OWN:                              # this side's own location fields
            if k in cur:
                new[k] = cur[k]
            else:
                new.pop(k, None)
        if path == src and "lorastudio" not in cur:      # the training card stays in LoRA Studio's copy
            new.pop("lorastudio", None)
        if new != cur:
            try:
                write_sidecar(path, new)
                changed = True
                stats["merged"] = stats.get("merged", 0) + 1
                if path == src:
                    stats["back"] = stats.get("back", 0) + 1
            except OSError as e:
                stats["errors"].append(f"{path.name}: {e}")
    if changed:
        stats["updated"] += 1
    else:
        stats["same"] += 1


# ----------------------------------------------------------------------------------- housekeeping (Clean up tab)
def _trainer():
    """sdgui when it runs in the same program (LoRA Studio): for the active run and the musubi-tuner folder."""
    return sys.modules.get("sdgui")


def busy_datasets() -> list[Path]:
    sg = _trainer()
    try:
        if sg is not None and sg.RUN.running() and sg.RUN.script:
            return [sg.script_home(sg.RUN.script)]
    except Exception:                  # noqa: BLE001
        pass
    return []


def musubi_cache_dir() -> Path | None:
    sg = _trainer()
    try:
        return (sg.musubi_root() / "cache") if sg is not None else None
    except Exception:                  # noqa: BLE001
        return None


def dataset_roots(lib: Library) -> list[Path]:
    lib.ensure_loaded()
    return sorted({d.root for d in lib.datasets.values()})


def leftovers_md(lib: Library):
    """(markdown, [paths]) of the training leftovers of finished LoRAs in every dataset. Nothing is touched."""
    if HK is None:
        return "housekeeping.py is missing next to lora_library.py.", []
    plan = HK.leftovers(dataset_roots(lib), musubi_cache_dir(), busy_datasets())
    items = plan["items"]
    L = []
    if items:
        total = sum(i[3] for i in items)
        L.append(f"**{len(items)} items, {HK.human(total)}** can go to the Trash:\n")
        L += ["| Dataset | Epoch files | States | .npz caches | Other | Size |", "|---|---|---|---|---|---|"]
        per = {}
        for p, kind, ds, size in items:
            r = per.setdefault(ds, {"epoch checkpoint": 0, "saved state": 0, "latent cache (.npz)": 0, "other": 0, "size": 0})
            r[kind if kind in r else "other"] += 1
            r["size"] += size
        for ds, r in sorted(per.items(), key=lambda kv: -kv[1]["size"]):
            L.append(f"| {html.escape(ds)} | {r['epoch checkpoint']} | {r['saved state']} | {r['latent cache (.npz)']} | "
                     f"{r['other']} | {HK.human(r['size'])} |")
        L.append("\nKept: pictures, captions, launch scripts, logs, `LoraManager/` and every **final** LoRA file. "
                 "Epoch files and states are only taken for LoRAs that have a final file; a later run re-creates the caches.")
    else:
        L.append("Nothing to clean up: no leftovers of finished LoRAs found.")
    if plan["skipped"]:
        L.append("\n**Skipped:** " + "; ".join(f"{html.escape(d)} ({why})" for d, why in plan["skipped"]))
    return "\n".join(L), [str(i[0]) for i in items]


def orphans_list(lib: Library):
    """(markdown, checkbox choices, all values) for LoRA Manager meta files whose LoRA no longer exists."""
    if HK is None:
        return "housekeeping.py is missing next to lora_library.py.", [], []
    found = HK.orphans(lib.settings.get("lora_dirs") or [], dataset_roots(lib))
    if not found:
        return "No orphaned meta files: every `.metadata.json` / `.civitai.info` has its LoRA file.", [], []
    choices = []
    for o in found:
        label = (f"[{o['where']}] {o['folder']} / {o['stem']} — {len(o['files'])} file(s), {HK.human(o['size'])}")
        choices.append((label, "\x1f".join(str(f) for f in o["files"])))
    md = (f"**{len(found)} meta file group(s)** whose LoRA file is gone ({HK.human(sum(o['size'] for o in found))}). "
          "Untick what you want to keep, tick the confirmation and press **Move ticked to Trash**. In a dataset's "
          "`LoraManager/`, a group only counts as orphaned when the LoRA is in neither the dataset's `model/` folder "
          "nor a LoRA folder (your Details edits live there).")
    return md, choices, [c[1] for c in choices]


def personal_sync(lib: Library, progress=None) -> str:
    """Read the LoRA folder(s) once and copy the LoRA Manager files of your own LoRAs (.metadata.json, preview
    and example images) into each dataset's LoraManager/ folder, named after the dataset's model file.
    A LoRA in the LoRA folder belongs to a dataset when its name and size match a model file in the dataset,
    or (for renamed copies) its sha256 does. Everything else in the LoRA folder (downloads) is left alone."""
    lib.ensure_loaded()
    by_name, by_size = {}, {}
    for l in lib.loras.values():
        p = l.dataset_path
        if l.dataset is None or p is None:
            continue
        try:
            size = p.stat().st_size
        except OSError:
            continue
        by_name[(p.stem.lower(), size)] = l
        by_size.setdefault(size, []).append(l)

    _LISTINGS.clear()
    ds_roots = {str(ds.root.resolve()) for ds in lib.datasets.values()}
    candidates = []
    for root in lib.settings["lora_dirs"]:
        for d, dns, fns in walk(root):
            if str(d.resolve()) in ds_roots:
                dns[:] = []
                continue
            candidates += [d / f for f in fns if f.endswith(".safetensors")]

    stats = {"new": 0, "updated": 0, "same": 0, "errors": []}
    matched, done_loras = 0, []
    total = max(1, len(candidates))
    for n, f in enumerate(candidates):
        if progress is not None:
            progress((n + 1) / total, desc=f"Checking {f.name}")
        try:
            size = f.stat().st_size
        except OSError:
            continue
        lora = by_name.get((f.stem.lower(), size))
        if lora is None:
            for cand in by_size.get(size, []):
                if file_sha256(cand.dataset_path, lib.hash_cache) == file_sha256(f, lib.hash_cache):
                    lora = cand
                    break
        if lora is None:
            continue
        matched += 1
        src_dir, old, new = f.parent, f.stem, lora.stem
        dst_dir = lora.dataset.archive_dir
        before = stats["new"] + stats["updated"]
        sc_path = f.with_name(old + ".metadata.json")
        sidecar = _load_json(sc_path, None) if sc_path.exists() else None
        ls_path = dst_dir / (new + ".metadata.json")
        if sc_path.exists():
            _sync_meta(sc_path, ls_path, stats)
            try:                        # the training card into LoRA Studio's copy, then what it adds to LoRA Manager
                import trainingcard as TC
                sc_new = _load_json(ls_path, None)
                lora.sidecar, lora.sidecar_path = (sc_new if isinstance(sc_new, dict) else None), ls_path
                if TC.scan_one(lib, lora) == "scanned":
                    _sync_meta(sc_path, ls_path, {"new": 0, "updated": 0, "same": 0, "errors": stats["errors"]})
            except Exception as e:      # noqa: BLE001
                stats["errors"].append(f"{new} training card: {e}")
        preview = find_preview(src_dir, old, sidecar if isinstance(sidecar, dict) else None)
        if preview:                     # the preview LoRA Manager's card shows: the LoRA page shows the same one
            target = dst_dir / f"{new}.preview{preview.suffix.lower()}"
            for ext in PREVIEW_EXT:
                other = dst_dir / f"{new}.preview{ext}"
                if other != target and other.exists() and preview.stat().st_mtime > other.stat().st_mtime:
                    try:
                        other.unlink()
                    except OSError:
                        pass
            _sync_one(preview, target, stats)
        sha = (((sidecar or {}) if isinstance(sidecar, dict) else {}).get("sha256") or "").lower()
        exs = [str(x) for x in find_examples(src_dir, old, sha, preview, list(lib.settings["lora_dirs"]) + lm_example_roots(lib))]
        if exs or ls_path.exists():                     # LoRA Manager's example pictures: linked, not copied
            cur = _load_json(ls_path, None) if ls_path.exists() else None
            cur = cur if isinstance(cur, dict) else {"file_name": new}
            lsd = dict(cur.get("lorastudio") or {})
            if (lsd.get("examples") or []) != exs:
                lsd["examples"] = exs
                cur["lorastudio"] = lsd
                try:
                    write_sidecar(ls_path, cur)
                    stats["linked"] = stats.get("linked", 0) + len(exs)
                    stats["updated"] += 1
                except OSError as e:
                    stats["errors"].append(f"{ls_path.name}: {e}")
        if stats["new"] + stats["updated"] > before:
            done_loras.append(f"{lora.dataset.name} · {new}")
    _LISTINGS.clear()
    lib.save_hash_cache()
    lib.scan()          # pick up the copied files

    lines = [f"Checked **{len(candidates)}** LoRA files in the LoRA folder; **{matched}** belong to your datasets.",
             f"Copied **{stats['new']}** new and **{stats['updated']}** updated files; "
             f"{stats['same']} were already up to date."]
    if done_loras:
        shown = ", ".join(done_loras[:30]) + (f" and {len(done_loras) - 30} more" if len(done_loras) > 30 else "")
        lines.append(f"Updated: {shown}.")
    if stats.get("linked"):
        lines.append(f"Linked **{stats['linked']}** example pictures from LoRA Manager (shown on the LoRA page; not "
                     "copied).")
    if stats.get("back"):
        lines.append(f"**{stats['back']}** `.metadata.json` file(s) in the LoRA folder got data that only the dataset "
                     "copy had (a `.bak` was made next to each).")
    if stats["errors"]:
        lines.append("Could not copy: " + "; ".join(stats["errors"][:10]))
    return "\n\n".join(lines)


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #

CSS = """
/* Normal (light) palette; the dark one below is used when Gradio runs in dark mode */
:root { --ll-card:#FDFDFB; --ll-line:#C9CDD3; --ll-line2:#BCC2CA; --ll-text:#1C1F24; --ll-muted:#5A626C;
        --ll-dim:#7D858F; --ll-ok:#21857C; --ll-warn:#A86A0C; --ll-prob:#C0392B; --ll-chip:#EDEFF2;
        --ll-chip-text:#3A4048; --ll-imgbg:#E6E9ED; --ll-dash:#B5BBC3; --ll-warn-bg:#FBF1DF; --ll-warn-fg:#7A4E05;
        --ll-prob-bg:#FBE6E3; --ll-prob-fg:#8E2A1F; --ll-focus:#E8A33D88; }
.dark { --ll-card:#1B1E23; --ll-line:#2A2E36; --ll-line2:#343944; --ll-text:#E6E8EB; --ll-muted:#9AA1AB;
        --ll-dim:#6B7280; --ll-ok:#4FB3A9; --ll-warn:#E8A33D; --ll-prob:#F07C6A; --ll-chip:#262A31;
        --ll-chip-text:#C3C8CF; --ll-imgbg:#0E0F12; --ll-dash:#3A404B; --ll-warn-bg:#2B2417; --ll-warn-fg:#F4C06F;
        --ll-prob-bg:#2E1D1B; --ll-prob-fg:#F2A497; --ll-focus:#E8A33D55; }
.dark .gradio-container, .dark body { background:#14161A; }
#pick_box { display:none !important; }
#pic_box { display:none !important; }
.ll-pgrid { display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr)); gap:10px; }
.ll-ptile { all:unset; cursor:pointer; display:flex; flex-direction:column; border-radius:10px; overflow:hidden;
  background:var(--block-background-fill); border:1px solid var(--border-color-primary); }
.ll-ptile:hover { border-color:var(--color-accent); }
.ll-ptile.ll-psel { outline:3px solid var(--color-accent); outline-offset:-3px; }
.ll-pimg { display:block; aspect-ratio:3/4; overflow:hidden; background:var(--background-fill-secondary); }
.ll-pimg img { width:100%; height:100%; object-fit:cover; display:block; }
.ll-pcap { display:flex; justify-content:space-between; gap:6px; padding:5px 8px; font-size:12px; }
.ll-pcap span { opacity:.65; }
#pg_view { position:sticky; top:8px; align-self:flex-start; }
#ls_top h1 { margin:0; font-size:22px; }
.ll-wrap { color:var(--ll-text); }
.ll-stats { display:flex; flex-wrap:wrap; gap:18px; font-size:14px; color:var(--ll-muted); margin:4px 0 8px; }
.ll-stats b { color:var(--ll-text); font-family:'JetBrains Mono',monospace; }
.ll-c-warn, .ll-c-warn b { color:var(--ll-warn) !important; }
.ll-c-prob, .ll-c-prob b { color:var(--ll-prob) !important; }
.ll-ok { color:var(--ll-ok); }
.ll-muted { color:var(--ll-muted); }
.ll-upper { text-transform:uppercase; letter-spacing:.06em; font-size:12px; }
.ll-h { margin:22px 0 10px; font-size:15px; font-weight:600; color:var(--ll-muted); }
.ll-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(240px,1fr)); gap:16px; }
.ll-card { all:unset; box-sizing:border-box; display:flex; flex-direction:column; background:var(--ll-card);
           border:1px solid var(--ll-line); border-radius:10px; overflow:hidden; cursor:pointer; color:var(--ll-text);
           font-family:inherit; text-align:left; min-width:0; }
.ll-card:hover, .ll-card:focus-visible { border-color:var(--ll-line2); outline:2px solid var(--ll-focus); }
.ll-st-problem { border-color:var(--ll-prob); }
.ll-img { display:block; aspect-ratio:3/4; overflow:hidden; background:var(--ll-imgbg); }
.ll-img img { width:100%; height:100%; object-fit:cover; display:block; transition:filter .2s; }
.ll-blur img { filter:blur(22px) brightness(.8); }
.ll-blur:hover img { filter:none; }
.ll-bar { display:flex; flex-direction:column; gap:5px; padding:9px 12px 10px; }
.ll-row { display:flex; justify-content:space-between; align-items:center; gap:8px; min-width:0; }
.ll-name { font-weight:600; font-size:14px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; min-width:0; }
.ll-big { font-size:16px; }
.ll-mono { font-family:'JetBrains Mono',monospace; font-size:11px; color:var(--ll-muted); }
.ll-trunc { white-space:nowrap; overflow:hidden; text-overflow:ellipsis; min-width:0; }
.ll-gtile.ll-pcol { border:2px solid var(--pc) !important; }
.ll-gtile.ll-pcol .ll-tchip-cur { background:var(--pc) !important; color:#fff !important; border-color:var(--pc) !important; }
.ll-pchip { font-size:10px; padding:1px 7px; border-radius:99px; background:var(--pc); color:#fff; font-weight:600; }
.ll-exn { font-size:10px; color:var(--ll-muted); margin-left:4px; }
.ll-badge { flex:none; font-family:'JetBrains Mono',monospace; font-size:11px; padding:2px 8px; border-radius:99px;
            background:var(--ll-chip); color:var(--ll-chip-text); }
.ll-note { font-size:11px; color:var(--ll-warn); }
.ll-dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; vertical-align:middle; }
.ll-dot-ok { background:var(--ll-ok); } .ll-dot-warning { background:var(--ll-warn); } .ll-dot-problem { background:var(--ll-prob); }
.ll-fav { color:var(--ll-warn); margin-right:5px; }
.ll-card-data { padding:14px 16px; gap:10px; min-height:300px; }
.ll-sub { margin-top:-4px; }
.ll-dl { display:grid; grid-template-columns:auto 1fr; gap:5px 12px; font-size:12px; }
.ll-dt { color:var(--ll-muted); } .ll-dd { font-family:'JetBrains Mono',monospace; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.ll-tags { display:flex; flex-wrap:wrap; gap:6px; }
.ll-tag { font-family:'JetBrains Mono',monospace; font-size:11px; padding:3px 8px; background:var(--ll-chip);
          color:var(--ll-chip-text); border-radius:99px; }
.ll-spark { width:100%; height:44px; margin-top:auto; }
.ll-foot { font-size:11px; color:var(--ll-dim); border-top:1px solid var(--ll-line); padding-top:8px; }
.ll-issue { font-size:12px; padding:6px 10px; border-radius:6px; margin-top:6px; display:block; }
.ll-issue-warning { background:var(--ll-warn-bg); color:var(--ll-warn-fg); }
.ll-issue-problem { background:var(--ll-prob-bg); color:var(--ll-prob-fg); }
.ll-missing { cursor:default; background:transparent; border:1px dashed var(--ll-dash); align-items:center; justify-content:center;
              gap:6px; padding:24px; min-height:200px; color:var(--ll-dim); font-size:13px; text-align:center; }
.ll-miss-title { font-size:15px; color:var(--ll-muted); }
.ll-untrained { cursor:default; border:1px dashed var(--ll-warn); padding:16px; gap:8px; font-size:13px; min-height:140px; }
.ll-warn-text { color:var(--ll-warn); }
.ll-dhead h2 { margin:4px 0 4px; font-size:26px; }
.ll-ddl { display:grid; grid-template-columns:auto 1fr; gap:6px 16px; margin:0; font-size:13px; }
.ll-ddl dt { color:var(--ll-muted); } .ll-ddl dd { margin:0; font-family:'JetBrains Mono',monospace; }
.ll-tgrid { display:grid; grid-template-columns:repeat(auto-fit,minmax(420px,1fr)); gap:10px; }
.ll-tcard { border:1px solid var(--ll-line); border-radius:10px; padding:10px 12px; background:var(--ll-card); }
.ll-tcard .rs-label { font-size:11px; letter-spacing:.06em; color:var(--ll-muted); margin-bottom:6px; }
.ll-tcard .ll-ddl { grid-template-columns:minmax(120px,max-content) 1fr; }
.ll-tcard .ll-ddl dd { overflow-wrap:anywhere; font-size:12.5px; }
.ll-check { list-style:none; padding:0; margin:0; display:flex; flex-direction:column; gap:6px; font-size:13px; }
.ll-check li { display:flex; justify-content:space-between; gap:12px; }
h4 { margin:18px 0 8px; font-size:14px; }
.ll-ggrid { display:grid; grid-template-columns:repeat(auto-fill,minmax(190px,1fr)); gap:14px; }
.ll-gimg { display:flex; aspect-ratio:3/4; overflow:hidden; background:var(--ll-imgbg); }
.ll-gimg img { width:100%; height:100%; object-fit:cover; display:block; transition:filter .2s; }
.ll-gph { flex-direction:column; align-items:center; justify-content:center; gap:6px; color:var(--ll-dim); font-size:12px; }
.ll-gph-letter { font-size:40px; font-weight:600; color:var(--ll-dash); }
.ll-gcap { display:flex; justify-content:space-between; align-items:center; gap:6px; padding:8px 10px; }
.ll-gcap .ll-name { font-size:13px; }
.ll-gcap .ll-badge { font-size:10px; padding:2px 7px; }
.ll-ggroup { grid-column:1 / -1; border:1px solid var(--ll-line2); border-radius:12px; padding:10px 12px 12px;
             display:flex; flex-direction:column; gap:10px; }
.ll-ghead { display:flex; align-items:center; gap:12px; flex-wrap:wrap; }
.ll-ghead .ll-name { font-size:15px; }
.ll-grid { grid-auto-flow:dense; }
.ll-vers { display:none; grid-column:1 / -1; border:1px solid var(--ll-line2); border-radius:12px;
           padding:12px; flex-direction:column; gap:10px; background:var(--ll-imgbg); }
.ll-vers.ll-show { display:flex; }
.ll-vers-head { display:flex; gap:12px; align-items:baseline; flex-wrap:wrap; font-size:14px; }
.ll-card.ll-open { outline:2px solid var(--ll-warn); }
.ll-done { align-self:flex-start; font-size:11px; font-weight:600; padding:2px 8px; border-radius:99px;
           background:var(--ll-ok); color:#0E1412; margin-right:6px; }
.ll-done-tile { padding:1px 6px; }
#sel_box { display:none !important; }
.ll-card { position:relative; }
.ll-selmark { display:none; position:absolute; top:8px; left:8px; z-index:3; width:22px; height:22px; border-radius:6px;
              border:2px solid #fff; background:rgba(0,0,0,.45); box-shadow:0 1px 3px rgba(0,0,0,.5); }
body.ll-sel-ll_lib #ll_lib .ll-selmark, body.ll-sel-ll_gal #ll_gal .ll-selmark { display:block; }
body.ll-sel-ll_lib #ll_lib .ll-card-data[data-gid] { padding-top:38px; }
.ll-card.ll-picked { outline:3px solid var(--ll-ok) !important; }
.ll-card.ll-picked .ll-selmark { background:var(--ll-ok); border-color:var(--ll-ok); }
.ll-card.ll-picked .ll-selmark::after { content:"\\2713"; color:#0E1412; font-weight:700; font-size:15px;
              position:absolute; left:3px; top:-1px; }
.ll-selbar { align-items:center !important; }
.ll-selcount { font-family:'JetBrains Mono',monospace; font-size:13px; color:var(--ll-muted); }
.ll-gcap2 { flex-direction:column; align-items:flex-start; gap:6px; }
.ll-tchips { display:flex; flex-wrap:wrap; gap:4px; }
.ll-tchip { font-family:'JetBrains Mono',monospace; font-size:10px; font-weight:600; padding:2px 6px; border-radius:6px;
            border:1px solid var(--ll-line2); color:var(--ll-muted); }
.ll-tchip { cursor:pointer; } .ll-tchip:hover { border-color:var(--ll-warn); }
.ll-tchip small { font-size:9px; margin-left:2px; opacity:.8; }
.ll-tchip-cur { color:var(--ll-text, inherit); border-color:var(--ll-muted); }
.ll-tchip-done { background:var(--ll-ok); border-color:var(--ll-ok); color:#0E1412; }
.ll-card.ll-jump { outline:3px solid var(--ll-warn) !important; }
.ll-vchip-btn { cursor:pointer; } .ll-vchip-btn:hover { border-color:var(--ll-warn); color:var(--ll-warn); }
.ll-vlabel { font-family:'JetBrains Mono',monospace; font-size:12px; font-weight:600; color:var(--ll-warn); }
.ll-gver { flex:none; font-family:'JetBrains Mono',monospace; font-size:10px; padding:2px 7px; border-radius:99px;
           border:1px solid var(--ll-line2); color:var(--ll-muted); margin-right:4px; }
.ll-vchip { align-self:flex-start; font-family:'JetBrains Mono',monospace; font-size:11px; padding:2px 8px;
            border-radius:99px; border:1px solid var(--ll-line2); color:var(--ll-muted); }

/* ---- LoRA Studio frame: top tabs, the status strip and the shared page parts ---- */
#ls_top { align-items:center; gap:14px; }
#ls_top h1 { font-weight:700; letter-spacing:-.01em; }
#ls_top .ls-sub { color:var(--ll-muted); font-size:13px; }
#ls_tabs > .tab-wrapper button[role="tab"], #ls_tabs > div > .tab-container button[role="tab"] { font-weight:600; font-size:15px; }
button[role="tab"][aria-selected="true"] { color:var(--ll-text) !important; border-color:#E8A33D !important; }
.ls-strip { display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:10px 18px;
            padding:8px 14px; border-radius:10px; background:#262A31; color:#E6E8EB; font-size:13px; }
.ls-run { cursor:pointer; font-weight:600; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; flex:1 1 320px; }
.ls-st-run .ls-run { color:#F4C06F; } .ls-st-done .ls-run { color:#7FD3C9; } .ls-st-fail .ls-run { color:#F2A497; }
.ls-st-idle .ls-run { color:#C3C8CF; font-weight:500; }
.ls-machine { display:flex; flex-wrap:wrap; gap:6px 14px; align-items:center; }
.ls-meter { display:inline-flex; flex-direction:column; gap:3px; min-width:96px; }
.ls-mlab { font-family:'JetBrains Mono',monospace; font-size:11px; color:#C3C8CF; white-space:nowrap; }
.ls-mbar { display:block; height:4px; border-radius:2px; background:#3A404B; overflow:hidden; }
.ls-mbar > span { display:block; height:100%; background:#21857C; }
.ls-hot .ls-mbar > span { background:#E8A33D; } .ls-hot .ls-mlab { color:#F4C06F; }
.ls-mnote { font-size:11px; color:#9AA1AB; }
.ls-oll { font-family:'JetBrains Mono',monospace; font-size:12px; white-space:nowrap; align-self:center; }
.ls-oll-on { color:#7FDDD3; font-weight:600; } .ls-oll-off { color:#B8BEC7; }
#ls_strip { padding:0 !important; border:0 !important; background:transparent !important; }
.ls-empty, .ls-empty p { color:var(--ll-muted); font-style:italic; }
.ls-filters { border-right:1px solid var(--ll-line); padding-right:10px; }
.ls-actions { align-items:center !important; }
.pp-box { display:flex; flex-direction:column; gap:6px; padding:10px 12px; border:1px solid var(--ll-line); border-radius:10px;
          background:var(--ll-card); }
.pp-row { display:grid; grid-template-columns:170px 110px 1fr; gap:10px; align-items:center; font-size:14px; }
.pp-name { font-weight:600; }
.pp-chip { justify-self:start; font-family:'JetBrains Mono',monospace; font-size:11px; font-weight:600; padding:2px 9px; border-radius:99px; }
.pp-ok { background:#21857C; color:#fff; } .pp-todo { background:#FBE9C8; color:#8F560B; }
.dark .pp-todo { background:#3A2E1A; color:#F4C06F; }
.pp-text { color:var(--ll-muted); font-size:13px; }
.pp-next { margin:6px 0 0; font-weight:600; color:var(--ll-ok); }
.pp-empty { margin:0; color:var(--ll-muted); }
#ov_df td, #hist_df td { cursor:pointer; }
@media (max-width: 700px) { .pp-row { grid-template-columns:1fr; gap:2px; } .ls-filters { border-right:0; } }
"""

PICK_JS = """
if (!window.llSelecting) {
  // Select mode (per view: ll_lib = Library, ll_gal = Gallery): clicking a card selects it instead of opening it
  window.llScope = function(el){ var sc = el && el.closest && el.closest('#ll_lib, #ll_gal'); return sc ? sc.id : ''; };
  window.llSelecting = function(el){ var id = llScope(el); return !!id && document.body.classList.contains('ll-sel-' + id); };
  window.llSelCount = function(){
    ['ll_lib', 'll_gal'].forEach(function(id){
      var n = document.querySelectorAll('#' + id + ' .ll-card.ll-picked[data-gid]').length;
      document.querySelectorAll('.ll-count-' + id).forEach(function(e){ e.textContent = n + ' selected'; });
    });
  };
  window.llSelMode = function(id, on){
    document.body.classList.toggle('ll-sel-' + id, !!on);
    if (!on) document.querySelectorAll('#' + id + ' .ll-card.ll-picked').forEach(function(e){ e.classList.remove('ll-picked'); });
    llSelCount(); return on;
  };
  window.llSelAll = function(id, on){
    document.querySelectorAll('#' + id + ' .ll-card[data-gid]').forEach(function(e){ e.classList.toggle('ll-picked', !!on); });
    llSelCount();
  };
  window.llSelGet = function(id){
    var g = {};
    document.querySelectorAll('#' + id + ' .ll-card.ll-picked[data-gid]').forEach(function(e){ g[e.dataset.gid] = 1; });
    return Object.keys(g).join(',');
  };
}
if (!window.llToggle) {
  window.llToggle = function(gid, btn){
    if (llSelecting(btn) && btn.dataset.gid) { btn.classList.toggle('ll-picked'); llSelCount(); return; }
    var root = (btn && btn.closest('.ll-wrap')) || document;
    var target = root.querySelector('.ll-vers[data-gid="' + gid + '"]');
    var wasOpen = target && target.classList.contains('ll-show');
    root.querySelectorAll('.ll-vers.ll-show').forEach(function(e){ e.classList.remove('ll-show'); });
    root.querySelectorAll('.ll-card.ll-open').forEach(function(e){ e.classList.remove('ll-open'); });
    if (target && !wasOpen) { target.classList.add('ll-show'); if (btn) btn.classList.add('ll-open'); }
  };
}
if (!window.llOpenCard) {
  // Browse tile or type label: open that LoRA's page (its newest version)
  window.llOpenCard = function(gid, card, ds){
    if (card && llSelecting(card) && card.dataset.gid) { card.classList.toggle('ll-picked'); llSelCount(); return; }
    var el = document.querySelector('#pick_box textarea, #pick_box input');
    if (!el || !gid) return;
    var setter = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(el), 'value').set;
    setter.call(el, 'gid:' + gid + '|' + Date.now());
    el.dispatchEvent(new Event('input', {bubbles:true}));
    window.scrollTo({top: 0, behavior: 'smooth'});
  };
}
if (!window.llPicOpen) {
  // Gallery tab: show one picture with its prompt in the viewer next to the grid
  window.llPicOpen = function(path, tile){
    var el = document.querySelector('#pic_box textarea, #pic_box input');
    if (!el) return;
    document.querySelectorAll('.ll-ptile.ll-psel').forEach(function(e){ e.classList.remove('ll-psel'); });
    if (tile) tile.classList.add('ll-psel');
    var setter = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(el), 'value').set;
    setter.call(el, path + '|' + Date.now());
    el.dispatchEvent(new Event('input', {bubbles:true}));
  };
}
if (!window.loraPick) {
  window.loraPick = function(key, card){
    if (card && llSelecting(card) && card.dataset.gid) { card.classList.toggle('ll-picked'); llSelCount(); return; }
    var el = document.querySelector('#pick_box textarea, #pick_box input');
    if(!el) return;
    var setter = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(el), 'value').set;
    setter.call(el, key + '|' + Date.now());
    el.dispatchEvent(new Event('input', {bubbles:true}));
  };
}
"""

def head_html(style: str = "dark") -> str:
    """<head> script: open in the saved style unless the URL already picks one (?__theme=dark|light),
    plus the card-click helper and the Dark/Normal switch."""
    theme = "light" if style == "normal" else "dark"
    return """
<script>
(function(){
  try {
    var u = new URL(window.location.href);
    if (!u.searchParams.has('__theme')) {          // the saved style, without loading the page a second time
      u.searchParams.set('__theme','%s'); window.history.replaceState(null, '', u.toString());
      var dk = u.searchParams.get('__theme') === 'dark';
      var fix = function(){ document.body.classList.toggle('dark', dk); };
      fix(); setTimeout(fix, 0); setTimeout(fix, 400);
      try { window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function(){ setTimeout(fix, 0); }); } catch(e) {}
    }
  } catch(e) {}
})();
window.llToggleStyle = function(){
  try {
    var u = new URL(window.location.href);
    var dark = document.body.classList.contains('dark') || u.searchParams.get('__theme') === 'dark';
    u.searchParams.set('__theme', dark ? 'light' : 'dark');
    window.location.replace(u.toString());
  } catch(e) {}
};
// A page left open across a restart of the program sends its values to the wrong (renumbered) components,
// e.g. "Value: 'A' is not in the list of choices". Notice the restart and ask for a reload.
(function(){
  var first = null, shown = false;
  function banner(){
    if (shown || !document.body) return; shown = true;
    var d = document.createElement('div');
    d.id = 'll_restart';
    d.style.cssText = 'position:fixed;top:0;left:0;right:0;z-index:99999;padding:10px 16px;background:#a85a08;color:#fff;' +
                      'font:600 14px sans-serif;display:flex;gap:14px;align-items:center;justify-content:center;box-shadow:0 2px 8px rgba(0,0,0,.4)';
    d.innerHTML = 'LoRA Studio was restarted: this page is out of date. Reload it before you continue.';
    var b = document.createElement('button');
    b.textContent = 'Reload now';
    b.style.cssText = 'padding:4px 14px;border-radius:6px;border:0;background:#fff;color:#a85a08;font-weight:700;cursor:pointer';
    b.onclick = function(){ window.location.reload(); };
    d.appendChild(b);
    document.body.appendChild(d);
  }
  function check(){
    fetch('gradio_api/app_id', {cache: 'no-store'}).then(function(r){ return r.ok ? r.text() : null; }).then(function(t){
      if (!t) return;
      if (first === null) first = t; else if (t !== first) banner();
    }).catch(function(){});
  }
  check();
  setInterval(check, 10000);
})();
%s
</script>
""" % (theme, PICK_JS)


TOGGLE_JS = "() => { if (window.llToggleStyle) window.llToggleStyle(); }"


# The LoRA Studio palette (the same as the layout slides): warm paper, ink, amber for actions, teal for "done".
PALETTE = {"paper": "#F6F5F1", "card": "#FDFDFB", "ink": "#1C1F24", "muted": "#4A5160", "amber": "#E8A33D",
           "amber_text": "#8F560B", "amber_tint": "#FBE9C8", "line": "#C9CDD3", "teal": "#21857C", "chip": "#ECEEF1",
           "dark_paper": "#14161A", "dark_card": "#1B1E23", "dark_card2": "#262A31", "dark_line": "#343944",
           "dark_ink": "#E6E8EB"}


def make_theme():
    P = PALETTE
    t = gr.themes.Base(
        primary_hue="amber", neutral_hue="slate", radius_size="md",
        font=["IBM Plex Sans", "Inter", "system-ui", "Segoe UI", "Roboto", "sans-serif"],     # local fonts only: no request
        font_mono=["JetBrains Mono", "Fira Code", "DejaVu Sans Mono", "Consolas", "monospace"],  # to Google at every start
    )
    try:
        t = t.set(
            body_background_fill=P["paper"], body_background_fill_dark=P["dark_paper"],
            body_text_color=P["ink"], body_text_color_dark=P["dark_ink"],
            background_fill_primary=P["card"], background_fill_primary_dark=P["dark_card"],
            background_fill_secondary=P["paper"], background_fill_secondary_dark=P["dark_paper"],
            block_background_fill=P["card"], block_background_fill_dark=P["dark_card"],
            block_border_color=P["line"], block_border_color_dark=P["dark_line"],
            border_color_primary=P["line"], border_color_primary_dark=P["dark_line"],
            block_label_text_color=P["muted"], block_title_text_color=P["ink"], block_title_text_color_dark=P["dark_ink"],
            color_accent=P["amber"], color_accent_soft=P["amber_tint"], color_accent_soft_dark="#3A2E1A",
            link_text_color=P["amber_text"], link_text_color_dark=P["amber"],
            button_primary_background_fill=P["amber"], button_primary_background_fill_dark=P["amber"],
            button_primary_background_fill_hover="#D99227", button_primary_background_fill_hover_dark="#F0B457",
            button_primary_text_color=P["ink"], button_primary_text_color_dark=P["ink"],
            button_primary_border_color=P["amber"], button_primary_border_color_dark=P["amber"],
            button_secondary_background_fill=P["card"], button_secondary_background_fill_dark=P["dark_card2"],
            button_secondary_border_color=P["line"], button_secondary_border_color_dark=P["dark_line"],
            input_background_fill=P["card"], input_background_fill_dark=P["dark_card2"],
            input_border_color=P["line"], input_border_color_dark=P["dark_line"],
            input_border_color_focus=P["amber"], input_border_color_focus_dark=P["amber"],
            checkbox_background_color_selected=P["amber"], checkbox_background_color_selected_dark=P["amber"],
            checkbox_border_color_selected=P["amber"], checkbox_border_color_selected_dark=P["amber"],
            slider_color=P["amber"], slider_color_dark=P["amber"],
            table_even_background_fill=P["card"], table_odd_background_fill=P["paper"],
            table_even_background_fill_dark=P["dark_card"], table_odd_background_fill_dark=P["dark_paper"],
        )
    except TypeError as e:                  # an older Gradio without one of these names: keep the base theme
        print(f"[LoRA Studio] theme: {e}")
    return t


STATUS_CHOICES = ["OK", "Warnings", "Problems"]

LIB = Library()


def personal_bases() -> list:
    """Base models used by your own LoRAs (those with a dataset), in the Settings order."""
    used = {l.base for l in LIB.loras.values() if l.dataset is not None}
    order = LIB.base_names() + sorted(used - set(LIB.base_names()))
    return [b for b in order if b in used]


def lora_choices(extra_gid: str = ""):
    """Details picker: one entry per version group of YOUR LoRAs (value = group id); the Version dropdown picks
    the file. `extra_gid` keeps a download in the list while it is open (opened from Library with downloads on)."""
    out = []
    for gid, members in LIB.groups.items():
        if not members or (members[-1].dataset is None and gid != extra_gid):
            continue
        n = members[-1]
        extra = f"{len(members)} versions" if len(members) > 1 else n.stem
        out.append((f"{group_title(members)}  ({n.base}, {extra})", gid))
    return sorted(out, key=lambda c: c[0].lower())


def version_choices(lora) -> list:
    members = LIB.group_of(lora)
    newest, best = members[-1], best_member(members)
    return [(f"{m.vlabel}{' (newest)' if m is newest and len(members) > 1 else ''}"
             f"{' ★ best' if m is best else ''}  ·  {m.stem}", m.key)
            for m in reversed(members)]


UI_HOOKS: dict = {}          # components lora_studio.py wires to other tabs


def versions_html(lora) -> str:
    """All versions of a LoRA (one row each): trained, pictures, steps, final loss, completed."""
    rows = []
    for l in sorted(LIB.group_of(lora), key=lambda x: -(x.version or 0)):
        m = l.meta or {}
        when = ""
        try:
            t = float(m.get("ss_training_finished_at") or 0) or (l.path.stat().st_mtime if l.path else 0)
            when = time.strftime("%Y-%m-%d", time.localtime(t)) if t else ""
        except (OSError, ValueError, TypeError):
            pass
        import trainingcard as TC
        card = TC.stored(l.sidecar)                      # the training card, when the version was scanned
        when = (card.get("trained_on") or "")[:10] or when
        pics = card.get("pictures") or m.get("ss_num_train_images") or ""
        steps = card.get("steps") or m.get("ss_max_train_steps") or m.get("ss_steps") or ""
        loss = f"{l.loss[-1][1]:.3f}" if l.loss else (card.get("final_loss") or "")
        done = "✓ yes" if is_completed(l) else "no"
        rows.append(f"<tr><td><b>{esc(l.vlabel)}</b></td><td>{esc(when)}</td><td>{esc(pics)}</td><td>{esc(steps)}</td>"
                    f"<td>{esc(loss)}</td><td>{done}</td></tr>")
    return ('<table class="rs-table"><thead><tr><th>Version</th><th>Trained</th><th>Pictures</th><th>Steps</th>'
            '<th>Final loss</th><th>Completed</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table>")


def build_library_ui(app, detach_settings=False, maint_extra=None):
    """Render the Library tabs (Carddex, Library, Details, Gallery, Personal sync, Settings) into the current layout
    context of `app`. Used standalone by build_app() and inside lora_studio.py. With detach_settings the
    two settings panels are built hidden (FOLDERS_BLOCK, LOOK_BLOCK) for the caller to .render() elsewhere."""
    global FOLDERS_BLOCK, LOOK_BLOCK
    s = LIB.settings
    # Hidden helper that turns a card/tile click into "open Details". It sits outside the tabs because a tab's
    # contents only exist in the page once that tab has been opened.
    pick_box = gr.Textbox(elem_id="pick_box", show_label=False, container=False)
    sel_box = gr.Textbox(elem_id="sel_box", show_label=False, container=False)
    pic_box = gr.Textbox(elem_id="pic_box", show_label=False, container=False)

    def sel_toolbar(scope: str) -> dict:
        """Select mode for a card view: tick Select, click cards, then mark them all at once."""
        with gr.Row(elem_classes=["ll-selbar"]):
            cb = gr.Checkbox(label="Select", value=False, scale=0, min_width=100)
            all_b = gr.Button("Select all shown", size="sm", scale=0, min_width=140)
            none_b = gr.Button("Clear selection", size="sm", scale=0, min_width=130)
            gr.HTML(f'<span class="ll-selcount ll-count-{scope}">0 selected</span>')
            done_b = gr.Button("\u2713 Mark Completed", size="sm", variant="primary", scale=0, min_width=160)
            undone_b = gr.Button("Mark not completed", size="sm", scale=0, min_width=160)
        msg = gr.Markdown()
        return {"scope": scope, "cb": cb, "all": all_b, "none": none_b, "done": done_b, "undone": undone_b, "msg": msg}

    with gr.Tabs() as tabs:

        # ---------------- Browse: one page, two looks (Pictures / Status) ---------------- #
        with gr.Tab("Browse", id="gallery"):
            with gr.Row():
                with gr.Column(scale=1, min_width=240, elem_classes=["ls-filters"]):
                    gal_search = gr.Textbox(label="Search", placeholder="Name, character, trigger word")
                    search = gal_search
                    gal_base = gr.Dropdown(["All"] + LIB.base_names(), value="All", label="Base model")
                    base_f = gr.CheckboxGroup(label="Base model", choices=LIB.base_names() + ["Unknown"],
                                              value=LIB.base_names() + ["Unknown"], visible=False)
                    status_f = gr.CheckboxGroup(label="Status", choices=STATUS_CHOICES, value=["Warnings", "Problems"],
                                                visible=False)
                    group_f = gr.Radio(["Character", "Base model"], value="Character", label="Group by", visible=False)
                    gal_sort = gr.Radio(["Newest first", "Name"], value="Newest first", label="Sort")
                    gal_prev = gr.Checkbox(label="Only with preview picture")
                    gal_fav = gr.Checkbox(label="Favorites only")
                    rescan = gr.Button("Rescan folders", variant="primary")
                with gr.Column(scale=4):
                    look = gr.Radio(["Pictures", "Status"], value="Pictures", show_label=False, container=False,
                                    elem_id="col_look")
                    with gr.Column(visible=True) as look_pics:
                        gal_sel = sel_toolbar("ll_gal")
                        gal_html = gr.HTML(elem_id="ll_gal", **({"js_on_load": PICK_JS}
                                           if "js_on_load" in inspect.signature(gr.HTML.__init__).parameters else {}))
                        gr.Markdown("Click a tile: its LoRA page. A green chip = Completed.", elem_classes=["ls-note"])
                    with gr.Column(visible=False) as look_status:
                        html_kwargs = {}
                        if "js_on_load" in inspect.signature(gr.HTML.__init__).parameters:
                            html_kwargs["js_on_load"] = PICK_JS
                        lib_sel = sel_toolbar("ll_lib")
                        lib_html = gr.HTML(elem_id="ll_lib", **html_kwargs)
                        gr.Markdown("Only problems of the LoRA files themselves. What a dataset still needs (untrained, "
                                    "too few pictures) is in Datasets › Improvements.", elem_classes=["ls-note"])

        # ---------------- One LoRA: Overview · Pictures · Training · Edit ---------------- #
        with gr.Tab("LoRA page", id="details"):
            with gr.Row(elem_id="lp_head", equal_height=True):
                lp_back = gr.Button("← Browse", size="sm", scale=0, min_width=110)
                with gr.Column(scale=1, min_width=200):
                    lp_title = gr.HTML("", elem_id="lp_title")
                lp_types = gr.Radio(choices=[], label="LoRA type", show_label=False, container=False, scale=0,
                                    min_width=240, elem_id="lp_types")
                ver_dd = gr.Dropdown(label="Version", choices=[], interactive=True, scale=0, min_width=200)
            pick_dd = gr.Dropdown(label="LoRA", choices=[], interactive=True, filterable=True, visible=False)
            cur_key = gr.State("")
            d_empty = gr.Markdown("**No LoRA open.** Click a tile in Browse.", elem_classes=["ls-empty"])
            d_body = gr.Column(visible=False)
            with d_body:
              with gr.Tabs(selected="lp_over") as lp_tabs:
                with gr.Tab("Overview", id="lp_over"):
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=2, min_width=280):
                            d_preview = gr.Image(label="Preview", type="pil", interactive=False, height=460)
                            gr.Markdown("Preview from LoRA Manager · NSFW blur follows Settings › Look and base models",
                                        elem_classes=["ls-note"])
                            d_examples = gr.Gallery(label="Example pictures from LoRA Manager", columns=4, height=200,
                                                    object_fit="cover", visible=False, allow_preview=True)
                        with gr.Column(scale=3):
                            e_done = gr.Checkbox(label="Completed", value=False, interactive=False,
                                                 info="Ticked automatically when training finishes. Completed hides this "
                                                      "launch script in Training › Run.")
                            e_done_msg = gr.Markdown()
                            d_head = gr.HTML()
                            with gr.Row(elem_classes=["ls-bar"]):
                                d_accept = gr.Button("Accept the warnings", size="sm", scale=0, min_width=190)
                                d_accept_msg = gr.Markdown(
                                    "A warning you have looked at (for example *Run stopped at epoch 8 of 10* when "
                                    "you stopped it on purpose): accept it, and the orange mark goes away. Press "
                                    "again to show accepted warnings again.", elem_classes=["ls-note"])
                            with gr.Accordion("Copy this LoRA to your LoRA folder", open=False) as d_cp_acc:
                                with gr.Row():
                                    d_cp_dest = gr.Textbox(label="Copy to folder", scale=4)
                                    d_cp_btn = gr.Button("Copy to LoRA folder", variant="primary", scale=1)
                                d_cp_note = gr.Markdown()
                                d_cp_over = gr.Checkbox(label="Replace a different file with the same name (the old one "
                                                              "goes to the Trash)", value=False)
                                d_cp_msg = gr.Markdown()
                            with gr.Row(elem_classes=["ls-bar"]):
                                d_pics = gr.Button("Pictures made with it", size="sm", scale=0, min_width=200)
                                d_copy_jump = gr.Button("Copy to LoRA folder", size="sm", scale=0, min_width=170)
                with gr.Tab("Pictures", id="lp_pics"):
                    gr.Markdown("Every picture you made with this LoRA, **all its versions together**. Found through the "
                                "`.json` ComfyUI saves next to each picture (or the PNG's own metadata). A scan only reads "
                                "new and changed pictures.", elem_classes=["ls-note"])
                    pg_lora = gr.Dropdown(label="LoRA", choices=[], interactive=True, filterable=True, visible=False)
                    with gr.Row(equal_height=True):
                        pg_ver = gr.Dropdown(label="Version", choices=[("All versions", "")], value="", interactive=True,
                                             scale=1)
                        pg_folder = gr.Textbox(label="Picture folder for this LoRA type",
                                               placeholder="Where ComfyUI saves the pictures of this LoRA type", scale=3)
                        pg_scan = gr.Button("Scan for new pictures", variant="primary", scale=1, min_width=170)
                        pg_all = gr.Button("Scan all picture folders", scale=1, min_width=170)
                        pg_stop = gr.Button("Stop scan", variant="stop", scale=0, min_width=110)
                    pg_msg = gr.Markdown()
                    with gr.Accordion("Old names of this LoRA (pictures made before a rename)", open=False) as pg_old_acc:
                        gr.Markdown("Names found in your pictures that no LoRA file has now, but that look like this LoRA. "
                                    "Tick the ones that are this LoRA under an older name; their pictures join as *old name*.")
                        pg_old = gr.CheckboxGroup(label="Old names", choices=[], value=[])
                        with gr.Row(elem_classes=["ls-bar"]):
                            pg_old_add = gr.Textbox(label="Or type a name", placeholder="e.g. mycharILLUSTRIOUS", scale=3)
                            pg_old_save = gr.Button("Save old names", variant="primary", scale=1, min_width=150)
                        pg_old_msg = gr.Markdown()
                    with gr.Row():
                        with gr.Column(scale=3):
                            with gr.Row(elem_classes=["ls-bar"]):
                                pg_prev = gr.Button("\u25C0 Newer", size="sm", min_width=90)
                                pg_page = gr.Number(value=1, precision=0, minimum=1, label="Page", min_width=70)
                                pg_pages = gr.Markdown()
                                pg_next = gr.Button("Older \u25B6", size="sm", min_width=90)
                            pg_html = gr.HTML('<p class="ls-empty">No pictures yet.</p>', elem_id="pg_grid")
                        with gr.Column(scale=2, elem_id="pg_view"):
                            pg_img = gr.Image(label="Picture", type="pil", interactive=False, height=560)
                            pg_info = gr.Markdown()
                            pg_prompt = gr.Textbox(label="Prompt", lines=4, interactive=False,
                                                   **({"buttons": ["copy"]} if "buttons" in inspect.signature(gr.Textbox.__init__).parameters
                                                      else {"show_copy_button": True}))
                with gr.Tab("Training", id="lp_train"):
                    d_settings = gr.HTML()
                    with gr.Row(elem_classes=["ls-bar"]):
                        d_rescan = gr.Button("Read the training again", size="sm", scale=0, min_width=210)
                        gr.Markdown("after you changed the launch script or the log", elem_classes=["ls-note"])
                    d_loss = gr.LinePlot(x="step", y="loss", label="Loss", height=260)
                    gr.Markdown("#### All versions")
                    lp_versions = gr.HTML(elem_id="lp_versions")
                    with gr.Row(elem_classes=["ls-bar"]):
                        lp_newv = gr.Button("Train a new version…", size="sm", scale=0, min_width=190)
                        gr.Markdown("opens Datasets › Edit existing › 9 Launch script", elem_classes=["ls-note"])
                with gr.Tab("Edit", id="lp_edit"):
                    with gr.Row():
                        e_name = gr.Textbox(label="Display name")
                        e_char = gr.Textbox(label="Character (groups it in Browse)")
                    with gr.Row():
                        e_ver = gr.Textbox(label="Version", info="shown as the version in ComfyUI's LoRA Manager", scale=1)
                        e_cskip = gr.Number(label="Clip skip", value=None, precision=0, minimum=1, scale=1)
                        e_rate = gr.Radio([("–", 0), ("1", 1), ("2", 2), ("3", 3), ("4", 4), ("5", 5)], value=0,
                                          label="Your rating", info="4-5 good, 1-2 weak: Collection › Insights compares "
                                          "the datasets behind them", scale=2)
                    e_trig = gr.Textbox(label="Trigger words (one per line)", lines=3)
                    e_desc = gr.Textbox(label="Description", lines=3)
                    e_notes = gr.Textbox(label="Notes", lines=3)
                    with gr.Row():
                        e_str = gr.Number(label="Strength", value=None)
                        e_clip = gr.Number(label="CLIP strength", value=None)
                        e_nsfw = gr.Dropdown(label="Preview NSFW level", choices=[("Unknown (0)", 0)]
                                             + [(k, v) for k, v in NSFW_LEVELS.items()], value=0)
                        e_fav = gr.Checkbox(label="Favorite")
                    with gr.Row():
                        e_flag = gr.Dropdown(label="Flag", choices=["none", "warning", "problem"], value="none")
                        e_reason = gr.Textbox(label="Flag reason")
                    with gr.Row(elem_classes=["ls-bar"]):
                        e_save = gr.Button("Save", variant="primary", scale=0, min_width=120)
                        gr.Markdown(f"Saved in the dataset's {ARCHIVE_DIR}/ folder, a .bak is kept. Training settings "
                                    "(the Training tab) come from the LoRA file, its launch script and its log and cannot be edited.",
                                    elem_classes=["ls-note"])
                    e_msg = gr.Markdown()

        # ---------------- Insights: what your datasets, LoRAs and runs say ---------------- #
        with gr.Tab("Insights", id="insights") as ins_tab:
            gr.Markdown("### Insights", elem_classes=["ls-h"])
            gr.Markdown("What your own datasets, LoRAs and runs say, read on this computer: the datasets behind your good "
                        "and weak LoRAs, and what your graphics card managed per LoRA type. With few LoRAs these are "
                        "hints, not rules: each says how many it comes from. Step 5 and step 9 show the short version.",
                        elem_classes=["ls-note"])
            with gr.Row(elem_classes=["ls-bar"]):
                ins_btn = gr.Button("Read again", size="sm", scale=0, min_width=130)
            ins_html = gr.HTML()

        # ---------------- Maintenance: Sync | Clean up ---------------- #
        with gr.Tab("Maintenance", id="maint"):
          with gr.Row(equal_height=False):
            with gr.Column(scale=1, elem_classes=["ls-card"]):
                gr.HTML('<div class="rs-label">IN · SYNC</div>')
                gr.Markdown("### Personal sync from your LoRA folder")
                gr.Markdown(
                    "Copies the LoRA Manager files of your own LoRAs (`.metadata.json`, preview and example pictures) "
                    f"from the LoRA folder into each dataset's **`{ARCHIVE_DIR}/`** folder, where the Collection reads them. "
                    "A LoRA belongs to a dataset when it has the same name and size as a model file in the dataset, or "
                    "the same SHA256 hash (renamed copies). Downloads are left alone. The `.metadata.json` files are synced both "
                    "ways: what only one side has (notes, tags, trigger words, Civitai data …) is copied to the other, "
                    "an empty field takes the other side's value, and where both differ the newer file wins. A `.bak` "
                    "is made before a file is changed.")
                with gr.Accordion("LoRA Manager's example pictures folder", open=False):
                    _lmr = lm_example_roots(LIB)
                    lm_dir = gr.Textbox(label="Example pictures folder (empty = read from LoRA Manager's settings)",
                                        value=LIB.data.get("lm_examples_dir") or "",
                                        placeholder=_lmr[0] if _lmr else "the 'Example images' folder set in LoRA Manager")
                    gr.Markdown("The pictures you add to a LoRA's card in LoRA Manager are kept there, one folder per "
                                "LoRA (its SHA256). Sync links them to the LoRA page (they are not copied). Found now: "
                                + (f"`{_lmr[0]}`" if _lmr else "nothing (set the folder here)") + ".",
                                elem_classes=["ls-note"])
                    lm_dir_save = gr.Button("Save folder", size="sm")
                    lm_dir_msg = gr.Markdown()
                sync_btn = gr.Button("Sync now", variant="primary")
                sync_msg = gr.Markdown()
                gr.HTML('<div class="rs-label" style="margin-top:14px">TRAINING CARDS</div>')
                gr.Markdown("### Scan all LoRAs")
                gr.Markdown(
                    "Reads every LoRA of your datasets once: its launch script, the LoRA file's training data, the repeat "
                    "folders and the run log, into separate fields on the LoRA page (Training tab), saved in the "
                    f"dataset's `{ARCHIVE_DIR}/` copy of the LoRA Manager file. A version that was scanned is read again "
                    "only when its file changes; a new version is read the first time it is opened or scanned. The long "
                    "training text older versions wrote into *Notes* is taken out (your own notes stay).",
                    elem_classes=["ls-note"])
                with gr.Row(elem_classes=["ls-bar"]):
                    tc_btn = gr.Button("Scan all LoRAs", variant="primary", scale=0, min_width=170)
                    tc_force = gr.Checkbox(label="Read the scanned ones again too", value=False, scale=0, min_width=260)
                tc_msg = gr.Markdown()
            with gr.Column(scale=1, elem_classes=["ls-card"]):
                gr.HTML('<div class="rs-label">OUT · CLEAN UP</div>')
                gr.Markdown("### Free disk space")
                gr.Markdown("Everything goes to the desktop **Trash**; restore it from the file manager. One run at a "
                            "time: Training › Results.", elem_classes=["ls-note"])
                with gr.Accordion("Leftovers of all finished LoRAs", open=True):
                    gr.Markdown("Epoch files, saved states (`-state` folders), `.tmp.safetensors`, `.npz` latent caches and "
                                "musubi-tuner's cache folders. Only for LoRAs that have a final file; a dataset with a "
                                "training run in progress is skipped.")
                    with gr.Row(elem_classes=["ls-bar"]):
                        lo_scan = gr.Button("Find leftovers", size="sm", scale=1, min_width=140)
                        lo_ok = gr.Checkbox(label="Yes, move all of these to the Trash", value=False, scale=3)
                        lo_go = gr.Button("Move to Trash", variant="stop", size="sm", scale=1, min_width=140)
                    lo_md = gr.Markdown()
                    lo_paths = gr.State([])
                with gr.Accordion("Orphaned LoRA Manager files", open=False):
                    gr.Markdown("A `.metadata.json` / `.civitai.info` with its preview and example images, whose "
                                "`.safetensors` file no longer exists (deleted or renamed LoRAs).")
                    with gr.Row(elem_classes=["ls-bar"]):
                        or_scan = gr.Button("Scan", size="sm", scale=1, min_width=100)
                        or_ok = gr.Checkbox(label="Yes, move the ticked ones to the Trash", value=False, scale=3)
                        or_go = gr.Button("Move ticked to Trash", variant="stop", size="sm", scale=1, min_width=170)
                    or_md = gr.Markdown()
                    or_pick = gr.CheckboxGroup(choices=[], label="Orphaned meta files", value=[])
                if maint_extra is not None:
                    maint_extra()

        # ---------------- Settings ---------------- #
        # Two panels (folders / library and look). Standalone they sit in a Settings tab here;
        # in LoRA Studio they are moved to the Settings top tab (detach_settings=True).
        with (contextlib.nullcontext() if detach_settings else gr.Tab("Settings", id="settings")):
            with gr.Column(render=not detach_settings) as FOLDERS_BLOCK:
                gr.Markdown(f"Where your own data is: datasets, LoRAs, model files and ComfyUI pictures. Saved to "
                            f"`{SETTINGS_FILE.name}` (you can also edit that file). One folder per line.")
                st_sd = gr.Textbox(
                    label="sd-scripts folder",
                    info="Your sd-scripts install (with venv/ and the *_train_network.py files). "
                         "Used by the Training LoRA tabs. Changing it needs a restart.",
                    value=s["sd_scripts_dir"], visible=not detach_settings)
                with gr.Row():
                    st_ds = gr.Textbox(
                        label="Dataset folders",
                        info="Where ALL your datasets are. Each dataset folder holds repeat folders (10_name), "
                             f"log/, model/, launch scripts and {ARCHIVE_DIR}/. The FIRST one is where the "
                             "Training tabs list and create datasets.",
                        lines=4, value="\n".join(s["dataset_dirs"]))
                    st_lora = gr.Textbox(
                        label="LoRA folders",
                        info="Where ALL your LoRAs are, with the LoRA Manager files from ComfyUI "
                             "(.metadata.json, preview and example images). Only read by Personal sync.",
                        lines=4, value="\n".join(s["lora_dirs"]))
                with gr.Row():
                    st_scripts = gr.Textbox(
                        label="Extra launch script folders (optional)",
                        info="Folders with .sh launch scripts for Training › Run, besides your dataset folders.",
                        lines=2, value="\n".join(s["script_dirs"]))
                    st_sample = gr.Textbox(
                        label="Monitor sample folder (optional)",
                        info="Where the Monitor tab looks for the newest sample images. Empty = <sd-scripts>/output.",
                        value=s["sample_dir"], visible=False)
                st_models = gr.Textbox(
                    label="Model folder (optional)",
                    info="Default folder for training model files (base model, text encoder, VAE) when a LoRA type "
                         "has no folder of its own yet in Settings › Model files.",
                    value=s["models_dir"])
                st_pics = gr.Textbox(
                    label="ComfyUI picture folder (optional)",
                    info="Where ComfyUI saves your pictures (its output folder). The Gallery tab looks in here for a "
                         "folder per LoRA type (Anima, Illustrious, Pony, ...).",
                    value=s.get("pictures_dir", ""))
                st_save = gr.Button("Save folders and rescan", variant="primary")
                st_msg = gr.Markdown()
            with gr.Column(render=not detach_settings) as LOOK_BLOCK:
                st_style = gr.Radio(["Dark", "Normal"], label="Style (default when the program opens)",
                                    value="Normal" if s["style"] == "normal" else "Dark")
                st_bases = gr.Textbox(label="Base models: Name = keywords (checked top to bottom)", lines=6,
                                      value="\n".join(f"{b} = {', '.join(k)}" for b, k in s["bases"]))
                st_expected = gr.CheckboxGroup(label="Expected bases per character", choices=LIB.base_names(),
                                               value=[b for b in s["expected_bases"] if b in LIB.base_names()])
                with gr.Row():
                    st_blur = gr.Checkbox(label="Blur NSFW previews (hover to show)", value=s["nsfw_blur"])
                    st_level = gr.Dropdown(label="Blur at level and above", choices=list(NSFW_LEVELS),
                                           value=next((k for k, v in NSFW_LEVELS.items() if v == s["nsfw_level"]),
                                                      "X (8)"))
                    st_unknown = gr.Checkbox(label="Also blur unknown level when model is marked NSFW",
                                             value=s["blur_unknown_nsfw"])
                st_save2 = gr.Button("Save settings and rescan", variant="primary")
                st_msg2 = gr.Markdown()
                _pc = purpose_colors(LIB)
                with gr.Group():
                    pc_on = gr.Checkbox(label="Colour the Collection cards by purpose (border and label; off = all "
                                              "white as before)", value=_pc["on"])
                    with gr.Row():
                        pc_pick = [gr.ColorPicker(label=lab, value=_pc[k], scale=1) for k, lab in PURPOSES]
                    with gr.Row(elem_classes=["ls-bar"]):
                        pc_save = gr.Button("Save colours", size="sm", scale=0, min_width=140)
                        pc_reset = gr.Button("Default colours", size="sm", scale=0, min_width=150)
                    pc_msg = gr.Markdown()

    # ---------------- Callbacks ---------------- #
    filters = [search, base_f, status_f, group_f]
    gal_inputs = [gal_search, gal_base, gal_sort, gal_prev, gal_fav]

    def do_gallery(q, b, so, pv, fv):
        return render_gallery(LIB, q, b, so, pv, fv)

    for f in gal_inputs:
        f.change(do_gallery, gal_inputs, gal_html)

    def pc_store(on, *cols):
        LIB.data["purpose_colors"] = {"on": bool(on), **{k: (c or PURPOSE_COLORS[k]) for (k, _), c in zip(PURPOSES, cols)}}
        LIB.save_data()
        return ("✔ Saved: the cards are coloured by purpose." if on else "✔ Saved: the cards are white.")
    pc_save.click(pc_store, [pc_on] + pc_pick, pc_msg).then(do_gallery, gal_inputs, gal_html)
    pc_on.input(pc_store, [pc_on] + pc_pick, pc_msg).then(do_gallery, gal_inputs, gal_html)
    pc_reset.click(lambda: [PURPOSE_COLORS[k] for k, _ in PURPOSES], None, pc_pick)

    def set_look(v):
        st = v == "Status"
        return (gr.update(visible=not st), gr.update(visible=st), gr.update(visible=st), gr.update(visible=st),
                gr.update(visible=not st), gr.update(visible=not st), gr.update(visible=not st))
    look.change(set_look, look, [look_pics, look_status, status_f, group_f, gal_sort, gal_prev, gal_fav])
    gal_base.change(lambda b: LIB.base_names() + ["Unknown"] if b in (None, "", "All") else [b], gal_base, base_f)

    def do_render(q, b, st, g):
        return render_library(LIB, q, b, st, g)

    def list_updates(q, b, st, g):
        """Library + the three lists that should only offer your own LoRAs and their base models."""
        names = personal_bases()
        keep = [x for x in (b or []) if x in names] or names
        return (render_library(LIB, q, keep, st, g),
                gr.update(choices=names, value=keep),
                gr.update(choices=lora_choices()),
                gr.update(choices=["All"] + names))

    def do_rescan(q, b, st, g, progress=gr.Progress()):
        LIB.scan(progress)
        return list_updates(q, b, st, g)

    def do_startup(q, b, st, g, progress=gr.Progress()):
        LIB.ensure_loaded(progress)
        return list_updates(q, b, st, g)

    for f in filters:
        f.change(do_render, filters, lib_html)
    rescan.click(do_rescan, filters, [lib_html, base_f, pick_dd, gal_base]).then(do_gallery, gal_inputs, gal_html
                 ).then(lambda g: gr.update(choices=lora_choices(), value=g if g in LIB.groups else None), pg_lora, pg_lora)
    app.load(do_startup, filters, [lib_html, base_f, pick_dd, gal_base]).then(do_gallery, gal_inputs, gal_html
             ).then(lambda: gr.update(choices=lora_choices()), None, pg_lora)

    def mass_mark(gids, done):
        """Tick or untick Completed for every selected card (Library card = one dataset + LoRA type,
        Carddex tile = every LoRA type of one dataset)."""
        picked = [g for g in (gids or "").split(",") if g]
        if not picked:
            return "Nothing selected: tick **Select**, click the cards you want (or *Select all shown*), then press the button again."
        changed, skipped = [], 0
        # a Carddex tile is a whole dataset ("ds:mychar"): it stands for every LoRA type of that dataset
        expanded = []
        for gid in picked:
            if gid.startswith("ds:"):
                name = gid[3:]
                expanded += [g for g, m in LIB.groups.items()
                             if m and m[-1].dataset is not None and m[-1].dataset.name.lower() == name]
            else:
                expanded.append(gid)
        for gid in dict.fromkeys(expanded):
            l = LIB.newest(gid)
            ds, t = done_key(l) if l else ("", "")
            if not ds:
                skipped += 1
                continue
            COMPLETED.set_done(ds, t, done, how="manual", lora=l.stem)
            changed.append(f"{ds} · {l.base}")
        what = "Completed" if done else "not completed"
        msg = f"Marked **{len(changed)}** as {what}: " + ", ".join(sorted(set(changed))) if changed else ""
        if skipped:
            msg += f"{' ' if msg else ''}Skipped {skipped} without a dataset (downloads cannot be marked)."
        return msg + " Training › Run shows it after *Refresh*."

    for tb in (gal_sel, lib_sel):
        sc = tb["scope"]
        tb["cb"].change(None, tb["cb"], None, js=f"(v) => llSelMode('{sc}', v)")
        tb["all"].click(lambda: True, None, tb["cb"], js=f"() => {{ llSelMode('{sc}', true); llSelAll('{sc}', true); }}")
        tb["none"].click(None, None, None, js=f"() => llSelAll('{sc}', false)")
        for btn, flag in ((tb["done"], True), (tb["undone"], False)):
            btn.click(lambda g, f=flag: mass_mark(g, f), sel_box, tb["msg"], js=f"(v) => [llSelGet('{sc}')]"
                      ).then(do_render, filters, lib_html).then(do_gallery, gal_inputs, gal_html
                      ).then(None, None, None, js="() => { if (window.llSelCount) llSelCount(); }")

    detail_outputs = [d_preview, d_head, d_settings, d_loss,
                      e_name, e_char, e_trig, e_notes, e_str, e_clip, e_nsfw, e_fav, e_flag, e_reason, e_msg,
                      pick_dd, ver_dd, cur_key, e_done, d_empty, d_body, e_ver, e_cskip, e_desc, d_examples, e_rate]

    def on_pick(v):
        key = (v or "").split("|")[0]
        if key.startswith("gid:"):
            n = LIB.newest(key[4:])
            key = n.key if n else ""
        if key not in LIB.loras:
            return [gr.update()] * (len(detail_outputs) + 1)
        return show_details(key) + [gr.Tabs(selected="details")]

    pick_box.input(on_pick, pick_box, detail_outputs + [tabs])
    pick_box.change(on_pick, pick_box, detail_outputs + [tabs])

    def show_details(key):
        lora = LIB.loras.get(key)
        if not lora:
            return ([None, "", "", pd.DataFrame(columns=["step", "loss"])] + [gr.update()] * 10
                    + ["", gr.update(), gr.update(), "", gr.update(value=False, interactive=False), gr.update(visible=True),
                       gr.update(visible=False), "", None, "", gr.update(visible=False, value=None), 0])
        import trainingcard as TC
        if lora.dataset is not None and lora.final and not TC.is_scanned(lora):
            try:                                 # a version not read yet: its training card is made now
                TC.scan_one(LIB, lora)
            except Exception as e:               # noqa: BLE001
                print(f"[LoRA Library] training card of {lora.stem}: {e!r}")
        head, settings = details_html(LIB, lora)
        prev = load_image(lora.preview) if lora.preview else None
        prev_upd = gr.update(value=prev, elem_classes=["ll-blur"] if LIB.blur(lora) else [])
        tips = usage_tips(lora)
        level, _ = lora.nsfw_level()
        ov = LIB.overrides(lora.key)
        trig = "\n".join(lora.trigger_words)
        notes = (lora.sidecar or {}).get("notes", "")
        strength = tips.get("strength")
        filled = []
        try:                                     # empty fields: filled from the training, saved only with Save
            sug = suggest_edit(lora)
        except Exception:                        # noqa: BLE001
            sug = {}
        if not trig.strip() and sug.get("trigger"):
            trig = sug["trigger"]
            filled.append("trigger word")
        notes = TC.strip_auto_notes(notes or "")          # the old training text is in the Training tab now
        use = None
        try:
            import insights
            use = insights.usage_of(lora)                # how you use it in your ComfyUI pictures
        except Exception:                        # noqa: BLE001
            pass
        if strength is None and use and use.get("strength") is not None and use["pictures"] >= 3:
            strength = round(float(use["strength"]), 2)
            filled.append(f"strength {strength}: the median of the {use['pictures']} pictures you made with it")
        elif strength is None and sug.get("strength") is not None:
            strength = sug["strength"]
            filled.append("strength")
        clip_s = tips.get("clip_strength")
        if clip_s is None and sug.get("clip_strength") is not None:
            clip_s = sug["clip_strength"]
            filled.append("CLIP strength")
        char = lora.character
        if not (char or "").strip() and sug.get("character"):
            char = sug["character"]
            filled.append("character")
        msg = ("✎ **Filled in from the training** (" + ", ".join(filled) + ": from the dataset, the LoRA file, its "
               "launch script and log). They are not saved yet: check them and press **Save** to keep them."
               if filled else "")
        return [prev_upd, head, settings, loss_frame(lora),
                lora.display_name, char, trig,
                notes,
                strength, clip_s,
                level if level in (0, 1, 2, 4, 8, 16) else 0,
                bool((lora.sidecar or {}).get("favorite")),
                ov.get("flag", "none"), ov.get("flag_reason", ""), msg,
                gr.update(choices=lora_choices(lora.gid), value=lora.gid),
                gr.update(choices=version_choices(lora), value=lora.key,
                          interactive=len(LIB.group_of(lora)) > 1),
                lora.key, done_checkbox(lora), gr.update(visible=False), gr.update(visible=True),
                (lora.sidecar or {}).get("civitai", {}).get("name") or (lora.vlabel if lora.version is not None else ""),
                tips.get("clip_skip") if tips.get("clip_skip") not in (None, "") else
                (int(TC.stored(lora.sidecar).get("clip_skip") or 1) if lora.sidecar else None),
                (lora.sidecar or {}).get("modelDescription", ""),
                gr.update(value=[str(p) for p in lora.examples[:40]] or None, visible=bool(lora.examples)),
                int(ov.get("rating") or 0)]

    def done_checkbox(lora):
        ds, t = done_key(lora)
        if not ds:
            return gr.update(value=False, interactive=False,
                             label="Completed (only for LoRAs with a dataset in your dataset folders)")
        inf = COMPLETED.info(ds, t, version=COMPLETED.version_of(lora.stem))
        how = (f" (set {'automatically' if inf.get('how') == 'auto' else 'by you'} {inf.get('at', '')})"
               if inf.get("done") else "")
        return gr.update(value=bool(inf.get("done")), interactive=True,
                         label=f"Completed: {ds} · {lora.base} {lora.vlabel}{how}")

    def toggle_done(key, value):
        lora = LIB.loras.get(key)
        if not lora:
            return gr.update(), ""
        ds, t = done_key(lora)
        if not ds:
            return gr.update(value=False), ""
        COMPLETED.set_done(ds, t, bool(value), how="manual", lora=lora.stem)
        msg = (f"**{ds} · {lora.base} {lora.vlabel}** marked Completed: the launch script of this version is hidden "
               "in Training › Run (press *Refresh* there). A new version starts unmarked." if value else
               f"**{ds} · {lora.base} {lora.vlabel}** is no longer Completed: its launch script shows in Training › Run "
               "again (press *Refresh* there).")
        return done_checkbox(lora), msg

    e_done.input(toggle_done, [cur_key, e_done], [e_done, e_done_msg]
                 ).then(do_render, filters, lib_html).then(do_gallery, gal_inputs, gal_html)

    def show_group(gid):
        newest = LIB.newest(gid)
        return show_details(newest.key if newest else "")

    # .input = only when you pick something yourself (not when the code updates the dropdowns)
    pick_dd.input(show_group, pick_dd, detail_outputs)
    ver_dd.input(show_details, ver_dd, detail_outputs)

    def save_edits(key, name, char, trig, notes, strength, clip, nsfw, fav, flag, reason, ver="", cskip=None, desc="",
                   rate=0):
        lora = LIB.loras.get(key)
        if not lora:
            return "Pick a LoRA first.", gr.update()
        data = copy.deepcopy(lora.sidecar) if lora.sidecar else new_sidecar(LIB, lora)
        data["model_name"] = (name or lora.stem).strip()
        civ = data.get("civitai") or {}
        data["civitai"] = civ
        civ["trainedWords"] = [t.strip() for t in (trig or "").splitlines() if t.strip()]
        data["notes"] = notes or ""
        if (ver or "").strip():
            civ["name"] = ver.strip()
        data["modelDescription"] = desc or ""
        tips = usage_tips(lora)
        if cskip is None or (isinstance(cskip, float) and math.isnan(cskip)):
            tips.pop("clip_skip", None)
        else:
            tips["clip_skip"] = int(cskip)
        for k, v in (("strength", strength), ("clip_strength", clip)):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                tips.pop(k, None)
            else:
                tips[k] = float(v)
        data["usage_tips"] = json.dumps(tips)
        data["favorite"] = bool(fav)
        data["preview_nsfw_level"] = int(nsfw or 0)
        target = lora.sidecar_path        # <dataset>/LoraManager/<model file>.metadata.json (LoRA folder untouched)
        if target is None:
            return "No place to save: this LoRA has no dataset.", gr.update()
        write_sidecar(target, data)

        ov = LIB.overrides(key)
        char = (char or "").strip()
        if char and char != lora.character:
            ov["character"] = char
        if rate:
            ov["rating"] = int(rate)
        else:
            ov.pop("rating", None)
        try:
            import insights
            insights.forget()                    # the Insights page and hints count the new rating
        except Exception:                        # noqa: BLE001
            pass
        if flag and flag != "none":
            ov["flag"], ov["flag_reason"] = flag, reason or ""
        else:
            ov.pop("flag", None)
            ov.pop("flag_reason", None)
        LIB.save_data()
        lora.sidecar, lora.sidecar_path, lora.sidecar_from_archive = data, target, False
        if char and LIB.dataset_by_name(char) and lora.dataset is None:
            lora.dataset = LIB.dataset_by_name(char)
        if ov.get("character"):
            lora.character = ov["character"]
        LIB._evaluate(lora)
        LIB.save_cache()
        return f"Saved to `{target}`.", gr.update(choices=lora_choices(lora.gid), value=lora.gid)

    def accept_warnings(key):
        lora = LIB.loras.get(key)
        if not lora:
            return gr.update(), "Open a LoRA first."
        ov = LIB.overrides(key)
        warn = [t for lvl, t in lora.issues if lvl == "warning"]
        if warn:
            ov["accepted"] = sorted(set(ov.get("accepted") or []) | set(warn))
            msg = f"Accepted: {'; '.join(warn)}. The orange mark is gone."
        elif ov.get("accepted"):
            ov.pop("accepted", None)
            msg = "The accepted warnings show again."
        else:
            return gr.update(), "This version has no warnings to accept" + (
                " (a red problem, such as a NaN loss, cannot be accepted: train it again)." if lora.issues else ".")
        LIB.save_data()
        LIB._evaluate(lora)
        LIB.save_cache()
        return details_html(LIB, lora)[0], msg

    d_accept.click(accept_warnings, cur_key, [d_head, d_accept_msg]
                   ).then(do_render, filters, lib_html).then(do_gallery, gal_inputs, gal_html)

    e_save.click(save_edits, [cur_key, e_name, e_char, e_trig, e_notes, e_str, e_clip, e_nsfw,
                              e_fav, e_flag, e_reason, e_ver, e_cskip, e_desc, e_rate], [e_msg, pick_dd]
                 ).then(do_render, filters, lib_html).then(do_gallery, gal_inputs, gal_html)

    def do_sync(q, b, st, g, progress=gr.Progress()):
        msg = personal_sync(LIB, progress)
        return (msg,) + tuple(list_updates(q, b, st, g))

    sync_btn.click(do_sync, filters, [sync_msg, lib_html, base_f, pick_dd, gal_base]
                   ).then(do_gallery, gal_inputs, gal_html)

    def lm_dir_store(d):
        d = (d or "").strip()
        if d and not Path(os.path.expanduser(d)).is_dir():
            return f"❌ Not a folder: `{d}`"
        LIB.data["lm_examples_dir"] = d
        LIB.save_data()
        _LM_ROOTS.clear()
        r = lm_example_roots(LIB)
        return (f"✔ Saved. Example pictures are read from `{r[0]}`. Restart LoRA Studio once so the LoRA page may show "
                "pictures from there, then press Sync now." if r else "✔ Saved (read from LoRA Manager's settings).")
    lm_dir_save.click(lm_dir_store, lm_dir, lm_dir_msg)

    def do_cards(force, progress=gr.Progress()):
        import trainingcard as TC
        return TC.scan_all(LIB, bool(force), progress)

    tc_btn.click(do_cards, tc_force, tc_msg)

    def do_insights(force=False):
        import insights
        return insights.page_html(insights.compute(LIB, force=bool(force)), esc)

    ins_tab.select(lambda: do_insights(False), None, ins_html)
    ins_btn.click(lambda: do_insights(True), None, ins_html)

    def rescan_one(key):
        import trainingcard as TC
        lora = LIB.loras.get(key)
        if lora is not None and lora.dataset is not None:
            TC.scan_one(LIB, lora, force=True)
        return show_details(key)

    d_rescan.click(rescan_one, cur_key, detail_outputs)

    # ---- copy to LoRA folder (Details)
    def details_copy_info(key):
        lora = LIB.loras.get(key)
        if HK is None or not lora or not lora.dataset_path:
            return "", "This LoRA has no file in a dataset's model folder."
        t = HK.copy_target(lora.stem, LIB.settings.get("lora_dirs") or [], lora.base)
        same = [f for f in HK.lora_files(LIB.settings.get("lora_dirs") or []) if f.name == lora.dataset_path.name]
        note = f"`{lora.dataset_path.name}` → **{t['reason']}**."
        if same:
            note += f" Already in `{same[0].parent}`" + (" (same size)." if same[0].stat().st_size ==
                                                          lora.dataset_path.stat().st_size else " (a DIFFERENT file).")
        return str(t["dir"] or ""), note

    def details_copy(key, dest, over):
        lora = LIB.loras.get(key)
        if HK is None or not lora or not lora.dataset_path:
            return "Pick a LoRA first."
        if not (dest or "").strip():
            return "Enter the LoRA folder to copy to."
        ok, msg = HK.copy_lora(lora.dataset_path, Path(os.path.expanduser(dest.strip())), bool(over))
        return ("✔ " if ok else "❌ ") + msg

    cur_key.change(details_copy_info, cur_key, [d_cp_dest, d_cp_note])
    d_cp_btn.click(details_copy, [cur_key, d_cp_dest, d_cp_over], d_cp_msg).then(lambda: False, None, d_cp_over)
    d_copy_jump.click(lambda: gr.Accordion(open=True), None, d_cp_acc)


    # ---- Gallery tab
    def pg_base(gid):
        n = LIB.newest(gid)
        return n.base if n else ""

    def pg_folder_note(gid):
        base = pg_base(gid)
        if not base:
            return "", ""
        folder, how = picture_folder(base)
        if folder:
            done = PG.index()["folders"].get(str(Path(folder).resolve()))
            when = (f" Last scanned {done['scanned']} ({done['pictures']} pictures)." if done
                    else " Not scanned yet: press **Scan for new pictures**.")
            return folder, f"{base} pictures: {how}.{when}"
        return "", (f"{how} **Enter where ComfyUI saves your {base} pictures** above and press *Scan for new "
                    "pictures*; the folder is remembered for this LoRA type.")

    def pg_show(gid, ver, page):
        html_, page, pages = render_pictures(gid, ver, page)
        return html_, page, (f"of {pages}" if pages > 1 else "")

    def pg_old_update(gid):
        ch = alias_suggestions(gid) if gid in LIB.groups else []
        return gr.update(choices=ch, value=[n for _, n in ch if n in set(picture_aliases(gid))])

    def pg_pick(gid):
        folder, note = pg_folder_note(gid)
        html_, page, pages = pg_show(gid, "", 1)
        return (folder, note, gr.update(choices=picture_versions(gid), value=""), html_, page, pages,
                None, "", "", pg_old_update(gid))

    pg_outs = [pg_folder, pg_msg, pg_ver, pg_html, pg_page, pg_pages, pg_img, pg_info, pg_prompt, pg_old]

    def pg_old_do(gid, picked, typed):
        if gid not in LIB.groups:
            return "Pick a LoRA first.", gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        names = list(picked or []) + [t for t in re.split(r"[,\n]", typed or "") if t.strip()]
        set_picture_aliases(gid, names)
        n = len(picture_aliases(gid))
        html_, page, pages = pg_show(gid, "", 1)
        return ((f"✔ {n} old name{'s' if n != 1 else ''} linked to this LoRA." if n else "No old names linked."),
                pg_old_update(gid), "", gr.update(choices=picture_versions(gid), value=""), html_, pages)

    pg_old_save.click(pg_old_do, [pg_lora, pg_old, pg_old_add], [pg_old_msg, pg_old, pg_old_add, pg_ver, pg_html, pg_pages])
    pg_lora.input(pg_pick, pg_lora, pg_outs)

    def pg_do_scan(gid, folder, progress=gr.Progress()):
        base = pg_base(gid)
        if not base:
            return "Pick a LoRA first.", gr.update(), gr.update()
        folder = os.path.expanduser((folder or "").strip())
        if not folder:
            return (f"**Where does ComfyUI save your {base} pictures?** Enter the folder above, then press the button "
                    "again.", gr.update(), gr.update())
        if not Path(folder).is_dir():
            return f"❌ The folder `{folder}` does not exist.", gr.update(), gr.update()
        auto, _ = picture_folder(base)
        saved = ""
        if not auto or Path(auto).resolve() != Path(folder).resolve():
            set_picture_dir(base, folder)
            LIB.settings["picture_dirs"] = load_settings().get("picture_dirs") or {}
            saved = f" The folder is remembered for {base}."
        st = PG.scan(folder, progress)
        if st["stopped"]:
            return (f"⏸ Stopped after {st['seconds']} s: {st['read']} pictures read and kept. Press *Scan for new "
                    f"pictures* to continue where it stopped.{saved}", gr.update(choices=picture_versions(gid), value=""), folder)
        return (f"✔ Scanned `{folder}` in {st['seconds']} s: {st['pictures']} pictures, {st['read']} new or changed, "
                f"{st['removed']} gone, {st['with_lora']} with LoRAs, {st['no_meta']} without metadata (skipped).{saved}",
                gr.update(choices=picture_versions(gid), value=""), folder)

    def pg_do_all(gid, progress=gr.Progress()):
        s_ = LIB.settings
        roots = [d for d in [s_.get("pictures_dir")] + list((s_.get("picture_dirs") or {}).values()) if d and Path(d).is_dir()]
        keep = []
        for r in sorted(set(str(Path(d).resolve()) for d in roots), key=len):
            if not any(r.startswith(k + os.sep) for k in keep):
                keep.append(r)
        if not keep:
            return ("No picture folders yet: set the **ComfyUI picture folder** in Settings > Folders, or enter a folder "
                    "above and press *Scan for new pictures*.", gr.update())
        lines, total = [], 0
        for r in keep:
            st = PG.scan(r, progress)
            if st["stopped"]:
                lines.append(f"`{r}`: stopped, {st['read']} read so far (press again to continue)")
                break
            total += st["with_lora"]
            lines.append(f"`{r}`: {st['pictures']} pictures, {st['read']} new or changed, {st['removed']} gone")
        return (f"✔ Scanned {len(keep)} folder(s), {total} pictures with LoRAs.  \n" + "  \n".join(lines),
                gr.update(choices=picture_versions(gid), value="") if gid else gr.update())

    pg_scan.click(pg_do_scan, [pg_lora, pg_folder], [pg_msg, pg_ver, pg_folder]
                  ).then(lambda g: pg_show(g, "", 1), pg_lora, [pg_html, pg_page, pg_pages]).then(pg_old_update, pg_lora, pg_old)
    pg_all.click(pg_do_all, pg_lora, [pg_msg, pg_ver]).then(lambda g: pg_show(g, "", 1), pg_lora, [pg_html, pg_page, pg_pages]
                 ).then(pg_old_update, pg_lora, pg_old)
    pg_stop.click(lambda: (PG.stop(), "Stopping ... (what was read so far is kept)")[1], None, pg_msg,
                  queue=False)
    pg_ver.input(lambda g, v: pg_show(g, v, 1), [pg_lora, pg_ver], [pg_html, pg_page, pg_pages])
    pg_page.submit(pg_show, [pg_lora, pg_ver, pg_page], [pg_html, pg_page, pg_pages])
    pg_prev.click(lambda g, v, p: pg_show(g, v, (p or 1) - 1), [pg_lora, pg_ver, pg_page], [pg_html, pg_page, pg_pages])
    pg_next.click(lambda g, v, p: pg_show(g, v, (p or 1) + 1), [pg_lora, pg_ver, pg_page], [pg_html, pg_page, pg_pages])

    def pg_open(v, gid):
        path = (v or "").rsplit("|", 1)[0]
        if not path:
            return gr.update(), gr.update(), gr.update()
        img, info, prompt = picture_details(path, gid)
        blur = LIB.newest(gid) is not None and LIB.blur(LIB.newest(gid))
        return gr.update(value=img, elem_classes=["ll-blur"] if blur else []), info, prompt

    pic_box.change(pg_open, [pic_box, pg_lora], [pg_img, pg_info, pg_prompt])   # .input is not fired by the JS

    def from_details(key):
        lora = LIB.loras.get(key)
        if not lora:
            return [gr.update()] * (len(pg_outs) + 2)
        gid = lora.gid
        return [gr.update(choices=lora_choices(gid), value=gid)] + list(pg_pick(gid)) + [gr.Tabs(selected="lp_pics")]

    def details_scan(gid, folder, progress=gr.Progress()):
        """Coming from Details: scan right away when the folder is known (only new pictures are read)."""
        if not folder:
            return gr.update(), gr.update(), gr.update()
        return pg_do_scan(gid, folder, progress)

    d_pics.click(from_details, cur_key, [pg_lora] + pg_outs + [lp_tabs]
                 ).then(details_scan, [pg_lora, pg_folder], [pg_msg, pg_ver, pg_folder]
                 ).then(lambda g: pg_show(g, "", 1), pg_lora, [pg_html, pg_page, pg_pages]).then(pg_old_update, pg_lora, pg_old)

    # ---- the LoRA page header, its versions and its pictures follow the open LoRA
    def lp_after(key):
        lora = LIB.loras.get(key)
        if not lora:
            return "", gr.update(choices=[], value=None), "", gr.update()
        ds = lora.dataset
        sib = [g for g, m in LIB.groups.items() if m and (
            (ds is not None and m[-1].dataset is not None and m[-1].dataset.name == ds.name)
            or (ds is None and m[-1].character.lower() == lora.character.lower()))]
        newest = {g: LIB.newest(g) for g in sib}
        choices = sorted(((newest[g].base if newest[g] else g), g) for g in sib)
        title = (f'<div class="lp-title"><b>{esc(lora.character or lora.display_name)}</b>'
                 f'<span class="ll-muted">{esc(lora.display_name)}</span></div>')
        return (title, gr.update(choices=choices, value=lora.gid, visible=len(choices) > 1), versions_html(lora),
                gr.update(choices=lora_choices(lora.gid), value=lora.gid))

    cur_key.change(lp_after, cur_key, [lp_title, lp_types, lp_versions, pg_lora]) \
        .then(lambda g: pg_pick(g) if g in LIB.groups else [gr.update()] * len(pg_outs), pg_lora, pg_outs)
    lp_types.input(show_group, lp_types, detail_outputs)
    lp_back.click(lambda: gr.Tabs(selected="gallery"), None, tabs)
    UI_HOOKS["lp_newv"] = (lp_newv, cur_key)

    # ---- Clean up tab
    def do_lo_scan():
        md, paths = leftovers_md(LIB)
        return md, paths, False

    def do_lo_go(paths, ok):
        if not ok:
            gr.Warning("Tick the confirmation first.")
            return "Tick the confirmation first.", paths, False
        if not paths:
            return "Press **Find leftovers** first.", paths, False
        busy = busy_datasets()
        todo = [p for p in paths if not any(Path(p).is_relative_to(b) for b in busy)]
        size = sum(HK.size_of(Path(p)) for p in todo)
        moved, err = HK.trash(todo)
        msg = f"✔ Moved {moved} item(s), {HK.human(size)}, to the Trash."
        if len(todo) < len(paths):
            msg += f" {len(paths) - len(todo)} item(s) skipped: a training run started in that dataset."
        if err:
            msg += "\n\nCould not move: " + "; ".join(err[:5])
        return msg, [], False

    lo_scan.click(do_lo_scan, None, [lo_md, lo_paths, lo_ok])
    lo_go.click(do_lo_go, [lo_paths, lo_ok], [lo_md, lo_paths, lo_ok])

    def do_or_scan():
        md, choices, values = orphans_list(LIB)
        return md, gr.update(choices=choices, value=values), False

    def do_or_go(picked, ok):
        if not ok:
            gr.Warning("Tick the confirmation first.")
            return "Tick the confirmation first.", gr.update(), False
        files = [f for group in (picked or []) for f in group.split("\x1f") if f]
        if not files:
            return "Nothing ticked.", gr.update(), False
        moved, err = HK.trash(files)
        md, choices, values = orphans_list(LIB)
        msg = f"✔ Moved {moved} file(s) / folder(s) to the Trash." + ("\n\nCould not move: " + "; ".join(err[:5]) if err else "")
        return msg + "\n\n" + md, gr.update(choices=choices, value=values), False

    or_scan.click(do_or_scan, None, [or_md, or_pick, or_ok])
    or_go.click(do_or_go, [or_pick, or_ok], [or_md, or_pick, or_ok])

    def do_save_settings(*vals, progress=gr.Progress()):
        old = dict(LIB.settings)
        write_settings(*vals)
        LIB.scan(progress)
        names = LIB.base_names()
        msg = "Settings saved and folders rescanned."
        if LIB.settings["sd_scripts_dir"] != old["sd_scripts_dir"]:
            msg += " **Restart the program** so the Training tabs use the new sd-scripts folder."
        if LIB.settings["style"] != old["style"]:
            msg += " The new style is used the next time the program opens (or use the Dark / Normal button)."
        return (msg, msg,
                gr.update(choices=personal_bases(), value=personal_bases()),
                gr.update(choices=names, value=[b for b in LIB.settings["expected_bases"] if b in names]),
                gr.update(choices=lora_choices()))

    st_inputs = [st_sd, st_ds, st_lora, st_scripts, st_sample, st_models, st_bases, st_expected, st_style,
                 st_blur, st_level, st_unknown, st_pics]
    for btn in (st_save, st_save2):
        btn.click(do_save_settings, st_inputs, [st_msg, st_msg2, base_f, st_expected, pick_dd]
                  ).then(do_render, filters, lib_html).then(do_gallery, gal_inputs, gal_html)


def build_app():
    style_in_launch = "css" in inspect.signature(gr.Blocks.launch).parameters
    theme, head = make_theme(), head_html(LIB.settings["style"])
    blocks_kwargs = {"title": "LoRA Library"}
    if not style_in_launch:
        blocks_kwargs.update(css=CSS, head=head, theme=theme)
    with gr.Blocks(**blocks_kwargs) as app:
        with gr.Row(elem_id="ls_top"):
            gr.Markdown("# LoRA Library")
            gr.Button("Dark / Normal", size="sm", scale=0).click(None, None, None, js=TOGGLE_JS)
        build_library_ui(app)
    launch_kwargs = {"css": CSS, "head": head, "theme": theme} if style_in_launch else {}
    return app, launch_kwargs


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="LoRA Library")
    ap.add_argument("--port", type=int, default=7870)
    ap.add_argument("--listen", action="store_true", help="Listen on all interfaces (LAN access)")
    ap.add_argument("--no-browser", action="store_true", help="Don't open a browser tab")
    args = ap.parse_args()
    app, extra = build_app()
    THUMB_DIR.mkdir(exist_ok=True)
    threading.Thread(target=lambda: (LIB.ensure_loaded(), warm_thumbnails(LIB)), daemon=True).start()
    app.launch(server_name="0.0.0.0" if args.listen else "127.0.0.1", server_port=args.port,
               inbrowser=not args.no_browser, allowed_paths=[str(THUMB_DIR)], **extra)
