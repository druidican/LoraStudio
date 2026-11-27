#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""
lora_library.py - the Collection's data: a catalogue of your LoRAs and their datasets.

Expected dataset layout (one folder per dataset inside a "dataset folder"):

    <dataset folder>/
      Name/
        10_name/        images + .txt captions (repeat folder, any number of them)
        log/            TensorBoard logs (sd-scripts --logging_dir)
        model/          trained .safetensors (these are "your LoRAs")
        *.sh            launch scripts
        LoraManager/    LoRA Manager data (.metadata.json, preview, example images), filled by the Personal sync

Here: the settings (library_settings.txt), the readers (safetensors header, TensorBoard events, thumbnails), the data
model (Dataset, Lora), the scan (Library, LIB), version groups and Completed marks. Elsewhere: the HTML of cards,
tiles and tables in libcards.py, your pictures made with a LoRA in libpictures.py, LoRA Manager's files (sync,
leftovers) in libsync.py.

Training settings always come from the .safetensors metadata (read-only). Edits go to the LoRA Manager sidecar in the
dataset (LoraManager/<name>.metadata.json), never into the .safetensors file.
"""

from __future__ import annotations
__version__ = "1.271126b"  # LoRA Studio version, see version.py

import hashlib
import html
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


try:                                   # shared "Completed" marks (completed.py next to this file)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import completed as COMPLETED
except Exception:
    COMPLETED = None
try:                                   # copy to LoRA folder, Trash, leftovers, orphaned meta files
    import housekeeping as HK
except Exception:                      # noqa: BLE001
    HK = None
from PIL import Image, ImageOps

# --------------------------------------------------------------------------- #
# Files and constants
# --------------------------------------------------------------------------- #

import lspaths as _LP  # noqa: E402  (program/lspaths.py: ROOT, PROG, USER)
APP_DIR = _LP.USER                  # your settings, data and caches (userdata/)
SETTINGS_FILE = APP_DIR / "library_settings.txt"  # (version 1 used settings.txt, migrated once)
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
             ".thumbs", "sample", "samples", ARCHIVE_DIR, "_set_aside", "_compare"}   # _compare: Training/compare.py
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
    ("NoobAI V-pred", "noobvp, noobai_vpred, noobai-vpred, noob_vpred"),   # before NoobAI (EPS): "noobvp" holds "noob"
    ("NoobAI", "noobai, noob"),                 # before Illustrious: NoobAI is trained on top of it
    ("Illustrious", "illustrious, nova"),
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

# Extra folders with launch scripts (.sh) for the Queue, besides the dataset folders
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
    for new_name, kw in (("NoobAI V-pred", ["noobvp", "noobai_vpred", "noobai-vpred", "noob_vpred"]),
                         ("NoobAI", ["noobai", "noob"])):
        # settings from before NoobAI was its own type: in front of Illustrious, which no longer claims "noobai"
        if bases and not any(n.lower() == new_name.lower() for n, _ in bases):
            bases = [(n, [k for k in kws if not k.startswith("noob")]) if n.lower() == "illustrious" else (n, kws)
                     for n, kws in bases]
            at = next((i for i, (n, kws) in enumerate(bases)
                       if n.lower() in ("noobai", "illustrious") or "illustrious" in kws), len(bases))
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
    import model_guide
    try:
        return model_guide.read_header(path).get("__metadata__") or {}
    except Exception:                                  # noqa: BLE001
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
    """Link to the thumbnail (served from .thumbs, lspaths.file_url). The browser caches it, so pages stay small:
    pictures used to be embedded in the page itself and sent again on every page load and filter change."""
    tp = thumb_file(path, width)
    if not tp:
        return ""
    import lspaths
    return lspaths.file_url(tp)


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
    tokens = ["sdxl", "ill", "illustrious", "noob", "noobvp", "noobai", "ani", "anima", "pony", "flux", "zit", "zib", "zimage", "qwi", "f2k",
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
        """The version a card and the LoRA page show: the one marked best in a dataset's Results, else the newest."""
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
# Version groups, edits suggested from the training
# --------------------------------------------------------------------------- #


def suggest_edit(lora: Lora) -> dict:
    """What LoRA Studio knows about a LoRA it trained, for the empty fields of LoRA page › Edit: the trigger word (the
    dataset's), notes with the training (from the LoRA file, its launch script and its run log) and a strength."""
    import sd_script
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
            home = sd_script.script_home(sp)
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


# --------------------------------------------------------------------------- #
# Details: the strength from LoRA Manager's usage tips
# --------------------------------------------------------------------------- #


def usage_tips(lora: Lora) -> dict:
    raw = (lora.sidecar or {}).get("usage_tips") or "{}"
    try:
        tips = json.loads(raw) if isinstance(raw, str) else dict(raw)
        return tips if isinstance(tips, dict) else {}
    except Exception:
        return {}


# --------------------------------------------------------------------------- #
# Completed marks
# --------------------------------------------------------------------------- #


def mark_completed(gids, done) -> str:
    """Tick or untick Completed for every selected card (Library card = one dataset + LoRA type,
    Carddex tile = every LoRA type of one dataset)."""
    picked = [g for g in (gids.split(",") if isinstance(gids, str) else gids or []) if g]
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
    return msg + " The Queue shows it after *Refresh*."


LIB = Library()


