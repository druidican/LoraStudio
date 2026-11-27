# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""dsfiles.py - the dataset folders and their files, shared by every part: which folders are datasets (list_lora_dirs,
is_dataset_dir), the repeat folders (N_name), the pictures and captions in them, thumbnails, and the LoRA type
presets file (Lora_Datatypes.py, load_scan_module).
"""
from __future__ import annotations
__version__ = "0"  # LoRA Studio version, see version.py

import hashlib
import importlib.util
import os
from pathlib import Path

import sdgui as S


# --------------------------------------------------------------------------- dataset tab
DATATYPES_FILE = "Lora_Datatypes.py"   # ALL LoRA types (PRESETS) live in this file


def find_scan_script(explicit):
    """Lora_Datatypes.py: the given path, else next to sdgui.py, in <sd-scripts>, or in your home folder."""
    cands = []
    if explicit:
        cands.append(Path(os.path.expanduser(explicit)))
    for d in (Path(S.__file__).resolve().parent, S.ROOT, S.HOME):
        cands.append(d / DATATYPES_FILE)
    for c in cands:
        if c.is_file():
            return c
    return None


_SCAN_MOD: dict = {}


def load_scan_module():
    """Lora_Datatypes.py, read again only when the file changed, so edits to PRESETS apply without a restart
    (it used to be read on every call: dozens of times for one page)."""
    if S.SCAN_SCRIPT is None:
        raise FileNotFoundError("Lora_Datatypes.py not found. Put it next to sdgui.py or pass --datatypes.")
    st = Path(S.SCAN_SCRIPT).stat()
    key = (str(S.SCAN_SCRIPT), st.st_mtime_ns, st.st_size)
    if _SCAN_MOD.get("key") != key:
        spec = importlib.util.spec_from_file_location("Lora_Datatypes", str(S.SCAN_SCRIPT))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _SCAN_MOD.update(key=key, mod=mod)
    return _SCAN_MOD["mod"]


PROGRAM_MARKS = ("lora_studio.py", "sdgui.py", "lora_library.py", "start.sh", "install.sh", "setup.py", "pyproject.toml",
                 "requirements.txt", ".git")


def is_dataset_dir(p: Path) -> bool:
    """A folder in the dataset folder that is a dataset: pictures (in it or one level down), a repeat folder, launch
    scripts or LoRA Studio's own notes. Program folders (a copy of LoRA Studio, a git checkout, a venv) are not."""
    if not p.is_dir() or p.name.startswith(".") or p.name.lower() in S.IGNORE_DIRS:
        return False
    try:
        names = {c.name for c in p.iterdir()}
    except OSError:
        return False
    if any(m in names for m in PROGRAM_MARKS) or "pyvenv.cfg" in names:
        return False
    if ".lsflow" in names or any(n.endswith(".sh") for n in names) or any(S.REPEAT_RE.match(n) for n in names):
        return True
    if any(Path(n).suffix.lower() in S.IMAGE_EXTS for n in names):
        return True
    for c in p.iterdir():                       # pictures one level down (a dataset without a repeat folder yet)
        if c.is_dir() and not c.name.startswith(".") and c.name.lower() not in S.IGNORE_DIRS:
            try:
                if any(Path(n).suffix.lower() in S.IMAGE_EXTS for n in os.listdir(c)):
                    return True
            except OSError:
                pass
    return False


def list_lora_dirs():
    img = S.images_dir()
    return sorted(p.name for p in img.iterdir() if is_dataset_dir(p)) if img.is_dir() else []


# --------------------------------------------------------------------------- shared dataset helpers
def lora_path(name) -> Path:
    return S.images_dir() / name


def repeat_folders(lora_dir: Path):
    """[(folder, repeats)] using the same rules as the scan script (repeats=None: not a '<N>_name'
    folder, so sd-scripts ignores it). Skips model/output/sample folders."""
    m = S.REPEAT_RE.match(lora_dir.name)
    if m:
        return [(lora_dir, int(m.group(1)))]
    found, loose = [], []
    for dirpath, dirnames, filenames in os.walk(lora_dir):
        p = Path(dirpath)
        mm = S.REPEAT_RE.match(p.name)
        if p != lora_dir and mm:
            found.append((p, int(mm.group(1))))
            dirnames[:] = []
            continue
        dirnames[:] = sorted(d for d in dirnames if d.lower() not in S.IGNORE_DIRS and not d.startswith(".")
                             and not S.ARCHIVE_RENAMED_RE.match(d))
        if any(Path(f).suffix.lower() in S.IMAGE_EXTS for f in filenames):
            loose.append((p, None))
    return found + loose


def renamed_archives(lora_dir: Path):
    """LoraManager folders (Personal sync: previews / example images) that were renamed to <N>_LoraManager.
    sd-scripts and musubi would train on those pictures, so they are left out of the plan and reported."""
    try:
        return sorted(d for d in lora_dir.iterdir() if d.is_dir() and S.ARCHIVE_RENAMED_RE.match(d.name))
    except OSError:
        return []


def list_images(folder: Path):
    try:
        return sorted((f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in S.IMAGE_EXTS),
                      key=lambda f: f.name.lower())
    except OSError:
        return []


def caption_file(img: Path) -> Path:
    for ext in S.CAPTION_EXTS:
        c = img.with_name(img.stem + ext)
        if c.is_file():
            return c
    return img.with_name(img.stem + ".txt")


def read_caption(img: Path):
    c = caption_file(img)
    if not c.is_file():
        return None
    return c.read_text(encoding="utf-8", errors="ignore").strip()


def thumb(img: Path, size: int = 384) -> str:
    """Cached JPEG thumbnail (keeps the gallery fast with 1024px PNGs)."""
    try:
        from PIL import Image, ImageOps
    except ImportError:
        return str(img)
    try:
        st = img.stat()
        key = hashlib.sha1(f"{img}|{st.st_mtime_ns}|{st.st_size}".encode()).hexdigest()[:20]
        dst = S.CACHE_DIR / "thumbs" / f"{key}.jpg"
        if not dst.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(img) as im:
                im = ImageOps.exif_transpose(im)
                im.thumbnail((size, size))
                if im.mode != "RGB":
                    im = im.convert("RGB")
                im.save(dst, "JPEG", quality=85)
        return str(dst)
    except Exception:
        return str(img)


def split_tags(text: str):
    return [t.strip() for t in (text or "").replace("\n", ",").split(",") if t.strip()]
