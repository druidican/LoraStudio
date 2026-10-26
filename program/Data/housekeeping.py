#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""
housekeeping.py - file chores for LoRA Studio, without any UI:

  * copy a finished LoRA to the LoRA folder where its older versions are (or <LoRA folder>/<type>/ for a first one)
  * training leftovers of FINISHED LoRAs: epoch checkpoints, saved states, .npz caches, musubi cache folders
  * LoRA Manager meta files (.metadata.json, preview, example images) whose LoRA file no longer exists

Everything that is removed goes to the desktop's normal Trash (freedesktop.org: restorable from the file manager,
the space is freed when the Trash is emptied). `gio trash` is used when present, else the Trash spec directly.
"""
from __future__ import annotations
__version__ = "1.261026v"  # LoRA Studio version, see version.py

import hashlib
import os
import re
import shutil
import subprocess
import urllib.parse
from datetime import datetime
from pathlib import Path

LORA_EXT = (".safetensors", ".ckpt", ".pt")
EPOCH_RE = re.compile(r"-(?:step)?\d{3,}$", re.I)                      # mycharAnimaV2-000004
STATE_RE = re.compile(r"^(?P<name>.+?)(?:-(?:step)?\d{3,})?-state$", re.I)  # mycharAnimaV2-000004-state / mycharAnimaV2-state
VERSION_TAIL_RE = re.compile(r"[_\-\s]?[vV]\d{1,3}$")
PREVIEW_EXT = (".webp", ".png", ".jpg", ".jpeg", ".gif", ".mp4", ".webm")
IMG_EXT = (".webp", ".png", ".jpg", ".jpeg", ".gif", ".bmp")
ANCHOR_SUFFIXES = (".metadata.json", ".civitai.info")                  # proof that a stem is a LoRA Manager entry
SKIP_DIRS = {"venv", ".git", "__pycache__", "node_modules", ".thumbs"}


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.2f} {unit}"
        n /= 1024
    return f"{n} B"


def size_of(p: Path) -> int:
    try:
        if p.is_symlink():
            return 0
        if p.is_file():
            return p.stat().st_size
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and not f.is_symlink())
    except OSError:
        return 0


# ----------------------------------------------------------------------------------------------- trash
def _trash_dirs_for(p: Path) -> Path:
    """The freedesktop Trash that files on p's filesystem go to: the home Trash on the home filesystem, else
    <mount point>/.Trash-<uid> (moving 10 GB of states across disks would be slow and fill the home disk)."""
    home_trash = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share") / "Trash"
    try:
        home_dev = (home_trash if home_trash.exists() else Path.home()).stat().st_dev
        if p.stat().st_dev == home_dev:
            return home_trash
        top = p.resolve()
        while top.parent != top and top.parent.stat().st_dev == p.stat().st_dev:
            top = top.parent
        return top / f".Trash-{os.getuid()}"
    except OSError:
        return home_trash


def _own_trash(p: Path):
    trash = _trash_dirs_for(p)
    (trash / "files").mkdir(parents=True, exist_ok=True)
    (trash / "info").mkdir(parents=True, exist_ok=True)
    name, n = p.name, 1
    while (trash / "files" / name).exists() or (trash / "info" / (name + ".trashinfo")).exists():
        n += 1
        name = f"{p.stem}.{n}{p.suffix}" if p.is_file() else f"{p.name}.{n}"
    info = trash / "info" / (name + ".trashinfo")
    info.write_text("[Trash Info]\nPath=" + urllib.parse.quote(str(p.resolve())) + "\nDeletionDate="
                    + datetime.now().strftime("%Y-%m-%dT%H:%M:%S") + "\n", encoding="utf-8")
    try:
        os.rename(p, trash / "files" / name)
    except OSError:
        shutil.move(str(p), str(trash / "files" / name))


def trash(paths, progress=None) -> tuple[int, list[str]]:
    """Move files / folders to the desktop Trash. (moved, [error lines])."""
    paths = [Path(p) for p in paths]
    gio = shutil.which("gio")
    moved, errors = 0, []
    for i, p in enumerate(paths):
        if progress:
            progress(i, len(paths), p)
        if not p.exists() and not p.is_symlink():
            continue
        try:
            if gio:
                r = subprocess.run([gio, "trash", "--", str(p)], capture_output=True, text=True, timeout=600)
                if r.returncode != 0:
                    raise OSError((r.stderr or r.stdout).strip() or f"gio trash exit {r.returncode}")
            else:
                _own_trash(p)
            moved += 1
        except (OSError, subprocess.TimeoutExpired) as e:
            if gio:                                    # gio can refuse some mounts: try the spec directly
                try:
                    _own_trash(p)
                    moved += 1
                    continue
                except OSError as e2:
                    e = e2
            errors.append(f"{p}: {e}")
    return moved, errors


# ----------------------------------------------------------------------------------------------- copy to LoRA folder
def version_base(stem: str) -> str:
    """mycharAnimaV2-000004 -> mycharanima (version and epoch number removed, lower case)."""
    s = EPOCH_RE.sub("", stem)
    return VERSION_TAIL_RE.sub("", s).lower()


def version_of(stem: str) -> int:
    m = re.search(r"[vV](\d{1,3})$", EPOCH_RE.sub("", stem))
    return int(m.group(1)) if m else 0


def type_folder(label: str) -> str:
    """'Z-Image Turbo (musubi-tuner)' -> 'ZImageTurbo', 'Flux.2' -> 'Flux2', 'Illustrious' -> 'Illustrious'."""
    label = re.sub(r"\(.*?\)", "", label or "").strip() or "Other"
    return re.sub(r"[^A-Za-z0-9]+", "", label.title() if label.islower() else label) or "Other"


def lora_files(dirs) -> list[Path]:
    out = []
    for d in dirs:
        d = Path(os.path.expanduser(str(d)))
        if not d.is_dir():
            continue
        for root, dnames, fnames in os.walk(d):
            dnames[:] = [x for x in dnames if x not in SKIP_DIRS and not x.startswith(".")]
            out += [Path(root) / f for f in fnames if f.lower().endswith(LORA_EXT)]
    return out


def copy_target(stem: str, lora_dirs, type_label: str) -> dict:
    """{'dir', 'reason', 'siblings': [Path]} for a new LoRA file named <stem>."""
    base = version_base(stem)
    every = lora_files(lora_dirs)
    sib = [f for f in every if version_base(f.stem) == base and f.stem.lower() != stem.lower()]
    if sib:
        best = max(sib, key=lambda f: (version_of(f.stem), f.stat().st_mtime if f.exists() else 0))
        return {"dir": best.parent, "siblings": sorted(sib),
                "reason": f"next to {best.name} (an older version)"}
    same = [f for f in every if f.stem.lower() == stem.lower()]
    if same:
        return {"dir": same[0].parent, "siblings": [], "reason": f"where {same[0].name} already is"}
    dirs = [Path(os.path.expanduser(str(d))) for d in lora_dirs if str(d).strip()]
    if not dirs:
        return {"dir": None, "siblings": [], "reason": "no LoRA folder set (Settings > Folders)"}
    return {"dir": dirs[0] / type_folder(type_label), "siblings": [],
            "reason": f"first version: a subfolder for its type in your first LoRA folder"}


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_lora(src: Path, dest_dir: Path, overwrite: bool = False, name: str = "") -> tuple[bool, str]:
    """Copy src into dest_dir (created), as `name` when given. Same file already there: nothing to do. Different
    file with that name: only with overwrite (the old one goes to the Trash first)."""
    src, dest_dir = Path(src), Path(os.path.expanduser(str(dest_dir)))
    if not src.is_file():
        return False, f"LoRA file not found: {src}"
    dest = dest_dir / (name or src.name)
    if dest.exists():
        if dest.stat().st_size == src.stat().st_size and _sha256(dest) == _sha256(src):
            return True, f"Already there (identical): {dest}"
        if not overwrite:
            return False, (f"{dest} already exists and is a different file. Tick 'Replace a different file with the "
                           "same name' to put the old one in the Trash and copy the new one.")
        _, err = trash([dest])
        if err:
            return False, "Could not move the old file to the Trash: " + err[0]
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        tmp = dest_dir / (dest.name + ".part")
        shutil.copy2(src, tmp)
        if tmp.stat().st_size != src.stat().st_size:
            tmp.unlink(missing_ok=True)
            return False, "Copy incomplete (size differs): disk full?"
        os.replace(tmp, dest)
    except OSError as e:
        return False, f"Copy failed: {e}"
    return True, f"Copied to {dest} ({human(dest.stat().st_size)})"


# ----------------------------------------------------------------------------------------------- training leftovers
def _finals(model_dir: Path) -> set[str]:
    if not model_dir.is_dir():
        return set()
    return {f.stem for f in model_dir.iterdir()
            if f.is_file() and f.suffix.lower() in LORA_EXT and not EPOCH_RE.search(f.stem) and ".tmp" not in f.name}


def leftovers(dataset_roots, musubi_cache: Path | None = None, busy=()) -> dict:
    """What can go for every dataset whose LoRA has a final file. Nothing is touched.
    {'items': [(Path, kind, dataset name, size)], 'skipped': [(dataset name, why)]}
    - epoch checkpoints  <name>-000004.safetensors   when <name>.safetensors exists
    - saved states       <name>[-000004]-state/      when <name>.safetensors exists
    - temporary files    *.tmp.safetensors, *.part
    - latent caches      *.npz anywhere in the dataset (rebuilt by the next run)
    - musubi cache       <musubi-tuner>/cache/<dataset>/ (rebuilt by the next run)"""
    busy = {Path(b).resolve() for b in busy}
    items, skipped = [], []
    for root in sorted({Path(r) for r in dataset_roots}):
        name = root.name
        if root.resolve() in busy:
            skipped.append((name, "a training run is active in this dataset"))
            continue
        finals = _finals(root / "model")
        if not finals:
            skipped.append((name, "no finished LoRA yet (you may still resume it)"))
            continue
        found = []
        model = root / "model"
        for f in sorted(model.iterdir()) if model.is_dir() else []:
            if f.is_file() and f.suffix.lower() in LORA_EXT and EPOCH_RE.search(f.stem) \
                    and EPOCH_RE.sub("", f.stem) in finals:
                found.append((f, "epoch checkpoint"))
            elif f.is_file() and (f.name.endswith(".tmp.safetensors") or f.name.endswith(".part")):
                found.append((f, "temporary file"))
        for dpath, dnames, fnames in os.walk(root):
            dp = Path(dpath)
            keep = []
            for d in dnames:
                m = STATE_RE.match(d)
                if m and m.group("name") in finals:
                    found.append((dp / d, "saved state"))
                elif d.lower() not in ("loramanager",) and d not in SKIP_DIRS and not d.startswith("."):
                    keep.append(d)
            dnames[:] = keep
            found += [(dp / f, "latent cache (.npz)") for f in sorted(fnames) if f.lower().endswith(".npz")]
        if musubi_cache:
            mc = Path(musubi_cache) / name
            if mc.is_dir():
                found.append((mc, "musubi cache"))
        if not found:
            continue
        for p, kind in found:
            items.append((p, kind, name, size_of(p)))
    return {"items": items, "skipped": skipped}


# ----------------------------------------------------------------------------------------------- orphaned meta files
def _meta_groups(folder: Path) -> dict[str, list[Path]]:
    """{stem: [files and folders that belong to that LoRA Manager entry]} for the anchors in one folder."""
    try:
        names = os.listdir(folder)
    except OSError:
        return {}
    stems = {}
    for n in names:
        for suf in ANCHOR_SUFFIXES:
            if n.endswith(suf) or n.endswith(suf + ".bak"):
                stems[n[:n.rindex(suf)]] = []
    if not stems:
        return {}
    for n in names:
        p = folder / n
        for stem in stems:
            if not n.startswith(stem):
                continue
            rest = n[len(stem):]
            if rest.startswith(".") and (rest.endswith(ANCHOR_SUFFIXES) or rest.endswith(tuple(s + ".bak" for s in ANCHOR_SUFFIXES))
                                         or rest in (".sha256",) or rest.lower() in PREVIEW_EXT
                                         or rest.lower().startswith(".preview.") and rest.lower().endswith(PREVIEW_EXT)
                                         or (p.is_file() and rest.lower().endswith(IMG_EXT) and rest.count(".") >= 2)):
                stems[stem].append(p)
            elif rest == "_examples" and p.is_dir():          # Personal sync's example folder
                stems[stem].append(p)
    ex = folder / "example_images"
    for stem in stems:
        if (ex / stem).is_dir():
            stems[stem].append(ex / stem)
    return stems


def orphans(lora_dirs, dataset_roots) -> list[dict]:
    """LoRA Manager entries without their LoRA file. In a LoRA folder: no <stem>.safetensors next to the meta file.
    In a dataset's LoraManager/: no <stem>.safetensors in the dataset's model/ folder NOR in any LoRA folder.
    [{'where': 'LoRA folder'|'dataset', 'folder', 'stem', 'files': [Path], 'size', 'looked_for'}]"""
    out = []
    in_lora_dirs = {f.stem for f in lora_files(lora_dirs)}     # a dataset's meta stays while its LoRA is there
    for d in lora_dirs:
        d = Path(os.path.expanduser(str(d)))
        if not d.is_dir():
            continue
        for root, dnames, _ in os.walk(d):
            dnames[:] = [x for x in dnames if x not in SKIP_DIRS and not x.startswith(".") and x != "example_images"]
            folder = Path(root)
            for stem, files in _meta_groups(folder).items():
                if any((folder / (stem + e)).exists() for e in LORA_EXT):
                    continue
                out.append({"where": "LoRA folder", "folder": folder, "stem": stem, "files": sorted(files),
                            "size": sum(size_of(f) for f in files), "looked_for": f"{folder / stem}.safetensors"})
    for r in dataset_roots:
        r = Path(r)
        arch = r / "LoraManager"
        if not arch.is_dir():
            continue
        have = {f.stem for f in (r / "model").iterdir() if f.suffix.lower() in LORA_EXT} if (r / "model").is_dir() else set()
        for stem, files in _meta_groups(arch).items():
            if stem in have or stem in in_lora_dirs:
                continue
            out.append({"where": "dataset", "folder": arch, "stem": stem, "files": sorted(files),
                        "size": sum(size_of(f) for f in files), "looked_for": f"{r / 'model' / stem}.safetensors"})
    return out
