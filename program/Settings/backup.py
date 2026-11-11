# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Backup and restore of LoRA Studio's settings and databases (Settings > Backup, and the setup page).

A backup is one zip, LoraStudio-backup-<date>-<time>.zip, with
  library_settings.txt, training_settings.txt      folders, model files per LoRA type, Ollama, ...
  completed.json, library_data.json                 Completed marks, your edits, old LoRA names, flags
  hash_cache.json, model_ids.json                   file hashes and model identification (saves time)
  picture_index.json                                the Gallery's picture index
  launch_settings.json                              Settings > Launch options (port, login, start page, ...)
  batch_settings.json                               Datasets › Improvements › What Fix does
  huggingtoken.txt                                  only when "include the token" is ticked
  gui_logs/                                         training logs shown by the Monitor
  manifest.json                                     when and where it was made
Not in it: the venv, sd-scripts / musubi-tuner, your datasets, models, LoRAs and pictures (they stay where
they are) and caches that are rebuilt by themselves (scan cache, thumbnails).

Restoring on another computer or user: paths that started with the old home folder / old LoRA Studio
folder are changed to the new ones. The current files are kept in restore_backups/<time>/ first.
"""
from __future__ import annotations
__version__ = "1.111126j"  # LoRA Studio version, see version.py

import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

import lspaths as _LP  # noqa: E402  (program/lspaths.py: ROOT, PROG, USER)
APP_DIR = _LP.USER                   # the files a backup holds live in userdata/
FILES = ["library_settings.txt", "training_settings.txt", "completed.json", "library_data.json",
         "hash_cache.json", "model_ids.json", "picture_index.json", "datacheck_cache.json",
         "launch_settings.json", "batch_settings.json", "comfy_settings.json"]
TOKEN = "huggingtoken.txt"
DIRS = ["gui_logs"]
MAX_FILE = 200 << 20               # skip single files above 200 MB (a runaway log)
PREFIX = "LoraStudio-backup-"
DEFAULT_DIR = Path.home() / "LoraStudio-backups"
TEXT_EXT = {".txt", ".json", ".log", ".sh", ".toml"}


def tilde(p) -> str:
    p, home = str(p), str(Path.home())
    return "~" + p[len(home):] if p == home or p.startswith(home + os.sep) else p


def backup_dir() -> Path:
    """The folder saved in library_settings.txt ([folders] backup_dir), else ~/LoraStudio-backups."""
    # read straight from the file: importing lora_library would create a default settings file on a new install
    import re
    try:
        text = (APP_DIR / "library_settings.txt").read_text(encoding="utf-8")
        m = re.search(r"^\s*backup_dir\s*=\s*(\S.*)$", text, re.M)
        if m:
            return Path(os.path.expanduser(m.group(1).strip()))
    except OSError:
        pass
    return DEFAULT_DIR


def remember_dir(d: Path):
    try:
        import lora_library as ll
        ll.set_backup_dir(str(d))
    except Exception as e:
        print(f"[Backup] could not save the backup folder in the settings: {e!r}")


# --------------------------------------------------------------------------- #
# Make
# --------------------------------------------------------------------------- #

def make_backup(dest: str | Path | None = None, include_token: bool = True) -> tuple[Path, dict]:
    dest = Path(os.path.expanduser(str(dest))) if dest else backup_dir()
    dest.mkdir(parents=True, exist_ok=True)
    name = f"{PREFIX}{time.strftime('%Y%m%d-%H%M%S')}.zip"
    out = dest / name
    tmp = out.with_suffix(".zip.part")
    files = [f for f in FILES if (APP_DIR / f).is_file()]
    if include_token and (APP_DIR / TOKEN).is_file():
        files.append(TOKEN)
    extra = []
    for d in DIRS:
        for p in sorted((APP_DIR / d).rglob("*")) if (APP_DIR / d).is_dir() else []:
            if p.is_file() and p.stat().st_size <= MAX_FILE:
                extra.append(p.relative_to(APP_DIR).as_posix())
    manifest = {"app": "LoRA Studio", "kind": "settings-backup", "version": 1,
                "created": time.strftime("%Y-%m-%d %H:%M:%S"), "home": str(Path.home()), "app_dir": str(_LP.ROOT),
                "user": os.environ.get("USER", ""), "files": files + extra, "token": TOKEN in files}
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(manifest, indent=1))
        for f in files + extra:
            z.write(APP_DIR / f, f)
    tmp.replace(out)
    if manifest["token"]:
        os.chmod(out, 0o600)                    # holds the Hugging Face token: readable only by you
    return out, manifest


# --------------------------------------------------------------------------- #
# Find / read
# --------------------------------------------------------------------------- #

def read_manifest(path: str | Path) -> dict | None:
    try:
        with zipfile.ZipFile(path) as z:
            m = json.loads(z.read("manifest.json"))
        return m if m.get("app") == "LoRA Studio" else None
    except Exception:
        return None


def find_backups(extra_dirs: list | None = None) -> list[tuple[str, str]]:
    """Dropdown choices [(label, path)], newest first: the backup folder, ~/LoraStudio-backups, Downloads, ~."""
    dirs = [backup_dir(), DEFAULT_DIR, Path.home() / "Downloads", Path.home()] + [Path(d) for d in extra_dirs or []]
    seen, out = set(), []
    for d in dirs:
        try:
            cands = list(d.glob(f"{PREFIX}*.zip")) if d.is_dir() else []
        except OSError:
            continue
        for p in cands:
            if p in seen:
                continue
            seen.add(p)
            m = read_manifest(p)
            if not m:
                continue
            label = (f"{m.get('created', '?')}  ·  {tilde(p.parent)}  ·  {len(m.get('files', []))} files"
                     f"{'  ·  with token' if m.get('token') else ''}")
            out.append((label, str(p), p.stat().st_mtime))
    out.sort(key=lambda t: -t[2])
    return [(a, b) for a, b, _ in out]


def describe(path: str | Path) -> str:
    m = read_manifest(path)
    if not m:
        return f"`{path}` is not a LoRA Studio backup."
    names = [f for f in m.get("files", []) if "/" not in f]
    logs = len(m.get("files", [])) - len(names)
    s = f"Backup from **{m.get('created')}**, made in `{m.get('app_dir')}`: " + ", ".join(names)
    if logs:
        s += f", {logs} training log file{'s' if logs != 1 else ''}"
    s += "." + ("" if m.get("token") else " No Hugging Face token in it (add it later in Settings > Training).")
    return s


# --------------------------------------------------------------------------- #
# Restore
# --------------------------------------------------------------------------- #

def _remap(text: str, pairs: list[tuple[str, str]]) -> str:
    for old, new in pairs:
        if old and old != new:
            text = text.replace(old + "/", new + "/").replace(old + '"', new + '"').replace(old + "\n", new + "\n")
    return text


def restore(path: str | Path) -> str:
    """Put the backup's files into this LoRA Studio folder. Returns a summary (markdown)."""
    path = Path(os.path.expanduser(str(path)))
    m = read_manifest(path)
    if not m:
        raise ValueError(f"{path} is not a LoRA Studio backup")
    allowed = set(FILES) | {TOKEN}
    keep = APP_DIR / "restore_backups" / time.strftime("%Y%m%d-%H%M%S")
    old_app, old_home = m.get("app_dir", ""), m.get("home", "")
    pairs = {(old_app, str(_LP.ROOT)), (old_home, str(Path.home()))}
    if old_app and old_home and old_app.startswith(old_home + "/"):
        pairs.add(("~" + old_app[len(old_home):], tilde(_LP.ROOT)))     # settings keep paths as ~/LoraStudio/...
    pairs = sorted(pairs, key=lambda t: -len(t[0]))     # the longer (more specific) path first
    done = []
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            name = info.filename
            if name == "manifest.json" or info.is_dir():
                continue
            ok = name in allowed or any(name.startswith(d + "/") and ".." not in name for d in DIRS)
            if not ok or name.startswith("/") or ".." in Path(name).parts:
                continue                           # only the files LoRA Studio writes; nothing outside its folder
            target = APP_DIR / name
            if target.exists():
                (keep / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, keep / name)
            target.parent.mkdir(parents=True, exist_ok=True)
            data = z.read(info)
            if target.suffix in TEXT_EXT:
                data = _remap(data.decode("utf-8", "replace"), pairs).encode("utf-8")
            target.write_bytes(data)
            if name == TOKEN:
                os.chmod(target, 0o600)
            done.append(name)
    (APP_DIR / ".setup_done").write_text(f"restored from {path} {time.strftime('%Y-%m-%d %H:%M')}\n")
    for stale in ("library_cache.pkl",):        # rebuilt from the restored settings
        try:
            (APP_DIR / stale).unlink()
        except OSError:
            pass
    moved = [f"`{a}` → `{b}`" for a, b in pairs if a and a != b]
    top = [d for d in done if "/" not in d]
    return (f"✔ Restored {len(top)} files" + (f" and {len(done) - len(top)} log files" if len(done) > len(top) else "")
            + f" from `{path.name}`." + (" Paths changed: " + ", ".join(moved) + "." if moved else "")
            + (f" The files that were here are in `{keep}`." if keep.exists() else ""))


# --------------------------------------------------------------------------- #
# Restart the program (after a restore or the setup page)
# --------------------------------------------------------------------------- #

def restart_program(delay: float = 1.5):
    """Start lora_studio.py again in the same process (same port); the open page reloads by itself."""
    import threading

    def go():
        time.sleep(delay)
        args = [a for a in sys.argv[1:] if a != "--setup"]
        if "--no-browser" not in args:
            args.append("--no-browser")
        script = str(_LP.ROOT / "lora_studio.py")
        print("[LoRA Studio] restarting with the new settings ...", flush=True)
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        os.execv(sys.executable, [sys.executable, script] + args)

    threading.Thread(target=go, daemon=True).start()


# --------------------------------------------------------------------------- #
# Settings > Backup
# --------------------------------------------------------------------------- #

def build_ui():
    import gradio as gr
    gr.Markdown("A backup is one zip with your settings and LoRA Studio's own databases: folders, model files per "
                "LoRA type, Completed marks, your edits in Details, old LoRA names, the Gallery's picture index, file "
                "hashes and training logs. Your datasets, models, LoRAs and pictures are **not** in it (they stay "
                "where they are). Restore it here or on the setup page of a new install.")
    with gr.Row(equal_height=True):
        folder = gr.Textbox(label="Backup folder (remembered)", value=tilde(backup_dir()), scale=4)
        tok = gr.Checkbox(label="Include the Hugging Face token (keep the backup private)", value=True, scale=2)
        go = gr.Button("Back up now", variant="primary", scale=1, min_width=140)
    msg = gr.Markdown()
    with gr.Accordion("Restore from a backup", open=False):
        with gr.Row(equal_height=True):
            pick = gr.Dropdown(label="Backups found", choices=find_backups(), interactive=True, scale=4)
            refresh = gr.Button("Refresh", size="sm", scale=0, min_width=90)
        path = gr.Textbox(label="Or the path of a backup .zip")
        info = gr.Markdown()
        with gr.Row(equal_height=True):
            ok = gr.Checkbox(label="Yes, replace the current settings with this backup (they are kept in "
                                   "restore_backups/ first); LoRA Studio restarts", value=False, scale=4)
            rs = gr.Button("Restore", variant="stop", scale=1, min_width=120)
        rmsg = gr.Markdown()

    def do_backup(d, t):
        d = os.path.expanduser((d or "").strip()) or str(DEFAULT_DIR)
        try:
            out, man = make_backup(d, bool(t))
        except Exception as e:
            return f"❌ Could not write the backup to `{d}`: {e}", gr.update()
        remember_dir(Path(d))
        size = out.stat().st_size / 1024
        return (f"✔ Saved `{out}` ({size:,.0f} KB, {len(man['files'])} files"
                f"{', with your Hugging Face token: keep it private' if man['token'] else ''}). "
                f"The folder is remembered for next time.", gr.update(choices=find_backups([d])))

    def show(p):
        return describe(p) if p else ""

    def do_restore(p, typed, yes):
        src = (typed or "").strip() or p
        if not src:
            return "Pick a backup or enter its path."
        if not yes:
            return "Tick the confirmation first."
        try:
            text = restore(src)
        except Exception as e:
            return f"❌ {e}"
        restart_program()
        return text + "\n\n**LoRA Studio restarts now**; this page reloads by itself in a few seconds."

    go.click(do_backup, [folder, tok], [msg, pick])
    refresh.click(lambda d: gr.update(choices=find_backups([os.path.expanduser(d or "")])), folder, pick)
    pick.change(show, pick, info)
    path.change(lambda t: describe(t.strip()) if t and t.strip() else "", path, info)
    rs.click(do_restore, [pick, path, ok], rmsg)
