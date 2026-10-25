#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""
LoRA Studio first-time setup (run by start.sh on the first start, or with ./start.sh --setup).

Asks in the terminal:
  1. an existing library_settings.txt to import           (Enter = no)
  2. an existing training settings file to import          (Enter = no, a new one is created)
     (sd-scripts' settings.txt or a training_settings.txt)
  3. the Hugging Face token: a huggingtoken.txt to import, or paste the token   (Enter = empty file)
  4. folders, when no library settings were imported: sd-scripts, datasets, LoRAs
     (defaults are subfolders of your home folder, saved as ~ so they follow the logged-in user)
  5. the model folder for training model files (base model, text encoder, VAE)

Imported files are copied into this folder; the originals are not changed. Nothing is overwritten
without asking. Without a terminal (no keyboard), the defaults are used.
"""

from __future__ import annotations
__version__ = "1.251026g"  # LoRA Studio version, see version.py

import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIBRARY_SETTINGS = HERE / "library_settings.txt"
TRAINING_SETTINGS = HERE / "training_settings.txt"
HF_TOKEN_FILE = HERE / "huggingtoken.txt"
HOME = Path.home()
DATA_HOME = HOME / "LoraStudio-data"
INTERACTIVE = sys.stdin.isatty()

TRAINING_HEADER = (
    "# sdgui settings: model, VAE and text-encoder files for each LoRA type.\n"
    "# Written by sdgui.py (Settings tab); safe to edit by hand.\n"
    "# One [section] per LoRA type, `flag = path` lines, and `<flag>.dir` remembers the folder shown in the file list.\n"
)
TOKEN_HEADER = "# Put your Hugging Face token (hf_...) on its own line below. Lines starting with # are ignored.\n"


# ----------------------------------------------------------------------------- helpers
def say(text=""):
    print(text, flush=True)


def ask(prompt: str, default: str = "") -> str:
    if not INTERACTIVE:
        return default
    shown = f" [{default}]" if default else ""
    try:
        val = input(f"{prompt}{shown}: ").strip()
    except EOFError:
        return default
    return val or default


def yes(prompt: str, default: bool) -> bool:
    ans = ask(f"{prompt} ({'Y/n' if default else 'y/N'})", "y" if default else "n").lower()
    return ans.startswith("y")


def expand(path: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(path.strip().strip('"').strip("'")))).absolute()


def tilde(path) -> str:
    p = str(path)
    home = str(HOME)
    return "~" + p[len(home):] if p == home or p.startswith(home + os.sep) else p


def pick_file(prompt: str, names: tuple[str, ...]) -> Path | None:
    """Ask for a file (or a folder containing one of `names`). Enter = none."""
    while True:
        raw = ask(prompt)
        if not raw:
            return None
        p = expand(raw)
        if p.is_dir():
            found = next((p / n for n in names if (p / n).is_file()), None)
            if found:
                return found
            say(f"  No {' / '.join(names)} in {p}. Try again, or press Enter to skip.")
            continue
        if p.is_file():
            return p
        say(f"  {p} does not exist. Try again, or press Enter to skip.")


def may_write(path: Path) -> bool:
    """True if `path` may be (over)written. Existing files are only replaced after asking."""
    if not path.exists():
        return True
    if yes(f"  {path.name} already exists. Replace it (a .bak copy is kept)?", False):
        shutil.copy2(path, path.with_name(path.name + ".bak"))
        return True
    return False


def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


# ----------------------------------------------------------------------------- steps
def step_library() -> bool:
    """Returns True if library settings were imported (then no folder questions are needed)."""
    say("1/4  Existing LoRA Studio library settings")
    say("     A library_settings.txt (or the old Library's settings.txt) from an earlier install.")
    src = pick_file("     Path to the file or its folder (Enter = no, start fresh)",
                    ("library_settings.txt", "settings.txt"))
    if not src:
        return False
    text = read_text(src)
    if "[folders]" not in text:
        say(f"     {src.name} has no [folders] section, so it is not a LoRA Studio library settings file. Skipped.")
        return False
    if may_write(LIBRARY_SETTINGS):
        shutil.copyfile(src, LIBRARY_SETTINGS)
        say(f"     Imported {src}")
        return True
    return LIBRARY_SETTINGS.exists()


def step_training():
    say()
    say("2/4  Existing training settings")
    say("     sd-scripts' settings.txt (from sdgui) or a training_settings.txt: model files per LoRA type,")
    say("     venv, GPU / ROCm environment, subtypes.")
    src = pick_file("     Path to the file or its folder (Enter = no, create a new one)",
                    ("training_settings.txt", "settings.txt"))
    if src:
        text = read_text(src)
        if "[folders]" in text and "[general]" not in text:
            say("     That looks like library settings, not training settings. Skipped.")
            src = None
        elif may_write(TRAINING_SETTINGS):
            shutil.copyfile(src, TRAINING_SETTINGS)
            say(f"     Imported {src}")
            return
        else:
            return
    if not TRAINING_SETTINGS.exists():
        TRAINING_SETTINGS.write_text(TRAINING_HEADER, encoding="utf-8")
        say(f"     Created a new {TRAINING_SETTINGS.name}")


def step_token():
    say()
    say("3/4  Hugging Face token (only needed for gated models)")
    raw = ask("     Path to huggingtoken.txt or its folder, or paste the token itself (Enter = none)")
    token = None
    if raw.startswith("hf_") and " " not in raw:
        token = raw
    elif raw:
        p = expand(raw)
        if p.is_dir():
            p = p / "huggingtoken.txt"
        if p.is_file():
            if may_write(HF_TOKEN_FILE):
                shutil.copyfile(p, HF_TOKEN_FILE)
                say(f"     Imported {p}")
            return
        say(f"     {p} not found.")
    if token:
        if may_write(HF_TOKEN_FILE):
            HF_TOKEN_FILE.write_text(TOKEN_HEADER + token + "\n", encoding="utf-8")
            os.chmod(HF_TOKEN_FILE, 0o600)
            say(f"     Saved the token in {HF_TOKEN_FILE.name} (readable only by you)")
        return
    if not HF_TOKEN_FILE.exists():
        HF_TOKEN_FILE.write_text(TOKEN_HEADER, encoding="utf-8")
        os.chmod(HF_TOKEN_FILE, 0o600)
        say(f"     Created an empty {HF_TOKEN_FILE.name}; put your token in it later if you need one")


def step_folders(imported: bool):
    say()
    say("4/4  Folders  (Enter accepts the value in [brackets]; ~ means your home folder)")
    sys.path.insert(0, str(HERE))
    import lora_library as ll   # noqa: E402  (imported here: it reads/creates library_settings.txt)

    current = ll.load_settings()
    if imported:
        values = {**ll.default_settings_values(), **current}
        values["bases"] = [(n, ", ".join(k)) for n, k in current["bases"]]
        if current.get("models_dir"):
            say("     Folders come from the imported library settings. Change them later in Settings > Folders.")
            return
    else:
        d = ll.default_settings_values()
        sd = ask("     sd-scripts folder", tilde(d["sd_scripts_dir"]))
        ds = ask("     Dataset folder (one folder per LoRA dataset)", tilde(d["dataset_dirs"][0]))
        lo = ask("     LoRA folder (your finished LoRAs, LoRA Manager files)", tilde(d["lora_dirs"][0]))
        values = {**d, "sd_scripts_dir": str(expand(sd)), "dataset_dirs": [str(expand(ds))],
                  "lora_dirs": [str(expand(lo))]}

    say()
    say("     Model folder: where your base models, text encoders and VAEs are. Model-file slots in")
    say("     Settings > Training that have no folder yet start here.")
    md = ask("     Model folder", tilde(values.get("models_dir") or ll.default_settings_values()["models_dir"]))
    values["models_dir"] = str(expand(md))

    missing = [Path(p) for p in values["dataset_dirs"] + values["lora_dirs"] + [values["models_dir"]]
               if p and not Path(p).exists()]
    if missing:
        say()
        say("     These folders do not exist yet:")
        for m in missing:
            say(f"       {tilde(m)}")
        if yes("     Create them now?", True):
            for m in missing:
                m.mkdir(parents=True, exist_ok=True)
    if not Path(values["sd_scripts_dir"]).is_dir():
        say(f"     Note: {tilde(values['sd_scripts_dir'])} does not exist. Training needs sd-scripts there "
            "(or change the folder in Settings > Folders).")
    LIBRARY_SETTINGS.write_text(ll.render_settings(values), encoding="utf-8")
    say(f"     Saved {LIBRARY_SETTINGS.name}")


def main():
    say()
    say("=" * 70)
    say(" LoRA Studio setup")
    say("=" * 70)
    if not INTERACTIVE:
        say(" No terminal to ask questions in: using the defaults. Run ./start.sh --setup from a terminal")
        say(" to import existing settings or choose folders.")
    say(" Press Enter to skip a question or accept the value in [brackets].")
    say()
    imported = step_library()
    step_training()
    step_token()
    step_folders(imported)
    say()
    say(" Setup done. Everything can be changed later in the Settings tab.")
    say("=" * 70)
    say()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        say("\n Setup cancelled. Run ./start.sh --setup to do it again.")
        sys.exit(1)
