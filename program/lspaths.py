# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Where everything lives (from 1.261026 on: 1.011126 is the first release with this layout).

    LoraStudio/                 ROOT: start.sh, install.sh, lora_studio.py (the launcher), version.py, selftest.py,
                                README, LICENSE, NOTICE, requirements.txt, venv/, sd-scripts/, musubi-tuner/,
                                constraints.txt
    LoraStudio/program/         PROG: the program, by topic: Gallery/ Data/ Training/ Settings/ Other/,
                                plus manual/ and logo/
    LoraStudio/userdata/        USER: your settings, token, databases, caches, logs, reports, update backups

bootstrap() puts the program folders on sys.path, so every module imports the others by name as before.
migrate() moves an older install (everything in one folder) into this layout: your files to userdata/, the old
program files to userdata/update_backups/old_layout_<time>/. It runs at every start and does nothing once done.
"""
from __future__ import annotations
__version__ = "1.011126"  # LoRA Studio version, see version.py

import shutil
import sys
import time
from pathlib import Path

PROG = Path(__file__).resolve().parent
ROOT = PROG.parent
USER = ROOT / "userdata"
TOPICS = ("Gallery", "Data", "Training", "Settings", "Other")
try:                                    # every module may write to it at import (settings made on first use)
    USER.mkdir(exist_ok=True)
except OSError:
    pass

# your files: they move from the old single folder into userdata/
USER_FILES = ("library_settings.txt", "training_settings.txt", "settings.txt", "huggingtoken.txt", "completed.json",
              "library_data.json", "model_ids.json", "hash_cache.json", "picture_index.json", "launch_settings.json",
              "library_cache.pkl", "features.json", "notice_accepted.json", "batch_settings.json",
              "comfy_settings.json", "datacheck_cache.json", "custom.css", ".setup_done", "settings.txt.bak",
              "library_settings.txt.bak", "training_settings.txt.bak")
USER_DIRS = ("gui_logs", "gui_cache", ".thumbs", "reports", "update_backups", "comfy_workflows", "pictools_models",
             "restore_backups")
USER_GLOBS = ("lorastudio_check_*.md",)
# the program files of the old layout (they now live in program/<topic>/)
OLD_PROGRAM = ("lora_library.py picgallery.py completed.py model_guide.py datasets_ui.py dsflow.py captioner.py "
               "datacheck.py consistency.py picker.py trim.py pictools.py pictools_worker.py sd_prepare.py "
               "housekeeping.py sdgui.py sd_script.py sd_settings.py training_ui.py savepoint.py failcodes.py "
               "comfytest.py Lora_Datatypes.py Training_Recommendations.py settings_ui.py launchopts.py firstrun.py "
               "setup.py backup.py features.py notice.py venvcheck.py selfupdate.py machinecheck.py updater.py "
               "ollamactl.py statusbar.py studio_css.py lora_studio.css ticker.py report.py manual.py").split()
OLD_DIRS = ("manual", "logo")


def folders() -> list[str]:
    return [str(PROG)] + [str(PROG / t) for t in TOPICS]


def bootstrap() -> None:
    """The program folders first on sys.path (before the launcher's own folder, where an old copy could lie)."""
    for p in reversed(folders()):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)


def migrate(say=print) -> list[str]:
    """Move an old single-folder install into this layout. Returns what was moved (empty when nothing was to do)."""
    done = []
    try:
        USER.mkdir(exist_ok=True)
    except OSError:
        return done
    names = list(USER_FILES) + list(USER_DIRS)
    for g in USER_GLOBS:
        names += [p.name for p in ROOT.glob(g)]
    for n in names:
        src, dst = ROOT / n, USER / n
        if not src.exists() or src.is_symlink():
            continue
        try:
            if dst.exists():
                if src.is_dir():                        # both: merge the old folder's files in, then drop it
                    for f in src.rglob("*"):
                        t = dst / f.relative_to(src)
                        if f.is_file() and not t.exists():
                            t.parent.mkdir(parents=True, exist_ok=True)
                            shutil.move(str(f), str(t))
                    shutil.rmtree(src, ignore_errors=True)
                else:                                   # the userdata copy is newer: keep the old one aside
                    shutil.move(str(src), str(USER / f"{n}.old_layout"))
            else:
                shutil.move(str(src), str(dst))
            done.append(n)
        except OSError as e:
            say(f"[Layout] could not move {n}: {e}")
    old = [n for n in OLD_PROGRAM if (ROOT / n).is_file()] + [d for d in OLD_DIRS if (ROOT / d).is_dir()]
    if old:
        keep = USER / "update_backups" / f"old_layout_{time.strftime('%Y%m%d_%H%M%S')}"
        keep.mkdir(parents=True, exist_ok=True)
        for n in old:
            try:
                shutil.move(str(ROOT / n), str(keep / n))
                done.append(n)
            except OSError as e:
                say(f"[Layout] could not move the old {n}: {e}")
        shutil.rmtree(ROOT / "__pycache__", ignore_errors=True)
        say(f"[Layout] the old program files are in {keep.relative_to(ROOT)}/ (delete them when all works)")
    if done:
        say(f"[Layout] moved to the new layout: {len(done)} files and folders (your settings are in userdata/)")
    return done
