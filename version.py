# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""The version of LoRA Studio and of each of its parts.

Form: <release>.<ddmmyy>[letter]
    1.241026   release 1, made 24 Oct 2026 (the first version with numbers)
    1.051126b  a later update of release 1: the date of that update and a letter at the end (b, then c, ...)
Every part carries the version of the update that last CHANGED it: __version__ in the Python files, a "LoRA Studio
version" line in the .css and .sh files. A part that an update does not touch keeps its older version, so
Settings › Version shows what each update changed. RELEASE below says which version each part must have in this
release; a part with another version (or missing) was left behind by a half-done copy and is shown with a warning.
"""
from __future__ import annotations

import re
from pathlib import Path

__version__ = "1.251026m"
VERSION = "1.251026m"
# the version each part has in this release (written by the release script; a part not listed: VERSION)
RELEASE: dict = {
    "lora_studio.py": "1.251026i",
    "version.py": "1.251026m",
    "sdgui.py": "1.251026g",
    "sd_script.py": "1.251026g",
    "sd_prepare.py": "1.251026g",
    "sd_settings.py": "1.251026g",
    "datasets_ui.py": "1.251026g",
    "dsflow.py": "1.251026g",
    "training_ui.py": "1.251026g",
    "settings_ui.py": "1.251026k",
    "lora_library.py": "1.251026g",
    "comfytest.py": "1.251026g",
    "captioner.py": "1.251026g",
    "ollamactl.py": "1.251026g",
    "datacheck.py": "1.251026g",
    "consistency.py": "1.251026g",
    "pictools.py": "1.251026g",
    "pictools_worker.py": "1.251026g",
    "picgallery.py": "1.251026g",
    "picker.py": "1.251026g",
    "trim.py": "1.251026g",
    "housekeeping.py": "1.251026g",
    "completed.py": "1.251026g",
    "model_guide.py": "1.251026g",
    "ticker.py": "1.251026g",
    "statusbar.py": "1.251026g",
    "savepoint.py": "1.251026g",
    "launchopts.py": "1.251026g",
    "firstrun.py": "1.251026g",
    "setup.py": "1.251026g",
    "updater.py": "1.251026g",
    "backup.py": "1.251026g",
    "selftest.py": "1.251026m",
    "machinecheck.py": "1.251026h",
    "selfupdate.py": "1.251026g",
    "failcodes.py": "1.251026h",
    "report.py": "1.251026g",
    "manual.py": "1.251026g",
    "venvcheck.py": "1.251026m",
    "features.py": "1.251026g",
    "studio_css.py": "1.251026g",
    "Lora_Datatypes.py": "1.251026g",
    "Training_Recommendations.py": "1.251026g",
    "lora_studio.css": "1.251026g",
    "install.sh": "1.251026k",
    "start.sh": "1.251026i",
}
# what changed in each update, newest first (shown in Settings › Version)
CHANGES = [
    ("1.251026m", "Self-test fix: the training venv check is tested with made-up packages only, so it also passes on a computer whose venv has real ROCm packages (1.251026l failed its self-test there and was rolled back)."),
    ("1.251026l", "The PyTorch upgrade installs ROCm itself (rocm[libraries,devel,device-gfx…]) in the same pip run, at exactly the version the new PyTorch is built for."),
    ("1.251026k", "Settings › Updates › The training venv › Look for a newer PyTorch: asks AMD's server (ROCm) or PyPI (CUDA) for a newer stable PyTorch build and upgrades on a click (old versions kept, GPU test, back when it fails, locks and requirements renewed); install.sh keeps a PyTorch that is newer than its own."),
    ("1.251026j", "ROCm 10.1.0: install.sh installs torch 2.13.0+rocm10.1.0 (the same PyTorch, built for ROCm 10.1) with every ROCm part pinned to 10.1.0, and offers the upgrade to an install with an older ROCm build; a note for WSL with a Ryzen that has built-in graphics."),
    ("1.251026i", "The training venv is checked at every start and after every update (venvcheck.py, no download): every ROCm or CUDA part must be the version PyTorch is built for; Settings › Updates › The training venv shows it and Fix the training venv puts parts back, writes the locks in constraints.txt again and, when PyTorch was upgraded, installs the requirements again within the new locks (your own venv only gets the commands); a line in the Health check."),
    ("1.251026h", "ROCm fix: install.sh pins every ROCm part to the version PyTorch is built for and puts back a part that does not match (a newer rocm-sdk-devel made SDXL runs crash with 'Memory access fault by GPU'); the machine check shows mixed ROCm parts; a failed run with a GPU memory fault is explained in plain words."),
    ("1.251026g", "The painted logo (title icon, browser tab, Windows starter) and the login page with the login fields on the painting; an install report at the end of install.sh and the Windows installer (no tokens or home folder; Report for Claude includes it); Manual chapters Testing LoRA Studio and Start and network, and the login; GPL-3.0 licence, a README for new users (the details moved to REFERENCE.md); faster start of the page (it no longer loads twice, no fonts from Google); compressed pages when reachable on the network."),
    ("1.251026f", "Install for what you have: install.sh asks about sd-scripts, musubi-tuner and Ollama one by one (install it, use mine, or none; Ollama installed for you on Linux); a missing part has its buttons hidden and its steps and tabs greyed out with '(… missing)', Settings › Start and network › What is installed; Windows through WSL2 (Install LoRA Studio.cmd); Manual chapter Installing."),
    ("1.251026e", "Updates of LoRA Studio itself from a LoraStudio.zip only (the GitHub check is taken out again)."),
    ("1.251026d", "What went wrong in a failed run, in a few words; Report for Claude; the Manual tab; the logo; install.sh asks whether to install the trainers or use yours; a fix: try first was missing for runs whose launch script moved to _old_runs."),
    ("1.241026c", "Only the repeat folder <N>_<dataset name> is the dataset (other pictures only when step 2 says so, and kept ones join it; other repeat folders go to _extra/ when a launch script is saved); LoRA Studio updates itself from a LoraStudio.zip (Settings › Updates: backup, self-test, roll back)."),
    ("1.241026b", "Machine check with a tiny training (Settings › Version, selftest.py --machine); Training › Results: compare runs (loss curves, settings difference, test pictures side by side) and a best-version mark the Collection shows first; one look for buttons, bars, cards and tables; Datasets › 6 starts with the watermarks and folds the pictures that stay as they are; Settings › Version shows what each update changed."),
    ("1.241026", "First version with numbers: the program split by topic (sd_script.py, sd_prepare.py, "
                 "sd_settings.py), unused code removed, version numbers, Settings › Version, selftest.py."),
]
HERE = Path(__file__).resolve().parent

# every file that belongs to the program (the same list as APP_FILES in install.sh, without the data files)
PARTS = [
    ("lora_studio.py", "the program: starts everything, the page"),
    ("version.py", "this list"),
    ("sdgui.py", "the trainer side: runs, queue, history, the page layout of the old tabs"),
    ("sd_script.py", "launch scripts: plans, explanations, the step 9 page"),
    ("sd_prepare.py", "preparing a dataset: captions, picture check, watermarks, consistency"),
    ("sd_settings.py", "trainer settings: token, model files, repeat folders, subtypes, GPU"),
    ("datasets_ui.py", "Datasets: the nine steps"),
    ("dsflow.py", "Datasets: the logic behind the steps, automatic improvement"),
    ("training_ui.py", "Training: run, queue, results"),
    ("settings_ui.py", "Settings"),
    ("lora_library.py", "Collection (the LoRA library)"),
    ("comfytest.py", "test pictures with ComfyUI"),
    ("captioner.py", "captions with Ollama"),
    ("ollamactl.py", "Ollama control"),
    ("datacheck.py", "picture check"),
    ("consistency.py", "consistency check"),
    ("pictools.py", "picture tools (watermark removal, upscaling)"),
    ("pictools_worker.py", "picture tools worker (runs in the sd-scripts venv)"),
    ("picgallery.py", "picture gallery"),
    ("picker.py", "picture picker"),
    ("trim.py", "trimming pictures"),
    ("housekeeping.py", "housekeeping"),
    ("completed.py", "finished-run marks"),
    ("model_guide.py", "model guide"),
    ("ticker.py", "the one page timer"),
    ("statusbar.py", "status bar"),
    ("savepoint.py", "save points"),
    ("launchopts.py", "start options"),
    ("firstrun.py", "first start"),
    ("setup.py", "setup"),
    ("updater.py", "updates of the trainers"),
    ("backup.py", "backup"),
    ("selftest.py", "self-test of the main parts"),
    ("machinecheck.py", "the machine check (selftest.py --machine)"),
    ("selfupdate.py", "updates of LoRA Studio itself (Settings › Updates)"),
    ("failcodes.py", "what went wrong in a failed run, in a few words"),
    ("report.py", "Report for Claude"),
    ("manual.py", "the Manual tab"),
    ("venvcheck.py", "the training venv: PyTorch and its ROCm / CUDA parts belong together"),
    ("features.py", "what is installed (Ollama, sd-scripts, musubi-tuner): missing parts are greyed out"),
    ("studio_css.py", "the look (loads lora_studio.css)"),
    ("Lora_Datatypes.py", "LoRA types"),
    ("Training_Recommendations.py", "tuned settings"),
    ("lora_studio.css", "the look"),
    ("install.sh", "installer"),
    ("start.sh", "start script"),
]

_PY = re.compile(r'^__version__\s*=\s*["\']([^"\']+)["\']', re.M)
_TXT = re.compile(r"LoRA Studio version\s+([0-9][0-9A-Za-z.]*)")


def version_of(name: str, folder: Path | None = None) -> str | None:
    """The version written in a part, None when the file is missing, "" when it has no version."""
    p = (folder or HERE) / name
    try:
        text = p.read_text(encoding="utf-8", errors="replace")[:6000]
    except OSError:
        return None
    m = (_PY if name.endswith(".py") else _TXT).search(text)
    return m.group(1) if m else ""


def expected(name: str) -> str:
    return RELEASE.get(name, VERSION)


def parts(folder: Path | None = None) -> list[dict]:
    """[{name, what, version, want, new, state}] state: ok / old (not the version of this release) / none (no
    version) / missing; new = changed in the newest update."""
    out = []
    for name, what in PARTS:
        v = version_of(name, folder)
        want = expected(name)
        state = "missing" if v is None else "none" if v == "" else "ok" if v == want else "old"
        out.append({"name": name, "what": what, "version": v or "", "want": want, "new": want == VERSION,
                    "state": state})
    return out


def problems(folder: Path | None = None) -> list[dict]:
    return [p for p in parts(folder) if p["state"] != "ok"]


def banner() -> str:
    bad = problems()
    return f"LoRA Studio {VERSION}" + (f"  ({len(bad)} part(s) not of this release, see Settings › Version)" if bad else "")


if __name__ == "__main__":
    print(banner())
    for p in parts():
        print(f"  {p['name']:30} {p['version'] or '-':12} {p['state']}" + ("  (changed in this update)" if p["new"] else ""))
