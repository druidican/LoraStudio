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

__version__ = "1.261026z"
VERSION = "1.261026z"
# the version each part has in this release (written by the release script; a part not listed: VERSION)
RELEASE: dict = {
    "lora_studio.py": "1.251026o",
    "version.py": "1.261026z",
    "program/lspaths.py": "1.251026o",
    "program/Training/sdgui.py": "1.261026y",
    "program/Training/sd_script.py": "1.261026v",
    "program/Data/sd_prepare.py": "1.261026v",
    "program/Training/sd_settings.py": "1.261026v",
    "program/Data/datasets_ui.py": "1.261026z",
    "program/Data/dsflow.py": "1.261026w",
    "program/Training/training_ui.py": "1.261026v",
    "program/Settings/settings_ui.py": "1.251026o",
    "program/Gallery/lora_library.py": "1.261026z",
    "program/Training/comfytest.py": "1.251026o",
    "program/Data/captioner.py": "1.261026v",
    "program/Data/wdtagger.py": "1.261026v",
    "program/Settings/hfdownload.py": "1.261026v",
    "program/Settings/ollamactl.py": "1.251026o",
    "program/Data/datacheck.py": "1.261026w",
    "program/Data/consistency.py": "1.251026o",
    "program/Data/pictools.py": "1.261026v",
    "program/Data/pictools_worker.py": "1.251026o",
    "program/Gallery/picgallery.py": "1.251026o",
    "program/Data/trim.py": "1.251026o",
    "program/Data/housekeeping.py": "1.261026v",
    "program/Gallery/completed.py": "1.261026v",
    "program/Gallery/model_guide.py": "1.261026v",
    "program/Gallery/trainingcard.py": "1.261026z",
    "program/Gallery/insights.py": "1.261026z",
    "program/Other/ticker.py": "1.251026o",
    "program/Other/statusbar.py": "1.251026o",
    "program/Training/savepoint.py": "1.251026o",
    "program/Settings/launchopts.py": "1.261026v",
    "program/Settings/firstrun.py": "1.261026v",
    "program/Settings/setup.py": "1.251026o",
    "program/Settings/updater.py": "1.251026o",
    "program/Settings/backup.py": "1.251026o",
    "selftest.py": "1.261026z",
    "program/Settings/machinecheck.py": "1.251026o",
    "program/Settings/selfupdate.py": "1.261026x",
    "program/Training/failcodes.py": "1.251026o",
    "program/Other/report.py": "1.251026o",
    "program/Other/manual.py": "1.261026v",
    "program/Settings/notice.py": "1.261026v",
    "program/Settings/venvcheck.py": "1.251026o",
    "program/Settings/features.py": "1.251026o",
    "program/Other/studio_css.py": "1.251026o",
    "program/Training/Lora_Datatypes.py": "1.251026o",
    "program/Training/Training_Recommendations.py": "1.251026o",
    "program/Other/lora_studio.css": "1.261026v",
    "install.sh": "1.261026x",
    "start.sh": "1.251026o",
}
# what changed in each update, newest first (shown in Settings › Version)
CHANGES = [
    ("1.261026z", "Collection › Insights: the datasets behind your good and weak LoRAs side by side per LoRA type (pictures, repeats, steps, close-ups, full body, outfits, places, detail shots, caption length), and what your graphics card managed (batch sizes that finished or ran out of video memory, seconds per step). A version is good when you rate it 4-5 (new: Your rating on the LoRA page › Edit), or is the ★ best version, or has pictures made with it; weak when rated 1-2. Step 5 shows what your good LoRAs of the type had; step 9 warns about a batch that ran out of memory before and estimates the time from your runs at that batch size."),
    ("1.261026y", "Collection › LoRA page: a Training tab with the training card: dataset, LoRA type, version, trigger words, pictures and repeat folders, epochs and how the steps add up, batch, network, learning rate, optimizer, scheduler, clip skip and more, training time, final loss and the save point to try first, each in its own field, read from the LoRA file, its launch script (also in _old_runs), the repeat folders and the logs. Saved in the dataset's copy of the LoRA Manager file with a scanned mark; Maintenance › Scan all LoRAs reads every version once (again only when its file changes); the old training text in Notes is taken out. Edit has Version, Clip skip and Description. Sync now: the version name goes to LoRA Manager, trigger words come from it when LoRA Studio has none, usage tips are merged and clip skip defaults to 1, LoRA Manager's chosen preview and example pictures show on the LoRA page. New runs write the GPU name into their log."),
    ("1.261026x", "Documentation: a new INSTALL.md (step by step for Linux and Windows: what you need, the installer's questions, the first start, updates, moving to a new computer, removing it, and a troubleshooting table), linked from the README and part of the zip. The Manual describes the Hugging Face downloads, the two-way metadata sync, the usefulness check and the pins, with new pictures of step 4, step 5 and Settings › Model files; step 4's own text was updated too."),
    ("1.261026w", "Datasets › 4 Sort, for characters: pictures where clothes, pose or angle hide nearly everything about the character, or where the figure is small and the face tiny, are left out; plain hands, feet, legs or arms are left out instead of kept as detail shots (a tail, horns, a tattoo stay); the face size counts, and sharpness weighs less, so a slightly soft picture that shows the character well beats a crisp one that shows little."),
    ("1.261026v", "Datasets › 4 Sort / 5 Select: the Ollama sort also rates how much each picture teaches about the subject (1-5) and notes a detail it shows well; pictures that teach almost nothing are left out, the rating counts in the quality score. Step 5 has pins under every picture: 📌 Always keeps it and ⛔ Never leaves it out whatever the rules say, kept through new sorts and used by the automatic improvement. Clean-up: dead code removed (the old picture picker, unused functions and imports, 80 unused re-exports in sdgui), and Lora_Datatypes.py is no longer re-read on every call (read again only when it changes): opening a dataset step is up to 7× faster."),
    ("1.261026u", "Collection › Maintenance › Sync now syncs the .metadata.json files both ways: data that only one side has (notes, tags, trigger words, Civitai data, usage tips …) is copied to the other, also into the LoRA folder's file, so ComfyUI's LoRA Manager sees your edits; an empty field takes the other side's value; where both differ, the newer file wins. File paths stay as they are on each side, a .bak is made before a file changes."),
    ("1.261026t", "Model files can be downloaded from Hugging Face on a click, with your token: a ⬇ From Hugging Face button next to every missing file in Settings › Model files (and Download all missing files), the same card on Datasets › 9 Launch script, and a field to paste any Hugging Face file link for a slot. The usual files per LoRA type come from the sd-scripts and musubi-tuner documentation; they go into your models folder (checkpoints, diffusion_models, text_encoders, vae, loras), files in several parts are fetched whole, a stopped download goes on later, gated repos say what to do, and the slot is saved."),
    ("1.261026s", "Datasets › 4 Sort stays done when pictures are removed, moved (step 5), saved again (step 6) or when the repeat folder is renamed: a picture is matched by its file name, not only by its full path; only new pictures ask for a new sort. Step 5 also keeps the sort results of moved pictures."),
    ("1.261026r", "Datasets › 7 Tags: the tag-style caption sets (Illustrious, Pony, SDXL, Anima) are written by the WD tagger (SmilingWolf/wd-eva02-large-tagger-v3, through sd-scripts' own tagger) instead of Ollama: general tags only (no character names, the trigger word stands for them), the Never describe list and the consistency check as usual; works without Ollama; onnx/onnxruntime are installed into the training venv the first time. Tag cleaning keeps tags like 1girl."),
    ("1.261026q", "Datasets › 3 Setup: Object, Action / pose and Concept need the words for the thing; when they were missing, Continue showed the message but the page went back to the saved purpose and hid the words field, so step 3 just repeated. Your choice now stays, the words field stays open and the message says where to write."),
    ("1.261026p", "Collection: a warning you have looked at (for example 'Run stopped at epoch 8 of 10' after stopping a run on purpose) can be accepted on the LoRA page (Accept the warnings): the orange mark goes away; pressing again shows it again."),
    ("1.251026o", "New folder layout: program/ holds the program by topic (Gallery, Data, Training, Settings, Other, plus the Manual and the logo), userdata/ holds everything that is yours (settings, token, marks, caches, logs, reports, update backups); the launcher, version.py, selftest.py, the installers and the documents stay in the main folder. An older install moves into it by itself at the first start or with ./install.sh (the old program files are kept aside); backups of the old layout import as before. On an NVIDIA card Settings › Trainers and GPU shows CUDA values instead of the ROCm ones, and launch scripts leave the ROCm-only values out. README: made on Linux with ROCm, testers wanted."),
    ("1.251026n", "The notice about sd-scripts and musubi-tuner (made by kohya-ss, Apache License 2.0; LoRA Studio does not own or include them): confirmed once in install.sh and on the very first page in the browser; also in the NOTICE file and Settings › Version › Licences and credits."),
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
    ("program/lspaths.py", "where everything lives: program/, userdata/; moves an old install into this layout"),
    ("program/Training/sdgui.py", "the trainer side: runs, queue, history, the page layout of the old tabs"),
    ("program/Training/sd_script.py", "launch scripts: plans, explanations, the step 9 page"),
    ("program/Data/sd_prepare.py", "preparing a dataset: captions, picture check, watermarks, consistency"),
    ("program/Training/sd_settings.py", "trainer settings: token, model files, repeat folders, subtypes, GPU"),
    ("program/Data/datasets_ui.py", "Datasets: the nine steps"),
    ("program/Data/dsflow.py", "Datasets: the logic behind the steps, automatic improvement"),
    ("program/Training/training_ui.py", "Training: run, queue, results"),
    ("program/Settings/settings_ui.py", "Settings"),
    ("program/Gallery/lora_library.py", "Collection (the LoRA library)"),
    ("program/Training/comfytest.py", "test pictures with ComfyUI"),
    ("program/Data/captioner.py", "captions with Ollama"),
    ("program/Data/wdtagger.py", "tags with the WD tagger (wd-eva02-large-tagger-v3) for the tag-style caption sets"),
    ("program/Settings/hfdownload.py", "downloads missing model files from Hugging Face on a click, with your token"),
    ("program/Settings/ollamactl.py", "Ollama control"),
    ("program/Data/datacheck.py", "picture check"),
    ("program/Data/consistency.py", "consistency check"),
    ("program/Data/pictools.py", "picture tools (watermark removal, upscaling)"),
    ("program/Data/pictools_worker.py", "picture tools worker (runs in the sd-scripts venv)"),
    ("program/Gallery/picgallery.py", "picture gallery"),
    ("program/Data/trim.py", "trimming pictures"),
    ("program/Data/housekeeping.py", "housekeeping"),
    ("program/Gallery/completed.py", "finished-run marks"),
    ("program/Gallery/model_guide.py", "model guide"),
    ("program/Gallery/trainingcard.py", "the training card of a LoRA: its training data in separate fields"),
    ("program/Gallery/insights.py", "Insights: the datasets behind good and weak LoRAs, what your graphics card managed"),
    ("program/Other/ticker.py", "the one page timer"),
    ("program/Other/statusbar.py", "status bar"),
    ("program/Training/savepoint.py", "save points"),
    ("program/Settings/launchopts.py", "start options"),
    ("program/Settings/firstrun.py", "first start"),
    ("program/Settings/setup.py", "setup"),
    ("program/Settings/updater.py", "updates of the trainers"),
    ("program/Settings/backup.py", "backup"),
    ("selftest.py", "self-test of the main parts"),
    ("program/Settings/machinecheck.py", "the machine check (selftest.py --machine)"),
    ("program/Settings/selfupdate.py", "updates of LoRA Studio itself (Settings › Updates)"),
    ("program/Training/failcodes.py", "what went wrong in a failed run, in a few words"),
    ("program/Other/report.py", "Report for Claude"),
    ("program/Other/manual.py", "the Manual tab"),
    ("program/Settings/notice.py", "the notice about sd-scripts / musubi-tuner (first page, once)"),
    ("program/Settings/venvcheck.py", "the training venv: PyTorch and its ROCm / CUDA parts belong together"),
    ("program/Settings/features.py", "what is installed (Ollama, sd-scripts, musubi-tuner): missing parts are greyed out"),
    ("program/Other/studio_css.py", "the look (loads lora_studio.css)"),
    ("program/Training/Lora_Datatypes.py", "LoRA types"),
    ("program/Training/Training_Recommendations.py", "tuned settings"),
    ("program/Other/lora_studio.css", "the look"),
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
