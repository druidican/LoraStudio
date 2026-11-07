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

__version__ = "1.071126c"
VERSION = "1.071126c"
# the version each part has in this release (written by the release script; a part not listed: VERSION)
RELEASE: dict = {
    "lora_studio.py": "1.011126",
    "version.py": "1.071126c",
    "program/lspaths.py": "1.011126",
    "program/Training/sdgui.py": "1.071126b",
    "program/Training/sd_script.py": "1.071126b",
    "program/Data/sd_prepare.py": "1.031126",
    "program/Training/sd_settings.py": "1.051126b",
    "program/Data/datasets_ui.py": "1.031126",
    "program/Data/dsflow.py": "1.051126",
    "program/Training/training_ui.py": "1.011126",
    "program/Settings/settings_ui.py": "1.051126b",
    "program/Gallery/lora_library.py": "1.031126",
    "program/Training/comfytest.py": "1.011126",
    "program/Data/captioner.py": "1.031126",
    "program/Data/wdtagger.py": "1.051126",
    "program/Settings/ghupdate.py": "1.011126",
    "program/Settings/hfdownload.py": "1.011126",
    "program/Settings/ollamactl.py": "1.011126",
    "program/Data/datacheck.py": "1.031126",
    "program/Data/consistency.py": "1.011126",
    "program/Data/pictools.py": "1.011126",
    "program/Data/pictools_worker.py": "1.011126",
    "program/Gallery/picgallery.py": "1.011126",
    "program/Data/trim.py": "1.011126",
    "program/Data/housekeeping.py": "1.011126",
    "program/Gallery/completed.py": "1.011126",
    "program/Gallery/model_guide.py": "1.011126",
    "program/Gallery/trainingcard.py": "1.011126",
    "program/Gallery/insights.py": "1.011126",
    "program/Other/ticker.py": "1.011126",
    "program/Other/statusbar.py": "1.051126b",
    "program/Other/gpusmi.py": "1.051126b",
    "program/Training/savepoint.py": "1.011126",
    "program/Settings/launchopts.py": "1.011126",
    "program/Settings/firstrun.py": "1.011126",
    "program/Settings/setup.py": "1.011126",
    "program/Settings/updater.py": "1.011126",
    "program/Settings/backup.py": "1.011126",
    "selftest.py": "1.071126b",
    "program/Settings/machinecheck.py": "1.051126b",
    "program/Settings/selfupdate.py": "1.011126",
    "program/Training/failcodes.py": "1.011126",
    "program/Other/report.py": "1.011126",
    "program/Other/manual.py": "1.011126",
    "program/Settings/notice.py": "1.011126",
    "program/Settings/venvcheck.py": "1.011126",
    "program/Settings/features.py": "1.011126",
    "program/Other/studio_css.py": "1.011126",
    "program/Training/Lora_Datatypes.py": "1.071126c",
    "program/Training/Training_Recommendations.py": "1.071126c",
    "program/Other/lora_studio.css": "1.011126",
    "install.sh": "1.051126b",
    "start.sh": "1.051126",
}
# what changed in each update, newest first (shown in Settings › Version)
CHANGES = [
    ("1.071126c", "SDXL, Illustrious and Pony character LoRAs now train with alpha = dim (32/32) instead of 32/16, which learned at half strength."),
    ("1.071126b", "Launch scripts: sampling is never written or run without a prompts file (sd-scripts stopped at the first sample), and SDXL, Illustrious and Pony scripts no longer get --clip_skip. A script that has them is fixed when it runs."),
    ("1.071126", "Quality words: Pony LoRA types train with score_9, score_8_up, score_7_up, source_anime in front of every caption and Illustrious with masterpiece, best quality behind it (--caption_prefix / --caption_suffix in the launch script; the caption files stay as they are; for Pony --keep_tokens also keeps the quality words and the trigger word in front); newly generated sample prompts get them too."),
    ("1.051126b", "AMD GPU values from amd-smi: ROCm 10 and newer have no rocm-smi, so the top bar, the machine check, the GPU name for the run log and Detect GPU read amd-smi (metric and static, in JSON) there, and rocm-smi on older ROCm. install.sh looks for both, tells which one it uses (and how to install amd-smi when it is missing) and saves the choice."),
    ("1.051126", "Install: the installer asks for the install folder (Enter = ~/LoraStudio; --dir still works; run inside an install it updates that one), and a moved or renamed LoRA Studio folder starts again: start.sh puts the paths of its Python environment and settings right by itself (a training environment is never thrown away). Datasets › 7: when the WD tagger writes no captions, its last lines are shown instead of a done message."),
    ("1.031126", "Worn items: a new purpose for clothes, equipment and jewelry worn by people (step 3 asks which). Step 7 writes every tag; Kinds and where they are worn keeps the kind and place words the same in every caption (other words become the name, the name is added where a tag mentions it, trigger, kinds and places first) and counts the pictures per kind (step 5 shows it too); Item tags → trigger words moves the item's own look out of the captions into the Item description next to the trigger word, keeping the kind. Sentence captions describe the person in full and only name the kind and where it is worn. Manual and reference updated."),
    ("1.011126", "The first public release of LoRA Studio. Datasets: nine steps from a folder of pictures to a launch script (usable scan, sort with pins and a usefulness check, selection, resize, WD tags or Ollama sentences with the character tags moved to the trigger words, review, launch script). Training: a queue that frees video memory between models, the live run, and Results with save points to try, test pictures through ComfyUI and comparisons. Collection: your LoRAs with training cards, LoRA Manager sync both ways, example pictures, ratings and Insights. Settings: health check, folders, Hugging Face downloads, backups, updates from a zip or from GitHub releases. Install with its own Python 3.12 on Linux, Windows (WSL2) and macOS (without training). GPL-3.0-only."),
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
    ("program/Settings/ghupdate.py", "updates of LoRA Studio from GitHub (the newest upload), on a button"),
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
    ("program/Other/gpusmi.py", "AMD GPU values: amd-smi (ROCm 10 and newer) or rocm-smi"),
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
