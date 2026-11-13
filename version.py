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

__version__ = "1.131126b"
VERSION = "1.131126b"
# the version each part has in this release (written by the release script; a part not listed: VERSION)
RELEASE: dict = {
    "lora_studio.py": "1.131126b",
    "version.py": "1.131126b",
    "program/lspaths.py": "1.011126",
    "program/Training/sdgui.py": "1.131126b",
    "program/Training/sd_script.py": "1.131126b",
    "program/Data/sd_prepare.py": "1.131126b",
    "program/Training/sd_settings.py": "1.131126b",
    "program/Data/datasets_ui.py": "1.131126b",
    "program/Data/dsflow.py": "1.111126j",
    "program/Training/training_ui.py": "1.131126b",
    "program/Settings/settings_ui.py": "1.131126b",
    "program/Gallery/lora_library.py": "1.131126b",
    "program/Training/comfytest.py": "1.011126",
    "program/Training/speedtest.py": "1.111126b",
    "program/Training/speedjobs.py": "1.111126b",
    "program/Training/comfyupscale.py": "1.101126",
    "program/Data/detailscan.py": "1.101126b",
    "program/Data/captioner.py": "1.091126",
    "program/Data/wdtagger.py": "1.051126",
    "program/Settings/ghupdate.py": "1.011126",
    "program/Settings/hfdownload.py": "1.111126h",
    "program/Settings/ollamactl.py": "1.011126",
    "program/Data/datacheck.py": "1.031126",
    "program/Data/consistency.py": "1.011126",
    "program/Data/pictools.py": "1.011126",
    "program/Data/pictools_worker.py": "1.011126",
    "program/Gallery/picgallery.py": "1.011126",
    "program/Data/trim.py": "1.111126g",
    "program/Data/housekeeping.py": "1.011126",
    "program/Gallery/completed.py": "1.131126b",
    "program/Gallery/model_guide.py": "1.091126",
    "program/Gallery/trainingcard.py": "1.111126g",
    "program/Gallery/insights.py": "1.131126b",
    "program/Other/statusbar.py": "1.131126b",
    "program/Other/home.py": "1.131126b",
    "program/Other/gpusmi.py": "1.051126b",
    "program/Training/savepoint.py": "1.011126",
    "program/Settings/launchopts.py": "1.131126b",
    "program/Settings/firstrun.py": "1.011126",
    "program/Settings/setup.py": "1.011126",
    "program/Settings/updater.py": "1.011126",
    "program/Settings/backup.py": "1.111126j",
    "selftest.py": "1.111126h",
    "program/Settings/machinecheck.py": "1.111126d",
    "program/Settings/selfupdate.py": "1.011126",
    "program/Training/failcodes.py": "1.131126b",
    "program/Other/report.py": "1.011126",
    "program/Other/manual.py": "1.131126",
    "program/Settings/notice.py": "1.011126",
    "program/Settings/venvcheck.py": "1.011126",
    "program/Settings/features.py": "1.091126",
    "program/Other/studio_css.py": "1.011126",
    "program/Training/Lora_Datatypes.py": "1.111126h",
    "program/Training/Training_Recommendations.py": "1.091126",
    "program/Other/lora_studio.css": "1.131126b",
    "install.sh": "1.051126b",
    "start.sh": "1.051126",
}
# what changed in each update, newest first (shown in Settings › Version)
CHANGES = [
    ("1.131126b", "One page per dataset: an open dataset has four parts in a bar under its name: Prepare (steps 1-8, it reopens at the step you were at), Train (the launch script, plus this dataset's queue items and last runs), Results (its runs, moved here from Training › Results; 'Runs of' still shows other datasets or all) and LoRA (its LoRA files with versions, status, best and Completed, and a button to the LoRA page). The Training tab is now the Queue tab; Insights moved to the Collection. A Collection tile opens the dataset page's LoRA part (a type label on the tile still opens that LoRA page). The status strip and Home's Last results open the run in its dataset's Results. The old Training › Results start-up load and its wiring are removed. Manual updated."),
    ("1.131126", "Redesign 4: Home is the new start page: what is training now, what needs a look (setup problems, datasets that could be better, warnings in your LoRA files), the last five results and the datasets you worked on last, each a click away. The Manual left the tab row: the ? button at the top opens it at the chapter of the page you are on, and ← Back returns there. If Collection was your start page, Home takes its place (Settings › Start and network changes it)."),
    ("1.111126j", "Redesign 3: Datasets is one list with + New dataset on top (New dataset and Edit existing are no longer two pages). Improvements has its own choices folded on the page (What Fix does; they left Settings) and Check the next fixed dataset, which opens each fixed dataset at step 5 to look it over. Training › Run's Mark as run is now Mark Completed, the one done mark. Settings is grouped like the work: Start, Datasets, Training, Results, Collection, Care."),
    ("1.111126i", "Redesign 2 (from dataset to training in one go): step 9's last button is now Save and add to the queue; with Start the queue when nothing is training ticked (remembered) the run starts at once. When a run ends, clicking the status strip opens that run in Training › Results. Results has Your rating next to the best-version mark, and Insights moved to Training as its third page (Training › Insights). Every click inside the Results lists is one event instead of four or five."),
    ("1.111126h", "Redesign 1 (Datasets steps in a natural order): Setup is now step 2 and the usable scan step 3, so the scan knows which sizes your LoRA types need. Setup shows whether the model files of the ticked types are on this computer, whether their trainer is installed and whether Ollama has a model that reads pictures; missing model files download in the background while you go on (step 9's download button does the same). Steps renamed: 6 Watermarks and size, 7 Captions, 9 Train. The manual's character description example no longer suggests body traits."),
    ("1.111126g", "Clean-up 4 (Collection and start): the old stand-alone lora_library.py program is gone (LoRA Studio is the only way in), and leftover code that nothing used any more is removed. Nothing you use changes. This ends the clean-up run: four releases, about 900 lines less, and a quicker page."),
    ("1.111126f", "Clean-up 3 (Datasets): moving between steps and opening a dataset is much quicker. A new step was four events and opening a dataset about eight more; each is now one (on the 80-dataset test copy most step changes went from 4.5-9 s of a busy browser to 1.5-2.7 s). A hidden 'Prepare' page that recalculated after every change is gone, with its unused picture check code. The example text in the character description no longer suggests body traits."),
    ("1.111126e", "Clean-up 2 (Training): the old stand-alone sdgui.py program is gone (LoRA Studio is the only way in; it had no Run page any more), with its own header, GPU display, timers and the hidden 'Run all steps' preparation, which also ran three extra steps every time you picked a LoRA type in step 9. Leftovers of the old queue table removed. Training › Results no longer searches all dataset folders for every old log without a script line on each refresh. Nothing you use changes."),
    ("1.111126d", "Clean-up 1 (status strip, timer, start-up, Settings): the page is quicker to open. Fields that the start-up already filled no longer each send their own event afterwards (the model file fields made four, the Collection's base-model filter, the caption example and the speed test list one each), the shared timer no longer redraws parts that have nothing new, and the training venv status loads with the other start-up parts. On a test copy with 80 datasets and 200 LoRAs the browser was busy about 9 s instead of 22 s while opening the page. Leftovers removed. LS_LOAD_DEBUG=1 prints how long each start-up part takes."),
    ("1.111126c", "Runs stopped before their final LoRA count as not run: listed as stopped, ticked by 'Tick the ones not run', and can be queued again."),
    ("1.111126b", "New: Settings › Speed test. Tick the LoRA types to test (quick, full, or only how fast they train now); each trains a few steps on an invented test dataset with different speed settings. Results stay per type, and 'Use the faster settings' makes new launch scripts of that type use the fastest one. Fixed: live parts with one output (for example download progress) could show 'Error' since 1.101126c."),
    ("1.111126", "New: a training speed test for the terminal (program/Training/speedtest.py). It tries faster settings on one of your own launch scripts (block swapping, pinned memory, checkpointing, attention, fp8, torch.compile, ROCm settings), measures seconds per step and video memory, and writes a report with what to change."),
    ("1.101126d", "Z-Image, Qwen-Image and FLUX.2 (musubi-tuner): a dataset with a second caption set trained on no pictures at all (musubi did not recognise IMG.sentences.txt). The pictures are now linked into musubi's cache with the right captions, and a missing caption set stops the run with a clear message."),
    ("1.101126c", "Speed: much less work for the browser. The start-up loaders run as one event, the status strip and the queue refresh without Gradio events, the shared timer sleeps while nothing runs and redraws only what changed, and launch scripts are no longer re-read on every list redraw. Typing and clicks no longer stutter, and switching tabs is near instant."),
    ("1.101126b", "Detailed scan: a face crop and a detail crop that show nearly the same (a wreath on the head) become one new picture instead of two near-duplicates."),
    ("1.101126", "Detailed scan for character datasets (step 4): finds gaps in angles, poses, expressions, outfits, places and light, and suggests per picture what to add, upscale, sharpen or crop into a new picture (step 5, only when ticked). Upscaling and sharpening use your own ComfyUI workflow, for example SeedVR2 (Settings › Test pictures), with the size and noise set per picture."),
    ("1.091126b", "Caption sets: Anima, FLUX and Chroma can use tags, sentences or both (step 3); with both, step 9 picks the set a launch script trains on. Fixed: a second LoRA type in a dataset trained on the first caption set instead of its own."),
    ("1.091126", "Character settings revised: SDXL, Illustrious and Pony train the text encoders lightly (5e-5); FLUX and Chroma use rank 32 / alpha 32 and up to 3000 steps; Z-Image Turbo up to 3000 steps; new LoRA type Z-Image Base (no training adapter); a class word in step 3 makes sentence captions begin 'trigger, a class word, ...'."),
    ("1.081126", "Datasets: the error 'desc_load didn't return enough output values' in the terminal when moving between pages is gone."),
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
    ("lora_studio.py", "the program: starts everything"),
    ("version.py", "this list"),
    ("program/lspaths.py", "where everything lives: program/, userdata/; moves an old install into this layout"),
    ("program/Training/sdgui.py", "the trainer side: runs, queue, history"),
    ("program/Training/sd_script.py", "launch scripts: plans, explanations, the step 9 page"),
    ("program/Data/sd_prepare.py", "preparing a dataset: captions, picture check, watermarks, consistency"),
    ("program/Training/sd_settings.py", "trainer settings: token, model files, repeat folders, subtypes, GPU"),
    ("program/Data/datasets_ui.py", "Datasets: the nine steps"),
    ("program/Data/dsflow.py", "Datasets: the logic behind the steps, automatic improvement"),
    ("program/Training/training_ui.py", "Training: run, queue, results"),
    ("program/Settings/settings_ui.py", "Settings"),
    ("program/Gallery/lora_library.py", "Collection (the LoRA library)"),
    ("program/Training/comfytest.py", "test pictures with ComfyUI"),
    ("program/Training/speedtest.py", "training speed test for a launch script (terminal)"),
    ("program/Training/speedjobs.py", "speed tests per LoRA type (Settings › Speed test)"),
    ("program/Training/comfyupscale.py", "upscale and sharpen pictures with your ComfyUI workflow"),
    ("program/Data/detailscan.py", "the detailed scan: gaps and per-picture suggestions"),
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
    ("program/UI/shell.py", "the page: the top bar, tabs, status strip and the ? manual"),
    ("program/UI/widgets.py", "small shared page parts: pictures, cards, crop and paint panels, file links"),
    ("program/UI/theme.css", "the look of the NiceGUI pages, light and dark"),
    ("program/UI/login.py", "the login page"),
    ("program/UI/setup_page.py", "the setup page and the notice shown once"),
    ("program/UI/pg_home.py", "Home"),
    ("program/UI/pg_datasets.py", "Datasets: the list, Improvements and the dataset page"),
    ("program/UI/ds_steps.py", "Datasets: steps 2-8 and the LoRA part"),
    ("program/UI/ds_train.py", "Datasets: step 9 Train"),
    ("program/UI/ds_results.py", "Datasets: Results (runs, compare, test pictures)"),
    ("program/UI/ds_batch.py", "Datasets › Improvements: Fix the ticked datasets"),
    ("program/UI/pg_queue.py", "Queue: the launch scripts, the queue and the live run"),
    ("program/UI/pg_collection.py", "Collection: Browse, Insights, Maintenance"),
    ("program/UI/lora_page.py", "Collection: the LoRA page"),
    ("program/UI/pg_settings.py", "Settings: the list on the left"),
    ("program/UI/set_panels.py", "Settings: the panels"),
    ("program/Gallery/lorapage.py", "the LoRA page's logic (details, edits, pictures)"),
    ("program/Gallery/maint.py", "Collection › Maintenance logic (sync, leftovers, orphans)"),
    ("program/Other/notify.py", "short messages at the corner of the page"),
    ("program/Other/statusbar.py", "status bar"),
    ("program/Other/home.py", "Home page"),
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
    ("program/Other/manual.py", "the manual (the ? button)"),
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
