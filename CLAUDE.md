# LoRA Studio: notes for the next session

Read this first. It says how the project is built, the rules that are not visible in the code, why some things
are the way they are, and what is still open. The owner (L. G. H. Druidican, GitHub `druidican`) does not write
code: the work is done by Claude, the owner tests on real hardware and reports back.

## What it is

A Gradio GUI (Python 3.12) for training LoRAs with kohya-ss **sd-scripts** (SD 1.5, SDXL, Illustrious, Pony,
Anima, FLUX, SD3, Lumina, Chroma …) and **musubi-tuner** (Z-Image, Qwen-Image, FLUX.2), mainly on AMD ROCm under
Linux, also NVIDIA. It also makes datasets (sort, select, crop, captions), keeps a catalogue of the owner's LoRAs
(the Collection, with LoRA Manager data), runs a training queue and makes test pictures through ComfyUI.

- `lora_studio.py`: entry point (the only one: the old stand-alone sdgui.py / lora_library.py modes were removed in 1.111126e/g). `start.sh` / `install.sh` (Linux, WSL, macOS without trainers),
  `install-windows.ps1` + `Install LoRA Studio.cmd` (installs into WSL2).
- `program/`: `Data/` (datasets, the nine steps: `dsflow.py`, `datasets_ui.py`, `captioner.py`, `wdtagger.py`,
  `consistency.py`), `Training/` (`sdgui.py` runner/queue, `sd_script.py` launch scripts, `Lora_Datatypes.py`
  presets per LoRA type, `Training_Recommendations.py` numbers per type and purpose), `Gallery/` (Collection,
  `lora_library.py`), `Settings/` (`settings_ui.py`, `ghupdate.py` updater), `Other/` (`gpusmi.py`, `lspaths.py` …),
  `manual/` (the built-in manual, Markdown + `img/`).
- `version.py`: VERSION, PARTS (every program file), RELEASE (version of each part), CHANGES (release notes).
- `selftest.py`: checks that must pass before every release (`python selftest.py`).
- `tools/`: release and test helpers (not shipped in the zip).

## Rules

- **Licence:** GPL-3.0-only, © 2026 L. G. H. Druidican. Every source file starts with
  `# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE` and has
  `__version__ = "…"  # LoRA Studio version, see version.py`.
- **A new program file** needs an entry in `version.py` PARTS and RELEASE, the header above, and the install
  lists if it is outside `program/`.
- **Versions:** `1.ddmmyy`, plus `b`, `c` … for further releases the same day. Never reuse a released number,
  except when the owner explicitly says a change belongs to an existing release.
- **Releases:** `tools/release.sh <version> "<text>"` (stamps, selftest, builds `dist/LoraStudio.zip`), then
  commit and push to `main`, and send the owner the zip. Tag pushes are refused by the session proxy: the owner
  makes the GitHub release by hand. Commit messages end with the attribution lines the session provides.
- **Never** put the owner's token files (`huggingtoken.txt`, `github_token.txt`) or anything from `userdata/` in
  the repo or a zip. When sharing, share only `LoraStudio.zip`, never a folder of the owner's.
- **Manual:** every user-visible feature is described in `program/manual/`. Screenshots are taken only from
  an invented demo world (a demo robot called *pip*, a throw-away HOME), never from the owner's real datasets,
  LoRAs or ComfyUI prompts. Back up the demo world before a session and restore it afterwards.
- **People in datasets:** never analyse, compare or rate the likeness of a real person in a dataset, never add
  face or likeness matching, and never suggest ethnicity, skin colour, age or body parts for character captions.
  Problems with such datasets get generic tool fixes only.
- **ComfyUI workflows** of the owner hold their own prompts: use only their settings (sampler, size, model),
  never copy or show the prompts.

## Working with the owner

- Ask clarifying questions before giving detailed answers (their stated preference).
- Commands and instructions for Ubuntu (apt), not Fedora. Their main machine: Ubuntu 26.10, RX 9070 XT (gfx1201),
  ROCm 10.x, so `amd-smi`, not `rocm-smi`.
- Short, plain answers; no code they have to edit by hand when a release can do it.

## Why things are the way they are

- **Python 3.12 via uv** in `.tools/python` (install.sh), because system Pythons broke installs on Windows/macOS.
  uv makes absolute symlinks, so `start.sh` repairs them, the venv, shebangs and settings paths when the folder
  was moved (`venv/.lorastudio-root` remembers the old place). The installer asks where to install.
- **requirements.txt has no versions** (owner's wish: latest gradio, pandas, pillow, tensorboard).
- **GPU status:** `gpusmi.py` uses amd-smi on ROCm ≥ 10 (rocm-smi is gone there), rocm-smi before; install.sh
  writes the choice to `features.json` `gpu_tool`. amd-smi's JSON differs between versions: keys matched loosely.
- **Updater** (`ghupdate.py`): channel *Releases only* (default) or *Betas too*; the token is never sent along a
  redirect to another host (codeload); *Test token* explains each failure.
- **WD tagger writes every tag** for characters and worn items; step 7's *tags → trigger words* card moves the
  ticked ones into the description (undo: *Put them back*). Earlier versions left trait tags out, which hid
  information and errors. A failing WD tagger must always show its error, never "done".
- **Caption sets:** one per caption style (`IMG.txt`, `IMG.<style>.txt`); every launch script gets
  `--caption_extension` of its own set (`dsflow.caption_ext`). Anima, FLUX and Chroma read both styles: step 3
  chooses tags / sentences / both, step 9 picks the set per script (`train_style` in the dataset's meta).
  musubi pairs captions by cutting ONE extension (IMG.sentences.txt never matches IMG.png), so musubi scripts link
  the pictures into `cache/<name>/images/<repeat folder>` with the chosen captions as IMG.txt (`CAPTION_EXT`).
- **Class word** (step 3, characters): sentence captions begin "trigger, a <class word>, …" (`captioner.with_class`).
- **Detailed scan** (`Data/detailscan.py`, characters, on per dataset in step 4): an extra Ollama question per
  picture, coverage over the kept pictures, suggestions in step 5 (add / upscale / sharpen / crop into a new picture),
  done only when ticked. Upscales go through the owner's ComfyUI workflow (`Training/comfyupscale.py`: UI or API
  format, SeedVR2 size and noise set per picture), else the picture tools. Judges visibility, never identity.
- **Speed test**: `Training/speedtest.py` (terminal, one launch script) and `Training/speedjobs.py` (Settings ›
  Speed test, per LoRA type, invented dataset in `userdata/speedtest/`, results.json, use.json applied in
  `sd_script.make_plan` via `_speed`). The launch script is run up to `accelerate` with a bash function recording it.
- **Worn item** purpose (clothes / equipment / jewelry): kinds and places, *Make the words the same*, balance check.
- **Quality words:** Pony trains with `score_9, score_8_up, score_7_up, source_anime` as `--caption_prefix`
  (keep_tokens raised to cover them), Illustrious with `masterpiece, best quality` as `--caption_suffix`; caption
  files stay untouched; new sample prompts get them too.
- **SDXL / Illustrious / Pony characters: dim 32, alpha 32** (1.071126c). The owner's test: 32/16 learned at half
  strength and was clearly worse. Anima characters use 16/16. Since 1.091126 SDXL-family characters also train the
  text encoders at 5e-5 (batch ≤ 4); FLUX / Chroma characters rank 32/32, 1600-3000 steps; Z-Image Base is its own
  type (no adapter).
- **Never sample without prompts:** sd-scripts crashes at the first sample when `--sample_every_n_epochs` has no
  `--sample_prompts`; `sdgui.fix_flags` removes it (and `--clip_skip` from SDXL scripts, which ignore it) when
  saving, when a form field is applied and when a script runs.
- **Training stops Ollama** to free video memory and does not start it again.
- Model choice matters as much as settings: the owner trains and generates Illustrious LoRAs on
  Illustrious-XL v0.1 by choice. Watermarks in training pictures are learned; step 6 removes them, but check.

## Speed (1.101126c)

Every Gradio event costs the browser ~0.3 s on this big page, whatever it outputs. So: the start-up loaders are
joined into one event (`Other/loadjoin.py`, `LS_SEPARATE_LOADS=1` to compare); the status strip and the queue are
fetched by the page itself from `/ls_live` (`Other/livepoll.py`, behind the login); the shared timer
(`Other/ticker.py`) redraws only parts that changed and sleeps while nothing runs (the page wakes it).
`script_type` is cached per file. A Datasets step change is one event (`nav_all`), a new dataset/type one more (`sync_all` with sdgui's `ds_all`); don't add `.change` chains that follow programmatic sets, chain or join instead. `LS_LOAD_DEBUG=1` prints the start-up steps' times. Measure with a test copy at the owner's size (80 datasets, 200 LoRAs) and
Playwright long tasks; `LS_TICK_DEBUG=1` prints what each tick changes.

## Testing

- `python selftest.py` (needs gradio etc.; use the program's venv).
- `tools/run_test_app.sh <work folder> [port]`: a copy of the app with its own HOME and a pretend Ollama
  (`tools/fake_ollama.py`); `tools/fake_comfy.py` and `tools/fake_github.py` pretend ComfyUI and a GitHub release.
- Browser checks with Playwright (chromium). Don't use `pkill -f` / `pgrep -f` with a pattern that also matches
  your own shell command; it kills the shell. Use `fuser -k <port>/tcp` or `ps … | awk` with a precise pattern.

## Open

- Worn items, later (owner deferred): sort and crop around the item; weight per kind.
- amd-smi: values that stay empty on real hardware need the owner's `amd-smi metric --json` output.
- WD tagger on Ubuntu 26.10: if it still fails, the error text it now shows is needed.
- RPGAssistant (separate repo `druidican/rpgassistant`): 8 changes on branch `wip-8-changes`, built and
  selftest passing, browser test and version stamp still to do, then merge to main.
