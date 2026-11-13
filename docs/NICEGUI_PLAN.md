# LoRA Studio on NiceGUI: the plan

Owner's decisions (2026-11-13):

- The whole program moves from Gradio to NiceGUI ("a complete redo"), built **all at once** on the branch `nicegui`;
  `main` stays the Gradio version until the new one is complete.
- **Look:** today's layout (the redesign: Home, the dataset page with Prepare / Train / Results / LoRA, Queue,
  Collection, Settings), redrawn cleaner with NiceGUI's controls.
- **Order:** Collection, Home, Queue, Datasets, Settings.
- **Releases:** when complete, as betas (updater channel *Betas too*); *Releases only* stays on Gradio until the owner
  says the new version is good.
- Home's *Continue where you left off* stays as it is.
- **New: after training.** The status strip and Home link straight to the new run's test pictures. One LoRA trained:
  test pictures of its best epoch and its latest (final) file. Several LoRAs trained (a queue): the final file of
  each new LoRA, made together on one page.

## How it is built

- `lora_studio.py` becomes a NiceGUI program (during the work: `lora_studio_ng.py`, renamed at the end).
- `program/UI/`: the pages, one file per page, no logic in them:
  `shell.py` (header, tabs, status strip, ? help, login, theme), `home.py`, `collection.py`, `lora_page.py`,
  `queue.py`, `datasets.py` (list, Improvements), `dataset_page.py` (+ one file per step where large),
  `results.py`, `settings.py`, `manual.py`, `widgets.py` (cards, chips, picture grids, confirm dialogs).
- The logic stays in the existing modules (`lora_library`, `sdgui`, `dsflow`, `sd_script`, …) and loses every Gradio
  part: `import gradio` is gone from all of them at the end, and `datasets_ui.py`, `training_ui.py`, `settings_ui.py`,
  `loadjoin.py`, `livepoll.py`, `ticker.py` and the JavaScript bridges are deleted (NiceGUI updates elements directly).
- Live parts (status strip, queue, run progress) are `ui.timer`s that change only what changed.
- Pictures are served by the program (thumbnails from the Collection's cache), never as inline data.
- `requirements.txt` gets `nicegui` and loses `gradio` at the end.

## Notes for the next session

- Test world: `tools/make_test_world.py /tmp/ngw`, then copy `lora_studio_ng.py`, `program/`, `version.py` into
  `/tmp/ngw/app` and run it with `HOME=/tmp/ngw/home`; `tools/fake_comfy.py` + `userdata/comfy_settings.json` for test
  pictures. HTML parts drawn by the old renderers send `lsSend(box, v)` → NiceGUI event `ls_<box>` (shell.BRIDGE_JS).

## Progress

- [x] Test page: Collection › Browse (`try-nicegui/`), owner: "amazingly fast".
- [x] Shell: tabs, status strip, login, theme, ? help (first version)
- [x] Collection: Browse, LoRA page (Overview, Pictures, Training, Edit), Insights, Maintenance (logic in Gallery/lorapage.py, maint.py)
- [x] Home (+ Just trained: the last queue's runs with their test pictures; the status strip opens it)
- [x] Queue (+ test pictures after a queue: one LoRA best + final, several the final file of each; comfytest.WHICH, last_batch.json)
- [x] Datasets: list, new dataset, dataset page (steps 2-8 in UI/ds_steps.py, Train UI/ds_train.py, Results
  UI/ds_results.py, LoRA). Still open: Improvements' *Fix the ticked datasets* (batch, settings_ui.batch_panel).
- [ ] Settings (all groups), first-run setup page
- [ ] Manual text and pip screenshots; selftest; Gradio removed; beta release
