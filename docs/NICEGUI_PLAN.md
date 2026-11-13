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

## How it is built (as done)

- `lora_studio.py` is the NiceGUI program (`ui.run`, login, the setup page first when there are no settings).
- `program/UI/`: the pages: `shell.py` (header, tabs, status strip, ? manual, notice), `pg_home.py`,
  `pg_datasets.py` (list, Improvements, the dataset page) with `ds_steps.py` (steps 2-8, LoRA), `ds_train.py` (step 9),
  `ds_results.py` (Results), `ds_batch.py` (Fix the ticked); `pg_queue.py`; `pg_collection.py` + `lora_page.py`;
  `pg_settings.py` + `set_panels.py`; `setup_page.py` (first start), `login.py`, `widgets.py`, `theme.css`.
- The logic stays in the existing modules. `datasets_ui.py`, `training_ui.py`, `settings_ui.py` and `home.py` keep
  only their HTML renderers and data functions (no Gradio). Clicks inside that HTML call `lsSend(box, v)`, which the
  shell turns into the NiceGUI event `ls_<box>` (shell.BRIDGE_JS); `lsSetNav`, `lsGoQueue`, `lsGoRun`, `lsOpenLora`,
  `lsManual` likewise. Short messages from the logic go through `Other/notify.py`.
- Live parts are `ui.timer`s that redraw only what changed (`set_html` keeps the last HTML per part).
- Pictures are links: `widgets.pic_url` (/pic/…, thumbnails from the Collection's cache) and `lspaths.file_url`
  (/lsfile/…, only inside the allowed folders, `widgets.allowed_roots`). `datacheck.URL_DIR` is set at the start.
- `selftest.py` starts the program once with `LS_PAGE_CHECK=1` (no login, queue or updates) and opens the page.

## Progress

- [x] Test page: Collection › Browse (`try-nicegui/`), owner: "amazingly fast".
- [x] Shell: tabs, status strip, login, theme, ? help (first version)
- [x] Collection: Browse, LoRA page (Overview, Pictures, Training, Edit), Insights, Maintenance (logic in Gallery/lorapage.py, maint.py)
- [x] Home (+ Just trained: the last queue's runs with their test pictures; the status strip opens it)
- [x] Queue (+ test pictures after a queue: one LoRA best + final, several the final file of each; comfytest.WHICH, last_batch.json)
- [x] Datasets: list, new dataset, dataset page (steps 2-8, Train, Results, LoRA), Improvements with *Fix the ticked*
- [x] Settings: every group (UI/pg_settings.py, UI/set_panels.py); the first-start setup page (UI/setup_page.py)
- [x] Gradio removed (ticker, loadjoin, livepoll deleted; the old build functions gone); selftest; install/start scripts
- [x] Manual text and new pip screenshots (`tools/make_test_world.py`: the pretend trainer now writes a loss log)
- [x] Beta release 1.131126c
- [x] When complete (owner, 2026-11-13): the old `main` kept as branch `old`, `nicegui` became the new `main`.
