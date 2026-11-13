#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""
LoRA Studio: train LoRAs with sd-scripts and musubi-tuner, make the datasets, keep the Collection

  Home · Datasets (a page per dataset: Prepare, Train, Results, LoRA) · Queue · Collection · Settings
  A status strip on every page shows the training run and the machine (GPU, VRAM, CPU, RAM).

Run:  python lora_studio.py [--port 7860] [--host 127.0.0.1] [--auth user:password] [--no-browser]
"""
from __future__ import annotations
__version__ = "1.131126b"  # LoRA Studio version, see version.py

import argparse
import os
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "program"))
import lspaths  # noqa: E402
lspaths.bootstrap()
lspaths.migrate()
USER, PROG = lspaths.USER, lspaths.PROG


def apply_folders(s: dict):
    """Point the trainer at the folders from library_settings.txt (also after saving Settings)."""
    import sdgui
    sdgui.ROOT = Path(os.path.expanduser(s["sd_scripts_dir"])).absolute()
    datasets = [Path(os.path.expanduser(d)).absolute() for d in s["dataset_dirs"]]
    sdgui.IMAGES_DIR = datasets[0] if datasets else None
    sdgui.EXTRA_SCRIPT_DIRS[:] = datasets[1:] + [Path(os.path.expanduser(d)).absolute() for d in s["script_dirs"]]
    sdgui.DEFAULT_MODEL_DIR = os.path.expanduser(s.get("models_dir") or "")


def use_own_config_folder():
    """The training settings and the Hugging Face token live in userdata/, not inside sd-scripts."""
    import sdgui
    sdgui.CONFIG_DIR, sdgui.SETTINGS_FILE = USER, "training_settings.txt"


def setup_page_first():
    """First start (no settings yet) or --setup: the setup page instead of the program. It runs BEFORE lora_library
    and sdgui are imported, because those make default settings files when there are none. When it finishes, it
    starts this program again (same port) with the new settings."""
    import firstrun
    if os.environ.get("LS_PAGE_CHECK") or ("--setup" not in sys.argv and not firstrun.needs_setup()):
        return
    import launchopts
    lo = launchopts.load()
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--port", type=int, default=lo["port"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--auth")
    ap.add_argument("--no-browser", action="store_true")
    a, _ = ap.parse_known_args()
    auth = a.auth or os.environ.get("SDGUI_AUTH")
    auth = tuple(auth.split(":", 1)) if auth and ":" in auth else None
    firstrun.run(a.host, a.port, not a.no_browser, auth)
    sys.exit(0)


def start_tasks(lo: dict):
    """The quiet jobs after the start: the Collection scan and thumbnails, the queue, old launch scripts, the
    program's own folders, the training venv check and the trainers' update check."""
    import lora_library as LL
    import sdgui
    try:                                   # the pictures on the Datasets pages: links to small files, not inline data
        import time
        import datacheck
        datacheck.URL_DIR["dir"] = LL.THUMB_DIR / "pages"
        for f in (LL.THUMB_DIR / "pages").glob("*.jpg"):           # old page pictures: made again when needed
            if time.time() - f.stat().st_mtime > 30 * 86400:
                f.unlink(missing_ok=True)
    except Exception:                      # noqa: BLE001
        pass
    threading.Thread(target=lambda: (LL.LIB.ensure_loaded(), LL.warm_thumbnails(LL.LIB)),
                     daemon=True, name="library-load").start()
    note = sdgui.queue_restore(start=lo["resume_queue"])
    if note:
        print(f"[Queue] {note}")
    n_old = sdgui.archive_all_datasets()
    if n_old:
        print(f"[Scripts] {n_old} older launch script(s) moved to {sdgui.OLD_RUNS}/ (one script per LoRA type is shown)")
    try:                                   # an update from before 1.251026d: bring the manual and the logo
        import selfupdate
        selfupdate.repair_dirs()
    except Exception as e:                 # noqa: BLE001
        print(f"[Updates] {e!r}")
    try:                                   # PyTorch and its ROCm / CUDA parts in the training venv (no download)
        import venvcheck
        venvcheck.start_check()
    except Exception as e:                 # noqa: BLE001
        print(f"[Training venv] {e!r}")
    if lo["check_updates"]:
        try:                               # quiet check for sd-scripts / musubi-tuner updates (never installs)
            import updater
            updater.start_background_check()
        except Exception as e:             # noqa: BLE001
            print(f"[Updates] not available: {e!r}")


def main():
    import launchopts
    lo = launchopts.load()
    ap = argparse.ArgumentParser(description="LoRA Studio: your LoRA collection, datasets and training")
    ap.add_argument("--port", type=int, default=lo["port"])
    ap.add_argument("--host", default="0.0.0.0" if lo["network"] else "127.0.0.1")
    ap.add_argument("--auth", help="login as user:password; or set SDGUI_AUTH")
    ap.add_argument("--venv", help="venv for the trainer (default: <sd-scripts>/venv)")
    ap.add_argument("--datatypes", "--scan-script", dest="scan_script",
                    help="path to Lora_Datatypes.py, the file with all LoRA types (default: next to this file)")
    ap.add_argument("--allow-path", action="append", default=[],
                    help="extra folder the pages may show pictures from (repeatable)")
    ap.add_argument("--no-browser", action="store_true", default=not lo["browser"])
    ap.add_argument("--setup", action="store_true", help="show the setup page again (folders, token, backup import)")
    args = ap.parse_args()

    import lora_library as LL
    import sdgui
    s = LL.LIB.settings
    apply_folders(s)
    LL.ON_SETTINGS_SAVED.append(apply_folders)
    use_own_config_folder()
    sdgui.DEFAULT_VENV = os.path.expanduser(args.venv) if args.venv else sdgui.detect_venv()
    sdgui.SCAN_SCRIPT = sdgui.find_scan_script(args.scan_script)
    sdgui.apply_gfx_env()
    sdgui.BIND_HOST = args.host

    def sd_dir_saved(path):                 # Settings › Trainers and GPU is the one place for the sd-scripts folder
        v = LL.load_settings()
        if os.path.expanduser(v.get("sd_scripts_dir") or "") != os.path.expanduser(path):
            v["sd_scripts_dir"] = path
            LL._write_values(v)
            LL.LIB.settings["sd_scripts_dir"] = path
    sdgui.ON_SD_DIR.append(sd_dir_saved)

    check_only = bool(os.environ.get("LS_PAGE_CHECK"))   # selftest.py: build the page once, start nothing else
    auth = None if check_only else args.auth or os.environ.get("SDGUI_AUTH") or (f"{lo['user']}:{lo['password']}"
                                                           if lo["user"] and lo["password"] else None)
    from nicegui import app, ui
    import login
    import shell
    import widgets  # the file routes
    widgets.EXTRA_ROOTS[:] = [Path(os.path.expanduser(p)) for p in args.allow_path] + [Path(sdgui.CACHE_DIR)]

    if auth:
        user, sep, pw = auth.partition(":")
        if not sep or not user or not pw:
            ap.error("--auth must look like user:password")
        login.install(user, pw)

    @ui.page("/")
    def index():
        if auth and not app.storage.user.get("ok"):
            return ui.navigate.to("/login")
        shell.build_page()

    import version
    print(version.banner())
    if not check_only:
        start_tasks(lo)
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"\n*** Listening on {args.host}:{args.port}: anyone on your network can open the program. "
              f"{'A login is required.' if auth else 'No login is set: use --auth user:password.'} ***\n")
    fav = PROG / "logo" / "favicon.png"
    ui.run(host=args.host, port=args.port, title="LoRA Studio", favicon=fav if fav.is_file() else None,
           show=not args.no_browser, reload=False, storage_secret=login.storage_secret(), show_welcome_message=True)


if __name__ == "__main__":
    setup_page_first()
    main()
