#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""
LoRA Studio on NiceGUI (being built; replaces lora_studio.py when complete, see docs/NICEGUI_PLAN.md)

  Home · Datasets (a page per dataset: Prepare, Train, Results, LoRA) · Queue · Collection · Settings
  A status strip on every page shows the training run and the machine (GPU, VRAM, CPU, RAM).

Run:  python lora_studio_ng.py [--port 7860] [--host 127.0.0.1] [--auth user:password] [--no-browser]
"""
from __future__ import annotations
__version__ = "1.131126b"  # LoRA Studio version, see version.py

import argparse
import os
import secrets
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


def storage_secret() -> str:
    """The key that signs the browser's login cookie (made once, kept in userdata)."""
    f = USER / ".session_key"
    try:
        return f.read_text().strip()
    except OSError:
        k = secrets.token_hex(32)
        USER.mkdir(exist_ok=True)
        f.write_text(k)
        f.chmod(0o600)
        return k


def main():
    import launchopts
    lo = launchopts.load()
    ap = argparse.ArgumentParser(description="LoRA Studio: your LoRA collection, datasets and training")
    ap.add_argument("--port", type=int, default=lo["port"])
    ap.add_argument("--host", default="0.0.0.0" if lo["network"] else "127.0.0.1")
    ap.add_argument("--auth", help="login as user:password; or set SDGUI_AUTH")
    ap.add_argument("--venv", help="venv for the trainer (default: <sd-scripts>/venv)")
    ap.add_argument("--no-browser", action="store_true", default=not lo["browser"])
    args = ap.parse_args()

    import lora_library as LL
    import sdgui
    s = LL.LIB.settings
    apply_folders(s)
    LL.ON_SETTINGS_SAVED.append(apply_folders)
    sdgui.CONFIG_DIR, sdgui.SETTINGS_FILE = USER, "training_settings.txt"
    sdgui.DEFAULT_VENV = os.path.expanduser(args.venv) if args.venv else sdgui.detect_venv()
    sdgui.SCAN_SCRIPT = sdgui.find_scan_script(None)
    sdgui.apply_gfx_env()
    sdgui.BIND_HOST = args.host

    auth = args.auth or os.environ.get("SDGUI_AUTH") or (f"{lo['user']}:{lo['password']}"
                                                           if lo["user"] and lo["password"] else None)
    from nicegui import app, ui
    import shell
    import widgets  # noqa: F401  (the file routes)

    if auth:
        user, _, pw = auth.partition(":")
        _login(app, ui, user, pw)

    @ui.page("/")
    def index():
        if auth and not app.storage.user.get("ok"):
            return ui.navigate.to("/login")
        shell.build_page()

    threading.Thread(target=LL.LIB.ensure_loaded, daemon=True, name="library-load").start()
    note = sdgui.queue_restore(start=lo["resume_queue"])
    if note:
        print(f"[Queue] {note}")
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"\n*** Listening on {args.host}:{args.port}: anyone on your network can open the program. "
              f"{'A login is required.' if auth else 'No login is set: use --auth user:password.'} ***\n")
    fav = PROG / "logo" / "favicon.png"
    ui.run(host=args.host, port=args.port, title="LoRA Studio", favicon=fav if fav.is_file() else None,
           show=not args.no_browser, reload=False, storage_secret=storage_secret(), show_welcome_message=True)


def _login(app, ui, user: str, pw: str):
    """A login page; the browser keeps a signed cookie, so a reload does not ask again."""
    import hmac

    @ui.page("/login")
    def login():
        def check():
            if hmac.compare_digest(u.value, user) and hmac.compare_digest(p.value, pw):
                app.storage.user["ok"] = True
                ui.navigate.to("/")
            else:
                msg.text = "Wrong user name or password."
        ui.query("body").style("background:#1C1F24")
        with ui.card().classes("absolute-center w-80 items-stretch"):
            ui.label("LoRA Studio").classes("text-xl font-bold text-center")
            u = ui.input("User name").props("outlined dense")
            p = ui.input("Password", password=True).props("outlined dense").on("keydown.enter", check)
            ui.button("Log in", on_click=check).props("unelevated no-caps color=primary")
            msg = ui.label("").classes("text-red-600 text-sm")


if __name__ == "__main__":
    main()
