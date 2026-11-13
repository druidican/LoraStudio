# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Settings > Launch options: how LoRA Studio starts (network, port, browser, login, start page, ...).

Saved in launch_settings.json next to the program (readable only by you: it can hold the login password).
lora_studio.py reads it at start; an option given on the command line (./start.sh --port 7861) still wins.
"""
from __future__ import annotations
__version__ = "1.131126c"  # LoRA Studio version, see version.py

import json
import os
import sys
import threading
import time

import lspaths as _LP  # noqa: E402  (program/lspaths.py: ROOT, PROG, USER)
APP_DIR = _LP.USER
FILE = APP_DIR / "launch_settings.json"
START_PAGES = [("Home", "home"), ("Datasets", "datasets"), ("Queue", "queue"), ("Collection", "collection"),
               ("Settings", "settings")]
DEFAULTS = {"network": False, "port": 7860, "browser": True, "user": "", "password": "", "start_page": "home",
            "check_updates": True, "resume_queue": False, "notify": True, "queue_autostart": True,
            "home_page": True}          # set once Home exists (1.131126): the old default start page becomes Home


def load() -> dict:
    try:
        d = json.loads(FILE.read_text(encoding="utf-8"))
        if "home_page" not in d and d.get("start_page") == "collection":
            d["start_page"] = "home"            # Collection was the start page before Home existed
        if d.get("start_page") == "training":
            d["start_page"] = "queue"           # Training became the Queue tab (1.131126b)
        return {**DEFAULTS, **{k: d[k] for k in DEFAULTS if k in d}}
    except Exception:
        return dict(DEFAULTS)


def save(v: dict):
    tmp = FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps({**DEFAULTS, **v}, indent=1), encoding="utf-8")
    os.chmod(tmp, 0o600)
    tmp.replace(FILE)


def restart(delay: float = 1.5, extra: list | None = None):
    """Start lora_studio.py again with the saved options (the command-line ones that the file now covers dropped)."""
    def go():
        time.sleep(delay)
        drop_val = {"--host", "--port", "--auth"}
        args, skip = [], False
        for a in sys.argv[1:]:
            if skip:
                skip = False
                continue
            if a in drop_val:
                skip = True
                continue
            if any(a.startswith(f + "=") for f in drop_val) or a in ("--no-browser", "--setup"):
                continue
            args.append(a)
        args.append("--no-browser")                   # the open page reloads by itself
        args += list(extra or [])
        print("[LoRA Studio] restarting with the new launch options ...", flush=True)
        script = os.path.abspath(sys.argv[0]) if sys.argv[0].endswith(".py") else str(_LP.ROOT / "lora_studio.py")
        os.execv(sys.executable, [sys.executable, script] + args)
    threading.Thread(target=go, daemon=True).start()

