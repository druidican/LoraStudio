# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Settings > Launch options: how LoRA Studio starts (network, port, browser, login, start page, ...).

Saved in launch_settings.json next to the program (readable only by you: it can hold the login password).
lora_studio.py reads it at start; an option given on the command line (./start.sh --port 7861) still wins.
"""
from __future__ import annotations
__version__ = "1.011126"  # LoRA Studio version, see version.py

import json
import os
import sys
import threading
import time

import lspaths as _LP  # noqa: E402  (program/lspaths.py: ROOT, PROG, USER)
APP_DIR = _LP.USER
FILE = APP_DIR / "launch_settings.json"
START_PAGES = [("Collection", "collection"), ("Datasets", "datasets"), ("Training", "training"), ("Settings", "settings")]
DEFAULTS = {"network": False, "port": 7860, "browser": True, "user": "", "password": "", "start_page": "collection",
            "check_updates": True, "resume_queue": False, "notify": True}


def load() -> dict:
    try:
        d = json.loads(FILE.read_text(encoding="utf-8"))
        return {**DEFAULTS, **{k: d[k] for k in DEFAULTS if k in d}}
    except Exception:
        return dict(DEFAULTS)


def save(v: dict):
    tmp = FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps({**DEFAULTS, **v}, indent=1), encoding="utf-8")
    os.chmod(tmp, 0o600)
    tmp.replace(FILE)


def cli_given(flag: str) -> bool:
    return any(a == flag or a.startswith(flag + "=") for a in sys.argv[1:])


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
        os.execv(sys.executable, [sys.executable, str(_LP.ROOT / "lora_studio.py")] + args)
    threading.Thread(target=go, daemon=True).start()


def build_ui():
    import gradio as gr
    v = load()
    gr.Markdown("How LoRA Studio starts. Saved in `launch_settings.json` (readable only by you); used by `./start.sh` "
                "and the desktop shortcut from the next start. An option typed on the command line still wins.")
    with gr.Row(equal_height=True):
        net = gr.Checkbox(label="Reachable on the network", value=v["network"], scale=2,
                          info="Other computers on your network can open LoRA Studio (needs a login below).")
        port = gr.Number(label="Port", value=v["port"], precision=0, minimum=1024, maximum=65535, scale=1,
                         info="The address ends in :port")
    with gr.Row(equal_height=True):
        user = gr.Textbox(label="Login: user name", value=v["user"], scale=1)
        pw = gr.Textbox(label="Login: password", value=v["password"], type="password", scale=1,
                        info="Required when reachable on the network; also asked on this computer.")
    with gr.Row(equal_height=True):
        browser = gr.Checkbox(label="Open the browser at start", value=v["browser"], scale=1,
                              info="Off for a computer without a screen.")
        start = gr.Dropdown(label="Start page", choices=START_PAGES, value=v["start_page"], scale=1)
    with gr.Row(equal_height=True):
        upd = gr.Checkbox(label="Check for trainer updates at start", value=v["check_updates"], scale=1,
                          info="Only looks; installing is always your click (Settings > Updates).")
        resume = gr.Checkbox(label="Resume the queue after a restart", value=v["resume_queue"], scale=1,
                             info="Waiting items start again; a run that was cut off is marked stopped.")
        notify = gr.Checkbox(label="Notice when a run finishes", value=v["notify"], scale=1,
                             info="A browser notification (the browser asks once).")
    with gr.Row(elem_classes=["ls-bar"]):
        save_b = gr.Button("Save", scale=1, min_width=120)
        restart_b = gr.Button("Save and restart", variant="primary", scale=1, min_width=180)
    msg = gr.Markdown()

    def collect(n, p, u, w, b, s, c, r, no):
        return {"network": bool(n), "port": int(p or 7860), "browser": bool(b), "user": (u or "").strip(),
                "password": w or "", "start_page": s or "collection", "check_updates": bool(c),
                "resume_queue": bool(r), "notify": bool(no)}

    def check(d) -> str:
        if d["network"] and not (d["user"] and d["password"]):
            return ("❌ Not saved: reachable on the network needs a user name and password, otherwise anyone on the "
                    "network could run commands through LoRA Studio.")
        if bool(d["user"]) != bool(d["password"]):
            return "❌ Not saved: give both a user name and a password, or neither."
        if not 1024 <= d["port"] <= 65535:
            return "❌ Not saved: the port must be between 1024 and 65535."
        return ""

    def do_save(*a):
        d = collect(*a)
        err = check(d)
        if err:
            return err
        save(d)
        given = [f for f in ("--host", "--port", "--auth", "--no-browser") if cli_given(f)]
        note = (f" This start used {', '.join(given)} from the command line; those win until you start without them."
                if given else "")
        return "✔ Saved. Used from the next start." + note

    def do_restart(*a):
        d = collect(*a)
        err = check(d)
        if err:
            return err
        save(d)
        host = "this computer's network address" if d["network"] else "127.0.0.1"
        restart()
        return (f"✔ Saved. **Restarting** at `http://{host}:{d['port']}` ... this page reloads by itself when the port "
                "stays the same; after a port change, open the new address.")

    ins = [net, port, user, pw, browser, start, upd, resume, notify]
    save_b.click(do_save, ins, msg)
    restart_b.click(do_restart, ins, msg)
