#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""
sdgui.py - the training engine of LoRA Studio, and the shared parts of the launch script and dataset pages.

  runs           Run: one training run (bash on a temporary copy of the launch script, the log streamed and kept in
                 userdata/gui_logs/), its stages (##LS_STAGE markers of the musubi pipelines), loss and NaN watch
  the queue      QUEUE and _queue_worker: one script after the other, making room in video memory when the model
                 changes, stop on fail, Completed marks after a run that wrote its final LoRA
  launch scripts reading and writing their options (the form), the LoRA types (Lora_Datatypes.py) and the
                 recommendations; split over sd_script.py, sd_prepare.py and sd_settings.py, imported back here
  build_ui(app)  the training parts of the page, built hidden in BLOCKS for lora_studio.py / datasets_ui.py to place

Security: this runs shell text from a web page and edits files. LoRA Studio binds to 127.0.0.1 unless you choose
otherwise, with a login then (Settings › Start and network).
"""
__version__ = "1.111126e"  # LoRA Studio version, see version.py
import atexit
import codecs
import contextlib
import glob
import hashlib
import importlib.util
import inspect
import html
import json
import math
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import gradio as gr

import lspaths as _LP  # noqa: E402

try:                                   # shared "Completed" marks (completed.py next to this file)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import completed as COMPLETED
except Exception:                      # running without it: nothing is ever hidden
    COMPLETED = None
try:                                   # which model files belong together (Settings > Training suggestions)
    import model_guide as GUIDE
except Exception:                      # noqa: BLE001
    GUIDE = None
try:
    import datacheck as DC                      # picture checks for a dataset (Create Data Launch Script tab)
except Exception:                               # noqa: BLE001
    DC = None
try:
    import housekeeping as HK                   # copy to LoRA folder, Trash, leftovers (no UI inside)
except Exception:                               # noqa: BLE001
    HK = None
try:
    import captioner as CAP                     # Recaption with Ollama (Create Data Launch Script tab)
except Exception:                               # noqa: BLE001 - the tab shows that it is missing
    CAP = None

HOME = Path.home()
LOG_DIR = _LP.USER / "gui_logs"
MAX_LOG_LINES = 400      # lines shown in the UI
MAX_KEPT_LINES = 5000    # lines kept in memory (full log is also on disk)
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp"}
VAR_RE = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
TRAIN_RE = re.compile(r"\b[\w.-]*train[\w.-]*\.py\b")
# the tqdm line sd-scripts prints while training, e.g.
#   steps:  12%|█▏        | 300/2430 [05:12<36:30,  1.03s/it, avr_loss=0.0831]
PROG_RE = re.compile(r"(steps:\s*\d+%\|[^|\n]*\|\s*\d+/\d+\s*\[[^\]\n]*\])")
NO_STATE = "none"
STATE_MARKERS = ("train_state.json", "random_states_0.pkl", "optimizer.bin")

ROOT: Path = HOME / "sd-scripts"
CONFIG_DIR: Path | None = None   # where settings.txt / huggingtoken.txt live; None = <root> (lora_studio.py sets its own folder)
DEFAULT_MODEL_DIR = ""            # folder for model-file slots that have none yet (set by lora_studio.py)
IMAGES_DIR: Path | None = None   # dataset folder; None = <root>/images (set by --images-dir or lora_studio.py)
EXTRA_SCRIPT_DIRS: list[Path] = []
DEFAULT_VENV = ""
SCAN_SCRIPT: Path | None = None
CACHE_DIR = _LP.USER / "gui_cache"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
CAPTION_EXTS = (".txt", ".caption")
REPEAT_RE = re.compile(r"^(\d+)_")
IGNORE_DIRS = {"model", "models", "output", "outputs", "samples", "sample", "log", "logs",
               "loramanager", "_set_aside", "_reserve", "_originals", ".lsflow", "_old_runs", "_extra"}   # never dataset folders (LoraManager = Personal sync: previews and examples)
ARCHIVE_RENAMED_RE = re.compile(r"^\d+_loramanager$", re.I)   # a LoraManager folder renamed into a repeat folder
CSS = ('#wm_tool {display:none !important;} '
       '.ls-bar {gap: 8px !important; align-items: center !important; flex-wrap: wrap;} '
       '.ls-bar > * {flex-grow: 0;} .ls-bar .block {padding: 4px 8px !important;} '
       '.ls-prog {position: relative; height: 32px; border-radius: 8px; overflow: hidden; background: var(--block-background-fill); border: 1px solid var(--border-color-primary);} '
       '.ls-prog-fill {position: absolute; left: 0; top: 0; bottom: 0; transition: width .9s ease; background: linear-gradient(90deg, #a85a08, #e8a33d);} '
       '.ls-prog-ok .ls-prog-fill {background: linear-gradient(90deg, #1f7a72, #4fb3a9);} '
       '.ls-prog-bad .ls-prog-fill {background: linear-gradient(90deg, #8e2a1f, #f07c6a);} '
       ".ls-prog-text {position: relative; z-index: 1; line-height: 32px; padding: 0 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 13px; text-shadow: 0 1px 2px rgba(0,0,0,.45);} "
       '#loss_box .lossbox {--ls-ok:#2e9e5b; --ls-warn:#d98a1a; --ls-bad:#d64545; --ls-info:#8a93a0; --ls-line:#4c7fd6; padding: 10px 12px; border-radius: 8px; background: rgba(128,128,128,.10); font-size: .9rem;} '
       '#loss_box .lossmuted {opacity: .72; font-size: .8rem;} '
       '#loss_box .losscells {display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 8px 0;} '
       '#loss_box .losscell {padding: 6px 10px; background: rgba(128,128,128,.12); border-radius: 6px;} '
       '@media (max-width: 700px) {#loss_box .losscells {grid-template-columns: 1fr;}} '
       '#loss_box .losschart {display:block; background: rgba(128,128,128,.08); border-radius: 6px;} '
       '#progress_line textarea, #progress_line input {font-family: monospace; font-size: 0.95em;} '
       '#stage_box .lsstage {padding: 8px 10px; border-radius: 8px; background: rgba(128,128,128,.10);} '
       '#stage_box .lschips {display: flex; flex-wrap: wrap; gap: 6px;} '
       '#stage_box .lschip {padding: 3px 10px; border-radius: 14px; font-size: .85rem; border: 1px solid var(--border-color-primary); background: rgba(128,128,128,.12); opacity: .75;} '
       '#stage_box .lschip-run {background: #a85a08; color: #fff; opacity: 1; border-color: #e8a33d;} '
       '#stage_box .lschip-done {background: #1f7a72; color: #fff; opacity: 1; border-color: #4fb3a9;} '
       '#stage_box .lschip-fail {background: #8e2a1f; color: #fff; opacity: 1; border-color: #f07c6a;} '
       '#stage_box .lschip-stopped {background: #5b6470; color: #fff; opacity: 1;} '
       '#stage_box .lsdur {opacity: .8; margin-left: 6px; font-size: .78rem;} '
       '#stage_box .lsmsg {margin-top: 6px; font-size: .9rem;} #stage_box .lsbad {color: #f07c6a;} '
       '#stage_box .lsok {color: #4fb3a9;} '
       '#wm_html .wm-row {display: flex; gap: 12px; align-items: flex-start; padding: 10px 0; border-bottom: 1px solid var(--border-color-primary); flex-wrap: wrap;} '
       '#wm_html .wm-src {width: 260px;} '
       '#wm_html .wm-src img {max-width: 260px; max-height: 260px; border-radius: 6px; display: block;} '
       '#wm_html .wm-src .wm-name {font-weight: 600; font-size: .85rem; margin-top: 4px; word-break: break-all;} '
       '#wm_html .wm-src .wm-marks {font-size: .8rem; opacity: .85;} '
       '#wm_html .wm-opts {display: flex; gap: 10px; flex-wrap: wrap; flex: 1;} '
       '#wm_html .wm-opt {all: unset; cursor: pointer; width: 200px; border: 2px solid var(--border-color-primary); border-radius: 8px; padding: 6px; display: flex; flex-direction: column; gap: 4px; font-size: .8rem;} '
       '#wm_html .wm-opt img {max-width: 200px; max-height: 200px; border-radius: 4px; display: block; margin: 0 auto;} '
       '#wm_html .wm-opt .wm-big {height: 120px; display: flex; align-items: center; justify-content: center; font-size: 1.6rem; opacity: .6;} '
       '#wm_html .wm-opt.sel {border-color: #e8a33d; background: rgba(232,163,61,.12);} '
       '#wm_html .wm-opt b {font-size: .85rem;} #wm_html .wm-now {color: #4fb3a9; font-weight: 600;} '
       '#cc_box {display: none !important;} '
       '#cc_html input[type=radio] {appearance: auto; -webkit-appearance: radio; accent-color: #E8A33D; width: 14px; height: 14px; margin: 0 4px 0 0; vertical-align: -2px; opacity: 1;} '
       '#cc_html label {cursor: pointer;} '
       '#cc_html .cc-group {border: 1px solid var(--border-color-primary); border-radius: 10px; padding: 10px 12px; margin: 0 0 10px;} '
       '#cc_html .cc-head {display: flex; gap: 10px; align-items: baseline; flex-wrap: wrap;} '
       '#cc_html .cc-trait {font-weight: 700; font-size: 1rem;} '
       '#cc_html .cc-cov {opacity: .7; font-size: .85rem;} '
       '#cc_html .cc-kind {font-size: .72rem; font-weight: 600; padding: 1px 8px; border-radius: 99px;} '
       '#cc_html .cc-lock {background: #262A31; color: #F4C06F;} '
       '#cc_html .cc-exc {margin-top: 8px; font-size: .82rem; opacity: .8;} '
       '#cc_html .cc-contra {background: #FBE9C8; color: #8F560B;} '
       '#cc_html .cc-same {background: #21857C; color: #fff;} '
       '#cc_html .cc-vals {display: flex; gap: 8px; flex-wrap: wrap; margin: 8px 0;} '
       '#cc_html .cc-val {padding: 3px 10px; border-radius: 8px; border: 1px solid var(--border-color-primary); font-size: .88rem;} '
       '#cc_html .cc-val.cc-main {border-color: #E8A33D;} #cc_html .cc-val small {opacity: .7;} '
       '#cc_html .cc-word {font: inherit; font-weight: 600; background: transparent; color: inherit; border: 0; border-bottom: 1px dashed #E8A33D;} '
       '#cc_html .cc-modes {display: flex; gap: 6px 18px; flex-wrap: wrap; font-size: .9rem;} '
       '#cc_html .cc-modes i {color: #21857C; font-style: normal; font-size: .78rem; margin-left: 4px;} '
       '#cc_html .cc-oddhead {margin: 10px 0 6px; font-size: .85rem; opacity: .85;} '
       '#cc_html .cc-odds {display: flex; gap: 10px; flex-wrap: wrap;} '
       '#cc_html .cc-odd {display: flex; gap: 8px; width: 330px; padding: 6px; border: 2px solid var(--border-color-primary); border-radius: 8px;} '
       '#cc_html .cc-odd.cc-sel {border-color: #E8A33D;} #cc_html .cc-look {all: unset; cursor: zoom-in;} '
       '#cc_html .cc-look img {width: 96px; height: 96px; object-fit: cover; border-radius: 6px; display: block;} '
       '#cc_html .cc-oddtxt {display: flex; flex-direction: column; gap: 3px; font-size: .82rem; min-width: 0;} '
       '#cc_html .cc-name {font-weight: 600; word-break: break-all;} '
       '#cc_view {position: sticky; top: 8px; align-self: flex-start;} '
       '#dc_html .dctable {width: 100%; border-collapse: collapse;} '
       '#dc_html .dctable td {vertical-align: top; padding: 6px;} '
       '#dc_html .dctable tr {border-bottom: 1px solid var(--border-color-primary);} '
       '#dc_html img {width: 120px; height: 120px; object-fit: cover; border-radius: 6px;} '
       '#dc_html ul {margin: 4px 0 0 18px; padding: 0;} #dc_html li {font-size: .88rem;} '
       '#dc_html li.dcbad {color: #f07c6a;} #dc_html li.dcwarn {color: #d9a23d;} '
       '#dc_html .dctag {margin-left: 8px; padding: 1px 8px; border-radius: 10px; font-size: .75rem; color: #fff;} '
       '#dc_html .dctag-bad {background: #b03a2e;} #dc_html .dctag-warn {background: #a07012;} '
       '#oc_ex .ocex {padding: 8px 12px; border-radius: 8px; background: rgba(128,128,128,.10);} '
       '#oc_ex .ochead {font-size: .82rem; opacity: .85; margin-bottom: 4px;} '
       '#oc_ex .octext {font-family: var(--font-mono, monospace); font-size: .9rem; line-height: 1.45;} '
       '#oc_ex .ocnote {font-size: .8rem; opacity: .85; margin-top: 4px;} '
       '#oc_ex .ocnote.bad {color: #f07c6a; opacity: 1;} '
       '#oc_ex .ocbar {height: 8px; border-radius: 4px; background: rgba(128,128,128,.25); margin-top: 6px; overflow: hidden;} '
       '#oc_ex .ocfill {height: 100%; border-radius: 4px;} #oc_ex .ocfill.ok {background: #1f9e8f;} '
       '#oc_ex .ocfill.bad {background: #d9534f;} '
       '#stage_box .lstail {margin: 6px 0 0; padding: 6px 8px; max-height: 170px; overflow: auto; font-size: .78rem; background: rgba(0,0,0,.18); border-radius: 6px; white-space: pre-wrap;} ')
LAST_CHANGE = {"desc": "", "files": {}}   # last caption edit, for Undo


# --------------------------------------------------------------------------- run state
class Run:
    def __init__(self):
        self.proc = None
        self.lines: list[str] = []
        self.cur = ""            # in-progress line (tqdm uses \r)
        self.lock = threading.Lock()
        self.started = 0.0
        self.ended = None
        self.name = ""
        self.logfile = None
        self.tmp = None
        self.version = 0         # bumps whenever the log changes
        self.secret = ""         # HF token, masked out of the log
        self.progress = ""       # latest "steps: ..." tqdm line
        self.loss: list = []     # (step, avr_loss, epoch) samples
        self.epoch = 0           # current epoch, from the "epoch 5/10" log line
        self.epochs_total = 0
        self.step_offset = 0     # steps already done in a resumed state (sd-scripts restarts its bar at 0)
        self.script = ""         # the launch script this run belongs to (for the Completed mark)
        self.stages: list = []   # all-in-one scripts (musubi types): [{"name", "state": pending|run|done|fail|stopped, "t0", "t1"}]
        self.stage_no = 0        # stage the script announced last (##LS_STAGE i/n name)
        self.fail: dict | None = None     # {"no", "name", "msg", "tail"} from a ##LS_FAIL line
        self.nan_at: dict | None = None   # first NaN/inf avr_loss: {"step", "epoch", "stage"}

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None


RUN = Run()


def fmt(sec: float) -> str:
    sec = int(sec)
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def log_text() -> str:
    with RUN.lock:
        cur = RUN.cur
        # the live progress bar has its own line above the log, so keep it out of the log tail
        tail = RUN.lines[-MAX_LOG_LINES:] + ([cur] if cur and not PROG_RE.search(cur) else [])
    return "\n".join(tail)


_TQDM_RE = re.compile(r"steps:\s*\d+%\|[^|\n]*\|\s*(\d+)/(\d+)\s*\[([^<\]]*)<([^,\]]*),?\s*([^\]]*)\]")


def _hms_to_sec(t: str):
    try:
        parts = [int(x) for x in t.strip().split(":")]
    except ValueError:
        return None
    sec = 0
    for x in parts:
        sec = sec * 60 + x
    return sec


def shift_progress(line: str, offset: int) -> str:
    """sd-scripts restarts its step bar at 0 after --resume. Show the real position instead
    (steps done = offset + n), with percent and time left recalculated."""
    if not offset:
        return line
    m = _TQDM_RE.search(line)
    if not m:
        return line
    n, total = int(m.group(1)), int(m.group(2))
    elapsed, rest = m.group(3), m.group(5)
    done = min(total, offset + n)
    left_steps = max(0, total - done)
    el = _hms_to_sec(elapsed)
    if n > 0 and el:
        s = left_steps * el / n
        eta = fmt(s) if s >= 3600 else f"{int(s) // 60:02d}:{int(s) % 60:02d}"
    else:
        eta = "?"
    pct = int(100 * done / total) if total else 0
    width = 10
    full = int(width * done / total) if total else 0
    bar = "█" * full + " " * (width - full)
    return f"steps: {pct:3d}%|{bar}| {done}/{total} [{elapsed}<{eta}, {rest}] (resumed at {offset})"


def progress_html() -> str:
    """The step counter as a bar that fills with the progress, with the counter text on top."""
    txt = progress_text()
    m = re.search(r"(\d+)/(\d+)", txt)
    pct = 0.0
    if m and int(m.group(2)):
        pct = min(100.0, 100.0 * int(m.group(1)) / int(m.group(2)))
    state = "idle" if RUN.proc is None else ("run" if RUN.running() else ("ok" if (RUN.proc.returncode == 0) else "bad"))
    clean = re.sub(r"\|[█▏▎▍▌▋▊▉ ]*\|", "|", txt)
    return (f'<div class="ls-prog ls-prog-{state}"><div class="ls-prog-fill" style="width:{pct:.1f}%"></div>'
            f'<div class="ls-prog-text">{html.escape(clean)}</div></div>')


def progress_text() -> str:
    if RUN.proc is None:
        return "steps: idle"
    if RUN.progress:
        return shift_progress(RUN.progress, RUN.step_offset)
    return "steps: waiting for training to start..." if RUN.running() else "steps: no training progress was printed"


# ---- all-in-one scripts (musubi types) announce their stages: "##LS_STAGE 3/6 Cache latents" and, when something
# breaks, "##LS_FAIL 3/6 Cache latents exit=1" (or a message instead of exit=). Studio shows them as a pipeline.
STAGE_RE = re.compile(r"^##LS_STAGE (\d+)/(\d+) (.+?)\s*$")
FAIL_RE = re.compile(r"^##LS_FAIL (\d+)/(\d+) (.*?)\s*$")
DONE_RE = re.compile(r"^##LS_DONE\b")
STAGE_DECL_RE = re.compile(r'^stage "([^"]+)"', re.M)
STOP_CODES = (130, 143, 137, -2, -15, -9)


def _stage_slot(run: Run, no: int, name: str = ""):
    while len(run.stages) < no:
        run.stages.append({"name": "", "state": "pending", "t0": None, "t1": None})
    st = run.stages[no - 1]
    if name:
        st["name"] = name
    return st


def _note_marker(run: Run, line: str):
    """Handle one ##LS_ marker line. Called with run.lock held."""
    if not line.startswith("##LS_"):
        return
    now = time.time()
    m = STAGE_RE.match(line)
    if m:
        no, total = int(m.group(1)), int(m.group(2))
        if 1 <= run.stage_no <= len(run.stages) and run.stages[run.stage_no - 1]["state"] == "run":
            run.stages[run.stage_no - 1].update(state="done", t1=now)
        _stage_slot(run, total)
        st = _stage_slot(run, no, m.group(3))
        st.update(state="run", t0=now, t1=None)
        run.stage_no = no
        return
    m = FAIL_RE.match(line)
    if m:
        no = int(m.group(1))
        st = _stage_slot(run, no)
        st.update(state="fail", t1=now)
        if run.fail is None:
            rest, name = m.group(3), st["name"]
            msg = rest[len(name):].strip() if name and rest.startswith(name) else rest
            tail = [l.rstrip() for l in run.lines[-60:] if l.strip() and not l.startswith("##LS_")][-8:]
            run.fail = {"no": no, "name": name, "msg": msg, "tail": tail}
        return
    if DONE_RE.match(line) and 1 <= run.stage_no <= len(run.stages):
        run.stages[run.stage_no - 1].update(state="done", t1=now)


def _finish_stages(run: Run, code: int):
    """The process has ended: close the stage that was still running. Called with run.lock held."""
    if not run.stages or not (1 <= run.stage_no <= len(run.stages)):
        return
    st = run.stages[run.stage_no - 1]
    if st["state"] != "run":
        return
    st["t1"] = time.time()
    if code == 0:
        st["state"] = "done"
        return
    st["state"] = "stopped" if code in STOP_CODES else "fail"
    if run.fail is None:
        tail = [l.rstrip() for l in run.lines[-60:] if l.strip() and not l.startswith("##LS_")][-8:]
        run.fail = {"no": run.stage_no, "name": st["name"],
                    "msg": "stopped" if code in STOP_CODES else f"exit code {code}", "tail": tail}


def fail_where(run: Run) -> str:
    if run.fail:
        return f"stage {run.fail['no']}/{len(run.stages) or '?'} '{run.fail['name']}'"
    return ""


def nan_text(run: Run) -> str:
    if not run.nan_at:
        return ""
    n = run.nan_at
    return f"loss turned NaN at step {n['step']}" + (f" (epoch {n['epoch']})" if n.get("epoch") else "")


def stage_html() -> str:
    """The pipeline of an all-in-one script: one chip per stage (done / running / failed / waiting), what is happening
    now, and where a failure or a NaN happened. Empty for ordinary single-command scripts."""
    with RUN.lock:
        stages = [dict(s) for s in RUN.stages]
        fail, nan, no = (dict(RUN.fail) if RUN.fail else None), (dict(RUN.nan_at) if RUN.nan_at else None), RUN.stage_no
        prog = RUN.progress
    if RUN.proc is None or not stages:
        return ""
    now = time.time()
    chips = []
    for i, st in enumerate(stages, 1):
        state = st["state"]
        icon = {"done": "✔", "run": "▶", "fail": "✖", "stopped": "■"}.get(state, "○")
        dur = ""
        if st["t0"]:
            dur = " " + fmt((st["t1"] or now) - st["t0"])
        extra = ""
        if state == "run" and prog:
            mm = re.search(r"(\d+)/(\d+)", prog)
            if mm and int(mm.group(2)):
                extra = f" · {mm.group(1)}/{mm.group(2)} steps"
        chips.append(f"<span class='lschip lschip-{state}'>{icon} {i}. {html.escape(st['name'] or '...')}"
                     f"<span class='lsdur'>{dur}{html.escape(extra)}</span></span>")
    body = ["<div class='lsstage'><div class='lschips'>" + "".join(chips) + "</div>"]
    running = RUN.running()
    if fail:
        what = "stopped" if fail["msg"] == "stopped" else "failed"
        body.append(f"<div class='lsmsg lsbad'>{'■' if what == 'stopped' else '❌'} Stage {fail['no']}/{len(stages)} "
                    f"<b>{html.escape(fail['name'])}</b> {what}: {html.escape(fail['msg'])}</div>")
        if fail["tail"] and what == "failed":
            body.append("<pre class='lstail'>" + html.escape("\n".join(l[-220:] for l in fail["tail"])) + "</pre>")
    elif running and 1 <= no <= len(stages):
        body.append(f"<div class='lsmsg'>Stage {no}/{len(stages)}: <b>{html.escape(stages[no - 1]['name'])}</b></div>")
    elif not running and all(s["state"] == "done" for s in stages):
        body.append(f"<div class='lsmsg lsok'>✔ All {len(stages)} stages finished.</div>")
    if nan:
        sname = stages[nan["stage"] - 1]["name"] if 1 <= nan.get("stage", 0) <= len(stages) else "training"
        body.append(f"<div class='lsmsg lsbad'>⚠️ <b>NaN loss</b> first seen at step {nan['step']}"
                    + (f" (epoch {nan['epoch']})" if nan.get("epoch") else "") + f" in stage <b>{html.escape(sname)}</b>. "
                    "The weights are broken from this step on: " + ("press <b>Stop now</b>, then retrain at half the learning rate."
                                                                    if running else "this run is counted as failed and not marked Completed.")
                    + "</div>")
    body.append("</div>")
    return "".join(body)


def _reader(run: Run):
    dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
    fd = run.proc.stdout.fileno()
    pending_cr = False
    with open(run.logfile, "ab") as lf:
        while True:
            try:
                chunk = os.read(fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            text = dec.decode(chunk)
            if run.secret:
                text = text.replace(run.secret, mask_token(run.secret))
            lf.write(text.encode("utf-8"))
            lf.flush()
            with run.lock:
                for ch in text:
                    if ch == "\r":
                        pending_cr = True
                        continue
                    if ch == "\n":
                        mp = PROG_RE.search(run.cur)
                        if mp:
                            run.progress = mp.group(1)
                            _note_loss(run, mp.group(1))
                        me = EPOCH_RE.match(run.cur)
                        if me:
                            run.epoch, run.epochs_total = int(me.group(1)), int(me.group(2))
                        _note_marker(run, run.cur)
                        run.lines.append(run.cur)
                        run.cur = ""
                        pending_cr = False
                        continue
                    if pending_cr:          # bare \r = overwrite current line (progress bars)
                        run.cur = ""
                        pending_cr = False
                    run.cur += ch
                mp = PROG_RE.search(run.cur)
                if mp:
                    run.progress = mp.group(1)
                    _note_loss(run, mp.group(1))
                if len(run.lines) > MAX_KEPT_LINES:
                    del run.lines[:1000]
                run.version += 1
    code = run.proc.wait()
    with run.lock:
        if run.cur:
            run.lines.append(run.cur)
            run.cur = ""
        run.lines.append(f"--- exited with code {code} ---")
        run.ended = time.time()
        _finish_stages(run, code)
        run.version += 1
    if run.stages or run.nan_at:                 # outside the lock: run_failure takes it itself
        why = run_failure(run, code)
        if why:
            with run.lock:
                run.lines.append(f"--- FAILED: {why} ---")
                run.version += 1
    try:                                    # also in the log file, for the report and for later reading
        with open(run.logfile, "a", encoding="utf-8") as fh:
            fh.write(f"\n--- exited with code {code} --- (ran {fmt(run.ended - run.started)})\n")
    except (OSError, TypeError):
        pass
    try:
        note = mark_completed_after_run(run, code)
    except Exception as e:
        note = f"--- could not set the Completed mark: {e} ---"
    if code == 0 and run.script:                # the LoRA Manager data of the new LoRA, filled from the training
        try:
            import training_ui
            training_ui.sidecar_after_run(run.script)
        except Exception:                   # noqa: BLE001
            pass
    if note:
        with run.lock:
            run.lines.append(note)
            run.version += 1
    try:                                    # the status strip announces the end, with the save point to try first
        _announce_done(run, code)
    except Exception:                       # noqa: BLE001
        pass
    try:
        run.tmp.unlink()
    except OSError:
        pass


def _announce_done(run, code):
    import statusbar as SB
    advice = ""
    if code == 0 and run.script:
        try:
            import savepoint as SP
            a = SP.for_script(run.script)
            advice = a["short"] if a else ""
        except Exception:                   # noqa: BLE001
            advice = ""
    if code != 0 and run.logfile:
        try:
            import failcodes
            fc = failcodes.for_log(run.logfile, code)
            advice = fc["title"] if fc else ""
        except Exception:                   # noqa: BLE001
            advice = ""
    SB.DONE_EVENTS.append({"t": time.time(), "name": run.name, "ok": code == 0, "advice": advice})
    del SB.DONE_EVENTS[:-20]


# --------------------------------------------------------------------------- scripts
def images_dir() -> Path:
    """The dataset folder: one folder per LoRA (repeat folders, model/, log/, launch scripts)."""
    return IMAGES_DIR if IMAGES_DIR is not None else ROOT / "images"


def list_scripts():
    """<root>/*.sh, <root>/images/**/*.sh, plus any --scripts-dir (recursive)."""
    found = {}

    def add(p: Path):
        if not p.name.startswith(".gui_run_") and OLD_RUNS not in p.parts:
            found[p.resolve()] = p

    if ROOT.is_dir():
        for p in ROOT.glob("*.sh"):
            add(p)
        img = images_dir()
        if img.is_dir():
            for top in img.iterdir():
                # a program folder (lora_library, a git checkout) is no dataset
                if not top.is_dir() or not is_dataset_dir(top):
                    continue
                for dirpath, dirnames, filenames in os.walk(top):
                    depth = len(Path(dirpath).relative_to(top).parts)
                    # never into the picture folders: they hold thousands of files and no launch scripts
                    dirnames[:] = [] if depth >= 2 else [
                        d for d in dirnames if not d.startswith(".") and d.lower() not in IGNORE_DIRS
                        and not REPEAT_RE.match(d) and d not in ("venv", ".venv", "node_modules", OLD_RUNS)]
                    for fn in filenames:
                        if fn.endswith(".sh"):
                            add(Path(dirpath) / fn)
    for d in EXTRA_SCRIPT_DIRS:
        if d.is_dir():
            for p in d.rglob("*.sh"):
                add(p)
    out = []
    for rp, p in sorted(found.items(), key=lambda kv: str(kv[0]).lower()):
        try:
            label = str(p.relative_to(ROOT))
        except ValueError:
            label = str(p)
        out.append((label, str(p)))
    return out


# --------------------------------------------------------------------------- resume / states
def _script_vars(body: str) -> dict:
    vars_ = {}
    for line in body.splitlines():
        m = VAR_RE.match(line)
        if m and m.group(1) not in vars_:
            v = m.group(2).strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
                v = v[1:-1]
            vars_[m.group(1)] = v
    return vars_


def _expand(val: str, vars_: dict) -> str:
    def sub(m):
        k = m.group(1) or m.group(2)
        return vars_.get(k, os.environ.get(k, m.group(0)))
    for _ in range(6):
        new = re.sub(r"\$\{(\w+)\}|\$(\w+)", sub, val)
        if new == val:
            break
        val = new
    return os.path.expanduser(val)


def script_vars(text: str, script_dir=None) -> dict:
    """Variables set in a launch script. SCRIPT_DIR and CHARACTER_NAME are computed by the script at run
    time (`$(cd ...)`), so when the script's folder is known they are filled in directly."""
    v = _script_vars(text)
    v.setdefault("HOME", str(HOME))
    if script_dir is not None:
        v["SCRIPT_DIR"] = str(script_dir)
        v["CHARACTER_NAME"] = Path(script_dir).name
    return v


def script_arg(text: str, name: str, script_dir=None):
    """Value of --<name> in a launch script, with simple $VARs expanded (None if absent or not expandable)."""
    vars_ = script_vars(text, script_dir)
    m = re.search(rf"--{name}[ =]+(?:\"([^\"]*)\"|'([^']*)'|(\S+))", text)
    if not m:
        return None
    raw = next(g for g in m.groups() if g is not None)
    val = _expand(raw, vars_)
    return None if "$" in val or "`" in val else val


def script_args(text: str, script_dir=None):
    """Best-effort (output_dir, output_name) from the script."""
    return script_arg(text, "output_dir", script_dir), script_arg(text, "output_name", script_dir)


def script_paths(path):
    """(model_dir, log_dir) of a launch script. Every base folder has a `model` folder that receives all the
    tensor files (LoRAs, saved states, and the `sample` images), and a `log` folder for TensorBoard."""
    if not path or not Path(path).is_file():
        return None, None
    base = script_home(path)
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    out = script_arg(text, "output_dir", base)
    log = script_arg(text, "logging_dir", base)
    return Path(out) if out else base / "model", Path(log) if log else base / "log"


def scan_states(root: Path, max_depth: int = 3):
    out = []
    if not root.is_dir():
        return out
    base = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root):
        d = Path(dirpath)
        if d.name.endswith("-state") or any(f in filenames for f in STATE_MARKERS):
            out.append(d)
            dirnames[:] = []
            continue
        if len(d.parts) - base >= max_depth:
            dirnames[:] = []
    out.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return out


def append_args(body: str, argstr: str):
    """Append `argstr` to the end of the first training command (the line with *train*.py and its
    backslash continuations). Returns (body, ok)."""
    lines = body.split("\n")
    for i, line in enumerate(lines):
        if line.lstrip().startswith("#") or not TRAIN_RE.search(line):
            continue
        j = i
        while lines[j].rstrip().endswith("\\") and j + 1 < len(lines):
            j += 1
        if re.search(r"\s#", lines[j]):          # trailing comment: can't append safely
            lines.insert(i + 1, f"  {argstr} \\")
        else:
            lines[j] = lines[j].rstrip() + f" {argstr}"
        return "\n".join(lines), True
    return body, False


_SAMPLE_FLAGS = ("sample_every_n_epochs", "sample_every_n_steps", "sample_at_first", "sample_sampler")


def _drop_arg(body: str, name: str) -> tuple[str, bool]:
    """Remove --name and its value (if any) from the script text."""
    pat = re.compile(rf"[ \t]*(?<![\w-])--{name}(?![\w-])(?:(?:=|[ \t]+)(?!--)(?:\"[^\"]*\"|'[^']*'|[^\s\\]+))?")
    new = pat.sub("", body)
    if new == body:
        return body, False
    new = re.sub(r"^[ \t]*\\[ \t]*\n", "", new, flags=re.M)          # a line that held only that option
    return new, True


def fix_flags(body: str) -> tuple[str, list]:
    """Options that make sd-scripts fail or are ignored, taken out: sampling without --sample_prompts (it crashes
    at the first sample) and --clip_skip in an SDXL script (SDXL does not use it). Returns (body, notes)."""
    notes = []
    if not re.search(r"(?<![\w-])--sample_prompts(?![\w-])", body):
        gone = [f for f in _SAMPLE_FLAGS if _drop_arg(body, f)[1]]
        for f in gone:
            body = _drop_arg(body, f)[0]
        if gone:
            notes.append("removed " + ", ".join("--" + f for f in gone) + " (no --sample_prompts: sd-scripts would stop at the first sample)")
    if re.search(r"sdxl_train\w*\.py", body) and re.search(r"(?<![\w-])--clip_skip(?![\w-])", body):
        body = _drop_arg(body, "clip_skip")[0]
        notes.append("removed --clip_skip (SDXL, Illustrious and Pony do not use it)")
    return body, notes


def inject_resume(body: str, state: str):
    """Append `--resume <state>` to the end of the first training command. Returns (body, ok)."""
    return append_args(body, f"--resume {shlex.quote(state)}")


# --------------------------------------------------------------------------- start / stop
def _start_run(path, workdir, venv, trace, resume, warn, info):
    """Start one script. warn/info report problems; returns True when the process was started."""
    global RUN
    if RUN.running():
        warn("A run is already in progress.")
        return False
    if not path:
        warn("Pick a script first.")
        return False
    src = Path(path)
    cwd = Path(os.path.expanduser(workdir.strip())) if (workdir or "").strip() else ROOT
    if not cwd.is_dir():
        warn(f"Working directory not found: {cwd}")
        return False
    src = Path(path)
    if not src.is_file():
        warn(f"Launch script not found: {src}")
        return False
    body = src.read_text(encoding="utf-8", errors="ignore")      # always the saved file, never a stale copy
    body, fixed = fix_flags(body)

    resume = "" if resume in (None, "", NO_STATE) else resume
    if resume:
        if not Path(resume).exists():
            warn(f"State folder no longer exists: {resume}")
            return False
        body, ok = inject_resume(body, resume)
        if not ok:
            warn("Couldn't find the training command (a line with *train*.py) to add --resume to.")
            return False

    pre = []
    if trace:
        pre.append("set -x")
    if (venv or "").strip():
        act = Path(os.path.expanduser(venv.strip()))
        if act.is_dir():
            act = act / "bin" / "activate"
        if not act.is_file():
            warn(f"activate file not found: {act}")
            return False
        pre.append(f"source {shlex.quote(str(act))}")
    if pre:
        lines = body.split("\n")
        idx = 1 if lines and lines[0].startswith("#!") else 0
        lines[idx:idx] = pre
        body = "\n".join(lines)

    tmp = src.parent / f".gui_run_{src.stem}_{int(time.time())}.sh"
    tmp.write_text(body if body.endswith("\n") else body + "\n")
    tmp.chmod(0o755)

    LOG_DIR.mkdir(exist_ok=True)
    logfile = LOG_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_{src.stem}.log"
    try:                                            # Training › Results finds the launch script of a run from this line
        gpu = gpu_name()
        logfile.write_text(f"##LS_SCRIPT {src}\n" + (f"##LS_RESUME {resume}\n" if resume else "")
                           + (f"##LS_GPU {gpu}\n" if gpu else ""), encoding="utf-8")
    except OSError:
        pass

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    venv_bin = str(Path(sys.executable).parent)   # keep the GUI's venv out of the script's PATH
    env["PATH"] = os.pathsep.join(p for p in env.get("PATH", "").split(os.pathsep) if p != venv_bin)
    env.pop("VIRTUAL_ENV", None)

    run = Run()
    run.name = src.name
    run.script = str(src)
    run.logfile = logfile
    run.tmp = tmp
    run.secret = read_hf_token()
    run.started = time.time()
    for nm in STAGE_DECL_RE.findall(body):                  # all-in-one script: show the whole pipeline from the start
        run.stages.append({"name": nm, "state": "pending", "t0": None, "t1": None})
    if resume:
        try:
            run.step_offset = int(json.loads((Path(resume) / "train_state.json").read_text()).get("current_step", 0))
        except Exception:
            run.step_offset = 0
    run.lines.append(f"$ cd {cwd} && bash {src}")
    run.lines.append(f"venv prelude: {bool((venv or '').strip())}, resume: {resume or 'no'}"
                     + (f" (saved step {run.step_offset})" if run.step_offset else ""))
    run.lines.append(f"log file: {logfile}")
    for n in fixed:
        run.lines.append(f"LoRA Studio {n}.")
    run.proc = subprocess.Popen(
        ["bash", str(tmp)],
        cwd=str(cwd),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,                    # own process group, so Stop reaches accelerate's children
    )
    RUN = run
    threading.Thread(target=_reader, args=(run,), daemon=True).start()
    try:                                            # free the video memory: Ollama is not started again afterwards
        import ollamactl
        threading.Thread(target=ollamactl.stop_all, daemon=True, name="ollama-stop").start()
    except ImportError:
        pass
    try:
        import pictools
        threading.Thread(target=pictools.stop, daemon=True, name="pictools-stop").start()
    except ImportError:
        pass
    info(f"Started {src.name}")
    return True


# ---- run settings (Settings tab, stored in settings.txt [general])


# ---- the two trainer folders (settings.txt [general]): SDGUI_SD_SCRIPTS (sd-scripts) and SDGUI_MUSUBI (musubi-tuner)
SD_KEY, MUSUBI_KEY = "SDGUI_SD_SCRIPTS", "SDGUI_MUSUBI"
_HERE_MU = _LP.ROOT / "musubi-tuner"            # install.sh puts it next to LoRA Studio
DEFAULT_MUSUBI = _HERE_MU if _HERE_MU.is_dir() else HOME / "musubi-tuner"


def _general_path(key: str):
    v = (read_settings().get("general", {}).get(key) or "").strip()
    return Path(os.path.expanduser(v)).absolute() if v else None


def sd_root() -> Path:
    """sd-scripts folder: SDGUI_SD_SCRIPTS from the settings file, else the one chosen at start-up / in Settings > Folders."""
    return _general_path(SD_KEY) or ROOT


def musubi_root() -> Path:
    """musubi-tuner folder: SDGUI_MUSUBI from the settings file, else ~/musubi-tuner."""
    return _general_path(MUSUBI_KEY) or DEFAULT_MUSUBI


def is_musubi_cfg(cfg) -> bool:
    return (cfg or {}).get("trainer") == "musubi"


def trainer_root(cfg) -> Path:
    return musubi_root() if is_musubi_cfg(cfg) else sd_root()


def is_musubi_text(text) -> bool:
    """A launch script written for musubi-tuner (the all-in-one pipeline) has a MUSUBI_DIR= line."""
    return re.search(r"^MUSUBI_DIR=", text or "", re.M) is not None


def trainer_script_path(script: str, musubi: bool) -> Path:
    return musubi_root() / "src" / "musubi_tuner" / script if musubi else sd_root() / script


def runner_settings():
    """(working directory, venv, trace) from settings.txt, falling back to the start-up defaults."""
    g = read_settings().get("general", {})
    wd = (g.get("SDGUI_WORKDIR") or "").strip() or str(ROOT)
    venv = (g.get("SDGUI_VENV") or "").strip()
    venv = "" if venv.lower() == "none" else (venv or DEFAULT_VENV)
    return wd, venv, (g.get("SDGUI_TRACE") or "").strip() == "1"


def runner_load():
    wd, venv, trace = runner_settings()
    g = read_settings().get("general", {})
    return wd, venv, trace, (g.get(SD_KEY) or "").strip(), (g.get(MUSUBI_KEY) or "").strip()


def runner_save(wd, venv, trace, sd_path="", mu_path=""):
    wd = (wd or "").strip()
    venv = (venv or "").strip()
    sd_path, mu_path = (sd_path or "").strip(), (mu_path or "").strip()
    for n, v in (("Working directory", wd), ("Venv", venv), ("sd-scripts path", sd_path), ("musubi-tuner path", mu_path)):
        if "\n" in v:
            gr.Warning(f"{n}: no line breaks please.")
            return f"{n}: no line breaks please.", st_raw()
    if wd and not Path(os.path.expanduser(wd)).is_dir():
        gr.Warning(f"Working directory does not exist: {wd}")
        return f"Working directory does not exist: `{wd}`. Nothing was saved.", st_raw()
    for n, v in (("sd-scripts path", sd_path), ("musubi-tuner path", mu_path)):
        if v and not Path(os.path.expanduser(v)).is_dir():
            gr.Warning(f"{n} does not exist: {v}")
            return f"{n} does not exist: `{v}`. Nothing was saved.", st_raw()
    data = read_settings()
    sec = data.setdefault("general", {})
    sec[SD_KEY] = sd_path                          # empty = use the sd-scripts folder from Settings > Folders
    sec[MUSUBI_KEY] = mu_path                      # empty = ~/musubi-tuner
    sec["SDGUI_WORKDIR"] = wd or str(ROOT)
    sec["SDGUI_VENV"] = venv or "none"            # "none" = deliberately no venv (an empty value would mean 'use default')
    sec["SDGUI_TRACE"] = "1" if trace else "0"
    try:
        write_settings(data)
    except OSError as e:
        return f"Could not write `{settings_path()}`: {e}", st_raw()
    gr.Info(f"Saved {Path(settings_path()).name}")
    return (f"Saved to `{settings_path()}` (`[general]`): working directory `{sec['SDGUI_WORKDIR']}`, venv "
            f"`{venv or 'none'}`, trace {'on' if trace else 'off'}; sd-scripts `{sd_root()}`, musubi-tuner `{musubi_root()}`. "
            "Used by the next Start or queued script, and by new launch scripts."), st_raw()


def dataset_of_script(path) -> str:
    """Dataset a launch script belongs to: its folder directly under the dataset folder
    (images/mychar/mycharANIMA_lora.sh -> mychar), else the folder the script is in."""
    p = Path(path)
    root = images_dir()
    if root is not None:                       # the quick way first: no symlink lookups (slow on synced folders)
        try:
            return p.relative_to(root).parts[0]
        except (ValueError, IndexError):
            pass
    try:
        return p.resolve().relative_to(Path(root).resolve()).parts[0]
    except (ValueError, IndexError, TypeError):
        return p.parent.name


_TYPE_CACHE: dict = {}


def _settings_sig():
    try:
        st = settings_path().stat()
        return st.st_mtime_ns, st.st_size
    except OSError:
        return 0, 0


def script_type(path, text=None) -> str:
    """LoRA type key of a launch script (sdxl, anima, illustrious, flux, pony, ...), '' if unknown.
    Remembered per script text and settings.txt (the subtypes live there): the lists ask for it very often."""
    try:
        if text is None:
            st = Path(path).stat()
            fkey = (str(path), st.st_mtime_ns, st.st_size, _settings_sig())
            if fkey in _TYPE_CACHE:
                return _TYPE_CACHE[fkey]
            text = Path(path).read_text(encoding="utf-8", errors="ignore")
            _TYPE_CACHE[fkey] = script_type(path, text)
            return _TYPE_CACHE[fkey]
        key = (hash(text), _settings_sig())
        if key not in _TYPE_CACHE:
            if len(_TYPE_CACHE) > 5000:
                _TYPE_CACHE.clear()
            _TYPE_CACHE[key] = infer_type(text, merged_presets()[1], "") or ""
        return _TYPE_CACHE[key]
    except Exception:
        return ""


def script_completed(path, text=None, data=None) -> bool:
    if COMPLETED is None:
        return False
    t = script_type(path, text)
    if not t:
        return False
    try:
        text = text if text is not None else Path(path).read_text(encoding="utf-8", errors="ignore")
        _, out_name = script_args(text, script_home(path))
    except Exception:                       # noqa: BLE001
        out_name = ""
    # only the version this script writes counts: a new V4 script is not hidden by V3's mark
    return COMPLETED.is_done(dataset_of_script(path), t, data, version=COMPLETED.version_of(out_name))


def mark_completed_after_run(run, code) -> str:
    """After a run: if it ended without error and wrote its final .safetensors, tick 'Completed' for this
    dataset + LoRA type. Returns a log line ('' when nothing was marked)."""
    if COMPLETED is None or code != 0 or not run.script:
        return ""
    if run.nan_at:
        return f"--- {nan_text(run)}: this dataset + type is NOT marked Completed ---"
    p = Path(run.script)
    final = final_lora(run.script)
    if final is None:
        return ""
    if not wrote_final(run):
        return "--- no new final LoRA file found, so this dataset + type is NOT marked Completed ---"
    out_name = final.stem
    t = script_type(p)
    if not t:
        return ""
    ds = dataset_of_script(p)
    COMPLETED.set_done(ds, t, True, how="auto", lora=out_name)
    label = merged_presets()[1].get(t, {}).get("label", t)
    v = COMPLETED.version_of(out_name)
    label += f" V{v}" if v else ""
    return (f"--- {final.name} finished: '{ds} · {label}' is marked Completed. Its launch scripts are hidden on the "
            f"Training › Run after Refresh (tick 'show finished' to see them; untick Completed in Collection › "
            f"Details to undo) ---")


# --------------------------------------------------------------------------- queue of scripts and states
QUEUE = {"items": [], "running": False, "stop_flag": False, "stop_on_fail": True, "killed": False}
QLOCK = threading.Lock()


def queue_failstop(v):
    QUEUE["stop_on_fail"] = bool(v)


_Q_DONE_LOGS: list = []


# ---- between queue items: the models of each script, and making room when they change
MODEL_FLAGS = ("pretrained_model_name_or_path", "dit", "vae", "ae", "text_encoder", "clip_l", "clip_g", "t5xxl",
               "qwen3", "gemma2", "byt5", "base_weights")


def script_models(path) -> frozenset:
    """The model files a launch script trains with (base model / DiT, VAE, text encoders, adapter)."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return frozenset()
    home = script_home(path)
    out = set()
    for k in MODEL_FLAGS:
        v = script_arg(text, k, home)
        if v:
            out.add((k, os.path.normpath(os.path.expanduser(v))))
    for k, v in re.findall(r'^(DIT|VAE|TEXT_ENCODER|BASE_MODEL)="?([^"\n]+)"?', text, re.M):   # musubi scripts
        out.add((k.lower(), v))
    return frozenset(out)


def vram_used():
    """(used, total) video memory in bytes of the first card, or None when it cannot be read."""
    try:
        if shutil.which("nvidia-smi"):
            r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=8)
            u, t = [float(x) for x in r.stdout.strip().splitlines()[0].split(",")]
            return u * 2 ** 20, t * 2 ** 20
        import gpusmi
        cards = gpusmi.cards()                   # amd-smi (ROCm 10+) or rocm-smi, in bytes
        if cards and cards[0]["vram_total"]:
                return float(cards[0]["vram_used"] or 0), float(cards[0]["vram_total"])
    except Exception:                       # noqa: BLE001
        pass
    return None


def sweep_run(proc) -> int:
    """Stop whatever is left of a finished run (helper processes of the trainer that did not end with it, which
    would keep video memory). Returns 1 when something had to be stopped."""
    if proc is None:
        return 0
    try:
        os.killpg(proc.pid, 0)              # anything left in the run's process group?
    except (ProcessLookupError, PermissionError, OSError):
        return 0
    _signal(proc, signal.SIGTERM)
    deadline = time.time() + 8
    while time.time() < deadline:
        try:
            os.killpg(proc.pid, 0)
        except (ProcessLookupError, PermissionError, OSError):
            return 1
        time.sleep(0.5)
    _signal(proc, signal.SIGKILL)
    return 1


def make_room(note, idle, wait=90) -> str:
    """The base model changes between two queue items: ComfyUI and Ollama let go of their models, and the queue
    waits (up to `wait` seconds) until the video memory is back near what it was before the queue started."""
    bits = []
    try:
        import comfytest as CT
        url = CT.settings()["url"]
        if CT.alive(url):
            CT._req(url, "/free", {"unload_models": True, "free_memory": True}, timeout=10)
            bits.append("ComfyUI unloaded its models")
    except Exception:                       # noqa: BLE001
        pass
    try:
        import ollamactl
        ollamactl.stop_all()
    except Exception:                       # noqa: BLE001
        pass
    v = vram_used()
    if v is None:
        return "; ".join(bits)
    used, total = v
    goal = max((idle or 0) + 768 * 2 ** 20, 0.08 * total)
    t0 = time.time()
    while used > goal and time.time() - t0 < wait:
        note(f"waiting for the video memory to empty: {used / 2 ** 30:.1f} of {total / 2 ** 30:.1f} GB in use")
        time.sleep(2)
        v = vram_used()
        if v is None:
            break
        used, total = v
    bits.append(f"video memory free: {(total - used) / 2 ** 30:.1f} of {total / 2 ** 30:.1f} GB"
                + ("" if used <= goal else " (some is still in use by another program)"))
    return "; ".join(bits)


def _queue_worker():
    prev_models, prev_label, idle = None, "", None
    v0 = vram_used()
    if v0 is not None and not RUN.running():
        idle = v0[0]                        # the video memory in use before the queue started (the desktop, ...)
    try:
        while True:
            with QLOCK:
                if QUEUE["stop_flag"]:
                    break
                it = next((i for i in QUEUE["items"] if i["state"] == "queued"), None)
                if it is None:
                    break
                it["state"], it["note"] = "running", ""
            def warn(msg, it=it):
                it["note"] = str(msg)
            try:
                import features as FT
                txt = Path(it["path"]).read_text(encoding="utf-8", errors="ignore")
                if not FT.can_train_text(txt):
                    it["state"], it["note"] = "failed", ("musubi-tuner" if "MUSUBI_DIR=" in txt else "sd-scripts") + \
                        " is not installed here (Settings › Start and network)"
                    continue
            except OSError:
                pass
            if RUN.running():
                it["state"], it["note"] = "failed", "another run is still active"
                break
            models = script_models(it["path"])
            t_ = script_type(it["path"])
            label = ((merged_presets()[1].get(t_) or {}).get("label") or t_ or Path(it["path"]).stem).split(" (")[0]
            room = ""
            if prev_models is not None and models != prev_models:   # another base model: make room for it first
                it["note"] = f"another model ({prev_label} → {label}): freeing memory …"
                room = f"model changed ({prev_label} → {label}): " + (make_room(warn, idle) or "memory freed")
            workdir, venv, trace = runner_settings()          # read fresh for every item
            if not _start_run(it["path"], workdir, venv, trace, it.get("resume", ""), warn, lambda m: None):
                it["state"] = "failed"
                it["note"] = it["note"] or "could not start"
                if QUEUE["stop_on_fail"]:
                    break
                continue
            if room:
                it["note"] = room
            run = RUN
            while run.proc.poll() is None:
                time.sleep(1)
            time.sleep(1.5)                                   # let the log reader finish
            code = run.proc.returncode
            if sweep_run(run.proc):                           # helper processes left behind would keep video memory
                print(f"[Queue] stopped processes left over from {Path(it['path']).name}", flush=True)
            prev_models, prev_label = models, label
            took = fmt((run.ended or time.time()) - run.started)
            if QUEUE.get("killed"):
                it["state"], it["note"] = "stopped", f"stopped by you after {took}"
                break
            why = run_failure(run, code)
            if not why and not wrote_final(run):
                it["state"], it["note"] = "stopped", f"ended after {took} without its final LoRA (stopped early?)"
            elif not why:
                it["state"], it["note"] = "done", f"finished in {took}" + (f" · {room}" if room else "")
                if run.logfile:
                    _Q_DONE_LOGS.append(Path(run.logfile).name)
            else:
                try:
                    import failcodes
                    fc = failcodes.for_log(run.logfile, code) if run.logfile else None
                except Exception:                     # noqa: BLE001
                    fc = None
                it["state"], it["note"] = "failed", ((fc["title"] + " · ") if fc else "") + \
                    f"{why} after {took}, log: {Path(run.logfile).name if run.logfile else '-'}"
                if QUEUE["stop_on_fail"]:
                    break
            if QUEUE["stop_flag"]:                            # Stop after current
                break
    finally:
        with QLOCK:
            QUEUE["running"] = False
            QUEUE["stop_flag"] = False
            QUEUE["killed"] = False
        done, _Q_DONE_LOGS[:] = list(_Q_DONE_LOGS), []
        if done:                                    # test pictures in ComfyUI, after the whole queue (LoRA Studio)
            try:
                import comfytest
                comfytest.after_queue(done)
            except Exception:                       # noqa: BLE001
                pass


def final_lora(script) -> Path | None:
    """The final LoRA file a launch script writes (<output_dir>/<output_name>.safetensors), None when unknown."""
    try:
        p = Path(script)
        text = p.read_text(encoding="utf-8", errors="ignore")
        out_dir, out_name = script_args(text, script_home(p))
    except (OSError, TypeError):
        return None
    if not out_name:
        return None
    return (Path(os.path.expanduser(out_dir)) if out_dir else script_home(p) / "model") / f"{out_name}.safetensors"


def wrote_final(run) -> bool:
    """Did this run write its final LoRA (newer than the start)? A run stopped with Ctrl+C can still end with exit
    code 0, so the exit code alone does not say it finished."""
    final = final_lora(run.script)
    if final is None:
        return True                                        # nothing to check against
    try:
        return final.stat().st_mtime >= run.started - 5
    except OSError:
        return False


def run_failure(run, code) -> str:
    """Why a finished run counts as failed ('' = it succeeded): a non-zero exit code, or a Python traceback in the
    last part of the log (a launch script without 'set -e' can end with exit code 0 after training crashed)."""
    nan = nan_text(run)
    if code != 0:
        why = f"exit code {code}"
        if fail_where(run):
            why += f", failed in {fail_where(run)}"
        return why + (f"; {nan}" if nan else "")
    if run.fail:
        return f"failed in {fail_where(run)}: {run.fail['msg']}"
    if nan:
        return nan
    with run.lock:
        tail = "\n".join(run.lines[-300:])
    if "Traceback (most recent call last)" in tail or "CalledProcessError" in tail:
        last = next((l.strip() for l in reversed(tail.splitlines())
                     if re.match(r"\s*\w+(\.\w+)*(Error|Exception)\b", l)), "")
        return "crashed" + (f" ({last[:120]})" if last else "")
    return ""


def queue_start(stop_on_fail):
    if QUEUE["running"]:
        gr.Warning("The queue is already running.")
        return
    if RUN.running():
        gr.Warning("A run is in progress. Wait for it to finish (or press Stop now), then start the queue.")
        return
    try:
        import speedjobs
        if speedjobs.busy():
            gr.Warning("A speed test is running (Settings › Speed test). Start the queue when it has finished, "
                       "or stop it there.")
            return
    except ImportError:
        pass
    with QLOCK:
        if not any(i["state"] == "queued" for i in QUEUE["items"]):
            gr.Warning("No waiting items in the queue.")
            return
        QUEUE["running"], QUEUE["stop_flag"], QUEUE["stop_on_fail"] = True, False, bool(stop_on_fail)
        QUEUE["killed"] = False
    threading.Thread(target=_queue_worker, daemon=True).start()
    gr.Info("Queue started.")


def queue_stop_after_current():
    if not QUEUE["running"]:
        gr.Info("The queue is not running.")
        return
    QUEUE["stop_flag"] = True
    gr.Info("The queue will stop after the current script finishes.")


def _signal(proc, sig):
    try:
        os.killpg(proc.pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def _escalate(proc):
    for delay, sig in ((20, signal.SIGTERM), (10, signal.SIGKILL)):
        deadline = time.time() + delay
        while time.time() < deadline:
            if proc.poll() is not None:
                return
            time.sleep(0.5)
        _signal(proc, sig)


def stop_run():
    """Stop now: interrupt the running script, mark its queue item stopped and end the queue."""
    if QUEUE["running"]:
        QUEUE["stop_flag"] = True
        QUEUE["killed"] = True
    if not RUN.running():
        gr.Info("Nothing is running.")
        return
    _signal(RUN.proc, signal.SIGINT)
    threading.Thread(target=_escalate, args=(RUN.proc,), daemon=True).start()
    gr.Info("Sent Ctrl+C. Escalates to SIGTERM after 20s and SIGKILL after 10s more if it hangs.")


# --------------------------------------------------------------------------- loss indicator
EPOCH_RE = re.compile(r"\s*epoch\s+(\d+)\s*/\s*(\d+)\s*$")
LOSS_RE = re.compile(r"\|\s*(\d+)/\d+\s*\[.*?avr_loss=([0-9.eE+\-]+|nan|inf)\s*\]")


def _note_loss(run: Run, prog: str):
    """Record one (step, avr_loss, epoch) sample from a tqdm line. Called with run.lock held."""
    m = LOSS_RE.search(prog)
    if not m:
        return
    step = int(m.group(1)) + run.step_offset
    try:
        val = float(m.group(2))
    except ValueError:
        return
    if not math.isfinite(val) and run.nan_at is None:
        run.nan_at = {"step": step, "epoch": run.epoch, "stage": run.stage_no}
    if run.loss and run.loss[-1][0] == step:
        return
    run.loss.append((step, val, run.epoch))
    if len(run.loss) > 40000:
        del run.loss[1:len(run.loss):2]          # thin out very long runs


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def loss_analysis(samples):
    """Plain-language reading of the avr_loss history. Returns dict with epochs, best/worst, verdict.
    avr_loss is only a smoothed training signal: it does not measure likeness. The verdicts compare the
    run with itself (its own best so far), not with a fixed number."""
    import math
    bad = [x for x in samples if not math.isfinite(x[1])]
    good = [x for x in samples if math.isfinite(x[1])]
    out = {"n": len(good), "epochs": [], "best": None, "worst": None, "level": "info", "msgs": []}
    if bad:
        out["level"] = "bad"
        out["msgs"].append("avr_loss is NaN/inf: training has diverged. Stop it, halve the learning rate and restart "
                           "(or resume from the last good state).")
    if len(good) < 8:
        if not bad:
            out["msgs"].append("Collecting data... (needs a few dozen steps)")
        return out
    by_ep = {}
    for step, v, ep in good:
        by_ep.setdefault(ep, []).append(v)
    rows = []
    for ep in sorted(by_ep):
        vs = by_ep[ep]
        rows.append({"epoch": ep, "mean": _mean(vs), "end": vs[-1], "n": len(vs)})
    out["epochs"] = rows
    # only epochs with enough samples count for best / worst (the running epoch is judged by its samples so far)
    full = [r for r in rows if r["n"] >= 5]
    if full:
        out["best"] = min(full, key=lambda r: r["mean"])
        out["worst"] = max(full, key=lambda r: r["mean"])
    vals = [v for _, v, _ in good]
    k = max(5, len(vals) // 20)
    cur = _mean(vals[-k:])
    sm = [_mean(vals[max(0, i - k + 1):i + 1]) for i in range(len(vals))]
    warm = max(k, len(vals) // 6)                 # ignore the noisy start when looking for the best point
    best_i = min(range(warm, len(sm)), key=lambda i: sm[i]) if len(sm) > warm else len(sm) - 1
    best_sm = sm[best_i]
    out["best_pt"] = (good[best_i][0], best_sm)
    out["worst_pt"] = max(((good[i][0], sm[i]) for i in range(warm, len(sm))), key=lambda t: t[1], default=None)
    msgs = out["msgs"]
    level = out["level"]
    rise = (cur - best_sm) / best_sm if best_sm > 0 else 0
    last20 = vals[-20:]
    med = sorted(last20)[len(last20) // 2]
    if len(good) > 40 and vals[-1] > 1.5 * med and med > 0:
        msgs.append(f"Spike: avr_loss jumped to {vals[-1]:.4f} (typical {med:.4f}). One spike is usually a bad batch; "
                    "repeated spikes mean the learning rate is too high.")
        level = max(level, "warn", key=["info", "ok", "warn", "bad"].index)
    if len(good) > 3 * warm and rise > 0.30:
        msgs.append(f"**Poor-likeness zone:** avr_loss {cur:.4f} is {rise:.0%} above its best ({best_sm:.4f} at step {good[best_i][0]}). "
                    "The model is moving away from the data: the learning rate may be too high, or the run is overfitting/unstable. "
                    "Compare the sample images; the checkpoint from near the best step is the safer pick.")
        level = max(level, "bad", key=["info", "ok", "warn", "bad"].index)
    elif len(good) > 3 * warm and rise > 0.12:
        msgs.append(f"Drifting up: avr_loss {cur:.4f} is {rise:.0%} above its best ({best_sm:.4f} at step {good[best_i][0]}). "
                    "Watch the next epochs; if it keeps rising, prefer the earlier checkpoint.")
        level = max(level, "warn", key=["info", "ok", "warn", "bad"].index)
    tail = vals[-max(20, len(vals) // 4):]
    if len(good) > 4 * warm and max(tail) > 0 and (max(tail) - min(tail)) / _mean(tail) < 0.04 and rise <= 0.12:
        msgs.append("Flat: avr_loss has stopped moving. The model has learned most of what it can from this data; "
                    "extra epochs mostly add overfitting risk.")
        level = max(level, "ok", key=["info", "ok", "warn", "bad"].index)
    if len(full) >= 3:
        a, b, c = full[-3]["mean"], full[-2]["mean"], full[-1]["mean"]
        if c > b > a and out["best"] is not full[-1]:
            msgs.append("The last 3 epoch averages are rising; compare the sample images from the best epoch with the latest.")
            level = max(level, "warn", key=["info", "ok", "warn", "bad"].index)
    if not msgs:
        first = vals[: max(5, len(vals) // 10)]
        drop = (_mean(first) - cur) / _mean(first) if _mean(first) else 0
        msgs.append(f"Healthy: avr_loss is {cur:.4f}, {drop:.0%} below the start and near its best." if drop > 0.02
                    else f"avr_loss is {cur:.4f}, steady. Diffusion loss falls only slowly; judge likeness from the sample images.")
        level = "ok"
    out["level"] = level
    return out


def _why_best_worst(an):
    rows = an["epochs"]
    if not an["best"] or not an["worst"] or len(rows) < 2:
        return "", ""
    b, w, f = an["best"], an["worst"], rows[0]
    last = rows[-1]["epoch"]
    why_b = f"Lowest epoch average ({b['mean']:.4f}), {100 * (f['mean'] - b['mean']) / f['mean']:.0f}% below epoch {f['epoch']}."
    if b["epoch"] == last:
        why_b += " Still improving: nothing later to compare with yet."
    else:
        why_b += " Later epochs are not lower, so extra training since then did not reduce the loss."
    if w["epoch"] == rows[0]["epoch"]:
        why_w = "Highest average, as expected: the model has not learned the data yet (early epoch)."
    elif w["epoch"] == last:
        why_w = "The newest epoch is the highest: loss is rising. Possible instability or overfitting; check the samples."
    else:
        why_w = "Highest average although later epochs are lower. Likely an unstable stretch (learning-rate spike or a hard batch)."
    return why_b, why_w


def loss_html() -> str:
    with RUN.lock:
        samples = list(RUN.loss)
    if RUN.proc is None or not samples:
        return ("<div class='lossbox'><span class='lossmuted'>Loss indicator: waiting for avr_loss values "
                "(they appear once training prints its progress bar).</span></div>")
    an = loss_analysis(samples)
    col = {"ok": "var(--ls-ok)", "info": "var(--ls-info)", "warn": "var(--ls-warn)", "bad": "var(--ls-bad)"}[an["level"]]
    icon = {"ok": "🟢", "info": "⚪", "warn": "🟠", "bad": "🔴"}[an["level"]]
    msgs = "".join(f"<div>{html.escape(m).replace('**', '')}</div>" for m in an["msgs"])
    why_b, why_w = _why_best_worst(an)

    def cell(title, r, why, c):
        if not r:
            return f"<div class='losscell'><b>{title}</b><br><span class='lossmuted'>needs at least 2 epochs</span></div>"
        return (f"<div class='losscell' style='border-left:4px solid {c}'><b>{title}</b>: epoch {r['epoch']}, "
                f"avg avr_loss <b>{r['mean']:.4f}</b><br><span class='lossmuted'>{html.escape(why)}</span></div>")
    cells = cell("Best epoch", an["best"], why_b, "var(--ls-ok)") + cell("Worst epoch", an["worst"], why_w, "var(--ls-bad)")

    # chart
    W, H, P = 600, 120, 8
    pts = [(s, v) for s, v, _ in samples if v == v and abs(v) != float("inf")]
    svg = ""
    if len(pts) >= 2:
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        x0, x1 = min(xs), max(xs); y0, y1 = min(ys), max(ys)
        if x1 == x0: x1 = x0 + 1
        if y1 == y0: y1 = y0 + 1e-6
        X = lambda x: P + (W - 2 * P) * (x - x0) / (x1 - x0)
        Y = lambda y: H - P - (H - 2 * P) * (y - y0) / (y1 - y0)
        step = max(1, len(pts) // 300)
        d = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in pts[::step])
        svg = (f"<svg viewBox='0 0 {W} {H}' width='100%' height='{H}' preserveAspectRatio='none' class='losschart'>"
               f"<polyline fill='none' stroke='var(--ls-line)' stroke-width='1.6' points='{d}'/>")
        for key, c in (("best_pt", "var(--ls-ok)"), ("worst_pt", "var(--ls-bad)")):
            pt = an.get(key)
            if pt:
                svg += (f"<line x1='{X(pt[0]):.1f}' x2='{X(pt[0]):.1f}' y1='{P}' y2='{H - P}' stroke='{c}' stroke-width='2' "
                        f"stroke-dasharray='4 3' vector-effect='non-scaling-stroke'/>")
        svg += "</svg>"
        svg += (f"<div class='lossmuted' style='display:flex;justify-content:space-between'><span>step {x0}</span>"
                f"<span>avr_loss {y0:.4f} to {y1:.4f}; <span style='color:var(--ls-ok)'>┆</span> best point, "
                f"<span style='color:var(--ls-bad)'>┆</span> worst point (after warm-up)</span><span>step {x1}</span></div>")
    ep = ""
    if an["epochs"]:
        best_e = an["best"]["epoch"] if an["best"] else None
        ep = "<div class='lossmuted' style='margin-top:6px'>" + " &nbsp;|&nbsp; ".join(
            (f"<b>ep {r['epoch']}: {r['mean']:.4f}</b>" if r["epoch"] == best_e else f"ep {r['epoch']}: {r['mean']:.4f}")
            for r in an["epochs"]) + "</div>"
    note = ("<div class='lossmuted' style='margin-top:6px'>avr_loss is a smoothed training signal, not a likeness score. "
            "It is compared with this run's own best, so the verdict is a hint. Always confirm with the sample images.</div>")
    return (f"<div class='lossbox' style='border-left:6px solid {col}'><div style='font-weight:600;margin-bottom:4px'>"
            f"{icon} Loss indicator</div>{msgs}<div class='losscells'>{cells}</div>{svg}{ep}{note}</div>")


# --------------------------------------------------------------------------- monitor


TB = {"proc": None}


def tb_stop():
    p = TB["proc"]
    if p and p.poll() is None:
        _signal(p, signal.SIGTERM)
    TB["proc"] = None


BIND_HOST = "127.0.0.1"


def display_host() -> str:
    """Address to show in links: loopback, or this computer's LAN address when listening on all interfaces."""
    if BIND_HOST not in ("0.0.0.0", "::"):
        return BIND_HOST
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as so:
            so.connect(("10.255.255.255", 1))       # no packet is sent; this just picks the outgoing interface
            return so.getsockname()[0]
    except OSError:
        return socket.gethostname()


def tb_start(logdir, port):
    tb_stop()
    logdir = os.path.expanduser(logdir or "")
    if not Path(logdir).is_dir():
        gr.Warning(f"Log dir not found: {logdir}")
        return "TensorBoard: not running"
    port = int(port)
    TB["proc"] = subprocess.Popen(
        [sys.executable, "-m", "tensorboard.main", "--logdir", logdir, "--host", BIND_HOST, "--port", str(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
    )
    return f"TensorBoard: http://{display_host()}:{port} (give it a few seconds to start)"


def tb_stop_ui():
    tb_stop()
    return "TensorBoard: not running"


atexit.register(tb_stop)


# --------------------------------------------------------------------------- dataset tab
DATATYPES_FILE = "Lora_Datatypes.py"   # ALL LoRA types (PRESETS) live in this file


def find_scan_script(explicit):
    """Lora_Datatypes.py: the given path, else next to sdgui.py, in <sd-scripts>, or in your home folder."""
    cands = []
    if explicit:
        cands.append(Path(os.path.expanduser(explicit)))
    for d in (Path(__file__).resolve().parent, ROOT, HOME):
        cands.append(d / DATATYPES_FILE)
    for c in cands:
        if c.is_file():
            return c
    return None


_SCAN_MOD: dict = {}


def load_scan_module():
    """Lora_Datatypes.py, read again only when the file changed, so edits to PRESETS apply without a restart
    (it used to be read on every call: dozens of times for one page)."""
    if SCAN_SCRIPT is None:
        raise FileNotFoundError("Lora_Datatypes.py not found. Put it next to sdgui.py or pass --datatypes.")
    st = Path(SCAN_SCRIPT).stat()
    key = (str(SCAN_SCRIPT), st.st_mtime_ns, st.st_size)
    if _SCAN_MOD.get("key") != key:
        spec = importlib.util.spec_from_file_location("Lora_Datatypes", str(SCAN_SCRIPT))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _SCAN_MOD.update(key=key, mod=mod)
    return _SCAN_MOD["mod"]


PROGRAM_MARKS = ("lora_studio.py", "sdgui.py", "lora_library.py", "start.sh", "install.sh", "setup.py", "pyproject.toml",
                 "requirements.txt", ".git")


def is_dataset_dir(p: Path) -> bool:
    """A folder in the dataset folder that is a dataset: pictures (in it or one level down), a repeat folder, launch
    scripts or LoRA Studio's own notes. Program folders (a copy of LoRA Studio, a git checkout, a venv) are not."""
    if not p.is_dir() or p.name.startswith(".") or p.name.lower() in IGNORE_DIRS:
        return False
    try:
        names = {c.name for c in p.iterdir()}
    except OSError:
        return False
    if any(m in names for m in PROGRAM_MARKS) or "pyvenv.cfg" in names:
        return False
    if ".lsflow" in names or any(n.endswith(".sh") for n in names) or any(REPEAT_RE.match(n) for n in names):
        return True
    if any(Path(n).suffix.lower() in IMAGE_EXTS for n in names):
        return True
    for c in p.iterdir():                       # pictures one level down (a dataset without a repeat folder yet)
        if c.is_dir() and not c.name.startswith(".") and c.name.lower() not in IGNORE_DIRS:
            try:
                if any(Path(n).suffix.lower() in IMAGE_EXTS for n in os.listdir(c)):
                    return True
            except OSError:
                pass
    return False


def list_lora_dirs():
    img = images_dir()
    return sorted(p.name for p in img.iterdir() if is_dataset_dir(p)) if img.is_dir() else []


# --------------------------------------------------------------------------- shared dataset helpers
def lora_path(name) -> Path:
    return images_dir() / name


def repeat_folders(lora_dir: Path):
    """[(folder, repeats)] using the same rules as the scan script (repeats=None: not a '<N>_name'
    folder, so sd-scripts ignores it). Skips model/output/sample folders."""
    m = REPEAT_RE.match(lora_dir.name)
    if m:
        return [(lora_dir, int(m.group(1)))]
    found, loose = [], []
    for dirpath, dirnames, filenames in os.walk(lora_dir):
        p = Path(dirpath)
        mm = REPEAT_RE.match(p.name)
        if p != lora_dir and mm:
            found.append((p, int(mm.group(1))))
            dirnames[:] = []
            continue
        dirnames[:] = sorted(d for d in dirnames if d.lower() not in IGNORE_DIRS and not d.startswith(".")
                             and not ARCHIVE_RENAMED_RE.match(d))
        if any(Path(f).suffix.lower() in IMAGE_EXTS for f in filenames):
            loose.append((p, None))
    return found + loose


def renamed_archives(lora_dir: Path):
    """LoraManager folders (Personal sync: previews / example images) that were renamed to <N>_LoraManager.
    sd-scripts and musubi would train on those pictures, so they are left out of the plan and reported."""
    try:
        return sorted(d for d in lora_dir.iterdir() if d.is_dir() and ARCHIVE_RENAMED_RE.match(d.name))
    except OSError:
        return []


def list_images(folder: Path):
    try:
        return sorted((f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS),
                      key=lambda f: f.name.lower())
    except OSError:
        return []


def caption_file(img: Path) -> Path:
    for ext in CAPTION_EXTS:
        c = img.with_name(img.stem + ext)
        if c.is_file():
            return c
    return img.with_name(img.stem + ".txt")


def read_caption(img: Path):
    c = caption_file(img)
    if not c.is_file():
        return None
    return c.read_text(encoding="utf-8", errors="ignore").strip()


def thumb(img: Path, size: int = 384) -> str:
    """Cached JPEG thumbnail (keeps the gallery fast with 1024px PNGs)."""
    try:
        from PIL import Image, ImageOps
    except ImportError:
        return str(img)
    try:
        st = img.stat()
        key = hashlib.sha1(f"{img}|{st.st_mtime_ns}|{st.st_size}".encode()).hexdigest()[:20]
        dst = CACHE_DIR / "thumbs" / f"{key}.jpg"
        if not dst.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(img) as im:
                im = ImageOps.exif_transpose(im)
                im.thumbnail((size, size))
                if im.mode != "RGB":
                    im = im.convert("RGB")
                im.save(dst, "JPEG", quality=85)
        return str(dst)
    except Exception:
        return str(img)


def split_tags(text: str):
    return [t.strip() for t in (text or "").replace("\n", ",").split(",") if t.strip()]


def S_NVIDIA() -> bool:
    """True when this computer trains on an NVIDIA card (features.gpu_kind): the ROCm settings are left out."""
    try:
        import features
        return features.gpu_kind() == "nvidia"
    except Exception:                           # noqa: BLE001
        return False


# --------------------------------------------------------------------------- GPU (rocm-smi on old ROCm, see gpusmi.py)
def _num_in(v):
    m = re.search(r"-?\d+(?:\.\d+)?", str(v))
    return float(m.group()) if m else None


def _pick(d, *needles, exclude=()):
    for k, v in d.items():
        kl = str(k).lower()
        if all(n in kl for n in needles) and not any(x in kl for x in exclude):
            return v
    return None


def parse_rocm_smi(data):
    """rocm-smi --json -> [{card, load, vram_total, vram_used, temp, junction, power}] (keys matched loosely,
    because the exact JSON key names differ between ROCm versions)."""
    cards = []
    for key in sorted(k for k in data if str(k).lower().startswith("card")):
        d = data[key]
        if not isinstance(d, dict):
            continue
        cards.append(dict(
            card=key,
            load=_num_in(_pick(d, "gpu use")),
            vram_total=_num_in(_pick(d, "vram total memory")),
            vram_used=_num_in(_pick(d, "vram total used")),
            temp=_num_in(_pick(d, "temperature", "edge")),
            junction=_num_in(_pick(d, "temperature", "junction")),
            power=_num_in(_pick(d, "power", "(w)")),
        ))
    return cards


_GPU_NAME: dict = {}


def gpu_name() -> str:
    """The graphics card's name (nvidia-smi or rocm-smi), for the run log and the LoRA's training card."""
    if "v" in _GPU_NAME:
        return _GPU_NAME["v"]
    name = ""
    try:
        if shutil.which("nvidia-smi"):
            r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                               text=True, timeout=8)
            name = (r.stdout.strip().splitlines() or [""])[0].strip()
        if not name:
            import gpusmi
            name = gpusmi.name()                 # amd-smi (ROCm 10+) or rocm-smi
    except Exception:                       # noqa: BLE001
        name = ""
    _GPU_NAME["v"] = name
    return name


# ---- the launch script page, preparing a dataset and the trainer settings live in sd_script.py,
# sd_prepare.py and sd_settings.py (imported back here, so sdgui.<name> works as before)
from sd_script import (  # noqa: F401  (also used as sdgui.<name> by the other parts)
    EXTRA_DIR,
    CHOICE,
    FORM_FIELDS,
    FORM_GROUPS,
    META_KEYS,
    NEW,
    OLD_RUNS,
    PRESETS_ERROR,
    RESERVED_TYPES,
    STUDIO_TABLE,
    SUBTYPE_BASES,
    _PH,
    _arg_re,
    _cs_result,
    _num,
    _plan_state,
    apply_form,
    archive_all_datasets,
    archive_old_scripts,
    builtin_presets,
    cs_apply,
    cs_apply_quiet,
    cs_check,
    cs_ds_change,
    cs_read,
    cs_retype,
    cs_save_new,
    cs_save_over,
    cs_script_change,
    cs_suggest,
    custom_subtypes,
    dataset_stats,
    ensure_folders,
    explain_for,
    export_report,
    find_script,
    form_dict,
    infer_type,
    launch_scripts,
    load_recs,
    make_plan,
    merged_presets,
    next_script_version,
    read_form,
    recommended_repeats,
    recs_status,
    script_home,
    set_choice,
    set_script_version,
    training_command,
    type_choices,
    with_virtual_row,
)
from sd_prepare import (  # noqa: F401  (also used as sdgui.<name> by the other parts)
    CC_JS,
    CC_OPEN_JS,
    OC_FAMILY,
    OC_STOP,
    _dc_cache_load,
    _dc_cache_save,
    _dc_min_side,
    _oc_check,
    _oc_family,
    cc_apply,
    cc_find,
    cc_leave_out,
    cc_open,
    cc_render,
    cc_state,
    cc_stop,
    dc_back,
    dc_run,
    dc_set_aside,
    dc_stop,
    oc_apply_recommendation,
    oc_example,
    oc_images,
    oc_preview,
    oc_recommend,
    oc_run,
    oc_save_settings,
    oc_settings,
    oc_stop,
    record_change,
    rv_undo,
    wm_apply,
    wm_apply_live,
    wm_crop_apply,
    wm_crop_show,
    wm_paint_apply,
    wm_render,
    wm_tool_open,
)
from sd_settings import (  # noqa: F401  (also used as sdgui.<name> by the other parts)
    ENV_EXTRA,
    GFX_VAR,
    KNOWN_GFX,
    SECTION_NAMES,
    _type_base,
    apply_gfx_env,
    base_name,
    cs_fill_hf,
    cs_set_epochs,
    cs_use_settings,
    detect_gfx,
    env_save,
    env_saved,
    vendor_env,
    env_update_scripts,
    fill_hf,
    folder_rows,
    get_export,
    get_gfx,
    gfx_load,
    gfx_pick,
    hf_path,
    hf_status_text,
    mask_token,
    read_hf_token,
    read_settings,
    rp_make,
    rp_name_update,
    rp_refresh,
    rp_rename,
    scan_short,
    set_arg,
    settings_models,
    settings_path,
    slot_label,
    st_apply_suggestions,
    st_load,
    st_online,
    st_dl_visible,
    st_download,
    st_download_all,
    st_download_link,
    st_download_stop,
    st_raw,
    st_refresh,
    st_refresh_all,
    st_save,
    st_suggest,
    sub_add,
    sub_choices,
    sub_list_md,
    sub_reload,
    sub_remove,
    type_slots,
    vae_optional,
    write_settings,
)


# ---- sample images on/off, logging, folder creation (all work on the script text)


# --------------------------------------------------------------------------- cleanup after a finished run
TRASH_DIR = "_cleanup_trash"
EPOCH_CKPT_RE = re.compile(r"-(?:step)?\d{3,}\.(safetensors|ckpt|pt)$", re.I)


def cleanup_plan(path, with_samples=True, keep_copied=False):
    """(base, model_dir, keep, remove, error). remove = [(Path, kind)]; nothing is touched. With keep_copied, epoch
    files that are already in a LoRA folder (same name and size, or copied under the -eN name) stay."""
    model_dir, _ = script_paths(path)
    if not model_dir:
        return None, None, [], [], "Pick a launch script first."
    base = script_home(path)
    if RUN.proc is not None and RUN.proc.poll() is None:
        return base, model_dir, [], [], "A run is still in progress. Wait for it to finish (or Stop it) before cleaning up."
    keep = []
    if model_dir.is_dir():
        keep = sorted(f for f in model_dir.iterdir()
                      if f.is_file() and f.suffix.lower() in (".safetensors", ".ckpt", ".pt") and not EPOCH_CKPT_RE.search(f.name))
    if not keep:
        return base, model_dir, [], [], (f"No finished LoRA file found in `{model_dir}` (a file without an `-000x` epoch number). "
                                         "The run probably did not finish, so nothing is removed: you may still want to resume it.")
    remove = []
    copied = set()
    if keep_copied and HK is not None:
        try:
            copied = {(f.name, f.stat().st_size) for f in HK.lora_files(lora_dirs_setting())}
        except OSError:
            copied = set()
    if model_dir.is_dir():
        for f in sorted(model_dir.iterdir()):
            if f.is_file() and EPOCH_CKPT_RE.search(f.name):
                if copied:
                    sz = f.stat().st_size
                    m = HK.EPOCH_RE.search(f.stem)
                    ep = int(re.sub(r"\D", "", m.group(0)) or 0) if m else 0
                    alt = f"{HK.EPOCH_RE.sub('', f.stem)}-e{ep}.safetensors" if m else ""
                    if (f.name, sz) in copied or (alt, sz) in copied:
                        keep.append(f)
                        continue
                remove.append((f, "epoch file you did not copy" if keep_copied else "epoch checkpoint"))
    for d in scan_states(base, 3):
        if TRASH_DIR not in d.parts:
            remove.append((d, "saved state"))
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d != TRASH_DIR]
        remove += [(Path(dirpath) / f, "cache (.npz)") for f in sorted(filenames) if f.lower().endswith(".npz")]
    sample = model_dir / "sample"
    if with_samples and sample.is_dir():
        remove.append((sample, "sample images"))
    seen, uniq = set(), []
    for item in remove:      # states may sit inside each other's search; avoid duplicates and children of removed folders
        if item[0] in seen or any(par in seen for par in item[0].parents):
            continue
        seen.add(item[0])
        uniq.append(item)
    return base, model_dir, keep, uniq, ""


def _size(p: Path) -> int:
    try:
        if p.is_file():
            return p.stat().st_size
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    except OSError:
        return 0


# --------------------------------------------------------------------------- copy a finished LoRA to the LoRA folder
def _library():
    try:
        import lora_library as LL
        return LL
    except Exception:                       # noqa: BLE001 - sdgui on its own has no library
        return None


def lora_dirs_setting() -> list[str]:
    LL = _library()
    if LL is None:
        return []
    try:
        return list((getattr(LL.LIB, "settings", None) or LL.load_settings()).get("lora_dirs") or [])
    except Exception:                       # noqa: BLE001
        return []


# --------------------------------------------------------------------------- settings tab (model files)
SETTINGS_FILE = "settings.txt"
ST_N = 4            # number of file slots in the Settings tab (set from the presets in build_ui)


# --------------------------------------------------------------------------- UI
# --------------------------------------------------------------------------- test pictures from LoRA files (Review tab)
GEN_HELP_CACHE: dict = {}


# --------------------------------------------------------------------------- Prepare > 5 Consistency
try:
    import consistency as CC
except Exception:                           # noqa: BLE001
    CC = None



def script_saved_hook(ds) -> str:
    """After a new launch script: dsflow moves other repeat folders to _extra/ and ends 'look outside the repeat
    folder' (Datasets › 2). Returns its note ('' when there is nothing to say)."""
    try:
        import dsflow
        return dsflow.script_saved(ds)
    except Exception as e:                  # noqa: BLE001
        return f"(tidying the dataset folder failed: {e})"


def auto_script(ds, tkey) -> str:
    """Write a launch script with the recommended settings for the type (the next version). Returns a note."""
    if not ds or not tkey:
        return ""
    try:
        plan = make_plan(ds, tkey)
        dest = lora_path(ds) / plan["filename"]
        dest.write_text(plan["script_text"].rstrip("\n") + "\n", encoding="utf-8")
        dest.chmod(dest.stat().st_mode | 0o755)
        ensure_folders(ds, plan["script_text"], plan, tkey)
        archive_old_scripts(lora_path(ds), keep=dest)
        tidy = script_saved_hook(ds)
        return (f" New launch script **`{dest.name}`** written with the recommended settings for "
                f"{dict((v, l) for l, v in type_choices()).get(tkey, tkey)} (check it in Training › Launch script)."
                + (f" {tidy[:1].upper()}{tidy[1:]}." if tidy else ""))
    except Exception as e:                  # noqa: BLE001
        return f" (No launch script written: {e})"


# --------------------------------------------------------------------------- LoRA Studio pages: helpers
ON_SD_DIR: list = []   # callbacks(path) when Settings > Trainers saves a new sd-scripts folder
BLOCKS: dict = {}      # page name -> gr.Column(render=False), placed by lora_studio.py and datasets_ui.py
UI: dict = {}          # components and chains lora_studio.py wires across pages


_KEYS_MEMO = {"t": 0.0, "keys": set()}


def _checked_keys() -> set:
    if time.time() - _KEYS_MEMO["t"] > 5:
        _KEYS_MEMO["keys"] = set(k.rsplit("|", 1)[0] for k in _dc_cache_load())
        _KEYS_MEMO["t"] = time.time()
    return _KEYS_MEMO["keys"]


def _checked_count(imgs) -> int:
    """How many of these pictures have a content-check result for their current file (any model)."""
    keys = _checked_keys()
    if not keys:
        return 0
    n = 0
    for p in imgs:
        try:
            st = p.stat()
        except OSError:
            continue
        if f"{p}|{st.st_size}|{int(st.st_mtime)}" in keys:
            n += 1
    return n


def _marks_counts(ds) -> tuple[int, int]:
    """(pictures with watermarks found, of those still undecided)."""
    if DC is None or not ds:
        return 0, 0
    try:
        root = lora_path(ds)
        marks = DC.reconcile(root, DC.marks_load(root))
    except Exception:                       # noqa: BLE001
        return 0, 0
    found = [e for e in marks.values() if e.get("marks")]
    return len(found), sum(1 for e in found if not e.get("choice"))


def _row_state(label, state, text, ok):
    chip = (f'<span class="pp-chip pp-ok">{state}</span>' if ok else f'<span class="pp-chip pp-todo">{state}</span>')
    return f'<div class="pp-row"><span class="pp-name">{label}</span>{chip}<span class="pp-text">{html.escape(text)}</span></div>'


def pp_status(ds, tkey):
    """The four preparation steps of a dataset with their state (Datasets > Prepare)."""
    if not ds:
        return '<div class="pp-box"><p class="pp-empty">Pick a dataset above to see its five steps.</p></div>'
    imgs = oc_images(ds)
    n = len(imgs)
    if not n:
        return ('<div class="pp-box"><p class="pp-empty">No pictures in this dataset yet. Put them in its folder '
                f'(<code>{html.escape(str(lora_path(ds)))}</code>), then pick it again.</p></div>')
    checked = _checked_count(imgs)
    found, pending = _marks_counts(ds)
    caps = sum(1 for im in imgs if read_caption(im))
    rows = folder_rows(ds)
    try:
        rec = recommended_repeats(ds, tkey, 0) if tkey else None
    except Exception:                       # noqa: BLE001
        rec = None
    if len(rows) == 1:
        f, r, _ = rows[0]
        rep_txt = f"{f.name}" + (f" · recommended {rec}" if rec else "")
        rep_ok = r is not None and (rec is None or r == rec)
    else:
        rep_txt, rep_ok = f"{len(rows)} repeat folders", True
    out = [_row_state("1 Check pictures", "done" if checked >= n else ("partly" if checked else "to do"),
                      f"{checked} of {n} checked", checked >= n),
           _row_state("2 Watermarks", ("decide" if pending else ("done" if found else "none found")) if checked else "after the check",
                      f"{found} with watermarks, {pending} undecided" if found else
                      ("no watermarks or stamps found" if checked else "found by the content check"),
                      checked > 0 and not pending),
           _row_state("3 Repeats", "done" if rep_ok else "to do", rep_txt, rep_ok),
           _row_state("4 Captions", "done" if caps >= n else "to do", f"{caps} of {n} have a caption", caps >= n)]
    if CC is not None:
        st, txt = cc_state(ds)
        out.append(_row_state("5 Consistency", st or "-", txt or "", st == "done") if caps else
                   _row_state("5 Consistency", "after captions", "checks the captions against each other", False))
    nxt = ("Next: write the launch script" if (caps >= n and rep_ok and not pending) else "")
    return ('<div class="pp-box">' + "".join(out)
            + (f'<p class="pp-next">✓ Ready. {nxt} (Training › Launch script).</p>' if nxt else "") + "</div>")


RUN_LOG_RE = re.compile(r"^(\d{8})_(\d{6})_(.+)\.log$")


def run_history(limit: int = 100):
    """Training › History: one row per run log in gui_logs/ (newest first)."""
    rows = []
    if not LOG_DIR.is_dir():
        return rows
    logs = sorted((f for f in LOG_DIR.glob("*.log") if RUN_LOG_RE.match(f.name)), reverse=True)[:limit]
    for f in logs:
        m = RUN_LOG_RE.match(f.name)
        try:
            fst = f.stat()
        except OSError:
            continue
        live = RUN.running() and RUN.logfile and Path(RUN.logfile) == f
        ck = (f.name, fst.st_size, fst.st_mtime)
        hit = _HIST_CACHE.get(f.name)
        if hit and hit[0] == ck and not live:          # a finished log is read once, not every 2 seconds
            rows.append(list(hit[1]))
            continue
        try:
            started = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
            size = f.stat().st_size
            with open(f, "rb") as fh:
                fh.seek(max(0, size - 400_000))
                tail = fh.read().decode("utf-8", "replace")
        except (OSError, ValueError):
            continue
        ex = re.findall(r"--- exited with code (-?\d+) ---(?: \(ran ([\d:]+)\))?", tail)
        loss = re.findall(r"avr_loss[=:]\s*([0-9.]+(?:e-?\d+)?|nan|inf)", tail)
        if ex:
            code, ran = ex[-1]
            result = "done" if code == "0" else f"failed (exit {code})"
            dur = ran or fmt(f.stat().st_mtime - started.timestamp())
        elif RUN.running() and RUN.logfile and Path(RUN.logfile) == f:
            result, dur = "running", fmt(time.time() - started.timestamp())
        else:
            result, dur = "cut off", fmt(max(0, f.stat().st_mtime - started.timestamp()))
        if "--- FAILED:" in tail and result == "done":
            result = "failed"
        if result == "done" and "no new final LoRA file found" in tail:
            result = "stopped (no final LoRA)"            # ended with code 0, but before the end (Ctrl+C)
        row = [started.strftime("%Y-%m-%d %H:%M"), m.group(3), dur, result, loss[-1] if loss else "", f.name]
        if not live:
            _HIST_CACHE[f.name] = (ck, tuple(row))
        rows.append(row)
    return rows


_HIST_CACHE: dict = {}


# ---- the queue survives a restart: its list is saved, and waiting items can start again (Launch options)
QUEUE_FILE = CACHE_DIR / "queue.json"
_Q_SAVED = {"key": None}


def _queue_saver():
    while True:
        time.sleep(3)
        try:
            with QLOCK:
                items = [dict(i) for i in QUEUE["items"]]
            key = json.dumps(items, sort_keys=True)
            if key != _Q_SAVED["key"]:
                CACHE_DIR.mkdir(exist_ok=True)
                QUEUE_FILE.write_text(key, encoding="utf-8")
                _Q_SAVED["key"] = key
        except Exception:                   # noqa: BLE001
            pass


def queue_restore(start: bool = False) -> str:
    """Load the saved queue (once, at start). A run that was cut off by the restart is marked stopped; with
    `start`, waiting items start again. Returns a one-line note for the log."""
    try:
        items = json.loads(QUEUE_FILE.read_text(encoding="utf-8"))
    except Exception:                       # noqa: BLE001
        items = []
    items = [i for i in items if isinstance(i, dict) and i.get("path") and Path(i["path"]).is_file()]
    for i in items:
        if i.get("state") == "running":
            i["state"], i["note"] = "stopped", "cut off by a restart"
    with QLOCK:
        if not QUEUE["items"]:
            QUEUE["items"] = items
    _Q_SAVED["key"] = json.dumps(items, sort_keys=True)
    threading.Thread(target=_queue_saver, daemon=True, name="queue-saver").start()
    waiting = sum(1 for i in items if i.get("state") == "queued")
    if start and waiting and not RUN.running() and not QUEUE["running"]:
        with QLOCK:
            QUEUE["running"], QUEUE["stop_flag"], QUEUE["killed"] = True, False, False
        threading.Thread(target=_queue_worker, daemon=True).start()
        return f"queue resumed: {waiting} waiting item(s) started again"
    return f"queue restored: {waiting} waiting" if items else ""


def save_hf_token(tok):
    tok = (tok or "").strip()
    if not tok:
        return "Paste a token first.", gr.update(), hf_status_text()
    if " " in tok or "\n" in tok:
        return "That does not look like a token (no spaces).", gr.update(), hf_status_text()
    p = hf_path()
    p.write_text("# Put your Hugging Face token (hf_...) on its own line below. Lines starting with # are ignored.\n"
                 + tok + "\n", encoding="utf-8")
    try:
        os.chmod(p, 0o600)
    except OSError:
        pass
    return f"✔ Saved in `{p}` (readable only by you).", read_hf_token(), hf_status_text()


def ollama_ctl():
    import ollamactl
    return ollamactl


def ai_models_html(url) -> str:
    """The table of installed Ollama models: use, name, size, reads pictures, state, start / stop."""
    oc = ollama_ctl()
    try:
        ms = oc.installed(url)
    except Exception as e:                      # noqa: BLE001
        return f'<p class="ls-empty">Ollama is not reachable at {html.escape(url or "")}: {html.escape(str(e))}</p>'
    if not ms:
        return '<p class="ls-empty">Ollama runs, but has no models yet. Download one below, for example qwen2.5vl:7b.</p>'
    now = {m["name"]: m["vram"] for m in (oc.running(url, fresh=True) or [])}
    _, use = oc_settings()
    used = sum(now.values()) / 1024 ** 3
    rows = []
    for m in ms:
        n = html.escape(m["name"])
        q = m["name"].replace("\\", "\\\\").replace("'", "\\'")
        vis = m.get("vision")
        radio = (f'<input type="radio" name="ai_use" aria-label="Use {n}" {"checked" if m["name"] == use else ""} '
                 f'onclick="lsSend(\'ai_box\', \'use|{q}\')">' if vis else
                 f'<input type="radio" name="ai_use" disabled aria-label="{n} cannot read pictures">')
        if m["name"] in now:
            state = f'<span class="ai-run">● running · {now[m["name"]] / 1024 ** 3:.1f} GB</span>'
            btn = f'<button class="ai-btn ai-stop" onclick="lsSend(\'ai_box\', \'stop|{q}\')">Stop</button>'
        else:
            state = '<span class="ai-off">stopped</span>'
            btn = f'<button class="ai-btn" onclick="lsSend(\'ai_box\', \'start|{q}\')">Start</button>'
        chip = ' <span class="ls-chip ok">default</span>' if m["name"] == use else ""
        rows.append(f'<tr><td>{radio}</td><td class="ai-name"><b>{n}</b>{chip}</td><td>{m.get("size_gb", 0)} GB</td>'
                    f'<td>{"yes" if vis else "no, text only"}</td><td>{state}</td><td>{btn}</td></tr>')
    return (f'<div class="ai-head"><b>Installed Ollama models · {len(ms)}</b><span>In memory now: {used:.1f} GB</span></div>'
            '<div class="ai-wrap"><table class="ai-table"><thead><tr><th>Default</th><th>Model</th><th>Size</th>'
            '<th>Reads pictures</th><th>State</th><th></th></tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>")


def ai_refresh(url, msg=""):
    url = (url or "").strip() or ollama_ctl().DEFAULT_URL
    try:
        ms = ollama_ctl().installed(url)
    except Exception:                           # noqa: BLE001
        ms = []
    names = [m["name"] for m in ms]
    vision = [m["name"] for m in ms if m.get("vision")]
    _, use = oc_settings()
    choices = vision + ([use] if use and use not in vision else [])
    return (ai_models_html(url), msg, gr.update(choices=choices, value=use or None),
            gr.update(choices=names, value=None))


def ai_default(name, url):
    """The default model picked in the dropdown at the top of AI helpers."""
    if not name:
        return ai_refresh(url)
    oc_save_settings((url or "").strip() or ollama_ctl().DEFAULT_URL, name)
    return ai_refresh(url, f"✔ **{name}** is now the default: used for captions, sorting and picture checks.")


def ai_test(url):
    url = (url or "").strip()
    if not url:
        return ("Enter the Ollama address.",) + ai_refresh(url)
    _, model = oc_settings()
    oc_save_settings(url, model)
    oc = ollama_ctl()
    ok = oc.running(url, fresh=True) is not None
    return (("✔ Ollama answers at " if ok else "❌ Ollama does not answer at ") + f"`{url}`.",) + ai_refresh(url)


def ai_action(v, url):
    """A click in the models table: use|name, start|name or stop|name."""
    act, _, name = (v or "").partition("|")
    name = name.rsplit("|", 1)[0] if name.count("|") else name
    oc = ollama_ctl()
    if not name:
        return ai_refresh(url)
    if act == "use":
        oc_save_settings((url or "").strip() or oc.DEFAULT_URL, name)
        return ai_refresh(url, f"✔ **{name}** is used for captions, sorting and picture checks.")
    if act == "start":
        return ai_refresh(url, oc.start(name, url))
    if act == "stop":
        return ai_refresh(url, oc.stop(name, url))
    return ai_refresh(url)


def ai_remove(name, ok, url):
    if not name:
        return ai_refresh(url, "Pick a model to remove.")
    if not ok:
        return ai_refresh(url, "Tick the confirmation first.")
    return ai_refresh(url, ollama_ctl().delete(name, url))


def ai_pull_tick(cur, url):
    oc = ollama_ctl()
    txt = oc.pull_text()
    if txt == (cur or ""):
        return gr.update(), gr.update()
    return txt, (ai_models_html(url) if oc.PULL["done"] else gr.update())


def trainer_facts_html() -> str:
    """Found on this computer: the trainers, the GPU, its memory and the ROCm version."""
    def ok(b):
        return '<span class="tf-ok">✓</span>' if b else '<span class="tf-no">✕</span>'
    sd, mu = sd_root(), musubi_root()
    sd_ok = sd.is_dir() and any(sd.glob("*train_network.py"))
    sd_venv = (sd / "venv" / "bin" / "activate").is_file() or bool(DEFAULT_VENV)
    mu_ok = (mu / "src" / "musubi_tuner").is_dir()
    mu_venv = any((mu / v / "bin" / "activate").is_file() for v in (".venv", "venv"))
    gpu, vram = "not found", ""
    try:
        import statusbar
        g = statusbar.machine()["gpus"]
        if g:
            gpu = g[0]["name"]
            vram = f"{g[0]['vram_total'] / 1024 ** 3:.0f} GB" if g[0].get("vram_total") else ""
    except Exception:                           # noqa: BLE001
        pass
    rocm = ""
    for f in sorted(glob.glob("/opt/rocm*/.info/version")):
        try:
            rocm = Path(f).read_text().strip()
        except OSError:
            pass
    tv = ""
    try:                                        # the ROCm / CUDA that comes with PyTorch in the training venv
        import venvcheck
        tv = (venvcheck.LAST or {}).get("torch", "")
    except Exception:                           # noqa: BLE001
        pass
    if S_NVIDIA():
        drv = ""
        try:
            drv = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        except Exception:                       # noqa: BLE001
            pass
        cu = tv.partition("+cu")[2]
        cuda = (f"CUDA {cu[:-1]}.{cu[-1]} (PyTorch)" if cu.isdigit() and len(cu) >= 2 else "in PyTorch") if tv else "?"
        gpu_facts = [("CUDA", cuda), ("NVIDIA driver", drv or "?")]
    else:
        if not rocm and "+rocm" in tv:
            rocm = tv.partition("+rocm")[2] + " (in the venv)"
        gfx = get_gfx() or os.environ.get(GFX_VAR, "")
        gpu_facts = [("ROCm", rocm or "not found"), ("GFX version", gfx or "auto")]
    facts = [("sd-scripts", f'{ok(sd_ok)} {"found" if sd_ok else "not found"} · venv {"found" if sd_venv else "missing"}'),
             ("musubi-tuner", f'{ok(mu_ok)} {"found" if mu_ok else "not installed (only for Z-Image, Qwen-Image, FLUX.2)"}'
                              + (f' · venv {"found" if mu_venv else "missing"}' if mu_ok else "")),
             ("GPU", html.escape(gpu)), ("Video memory", vram or "?")] + gpu_facts
    return ('<div class="tf-title">Found on this computer</div><div class="tf-grid">'
            + "".join(f'<div class="tf-fact"><span>{a}</span>{b}</div>' for a, b in facts) + "</div>")


def model_vis(tkey, *vals):
    """Show only the model-file fields the LoRA type uses (plus any that hold a value in the script)."""
    try:
        used = {k for k, _ in type_slots(tkey)} if tkey else set()
    except Exception:                       # noqa: BLE001
        used = set()
    names = [n for n, _ in FORM_GROUPS[3][1]]
    return [gr.update(visible=(not used) or n in used or bool((v or "").strip())) for n, v in zip(names, vals)]


_FIN_MEMO: dict = {}


def build_ui(parent):
    """The training parts of LoRA Studio, built into the open gr.Blocks `parent`: every page hidden in BLOCKS (a
    gr.Column made with render=False) for lora_studio.py / datasets_ui.py to place with .render(); the components
    other pages wire to are in UI."""
    global ST_N
    BLOCKS.clear()
    UI.clear()
    try:
        ST_N = max(4, max(len(v.get("model_args", [])) for v in merged_presets()[1].values()))
    except Exception:
        ST_N = 4
    sname = Path(settings_path()).name

    def page(key, title=""):
        BLOCKS[key] = gr.Column(render=False)
        return BLOCKS[key]
    sub = page                                  # a part of a page, placed by datasets_ui

    with contextlib.nullcontext(parent) as demo:
        hf_md = gr.Markdown(visible=False)      # the token status (shown in Settings › Model files)
        _types = type_choices()
        _purposes = [("Character / person", "character"), ("Art style", "style"), ("Object / item", "object"),
                     ("Worn item (clothes, equipment, jewelry)", "worn"),
                     ("Action / pose", "action"), ("Concept / effect", "concept")]
        _oc_url, _oc_model = oc_settings()

        # ================================================================ Training > Launch script
        with page("script", "Launch script"):
            # LoRA Studio places the parts in Datasets › 9 in its own order (sub-pages)
            with sub("s9_head"):
                with gr.Row():
                    cs_ds = gr.Dropdown(choices=list_lora_dirs(), value=None, label="Dataset", scale=3,
                                        visible=False)
                    cs_script = gr.Dropdown(choices=[("+ New launch script", NEW)], value=NEW,
                                            label="Launch script", scale=4)
                    cs_type = gr.Dropdown(choices=_types, value=_types[0][1] if _types else None,
                                          label="LoRA type (model architecture)", scale=3, visible=False)
                    cs_purpose = gr.Dropdown(choices=_purposes, value="character", label="Purpose", scale=2)
                    cs_priority = gr.Dropdown(choices=[("Balanced", "balanced"), ("Best quality", "quality"),
                                                       ("Fastest", "fast")], value="balanced", label="Priority", scale=2)
                    cs_refresh = gr.Button("Scan library", scale=1, visible=False)
                cs_status = gr.Markdown("")
            with sub("s9_opts"):
                with gr.Row():
                    cs_steps = gr.Number(label="Target total steps (0 = recommended for the type, press Enter)",
                                         value=0, precision=0, scale=2)
                    cs_userep = gr.Checkbox(label=("Plan with the recommended repeat count (the repeat folder is "
                                                   "renamed by Fill with recommended)"), value=True, scale=3)
                    cs_hfauto = gr.Checkbox(label="Fill HF_TOKEN from the token file", value=True, scale=2)
                    cs_samples = gr.State(False)   # sample images are not made (removed as an option)
                with gr.Row():
                    with gr.Column(scale=3):
                        cs_recs_md = gr.Markdown(recs_status())
                    cs_export_btn = gr.Button("Export report for Claude", scale=1, size="sm")
                    cs_export_file = gr.File(label="Report", visible=False, scale=2)
            with sub("s9_why"):
                cs_explain = gr.Markdown("")
            with sub("s9_fields"):
                cs_fields = []
                for gname, flds in FORM_GROUPS:
                    with gr.Accordion(gname + (" (only the files this LoRA type uses)" if gname == "Model files" else ""),
                                      open=False):
                        per = 2 if gname == "Model files" else 4
                        for i in range(0, len(flds), per):
                            with gr.Row():
                                for name, label in flds[i:i + per]:
                                    cs_fields.append(gr.Textbox(label=label, info=f"--{name}",
                                                                visible=name not in STUDIO_TABLE))
                model_fields = cs_fields[-len(FORM_GROUPS[3][1]):]
                with gr.Row():
                    cs_models_btn = gr.Button("Use model files from Settings", size="sm")
                    cs_hf_fill = gr.Button("Fill HF_TOKEN in the script text", size="sm")
                cs_makedirs = gr.Checkbox(
                    label="Create missing folders when saving: the repeat folder (<N>_<name>), model and log. "
                          "Images lying directly in the dataset folder are moved into a newly created repeat folder.",
                    value=True)
            with sub("s9_text"):
                cs_text = gr.Code(language="shell", label=("The script text (edit by hand; the settings above follow "
                                                           "it, and it follows them)"),
                                  lines=26, interactive=True, elem_id="cs_text")
                cs_prompts_path = gr.State("")       # no sample prompts file (sample images were removed)
                cs_prompts_text = gr.State("")
            with sub("s9_save"):
                with gr.Row(elem_classes=["ls-bar"], elem_id="s9_bar"):
                    cs_suggest_btn = gr.Button("Fill with recommended settings", size="sm", scale=0, min_width=230)
                    cs_check_btn = gr.Button("Check flags against my trainer", size="sm", scale=0, min_width=240)
                    cs_save_over_btn = gr.Button("Save", variant="primary", visible=False, size="sm", scale=0,
                                                 min_width=110)
                    cs_newname = gr.Textbox(label="Save as", scale=0, min_width=220)
                    cs_save_new_btn = gr.Button("Save as new version", variant="primary", scale=0, size="sm",
                                                min_width=200)
                    cs_queue_btn = gr.Button("Add to the queue →", size="sm", scale=0, min_width=180)
                cs_check_out = gr.Markdown("")
                cs_save_msg = gr.Markdown("")
            cs_plan = gr.State({})

        # ================================================================ Datasets > Prepare
        # The Datasets flow (datasets_ui.py) places these parts in its steps: p_wm in 6 Resize, p_caption and
        # p_cons in 7 Tags, p_repeat and the launch script page in 9 Launch script. "prepare" holds the shared
        # controls and the full picture check; LoRA Studio keeps it hidden.
        with page("prepare", "Prepare"):
            with gr.Row():
                pp_ds = gr.Dropdown(choices=list_lora_dirs(), value=None, label="Dataset", scale=3)
                pp_type = gr.Dropdown(choices=_types, value=_types[0][1] if _types else None,
                                      label="LoRA type", scale=3)
                pp_purpose = gr.Dropdown(choices=_purposes, value="character", label="Purpose", scale=2)
            pp_html = gr.HTML(pp_status(None, None), elem_id="pp_html")
            with gr.Accordion("1 · Check pictures: quality, content and balance", open=True):
                gr.Markdown("Finds pictures that are **not recommended** for training and says why. **Technical** "
                            "(no AI, seconds): too small for the LoRA type, blurry, near-duplicates, extreme crops, heavy "
                            "JPEG compression. **Content** (the Ollama model from Settings › AI helpers, a few seconds per "
                            "picture, remembered): subject visible and in focus, number of people, face hidden, text / "
                            "watermark, filters, collages and screenshots. **Balance**: framing, outfits and places that "
                            "repeat too much. The rules follow the *Purpose* above.")
                with gr.Row():
                    dc_checks = gr.CheckboxGroup(choices=[("Technical", "technical"), ("Content (Ollama)", "content"),
                                                          ("Balance (Ollama)", "balance")],
                                                 value=["technical", "content", "balance"], label="Checks", scale=3)
                    dc_btn = gr.Button("Check pictures", variant="primary", size="sm", scale=1, min_width=140)
                    dc_stop_btn = gr.Button("Stop", variant="stop", size="sm", scale=0, min_width=70)
                dc_sum = gr.Markdown("")
                dc_html = gr.HTML("", elem_id="dc_html")
                dc_pick = gr.CheckboxGroup(choices=[], value=[], label="Pictures to set aside (not recommended ones are ticked)")
                with gr.Row(elem_classes=["ls-bar"]):
                    dc_aside_btn = gr.Button("Move ticked to _set_aside", variant="stop", size="sm", scale=1)
                    dc_back_btn = gr.Button("Move all back", size="sm", scale=1)
                dc_msg = gr.Markdown("")
        with page("p_wm", "Watermarks"):
            gr.Markdown("Pictures where the sort found marks **added on top** of the picture (watermarks, site "
                        "names, logo stamps, signatures, date stamps): remove them (picture tools), crop them off, set "
                        "the picture aside, or keep it (it may be learned). Text that belongs to the scene (signs, shirts) is left alone.",
                        elem_classes=["ls-note"])
            wm_sum = gr.Markdown("")
            wm_html = gr.HTML("", elem_id="wm_html")
            wm_box = gr.Textbox(visible=False)
            with gr.Row(elem_classes=["ls-bar"], elem_id="wm_btnrow"):
                wm_apply_btn = gr.Button("Apply choices", variant="primary", size="sm", scale=1,
                                         elem_id="wm_apply_btn")
                wm_reco_btn = gr.Button("Use the recommended option for the undecided ones", size="sm", scale=2,
                                        elem_id="wm_reco_btn")
            wm_msg = gr.Markdown("")
            wm_tool = gr.Textbox(elem_id="wm_tool", show_label=False, container=False)
            with gr.Column(visible=False, elem_classes=["ls-card"], elem_id="wm_panel") as wm_panel:
                wm_ptitle = gr.Markdown()
                with gr.Column(visible=True) as wm_paint_col:
                    gr.Markdown("Paint over the mark with the brush (a little over its edges is fine). The "
                                "picture tools fill in what is under it; the rest of the picture stays as it is.",
                                elem_classes=["ls-note"])
                    wm_editor = gr.ImageEditor(type="pil", image_mode="RGBA", sources=(), transforms=(),
                                               layers=False, height=560, show_label=False,
                                               brush=gr.Brush(colors=["#ff3b30"], default_color="#ff3b30",
                                                              default_size=28, color_mode="fixed"),
                                               buttons=[], elem_id="wm_editor")
                with gr.Column(visible=False) as wm_crop_col:
                    gr.Markdown("Move the edges until the mark is outside; the shaded part is cut off.",
                                elem_classes=["ls-note"])
                    wm_cimg = gr.Image(type="pil", interactive=False, height=420, show_label=False)
                    wm_aspect = gr.Radio(["free", "3:4", "1:1", "4:3"], value="free", label="Shape")
                    with gr.Row():
                        wm_l = gr.Slider(0, 100, value=0, step=1, label="Left %")
                        wm_r = gr.Slider(0, 100, value=100, step=1, label="Right %")
                    with gr.Row():
                        wm_t = gr.Slider(0, 100, value=0, step=1, label="Top %")
                        wm_b = gr.Slider(0, 100, value=100, step=1, label="Bottom %")
                    wm_cres = gr.Markdown()
                with gr.Row(elem_classes=["ls-bar"]):
                    wm_paint_go = gr.Button("Remove what I painted", variant="primary", size="sm", scale=0,
                                            min_width=200)
                    wm_crop_go = gr.Button("Use this crop", variant="primary", size="sm", scale=0,
                                           min_width=160, visible=False)
                    wm_close = gr.Button("Close", size="sm", scale=0, min_width=100)
                wm_pmsg = gr.Markdown()
                wm_rel = gr.State("")
        with page("p_repeat", "Repeat folder"):
            rp_info = gr.Markdown("")
            with gr.Row():
                rp_folder = gr.Dropdown(choices=[], label="Image folder", scale=4)
                rp_repeats = gr.Number(label="Repeats", value=0, precision=0, scale=1)
                rp_newname = gr.Textbox(label="New folder name", scale=3)
            with gr.Row():
                rp_confirm = gr.Checkbox(label="Allow moving images that sit directly in the dataset folder into a new repeat folder",
                                         value=False, scale=3)
                rp_btn = gr.Button("Rename / create repeat folder", variant="primary", scale=2)
                rp_epochs_btn = gr.Button("Set epochs to hit the target steps", scale=2)
                rp_make_btn = gr.Button("Create missing folders now (repeat folder, model, log)", scale=2)
            rp_msg = gr.Markdown("")
        with page("p_caption", "Captions"):
            gr.Markdown(
                "The Ollama model (Settings › AI helpers) looks at every picture and writes a caption that starts "
                "with the trigger word. For a person, captions describe what changes (clothes, pose, place, light, "
                "framing) and leave out what is always that person, so the trigger word learns it. **Try on 3 "
                "pictures** writes nothing; **Write captions** replaces the `.txt` captions (Undo restores them).",
                elem_classes=["ls-note"])
            with gr.Row():
                oc_trigger = gr.Textbox(label="Trigger word (first word of every caption)", scale=1)
                oc_leave = gr.Textbox(label="Never describe (always part of the person)",
                                      value=CAP.DEFAULT_LEAVE_OUT if CAP else "", scale=3)
            with gr.Row():
                oc_style = gr.Radio(choices=[("Sentences", "sentences"), ("Tags", "tags")], value="sentences",
                                    label="Caption format", scale=1)
                oc_length = gr.Slider(1, 5, value=3, step=1, label="Length (1 very short ... 5 very long)", scale=2)
                oc_detail = gr.Slider(1, 4, value=3, step=1, label="Detail (1 basic ... 4 everything)", scale=2)
                oc_rec_btn = gr.Button("Use recommended", size="sm", scale=0, min_width=140)
            oc_rec = gr.Markdown("", elem_id="oc_rec")
            oc_ex = gr.HTML("", elem_id="oc_ex")
            with gr.Row(elem_classes=["ls-bar"]):
                oc_hints = gr.Checkbox(label="Give the old tags to the model as hints (they can be wrong)", value=False, scale=3)
                oc_unload = gr.Checkbox(label="Free the VRAM when done", value=True, scale=2)
                oc_prev_btn = gr.Button("Try on 3 pictures", size="sm", scale=1, min_width=120)
                oc_run_btn = gr.Button("Write captions", variant="primary", size="sm", scale=1, min_width=150)
                oc_stop_btn = gr.Button("Stop", variant="stop", size="sm", scale=1, min_width=70)
                oc_undo_btn = gr.Button("Undo last change", size="sm", scale=1, min_width=130)
            oc_out = gr.Markdown("")
        with page("p_cons", "Consistency"):
            gr.Markdown("The traits of the character itself that the captions mention (hair, eyes, build, skin, "
                        "horns, tattoos, a signature object): the same in most pictures is **taken out of the "
                        "captions** so the trigger word learns it; words that mean the same (athletic, fit, toned) "
                        "become one word; **odd pictures** that say something else are shown so you can look. "
                        "Clothes, places, poses, expressions and light stay in each caption.",
                        elem_classes=["ls-note"])
            with gr.Row(elem_classes=["ls-bar"]):
                cc_btn = gr.Button("Check the captions", size="sm", scale=0, min_width=170)
                cc_stop_btn = gr.Button("Stop", variant="stop", size="sm", scale=0, min_width=70)
                cc_apply_btn = gr.Button("Apply choices", variant="primary", size="sm", scale=0, min_width=150)
            cc_sum = gr.Markdown("")
            cc_box = gr.Textbox(elem_id="cc_box", show_label=False, container=False)
            with gr.Row(equal_height=False):
                with gr.Column(scale=3):
                    cc_html = gr.HTML("", elem_id="cc_html", **({"js_on_load": CC_OPEN_JS} if "js_on_load" in
                                      inspect.signature(gr.HTML.__init__).parameters else {}))
                with gr.Column(scale=2, elem_id="cc_view"):
                    cc_img = gr.Image(label="Picture", type="filepath", interactive=False, height=420)
                    cc_cap = gr.Markdown("*Click a picture that says something else to look at it with its caption.*")
            cc_msg = gr.Markdown("")
            cc_confirm = gr.Checkbox(label="Confirm", value=False, visible=False)

        # ================================================================ Settings > Trainers and GPU
        with page("trainers", "Trainers + GPU"):
            gr.Markdown("The two trainers and the GPU they run on. Saved in "
                        f"`{sname}` under `[general]`.")
            with gr.Row(equal_height=False):
                rn_sd = gr.Textbox(label="sd-scripts folder", scale=1,
                                   info="Illustrious, SDXL, Pony, Anima, FLUX.1, SD 1.5 ...")
                rn_mu = gr.Textbox(label="musubi-tuner folder", scale=1,
                                   info="Z-Image Turbo, Qwen-Image, FLUX.2 klein (with its own venv/). Empty = ~/musubi-tuner.")
            tr_facts = gr.HTML("", elem_id="tr_facts")
            with gr.Column(elem_classes=["ls-card"]):
                gr.Markdown("**Picture tools** · quality rating, look-alikes, crops around the person, watermark "
                            "removal (LaMa) and gentle upscaling (Real-ESRGAN). They run in the sd-scripts venv on "
                            "your GPU, never during a training run, and stop after 5 idle minutes.")
                pt_status = gr.HTML("")
                with gr.Row():
                    pt_install = gr.Button("Install the picture tools", size="sm", scale=0, min_width=220)
                    pt_check = gr.Button("Check again", size="sm", scale=0, min_width=130)
                pt_msg = gr.Markdown()
            with gr.Accordion("Advanced: low-level trainer and GPU values", open=False):
                with gr.Row():
                    rn_workdir = gr.Textbox(label="Working folder", scale=1,
                                            info="Scripts run from here, so relative paths like venv/bin/activate work.")
                    rn_venv = gr.Textbox(label="Venv to activate first (optional)", scale=1,
                                         info="Folder or .../bin/activate. Your script's own activate line still runs after it.")
                rn_trace = gr.Checkbox(label="Write a full trace to the log (set -x; for bug reports)")
                _nv = S_NVIDIA()
                gr.Markdown(("**GPU and CUDA environment (NVIDIA).** Used by the program, put into every new launch "
                             "script, and written into existing scripts with the button below. The AMD ROCm values "
                             "(GFX version, AOTriton, ROCm allocator) are not needed on NVIDIA and are left out.")
                            if _nv else
                            "**GPU and ROCm environment.** Change the GFX version when you change GPU. Used by the "
                            "program, put into every new launch script, and written into existing scripts with the "
                            "button below.")
                with gr.Row(visible=not _nv):
                    gx_pick = gr.Dropdown(choices=[(f"{lab}: {val}", val) for lab, val in KNOWN_GFX], label="Known GPUs",
                                          scale=3)
                    gx_val = gr.Textbox(label="GFX version override", placeholder="e.g. 12.0.1", scale=1, min_width=110)
                gx_extra = [gr.Textbox(label=lab, placeholder=f"empty = preset default ({d})")
                            for _, lab, d in ENV_EXTRA]
                with gr.Row(elem_classes=["ls-bar"]):
                    gx_detect = gr.Button("Detect GPU", size="sm", scale=1, visible=not _nv)
                    gx_save_btn = gr.Button("Save environment", size="sm", scale=1)
                with gr.Row():
                    gx_confirm = gr.Checkbox(label="Yes, rewrite the export line in all my launch scripts (a .bak is kept)", value=False, scale=3)
                    gx_upd_btn = gr.Button("Update all launch scripts", scale=1)
                gx_msg = gr.Markdown("")
            with gr.Row(elem_classes=["ls-bar"]):
                rn_save_btn = gr.Button("Save", variant="primary", size="sm", scale=0, min_width=120)
            rn_msg = gr.Markdown("")

        # ================================================================ Settings > Model files
        with page("models", "Model files"):
            gr.Markdown(f"Model, VAE and text-encoder files for each LoRA type, saved in `{sname}`. Pick the type, then "
                        "for every file type or paste a folder (press Enter) and pick a file. New launch scripts use "
                        "these files instead of the preset defaults.")
            _st_types = type_choices()
            with gr.Row(elem_classes=["ls-bar"]):
                st_type = gr.Dropdown(choices=_st_types, value=_st_types[0][1] if _st_types else None,
                                      label="LoRA type", scale=3)
                st_rec = gr.Checkbox(label="Also list files in subfolders (2 levels)", value=True, scale=2)
                st_reload = gr.Button("Reload from file", size="sm", scale=1)
                st_refresh_btn = gr.Button("Refresh file lists", size="sm", scale=1)
            st_info = gr.Markdown("")
            st_flags = gr.State([""] * ST_N)
            st_rows, st_dirs, st_files, st_dl_btns = [], [], [], []
            for i in range(ST_N):
                with gr.Row(visible=False, equal_height=True) as _row:
                    _d = gr.Textbox(label="Folder", scale=2)
                    _f = gr.Dropdown(choices=[], label="File", allow_custom_value=True, scale=3)
                    _b = gr.Button("⬇ From Hugging Face", size="sm", visible=False, scale=0, min_width=170)
                st_rows.append(_row)
                st_dirs.append(_d)
                st_files.append(_f)
                st_dl_btns.append(_b)
            with gr.Row():
                st_save_btn = gr.Button(f"Save to {sname}", variant="primary")
                st_dl_all = gr.Button("⬇ Download all missing files", visible=False)
                st_dl_stop = gr.Button("Stop download", variant="stop", size="sm", scale=0, min_width=140)
            st_dl_md = gr.Markdown("")
            with gr.Accordion("Download any file from Hugging Face (paste its link)", open=False):
                gr.Markdown("For a file that is not in the built-in list, or another version of it: open the file on "
                            "huggingface.co and copy the link from the address bar (…/blob/main/…) or from its download "
                            "button (…/resolve/main/…). Your Hugging Face token is used, so gated repos work once you "
                            "have accepted their licence there. The file goes into your models folder (or the slot's "
                            "folder) and the slot is saved. A file in several parts (…-00001-of-00003) is fetched whole.")
                with gr.Row():
                    st_dl_slot = gr.Dropdown(choices=[], label="For the slot", scale=2)
                    st_dl_link = gr.Textbox(label="Hugging Face file link", scale=5,
                                            placeholder="https://huggingface.co/<owner>/<repo>/blob/main/<file>.safetensors")
                    st_dl_go = gr.Button("⬇ Download", variant="primary", scale=1)
            st_msg = gr.Markdown("")
            with gr.Accordion("Suggestions: which files fit this LoRA type and base model", open=True):
                gr.Markdown("Every model file in the slot folders (and the `models` folder above them) is identified from "
                            "its header: model family, text encoder / VAE kind, fp16 / fp8 / ComfyUI fp8_scaled. The best "
                            "match for each slot is marked ★. **Apply suggestions** fills the empty or unsuitable slots "
                            "(the base model only with a file of the same family); nothing is saved until you press Save.")
                st_sug_md = gr.Markdown("")
                st_sug_picks = gr.State({})
                with gr.Row():
                    st_sug_apply = gr.Button("Apply suggestions", variant="secondary", scale=1)
                    st_sug_all = gr.Checkbox(label="Also replace files that already fit", value=False, scale=1)
                    st_sug_rescan = gr.Button("Check again", scale=1)
                    st_online_btn = gr.Button("Look up online (Hugging Face, Civitai)", scale=1)
                st_online_md = gr.Markdown("")
            with gr.Accordion("LoRA subtypes (for example Pony: its own model files and file tag)", open=False):
                gr.Markdown("A subtype trains exactly like its base type (same trainer, flags and recommended settings) "
                            "but has its own model files and file tag, so launch scripts are named `<folder><TAG>_lora.sh` "
                            f"and the LoRA `<folder><TAG>V1`. Saved in `{sname}` as a section such as `[pony]` with "
                            "`base = sdxl`, `label` and `tag`, plus the model files. After adding one, pick it as "
                            "**LoRA type** above and choose its files.")
                sub_md = gr.Markdown(sub_list_md())
                with gr.Row():
                    sub_name = gr.Textbox(label="Name", info="e.g. Pony", scale=2)
                    sub_base = gr.Dropdown(choices=[("SDXL", "sdxl"), ("Anima", "anima")], value="sdxl", label="Based on", scale=1)
                    sub_tag = gr.Textbox(label="File tag", info="empty = the name in capitals (PONY)", scale=2)
                    sub_add_btn = gr.Button("Add subtype", variant="primary", scale=1)
                with gr.Row():
                    sub_rm = gr.Dropdown(choices=sub_choices(), label="Subtype to remove", scale=3)
                    sub_confirm = gr.Checkbox(label=f"Yes, remove its section from {sname}", value=False, scale=3)
                    sub_rm_btn = gr.Button("Remove subtype", variant="stop", scale=1)
                sub_msg = gr.Markdown("")
            with gr.Accordion(f"{sname} as it is on disk", open=False):
                st_raw_box = gr.Textbox(value=st_raw(), lines=14, interactive=False, show_label=False)

        # ================================================================ Settings > AI helpers
        with page("ai", "AI helpers"):
            gr.Markdown("#### Ollama: reads pictures and writes captions")
            with gr.Row(equal_height=True):
                oc_url = gr.Textbox(label="Ollama address", value=_oc_url, scale=3)
                oc_find_btn = gr.Button("Test", size="sm", scale=0, min_width=90)
                ai_stopall = gr.Button("Stop Ollama", variant="stop", size="sm", scale=0, min_width=130)
            oc_model = gr.Dropdown(choices=[_oc_model] if _oc_model else [], value=_oc_model or None,
                                   label="Default model (captions, sorting and picture checks)",
                                   info="Only models that can read pictures are listed. It is started when a step "
                                        "needs it; the Start buttons below only load it ahead of time.",
                                   allow_custom_value=True, elem_id="ai_default")
            oc_find_md = gr.Markdown("")
            ai_box = gr.Textbox(elem_id="ai_box", show_label=False, container=False)
            ai_html = gr.HTML("", elem_id="ai_models")
            ai_msg = gr.Markdown("")
            with gr.Accordion("Get another model", open=False):
                with gr.Row(equal_height=True):
                    ai_pull_name = gr.Textbox(label="Model name", placeholder="e.g. minicpm-v:8b", scale=3)
                    ai_pull_btn = gr.Button("Download", size="sm", scale=0, min_width=110)
                ai_pull_md = gr.Markdown("")
                with gr.Row(equal_height=True):
                    ai_rm = gr.Dropdown(choices=[], label="Model to remove", scale=3)
                    ai_rm_ok = gr.Checkbox(label="Yes, remove it from Ollama", value=False, scale=2)
                    ai_rm_btn = gr.Button("Remove", variant="stop", size="sm", scale=0, min_width=100)
            gr.Markdown("Only one model runs at a time: **Start** on another model stops the running one first. "
                        "The **default model** (at the top, or the round button in the table) is used for captions, "
                        "sorting and picture checks; models that cannot read pictures cannot be picked. Starting a training run stops Ollama to free video memory; it is "
                        "not started again by itself. The top bar always shows which model is running.",
                        elem_classes=["ls-note"])
            gr.Markdown("#### Hugging Face token")
            gr.Markdown(f"Kept in `{hf_path()}` (readable only by you) and written into `export HF_TOKEN=\"...\"` of "
                        "launch scripts that download models. The token is masked in the run log.")
            with gr.Row():
                cs_hf_tok = gr.Textbox(label="Current token", value=read_hf_token(), type="password", interactive=False, scale=4)
                cs_hf_show = gr.Checkbox(label="Show", value=False, scale=1)
                cs_hf_reload = gr.Button("Reload token file", scale=1)
            with gr.Row():
                hf_new = gr.Textbox(label="Paste a new token", type="password", placeholder="hf_...", scale=4)
                hf_save_btn = gr.Button("Save token", variant="primary", scale=1)
            hf_msg = gr.Markdown("")

        # ================================================================ events
        def pp_after(ev):
            return ev.then(pp_status, [cs_ds, cs_type], pp_html)


        def rn_save(wd, venv, trace, sd, mu):
            msg, raw = runner_save(wd, venv, trace, sd, mu)
            sd = (sd or "").strip()
            if sd and msg.startswith("Saved to") and Path(os.path.expanduser(sd)).is_dir():
                for cb in ON_SD_DIR:
                    try:
                        cb(sd)
                    except Exception as e:          # noqa: BLE001
                        msg += f" (could not update the folder list: {e})"
            return msg, raw

        def rn_load():
            wd, venv, trace, sd, mu = runner_load()
            return wd, venv, trace, sd or str(ROOT), mu
        rn_save_btn.click(rn_save, [rn_workdir, rn_venv, rn_trace, rn_sd, rn_mu], [rn_msg, st_raw_box]) \
            .then(trainer_facts_html, None, tr_facts)
        demo.load(rn_load, None, [rn_workdir, rn_venv, rn_trace, rn_sd, rn_mu]).then(trainer_facts_html, None, tr_facts)

        def pt_html(fresh=False):
            try:
                import pictools
                st = pictools.status(fresh)
            except Exception as e:                       # noqa: BLE001
                return f'<div class="ls-note">✕ {html.escape(str(e))}</div>', gr.update()
            have = sorted(p.name for p in pictools.MODELS_DIR.glob("*") if p.suffix in (".pt", ".pth"))
            models = (f" · model files here: {html.escape(', '.join(have))}" if have else
                      " · the model files (about 2.5 GB) download the first time a tool is used")
            line = ("● ready" + models if st["ok"] else "○ " + html.escape(st["text"]))
            return (f'<div class="ls-note">{line}<br><span class="rs-mono">{html.escape(st["python"] or "")}</span></div>',
                    gr.update(visible=not st["ok"]))
        pt_check.click(lambda: pt_html(True), None, [pt_status, pt_install])
        demo.load(lambda: pt_html(False), None, [pt_status, pt_install])

        def pt_do():
            import pictools
            yield "Installing spandrel into the sd-scripts venv (PyTorch is left as it is) …"
            yield pictools.install()
        pt_install.click(pt_do, None, pt_msg).then(lambda: pt_html(True), None, [pt_status, pt_install])

        # ---- launch script + prepare (the dataset, type and purpose are the same on both pages)
        cs_outs = [cs_script, cs_type, cs_text] + cs_fields + [cs_explain, cs_prompts_path, cs_prompts_text,
                                                                cs_newname, cs_save_over_btn, cs_status, cs_plan, cs_samples]
        cs_ctl = [cs_ds, cs_script, cs_type, cs_steps, cs_userep, cs_hfauto, cs_samples]
        cs_ret = [cs_ds, cs_script, cs_type, cs_text, cs_steps, cs_userep, cs_hfauto, cs_samples]
        rp_outs = [rp_folder, rp_repeats, rp_newname, rp_info]
        rp_in = [cs_ds, cs_type, cs_steps]
        mvis = lambda ev: ev.then(model_vis, [cs_type] + model_fields, model_fields)

        def ds_both(v):
            ch = list_lora_dirs()
            v = v if v in ch else None
            return gr.update(choices=ch, value=v), gr.update(choices=ch, value=v)

        def ds_chain(ev):
            """Everything that follows a new dataset (on either page)."""
            ev = ev.then(cs_ds_change, [cs_ds, cs_type, cs_steps, cs_userep, cs_hfauto, cs_samples], cs_outs) \
                   .then(rp_refresh, rp_in, rp_outs).then(wm_render, cs_ds, [wm_sum, wm_html]) \
                   .then(cc_render, cs_ds, [cc_sum, cc_html]) \
                   .then(lambda ds: gr.update(value=ds or ""), cs_ds, oc_trigger) \
                   .then(lambda t: gr.update(value=t), cs_type, pp_type)
            return pp_after(mvis(ev))

        def type_chain(ev):
            ev = ev.then(cs_retype, cs_ret, cs_outs).then(rp_refresh, rp_in, rp_outs) \
                   .then(wm_render, cs_ds, [wm_sum, wm_html])
            return pp_after(mvis(ev))

        ds_chain(cs_ds.input(lambda v: v, cs_ds, pp_ds))
        ds_chain(pp_ds.input(lambda v: v, pp_ds, cs_ds))
        type_chain(cs_type.input(lambda v: v, cs_type, pp_type))
        type_chain(pp_type.input(lambda v: v, pp_type, cs_type))
        UI.update(ds_chain=ds_chain, ds_both=ds_both, cs_ds=cs_ds, pp_ds=pp_ds, cs_type=cs_type,
                  pp_type=pp_type, cs_purpose=cs_purpose, pp_purpose=pp_purpose, cs_retype=(cs_retype, cs_ret, cs_outs),
                  rp_refresh=(rp_refresh, rp_in, rp_outs), wm=(wm_render, [wm_sum, wm_html]),
                  cc=(cc_render, [cc_sum, cc_html]), oc=dict(trigger=oc_trigger, leave=oc_leave, style=oc_style,
                  length=oc_length, detail=oc_detail, url=oc_url, model=oc_model, out=oc_out, run=oc_run_btn),
                  cs_text=cs_text, cs_plan=cs_plan, cs_script=cs_script, cs_steps=cs_steps, cs_fields=cs_fields,
                  cs_save_new_btn=cs_save_new_btn, cs_save_over_btn=cs_save_over_btn, cs_status=cs_status,
                  cs_newname=cs_newname, cs_apply=(cs_apply, [cs_ds, cs_type, cs_script, cs_text, cs_plan] + cs_fields, cs_outs),
                  cs_models=(cs_use_settings, [cs_ds, cs_type, cs_script, cs_text, cs_plan], cs_outs),
                  cc_apply_btn=cc_apply_btn, cs_queue_btn=cs_queue_btn, cs_save_msg=cs_save_msg)
        mvis(cs_script.input(cs_script_change, cs_ctl, cs_outs))
        cs_steps.submit(cs_retype, cs_ret, cs_outs).then(rp_refresh, rp_in, rp_outs)

        def purpose_chain(ev):
            return ev.then(set_choice, [cs_purpose, cs_priority], None).then(cs_retype, cs_ret, cs_outs) \
                .then(lambda: recs_status(), None, cs_recs_md).then(rp_refresh, rp_in, rp_outs)
        purpose_chain(cs_purpose.input(lambda v: v, cs_purpose, pp_purpose))
        purpose_chain(pp_purpose.input(lambda v: v, pp_purpose, cs_purpose))
        purpose_chain(cs_priority.input(lambda: None, None, None))
        cs_export_btn.click(export_report, None, [cs_export_file, cs_status])
        cs_userep.input(cs_retype, cs_ret, cs_outs)
        cs_hfauto.input(cs_retype, cs_ret, cs_outs)
        pp_after(rp_make_btn.click(rp_make, [cs_ds, cs_type, cs_text, cs_plan], rp_msg).then(rp_refresh, rp_in, rp_outs))
        cs_suggest_btn.click(cs_suggest, cs_ctl, cs_outs).then(rp_refresh, rp_in, rp_outs)
        rp_folder.input(rp_name_update, [cs_ds, rp_folder, rp_repeats], rp_newname)
        rp_repeats.input(rp_name_update, [cs_ds, rp_folder, rp_repeats], rp_newname)
        pp_after(rp_btn.click(rp_rename, [cs_ds, rp_folder, rp_newname, rp_confirm], [rp_msg, cs_userep])
                 .then(cs_retype, cs_ret, cs_outs).then(rp_refresh, rp_in, rp_outs))
        ai_outs = [ai_html, ai_msg, oc_model, ai_rm]
        oc_find_btn.click(ai_test, oc_url, [oc_find_md] + ai_outs)
        oc_url.submit(ai_test, oc_url, [oc_find_md] + ai_outs)
        ai_box.change(ai_action, [ai_box, oc_url], ai_outs)
        oc_model.input(ai_default, [oc_model, oc_url], ai_outs)
        ai_stopall.click(lambda u: ai_refresh(u, ollama_ctl().stop_all(u)), oc_url, ai_outs)
        demo.load(lambda u: ai_refresh(u, ""), oc_url, ai_outs)
        ai_pull_btn.click(lambda n, u: ollama_ctl().pull(n, u), [ai_pull_name, oc_url], ai_pull_md)
        ai_rm_btn.click(ai_remove, [ai_rm, ai_rm_ok, oc_url], ai_outs).then(lambda: False, None, ai_rm_ok)
        import ticker
        ticker.timer(3.0, ai_pull_tick, [ai_pull_md, oc_url], [ai_pull_md, ai_html], show_progress="hidden")
        oc_prev_btn.click(oc_preview, [cs_ds, oc_url, oc_model, oc_trigger, oc_leave, oc_hints,
                                       oc_style, oc_length, oc_detail], oc_out)
        oc_run_evt = oc_run_btn.click(oc_run, [cs_ds, oc_url, oc_model, oc_trigger, oc_leave, oc_hints, oc_unload,
                                               oc_style, oc_length, oc_detail], oc_out)
        oc_ex_in = [oc_style, oc_length, oc_detail, oc_trigger, cs_type]
        for comp in (oc_style, oc_length, oc_detail):     # set by you, or by the recommendation (which redraws it)
            comp.input(oc_example, oc_ex_in, oc_ex)
        oc_trigger.change(oc_example, oc_ex_in, oc_ex)
        oc_rec_outs = [oc_rec, oc_style, oc_length, oc_detail]
        cs_type.change(oc_apply_recommendation, [cs_type, cs_purpose], oc_rec_outs).then(oc_example, oc_ex_in, oc_ex)
        cs_purpose.change(oc_apply_recommendation, [cs_type, cs_purpose], oc_rec_outs).then(oc_example, oc_ex_in, oc_ex)
        oc_rec_btn.click(oc_apply_recommendation, [cs_type, cs_purpose], oc_rec_outs).then(oc_example, oc_ex_in, oc_ex)
        demo.load(oc_apply_recommendation, [cs_type, cs_purpose], oc_rec_outs).then(oc_example, oc_ex_in, oc_ex)
        pp_after(oc_run_evt.then(cs_retype, cs_ret, cs_outs))              # the analysis re-reads the new captions
        oc_stop_btn.click(oc_stop, None, oc_out)
        pp_after(dc_btn.click(dc_run, [cs_ds, cs_type, cs_purpose, dc_checks, oc_url, oc_model], [dc_sum, dc_html, dc_pick])
                 .then(wm_render, cs_ds, [wm_sum, wm_html]))
        WM_JS = ("(ds, x) => { var o = {}; document.querySelectorAll('#wm_html .wm-row').forEach(function(r){ "
                 "var s = r.querySelector('.wm-opt.sel'); if (s) o[r.dataset.p] = s.dataset.c; }); return [ds, JSON.stringify(o)]; }")
        pp_after(wm_apply_btn.click(wm_apply_live, [cs_ds, wm_box], wm_msg, js=WM_JS).then(wm_render, cs_ds, [wm_sum, wm_html])
                 .then(cs_retype, cs_ret, cs_outs))
        pp_after(wm_reco_btn.click(lambda ds: (yield from wm_apply_live(ds, "", only_pending=True)), cs_ds, wm_msg)
                 .then(wm_render, cs_ds, [wm_sum, wm_html]).then(cs_retype, cs_ret, cs_outs))
        wm_p_outs = [wm_panel, wm_ptitle, wm_paint_col, wm_crop_col, wm_editor, wm_cimg, wm_l, wm_r, wm_t, wm_b,
                     wm_paint_go, wm_crop_go, wm_rel, wm_pmsg, wm_cres]
        wm_tool.change(wm_tool_open, [cs_ds, wm_tool], wm_p_outs, show_progress="hidden")
        wm_close.click(lambda: gr.update(visible=False), None, wm_panel)
        wm_cin = [cs_ds, wm_rel, wm_l, wm_r, wm_t, wm_b, wm_aspect]
        for c_ in (wm_l, wm_r, wm_t, wm_b):
            c_.release(wm_crop_show, wm_cin, [wm_cimg, wm_cres], show_progress="hidden")
        wm_aspect.change(wm_crop_show, wm_cin, [wm_cimg, wm_cres], show_progress="hidden")
        pp_after(wm_paint_go.click(lambda: "⏳ The picture tools fill in what you painted …", None, wm_pmsg)
                 .then(wm_paint_apply, [cs_ds, wm_rel, wm_editor], [wm_msg, wm_panel])
                 .then(wm_render, cs_ds, [wm_sum, wm_html]))
        pp_after(wm_crop_go.click(wm_crop_apply, wm_cin, [wm_msg, wm_panel])
                 .then(wm_render, cs_ds, [wm_sum, wm_html]))
        dc_stop_btn.click(dc_stop, None, dc_msg)
        pp_after(cc_btn.click(cc_find, [cs_ds, oc_url, oc_model, oc_trigger], [cc_sum, cc_html]))
        cc_stop_btn.click(cc_stop, None, cc_msg, queue=False)
        pp_after(cc_apply_btn.click(cc_apply, [cs_ds, cc_box, oc_url, oc_model, oc_trigger, cc_confirm], [cc_msg, cc_confirm],
                                    js=CC_JS)
                 .then(cc_render, cs_ds, [cc_sum, cc_html]).then(cc_leave_out, [cs_ds, oc_leave], oc_leave)
                 .then(cs_retype, cs_ret, cs_outs))
        cc_box.change(cc_open, cc_box, [cc_img, cc_cap])
        pp_after(dc_aside_btn.click(dc_set_aside, [cs_ds, dc_pick], dc_msg).then(cs_retype, cs_ret, cs_outs)
                 .then(wm_render, cs_ds, [wm_sum, wm_html]))
        pp_after(dc_back_btn.click(dc_back, cs_ds, dc_msg).then(cs_retype, cs_ret, cs_outs)
                 .then(wm_render, cs_ds, [wm_sum, wm_html]))
        pp_after(oc_undo_btn.click(rv_undo, None, oc_out).then(cs_retype, cs_ret, cs_outs))
        rp_epochs_btn.click(cs_set_epochs, [cs_ds, cs_type, cs_script, cs_text, cs_plan, cs_steps], cs_outs)
        cs_hf_show.input(lambda show: gr.update(type="text" if show else "password"), cs_hf_show, cs_hf_tok)
        cs_hf_reload.click(lambda: (read_hf_token(), hf_status_text()), None, [cs_hf_tok, hf_md])
        hf_save_btn.click(save_hf_token, hf_new, [hf_msg, cs_hf_tok, hf_md]).then(lambda: "", None, hf_new)
        cs_hf_fill.click(cs_fill_hf, [cs_text, cs_plan], cs_outs)
        # the form and the text follow each other, without buttons
        for fld, (fname, _) in zip(cs_fields, FORM_FIELDS):
            if fname not in STUDIO_TABLE:
                fld.blur(cs_apply_quiet, [cs_ds, cs_type, cs_script, cs_text, cs_plan] + cs_fields, cs_outs,
                         show_progress="hidden")
        cs_text.change(cs_read, [cs_ds, cs_type, cs_script, cs_text, cs_plan], cs_outs, show_progress="hidden")
        cs_models_btn.click(cs_use_settings, [cs_ds, cs_type, cs_script, cs_text, cs_plan], cs_outs)
        cs_check_btn.click(cs_check, cs_text, cs_check_out)
        pp_after(cs_save_over_btn.click(cs_save_over, [cs_ds, cs_script, cs_text, cs_type, cs_plan, cs_makedirs], cs_status)
                 .then(lambda s: s, cs_status, cs_save_msg).then(rp_refresh, rp_in, rp_outs))
        pp_after(cs_save_new_btn.click(lambda *a: cs_save_new(*a)[:-1], [cs_ds, cs_newname, cs_text, cs_prompts_path,
                                                                        cs_prompts_text, cs_type, cs_script, cs_plan,
                                                                        cs_makedirs], cs_outs)
                 .then(lambda s: s, cs_status, cs_save_msg)
                 .then(rp_refresh, rp_in, rp_outs))
        cs_refresh.click(scan_short, cs_ds, [cs_ds, cs_status]).then(lambda v: ds_both(v)[1], cs_ds, pp_ds)
        demo.load(model_vis, [cs_type] + model_fields, model_fields)

        # ---- model files settings
        st_slot_outs = []
        for _r, _d, _f in zip(st_rows, st_dirs, st_files):
            st_slot_outs += [_r, _d, _f]
        st_outs = st_slot_outs + [st_flags, st_info]
        sug_in = [st_type, st_flags] + st_dirs + st_files
        st_dl_vis_in = [st_type, st_flags] + st_files
        st_dl_vis_out = st_dl_btns + [st_dl_all, st_dl_slot]

        def st_files_changed(*vals):
            n = len(st_dl_vis_in)
            return list(st_dl_visible(*vals[:n])) + list(st_suggest(*vals[n:]))
        # after anything that changes the model file fields: the download buttons and the suggestions follow, in one
        # event (a .change per field cost four events, also when the page opened)
        _stv = lambda ev: ev.then(st_files_changed, st_dl_vis_in + sug_in, st_dl_vis_out + [st_sug_md, st_sug_picks],
                                  show_progress="hidden")
        _stv(st_type.input(st_load, [st_type, st_rec], st_outs))
        _stv(demo.load(st_load, [st_type, st_rec], st_outs))
        _stv(st_rec.input(st_refresh_all, [st_rec] + st_dirs + st_files, st_files))
        _stv(st_refresh_btn.click(st_refresh_all, [st_rec] + st_dirs + st_files, st_files))
        for _d, _f in zip(st_dirs, st_files):
            _stv(_d.submit(st_refresh, [_d, st_rec, _f], _f))
            _stv(_d.blur(st_refresh, [_d, st_rec, _f], _f))
            for _ev in (_f.input,):                            # a file picked or typed by you
                _ev(st_files_changed, st_dl_vis_in + sug_in, st_dl_vis_out + [st_sug_md, st_sug_picks],
                    show_progress="hidden")
        sub_outs = [sub_msg, st_type, cs_type, sub_rm, sub_md, st_raw_box]
        pp_types = lambda ev: ev.then(lambda t: gr.update(choices=type_choices(), value=t), cs_type, pp_type)
        pp_types(_stv(sub_add_btn.click(sub_add, [sub_name, sub_base, sub_tag], sub_outs)
                      .then(st_load, [st_type, st_rec], st_outs)))
        pp_types(_stv(sub_rm_btn.click(sub_remove, [sub_rm, sub_confirm, st_type, cs_type], sub_outs)
                      .then(lambda: False, None, sub_confirm).then(st_load, [st_type, st_rec], st_outs)))
        pp_types(_stv(st_reload.click(sub_reload, None, sub_outs).then(st_load, [st_type, st_rec], st_outs)))
        st_save_btn.click(st_save, [st_type, st_flags] + st_dirs + st_files, [st_msg, st_raw_box])
        st_type.input(lambda: "", None, st_online_md)
        st_sug_rescan.click(st_suggest, sug_in, [st_sug_md, st_sug_picks])
        _stv(st_sug_apply.click(st_apply_suggestions, [st_type, st_flags, st_sug_picks, st_sug_all, st_rec] + st_dirs
                                + st_files, st_dirs + st_files))
        st_online_btn.click(st_online, [st_type, st_files[0]], st_online_md)
        # Hugging Face downloads (Settings > Hugging Face token is used)
        for _i, (_d, _f, _b) in enumerate(zip(st_dirs, st_files, st_dl_btns)):
            _stv(_b.click(lambda t, fl, d, f, i=_i: (yield from st_download(t, fl, i, d, f)), [st_type, st_flags, _d, _f],
                          [st_dl_md, _d, _f], concurrency_id="hfdl", concurrency_limit=1))
        _stv(st_dl_all.click(st_download_all, [st_type, st_flags, st_rec] + st_dirs + st_files,
                             [st_dl_md] + st_dirs + st_files, concurrency_id="hfdl", concurrency_limit=1))
        _stv(st_dl_go.click(st_download_link, [st_type, st_flags, st_dl_slot, st_dl_link, st_rec] + st_dirs + st_files,
                            [st_dl_md] + st_dirs + st_files, concurrency_id="hfdl", concurrency_limit=1))
        st_dl_stop.click(st_download_stop, None, None, queue=False)
        st_type.input(lambda: "", None, st_dl_md)
        gx_pick.input(gfx_pick, gx_pick, gx_val)
        gx_detect.click(detect_gfx, None, [gx_val, gx_msg])
        gx_save_btn.click(env_save, [gx_val] + gx_extra, [gx_msg, st_raw_box])
        gx_upd_btn.click(env_update_scripts, gx_confirm, gx_msg).then(lambda: False, None, gx_confirm)
        demo.load(gfx_load, None, [gx_val, gx_pick, gx_msg] + gx_extra)
    return demo


def detect_venv() -> str:
    for c in (ROOT / "venv", HOME / "sd-script" / "venv"):
        if (c / "bin" / "activate").is_file():
            return str(c)
    return ""
