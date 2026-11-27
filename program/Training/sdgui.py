#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""
sdgui.py - the training engine of LoRA Studio, and the shared parts of the launch script and dataset pages.

  runs           Run: one training run (bash on a temporary copy of the launch script, the log streamed and kept in
                 userdata/gui_logs/), its stages (##LS_STAGE markers of the musubi pipelines), loss and NaN watch
  the queue      in sdqueue.py; the loss of a run in sdloss.py; the dataset folders and files in dsfiles.py
  launch scripts reading and writing their options (the form), the LoRA types (Lora_Datatypes.py) and the
                 recommendations; in sd_script.py, sd_prepare.py and sd_settings.py (use them directly)

Security: this runs shell text from a web page and edits files. LoRA Studio binds to 127.0.0.1 unless you choose
otherwise, with a login then (Settings › Start and network).
"""
__version__ = "1.271126b"  # LoRA Studio version, see version.py
import atexit
import notify
import codecs
import glob
import html
import json
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
CSS = ('.ls-bar {gap: 8px !important; align-items: center !important; flex-wrap: wrap;} '
       '.ls-bar > * {flex-grow: 0;} '
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
                text = text.replace(run.secret, sd_settings.mask_token(run.secret))
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
                            sdloss._note_loss(run, mp.group(1))
                        me = sdloss.EPOCH_RE.match(run.cur)
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
                    sdloss._note_loss(run, mp.group(1))
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
        why = sdqueue.run_failure(run, code)
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
    SB.DONE_EVENTS.append({"t": time.time(), "name": run.name, "ok": code == 0, "advice": advice,
                           "log": Path(run.logfile).name if run.logfile else ""})
    del SB.DONE_EVENTS[:-20]


# --------------------------------------------------------------------------- scripts
def images_dir() -> Path:
    """The dataset folder: one folder per LoRA (repeat folders, model/, log/, launch scripts)."""
    return IMAGES_DIR if IMAGES_DIR is not None else ROOT / "images"


def list_scripts():
    """<root>/*.sh, <root>/images/**/*.sh, plus any --scripts-dir (recursive)."""
    found = {}

    def add(p: Path):
        if not p.name.startswith(".gui_run_") and sd_script.OLD_RUNS not in p.parts and "_compare" not in p.parts:
            found[p.resolve()] = p

    if ROOT.is_dir():
        for p in ROOT.glob("*.sh"):
            add(p)
        img = images_dir()
        if img.is_dir():
            for top in img.iterdir():
                # a program folder (lora_library, a git checkout) is no dataset
                if not top.is_dir() or not dsfiles.is_dataset_dir(top):
                    continue
                for dirpath, dirnames, filenames in os.walk(top):
                    depth = len(Path(dirpath).relative_to(top).parts)
                    # never into the picture folders: they hold thousands of files and no launch scripts
                    dirnames[:] = [] if depth >= 2 else [
                        d for d in dirnames if not d.startswith(".") and d.lower() not in IGNORE_DIRS
                        and not REPEAT_RE.match(d) and d not in ("venv", ".venv", "node_modules", sd_script.OLD_RUNS, "_compare")]
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
    base = sd_script.script_home(path)
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


def drop_arg(body: str, name: str) -> tuple[str, bool]:
    """Remove --name and its value (if any) from the script text."""
    pat = re.compile(rf"[ \t]*(?<![\w-])--{name}(?![\w-])(?:(?:=|[ \t]+)(?!--)(?:\"[^\"]*\"|'[^']*'|[^\s\\]+))?")
    new = pat.sub("", body)
    if new == body:
        return body, False
    new = re.sub(r"^[ \t]*\\[ \t]*\n", "", new, flags=re.M)          # a line that held only that option
    return new, True


def set_switch(body: str, name: str, on: bool) -> tuple[str, bool]:
    """An option without a value (--network_train_unet_only): added to the training command, or taken out.
    Returns (body, changed)."""
    has = re.search(rf"(?<![\w-])--{name}(?![\w-])", body) is not None
    if on == has:
        return body, False
    return append_args(body, f"--{name}") if on else drop_arg(body, name)


def fix_flags(body: str) -> tuple[str, list]:
    """Options that make sd-scripts fail, taken out: sampling without --sample_prompts (it crashes at the first
    sample). --clip_skip stays (owner's tested Illustrious script carries --clip_skip 2). A Hugging Face token in the
    script is replaced by the lines that read it when the script runs (scrub_hf). Returns (body, notes)."""
    notes = []
    if not re.search(r"(?<![\w-])--sample_prompts(?![\w-])", body):
        gone = [f for f in _SAMPLE_FLAGS if drop_arg(body, f)[1]]
        for f in gone:
            body = drop_arg(body, f)[0]
        if gone:
            notes.append("removed " + ", ".join("--" + f for f in gone) + " (no --sample_prompts: sd-scripts would stop at the first sample)")
    body, had = sd_settings.scrub_hf(body)                         # the token is passed when the script runs, never stored in it
    if had:
        notes.append("took the Hugging Face token out of the script (it is passed when the script runs)")
    return body, notes


def inject_resume(body: str, state: str):
    """Append `--resume <state>` to the end of the first training command. Returns (body, ok)."""
    return append_args(body, f"--resume {shlex.quote(state)}")


# --------------------------------------------------------------------------- start / stop
# The VAE a type needs, checked before a run starts (a wrong one fails only after loading everything else):
# script → (the kind model_guide.identify must report, what to use)
VAE_NEEDS = {"anima_train_network.py": ("qwen_vae", "Anima", "qwen_image_vae.safetensors")}


def model_problem(body: str) -> str:
    """'' or why the script's VAE does not fit its type (read from the file's header, cached)."""
    for script, (kind, label, need) in VAE_NEEDS.items():
        if script not in body:
            continue
        m = re.search(r'--vae[ =]+(?:"([^"]+)"|\'([^\']+)\'|(\S+))', body)
        if not m:
            return ""
        path = os.path.expanduser(next(g for g in m.groups() if g))
        if "$" in path or not Path(path).is_file():
            return ""                                    # a variable or a missing file: the trainer says so
        try:
            import model_guide
            info = model_guide.identify(path)
        except Exception:                                # noqa: BLE001
            return ""
        if info.get("kind") != kind:
            why = info.get("note") or f"it is a {info.get('kind') or 'different'} file"
            return (f"❌ Not started: the VAE `{Path(path).name}` does not fit {label}: {why}. {label} needs "
                    f"`{need}`: pick it in Settings › Model files (or in step 9's More settings) and save the script.")
    return ""


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
    bad = model_problem(body)
    if bad:
        warn(bad)
        return False

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
    try:                                            # Results finds the launch script of a run from this line
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
    tok = sd_settings.read_hf_token()
    if tok:                                         # scripts read HF_TOKEN from here; it is never written into them
        env["HF_TOKEN"] = tok

    run = Run()
    run.name = src.name
    run.script = str(src)
    run.logfile = logfile
    run.tmp = tmp
    run.secret = sd_settings.read_hf_token()
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



# ---- the two trainer folders (settings.txt [general]): SDGUI_SD_SCRIPTS (sd-scripts) and SDGUI_MUSUBI (musubi-tuner)
SD_KEY, MUSUBI_KEY = "SDGUI_SD_SCRIPTS", "SDGUI_MUSUBI"
_HERE_MU = _LP.ROOT / "musubi-tuner"            # install.sh puts it next to LoRA Studio
DEFAULT_MUSUBI = _HERE_MU if _HERE_MU.is_dir() else HOME / "musubi-tuner"


def _general_path(key: str):
    v = (sd_settings.read_settings().get("general", {}).get(key) or "").strip()
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
    g = sd_settings.read_settings().get("general", {})
    wd = (g.get("SDGUI_WORKDIR") or "").strip() or str(ROOT)
    venv = (g.get("SDGUI_VENV") or "").strip()
    venv = "" if venv.lower() == "none" else (venv or DEFAULT_VENV)
    return wd, venv, (g.get("SDGUI_TRACE") or "").strip() == "1"


def runner_load():
    wd, venv, trace = runner_settings()
    g = sd_settings.read_settings().get("general", {})
    return wd, venv, trace, (g.get(SD_KEY) or "").strip(), (g.get(MUSUBI_KEY) or "").strip()


def runner_save(wd, venv, trace, sd_path="", mu_path=""):
    wd = (wd or "").strip()
    venv = (venv or "").strip()
    sd_path, mu_path = (sd_path or "").strip(), (mu_path or "").strip()
    for n, v in (("Working directory", wd), ("Venv", venv), ("sd-scripts path", sd_path), ("musubi-tuner path", mu_path)):
        if "\n" in v:
            return f"{n}: no line breaks please.", sd_settings.st_raw()
    if wd and not Path(os.path.expanduser(wd)).is_dir():
        return f"Working directory does not exist: `{wd}`. Nothing was saved.", sd_settings.st_raw()
    for n, v in (("sd-scripts path", sd_path), ("musubi-tuner path", mu_path)):
        if v and not Path(os.path.expanduser(v)).is_dir():
            return f"{n} does not exist: `{v}`. Nothing was saved.", sd_settings.st_raw()
    data = sd_settings.read_settings()
    sec = data.setdefault("general", {})
    sec[SD_KEY] = sd_path                          # empty = use the sd-scripts folder from Settings > Folders
    sec[MUSUBI_KEY] = mu_path                      # empty = ~/musubi-tuner
    sec["SDGUI_WORKDIR"] = wd or str(ROOT)
    sec["SDGUI_VENV"] = venv or "none"            # "none" = deliberately no venv (an empty value would mean 'use default')
    sec["SDGUI_TRACE"] = "1" if trace else "0"
    try:
        sd_settings.write_settings(data)
    except OSError as e:
        return f"Could not write `{sd_settings.settings_path()}`: {e}", sd_settings.st_raw()
    notify.info(f"Saved {Path(sd_settings.settings_path()).name}")
    return (f"Saved to `{sd_settings.settings_path()}` (`[general]`): working directory `{sec['SDGUI_WORKDIR']}`, venv "
            f"`{venv or 'none'}`, trace {'on' if trace else 'off'}; sd-scripts `{sd_root()}`, musubi-tuner `{musubi_root()}`. "
            "Used by the next Start or queued script, and by new launch scripts."), sd_settings.st_raw()


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
        st = sd_settings.settings_path().stat()
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
            _TYPE_CACHE[key] = sd_script.infer_type(text, sd_script.merged_presets()[1], "") or ""
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
        _, out_name = script_args(text, sd_script.script_home(path))
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
    final = sdqueue.final_lora(run.script)
    if final is None:
        return ""
    if not sdqueue.wrote_final(run):
        return "--- no new final LoRA file found, so this dataset + type is NOT marked Completed ---"
    out_name = final.stem
    t = script_type(p)
    if not t:
        return ""
    ds = dataset_of_script(p)
    COMPLETED.set_done(ds, t, True, how="auto", lora=out_name)
    label = sd_script.merged_presets()[1].get(t, {}).get("label", t)
    v = COMPLETED.version_of(out_name)
    label += f" V{v}" if v else ""
    return (f"--- {final.name} finished: '{ds} · {label}' is marked Completed. Its launch scripts are hidden on the "
            f"the Queue after Refresh (tick 'show finished' to see them; untick Completed in Collection › "
            f"Details to undo) ---")


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
    if sdqueue.QUEUE["running"]:
        sdqueue.QUEUE["stop_flag"] = True
        sdqueue.QUEUE["killed"] = True
    if not RUN.running():
        notify.info("Nothing is running.")
        return
    _signal(RUN.proc, signal.SIGINT)
    threading.Thread(target=_escalate, args=(RUN.proc,), daemon=True).start()
    notify.info("Sent Ctrl+C. Escalates to SIGTERM after 20s and SIGKILL after 10s more if it hangs.")


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
        return f"TensorBoard: not running (log folder not found: {logdir})"
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


# ---- the launch script page, preparing a dataset and the trainer settings live in sd_script.py, sd_prepare.py and
# sd_settings.py (callers use those modules directly)
import sd_script  # noqa: E402  (they import this file: after its names)
import sd_prepare  # noqa: E402
import sd_settings  # noqa: E402
import sdqueue
import sdloss
import dsfiles


# --------------------------------------------------------------------------- cleanup after a finished run
TRASH_DIR = "_cleanup_trash"
EPOCH_CKPT_RE = re.compile(r"-(?:step)?\d{3,}\.(safetensors|ckpt|pt)$", re.I)


def cleanup_plan(path, with_samples=True, keep_copied=False):
    """(base, model_dir, keep, remove, error). remove = [(Path, kind)]; nothing is touched. With keep_copied, epoch
    files that are already in a LoRA folder (same name and size, or copied under the -eN name) stay."""
    model_dir, _ = script_paths(path)
    if not model_dir:
        return None, None, [], [], "Pick a launch script first."
    base = sd_script.script_home(path)
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


def script_saved_hook(ds) -> str:
    """After a new launch script: dsflow moves other repeat folders to _extra/ and ends 'look outside the repeat
    folder' (Datasets › 3). Returns its note ('' when there is nothing to say)."""
    try:
        import dsflow
        return dsflow.script_saved(ds)
    except Exception as e:                  # noqa: BLE001
        return f"(tidying the dataset folder failed: {e})"


# --------------------------------------------------------------------------- LoRA Studio pages: helpers
ON_SD_DIR: list = []   # callbacks(path) when Settings > Trainers saves a new sd-scripts folder


def _marks_counts(ds) -> tuple[int, int]:
    """(pictures with watermarks found, of those still undecided)."""
    if DC is None or not ds:
        return 0, 0
    try:
        root = dsfiles.lora_path(ds)
        marks = DC.reconcile(root, DC.marks_load(root))
    except Exception:                       # noqa: BLE001
        return 0, 0
    found = [e for e in marks.values() if e.get("marks") or e.get("border")]
    return len(found), sum(1 for e in found if not e.get("choice"))


RUN_LOG_RE = re.compile(r"^(\d{8})_(\d{6})_(.+)\.log$")


def run_history(limit: int = 100):
    """The run history: one row per run log in gui_logs/ (newest first)."""
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


def save_hf_token(tok):
    tok = (tok or "").strip()
    if not tok:
        return "Paste a token first.", None, sd_settings.hf_status_text()
    if " " in tok or "\n" in tok:
        return "That does not look like a token (no spaces).", None, sd_settings.hf_status_text()
    p = sd_settings.hf_path()
    p.write_text("# Put your Hugging Face token (hf_...) on its own line below. Lines starting with # are ignored.\n"
                 + tok + "\n", encoding="utf-8")
    try:
        os.chmod(p, 0o600)
    except OSError:
        pass
    return f"✔ Saved in `{p}` (readable only by you).", sd_settings.read_hf_token(), sd_settings.hf_status_text()


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
    _, use = sd_prepare.oc_settings()
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
        gfx = sd_settings.get_gfx() or os.environ.get(sd_settings.GFX_VAR, "")
        gpu_facts = [("ROCm", rocm or "not found"), ("GFX version", gfx or "auto")]
    facts = [("sd-scripts", f'{ok(sd_ok)} {"found" if sd_ok else "not found"} · venv {"found" if sd_venv else "missing"}'),
             ("musubi-tuner", f'{ok(mu_ok)} {"found" if mu_ok else "not installed (only for Z-Image, Qwen-Image, FLUX.2)"}'
                              + (f' · venv {"found" if mu_venv else "missing"}' if mu_ok else "")),
             ("GPU", html.escape(gpu)), ("Video memory", vram or "?")] + gpu_facts
    return ('<div class="tf-title">Found on this computer</div><div class="tf-grid">'
            + "".join(f'<div class="tf-fact"><span>{a}</span>{b}</div>' for a, b in facts) + "</div>")


_FIN_MEMO: dict = {}


def detect_venv() -> str:
    for c in (ROOT / "venv", HOME / "sd-script" / "venv"):
        if (c / "bin" / "activate").is_file():
            return str(c)
    return ""
