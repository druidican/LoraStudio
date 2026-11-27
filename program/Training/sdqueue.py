# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""sdqueue.py - the training queue: launch scripts trained one after the other (QUEUE, _queue_worker), with stop on fail,
Completed marks after a run that wrote its final LoRA, and room in video memory when the next script uses another
model (make_room: ComfyUI unloads, the queue waits). The queue survives a restart (queue.json, queue_restore).
One run itself (Run, the reader, start and stop) is in sdgui.py.
"""
from __future__ import annotations
__version__ = "0"  # LoRA Studio version, see version.py

import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path

import notify
import sd_script
import sdgui as S


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
    home = sd_script.script_home(path)
    out = set()
    for k in MODEL_FLAGS:
        v = S.script_arg(text, k, home)
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
    S._signal(proc, signal.SIGTERM)
    deadline = time.time() + 8
    while time.time() < deadline:
        try:
            os.killpg(proc.pid, 0)
        except (ProcessLookupError, PermissionError, OSError):
            return 1
        time.sleep(0.5)
    S._signal(proc, signal.SIGKILL)
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
    if v0 is not None and not S.RUN.running():
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
            if S.RUN.running():
                it["state"], it["note"] = "failed", "another run is still active"
                break
            models = script_models(it["path"])
            t_ = S.script_type(it["path"])
            label = ((sd_script.merged_presets()[1].get(t_) or {}).get("label") or t_ or Path(it["path"]).stem).split(" (")[0]
            room = ""
            if prev_models is not None and models != prev_models:   # another base model: make room for it first
                it["note"] = f"another model ({prev_label} → {label}): freeing memory …"
                room = f"model changed ({prev_label} → {label}): " + (make_room(warn, idle) or "memory freed")
            workdir, venv, trace = S.runner_settings()          # read fresh for every item
            if not S._start_run(it["path"], workdir, venv, trace, it.get("resume", ""), warn, lambda m: None):
                it["state"] = "failed"
                it["note"] = it["note"] or "could not start"
                if QUEUE["stop_on_fail"]:
                    break
                continue
            if room:
                it["note"] = room
            run = S.RUN
            while run.proc.poll() is None:
                time.sleep(1)
            time.sleep(1.5)                                   # let the log reader finish
            code = run.proc.returncode
            if sweep_run(run.proc):                           # helper processes left behind would keep video memory
                print(f"[Queue] stopped processes left over from {Path(it['path']).name}", flush=True)
            prev_models, prev_label = models, label
            took = S.fmt((run.ended or time.time()) - run.started)
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
                    try:                                  # test pictures and the LoRA card before the next run
                        import comfytest
                        base_note = it["note"]
                        it["note"] = base_note + " · making the test pictures …"
                        res = comfytest.after_run(Path(run.logfile).name,
                                                  lambda t, it=it, b=base_note: it.__setitem__("note", f"{b} · {t}"))
                        it["note"] = base_note + " · " + res.split(" · ")[0][:160]
                    except Exception as e:            # noqa: BLE001
                        it["note"] += f" · test pictures: {e}"
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
        if done:                                    # Home › Just trained (the test pictures were made after each run)
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
        out_dir, out_name = S.script_args(text, sd_script.script_home(p))
    except (OSError, TypeError):
        return None
    if not out_name:
        return None
    return (Path(os.path.expanduser(out_dir)) if out_dir else sd_script.script_home(p) / "model") / f"{out_name}.safetensors"


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
    nan = S.nan_text(run)
    if code != 0:
        why = f"exit code {code}"
        if S.fail_where(run):
            why += f", failed in {S.fail_where(run)}"
        return why + (f"; {nan}" if nan else "")
    if run.fail:
        return f"failed in {S.fail_where(run)}: {run.fail['msg']}"
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
        notify.warning("The queue is already running.")
        return
    if S.RUN.running():
        notify.warning("A run is in progress. Wait for it to finish (or press Stop now), then start the queue.")
        return
    try:
        import speedjobs
        if speedjobs.busy():
            notify.warning("A speed test is running (Settings › Speed test). Start the queue when it has finished, "
                       "or stop it there.")
            return
    except ImportError:
        pass
    with QLOCK:
        if not any(i["state"] == "queued" for i in QUEUE["items"]):
            notify.warning("No waiting items in the queue.")
            return
        QUEUE["running"], QUEUE["stop_flag"], QUEUE["stop_on_fail"] = True, False, bool(stop_on_fail)
        QUEUE["killed"] = False
    threading.Thread(target=_queue_worker, daemon=True).start()
    notify.info("Queue started.")


def queue_stop_after_current():
    if not QUEUE["running"]:
        notify.info("The queue is not running.")
        return
    QUEUE["stop_flag"] = True
    notify.info("The queue will stop after the current script finishes.")


# ---- the queue survives a restart: its list is saved, and waiting items can start again (Launch options)
QUEUE_FILE = S.CACHE_DIR / "queue.json"


_Q_SAVED = {"key": None}


def _queue_saver():
    while True:
        time.sleep(3)
        try:
            with QLOCK:
                items = [dict(i) for i in QUEUE["items"]]
            key = json.dumps(items, sort_keys=True)
            if key != _Q_SAVED["key"]:
                S.CACHE_DIR.mkdir(exist_ok=True)
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
    if start and waiting and not S.RUN.running() and not QUEUE["running"]:
        with QLOCK:
            QUEUE["running"], QUEUE["stop_flag"], QUEUE["killed"] = True, False, False
        threading.Thread(target=_queue_worker, daemon=True).start()
        return f"queue resumed: {waiting} waiting item(s) started again"
    return f"queue restored: {waiting} waiting" if items else ""
