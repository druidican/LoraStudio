# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Training: Run and Results.

Run      one row per launch script (with "Resume from epoch N" when a saved state exists), the queue, and the live
         run (stages, progress, loss, save points so far, the next item, the log).
Results  every finished run from the run logs, newest first. Opening one shows the save point to try first, the
         save points with Copy buttons, Clean up for that run (leftovers to the Trash), its settings and the end of
         its log. TensorBoard is tucked away in a fold-out there.
"""
from __future__ import annotations
__version__ = "1.111126i"  # LoRA Studio version, see version.py

import hashlib
import html
import json
import math
import os
import re
import time
from datetime import datetime
from pathlib import Path

import gradio as gr

import sdgui as sg
import features as FT

REG: dict = {}             # short key -> {"path", "state"} for the buttons in the script list
TYPE_LABEL: dict = {}


def _key(*parts) -> str:
    k = hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()[:10]
    return k


def _esc(s) -> str:
    return html.escape(str(s if s is not None else ""))


def type_label(tkey: str) -> str:
    if not TYPE_LABEL:
        try:
            TYPE_LABEL.update({v: re.sub(r"\s*[\(\[].*?[\)\]]", "", l).strip() for l, v in sg.type_choices()})
        except Exception:                                # noqa: BLE001
            pass
    return TYPE_LABEL.get(tkey, tkey or "unknown type")


_INFO_CACHE: dict = {}


def script_info(path: str) -> dict:
    """{dataset, tkey, type, version, out_dir, out_name, text} of a launch script (cached by file time and size)."""
    try:
        st = Path(path).stat()
        ck = (st.st_mtime, st.st_size)
    except OSError:
        ck = None
    hit = _INFO_CACHE.get(path)
    if ck and hit and hit[0] == ck:
        return dict(hit[1])
    info = _script_info(path)
    if ck:
        _INFO_CACHE[path] = (ck, dict(info))
    return info


def _script_info(path: str) -> dict:
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        text = ""
    home = sg.script_home(p)
    out_dir, out_name = sg.script_args(text, home)
    tkey = sg.script_type(path, text)
    m = re.search(r"[vV](\d{1,3})$", out_name or "")
    return {"dataset": home.name, "tkey": tkey, "type": type_label(tkey), "version": f"V{m.group(1)}" if m else "",
            "out_dir": Path(os.path.expanduser(out_dir)) if out_dir else home / "model", "out_name": out_name or "",
            "text": text, "path": str(p)}


def run_name(info: dict) -> str:
    return f"{info['dataset']} · {info['type']}" + (f" {info['version']}" if info["version"] else "")


# --------------------------------------------------------------------------- run logs
_LOGSCRIPT: dict = {}


def log_script(logfile: Path) -> str:
    """The launch script of a run: the ##LS_SCRIPT line, else the script with the log's name (remembered per log)."""
    hit = _LOGSCRIPT.get(logfile.name)
    if hit and Path(hit).is_file():
        return hit
    v = _log_script(logfile)
    if v:
        _LOGSCRIPT[logfile.name] = v
    return v


def _log_script(logfile: Path) -> str:
    try:
        with open(logfile, encoding="utf-8", errors="replace") as fh:
            first = fh.readline()
        if first.startswith("##LS_SCRIPT "):
            return sg.find_script(first[len("##LS_SCRIPT "):].strip())
    except OSError:
        pass
    m = sg.RUN_LOG_RE.match(logfile.name)
    return _scripts_by_stem().get(m.group(3) if m else "", "")


_STEMS = {"t": 0.0, "map": {}}


def _scripts_by_stem() -> dict:
    """{script name without .sh: path} of every launch script, looked up at most every 30 s (old logs without a
    ##LS_SCRIPT line each searched all dataset folders, on every refresh of the lists)."""
    if time.time() - _STEMS["t"] > 30:
        found = {}
        for _, p in sg.list_scripts():
            found.setdefault(Path(p).stem, p)
        _STEMS.update(t=time.time(), map=found)
    return _STEMS["map"]


def runs(limit: int = 200) -> list[dict]:
    """Every run from gui_logs/, newest first: {log, started, script, dur, result, loss, info}."""
    out = []
    for row in sg.run_history(limit):
        started, stem, dur, result, loss, logname = row
        f = sg.LOG_DIR / logname
        script = log_script(f)
        info = script_info(script) if script and Path(script).is_file() else None
        out.append({"log": logname, "started": started, "stem": stem, "script": script, "dur": dur,
                    "result": result, "loss": loss, "info": info})
    return out


_FAIL: dict = {}


def fail_info(logname: str) -> dict | None:
    """The short description of why a run failed (failcodes.py), cached per log file."""
    f = sg.LOG_DIR / (logname or "")
    try:
        st = f.stat()
    except OSError:
        return None
    key = (logname, st.st_size, st.st_mtime)
    if key not in _FAIL:
        try:
            import failcodes
            _FAIL[key] = failcodes.for_log(f)
        except Exception:                                # noqa: BLE001
            _FAIL[key] = None
    return _FAIL[key]


def fail_html(logname: str) -> str:
    fc = fail_info(logname)
    if not fc:
        return ""
    return (f'<div class="rs-failcard"><span class="rs-label">WHAT WENT WRONG</span><div class="rs-big">{_esc(fc["title"])}'
            f'</div><div>{_esc(fc["what"])}</div><div class="rs-failfix"><b>What to do:</b> {_esc(fc["fix"])}</div>'
            + (f'<div class="rs-mono rs-failline">{_esc(fc["line"])}</div>' if fc.get("line") else "")
            + '<p class="ls-note">Press <b>Report for Claude</b> below for a file with everything needed to ask about it.'
              '</p></div>')


def _epoch_of_stop(info: dict) -> int | None:
    """Newest saved epoch file of a script's output (for 'stopped at epoch N')."""
    if not info or not info["out_dir"].is_dir() or not info["out_name"]:
        return None
    eps = []
    for f in info["out_dir"].glob(info["out_name"] + "-*.safetensors"):
        m = re.search(r"-(\d{6})$", f.stem)
        if m:
            eps.append(int(m.group(1)))
    return max(eps) if eps else None


def _states(info: dict) -> list[tuple[Path, int | None]]:
    """[(state folder, epoch)] of a script, newest first."""
    if not info or not info["out_dir"].is_dir():
        return []
    out = []
    for st in sg.scan_states(info["out_dir"]):
        if info["out_name"] and not st.name.startswith(info["out_name"] + "-"):
            continue
        ep = None
        try:
            ep = json.loads((st / "train_state.json").read_text()).get("current_epoch")
        except Exception:                                # noqa: BLE001
            m = re.search(r"-(\d{6})-state$", st.name)
            ep = int(m.group(1)) if m else None
        out.append((st, ep))
    return out


def _secs(dur: str) -> int:
    try:
        parts = [int(x) for x in dur.split(":")]
    except (ValueError, AttributeError):
        return 0
    while len(parts) < 3:
        parts.insert(0, 0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def _human(sec: int) -> str:
    if sec <= 0:
        return "?"
    h, m = divmod(int(sec) // 60, 60)
    return f"{h} h {m} min" if h else f"{m} min"


def estimate(path: str, hist: list | None = None) -> int:
    """Seconds a script will take: its own last finished run, else the average of finished runs of its type."""
    hist = hist if hist is not None else runs()
    mine = [r for r in hist if r["script"] == path and r["result"] == "done"]
    if mine:
        return _secs(mine[0]["dur"])
    t = script_info(path)["tkey"]
    same = [_secs(r["dur"]) for r in hist if r["result"] == "done" and r["info"] and r["info"]["tkey"] == t]
    same = [s for s in same if s > 60]
    return int(sum(same) / len(same)) if same else 0


# --------------------------------------------------------------------------- Run: scripts and queue
def scripts_html(show_done: bool = False) -> str:
    hist = runs()
    last = {}
    for r in hist:
        last.setdefault(r["script"], r)
    with sg.QLOCK:
        queued = {i["path"] for i in sg.QUEUE["items"] if i["state"] in ("queued", "running")}
    done_data = sg.COMPLETED.load() if sg.COMPLETED else {}
    rows, hidden = [], 0
    REG.clear()
    for _, path in sg.list_scripts():
        info = script_info(path)
        is_done = sg.script_completed(path, info["text"], done_data)
        if is_done and not show_done:
            hidden += 1
            continue
        k = _key(path)
        REG[k] = {"path": path, "state": ""}
        r = last.get(path)
        running = sg.RUN.running() and sg.RUN.script == path
        if running:
            state = '<span class="ls-chip warn">running</span>'
        elif not r:
            state = '<span class="tr-muted">never run</span>'
        elif r["result"] == "done" and (is_done or sg.COMPLETED is None):
            state = f'<span class="tr-muted">done {_esc(r["started"][5:10])}</span>'
        elif r["result"].startswith("failed"):
            fc = fail_info(r["log"])
            state = (f'<span class="tr-bad" title="{_esc(fc["what"]) if fc else ""}">failed {_esc(r["started"][5:10])}'
                     f'{": " + _esc(fc["title"]) if fc else ""}</span>')
        else:
            ep = _epoch_of_stop(info)
            state = f'<span class="tr-warn">stopped{f" at epoch {ep}" if ep else ""}</span>'
        finished = bool(r) and r["result"] == "done" and (is_done or sg.COMPLETED is None)
        btn = ('<span class="ls-chip">in the queue</span>' if path in queued else
               f'<button class="tr-btn" onclick="lsSend(\'tr_box\', \'add|{k}\')">{"Add again" if finished else "Add"}</button>')
        not_run = not running and not finished           # stopped, failed or cut off runs count as not run
        tick = (f'<input type="checkbox" class="tr-tick" data-k="{k}" data-notrun="{1 if not_run else 0}" '
                f'aria-label="Tick {_esc(info["dataset"])} {_esc(info["type"])}"'
                + (' disabled title="already in the queue"' if path in queued or running else "") + ">")
        mark = (f'<button class="tr-btn tr-small" title="You ran it another way, or do not need it: mark this version as '
                f'run and finished (hidden; Show finished shows it again)" onclick="lsSend(\'tr_box\', \'mark|{k}\')">'
                'Mark as run</button>' if not_run and not is_done and path not in queued else "")
        extra = ""
        if not FT.can_train_text(info["text"]):          # its trainer is not installed: made here, run elsewhere
            part = "musubi" if "MUSUBI_DIR=" in info["text"] else "sd_scripts"
            btn = f'<span class="tr-muted ls-missing">{_esc(FT.missing(part))}</span>'
            tick = '<input type="checkbox" class="tr-tick" disabled title="its trainer is not installed">'
            mark = ""
        st = _states(info) if FT.can_train_text(info["text"]) else []
        if st and not running:
            sk = _key(path, st[0][0])
            REG[sk] = {"path": path, "state": str(st[0][0])}
            lab = f"Resume from epoch {st[0][1]}" if st[0][1] else "Resume from the saved state"
            extra = f'<div class="tr-sub"><button class="tr-btn tr-small" onclick="lsSend(\'tr_box\', \'add|{sk}\')">{lab}</button></div>'
        done_chip = '<span class="ls-chip ok">✓ completed</span>' if is_done else ""
        rows.append(f'<div class="tr-row">{tick}<div class="tr-main"><b>{_esc(info["dataset"])}</b> '
                    f'<span>{_esc(info["type"])} {_esc(info["version"])}</span> {done_chip}</div>'
                    f'{state}{mark}{btn}</div>{extra}')
    if not rows:
        rows.append('<p class="ls-empty">No launch scripts yet. Make one in Datasets › New dataset or Edit existing '
                    '(step 9).</p>' if not hidden else '<p class="ls-empty">Every launch script is completed. Tick '
                    '<i>show finished</i> to see them.</p>')
    note = (f'<p class="ls-note">{hidden} completed launch script{"s" if hidden != 1 else ""} hidden.</p>' if hidden and not show_done else "")
    return ('<div class="tr-list">' + "".join(rows) + "</div>" + note +
            '<p class="ls-note">One row per launch script. Epoch files are not listed here: they are with each run '
            'in Results. A saved state shows as “Resume from epoch N”.</p>')


def queue_html() -> str:
    with sg.QLOCK:
        items = [dict(i) for i in sg.QUEUE["items"]]
        running_q = sg.QUEUE["running"]
    hist = runs()
    waiting = [i for i in items if i["state"] == "queued"]
    total = sum(estimate(i["path"], hist) for i in waiting)
    head = (f'{len(waiting)} waiting' + (f' · about {_human(total)} in all' if total else '')
            + (' · the queue is running' if running_q else ''))
    rows = []
    icon = {"queued": "", "running": '<span class="ls-chip warn">running</span>',
            "done": '<span class="ls-chip ok">✓ done</span>', "failed": '<span class="ls-chip bad">✕ failed</span>',
            "stopped": '<span class="ls-chip">stopped</span>'}
    for n, it in enumerate(items):
        info = script_info(it["path"])
        res = ""
        if it.get("resume"):
            m = re.search(r"-(\d{6})-state$", Path(it["resume"]).name)
            res = f'<span class="ls-chip warn">resume{f" from epoch {int(m.group(1))}" if m else ""}</span>'
        est = estimate(it["path"], hist) if it["state"] == "queued" else 0
        ctrl = ""
        if it["state"] == "queued":
            ctrl = (f'<button class="tr-btn tr-small" title="Move up" onclick="lsSend(\'tr_box\', \'up|{n}\')">↑</button>'
                    f'<button class="tr-btn tr-small" title="Move down" onclick="lsSend(\'tr_box\', \'down|{n}\')">↓</button>'
                    f'<button class="tr-btn tr-small" onclick="lsSend(\'tr_box\', \'rm|{n}\')">Remove</button>')
        elif it["state"] in ("done", "failed", "stopped"):
            ctrl = f'<button class="tr-btn tr-small" onclick="lsSend(\'tr_box\', \'rm|{n}\')">Remove</button>'
        note = f'<span class="tr-muted">{_esc(it.get("note", ""))}</span>' if it.get("note") else ""
        rows.append(f'<div class="tr-row tr-q-{it["state"]}"><div class="tr-main"><b>{_esc(run_name(info))}</b> {res} {note}</div>'
                    f'{icon.get(it["state"], "")}<span class="tr-muted">{"≈ " + _human(est) if est else ""}</span>'
                    f'<span class="tr-ctrl">{ctrl}</span></div>')
    if not rows:
        rows.append('<p class="ls-empty">The queue is empty. Press <b>Add</b> next to a launch script.</p>')
    return f'<div class="tr-qhead">{head}</div><div class="tr-list">' + "".join(rows) + "</div>"


def tr_action(v, show_done):
    act, _, rest = (v or "").partition("|")
    arg = rest.rsplit("|", 1)[0] if "|" in rest else rest
    msg = ""
    if act == "add":
        it = REG.get(arg)
        if not it:
            msg = "That list is out of date; it was refreshed. Press the button again."
        elif not FT.can_train_text(script_info(it["path"]).get("text", "")):
            msg = "Its trainer is not installed here (Settings › Start and network › What is installed)."
        else:
            with sg.QLOCK:
                sg.QUEUE["items"].append({"path": it["path"], "resume": it["state"], "state": "queued", "note": ""})
            msg = f"Added {run_name(script_info(it['path']))}" + (" (resume)" if it["state"] else "") + " to the queue."
    elif act == "mark":
        it = REG.get(arg)
        if not it:
            msg = "That list is out of date; it was refreshed. Press the button again."
        elif sg.COMPLETED is None:
            msg = "Completed marks are not available (completed.py is missing)."
        else:
            info = script_info(it["path"])
            sg.COMPLETED.set_done(info["dataset"], info["tkey"], True, how="manual", lora=info["out_name"])
            msg = (f"✔ {run_name(info)} is marked as run and finished, so it is hidden. Tick *show finished* to see it; "
                   "untick it there (Collection › LoRA page) to bring it back.")
    elif act in ("up", "down", "rm"):
        try:
            i = int(arg)
        except ValueError:
            i = -1
        with sg.QLOCK:
            items = sg.QUEUE["items"]
            if 0 <= i < len(items):
                if act == "rm":
                    if items[i]["state"] == "running":
                        msg = "The running item cannot be removed: press Stop now first."
                    else:
                        del items[i]
                else:
                    j = i + (-1 if act == "up" else 1)
                    if 0 <= j < len(items) and "running" not in (items[i]["state"], items[j]["state"]):
                        items[i], items[j] = items[j], items[i]
    return scripts_html(show_done), queue_html(), msg


def add_many(keys_json, show_done):
    """The ticked launch scripts, in list order, to the queue (skipping ones already waiting)."""
    try:
        keys = json.loads(keys_json or "[]")
    except ValueError:
        keys = []
    added, skipped = [], 0
    with sg.QLOCK:
        queued = {i["path"] for i in sg.QUEUE["items"] if i["state"] in ("queued", "running")}
        for k in keys:
            it = REG.get(k)
            if not it or it["path"] in queued:
                skipped += 1
                continue
            sg.QUEUE["items"].append({"path": it["path"], "resume": "", "state": "queued", "note": ""})
            queued.add(it["path"])
            added.append(it["path"])
    if not keys:
        msg = "Tick one or more launch scripts first."
    else:
        msg = (f"Added {len(added)} launch script{'s' if len(added) != 1 else ''} to the queue"
               + (f" ({skipped} already there or out of date)" if skipped else "") + ".")
    return scripts_html(show_done), queue_html(), msg


def clear_queue(finished_only: bool):
    with sg.QLOCK:
        sg.QUEUE["items"] = [i for i in sg.QUEUE["items"]
                             if i["state"] == "running" or (finished_only and i["state"] == "queued")]
    return queue_html()


def start_queue(stop_on_fail):
    sg.queue_start(stop_on_fail)
    return queue_html()


# --------------------------------------------------------------------------- Run: the live run
def live_head() -> str:
    run = sg.RUN
    if not run.name:
        return ""
    info = script_info(run.script) if run.script else {"dataset": run.name, "type": "", "version": ""}
    started = datetime.fromtimestamp(run.started).strftime("%H:%M") if run.started else "?"
    if run.running():
        when = f"started {started} · running {_human(time.time() - run.started)}"
    else:
        when = f"started {started} · ended after {_human((run.ended or time.time()) - run.started)}"
    return (f'<div class="tr-livehead"><div><div class="tr-livename">{_esc(run_name(info))}</div>'
            f'<div class="tr-muted">{when}</div></div></div>')


def savepoints_html() -> str:
    run = sg.RUN
    if not run.script:
        return ""
    info = script_info(run.script)
    eps = []
    if info["out_dir"].is_dir() and info["out_name"]:
        for f in info["out_dir"].glob(info["out_name"] + "-*.safetensors"):
            m = re.search(r"-(\d{6})$", f.stem)
            if m:
                eps.append(int(m.group(1)))
    eps.sort()
    total = run.epochs_total or None
    chips = "".join(f'<span class="ls-chip ok">epoch {e}</span>' for e in eps) or '<span class="tr-muted">none yet</span>'
    with sg.QLOCK:
        nxt = next((i for i in sg.QUEUE["items"] if i["state"] == "queued"), None)
    nxt_html = ""
    if nxt:
        ni = script_info(nxt["path"])
        nxt_html = (f'<div class="tr-next"><b>Next in the queue</b><br>{_esc(run_name(ni))}'
                    + (" (resume)" if nxt.get("resume") else "") + f' <span class="tr-muted">≈ {_human(estimate(nxt["path"]))}</span></div>')
    return (f'<div class="tr-sp"><b>Save points so far</b><div class="tr-chips">{chips}'
            + (f'<span class="tr-muted">of {total} epochs</span>' if total else "") + '</div>'
              '<p class="ls-note">Results gives the advice on which one to try first when the run ends.</p></div>' + nxt_html)


def live_tick(seen):
    run = sg.RUN
    show = bool(run.name)
    v = run.version
    return (gr.update(visible=show), live_head(), sg.stage_html(), sg.progress_html(), sg.loss_html(),
            savepoints_html(), (sg.log_text() if v != seen else gr.update()), v)


# --------------------------------------------------------------------------- Results
def results_rows(ds_filter: str, res_filter: str, not_copied: bool) -> list[dict]:
    hist = [r for r in runs() if r["result"] != "running"]
    out = []
    lora_names = None
    for r in hist:
        info = r["info"]
        if ds_filter and ds_filter != "All" and (not info or info["dataset"] != ds_filter):
            continue
        res = r["result"]
        kind = "done" if res == "done" else "failed" if res.startswith("failed") else "stopped"
        if res_filter and res_filter != "All" and kind != res_filter.lower():
            continue
        copied = ""
        if info and info["out_name"]:
            if lora_names is None:
                try:
                    lora_names = {f.name: f.stat().st_size for f in sg.HK.lora_files(sg.lora_dirs_setting())} if sg.HK else {}
                except OSError:
                    lora_names = {}
            final = info["out_dir"] / f"{info['out_name']}.safetensors"
            names = {n for n in lora_names if n.startswith(info["out_name"])}
            copied = "copied" if (final.name in names) else ("epoch copied" if names else "not yet")
        if not_copied and copied in ("copied", "epoch copied"):
            continue
        out.append({**r, "kind": kind, "copied": copied})
    return out


def _advice(script: str):
    try:
        import savepoint as SP
        return SP.for_script(script)
    except Exception:                                    # noqa: BLE001
        return None


_LEFT = {}


def _leftovers(script: str) -> tuple[int, str]:
    """(bytes, short text) of what Clean up would move to the Trash for this run's script."""
    key = (script, int(time.time() // 30))
    if key in _LEFT:
        return _LEFT[key]
    try:
        base, md, keep, remove, err = sg.cleanup_plan(script, True, keep_copied=True)
    except Exception:                                    # noqa: BLE001
        err, remove = "?", []
    if err:
        res = (0, "")
    else:
        size = sum(sg._size(p) for p, _ in remove)
        res = (size, "cleaned" if not remove else _gb(size))
    _LEFT[key] = res
    return res


def _gb(n) -> str:
    return f"{n / 1024 ** 3:.1f} GB" if n >= 1024 ** 3 * 0.1 else f"{n / 1024 ** 2:.0f} MB"


def results_html(ds_filter="All", res_filter="All", not_copied=False) -> str:
    rows = results_rows(ds_filter, res_filter, not_copied)
    if not rows:
        return ('<p class="ls-empty">No finished runs yet. They appear here when the queue has run a launch script.</p>'
                if ds_filter in ("", "All") and res_filter in ("", "All") and not not_copied
                else '<p class="ls-empty">No run matches these filters.</p>')
    body = []
    for r in rows[:150]:
        info = r["info"]
        name = run_name(info) if info else r["stem"]
        if r["kind"] == "done":
            res = '<span class="ls-chip ok">done</span>'
        elif r["kind"] == "failed":
            fc = fail_info(r["log"])
            res = (f'<span class="ls-chip bad">{_esc(r["result"])}</span>'
                   + (f'<div class="rs-fail">{_esc(fc["title"])}</div>' if fc else ""))
        else:
            ep = _epoch_of_stop(info) if info else None
            res = f'<span class="ls-chip">stopped{f" at epoch {ep}" if ep else ""}</span>'
        a = _advice(r["script"]) if r["script"] and r["kind"] != "failed" else None
        left = _leftovers(r["script"])[1] if r["script"] and Path(r["script"]).is_file() else ""
        star = ' <span class="rs-star" title="Marked as the best version">★</span>' if is_best_run(info) else ""
        body.append(f'<tr><td><input type="checkbox" class="rs-tick" data-log="{_esc(r["log"])}" '
                    f'title="Tick 2-4 runs to compare"></td>'
                    f'<td>{_esc(r["started"])}</td><td><b>{_esc(name)}</b>{star}</td><td>{_esc(r["dur"])}</td>'
                    f'<td>{res}</td><td>{_esc(a["short"] if a else "")}</td><td>{_esc(r["copied"])}</td>'
                    f'<td>{_esc(left)}</td><td><button class="tr-btn tr-small" '
                    f'onclick="lsSend(\'rs_box\', \'open|{_esc(r["log"])}\')">Open</button></td></tr>')
    return ('<div class="rs-wrap"><table class="rs-table"><thead><tr><th></th><th>Finished</th><th>LoRA</th><th>Time</th>'
            '<th>Result</th><th>Try first</th><th>Copied</th><th>Leftovers</th><th></th></tr></thead><tbody>'
            + "".join(body) + '</tbody></table></div><p class="ls-note">Every run from the run logs, newest first. '
            '“Leftovers” = epoch files you did not copy, saved states and caches that Clean up can move to the Trash. '
            'Tick 2-4 runs and press <b>Compare the ticked</b> to see them side by side. ★ = the version marked as the '
            'best one.</p>')


def _final_and_epochs(info) -> tuple[Path | None, dict]:
    files, final = {}, None
    if info and info["out_dir"].is_dir() and info["out_name"]:
        for f in info["out_dir"].glob(info["out_name"] + "*.safetensors"):
            if f.stem == info["out_name"]:
                final = f
            m = re.search(r"-(\d{6})$", f.stem)
            if m and f.stem[:-7] == info["out_name"]:
                files[int(m.group(1))] = f
    return final, files


def open_run(logname: str):
    """Everything the one-run page shows: (head, try-first, plot, save points, cleanup text, this-run, log, dest,
    state)."""
    import pandas as pd
    empty = pd.DataFrame({"epoch": [], "loss": []})
    f = sg.LOG_DIR / (logname or "")
    if not logname or not f.is_file():
        return ("", "", empty, "", "", "", "", "", {})
    hist = {r["log"]: r for r in runs()}
    r = hist.get(logname) or {"started": "", "dur": "", "result": "", "script": log_script(f), "info": None}
    info = r["info"] or (script_info(r["script"]) if r["script"] and Path(r["script"]).is_file() else None)
    name = run_name(info) if info else logname
    if is_best_run(info):
        name += "  ★ best"
    head = (f'<div class="rs-head"><div class="tr-livename">{_esc(name)}</div><div class="tr-muted">'
            f'{_esc(r["result"])} · started {_esc(r["started"])} · {_esc(r["dur"])}</div></div>')
    a = _advice(r["script"]) if r.get("script") else None
    final, eps = _final_and_epochs(info)
    best = a["best"] if a else None
    if a:
        txt = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", a["text"])
        big = _esc(a["short"][4:].replace(" first", "").capitalize())
        try_first = (f'<div class="rs-try"><span class="rs-label">TRY FIRST</span><div class="rs-big">{big}</div>'
                     f'<div>{txt}</div><p class="ls-note">Loss is a rough guide for image LoRAs: make a few test '
                     'pictures with both.</p></div>')
    elif final:
        try_first = ('<div class="rs-try"><span class="rs-label">TRY FIRST</span><div class="rs-big">The final file</div>'
                     '<p class="ls-note">No loss log to compare the save points (needs --logging_dir).</p></div>')
    else:
        try_first = '<div class="rs-try"><p class="ls-empty">No LoRA file from this run.</p></div>'
    if str(r.get("result", "")).startswith("failed") or "cut off" in str(r.get("result", "")):
        fh = fail_html(logname)
        if fh:
            try_first = fh + (try_first if (a or final) else "")
    plot = pd.DataFrame({"epoch": [e for e, _ in a["points"]], "loss": [v for _, v in a["points"]]}) if a else empty
    pts = dict(a["points"]) if a else {}
    try:
        lora_names = {f_.name for f_ in sg.HK.lora_files(sg.lora_dirs_setting())} if sg.HK else set()
    except OSError:
        lora_names = set()
    rows = []
    items = sorted(eps.items())
    if final:
        items.append((a["final"] if a else "final", final))
    for ep, path in items:
        is_final = path == final
        tag = " · final" if is_final and ep != "final" else ""
        tag += " · try first" if best is not None and ep == best else ""
        loss = pts.get(ep)
        alt = f"{info['out_name']}-e{ep}.safetensors" if not is_final else path.name
        copied = "yes" if (path.name in lora_names or alt in lora_names) else "no"
        try:
            size = _gb(path.stat().st_size)
        except OSError:
            size = ""
        rows.append(f'<tr{" class=rs-best" if best is not None and ep == best else ""}><td class="rs-mono">{_esc(path.stem)}</td>'
                    f'<td>{ep}{tag}</td><td>{f"{loss:.4f}" if isinstance(loss, float) and math.isfinite(loss) else ""}</td>'
                    f'<td>{size}</td><td>{copied}</td><td><button class="tr-btn tr-small" '
                    f'onclick="lsSend(\'rs_box\', \'copy|{_esc(path.name)}\')">Copy</button></td></tr>')
    sp = ('<table class="rs-table"><thead><tr><th>File</th><th>Epoch</th><th>Loss</th><th>Size</th>'
          '<th>In the LoRA folder</th><th></th></tr></thead><tbody>' + "".join(rows) + "</tbody></table>") if rows else \
         '<p class="ls-empty">No save points in the model folder.</p>'
    # cleanup
    clean = ""
    if r.get("script") and Path(r["script"]).is_file():
        base, md, keep, remove, err = sg.cleanup_plan(r["script"], True, keep_copied=True)
        if err:
            clean = _esc(err)
        elif not remove:
            clean = "Nothing to clean up: no epoch files you did not copy, no saved states, no caches."
        else:
            kinds = {}
            for p, k in remove:
                kinds[k] = kinds.get(k, 0) + 1
            size = sum(sg._size(p) for p, _ in remove)
            plural = {"epoch file you did not copy": "epoch files you did not copy", "saved state": "saved states",
                      "cache (.npz)": "latent cache files (.npz)", "epoch checkpoint": "epoch checkpoints"}
            clean = (f"**{_gb(size)}** of leftovers: " + ", ".join(f"{n} {plural.get(k, k) if n != 1 else k}"
                                                                  for k, n in kinds.items())
                     + ". The final file, the copied ones, the pictures, captions, launch script and logs stay.")
    # this run
    this = ""
    if info:
        t = info["text"]
        base = Path(info["path"]).parent

        def arg(n):
            return sg.script_arg(t, n, base) or "?"
        try:
            st = sg.dataset_stats(info["dataset"])
            pics = " + ".join(f'{x["n"]} × {x["repeats"]}' for x in st["rows"]) or "?"
        except Exception:                                # noqa: BLE001
            pics = "?"
        this = (f'<div class="rs-this">pictures × repeats {pics} · {arg("max_train_epochs")} epochs · LR {arg("learning_rate")}'
                f' · dim {arg("network_dim")}/{arg("network_alpha")} · {arg("resolution")}</div>')
    tail = ""
    try:
        size = f.stat().st_size
        with open(f, "rb") as fh:
            fh.seek(max(0, size - 60_000))
            lines = fh.read().decode("utf-8", "replace").replace("\r", "\n").splitlines()
        tail = "\n".join(l for l in lines if l.strip() and not l.startswith("##LS_"))[-4000:]
        tail = "\n".join(tail.splitlines()[-40:])
    except OSError:
        pass
    dest = ""
    if info and sg.HK and (final or eps):
        try:
            tgt = sg.HK.copy_target(info["out_name"], sg.lora_dirs_setting(), info["type"])
            dest = str(tgt["dir"] or "")
        except Exception:                                # noqa: BLE001
            dest = ""
    state = {"log": logname, "script": r.get("script") or "", "best": best,
             "final": str(final) if final else "", "out": info["out_name"] if info else "",
             "model": str(info["out_dir"]) if info else ""}
    return head, try_first, plot, sp, clean, this, tail, dest, state


def copy_file(state: dict, filename: str, dest: str, overwrite: bool = False) -> str:
    if not state or not state.get("model"):
        return "Open a run first."
    src = Path(state["model"]) / filename
    if not src.is_file():
        return f"Not found: {filename}"
    if not (dest or "").strip():
        return "Enter the LoRA folder to copy to."
    name = ""
    m = re.search(r"-(\d{6})$", src.stem)
    if m and state.get("out") and src.stem[:-7] == state["out"]:
        name = f"{state['out']}-e{int(m.group(1))}.safetensors"
    ok, msg = sg.HK.copy_lora(src, Path(os.path.expanduser(dest.strip())), bool(overwrite), name=name)
    if ok and state.get("script"):
        try:
            copied = Path(os.path.expanduser(dest.strip())) / (name or src.name)
            note = write_sidecar_for(copied, state["script"], int(m.group(1)) if m else None)
            if note:
                msg += f" · {note} (`{copied.stem}.metadata.json`)"
        except Exception as e:                           # noqa: BLE001
            msg += f" · LoRA Manager data not written: {e}"
    return ("✔ " if ok else "❌ ") + msg


# --------------------------------------------------------------------------- LoRA Manager data (.metadata.json)
def _training_facts(script: str) -> dict:
    info = script_info(script)
    text, home = info["text"], sg.script_home(script)
    arg = lambda n: sg.script_arg(text, n, home) or ""
    try:
        import dsflow
        m = dsflow.meta(info["dataset"])
    except Exception:                                    # noqa: BLE001
        m = {}
    try:
        stats = sg.dataset_stats(info["dataset"])
        pics, reps = stats["total"], "/".join(str(r["repeats"]) for r in stats["rows"])
    except Exception:                                    # noqa: BLE001
        pics, reps = 0, ""
    return {"info": info, "trigger": (m.get("trigger") or info["dataset"]), "purpose": m.get("purpose") or "",
            "pics": pics, "repeats": reps, "epochs": arg("max_train_epochs"), "lr": arg("learning_rate"),
            "dim": arg("network_dim"), "alpha": arg("network_alpha"), "res": arg("resolution"),
            "base_file": Path(arg("pretrained_model_name_or_path") or arg("dit")).name}


def _sidecar(lora_file: Path, script: str, epoch: int | None) -> dict:
    """LoRA Manager data for a LoRA LoRA Studio trained: trigger word, base model and the training in the notes."""
    import lora_library as ll
    f = _training_facts(script)
    info = f["info"]
    data = ll.new_sidecar(ll.LIB, type("L", (), {"stem": lora_file.stem, "path": lora_file, "preview": None,
                                                  "base": "Unknown"})())
    try:
        data["base_model"] = ll.detect_base(ll.read_safetensors_meta(lora_file), None, lora_file, ll.LIB.settings["bases"])
    except Exception:                                    # noqa: BLE001
        data["base_model"] = info["type"]
    data["civitai"] = {"trainedWords": [f["trigger"]] if f["trigger"] else []}
    bits = [f"Trained with LoRA Studio on {time.strftime('%Y-%m-%d')}: dataset {info['dataset']}"
            + (f" ({f['pics']} picture{'s' if f['pics'] != 1 else ''} × {f['repeats']} repeats)" if f["pics"] else ""),
            f"{info['type']} {info['version']}".strip() + (f", save point epoch {epoch}" if epoch else ", final file"),
            ", ".join(x for x in (f"{f['epochs']} epochs" if f["epochs"] else "", f"learning rate {f['lr']}" if f["lr"] else "",
                                  f"dim {f['dim']} / alpha {f['alpha']}" if f["dim"] else "",
                                  f"resolution {f['res']}" if f["res"] else "") if x),
            f"base model file {f['base_file']}" if f["base_file"] else ""]
    data["notes"] = ". ".join(b for b in bits if b) + "."
    data["tags"] = [t for t in (f["purpose"], info["type"]) if t]
    data["usage_tips"] = json.dumps({"strength": 1.0})
    return data


def write_sidecar_for(lora_file: Path, script: str, epoch: int | None = None) -> str:
    """Write <lora>.metadata.json next to a LoRA file. An existing one keeps everything you filled in; only empty
    fields (trigger word, notes, base model, tags) are filled. Returns a short note."""
    import lora_library as ll
    lora_file = Path(lora_file)
    target = lora_file.with_name(lora_file.stem + ".metadata.json")
    new = _sidecar(lora_file, script, epoch)
    if target.is_file():
        try:
            old = json.loads(target.read_text(encoding="utf-8"))
        except Exception:                                # noqa: BLE001
            old = {}
        merged = dict(old)
        if not ((old.get("civitai") or {}).get("trainedWords")):
            merged.setdefault("civitai", {})
            merged["civitai"] = dict(merged.get("civitai") or {}, trainedWords=new["civitai"]["trainedWords"])
        for k in ("notes", "tags"):
            if not old.get(k):
                merged[k] = new[k]
        if not old.get("base_model") or str(old.get("base_model")).lower() == "unknown":
            merged["base_model"] = new["base_model"]
        for k in ("file_name", "file_path", "size", "modified", "sha256"):
            merged[k] = new[k]
        if merged == old:
            return ""
        ll.write_sidecar(target, merged)
        return "LoRA Manager data completed"
    ll.write_sidecar(target, new)
    return "LoRA Manager data written"


def sidecar_after_run(script: str):
    """After a finished run: the dataset's copy of the LoRA Manager data for the final file (Collection › Edit)."""
    info = script_info(script)
    final = info["out_dir"] / f"{info['out_name']}.safetensors"
    if not info["out_name"] or not final.is_file():
        return
    import lora_library as ll
    arch = sg.script_home(script) / ll.ARCHIVE_DIR
    tgt = arch / (final.stem + ".metadata.json")
    if tgt.is_file():
        return
    ll.write_sidecar(tgt, _sidecar(final, script, None) | {"file_path": str(final)})


def copy_best(state, dest, overwrite):
    if not state:
        return "Open a run first."
    best = state.get("best")
    if best is None:
        return copy_final(state, dest, overwrite)
    md, out = Path(state["model"]), state["out"]
    f = md / f"{out}-{int(best):06d}.safetensors"
    if not f.is_file():
        return copy_final(state, dest, overwrite)
    return copy_file(state, f.name, dest, overwrite)


def copy_final(state, dest, overwrite):
    if not state or not state.get("final"):
        return "This run has no final file."
    return copy_file(state, Path(state["final"]).name, dest, overwrite)


def cleanup(state, ok):
    if not state or not state.get("script"):
        return "Open a run first."
    if not ok:
        return "Tick the confirmation first."
    base, md, keep, remove, err = sg.cleanup_plan(state["script"], True, keep_copied=True)
    if err:
        return err
    if not remove:
        return "Nothing to clean up."
    size = sum(sg._size(p) for p, _ in remove)
    moved, failed = sg.HK.trash([p for p, _ in remove]) if sg.HK else (0, ["housekeeping.py missing"])
    _LEFT.clear()
    return f"✔ Moved {moved} item(s), {_gb(size)}, to the Trash." + (f" Could not move: {'; '.join(failed[:3])}" if failed else "")


def run_again(state):
    if not state or not state.get("script") or not Path(state["script"]).is_file():
        return "The launch script of this run no longer exists."
    with sg.QLOCK:
        sg.QUEUE["items"].append({"path": state["script"], "resume": "", "state": "queued", "note": ""})
    return "Added to the queue: Training › Run."


# --------------------------------------------------------------------------- the pages
def build_run(app):
    import features as FT
    trains = FT.have("sd_scripts") or FT.have("musubi")
    if not trains:
        gr.Markdown("**Training (sd-scripts and musubi-tuner missing)**: neither trainer is installed here, so the "
                    "queue and Start are hidden. The launch scripts below are made by Datasets › 9 as usual: copy the "
                    "dataset folder to the computer with the trainer and run them there, or install a trainer and "
                    "tick it in Settings › Start and network.", elem_classes=["ls-missing-note"])
    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=380):
            gr.Markdown("### Launch scripts")
            with gr.Row(elem_classes=["ls-bar"]):
                show_done = gr.Checkbox(label="show finished", value=False, scale=0, min_width=150)
                refresh = gr.Button("Refresh", size="sm", scale=0, min_width=100)
            with gr.Row(elem_classes=["ls-bar"], visible=trains):
                t_notrun = gr.Button("Tick the ones not run", size="sm", scale=0, min_width=180)
                t_none = gr.Button("Untick all", size="sm", scale=0, min_width=110)
                t_add = gr.Button("Add the ticked to the queue", variant="primary", size="sm", scale=0, min_width=230)
            t_sel = gr.Textbox(visible=False)
            s_html = gr.HTML(elem_id="tr_scripts")
        with gr.Column(scale=1, min_width=380, visible=trains):
            gr.Markdown("### Queue")
            q_html = gr.HTML(elem_id="tr_queue")
            with gr.Row(elem_classes=["ls-bar"]):
                failstop = gr.Checkbox(label="Stop the queue when a run fails", value=True, scale=0, min_width=260)
                clear_done = gr.Button("Clear finished", size="sm", scale=0, min_width=120)
                clear_all = gr.Button("Clear the queue", size="sm", scale=0, min_width=130)
                start = gr.Button("Start queue", variant="primary", size="sm", scale=0, min_width=130)
            gr.Markdown("Starting the queue stops Ollama to free video memory. It is not started again by itself.",
                        elem_classes=["ls-note"])
            msg = gr.Markdown()
    box = gr.Textbox(elem_id="tr_box", show_label=False, container=False)
    with gr.Column(visible=bool(sg.RUN.name), elem_classes=["ls-card"]) as live:
        with gr.Row(equal_height=True):
            with gr.Column(scale=1, min_width=260):
                head = gr.HTML(live_head())
            stop_after = gr.Button("Stop after this run", size="sm", scale=0, min_width=160)
            stop_now = gr.Button("Stop now", variant="stop", size="sm", scale=0, min_width=110)
        stage = gr.HTML(sg.stage_html(), elem_id="stage_box")
        prog = gr.HTML(sg.progress_html(), elem_id="progress_line")
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                loss = gr.HTML(sg.loss_html(), elem_id="loss_box")
            with gr.Column(scale=2):
                sp = gr.HTML(savepoints_html(), elem_id="tr_sp")
        with gr.Accordion("Run log", open=False):
            log = gr.Textbox(lines=18, max_lines=18, autoscroll=True, show_label=False, interactive=False,
                             value=sg.log_text())
    seen = gr.State(-1)

    box.change(tr_action, [box, show_done], [s_html, q_html, msg])
    t_notrun.click(None, None, None, js="() => document.querySelectorAll('#tr_scripts .tr-tick:not([disabled])')"
                                        ".forEach(function(i){ i.checked = i.dataset.notrun === '1'; })")
    t_none.click(None, None, None, js="() => document.querySelectorAll('#tr_scripts .tr-tick').forEach(function(i){ "
                                      "i.checked = false; })")
    t_add.click(add_many, [t_sel, show_done], [s_html, q_html, msg],
                js="(x, sd) => [JSON.stringify(Array.from(document.querySelectorAll('#tr_scripts .tr-tick:checked'))"
                   ".map(function(e){ return e.dataset.k; })), sd]")
    show_done.change(scripts_html, show_done, s_html)
    refresh.click(scripts_html, show_done, s_html).then(queue_html, None, q_html)
    clear_done.click(lambda sd: (clear_queue(True), scripts_html(sd)), show_done, [q_html, s_html])
    clear_all.click(lambda sd: (clear_queue(False), scripts_html(sd)), show_done, [q_html, s_html])
    start.click(start_queue, failstop, q_html)
    failstop.change(sg.queue_failstop, failstop, None)
    stop_after.click(sg.queue_stop_after_current, None, None)
    stop_now.click(sg.stop_run, None, None)
    app.load(scripts_html, show_done, s_html).then(queue_html, None, q_html)
    last = gr.State("")

    def q_tick(prev):
        h = queue_html()
        return (gr.update(), prev) if h == prev else (h, h)
    import ticker
    if ticker.ACTIVE["on"]:                    # in LoRA Studio the page fetches the queue itself (livepoll.py)
        import livepoll
        livepoll.register("tr_queue", queue_html)
    else:
        ticker.timer(2.0, q_tick, last, [q_html, last], show_progress="hidden")
    ticker.timer(1.0, live_tick, seen, [live, head, stage, prog, loss, sp, log, seen], show_progress="hidden")
    return {"scripts": s_html, "queue": q_html, "show_done": show_done,
            "refresh": (lambda sd: (scripts_html(sd), queue_html()), [show_done], [s_html, q_html])}


def compare_html(st: dict) -> str:
    """The test pictures of a run: one column per save point, one row per prompt."""
    import comfytest as CT
    if not st or not st.get("log"):
        return ""
    info = script_info(st["script"]) if st.get("script") and Path(st["script"]).is_file() else None
    if not info:
        return '<p class="ls-empty">The launch script of this run is gone.</p>'
    fam = CT.family(info["tkey"])
    cfg = CT.settings()
    job = CT.JOBS.get(st["log"]) or {}
    head = ""
    if job.get("state") == "running":
        head = (f'<div class="ls-note">⏳ Making test pictures in ComfyUI … {_esc(job.get("text") or "")}'
                + (f' ({job["done"]} of {job["total"]})' if job.get("total") else "") + "</div>")
    elif job.get("text"):
        head = f'<div class="ls-note">{_esc(job["text"])}</div>'
    elif st["log"] in CT.PENDING:
        head = '<div class="ls-note">Waiting for ComfyUI: the test pictures are made as soon as it answers.</div>'
    if not fam:
        return head + '<p class="ls-empty">Test pictures are made for SDXL, Illustrious, Pony and Anima LoRAs.</p>'
    if not cfg["families"].get(fam):
        return head + (f'<p class="ls-empty">No ComfyUI workflow for {_esc(CT.FAMILIES[fam])} yet: add it in '
                       '<a href="#" onclick="lsSetNav(\'comfy\'); return false;">Settings › Test pictures</a>.</p>')
    grid = CT.results(info)
    if not grid:
        return head + '<p class="ls-empty">No test pictures yet. Press <b>Make test pictures</b> (ComfyUI must be running).</p>'
    labs = grid["labels"]
    best = st.get("best")
    cols = []
    for lab in labs:
        mark = " rs-cbest" if best is not None and lab == f"epoch {best}" else ""
        cols.append(f'<th class="rs-chead{mark}"><div>{_esc(lab)}' + (" · try first" if mark else "") + '</div>'
                    f'<button class="tr-btn tr-small" onclick="lsSend(\'rs_box\', \'cmpuse|{_esc(lab)}\')">Use this one'
                    '</button></th>')
    rows = []
    for i, pr in enumerate(grid.get("prompts") or []):
        cells = []
        for lab in labs:
            pics = (grid.get("pics") or {}).get(lab) or []
            f = pics[i] if i < len(pics) else None
            cells.append(f'<td>{"<img loading=lazy src=" + chr(39) + sg.DC.thumb_uri(Path(f), 320) + chr(39) + ">" if f else "<span class=tr-muted>—</span>"}</td>')
        tip = _esc(pr.get("text", "")[:300])
        rows.append(f'<tr><td class="rs-cprompt" title="{tip}">{i + 1}</td>{"".join(cells)}</tr>')
    return (head + f'<div class="rs-cwrap"><table class="rs-cgrid"><thead><tr><th></th>{"".join(cols)}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div><p class="ls-note">Made {_esc(grid.get("at", ""))} · the same '
            'prompt and seed in each row · hover a row number for its prompt. <b>Use this one</b> copies that save point '
            'to the folder above with its LoRA Manager data, and its first picture as the preview.</p>')


COPY_JS = """(t) => { try { navigator.clipboard.writeText(t); } catch (e) {
  var a = document.createElement('textarea'); a.value = t; document.body.appendChild(a); a.select();
  try { document.execCommand('copy'); } catch (e2) {} a.remove(); } }"""


# --------------------------------------------------------------------------- comparing runs
CMP_MAX = 4


def _ver_num(info) -> int | None:
    m = re.search(r"(\d+)", (info or {}).get("version") or "")
    return int(m.group(1)) if m else None


def is_best_run(info) -> bool:
    if not info or not sg.COMPLETED:
        return False
    v = _ver_num(info)
    return v is not None and sg.COMPLETED.best_version(info["dataset"], info["tkey"]) == v


def _lora_of(info):
    """The Collection's LoRA of a run (by its output name and dataset), or None before the Collection has seen it."""
    if not info or not info.get("out_name"):
        return None
    try:
        import lora_library as LL
        LL.LIB.ensure_loaded()
        for l in LL.LIB.loras.values():
            if l.stem == info["out_name"] and (l.dataset is None or l.dataset.name == info["dataset"]):
                return l
    except Exception:                                    # noqa: BLE001
        pass
    return None


def rating_of(info) -> int:
    l = _lora_of(info)
    if l is None:
        return 0
    import lora_library as LL
    return int(LL.LIB.overrides(l.key).get("rating") or 0)


def set_rating(st: dict, rate) -> str:
    """Your rating of the run's LoRA (the same one as on its LoRA page › Edit; Insights counts it)."""
    info = script_info(st["script"]) if st and st.get("script") and Path(st["script"]).is_file() else None
    l = _lora_of(info)
    if l is None:
        return ("The Collection does not know this LoRA yet: press Collection › Browse › *Rescan folders*, "
                "then rate it.")
    import lora_library as LL
    ov = LL.LIB.overrides(l.key)
    if rate:
        ov["rating"] = int(rate)
    else:
        ov.pop("rating", None)
    LL.LIB.save_data()
    try:
        import insights
        insights.forget()
    except Exception:                                    # noqa: BLE001
        pass
    return f"✔ Rated {rate} of 5: Insights counts it." if rate else "✔ Rating removed."


def toggle_best(logname: str) -> str:
    """Mark the version of a run as the best one of its dataset + LoRA type (again: remove the mark)."""
    f = sg.LOG_DIR / (logname or "")
    script = log_script(f) if logname and f.is_file() else ""
    info = script_info(script) if script and Path(script).is_file() else None
    if not info or not sg.COMPLETED:
        return "❌ The launch script of this run is gone."
    v = _ver_num(info)
    if v is None:
        return "❌ This run has no version number in its output name (V1, V2, …), so it cannot be marked."
    if is_best_run(info):
        sg.COMPLETED.set_best(info["dataset"], info["tkey"], None)
        return f"✔ {run_name(info)} is no longer marked as the best; the Collection shows the newest version again."
    sg.COMPLETED.set_best(info["dataset"], info["tkey"], v)
    try:
        import lora_library as LL
        LL.LIB.ensure_loaded()
    except Exception:                                    # noqa: BLE001
        pass
    return f"✔ ★ {run_name(info)} is marked as the best version: the Collection shows it first."


def _cmp_runs(logs: list[str]) -> list[dict]:
    hist = {r["log"]: r for r in runs()}
    out = []
    for ln in logs[:CMP_MAX]:
        r = hist.get(ln)
        if not r:
            continue
        info = r["info"]
        a = _advice(r["script"]) if r.get("script") and Path(r["script"]).is_file() else None
        out.append({**r, "name": run_name(info) if info else r["stem"], "advice": a})
    return out


def compare_plot(logs: list[str]):
    import pandas as pd
    rows = []
    for r in _cmp_runs(logs):
        for e, v in (r["advice"] or {}).get("points") or []:
            if isinstance(v, float) and math.isfinite(v):
                rows.append({"epoch": e, "loss": v, "run": r["name"]})
    return pd.DataFrame(rows or [], columns=["epoch", "loss", "run"])


def _short(v: str) -> str:
    v = (v or "").strip()
    return Path(v).name if "/" in v else v


def compare_head_html(logs: list[str]) -> str:
    rs = _cmp_runs(logs)
    if len(rs) < 2:
        return '<p class="ls-empty">Tick two to four runs in the list and press <b>Compare the ticked</b>.</p>'
    cards = []
    for r in rs:
        info = r["info"]
        best = is_best_run(info)
        a = r["advice"]
        cards.append(
            f'<div class="rs-ccard{" rs-ccard-best" if best else ""}"><div class="rs-cname">{_esc(r["name"])}'
            f'{" <span class=rs-star>★ best</span>" if best else ""}</div>'
            f'<div class="tr-muted">{_esc(r["started"])} · {_esc(r["dur"])} · {_esc(r["result"])}</div>'
            f'<div>{_esc(a["short"]) if a else "<span class=tr-muted>no loss log</span>"}</div>'
            f'<div class="rs-cbtns"><button class="tr-btn tr-small" onclick="lsSend(\'rs_box\', \'best|{_esc(r["log"])}\')">'
            f'{"Remove the best mark" if best else "★ Mark as the best version"}</button>'
            f'<button class="tr-btn tr-small" onclick="lsSend(\'rs_box\', \'open|{_esc(r["log"])}\')">Open</button>'
            '</div></div>')
    return f'<div class="rs-ccards">{"".join(cards)}</div>'


def compare_settings_html(logs: list[str], show_all: bool = False) -> str:
    rs = [r for r in _cmp_runs(logs) if r["info"]]
    if len(rs) < 2:
        return ""
    vals = []
    for r in rs:
        info = r["info"]
        d = {"LoRA type": info["type"]}
        try:
            st = sg.dataset_stats(info["dataset"])
            d["Pictures × repeats (now)"] = " + ".join(f'{x["n"]} × {x["repeats"]}' for x in st["rows"]) or "?"
        except Exception:                                # noqa: BLE001
            d["Pictures × repeats (now)"] = "?"
        fd = sg.form_dict(info["text"])
        for name, label in sg.FORM_FIELDS:
            if name in ("output_name", "logging_dir"):
                continue
            d[label] = _short(fd.get(name, ""))
        vals.append(d)
    keys = list(vals[0])
    for d in vals[1:]:
        keys += [k for k in d if k not in keys]
    diff = [k for k in keys if len({d.get(k, "") for d in vals}) > 1]
    same = [k for k in keys if k not in diff and any(d.get(k) for d in vals)]
    show = keys if show_all else diff
    show = [k for k in show if any(d.get(k) for d in vals)]
    head = "".join(f"<th>{_esc(r['name'])}</th>" for r in rs)
    body = []
    for k in show:
        cls = ' class="rs-diff"' if k in diff else ""
        body.append(f"<tr{cls}><td>{_esc(k)}</td>" + "".join(f"<td>{_esc(d.get(k, '') or '—')}</td>" for d in vals)
                    + "</tr>")
    if not body:
        return '<p class="ls-note">These runs have the same settings.</p>'
    note = (f"{len(diff)} setting(s) differ" + ("" if show_all else f"; {len(same)} are the same (tick “show all settings”)")
            + ". Pictures × repeats is the dataset as it is now.")
    return (f'<div class="rs-wrap"><table class="rs-table rs-ctable"><thead><tr><th>Setting</th>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div><p class="ls-note">{_esc(note)}</p>')


def _pick_label(grid: dict, a, which: str) -> str | None:
    labs = grid.get("labels") or []
    if which != "Final" and a and f"epoch {a['best']}" in labs:
        return f"epoch {a['best']}"
    if "final" in labs:
        return "final"
    return labs[-1] if labs else None


def compare_pics_html(logs: list[str], which: str = "Try first") -> str:
    import comfytest as CT
    rs = [r for r in _cmp_runs(logs) if r["info"]]
    if len(rs) < 2:
        return ""
    grids, heads, notes = [], [], []
    running = [r["name"] for r in rs if (CT.JOBS.get(r["log"]) or {}).get("state") == "running"]
    waiting = [r["name"] for r in rs if r["log"] in CT.PENDING]
    for r in rs:
        g = CT.results(r["info"]) or {}
        lab = _pick_label(g, r["advice"], which) if g else None
        grids.append((g, lab))
        heads.append(f'<th class="rs-chead"><div>{_esc(r["name"])}</div><div class="tr-muted">'
                     f'{_esc(lab or "no test pictures")}</div></th>')
    base = next((g.get("prompts") for g, _ in grids if g.get("prompts")), None)
    if not base:
        msg = '<p class="ls-empty">No test pictures yet. Press <b>Make test pictures with the same prompts</b>.</p>'
        return (_cmp_status(running, waiting) + msg)
    texts = [p.get("text", "") for p in base]
    rows, other = [], set()
    for i, txt in enumerate(texts):
        cells = []
        for (g, lab), r in zip(grids, rs):
            ptexts = [p.get("text", "") for p in g.get("prompts") or []]
            j = ptexts.index(txt) if txt in ptexts else None
            if g and j is None:
                other.add(r["name"])
            pics = (g.get("pics") or {}).get(lab) or [] if lab else []
            f = pics[j] if j is not None and j < len(pics) else None
            cells.append(f'<td>{"<img loading=lazy src=" + chr(39) + sg.DC.thumb_uri(Path(f), 320) + chr(39) + ">" if f else "<span class=tr-muted>—</span>"}</td>')
        rows.append(f'<tr><td class="rs-cprompt" title="{_esc(txt[:300])}">{i + 1}</td>{"".join(cells)}</tr>')
    if other:
        notes.append(f'{", ".join(sorted(other))}: made with other prompts. Press <b>Make test pictures with the same '
                     'prompts</b> to compare picture by picture.')
    return (_cmp_status(running, waiting)
            + "".join(f'<div class="ls-note">{n}</div>' for n in notes)
            + f'<div class="rs-cwrap"><table class="rs-cgrid"><thead><tr><th></th>{"".join(heads)}</tr></thead>'
              f'<tbody>{"".join(rows)}</tbody></table></div><p class="ls-note">One row per prompt (same prompt and seed '
              'in a row; hover a row number for its prompt). Each column is the chosen save point of that run.</p>')


def _cmp_status(running, waiting) -> str:
    out = ""
    if running:
        out += f'<div class="ls-note">⏳ Making test pictures in ComfyUI: {_esc(", ".join(running))}</div>'
    if waiting:
        out += f'<div class="ls-note">Waiting for their turn (or for ComfyUI): {_esc(", ".join(waiting))}</div>'
    return out


def build_results(app):
    with gr.Column(visible=True) as list_col:
        with gr.Row(elem_classes=["ls-bar"]):
            ds = gr.Dropdown(["All"] + sg.list_lora_dirs(), value="All", label="Dataset", scale=0, min_width=220)
            res = gr.Radio(["All", "Done", "Failed", "Stopped"], value="All", label="Result", scale=0, min_width=330)
            notc = gr.Checkbox(label="only not copied yet", value=False, scale=0, min_width=190)
            refresh = gr.Button("Refresh", size="sm", scale=0, min_width=100)
            cmp_go = gr.Button("Compare the ticked", variant="primary", size="sm", scale=0, min_width=180)
        cmp_sel = gr.Textbox(visible=False)
        cmp_msg0 = gr.Markdown()
        lst = gr.HTML(elem_id="rs_list")
    box = gr.Textbox(elem_id="rs_box", show_label=False, container=False)
    state = gr.State({})
    cmp_logs = gr.State([])
    with gr.Column(visible=False) as cmp_col:
        cmp_back = gr.Button("← All results", size="sm", scale=0, min_width=140)
        gr.Markdown("### Compare runs")
        cmp_head = gr.HTML(elem_id="rs_cmphead")
        cmp_bmsg = gr.Markdown()
        with gr.Row(equal_height=False):
            with gr.Column(scale=3, elem_classes=["ls-card"]):
                gr.Markdown("#### Loss per epoch")
                cmp_plot = gr.LinePlot(x="epoch", y="loss", color="run", height=260, show_label=False)
                gr.Markdown("Lower is not always better for image LoRAs; the shape matters more: a run that is still "
                            "falling could train longer, one that rises at the end trained too long.",
                            elem_classes=["ls-note"])
            with gr.Column(scale=4, elem_classes=["ls-card"]):
                gr.Markdown("#### Settings difference")
                cmp_set = gr.HTML(elem_id="rs_cmpset")
                cmp_all = gr.Checkbox(label="show all settings, also the ones that are the same", value=False)
        with gr.Column(elem_classes=["ls-card"]):
            gr.Markdown("#### Test pictures side by side")
            with gr.Row(elem_classes=["ls-bar"]):
                cmp_which = gr.Radio(["Try first", "Final"], value="Try first", label="Save point of each run",
                                     scale=0, min_width=260)
                cmp_same = gr.Button("Make test pictures with the same prompts", size="sm", scale=0, min_width=300)
                cmp_rf = gr.Button("Refresh", size="sm", scale=0, min_width=100)
            cmp_pics = gr.HTML(elem_id="rs_cmppics")
    with gr.Column(visible=False) as one:
        back = gr.Button("← All results", size="sm", scale=0, min_width=140)
        head = gr.HTML()
        with gr.Row(equal_height=False):
            with gr.Column(scale=2, elem_classes=["ls-card"]):
                tryf = gr.HTML()
                with gr.Row(elem_classes=["ls-bar"]):
                    c_best = gr.Button("Copy the one to try first", variant="primary", size="sm", scale=0, min_width=220)
                    c_final = gr.Button("Copy the final", size="sm", scale=0, min_width=140)
                dest = gr.Textbox(label="Copy to", info="The folder where older versions of this LoRA are; an epoch file is "
                                                         "copied with a name like mycharZITV1-e8.")
                over = gr.Checkbox(label="Replace a different file with the same name (the old one goes to the Trash)",
                                   value=False)
                cmsg = gr.Markdown()
            with gr.Column(scale=3, elem_classes=["ls-card"]):
                plot = gr.LinePlot(x="epoch", y="loss", label="Loss per epoch", height=240)
        gr.Markdown("#### Save points")
        sp = gr.HTML(elem_id="rs_sp")
        with gr.Column(elem_classes=["ls-card"]):
            gr.Markdown("#### Test pictures (ComfyUI)")
            with gr.Row(elem_classes=["ls-bar"]):
                cmp_make = gr.Button("Make test pictures", variant="primary", size="sm", scale=0, min_width=190)
                cmp_ref = gr.Button("Refresh", size="sm", scale=0, min_width=100)
            cmp_html = gr.HTML(elem_id="rs_cmp")
        with gr.Column(elem_classes=["ls-card"]):
            gr.Markdown("#### Clean up this run")
            clean_md = gr.Markdown()
            with gr.Row(elem_classes=["ls-bar"]):
                clean_ok = gr.Checkbox(label="Yes, move these to the Trash", value=False, scale=0, min_width=260)
                clean_btn = gr.Button("Move to the Trash", variant="stop", size="sm", scale=0, min_width=160)
            clean_msg = gr.Markdown()
        with gr.Column(elem_classes=["ls-card"]):
            gr.Markdown("#### This run")
            this = gr.HTML()
            with gr.Row(elem_classes=["ls-bar"]):
                newv = gr.Button("Train a new version…", size="sm", scale=0, min_width=190)
                import features as FT
                again = gr.Button("Run it again", size="sm", scale=0, min_width=130,
                                  visible=FT.have("sd_scripts") or FT.have("musubi"))
                best_b = gr.Button("★ Best version: mark / unmark", size="sm", scale=0, min_width=240)
                rate = gr.Radio([("–", 0), ("1", 1), ("2", 2), ("3", 3), ("4", 4), ("5", 5)], value=0, scale=0,
                                min_width=330, label="Your rating (4-5 good, 1-2 weak; Insights learns from it)")
                rep_b = gr.Button("Report for Claude", size="sm", scale=0, min_width=170)
            rep_note = gr.Textbox(label="What do you see? (optional, goes into the report)", lines=1,
                                  placeholder="for example: it stopped at epoch 3 / the pictures look burnt")
            with gr.Row(equal_height=False):
                rep_file = gr.File(label="The report", visible=False, interactive=False, scale=1)
                rep_msg = gr.Markdown()
            rep_text = gr.Textbox(visible=False)
            gr.Markdown("A new version opens Datasets › Edit existing › 9 Train for this dataset.",
                        elem_classes=["ls-note"])
            again_msg = gr.Markdown()
            with gr.Accordion("TensorBoard", open=False):
                with gr.Row():
                    tb_dir = gr.Textbox(label="Log folder", scale=4)
                    tb_port = gr.Number(label="Port", value=6006, precision=0, scale=1)
                with gr.Row():
                    tb_start = gr.Button("Start TensorBoard", size="sm")
                    tb_stop = gr.Button("Stop TensorBoard", size="sm")
                tb_md = gr.Markdown("TensorBoard: not running")
        with gr.Accordion("End of the run log", open=True):
            tail = gr.Textbox(lines=14, max_lines=14, show_label=False, interactive=False)

    filt = [ds, res, notc]
    for c in filt:
        c.change(results_html, filt, lst)
    refresh.click(lambda *a: (gr.update(choices=["All"] + sg.list_lora_dirs()), results_html(*a)), filt, [ds, lst])
    app.load(results_html, filt, lst)

    def do_open(v):
        act, _, rest = (v or "").partition("|")
        arg = rest.rsplit("|", 1)[0] if "|" in rest else rest
        if act != "open":
            return [gr.update()] * 16
        h, t, p, s, c, th, tl, d, st = open_run(arg)
        logdir = ""
        if st.get("script"):
            try:
                logdir = str(sg.script_paths(st["script"])[1] or "")
            except Exception:                            # noqa: BLE001
                logdir = ""
        info = script_info(st["script"]) if st.get("script") and Path(st["script"]).is_file() else None
        return (gr.update(visible=False), gr.update(visible=True), h, t, p, s, c, th, tl, d, st, logdir, "", "",
                gr.update(visible=False), rating_of(info))

    open_outs = [list_col, one, head, tryf, plot, sp, clean_md, this, tail, dest, state, tb_dir, cmsg, clean_msg, cmp_col,
                 rate]
    at_cmsg = open_outs.index(cmsg)

    def rs_click(v, st, d, o, logs):
        """Every click inside the Results lists, in ONE event (it was four, each costing the browser time):
        open a run, copy a save point, use a test picture, mark the best version."""
        act, _, rest = (v or "").partition("|")
        arg = rest.rsplit("|", 1)[0] if "|" in rest else rest
        out, extra = [gr.update()] * len(open_outs), [gr.update()] * 4      # + cmp_html, cmp_bmsg, cmp_head, again_msg
        if act == "open":
            out = list(do_open(v))
            extra[0] = compare_html(out[open_outs.index(state)])
        elif act == "copy":
            out[at_cmsg] = copy_file(st, arg, d, o)
        elif act == "cmpuse":
            import comfytest as CT
            out[at_cmsg] = CT.use(st.get("log"), arg, d, st, o)
            extra[0] = compare_html(st)
        elif act == "best":
            msg = toggle_best(arg)
            extra[1], extra[2], extra[3] = msg, (compare_head_html(logs) if logs else gr.update()), msg
        return out + extra
    box.change(rs_click, [box, state, dest, over, cmp_logs], open_outs + [cmp_html, cmp_bmsg, cmp_head, again_msg],
               show_progress="hidden")

    def cmp_start(st):
        import comfytest as CT
        if not st or not st.get("log"):
            return "Open a run first."
        return CT.make_in_background(st["log"])
    cmp_make.click(cmp_start, state, cmsg).then(compare_html, state, cmp_html)
    cmp_ref.click(compare_html, state, cmp_html)
    import ticker

    def cmp_tick(st):
        import comfytest as CT
        if st and st.get("log") and (CT.JOBS.get(st["log"]) or {}).get("state") == "running":
            return compare_html(st)
        return gr.update()
    ticker.timer(4.0, cmp_tick, state, cmp_html)
    back.click(lambda *a: (gr.update(visible=True), gr.update(visible=False), results_html(*a)), filt, [list_col, one, lst])

    # ---- comparing runs
    def cmp_open(sel, which, show_all):
        try:
            logs = [x for x in json.loads(sel or "[]") if x][:CMP_MAX]
        except ValueError:
            logs = []
        if len(logs) < 2:
            return ([gr.update()] * 7) + ["Tick two to four runs in the list first."]
        return (gr.update(visible=False), gr.update(visible=True), logs, compare_head_html(logs), compare_plot(logs),
                compare_settings_html(logs, show_all), compare_pics_html(logs, which), "")
    cmp_go.click(cmp_open, [cmp_sel, cmp_which, cmp_all],
                 [list_col, cmp_col, cmp_logs, cmp_head, cmp_plot, cmp_set, cmp_pics, cmp_msg0],
                 js="(x, w, a) => [JSON.stringify(Array.from(document.querySelectorAll('#rs_list .rs-tick:checked'))"
                    ".map(function(e){ return e.dataset.log; })), w, a]")
    cmp_back.click(lambda *a: (gr.update(visible=True), gr.update(visible=False), results_html(*a), ""), filt,
                   [list_col, cmp_col, lst, cmp_bmsg])
    cmp_all.change(compare_settings_html, [cmp_logs, cmp_all], cmp_set)
    cmp_which.change(compare_pics_html, [cmp_logs, cmp_which], cmp_pics)
    cmp_rf.click(compare_pics_html, [cmp_logs, cmp_which], cmp_pics)

    def cmp_same_fn(logs, which):
        import comfytest as CT
        return CT.share_prompts(logs or []), compare_pics_html(logs or [], which)
    cmp_same.click(cmp_same_fn, [cmp_logs, cmp_which], [cmp_bmsg, cmp_pics])


    def cmp_tick2(logs, which):
        import comfytest as CT
        if logs and any((CT.JOBS.get(ln) or {}).get("state") == "running" or ln in CT.PENDING for ln in logs):
            return compare_pics_html(logs, which)
        return gr.update()
    ticker.timer(5.0, cmp_tick2, [cmp_logs, cmp_which], cmp_pics)

    def reopen(st):
        if not st:
            return [gr.update()] * 4
        h, t, p, s, c, th, tl, d, _ = open_run(st["log"])
        return s, c, t, p
    c_best.click(copy_best, [state, dest, over], cmsg).then(reopen, state, [sp, clean_md, tryf, plot])
    c_final.click(copy_final, [state, dest, over], cmsg).then(reopen, state, [sp, clean_md, tryf, plot])
    clean_btn.click(cleanup, [state, clean_ok], clean_msg).then(lambda: False, None, clean_ok) \
        .then(reopen, state, [sp, clean_md, tryf, plot])
    again.click(run_again, state, again_msg)
    best_b.click(lambda st: toggle_best((st or {}).get("log", "")), state, again_msg)
    rate.input(set_rating, [state, rate], again_msg)

    def make_report(st, note):
        import report
        path, text = report.make((st or {}).get("log", ""), note or "")
        return (gr.update(value=str(path), visible=True), text,
                f"✔ Report made (`userdata/reports/{path.name}`, token and home folder removed) and copied: paste it into a "
                "chat with Claude, or drop the file there.")
    rep_b.click(make_report, [state, rep_note], [rep_file, rep_text, rep_msg]).then(
        None, rep_text, None, js=COPY_JS)
    tb_start.click(sg.tb_start, [tb_dir, tb_port], tb_md)
    tb_stop.click(sg.tb_stop_ui, None, tb_md)
    return {"newv": newv, "state": state, "list": lst, "refresh": (results_html, filt, lst)}
