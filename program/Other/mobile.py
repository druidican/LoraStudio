# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""The phone page's data (/m, UI/mobile_page.py): what is training, how far, time left, whether it failed (with the
last lines of its log), the queue and the newest sample picture. Small on purpose: the phone asks for it every few
seconds while a run is going and every minute when nothing runs."""
from __future__ import annotations
__version__ = "1.201126c"  # LoRA Studio version, see version.py

import re
import sys
import time
from pathlib import Path

PIC_EXT = (".png", ".jpg", ".jpeg", ".webp")
_SAMPLE = {"t": 0.0, "key": None, "file": None}


def _sg():
    return sys.modules.get("sdgui")


def _queue(sg) -> dict:
    items = list(sg.QUEUE["items"])
    waiting = [i for i in items if i.get("state") == "queued"]
    return {"waiting": len(waiting), "running": bool(sg.QUEUE["running"]), "stop_after": bool(sg.QUEUE["stop_flag"]),
            "next": [Path(i["path"]).stem for i in waiting[:6]]}


def newest_sample(script: str) -> Path | None:
    """The newest sample picture of a launch script (its model/sample folder), looked up at most every 10 s."""
    sg = _sg()
    if not script or sg is None:
        return None
    if _SAMPLE["key"] == script and time.time() - _SAMPLE["t"] < 10:
        return _SAMPLE["file"]
    best = None
    try:
        model_dir, _ = sg.script_paths(script)
        d = model_dir / "sample" if model_dir else None
        if d and d.is_dir():
            pics = [p for p in d.iterdir() if p.suffix.lower() in PIC_EXT]
            best = max(pics, key=lambda p: p.stat().st_mtime) if pics else None
    except OSError:
        best = None
    _SAMPLE.update(t=time.time(), key=script, file=best)
    return best


def status() -> dict:
    """Everything the phone page shows, as plain values."""
    sg = _sg()
    out = {"t": int(time.time()), "state": "idle", "name": "", "step": 0, "total": 0, "pct": 0.0, "eta": "",
           "elapsed": "", "epoch": 0, "epochs": 0, "loss": None, "nan": False, "stage": "", "line": "",
           "queue": {"waiting": 0, "running": False, "stop_after": False, "next": []}, "last": None, "error": [],
           "sample": None, "pics": ""}
    if sg is None:
        return out
    run = sg.RUN
    out["queue"] = _queue(sg)
    ct = sys.modules.get("comfytest")
    out["pics"] = ct.now_text() if ct is not None else ""
    if run.running():
        prog = sg.progress_text()
        m = re.search(r"(\d+)/(\d+)", prog)
        if m and int(m.group(2)):
            out["step"], out["total"] = int(m.group(1)), int(m.group(2))
            out["pct"] = round(100.0 * out["step"] / out["total"], 1)
        eta = re.search(r"<([\d:?]+)", prog)
        out["eta"] = eta.group(1) if eta and eta.group(1) != "?" else ""
        out.update(state="run", name=re.sub(r"\.sh$", "", run.name), epoch=run.epoch, epochs=run.epochs_total, nan=bool(run.nan_at),
                   elapsed=sg.fmt(time.time() - run.started) if run.started else "")
        if run.loss:
            out["loss"] = round(float(run.loss[-1][1]), 4) if run.loss[-1][1] == run.loss[-1][1] else None
        if run.stages:
            st = next((s for s in run.stages if s["state"] == "run"), None)
            if st:
                out["stage"] = f"{run.stages.index(st) + 1}/{len(run.stages)} {st['name']}"
        if not out["total"]:
            out["line"] = re.sub(r"^steps:\s*", "", prog)
    elif out["pics"]:
        out["state"] = "pics"
    sj = sys.modules.get("speedjobs")                    # Settings › Speed test
    if sj is not None and sj.JOB["running"] and not run.running():
        st = sj.status()
        if st.get("running"):
            out.update(state="speed", name="Speed test: " + st["type"], pct=st["pct"], eta=st["eta"],
                       step=st["step"], total=st["steps"], elapsed=st["elapsed"], line=st["line"][:160],
                       speed={"n": st["n"], "of": st["of"], "variant": st["variant"], "phase": st["phase"],
                              "s_per_step": st["s_per_step"], "gpu": st["gpu"],
                              "vram": round(st["vram"] / 2 ** 30, 1) if st["vram"] else None,
                              "types": f"{st['types_n']} of {st['types_of']}" if st["types_of"] > 1 else ""})
    import statusbar as SB
    if SB.DONE_EVENTS:
        ev = SB.DONE_EVENTS[-1]
        out["last"] = {"name": re.sub(r"\.sh$", "", ev["name"]), "ok": bool(ev["ok"]), "when": time.strftime("%H:%M", time.localtime(ev["t"])),
                       "day": time.strftime("%d %b", time.localtime(ev["t"])), "advice": ev.get("advice") or ""}
        if out["state"] == "idle" and SB.DONE_EVENTS and time.time() - SB.DONE_EVENTS[-1]["t"] < 12 * 3600:
            out["state"] = "done" if ev["ok"] else "fail"
    if out["state"] == "fail" and run.proc is not None and not run.running():
        with run.lock:
            tail = [l for l in run.lines[-60:] if l.strip()]
        keep = [l for l in tail if not re.search(r"steps:\s*\d+%\|", l)]
        out["error"] = [l[:300] for l in keep[-14:]]
    out["rate"] = to_rate()
    if run.script:
        p = newest_sample(run.script)
        if p is not None:
            try:
                out["sample"] = {"v": p.stat().st_mtime_ns, "name": p.name,
                                 "when": time.strftime("%H:%M", time.localtime(p.stat().st_mtime))}
            except OSError:
                pass
    return out


def sample_file() -> Path | None:
    sg = _sg()
    return newest_sample(sg.RUN.script) if sg is not None and sg.RUN.script else None


def stop_now() -> str:
    sg = _sg()
    sj = sys.modules.get("speedjobs")
    if sj is not None and sj.JOB["running"] and not (sg is not None and sg.RUN.running()):
        return sj.stop()
    if sg is None or not sg.RUN.running():
        return "Nothing is running."
    sg.stop_run()
    return "Stopping: Ctrl+C was sent to the training."


def stop_after() -> str:
    sg = _sg()
    if sg is None or not sg.QUEUE["running"]:
        return "The queue is not running."
    sg.queue_stop_after_current()
    return "The queue stops after the current run."


def start_queue() -> str:
    sg = _sg()
    if sg is None:
        return "Not ready."
    if sg.QUEUE["running"] or sg.RUN.running():
        return "Something is running already."
    if not any(i["state"] == "queued" for i in sg.QUEUE["items"]):
        return "No waiting items in the queue."
    sg.queue_start(sg.QUEUE.get("stop_on_fail", True))
    return "The queue is started."


def lan_address() -> str:
    """This computer's address on the home network (no traffic is sent), '' when there is none."""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            ip = s.getsockname()[0]
        return "" if ip.startswith("127.") else ip
    except OSError:
        return ""


def phone_text(v: dict) -> str:
    """Settings › Start and network: where the phone page is."""
    if not v.get("network"):
        return ("**Phone page:** tick *Reachable on the network* (with a login) and restart, then open "
                "`http://<this computer>:<port>/m` on your phone on the same Wi-Fi: the run, how far it is, time left, "
                "a failure with its error, the queue and the newest sample picture.")
    ip = lan_address() or "<this computer's address>"
    return (f"**Phone page:** open `http://{ip}:{int(v.get('port') or 7860)}/m` on your phone on the same Wi-Fi "
            "(the same login): the run, how far it is, time left, a failure with its error, the queue and the "
            "newest sample picture. Add it to the home screen to open it like an app.")


def to_rate() -> dict | None:
    """The newest comparison (Training/compare.py) with pictures not rated yet: {"id", "left", "rows", "name"}."""
    try:
        import compare as CMP
        for e in CMP.all_comparisons():
            rws = e.get("rows") or []
            left = sum(1 for r in rws if not r.get("pick"))
            if rws and left:
                return {"id": e["id"], "left": left, "rows": len(rws), "name": f"{e['ds']} · {e['tkey']}"}
    except Exception:                                    # noqa: BLE001
        pass
    return None


def rate_rows(cid: str) -> list[dict]:
    """The rows for the phone, without the names: [{"i", "prompt", "n", "pick": [index] | "none" | "same" | None}]."""
    import compare as CMP
    out = []
    for r in CMP.rows(cid):
        pk = r["pick"]
        if isinstance(pk, list):
            pk = [r["order"].index(k) for k in pk if k in r["order"]]
        out.append({"i": r["i"], "prompt": r["prompt"][:200], "n": len(r["order"]), "pick": pk})
    return out


def rate_pic(cid: str, i: int, n: int):
    import compare as CMP
    for r in CMP.rows(cid):
        if r["i"] == i and 0 <= n < len(r["pics"]):
            return Path(r["pics"][n])
    return None


def rate_pick(cid: str, i: int, pick) -> str:
    """pick: a list of picture indexes, "none", "same" or None."""
    import compare as CMP
    for r in CMP.rows(cid):
        if r["i"] == i:
            if isinstance(pick, list):
                pick = [r["order"][n] for n in pick if isinstance(n, int) and 0 <= n < len(r["order"])][:2] or None
            elif pick not in ("none", "same"):
                pick = None
            CMP.pick(cid, i, pick)
            return "saved"
    return "not found"
