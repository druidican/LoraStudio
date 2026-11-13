# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""The status strip at the top of every page: the training run, and the machine (GPU, VRAM, CPU, RAM).

GPU: amd-smi or rocm-smi (AMD, see gpusmi.py) or nvidia-smi (NVIDIA); CPU and RAM: /proc/stat and /proc/meminfo. Everything is cached for
two seconds, so many open pages share one reading. A missing tool just leaves that part out.
"""
from __future__ import annotations
__version__ = "1.131126c"  # LoRA Studio version, see version.py

import html
import re
import shutil
import subprocess
import sys
import threading
import time

_CACHE = {"t": 0.0, "data": None}
_LOCK = threading.Lock()
_CPU_LAST = {"idle": None, "total": None}
DONE_EVENTS: list = []          # [{"t", "name", "ok", "advice"}] runs that finished, newest last


def _cpu_percent():
    try:
        with open("/proc/stat") as f:
            parts = [float(x) for x in f.readline().split()[1:]]
    except OSError:
        return None
    idle, total = parts[3] + (parts[4] if len(parts) > 4 else 0), sum(parts)
    li, lt = _CPU_LAST["idle"], _CPU_LAST["total"]
    _CPU_LAST["idle"], _CPU_LAST["total"] = idle, total
    if li is None or total == lt:
        return None
    return max(0.0, min(100.0, 100.0 * (1 - (idle - li) / (total - lt))))


def _ram():
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, _, v = line.partition(":")
                info[k] = float(v.split()[0]) * 1024
        tot = info["MemTotal"]
        return tot - info.get("MemAvailable", info.get("MemFree", 0)), tot
    except (OSError, KeyError, ValueError):
        return None


def _gpu():
    """[{name, load, temp, vram_used, vram_total}] in %, °C and bytes."""
    try:                                                   # AMD: amd-smi (ROCm 10+) or rocm-smi, values in bytes
        import gpusmi
        res = [{"name": str(c["card"]).replace("card", "GPU "), "load": c.get("load"),
                "temp": c.get("junction") if c.get("temp") is None else c.get("temp"),
                "vram_used": c.get("vram_used") or 0, "vram_total": c.get("vram_total") or 0}
               for c in gpusmi.cards()]
        if res:
            return res
    except Exception:                                      # noqa: BLE001
        pass
    exe = shutil.which("nvidia-smi")
    if exe:
        try:
            r = subprocess.run([exe, "--query-gpu=name,utilization.gpu,temperature.gpu,memory.used,memory.total",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=6)
            res = []
            for line in r.stdout.strip().splitlines():
                n, load, temp, used, tot = [x.strip() for x in line.split(",")]
                res.append({"name": n, "load": float(load), "temp": float(temp),
                            "vram_used": float(used) * 1024 ** 2, "vram_total": float(tot) * 1024 ** 2})
            return res
        except Exception:
            pass
    return []


def machine() -> dict:
    with _LOCK:
        if _CACHE["data"] is not None and time.time() - _CACHE["t"] < 2.0:
            return _CACHE["data"]
        data = {"gpus": _gpu(), "cpu": _cpu_percent(), "ram": _ram()}
        _CACHE["t"], _CACHE["data"] = time.time(), data
        return data


def _meter(label: str, frac, text: str) -> str:
    frac = 0.0 if frac is None else max(0.0, min(1.0, frac))
    hot = " ls-hot" if frac >= 0.85 else ""
    return (f'<span class="ls-meter{hot}" title="{html.escape(label)}"><span class="ls-mlab">{html.escape(text)}</span>'
            f'<span class="ls-mbar"><span style="width:{frac * 100:.0f}%"></span></span></span>')


def machine_html() -> str:
    m = machine()
    parts = []
    gb = 1024 ** 3
    for g in m["gpus"][:2]:
        if g.get("load") is not None:
            t = f" · {g['temp']:.0f} °C" if g.get("temp") is not None else ""
            parts.append(_meter(f"{g['name']} load", g["load"] / 100, f"GPU {g['load']:.0f}%{t}"))
        if g.get("vram_total"):
            parts.append(_meter("VRAM", g["vram_used"] / g["vram_total"],
                                f"VRAM {g['vram_used'] / gb:.1f}/{g['vram_total'] / gb:.0f} GB"))
    if m["cpu"] is not None:
        parts.append(_meter("CPU", m["cpu"] / 100, f"CPU {m['cpu']:.0f}%"))
    if m["ram"]:
        used, tot = m["ram"]
        parts.append(_meter("RAM", used / tot, f"RAM {used / gb:.1f}/{tot / gb:.0f} GB"))
    if not m["gpus"]:
        parts.insert(0, '<span class="ls-mnote">GPU: no amd-smi / rocm-smi / nvidia-smi</span>')
    return "".join(parts)


def _waiting(sg) -> int:
    return sum(1 for i in sg.QUEUE["items"] if i["state"] == "queued")


def run_html() -> tuple[str, str]:
    """(line, state): state = run | idle | done | fail."""
    sg = sys.modules.get("sdgui")
    if sg is None:
        return "", "idle"
    run = sg.RUN
    if run.running():
        prog = sg.progress_text()
        m = re.search(r"(\d+)/(\d+)", prog)
        eta = re.search(r"<([\d:]+)", prog)
        step = f" · step {int(m.group(1)):,}/{int(m.group(2)):,}" if m else ""
        loss = f" · loss {run.loss[-1][1]:.3f}" if run.loss else ""
        left = f" · {eta.group(1)} left" if eta else ""
        nan = " · ⚠ loss is NaN" if run.nan_at else ""
        waiting = _waiting(sg)
        more = f" · {waiting} more in the queue" if waiting else ""
        return f"● Training {html.escape(run.name)}{step}{loss}{left}{nan}{more}", "run"
    bits, state = [], "idle"
    recent = DONE_EVENTS and time.time() - DONE_EVENTS[-1]["t"] < 12 * 3600
    if recent:
        ev = DONE_EVENTS[-1]
        state = "done" if ev["ok"] else "fail"
        when = time.strftime("%H:%M", time.localtime(ev["t"]))
        if ev["ok"]:
            bits.append(f"✓ {html.escape(ev['name'])} finished {when}" + (f" · {html.escape(ev['advice'])}" if ev.get("advice") else ""))
        else:
            bits.append(f"✕ {html.escape(ev['name'])} failed {when}" + (f": {html.escape(ev['advice'])}" if ev.get("advice") else ""))
    waiting = _waiting(sg)
    bits.append(f"queue: {waiting} waiting" if waiting else "queue empty")
    return ("○ Idle · " if state == "idle" else "") + " · ".join(bits), state


def ollama_html() -> str:
    try:
        import ollamactl
        return ollamactl.indicator_html()
    except Exception:                                    # noqa: BLE001
        return ""
