# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Ollama models: which are installed, which one is in video memory, start / stop / download / remove.

Only one model runs at a time: starting one stops the others first. A training run stops every model to free
video memory and does not start one again afterwards (the top bar shows "Ollama: none" until you start one, or
until a page that needs Ollama asks it a question).
"""
from __future__ import annotations
__version__ = "1.251026o"  # LoRA Studio version, see version.py

import html
import json
import sys
import threading
import time
import urllib.error
import urllib.request

DEFAULT_URL = "http://127.0.0.1:11434"
_PS = {"t": 0.0, "url": "", "data": None}
_LOCK = threading.Lock()
PULL = {"name": "", "status": "", "done": True, "error": ""}


def settings() -> tuple[str, str]:
    """(url, model chosen for pictures) from Settings › AI helpers."""
    sg = sys.modules.get("sdgui")
    if sg is not None:
        try:
            return sg.oc_settings()
        except Exception:                                # noqa: BLE001
            pass
    return DEFAULT_URL, ""


def _call(url: str, path: str, payload=None, method: str | None = None, timeout: float = 10.0) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url.rstrip("/") + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode("utf-8", "replace")
    try:
        return json.loads(body or "{}")
    except ValueError:
        return {}


def running(url: str | None = None, fresh: bool = False):
    """[{'name', 'vram'}] of the models in memory now, or None when Ollama cannot be reached. Cached 3 s."""
    url = url or settings()[0]
    with _LOCK:
        if not fresh and _PS["url"] == url and time.time() - _PS["t"] < 3.0:
            return _PS["data"]
    try:
        ms = _call(url, "/api/ps", timeout=2.5).get("models", [])
        data = [{"name": m.get("name") or m.get("model") or "", "vram": m.get("size_vram") or m.get("size") or 0}
                for m in ms]
    except Exception:                                    # noqa: BLE001
        data = None
    with _LOCK:
        _PS.update(t=time.time(), url=url, data=data)
    return data


def installed(url: str | None = None) -> list[dict]:
    """[{'name', 'vision', 'size_gb'}] (vision models first). Raises when Ollama cannot be reached."""
    url = url or settings()[0]
    try:
        import captioner
        return captioner.list_models(url)
    except ImportError:
        out = []
        for m in _call(url, "/api/tags").get("models", []):
            out.append({"name": m.get("name", ""), "vision": None, "size_gb": round((m.get("size") or 0) / 1e9, 1)})
        return out


def _forget():
    with _LOCK:
        _PS["t"] = 0.0


def stop(name: str, url: str | None = None) -> str:
    url = url or settings()[0]
    try:
        _call(url, "/api/generate", {"model": name, "keep_alive": 0}, timeout=60)
    except Exception as e:                               # noqa: BLE001
        return f"Could not stop {name}: {e}"
    _forget()
    return f"Stopped {name}."


def stop_all(url: str | None = None) -> str:
    """Unload every model (used when a training run starts)."""
    url = url or settings()[0]
    now = running(url, fresh=True)
    if not now:
        return "No model was running." if now is not None else "Ollama is not reachable."
    for m in now:
        stop(m["name"], url)
    _forget()
    return "Stopped " + ", ".join(m["name"] for m in now) + "."


def start(name: str, url: str | None = None) -> str:
    """Stop the other models, then load this one and keep it until it is stopped."""
    url = url or settings()[0]
    for m in running(url, fresh=True) or []:
        if m["name"] != name:
            stop(m["name"], url)
    try:
        _call(url, "/api/generate", {"model": name, "keep_alive": -1}, timeout=300)
    except Exception as e:                               # noqa: BLE001
        return f"Could not start {name}: {e}"
    _forget()
    return f"Started {name}."


def delete(name: str, url: str | None = None) -> str:
    url = url or settings()[0]
    try:
        stop(name, url)
        _call(url, "/api/delete", {"model": name}, method="DELETE", timeout=60)
    except Exception as e:                               # noqa: BLE001
        return f"Could not remove {name}: {e}"
    return f"Removed {name}."


def pull(name: str, url: str | None = None):
    """Download a model in the background; PULL holds the progress."""
    url = url or settings()[0]
    name = (name or "").strip()
    if not name:
        return "Type a model name, for example minicpm-v:8b."
    if not PULL["done"]:
        return f"Already downloading {PULL['name']}."
    PULL.update(name=name, status="starting", done=False, error="")

    def work():
        try:
            req = urllib.request.Request(url.rstrip("/") + "/api/pull", data=json.dumps({"model": name}).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=3600) as r:
                for line in r:
                    try:
                        d = json.loads(line)
                    except ValueError:
                        continue
                    if d.get("error"):
                        PULL["error"] = d["error"]
                        break
                    st = d.get("status", "")
                    if d.get("total"):
                        st += f" {100 * (d.get('completed') or 0) / d['total']:.0f}%"
                    PULL["status"] = st
        except Exception as e:                           # noqa: BLE001
            PULL["error"] = str(e)
        PULL["done"] = True
    threading.Thread(target=work, daemon=True, name="ollama-pull").start()
    return f"Downloading {name} …"


def pull_text() -> str:
    if not PULL["name"]:
        return ""
    if PULL["error"]:
        return f"✕ {PULL['name']}: {PULL['error']}"
    if PULL["done"]:
        return f"✓ {PULL['name']} downloaded."
    return f"Downloading {PULL['name']}: {PULL['status']}"


def indicator_html() -> str:
    """The Ollama part of the top bar: '● Ollama: name' or '○ Ollama: none'."""
    try:
        import features as FT
        if not FT.have("ollama"):
            return '<span class="ls-oll ls-oll-off" title="Ollama is not installed">– Ollama missing</span>'
    except ImportError:
        pass
    now = running()
    if now is None:
        return '<span class="ls-oll ls-oll-off" title="Ollama is not reachable">○ Ollama: off</span>'
    if not now:
        return '<span class="ls-oll ls-oll-off" title="No Ollama model in memory">○ Ollama: none</span>'
    names = ", ".join(m["name"] for m in now)
    gb = sum(m["vram"] for m in now) / 1024 ** 3
    return (f'<span class="ls-oll ls-oll-on" title="In video memory: {gb:.1f} GB">● Ollama: '
            f'{html.escape(names)}</span>')
