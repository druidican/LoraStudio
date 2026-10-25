# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""The picture tools from LoRA Studio's side: quality rating, look-alikes, person boxes for smart crops, watermark
removal and upscaling. The work is done by pictools_worker.py inside the sd-scripts venv (its PyTorch runs on the GPU);
this module starts it when needed, talks to it, remembers the answers, and stops it before a training run.

Settings › Trainers and GPU shows whether the tools are ready; "Install the picture tools" adds the one small package
the venv lacks (spandrel, for the upscalers) without touching PyTorch.
"""
from __future__ import annotations
__version__ = "1.251026g"  # LoRA Studio version, see version.py

import base64
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
WORKER = APP_DIR / "pictools_worker.py"
MODELS_DIR = APP_DIR / "pictools_models"
CACHE_FILE = APP_DIR / "gui_cache" / "pictools_cache.json"
IDLE_SECS = 300
_P = {"proc": None, "last": 0.0, "lock": threading.Lock(), "status": None, "status_t": 0.0}
_CACHE = {"data": None}


def sg():
    return sys.modules.get("sdgui")


def python() -> Path | None:
    """The python of the sd-scripts venv (or the venv set in Settings › Trainers and GPU)."""
    S = sg()
    cands = []
    if S is not None:
        try:
            venv = (S.runner_settings()[1] or "").strip()
            if venv:
                v = Path(os.path.expanduser(venv))
                cands.append((v.parent.parent if v.name == "activate" else v) / "bin" / "python")
        except Exception:                                # noqa: BLE001
            pass
        cands += [S.sd_root() / "venv" / "bin" / "python", S.sd_root() / ".venv" / "bin" / "python"]
    for c in cands:
        if c.is_file():
            return c
    return None


def status(fresh: bool = False) -> dict:
    """{"python", "ok", "missing": [...], "text"}; cached for a minute."""
    if not fresh and _P["status"] and time.time() - _P["status_t"] < 60:
        return _P["status"]
    py = python()
    st = {"python": str(py) if py else "", "ok": False, "missing": [], "text": ""}
    if not py:
        st["text"] = "the sd-scripts venv was not found (Settings › Trainers and GPU)"
    else:
        code = ("import importlib.util as u, json; print(json.dumps([m for m in "
                "('torch','torchvision','transformers','spandrel','PIL','numpy') if u.find_spec(m) is None]))")
        try:
            r = subprocess.run([str(py), "-c", code], capture_output=True, text=True, timeout=60)
            st["missing"] = json.loads(r.stdout.strip() or "[]")
            st["ok"] = not st["missing"]
            st["text"] = "ready" if st["ok"] else "missing in the sd-scripts venv: " + ", ".join(st["missing"])
        except Exception as e:                           # noqa: BLE001
            st["text"] = f"could not check the venv: {e}"
    _P["status"], _P["status_t"] = st, time.time()
    return st


def ready() -> bool:
    return bool(status()["ok"])


def install() -> str:
    """Add spandrel (and einops) to the sd-scripts venv without touching PyTorch."""
    py = python()
    if not py:
        return "The sd-scripts venv was not found: set it in Settings › Trainers and GPU."
    st = status(fresh=True)
    heavy = [m for m in st["missing"] if m in ("torch", "torchvision", "transformers")]
    if heavy:
        return ("The sd-scripts venv lacks " + ", ".join(heavy) + ": that is a broken trainer install, which LoRA Studio "
                "does not change. Reinstall sd-scripts first (Settings › Updates).")
    r = subprocess.run([str(py), "-m", "pip", "install", "--no-deps", "spandrel", "einops"], capture_output=True,
                       text=True, timeout=900)
    st = status(fresh=True)
    if st["ok"]:
        return "✔ The picture tools are ready. Their model files (about 2.5 GB) download the first time they are used."
    return "❌ Could not install: " + (r.stderr or r.stdout)[-600:]


# --------------------------------------------------------------------------- the worker
def _start():
    py = python()
    if not py:
        raise RuntimeError("the sd-scripts venv was not found")
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    _P["proc"] = subprocess.Popen([str(py), str(WORKER), str(MODELS_DIR)], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env,
                                  start_new_session=True)
    threading.Thread(target=_idle_watch, daemon=True, name="pictools-idle").start()


def _idle_watch():
    while True:
        time.sleep(30)
        p = _P["proc"]
        if p is None or p.poll() is not None:
            return
        if time.time() - _P["last"] > IDLE_SECS and _P["lock"].acquire(blocking=False):
            try:
                stop_locked()
            finally:
                _P["lock"].release()
            return


def call(cmd: dict, timeout: float = 1800) -> dict:
    """Send one command; start the worker when needed. Raises RuntimeError with the worker's message."""
    S = sg()
    if S is not None and S.RUN.running():
        raise RuntimeError("a training run is using the GPU")
    with _P["lock"]:
        p = _P["proc"]
        if p is None or p.poll() is not None:
            _start()
            p = _P["proc"]
        _P["last"] = time.time()
        p.stdin.write(json.dumps(cmd) + "\n")
        p.stdin.flush()
        t0 = time.time()
        line = ""
        while time.time() - t0 < timeout:
            line = p.stdout.readline()
            if line or p.poll() is not None:
                break
        _P["last"] = time.time()
    if not line:
        stop()
        raise RuntimeError("the picture tools stopped answering")
    r = json.loads(line)
    if not r.get("ok"):
        raise RuntimeError(r.get("error") or "failed")
    return r


def stop_locked():
    p = _P["proc"]
    _P["proc"] = None
    if p is not None and p.poll() is None:
        try:
            p.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
            p.stdin.flush()
            p.wait(timeout=10)
        except Exception:                                # noqa: BLE001
            p.kill()


def stop():
    """Free the video memory (called before a training run)."""
    if _P["proc"] is None:
        return
    with _P["lock"]:
        stop_locked()


def ping() -> dict:
    return call({"cmd": "ping"}, timeout=300)


# --------------------------------------------------------------------------- analyze, with a cache per file
def _sig(p: Path) -> str:
    st = p.stat()
    return f"{p}|{st.st_size}|{int(st.st_mtime)}"


def _cache():
    if _CACHE["data"] is None:
        try:
            _CACHE["data"] = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:                                # noqa: BLE001
            _CACHE["data"] = {}
    return _CACHE["data"]


def _cache_save():
    try:
        CACHE_FILE.parent.mkdir(exist_ok=True)
        tmp = CACHE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(_CACHE["data"]), encoding="utf-8")
        tmp.replace(CACHE_FILE)
    except OSError:
        pass


def analyze(paths, say=None, chunk: int = 8) -> dict:
    """{path: {"aes", "kind", "persons", "emb"}} (emb = list of floats); remembered by file size and time."""
    c = _cache()
    out, todo = {}, []
    for p in map(Path, paths):
        try:
            k = _sig(p)
        except OSError:
            continue
        if k in c:
            out[str(p)] = c[k]
        else:
            todo.append(p)
    for i in range(0, len(todo), chunk):
        part = todo[i:i + chunk]
        if say:
            say(f"{i + len(part)} of {len(todo)}")
        r = call({"cmd": "analyze", "paths": [str(p) for p in part]})["result"]
        for p in part:
            e = r.get(str(p))
            if e and "error" not in e:
                c[_sig(p)] = e
                out[str(p)] = e
        _cache_save()
    return out


def cached(p) -> dict | None:
    """What the tools already know about a picture, without starting them."""
    try:
        return _cache().get(_sig(Path(p)))
    except OSError:
        return None


def emb(e) -> list:
    import array
    a = array.array("e")
    a.frombytes(base64.b64decode(e["emb"]))
    return list(a)


def cos(a, b) -> float:
    return sum(x * y for x, y in zip(a, b))


def lookalikes(items: dict, limit: float = 0.95) -> list[tuple[str, str, float]]:
    """[(path a, path b, similarity)] of pictures that show the same thing (CLIP embeddings)."""
    keys = list(items)
    vecs = {k: emb(items[k]) for k in keys}
    try:
        import numpy as np
        m = np.array([vecs[k] for k in keys], dtype="float32")
        sim = m @ m.T
        out = []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                if sim[i, j] >= limit:
                    out.append((keys[i], keys[j], float(sim[i, j])))
        return out
    except ImportError:
        return [(keys[i], keys[j], cos(vecs[keys[i]], vecs[keys[j]])) for i in range(len(keys))
                for j in range(i + 1, len(keys)) if cos(vecs[keys[i]], vecs[keys[j]]) >= limit]


def smart_box(e: dict, w: int, h: int, aspect: float | None = None) -> list[float] | None:
    """A crop box (fractions) around the main person, with room above the head, in the picture's shape (or `aspect`).
    None when the person finder is unsure, or when the person fills the picture anyway (nothing to crop)."""
    ps = [p for p in (e or {}).get("persons") or [] if len(p) >= 5 and p[4] >= 0.7]
    if not ps:
        return None
    area = lambda p: max(0.0, p[2] - p[0]) * max(0.0, p[3] - p[1])
    main = max(ps, key=lambda p: p[4] * area(p))
    if area(main) < 0.06 * w * h:
        return None                                       # a small figure: probably not the subject, or a wrong find
    x1, y1, x2, y2 = main[:4]
    for p in ps:                                          # a second person next to the main one, about as big
        if p is main or area(p) < 0.4 * area(main):
            continue
        near = not (p[0] > x2 + 0.1 * w or p[2] < x1 - 0.1 * w or p[1] > y2 or p[3] < y1)
        if near:
            x1, y1, x2, y2 = min(x1, p[0]), min(y1, p[1]), max(x2, p[2]), max(y2, p[3])
    pw, ph = x2 - x1, y2 - y1
    x1, x2 = x1 - 0.10 * pw, x2 + 0.10 * pw               # room around the person, most of it above the head
    y1, y2 = y1 - 0.12 * ph, y2 + 0.05 * ph
    x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2)
    aspect = aspect or (w / h)
    bw, bh = x2 - x1, y2 - y1
    if bw / bh < aspect:                                  # too narrow: widen, centred on the person, shifted inside
        nb = min(float(w), bh * aspect)
        cx = (x1 + x2) / 2
        x1 = min(max(0.0, cx - nb / 2), w - nb)
        x2 = x1 + nb
        if nb >= w and bh * aspect > w:                   # cannot get wide enough: lower the height instead
            nh = w / aspect
            y2 = min(float(h), y1 + nh)
    else:                                                 # too wide: make it taller, keeping the head in
        nb = min(float(h), bw / aspect)
        y1 = min(max(0.0, y1 - (nb - bh) * 0.25), h - nb)  # most of the extra height goes below
        y2 = y1 + nb
        if nb >= h and bw / aspect > h:
            nw = h * aspect
            cx = (x1 + x2) / 2
            x1 = min(max(0.0, cx - nw / 2), w - nw)
            x2 = x1 + nw
    box = [round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4)]
    if (box[2] - box[0]) * (box[3] - box[1]) > 0.92:
        return None                                       # the person fills the picture: nothing to crop
    return box


def inpaint(src, dst, boxes, mask=None) -> dict:
    return call({"cmd": "inpaint", "src": str(src), "dst": str(dst), "boxes": boxes,
                 "mask": str(mask) if mask else None}, timeout=900)


def upscale(src, dst, target_short: int, kind: str = "photo") -> dict:
    return call({"cmd": "upscale", "src": str(src), "dst": str(dst), "target_short": int(target_short),
                 "kind": kind}, timeout=1800)


# --------------------------------------------------------------------------- watermark cut-outs
def keep_cutout(root: Path, rp: str, orig: Path, marks: list) -> None:
    """Save each removed watermark once per different mark in <dataset>/_originals/_watermarks/, with the pictures
    it was on (watermarks.json). The same mark again keeps the clearest (largest) cut-out."""
    from PIL import Image, ImageOps
    d = Path(root) / "_originals" / "_watermarks"
    d.mkdir(parents=True, exist_ok=True)
    index_file = d / "watermarks.json"
    try:
        index = json.loads(index_file.read_text(encoding="utf-8"))
    except Exception:                                    # noqa: BLE001
        index = {}
    try:
        with Image.open(orig) as im:
            im = ImageOps.exif_transpose(im).convert("RGBA")
            for m in marks:
                b = [int(v) for v in m["box"]]
                cut = im.crop(tuple(b))
                if cut.size[0] < 4 or cut.size[1] < 4:
                    continue
                key = _mark_key(m, cut, index)
                e = index.setdefault(key, {"file": f"{key}.png", "kind": m.get("kind", ""), "text": m.get("text", ""),
                                           "pictures": [], "area": 0})
                if rp not in e["pictures"]:
                    e["pictures"].append(rp)
                area = cut.size[0] * cut.size[1]
                if area > e.get("area", 0):
                    cut.save(d / e["file"])
                    e["area"] = area
                    e["size"] = list(cut.size)
                    e["hash"] = _dhash(cut)
    except OSError:
        return
    index_file.write_text(json.dumps(index, indent=1), encoding="utf-8")


def _dhash(im) -> int:
    g = im.convert("L").resize((9, 8))
    px = list(g.tobytes())
    return sum(1 << i for i in range(64) if px[(i // 8) * 9 + i % 8] > px[(i // 8) * 9 + i % 8 + 1])


def _mark_key(m, cut, index) -> str:
    import re
    text = re.sub(r"[^a-z0-9]+", "", (m.get("text") or "").lower())
    if text:
        return f"wm_{text[:40]}"
    h = _dhash(cut)
    ar = cut.size[0] / max(1, cut.size[1])
    for k, e in index.items():
        if e.get("text") or "hash" not in e:
            continue
        ear = e["size"][0] / max(1, e["size"][1])
        if bin(h ^ e["hash"]).count("1") <= 12 and 0.7 < ar / ear < 1.4:
            return k
    n = 1
    while f"wm_{m.get('kind', 'mark')}_{n}" in index:
        n += 1
    return f"wm_{(m.get('kind') or 'mark').replace(' ', '_')}_{n}"
