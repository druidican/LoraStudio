# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Picture index for the Gallery tab: which of your generated pictures were made with which LoRA.

ComfyUI pictures are read through their metadata, never by looking at the picture:
  * a sidecar .json with the same file name (ComfyUI_00012_.png + ComfyUI_00012_.json, or .png.json)
  * otherwise the PNG's own text chunks ("prompt" / "workflow" from ComfyUI, "parameters" A1111-style)
Pictures without either are skipped.

Understood formats: the ComfyUI API prompt (what actually ran: bypassed nodes are not in it), the UI workflow
(bypassed / muted nodes are skipped), LoRA loader nodes (core LoraLoader, rgthree Power Lora Loader, LoRA
Manager's loader, stacks), <lora:name:0.8> tags and "Lora hashes" lines (Image Saver, A1111 style).

The index is kept in picture_index.json next to the program. A scan only reads pictures that are new or
changed since the last scan (file time + size of the picture and its .json); deleted pictures are dropped.
"""
from __future__ import annotations
__version__ = "1.011126"  # LoRA Studio version, see version.py

import json
import os
import re
import threading
import time
from pathlib import Path

import lspaths as _LP  # noqa: E402  (program/lspaths.py: ROOT, PROG, USER)
APP_DIR = _LP.USER
INDEX_FILE = APP_DIR / "picture_index.json"
INDEX_VERSION = 3          # 3: + prompt/seed through UI workflow links, LoRA hashes from the PNG
PIC_EXT = {".png", ".jpg", ".jpeg", ".webp"}
LORA_EXT = (".safetensors", ".pt", ".ckpt", ".bin", ".pth")
SKIP_DIRS = {".git", "__pycache__", ".thumbs", "_set_aside", ".Trash-1000"}
MAX_PROMPT = 1500
LOCK = threading.RLock()

TAG_RE = re.compile(r"<lora:([^:>]+)(?::\s*([-+]?\d*\.?\d+))?[^>]*>", re.I)
HASHES_RE = re.compile(r'Lora hashes:\s*"([^"]*)"', re.I)
BYPASSED = (2, 4)          # ComfyUI node mode: 2 = muted, 4 = bypassed


# --------------------------------------------------------------------------- #
# Reading one picture's metadata
# --------------------------------------------------------------------------- #

def norm(name: str) -> str:
    """'Anima/mycharAnimaV2.safetensors' -> 'mycharanimav2'."""
    base = re.split(r"[\\/]", str(name).strip())[-1].strip()
    low = base.lower()
    for ext in LORA_EXT:
        if low.endswith(ext):
            return low[: -len(ext)].strip()
    return low


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _is_file(v) -> bool:
    return isinstance(v, str) and v.strip().lower().endswith(LORA_EXT)


def _is_link(v) -> bool:
    """[node_id, output_index] in the API format."""
    return isinstance(v, list) and len(v) == 2 and isinstance(v[0], (str, int)) and isinstance(v[1], int)


class _Found:
    def __init__(self):
        self.loras: dict[str, float | None] = {}
        self.hashes: dict[str, str] = {}     # short hash -> name
        self.ckpt = ""
        self.prompts: list[str] = []
        self.positive = ""
        self.seed = None

    def add(self, name, strength=None):
        if not isinstance(name, str) or not name.strip() or name.strip().lower() in ("none", "null"):
            return
        s = _num(strength)
        if s is not None and s == 0:
            return                            # strength 0 = switched off
        n = norm(name)
        if n and (n not in self.loras or self.loras[n] is None):
            self.loras[n] = s

    def text(self, s: str):
        if not isinstance(s, str):
            return
        for m in TAG_RE.finditer(s):
            self.add(m.group(1), m.group(2))
        for m in HASHES_RE.finditer(s):
            for part in m.group(1).split(","):
                if ":" in part:
                    name, h = part.rsplit(":", 1)
                    self.hashes[h.strip().lower()] = norm(name)
                    self.add(name)


def _lora_dict(d: dict, f: _Found):
    """{"on": true, "lora": "x.safetensors", "strength": 1} (rgthree) or {"name": "x", "strength": 1,
    "active": true} (LoRA Manager)."""
    if d.get("on") is False or d.get("active") is False or d.get("enabled") is False:
        return
    name = d.get("lora") or d.get("lora_name") or d.get("name") or d.get("file")
    if isinstance(name, str):
        f.add(name, d.get("strength", d.get("strength_model", d.get("weight"))))


def _api(prompt: dict, f: _Found):
    nodes = {str(k): v for k, v in prompt.items() if isinstance(v, dict) and isinstance(v.get("inputs"), dict)}
    for nid, node in nodes.items():
        ct = str(node.get("class_type", ""))
        is_lora = "lora" in ct.lower()
        inp = node["inputs"]
        for k, v in inp.items():
            kl = k.lower()
            if isinstance(v, str):
                if is_lora and ("lora" in kl or kl in ("name", "file")) and (_is_file(v) or "lora" in kl):
                    suffix = kl.split("lora_name")[-1] if "lora_name" in kl else ""
                    strength = next((inp[k2] for k2 in (f"strength_model{suffix}", f"model_weight{suffix}",
                                                         f"model_str{suffix}", f"lora_wt{suffix}", f"strength{suffix}",
                                                         "strength_model", "strength") if k2 in inp), None)
                    f.add(v, strength if not _is_link(strength) else None)
                elif "lora" in kl and _is_file(v):
                    f.add(v)
                elif not is_lora and kl in ("ckpt_name", "unet_name", "model_name") and not f.ckpt:
                    f.ckpt = v
                if "cliptextencode" in ct.lower().replace(" ", "") and kl in ("text", "text_g", "prompt"):
                    f.prompts.append(v)
                f.text(v)
            elif isinstance(v, dict) and is_lora:
                _lora_dict(v, f)
            elif isinstance(v, list) and not _is_link(v):
                for item in v:
                    if isinstance(item, dict) and ("lora" in kl or is_lora):
                        _lora_dict(item, f)
            elif kl in ("seed", "noise_seed") and isinstance(v, int) and f.seed is None:
                f.seed = v
    # the positive prompt: follow the sampler's "positive" link to its text node
    for node in nodes.values():
        pos = node["inputs"].get("positive")
        if _is_link(pos):
            src = nodes.get(str(pos[0]))
            if src:
                t = next((src["inputs"].get(k) for k in ("text", "text_g", "prompt")
                          if isinstance(src["inputs"].get(k), str)), "")
                if t:
                    f.positive = t
                    break


def _workflow(wf: dict, f: _Found):
    for node in wf.get("nodes") or []:
        if not isinstance(node, dict) or node.get("mode") in BYPASSED:
            continue
        t = str(node.get("type", ""))
        wv = node.get("widgets_values")
        vals = list(wv.values()) if isinstance(wv, dict) else (wv if isinstance(wv, list) else [])
        is_lora = "lora" in t.lower()
        for i, v in enumerate(vals):
            if isinstance(v, str):
                if is_lora and _is_file(v):
                    nxt = vals[i + 1] if i + 1 < len(vals) else None
                    f.add(v, nxt)
                elif not is_lora and _is_file(v) and not f.ckpt and ("checkpoint" in t.lower() or "unet" in t.lower()):
                    f.ckpt = v
                if "cliptextencode" in t.lower().replace(" ", "") and len(v) > 3:
                    f.prompts.append(v)
                f.text(v)
            elif isinstance(v, dict) and is_lora:
                _lora_dict(v, f)
            elif isinstance(v, list) and is_lora:
                for item in v:
                    if isinstance(item, dict):
                        _lora_dict(item, f)
    for sg in (wf.get("definitions") or {}).get("subgraphs") or []:
        if isinstance(sg, dict):
            _workflow(sg, f)
    try:
        _workflow_prompt_seed(wf, f)
    except Exception:
        pass


_TEXT_IN = ("text", "positive", "prompt", "part", "string", "text_g", "text_l")
_TEXT_KEYS = ("positive", "text", "text_g", "prompt", "string", "value", "text_0")


def _workflow_prompt_seed(wf: dict, f: _Found):
    """UI workflow: follow the links from the saver / sampler's "positive" input back to the text nodes
    (e.g. easy positive -> CR Combine Prompt -> CLIPTextEncode -> Image Saver), and its seed input."""
    nodes = {n.get("id"): n for n in wf.get("nodes") or [] if isinstance(n, dict)}
    links = {}
    for l in wf.get("links") or []:
        if isinstance(l, list) and len(l) >= 3:
            links[l[0]] = l[1]
        elif isinstance(l, dict) and "id" in l:
            links[l["id"]] = l.get("origin_id")

    def widgets(n):
        named = n.get("widgets_values_named")
        if isinstance(named, dict):
            return named
        wv = n.get("widgets_values")
        return dict(wv) if isinstance(wv, dict) else {}

    def text_of(nid, depth=0, seen=None):
        seen = seen if seen is not None else set()
        n = nodes.get(nid)
        if not n or depth > 12 or nid in seen or n.get("mode") in BYPASSED:
            return ""
        seen.add(nid)
        parts = []
        for inp in n.get("inputs") or []:
            name = str(inp.get("name", "")).lower()
            if inp.get("link") is not None and any(k in name for k in _TEXT_IN) and "negative" not in name:
                t = text_of(links.get(inp["link"]), depth + 1, seen)
                if t and t not in parts:
                    parts.append(t)
        named = widgets(n)
        if parts:
            sep = named.get("separator") if isinstance(named.get("separator"), str) else ", "
            return sep.join(parts)
        for k in _TEXT_KEYS:
            v = named.get(k)
            if isinstance(v, str) and v.strip() and not _is_file(v):
                return v.strip()
        strs = [v for v in (n.get("widgets_values") or []) if isinstance(v, str) and len(v) > 15 and not _is_file(v)
                and "%" not in v] if isinstance(n.get("widgets_values"), list) else []
        return max(strs, key=len) if strs else ""

    def seed_of(nid, depth=0):
        n = nodes.get(nid)
        if not n or depth > 6:
            return None
        named = widgets(n)
        for k in ("seed", "noise_seed", "seed_value", "value", "int"):
            if isinstance(named.get(k), int) and not isinstance(named.get(k), bool):
                return named[k]
        for inp in n.get("inputs") or []:
            if "seed" in str(inp.get("name", "")).lower() and inp.get("link") is not None:
                return seed_of(links.get(inp["link"]), depth + 1)
        wv = n.get("widgets_values")
        if isinstance(wv, list) and wv and isinstance(wv[0], int) and not isinstance(wv[0], bool):
            return wv[0]
        return None

    def rank(n):
        t = str(n.get("type", "")).lower()
        return 0 if "save" in t else 1 if "sampler" in t else 2

    for n in sorted(nodes.values(), key=rank):
        if n.get("mode") in BYPASSED:
            continue
        ins = {str(i.get("name", "")).lower(): i for i in n.get("inputs") or []}
        if not f.positive and ins.get("positive", {}).get("link") is not None:
            f.positive = text_of(links.get(ins["positive"]["link"]))
        if f.seed is None:
            for k in ("seed_value", "seed", "noise_seed"):
                if ins.get(k, {}).get("link") is not None:
                    f.seed = seed_of(links.get(ins[k]["link"]))
                    break
        if f.positive and f.seed is not None:
            break


def _a1111(text: str, f: _Found):
    f.text(text)
    head = text.split("\nNegative prompt:")[0].split("\nSteps:")[0]
    f.positive = f.positive or head.strip()
    m = re.search(r"(?:^|,\s*)Model:\s*([^,\n]+)", text)
    if m and not f.ckpt:
        f.ckpt = m.group(1).strip()
    m = re.search(r"(?:^|,\s*)Seed:\s*(\d+)", text)
    if m and f.seed is None:
        f.seed = int(m.group(1))
    _hashes_json(text, f)


def _hashes_json(text: str, f: _Found):
    """Image Saver / Civitai style: Hashes: {"model": "...", "lora:name": "abcdef1234"} (AutoV2 = sha256[:10])."""
    m = re.search(r"Hashes:\s*(\{[^{}]*\})", text)
    if not m:
        return
    try:
        d = json.loads(m.group(1))
    except ValueError:
        return
    for k, v in d.items():
        if isinstance(k, str) and k.lower().startswith("lora:") and isinstance(v, str) and len(v) >= 8:
            f.hashes[v.strip().lower()] = norm(k[5:])


def _generic(obj, f: _Found, key: str = "", depth: int = 0):
    """Last resort for other savers: lora-looking keys and <lora:> tags anywhere."""
    if depth > 12:
        return
    if isinstance(obj, dict):
        if any(k in obj for k in ("lora", "lora_name")) and any(k in obj for k in ("on", "active", "strength", "weight")):
            _lora_dict(obj, f)
        for k, v in obj.items():
            _generic(v, f, str(k), depth + 1)
    elif isinstance(obj, list):
        for v in obj:
            _generic(v, f, key, depth + 1)
    elif isinstance(obj, str):
        if "lora" in key.lower() and _is_file(obj):
            f.add(obj)
        if key.lower() in ("parameters", "prompt", "positive", "text") or "<lora:" in obj or "Lora hashes" in obj:
            if "Steps:" in obj or "Negative prompt:" in obj:
                _a1111(obj, f)
            else:
                f.text(obj)


def _maybe_json(v):
    if isinstance(v, str) and v.strip()[:1] in "{[":
        try:
            return json.loads(v)
        except ValueError:
            return None
    return v


def parse_meta(data) -> _Found:
    """Everything we can learn from a sidecar .json / PNG text chunks (dict)."""
    f = _Found()
    if not isinstance(data, dict):
        return f
    prompt = _maybe_json(data.get("prompt"))
    workflow = _maybe_json(data.get("workflow"))
    is_api = not prompt and any(isinstance(v, dict) and "class_type" in v for v in data.values())
    if is_api:
        prompt = data
    if not workflow and isinstance(data.get("nodes"), list):
        workflow = data
    if isinstance(prompt, dict):
        _api(prompt, f)                       # what actually ran
    if not f.loras and isinstance(workflow, dict):
        _workflow(workflow, f)                # UI workflow: bypassed nodes skipped
    params = data.get("parameters")
    if isinstance(params, str):
        _a1111(params, f)
    if not f.loras and not f.hashes:
        rest = {k: v for k, v in data.items() if k not in ("prompt", "workflow")} if (prompt or workflow) else data
        _generic(rest, f)
    return f


def read_png_text(path: Path, max_bytes: int = 64 << 20) -> dict:
    """The PNG's text chunks (tEXt / zTXt / iTXt), read straight from the file: the picture itself is skipped,
    never decoded (asking PIL for .text decodes the whole picture first, far too slow for thousands)."""
    import struct
    import zlib
    out: dict = {}
    try:
        with open(path, "rb") as fh:
            if fh.read(8) != b"\x89PNG\r\n\x1a\n":
                return {}
            while True:
                hdr = fh.read(8)
                if len(hdr) < 8:
                    break
                n, typ = struct.unpack(">I4s", hdr)
                if typ == b"IEND":
                    break
                if typ in (b"tEXt", b"zTXt", b"iTXt") and n <= max_bytes:
                    data = fh.read(n)
                    fh.seek(4, 1)                      # CRC
                    key, _, rest = data.partition(b"\0")
                    k = key.decode("latin-1", "replace")
                    try:
                        if typ == b"tEXt":
                            out[k] = rest.decode("latin-1", "replace")
                        elif typ == b"zTXt":
                            out[k] = zlib.decompress(rest[1:]).decode("latin-1", "replace")
                        else:
                            comp, rest2 = rest[0], rest[2:]
                            _lang, _, rest2 = rest2.partition(b"\0")
                            _tkey, _, txt = rest2.partition(b"\0")
                            out[k] = (zlib.decompress(txt) if comp else txt).decode("utf-8", "replace")
                    except Exception:
                        pass
                else:
                    fh.seek(n + 4, 1)                  # skip picture data + CRC without reading it
    except OSError:
        return {}
    return out


def sidecar_for(img: Path, names: set) -> Path | None:
    for cand in (img.stem + ".json", img.name + ".json"):
        if cand in names:
            return img.with_name(cand)
    return None


def read_picture(img: Path, js: Path | None) -> dict:
    data, src = None, ""
    if js is not None:
        try:
            data, src = json.loads(js.read_text(encoding="utf-8", errors="replace")), "json"
        except Exception:
            data = None
    if data is None and img.suffix.lower() == ".png":
        data, src = read_png_text(img), "png"
        if not data:
            data = None
    if data is None:
        return {"src": "", "loras": {}}
    f = parse_meta(data if isinstance(data, dict) else {"_": data})
    if src == "json" and not f.hashes and img.suffix.lower() == ".png":
        # the workflow .json has no file hashes; Image Saver puts them in the PNG's "parameters" text
        params = read_png_text(img).get("parameters")
        if params:
            g = _Found()
            g.text(params)
            _hashes_json(params, g)
            f.hashes.update(g.hashes)
    pos = f.positive or (max(f.prompts, key=len) if f.prompts else "")
    return {"src": src, "loras": f.loras, "hashes": f.hashes, "ckpt": norm(f.ckpt) if f.ckpt else "",
            "seed": f.seed, "prompt": TAG_RE.sub("", pos).strip()[:MAX_PROMPT]}


# --------------------------------------------------------------------------- #
# The index
# --------------------------------------------------------------------------- #

def _sig(st, jst) -> list:
    return [int(st.st_mtime), st.st_size, int(jst.st_mtime) if jst else 0, jst.st_size if jst else 0]


def load_index() -> dict:
    try:
        d = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
        if d.get("version") == INDEX_VERSION and isinstance(d.get("pics"), dict):
            return d
    except Exception:
        pass
    return {"version": INDEX_VERSION, "pics": {}, "folders": {}}


def save_index(idx: dict):
    tmp = INDEX_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(idx, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    tmp.replace(INDEX_FILE)


_INDEX: dict | None = None


def index() -> dict:
    global _INDEX
    with LOCK:
        if _INDEX is None:
            _INDEX = load_index()
        return _INDEX


STOP = threading.Event()      # set by the Gallery tab's Stop button
SAVE_EVERY = 60               # seconds: an interrupted scan keeps what it read


def stop():
    STOP.set()


def scan(folder: str | Path, progress=None, full: bool = False, workers: int = 8) -> dict:
    """Bring the index up to date for one folder (with its subfolders). Returns counts.
    Only new or changed pictures are read (several at a time); the index is saved every minute,
    so a stopped or interrupted scan continues where it was the next time."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    root = Path(os.path.expanduser(str(folder))).resolve()
    if not root.is_dir():
        raise FileNotFoundError(str(root))
    STOP.clear()
    t0 = time.time()
    tag = f"[Gallery] {root}"
    print(f"{tag}: looking for pictures ...", flush=True)
    found: list[tuple[Path, Path | None]] = []
    last = time.time()
    for dirpath, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        names = set(files)
        for fn in files:
            if os.path.splitext(fn)[1].lower() in PIC_EXT:
                p = Path(dirpath) / fn
                found.append((p, sidecar_for(p, names)))
        if time.time() - last > 0.5:
            last = time.time()
            if progress:
                progress(0.0, desc=f"Looking for pictures: {len(found)}")
        if STOP.is_set():
            break
    print(f"{tag}: {len(found)} pictures ({time.time() - t0:.1f} s)", flush=True)

    stats = {"pictures": len(found), "read": 0, "kept": 0, "removed": 0, "with_lora": 0, "no_meta": 0,
             "stopped": False}
    idx = index()
    with LOCK:
        pics = idx["pics"]
        todo = []
        for p, js in found:
            try:
                st = p.stat()
                jst = js.stat() if js else None
            except OSError:
                continue
            sig = _sig(st, jst)
            old = pics.get(str(p))
            if old and old.get("sig") == sig and not full:
                stats["kept"] += 1
            else:
                todo.append((p, js, sig, st.st_mtime))
    print(f"{tag}: {stats['kept']} unchanged, {len(todo)} to read", flush=True)

    def work(item):
        p, js, sig, mt = item
        e = read_picture(p, js)
        e["sig"], e["t"] = sig, mt
        return str(p), e

    t1 = last = last_save = time.time()
    done = 0
    if todo:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(work, it) for it in todo]
            for fut in as_completed(futs):
                try:
                    key, e = fut.result()
                except Exception:
                    continue
                with LOCK:
                    pics[key] = e
                done += 1
                stats["read"] += 1
                now = time.time()
                if now - last > 0.5:
                    last = now
                    rate = done / max(0.01, now - t1)
                    left = (len(todo) - done) / max(rate, 0.01)
                    msg = f"Reading pictures {done}/{len(todo)} ({rate:.0f}/s, about {left / 60:.0f} min left)"
                    if progress:
                        progress(done / len(todo), desc=msg)
                    if now - last_save > SAVE_EVERY:
                        last_save = now
                        print(f"{tag}: {msg}", flush=True)
                        with LOCK:
                            save_index(idx)
                if STOP.is_set():
                    stats["stopped"] = True
                    for f in futs:
                        f.cancel()
                    break
    with LOCK:
        if not stats["stopped"] and not STOP.is_set():
            seen = {str(p) for p, _ in found}
            prefix = str(root) + os.sep
            for key in [k for k in pics if k.startswith(prefix) and k not in seen]:
                del pics[key]
                stats["removed"] += 1
            idx["folders"][str(root)] = {"scanned": time.strftime("%Y-%m-%d %H:%M"), "pictures": len(found)}
        else:
            stats["stopped"] = True
        for p, _ in found:
            e = pics.get(str(p))
            if e is not None:
                stats["with_lora"] += bool(e.get("loras"))
                stats["no_meta"] += not e.get("src")
        save_index(idx)
    stats["seconds"] = round(time.time() - t0, 1)
    print(f"{tag}: {'stopped' if stats['stopped'] else 'done'} in {stats['seconds']} s, "
          f"{stats['read']} read, {stats['with_lora']} with LoRAs", flush=True)
    return stats


def pictures_for(names: set, hashes: set) -> list[tuple[str, dict, str]]:
    """[(path, entry, matched name)] for pictures using any of these LoRA names (normalised) or short hashes."""
    out = []
    with LOCK:
        for path, e in index()["pics"].items():
            hit = next((n for n in (e.get("loras") or {}) if n in names), None)
            if hit is None and hashes:
                hit = next((n for h, n in (e.get("hashes") or {}).items()
                            if any(h.startswith(x) or x.startswith(h) for x in hashes if len(h) >= 8)), None)
                if hit is not None and hit not in names:
                    hit = "#" + hit
            if hit is not None:
                out.append((path, e, hit))
    out.sort(key=lambda t: t[1].get("t") or 0, reverse=True)
    return out


def name_counts() -> dict:
    """LoRA name -> number of pictures, over the whole index."""
    out: dict[str, int] = {}
    with LOCK:
        for e in index()["pics"].values():
            for n in e.get("loras") or {}:
                out[n] = out.get(n, 0) + 1
    return out


def entry(path: str) -> dict | None:
    with LOCK:
        return index()["pics"].get(path)
