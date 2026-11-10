# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Upscale and sharpen dataset pictures with your own ComfyUI workflow (for example SeedVR2).

Settings › Test pictures (ComfyUI) › Upscale workflow: you give LoRA Studio a workflow saved from ComfyUI (the normal
'Save' format or 'Export (API)'). It is run as it is, with three things changed per picture:

  the picture   its Load Image node gets the picture (sent to ComfyUI's input folder first)
  the size      a SeedVR2 Video Upscaler node gets the target short side (resolution) and a cap for the long side;
                other upscalers run at the size the workflow sets
  the noise     SeedVR2's input / latent noise, by how much the picture grows (Automatic), or the workflow's own

The result is taken back from its Save Image node. ComfyUI's own models, GGUF files and settings stay yours.
"""
from __future__ import annotations
__version__ = "1.101126"  # LoRA Studio version, see version.py

import json
import shutil
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

import lspaths as _LP  # noqa: E402

FILE = _LP.USER / "comfy_upscale.json"
WF = _LP.USER / "comfy_workflows" / "upscale.json"
DEFAULTS = {"noise": "auto", "name": "", "note": ""}
NOISE_CHOICES = [("Automatic: more noise the more a picture grows", "auto"), ("The workflow's own values", "workflow")]
CONTROL = {"fixed", "increment", "decrement", "randomize"}
SKIP_TYPES = {"Note", "MarkdownNote", "PrimitiveNode", "Reroute"}
UPSCALER = "SeedVR2VideoUpscaler"


# --------------------------------------------------------------------------- settings
def settings() -> dict:
    try:
        d = json.loads(FILE.read_text(encoding="utf-8"))
    except Exception:                                    # noqa: BLE001
        d = {}
    return {**DEFAULTS, **{k: v for k, v in d.items() if k in DEFAULTS}}


def save_settings(**kw) -> dict:
    s = settings()
    s.update({k: v for k, v in kw.items() if k in DEFAULTS})
    FILE.parent.mkdir(parents=True, exist_ok=True)
    FILE.write_text(json.dumps(s, indent=1), encoding="utf-8")
    return s


def ready() -> bool:
    return WF.is_file()


# --------------------------------------------------------------------------- the workflow as an API graph
def _widget_inputs(node: dict) -> list[dict]:
    return [i for i in node.get("inputs") or [] if i.get("widget")]


def _widgets(node: dict, info: dict | None) -> dict:
    """{input name: value} of a UI-format node. Uses the widget markers of its inputs (new ComfyUI), else the input
    order ComfyUI reports for the node type (object_info). The value after a seed (fixed / randomize …) is skipped."""
    vals = node.get("widgets_values")
    if isinstance(vals, dict):
        return dict(vals)
    vals = list(vals or [])
    names = [(i["widget"].get("name") or i.get("name"), i.get("type")) for i in _widget_inputs(node)]
    if not names and info:
        req = {**(info.get("input", {}).get("required") or {}), **(info.get("input", {}).get("optional") or {})}
        for k, spec in req.items():
            t = spec[0] if isinstance(spec, (list, tuple)) and spec else None
            if isinstance(t, list) or t in ("INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"):
                names.append((k, "COMBO" if isinstance(t, list) else t))
    out, i = {}, 0
    for name, typ in names:
        if i >= len(vals):
            break
        out[name] = vals[i]
        i += 1
        if i < len(vals) and isinstance(vals[i], str) and vals[i] in CONTROL and (typ == "INT" or "seed" in name):
            i += 1
    out.pop("upload", None)                               # Load Image's upload button
    return out


def to_api(d: dict, object_info: dict | None = None) -> dict:
    """A workflow in API format: unchanged when it already is one, else made from the normal 'Save' format.
    Muted and bypassed nodes are left out (a bypassed node passes its input of the same type through)."""
    if "nodes" not in d:
        if all(isinstance(v, dict) and "class_type" in v for v in d.values()):
            return json.loads(json.dumps(d))
        raise ValueError("This is not a ComfyUI workflow.")
    nodes = {n["id"]: n for n in d.get("nodes") or []}
    links = {ln[0]: ln for ln in d.get("links") or [] if isinstance(ln, list) and len(ln) >= 5}

    def source(link_id, depth=0):
        ln = links.get(link_id)
        if not ln or depth > 20:
            return None
        src = nodes.get(ln[1])
        if not src or src.get("mode") == 2:
            return None
        if src.get("type") == "Reroute" or src.get("mode") == 4:
            want = (src.get("outputs") or [{}])[ln[2]].get("type") if ln[2] < len(src.get("outputs") or []) else None
            for inp in src.get("inputs") or []:
                if inp.get("link") is not None and (src.get("type") == "Reroute" or inp.get("type") == want):
                    return source(inp["link"], depth + 1)
            return None
        return [str(src["id"]), ln[2]]

    api = {}
    for nid, n in nodes.items():
        if n.get("type") in SKIP_TYPES or n.get("mode") in (2, 4):
            continue
        inputs = _widgets(n, (object_info or {}).get(n.get("type")))
        for inp in n.get("inputs") or []:
            if inp.get("link") is not None:
                s = source(inp["link"])
                if s:
                    inputs[inp["name"]] = s
                else:
                    inputs.pop(inp["name"], None)
        api[str(nid)] = {"class_type": n["type"], "inputs": inputs}
    return api


def check(api: dict) -> dict:
    """What LoRA Studio needs in the graph: {'load', 'save', 'upscaler'} node ids (upscaler may be '')."""
    load = [k for k, v in api.items() if v["class_type"] == "LoadImage"]
    save = [k for k, v in api.items() if v["class_type"] in ("SaveImage", "Image Save")]
    up = [k for k, v in api.items() if v["class_type"] == UPSCALER]
    if len(load) != 1:
        raise ValueError(f"The workflow needs exactly one Load Image node (it has {len(load)}).")
    if not save:
        raise ValueError("The workflow has no Save Image node.")
    return {"load": load[0], "save": save[0], "upscaler": up[0] if up else ""}


def import_workflow(path, url: str | None = None) -> str:
    """Keep a copy of the workflow; returns a line for Settings."""
    p = Path(path)
    d = json.loads(p.read_text(encoding="utf-8"))
    info = _object_info(url) if "nodes" in d else None
    api = to_api(d, info)
    ids = check(api)
    WF.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, WF)
    model = next((v["inputs"].get("model") for v in api.values() if "LoadDiT" in v["class_type"] or
                  v["class_type"].endswith("LoadDiTModel")), "")
    note = (f"SeedVR2 · {model or 'model as in the workflow'} · LoRA Studio sets the size and noise per picture"
            if ids["upscaler"] else "no SeedVR2 Video Upscaler node: the workflow runs at its own size")
    save_settings(name=p.name, note=note)
    return f"✔ Upscale workflow read ({len(api)} nodes): {note}."


def _object_info(url):
    try:
        import comfytest as CT
        return json.loads(CT._req(url or CT.settings()["url"], "/object_info", timeout=30))
    except Exception:                                    # noqa: BLE001
        return None


# --------------------------------------------------------------------------- size and noise
def noise_for(factor: float) -> tuple[float, float]:
    """(input noise, latent noise) for SeedVR2 by how much the picture grows. Input noise hides the artefacts of small
    sources; latent noise is added only for big jumps, as it can soften detail."""
    if factor <= 1.3:
        return 0.0, 0.0
    if factor <= 2.0:
        return 0.1, 0.0
    if factor <= 3.0:
        return 0.2, 0.0
    if factor <= 4.0:
        return 0.3, 0.05
    return 0.4, 0.1


def plan(w: int, h: int, target_short: int) -> dict:
    """What a run does to a w×h picture: the short side it gets, the cap of the long side and the noise."""
    short = max(1, min(w, h))
    target = max(int(target_short), short) // 2 * 2         # never smaller; even values work best
    factor = target / short
    inn, lat = noise_for(factor)
    return {"resolution": target, "max_resolution": int(target * max(w, h) / short) + 2, "factor": round(factor, 2),
            "input_noise": inn, "latent_noise": lat}


def graph(api: dict, image_name: str, pl: dict, prefix: str, noise: str = "auto") -> dict:
    g = json.loads(json.dumps(api))
    ids = check(g)
    g[ids["load"]]["inputs"]["image"] = image_name
    g[ids["save"]]["inputs"]["filename_prefix"] = prefix
    if ids["upscaler"]:
        i = g[ids["upscaler"]]["inputs"]
        i["resolution"] = pl["resolution"]
        i["max_resolution"] = pl["max_resolution"]
        if noise == "auto":
            i["input_noise_scale"], i["latent_noise_scale"] = pl["input_noise"], pl["latent_noise"]
    return g


# --------------------------------------------------------------------------- running it
def _upload(url: str, src: Path) -> str:
    """Send a picture to ComfyUI's input folder; returns the name its Load Image node uses."""
    b = uuid.uuid4().hex
    name = f"ls_{uuid.uuid4().hex[:8]}_{src.name}"
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
            f"Content-Type: application/octet-stream\r\n\r\n").encode() + src.read_bytes() + \
        f"\r\n--{b}\r\nContent-Disposition: form-data; name=\"subfolder\"\r\n\r\nlorastudio\r\n" \
        f"--{b}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n--{b}--\r\n".encode()
    r = urllib.request.Request(url.rstrip("/") + "/upload/image", data=body,
                               headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    with urllib.request.urlopen(r, timeout=120) as resp:
        d = json.loads(resp.read())
    return f"{d.get('subfolder')}/{d['name']}" if d.get("subfolder") else d["name"]


def why_not() -> str:
    """'' when an upscale can run now, else the reason."""
    import comfytest as CT
    if not ready():
        return "no upscale workflow (Settings › Test pictures (ComfyUI) › Upscale workflow)"
    if not CT.alive():
        return f"ComfyUI does not answer at {CT.settings()['url']}"
    try:
        import sdgui as S
        if S.RUN.running():
            return "a training run is active"
    except Exception:                                    # noqa: BLE001
        pass
    return ""


def upscale(src: Path, dst: Path, target_short: int, timeout: float = 1800) -> dict:
    """Run the upscale workflow on one picture; the result is written to dst (as PNG, or dst's format)."""
    import comfytest as CT
    from PIL import Image
    url = CT.settings()["url"]
    d = json.loads(WF.read_text(encoding="utf-8"))
    api = to_api(d, _object_info(url) if "nodes" in d else None)
    with Image.open(src) as im:
        w, h = im.size
    pl = plan(w, h, target_short)
    _free_ollama()
    name = _upload(url, Path(src))
    g = graph(api, name, pl, f"lorastudio/up_{Path(src).stem}", settings()["noise"])
    tmp = Path(dst).with_name(Path(dst).stem + ".comfy.png")
    t0 = time.time()
    CT._run_one(url, g, tmp, timeout=timeout)
    with Image.open(tmp) as im:
        im.load()
        out = im.convert("RGB") if Path(dst).suffix.lower() in (".jpg", ".jpeg") else im.copy()
        size = out.size
    out.save(dst, **({"quality": 95} if Path(dst).suffix.lower() in (".jpg", ".jpeg") else {}))
    if tmp != Path(dst):
        tmp.unlink(missing_ok=True)
    return {**pl, "w": size[0], "h": size[1], "seconds": round(time.time() - t0, 1)}


def _free_ollama():
    """The vision model of the sort and SeedVR2 do not fit in video memory together: unload Ollama's models."""
    try:
        import ollamactl
        ollamactl.stop_all()
    except Exception:                                    # noqa: BLE001
        pass

