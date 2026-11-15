# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Remove watermarks, logos and text with your own ComfyUI inpaint workflow (step 6: Remove, Paint over it yourself).

Settings › Test pictures (ComfyUI) › Inpaint workflow: a workflow saved from ComfyUI with one Load Image node whose
MASK output says what to fill in (as in lorastudio_inpaint_sdxl.json: Load Image → Grow Mask → Inpaint Model
Conditioning → KSampler → VAE Decode → Image Composite Masked → Save Image). Per mark LoRA Studio
  1. cuts a window around it (with room around the mark, so the model sees what belongs there),
  2. scales the window to about 1024 px (what SDXL works best at),
  3. sends it as a PNG whose transparent part is the mask (ComfyUI's Load Image turns transparency into its MASK),
  4. scales the answer back and pastes only the masked part (with soft edges) into the full picture.
Marks close together share one window. The seed is new every time; everything else is the workflow's own.
"""
from __future__ import annotations
__version__ = "1.151126f"  # LoRA Studio version, see version.py

import json
import random
import shutil
import time
from pathlib import Path

import lspaths as _LP  # noqa: E402

FILE = _LP.USER / "comfy_inpaint.json"
WF = _LP.USER / "comfy_workflows" / "inpaint.json"
DEFAULTS = {"name": "", "note": ""}
SIDE = 1024                     # the window is scaled to this long side
GROW = 6                        # px the mask grows at full size before it is sent
_ALIVE = {"t": 0.0, "ok": False}


# --------------------------------------------------------------------------- settings
def settings() -> dict:
    try:
        d = json.loads(FILE.read_text(encoding="utf-8"))
    except Exception:                                    # noqa: BLE001
        d = {}
    return {**DEFAULTS, **{k: v for k, v in d.items() if k in DEFAULTS}}


def _save_settings(**kw):
    s = settings()
    s.update({k: v for k, v in kw.items() if k in DEFAULTS})
    FILE.parent.mkdir(parents=True, exist_ok=True)
    FILE.write_text(json.dumps(s, indent=1), encoding="utf-8")


def ready() -> bool:
    return WF.is_file()


def usable() -> bool:
    """A workflow is set and ComfyUI answers (asked at most every 20 s, as step 6 asks for every row)."""
    if not ready():
        return False
    if time.time() - _ALIVE["t"] > 20:
        import comfytest as CT
        _ALIVE["ok"], _ALIVE["t"] = CT.alive(), time.time()
    return _ALIVE["ok"]


def _api(url=None) -> dict:
    import comfyupscale as U
    d = json.loads(WF.read_text(encoding="utf-8"))
    return U.to_api(d, U._object_info(url) if "nodes" in d else None)


def check(api: dict) -> dict:
    """{'load', 'save', 'masked', 'seeds'}: the Load Image, the Save Image, whether something uses the mask, seed ids."""
    import comfyupscale as U
    ids = U.check(api)
    masked = any(isinstance(v, list) and v[:2] == [ids["load"], 1]
                 for n in api.values() for v in n["inputs"].values())
    seeds = [(k, name) for k, n in api.items() for name in ("seed", "noise_seed") if isinstance(n["inputs"].get(name), int)]
    return {**ids, "masked": masked, "seeds": seeds}


def import_workflow(path, url: str | None = None) -> str:
    import comfyupscale as U
    p = Path(path)
    d = json.loads(p.read_text(encoding="utf-8"))
    api = U.to_api(d, U._object_info(url) if "nodes" in d else None)
    ids = check(api)
    if not ids["masked"]:
        raise ValueError("Nothing in the workflow uses the Load Image node's MASK output, so it would not know what "
                         "to fill in.")
    WF.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, WF)
    ck = next((n["inputs"].get("ckpt_name") or n["inputs"].get("unet_name") for n in api.values()
               if n["inputs"].get("ckpt_name") or n["inputs"].get("unet_name")), "")
    note = f"{ck or 'model as in the workflow'} · {len(api)} nodes"
    _save_settings(name=p.name, note=note)
    _ALIVE["t"] = 0
    return f"✔ Inpaint workflow read: {note}. Step 6's Remove and Paint over it yourself now use it while ComfyUI runs."


def forget() -> str:
    WF.unlink(missing_ok=True)
    _save_settings(name="", note="")
    return "The inpaint workflow is removed: step 6 uses the picture tools again."


# --------------------------------------------------------------------------- windows around the marks
def _windows(boxes, W, H) -> list[list[int]]:
    """A window per mark, with room around it; windows that touch are joined."""
    wins = []
    for b in boxes:
        x0, y0, x1, y1 = b
        pad = max(96, int(0.6 * max(x1 - x0, y1 - y0)))
        side = max(x1 - x0, y1 - y0) + 2 * pad
        side = max(side, min(512, W, H))
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        w, h = min(W, max(side, x1 - x0 + 2 * pad)), min(H, max(side, y1 - y0 + 2 * pad))
        l = int(min(max(0, cx - w / 2), W - w))
        t = int(min(max(0, cy - h / 2), H - h))
        wins.append([l, t, int(l + w), int(t + h)])
    changed = True
    while changed:
        changed = False
        for i in range(len(wins)):
            for j in range(i + 1, len(wins)):
                a, b = wins[i], wins[j]
                if not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]):
                    wins[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    del wins[j]
                    changed = True
                    break
            if changed:
                break
    return wins


def _mask_of(W, H, boxes=None, mask_file=None):
    """'L' mask at full size, 255 = fill in."""
    from PIL import Image, ImageDraw, ImageFilter
    if mask_file:
        with Image.open(mask_file) as mk:
            m = mk.convert("L").resize((W, H)).point(lambda v: 255 if v > 20 else 0)
    else:
        m = Image.new("L", (W, H), 0)
        d = ImageDraw.Draw(m)
        for b in boxes or []:
            d.rectangle([int(v) for v in b], fill=255)
    return m.filter(ImageFilter.MaxFilter(2 * GROW + 1)) if GROW else m


def _boxes_of(mask) -> list[list[int]]:
    """Boxes of the separate painted parts (on a small copy; enough to place the windows)."""
    import numpy as np
    small = mask.copy()
    f = min(1.0, 256 / max(mask.size))
    small = small.resize((max(1, int(mask.size[0] * f)), max(1, int(mask.size[1] * f))))
    a = np.asarray(small) > 0
    seen = np.zeros_like(a)
    out = []
    for y, x in zip(*np.nonzero(a)):
        if seen[y, x]:
            continue
        stack, xs, ys = [(y, x)], [], []
        seen[y, x] = True
        while stack:
            cy, cx = stack.pop()
            xs.append(cx)
            ys.append(cy)
            for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                if 0 <= ny < a.shape[0] and 0 <= nx < a.shape[1] and a[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        out.append([int(min(xs) / f), int(min(ys) / f), int((max(xs) + 1) / f), int((max(ys) + 1) / f)])
    return out


# --------------------------------------------------------------------------- running it
def inpaint(src: Path, boxes=None, mask_file=None, say=None, timeout: float = 900):
    """The picture with the marks (boxes) or the painted part (mask_file) filled in by the workflow: a PIL image
    (upright, RGB) of the same size as src."""
    import comfytest as CT
    import comfyupscale as U
    from PIL import Image, ImageFilter, ImageOps
    url = CT.settings()["url"]
    api = _api(url)
    ids = check(api)
    with Image.open(src) as im:
        pic = ImageOps.exif_transpose(im).convert("RGB")
    W, H = pic.size
    mask = _mask_of(W, H, boxes, mask_file)
    if not mask.getbbox():
        raise ValueError("nothing to fill in")
    wins = _windows(_boxes_of(mask) if mask_file else [[int(v) for v in b] for b in boxes], W, H)
    U._free_ollama()
    tmpdir = _LP.USER / "cache" / "inpaint"
    tmpdir.mkdir(parents=True, exist_ok=True)
    for n, (l, t, r, b) in enumerate(wins, 1):
        if say:
            say(f"part {n} of {len(wins)}")
        part, pm = pic.crop((l, t, r, b)), mask.crop((l, t, r, b))
        if not pm.getbbox():
            continue
        w, h = part.size
        f = SIDE / max(w, h)
        sw, sh = max(64, round(w * f / 8) * 8), max(64, round(h * f / 8) * 8)
        send = part.resize((sw, sh), Image.LANCZOS).convert("RGBA")
        send.putalpha(pm.resize((sw, sh), Image.NEAREST).point(lambda v: 0 if v > 127 else 255))
        tmp = tmpdir / f"ls_inpaint_{int(time.time() * 1000)}_{n}.png"
        send.save(tmp)
        try:
            name = U._upload(url, tmp)
            g = json.loads(json.dumps(api))
            g[ids["load"]]["inputs"]["image"] = name
            g[ids["save"]]["inputs"]["filename_prefix"] = "lorastudio/inpaint"
            for k, field in ids["seeds"]:
                g[k]["inputs"][field] = random.randint(1, 2 ** 48)
            got = tmpdir / (tmp.stem + ".out.png")
            CT._run_one(url, g, got, timeout=timeout)
            with Image.open(got) as gi:
                back = gi.convert("RGB").resize((w, h), Image.LANCZOS)
            got.unlink(missing_ok=True)
        finally:
            tmp.unlink(missing_ok=True)
        soft = pm.filter(ImageFilter.GaussianBlur(2))
        pic.paste(Image.composite(back, part, soft), (l, t))
    return pic


def why_not() -> str:
    """'' when the workflow can run now, else the reason."""
    import comfytest as CT
    if not ready():
        return "no inpaint workflow (Settings › Test pictures (ComfyUI) › Inpaint workflow)"
    if not CT.alive():
        return f"ComfyUI does not answer at {CT.settings()['url']}"
    return ""
