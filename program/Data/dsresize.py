# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""dsresize.py - step 6 of a dataset, watermarks and size: whether the watermark check is done (wm_done), the plan per
picture (resize_plan: crop around the person, fit, upscale small ones with the picture tools; your own choices in
resize_overrides), the preview of a crop, applying it (apply_resize_steps; the originals stay in _originals/) and
putting the originals back (restore_originals).
"""
from __future__ import annotations
__version__ = "0"  # LoRA Studio version, see version.py

import shutil
import time
from pathlib import Path

from PIL import Image
from PIL import ImageDraw
from PIL import ImageOps
import datacheck as DC
import dsflow as F


UPSCALE_FLOOR = 0.4          # with the picture tools, a short side down to 40 % of the type's size is upscaled in step 6


def soften_small(issues, w, h, side, tools):
    """With the picture tools, 'too small' between 40 % and 60 % of the type's size is a warning: step 6 upscales it."""
    if not (tools and side and w and h and min(w, h) >= side * UPSCALE_FLOOR):
        return issues
    return [(DC.WARN, f"small: {w}x{h}, upscaled with Real-ESRGAN in step 6") if s == DC.BAD and t.startswith("too small")
            else (s, t) for s, t in issues]


# --------------------------------------------------------------------------- watermarks
def main_person(persons, W, H):
    """The box of the main person from the picture tools: the most certain, biggest one; None when unsure."""
    ps = [p for p in persons or [] if len(p) >= 5 and p[4] >= 0.7]
    if not ps or not W or not H:
        return None
    best = max(ps, key=lambda p: p[4] * (p[2] - p[0]) * (p[3] - p[1]))
    if (best[2] - best[0]) * (best[3] - best[1]) < 0.06 * W * H:
        return None
    return [int(v) for v in best[:4]]


def wm_done(ds) -> str:
    """You are happy with the watermarks of this dataset (also when some could not be removed): the ones not decided
    stay as they are, and the dataset no longer asks about watermarks."""
    if not ds:
        return ""
    r = F.root(ds)
    marks = DC.reconcile(r, DC.marks_load(r))
    n = 0
    for e in marks.values():
        if (e.get("marks") or e.get("border")) and not e.get("choice"):
            e["choice"] = "keep"
            n += 1
    DC.marks_save(r, marks)
    m = F.meta(ds)
    m["wm_done"] = True
    F.meta_save(ds, m)
    return ("✔ Watermarks marked as done" + (f": {n} undecided picture{'s' if n != 1 else ''} kept as they are." if n
                                              else ".") + " Change any picture in the list above at any time.")


# --------------------------------------------------------------------------- 6 Watermarks and size
def resize_overrides(ds) -> dict:
    return (F._load(ds, "resize.json", {}) or {}).get("over", {})


def resize_save_over(ds, over: dict, max_side: int):
    F._save(ds, "resize.json", {"over": over, "max_side": int(max_side)})


def resize_plan(ds, max_side: int, over: dict | None = None, upscale: bool = True) -> list[dict]:
    over = over if over is not None else resize_overrides(ds)
    rows = []
    r = F.root(ds)
    side = F.min_side(F.meta(ds)["types"])
    tools = F.tools_ok() and upscale
    for p in F.images(ds):
        try:
            w, h = DC.shown_size(p)
        except Exception:                                # noqa: BLE001
            continue
        rel = str(p.relative_to(r))
        o = over.get(rel) or {}
        sugg = suggest_mode(w, h, max_side, side, tools)
        mode = o.get("mode") or sugg
        if mode == "upscale" and not tools:
            mode = "none"
        cw, ch = w, h
        if mode == "crop" and o.get("box"):
            l, t, rr, b = o["box"]
            cw, ch = max(1, int((rr - l) * w)), max(1, int((b - t) * h))
        if mode == "upscale":
            f = min(max(1.0, side / max(1, min(cw, ch))), max_side / max(cw, ch), 4.0)
        else:
            f = min(1.0, max_side / max(cw, ch)) if mode in ("scale", "crop") else 1.0
        rows.append({"path": str(p), "rel": rel, "w": w, "h": h, "mode": mode, "by_you": rel in over, "sugg": sugg,
                     "new": (round(cw * f), round(ch * f)), "box": o.get("box")})
    return rows


def suggest_mode(w, h, max_side, side, tools) -> str:
    if max(w, h) > max_side:
        return "scale"
    if tools and side and side * UPSCALE_FLOOR <= min(w, h) < side:
        return "upscale"
    return "none"


def person_box(path: str, aspect: str = "free", run: bool = False) -> list[float] | None:
    """A crop around the person(s) the picture tools found; run=False only uses what they already know."""
    try:
        import pictools
        p = Path(path)
        if run:
            if not F.tools_ok():
                return None
            e = pictools.analyze([p]).get(str(p))
        else:
            e = pictools.cached(p)
        if not e:
            return None
        w, h = DC.shown_size(p)
        a = {"1:1": 1.0, "3:4": 0.75, "4:3": 4 / 3}.get(aspect)
        return pictools.smart_box(e, w, h, a)
    except Exception:                                    # noqa: BLE001
        return None


def crop_preview(path: str, box, side: int = 640) -> Image.Image | None:
    try:
        im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    except Exception:                                    # noqa: BLE001
        return None
    im.thumbnail((side, side))
    if box:
        w, h = im.size
        l, t, r, b = box
        shade = Image.new("RGBA", im.size, (0, 0, 0, 120))
        d = ImageDraw.Draw(shade)
        d.rectangle([int(l * w), int(t * h), int(r * w), int(b * h)], fill=(0, 0, 0, 0))
        im = Image.alpha_composite(im.convert("RGBA"), shade).convert("RGB")
        ImageDraw.Draw(im).rectangle([int(l * w), int(t * h), int(r * w) - 1, int(b * h) - 1], outline=(232, 163, 61),
                                     width=3)
    return im


def fit_box(w, h, l, t, r, b, aspect: str):
    """Make a box (fractions) follow an aspect ratio ('free', '1:1', '3:4', '4:3'), shrinking around its centre."""
    l, t, r, b = [max(0.0, min(1.0, float(x))) for x in (l, t, r, b)]
    if r - l < 0.05:
        r = min(1.0, l + 0.05)
    if b - t < 0.05:
        b = min(1.0, t + 0.05)
    if aspect in ("1:1", "3:4", "4:3"):
        aw, ah = (int(x) for x in aspect.split(":"))
        bw, bh = (r - l) * w, (b - t) * h
        want = aw / ah
        if bw / bh > want:
            nbw = bh * want
            cx = (l + r) / 2 * w
            l, r = (cx - nbw / 2) / w, (cx + nbw / 2) / w
        else:
            nbh = bw / want
            cy = (t + b) / 2 * h
            t, b = (cy - nbh / 2) / h, (cy + nbh / 2) / h
    return [round(max(0.0, l), 4), round(max(0.0, t), 4), round(min(1.0, r), 4), round(min(1.0, b), 4)]


def apply_resize_steps(ds, max_side: int, upscale: bool = True):
    """Generator: yields a progress line per picture; its return value is the message."""
    over = resize_overrides(ds)
    rows = resize_plan(ds, max_side, over, upscale)
    todo = [x for x in rows if x["mode"] in ("scale", "crop", "upscale")
            and (x["new"] != (x["w"], x["h"]) or x["mode"] == "crop")]
    F.ensure_originals(ds, [x["path"] for x in todo])
    n_s = n_c = n_u = 0
    errs = []
    t0 = time.time()
    for i, x in enumerate(todo):
        p = Path(x["path"])
        left = ""
        if i:
            left = f" · about {int((time.time() - t0) / i * (len(todo) - i))} s left"
        yield (f"⏳ {'Upscaling' if x['mode'] == 'upscale' else 'Cropping' if x['mode'] == 'crop' else 'Scaling'} "
               f"{p.name} ({i + 1} of {len(todo)}){left}")
        if x["mode"] == "upscale":
            try:
                import pictools
                pictools.upscale(p, p, min(x["new"]), "photo")
                n_u += 1
            except Exception as e:                       # noqa: BLE001
                errs.append(f"{p.name}: {e}")
            continue
        try:
            with Image.open(p) as im0:
                info = im0.info
                fmt = im0.format
                im = ImageOps.exif_transpose(im0)
                im.load()
            if x["mode"] == "crop" and x.get("box"):
                l, t, r, b = x["box"]
                w, h = im.size
                im = im.crop((int(l * w), int(t * h), int(r * w), int(b * h)))
                n_c += 1
            if max(im.size) > max_side:
                f = max_side / max(im.size)
                im = im.resize((max(1, round(im.size[0] * f)), max(1, round(im.size[1] * f))), Image.LANCZOS)
                n_s += 1 if x["mode"] == "scale" else 0
            kw = {}
            if fmt == "JPEG":
                kw = {"quality": 95, "subsampling": 0}
                if im.mode not in ("RGB", "L"):
                    im = im.convert("RGB")
            elif fmt == "PNG" and info.get("icc_profile"):
                kw = {"icc_profile": info["icc_profile"]}
            im.save(p, format=fmt or None, **kw)
        except Exception as e:                           # noqa: BLE001
            errs.append(f"{p.name}: {e}")
    over = {k: v for k, v in over.items() if v.get("mode") == "keep"}  # done crops are part of the picture now
    resize_save_over(ds, over, max_side)
    F.mark_step(ds, "resize", {"scaled": n_s, "cropped": n_c, "upscaled": n_u, "max_side": int(max_side)})
    return (f"✔ {n_s} picture{'s' if n_s != 1 else ''} scaled down, {n_c} cropped"
            + (f", {n_u} upscaled" if n_u else "") + ". The untouched pictures are in "
            f"`{F.ORIGINALS}/`." + (f" Could not change: {'; '.join(errs[:3])}" if errs else ""))


def restore_originals(ds) -> str:
    r = F.root(ds)
    n = 0
    for p in F.images(ds):
        o = r / F.ORIGINALS / p.relative_to(r)
        if o.is_file():
            shutil.copy2(o, p)
            n += 1
    m = F.meta(ds)
    m["steps"].pop("resize", None)
    F.meta_save(ds, m)
    return f"✔ {n} pictures put back from `{F.ORIGINALS}/`."
