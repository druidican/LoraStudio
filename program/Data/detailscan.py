# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Detailed scan (character datasets, on per dataset in step 4): after the sort, the dataset is looked at as a whole.

  pass 3     the Ollama model is asked once more per picture: angle, pose, expression, light, where the face is and a
             detail worth a close-up (cached, like the sort)
  coverage   framings, angles, poses, expressions, outfits, places and light are counted over the pictures the sort
             keeps; what is missing or rare is a gap
  suggests   per picture, with the reason, nothing is done until you tick it (step 5):
               add       a left-out picture that fills a gap (it gets an 📌 Always pin)
               upscale   a kept picture smaller than the LoRA type trains at (your ComfyUI upscale workflow)
               sharpen   a kept picture that is soft (the same workflow at the same size)
               crop      a small face or detail in a big picture becomes a NEW picture next to it, upscaled when small
  originals  stay in _originals/; new crops get a copy there too, so 'Put the originals back' knows them.

It judges how well a picture shows things (visible, sharp, big enough, which angle), never whose face it is.
"""
from __future__ import annotations
__version__ = "1.101126"  # LoRA Studio version, see version.py

import json
import re
import time
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps

import datacheck as DC
import dsflow as F

DIMS = {"framing": ("Framing", ["close-up", "upper body", "full body"]),
        "angle": ("Angle", ["front", "three-quarter", "side", "back"]),
        "pose": ("Pose", ["standing", "sitting"]),
        "expression": ("Expression", ["neutral", "smiling"]),
        "lighting": ("Light", []), "outfit": ("Outfit", []), "place": ("Place", [])}
VARIETY = {"lighting": 2, "outfit": 3, "place": 3}       # at least this many different ones
RARE = 2                                                  # a wanted value seen fewer times than this is a gap
FACE_SMALL = 0.06                                         # a face under 6 % of the picture is worth its own crop
DETAIL_SMALL = 0.10
MIN_CROP = 128                                            # smaller crops are not worth it, even upscaled
KEPT = ("good", "detail")

PROMPT = """Look at this picture of a character for an image-model training set. The picture is {w}x{h} pixels.
Answer ONLY with a JSON object:
{{"angle": "<front | three-quarter | side | back | from above | from below>: how the main subject faces the camera",
 "pose": "<standing | sitting | lying | kneeling | walking | running | action | other>",
 "expression": "<neutral | smiling | laughing | serious | surprised | sad | angry | other | not visible>",
 "lighting": "<daylight | indoor | studio | night | low light | dramatic | other>",
 "face_box": <[x1, y1, x2, y2] in pixels around the main subject's face, or null when no face is visible>,
 "detail": "<'' or one visible detail of the subject worth a close-up: a tattoo, jewellery, a pattern, an accessory>",
 "detail_box": <[x1, y1, x2, y2] in pixels around that detail, or null>}}"""


def enabled(ds) -> bool:
    m = F.meta(ds)
    return bool(m.get("detailed_scan")) and DC.base_purpose(m.get("purpose")) == "character"


def set_enabled(ds, on: bool):
    m = F.meta(ds)
    m["detailed_scan"] = bool(on)
    F.meta_save(ds, m)


def result(ds) -> dict | None:
    return F._load(ds, "detail.json")


# --------------------------------------------------------------------------- asking the vision model
def ask(url: str, model: str, img: Path, timeout: float = 600.0) -> dict:
    import captioner as CAP
    W, H, w, h = DC._sent_size(img)
    r = CAP._req(url, "/api/chat", {
        "model": model, "stream": False, "format": "json", "keep_alive": "5m",
        "options": {"temperature": 0.0, "num_predict": 300},
        "messages": [{"role": "user", "content": PROMPT.format(w=w, h=h), "images": [CAP.image_b64(img, 1024)]}],
    }, timeout=timeout)
    txt = re.sub(r"^```(json)?|```$", "", ((r.get("message") or {}).get("content") or r.get("response") or "").strip())
    try:
        d = json.loads(txt)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", txt, re.S)
        d = json.loads(m.group(0)) if m else {}
    if not isinstance(d, dict):
        d = {}
    sx, sy = W / w, H / h
    out = {k: F._norm(d.get(k)) for k in ("angle", "pose", "expression", "lighting")}
    for k in ("face_box", "detail_box"):
        b = DC._box(d.get(k), w, h)
        out[k] = [round(b[0] * sx), round(b[1] * sy), round(b[2] * sx), round(b[3] * sy)] if b else None
    out["detail"] = DC.unique_detail({"unique_detail": d.get("detail")})
    out["W"], out["H"] = W, H
    return out


def _cache_file(ds) -> Path:
    return F.root(ds) / ".lsflow" / "detail_cache.json"


# --------------------------------------------------------------------------- the scan
def _crop_box(box, W, H, grow: float, side_min: int):
    """A crop around a box, `grow` times its size (a little taller than wide), inside the picture."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    s = max(max(x2 - x1, y2 - y1) * grow, side_min)
    w_, h_ = min(s, W), min(s * 1.15, H)
    left = max(0.0, min(W - w_, cx - w_ / 2))
    top = max(0.0, min(H - h_, cy - h_ * 0.45))
    return [int(left), int(top), int(left + w_), int(top + h_)]


def coverage(entries: list[dict]) -> tuple[dict, list[str], dict]:
    """(counts per dimension, gap lines, {dim: [missing or rare values]})"""
    counts = {d: Counter(e[d] for e in entries if e.get(d)) for d in DIMS}
    gaps, want = [], {}
    for d, (label, wanted) in DIMS.items():
        c = counts[d]
        miss = [v for v in wanted if c.get(v, 0) < RARE]
        if miss:
            want[d] = miss
            gaps.append(f"{label}: " + ", ".join(f"{v} ({c.get(v) or 'none'})" for v in miss))
        if d in VARIETY and len(c) < VARIETY[d]:
            want.setdefault(d, ["*new*"])
            gaps.append(f"{label}: only {len(c)} different ({', '.join(c) or 'none'})")
    return counts, gaps, want


def run(ds, url: str | None = None, model: str | None = None):
    """Generator: progress lines, then 'done'. Needs the sort result (step 4) and Ollama."""
    S = F.sg()
    sr = F.sort_result(ds)
    if not sr or not sr.get("pics"):
        yield "The detailed scan needs the sort first."
        return
    url = (url or "").strip() or S.oc_settings()[0]
    model = model or S.oc_settings()[1]
    m = F.meta(ds)
    need = F.min_side(m["types"]) or 896                  # smaller than this is too small for the LoRA type
    side = 512 if need < 600 else 1024                     # what an upscale aims at: the size the type trains at
    pins = F.pins(ds)
    cache_p = _cache_file(ds)
    try:
        cache = json.loads(cache_p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cache = {}
    t0, asked = time.time(), 0
    pics = {p: e for p, e in sr["pics"].items() if Path(p).is_file()}
    entries = {}
    for i, (p, e) in enumerate(pics.items(), 1):
        if F.STOP.is_set():
            yield "stopped"
            break
        if e.get("cat") == "out" and "small" not in (e.get("why") or ""):
            continue                                      # broken or blurry: nothing to win
        pp = Path(p)
        st = pp.stat()
        key = f"{pp.name}|{st.st_size}|{int(st.st_mtime)}|{model}"
        d = cache.get(key)
        if d is None:
            left = f", about {S.fmt((time.time() - t0) / max(asked, 1) * (len(pics) - i + 1))} left" if asked else ""
            yield f"Detailed scan (Ollama): {i} of {len(pics)}{left}"
            try:
                d = ask(url, model, pp)
            except Exception as ex:                       # noqa: BLE001
                yield f"Ollama stopped answering: {ex}"
                break
            cache[key] = d
            asked += 1
            if asked % 5 == 0:
                cache_p.write_text(json.dumps(cache), encoding="utf-8")
        entries[p] = {**d, "framing": e.get("framing") or "", "outfit": e.get("outfit") or "",
                      "place": e.get("place") or "", "cat": e.get("cat"), "why": e.get("why") or "",
                      "w": e.get("w") or d.get("W"), "h": e.get("h") or d.get("H"), "warn": e.get("warn") or [],
                      "face": e.get("face") or "", "pin": F.pin_of(pins, p)}
    cache_p.write_text(json.dumps(cache), encoding="utf-8")
    kept = [x for p, x in entries.items() if (x["cat"] in KEPT or x["pin"] == "keep") and x["pin"] != "never"]
    counts, gaps, want = coverage(kept)
    items = []
    for p, x in entries.items():
        name = Path(p).name
        is_kept = (x["cat"] in KEPT or x["pin"] == "keep") and x["pin"] != "never"
        if not is_kept and x["pin"] != "never":
            adds = [v for d, vals in want.items() for v in ([x.get(d)] if x.get(d) else [])
                    if v in vals or ("*new*" in vals and v not in counts[d])]
            if adds:
                items.append({"kind": "add", "path": p, "label": f"{name}: add it, it shows {', '.join(adds)}",
                              "why": f"left out by the sort ({x['why'] or x['cat']}), but the selection lacks "
                                     + ", ".join(adds)})
                is_kept = True
        if not is_kept:
            continue
        short = min(x["w"] or 0, x["h"] or 0)
        if short and short < need:
            items.append({"kind": "upscale", "path": p, "target": side,
                          "label": f"{name}: upscale {short} → {side} px (×{side / short:.1f})",
                          "why": f"smaller than the {need} px the LoRA type needs (it trains at {side} px)"})
        elif any("soft focus" in w for w in x["warn"]):
            items.append({"kind": "sharpen", "path": p, "target": short,
                          "label": f"{name}: sharpen (same size)", "why": "slightly soft"})
        W, H = x.get("W") or x["w"], x.get("H") or x["h"]
        for k, small, what, grow in (("face_box", FACE_SMALL, "face", 2.4), ("detail_box", DETAIL_SMALL, "detail", 1.8)):
            b = x.get(k)
            if not b or not W or not H:
                continue
            area = (b[2] - b[0]) * (b[3] - b[1]) / (W * H)
            label = "face" if what == "face" else re.sub(r"^(an?|the)\s+", "", x.get("detail") or "detail", flags=re.I)
            if area >= small or (what == "detail" and not x.get("detail")):
                continue
            box = _crop_box(b, W, H, grow, MIN_CROP)
            cs = min(box[2] - box[0], box[3] - box[1])
            if cs < MIN_CROP:
                continue
            up = f", upscaled {cs} → {side} px" if cs < need else ""
            items.append({"kind": "crop", "path": p, "box": box, "what": what, "target": side,
                          "label": f"{name}: new picture of the {label} ({cs} px{up})",
                          "why": f"the {label} is only {max(area, 0.01):.0%} of the picture", "need": need})
    for n, it in enumerate(items):
        it["id"] = f"{n}:{it['kind']}:{Path(it['path']).name}"
    res = {"at": time.strftime("%Y-%m-%d %H:%M"), "side": side, "counts": {d: dict(c) for d, c in counts.items()},
           "gaps": gaps, "items": items, "kept": len(kept), "seconds": round(time.time() - t0, 1)}
    F._save(ds, "detail.json", res)
    yield "done"


# --------------------------------------------------------------------------- doing what you ticked
def _new_name(src: Path, what: str) -> Path:
    i = 1
    while True:
        p = src.with_name(f"{src.stem}_{what}{i}.png")
        if not p.exists():
            return p
        i += 1


def _enlarge(src: Path, dst: Path, target: int) -> str:
    """Upscale with the ComfyUI workflow, else LoRA Studio's own picture tools; '' when done, else why not."""
    import comfyupscale as U
    why = U.why_not()
    if not why:
        U.upscale(src, dst, target)
        return ""
    if F.tools_ok():
        import pictools
        pictools.upscale(src, dst, target)
        return ""
    return why


def apply(ds, ids: list[str]):
    """Generator: progress lines, then a summary line starting with ✔ or ⚠."""
    res = result(ds) or {}
    todo = [it for it in res.get("items") or [] if it["id"] in set(ids or [])]
    if not todo:
        yield "⚠ Nothing ticked."
        return
    import comfyupscale as U
    done, skipped, made = [], [], []
    F.STOP.clear()
    for i, it in enumerate(todo, 1):
        if F.STOP.is_set():
            skipped.append(f"stopped before {len(todo) - i + 1} more")
            break
        p = Path(it["path"])
        if not p.is_file():
            skipped.append(f"{p.name}: the file is gone")
            continue
        yield f"{i} of {len(todo)}: {it['label']}"
        try:
            if it["kind"] == "add":
                F.set_pin(ds, str(p), "keep")
                done.append(it["label"])
            elif it["kind"] in ("upscale", "sharpen"):
                why = U.why_not()
                if why and it["kind"] == "sharpen":
                    skipped.append(f"{p.name}: {why}")
                    continue
                F.ensure_originals(ds, [p])
                why = _enlarge(p, p, int(it["target"]))
                (skipped.append(f"{p.name}: {why}") if why else done.append(it["label"]))
            elif it["kind"] == "crop":
                new = _new_name(p, it["what"])
                with Image.open(p) as im:
                    im = ImageOps.exif_transpose(im)
                    im.crop(tuple(it["box"])).convert("RGB").save(new)
                cs = min(it["box"][2] - it["box"][0], it["box"][3] - it["box"][1])
                note = ""
                F.ensure_originals(ds, [new])               # the crop as cut, before any upscale
                if cs < int(it.get("need") or it["target"]):
                    why = _enlarge(new, new, int(it["target"]))
                    note = f" (not upscaled: {why})" if why else ""
                F.set_pin(ds, str(new), "keep")
                made.append(new.name)
                done.append(it["label"] + note)
        except Exception as ex:                           # noqa: BLE001
            skipped.append(f"{p.name}: {ex}")
    lines = [f"✔ Done {len(done)} of {len(todo)}."]
    if made:
        lines.append(f"New pictures (pinned 📌 Always, no caption yet; step 7 *Only pictures without a caption* "
                     f"writes theirs): {', '.join(made)}.")
    if skipped:
        lines.append("⚠ Not done: " + "; ".join(skipped))
    yield "\n\n".join(lines)
