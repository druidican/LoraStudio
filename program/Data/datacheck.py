#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""
datacheck.py - checks the pictures of a training dataset and says which ones are not recommended, and why.

Three passes (each optional):
  1. technical (no AI, seconds): too small for the LoRA type, blurry, near-duplicates, extreme aspect ratio,
     heavy JPEG compression
  2. picture content with a local vision model in Ollama (same connection as Recaption): subject visible and in
     focus, how many people, face hidden or cut off, text / watermark, heavy filter, collage / screenshot
  3. dataset balance from the content pass: framing, outfits and places that repeat too much

The rules follow the LoRA's purpose: character (one person, face visible), style (the look; crowds are fine,
text and UI are not), concept (one clear main subject).

Not-recommended pictures can be moved, with their captions, to <dataset>/_set_aside/from_<repeat folder>/,
which no trainer reads (no <N>_ prefix), and moved back from there.

Watermarks, stamps and site names (marks added on top of a picture after it was made): for pictures the content
pass flags, the vision model is asked WHERE they are and where the main subject is. Up to two crops that cut the
marks off and keep the subject whole are proposed, next to "set aside" and "keep". The original of a cropped picture
is kept in <dataset>/_set_aside/originals/<repeat folder>/, so every choice can be changed again later.
"""
from __future__ import annotations
__version__ = "1.261026w"  # LoRA Studio version, see version.py

import base64
import io
import glob
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path

try:
    import numpy as np
except ImportError:                       # sharpness is skipped without numpy
    np = None
from PIL import Image, ImageOps

SET_ASIDE = "_set_aside"
ORIGINALS = "originals"                   # _set_aside/originals/<repeat folder>/: the uncropped pictures
MARKS_FILE = "marks.json"                 # _set_aside/marks.json: watermark boxes, crop options, your choices
IMG_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
CAPTION_EXT = (".txt", ".caption")
BAD, WARN = "bad", "warn"


# ----------------------------------------------------------------------------------------------- technical checks
def _gray(im: Image.Image, side: int = 512) -> Image.Image:
    g = ImageOps.exif_transpose(im).convert("L")
    g.thumbnail((side, side))
    return g


def sharpness(g: Image.Image) -> float | None:
    """Variance of the Laplacian on a 512 px grayscale copy: low = blurry / out of focus."""
    if np is None:
        return None
    a = np.asarray(g, dtype=np.float32)
    if a.shape[0] < 3 or a.shape[1] < 3:
        return 0.0
    lap = (a[1:-1, :-2] + a[1:-1, 2:] + a[:-2, 1:-1] + a[2:, 1:-1] - 4 * a[1:-1, 1:-1])
    return float(lap.var())


def dhash(g: Image.Image) -> int:
    small = g.resize((9, 8), Image.Resampling.LANCZOS)
    px = list(small.getdata())
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | (px[row * 9 + col] > px[row * 9 + col + 1])
    return bits


def jpeg_quality(im: Image.Image) -> int | None:
    """Estimated JPEG quality (IJG scale) from the luminance quantisation table; None for other formats."""
    q = getattr(im, "quantization", None)
    if not q or 0 not in q:
        return None
    table = list(q[0])
    if not table:
        return None
    # IJG standard luminance table, scaled: S = 5000/Q (Q<50) or 200-2Q
    std = [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29,
           51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92, 49, 64, 78, 87, 103, 121, 120,
           101, 72, 92, 95, 98, 112, 100, 103, 99]
    scale = sum(t * 100 / s for t, s in zip(table, std)) / len(std)
    qual = (200 - scale) / 2 if scale <= 100 else 5000 / scale
    return int(max(1, min(100, round(qual))))


def technical(paths: list[Path], min_side: int = 0, progress=None) -> dict[Path, dict]:
    """{path: {'w','h','sharp','hash','jpeg_q','issues':[(sev, text)]}} plus duplicate and relative-blur issues."""
    out = {}
    for i, p in enumerate(paths):
        if progress:
            progress(i, len(paths), p)
        r = {"issues": []}
        try:
            with Image.open(p) as im:
                im.load()
                w, h = ImageOps.exif_transpose(im).size
                g = _gray(im)
                r.update(w=w, h=h, sharp=sharpness(g), hash=dhash(g), jpeg_q=jpeg_quality(im))
        except Exception as e:                       # noqa: BLE001 - broken file
            r["issues"].append((BAD, f"cannot be read ({type(e).__name__}): broken or not an image"))
            out[p] = r
            continue
        short = min(w, h)
        if min_side and short < min_side * 0.6:
            r["issues"].append((BAD, f"too small: {w}x{h}, the short side is far below the {min_side} px this type "
                                     "trains at (upscaled = soft, smeary details)"))
        elif min_side and short < min_side:
            r["issues"].append((WARN, f"small: {w}x{h}, short side below {min_side} px (gets upscaled)"))
        ratio = max(w, h) / max(1, short)
        if ratio > 2.4:
            r["issues"].append((WARN, f"extreme aspect ratio {ratio:.1f}:1 (a narrow strip after bucketing; crop it)"))
        if r["jpeg_q"] is not None and r["jpeg_q"] < 60:
            r["issues"].append((WARN, f"heavy JPEG compression (quality about {r['jpeg_q']}): blocks and smearing "
                                      "can be learned"))
        out[p] = r
    # blur: absolute floor and relative to this dataset (phone photos and renders differ a lot)
    vals = sorted(v["sharp"] for v in out.values() if v.get("sharp") is not None)
    if vals:
        med = vals[len(vals) // 2]
        for p, r in out.items():
            s = r.get("sharp")
            if s is None:
                continue
            if s < max(25.0, med * 0.2):
                r["issues"].append((BAD, f"blurry / out of focus (sharpness {s:.0f}, dataset median {med:.0f})"))
            elif s < max(60.0, med * 0.4):
                r["issues"].append((WARN, f"soft focus (sharpness {s:.0f}, dataset median {med:.0f})"))
    # near-duplicates: keep the larger / sharper one of each pair
    items = [(p, r) for p, r in out.items() if "hash" in r]
    for a in range(len(items)):
        pa, ra = items[a]
        for b in range(a + 1, len(items)):
            pb, rb = items[b]
            if bin(ra["hash"] ^ rb["hash"]).count("1") <= 5:
                score = lambda r: (r["w"] * r["h"], r.get("sharp") or 0)
                worse, better = (pa, pb) if score(ra) < score(rb) else (pb, pa)
                note = (WARN, f"near-duplicate of {better.parent.name}/{better.name} (that one is larger / sharper)")
                if note not in out[worse]["issues"]:
                    out[worse]["issues"].append(note)
    return out


# ----------------------------------------------------------------------------------------------- Ollama content pass
CONTENT_PROMPT = """Look at this photo for an image-model training set and answer ONLY with a JSON object:
{"people": <number of people visible>,
 "main_subject_clear": <true if one main subject is clearly visible and not tiny>,
 "in_focus": <true if the main subject is sharp>,
 "face_visible": <true / false / null when there is no person>,
 "face_problem": "<'' or what hides or cuts the face: sunglasses, mask, hand, turned away, cropped, too small, motion blur>",
 "text_or_watermark": <true if readable text, a watermark, logo or caption is on the image>,
 "heavy_filter": <true if a strong filter, beauty retouch, sticker or heavy edit is applied>,
 "collage_or_screenshot": <true if it is a collage, a screenshot with app interface, or a photo of a screen>,
 "framing": "<close-up | upper body | full body | wide | other>",
 "part_only": "<'' or, when the picture shows only a PART of the person or character on purpose, which part: hands, tail, horns, back, feet, a tattoo, ...>",
 "outfit": "<2-5 words for the clothing, '' if none>",
 "place": "<2-4 words for the place or background>",
 "face_size": "<none | tiny | small | medium | large: how big the main subject's face is in the picture>",
 "features_visible": <true if what identifies the subject can be seen: face, hair, body shape, skin, marks; false if clothing, pose, objects or the angle hide nearly all of it (a huge dress that hides the body, a pose with the face and body turned away or covered)>,
 "teaches": <1-5: how much a model can learn about the main subject from this picture. 1 = almost nothing (the subject is tiny, mostly hidden, cut off, blurred or lost in clutter, or the picture is mostly empty background); 3 = usable; 5 = the subject is clearly shown with its details>,
 "unique_detail": "<'' or a visible detail of the subject that this picture shows well and that is easy to miss elsewhere: a tattoo, a scar, jewellery, a pattern, an accessory>",
 "note": "<anything else that makes it a poor training picture, '' if none>"}"""


def content(url: str, model: str, img: Path, timeout: float = 600.0) -> dict:
    """One image through the Ollama vision model; a dict with the fields above (missing ones absent)."""
    import captioner as CAP                        # same connection and image handling as Recaption
    r = CAP._req(url, "/api/chat", {
        "model": model, "stream": False, "format": "json", "keep_alive": "5m",
        "options": {"temperature": 0.0, "num_predict": 420},
        "messages": [{"role": "user", "content": CONTENT_PROMPT, "images": [CAP.image_b64(img, 896)]}],
    }, timeout=timeout)
    txt = ((r.get("message") or {}).get("content") or r.get("response") or "").strip()
    txt = re.sub(r"^```(json)?|```$", "", txt).strip()
    try:
        d = json.loads(txt)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", txt, re.S)
        d = json.loads(m.group(0)) if m else {}
    return d if isinstance(d, dict) else {}


def _true(v) -> bool:
    return v is True or (isinstance(v, str) and v.strip().lower() in ("true", "yes"))


CONCEPT_KINDS = ("object", "action", "concept")


def base_purpose(p) -> str:
    """character / style / concept (object, action and effect LoRAs follow the concept rules)."""
    return "concept" if p in CONCEPT_KINDS else (p or "character")


def teaches(d: dict):
    """The vision model's 1-5 rating of how much the picture shows of the subject; None when it gave none."""
    try:
        v = int(float(str(d.get("teaches")).strip()))
    except (TypeError, ValueError):
        return None
    return min(5, max(1, v))


def face_size(d: dict) -> str:
    v = str(d.get("face_size") or "").strip().lower()
    return v if v in ("none", "tiny", "small", "medium", "large") else ""


# parts that say nothing about who the character is, unless they show something special (claws, a tattoo …)
PLAIN_PARTS = ("hand", "hands", "finger", "fingers", "foot", "feet", "toe", "toes", "leg", "legs", "arm", "arms",
               "shoe", "shoes", "boot", "boots", "knee", "knees", "lap", "torso", "belly", "chest")


def plain_part(part: str) -> bool:
    words = re.findall(r"[a-z]+", (part or "").lower())
    return bool(words) and all(w in PLAIN_PARTS or w in ("and", "the", "both", "left", "right", "bare", "only")
                               for w in words)


def unique_detail(d: dict) -> str:
    v = re.sub(r"\s+", " ", str(d.get("unique_detail") or "").strip())
    return "" if v.lower() in ("", "none", "no", "n/a", "''", "null") else v[:60]


def content_issues(d: dict, purpose: str) -> list[tuple[str, str]]:
    """Issues from the Ollama answer, by purpose: character / style / concept."""
    purpose = base_purpose(purpose)
    out = []
    if not d:
        return [(WARN, "the vision model gave no usable answer for this picture")]
    try:
        people = int(d.get("people") or 0)
    except (TypeError, ValueError):
        people = 0
    if _true(d.get("collage_or_screenshot")):
        out.append((BAD, "collage, screenshot with app interface or a photo of a screen"))
    if _true(d.get("text_or_watermark")):
        out.append((BAD if purpose == "style" else WARN,
                    "text, watermark or logo on the picture (it gets learned" + (" as part of the style)" if purpose == "style" else ")")))
    if d.get("in_focus") is False or str(d.get("in_focus")).lower() == "false":
        out.append((BAD if purpose != "style" else WARN, "main subject not in focus"))
    if purpose in ("character", "concept") and (d.get("main_subject_clear") is False
                                                or str(d.get("main_subject_clear")).lower() == "false"):
        out.append((BAD, "no clear main subject (too small, hidden or not there)"))
    if purpose == "character":
        if people > 1:
            out.append((BAD, f"{people} people: the LoRA can mix the person up with the others (crop to one person)"))
        elif people == 0:
            out.append((BAD, "no person visible"))
        fv, fp = d.get("face_visible"), str(d.get("face_problem") or "").strip()
        if (fv is False or str(fv).lower() == "false") and people >= 1:
            out.append((WARN, "face not visible" + (f" ({fp})" if fp else "") + ": fine for a few pictures, not many"))
        elif fp and fp.lower() not in ("none", "no", "''", "n/a"):
            out.append((WARN, f"face partly hidden or cut: {fp}"))
        if _true(d.get("heavy_filter")):
            out.append((WARN, "strong filter or retouch: the LoRA learns the filtered look"))
        face = face_size(d)
        hidden = d.get("features_visible") is False or str(d.get("features_visible")).lower() == "false"
        part = str(d.get("part_only") or "").strip().lower() not in ("", "none", "no", "n/a", "''")
        if people >= 1 and not part:
            if hidden:
                out.append((BAD, "hides nearly everything about the character (clothes, pose or angle)"))
            elif face in ("tiny", "none") and (teaches(d) or 3) <= 3:
                out.append((BAD, "face too small and little else to learn from (the character fills only a small part)"))
            elif face == "tiny":
                out.append((WARN, "face very small: fine for a few full-body pictures, not many"))
    elif purpose == "concept" and _true(d.get("heavy_filter")):
        out.append((WARN, "strong filter or edit"))
    t = teaches(d) if purpose != "style" else None     # a style LoRA learns the whole picture, not one subject
    if t is not None and t <= 1:
        out.append((BAD, "teaches almost nothing about the subject (tiny, hidden, cut off or lost in the background)"))
    elif t == 2:
        out.append((WARN, "shows little of the subject"))
    note = str(d.get("note") or "").strip()
    if note and note.lower() not in ("none", "no", "n/a"):
        out.append((WARN, f"vision model: {note}"))
    return out


# ----------------------------------------------------------------------------------------------- balance
def balance(results: dict[Path, dict], purpose: str) -> list[str]:
    """Notes about the dataset as a whole, from the content answers."""
    purpose = base_purpose(purpose)
    answers = [r["content"] for r in results.values() if r.get("content")]
    n = len(answers)
    if n < 6:
        return []
    notes = []
    fr = Counter(str(a.get("framing") or "other").strip().lower() for a in answers)
    share = lambda k: fr.get(k, 0) / n
    if purpose == "character":
        if share("close-up") > 0.6:
            notes.append(f"{share('close-up'):.0%} close-ups: add upper-body and full-body pictures, or the LoRA "
                         "struggles with anything but portraits.")
        if share("full body") < 0.1:
            nfb = fr.get("full body", 0)
            notes.append(("no full-body pictures" if not nfb else f"only {nfb} full-body picture(s)")
                         + ": add some (10-20 %) so the body proportions and whole outfits are learned.")
        if share("close-up") + share("upper body") < 0.3:
            notes.append("few close-ups / upper-body pictures: the face may come out weak; add some where it is large.")
    for field, what in (("outfit", "outfit"), ("place", "place / background")):
        vals = Counter(re.sub(r"\s+", " ", str(a.get(field) or "").strip().lower()) for a in answers)
        vals.pop("", None)
        if vals:
            top, cnt = vals.most_common(1)[0]
            if cnt / n > 0.4:
                notes.append(f"the same {what} ('{top}') in {cnt} of {n} pictures: it may stick to the trigger word. "
                             "Vary it, or name it in the captions so it stays separate.")
    if purpose == "style":
        people = Counter(min(int(a.get("people") or 0), 3) if str(a.get("people") or "0").isdigit() else 0 for a in answers)
        if people.get(1, 0) / n > 0.85:
            notes.append("almost every picture shows one person: the style may only work on portraits; add scenes, "
                         "objects or landscapes in the same style.")
    return notes


# ----------------------------------------------------------------------------------------------- set aside / back
def caption_of(img: Path) -> list[Path]:
    """The caption files of a picture: IMG.txt / IMG.caption, and extra caption sets such as IMG.sentences.txt."""
    out = [img.with_suffix(e) for e in CAPTION_EXT if img.with_suffix(e).is_file()]
    try:
        out += sorted(f for f in img.parent.glob(glob.escape(img.stem) + ".*.txt") if f.is_file() and f not in out)
    except OSError:
        pass
    return out


def set_aside(dataset_root: Path, images: list[Path]) -> tuple[int, list[str]]:
    """Move images (and their captions) to <dataset>/_set_aside/from_<repeat folder>/."""
    moved, errors = 0, []
    for img in images:
        img = Path(img)
        dest = Path(dataset_root) / SET_ASIDE / f"from_{img.parent.name}"
        try:
            dest.mkdir(parents=True, exist_ok=True)
            for f in [img] + caption_of(img):
                target = dest / f.name
                if target.exists():
                    target = dest / f"{f.stem}_{os.getpid()}{f.suffix}"
                shutil.move(str(f), str(target))
            moved += 1
        except OSError as e:
            errors.append(f"{img.name}: {e}")
    return moved, errors


def set_aside_list(dataset_root: Path) -> list[Path]:
    """Set-aside pictures (not the originals of cropped pictures, which live in _set_aside/originals/)."""
    base = Path(dataset_root) / SET_ASIDE
    if not base.is_dir():
        return []
    return sorted(p for p in base.rglob("*") if p.suffix.lower() in IMG_EXT and ORIGINALS not in p.relative_to(base).parts)


def move_back(dataset_root: Path) -> tuple[int, list[str]]:
    """Every set-aside picture back to the repeat folder it came from (from_<name> -> <name>)."""
    moved, errors = 0, []
    base = Path(dataset_root) / SET_ASIDE
    for img in set_aside_list(dataset_root):
        src_dir = img.parent
        name = src_dir.name[5:] if src_dir.name.startswith("from_") else src_dir.name
        # pictures that sat directly in the dataset folder (no repeat folder yet) go back there
        dest = Path(dataset_root) if name == Path(dataset_root).name else Path(dataset_root) / name
        try:
            dest.mkdir(exist_ok=True)
            for f in [img] + caption_of(img):
                target = dest / f.name
                if target.exists():
                    errors.append(f"{f.name}: a file with that name is back in {name}/ already")
                    continue
                shutil.move(str(f), str(target))
            moved += 1
        except OSError as e:
            errors.append(f"{img.name}: {e}")
    for d in sorted(base.glob("*"), reverse=True):        # remove empty folders
        if d.name == ORIGINALS:
            continue
        try:
            d.rmdir()
        except OSError:
            pass
    try:
        base.rmdir()
    except OSError:
        pass
    return moved, errors


# LoRA Studio sets this to a folder Gradio may serve: pictures on the pages are then links to small files there,
# which the browser keeps, instead of being sent inside every page update (that made every click slow).
URL_DIR: dict = {"dir": None}


def _url_of(im, key: str) -> str:
    """A small JPEG of `im` as a link (URL_DIR set) or as a data URI."""
    d = URL_DIR["dir"]
    if d is not None:
        import hashlib
        from urllib.parse import quote
        fp = Path(d) / (hashlib.md5(key.encode()).hexdigest() + ".jpg")
        if not fp.exists():
            Path(d).mkdir(parents=True, exist_ok=True)
            tmp = fp.with_suffix(".tmp.jpg")
            im.save(tmp, "JPEG", quality=82)
            tmp.replace(fp)
        return "/gradio_api/file=" + quote(str(fp))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _sigkey(p, *extra) -> str:
    st = Path(p).stat()
    return "|".join(str(x) for x in (p, st.st_mtime, st.st_size) + extra)


def _cached_url(key: str):
    d = URL_DIR["dir"]
    if d is None:
        return None
    import hashlib
    from urllib.parse import quote
    fp = Path(d) / (hashlib.md5(key.encode()).hexdigest() + ".jpg")
    return "/gradio_api/file=" + quote(str(fp)) if fp.exists() else None


def thumb_uri(p: Path, side: int = 160) -> str:
    try:
        key = _sigkey(p, "thumb", side)
        hit = _cached_url(key)
        if hit:
            return hit
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((side, side))
            return _url_of(im, key)
    except Exception:                                    # noqa: BLE001
        return ""


# ----------------------------------------------------------------------------------------------- watermarks and stamps
MARKS_PROMPT = """This picture is {w}x{h} pixels. Find every mark that was ADDED ON TOP of the picture after it was made:
watermarks (also faint or transparent ones), website names, URLs, social media handles, logo stamps, copyright notes,
signatures, date / time stamps printed by a camera, captions or text overlays, stickers or emoji added later.
Do NOT include text that is part of the scene itself (signs, shop names, shirts, books, screens, tattoos).
Answer ONLY with JSON:
{{"marks": [{{"kind": "<watermark | site | logo | signature | date stamp | caption | sticker | other>",
             "text": "<the text of the mark, '' if none>", "box": [x1, y1, x2, y2]}}],
 "subject": [x1, y1, x2, y2]}}
Boxes are pixel coordinates in this {w}x{h} picture (x1, y1 = top-left corner, x2, y2 = bottom-right corner).
"subject" is the box around the main subject (the whole person, animal or object the picture is about).
Use "marks": [] when there are none."""
MAX_MARKS = 4


def _sent_size(img: Path, max_side: int = 1024) -> tuple[int, int, int, int]:
    """(W, H) of the picture (upright) and (w, h) of the copy sent to the vision model."""
    with Image.open(img) as im:
        im = ImageOps.exif_transpose(im)
        W, H = im.size
    f = min(1.0, max_side / max(W, H))
    return W, H, max(1, int(W * f)), max(1, int(H * f))


def _box(v, w: int, h: int):
    """A model box -> (x1, y1, x2, y2) in the sent picture, or None. Pixel, 0-1 and 0-1000 coordinates are understood."""
    try:
        x1, y1, x2, y2 = [float(t) for t in list(v)[:4]]
    except (TypeError, ValueError):
        return None
    if max(x1, y1, x2, y2) <= 1.0:
        x1, x2, y1, y2 = x1 * w, x2 * w, y1 * h, y2 * h
    elif max(x2, y2) > max(w, h) * 1.05 and max(x1, y1, x2, y2) <= 1000:
        x1, x2, y1, y2 = x1 * w / 1000, x2 * w / 1000, y1 * h / 1000, y2 * h / 1000
    x1, x2 = sorted((max(0.0, min(w, x1)), max(0.0, min(w, x2))))
    y1, y2 = sorted((max(0.0, min(h, y1)), max(0.0, min(h, y2))))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return x1, y1, x2, y2


def find_marks(url: str, model: str, img: Path, timeout: float = 600.0) -> dict:
    """{"W", "H", "marks": [{"kind", "text", "box"}], "subject": box | None}, boxes in the picture's own pixels."""
    import captioner as CAP
    W, H, w, h = _sent_size(img)
    r = CAP._req(url, "/api/chat", {
        "model": model, "stream": False, "format": "json", "keep_alive": "5m",
        "options": {"temperature": 0.0, "num_predict": 400},
        "messages": [{"role": "user", "content": MARKS_PROMPT.format(w=w, h=h), "images": [CAP.image_b64(img, 1024)]}],
    }, timeout=timeout)
    txt = ((r.get("message") or {}).get("content") or r.get("response") or "").strip()
    txt = re.sub(r"^```(json)?|```$", "", txt).strip()
    try:
        d = json.loads(txt)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", txt, re.S)
        d = json.loads(m.group(0)) if m else {}
    sx, sy = W / w, H / h
    up = lambda b: [round(b[0] * sx), round(b[1] * sy), round(b[2] * sx), round(b[3] * sy)] if b else None
    marks = []
    for m in (d.get("marks") or []) if isinstance(d, dict) else []:
        if not isinstance(m, dict):
            continue
        b = _box(m.get("box") or m.get("bbox") or m.get("bbox_2d"), w, h)
        if b and (b[2] - b[0]) * (b[3] - b[1]) < 0.6 * w * h:        # a "mark" covering most of the picture is no mark
            marks.append({"kind": str(m.get("kind") or "mark")[:20], "text": str(m.get("text") or "")[:60], "box": up(b)})
    subject = _box(d.get("subject"), w, h) if isinstance(d, dict) else None
    return {"W": W, "H": H, "marks": marks[:MAX_MARKS], "subject": up(subject)}


def _describe_cut(W, H, r) -> str:
    parts = []
    for name, cut, full in (("left", r[0], W), ("top", r[1], H), ("right", W - r[2], W), ("bottom", H - r[3], H)):
        if cut > 0:
            parts.append(f"{round(100 * cut / full)}% {name}")
    return "cuts " + ", ".join(parts) if parts else "no cut"


def crop_options(W: int, H: int, marks: list, subject, min_side: int = 512) -> list[dict]:
    """Up to two crops (x0, y0, x1, y1) that cut every mark off and keep the subject whole, biggest first;
    the second one keeps the original shape best. [] when no crop works (marks over the middle, too small after)."""
    from itertools import product
    if not marks:
        return []
    m = max(8, round(0.025 * max(W, H)))                           # safety margin: model boxes are approximate
    boxes = [(max(0, b[0] - m), max(0, b[1] - m), min(W, b[2] + m), min(H, b[3] + m)) for b in (x["box"] for x in marks)]
    subj = None
    if subject:
        t = 0.02
        subj = (subject[0] + t * W, subject[1] + t * H, subject[2] - t * W, subject[3] - t * H)
    min_side = max(256, int(min_side or 0))
    seen, cands = set(), []
    for sides in product("lrtb", repeat=len(boxes)):
        x0, y0, x1, y1 = 0, 0, W, H
        for (bx0, by0, bx1, by1), sd in zip(boxes, sides):
            if bx1 <= x0 or bx0 >= x1 or by1 <= y0 or by0 >= y1:
                continue                                           # already outside
            if sd == "l":
                x0 = max(x0, bx1)
            elif sd == "r":
                x1 = min(x1, bx0)
            elif sd == "t":
                y0 = max(y0, by1)
            else:
                y1 = min(y1, by0)
        r = (int(x0), int(y0), int(x1), int(y1))
        if r in seen or r[2] - r[0] < min_side or r[3] - r[1] < min_side:
            continue
        seen.add(r)
        if any(not (bx1 <= r[0] or bx0 >= r[2] or by1 <= r[1] or by0 >= r[3]) for bx0, by0, bx1, by1 in boxes):
            continue
        if subj:                                                   # most of the subject stays (its box is approximate)
            sx = max(0, min(r[2], subj[2]) - max(r[0], subj[0]))
            sy = max(0, min(r[3], subj[3]) - max(r[1], subj[1]))
            sa = max(1.0, (subj[2] - subj[0]) * (subj[3] - subj[1]))
            if sx * sy / sa < 0.85 or (subj[1] < r[1] - 0.04 * H):  # and never cut off the top (the head)
                continue
        area = (r[2] - r[0]) * (r[3] - r[1]) / (W * H)
        if area < 0.3:
            continue
        cands.append((area, r))
    if not cands:
        return []
    cands.sort(key=lambda c: -c[0])
    out = [cands[0]]
    shape = W / H
    rest = [c for c in cands[1:] if _iou(c[1], out[0][1]) < 0.9]
    if rest:
        out.append(min(rest, key=lambda c: (abs((c[1][2] - c[1][0]) / (c[1][3] - c[1][1]) - shape), -c[0])))
    return [{"id": "AB"[i], "box": list(r), "keep": round(100 * a), "what": _describe_cut(W, H, r),
             "size": f"{r[2] - r[0]}x{r[3] - r[1]}"} for i, (a, r) in enumerate(out)]


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua else 0.0


# ---- the marks file and applying a choice
def marks_load(dataset_root: Path) -> dict:
    try:
        return json.loads((Path(dataset_root) / SET_ASIDE / MARKS_FILE).read_text(encoding="utf-8"))
    except Exception:                                    # noqa: BLE001
        return {}


def marks_save(dataset_root: Path, data: dict):
    f = Path(dataset_root) / SET_ASIDE / MARKS_FILE
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(data, indent=1), encoding="utf-8")


def rel(dataset_root: Path, p: Path) -> str:
    return Path(p).relative_to(dataset_root).as_posix()


def original_of(dataset_root: Path, relpath: str) -> Path:
    return Path(dataset_root) / SET_ASIDE / ORIGINALS / relpath


def reconcile(dataset_root: Path, data: dict) -> dict:
    """Bring the recorded choices in line with the files (Move all back, files moved by hand, ...)."""
    root = Path(dataset_root)
    for rp, e in list(data.items()):
        here, orig = root / rp, original_of(root, rp)
        ch = e.get("choice") or ""
        if ch in ("A", "B") and not orig.exists():
            e["choice"] = "keep" if here.exists() else ""
        elif ch == "aside" and here.exists():
            e["choice"] = "keep"
        elif not ch and not here.exists() and not orig.exists():
            data.pop(rp)
    return data


def _save_like(im: Image.Image, src: Path, dest: Path, info: dict):
    ext = dest.suffix.lower()
    kw = {}
    if info.get("icc_profile"):
        kw["icc_profile"] = info["icc_profile"]
    if ext in (".jpg", ".jpeg"):
        im.convert("RGB").save(dest, "JPEG", quality=95, subsampling=0, **kw)
    elif ext == ".webp":
        im.save(dest, "WEBP", quality=95, **kw)
    elif ext == ".png":
        im.save(dest, "PNG", **kw)
    else:
        im.save(dest, **kw)


def _back_to_original(root: Path, rp: str, e: dict) -> str | None:
    """Undo the current choice: the original picture back at its place. Returns an error or None."""
    here, orig = root / rp, original_of(root, rp)
    ch = e.get("choice") or ""
    try:
        if ch in ("A", "B", "R", "P", "C") and orig.exists():
            shutil.move(str(orig), str(here))
        elif ch == "aside":
            src = Path(e.get("aside_path") or "")
            src = root / src if src and not src.is_absolute() else src
            if not src or not Path(src).exists():
                return f"{rp}: the set-aside picture is gone from _set_aside"
            here.parent.mkdir(parents=True, exist_ok=True)
            for f in [Path(src)] + caption_of(Path(src)):
                shutil.move(str(f), str(here.parent / (here.stem + f.suffix if f.suffix.lower() in CAPTION_EXT else here.name)))
        e["choice"] = "keep" if ch else ""
    except OSError as ex:
        return f"{rp}: {ex}"
    return None


def apply_choice(dataset_root: Path, rp: str, e: dict, choice: str, mask: Path | None = None,
                 box: list | None = None) -> str | None:
    """choice: "A" / "B" (crop), "R" (remove: paint over with the picture tools), "aside", "keep". The original is
    kept; returns an error or None."""
    root = Path(dataset_root)
    if choice not in ("P", "C") and ((e.get("choice") or "keep") == choice or (choice == "keep" and not e.get("choice"))):
        e["choice"] = choice
        return None
    err = _back_to_original(root, rp, e)
    if err:
        return err
    here, orig = root / rp, original_of(root, rp)
    try:
        if choice in ("A", "B"):
            opt = next((o for o in e.get("options") or [] if o["id"] == choice), None)
            if not opt:
                return f"{rp}: there is no crop {choice}"
            orig.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(here), str(orig))
            with Image.open(orig) as im:
                info = dict(im.info)
                im = ImageOps.exif_transpose(im)
                _save_like(im.crop(tuple(opt["box"])), orig, here, info)
        elif choice == "R":
            import pictools
            W, H = e.get("W") or 1, e.get("H") or 1
            area = sum((m["box"][2] - m["box"][0]) * (m["box"][3] - m["box"][1]) for m in e.get("marks") or [])
            if area > 0.2 * W * H:
                return f"{rp}: the marks cover too much of the picture to paint over"
            orig.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(here), str(orig))
            try:
                pictools.inpaint(orig, here, [m["box"] for m in e.get("marks") or []])
            except Exception as ex:                      # noqa: BLE001
                shutil.move(str(orig), str(here))
                return f"{rp}: could not paint over the marks ({ex})"
            try:
                pictools.keep_cutout(root, rp, orig, e.get("marks") or [])
            except Exception:                            # noqa: BLE001
                pass
        elif choice == "C":                              # your own crop
            if not box:
                return f"{rp}: no crop box"
            orig.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(here), str(orig))
            with Image.open(orig) as im:
                info = dict(im.info)
                im = ImageOps.exif_transpose(im)
                _save_like(im.crop(tuple(int(v) for v in box)), orig, here, info)
            e["custom_box"] = [int(v) for v in box]
        elif choice == "P":                              # painted over by you, removed by the picture tools
            import pictools
            if not mask:
                return f"{rp}: nothing was painted"
            orig.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(here), str(orig))
            try:
                r = pictools.inpaint(orig, here, [], mask=mask)
                if not r.get("ok", True):
                    raise RuntimeError(r.get("error") or "failed")
            except Exception as ex:                      # noqa: BLE001
                shutil.move(str(orig), str(here))
                return f"{rp}: could not paint over it ({ex})"
            try:
                from PIL import Image as _I
                with _I.open(mask) as mk:
                    with Image.open(orig) as o:
                        W, H = ImageOps.exif_transpose(o).size
                    bb = mk.convert("L").resize((W, H)).point(lambda v: 255 if v > 20 else 0).getbbox()
                if bb:
                    pictools.keep_cutout(root, rp, orig, [{"box": list(bb), "kind": "painted", "text": ""}])
            except Exception:                            # noqa: BLE001
                pass
        elif choice == "aside":
            dest = root / SET_ASIDE / f"from_{here.parent.name}"
            target = dest / here.name
            moved, errs = set_aside(root, [here])
            if errs:
                return errs[0]
            e["aside_path"] = rel(root, target if target.exists() else next(dest.glob(here.stem + "*"), target))
        e["choice"] = choice
    except OSError as ex:
        return f"{rp}: {ex}"
    return None


def marked_preview(img: Path, e: dict, side: int = 260) -> str:
    """Thumbnail with the marks in red and the subject in green, as a data URI."""
    from PIL import ImageDraw
    try:
        key = _sigkey(img, "marked", side, json.dumps([e.get("subject"), [m.get("box") for m in e.get("marks") or []]]))
        hit = _cached_url(key)
        if hit:
            return hit
        with Image.open(img) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            W, H = im.size
            im.thumbnail((side, side))
            f = im.size[0] / W
            d = ImageDraw.Draw(im)
            if e.get("subject"):
                d.rectangle([v * f for v in e["subject"]], outline=(60, 200, 120), width=2)
            for m in e.get("marks") or []:
                d.rectangle([v * f for v in m["box"]], outline=(240, 60, 50), width=3)
            return _url_of(im, key)
    except Exception:                                    # noqa: BLE001
        return ""


def crop_preview(img: Path, box, side: int = 200) -> str:
    try:
        key = _sigkey(img, "crop", side, list(box))
        hit = _cached_url(key)
        if hit:
            return hit
        with Image.open(img) as im:
            im = ImageOps.exif_transpose(im).convert("RGB").crop(tuple(box))
            im.thumbnail((side, side))
            return _url_of(im, key)
    except Exception:                                    # noqa: BLE001
        return ""
