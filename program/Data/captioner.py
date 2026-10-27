#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""
captioner.py - writes natural-language training captions with a local vision model in Ollama.

Used by LoRA Studio (Create Data Launch Script > Recaption with Ollama). For a character / person LoRA the captions
describe what CHANGES from picture to picture (clothes, pose, place, light, framing) and leave out what is ALWAYS the
person (face, eye and hair colour, freckles, tattoos, body): those traits then end up in the trigger word.

Only the Python standard library is used (urllib); Pillow is used to shrink the image before it is sent, when present.
Run it on its own to test the connection:   python3 captioner.py [--url http://127.0.0.1:11434] [image.jpg]
"""
from __future__ import annotations
__version__ = "1.261026v"  # LoRA Studio version, see version.py

import base64
import io
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_URL = "http://127.0.0.1:11434"
# Names of vision models, for Ollama versions that do not report "capabilities" in /api/show
VISION_HINTS = ("vl", "vision", "llava", "bakllava", "moondream", "minicpm-v", "gemma3", "llama4", "mistral-small3",
                "granite3.2-vision", "joycaption")
DEFAULT_LEAVE_OUT = ("face, facial features, eye colour, hair colour, freckles, tattoos, skin, body shape, breasts, "
                     "age, ethnicity, makeup")

# What the LoRA is for: "character" (default), "style" or "concept". Set by the Datasets steps (and the launch script
# page's Purpose) so every caption writer uses the right prompt.
PURPOSE = {"value": "character", "words": ""}
CONCEPT_KINDS = ("object", "action", "concept")      # Object / item, Action / pose, Concept / effect
CONCEPT_WHAT = {
    "object": ("an OBJECT (an item)", "the object itself: its shape, colours, material, parts and decorations",
               "who holds, wears or uses it, the clothes, the pose, the place, the light and the camera angle"),
    "action": ("an ACTION or POSE", "the pose or action itself: how the body, arms, legs and hands are placed or move",
               "the person (hair, clothes, expression), what they hold, the place, the light and the camera angle"),
    "concept": ("a CONCEPT or EFFECT", "the effect itself and how it looks",
                "everything else: the people, their clothes and pose, objects, the place, the light"),
}
# For a style LoRA the style itself is what the trigger word learns: these are never named in the captions.
STYLE_LEAVE_OUT = ("art style, medium, painting, drawing, illustration, sketch, watercolor, oil painting, digital art, "
                   "anime style, cartoon, line art, linework, brush strokes, cel shading, shading, color palette, "
                   "colour palette, texture, artist")

# ---- caption style, length and detail -------------------------------------------------------------------------------
STYLES = ("sentences", "tags")
# Length 1..5: words for sentences, number of tags for tag lists
LENGTH_LABELS = {1: "Very short", 2: "Short", 3: "Medium", 4: "Long", 5: "Very long"}
SENT_WORDS = {1: 20, 2: 35, 3: 55, 4: 80, 5: 110}
TAG_COUNT = {1: 8, 2: 14, 3: 20, 4: 28, 5: 40}
# Detail 1..4: which aspects of the picture are described (each level adds to the one before)
DETAIL_LABELS = {1: "Basic", 2: "Normal", 3: "Detailed", 4: "Everything"}
DETAIL_ASPECTS = {
    1: ["clothing", "pose and action", "framing (close-up, upper body, full body, selfie, mirror selfie)"],
    2: ["facial expression and where the person looks", "the place and the main background"],
    3: ["light and time of day", "camera angle", "accessories, and the colours and materials of the clothes"],
    4: ["objects in the background", "other people or animals", "photo style (candid, studio, phone photo, depth of field)",
        "composition"],
}


def aspects(detail: int) -> list[str]:
    return [a for lvl in range(1, max(1, min(4, int(detail))) + 1) for a in DETAIL_ASPECTS[lvl]]


def _concept_prompt(kind: str, style: str, trigger: str, leave_out: str, length: int, detail: int) -> str:
    """An object, action or effect LoRA: the trigger word stands for it, so it is named, never described."""
    what_is, never_desc, describe = CONCEPT_WHAT.get(kind, CONCEPT_WHAT["concept"])
    words_ = f" ({leave_out})" if (leave_out or "").strip() else ""
    never = (f"- {trigger} is {what_is}{words_}. Write {trigger} where it is, and NEVER describe {never_desc}: "
             "the trigger word learns that.\n")
    if style == "tags":
        n = TAG_COUNT.get(int(length), 20)
        return (f"You write training captions for a LoRA that learns {what_is}, called {trigger}, in the style of booru "
                f"tags.\nAnswer with ONE line of comma-separated tags, at most {n} tags, and nothing else.\n\nRules:\n"
                f"- The first tag is exactly \"{trigger}\".\n"
                "- Short lowercase tags of one to three words. No sentences, no numbering, no quotes, no duplicates.\n"
                f"- Tag {describe}, the most important first.\n{never}")
    words = SENT_WORDS.get(int(length), 55)
    nsent = "one sentence" if words <= 30 else "one or two sentences" if words <= 60 else "two to four sentences"
    return (f"You write training captions for a LoRA that learns {what_is}, called {trigger}. The people and places "
            f"change from picture to picture.\nWrite ONE caption in plain English: {nsent}, at most {words} words, and "
            f"nothing else.\n\nRules:\n- Mention {trigger} once, where it is in the picture.\n- Describe {describe}.\n"
            f"{never}- No lists, no tags, no quotes, no line breaks, no opinions, and do not start with \"The image\" or "
            "\"This picture\".")


def _style_prompt(style: str, trigger: str, leave_out: str, length: int, detail: int) -> str:
    """A STYLE LoRA: describe everything that is in the picture, never the style, medium or technique."""
    asp = [a for a in aspects(max(3, int(detail))) if "photo style" not in a]
    what = "\n".join(f"  - {a}" for a in ["who or what is shown: people with their hair, eyes, clothes, expression and "
                                           "pose; animals; objects"] + asp)
    never = (f"- NEVER name the art style, the medium, the technique or the artist: {leave_out}. They all belong to "
             f"{trigger}, which stands for the style.\n")
    if style == "tags":
        n = TAG_COUNT.get(int(length), 20)
        return (f"You write training captions for a STYLE LoRA in the style of booru tags. Every picture is in the same "
                f"art style, called {trigger}.\nAnswer with ONE line of comma-separated tags, at most {n} tags, and "
                f"nothing else.\n\nRules:\n- The first tag is exactly \"{trigger}\".\n"
                "- Short lowercase tags of one to three words. No sentences, no numbering, no quotes, no duplicates.\n"
                f"- Tag what is IN the picture, the most important first:\n{what}\n{never}")
    words = SENT_WORDS.get(int(length), 55)
    nsent = "one sentence" if words <= 30 else "one or two sentences" if words <= 60 else "two to four sentences"
    return (f"You write training captions for a STYLE LoRA. Every picture is in the same art style, called {trigger}.\n"
            f"Write ONE caption in plain English: {nsent}, at most {words} words, and nothing else.\n\nRules:\n"
            f"- Begin with \"{trigger}, \" and then describe what is IN the picture, the most important first:\n{what}\n"
            f"{never}- No lists, no tags, no quotes, no line breaks, no opinions, and do not start with \"The image\" or "
            "\"This picture\".")


def build_prompt(style: str, trigger: str, leave_out: str, length: int, detail: int, purpose: str | None = None) -> str:
    pu = purpose or PURPOSE["value"]
    if pu == "style":
        return _style_prompt(style, trigger, leave_out or STYLE_LEAVE_OUT, length, detail)
    if pu in CONCEPT_KINDS:
        return _concept_prompt(pu, style, trigger, leave_out or PURPOSE.get("words") or "", length, detail)
    what = "\n".join(f"  - {a}" for a in aspects(detail))
    never = (f"- NEVER mention these, because they always belong to {trigger}: {leave_out}.\n"
             "  Do not describe the face or the body. Do not guess age, nationality or ethnicity.\n"
             "- Hairstyle (ponytail, bun, wet hair) may be mentioned, hair colour may not.\n") if (leave_out or "").strip() else ""
    if style == "tags":
        n = TAG_COUNT.get(int(length), 20)
        return (f"You write training captions for an image model in the style of booru tags. The subject is called {trigger}.\n"
                f"Answer with ONE line of comma-separated tags, at most {n} tags, and nothing else.\n\nRules:\n"
                f"- The first tag is exactly \"{trigger}\".\n"
                "- Short lowercase tags of one to three words (for example: white bikini, sitting, beach chair, from side, "
                "sunlight). No sentences, no numbering, no quotes, no duplicates.\n"
                f"- Tag only these things, the most important first:\n{what}\n{never}")
    words = SENT_WORDS.get(int(length), 55)
    nsent = "one sentence" if words <= 30 else "one or two sentences" if words <= 60 else "two to four sentences"
    return (f"You write training captions for an image model. Every picture shows the same subject, called {trigger}.\n"
            f"Write ONE caption in plain English: {nsent}, at most {words} words, and nothing else.\n\nRules:\n"
            f"- Begin with \"{trigger}\" as the subject, for example \"{trigger} sits on a beach chair ...\".\n"
            f"- Describe only these things, the most important first:\n{what}\n{never}"
            "- No lists, no tags, no quotes, no line breaks, no opinions, and do not start with \"The image\" or \"This photo\".")


# Kept for callers of the first version (sentences, length 3, detail 4)
USER = "Write the caption for this photo."
USER_HINT = ("Write the caption for this photo. An automatic tagger listed these words for it; they may be wrong, so "
             "use only what you can see yourself: {tags}")


class OllamaError(RuntimeError):
    pass


def _req(url: str, path: str, payload: dict | None = None, timeout: float = 15.0) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url.rstrip("/") + path, data=data,
                                 headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace") or "{}")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:300]
        try:
            body = json.loads(body).get("error", body)
        except Exception:
            pass
        raise OllamaError(f"Ollama answered {e.code}: {body}") from None
    except (urllib.error.URLError, OSError) as e:
        raise OllamaError(f"Ollama is not reachable at {url} ({getattr(e, 'reason', e)}). "
                          "Is it running? (systemctl status ollama, or start it with: ollama serve)") from None


def list_models(url: str = DEFAULT_URL) -> list[dict]:
    """[{'name', 'vision': True/False/None, 'size_gb'}] for every model in Ollama, vision models first."""
    out = []
    for m in _req(url, "/api/tags").get("models", []):
        name = m.get("name") or m.get("model") or ""
        vision = None
        try:
            caps = _req(url, "/api/show", {"model": name}).get("capabilities")
            if isinstance(caps, list):
                vision = "vision" in caps
        except OllamaError:
            pass
        if vision is None:
            fams = " ".join((m.get("details") or {}).get("families") or []).lower()
            vision = any(h in name.lower() for h in VISION_HINTS) or "clip" in fams or "mllama" in fams
        out.append({"name": name, "vision": vision, "size_gb": round((m.get("size") or 0) / 1e9, 1)})
    out.sort(key=lambda d: (not d["vision"], d["name"]))
    return out


def image_b64(path: Path, max_side: int = 1024) -> str:
    """The image as base64 JPEG, shrunk to max_side (faster, and the model sees no more detail than that anyway)."""
    try:
        from PIL import Image, ImageOps
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((max_side, max_side))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=90)
            return base64.b64encode(buf.getvalue()).decode()
    except ImportError:
        return base64.b64encode(Path(path).read_bytes()).decode()


def _terms(leave_out: str) -> list[str]:
    return [t.strip().lower() for t in re.split(r"[,;\n]", leave_out or "") if t.strip()]


# words that betray a described trait (checked after generation; caption is retried once when one shows up)
_TRAIT_WORDS = {
    "eye colour": r"\b(blue|green|brown|grey|gray|hazel|amber)[- ]eyed?\b|\b(blue|green|brown|grey|gray|hazel) eyes\b",
    "hair colour": r"\b(blonde?|brunette|red-haired|redhead|ginger|black-haired|brown-haired|platinum|dark-haired|"
                   r"fair-haired)\b|\b(blonde?|black|brown|red|auburn|golden|dark|light|grey|gray|white|silver) hair\b",
    "freckles": r"\bfreckle", "tattoos": r"\btattoo", "breasts": r"\bbreast|\bcleavage\b",
    "ethnicity": r"\b(asian|caucasian|european|african|latina|japanese|korean|chinese|scandinavian)\b(?! (food|restaurant|style|garden))",
    "age": r"\b(young|old|elderly|teen|teenage|middle-aged|\d{2}-year-old)\b",
}


def leaks(caption: str, leave_out: str) -> list[str]:
    """Traits from the leave-out list that the caption still mentions."""
    low = caption.lower()
    hits = []
    for t in _terms(leave_out):
        pat = _TRAIT_WORDS.get(t)
        if pat:
            if re.search(pat, low):
                hits.append(t)
        elif len(t) > 3 and " " not in t and t not in ("face", "skin", "facial features", "body shape", "makeup") \
                and re.search(rf"\b{re.escape(t.rstrip('s'))}", low):
            hits.append(t)
    return hits


# trait categories a character description may be suggested from (never ethnicity, skin, age or body parts)
DESCRIBE_OK = ("hair colour", "eye colour", "freckles", "tattoos", "body shape", "facial features", "face", "makeup")


def trait_tags(text: str, leave_out: str) -> list[str]:
    """The tags of a raw tag line that name a left-out trait of a describable kind (blonde hair, blue eyes, slim,
    freckles): the material for a suggested character description."""
    out = []
    for x in re.split(r"[,\n]", text or ""):
        x = x.strip().strip('"\'.').lower().replace("_", " ")
        if not x or len(x.split()) > 4:
            continue
        hit = leaks(x, leave_out)
        if hit and all(h in DESCRIBE_OK for h in hit):
            out.append(x)
    return out


def clean(text: str, trigger: str) -> str:
    """One tidy line that contains the trigger word (put in front when the model left it out)."""
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    t = re.sub(r"^\s*(caption|answer)\s*:\s*", "", t.strip(), flags=re.I)
    t = t.strip().strip('"').strip("'").strip()
    t = re.sub(r"\s+", " ", t)
    if trigger and not re.search(rf"\b{re.escape(trigger)}\b", t, re.I):
        if PURPOSE["value"] == "character":              # only a character's trigger word is the person in the picture
            t = re.sub(r"^(a|the) (woman|man|person|girl|boy|lady|guy)\b", trigger, t, flags=re.I)
        if not re.search(rf"\b{re.escape(trigger)}\b", t, re.I):
            t = f"{trigger}, {t[:1].lower() + t[1:]}" if t else trigger
    return t


def clean_tags(text: str, trigger: str, leave_out: str, max_tags: int) -> tuple[str, list[str]]:
    """Tidy tag line: trigger first, lowercase, no duplicates, at most max_tags; tags naming a left-out trait are
    DROPPED (simpler and safer than asking again). Returns (line, dropped traits)."""
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    t = re.sub(r"^\s*(caption|tags|answer)\s*:\s*", "", t.strip(), flags=re.I)
    raw = [x.strip().strip('"\'.').lower() for x in re.split(r"[,\n]", t)]
    raw = [re.sub(r"^\s*(?:[-*•]+|\d+[.)])\s*", "", x).replace("_", " ").strip() for x in raw]   # list marks, not 1girl
    out, seen, dropped = [], set(), []
    for x in raw:
        if not x or x in seen or x == (trigger or "").lower() or len(x.split()) > 5:
            continue
        hit = leaks(x, leave_out)
        if hit:
            dropped += hit
            continue
        seen.add(x)
        out.append(x)
    out = out[:max(1, max_tags - 1)]
    return ", ".join([trigger] + out if trigger else out), sorted(set(dropped))


def caption(url: str, model: str, img: Path, trigger: str, leave_out: str = DEFAULT_LEAVE_OUT, hint_tags: str = "",
            words: int = 0, keep_alive: str = "5m", timeout: float = 600.0, style: str = "sentences",
            length: int = 3, detail: int = 4) -> dict:
    """{'caption', 'leaks': [...], 'retried': bool, 'dropped': [...]} for one image.
    style 'sentences' or 'tags'; length 1..5; detail 1..4 (see LENGTH_LABELS / DETAIL_LABELS)."""
    if words and style == "sentences":                    # old call style: caption(..., words=60)
        length = min(SENT_WORDS, key=lambda k: abs(SENT_WORDS[k] - words))
    system = build_prompt(style, trigger, leave_out, length, detail)
    user = USER_HINT.format(tags=hint_tags) if hint_tags else USER
    if style == "tags":
        user = user.replace("Write the caption", "Write the tags")
    b64 = image_b64(img)

    def ask(extra: str = "") -> str:
        r = _req(url, "/api/chat", {
            "model": model, "stream": False, "keep_alive": keep_alive,
            "options": {"temperature": 0.2, "num_predict": 400},
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user + extra, "images": [b64]}],
        }, timeout=timeout)
        msg = r.get("message") or {}
        return msg.get("content") or r.get("response") or ""

    if style == "tags":
        line, dropped = clean_tags(ask(), trigger, leave_out, TAG_COUNT.get(int(length), 20))
        retried = False
        if line.count(",") < 2:                           # nearly empty: ask once more
            retried = True
            line2, dropped2 = clean_tags(ask(" Answer only with the comma-separated tags."), trigger, leave_out,
                                         TAG_COUNT.get(int(length), 20))
            if line2.count(",") > line.count(","):
                line, dropped = line2, dropped2
        return {"caption": line, "leaks": [], "retried": retried, "dropped": dropped}

    text = clean(ask(), trigger)
    hits, retried = leaks(text, leave_out), False
    if hits or len(text) < len(trigger) + 12:
        retried = True
        again = clean(ask(f" Your first answer mentioned {', '.join(hits) or 'too little'}; write it again without "
                          f"that and follow every rule."), trigger)
        if len(leaks(again, leave_out)) < len(hits) or (not hits and len(again) > len(text)):
            text, hits = again, leaks(again, leave_out)
    return {"caption": text, "leaks": hits, "retried": retried, "dropped": []}


# ---- the example shown in the GUI (no model involved): one invented photo, described at every setting --------------
_EX_TAGS = {
    1: ["white linen shirt", "denim shorts", "sitting", "holding coffee cup", "upper body"],
    2: ["smiling", "looking at viewer", "outdoor cafe", "street"],
    3: ["sunlight", "afternoon", "from side", "sunglasses on head", "silver bracelet", "rolled-up sleeves"],
    4: ["wooden table", "potted plants", "parked bicycle", "people in background", "candid photo",
        "shallow depth of field", "rule of thirds"],
}
# sentence pieces per detail level, most important first; the first piece starts the caption
_EX_SENT = {
    1: ["{t} sits with a coffee cup in hand", "wearing a white linen shirt and denim shorts", "in an upper-body shot"],
    2: ["{t} smiles and looks at the camera", "at an outdoor cafe on a quiet street"],
    3: ["warm afternoon sunlight falls from the side", "sunglasses are pushed up on the head",
        "a thin silver bracelet sits on the wrist", "the shirt sleeves are rolled up"],
    4: ["potted plants and a parked bicycle stand in the background", "a few people walk past, blurred",
        "it is a candid phone photo with shallow depth of field", "the subject sits on the left third of the frame"],
}


# more about the SAME aspects, used when the length leaves room (a real model elaborates the same way)
_EX_MORE_TAGS = {
    1: ["loose shirt", "frayed shorts", "crossed legs", "both hands on cup"],
    2: ["head tilt", "closed-mouth smile", "cafe chairs", "cobblestone street"],
    3: ["warm lighting", "soft shadows", "eye-level shot", "beige tones"],
    4: ["menu board", "string lights", "bokeh", "vertical framing"],
}
_EX_MORE_SENT = {
    1: ["the loose shirt is half buttoned and the shorts are frayed at the hem",
        "both hands hold the cup, legs crossed on the chair"],
    2: ["the head is tilted slightly with a closed-mouth smile",
        "cafe chairs line a cobblestone street"],
    3: ["the light is warm with soft shadows, seen at eye level", "the colours are beige and white"],
    4: ["a menu board and string lights hang behind, softly out of focus", "the photo is framed vertically"],
}


def example(style: str, length: int, detail: int, trigger: str = "ohwx") -> str:
    """The invented example caption for these settings (illustrates what length and detail do).
    Detail decides WHICH aspects are described; length decides HOW MUCH is said about them."""
    t = (trigger or "ohwx").strip() or "ohwx"
    detail, length = max(1, min(4, int(detail))), max(1, min(5, int(length)))
    lv = range(1, detail + 1)
    if PURPOSE["value"] != "character":                 # style / object / action / effect: the person is described
        if style == "tags":
            tags = (["1girl", "long red hair", "green eyes"] + [x for l in lv for x in _EX_TAGS[l]]
                    + [x for l in lv for x in _EX_MORE_TAGS[l]])
            return ", ".join([t] + tags[:TAG_COUNT[length] - 1])
        subj = "a woman with long red hair and green eyes"
        body = _example_sentences(subj, length, max(detail, 3)).replace(". " + subj, ". She")
        body = body.replace(" it is a candid phone photo with shallow depth of field,", "")
        return f"{t}, {body[:1].lower()}{body[1:]}"
    if style == "tags":
        tags = [x for l in lv for x in _EX_TAGS[l]] + [x for l in lv for x in _EX_MORE_TAGS[l]]
        return ", ".join([t] + tags[:TAG_COUNT[length] - 1])
    return _example_sentences(t, length, detail)


def _example_sentences(t: str, length: int, detail: int) -> str:
    lv = range(1, detail + 1)
    groups = []                                           # each group becomes one sentence
    for l in lv:
        pieces = [x.format(t=t) for x in _EX_SENT[l]]
        groups += [pieces] if l <= 2 else [pieces[k:k + 2] for k in range(0, len(pieces), 2)]
    groups += [[x] for l in lv for x in _EX_MORE_SENT[l]]
    budget, n, sentences = SENT_WORDS[length], 0, []
    for g in groups:
        kept = []
        for x in g:
            w = len(x.split())
            if (sentences or kept) and n + w > budget:
                break
            kept.append(x)
            n += w
        if kept:
            sent = ", ".join(kept)
            sentences.append(sent if sent.startswith(t) else sent[:1].upper() + sent[1:])
        if len(kept) < len(g):
            break
    return " ".join(x + "." for x in sentences)


def clip_tokens(text: str) -> int:
    """Rough CLIP token count (CLIP reads 75 per chunk): about 1.3 tokens per word plus one per comma."""
    return int(round(len(re.findall(r"[A-Za-z0-9']+", text)) * 1.3 + text.count(",")))


def unload(url: str, model: str):
    """Free the model's VRAM right away (before training starts)."""
    try:
        _req(url, "/api/chat", {"model": model, "messages": [], "keep_alive": 0}, timeout=30)
    except OllamaError:
        pass


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--model")
    ap.add_argument("--trigger", default="ohwx")
    ap.add_argument("image", nargs="?")
    a = ap.parse_args()
    ms = list_models(a.url)
    for m in ms:
        print(f"{'vision' if m['vision'] else 'text  '}  {m['name']}  ({m['size_gb']} GB)")
    if a.image:
        mdl = a.model or next((m["name"] for m in ms if m["vision"]), None)
        if not mdl:
            raise SystemExit("No vision model in Ollama. Try: ollama pull qwen2.5vl:7b")
        print(caption(a.url, mdl, Path(a.image), a.trigger))
