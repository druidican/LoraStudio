# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Fixed features of a dataset (step 2): what must stay right, in your own words (an outfit, an object, a colour, a
style, the gender), each with the wrong words the WD tagger or Ollama use for it. Every fixed feature is part of the
trigger: the word and its wrong words are taken out of every caption (tags and sentences) when captions are written,
Ollama is told never to name them, and step 8 lists captions that still name one (written by hand or by an older
caption run). When the gender is a fixed feature, the class word goes too (sentence captions begin "trigger, …").
LoRA Studio never suggests features itself: you write them. Kept in the dataset's meta as `fixed`:
[{"word": "orange visor", "wrong": "goggles, visor"}].
"""
from __future__ import annotations
__version__ = "1.271126b"  # LoRA Studio version, see version.py

import re
from pathlib import Path

GENDER = {"male", "female", "man", "woman", "men", "women", "boy", "girl", "boys", "girls", "1boy", "1girl",
          "2boys", "2girls", "multiple boys", "multiple girls", "guy", "lady", "masculine", "feminine", "androgynous",
          "gender", "male focus", "female focus"}


def _F():
    import dsflow
    return dsflow


def rows(ds: str) -> list[dict]:
    """[{"word", "wrong"}] of the dataset (empty rows left out)."""
    out = []
    for r in _F().meta(ds).get("fixed") or []:
        w, x = str(r.get("word") or "").strip(), str(r.get("wrong") or "").strip()
        if w or x:
            out.append({"word": w, "wrong": x})
    return out


def save(ds: str, new_rows: list[dict]) -> str:
    F = _F()
    m = F.meta(ds)
    clean = []
    for r in new_rows or []:
        w = re.sub(r"\s+", " ", str(r.get("word") or "")).strip(" ,")
        x = ", ".join(t for t in _split(r.get("wrong") or "") if t.lower() != w.lower())
        if w or x:
            clean.append({"word": w, "wrong": x})
    m["fixed"] = clean
    F.meta_save(ds, m)
    n = len(clean)
    return (f"✔ {n} fixed feature{'s' if n != 1 else ''} kept. They are taken out of the captions the next time "
            "they are written (step 7); step 8 lists captions that still name one.") if n else "No fixed features."


def _split(text: str) -> list[str]:
    out = []
    for t in re.split(r"[,;\n]", text or ""):
        t = re.sub(r"\s+", " ", t.replace("_", " ")).strip().lower()
        if t and t not in out:
            out.append(t)
    return out


def _cls(ds: str) -> str:
    """The class word the captions use ('' when the dataset is not a character)."""
    import datacheck as DC
    m = _F().meta(ds)
    if DC.base_purpose(m.get("purpose") or "character") != "character":
        return ""
    return (m.get("class_word", "person") or "").strip().lower()


def drops_class(ds: str) -> bool:
    """The gender is a fixed feature: the class word goes too."""
    t = set(_plain_terms(ds))
    cls = _cls(ds)
    return bool(t & GENDER) or (bool(cls) and cls in t)


def _plain_terms(ds: str) -> list[str]:
    out = []
    for r in rows(ds):
        for t in _split(r["word"]) + _split(r["wrong"]):
            if t not in out:
                out.append(t)
    return out


def terms(ds: str) -> list[str]:
    """Every word that is taken out of the captions (the fixed words, their wrong words and, when the gender is a
    fixed feature, the class word), longest first."""
    out = _plain_terms(ds)
    if out and drops_class(ds):
        cls = _cls(ds)
        if cls and cls not in out:
            out.append(cls)
    return sorted(out, key=lambda t: (-len(t), t))


def class_word(ds: str) -> str:
    """The class word sentence captions begin with: '' when the gender is a fixed feature."""
    return "" if drops_class(ds) else _cls(ds)


def prompt_rule(ds: str, trigger: str) -> str:
    """The line Ollama's caption prompt gets."""
    ws = [r["word"] for r in rows(ds) if r["word"]]
    wrong = [t for r in rows(ds) for t in _split(r["wrong"])]
    if not ws and not wrong:
        return ""
    return (f"- NEVER name or describe these, they are part of {trigger}: " + ", ".join(ws + wrong) + ".\n")


# --------------------------------------------------------------------------- taking them out
def _pat(term: str) -> str:
    """A term as a whole word or phrase (also with s/es, - or _ between its words)."""
    words = [re.escape(w) for w in term.split()]
    return r"(?<![\w-])" + r"[\s_-]+".join(words) + r"(?:e?s)?(?![\w-])"


def found(text: str, ts: list[str]) -> list[str]:
    low = (text or "").lower()
    return [t for t in ts if re.search(_pat(t), low)]


def strip_tags(line: str, ts: list[str], trigger: str = "") -> str:
    """A tag line without the tags that name a term (the trigger word stays)."""
    trig = (trigger or "").strip().lower()
    keep = []
    for x in re.split(r",", line or ""):
        x = x.strip()
        if not x:
            continue
        if x.lower() != trig and found(x.replace("_", " "), ts):
            continue
        keep.append(x)
    return ", ".join(keep)


_LEAD = r"(?:(?:wears|is wearing|wearing|has|with|and|in)\s+)?(?:(?:a|an|the|his|her|its|their)\s+)?"
_DANGLING = r"\b(?:and|with|near|of|in|on|at|beside|next to|wearing|wears)\s*(?=[,;.]|$)"


def strip_sentences(text: str, ts: list[str], trigger: str = "") -> str:
    """Sentences without the terms: a part (between commas) that names one goes; in the part with the trigger word
    only the words themselves go."""
    trig = (trigger or "").strip().lower()
    parts = re.split(r"(?<=[,;.])\s*", (text or "").strip())
    out = []
    for i, p in enumerate(parts):
        if not p:
            continue
        if not found(p, ts):
            out.append(p)
            continue
        if i == 0 or (trig and trig in p.lower()):
            q = p
            for t in ts:
                q = re.sub(_LEAD + _pat(t), "", q, flags=re.I)
            q = re.sub(_DANGLING, "", re.sub(r"\s{2,}", " ", q)).strip()
            if trig:
                q = re.sub(rf"^({re.escape(trig)})\s+and\s+", r"\1 ", q, flags=re.I)
            if re.sub(r"[\s,;.]", "", q):
                out.append(q)
        # another part that names a term goes as a whole
    t = " ".join(out).strip()
    t = re.sub(r"\s+([,;.])", r"\1", t)
    t = re.sub(r"([,;])\s*([,;.])", r"\2", t)
    t = re.sub(r"\s{2,}", " ", t).strip(" ,;")
    if t and not t.endswith((".", "!", "?")) and (text or "").strip().endswith("."):
        t += "."
    return t


def strip(ds: str, text: str, style: str, trigger: str = "") -> str:
    ts = terms(ds)
    if not ts or not text:
        return text
    trigger = trigger or (_F().meta(ds).get("trigger") or ds)
    return strip_tags(text, ts, trigger) if style == "tags" else strip_sentences(text, ts, trigger)


def style_of_ext(ds: str, ext: str) -> str:
    import dscaptions
    return next((x["style"] for x in dscaptions.caption_sets(ds) if x["ext"] == ext), "sentences")


# --------------------------------------------------------------------------- step 8
def check(ds: str) -> list[dict]:
    """Captions that still name a fixed feature: [{"path", "ext", "caption", "hits"}]."""
    import dscaptions
    ts = terms(ds)
    if not ts:
        return []
    F = _F()
    out = []
    for s in dscaptions.caption_sets(ds):
        for p in F.images(ds):
            c = dscaptions.read_set(p, s["ext"])
            hits = found(c, ts) if c else []
            if hits:
                out.append({"path": str(p), "ext": s["ext"], "caption": c, "hits": hits})
    return out


def take_out(ds: str, path: str = "") -> str:
    """Take the fixed features out of the captions (one picture, or every one the check lists); undo with Undo."""
    import sd_prepare
    import dscaptions
    F = _F()
    items = [x for x in check(ds) if not path or x["path"] == path]
    if not items:
        return "No caption names a fixed feature."
    files = {dscaptions.set_file(Path(x["path"]), x["ext"]): x for x in items}
    try:
        sd_prepare.record_change(f"fixed features out of the captions of {ds}",
                             {f: f.read_text(encoding="utf-8", errors="ignore") for f in files})
    except Exception:                                    # noqa: BLE001
        pass
    trig = F.meta(ds).get("trigger") or ds
    for f, x in files.items():
        f.write_text(strip(ds, x["caption"], style_of_ext(ds, x["ext"]), trig) + "\n", encoding="utf-8")
    n = len(files)
    return f"✔ Taken out of {n} caption{'s' if n != 1 else ''}."
