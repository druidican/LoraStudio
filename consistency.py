# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Caption consistency (Datasets > Prepare > 5 Consistency).

After the captions are written, this reads every caption of a dataset and lists the traits that belong to the
character itself (hair colour and length, eye colour, build, skin, horns, tattoos, a signature object ...), grouped
per trait. Clothes, their colours, places, poses, expressions, light and camera are left out on purpose: those are
meant to differ from picture to picture.

For every trait you decide:
  * Keep in the LoRA  - the trait is removed from the captions, so the trigger word learns it. Pictures that say
                        something else (3 say brown eyes, 40 say blue) are shown one by one: fix (remove it) or keep
                        it for that picture only (a costume, contact lenses, odd light).
  * This picture only - every caption keeps its own value. Words that mean the same (athletic / fit / toned) are
                        merged into the one word you pick, so the LoRA does not learn three names for one thing.
  * Leave as it is

The traits of each caption come from the Ollama model (text only, one short question per caption, remembered by the
caption's text). Without Ollama a built-in word list finds the common ones (eyes, hair, build, skin).
"""
from __future__ import annotations
__version__ = "1.251026g"  # LoRA Studio version, see version.py

import hashlib
import json
import re
import threading
import time
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
CACHE_DIR = APP_DIR / "gui_cache"
CACHE_FILE = CACHE_DIR / "consistency_cache.json"      # caption text -> traits; values -> synonym groups
RESULT_DIR = CACHE_DIR / "consistency"                  # last scan per dataset (for the page after a reload)
STOP = threading.Event()
_LOCK = threading.Lock()

EXTRACT_PROMPT = """This is the caption of one training picture of a character or person{trig}:
"{caption}"

List ONLY traits that belong to the character itself and normally stay the same from picture to picture:
hair colour, hair length, hair style, eye colour, skin tone, body build, height, apparent age, facial hair, freckles,
moles, scars, tattoos, piercings, ears, horns, tail, wings, fur, other body or species features, and a signature
object the character always carries.
Do NOT list clothing or the colours of clothing, accessories, pose, expression, action, place, background, light,
camera, art style, or anything about other people.
For every trait give: "trait" (short name, e.g. "eye colour"), "value" (one or two words, e.g. "blue"), and "phrase"
(the exact words as they are written in the caption, e.g. "bright blue eyes").
Answer only JSON: {{"traits": [{{"trait": "...", "value": "...", "phrase": "..."}}]}}  (an empty list when there are none)."""

GROUP_PROMPT = """In different captions of the SAME character, these words were used for the character's {trait}:
{values}

Group the words that mean the same thing (synonyms or spelling variants, e.g. "athletic", "fit" and "toned", or
"grey" and "gray"). Words with a different meaning (e.g. "blue" and "brown", or "long" and "short") must be in
different groups. Every word must be in exactly one group.
Answer only JSON: {{"groups": [["word", "word"], ["word"]]}}"""

REWRITE_PROMPT = """Rewrite this picture caption with these changes and NOTHING else:
{changes}
Keep every other word, the order and the style. Keep "{trigger}" where it is. Answer with the new caption only.

Caption: {caption}"""

TRAIT_ALIASES = {
    "eye color": "eye colour", "eyes": "eye colour", "eye": "eye colour", "eye colors": "eye colour",
    "hair color": "hair colour", "haircolor": "hair colour", "hair colours": "hair colour",
    "hairstyle": "hair style", "hair": "hair colour",
    "body type": "build", "physique": "build", "body build": "build", "body shape": "build", "figure": "build",
    "body": "build", "body type/build": "build",
    "skin color": "skin tone", "skin colour": "skin tone", "skin": "skin tone", "complexion": "skin tone",
    "age": "apparent age", "beard": "facial hair", "moustache": "facial hair", "mustache": "facial hair",
}
# a trait is about the character when its name holds one of these ...
IDENTITY_WORDS = ("hair", "eye", "skin", "build", "body", "height", "age", "face", "facial", "beard", "freckle", "mole",
                  "scar", "tattoo", "pierc", "ear", "horn", "tail", "wing", "fur", "scale", "species", "muscle", "lip",
                  "nose", "teeth", "fang", "claw", "complexion", "physique", "signature", "birthmark", "makeup",
                  "breast", "chest", "hoof", "hooves", "antler", "mark", "feature")
# ... and none of these
NOT_IDENTITY = ("cloth", "dress", "shirt", "outfit", "wear", "garment", "shoe", "boot", "hat", "jacket", "accessor",
                "jewel", "pose", "expression", "background", "light", "camera", "style of", "art style", "action",
                "location", "setting", "emotion", "mood")

# built-in synonym sets (lowercase); the model groups the rest
SYNONYMS = {
    "build": [{"athletic", "fit", "toned", "sporty", "athletic build", "fit body"}, {"slim", "slender", "thin", "lean"},
              {"curvy", "voluptuous", "shapely", "hourglass"}, {"muscular", "buff", "ripped", "strong"},
              {"chubby", "plump", "full-figured"}],
    "hair colour": [{"blonde", "blond", "golden", "fair"}, {"brown", "brunette", "chestnut"}, {"red", "ginger"},
                    {"grey", "gray"}, {"white", "snow-white", "snow white"}, {"silver", "platinum", "silvery"}],
    "eye colour": [{"grey", "gray"}, {"amber", "golden", "gold"}, {"violet", "purple"}],
    "hair length": [{"long", "very long", "waist-length", "waist length"}, {"short", "cropped"},
                    {"medium", "medium-length", "shoulder-length", "shoulder length"}],
    "skin tone": [{"pale", "fair", "light", "porcelain"}, {"tan", "tanned", "sun-kissed"}],
}

LEXICON = [
    ("eye colour", r"\b(blue|green|brown|grey|gray|hazel|amber|red|purple|violet|golden|gold|yellow|pink|heterochromatic)[- ]eye[ds]?\b"),
    ("hair colour", r"\b(blonde?|black|brown|red|auburn|ginger|golden|grey|gray|white|silver|platinum|pink|blue|green|purple|brunette)[- ]hair(?:ed)?\b"),
    ("hair length", r"\b(very long|long|short|medium[- ]length|shoulder[- ]length|waist[- ]length|cropped)[- ]hair\b"),
    ("build", r"\b(athletic|fit|toned|slim|slender|skinny|thin|petite|curvy|voluptuous|muscular|chubby|plump|stocky)(?:[- ](?:build|body|figure|physique))?\b"),
    ("skin tone", r"\b(pale|fair|tan|tanned|dark|olive|light)[- ]skin(?:ned)?\b"),
    ("freckles", r"\b(freckles|freckled)\b"),
    ("horns", r"\b((?:\w+ )?horns)\b"),
    ("tattoos", r"\b((?:\w+ )?tattoos?)\b"),
]


class Error(Exception):
    pass


# --------------------------------------------------------------------------- small helpers
def _key(*parts) -> str:
    return hashlib.sha1("\x1f".join(str(p) for p in parts).encode("utf-8", "replace")).hexdigest()[:20]


def _cache_load() -> dict:
    try:
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _cache_save(c: dict):
    try:
        CACHE_DIR.mkdir(exist_ok=True)
        tmp = CACHE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(c), encoding="utf-8")
        tmp.replace(CACHE_FILE)
    except OSError:
        pass


def norm_trait(t: str) -> str:
    t = re.sub(r"\s+", " ", (t or "").strip().lower().replace("_", " "))
    return TRAIT_ALIASES.get(t, t)


def is_identity(trait: str) -> bool:
    t = trait.lower()
    return any(w in t for w in IDENTITY_WORDS) and not any(w in t for w in NOT_IDENTITY)


def norm_value(v: str, trait: str) -> str:
    v = re.sub(r"\s+", " ", (v or "").strip().lower().strip(".,;:'\""))
    for w in ("eyes", "eye", "hair", "haired", "eyed", "skin", "skinned", "build", "body", "colored", "coloured"):
        v = re.sub(rf"[- ]{w}$", "", v)
    return v.strip()


def is_tags(caption: str) -> bool:
    """A tag caption: comma separated short items (booru style), not sentences."""
    items = [x.strip() for x in (caption or "").split(",") if x.strip()]
    if len(items) < 3:
        return False
    return sum(len(x.split()) for x in items) / len(items) <= 4 and not re.search(r"[.!?]\s+[A-Z]", caption)


def _json(txt: str):
    txt = re.sub(r"<think>.*?</think>", "", txt or "", flags=re.S)
    txt = re.sub(r"^```(json)?|```$", "", txt.strip()).strip()
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", txt, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


def _ask(url, model, prompt, *, as_json=True, timeout=300.0, num_predict=500):
    import captioner as CAP
    payload = {"model": model, "stream": False, "keep_alive": "5m",
               "options": {"temperature": 0.0, "num_predict": num_predict},
               "messages": [{"role": "user", "content": prompt}]}
    if as_json:
        payload["format"] = "json"
    try:
        r = CAP._req(url, "/api/chat", payload, timeout=timeout)
    except CAP.OllamaError as e:
        raise Error(str(e)) from None
    return ((r.get("message") or {}).get("content") or r.get("response") or "").strip()


def _find_phrase(caption: str, phrase: str, value: str) -> str:
    """The phrase as it is written in the caption ('' when it cannot be found)."""
    low = caption.lower()
    for cand in (phrase, value):
        cand = (cand or "").strip().strip(".,")
        if cand and len(cand) > 1:
            i = low.find(cand.lower())
            if i >= 0:
                return caption[i:i + len(cand)]
    return ""


# --------------------------------------------------------------------------- reading the traits of one caption
def lexicon_traits(caption: str) -> list:
    out, seen = [], set()
    for trait, pat in LEXICON:
        for m in re.finditer(pat, caption, re.I):
            val = norm_value(m.group(1), trait)
            if trait in ("horns", "tattoos", "freckles"):
                val = m.group(1).lower()
            k = (trait, m.group(0).lower())
            if k not in seen:
                seen.add(k)
                out.append({"trait": trait, "value": val, "phrase": m.group(0)})
    return out


def caption_traits(caption: str, url: str, model: str, trigger: str, cache: dict) -> list:
    """[{trait, value, phrase}] of one caption: only identity traits, phrases as written in the caption."""
    if not caption:
        return []
    k = "x:" + _key(model, trigger, caption)
    if k in cache:
        return cache[k]
    if url and model:
        raw = _json(_ask(url, model, EXTRACT_PROMPT.format(
            caption=caption.replace('"', "'"), trig=f' (trigger word "{trigger}")' if trigger else "")))
        items = (raw or {}).get("traits") if isinstance(raw, dict) else raw if isinstance(raw, list) else []
    else:
        items = lexicon_traits(caption)
    out = []
    for it in items or []:
        if not isinstance(it, dict):
            continue
        trait = norm_trait(str(it.get("trait") or ""))
        value = norm_value(str(it.get("value") or ""), trait)
        if not trait or not value or not is_identity(trait):
            continue
        if trigger and value == trigger.lower():
            continue
        phrase = _find_phrase(caption, str(it.get("phrase") or ""), value)
        out.append({"trait": trait, "value": value, "phrase": phrase})
    if url and model:
        cache[k] = out
    return out


def group_values(trait: str, values: list, url: str, model: str, cache: dict) -> list:
    """[[value, ...], ...]: values that mean the same are in one group."""
    values = sorted(set(values))
    if len(values) < 2:
        return [values] if values else []
    groups, left = [], set(values)
    for syn in SYNONYMS.get(trait, []):
        g = [v for v in values if v in syn]
        if g:
            groups.append(g)
            left -= set(g)
    rest = sorted(left)
    if len(rest) >= 2 and url and model:
        k = "g:" + _key(model, trait, *rest)
        if k not in cache:
            raw = _json(_ask(url, model, GROUP_PROMPT.format(trait=trait, values=", ".join(f'"{v}"' for v in rest)),
                             num_predict=300))
            got, used = [], set()
            for g in (raw or {}).get("groups", []) if isinstance(raw, dict) else []:
                g = [str(x).strip().lower() for x in g if str(x).strip().lower() in left and str(x).strip().lower() not in used]
                if g:
                    used |= set(g)
                    got.append(g)
            got += [[v] for v in rest if v not in used]
            cache[k] = got
        groups += cache[k]
    else:
        groups += [[v] for v in rest]
    # a built-in group may overlap with one from the model: join them
    merged: list = []
    for g in groups:
        hit = next((m for m in merged if set(m) & set(g)), None)
        if hit is not None:
            hit.extend(x for x in g if x not in hit)
        else:
            merged.append(list(g))
    return merged


# --------------------------------------------------------------------------- the scan
def result_path(ds: str) -> Path:
    return RESULT_DIR / (re.sub(r"[^\w.-]+", "_", ds) + ".json")


def load_result(ds: str):
    try:
        return json.loads(result_path(ds).read_text(encoding="utf-8"))
    except Exception:
        return None


def captions_key(pics: dict) -> str:
    return _key(*sorted(f"{p}|{c}" for p, c in pics.items()))


def scan(ds: str, images: list, read_caption, url: str, model: str, trigger: str, progress=None) -> dict:
    """Read every caption and group the traits. images: [Path]; read_caption(Path) -> str | None."""
    STOP.clear()
    with _LOCK:
        cache = _cache_load()
    pics, traits, used_lexicon, err = {}, {}, False, ""
    caps = [(p, read_caption(p)) for p in images]
    caps = [(p, c) for p, c in caps if c]
    t0, last_save = time.time(), time.time()
    for i, (p, c) in enumerate(caps):
        if STOP.is_set():
            err = "stopped"
            break
        if progress:
            progress((i, len(caps)), desc=f"Reading caption {i + 1} of {len(caps)}")
        try:
            tr = caption_traits(c, url, model, trigger, cache)
        except Error as e:
            if not used_lexicon:
                err = str(e)
            used_lexicon = True
            tr = caption_traits(c, "", "", trigger, cache)
            url = model = ""                                    # Ollama is gone: the word list for the rest
        pics[str(p)] = c
        traits[str(p)] = tr
        if time.time() - last_save > 30:
            with _LOCK:
                _cache_save(cache)
            last_save = time.time()
    # per trait: value -> pictures
    by_trait: dict = {}
    for p, tr in traits.items():
        for t in tr:
            e = by_trait.setdefault(t["trait"], {})
            e.setdefault(t["value"], {})[p] = t["phrase"]
    groups = []
    for n, (trait, vals) in enumerate(sorted(by_trait.items(), key=lambda kv: -sum(len(x) for x in kv[1].values()))):
        try:
            vg = group_values(trait, list(vals), url, model, cache)
        except Error:
            vg = [[v] for v in sorted(vals)]
        values = []
        for g in vg:
            ph = {}
            for v in g:
                ph.update({p: vals[v][p] for p in vals[v]})
            count = {v: len(vals[v]) for v in g}
            values.append({"words": sorted(g, key=lambda v: -count[v]), "count": count, "pics": ph})
        values.sort(key=lambda v: -len(v["pics"]))
        mentioned = len({p for v in values for p in v["pics"]})
        groups.append({"id": f"t{n}", "trait": trait, "values": values, "mentioned": mentioned,
                       "kind": "contradiction" if len(values) > 1 else "same"})
    groups.sort(key=lambda g: (g["kind"] != "contradiction", -g["mentioned"]))
    with _LOCK:
        _cache_save(cache)
    res = {"ds": ds, "at": time.strftime("%Y-%m-%d %H:%M"), "total": len(caps), "groups": groups,
           "captions": captions_key(pics), "lexicon": used_lexicon, "error": err,
           "seconds": round(time.time() - t0, 1)}
    try:
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        result_path(ds).write_text(json.dumps(res), encoding="utf-8")
    except OSError:
        pass
    return res


# --------------------------------------------------------------------------- changing captions
def _tidy(s: str) -> str:
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s+([,.;:])", r"\1", s)
    s = re.sub(r"([,;])\s*(?=[,.;])", "", s)
    s = re.sub(r"\b(with|and|has|having|of)\s*([,.])", r"\2", s)
    s = re.sub(r"\b(with|and)\s+(and|with)\b", r"\1", s)
    s = re.sub(r",\s*,", ",", s)
    s = re.sub(r"^[\s,;]+|[\s,;]+$", "", s)
    return s.strip()


KEEP_BEFORE = {"with", "and", "a", "an", "the", "her", "his", "their", "its", "has", "have", "of"}


def _without_value(text: str, value: str) -> str:
    """'long white hair' without 'white' -> 'long hair'; '' when only the noun would be left."""
    rest = re.sub(rf"\b{re.escape(value)}\b[- ]?", "", text, count=1, flags=re.I).strip(" -")
    rest = re.sub(r"\s+", " ", rest)
    return "" if (not rest or rest.lower() in ("hair", "eyes", "eye", "skin", "body", "build", "figure")) else rest


def remove_phrase(caption: str, phrase: str, value: str = ""):
    """Take a phrase out of a sentence caption with the joining word around it ("with", "and", commas).
    When the phrase carries another trait in front ("long white hair", removing the colour), only the value goes.
    None when that would break the sentence ("She has an athletic build, ..."): such a caption needs the model."""
    m = re.search(rf"(\w+)[- ]+{re.escape(phrase)}", caption, re.I)
    if value and m and m.group(1).lower() not in KEEP_BEFORE:
        i = m.end() - len(phrase)
        short = _without_value(phrase, value) or re.sub(r"^\S+\s*", "", phrase)
        return _tidy(caption[:i] + short + caption[i + len(phrase):])
    p = re.escape(phrase)
    if re.search(rf"\b(has|have|is|are|was|were|looks?|appears?)\s+(a|an|the|very|quite|her|his)?\s*{p}", caption, re.I):
        return None
    s = caption
    for pat, rep in ((rf",\s*{p}(?=\s*[,.;]|\s*$)", ""), (rf"\s+and\s+{p}\b", ""), (rf"\b{p}\s+and\s+", ""),
                     (rf"\s+with\s+{p}\b", ""), (rf"\b{p}\s*,\s*", ""), (rf"\b{p}\b", "")):
        new = re.sub(pat, rep, s, count=1, flags=re.I)
        if new != s:
            s = new
            break
    return _tidy(s)


def edit_tags(caption: str, removes: list, replaces: list) -> str:
    """removes: [(phrase, value)]. A tag that is just the phrase goes; a longer tag loses only the value
    ("long white hair" without the colour -> "long hair")."""
    items = [x.strip() for x in caption.split(",")]
    out = []
    for it in items:
        low = it.lower()
        drop = False
        for r, val in removes:
            if not r or r.lower() not in low:
                continue
            if low == r.lower():
                drop = True
                break
            it = _without_value(it, val or r) if val else ""
            low = it.lower()
            if not it:
                drop = True
                break
        if drop:
            continue
        for old, new in replaces:
            if old and old.lower() in low:
                it = re.sub(re.escape(old), new, it, flags=re.I)
                low = it.lower()
        if it and it.lower() not in (x.lower() for x in out):
            out.append(it)
    return ", ".join(out)


def edit_sentence(caption: str, removes: list, replaces: list, url: str, model: str, trigger: str) -> str:
    s = caption
    for old, new in replaces:
        if old:
            s = re.sub(re.escape(old), new, s, flags=re.I)
            art = "an" if new[:1].lower() in "aeiou" else "a"
            s = re.sub(rf"\b(a|an|A|An) (?={re.escape(new)})", lambda m: (art.capitalize() if m.group(1)[0] == "A" else art) + " ", s)
    if not removes:
        return s
    if url and model:                       # the model rewrites the sentence so it still reads well
        changes = "\n".join(f'- remove "{v}" from "{r}" (keep the other words about it, if any)' if v and v.lower() != r.lower()
                             else f'- remove any mention of "{r}"' for r, v in removes)
        try:
            new = _ask(url, model, REWRITE_PROMPT.format(changes=changes, trigger=trigger or "", caption=s),
                       as_json=False, num_predict=400)
            new = re.sub(r"<think>.*?</think>", "", new, flags=re.S).strip().strip('"').strip()
            new = re.sub(r"^(new )?caption\s*:\s*", "", new, flags=re.I)
            ok = (new and all(not re.search(rf"\b{re.escape(v or r)}\b", new, re.I) for r, v in removes)
                  and 0.4 * len(s) <= len(new) <= 1.1 * len(s)
                  and (not trigger or trigger.lower() not in s.lower() or trigger.lower() in new.lower()))
            if ok:
                return new
        except Error:
            pass
    for r, v in removes:
        new = remove_phrase(s, r, v)
        if new is None:
            raise Error(f"needs the Ollama model to rewrite: {caption[:60]}")
        s = new
    return s


def plan(res: dict, choices: dict) -> dict:
    """{picture: {"remove": [phrase], "replace": [(old, new)]}} from the choices.
    choices = {trait id: {"mode": "lora"|"each"|"skip", "fix": [odd pictures whose value is removed too],
                          "words": {value group index: chosen word}}}"""
    ops: dict = {}
    for g in res.get("groups", []):
        ch = choices.get(g["id"]) or {}
        mode, _, mi = str(ch.get("mode", "skip")).partition(":")
        if mode == "lora":
            # the kept value is removed everywhere; a picture that says something else only when you chose "fix"
            # for it (after looking at it); the others keep their own value ("this picture only", or a locked exception)
            fix = set(ch.get("fix") or []) - set(g.get("exceptions") or [])
            k = int(mi) if mi.isdigit() and int(mi) < len(g["values"]) else 0
            main = g["values"][k]["pics"] if g["values"] else {}
            for v in g["values"]:
                for p, phrase in v["pics"].items():
                    if p not in main and p not in fix:
                        continue
                    if phrase:
                        val = next((w for w in v["words"] if re.search(rf"\b{re.escape(w)}\b", phrase, re.I)), "")
                        ops.setdefault(p, {"remove": [], "replace": []})["remove"].append((phrase, val))
        elif mode == "each":
            for i, v in enumerate(g["values"]):
                word = (ch.get("words") or {}).get(str(i))
                if not word or len(v["words"]) < 2:
                    continue
                for p, phrase in v["pics"].items():
                    if not phrase:
                        continue
                    for w in v["words"]:
                        if w != word and re.search(rf"\b{re.escape(w)}\b", phrase, re.I):
                            new = re.sub(rf"\b{re.escape(w)}\b", word, phrase, count=1, flags=re.I)
                            ops.setdefault(p, {"remove": [], "replace": []})["replace"].append((phrase, new))
                            break
    return ops


def apply(res: dict, choices: dict, read_caption, caption_file, url: str = "", model: str = "", trigger: str = "",
          progress=None) -> tuple:
    """Change the captions. Returns ({caption file: old text} for Undo, [(picture, old, new)], kept-in-LoRA traits,
    [pictures skipped because their sentence needs the model to rewrite it])."""
    ops = plan(res, choices)
    backup, changed, skipped = {}, [], []
    for i, (p, op) in enumerate(ops.items()):
        if progress:
            progress((i, len(ops)), desc=f"Changing caption {i + 1} of {len(ops)}")
        img = Path(p)
        old = read_caption(img)
        if not old:
            continue
        if is_tags(old):
            new = edit_tags(old, op["remove"], op["replace"])
        else:
            try:
                new = edit_sentence(old, op["remove"], op["replace"], url, model, trigger)
            except Error:
                skipped.append(p)
                continue
        if new and new.strip() != old.strip():
            cf = caption_file(img)
            backup[cf] = cf.read_text(encoding="utf-8", errors="ignore") if cf.is_file() else None
            cf.write_text(new.strip() + "\n", encoding="utf-8")
            changed.append((p, old, new))
    kept = [g["trait"] for g in res.get("groups", [])
            if str((choices.get(g["id"]) or {}).get("mode", "")).startswith("lora")]
    return backup, changed, kept, skipped


# --------------------------------------------------------------------------- locked traits (<dataset>/traits.txt)
LOCK_NAME = "traits.txt"
LOCKS_ON = False            # locked traits were removed in the Datasets redesign: an old traits.txt is ignored
LOCK_HEAD = """# LoRA Studio: locked traits of this dataset. They come first: before the caption model, the recommendations
# and new choices (a new choice that clashes with a lock is only written after you confirm it).
# Written by Datasets > Prepare > 5 Consistency; you can also edit this file by hand.
#
#   trait = keep VALUE | words: phrase, phrase    kept in the LoRA: left out of every caption (the trigger word learns it)
#   trait = each | use WORD for other, other      each caption keeps its own value; WORD replaces the others
#   [pictures]  picture: trait = phrase           exceptions: this picture keeps its own value
"""


def lock_file(root) -> Path:
    return Path(root) / LOCK_NAME


def _split(x: str) -> list:
    return [w.strip() for w in x.split(",") if w.strip()]


def locks_load(root) -> dict:
    out = {"traits": {}, "pictures": {}}
    if not LOCKS_ON:
        return out
    try:
        text = lock_file(root).read_text(encoding="utf-8")
    except OSError:
        return out
    sec = "traits"
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower() in ("[traits]", "[pictures]"):
            sec = line.lower()[1:-1]
            continue
        if sec == "pictures":
            pic, _, rest = line.partition(":")
            trait, _, phrase = rest.partition("=")
            if pic.strip() and trait.strip():
                out["pictures"].setdefault(pic.strip(), {})[norm_trait(trait)] = phrase.strip()
            continue
        name, _, rest = line.partition("=")
        trait = norm_trait(name)
        parts = [x.strip() for x in rest.split("|")]
        if not trait or not parts or not parts[0]:
            continue
        head = parts[0].split(None, 1)
        mode = head[0].lower()
        e = {"mode": "keep" if mode == "keep" else "each", "value": (head[1].strip().lower() if len(head) > 1 else ""),
             "words": [], "use": {}}
        for x in parts[1:]:
            if x.lower().startswith("words:"):
                e["words"] = _split(x[6:])
            elif x.lower().startswith("use "):
                w, _, others = x[4:].partition(" for ")
                if w.strip():
                    e["use"][w.strip().lower()] = [o.lower() for o in _split(others)]
        out["traits"][trait] = e
    return out


def locks_save(root, locks: dict):
    if not LOCKS_ON:
        return
    L = [LOCK_HEAD, "[traits]"]
    for trait, e in sorted(locks.get("traits", {}).items()):
        if e["mode"] == "keep":
            line = f"{trait} = keep {e.get('value', '')}".rstrip()
            if e.get("words"):
                line += " | words: " + ", ".join(e["words"])
        else:
            line = f"{trait} = each" + "".join(f" | use {w} for {', '.join(o)}" for w, o in e.get("use", {}).items() if o)
        L.append(line)
    if locks.get("pictures"):
        L += ["", "[pictures]"]
        for pic, tr in sorted(locks["pictures"].items()):
            L += [f"{pic}: {t} = {ph}" for t, ph in sorted(tr.items())]
    lock_file(root).write_text("\n".join(L) + "\n", encoding="utf-8")


def _rel(root, p) -> str:
    try:
        return str(Path(p).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(p)


def with_locks(res: dict, root) -> dict:
    """Mark the groups with their lock: the kept value first, locked exceptions left out of the odd pictures."""
    locks = locks_load(root)
    for g in res.get("groups", []):
        lk = locks["traits"].get(g["trait"])
        g["lock"] = lk
        g["exceptions"] = [p for v in g["values"] for p in v["pics"]
                           if g["trait"] in locks["pictures"].get(_rel(root, p), {})]
        # a locked exception keeps its own value: it is never asked about or changed again
        exc = set(g["exceptions"])
        for v in g["values"]:
            v["pics"] = {p: ph for p, ph in v["pics"].items() if p not in exc}
        g["values"] = [v for v in g["values"] if v["pics"]]
        g["kind"] = "contradiction" if len(g["values"]) > 1 else "same"
        g["mentioned"] = len({p for v in g["values"] for p in v["pics"]})
        if lk and lk["mode"] == "keep" and lk.get("value"):
            want = {lk["value"], *[w.lower() for w in lk.get("words", [])]}
            k = next((i for i, v in enumerate(g["values"])
                      if want & set(v["words"]) or any(ph.lower() in want for ph in v["pics"].values())), None)
            if k:
                g["values"].insert(0, g["values"].pop(k))
            g["lock_found"] = k is not None
    res["groups"] = [g for g in res.get("groups", []) if g["values"]]
    return res


def lock_changes(res: dict, choices: dict, root) -> tuple:
    """(new locks, [clash text]) for these choices. Leave as it is (skip) changes no lock."""
    locks = locks_load(root)
    new = {"traits": dict(locks["traits"]), "pictures": {k: dict(v) for k, v in locks["pictures"].items()}}
    clashes = []
    for g in res.get("groups", []):
        ch = choices.get(g["id"]) or {}
        mode, _, mi = str(ch.get("mode", "skip")).partition(":")
        if mode not in ("lora", "each") or not g["values"]:
            continue
        old = locks["traits"].get(g["trait"])
        if mode == "lora":
            k = int(mi) if mi.isdigit() and int(mi) < len(g["values"]) else 0
            v = g["values"][k]
            e = {"mode": "keep", "value": v["words"][0],
                 "words": sorted({ph.lower() for ph in v["pics"].values() if ph} | set(v["words"]) |
                                 set((old or {}).get("words", []) if old and old.get("value") == v["words"][0] else [])),
                 "use": {}}
            own = set(ch.get("own") or [])        # odd pictures you chose "this picture only" for (after looking)
            for i, other in enumerate(g["values"]):
                if i == k:
                    continue
                for p, ph in other["pics"].items():
                    if p not in v["pics"] and p in own:
                        new["pictures"].setdefault(_rel(root, p), {})[g["trait"]] = ph or other["words"][0]
            if old and (old["mode"] != "keep" or old.get("value") != e["value"]):
                clashes.append(f"{g['trait']}: locked as " + (f"kept in the LoRA ({old.get('value')})" if old["mode"] == "keep"
                               else "this picture only") + f", now kept in the LoRA ({e['value']})")
        else:
            use = {}
            for i, v in enumerate(g["values"]):
                w = (ch.get("words") or {}).get(str(i))
                if w and len(v["words"]) > 1:
                    others = [x for x in v["words"] if x != w]
                    phrases = sorted({ph.lower() for ph in v["pics"].values() if ph and " " in ph
                                      and any(re.search(rf"\b{re.escape(o)}\b", ph, re.I) for o in others)})
                    use[w] = others + phrases
            e = {"mode": "each", "value": "", "words": [], "use": {**((old or {}).get("use") or {}), **use}}
            if old and old["mode"] == "keep":
                clashes.append(f"{g['trait']}: locked as kept in the LoRA ({old.get('value')}), now this picture only")
        new["traits"][g["trait"]] = e
    return new, clashes


def kept_traits(root) -> list:
    return [t for t, e in locks_load(root)["traits"].items() if e["mode"] == "keep"]


def lock_caption(root, img, caption: str) -> str:
    """A new caption (step 4) made to follow the locks: kept traits out, locked words in. Without the model, so a
    sentence that would break is left as it is (step 5 finds it)."""
    if not caption:
        return caption
    locks = locks_load(root)
    if not locks["traits"]:
        return caption
    exc = locks["pictures"].get(_rel(root, img), {})
    removes, replaces = [], []
    for trait, e in locks["traits"].items():
        if trait in exc:
            continue
        if e["mode"] == "keep":
            val = e.get("value") or ""
            cands = sorted({c.lower() for c in (e.get("words") or [])} | ({val} if val else set()), key=len, reverse=True)
            if is_tags(caption):
                # a tag that IS the value or a phrase ("blue eyes", "athletic"), or holds a longer phrase
                # ("long white hair"); a bare colour word inside another tag ("blue dress") is never touched
                for it in (x.strip() for x in caption.split(",")):
                    low = it.lower()
                    hit = next((c for c in cands if low == c or (" " in c and re.search(rf"\b{re.escape(c)}\b", low))), None)
                    if hit:
                        removes.append((it if low == hit else hit, val if val and val in hit else ""))
            else:
                for c in cands:
                    m = re.search(rf"\b{re.escape(c)}\b", caption, re.I) if " " in c else None
                    if m:
                        removes.append((m.group(0), val if val and val in c else ""))
                        break
        else:
            for w, others in e.get("use", {}).items():
                words = [o for o in others if " " not in o]
                for o in sorted(others, key=len, reverse=True):
                    if " " in o:            # a phrase ("fit body"): its synonym word becomes the locked word
                        syn = next((x for x in words if re.search(rf"\b{re.escape(x)}\b", o)), "")
                        new_o = re.sub(rf"\b{re.escape(syn)}\b", w, o) if syn else ""
                        if new_o and re.search(rf"\b{re.escape(o)}\b", caption, re.I):
                            replaces.append((o, new_o))
                    elif is_tags(caption) and any(x.strip().lower() == o for x in caption.split(",")):
                        replaces.append((o, w))
    if not removes and not replaces:
        return caption
    if is_tags(caption):
        return edit_tags(caption, removes, replaces)
    try:
        return edit_sentence(caption, removes, replaces, "", "", "")
    except Error:
        return caption
