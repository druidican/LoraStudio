# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""dscaptions.py - step 7 of a dataset, the captions: one caption set per caption style (IMG.txt, IMG.<style>.txt;
caption_sets, caption_ext, the set each launch script trains on), writing a set (write_set: Ollama sentences or the
WD tagger's tags, fixed features taken out), the tags → trigger words card (move_tags, put_back) and the description
of the subject.
"""
from __future__ import annotations
__version__ = "0"  # LoRA Studio version, see version.py

import re
import shutil
import time
from pathlib import Path

import dsflow as F
import dsworn
import fixedfeat as FX


# --------------------------------------------------------------------------- 7 Captions
def style_of(tkey) -> str:
    import sd_prepare
    fam = sd_prepare._oc_family(tkey)
    return sd_prepare.OC_FAMILY.get(fam, ("sentences",))[0]


# Model families whose text encoder reads both booru tags and sentences: the dataset decides (step 2), and with both
# caption sets the launch script picks one (step 9).
BOTH_STYLES = {"anima", "flux", "chroma"}
CAPTION_CHOICES = [("Each type's usual style", "auto"), ("Tags", "tags"), ("Sentences", "sentences"),
                   ("Both: pick per run in step 9", "both")]


def can_both(tkey) -> bool:
    import sd_prepare
    return sd_prepare._oc_family(tkey) in BOTH_STYLES


def styles_for(tkey, m: dict) -> list[str]:
    """The caption styles this LoRA type gets in this dataset, its usual one first."""
    base = style_of(tkey)
    if not can_both(tkey):
        return [base]
    ch = m.get("caption_choice") or "auto"
    if ch == "both":
        return [base, "sentences" if base == "tags" else "tags"]
    return [ch] if ch in ("tags", "sentences") else [base]


def caption_sets(ds) -> list[dict]:
    """[{'style', 'ext', 'types': [labels], 'tkeys'}]: the first LoRA type's style is IMG.txt."""
    m = F.meta(ds)
    labels = F.type_labels()
    out = []
    for t in m["types"] or []:
        for st in styles_for(t, m):
            e = next((x for x in out if x["style"] == st), None)
            if e is None:
                e = {"style": st, "ext": ".txt" if not out else f".{st}.txt", "types": [], "tkeys": []}
                out.append(e)
            e["types"].append(labels.get(t, t))
            e["tkeys"].append(t)
    if not out:
        out.append({"style": "sentences", "ext": ".txt", "types": [], "tkeys": []})
    return out


def train_sets(ds, tkey) -> list[dict]:
    """The caption sets a launch script of this type can train on."""
    return [x for x in caption_sets(ds) if tkey in x["tkeys"]]


def train_style(ds, tkey) -> str:
    """The style a launch script of this type trains on: your pick in step 9, else the type's usual one."""
    sets = train_sets(ds, tkey)
    pick = (F.meta(ds).get("train_style") or {}).get(tkey)
    if pick and any(x["style"] == pick for x in sets):
        return pick
    return sets[0]["style"] if sets else style_of(tkey)


def caption_ext(ds, tkey) -> str:
    """--caption_extension for a launch script of this type (".txt", ".sentences.txt", ...)."""
    st = train_style(ds, tkey)
    return next((x["ext"] for x in train_sets(ds, tkey) if x["style"] == st), ".txt")


def set_train_style(ds, tkey, style):
    m = F.meta(ds)
    m.setdefault("train_style", {})[tkey] = style
    F.meta_save(ds, m)


def set_file(img: Path, ext: str) -> Path:
    import dsfiles
    if ext == ".txt":
        return dsfiles.caption_file(img)
    return img.with_name(img.stem + ext)


def read_set(img: Path, ext: str):
    f = set_file(img, ext)
    if not f.is_file():
        return None
    return f.read_text(encoding="utf-8", errors="ignore").strip() or None


def set_count(ds, ext) -> tuple[int, int]:
    imgs = F.images(ds)
    return sum(1 for p in imgs if read_set(p, ext)), len(imgs)


def write_set(ds, url, model, trigger, leave_out, style, length, detail, ext, only_missing=False):
    """Generator: writes a caption set (IMG.sentences.txt ...). Tag sets are written by the WD
    tagger (wdtagger.py), sentence sets by Ollama."""
    import sd_prepare
    S = F.sg()
    tell_fixed(ds, trigger)
    if style == "tags":
        yield from _write_set_wd(ds, trigger, leave_out, length, ext, only_missing)
        return
    import features as FT
    if not FT.have("ollama"):
        yield f"Sentences (`IMG{ext}`) need Ollama, which is not installed: write them in 8 Review, or install Ollama."
        return
    bad = sd_prepare._oc_check(ds, url, model, trigger)
    if bad:
        yield bad
        return
    url = (url or "").strip() or S.CAP.DEFAULT_URL
    imgs = F.images(ds)
    if only_missing:
        imgs = [p for p in imgs if not read_set(p, ext)]
    if not imgs:
        yield "Every picture already has a caption in this set."
        return
    sd_prepare.record_change(f"captions {ext} of {ds}", {set_file(p, ext): (set_file(p, ext).read_text(encoding="utf-8", errors="ignore")
                                                                 if set_file(p, ext).is_file() else None) for p in imgs})
    F.STOP.clear()
    done, failed, t0 = 0, [], time.time()
    for i, p in enumerate(imgs, 1):
        if F.STOP.is_set():
            break
        eta = f", about {S.fmt((time.time() - t0) / max(done, 1) * (len(imgs) - done))} left" if done else ""
        yield f"⏳ {style} {i}/{len(imgs)}: {p.name}{eta}"
        try:
            r = S.CAP.caption(url, model, p, (trigger or ds).strip(), leave_out or "", style=style,
                              length=int(length or 3), detail=int(detail or 3))
        except S.CAP.OllamaError as e:
            failed.append(f"{p.name}: {e}")
            if "not reachable" in str(e):
                break
            continue
        set_file(p, ext).write_text(FX.strip(ds, r["caption"], style, (trigger or ds).strip()) + "\n", encoding="utf-8")
        done += 1
    msg = f"{'■ Stopped' if F.STOP.is_set() else '✔ Done'}: {done} of {len(imgs)} captions written to `IMG{ext}`."
    if failed:
        msg += f" {len(failed)} failed: " + "; ".join(failed[:3])
    yield msg


def tell_fixed(ds, trigger=""):
    """The caption writer (captioner.PURPOSE) gets the dataset's fixed features and its class word (none when the
    gender is a fixed feature)."""
    S = F.sg()
    if S is not None and S.CAP is not None:
        S.CAP.PURPOSE["fixed"] = FX.prompt_rule(ds, (trigger or F.meta(ds).get("trigger") or ds).strip())
        S.CAP.PURPOSE["class"] = FX.class_word(ds)


def _write_set_wd(ds, trigger, leave_out, length, ext, only_missing=False):
    """A tag set with the WD tagger (no Ollama needed)."""
    import sd_prepare
    import wdtagger as WD
    S = F.sg()
    imgs = F.images(ds)
    if only_missing:
        imgs = [p for p in imgs if not read_set(p, ext)]
    if not imgs:
        yield "Every picture already has a caption in this set."
        return
    sd_prepare.record_change(f"captions {ext} of {ds}", {set_file(p, ext): (set_file(p, ext).read_text(encoding="utf-8", errors="ignore")
                                                                 if set_file(p, ext).is_file() else None) for p in imgs})
    F.STOP.clear()
    cap = S.CAP if S.CAP is not None else __import__("captioner")
    max_tags = cap.TAG_COUNT.get(int(length or 3), 20)
    traits: dict = {}
    import datacheck as DC
    purpose = F.meta(ds).get("purpose", "character")
    character = DC.base_purpose(purpose) == "character"
    # a character or a worn item: ALL tags are written; afterwards step 7 moves the character's own ones (or the
    # item's own look) to the trigger words
    write_leave = "" if character or purpose == "worn" else (leave_out or "")
    last = ""
    for last in WD.tag(imgs, (trigger or ds).strip(), write_leave, max_tags,
                       lambda img, line: set_file(img, ext).write_text(
                           FX.strip(ds, line, "tags", (trigger or ds).strip()) + "\n", encoding="utf-8"), F.STOP, traits,
                       trait_leave=(leave_out or cap.DEFAULT_LEAVE_OUT) if character else
                       ("" if purpose == "worn" else None)):
        yield last
    ok = str(last).startswith("✔") and traits.get("_n")      # only after captions were really written
    m = F.meta(ds)
    if traits.get("_n"):
        m["trait_tags"] = traits                         # Character description › Suggest (step 7)
    if (character or purpose == "worn") and traits.get("_n"):
        m["wd_all"] = {"ext": ext, "at": time.strftime("%Y-%m-%d %H:%M")}
        shutil.rmtree(_moved_dir(ds, ext), ignore_errors=True)    # new captions with all tags: nothing moved yet
        (m.get("moved_tags") or {}).pop(ext, None)
    F.meta_save(ds, m)
    if ok and character:
        yield (f"{last}\n\nEvery tag is in the captions now, also the character's own (hair, eyes ...). Next: "
               "**Character tags → trigger words** below moves the ones you tick out of the captions and next to the "
               "trigger word.")
    elif ok and purpose == "worn":
        yield (f"{last}\n\nEvery tag is in the captions now, also the item's own look (metal, stones, pattern ...). "
               "Next: **Kinds and where they are worn** makes those words the same in every caption, and **Item tags "
               "→ trigger words** moves the item's look out of the captions.")


# ---- step 7, character tags: all tags are written first, then the ticked ones move to the trigger words
def _moved_dir(ds, ext) -> Path:
    return F._flow(ds) / f"captions_alltags{ext.replace('.', '_')}"


def tag_counts(ds, ext=".txt", trigger=""):
    """[(tag, pictures)] of a caption set, most frequent first, and the number of pictures with a caption."""
    trig = (trigger or ds or "").strip().lower()
    counts, n = {}, 0
    for p in F.images(ds):
        t = read_set(p, ext)
        if not t:
            continue
        n += 1
        for x in dict.fromkeys(w.strip().lower() for w in re.split(r"[,\n]", t) if w.strip()):
            if x != trig:
                counts[x] = counts.get(x, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])), n


def default_ticks(rows, n, purpose="character", worn_kind="jewelry", kinds=None, places=None) -> list[str]:
    """Pre-ticked: tags in at least 40 % of the pictures that describe the character (hair, eyes, freckles, build,
    face, tattoos, makeup). Never ethnicity, skin or age: those you tick yourself if you want them.
    A worn item: tags in at least 30 % of the pictures that describe the item's own look (metal, stones, pattern,
    material ...), never the kind words or the places it is worn."""
    import captioner as CAP
    if purpose == "worn":
        look = dsworn.WORN_LOOK.get(worn_kind or "jewelry", dsworn.WORN_LOOK["jewelry"])
        own = {w.lower() for w in (kinds or []) + (places or [])}
        out = []
        for t, c in rows:
            if c < max(2, 0.3 * n) or t in own:
                continue
            words = set(re.findall(r"[a-z]+", t))
            if words & look or any(re.search(rf"\b{re.escape(k)}\b", t) for k in (kinds or [])):
                out.append(t)
        return out
    out = []
    for t, c in rows:
        if c < max(2, 0.4 * n):
            continue
        hit = CAP.leaks(t, CAP.DEFAULT_LEAVE_OUT)
        if hit and all(h in CAP.DESCRIBE_OK for h in hit):
            out.append(t)
    return out


def moved_tags(ds, ext=".txt") -> list[str]:
    return list((F.meta(ds).get("moved_tags") or {}).get(ext, {}).get("tags") or []) if ds else []


def move_tags(ds, tags, ext=".txt", trigger="") -> str:
    """Take the ticked tags out of every caption of the set and add them to the character description. The captions
    with all tags are kept in .lsflow/ first, so put_back() can undo it."""
    tags = [t.strip().lower() for t in tags or [] if t.strip()]
    if not tags:
        return "Tick the tags that belong to the character first."
    trig = (trigger or ds).strip().lower()
    tags = [t for t in tags if t != trig]
    keep = _moved_dir(ds, ext)
    kinds_keep = dsworn.worn_names(ds)[0] if F.meta(ds).get("purpose") == "worn" else []
    first = not keep.is_dir()
    keep.mkdir(parents=True, exist_ok=True)
    changed = 0
    for p in F.images(ds):
        f = set_file(p, ext)
        if not f.is_file():
            continue
        text = f.read_text(encoding="utf-8", errors="ignore")
        if first or not (keep / f.name).is_file():
            shutil.copy2(f, keep / f.name)                # the version with all tags
        parts = [w.strip() for w in re.split(r"[,\n]", text) if w.strip()]
        new = []
        for w in parts:
            if w.lower() not in set(tags):
                new.append(w)
            else:                                        # 'gold ring' leaves 'ring' (a kind of a worn item)
                new += [k for k in kinds_keep if re.search(rf"\b{re.escape(k)}\b", w.lower())]
        new = list(dict.fromkeys(new))
        if len(new) != len(parts):
            f.write_text(", ".join(new) + "\n", encoding="utf-8")
            changed += 1
    m = F.meta(ds)
    mv = m.setdefault("moved_tags", {}).setdefault(ext, {"tags": [], "desc_before": m.get("description", "")})
    mv["tags"] = list(dict.fromkeys(mv["tags"] + tags))
    F.meta_save(ds, m)
    desc = set_description(ds, ", ".join([description(ds)] + tags))
    return (f"✔ Moved {len(tags)} tag(s) out of {changed} caption(s) and next to the trigger word: *{desc}*. "
            "The captions with all tags are kept: **Put them back** undoes it.")


def put_back(ds, ext=".txt") -> str:
    """Undo move_tags: the captions with all tags back, the moved tags out of the description again."""
    keep = _moved_dir(ds, ext)
    m = F.meta(ds)
    mv = (m.get("moved_tags") or {}).get(ext)
    if not keep.is_dir() or not mv:
        return "Nothing was moved in this caption set, so there is nothing to put back."
    n = 0
    for p in F.images(ds):
        f = set_file(p, ext)
        if (keep / f.name).is_file():
            shutil.copy2(keep / f.name, f)
            n += 1
    shutil.rmtree(keep, ignore_errors=True)
    gone = {t.lower() for t in mv.get("tags") or []}
    m = F.meta(ds)
    m.get("moved_tags", {}).pop(ext, None)
    F.meta_save(ds, m)
    left = [w for w in re.split(r",", description(ds)) if w.strip() and w.strip().lower() not in gone]
    set_description(ds, ", ".join(left))
    return f"✔ Put back: {n} caption(s) have all their tags again; the moved tags are out of the description."


def suggest_description(ds) -> str:
    """The traits the WD tagger saw in at least 40 % of the pictures (left out of the captions): a starting point for
    the character description. Only hair, eyes, build, freckles, tattoos and such: never ethnicity, skin or age."""
    t = dict(F.meta(ds).get("trait_tags") or {})
    n = t.pop("_n", 0)
    if not n:
        return ""
    keep = [k for k, c in sorted(t.items(), key=lambda kv: -kv[1]) if c >= max(2, 0.4 * n)]
    return ", ".join(keep[:10])


def description(ds) -> str:
    return (F.meta(ds).get("description") or "").strip() if ds else ""


def set_description(ds, text: str) -> str:
    words = [w.strip() for w in re.split(r"[,\n]", text or "") if w.strip()]
    m = F.meta(ds)
    m["description"] = ", ".join(dict.fromkeys(words))
    F.meta_save(ds, m)
    return m["description"]
