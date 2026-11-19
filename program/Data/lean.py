# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""The look of the result: what kind of pictures a dataset holds (photos, drawings, 3D renders, paintings: Ollama
tells per picture) and what the LoRA should lean to on each LoRA type (step 7, again per launch script in step 9).

Your captions say what is IN a picture, not what KIND of picture it is, so a LoRA made from photos ties the photo look
to the trigger word; on a drawing model (Illustrious, NoobAI, Pony, Anima) that look fights the model. So:
  * training adds the words that describe the pictures (photos: realistic, photorealistic, photo (medium) on the
    Danbooru-tag models) behind every caption with --caption_suffix (caption files stay as they are), and Pony's
    source_anime is no longer put in front of photos;
  * the lean decides where those words go when pictures are made: the model's own style → into the negative prompt,
    Photo → into the prompt; the test pictures and the sample prompts follow it.
The words come from the models' own pages (Danbooru tags for Illustrious, NoobAI and Anima; Pony V6's source_ tags,
which have none for photos).
"""
from __future__ import annotations
__version__ = "1.191126e"  # LoRA Studio version, see version.py

import json
import re
import time
from collections import Counter
from pathlib import Path

MEDIA = {"photo": "photos", "drawing": "drawings", "3d": "3D renders", "painting": "paintings"}
MIXED_SHARE = 0.2                 # more than this share of another kind: step 8 warns

# the words that describe a kind of picture, per word group
WORDS = {
    "booru": {"photo": "realistic, photorealistic, photo (medium)", "3d": "3d", "painting": "painting (medium)",
              "drawing": ""},
    "pony": {"photo": "photo, realistic", "3d": "3d", "painting": "painting", "drawing": ""},
    "plain": {"photo": "photo", "3d": "3d render", "painting": "painting", "drawing": "illustration"},
}
PONY_SCORE = "score_9, score_8_up, score_7_up"
PONY_SOURCE = {"anime": "source_anime", "cartoon": "source_cartoon"}
OWN = {"illustrious": "Manga (its own style)", "noobai": "Anime (its own style)", "anima": "Anime (its own style)"}
LEANS = {
    "booru": [("own", "Its own style"), ("photo", "Photo"), ("3d", "3D")],
    "anima": [("own", "Anime (its own style)"), ("3d", "3D")],        # Anima's card: it does not do realism
    "pony": [("anime", "Anime"), ("cartoon", "Western cartoon"), ("photo", "Photo")],
    "plain": [("photo", "Photo"), ("drawing", "Drawing")],
}
DEFAULT = {"booru": "own", "anima": "own", "pony": "cartoon", "plain": "photo"}
LEAN_MEDIUM = {"own": "drawing", "anime": "drawing", "cartoon": "drawing", "drawing": "drawing", "photo": "photo",
               "3d": "3d"}
REAL = re.compile(r"real|photo", re.I)      # Pony Realism, CyberRealistic, realismIllustrious, …


def _F():
    import dsflow
    return dsflow


# --------------------------------------------------------------------------- the kind of each picture
def norm_medium(s) -> str:
    s = str(s or "").strip().lower()
    if not s:
        return ""
    if "3d" in s or "render" in s or "cgi" in s:
        return "3d"
    if "photo" in s or "camera" in s:
        return "photo"
    if "paint" in s or "watercolo" in s or "oil" in s:
        return "painting"
    if any(w in s for w in ("draw", "illustr", "anime", "cartoon", "manga", "sketch", "comic", "digital art", "art")):
        return "drawing"
    return ""


def media(ds) -> dict:
    """{picture path relative to the dataset: kind} as Ollama told it."""
    return _F()._load(ds, "medium.json", {}) or {}


def remember(ds, found: dict):
    """found: {path: Ollama's answer (any wording)}; kept by the path relative to the dataset."""
    F = _F()
    root = F.root(ds)
    cur = media(ds)
    for p, v in found.items():
        m = norm_medium(v)
        if not m:
            continue
        try:
            cur[str(Path(p).relative_to(root))] = m
        except ValueError:
            cur[str(p)] = m
    F._save(ds, "medium.json", cur)


def summary(ds) -> dict:
    """{"counts": {kind: n}, "main": kind or '', "share": share of main among the known, "unknown": n,
    "mixed": another kind is more than MIXED_SHARE, "total": n, "others": the pictures not of the most common kind}
    over the pictures in the repeat folder."""
    F = _F()
    root = F.root(ds)
    have = media(ds)
    c, unknown, total, by = Counter(), 0, 0, {}
    for p in F.images(ds):
        total += 1
        try:
            k = have.get(str(p.relative_to(root)))
        except ValueError:
            k = None
        if k:
            c[k] += 1
            by.setdefault(k, []).append(str(p))
        else:
            unknown += 1
    known = sum(c.values())
    main, n = (c.most_common(1)[0] if c else ("", 0))
    return {"counts": dict(c), "main": main if known >= max(1, total // 2) else "", "share": n / known if known else 0,
            "unknown": unknown, "mixed": bool(known) and (known - n) / known > MIXED_SHARE, "total": total,
            "others": sorted(x for k, v in by.items() if k != main for x in v)}


def summary_text(sm: dict) -> str:
    if not sm["total"]:
        return "no pictures in the repeat folder yet"
    bits = [f"{n} {MEDIA[k] if n != 1 else MEDIA[k][:-1]}" for k, n in sorted(sm["counts"].items(), key=lambda x: -x[1])]
    if sm["unknown"]:
        bits.append(f"{sm['unknown']} not checked")
    return ", ".join(bits)


ASK = ('What kind of picture is this? Answer ONLY with a JSON object: {"medium": "<photo | drawing | 3d render | '
       'painting>"} (photo: taken with a camera, also edited photos; drawing: anime, manga, cartoon, comic, digital '
       'illustration; 3d render: CGI, a game or 3D model; painting: oil, watercolour, a traditional painting)')


def ask(url: str, model: str, img: Path, timeout: float = 300.0) -> str:
    import captioner as CAP
    import datacheck as DC
    r = CAP._req(url, "/api/chat", {
        "model": model, "stream": False, "format": "json", "keep_alive": "5m",
        "options": {"temperature": 0.0, "num_predict": 30},
        "messages": [{"role": "user", "content": ASK, "images": [CAP.image_b64(img, DC.SENT_SIDE)]}],
    }, timeout=timeout)
    txt = ((r.get("message") or {}).get("content") or r.get("response") or "").strip()
    try:
        return str(json.loads(re.sub(r"^```(json)?|```$", "", txt).strip()).get("medium") or "")
    except (ValueError, AttributeError):
        return txt


def check_run(ds, url: str, model: str, everything: bool = False):
    """Generator: asks Ollama the kind of every picture in the repeat folder that has none yet (or all of them)."""
    F = _F()
    import datacheck as DC
    from concurrent.futures import ThreadPoolExecutor
    F.STOP.clear()
    root = F.root(ds)
    have = media(ds)

    def known(p):
        try:
            return have.get(str(p.relative_to(root)))
        except ValueError:
            return None
    todo = [p for p in F.images(ds) if everything or not known(p)]
    if not todo:
        yield "✔ Every picture's kind is known: " + summary_text(summary(ds))
        return
    found, t0, err = {}, time.time(), ""
    with ThreadPoolExecutor(max_workers=DC.PARALLEL) as pool:
        futs = [(p, pool.submit(ask, url, model, p)) for p in todo]
        for i, (p, f) in enumerate(futs, 1):
            if F.STOP.is_set():
                err = "stopped"
                break
            eta = f", about {int((time.time() - t0) / max(i - 1, 1) * (len(todo) - i + 1))} s left" if i > 1 else ""
            yield f"Asking Ollama what kind of picture each one is: {i} of {len(todo)}{eta}"
            try:
                found[p] = f.result()
            except Exception as e:                       # noqa: BLE001
                err = f"Ollama stopped answering: {e}"
                break
            if i % 10 == 0:
                remember(ds, found)
        if err:
            for _, f in futs:
                f.cancel()
    remember(ds, found)
    yield ("❌ " + err + ". " if err else "✔ ") + "Kinds: " + summary_text(summary(ds))


# --------------------------------------------------------------------------- the lean per LoRA type
def group(tkey: str) -> str:
    """'booru' (Illustrious, NoobAI), 'anima', 'pony' or 'plain' (SDXL, FLUX, Chroma, Z-Image, Qwen-Image, …)."""
    import sd_script as SS
    try:
        _, presets = SS.merged_presets()
        p = presets.get(tkey) or {}
    except Exception:                                    # noqa: BLE001
        p = {}
    fam = SS.quality_family(tkey, p)
    if fam in ("illustrious", "noobai"):
        return "booru"
    if fam == "pony":
        return "pony"
    words = f"{tkey} {p.get('label', '')} {p.get('base', '')}".lower()
    return "anima" if "anima" in words else "plain"


def family(tkey: str) -> str:
    import sd_script as SS
    try:
        _, presets = SS.merged_presets()
        return SS.quality_family(tkey, presets.get(tkey) or {}) or ("anima" if group(tkey) == "anima" else "")
    except Exception:                                    # noqa: BLE001
        return ""


def realistic(tkey: str) -> bool:
    """The base model chosen in Settings › Model files has a realism name (Pony Realism, CyberRealistic …)."""
    try:
        import sd_settings as SSET
        files = SSET.settings_models(tkey)
    except Exception:                                    # noqa: BLE001
        return False
    return any(REAL.search(Path(v).name) for k, v in files.items() if k not in ("vae", "ae") and v)


def choices(tkey: str) -> dict:
    g = group(tkey)
    out = dict(LEANS[g])
    if g == "booru" and "own" in out:
        out["own"] = OWN.get(family(tkey), out["own"])
    return out


def default(tkey: str) -> str:
    g = group(tkey)
    if g in ("booru", "pony") and realistic(tkey):
        return "photo"
    return DEFAULT[g]


def lean_of(ds, tkey: str) -> str:
    v = (_F().meta(ds).get("lean") or {}).get(tkey)
    return v if v in choices(tkey) else default(tkey)


def set_lean(ds, tkey: str, v: str):
    F = _F()
    m = F.meta(ds)
    m.setdefault("lean", {})[tkey] = v
    F.meta_save(ds, m)


def _join(*xs) -> str:
    return ", ".join(x.strip().strip(",").strip() for x in xs if x and x.strip().strip(","))


def words(ds, tkey: str, sm: dict | None = None) -> dict:
    """What training and the pictures get for this dataset and type:
    {"medium", "lean", "train" (behind every caption), "pony_prefix" (Pony: the words in front, '' otherwise),
     "pos" (in the prompt of test and sample pictures), "neg" (in their negative prompt)}."""
    sm = sm or summary(ds)
    g = group(tkey)
    W = WORDS["booru" if g in ("booru", "anima") else g]
    med, lean = sm["main"], lean_of(ds, tkey)
    want = LEAN_MEDIUM[lean]
    out = {"medium": med, "lean": lean, "train": W.get(med, "") if med else "", "pony_prefix": "",
           "pos": W.get(want, "") if med else "", "neg": W.get(med, "") if med and med != want else ""}
    if g == "pony":
        src = PONY_SOURCE.get(lean, "")
        # in front while training: the source word only for drawn pictures (unknown: as before, source_anime)
        out["pony_prefix"] = _join(PONY_SCORE, (src or "source_anime") if med in ("drawing", "") else "")
        out["pos"] = _join(src, out["pos"])
        if lean == "photo":
            out["neg"] = _join(out["neg"], "source_anime, source_cartoon")
    return out


def explain(ds, tkey: str, w: dict | None = None) -> str:
    """One line for the cards: what training adds and what the pictures get."""
    w = w or words(ds, tkey)
    if not w["medium"]:
        return "The kind of the pictures is not known yet: nothing is added (Check picture kinds)."
    bits = [f"Training adds `{w['train']}` behind every caption" if w["train"] else "Training adds nothing"]
    if w["pony_prefix"]:
        bits.append(f"`{w['pony_prefix']}` in front")
    pics = []
    if w["pos"]:
        pics.append(f"`{w['pos']}` in the prompt")
    if w["neg"]:
        pics.append(f"`{w['neg']}` in the negative prompt")
    return "; ".join(bits) + (". Test and sample pictures: " + ", ".join(pics) + "." if pics else ".")


# --------------------------------------------------------------------------- into the launch script
def _n(words_: str) -> int:
    return len([x for x in words_.split(",") if x.strip()])


def apply_cfg(ds, tkey: str, cfg: dict) -> dict:
    """A new launch script's settings with the words (sd_script.make_plan): --caption_prefix / --caption_suffix
    (--keep_tokens follows the words in front), the musubi CAPTION_SUF, and the sample prompts' words."""
    import sd_script as SS
    w = words(ds, tkey)
    fam = SS.quality_family(tkey, cfg)
    q_pre, q_suf = SS.QUALITY_PREFIX.get(fam, ""), SS.QUALITY_SUFFIX.get(fam, "")
    pre = w["pony_prefix"] or q_pre
    suf = _join(w["train"], q_suf)
    flags = list(cfg.get("flags") or [])
    old = next((m.group(1) for f in flags for m in [re.search(r'--caption_prefix "([^"]*)"', f)] if m), "")
    flags = [f for f in flags if not re.match(r'\s*--caption_(prefix|suffix) ', f)]
    delta = _n(pre) - _n(old)
    if delta:
        flags = [re.sub(r"--keep_tokens\s+(\d+)", lambda m: f"--keep_tokens {max(0, int(m.group(1)) + delta)}", f)
                 for f in flags]
    if not SS.S.is_musubi_cfg(cfg):
        if pre:
            flags.append(f'--caption_prefix "{pre},"')
        if suf:
            flags.append(f'--caption_suffix ", {suf}"')
    cfg = {**cfg, "flags": flags, "caption_add": ", " + w["train"] if w["train"] else ""}
    s_pre = w["pony_prefix"] and _join(PONY_SCORE) or q_pre            # sample prompts: score words, then the lean's
    cfg["quality_words"] = (_join(s_pre, w["pos"] if fam == "pony" else ""), _join("" if fam == "pony" else w["pos"], q_suf))
    cfg["lean_negative"] = w["neg"]
    return cfg


_PRE = re.compile(r'--caption_prefix "([^"]*)"')
_SUF = re.compile(r'--caption_suffix "([^"]*)"')
_MSUF = re.compile(r'^CAPTION_SUF="([^"]*)"', re.M)


def in_text(text: str) -> tuple[str, str] | None:
    """(prefix, suffix) words a launch script trains with; None for an old musubi script without CAPTION_SUF."""
    if "CAPTION_EXT=" in (text or ""):
        m = _MSUF.search(text)
        return ("", m.group(1).lstrip(", ")) if m else None
    p, s = _PRE.search(text or ""), _SUF.search(text or "")
    return ((p.group(1).rstrip(",").strip() if p else ""), (s.group(1).lstrip(", ").strip() if s else ""))


def wanted(ds, tkey: str, text: str) -> tuple[str, str]:
    import sd_script as SS
    w = words(ds, tkey)
    if "CAPTION_EXT=" in (text or ""):
        return "", w["train"]
    fam = family(tkey)
    return (w["pony_prefix"] or SS.QUALITY_PREFIX.get(fam, "")), _join(w["train"], SS.QUALITY_SUFFIX.get(fam, ""))


def differs(ds, tkey: str, text: str) -> bool:
    have = in_text(text)
    if have is None or not (text or "").strip():
        return False
    return tuple(_join(x) for x in have) != tuple(_join(x) for x in wanted(ds, tkey, text))


def rewrite(ds, tkey: str, text: str) -> str:
    """The launch script text with this dataset's words (step 9's 'Use the chosen look')."""
    have = in_text(text)
    if have is None:
        return text
    pre, suf = wanted(ds, tkey, text)
    if "CAPTION_EXT=" in text:
        return _MSUF.sub(lambda m: f'CAPTION_SUF="{", " + suf if suf else ""}"', text)
    delta = _n(pre) - _n(have[0])
    if delta:
        text = re.sub(r"--keep_tokens\s+(\d+)", lambda m: f"--keep_tokens {max(0, int(m.group(1)) + delta)}", text)
    for rx, flag, val in ((_PRE, "--caption_prefix", f"{pre}," if pre else ""),
                          (_SUF, "--caption_suffix", f", {suf}" if suf else "")):
        if rx.search(text):
            if val:
                text = rx.sub(lambda m: f'{flag} "{val}"', text, count=1)
            else:                                        # drop the flag with its line break
                text = re.sub(r'[ \t]*' + flag + r' "[^"]*"[ \t]*\\?\n?', "", text, count=1)
        elif val:                                        # add it after --caption_extension (or the trainer's .py)
            at = r'(--caption_extension "[^"]*")' if "--caption_extension" in text else r'(accelerate launch[^\n]*?\.py)'
            text = re.sub(at, lambda m: f'{m.group(1)} \\\n  {flag} "{val}"', text, count=1)
    return text


def prompt_words(ds, tkey: str) -> tuple[str, str]:
    """(prompt words, negative words) for the test pictures of a LoRA of this dataset and type."""
    try:
        w = words(ds, tkey)
    except Exception:                                    # noqa: BLE001
        return "", ""
    return w["pos"], w["neg"]
