# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Fill the gaps the detailed scan found (step 5), for a new version of a LoRA:

  too few    pictures of a framing, angle, pose, expression, light, outfit or place: first your own ComfyUI pictures
             made with this dataset's LoRA (and no other LoRA) are looked through: Ollama tells what each one shows,
             the same questions as the sort and the detailed scan; what they cannot fill is made new with the LoRA
             (its newest version) through your ComfyUI test-picture workflow, with a prompt written for what is
             missing ("<trigger>, <class word>, close-up portrait, …")
  too many   one kind is most of the dataset: it can be outweighed by the new pictures, or some of those pictures
             (the ones the sort ranked lowest) get the 'never' pin, so Select leaves them out
Found and made pictures wait in .lsflow/gapfill/ (found ones as copies) until you pick: the ticked ones join the repeat
folder as found_<kind>_N / made_<kind>_N with an 'always keep' pin; the rest go to the Trash (the originals of found
pictures stay where they are and are not offered again). More than a quarter LoRA-made pictures gets a warning: a LoRA
trained on its own pictures learns its own mistakes harder. The prompts only name framing, angle, pose, expression, light,
outfit and place, never anything about the person's body or looks.
"""
from __future__ import annotations
__version__ = "1.171126d"  # LoRA Studio version, see version.py

import json
import random
import shutil
import time
from collections import Counter
from pathlib import Path

import detailscan as DSC
import dsflow as F

PHRASE = {
    ("framing", "close-up"): "close-up portrait, face in focus",
    ("framing", "upper body"): "upper body shot",
    ("framing", "full body"): "full body shot, whole figure visible from head to feet",
    ("angle", "front"): "facing the viewer",
    ("angle", "three-quarter"): "three-quarter view",
    ("angle", "side"): "side view, profile",
    ("angle", "back"): "from behind, back view",
    ("pose", "standing"): "standing",
    ("pose", "sitting"): "sitting on a chair",
    ("expression", "neutral"): "neutral expression",
    ("expression", "smiling"): "smiling",
}
NEW = {   # for 'not enough different ones': values the dataset does not have yet are taken from these
    "lighting": ["soft daylight", "warm indoor light", "evening light", "studio lighting", "overcast light"],
    "place": ["in a park", "on a city street", "in a cosy living room", "on a beach", "in a forest", "in a library",
              "in a cafe", "in a garden"],
    "outfit": ["wearing a casual hoodie and jeans", "wearing a smart coat", "wearing a t-shirt and shorts",
               "wearing a knitted sweater", "wearing a rain jacket"],
}
LABEL = {d: lab for d, (lab, _) in DSC.DIMS.items()}
TOO_MANY = 0.6                                           # one kind is more than 60 % of the kept pictures …
TOO_MANY_MIN = 6                                         # … and at least this many pictures
DIR = "gapfill"
SHARE_WARN = 0.25                                        # warn when more of the repeat folder than this is LoRA-made
MAX_ASK = 60                                             # pictures Ollama is asked about per search (answers are kept)


def _dir(ds) -> Path:
    return F.root(ds) / F.FLOW_DIR / DIR


def state(ds) -> dict:
    """{made: {file: {"dim", "value", "prompt", "seed", "lora"}}} of the pictures waiting to be picked."""
    try:
        d = json.loads((_dir(ds) / "made.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        d = {}
    d["made"] = {k: v for k, v in (d.get("made") or {}).items() if (_dir(ds) / k).is_file()}
    return d


def _save_state(ds, st: dict):
    _dir(ds).mkdir(parents=True, exist_ok=True)
    (_dir(ds) / "made.json").write_text(json.dumps(st, indent=1), encoding="utf-8")


# --------------------------------------------------------------------------- what is missing, what is too much
def analyse(ds) -> dict:
    """{"few": [(dim, value, count)], "many": [(dim, value, count, total)], "kept": n}: the detailed scan's answers
    for the pictures in the repeat folder now (what trains), not for every picture the scan looked at."""
    counts, _, _, kept = DSC.active(ds)
    if not kept:                                         # no answers for the folder yet: the scan's own numbers
        r = DSC.result(ds) or {}
        counts, kept = r.get("counts") or {}, int(r.get("kept") or 0)
    counts = {d: Counter(c) for d, c in counts.items()}
    few, many = [], []
    for d, (label, wanted) in DSC.DIMS.items():
        c = counts.get(d) or Counter()
        for v in wanted:
            if c.get(v, 0) < DSC.RARE and (d, v) in PHRASE:
                few.append((d, v, c.get(v, 0)))
        if d in DSC.VARIETY and len(c) < DSC.VARIETY[d]:
            for v in [x for x in NEW.get(d, []) if x not in c][:DSC.VARIETY[d] - len(c)]:
                few.append((d, v, 0))
        tot = sum(c.values())
        for v, n in c.items():
            if tot and n >= TOO_MANY_MIN and n / tot > TOO_MANY and d in ("framing", "angle", "pose", "expression"):
                many.append((d, v, n, tot))
                for w in wanted:                         # outweigh it with more of the other kinds
                    if w != v and (d, w) in PHRASE and not any(x[:2] == (d, w) for x in few):
                        few.append((d, w, c.get(w, 0)))
    return {"few": few, "many": many, "kept": kept}


# --------------------------------------------------------------------------- the LoRA that makes them
def lora_for(ds) -> dict | None:
    """The dataset's own LoRA to make pictures with: {"file", "name", "tkey", "fam", "label"} or None (with why)."""
    import comfytest as CT
    import lora_library as LL
    import training_ui as T
    cfg = CT.settings()
    LL.LIB.ensure_loaded()
    best = None
    for gid, members in LL.LIB.groups.items():
        if not members or members[-1].dataset is None or members[-1].dataset.name != ds:
            continue
        lo = LL.LIB.newest(gid)
        f = lo.dataset_path or lo.path if lo else None
        if not f or not Path(f).is_file():
            continue
        tkey = ""
        for s, _ in lo.scripts or []:
            info = T.script_info(str(s)) if Path(s).is_file() else None
            if info and info.get("tkey"):
                tkey = info["tkey"]
                break
        fam = CT.family(tkey)
        if not fam or not CT.workflow(cfg, fam)[0]:
            continue
        cand = {"file": str(f), "name": lo.stem, "tkey": tkey, "fam": fam, "label": f"{lo.stem} ({lo.vlabel})",
                "t": Path(f).stat().st_mtime}
        if best is None or cand["t"] > best["t"]:
            best = cand
    return best


def prompt(ds, dim: str, value: str) -> str:
    m = F.meta(ds)
    trig = (m.get("trigger") or ds).strip()
    cls = (m.get("class_word") or "").strip()
    what = PHRASE.get((dim, value)) or value
    return ", ".join(x for x in (trig, cls, what) if x)


# --------------------------------------------------------------------------- looking in your own pictures
def _groups(ds) -> list[str]:
    import lora_library as LL
    LL.LIB.ensure_loaded()
    return [g for g, m in LL.LIB.groups.items() if m and m[-1].dataset is not None and m[-1].dataset.name == ds]


def candidates(ds) -> list[dict]:
    """Your ComfyUI pictures made with this dataset's LoRA and no other LoRA, not offered before: newest version,
    highest strength and newest picture first. [{"path", "lora", "strength"}]"""
    import lora_library as LL
    seen = set(state(ds).get("seen") or [])
    out = []
    for gid in _groups(ds):
        members = LL.LIB.groups[gid]
        order = {m.key: i for i, m in enumerate(members)}
        for x in LL.group_pictures(gid):
            loras = (x["e"].get("loras") or {})
            if len(loras) != 1 or str(x["path"]) in seen or not Path(x["path"]).is_file():
                continue
            try:
                strength = float(x.get("strength") or 1.0)
            except (TypeError, ValueError):
                strength = 1.0
            out.append({"path": str(x["path"]), "lora": f"{x['lora'].stem} ({x['lora'].vlabel})", "strength": strength,
                        "rank": (order.get(x["lora"].key, -1), strength, Path(x["path"]).stat().st_mtime)})
    out.sort(key=lambda c: c["rank"], reverse=True)
    return out


def _what(url, model, p: Path, purpose: str) -> dict:
    """What a picture shows, in the words of the sort and the detailed scan; {"bad": reason} when it is not usable."""
    import datacheck as DC
    d = DC.content(url, model, p, detail=True)          # the sort's and the detailed scan's questions in one go
    bad = [t for s_, t in DC.content_issues(d, purpose) if s_ == DC.BAD]
    if bad:
        return {"bad": bad[0]}
    x = DSC.parse(d, *DC._sent_size(p, DC.SENT_SIDE))
    return {"framing": F._norm(d.get("framing")), "outfit": F._norm(d.get("outfit")), "place": F._norm(d.get("place")),
            **{k: x.get(k) or "" for k in ("angle", "pose", "expression", "lighting")}}


def search(ds, jobs: list[tuple[str, str, int]], say=None) -> tuple[dict, str]:
    """Fill the jobs from your own pictures. Returns ({(dim, value): found}, message); found pictures are copied into
    .lsflow/gapfill/ to be picked like made ones."""
    import features as FT
    say = say or (lambda t: None)
    S = F.sg()
    cands = candidates(ds)
    if not cands:
        return {}, "No pictures of your own with only this LoRA (scan your ComfyUI picture folders on the LoRA page)."
    url, model = S.oc_settings()
    if FT.have("ollama") and S.CAP is not None and not model:   # no default chosen: the first vision model
        try:
            model = next((m["name"] for m in S.CAP.list_models(url) if m["vision"]), "")
        except Exception:                                # noqa: BLE001
            model = ""
    if not (FT.have("ollama") and model and S.CAP is not None):
        return {}, "Your own pictures were not looked through: Ollama is not available (Settings › AI helpers)."
    counts, _, _, _ = DSC.active(ds)
    have = {d: set(c) for d, c in counts.items()}
    need = {(d, v): n for d, v, n in jobs if n > 0}
    purpose = F.meta(ds).get("purpose") or "character"
    cache_f = _dir(ds) / "found_cache.json"
    try:
        cache = json.loads(cache_f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cache = {}
    st = state(ds)
    out, used, asked = Counter(), set(), 0
    _dir(ds).mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(cands, 1):
        if not any(need[k] > out[k] for k in need) or F.STOP.is_set():
            break
        p = Path(c["path"])
        try:
            sig = f"{p}|{p.stat().st_size}|{int(p.stat().st_mtime)}"
        except OSError:
            continue
        w = cache.get(sig)
        if w is None:
            if asked >= MAX_ASK:
                continue
            say(f"⏳ Looking through your pictures with this LoRA: {i} of {len(cands)} (Ollama)")
            try:
                w = _what(url, model, p, purpose)
            except Exception as e:                       # noqa: BLE001
                return dict(out), f"Ollama stopped answering while looking through your pictures: {e}"
            cache[sig] = w
            asked += 1
            if asked % 5 == 0:
                cache_f.write_text(json.dumps(cache), encoding="utf-8")
        if w.get("bad"):
            continue
        for (dim, value), n in need.items():
            if out[(dim, value)] >= n:
                continue
            v = w.get(dim) or ""
            fits = (v and v not in have.get(dim, set()) and v not in used) if dim in DSC.VARIETY else v == value
            if fits:
                k = 1
                stem = f"found_{dim}_{value}".replace(" ", "-").replace("/", "-")
                while (_dir(ds) / f"{stem}_{k}{p.suffix.lower()}").exists():
                    k += 1
                fname = f"{stem}_{k}{p.suffix.lower()}"
                shutil.copy2(p, _dir(ds) / fname)
                st["made"][fname] = {"dim": dim, "value": v if dim in DSC.VARIETY else value, "prompt": "",
                                     "found": str(p), "lora": c["lora"], "strength": c["strength"]}
                st.setdefault("seen", []).append(str(p))
                if dim in DSC.VARIETY:
                    used.add(v)
                out[(dim, value)] += 1
                break
    cache_f.write_text(json.dumps(cache), encoding="utf-8")
    _save_state(ds, st)
    n = sum(out.values())
    left = len([c for c in cands if f"{c['path']}" not in set(st.get("seen") or [])])
    return dict(out), (f"✔ {n} of your own picture(s) with this LoRA fit a gap." if n else
                       "None of your own pictures with this LoRA fit the gaps.") + \
        (f" ({asked} looked at now; answers are kept, so the next search is quicker.)" if asked else "") + \
        (f" {left} not looked at yet: search again for more." if asked >= MAX_ASK and left else "")


def lora_made_share(ds) -> tuple[int, int]:
    """(LoRA-made pictures in the repeat folder: found and made ones, all pictures there)."""
    rows = F.sg().dataset_stats(ds)["rows"]
    if not rows:
        return 0, 0
    rep = Path(rows[0]["folder"])
    pics = [p for p in rep.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")]
    ours = set((F.meta(ds).get("made_pictures") or []))
    return sum(1 for p in pics if p.name in ours or p.name.startswith(("made_", "found_"))), len(pics)


def share_note(ds, extra: int = 0) -> str:
    n, tot = lora_made_share(ds)
    n, tot = n + extra, tot + extra
    if tot and n / tot > SHARE_WARN:
        return (f"⚠ {n} of {tot} pictures ({n / tot:.0%}) are made by the LoRA. More than a quarter: the LoRA learns "
                "its own mistakes harder (a face that drifts, hands). Real pictures are better where you have them.")
    return ""


def fill(ds, jobs: list[tuple[str, str, int]], say=None) -> str:
    """Search first, then make what is still missing."""
    say = say or (lambda t: None)
    found, msg1 = search(ds, jobs, say)
    rest = [(d, v, n - found.get((d, v), 0)) for d, v, n in jobs if n - found.get((d, v), 0) > 0]
    if not rest:
        return msg1 + " Nothing left to make. Tick the good ones and press *Add the ticked to the dataset*."
    if F.STOP.is_set():
        return msg1 + " Stopped."
    msg2 = make(ds, rest, say)
    return msg1 + "\n\n" + msg2


# --------------------------------------------------------------------------- making them
def make(ds, jobs: list[tuple[str, str, int]], say=None) -> str:
    """jobs = [(dim, value, how many)]. Makes the pictures into .lsflow/gapfill/. Returns a message."""
    import comfytest as CT
    say = say or (lambda t: None)
    S = F.sg()
    if S.RUN.running():
        return "❌ A training run is using the GPU: make the pictures when it is done."
    lo = lora_for(ds)
    if not lo:
        return ("❌ No LoRA of this dataset can make pictures: it needs a trained version and a workflow for its "
                "model family in Settings › Test pictures.")
    cfg = CT.settings()
    url = cfg["url"]
    if not CT.alive(url):
        return f"❌ ComfyUI does not answer at {url}. Start it and press the button again."
    try:
        import comfyupscale as U
        U._free_ollama()                                 # Ollama (the search) and the image model share the GPU
    except Exception:                                    # noqa: BLE001
        pass
    wf, ffam = CT.workflow(cfg, lo["fam"])
    if lo["fam"] == "vpred" and not wf.get("vpred"):
        wf = {**wf, "vpred": True, "zsnr": True}
    fixed = (cfg["fixed"].get(ffam) or "").strip().rstrip(",")
    st = state(ds)
    out = _dir(ds)
    out.mkdir(parents=True, exist_ok=True)
    total = sum(n for _, _, n in jobs)
    done, errors = 0, []
    rnd = random.Random()
    try:
        with CT.staged_lora(Path(lo["file"]), url) as name:
            for dim, value, n in jobs:
                for i in range(n):
                    if F.STOP.is_set():
                        return f"Stopped: {done} picture(s) made."
                    text = (fixed + ", " if fixed else "") + prompt(ds, dim, value)
                    seed = rnd.randint(1, 2 ** 31 - 1)
                    fname = f"{dim}_{value}_{int(time.time())}_{i + 1}.png".replace(" ", "-").replace("/", "-")
                    say(f"⏳ {LABEL.get(dim, dim)} · {value}: picture {i + 1} of {n} ({done + 1} of {total})")
                    try:
                        CT._run_one(url, CT.graph(wf, name, text, seed, f"lorastudio/gapfill_{ds}"), out / fname)
                    except Exception as e:               # noqa: BLE001
                        errors.append(str(e))
                        if done == 0:
                            return f"❌ {e}"
                        continue
                    st["made"][fname] = {"dim": dim, "value": value, "prompt": text, "seed": seed, "lora": lo["label"]}
                    _save_state(ds, st)
                    done += 1
    except RuntimeError as e:
        return f"❌ {e}"
    return (f"✔ {done} picture(s) made with {lo['label']}. Tick the good ones and press *Add the ticked to the dataset*."
            + (f" {len(errors)} failed: {errors[0][:200]}" if errors else ""))


# --------------------------------------------------------------------------- picking
def add(ds, files: list[str]) -> str:
    """The ticked pictures join the repeat folder (made_<kind>_N.png, pinned 'always keep'); the rest go to the Trash."""
    import housekeeping as HK
    st = state(ds)
    rows = F.sg().dataset_stats(ds)["rows"]
    if not rows:
        return "This dataset has no repeat folder."
    rep = Path(rows[0]["folder"])
    added = []
    for f in files:
        meta_ = st["made"].get(f)
        src = _dir(ds) / f
        if not meta_ or not src.is_file():
            continue
        kind = "found" if meta_.get("found") else "made"
        base = f"{kind}_{meta_['dim']}_{meta_['value']}".replace(" ", "-").replace("/", "-")
        ext = src.suffix.lower() or ".png"
        k = 1
        while (rep / f"{base}_{k}{ext}").exists():
            k += 1
        dest = rep / f"{base}_{k}{ext}"
        shutil.move(str(src), str(dest))
        F.set_pin(ds, dest, "keep")
        added.append(dest)
        st["made"].pop(f, None)
    F.ensure_originals(ds, added)
    rest = [_dir(ds) / f for f in st["made"]]
    HK.trash(rest)
    st["made"] = {}
    _save_state(ds, st)
    m = F.meta(ds)
    m.setdefault("made_pictures", [])
    m["made_pictures"] = sorted(set(m["made_pictures"]) | {p.name for p in added})
    F.meta_save(ds, m)
    warn = share_note(ds)
    return (f"✔ {len(added)} picture(s) added to {rep.name}/ (pinned 📌 always keep), {len(rest)} thrown away (Trash; "
            "the originals of your own pictures stay where they are). Write their captions in step 7."
            + (f"\n\n{warn}" if warn else ""))


def discard(ds) -> str:
    import housekeeping as HK
    st = state(ds)
    n, _ = HK.trash([_dir(ds) / f for f in st["made"]])
    st["made"] = {}
    _save_state(ds, st)
    return f"✔ {n} picture(s) thrown away (Trash)."


def set_aside(ds, dim: str, value: str, keep_share: float = 0.5) -> str:
    """Too many of one framing: the lowest-ranked of them get the 'never' pin until it is about keep_share. (Only the
    framing is stored per picture by the sort; angle, pose and expression are outweighed by making others.)"""
    r = DSC.result(ds) or {}
    tot = sum((r.get("counts") or {}).get(dim, {}).values())
    sr = F.sort_result(ds) or {}
    pins = F.pins(ds)
    pics = [(p, e) for p, e in (sr.get("pics") or {}).items()
            if (e.get(dim) or "") == value and e.get("cat") in DSC.KEPT and F.pin_of(pins, p) != "keep" and Path(p).is_file()]
    have = len(pics)
    others = tot - have
    target = max(1, int(others * keep_share / (1 - keep_share))) if others else have
    n = max(0, have - target)
    if not n:
        return f"Nothing to set aside: {have} {value} pictures against {others} others."
    pics.sort(key=lambda pe: pe[1].get("q") or 0)
    for p, _ in pics[:n]:
        F.set_pin(ds, p, "never")
    return (f"✔ {n} of the {have} {value} pictures got the 🚫 never pin (the ones the sort ranked lowest); Select leaves "
            "them out. Remove the pins in step 5's picture list to undo.")
