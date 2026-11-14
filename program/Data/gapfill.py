# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Fill the gaps the detailed scan found (step 5), for a new version of a LoRA:

  too few    pictures of a framing, angle, pose, expression, light, outfit or place: new pictures are made with the
             dataset's own LoRA (its newest or best version) through your ComfyUI test-picture workflow, with a prompt
             written for what is missing ("<trigger>, <class word>, close-up portrait, …")
  too many   one kind is most of the dataset: it can be outweighed by the new pictures, or some of those pictures
             (the ones the sort ranked lowest) get the 'never' pin, so Select leaves them out
The made pictures wait in .lsflow/gapfill/ until you pick: the ticked ones join the repeat folder as made_<kind>_N.png
with an 'always keep' pin; the rest go to the Trash. The prompts only name framing, angle, pose, expression, light,
outfit and place, never anything about the person's body or looks.
"""
from __future__ import annotations
__version__ = "1.141126b"  # LoRA Studio version, see version.py

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
    """{"few": [(dim, value, count)], "many": [(dim, value, count, total)], "kept": n} from the detailed scan."""
    r = DSC.result(ds) or {}
    counts = {d: Counter(c) for d, c in (r.get("counts") or {}).items()}
    kept = int(r.get("kept") or 0)
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
        if not fam or not cfg["families"].get(fam):
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
    wf = cfg["families"][lo["fam"]]
    fixed = (cfg["fixed"].get(lo["fam"]) or "").strip().rstrip(",")
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
        base = f"made_{meta_['dim']}_{meta_['value']}".replace(" ", "-").replace("/", "-")
        k = 1
        while (rep / f"{base}_{k}.png").exists():
            k += 1
        dest = rep / f"{base}_{k}.png"
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
    return (f"✔ {len(added)} picture(s) added to {rep.name}/ (pinned 📌 always keep), {len(rest)} thrown away (Trash). "
            "Write their captions in step 7.")


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
