# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Compare settings: train one dataset several times with ONE setting changed per run, make the same test pictures
of every run, rate them blind, and learn per LoRA type which settings make the best LoRAs.

  1. start(ds, tkey, groups): the standard launch script of the type (sd_script.make_plan) is the baseline; each
     variant changes one thing (learning rate, rank, text encoders). The scripts go to <dataset>/_compare/ and train
     into <dataset>/_compare/model (the Collection skips that folder), all in the queue as one group. "epochs" costs
     no run: the baseline's save points are compared too.
  2. after_queue() (called by comfytest.after_queue) makes the test pictures of every comparison whose runs are done:
     3 prompts from the dataset's captions × 2 seeds, the same for every variant, through ComfyUI.
  3. rows(): one row per prompt and seed with every variant's picture, shuffled and shown as A, B, C …; pick() keeps
     the best one or two (or "none" / "same").
  4. summary() per comparison and best() per LoRA type over all comparisons; use() makes a clear winner the default
     for new launch scripts of the type (apply(), called by sd_script.make_plan), like the speed test's faster
     settings.
Every comparison is one JSON file in userdata/compare/.
"""
from __future__ import annotations
__version__ = "1.261126b"  # LoRA Studio version, see version.py

import json
import random
import re
import sys
import threading
import time
from pathlib import Path

import lspaths as _LP

DIR = _LP.USER / "compare"
USE_FILE = _LP.USER / "compare_use.json"
SUB = "_compare"                       # in the dataset folder: the scripts, and model/ with their LoRA files
PROMPTS, SEEDS = 3, 2                  # 6 test pictures per variant
LR_CHOICES = ["5e-5", "1e-4", "2e-4"]
RANK_CHOICES = [(16, 16), (32, 32), (64, 64)]
GROUPS = {"lr": "Learning rate", "rank": "Rank (dim / alpha)", "te": "Text encoders", "epochs": "Steps / epochs"}
LETTERS = "ABCDEFGHIJKLMNOP"
_LOCK = threading.Lock()
JOB: dict = {"running": False, "id": "", "text": ""}


def _sg():
    return sys.modules.get("sdgui") or __import__("sdgui")


# --------------------------------------------------------------------------- the settings in a launch script
def lr_of(text: str) -> str:
    import sd_script
    return sd_script.form_value(text or "", "learning_rate")


def rank_of(text: str) -> tuple[int, int] | None:
    import sd_script
    d, a = sd_script.form_value(text or "", "network_dim"), sd_script.form_value(text or "", "network_alpha")
    try:
        return (int(float(d)), int(float(a))) if d and a else None
    except ValueError:
        return None


def te_trained(text: str) -> bool | None:
    """True: the text encoders learn; False: the image model only; None: the script has no text encoders to train."""
    if "--network_train_unet_only" in (text or ""):
        return False
    return True if re.search(r"--text_encoder_lr\b", text or "") else None


def _same_lr(a: str, b: str) -> bool:
    try:
        return float(a) == float(b)
    except ValueError:
        return a == b


def set_lr(text: str, v: str) -> str:
    import sd_settings
    return sd_settings.set_arg(text, "learning_rate", v)[0]


def set_rank(text: str, d: int, a: int) -> str:
    import sd_settings
    return sd_settings.set_arg(sd_settings.set_arg(text, "network_dim", str(d))[0], "network_alpha", str(a))[0]


def set_te(text: str, on: bool, lr: str = "5e-5") -> str:
    """on: the text encoders learn (at lr, unless the script has its own); off: the image model only."""
    import sd_settings
    import sd_script
    S = _sg()
    text = S.set_switch(text, "network_train_unet_only", not on)[0]
    if on and not sd_script.form_value(text, "text_encoder_lr"):
        text = sd_settings.set_arg(text, "text_encoder_lr", lr)[0]
    return text


def variants_for(text: str, groups: list[str]) -> list[dict]:
    """The runs a comparison trains: the baseline and one changed setting per variant.
    [{"key", "group", "label", "value"}]; "epochs" adds no run (the baseline's save points are compared)."""
    out = [{"key": "base", "group": "", "label": "your standard script", "value": ""}]
    lr = lr_of(text)
    if "lr" in groups and lr:
        for v in LR_CHOICES:
            if not _same_lr(v, lr):
                out.append({"key": "lr" + v.replace("-", ""), "group": "lr", "label": f"learning rate {v}", "value": v})
    rk = rank_of(text)
    if "rank" in groups and rk:
        for d, a in RANK_CHOICES:
            if (d, a) != rk:
                out.append({"key": f"r{d}x{a}", "group": "rank", "label": f"rank {d}/{a}", "value": f"{d}/{a}"})
    te = te_trained(text)
    if "te" in groups and te is not None:
        out.append({"key": "teoff" if te else "teon", "group": "te", "value": "off" if te else "on",
                    "label": "image model only" if te else "text encoders trained (5e-5)"})
    return out


def base_values(text: str) -> dict:
    rk = rank_of(text)
    te = te_trained(text)
    return {"lr": lr_of(text), "rank": f"{rk[0]}/{rk[1]}" if rk else "",
            "te": "" if te is None else ("on" if te else "off")}


def value_label(group: str, v: str) -> str:
    if group == "te":
        return {"on": "text encoders trained", "off": "image model only"}.get(v, v)
    if group == "rank":
        return f"rank {v}"
    if group == "lr":
        return f"learning rate {v}"
    return v


def apply_variant(text: str, var: dict) -> str:
    g, v = var["group"], var["value"]
    if g == "lr":
        return set_lr(text, v)
    if g == "rank":
        d, a = (int(x) for x in v.split("/"))
        return set_rank(text, d, a)
    if g == "te":
        return set_te(text, v == "on")
    return text


# --------------------------------------------------------------------------- 1. start a comparison
def _file(cid: str) -> Path:
    return DIR / f"{cid}.json"


def load(cid: str) -> dict:
    try:
        return json.loads(_file(cid).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save(exp: dict):
    DIR.mkdir(parents=True, exist_ok=True)
    tmp = _file(exp["id"]).with_suffix(".tmp")
    tmp.write_text(json.dumps(exp, indent=1), encoding="utf-8")
    tmp.replace(_file(exp["id"]))


def all_comparisons(ds: str | None = None) -> list[dict]:
    out = []
    for f in sorted(DIR.glob("*.json"), reverse=True) if DIR.is_dir() else []:
        try:
            e = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if ds is None or e.get("ds") == ds:
            out.append(e)
    return out


def preview(ds: str, tkey: str, groups: list[str]) -> dict:
    """What start() would train: {"variants", "runs", "steps", "text"} (no files written)."""
    import sd_script as SS
    plan = SS.make_plan(ds, tkey)
    text = plan.get("script_text") or ""
    vs = variants_for(text, groups)
    steps = (plan.get("rec") or {}).get("total_steps") or 0
    return {"variants": vs, "runs": len(vs), "text": text, "steps": steps, "hours": hours(tkey, steps, len(vs))}


def hours(tkey: str, steps: int, runs: int) -> float | None:
    """Rough training time of a comparison from the speed test's seconds per step (None when never measured)."""
    try:
        import speedjobs as SJ
        r = SJ.results().get(tkey) or {}
        sps = r.get("best_s") or r.get("baseline")
        sps = float(sps) if sps else None
    except Exception:                                    # noqa: BLE001
        sps = None
    if not sps or not steps:
        return None
    return round(runs * (steps * sps + 150) / 3600, 1)       # + about 2-3 minutes per run to load and save


def start(ds: str, tkey: str, groups: list[str]) -> str:
    """Write the scripts of a comparison, put them in the queue (as one group) and start it when nothing runs."""
    import dsfiles
    import sdqueue
    import sd_script as SS
    S = _sg()
    if not ds or not tkey:
        return "❌ Pick a dataset and a LoRA type."
    groups = [g for g in groups if g in GROUPS]
    if not groups:
        return "❌ Tick at least one setting to compare."
    plan = SS.make_plan(ds, tkey)
    text = plan.get("script_text") or ""
    if not text.strip() or "accelerate launch" not in text and "MUSUBI_DIR=" not in text:
        return "❌ No launch script could be made for this type (see step 9)."
    vs = variants_for(text, groups)
    if len(vs) < 2 and "epochs" not in groups:
        return "❌ This type has none of these settings to change."
    cid = time.strftime("%Y%m%d_%H%M%S")
    home = dsfiles.lora_path(ds)
    sub = home / SUB
    sub.mkdir(exist_ok=True)
    rec = plan.get("rec") or {}
    if rec.get("rename") and rec.get("repeats"):        # the repeats the standard script wants, as step 9 does
        try:
            SS._auto_rename(ds, rec["repeats"][0])
        except Exception:                                # noqa: BLE001
            pass
    pp = Path(str(plan.get("prompts_path") or "")).expanduser() if plan.get("prompts_path") else None
    if pp and SS.has_samples(text) and (plan.get("prompts_text") or "").strip() and not pp.exists():
        pp.write_text(plan["prompts_text"].rstrip("\n") + "\n", encoding="utf-8")
    bl = base_values(text)
    runs = []
    for v in vs:
        t = apply_variant(text, v)
        t = t.replace(SS.SCRIPT_DIR_LINE, 'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   '
                      '# in _compare/: the dataset is one up', 1)
        t = re.sub(r'^OUTPUT_DIR="\$\{TRAIN_DIR\}/model"', f'OUTPUT_DIR="${{TRAIN_DIR}}/{SUB}/model"', t, count=1,
                   flags=re.M)
        t = re.sub(r'--output_name "([^"]*)"', lambda m: f'--output_name "{m.group(1)}_cmp{cid[-6:]}{v["key"]}"', t,
                   count=1)
        t = re.sub(r"^(#!/bin/bash\n)", rf"\1# LoRA Studio comparison {cid}: {v['label']}\n", t, count=1)
        f = sub / f"cmp{cid}_{v['key']}.sh"
        f.write_text(S.fix_flags(t)[0].rstrip("\n") + "\n", encoding="utf-8")
        f.chmod(0o755)
        runs.append({**v, "script": str(f)})
    exp = {"id": cid, "ds": ds, "tkey": tkey, "groups": groups, "base": bl, "runs": runs, "created": time.time(),
           "prompts": [], "pics": {}, "rows": [], "state": "training"}
    save(exp)
    with sdqueue.QLOCK:
        for r in runs:
            sdqueue.QUEUE["items"].append({"path": r["script"], "resume": "", "state": "queued",
                                     "note": f"comparison {cid[-6:]}: {r['label']}"})
    msg = f"✔ Comparison started: {len(runs)} run(s) in the queue ({', '.join(r['label'] for r in runs)})."
    if "epochs" in groups:
        msg += " The save points of your standard script are compared too."
    if not sdqueue.QUEUE.get("running") and not S.RUN.running():
        sdqueue.queue_start(sdqueue.QUEUE.get("stop_on_fail", True))
        if sdqueue.QUEUE.get("running"):
            msg += " Training starts now."
    else:
        msg += " They train after what is in the queue now."
    return msg


# --------------------------------------------------------------------------- 2. state and test pictures
def _files(script: str):
    """(final file, {epoch: file}) of a run."""
    import training_ui as T
    if not Path(script).is_file():
        return None, {}
    return T._final_and_epochs(T.script_info(script))


def run_states(exp: dict) -> dict:
    """{key: queued | running | done | failed | gone}."""
    import sdqueue
    q = {i["path"]: i["state"] for i in sdqueue.QUEUE["items"]}
    out = {}
    for r in exp["runs"]:
        st = q.get(r["script"])
        final, _ = _files(r["script"])
        if final is not None and st not in ("running", "queued"):
            out[r["key"]] = "done"
        elif st in ("queued", "running"):
            out[r["key"]] = st
        elif st in ("failed", "stopped"):
            out[r["key"]] = "failed"
        else:
            out[r["key"]] = "gone" if final is None else "done"
    return out


def entries(exp: dict) -> list[dict]:
    """What gets test pictures: every finished run, and (epochs) the baseline's save points at about 1/2 and 3/4.
    [{"key", "label", "file", "group", "value"}]."""
    st = run_states(exp)
    out = []
    for r in exp["runs"]:
        if st.get(r["key"]) != "done":
            continue
        final, eps = _files(r["script"])
        out.append({"key": r["key"], "label": r["label"], "file": str(final), "group": r["group"], "value": r["value"]})
        if r["key"] == "base" and "epochs" in exp["groups"] and eps:
            keys = sorted(eps)
            last = keys[-1] + 1
            for frac in (0.5, 0.75):
                k = min(keys, key=lambda e: abs(e - last * frac))
                if not any(x["key"] == f"ep{k}" for x in out):
                    out.append({"key": f"ep{k}", "label": f"save point epoch {k} of {last}", "file": str(eps[k]),
                                "group": "epochs", "value": f"{round(100 * k / last)}%"})
    return out


def _prompts(exp: dict) -> list[dict]:
    """3 captions of the dataset (picked once and kept) × 2 seeds: [{"text", "seed"}]."""
    import dsfiles
    if exp.get("prompts"):
        return exp["prompts"]
    import dsflow as F
    caps = []
    for im in F.images(exp["ds"]):
        c = dsfiles.read_caption(im)
        if c:
            caps.append(c)
    rnd = random.Random(exp["id"])
    rnd.shuffle(caps)
    caps = caps[:PROMPTS] or [exp["ds"]]
    seeds = [rnd.randint(1, 2 ** 31 - 1) for _ in range(SEEDS)]
    exp["prompts"] = [{"text": c, "seed": s} for c in caps for s in seeds]
    return exp["prompts"]


def pic_dir(exp: dict) -> Path:
    import dsfiles
    return dsfiles.lora_path(exp["ds"]) / SUB / "pictures" / exp["id"]


def make_pictures(cid: str, say=None) -> str:
    """The test pictures of every finished run that has none yet. Returns a message (✔ / ❌)."""
    import comfytest as CT
    say = say or (lambda t: None)
    S = _sg()
    exp = load(cid)
    if not exp:
        return "❌ This comparison is gone."
    if S.RUN.running():
        return "❌ A training run is using the GPU: the pictures are made after the queue."
    ents = entries(exp)
    if not ents:
        return "❌ No run of this comparison has finished yet."
    try:
        _cfg, wf, url, root, fixed = CT.prepare(exp["ds"], exp["tkey"])
    except RuntimeError as e:
        return f"❌ {e}"
    prompts = _prompts(exp)
    save(exp)
    d = pic_dir(exp)
    todo = [(e, i) for e in ents for i in range(len(prompts))
            if not (exp["pics"].get(e["key"]) or [None] * len(prompts))[i]
            or not Path((exp["pics"].get(e["key"]) or [""] * len(prompts))[i] or "").is_file()]
    if not todo:
        return "✔ Every finished run has its test pictures."
    n, total = 0, len(todo)
    JOB.update(running=True, id=cid, text=f"0 of {total}")
    try:
        by_file: dict = {}
        for e, i in todo:
            by_file.setdefault(e["file"], []).append((e, i))
        for f, items in by_file.items():
            with CT.staged_lora(Path(f), url) as name:
                for e, i in items:
                    n += 1
                    JOB["text"] = f"{n} of {total}: {e['label']}"
                    say(f"⏳ picture {n} of {total}: {e['label']}")
                    pr = prompts[i]
                    dest = d / f"{e['key']}_{i + 1}.png"
                    text = (fixed + ", " if fixed else "") + pr["text"]
                    CT._run_one(url, CT.graph(wf, name, text, pr["seed"], f"lorastudio/compare_{cid}"), dest)
                    exp = load(cid) or exp
                    exp.setdefault("pics", {}).setdefault(e["key"], [None] * len(prompts))[i] = str(dest)
                    save(exp)
    except Exception as e:                               # noqa: BLE001
        return f"❌ Stopped after {n - 1} of {total} pictures: {e}"
    finally:
        JOB.update(running=False, text="")
        CT.free_memory(url)
    exp = load(cid)
    _make_rows(exp)
    save(exp)
    return f"✔ {total} test pictures made: rate them in Results › Compare settings."


def make_in_background(cid: str) -> str:
    if JOB["running"]:
        return "Test pictures of a comparison are being made right now."

    def work():
        msg = make_pictures(cid)
        exp = load(cid)
        if exp:
            exp["msg"] = msg
            save(exp)
    threading.Thread(target=work, daemon=True, name="compare-pics").start()
    return "Making the test pictures in ComfyUI …"


def after_queue():
    """After the training queue: make the pictures of every comparison whose runs are all finished (when ComfyUI
    answers; tried again every minute for up to 6 hours)."""
    def ready():
        out = []
        for e in all_comparisons():
            st = run_states(e)
            if not any(v in ("queued", "running") for v in st.values()) and any(v == "done" for v in st.values()):
                ents = entries(e)
                have = e.get("pics") or {}
                if any(not have.get(x["key"]) or None in have[x["key"]] for x in ents):
                    out.append(e["id"])
        return out

    def work():
        import comfytest as CT
        t0 = time.time()
        while time.time() - t0 < 6 * 3600:
            ids = ready()
            if not ids:
                return
            S = _sg()
            if S.RUN.running() or not CT.alive():
                time.sleep(60)
                continue
            for cid in ids:
                msg = make_pictures(cid)
                e = load(cid)
                if e:
                    e["msg"] = msg
                    save(e)
            return
    if not any(t.name == "compare-after" for t in threading.enumerate()):
        threading.Thread(target=work, daemon=True, name="compare-after").start()


# --------------------------------------------------------------------------- 3. rating
def _make_rows(exp: dict):
    """One row per prompt and seed with the variants that have that picture, shuffled once (kept)."""
    old = {(r["i"]): r for r in exp.get("rows") or []}
    rows = []
    rnd = random.Random(exp["id"] + "rows")
    for i in range(len(exp.get("prompts") or [])):
        keys = [k for k, files in (exp.get("pics") or {}).items() if i < len(files) and files[i]]
        if len(keys) < 2:
            continue
        o = old.get(i)
        if o and set(o["order"]) == set(keys):
            rows.append(o)
            continue
        rnd.shuffle(keys)
        rows.append({"i": i, "order": keys, "pick": None})
    exp["rows"] = rows


def rows(cid: str) -> list[dict]:
    """[{"i", "prompt", "order", "pick", "pics": [path per order]}] (no names: the pictures are shown as A, B …)."""
    exp = load(cid)
    if not exp:
        return []
    if not exp.get("rows") or any(len(r["order"]) != len([k for k, f in exp["pics"].items() if r["i"] < len(f) and f[r["i"]]])
                                  for r in exp["rows"]):
        _make_rows(exp)
        save(exp)
    out = []
    for r in exp["rows"]:
        out.append({**r, "prompt": exp["prompts"][r["i"]]["text"],
                    "pics": [exp["pics"][k][r["i"]] for k in r["order"]]})
    return out


def pick(cid: str, i: int, pick_):
    """pick_: a list of 1-2 keys, "none", "same", or None (undo)."""
    with _LOCK:
        exp = load(cid)
        for r in exp.get("rows") or []:
            if r["i"] == i:
                r["pick"] = pick_
        save(exp)


def labels(exp: dict) -> dict:
    out = {r["key"]: r["label"] for r in exp["runs"]}
    for e in entries(exp):
        out.setdefault(e["key"], e["label"])
    return out


# --------------------------------------------------------------------------- 4. results
def summary(exp: dict) -> dict:
    """{"rated", "rows", "wins": {key: n}, "groups": {group: {"winner": key or "", "clear": bool, "counts": {key: n},
    "text"}}}. A pick counts for every picked picture; per group only the baseline and that group's variants count."""
    rws = exp.get("rows") or []
    rated = [r for r in rws if r.get("pick")]
    wins: dict = {}
    for r in rated:
        if isinstance(r["pick"], list):
            for k in r["pick"]:
                wins[k] = wins.get(k, 0) + 1
    lab = labels(exp)
    groups = {}
    for g in exp["groups"]:
        members = ["base"] + [r["key"] for r in exp["runs"] if r["group"] == g]
        if g == "epochs":
            members = ["base"] + sorted({k for r in rws for k in r["order"] if k.startswith("ep")})
        counts = {k: wins.get(k, 0) for k in members if any(k in r["order"] for r in rws)}
        if not counts:
            continue
        top = sorted(counts.items(), key=lambda kv: -kv[1])
        best, n1 = top[0]
        n2 = top[1][1] if len(top) > 1 else 0
        clear = n1 >= 3 and n1 >= 2 * n2                # at least 3 picks, and twice the next one
        name = lab.get(best, best) if best != "base" else _base_label(exp, g)
        text = (f"{name} won {n1} of {len(rated)} rows" if n1 else "no picks yet") + ("" if clear or not n1 else
                                                                                       " (not clearly)")
        groups[g] = {"winner": best if n1 else "", "clear": clear, "counts": counts, "text": text}
    return {"rated": len(rated), "rows": len(rws), "wins": wins, "groups": groups,
            "none": sum(1 for r in rated if r["pick"] == "none"), "same": sum(1 for r in rated if r["pick"] == "same")}


def _base_label(exp: dict, g: str) -> str:
    if g == "epochs":
        return "the final file"
    return value_label(g, exp["base"].get(g, "")) + " (your standard)"


def winner_value(exp: dict, g: str, key: str) -> str:
    if key == "base":
        return exp["base"].get(g, "") if g != "epochs" else "100%"
    for r in exp["runs"]:
        if r["key"] == key:
            return r["value"]
    for e in entries(exp):
        if e["key"] == key:
            return e["value"]
    return ""


def best(tkey: str | None = None) -> dict:
    """Over every rated comparison: {tkey: {group: {"wins": {value: n comparisons}, "n": comparisons, "offer": value
    or '', "text"}}}. A value is offered when it won clearly in at least 2 comparisons and nothing else won as often."""
    out: dict = {}
    for e in all_comparisons():
        if tkey and e["tkey"] != tkey:
            continue
        sm = summary(e)
        if not sm["rated"]:
            continue
        for g, r in sm["groups"].items():
            d = out.setdefault(e["tkey"], {}).setdefault(g, {"wins": {}, "n": 0, "unclear": 0})
            d["n"] += 1
            if r["winner"] and r["clear"]:
                v = winner_value(e, g, r["winner"])
                d["wins"][v] = d["wins"].get(v, 0) + 1
            else:
                d["unclear"] += 1
    for t, gs in out.items():
        for g, d in gs.items():
            top = sorted(d["wins"].items(), key=lambda kv: -kv[1])
            d["offer"] = ""
            if top and top[0][1] >= 2 and (len(top) == 1 or top[0][1] > top[1][1]) and g != "epochs":
                d["offer"] = top[0][0]
            parts = [f"{value_label(g, v)}: {n}" for v, n in top]
            d["text"] = (f"clear winner in {sum(d['wins'].values())} of {d['n']} comparison(s): " + ", ".join(parts)
                         if top else f"no clear winner in {d['n']} comparison(s)")
    return out


# --------------------------------------------------------------------------- the winner as the default
def uses() -> dict:
    try:
        return json.loads(USE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def set_use(tkey: str, group: str, value: str | None) -> str:
    u = uses()
    t = u.setdefault(tkey, {})
    if value:
        t[group] = value
    else:
        t.pop(group, None)
    if not t:
        u.pop(tkey, None)
    USE_FILE.parent.mkdir(parents=True, exist_ok=True)
    USE_FILE.write_text(json.dumps(u, indent=1), encoding="utf-8")
    return (f"✔ New launch scripts of this type use {value_label(group, value)}." if value else
            "✔ New launch scripts of this type use the standard setting again.")


def apply(tkey: str, text: str) -> tuple[str, list]:
    """A new launch script with the settings that won comparisons for this type (and ticked to use): (text, notes)."""
    u = uses().get(tkey) or {}
    notes = []
    for g, v in u.items():
        if g in ("lr", "rank", "te") and v:
            text = apply_variant(text, {"group": g, "value": v})
            notes.append(f"{value_label(g, v)}: it won your comparisons of this type (Collection › Insights).")
    return text, notes


def delete(cid: str) -> str:
    """A comparison's scripts, LoRA files (and save points), sample pictures and test pictures to the Trash."""
    import sdqueue
    import housekeeping as HK
    exp = load(cid)
    if not exp:
        return "This comparison is gone already."
    if any(v in ("queued", "running") for v in run_states(exp).values()):
        with sdqueue.QLOCK:                                    # its waiting runs leave the queue; a running one goes on
            sdqueue.QUEUE["items"] = [i for i in sdqueue.QUEUE["items"]
                                if not (i["state"] == "queued" and i["path"] in {r["script"] for r in exp["runs"]})]
    paths = []
    for r in exp["runs"]:
        if r["script"] and Path(r["script"]).is_file():
            paths.append(r["script"])
        final, eps = _files(r["script"]) if Path(r["script"]).is_file() else (None, {})
        paths += [str(f) for f in [final, *eps.values()] if f]
    d = pic_dir(exp)
    if d.is_dir():
        paths.append(str(d))
    n, errs = HK.trash(paths) if paths else (0, [])
    _file(cid).unlink(missing_ok=True)
    return f"✔ Comparison deleted ({n} file(s) to the Trash)." + (f" Not moved: {', '.join(errs[:3])}" if errs else "")
