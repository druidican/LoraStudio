# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Insights: what your own datasets, LoRAs and runs say (Collection › Insights, and short hints in the steps).

Everything is read on this computer from what LoRA Studio already has; nothing is sent anywhere.

1. Datasets behind good and weak LoRAs (and, the same way, the settings of the good ones: "settings that work for
   you", shown in step 9 next to the general recommendations). A version counts as GOOD when you rated it 4 or 5 on its LoRA page, or (not
   rated) when it is the ★ best version or you made pictures with it in ComfyUI; WEAK when you rated it 1 or 2.
   Unrated versions without either sign are left out. Per LoRA type the datasets of the good ones are compared with
   the others: pictures, pictures × repeats, framings, outfits, places, detail shots and caption length.
   The framings, outfits and places come from the dataset's sort (step 4) as it is NOW, so for an older version they
   are a guide only.
2. Your graphics card. Every run log in userdata/gui_logs/ with its launch script: LoRA type, batch size,
   resolution, whether it finished, ran out of video memory or failed otherwise, and the time per step. Per LoRA
   type: the batch sizes that worked, the ones that ran out of memory, and seconds per step, for a better time
   estimate in step 9.

3. Training too long? Per LoRA type, where in the run the save point to try first was (savepoint.py's advice, kept in
   the training card): usually early means fewer epochs would do (step 9 says how many).
4. How you use your LoRAs: from the picture index (LoRA page › Pictures), how many pictures you made with each LoRA,
   the strength you use (the LoRA page offers it as its strength) and when last; LoRAs never used.

With few LoRAs these are hints, not rules: every number says how many LoRAs or runs it comes from.
"""
from __future__ import annotations
__version__ = "1.111126g"  # LoRA Studio version, see version.py

import re
import statistics
import sys
import threading
import time
from collections import Counter
from pathlib import Path

MIN_PICS_USED = 3                 # pictures made with a version before it counts as used
_CACHE: dict = {"t": 0.0, "v": None}
_LOCK = threading.Lock()
TTL = 600


def _sg():
    return sys.modules.get("sdgui")


def _med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def _fmt(v, pct=False):
    if v is None:
        return "–"
    if pct:
        return f"{round(100 * v)} %"
    return f"{v:.0f}" if abs(v) >= 10 or float(v).is_integer() else f"{v:.1f}"


# --------------------------------------------------------------------------- 1. datasets
def verdict(lib, lora, counts: dict) -> tuple[str, str]:
    """('good' | 'weak' | '', why)."""
    import lora_library as LL
    import picgallery as PG
    r = int(lib.overrides(lora.key).get("rating") or 0)
    if r >= 4:
        return "good", f"rated {r}"
    if 1 <= r <= 2:
        return "weak", f"rated {r}"
    if r == 3:
        return "", "rated 3"
    members = lib.group_of(lora)
    if LL.best_member(members) is lora:
        return "good", "★ best version"
    n = counts.get(PG.norm(lora.stem), 0)
    if n >= MIN_PICS_USED:
        return "good", f"{n} pictures made with it"
    return "", ""


def _dataset_facts(ds_name: str, root: Path) -> dict:
    out = {}
    try:
        import dsflow as F
        res = F.sort_result(ds_name) or {}
        pics = [e for e in (res.get("pics") or {}).values() if e.get("cat") in ("good", "detail", "alike")]
        if pics:
            fr = Counter(e.get("framing") or "" for e in pics if e.get("cat") != "detail")
            n = max(1, sum(fr.values()))
            out["close"] = fr.get("close-up", 0) / n
            out["full"] = fr.get("full body", 0) / n
            out["outfits"] = len({e.get("outfit") for e in pics if e.get("outfit")})
            out["places"] = len({e.get("place") for e in pics if e.get("place")})
            out["detail"] = sum(1 for e in pics if e.get("cat") == "detail") / len(pics)
    except Exception:                                  # noqa: BLE001
        pass
    lens = []
    for d in sorted(root.iterdir()) if root.is_dir() else []:
        if d.is_dir() and re.match(r"^\d+_", d.name):
            for t in list(d.glob("*.txt"))[:300]:
                try:
                    txt = t.read_text(encoding="utf-8", errors="ignore").strip()
                except OSError:
                    continue
                if txt:
                    lens.append(len([w for w in re.split(r",\s*", txt) if w]) if txt.count(",") >= 3
                                else len(txt.split()))
    if lens:
        out["caption"] = statistics.median(lens)
    return out


DS_ROWS = [("pictures", "Pictures", False), ("samples", "Pictures × repeats per epoch", False),
           ("steps", "Steps", False), ("close", "Close-ups", True), ("full", "Full body", True),
           ("outfits", "Outfits", False), ("places", "Places", False), ("detail", "Detail shots", True),
           ("caption", "Caption length (tags or words)", False)]


def _num(v):
    m = re.match(r"\s*(\d+(?:\.\d+)?)", str(v or ""))
    return float(m.group(1)) if m else None


def dataset_rows(lib) -> list[dict]:
    """One row per version of your datasets with a verdict: {type, label, verdict, why, stem, ds, facts}."""
    import picgallery as PG
    import trainingcard as TC
    lib.ensure_loaded()
    counts = PG.name_counts()
    facts_ds: dict = {}
    rows = []
    for l in lib.loras.values():
        if l.dataset is None or not l.final:
            continue
        v, why = verdict(lib, l, counts)
        if not TC.is_scanned(l):                         # a version without an up-to-date training card: read it now
            try:
                TC.scan_one(lib, l)
            except Exception:                            # noqa: BLE001
                pass
        card = TC.stored(l.sidecar)
        f = dict(facts_ds.setdefault(l.dataset.name, _dataset_facts(l.dataset.name, l.dataset.root)))
        f["pictures"] = _num(card.get("pictures"))
        calc = re.match(r"(\d+) pictures with repeats", card.get("steps_calc") or "")
        f["samples"] = float(calc.group(1)) if calc else None
        f["steps"] = _num(card.get("steps"))
        st = {}                                          # the training settings, for "settings that work for you"
        try:
            st["lr"] = float(str(card.get("lr") or "").split()[0])
        except (ValueError, IndexError):
            pass
        nw = re.match(r"\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", card.get("network") or "")
        if nw:
            st["dim"], st["alpha"] = float(nw.group(1)), float(nw.group(2))
        ep = re.findall(r"\d+", card.get("epochs") or "")
        if ep:
            st["epochs"] = float(ep[-1])
        st["steps"] = _num(re.findall(r"\d+", card.get("steps") or "")[-1]) if re.findall(r"\d+", card.get("steps") or "") else None
        st["batch"] = _num(card.get("batch"))
        for k in ("optimizer", "resolution"):
            if card.get(k):
                st[k] = str(card[k])
        if card.get("scheduler"):
            st["scheduler"] = str(card["scheduler"]).split(" ·")[0]
        be = _best_epoch(card.get("best_epoch") or "")
        if be and st.get("epochs"):
            st["best_ratio"] = be / st["epochs"]
            st["best_epoch"] = be
        rows.append({"type": card.get("_tkey") or "", "label": card.get("lora_type") or l.base, "verdict": v,
                     "why": why, "stem": l.stem, "ds": l.dataset.name, "facts": f, "settings": st,
                     "gid": l.gid, "size": _size(l)})
    return rows


def _size(l) -> int:
    try:
        return Path(l.path).stat().st_size
    except (OSError, TypeError):
        return 0


def _best_epoch(text: str):
    """'Try epoch 10 first, then the final (12)' -> 10; 'Try the final (12) first' -> 12."""
    m = re.match(r"\s*Try epoch (\d+)", text)
    if m:
        return int(m.group(1))
    m = re.match(r"\s*Try the final \((\d+)\)", text)
    return int(m.group(1)) if m else None


# --------------------------------------------------------------------------- 1. settings that work for you
SET_NUM = [("lr", "Learning rate"), ("dim", "Network dim"), ("alpha", "Network alpha"), ("epochs", "Epochs"),
           ("steps", "Steps"), ("batch", "Batch size")]
SET_CAT = [("optimizer", "Optimizer"), ("scheduler", "Scheduler"), ("resolution", "Resolution")]


def _lr(v) -> str:
    return "–" if v is None else (f"{v:.0e}".replace("e-0", "e-") if 0 < v < 0.01 else f"{v:g}")


def settings_summary(rows: list[dict]) -> list[dict]:
    """Per LoRA type: {type, label, n_good, n_rest, rows: [(label, good, rest)], notes, good: {key: value}}."""
    out = []
    by: dict = {}
    for r in rows:
        by.setdefault(r["type"] or r["label"], []).append(r)
    for t, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        good = [r for r in rs if r["verdict"] == "good"]
        rest = [r for r in rs if r["verdict"] == "weak"] or [r for r in rs if r["verdict"] != "good"]
        table, notes, gv = [], [], {}
        for k, lab in SET_NUM:
            g = _med([r["settings"].get(k) for r in good])
            o = _med([r["settings"].get(k) for r in rest])
            if g is None and o is None:
                continue
            f = _lr if k == "lr" else _fmt
            table.append((lab, f(g), f(o)))
            if g is not None:
                gv[k] = g
            if g is not None and o is not None and len(good) >= 2 and len(rest) >= 2 and abs(g - o) / (max(g, o) or 1) >= 0.25:
                notes.append(f"{lab}: your good LoRAs used {'more' if g > o else 'less'} ({f(g)} against {f(o)}).")
        for k, lab in SET_CAT:
            cg = Counter(r["settings"].get(k) for r in good if r["settings"].get(k)).most_common(1)
            co = Counter(r["settings"].get(k) for r in rest if r["settings"].get(k)).most_common(1)
            if cg or co:
                table.append((lab, cg[0][0] if cg else "–", co[0][0] if co else "–"))
                if cg:
                    gv[k] = cg[0][0]
        out.append({"type": t, "label": rs[0]["label"], "n_good": len(good), "n_rest": len(rest), "rows": table,
                    "notes": notes, "good": gv})
    return out


# --------------------------------------------------------------------------- 2. training too long
def epochs_summary(rows: list[dict]) -> list[dict]:
    """Per LoRA type, over every version with a save-point advice: where in the run the one to try first was."""
    out = []
    by: dict = {}
    for r in rows:
        if r["settings"].get("best_ratio"):
            by.setdefault(r["type"] or r["label"], []).append(r)
    for t, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        ratios = [r["settings"]["best_ratio"] for r in rs]
        med = statistics.median(ratios)
        early = sum(1 for x in ratios if x <= 0.8)
        note = ""
        if len(rs) >= 3 and med <= 0.8:
            note = (f"the save point to try first was usually at about {round(100 * med)} % of the run "
                    f"({early} of {len(rs)} runs ended more than 20 % later than needed): fewer epochs would do")
        elif len(rs) >= 3:
            note = "the best save point was usually near the end: the runs were not too long"
        out.append({"type": t, "label": rs[0]["label"], "n": len(rs), "ratio": med, "early": early, "note": note,
                    "runs": [(r["stem"], r["settings"].get("best_epoch"), r["settings"].get("epochs")) for r in rs]})
    return out


# --------------------------------------------------------------------------- 5. how you use your LoRAs
def usage_rows(lib) -> list[dict]:
    """Per LoRA (all versions together): pictures made with it, the strength you use, when last, file sizes."""
    import lora_library as LL
    import picgallery as PG
    lib.ensure_loaded()
    pics = PG.index()["pics"]
    by_name: dict = {}
    for path, e in pics.items():
        when = (e.get("sig") or [0])[0]
        for n, st in (e.get("loras") or {}).items():
            by_name.setdefault(n, []).append((st, when))
    out = []
    for gid, members in lib.groups.items():
        mine = [m for m in members if m.dataset is not None]
        if not mine:
            continue
        names = {PG.norm(m.stem) for m in mine} | set(LL.picture_aliases(gid))
        uses = [u for n in names for u in by_name.get(n, [])]
        per = {m.stem: len(by_name.get(PG.norm(m.stem), [])) for m in mine}
        strengths = [s for s, _ in uses if isinstance(s, (int, float))]
        last = max((w for _, w in uses), default=0)
        out.append({"gid": gid, "name": mine[-1].display_name or gid, "label": mine[-1].base, "pictures": len(uses),
                    "strength": statistics.median(strengths) if strengths else None, "per_version": per,
                    "last": time.strftime("%Y-%m-%d", time.localtime(last)) if last else "",
                    "size": sum(_size(m) for m in mine), "versions": len(mine)})
    return sorted(out, key=lambda r: -r["pictures"])


def usage_of(lora) -> dict | None:
    """{pictures, strength} for one LoRA (all its versions), from the cached Insights; None when not known."""
    with _LOCK:
        hit = _USE if time.time() - _USE.get("t", 0) < TTL else None
    if hit is None:                                    # only the picture index: quick, also on the first LoRA page
        try:
            import lora_library as LL
            rows = usage_rows(LL.LIB)
        except Exception:                              # noqa: BLE001
            return None
        with _LOCK:
            _USE.clear()
            _USE.update(t=time.time(), rows={r["gid"]: r for r in rows})
        hit = _USE
    return hit["rows"].get(lora.gid)


_USE: dict = {}


def dataset_summary(rows: list[dict]) -> list[dict]:
    """Per LoRA type: {label, type, n_good, n_weak, n_other, rows: [(label, good, rest, pct)], notes}."""
    out = []
    by: dict = {}
    for r in rows:
        by.setdefault(r["type"] or r["label"], []).append(r)
    for t, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        good = [r for r in rs if r["verdict"] == "good"]
        weak = [r for r in rs if r["verdict"] == "weak"]
        rest = weak or [r for r in rs if r["verdict"] != "good"]
        table, notes = [], []
        for key, lab, pct in DS_ROWS:
            g = _med([r["facts"].get(key) for r in good])
            o = _med([r["facts"].get(key) for r in rest])
            if g is None and o is None:
                continue
            table.append((lab, g, o, pct))
            if g is not None and o is not None and len(good) >= 2 and len(rest) >= 2:
                big = max(abs(g), abs(o)) or 1
                if abs(g - o) / big >= (0.25 if not pct else 0.0) and (not pct or abs(g - o) >= 0.10):
                    more = "more" if g > o else "fewer" if not pct else "less"
                    notes.append(f"{lab}: your good LoRAs had {more} ({_fmt(g, pct)} against {_fmt(o, pct)}).")
        out.append({"type": t, "label": rs[0]["label"], "n_good": len(good), "n_weak": len(weak),
                    "n_rest": len(rest), "weak_is_rated": bool(weak), "rows": table, "notes": notes})
    return out


# --------------------------------------------------------------------------- 2. the graphics card
RUN_LOG_RE = re.compile(r"^(\d{8})_(\d{6})_(.+)\.log$")


def _speed(tail: str):
    """Seconds per step: the median of the last tqdm lines (it/s or s/it)."""
    vals = []
    for v, unit in re.findall(r"(\d+(?:\.\d+)?)(it/s|s/it)", tail)[-60:]:
        x = float(v)
        if x > 0:
            vals.append(1 / x if unit == "it/s" else x)
    return statistics.median(vals) if vals else None


def run_rows() -> list[dict]:
    """One row per run log: {type, label, batch, res, result, why, sps, when, script}."""
    sg = _sg()
    d = getattr(sg, "LOG_DIR", None) if sg is not None else None
    if not d or not Path(d).is_dir():
        return []
    import failcodes as FC
    labels = {}
    try:
        labels = {k: re.sub(r"\s*\[template\]", "", lab) for lab, k in sg.type_choices()}
    except Exception:                                  # noqa: BLE001
        pass
    rows = []
    for f in sorted(Path(d).glob("*.log")):
        m = RUN_LOG_RE.match(f.name)
        if not m:
            continue
        try:
            size = f.stat().st_size
            with open(f, "rb") as fh:
                head = fh.read(3000).decode("utf-8", "replace")
                fh.seek(max(0, size - 300_000))
                tail = fh.read().decode("utf-8", "replace")
        except OSError:
            continue
        sm = re.search(r"^##LS_SCRIPT (.+)$", head, re.M)
        script = Path(sm.group(1).strip()) if sm else None
        if script is not None and not script.is_file():
            alt = script.parent / "_old_runs" / script.name
            script = alt if alt.is_file() else None
        if script is None:
            continue
        try:
            text = script.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        home = sg.script_home(script)
        t = sg.script_type(str(script), text)
        batch = _num(sg.script_arg(text, "train_batch_size", home) or sg.script_arg(text, "batch_size", home)) or 1
        res = (sg.script_arg(text, "resolution", home) or "").replace(",", "×")
        ex = re.findall(r"--- exited with code (-?\d+) ---", tail)
        code = ex[-1] if ex else None
        if code == "0" and "--- FAILED:" not in tail:
            result, why = "done", ""
        elif code is None and sg.RUN.running() and sg.RUN.logfile and Path(sg.RUN.logfile) == f:
            continue                                   # the run that is going on now
        else:
            e = FC.explain(tail, code)
            result = "oom" if e and e["code"] == "OOM" else ("stopped" if code is None else "failed")
            why = e["title"] if e else ("stopped or cut off" if code is None else f"exit {code}")
        gpu = re.search(r"^##LS_GPU (.+)$", head, re.M)
        rows.append({"type": t, "label": labels.get(t, t or "?"), "batch": int(batch), "res": res, "result": result,
                     "why": why, "sps": _speed(tail) if result == "done" else None,
                     "when": f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:]}", "script": script.name,
                     "gpu": gpu.group(1).strip() if gpu else "",
                     "ckpt": "--gradient_checkpointing" in text, "swap": sg.script_arg(text, "blocks_to_swap", home)})
    return rows


def gpu_summary(rows: list[dict]) -> list[dict]:
    """Per LoRA type: {type, label, batches: [{batch, res, done, oom, failed, sps}], safe, oom_at, note}."""
    out = []
    by: dict = {}
    for r in rows:
        by.setdefault(r["type"] or r["label"], []).append(r)
    for t, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        cells: dict = {}
        for r in rs:
            c = cells.setdefault((r["batch"], r["res"]), {"batch": r["batch"], "res": r["res"], "done": 0, "oom": 0,
                                                          "failed": 0, "sps": []})
            c["done" if r["result"] == "done" else "oom" if r["result"] == "oom" else "failed"] += 1
            if r["sps"]:
                c["sps"].append(r["sps"])
        batches = sorted(cells.values(), key=lambda c: (c["res"], c["batch"]))
        for c in batches:
            c["sps"] = _med(c["sps"])
        ok = [c["batch"] for c in batches if c["done"]]
        oom = [c["batch"] for c in batches if c["oom"]]
        safe = max(ok) if ok else None
        low_oom = min([b for b in oom if safe is None or b > safe] or [None]) if oom else None
        if safe and low_oom:
            note = f"batch {safe} finished; batch {low_oom} ran out of video memory"
        elif safe:
            note = f"batch {safe} is the largest that finished"
        elif oom:
            note = f"batch {min(oom)} ran out of video memory; try 1 with gradient checkpointing"
        else:
            note = ""
        out.append({"type": t, "label": rs[0]["label"], "runs": len(rs), "batches": batches, "safe": safe,
                    "oom_at": low_oom, "note": note})
    return out


# --------------------------------------------------------------------------- cached, for the hints
def compute(lib=None, force=False) -> dict:
    with _LOCK:
        if not force and _CACHE["v"] is not None and time.time() - _CACHE["t"] < TTL:
            return _CACHE["v"]
    import lora_library as LL
    lib = lib or LL.LIB
    try:
        drows = dataset_rows(lib)
    except Exception as e:                             # noqa: BLE001
        print(f"[Insights] datasets: {e!r}")
        drows = []
    try:
        rrows = run_rows()
    except Exception as e:                             # noqa: BLE001
        print(f"[Insights] runs: {e!r}")
        rrows = []
    try:
        urows = usage_rows(lib)
    except Exception as e:                             # noqa: BLE001
        print(f"[Insights] usage: {e!r}")
        urows = []
    v = {"datasets": dataset_summary(drows), "drows": drows, "gpu": gpu_summary(rrows), "rrows": rrows,
         "settings": settings_summary(drows), "epochs": epochs_summary(drows), "usage": urows,
         "at": time.strftime("%H:%M")}
    with _LOCK:
        _CACHE.update(t=time.time(), v=v)
    return v


def forget():
    with _LOCK:
        _CACHE["v"] = None
        _USE.clear()


def dataset_hint(tkey: str) -> str:
    """Step 5: what the datasets of your good LoRAs of this type looked like ('' when there are fewer than 2)."""
    try:
        v = compute()
    except Exception:                                  # noqa: BLE001
        return ""
    s = next((x for x in v["datasets"] if x["type"] == tkey), None)
    if not s or s["n_good"] < 2:
        return ""
    vals = {lab: g for lab, g, _, _ in s["rows"]}
    bits = []
    if vals.get("Pictures") is not None:
        bits.append(f"{_fmt(vals['Pictures'])} pictures")
    if vals.get("Close-ups") is not None:
        bits.append(f"{_fmt(vals['Close-ups'], True)} close-ups")
    if vals.get("Full body") is not None:
        bits.append(f"{_fmt(vals['Full body'], True)} full body")
    if vals.get("Outfits") is not None:
        bits.append(f"{_fmt(vals['Outfits'])} outfits")
    return (f"Your {s['n_good']} good {s['label']} LoRAs had (median) " + ", ".join(bits) + ". Collection › Insights."
            if bits else "")


def settings_hint(tkey: str) -> str:
    """Step 9: the settings of your good LoRAs of this type ('' with fewer than 2)."""
    try:
        v = compute()
    except Exception:                                  # noqa: BLE001
        return ""
    s = next((x for x in v["settings"] if x["type"] == tkey), None)
    if not s or s["n_good"] < 2:
        return ""
    g = s["good"]
    bits = [f"learning rate {_lr(g['lr'])}" if g.get("lr") else "",
            f"dim {_fmt(g['dim'])} / alpha {_fmt(g['alpha'])}" if g.get("dim") and g.get("alpha") else "",
            f"{_fmt(g['epochs'])} epochs" if g.get("epochs") else "", f"{_fmt(g['steps'])} steps" if g.get("steps") else "",
            f"{g['optimizer']}" if g.get("optimizer") else ""]
    bits = [b for b in bits if b]
    return f"your {s['n_good']} good {s['label']} LoRAs (median): " + ", ".join(bits) if bits else ""


def epochs_hint(tkey: str, epochs: int | None) -> str:
    """Step 9: whether your runs of this type usually peaked early."""
    try:
        v = compute()
    except Exception:                                  # noqa: BLE001
        return ""
    s = next((x for x in v["epochs"] if x["type"] == tkey), None)
    if not s or s["n"] < 3 or s["ratio"] > 0.8:
        return ""
    if epochs:
        want = max(1, round(epochs * s["ratio"] + 0.49))
        return (f"in your {s['n']} runs of this type the save point to try first was at about {round(100 * s['ratio'])} "
                f"% of the run: about {want} of these {epochs} epochs may be enough (save points let you pick)")
    return s["note"]


# --------------------------------------------------------------------------- the page
def _sps(v) -> str:
    return f"{v:.2f}" if v else "–"


def page_html(v: dict, esc) -> str:
    H = [f"<p class='ll-muted'>Read {esc(v['at'])} from your datasets, LoRAs and run logs. A version counts as "
         "<b>good</b> when you rated it 4-5, or (not rated) when it is the ★ best version or you made pictures with it; "
         "<b>weak</b> when you rated it 1-2. Rate versions on their LoRA page › Edit.</p>"]
    H.append("<h3>Datasets behind good and weak LoRAs</h3>")
    ds = [s for s in v["datasets"] if s["n_good"] or s["n_weak"]]
    if not ds:
        H.append("<p class='ll-muted'>No version counts as good or weak yet: mark the best version in Training › "
                 "Results, rate versions on their LoRA page, or scan your ComfyUI pictures (LoRA page › Pictures).</p>")
    for s in ds:
        other = "weak (rated 1-2)" if s["weak_is_rated"] else "the others"
        H.append(f"<div class='ll-tcard' style='margin:8px 0'><div class='rs-label'>{esc(s['label'].upper())}</div>"
                 f"<p class='ll-muted'>{s['n_good']} good · {s['n_rest']} {esc(other)} (medians)</p>")
        if s["rows"]:
            H.append("<table class='rs-table'><thead><tr><th></th><th>Good</th><th>" + esc(other.split(' (')[0].capitalize())
                     + "</th></tr></thead><tbody>" + "".join(
                         f"<tr><td>{esc(lab)}</td><td><b>{_fmt(g, p)}</b></td><td>{_fmt(o, p)}</td></tr>"
                         for lab, g, o, p in s["rows"]) + "</tbody></table>")
        if s["notes"]:
            H.append("<ul>" + "".join(f"<li>{esc(n)}</li>" for n in s["notes"]) + "</ul>")
        elif s["n_good"] < 2 or s["n_rest"] < 2:
            H.append("<p class='ll-muted'>Too few LoRAs of this type on one side to compare yet (2 each are needed).</p>")
        H.append("</div>")
    H.append("<h3>Settings that work for you</h3>")
    st = [s for s in v.get("settings") or [] if s["n_good"]]
    if not st:
        H.append("<p class='ll-muted'>No good LoRAs yet to learn from (see above how a version counts as good).</p>")
    for s in st:
        H.append(f"<div class='ll-tcard' style='margin:8px 0'><div class='rs-label'>{esc(s['label'].upper())}</div>"
                 f"<p class='ll-muted'>{s['n_good']} good · {s['n_rest']} others (medians; the most used for names)</p>"
                 "<table class='rs-table'><thead><tr><th></th><th>Good</th><th>Others</th></tr></thead><tbody>"
                 + "".join(f"<tr><td>{esc(a)}</td><td><b>{esc(str(b))}</b></td><td>{esc(str(c))}</td></tr>"
                           for a, b, c in s["rows"]) + "</tbody></table>"
                 + ("<ul>" + "".join(f"<li>{esc(n)}</li>" for n in s["notes"]) + "</ul>" if s["notes"] else "")
                 + "</div>")
    H.append("<h3>Training too long?</h3>")
    ep = v.get("epochs") or []
    if not ep:
        H.append("<p class='ll-muted'>No finished runs with save points and a log yet.</p>")
    for s in ep:
        runs = " · ".join(f"{esc(n)}: {b} of {int(t)}" for n, b, t in s["runs"][:12])
        H.append(f"<div class='ll-tcard' style='margin:8px 0'><div class='rs-label'>{esc(s['label'].upper())}</div>"
                 f"<p>{s['n']} run{'s' if s['n'] != 1 else ''} · the save point to try first was at "
                 f"<b>{round(100 * s['ratio'])} %</b> of the run (median)"
                 + (f": {esc(s['note'])}" if s["note"] else " (3 runs are needed for advice)") + "</p>"
                 f"<p class='ll-muted ll-mono'>{runs}</p></div>")
    H.append("<h3>How you use your LoRAs</h3>")
    us = v.get("usage") or []
    if not any(u["pictures"] for u in us):
        H.append("<p class='ll-muted'>No pictures found yet: scan your ComfyUI picture folders (LoRA page › Pictures › "
                 "Scan all picture folders).</p>")
    else:
        rows = "".join(
            f"<tr><td>{esc(u['name'])}</td><td>{esc(u['label'])}</td><td>{u['pictures']}</td>"
            f"<td>{_sps(u['strength'])}</td>"
            f"<td>{esc(u['last'])}</td></tr>" for u in us if u["pictures"])
        H.append("<table class='rs-table'><thead><tr><th>LoRA</th><th>Type</th><th>Pictures</th><th>Strength you use</th>"
                 "<th>Last used</th></tr></thead><tbody>" + rows + "</tbody></table>")
        unused = [u for u in us if not u["pictures"]]
        if unused:
            mb = sum(u["size"] for u in unused) / 1e6
            H.append(f"<p><b>Never used in your pictures</b> ({len(unused)}, {mb:.0f} MB): "
                     + ", ".join(esc(u["name"]) for u in unused[:40])
                     + ". Candidates for a clean-up, or for a test (Training › Results › test pictures).</p>")
    H.append("<h3>Your graphics card</h3>")
    if not v["gpu"]:
        H.append("<p class='ll-muted'>No run logs yet (userdata/gui_logs/): they are written by every run started "
                 "from LoRA Studio.</p>")
    for s in v["gpu"]:
        rows = "".join(
            f"<tr><td>{c['batch']}</td><td>{esc(c['res'] or '–')}</td><td>{c['done']}</td>"
            f"<td>{'<b>' + str(c['oom']) + '</b>' if c['oom'] else 0}</td><td>{c['failed']}</td>"
            f"<td>{_sps(c['sps'])}</td></tr>" for c in s["batches"])
        H.append(f"<div class='ll-tcard' style='margin:8px 0'><div class='rs-label'>{esc(s['label'].upper())}</div>"
                 f"<p class='ll-muted'>{s['runs']} run{'s' if s['runs'] != 1 else ''}"
                 + (f" · {esc(s['note'])}" if s["note"] else "") + "</p>"
                 "<table class='rs-table'><thead><tr><th>Batch</th><th>Resolution</th><th>Finished</th>"
                 "<th>Out of video memory</th><th>Other failures</th><th>Seconds per step</th></tr></thead><tbody>"
                 + rows + "</tbody></table></div>")
    return "".join(H)
