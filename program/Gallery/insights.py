# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Insights: what your own datasets, LoRAs and runs say (Collection › Insights, and short hints in the steps).

Everything is read on this computer from what LoRA Studio already has; nothing is sent anywhere.

1. Datasets behind good and weak LoRAs. A version counts as GOOD when you rated it 4 or 5 on its LoRA page, or (not
   rated) when it is the ★ best version or you made pictures with it in ComfyUI; WEAK when you rated it 1 or 2.
   Unrated versions without either sign are left out. Per LoRA type the datasets of the good ones are compared with
   the others: pictures, pictures × repeats, framings, outfits, places, detail shots and caption length.
   The framings, outfits and places come from the dataset's sort (step 4) as it is NOW, so for an older version they
   are a guide only.
2. Your graphics card. Every run log in userdata/gui_logs/ with its launch script: LoRA type, batch size,
   resolution, whether it finished, ran out of video memory or failed otherwise, and the time per step. Per LoRA
   type: the batch sizes that worked, the ones that ran out of memory, and seconds per step, for a better time
   estimate in step 9.

With few LoRAs these are hints, not rules: every number says how many LoRAs or runs it comes from.
"""
from __future__ import annotations
__version__ = "1.261026z"  # LoRA Studio version, see version.py

import re
import statistics
import sys
import threading
import time
from collections import Counter
from pathlib import Path

RATING_LABEL = {0: "not rated", 1: "1 · poor", 2: "2 · weak", 3: "3 · fine", 4: "4 · good", 5: "5 · very good"}
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
        rows.append({"type": card.get("_tkey") or "", "label": card.get("lora_type") or l.base, "verdict": v,
                     "why": why, "stem": l.stem, "ds": l.dataset.name, "facts": f})
    return rows


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
    v = {"datasets": dataset_summary(drows), "drows": drows, "gpu": gpu_summary(rrows), "rrows": rrows,
         "at": time.strftime("%H:%M")}
    with _LOCK:
        _CACHE.update(t=time.time(), v=v)
    return v


def forget():
    with _LOCK:
        _CACHE["v"] = None


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


def gpu_hint(tkey: str, batch: int | None, steps: int | None) -> str:
    """Step 9: what your graphics card did with this type before, and a time estimate from your own runs."""
    try:
        v = compute()
    except Exception:                                  # noqa: BLE001
        return ""
    s = next((x for x in v["gpu"] if x["type"] == tkey), None)
    if not s:
        return ""
    bits = []
    if batch and s["oom_at"] and batch >= s["oom_at"]:
        bits.append(f"⚠ batch {s['oom_at']} ran out of video memory before for this type"
                    + (f"; batch {s['safe']} finished" if s["safe"] else ""))
    elif s["note"]:
        bits.append(s["note"].capitalize())
    sps = [c["sps"] for c in s["batches"] if c["sps"] and (batch is None or c["batch"] == batch)]
    if sps and steps:
        sec = statistics.median(sps) * steps
        h, m_ = divmod(int(sec // 60), 60)
        bits.append(f"about {f'{h} h {m_:02d} min' if h else f'{m_} min'} for {steps} steps "
                    f"({statistics.median(sps):.2f} s per step in your earlier runs)")
    return ("On your graphics card: " + "; ".join(bits) + ".") if bits else ""


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
