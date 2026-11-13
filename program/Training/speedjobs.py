# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Speed tests inside LoRA Studio (Settings › Speed test), at your own pace, one or more LoRA types at a time.

  test dataset   userdata/speedtest/lsspeedtest/: 20 invented pictures with captions, made once; none of your
                 datasets is used and nothing appears in the Collection (a dot-free name inside userdata)
  a test         per ticked type: a temporary launch script for that dataset (exactly as step 9 would write it),
                 then speedtest.test_script() on it, in the background; Stop ends it after the running variant
  results        userdata/speedtest/results.json (per type: when, how deep, now and best s/step, what changed) and
                 the reports in userdata/speedtest/reports/
  use            userdata/speedtest/use.json: per type, "use the faster settings": new launch scripts of that type
                 get the changes and the environment values of its best result (sd_script.make_plan → apply())

Nothing runs while a training run is active, and the training queue does not start while a speed test runs.
"""
from __future__ import annotations
__version__ = "1.111126b"  # LoRA Studio version, see version.py

import json
import shlex
import threading
import time
from pathlib import Path

import lspaths as _LP

DIR = _LP.USER / "speedtest"
DATASET = DIR / "lsspeedtest"
RESULTS = DIR / "results.json"
USE = DIR / "use.json"
DEPTHS = [("Quick: the settings that usually matter most (about 20-40 minutes per type)", "quick"),
          ("Full: every setting, then the faster ones together (about 1-2 hours per type)", "full"),
          ("Only how fast it trains now (a few minutes per type)", "now")]
JOB = {"running": False, "lines": [], "now": "", "stop": threading.Event(), "todo": [], "done": [],
       "started": 0.0}


# --------------------------------------------------------------------------- stored results and choices
def _read(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write(p: Path, d: dict):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=1), encoding="utf-8")


def results() -> dict:
    return _read(RESULTS)


def use_on(tkey: str) -> bool:
    return bool(_read(USE).get(tkey))


def set_use(tkey: str, on: bool) -> str:
    r = results().get(tkey)
    u = _read(USE)
    if on:
        if not r or not (r.get("changes") or r.get("env")):
            return "There is no faster setting for this type yet: test it first."
        u[tkey] = {"changes": r["changes"], "env": r.get("env") or {}, "from": r.get("date", "")}
    else:
        u.pop(tkey, None)
    _write(USE, u)
    return ("✔ New launch scripts of this type use the faster settings." if on
            else "✔ New launch scripts of this type use the normal settings again.")


def apply(tkey: str, cfg: dict) -> dict:
    """The faster settings of this type, when chosen, applied to a launch script's settings (flags and environment)."""
    u = _read(USE).get(tkey)
    if not u:
        return cfg
    import speedtest as ST
    toks = shlex.split(" ".join(cfg.get("flags") or []))
    for c in u.get("changes") or []:
        toks = ST.drop(toks, c["opt"]) if c["op"] == "remove" else ST.setopt(toks, c["opt"], c.get("new"))
    flags, n = [], 0
    while n < len(toks):                                  # back to one string per option
        part = [toks[n]]
        if n + 1 < len(toks) and not toks[n + 1].startswith("--"):
            part.append(toks[n + 1])
            n += 1
        flags.append(" ".join(shlex.quote(x) for x in part))
        n += 1
    return {**cfg, "flags": flags, "env": {**(cfg.get("env") or {}), **(u.get("env") or {})},
            "speed_note": f"faster settings from the speed test of {u.get('from', '')}"}


# --------------------------------------------------------------------------- which types can be tested
def types() -> list[dict]:
    """Every LoRA type, with whether it can be tested now (its model files and trainer are there)."""
    import features as FT
    import hfdownload as HD
    import sd_script as SS
    _, presets = SS.merged_presets()
    res = results()
    out = []
    for k, v in presets.items():
        why = ""
        try:
            files = HD.current_files(k)
            miss = [f for f, p in files.items() if HD.is_missing(p) and not (f == "vae" and not p)]
            if miss:
                why = "model files missing: " + ", ".join(miss)
        except Exception as e:                           # noqa: BLE001
            why = f"model files unknown ({e})"
        musubi = v.get("trainer") == "musubi"
        if not why and not FT.have("musubi" if musubi else "sd_scripts"):
            why = ("musubi-tuner" if musubi else "sd-scripts") + " is not installed"
        out.append({"key": k, "label": v["label"], "ready": not why, "why": why, "result": res.get(k),
                    "use": use_on(k)})
    return out


# --------------------------------------------------------------------------- the invented test dataset
def make_dataset() -> Path:
    rep = DATASET / "10_lsspeedtest"
    if rep.is_dir() and len(list(rep.glob("*.png"))) >= 20:
        return DATASET
    from PIL import Image, ImageDraw
    rep.mkdir(parents=True, exist_ok=True)
    sizes = [(1024, 1024), (832, 1216), (1216, 832)]
    for i in range(20):
        w, h = sizes[i % 3]
        im = Image.new("RGB", (w, h), (40 + i * 9, 90 + (i * 37) % 120, 160 - i * 5))
        d = ImageDraw.Draw(im)
        for k in range(6):
            x, y = (i * 97 + k * 151) % w, (i * 53 + k * 211) % h
            d.ellipse([x, y, x + 180 + k * 20, y + 140 + k * 15], fill=((k * 50) % 255, 200 - k * 25, 90 + k * 30))
        im.save(rep / f"t{i:02d}.png")
        (rep / f"t{i:02d}.txt").write_text("lsspeedtest, a robot, standing in a room, simple shapes\n")
        (rep / f"t{i:02d}.sentences.txt").write_text("lsspeedtest, a robot, stands in a bright room.\n")
    return DATASET


def make_script(tkey: str) -> Path:
    import sd_script as SS
    ds = make_dataset()
    plan = SS.make_plan(str(ds), tkey, hf=True, samples=False)
    p = ds / f"lsspeedtest_{tkey}.sh"
    p.write_text(plan["script_text"], encoding="utf-8")
    p.chmod(0o755)
    return p


# --------------------------------------------------------------------------- the background job
def busy() -> bool:
    return JOB["running"]


def _say(line: str, progress: bool = False):
    if progress:
        JOB["now"] = line.strip()
        return
    JOB["lines"] = (JOB["lines"] + [time.strftime("%H:%M ") + line])[-400:]
    JOB["now"] = ""


def start(tkeys: list[str], depth: str = "quick") -> str:
    if JOB["running"]:
        return "A speed test is already running."
    try:
        import sdgui as S
        if S.RUN.running() or S.QUEUE.get("running"):
            return "A training run or the queue is active: start the speed test when it has finished."
    except Exception:                                    # noqa: BLE001
        pass
    ok = {t["key"]: t for t in types()}
    todo = [k for k in tkeys or [] if k in ok and ok[k]["ready"]]
    if not todo:
        return "Tick at least one LoRA type that is ready (its model files are there)."
    JOB.update(running=True, lines=[], now="", todo=list(todo), done=[], started=time.time())
    JOB["stop"].clear()
    threading.Thread(target=_work, args=(todo, depth), daemon=True).start()
    return f"Speed test started: {', '.join(ok[k]['label'] for k in todo)}."


def stop() -> str:
    JOB["stop"].set()
    return "Stopping after the running step …" if JOB["running"] else "No speed test is running."


def _work(todo, depth):
    import speedtest as ST
    try:
        try:
            import ollamactl
            ollamactl.stop_all()                        # the video memory belongs to the test
        except Exception:                                # noqa: BLE001
            pass
        labels = {t["key"]: t["label"] for t in types()}
        for k in todo:
            if JOB["stop"].is_set():
                break
            _say(f"── {labels.get(k, k)} ──")
            try:
                script = make_script(k)
                r = ST.test_script(script, depth, say=_say, stop=JOB["stop"], report_dir=DIR / "reports")
            except SystemExit as e:                     # speedtest's own clear stop messages
                _say(f"❌ {e}")
                continue
            except Exception as e:                       # noqa: BLE001
                _say(f"❌ {labels.get(k, k)}: {e!r}")
                continue
            if r.get("results"):
                res = results()
                res[k] = {**{x: r.get(x) for x in ("baseline", "best", "best_s", "changes", "env", "report", "depth",
                                                   "status", "vram_total")},
                          "date": time.strftime("%Y-%m-%d %H:%M"), "label": labels.get(k, k), "results": r["results"]}
                _write(RESULTS, res)
                u = _read(USE)
                if k in u:                              # chosen before: follow the new best result
                    if r.get("changes") or r.get("env"):
                        u[k] = {"changes": r["changes"], "env": r.get("env") or {}, "from": res[k]["date"]}
                    _write(USE, u)
                gain = (f", fastest {r['best_s']} s/step ({(1 - r['best_s'] / r['baseline']) * 100:.0f}% faster)"
                        if r.get("best_s") and r.get("baseline") and r["best"] != "baseline" else "")
                _say(f"✔ {labels.get(k, k)}: now {r.get('baseline') or '-'} s/step{gain}")
            JOB["done"].append(k)
    finally:
        JOB["running"] = False
        JOB["now"] = ""
        _say("stopped" if JOB["stop"].is_set() else "Speed test finished.")


def table_html() -> str:
    """Settings › Speed test: one row per LoRA type (can it be tested, last result, faster settings in use)."""
    import html
    rows = []
    for t in types():
        r = t["result"] or {}
        if r.get("baseline"):
            gain = (f" → <b>{r['best_s']}</b> ({(1 - r['best_s'] / r['baseline']) * 100:.0f}% faster)"
                    if r.get("best_s") and r.get("best") not in (None, "baseline") else " (already the fastest)")
            res = f"{r['date']} · {r.get('depth', '')} · {r['baseline']} s/step{gain}"
        else:
            res = "not tested"
        state = "✔ in use" if t["use"] else ""
        rows.append(f"<tr><td>{html.escape(t['label'])}</td><td>{'ready' if t['ready'] else html.escape(t['why'])}"
                    f"</td><td>{res}</td><td>{state}</td></tr>")
    return ('<table class="ls-table"><tr><th>LoRA type</th><th>Can be tested</th><th>Last test: now → fastest'
            '</th><th>Faster settings</th></tr>' + "".join(rows) + "</table>")


def report_of(tkey: str) -> tuple[str, bool]:
    """The report text of a tested type and whether it found faster settings."""
    r = results().get(tkey) or {}
    try:
        text = Path(r.get("report", "")).read_text(encoding="utf-8")
    except OSError:
        text = "The report file is gone."
    return text, bool(r.get("changes") or r.get("env"))


def live_text() -> str:
    lines = JOB["lines"][-14:] + ([f"⏳ {JOB['now']}"] if JOB["now"] else [])
    return "\n".join(lines)
