# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Home: the start page. What is training, what needs a look, the last results and where you left off.

  Just trained     the last training (one LoRA or a queue) and its test pictures (comfytest.last_batch)
  Training now     the run and the queue → the queue
  Needs a look     Settings' health check, Datasets › Improvements and the Collection's warnings, counted, with the
                   first few lines and a way to each
  Last results     the last five finished runs → that run
  Continue         the datasets you opened last, at the step you were at → that dataset
  Start            + New dataset

Nothing is scanned here: every part reads what the other pages already know (their caches). This file holds the
data of the cards; UI/pg_home.py draws them.
"""
from __future__ import annotations
__version__ = "1.131126b"  # LoRA Studio version, see version.py

import html
import json
import time


import lspaths as _LP

RECENT = _LP.USER / "recent_datasets.json"
esc = html.escape


# --------------------------------------------------------------------------- where you left off
def remember(ds: str, step: int):
    """Called by Datasets when a dataset is open at a step: the newest five are kept."""
    if not ds:
        return
    try:
        old = json.loads(RECENT.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        old = []
    if old and old[0].get("ds") == ds and old[0].get("step") == step:
        return                                            # nothing new: no write on every step click
    new = [{"ds": ds, "step": int(step or 0), "t": time.time()}] + [r for r in old if r.get("ds") != ds]
    try:
        RECENT.parent.mkdir(parents=True, exist_ok=True)
        RECENT.write_text(json.dumps(new[:5]), encoding="utf-8")
    except OSError:
        pass


def recent() -> list[dict]:
    try:
        return [r for r in json.loads(RECENT.read_text(encoding="utf-8")) if r.get("ds")]
    except (OSError, ValueError):
        return []


# --------------------------------------------------------------------------- the data of the Home cards
def look_items() -> list[dict]:
    """Needs a look: [{"level": bad|warn, "title", "detail", "go": (page, kw)}] from the health check, the
    datasets' improvements and the Collection's warnings (each read from what those pages already know)."""
    out = []
    try:
        import settings_ui
        bad = [c for c in settings_ui.checks() if c["state"] in ("bad", "warn")]
    except Exception:                                    # noqa: BLE001
        bad = []
    for c in bad:
        out.append({"level": c["state"], "title": c["title"], "detail": c["detail"],
                    "go": ("settings", {"group": c["fix"] or "health"})})
    try:
        import dsflow as F
        import sdgui as S
        imp = [(ds, F.improvements(ds)) for ds in S.list_lora_dirs()]
    except Exception:                                    # noqa: BLE001
        imp = []
    for ds, it in imp:
        if it:
            out.append({"level": "warn", "title": ds, "detail": it[0][0], "go": ("datasets", {"ds": ds})})
    try:
        import lora_library as LL
        st = [LL.group_status(m)[0] for m in LL.LIB.groups.values() if m and m[-1].dataset]
        n_prob, n_warn = st.count("problem"), st.count("warning")
    except Exception:                                    # noqa: BLE001
        n_prob = n_warn = 0
    if n_prob or n_warn:
        out.append({"level": "bad" if n_prob else "warn", "title": "Collection",
                    "detail": f"{n_prob} problems, {n_warn} warnings in your LoRA files",
                    "go": ("collection", {"look": "Status"})})
    return out


def last_runs(n: int = 5) -> list[dict]:
    try:
        import training_ui as T
        return [r for r in T.runs(30) if r["result"] != "running"][:n]
    except Exception:                                    # noqa: BLE001
        return []


def continue_items() -> list[dict]:
    """[{"ds", "step", "label"}] of the datasets opened last that still exist."""
    try:
        import dsflow as F
        import sdgui as S
        labels = {n: lab for n, _, lab in F.STEPS}
        labels.update({10: "Results", 11: "LoRA"})
        have = set(S.list_lora_dirs())
    except Exception:                                    # noqa: BLE001
        labels, have = {}, set()
    out = []
    for r in recent():
        if r["ds"] in have:
            step = r.get("step") or 0
            out.append({"ds": r["ds"], "step": step,
                        "label": ((str(step) + " ") if step <= 9 else "") + labels.get(step, "")})
    return out


# --------------------------------------------------------------------------- the parts (Gradio page, until it is removed)

