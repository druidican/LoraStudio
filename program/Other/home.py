# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Home: the start page. What is training, what needs a look, the last results and where you left off.

  Training now     the run and the queue (refreshed by the page itself, livepoll.py) → the queue
  Needs a look     Settings' health check, Datasets › Improvements and the Collection's warnings, counted, with the
                   first few lines and a way to each
  Last results     the last five finished runs → that run
  Continue         the datasets you opened last, at the step you were at → that dataset
  Start            + New dataset

Nothing is scanned here: every part reads what the other pages already know (their caches). The clicks are sent to
the hidden #home_box (lora_studio.py opens the dataset or the page); jumps that need no Python are plain JavaScript.
"""
from __future__ import annotations
__version__ = "1.131126b"  # LoRA Studio version, see version.py

import html
import json
import time

import gradio as gr

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
def _send(v: str) -> str:
    return f"lsSend('home_box', {json.dumps(v)})".replace('"', "&quot;")


def _card(title: str, body: str, foot: str = "") -> str:
    return (f'<div class="ls-card hm-card"><div class="hm-h">{esc(title)}</div>{body}'
            + (f'<div class="hm-foot">{foot}</div>' if foot else "") + "</div>")


def training_html() -> str:
    """Training now (cheap: read every 2.5 s by the page)."""
    import sdgui as S
    waiting = sum(1 for i in S.QUEUE["items"] if i["state"] == "queued")
    if S.RUN.running():
        prog = S.progress_text()
        body = (f'<p class="hm-big">● {esc(S.RUN.name or "training")}</p><p class="ls-note">{esc(prog)}</p>'
                f'<p class="ls-note">{waiting} more waiting in the queue</p>')
    else:
        body = ('<p class="hm-big">Nothing is training</p><p class="ls-note">'
                + (f"{waiting} waiting in the queue." if waiting else "The queue is empty.") + "</p>")
    btn = '<button class="tr-btn" onclick="window.lsGoQueue && lsGoQueue()">Open the queue</button>'
    return _card("Training now", body, btn)


def look_html() -> str:
    """Needs a look: setup, datasets, LoRA files."""
    rows = []
    try:
        import settings_ui
        bad = [c for c in settings_ui.checks() if c["state"] in ("bad", "warn")]
    except Exception:                                    # noqa: BLE001
        bad = []
    for c in bad[:3]:
        rows.append(f'<a href="#" class="hm-row" onclick="lsSetNav(\'{esc(c["fix"] or "health")}\'); return false;">'
                    f'<span class="hm-dot hm-{"bad" if c["state"] == "bad" else "warn"}"></span>'
                    f'<b>{esc(c["title"])}</b> · {esc(c["detail"])}</a>')
    if len(bad) > 3:
        rows.append(f'<a href="#" class="hm-row" onclick="lsSetNav(\'health\'); return false;">'
                    f'{len(bad) - 3} more in Settings › Health check</a>')
    try:
        import dsflow as F
        import sdgui as S
        imp = [(ds, F.improvements(ds)) for ds in S.list_lora_dirs()]
        imp = [(ds, it) for ds, it in imp if it]
    except Exception:                                    # noqa: BLE001
        imp = []
    for ds, it in imp[:3]:
        rows.append(f'<a href="#" class="hm-row" onclick="{_send("ds|" + ds)}; return false;">'
                    f'<span class="hm-dot hm-warn"></span><b>{esc(ds)}</b> · {esc(it[0][0])}</a>')
    if len(imp) > 3:
        rows.append(f'<a href="#" class="hm-row" onclick="{_send("imp")}; return false;">'
                    f'{len(imp) - 3} more datasets in Datasets › Improvements</a>')
    try:
        import lora_library as LL
        st = [LL.group_status(m)[0] for m in LL.LIB.groups.values() if m and m[-1].dataset]
        n_prob, n_warn = st.count("problem"), st.count("warning")
    except Exception:                                    # noqa: BLE001
        n_prob = n_warn = 0
    if n_prob or n_warn:
        rows.append(f'<a href="#" class="hm-row" onclick="lsClickTab(\'Collection\'); return false;">'
                    f'<span class="hm-dot hm-{"bad" if n_prob else "warn"}"></span>'
                    f'<b>Collection</b> · {n_prob} problems, {n_warn} warnings in your LoRA files</a>')
    body = "".join(rows) or '<p class="ls-note">Nothing needs a look.</p>'
    return _card("Needs a look", body)


def results_html() -> str:
    try:
        import training_ui as T
        rows = [r for r in T.runs(30) if r["result"] != "running"][:5]
    except Exception:                                    # noqa: BLE001
        rows = []
    out = []
    for r in rows:
        ok = r["result"] == "done"
        label = (f'{r["info"]["dataset"]} · {r["info"]["type"]}' if r.get("info") else r["stem"])
        out.append(f'<a href="#" class="hm-row" onclick="lsGoRun(\'{esc(r["log"])}\'); return false;">'
                   f'<span class="hm-dot hm-{"ok" if ok else "bad"}"></span><b>{esc(label)}</b>'
                   f' · {esc(r["result"])} · {esc(r["started"][5:])}</a>')
    return _card("Last results", "".join(out) or '<p class="ls-note">No finished runs yet.</p>')


def continue_html() -> str:
    try:
        import dsflow as F
        labels = {n: lab for n, _, lab in F.STEPS}
        labels.update({10: "Results", 11: "LoRA"})
        import sdgui as S
        have = set(S.list_lora_dirs())
    except Exception:                                    # noqa: BLE001
        labels, have = {}, set()
    out = []
    for r in recent():
        if r["ds"] not in have:
            continue
        step = r.get("step") or 0
        out.append(f'<a href="#" class="hm-row" onclick="{_send("dsstep|" + r["ds"] + "|" + str(step))}; return false;">'
                   f'<b>{esc(r["ds"])}</b> · {(str(step) + " ") if step <= 9 else ""}{esc(labels.get(step, ""))}</a>')
    body = "".join(out) or '<p class="ls-note">The datasets you open appear here.</p>'
    foot = f'<button class="tr-btn" onclick="{_send("new")}">+ New dataset</button>'
    return _card("Continue where you left off", body, foot)


def page_html() -> str:
    return (f'<div class="hm-grid"><div id="hm_live">{training_html()}</div>{continue_html()}'
            f'{look_html()}{results_html()}</div>')


def build():
    """The Home tab's parts. Returns the box (clicks that need Python) and (fn, inputs, outputs) to fill it."""
    gr.Markdown("### Home", elem_classes=["ls-h"])
    with gr.Row(elem_classes=["ls-bar"]):
        ref = gr.Button("Refresh", size="sm", scale=0, min_width=110)
    page = gr.HTML(elem_id="hm_page")
    box = gr.Textbox(elem_id="home_box", show_label=False, container=False, elem_classes=["ls-hidden"])
    ref.click(page_html, None, page, show_progress="hidden")
    return {"box": box, "load": (page_html, None, page)}


CSS = """
.hm-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap:16px; }
.hm-card { display:flex; flex-direction:column; gap:6px; min-height:150px; }
.hm-h { font-weight:600; font-size:15px; margin-bottom:4px; }
.hm-big { font-size:17px; font-weight:600; margin:2px 0; }
#ls_manual_tab-button { display:none !important; }      /* the Manual opens with the ? button */
#hm_page, .hm-card { text-align:left; }
.hm-row { display:block; padding:6px 8px; border-radius:6px; text-decoration:none !important; color:inherit !important;
  font-size:14px; }
.hm-row:hover { background: var(--ls-hover, rgba(127,127,127,.12)); }
.hm-dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:8px; vertical-align:middle; }
.hm-ok { background:#21857C; } .hm-warn { background:#D9902B; } .hm-bad { background:#C0392B; }
.hm-foot { margin-top:auto; padding-top:6px; }
"""
