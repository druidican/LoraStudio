# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Settings: a grouped list on the left, one panel at a time on the right, opening on a Health check.

Program     Start and network
Your files  Folders, Model files
Tools       Trainers and GPU, AI helpers
Library     Look and base models
Care        Backup, Updates

The Health check lists what is set and working, what needs a look and what is missing, each with a link that
opens the panel where it is fixed.
"""
from __future__ import annotations
__version__ = "1.111126b"  # LoRA Studio version, see version.py

import html
import os
import shutil
import sys
import threading
import time
from pathlib import Path

import gradio as gr

NAV = [
    ("", [("health", "Health check")]),
    ("PROGRAM", [("start", "Start and network")]),
    ("YOUR FILES", [("folders", "Folders"), ("models", "Model files")]),
    ("TOOLS", [("trainers", "Trainers and GPU"), ("speed", "Speed test"), ("ai", "AI helpers"),
               ("batch", "Automatic improvement"), ("comfy", "Test pictures (ComfyUI)")]),
    ("LIBRARY", [("look", "Look and base models")]),
    ("CARE", [("backup", "Backup"), ("updates", "Updates"), ("version", "Version")]),
]
KEYS = [k for _, items in NAV for k, _ in items]
NAV_PART = {"ai": "ollama", "trainers": "sd_scripts"}       # greyed with "(… missing)" when that part is not installed


def nav_label(key: str) -> str:
    import features as FT
    label = dict(i for _, it in NAV for i in it)[key]
    return FT.label(label, NAV_PART[key]) if key in NAV_PART else label


def nav_classes(key: str, on: bool) -> list:
    import features as FT
    return (["set-item"] + (["set-on"] if on else [])
            + (["set-missing"] if key in NAV_PART and not FT.have(NAV_PART[key]) else []))


def _sg():
    return sys.modules.get("sdgui")


def _ll():
    return sys.modules.get("lora_library")


def _tilde(p) -> str:
    p, home = str(p), str(Path.home())
    return "~" + p[len(home):] if p.startswith(home) else p


# --------------------------------------------------------------------------- health check
class _Skip(Exception):
    pass


def checks() -> list[dict]:
    """[{state: ok|warn|bad, title, detail, fix, fix_label}]"""
    out = []
    sg, ll = _sg(), _ll()

    def add(state, title, detail="", fix="", fix_label="Fix"):
        out.append({"state": state, "title": title, "detail": detail, "fix": fix, "fix_label": fix_label})

    s = ll.LIB.settings if ll else {}
    # folders
    ds_dirs = [Path(os.path.expanduser(d)) for d in s.get("dataset_dirs") or []]
    if not ds_dirs:
        add("bad", "Dataset folder", "not set", "folders")
    elif not ds_dirs[0].is_dir():
        add("bad", "Dataset folder", f"{_tilde(ds_dirs[0])} does not exist", "folders")
    else:
        n = len(sg.list_lora_dirs()) if sg else 0
        add("ok", "Dataset folder", f"{_tilde(ds_dirs[0])} · {n} datasets", "folders")
    lora_dirs = [Path(os.path.expanduser(d)) for d in s.get("lora_dirs") or []]
    missing = [d for d in lora_dirs if not d.is_dir()]
    if not lora_dirs:
        add("warn", "LoRA folder", "not set: finished LoRAs cannot be copied to ComfyUI", "folders")
    elif missing:
        add("warn", "LoRA folder", f"{_tilde(missing[0])} does not exist", "folders")
    else:
        add("ok", "LoRA folder", ", ".join(_tilde(d) for d in lora_dirs[:2]), "folders")
    pics = os.path.expanduser(s.get("pictures_dir") or "")
    if pics and not Path(pics).is_dir():
        add("warn", "ComfyUI picture folder", f"{_tilde(pics)} does not exist", "folders")
    elif pics:
        add("ok", "ComfyUI picture folder", _tilde(pics), "folders")
    # trainers
    import features as FT
    for part in ("sd_scripts", "musubi", "ollama"):
        if not FT.have(part):
            add("off", FT.PARTS[part], f"not installed (your choice): {FT.WHAT[part]} is hidden", "start", "Change")
    if sg and FT.have("sd_scripts"):
        sd = sg.sd_root()
        sd_ok = sd.is_dir() and any(sd.glob("*train_network.py"))
        add("ok" if sd_ok else "bad", "sd-scripts", _tilde(sd) if sd_ok else f"not found in {_tilde(sd)}", "trainers")
    if sg:
        mu = sg.musubi_root()
        if (mu / "src" / "musubi_tuner").is_dir() and FT.have("musubi"):
            add("ok", "musubi-tuner", _tilde(mu), "trainers")
        used = {sg.script_type(p) for _, p in sg.list_scripts()}
        if FT.have("musubi") and any(t in used for t in ("zimage", "zimagebase", "qwenimage", "flux2")) and \
                not (mu / "src" / "musubi_tuner").is_dir():
            add("bad", "musubi-tuner", "not found, but you have launch scripts that need it", "trainers")
        if FT.have("sd_scripts") and sg.sd_root().is_dir():
            try:
                import pictools
                pst = pictools.status()
                add("ok" if pst["ok"] else "warn", "Picture tools",
                    "ready (quality, look-alikes, watermark removal, upscaling)" if pst["ok"] else
                    pst["text"] + " · Install them here", "trainers")
            except Exception:                            # noqa: BLE001
                pass
    # the training venv: PyTorch and its ROCm / CUDA parts
    try:
        import venvcheck as VC
        r = VC.LAST
        if r.get("torch"):
            add("ok" if r["ok"] else "bad", "Training venv", r["text"], "updates",
                "Fix" if r.get("managed") else "Show the commands")
    except Exception:                                    # noqa: BLE001
        pass
    # the program's own files: all of the same version?
    try:
        import version
        bad = version.problems()
        if bad:
            add("warn", "Program files", f"{len(bad)} not the ones of this release ({bad[0]['name']}"
                + (", …" if len(bad) > 1 else "") + ")", "version", "See")
        else:
            add("ok", "Program files", f"all {len(version.PARTS)} parts of LoRA Studio {version.VERSION}", "version")
    except Exception:                                    # noqa: BLE001
        pass
    # GPU
    try:
        import statusbar
        g = statusbar.machine()["gpus"]
        if g:
            vram = f" · {g[0]['vram_total'] / 1024 ** 3:.0f} GB" if g[0].get("vram_total") else ""
            add("ok", "GPU", f"{g[0]['name']}{vram}", "trainers")
        else:
            add("warn", "GPU", "not detected (no amd-smi, rocm-smi or nvidia-smi)", "trainers")
    except Exception:                                    # noqa: BLE001
        pass
    # Ollama
    try:
        if not FT.have("ollama"):
            raise _Skip
        import ollamactl
        url, model = ollamactl.settings()
        now = ollamactl.running(url, fresh=True)
        if now is None:
            add("warn", "Ollama", f"not reachable at {url}: sorting, checks and captions need it", "ai")
        else:
            names = {m["name"]: m for m in ollamactl.installed(url)}
            if not model:
                add("warn", "Ollama", "no model chosen for pictures", "ai", "Choose one")
            elif model not in names:
                add("warn", "Ollama", f"{model} is not installed", "ai")
            elif names[model].get("vision") is False:
                add("warn", "Ollama", f"{model} cannot read pictures", "ai")
            else:
                state = "running" if any(m["name"] == model for m in now) else "installed, not running now"
                add("ok", "Ollama", f"{model} · {state}", "ai")
    except _Skip:
        pass
    except Exception as e:                               # noqa: BLE001
        add("warn", "Ollama", f"could not check: {e}", "ai")
    # model files: the LoRA types you use (launch scripts) or started to set up
    if sg:
        try:
            labels = dict((v, l) for l, v in sg.type_choices())
            used = {sg.script_type(p) for _, p in sg.list_scripts()}
            used |= {k for k in sg.read_settings() if k in labels and sg.settings_models(k)}
            full = []
            for t in sorted(u for u in used if u in labels):
                saved = sg.settings_models(t)
                _, presets = sg.merged_presets()
                gaps = []
                for flag, default in sg.type_slots(t):
                    if flag in ("vae",) and sg.vae_optional(t, presets):
                        continue
                    have = saved.get(flag) or ("" if (not default or sg._PH in default) else default)
                    if not have:
                        gaps.append(sg.slot_label(flag))
                if gaps:
                    add("warn", "Model files", f"{labels[t]}: {', '.join(gaps)} not chosen", "models")
                else:
                    full.append(labels[t])
            if full:
                add("ok", "Model files", ", ".join(full) + ": all slots filled", "models")
        except Exception as e:                           # noqa: BLE001
            add("warn", "Model files", f"could not check: {e}", "models")
        add("ok" if sg.read_hf_token() else "warn", "Hugging Face token",
            "saved" if sg.read_hf_token() else "not saved: launch scripts that download models will fail", "ai")
    # backup
    try:
        import backup
        found = backup.find_backups()
        if not found:
            add("warn", "Backup", "no backup yet", "backup", "Back up now")
        else:
            age = (time.time() - Path(found[0][1]).stat().st_mtime) / 86400
            add("warn" if age > 14 else "ok", "Backup",
                f"last backup {age:.0f} days ago" if age >= 1 else "last backup today", "backup",
                "Back up now")
    except Exception:                                    # noqa: BLE001
        pass
    # disk space where the datasets are
    if ds_dirs and ds_dirs[0].is_dir():
        free = shutil.disk_usage(ds_dirs[0]).free / 1024 ** 3
        add("warn" if free < 20 else "ok", "Disk space", f"{free:.0f} GB free where the datasets are", "folders")
    return out


def health_html() -> str:
    items = checks()
    ok = [i for i in items if i["state"] == "ok"]
    warn = [i for i in items if i["state"] == "warn"]
    bad = [i for i in items if i["state"] == "bad"]

    def row(i):
        mark = {"ok": "✓", "warn": "!", "bad": "✕", "off": "–"}[i["state"]]
        fix = (f'<a href="#" class="hc-fix" onclick="lsSetNav(\'{i["fix"]}\'); return false;">{html.escape(i["fix_label"])} →</a>'
               if i["fix"] and i["state"] not in ("ok", "off") else
               (f'<a href="#" class="hc-open" onclick="lsSetNav(\'{i["fix"]}\'); return false;">Open</a>' if i["fix"] else ""))
        return (f'<div class="hc-row"><span class="hc-dot hc-{i["state"]}">{mark}</span>'
                f'<span class="hc-t"><b>{html.escape(i["title"])}</b> · {html.escape(i["detail"])}</span>{fix}</div>')

    tiles = (f'<div class="hc-tiles"><div class="hc-tile"><b class="hc-n-ok">{len(ok)}</b><span>set and working</span></div>'
             f'<div class="hc-tile"><b class="hc-n-warn">{len(warn)}</b><span>need a look</span></div>'
             f'<div class="hc-tile"><b>{len(bad)}</b><span>missing</span></div></div>')
    body = ""
    if bad or warn:
        body += '<div class="hc-card"><div class="hc-h">Needs a look</div>' + "".join(row(i) for i in bad + warn) + "</div>"
    if ok:
        body += '<div class="hc-card"><div class="hc-h">Set and working</div>' + "".join(row(i) for i in ok) + "</div>"
    off = [i for i in items if i["state"] == "off"]
    if off:
        body += ('<div class="hc-card hc-offcard"><div class="hc-h">Not installed</div>' + "".join(row(i) for i in off)
                 + '<div class="ls-note">Change it in Settings › Start and network › What is installed.</div></div>')
    when = time.strftime("%H:%M")
    return f'<div class="hc"><div class="hc-when">checked {when}</div>{tiles}{body}</div>'


def nav_counts() -> dict:
    """{nav key: number of things that need a look there} for the badges in the left list."""
    c = {}
    for i in checks():
        if i["state"] not in ("ok", "off") and i["fix"]:
            c[i["fix"]] = c.get(i["fix"], 0) + 1
    c["health"] = sum(c.values())
    return c


NAV_JS = """
window.lsSetNav = function(key){
  var t = Array.from(document.querySelectorAll('#ls_tabs > div button[role="tab"], #ls_tabs button[role="tab"]'))
              .find(function(x){ return x.textContent.trim() === 'Settings'; });
  var away = t && t.getAttribute('aria-selected') !== 'true';
  if (away) t.click();
  setTimeout(function(){                       // after the tab has opened (it shows the Health check first)
    var b = document.getElementById('setnav_' + key);
    if (b) { b.click(); window.scrollTo({top: 0, behavior: 'smooth'}); }
  }, away ? 600 : 0);
};
"""


# --------------------------------------------------------------------------- Automatic improvement
KEEP_CHOICES = [("The suggestion for the LoRA type, at most the number below", "suggested"),
                ("Always the number below (fewer when there are not that many)", "fixed")]
WM_CHOICES = [("Remove them (the picture tools paint over them; crop if that fails)", "remove"),
              ("Crop them off (the recommended crop)", "crop"), ("Set those pictures aside", "aside"),
              ("Leave them (decide later in Datasets)", "leave")]
CAP_CHOICES = [("Only pictures without a caption", "missing"), ("Write every caption again", "all")]
CONS_CHOICES = [("Leave it out", "ignore"), ("Run it", "check")]


def batch_panel(app):
    import dsflow as F
    gr.Markdown("What **Datasets › Improvements › Fix the ticked datasets** does to each dataset. Saved in "
                "`batch_settings.json` (in the backup too). Everything moved can be put back (`_reserve/`, `_originals/`).",
                elem_classes=["ls-note"])
    with gr.Column(elem_classes=["ls-card"]):
        gr.Markdown("**Pictures to keep** (step 5 Select; the rest go to `_reserve/`)")
        keep_mode = gr.Radio(KEEP_CHOICES, label="How many", value="suggested")
        max_pics = gr.Number(label="Number of pictures to keep in the dataset folder", value=F.BATCH_MAX, precision=0,
                             minimum=5, maximum=1000)
        sel = gr.Radio([("Safe: the cleanest pictures first", "safe"),
                        ("Varied: more close-ups, angles, outfits and details", "varied")],
                       label="How to pick them", value="safe")
        safety = gr.Slider(5, 100, value=50, step=5, label="Safety: move nothing in step 2 when more than this % "
                                                            "of the pictures look not usable")
    with gr.Column(elem_classes=["ls-card"]):
        gr.Markdown("**LoRA type and purpose** (step 3; only for a dataset that has none yet)")
        with gr.Row():
            dtype = gr.Dropdown(choices=[], label="LoRA type", scale=1)
            dpurp = gr.Dropdown(choices=F.PURPOSES, value="character", label="Purpose", scale=1)
    with gr.Column(elem_classes=["ls-card"]):
        gr.Markdown("**Watermarks and size** (step 6)")
        wm = gr.Radio(WM_CHOICES, label="Pictures with a watermark", value="remove")
        up = gr.Checkbox(label="Upscale pictures that are a little too small (picture tools, Real-ESRGAN)", value=True)
    with gr.Column(elem_classes=["ls-card"]):
        gr.Markdown("**Captions** (step 7) and the end")
        import features as FT
        cap = gr.Radio(CAP_CHOICES, label=FT.label("Captions", "ollama"), value="missing", interactive=FT.have("ollama"),
                       elem_classes=[] if FT.have("ollama") else ["ls-missing"])
        cons = gr.Radio(CONS_CHOICES, label="Consistency check", value="ignore")
        each = gr.Checkbox(label="Answer every trait with “This picture only” (each caption keeps its own; untick to "
                                 "walk through them yourself later in step 7)", value=True, visible=False)
        cons.change(lambda c: gr.update(visible=c == "check"), cons, each)
        ls = gr.Checkbox(label="Write a launch script with the recommended settings at the end (step 9)", value=False)
    with gr.Row():
        save = gr.Button("Save", variant="primary", size="sm", scale=0, min_width=120)
        reset = gr.Button("Back to the defaults", size="sm", scale=0, min_width=200)
    msg = gr.Markdown()
    summ = gr.Markdown(elem_classes=["ls-note"])
    comps = [keep_mode, max_pics, sel, safety, dtype, dpurp, wm, up, cap, cons, each, ls]
    keys = ["keep_mode", "max_pics", "select_mode", "safety_pct", "default_type", "default_purpose", "watermarks", "upscale",
            "captions", "consistency", "cons_each", "launch_script"]

    def load(b=None):
        b = b or F.batch_settings()
        S = _sg()
        types = [(l, v) for l, v in (S.type_choices() if S else [])]
        out = []
        for k, c in zip(keys, comps):
            if k == "default_type":
                out.append(gr.update(choices=types, value=b[k]))
            elif k == "cons_each":
                out.append(gr.update(value=b[k], visible=b["consistency"] == "check"))
            else:
                out.append(b[k])
        return out + ["**Now:** " + F.batch_summary(b)]

    def do_save(*vals):
        b = F.batch_settings_save(dict(zip(keys, vals)))
        return load(b)[:-1] + ["✔ Saved. The next *Fix the ticked datasets* uses these.", "**Now:** " + F.batch_summary(b)]

    def do_reset():
        b = F.batch_settings_save(dict(F.BATCH_DEFAULTS))
        return load(b)[:-1] + ["✔ Back to the defaults.", "**Now:** " + F.batch_summary(b)]
    app.load(load, None, comps + [summ])
    save.click(do_save, comps, comps + [msg, summ])
    reset.click(do_reset, None, comps + [msg, summ])


# --------------------------------------------------------------------------- What is installed
def features_panel(app):
    import features as FT
    with gr.Column(elem_classes=["ls-card"]):
        gr.Markdown("### What is installed")
        gr.Markdown("A part that is not installed has its buttons hidden, so nothing can be started by mistake; its "
                    "steps and pages stay, greyed out with *(… missing)*. Launch scripts are still made without a "
                    "trainer: run them where the trainer is. `install.sh` sets this; change it here when you add or "
                    "remove a part, then restart LoRA Studio.", elem_classes=["ls-note"])
        boxes = {}
        for key, name in FT.PARTS.items():
            boxes[key] = gr.Checkbox(label=f"{name} is installed", value=FT.have(key),
                                     info=f"Needed for {FT.WHAT[key]}.")
        gr.Markdown(f"Ollama: get it from [{FT.ollama_guide()}]({FT.ollama_guide()})"
                    + (" (on Windows: install it in Windows, LoRA Studio in WSL reaches it at the address in Settings › "
                       "AI helpers)." if FT.is_wsl() else "; or on Linux: `curl -fsSL https://ollama.com/install.sh | sh`."),
                    elem_classes=["ls-note"])
        with gr.Row(elem_classes=["ls-bar"]):
            save = gr.Button("Save", variant="primary", size="sm", scale=0, min_width=110)
            rest = gr.Button("Restart LoRA Studio", size="sm", scale=0, min_width=180)
        msg = gr.Markdown()

    def do_save(*vals):
        cur = FT.load()
        new = {}
        for (key, _), on in zip(FT.PARTS.items(), vals):
            old = cur.get(key, "")
            new[key] = (old if old and old != "none" else "installed") if on else "none"
        FT.save(**new)
        return "✔ Saved. Restart LoRA Studio to apply it (the page is built once at start)."
    save.click(do_save, list(boxes.values()), msg)

    def do_restart():
        import selfupdate as SU
        return SU.restart()
    rest.click(do_restart, None, msg, js="""() => { setTimeout(function wait(){
        fetch(location.href, {cache: 'no-store'}).then(function(r){ if (r.ok) location.reload(); else setTimeout(wait, 2000); })
        .catch(function(){ setTimeout(wait, 2000); }); }, 6000); }""")


# --------------------------------------------------------------------------- Updates › LoRA Studio itself
def selfupdate_plan_html(path) -> str:
    import selfupdate as SU
    import version  # noqa: F811
    if not path:
        return ""
    try:
        pl = SU.plan(path)
    except Exception as e:                               # noqa: BLE001
        return f'<p class="ls-empty">❌ {html.escape(str(e))}</p>'
    z = pl["zip"]
    ch = [r for r in pl["rows"] if r["state"] in ("changed", "new")]
    head = (f'<div class="hc-card"><div class="hc-h">LoraStudio.zip: version {html.escape(z["version"])}</div>'
            f'<div>You have {html.escape(version.VERSION)}. '
            + ("This zip is <b>newer</b>." if pl["newer"] else "This is the version you have." if pl["same"] else
               "<b>This zip is older than what you have</b> (installing it goes back).")
            + f' {len(ch)} part(s) would change.</div>'
            + "".join(f'<div class="ls-ver-h"><b>{html.escape(v)}</b> {html.escape(txt)}</div>'
                      for v, txt in z["changes"] if SU._vkey(v) > SU._vkey(version.VERSION)) + "</div>")

    def row(r):
        return (f'<tr><td><code>{html.escape(r["name"])}</code></td><td>{html.escape(r["have"] or "—")}</td>'
                f'<td>{html.escape(r["new"])}</td><td>'
                + {"changed": '<span class="ls-chip warn">changes</span>', "new": '<span class="ls-chip ok">new</span>',
                   "same": '<span class="tr-muted">stays</span>',
                   "missing in zip": '<span class="ls-chip bad">not in the zip</span>'}[r["state"]]
                + f'</td><td>{html.escape(r["what"])}</td></tr>')
    th = ('<thead><tr><th>Part</th><th>You have</th><th>In the zip</th><th></th><th>What it does</th></tr></thead>')
    moving = [r for r in sorted(pl["rows"], key=lambda r: r["name"]) if r["state"] != "same"]
    stay = [r for r in sorted(pl["rows"], key=lambda r: r["name"]) if r["state"] == "same"]
    out = head
    if moving:
        out += (f'<div class="rs-wrap"><table class="rs-table ls-ver-t">{th}<tbody>{"".join(row(r) for r in moving)}'
                '</tbody></table></div>')
    if stay:
        out += (f'<details class="ds-fold"><summary>{len(stay)} parts stay as they are</summary><div class="rs-wrap">'
                f'<table class="rs-table ls-ver-t">{th}<tbody>{"".join(row(r) for r in stay)}</tbody></table></div>'
                '</details>')
    return out


def venv_html() -> str:
    import venvcheck as VC
    r = VC.LAST or VC.inspect()
    if not r.get("torch"):
        return f'<div class="ls-note">{html.escape(r.get("text", ""))}</div>'
    who = "installed by LoRA Studio" if r.get("managed") else "your own venv: LoRA Studio shows the commands, it does not change it"
    head = (f'<div class="hc-row"><span class="hc-dot hc-{"ok" if r["ok"] else "bad"}">{"✓" if r["ok"] else "✕"}</span>'
            f'<span class="hc-t"><b>Training venv</b> · {html.escape(r["text"])}</span></div>'
            f'<div class="ls-note"><code>{html.escape(_tilde(r["py"]))}</code> · {html.escape(who)}</div>')
    if not r["ok"] and not r.get("managed"):
        head += "<pre>" + html.escape("\n".join(__import__("venvcheck").commands(r))) + "</pre>"
    return head


def venv_panel(app):
    import venvcheck as VC
    with gr.Column(elem_classes=["ls-card"]):
        gr.Markdown("### The training venv")
        gr.Markdown("PyTorch decides: every ROCm or CUDA part must be the version PyTorch is built for (a newer part "
                    "is wrong too: it crashes trainings with *Memory access fault by GPU*). Checked at every start and "
                    "after an update, without downloading anything. **Fix** puts the parts that do not fit back, "
                    "writes the locks in `constraints.txt` again and, when PyTorch itself was upgraded, installs the "
                    "requirements again within the new locks.", elem_classes=["ls-note"])
        st = gr.HTML(venv_html)
        with gr.Row(elem_classes=["ls-bar"]):
            again = gr.Button("Check again", size="sm", scale=0, min_width=150)
            fixb = gr.Button("Fix the training venv", variant="primary", size="sm", scale=0, min_width=210)
        vlog = gr.Textbox(label="Training venv", lines=8, max_lines=24, interactive=False, autoscroll=True,
                          visible=False)

    def recheck():
        VC.inspect()
        return venv_html()
    again.click(recheck, None, st)

    def run_fix():
        for text in VC.fix():
            yield gr.update(value=text, visible=True), gr.update()
        yield gr.update(), venv_html()
    fixb.click(run_fix, None, [vlog, st])

    with gr.Column(elem_classes=["ls-card"]):
        gr.Markdown("#### A newer PyTorch / ROCm")
        gr.Markdown("Asks AMD's server (ROCm) or PyPI (CUDA) which PyTorch builds exist; only stable PyTorch releases "
                    "count (torchvision / torchaudio as AMD pairs them). Only the list of versions is downloaded until "
                    "you upgrade. The upgrade keeps the old versions in `update_backups/`, tests the GPU and puts them "
                    "back when the test fails; then the locks and the requirements are renewed.",
                    elem_classes=["ls-note"])
        nmd = gr.Markdown()
        with gr.Row(elem_classes=["ls-bar"]):
            lookb = gr.Button("Look for a newer PyTorch", size="sm", scale=0, min_width=220)
            sure = gr.Checkbox(label="Yes, upgrade (several GB; no training until it has finished)", value=False,
                               visible=False, scale=1)
            upb = gr.Button("Upgrade", variant="primary", size="sm", scale=0, min_width=200, visible=False)

    def do_look():
        n = VC.look()
        show = n["newer"]
        md = ("**" + html.escape(n["text"]) + "**" if show else html.escape(n["text"]))
        if show and not n["managed"]:
            md += "\n\nYour own venv: LoRA Studio shows the command, it does not change it:\n\n```\n" + \
                  VC._upgrade_cmd(VC.LAST, n["newest"]) + "\n```"
        return (md, gr.update(visible=show and n["managed"], value=False),
                gr.update(visible=show and n["managed"], value=f"Upgrade to torch {n['newest']}"))
    lookb.click(lambda: "Asking the server …", None, nmd).then(do_look, None, [nmd, sure, upb])

    def do_up(ok):
        if not ok:
            yield gr.update(value="Tick 'Yes, upgrade' first.", visible=True), gr.update()
            return
        tgt = VC.NEWER.get("newest")
        if not tgt:
            yield gr.update(value="Look for a newer PyTorch first.", visible=True), gr.update()
            return
        for text in VC.upgrade(tgt):
            yield gr.update(value=text, visible=True), gr.update()
        yield gr.update(), venv_html()
    upb.click(do_up, sure, [vlog, st])


def selfupdate_panel(app):
    import selfupdate as SU
    with gr.Column(elem_classes=["ls-card"]):
        gr.Markdown("### LoRA Studio itself")
        gr.Markdown("Install a newer **LoraStudio.zip**: only the program files change (your settings, databases, "
                    "token, logs and venv are never touched). The files it replaces are kept in `update_backups/` "
                    "first, the self-test runs with the new files, and when it fails the old files go back by "
                    "themselves. Then restart LoRA Studio.", elem_classes=["ls-note"])
        zf = gr.File(label="LoraStudio.zip", file_types=[".zip"], type="filepath", height=110)
        plan_html = gr.HTML()
        with gr.Row(elem_classes=["ls-bar"]):
            inst = gr.Button("Install this update", variant="primary", size="sm", scale=0, min_width=190,
                             interactive=False)
            rest = gr.Button("Restart LoRA Studio", size="sm", scale=0, min_width=180)
            back = gr.Button("Roll back the last update", size="sm", scale=0, min_width=210)
        ulog = gr.Textbox(label="LoRA Studio update", lines=10, max_lines=24, interactive=False, autoscroll=True,
                          elem_id="ls_selfupd")
        import ghupdate as GH
        with gr.Accordion("From GitHub: updates without a zip", open=True):
            gr.Markdown(f"Checks [github.com/{GH.repo()}](https://github.com/{GH.repo()}) (branch `{GH.branch()}`) for a new "
                        "release (or, with *Betas too*, the newest upload) and installs it the same safe way as a zip: a backup first, the self-test, and "
                        "back by itself when the test fails. Only when you press the buttons. While the repository is "
                        "private, a **GitHub token** with read access to it is needed (github.com › Settings › Developer "
                        "settings › Fine-grained tokens: this repository, *Contents: Read-only*); it is saved in "
                        "`userdata/github_token.txt`, readable only by you, and left out of every report.",
                        elem_classes=["ls-note"])
            with gr.Row(elem_classes=["ls-bar"]):
                gh_tok = gr.Textbox(label="GitHub token", type="password", scale=4,
                                    placeholder=(f"saved: {GH.mask(GH.read_token())}" if GH.read_token()
                                                 else "github_pat_… (empty: no token)"))
                gh_tok_save = gr.Button("Save token", size="sm", scale=0, min_width=120)
                gh_tok_test = gr.Button("Test token", size="sm", scale=0, min_width=120)
            gh_chan = gr.Radio(list(GH.CHANNELS.values()), value=GH.CHANNELS[GH.channel()], label="Updates from",
                               info="Releases only: the versions published as a release (tested). Betas too: also "
                                    "every newer upload in between, which may still have rough edges.")
            with gr.Row(elem_classes=["ls-bar"]):
                gh_check = gr.Button("Check GitHub", size="sm", scale=0, min_width=150)
                gh_inst = gr.Button("Install from GitHub", variant="primary", size="sm", scale=0, min_width=190,
                                    interactive=False)
            gh_md = gr.Markdown()

    def on_file(path):
        h = selfupdate_plan_html(path)
        try:
            ok = bool(path) and any(r["state"] in ("changed", "new") for r in SU.plan(path)["rows"])
        except Exception:                                # noqa: BLE001
            ok = False
        return h, gr.update(interactive=ok)
    zf.change(on_file, zf, [plan_html, inst])

    def do_install(path):
        lines = []
        if not path:
            yield "Pick the LoraStudio.zip first."
            return
        done = {"ok": None}

        def work():
            done["ok"] = SU.install(path, say=lines.append)
        th = threading.Thread(target=work, daemon=True, name="selfupdate")
        th.start()
        while th.is_alive():
            time.sleep(0.7)
            yield "\n".join(lines)
        if done["ok"]:
            try:                                         # the training venv is checked after every update too
                import venvcheck as VC
                r = VC.inspect()
                lines.append(("✔ " if r["ok"] else "! ") + "Training venv: " + r["text"]
                             + ("" if r["ok"] else " · Settings › Updates › The training venv › Fix the training venv"))
            except Exception:                            # noqa: BLE001
                pass
        yield "\n".join(lines)
    inst.click(do_install, zf, ulog)

    def do_back():
        lines = []
        SU.rollback(say=lines.append)
        return "\n".join(lines)
    back.click(do_back, None, ulog)

    def gh_save(tok):
        msg = GH.save_token(tok)
        t = GH.read_token()
        return msg, gr.update(value="", placeholder=f"saved: {GH.mask(t)}" if t else "github_pat_… (empty: no token)")
    gh_tok_save.click(gh_save, gh_tok, [gh_md, gh_tok])
    gh_tok_test.click(GH.test_token, None, gh_md)

    def gh_do_check():
        c = GH.check()
        return c["text"], gr.update(interactive=bool(c.get("ok")) and not c.get("same"))
    gh_check.click(gh_do_check, None, [gh_md, gh_inst])

    def gh_set_chan(c):
        GH.set_channel(c)
        return "", gr.update(interactive=False)          # check again for the chosen kind
    gh_chan.change(gh_set_chan, gh_chan, [gh_md, gh_inst])

    def gh_do_install():
        lines = []
        done = {"ok": None}

        def work():
            done["ok"] = GH.install(say=lines.append)
        th = threading.Thread(target=work, daemon=True, name="ghupdate")
        th.start()
        while th.is_alive():
            time.sleep(0.7)
            yield "\n".join(lines), gr.update()
        yield "\n".join(lines), gr.update(interactive=not done["ok"])
    gh_inst.click(gh_do_install, None, [ulog, gh_inst])
    rest.click(SU.restart, None, ulog, js="""() => { setTimeout(function wait(){
        fetch(location.href, {cache: 'no-store'}).then(function(r){ if (r.ok) location.reload(); else setTimeout(wait, 2000); })
        .catch(function(){ setTimeout(wait, 2000); }); }, 6000); }""")


# --------------------------------------------------------------------------- Version
def version_html() -> str:
    import version
    rows = []
    label = {"ok": "", "old": "should be {want}", "none": "no version", "missing": "missing"}
    ps = version.parts()
    for p in ps:
        cls = "hc-ok" if p["state"] == "ok" else "hc-warn"
        note = label[p["state"]].format(want=p["want"])
        upd = '<span class="ls-chip ok">changed</span>' if p["new"] and len(version.CHANGES) > 1 else ""
        rows.append(f'<tr><td><span class="hc-dot {cls}">{"✓" if p["state"] == "ok" else "!"}</span></td>'
                    f'<td><code>{html.escape(p["name"])}</code></td><td>{html.escape(p["version"] or "-")}</td>'
                    f'<td>{upd}</td><td>{html.escape(p["what"])}</td><td class="ls-ver-bad">{html.escape(note)}</td></tr>')
    bad = version.problems()
    n_new = sum(1 for p in ps if p["new"])
    hist = "".join(f'<div class="ls-ver-h"><b>{html.escape(v)}</b> {html.escape(txt)}</div>'
                   for v, txt in version.CHANGES)
    head = (f'<div class="hc-card"><div class="hc-h">LoRA Studio {version.VERSION}</div>'
            + (f'<div>{n_new} of {len(ps)} parts were changed by this update; the others keep the version of the update '
               'that last changed them.</div>' if len(version.CHANGES) > 1 else "")
            + ("<div>All parts are the ones of this release.</div>" if not bad else
               f"<div><b>{len(bad)} part(s) are not the ones of this release.</b> Usually a file left behind by a "
               "half-done copy: put all the files from LoraStudio.zip in the program folder again (your settings are "
               "not touched).</div>")
            + f'<div class="ls-ver-hist">{hist}</div>'
            '<div class="ls-note">Version form: release.ddmmyy, and a letter at the end for a later update of the '
            'same release (1.241026 = release 1 of 24 Oct 2026; 1.241026b = an update of it).</div>'
            '<div class="ls-note">Free software under the GNU General Public License, version 3 (LICENSE in the program '
            'folder); no warranty. © 2026 L. G. H. Druidican.</div></div>')
    return (head + '<div class="hc-card"><div class="rs-wrap"><table class="rs-table ls-ver-t"><thead><tr><th></th>'
            '<th>Part</th><th>Version</th><th>This update</th><th>What it does</th><th></th></tr></thead><tbody>'
            + "".join(rows) + "</tbody></table></div></div>")


MC = {"proc": None}


def machine_check(train: bool):
    """Run selftest.py --machine in its own process and show its output as it comes (generator)."""
    import subprocess
    sg = sys.modules.get("sdgui")
    if train and sg is not None and (sg.RUN.running() or (sg.QUEUE.get("running") if hasattr(sg, "QUEUE") else False)):
        yield "A training run or the queue is active: the tiny training would compete for the GPU. Try again when it is done."
        return
    if MC["proc"] is not None and MC["proc"].poll() is None:
        yield "A check is running already."
        return
    here = __import__("lspaths").ROOT                 # selftest.py is in the LoRA Studio folder
    cmd = [sys.executable, "-u", str(here / "selftest.py"), "--machine"] + ([] if train else ["--no-train"])
    proc = subprocess.Popen(cmd, cwd=str(here), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            errors="replace")
    MC["proc"] = proc
    lines = []
    for line in proc.stdout:
        lines.append(line.rstrip("\n"))
        yield "\n".join(lines)
    proc.wait()
    lines.append("" if proc.returncode == 0 else f"(the check stopped with exit code {proc.returncode})")
    yield "\n".join(lines)


def version_panel(app):
    out = gr.HTML(version_html())
    with gr.Row(elem_classes=["ls-bar"]):
        gr.Button("Check again", size="sm", scale=0, min_width=130).click(version_html, None, out)
    with gr.Column(elem_classes=["ls-card"]):
        gr.Markdown("**Check this machine.** Looks at your GPU, the trainers and their venv, the model files of every "
                    "LoRA type, Ollama, ComfyUI and the picture tools, and can train a tiny LoRA (10 steps at 512 px "
                    "on 8 made-up pictures, in a temporary folder that is deleted afterwards) to see that a real run "
                    "starts, uses the GPU and saves a file. Nothing of yours is changed. The report "
                    "(`userdata/lorastudio_check_<time>.md`; no token, home folder as ~, no dataset "
                    "names) is the file to upload when something is not right. Same as "
                    "`./venv/bin/python selftest.py --machine` in a terminal.", elem_classes=["ls-note"])
        with gr.Row(elem_classes=["ls-bar"]):
            quick = gr.Button("Check (no training, about a minute)", size="sm", scale=0, min_width=280)
            full = gr.Button("Check + tiny training (a few minutes)", size="sm", variant="primary", scale=0,
                             min_width=300)
        log = gr.Textbox(label="Machine check", lines=14, max_lines=30, autoscroll=True, interactive=False,
                         elem_id="ls_mcheck")

    def run_quick():
        yield from machine_check(False)

    def run_full():
        yield from machine_check(True)
    quick.click(run_quick, None, log)
    full.click(run_full, None, log)
    import notice
    with gr.Accordion("Licences and credits", open=False, elem_classes=["ls-card"]):
        gr.Markdown(f"### {notice.TITLE}\n\n{notice.TEXT}\n\nThe full GPL-3.0 text is in `LICENSE`, this text in "
                    "`NOTICE`, both in the program folder.")
    with gr.Column(elem_classes=["ls-card"]):
        gr.Markdown("**Report for Claude.** One file with the version, the machine and the newest machine check "
                    "(token and home folder removed), copied so you can paste it into a chat. For a failed run, use "
                    "the button on that run in Training › Results: it adds the launch script and the end of the log.",
                    elem_classes=["ls-note"])
        rnote = gr.Textbox(label="What do you see? (optional)", lines=2)
        with gr.Row(elem_classes=["ls-bar"]):
            rbtn = gr.Button("Report for Claude", size="sm", scale=0, min_width=170)
        rfile = gr.File(label="The report", visible=False, interactive=False)
        rtext = gr.Textbox(visible=False)
        rmsg = gr.Markdown()

    def make_report(note):
        import report
        path, text = report.make("", note or "")
        return gr.update(value=str(path), visible=True), text, f"✔ Made and copied: `userdata/reports/{path.name}`."
    import training_ui
    rbtn.click(make_report, rnote, [rfile, rtext, rmsg]).then(None, rtext, None, js=training_ui.COPY_JS)
    gr.Markdown("A quick self-test of the program itself (no training, nothing is changed): "
                "`./venv/bin/python selftest.py` in the program folder. It checks the versions, that every part "
                "loads, that the page builds, and the launch script, dataset and test-picture logic on a small "
                "made-up dataset.", elem_classes=["ls-note"])


# --------------------------------------------------------------------------- Test pictures (ComfyUI)
def comfy_panel(app):
    import comfytest as CT
    gr.Markdown("After a training run, LoRA Studio can make the same test pictures with each save point in your "
                "ComfyUI, side by side in **Training › Results**, to pick the best epoch. It reads your own workflow for "
                "the model, sampler, steps, CFG, size and negative prompt, and sends ComfyUI a small graph of its own "
                "nodes with those settings (no custom nodes needed; upscaling and face detailers are left out). "
                "Saved in `comfy_settings.json`; your workflows are kept in `comfy_workflows/`.", elem_classes=["ls-note"])
    with gr.Column(elem_classes=["ls-card"]):
        with gr.Row(equal_height=True):
            url = gr.Textbox(label="ComfyUI address", scale=3)
            test = gr.Button("Test", size="sm", scale=0, min_width=90)
        lora_dir = gr.Textbox(label="The LoRA folder ComfyUI reads (empty = your first LoRA folder, Settings › Folders)",
                              info="Each save point is copied into _lorastudio_test/ in it for a moment, so ComfyUI "
                                   "can load it.")
        with gr.Row():
            n = gr.Number(label="Test prompts per LoRA", value=8, precision=0, minimum=1, maximum=24, scale=1)
            around = gr.Number(label="Save points on each side of the one to try first", value=2, precision=0,
                               minimum=0, maximum=6, scale=1)
        auto = gr.Checkbox(label="Make them by themselves after the training queue has finished (when ComfyUI "
                                 "answers; it is tried again every minute)", value=True)
        tmsg = gr.Markdown()
    fams = {}
    for fam, lab in CT.FAMILIES.items():
        with gr.Column(elem_classes=["ls-card"]):
            gr.Markdown(f"**{lab}**")
            info = gr.Markdown()
            with gr.Row(equal_height=True):
                up = gr.File(label="Your workflow (ComfyUI › Workflow › Save / Export, the .json file)",
                             file_types=[".json"], scale=3, height=110)
            fixed = gr.Textbox(label="Fixed tags in front of every test prompt (e.g. quality tags)", lines=2)
            fams[fam] = (info, up, fixed)
    with gr.Row():
        save = gr.Button("Save", variant="primary", size="sm", scale=0, min_width=120)
    msg = gr.Markdown()
    gr.Markdown("The prompts are 8 captions of the dataset (picked once per run), each with the fixed tags in front, "
                "and the same seed in every column, so only the epoch changes.", elem_classes=["ls-note"])

    def summary(fam):
        wf = CT.settings()["families"].get(fam)
        if not wf:
            return "No workflow yet."
        model = wf["ckpt"] or wf["unet"]
        return (f"Model `{model}` · {wf['sampler']} / {wf['scheduler']} · {wf['steps']} steps · CFG {wf['cfg']} · "
                f"{wf['width']}×{wf['height']}" + (f" · CLIP skip {abs(wf['clip_skip'])}" if wf["clip_skip"] else "")
                + (f" · text encoder `{wf['clip']}` · VAE `{wf['vae']}`" if wf["kind"] == "unet" else ""))

    outs = [url, lora_dir, n, around, auto] + [x for f in fams.values() for x in (f[0], f[2])]

    def load():
        s = CT.settings()
        vals = [s["url"], s["lora_dir"], s["prompts"], s["around"], s["auto"]]
        for fam in fams:
            vals += [summary(fam), s["fixed"].get(fam, "")]
        return vals
    app.load(load, None, outs)
    for fam, (info, up, fixed) in fams.items():
        def on_up(f, fam=fam):
            if not f:
                return gr.update(), gr.update(), ""
            try:
                wf, note = CT.import_workflow(fam, f if isinstance(f, str) else f.name)
            except Exception as e:                       # noqa: BLE001
                return gr.update(), gr.update(), f"❌ Could not read that workflow: {e}"
            return summary(fam), CT.settings()["fixed"].get(fam, ""), note
        up.upload(on_up, up, [info, fixed, msg])

    def do_save(u, ld, n_, ar, au, *rest):
        fixed = {fam: rest[2 * i + 1] for i, fam in enumerate(fams)}
        CT.save_settings(url=(u or "").strip() or CT.DEFAULTS["url"], lora_dir=(ld or "").strip(),
                         prompts=int(n_ or 8), around=int(ar or 0), auto=bool(au), fixed=fixed)
        return "✔ Saved."
    save.click(do_save, outs, msg)

    def do_test(u):
        u = (u or "").strip() or CT.DEFAULTS["url"]
        if not CT.alive(u):
            return f"❌ ComfyUI does not answer at `{u}`. Start it, then press Test again."
        names = CT.lora_names(u)
        return f"✔ ComfyUI answers at `{u}` ({len(names)} LoRAs in its list)."
    test.click(do_test, url, tmsg)
    upscale_card(app)


def speed_panel(app):
    """Settings › Speed test: which settings make each LoRA type train fastest on this computer (speedjobs.py)."""
    import speedjobs as SJ
    import ticker
    gr.Markdown("Tick the LoRA types to test. Each one trains a few steps on an invented test dataset (20 plain "
                "pictures; none of your datasets is used) with different settings: fewer swapped blocks, pinned "
                "memory, no gradient checkpointing, other attention, fp8, torch.compile and ROCm speed settings. It "
                "measures seconds per step and video memory. Test some types now and others later; the results stay. "
                "When a type has faster settings, tick **Use the faster settings** below to make every new launch "
                "script of that type use them. Close ComfyUI before a test; Ollama is stopped by itself.",
                elem_classes=["ls-note"])
    table = gr.HTML()
    with gr.Column(elem_classes=["ls-card"]):
        pick = gr.CheckboxGroup(choices=[], label="Test these LoRA types", elem_id="sp_pick")
        depth = gr.Radio(SJ.DEPTHS, value="quick", label="How thorough")
        with gr.Row(elem_classes=["ls-bar"]):
            go = gr.Button("Start the speed test", variant="primary", size="sm", scale=0, min_width=190)
            halt = gr.Button("Stop", variant="stop", size="sm", scale=0, min_width=90)
        msg = gr.Markdown()
        live = gr.Markdown(elem_id="sp_live")
    with gr.Column(elem_classes=["ls-card"]):
        gr.Markdown("**Results**")
        which = gr.Dropdown(choices=[], label="LoRA type", interactive=True)
        use = gr.Checkbox(label="Use the faster settings for new launch scripts of this type",
                          info="Existing launch scripts stay as they are; make a new version in step 9. Untick to "
                               "go back to the normal settings.")
        use_msg = gr.Markdown()
        rep_md = gr.Markdown(elem_id="sp_report")

    def tbl():
        rows = []
        for t in SJ.types():
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

    def choices():
        ts = SJ.types()
        ready = [(t["label"], t["key"]) for t in ts if t["ready"]]
        tested = [(t["label"], t["key"]) for t in ts if t["result"]]
        return ready, tested

    def load():
        ready, tested = choices()
        cur = tested[0][1] if tested else None
        return (tbl(), gr.update(choices=ready), gr.update(choices=tested, value=cur)) + show(cur)

    def show(k):
        if not k:
            return gr.update(value=False, interactive=False), "", ""
        r = SJ.results().get(k) or {}
        try:
            text = Path(r.get("report", "")).read_text(encoding="utf-8")
        except OSError:
            text = "The report file is gone."
        can = bool(r.get("changes") or r.get("env"))
        return gr.update(value=SJ.use_on(k), interactive=can), "", text

    def live_view():
        if not SJ.JOB["running"] and not SJ.JOB["lines"]:
            return gr.update()
        lines = SJ.JOB["lines"][-14:] + ([f"⏳ {SJ.JOB['now']}"] if SJ.JOB["now"] else [])
        return "```\n" + "\n".join(lines) + "\n```"

    def started(ks, d):
        return SJ.start(ks or [], d or "quick"), live_view()

    app.load(load, None, [table, pick, which, use, use_msg, rep_md])
    go.click(started, [pick, depth], [msg, live])
    halt.click(lambda: SJ.stop(), None, msg)
    which.change(show, which, [use, use_msg, rep_md])
    use.input(lambda k, v: SJ.set_use(k, v) if k else "", [which, use], use_msg).then(tbl, None, table)
    ticker.timer(2.0, live_view, None, live, show_progress="hidden")
    def follow(cur):
        """While a test runs and after it: the table, the list of tested types and the shown result follow it."""
        if not (SJ.JOB["running"] or SJ.JOB["done"]):
            return [gr.update()] * 5
        _, tested = choices()
        keys = [k for _, k in tested]
        cur = cur if cur in keys else (SJ.JOB["done"][-1] if SJ.JOB["done"] else (keys[0] if keys else None))
        return (tbl(), gr.update(choices=tested, value=cur)) + show(cur)
    ticker.timer(2.0, follow, which, [table, which, use, use_msg, rep_md], show_progress="hidden")
    ticker.BUSY.append(SJ.busy)


def upscale_card(app):
    """Settings › Test pictures › Upscale workflow: your ComfyUI workflow for upscaling and sharpening pictures."""
    import comfyupscale as U
    with gr.Column(elem_classes=["ls-card"]):
        gr.Markdown("**Upscale workflow** (dataset pictures)")
        gr.Markdown("Used by the **Detailed scan** (Datasets › Sort, characters) to upscale small pictures, sharpen "
                    "soft ones and enlarge detail crops. Give it your own upscale workflow, for example SeedVR2, saved "
                    "from ComfyUI (Workflow › Save or Export (API)). It runs as it is, with your models (GGUF too); "
                    "LoRA Studio only puts the picture into its Load Image node and, in a SeedVR2 Video Upscaler node, "
                    "sets the size (the short side the LoRA type trains at) and the noise. The originals stay in "
                    "`_originals/`.", elem_classes=["ls-note"])
        uinfo = gr.Markdown()
        uup = gr.File(label="Your upscale workflow (.json)", file_types=[".json"], height=110)
        noise = gr.Radio(U.NOISE_CHOICES, label="SeedVR2 noise", value="auto",
                         info="Automatic: none for sharpening at the same size, input noise 0.1-0.4 the more a picture "
                              "grows (it hides the blocks and smearing of small sources), and a little latent noise "
                              "above 3×, which can soften detail.")
        umsg = gr.Markdown()

    def uload():
        s = U.settings()
        return (f"Workflow: `{s['name']}` · {s['note']}" if U.ready() else "No upscale workflow yet."), s["noise"]
    app.load(uload, None, [uinfo, noise])

    def on_up(f):
        if not f:
            return gr.update(), ""
        try:
            note = U.import_workflow(f if isinstance(f, str) else f.name)
        except Exception as e:                           # noqa: BLE001
            return gr.update(), f"❌ Could not use that workflow: {e}"
        return uload()[0], note
    uup.upload(on_up, uup, [uinfo, umsg])
    noise.input(lambda v: (U.save_settings(noise=v), "✔ Saved.")[1], noise, umsg)
    return uinfo, noise, uload


# --------------------------------------------------------------------------- the page
def build(app, panels: dict):
    """panels: {key: callable that renders that panel's content}. Returns the components the caller may need."""
    btns, cols = {}, {}
    with gr.Row(equal_height=False, elem_id="set_page"):
        with gr.Column(scale=1, min_width=220, elem_id="set_nav"):
            for grp, items in NAV:
                if grp:
                    gr.HTML(f'<div class="set-grp">{grp}</div>')
                for key, label in items:
                    btns[key] = gr.Button(nav_label(key), elem_id=f"setnav_{key}", size="sm",
                                          elem_classes=nav_classes(key, key == "health"))
        with gr.Column(scale=5, min_width=480, elem_id="set_body"):
            for grp, items in NAV:
                for key, label in items:
                    with gr.Column(visible=key == "health", elem_classes=["set-panel"]) as col:
                        if key == "health":
                            gr.Markdown("## Health check")
                            hc = gr.HTML(elem_id="hc_html")
                            with gr.Row():
                                again = gr.Button("Check again", size="sm", scale=0, min_width=130)
                                setup = gr.Button("Run the setup page again", size="sm", scale=0, min_width=220)
                            gr.Markdown("*Run the setup page again* walks through folders, trainers, models and AI helpers "
                                        "in order; LoRA Studio restarts on the setup page.", elem_classes=["ls-note"])
                            setup_msg = gr.Markdown()
                        else:
                            gr.Markdown(f"## {nav_label(key)}")
                            try:
                                panels[key]()
                            except Exception as e:      # noqa: BLE001
                                gr.Markdown(f"This panel could not be built: {e!r}")
                    cols[key] = col

    def pick(key):
        counts = nav_counts()
        outs = [gr.update(visible=k == key) for k in KEYS]
        for k in KEYS:
            label = nav_label(k)
            n = counts.get(k, 0)
            outs.append(gr.update(value=f"{label}  ·  {n}" if n else label, elem_classes=nav_classes(k, k == key)))
        return outs

    outs = [cols[k] for k in KEYS] + [btns[k] for k in KEYS]
    for k in KEYS:
        btns[k].click(lambda k=k: pick(k), None, outs, show_progress="hidden")
    again.click(health_html, None, hc)
    btns["health"].click(health_html, None, hc)
    app.load(health_html, None, hc).then(lambda: pick("health"), None, outs)

    def restart_setup():
        try:
            import launchopts
            launchopts.restart(extra=["--setup"])
            return "Restarting on the setup page … this page reloads by itself."
        except TypeError:
            return "Run `./start.sh --setup` to see the setup page again."
        except Exception as e:                           # noqa: BLE001
            return f"Could not restart: {e!r}. Run `./start.sh --setup`."
    setup.click(restart_setup, None, setup_msg)
    return {"buttons": btns, "health": hc}
