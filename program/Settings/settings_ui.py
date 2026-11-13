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
__version__ = "1.131126b"  # LoRA Studio version, see version.py

import html
import os
import shutil
import sys
import time
from pathlib import Path


NAV = [                                       # grouped like the work: start, datasets, training, results, collection
    ("", [("health", "Health check")]),
    ("START", [("start", "Start and network"), ("folders", "Folders")]),
    ("DATASETS", [("ai", "AI helpers")]),
    ("TRAINING", [("trainers", "Trainers and GPU"), ("models", "Model files"), ("speed", "Speed test")]),
    ("RESULTS", [("comfy", "Test pictures (ComfyUI)")]),
    ("COLLECTION", [("look", "Look and base models")]),
    ("CARE", [("backup", "Backup"), ("updates", "Updates"), ("version", "Version")]),
]
KEYS = [k for _, items in NAV for k, _ in items]
NAV_PART = {"ai": "ollama", "trainers": "sd_scripts"}       # greyed with "(… missing)" when that part is not installed


def nav_label(key: str) -> str:
    import features as FT
    label = dict(i for _, it in NAV for i in it)[key]
    return FT.label(label, NAV_PART[key]) if key in NAV_PART else label


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


# --------------------------------------------------------------------------- Automatic improvement
KEEP_CHOICES = [("The suggestion for the LoRA type, at most the number below", "suggested"),
                ("Always the number below (fewer when there are not that many)", "fixed")]
WM_CHOICES = [("Remove them (the picture tools paint over them; crop if that fails)", "remove"),
              ("Crop them off (the recommended crop)", "crop"), ("Set those pictures aside", "aside"),
              ("Leave them (decide later in Datasets)", "leave")]
CAP_CHOICES = [("Only pictures without a caption", "missing"), ("Write every caption again", "all")]
CONS_CHOICES = [("Leave it out", "ignore"), ("Run it", "check")]


# --------------------------------------------------------------------------- What is installed


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


# --------------------------------------------------------------------------- Test pictures (ComfyUI)


# --------------------------------------------------------------------------- the page
