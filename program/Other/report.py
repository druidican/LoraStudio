# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Report for Claude: one file with everything needed to ask about a problem, instead of copy-pasting by hand.

make(logname="") -> (path, text). With a run log: the run (result, time, what went wrong), its launch script and the
end of its log. Always: the LoRA Studio version and parts, the machine (GPU, system, Python, Gradio) and the newest
machine check. Before it is written, the Hugging Face token and anything that looks like a token are removed and
your home folder is shown as ~. Saved as reports/lorastudio_report_<time>.md in the program folder; the page also
copies it, so it can be pasted or dropped into a chat.
"""
from __future__ import annotations
__version__ = "1.251026o"  # LoRA Studio version, see version.py

import platform
import re
import sys
import time
from pathlib import Path

import lspaths as _LP  # noqa: E402  (program/lspaths.py: ROOT, PROG, USER)
HERE = _LP.USER
OUT_DIR = HERE / "reports"
HOME = str(Path.home())
TOKEN_RE = re.compile(r"\b(hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
LOG_LINES = 200


def redact(text: str) -> str:
    text = TOKEN_RE.sub("[token removed]", text or "")
    sg = sys.modules.get("sdgui")
    try:
        tok = sg.read_hf_token() if sg else ""
        if tok and len(tok) > 6:
            text = text.replace(tok, "[token removed]")
    except Exception:                                    # noqa: BLE001
        pass
    text = re.sub(r'^(\s*(export\s+)?(HF_TOKEN|HUGGING_FACE_HUB_TOKEN|HUGGINGFACE_TOKEN)=).*$', r'\1"[removed]"', text,
                  flags=re.M)
    if HOME and HOME != "/":
        text = text.replace(HOME, "~")
    return text


def _log_end(path: Path, n: int = LOG_LINES) -> str:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 400_000))
            raw = fh.read().decode("utf-8", "replace")
    except OSError:
        return ""
    raw = ANSI_RE.sub("", raw).replace("\r", "\n")
    out, last_prog = [], None
    for line in raw.splitlines():
        if not line.strip() or line.startswith("##LS_"):
            continue
        if re.match(r"^\s*steps:\s*\d+%", line):          # progress bars: keep only every change of the percentage
            pct = re.match(r"^\s*steps:\s*(\d+)%", line).group(1)
            if pct == last_prog:
                continue
            last_prog = pct
        out.append(line)
    return "\n".join(out[-n:])


def _machine() -> list[str]:
    lines = [f"- System: {platform.platform()} · Python {platform.python_version()}"]
    try:
        import gradio
        lines.append(f"- Gradio {gradio.__version__}")
    except Exception:                                    # noqa: BLE001
        pass
    try:
        import statusbar
        for g in statusbar.machine()["gpus"]:
            lines.append(f"- GPU: {g['name']} · {g['vram_total'] / 1024 ** 3:.0f} GB")
    except Exception:                                    # noqa: BLE001
        pass
    sg = sys.modules.get("sdgui")
    if sg is not None:
        try:
            env = sg.env_saved()
            if env:
                lines.append("- GPU settings: " + ", ".join(f"{k}={v}" for k, v in env.items()))
        except Exception:                                # noqa: BLE001
            pass
    return lines


def _newest_check() -> str:
    checks = sorted(HERE.glob("lorastudio_check_*.md"))
    if not checks or time.time() - checks[-1].stat().st_mtime > 14 * 86400:
        return ""
    text = checks[-1].read_text(encoding="utf-8", errors="replace")
    keep = [l for l in text.splitlines() if l.startswith(("| ✕", "| !", "- Result", "- LoRA Studio"))
            or "PyTorch" in l or "Training ran" in l]
    return f"Newest machine check ({checks[-1].name}):\n" + "\n".join(keep[:40])


def _newest_install() -> str:
    """The main lines of the newest install report (install.sh writes reports/install_report_<time>.md)."""
    reps = sorted(OUT_DIR.glob("install_report_*.md"))
    if not reps:
        return ""
    text = reps[-1].read_text(encoding="utf-8", errors="replace")
    keep = [l for l in text.splitlines() if l.startswith(("- Result", "- LoRA Studio", "- System", "- Choices", "- Graphics",
                                                          "- ! ", "**Stopped"))]
    return f"Newest install report ({reps[-1].name}):\n" + "\n".join(keep[:40])


def make(logname: str = "", note: str = "") -> tuple[Path, str]:
    import version
    parts = []
    parts.append(f"# LoRA Studio report · {time.strftime('%Y-%m-%d %H:%M')}")
    parts.append("")
    parts.append("Please help me with this problem in LoRA Studio (a Gradio app that trains LoRAs with sd-scripts / "
                 "musubi-tuner). Everything below was collected by the program.")
    if note.strip():
        parts += ["", "## What I see", "", note.strip()]
    bad = version.problems()
    parts += ["", "## Program", "", f"- LoRA Studio {version.VERSION}"
              + (f" · {len(bad)} part(s) not of this release: " + ", ".join(f"{p['name']} {p['version']}" for p in bad)
                 if bad else " · all parts of this release")]
    parts += _machine()
    sg = sys.modules.get("sdgui")
    if logname and sg is not None:
        f = sg.LOG_DIR / logname
        try:
            import training_ui as T
            hist = {r["log"]: r for r in T.runs()}
            r = hist.get(logname) or {}
            info = r.get("info")
            name = T.run_name(info) if info else logname
            parts += ["", "## The run", "", f"- {name} · {r.get('result', '?')} · started {r.get('started', '?')} · "
                      f"took {r.get('dur', '?')}", f"- log: gui_logs/{logname}"]
            fc = T.fail_info(logname)
            if fc:
                parts += [f"- LoRA Studio thinks: **{fc['title']}** ({fc['code']}): {fc['what']}",
                          f"- Its advice: {fc['fix']}"]
                if fc.get("line"):
                    parts.append(f"- The line it found: `{fc['line']}`")
            if info:
                try:
                    st = sg.dataset_stats(info["dataset"])
                    parts.append("- Dataset: " + " + ".join(f"{x['folder'].name}: {x['n']} pictures, "
                                                           f"{x['captioned']} captions" for x in st["rows"]))
                except Exception:                        # noqa: BLE001
                    pass
                parts += ["", "## The launch script", "", "```bash", info["text"].rstrip(), "```"]
        except Exception as e:                           # noqa: BLE001
            parts += ["", f"(the run could not be read: {e})"]
        parts += ["", f"## The end of the log (last {LOG_LINES} lines)", "", "```", _log_end(f), "```"]
    inst = _newest_install()
    if inst:
        parts += ["", "## Install", "", "```", inst, "```"]
    chk = _newest_check()
    if chk:
        parts += ["", "## Machine check", "", "```", chk, "```"]
    text = redact("\n".join(parts) + "\n")
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / f"lorastudio_report_{time.strftime('%Y%m%d_%H%M%S')}.md"
    path.write_text(text, encoding="utf-8")
    for old in (sorted(OUT_DIR.glob("lorastudio_report_*.md"))[:-30] + sorted(OUT_DIR.glob("install_report_*.md"))[:-10]):      # keep the newest 30
        old.unlink(missing_ok=True)
    return path, text
