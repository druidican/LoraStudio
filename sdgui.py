#!/usr/bin/env python3
"""
sdgui.py - small Gradio front-end for your existing sd-scripts launch scripts.

It does NOT touch your sd-scripts install or its venv. It:
  * finds your *.sh launch scripts in <root>/ and <root>/images/** (recursively)
  * runs the saved script (bash, default working dir = <root>) and streams the log
  * can resume a crashed run from a saved --save_state folder (adds --resume to a temp copy)
  * Stop button (Ctrl+C first, then escalates to SIGTERM / SIGKILL)
  * Create Data Launch Script tab: pick a LoRA folder and an existing launch script (settings are loaded
    into a form) or '+ New launch script' (recommended settings for any sd-scripts architecture, with the
    reasoning and step calculation). Edit the form or the raw text, save over the script or as a new one.
    Uses your scan_lora_dataset_ny.py for presets and templates (imported, not copied).
    Saving also creates the missing <N>_name repeat folder plus model/ and log/ folders; sample images are a yes/no checkbox.
  * Settings tab: model / VAE / text-encoder files per LoRA type (folder + file dropdown), saved in <sd-scripts>/settings.txt
    and used for new launch scripts
  * Review tab: browse a dataset's images with their captions, edit captions, bulk find/replace + undo
  * Monitor tab: newest sample images from disk + optional TensorBoard
  * GPU status (top right) from rocm-smi: load, VRAM, temperature, power; Hugging Face token status
    (token read from <sd-scripts>/huggingtoken.txt and written into `export HF_TOKEN=` in launch scripts)
  * Scan library button: finds new image libraries in <sd-scripts>/images
  * Repeat folders panel: rename a folder to the correct <N>_name format

Install (in its OWN venv, not your training venv):
    python3 -m venv ~/sdgui-venv
    ~/sdgui-venv/bin/pip install "gradio>=5" tensorboard

Run:
    ~/sdgui-venv/bin/python sdgui.py --root ~/sd-scripts
    then open http://127.0.0.1:7860

Security: this runs shell text from a web page. It binds to 127.0.0.1 only; don't expose it.
"""
import argparse
import atexit
import codecs
import difflib
import glob
import hashlib
import importlib.util
import inspect
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import gradio as gr

HOME = Path.home()
LOG_DIR = Path(__file__).resolve().parent / "gui_logs"
MAX_LOG_LINES = 400      # lines shown in the UI
MAX_KEPT_LINES = 5000    # lines kept in memory (full log is also on disk)
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp"}
VAR_RE = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
TRAIN_RE = re.compile(r"\b[\w.-]*train[\w.-]*\.py\b")
# the tqdm line sd-scripts prints while training, e.g.
#   steps:  12%|█▏        | 300/2430 [05:12<36:30,  1.03s/it, avr_loss=0.0831]
PROG_RE = re.compile(r"(steps:\s*\d+%\|[^|\n]*\|\s*\d+/\d+\s*\[[^\]\n]*\])")
NO_STATE = "none"
STATE_MARKERS = ("train_state.json", "random_states_0.pkl", "optimizer.bin")

ROOT: Path = HOME / "sd-scripts"
EXTRA_SCRIPT_DIRS: list[Path] = []
DEFAULT_VENV = ""
SCAN_SCRIPT: Path | None = None
CACHE_DIR = Path(__file__).resolve().parent / "gui_cache"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
CAPTION_EXTS = (".txt", ".caption")
REPEAT_RE = re.compile(r"^(\d+)_")
IGNORE_DIRS = {"model", "models", "output", "outputs", "samples", "sample", "log", "logs"}   # never dataset folders
CSS = ("#gpu_status, #hf_status {text-align: right;} #gpu_status p, #hf_status p {margin: 0;} "
       "#gpu_status .grow {display: flex; flex-wrap: wrap; gap: 6px; justify-content: flex-end; align-items: center;} "
       "#gpu_status .gname {font-weight: 600; margin-right: 2px;} #gpu_status .gmsg {opacity: .7;} "
       "#gpu_status .gbar {position: relative; width: 175px; height: 22px; border-radius: 6px; overflow: hidden; "
       "background: #5b6470;} "
       "#gpu_status .gfill {position: absolute; inset: 0 auto 0 0; transition: width .6s ease, background .6s;} "
       "#gpu_status .gtxt {position: relative; display: flex; justify-content: space-between; padding: 0 8px; "
       "line-height: 22px; font-size: .78rem; white-space: nowrap;} "
       "#gpu_status .gtxt, #gpu_status .gtxt * {color: #fff !important; text-shadow: 0 1px 2px rgba(0,0,0,.6);} "
       "#progress_line textarea, #progress_line input {font-family: monospace; font-size: 0.95em;}")
LAUNCH_HAS_CSS = "css" in inspect.signature(gr.Blocks.launch).parameters   # Gradio 6 moved css to launch()
LAST_CHANGE = {"desc": "", "files": {}}   # last caption edit, for Undo


# --------------------------------------------------------------------------- run state
class Run:
    def __init__(self):
        self.proc = None
        self.lines: list[str] = []
        self.cur = ""            # in-progress line (tqdm uses \r)
        self.lock = threading.Lock()
        self.started = 0.0
        self.ended = None
        self.name = ""
        self.logfile = None
        self.tmp = None
        self.version = 0         # bumps whenever the log changes
        self.secret = ""         # HF token, masked out of the log
        self.progress = ""       # latest "steps: ..." tqdm line

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None


RUN = Run()


def fmt(sec: float) -> str:
    sec = int(sec)
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def status_text() -> str:
    if RUN.proc is None:
        return "**Status:** idle"
    el = (RUN.ended or time.time()) - RUN.started
    if RUN.running():
        return f"**Status:** 🟢 running `{RUN.name}`, {fmt(el)} elapsed"
    return f"**Status:** ⚪ finished `{RUN.name}`, exit code {RUN.proc.returncode}, {fmt(el)}"


def log_text() -> str:
    with RUN.lock:
        cur = RUN.cur
        # the live progress bar has its own line above the log, so keep it out of the log tail
        tail = RUN.lines[-MAX_LOG_LINES:] + ([cur] if cur and not PROG_RE.search(cur) else [])
    return "\n".join(tail)


def progress_text() -> str:
    if RUN.proc is None:
        return "steps: idle"
    if RUN.progress:
        return RUN.progress
    return "steps: waiting for training to start..." if RUN.running() else "steps: no training progress was printed"


def _reader(run: Run):
    dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
    fd = run.proc.stdout.fileno()
    pending_cr = False
    with open(run.logfile, "ab") as lf:
        while True:
            try:
                chunk = os.read(fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            text = dec.decode(chunk)
            if run.secret:
                text = text.replace(run.secret, mask_token(run.secret))
            lf.write(text.encode("utf-8"))
            lf.flush()
            with run.lock:
                for ch in text:
                    if ch == "\r":
                        pending_cr = True
                        continue
                    if ch == "\n":
                        mp = PROG_RE.search(run.cur)
                        if mp:
                            run.progress = mp.group(1)
                        run.lines.append(run.cur)
                        run.cur = ""
                        pending_cr = False
                        continue
                    if pending_cr:          # bare \r = overwrite current line (progress bars)
                        run.cur = ""
                        pending_cr = False
                    run.cur += ch
                mp = PROG_RE.search(run.cur)
                if mp:
                    run.progress = mp.group(1)
                if len(run.lines) > MAX_KEPT_LINES:
                    del run.lines[:1000]
                run.version += 1
    code = run.proc.wait()
    with run.lock:
        if run.cur:
            run.lines.append(run.cur)
            run.cur = ""
        run.lines.append(f"--- exited with code {code} ---")
        run.ended = time.time()
        run.version += 1
    try:
        run.tmp.unlink()
    except OSError:
        pass


# --------------------------------------------------------------------------- scripts
def list_scripts():
    """<root>/*.sh, <root>/images/**/*.sh, plus any --scripts-dir (recursive)."""
    found = {}

    def add(p: Path):
        if not p.name.startswith(".gui_run_"):
            found[p.resolve()] = p

    if ROOT.is_dir():
        for p in ROOT.glob("*.sh"):
            add(p)
        img = ROOT / "images"
        if img.is_dir():
            for p in img.rglob("*.sh"):
                add(p)
    for d in EXTRA_SCRIPT_DIRS:
        if d.is_dir():
            for p in d.rglob("*.sh"):
                add(p)
    out = []
    for rp, p in sorted(found.items(), key=lambda kv: str(kv[0]).lower()):
        try:
            label = str(p.relative_to(ROOT))
        except ValueError:
            label = str(p)
        out.append((label, str(p)))
    return out


def refresh_scripts():
    return gr.update(choices=list_scripts())


# --------------------------------------------------------------------------- resume / states
def _script_vars(body: str) -> dict:
    vars_ = {}
    for line in body.splitlines():
        m = VAR_RE.match(line)
        if m and m.group(1) not in vars_:
            v = m.group(2).strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
                v = v[1:-1]
            vars_[m.group(1)] = v
    return vars_


def _expand(val: str, vars_: dict) -> str:
    def sub(m):
        k = m.group(1) or m.group(2)
        return vars_.get(k, os.environ.get(k, m.group(0)))
    for _ in range(6):
        new = re.sub(r"\$\{(\w+)\}|\$(\w+)", sub, val)
        if new == val:
            break
        val = new
    return os.path.expanduser(val)


def script_vars(text: str, script_dir=None) -> dict:
    """Variables set in a launch script. SCRIPT_DIR and CHARACTER_NAME are computed by the script at run
    time (`$(cd ...)`), so when the script's folder is known they are filled in directly."""
    v = _script_vars(text)
    v.setdefault("HOME", str(HOME))
    if script_dir is not None:
        v["SCRIPT_DIR"] = str(script_dir)
        v["CHARACTER_NAME"] = Path(script_dir).name
    return v


def script_arg(text: str, name: str, script_dir=None):
    """Value of --<name> in a launch script, with simple $VARs expanded (None if absent or not expandable)."""
    vars_ = script_vars(text, script_dir)
    m = re.search(rf"--{name}[ =]+(?:\"([^\"]*)\"|'([^']*)'|(\S+))", text)
    if not m:
        return None
    raw = next(g for g in m.groups() if g is not None)
    val = _expand(raw, vars_)
    return None if "$" in val or "`" in val else val


def script_args(text: str, script_dir=None):
    """Best-effort (output_dir, output_name) from the script."""
    return script_arg(text, "output_dir", script_dir), script_arg(text, "output_name", script_dir)


def script_paths(path):
    """(model_dir, log_dir) of a launch script. Every base folder has a `model` folder that receives all the
    tensor files (LoRAs, saved states, and the `sample` images), and a `log` folder for TensorBoard."""
    if not path or not Path(path).is_file():
        return None, None
    base = Path(path).parent
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    out = script_arg(text, "output_dir", base)
    log = script_arg(text, "logging_dir", base)
    return Path(out) if out else base / "model", Path(log) if log else base / "log"


def checkpoints_md(folder) -> str:
    """The finished tensor files and resumable states in a model folder, newest first."""
    p = Path(os.path.expanduser(folder or ""))
    if not (folder or "").strip():
        return "Pick a launch script in the Train tab to see its model folder."
    if not p.is_dir():
        return f"Model folder `{p}` does not exist yet. It is created when you save a launch script, or on the first run."
    files = []
    for f in p.iterdir():
        if f.is_file() and f.suffix.lower() in (".safetensors", ".ckpt", ".pt"):
            try:
                st = f.stat()
                files.append((st.st_mtime, st.st_size, f.name))
            except OSError:
                pass
    files.sort(reverse=True)
    states = scan_states(p, 1)
    head = f"**`{p}`**: {len(files)} tensor file(s)" + (f", {len(states)} saved state(s) you can resume from" if states else "")
    if not files:
        return head + ". Nothing has been saved yet."
    rows = ["", "| File | Size | Saved |", "|---|---|---|"]
    for mt, size, name in files[:40]:
        rows.append(f"| {name} | {size / 1024 ** 2:.0f} MB | {datetime.fromtimestamp(mt):%Y-%m-%d %H:%M} |")
    if len(files) > 40:
        rows.append(f"| ... and {len(files) - 40} older files | | |")
    return head + "\n".join(rows)


def scan_states(root: Path, max_depth: int = 3):
    out = []
    if not root.is_dir():
        return out
    base = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root):
        d = Path(dirpath)
        if d.name.endswith("-state") or any(f in filenames for f in STATE_MARKERS):
            out.append(d)
            dirnames[:] = []
            continue
        if len(d.parts) - base >= max_depth:
            dirnames[:] = []
    out.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return out


def state_label(d: Path) -> str:
    epoch = step = "?"
    j = d / "train_state.json"
    if j.is_file():
        try:
            js = json.loads(j.read_text())
            epoch, step = js.get("current_epoch", "?"), js.get("current_step", "?")
        except Exception:
            pass
    return f"{d.name}  |  epoch {epoch}, step {step}  |  {datetime.fromtimestamp(d.stat().st_mtime):%Y-%m-%d %H:%M}"


def find_states_ui(path, state_dir):
    text = ""
    if path and Path(path).is_file():
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
    out_dir, out_name = script_args(text, Path(path).parent if path else None)
    folder = (state_dir or "").strip() or out_dir
    root = Path(os.path.expanduser(folder)) if folder else None
    if root is None or not root.is_dir():
        gr.Warning("Set a valid search folder (usually the script's --output_dir).")
        return gr.update(choices=[("(start fresh, no resume)", NO_STATE)], value=NO_STATE), "No folder to search."
    states = scan_states(root)
    matching = [s for s in states if out_name and s.name.startswith(out_name)]
    shown = matching if matching else states
    if not states:
        msg = f"No saved states found under `{root}`. (The run needs `--save_state` to produce them.)"
    elif out_name and matching:
        msg = f"Showing {len(shown)} state(s) for output name `{out_name}` in `{root}` ({len(states) - len(shown)} others hidden)."
    elif out_name:
        msg = f"No state starts with `{out_name}`; showing all {len(states)} in `{root}`."
    else:
        msg = f"Found {len(states)} state(s) in `{root}`."
    choices = [("(start fresh, no resume)", NO_STATE)] + [(state_label(s), str(s)) for s in shown]
    return gr.update(choices=choices, value=NO_STATE), msg


def append_args(body: str, argstr: str):
    """Append `argstr` to the end of the first training command (the line with *train*.py and its
    backslash continuations). Returns (body, ok)."""
    lines = body.split("\n")
    for i, line in enumerate(lines):
        if line.lstrip().startswith("#") or not TRAIN_RE.search(line):
            continue
        j = i
        while lines[j].rstrip().endswith("\\") and j + 1 < len(lines):
            j += 1
        if re.search(r"\s#", lines[j]):          # trailing comment: can't append safely
            lines.insert(i + 1, f"  {argstr} \\")
        else:
            lines[j] = lines[j].rstrip() + f" {argstr}"
        return "\n".join(lines), True
    return body, False


def inject_resume(body: str, state: str):
    """Append `--resume <state>` to the end of the first training command. Returns (body, ok)."""
    return append_args(body, f"--resume {shlex.quote(state)}")


def on_script_change(path):
    model, log = script_paths(path)
    model = model or (ROOT / "output")
    return (str(model),
            gr.update(choices=[("(start fresh, no resume)", NO_STATE)], value=NO_STATE), "",
            str(model), str(model / "sample"), checkpoints_md(str(model)), str(log or (ROOT / "logs")))


# --------------------------------------------------------------------------- start / stop
def start_run(path, workdir, venv, trace, resume):
    global RUN
    if RUN.running():
        gr.Warning("A run is already in progress.")
        return
    if not path:
        gr.Warning("Pick a script first.")
        return
    src = Path(path)
    cwd = Path(os.path.expanduser(workdir.strip())) if (workdir or "").strip() else ROOT
    if not cwd.is_dir():
        gr.Warning(f"Working directory not found: {cwd}")
        return
    src = Path(path)
    if not src.is_file():
        gr.Warning(f"Launch script not found: {src}")
        return
    body = src.read_text(encoding="utf-8", errors="ignore")      # always the saved file, never a stale copy

    resume = "" if resume in (None, "", NO_STATE) else resume
    if resume:
        if not Path(resume).exists():
            gr.Warning(f"State folder no longer exists: {resume}")
            return
        body, ok = inject_resume(body, resume)
        if not ok:
            gr.Warning("Couldn't find the training command (a line with *train*.py) to add --resume to.")
            return

    pre = []
    if trace:
        pre.append("set -x")
    if (venv or "").strip():
        act = Path(os.path.expanduser(venv.strip()))
        if act.is_dir():
            act = act / "bin" / "activate"
        if not act.is_file():
            gr.Warning(f"activate file not found: {act}")
            return
        pre.append(f"source {shlex.quote(str(act))}")
    if pre:
        lines = body.split("\n")
        idx = 1 if lines and lines[0].startswith("#!") else 0
        lines[idx:idx] = pre
        body = "\n".join(lines)

    tmp = src.parent / f".gui_run_{src.stem}_{int(time.time())}.sh"
    tmp.write_text(body if body.endswith("\n") else body + "\n")
    tmp.chmod(0o755)

    LOG_DIR.mkdir(exist_ok=True)
    logfile = LOG_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_{src.stem}.log"

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    venv_bin = str(Path(sys.executable).parent)   # keep the GUI's venv out of the script's PATH
    env["PATH"] = os.pathsep.join(p for p in env.get("PATH", "").split(os.pathsep) if p != venv_bin)
    env.pop("VIRTUAL_ENV", None)

    run = Run()
    run.name = src.name
    run.logfile = logfile
    run.tmp = tmp
    run.secret = read_hf_token()
    run.started = time.time()
    run.lines.append(f"$ cd {cwd} && bash {src}")
    run.lines.append(f"venv prelude: {bool((venv or '').strip())}, resume: {resume or 'no'}")
    run.lines.append(f"log file: {logfile}")
    run.proc = subprocess.Popen(
        ["bash", str(tmp)],
        cwd=str(cwd),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,                    # own process group, so Stop reaches accelerate's children
    )
    RUN = run
    threading.Thread(target=_reader, args=(run,), daemon=True).start()
    gr.Info(f"Started {src.name}")


def _signal(proc, sig):
    try:
        os.killpg(proc.pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def _escalate(proc):
    for delay, sig in ((20, signal.SIGTERM), (10, signal.SIGKILL)):
        deadline = time.time() + delay
        while time.time() < deadline:
            if proc.poll() is not None:
                return
            time.sleep(0.5)
        _signal(proc, sig)


def stop_run():
    if not RUN.running():
        gr.Info("Nothing is running.")
        return
    _signal(RUN.proc, signal.SIGINT)
    threading.Thread(target=_escalate, args=(RUN.proc,), daemon=True).start()
    gr.Info("Sent Ctrl+C. Escalates to SIGTERM after 20s and SIGKILL after 10s more if it hangs.")


def tick(seen):
    v = RUN.version
    return status_text(), (log_text() if v != seen else gr.update()), v, progress_text()


# --------------------------------------------------------------------------- monitor
def latest_images(folder, n=12):
    p = Path(os.path.expanduser(folder or ""))
    if not p.is_dir():
        return []
    files = []
    for f in p.rglob("*"):
        if f.suffix.lower() in IMG_EXT:
            try:
                files.append((f.stat().st_mtime, f))
            except OSError:
                pass
    files.sort(reverse=True)
    return [(str(f), f.name) for _, f in files[:n]]


TB = {"proc": None}


def tb_stop():
    p = TB["proc"]
    if p and p.poll() is None:
        _signal(p, signal.SIGTERM)
    TB["proc"] = None


def tb_start(logdir, port):
    tb_stop()
    logdir = os.path.expanduser(logdir or "")
    if not Path(logdir).is_dir():
        gr.Warning(f"Log dir not found: {logdir}")
        return "TensorBoard: not running"
    port = int(port)
    TB["proc"] = subprocess.Popen(
        [sys.executable, "-m", "tensorboard.main", "--logdir", logdir, "--host", "127.0.0.1", "--port", str(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
    )
    return f"TensorBoard: http://127.0.0.1:{port} (give it a few seconds to start)"


def tb_stop_ui():
    tb_stop()
    return "TensorBoard: not running"


atexit.register(tb_stop)



# --------------------------------------------------------------------------- dataset tab
def find_scan_script(explicit):
    cands = []
    if explicit:
        cands.append(Path(os.path.expanduser(explicit)))
    for d in (Path(__file__).resolve().parent, ROOT, HOME):
        for n in ("scan_lora_dataset_ny.py", "scan_lora_dataset.py"):
            cands.append(d / n)
    for c in cands:
        if c.is_file():
            return c
    return None


def load_scan_module():
    """Import the user's scan script fresh each time, so edits to PRESETS apply without a restart."""
    if SCAN_SCRIPT is None:
        raise FileNotFoundError("scan_lora_dataset_ny.py not found. Put it next to sdgui.py or pass --scan-script.")
    spec = importlib.util.spec_from_file_location("scan_lora_dataset_ny", str(SCAN_SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def list_lora_dirs():
    img = ROOT / "images"
    return sorted(p.name for p in img.iterdir() if p.is_dir() and not p.name.startswith(".")) if img.is_dir() else []


def refresh_lora_dirs():
    return gr.update(choices=list_lora_dirs())


# --------------------------------------------------------------------------- shared dataset helpers
def lora_path(name) -> Path:
    return ROOT / "images" / name


def repeat_folders(lora_dir: Path):
    """[(folder, repeats)] using the same rules as the scan script (repeats=None: not a '<N>_name'
    folder, so sd-scripts ignores it). Skips model/output/sample folders."""
    m = REPEAT_RE.match(lora_dir.name)
    if m:
        return [(lora_dir, int(m.group(1)))]
    found, loose = [], []
    for dirpath, dirnames, filenames in os.walk(lora_dir):
        p = Path(dirpath)
        mm = REPEAT_RE.match(p.name)
        if p != lora_dir and mm:
            found.append((p, int(mm.group(1))))
            dirnames[:] = []
            continue
        dirnames[:] = sorted(d for d in dirnames if d.lower() not in IGNORE_DIRS and not d.startswith("."))
        if any(Path(f).suffix.lower() in IMAGE_EXTS for f in filenames):
            loose.append((p, None))
    return found + loose


def list_images(folder: Path):
    try:
        return sorted((f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS),
                      key=lambda f: f.name.lower())
    except OSError:
        return []


def caption_file(img: Path) -> Path:
    for ext in CAPTION_EXTS:
        c = img.with_name(img.stem + ext)
        if c.is_file():
            return c
    return img.with_name(img.stem + ".txt")


def read_caption(img: Path):
    c = caption_file(img)
    if not c.is_file():
        return None
    return c.read_text(encoding="utf-8", errors="ignore").strip()


def thumb(img: Path, size: int = 384) -> str:
    """Cached JPEG thumbnail (keeps the gallery fast with 1024px PNGs)."""
    try:
        from PIL import Image, ImageOps
    except ImportError:
        return str(img)
    try:
        st = img.stat()
        key = hashlib.sha1(f"{img}|{st.st_mtime_ns}|{st.st_size}".encode()).hexdigest()[:20]
        dst = CACHE_DIR / "thumbs" / f"{key}.jpg"
        if not dst.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(img) as im:
                im = ImageOps.exif_transpose(im)
                im.thumbnail((size, size))
                if im.mode != "RGB":
                    im = im.convert("RGB")
                im.save(dst, "JPEG", quality=85)
        return str(dst)
    except Exception:
        return str(img)


def split_tags(text: str):
    return [t.strip() for t in (text or "").replace("\n", ",").split(",") if t.strip()]


# --------------------------------------------------------------------------- GPU status (rocm-smi)
_GPU = {"t": 0.0, "txt": "**GPU:** checking..."}


def _num_in(v):
    m = re.search(r"-?\d+(?:\.\d+)?", str(v))
    return float(m.group()) if m else None


def _pick(d, *needles, exclude=()):
    for k, v in d.items():
        kl = str(k).lower()
        if all(n in kl for n in needles) and not any(x in kl for x in exclude):
            return v
    return None


def parse_rocm_smi(data):
    """rocm-smi --json -> [{card, load, vram_total, vram_used, temp, junction, power}] (keys matched loosely,
    because the exact JSON key names differ between ROCm versions)."""
    cards = []
    for key in sorted(k for k in data if str(k).lower().startswith("card")):
        d = data[key]
        if not isinstance(d, dict):
            continue
        cards.append(dict(
            card=key,
            load=_num_in(_pick(d, "gpu use")),
            vram_total=_num_in(_pick(d, "vram total memory")),
            vram_used=_num_in(_pick(d, "vram total used")),
            temp=_num_in(_pick(d, "temperature", "edge")),
            junction=_num_in(_pick(d, "temperature", "junction")),
            power=_num_in(_pick(d, "power", "(w)")),
        ))
    return cards


_GPU_PEAK = {"power": 300.0}


def _bar(label, frac, text, warn=0.6, crit=0.85):
    """One colored meter: label on the left, value inside the bar. frac is 0..1."""
    frac = min(max(frac or 0.0, 0.0), 1.0)
    color = "#2e9e5b" if frac < warn else ("#d99a1f" if frac < crit else "#d64545")
    return (f'<div class="gbar" title="{label}"><div class="gfill" style="width:{frac * 100:.0f}%;background:{color}"></div>'
            f'<div class="gtxt"><span>{label}</span><b>{text}</b></div></div>')


def format_gpu(cards):
    if not cards:
        return '<div class="gmsg">GPU: no data from rocm-smi</div>'
    out = []
    for c in cards:
        bars = []
        if c["load"] is not None:
            bars.append(_bar("Load", c["load"] / 100, f"{c['load']:.0f}%"))
        if c["vram_total"]:
            div = 1024 ** 3 if c["vram_total"] > 1e6 else 1024.0      # bytes or MB
            used, tot = (c["vram_used"] or 0) / div, c["vram_total"] / div
            bars.append(_bar("VRAM", used / tot if tot else 0, f"{used:.1f} / {tot:.1f} GiB", 0.75, 0.92))
        t = c["junction"] if c["junction"] is not None else c["temp"]
        if t is not None:
            txt = f"{c['temp']:.0f}°C" if c["temp"] is not None else f"{t:.0f}°C"
            if c["junction"] is not None:
                txt += f" · jct {c['junction']:.0f}°C"
            bars.append(_bar("Temp", t / 110, txt, 0.64, 0.8))
        if c["power"] is not None:
            _GPU_PEAK["power"] = max(_GPU_PEAK["power"], c["power"])
            bars.append(_bar("Power", c["power"] / _GPU_PEAK["power"], f"{c['power']:.0f} W", 0.7, 0.92))
        name = str(c["card"]).replace("card", "GPU ")
        out.append(f'<div class="grow"><span class="gname">{name}</span>{"".join(bars)}</div>')
    return "".join(out)


def gpu_status_text():
    now = time.time()
    if now - _GPU["t"] < 2.0:
        return _GPU["txt"]
    exe = shutil.which("rocm-smi") or next(iter(sorted(glob.glob("/opt/rocm*/bin/rocm-smi"))), None)
    if not exe:
        txt = '<div class="gmsg">GPU: rocm-smi not found</div>'
    else:
        try:
            r = subprocess.run([exe, "--showuse", "--showmeminfo", "vram", "--showtemp", "--showpower", "--json"],
                               capture_output=True, text=True, timeout=6)
            out = r.stdout
            txt = format_gpu(parse_rocm_smi(json.loads(out[out.index("{"): out.rindex("}") + 1])))
        except subprocess.TimeoutExpired:
            txt = '<div class="gmsg">GPU: rocm-smi timed out</div>'
        except Exception:
            txt = '<div class="gmsg">GPU: could not read rocm-smi output</div>'
    _GPU["t"], _GPU["txt"] = now, txt
    return txt


# --------------------------------------------------------------------------- LoRA types (architectures)
VIEWS_OK = (20, 100)       # rough "views per image" window that works for character LoRAs
_BIG_ENV = {"PYTORCH_ROCM_ALLOC_CONF": "max_split_size_mb:512,garbage_collection_threshold:0.8"}
_PH = "/PATH/TO/"          # placeholder for model files this tool cannot know

# Architectures sd-scripts can train LoRAs for. Your own PRESETS in scan_lora_dataset_ny.py (sdxl, anima,
# illustrious) always win over these. These are TEMPLATES written from the sd-scripts documentation: set the
# model paths, and use "Check flags" to confirm your sd-scripts version accepts every option.
EXTRA_PRESETS = {
    "sd15": {
        "label": "SD 1.5 (for SD 2.x add --v2)", "suffix": "SD15", "max_batch": 8,
        "steps": 1800, "epochs": 15, "views": 60, "targets": (1300, 1800, 2400),
        "lr": "1e-4", "optimizer": "AdamW8bit", "scheduler": "cosine_with_restarts", "warmup": 80,
        "script": "train_network.py", "sample_args": "--w 512 --h 512 --s 28 --l 7",
        "prompts_name": "SD15prompts.txt", "script_name": "{name}SD15training.sh",
        "model_args": [("--pretrained_model_name_or_path", _PH + "sd15-checkpoint.safetensors")],
        "flags": [
            "--resolution 512,512",
            "--enable_bucket --min_bucket_reso 256 --max_bucket_reso 1024 --bucket_reso_steps 64",
            "--network_module networks.lora", "--network_dim 32 --network_alpha 16",
            "--mixed_precision bf16 --save_precision bf16", "--sdpa",
            "--cache_latents --cache_latents_to_disk",
            "--text_encoder_lr 5e-5", "--clip_skip 2",
            "--shuffle_caption --keep_tokens 1", "--max_grad_norm 1.0",
            "--min_snr_gamma 5", "--noise_offset 0.0357",
        ],
        "sample_extra": "--sample_sampler euler_a",
        "notes": ["--clip_skip 2 is right for anime-derived checkpoints; use 1 for photographic SD 1.5 models.",
                  "Add --network_train_unet_only if you want to freeze the text encoder (less overfit to caption wording)."],
        "template": True,
    },
    "sd3": {
        "label": "SD3 / SD3.5", "suffix": "SD3", "max_batch": 2,
        "steps": 1600, "epochs": 10, "views": 40, "targets": (1200, 1600, 2200),
        "lr": "1e-4", "optimizer": "AdamW8bit", "scheduler": "cosine", "warmup": 80,
        "script": "sd3_train_network.py", "sample_args": "--w 1024 --h 1024 --s 28 --l 4.5",
        "prompts_name": "SD3prompts.txt", "script_name": "{name}SD3training.sh",
        "model_args": [("--pretrained_model_name_or_path", _PH + "sd3.5_medium.safetensors"),
                       ("--clip_l", _PH + "clip_l.safetensors"), ("--clip_g", _PH + "clip_g.safetensors"),
                       ("--t5xxl", _PH + "t5xxl_fp16.safetensors")],
        "flags": [
            "--resolution 1024,1024",
            "--enable_bucket --min_bucket_reso 512 --max_bucket_reso 1536 --bucket_reso_steps 64",
            "--network_module networks.lora_sd3", "--network_dim 16 --network_alpha 16",
            "--mixed_precision bf16 --save_precision bf16", "--sdpa --gradient_checkpointing",
            "--cache_latents --cache_latents_to_disk",
            "--cache_text_encoder_outputs --cache_text_encoder_outputs_to_disk",
            "--network_train_unet_only", "--max_grad_norm 1.0",
        ],
        "sample_extra": "", "env": _BIG_ENV,
        "notes": ["SD3.5 Large (8B) will not fit 16 GB as is: add --fp8_base and --blocks_to_swap N.",
                  "SD3.5 Medium (2.5B) fits comfortably with gradient checkpointing."],
        "template": True,
    },
    "flux": {
        "label": "FLUX.1 (dev / schnell)", "suffix": "FLUX", "max_batch": 1,
        "steps": 1600, "epochs": 10, "views": 40, "targets": (1200, 1600, 2200),
        "lr": "1e-4", "optimizer": "AdamW8bit", "scheduler": "cosine", "warmup": 80,
        "script": "flux_train_network.py", "sample_args": "--w 1024 --h 1024 --s 20 --g 3.5",
        "prompts_name": "FLUXprompts.txt", "script_name": "{name}FLUXtraining.sh",
        "model_args": [("--pretrained_model_name_or_path", _PH + "flux1-dev.safetensors"),
                       ("--clip_l", _PH + "clip_l.safetensors"), ("--t5xxl", _PH + "t5xxl_fp16.safetensors"),
                       ("--ae", _PH + "ae.safetensors")],
        "flags": [
            "--resolution 1024,1024",
            "--enable_bucket --min_bucket_reso 512 --max_bucket_reso 1536 --bucket_reso_steps 64",
            "--network_module networks.lora_flux", "--network_dim 16 --network_alpha 16",
            "--mixed_precision bf16 --save_precision bf16", "--sdpa --gradient_checkpointing",
            "--cache_latents --cache_latents_to_disk",
            "--cache_text_encoder_outputs --cache_text_encoder_outputs_to_disk",
            "--fp8_base", "--network_train_unet_only",
            "--timestep_sampling shift --discrete_flow_shift 3.1582 --model_prediction_type raw --guidance_scale 1.0",
        ],
        "sample_extra": "", "env": _BIG_ENV, "workers": 2,
        "notes": ["FLUX.1 is a 12B model: --fp8_base is what makes it fit 16 GB. If you still run out of memory, "
                  "add --blocks_to_swap 8 (raise as needed).",
                  "Captions can be natural language instead of tags; the trigger word still goes first."],
        "template": True,
    },
    "lumina": {
        "label": "Lumina Image 2.0", "suffix": "LUM", "max_batch": 2,
        "steps": 1600, "epochs": 10, "views": 40, "targets": (1200, 1600, 2200),
        "lr": "1e-4", "optimizer": "AdamW8bit", "scheduler": "cosine", "warmup": 80,
        "script": "lumina_train_network.py", "sample_args": "--w 1024 --h 1024 --s 36 --l 4.0",
        "prompts_name": "LUMprompts.txt", "script_name": "{name}LUMtraining.sh",
        "model_args": [("--pretrained_model_name_or_path", _PH + "lumina_2_model_bf16.safetensors"),
                       ("--gemma2", _PH + "gemma_2_2b_fp16.safetensors"), ("--ae", _PH + "ae.safetensors")],
        "flags": [
            "--resolution 1024,1024",
            "--enable_bucket --min_bucket_reso 512 --max_bucket_reso 1536 --bucket_reso_steps 64",
            "--network_module networks.lora_lumina", "--network_dim 16 --network_alpha 16",
            "--mixed_precision bf16 --save_precision bf16", "--sdpa --gradient_checkpointing",
            "--cache_latents --cache_latents_to_disk",
            "--cache_text_encoder_outputs --cache_text_encoder_outputs_to_disk",
            "--network_train_unet_only",
            "--timestep_sampling nextdit_shift --discrete_flow_shift 6.0 --model_prediction_type raw",
        ],
        "sample_extra": "", "env": _BIG_ENV,
        "notes": ["Lumina 2.0 is a 2B model and fits 16 GB at batch 2; it uses the Gemma-2 2B text encoder."],
        "template": True,
    },
    "hunyuan": {
        "label": "HunyuanImage-2.1", "suffix": "HYI", "max_batch": 1,
        "steps": 1500, "epochs": 10, "views": 35, "targets": (1100, 1500, 2000),
        "lr": "1e-4", "optimizer": "AdamW8bit", "scheduler": "cosine", "warmup": 80,
        "script": "hunyuan_image_train_network.py", "sample_args": "--w 2048 --h 2048 --s 50 --g 3.5",
        "prompts_name": "HYIprompts.txt", "script_name": "{name}HYItraining.sh",
        "model_args": [("--pretrained_model_name_or_path", _PH + "hunyuanimage2.1.safetensors"),
                       ("--text_encoder", _PH + "qwen_2.5_vl_7b.safetensors"),
                       ("--byt5", _PH + "byt5_small_glyphxl_fp16.safetensors"),
                       ("--vae", _PH + "hunyuan_image_2.1_vae_fp16.safetensors")],
        "flags": [
            "--resolution 2048,2048",
            "--enable_bucket --min_bucket_reso 1024 --max_bucket_reso 2560 --bucket_reso_steps 64",
            "--network_module networks.lora_hunyuan_image", "--network_dim 16 --network_alpha 16",
            "--mixed_precision bf16 --save_precision bf16", "--sdpa --gradient_checkpointing",
            "--cache_latents --cache_latents_to_disk",
            "--cache_text_encoder_outputs --cache_text_encoder_outputs_to_disk",
            "--fp8_base --fp8_scaled --blocks_to_swap 20", "--network_train_unet_only",
            "--timestep_sampling shift --discrete_flow_shift 5.0",
        ],
        "sample_extra": "", "env": _BIG_ENV, "workers": 2,
        "notes": ["A 17B-parameter model at 2K resolution: on 16 GB it only works with fp8 plus heavy block "
                  "swapping, and will be slow. 24 GB+ is the comfortable minimum.",
                  "This is the least-tested template here: run 'Check flags' before training."],
        "template": True,
    },
}


def merged_presets():
    """(scan module or None, {key: preset}). Your scan script's PRESETS win; extras fill in the rest."""
    try:
        sc = load_scan_module()
    except Exception:
        return None, dict(EXTRA_PRESETS)
    merged = dict(sc.PRESETS)
    for k, v in EXTRA_PRESETS.items():
        merged.setdefault(k, v)
    return sc, merged


def type_choices():
    _, presets = merged_presets()
    return [(v["label"] + ("  [template]" if v.get("template") else ""), k) for k, v in presets.items()]


# --------------------------------------------------------------------------- create launch script tab
NEW = "__new__"

FORM_GROUPS = [
    ("Training", [
        ("max_train_epochs", "Epochs"), ("train_batch_size", "Batch size"),
        ("gradient_accumulation_steps", "Grad accum"), ("learning_rate", "Learning rate"),
        ("optimizer_type", "Optimizer"), ("lr_scheduler", "LR scheduler"),
        ("lr_warmup_steps", "Warmup steps"), ("network_dim", "Network dim"),
        ("network_alpha", "Network alpha"), ("resolution", "Resolution"),
        ("seed", "Seed"), ("mixed_precision", "Mixed precision")]),
    ("Saving and samples", [
        ("output_name", "Output name"), ("save_every_n_epochs", "Save every N epochs"),
        ("sample_every_n_epochs", "Sample every N epochs"), ("max_data_loader_n_workers", "Data loader workers"),
        ("logging_dir", "Logging dir (TensorBoard)")]),
    ("Advanced", [
        ("network_module", "Network module"), ("attn_mode", "Attn mode (Anima)"),
        ("noise_offset", "Noise offset"), ("min_snr_gamma", "Min SNR gamma"),
        ("max_grad_norm", "Max grad norm"), ("keep_tokens", "Keep tokens"),
        ("clip_skip", "Clip skip"), ("timestep_sampling", "Timestep sampling"),
        ("discrete_flow_shift", "Discrete flow shift"), ("blocks_to_swap", "Blocks to swap")]),
    ("Model files", [
        ("pretrained_model_name_or_path", "Base model"), ("vae", "VAE"), ("clip_l", "CLIP-L"),
        ("clip_g", "CLIP-G"), ("t5xxl", "T5-XXL"), ("ae", "AE (Flux/Lumina)"), ("gemma2", "Gemma2"),
        ("qwen3", "Qwen3 (Anima)"), ("text_encoder", "Text encoder"), ("byt5", "ByT5")]),
]
FORM_FIELDS = [f for _, fs in FORM_GROUPS for f in fs]
MODEL_FIELDS = {name for name, _ in FORM_GROUPS[3][1]}


def _arg_re(name: str):
    return re.compile(rf"((?<![\w-])--{name}(?:=|[ \t]+))(\"[^\"]*\"|'[^']*'|[^\s\\]+)")


def read_form(text: str):
    vals = []
    for name, _ in FORM_FIELDS:
        m = _arg_re(name).search(text)
        v = m.group(2) if m else ""
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        vals.append(v)
    return vals


def form_dict(text: str) -> dict:
    return {name: v for (name, _), v in zip(FORM_FIELDS, read_form(text))}


def apply_form(text: str, vals):
    """Write non-empty form values into the script text. Existing args are changed in place; missing ones
    are appended to the training command. An empty field leaves the script alone."""
    changes, failed = [], []
    for (name, label), v in zip(FORM_FIELDS, vals):
        v = (v or "").strip()
        if not v:
            continue
        m = _arg_re(name).search(text)
        if m:
            old = m.group(2)
            q = old[0] if old[0] in "\"'" else ""
            oldv = old[1:-1] if q else old
            if oldv == v:
                continue
            tok = f"{q}{v}{q}" if q else (f'"{v}"' if re.search(r"\s", v) else v)
            text = text[:m.start(2)] + tok + text[m.end(2):]
            changes.append(f"{label}: {oldv} -> {v}")
        else:
            tok = f'"{v}"' if re.search(r"\s", v) else v
            text, ok = append_args(text, f"--{name} {tok}")
            if ok:
                changes.append(f"{label}: added {v}")
            else:
                failed.append(label)
    return text, changes, failed


def _num(v):
    try:
        return int(v) if v and int(v) > 0 else None
    except (TypeError, ValueError):
        return None


def training_command(text: str):
    """(trainer script name, text after it) for the first training command, or (None, "")."""
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.lstrip().startswith("#") or not TRAIN_RE.search(line):
            continue
        j = i
        while lines[j].rstrip().endswith("\\") and j + 1 < len(lines):
            j += 1
        cmd = "\n".join(lines[i:j + 1])
        m = TRAIN_RE.search(cmd)
        return m.group(0), cmd[m.end():]
    return None, ""


def infer_type(text, presets, fallback):
    script, _ = training_command(text)
    if not script:
        return fallback
    cands = [k for k, v in presets.items() if v.get("script") == script]
    if not cands:
        return fallback
    if len(cands) == 1:
        return cands[0]
    hay = form_dict(text).get("output_name", "") + " " + text[:600]
    for k in sorted(cands, key=lambda k: -len(presets[k].get("suffix", ""))):
        suf = presets[k].get("suffix", "")
        if suf and suf in hay:
            return k
    return cands[0]


def prompts_path_from_text(text: str, lora: Path):
    m = re.search(r"^PROMPTS_FILE=(?:\"([^\"]*)\"|'([^']*)'|(\S+))", text, re.M)
    if not m:
        return None
    raw = next(g for g in m.groups() if g is not None)
    raw = raw.replace("${SCRIPT_DIR}", str(lora)).replace("$SCRIPT_DIR", str(lora)).replace("${HOME}", str(HOME))
    return Path(os.path.expanduser(raw))


def launch_scripts(ds):
    d = lora_path(ds) if ds else None
    if not d or not d.is_dir():
        return []
    return sorted((p for p in d.glob("*.sh") if not p.name.startswith(".gui_run_")), key=lambda p: p.name.lower())


def cs_script_choices(ds):
    return [("+ New launch script", NEW)] + [(p.name, str(p)) for p in launch_scripts(ds)]


TYPE_TAG = {"sdxl": "SDXL", "anima": "ANIMA", "illustrious": "Illustrious", "sd15": "SD15", "sd3": "SD3",
            "flux": "Flux1", "lumina": "Lumina", "hunyuan": "HunyuanImage"}


def new_script_filename(lora: Path, tkey: str) -> str:
    """<folder><TYPE>_lora.sh, e.g. bubbaANIMA_lora.sh, bubbaSDXL_lora.sh, bubbaFlux1_lora.sh."""
    tag = TYPE_TAG.get(tkey) or re.sub(r"[^A-Za-z0-9]+", "", str(tkey)).upper()
    return unique_script_name(lora, f"{lora.name}{tag}_lora.sh")


def unique_script_name(lora: Path, name: str) -> str:
    if not (lora / name).exists():
        return name
    stem, n = name[:-3], 2
    while (lora / f"{stem}_{n}.sh").exists():
        n += 1
    return f"{stem}_{n}.sh"


# ---- dataset numbers
def dataset_stats(ds):
    lora = lora_path(ds)
    rows, loose = [], []
    for f, r in repeat_folders(lora):
        imgs = list_images(f)
        if r is None:
            loose.append((f, len(imgs)))
            continue
        rows.append(dict(folder=f, repeats=r, n=len(imgs), captioned=sum(1 for im in imgs if read_caption(im))))
    return dict(lora=lora, rows=rows, loose=loose, total=sum(r["n"] for r in rows),
                captioned=sum(r["captioned"] for r in rows))


def with_virtual_row(stats):
    """A dataset whose images sit directly in the LoRA folder has no <N>_name folder yet. Plan it as if
    the folder existed: one virtual row, which Save creates (and moves the images into)."""
    root_imgs = list_images(stats["lora"])
    if stats["rows"] or not root_imgs:
        return stats
    caps = sum(1 for im in root_imgs if read_caption(im))
    row = dict(folder=stats["lora"], repeats=1, n=len(root_imgs), captioned=caps, virtual=True)
    return {**stats, "rows": [row], "loose": [(f, n) for f, n in stats["loose"] if f != stats["lora"]],
            "total": len(root_imgs), "captioned": caps}


def recommended_repeats(ds, tkey, steps_over=0):
    stats = with_virtual_row(dataset_stats(ds))
    n = stats["total"]
    try:
        sc, presets = merged_presets()
        if not n or sc is None or tkey not in presets:
            return 10
        cfg = dict(presets[tkey])
        if _num(steps_over):
            cfg["steps"] = _num(steps_over)
        bs, ga, _ = sc.suggest_settings(n, cfg)
        return sc.suggest_repeats(n, bs, ga, cfg["epochs"], cfg["steps"])
    except Exception:
        return 10


def build_prompts(sc, stats, cfg):
    """(path, text, is_new): the type's sample prompts file for this dataset (existing file, else generated)."""
    lora = stats["lora"]
    path = lora / cfg["prompts_name"]
    if path.is_file():
        return path, path.read_text(encoding="utf-8", errors="ignore"), False
    caps = sc.read_captions([(r["folder"], r["repeats"]) for r in stats["rows"]]) if stats["rows"] else []
    return path, sc.build_prompts_text(caps, lora.name.lower(), lora.name, cfg)[0], True


# ---- sample images on/off, logging, folder creation (all work on the script text)
SAMPLE_ARG_RES = [r"--sample_every_n_epochs[ =]\S+", r"--sample_every_n_steps[ =]\S+",
                  r"--sample_prompts[ =](?:\"[^\"]*\"|'[^']*'|\S+)", r"--sample_sampler[ =]\S+", r"--sample_at_first"]


def has_samples(text) -> bool:
    return re.search(r"(?<![\w-])--sample_prompts(?![\w-])", text or "") is not None


def _command_span(lines):
    for i, line in enumerate(lines):
        if line.lstrip().startswith("#") or not TRAIN_RE.search(line):
            continue
        j = i
        while lines[j].rstrip().endswith("\\") and j + 1 < len(lines):
            j += 1
        return i, j
    return None


def strip_samples(text: str) -> str:
    """Remove the sample-image arguments from the training command and tidy the line continuations."""
    lines = text.split("\n")
    span = _command_span(lines)
    if not span:
        return text
    i, j = span
    block = []
    for k in range(i, j + 1):
        ln = new = lines[k]
        for pat in SAMPLE_ARG_RES:
            new = re.sub(r"(?<![\w-])" + pat, "", new)
        if new != ln:
            new = re.sub(r"(?<=\S)[ \t]{2,}(?=\S)", " ", new).rstrip()
            if ln[:1] in " \t":
                new = "  " + new.lstrip()
        if re.fullmatch(r"\s*\\?\s*", new):
            continue
        block.append(new)
    if block:
        block[-1] = re.sub(r"[ \t]*\\\s*$", "", block[-1])
    lines[i:j + 1] = block
    return "\n".join(lines)


def add_samples(text: str, cfg: dict, lora: Path):
    """Add per-epoch sample images (and the PROMPTS_FILE variable) to a script. Returns (text, ok)."""
    if has_samples(text):
        return text, True
    lines = text.split("\n")
    span = _command_span(lines)
    if not span:
        return text, False
    if not re.search(r"^PROMPTS_FILE=", text, re.M):
        base = "${SCRIPT_DIR}" if re.search(r"^SCRIPT_DIR=", text, re.M) else str(lora)
        lines.insert(span[0], f'PROMPTS_FILE="{base}/{cfg["prompts_name"]}"')
        text = "\n".join(lines)
    args = '--sample_every_n_epochs 1 --sample_prompts "${PROMPTS_FILE}"'
    if cfg.get("sample_extra"):
        args += " " + cfg["sample_extra"]
    return append_args(text, args)


def add_logging(text: str) -> str:
    """New scripts write TensorBoard logs to <train dir>/log (sd-scripts turns on TensorBoard by itself
    when --logging_dir is given)."""
    if _has(text, "--logging_dir"):
        return text
    if not re.search(r"^LOG_DIR=", text, re.M):
        m = re.search(r"^OUTPUT_DIR=.*$", text, re.M)
        if not m:
            return text
        text = text[:m.end()] + '\nLOG_DIR="${TRAIN_DIR}/log"' + text[m.end():]
    return append_args(text, '--logging_dir "${LOG_DIR}"')[0]


def _inside(p: Path, lora: Path) -> bool:
    try:
        p.resolve().relative_to(lora.resolve())
        return True
    except ValueError:
        return False


def _rel(p: Path, lora: Path) -> str:
    try:
        return str(p.resolve().relative_to(lora.resolve())) + "/"
    except ValueError:
        return str(p)


def ensure_folders(ds, text, plan, tkey):
    """Create what a training run needs inside the LoRA folder: the <N>_name repeat folder (if there is none),
    and the model and log folders the script writes to. Images lying directly in the LoRA folder are moved
    into a newly created repeat folder. Returns a list of what was done."""
    lora = lora_path(ds)
    notes = []
    v = script_vars(text or "", lora)

    def resolve(name, default):
        raw = v.get(name)
        val = _expand(raw, v) if raw else ""
        return Path(val) if val and "$" not in val and "`" not in val else default

    train_dir = resolve("TRAIN_DIR", lora)
    v["TRAIN_DIR"] = str(train_dir)
    for label, d in (("model", resolve("OUTPUT_DIR", train_dir / "model")), ("log", resolve("LOG_DIR", train_dir / "log"))):
        if not _inside(d, lora):
            notes.append(f"`{d}` is outside the LoRA folder, so it was not created")
        elif not d.is_dir():
            d.mkdir(parents=True, exist_ok=True)
            notes.append(f"created `{_rel(d, lora)}`")
    stats = dataset_stats(ds)
    if not stats["rows"]:
        name = (plan or {}).get("create") or f"{recommended_repeats(ds, tkey)}_{lora.name.lower()}"
        dest = train_dir / name
        if not _inside(dest, lora):
            notes.append(f"`{dest}` is outside the LoRA folder, so it was not created")
        elif not dest.exists():
            dest.mkdir(parents=True)
            notes.append(f"created repeat folder `{_rel(dest, lora)}`")
            if train_dir.resolve() == lora.resolve():
                moved = 0
                for im in list_images(lora):
                    for src in [im] + [im.with_name(im.stem + e) for e in CAPTION_EXTS]:
                        if src.is_file():
                            shutil.move(str(src), str(dest / src.name))
                            moved += 1
                if moved:
                    notes.append(f"moved {moved} files (images and captions) from the LoRA folder into `{name}/`")
            if not list_images(dest):
                notes.append(f"`{name}/` is empty: copy your images and .txt captions into it")
        left = [f for f, n in dataset_stats(ds)["loose"] if n]          # re-read: images may just have been moved
        if left:
            notes.append("images in " + ", ".join(f"`{_rel(f, lora)}`" for f in left)
                         + " have no repeat count: use the Repeat folders panel to rename that folder")
    return notes


def recommend_numbers(sc, cfg, stats, use_sug_rep=True):
    """Batch / accumulation / repeats / epochs recommended for this type and dataset (same maths as your scan script)."""
    total, rows = stats["total"], stats["rows"]
    bs, ga, _ = sc.suggest_settings(total, cfg)
    eff = bs * ga
    repeats = [r["repeats"] for r in rows]
    rename, want, create = None, None, None
    if len(rows) == 1:
        want = sc.suggest_repeats(total, bs, ga, cfg["epochs"], cfg["steps"])
        if rows[0].get("virtual"):
            repeats = [want]
            create = f"{want}_{rows[0]['folder'].name.lower()}"
        elif want != repeats[0] and use_sug_rep:
            old = rows[0]["folder"]
            rename = f'mv "{old}" "{old.parent / REPEAT_RE.sub(f"{want}_", old.name, count=1)}"'
            repeats = [want]
    samples = sum(r["n"] * rep for r, rep in zip(rows, repeats))
    spe = max(1, math.ceil(samples / eff))
    targets = cfg["targets"]
    main_target = targets[len(targets) // 2]
    plan = {t: max(1, round(t / spe)) for t in targets}
    epochs = plan[main_target]
    return dict(bs=bs, ga=ga, eff=eff, repeats=repeats, want_repeats=want, rename=rename, create=create, samples=samples,
                spe=spe, plan=plan, main_target=main_target, epochs=epochs, total_steps=epochs * spe)


def make_plan(ds, tkey, steps_over=0, use_sug_rep=True, hf=True, samples=True):
    sc, presets = merged_presets()
    if sc is None:
        raise FileNotFoundError("scan_lora_dataset_ny.py not found: put it next to sdgui.py or pass --scan-script.")
    if tkey not in presets:
        raise ValueError(f"Unknown LoRA type: {tkey}")
    cfg = dict(presets[tkey])
    ov = settings_models(tkey)                       # model files chosen in the Settings tab win over the preset
    if ov:
        have = {f.lstrip("-") for f, _ in cfg["model_args"]}
        cfg["model_args"] = [(f, ov.get(f.lstrip("-"), pth)) for f, pth in cfg["model_args"]]
        if tkey in OPTIONAL_VAE_TYPES and "vae" not in have and ov.get("vae"):
            cfg["model_args"].append(("--vae", ov["vae"]))
    so = _num(steps_over)
    if so:
        cfg["steps"], cfg["targets"] = so, (round(so * 0.7), so, round(so * 1.3))
    stats = with_virtual_row(dataset_stats(ds))
    virtual = any(r.get("virtual") for r in stats["rows"])
    if not stats["rows"]:
        if stats["loose"]:
            raise ValueError("The images are not in a `<N>_name` repeat folder, so sd-scripts would ignore them. "
                             "Rename their folder in the **Repeat folders** panel below (for example to `4_name`).")
        raise ValueError("No images found in this dataset yet. Copy your images (and .txt captions) into the folder, "
                         "or press **Create missing folders now** in the Repeat folders panel to set up the folders first.")
    if not stats["total"]:
        raise ValueError("Repeat folders found, but no images inside them.")
    rec = recommend_numbers(sc, cfg, stats, use_sug_rep)
    lora = stats["lora"]
    train_dir = lora if virtual else sorted({r["folder"].parent for r in stats["rows"]})[0]
    if samples:
        prompts_path, prompts_text, prompts_new = build_prompts(sc, stats, cfg)
    else:
        prompts_path, prompts_text, prompts_new = lora / cfg["prompts_name"], "", False
    base_name = lora.name + cfg["suffix"]
    version = sc.next_version(train_dir / "model", base_name)
    out_name = f"{base_name}V{version}"
    run_cfg = {**cfg, "suffix": f"{cfg['suffix']}V{version}"}
    cmd = sc.build_command(run_cfg, "${CHARACTER_NAME}", "${TRAIN_DIR}", "${OUTPUT_DIR}", rec["bs"], rec["ga"],
                           rec["epochs"], cfg.get("save_every", 1), cfg["lr"], "${PROMPTS_FILE}" if samples else None)
    if not cmd:
        raise ValueError("This LoRA type has no launch template (no 'script' in its preset).")
    plan_line = (f"Output {out_name}. Plan: {stats['total']} images x {'/'.join(str(x) for x in rec['repeats'])} "
                 f"repeats, batch {rec['bs']} x accum {rec['ga']}, ~{rec['spe']} steps/epoch, {rec['epochs']} epochs "
                 f"= ~{rec['total_steps']} steps (target {rec['main_target']})")
    text = sc.render_training_script(sc.under(ROOT, HOME, "${HOME}"), sc.under(train_dir, lora, "${SCRIPT_DIR}"),
                                     sc.under(prompts_path, lora, "${SCRIPT_DIR}") if samples else None, plan_line, cmd,
                                     {**sc.ROCM_ENV, **cfg.get("env", {}), **({GFX_VAR: get_gfx()} if get_gfx() else {})})
    text = add_logging(text)
    if hf:
        text, _ = fill_hf(text, force=True)
    return dict(cfg=cfg, stats=stats, rec=rec, script_text=text, prompts_path=prompts_path,
                prompts_text=prompts_text, prompts_new=prompts_new, out_name=out_name,
                filename=new_script_filename(lora, tkey))


# ---- explanation
OPT_WHY = {
    "adamw8bit": "AdamW8bit keeps optimizer state in 8 bit (about a quarter of AdamW's memory) with practically the same quality. That is why it is the usual choice for LoRA on 16 GB.",
    "adamw": "AdamW is the reference optimizer. It needs more VRAM than the 8-bit version for the same result.",
    "adafactor": "Adafactor needs very little optimizer memory. With scale_parameter=False relative_step=False it behaves like a fixed-LR optimizer.",
    "prodigy": "Prodigy adapts the learning rate by itself, so the LR should stay at 1.0. It removes LR guesswork at the cost of some memory.",
    "lion": "Lion is memory-light but wants a 3-10x lower LR than AdamW.",
}
SCHED_WHY = {
    "cosine": "cosine decays the LR smoothly towards 0, so the last epochs refine what was learned instead of overwriting it.",
    "cosine_with_restarts": "cosine_with_restarts decays like cosine but restarts a few times, which can shake the run out of a weak solution. The last cycle still ends near 0.",
    "constant": "constant keeps the LR fixed: predictable, but later epochs can overshoot.",
    "constant_with_warmup": "constant_with_warmup holds the LR fixed after the warmup: predictable, but later epochs can overshoot.",
    "linear": "linear decays the LR in a straight line to 0.",
    "polynomial": "polynomial decays the LR along a curve to 0.",
}
FLAG_WHY = [
    ("--gradient_checkpointing", "Gradient checkpointing recomputes activations instead of storing them: roughly 20-30% slower, but it is what keeps this model inside 16 GB."),
    ("--cache_latents", "Images are encoded by the VAE once and cached (to disk with --cache_latents_to_disk), so the VAE does not run every step. Faster and lighter; random crop/flip augmentation is off."),
    ("--cache_text_encoder_outputs", "Text-encoder outputs are cached, so the text encoder is not held in VRAM while training. Caption shuffling is therefore off and tag order stays as written."),
    ("--network_train_unet_only", "Only the image model is trained (text encoder frozen): less VRAM, and less overfitting to caption wording."),
    ("--fp8_base", "The base model is held in 8-bit float and upcast layer by layer, roughly halving its VRAM so large models fit 16 GB. The quality cost is small for LoRA training."),
    ("--blocks_to_swap", "Transformer blocks are swapped between GPU and system RAM to save VRAM, at a speed cost. Raise the number if you still run out of memory."),
    ("--sdpa", "PyTorch scaled-dot-product attention is the fast attention path on ROCm (xformers is not available there)."),
    ("--shuffle_caption", "Tags are shuffled every step so the model does not memorize their order. --keep_tokens pins the first tag (the trigger word)."),
    ("--min_snr_gamma", "Min-SNR weighting (gamma 5) balances the loss across noise levels, which speeds up convergence and reduces blotchy artifacts."),
    ("--noise_offset", "A small noise offset lets the model learn very dark and very bright images. Keep it small (0.05 or less) or colors drift."),
    ("--save_state", "Training state is saved with the checkpoints, so a crashed run can be resumed from the Train tab."),
]


def _has(text, flag):
    return re.search(rf"(?<![\w-]){re.escape(flag)}(?![\w-])", text) is not None


def explain_md(ds, tkey, cfg, stats, text, rec, existing, planned_repeats=None, rename=None, create=None):
    f = form_dict(text)

    def I(name, default=None):
        try:
            return int(float(str(f.get(name, "")).strip()))
        except ValueError:
            return default

    rows, imgs = stats["rows"], stats["total"]
    reps = planned_repeats or [r["repeats"] for r in rows]
    samples = sum(r["n"] * rep for r, rep in zip(rows, reps))
    bs, ga = max(1, I("train_batch_size", 1) or 1), max(1, I("gradient_accumulation_steps", 1) or 1)
    ep = I("max_train_epochs")
    eff = bs * ga
    spe = max(1, math.ceil(samples / eff))
    total = spe * ep if ep else None
    views = total * eff / imgs if total else None
    lo, hi = cfg["targets"][0], cfg["targets"][-1]
    rep_txt = " / ".join(str(x) for x in sorted(set(reps)))

    L = []
    L.append(f"### {'Your launch script compared with the recommendation' if existing else 'Recommended plan'}: {cfg['label']}")
    miss = imgs - stats["captioned"]
    L.append(f"**Dataset `{ds}`:** {imgs} images in {len(rows)} repeat folder(s), {stats['captioned']} with captions"
             + (f", **{miss} without**" if miss else "") + ".")
    L += ["", "**How the step count is calculated**", "", "| Quantity | Calculation | Result |", "|---|---|---|",
          f"| Images | counted in the repeat folders | {imgs} |",
          f"| Samples per epoch | images × repeats ({rep_txt}) | {samples} |",
          f"| Effective batch | batch size × accumulation = {bs} × {ga} | {eff} |",
          f"| Steps per epoch | ceil({samples} / {eff}) | {spe} |"]
    if total:
        L += [f"| Total steps | {spe} steps × {ep} epochs | **{total}** |",
              f"| Views per image | total steps × effective batch / images | {views:.0f} |"]
    L += ["", f"**Recommended for {cfg['label']}:** about {lo}-{hi} total steps (aim for {cfg['steps']}), roughly "
          f"{cfg['views']} views per image ({VIEWS_OK[0]}-{VIEWS_OK[1]} is workable), spread over about "
          f"{cfg['epochs']} epochs. Steps per epoch are approximate: aspect-ratio bucketing can add a few, so "
          "check the count sd-scripts prints at startup."]

    if existing and rec:
        cf = form_dict(" ".join(cfg.get("flags", [])))
        L += ["", "| | Your script | Recommended |", "|---|---|---|",
              f"| Batch × accumulation | {bs} × {ga} | {rec['bs']} × {rec['ga']} |",
              f"| Repeats | {rep_txt} | {'/'.join(str(x) for x in rec['repeats'])} |",
              f"| Epochs | {ep if ep else '?'} | {rec['epochs']} |",
              f"| Total steps | {total if total else '?'} | ~{rec['total_steps']} |",
              f"| Learning rate / optimizer | {f.get('learning_rate') or '?'} / {f.get('optimizer_type') or '?'} | {cfg['lr']} / {cfg['optimizer']} |",
              f"| Network dim / alpha | {f.get('network_dim') or '?'} / {f.get('network_alpha') or '?'} | {cf.get('network_dim') or '?'} / {cf.get('network_alpha') or '?'} |"]

    # why these settings
    W = []
    if ga > 1:
        W.append(f"**Batch {bs} × accumulation {ga} = {eff}.** A larger effective batch gives smoother gradients. "
                 f"The batch size alone is capped at {cfg['max_batch']} for this architecture on 16 GB, so gradient "
                 "accumulation supplies the rest without extra VRAM.")
    else:
        W.append(f"**Batch {bs}, no accumulation.** At this dataset size one batch of {bs} already gives about the "
                 f"{cfg['views']} views per image the plan aims for, so no accumulation is needed.")
    save_every = I("save_every_n_epochs", 1) or 1
    if ep:
        W.append(f"**{rep_txt} repeats × {ep} epochs.** Low repeats with more epochs give a checkpoint every {save_every} "
                 f"epoch(s) ({max(1, ep // save_every)} in total) to compare. The best one is usually around epoch "
                 f"{max(1, round(ep * 0.6))}-{max(1, round(ep * 0.8))} of {ep}, not the last.")
    opt, lr = (f.get("optimizer_type") or "").lower(), f.get("learning_rate") or ""
    if opt or lr:
        W.append(f"**{f.get('optimizer_type') or 'optimizer'}, LR {lr or '?'}.** " + OPT_WHY.get(opt, "")
                 + " If the loss turns to NaN, retry at half the LR. If likeness is weak at the end, raise it or add epochs.")
    sched = (f.get("lr_scheduler") or "").lower()
    warm = I("lr_warmup_steps")
    if sched:
        pct = f" The {warm} warmup steps ({100 * warm / total:.0f}% of the run) ramp the LR up gradually so the first noisy steps cannot damage the weights." if warm and total else ""
        W.append(f"**Scheduler {sched}.** " + SCHED_WHY.get(sched, "") + pct)
    dim, alpha = I("network_dim"), I("network_alpha")
    if dim and alpha:
        W.append(f"**Dim {dim}, alpha {alpha} (scale {alpha / dim:.2f}).** Dim is the LoRA's capacity; 16-32 is plenty for "
                 "one character, and more mostly adds file size and overfitting. A scale below 1 acts like a lower "
                 "learning rate and keeps training stable.")
    mp = (f.get("mixed_precision") or "").lower()
    if mp:
        W.append("**Precision " + mp + ".** " + ("bf16 has fp32's number range, which avoids the NaN/overflow problems of fp16, and your RDNA4 GPU supports it natively." if mp == "bf16" else "fp16 is fast but can overflow; switch to bf16 if you see NaN."))
    if f.get("resolution"):
        W.append(f"**Resolution {f['resolution']}** with aspect-ratio bucketing, so non-square images are trained without cropping.")
    if f.get("timestep_sampling") or f.get("discrete_flow_shift"):
        W.append(f"**Flow-matching settings** (timestep sampling {f.get('timestep_sampling') or 'default'}, shift "
                 f"{f.get('discrete_flow_shift') or 'default'}). The shift moves training towards noisier timesteps, which suits large images; "
                 "it is a common starting value for LoRA training of this model family.")
    for flag, why in FLAG_WHY:
        if _has(text, flag):
            W.append(f"**`{flag}`.** {why}")
    if f.get("sample_every_n_epochs"):
        W.append(f"**Sample images every {f['sample_every_n_epochs']} epoch(s)** from the prompts file with a fixed seed, so you can compare epochs visually.")
    L += ["", "**Why these settings**", ""] + [f"- {w}" for w in W]

    # warnings
    A = []
    if create:
        A.append(f"There is no `<N>_name` repeat folder yet. **`{create}/` will be created when you save** (with the `model` and `log` "
                 "folders), and images lying directly in the LoRA folder are moved into it, if the checkbox next to Save is ticked.")
    if rename:
        A.append(f"Rename the repeat folder (see **Repeat folders** above, or run `{rename}`): sd-scripts reads the repeat count from the folder name.")
    elif not existing and rec and rec.get("want_repeats") and rec["want_repeats"] != rec["repeats"][0]:
        A.append(f"The folder uses {rec['repeats'][0]} repeats; about {rec['want_repeats']} would hit the step target best.")
    if total is not None:
        if total < lo:
            A.append(f"Total steps ({total}) is below the recommended {lo}-{hi}: likely undertrained. About {max(1, round(cfg['steps'] / spe))} epochs would give ~{cfg['steps']} steps.")
        elif total > hi:
            A.append(f"Total steps ({total}) is above the recommended {lo}-{hi}: overfit risk. About {max(1, round(cfg['steps'] / spe))} epochs would give ~{cfg['steps']} steps.")
    if views is not None and views < VIEWS_OK[0]:
        A.append(f"Only ~{views:.0f} views per image: weak likeness is likely.")
    if views is not None and views > VIEWS_OK[1]:
        A.append(f"~{views:.0f} views per image: overfitting and memorized poses are likely.")
    if bs > cfg["max_batch"]:
        A.append(f"Batch size {bs} is above the {cfg['max_batch']} that fits in 16 GB for this type: out-of-memory risk.")
    if miss:
        A.append(f"{miss} image(s) have no caption file (check the Review tab).")
    if stats["loose"]:
        A.append("Images outside `<N>_name` repeat folders are ignored by sd-scripts: " + ", ".join(f"{p.name} ({n})" for p, n in stats["loose"]))
    if imgs < 15:
        A.append(f"Only {imgs} images: expect weak or overfit results. 20-100 good, varied images is the sweet spot.")
    if get_export(text, "HF_TOKEN") == "" and read_hf_token():
        A.append("`HF_TOKEN` is empty in this script but huggingtoken.txt has a token. Press **Fill HF_TOKEN** if you use gated Hugging Face models.")
    if not _has(text, "--logging_dir"):
        A.append("No `--logging_dir`: nothing is written to the `log` folder, so TensorBoard has no loss curves. "
                 "Put `${LOG_DIR}` (or a folder path) in the form field 'Logging dir'.")
    if not _has(text, "--save_state"):
        A.append("No `--save_state`: a crashed run cannot be resumed (the Train tab's resume needs it).")
    for name in MODEL_FIELDS:
        v = os.path.expandvars(os.path.expanduser(f.get(name, "") or ""))
        if v and "$" not in v and not Path(v).exists():
            A.append(f"Model file not found for `--{name}`: `{v}`" + (" (placeholder: set the real path)" if "/PATH/TO/" in v else ""))
    script = cfg.get("script", "")
    if script and not (ROOT / script).is_file():
        A.append(f"`{script}` is not in `{ROOT}`: this architecture is not supported by your sd-scripts version.")
    if not (ROOT / "venv").is_dir():
        A.append(f"No `venv/` in `{ROOT}`.")
    if cfg.get("template"):
        A.append("This type's settings are a template written from the sd-scripts documentation, not tested on your machine. Use **Check flags** before training.")
    if A:
        L += ["", "**Check before you train**", ""] + [f"- {a}" for a in A]
    if cfg.get("notes"):
        L += ["", f"**Notes for {cfg['label']}**", ""] + [f"- {n}" for n in cfg["notes"]]
    return "\n".join(L)


def explain_for(ds, tkey, script, text, plan):
    """Explanation markdown for whatever is in the editor right now."""
    try:
        sc, presets = merged_presets()
        if tkey not in presets:
            return "Pick a LoRA type."
        cfg = dict(presets[tkey])
        stats = with_virtual_row(dataset_stats(ds))
        if not stats["rows"] or not stats["total"]:
            return "**No analysis:** this dataset has no images in `<N>_name` repeat folders yet."
        rec = recommend_numbers(sc, cfg, stats, True) if sc else None
        return explain_md(ds, tkey, cfg, stats, text, rec, existing=(script != NEW),
                          planned_repeats=(plan or {}).get("repeats") or (rec or {}).get("repeats"),
                          rename=(plan or {}).get("rename"), create=(plan or {}).get("create") or (rec or {}).get("create"))
    except Exception as e:
        return f"**Could not build the explanation:** {e}"


# ---- handlers
KEEP = object()


def _cs_result(script=KEEP, type_=KEEP, text=KEEP, fields=None, explain=KEEP, prompts_path=KEEP,
               prompts_text=KEEP, newname=KEEP, save_over=KEEP, status=KEEP, plan=None, samples=KEEP):
    def k(v):
        return gr.update() if v is KEEP else v
    flds = fields if fields is not None else [gr.update()] * len(FORM_FIELDS)
    return ([k(script), k(type_), k(text)] + list(flds)
            + [k(explain), k(prompts_path), k(prompts_text), k(newname), k(save_over), k(status), plan or {}, k(samples)])


def _plan_state(plan):
    return {"repeats": plan["rec"]["repeats"], "rename": plan["rec"]["rename"], "create": plan["rec"]["create"]}


def _new_status(plan):
    n = " (a launch script for this folder already exists; the new one gets a different name)" if launch_scripts(plan["stats"]["lora"].name) else ""
    return (f"**New launch script:** recommended settings for {plan['cfg']['label']}{n}. Nothing is written until you press Save as new.")


def cs_load(ds, script, tkey, steps_over, use_rep, hf_auto=True, samples=True):
    if not ds:
        return _cs_result(status="Pick a LoRA folder.")
    files = launch_scripts(ds)
    if script != NEW and script not in {str(p) for p in files}:
        script = str(files[0]) if files else NEW
    upd = gr.update(choices=cs_script_choices(ds), value=script)
    n = len(FORM_FIELDS)
    if script == NEW:
        try:
            plan = make_plan(ds, tkey, steps_over, use_rep, hf_auto, samples)
        except Exception as e:
            return _cs_result(script=upd, text="", fields=[""] * n, explain="", prompts_path="", prompts_text="",
                              newname="", save_over=gr.update(visible=False),
                              status=f"**Cannot build a recommendation:** {e}")
        text, ps = plan["script_text"], _plan_state(plan)
        return _cs_result(script=upd, text=text, fields=read_form(text),
                          explain=explain_md(ds, tkey, plan["cfg"], plan["stats"], text, plan["rec"], False,
                                             ps["repeats"], ps["rename"], ps["create"]),
                          prompts_path=str(plan["prompts_path"]) if samples else "",
                          prompts_text=plan["prompts_text"] if samples else "",
                          newname=plan["filename"], save_over=gr.update(visible=False),
                          status=_new_status(plan), plan=ps)
    text = Path(script).read_text(encoding="utf-8", errors="ignore")
    filled = False
    if hf_auto:
        text, filled = fill_hf(text, force=False)
    _, presets = merged_presets()
    tk = infer_type(text, presets, tkey)
    pp = prompts_path_from_text(text, lora_path(ds))
    ptext = pp.read_text(encoding="utf-8", errors="ignore") if pp and pp.is_file() else ""
    label = presets.get(tk, {}).get("label", tk)
    return _cs_result(script=upd, type_=tk, text=text, fields=read_form(text),
                      explain=explain_for(ds, tk, script, text, {}),
                      samples=has_samples(text),
                      prompts_path=str(pp) if pp else "", prompts_text=ptext,
                      newname=unique_script_name(lora_path(ds), Path(script).stem + "_copy.sh"),
                      save_over=gr.update(visible=True),
                      status=f"**Loaded `{Path(script).name}`** (detected type: {label}). Edit the form or the text, then Save. "
                             "Use Save as new to keep the original untouched."
                             + (" `HF_TOKEN` was empty and has been filled from huggingtoken.txt (not saved yet)." if filled else ""))


def cs_ds_change(ds, tkey, steps_over, use_rep, hf_auto=True, samples=True):
    return cs_load(ds, "", tkey, steps_over, use_rep, hf_auto, samples)


def cs_script_change(ds, script, tkey, steps_over, use_rep, hf_auto=True, samples=True):
    return cs_load(ds, script, tkey, steps_over, use_rep, hf_auto, samples)


def cs_suggest(ds, script, tkey, steps_over, use_rep, hf_auto=True, samples=True):
    """Replace the editor with the recommended script for the chosen type (does not save anything)."""
    if not ds:
        return _cs_result(status="Pick a LoRA folder.")
    try:
        plan = make_plan(ds, tkey, steps_over, use_rep, hf_auto, samples)
    except Exception as e:
        return _cs_result(status=f"**Cannot build a recommendation:** {e}")
    text, ps = plan["script_text"], _plan_state(plan)
    gr.Info("The editor now shows the recommended script. Nothing is saved until you press a Save button.")
    return _cs_result(text=text, fields=read_form(text),
                      explain=explain_md(ds, tkey, plan["cfg"], plan["stats"], text, plan["rec"], False,
                                         ps["repeats"], ps["rename"], ps["create"]),
                      prompts_path=(str(plan["prompts_path"]) if samples else "") if script == NEW else KEEP,
                      prompts_text=(plan["prompts_text"] if samples else "") if script == NEW else KEEP,
                      status=f"Recommended settings for {plan['cfg']['label']} loaded into the editor (not saved).",
                      plan=ps if script == NEW else {})


def cs_retype(ds, script, tkey, text, steps_over, use_rep, hf_auto=True, samples=True):
    if script == NEW:
        return cs_load(ds, NEW, tkey, steps_over, use_rep, hf_auto, samples)
    return _cs_result(explain=explain_for(ds, tkey, script, text, {}),
                      status="Comparison updated for the chosen type. Your script text is unchanged.")


def cs_samples_toggle(ds, script, tkey, text, plan, steps_over, use_rep, hf_auto, samples):
    """Yes/no: generate sample images during training. New script: rebuild it. Existing script: add or remove
    the sample arguments in the script text (nothing is saved until you press a Save button)."""
    if script == NEW:
        return cs_load(ds, NEW, tkey, steps_over, use_rep, hf_auto, samples)
    if not samples:
        new = strip_samples(text or "")
        return _cs_result(text=new, fields=read_form(new), explain=explain_for(ds, tkey, script, new, plan),
                          status="Sample images switched **off** in the script text (not saved yet).", plan=plan)
    try:
        sc, presets = merged_presets()
        cfg = dict(presets[tkey])
        lora = lora_path(ds)
        new, ok = add_samples(text or "", cfg, lora)
        if not ok:
            gr.Warning("No training command found to add the sample arguments to.")
            return _cs_result(status="No training command found to add the sample arguments to.", samples=False, plan=plan)
        pp = prompts_path_from_text(new, lora) or (lora / cfg["prompts_name"])
        if pp.is_file():
            ptext = pp.read_text(encoding="utf-8", errors="ignore")
        else:
            ptext = build_prompts(sc, with_virtual_row(dataset_stats(ds)), cfg)[1]
    except Exception as e:
        gr.Warning(f"Could not switch samples on: {e}")
        return _cs_result(status=f"Could not switch samples on: {e}", samples=False, plan=plan)
    return _cs_result(text=new, fields=read_form(new), explain=explain_for(ds, tkey, script, new, plan),
                      prompts_path=str(pp), prompts_text=ptext, plan=plan,
                      status="Sample images switched **on** in the script text (not saved yet). "
                             "The prompts file is created on save if it does not exist.")


def cs_apply(ds, tkey, script, text, plan, *vals):
    new, changes, failed = apply_form(text or "", list(vals))
    if changes:
        gr.Info("Applied to the script text (not saved yet): " + "; ".join(changes))
    if failed:
        gr.Warning("No training command found to add these to: " + ", ".join(failed))
    if not changes and not failed:
        gr.Info("Nothing to change: the form matches the text, or its fields are empty.")
    return _cs_result(text=new, explain=explain_for(ds, tkey, script, new, plan), plan=plan)


def cs_read(ds, tkey, script, text, plan):
    return _cs_result(fields=read_form(text or ""), explain=explain_for(ds, tkey, script, text or "", plan), plan=plan)


def cs_check(text):
    script, tail = training_command(text or "")
    if not script:
        return "No training command (a line with `*train*.py`) found in the script."
    py, sp = ROOT / "venv" / "bin" / "python", ROOT / script
    if not py.is_file():
        return f"venv python not found: `{py}`"
    if not sp.is_file():
        return f"`{script}` is not in `{ROOT}`: your sd-scripts version does not support this architecture."
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    try:
        r = subprocess.run([str(py), str(sp), "--help"], cwd=str(ROOT), capture_output=True, text=True,
                           timeout=240, env=env)
    except subprocess.TimeoutExpired:
        return "`--help` timed out after 4 minutes."
    helptxt = (r.stdout or "") + (r.stderr or "")
    known = set(re.findall(r"--[A-Za-z0-9][A-Za-z0-9_-]*", helptxt))
    if len(known) < 20:
        return "Could not read the option list from `--help`:\n\n```\n" + helptxt[-800:] + "\n```"
    used = []
    for fl in re.findall(r"(?<![\w-])(--[A-Za-z0-9][A-Za-z0-9_-]*)", tail):
        if fl not in used:
            used.append(fl)
    bad = [x for x in used if x not in known]
    if not bad:
        return f"All {len(used)} options in the training command are accepted by `{script}` in your sd-scripts."
    out = [f"{len(bad)} of {len(used)} options are **not** known to `{script}` in your sd-scripts:"]
    for b in bad:
        close = difflib.get_close_matches(b, known, n=2, cutoff=0.7)
        out.append(f"- `{b}`" + (f"  (did you mean {', '.join('`' + c + '`' for c in close)}?)" if close else ""))
    return "\n".join(out)


def _folder_note(notes):
    return (" Folders: " + "; ".join(notes) + ".") if notes else ""


def cs_save_over(ds, script, text, tkey, plan, makedirs):
    if script in (None, "", NEW) or not Path(script).is_file():
        gr.Warning("No existing launch script is selected: use Save as new.")
        return "No existing launch script is selected."
    p = Path(script)
    shutil.copy2(p, p.with_name(p.name + ".bak"))
    p.write_text((text or "").rstrip("\n") + "\n", encoding="utf-8")
    msg = f"**Saved `{p.name}`** (previous version: `{p.name}.bak`). The Train tab always runs the saved file."
    if has_samples(text):                     # create the prompts file if the script points at a missing one
        pp = prompts_path_from_text(text, lora_path(ds))
        if pp and not pp.exists() and _inside(pp, lora_path(ds)):
            try:
                _, presets = merged_presets()
                pp.write_text(build_prompts(merged_presets()[0], with_virtual_row(dataset_stats(ds)), presets[tkey])[1], encoding="utf-8")
                msg += f" Created the prompts file `{pp.name}`."
            except Exception:
                pass
    notes = ensure_folders(ds, text, plan, tkey) if makedirs else []
    gr.Info(f"Saved {p.name}")
    return msg + _folder_note(notes)


def cs_save_new(ds, newname, text, prompts_path, prompts_text, tkey, script, plan, makedirs):
    if not ds:
        return _cs_result(status="Pick a LoRA folder.", plan=plan) + [gr.update()]
    name = (newname or "").strip()
    if name and not name.endswith(".sh"):
        name += ".sh"
    if not name or not re.fullmatch(r"[\w][\w .\-]*\.sh", name):
        gr.Warning("Enter a file name like MyCharacterFLUXtraining.sh (letters, digits, space, _ - . only).")
        return _cs_result(status="Invalid file name.", plan=plan) + [gr.update()]
    dest = lora_path(ds) / name
    if dest.exists():
        gr.Warning(f"{name} already exists. Pick another name, or select it and use Save (overwrite).")
        return _cs_result(status=f"`{name}` already exists. Choose another name.", plan=plan) + [gr.update()]
    dest.write_text((text or "").rstrip("\n") + "\n", encoding="utf-8")
    dest.chmod(dest.stat().st_mode | 0o755)
    msg = f"**Saved new launch script `{name}`**"
    pp = Path(os.path.expanduser(prompts_path)) if (prompts_path or "").strip() else None
    if pp and has_samples(text) and (prompts_text or "").strip() and not pp.exists():
        pp.write_text(prompts_text.rstrip("\n") + "\n", encoding="utf-8")
        msg += f" and the prompts file `{pp.name}`"
    elif not has_samples(text):
        msg += " (no sample images, so no prompts file)"
    msg += "."
    notes = ensure_folders(ds, text, plan, tkey) if makedirs else []
    msg += _folder_note(notes) + " It is selected here and in the Train tab."
    gr.Info(f"Saved {name}")
    res = _cs_result(script=gr.update(choices=cs_script_choices(ds), value=str(dest)), save_over=gr.update(visible=True),
                     status=msg, plan={})
    return res + [gr.update(choices=list_scripts(), value=str(dest))]


def cs_save_prompts(path, text):
    if not (path or "").strip():
        gr.Warning("No prompts file path.")
        return "No prompts file path."
    p = Path(os.path.expanduser(path.strip()))
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        shutil.copy2(p, p.with_name(p.name + ".bak"))
    p.write_text((text or "").rstrip("\n") + "\n", encoding="utf-8")
    gr.Info(f"Saved {p.name}")
    return f"Saved `{p}`."



# --------------------------------------------------------------------------- Hugging Face token
HF_FILE = "huggingtoken.txt"


def read_hf_token() -> str:
    """First non-empty, non-comment line of <sd-scripts>/huggingtoken.txt (empty string if missing)."""
    try:
        for line in (ROOT / HF_FILE).read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                return line.strip("\"'")
    except OSError:
        pass
    return ""


def mask_token(tok: str) -> str:
    if not tok:
        return ""
    return f"{tok[:4]}…{tok[-4:]}" if len(tok) > 12 else "•" * len(tok)


def hf_status_text() -> str:
    tok = read_hf_token()
    if tok:
        return f"**HF token:** `{mask_token(tok)}` (from {HF_FILE})"
    return f"**HF token:** not set. Put it in `{ROOT / HF_FILE}`"


def header_status():
    return gpu_status_text(), hf_status_text()


def get_export(text: str, name: str):
    m = re.search(rf"^export\s+{name}=(?:\"([^\"]*)\"|'([^']*)'|(\S*))", text, re.M)
    if not m:
        return None
    return next((g for g in m.groups() if g is not None), "")


def set_export(text: str, name: str, value: str) -> str:
    line = f'export {name}="{value}"'
    pat = re.compile(rf"^export\s+{name}=.*$", re.M)
    if pat.search(text):
        return pat.sub(lambda m: line, text, count=1)
    lines = text.split("\n")
    for i, l in enumerate(lines):
        if l.startswith("cd "):                       # keep it above the cd into sd-scripts
            lines.insert(i, line)
            return "\n".join(lines)
    lines.insert(1 if lines and lines[0].startswith("#!") else 0, line)
    return "\n".join(lines)


def fill_hf(text: str, force: bool):
    """Write the token from huggingtoken.txt into `export HF_TOKEN=...`.
    force=False only fills an empty value. Returns (text, changed)."""
    tok = read_hf_token()
    if not tok or not re.fullmatch(r"[\w.\-]+", tok):
        return text, False
    cur = get_export(text, "HF_TOKEN")
    if cur == tok or (cur and not force):
        return text, False
    return set_export(text, "HF_TOKEN", tok), True


def cs_fill_hf(text, plan):
    tok = read_hf_token()
    if not tok:
        gr.Warning(f"No token found in {ROOT / HF_FILE}.")
        return _cs_result(status=f"No token found in `{ROOT / HF_FILE}`.", plan=plan)
    if not re.fullmatch(r"[\w.\-]+", tok):
        gr.Warning("The token contains unusual characters; not written into the script.")
        return _cs_result(status="The token contains unusual characters, so it was not written into the script.", plan=plan)
    new, changed = fill_hf(text or "", force=True)
    return _cs_result(text=new, plan=plan,
                      status="`HF_TOKEN` set in the script text (not saved yet)." if changed
                      else "`HF_TOKEN` already matches huggingtoken.txt.")


# --------------------------------------------------------------------------- library scan
KNOWN_LIBS = CACHE_DIR / "known_libraries.json"


def scan_library(cur_cs, cur_rv):
    img = ROOT / "images"
    if not img.is_dir():
        return gr.update(), gr.update(), f"`{img}` does not exist.", gr.update()
    try:
        known = set(json.loads(KNOWN_LIBS.read_text()))
    except Exception:
        known = None
    names = list_lora_dirs()
    rows, new = [], []
    for name in names:
        fr = repeat_folders(lora_path(name))
        imgs = capt = 0
        for f, _ in fr:
            for im in list_images(f):
                imgs += 1
                capt += bool(read_caption(im))
        reps = sorted({r for _, r in fr if r})
        loose = [f for f, r in fr if r is None]
        nscr = len(launch_scripts(name))
        notes = []
        if known is not None and name not in known:
            new.append(name)
            notes.append("**NEW**")
        if not imgs:
            notes.append("no images")
        elif not reps:
            notes.append("no `N_name` repeat folder: sd-scripts would ignore the images")
        elif loose:
            notes.append("some images are outside repeat folders")
        if imgs and capt < imgs:
            notes.append(f"{imgs - capt} without caption")
        if imgs and not nscr:
            notes.append("no launch script yet")
        rows.append(f"| {name} | {imgs} | {capt} | {', '.join(str(r) for r in reps) or '-'} | {nscr} | {'; '.join(notes) or 'ready'} |")
    gone = sorted(known - set(names)) if known else []
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        KNOWN_LIBS.write_text(json.dumps(names))
    except OSError:
        pass
    if known is None:
        head = f"**First scan:** {len(names)} image libraries in `{img}`. They are remembered, so the next scan can point out new ones."
    elif new:
        head = f"**{len(new)} new image librar{'y' if len(new) == 1 else 'ies'} since the last scan:** {', '.join(new)}. ({len(names)} in total.)"
    else:
        head = f"No new image libraries since the last scan. ({len(names)} in total.)"
    if gone:
        head += f" No longer present: {', '.join(gone)}."
    table = ["", "| Library | Images | Captions | Repeats | Launch scripts | Notes |", "|---|---|---|---|---|---|"] + rows
    ds_cs = gr.update(choices=names, value=cur_cs if cur_cs in names else None)
    ds_rv = gr.update(choices=names, value=cur_rv if cur_rv in names else None)
    return ds_cs, ds_rv, head + "\n".join(table), gr.update(open=True)


# --------------------------------------------------------------------------- repeat folders (rename)
def base_name(name: str) -> str:
    return REPEAT_RE.sub("", name, count=1)


def folder_rows(ds):
    return [(f, r, len(list_images(f))) for f, r in repeat_folders(lora_path(ds))]


def rp_choices(ds):
    lora, out = lora_path(ds), []
    for f, r, n in folder_rows(ds):
        if f == lora:
            label = f"(images directly in {lora.name}/)  {n} images, no repeat folder"
        else:
            label = f"{f.relative_to(lora)}  ({n} images, " + (f"{r} repeats)" if r else "no repeat count)")
        out.append((label, str(f)))
    return out


def rp_new_name(ds, folder, repeats):
    if not ds or not folder:
        return ""
    lora, f = lora_path(ds), Path(folder)
    base = lora.name.lower() if f == lora else base_name(f.name)
    try:
        r = int(repeats)
    except (TypeError, ValueError):
        r = 0
    return f"{r}_{base}" if r > 0 else ""


def rp_refresh(ds, tkey, steps_over):
    if not ds or not lora_path(ds).is_dir():
        return [gr.update(choices=[], value=None), 0, "", ""]
    rows = folder_rows(ds)
    if not rows:
        return [gr.update(choices=[], value=None), 0, "", "No folders with images found in this dataset."]
    n_total = sum(n for _, _, n in rows)
    rec, label = None, tkey
    try:
        sc, presets = merged_presets()
        cfg = dict(presets[tkey])
        so = _num(steps_over)
        if so:
            cfg["steps"] = so
        label = cfg["label"]
        if sc:
            bs, ga, _ = sc.suggest_settings(n_total, cfg)
            rec = sc.suggest_repeats(n_total, bs, ga, cfg["epochs"], cfg["steps"])
    except Exception:
        pass
    pick = next((f for f, r, n in rows if r is None), None)
    if pick is None and rec:
        pick = next((f for f, r, n in rows if r != rec), None)
    if pick is None:
        pick = rows[0][0]
    repeats = rec or next((r for f, r, n in rows if f == pick and r), 0)
    lora = lora_path(ds)
    lines = [f"Recommended repeats for {label} with {n_total} images: **{rec}**." if rec else f"{n_total} images."]
    for f, r, n in rows:
        nm = "(LoRA folder itself)" if f == lora else f"`{f.relative_to(lora)}`"
        lines.append(f"- {nm}: {n} images, " + (f"{r} repeats" + (" (matches)" if rec and r == rec else "")
                                                if r else "**no repeat count: sd-scripts ignores these images**"))
    return [gr.update(choices=rp_choices(ds), value=str(pick)), repeats, rp_new_name(ds, str(pick), repeats), "\n".join(lines)]


def rp_name_update(ds, folder, repeats):
    return rp_new_name(ds, folder, repeats)


def rp_rename(ds, folder, newname, move_ok):
    """Rename a repeat folder to <N>_name, or move loose images into a new repeat folder. Returns (message, userep)."""
    keep = gr.update()
    if not ds or not folder:
        gr.Warning("Pick a folder first.")
        return "Pick a folder first.", keep
    lora, f, name = lora_path(ds), Path(folder), (newname or "").strip()
    if not re.fullmatch(r"\d+_[^/\\\x00]+", name):
        gr.Warning("The name must look like 4_name (a repeat count, an underscore, then a name).")
        return "The name must look like `4_name`.", keep
    if f == lora:                                   # loose images directly in the LoRA folder
        if not move_ok:
            gr.Warning("Tick the box to allow moving the images into a new folder.")
            return "Tick the box above to allow moving these images into a new folder.", keep
        dest = lora / name
        if dest.exists():
            gr.Warning(f"{name} already exists.")
            return f"`{name}` already exists.", keep
        dest.mkdir()
        moved = 0
        for im in list_images(lora):
            for src in [im] + [im.with_name(im.stem + e) for e in CAPTION_EXTS]:
                if src.is_file():
                    shutil.move(str(src), str(dest / src.name))
                    moved += 1
        gr.Info(f"Moved {moved} files into {name}/")
        return f"Moved {moved} files (images and captions) into `{name}/`.", gr.update(value=False)
    dest = f.with_name(name)
    if dest == f:
        return "The folder already has that name.", keep
    if dest.exists():
        gr.Warning(f"{name} already exists.")
        return f"`{name}` already exists next to it.", keep
    f.rename(dest)
    gr.Info(f"Renamed {f.name} -> {name}")
    return f"Renamed `{f.name}` to `{name}`. Epochs are re-planned for the folder's actual repeat count.", gr.update(value=False)


def rp_make(ds, tkey, text, plan):
    if not ds:
        gr.Warning("Pick a LoRA folder first.")
        return "Pick a LoRA folder first."
    notes = ensure_folders(ds, text or "", plan, tkey)
    gr.Info("Folders checked.")
    return ("Done: " + "; ".join(notes) + ".") if notes else "Nothing to create: the repeat folder, `model` and `log` already exist."


def cs_set_epochs(ds, tkey, script, text, plan, steps_over):
    """Set epochs so the run lands on the target step count with the script's batch/accum and the folder's repeats."""
    try:
        _, presets = merged_presets()
        cfg = dict(presets[tkey])
        steps = _num(steps_over) or cfg["steps"]
        stats = dataset_stats(ds)
        f = form_dict(text or "")
        bs = max(1, int(float(f.get("train_batch_size") or 1)))
        ga = max(1, int(float(f.get("gradient_accumulation_steps") or 1)))
        reps = (plan or {}).get("repeats") or [r["repeats"] for r in stats["rows"]]
        samples = sum(r["n"] * rep for r, rep in zip(stats["rows"], reps))
        spe = max(1, math.ceil(samples / (bs * ga)))
        ep = max(1, round(steps / spe))
    except Exception as e:
        gr.Warning(f"Could not calculate epochs: {e}")
        return _cs_result(status=f"Could not calculate epochs: {e}", plan=plan)
    vals = [""] * len(FORM_FIELDS)
    vals[[n for n, _ in FORM_FIELDS].index("max_train_epochs")] = str(ep)
    new, _, _ = apply_form(text or "", vals)
    return _cs_result(text=new, fields=read_form(new), explain=explain_for(ds, tkey, script, new, plan),
                      status=f"Epochs set to {ep}: {spe} steps/epoch × {ep} = ~{spe * ep} steps (target {steps}). Not saved yet.",
                      plan=plan)



# --------------------------------------------------------------------------- review tab
def sub_choices(ds):
    lora = lora_path(ds)
    out = [("All folders", "__all__")]
    for f, r in repeat_folders(lora):
        rel = "." if f == lora else str(f.relative_to(lora))
        info = f"{r} repeats" if r else "no repeat count, sd-scripts ignores it"
        out.append((f"{rel}  ({len(list_images(f))} images, {info})", str(f)))
    return out


def scope_folders(ds, sub):
    if not sub or sub == "__all__":
        return [f for f, _ in repeat_folders(lora_path(ds))]
    return [Path(sub)]


def collect_items(ds, sub, filt, only_missing):
    items = []
    filt = (filt or "").strip().lower()
    for folder in scope_folders(ds, sub):
        for img in list_images(folder):
            cap = read_caption(img)
            if only_missing and cap:
                continue
            if filt and filt not in (cap or "").lower():
                continue
            items.append((img, cap))
    return items


def review_page(ds, sub, filt, missing, size, page):
    if not ds:
        return [], [], 1, "Pick a dataset folder."
    items = collect_items(ds, sub, filt, missing)
    size = int(size or 24)
    pages = max(1, math.ceil(len(items) / size))
    page = min(max(1, int(page or 1)), pages)
    chunk = items[(page - 1) * size: page * size]
    gallery = [(thumb(img), f"{img.name}: {(cap[:80] if cap else '(no caption)')}") for img, cap in chunk]
    paths = [str(img) for img, _ in chunk]
    capt = sum(1 for _, c in items if c)
    return gallery, paths, page, f"Page {page}/{pages}: {len(items)} image(s) match the filter, {capt} captioned. Click an image to edit its caption."


def rv_first(ds, sub, filt, missing, size):
    return review_page(ds, sub, filt, missing, size, 1)


def rv_prev(ds, sub, filt, missing, size, page):
    return review_page(ds, sub, filt, missing, size, (page or 1) - 1)


def rv_next(ds, sub, filt, missing, size, page):
    return review_page(ds, sub, filt, missing, size, (page or 1) + 1)


def rv_ds_change(ds):
    if not ds or not lora_path(ds).is_dir():
        return gr.update(choices=[("All folders", "__all__")], value="__all__"), ""
    total = capt = 0
    for f, _ in repeat_folders(lora_path(ds)):
        for img in list_images(f):
            total += 1
            capt += bool(read_caption(img))
    summary = f"**{ds}:** {total} images, {capt} with a caption, {total - capt} without."
    return gr.update(choices=sub_choices(ds), value="__all__"), summary


def rv_refresh_dirs():
    return gr.update(choices=list_lora_dirs())


def rv_select(paths, evt: gr.SelectData):
    idx = evt.index
    if isinstance(idx, (list, tuple)):
        idx = idx[0]
    if idx is None or not paths or idx >= len(paths):
        return None, "", "", ""
    p = Path(paths[idx])
    cap = read_caption(p)
    n = len(split_tags(cap or ""))
    info = f"**{p.name}**  ({n} tags)" + ("" if cap is not None else "  (no caption file yet, saving will create one)")
    return str(p), info, cap or "", str(p)


def record_change(desc, files):
    """files: {Path: old_text_or_None}. Kept in memory for Undo and dumped to gui_logs/ as a safety copy."""
    LAST_CHANGE["desc"] = desc
    LAST_CHANGE["files"] = dict(files)
    try:
        LOG_DIR.mkdir(exist_ok=True)
        dump = LOG_DIR / f"caption_backup_{datetime.now():%Y%m%d_%H%M%S}.json"
        dump.write_text(json.dumps({"desc": desc, "files": {str(k): v for k, v in files.items()}}))
    except OSError:
        pass


def rv_save_caption(sel, text):
    if not sel:
        gr.Warning("Select an image first.")
        return "No image selected."
    cf = caption_file(Path(sel))
    old = cf.read_text(encoding="utf-8", errors="ignore") if cf.is_file() else None
    new = (text or "").strip()
    if old is not None and old.strip() == new:
        return "No change."
    record_change(f"caption of {Path(sel).name}", {cf: old})
    cf.write_text(new, encoding="utf-8")
    return f"Saved {cf.name}. (Undo available.)"


def rv_undo():
    files = LAST_CHANGE["files"]
    if not files:
        gr.Info("Nothing to undo.")
        return "Nothing to undo."
    n = 0
    for p, old in files.items():
        try:
            if old is None:
                p.unlink(missing_ok=True)
            else:
                p.write_text(old, encoding="utf-8")
            n += 1
        except OSError:
            pass
    desc = LAST_CHANGE["desc"]
    LAST_CHANGE["files"] = {}
    return f"Undid '{desc}': restored {n} file(s)."


def bulk_files(ds, sub, scope):
    folders = scope_folders(ds, sub if str(scope).startswith("Selected") else "__all__")
    out = []
    for f in folders:
        for img in list_images(f):
            cf = caption_file(img)
            if cf.is_file():
                out.append(cf)
    return out


def transform(text, find, repl, mode, ignore_case):
    """Returns the new caption text, or None if unchanged. Raises re.error for a bad regex."""
    if mode == "tag":
        old = split_tags(text)
        out, hit = [], False
        for t in old:
            same = t.lower() == find.lower() if ignore_case else t == find
            if same:
                hit = True
                if repl.strip():
                    out.append(repl.strip())
            else:
                out.append(t)
        if not hit:
            return None
        seen, dedup = set(), []                  # drop duplicates the replacement may have created
        for t in out:
            k = t.lower()
            if k not in seen:
                seen.add(k)
                dedup.append(t)
        return ", ".join(dedup) if dedup != old else None
    flags = re.I if ignore_case else 0
    if mode == "text":
        new = re.sub(re.escape(find), lambda m: repl, text, flags=flags)
    else:
        new = re.sub(find, repl, text, flags=flags)
    return None if new == text else new


def bulk_changes(ds, sub, scope, find, repl, mode, ic):
    if not find:
        raise ValueError("Enter something to find.")
    files = bulk_files(ds, sub, scope)
    changes = []
    for cf in files:
        old = cf.read_text(encoding="utf-8", errors="ignore")
        new = transform(old.strip(), find, repl or "", mode, ic)
        if new is not None:
            changes.append((cf, old, new))
    return files, changes


def bulk_preview(ds, sub, scope, find, repl, mode, ic):
    if not ds:
        return "Pick a dataset folder first."
    try:
        files, changes = bulk_changes(ds, sub, scope, find, repl, mode, ic)
    except (ValueError, re.error) as e:
        return f"Error: {e}"
    lines = [f"{len(changes)} of {len(files)} captions would change. Nothing has been written."]
    if mode == "tag":
        lines.append("(Whole-tag mode; duplicate tags created by the replacement are removed.)")
    for cf, old, new in changes[:12]:
        lines.append(f"\n{cf.name}\n  before: {old.strip()[:160]}\n  after:  {new.strip()[:160]}")
    if len(changes) > 12:
        lines.append(f"\n... and {len(changes) - 12} more")
    return "\n".join(lines)


def bulk_apply(ds, sub, scope, find, repl, mode, ic):
    if not ds:
        return "Pick a dataset folder first."
    try:
        files, changes = bulk_changes(ds, sub, scope, find, repl, mode, ic)
    except (ValueError, re.error) as e:
        return f"Error: {e}"
    if not changes:
        return f"0 of {len(files)} captions match. Nothing changed."
    record_change(f"bulk replace '{find}' -> '{repl}'", {cf: old for cf, old, _ in changes})
    for cf, _, new in changes:
        cf.write_text(new, encoding="utf-8")
    return f"Changed {len(changes)} of {len(files)} captions. Use 'Undo last change' to revert (safety copy in gui_logs/)."


def rv_stats(ds, sub, scope):
    if not ds:
        return "Pick a dataset folder first."
    files = bulk_files(ds, sub, scope)
    cnt = Counter()
    for cf in files:
        cnt.update({t.lower() for t in split_tags(cf.read_text(encoding="utf-8", errors="ignore"))})
    head = f"{len(files)} captions, {len(cnt)} distinct tags (count, tag; top 150)\n"
    return head + "\n".join(f"{c:>5}  {t}" for t, c in cnt.most_common(150))



# --------------------------------------------------------------------------- cleanup after a finished run
TRASH_DIR = "_cleanup_trash"
EPOCH_CKPT_RE = re.compile(r"-(?:step)?\d{3,}\.(safetensors|ckpt|pt)$", re.I)


def cleanup_plan(path, with_samples=True):
    """(base, model_dir, keep, remove, error). remove = [(Path, kind)]; nothing is touched."""
    model_dir, _ = script_paths(path)
    if not model_dir:
        return None, None, [], [], "Pick a launch script first."
    base = Path(path).parent
    if RUN.proc is not None and RUN.proc.poll() is None:
        return base, model_dir, [], [], "A run is still in progress. Wait for it to finish (or Stop it) before cleaning up."
    keep = []
    if model_dir.is_dir():
        keep = sorted(f for f in model_dir.iterdir()
                      if f.is_file() and f.suffix.lower() in (".safetensors", ".ckpt", ".pt") and not EPOCH_CKPT_RE.search(f.name))
    if not keep:
        return base, model_dir, [], [], (f"No finished LoRA file found in `{model_dir}` (a file without an `-000x` epoch number). "
                                         "The run probably did not finish, so nothing is removed: you may still want to resume it.")
    remove = []
    if model_dir.is_dir():
        for f in sorted(model_dir.iterdir()):
            if f.is_file() and EPOCH_CKPT_RE.search(f.name):
                remove.append((f, "epoch checkpoint"))
    for d in scan_states(base, 3):
        if TRASH_DIR not in d.parts:
            remove.append((d, "saved state"))
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d != TRASH_DIR]
        remove += [(Path(dirpath) / f, "cache (.npz)") for f in sorted(filenames) if f.lower().endswith(".npz")]
    sample = model_dir / "sample"
    if with_samples and sample.is_dir():
        remove.append((sample, "sample images"))
    seen, uniq = set(), []
    for item in remove:      # states may sit inside each other's search; avoid duplicates and children of removed folders
        if item[0] in seen or any(par in seen for par in item[0].parents):
            continue
        seen.add(item[0])
        uniq.append(item)
    return base, model_dir, keep, uniq, ""


def _size(p: Path) -> int:
    try:
        if p.is_file():
            return p.stat().st_size
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    except OSError:
        return 0


def cleanup_preview(path, with_samples):
    base, model_dir, keep, remove, err = cleanup_plan(path, with_samples)
    if err:
        return err
    total = sum(_size(p) for p, _ in remove)
    if not remove:
        return "Nothing to clean up: no checkpoints, states or .npz files found. Final file: " + ", ".join(f"`{k.name}`" for k in keep)
    kinds = {}
    for p, k in remove:
        kinds.setdefault(k, []).append(p)
    rows = ["| What | Count | Size |", "|---|---|---|"]
    for k, items in kinds.items():
        rows.append(f"| {k} | {len(items)} | {sum(_size(i) for i in items) / 1024 ** 3:.2f} GB |")
    lst = "\n".join(f"- `{p.relative_to(base) if base in p.parents else p}`" for p, _ in remove[:25])
    more = f"\n- ... and {len(remove) - 25} more" if len(remove) > 25 else ""
    return (f"**Will be moved to the trash** ({total / 1024 ** 3:.2f} GB):\n\n" + "\n".join(rows)
            + f"\n\n**Kept:** original pictures and captions, launch script and prompts file, TensorBoard logs, and the final "
            + ", ".join(f"`{k.name}`" for k in keep) + f".\n\nTrash folder: `{ROOT / TRASH_DIR}/<timestamp>_{base.name}`. "
            "Tick the confirmation box and press **Move to trash** to proceed.\n\n" + lst + more)


def cleanup_run(path, with_samples, confirm):
    if not confirm:
        gr.Warning("Tick the confirmation box first.")
        return "Tick the confirmation box first."
    base, model_dir, keep, remove, err = cleanup_plan(path, with_samples)
    if err:
        return err
    if not remove:
        return "Nothing to clean up."
    dest = ROOT / TRASH_DIR / f"{datetime.now():%Y%m%d-%H%M%S}_{base.name}"
    moved, failed, total = 0, [], 0
    for p, _ in remove:
        rel = p.relative_to(base) if base in p.parents else Path(p.name)
        target = dest / rel
        try:
            size = _size(p)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), str(target))
            moved += 1
            total += size
        except OSError as e:
            failed.append(f"{p.name}: {e}")
    msg = (f"Moved {moved} item(s), {total / 1024 ** 3:.2f} GB, to `{dest}`. Delete that folder when you are sure you do not need it. "
           "Final file kept: " + ", ".join(f"`{k.name}`" for k in keep) + ".")
    if failed:
        msg += "\n\nCould not move: " + "; ".join(failed[:5])
    gr.Info("Cleanup done")
    return msg


# --------------------------------------------------------------------------- settings tab (model files)
SETTINGS_FILE = "settings.txt"
MODEL_EXTS = {".safetensors", ".sft", ".ckpt", ".pt", ".pth", ".bin"}
ST_N = 4            # number of file slots in the Settings tab (set from the presets in build_ui)
SLOT_LABELS = {
    "pretrained_model_name_or_path": "Model (checkpoint / diffusion model)", "vae": "VAE", "ae": "AE (VAE)",
    "qwen3": "Qwen3 text encoder", "clip_l": "CLIP-L text encoder", "clip_g": "CLIP-G text encoder",
    "t5xxl": "T5-XXL text encoder", "gemma2": "Gemma2 text encoder", "text_encoder": "Text encoder",
    "byt5": "ByT5 text encoder",
}


OPTIONAL_VAE_TYPES = {"sdxl", "illustrious", "sd15"}


def settings_path() -> Path:
    return ROOT / SETTINGS_FILE


def slot_label(key: str) -> str:
    return SLOT_LABELS.get(key, key.replace("_", " "))


def read_settings() -> dict:
    """settings.txt -> {lora_type: {flag: path, "<flag>.dir": folder}}."""
    out, sec = {}, None
    try:
        lines = settings_path().read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    for ln in lines:
        t = ln.strip()
        if not t or t.startswith("#"):
            continue
        if t.startswith("[") and t.endswith("]"):
            sec = t[1:-1].strip()
            out.setdefault(sec, {})
        elif sec and "=" in t:
            k, _, v = t.partition("=")
            out[sec][k.strip()] = v.strip()
    return out


def write_settings(data: dict):
    lines = ["# sdgui settings: model, VAE and text-encoder files for each LoRA type.",
             "# Written by sdgui.py (Settings tab); safe to edit by hand.",
             "# One [section] per LoRA type, `flag = path` lines, and `<flag>.dir` remembers the folder shown in the file list.", ""]
    for sec, kv in data.items():
        kv = {k: v for k, v in kv.items() if v}
        if kv:
            lines.append(f"[{sec}]")
            lines += [f"{k} = {v}" for k, v in kv.items()]
            lines.append("")
    path = settings_path()
    if path.is_file():
        shutil.copy2(path, path.with_name(path.name + ".bak"))
    path.write_text("\n".join(lines), encoding="utf-8")


def settings_models(tkey) -> dict:
    """{flag_without_dashes: path} saved for a LoRA type (used for new launch scripts)."""
    return {k: v for k, v in read_settings().get(tkey, {}).items() if v and not k.endswith(".dir")}


def type_slots(tkey):
    _, presets = merged_presets()
    cfg = presets.get(tkey) or {}
    slots = [(f.lstrip("-"), path) for f, path in cfg.get("model_args", [])]
    if tkey in OPTIONAL_VAE_TYPES and not any(k in ("vae", "ae") for k, _ in slots):
        slots.append(("vae", ""))                    # optional external VAE (--vae) for SD1.5 / SDXL family
    return slots


def list_model_files(folder, recursive=True, limit=1000):
    """[(label, full_path)] of model files in a folder (and up to two levels of subfolders)."""
    base = Path(os.path.expanduser((folder or "").strip()))
    if not (folder or "").strip() or not base.is_dir():
        return []
    found = []
    try:
        if recursive:
            for dirpath, dirnames, filenames in os.walk(base):
                depth = len(Path(dirpath).relative_to(base).parts)
                dirnames[:] = sorted(d for d in dirnames if not d.startswith(".")) if depth < 2 else []
                found += [Path(dirpath) / f for f in filenames if Path(f).suffix.lower() in MODEL_EXTS]
                if len(found) >= limit:
                    break
        else:
            found = [f for f in base.iterdir() if f.is_file() and f.suffix.lower() in MODEL_EXTS]
    except OSError:
        pass
    out = []
    for f in sorted(found, key=lambda x: str(x).lower())[:limit]:
        try:
            size = f" ({f.stat().st_size / 1024 ** 3:.1f} GB)"
        except OSError:
            size = ""
        out.append((f"{f.relative_to(base)}{size}", str(f)))
    return out


def _slot_dir(path, saved_dir):
    if saved_dir:
        return saved_dir
    return str(Path(path).parent) if path and "/PATH/TO" not in path else ""


def st_load(tkey, recursive):
    if not tkey:
        return [gr.update()] * (3 * ST_N) + [[""] * ST_N, "Pick a LoRA type."]
    slots, saved = type_slots(tkey), read_settings().get(tkey, {})
    out, flags = [], []
    for i in range(ST_N):
        if i < len(slots):
            key, default = slots[i]
            path = saved.get(key) or default
            d = _slot_dir(path, saved.get(key + ".dir"))
            lab = slot_label(key)
            flags.append(key)
            out += [gr.update(visible=True), gr.update(label=f"{lab}: folder", value=d),
                    gr.update(label=f"{lab}: file (--{key})", choices=list_model_files(d, recursive), value=path or None)]
        else:
            flags.append("")
            out += [gr.update(visible=False), gr.update(value=""), gr.update(choices=[], value=None)]
    have = settings_path().is_file()
    info = (f"Settings for this type are read from `{settings_path()}`." if saved else
            (f"No saved settings for this type yet: showing the defaults from your preset. **Save** writes them to `{settings_path()}`."
             if have else f"`{settings_path()}` does not exist yet: showing the defaults from your preset. **Save** creates it."))
    return out + [flags, info]


def st_refresh(folder, recursive, current):
    return gr.update(choices=list_model_files(folder, recursive), value=current)


def st_refresh_all(recursive, *args):
    dirs, cur = args[:ST_N], args[ST_N:2 * ST_N]
    return [st_refresh(d, recursive, c) for d, c in zip(dirs, cur)]


def st_raw() -> str:
    try:
        return settings_path().read_text(encoding="utf-8")
    except OSError:
        return "(settings.txt does not exist yet)"


def st_save(tkey, flags, *args):
    dirs, files = args[:ST_N], args[ST_N:2 * ST_N]
    if not tkey:
        gr.Warning("Pick a LoRA type first.")
        return "Pick a LoRA type first.", st_raw()
    data = read_settings()
    sec = {}
    rows = ["| Slot | File | |", "|---|---|---|"]
    for key, d, f in zip(flags, dirs, files):
        if not key:
            continue
        f = (f or "").strip()
        d = (d or "").strip()
        if "\n" in f or "\n" in d:
            gr.Warning(f"{key}: paths cannot contain line breaks.")
            return f"{key}: paths cannot contain line breaks.", st_raw()
        if d:
            sec[key + ".dir"] = d
        if f:
            sec[key] = f
        ok = bool(f) and Path(os.path.expanduser(f)).is_file()
        rows.append(f"| {slot_label(key)} | `{f or '(not set: the preset default is used)'}` | "
                    + ("found" if ok else ("**file not found**" if f else "")) + " |")
    data[tkey] = sec
    try:
        write_settings(data)
    except OSError as e:
        gr.Warning(f"Could not write settings.txt: {e}")
        return f"Could not write `{settings_path()}`: {e}", st_raw()
    gr.Info("Saved settings.txt")
    return (f"**Saved to `{settings_path()}`** for this LoRA type (other types are untouched; the previous file is kept as "
            "`settings.txt.bak`). New launch scripts of this type now use these files. For an existing script press "
            "**Use model files from Settings** in the Create Data Launch Script tab.\n\n" + "\n".join(rows)), st_raw()


# ---- GPU / GFX (HSA_OVERRIDE_GFX_VERSION), stored in settings.txt under [general]
GFX_VAR = "HSA_OVERRIDE_GFX_VERSION"
# (label, value). The value is what the ROCm override expects. Cards with a native ROCm target need no override
# in theory, but your scripts set one, so the list gives the matching number. Unsupported chips are mapped to
# the nearest supported target (RDNA2 -> 10.3.0, 780M -> 11.0.0). Check ROCm's compatibility page for your version.
KNOWN_GFX = [
    ("RX 9070 XT / 9070 / 9070 GRE, RDNA4 (gfx1201)", "12.0.1"),
    ("RX 9060 XT / 9060, RDNA4 (gfx1200)", "12.0.0"),
    ("RX 7900 XTX / XT / GRE, RDNA3 (gfx1100)", "11.0.0"),
    ("RX 7800 XT / 7700 XT, RDNA3 (gfx1101)", "11.0.1"),
    ("RX 7600 / 7600 XT, RDNA3 (gfx1102)", "11.0.2"),
    ("Radeon 780M / 760M iGPU (gfx1103, use gfx1100 override)", "11.0.0"),
    ("Strix Halo / Radeon 8060S (gfx1151)", "11.5.1"),
    ("Strix Point / Radeon 890M (gfx1150)", "11.5.0"),
    ("RX 6950 XT / 6900 XT / 6800 XT / 6800, RDNA2 (gfx1030)", "10.3.0"),
    ("RX 6750 XT / 6700 XT, RDNA2 (gfx1031, use gfx1030 override)", "10.3.0"),
    ("RX 6650 XT / 6600 XT / 6600, RDNA2 (gfx1032, use gfx1030 override)", "10.3.0"),
    ("RX 5700 XT / 5700, RDNA1 (gfx1010)", "10.1.0"),
    ("Radeon VII / MI50 (gfx906)", "9.0.6"),
    ("Vega 56 / 64 (gfx900)", "9.0.0"),
    ("Instinct MI100 (gfx908)", "9.0.8"),
    ("Instinct MI200 series (gfx90a)", "9.0.10"),
    ("Instinct MI300 series (gfx942)", "9.4.2"),
]


def gfx_name_to_version(name: str) -> str:
    """'gfx1201' -> '12.0.1', 'gfx90a' -> '9.0.10', 'gfx942' -> '9.4.2'."""
    m = re.fullmatch(r"gfx([0-9]+)([0-9a-f])([0-9a-f])", (name or "").strip().lower())
    if not m:
        return ""
    major, minor, step = m.groups()
    return f"{int(major)}.{int(minor, 16)}.{int(step, 16)}"


def get_gfx() -> str:
    return (read_settings().get("general", {}).get(GFX_VAR) or "").strip()


def apply_gfx_env():
    """Make this process (and everything it starts) use the saved value."""
    v = get_gfx()
    if v:
        os.environ[GFX_VAR] = v


def detect_gfx():
    """(version, message) from rocminfo, falling back to rocm-smi."""
    names = []
    exe = shutil.which("rocminfo") or next(iter(sorted(glob.glob("/opt/rocm*/bin/rocminfo"))), None)
    if exe:
        try:
            out = subprocess.run([exe], capture_output=True, text=True, timeout=15).stdout
            names = re.findall(r"\bgfx[0-9a-f]{3,4}\b", out)
        except Exception:
            pass
    if not names:
        exe = shutil.which("rocm-smi") or next(iter(sorted(glob.glob("/opt/rocm*/bin/rocm-smi"))), None)
        if exe:
            try:
                out = subprocess.run([exe, "--showproductname"], capture_output=True, text=True, timeout=10).stdout
                names = re.findall(r"\bgfx[0-9a-f]{3,4}\b", out)
            except Exception:
                pass
    names = [n for n in names if n != "gfx000"]
    if not names:
        return gr.update(), "Could not detect the GPU (rocminfo / rocm-smi gave no gfx name). Pick it from the list instead."
    n = names[0]
    v = gfx_name_to_version(n)
    return v, f"Detected **{n}**, which is `{v}`. If the card is not officially supported by your ROCm, pick the nearest supported entry from the list instead."


def gfx_pick(choice):
    return choice or gr.update()


def gfx_load():
    v = get_gfx() or os.environ.get(GFX_VAR, "")
    known = {val for _, val in KNOWN_GFX}
    label = next((lab for lab, val in KNOWN_GFX if val == v), None)
    note = (f"`{GFX_VAR}` is `{v}` ({'from settings.txt' if get_gfx() else 'from the environment, not saved in settings.txt'})."
            if v else f"No `{GFX_VAR}` saved yet.")
    return v, label, note


def gfx_save(value):
    value = (value or "").strip()
    if value and not re.fullmatch(r"\d+\.\d+\.\d+", value):
        gr.Warning("The value must look like 12.0.1")
        return "The value must look like `12.0.1` (major.minor.step).", st_raw()
    data = read_settings()
    sec = data.setdefault("general", {})
    if value:
        sec[GFX_VAR] = value
    else:
        sec.pop(GFX_VAR, None)
    try:
        write_settings(data)
    except OSError as e:
        return f"Could not write `{settings_path()}`: {e}", st_raw()
    if value:
        os.environ[GFX_VAR] = value
    else:
        os.environ.pop(GFX_VAR, None)
    gr.Info("Saved settings.txt")
    return (f"Saved `{GFX_VAR} = {value or '(empty)'}` to `{settings_path()}`. The GUI and anything it starts use it now. "
            "New launch scripts get it automatically; use **Update all launch scripts** for existing ones."), st_raw()


def gfx_update_scripts(confirm):
    v = get_gfx()
    if not v:
        return "Save a GFX value first."
    if not confirm:
        gr.Warning("Tick the confirmation box first.")
        return "Tick the confirmation box first."
    pat = re.compile(rf"^(\s*export\s+{GFX_VAR}=)(?:\"[^\"]*\"|'[^']*'|\S*)(.*)$", re.M)
    changed, same, none = [], 0, 0
    for _, path in list_scripts():
        f = Path(path)
        try:
            text = f.read_text(encoding="utf-8")
        except OSError:
            continue
        if not pat.search(text):
            none += 1
            continue
        new = pat.sub(lambda m: f'{m.group(1)}"{v}"{m.group(2)}', text)
        if new == text:
            same += 1
            continue
        try:
            shutil.copy2(f, f.with_name(f.name + ".bak"))
            f.write_text(new, encoding="utf-8")
            changed.append(str(f))
        except OSError:
            pass
    msg = f"Set `{GFX_VAR}={v}` in {len(changed)} launch script(s) (a `.bak` copy of each was kept); {same} already had it"
    if none:
        msg += f"; {none} have no `export {GFX_VAR}` line and were left alone"
    return msg + "." + ("\n\n" + "\n".join(f"- `{c}`" for c in changed[:20]) if changed else "")



def set_arg(text: str, name: str, value: str):
    """Set --name to value in the script text. Returns (text, changed, ok)."""
    m = _arg_re(name).search(text)
    if m:
        old = m.group(2)
        q = old[0] if old[0] in "\"'" else ""
        if (old[1:-1] if q else old) == value:
            return text, False, True
        tok = f"{q}{value}{q}" if q else (f'"{value}"' if re.search(r"\s", value) else value)
        return text[:m.start(2)] + tok + text[m.end(2):], True, True
    tok = f'"{value}"' if re.search(r"\s", value) else value
    new, ok = append_args(text, f"--{name} {tok}")
    return new, ok, ok


def cs_use_settings(ds, tkey, script, text, plan):
    """Write the model files saved in settings.txt for this LoRA type into the script text."""
    ov = settings_models(tkey)
    if not ov and not get_gfx():
        gr.Warning("No saved model files for this LoRA type: set them in the Settings tab first.")
        return _cs_result(status="No saved model files for this LoRA type: set them in the **Settings** tab first.", plan=plan)
    new, changed, failed = text or "", [], []
    for key, val in ov.items():
        new, ch, ok = set_arg(new, key, val)
        if ch:
            changed.append(key)
        if not ok:
            failed.append(key)
    if get_gfx():
        g = get_gfx()
        if get_export(new, GFX_VAR) != g:
            new = set_export(new, GFX_VAR, g)
            changed.append(GFX_VAR)
    msg = (f"Settings from settings.txt applied to the script text (not saved yet): {', '.join(k if k.isupper() else '--' + k for k in changed)}."
           if changed else "The script already uses the model files from settings.txt.")
    if failed:
        msg += f" No training command found to add: {', '.join(failed)}."
    return _cs_result(text=new, fields=read_form(new), explain=explain_for(ds, tkey, script, new, plan), status=msg, plan=plan)


# --------------------------------------------------------------------------- UI
def build_ui(default_sample_dir: str):
    global ST_N
    try:
        ST_N = max(4, max(len(v.get("model_args", [])) for v in merged_presets()[1].values()))
    except Exception:
        ST_N = 4
    scripts = list_scripts()
    with gr.Blocks(title="sd-scripts launcher", **({} if LAUNCH_HAS_CSS else {"css": CSS})) as demo:
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("## sd-scripts launcher")
            with gr.Column(scale=2):
                gpu_md = gr.HTML(gpu_status_text(), elem_id="gpu_status")
                hf_md = gr.Markdown(hf_status_text(), elem_id="hf_status")
        status = gr.Markdown(status_text())
        seen = gr.State(-1)

        with gr.Tabs():
            with gr.Tab("Train"):
                with gr.Row():
                    script_dd = gr.Dropdown(choices=scripts, value=scripts[0][1] if scripts else None,
                                            label=f"Launch script (found under {ROOT} and {ROOT}/images)", scale=5)
                    refresh_btn = gr.Button("Refresh list", scale=1)
                with gr.Row():
                    workdir = gr.Textbox(label="Working directory", value=str(ROOT), scale=3,
                                         info="Scripts run from here, so relative paths like venv/bin/activate work.")
                    venv = gr.Textbox(label="Venv to activate first (optional)", value=DEFAULT_VENV, scale=3,
                                      info="Folder or .../bin/activate. Your script's own activate line still runs after it.")
                    trace = gr.Checkbox(label="Trace commands (set -x)", value=False, scale=1)
                with gr.Accordion("Resume a crashed run from a saved state", open=False):
                    gr.Markdown("Needs the original run to have used `--save_state`. Adds `--resume <state>` "
                                "to the training command in a temp copy. Keep the same settings as the original run.")
                    with gr.Row():
                        state_dir = gr.Textbox(label="Search folder (defaults to the script's --output_dir)", scale=5)
                        find_btn = gr.Button("Find states", scale=1)
                    state_msg = gr.Markdown("")
                    state_dd = gr.Dropdown(choices=[("(start fresh, no resume)", NO_STATE)], value=NO_STATE,
                                           label="State to resume from")
                with gr.Row():
                    start_btn = gr.Button("Start", variant="primary")
                    stop_btn = gr.Button("Stop", variant="stop")
                with gr.Accordion("Clean up after a finished run", open=False):
                    gr.Markdown("Moves the epoch checkpoints (`-000x.safetensors`), saved states and all `.npz` cache files of the "
                                "selected script's folder to a trash folder, so only your pictures, captions, the final LoRA, "
                                "the launch script and the TensorBoard logs remain. Refuses to run while training is going or "
                                "when no final LoRA exists yet. Without the `.npz` caches and states, a later run re-caches and cannot resume.")
                    with gr.Row():
                        cl_samples = gr.Checkbox(label="Also move the sample images", value=True, scale=2)
                        cl_prev_btn = gr.Button("Preview cleanup", scale=1)
                        cl_confirm = gr.Checkbox(label="Yes, move these files to the trash", value=False, scale=2)
                        cl_btn = gr.Button("Move to trash", variant="stop", scale=1)
                    cl_msg = gr.Markdown("")
                progress = gr.Textbox(value=progress_text(), show_label=False, interactive=False, lines=1,
                                      max_lines=1, elem_id="progress_line")
                log = gr.Textbox(label="Log", lines=22, max_lines=22, autoscroll=True,
                                 interactive=False, value=log_text())

            with gr.Tab("Create Data Launch Script"):
                if SCAN_SCRIPT is None:
                    gr.Markdown("**scan_lora_dataset_ny.py not found.** Put it next to `sdgui.py` (or in your "
                                "sd-scripts folder) or start with `--scan-script /path/to/it.py`, then restart. "
                                "Without it only editing existing launch scripts works.")
                else:
                    gr.Markdown("Pick a LoRA folder, then either an **existing launch script** (its settings are loaded into "
                                "the form) or **+ New launch script** (recommended settings for the LoRA type, with the "
                                "reasoning and the step calculation). Nothing is written until you press a Save button.")
                _types = type_choices()
                with gr.Row():
                    cs_ds = gr.Dropdown(choices=list_lora_dirs(), label="LoRA folder", scale=3)
                    cs_script = gr.Dropdown(choices=[("+ New launch script", NEW)], value=NEW,
                                            label="Launch script", scale=4)
                    cs_type = gr.Dropdown(choices=_types, value=_types[0][1] if _types else None,
                                          label="LoRA type (model architecture)", scale=3)
                    cs_refresh = gr.Button("Scan library", scale=1)
                cs_status = gr.Markdown("")
                with gr.Accordion("Library scan (new image libraries in the sd-scripts images folder)", open=False) as cs_lib:
                    cs_lib_md = gr.Markdown("Press **Scan library** to list every image library and flag new ones.")
                with gr.Row():
                    cs_steps = gr.Number(label="Target total steps (0 = recommended for the type, press Enter)",
                                         value=0, precision=0, scale=2)
                    cs_userep = gr.Checkbox(label="Plan with the recommended repeat count (shows a rename command)",
                                            value=True, scale=3)
                    cs_hfauto = gr.Checkbox(label="Fill HF_TOKEN from huggingtoken.txt", value=True, scale=2)
                    cs_samples = gr.Checkbox(label="Make sample images during training (every epoch, from the prompts file)",
                                             value=True, scale=3)
                    cs_suggest_btn = gr.Button("Fill with recommended settings", scale=2)
                with gr.Accordion("Why these settings, with the step calculation", open=True):
                    cs_explain = gr.Markdown("")
                with gr.Accordion("Repeat folders: rename to the correct N_name format", open=True):
                    rp_info = gr.Markdown("")
                    with gr.Row():
                        rp_folder = gr.Dropdown(choices=[], label="Image folder", scale=4)
                        rp_repeats = gr.Number(label="Repeats", value=0, precision=0, scale=1)
                        rp_newname = gr.Textbox(label="New folder name", scale=3)
                    with gr.Row():
                        rp_confirm = gr.Checkbox(label="Allow moving images that sit directly in the LoRA folder into a new repeat folder",
                                                 value=False, scale=3)
                        rp_btn = gr.Button("Rename / create repeat folder", variant="primary", scale=2)
                        rp_epochs_btn = gr.Button("Set epochs to hit the target steps", scale=2)
                        rp_make_btn = gr.Button("Create missing folders now (repeat folder, model, log)", scale=2)
                    rp_msg = gr.Markdown("")
                with gr.Accordion("Hugging Face token", open=False):
                    gr.Markdown(f"Read from `{ROOT / HF_FILE}` and written into `export HF_TOKEN=\"...\"` in the launch script. "
                                "The saved script then contains the token in plain text, so keep launch scripts out of git and shared folders. "
                                "The token is masked in the run log.")
                    with gr.Row():
                        cs_hf_tok = gr.Textbox(label="Token", value=read_hf_token(), type="password", interactive=False, scale=4)
                        cs_hf_show = gr.Checkbox(label="Show", value=False, scale=1)
                        cs_hf_reload = gr.Button("Reload token file", scale=1)
                        cs_hf_fill = gr.Button("Fill HF_TOKEN in the script text", scale=2)
                cs_fields = []
                for gname, flds in FORM_GROUPS:
                    with gr.Accordion(gname, open=(gname in ("Training", "Model files"))):
                        per = 2 if gname == "Model files" else 4
                        for i in range(0, len(flds), per):
                            with gr.Row():
                                for name, label in flds[i:i + per]:
                                    cs_fields.append(gr.Textbox(label=label, info=f"--{name}"))
                with gr.Row():
                    cs_read_btn = gr.Button("Read form from script text")
                    cs_apply_btn = gr.Button("Apply form to script text", variant="primary")
                    cs_models_btn = gr.Button("Use model files from Settings")
                    cs_check_btn = gr.Button("Check flags against my sd-scripts")
                cs_check_out = gr.Markdown("")
                cs_text = gr.Code(language="shell", label="Launch script (not written until you save)",
                                  lines=26, interactive=True, elem_id="cs_text")
                with gr.Accordion("Sample prompts file", open=False):
                    cs_prompts_path = gr.Textbox(label="Prompts file")
                    cs_prompts_text = gr.Code(label="Prompts", lines=12, interactive=True)
                    with gr.Row():
                        cs_prompts_save = gr.Button("Save prompts file")
                        cs_prompts_msg = gr.Markdown("")
                cs_makedirs = gr.Checkbox(
                    label="Create missing folders when saving: the repeat folder (<N>_<name>), model and log. "
                          "Images lying directly in the LoRA folder are moved into a newly created repeat folder.",
                    value=True)
                with gr.Row():
                    cs_save_over_btn = gr.Button("Save (overwrite the selected script)", variant="primary", visible=False)
                    cs_newname = gr.Textbox(label="File name for a new launch script", scale=3)
                    cs_save_new_btn = gr.Button("Save as new launch script", variant="primary", scale=2)
                cs_plan = gr.State({})

            with gr.Tab("Review"):
                with gr.Row():
                    rv_ds = gr.Dropdown(choices=list_lora_dirs(), label="Dataset folder", scale=3)
                    rv_sub = gr.Dropdown(choices=[("All folders", "__all__")], value="__all__",
                                         label="Repeat folder", scale=4)
                    rv_refresh_btn = gr.Button("Refresh folders", scale=1)
                rv_summary = gr.Markdown("")
                with gr.Row():
                    rv_filter = gr.Textbox(label="Show only captions containing (press Enter)", scale=3)
                    rv_missing = gr.Checkbox(label="Only images without a caption", value=False, scale=1)
                    rv_size = gr.Dropdown(choices=[12, 24, 48, 96], value=24, label="Per page", scale=1)
                    rv_prev_btn = gr.Button("Prev", scale=1)
                    rv_next_btn = gr.Button("Next", scale=1)
                rv_pageinfo = gr.Markdown("")
                rv_gallery = gr.Gallery(label="Images", columns=6, height=480, object_fit="cover",
                                        allow_preview=False)
                rv_paths = gr.State([])
                rv_page = gr.State(1)
                rv_sel = gr.State("")
                with gr.Row():
                    with gr.Column(scale=2):
                        rv_img = gr.Image(label="Selected image", interactive=False, height=420)
                    with gr.Column(scale=3):
                        rv_selinfo = gr.Markdown("Click an image above.")
                        rv_caption = gr.Textbox(label="Caption (comma-separated tags)", lines=8, interactive=True)
                        with gr.Row():
                            rv_save_btn = gr.Button("Save caption", variant="primary")
                            rv_undo_btn = gr.Button("Undo last change")
                        rv_msg = gr.Markdown("")
                with gr.Accordion("Bulk find / replace in captions", open=False):
                    with gr.Row():
                        bk_find = gr.Textbox(label="Find", scale=3)
                        bk_repl = gr.Textbox(label="Replace with (empty = delete the tag/text)", scale=3)
                    with gr.Row():
                        bk_mode = gr.Radio(choices=[("Whole tag", "tag"), ("Text anywhere", "text"), ("Regex", "regex")],
                                           value="tag", label="Match")
                        bk_ic = gr.Checkbox(label="Ignore case", value=True)
                        bk_scope = gr.Radio(choices=["Whole dataset", "Selected repeat folder"],
                                            value="Whole dataset", label="Apply to")
                    with gr.Row():
                        bk_preview_btn = gr.Button("Preview (writes nothing)")
                        bk_apply_btn = gr.Button("Apply to all matching captions", variant="primary")
                        bk_undo_btn = gr.Button("Undo last change")
                    bk_out = gr.Textbox(label="Result", lines=14, interactive=False)
                with gr.Accordion("Tag statistics", open=False):
                    st_btn = gr.Button("Count tags (uses the 'Apply to' scope above)")
                    st_out = gr.Textbox(label="Tags", lines=16, interactive=False)

            with gr.Tab("Monitor"):
                gr.Markdown("### Sample images (newest first, read from disk)")
                gr.Markdown("Follows the launch script selected in the Train tab: its `model` folder (all tensor files, saved states "
                            "and the `sample` images) and its `log` folder.")
                with gr.Row():
                    mon_model = gr.Textbox(label="Model folder (output of the script)", value=default_sample_dir, scale=4)
                    ck_btn = gr.Button("Refresh files", scale=1)
                ck_md = gr.Markdown(checkpoints_md(""))
                with gr.Row():
                    sample_dir = gr.Textbox(label="Sample images folder", value=default_sample_dir, scale=5)
                    sample_btn = gr.Button("Refresh", scale=1)
                gallery = gr.Gallery(label="Samples", columns=4, height=520, object_fit="contain")
                gr.Markdown("### TensorBoard")
                with gr.Row():
                    tb_dir = gr.Textbox(label="Log folder (--logging_dir of your script)",
                                        value=str(ROOT / "logs"), scale=4)
                    tb_port = gr.Number(label="Port", value=6006, precision=0, scale=1)
                with gr.Row():
                    tb_start_btn = gr.Button("Start TensorBoard")
                    tb_stop_btn = gr.Button("Stop TensorBoard")
                tb_status = gr.Markdown("TensorBoard: not running")

            with gr.Tab("Settings"):
                gr.Markdown(f"Model, VAE and text-encoder files for each LoRA type, saved in `{settings_path()}`. For every file: "
                            "type or paste a folder (press Enter), then pick a file from the list. New launch scripts use "
                            "these files instead of the preset defaults.")
                with gr.Accordion("GPU: GFX version (HSA_OVERRIDE_GFX_VERSION)", open=True):
                    gr.Markdown("Change this when you change GPU. It is saved in settings.txt under `[general]`, used by the GUI itself, "
                                "put into every new launch script, and can be written into existing scripts with the button below.")
                    with gr.Row():
                        gx_pick = gr.Dropdown(choices=[(f"{lab}: {val}", val) for lab, val in KNOWN_GFX], label="Known GPUs",
                                              scale=4)
                        gx_val = gr.Textbox(label="GFX version", info="e.g. 12.0.1", scale=1)
                        gx_detect = gr.Button("Detect", scale=1)
                        gx_save_btn = gr.Button("Save GFX", variant="primary", scale=1)
                    with gr.Row():
                        gx_confirm = gr.Checkbox(label="Yes, rewrite the export line in all my launch scripts (a .bak is kept)", value=False, scale=3)
                        gx_upd_btn = gr.Button("Update all launch scripts", scale=1)
                    gx_msg = gr.Markdown("")
                _st_types = type_choices()
                with gr.Row():
                    st_type = gr.Dropdown(choices=_st_types, value=_st_types[0][1] if _st_types else None,
                                          label="LoRA type", scale=3)
                    st_rec = gr.Checkbox(label="Also list files in subfolders (2 levels)", value=True, scale=2)
                    st_reload = gr.Button("Reload from settings.txt", scale=1)
                    st_refresh_btn = gr.Button("Refresh file lists", scale=1)
                st_info = gr.Markdown("")
                st_flags = gr.State([""] * ST_N)
                st_rows, st_dirs, st_files = [], [], []
                for i in range(ST_N):
                    with gr.Row(visible=False) as _row:
                        _d = gr.Textbox(label="Folder", scale=2)
                        _f = gr.Dropdown(choices=[], label="File", allow_custom_value=True, scale=3)
                    st_rows.append(_row)
                    st_dirs.append(_d)
                    st_files.append(_f)
                with gr.Row():
                    st_save_btn = gr.Button("Save to settings.txt", variant="primary")
                st_msg = gr.Markdown("")
                with gr.Accordion("settings.txt as it is on disk", open=False):
                    st_raw_box = gr.Textbox(value=st_raw(), lines=14, interactive=False, show_label=False)

        # events
        script_change_outputs = [state_dir, state_dd, state_msg, mon_model, sample_dir, ck_md, tb_dir]
        script_dd.change(on_script_change, script_dd, script_change_outputs).then(latest_images, sample_dir, gallery)
        refresh_btn.click(refresh_scripts, None, script_dd)
        find_btn.click(find_states_ui, [script_dd, state_dir], [state_dd, state_msg])
        start_btn.click(start_run, [script_dd, workdir, venv, trace, state_dd], None)
        stop_btn.click(stop_run, None, None)
        cl_prev_btn.click(cleanup_preview, [script_dd, cl_samples], cl_msg)
        cl_btn.click(cleanup_run, [script_dd, cl_samples, cl_confirm], cl_msg).then(lambda: False, None, cl_confirm)

        # create data launch script tab
        cs_outs = [cs_script, cs_type, cs_text] + cs_fields + [cs_explain, cs_prompts_path, cs_prompts_text,
                                                                cs_newname, cs_save_over_btn, cs_status, cs_plan, cs_samples]
        cs_ctl = [cs_ds, cs_script, cs_type, cs_steps, cs_userep, cs_hfauto, cs_samples]
        cs_ret = [cs_ds, cs_script, cs_type, cs_text, cs_steps, cs_userep, cs_hfauto, cs_samples]
        rp_outs = [rp_folder, rp_repeats, rp_newname, rp_info]
        rp_in = [cs_ds, cs_type, cs_steps]
        cs_ds.input(cs_ds_change, [cs_ds, cs_type, cs_steps, cs_userep, cs_hfauto, cs_samples], cs_outs).then(rp_refresh, rp_in, rp_outs)
        cs_script.input(cs_script_change, cs_ctl, cs_outs)
        cs_type.input(cs_retype, cs_ret, cs_outs).then(rp_refresh, rp_in, rp_outs)
        cs_steps.submit(cs_retype, cs_ret, cs_outs).then(rp_refresh, rp_in, rp_outs)
        cs_userep.input(cs_retype, cs_ret, cs_outs)
        cs_hfauto.input(cs_retype, cs_ret, cs_outs)
        cs_samples.input(cs_samples_toggle, [cs_ds, cs_script, cs_type, cs_text, cs_plan, cs_steps, cs_userep, cs_hfauto, cs_samples], cs_outs)
        rp_make_btn.click(rp_make, [cs_ds, cs_type, cs_text, cs_plan], rp_msg).then(rp_refresh, rp_in, rp_outs)
        cs_suggest_btn.click(cs_suggest, cs_ctl, cs_outs).then(rp_refresh, rp_in, rp_outs)
        rp_folder.input(rp_name_update, [cs_ds, rp_folder, rp_repeats], rp_newname)
        rp_repeats.input(rp_name_update, [cs_ds, rp_folder, rp_repeats], rp_newname)
        rp_btn.click(rp_rename, [cs_ds, rp_folder, rp_newname, rp_confirm], [rp_msg, cs_userep]) \
            .then(cs_retype, cs_ret, cs_outs).then(rp_refresh, rp_in, rp_outs)
        rp_epochs_btn.click(cs_set_epochs, [cs_ds, cs_type, cs_script, cs_text, cs_plan, cs_steps], cs_outs)
        cs_hf_show.input(lambda show: gr.update(type="text" if show else "password"), cs_hf_show, cs_hf_tok)
        cs_hf_reload.click(lambda: (read_hf_token(), hf_status_text()), None, [cs_hf_tok, hf_md])
        cs_hf_fill.click(cs_fill_hf, [cs_text, cs_plan], cs_outs)
        cs_read_btn.click(cs_read, [cs_ds, cs_type, cs_script, cs_text, cs_plan], cs_outs)
        cs_apply_btn.click(cs_apply, [cs_ds, cs_type, cs_script, cs_text, cs_plan] + cs_fields, cs_outs)
        cs_models_btn.click(cs_use_settings, [cs_ds, cs_type, cs_script, cs_text, cs_plan], cs_outs)
        cs_check_btn.click(cs_check, cs_text, cs_check_out)
        cs_save_over_btn.click(cs_save_over, [cs_ds, cs_script, cs_text, cs_type, cs_plan, cs_makedirs], cs_status) \
            .then(rp_refresh, rp_in, rp_outs)
        cs_save_new_btn.click(cs_save_new, [cs_ds, cs_newname, cs_text, cs_prompts_path, cs_prompts_text,
                                            cs_type, cs_script, cs_plan, cs_makedirs], cs_outs + [script_dd]) \
            .then(rp_refresh, rp_in, rp_outs)
        cs_prompts_save.click(cs_save_prompts, [cs_prompts_path, cs_prompts_text], cs_prompts_msg)
        cs_refresh.click(scan_library, [cs_ds, rv_ds], [cs_ds, rv_ds, cs_lib_md, cs_lib])

        # settings tab
        st_slot_outs = []
        for _r, _d, _f in zip(st_rows, st_dirs, st_files):
            st_slot_outs += [_r, _d, _f]
        st_outs = st_slot_outs + [st_flags, st_info]
        st_type.input(st_load, [st_type, st_rec], st_outs)
        st_reload.click(st_load, [st_type, st_rec], st_outs)
        demo.load(st_load, [st_type, st_rec], st_outs)
        st_rec.input(st_refresh_all, [st_rec] + st_dirs + st_files, st_files)
        st_refresh_btn.click(st_refresh_all, [st_rec] + st_dirs + st_files, st_files)
        for _d, _f in zip(st_dirs, st_files):
            _d.submit(st_refresh, [_d, st_rec, _f], _f)
            _d.blur(st_refresh, [_d, st_rec, _f], _f)
        st_save_btn.click(st_save, [st_type, st_flags] + st_dirs + st_files, [st_msg, st_raw_box])
        gx_pick.input(gfx_pick, gx_pick, gx_val)
        gx_detect.click(detect_gfx, None, [gx_val, gx_msg])
        gx_save_btn.click(gfx_save, gx_val, [gx_msg, st_raw_box])
        gx_upd_btn.click(gfx_update_scripts, gx_confirm, gx_msg).then(lambda: False, None, gx_confirm)
        demo.load(gfx_load, None, [gx_val, gx_pick, gx_msg])

        # review tab
        rv_filters = [rv_ds, rv_sub, rv_filter, rv_missing, rv_size]
        rv_outs = [rv_gallery, rv_paths, rv_page, rv_pageinfo]
        rv_ds.input(rv_ds_change, rv_ds, [rv_sub, rv_summary]).then(rv_first, rv_filters, rv_outs)
        rv_refresh_btn.click(rv_refresh_dirs, None, rv_ds)
        rv_sub.input(rv_first, rv_filters, rv_outs)
        rv_filter.submit(rv_first, rv_filters, rv_outs)
        rv_missing.input(rv_first, rv_filters, rv_outs)
        rv_size.input(rv_first, rv_filters, rv_outs)
        rv_prev_btn.click(rv_prev, rv_filters + [rv_page], rv_outs)
        rv_next_btn.click(rv_next, rv_filters + [rv_page], rv_outs)
        rv_gallery.select(rv_select, rv_paths, [rv_img, rv_selinfo, rv_caption, rv_sel])
        rv_save_btn.click(rv_save_caption, [rv_sel, rv_caption], rv_msg).then(
            review_page, rv_filters + [rv_page], rv_outs)
        rv_undo_btn.click(rv_undo, None, rv_msg).then(review_page, rv_filters + [rv_page], rv_outs)
        bk_args = [rv_ds, rv_sub, bk_scope, bk_find, bk_repl, bk_mode, bk_ic]
        bk_preview_btn.click(bulk_preview, bk_args, bk_out)
        bk_apply_btn.click(bulk_apply, bk_args, bk_out).then(review_page, rv_filters + [rv_page], rv_outs)
        bk_undo_btn.click(rv_undo, None, bk_out).then(review_page, rv_filters + [rv_page], rv_outs)
        st_btn.click(rv_stats, [rv_ds, rv_sub, bk_scope], st_out)

        sample_btn.click(latest_images, sample_dir, gallery)
        ck_btn.click(checkpoints_md, mon_model, ck_md)
        mon_model.submit(checkpoints_md, mon_model, ck_md)
        tb_start_btn.click(tb_start, [tb_dir, tb_port], tb_status)
        tb_stop_btn.click(tb_stop_ui, None, tb_status)

        demo.load(on_script_change, script_dd, script_change_outputs).then(latest_images, sample_dir, gallery)

        gr.Timer(1.0).tick(tick, seen, [status, log, seen, progress])
        gr.Timer(3.0).tick(header_status, None, [gpu_md, hf_md])
        gr.Timer(20.0).tick(latest_images, sample_dir, gallery)
        gr.Timer(20.0).tick(checkpoints_md, mon_model, ck_md)
    return demo


def detect_venv() -> str:
    for c in (ROOT / "venv", HOME / "sd-script" / "venv"):
        if (c / "bin" / "activate").is_file():
            return str(c)
    return ""


def main():
    global ROOT, DEFAULT_VENV, SCAN_SCRIPT
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HOME / "sd-scripts"), help="your sd-scripts folder")
    ap.add_argument("--scripts-dir", action="append", default=[],
                    help="extra folder searched recursively for *.sh (repeatable)")
    ap.add_argument("--venv", help="venv folder to pre-fill (default: <root>/venv, else ~/sd-script/venv)")
    ap.add_argument("--scan-script", help="path to scan_lora_dataset_ny.py (default: searched next to sdgui.py, in <root>, in ~)")
    ap.add_argument("--sample-dir", help="folder scanned for sample images (default: <root>/output)")
    ap.add_argument("--allow-path", action="append", default=[],
                    help="extra folder the gallery may serve images from (repeatable)")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    ROOT = Path(os.path.expanduser(args.root)).absolute()
    EXTRA_SCRIPT_DIRS.extend(Path(os.path.expanduser(d)).absolute() for d in args.scripts_dir)
    DEFAULT_VENV = os.path.expanduser(args.venv) if args.venv else detect_venv()
    SCAN_SCRIPT = find_scan_script(args.scan_script)
    apply_gfx_env()
    sample_dir = args.sample_dir or str(ROOT / "output")
    allowed = [str(ROOT)] + [str(d) for d in EXTRA_SCRIPT_DIRS] + [os.path.expanduser(p) for p in args.allow_path] \
        + [os.path.expanduser(sample_dir)]

    demo = build_ui(sample_dir)
    allowed.append(str(CACHE_DIR))
    kw = {"css": CSS} if LAUNCH_HAS_CSS else {}
    demo.launch(server_name="127.0.0.1", server_port=args.port, allowed_paths=allowed, **kw)


if __name__ == "__main__":
    main()