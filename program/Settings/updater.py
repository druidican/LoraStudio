# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Updates for sd-scripts and musubi-tuner (Settings > Updates).

* Checking: `git fetch` in both folders (quietly at start-up, or with the button); shows how many commits each
  is behind its branch, the newest commit messages and whether the requirements changed.
* Updating (your click, never automatic), for both trainers together because they share one venv:
    1. refused while a training runs, or when a folder has local changes to tracked files
    2. baseline tests: GPU, bitsandbytes, the trainers' --help (only tests that pass now can fail the update)
    3. snapshot: the commits and every installed package version (kept in update_backups/<time>/)
    4. git pull --ff-only on the branch you have
    5. pip dry run of both trainers' requirements with PyTorch / ROCm locked: a conflict stops here
    6. the real install, then the tests again
    7. anything fails -> back to the old commits and the old package versions, tests once more
    8. success -> constraints.txt is written again (PyTorch locked as before, trainer libraries at their new versions)
"""
from __future__ import annotations
__version__ = "1.011126"  # LoRA Studio version, see version.py

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import lspaths as _LP  # noqa: E402  (program/lspaths.py: ROOT, PROG, USER)
APP_DIR = _LP.ROOT                    # venv/, the trainers, constraints.txt
BACKUP_DIR = _LP.USER / "update_backups"
CONSTRAINTS = APP_DIR / "constraints.txt"
TORCH_RE = r"(torch|torchvision|torchaudio|triton|pytorch-triton-rocm|rocm[a-z0-9_-]*)"
TRAINER_LIBS_RE = r"(transformers|diffusers|accelerate|huggingface[-_]hub|safetensors|tokenizers|bitsandbytes)"
REQ_FILES = ("requirements.txt", "pyproject.toml", "setup.py", "setup.cfg")
HELP_TESTS = (("sd-scripts", "sdxl_train_network.py"), ("sd-scripts", "flux_train_network.py"),
              ("musubi-tuner", "src/musubi_tuner/zimage_train_network.py"))

STATUS: dict = {"checked": "", "repos": [], "running": False}
FAILED_FILE = BACKUP_DIR / "failed.json"     # {repo name: {"upstream": commit, "why": text, "at": time}}


def _failed() -> dict:
    try:
        return json.loads(FAILED_FILE.read_text())
    except Exception:
        return {}


def _set_failed(data: dict):
    try:
        BACKUP_DIR.mkdir(exist_ok=True)
        FAILED_FILE.write_text(json.dumps(data, indent=1))
    except OSError:
        pass
LOCK = threading.Lock()          # one check / update at a time


# --------------------------------------------------------------------------- #
# Where things are
# --------------------------------------------------------------------------- #

def repos() -> list[tuple[str, Path]]:
    sg = sys.modules.get("sdgui")
    sd = mu = None
    try:
        if sg is not None:
            sd, mu = Path(sg.sd_root()), Path(sg.musubi_root())
    except Exception:
        pass
    return [("sd-scripts", sd or APP_DIR / "sd-scripts"), ("musubi-tuner", mu or APP_DIR / "musubi-tuner")]


def venv_python() -> Path:
    """The shared venv made by install.sh (APP_DIR/venv), else the Python running the program."""
    p = APP_DIR / "venv" / "bin" / "python"
    return p if p.exists() else Path(sys.executable)


def shared_venv() -> bool:
    return (APP_DIR / "venv" / ".lorastudio-shared").exists()


def training_running() -> bool:
    sg = sys.modules.get("sdgui")
    try:
        return bool(sg is not None and sg.RUN.running())
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def git(d: Path, *args, timeout: int = 120) -> tuple[int, str]:
    try:
        r = subprocess.run(["git", "-C", str(d), *args], capture_output=True, text=True, timeout=timeout,
                           env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
        return r.returncode, (r.stdout + r.stderr).strip()
    except FileNotFoundError:
        return 127, "git is not installed"
    except subprocess.TimeoutExpired:
        return 124, f"git {' '.join(args)} took longer than {timeout} s (no network?)"


def packages(py: Path) -> dict:
    """name (lower case) -> version of every installed distribution."""
    code = ("import json\nfrom importlib.metadata import distributions\nout={}\n"
            "for d in distributions():\n n=(d.metadata['Name'] or '').strip().lower()\n"
            " if n and n not in out: out[n]=d.version\nprint(json.dumps(out))")
    r = subprocess.run([str(py), "-c", code], capture_output=True, text=True, timeout=120)
    return json.loads(r.stdout or "{}")


def _norm(n: str) -> str:
    return re.sub(r"[-_.]+", "-", n.lower())


def torch_lines(pk: dict) -> list[str]:
    """PyTorch / ROCm lines from constraints.txt, else from the installed packages."""
    pat = re.compile(TORCH_RE, re.I)
    lines = []
    if CONSTRAINTS.exists():
        for l in CONSTRAINTS.read_text().splitlines():
            name = l.split("==")[0].strip()
            if "==" in l and not l.startswith("#") and pat.fullmatch(name):
                lines.append(l.strip())
    if not lines:
        lines = [f"{n}=={v}" for n, v in sorted(pk.items()) if pat.fullmatch(n)]
    return lines


def run_stream(cmd: list, cwd: Path | None, log: list, timeout: int = 3600, label: str = ""):
    """Run a command, appending its output to `log` line by line; yields while it runs; returns the exit code."""
    log.append(f"$ {label or ' '.join(str(c) for c in cmd)}")
    try:
        p = subprocess.Popen([str(c) for c in cmd], cwd=str(cwd) if cwd else None, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1,
                             env={**os.environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1", "PYTHONUNBUFFERED": "1"})
    except FileNotFoundError as e:
        log.append(f"   {e}")
        return 127
    t0 = time.time()
    last = 0.0
    for line in p.stdout:
        line = line.rstrip()
        if line:
            log.append("   " + line)
        if time.time() - last > 0.7:
            last = time.time()
            yield
        if time.time() - t0 > timeout:
            p.kill()
            log.append(f"   stopped: took longer than {timeout} s")
            break
    p.wait()
    yield
    return p.returncode


def _drain(gen):
    """Run a run_stream() generator to the end inside another generator: `rc = yield from _drain(...)`."""
    rc = yield from gen
    return rc


# --------------------------------------------------------------------------- #
# Check
# --------------------------------------------------------------------------- #

def check_repo(name: str, d: Path, fetch: bool = True) -> dict:
    st = {"name": name, "dir": str(d), "ok": False, "behind": 0, "log": [], "dirty": False, "req_changed": False,
          "error": "", "branch": "", "head": "", "upstream": ""}
    if not (d / ".git").exists():
        st["error"] = "not a git folder (installed without git?)"
        return st
    rc, br = git(d, "rev-parse", "--abbrev-ref", "HEAD")
    st["branch"] = br if rc == 0 else ""
    rc, up = git(d, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    if rc != 0:
        st["error"] = f"branch '{st['branch']}' follows no remote branch"
        return st
    st["upstream"] = up
    if fetch:
        rc, out = git(d, "fetch", "--quiet", up.split("/")[0], timeout=90)
        if rc != 0:
            st["error"] = "could not reach GitHub: " + (out.splitlines()[-1] if out else f"git exit {rc}")
            return st
    st["head"] = git(d, "log", "-1", "--format=%h %cd", "--date=short")[1]
    st["up_commit"] = git(d, "rev-parse", "@{u}")[1]
    rc, n = git(d, "rev-list", "--count", "HEAD..@{u}")
    st["behind"] = int(n) if rc == 0 and n.isdigit() else 0
    if st["behind"]:
        st["log"] = git(d, "log", "--format=%h %cd %s", "--date=short", "-n", "12", "HEAD..@{u}")[1].splitlines()
        changed = git(d, "diff", "--name-only", "HEAD..@{u}")[1].splitlines()
        st["req_changed"] = any(Path(f).name in REQ_FILES for f in changed)
    st["dirty"] = bool(git(d, "status", "--porcelain", "--untracked-files=no")[1])
    st["ok"] = True
    return st


def check(fetch: bool = True) -> dict:
    with LOCK:
        STATUS["running"] = True
        try:
            STATUS["repos"] = [check_repo(n, d, fetch) for n, d in repos()]
            STATUS["checked"] = time.strftime("%Y-%m-%d %H:%M")
        finally:
            STATUS["running"] = False
    return STATUS


_BG: threading.Thread | None = None


def start_background_check():
    """Quiet check at start-up (only fetches; never installs)."""
    global _BG
    if _BG is None:
        def go():
            try:
                check()
                n = sum(r.get("behind", 0) for r in STATUS["repos"])
                if n:
                    print(f"[Updates] {n} new commit(s) for the trainers: see Settings > Updates", flush=True)
            except Exception as e:
                print(f"[Updates] check failed: {e!r}", flush=True)
        _BG = threading.Thread(target=go, daemon=True, name="update-check")
        _BG.start()
    return _BG


def status_md() -> str:
    if STATUS["running"] and not STATUS["repos"]:
        return "Checking for updates ..."
    if not STATUS["repos"]:
        return "Not checked yet: press **Check for updates**."
    lines = [f"Checked {STATUS['checked']}."]
    any_new = False
    for r in STATUS["repos"]:
        head = f"**{r['name']}** (`{r['dir']}`)"
        if r["error"]:
            lines.append(f"- {head}: ⚠ {r['error']}")
            continue
        if not r["behind"]:
            lines.append(f"- {head}: ✔ up to date ({r['branch']}, {r['head']})")
            continue
        any_new = True
        extra = " · **its requirements changed**" if r["req_changed"] else ""
        dirty = " · ⚠ has local changes to its files: it will be skipped" if r["dirty"] else ""
        f = _failed().get(r["name"])
        if f and f.get("upstream") == r.get("up_commit"):
            dirty += (f"\n    - ⚠ **Updating to this version failed on {f.get('at')} and was rolled back**: {f.get('why')}. "
                      "Wait for a newer commit, or press Update now to try again.")
        lines.append(f"- {head}: 🆕 **{r['behind']} new commit{'s' if r['behind'] != 1 else ''}** on {r['upstream']} "
                     f"(you have {r['head']}){extra}{dirty}")
        lines += [f"    - `{l}`" for l in r["log"][:8]]
        if len(r["log"]) > 8 or r["behind"] > 8:
            lines.append(f"    - … {r['behind'] - min(8, len(r['log']))} more")
    if any_new:
        lines.append("\nPress **Update now** to install. Both trainers share one venv: PyTorch / ROCm stay as they are, "
                     "and if a test fails afterwards everything goes back to how it was.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

GPU_TEST = ("import torch\nok = torch.cuda.is_available()\nprint('torch', torch.__version__, 'GPU', ok)\n"
            "x = torch.randn(256, 256, device='cuda', dtype=torch.bfloat16) if ok else None\n"
            "assert ok and torch.isfinite(x @ x).all()\n")


def run_tests(py: Path, roots: dict, log: list):
    """Yields while running; returns {test name: passed}."""
    res = {}
    rc = yield from _drain(run_stream([py, "-c", GPU_TEST], None, log, timeout=300, label="test: PyTorch on the GPU"))
    res["PyTorch on the GPU"] = rc == 0
    rc = yield from _drain(run_stream([py, "-c", "import bitsandbytes as b; print('bitsandbytes', b.__version__)"],
                                      None, log, timeout=300, label="test: bitsandbytes loads"))
    res["bitsandbytes"] = rc == 0
    for name, script in HELP_TESTS:
        d = roots.get(name)
        if not d or not (d / script).exists():
            continue
        out: list = []
        rc = yield from _drain(run_stream([py, script, "--help"], d, out, timeout=300))
        res[f"{name}: {Path(script).name} starts"] = rc == 0
        log.append(f"$ {name}: {script} --help -> {'ok' if rc == 0 else 'FAILED'}")
        if rc != 0:
            log += out[-6:]
    return res


# --------------------------------------------------------------------------- #
# Update
# --------------------------------------------------------------------------- #

def _install_cmds(py: Path, roots: dict, cons: Path, dry: bool) -> list[tuple[list, Path | None]]:
    extra = ["--dry-run"] if dry else []
    cmds = []
    sd = roots.get("sd-scripts")
    if sd and (sd / "requirements.txt").exists():
        cmds.append(([py, "-m", "pip", "install", *extra, "-r", "requirements.txt", "-c", cons], sd))
    mu = roots.get("musubi-tuner")
    if mu and ((mu / "pyproject.toml").exists() or (mu / "setup.py").exists()):
        cmds.append(([py, "-m", "pip", "install", *extra, "-e", str(mu), "-c", cons], None))
    return cmds


def update():
    """Generator for the Update now button: yields (log text, status markdown)."""
    log: list[str] = []

    def out(msg=None):
        if msg:
            log.append(msg)
        return "\n".join(log[-400:]), status_md()

    if not LOCK.acquire(blocking=False):
        yield out("Another check or update is running: wait for it to finish.")
        return
    try:
        if training_running():
            yield out("❌ A training is running: update when it has finished (the trainers are in use).")
            return
        if not shared_venv():
            yield out("❌ Updating from here needs the install.sh layout (one shared venv in the LoRA Studio folder). "
                      "Run ./install.sh once; it also updates both trainers.")
            return
        py = venv_python()
        yield out("Checking GitHub ...")
        STATUS["repos"] = [check_repo(n, d) for n, d in repos()]
        STATUS["checked"] = time.strftime("%Y-%m-%d %H:%M")
        todo = [r for r in STATUS["repos"] if r["ok"] and r["behind"] and not r["dirty"]]
        for r in STATUS["repos"]:
            if r["ok"] and r["behind"] and r["dirty"]:
                log.append(f"⚠ {r['name']}: local changes to its files, not updated (git -C {r['dir']} status)")
            if r["error"]:
                log.append(f"⚠ {r['name']}: {r['error']}")
        if not todo:
            yield out("✔ Nothing to update.")
            return
        roots = {n: Path(d) for n, d in repos()}

        # 2. baseline tests
        yield out("\n== Tests before updating (only these may fail the update)")
        base = yield from _with_out(run_tests(py, roots, log), out)
        log.append("   " + ", ".join(f"{k}: {'ok' if v else 'fails already'}" for k, v in base.items()))

        # 3. snapshot
        stamp = time.strftime("%Y%m%d-%H%M%S")
        bdir = BACKUP_DIR / stamp
        old = sorted(d for d in BACKUP_DIR.glob("2*") if d.is_dir()) if BACKUP_DIR.exists() else []
        for d in old[:-9]:                   # keep the last 10 snapshots
            import shutil
            shutil.rmtree(d, ignore_errors=True)
        bdir.mkdir(parents=True, exist_ok=True)
        before = packages(py)
        heads = {r["name"]: git(Path(r["dir"]), "rev-parse", "HEAD")[1] for r in todo}
        (bdir / "packages.json").write_text(json.dumps(before, indent=1, sort_keys=True))
        (bdir / "commits.json").write_text(json.dumps(heads, indent=1))
        if CONSTRAINTS.exists():
            (bdir / "constraints.txt").write_text(CONSTRAINTS.read_text())
        yield out(f"\n== Snapshot saved in {bdir}")

        cons = bdir / "torch-lock.txt"
        cons.write_text("\n".join(torch_lines(before)) + "\n")

        def rollback(why: str):
            log.append(f"\n== ❌ {why}: going back to how it was")
            fails = _failed()
            for r in todo:
                fails[r["name"]] = {"upstream": r.get("up_commit"), "why": why, "at": time.strftime("%Y-%m-%d %H:%M")}
            _set_failed(fails)
            for r in todo:
                rc, o = git(Path(r["dir"]), "reset", "--hard", heads[r["name"]])
                log.append(f"   {r['name']}: back to {heads[r['name']][:9]}" + ("" if rc == 0 else f" (FAILED: {o})"))
            yield
            now = packages(py)
            changed = [f"{n}=={v}" for n, v in before.items() if now.get(n) != v and not re.fullmatch(TORCH_RE, n)
                       and n != "musubi-tuner"]
            added = [n for n in now if n not in before]
            if added:
                yield from _drain(run_stream([py, "-m", "pip", "uninstall", "-y", *added], None, log))
            if changed:
                yield from _drain(run_stream([py, "-m", "pip", "install", "--no-deps", *changed, "-c", cons], None, log))
            mu = roots.get("musubi-tuner")
            if mu and any(r["name"] == "musubi-tuner" for r in todo):
                yield from _drain(run_stream([py, "-m", "pip", "install", "--no-deps", "-e", str(mu)], None, log))
            left = {n: v for n, v in before.items() if packages(py).get(n) != v and n != "musubi-tuner"}
            log.append("   packages restored" if not left else f"   ⚠ not restored: {left}")
            res = yield from _drain(run_tests(py, roots, log))
            bad = [k for k, v in res.items() if base.get(k) and not v]
            log.append("   ✔ tests pass again: everything is as before" if not bad
                       else f"   ⚠ still failing after the rollback: {', '.join(bad)} (snapshot: {bdir})")

        # 4. pull
        yield out("\n== Getting the new versions")
        for r in todo:
            rc, o = git(Path(r["dir"]), "merge", "--ff-only", "@{u}")
            log.append(f"   {r['name']}: " + (git(Path(r['dir']), 'log', '-1', '--format=%h %cd %s', '--date=short')[1]
                                              if rc == 0 else f"FAILED: {o}"))
            if rc != 0:
                yield from _with_out(rollback(f"git pull failed for {r['name']}"), out)
                yield out()
                return
        yield out()

        # 5. dry run
        yield out("\n== Dry run of the new requirements (PyTorch / ROCm locked)")
        would = []
        for cmd, cwd in _install_cmds(py, roots, cons, dry=True):
            n0 = len(log)
            rc = yield from _with_out(run_stream(cmd, cwd, log), out)
            would += [l.split("Would install", 1)[1].strip() for l in log[n0:] if "Would install" in l]
            if rc != 0:
                yield from _with_out(rollback("the new requirements cannot be installed (pip dry run failed, see above)"), out)
                yield out()
                return
        pk = " ".join(would).split()
        if any(re.fullmatch(TORCH_RE + r"-.*", p, re.I) for p in pk):
            yield from _with_out(rollback("the update would replace PyTorch / ROCm"), out)
            yield out()
            return
        log.append("   would change: " + (", ".join(pk) if pk else "no packages"))
        yield out()

        # 6. install + tests
        if pk:
            yield out("\n== Installing")
            for cmd, cwd in _install_cmds(py, roots, cons, dry=False):
                rc = yield from _with_out(run_stream(cmd, cwd, log), out)
                if rc != 0:
                    yield from _with_out(rollback("pip install failed"), out)
                    yield out()
                    return
        else:
            # editable musubi-tuner: refresh its entry points without touching other packages
            mu = roots.get("musubi-tuner")
            if mu and any(r["name"] == "musubi-tuner" for r in todo):
                yield from _with_out(run_stream([py, "-m", "pip", "install", "-q", "--no-deps", "-e", str(mu)], None, log), out)
        yield out("\n== Tests after updating")
        res = yield from _with_out(run_tests(py, roots, log), out)
        bad = [k for k, v in res.items() if base.get(k) and not v]
        if bad:
            yield from _with_out(rollback("failed after the update: " + ", ".join(bad)), out)
            yield out()
            return

        # 7. constraints
        after = packages(py)
        lib_pat = re.compile(TRAINER_LIBS_RE, re.I)
        text = ("# Written by install.sh (updated by LoRA Studio > Settings > Updates " + stamp + "):\n"
                "# versions that later 'pip install' runs must keep (pip install ... -c constraints.txt)\n"
                + "\n".join(torch_lines(after)) + "\n"
                + "\n".join(f"{n}=={v}" for n, v in sorted(after.items()) if lib_pat.fullmatch(n)) + "\n")
        CONSTRAINTS.write_text(text)
        fails = _failed()
        for r in todo:
            fails.pop(r["name"], None)
        _set_failed(fails)
        diff = [f"{n} {before.get(n, '-')} → {v}" for n, v in sorted(after.items()) if before.get(n) != v]
        STATUS["repos"] = [check_repo(n, d, fetch=False) for n, d in repos()]
        log.append("\n== ✔ Updated. " + ("Package changes: " + "; ".join(diff) if diff else "No package changes."))
        log.append("   constraints.txt now locks the new trainer library versions (PyTorch unchanged). "
                   "New trainings use the new versions; restart LoRA Studio if a package of its own changed.")
        yield out()
    except Exception as e:                     # noqa: BLE001
        yield out(f"❌ The update stopped with an error: {e!r}. Snapshots are in {BACKUP_DIR}.")
    finally:
        LOCK.release()


def _with_out(gen, out):
    """Run a sub-generator that yields None, passing the current log to the UI each time; returns its value."""
    while True:
        try:
            next(gen)
        except StopIteration as stop:
            return stop.value
        yield out()


# --------------------------------------------------------------------------- #
# UI (Settings > Updates)
# --------------------------------------------------------------------------- #

