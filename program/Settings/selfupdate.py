# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Updates of LoRA Studio itself (Settings › Updates › LoRA Studio): install a newer LoraStudio.zip.

    1. Read the zip: its version.py says the version, the parts and the version of each part (nothing in it is run).
    2. Show what would change: the parts of another version, the new parts, the ones that stay, what the update says.
    3. Install (your click): refused while a training run, the queue or the automatic improvement is busy.
       Only the program files are written (the parts listed in its version.py, requirements.txt, README.md);
       your settings, databases, token, logs and venv are never touched. The files it replaces are kept first
       in update_backups/lorastudio_<old version>_<time>/.
    4. The self-test (selftest.py) runs with the new files. When it fails, the old files go back by themselves.
    5. Restart LoRA Studio (button) to use the new version. Roll back puts the files of before the last update back.
"""
from __future__ import annotations
__version__ = "1.011126"  # LoRA Studio version, see version.py

import ast
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

import lspaths as _LP  # noqa: E402  (program/lspaths.py: ROOT, PROG, USER)
HERE = _LP.ROOT                      # the files of a zip are placed relative to the LoRA Studio folder
BACKUPS = _LP.USER / "update_backups"
PREFIX = "lorastudio_"
EXTRA_FILES = ("requirements.txt", "README.md", "INSTALL.md", "REFERENCE.md", "LICENSE", "NOTICE")       # copied too, not in the parts list
DIRS = ("program/manual", "program/logo")             # folders that are copied as a whole (the manual, the logo)
STATE = {"busy": False}


def repair_dirs(say=print) -> str:
    """The manual and the logo came with 1.251026d; an update from an older LoRA Studio copies only single files.
    When those folders are missing, take them from a LoraStudio.zip of this version (in the program folder,
    update_backups/downloads/, ~/Downloads or ~)."""
    import version as V
    missing = [d for d in DIRS if not (HERE / d).is_dir()]
    extra = [f for f in EXTRA_FILES if not (HERE / f).is_file()]     # LICENSE, REFERENCE.md: new in 1.251026g
    if not missing and not extra:
        return ""
    places = [HERE, BACKUPS / "downloads", Path.home() / "Downloads", Path.home()]
    zips = []
    for d in places:
        try:
            zips += [z for z in d.glob("LoraStudio*.zip") if z.is_file()]
        except OSError:
            pass
    for z in sorted(zips, key=lambda f: f.stat().st_mtime, reverse=True):
        try:
            info = read_zip(z)
        except Exception:                                # noqa: BLE001
            continue
        if info["version"] != V.VERSION:
            continue
        n = 0
        with zipfile.ZipFile(z) as zz:
            for name in info["dir_files"]:
                if any(name.startswith(d + "/") for d in missing):
                    dest = HERE / name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(zz.read(info["root"] + name))
                    n += 1
            for name in extra:
                if name in info["names"]:
                    (HERE / name).write_bytes(zz.read(info["root"] + name))
                    n += 1
        msg = f"{', '.join(missing + extra)} ({n} files) taken from {z.name}"
        say(f"[Updates] {msg}")
        return msg
    if not missing:                                      # only LICENSE / REFERENCE.md: not worth a warning
        return ""
    msg = ("the manual and the logo are missing: install LoraStudio.zip " + V.VERSION +
           " once more in Settings › Updates (or unpack its manual/ and logo/ folders here)")
    say(f"[Updates] {msg}")
    return msg


def _literal(src: str, name: str):
    """The value of `name = <literal>` (or `name: type = <literal>`) in Python source, without running it."""
    tree = ast.parse(src)
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets):
            return ast.literal_eval(n.value)
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id == name and n.value:
            return ast.literal_eval(n.value)
    return None


def read_zip(path) -> dict:
    """{"version", "parts": [(name, what)], "release": {name: version}, "changes": [(v, text)], "root": prefix,
    "names": set of file names in the zip root}. Raises ValueError when it is not a LoraStudio.zip."""
    p = Path(path)
    if not zipfile.is_zipfile(p):
        raise ValueError("This is not a zip file.")
    with zipfile.ZipFile(p) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        vp = next((n for n in names if n.endswith("version.py") and n.count("/") <= 1), None)
        if not vp:
            raise ValueError("No version.py in the zip: this is not a LoraStudio.zip of version 1.241026 or later.")
        root = vp[: -len("version.py")]
        src = z.read(vp).decode("utf-8", "replace")
    version = _literal(src, "VERSION")
    parts = _literal(src, "PARTS") or []
    if not version or not parts:
        raise ValueError("The version.py in the zip has no VERSION or PARTS.")
    return {"version": version, "parts": parts, "release": _literal(src, "RELEASE") or {},
            "changes": _literal(src, "CHANGES") or [], "root": root,
            "names": {n[len(root):] for n in names if n.startswith(root) and ".." not in n.split("/")},
            "dir_files": sorted(n[len(root):] for n in names if n.startswith(root)
                                and n[len(root):].startswith(tuple(d + "/" for d in DIRS)) and ".." not in n.split("/"))}


def _vkey(v: str):
    """'1.241026b' -> (1, 2026-10-24, 'b') for comparing versions."""
    m = re.match(r"^(\d+)\.(\d{2})(\d{2})(\d{2})([a-z]*)$", v or "")
    if not m:
        return (0, "", "")
    rel, dd, mm, yy, letter = m.groups()
    return (int(rel), f"{yy}{mm}{dd}", letter)


def plan(path) -> dict:
    """What installing the zip would do: {"zip": read_zip(...), "rows": [{name, what, have, new, state}],
    "newer": bool, "same": bool, "text": markdown}."""
    import version as V
    info = read_zip(path)
    rows = []
    for name, what in info["parts"]:
        have = V.version_of(name)
        new = info["release"].get(name, info["version"])
        in_zip = name in info["names"]
        state = ("missing in zip" if not in_zip else "new" if have is None else
                 "same" if have == new and _same_file(path, info, name) else "changed")
        rows.append({"name": name, "what": what, "have": have or "", "new": new, "state": state})
    newer = _vkey(info["version"]) > _vkey(V.VERSION)
    same = info["version"] == V.VERSION
    return {"zip": info, "rows": rows, "newer": newer, "same": same}


def _same_file(path, info, name) -> bool:
    try:
        with zipfile.ZipFile(path) as z:
            return z.read(info["root"] + name) == (HERE / name).read_bytes()
    except (KeyError, OSError):
        return False


def busy_reason() -> str:
    sg = sys.modules.get("sdgui")
    try:
        if sg is not None and sg.RUN.running():
            return "a training run is active"
        if sg is not None and sg.QUEUE.get("running"):
            return "the training queue is running"
    except Exception:                                    # noqa: BLE001
        pass
    try:
        import dsflow
        if dsflow.BATCH.get("running"):
            return "the automatic improvement is running"
    except Exception:                                    # noqa: BLE001
        pass
    return ""


def install(path, say=print) -> bool:
    """Install the zip (see the top of this file). Returns True when the new files are in place and tested."""
    import version as V
    if STATE["busy"]:
        say("An update is being installed already.")
        return False
    why = busy_reason()
    if why:
        say(f"Not now: {why}. Install the update when it has finished.")
        return False
    try:
        pl = plan(path)
    except Exception as e:                               # noqa: BLE001
        say(f"❌ {e}")
        return False
    info = pl["zip"]
    todo = [r["name"] for r in pl["rows"] if r["state"] in ("changed", "new")]
    with zipfile.ZipFile(path) as z:
        dir_todo = [n for n in info.get("dir_files", []) if not (HERE / n).is_file()
                    or (HERE / n).read_bytes() != z.read(info["root"] + n)]
    if not todo and not dir_todo:
        say(f"Nothing to install: the files of {info['version']} are the ones you have.")
        return False
    todo += [n for n in EXTRA_FILES if n in info["names"]]
    todo += dir_todo
    STATE["busy"] = True
    stamp = time.strftime("%Y%m%d_%H%M%S")
    bdir = BACKUPS / f"{PREFIX}{V.VERSION}_{stamp}"
    manifest = {"from": V.VERSION, "to": info["version"], "at": time.strftime("%Y-%m-%d %H:%M"), "replaced": [],
                "added": []}
    try:
        bdir.mkdir(parents=True, exist_ok=True)
        say(f"== Backup of the files that change → {bdir.relative_to(HERE)}/")
        for n in todo:
            f = HERE / n
            if f.is_file():
                (bdir / n).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, bdir / n)
                manifest["replaced"].append(n)
            else:
                manifest["added"].append(n)
        (bdir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        say(f"== Writing {len(todo)} files of LoRA Studio {info['version']}")
        with zipfile.ZipFile(path) as z:
            for n in todo:
                data = z.read(info["root"] + n)
                (HERE / n).parent.mkdir(parents=True, exist_ok=True)
                tmp = (HERE / n).with_name(f".{Path(n).name}.new")
                tmp.write_bytes(data)
                if n.endswith(".sh"):
                    tmp.chmod(0o755)
                tmp.replace(HERE / n)
                if not n.startswith(DIRS):
                    say(f"   {n}")
        if dir_todo:
            say(f"   + {len(dir_todo)} files of the manual and the logo")
        say("== Self-test with the new files")
        r = subprocess.run([sys.executable, str(HERE / "selftest.py")], cwd=str(HERE), capture_output=True, text=True,
                           timeout=600)
        out = (r.stdout or "") + (r.stderr or "")
        for line in out.strip().splitlines()[-16:]:
            say("   " + line)
        if r.returncode != 0:
            say("❌ The self-test failed: the old files go back.")
            _restore(bdir, say)
            return False
        req = "requirements.txt" in todo and "requirements.txt" in manifest["replaced"] and \
            (bdir / "requirements.txt").read_bytes() != (HERE / "requirements.txt").read_bytes()
        say(f"✔ LoRA Studio {info['version']} is installed and tested. Restart LoRA Studio to use it"
            + (" (the program packages changed: start.sh installs them when it starts)." if req else "."))
        return True
    except Exception as e:                               # noqa: BLE001
        say(f"❌ {e}: the old files go back.")
        _restore(bdir, say)
        return False
    finally:
        STATE["busy"] = False


def _restore(bdir: Path, say=print) -> str:
    try:
        m = json.loads((bdir / "manifest.json").read_text(encoding="utf-8"))
    except Exception:                                    # noqa: BLE001
        return "No manifest in the backup."
    for n in m.get("replaced", []):
        shutil.copy2(bdir / n, HERE / n)
    for n in m.get("added", []):
        (HERE / n).unlink(missing_ok=True)
    say(f"   back to LoRA Studio {m.get('from')} ({len(m.get('replaced', []))} files put back, "
        f"{len(m.get('added', []))} new ones removed)")
    try:
        (bdir / "manifest.json").rename(bdir / "manifest.rolled_back.json")
    except OSError:
        pass
    return m.get("from", "")


def backups() -> list[dict]:
    """[{dir, from, to, at}] of the updates that can be rolled back, newest first."""
    out = []
    for d in sorted(BACKUPS.glob(f"{PREFIX}*"), reverse=True):
        try:
            m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
            out.append({"dir": d, **m})
        except Exception:                                # noqa: BLE001
            continue
    return out


def rollback(say=print) -> bool:
    b = backups()
    if not b:
        say("There is no update to roll back.")
        return False
    why = busy_reason()
    if why:
        say(f"Not now: {why}.")
        return False
    _restore(b[0]["dir"], say)
    say("✔ Restart LoRA Studio to use the old version again.")
    return True


def restart(delay: float = 1.5) -> str:
    """Start LoRA Studio again in this process (through start.sh, so new packages get installed)."""
    why = busy_reason()
    if why:
        return f"Not now: {why}."
    start = HERE / "start.sh"
    args = [a for a in sys.argv[1:] if a != "--setup"]
    if "--no-browser" not in args:
        args.append("--no-browser")                      # the page that is open reconnects by itself

    def go():
        time.sleep(delay)
        sys.stdout.flush()
        if start.is_file() and not os.environ.get("LS_RESTART_DIRECT"):
            os.execv("/bin/bash", ["bash", str(start)] + args)
        os.execv(sys.executable, [sys.executable, str(HERE / "lora_studio.py")] + args)
    threading.Thread(target=go, daemon=True, name="restart").start()
    return "Restarting LoRA Studio … the page reloads by itself when it is back (about 10-30 seconds)."
