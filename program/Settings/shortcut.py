# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Settings › Start and network › Shortcut: LoRA Studio in the applications menu (Linux desktops: GNOME, KDE …).

The menu entry (~/.local/share/applications/lorastudio.desktop) runs userdata/lorastudio-launch.sh: when LoRA Studio
answers at its port, it only opens the page; otherwise it starts ./start.sh quietly (no terminal; the output goes to
userdata/logs/shortcut-start.log), waits until the page answers and opens it. The port is read from
launch_settings.json each time. refresh() at every start keeps the paths right after the folder was moved.
"""
from __future__ import annotations
__version__ = "1.231126"  # LoRA Studio version, see version.py

import os
import shutil
import subprocess
import sys
from pathlib import Path

import lspaths as LP

NAME = "lorastudio.desktop"
ICON = LP.PROG / "logo" / "lorastudio_icon_256.png"
LAUNCHER = LP.USER / "lorastudio-launch.sh"


def apps_dir() -> Path:
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "applications"


def entry() -> Path:
    return apps_dir() / NAME


def possible() -> tuple[bool, str]:
    """(can make one here, why not)."""
    if not sys.platform.startswith("linux"):
        return False, "The applications-menu shortcut is for Linux desktops."
    if "microsoft" in (Path("/proc/version").read_text(errors="ignore").lower() if Path("/proc/version").exists() else ""):
        return False, "Under WSL, start LoRA Studio from the Windows shortcut the installer made."
    return True, ""


def exists() -> bool:
    return entry().is_file()


def _q(s: str) -> str:
    """For single quotes in bash."""
    return "'" + str(s).replace("'", "'\\''") + "'"


def _desk(path) -> str:
    """A path as one argument of a .desktop Exec line (double quotes; \\ " ` $ escaped, % doubled)."""
    t = str(path)
    for c in ("\\", '"', "`", "$"):
        t = t.replace(c, "\\" + c)
    return '"' + t.replace("%", "%%") + '"'


def launcher_text() -> str:
    root = LP.ROOT
    return f"""#!/bin/bash
# Made by LoRA Studio (Settings > Start and network > Shortcut); made again at every start, do not edit.
# Opens LoRA Studio in the browser; starts it first (quietly) when it is not running.
ROOT={_q(root)}
OPTS={_q(LP.USER / "launch_settings.json")}
PORT="$(grep -o '"port": *[0-9]*' "$OPTS" 2>/dev/null | grep -o '[0-9]*$')"
URL="http://127.0.0.1:${{PORT:-7860}}/"
up() {{ curl -s -o /dev/null --max-time 2 "$URL" 2>/dev/null || wget -q -O /dev/null --timeout=2 "$URL" 2>/dev/null \\
       || python3 -c "import urllib.request,sys; urllib.request.urlopen(sys.argv[1], timeout=2)" "$URL" 2>/dev/null; }}
if ! up; then
  mkdir -p {_q(LP.USER / "logs")}
  command -v notify-send >/dev/null && notify-send -i {_q(ICON)} "LoRA Studio" "Starting … the page opens when it is ready."
  cd "$ROOT" && setsid nohup ./start.sh --no-browser >> {_q(LP.USER / "logs" / "shortcut-start.log")} 2>&1 < /dev/null &
  for i in $(seq 1 240); do up && break; sleep 1; done
fi
xdg-open "$URL" >/dev/null 2>&1 &
"""


def entry_text() -> str:
    return f"""[Desktop Entry]
Type=Application
Version=1.0
Name=LoRA Studio
GenericName=LoRA training
Comment=Datasets, LoRA training and your LoRA collection
Exec={_desk(LAUNCHER)}
Icon={ICON}
Terminal=false
Categories=Graphics;Development;
Keywords=LoRA;training;Stable Diffusion;ComfyUI;dataset;
StartupNotify=false
"""


def _write():
    LAUNCHER.parent.mkdir(parents=True, exist_ok=True)
    LAUNCHER.write_text(launcher_text(), encoding="utf-8")
    LAUNCHER.chmod(0o755)
    apps_dir().mkdir(parents=True, exist_ok=True)
    entry().write_text(entry_text(), encoding="utf-8")
    entry().chmod(0o755)
    if shutil.which("update-desktop-database"):
        subprocess.run(["update-desktop-database", str(apps_dir())], capture_output=True, timeout=20)


def make() -> str:
    ok, why = possible()
    if not ok:
        return "❌ " + why
    try:
        _write()
    except OSError as e:
        return f"❌ Could not make the shortcut: {e}"
    return ("✔ LoRA Studio is in your applications menu (search for LoRA Studio; right-click it in the dock to pin "
            "it). It opens the page, and starts LoRA Studio first when it is not running.")


def remove() -> str:
    try:
        entry().unlink(missing_ok=True)
        LAUNCHER.unlink(missing_ok=True)
    except OSError as e:
        return f"❌ Could not remove the shortcut: {e}"
    return "✔ The shortcut is removed from the applications menu."


def refresh():
    """At start: an existing shortcut gets the current paths (the folder may have been moved)."""
    if exists() and possible()[0]:
        try:
            if entry().read_text(encoding="utf-8") != entry_text() or not LAUNCHER.is_file() \
                    or LAUNCHER.read_text(encoding="utf-8") != launcher_text():
                _write()
        except OSError:
            pass
