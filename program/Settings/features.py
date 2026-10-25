# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""What is installed: Ollama, sd-scripts, musubi-tuner. A part that is missing has its buttons hidden, so nothing
can be started by mistake; the steps, tabs and names stay, greyed out with "(Ollama missing)" and so on.

features.json next to the program (written by install.sh, changed in Settings › Start and network):
    {"ollama": "installed" | "remote" | "none", "sd_scripts": "installed" | "existing" | "none",
     "musubi": "installed" | "existing" | "none", "platform": "linux" | "wsl"}
Without the file (an older install) everything counts as present, as before. A change needs a restart, because the
page is built once at start.
"""
from __future__ import annotations
__version__ = "1.251026o"  # LoRA Studio version, see version.py

import json
import os
import shutil
from pathlib import Path

import lspaths as _LP  # noqa: E402  (program/lspaths.py: ROOT, PROG, USER)
HERE = _LP.USER
FILE = HERE / "features.json"
PARTS = {"ollama": "Ollama", "sd_scripts": "sd-scripts", "musubi": "musubi-tuner"}
WHAT = {"ollama": "the detailed sort (pass 2), writing captions and the consistency check",
        "sd_scripts": "training SDXL, Illustrious, Pony, Anima, FLUX.1, SD 1.5, … and the picture tools",
        "musubi": "training Z-Image Turbo, Qwen-Image and FLUX.2"}
MUSUBI_TYPES = {"zimage", "qwenimage", "flux2"}
_CACHE: dict = {}


def load() -> dict:
    if "data" not in _CACHE:
        try:
            d = json.loads(FILE.read_text(encoding="utf-8"))
        except Exception:                                # noqa: BLE001
            d = {}
        _CACHE["data"] = d if isinstance(d, dict) else {}
    return dict(_CACHE["data"])


def save(**kw) -> dict:
    d = load()
    d.update({k: v for k, v in kw.items() if v is not None})
    FILE.write_text(json.dumps(d, indent=1), encoding="utf-8")
    _CACHE["data"] = d
    return d


def have(part: str) -> bool:
    """True unless features.json says the part is not installed ("none")."""
    if os.environ.get("LS_ALL_FEATURES"):
        return True
    return load().get(part, "") != "none"


def missing(part: str) -> str:
    """'' when the part is there, else '(Ollama missing)' etc."""
    return "" if have(part) else f"({PARTS[part]} missing)"


def label(text: str, part: str) -> str:
    """A label with '(… missing)' added when the part is missing."""
    m = missing(part)
    return f"{text} {m}" if m else text


def trainer_for(tkey: str) -> str:
    return "musubi" if tkey in MUSUBI_TYPES else "sd_scripts"


def can_train(tkey: str) -> bool:
    return have(trainer_for(tkey))


def can_train_text(text: str) -> bool:
    """A launch script can be run here: its trainer is installed."""
    return have("musubi") if "MUSUBI_DIR=" in (text or "") else have("sd_scripts")


def summary() -> list[tuple[str, str, bool]]:
    """[(name, what it is for, present)] for Settings."""
    return [(PARTS[k], WHAT[k], have(k)) for k in PARTS]


def is_wsl() -> bool:
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


OLLAMA_GUIDE_WINDOWS = "https://ollama.com/download/windows"
OLLAMA_GUIDE_LINUX = "https://ollama.com/download/linux"


def ollama_guide() -> str:
    return OLLAMA_GUIDE_WINDOWS if is_wsl() else OLLAMA_GUIDE_LINUX


def gpu_kind() -> str:
    """"amd" or "nvidia": which graphics card the trainings use. From features.json ("gpu", written by install.sh),
    else from the computer (/dev/kfd = AMD ROCm, nvidia-smi = NVIDIA); LS_GPU=amd|nvidia overrides it.
    Settings › Trainers and GPU shows the ROCm values for AMD and the CUDA values for NVIDIA."""
    if "gpu" in _CACHE:
        return _CACHE["gpu"]
    g = (os.environ.get("LS_GPU") or load().get("gpu") or "").lower()
    if g not in ("amd", "nvidia"):
        kfd, nv = Path("/dev/kfd").exists(), bool(shutil.which("nvidia-smi"))
        g = "nvidia" if nv and not kfd else "amd"       # both or neither: AMD, the setup it was made for
    _CACHE["gpu"] = g
    return g

