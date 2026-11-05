#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
# LoRA Studio version 1.051126
# Start LoRA Studio (Training LoRA + Personal Data + Settings) and open it in the browser.
#
# Installed with ./install.sh (recommended): uses the shared ./venv (Python 3.12 + PyTorch + sd-scripts +
# musubi-tuner + the GUI). Without install.sh, the first start creates a small GUI-only ./venv instead
# (training then needs your own sd-scripts / musubi-tuner). Packages are only reinstalled when
# requirements.txt has changed, always within the versions locked in constraints.txt.
#
# The first start opens a setup page in the browser: folders, models, ComfyUI, Hugging Face token, Ollama,
# backup folder (every step can be skipped), or import a backup. Show it again any time: ./start.sh --setup
#
# Extra options are passed on, e.g.:  ./start.sh --port 7861   or   ./start.sh --no-browser
# Rebuild a GUI-only environment from scratch:  ./start.sh --rebuild-venv
# (the shared environment from install.sh is rebuilt with:  ./install.sh --reinstall-torch)

set -o pipefail

# Work from the folder this script is in (also when started from a desktop shortcut)
cd "$(dirname "$0")" && cd "$(pwd -P)" || exit 1
HERE="$(pwd)"
uv_from_github() {   # uv_from_github DIR: the uv program straight from its GitHub release (when astral.sh is not reachable)
    local a t
    case "$(uname -s)-$(uname -m)" in
        Linux-x86_64) t=x86_64-unknown-linux-gnu ;; Linux-aarch64|Linux-arm64) t=aarch64-unknown-linux-gnu ;;
        Darwin-arm64) t=aarch64-apple-darwin ;; Darwin-x86_64) t=x86_64-apple-darwin ;; *) return 1 ;;
    esac
    a="https://github.com/astral-sh/uv/releases/latest/download/uv-$t.tar.gz"
    mkdir -p "$1"
    if command -v curl >/dev/null 2>&1; then curl -LsSf "$a" | tar -xz -C "$1" --strip-components=1
    else wget -qO- "$a" | tar -xz -C "$1" --strip-components=1; fi
}
sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }

VENV="./venv"
PY="$VENV/bin/python"
STAMP="$VENV/.requirements.sha256"

SHARED="$VENV/.lorastudio-shared"      # written by install.sh
CONS=()
[ -f constraints.txt ] && CONS=(-c constraints.txt)

for f in lora_studio.py version.py selftest.py requirements.txt program/lspaths.py program/Training/sdgui.py; do
    if [ ! -f "$f" ]; then
        if [ "$f" = program/lspaths.py ] && [ -f sdgui.py ]; then
            echo "This LoRA Studio has the new folder layout (program/, userdata/), but the program/ folder is missing."
            echo "Unpack the whole LoraStudio.zip and run ./install.sh once: your settings move to userdata/ by themselves."
        else
            echo "Missing $f - put all the files from LoraStudio.zip in the same folder."
        fi
        exit 1
    fi
done

# --rebuild-venv: throw the environment away and make a fresh one
ARGS=()
for a in "$@"; do
    if [ "$a" = "--rebuild-venv" ]; then
        if [ -f "$SHARED" ]; then
            echo "$VENV is the shared training environment from install.sh (PyTorch, sd-scripts, musubi-tuner)."
            echo "Rebuild it with:  ./install.sh --reinstall-torch"
            exit 1
        fi
        echo "Removing $VENV to rebuild it..."
        rm -rf "$VENV"
    else
        ARGS+=("$a")
    fi
done

# 0. The LoRA Studio folder was moved or renamed: the Python environment remembers where it was made, so the paths
#    in it (and in the settings that point at the trainers in this folder) are put right before anything starts.
OLD_ROOT="$(cat "$VENV/.lorastudio-root" 2>/dev/null)"
if [ -n "$OLD_ROOT" ] && [ "$OLD_ROOT" != "$HERE" ] && [ -d "$VENV" ]; then
    echo "LoRA Studio was moved from $OLD_ROOT to $HERE: updating the paths in the Python environment ..."
    for f in "$HERE"/.tools/python/*; do          # uv's short-version links inside .tools
        [ -L "$f" ] || continue
        t="$(readlink "$f")"
        case "$t" in "$OLD_ROOT"/*) ln -sfn "$HERE${t#$OLD_ROOT}" "$f" ;; esac
    done
    for f in "$VENV"/bin/*; do
        if [ -L "$f" ]; then                       # python -> .../.tools/python/... (made by uv)
            t="$(readlink "$f")"
            case "$t" in "$OLD_ROOT"/*) ln -sfn "$HERE${t#$OLD_ROOT}" "$f" ;; esac
        elif [ -f "$f" ] && head -c 200 "$f" 2>/dev/null | grep -q "$OLD_ROOT"; then
            sed -i.bak "s#$OLD_ROOT#$HERE#g" "$f" 2>/dev/null && rm -f "$f.bak"   # #! lines and activate scripts
        fi
    done
    [ -f "$VENV/pyvenv.cfg" ] && sed -i.bak "s#$OLD_ROOT#$HERE#g" "$VENV/pyvenv.cfg" && rm -f "$VENV/pyvenv.cfg.bak"
    old_t="${OLD_ROOT/#$HOME/~}"; new_t="${HERE/#$HOME/~}"
    for f in userdata/training_settings.txt userdata/library_settings.txt; do
        [ -f "$f" ] && sed -i.bak "s#$OLD_ROOT/#$HERE/#g; s#$old_t/#$new_t/#g" "$f" && rm -f "$f.bak"
    done
    echo "$HERE" > "$VENV/.lorastudio-root"
fi

# 1. Create the venv if it does not exist (or is broken): with LoRA Studio's own Python 3.12 from uv (in .tools/, no admin
#    rights, the system Python is not touched); only when uv cannot be downloaded the computer's python3.12 / python3
if [ -e "$VENV" ] && ! "$PY" -c 'import sys' >/dev/null 2>&1; then
    if [ -f "$SHARED" ]; then                     # the training environment (PyTorch, trainers): never thrown away here
        echo "The training environment in $VENV does not start. Run ./install.sh (it repairs it; your settings stay)."
        exit 1
    fi
    echo "The Python environment in $VENV does not work any more: making a new one ..."
    rm -rf "$VENV"
fi
if [ ! -x "$PY" ]; then
    if [ -f install.sh ]; then
        echo "No Python environment yet. Recommended: ./install.sh  (sets up PyTorch, sd-scripts and musubi-tuner here)."
        echo "Creating a small GUI-only environment instead in 5 seconds (Ctrl+C to stop)..."
        sleep 5
    fi
    echo "First start: creating the Python 3.12 environment in $VENV ..."
    export UV_PYTHON_INSTALL_DIR="$HERE/.tools/python" UV_CACHE_DIR="$HERE/.tools/cache" UV_NO_MODIFY_PATH=1
    UV=""
    for f in "$HERE/.tools/uv" "$HERE/.tools/bin/uv"; do [ -x "$f" ] && UV="$f" && break; done
    if [ -z "$UV" ]; then
        mkdir -p "$HERE/.tools"
        if command -v curl >/dev/null 2>&1; then curl -LsSf https://astral.sh/uv/install.sh 2>/dev/null | env UV_UNMANAGED_INSTALL="$HERE/.tools" sh >/dev/null 2>&1
        elif command -v wget >/dev/null 2>&1; then wget -qO- https://astral.sh/uv/install.sh 2>/dev/null | env UV_UNMANAGED_INSTALL="$HERE/.tools" sh >/dev/null 2>&1
        fi
        [ -x "$HERE/.tools/uv" ] || uv_from_github "$HERE/.tools" >/dev/null 2>&1
        [ -x "$HERE/.tools/uv" ] && UV="$HERE/.tools/uv"
    fi
    rm -rf "$VENV"
    if [ -n "$UV" ] && "$UV" venv --python 3.12 --python-preference only-managed --seed "$VENV" >/dev/null 2>&1; then
        :
    else
        SYS=""
        for c in python3.12 python3; do command -v "$c" >/dev/null 2>&1 && SYS="$c" && break; done
        if [ -z "$SYS" ]; then
            echo "Could not download Python 3.12 (no internet?) and no python3 was found. Connect to the internet and try again."
            exit 1
        fi
        echo "uv could not be used: making the environment with $SYS ($($SYS --version 2>&1))."
        if ! "$SYS" -m venv "$VENV"; then
            echo "Could not create the venv. Install the venv package of your Python (e.g. sudo apt install python3.12-venv)."
            exit 1
        fi
    fi
    "$PY" -m pip install --upgrade pip >/dev/null
fi
[ -f "$VENV/.lorastudio-root" ] || echo "$HERE" > "$VENV/.lorastudio-root"

# 2. Install / update the packages when requirements.txt is new or changed
WANT="$(sha256 requirements.txt)"
HAVE="$(cat "$STAMP" 2>/dev/null)"
if [ "$WANT" != "$HAVE" ] || ! "$PY" -c "import gradio, pandas, PIL" >/dev/null 2>&1; then
    echo "Installing packages from requirements.txt (only needed once) ..."
    if ! "$PY" -m pip install --upgrade -r requirements.txt "${CONS[@]}"; then
        echo
        echo "Package installation failed. Check your internet connection and run ./start.sh again."
        exit 1
    fi
    echo "$WANT" > "$STAMP"
fi

# 3. First start (no settings yet) or --setup: LoRA Studio opens its setup page in the browser
#    (folders, token, Ollama, or import a backup). Without a browser: venv/bin/python program/Settings/setup.py asks in the terminal.

echo "Starting LoRA Studio with $("$PY" -c 'import sys, gradio; print(f"Python {sys.version.split()[0]}, Gradio {gradio.__version__}")')"
exec "$PY" lora_studio.py "${ARGS[@]}"
