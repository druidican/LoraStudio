#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
# LoRA Studio version 1.251026n
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
cd "$(dirname "$(readlink -f "$0")")" || exit 1

VENV="./venv"
PY="$VENV/bin/python"
STAMP="$VENV/.requirements.sha256"

SHARED="$VENV/.lorastudio-shared"      # written by install.sh
CONS=()
[ -f constraints.txt ] && CONS=(-c constraints.txt)

for f in lora_studio.py sdgui.py lora_library.py setup.py completed.py model_guide.py captioner.py housekeeping.py datacheck.py picgallery.py updater.py backup.py firstrun.py statusbar.py savepoint.py launchopts.py consistency.py picker.py trim.py ollamactl.py ticker.py settings_ui.py training_ui.py datasets_ui.py dsflow.py studio_css.py lora_studio.css comfytest.py pictools.py pictools_worker.py version.py sd_script.py sd_prepare.py sd_settings.py selftest.py machinecheck.py selfupdate.py failcodes.py report.py manual.py features.py venvcheck.py notice.py Lora_Datatypes.py Training_Recommendations.py requirements.txt; do
    if [ ! -f "$f" ]; then
        echo "Missing $f - put all the files from LoraStudio.zip in the same folder."
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

# 1. Create the venv if it does not exist (or is broken)
if [ ! -x "$PY" ]; then
    if [ -f install.sh ]; then
        echo "No Python environment yet. Recommended: ./install.sh  (sets up PyTorch, sd-scripts and musubi-tuner here)."
        echo "Creating a small GUI-only environment instead in 5 seconds (Ctrl+C to stop)..."
        sleep 5
    fi
    echo "First start: creating the Python environment in $VENV ..."
    if ! command -v python3 >/dev/null 2>&1; then
        echo "python3 was not found. On Fedora: sudo dnf install python3"
        exit 1
    fi
    rm -rf "$VENV"
    if ! python3 -m venv "$VENV"; then
        echo "Could not create the venv. On Fedora: sudo dnf install python3-virtualenv"
        exit 1
    fi
    "$PY" -m pip install --upgrade pip >/dev/null
fi

# 2. Install / update the packages when requirements.txt is new or changed
WANT="$(sha256sum requirements.txt | cut -d' ' -f1)"
HAVE="$(cat "$STAMP" 2>/dev/null)"
if [ "$WANT" != "$HAVE" ] || ! "$PY" -c "import gradio, pandas, PIL" >/dev/null 2>&1; then
    echo "Installing packages from requirements.txt (only needed once) ..."
    if ! "$PY" -m pip install -r requirements.txt "${CONS[@]}"; then
        echo
        echo "Package installation failed. Check your internet connection and run ./start.sh again."
        exit 1
    fi
    echo "$WANT" > "$STAMP"
fi

# 3. First start (no settings yet) or --setup: LoRA Studio opens its setup page in the browser
#    (folders, token, Ollama, or import a backup). Without a browser: python setup.py asks in the terminal.

echo "Starting LoRA Studio with $("$PY" -c 'import sys, gradio; print(f"Python {sys.version.split()[0]}, Gradio {gradio.__version__}")')"
exec "$PY" lora_studio.py "${ARGS[@]}"
