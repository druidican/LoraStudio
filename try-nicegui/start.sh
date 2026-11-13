#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
# The NiceGUI test of Collection › Browse. Its own venv (try-nicegui/venv): LoRA Studio's venv is not touched.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
cd "$HERE"
if [ ! -x venv/bin/python ]; then
    if [ -x "$ROOT/venv/bin/python" ]; then PY="$ROOT/venv/bin/python"; else PY="$(command -v python3)"; fi
    echo "First start: making a venv for the test with NiceGUI (a few minutes, about 300 MB) ..."
    "$PY" -m venv venv
    venv/bin/python -m pip install -q --upgrade pip
    venv/bin/python -m pip install -q nicegui gradio pandas pillow
fi
echo "Opening http://127.0.0.1:7870  (Ctrl+C here stops it)"
exec venv/bin/python browse.py
