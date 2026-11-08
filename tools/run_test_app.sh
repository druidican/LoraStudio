#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
# Start a throw-away copy of LoRA Studio for testing: its own HOME (so no real settings are touched) and
# the pretend Ollama.   tools/run_test_app.sh <work folder> [port, default 7931]
# The work folder gets app/ (a copy of this repo) and home/. Stop it with: fuser -k <port>/tcp
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
W="$(mkdir -p "$1" && cd "$1" && pwd)"; PORT="${2:-7931}"; OPORT=$((PORT + 3570))
PY="${PY:-$( [ -x "$ROOT/venv/bin/python" ] && echo "$ROOT/venv/bin/python" || echo python3 )}"
fuser -k "$PORT/tcp" "$OPORT/tcp" 2>/dev/null || true
rm -rf "$W/app"; mkdir -p "$W/app" "$W/home"
cp -a "$ROOT"/. "$W/app/"; rm -rf "$W/app/.git" "$W/app/userdata"
(nohup "$PY" "$ROOT/tools/fake_ollama.py" "$OPORT" "$W/app" > "$W/ollama.log" 2>&1 &)
(cd "$W/app" && HOME="$W/home" OLLAMA_HOST="http://127.0.0.1:$OPORT" nohup "$PY" lora_studio.py --port "$PORT" --no-browser > "$W/server.log" 2>&1 &)
for i in $(seq 1 60); do curl -s -o /dev/null "http://127.0.0.1:$PORT/" && break; sleep 1; done
echo "LoRA Studio test copy: http://127.0.0.1:$PORT  (pretend Ollama on $OPORT; set it in Settings › AI helpers)"
tail -3 "$W/server.log"
