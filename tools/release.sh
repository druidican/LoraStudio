#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
# Make a release from this repository.
#   tools/release.sh <version> "<one line about what changed, for the user>"
# Version: 1.ddmmyy (day month year), plus b, c, ... for more releases the same day.
# 1. stamps every program part that changed since the last commit (HEAD = the previous release) with the new
#    version and adds the line to version.py CHANGES   2. runs selftest.py   3. builds dist/LoraStudio.zip
# It does not commit or push: check `git diff`, then commit and push yourself.
# Python for the selftest: $PY, else ./venv/bin/python, else python3 (it needs gradio, pandas, pillow).
set -e
[ $# -eq 2 ] || { sed -n '3,10p' "$0"; exit 1; }
VER="$1"; TEXT="$2"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
[[ "$VER" =~ ^1\.[0-9]{6}[a-z]?$ ]] || { echo "Version must look like 1.081126 or 1.081126b" >&2; exit 1; }
PREV="$(mktemp -d)"; trap 'rm -rf "$PREV"' EXIT
git archive HEAD | tar -x -C "$PREV"                  # the previous release, to see which parts changed
python3 tools/stamp_release.py "$ROOT" "$PREV" "$VER" "$TEXT"
PY="${PY:-$( [ -x venv/bin/python ] && echo venv/bin/python || echo python3 )}"
"$PY" selftest.py | tail -3
bash tools/build_zip.sh
echo "Now: git diff, then git commit -am \"$VER: ...\" and git push. Send dist/LoraStudio.zip to the owner."
