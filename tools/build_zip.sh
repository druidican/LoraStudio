#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
# Build dist/LoraStudio.zip from this repository: the files in install.sh's APP_FILES, the Windows installer
# and program/. Nothing else goes in (no tools/, no CLAUDE.md, no userdata/, no tokens).
set -e
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$SRC/dist/LoraStudio.zip}"
B="$(mktemp -d)"; trap 'rm -rf "$B"' EXIT
mkdir -p "$B/LoraStudio" "$(dirname "$OUT")"
FILES=$(bash -c "source <(sed -n '/^APP_FILES=\"/,/\"\$/p' '$SRC/install.sh'); echo \$APP_FILES")
for f in $FILES; do cp -p "$SRC/$f" "$B/LoraStudio/"; done
cp -p "$SRC/install-windows.ps1" "$SRC/Install LoRA Studio.cmd" "$B/LoraStudio/"
cp -r "$SRC/program" "$B/LoraStudio/"
find "$B" -name __pycache__ -prune -exec rm -rf {} +
chmod +x "$B/LoraStudio/install.sh" "$B/LoraStudio/start.sh"
for bad in huggingtoken.txt github_token.txt; do
    if find "$B" -name "$bad" | grep -q .; then echo "STOP: $bad would go into the zip" >&2; exit 1; fi
done
rm -f "$OUT"; (cd "$B" && zip -qr -X "$OUT" LoraStudio)
echo "Built $OUT ($(unzip -l "$OUT" | tail -1 | awk '{print $2}') files)"
