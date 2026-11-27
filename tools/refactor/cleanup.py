# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Remove what the rewrite left unused: `S = _sg()` lines, lone local `import x` lines, names in from-import lists."""
import re, subprocess, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
PY = sys.executable
def flakes(paths):
    r = subprocess.run([PY, "-m", "pyflakes", *paths], capture_output=True, text=True, cwd=REPO)
    return r.stdout.splitlines()
paths = sys.argv[1:]
for _ in range(6):
    out = flakes(paths)
    todo = {}
    for line in out:
        m = re.match(r"(.+?):(\d+):\d+:? (.*)", line)
        if not m: continue
        f, ln, msg = m.group(1), int(m.group(2)), m.group(3)
        todo.setdefault(f, []).append((ln, msg))
    changed = False
    for f, items in todo.items():
        lines = (REPO / f).read_text(encoding="utf-8").splitlines(keepends=True)
        dels = set()
        for ln, msg in items:
            L = lines[ln - 1]
            mv = re.match(r"local variable '(\w+)' is assigned to but never used", msg)
            if mv and re.fullmatch(rf"\s*{mv.group(1)} = _?\w*\(\)\s*(#.*)?\n?", L):
                dels.add(ln)
            mi = re.match(r"'([\w.]+)(?: as (\w+))?' imported but unused", msg)
            if mi:
                nm = mi.group(1)
                if re.fullmatch(rf"\s*import {re.escape(nm)}( as \w+)?\s*(#.*)?\n?", L):
                    dels.add(ln)
                else:
                    # a name inside a parenthesised from-import list: its own line "    name,"
                    short = nm.split(".")[-1]
                    for k in range(ln - 1, min(ln + 200, len(lines))):
                        if re.fullmatch(rf"\s*{re.escape(short)},\s*\n?", lines[k]):
                            dels.add(k + 1); break
                        if lines[k].strip() == ")": break
        if dels:
            lines = [l for i, l in enumerate(lines, 1) if i not in dels]
            (REPO / f).write_text("".join(lines), encoding="utf-8")
            changed = True
    if not changed: break
for l in flakes(paths):
    print(l)
