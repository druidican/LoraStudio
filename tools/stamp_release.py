# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Stamp a release: parts that changed since the previous release get the new version, the others keep theirs.
Usage: stamp_release.py <src> <previous release folder> <new version> <change text>"""
import re, sys, importlib.util
from pathlib import Path
src, old, new, text = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
spec = importlib.util.spec_from_file_location("v", src / "version.py"); v = importlib.util.module_from_spec(spec); spec.loader.exec_module(v)
PY = re.compile(r'^(__version__\s*=\s*")([^"]+)(")', re.M)
TX = re.compile(r"(LoRA Studio version\s+)([0-9][0-9A-Za-z.]*)")
def norm(t):  # content without the version stamps (and without version.py's RELEASE / CHANGES / VERSION lines)
    t = PY.sub(r"\1X\3", t); t = TX.sub(r"\1X", t)
    return t
def stamp(p, ver):
    t = p.read_text(encoding="utf-8")
    t2 = PY.sub(lambda m: m.group(1) + ver + m.group(3), t, count=1) if p.suffix == ".py" else TX.sub(lambda m: m.group(1) + ver, t, count=1)
    p.write_text(t2, encoding="utf-8")
release = {}
for name, _ in v.PARTS:
    f, o = src / name, old / name
    if name == "version.py":
        release[name] = new; continue
    if o.is_file() and norm(o.read_text(encoding="utf-8")) == norm(f.read_text(encoding="utf-8")):
        ov = (PY if name.endswith(".py") else TX).search(o.read_text(encoding="utf-8"))
        ver = ov.group(2) if ov else new
    else:
        ver = new
    stamp(f, ver)
    release[name] = ver
vp = src / "version.py"; t = vp.read_text(encoding="utf-8")
t = PY.sub(lambda m: m.group(1) + new + m.group(3), t, count=1)
t = re.sub(r'^VERSION = "[^"]+"', f'VERSION = "{new}"', t, count=1, flags=re.M)
body = ",\n".join(f'    "{k}": "{val}"' for k, val in release.items())
t = re.sub(r"^RELEASE: dict = \{.*?\}\n", "RELEASE: dict = {\n" + body + ",\n}\n", t, count=1, flags=re.M | re.S)
if f'("{new}",' not in t:
    t = t.replace("CHANGES = [\n", f'CHANGES = [\n    ("{new}", "{text}"),\n', 1)
vp.write_text(t, encoding="utf-8")
ch = [k for k, val in release.items() if val == new]
print(len(ch), "changed:", ", ".join(ch))
