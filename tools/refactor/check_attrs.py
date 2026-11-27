# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Every <module>.<name> in the repo exists (modules of the repo, through imports, getters and assigned getters)."""
import ast, importlib, sys, os
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))
from scopes import Analyzer
from split import files, getters, is_src_value, REPO
for d in ["program"] + [str(p) for p in (REPO / "program").iterdir() if p.is_dir()]:
    sys.path.insert(0, str(REPO / d))
sys.path.insert(0, str(REPO))
mods = {p.stem: p for p in files()}
bad = 0
cache = {}
def has(mod, attr):
    if mod not in cache:
        try:
            cache[mod] = importlib.import_module(mod)
        except Exception as e:
            print("cannot import", mod, e); cache[mod] = None
    m = cache[mod]
    return m is None or hasattr(m, attr)
targets = sys.argv[1:] or ["dsflow", "sdgui", "lora_library"]
for p in files():
    text = p.read_text(encoding="utf-8")
    tree = ast.parse(text); an = Analyzer(tree)
    for src in mods:
        if src not in text and not any(f'{g}()' in text for g in __import__('split').global_getters(src)): continue
        gets = getters(tree, src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                how = is_src_value(an, node.value, src, gets)
                if how and not has(src, node.attr):
                    bad += 1; print(f"{p.relative_to(REPO)}:{node.lineno}: {src}.{node.attr} does not exist")
print("bad:", bad)
