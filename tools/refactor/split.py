# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Move top-level names of a module into a new module and rewrite every caller (no re-exports left behind).

    python split.py move <src_mod> <new_mod> <names-file> [docstring-file]
    python split.py unexport <src_mod> <name> ...     (callers use the module the name comes from)
"""
import ast
import re
import sys
from pathlib import Path

from scopes import Analyzer

REPO = Path(__file__).resolve().parents[2]
ALIAS = {"dsflow": "F", "sdgui": "S", "lora_library": "LL"}
HEADER = "# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE\n"
NOTES: list[str] = []


def files():
    out = []
    for p in sorted(REPO.rglob("*.py")):
        rel = p.relative_to(REPO).as_posix()
        if rel.startswith(("venv/", ".tools/", "dist/", "userdata/", "musubi-tuner/", "sd-scripts/")) or "__pycache__" in rel:
            continue
        out.append(p)
    return out


def mod_path(mod: str) -> Path:
    hits = [p for p in files() if p.stem == mod]
    assert len(hits) == 1, (mod, hits)
    return hits[0]


# --------------------------------------------------------------------------- text edits
def line_offsets(text):
    offs, n = [0], 0
    for line in text.splitlines(keepends=True):
        n += len(line.encode("utf-8"))
        offs.append(n)
    return offs


class Edits:
    """Edits in (line, utf-8 col) coordinates applied at once."""

    def __init__(self, text):
        self.text = text
        self.b = text.encode("utf-8")
        self.offs = line_offsets(text)
        self.items = []

    def pos(self, line, col):
        return self.offs[line - 1] + col

    def replace(self, node, new):
        self.items.append((self.pos(node.lineno, node.col_offset), self.pos(node.end_lineno, node.end_col_offset), new))

    def insert_line_after(self, line, new_line):
        """Insert a whole line after line number `line` (1-based)."""
        p = self.offs[line]
        self.items.append((p, p, new_line.encode("utf-8") if isinstance(new_line, str) else new_line))

    def insert_line_before(self, line, new_line):
        p = self.offs[line - 1]
        self.items.append((p, p, new_line))

    def apply(self) -> str:
        b = self.b
        seen = set()
        for a, z, new in sorted(self.items, key=lambda t: (t[0], t[1]), reverse=True):
            key = (a, z, new if isinstance(new, str) else new.decode())
            if key in seen:
                continue
            seen.add(key)
            nb = new.encode("utf-8") if isinstance(new, str) else new
            b = b[:a] + nb + b[z:]
        return b.decode("utf-8")


# --------------------------------------------------------------------------- module facts
def top_defs(tree):
    """[(stmt, names)] of the module's top-level statements."""
    out = []
    for st in tree.body:
        names = set()
        if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(st.name)
        elif isinstance(st, ast.Assign):
            for t in st.targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        names.add(n.id)
        elif isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
            names.add(st.target.id)
        out.append((st, names))
    return out


def import_bindings(tree):
    """name -> (stmt, alias) for plain top-level imports (not inside try/if)."""
    out = {}
    for st in tree.body:
        if isinstance(st, (ast.Import, ast.ImportFrom)):
            for a in st.names:
                nm = a.asname or a.name.split(".")[0]
                out[nm] = (st, a)
    return out


def import_text(st, a) -> str:
    if isinstance(st, ast.Import):
        return f"import {a.name}" + (f" as {a.asname}" if a.asname else "")
    mod = "." * st.level + (st.module or "")
    return f"from {mod} import {a.name}" + (f" as {a.asname}" if a.asname else "")


def getters(tree, src):
    """Module-level functions that return the module `src` (sys.modules.get / __import__ / a local import)."""
    pat = re.compile(rf"""(sys\.modules(\.get\(|\[)['"]{src}['"](\)|\])|__import__\(['"]{src}['"]\))""")
    out = set()
    for st in tree.body:
        if not isinstance(st, ast.FunctionDef) or len(st.body) > 4:
            continue
        local_mod = {(a.asname or a.name) for n in ast.walk(st) if isinstance(n, ast.Import) for a in n.names if a.name == src}
        for r in ast.walk(st):
            if isinstance(r, ast.Return) and r.value is not None:
                vals = r.value.values if isinstance(r.value, ast.BoolOp) else [r.value]
                if any((isinstance(v, ast.Name) and v.id in local_mod) or pat.fullmatch(ast.unparse(v)) for v in vals):
                    out.add(st.name)
    return out


GLOBAL_GETS: dict[str, set] = {}


def global_getters(src):
    """Names of getter functions for src in any file (called as <module>.<name>() elsewhere)."""
    if src not in GLOBAL_GETS:
        out = set()
        for p in files():
            t = p.read_text(encoding="utf-8")
            if src in t:
                out |= getters(ast.parse(t), src)
        GLOBAL_GETS[src] = out
    return GLOBAL_GETS[src]


def _is_getter_call(v, src, gets):
    if not isinstance(v, ast.Call):
        return False
    if isinstance(v.func, ast.Name) and v.func.id in gets:
        return True
    return isinstance(v.func, ast.Attribute) and v.func.attr in global_getters(src)


def _import_bound(an, name_node, src):
    sc = an.resolve(name_node)
    return any(isinstance(b, ast.Import) and any(a.name == src and (a.asname or a.name) == name_node.id for a in b.names)
               for b in sc.bindings.get(name_node.id, []))


def _module_expr(an, v, src, gets, own):
    """The expression is the module itself (a getter call, an import-bound name, sys.modules / __import__)."""
    if _is_getter_call(v, src, gets):
        return True
    if isinstance(v, ast.Name) and v.id != own and _import_bound(an, v, src):
        return True
    if isinstance(v, ast.BoolOp):
        return any(_module_expr(an, x, src, gets, own) for x in v.values)
    if isinstance(v, ast.IfExp):
        return _module_expr(an, v.body, src, gets, own) or _module_expr(an, v.orelse, src, gets, own)
    t = ast.unparse(v)
    return bool(re.fullmatch(rf"""sys\.modules(\.get\(|\[)['"]{src}['"](\)|\])|__import__\(['"]{src}['"]\)""", t))


def is_src_value(an, node, src, gets):
    """How `node` (an Attribute's value) refers to module src: ('import', stmt) / ('getter', None) / None."""
    if _is_getter_call(node, src, gets):
        return ("getter", None)
    if isinstance(node, ast.Subscript) and ast.unparse(node) in (f"sys.modules['{src}']", f'sys.modules["{src}"]'):
        return ("getter", None)
    if isinstance(node, ast.Call) and ast.unparse(node).startswith(("sys.modules.get(", "__import__(")) and src in ast.unparse(node):
        return ("getter", None)
    if not isinstance(node, ast.Name):
        return None
    sc = an.resolve(node)
    for b in sc.bindings.get(node.id, []):
        if isinstance(b, ast.Import):
            for a in b.names:
                if a.name == src and (a.asname or a.name) == node.id:
                    return ("import", b)
        if isinstance(b, ast.ImportFrom):
            continue
        if isinstance(b, (ast.Assign, ast.AnnAssign)) and b.value is not None:
            v = b.value
            if isinstance(b, ast.Assign) and isinstance(v, ast.Tuple):
                for t in b.targets:
                    if isinstance(t, ast.Tuple) and len(t.elts) == len(v.elts):
                        for te, ve in zip(t.elts, v.elts):
                            if isinstance(te, ast.Name) and te.id == node.id and _module_expr(an, ve, src, gets, node.id):
                                return ("getter", None)
            elif _module_expr(an, v, src, gets, node.id):
                return ("getter", None)
    return None


def first_body_line(fn):
    """Line to insert a statement before: the first statement after the docstring."""
    body = fn.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        if len(body) > 1:
            return body[1].lineno, body[1].col_offset
        return body[0].end_lineno + 1, body[0].col_offset
    return body[0].lineno, body[0].col_offset


# --------------------------------------------------------------------------- callers
def rewrite_callers(src, names_to_mod: dict[str, str], skip: set[Path]):
    """In every file: <src alias>.<name> -> <new module>.<name> (with the import)."""
    changed = []
    for p in files():
        if p in skip:
            continue
        text = p.read_text(encoding="utf-8")
        if src not in text and not any(f"{g}()" in text for g in global_getters(src)):
            continue
        tree = ast.parse(text)
        an = Analyzer(tree)
        gets = getters(tree, src)
        ed = Edits(text)
        need_top: set[str] = set()
        did = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in names_to_mod:
                how = is_src_value(an, node.value, src, gets)
                if how is None:
                    if isinstance(node.value, ast.Name) and node.value.id in ("S", "sg", "G", "F", "LL", "ll", "_S", "dsflow", "sdgui", "lora_library", "_sg", "DF"):
                        NOTES.append(f"? {p.relative_to(REPO)}:{node.lineno} {ast.unparse(node)}")
                    continue
                new = names_to_mod[node.attr]
                ed.replace(node.value, new)
                did = True
                kind, stmt = how
                if kind == "import":
                    ind = " " * stmt.col_offset
                    ed.insert_line_after(stmt.end_lineno, f"{ind}import {new}\n")
                else:
                    fn = an.enclosing_function(node)
                    if fn is None or fn.kind == "module":
                        need_top.add(new)
                    else:
                        ln, col = first_body_line(fn.node)
                        ed.insert_line_before(ln, f"{' ' * col}import {new}\n")
            elif isinstance(node, ast.ImportFrom) and node.module == src and any(a.name in names_to_mod for a in node.names):
                keep = [a for a in node.names if a.name not in names_to_mod]
                moved = [a for a in node.names if a.name in names_to_mod]
                ind = " " * node.col_offset
                lines = []
                if keep:
                    lines.append(f"from {src} import " + ", ".join(a.name + (f" as {a.asname}" if a.asname else "") for a in keep))
                by = {}
                for a in moved:
                    by.setdefault(names_to_mod[a.name], []).append(a)
                for m, al in by.items():
                    lines.append(f"from {m} import " + ", ".join(a.name + (f" as {a.asname}" if a.asname else "") for a in al))
                ed.replace(node, ("\n" + ind).join(lines))
                did = True
        if need_top:
            last = max((s for s in tree.body if isinstance(s, (ast.Import, ast.ImportFrom))), key=lambda s: s.end_lineno, default=None)
            ln = last.end_lineno if last else 1
            for m in sorted(need_top):
                ed.insert_line_after(ln, f"import {m}\n")
        if did:
            new_text = ed.apply()
            ast.parse(new_text)
            p.write_text(new_text, encoding="utf-8")
            changed.append(p.relative_to(REPO).as_posix())
    return changed


# --------------------------------------------------------------------------- move
def segment_start(lines, st):
    """First line (1-based) of a statement with its decorators and the comment lines right above it."""
    start = min([st.lineno] + [d.lineno for d in getattr(st, "decorator_list", [])])
    i = start - 1
    while i - 1 >= 0 and lines[i - 1].lstrip().startswith("#") and not lines[i - 1].startswith("# ---"):
        i -= 1
    # a section header right above (# ---- …) goes along when the comment block starts there
    while i - 1 >= 0 and lines[i - 1].startswith("# ---"):
        i -= 1
    return i + 1


def move(src, new, names, doc):
    sp = mod_path(src)
    text = sp.read_text(encoding="utf-8")
    tree = ast.parse(text)
    an = Analyzer(tree)
    lines = text.splitlines(keepends=True)
    tops = top_defs(tree)
    imps = import_bindings(tree)
    names = set(names)
    defined = set().union(*[n for _, n in tops])
    missing = names - defined
    assert not missing, f"not defined in {src}: {sorted(missing)}"
    moved_st = [st for st, n in tops if n and n <= names]
    partial = [(st.lineno, n) for st, n in tops if n & names and not n <= names]
    assert not partial, f"statements only partly moved: {partial}"
    remaining = set(an.module.locals) - names
    alias = ALIAS[src]
    # does the alias collide with a name in the moved code?
    moved_ids = {n.id for st in moved_st for n in ast.walk(st) if isinstance(n, ast.Name)} | \
                {a.arg for st in moved_st for a in ast.walk(st) if isinstance(a, ast.arg)}
    if alias in moved_ids:
        alias = src
    # 1. the moved code: names of src's remaining globals qualified, imports collected
    ed = Edits(text)
    need_imports = {}
    module_level_refs = []
    use_alias = False
    moved_ranges = []
    for st in moved_st:
        moved_ranges.append((segment_start(lines, st), st.end_lineno))
        for n in ast.walk(st):
            if isinstance(n, ast.Name) and an.is_global(n):
                fn = an.enclosing_function(n)
                at_import = fn is None or fn.kind == "module"
                if n.id in remaining and n.id not in imps:
                    if isinstance(n.ctx, ast.Store) or isinstance(n.ctx, ast.Del):
                        NOTES.append(f"! moved code assigns {src}.{n.id} (line {n.lineno}): check")
                    ed.replace(n, f"{alias}.{n.id}")
                    use_alias = True
                    if at_import:
                        module_level_refs.append(f"{n.id} (line {n.lineno})")
                elif n.id in imps:
                    need_imports[n.id] = imps[n.id]
    if any("__file__" in ast.unparse(st) for st in moved_st):
        NOTES.append(f"! {new}: the moved code uses __file__: check the folder it means")
    if module_level_refs:
        NOTES.append(f"! {new}: refers to {src} at import time: {', '.join(module_level_refs)}")
    qualified = ed.apply()
    qlines = qualified.splitlines(keepends=True)
    merged = []                      # neighbours in the source (only blank lines between) stay as they were
    for a, z in sorted(moved_ranges):
        if merged and all(not qlines[i].strip() for i in range(merged[-1][1], a - 1)):
            merged[-1] = (merged[-1][0], z)
        else:
            merged.append((a, z))
    body = []
    for a, z in merged:
        chunk = "".join(qlines[a - 1:z]).rstrip("\n") + "\n"
        body.append(chunk)
    # new module text
    out = [HEADER, f'"""{doc.strip()}\n"""\n', "from __future__ import annotations\n",
           f'__version__ = "0"  # LoRA Studio version, see version.py\n\n']
    std, own = [], []
    for nm, (st, a) in sorted(need_imports.items(), key=lambda kv: kv[1][0].lineno):
        if nm == "annotations":
            continue
        t = import_text(st, a)
        top = (a.name if isinstance(st, ast.Import) else (st.module or "")).split(".")[0]
        (std if top in sys.stdlib_module_names else own).append(t)
    std = sorted(set(std), key=lambda t: (t.startswith("from"), t))
    own = sorted(set(own) | ({f"import {src}" + (f" as {alias}" if alias != src else "")} if use_alias else set()))
    out.append("\n".join(std) + ("\n\n" if std and own else "") + "\n".join(own) + "\n\n\n")
    out.append("\n\n".join(body))
    new_text = "".join(out)
    np_ = sp.parent / f"{new}.py"
    assert not np_.exists(), np_
    # 2. src without the moved code; its own references to moved names -> new.<name>
    ed2 = Edits(text)
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and n.id in names and an.is_global(n):
            inside = any(a <= n.lineno <= z for a, z in moved_ranges)
            if not inside:
                fn = an.enclosing_function(n)
                if fn is None or fn.kind == "module":
                    NOTES.append(f"! {src} uses {n.id} at import time (line {n.lineno})")
                ed2.replace(n, f"{new}.{n.id}")
    for a, z in moved_ranges:
        p0 = ed2.pos(a, 0)
        p1 = ed2.offs[z] if z < len(ed2.offs) else len(ed2.b)
        ed2.items.append((p0, p1, ""))
    src_text = ed2.apply()
    src_text = re.sub(r"\n{4,}", "\n\n\n", src_text)
    # import of the new module at the top of src (after the last plain top import)
    t2 = ast.parse(src_text)
    if re.search(rf"\b{new}\.", src_text):
        last = max((s for s in t2.body if isinstance(s, (ast.Import, ast.ImportFrom))), key=lambda s: s.end_lineno)
        e3 = Edits(src_text)
        e3.insert_line_after(last.end_lineno, f"import {new}\n")
        src_text = e3.apply()
    ast.parse(src_text)
    ast.parse(new_text)
    sp.write_text(src_text, encoding="utf-8")
    np_.write_text(new_text, encoding="utf-8")
    # 3. callers
    ch = rewrite_callers(src, {n: new for n in names}, {sp})
    return np_, ch


def unexport(src, names):
    """Names src only imports from elsewhere: callers use the module they come from."""
    sp = mod_path(src)
    tree = ast.parse(sp.read_text(encoding="utf-8"))
    origin = {}
    for st in ast.walk(tree):
        if isinstance(st, ast.ImportFrom) and st.level == 0:
            for a in st.names:
                if (a.asname or a.name) in names:
                    assert not a.asname, a.asname
                    origin[a.name] = st.module
    miss = set(names) - set(origin)
    assert not miss, miss
    return rewrite_callers(src, origin, {sp})


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "move":
        src, new, nf = sys.argv[2:5]
        names = [x.strip() for x in Path(nf).read_text().split() if x.strip() and not x.startswith("#")]
        doc = Path(sys.argv[5]).read_text() if len(sys.argv) > 5 else f"Part of {src}."
        np_, ch = move(src, new, names, doc)
        print("new:", np_.relative_to(REPO), "\ncallers:", " ".join(ch))
    elif cmd == "unexport":
        ch = unexport(sys.argv[2], sys.argv[3:])
        print("callers:", " ".join(ch))
    for n in NOTES:
        print(n)
