# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Scope analysis: which Name nodes refer to module globals, and how a name is bound in a scope."""
import ast


class Scope:
    def __init__(self, node, parent, kind):
        self.node, self.parent, self.kind = node, parent, kind   # kind: module / function / class / comp
        self.locals: set[str] = set()
        self.globals_decl: set[str] = set()
        self.nonlocal_decl: set[str] = set()
        self.bindings: dict[str, list] = {}   # name -> list of binding nodes (Import alias, Assign value, ...)


def _targets(t):
    if isinstance(t, ast.Name):
        yield t.id, t
    elif isinstance(t, (ast.Tuple, ast.List)):
        for e in t.elts:
            yield from _targets(e)
    elif isinstance(t, ast.Starred):
        yield from _targets(t.value)


class Analyzer:
    """Build scopes; then resolve(name_node) -> scope where it is bound (module scope = global)."""

    def __init__(self, tree):
        self.tree = tree
        self.scope_of: dict[int, Scope] = {}     # id(node) -> scope that the node lives in
        self.module = Scope(tree, None, "module")
        self._build(tree, self.module)

    # -------------------------------------------------------------- building
    def _new(self, node, parent, kind):
        s = Scope(node, parent, kind)
        self.scope_of[id(node)] = parent          # the def itself lives in the parent
        return s

    def _bind(self, scope, name, how):
        if scope.kind == "comp" and how is not None and isinstance(how, ast.NamedExpr):
            # walrus binds in the enclosing function/module
            s = scope
            while s.kind == "comp":
                s = s.parent
            scope = s
        if name in scope.globals_decl:
            scope = self.module
        scope.locals.add(name)
        scope.bindings.setdefault(name, []).append(how)

    def _build(self, node, scope):
        for child in ast.iter_child_nodes(node):
            self._visit(child, scope)

    def _visit(self, node, scope):
        self.scope_of[id(node)] = scope
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for d in node.decorator_list:
                self._visit(d, scope)
            self._defaults(node.args, scope)
            if node.returns:
                self._visit(node.returns, scope)
            self._bind(scope, node.name, node)
            s = self._new(node, scope, "function")
            self._args(node.args, s, scope)
            self._prescan_decl(node.body, s)
            for st in node.body:
                self._visit(st, s)
            return
        if isinstance(node, ast.Lambda):
            self._defaults(node.args, scope)
            s = self._new(node, scope, "function")
            self._args(node.args, s, scope)
            self._visit(node.body, s)
            return
        if isinstance(node, ast.ClassDef):
            for d in node.decorator_list:
                self._visit(d, scope)
            for b in node.bases + [k.value for k in node.keywords]:
                self._visit(b, scope)
            self._bind(scope, node.name, node)
            s = self._new(node, scope, "class")
            for st in node.body:
                self._visit(st, s)
            return
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            gens = node.generators
            self._visit(gens[0].iter, scope)          # first iterable in the enclosing scope
            s = self._new(node, scope, "comp")
            for i, g in enumerate(gens):
                if i:
                    self._visit(g.iter, s)
                for n, t in _targets(g.target):
                    self._bind(s, n, g)
                    self.scope_of[id(t)] = s
                for c in g.ifs:
                    self._visit(c, s)
            if isinstance(node, ast.DictComp):
                self._visit(node.key, s)
                self._visit(node.value, s)
            else:
                self._visit(node.elt, s)
            return
        if isinstance(node, ast.Global):
            scope.globals_decl.update(node.names)
            return
        if isinstance(node, ast.Nonlocal):
            scope.nonlocal_decl.update(node.names)
            return
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                nm = a.asname or a.name.split(".")[0]
                if nm != "*":
                    self._bind(scope, nm, node)
            return
        if isinstance(node, ast.Assign):
            for t in node.targets:
                for n, tn in _targets(t):
                    self._bind(scope, n, node)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            for n, tn in _targets(node.target):
                self._bind(scope, n, node)
        elif isinstance(node, ast.NamedExpr):
            self._bind(scope, node.target.id, node)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            for n, tn in _targets(node.target):
                self._bind(scope, n, node)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for it in node.items:
                if it.optional_vars is not None:
                    for n, tn in _targets(it.optional_vars):
                        self._bind(scope, n, node)
        elif isinstance(node, ast.ExceptHandler):
            if node.name:
                self._bind(scope, node.name, node)
        elif isinstance(node, ast.Delete):
            for t in node.targets:
                for n, tn in _targets(t):
                    self._bind(scope, n, node)
        elif isinstance(node, ast.match_case):
            pass
        for child in ast.iter_child_nodes(node):
            self._visit(child, scope)

    def _prescan_decl(self, body, s):
        for st in body:
            for n in ast.walk(st):
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)) and n is not st:
                    pass
                if isinstance(n, ast.Global):
                    s.globals_decl.update(n.names)
                if isinstance(n, ast.Nonlocal):
                    s.nonlocal_decl.update(n.names)

    def _defaults(self, args, scope):
        for d in args.defaults + [d for d in args.kw_defaults if d is not None]:
            self._visit(d, scope)
        for a in args.posonlyargs + args.args + args.kwonlyargs + [args.vararg, args.kwarg]:
            if a is not None and a.annotation is not None:
                self._visit(a.annotation, scope)

    def _args(self, args, s, outer):
        for a in args.posonlyargs + args.args + args.kwonlyargs + [args.vararg, args.kwarg]:
            if a is not None:
                self._bind(s, a.arg, a)

    # -------------------------------------------------------------- resolving
    def resolve(self, name_node) -> Scope:
        """The scope a Name node's name is bound in (module scope when global or unbound)."""
        s = self.scope_of[id(name_node)]
        nm = name_node.id
        first = True
        while s is not None:
            if s.kind == "class" and not first:
                s = s.parent
                continue
            if nm in s.globals_decl:
                return self.module
            if nm in s.locals and nm not in s.nonlocal_decl:
                return s
            first = False
            s = s.parent
        return self.module

    def is_global(self, name_node) -> bool:
        return self.resolve(name_node) is self.module

    def enclosing_function(self, node):
        s = self.scope_of.get(id(node))
        while s is not None and s.kind not in ("function", "module"):
            s = s.parent
        return s
