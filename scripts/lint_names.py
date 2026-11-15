#!/usr/bin/env python
"""Contrôle statique minimal (sans dépendance) : noms globaux non définis et imports inutilisés.

Sert à relire le code PyTorch, que l'on ne peut pas exécuter partout : une faute de frappe sur un nom
est attrapée ici. Ne remplace ni pyflakes ni l'exécution (`scripts/smoke_test.py`).

    uv run scripts/lint_names.py src/flou/models src/flou/train scripts
"""
import ast
import builtins
import symtable
import sys
from pathlib import Path


def module_names(tree: ast.Module) -> set[str]:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                names.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
    for node in tree.body:
        for sub in ast.walk(node) if isinstance(node, (ast.Assign, ast.AnnAssign, ast.For, ast.With, ast.Try, ast.If)) else []:
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                names.add(sub.id)
    return names


def undefined_globals(src: str, path: str) -> list[str]:
    tree = ast.parse(src)
    defined = module_names(tree) | set(dir(builtins)) | {"__file__", "__name__", "__doc__"}
    problems = []

    def walk(tab: symtable.SymbolTable):
        for sym in tab.get_symbols():
            if sym.is_global() and sym.is_referenced() and sym.get_name() not in defined:
                problems.append(f"{path}: nom non défini « {sym.get_name()} » (portée {tab.get_name()})")
        for child in tab.get_children():
            walk(child)

    walk(symtable.symtable(src, path, "exec"))
    return problems


def unused_imports(src: str, path: str) -> list[str]:
    tree = ast.parse(src)
    imported = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                imported[(a.asname or a.name).split(".")[0]] = node.lineno
        elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
            for a in node.names:
                imported[a.asname or a.name] = node.lineno
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    used |= {n.value.id for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}
    exported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
            exported |= {e.value for e in node.value.elts if isinstance(e, ast.Constant)}
    return [f"{path}:{ln}: import inutilisé « {n} »" for n, ln in imported.items()
            if n not in used and n not in exported and not n.startswith("_") and not path.endswith("__init__.py")]


def main(args: list[str]) -> int:
    files = []
    for a in args:
        p = Path(a)
        files += sorted(p.rglob("*.py")) if p.is_dir() else [p]
    bad = []
    for f in files:
        src = f.read_text(encoding="utf-8")
        bad += undefined_globals(src, str(f)) + unused_imports(src, str(f))
    print("\n".join(bad) if bad else f"ok : {len(files)} fichiers, aucun problème")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["src", "scripts"]))
