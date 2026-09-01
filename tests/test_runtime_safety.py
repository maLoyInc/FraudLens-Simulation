"""Static guard: the application package must never train or write artifacts.

CLAUDE.md section 7 requires training and inference to stay separate. This walks
the AST of every module under ``fraudlens/`` and fails on the calls that would
mean the runtime is fitting a model or writing a file.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "fraudlens"

BANNED_METHODS = {"fit", "fit_transform", "partial_fit", "dump", "to_pickle",
                  "to_parquet", "to_csv", "write_text", "write_bytes", "savefig"}
BANNED_FUNCTIONS = {"open"}
WRITE_MODES = {"w", "wb", "a", "ab", "w+", "wb+", "r+", "x", "xb"}


def modules() -> list[Path]:
    return sorted(PACKAGE.rglob("*.py"))


def calls(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            yield node


def test_the_package_has_modules_to_check():
    assert modules()


def test_no_runtime_fitting_or_serialisation():
    offences: list[str] = []
    for path in modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in calls(tree):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in BANNED_METHODS:
                offences.append(f"{path.name}:{node.lineno} .{func.attr}()")
            if isinstance(func, ast.Name) and func.id in BANNED_FUNCTIONS:
                offences.append(f"{path.name}:{node.lineno} {func.id}()")
    assert not offences, "runtime training or file writing found: " + ", ".join(offences)


def test_no_write_mode_string_appears_anywhere():
    offences = []
    for path in modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value in WRITE_MODES:
                offences.append(f"{path.name}:{node.lineno} {node.value!r}")
    assert not offences, "file write mode found: " + ", ".join(offences)


def test_no_training_only_imports_reach_the_application():
    banned_roots = {"imblearn", "sklearn.model_selection"}
    offences = []
    for path in modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if any(name == root or name.startswith(root + ".")
                       for root in banned_roots):
                    offences.append(f"{path.name}:{node.lineno} {name}")
    assert not offences, "training-only import found: " + ", ".join(offences)
