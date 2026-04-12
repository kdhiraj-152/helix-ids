#!/usr/bin/env python3
"""Fail if protected executable entrypoints are not wrapped with @governed_entrypoint."""

from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROTECTED_ENTRYPOINTS = [
    PROJECT_ROOT / "scripts" / "training" / "train_multidataset_v2_fixed.py",
    PROJECT_ROOT / "scripts" / "training" / "train_helix_ids_full.py",
    PROJECT_ROOT / "scripts" / "evaluation" / "benchmark_e2e_v2_fixed.py",
    PROJECT_ROOT / "scripts" / "evaluation" / "holdout_evaluation_v2.py",
]


def _decorator_name(decorator: ast.expr) -> str:
    if isinstance(decorator, ast.Name):
        return decorator.id
    if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
        return decorator.func.id
    if isinstance(decorator, ast.Attribute):
        return decorator.attr
    if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute):
        return decorator.func.attr
    return ""


def validate_file(file_path: Path) -> list[str]:
    tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
    errors: list[str] = []

    main_fn: ast.FunctionDef | None = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            main_fn = node
            break

    if main_fn is None:
        return [f"{file_path}: missing main() for governed entrypoint"]

    decorator_names = {_decorator_name(dec) for dec in main_fn.decorator_list}
    if "governed_entrypoint" not in decorator_names:
        errors.append(f"{file_path}: main() is missing @governed_entrypoint")

    return errors


def main() -> int:
    errors: list[str] = []
    for file_path in PROTECTED_ENTRYPOINTS:
        if not file_path.exists():
            errors.append(f"{file_path}: protected entrypoint file not found")
            continue
        errors.extend(validate_file(file_path))

    if errors:
        print("E-GATE-ENTRYPOINT-BYPASS")
        for error in errors:
            print(error)
        return 1

    print("Entrypoint governance check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
