#!/usr/bin/env python3
"""Fail if protected files call restricted sklearn metric APIs directly."""

from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROTECTED_FILES = [
    PROJECT_ROOT / "scripts" / "train_multidataset_v2_fixed.py",
    PROJECT_ROOT / "scripts" / "train_helix_ids_full.py",
    PROJECT_ROOT / "scripts" / "benchmark_e2e_v2_fixed.py",
    PROJECT_ROOT / "scripts" / "holdout_evaluation_v2.py",
]

FORBIDDEN_SYMBOLS = {
    "f1_score",
    "precision_score",
    "recall_score",
    "accuracy_score",
    "classification_report",
    "confusion_matrix",
}


class MetricBypassVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.errors: list[str] = []

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module != "sklearn.metrics":
            return
        for alias in node.names:
            if alias.name in FORBIDDEN_SYMBOLS:
                self.errors.append(
                    f"line {node.lineno}: direct sklearn.metrics import '{alias.name}' is forbidden"
                )

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_SYMBOLS:
            self.errors.append(
                f"line {node.lineno}: direct call '{node.func.id}(...)' is forbidden"
            )
        self.generic_visit(node)


def validate_file(file_path: Path) -> list[str]:
    tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
    visitor = MetricBypassVisitor()
    visitor.visit(tree)
    return visitor.errors


def main() -> int:
    failures: list[str] = []
    for file_path in PROTECTED_FILES:
        if not file_path.exists():
            failures.append(f"{file_path}: protected file not found")
            continue

        errors = validate_file(file_path)
        for error in errors:
            failures.append(f"{file_path}: {error}")

    if failures:
        print("E-METRICS-BYPASS")
        for failure in failures:
            print(failure)
        return 1

    print("Metrics contract bypass check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
