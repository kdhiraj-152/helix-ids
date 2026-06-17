"""Verify no reverse dependencies from src/helix_ids into scripts/.

Rule: src/helix_ids must never import from scripts/training (or any scripts/ module).
scripts/ imports from src/helix_ids are allowed.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src" / "helix_ids"
SCRIPTS_TRAINING_ROOT = PROJECT_ROOT / "scripts" / "training"


def _collect_import_targets(root: Path) -> dict[str, set[str]]:
    """Walk a directory tree and collect all absolute imports per file.

    Returns {relative_file_path: {imported_module, ...}} where each
    imported_module is a dotted module name at the top level.
    """
    imports: dict[str, set[str]] = {}
    for pyfile in sorted(root.rglob("*.py")):
        if pyfile.name == "__init__.py":
            continue
        # Only consider files actually under the root (skip venv, etc.)
        if not str(pyfile.resolve()).startswith(str(root.resolve())):
            continue
        try:
            tree = ast.parse(pyfile.read_text("utf-8"))
        except SyntaxError:
            continue  # pragma: no cover -- malformed file
        rel = pyfile.relative_to(PROJECT_ROOT).as_posix()
        targets: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    targets.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    targets.add(node.module.split(".")[0])
        if targets:
            imports[rel] = targets
    return imports


@pytest.fixture(scope="session")
def src_imports() -> dict[str, set[str]]:
    """All top-level import targets from src/helix_ids."""
    return _collect_import_targets(SRC_ROOT)


def test_no_reverse_dependency_to_scripts(src_imports: dict[str, set[str]]) -> None:
    """src/helix_ids must not import from scripts/."""
    offenders: list[str] = []
    for filepath, targets in sorted(src_imports.items()):
        if "scripts" in targets:
            offenders.append(f"{filepath} imports scripts")
    assert not offenders, (
        f"Reverse dependency violations ({len(offenders)}):\n" + "\n".join(offenders)
    )


def test_no_reverse_dependency_to_scripts_training(
    src_imports: dict[str, set[str]],
) -> None:
    """src/helix_ids must not import from scripts/training specifically."""
    offenders: list[str] = []
    for filepath, targets in sorted(src_imports.items()):
        for t in targets:
            if t.startswith("scripts") or (t == "train_helix_ids_full"):
                offenders.append(f"{filepath} -> {t}")
    assert not offenders, (
        f"scripts/training reverse dependency violations ({len(offenders)}):\n"
        + "\n".join(offenders)
    )


# Edge-case: ensure the test itself doesn't have import issues
def test_src_imports_collected(src_imports: dict[str, set[str]]) -> None:
    """Sanity check that the collector found real imports in src."""
    assert len(src_imports) > 0, "No imports collected from src/helix_ids"
