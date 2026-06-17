"""Verify the trainer-boundary rule: scripts/training may import from
src/helix_ids but must NOT import from helix_ids via `src.` prefix
(only bare `helix_ids` imports from the installed package).
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_TRAINING_ROOT = PROJECT_ROOT / "scripts" / "training"
SRC_HELIX_IDS_ROOT = PROJECT_ROOT / "src" / "helix_ids"


def _collect_file_imports(
    root: Path, project_root: Path
) -> dict[str, set[str]]:
    """Collect all import targets (top-level module) per file."""
    result: dict[str, set[str]] = {}
    for pyfile in sorted(root.rglob("*.py")):
        if pyfile.name == "__init__.py":
            continue
        rel = pyfile.relative_to(project_root).as_posix()
        try:
            tree = ast.parse(pyfile.read_text("utf-8"))
        except SyntaxError:
            continue
        targets: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    targets.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                targets.add(node.module.split(".")[0])
        if targets:
            result[rel] = targets
    return result


def test_training_may_import_helix_ids() -> None:
    """scripts/training may import from helix_ids (it's the allowed direction)."""
    imports = _collect_file_imports(SCRIPTS_TRAINING_ROOT, PROJECT_ROOT)
    helix_ids_importers = [
        fp for fp, targets in imports.items() if "helix_ids" in targets
    ]
    # At minimum, the main training script should import helix_ids
    assert len(helix_ids_importers) > 0, (
        "No files in scripts/training import from helix_ids — "
        "expected at least some to use the core library"
    )
    # All helix_ids imports (if any) are fine — this is the allowed direction


def test_src_does_not_import_src_prefix() -> None:
    """Files under src/ must not use 'src.' prefix in imports."""
    imports = _collect_file_imports(SRC_HELIX_IDS_ROOT, PROJECT_ROOT)
    offenders = [
        fp
        for fp, targets in imports.items()
        if "src" in targets
    ]
    assert not offenders, (
        f"Files using 'src' prefix import ({len(offenders)}):\n"
        + "\n".join(offenders)
    )


def test_training_imports_are_from_helix_ids_not_src() -> None:
    """train_helix_ids_full.py should import from helix_ids, not src.helix_ids."""
    target_file = SCRIPTS_TRAINING_ROOT / "train_helix_ids_full.py"
    rel = "scripts/training/train_helix_ids_full.py"
    try:
        tree = ast.parse(target_file.read_text("utf-8"))
    except SyntaxError:
        pytest.fail(f"Cannot parse {rel}")
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("src."):
                offenders.append(f"{rel}: {node.module}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("src."):
                    offenders.append(f"{rel}: {alias.name}")
    assert not offenders, (
        "train_helix_ids_full.py uses 'src.' prefix imports:\n"
        + "\n".join(offenders)
    )


def test_no_implicit_src_import_via_syspath() -> None:
    """train_helix_ids_full.py should not contain 'src.helix_ids' import strings."""
    target_file = SCRIPTS_TRAINING_ROOT / "train_helix_ids_full.py"
    content = target_file.read_text("utf-8")
    if "import src.helix_ids" in content or "from src.helix_ids" in content:
        pytest.fail(
            "train_helix_ids_full.py contains 'src.helix_ids' import strings"
        )
