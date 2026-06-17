"""Phase 20 — Architecture Lockdown Enforcement.

This test module codifies the frozen architecture contract for RC1 readiness.
All checks are designed to fail CI if any architectural invariant is violated.

Frozen invariants:
  - HelixFullTrainer LOC <= 2000
  - HelixFullTrainer methods <= 100
  - src -> scripts imports = 0 (no reverse dependencies)
  - Package-level cycles = 0
  - Self-imports = 0
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src" / "helix_ids"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
SCRIPTS_TRAINING_ROOT = PROJECT_ROOT / "scripts" / "training"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_imports(root: Path, project_root: Path) -> dict[str, set[str]]:
    """Collect all top-level import targets per file under *root*.

    Returns {relative_filepath: {import_target, ...}} where each
    import_target is the first dotted component (``helix_ids``, ``torch``).
    """
    imports: dict[str, set[str]] = {}
    for pyfile in sorted(root.rglob("*.py")):
        if pyfile.name == "__init__.py":
            continue
        if not str(pyfile.resolve()).startswith(str(root.resolve())):
            continue
        try:
            tree = ast.parse(pyfile.read_text("utf-8"))
        except SyntaxError:
            continue
        rel = pyfile.relative_to(project_root).as_posix()
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


def _collect_full_imports(root: Path, project_root: Path) -> dict[str, set[str]]:
    """Collect full dotted import paths (not just top-level)."""
    imports: dict[str, set[str]] = {}
    for pyfile in sorted(root.rglob("*.py")):
        if pyfile.name == "__init__.py":
            continue
        if not str(pyfile.resolve()).startswith(str(root.resolve())):
            continue
        try:
            tree = ast.parse(pyfile.read_text("utf-8"))
        except SyntaxError:
            continue
        rel = pyfile.relative_to(project_root).as_posix()
        targets: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    targets.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    targets.add(node.module)
        if targets:
            imports[rel] = targets
    return imports


@pytest.fixture(scope="session")
def src_imports() -> dict[str, set[str]]:
    return _collect_imports(SRC_ROOT, PROJECT_ROOT)


# ===================================================================
# Rule 1 — Frozen Trainer Size Limits
# ===================================================================

def test_trainer_loc_frozen_2000() -> None:
    """HelixFullTrainer class body must not exceed 2000 LOC (RC1 freeze gate)."""
    trainer_file = SCRIPTS_TRAINING_ROOT / "train_helix_ids_full.py"
    assert trainer_file.exists(), "train_helix_ids_full.py not found"
    tree = ast.parse(trainer_file.read_text("utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "HelixFullTrainer":
            class_loc = node.end_lineno - node.lineno + 1
            assert class_loc <= 2000, (
                f"HelixFullTrainer is {class_loc} LOC — exceeds RC1 freeze gate of 2000. "
                f"Architecture lockdown prohibits size regrowth."
            )
            break


def test_trainer_methods_frozen_100() -> None:
    """HelixFullTrainer must not exceed 100 methods (RC1 freeze gate)."""
    trainer_file = SCRIPTS_TRAINING_ROOT / "train_helix_ids_full.py"
    assert trainer_file.exists(), "train_helix_ids_full.py not found"
    tree = ast.parse(trainer_file.read_text("utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "HelixFullTrainer":
            count = sum(
                1 for n in node.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            )
            assert count <= 100, (
                f"HelixFullTrainer has {count} methods — exceeds RC1 freeze gate of 100. "
                f"Architecture lockdown prohibits method regrowth."
            )
            break


# ===================================================================
# Rule 2 — No Reverse Dependencies (src -> scripts)
# ===================================================================

def test_src_does_not_import_scripts(src_imports: dict[str, set[str]]) -> None:
    """src/helix_ids must not import from scripts/ (zero reverse deps)."""
    offenders: list[str] = []
    for filepath, targets in sorted(src_imports.items()):
        if "scripts" in targets:
            offenders.append(f"{filepath} imports scripts")
    assert not offenders, (
        f"Reverse dependency violations ({len(offenders)}):\n" + "\n".join(offenders)
    )


def test_src_imports_collected(src_imports: dict[str, set[str]]) -> None:
    """Sanity — imports collector works."""
    assert len(src_imports) > 0, "No imports collected from src/helix_ids"


# ===================================================================
# Rule 3 — No Package Cycles
# ===================================================================

BOUNDARIES: dict[str, str] = {
    "src/helix_ids": "helix_ids",
    "scripts/training": "scripts.training",
    "scripts/operations": "scripts.operations",
    "scripts/evaluation": "scripts.evaluation",
    "scripts/data": "scripts.data",
    "scripts/ci": "scripts.ci",
}


@pytest.fixture(scope="session")
def boundary_graph() -> dict[str, dict[str, set[str]]]:
    """Build a directed graph of cross-boundary imports."""
    graph: dict[str, dict[str, set[str]]] = {label: {} for label in BOUNDARIES.values()}
    boundary_by_prefix: dict[str, str] = {
        str(PROJECT_ROOT / d): label for d, label in BOUNDARIES.items()
    }
    for dirpath_str, label in boundary_by_prefix.items():
        dirpath = Path(dirpath_str)
        if not dirpath.is_dir():
            continue
        for pyfile in sorted(dirpath.rglob("*.py")):
            if any(p.startswith(".") for p in pyfile.parent.relative_to(dirpath).parts):
                continue
            rel = pyfile.relative_to(PROJECT_ROOT).as_posix()
            try:
                tree = ast.parse(pyfile.read_text("utf-8"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                mod_name: str | None = None
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        mod_name = alias.name
                        break
                elif isinstance(node, ast.ImportFrom) and node.module:
                    mod_name = node.module
                if mod_name:
                    imported_label = _resolve_boundary(mod_name, boundary_by_prefix)
                    if imported_label and imported_label != label:
                        graph[label].setdefault(imported_label, set()).add(rel)
    return graph


def _resolve_boundary(mod_name: str, boundaries: dict[str, str]) -> str | None:
    """Return boundary LABEL if mod_name is inside a known boundary dir."""
    mod_path = mod_name.replace(".", "/")
    for _dirpath_str, label in boundaries.items():
        boundary_top = label.split(".")[0]
        if mod_path.split("/")[0] == boundary_top:
            return label
    return None


def test_zero_package_cycles(boundary_graph: dict[str, dict[str, set[str]]]) -> None:
    """The cross-boundary import graph must have zero cycles."""
    labels = list(boundary_graph.keys())
    edges: list[tuple[str, str]] = []
    for src_label, targets in boundary_graph.items():
        for tgt_label in targets:
            if src_label != tgt_label:
                edges.append((src_label, tgt_label))
    adj: dict[str, list[str]] = {lbl: [] for lbl in labels}
    for src, tgt in edges:
        adj[src].append(tgt)
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = dict.fromkeys(labels, WHITE)

    def dfs(node: str, path: list[str] | None = None) -> list[str] | None:
        if path is None:
            path = []
        color[node] = GRAY
        path.append(node)
        for neighbor in adj.get(node, []):
            if color[neighbor] == GRAY:
                cycle_start = path.index(neighbor)
                return path[cycle_start:] + [neighbor]
            if color[neighbor] == WHITE:
                result = dfs(neighbor, path)
                if result is not None:
                    return result
        path.pop()
        color[node] = BLACK
        return None

    for label in labels:
        if color[label] == WHITE:
            cycle = dfs(label)
            if cycle is not None:
                pytest.fail(f"Dependency cycle detected: {' -> '.join(cycle)}")


def test_zero_src_internal_cycles() -> None:
    """src/helix_ids/ subpackages must form a DAG (zero cycles)."""
    imports = _collect_full_imports(SRC_ROOT, PROJECT_ROOT)
    pkg_edges: dict[str, set[str]] = {}
    for filepath, targets in imports.items():
        file_pkg = ".".join(filepath.replace(".py", "").split("/")[1:3])
        for t in targets:
            if t.startswith("helix_ids."):
                target_pkg = ".".join(t.split(".")[:2])
                if target_pkg != file_pkg:
                    pkg_edges.setdefault(file_pkg, set()).add(target_pkg)
    nodes = list(pkg_edges.keys())
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = dict.fromkeys(nodes, WHITE)

    def dfs(n: str, path: list[str]) -> list[str] | None:
        color[n] = GRAY
        path.append(n)
        for neighbor in pkg_edges.get(n, []):
            if color.get(neighbor) == GRAY:
                cycle_start = path.index(neighbor)
                return path[cycle_start:] + [neighbor]
            if color.get(neighbor) == WHITE:
                result = dfs(neighbor, path)
                if result:
                    return result
        path.pop()
        color[n] = BLACK
        return None

    for n in nodes:
        if color[n] == WHITE:
            cycle = dfs(n, [])
            if cycle:
                pytest.fail(f"Internal src cycle: {' -> '.join(cycle)}")


# ===================================================================
# Rule 4 — No Self-Imports
# ===================================================================

def test_zero_self_imports_in_training() -> None:
    """scripts/training files must not self-import."""
    for pyfile in sorted(SCRIPTS_TRAINING_ROOT.rglob("*.py")):
        if pyfile.name == "__init__.py":
            continue
        rel = pyfile.relative_to(PROJECT_ROOT).as_posix()
        module_path = rel.replace("/", ".").replace(".py", "")
        try:
            tree = ast.parse(pyfile.read_text("utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == module_path:
                pytest.fail(
                    f"Self-import detected in {rel}: "
                    f"'from {module_path} import ...'"
                )


def test_zero_self_imports_in_src() -> None:
    """src/helix_ids files must not self-import."""
    for pyfile in sorted(SRC_ROOT.rglob("*.py")):
        if pyfile.name == "__init__.py":
            continue
        rel = pyfile.relative_to(PROJECT_ROOT).as_posix()
        module_path = rel.replace("/", ".").replace(".py", "")
        try:
            tree = ast.parse(pyfile.read_text("utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == module_path:
                pytest.fail(
                    f"Self-import detected in {rel}: "
                    f"'from {module_path} import ...'"
                )


# ===================================================================
# Rule 5 — No Forbidden Imports (src. prefix)
# ===================================================================

def test_src_does_not_import_src_prefix() -> None:
    """Files under src/ must not use 'src.' prefix in imports."""
    imports = _collect_imports(SRC_ROOT, PROJECT_ROOT)
    offenders = [fp for fp, targets in imports.items() if "src" in targets]
    assert not offenders, (
        f"Files using 'src' prefix import ({len(offenders)}):\n" + "\n".join(offenders)
    )


# ===================================================================
# Smoke
# ===================================================================

def test_architecture_lockdown_smoke() -> None:
    """Minimal smoke — module loads."""
    assert True
