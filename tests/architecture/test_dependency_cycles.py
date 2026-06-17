"""Verify zero cyclic dependencies between high-level packages.

Tracks imports at the boundary level between:
  - src/helix_ids/* (the core library — data, models, governance, etc.)
  - scripts/training/* (the training pipeline)
  - scripts/operations/* (deployment & operations)

This is a coarse cycle-detection layer test.  Fine-grained cell-level
cycle detection is left to static analysis tools.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Boundary packages we track (directory -> label)
BOUNDARIES: dict[str, str] = {
    "src/helix_ids": "helix_ids",
    "scripts/training": "scripts.training",
}

# Directories that do NOT live inside any boundary but may be imported
IGNORE_PREFIXES = (".venv", "venv", "__pycache__", ".git", ".mypy_cache", ".pytest_cache")


@pytest.fixture(scope="session")
def boundary_graph() -> dict[str, dict[str, set[str]]]:
    """Build a directed graph of cross-boundary imports.

    Returns {boundary_label: {imported_boundary_label: {files...}}}.
    """
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
                continue  # skip hidden dirs inside boundary
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
                        break  # only check first top-level name per node
                elif isinstance(node, ast.ImportFrom) and node.module:
                    mod_name = node.module
                if mod_name:
                    # Check if the imported module lives inside a different boundary
                    imported_label = _resolve_boundary(mod_name, boundary_by_prefix)
                    if imported_label and imported_label != label:
                        graph[label].setdefault(imported_label, set()).add(rel)
    return graph


def _resolve_boundary(
    mod_name: str, boundaries: dict[str, str]
) -> str | None:
    """Return the boundary LABEL if mod_name lives inside a known boundary dir."""
    mod_path = mod_name.replace(".", "/")
    for _dirpath, label in boundaries.items():
        # The boundary label corresponds to the top-level package
        # e.g. "helix_ids" for "src/helix_ids", "scripts" for "scripts/training"
        boundary_top = label.split(".")[0]
        if mod_path.split("/")[0] == boundary_top:
            return label
    return None


def test_no_cycles_in_boundary_graph(
    boundary_graph: dict[str, dict[str, set[str]]],
) -> None:
    """Verify the directed cross-boundary import graph is acyclic.

    NOTE: Intra-boundary imports (e.g. helix_ids/foo importing helix_ids/bar)
    are intentionally excluded — we only care about cross-boundary edges.
    """
    labels = list(boundary_graph.keys())
    edges: list[tuple[str, str]] = []
    for src_label, targets in boundary_graph.items():
        for tgt_label, _files in targets.items():
            if src_label != tgt_label:  # skip intra-boundary
                edges.append((src_label, tgt_label))

    # Simple DFS cycle detection
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
                # Found a cycle
                cycle_start = path.index(neighbor)
                return path[cycle_start:] + [neighbor]
            if color[neighbor] == WHITE:
                result = dfs(neighbor, path)
                if result is not None:
                    return result
        path.pop()
        color[node] = BLACK
        return None

    cycle = dfs(labels[0] if labels else "")
    assert cycle is None, (
        f"Dependency cycle detected: {' -> '.join(cycle)}"
    )


def test_no_self_imports_in_training() -> None:
    """Verify scripts/training files don't import from themselves.

    A self-import is when file imports from its own module path.
    """
    training_dir = PROJECT_ROOT / "scripts" / "training"
    for pyfile in sorted(training_dir.rglob("*.py")):
        if pyfile.name == "__init__.py":
            continue
        # Determine the dotted module path
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
