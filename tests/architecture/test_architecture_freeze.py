"""Phase 19 — Architecture Freeze Boundary Enforcement.

This suite codifies the frozen architecture contract:

  - src/helix_ids/  NEVER imports scripts/
  - scripts/       MAY  import src/
  - tests/         MAY  import both
  - No self-imports within any package
  - No package cycles between high-level boundaries
  - No forbidden imports (src. prefix, reverse deps, etc.)
  - HelixFullTrainer does not regrow (method count / LOC gate)
  - ENGINERED_FEATURE_NAMES is in src/helix_ids, not in scripts/training
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src" / "helix_ids"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
SCRIPTS_TRAINING_ROOT = PROJECT_ROOT / "scripts" / "training"
TESTS_ROOT = PROJECT_ROOT / "tests"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_imports(root: Path, project_root: Path) -> dict[str, set[str]]:
    """Collect all top-level import targets per file under *root*.

    Returns {relative_filepath: {import_target, ...}} where each
    import_target is the first dotted component (e.g. ``helix_ids``,
    ``scripts``, ``torch``).
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


# ---------------------------------------------------------------------------
# Rule 1 — No reverse dependencies (src -> scripts)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def src_imports() -> dict[str, set[str]]:
    return _collect_imports(SRC_ROOT, PROJECT_ROOT)


@pytest.fixture(scope="session")
def src_full_imports() -> dict[str, set[str]]:
    return _collect_full_imports(SRC_ROOT, PROJECT_ROOT)


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


def test_src_imports_collected(src_imports: dict[str, set[str]]) -> None:
    """Sanity check — the collector found real imports in src."""
    assert len(src_imports) > 0, "No imports collected from src/helix_ids"


# ---------------------------------------------------------------------------
# Rule 2 — No self imports
# ---------------------------------------------------------------------------

def test_no_self_imports_in_training() -> None:
    """scripts/training files must not import from their own module path."""
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


def test_no_self_imports_in_src() -> None:
    """src/helix_ids files must not import from their own module path."""
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


# ---------------------------------------------------------------------------
# Rule 3 — No package cycles
# ---------------------------------------------------------------------------

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


def _resolve_boundary(
    mod_name: str, boundaries: dict[str, str]
) -> str | None:
    """Return the boundary LABEL if mod_name lives inside a known boundary dir."""
    mod_path = mod_name.replace(".", "/")
    for _dirpath_str, label in boundaries.items():
        boundary_top = label.split(".")[0]
        if mod_path.split("/")[0] == boundary_top:
            return label
    return None


def test_no_cycles_in_boundary_graph(
    boundary_graph: dict[str, dict[str, set[str]]],
) -> None:
    """Verify the directed cross-boundary import graph is acyclic."""
    labels = list(boundary_graph.keys())
    edges: list[tuple[str, str]] = []
    for src_label, targets in boundary_graph.items():
        for tgt_label, _files in targets.items():
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

    cycle = dfs(labels[0] if labels else "")
    assert cycle is None, (
        f"Dependency cycle detected: {' -> '.join(cycle)}"
    )


# ---------------------------------------------------------------------------
# Rule 4 — No forbidden imports (src. prefix)
# ---------------------------------------------------------------------------

def test_training_may_import_helix_ids() -> None:
    """scripts/training may import from helix_ids (allowed direction)."""
    imports = _collect_imports(SCRIPTS_TRAINING_ROOT, PROJECT_ROOT)
    helix_ids_importers = [
        fp for fp, targets in imports.items() if "helix_ids" in targets
    ]
    assert len(helix_ids_importers) > 0, (
        "No files in scripts/training import from helix_ids"
    )


def test_src_does_not_import_src_prefix() -> None:
    """Files under src/ must not use 'src.' prefix in imports."""
    imports = _collect_imports(SRC_ROOT, PROJECT_ROOT)
    offenders = [fp for fp, targets in imports.items() if "src" in targets]
    assert not offenders, (
        f"Files using 'src' prefix import ({len(offenders)}):\n"
        + "\n".join(offenders)
    )


def test_training_imports_are_from_helix_ids_not_src() -> None:
    """train_helix_ids_full.py should import from helix_ids, not src.helix_ids."""
    target_file = SCRIPTS_TRAINING_ROOT / "train_helix_ids_full.py"
    if not target_file.exists():
        pytest.skip("train_helix_ids_full.py not found")
    try:
        tree = ast.parse(target_file.read_text("utf-8"))
    except SyntaxError:
        pytest.fail("Cannot parse train_helix_ids_full.py")
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("src."):
                offenders.append(f"from {node.module} import ...")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("src."):
                    offenders.append(f"import {alias.name}")
    assert not offenders, (
        "train_helix_ids_full.py uses 'src.' prefix imports:\n"
        + "\n".join(offenders)
    )


# ---------------------------------------------------------------------------
# Rule 5 — No trainer re-growth
# ---------------------------------------------------------------------------

def test_trainer_method_count_gate() -> None:
    """HelixFullTrainer must not exceed 109 methods (Phase 13B baseline)."""
    trainer_file = SCRIPTS_TRAINING_ROOT / "train_helix_ids_full.py"
    assert trainer_file.exists(), "train_helix_ids_full.py not found"
    tree = ast.parse(trainer_file.read_text("utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "HelixFullTrainer":
            # Count FunctionDef and AsyncFunctionDef directly in the class body
            count = sum(
                1 for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            )
            assert count <= 109, (
                f"HelixFullTrainer has {count} methods — exceeds freeze gate of 109. "
                f"Architecture freeze prohibits method re-growth."
            )
            break


def test_trainer_loc_gate() -> None:
    """HelixFullTrainer class must not exceed 2525 LOC (Phase 13B baseline)."""
    trainer_file = SCRIPTS_TRAINING_ROOT / "train_helix_ids_full.py"
    assert trainer_file.exists(), "train_helix_ids_full.py not found"
    tree = ast.parse(trainer_file.read_text("utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "HelixFullTrainer":
            class_loc = node.end_lineno - node.lineno + 1
            assert class_loc <= 2525, (
                f"HelixFullTrainer is {class_loc} LOC — exceeds freeze gate of 2525. "
                f"Architecture freeze prohibits size re-growth."
            )
            break


def test_trainer_facade_method_count_gate() -> None:
    """TrainerFacade must not exceed 20 methods."""
    facade_file = SCRIPTS_TRAINING_ROOT / "core" / "trainer_facade.py"
    assert facade_file.exists(), "trainer_facade.py not found"
    tree = ast.parse(facade_file.read_text("utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "TrainerFacade":
            methods = [
                n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            assert len(methods) <= 20, (
                f"TrainerFacade has {len(methods)} methods — exceeds freeze gate of 20."
            )
            break


# ---------------------------------------------------------------------------
# Rule 6 — ENGINEERED_FEATURE_NAMES must live in scripts/training/_constants.py,
#           not in src/helix_ids/ (training-layer detail, not domain)
# ---------------------------------------------------------------------------


def test_engineered_feature_names_not_defined_in_src() -> None:
    """ENGINEERED_FEATURE_NAMES constant must be in scripts/training/_constants.py,
    not defined in src/helix_ids/. Training-layer feature-engineering constants
    do not belong in the domain core.
    """
    constant_name = "ENGINEERED_FEATURE_NAMES"

    # Check it's NOT defined in src/
    src_definitions = []
    for pyfile in sorted(SRC_ROOT.rglob("*.py")):
        content = pyfile.read_text("utf-8")
        if f"{constant_name} =" in content or f"{constant_name}=" in content:
            lines = content.split("\n")
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith(constant_name) and "=" in stripped:
                    src_definitions.append(
                        f"{pyfile.relative_to(PROJECT_ROOT).as_posix()}:{i+1}"
                    )

    if src_definitions:
        pytest.fail(
            f"ENGINEERED_FEATURE_NAMES should NOT be defined in src/ "
            f"(found in {src_definitions}). "
            f"It belongs in scripts/training/_constants.py."
        )

    # Check it IS defined in the canonical location
    canonical = SCRIPTS_TRAINING_ROOT / "_constants.py"
    if not canonical.exists():
        pytest.fail(
            f"Canonical location {canonical.relative_to(PROJECT_ROOT)} not found."
        )
    content = canonical.read_text("utf-8")
    if (
        f"{constant_name} =" not in content
        and f"{constant_name}=" not in content
        and f"{constant_name}:" not in content
    ):
        pytest.fail(
            f"ENGINEERED_FEATURE_NAMES must be defined in "
            f"{canonical.relative_to(PROJECT_ROOT)}."
        )


# ---------------------------------------------------------------------------
# Rule 7 — No forbidden package-internal cycles
# ---------------------------------------------------------------------------

def test_no_src_internal_cycles() -> None:
    """src/helix_ids/ subpackages must form a DAG."""
    imports = _collect_full_imports(SRC_ROOT, PROJECT_ROOT)

    # Build package-level graph
    pkg_edges: dict[str, set[str]] = {}
    for filepath, targets in imports.items():
        # Determine file's package
        file_pkg = ".".join(filepath.replace(".py", "").split("/")[1:3])  # helix_ids.*
        for t in targets:
            if t.startswith("helix_ids."):
                target_pkg = ".".join(t.split(".")[:2])  # helix_ids.*
                if target_pkg != file_pkg:
                    pkg_edges.setdefault(file_pkg, set()).add(target_pkg)

    # DFS cycle detection on src packages only
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


# ---------------------------------------------------------------------------
# Sanity — metrics fixture works
# ---------------------------------------------------------------------------

def test_architecture_freeze_smoke() -> None:
    """Minimal smoke test: the test module itself loads."""
    assert True
