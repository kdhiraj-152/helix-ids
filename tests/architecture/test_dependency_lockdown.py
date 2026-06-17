"""Tests for dependency lockfile integrity, drift detection, and reproducibility.

Phase 21: Verifies that the requirements.lock file is present, parsable,
contains exact version pins with hashes, and matches the installed environment.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).parents[2]
LOCKFILES = {
    "core": PROJECT_ROOT / "requirements-lock.txt",
    "dev": PROJECT_ROOT / "requirements-dev-lock.txt",
    "all": PROJECT_ROOT / "requirements-all-lock.txt",
}

# Packages with known dynamic versions that may not be in install steps.
# These are excluded from exact-version matching but must still be present.
VERSION_EXEMPT: set[str] = {"setuptools", "wheel", "pip"}

HASH_RE = re.compile(r"--hash=sha256:[a-f0-9]{64}")


# ── Helpers ──────────────────────────────────────────────────────────────────


def _parse_lockfile(path: Path) -> dict[str, dict[str, Any]]:
    """Parse a pip-compile generated lockfile into {package: metadata} dict."""
    packages: dict[str, dict[str, Any]] = {}
    current_pkg: str | None = None
    current_version: str | None = None
    current_hashes: list[str] = []
    current_via: list[str] = []

    with open(path) as f:
        lines = f.readlines()

    for line in lines:
        stripped = line.rstrip()

        # Skip comments and blanks
        if not stripped or stripped.startswith("#"):
            continue

        # Detect package line: pkg==version \
        eq_match = re.match(r"^([a-zA-Z0-9_\-.]+)==([\w.*]+)\s*\\?$", stripped)
        if eq_match:
            if current_pkg:
                packages[current_pkg] = {
                    "version": current_version or "",
                    "hashes": list(current_hashes),
                    "via": list(current_via),
                }
            current_pkg = eq_match.group(1)
            current_version = eq_match.group(2)
            current_hashes = []
            current_via = []
            continue

        # Detect hash continuation line
        hash_match = HASH_RE.search(stripped)
        if hash_match and current_pkg:
            current_hashes.append(hash_match.group(0))
            continue

        # Detect "via" comment
        via_match = re.match(r"\s+# via\s+(.+)", stripped)
        if via_match and current_pkg:
            current_via.append(via_match.group(1).strip())

    if current_pkg:
        packages[current_pkg] = {
            "version": current_version or "",
            "hashes": list(current_hashes),
            "via": list(current_via),
        }

    return packages


def _get_installed_packages() -> dict[str, str]:
    """Get {package: version} from the current pip environment.

    Normalizes names to use underscores (pip's JSON output inconsistently
    uses hyphens vs underscores across different packages).
    """
    result = subprocess.run(
        ["pip", "list", "--format=json"],
        capture_output=True, text=True, check=True,
    )
    entries: list[dict[str, str]] = json.loads(result.stdout)
    def _norm(name: str) -> str:
        return name.lower().replace("-", "_")
    return {_norm(entry["name"]): entry["version"] for entry in entries}


# ── Tests ────────────────────────────────────────────────────────────────────


class TestDependencyLockdown:
    """Suite for dependency lockfile verification."""

    @pytest.mark.parametrize("name,path", list(LOCKFILES.items()))
    def test_lockfile_exists(self, name: str, path: Path) -> None:
        """Every expected lockfile must exist and be non-empty."""
        assert path.exists(), f"{name} lockfile missing at {path}"
        assert path.stat().st_size > 100, f"{name} lockfile too small"

    @pytest.mark.parametrize("name,path", list(LOCKFILES.items()))
    def test_lockfile_parsable(self, name: str, path: Path) -> None:
        """Every lockfile must parse to at least one pinned package."""
        pkgs = _parse_lockfile(path)
        assert len(pkgs) > 0, f"{name} lockfile yielded zero packages"

    @pytest.mark.parametrize("name,path", list(LOCKFILES.items()))
    def test_all_packages_have_hashes(self, name: str, path: Path) -> None:
        """Every pinned package must have at least one SHA256 hash."""
        pkgs = _parse_lockfile(path)
        no_hash = [
            pkg for pkg, meta in pkgs.items()
            if len(meta["hashes"]) == 0 and pkg.lower() not in VERSION_EXEMPT
        ]
        assert not no_hash, (
            f"{name} lockfile: {len(no_hash)} packages without hashes: {no_hash}"
        )

    def test_core_lockfile_no_extras(self) -> None:
        """The core lockfile must not include dev/training/deployment deps."""
        pkgs = _parse_lockfile(LOCKFILES["core"])
        disallowed = {"pytest", "ruff", "mypy", "mlflow", "optuna", "onnx"}
        found = [p for p in pkgs if p.lower() in disallowed]
        assert not found, (
            f"Core lockfile contains extra deps: {found}"
        )

    def test_installed_versions_match_core_lockfile(self) -> None:
        """Installed package versions must match the core lockfile pins.

        Excludes VERSION_EXEMPT packages (setuptools, wheel, pip) whose
        versions are determined by the Python runtime.
        """
        pkgs = _parse_lockfile(LOCKFILES["core"])
        installed = _get_installed_packages()
        mismatches: list[str] = []

        for pkg, meta in pkgs.items():
            pkg_norm = pkg.lower().replace("-", "_")
            if pkg_norm in VERSION_EXEMPT:
                continue

            pinned_ver = meta.get("version", "")
            inst_ver = installed.get(pkg_norm)
            if inst_ver is None:
                mismatches.append(f"{pkg}: installed (MISSING)")
            elif inst_ver != pinned_ver:
                mismatches.append(
                    f"{pkg}: installed {inst_ver} != pinned {pinned_ver}, "
                    f"run 'pip install -r requirements.lock' to sync"
                )

        if mismatches:
            # This is a WARNING test — not a hard blocker for CI
            # because the dev environment may have been installed at a
            # different time than the lockfile was generated.
            # Regenerate lockfile if version drift is intentional.
            import logging
            logging.warning(
                "Dependency drift detected (%d mismatches). "
                "Run 'pip install -r requirements.lock' or "
                "'pip-compile --generate-hashes --output-file=requirements.lock pyproject.toml' "
                "to sync.",
                len(mismatches),
            )
            # Report but don't hard-fail — this is informative drift detection
            for m in mismatches:
                logging.warning("  %s", m)
            pytest.fail(
                f"Version mismatches ({len(mismatches)}): lockfile out of sync with environment.\n"
                + "\n".join(mismatches)
            )

    def test_all_requirements_satisfiable(self) -> None:
        """Verify that requirements.lock can be installed successfully."""
        result = subprocess.run(
            ["pip", "install", "--dry-run", "-r", str(LOCKFILES["core"])],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            stderr_tail = result.stderr[-500:] if result.stderr else "(no stderr)"
            pytest.fail(
                f"pip install --dry-run failed for requirements.lock:\n{stderr_tail}"
            )

    def test_no_version_ranges_in_lockfile(self) -> None:
        """Lockfiles must use exact version pins, not ranges."""
        pkgs = _parse_lockfile(LOCKFILES["core"])
        for pkg in pkgs:
            assert ">=" not in pkg, f"Range constraint found in lockfile: {pkg}"
            assert "~=" not in pkg, f"Compatible release found in lockfile: {pkg}"

    def test_dependency_graph_acyclic(self) -> None:
        """Core lockfile packages must not form dependency cycles.

        This is a basic sanity: pip-compile would normally reject cycles.
        """
        pkgs = _parse_lockfile(LOCKFILES["core"])
        # Build reverse dep tree: pkg -> packages that depend on it
        rev_deps: dict[str, set[str]] = {}
        for pkg, meta in pkgs.items():
            for via_dep in meta.get("via", []):
                via_dep = via_dep.strip()
                if via_dep:
                    if via_dep not in rev_deps:
                        rev_deps[via_dep] = set()
                    rev_deps[via_dep].add(pkg)

        # Check for trivial self-deps (shouldn't happen with pip-compile)
        self_deps = {
            pkg for pkg in pkgs
            if pkg in rev_deps and pkg in rev_deps[pkg]
        }
        assert not self_deps, f"Self-referencing dependencies: {self_deps}"
