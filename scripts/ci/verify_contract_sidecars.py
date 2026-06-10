#!/usr/bin/env python3
"""AST-based contract enforcement for artifact producers.

This guard ensures any on-disk torch.save usage is paired with manifest
sidecars and provenance finalization, and it flags forbidden runtime
compatibility calls in production/ingress code paths.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SCAN_DIRS = [ROOT / "src", ROOT / "scripts"]
SKIP_PARTS = {".venv", "venv", "site-packages", "__pycache__"}
SKIP_DIRS = {"tests", "fixtures"}

MANIFEST_MARKERS = {
    "write_contract_sidecars",
    "finalize_export_artifact",
    "finalize_artifact_manifest",
    "write_artifact_manifest_sidecar",
}

FORBIDDEN_CALL_ATTRS = {"warn", "reindex", "align", "pad", "truncate", "repair", "recover"}
FORBIDDEN_SCOPES = (
    "src/helix_ids/operations",
    "src/helix_ids/governance",
    "src/helix_ids/utils",
    "scripts/operations",
    "scripts/deployment",
)


def _iter_python_files() -> list[Path]:
    files: list[Path] = []
    for base in SCAN_DIRS:
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            if any(part in SKIP_PARTS for part in path.parts):
                continue
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            if path.name.startswith("test_"):
                continue
            files.append(path)
    return files


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Attribute):
        if isinstance(func.value, ast.Name):
            return f"{func.value.id}.{func.attr}"
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _is_bytesio_call(node: ast.AST | None) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name) and func.id == "BytesIO":
        return True
    if isinstance(func, ast.Attribute) and func.attr == "BytesIO":
        return True
    return False


def _collect_bytesio_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not _is_bytesio_call(node.value):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                names.add(target.id)
    return names


def _is_torch_save(node: ast.Call) -> bool:
    func = node.func
    return isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "torch" and func.attr == "save"


def _is_in_memory_save(node: ast.Call, bytesio_names: set[str]) -> bool:
    # torch.save(obj, f=BytesIO()) or torch.save(obj, buffer)
    dest: ast.AST | None = None
    if len(node.args) > 1:
        dest = node.args[1]
    for keyword in node.keywords:
        if keyword.arg in {"f", "file"}:
            dest = keyword.value
    if _is_bytesio_call(dest):
        return True
    if isinstance(dest, ast.Name) and dest.id in bytesio_names:
        return True
    return False


def _in_forbidden_scope(path: Path) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    return any(rel.startswith(scope) for scope in FORBIDDEN_SCOPES)


def _iter_calls(tree: ast.AST) -> list[ast.Call]:
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)]


def _is_manifest_marker(call: ast.Call, name: str) -> bool:
    func = call.func
    if name in MANIFEST_MARKERS:
        return True
    return isinstance(func, ast.Name) and func.id in MANIFEST_MARKERS


def _is_forbidden_call(call: ast.Call, name: str) -> bool:
    func = call.func
    if isinstance(func, ast.Attribute) and func.attr in FORBIDDEN_CALL_ATTRS:
        return True
    return name == "warnings.warn"


def _collect_manifest_markers(calls: list[ast.Call]) -> bool:
    for call in calls:
        name = _call_name(call)
        if _is_manifest_marker(call, name):
            return True
    return False


def _collect_torch_save_lines(calls: list[ast.Call], bytesio_names: set[str]) -> list[int]:
    lines: list[int] = []
    for call in calls:
        if _is_torch_save(call) and not _is_in_memory_save(call, bytesio_names):
            lines.append(call.lineno)
    return lines


def _collect_forbidden_lines(calls: list[ast.Call]) -> list[int]:
    lines: list[int] = []
    for call in calls:
        name = _call_name(call)
        if _is_forbidden_call(call, name):
            lines.append(call.lineno)
    return lines


def analyze_file(path: Path) -> list[str]:
    issues: list[str] = []
    source = path.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"{path}: AST parse failed: {exc}"]

    bytesio_names = _collect_bytesio_names(tree)
    calls = _iter_calls(tree)
    has_manifest_marker = _collect_manifest_markers(calls)
    torch_save_lines = _collect_torch_save_lines(calls, bytesio_names)
    forbidden_lines = _collect_forbidden_lines(calls) if _in_forbidden_scope(path) else []

    if torch_save_lines and not has_manifest_marker:
        line_list = ", ".join(str(line) for line in torch_save_lines)
        issues.append(f"{path}: torch.save without manifest/sidecar enforcement (lines {line_list})")
    if forbidden_lines:
        line_list = ", ".join(str(line) for line in forbidden_lines)
        issues.append(f"{path}: forbidden runtime compatibility calls (lines {line_list})")
    return issues


def main() -> None:
    issues: list[str] = []
    for path in _iter_python_files():
        issues.extend(analyze_file(path))
    if issues:
        print("❌ Contract enforcement violations found:")
        for issue in issues:
            print(f" - {issue}")
        sys.exit(1)
    print("✅ AST contract enforcement passed.")


if __name__ == "__main__":
    main()
