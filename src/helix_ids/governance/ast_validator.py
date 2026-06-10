"""AST-based governance enforcement for contract-sensitive code."""

from __future__ import annotations

import argparse
import ast
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ALLOWLIST_SUFFIXES: tuple[str, ...] = (
    "feature_engineering.py",
)
DEFAULT_SENSITIVE_PATH_TOKENS: tuple[str, ...] = (
    "runtime",
    "governance",
    "export",
    "inference",
    "contracts",
)
DEFAULT_EXCLUDE_DIRS: tuple[str, ...] = (
    "tests",
    "fixtures",
    "notebooks",
    "artifacts",
    "results",
    "checkpoints",
)
APPROVED_SERIALIZATION_MODULES: tuple[str, ...] = (
    "src/helix_ids/utils/export.py",
    "src/helix_ids/governance/provenance.py",
    "src/helix_ids/governance/lifecycle_verifier.py",
)

FORBIDDEN_SILENT_CALLS: tuple[str, ...] = (
    "fillna",
    "replace",
    "reindex",
    "align",
    "infer_objects",
    "astype",
)
FALLBACK_KEYWORDS: tuple[str, ...] = (
    "fallback",
    "auto_fix",
)
MANIFEST_WIRING_CALLS: tuple[str, ...] = (
    "build_artifact_manifest",
    "build_provenance_chain",
    "checkpoint_manifest_payload",
    "finalize_artifact_manifest",
    "finalize_export_artifact",
    "verify_export_artifact",
    "write_contract_sidecars",
)


@dataclass(frozen=True)
class ASTViolation:
    file: str
    line: int
    rule_id: str
    symbol: str
    message: str

    def as_dict(self) -> dict[str, object]:
        return {
            "file": self.file,
            "line": self.line,
            "rule_id": self.rule_id,
            "symbol": self.symbol,
            "message": self.message,
        }


@dataclass(frozen=True)
class ASTValidatorConfig:
    allowlist_suffixes: tuple[str, ...] = DEFAULT_ALLOWLIST_SUFFIXES
    sensitive_path_tokens: tuple[str, ...] = DEFAULT_SENSITIVE_PATH_TOKENS
    exclude_dirs: tuple[str, ...] = DEFAULT_EXCLUDE_DIRS
    manifest_wiring_calls: tuple[str, ...] = MANIFEST_WIRING_CALLS
    approved_serialization_suffixes: tuple[str, ...] = APPROVED_SERIALIZATION_MODULES


class ASTValidator:
    def __init__(self, config: ASTValidatorConfig | None = None) -> None:
        self.config = config or ASTValidatorConfig()

    def validate_paths(self, paths: Sequence[Path]) -> list[ASTViolation]:
        files = discover_python_files(paths, exclude_dirs=self.config.exclude_dirs)
        violations: list[ASTViolation] = []
        for path in files:
            violations.extend(self.validate_file(path))
        return sorted(
            violations,
            key=lambda item: (item.file, item.line, item.rule_id, item.symbol),
        )

    def validate_file(self, path: Path) -> list[ASTViolation]:
        content = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(content, filename=str(path))
        except SyntaxError as exc:
            return [
                ASTViolation(
                    file=_relative_path(path),
                    line=exc.lineno or 1,
                    rule_id="GOV900",
                    symbol="syntax_error",
                    message="AST parse failed",
                )
            ]
        visitor = _RuleVisitor(path, self.config)
        visitor.visit(tree)
        return visitor.violations


class _RuleVisitor(ast.NodeVisitor):
    def __init__(self, path: Path, config: ASTValidatorConfig) -> None:
        self.path = path
        self.config = config
        self.violations: list[ASTViolation] = []
        self._allowlisted = _is_allowlisted(path, config.allowlist_suffixes)
        self._sensitive = _is_sensitive(path, config.sensitive_path_tokens)
        self._serialization_allowed = _is_allowlisted(path, config.approved_serialization_suffixes)

    def visit_Call(self, node: ast.Call) -> None:
        name = _call_name(node.func)
        if name in FORBIDDEN_SILENT_CALLS and not self._allowlisted:
            self._add_violation(
                node,
                rule_id="GOV001",
                symbol=name,
                message="Forbidden silent repair outside canonical derivation layer",
            )
        if self._sensitive:
            if _is_warnings_warn(node.func):
                self._add_violation(
                    node,
                    rule_id="GOV010",
                    symbol="warnings.warn",
                    message="Runtime fallback warnings are forbidden in sensitive paths",
                )
            for keyword in node.keywords:
                if keyword.arg in FALLBACK_KEYWORDS:
                    self._add_violation(
                        node,
                        rule_id="GOV013",
                        symbol=f"{keyword.arg}=",
                        message="Fallback/auto-fix behavior is forbidden in sensitive paths",
                    )
            if _is_sorted_columns_call(node) or _is_set_columns_call(node):
                self._add_violation(
                    node,
                    rule_id="GOV030",
                    symbol=name or "sorted/set",
                    message="Dynamic schema mutation is forbidden in sensitive paths",
                )
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if self._sensitive and any(_is_columns_assignment(target) for target in node.targets):
            self._add_violation(
                node,
                rule_id="GOV030",
                symbol="columns=",
                message="Dynamic schema mutation is forbidden in sensitive paths",
            )
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        if self._sensitive:
            for handler in node.handlers:
                if _is_broad_exception(handler):
                    self._add_violation(
                        handler,
                        rule_id="GOV011",
                        symbol=_handler_symbol(handler),
                        message="Broad exception handling is forbidden in sensitive paths",
                    )
                if _is_pass_only(handler):
                    self._add_violation(
                        handler,
                        rule_id="GOV012",
                        symbol="except: pass",
                        message="try/except pass is forbidden in sensitive paths",
                    )
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_artifact_bypass(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._check_artifact_bypass(node)
        self.generic_visit(node)

    def visit_Module(self, node: ast.Module) -> None:
        self._check_artifact_bypass(node)
        self.generic_visit(node)

    def _check_artifact_bypass(self, node: ast.AST) -> None:
        calls = _gather_calls_excluding_scopes(node)
        wiring_present = any(_call_name(call.func) in self.config.manifest_wiring_calls for call in calls)
        for call in calls:
            if _is_pickle_dump(call):
                self._add_violation(
                    call,
                    rule_id="GOV022",
                    symbol="pickle.dump",
                    message="Ungoverned serializer is forbidden",
                )
                continue
            if _is_joblib_dump(call):
                self._add_violation(
                    call,
                    rule_id="GOV022",
                    symbol="joblib.dump",
                    message="Ungoverned serializer is forbidden",
                )
                continue
            if _is_torch_save(call):
                if not self._serialization_allowed:
                    self._add_violation(
                        call,
                        rule_id="GOV021",
                        symbol="torch.save",
                        message="Serialization only allowed through governed exporters",
                    )
                    continue
                if not wiring_present:
                    self._add_violation(
                        call,
                        rule_id="GOV020",
                        symbol="torch.save",
                        message="Artifact save without manifest/provenance wiring",
                    )
                    continue
                if _args_include_state_dict(call.args):
                    continue

    def _add_violation(self, node: ast.AST, *, rule_id: str, symbol: str, message: str) -> None:
        line = getattr(node, "lineno", 1)
        self.violations.append(
            ASTViolation(
                file=_relative_path(self.path),
                line=int(line),
                rule_id=rule_id,
                symbol=symbol,
                message=message,
            )
        )


def _relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _is_allowlisted(path: Path, allowlist_suffixes: Sequence[str]) -> bool:
    path_str = path.as_posix()
    for suffix in allowlist_suffixes:
        if path_str.endswith(suffix):
            return True
    return False


def _is_sensitive(path: Path, tokens: Sequence[str]) -> bool:
    path_str = path.as_posix().lower()
    return any(token in path_str for token in tokens)


def _call_name(func: ast.AST) -> str:
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _is_warnings_warn(func: ast.AST) -> bool:
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "warn"
        and isinstance(func.value, ast.Name)
        and func.value.id == "warnings"
    )


def _is_broad_exception(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    if isinstance(handler.type, ast.Name) and handler.type.id in {"Exception", "BaseException"}:
        return True
    if isinstance(handler.type, ast.Attribute) and handler.type.attr in {"Exception", "BaseException"}:
        return True
    return False


def _handler_symbol(handler: ast.ExceptHandler) -> str:
    if handler.type is None:
        return "except:"
    if isinstance(handler.type, ast.Name):
        return f"except {handler.type.id}"
    if isinstance(handler.type, ast.Attribute):
        return f"except {handler.type.attr}"
    return "except"


def _is_pass_only(handler: ast.ExceptHandler) -> bool:
    return bool(handler.body) and all(isinstance(node, ast.Pass) for node in handler.body)


def _is_torch_save(call: ast.Call) -> bool:
    func = call.func
    if isinstance(func, ast.Attribute) and func.attr == "save":
        return isinstance(func.value, ast.Name) and func.value.id == "torch"
    if isinstance(func, ast.Name) and func.id == "save":
        return True
    return False


def _is_pickle_dump(call: ast.Call) -> bool:
    func = call.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "dump"
        and isinstance(func.value, ast.Name)
        and func.value.id == "pickle"
    )


def _is_joblib_dump(call: ast.Call) -> bool:
    func = call.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "dump"
        and isinstance(func.value, ast.Name)
        and func.value.id == "joblib"
    )


def _args_include_state_dict(args: Sequence[ast.AST]) -> bool:
    for arg in args:
        if _contains_state_dict(arg):
            return True
    return False


def _contains_state_dict(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id.endswith("state_dict")
    if isinstance(node, ast.Attribute):
        return node.attr == "state_dict"
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        return node.func.attr == "state_dict"
    if isinstance(node, ast.Dict):
        return any(_contains_state_dict(value) for value in node.values)
    return False


def _is_columns_assignment(target: ast.AST) -> bool:
    return isinstance(target, ast.Attribute) and target.attr == "columns"


def _is_sorted_columns_call(call: ast.Call) -> bool:
    if not (isinstance(call.func, ast.Name) and call.func.id == "sorted"):
        return False
    return any(_is_columns_arg(arg) for arg in call.args)


def _is_set_columns_call(call: ast.Call) -> bool:
    if not (isinstance(call.func, ast.Name) and call.func.id == "set"):
        return False
    return any(_is_columns_arg(arg) for arg in call.args)


def _is_columns_arg(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id in {"columns", "column_names"}
    if isinstance(node, ast.Attribute):
        return node.attr == "columns"
    return False


def discover_python_files(paths: Sequence[Path], *, exclude_dirs: Sequence[str]) -> list[Path]:
    python_files: list[Path] = []
    excluded = {item.lower() for item in exclude_dirs}
    for path in paths:
        if path.is_file() and path.suffix == ".py":
            if not _is_excluded(path, excluded):
                python_files.append(path)
            continue
        if path.is_dir():
            for candidate in path.rglob("*.py"):
                if not _is_excluded(candidate, excluded):
                    python_files.append(candidate)
    return sorted(set(python_files), key=lambda item: item.as_posix())


def _is_excluded(path: Path, excluded: set[str]) -> bool:
    return any(part.lower() in excluded for part in path.parts)


def _gather_calls_excluding_scopes(node: ast.AST) -> list[ast.Call]:
    calls: list[ast.Call] = []
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        if isinstance(child, ast.Call):
            calls.append(child)
        calls.extend(_gather_calls_excluding_scopes(child))
    return calls


def validate_paths(
    paths: Sequence[Path | str],
    *,
    config: ASTValidatorConfig | None = None,
) -> list[ASTViolation]:
    resolved = [Path(path) for path in paths]
    validator = ASTValidator(config=config)
    return validator.validate_paths(resolved)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Helix governance AST validator")
    parser.add_argument("--paths", nargs="+", default=["src", "scripts"], help="Paths to scan")
    parser.add_argument("--ci", action="store_true", help="Enable CI mode for fail-fast checks")
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    args = parser.parse_args(argv)

    violations = validate_paths(args.paths)
    payload = [violation.as_dict() for violation in violations]
    output = json.dumps(payload, sort_keys=True, indent=2)
    print(output)
    if violations:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
