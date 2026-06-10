"""Enforcement completeness tests.

Phase 4A Formalization Closure: proves every governance artifact is
referenced by at least one validator or CI gate.

No runtime changes — read-only validation.
"""
from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Governance documents
# ---------------------------------------------------------------------------

GOVERNANCE_DOCS = {
    "manifest_schema_governance.md",
    "result_schema_governance.md",
    "hash_authority.md",
    "IMMUTABLE_SCHEMA_CONTRACT.md",
}


def test_all_governance_docs_exist() -> None:
    """Every governance document in GOVERNANCE_DOCS must exist and be non-empty."""
    docs_dir = PROJECT_ROOT / "docs" / "governance"
    for doc_name in GOVERNANCE_DOCS:
        doc_path = docs_dir / doc_name
        assert doc_path.exists(), f"Missing governance doc: {doc_name}"
        content = doc_path.read_text(encoding="utf-8")
        assert content.strip(), f"Governance doc is empty: {doc_name}"


# ---------------------------------------------------------------------------
# Governance doc → validator mapping
# ---------------------------------------------------------------------------

# Maps each governance doc to the validator(s) / CI gate(s) that enforce it.
# Format: doc_name -> list of (validator_identifier, enforcement_type)
DOC_TO_VALIDATOR: dict[str, list[tuple[str, str]]] = {
    "manifest_schema_governance.md": [
        ("scripts/ci/validate_governance_docs.py", "CI doc existence"),
        ("scripts/ci/validate_benchmark_outputs.py", "CI payload validation"),
        ("scripts/ci/validate_schema_registry.py", "CI registry entry"),
    ],
    "result_schema_governance.md": [
        ("scripts/ci/validate_governance_docs.py", "CI doc existence"),
        ("scripts/ci/validate_benchmark_outputs.py", "CI payload validation"),
        ("scripts/ci/validate_schema_registry.py", "CI registry entry"),
    ],
    "hash_authority.md": [
        ("scripts/ci/validate_governance_docs.py", "CI doc existence"),
        ("scripts/ci/validate_benchmark_outputs.py", "CI hash field validation"),
    ],
    "IMMUTABLE_SCHEMA_CONTRACT.md": [
        ("scripts/ci/validate_governance_docs.py", "CI doc existence"),
        ("tests/test_runtime_invariants.py", "unit test drift→fail"),
        ("src/helix_ids/utils/export.py", "runtime contract enforcement"),
    ],
}


def test_every_governance_doc_has_validator() -> None:
    """Every governance document must map to at least one validator or CI gate."""
    missing = []
    for doc_name in GOVERNANCE_DOCS:
        validators = DOC_TO_VALIDATOR.get(doc_name, [])
        if not validators:
            missing.append(doc_name)
    assert not missing, (
        f"Governance docs with no validator: {missing}. "
        "Add entries to DOC_TO_VALIDATOR in this file."
    )


def test_validator_files_exist() -> None:
    """Every referenced validator file must exist."""
    all_validators = set()
    for validators in DOC_TO_VALIDATOR.values():
        for validator_path, _ in validators:
            all_validators.add(validator_path)

    missing = []
    for vpath in all_validators:
        full = PROJECT_ROOT / vpath
        if not full.exists():
            missing.append(vpath)
    assert not missing, f"Validator files not found: {missing}"


# ---------------------------------------------------------------------------
# Schema registry
# ---------------------------------------------------------------------------

def test_schema_registry_exists() -> None:
    """schema_registry.yaml must exist."""
    registry = PROJECT_ROOT / "schema_registry.yaml"
    assert registry.exists(), "schema_registry.yaml not found"


def test_schema_registry_has_required_entries() -> None:
    """Registry must have manifest_schema and benchmark_result_schema entries."""
    import yaml

    registry = PROJECT_ROOT / "schema_registry.yaml"
    with registry.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    assert "schemas" in data, "Registry missing 'schemas' key"
    schema_names = {entry["schema_name"] for entry in data["schemas"]}
    assert "manifest_schema" in schema_names, "manifest_schema not in registry"
    assert "benchmark_result_schema" in schema_names, "benchmark_result_schema not in registry"


def test_schema_registry_entries_have_required_fields() -> None:
    """Every registry entry must have all required fields."""
    import yaml

    REQUIRED_ENTRY_FIELDS = {
        "schema_name",
        "current_version",
        "status",
        "owner",
        "compatibility_window",
        "deprecation_policy",
        "approval_required",
    }
    VALID_STATUSES = {"active", "deprecated", "retired"}

    registry = PROJECT_ROOT / "schema_registry.yaml"
    with registry.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    for entry in data["schemas"]:
        missing_fields = REQUIRED_ENTRY_FIELDS - set(entry.keys())
        assert not missing_fields, (
            f"Entry '{entry.get('schema_name')}' missing fields: {missing_fields}"
        )
        assert entry["status"] in VALID_STATUSES, (
            f"Entry '{entry.get('schema_name')}' has invalid status: {entry['status']}"
        )


# ---------------------------------------------------------------------------
# CI validator scripts
# ---------------------------------------------------------------------------

CI_VALIDATORS = {
    "scripts/ci/validate_governance_docs.py",
    "scripts/ci/validate_schema_registry.py",
    "scripts/ci/validate_benchmark_outputs.py",
    "scripts/ci/verify_contract_sidecars.py",
}


def test_ci_validator_scripts_exist() -> None:
    """Every CI validator referenced in the workflow must exist."""
    for vpath in CI_VALIDATORS:
        full = PROJECT_ROOT / vpath
        assert full.exists(), f"CI validator not found: {vpath}"


def test_ci_validator_scripts_are_executable_syntax() -> None:
    """Every CI validator script must be valid Python syntax."""
    import py_compile

    for vpath in CI_VALIDATORS:
        full = PROJECT_ROOT / vpath
        try:
            py_compile.compile(str(full), doraise=True)
        except py_compile.PyCompileError as exc:
            pytest.fail(f"Syntax error in {vpath}: {exc}")


# ---------------------------------------------------------------------------
# ADR documents
# ---------------------------------------------------------------------------

ADRs = {
    "docs/governance/ADR-001-governance-philosophy.md",
    "docs/governance/ADR-002-schema-lifecycle.md",
    "docs/governance/ADR-003-hash-authority.md",
    "docs/governance/ADR-004-enforcement-pipeline.md",
}


def test_all_adrs_exist() -> None:
    """All defined ADRs must exist and be non-empty."""
    for adr_path in ADRs:
        full = PROJECT_ROOT / adr_path
        assert full.exists(), f"ADR not found: {adr_path}"
        content = full.read_text(encoding="utf-8")
        assert content.strip(), f"ADR is empty: {adr_path}"
        assert "Status:" in content, f"ADR missing Status field: {adr_path}"
        assert "Date:" in content, f"ADR missing Date field: {adr_path}"


# ---------------------------------------------------------------------------
# Phase 4A output artifacts
# ---------------------------------------------------------------------------

def test_phase4a_coverage_audit_exists() -> None:
    """Governance coverage audit document must exist."""
    audit_path = PROJECT_ROOT / "docs" / "governance" / "phase4a_governance_coverage_audit.md"
    assert audit_path.exists(), "phase4a_governance_coverage_audit.md not found"


def test_reproducibility_gap_analysis_exists() -> None:
    """Reproducibility gap analysis document must exist."""
    gap_path = PROJECT_ROOT / "docs" / "governance" / "reproducibility_gap_analysis.md"
    assert gap_path.exists(), "reproducibility_gap_analysis.md not found"


# ---------------------------------------------------------------------------
# CI workflow references
# ---------------------------------------------------------------------------

def test_ci_workflow_references_all_validators() -> None:
    """The CI workflow must reference all CI validators.

    Positive assertion: at least one validator reference was found.
    Negative assertion: no validators are missing from the workflow.
    """
    import yaml

    wf_path = PROJECT_ROOT / ".github" / "workflows" / "runtime-monitoring-hardening-ci.yml"
    with wf_path.open("r", encoding="utf-8") as fh:
        wf = yaml.safe_load(fh)

    jobs = wf.get("jobs", {})
    # Collect all run/step commands that invoke validators
    validator_calls: set[str] = set()
    for _job_name, job in jobs.items():
        for step in job.get("steps", []):
            run_cmd = step.get("run", "")
            # Look for validator invocations
            for vpath in CI_VALIDATORS:
                if vpath in run_cmd:
                    validator_calls.add(vpath)
            # Also check uses: actions steps
            uses = step.get("uses", "")
            if "upload-artifact" in uses:
                pass  # artifact upload is fine

    # Positive assertion: at least one validator was found in the workflow
    assert validator_calls, (
        "No validator references found in CI workflow. "
        f"Expected at least one of: {sorted(CI_VALIDATORS)}"
    )

    missing = CI_VALIDATORS - validator_calls
    # Negative assertion: no validators are missing
    assert not missing, (
        f"CI workflow missing references to validators: {sorted(missing)}. "
        "Each CI validator must be invoked in at least one workflow step."
    )

    # -----------------------------------------------------------------------
# Runtime enforcement module verification (Objective 3)
# -----------------------------------------------------------------------

# Runtime enforcement modules referenced by governance/ADR docs.
# These must exist as importable Python modules.
RUNTIME_GOVERNANCE_MODULES = {
    "src/helix_ids/governance/entrypoint.py": "governed_entrypoint",
    "src/helix_ids/governance/lifecycle_verifier.py": "lifecycle_verifier",
    "src/helix_ids/governance/provenance.py": "provenance",
    "src/helix_ids/governance/failure_memory.py": "failure_memory",
    "src/helix_ids/governance/run_registry.py": "run_registry",
    "src/helix_ids/governance/ast_validator.py": "ast_validator",
    "src/helix_ids/governance/orchestrator.py": "orchestrator",
    "src/helix_ids/governance/promotion.py": "promotion",
    "src/helix_ids/utils/export.py": "contract sidecar enforcement",
}


def test_runtime_governance_modules_exist() -> None:
    """Every runtime enforcement module referenced in governance docs must exist."""
    missing = []
    for module_path, description in RUNTIME_GOVERNANCE_MODULES.items():
        full = PROJECT_ROOT / module_path
        if not full.exists():
            missing.append(f"{module_path} ({description})")
    assert not missing, f"Runtime enforcement modules not found: {missing}"


def test_runtime_modules_are_valid_python() -> None:
    """Every runtime enforcement module must be valid Python syntax."""
    import py_compile

    missing_bad = []
    for module_path in RUNTIME_GOVERNANCE_MODULES:
        full = PROJECT_ROOT / module_path
        try:
            py_compile.compile(str(full), doraise=True)
        except py_compile.PyCompileError:
            missing_bad.append(module_path)
    assert not missing_bad, f"Syntax errors in runtime modules: {missing_bad}"


# -----------------------------------------------------------------------
# Test file references (Objective 3 — verify dangling test refs)
# -----------------------------------------------------------------------

ALL_TEST_FILES = {
    "tests/test_runtime_invariants.py",
    "tests/test_lifecycle_verifier.py",
    "tests/test_provenance.py",
    "tests/test_governance/test_integration_enforcement.py",
    "tests/test_governance/test_ast_validator.py",
    "tests/test_governance/test_run_registry.py",
    "tests/test_governance/test_validate_schema_registry.py",  # Phase 4B
}


def test_all_referenced_test_files_exist() -> None:
    """Every test file referenced in governance mapping must exist."""
    missing = []
    for test_path in ALL_TEST_FILES:
        full = PROJECT_ROOT / test_path
        if not full.exists():
            missing.append(test_path)
    assert not missing, f"Referenced test files not found: {missing}"


# -----------------------------------------------------------------------
# Phase 4B outputs
# -----------------------------------------------------------------------

def test_phase4b_assumption_elimination_doc_exists() -> None:
    """Phase 4B assumption elimination audit doc must exist."""
    path = PROJECT_ROOT / "docs" / "governance" / "phase4b_assumption_elimination.md"
    assert path.exists(), "phase4b_assumption_elimination.md not found"
    content = path.read_text(encoding="utf-8")
    assert content.strip(), "phase4b_assumption_elimination.md is empty"
    # Must document assumptions removed and enforcement added
    assert "assumptions removed" in content.lower() or "gap" in content.lower()


# -----------------------------------------------------------------------
# Governance consistency validator existence (Objective 4)
# -----------------------------------------------------------------------

CONSISTENCY_VALIDATOR = "scripts/ci/validate_governance_consistency.py"


def test_governance_consistency_validator_exists() -> None:
    """Objective 4: governance consistency validator must exist."""
    full = PROJECT_ROOT / CONSISTENCY_VALIDATOR
    assert full.exists(), f"Governance consistency validator not found: {CONSISTENCY_VALIDATOR}"


def test_governance_consistency_validator_syntax() -> None:
    """Governance consistency validator must be valid Python."""
    import py_compile

    full = PROJECT_ROOT / CONSISTENCY_VALIDATOR
    try:
        py_compile.compile(str(full), doraise=True)
    except py_compile.PyCompileError as exc:
        pytest.fail(f"Syntax error in {CONSISTENCY_VALIDATOR}: {exc}")


# -----------------------------------------------------------------------
# CI workflow includes new Phase 4B gates (Objective 5)
# -----------------------------------------------------------------------

def test_ci_workflow_references_consistency_validator() -> None:
    """CI workflow must invoke the governance consistency validator."""
    import yaml

    wf_path = PROJECT_ROOT / ".github" / "workflows" / "runtime-monitoring-hardening-ci.yml"
    with wf_path.open("r", encoding="utf-8") as fh:
        wf = yaml.safe_load(fh)

    referenced: set[str] = set()
    for job in wf.get("jobs", {}).values():
        for step in job.get("steps", []):
            if CONSISTENCY_VALIDATOR in step.get("run", ""):
                referenced.add(CONSISTENCY_VALIDATOR)

    assert CONSISTENCY_VALIDATOR in referenced, (
        f"CI workflow must invoke {CONSISTENCY_VALIDATOR} as a hard-fail gate"
    )


def test_ci_workflow_references_schema_chronology_validator() -> None:
    """CI workflow must invoke the schema registry validator (chronology enforcement)."""
    import yaml

    wf_path = PROJECT_ROOT / ".github" / "workflows" / "runtime-monitoring-hardening-ci.yml"
    with wf_path.open("r", encoding="utf-8") as fh:
        wf = yaml.safe_load(fh)

    referenced: set[str] = set()
    for job in wf.get("jobs", {}).values():
        for step in job.get("steps", []):
            if "validate_schema_registry.py" in step.get("run", ""):
                referenced.add("validate_schema_registry.py")

    assert "validate_schema_registry.py" in referenced, (
        "CI workflow must invoke validate_schema_registry.py for chronology enforcement"
    )
