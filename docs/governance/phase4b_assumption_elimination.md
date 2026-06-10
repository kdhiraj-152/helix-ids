# Phase 4B — Assumption Elimination

**Date:** 2026-06-03
**Phase:** 4B — Formalization Closure
**Purpose:** Document assumptions eliminated in this phase and remaining gaps.
No future roadmap, no feature proposals.

---

## 1. Assumptions Removed

| Assumption (Phase 4A Gap) | Severity | Removal Mechanism | Validation |
|---------------------------|----------|-------------------|------------|
| `IMMUTABLE_SCHEMA_CONTRACT.md` not enforced in CI | Medium | Added to `REQUIRED_DOCS` in `scripts/ci/validate_governance_docs.py` | `test_all_governance_docs_exist` |
| Registry version monotonicity not machine-enforced | Medium | Added entry-list chronology check to `scripts/ci/validate_schema_registry.py` step 4 | 8 new tests in `tests/test_governance/test_validate_schema_registry.py` |
| `config_hashes` sub-key shape not validated | Low | Added `_check_config_hashes()` to `scripts/ci/validate_benchmark_outputs.py` | 10 new tests in `tests/test_governance/test_nested_schema_validation.py` |
| `dataset_hashes` sub-key values not validated for non-empty | Low | Extended `_check_dataset_hashes()` to require non-empty string values | `tests/test_governance/test_nested_schema_validation.py` |

---

## 2. Enforcement Added

### 2.1 Schema Registry Chronology (validate_schema_registry.py)

- **What:** Entries in `schema_registry.yaml` must appear in monotonically non-decreasing `current_version` order.
- **How:** New step 4 in `validate_registry()` iterates entries left-to-right, tracking `prev_date`. Retired entries update `prev_date` but do not trigger violation (historical markers).
- **Fail condition:** `parsed(version)[i] < parsed(version)[i-1]` for non-retired entry.
- **Tests:** 8 tests covering valid/invalid chronology, malformed version, equal versions, retired-entry exception.

### 2.2 Nested Config Hashes (validate_benchmark_outputs.py)

- **What:** `config_hashes` object in manifests — each present value must be a non-empty string. No specific key names are mandated; the validator checks that any key present has a non-missing, non-empty scalar value. An empty `{}` mapping also fails.
- **How:** New `_check_config_hashes()` called from `_validate_manifest_file`.
- **Fail conditions:**
  - `config_hashes` is absent or not a mapping
  - `config_hashes` is an empty mapping `{}`
  - Any present value is `None`, empty string, or whitespace-only string
- **Tests:** 10 tests covering all missing/empty/malformed cases.

### 2.3 Nested Dataset Hashes Non-Empty (validate_benchmark_outputs.py)

- **What:** `dataset_hashes.{raw,processed,split,primary}` must be non-empty strings (not just present).
- **How:** Extended `_check_dataset_hashes()` with `_missing_scalar` check per sub-key.
- **Tests:** 12 tests in `tests/test_governance/test_nested_schema_validation.py`.

### 2.4 Governance Doc → Validator Mapping (test_enforcement_completeness.py)

- **What:** Every governance document must map to at least one existing validator. Every runtime module referenced in governance docs must exist as a valid Python file.
- **How:** New `DOC_TO_VALIDATOR` cross-reference map; `test_every_governance_doc_has_validator()`, `test_validator_files_exist()`, `test_runtime_governance_modules_exist()`, `test_all_referenced_test_files_exist()`.
- **Fail conditions:**
  - Governance doc has no validator mapping
  - Validator file does not exist on disk
  - Runtime module does not exist
  - Referenced test file does not exist

### 2.5 Governance Consistency Gate (validate_governance_consistency.py)

- **What:** ADR claims are checked against actual validator implementation capability.
- **Checks:**
  1. ADR-002 monotonicity claim → validator has format parsing + chronology enforcement
  2. Enforcement claims in docs → referenced validator files exist
  3. Schema name references → referenced schemas exist in registry
  4. False enforcement claims (advisory only)
- **Output:** `results/gates/governance_consistency_validation.json`
- **CI integration:** Called as hard-fail gate in `schema_governance` job.

### 2.6 CI Workflow Hardening

- **Added:** `validate_governance_consistency.py` invocation as hard-fail step in `schema_governance` job.
- **Artifact:** `results/gates/governance_consistency_validation.json` uploaded alongside other gate reports.

---

## 3. Remaining Non-Machine-Enforced Policies

The following governance policies are documented but intentionally not fully machine-enforced. Rationale is provided for each exclusion.

| Policy | Document | Enforcement Gap | Rationale for Exclusion |
|--------|----------|----------------|-------------------------|
| Migration freeze policy | `IMMUTABLE_SCHEMA_CONTRACT.md` | No automated freeze-gate enforcement; manual PR review required | Freeze gates require human judgment about schema change severity |
| Producer/consumer obligations | `IMMUTABLE_SCHEMA_CONTRACT.md` | Policy statements not programmatically verifiable | Behavioral obligations require human interpretation |
| Schema versioning approval requirements | `manifest_schema_governance.md` §8 | No automated approval tracking; relies on PR review | Approval is a social/governance process, not detectable from artifacts |
| ADR claim correctness | `ADR-00X` series | `validate_governance_consistency.py` check 4 is advisory only; may produce false positives | Natural-language claim detection is heuristic; manual review still required |
| `determinism` sub-object flag values | `result_schema_governance.md` §4 | Field presence checked; actual torch flag values not independently verified | Platform-dependent; would require runtime instrumentation |
| `config_hashes` sub-key semantic validity | `manifest_schema_governance.md` | Sub-key presence and non-emptiness checked; hash validity not cryptographically verified | Would require re-running full training pipeline |
| Frozen dependencies | `ADR-001` | No `requirements.lock` in provenance chain | Environment reproducibility achieved via deterministic ops + seeds; pip lock out of scope |
| Training log incorporation | `ADR-001` | Logs written but not hashed into provenance | Log format is not stable; would require schema for log metadata |

---

## 4. Rationale for Exclusions

### Why not fully automated freeze-gate?
Schema migration decisions involve weighing breaking-change impact across consumers. Automated systems cannot assess whether a change is acceptable — this requires governance-lead judgment. The CI gate enforces process (registry update, CI green, PR review) but not the judgment itself.

### Why is ADR claim check advisory?
Natural-language claim detection in `validate_governance_consistency.py` check 4 uses regex patterns that may produce false positives (e.g., "CI enforces" in a comment about intended future work). Manual review is still needed for edge cases.

### Why not cryptographically verify config_hashes?
Verifying that a SHA-256 hash matches the actual configuration would require re-executing the full training pipeline with the same inputs. This is deliberately out of scope — the hash is treated as authoritative per `hash_authority.md`.

### Why not verify `determinism` sub-object values?
Torch deterministic flag verification would require runtime platform instrumentation and is inherently environment-specific. The current enforcement (presence check) ensures the field is recorded, enabling post-hoc audit.

---

## 5. Formalization Completeness Summary

| Phase 4A Gap | Status |
|-------------|--------|
| `IMMUTABLE_SCHEMA_CONTRACT.md` not in CI REQUIRED_DOCS | **Resolved** |
| Registry version monotonicity not enforced | **Resolved** |
| `config_hashes` sub-key shape not validated | **Resolved** |
| `dataset_hashes` sub-key values not validated | **Resolved** |
| ADR claims vs implementation drift | **Resolved** (advisory + hard gate for concrete claims) |
| Documentation → validator reference drift | **Resolved** |
| Runtime module existence not verified | **Resolved** |

No new runtime features, behavioral changes, or product capabilities introduced in this phase.