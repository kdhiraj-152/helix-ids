# ADR-004: Enforcement Pipeline Architecture

**Status:** Accepted
**Date:** 2026-06-03
**Deciders:** HELIX-IDS governance lead

---

## Context

HELIX-IDS enforces governance through a layered CI + runtime pipeline. This ADR documents the enforcement architecture and identifies coverage gaps.

## Current Guarantees

### CI Layer

| Job | Trigger | Enforcement |
|-----|---------|-------------|
| `checks` | Every PR | `pytest -q`, `py_compile`, `verify_contract_sidecars.py` |
| `governance_ast` | Every PR | `ast_validator.py --ci --json` — no dynamic code |
| `contract_lifecycle` | After `governance_ast` | Export contract tests, runtime invariants, lifecycle verifier |
| `schema_governance` | After `governance_ast` | Schema registry validation, governance doc existence |
| `benchmark_enforcement` | After `contract_lifecycle` + `schema_governance` | Manifest expansion, benchmark dry-run, output validation |

### Runtime Layer

| Component | Enforcement |
|-----------|-------------|
| `governed_entrypoint` | Non-bypassable; fails if governance context missing |
| `lifecycle_verifier` | Stage timeout enforcement, quality gates, promotion contract |
| `ast_validator` | Runs on source files; blocks unsafe patterns |
| `run_registry` | Persists every governed run for audit |
| `failure_memory` | Stores failure events for post-hoc analysis |

## Enforcement Coverage Matrix

| Governance Rule | CI | Runtime | Test |
|----------------|-----|---------|------|
| Manifest schema compliance | `validate_benchmark_outputs.py` | — | `test_benchmark_formalization.py` |
| Result schema compliance | `validate_benchmark_outputs.py` | — | `test_export_contract.py` |
| Governance doc existence | `validate_governance_docs.py` | — | Implicit |
| Schema registry validity | `validate_schema_registry.py` | — | Implicit |
| AST safety (no dynamic code) | `ast_validator.py` | — | `test_ast_validator.py` |
| Contract sidecar consistency | `verify_contract_sidecars.py` | — | `test_export_contract.py` |
| Lifecycle promotion gates | — | `lifecycle_verifier.py` | `test_lifecycle_verifier.py` |
| Provenance roundtrip | — | `provenance.py` | `test_provenance.py` |
| Non-bypassable entrypoint | — | `governed_entrypoint` | `test_integration_enforcement.py` |
| Drift → fail | — | `export.py` | `test_runtime_invariants.py` |
| Hash authority | `validate_benchmark_outputs.py` | `provenance.py` | `test_provenance.py` |

## Known Limitations

| Gap | CI/Runtime | Mitigation |
|-----|-----------|-----------|
| Governance doc policy (migration freeze, producer obligations) | CI only (doc existence) | Manual review on PR |
| `determinism` sub-object flag values | CI only (field presence) | Torch determinism tested via integration |
| `config_hashes` sub-key completeness | CI only (object existence) | No known impact on current use |
| Runtime feature schema not in registry | CI only (sidecar check) | Enforced via `IMMUTABLE_SCHEMA_CONTRACT.md` |
| Legacy artifact policy | CI only (env var) | Enforced via `HELIX_ALLOW_LEGACY_*` flags |

## Future Provenance-Locking Roadmap

1. **Policy test coverage** — Add programmatic tests for governance doc policy enforcement (not just existence).
2. **Runtime determinism verification** — Test that same seed produces identical model weights (not just similar metrics).
3. **Full schema registry** — Add runtime feature schema to `schema_registry.yaml`.
4. **Policy-as-code** — Encode governance policies as machine-readable rules validated by `ast_validator`.

## Consequences

- **Positive:** Multi-layer enforcement prevents both intentional bypass and accidental drift.
- **Negative:** Complex CI pipeline (~5 jobs, dependency chains) — failure attribution can be non-obvious.
- **Neutral:** Runtime enforcement adds overhead; acceptable for production but may slow research iteration.