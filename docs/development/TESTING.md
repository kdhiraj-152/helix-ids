# Testing

> Last updated: 2026-06-18  
> Single authoritative reference for all testing in HELIX-IDS.

## Test Suite Overview

| Test Type | Count | Locations | CI Stage |
|-----------|-------|-----------|----------|
| Unit | 50+ | `tests/`, `tests/test_*`, `tests/training/` | `ci.yml` |
| Integration | 10+ | `tests/test_operations/`, `tests/training/` | `ci.yml` |
| Architecture | 6 | `tests/architecture/` | `architecture.yml` |
| Governance | 10 | `tests/test_governance/` | `architecture.yml` |
| Property-based | 1 | `tests/test_property_based.py` | `ci.yml` |
| Fuzz | 1 | `tests/test_fuzz.py` | `nightly.yml` |
| Chaos | 1 | `tests/test_checkpoint_chaos.py` | `nightly.yml` |
| Fault injection | 1 | `tests/test_fault_injection.py` | `nightly.yml` |
| Memory leak | 1 | `tests/test_memory_leak_detection.py` | `nightly.yml` |
| Load | 1 | `scripts/benchmarks/load_test.py` | Manual |
| Soak | 3 | `scripts/benchmarks/soak_*.py` | Manual |
| **Total** | **~85+ files** | | |

## Running Tests

```bash
# Full suite
pytest -q

# Specific areas
pytest tests/test_operations -q
pytest tests/test_governance -q
pytest tests/architecture -q

# With coverage
pytest --cov=src/helix_ids --cov-fail-under=65

# Lint and type checking
ruff check src scripts tests
mypy src
```

## Unit Tests

Located in `tests/` (~50+ files). Cover all core modules under `src/helix_ids/`:
- Data loading and harmonization
- Model construction and forward pass
- Loss computation
- Governance entrypoints and lifecycle verification
- Operations and monitoring
- Utils and export

Run: `pytest tests/ -q`

## Integration Tests

Located in `tests/test_operations/` and `tests/training/`. Cover:
- End-to-end training smoke test (`test_e2e_smoke.py`)
- REST server metrics and endpoint behavior
- Governance integration (entrypoint wrapping + contract validation)
- Staging gate check script

Run: `pytest tests/test_operations/ tests/training/ -q`

## Architecture Tests

6 test files in `tests/architecture/` enforcing:
- DAG constraints (no cycles)
- Package boundaries (no forbidden imports)
- Method count limits (≤100 per class, ≤2000 LOC)
- Reverse dependency constraints
- Architecture freeze enforcement

Run: `pytest tests/architecture/ -q`

## Property Tests

`tests/test_property_based.py` uses Hypothesis to verify:
- Feature harmonization is idempotent
- Model output shape invariants
- Schema contract constraints

Run: `pytest tests/test_property_based.py -q`

## Mutation Tests

**Tool**: Cosmic-Ray 8.4.6 (replaced mutmut — Python 3.11 crash)

**Configuration**: 16 cosmic-ray configs at project root.

**Current score**: 100% killed across all 15 modules (8,479 mutants, 0 survivors).

| Module | Mutants | Score |
|--------|---------|-------|
| `utils/metrics.py` | 56 | 100% |
| `models/loss.py` | 87 | 100% |
| `models/adaptation/coral_loss.py` | 215 | 100% |
| `governance/lifecycle_verifier.py` | 905 | 100% |
| `governance/provenance.py` | 194 | 100% |
| `utils/export.py` | 811 | 100% |
| `governance/ast_validator.py` | 321 | 100% |
| `contracts/diagnostic_contract.py` | 53 | 100% |
| `contracts/schema_contract.py` | 99 | 100% |
| `operations/baseline_freeze.py` | 358 | 100% |
| `data/preprocessing.py` | 401 | 100% |
| `operations/determinism.py` | 117 | 100% |
| `models/inference_runtime.py` | 1,565 | 100% |
| `feature_harmonization.py` | 1,875 | 100% |
| `transfer_learning.py` | 1,565 | 100% |

**CI**: Mutation testing runs weekly (Monday 06:00 UTC) via `test-reliability.yml`.

## Chaos Tests

`tests/test_checkpoint_chaos.py` — Verifies system resilience under:
- Corrupted checkpoint files
- Partial save failures
- Concurrent checkpoint operations

## Fault Injection

`tests/test_fault_injection.py` — Tests recovery under:
- Data loader failures
- Network timeout during inference
- NaN propagation
- OOM simulation

## Coverage Expectations

| Gate | Threshold | Enforcement |
|------|-----------|-------------|
| Line coverage | ≥65% | Blocking CI gate (`--cov-fail-under=65`) |
| Current coverage | ≥70.09% | `src/helix_ids/` only (excludes tests/scripts) |
| Report | Uploaded as `coverage.xml` artifact | CI pipeline |

## CI Gates

| Gate | Workflow | Command |
|------|----------|---------|
| Lint | `quality.yml` | `ruff check src scripts tests` |
| Types | `quality.yml` | `mypy src` |
| Tests | `ci.yml` | `pytest -q` |
| Coverage | `ci.yml` | `pytest --cov-fail-under=65` |
| Architecture | `architecture.yml` | `pytest tests/architecture/ -q` |
| Security | `quality.yml` | `bandit -r src/` |
| Dependencies | `dependency-review.yml` | Block on HIGH severity |
| Performance | `nightly.yml` | Regression threshold check |
| Mutation | `test-reliability.yml` | Weekly, 100% target |
