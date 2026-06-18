# Result Schema Governance

**Registry entry:** `benchmark_result_schema` (versioned YYYY-MM-DD per ADR-002)

## Scope

Governs the JSON schema for benchmark result payloads produced by the evaluation pipeline.

## Rules

1. **Version monotonicity** — Same YYYY-MM-DD date-stamp rules as manifest schema governance (enforced by `validate_schema_registry.py`).
2. **Field retention** — Once a field is published in an active result schema, it must not be removed or have its type changed without a deprecation cycle.
3. **Required fields** — Every result must include `schema_version`, `benchmark_id`, `metrics`, `environment`, and `timestamp`.
4. **Deprecation** — Follows the same announce-deprecate-then-retire cycle as manifest schemas.

## Validator

- `validate_schema_registry.py` enforces registry integrity for both manifest and result schemas.
- Benchmark output validation is performed by `scripts/ci/validate_benchmark_outputs.py`.
