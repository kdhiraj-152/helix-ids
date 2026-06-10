# ADR-002: Schema Lifecycle and Versioning Policy

**Status:** Accepted
**Date:** 2026-06-03
**Deciders:** HELIX-IDS governance lead

---

## Context

HELIX-IDS manages two primary schemas: the **runtime feature schema** (immutable, enforced by `IMMUTABLE_SCHEMA_CONTRACT.md`) and the **manifest/result schema** (versioned, governed by `schema_registry.yaml` and `manifest_schema_governance.md` / `result_schema_governance.md`).

This ADR defines how schema changes are handled without breaking existing artifacts.

## Current Guarantees

1. **Schema registry** (`schema_registry.yaml`) tracks all schema versions with owner, compatibility window (30d), and deprecation policy.
2. **Version format** is `YYYY-MM-DD` date-stamp, enforced monotonically.
3. **Breaking change definition** is explicitly documented — field removal, renaming, type change, hash algorithm change.
4. **Dual-read window** — consumers support old and new versions during compatibility window.
5. **Producers first** — schema producers update before consumers require the new version.
6. **CI enforcement** — `validate_schema_registry.py` blocks merges that modify schema structure without registry update.

## Known Limitations

| Limitation | Impact |
|-----------|--------|
| Version monotonicity not programmatically enforced beyond string non-empty check | Manual review required |
| `config_hashes` sub-key completeness not validated | Partial field sets may pass CI |
| `governance_state` field extensions not formally bounded | Extensible but informal |
| Registry has only 2 entries (manifest_schema, benchmark_result_schema) | Runtime feature schema not in registry |

## Future Provenance-Locking Roadmap

1. **Add runtime feature schema to `schema_registry.yaml`** — currently governed by `IMMUTABLE_SCHEMA_CONTRACT.md` but not in the registry.
2. **Programmatic version monotonicity check** — parse dates, enforce chronological ordering in CI.
3. **Formal `config_hashes` sub-key schema** — JSONSchema for the nested object validated in CI.
4. **Automated migration assistant** — CLI tool to update registry and regenerate artifacts.

## Consequences

- **Positive:** Schema changes are tracked and reversible.
- **Negative:** Breaking changes require a formal migration cycle (min 30d compatibility window).
- **Neutral:** Registry ownership is a single point — governance lead must remain active.