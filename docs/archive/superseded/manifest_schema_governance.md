# Manifest Schema Governance

**Registry entry:** `manifest_schema` (versioned YYYY-MM-DD per ADR-002)

## Scope

Governs the JSON schema for pipeline artifact manifests including dataset manifests, model manifests, and experiment manifests.

## Rules

1. **Version monotonicity** — Every manifest schema version must be a YYYY-MM-DD date-stamp; versions must not decrease (enforced by `validate_schema_registry.py`).
2. **Backward compatibility** — A new schema version must accept all valid payloads from the previous version within the 30-day compatibility window.
3. **Deprecation** — Follows the announce-deprecate-then-retire cycle defined in the registry. Schema status transitions: `active` -> `deprecated` -> `retired`.
4. **Approval** — Any schema change requires governance lead approval (`approval_required: true` in the registry).

## Validator

- `validate_schema_registry.py` enforces version format, chronology, and required fields.
- `validate_governance_consistency.py` verifies that documented claims match validator capabilities.
