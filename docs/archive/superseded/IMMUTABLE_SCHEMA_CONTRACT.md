# Immutable Schema Contract

## Canonical Schema Philosophy

Helix IDS treats the runtime feature schema as an immutable contract. The canonical feature order, schema hash, and sidecar files are authoritative; they are not inferred, repaired, or negotiated at runtime.

## Drift -> Fail Rules

- Any reordered column set is a contract violation.
- Any missing feature is a contract violation.
- Any extra feature is a contract violation.
- Any schema hash mismatch is a contract violation.
- Any label-space mismatch is a contract violation.
- Any attempt to continue after drift detection is prohibited.

## Prohibited Behaviors

- Silent coercion.
- Compatibility fallbacks.
- Dynamic alignment or reindexing.
- Column inference from partially matching inputs.
- Warning-based recovery paths.
- Repairing artifacts in consumer code.

## Exporter Obligations

- Emit the canonical runtime contract payload with every model artifact.
- Write the exact `.contract.json`, `.feature_order.json`, and `.schema_hash.txt` sidecars.
- Keep the payload stable and versioned.
- Fail the export if the emitted contract does not match the canonical schema.

## Runtime Obligations

- Validate every checkpoint and sidecar before use.
- Fail fast on any schema drift.
- Emit structured telemetry for every drift event.
- Never infer or repair missing runtime contract information.
- Never continue execution after a drift event is detected.

## Telemetry Guarantees

- Single JSON schema for drift events.
- Stable event name: `schema_drift_detected`.
- UTC timestamps only.
- Immutable payload fields.
- Deterministic field names and ordering.
- Payloads must include producer identity, artifact path, expected and actual schema hashes, feature-name sets, and observed versus expected cardinality.

## Migration Freeze Policy

- Canonical contract changes are freeze-gated and must be treated as release changes.
- Producers must be updated first, then consumers, then CI enforcement.
- The freeze branch/tag remains the canonical rollback point.
- Any future schema change must be approved as a deliberate migration, not as a compatibility patch.
