# Result Schema Governance

## Scope

This document defines governance policy for the HELIX-IDS benchmark result
schema — the JSON structure emitted to `results/metrics/*.result.json` and
`results/metrics/*.metrics.json` by the benchmark execution pipeline. It
establishes authority, compatibility guarantees, lineage requirements, and
lifecycle management for result payloads.

---

## 1. Result Schema Authority

| Role              | Party                         | Responsibility                                   |
|-------------------|-------------------------------|---------------------------------------------------|
| Schema Owner      | HELIX-IDS governance lead     | Approve result schema modifications; maintain registry |
| Schema Steward    | CI enforcement pipeline       | Validate result payload compliance; block violations    |
| Schema Producer   | Benchmark evaluation scripts  | Emit result payloads conforming to current version      |
| Schema Consumer   | Reporting, staging, deployment | Read results; never infer or repair missing data       |

The Schema Owner is recorded in `schema_registry.yaml` under the
`benchmark_result_schema` entry's `owner` field.

---

## 2. Result Payload Compatibility Guarantees

1. **Backward read**: Newer consumers must be able to read result payloads
   produced under any non-deprecated schema version within the compatibility
   window.
2. **Forward silent**: Older consumers encountering unknown optional fields
   must silently ignore them rather than fail.
3. **No silent coercion**: If a required field is missing or has wrong type,
   the consumer must fail — never guess, default, or repair.
4. **Hash immutability**: All hash fields in a result payload are immutable
   once written. No post-hoc correction is permitted.

---

## 3. Required Lineage Fields

Every result JSON file **must** contain a `lineage` object with the following
fields:

| Field                | Type   | Description                                        |
|----------------------|--------|----------------------------------------------------|
| `run_id`             | string | Globally unique run identifier                     |
| `benchmark_id`       | string | Benchmark spec identifier                          |
| `variant_id`         | string | Variant within the benchmark                       |
| `manifest_hash`      | string | SHA-256 of the source manifest JSON                |
| `manifest_path`      | string | Filesystem path to the source manifest             |
| `schema_hash`        | string | Runtime feature schema hash                        |
| `mapping_version`    | string | Label-mapping version tag                          |
| `mapping_hash`       | string | SHA-256 of label-mapping definition                |
| `dataset_id`         | string | Canonical dataset identifier                       |
| `dataset_hash_primary`| string | Authoritative dataset identity hash               |
| `dataset_hashes`     | object | Per-stage dataset hashes (raw, processed, split, primary) |
| `config_hashes`      | object | Config-derived SHA-256 hashes                      |
| `model_architecture` | string | Model architecture identifier                      |
| `git_commit`         | string | Git commit SHA at execution time                   |
| `fingerprint`        | string | Deterministic run fingerprint                      |

A result payload missing any required lineage field is a **contract violation**
and must cause CI to fail.

---

## 4. Required Reproducibility Fields

Every result JSON file **must** contain a `reproducibility_metadata` object
with the following fields:

| Field    | Type   | Description                                        |
|----------|--------|----------------------------------------------------|
| `seed`   | int    | Primary random seed                                |
| `determinism` | object | Determinism configuration (torch flags, hash seed) |
| `run_id` | string | Run identifier (must match lineage.run_id)         |

The `determinism` object must include:
- `seed` (int)
- `torch_deterministic_algorithms` (bool)
- `torch_cudnn_deterministic` (bool)
- `torch_cudnn_benchmark` (bool)
- `python_hash_seed` (string)

---

## 5. Backward Compatibility Requirements

1. **Field additions**: New optional fields may be added at any nesting level
   without a schema version bump.
2. **Field removals**: Removing any required field is a breaking change
   requiring a migration plan (see Section 7).
3. **Type changes**: Changing the type of any existing field is a breaking
   change.
4. **Hash stability**: The hash computation algorithm (SHA-256, canonical JSON
   encoding with sorted keys and compact separators) must not change without a
   full migration.
5. **Nested structure**: Adding new required sub-keys inside existing objects
   (e.g., `config_hashes`) is a breaking change unless the parent object is
   optional.

---

## 6. Version Lifecycle Policy

| Phase        | Duration              | Behavior                                              |
|--------------|-----------------------|-------------------------------------------------------|
| Active       | Until superseded      | Full CI enforcement; all required fields validated    |
| Deprecated   | 30 days minimum       | CI warns but does not fail; new production uses newer |
| Retired      | After deprecation     | CI fails on retired schema versions                  |

### Version Transition Rules

1. A new version becomes `Active` when the first conforming result payload is
   produced and CI validates it.
2. The prior version enters `Deprecated` on the same commit.
3. After the compatibility window, the deprecated version moves to `Retired`
   via a schema-registry update.
4. No result payload may be produced under a `Retired` schema version.

### Schema Version Format

Result schema versions follow the same `YYYY-MM-DD` date-stamp convention as
manifest schema versions. The `schema_version` field in a result payload
identifies the result schema shape (distinct from the manifest schema version).
