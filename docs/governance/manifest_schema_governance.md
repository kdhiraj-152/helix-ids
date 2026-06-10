# Manifest Schema Governance

## Scope

This document defines governance policy for the HELIX-IDS benchmark manifest
schema — the JSON structure emitted to `results/manifests/*.manifest.json` by
the benchmark execution pipeline. It complements the existing
[IMMUTABLE_SCHEMA_CONTRACT.md](IMMUTABLE_SCHEMA_CONTRACT.md) which governs the
runtime feature schema; this document governs the manifest payload structure
itself.

---

## 1. Schema Ownership

| Role            | Party                         | Responsibility                                    |
|-----------------|-------------------------------|----------------------------------------------------|
| Schema Owner    | HELIX-IDS governance lead     | Approve schema modifications; maintain registry    |
| Schema Steward  | CI enforcement pipeline       | Validate compliance; block non-conforming changes  |
| Schema Producer | Benchmark execution scripts   | Emit manifests conforming to the current version   |
| Schema Consumer | Validation, evaluation, CI    | Read manifests; never infer or repair missing data |

The Schema Owner is recorded in `schema_registry.yaml` under the
`manifest_schema` entry's `owner` field.

---

## 2. Required Fields

Every manifest JSON file **must** contain the following top-level fields
(as defined by `scripts/evaluation/benchmarks.py` `REQUIRED_MANIFEST_FIELDS`):

|| Field                  | Type   | Description                                      |
||------------------------|--------|--------------------------------------------------|
|| `dataset_id`           | string | Canonical dataset identifier                     |
|| `dataset_roots`        | array  | Dataset root directory paths                     |
|| `raw_hash`             | string | SHA-256 of raw dataset files                     |
|| `processed_hash`       | string | SHA-256 of processed dataset artifacts           |
|| `split_hash`           | string | SHA-256 of data-split artifacts                  |
|| `dataset_hash_primary` | string | Authoritative dataset identity hash              |
|| `model_architecture`  | string | Model architecture identifier                    |
|| `model_architecture_source` | string | Origin of architecture declaration         |
|| `governance_state`     | object | Governance enforcement state snapshot            |
|| `evaluation_mode`      | string | Evaluation mode tag                              |
|| `platform_targets`     | array  | Target deployment platforms                      |
|| `config_hashes`       | object | Nested config-derived hashes                     |
|| `schema_hash`          | string | Runtime feature schema hash                       |
|| `mapping_version`      | string | Label-mapping version tag                         |
|| `seed`                 | int    | Random seed for reproducibility                  |
|| `experiment_id`        | string | Unique experiment identifier                     |

A manifest missing any required field is a **contract violation** and must
cause CI to fail.

---

## 3. Optional Fields

The following fields are permitted but not required:

|| Field                      | Type   | Description                               |
||----------------------------|--------|-------------------------------------------|
| `schema_version`           | string | Manifest schema version (YYYY-MM-DD)       |
| `benchmark_id`             | string | Benchmark spec identifier                  |
| `variant_id`               | string | Variant identifier within benchmark        |
| `mapping_hash`             | string | SHA-256 of label-mapping definition       |
| `outputs`                  | object | Evaluation outputs container              |
| `dataset_roots`            | array  | Dataset root directory paths              |
| `evaluation_mode`          | string | Evaluation mode tag                        |
| `platform_targets`         | array  | Target deployment platforms               |
| `entrypoint`               | string | Benchmark entrypoint function name        |
| `manifest_source`          | string | Source YAML config path                    |
| `config`                   | object | Embedded config snapshot                   |
| `dataset`                  | object | Nested dataset metadata                    |

Producers **may** include optional fields. Consumers **must not** require
optional fields to be present.

---

## 4. Schema Versioning Rules

1. **Format**: `YYYY-MM-DD` date-stamp (e.g. `2026-06-02`).
2. **Monotonic**: New versions must be chronologically later than all prior
   versions.
3. **Single-version per artifact**: Each manifest file carries exactly one
   `schema_version`.
4. **Version is not a hash**: The version identifies the schema shape; the
   `schema_hash` field validates content integrity.

---

## 5. Breaking-Change Criteria

A change is **breaking** if it:

- Removes a required field.
- Renames a required or optional field.
- Changes the type of any field.
- Reorders nested object keys that participate in hash computation.
- Alters the hash computation algorithm (e.g. SHA-256 to SHA-512).
- Modifies the `config_hashes` required sub-keys.
- Changes the label vocabulary or mapping version semantics.

A change is **non-breaking** if it:

- Adds a new optional field.
- Adds a new sub-key under `config_hashes` (additive only).
- Extends `governance_state` with new keys.
- Updates documentation or comments.

---

## 6. Deprecation Policy

1. **Announcement**: Deprecated fields are marked in this document and in the
   schema registry with `status: deprecated`.
2. **Compatibility window**: Deprecated fields must remain accepted by
   consumers for at least one full release cycle (minimum 30 days).
3. **Removal**: After the compatibility window, a breaking-change proposal is
   required to remove the field entirely.
4. **CI signaling**: During the deprecation window, CI emits warnings but does
   not fail on deprecated fields.

---

## 7. Migration Requirements

When a breaking change is approved:

1. **Producers first**: Manifest producers must emit the new schema version
   before any consumer requires it.
2. **Dual-read window**: Consumers must support both the old and new schema
   versions during the compatibility window.
3. **Freeze tag**: The commit introducing the new schema version must be tagged
   as the canonical rollback point.
4. **Registry update**: The `schema_registry.yaml` entry must be updated
   atomically with the code change.
5. **CI gate**: CI validation must pass under both schema versions until the
   old version is formally retired.

---

## 8. Approval Requirements for Schema Modifications

| Change type       | Required approvals                        | Process                          |
|-------------------|-------------------------------------------|----------------------------------|
| Non-breaking      | Schema Owner (1)                          | PR + CI green                    |
| Breaking          | Schema Owner + 1 reviewer                 | PR + migration plan + CI green   |
| Hash algorithm    | Schema Owner + 2 reviewers               | Full migration plan required     |
| Deprecation       | Schema Owner (1)                          | PR + registry update             |

All schema modifications must be reflected in `schema_registry.yaml` before
merge. CI enforcement blocks merges that modify manifest structure without a
corresponding registry update.
