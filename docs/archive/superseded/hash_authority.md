# Hash Authority and Artifact Integrity

**See also:** ADR-003-hash-authority.md

## Hash Types and Sources

| Hash Field | Authoritative Source |
|---|---|
| `raw_hash` | `build_dataset_manifest_hash` |
| `processed_hash` | `feature_harmonization` writer |
| `split_hash` | `multi_dataset_loader` split writer |
| `model_hash` | `train_helix_ids_full` checkpoint writer |
| `contract_hash` | `export_contract` exporter |

## Canonical Encoding

All hash inputs use SHA-256 over canonical JSON (`sort_keys=True`, `separators=(",",":")`, UTF-8).

## Conflict Resolution

If two stages produce different hashes for the same logical artifact, the earlier pipeline stage is authoritative (raw > processed > split > model).
