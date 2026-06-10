# ADR-003: Hash Authority and Artifact Integrity

**Status:** Accepted
**Date:** 2026-06-03
**Deciders:** HELIX-IDS governance lead

---

## Context

HELIX-IDS tracks artifact integrity through SHA-256 hashes at every pipeline stage. This ADR documents which component is authoritative for each hash type and how conflicts are resolved.

## Current Guarantees

1. **Single hash algorithm** — SHA-256 throughout; no algorithm flexibility.
2. **Canonical JSON encoding** — `sort_keys=True`, `separators=(",",":")`, UTF-8 for all JSON hash inputs.
3. **Hash types** (defined in `hash_authority.md`):
   - `raw_hash`: authoritative source `build_dataset_manifest_hash`
   - `processed_hash`: authoritative source `feature_harmonization` writer
   - `split_hash`: authoritative source `multi_dataset_loader` split writer
   - `dataset_hash_primary`: authoritative source `build_dataset_manifest_hash`
   - `schema_hash`: runtime feature schema hash from canonical contract
   - `manifest_hash`: SHA-256 of the manifest JSON file
   - `artifact_sha256`: final checkpoint/TorchScript artifact digest
4. **Immutable hash fields** — once written, no post-hoc correction permitted.
5. **Tamper detection** — `verify_artifact_manifest()` raises `ArtifactManifestError` on mismatch.
6. **File ordering** — files sorted lexicographically before hashing to ensure deterministic manifest.

## Known Limitations

| Limitation | Impact |
|-----------|--------|
| SHA-256 only — no algorithm agility | Future quantum threat to long-term integrity verification |
| `raw_hash` computed from file metadata + content; no streaming hash | Very large files incur full read on every run |
| No hash revocation mechanism | Compromised hash requires schema version bump |
| Individual pipeline step hashes not tracked separately | Can identify which stage changed but not which operation within stage |

## Future Provenance-Locking Roadmap

1. **Step-level hashes** — track `raw_hash → harmonized_hash → split_hash` as separate artifacts with explicit dependency chain.
2. **Algorithm agility** — add SHA-3 family support with migration path from SHA-256.
3. **Streaming hash** — replace full-file reads with incremental SHA-256 (content-addressable).
4. **Hash revocation registry** — formal mechanism to mark compromised hashes as revoked.

## Consequences

- **Positive:** Strong integrity guarantees at every pipeline stage with tamper-evident artifacts.
- **Negative:** Full file hashing adds overhead to dataset pipeline (~2% for typical NSL-KDD splits).
- **Neutral:** Hash fields are immutable — errors require new run with corrected inputs.