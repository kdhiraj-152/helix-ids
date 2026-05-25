# CHECKPOINT AUDIT

## Active canonical paths
- `scripts/training/train_helix_ids_full.py` writes checkpoint payloads with canonical feature order and contract metadata.
- `src/helix_ids/operations/inference_runtime.py` rejects checkpoints missing canonical metadata or carrying legacy dimensions.
- `src/helix_ids/operations/baseline_freeze.py` validates checkpoint metadata before freezing a release bundle.

## Legacy compatibility paths
- None retained in runtime or release packaging. Legacy checkpoints are rejected instead of adapted.

## Dead paths
- Checkpoints that omit `canonical_input_dim`, `canonical_binary_classes`, or `canonical_family_classes`.
- Checkpoints with reordered feature order.
- Checkpoints with 19-feature or 41-feature input dimensions.

## Contradictory paths
- Older checkpoints saved before the canonical contract fields were added are now intentionally invalid.
- Any checkpoint whose model tensors imply a different input/output width than the canonical contract is rejected immediately.
