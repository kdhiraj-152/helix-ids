# EXPORT CONTRACT REPORT

## Active canonical paths
- `src/helix_ids/operations/inference_runtime.py` exports TorchScript and ONNX artifacts with sidecar contract metadata.
- `scripts/operations/export_inference_bundle.py` emits a service contract that mirrors runtime metadata.
- `src/helix_ids/utils/export.py` defaults to the canonical 17-feature / 2-class / 7-class contract.

## Legacy compatibility paths
- None retained for canonical export bundles. No silent fallback metadata is emitted.

## Dead paths
- Export metadata that omits feature order, canonical input dimension, or class counts.
- ONNX/TorchScript bundles that rely on implicit class counts.
- Service contracts that describe non-canonical feature dimensions.

## Contradictory paths
- Any exported artifact whose metadata disagrees with the runtime constants is treated as invalid.
- Generic ONNX export helpers remain available, but they now default to the canonical HELIX contract and should not be used to describe alternate schemas.
