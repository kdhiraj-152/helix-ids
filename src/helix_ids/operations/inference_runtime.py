from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import torch

from helix_ids.contracts import CONTRACT_VERSION, FEATURE_ORDER_HASH
from helix_ids.contracts.schema_contract import (
    CANONICAL_BINARY_CLASSES,
    CANONICAL_FEATURE_ORDER,
    CANONICAL_FAMILY_CLASSES,
    CANONICAL_INPUT_DIM,
    SCHEMA_HASH,
    SCHEMA_VERSION,
    assert_runtime_contract,
    runtime_contract_payload,
    validate_feature_order,
)
from helix_ids.governance import (
    embed_manifest_in_onnx_metadata,
    verify_artifact_provenance,
    verify_ingress_artifact,
    write_contract_sidecars,
)
from helix_ids.utils.export import build_export_manifest, finalize_export_artifact, verify_export_artifact
from helix_ids.models.full import HelixFullConfig, create_helix_full


@dataclass(frozen=True)
class InferenceConfig:
    fixed_temperature: float = 1.0
    prediction_floor: float = 1e-6
    global_coverage_floor: bool = False
    global_coverage_quantile: float = 0.95
    class_margin_override_enabled: bool = True
    class_margin_override_class_id: int = 4
    class_margin_override_tau: float = 70_000.0
    class_margin_override_use_percentile: bool = True
    class_margin_override_percentile_k: float = 75.0
    class_margin_override_hybrid_and: bool = True
    class_margin_override_buffer_size: int = 1_000
    class_margin_override_warmup_min_samples: int = 200
    class_margin_override_rate_guard_threshold: float = 0.5
    class_margin_override_rate_guard_window: int = 1_000
    class_margin_override_baseline_precision: float | None = None
    class_margin_override_precision_disable_ratio: float = 0.8
    class_margin_override_freeze_adaptive_tau: bool = False
    class_margin_override_frozen_tau_adaptive: float | None = None
    class_margin_override_use_margin_zscore: bool = False
    class_margin_override_margin_zscore_epsilon: float = 1e-9
    class_margin_buffer_variance_epsilon: float = 1e-9


class HelixInferenceRuntime:
    """Inference-only HELIX runtime stripped from training components."""

    @staticmethod
    def _contract_metadata() -> dict[str, Any]:
        payload = runtime_contract_payload()
        return {str(key): value for key, value in payload.items()}

    @staticmethod
    def _extract_backbone_linears(state_dict: dict[str, Any]) -> list[tuple[int, torch.Tensor]]:
        backbone_linears: list[tuple[int, torch.Tensor]] = []
        for key, tensor in state_dict.items():
            if not isinstance(tensor, torch.Tensor):
                continue
            if not key.startswith("backbone.") or not key.endswith(".weight"):
                continue
            if tensor.ndim != 2:
                continue
            parts = key.split(".")
            if len(parts) < 3:
                continue
            try:
                layer_idx = int(parts[1])
            except ValueError:
                continue
            backbone_linears.append((layer_idx, tensor))
        return backbone_linears

    @staticmethod
    def _infer_model_config(state_dict: dict[str, Any]) -> HelixFullConfig:
        backbone_linears = HelixInferenceRuntime._extract_backbone_linears(state_dict)
        if not backbone_linears:
            raise ValueError("Unable to infer backbone topology from checkpoint")

        backbone_linears.sort(key=lambda item: item[0])
        input_dim = int(backbone_linears[0][1].shape[1])
        hidden_dims = tuple(int(t.shape[0]) for _, t in backbone_linears)

        family_out_dim = CANONICAL_FAMILY_CLASSES
        binary_out_dim = CANONICAL_BINARY_CLASSES
        family_bias = state_dict.get("family_head.3.bias")
        if isinstance(family_bias, torch.Tensor):
            family_out_dim = int(family_bias.shape[0])
        binary_bias = state_dict.get("binary_head.3.bias")
        if isinstance(binary_bias, torch.Tensor):
            binary_out_dim = int(binary_bias.shape[0])

        if input_dim != CANONICAL_INPUT_DIM:
            raise ValueError(
                f"Checkpoint input_dim mismatch: expected {CANONICAL_INPUT_DIM}, got {input_dim}"
            )
        if binary_out_dim != CANONICAL_BINARY_CLASSES:
            raise ValueError(
                f"Checkpoint binary output mismatch: expected {CANONICAL_BINARY_CLASSES}, got {binary_out_dim}"
            )
        if family_out_dim != CANONICAL_FAMILY_CLASSES:
            raise ValueError(
                f"Checkpoint family output mismatch: expected {CANONICAL_FAMILY_CLASSES}, got {family_out_dim}"
            )

        dropout_rates = (
            (0.3, 0.3, 0.25, 0.2)
            if len(hidden_dims) == 4
            else tuple([0.2] * len(hidden_dims))
        )
        return HelixFullConfig(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            binary_output_dim=binary_out_dim,
            family_output_dim=family_out_dim,
            dropout_rates=dropout_rates,
        )

    @staticmethod
    def _validate_checkpoint_contract(payload: dict[str, Any], model_cfg: HelixFullConfig) -> None:
        required_keys = [
            "schema_version",
            "feature_order",
            "schema_hash",
            "contract_version",
            "feature_order_hash",
            "input_dim",
            "binary_output_dim",
            "family_output_dim",
        ]
        missing_keys = [key for key in required_keys if key not in payload]
        if missing_keys:
            raise AssertionError(f"Missing required checkpoint contract metadata: {missing_keys}")

        input_dim = int(model_cfg.input_dim)
        if input_dim != CANONICAL_INPUT_DIM:
            raise AssertionError(
                f"HelixInferenceRuntime supports only canonical HelixIDS-Full input_dim {CANONICAL_INPUT_DIM}; got {input_dim}"
            )

        assert_runtime_contract(
            schema_version=str(payload["schema_version"]),
            schema_hash=str(payload["schema_hash"]),
            feature_order=[str(feature) for feature in payload["feature_order"]],
            input_dim=int(payload["input_dim"]),
            binary_output_dim=int(payload["binary_output_dim"]),
            family_output_dim=int(payload["family_output_dim"]),
            context="checkpoint contract",
        )

        if str(payload["contract_version"]) != CONTRACT_VERSION:
            raise AssertionError(
                f"Checkpoint contract_version mismatch: expected {CONTRACT_VERSION}, got {payload['contract_version']!r}"
            )
        if str(payload["feature_order_hash"]) != FEATURE_ORDER_HASH:
            raise AssertionError(
                "Checkpoint feature_order_hash does not match the immutable runtime contract"
            )

        if int(payload["input_dim"]) != input_dim:
            raise AssertionError(
                f"Checkpoint input_dim mismatch: expected {input_dim}, got {payload['input_dim']!r}"
            )

    @staticmethod
    def _require_ingress_fields(contract: Mapping[str, Any], *, context: str) -> None:
        required = ["schema_hash", "feature_order", "contract_version", "feature_order_hash"]
        missing = [field for field in required if field not in contract]
        if missing:
            raise RuntimeError(
                f"Missing required {context} fields for ingress verification: {', '.join(missing)}"
            )

    def verify_ingress_artifact(
        self,
        artifact_path: Path,
        *,
        kind: str,
        contract: Mapping[str, Any],
        embedded_manifest: Mapping[str, Any] | None,
    ) -> None:
        self._require_ingress_fields(contract, context="contract")
        sidecars = {
            "contract": artifact_path.with_suffix(artifact_path.suffix + ".contract.json"),
            "feature_order": artifact_path.with_suffix(artifact_path.suffix + ".feature_order.json"),
            "schema_hash": artifact_path.with_suffix(artifact_path.suffix + ".schema_hash.txt"),
        }
        deployment_manifest = None
        candidate_deploy = artifact_path.parent / "deployment.manifest.json"
        if candidate_deploy.exists():
            deployment_manifest = candidate_deploy
        sidecar = verify_ingress_artifact(
            artifact_path,
            kind=kind,
            contract=contract,
            embedded_manifest=embedded_manifest,
            allow_legacy_local_dev=True,
            sidecars=sidecars,
            deployment_manifest=deployment_manifest,
        )
        manifest = sidecar or embedded_manifest
        if not manifest:
            # Legacy artifacts may bypass manifest checks only when explicitly allowed
            # via HELIX_ALLOW_LEGACY_ARTIFACTS.
            return
        for field in ("schema_hash", "feature_order_hash", "contract_version"):
            if str(manifest[field]) != str(contract[field]):
                raise RuntimeError(
                    f"Manifest {field} mismatch: manifest={manifest[field]} contract={contract[field]}"
                )
        if str(manifest.get("contract_version")) != CONTRACT_VERSION:
            raise RuntimeError(
                f"Manifest contract_version mismatch: expected {CONTRACT_VERSION}, got {manifest.get('contract_version')}"
            )
        if str(manifest.get("feature_order_hash")) != FEATURE_ORDER_HASH:
            raise RuntimeError(
                "Manifest feature_order_hash does not match immutable feature order hash"
            )
        for field in ("exporter_version", "git_commit"):
            value = str(manifest.get(field, ""))
            if not value or value == "unknown":
                raise RuntimeError(f"Manifest missing required ingress field: {field}")

    @staticmethod
    def _to_2d_float32(features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2:
            raise ValueError(f"Expected 2D feature matrix, got shape={x.shape}")
        return x

    @staticmethod
    def _assert_output_parity(
        reference_outputs: tuple[torch.Tensor, ...],
        candidate_outputs: tuple[torch.Tensor, ...],
        *,
        context: str,
        max_abs_diff: float = 1e-5,
    ) -> None:
        if len(reference_outputs) != len(candidate_outputs):
            raise RuntimeError(f"{context}: output count mismatch")

        for index, (reference, candidate) in enumerate(zip(reference_outputs, candidate_outputs)):
            reference_tensor = reference.detach().cpu()
            candidate_tensor = candidate.detach().cpu() if isinstance(candidate, torch.Tensor) else torch.from_numpy(np.asarray(candidate))
            if reference_tensor.shape != candidate_tensor.shape:
                raise RuntimeError(
                    f"{context}: output[{index}] shape mismatch {tuple(reference_tensor.shape)} != {tuple(candidate_tensor.shape)}"
                )
            if reference_tensor.dtype != candidate_tensor.dtype:
                raise RuntimeError(
                    f"{context}: output[{index}] dtype mismatch {reference_tensor.dtype} != {candidate_tensor.dtype}"
                )
            if not torch.isfinite(reference_tensor).all() or not torch.isfinite(candidate_tensor).all():
                raise RuntimeError(f"{context}: output[{index}] contains non-finite values")
            diff = torch.max(torch.abs(reference_tensor - candidate_tensor)).item()
            if diff > float(max_abs_diff):
                raise RuntimeError(
                    f"{context}: output[{index}] max abs diff {diff:.6g} exceeds threshold {max_abs_diff:.6g}"
                )

    def _write_export_sidecars(self, output_path: Path, manifest: Mapping[str, Any]) -> None:
        contract = self._contract_metadata()
        sidecars = write_contract_sidecars(output_path, contract)
        finalize_export_artifact(output_path, manifest, sidecars=sidecars)

    def _validate_torchscript_parity(self, traced: torch.jit.ScriptModule, example: torch.Tensor) -> None:
        with torch.no_grad():
            reference_outputs = self.model(example.to(self.device))
            candidate_outputs = traced(example.cpu())
        if not isinstance(reference_outputs, tuple) or not isinstance(candidate_outputs, tuple):
            raise RuntimeError("TorchScript parity validation expects tuple outputs")
        self._assert_output_parity(
            tuple(reference_outputs),
            tuple(candidate_outputs),
            context="TorchScript parity",
        )

    def _validate_onnx_parity(self, output_path: Path, dummy: torch.Tensor) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError("ONNX Runtime is required for export parity validation") from exc

        session = ort.InferenceSession(str(output_path))
        input_name = session.get_inputs()[0].name
        with torch.no_grad():
            reference = self.model(dummy)
        if not isinstance(reference, tuple):
            raise RuntimeError("ONNX parity validation expects tuple outputs")
        candidate_outputs = tuple(torch.from_numpy(output) for output in session.run(None, {input_name: dummy.detach().cpu().numpy()}))
        self._assert_output_parity(
            tuple(reference),
            candidate_outputs,
            context="ONNX parity",
        )

    def _validate_onnx_metadata(self, output_path: Path) -> None:
        import onnx

        model = onnx.load(str(output_path))
        metadata = {prop.key: prop.value for prop in model.metadata_props}
        contract = self._contract_metadata()
        for key, value in contract.items():
            expected = json.dumps(value) if isinstance(value, (list, dict)) else str(value)
            if metadata.get(key) != expected:
                raise RuntimeError(f"ONNX metadata mismatch for {key}: expected {expected!r}, got {metadata.get(key)!r}")

    @staticmethod
    def _resolve_target_classes(num_classes: int, active_classes: list[int] | None) -> list[int]:
        if active_classes is None:
            return list(range(num_classes))
        return [int(c) for c in active_classes]

    @staticmethod
    def _choose_override_class(
        sample_logits: torch.Tensor,
        *,
        predicted_class: int,
        target_classes: list[int],
        num_classes: int,
        sample_threshold: torch.Tensor,
    ) -> tuple[int, float] | None:
        for cls in target_classes:
            if cls < 0 or cls >= num_classes or cls == predicted_class:
                continue
            cls_logit = sample_logits[cls]
            if bool(cls_logit > sample_threshold):
                return int(cls), float(cls_logit.detach().cpu().item())
        return None

    def _apply_global_coverage_override(
        self,
        pred: torch.Tensor,
        family_logits: torch.Tensor,
        *,
        target_classes: list[int],
    ) -> tuple[np.ndarray, list[int | None], list[float | None], list[float | None]]:
        batch_size = int(pred.shape[0])
        num_classes = int(family_logits.shape[1])
        q = float(np.clip(self.config.global_coverage_quantile, 0.0, 1.0))

        override_applied_arr = np.zeros((batch_size,), dtype=bool)
        override_class_arr: list[int | None] = [None] * batch_size
        override_logit_arr: list[float | None] = [None] * batch_size
        override_threshold_arr: list[float | None] = [None] * batch_size

        for i in range(batch_size):
            predicted_class = int(pred[i].item())
            sample_logits = family_logits[i, :]
            sample_threshold = torch.quantile(sample_logits, q=q, dim=0)
            override = self._choose_override_class(
                sample_logits,
                predicted_class=predicted_class,
                target_classes=target_classes,
                num_classes=num_classes,
                sample_threshold=sample_threshold,
            )
            if override is None:
                continue

            override_class, override_logit = override
            pred[i] = override_class
            override_applied_arr[i] = True
            override_class_arr[i] = override_class
            override_logit_arr[i] = override_logit
            override_threshold_arr[i] = float(sample_threshold.detach().cpu().item())

        return (
            override_applied_arr,
            override_class_arr,
            override_logit_arr,
            override_threshold_arr,
        )

    @staticmethod
    def _format_predict_response(
        pred: torch.Tensor,
        conf: torch.Tensor,
        probs: torch.Tensor,
        override_applied_arr: np.ndarray,
        override_class_arr: list[int | None],
        override_logit_arr: list[float | None],
        override_threshold_arr: list[float | None],
        margin_override_applied_arr: np.ndarray,
        margin_override_margin_arr: np.ndarray,
        margin_override_second_class_arr: np.ndarray,
    ) -> dict[str, Any]:
        pred_np = pred.cpu().numpy().astype(int)
        conf_np = conf.cpu().numpy().astype(float)
        probs_np = probs.cpu().numpy().astype(float)

        batch_mode = len(pred_np) > 1
        override_any = bool(np.any(override_applied_arr))
        first_idx = int(np.argmax(override_applied_arr)) if override_any else -1
        margin_override_any = bool(np.any(margin_override_applied_arr))
        margin_first_idx = int(np.argmax(margin_override_applied_arr)) if margin_override_any else -1
        margin_value = (
            float(margin_override_margin_arr[margin_first_idx])
            if margin_override_any and np.isfinite(margin_override_margin_arr[margin_first_idx])
            else None
        )
        margin_second_class = (
            int(margin_override_second_class_arr[margin_first_idx])
            if margin_override_any and int(margin_override_second_class_arr[margin_first_idx]) >= 0
            else None
        )
        return {
            "family_class": pred_np.tolist() if batch_mode else int(pred_np[0]),
            "confidence": conf_np.tolist() if batch_mode else float(conf_np[0]),
            "coverage_override_applied": override_any,
            "coverage_override_class": (override_class_arr[first_idx] if override_any else None),
            "coverage_override_logit": (override_logit_arr[first_idx] if override_any else None),
            "coverage_override_threshold": (override_threshold_arr[first_idx] if override_any else None),
            "class_margin_override_applied": margin_override_any,
            "class_margin_override_second_class": margin_second_class,
            "class_margin_override_margin": margin_value,
            "probabilities": probs_np.tolist(),
        }

    def __init__(
        self,
        checkpoint_path: Path,
        *,
        device: str = "cpu",
        config: InferenceConfig = InferenceConfig(),
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device)
        self.config = config

        payload = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = payload.get("model_state_dict") or payload.get("model")
        if not isinstance(state_dict, dict):
            raise ValueError("Checkpoint missing model state dict")

        model_cfg = self._infer_model_config(state_dict)
        self._validate_checkpoint_contract(payload, model_cfg)
        self.verify_ingress_artifact(
            self.checkpoint_path,
            kind="checkpoint",
            contract=payload,
            embedded_manifest=payload.get("artifact_manifest"),
        )

        self.model = create_helix_full(model_cfg)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

        self.feature_order = list(payload["feature_order"])
        self.schema_hash = str(payload["schema_hash"])
        self.schema_version = str(payload["schema_version"])

        checkpoint_contract_path = self.checkpoint_path.with_suffix(self.checkpoint_path.suffix + ".contract.json")
        checkpoint_feature_order_path = self.checkpoint_path.with_suffix(self.checkpoint_path.suffix + ".feature_order.json")
        checkpoint_schema_hash_path = self.checkpoint_path.with_suffix(self.checkpoint_path.suffix + ".schema_hash.txt")
        for sidecar_path in [checkpoint_contract_path, checkpoint_feature_order_path, checkpoint_schema_hash_path]:
            if not sidecar_path.exists():
                raise RuntimeError(f"Missing checkpoint sidecar: {sidecar_path}")

        sidecar_contract = json.loads(checkpoint_contract_path.read_text(encoding="utf-8"))
        validate_feature_order(json.loads(checkpoint_feature_order_path.read_text(encoding="utf-8")), context="checkpoint sidecar")
        sidecar_schema_hash = checkpoint_schema_hash_path.read_text(encoding="utf-8").strip()
        assert_runtime_contract(
            schema_version=str(sidecar_contract["schema_version"]),
            schema_hash=str(sidecar_contract["schema_hash"]),
            feature_order=[str(feature) for feature in sidecar_contract["feature_order"]],
            input_dim=int(sidecar_contract["input_dim"]),
            binary_output_dim=int(sidecar_contract["binary_output_dim"]),
            family_output_dim=int(sidecar_contract["family_output_dim"]),
            context="checkpoint sidecar",
        )
        if sidecar_contract != runtime_contract_payload():
            raise RuntimeError("Checkpoint sidecar contract does not match the immutable runtime contract")
        if sidecar_schema_hash != self.schema_hash:
            raise RuntimeError("Checkpoint schema_hash sidecar does not match checkpoint metadata")
        if list(self.feature_order) != list(CANONICAL_FEATURE_ORDER):
            raise RuntimeError("Checkpoint feature order does not match the immutable runtime contract")

        # Rolling margins buffer and online guards for class-margin override.
        self._class_margin_buffer: deque[float] = deque(
            maxlen=max(1, int(self.config.class_margin_override_buffer_size))
        )
        self._class_margin_guard_overrides: deque[int] = deque(
            maxlen=max(1, int(self.config.class_margin_override_rate_guard_window))
        )
        self._class_margin_guard_total: deque[int] = deque(
            maxlen=max(1, int(self.config.class_margin_override_rate_guard_window))
        )
        self._class_margin_guard_tp: deque[int] = deque(
            maxlen=max(1, int(self.config.class_margin_override_rate_guard_window))
        )
        self._class_margin_guard_pred4: deque[int] = deque(
            maxlen=max(1, int(self.config.class_margin_override_rate_guard_window))
        )
        self._class_margin_guard_pred4_labeled: deque[int] = deque(
            maxlen=max(1, int(self.config.class_margin_override_rate_guard_window))
        )
        self._class_margin_guard_labeled: deque[int] = deque(
            maxlen=max(1, int(self.config.class_margin_override_rate_guard_window))
        )
        self._class_margin_enabled_runtime = bool(self.config.class_margin_override_enabled)
        self._class_margin_adaptive_frozen = bool(self.config.class_margin_override_freeze_adaptive_tau)
        self._class_margin_collapse_alert = False
        self._class_margin_last_tau_adaptive: float | None = None

    def _normalize_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return logits - logits.max(dim=1, keepdim=True).values

    def _apply_temperature(self, logits: torch.Tensor) -> torch.Tensor:
        temp = max(1e-6, float(self.config.fixed_temperature))
        return logits / temp

    def _apply_prediction_floor(self, probs: torch.Tensor) -> torch.Tensor:
        floor = float(self.config.prediction_floor)
        if floor <= 0.0:
            return probs
        floored = torch.clamp(probs, min=floor)
        return floored / floored.sum(dim=1, keepdim=True)

    def _compute_precision_guard_ok(self) -> bool:
        baseline = self.config.class_margin_override_baseline_precision
        if baseline is None:
            return True

        labeled_n = int(sum(self._class_margin_guard_labeled))
        if labeled_n <= 0:
            return True

        pred4 = int(sum(self._class_margin_guard_pred4_labeled))
        if pred4 <= 0:
            # Undefined precision (no predicted positives); do not hard-disable.
            return True

        tp = int(sum(self._class_margin_guard_tp))
        live_precision = float(tp) / float(max(1, pred4))
        min_allowed = float(baseline) * float(self.config.class_margin_override_precision_disable_ratio)
        return bool(live_precision >= min_allowed)

    def _update_class_margin_guards(
        self,
        *,
        applied_mask: np.ndarray,
        pred: torch.Tensor,
        labels: np.ndarray | None = None,
    ) -> None:
        batch_size = int(pred.shape[0])
        self._class_margin_guard_overrides.append(int(np.sum(applied_mask)))
        self._class_margin_guard_total.append(batch_size)
        pred_np = pred.detach().cpu().numpy().astype(np.int64)
        cls = int(self.config.class_margin_override_class_id)
        pred4_mask = pred_np == cls
        self._class_margin_guard_pred4.append(int(np.sum(pred4_mask)))

        if labels is None:
            self._class_margin_guard_tp.append(0)
            self._class_margin_guard_pred4_labeled.append(0)
            self._class_margin_guard_labeled.append(0)
            return

        y = np.asarray(labels, dtype=np.int64)
        if y.ndim != 1 or int(y.shape[0]) != int(pred_np.shape[0]):
            self._class_margin_guard_tp.append(0)
            self._class_margin_guard_pred4_labeled.append(0)
            self._class_margin_guard_labeled.append(0)
            return

        tp = int(np.sum((y == cls) & pred4_mask))
        self._class_margin_guard_tp.append(tp)
        self._class_margin_guard_pred4_labeled.append(int(np.sum(pred4_mask)))
        self._class_margin_guard_labeled.append(1)

    def _adaptive_tau_from_buffer(self) -> float | None:
        frozen_tau = self.config.class_margin_override_frozen_tau_adaptive
        if frozen_tau is not None:
            return float(max(0.0, float(frozen_tau)))
        if len(self._class_margin_buffer) < int(max(1, self.config.class_margin_override_warmup_min_samples)):
            return None
        k = float(np.clip(float(self.config.class_margin_override_percentile_k), 0.0, 100.0))
        arr = np.asarray(list(self._class_margin_buffer), dtype=np.float64)
        if arr.size <= 0:
            return None
        return float(np.percentile(arr, k))

    def _margin_signal(self, margin: torch.Tensor) -> torch.Tensor:
        """Return threshold signal: raw margin or z-normalized margin based on config."""
        if not bool(self.config.class_margin_override_use_margin_zscore):
            return margin
        arr = np.asarray(list(self._class_margin_buffer), dtype=np.float64)
        if arr.size <= 1:
            return margin
        mu = float(np.mean(arr))
        sigma = float(np.std(arr))
        eps = float(max(1e-12, self.config.class_margin_override_margin_zscore_epsilon))
        denom = sigma if sigma > eps else eps
        return (margin - float(mu)) / float(denom)

    def _fixed_tau_signal(self) -> float:
        """Return fixed threshold in the same signal domain as margin signal."""
        tau_fixed = float(max(0.0, self.config.class_margin_override_tau))
        if not bool(self.config.class_margin_override_use_margin_zscore):
            return tau_fixed
        arr = np.asarray(list(self._class_margin_buffer), dtype=np.float64)
        if arr.size <= 1:
            return tau_fixed
        mu = float(np.mean(arr))
        sigma = float(np.std(arr))
        eps = float(max(1e-12, self.config.class_margin_override_margin_zscore_epsilon))
        denom = sigma if sigma > eps else eps
        return float((tau_fixed - mu) / denom)

    def _adaptive_tau_signal(self, tau_adaptive: float | None) -> float | None:
        if tau_adaptive is None:
            return None
        if not bool(self.config.class_margin_override_use_margin_zscore):
            return float(tau_adaptive)
        arr = np.asarray(list(self._class_margin_buffer), dtype=np.float64)
        if arr.size <= 1:
            return float(tau_adaptive)
        mu = float(np.mean(arr))
        sigma = float(np.std(arr))
        eps = float(max(1e-12, self.config.class_margin_override_margin_zscore_epsilon))
        denom = sigma if sigma > eps else eps
        return float((float(tau_adaptive) - mu) / denom)

    def _resolve_class_margin_switch(
        self,
        is_target_top1: torch.Tensor,
        fixed_cond: torch.Tensor,
        adaptive_cond: torch.Tensor,
        tau_adaptive: float | None,
    ) -> torch.Tensor:
        if bool(self.config.class_margin_override_hybrid_and):
            return is_target_top1 & fixed_cond & adaptive_cond
        if tau_adaptive is not None:
            return is_target_top1 & adaptive_cond
        return is_target_top1 & fixed_cond

    def _update_margin_buffer(self, *, is_target_top1: torch.Tensor, margin: torch.Tensor) -> None:
        mask = is_target_top1.detach().cpu().numpy().astype(bool)
        if not bool(np.any(mask)):
            return
        vals = margin.detach().cpu().numpy().astype(np.float64)
        for v in vals[mask].tolist():
            self._class_margin_buffer.append(float(v))

    def _apply_class_margin_override(
        self,
        pred: torch.Tensor,
        logits: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Optionally override target-class argmax to second-best under margin controls.

        Hybrid production rule (recommended):
            if argmax==class_id and margin<tau_adaptive and margin<tau_fixed -> switch

        Where:
            tau_adaptive is percentile_k from rolling buffer (when warmup satisfied),
            and tau_fixed is class_margin_override_tau.
        """
        batch_size = int(pred.shape[0])
        applied = np.zeros((batch_size,), dtype=bool)
        margin_arr = np.full((batch_size,), np.nan, dtype=np.float64)
        second_cls_arr = np.full((batch_size,), -1, dtype=np.int64)

        if not bool(self._class_margin_enabled_runtime):
            return applied, margin_arr, second_cls_arr

        class_id = int(self.config.class_margin_override_class_id)
        class_count = int(logits.shape[1])
        if class_id < 0 or class_id >= class_count:
            return applied, margin_arr, second_cls_arr

        top2 = torch.topk(logits, k=min(2, class_count), dim=1)
        top2_vals = top2.values
        top2_idx = top2.indices
        if int(top2_vals.shape[1]) < 2:
            return applied, margin_arr, second_cls_arr

        top1_idx = top2_idx[:, 0]
        top2_idx_only = top2_idx[:, 1]
        top1_val = top2_vals[:, 0]
        top2_val_only = top2_vals[:, 1]

        is_target_top1 = top1_idx == int(class_id)
        margin = top1_val - top2_val_only

        tau_fixed = float(max(0.0, self.config.class_margin_override_tau))
        tau_adaptive: float | None = None
        adaptive_requested = bool(self.config.class_margin_override_use_percentile)
        frozen_tau_cfg = self.config.class_margin_override_frozen_tau_adaptive
        use_adaptive = adaptive_requested and (
            frozen_tau_cfg is not None or not bool(self._class_margin_adaptive_frozen)
        )
        if use_adaptive:
            tau_adaptive = self._adaptive_tau_from_buffer()

        self._class_margin_last_tau_adaptive = float(tau_adaptive) if tau_adaptive is not None else None

        # Warmup guard for adaptive mode: disable overrides until enough class-id margins are buffered.
        # When adaptive tau is frozen by guard, fallback to fixed tau is still allowed.
        if adaptive_requested and not bool(self._class_margin_adaptive_frozen) and tau_adaptive is None:
            margin_arr = margin.detach().cpu().numpy().astype(np.float64)
            second_cls_arr = top2_idx_only.detach().cpu().numpy().astype(np.int64)
            return applied, margin_arr, second_cls_arr

        if tau_fixed <= 0.0 and tau_adaptive is None:
            margin_arr = margin.detach().cpu().numpy().astype(np.float64)
            second_cls_arr = top2_idx_only.detach().cpu().numpy().astype(np.int64)
            return applied, margin_arr, second_cls_arr

        tau_fixed_signal = self._fixed_tau_signal()
        margin_signal = self._margin_signal(margin)

        tau_adaptive_signal = self._adaptive_tau_signal(tau_adaptive)

        fixed_cond = (
            margin_signal < float(tau_fixed_signal)
            if tau_fixed > 0.0
            else torch.ones_like(is_target_top1)
        )
        adaptive_cond = (
            margin_signal < float(tau_adaptive_signal)
            if tau_adaptive_signal is not None
            else torch.ones_like(is_target_top1)
        )

        switch = self._resolve_class_margin_switch(
            is_target_top1,
            fixed_cond,
            adaptive_cond,
            tau_adaptive,
        )

        if bool(switch.any()):
            pred[switch] = top2_idx_only[switch]
            switch_np = switch.detach().cpu().numpy().astype(bool)
            applied[switch_np] = True

        # Update rolling buffer after decision to avoid current-batch self-influence in adaptive tau.
        self._update_margin_buffer(is_target_top1=is_target_top1, margin=margin)

        margin_arr = margin.detach().cpu().numpy().astype(np.float64)
        second_cls_arr = top2_idx_only.detach().cpu().numpy().astype(np.int64)
        return applied, margin_arr, second_cls_arr

    def predict(
        self,
        features: np.ndarray,
        *,
        active_classes: list[int] | None = None,
        enforce_global_coverage: bool | None = None,
        labels: np.ndarray | None = None,
    ) -> dict[str, Any]:
        x = self._to_2d_float32(features)

        with torch.no_grad():
            t = torch.from_numpy(x).to(self.device)
            binary_logits, family_logits = self.model(t)
            if int(binary_logits.shape[1]) != CANONICAL_BINARY_CLASSES:
                raise RuntimeError(
                    f"Runtime binary logits mismatch: expected {CANONICAL_BINARY_CLASSES}, got {int(binary_logits.shape[1])}"
                )
            if int(family_logits.shape[1]) != CANONICAL_FAMILY_CLASSES:
                raise RuntimeError(
                    f"Runtime family logits mismatch: expected {CANONICAL_FAMILY_CLASSES}, got {int(family_logits.shape[1])}"
                )
            family_logits = self._normalize_logits(family_logits)
            family_logits = self._apply_temperature(family_logits)
            probs = torch.softmax(family_logits, dim=1)
            probs = self._apply_prediction_floor(probs)

            pred = torch.argmax(probs, dim=1)

            batch_size = int(pred.shape[0])
            override_applied_arr = np.zeros((batch_size,), dtype=bool)
            override_class_arr: list[int | None] = [None] * batch_size
            override_logit_arr: list[float | None] = [None] * batch_size
            override_threshold_arr: list[float | None] = [None] * batch_size

            should_enforce = (
                self.config.global_coverage_floor if enforce_global_coverage is None else bool(enforce_global_coverage)
            )
            if should_enforce:
                target_classes = self._resolve_target_classes(int(probs.shape[1]), active_classes)
                (
                    override_applied_arr,
                    override_class_arr,
                    override_logit_arr,
                    override_threshold_arr,
                ) = self._apply_global_coverage_override(
                    pred,
                    family_logits,
                    target_classes=target_classes,
                )

            # Apply class-margin discontinuous decision rule after optional coverage override
            # so that class-margin policy is reflected in the final predicted class.
            margin_override_applied_arr, margin_override_margin_arr, margin_override_second_class_arr = (
                self._apply_class_margin_override(pred, family_logits)
            )

            # Online guards: override-rate guard and optional precision guard (when labels provided).
            self._update_class_margin_guards(
                applied_mask=margin_override_applied_arr,
                pred=pred,
                labels=labels,
            )
            guard_total = int(sum(self._class_margin_guard_total))
            guard_overrides = int(sum(self._class_margin_guard_overrides))
            override_rate = float(guard_overrides) / float(max(1, guard_total))
            if (
                override_rate > float(self.config.class_margin_override_rate_guard_threshold)
                and not self._class_margin_adaptive_frozen
            ):
                self._class_margin_adaptive_frozen = True
            if not self._compute_precision_guard_ok():
                self._class_margin_enabled_runtime = False

            if len(self._class_margin_buffer) > 1:
                buf = np.asarray(list(self._class_margin_buffer), dtype=np.float64)
                self._class_margin_collapse_alert = bool(
                    float(np.var(buf)) <= float(self.config.class_margin_buffer_variance_epsilon)
                )
            else:
                self._class_margin_collapse_alert = False

            conf = probs.gather(1, pred.unsqueeze(1)).squeeze(1)
        response = self._format_predict_response(
            pred,
            conf,
            probs,
            override_applied_arr,
            override_class_arr,
            override_logit_arr,
            override_threshold_arr,
            margin_override_applied_arr,
            margin_override_margin_arr,
            margin_override_second_class_arr,
        )
        response["class_margin_tau_adaptive"] = (
            None
            if self._class_margin_last_tau_adaptive is None
            else float(self._class_margin_last_tau_adaptive)
        )
        response["class_margin_adaptive_frozen"] = bool(self._class_margin_adaptive_frozen)
        response["class_margin_buffer_size"] = int(len(self._class_margin_buffer))
        response["class_margin_enabled"] = bool(self._class_margin_enabled_runtime)
        response["class_margin_collapse_alert"] = bool(self._class_margin_collapse_alert)
        return response

    def export_torchscript(self, output_path: Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        model_cpu = self.model.cpu().eval()
        example = torch.arange(self.model.input_dim, dtype=torch.float32).reshape(1, -1)
        traced = torch.jit.trace(model_cpu, example, strict=False)
        contract = self._contract_metadata()
        manifest_base = build_export_manifest(
            contract=contract,
            model_architecture=self.model.__class__.__name__,
            export_config={"format": "torchscript"},
            runtime_version=str(torch.__version__),
        )
        # Embed the manifest in the TorchScript extra files using the
        # canonical torchscript_extra_files_for_manifest format.
        from helix_ids.governance.provenance import torchscript_extra_files_for_manifest

        torch.jit.save(
            traced,
            str(output_path),
            _extra_files=torchscript_extra_files_for_manifest(manifest_base),
        )
        sidecars = write_contract_sidecars(output_path, contract)
        manifest = finalize_export_artifact(output_path, manifest_base, sidecars=sidecars)
        from helix_ids.governance.provenance import manifest_without_artifact_sha256

        verify_export_artifact(
            output_path,
            kind="torchscript",
            contract=contract,
            embedded_manifest=manifest_without_artifact_sha256(manifest),
        )
        self._validate_torchscript_parity(traced, example)
        self.model.to(self.device)
        return output_path

    def export_onnx(self, output_path: Path, *, opset: int = 13) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.eval()
        dummy = torch.arange(self.model.input_dim, dtype=torch.float32, device=self.device).reshape(1, -1)
        contract = self._contract_metadata()
        manifest_base = build_export_manifest(
            contract=contract,
            model_architecture=self.model.__class__.__name__,
            export_config={"format": "onnx", "opset": opset},
            onnx_opset=opset,
            runtime_version=str(torch.__version__),
        )
        torch.onnx.export(
            self.model,
            (dummy,),
            str(output_path),
            export_params=True,
            opset_version=opset,
            input_names=["features"],
            output_names=["binary_logits", "family_logits"],
            dynamic_axes={
                "features": {0: "batch"},
                "binary_logits": {0: "batch"},
                "family_logits": {0: "batch"},
            },
        )
        import onnx

        model = onnx.load(str(output_path))
        for key, value in contract.items():
            meta = model.metadata_props.add()
            meta.key = str(key)
            meta.value = json.dumps(value) if isinstance(value, (list, dict)) else str(value)
        embed_manifest_in_onnx_metadata(model, manifest_base)
        onnx.save(model, str(output_path))
        sidecars = write_contract_sidecars(output_path, contract)
        manifest = finalize_export_artifact(output_path, manifest_base, sidecars=sidecars)
        from helix_ids.governance.provenance import manifest_without_artifact_sha256

        verify_export_artifact(
            output_path,
            kind="onnx",
            contract=contract,
            embedded_manifest=manifest_without_artifact_sha256(manifest),
        )
        self._validate_onnx_parity(output_path, dummy)
        self._validate_onnx_metadata(output_path)
        return output_path
