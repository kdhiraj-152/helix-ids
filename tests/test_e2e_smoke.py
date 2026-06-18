"""
E2E smoke tests for HELIX-IDS training, inference, and governance flows.

Requirements:
- Runtime under 60 seconds.
- Purely synthetic data (no external datasets).
- Exercises real code paths for: training, checkpoint creation,
  inference runtime loading/prediction, and governance roundtrip.

Flow A: Train -> Checkpoint
  - Tiny synthetic dataset, 2 epochs.
  - Save checkpoint with full contract sidecars.
  - Verify file integrity and contract metadata.

Flow B: Export -> Load -> Predict
  - Load Flow A checkpoint through HelixInferenceRuntime.
  - Run prediction on synthetic inputs.
  - Verify output schema, shape, and confidence bounds.

Flow C: Governance Roundtrip
  - Run lifecycle_verifier end-to-end (creates + verifies artifacts).
  - Run GateOrchestrator stage sequence.
  - Run promotion consensus with seed run summaries.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from helix_ids import __version__ as HELIX_IDS_VERSION
from helix_ids.contracts.schema_contract import (
    CANONICAL_BINARY_CLASSES,
    CANONICAL_FAMILY_CLASSES,
    CANONICAL_FEATURE_ORDER,
    CANONICAL_INPUT_DIM,
    SCHEMA_VERSION,
    compute_schema_hash,
    runtime_contract_payload,
)
from helix_ids.governance.orchestrator import (
    GateOrchestrator,
)
from helix_ids.governance.parameters import DEFAULT_GOVERNANCE_POLICY
from helix_ids.governance.promotion import (
    SeedRunSummary,
    aggregate_seed_runs,
)
from helix_ids.governance.provenance import (
    ARTIFACT_MANIFEST_KEY,
    build_artifact_manifest,
    build_provenance_chain,
    finalize_artifact_manifest,
    write_contract_sidecars,
)
from helix_ids.models.helix_ids_full import (
    HelixIDSFull,
    MultiTaskLoss,
)
from helix_ids.operations.inference_runtime import HelixInferenceRuntime, InferenceConfig

# ============================================================================
# Helpers
# ============================================================================


def _make_synthetic_data(
    n: int = 16,
    *,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create synthetic (features, binary_labels, family_labels)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, CANONICAL_INPUT_DIM)).astype(np.float32)
    y_bin = (x.sum(axis=1) > 0).astype(np.int64)
    y_family = (np.abs(x[:, 0]) * 3 + np.abs(x[:, 1]) * 5).astype(np.int64) % CANONICAL_FAMILY_CLASSES
    return (
        torch.from_numpy(x),
        torch.from_numpy(y_bin),
        torch.from_numpy(y_family),
    )


def _train_helix_full(
    x: torch.Tensor,
    y_bin: torch.Tensor,
    y_family: torch.Tensor,
    *,
    epochs: int = 2,
) -> HelixIDSFull:
    """Train a HelixIDSFull model on synthetic data for `epochs` epochs."""
    model = HelixIDSFull()
    loss_fn = MultiTaskLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        bin_logits, fam_logits = model(x)
        loss, _ = loss_fn(bin_logits, y_bin, fam_logits, y_family)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def _save_checkpoint(
    model: HelixIDSFull,
    path: Path,
    *,
    git_commit: str = HELIX_IDS_VERSION,
) -> dict[str, Any]:
    """Save model checkpoint with full contract sidecars and provenance chain.

    This replicates the lifecycle_verifier._write_checkpoint pattern
    using the public provenance API so HelixInferenceRuntime can load it.
    """
    contract = runtime_contract_payload()
    manifest = build_artifact_manifest(
        contract=contract,
        model_architecture=model.__class__.__name__,
        export_config={"format": "checkpoint"},
        git_commit=git_commit,
        exporter_version=HELIX_IDS_VERSION,
        runtime_version=HELIX_IDS_VERSION,
    )
    payload: dict[str, Any] = {"model_state_dict": model.state_dict()}
    payload.update(contract)
    # checkpoint_manifest_payload excludes artifact_sha256 (not yet known)
    from helix_ids.governance.provenance import checkpoint_manifest_payload

    payload[ARTIFACT_MANIFEST_KEY] = checkpoint_manifest_payload(manifest)
    torch.save(payload, path)
    sidecars = write_contract_sidecars(path, contract)
    chain = build_provenance_chain(path, manifest=manifest, sidecars=sidecars)
    finalized = finalize_artifact_manifest(path, manifest, provenance_chain=chain)
    return finalized


# ============================================================================
# Flow A: Train -> Checkpoint
# ============================================================================


class TestFlowA_TrainCheckpoint:
    """Verify that training produces a loadable checkpoint with contract metadata."""

    def test_train_and_save_checkpoint(self, tmp_path: Path) -> None:
        """Train on synthetic data for 2 epochs and save checkpoint."""
        x, y_bin, y_family = _make_synthetic_data()
        model = _train_helix_full(x, y_bin, y_family)

        # Verify model produces meaningful output
        with torch.no_grad():
            bin_logits, fam_logits = model(x)
        assert bin_logits.shape == (16, CANONICAL_BINARY_CLASSES)
        assert fam_logits.shape == (16, CANONICAL_FAMILY_CLASSES)
        assert torch.isfinite(bin_logits).all()
        assert torch.isfinite(fam_logits).all()

        # Save checkpoint
        ckpt_path = tmp_path / "helix_e2e.pt"
        _save_checkpoint(model, ckpt_path)

        # Verify checkpoint file exists and is non-empty
        assert ckpt_path.exists()
        assert ckpt_path.stat().st_size > 0

        # Verify sidecars exist
        assert ckpt_path.with_suffix(ckpt_path.suffix + ".contract.json").exists()
        assert ckpt_path.with_suffix(ckpt_path.suffix + ".feature_order.json").exists()
        assert ckpt_path.with_suffix(ckpt_path.suffix + ".schema_hash.txt").exists()

        # Verify manifest sidecar exists
        assert ckpt_path.with_suffix(ckpt_path.suffix + ".manifest.json").exists()

        # Verify contract sidecar content matches runtime contract
        contract_sidecar = json.loads(
            ckpt_path.with_suffix(ckpt_path.suffix + ".contract.json").read_text(encoding="utf-8")
        )
        assert contract_sidecar["schema_version"] == SCHEMA_VERSION
        assert contract_sidecar["input_dim"] == CANONICAL_INPUT_DIM
        assert contract_sidecar["binary_output_dim"] == CANONICAL_BINARY_CLASSES
        assert contract_sidecar["family_output_dim"] == CANONICAL_FAMILY_CLASSES
        assert contract_sidecar["feature_order"] == CANONICAL_FEATURE_ORDER
        expected_hash = compute_schema_hash()
        assert contract_sidecar["schema_hash"] == expected_hash


# ============================================================================
# Flow B: Export -> Load -> Predict
# ============================================================================


class TestFlowB_ExportLoadPredict:
    """Verify checkpoint can be loaded by HelixInferenceRuntime and produce predictions."""

    @pytest.fixture(autouse=True)
    def _setup_flow_b(self, tmp_path: Path) -> None:
        """Train model and save checkpoint once for all test methods."""
        x, y_bin, y_family = _make_synthetic_data()
        model = _train_helix_full(x, y_bin, y_family)
        self._checkpoint_path = tmp_path / "helix_e2e_inference.pt"
        _save_checkpoint(model, self._checkpoint_path)
        self._checkpoint_path = self._checkpoint_path  # type: ignore[assignment]

    def test_load_checkpoint_through_runtime(self) -> None:
        """HelixInferenceRuntime loads checkpoint and configures model."""
        config = InferenceConfig(
            global_coverage_floor=False,
            class_margin_override_enabled=False,
        )
        runtime = HelixInferenceRuntime(
            self._checkpoint_path,
            device="cpu",
            config=config,
        )
        assert runtime.model is not None
        assert runtime.model.input_dim == CANONICAL_INPUT_DIM
        assert runtime.feature_order == list(CANONICAL_FEATURE_ORDER)
        assert runtime.schema_version == SCHEMA_VERSION

    def test_predict_output_schema(self) -> None:
        """predict() returns expected schema with correct shapes."""
        config = InferenceConfig(
            global_coverage_floor=False,
            class_margin_override_enabled=False,
        )
        runtime = HelixInferenceRuntime(
            self._checkpoint_path,
            device="cpu",
            config=config,
        )

        features = np.random.default_rng(42).standard_normal((4, CANONICAL_INPUT_DIM)).astype(np.float32)
        result = runtime.predict(features)

        # Required keys
        assert "family_class" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert "coverage_override_applied" in result
        assert "class_margin_override_applied" in result

        # Output shapes for single batch
        if isinstance(result["family_class"], int):
            assert 0 <= result["family_class"] < CANONICAL_FAMILY_CLASSES
        elif isinstance(result["family_class"], list):
            assert len(result["family_class"]) == 4
            for cls in result["family_class"]:
                assert 0 <= cls < CANONICAL_FAMILY_CLASSES

        # Confidence bound
        conf = result["confidence"]
        if isinstance(conf, float):
            assert 0.0 < conf <= 1.0
        elif isinstance(conf, list):
            assert len(conf) == 4
            for c in conf:
                assert 0.0 < c <= 1.0

        # Probability vector
        probs = result["probabilities"]
        if isinstance(probs, list) and len(probs) > 0:
            if isinstance(probs[0], list):
                # batch
                assert len(probs) == 4
                for row in probs:
                    assert len(row) == CANONICAL_FAMILY_CLASSES
                    assert abs(sum(row) - 1.0) < 1e-5
            else:
                assert len(probs) == CANONICAL_FAMILY_CLASSES
                assert abs(sum(probs) - 1.0) < 1e-5

    def test_predict_batch_output(self) -> None:
        """predict() handles batch inputs with correct post-processing."""
        config = InferenceConfig(
            global_coverage_floor=False,
            class_margin_override_enabled=False,
        )
        runtime = HelixInferenceRuntime(
            self._checkpoint_path,
            device="cpu",
            config=config,
        )

        for batch_size in (1, 2, 8):
            features = np.random.default_rng(batch_size).standard_normal(
                (batch_size, CANONICAL_INPUT_DIM)
            ).astype(np.float32)
            result = runtime.predict(features)

            if batch_size == 1:
                # Single-item output
                assert isinstance(result["family_class"], int)
            else:
                # Batch output
                assert isinstance(result["family_class"], list)
                assert len(result["family_class"]) == batch_size

            assert "coverage_override_applied" in result
            assert "class_margin_override_applied" in result

    def test_predict_1d_input_reshaped(self) -> None:
        """1D input is auto-reshaped to (1, input_dim)."""
        config = InferenceConfig(
            global_coverage_floor=False,
            class_margin_override_enabled=False,
        )
        runtime = HelixInferenceRuntime(
            self._checkpoint_path,
            device="cpu",
            config=config,
        )

        features_1d = np.random.default_rng(99).standard_normal(CANONICAL_INPUT_DIM).astype(np.float32)
        assert features_1d.ndim == 1
        result = runtime.predict(features_1d)
        assert isinstance(result["family_class"], int)


# ============================================================================
# Flow C: Governance Roundtrip
# ============================================================================


class TestFlowC_GovernanceRoundtrip:
    """Verify governance orchestration, lifecycle artifacts, and promotion."""

    def test_gate_orchestrator_stage_sequence(self) -> None:
        """GateOrchestrator runs full stage sequence with relaxed policy."""
        relaxed = dataclasses.replace(
            DEFAULT_GOVERNANCE_POLICY,
            stage_timeouts=dataclasses.replace(
                DEFAULT_GOVERNANCE_POLICY.stage_timeouts,
                preload_seconds=30,
                pretrain_seconds=30,
                posteval_seconds=30,
                prepromote_seconds=30,
            ),
        )
        orchestrator = GateOrchestrator(
            policy=relaxed,
            strict_missing_metrics=False,
        )
        context: dict[str, Any] = {
            "run_id": "smoke-test",
            "entrypoint": "test_e2e",
            "model_architecture": "HelixIDSFull",
            "config": {"epochs": 2},
            "dataset": "synthetic",
            "seed": 42,
            # prepromote stage required fields
            "seed_run_count": 3,
            "consensus_pass": True,
            "inter_seed_macro_f1_variance": 0.005,
            "reproducibility_delta": 0.005,
            # posteval stage required fields
            "macro_f1_ci_width": 0.03,
            "macro_f1_ci_lower": 0.82,
            "abs_macro_f1_drift": 0.01,
            "abs_macro_f1_zscore": 1.5,
        }
        decisions = orchestrator.run_stage_sequence(context)
        assert len(decisions) >= 1
        for decision in decisions:
            assert decision.status == "PASS", f"Gate {decision.gate_id} failed: {decision.reason_code}"

    def test_lifecycle_artifacts_pass_verification(self) -> None:
        """Lifecycle verification succeeds on synthetic tiny model (req ONNX)."""
        from helix_ids.governance.lifecycle_verifier import run_lifecycle_verification

        result = run_lifecycle_verification(
            Path("/tmp/helix_e2e_lifecycle"),
            require_onnx=True,
        )
        assert result["parity_ok"] is True
        assert "artifacts" in result
        assert result["artifacts"]["checkpoint"].exists()
        assert result["artifacts"]["torchscript"].exists()
        assert result["artifacts"]["onnx"].exists()

    def test_promotion_consensus_aggregation(self) -> None:
        """Seed-run aggregation produces a PromotionConsensus."""
        seed_runs = [
            SeedRunSummary(seed=1, macro_f1=0.85, macro_f1_ci_lower=0.82, macro_f1_ci_width=0.03, tier2_pass=True),
            SeedRunSummary(seed=2, macro_f1=0.83, macro_f1_ci_lower=0.80, macro_f1_ci_width=0.03, tier2_pass=True),
            SeedRunSummary(seed=3, macro_f1=0.84, macro_f1_ci_lower=0.81, macro_f1_ci_width=0.03, tier2_pass=True),
        ]
        consensus = aggregate_seed_runs(
            seed_runs=seed_runs,
            min_seed_runs=3,
            max_inter_seed_macro_f1_variance=0.01,
            reproducibility_tolerance=0.03,
            min_ci95_lower_bound=0.78,
            max_ci_width=0.05,
        )
        assert consensus.mean_macro_f1 >= 0.83
        assert consensus.consensus_pass is True
        assert consensus.seed_run_count == 3

    def test_gate_orchestrator_custom_stage(self) -> None:
        """GateOrchestrator supports custom stage registration."""
        orchestrator = GateOrchestrator(strict_missing_metrics=False)
        custom_calls: list[str] = []

        def custom_gate(context: dict) -> tuple[bool, float | None, float | None, str | None]:
            custom_calls.append("called")
            return True, 1.0, 1.0, None

        orchestrator.register_gate("preload", "custom_smoke_gate", custom_gate)
        decisions = orchestrator.run("preload", {"run_id": "smoke-custom", "entrypoint": "custom_test"})
        assert len(custom_calls) == 1
        assert all(d.status == "PASS" for d in decisions)
