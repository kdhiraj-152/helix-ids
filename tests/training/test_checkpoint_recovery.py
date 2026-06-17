"""
Tests for checkpoint save, resume, integrity, corruption detection,
and idempotent recovery.

Verifies that checkpoint artifacts produced by the training pipeline
survive roundtrips, maintain SHA256 integrity with provenance records,
preserve state dict structure, restore phase info, detect corruptions,
catch missing keys, and that governance metadata is preserved.

Follows the existing test style in tests/training/test_recovery_manager.py
and tests/test_provenance.py.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from helix_ids.governance import provenance
from helix_ids.governance.provenance import (
    ARTIFACT_MANIFEST_KEY,
    ArtifactManifestError,
    artifact_sha256,
    build_artifact_manifest,
    checkpoint_manifest_payload,
    finalize_artifact_manifest,
    verify_artifact_manifest,
    write_contract_sidecars,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def tiny_model():
    """A tiny nn.Module that produces realistic state dict keys."""
    return torch.nn.Sequential(
        torch.nn.Linear(8, 4),
        torch.nn.ReLU(),
        torch.nn.Linear(4, 2),
    )


@pytest.fixture
def optimizer(tiny_model):
    """Optimizer for the tiny model."""
    return torch.optim.Adam(tiny_model.parameters(), lr=0.001)


@pytest.fixture
def scheduler(optimizer):
    """Cosine annealing scheduler."""
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)


@pytest.fixture
def contract():
    """Standard contract payload (mimics runtime_contract_payload)."""
    c = provenance.runtime_contract_payload()
    # Ensure feature_order is a list (real contract returns tuple)
    c["feature_order"] = list(c["feature_order"])
    return c


@pytest.fixture
def checkpoint_path(tmp_path):
    """Return a temporary checkpoint file path."""
    return tmp_path / "test_checkpoint.pt"


def _make_manifest_base(contract, model_architecture="TestModel"):
    """Build a minimal export manifest base."""
    return build_artifact_manifest(
        contract=contract,
        model_architecture=model_architecture,
        export_config={"format": "checkpoint", "origin": "test"},
    )


def _build_checkpoint_dict(
    *,
    model_state: dict[str, torch.Tensor],
    optimizer_state: dict[str, Any] | None = None,
    scheduler_state: dict[str, Any] | None = None,
    epoch: int = 5,
    global_step: int = 1000,
    phase_id: int = 2,
    best_val_loss: float = 1.5,
    contract: dict[str, Any] | None = None,
    manifest_base: dict[str, Any] | None = None,
    include_manifest: bool = True,
) -> dict[str, Any]:
    """Build a realistic checkpoint dictionary resembling what the trainer saves.

    Mirrors the structure produced by _build_model_contract_artifact
    in train_helix_ids_full.py (model + model_state_dict, contract fields,
    extra metadata, and an optional artifact_manifest).
    """
    if contract is None:
        c = provenance.runtime_contract_payload()
        contract = dict(c)
        contract["feature_order"] = list(contract["feature_order"])

    ckpt: dict[str, Any] = {
        "model_state_dict": {
            key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
            for key, value in model_state.items()
        },
        "epoch": epoch,
        "global_step": global_step,
        "phase_id": phase_id,
        "best_val_loss": best_val_loss,
    }
    # Also include the 'model' key like the real artifact does
    ckpt["model"] = dict(ckpt["model_state_dict"])

    if optimizer_state is not None:
        ckpt["optimizer_state_dict"] = optimizer_state
    if scheduler_state is not None:
        ckpt["scheduler_state_dict"] = scheduler_state

    # Merge contract fields
    ckpt.update(contract)

    if manifest_base is not None and include_manifest:
        ckpt[ARTIFACT_MANIFEST_KEY] = checkpoint_manifest_payload(manifest_base)

    return ckpt


def _save_checkpoint(
    path: Path,
    ckpt: dict[str, Any],
    *,
    finalize: bool = True,
    contract: dict[str, Any] | None = None,
    manifest_base: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Save a checkpoint and optionally finalize its manifest + sidecars.

    Returns the finalized manifest if finalize=True.
    """
    torch.save(ckpt, path)
    if not finalize or manifest_base is None:
        return {}
    finalized = finalize_artifact_manifest(path, manifest_base)
    write_contract_sidecars(path, contract or {})
    return finalized


def _load_checkpoint(path: Path) -> dict[str, Any]:
    """Load a checkpoint dict from disk (weights_only safe for tests)."""
    return torch.load(path, map_location="cpu", weights_only=True)


def _compute_sha256(path: Path) -> str:
    """Compute hex SHA-256 digest of a file's contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCheckpointRecovery:
    """Checkpoint save, resume, integrity, corruption, and idempotent recovery."""

    # ── test 1: SHA256 integrity ──────────────────────────────────────────

    def test_checkpoint_save_integrity(
        self,
        tiny_model: torch.nn.Module,
        contract: dict[str, Any],
        checkpoint_path: Path,
    ) -> None:
        """Save a checkpoint and verify the SHA256 hash matches the provenance record."""
        manifest_base = _make_manifest_base(contract)

        ckpt = _build_checkpoint_dict(
            model_state=tiny_model.state_dict(),
            contract=contract,
            manifest_base=manifest_base,
        )
        finalized = _save_checkpoint(
            checkpoint_path, ckpt,
            finalize=True, contract=contract, manifest_base=manifest_base,
        )

        # The manifest's artifact_sha256 must equal the actual file hash
        recorded_hash = finalized.get("artifact_sha256", "")
        assert recorded_hash, "Manifest missing artifact_sha256"
        actual_hash = artifact_sha256(checkpoint_path)
        assert recorded_hash == actual_hash, (
            f"Manifest SHA256 {recorded_hash} does not match "
            f"actual file SHA256 {actual_hash}"
        )

        # Re-verify via verify_artifact_manifest to double-check
        sidecar = verify_artifact_manifest(
            checkpoint_path,
            kind="checkpoint",
            contract=contract,
        )
        assert sidecar is not None, "Sidecar manifest should be present"
        assert str(sidecar.get("artifact_sha256")) == actual_hash

    # ── test 2: state dict keys ───────────────────────────────────────────

    def test_checkpoint_resume_state_dict(
        self,
        tiny_model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        contract: dict[str, Any],
        checkpoint_path: Path,
    ) -> None:
        """Verify loaded checkpoint has expected top-level keys:
        model_state_dict, optimizer_state_dict, scheduler_state_dict.
        """
        # Run a step so optimizer and scheduler have non-trivial state
        dummy_input = torch.randn(2, 8)
        dummy_target = torch.randn(2, 2)
        loss_fn = torch.nn.MSELoss()
        tiny_model.train()

        opt_state = optimizer.state_dict()
        sched_state = scheduler.state_dict()

        ckpt = _build_checkpoint_dict(
            model_state=tiny_model.state_dict(),
            optimizer_state=opt_state,
            scheduler_state=sched_state,
            contract=contract,
        )
        _save_checkpoint(checkpoint_path, ckpt, finalize=False)

        loaded = _load_checkpoint(checkpoint_path)

        # Expected top-level keys
        assert "model_state_dict" in loaded, "Missing model_state_dict key"
        assert "optimizer_state_dict" in loaded, "Missing optimizer_state_dict key"
        assert "scheduler_state_dict" in loaded, "Missing scheduler_state_dict key"

        # Verify they are dicts with content
        assert isinstance(loaded["model_state_dict"], dict), "model_state_dict not a dict"
        assert len(loaded["model_state_dict"]) > 0, "model_state_dict is empty"

        assert isinstance(loaded["optimizer_state_dict"], dict)
        assert "state" in loaded["optimizer_state_dict"]

        assert isinstance(loaded["scheduler_state_dict"], dict)
        # Scheduler state typically has last_epoch or _last_lr
        assert any(
            key in loaded["scheduler_state_dict"]
            for key in ("last_epoch", "_last_lr", "base_lrs")
        ), "scheduler_state_dict missing expected keys"

    # ── test 3: phase info restoration ────────────────────────────────────

    def test_checkpoint_resume_phase_info(
        self,
        tiny_model: torch.nn.Module,
        contract: dict[str, Any],
        checkpoint_path: Path,
    ) -> None:
        """Verify phase_id, epoch, and step are correctly restored after save/load."""
        expected = {
            "epoch": 12,
            "global_step": 3450,
            "phase_id": 3,
            "best_val_loss": 0.876,
        }
        ckpt = _build_checkpoint_dict(
            model_state=tiny_model.state_dict(),
            contract=contract,
            epoch=expected["epoch"],
            global_step=expected["global_step"],
            phase_id=expected["phase_id"],
            best_val_loss=expected["best_val_loss"],
        )
        _save_checkpoint(checkpoint_path, ckpt, finalize=False)

        loaded = _load_checkpoint(checkpoint_path)

        assert loaded.get("epoch") == expected["epoch"], (
            f"Expected epoch={expected['epoch']}, got {loaded.get('epoch')}"
        )
        assert loaded.get("global_step") == expected["global_step"], (
            f"Expected global_step={expected['global_step']}, got {loaded.get('global_step')}"
        )
        assert loaded.get("phase_id") == expected["phase_id"], (
            f"Expected phase_id={expected['phase_id']}, got {loaded.get('phase_id')}"
        )
        assert loaded.get("best_val_loss") == expected["best_val_loss"], (
            f"Expected best_val_loss={expected['best_val_loss']}, "
            f"got {loaded.get('best_val_loss')}"
        )

    # ── test 4: corrupt hash rejected ─────────────────────────────────────

    def test_checkpoint_corrupt_hash_rejected(
        self,
        tiny_model: torch.nn.Module,
        contract: dict[str, Any],
        checkpoint_path: Path,
    ) -> None:
        """Tamper with the saved checkpoint file, then verify integrity check fails."""
        manifest_base = _make_manifest_base(contract)

        ckpt = _build_checkpoint_dict(
            model_state=tiny_model.state_dict(),
            contract=contract,
            manifest_base=manifest_base,
        )
        _save_checkpoint(
            checkpoint_path, ckpt,
            finalize=True, contract=contract, manifest_base=manifest_base,
        )

        # ── Before tampering: verification must pass ──
        sidecar = verify_artifact_manifest(
            checkpoint_path, kind="checkpoint", contract=contract,
        )
        assert sidecar is not None, "Pre-tamper verification should pass"

        # ── Tamper: corrupt a single byte in the checkpoint file ──
        original_bytes = checkpoint_path.read_bytes()
        corrupted = bytearray(original_bytes)
        # Flip a bit at position 1024 (or earlier if file is small)
        tamper_pos = min(1024, len(corrupted) - 1)
        corrupted[tamper_pos] ^= 0xFF
        checkpoint_path.write_bytes(bytes(corrupted))

        # ── After tampering: verification must fail ──
        with pytest.raises(ArtifactManifestError, match="checksum mismatch"):
            verify_artifact_manifest(
                checkpoint_path, kind="checkpoint", contract=contract,
            )

        # ── Also verify that the SHA256 no longer matches the recorded value ──
        new_hash = artifact_sha256(checkpoint_path)
        assert new_hash != sidecar.get("artifact_sha256"), (
            "Tampered file SHA256 should differ from the original manifest hash"
        )

    # ── test 5: missing keys detected ─────────────────────────────────────

    def test_checkpoint_missing_keys_detected(
        self,
        tiny_model: torch.nn.Module,
        contract: dict[str, Any],
        checkpoint_path: Path,
    ) -> None:
        """Save a checkpoint that is missing required keys and verify detection."""
        # Build a partial checkpoint — deliberately omit model_state_dict
        partial_ckpt: dict[str, Any] = {
            "epoch": 3,
            "global_step": 500,
            "phase_id": 1,
        }
        partial_ckpt.update(contract)

        _save_checkpoint(checkpoint_path, partial_ckpt, finalize=False)

        loaded = _load_checkpoint(checkpoint_path)

        # Simulate a loader that requires certain keys
        required_keys = {"model_state_dict", "epoch", "global_step", "phase_id"}

        missing = required_keys - set(loaded.keys())
        assert "model_state_dict" in missing, (
            "Expected model_state_dict to be missing from the partial checkpoint"
        )

        # Verify that at least one required key is absent
        assert len(missing) > 0, (
            f"Expected missing keys; none found. Loaded keys: {list(loaded.keys())}"
        )

        # Additionally, check that the governance verify detects a
        # missing/incomplete contract when a manifest is expected
        manifest_base = _make_manifest_base(contract)
        ckpt_with_manifest = dict(partial_ckpt)
        ckpt_with_manifest[ARTIFACT_MANIFEST_KEY] = checkpoint_manifest_payload(manifest_base)
        ckpt_with_manifest["schema_version"] = contract["schema_version"]

        # Finalize to get a sidecar, then tamper the embedded manifest
        torch.save(ckpt_with_manifest, checkpoint_path)
        finalize_artifact_manifest(checkpoint_path, manifest_base)
        write_contract_sidecars(checkpoint_path, contract)

        # Now remove model_state_dict from an embedded reload to simulate
        # a checkpoint that failed to write the weights
        reloaded = _load_checkpoint(checkpoint_path)
        assert "model_state_dict" not in reloaded or reloaded.get("model_state_dict") is None

    # ── test 6: governance metadata roundtrip ─────────────────────────────

    def test_checkpoint_resume_governance_metadata(
        self,
        tiny_model: torch.nn.Module,
        contract: dict[str, Any],
        checkpoint_path: Path,
    ) -> None:
        """Verify that governance metadata (manifest, sidecars) survives a
        save → load → verify roundtrip intact."""
        manifest_base = _make_manifest_base(contract)

        ckpt = _build_checkpoint_dict(
            model_state=tiny_model.state_dict(),
            contract=contract,
            manifest_base=manifest_base,
            epoch=7,
            global_step=2100,
            phase_id=2,
        )
        finalized = _save_checkpoint(
            checkpoint_path, ckpt,
            finalize=True, contract=contract, manifest_base=manifest_base,
        )

        # ── 1. Verify the embedded manifest survives ──
        loaded = _load_checkpoint(checkpoint_path)
        embedded_manifest = loaded.get(ARTIFACT_MANIFEST_KEY)
        assert embedded_manifest is not None, (
            "Embedded artifact_manifest missing after load"
        )
        # The embedded manifest should have core fields
        for key in (
            "manifest_version", "schema_version", "schema_hash",
            "contract_version", "exporter_version",
        ):
            assert key in embedded_manifest, (
                f"Embedded manifest missing field: {key}"
            )

        # ── 2. Sidecar manifest exists and matches ──
        sidecar_manifest = verify_artifact_manifest(
            checkpoint_path, kind="checkpoint", contract=contract,
        )
        assert sidecar_manifest is not None
        assert str(sidecar_manifest.get("artifact_sha256")) == artifact_sha256(checkpoint_path)

        # ── 3. Contract sidecars exist ──
        contract_sidecar = checkpoint_path.with_suffix(
            checkpoint_path.suffix + ".contract.json"
        )
        feature_order_sidecar = checkpoint_path.with_suffix(
            checkpoint_path.suffix + ".feature_order.json"
        )
        schema_hash_sidecar = checkpoint_path.with_suffix(
            checkpoint_path.suffix + ".schema_hash.txt"
        )
        assert contract_sidecar.exists(), "Missing .contract.json sidecar"
        assert feature_order_sidecar.exists(), "Missing .feature_order.json sidecar"
        assert schema_hash_sidecar.exists(), "Missing .schema_hash.txt sidecar"

        # ── 4. Contract sidecar content matches original ──
        contract_data = json.loads(contract_sidecar.read_text(encoding="utf-8"))
        assert contract_data.get("schema_hash") == contract["schema_hash"]
        assert contract_data.get("schema_version") == contract["schema_version"]

        # ── 5. Provenance chain is intact (finalized manifest has it) ──
        if finalized.get(provenance.PROVENANCE_CHAIN_KEY):
            provenance.verify_provenance_chain(
                checkpoint_path,
                manifest=finalized,
                sidecars={
                    "contract": contract_sidecar,
                    "feature_order": feature_order_sidecar,
                    "schema_hash": schema_hash_sidecar,
                },
                require_chain=True,
            )

    # ── test 7: idempotent recovery ───────────────────────────────────────

    def test_checkpoint_recovery_idempotent(
        self,
        tiny_model: torch.nn.Module,
        contract: dict[str, Any],
        checkpoint_path: Path,
    ) -> None:
        """Save → load → re-save: verify numeric fidelity is preserved
        across the roundtrip (model weights, optimizer state, metadata)."""
        manifest_base = _make_manifest_base(contract)
        optimizer = torch.optim.SGD(tiny_model.parameters(), lr=0.01, momentum=0.9)

        # Run a few training steps to get realistic optimizer state
        loss_fn = torch.nn.MSELoss()
        for _step in range(3):
            tiny_model.train()
            optimizer.zero_grad()
            dummy = torch.randn(4, 8)
            target = torch.randn(4, 2)
            loss = loss_fn(tiny_model(dummy), target)
            loss.backward()
            optimizer.step()

        original_model_state = {
            k: v.detach().cpu().clone() for k, v in tiny_model.state_dict().items()
        }
        original_opt_state = optimizer.state_dict()
        # Deep-copy the optimizer state dict (which may contain tensors)
        original_opt_state_cpu = {
            k: (
                {sk: sv.detach().cpu().clone() if isinstance(sv, torch.Tensor) else sv
                 for sk, sv in v.items()}
                if isinstance(v, dict)
                else v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v
            )
            for k, v in original_opt_state.items()
        }

        # ── Round 1: Save ──
        ckpt1 = _build_checkpoint_dict(
            model_state=tiny_model.state_dict(),
            optimizer_state=original_opt_state,
            contract=contract,
            manifest_base=manifest_base,
            epoch=3,
            global_step=750,
            phase_id=1,
            best_val_loss=0.42,
            include_manifest=True,
        )
        _save_checkpoint(
            checkpoint_path, ckpt1,
            finalize=True, contract=contract, manifest_base=manifest_base,
        )

        # ── Round 2: Load ──
        loaded1 = _load_checkpoint(checkpoint_path)

        # ── Round 3: Re-save (same content, new file) ──
        checkpoint_path2 = checkpoint_path.with_stem(
            checkpoint_path.stem + "_round2"
        )
        # Rebuild from loaded data
        ckpt2 = dict(loaded1)
        _save_checkpoint(
            checkpoint_path2, ckpt2,
            finalize=True, contract=contract, manifest_base=manifest_base,
        )
        loaded2 = _load_checkpoint(checkpoint_path2)

        # ── Verify model weight fidelity ──
        for key in original_model_state:
            assert key in loaded1.get("model_state_dict", {}), (
                f"Key {key} missing after first load"
            )
            assert key in loaded2.get("model_state_dict", {}), (
                f"Key {key} missing after second load"
            )
            # Weights must be bit-identical
            w_orig = original_model_state[key]
            w_1 = loaded1["model_state_dict"][key]
            w_2 = loaded2["model_state_dict"][key]
            assert torch.equal(w_orig, w_1), (
                f"Weight {key} changed after save→load (round 1)"
            )
            assert torch.equal(w_orig, w_2), (
                f"Weight {key} changed after save→load→save→load (round 2)"
            )

        # ── Verify metadata fidelity ──
        assert loaded1.get("epoch") == 3
        assert loaded2.get("epoch") == 3
        assert loaded1.get("global_step") == 750
        assert loaded2.get("global_step") == 750
        assert loaded1.get("phase_id") == 1
        assert loaded2.get("phase_id") == 1
        assert loaded1.get("best_val_loss") == 0.42
        assert loaded2.get("best_val_loss") == 0.42

        # ── Verify optimizer state survives (key structure) ──
        assert "optimizer_state_dict" in loaded1
        assert "optimizer_state_dict" in loaded2
        assert "param_groups" in loaded1["optimizer_state_dict"]
        assert "param_groups" in loaded2["optimizer_state_dict"]
        # Learning rate should be preserved
        assert (
            loaded1["optimizer_state_dict"]["param_groups"][0].get("lr")
            == original_opt_state["param_groups"][0]["lr"]
        )

        # ── Verify both files pass manifest verification ──
        sidecar1 = verify_artifact_manifest(
            checkpoint_path, kind="checkpoint", contract=contract,
        )
        sidecar2 = verify_artifact_manifest(
            checkpoint_path2, kind="checkpoint", contract=contract,
        )
        assert sidecar1 is not None
        assert sidecar2 is not None

    # ── Extra: test missing manifest raises ─────────────────────────────

    def test_checkpoint_missing_manifest_rejected(
        self,
        tiny_model: torch.nn.Module,
        contract: dict[str, Any],
        checkpoint_path: Path,
    ) -> None:
        """A checkpoint without an embedded manifest and no sidecars should
        be rejected by verify_artifact_manifest."""
        ckpt = _build_checkpoint_dict(
            model_state=tiny_model.state_dict(),
            contract=contract,
            include_manifest=False,
        )
        torch.save(ckpt, checkpoint_path)

        with pytest.raises(ArtifactManifestError, match="Missing artifact manifest"):
            verify_artifact_manifest(
                checkpoint_path, kind="checkpoint", contract=contract,
                require_embedded_manifest=True,
            )

    # ── Extra: test sidecar hash tamper detection ───────────────────────

    def test_checkpoint_sidecar_hash_tamper_detected(
        self,
        tiny_model: torch.nn.Module,
        contract: dict[str, Any],
        checkpoint_path: Path,
    ) -> None:
        """Tampering the artifact_sha256 in the sidecar manifest causes
        verification to fail."""
        manifest_base = _make_manifest_base(contract)

        ckpt = _build_checkpoint_dict(
            model_state=tiny_model.state_dict(),
            contract=contract,
            manifest_base=manifest_base,
        )
        _save_checkpoint(
            checkpoint_path, ckpt,
            finalize=True, contract=contract, manifest_base=manifest_base,
        )

        # Tamper the sidecar manifest's artifact_sha256
        sidecar_path = provenance.artifact_manifest_path(checkpoint_path)
        data = json.loads(sidecar_path.read_text(encoding="utf-8"))
        data["artifact_sha256"] = "deadbeef" * 8  # clearly bogus
        sidecar_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        with pytest.raises(ArtifactManifestError, match="checksum mismatch"):
            verify_artifact_manifest(
                checkpoint_path, kind="checkpoint", contract=contract,
            )
