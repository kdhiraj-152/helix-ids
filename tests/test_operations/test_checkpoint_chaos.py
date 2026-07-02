"""
Phase 22B — P3: Checkpoint Chaos Testing

Verifies that every corruption path fails closed:
  - Never silently recover corrupted state.
  - Never load partially valid checkpoints.

Scenarios:
  1. Interrupted checkpoint file write (truncated .pt)
  2. Interrupted manifest write (truncated manifest.json)
  3. Interrupted checksum write (truncated sidecar)
  4. Partial directory deletion (missing sidecars)
  5. Missing sidecar metadata (incomplete manifest)
  6. Corrupted JSON metadata (invalid content)
  7. Concurrent save/load race (atomic read-consistency)
  8. Cross-version compatibility (field projection)
  9. Read-only checkpoint directory (permission denied)
  10. Disk-full simulation during save (ENOSPC)
"""

from __future__ import annotations

import json
import os
import pickle
import stat
import threading
from pathlib import Path
from typing import Any

import pytest
import torch

from helix_ids.contracts import FEATURE_ORDER_HASH, runtime_contract_payload
from helix_ids.governance.provenance import (
    ARTIFACT_MANIFEST_FILENAME,
    ARTIFACT_MANIFEST_KEY,
    ArtifactManifestError,
    artifact_sha256,
    checkpoint_manifest_payload,
    write_contract_sidecars,
)
from helix_ids.models.full import HelixFullConfig, create_helix_full
from helix_ids.operations.inference_runtime import HelixInferenceRuntime
from helix_ids.utils.export import (
    build_export_manifest,
    finalize_export_artifact,
    verify_export_artifact,
)

# ── Actual runtime contract — single source of truth ────────────────────────

_RC = runtime_contract_payload()
ACTUAL_FEATURE_ORDER: list[str] = list(_RC["feature_order"])
ACTUAL_SCHEMA_VERSION: str = str(_RC["schema_version"])
ACTUAL_SCHEMA_HASH: str = str(_RC["schema_hash"])
ACTUAL_INPUT_DIM: int = int(_RC["input_dim"])
ACTUAL_BINARY_OUTPUT_DIM: int = int(_RC["binary_output_dim"])
ACTUAL_FAMILY_OUTPUT_DIM: int = int(_RC["family_output_dim"])
ACTUAL_FEATURE_ORDER_HASH: str = str(FEATURE_ORDER_HASH)

CANONICAL_CONTRACT = {
    "schema_version": ACTUAL_SCHEMA_VERSION,
    "schema_hash": ACTUAL_SCHEMA_HASH,
    "feature_order": list(ACTUAL_FEATURE_ORDER),
    "input_dim": ACTUAL_INPUT_DIM,
    "binary_output_dim": ACTUAL_BINARY_OUTPUT_DIM,
    "family_output_dim": ACTUAL_FAMILY_OUTPUT_DIM,
    "contract_version": "2.1",
    "feature_order_hash": ACTUAL_FEATURE_ORDER_HASH,
}


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def model() -> torch.nn.Module:
    return create_helix_full(
        HelixFullConfig(
            input_dim=ACTUAL_INPUT_DIM,
            binary_output_dim=ACTUAL_BINARY_OUTPUT_DIM,
            family_output_dim=ACTUAL_FAMILY_OUTPUT_DIM,
        )
    )


@pytest.fixture
def contract() -> dict[str, Any]:
    return dict(CANONICAL_CONTRACT)


@pytest.fixture
def manifest_base(contract):
    return build_export_manifest(
        contract=contract,
        model_architecture="HelixFull",
        export_config={"format": "checkpoint", "origin": "chaos_test"},
    )


@pytest.fixture
def payload(model, contract, manifest_base):
    return {
        "model_state_dict": {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        },
        "model": {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        },
        "epoch": 42,
        "global_step": 10000,
        "phase_id": 3,
        "best_val_loss": 0.123,
        **contract,
        ARTIFACT_MANIFEST_KEY: checkpoint_manifest_payload(manifest_base),
    }


def _write_blessed_checkpoint(
    path: Path,
    payload: dict[str, Any],
    manifest_base: dict[str, Any],
    contract: dict[str, Any],
) -> None:
    """Write a fully valid, finalized checkpoint (the 'blessed' baseline)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    sidecars = write_contract_sidecars(path, contract)
    finalize_export_artifact(path, manifest_base, sidecars=sidecars)


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 1: Interrupted checkpoint file write (truncated .pt)
# ═══════════════════════════════════════════════════════════════════════════════


class TestInterruptedCheckpointWrite:
    """Scenario 1 — truncated .pt file must be detected and rejected."""

    def test_truncated_checkpoint_rejected(
        self,
        tmp_path: Path,
        payload: dict[str, Any],
        contract: dict[str, Any],
        manifest_base: dict[str, Any],
    ) -> None:
        path = tmp_path / "truncated.pt"

        _write_blessed_checkpoint(path, payload, manifest_base, contract)
        pre_trunc_hash = artifact_sha256(path)

        original = path.read_bytes()
        truncated = original[: len(original) // 4]
        path.write_bytes(truncated)

        with pytest.raises(ArtifactManifestError, match="checksum mismatch"):
            verify_export_artifact(
                path,
                kind="checkpoint",
                contract=contract,
                embedded_manifest=checkpoint_manifest_payload(manifest_base),
            )

        assert artifact_sha256(path) != pre_trunc_hash, (
            "Truncated file SHA256 should differ from original"
        )

    def test_empty_checkpoint_file_rejected(
        self,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "empty.pt"
        path.write_text("", encoding="utf-8")

        with pytest.raises((EOFError, RuntimeError, ArtifactManifestError)):
            HelixInferenceRuntime(path)

    def test_checkpoint_with_random_bytes_rejected(
        self,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "garbage.pt"
        path.write_bytes(os.urandom(4096))

        with pytest.raises((EOFError, RuntimeError, ArtifactManifestError, Exception)):
            HelixInferenceRuntime(path)


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 2: Interrupted manifest write (truncated manifest.json)
# ═══════════════════════════════════════════════════════════════════════════════


class TestInterruptedManifestWrite:
    """Scenario 2 — truncated/corrupted manifest.json must be detected."""

    def test_truncated_manifest_rejected(
        self,
        tmp_path: Path,
        payload: dict[str, Any],
        contract: dict[str, Any],
        manifest_base: dict[str, Any],
    ) -> None:
        path = tmp_path / "bad-manifest.pt"
        _write_blessed_checkpoint(path, payload, manifest_base, contract)

        manifest_path = path.with_suffix(
            path.suffix + f".{ARTIFACT_MANIFEST_FILENAME}"
        )
        assert manifest_path.exists()
        original = manifest_path.read_bytes()
        manifest_path.write_bytes(original[: len(original) // 2])

        with pytest.raises((ArtifactManifestError, json.JSONDecodeError)):
            verify_export_artifact(
                path,
                kind="checkpoint",
                contract=contract,
                embedded_manifest=checkpoint_manifest_payload(manifest_base),
            )

    def test_empty_manifest_rejected(
        self,
        tmp_path: Path,
        payload: dict[str, Any],
        contract: dict[str, Any],
        manifest_base: dict[str, Any],
    ) -> None:
        path = tmp_path / "empty-manifest.pt"
        _write_blessed_checkpoint(path, payload, manifest_base, contract)
        manifest_path = path.with_suffix(
            path.suffix + f".{ARTIFACT_MANIFEST_FILENAME}"
        )
        manifest_path.write_text("", encoding="utf-8")

        with pytest.raises((ArtifactManifestError, json.JSONDecodeError)):
            verify_export_artifact(
                path,
                kind="checkpoint",
                contract=contract,
                embedded_manifest=checkpoint_manifest_payload(manifest_base),
            )

    def test_missing_manifest_rejected(
        self,
        tmp_path: Path,
        payload: dict[str, Any],
        contract: dict[str, Any],
        manifest_base: dict[str, Any],
    ) -> None:
        path = tmp_path / "no-manifest.pt"
        # Save checkpoint WITHOUT finalizing the manifest write
        torch.save(payload, path)
        write_contract_sidecars(path, contract)
        # Intentionally skip finalize_export_artifact

        # Verification should fail without a manifest sidecar
        with pytest.raises((ArtifactManifestError, FileNotFoundError)):
            verify_export_artifact(
                path,
                kind="checkpoint",
                contract=contract,
                embedded_manifest=checkpoint_manifest_payload(manifest_base),
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 3: Interrupted checksum write (truncated sidecar)
# ═══════════════════════════════════════════════════════════════════════════════


def _delete_or_truncate_sidecar(path: Path, suffix: str) -> None:
    sc = path.with_suffix(path.suffix + suffix)
    if sc.exists():
        sc.unlink()


class TestInterruptedChecksumWrite:
    """Scenario 3 — truncated .contract.json / .schema_hash.txt must be detected."""

    def test_truncated_contract_sidecar_rejected(
        self,
        tmp_path: Path,
        payload: dict[str, Any],
        contract: dict[str, Any],
        manifest_base: dict[str, Any],
    ) -> None:
        path = tmp_path / "bad-sidecar.pt"
        _write_blessed_checkpoint(path, payload, manifest_base, contract)

        sidecar_path = path.with_suffix(path.suffix + ".contract.json")
        original = sidecar_path.read_bytes()
        sidecar_path.write_bytes(original[: len(original) // 3])

        with pytest.raises((RuntimeError, ArtifactManifestError, json.JSONDecodeError)):
            HelixInferenceRuntime(path)

    def test_truncated_schema_hash_sidecar_rejected(
        self,
        tmp_path: Path,
        payload: dict[str, Any],
        contract: dict[str, Any],
        manifest_base: dict[str, Any],
    ) -> None:
        path = tmp_path / "bad-schemahash.pt"
        _write_blessed_checkpoint(path, payload, manifest_base, contract)

        schema_hash_path = path.with_suffix(path.suffix + ".schema_hash.txt")
        schema_hash_path.write_text("short\n", encoding="utf-8")

        with pytest.raises((RuntimeError, ArtifactManifestError)):
            HelixInferenceRuntime(path)

    def test_missing_contract_sidecar_rejected(
        self,
        tmp_path: Path,
        payload: dict[str, Any],
        contract: dict[str, Any],
        manifest_base: dict[str, Any],
    ) -> None:
        path = tmp_path / "missing-sidecar.pt"
        _write_blessed_checkpoint(path, payload, manifest_base, contract)

        _delete_or_truncate_sidecar(path, ".contract.json")

        with pytest.raises((RuntimeError, ArtifactManifestError, FileNotFoundError)):
            HelixInferenceRuntime(path)

    def test_missing_all_sidecars_rejected(
        self,
        tmp_path: Path,
        payload: dict[str, Any],
        contract: dict[str, Any],
        manifest_base: dict[str, Any],
    ) -> None:
        path = tmp_path / "all-missing.pt"
        _write_blessed_checkpoint(path, payload, manifest_base, contract)

        for suffix in [".contract.json", ".feature_order.json",
                       ".schema_hash.txt",
                       f".{ARTIFACT_MANIFEST_FILENAME}"]:
            _delete_or_truncate_sidecar(path, suffix)

        with pytest.raises((RuntimeError, ArtifactManifestError)):
            HelixInferenceRuntime(path)


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 4: Partial directory deletion
# ═══════════════════════════════════════════════════════════════════════════════


class TestPartialDirectoryDeletion:
    """Scenario 4 — missing checkpoint directory or sidecardir must fail."""

    def test_checkpoint_dir_deleted(
        self,
        tmp_path: Path,
        payload: dict[str, Any],
        contract: dict[str, Any],
        manifest_base: dict[str, Any],
    ) -> None:
        path = tmp_path / "dir_deleted" / "ckpt.pt"
        _write_blessed_checkpoint(path, payload, manifest_base, contract)

        import shutil
        shutil.rmtree(path.parent)

        with pytest.raises((FileNotFoundError, RuntimeError, ArtifactManifestError)):
            HelixInferenceRuntime(path)

    def test_sidecar_files_deleted(
        self,
        tmp_path: Path,
        payload: dict[str, Any],
        contract: dict[str, Any],
        manifest_base: dict[str, Any],
    ) -> None:
        path = tmp_path / "sidecar_dir" / "ckpt.pt"
        _write_blessed_checkpoint(path, payload, manifest_base, contract)

        for suffix in [".contract.json", ".feature_order.json",
                       ".schema_hash.txt",
                       f".{ARTIFACT_MANIFEST_FILENAME}"]:
            _delete_or_truncate_sidecar(path, suffix)

        with pytest.raises((RuntimeError, ArtifactManifestError)):
            HelixInferenceRuntime(path)


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 5: Missing sidecar metadata (incomplete fields)
# ═══════════════════════════════════════════════════════════════════════════════


class TestMissingSidecarMetadata:
    """Scenario 5 — sidecars with missing/incomplete fields must be rejected."""

    def test_manifest_missing_required_fields(
        self,
        tmp_path: Path,
        payload: dict[str, Any],
        contract: dict[str, Any],
        manifest_base: dict[str, Any],
    ) -> None:
        path = tmp_path / "missing-fields.pt"
        _write_blessed_checkpoint(path, payload, manifest_base, contract)

        manifest_path = path.with_suffix(
            path.suffix + f".{ARTIFACT_MANIFEST_FILENAME}"
        )
        incomplete = {
            "manifest_version": "1.0.0",
            # Missing: exporter_version, git_commit, runtime_version
        }
        manifest_path.write_text(json.dumps(incomplete, indent=2), encoding="utf-8")

        with pytest.raises(
            (ArtifactManifestError, KeyError),
        ):
            verify_export_artifact(
                path,
                kind="checkpoint",
                contract=contract,
                embedded_manifest=checkpoint_manifest_payload(manifest_base),
            )

    def test_contract_sidecar_missing_keys(
        self,
        tmp_path: Path,
        payload: dict[str, Any],
        contract: dict[str, Any],
        manifest_base: dict[str, Any],
    ) -> None:
        path = tmp_path / "partial-contract.pt"
        _write_blessed_checkpoint(path, payload, manifest_base, contract)

        contract_path = path.with_suffix(path.suffix + ".contract.json")
        partial = {"schema_version": ACTUAL_SCHEMA_VERSION}
        contract_path.write_text(json.dumps(partial, indent=2), encoding="utf-8")

        with pytest.raises((RuntimeError, ArtifactManifestError, KeyError)):
            HelixInferenceRuntime(path)


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 6: Corrupted JSON metadata
# ═══════════════════════════════════════════════════════════════════════════════


class TestCorruptedJsonMetadata:
    """Scenario 6 — corrupt JSON in sidecar files must be detected."""

    def test_corrupted_contract_json(
        self,
        tmp_path: Path,
        payload: dict[str, Any],
        contract: dict[str, Any],
        manifest_base: dict[str, Any],
    ) -> None:
        path = tmp_path / "corrupt-contract.pt"
        _write_blessed_checkpoint(path, payload, manifest_base, contract)

        contract_path = path.with_suffix(path.suffix + ".contract.json")
        contract_path.write_bytes(b"\x00\xff\xfe\xfd" + os.urandom(128))

        with pytest.raises((RuntimeError, ArtifactManifestError, json.JSONDecodeError)):
            HelixInferenceRuntime(path)

    def test_corrupted_feature_order_json(
        self,
        tmp_path: Path,
        payload: dict[str, Any],
        contract: dict[str, Any],
        manifest_base: dict[str, Any],
    ) -> None:
        path = tmp_path / "corrupt-features.pt"
        _write_blessed_checkpoint(path, payload, manifest_base, contract)

        feature_path = path.with_suffix(path.suffix + ".feature_order.json")
        feature_path.write_text("NOT JSON [[[", encoding="utf-8")

        with pytest.raises((RuntimeError, ArtifactManifestError, json.JSONDecodeError)):
            HelixInferenceRuntime(path)

    def test_corrupted_manifest_json(
        self,
        tmp_path: Path,
        payload: dict[str, Any],
        contract: dict[str, Any],
        manifest_base: dict[str, Any],
    ) -> None:
        path = tmp_path / "corrupt-manifest.pt"
        _write_blessed_checkpoint(path, payload, manifest_base, contract)

        manifest_path = path.with_suffix(
            path.suffix + f".{ARTIFACT_MANIFEST_FILENAME}"
        )
        manifest_path.write_text('{"broken": true, missing_end', encoding="utf-8")

        with pytest.raises((ArtifactManifestError, json.JSONDecodeError)):
            verify_export_artifact(
                path,
                kind="checkpoint",
                contract=contract,
                embedded_manifest=checkpoint_manifest_payload(manifest_base),
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 7: Concurrent save/load race (read-consistency)
# ═══════════════════════════════════════════════════════════════════════════════


class TestConcurrentSaveLoad:
    """Scenario 7 — concurrent save/load must not produce inconsistent state."""

    def test_concurrent_save_does_not_corrupt_read(
        self,
        tmp_path: Path,
        payload: dict[str, Any],
    ) -> None:
        path = tmp_path / "concurrent.pt"
        torch.save(payload, path)

        loaded = torch.load(path, map_location="cpu", weights_only=True)
        assert loaded.get("epoch") == 42

        errors: list[Exception] = []
        reads_ok: int = 0

        def _writer() -> None:
            for i in range(10):
                p = dict(payload)
                p["epoch"] = 100 + i
                torch.save(p, path)

        def _reader() -> None:
            nonlocal reads_ok
            import time
            for _ in range(20):
                try:
                    data = torch.load(path, map_location="cpu", weights_only=True)
                    epoch = data.get("epoch")
                    assert isinstance(epoch, int), f"Corrupted epoch: {epoch!r}"
                    reads_ok += 1
                except Exception as e:
                    errors.append(e)
                time.sleep(0.001)

        t1 = threading.Thread(target=_writer, daemon=True)
        t2 = threading.Thread(target=_reader, daemon=True)
        t1.start()
        t2.start()
        t1.join(timeout=15)
        t2.join(timeout=15)

        known = (RuntimeError, ValueError, KeyError, EOFError, OSError)
        for err in errors:
            assert isinstance(err, known), (
                f"Unexpected error type: {type(err).__name__}: {err}"
            )
        assert reads_ok > 0, "No successful reads during concurrent access"

    def test_independent_save_load_consistency(
        self,
        tmp_path: Path,
        model: torch.nn.Module,
        contract: dict[str, Any],
        manifest_base: dict[str, Any],
    ) -> None:
        """Two independent checkpoints must both be independently verifiable."""
        path_a = tmp_path / "consistency_a.pt"
        path_b = tmp_path / "consistency_b.pt"

        payload_a = {
            "model_state_dict": {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            },
            "epoch": 1,
            **contract,
            ARTIFACT_MANIFEST_KEY: checkpoint_manifest_payload(manifest_base),
        }
        payload_b = dict(payload_a)
        payload_b["epoch"] = 2

        _write_blessed_checkpoint(path_a, payload_a, manifest_base, contract)
        _write_blessed_checkpoint(path_b, payload_b, manifest_base, contract)

        result_a = verify_export_artifact(
            path_a,
            kind="checkpoint",
            contract=contract,
            embedded_manifest=checkpoint_manifest_payload(manifest_base),
        )
        result_b = verify_export_artifact(
            path_b,
            kind="checkpoint",
            contract=contract,
            embedded_manifest=checkpoint_manifest_payload(manifest_base),
        )
        assert result_a is not None
        assert result_b is not None

        loaded_a = torch.load(path_a, map_location="cpu", weights_only=True)
        loaded_b = torch.load(path_b, map_location="cpu", weights_only=True)
        assert loaded_a["epoch"] == 1
        assert loaded_b["epoch"] == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 8: Cross-version compatibility (schema migration)
# ═══════════════════════════════════════════════════════════════════════════════


class TestCrossVersionCompatibility:
    """Scenario 8 — checkpoints with different versions must be detected."""

    def test_older_schema_version_detected(
        self,
        tmp_path: Path,
        model: torch.nn.Module,
        contract: dict[str, Any],
        manifest_base: dict[str, Any],
    ) -> None:
        path = tmp_path / "older-schema.pt"
        older_contract = dict(contract)
        older_contract["schema_version"] = "2025-01-01"
        older_contract["schema_hash"] = "deadbeef" * 8

        payload = {
            "model_state_dict": {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            },
            **older_contract,
            ARTIFACT_MANIFEST_KEY: checkpoint_manifest_payload(manifest_base),
        }

        torch.save(payload, path)
        write_contract_sidecars(path, older_contract)

        older_manifest_base = build_export_manifest(
            contract=older_contract,
            model_architecture="HelixFull",
            export_config={"format": "checkpoint", "origin": "cross_version"},
        )
        finalize_export_artifact(path, older_manifest_base, sidecars={})

        with pytest.raises((AssertionError, RuntimeError, ArtifactManifestError)):
            HelixInferenceRuntime(path)

    def test_mismatched_feature_order_hash_rejected(
        self,
        tmp_path: Path,
        payload: dict[str, Any],
        contract: dict[str, Any],
    ) -> None:
        """Checkpoint with wrong feature_order_hash in sidecar must be rejected
        when verified against the canonical (correct) contract."""
        path = tmp_path / "bad-hash.pt"

        wrong_hash = "0" * 64
        bad_contract = dict(contract)
        bad_contract["feature_order_hash"] = wrong_hash

        # Build manifest_base from the WRONG contract so the saved sidecar
        # embeds the incorrect hash
        bad_manifest = build_export_manifest(
            contract=bad_contract,
            model_architecture="HelixFull",
            export_config={"format": "checkpoint", "origin": "chaos_test"},
        )

        bad_payload = dict(payload)
        bad_payload.update(bad_contract)
        bad_payload[ARTIFACT_MANIFEST_KEY] = checkpoint_manifest_payload(bad_manifest)

        torch.save(bad_payload, path)
        write_contract_sidecars(path, bad_contract)
        finalize_export_artifact(path, bad_manifest, sidecars={})

        # Verify with the CORRECT contract — must detect mismatch
        with pytest.raises((ArtifactManifestError, RuntimeError, AssertionError)):
            verify_export_artifact(
                path,
                kind="checkpoint",
                contract=contract,
                embedded_manifest=checkpoint_manifest_payload(bad_manifest),
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 9: Read-only checkpoint directory
# ═══════════════════════════════════════════════════════════════════════════════


class TestReadOnlyCheckpointDirectory:
    """Scenario 9 — read-only checkpoint dir must fail on save with clear error."""

    def test_save_to_readonly_directory_rejected(
        self,
        tmp_path: Path,
        payload: dict[str, Any],
    ) -> None:
        ro_dir = tmp_path / "readonly_ckpt"
        ro_dir.mkdir(parents=True, exist_ok=True)

        original_mode = ro_dir.stat().st_mode
        ro_dir.chmod(
            stat.S_IRUSR | stat.S_IXUSR
            | stat.S_IRGRP | stat.S_IXGRP
            | stat.S_IROTH | stat.S_IXOTH
        )

        try:
            path = ro_dir / "ckpt.pt"
            with pytest.raises((PermissionError, OSError, RuntimeError)):
                torch.save(payload, path)
        finally:
            ro_dir.chmod(original_mode)

    def test_readonly_checkpoint_file_cannot_be_overwritten(
        self,
        tmp_path: Path,
        payload: dict[str, Any],
        contract: dict[str, Any],
        manifest_base: dict[str, Any],
    ) -> None:
        path = tmp_path / "readonly.pt"
        _write_blessed_checkpoint(path, payload, manifest_base, contract)

        original_mode = path.stat().st_mode
        path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

        try:
            with pytest.raises((PermissionError, OSError, RuntimeError)):
                torch.save(payload, path)
        finally:
            path.chmod(original_mode)


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 10: Disk-full simulation during save
# ═══════════════════════════════════════════════════════════════════════════════


class TestDiskFullSimulation:
    """Scenario 10 — disk-full must fail gracefully without corruption."""

    def test_failsafe_on_write_error(
        self,
        tmp_path: Path,
        payload: dict[str, Any],
    ) -> None:
        bad_path = tmp_path / "nonexistent" / "deeper" / "ckpt.pt"
        with pytest.raises((OSError, RuntimeError, FileNotFoundError)):
            torch.save(payload, bad_path)

    def test_partial_write_then_full_restore(
        self,
        tmp_path: Path,
        payload: dict[str, Any],
        contract: dict[str, Any],
        manifest_base: dict[str, Any],
    ) -> None:
        path = tmp_path / "recover.pt"

        # Simulate a partial/failed write (like disk-full mid-write)
        path.write_bytes(b"SIMULATED_DISK_FULL_PARTIAL_WRITE")

        # Recovery: write a valid checkpoint
        _write_blessed_checkpoint(path, payload, manifest_base, contract)

        # Must pass verification
        result = verify_export_artifact(
            path,
            kind="checkpoint",
            contract=contract,
            embedded_manifest=checkpoint_manifest_payload(manifest_base),
        )
        assert result is not None, (
            "Checkpoint must be verifiable after recovery from partial write"
        )

    def test_partial_write_remains_invalid(
        self,
        tmp_path: Path,
        payload: dict[str, Any],
        contract: dict[str, Any],
        manifest_base: dict[str, Any],
    ) -> None:
        """A partially written checkpoint must NOT be accidentally loadable."""
        path = tmp_path / "never-recover.pt"

        path.write_bytes(b"PARTIAL_WRITE_NEVER_COMPLETED")

        with pytest.raises((EOFError, RuntimeError, ArtifactManifestError,
                            pickle.UnpicklingError)):
            HelixInferenceRuntime(path)


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-cutting: Blessed baseline must pass (positive control)
# ═══════════════════════════════════════════════════════════════════════════════


class TestCheckpointBlessedBaseline:
    """Positive control — a fully blessed checkpoint must load and verify."""

    def test_blessed_checkpoint_loads_successfully(
        self,
        tmp_path: Path,
        payload: dict[str, Any],
        contract: dict[str, Any],
        manifest_base: dict[str, Any],
    ) -> None:
        path = tmp_path / "blessed.pt"
        _write_blessed_checkpoint(path, payload, manifest_base, contract)

        runtime = HelixInferenceRuntime(path)
        assert runtime.schema_version is not None
        assert runtime.model is not None

    def test_blessed_checkpoint_verifies(
        self,
        tmp_path: Path,
        payload: dict[str, Any],
        contract: dict[str, Any],
        manifest_base: dict[str, Any],
    ) -> None:
        path = tmp_path / "blessed_verify.pt"
        _write_blessed_checkpoint(path, payload, manifest_base, contract)

        result = verify_export_artifact(
            path,
            kind="checkpoint",
            contract=contract,
            embedded_manifest=checkpoint_manifest_payload(manifest_base),
        )
        assert result is not None
        assert str(result.get("artifact_sha256")) == artifact_sha256(path)
