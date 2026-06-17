"""Tests for auto-restart recovery system.

Covers:
  - crash detection via sentinel file
  - checkpoint discovery and ordering
  - checkpoint corruption detection
  - no-crash / fresh-start scenario
  - missing-checkpoint scenario
  - governance verification
  - recovery lock semantics
  - full resolve_restart lifecycle
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

import pytest
import torch

from helix_ids.governance.provenance import ArtifactManifestError
from helix_ids.operations.recovery.restart_manager import (
    CrashedState,
    RestartDecision,
    RestartManager,
    RestartOutcome,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def ckpt_dir(tmp_path: Path) -> Path:
    """Return a temporary checkpoint directory."""
    d = tmp_path / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def sentinel_dir(tmp_path: Path) -> Path:
    """Return a separate temporary sentinel directory."""
    d = tmp_path / "sentinels"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def restart_manager(ckpt_dir: Path) -> RestartManager:
    """Return a RestartManager with governance verification OFF (simpler tests)."""
    return RestartManager(
        checkpoint_dir=ckpt_dir,
        require_governance=False,
    )


@pytest.fixture
def restart_manager_with_gov(ckpt_dir: Path) -> RestartManager:
    """Return a RestartManager with governance verification ON."""
    return RestartManager(
        checkpoint_dir=ckpt_dir,
        require_governance=True,
    )


def _make_checkpoint(
    path: Path,
    epoch: int,
    step: int,
    phase_id: int = 1,
    best_val_loss: float = float("inf"),
) -> Path:
    """Create a minimal synthetic checkpoint file at *path*."""
    ckpt_path = path / f"checkpoint_epoch_{epoch}_step_{step}.pt"
    torch.save(
        {
            "model_state_dict": {"weight": torch.randn(4, 4)},
            "optimizer_state_dict": {"param_groups": [], "state": {}},
            "epoch": epoch,
            "global_step": step,
            "phase_id": phase_id,
            "best_val_loss": best_val_loss,
        },
        ckpt_path,
    )
    return ckpt_path


def _write_sentinel(
    sentinel_dir: Path,
    *,
    reason: str = "training_crashed",
    run_id: str | None = None,
) -> Path:
    """Write a crash sentinel file."""
    sentinel = sentinel_dir / "_crash_sentinel"
    payload: dict[str, str] = {
        "timestamp": "2026-06-16T12:00:00Z",
        "reason": reason,
    }
    if run_id:
        payload["run_id"] = run_id
    sentinel.write_text(json.dumps(payload), encoding="utf-8")
    return sentinel


# ── Crash Detection ──────────────────────────────────────────────────────────


class TestCrashDetection:
    def test_no_sentinel_no_crash(self, restart_manager: RestartManager) -> None:
        crash = restart_manager._detect_crash()
        assert crash is None, "No sentinel file should mean no crash"

    def test_sentinel_detects_crash(self, ckpt_dir: Path) -> None:
        _write_sentinel(ckpt_dir)
        mgr = RestartManager(checkpoint_dir=ckpt_dir, require_governance=False)
        crash = mgr._detect_crash()
        assert crash is not None
        assert crash.reason == "training_crashed"
        assert "2026-06-16" in crash.detected_at

    def test_sentinel_with_run_id(self, ckpt_dir: Path) -> None:
        _write_sentinel(ckpt_dir, run_id="exp_42")
        mgr = RestartManager(checkpoint_dir=ckpt_dir, require_governance=False)
        crash = mgr._detect_crash()
        assert crash is not None
        assert crash.run_id == "exp_42"

    def test_corrupt_sentinel(self, ckpt_dir: Path) -> None:
        """Corrupt sentinel should still be detected (graceful fallback)."""
        sentinel = ckpt_dir / "_crash_sentinel"
        sentinel.write_text("not valid json", encoding="utf-8")
        mgr = RestartManager(checkpoint_dir=ckpt_dir, require_governance=False)
        crash = mgr._detect_crash()
        assert crash is not None
        assert "unknown" in crash.reason

    def test_oserror_sentinel_fallback(self, ckpt_dir: Path) -> None:
        """OSError during sentinel read should fallback to empty payload."""
        sentinel = ckpt_dir / "_crash_sentinel"
        sentinel.write_text("{}", encoding="utf-8")
        # Make the file unreadable
        sentinel.chmod(0o000)
        mgr = RestartManager(checkpoint_dir=ckpt_dir, require_governance=False)
        try:
            crash = mgr._detect_crash()
            # Fallback should still produce a CrashedState
            assert crash is not None
            assert crash.reason == "unknown"
        finally:
            sentinel.chmod(0o644)

    def test_write_and_clear_sentinel(self, ckpt_dir: Path) -> None:
        mgr = RestartManager(checkpoint_dir=ckpt_dir, require_governance=False)

        sentinel = mgr.write_crash_sentinel(reason="OOM")
        assert sentinel.exists()

        mgr.clear_crash_sentinel()
        assert not sentinel.exists()


# ── Checkpoint Discovery ─────────────────────────────────────────────────────


class TestCheckpointDiscovery:
    def test_no_checkpoints(self, restart_manager: RestartManager) -> None:
        candidates = restart_manager._discover_checkpoints()
        assert candidates == []

    def test_discover_single(self, ckpt_dir: Path, restart_manager: RestartManager) -> None:
        _make_checkpoint(ckpt_dir, epoch=5, step=1000)
        candidates = restart_manager._discover_checkpoints()
        assert len(candidates) == 1
        assert "epoch_5" in candidates[0].name

    def test_discover_multiple_sorted_descending(
        self, ckpt_dir: Path, restart_manager: RestartManager
    ) -> None:
        _make_checkpoint(ckpt_dir, epoch=1, step=200)
        _make_checkpoint(ckpt_dir, epoch=3, step=600)
        _make_checkpoint(ckpt_dir, epoch=2, step=400)
        _make_checkpoint(ckpt_dir, epoch=3, step=700)

        candidates = restart_manager._discover_checkpoints()
        assert len(candidates) == 4
        # Newest first: epoch 3 step 700, epoch 3 step 600, epoch 2 step 400, epoch 1 step 200
        names = [p.name for p in candidates]
        assert names[0] == "checkpoint_epoch_3_step_700.pt"
        assert names[1] == "checkpoint_epoch_3_step_600.pt"
        assert names[2] == "checkpoint_epoch_2_step_400.pt"
        assert names[3] == "checkpoint_epoch_1_step_200.pt"

    def test_filters_non_checkpoint_files(
        self, ckpt_dir: Path, restart_manager: RestartManager
    ) -> None:
        (ckpt_dir / "other_file.txt").write_text("hello", encoding="utf-8")
        (ckpt_dir / "model_best.pt").write_text("not a real ckpt", encoding="utf-8")
        _make_checkpoint(ckpt_dir, epoch=2, step=400)

        candidates = restart_manager._discover_checkpoints()
        assert len(candidates) == 1


# ── Checkpoint Selection ──────────────────────────────────────────────────────


class TestCheckpointSelection:
    def test_selects_newest_valid(
        self, ckpt_dir: Path, restart_manager: RestartManager
    ) -> None:
        _make_checkpoint(ckpt_dir, epoch=1, step=200)
        best_path = _make_checkpoint(ckpt_dir, epoch=3, step=700)

        candidates = restart_manager._discover_checkpoints()
        best = restart_manager._select_best_checkpoint(candidates)
        assert best is not None
        assert best.name == best_path.name

    def test_empty_list_returns_none(
        self, restart_manager: RestartManager
    ) -> None:
        best = restart_manager._select_best_checkpoint([])
        assert best is None

    def test_all_candidates_invalid_returns_none(
        self, ckpt_dir: Path
    ) -> None:
        """When all checkpoints are corrupt (<100 bytes), select_best returns None."""
        mgr = RestartManager(checkpoint_dir=ckpt_dir, require_governance=False)

        # Two tiny (invalid) checkpoints
        bad1 = ckpt_dir / "checkpoint_epoch_5_step_1000.pt"
        bad1.write_text("x" * 50, encoding="utf-8")
        bad2 = ckpt_dir / "checkpoint_epoch_3_step_600.pt"
        bad2.write_text("y" * 50, encoding="utf-8")

        candidates = mgr._discover_checkpoints()
        best = mgr._select_best_checkpoint(candidates)
        assert best is None

    def test_corrupt_file_skipped(self, ckpt_dir: Path) -> None:
        """A tiny (<100 byte) file should be considered invalid."""
        mgr = RestartManager(checkpoint_dir=ckpt_dir, require_governance=False)

        bad_path = ckpt_dir / "checkpoint_epoch_5_step_1000.pt"
        bad_path.write_text("not a real checkpoint", encoding="utf-8")

        good_path = _make_checkpoint(ckpt_dir, epoch=3, step=600)

        candidates = mgr._discover_checkpoints()
        best = mgr._select_best_checkpoint(candidates)
        assert best is not None
        # Should skip the corrupt one and pick the valid one
        assert best.name == good_path.name

    def test_corrupt_file_then_none_left(
        self, ckpt_dir: Path
    ) -> None:
        """Only corrupt files → select_best returns None."""
        mgr = RestartManager(checkpoint_dir=ckpt_dir, require_governance=False)
        bad = ckpt_dir / "checkpoint_epoch_1_step_100.pt"
        bad.write_text("x" * 50, encoding="utf-8")

        candidates = mgr._discover_checkpoints()
        best = mgr._select_best_checkpoint(candidates)
        assert best is None


# ── Recovery Lock ─────────────────────────────────────────────────────────────


class TestRecoveryLock:
    def test_acquire_lock(self, ckpt_dir: Path) -> None:
        mgr = RestartManager(checkpoint_dir=ckpt_dir, require_governance=False)
        assert mgr.acquire_recovery_lock() is True

    def test_lock_blocks_concurrent(self, ckpt_dir: Path) -> None:
        mgr = RestartManager(checkpoint_dir=ckpt_dir, require_governance=False)
        mgr.acquire_recovery_lock(timeout_seconds=30)
        # Second acquire within timeout should fail
        assert mgr.acquire_recovery_lock(timeout_seconds=30) is False

    def test_lock_timeout_expired(self, ckpt_dir: Path) -> None:
        mgr = RestartManager(checkpoint_dir=ckpt_dir, require_governance=False)
        mgr.acquire_recovery_lock(timeout_seconds=0)
        import time
        time.sleep(0.01)  # Ensure timeout expired
        assert mgr.acquire_recovery_lock(timeout_seconds=0) is True

    def test_release_lock(self, ckpt_dir: Path) -> None:
        mgr = RestartManager(checkpoint_dir=ckpt_dir, require_governance=False)
        mgr.acquire_recovery_lock()
        mgr.release_recovery_lock()
        assert mgr.acquire_recovery_lock() is True

    def test_release_lock_when_none_exists(self, ckpt_dir: Path) -> None:
        """release_recovery_lock should not raise when no lock file exists."""
        mgr = RestartManager(checkpoint_dir=ckpt_dir, require_governance=False)
        # No lock acquired → release must not raise
        mgr.release_recovery_lock()  # must not raise


# ── Full resolve_restart ──────────────────────────────────────────────────────


class TestResolveRestart:
    def test_fresh_start_no_crash(
        self, restart_manager: RestartManager
    ) -> None:
        decision = restart_manager.resolve_restart()
        assert decision.should_restart is False
        assert "no crash" in decision.reason

    def test_crash_no_checkpoint(
        self, ckpt_dir: Path
    ) -> None:
        _write_sentinel(ckpt_dir)
        mgr = RestartManager(checkpoint_dir=ckpt_dir, require_governance=False)
        decision = mgr.resolve_restart()
        assert decision.should_restart is False
        assert "no checkpoints found" in decision.reason

    def test_crash_with_valid_checkpoint(
        self, ckpt_dir: Path
    ) -> None:
        _write_sentinel(ckpt_dir)
        _make_checkpoint(ckpt_dir, epoch=5, step=1000, phase_id=2, best_val_loss=1.5)
        mgr = RestartManager(checkpoint_dir=ckpt_dir, require_governance=False)

        decision = mgr.resolve_restart()
        assert decision.should_restart is True
        assert decision.resume_epoch == 5
        assert decision.resume_step == 1000
        assert decision.resume_phase_id == 2
        assert decision.resume_best_val_loss == 1.5
        assert decision.checkpoint is not None
        assert decision.checkpoint_path is not None
        assert decision.crash is not None

    def test_crash_picks_latest_checkpoint(
        self, ckpt_dir: Path
    ) -> None:
        _write_sentinel(ckpt_dir)
        _make_checkpoint(ckpt_dir, epoch=2, step=400, phase_id=1)
        _make_checkpoint(ckpt_dir, epoch=5, step=1200, phase_id=3)  # latest
        _make_checkpoint(ckpt_dir, epoch=3, step=700, phase_id=2)
        mgr = RestartManager(checkpoint_dir=ckpt_dir, require_governance=False)

        decision = mgr.resolve_restart()
        assert decision.should_restart is True
        assert decision.resume_epoch == 5
        assert decision.resume_phase_id == 3

    def test_crash_skips_corrupt_picks_earlier(
        self, ckpt_dir: Path
    ) -> None:
        _write_sentinel(ckpt_dir)
        # Latest but corrupt (tiny file)
        bad_path = ckpt_dir / "checkpoint_epoch_5_step_1000.pt"
        bad_path.write_text("corrupt", encoding="utf-8")
        # Earlier but valid
        _make_checkpoint(ckpt_dir, epoch=3, step=700, phase_id=2)

        mgr = RestartManager(checkpoint_dir=ckpt_dir, require_governance=False)
        decision = mgr.resolve_restart()
        assert decision.should_restart is True
        assert decision.resume_epoch == 3

    def test_crash_all_checkpoints_corrupt_returns_none(
        self, ckpt_dir: Path
    ) -> None:
        """All checkpoints are corrupt → should_restart=False with reason about no valid checkpoint."""
        _write_sentinel(ckpt_dir)
        bad1 = ckpt_dir / "checkpoint_epoch_5_step_1000.pt"
        bad1.write_text("x" * 50, encoding="utf-8")
        bad2 = ckpt_dir / "checkpoint_epoch_3_step_600.pt"
        bad2.write_text("y" * 50, encoding="utf-8")
        mgr = RestartManager(checkpoint_dir=ckpt_dir, require_governance=False)

        decision = mgr.resolve_restart()
        assert decision.should_restart is False
        assert "no valid (uncorrupted) checkpoint found" in decision.reason
        assert decision.crash is not None

    def test_crash_load_checkpoint_raises(
        self, ckpt_dir: Path
    ) -> None:
        """When torch.load raises on the best checkpoint, fallback decision is returned."""
        _write_sentinel(ckpt_dir)
        _make_checkpoint(ckpt_dir, epoch=5, step=1000)
        mgr = RestartManager(checkpoint_dir=ckpt_dir, require_governance=False)

        # Patch _load_checkpoint_metadata to raise
        original_load = mgr._load_checkpoint_metadata

        def failing_load(path):
            raise RuntimeError("corrupted tensor data")

        mgr._load_checkpoint_metadata = failing_load

        decision = mgr.resolve_restart()
        assert decision.should_restart is False
        assert "failed to load checkpoint" in decision.reason
        assert decision.crash is not None

    def test_clear_sentinel_after_restart(
        self, ckpt_dir: Path
    ) -> None:
        """After a successful restart, the sentinel should be cleared."""
        _write_sentinel(ckpt_dir)
        _make_checkpoint(ckpt_dir, epoch=5, step=1000)
        mgr = RestartManager(checkpoint_dir=ckpt_dir, require_governance=False)

        decision = mgr.resolve_restart()
        assert decision.should_restart is True

        # Simulate post-restart cleanup
        mgr.clear_crash_sentinel()
        crash = mgr._detect_crash()
        assert crash is None

    def test_restart_clears_sentinel_next_run(
        self, ckpt_dir: Path
    ) -> None:
        """After clearing sentinel, resolve_restart returns fresh start."""
        _write_sentinel(ckpt_dir)
        _make_checkpoint(ckpt_dir, epoch=3, step=600)
        mgr = RestartManager(checkpoint_dir=ckpt_dir, require_governance=False)

        # First call: detects crash
        d1 = mgr.resolve_restart()
        assert d1.should_restart is True

        # Clear (as if restart ran successfully)
        mgr.clear_crash_sentinel()

        # Second call: no crash, fresh start
        d2 = mgr.resolve_restart()
        assert d2.should_restart is False
        assert "no crash" in d2.reason

    def test_crash_separate_sentinel_dir(
        self, ckpt_dir: Path, sentinel_dir: Path
    ) -> None:
        """Crash sentinel in a separate directory from checkpoints."""
        _write_sentinel(sentinel_dir)
        _make_checkpoint(ckpt_dir, epoch=5, step=1000)
        mgr = RestartManager(
            checkpoint_dir=ckpt_dir,
            crash_sentinel_dir=sentinel_dir,
            require_governance=False,
        )

        decision = mgr.resolve_restart()
        assert decision.should_restart is True
        assert decision.resume_epoch == 5
        assert decision.crash is not None

    def test_crash_separate_sentinel_dir_no_ckpt(
        self, ckpt_dir: Path, sentinel_dir: Path
    ) -> None:
        """Sentinel in separate dir, no checkpoints in ckpt_dir."""
        _write_sentinel(sentinel_dir)
        mgr = RestartManager(
            checkpoint_dir=ckpt_dir,
            crash_sentinel_dir=sentinel_dir,
            require_governance=False,
        )

        decision = mgr.resolve_restart()
        assert decision.should_restart is False
        assert "no checkpoints found" in decision.reason
        assert decision.crash is not None

    def test_checkpoint_with_missing_keys(
        self, ckpt_dir: Path
    ) -> None:
        """Checkpoint missing 'epoch'/'global_step' keys should use defaults."""
        _write_sentinel(ckpt_dir)
        ckpt_path = ckpt_dir / "checkpoint_epoch_5_step_1000.pt"
        torch.save(
            {
                "model_state_dict": {"weight": torch.randn(4, 4)},
                # Deliberately omit 'epoch', 'global_step', 'phase_id', 'best_val_loss'
            },
            ckpt_path,
        )
        mgr = RestartManager(checkpoint_dir=ckpt_dir, require_governance=False)

        decision = mgr.resolve_restart()
        assert decision.should_restart is True
        assert decision.resume_epoch == 0  # default
        assert decision.resume_step == 0  # default
        assert decision.resume_phase_id == 1  # default
        assert decision.resume_best_val_loss == float("inf")  # default

    def test_resolve_restart_updates_run_id(
        self, ckpt_dir: Path
    ) -> None:
        """_detect_run_id picks up run_id file and crash sentinel uses it."""
        # Write a run_id file
        run_id_file = ckpt_dir / "_run_id.txt"
        run_id_file.write_text("my_run_007\n", encoding="utf-8")

        # Write sentinel WITHOUT run_id (so it falls back to _detect_run_id)
        sentinel = ckpt_dir / "_crash_sentinel"
        sentinel.write_text(
            json.dumps({"timestamp": "2026-06-16T12:00:00Z", "reason": "OOM"}),
            encoding="utf-8",
        )

        mgr = RestartManager(checkpoint_dir=ckpt_dir, require_governance=False)
        decision = mgr.resolve_restart()
        assert decision.crash is not None
        assert decision.crash.run_id == "my_run_007"

    def test_run_id_file_missing_defaults_to_unknown(
        self, ckpt_dir: Path
    ) -> None:
        """When no _run_id.txt exists, _detect_run_id returns 'unknown'."""
        mgr = RestartManager(checkpoint_dir=ckpt_dir, require_governance=False)
        run_id = mgr._detect_run_id()
        assert run_id == "unknown"


# ── RestartDecision Metadata ──────────────────────────────────────────────────


class TestRestartDecision:
    def test_metadata_fresh_start(self) -> None:
        d = RestartDecision(should_restart=False, reason="fresh start")
        meta = d.metadata
        assert meta["should_restart"] is False
        assert meta["reason"] == "fresh start"

    def test_metadata_with_checkpoint(self) -> None:
        d = RestartDecision(
            should_restart=True,
            checkpoint_path=Path("/tmp/ckpt.pt"),
            resume_epoch=10,
            resume_step=2000,
            resume_phase_id=3,
            reason="restarting",
        )
        meta = d.metadata
        assert meta["resume_epoch"] == 10
        assert meta["resume_step"] == 2000
        assert meta["resume_phase_id"] == 3

    def test_metadata_checkpoint_path_none(self) -> None:
        """When checkpoint_path is None, metadata returns None for that field."""
        d = RestartDecision(should_restart=False, reason="no checkpoint")
        meta = d.metadata
        assert meta["checkpoint_path"] is None


# ── Sentinel Operations ───────────────────────────────────────────────────────


class TestSentinelOperations:
    def test_write_sentinel_creates_file(self, ckpt_dir: Path) -> None:
        mgr = RestartManager(checkpoint_dir=ckpt_dir, require_governance=False)
        sentinel = mgr.write_crash_sentinel(reason="OOM crash")
        assert sentinel.exists()
        payload = json.loads(sentinel.read_text(encoding="utf-8"))
        assert payload["reason"] == "OOM crash"
        assert "timestamp" in payload

    def test_write_then_clear(self, ckpt_dir: Path) -> None:
        mgr = RestartManager(checkpoint_dir=ckpt_dir, require_governance=False)
        mgr.write_crash_sentinel()
        assert mgr._detect_crash() is not None
        mgr.clear_crash_sentinel()
        assert mgr._detect_crash() is None

    def test_clear_when_no_sentinel(self, ckpt_dir: Path) -> None:
        """clear_crash_sentinel should not raise when no sentinel exists."""
        mgr = RestartManager(checkpoint_dir=ckpt_dir, require_governance=False)
        mgr.clear_crash_sentinel()  # must not raise

    def test_write_sentinel_in_separate_dir(self, ckpt_dir: Path, sentinel_dir: Path) -> None:
        """Write sentinel in a separate directory from checkpoints."""
        mgr = RestartManager(
            checkpoint_dir=ckpt_dir,
            crash_sentinel_dir=sentinel_dir,
            require_governance=False,
        )
        sentinel = mgr.write_crash_sentinel(reason="test_separate")
        assert sentinel.exists()
        assert sentinel.parent == sentinel_dir
        # No sentinel in checkpoint_dir
        assert not (ckpt_dir / "_crash_sentinel").exists()

    def test_write_sentinel_default_reason(self, ckpt_dir: Path) -> None:
        """Default reason is 'training_crashed'."""
        mgr = RestartManager(checkpoint_dir=ckpt_dir, require_governance=False)
        sentinel = mgr.write_crash_sentinel()
        payload = json.loads(sentinel.read_text(encoding="utf-8"))
        assert payload["reason"] == "training_crashed"


# ── Checkpoint Validation (Minimal) ──────────────────────────────────────────


class TestCheckpointValidation:
    def test_invalid_size_rejected(self, ckpt_dir: Path) -> None:
        mgr = RestartManager(checkpoint_dir=ckpt_dir, require_governance=False)
        path = ckpt_dir / "checkpoint_epoch_1_step_100.pt"
        path.write_text("x" * 50, encoding="utf-8")
        assert mgr._is_checkpoint_valid(path) is False

    def test_missing_file_rejected(self, ckpt_dir: Path) -> None:
        mgr = RestartManager(checkpoint_dir=ckpt_dir, require_governance=False)
        path = ckpt_dir / "nonexistent.pt"
        assert mgr._is_checkpoint_valid(path) is False

    def test_valid_checkpoint_accepted(self, ckpt_dir: Path) -> None:
        mgr = RestartManager(checkpoint_dir=ckpt_dir, require_governance=False)
        path = _make_checkpoint(ckpt_dir, epoch=5, step=1000)
        assert mgr._is_checkpoint_valid(path) is True

    def test_load_checkpoint_metadata(self, ckpt_dir: Path) -> None:
        mgr = RestartManager(checkpoint_dir=ckpt_dir, require_governance=False)
        path = _make_checkpoint(
            ckpt_dir, epoch=7, step=1500, phase_id=2, best_val_loss=0.876
        )
        ckpt = mgr._load_checkpoint_metadata(path)
        assert ckpt["epoch"] == 7
        assert ckpt["global_step"] == 1500
        assert ckpt["phase_id"] == 2
        assert ckpt["best_val_loss"] == 0.876
        assert "model_state_dict" in ckpt


# ── Governance Verification ──────────────────────────────────────────────────


class TestGovernanceVerification:
    """Tests for require_governance=True path."""

    def test_gov_disabled_skips_verification(self, ckpt_dir: Path) -> None:
        """When require_governance=False, no governance check is performed."""
        mgr = RestartManager(checkpoint_dir=ckpt_dir, require_governance=False)
        path = _make_checkpoint(ckpt_dir, epoch=1, step=100)
        assert mgr._is_checkpoint_valid(path) is True

    def test_gov_enabled_valid_manifest(self, ckpt_dir: Path) -> None:
        """Governance verification succeeds when verify_artifact_manifest returns a dict."""
        mgr = RestartManager(checkpoint_dir=ckpt_dir, require_governance=True)
        path = _make_checkpoint(ckpt_dir, epoch=1, step=100)

        with patch(
            "helix_ids.operations.recovery.restart_manager.verify_artifact_manifest",
            return_value={"verified": True, "artifact_sha256": "abc123"},
        ) as mock_verify:
            result = mgr._is_checkpoint_valid(path)
            assert result is True
            mock_verify.assert_called_once_with(path, kind="checkpoint", contract=None)

    def test_gov_enabled_verify_returns_none(self, ckpt_dir: Path) -> None:
        """When verify_artifact_manifest returns None, checkpoint is rejected."""
        mgr = RestartManager(checkpoint_dir=ckpt_dir, require_governance=True)
        path = _make_checkpoint(ckpt_dir, epoch=1, step=100)

        with patch(
            "helix_ids.operations.recovery.restart_manager.verify_artifact_manifest",
            return_value=None,
        ) as mock_verify:
            result = mgr._is_checkpoint_valid(path)
            assert result is False
            mock_verify.assert_called_once()

    def test_gov_enabled_verify_raises_artifact_error(self, ckpt_dir: Path) -> None:
        """When verify_artifact_manifest raises ArtifactManifestError, checkpoint is rejected."""
        mgr = RestartManager(checkpoint_dir=ckpt_dir, require_governance=True)
        path = _make_checkpoint(ckpt_dir, epoch=1, step=100)

        with patch(
            "helix_ids.operations.recovery.restart_manager.verify_artifact_manifest",
            side_effect=ArtifactManifestError("checksum mismatch"),
        ) as mock_verify:
            result = mgr._is_checkpoint_valid(path)
            assert result is False
            mock_verify.assert_called_once()

    def test_gov_enabled_verify_raises_generic_error(self, ckpt_dir: Path) -> None:
        """Generic exception during governance check is caught and returns False."""
        mgr = RestartManager(checkpoint_dir=ckpt_dir, require_governance=True)
        path = _make_checkpoint(ckpt_dir, epoch=1, step=100)

        with patch(
            "helix_ids.operations.recovery.restart_manager.verify_artifact_manifest",
            side_effect=OSError("permission denied"),
        ):
            result = mgr._is_checkpoint_valid(path)
            assert result is False

    def test_gov_resolve_restart_valid(self, ckpt_dir: Path) -> None:
        """Full resolve_restart with governance enabled and valid manifest."""
        _write_sentinel(ckpt_dir)
        _make_checkpoint(ckpt_dir, epoch=5, step=1000, phase_id=2, best_val_loss=0.5)

        mgr = RestartManager(checkpoint_dir=ckpt_dir, require_governance=True)

        with patch(
            "helix_ids.operations.recovery.restart_manager.verify_artifact_manifest",
            return_value={"verified": True},
        ):
            decision = mgr.resolve_restart()
            assert decision.should_restart is True
            assert decision.resume_epoch == 5
            assert decision.resume_phase_id == 2

    def test_gov_resolve_restart_all_fail(self, ckpt_dir: Path) -> None:
        """Governance rejects all checkpoints → should_restart=False."""
        _write_sentinel(ckpt_dir)
        _make_checkpoint(ckpt_dir, epoch=5, step=1000)
        _make_checkpoint(ckpt_dir, epoch=3, step=600)

        mgr = RestartManager(checkpoint_dir=ckpt_dir, require_governance=True)

        with patch(
            "helix_ids.operations.recovery.restart_manager.verify_artifact_manifest",
            side_effect=ArtifactManifestError("invalid manifest"),
        ):
            decision = mgr.resolve_restart()
            assert decision.should_restart is False
            assert "no valid (uncorrupted) checkpoint found" in decision.reason


# ── RestartOutcome ────────────────────────────────────────────────────────────


class TestRestartOutcome:
    """Tests for the RestartOutcome dataclass."""

    def test_outcome_success(self) -> None:
        decision = RestartDecision(
            should_restart=True,
            checkpoint_path=Path("/tmp/ckpt.pt"),
            resume_epoch=10,
            reason="restarting",
        )
        outcome = RestartOutcome(
            success=True,
            decision=decision,
            loaded_checkpoint_path=Path("/tmp/ckpt.pt"),
        )
        assert outcome.success is True
        assert outcome.decision.should_restart is True
        assert outcome.loaded_checkpoint_path == Path("/tmp/ckpt.pt")
        assert outcome.error is None

    def test_outcome_failure(self) -> None:
        decision = RestartDecision(
            should_restart=False,
            reason="no valid checkpoint",
        )
        outcome = RestartOutcome(
            success=False,
            decision=decision,
            error="Failed to load checkpoint: corrupted data",
        )
        assert outcome.success is False
        assert outcome.decision.should_restart is False
        assert outcome.loaded_checkpoint_path is None
        assert "corrupted data" in outcome.error

    def test_outcome_default_loaded_path(self) -> None:
        """When no checkpoint was loaded, loaded_checkpoint_path defaults to None."""
        decision = RestartDecision(should_restart=False, reason="fresh start")
        outcome = RestartOutcome(success=True, decision=decision)
        assert outcome.loaded_checkpoint_path is None
        assert outcome.error is None

    def test_outcome_with_error(self) -> None:
        """Outcome with both decision and error populated."""
        decision = RestartDecision(should_restart=False, reason="crash detected")
        outcome = RestartOutcome(
            success=False,
            decision=decision,
            error="Recovery lock held by another process",
        )
        assert outcome.error == "Recovery lock held by another process"


# ── CrashedState ──────────────────────────────────────────────────────────────


class TestCrashedState:
    """Tests for the CrashedState dataclass."""

    def test_default_values(self) -> None:
        crash = CrashedState(run_id="test_run", detected_at="2026-06-16T12:00:00Z")
        assert crash.run_id == "test_run"
        assert crash.last_valid_checkpoint is None
        assert crash.reason == "unknown"
        assert crash.crashed_epoch is None
        assert crash.crashed_step is None

    def test_all_fields_populated(self) -> None:
        ckpt_path = Path("/tmp/ckpt.pt")
        crash = CrashedState(
            run_id="exp_01",
            detected_at="2026-06-16T12:00:00Z",
            last_valid_checkpoint=ckpt_path,
            reason="OOM",
            crashed_epoch=5,
            crashed_step=1000,
        )
        assert crash.last_valid_checkpoint == ckpt_path
        assert crash.crashed_epoch == 5
        assert crash.crashed_step == 1000


# ── Checkpoint Metadata Edge Cases ────────────────────────────────────────────


class TestCheckpointMetadataEdgeCases:
    """Edge cases for checkpoint metadata loading."""

    def test_checkpoint_missing_epoch_key(self, ckpt_dir: Path) -> None:
        """Checkpoint without 'epoch' key should still load but resume_epoch defaults to 0."""
        _write_sentinel(ckpt_dir)
        ckpt_path = ckpt_dir / "checkpoint_epoch_3_step_500.pt"
        torch.save(
            {
                "model_state_dict": {"weight": torch.randn(4, 4)},
                "global_step": 500,
                "phase_id": 2,
                "best_val_loss": 0.75,
                # no 'epoch' key
            },
            ckpt_path,
        )
        mgr = RestartManager(checkpoint_dir=ckpt_dir, require_governance=False)
        decision = mgr.resolve_restart()
        assert decision.should_restart is True
        assert decision.resume_epoch == 0  # default when key missing

    def test_checkpoint_missing_global_step_key(self, ckpt_dir: Path) -> None:
        """Checkpoint without 'global_step' key should default resume_step to 0."""
        _write_sentinel(ckpt_dir)
        ckpt_path = ckpt_dir / "checkpoint_epoch_4_step_800.pt"
        torch.save(
            {
                "model_state_dict": {"weight": torch.randn(4, 4)},
                "epoch": 4,
                # no 'global_step' key
            },
            ckpt_path,
        )
        mgr = RestartManager(checkpoint_dir=ckpt_dir, require_governance=False)
        decision = mgr.resolve_restart()
        assert decision.should_restart is True
        assert decision.resume_step == 0  # default

    def test_checkpoint_empty_dict(self, ckpt_dir: Path) -> None:
        """Empty checkpoint dict should use all defaults."""
        _write_sentinel(ckpt_dir)
        ckpt_path = ckpt_dir / "checkpoint_epoch_1_step_100.pt"
        torch.save({}, ckpt_path)
        mgr = RestartManager(checkpoint_dir=ckpt_dir, require_governance=False)
        decision = mgr.resolve_restart()
        assert decision.should_restart is True
        assert decision.resume_epoch == 0
        assert decision.resume_step == 0
        assert decision.resume_phase_id == 1
        assert decision.resume_best_val_loss == float("inf")


# ── __init__ Edge Cases ────────────────────────────────────────────────────────


class TestRestartManagerInit:
    """Edge cases for RestartManager.__init__."""

    def test_default_sentinel_dir_equals_ckpt_dir(self, ckpt_dir: Path) -> None:
        """When crash_sentinel_dir is None, sentinel dir = checkpoint dir."""
        mgr = RestartManager(checkpoint_dir=ckpt_dir, require_governance=False)
        assert mgr._sentinel_dir == mgr._ckpt_dir

    def test_separate_sentinel_dir(self, ckpt_dir: Path, sentinel_dir: Path) -> None:
        """When crash_sentinel_dir is provided, it's used as sentinel dir."""
        mgr = RestartManager(
            checkpoint_dir=ckpt_dir,
            crash_sentinel_dir=sentinel_dir,
            require_governance=False,
        )
        assert mgr._sentinel_dir == sentinel_dir
        assert mgr._sentinel_dir != mgr._ckpt_dir

    def test_both_dirs_created(self, tmp_path: Path) -> None:
        """Both checkpoint and sentinel dirs are created if they don't exist."""
        new_ckpt = tmp_path / "new_ckpt"
        new_sentinel = tmp_path / "new_sentinel"
        assert not new_ckpt.exists()
        assert not new_sentinel.exists()

        mgr = RestartManager(
            checkpoint_dir=new_ckpt,
            crash_sentinel_dir=new_sentinel,
            require_governance=False,
        )
        assert new_ckpt.exists()
        assert new_sentinel.exists()


# ── Multiple Crash Sentinels ──────────────────────────────────────────────────


class TestMultipleCrashSentinels:
    """Tests for scenarios involving multiple crash sentinel operations."""

    def test_overwrite_existing_sentinel(self, ckpt_dir: Path) -> None:
        """Writing a sentinel when one already exists overwrites it."""
        mgr = RestartManager(checkpoint_dir=ckpt_dir, require_governance=False)
        sentinel1 = mgr.write_crash_sentinel(reason="first_crash")
        payload1 = json.loads(sentinel1.read_text(encoding="utf-8"))
        assert payload1["reason"] == "first_crash"

        sentinel2 = mgr.write_crash_sentinel(reason="second_crash")
        assert sentinel1 == sentinel2  # same file
        payload2 = json.loads(sentinel2.read_text(encoding="utf-8"))
        assert payload2["reason"] == "second_crash"

    def test_clear_and_recreate_sentinel(self, ckpt_dir: Path) -> None:
        """Clearing and recreating a sentinel works correctly."""
        mgr = RestartManager(checkpoint_dir=ckpt_dir, require_governance=False)

        mgr.write_crash_sentinel(reason="crash_1")
        assert mgr._detect_crash() is not None

        mgr.clear_crash_sentinel()
        assert mgr._detect_crash() is None

        mgr.write_crash_sentinel(reason="crash_2")
        crash = mgr._detect_crash()
        assert crash is not None
        assert crash.reason == "crash_2"

    def test_sentinel_in_ckpt_dir_not_sentinel_dir(self, ckpt_dir: Path, sentinel_dir: Path) -> None:
        """A sentinel in checkpoint_dir is NOT detected when sentinel_dir is separate."""
        _write_sentinel(ckpt_dir)  # write to ckpt_dir
        mgr = RestartManager(
            checkpoint_dir=ckpt_dir,
            crash_sentinel_dir=sentinel_dir,
            require_governance=False,
        )
        crash = mgr._detect_crash()
        assert crash is None  # should look in sentinel_dir, not ckpt_dir

    def test_sentinel_only_in_sentinel_dir_detected(self, ckpt_dir: Path, sentinel_dir: Path) -> None:
        """A sentinel in sentinel_dir IS detected when sentinel_dir is separate."""
        _write_sentinel(sentinel_dir)  # write to sentinel dir
        mgr = RestartManager(
            checkpoint_dir=ckpt_dir,
            crash_sentinel_dir=sentinel_dir,
            require_governance=False,
        )
        crash = mgr._detect_crash()
        assert crash is not None


# ── Detection of Run ID ──────────────────────────────────────────────────────


class TestDetectRunId:
    """Tests for _detect_run_id."""

    def test_no_run_id_file_returns_unknown(self, ckpt_dir: Path) -> None:
        mgr = RestartManager(checkpoint_dir=ckpt_dir, require_governance=False)
        run_id = mgr._detect_run_id()
        assert run_id == "unknown"

    def test_run_id_file_read(self, ckpt_dir: Path) -> None:
        run_id_file = ckpt_dir / "_run_id.txt"
        run_id_file.write_text("experiment_42\n", encoding="utf-8")
        mgr = RestartManager(checkpoint_dir=ckpt_dir, require_governance=False)
        run_id = mgr._detect_run_id()
        assert run_id == "experiment_42"

    def test_run_id_file_via_sentinel_dir(self, ckpt_dir: Path, sentinel_dir: Path) -> None:
        """_detect_run_id looks in sentinel_dir, not checkpoint_dir."""
        # Write run_id to sentinel_dir
        run_id_file = sentinel_dir / "_run_id.txt"
        run_id_file.write_text("sentinel_run\n", encoding="utf-8")
        # Also write one to ckpt_dir (should be ignored)
        ckpt_run_id = ckpt_dir / "_run_id.txt"
        ckpt_run_id.write_text("ckpt_run\n", encoding="utf-8")

        mgr = RestartManager(
            checkpoint_dir=ckpt_dir,
            crash_sentinel_dir=sentinel_dir,
            require_governance=False,
        )
        run_id = mgr._detect_run_id()
        assert run_id == "sentinel_run"

    def test_crash_detected_with_run_id_from_file(self, ckpt_dir: Path) -> None:
        """When sentinel has no run_id, _detect_run_id reads from file."""
        # Write run_id file
        run_id_file = ckpt_dir / "_run_id.txt"
        run_id_file.write_text("auto_id\n", encoding="utf-8")
        # Write sentinel without run_id
        sentinel = ckpt_dir / "_crash_sentinel"
        sentinel.write_text(
            json.dumps({"timestamp": "2026-06-16T12:00:00Z", "reason": "crash"}),
            encoding="utf-8",
        )
        mgr = RestartManager(checkpoint_dir=ckpt_dir, require_governance=False)
        crash = mgr._detect_crash()
        assert crash is not None
        assert crash.run_id == "auto_id"
