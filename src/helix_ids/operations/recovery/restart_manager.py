"""Auto-restart recovery system for HELIX-IDS training pipelines.

Detects crashes, discovers the latest valid checkpoint, verifies integrity,
and returns a ``RestartDecision`` describing how to resume training without
manual intervention.

Usage
-----
    mgr = RestartManager(checkpoint_dir="/path/to/checkpoints")
    decision = mgr.resolve_restart()
    if decision.should_restart:
        ckpt = decision.checkpoint
        # resume from ckpt["epoch"], ckpt["phase_id"], etc.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from helix_ids.governance.provenance import (
    verify_artifact_manifest,
)

logger = logging.getLogger(__name__)

# ── Pattern for HELIX checkpoint filenames ───────────────────────────────────
# Expected: checkpoint_epoch_<N>_step_<M>.pt
_CHECKPOINT_RE = re.compile(
    r"checkpoint_epoch_(?P<epoch>\d+)_step_(?P<step>\d+)\.pt$"
)

# Sentinels
_CRASH_SENTINEL = "_crash_sentinel"
_RECOVERY_LOCK = "_recovery.lock"


@dataclass(frozen=True)
class CrashedState:
    """Describes the crash state of a training run."""

    run_id: str
    detected_at: str  # ISO 8601
    last_valid_checkpoint: Path | None = None
    reason: str = "unknown"
    crashed_epoch: int | None = None
    crashed_step: int | None = None


@dataclass(frozen=True)
class RestartDecision:
    """Decision produced by ``RestartManager.resolve_restart``."""

    should_restart: bool
    checkpoint: dict[str, Any] | None = None
    checkpoint_path: Path | None = None
    crash: CrashedState | None = None
    resume_epoch: int = 0
    resume_step: int = 0
    resume_phase_id: int = 1
    resume_best_val_loss: float = float("inf")
    reason: str = ""

    @property
    def metadata(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary of this decision."""
        return {
            "should_restart": self.should_restart,
            "resume_epoch": self.resume_epoch,
            "resume_step": self.resume_step,
            "resume_phase_id": self.resume_phase_id,
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
            "reason": self.reason,
        }


@dataclass
class RestartOutcome:
    """Result of executing a restart decision."""

    success: bool
    decision: RestartDecision
    loaded_checkpoint_path: Path | None = None
    error: str | None = None


# ═══════════════════════════════════════════════════════════════════════════════
# RestartManager
# ═══════════════════════════════════════════════════════════════════════════════


class RestartManager:
    """Detects crashes, discovers valid checkpoints, and produces restart decisions.

    Parameters
    ----------
    checkpoint_dir : str or Path
        Directory containing checkpoint ``.pt`` files.
    crash_sentinel_dir : str or Path or None
        Directory where crash sentinel files live. If None, uses
        *checkpoint_dir*.
    require_governance : bool
        If True (default), verify the artifact manifest (SHA-256) before
        accepting a checkpoint as valid.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        *,
        crash_sentinel_dir: str | Path | None = None,
        require_governance: bool = True,
    ):
        self._ckpt_dir = Path(checkpoint_dir)
        self._sentinel_dir = (
            Path(crash_sentinel_dir) if crash_sentinel_dir else self._ckpt_dir
        )
        self._require_governance = require_governance
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._sentinel_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ──────────────────────────────────────────────────────────

    def resolve_restart(self) -> RestartDecision:
        """Check for crashes, find the best checkpoint, and decide.

        Returns a ``RestartDecision`` with ``should_restart=True`` if a valid
        checkpoint was found and a crash was detected. Otherwise returns a
        fresh-start decision.
        """
        # 1. Detect crash
        crash = self._detect_crash()
        if crash is None:
            return RestartDecision(
                should_restart=False,
                reason="no crash detected, starting fresh",
            )

        # 2. Discover latest valid checkpoint
        candidates = self._discover_checkpoints()
        if not candidates:
            return RestartDecision(
                should_restart=False,
                crash=crash,
                reason=(
                    f"crash detected but no checkpoints found in {self._ckpt_dir}"
                ),
            )

        # 3. Verify and select the best checkpoint
        best = self._select_best_checkpoint(candidates)
        if best is None:
            return RestartDecision(
                should_restart=False,
                crash=crash,
                reason="no valid (uncorrupted) checkpoint found",
            )

        # 4. Load checkpoint metadata
        try:
            ckpt = self._load_checkpoint_metadata(best)
        except Exception as exc:
            return RestartDecision(
                should_restart=False,
                crash=crash,
                reason=f"failed to load checkpoint {best.name}: {exc}",
            )

        return RestartDecision(
            should_restart=True,
            checkpoint=ckpt,
            checkpoint_path=best,
            crash=crash,
            resume_epoch=ckpt.get("epoch", 0),
            resume_step=ckpt.get("global_step", 0),
            resume_phase_id=ckpt.get("phase_id", 1),
            resume_best_val_loss=ckpt.get("best_val_loss", float("inf")),
            reason=f"restarting from {best.name} (epoch {ckpt.get('epoch', '?')})",
        )

    def write_crash_sentinel(self, *, reason: str = "training_crashed") -> Path:
        """Write a crash sentinel file to mark an abrupt termination.

        This is called by the training loop wrapper when it catches an
        unexpected exception, so that the next invocation of
        ``resolve_restart`` can detect the crash.
        """
        sentinel = self._sentinel_dir / _CRASH_SENTINEL
        payload = {
            "timestamp": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "reason": reason,
        }
        sentinel.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Wrote crash sentinel: %s", sentinel)
        return sentinel

    def clear_crash_sentinel(self) -> None:
        """Remove the crash sentinel after a successful restart."""
        sentinel = self._sentinel_dir / _CRASH_SENTINEL
        if sentinel.exists():
            sentinel.unlink()
            logger.info("Cleared crash sentinel: %s", sentinel)

    def acquire_recovery_lock(self, *, timeout_seconds: int = 30) -> bool:
        """Acquire a recovery lock to prevent concurrent restarts.

        Returns True if the lock was acquired.
        """
        lock_file = self._sentinel_dir / _RECOVERY_LOCK
        if lock_file.exists():
            age = datetime.now(timezone.utc).timestamp() - lock_file.stat().st_mtime
            if age < timeout_seconds:
                return False
            logger.warning(
                "Recovery lock expired (age=%.1fs), overriding", age
            )
            lock_file.unlink(missing_ok=True)
        lock_file.write_text(
            json.dumps(
                {
                    "timestamp": datetime.now(timezone.utc).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    )
                }
            ),
            encoding="utf-8",
        )
        return True

    def release_recovery_lock(self) -> None:
        """Release the recovery lock."""
        lock_file = self._sentinel_dir / _RECOVERY_LOCK
        if lock_file.exists():
            lock_file.unlink()

    # ── Internal methods ────────────────────────────────────────────────────

    def _detect_run_id(self) -> str:
        """Extract or generate a run identifier."""
        # Could be read from environment or a run_id file
        run_id_file = self._sentinel_dir / "_run_id.txt"
        if run_id_file.exists():
            return run_id_file.read_text(encoding="utf-8").strip()
        return "unknown"

    def _detect_crash(self) -> CrashedState | None:
        """Check for the presence of a crash sentinel.

        Returns a ``CrashedState`` if a sentinel exists, otherwise ``None``.
        """
        sentinel = self._sentinel_dir / _CRASH_SENTINEL
        if not sentinel.exists():
            return None

        try:
            payload = json.loads(sentinel.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            payload = {}

        run_id = payload.get("run_id", self._detect_run_id())

        return CrashedState(
            run_id=run_id,
            detected_at=payload.get(
                "timestamp",
                datetime.fromtimestamp(
                    sentinel.stat().st_mtime, tz=timezone.utc
                ).strftime("%Y-%m-%dT%H:%M:%SZ"),
            ),
            reason=payload.get("reason", "unknown"),
        )

    def _discover_checkpoints(self) -> list[Path]:
        """Scan *checkpoint_dir* for ``.pt`` checkpoint files.

        Returns paths sorted by (epoch, step) descending — newest first.
        """
        candidates: list[tuple[int, int, Path]] = []
        for fpath in sorted(self._ckpt_dir.iterdir()):
            match = _CHECKPOINT_RE.search(str(fpath.name))
            if not match:
                continue
            epoch = int(match.group("epoch"))
            step = int(match.group("step"))
            candidates.append((epoch, step, fpath))

        # Sort by epoch desc, step desc
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [p for _, _, p in candidates]

    def _select_best_checkpoint(
        self, candidates: list[Path]
    ) -> Path | None:
        """Pick the first valid checkpoint from *candidates* (newest first).

        If ``_require_governance`` is set, each candidate must pass
        ``verify_artifact_manifest``.
        """
        for path in candidates:
            if self._is_checkpoint_valid(path):
                return path
        return None

    def _is_checkpoint_valid(self, path: Path) -> bool:
        """Check that a checkpoint file is readable and (optionally) verified."""
        if not path.exists() or path.stat().st_size < 100:
            return False

        if not self._require_governance:
            return True

        try:
            # Verify against its embedded manifest
            result = verify_artifact_manifest(
                path, kind="checkpoint", contract=None
            )
            # result is a dict if valid, or raises ArtifactManifestError
            return result is not None
        except Exception:
            return False

    def _load_checkpoint_metadata(self, path: Path) -> dict[str, Any]:
        """Load checkpoint metadata into memory.

        Uses ``weights_only=True`` for safe PyTorch deserialisation. Only
        loads the minimal dict — caller is responsible for applying state
        dicts to the model/optimizer.
        """
        import torch

        ckpt: dict[str, Any] = torch.load(
            path, map_location="cpu", weights_only=True
        )
        return ckpt
