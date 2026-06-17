"""
Phase 22B — P5: Memory Leak Detection

Instrumentation:
  - tracemalloc (peak + current memory)
  - gc statistics (collected objects per generation)
  - torch tensor count
  - file descriptor count
  - thread count

Required loops:
  1. 10k training-step loop  (forward + backward)
  2. 10k inference loop       (predict calls)
  3. 10k logging-context loop (LogContextManager enter/exit)
  4.  1k checkpoint save/load loop
  5.  1k restart-manager cycle loop

Gate: No monotonic growth trend. Temporary allocations may grow;
       retained allocations must not.
"""

from __future__ import annotations

import gc
import os
import threading
import tracemalloc
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

from helix_ids.contracts import (
    CANONICAL_BINARY_CLASSES,
    CANONICAL_FAMILY_CLASSES,
    CANONICAL_INPUT_DIM,
    FEATURE_ORDER_HASH,
    runtime_contract_payload,
)
from helix_ids.governance.provenance import (
    ARTIFACT_MANIFEST_KEY,
    checkpoint_manifest_payload,
    write_contract_sidecars,
)
from helix_ids.models.full import HelixFullConfig, create_helix_full
from helix_ids.operations.inference_runtime import HelixInferenceRuntime
from helix_ids.operations.logging.log_context import LogContextManager
from helix_ids.operations.recovery.restart_manager import RestartManager
from helix_ids.utils.export import (
    build_export_manifest,
    finalize_export_artifact,
)

_RC = runtime_contract_payload()
CANONICAL_CONTRACT = {
    "schema_version": str(_RC["schema_version"]),
    "schema_hash": str(_RC["schema_hash"]),
    "feature_order": list(_RC["feature_order"]),
    "input_dim": int(_RC["input_dim"]),
    "binary_output_dim": int(_RC["binary_output_dim"]),
    "family_output_dim": int(_RC["family_output_dim"]),
    "contract_version": "2.1",
    "feature_order_hash": FEATURE_ORDER_HASH,
}

# ── Snapshot helpers ─────────────────────────────────────────────────────────


@dataclass
class MemSnapshot:
    """A point-in-time memory snapshot."""

    label: str
    tracemalloc_current: int  # bytes
    tracemalloc_peak: int  # bytes
    gc_counts: tuple[int, int, int]  # gen0, gen1, gen2 collected
    gc_garbage: int
    tensor_count: int
    tensor_bytes: int
    fd_count: int
    thread_count: int


def _count_tensors() -> tuple[int, int]:
    """Count live tensors and their total byte size."""
    count = 0
    total_bytes = 0
    for obj in gc.get_objects():
        if isinstance(obj, torch.Tensor):
            count += 1
            try:
                total_bytes += obj.numel() * obj.element_size()
            except Exception:
                pass
    return count, total_bytes


def _count_fds() -> int:
    """Count open file descriptors."""
    try:
        return len(os.listdir(f"/proc/{os.getpid()}/fd"))
    except FileNotFoundError:
        # macOS fallback
        try:
            import resource
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            # Approximate by counting /dev/fd entries
            return len(os.listdir("/dev/fd"))
        except Exception:
            return -1


def _count_threads() -> int:
    return threading.active_count()


def take_snapshot(label: str) -> MemSnapshot:
    """Collect all instrumentation metrics at once."""
    gc.collect()
    gc.collect()
    tracemalloc_snapshot = tracemalloc.take_snapshot()
    top_stats = tracemalloc_snapshot.statistics("traceback")
    current = sum(stat.size for stat in top_stats)
    peak = tracemalloc.get_traced_memory()[1]

    gc_counts = gc.get_count()
    gc_garbage = len(gc.garbage)

    tensor_count, tensor_bytes = _count_tensors()
    fd_count = _count_fds()
    thread_count = _count_threads()

    return MemSnapshot(
        label=label,
        tracemalloc_current=current,
        tracemalloc_peak=peak,
        gc_counts=gc_counts,
        gc_garbage=gc_garbage,
        tensor_count=tensor_count,
        tensor_bytes=tensor_bytes,
        fd_count=fd_count,
        thread_count=thread_count,
    )


def assert_no_leak_trend(
    initial: MemSnapshot,
    midpoint: MemSnapshot,
    final: MemSnapshot,
    *,
    label: str = "",
) -> None:
    """Assert no monotonic growth in retained allocations.

    Temporary allocations may grow between initial→midpoint but must
    stabilise or drop by midpoint→final, or stay under a threshold.
    """
    max_allowed_growth = 5_000_000  # 5 MB max retained growth

    # Current tracemalloc should not grow monotonically
    if (midpoint.tracemalloc_current > initial.tracemalloc_current
            and final.tracemalloc_current > midpoint.tracemalloc_current):
        growth = final.tracemalloc_current - initial.tracemalloc_current
        assert growth < max_allowed_growth, (
            f"{label}: monotonic current-memory growth "
            f"({growth / 1e6:.1f} MB). "
            f"Initial={initial.tracemalloc_current / 1e6:.1f} MB, "
            f"Mid={midpoint.tracemalloc_current / 1e6:.1f} MB, "
            f"Final={final.tracemalloc_current / 1e6:.1f} MB"
        )

    # Final vs initial should not exceed max threshold
    retained = final.tracemalloc_current - initial.tracemalloc_current
    if retained > max_allowed_growth:
        # Allow one retry with forced GC
        gc.collect()
        gc.collect()
        retained_after_gc = tracemalloc.get_traced_memory()[0] - initial.tracemalloc_current
        if retained_after_gc > max_allowed_growth:
            assert False, (
                f"{label}: retained allocations {retained_after_gc / 1e6:.1f} MB "
                f"exceeds max {max_allowed_growth / 1e6:.1f} MB"
            )
        retained = retained_after_gc

    # Tensor count should not grow monotonically
    if (midpoint.tensor_count > initial.tensor_count
            and final.tensor_count > midpoint.tensor_count):
        growth = final.tensor_count - initial.tensor_count
        assert growth < 10, (
            f"{label}: monotonic tensor-count growth ({growth} tensors)"
        )

    # Thread count should not grow
    thread_growth = final.thread_count - initial.thread_count
    assert thread_growth <= 1, (
        f"{label}: thread count grew from {initial.thread_count} "
        f"to {final.thread_count}"
    )


def format_snapshot_diff(
    initial: MemSnapshot,
    midpoint: MemSnapshot,
    final: MemSnapshot,
) -> str:
    """Format a human-readable diff table."""
    lines = [
        f"{'Metric':<35} {'Initial':>15} {'Midpoint':>15} {'Final':>15}",
        "-" * 80,
    ]
    rows = [
        ("Tracemalloc Current (MB)", initial.tracemalloc_current / 1e6,
         midpoint.tracemalloc_current / 1e6, final.tracemalloc_current / 1e6),
        ("Tracemalloc Peak (MB)", initial.tracemalloc_peak / 1e6,
         midpoint.tracemalloc_peak / 1e6, final.tracemalloc_peak / 1e6),
        ("GC Gen0/1/2", f"{initial.gc_counts}",
         f"{midpoint.gc_counts}", f"{final.gc_counts}"),
        ("GC Garbage", initial.gc_garbage,
         midpoint.gc_garbage, final.gc_garbage),
        ("Tensor Count", initial.tensor_count,
         midpoint.tensor_count, final.tensor_count),
        ("Tensor Bytes (MB)", initial.tensor_bytes / 1e6,
         midpoint.tensor_bytes / 1e6, final.tensor_bytes / 1e6),
        ("File Descriptors", initial.fd_count,
         midpoint.fd_count, final.fd_count),
        ("Threads", initial.thread_count,
         midpoint.thread_count, final.thread_count),
    ]
    for name, i, m, f in rows:
        if isinstance(i, str):
            lines.append(f"{name:<35} {i:>15} {m:>15} {f:>15}")
        else:
            lines.append(f"{name:<35} {i:>15.2f} {m:>15.2f} {f:>15.2f}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="session", autouse=True)
def _start_tracemalloc() -> Generator[None, None, None]:
    tracemalloc.start(25)
    yield
    tracemalloc.stop()


@pytest.fixture
def model() -> torch.nn.Module:
    return create_helix_full(
        HelixFullConfig(
            input_dim=CANONICAL_INPUT_DIM,
            binary_output_dim=CANONICAL_BINARY_CLASSES,
            family_output_dim=CANONICAL_FAMILY_CLASSES,
        )
    )


@pytest.fixture
def blessed_checkpoint(
    tmp_path: Path,
    model: torch.nn.Module,
) -> Path:
    """Write a fully blessed checkpoint to a temp file and return its path."""
    contract = dict(CANONICAL_CONTRACT)
    manifest_base = build_export_manifest(
        contract=contract,
        model_architecture="HelixFull",
        export_config={"format": "checkpoint", "origin": "memleak_test"},
    )
    payload = {
        "model_state_dict": {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        },
        "model": {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        },
        **contract,
        ARTIFACT_MANIFEST_KEY: checkpoint_manifest_payload(manifest_base),
    }
    path = tmp_path / "blessed.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    sidecars = write_contract_sidecars(path, contract)
    finalize_export_artifact(path, manifest_base, sidecars=sidecars)
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# Loop 1: 10k training-step loop (forward + backward)
# ═══════════════════════════════════════════════════════════════════════════════


class TestTrainingLoopMemory:
    """10k training-step loop — forward pass + backward."""

    N_STEPS = 10_000

    def test_no_leak_in_training_loop(
        self,
        model: torch.nn.Module,
    ) -> None:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        loss_fn = torch.nn.CrossEntropyLoss()

        initial = take_snapshot("training_initial")

        for step in range(self.N_STEPS):
            x = torch.randn(32, CANONICAL_INPUT_DIM)
            y_binary = torch.randint(0, CANONICAL_BINARY_CLASSES, (32,))
            y_family = torch.randint(0, CANONICAL_FAMILY_CLASSES, (32,))

            optimizer.zero_grad()
            binary_logits, family_logits = model(x)
            loss = loss_fn(binary_logits, y_binary)
            loss += loss_fn(family_logits, y_family)
            loss.backward()
            optimizer.step()

            if step == self.N_STEPS // 2:
                midpoint = take_snapshot("training_midpoint")

            # Clear per-iteration tensors
            del x, y_binary, y_family, binary_logits, family_logits, loss

        final = take_snapshot("training_final")

        print("\n--- Training Loop Memory Snapshot ---")
        print(format_snapshot_diff(initial, midpoint, final))

        assert_no_leak_trend(initial, midpoint, final, label="10k training step")


# ═══════════════════════════════════════════════════════════════════════════════
# Loop 2: 10k inference loop (predict calls)
# ═══════════════════════════════════════════════════════════════════════════════


class TestInferenceLoopMemory:
    """10k inference loop — HelixInferenceRuntime.predict()."""

    N_STEPS = 10_000

    def test_no_leak_in_inference_loop(
        self,
        blessed_checkpoint: Path,
    ) -> None:
        runtime = HelixInferenceRuntime(blessed_checkpoint, device="cpu")

        initial = take_snapshot("inference_initial")

        for step in range(self.N_STEPS):
            features = np.random.randn(CANONICAL_INPUT_DIM).astype(np.float32)
            result = runtime.predict(features)

            if step == self.N_STEPS // 2:
                midpoint = take_snapshot("inference_midpoint")

            del features, result

        final = take_snapshot("inference_final")

        print("\n--- Inference Loop Memory Snapshot ---")
        print(format_snapshot_diff(initial, midpoint, final))

        assert_no_leak_trend(initial, midpoint, final, label="10k inference loop")


# ═══════════════════════════════════════════════════════════════════════════════
# Loop 3: 10k logging-context loop (LogContextManager enter/exit)
# ═══════════════════════════════════════════════════════════════════════════════


class TestLoggingContextLoopMemory:
    """10k logging-context loop — LogContextManager enter/exit."""

    N_STEPS = 10_000

    def test_no_leak_in_logging_context_loop(self) -> None:
        initial = take_snapshot("logging_initial")

        for step in range(self.N_STEPS):
            with LogContextManager(
                run_id=f"run_{step % 10}",
                phase="testing",
                epoch=step,
                step=step,
                extra={"batch": step},
            ):
                _ = {"msg": "hello", "value": step * 2}

            if step == self.N_STEPS // 2:
                midpoint = take_snapshot("logging_midpoint")

        final = take_snapshot("logging_final")

        print("\n--- Logging Context Loop Memory Snapshot ---")
        print(format_snapshot_diff(initial, midpoint, final))

        assert_no_leak_trend(initial, midpoint, final, label="10k logging context")


# ═══════════════════════════════════════════════════════════════════════════════
# Loop 4: 1k checkpoint save/load loop
# ═══════════════════════════════════════════════════════════════════════════════


class TestCheckpointSaveLoadLoopMemory:
    """1k checkpoint save/load cycle."""

    N_CYCLES = 1_000

    def test_no_leak_in_checkpoint_save_load(
        self,
        tmp_path: Path,
        model: torch.nn.Module,
    ) -> None:
        contract = dict(CANONICAL_CONTRACT)
        manifest_base = build_export_manifest(
            contract=contract,
            model_architecture="HelixFull",
            export_config={"format": "checkpoint", "origin": "memleak_saveload"},
        )

        initial = take_snapshot("saveload_initial")

        for cycle in range(self.N_CYCLES):
            payload = {
                "model_state_dict": {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                },
                "epoch": cycle,
                **contract,
                ARTIFACT_MANIFEST_KEY: checkpoint_manifest_payload(manifest_base),
            }
            ckpt_path = tmp_path / f"loop_ckpt_{cycle}.pt"
            torch.save(payload, ckpt_path)

            loaded = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            assert loaded["epoch"] == cycle

            if cycle == self.N_CYCLES // 2:
                midpoint = take_snapshot("saveload_midpoint")

            del payload, loaded
            ckpt_path.unlink(missing_ok=True)

        final = take_snapshot("saveload_final")

        print("\n--- Checkpoint Save/Load Memory Snapshot ---")
        print(format_snapshot_diff(initial, midpoint, final))

        assert_no_leak_trend(initial, midpoint, final, label="1k checkpoint save/load")


# ═══════════════════════════════════════════════════════════════════════════════
# Loop 5: 1k restart-manager cycle loop
# ═══════════════════════════════════════════════════════════════════════════════


class TestRestartManagerCycleMemory:
    """1k restart-manager cycle (sentinel + resolve)."""

    N_CYCLES = 1_000

    def test_no_leak_in_restart_manager_cycles(
        self,
        tmp_path: Path,
        model: torch.nn.Module,
    ) -> None:
        contract = dict(CANONICAL_CONTRACT)
        manifest_base = build_export_manifest(
            contract=contract,
            model_architecture="HelixFull",
            export_config={"format": "checkpoint", "origin": "memleak_restart"},
        )

        ckpt_dir = tmp_path / "restart_ckpts"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Write one baseline checkpoint
        baseline = ckpt_dir / "baseline.pt"
        payload = {
            "model_state_dict": {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            },
            "model": {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            },
            "epoch": 10,
            "global_step": 5000,
            "phase_id": 3,
            "best_val_loss": 0.123,
            **contract,
            ARTIFACT_MANIFEST_KEY: checkpoint_manifest_payload(manifest_base),
        }
        torch.save(payload, baseline)
        sidecars = write_contract_sidecars(baseline, contract)
        finalize_export_artifact(baseline, manifest_base, sidecars=sidecars)

        sentinel_dir = tmp_path / "sentinel"
        sentinel_dir.mkdir(parents=True, exist_ok=True)

        initial = take_snapshot("restart_initial")

        for cycle in range(self.N_CYCLES):
            rm = RestartManager(
                checkpoint_dir=ckpt_dir,
                crash_sentinel_dir=sentinel_dir,
            )
            rm.write_crash_sentinel()
            decision = rm.resolve_restart()

            assert decision is not None

            if cycle == self.N_CYCLES // 2:
                midpoint = take_snapshot("restart_midpoint")

            del rm, decision

        final = take_snapshot("restart_final")

        print("\n--- Restart Manager Cycle Memory Snapshot ---")
        print(format_snapshot_diff(initial, midpoint, final))

        assert_no_leak_trend(initial, midpoint, final, label="1k restart-manager cycle")
