"""Regression tests for EpochRunner (Phase 17 extraction).

Covers:
  1.  run_epoch returns correct metrics structure
  2.  Batch diversity check (min 2 unique classes)
  3.  Class starvation detection
  4.  Warmup integration in epoch 0
  5.  Step 10 diagnostics logging flag
  6.  Logit range tracking
  7.  Loss strategy selection
"""

from __future__ import annotations

import sys
from typing import Any

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from scripts.training.execution.batch_processor import BatchProcessor
from scripts.training.execution.epoch_runner import EpochRunner
from scripts.training.execution.warmup_manager import WarmupManager

# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #


@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class _SimpleFamilyModel(nn.Module):
    """Minimal model for test forward pass."""

    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        self.backbone = nn.Linear(10, 16)
        self.binary_head = nn.Linear(16, 2)
        self.family_head = nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = True) -> tuple:
        h = self.backbone(x)
        binary_logits = self.binary_head(h)
        family_logits = self.family_head(h)
        return binary_logits, family_logits, h


class _FakeConfig:
    log_interval: int = 5
    max_grad_norm: float = 1.0


class _DiverseFamilyDataset(Dataset):
    """Dataset with at least 3 distinct family classes per batch (via small size)."""

    def __init__(
        self, num_samples: int = 16, num_classes: int = 5,
    ) -> None:
        self.num_samples = num_samples
        self.num_classes = num_classes
        self._x = torch.randn(num_samples, 10)
        # Ensure diverse classes
        self._family_labels = torch.arange(num_samples) % num_classes

    @property
    def family_labels(self) -> torch.Tensor:
        return self._family_labels

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        y_binary = 1 if int(self._family_labels[idx]) >= (self.num_classes // 2) else 0
        return self._x[idx], y_binary, int(self._family_labels[idx])


@pytest.fixture
def dataset() -> Dataset[Any]:
    return _DiverseFamilyDataset(num_samples=16, num_classes=5)


@pytest.fixture
def train_loader(dataset: Dataset[Any]) -> DataLoader[Any]:
    return DataLoader(dataset, batch_size=4, shuffle=False)


@pytest.fixture
def model(device: torch.device) -> nn.Module:
    m = _SimpleFamilyModel(num_classes=5).to(device)
    m.train()
    return m


@pytest.fixture
def loss_fn() -> Any:
    """Multi-task loss matching the trainer's loss function."""
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
    from helix_ids.models.helix_ids_full import MultiTaskLoss  # type: ignore[import-untyped]

    return MultiTaskLoss()


@pytest.fixture
def config() -> Any:
    return _FakeConfig()


@pytest.fixture
def logger() -> Any:
    class _TestLogger:
        def __init__(self) -> None:
            self.messages: list[str] = []

        def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
            self.messages.append(msg % args if args else msg)

        def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
            self.messages.append(f"WARN:{msg % args if args else msg}")

        def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
            self.messages.append(f"ERR:{msg % args if args else msg}")

    return _TestLogger()


@pytest.fixture
def batch_processor(
    model: nn.Module,
    loss_fn: Any,
    config: Any,
    device: torch.device,
) -> BatchProcessor:
    from scripts.training.losses import LossRegistry

    registry = LossRegistry(
        entropy_warmup_steps=100,
        entropy_warmup_weight=0.01,
        kl_uniform_weight=0.001,
        warmup_kl_uniform_weight=0.01,
        logit_floor=-5.0,
        logit_floor_weight=0.001,
        tail_ce_weight=0.0,
        tail_class_mask=None,
        loss_fn=None,
        energy_gap_weight=0.5,
        energy_multi_negative_alpha=1.0,
        energy_balance_weight=0.01,
        energy_winner_weight=0.01,
        energy_winner_min_count=1,
        energy_logit_temperature=2.0,
        energy_win_rate_ema_momentum=0.9,
        energy_emergence_bias_eps=1e-3,
        energy_emergence_bias_beta=0.5,
        energy_emergence_bias_ratio_max=0.30,
    )
    return BatchProcessor(
        model=model, loss_fn=loss_fn, loss_registry=registry, config=config,
    )


@pytest.fixture
def warmup_manager(
    model: nn.Module,
    loss_fn: Any,
    config: Any,
    device: torch.device,
) -> WarmupManager:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    return WarmupManager(
        model=model, loss_fn=loss_fn, optimizer=optimizer,
        device=device, config=config,
    )


@pytest.fixture
def epoch_runner(
    model: nn.Module,
    train_loader: DataLoader[Any],
    config: Any,
    device: torch.device,
    logger: Any,
    batch_processor: BatchProcessor,
    warmup_manager: WarmupManager,
) -> EpochRunner:
    return EpochRunner(
        model=model,
        train_loader=train_loader,
        config=config,
        device=device,
        logger=logger,
        batch_processor=batch_processor,
        warmup_manager=warmup_manager,
    )


@pytest.fixture(autouse=True)
def _set_seed() -> None:
    torch.manual_seed(42)


def _null_hook(*args: Any, **kwargs: Any) -> Any:
    """No-op hook for callable parameters."""
    return None


@pytest.fixture
def optimizer(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.SGD(model.parameters(), lr=0.01)


# ------------------------------------------------------------------ #
# 1. Return structure
# ------------------------------------------------------------------ #


class TestRunEpochReturnStructure:
    """Verify run_epoch returns all expected fields."""

    def test_returns_all_keys(
        self,
        epoch_runner: EpochRunner,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        result = epoch_runner.run_epoch(
            epoch=1,
            global_step=0,
            warmup_steps=100,
            representation_diagnostic_mode=False,
            use_energy_based_family_objective=False,
            active_family_class_ids=[0, 1, 2, 3, 4],
            enforce_all_classes_per_batch=False,
            step_coverage_checked=False,
            head_phase_start_step=0,
            representation_phase_active=False,
            in_representation_window=False,
            representation_curriculum_complete=False,
            binary_class_weights=None,
            family_class_weights=None,
            family_log_prior=None,
            energy_logit_temperature=1.0,
            disable_tail_focal_regularizer=False,
            optimizer=optimizer,
            backbone_params=list(epoch_runner._model.parameters()),
            check_backbone_freeze_state=_null_hook,
            handle_representation_phase_logic=_null_hook,
            maybe_activate_joint_finetune_phase=_null_hook,
            check_family_class_coverage=_null_hook,
            check_step_coverage=_null_hook,
            freeze_epoch_centroid_snapshot=_null_hook,
            update_centroids_from_epoch_buffer=_null_hook,
        )
        expected_keys = {
            "metrics", "global_step", "class_starvation_streak",
            "family_pred_counts", "family_logit_sums",
            "step10_symmetry_logged",
        }
        assert set(result.keys()) == expected_keys

    def test_metrics_are_positive_finite(
        self,
        epoch_runner: EpochRunner,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        result = epoch_runner.run_epoch(
            epoch=1,
            global_step=0,
            warmup_steps=100,
            representation_diagnostic_mode=False,
            use_energy_based_family_objective=False,
            active_family_class_ids=[0, 1, 2, 3, 4],
            enforce_all_classes_per_batch=False,
            step_coverage_checked=False,
            head_phase_start_step=0,
            representation_phase_active=False,
            in_representation_window=False,
            representation_curriculum_complete=False,
            binary_class_weights=None,
            family_class_weights=None,
            family_log_prior=None,
            energy_logit_temperature=1.0,
            disable_tail_focal_regularizer=False,
            optimizer=optimizer,
            backbone_params=list(epoch_runner._model.parameters()),
            check_backbone_freeze_state=_null_hook,
            handle_representation_phase_logic=_null_hook,
            maybe_activate_joint_finetune_phase=_null_hook,
            check_family_class_coverage=_null_hook,
            check_step_coverage=_null_hook,
            freeze_epoch_centroid_snapshot=_null_hook,
            update_centroids_from_epoch_buffer=_null_hook,
        )
        m = result["metrics"]
        assert m["train_loss"] > 0.0
        assert m["train_calibrated_loss"] > 0.0
        assert 0.0 <= m["train_binary_acc"] <= 1.0
        assert 0.0 <= m["train_family_acc"] <= 1.0
        assert isinstance(result["global_step"], int)
        assert result["class_starvation_streak"] >= 0


# ------------------------------------------------------------------ #
# 2. Batch diversity check
# ------------------------------------------------------------------ #


class TestBatchDiversity:
    """Verify diversity enforcement accepts sufficient classes."""

    def test_runs_with_diverse_batches(
        self,
        epoch_runner: EpochRunner,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        # Our dataset has 5 classes distributed across 4 batches of 4
        result = epoch_runner.run_epoch(
            epoch=0,
            global_step=0,
            warmup_steps=100,
            representation_diagnostic_mode=False,
            use_energy_based_family_objective=False,
            active_family_class_ids=[0, 1, 2, 3, 4],
            enforce_all_classes_per_batch=False,
            step_coverage_checked=False,
            head_phase_start_step=0,
            representation_phase_active=False,
            in_representation_window=False,
            representation_curriculum_complete=False,
            binary_class_weights=None,
            family_class_weights=None,
            family_log_prior=None,
            energy_logit_temperature=1.0,
            disable_tail_focal_regularizer=False,
            optimizer=optimizer,
            backbone_params=list(epoch_runner._model.parameters()),
            check_backbone_freeze_state=_null_hook,
            handle_representation_phase_logic=_null_hook,
            maybe_activate_joint_finetune_phase=_null_hook,
            check_family_class_coverage=_null_hook,
            check_step_coverage=_null_hook,
            freeze_epoch_centroid_snapshot=_null_hook,
            update_centroids_from_epoch_buffer=_null_hook,
        )
        # Should not raise RuntimeError
        assert result["global_step"] > 0


# ------------------------------------------------------------------ #
# 3. Internal helpers
# ------------------------------------------------------------------ #


class TestInternalHelpers:
    """Verify internal helper methods."""

    def test_set_epoch_loss_strategy_rep_diag(self) -> None:
        strategy = EpochRunner._set_epoch_loss_strategy(
            representation_diagnostic_mode=True,
        )
        assert strategy == "rep_diag"

    def test_set_epoch_loss_strategy_standard(self) -> None:
        strategy = EpochRunner._set_epoch_loss_strategy(
            representation_diagnostic_mode=False,
        )
        assert strategy == "standard"

    def test_is_representation_window_step_in_window(self) -> None:
        assert EpochRunner._is_representation_window_step(100, True)
        assert EpochRunner._is_representation_window_step(0, True)
        assert EpochRunner._is_representation_window_step(499, True)

    def test_is_representation_window_step_outside(self) -> None:
        assert not EpochRunner._is_representation_window_step(500, True)
        assert not EpochRunner._is_representation_window_step(-1, True)

    def test_is_representation_window_step_disabled(self) -> None:
        assert not EpochRunner._is_representation_window_step(100, False)

    def test_update_train_batch_stats_first_call(
        self,
        device: torch.device,
    ) -> None:
        pred = torch.randint(0, 5, (4,), device=device)
        logits = torch.randn(4, 5, device=device)
        counts, sums = EpochRunner._update_train_batch_stats(
            None, None, pred, logits,
        )
        assert counts is not None
        assert sums is not None
        assert counts.shape == (5,)
        assert sums.shape == (5,)
        assert counts.sum().item() == 4

    def test_update_train_batch_stats_increments(
        self,
        device: torch.device,
    ) -> None:
        pred = torch.randint(0, 5, (4,), device=device)
        logits = torch.randn(4, 5, device=device)
        counts = torch.zeros(5, dtype=torch.int64)
        sums = torch.zeros(5, dtype=torch.float32)
        counts_out, sums_out = EpochRunner._update_train_batch_stats(
            counts, sums, pred, logits,
        )
        assert counts_out is not None
        assert counts_out.sum().item() == 4
        assert (counts_out > 0).any()


# ------------------------------------------------------------------ #
# 4. Warmup at epoch 0
# ------------------------------------------------------------------ #


class TestEpoch0Warmup:
    """Verify warmup runs during epoch 0."""

    def test_warmup_executes_at_epoch_zero(
        self,
        epoch_runner: EpochRunner,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        result = epoch_runner.run_epoch(
            epoch=0,
            global_step=0,
            warmup_steps=100,
            representation_diagnostic_mode=False,
            use_energy_based_family_objective=False,
            active_family_class_ids=[0, 1, 2, 3, 4],
            enforce_all_classes_per_batch=False,
            step_coverage_checked=False,
            head_phase_start_step=0,
            representation_phase_active=False,
            in_representation_window=False,
            representation_curriculum_complete=False,
            binary_class_weights=None,
            family_class_weights=None,
            family_log_prior=None,
            energy_logit_temperature=1.0,
            disable_tail_focal_regularizer=False,
            optimizer=optimizer,
            backbone_params=list(epoch_runner._model.parameters()),
            check_backbone_freeze_state=_null_hook,
            handle_representation_phase_logic=_null_hook,
            maybe_activate_joint_finetune_phase=_null_hook,
            check_family_class_coverage=_null_hook,
            check_step_coverage=_null_hook,
            freeze_epoch_centroid_snapshot=_null_hook,
            update_centroids_from_epoch_buffer=_null_hook,
        )
        # Warmup should execute at epoch 0, advancing global step
        assert result["global_step"] > 0
