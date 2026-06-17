"""Regression tests for WarmupManager (Phase 17 extraction).

Covers:
  1.  Early return when epoch != 0
  2.  Early return when model not training
  3.  Class-to-indices collection from datasets
  4.  Active class ID resolution
  5.  Warmup batch tensor construction
  6.  Logit control application
  7.  Full warmup forward pass and optimizer step
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from scripts.training.execution.warmup_manager import WarmupManager

# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #


@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class _SimpleFamilyModel(nn.Module):
    """Minimal model with a family_head attribute."""

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
    log_interval: int = 10
    max_grad_norm: float = 1.0


class _SimpleDataset(Dataset):
    """Dataset with family_labels for warmup class discovery."""

    def __init__(self, num_samples: int = 20, num_classes: int = 5) -> None:
        self.num_samples = num_samples
        self.num_classes = num_classes
        self._x = torch.randn(num_samples, 10)
        self._family_labels = torch.randint(0, num_classes, (num_samples,))

    @property
    def family_labels(self) -> torch.Tensor:
        return self._family_labels

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        y_binary = 1 if int(self._family_labels[idx]) >= (self.num_classes // 2) else 0
        return self._x[idx], y_binary, int(self._family_labels[idx])


@pytest.fixture
def train_dataset() -> Dataset[Any]:
    return _SimpleDataset(num_samples=32, num_classes=5)


@pytest.fixture
def model(device: torch.device) -> nn.Module:
    m = _SimpleFamilyModel(num_classes=5).to(device)
    m.train()
    return m


@pytest.fixture
def loss_fn() -> Any:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
    from helix_ids.models.helix_ids_full import MultiTaskLoss

    return MultiTaskLoss()


@pytest.fixture
def config() -> Any:
    return _FakeConfig()


@pytest.fixture
def warmup_manager(
    model: nn.Module,
    loss_fn: Any,
    config: Any,
    device: torch.device,
) -> WarmupManager:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    return WarmupManager(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        config=config,
    )


@pytest.fixture(autouse=True)
def _set_seed() -> None:
    torch.manual_seed(42)


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _dummy_logger() -> Any:
    """A minimal logger that swallows everything."""

    class _SilentLogger:
        def info(self, *args: Any, **kwargs: Any) -> None: ...
        def warning(self, *args: Any, **kwargs: Any) -> None: ...
        def error(self, *args: Any, **kwargs: Any) -> None: ...

    return _SilentLogger()


# ------------------------------------------------------------------ #
# 1. Early return conditions
# ------------------------------------------------------------------ #


class TestEarlyReturn:
    """Verify warmup returns immediately when conditions aren't met."""

    def test_skipped_when_epoch_not_zero(
        self,
        warmup_manager: WarmupManager,
        train_dataset: Dataset[Any],
    ) -> None:
        result = warmup_manager.run_warmup(
            train_dataset,
            epoch=1,
            model_training=True,
            global_step=0,
            warmup_steps=100,
            active_family_class_ids=None,
            use_energy_based_family_objective=False,
            binary_class_weights=None,
            family_class_weights=None,
            backbone_params=None,
            logger=_dummy_logger(),
        )
        assert not result["warmup_executed"]
        assert result["global_step_increment"] == 0

    def test_skipped_when_model_not_training(
        self,
        warmup_manager: WarmupManager,
        train_dataset: Dataset[Any],
    ) -> None:
        result = warmup_manager.run_warmup(
            train_dataset,
            epoch=0,
            model_training=False,
            global_step=0,
            warmup_steps=100,
            active_family_class_ids=None,
            use_energy_based_family_objective=False,
            binary_class_weights=None,
            family_class_weights=None,
            backbone_params=None,
            logger=_dummy_logger(),
        )
        assert not result["warmup_executed"]
        assert result["global_step_increment"] == 0


# ------------------------------------------------------------------ #
# 2. Class-to-indices collection
# ------------------------------------------------------------------ #


class TestCollectClassToIndices:
    """Verify dataset class index mapping."""

    def test_returns_all_classes(
        self,
    ) -> None:
        dataset = _SimpleDataset(num_samples=50, num_classes=5)
        mapping = WarmupManager._collect_class_to_indices(dataset)
        assert set(mapping.keys()) == {0, 1, 2, 3, 4}
        total = sum(len(v) for v in mapping.values())
        assert total == 50

    def test_empty_dataset_returns_empty_dicts(
        self,
    ) -> None:
        dataset = _SimpleDataset(num_samples=0, num_classes=5)
        mapping = WarmupManager._collect_class_to_indices(dataset)
        # With 0 samples, family_labels is still a 0-element tensor
        assert sum(len(v) for v in mapping.values()) == 0


# ------------------------------------------------------------------ #
# 3. Active class ID resolution
# ------------------------------------------------------------------ #


class TestResolveWarmupActiveClassIds:
    """Verify active class ID resolution for warmup."""

    def test_uses_provided_ids_when_given(
        self,
    ) -> None:
        class_to_indices = {0: [1, 2], 1: [3], 2: [4], 3: [5], 4: [6]}
        resolved = WarmupManager._resolve_warmup_active_class_ids(
            class_to_indices, class_count=10, active_family_class_ids=[1, 3, 5],
        )
        assert resolved == [1, 3, 5]

    def test_filters_invalid_ids(
        self,
    ) -> None:
        class_to_indices = {0: [1], 1: [2], 2: [3]}
        resolved = WarmupManager._resolve_warmup_active_class_ids(
            class_to_indices, class_count=5, active_family_class_ids=[0, 99, -1, 2],
        )
        assert resolved == [0, 2]

    def test_falls_back_to_dataset_classes(
        self,
    ) -> None:
        class_to_indices = {1: [0], 3: [1], 4: [2]}
        resolved = WarmupManager._resolve_warmup_active_class_ids(
            class_to_indices,
            class_count=10,
            active_family_class_ids=None,
        )
        assert resolved == [1, 3, 4]

    def test_raises_when_empty(
        self,
    ) -> None:
        with pytest.raises(RuntimeError, match="No active classes"):
            WarmupManager._resolve_warmup_active_class_ids(
                {}, class_count=5, active_family_class_ids=[],
            )


# ------------------------------------------------------------------ #
# 4. Warmup batch tensor construction
# ------------------------------------------------------------------ #


class TestBuildWarmupBatchTensors:
    """Verify warmup batch tensor shapes and types."""

    def test_returns_correct_shapes(
        self,
    ) -> None:
        dataset = _SimpleDataset(num_samples=20, num_classes=5)
        forced_indices = [0, 5, 10, 15]
        x, y_binary, y_family, y_rows = (
            WarmupManager._build_warmup_batch_tensors(dataset, forced_indices)
        )
        assert x.shape == (4, 10)
        assert y_binary.shape == (4,)
        assert y_family.shape == (4,)
        assert len(y_rows) == 4
        assert y_binary.dtype == torch.long
        assert y_family.dtype == torch.long

    def test_handles_single_item(
        self,
    ) -> None:
        dataset = _SimpleDataset(num_samples=5, num_classes=5)
        forced_indices = [2]
        x, y_binary, y_family, y_rows = (
            WarmupManager._build_warmup_batch_tensors(dataset, forced_indices)
        )
        assert x.shape == (1, 10)
        assert len(y_rows) == 1


# ------------------------------------------------------------------ #
# 5. Logit control
# ------------------------------------------------------------------ #


class TestApplyWarmupLogitControls:
    """Verify logit control application."""

    def test_returns_clone_shape(
        self,
        device: torch.device,
    ) -> None:
        logits = torch.randn(4, 5, device=device)
        controlled = WarmupManager._apply_warmup_logit_controls(
            logits,
            active_class_ids=[0, 1, 2, 3, 4],
            use_energy_based_family_objective=False,
        )
        assert controlled.shape == logits.shape
        assert torch.isfinite(controlled).all()

    def test_energy_division(
        self,
        device: torch.device,
    ) -> None:
        logits = torch.randn(4, 5, device=device)
        controlled = WarmupManager._apply_warmup_logit_controls(
            logits,
            active_class_ids=[0, 1, 2, 3, 4],
            use_energy_based_family_objective=True,
        )
        # With energy enabled, logits are divided by temperature (default 1.0)
        assert controlled.shape == logits.shape


# ------------------------------------------------------------------ #
# 6. Full warmup execution
# ------------------------------------------------------------------ #


class TestFullWarmupExecution:
    """Verify the warmup forward + backprop executes end-to-end."""

    def test_warmup_produces_valid_result(
        self,
        warmup_manager: WarmupManager,
        train_dataset: Dataset[Any],
    ) -> None:
        params_before = (
            list(warmup_manager._model.parameters())[0]
            .clone()
            .detach()
        )

        result = warmup_manager.run_warmup(
            train_dataset,
            epoch=0,
            model_training=True,
            global_step=0,
            warmup_steps=100,
            active_family_class_ids=None,
            use_energy_based_family_objective=False,
            binary_class_weights=None,
            family_class_weights=None,
            backbone_params=list(warmup_manager._model.parameters()),
            logger=_dummy_logger(),
        )
        assert result["warmup_executed"]
        assert result["global_step_increment"] == 1
        assert len(result["active_class_ids"]) > 0

        params_after = list(warmup_manager._model.parameters())[0]
        assert not torch.allclose(params_before, params_after), (
            "Parameters should be updated after warmup backprop"
        )
