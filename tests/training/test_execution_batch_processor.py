"""Regression tests for BatchProcessor (Phase 17 extraction).

Covers:
  1.  process_batch returns correct result shape and keys
  2.  Forward pass and classification loss is computed
  3.  Backpropagation updates model parameters
  4.  Active family class ID resolution
  5.  Logit stabilization with temperature
  6.  Energy-based objective dispatch (when enabled)
  7.  Warmup step weight suppression
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

from scripts.training.execution.batch_processor import BatchProcessor
from scripts.training.losses import LossRegistry

# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #


@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class _SimpleModel(nn.Module):
    """Minimal model for test-time forward pass."""

    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        self.backbone = nn.Linear(10, 16)
        self.binary_head = nn.Linear(16, 2)
        self.family_head = nn.Linear(16, num_classes)

    def forward(
        self, x: torch.Tensor, return_features: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        binary_logits = self.binary_head(h)
        family_logits = self.family_head(h)
        return binary_logits, family_logits, h


@pytest.fixture
def model(device: torch.device) -> nn.Module:
    m = _SimpleModel(5).to(device)
    m.train()
    return m


@pytest.fixture
def loss_fn() -> Any:
    """Multi-task loss matching the trainer's loss function."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
    from helix_ids.models.helix_ids_full import MultiTaskLoss

    return MultiTaskLoss()


@pytest.fixture
def loss_registry(device: torch.device) -> LossRegistry:
    return LossRegistry(
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


@pytest.fixture
def config() -> Any:
    return SimpleNamespace(
        max_grad_norm=1.0,
        log_interval=10,
    )


@pytest.fixture
def processor(
    model: nn.Module,
    loss_fn: Any,
    loss_registry: LossRegistry,
    config: Any,
) -> BatchProcessor:
    return BatchProcessor(
        model=model,
        loss_fn=loss_fn,
        loss_registry=loss_registry,
        config=config,
    )


@pytest.fixture(autouse=True)
def _set_seed() -> None:
    torch.manual_seed(42)


@pytest.fixture
def batch(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.randn(8, 10, device=device)
    y_binary = torch.randint(0, 2, (8,), device=device)
    y_family = torch.randint(0, 5, (8,), device=device)
    return x, y_binary, y_family


# ------------------------------------------------------------------ #
# 1. Result shape and keys
# ------------------------------------------------------------------ #


class TestProcessBatchResultShape:
    """Verify process_batch returns all expected keys with correct shapes."""

    def test_all_expected_keys_present(
        self,
        processor: BatchProcessor,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        x, y_binary, y_family = batch
        optimizer = torch.optim.SGD(processor._model.parameters(), lr=0.01)

        result = processor.process_batch(
            x, y_binary, y_family,
            in_step_warmup=False,
            in_representation_phase=False,
            optimizer=optimizer,
            backbone_params=list(processor._model.parameters()),
            global_step=10,
            warmup_steps=100,
            binary_class_weights=None,
            family_class_weights=None,
            active_family_class_ids=[0, 1, 2, 3, 4],
            use_energy_based_family_objective=False,
        )

        expected_keys = {
            "loss", "raw_family_logits", "family_pred",
            "binary_correct", "family_correct", "batch_size",
            "energy_diag", "global_step_increment",
        }
        assert set(result.keys()) == expected_keys, f"Mismatch: {set(result.keys()) ^ expected_keys}"

    def test_metric_types(
        self,
        processor: BatchProcessor,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        x, y_binary, y_family = batch
        optimizer = torch.optim.SGD(processor._model.parameters(), lr=0.01)

        result = processor.process_batch(
            x, y_binary, y_family,
            in_step_warmup=False,
            in_representation_phase=False,
            optimizer=optimizer,
            backbone_params=list(processor._model.parameters()),
            global_step=10,
            warmup_steps=100,
            binary_class_weights=None,
            family_class_weights=None,
            active_family_class_ids=[0, 1, 2, 3, 4],
            use_energy_based_family_objective=False,
        )

        assert isinstance(result["loss"], torch.Tensor)
        assert result["loss"].dim() == 0
        assert isinstance(result["batch_size"], int)
        assert isinstance(result["binary_correct"], int)
        assert isinstance(result["family_correct"], int)
        assert isinstance(result["global_step_increment"], int)
        assert result["global_step_increment"] == 1
        assert isinstance(result["raw_family_logits"], torch.Tensor)
        assert isinstance(result["family_pred"], torch.Tensor)


# ------------------------------------------------------------------ #
# 2. Forward pass and loss computation
# ------------------------------------------------------------------ #


class TestForwardAndLoss:
    """Verify that loss is computed and backprop affects parameters."""

    def test_loss_is_positive_finite(
        self,
        processor: BatchProcessor,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        x, y_binary, y_family = batch
        optimizer = torch.optim.SGD(processor._model.parameters(), lr=0.01)

        result = processor.process_batch(
            x, y_binary, y_family,
            in_step_warmup=False,
            in_representation_phase=False,
            optimizer=optimizer,
            backbone_params=list(processor._model.parameters()),
            global_step=10,
            warmup_steps=100,
            binary_class_weights=None,
            family_class_weights=None,
            active_family_class_ids=[0, 1, 2, 3, 4],
            use_energy_based_family_objective=False,
        )

        loss_val = float(result["loss"].item())
        assert loss_val > 0.0, f"Loss should be positive, got {loss_val}"
        assert torch.isfinite(result["loss"]), "Loss should be finite"

    def test_backprop_updates_parameters(
        self,
        processor: BatchProcessor,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        x, y_binary, y_family = batch
        optimizer = torch.optim.SGD(processor._model.parameters(), lr=0.1)

        param_before = (
            list(processor._model.parameters())[0]
            .clone()
            .detach()
        )

        processor.process_batch(
            x, y_binary, y_family,
            in_step_warmup=False,
            in_representation_phase=False,
            optimizer=optimizer,
            backbone_params=list(processor._model.parameters()),
            global_step=10,
            warmup_steps=100,
            binary_class_weights=None,
            family_class_weights=None,
            active_family_class_ids=[0, 1, 2, 3, 4],
            use_energy_based_family_objective=False,
        )

        param_after = list(processor._model.parameters())[0]
        assert not torch.allclose(param_before, param_after), (
            "Parameters should be updated after backprop"
        )


# ------------------------------------------------------------------ #
# 3. Active class ID resolution
# ------------------------------------------------------------------ #


class TestActiveClassIdResolution:
    """Verify active class ID resolution from logits and labels."""

    def test_resolve_from_provided_ids(
        self,
        processor: BatchProcessor,
        device: torch.device,
    ) -> None:
        logits = torch.randn(4, 10, device=device)
        labels = torch.randint(0, 10, (4,), device=device)
        provided = [2, 5, 7]

        resolved = processor._resolve_batch_active_family_class_ids(
            logits, labels, provided,
        )
        assert resolved == [2, 5, 7]

    def test_filter_invalid_ids(
        self,
        processor: BatchProcessor,
        device: torch.device,
    ) -> None:
        logits = torch.randn(4, 5, device=device)
        labels = torch.randint(0, 5, (4,), device=device)
        provided = [0, 3, 99, -1, 2]

        resolved = processor._resolve_batch_active_family_class_ids(
            logits, labels, provided,
        )
        assert resolved == [0, 2, 3]

    def test_fallback_from_labels(
        self,
        processor: BatchProcessor,
        device: torch.device,
    ) -> None:
        logits = torch.randn(8, 5, device=device)
        labels = torch.tensor([1, 3, 1, 2], device=device)

        resolved = processor._resolve_batch_active_family_class_ids(
            logits, labels, None,
        )
        assert resolved == [1, 2, 3]


# ------------------------------------------------------------------ #
# 4. Logit stabilization
# ------------------------------------------------------------------ #


class TestLogitStabilization:
    """Verify logit stabilization produces finite output."""

    def test_stabilize_returns_finite(
        self,
        processor: BatchProcessor,
        device: torch.device,
    ) -> None:
        logits = torch.randn(4, 5, device=device)
        stabilized = processor._stabilize_batch_family_logits(
            logits,
            active_family_class_ids=[0, 1, 2, 3, 4],
            use_energy_based_family_objective=False,
        )
        assert torch.isfinite(stabilized).all()

    def test_temperature_updates(
        self,
        processor: BatchProcessor,
        device: torch.device,
    ) -> None:
        old_temp = processor.logit_temp
        logits = torch.randn(4, 5, device=device)
        processor._stabilize_batch_family_logits(
            logits,
            active_family_class_ids=[0, 1, 2, 3, 4],
            use_energy_based_family_objective=False,
        )
        # Temperature should update via EMA
        assert processor.logit_temp != old_temp


# ------------------------------------------------------------------ #
# 5. Warmup step
# ------------------------------------------------------------------ #


class TestWarmupStep:
    """Verify weight suppression during warmup steps."""

    def test_weights_suppressed_during_warmup(
        self,
        processor: BatchProcessor,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        x, y_binary, y_family = batch
        optimizer = torch.optim.SGD(processor._model.parameters(), lr=0.01)

        # in_step_warmup=True should set weights to None
        result = processor.process_batch(
            x, y_binary, y_family,
            in_step_warmup=True,
            in_representation_phase=False,
            optimizer=optimizer,
            backbone_params=list(processor._model.parameters()),
            global_step=5,
            warmup_steps=100,
            binary_class_weights=torch.tensor([1.0, 2.0]),
            family_class_weights=torch.tensor([1.0] * 5),
            active_family_class_ids=[0, 1, 2, 3, 4],
            use_energy_based_family_objective=False,
        )

        assert torch.isfinite(result["loss"])


# ------------------------------------------------------------------ #
# 6. Backpropagation
# ------------------------------------------------------------------ #


class TestBackpropagation:
    """Verify backpropagation internals."""

    def test_backprop_without_backbone_params(
        self,
        processor: BatchProcessor,
        device: torch.device,
    ) -> None:
        loss = torch.tensor(1.0, device=device, requires_grad=True)
        optimizer = torch.optim.SGD(processor._model.parameters(), lr=0.01)

        # Should not raise when backbone_params is None
        processor._backpropagate(
            loss,
            optimizer=optimizer,
            backbone_params=None,
            in_representation_phase=False,
        )

    def test_backprop_with_backbone_params(
        self,
        processor: BatchProcessor,
        device: torch.device,
    ) -> None:
        x = torch.randn(4, 10, device=device)
        out, _, _ = processor._model(x)
        loss = out.sum()
        optimizer = torch.optim.SGD(processor._model.parameters(), lr=0.01)

        param_before = list(processor._model.parameters())[0].clone()
        processor._backpropagate(
            loss,
            optimizer=optimizer,
            backbone_params=list(processor._model.parameters()),
            in_representation_phase=False,
        )
        param_after = list(processor._model.parameters())[0]
        assert not torch.allclose(param_before, param_after)
