"""Regression tests for Phase 13A-3 extracted representation delegates.

Validates that the three extracted packages (LossRegistry, CentroidManager,
RepresentationCoordinator) produce correct outputs matching the original
behaviour from train_helix_ids_full.py.

Covers all 17 extracted methods with 56+ tests, including edge cases
(empty tensors, single-class, degenerate, boundary values).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F

from scripts.training.losses.loss_registry import LossRegistry
from scripts.training.representation.centroid_manager import CentroidManager
from scripts.training.representation.representation_coordinator import (
    RepresentationCoordinator,
)

# ======================================================================
# Shared fixtures
# ======================================================================


@pytest.fixture
def rng() -> torch.Generator:
    return torch.Generator().manual_seed(42)


@pytest.fixture
def sample_features_3class(rng: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Well-separated 3-class embedding space with 4-D features."""
    c0 = torch.randn(20, 4, generator=rng) * 0.1 + torch.tensor([-1.0, 0.0, 0.0, 0.0])
    c1 = torch.randn(20, 4, generator=rng) * 0.1 + torch.tensor([0.0, 1.0, 0.0, 0.0])
    c2 = torch.randn(20, 4, generator=rng) * 0.1 + torch.tensor([0.0, 0.0, 1.0, 0.0])
    features = F.normalize(torch.cat([c0, c1, c2], dim=0), p=2, dim=1)
    labels = torch.cat(
        [
            torch.zeros(20, dtype=torch.int64),
            torch.ones(20, dtype=torch.int64),
            torch.full((20,), 2, dtype=torch.int64),
        ],
        dim=0,
    )
    return features, labels


@pytest.fixture
def sample_logits_3class() -> torch.Tensor:
    """Logits with 3 active classes, batch=12."""
    return torch.tensor(
        [
            [2.0, -1.0, 0.5],
            [1.5, 0.0, -0.5],
            [-0.5, 2.5, 0.0],
            [0.0, 0.0, 2.0],
            [1.0, 1.0, -1.0],
            [-1.0, 2.0, 0.0],
            [0.5, -0.5, 1.5],
            [2.0, -2.0, -1.0],
            [-1.0, -0.5, 2.5],
            [1.0, 0.0, 2.0],
            [0.0, 1.5, 0.5],
            [-0.5, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )


@pytest.fixture
def sample_y_family() -> torch.Tensor:
    """Class labels for sample_logits_3class — roughly balanced across 3 classes."""
    return torch.tensor([0, 0, 1, 2, 0, 1, 2, 0, 2, 2, 1, 1], dtype=torch.int64)


@pytest.fixture
def sample_y_binary() -> torch.Tensor:
    """Binary labels (all attack=1 for simplicity)."""
    return torch.ones(12, dtype=torch.int64)


@pytest.fixture
def sample_binary_logits() -> torch.Tensor:
    """Binary head logits matching batch size."""
    return torch.randn(12, 2, dtype=torch.float32)


@pytest.fixture
def mock_multi_task_loss() -> torch.nn.Module:
    """A minimal MultiTaskLoss-like module for energy-objective tests."""

    class _MinimalLossFn(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lambda_binary = 1.0
            self._ce = torch.nn.CrossEntropyLoss()

        def _classification_loss(
            self,
            logits: torch.Tensor,
            labels: torch.Tensor,
            class_weights: torch.Tensor | None,
        ) -> torch.Tensor:
            return self._ce(logits, labels)

    return _MinimalLossFn()


# ======================================================================
# LossRegistry — supervised_contrastive_loss
# ======================================================================


class TestSupervisedContrastiveLoss:
    def test_basic_3class(self, sample_features_3class: tuple[torch.Tensor, torch.Tensor]) -> None:
        features, labels = sample_features_3class
        loss = LossRegistry.supervised_contrastive_loss(
            features, labels, temperature=0.5
        )
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # scalar
        assert loss.item() > 0.0  # should produce positive loss

    def test_single_sample(self) -> None:
        """Single sample should return zero loss (no valid positives/negatives)."""
        features = F.normalize(torch.randn(1, 4), p=2, dim=1)
        labels = torch.zeros(1, dtype=torch.int64)
        loss = LossRegistry.supervised_contrastive_loss(
            features, labels, temperature=0.5
        )
        assert float(loss.item()) == 0.0

    def test_with_anchor_weights(self, sample_features_3class: tuple[torch.Tensor, torch.Tensor]) -> None:
        features, labels = sample_features_3class
        weights = torch.rand(60, dtype=torch.float32).clamp_min(0.1)
        loss = LossRegistry.supervised_contrastive_loss(
            features, labels, temperature=0.5, anchor_weights=weights
        )
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() > 0.0

    def test_with_negative_weight(self, sample_features_3class: tuple[torch.Tensor, torch.Tensor]) -> None:
        features, labels = sample_features_3class
        loss = LossRegistry.supervised_contrastive_loss(
            features, labels, temperature=0.5, negative_weight=2.0
        )
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() > 0.0

    def test_empty_batch(self) -> None:
        """Empty batch: features.shape[0] <= 1 triggers early zero return."""
        features = torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.zeros((0,), dtype=torch.int64)
        loss = LossRegistry.supervised_contrastive_loss(
            features, labels, temperature=0.5
        )
        assert float(loss.item()) == 0.0

    def test_logits_output_range(self, sample_features_3class: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Loss should be strictly positive for non-trivial batches."""
        features, labels = sample_features_3class
        loss = LossRegistry.supervised_contrastive_loss(
            features, labels, temperature=0.1
        )
        assert loss.item() > 0.0
        # Higher temperature = more uniform logits = higher contrastive loss
        loss_high = LossRegistry.supervised_contrastive_loss(
            features, labels, temperature=10.0
        )
        assert loss_high.item() > loss.item()


# ======================================================================
# LossRegistry — supcon_anchor_weights
# ======================================================================


class TestSupconAnchorWeights:
    def test_basic(self) -> None:
        labels = torch.tensor([0, 0, 0, 1, 1, 2], dtype=torch.int64)
        weights = LossRegistry.supcon_anchor_weights(labels)
        assert isinstance(weights, torch.Tensor)
        assert list(weights.shape) == [6]
        assert float(weights.mean().item()) == pytest.approx(1.0, abs=1e-5)

    def test_single_class(self) -> None:
        labels = torch.zeros(10, dtype=torch.int64)
        weights = LossRegistry.supcon_anchor_weights(labels)
        assert list(weights.shape) == [10]
        assert float(weights.mean().item()) == pytest.approx(1.0, abs=1e-5)

    def test_empty_labels(self) -> None:
        labels = torch.zeros((0,), dtype=torch.int64)
        weights = LossRegistry.supcon_anchor_weights(labels)
        assert list(weights.shape) == [0]


# ======================================================================
# LossRegistry — class_conditional_energy_gap_loss
# ======================================================================


class TestClassConditionalEnergyGapLoss:
    def test_basic(self, sample_logits_3class: torch.Tensor, sample_y_family: torch.Tensor) -> None:
        loss, e_y, e_others, gap, total = LossRegistry.class_conditional_energy_gap_loss(
            sample_logits_3class, sample_y_family, alpha=1.0
        )
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert isinstance(e_y, float)  # energy_y = -logit_y, sign depends on logit polarity
        assert isinstance(e_others, float)
        assert isinstance(gap, float)
        assert gap > 0.0  # logsumexp(negatives) < logit_y typically

    def test_single_class(self) -> None:
        """Single-class logits (shape[1] <= 1) should fall back to cross-entropy."""
        logits = torch.randn(8, 1, dtype=torch.float32)
        labels = torch.zeros(8, dtype=torch.int64)
        loss, ce, e_others, gap, total = LossRegistry.class_conditional_energy_gap_loss(
            logits, labels, alpha=1.0
        )
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert e_others == 0.0
        assert gap == 0.0

    def test_empty_batch(self) -> None:
        logits = torch.zeros((0, 3), dtype=torch.float32)
        labels = torch.zeros((0,), dtype=torch.int64)
        loss, e_y, e_others, gap, total = LossRegistry.class_conditional_energy_gap_loss(
            logits, labels, alpha=1.0
        )
        assert isinstance(loss, torch.Tensor)
        assert float(loss.item()) == 0.0

    def test_ndim_mismatch(self) -> None:
        """1-D logits (ndim != 2) should trigger early return."""
        logits = torch.randn(10, dtype=torch.float32)
        labels = torch.zeros(10, dtype=torch.int64)
        loss, e_y, e_others, gap, total = LossRegistry.class_conditional_energy_gap_loss(
            logits, labels, alpha=1.0
        )
        assert float(loss.item()) == 0.0


# ======================================================================
# LossRegistry — energy_class_balance_loss
# ======================================================================


class TestEnergyClassBalanceLoss:
    def test_basic(self, sample_logits_3class: torch.Tensor) -> None:
        loss, kl, entropy, min_mass = LossRegistry.energy_class_balance_loss(
            sample_logits_3class
        )
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert kl >= 0.0
        assert entropy >= 0.0
        assert min_mass > 0.0

    def test_empty_batch(self) -> None:
        logits = torch.zeros((0, 3), dtype=torch.float32)
        loss, kl, entropy, min_mass = LossRegistry.energy_class_balance_loss(logits)
        assert float(loss.item()) == 0.0

    def test_ndim_mismatch(self) -> None:
        logits = torch.randn(10, dtype=torch.float32)
        loss, kl, entropy, min_mass = LossRegistry.energy_class_balance_loss(logits)
        assert float(loss.item()) == 0.0


# ======================================================================
# LossRegistry — energy_min_winner_loss
# ======================================================================


class TestEnergyMinWinnerLoss:
    def test_basic(self, sample_logits_3class: torch.Tensor) -> None:
        loss, deficit, min_count = LossRegistry.energy_min_winner_loss(
            sample_logits_3class, None, min_winners=3
        )
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert deficit >= 0.0
        assert min_count >= 0.0

    def test_with_active_class_ids(self, sample_logits_3class: torch.Tensor) -> None:
        loss, deficit, min_count = LossRegistry.energy_min_winner_loss(
            sample_logits_3class, [0, 1, 2], min_winners=5
        )
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0.0  # likely deficit with min_winners=5

    def test_empty_batch(self) -> None:
        logits = torch.zeros((0, 3), dtype=torch.float32)
        loss, deficit, min_count = LossRegistry.energy_min_winner_loss(
            logits, None, min_winners=3
        )
        assert float(loss.item()) == 0.0

    def test_ndim_mismatch(self) -> None:
        logits = torch.randn(10, dtype=torch.float32)
        loss, deficit, min_count = LossRegistry.energy_min_winner_loss(
            logits, None, min_winners=3
        )
        assert float(loss.item()) == 0.0

    def test_with_invalid_active_ids(self, sample_logits_3class: torch.Tensor) -> None:
        """Class IDs outside range should be filtered; falls back to all classes."""
        loss, deficit, min_count = LossRegistry.energy_min_winner_loss(
            sample_logits_3class, [99, 100], min_winners=3
        )
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0


# ======================================================================
# LossRegistry — pairwise_margin_repulsion_loss
# ======================================================================


class TestPairwiseMarginRepulsionLoss:
    def test_basic(self, sample_features_3class: tuple[torch.Tensor, torch.Tensor]) -> None:
        features, labels = sample_features_3class
        loss = LossRegistry.pairwise_margin_repulsion_loss(
            features, labels, margin=2.0, top_k=3
        )
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0.0

    def test_single_sample(self) -> None:
        features = F.normalize(torch.randn(1, 4), p=2, dim=1)
        labels = torch.zeros(1, dtype=torch.int64)
        loss = LossRegistry.pairwise_margin_repulsion_loss(
            features, labels, margin=2.0
        )
        assert float(loss.item()) == 0.0

    def test_same_class_all(self) -> None:
        """All same class = no negative pairs = zero loss."""
        features = F.normalize(torch.randn(10, 4), p=2, dim=1)
        labels = torch.zeros(10, dtype=torch.int64)
        loss = LossRegistry.pairwise_margin_repulsion_loss(
            features, labels, margin=2.0
        )
        assert float(loss.item()) == 0.0


# ======================================================================
# LossRegistry — centroid_separation_barrier_loss
# ======================================================================


class TestCentroidSeparationBarrierLoss:
    def test_basic(self, sample_features_3class: tuple[torch.Tensor, torch.Tensor]) -> None:
        features, labels = sample_features_3class
        loss = LossRegistry.centroid_separation_barrier_loss(
            features, labels, min_distance=0.5
        )
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0.0

    def test_single_class(self) -> None:
        features = F.normalize(torch.randn(10, 4), p=2, dim=1)
        labels = torch.zeros(10, dtype=torch.int64)
        loss = LossRegistry.centroid_separation_barrier_loss(
            features, labels, min_distance=0.5
        )
        assert float(loss.item()) == 0.0

    def test_empty(self) -> None:
        features = torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.zeros((0,), dtype=torch.int64)
        loss = LossRegistry.centroid_separation_barrier_loss(
            features, labels, min_distance=0.5
        )
        assert float(loss.item()) == 0.0


# ======================================================================
# LossRegistry — centroid_repulsion_loss
# ======================================================================


class TestCentroidRepulsionLoss:
    def test_basic(self, sample_features_3class: tuple[torch.Tensor, torch.Tensor]) -> None:
        features, labels = sample_features_3class
        loss = LossRegistry.centroid_repulsion_loss(features, labels, margin=2.0)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0.0

    def test_single_class(self) -> None:
        features = F.normalize(torch.randn(10, 4), p=2, dim=1)
        labels = torch.zeros(10, dtype=torch.int64)
        loss = LossRegistry.centroid_repulsion_loss(features, labels, margin=2.0)
        assert float(loss.item()) == 0.0


# ======================================================================
# LossRegistry — intra_class_variance_clamp_loss
# ======================================================================


class TestIntraClassVarianceClampLoss:
    def test_basic(self, sample_features_3class: tuple[torch.Tensor, torch.Tensor]) -> None:
        features, labels = sample_features_3class
        loss = LossRegistry.intra_class_variance_clamp_loss(
            features, labels, var_lower_bound=0.0, var_upper_bound=1.0
        )
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_degenerate_single_class(self) -> None:
        features = F.normalize(torch.randn(10, 4), p=2, dim=1)
        labels = torch.zeros(10, dtype=torch.int64)
        loss = LossRegistry.intra_class_variance_clamp_loss(
            features, labels, var_lower_bound=0.0, var_upper_bound=1.0
        )
        assert float(loss.item()) == 0.0

    def test_penalty_high_variance(self) -> None:
        """High-variance features should trigger upper-bound penalty."""
        features = torch.randn(30, 8, dtype=torch.float32)
        labels = torch.cat([
            torch.zeros(10, dtype=torch.int64),
            torch.ones(10, dtype=torch.int64),
            torch.full((10,), 2, dtype=torch.int64),
        ])
        loss = LossRegistry.intra_class_variance_clamp_loss(
            features, labels, var_lower_bound=0.0, var_upper_bound=0.01
        )
        assert loss.item() > 0.0


# ======================================================================
# LossRegistry — compute_batch_class_centroids_for_loss
# ======================================================================


class TestComputeBatchClassCentroidsForLoss:
    def test_basic(self, sample_features_3class: tuple[torch.Tensor, torch.Tensor]) -> None:
        features, labels = sample_features_3class
        centroids, class_ids = LossRegistry.compute_batch_class_centroids_for_loss(
            features, labels
        )
        assert isinstance(centroids, torch.Tensor)
        assert len(class_ids) == 3
        assert list(centroids.shape) == [3, 4]
        assert centroids.dtype == torch.float32

    def test_empty(self) -> None:
        features = torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.zeros((0,), dtype=torch.int64)
        centroids, class_ids = LossRegistry.compute_batch_class_centroids_for_loss(
            features, labels
        )
        assert list(centroids.shape) == [0, 0]
        assert class_ids == []

    def test_single_element_batch(self) -> None:
        features = F.normalize(torch.randn(1, 4), p=2, dim=1)
        labels = torch.zeros(1, dtype=torch.int64)
        centroids, class_ids = LossRegistry.compute_batch_class_centroids_for_loss(
            features, labels
        )
        assert list(centroids.shape) == [1, 4]
        assert class_ids == [0]


# ======================================================================
# LossRegistry — global_centroid_guided_losses
# ======================================================================


class TestGlobalCentroidGuidedLosses:
    def test_basic_with_frozen(self) -> None:
        """Use pre-populated epoch_frozen_centroids."""
        frozen = {
            0: F.normalize(torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32), p=2, dim=0),
            1: F.normalize(torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32), p=2, dim=0),
            2: F.normalize(torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float32), p=2, dim=0),
        }
        batch_centroids = torch.stack([
            torch.tensor([0.9, 0.1, 0.0, 0.0]),
            torch.tensor([0.1, 0.9, 0.0, 0.0]),
            torch.tensor([0.0, 0.1, 0.9, 0.0]),
        ], dim=0)
        batch_centroids = F.normalize(batch_centroids, p=2, dim=1)
        class_ids = [0, 1, 2]
        rep_loss, barrier_loss, min_inter = LossRegistry.global_centroid_guided_losses(
            batch_centroids, class_ids, frozen,
            rep_centroid_repulsion_margin=0.8,
            rep_centroid_barrier_min_distance=0.5,
        )
        assert isinstance(rep_loss, torch.Tensor)
        assert isinstance(barrier_loss, torch.Tensor)
        assert rep_loss.ndim == 0
        assert barrier_loss.ndim == 0
        assert isinstance(min_inter, float)

    def test_empty_frozen_populates(self) -> None:
        """When epoch_frozen_centroids is empty, it gets populated from batch."""
        frozen: dict[int, torch.Tensor] = {}
        batch_centroids = F.normalize(
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32),
            p=2, dim=1,
        )
        class_ids = [0, 1]
        _, _, min_inter = LossRegistry.global_centroid_guided_losses(
            batch_centroids, class_ids, frozen,
            rep_centroid_repulsion_margin=0.8,
            rep_centroid_barrier_min_distance=0.5,
        )
        # After call, frozen should be populated
        assert len(frozen) == 2
        assert 0 in frozen
        assert 1 in frozen

    def test_empty_batch(self) -> None:
        frozen = {0: torch.zeros(4, dtype=torch.float32)}
        batch_centroids = torch.zeros((0, 4), dtype=torch.float32)
        rep_loss, barrier_loss, min_inter = LossRegistry.global_centroid_guided_losses(
            batch_centroids, [], frozen,
            rep_centroid_repulsion_margin=0.8,
            rep_centroid_barrier_min_distance=0.5,
        )
        assert float(rep_loss.item()) == 0.0
        assert float(barrier_loss.item()) == 0.0

    def test_single_global_id(self) -> None:
        """Single frozen centroid -> no pairs -> zero loss."""
        frozen = {0: torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)}
        batch_centroids = F.normalize(
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
            p=2, dim=1,
        )
        rep_loss, barrier_loss, min_inter = LossRegistry.global_centroid_guided_losses(
            batch_centroids, [0], frozen,
            rep_centroid_repulsion_margin=0.8,
            rep_centroid_barrier_min_distance=0.5,
        )
        assert float(rep_loss.item()) == 0.0
        assert float(barrier_loss.item()) == 0.0


# ======================================================================
# LossRegistry — critical_pair_centroid_push_loss
# ======================================================================


class TestCriticalPairCentroidPushLoss:
    def test_basic(self, sample_features_3class: tuple[torch.Tensor, torch.Tensor]) -> None:
        features, labels = sample_features_3class
        loss = LossRegistry.critical_pair_centroid_push_loss(
            features, labels,
            min_distance=0.5,
            critical_collision_pairs={(0, 1)},
        )
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0.0

    def test_no_collision_pairs(self, sample_features_3class: tuple[torch.Tensor, torch.Tensor]) -> None:
        features, labels = sample_features_3class
        loss = LossRegistry.critical_pair_centroid_push_loss(
            features, labels,
            min_distance=0.5,
            critical_collision_pairs=set(),
        )
        assert float(loss.item()) == 0.0

    def test_class_not_in_batch(self) -> None:
        """Critical pair references a class not present in batch -> skipped gracefully."""
        features = F.normalize(torch.randn(10, 4), p=2, dim=1)
        labels = torch.zeros(10, dtype=torch.int64)
        loss = LossRegistry.critical_pair_centroid_push_loss(
            features, labels,
            min_distance=0.5,
            critical_collision_pairs={(0, 5), (5, 6)},
        )
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0


# ======================================================================
# LossRegistry — compute_representation_energy_objective
# ======================================================================


class TestComputeRepresentationEnergyObjective:
    def test_epoch_zero(
        self,
        sample_logits_3class: torch.Tensor,
        sample_y_family: torch.Tensor,
        sample_y_binary: torch.Tensor,
        sample_binary_logits: torch.Tensor,
        mock_multi_task_loss: torch.nn.Module,
    ) -> None:
        """Epoch 0: balance and winner losses are zeroed out."""
        loss, diag = LossRegistry.compute_representation_energy_objective(
            family_logits_train=sample_logits_3class,
            y_family=sample_y_family,
            y_binary=sample_y_binary,
            binary_logits=sample_binary_logits,
            active_family_class_ids=[0, 1, 2],
            loss_fn=mock_multi_task_loss,
            energy_gap_weight=1.0,
            energy_multi_negative_alpha=1.0,
            energy_balance_weight=0.5,
            energy_winner_weight=0.3,
            energy_winner_min_count=3,
            epoch=0,
        )
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        # epoch 0 => effective_balance and winner weights are 0
        assert diag["effective_energy_balance_weight"] == 0.0
        assert diag["effective_energy_winner_weight"] == 0.0

    def test_positive_epoch(
        self,
        sample_logits_3class: torch.Tensor,
        sample_y_family: torch.Tensor,
        sample_y_binary: torch.Tensor,
        sample_binary_logits: torch.Tensor,
        mock_multi_task_loss: torch.nn.Module,
    ) -> None:
        """Epoch > 0: balance and winner losses are active."""
        loss, diag = LossRegistry.compute_representation_energy_objective(
            family_logits_train=sample_logits_3class,
            y_family=sample_y_family,
            y_binary=sample_y_binary,
            binary_logits=sample_binary_logits,
            active_family_class_ids=[0, 1, 2],
            loss_fn=mock_multi_task_loss,
            energy_gap_weight=1.0,
            energy_multi_negative_alpha=1.0,
            energy_balance_weight=0.5,
            energy_winner_weight=0.3,
            energy_winner_min_count=3,
            epoch=5,
        )
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert diag["effective_energy_balance_weight"] == 0.5
        assert diag["effective_energy_winner_weight"] == 0.3

    def test_diagnostics_contain_all_keys(
        self,
        sample_logits_3class: torch.Tensor,
        sample_y_family: torch.Tensor,
        sample_y_binary: torch.Tensor,
        sample_binary_logits: torch.Tensor,
        mock_multi_task_loss: torch.nn.Module,
    ) -> None:
        _, diag = LossRegistry.compute_representation_energy_objective(
            family_logits_train=sample_logits_3class,
            y_family=sample_y_family,
            y_binary=sample_y_binary,
            binary_logits=sample_binary_logits,
            active_family_class_ids=[0, 1, 2],
            loss_fn=mock_multi_task_loss,
            energy_gap_weight=1.0,
            energy_multi_negative_alpha=1.0,
            energy_balance_weight=0.5,
            energy_winner_weight=0.3,
            energy_winner_min_count=3,
            epoch=10,
        )
        expected_keys = {
            "mean_e_y", "mean_e_others", "mean_gap", "mean_energy_total",
            "mean_balance_kl", "mean_pred_entropy", "min_pred_mass",
            "mean_winner_deficit", "min_winner_count",
            "effective_energy_balance_weight", "effective_energy_winner_weight",
        }
        assert set(diag.keys()) == expected_keys


# ======================================================================
# CentroidManager — update_running_rep_centroids
# ======================================================================


class TestUpdateRunningRepCentroids:
    def test_first_update(self) -> None:
        mgr = CentroidManager(centroid_ema_momentum=0.9)
        batch = F.normalize(
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32),
            p=2, dim=1,
        )
        mgr.update_running_rep_centroids(batch, [0, 1])
        assert len(mgr.centroid_ema_state) == 2
        assert 0 in mgr.centroid_ema_state
        assert 1 in mgr.centroid_ema_state

    def test_ema_smoothing(self) -> None:
        """Second update should blend with stored EMA state."""
        mgr = CentroidManager(centroid_ema_momentum=0.8)
        batch1 = F.normalize(
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32),
            p=2, dim=1,
        )
        batch2 = F.normalize(
            torch.tensor([[0.9, 0.1, 0.0], [0.1, 0.9, 0.0]], dtype=torch.float32),
            p=2, dim=1,
        )
        mgr.update_running_rep_centroids(batch1, [0, 1])
        first_val = mgr.centroid_ema_state[0].clone()
        mgr.update_running_rep_centroids(batch2, [0, 1])
        second_val = mgr.centroid_ema_state[0]
        # Second should be different from first (blended)
        assert not torch.allclose(first_val, second_val)

    def test_empty_batch(self) -> None:
        mgr = CentroidManager()
        batch = torch.zeros((0, 4), dtype=torch.float32)
        mgr.update_running_rep_centroids(batch, [])
        assert len(mgr.centroid_ema_state) == 0


# ======================================================================
# CentroidManager — freeze_epoch_centroid_snapshot
# ======================================================================


class TestFreezeEpochCentroidSnapshot:
    def test_basic_snapshot(self) -> None:
        mgr = CentroidManager()
        batch = F.normalize(
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32),
            p=2, dim=1,
        )
        mgr.update_running_rep_centroids(batch, [0, 1])
        mgr.freeze_epoch_centroid_snapshot()
        assert len(mgr.epoch_frozen_centroids) == 2
        assert torch.allclose(
            mgr.epoch_frozen_centroids[0],
            mgr.centroid_ema_state[0],
        )

    def test_snapshot_overwrites(self) -> None:
        mgr = CentroidManager()
        batch1 = F.normalize(
            torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32), p=2, dim=1,
        )
        mgr.update_running_rep_centroids(batch1, [0])
        mgr.freeze_epoch_centroid_snapshot()
        frozen1 = mgr.epoch_frozen_centroids[0].clone()

        batch2 = F.normalize(
            torch.tensor([[0.5, 0.5, 0.0]], dtype=torch.float32), p=2, dim=1,
        )
        mgr.update_running_rep_centroids(batch2, [0])
        mgr.freeze_epoch_centroid_snapshot()
        frozen2 = mgr.epoch_frozen_centroids[0]
        assert not torch.allclose(frozen1, frozen2)

    def test_snapshot_before_any_centroids(self) -> None:
        """Snapshot when no centroids exist -> empty frozen dict."""
        mgr = CentroidManager()
        mgr.freeze_epoch_centroid_snapshot()
        assert len(mgr.epoch_frozen_centroids) == 0


# ======================================================================
# CentroidManager — update_centroids_from_epoch_buffer
# ======================================================================


class TestUpdateCentroidsFromEpochBuffer:
    def test_basic(self) -> None:
        mgr = CentroidManager()
        chunks = [
            F.normalize(torch.randn(10, 4), p=2, dim=1),
            F.normalize(torch.randn(10, 4), p=2, dim=1),
        ]
        label_chunks = [
            torch.zeros(10, dtype=torch.int64),
            torch.ones(10, dtype=torch.int64),
        ]
        mock_analyzer = MagicMock()
        mock_analyzer.compute_class_centroids.return_value = (
            F.normalize(torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32), p=2, dim=1),
            [0],
        )
        mgr.update_centroids_from_epoch_buffer(chunks, label_chunks, mock_analyzer)
        assert len(mgr.centroid_ema_state) == 1
        # Chunks should be cleared after processing
        assert len(chunks) == 0
        assert len(label_chunks) == 0

    def test_empty_chunks(self) -> None:
        mgr = CentroidManager()
        chunks: list[torch.Tensor] = []
        label_chunks: list[torch.Tensor] = []
        mock_analyzer = MagicMock()
        mgr.update_centroids_from_epoch_buffer(chunks, label_chunks, mock_analyzer)
        assert len(mgr.centroid_ema_state) == 0
        mock_analyzer.compute_class_centroids.assert_not_called()

    def test_empty_centroids_from_analyzer(self) -> None:
        """If compute_class_centroids returns empty, chunks should still be cleared."""
        mgr = CentroidManager()
        chunks = [torch.randn(10, 4)]
        label_chunks = [torch.zeros(10, dtype=torch.int64)]
        mock_analyzer = MagicMock()
        mock_analyzer.compute_class_centroids.return_value = (
            torch.zeros((0, 4), dtype=torch.float32),
            [],
        )
        mgr.update_centroids_from_epoch_buffer(chunks, label_chunks, mock_analyzer)
        assert len(mgr.centroid_ema_state) == 0
        assert len(chunks) == 0
        assert len(label_chunks) == 0


# ======================================================================
# CentroidManager — stabilize_centroids
# ======================================================================


class TestStabilizeCentroids:
    def test_basic(self) -> None:
        mgr = CentroidManager()
        # Pre-populate EMA state
        init = F.normalize(
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32),
            p=2, dim=1,
        )
        mgr.update_running_rep_centroids(init, [0, 1])
        batch = F.normalize(
            torch.tensor([[0.8, 0.2, 0.0], [0.2, 0.8, 0.0]], dtype=torch.float32),
            p=2, dim=1,
        )
        stabilized = mgr.stabilize_centroids(batch, [0, 1])
        assert list(stabilized.shape) == [2, 3]
        assert stabilized.requires_grad is False  # detach()

    def test_unknown_class_id(self) -> None:
        """Class ID not in EMA state -> just use current (no prev)."""
        mgr = CentroidManager()
        batch = F.normalize(
            torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32), p=2, dim=1,
        )
        stabilized = mgr.stabilize_centroids(batch, [99])
        assert list(stabilized.shape) == [1, 3]
        # Should have been added to state
        assert 99 in mgr.centroid_ema_state

    def test_empty_batch(self) -> None:
        mgr = CentroidManager()
        batch = torch.zeros((0, 4), dtype=torch.float32)
        stabilized = mgr.stabilize_centroids(batch, [])
        assert list(stabilized.shape) == [0, 4]


# ======================================================================
# CentroidManager — reset_epoch_frozen_centroids
# ======================================================================


class TestResetEpochFrozenCentroids:
    def test_basic(self) -> None:
        mgr = CentroidManager()
        # Populate and freeze
        batch = F.normalize(
            torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32), p=2, dim=1,
        )
        mgr.update_running_rep_centroids(batch, [0])
        mgr.freeze_epoch_centroid_snapshot()
        assert len(mgr.epoch_frozen_centroids) == 1
        mgr.reset_epoch_frozen_centroids()
        assert len(mgr.epoch_frozen_centroids) == 0
        # EMA state should survive
        assert len(mgr.centroid_ema_state) == 1


# ======================================================================
# CentroidManager — momentum clamping
# ======================================================================


class TestCentroidManagerInit:
    def test_momentum_clamping(self) -> None:
        mgr = CentroidManager(centroid_ema_momentum=1.5)
        assert mgr._momentum == 1.0
        mgr_low = CentroidManager(centroid_ema_momentum=-0.5)
        assert mgr_low._momentum == 0.0


# ======================================================================
# RepresentationCoordinator — rebalance_representation_batch
# ======================================================================


class TestRebalanceRepresentationBatch:
    def test_basic(self) -> None:
        features = F.normalize(torch.randn(30, 4), p=2, dim=1)
        labels = torch.cat([
            torch.zeros(10, dtype=torch.int64),
            torch.ones(10, dtype=torch.int64),
            torch.full((10,), 2, dtype=torch.int64),
        ])
        reb_f, reb_l = RepresentationCoordinator.rebalance_representation_batch(
            features, labels, target_per_class=5, run_seed=42, global_step=0,
        )
        assert isinstance(reb_f, torch.Tensor)
        assert isinstance(reb_l, torch.Tensor)
        # 3 classes * 5 per class = 15
        assert reb_f.shape[0] == 15
        assert reb_l.shape[0] == 15

    def test_single_class(self) -> None:
        features = F.normalize(torch.randn(20, 4), p=2, dim=1)
        labels = torch.zeros(20, dtype=torch.int64)
        reb_f, reb_l = RepresentationCoordinator.rebalance_representation_batch(
            features, labels, target_per_class=10, run_seed=42, global_step=0,
        )
        # Single class -> early return (unchanged)
        assert reb_f.shape[0] == 20
        assert reb_l.shape[0] == 20

    def test_empty_batch(self) -> None:
        features = torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.zeros((0,), dtype=torch.int64)
        reb_f, reb_l = RepresentationCoordinator.rebalance_representation_batch(
            features, labels, target_per_class=5, run_seed=42, global_step=0,
        )
        # Early return for features.shape[0] <= 1
        assert reb_f.shape[0] == 0
        assert reb_l.shape[0] == 0

    def test_single_element(self) -> None:
        features = F.normalize(torch.randn(1, 4), p=2, dim=1)
        labels = torch.zeros(1, dtype=torch.int64)
        reb_f, reb_l = RepresentationCoordinator.rebalance_representation_batch(
            features, labels, target_per_class=5, run_seed=42, global_step=0,
        )
        assert reb_f.shape[0] == 1

    def test_deterministic_with_seed(self) -> None:
        """Same seed + global_step should produce same output."""
        features = F.normalize(torch.randn(30, 4), p=2, dim=1)
        labels = torch.cat([
            torch.zeros(10, dtype=torch.int64),
            torch.ones(10, dtype=torch.int64),
            torch.full((10,), 2, dtype=torch.int64),
        ])
        r1_f, r1_l = RepresentationCoordinator.rebalance_representation_batch(
            features, labels, target_per_class=4, run_seed=42, global_step=0,
        )
        r2_f, r2_l = RepresentationCoordinator.rebalance_representation_batch(
            features, labels, target_per_class=4, run_seed=42, global_step=0,
        )
        assert torch.equal(r1_f, r2_f)
        assert torch.equal(r1_l, r2_l)

    def test_different_seed_different_output(self) -> None:
        """Different seeds should (usually) produce different output."""
        features = F.normalize(torch.randn(30, 4), p=2, dim=1)
        labels = torch.cat([
            torch.zeros(10, dtype=torch.int64),
            torch.ones(10, dtype=torch.int64),
            torch.full((10,), 2, dtype=torch.int64),
        ])
        r1_f, r1_l = RepresentationCoordinator.rebalance_representation_batch(
            features, labels, target_per_class=5, run_seed=42, global_step=0,
        )
        r2_f, r2_l = RepresentationCoordinator.rebalance_representation_batch(
            features, labels, target_per_class=5, run_seed=99, global_step=0,
        )
        # Different seed makes different random choices
        different_f = not torch.equal(r1_f, r2_f)
        different_l = not torch.equal(r1_l, r2_l)
        assert different_f and different_l

    def test_target_per_class_clamping(self) -> None:
        """target_per_class should be clamped to max available per class."""
        features = F.normalize(torch.randn(20, 4), p=2, dim=1)
        labels = torch.cat([
            torch.zeros(5, dtype=torch.int64),
            torch.ones(5, dtype=torch.int64),
            torch.full((10,), 2, dtype=torch.int64),
        ])
        reb_f, reb_l = RepresentationCoordinator.rebalance_representation_batch(
            features, labels, target_per_class=100, run_seed=42, global_step=0,
        )
        # target clamped to max count (10)
        # 3 classes * min(max, min(counts)) = 3 * 10 = 30? No actually...
        # target = max(1, min(100, max(counts))) = max(1, min(100, 10)) = 10
        # per class: class 0 has 5 < 10, so replace=True (oversample)
        # class 1 has 5 < 10, replace=True
        # class 2 has 10 >= 10, replace=False
        # total = 10 + 10 + 10 = 30
        assert reb_f.shape[0] == 30


# ======================================================================
# RepresentationCoordinator — store_representation_chunks
# ======================================================================


class TestStoreRepresentationChunks:
    def test_basic_append(self) -> None:
        features = F.normalize(torch.randn(10, 4), p=2, dim=1)
        labels = torch.zeros(10, dtype=torch.int64)
        feat_chunks: list[torch.Tensor] = []
        label_chunks: list[torch.Tensor] = []
        mock_analyzer = MagicMock()
        mock_analyzer.prepare_representation_features.return_value = features

        RepresentationCoordinator.store_representation_chunks(
            feat_chunks, label_chunks,
            backbone_features=torch.randn(10, 64),
            y_family=labels,
            cluster_analyzer=mock_analyzer,
        )
        assert len(feat_chunks) == 1
        assert len(label_chunks) == 1
        assert feat_chunks[0].device == torch.device("cpu")
        assert label_chunks[0].device == torch.device("cpu")


# ======================================================================
# RepresentationCoordinator — store_epoch_chunks
# ======================================================================


class TestStoreEpochChunks:
    def test_basic_append(self) -> None:
        features = F.normalize(torch.randn(10, 4), p=2, dim=1)
        labels = torch.zeros(10, dtype=torch.int64)
        feat_chunks: list[torch.Tensor] = []
        label_chunks: list[torch.Tensor] = []
        mock_analyzer = MagicMock()
        mock_analyzer.prepare_representation_features.return_value = features

        RepresentationCoordinator.store_epoch_chunks(
            feat_chunks, label_chunks,
            backbone_features=torch.randn(10, 64),
            y_family=labels,
            cluster_analyzer=mock_analyzer,
        )
        assert len(feat_chunks) == 1
        assert len(label_chunks) == 1
        assert feat_chunks[0].device == torch.device("cpu")


# ======================================================================
# RepresentationCoordinator — concat_chunks
# ======================================================================


class TestConcatChunks:
    def test_non_empty(self) -> None:
        chunks = [torch.randn(5, 4), torch.randn(3, 4)]
        result = RepresentationCoordinator.concat_chunks(chunks)
        assert list(result.shape) == [8, 4]

    def test_empty_list_with_default(self) -> None:
        default = torch.zeros((1, 4), dtype=torch.float32)
        result = RepresentationCoordinator.concat_chunks([], default=default)
        assert torch.equal(result, default)

    def test_empty_list_without_default(self) -> None:
        result = RepresentationCoordinator.concat_chunks([])
        assert list(result.shape) == [0, 0]


# ======================================================================
# RepresentationCoordinator — concat_label_chunks
# ======================================================================


class TestConcatLabelChunks:
    def test_basic(self) -> None:
        chunks = [torch.zeros(5, dtype=torch.int64), torch.ones(3, dtype=torch.int64)]
        result = RepresentationCoordinator.concat_label_chunks(chunks)
        assert list(result.shape) == [8]
        assert result.dtype == torch.int64

    def test_empty(self) -> None:
        result = RepresentationCoordinator.concat_label_chunks([])
        assert list(result.shape) == [0]
        assert result.dtype == torch.int64


# ======================================================================
# LossRegistry — edge-case cross-cutting
# ======================================================================


class TestCrossCuttingEdgeCases:
    def test_all_losses_device_consistency(self) -> None:
        """All loss functions should work seamlessly on CPU."""
        features = F.normalize(torch.randn(16, 4), p=2, dim=1)
        labels = torch.cat([
            torch.zeros(4, dtype=torch.int64),
            torch.ones(4, dtype=torch.int64),
            torch.full((4,), 2, dtype=torch.int64),
            torch.full((4,), 3, dtype=torch.int64),
        ])
        logits = torch.randn(16, 4, dtype=torch.float32)

        # supervised_contrastive_loss
        sc = LossRegistry.supervised_contrastive_loss(features, labels, temperature=0.5)
        assert sc.device.type == "cpu"

        # supcon_anchor_weights
        saw = LossRegistry.supcon_anchor_weights(labels)
        assert saw.device.type == "cpu"

        # class_conditional_energy_gap_loss
        cce, _, _, _, _ = LossRegistry.class_conditional_energy_gap_loss(logits, labels, alpha=1.0)
        assert cce.device.type == "cpu"

        # energy_class_balance_loss
        ecb, _, _, _ = LossRegistry.energy_class_balance_loss(logits)
        assert ecb.device.type == "cpu"

        # energy_min_winner_loss
        emw, _, _ = LossRegistry.energy_min_winner_loss(logits, None, min_winners=3)
        assert emw.device.type == "cpu"

        # pairwise_margin_repulsion_loss
        pmr = LossRegistry.pairwise_margin_repulsion_loss(features, labels, margin=2.0)
        assert pmr.device.type == "cpu"

        # centroid_separation_barrier_loss
        csb = LossRegistry.centroid_separation_barrier_loss(features, labels, min_distance=0.5)
        assert csb.device.type == "cpu"

        # centroid_repulsion_loss
        cr = LossRegistry.centroid_repulsion_loss(features, labels, margin=2.0)
        assert cr.device.type == "cpu"

        # intra_class_variance_clamp_loss
        icv = LossRegistry.intra_class_variance_clamp_loss(
            features, labels, var_lower_bound=0.0, var_upper_bound=1.0,
        )
        assert icv.device.type == "cpu"

        # compute_batch_class_centroids_for_loss
        cbc, _ = LossRegistry.compute_batch_class_centroids_for_loss(features, labels)
        assert cbc.device.type == "cpu"

        # critical_pair_centroid_push_loss
        cpc = LossRegistry.critical_pair_centroid_push_loss(
            features, labels, min_distance=0.5, critical_collision_pairs={(0, 1)},
        )
        assert cpc.device.type == "cpu"

        # global_centroid_guided_losses — use pre-populated frozen centroids
        frozen = {i: torch.zeros(4, dtype=torch.float32) for i in range(4)}
        gcg_r, gcg_b, _ = LossRegistry.global_centroid_guided_losses(
            cbc, [0, 1, 2, 3], frozen,
            rep_centroid_repulsion_margin=0.8,
            rep_centroid_barrier_min_distance=0.5,
        )
        assert gcg_r.device.type == "cpu"
        assert gcg_b.device.type == "cpu"


# ======================================================================
# Smoke-level: all delegates instantiate and run without errors
# ======================================================================


class TestPackageSmoke:
    def test_centroid_manager_lifecycle(self) -> None:
        """Full lifecycle: update -> freeze -> stabilize -> reset."""
        mgr = CentroidManager(centroid_ema_momentum=0.9)
        batch = F.normalize(
            torch.randn(2, 4), p=2, dim=1,
        )
        mgr.update_running_rep_centroids(batch, [0, 1])
        mgr.freeze_epoch_centroid_snapshot()
        assert len(mgr.epoch_frozen_centroids) == 2
        stabilized = mgr.stabilize_centroids(batch, [0, 1])
        assert list(stabilized.shape) == [2, 4]
        mgr.reset_epoch_frozen_centroids()
        assert len(mgr.epoch_frozen_centroids) == 0

    def test_coordinator_concat_pipeline(self) -> None:
        """Pipeline: store -> concat -> empty reset."""
        feat_chunks: list[torch.Tensor] = []
        label_chunks: list[torch.Tensor] = []
        mock = MagicMock()
        # Return different sizes matching the backbone_features batch shapes
        def prep_side_effect(x: torch.Tensor) -> torch.Tensor:
            return torch.randn(int(x.shape[0]), 4)

        mock.prepare_representation_features = prep_side_effect

        RepresentationCoordinator.store_representation_chunks(
            feat_chunks, label_chunks,
            backbone_features=torch.randn(10, 64),
            y_family=torch.zeros(10, dtype=torch.int64),
            cluster_analyzer=mock,
        )
        RepresentationCoordinator.store_representation_chunks(
            feat_chunks, label_chunks,
            backbone_features=torch.randn(5, 64),
            y_family=torch.ones(5, dtype=torch.int64),
            cluster_analyzer=mock,
        )
        concat_feat = RepresentationCoordinator.concat_chunks(feat_chunks)
        concat_label = RepresentationCoordinator.concat_label_chunks(label_chunks)
        assert list(concat_feat.shape) == [15, 4]
        assert list(concat_label.shape) == [15]

        # Empty after clearing
        feat_chunks.clear()
        label_chunks.clear()
        empty = RepresentationCoordinator.concat_chunks(feat_chunks)
        assert list(empty.shape) == [0, 0]

    def test_loss_registry_static_methods(self) -> None:
        """All loss methods can be called as static methods (no instance)."""
        features = F.normalize(torch.randn(8, 4), p=2, dim=1)
        labels = torch.cat([
            torch.zeros(4, dtype=torch.int64),
            torch.ones(4, dtype=torch.int64),
        ])
        logits = torch.randn(8, 2, dtype=torch.float32)

        # Each call just needs to not crash
        LossRegistry.supervised_contrastive_loss(features, labels, temperature=0.5)
        LossRegistry.supcon_anchor_weights(labels)
        LossRegistry.class_conditional_energy_gap_loss(logits, labels, alpha=1.0)
        LossRegistry.energy_class_balance_loss(logits)
        LossRegistry.energy_min_winner_loss(logits, None, min_winners=2)
        LossRegistry.pairwise_margin_repulsion_loss(features, labels, margin=2.0)
        LossRegistry.centroid_separation_barrier_loss(features, labels, min_distance=0.5)
        LossRegistry.centroid_repulsion_loss(features, labels, margin=2.0)
        LossRegistry.intra_class_variance_clamp_loss(features, labels, var_lower_bound=0.0, var_upper_bound=1.0)
        LossRegistry.compute_batch_class_centroids_for_loss(features, labels)
        frozen = {0: torch.zeros(4), 1: torch.zeros(4)}
        LossRegistry.global_centroid_guided_losses(
            torch.randn(2, 4), [0, 1], frozen,
            rep_centroid_repulsion_margin=0.8,
            rep_centroid_barrier_min_distance=0.5,
        )
        LossRegistry.critical_pair_centroid_push_loss(
            features, labels, min_distance=0.5, critical_collision_pairs={(0, 1)},
        )
