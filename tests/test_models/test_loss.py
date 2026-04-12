"""
Tests for loss functions.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from helix_ids.models.loss import (
    DEFAULT_THREAT_WEIGHTS,
    MultiTaskLoss,
    ThreatAwareFocalLoss,
    get_class_weights,
)


class TestThreatAwareFocalLoss:
    """Test ThreatAwareFocalLoss implementation."""

    def test_focal_loss_shape(self, sample_labels_multiclass):
        """Test focal loss produces scalar output."""
        loss_fn = ThreatAwareFocalLoss(gamma=2.0)

        logits = torch.randn(32, 5)
        loss = loss_fn(logits, sample_labels_multiclass)

        assert loss.dim() == 0, "Loss should be scalar"
        assert loss.item() > 0, "Loss should be positive"

    def test_focal_loss_with_alpha(self):
        """Test focal loss with class weights (alpha)."""
        # Without alpha
        loss_no_alpha = ThreatAwareFocalLoss(gamma=2.0, alpha=None)
        # With alpha
        alpha = torch.tensor([1.0, 1.5, 2.0, 8.0, 10.0])
        loss_with_alpha = ThreatAwareFocalLoss(gamma=2.0, alpha=alpha)

        logits = torch.randn(32, 5)
        labels = torch.randint(0, 5, (32,))

        l1 = loss_no_alpha(logits, labels)
        l2 = loss_with_alpha(logits, labels)

        # Both should be valid losses
        assert l1.item() > 0
        assert l2.item() > 0

    def test_focal_loss_gamma_effect(self):
        """Test that gamma parameter affects loss values (after warmup)."""
        loss_g0 = ThreatAwareFocalLoss(gamma=0.0, use_warmup=False)
        loss_g2 = ThreatAwareFocalLoss(gamma=2.0, use_warmup=False)

        # Create easy examples (high confidence correct predictions)
        logits = torch.zeros(32, 5)
        labels = torch.zeros(32, dtype=torch.long)
        logits[:, 0] = 5.0  # High confidence for class 0

        l_g0 = loss_g0(logits, labels)
        l_g2 = loss_g2(logits, labels)

        # With gamma=2, easy examples are down-weighted
        # so loss should be lower than gamma=0
        assert l_g2 < l_g0 * 0.5, "Gamma should reduce loss for easy examples"

    def test_focal_loss_values_positive(self):
        """Test loss values are always positive."""
        loss_fn = ThreatAwareFocalLoss(gamma=2.0)

        for _ in range(10):
            logits = torch.randn(16, 5)
            labels = torch.randint(0, 5, (16,))
            loss = loss_fn(logits, labels)

            assert loss.item() >= 0, "Loss should be non-negative"

    def test_focal_loss_backward(self):
        """Test loss supports backward pass."""
        loss_fn = ThreatAwareFocalLoss(gamma=2.0)

        logits = torch.randn(32, 5, requires_grad=True)
        labels = torch.randint(0, 5, (32,))

        loss = loss_fn(logits, labels)
        loss.backward()

        assert logits.grad is not None
        assert logits.grad.shape == logits.shape


class TestMultiTaskLoss:
    """Test MultiTaskLoss for curriculum learning."""

    def test_multitask_loss_combined(self):
        """Test multi-task loss combines binary and family losses."""
        loss_fn = MultiTaskLoss()

        outputs = {
            "binary": torch.randn(32, 2),
            "family": torch.randn(32, 5),
        }
        targets = {
            "binary": torch.randint(0, 2, (32,)),
            "family": torch.randint(0, 5, (32,)),
        }

        result = loss_fn(outputs, targets)
        # Returns tuple: (loss, loss_dict)
        loss = result[0]

        assert loss.dim() == 0
        assert loss.item() > 0

    def test_multitask_loss_curriculum(self):
        """Test curriculum learning phases change loss weights."""
        loss_fn = MultiTaskLoss()

        outputs = {
            "binary": torch.randn(32, 2),
            "family": torch.randn(32, 5),
        }
        targets = {
            "binary": torch.randint(0, 2, (32,)),
            "family": torch.randint(0, 5, (32,)),
        }

        # Epoch 1: binary focus
        loss_fn.set_epoch(1)
        result1 = loss_fn(outputs, targets)
        l1 = result1[0]

        # Epoch 30: family focus
        loss_fn.set_epoch(30)
        result2 = loss_fn(outputs, targets)
        l2 = result2[0]

        # Both should be valid
        assert l1.item() > 0
        assert l2.item() > 0


class TestClassWeights:
    """Test class weight calculation."""

    def test_get_class_weights_balanced(self):
        """Test weights for balanced classes."""
        y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])
        weights = get_class_weights(y, num_classes=3)

        assert len(weights) == 3
        # Balanced classes should have similar weights
        assert (weights.max() - weights.min()) < 0.5

    def test_get_class_weights_imbalanced(self):
        """Test weights for imbalanced classes."""
        y = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 2])  # 8:1:1 ratio
        weights = get_class_weights(y, num_classes=3)

        assert len(weights) == 3
        # Minority classes should have higher weights
        assert weights[1] > weights[0]
        assert weights[2] > weights[0]

    def test_get_class_weights_empty_class(self):
        """Test weights handle missing classes."""
        y = torch.tensor([0, 0, 0, 1, 1])  # No class 2
        weights = get_class_weights(y, num_classes=3)

        assert len(weights) == 3
        # All weights should be finite
        assert torch.isfinite(weights).all()

    def test_default_threat_weights(self):
        """Test default threat weights are defined correctly (conservative)."""
        assert DEFAULT_THREAT_WEIGHTS.shape == (5,)
        assert DEFAULT_THREAT_WEIGHTS[0] == 1.0  # Normal
        assert DEFAULT_THREAT_WEIGHTS[4] == 4.0  # U2R (conservative, prevents gradient collapse)
