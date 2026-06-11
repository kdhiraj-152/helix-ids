"""
Targeted unit tests for loss.py uncovered code paths.

Focuses on lines missed in current coverage (68% → 85%+):
- get_class_weights: numpy input, inferred num_classes
- ThreatAwareFocalLoss: float alpha, label_smoothing=0, alpha gathering,
  reduction='sum','none', all branches
- CalibrationLoss: sum/none reduction
- MultiTaskLoss: fine/calibration branches, get_curriculum_weights
- FocalLoss: full coverage including alpha, all reductions
- create_loss_function: all loss_type branches + error
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from helix_ids.models.loss import (
    CalibrationLoss,
    FocalLoss,
    MultiTaskLoss,
    ThreatAwareFocalLoss,
    create_loss_function,
    get_class_weights,
)

# =============================================================================
# get_class_weights — uncovered paths
# =============================================================================


class TestGetClassWeightsUncovered:
    """Cover lines 69 and 74 in get_class_weights."""

    def test_numpy_input(self):
        """Line 69: pass numpy array instead of tensor."""
        y_np = np.array([0, 0, 0, 1, 2, 1, 2])
        weights = get_class_weights(y_np, num_classes=3)
        assert len(weights) == 3
        assert torch.isfinite(weights).all()

    def test_inferred_num_classes(self):
        """Line 74: infer num_classes from y.max() when num_classes is None."""
        y = torch.tensor([0, 1, 2, 3, 4, 0, 1])
        weights = get_class_weights(y, num_classes=None)
        assert len(weights) == 5  # classes 0..4
        assert torch.isfinite(weights).all()

    def test_numpy_inferred_num_classes(self):
        """Lines 69+74: numpy array with inferred num_classes."""
        y_np = np.array([0, 1, 2, 3, 4])
        weights = get_class_weights(y_np, num_classes=None)
        assert len(weights) == 5
        assert torch.isfinite(weights).all()


# =============================================================================
# ThreatAwareFocalLoss — uncovered paths
# =============================================================================


class TestThreatAwareFocalLossUncovered:
    """Cover lines: 156, 202, 222, 229-234, 246-249."""

    def test_alpha_as_float(self):
        """Line 156: alpha as float → converted to 2-element tensor."""
        loss_fn = ThreatAwareFocalLoss(alpha=0.75, use_warmup=False)
        assert loss_fn.alpha is not None
        assert loss_fn.alpha.shape == (2,), f"Expected (2,), got {loss_fn.alpha.shape}"
        logits = torch.randn(32, 2)
        labels = torch.randint(0, 2, (32,))
        loss = loss_fn(logits, labels)
        assert loss.item() > 0

    def test_label_smoothing_zero(self):
        """Line 202: label_smoothing=0 → targets_smooth is None → line 222."""
        loss_fn = ThreatAwareFocalLoss(label_smoothing=0.0, use_warmup=False)
        logits = torch.randn(32, 5)
        labels = torch.randint(0, 5, (32,))
        loss = loss_fn(logits, labels)
        assert loss.item() > 0

    def test_alpha_gathering_after_warmup(self):
        """Lines 229-234: alpha tensor gathered in forward (past warmup)."""
        alpha = torch.tensor([1.0, 1.5, 2.0, 8.0, 10.0])
        loss_fn = ThreatAwareFocalLoss(
            alpha=alpha, use_warmup=False, warmup_epochs=10
        )
        logits = torch.randn(32, 5)
        labels = torch.randint(0, 5, (32,))
        loss = loss_fn(logits, labels)
        assert loss.item() > 0

    def test_alpha_smaller_than_num_classes(self):
        """Line 230-232: expand alpha when alpha.size(0) < num_classes.
        .expand only works from singleton dims, so use 1-element alpha."""
        alpha = torch.tensor([1.0])  # only 1 weight, expandable to 5
        loss_fn = ThreatAwareFocalLoss(
            alpha=alpha, use_warmup=False, warmup_epochs=10
        )
        logits = torch.randn(32, 5)
        labels = torch.randint(0, 5, (32,))
        loss = loss_fn(logits, labels)
        assert loss.item() > 0

    def test_reduction_sum(self):
        """Line 246-247: reduction='sum'."""
        loss_fn = ThreatAwareFocalLoss(reduction="sum", use_warmup=False)
        logits = torch.randn(32, 5)
        labels = torch.randint(0, 5, (32,))
        loss = loss_fn(logits, labels)
        assert loss.dim() == 0  # scalar
        assert loss.item() > 0

    def test_reduction_none(self):
        """Line 248-249: reduction='none'."""
        loss_fn = ThreatAwareFocalLoss(reduction="none", use_warmup=False)
        logits = torch.randn(32, 5)
        labels = torch.randint(0, 5, (32,))
        loss = loss_fn(logits, labels)
        assert loss.shape == (32,)  # per-sample
        assert (loss >= 0).all()

    def test_warmup_path(self):
        """Line 190-191: warmup branch uses CrossEntropyLoss."""
        loss_fn = ThreatAwareFocalLoss(use_warmup=True, warmup_epochs=10)
        loss_fn.set_epoch(5)  # within warmup
        logits = torch.randn(32, 5)
        labels = torch.randint(0, 5, (32,))
        loss = loss_fn(logits, labels)
        assert loss.item() > 0

    def test_no_threat_weights_smaller(self):
        """Line 239-241: threat_w.size(0) < num_classes → skip threat weighting."""
        # Force threat_weights smaller than num_classes to skip the gather
        small_threat = torch.tensor([1.0])  # Only 1 weight
        loss_fn = ThreatAwareFocalLoss(
            threat_weights=small_threat,
            use_warmup=False,
        )
        logits = torch.randn(32, 5)
        labels = torch.randint(0, 5, (32,))
        loss = loss_fn(logits, labels)
        assert loss.item() > 0

    def test_repr(self):
        """__repr__ method."""
        loss_fn = ThreatAwareFocalLoss(gamma=1.5, reduction="mean", label_smoothing=0.1)
        r = repr(loss_fn)
        assert "ThreatAwareFocalLoss" in r
        assert "gamma=1.5" in r

    def test_alpha_none_after_warmup(self):
        """No alpha, after warmup — verify no crash when self.alpha is None."""
        loss_fn = ThreatAwareFocalLoss(alpha=None, use_warmup=False)
        logits = torch.randn(32, 5)
        labels = torch.randint(0, 5, (32,))
        loss = loss_fn(logits, labels)
        assert loss.item() > 0

    def test_set_epoch_warmup_to_post_warmup(self):
        """Progressive: set epoch, then verify warmup vs focal path."""
        loss_fn = ThreatAwareFocalLoss(use_warmup=True, warmup_epochs=10)
        loss_fn.set_epoch(1)
        logits = torch.randn(32, 5)
        labels = torch.randint(0, 5, (32,))
        loss_warmup = loss_fn(logits, labels)

        loss_fn.set_epoch(15)
        loss_post = loss_fn(logits, labels)
        assert loss_warmup.item() > 0
        assert loss_post.item() > 0


# =============================================================================
# CalibrationLoss — uncovered lines 305-328
# =============================================================================


class TestCalibrationLoss:
    """Cover CalibrationLoss forward all reduction paths."""

    def test_reduction_mean(self):
        loss_fn = CalibrationLoss(reduction="mean")
        logits = torch.randn(32, 5)
        labels = torch.randint(0, 5, (32,))
        loss = loss_fn(logits, labels)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_reduction_sum(self):
        """Line 325-326: reduction='sum'."""
        loss_fn = CalibrationLoss(reduction="sum")
        logits = torch.randn(32, 5)
        labels = torch.randint(0, 5, (32,))
        loss = loss_fn(logits, labels)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_reduction_none(self):
        """Line 327-328: reduction='none'."""
        loss_fn = CalibrationLoss(reduction="none")
        logits = torch.randn(32, 5)
        labels = torch.randint(0, 5, (32,))
        loss = loss_fn(logits, labels)
        assert loss.shape == (32,)
        assert (loss >= 0).all()

    def test_temperature_scaling(self):
        """Non-default temperature."""
        loss_fn = CalibrationLoss(temperature=2.0)
        logits = torch.randn(32, 5)
        labels = torch.randint(0, 5, (32,))
        loss = loss_fn(logits, labels)
        assert loss.item() > 0

    def test_perfect_prediction(self):
        """Edge case: perfectly correct predictions."""
        loss_fn = CalibrationLoss(reduction="mean")
        logits = torch.full((4, 2), -10.0)
        logits[:, 0] = 10.0  # all predict class 0
        labels = torch.zeros(4, dtype=torch.long)
        loss = loss_fn(logits, labels)
        # confidence ≈ 1.0, correctness = 1.0 → BCE should be near 0
        assert loss.item() < 0.01

    def test_all_wrong(self):
        """Edge case: all wrong predictions = high calibration loss."""
        loss_fn = CalibrationLoss(reduction="mean")
        logits = torch.full((4, 2), 10.0)
        logits[:, 0] = -10.0  # model very confident in wrong class 1
        labels = torch.zeros(4, dtype=torch.long)  # true class is 0
        loss = loss_fn(logits, labels)
        # confidence ≈ 1.0, correctness = 0.0 → BCE ≈ -log(0) ≈ large
        assert loss.item() > 1.0


# =============================================================================
# MultiTaskLoss — uncovered lines 491, 532-534, 539-541
# =============================================================================


class TestMultiTaskLossUncovered:
    """Cover curriculum weights getter, fine loss, calibration loss."""

    def test_get_curriculum_weights(self):
        """Line 491: get_curriculum_weights returns dict."""
        loss_fn = MultiTaskLoss()
        weights = loss_fn.get_curriculum_weights()
        assert isinstance(weights, dict)
        assert "alpha" in weights
        assert "beta" in weights
        assert "gamma" in weights
        assert "delta" in weights
        assert "epoch" in weights
        assert weights["epoch"] == 1

    def test_curriculum_weights_vary_by_epoch(self):
        """Verify weights change across curricula."""
        loss_fn = MultiTaskLoss()
        loss_fn.set_epoch(1)
        w1 = loss_fn.get_curriculum_weights()
        loss_fn.set_epoch(60)
        w60 = loss_fn.get_curriculum_weights()
        assert w1["alpha"] == 1.0 and w1["beta"] == 0.0  # binary only
        assert w60["delta"] == 0.2  # calibration active

    def test_fine_loss_branch(self):
        """Lines 532-534: add fine-grained loss when 'fine' in outputs."""
        loss_fn = MultiTaskLoss()
        loss_fn.set_epoch(40)  # gamma_weight > 0 at epoch 31+
        outputs = {
            "binary": torch.randn(32, 2),
            "family": torch.randn(32, 5),
            "fine": torch.randn(32, 23),
        }
        targets = {
            "binary": torch.randint(0, 2, (32,)),
            "family": torch.randint(0, 5, (32,)),
            "fine": torch.randint(0, 23, (32,)),
        }
        total_loss, loss_dict = loss_fn(outputs, targets)
        assert "loss_fine" in loss_dict
        assert "loss_total" in loss_dict
        assert total_loss.item() > 0

    def test_calibration_loss_branch(self):
        """Lines 539-541: add calibration loss when delta > 0."""
        loss_fn = MultiTaskLoss()
        loss_fn.set_epoch(60)  # delta=0.2 at epoch 51+
        outputs = {
            "binary": torch.randn(32, 2),
            "family": torch.randn(32, 5),
            "fine": torch.randn(32, 23),
        }
        targets = {
            "binary": torch.randint(0, 2, (32,)),
            "family": torch.randint(0, 5, (32,)),
            "fine": torch.randint(0, 23, (32,)),
        }
        total_loss, loss_dict = loss_fn(outputs, targets)
        assert "loss_calibration" in loss_dict
        assert "loss_fine" in loss_dict
        assert total_loss.item() > 0

    def test_missing_keys(self):
        """No crash when keys are missing from outputs."""
        loss_fn = MultiTaskLoss()
        loss_fn.set_epoch(1)
        outputs = {
            "binary": torch.randn(32, 2),
        }
        targets = {
            "binary": torch.randint(0, 2, (32,)),
        }
        total_loss, loss_dict = loss_fn(outputs, targets)
        assert "loss_total" in loss_dict
        assert total_loss.item() > 0

    def test_family_only_mid_curriculum(self):
        """Family branch only (epoch 11-30, beta>0, gamma=0, delta=0)."""
        loss_fn = MultiTaskLoss()
        loss_fn.set_epoch(20)
        outputs = {
            "family": torch.randn(32, 5),
        }
        targets = {
            "family": torch.randint(0, 5, (32,)),
        }
        total_loss, loss_dict = loss_fn(outputs, targets)
        assert "loss_family" in loss_dict
        assert total_loss.item() > 0

    def test_repr(self):
        loss_fn = MultiTaskLoss()
        r = repr(loss_fn)
        assert "MultiTaskLoss" in r
        assert "binary=2" in r

    def test_expand_threat_weights_custom(self):
        """Verify _expand_threat_weights with various num_fine_classes."""
        loss_fn = MultiTaskLoss()
        tw = torch.tensor([1.0, 1.2, 1.5, 3.0, 4.0])
        # Small num_fine_classes (only Normal class)
        small = loss_fn._expand_threat_weights(tw, 1)
        assert small.shape == (1,)
        assert small[0] == 1.0
        # Medium (up to Probe variants)
        medium = loss_fn._expand_threat_weights(tw, 15)
        assert medium.shape == (15,)
        assert medium[0] == 1.0
        assert medium[11] == 1.5
        # Full
        full = loss_fn._expand_threat_weights(tw, 23)
        assert full.shape == (23,)
        assert full[21] == 4.0


# =============================================================================
# FocalLoss — complete lines 578-604
# =============================================================================


class TestFocalLoss:
    """Cover FocalLoss.__init__ (578-585) and forward (589-604)."""

    def test_no_alpha(self):
        loss_fn = FocalLoss()
        logits = torch.randn(32, 5)
        labels = torch.randint(0, 5, (32,))
        loss = loss_fn(logits, labels)
        assert loss.item() > 0

    def test_with_alpha(self):
        """Lines 582-584, 596-598: register alpha buffer and gather."""
        alpha = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        loss_fn = FocalLoss(alpha=alpha)
        assert loss_fn.alpha is not None
        logits = torch.randn(32, 5)
        labels = torch.randint(0, 5, (32,))
        loss = loss_fn(logits, labels)
        assert loss.item() > 0

    def test_reduction_sum(self):
        """Line 602-603: reduction='sum'."""
        loss_fn = FocalLoss(reduction="sum")
        logits = torch.randn(32, 5)
        labels = torch.randint(0, 5, (32,))
        loss = loss_fn(logits, labels)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_reduction_none(self):
        """Line 604: reduction='none' fallthrough."""
        loss_fn = FocalLoss(reduction="none")
        logits = torch.randn(32, 5)
        labels = torch.randint(0, 5, (32,))
        loss = loss_fn(logits, labels)
        assert loss.shape == (32,)
        assert (loss >= 0).all()

    def test_non_default_gamma(self):
        loss_fn = FocalLoss(gamma=3.0)
        logits = torch.randn(32, 5)
        labels = torch.randint(0, 5, (32,))
        loss = loss_fn(logits, labels)
        assert loss.item() > 0

    def test_backward(self):
        loss_fn = FocalLoss()
        logits = torch.randn(32, 5, requires_grad=True)
        labels = torch.randint(0, 5, (32,))
        loss = loss_fn(logits, labels)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.shape == logits.shape


# =============================================================================
# create_loss_function — lines 625-642
# =============================================================================


class TestCreateLossFunction:
    """Cover all loss_type branches in create_loss_function."""

    def test_ce(self):
        """Line 625-627: standard cross-entropy."""
        loss_fn = create_loss_function("ce", num_classes=5)
        assert isinstance(loss_fn, torch.nn.CrossEntropyLoss)

    def test_ce_with_weights(self):
        """Cross-entropy with class weights."""
        weights = torch.tensor([1.0, 2.0, 1.0, 3.0, 4.0])
        loss_fn = create_loss_function("ce", num_classes=5, class_weights=weights)
        assert isinstance(loss_fn, torch.nn.CrossEntropyLoss)
        assert loss_fn.weight is not None

    def test_focal(self):
        """Line 629-631: standard focal loss."""
        loss_fn = create_loss_function("focal", num_classes=5)
        assert isinstance(loss_fn, FocalLoss)

    def test_focal_with_weights(self):
        """Focal loss with class weights (alpha)."""
        weights = torch.tensor([1.0, 2.0, 1.0, 3.0, 4.0])
        loss_fn = create_loss_function("focal", num_classes=5, class_weights=weights)
        assert isinstance(loss_fn, FocalLoss)
        assert loss_fn.alpha is not None

    def test_focal_with_kwargs(self):
        """Focal loss with extra kwargs (gamma)."""
        loss_fn = create_loss_function("focal", num_classes=5, gamma=3.0)
        assert isinstance(loss_fn, FocalLoss)
        assert loss_fn.gamma == 3.0

    def test_threat_focal(self):
        """Line 633-635: threat-aware focal loss."""
        loss_fn = create_loss_function("threat_focal", num_classes=5)
        assert isinstance(loss_fn, ThreatAwareFocalLoss)

    def test_threat_focal_with_alpha(self):
        """Threat-aware focal loss with class weights."""
        weights = torch.tensor([1.0, 2.0, 1.0, 3.0, 4.0])
        loss_fn = create_loss_function(
            "threat_focal", num_classes=5, class_weights=weights
        )
        assert isinstance(loss_fn, ThreatAwareFocalLoss)
        assert loss_fn.alpha is not None

    def test_multitask(self):
        """Line 637-639: multi-task loss."""
        loss_fn = create_loss_function("multitask", num_classes=23)
        assert isinstance(loss_fn, MultiTaskLoss)
        assert loss_fn.num_fine_classes == 23

    def test_unknown_loss_type(self):
        """Line 641-642: ValueError for unknown loss type."""
        import pytest
        with pytest.raises(ValueError, match="Unknown loss type"):
            create_loss_function("unknown_type")
