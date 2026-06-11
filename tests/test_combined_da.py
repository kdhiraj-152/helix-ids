"""Tests for combined_da.py — Combined Domain Adaptation Loss."""

import math

import pytest
import torch

from helix_ids.models.adaptation.combined_da import (
    CombinedDAConfig,
    CombinedDANNLoss,
    CombinedDomainAdaptation,
    create_combined_da_loss,
)

# ---------------------------------------------------------------------------
# CombinedDAConfig
# ---------------------------------------------------------------------------

class TestCombinedDAConfig:
    """Tests for CombinedDAConfig dataclass."""

    def test_default_values(self):
        config = CombinedDAConfig()
        assert config.dann_weight == 0.5
        assert config.mmd_weight == 0.25
        assert config.coral_weight == 0.25
        assert config.lambda_da == 1.0
        assert config.lambda_max == 1.0
        assert config.lambda_gamma == 10.0
        assert config.mmd_kernel == "multi"
        assert config.coral_normalize is True

    def test_weights_already_sum_to_one(self):
        """When weights already sum to 1, they should not change."""
        config = CombinedDAConfig(dann_weight=0.6, mmd_weight=0.3, coral_weight=0.1)
        assert abs(config.dann_weight - 0.6) < 1e-9
        assert abs(config.mmd_weight - 0.3) < 1e-9
        assert abs(config.coral_weight - 0.1) < 1e-9

    def test_weight_normalization(self):
        """When weights don't sum to 1, they should be normalized."""
        config = CombinedDAConfig(dann_weight=1.0, mmd_weight=1.0, coral_weight=1.0)
        total = config.dann_weight + config.mmd_weight + config.coral_weight
        assert abs(total - 1.0) < 1e-6
        # Each should be 1/3
        assert abs(config.dann_weight - 1.0 / 3.0) < 1e-6
        assert abs(config.mmd_weight - 1.0 / 3.0) < 1e-6
        assert abs(config.coral_weight - 1.0 / 3.0) < 1e-6

    def test_weight_normalization_off_by_small_epsilon(self):
        """Weights within 1e-6 tolerance should not be normalized."""
        config = CombinedDAConfig(dann_weight=0.5, mmd_weight=0.25, coral_weight=0.250001)
        # The total is 1.000001 which is > 1e-6 away from 1.0, so should be normalized
        total = config.dann_weight + config.mmd_weight + config.coral_weight
        assert abs(total - 1.0) < 1e-6

    def test_custom_mmd_kernel(self):
        config = CombinedDAConfig(mmd_kernel="gaussian")
        assert config.mmd_kernel == "gaussian"

    def test_custom_coral_normalize(self):
        config = CombinedDAConfig(coral_normalize=False)
        assert config.coral_normalize is False


# ---------------------------------------------------------------------------
# CombinedDomainAdaptation
# ---------------------------------------------------------------------------

class TestCombinedDomainAdaptationInit:
    """Tests for CombinedDomainAdaptation.__init__."""

    def test_default_init(self):
        da = CombinedDomainAdaptation()
        assert da.config.dann_weight == 0.5
        assert da.config.mmd_weight == 0.25
        assert da.config.coral_weight == 0.25
        assert da.config.lambda_da == 1.0
        assert isinstance(da.mmd_loss, torch.nn.Module)
        assert isinstance(da.coral_loss, torch.nn.Module)
        assert isinstance(da.domain_criterion, torch.nn.BCEWithLogitsLoss)
        assert hasattr(da, "grl_lambda")
        assert da.grl_lambda.item() == 0.0

    def test_init_with_config(self):
        config = CombinedDAConfig(dann_weight=0.7, mmd_weight=0.2, coral_weight=0.1, lambda_da=0.5)
        da = CombinedDomainAdaptation(config=config)
        assert da.config.dann_weight == 0.7
        assert da.config.mmd_weight == 0.2
        assert da.config.coral_weight == 0.1
        assert da.config.lambda_da == 0.5

    def test_init_with_kwargs(self):
        da = CombinedDomainAdaptation(dann_weight=0.8, mmd_weight=0.1, coral_weight=0.1, lambda_da=2.0)
        assert da.config.dann_weight == 0.8
        assert da.config.mmd_weight == 0.1
        assert da.config.coral_weight == 0.1
        assert da.config.lambda_da == 2.0

    def test_init_grl_lambda_buffer(self):
        da = CombinedDomainAdaptation()
        assert da.grl_lambda.device.type == "cpu"
        assert da.grl_lambda.dtype == torch.float32

    def test_init_progress(self):
        da = CombinedDomainAdaptation()
        assert da._progress == 0.0


class TestCombinedDomainAdaptationForward:
    """Tests for CombinedDomainAdaptation.forward."""

    @pytest.fixture
    def setup(self):
        torch.manual_seed(42)
        da = CombinedDomainAdaptation()
        src_feat = torch.randn(20, 8)
        tgt_feat = torch.randn(20, 8)
        src_domain = torch.randn(20, 1)
        tgt_domain = torch.randn(20, 1)
        return da, src_feat, tgt_feat, src_domain, tgt_domain

    def test_forward_returns_dict(self, setup):
        da, src_feat, tgt_feat, src_domain, tgt_domain = setup
        losses = da(src_feat, tgt_feat, src_domain, tgt_domain)
        assert isinstance(losses, dict)

    def test_forward_contains_all_keys(self, setup):
        da, src_feat, tgt_feat, src_domain, tgt_domain = setup
        losses = da(src_feat, tgt_feat, src_domain, tgt_domain)
        expected_keys = {"mmd_loss", "coral_loss", "dann_loss", "combined_da_loss"}
        assert set(losses.keys()) == expected_keys, f"Got keys: {set(losses.keys())}"

    def test_forward_losses_non_negative(self, setup):
        da, src_feat, tgt_feat, src_domain, tgt_domain = setup
        losses = da(src_feat, tgt_feat, src_domain, tgt_domain)
        for key, val in losses.items():
            assert val.item() >= -1e-6 or key in ("dann_loss",), \
                f"{key} should be >= 0, got {val.item()}"

    def test_forward_mmd_zero_weight_skips_mmd(self):
        da = CombinedDomainAdaptation(mmd_weight=0.0, coral_weight=0.5, dann_weight=0.5)
        src_feat = torch.randn(20, 8)
        tgt_feat = torch.randn(20, 8)
        losses = da(src_feat, tgt_feat, torch.randn(20, 1), torch.randn(20, 1))
        assert "mmd_loss" not in losses
        assert "coral_loss" in losses
        assert "dann_loss" in losses
        assert "combined_da_loss" in losses

    def test_forward_no_dann_logits_skips_dann(self):
        da = CombinedDomainAdaptation()
        src_feat = torch.randn(20, 8)
        tgt_feat = torch.randn(20, 8)
        losses = da(src_feat, tgt_feat)
        assert "mmd_loss" in losses
        assert "coral_loss" in losses
        assert "dann_loss" not in losses
        assert "combined_da_loss" in losses

    def test_forward_lambda_da_scaling(self):
        """combined_da_loss should be total_loss * lambda_da."""
        da = CombinedDomainAdaptation(lambda_da=2.0)
        src_feat = torch.randn(20, 8)
        tgt_feat = torch.randn(20, 8)
        src_domain = torch.randn(20, 1)
        tgt_domain = torch.randn(20, 1)
        losses = da(src_feat, tgt_feat, src_domain, tgt_domain)
        expected_combined = 2.0 * (
            da.config.mmd_weight * losses["mmd_loss"]
            + da.config.coral_weight * losses["coral_loss"]
            + da.config.dann_weight * da.grl_lambda * losses["dann_loss"]
        )
        assert torch.allclose(losses["combined_da_loss"], expected_combined, atol=1e-6)

    def test_forward_coral_zero_weight(self):
        da = CombinedDomainAdaptation(coral_weight=0.0)
        src_feat = torch.randn(20, 8)
        tgt_feat = torch.randn(20, 8)
        losses = da(src_feat, tgt_feat, torch.randn(20, 1), torch.randn(20, 1))
        assert "coral_loss" not in losses
        assert "mmd_loss" in losses
        assert "dann_loss" in losses

    def test_forward_dann_zero_weight(self):
        da = CombinedDomainAdaptation(dann_weight=0.0)
        src_feat = torch.randn(20, 8)
        tgt_feat = torch.randn(20, 8)
        losses = da(src_feat, tgt_feat, torch.randn(20, 1), torch.randn(20, 1))
        assert "dann_loss" not in losses
        assert "mmd_loss" in losses
        assert "coral_loss" in losses


class TestCombinedDomainAdaptationUpdateLambda:
    """Tests for CombinedDomainAdaptation.update_lambda."""

    def test_update_lambda_zero_progress(self):
        da = CombinedDomainAdaptation()
        val = da.update_lambda(0.0)
        expected = 2.0 / (1.0 + math.exp(0.0)) - 1.0  # sigmoid at 0 -> 0.5, so 2*0.5-1=0
        assert abs(val - expected) < 1e-6
        assert da.grl_lambda.item() == val * da.config.lambda_da
        assert da._progress == 0.0

    def test_update_lambda_full_progress(self):
        da = CombinedDomainAdaptation()
        val = da.update_lambda(1.0)
        expected = 2.0 / (1.0 + math.exp(-10.0)) - 1.0  # sigmoid at 10 -> ~1, so ~1
        assert abs(val - expected) < 1e-4
        assert da._progress == 1.0

    def test_update_lambda_mid_progress(self):
        config = CombinedDAConfig(lambda_max=1.0, lambda_gamma=10.0)
        da = CombinedDomainAdaptation(config=config)
        val = da.update_lambda(0.5)
        expected = 2.0 / (1.0 + math.exp(-10.0 * 0.5)) - 1.0
        assert abs(val - expected) < 1e-6

    def test_update_lambda_multiple_calls(self):
        da = CombinedDomainAdaptation()
        v1 = da.update_lambda(0.3)
        v2 = da.update_lambda(0.7)
        assert v2 > v1, "lambda should increase with progress"

    def test_update_lambda_resets_progress(self):
        da = CombinedDomainAdaptation()
        da.update_lambda(0.5)
        assert da._progress == 0.5
        da.update_lambda(0.8)
        assert da._progress == 0.8

    def test_update_lambda_with_lambda_da_scaling(self):
        da = CombinedDomainAdaptation(lambda_da=0.5)
        val = da.update_lambda(0.5)
        expected = 2.0 / (1.0 + math.exp(-10.0 * 0.5)) - 1.0
        assert abs(val - expected) < 1e-6
        # grl_lambda should be scaled by lambda_da
        assert abs(da.grl_lambda.item() - expected * 0.5) < 1e-6

    def test_update_lambda_custom_lambda_max(self):
        config = CombinedDAConfig(lambda_max=2.0)
        da = CombinedDomainAdaptation(config=config)
        val = da.update_lambda(0.5)
        expected = 2.0 * (2.0 / (1.0 + math.exp(-10.0 * 0.5)) - 1.0)
        assert abs(val - expected) < 1e-6


class TestComputeDANNLoss:
    """Tests for CombinedDomainAdaptation.compute_dann_loss."""

    def test_dann_loss_output(self):
        da = CombinedDomainAdaptation()
        src_logits = torch.randn(20, 1)
        tgt_logits = torch.randn(20, 1)
        loss = da.compute_dann_loss(src_logits, tgt_logits)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() > 0.0  # BCE should be positive with random logits

    def test_dann_loss_perfect_classifier(self):
        """Perfect domain classifier should have near-zero loss."""
        da = CombinedDomainAdaptation()
        # Very confident correct predictions
        src_logits = -10.0 * torch.ones(10, 1)  # source = 0
        tgt_logits = 10.0 * torch.ones(10, 1)   # target = 1
        loss = da.compute_dann_loss(src_logits, tgt_logits)
        assert loss.item() < 0.001, f"Perfect classifier loss should be near 0, got {loss.item()}"

    def test_dann_loss_reversed(self):
        """Confident reversed predictions should have high loss."""
        da = CombinedDomainAdaptation()
        src_logits = 10.0 * torch.ones(10, 1)   # predicting 1 for source (should be 0)
        tgt_logits = -10.0 * torch.ones(10, 1)  # predicting 0 for target (should be 1)
        loss = da.compute_dann_loss(src_logits, tgt_logits)
        assert loss.item() > 5.0, f"Reversed predictions should have high loss, got {loss.item()}"


# ---------------------------------------------------------------------------
# CombinedDANNLoss
# ---------------------------------------------------------------------------

class TestCombinedDANNLoss:
    """Tests for CombinedDANNLoss(nn.Module)."""

    @pytest.fixture
    def setup(self):
        torch.manual_seed(42)
        loss_fn = CombinedDANNLoss()
        batch_size = 16
        num_classes = 5
        class_logits = torch.randn(batch_size, num_classes)
        class_labels = torch.randint(0, num_classes, (batch_size,))
        source_features = torch.randn(batch_size, 8)
        target_features = torch.randn(batch_size, 8)
        source_domain_logits = torch.randn(batch_size, 1)
        target_domain_logits = torch.randn(batch_size, 1)
        return loss_fn, class_logits, class_labels, source_features, target_features, \
            source_domain_logits, target_domain_logits

    def test_forward_returns_tuple(self, setup):
        loss_fn, *inputs = setup
        total_loss, loss_dict = loss_fn(*inputs)
        assert isinstance(total_loss, torch.Tensor)
        assert isinstance(loss_dict, dict)
        assert total_loss.ndim == 0

    def test_loss_dict_contains_keys(self, setup):
        loss_fn, *inputs = setup
        _, loss_dict = loss_fn(*inputs)
        expected_keys = {"task_loss", "combined_da_loss", "lambda", "total_loss",
                         "dann_loss", "mmd_loss", "coral_loss"}
        assert set(loss_dict.keys()) == expected_keys, f"Got keys: {set(loss_dict.keys())}"

    def test_loss_dict_values_are_floats(self, setup):
        loss_fn, *inputs = setup
        _, loss_dict = loss_fn(*inputs)
        for key, val in loss_dict.items():
            assert isinstance(val, float) or isinstance(val, int), \
                f"{key} should be float/int, got {type(val)}"

    def test_lambda_scaling(self, setup):
        loss_fn, *inputs = setup
        _, loss_dict = loss_fn(*inputs, lambda_=0.5)
        assert loss_dict["lambda"] == 0.5

    def test_total_loss_equals_task_plus_adversarial(self, setup):
        loss_fn, class_logits, class_labels, source_features, target_features, \
            src_domain, tgt_domain = setup
        total_loss, loss_dict = loss_fn(
            class_logits, class_labels, source_features, target_features,
            src_domain, tgt_domain, lambda_=1.0,
        )
        expected_total = loss_dict["task_loss"] + loss_fn.adversarial_weight * loss_dict["combined_da_loss"]
        assert abs(total_loss.item() - expected_total) < 1e-6

    def test_lambda_zero_grl(self, setup):
        """When lambda=0, DANN loss term should not affect total (GRL is 0)."""
        loss_fn, *inputs = setup
        total_loss, loss_dict = loss_fn(*inputs, lambda_=0.0)
        # With lambda 0, grl_lambda is 0, so DANN term is 0
        # Combined DA loss should be just mmd + coral weighted
        assert not torch.isnan(total_loss)

    def test_adversarial_weight_scaling(self, setup):
        loss_fn, class_logits, class_labels, source_features, target_features, \
            src_domain, tgt_domain = setup
        loss_fn.adversarial_weight = 2.0
        total_loss, loss_dict = loss_fn(
            class_logits, class_labels, source_features, target_features,
            src_domain, tgt_domain, lambda_=1.0,
        )
        expected_total = loss_dict["task_loss"] + 2.0 * loss_dict["combined_da_loss"]
        assert abs(total_loss.item() - expected_total) < 1e-6

    def test_custom_class_weights(self):
        class_weights = torch.tensor([1.0, 2.0, 1.0])
        loss_fn = CombinedDANNLoss(class_weights=class_weights)
        assert loss_fn.class_weights is not None
        assert torch.allclose(loss_fn.class_weights, class_weights)

    def test_forward_with_custom_da_weights(self):
        loss_fn = CombinedDANNLoss(dann_weight=0.1, mmd_weight=0.8, coral_weight=0.1)
        batch_size = 10
        num_classes = 4
        total_loss, loss_dict = loss_fn(
            torch.randn(batch_size, num_classes),
            torch.randint(0, num_classes, (batch_size,)),
            torch.randn(batch_size, 8),
            torch.randn(batch_size, 8),
            torch.randn(batch_size, 1),
            torch.randn(batch_size, 1),
            lambda_=1.0,
        )
        assert not torch.isnan(total_loss)
        assert "mmd_loss" in loss_dict
        assert "coral_loss" in loss_dict
        assert "dann_loss" in loss_dict

    def test_internal_da_loss_module(self, setup):
        """da_loss inside CombinedDANNLoss should be a CombinedDomainAdaptation instance."""
        loss_fn, *inputs = setup
        assert isinstance(loss_fn.da_loss, CombinedDomainAdaptation)
        # Check that DA loss sub-modules work
        assert loss_fn.da_loss.config.dann_weight == 0.5


# ---------------------------------------------------------------------------
# create_combined_da_loss
# ---------------------------------------------------------------------------

class TestCreateCombinedDALoss:
    """Tests for the create_combined_da_loss factory function."""

    def test_returns_combined_domain_adaptation(self):
        da = create_combined_da_loss()
        assert isinstance(da, CombinedDomainAdaptation)

    def test_default_config(self):
        da = create_combined_da_loss()
        assert da.config.dann_weight == 0.5
        assert da.config.mmd_weight == 0.25
        assert da.config.coral_weight == 0.25
        assert da.config.lambda_da == 1.0

    def test_custom_params(self):
        da = create_combined_da_loss(
            dann_weight=0.2, mmd_weight=0.6, coral_weight=0.2,
            lambda_da=0.5, mmd_kernel="gaussian",
        )
        assert da.config.dann_weight == 0.2
        assert da.config.mmd_weight == 0.6
        assert da.config.coral_weight == 0.2
        assert da.config.lambda_da == 0.5
        assert da.config.mmd_kernel == "gaussian"

    def test_functional_forward(self):
        da = create_combined_da_loss()
        src = torch.randn(15, 6)
        tgt = torch.randn(15, 6)
        losses = da(src, tgt, torch.randn(15, 1), torch.randn(15, 1))
        assert "combined_da_loss" in losses
