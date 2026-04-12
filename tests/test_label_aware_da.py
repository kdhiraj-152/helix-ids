"""Tests for label-aware domain adaptation."""

import pytest
import torch
import torch.nn.functional as F

from src.helix_ids.models.adaptation import (
    ClassConditionalMMDLoss,
    ConditionalDomainDiscriminator,
    LabelAwareDAConfig,
    LabelAwareDALoss,
    LabelAwareDANN,
    PartialTransferReweighter,
    create_label_aware_dann,
)


class TestPartialTransferReweighter:
    """Test partial transfer reweighting for label shift."""

    def test_initialization(self):
        """Test reweighter initialization."""
        reweighter = PartialTransferReweighter(num_classes=5, reweight_coeff=0.5)
        assert reweighter.num_classes == 5
        assert reweighter.reweight_coeff == 0.5
        assert reweighter.target_class_dist.shape == (5,)

    def test_update_target_distribution(self):
        """Test updating target class distribution."""
        reweighter = PartialTransferReweighter(num_classes=5)

        # Create pseudo-labels with uneven distribution
        pseudo_labels = torch.tensor([0, 0, 1, 1, 1, 2, 3, 4, 4])
        reweighter.update_target_distribution(pseudo_labels)

        # Check distribution is updated
        assert reweighter.target_class_dist[0].item() > 0
        assert abs(reweighter.target_class_dist.sum().item() - 1.0) < 1e-5

    def test_compute_sample_weights(self):
        """Test sample weight computation for partial transfer."""
        reweighter = PartialTransferReweighter(num_classes=5, reweight_coeff=1.0)

        # Set target distribution (classes 0, 1, 2 present)
        target_dist = torch.tensor([0.3, 0.3, 0.4, 0.0, 0.0])
        reweighter.target_class_dist = target_dist

        # Source labels with all 5 classes
        source_labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 4])

        weights = reweighter.compute_sample_weights(source_labels)

        assert weights.shape == source_labels.shape
        assert (weights > 0).all()
        # Classes 3, 4 (not in target) should have lower weights
        assert weights[6] < weights[0]  # Class 3 < Class 0

    def test_partial_transfer_interpolation(self):
        """Test interpolation between no reweighting and full reweighting."""
        reweighter = PartialTransferReweighter(num_classes=5, reweight_coeff=0.0)

        target_dist = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2])
        reweighter.target_class_dist = target_dist

        source_labels = torch.tensor([0, 1, 2, 3, 4])
        weights = reweighter.compute_sample_weights(source_labels)

        # With coeff=0, weights should be close to 1.0
        assert weights.mean().item() >= 0.9


class TestConditionalDomainDiscriminator:
    """Test conditional (per-class) domain discriminators."""

    def test_initialization(self):
        """Test discriminator initialization."""
        disc = ConditionalDomainDiscriminator(
            num_classes=5,
            input_dim=64,
            hidden_dims=[32, 16],
        )
        assert len(disc.discriminators) == 5
        assert disc.num_classes == 5

    def test_forward_with_labels(self):
        """Test forward pass with class labels."""
        disc = ConditionalDomainDiscriminator(
            num_classes=5,
            input_dim=64,
            hidden_dims=[32, 16],
        )

        batch_size = 16
        features = torch.randn(batch_size, 64)
        labels = torch.randint(0, 5, (batch_size,))

        logits = disc(features, labels)

        assert logits.shape == (batch_size, 1)
        assert not torch.isnan(logits).any()

    def test_forward_per_class(self):
        """Test per-class forward pass without label filtering."""
        disc = ConditionalDomainDiscriminator(
            num_classes=5,
            input_dim=64,
        )

        features = torch.randn(16, 64)
        logits_list = disc.forward_per_class(features)

        assert len(logits_list) == 5
        for logits in logits_list:
            assert logits.shape == (16, 1)


class TestClassConditionalMMDLoss:
    """Test class-conditional MMD loss."""

    def test_initialization(self):
        """Test MMD loss initialization."""
        mmd = ClassConditionalMMDLoss(num_classes=5, kernel="multi")
        assert mmd.num_classes == 5
        assert mmd.kernel == "multi"

    def test_forward_single_class(self):
        """Test forward pass with single class present."""
        mmd = ClassConditionalMMDLoss(num_classes=5, kernel="multi")

        # Only class 0 present in both domains
        source_feats = torch.randn(20, 32)
        target_feats = torch.randn(15, 32)
        source_labels = torch.zeros(20, dtype=torch.long)
        target_labels = torch.zeros(15, dtype=torch.long)

        loss, loss_dict = mmd(source_feats, target_feats, source_labels, target_labels)

        assert loss.item() >= 0
        assert "total" in loss_dict
        assert "mmd_class_0" in loss_dict

    def test_forward_multiple_classes(self):
        """Test forward pass with multiple classes."""
        mmd = ClassConditionalMMDLoss(num_classes=3, kernel="gaussian")

        source_feats = torch.randn(30, 32)
        target_feats = torch.randn(25, 32)
        source_labels = torch.cat(
            [
                torch.zeros(10, dtype=torch.long),
                torch.ones(10, dtype=torch.long),
                torch.full((10,), 2, dtype=torch.long),
            ]
        )
        target_labels = torch.cat(
            [
                torch.zeros(8, dtype=torch.long),
                torch.ones(8, dtype=torch.long),
                torch.full((9,), 2, dtype=torch.long),
            ]
        )

        loss, loss_dict = mmd(source_feats, target_feats, source_labels, target_labels)

        assert loss.item() >= 0
        assert "total" in loss_dict
        # Should have losses for classes that appear in both domains
        class_losses = [k for k in loss_dict if k.startswith("mmd_class")]
        assert len(class_losses) > 0

    def test_skip_missing_classes(self):
        """Test that missing classes are skipped."""
        mmd = ClassConditionalMMDLoss(num_classes=5)

        source_feats = torch.randn(10, 32)
        target_feats = torch.randn(10, 32)
        source_labels = torch.zeros(10, dtype=torch.long)
        target_labels = torch.ones(10, dtype=torch.long)  # Different class

        loss, loss_dict = mmd(source_feats, target_feats, source_labels, target_labels)

        # Should return 0 loss when no shared classes
        assert loss.item() == 0.0


class TestLabelAwareDANN:
    """Test label-aware DANN model."""

    def test_initialization(self):
        """Test model initialization."""
        model = LabelAwareDANN()
        assert model.config.num_classes == 5
        assert model.feature_dim == 64

    def test_forward_classification(self):
        """Test forward pass for classification."""
        model = LabelAwareDANN()
        x = torch.randn(16, 41)

        logits = model(x)
        assert logits.shape == (16, 5)

        logits, features = model(x, return_features=True)
        assert logits.shape == (16, 5)
        assert features.shape == (16, 64)

    def test_forward_label_aware_da(self):
        """Test forward pass for label-aware DA training."""
        model = LabelAwareDANN()

        x_source = torch.randn(16, 41)
        x_target = torch.randn(12, 41)
        y_source = torch.randint(0, 5, (16,))
        y_target = torch.randint(0, 5, (12,))

        outputs = model.forward_label_aware_da(
            x_source,
            x_target,
            y_source,
            y_target,
        )

        assert "class_logits" in outputs
        assert "domain_logits_source" in outputs
        assert "domain_logits_target" in outputs
        assert "features_source" in outputs
        assert "mmd_loss" in outputs
        assert "coral_loss" in outputs

        assert outputs["class_logits"].shape == (16, 5)
        assert outputs["domain_logits_source"].shape == (16, 1)
        assert outputs["domain_logits_target"].shape == (12, 1)

    def test_update_lambda(self):
        """Test gradient reversal coefficient scheduling."""
        model = LabelAwareDANN()

        lambda0 = model.update_lambda(0.0)
        assert lambda0 >= 0

        lambda_mid = model.update_lambda(0.5)
        assert lambda_mid > lambda0

        lambda_end = model.update_lambda(1.0)
        assert lambda_end >= lambda_mid

    def test_get_features(self):
        """Test feature extraction."""
        model = LabelAwareDANN()
        x = torch.randn(16, 41)

        features = model.get_features(x)
        assert features.shape == (16, 64)

    def test_different_configurations(self):
        """Test different configuration options."""
        # Without conditional DANN
        config1 = LabelAwareDAConfig(use_conditional_dann=False)
        model1 = LabelAwareDANN(config1)
        assert model1.config.use_conditional_dann is False

        # Without partial transfer
        config2 = LabelAwareDAConfig(use_partial_transfer=False)
        model2 = LabelAwareDANN(config2)
        assert model2.reweighter is None

        # Without class-conditional MMD
        config3 = LabelAwareDAConfig(use_class_conditional_mmd=False)
        model3 = LabelAwareDANN(config3)
        assert not isinstance(model3.mmd_loss, ClassConditionalMMDLoss)


class TestLabelAwareDALoss:
    """Test combined label-aware DA loss."""

    def test_initialization(self):
        """Test loss function initialization."""
        loss_fn = LabelAwareDALoss(adversarial_weight=1.0, mmd_weight=0.5)
        assert loss_fn.adversarial_weight == 1.0
        assert loss_fn.mmd_weight == 0.5

    def test_forward(self):
        """Test loss computation."""
        loss_fn = LabelAwareDALoss()

        # Create forward outputs
        forward_outputs = {
            "class_logits": torch.randn(16, 5),
            "domain_logits_source": torch.randn(16, 1),
            "domain_logits_target": torch.randn(12, 1),
            "mmd_loss": torch.tensor(0.1),
            "coral_loss": torch.tensor(0.05),
            "sample_weights": None,
            "mmd_dict": {"mmd_class_0": 0.08},
        }

        total_loss, loss_dict = loss_fn(forward_outputs, lambda_=0.5)

        assert total_loss.item() >= 0
        assert "domain_loss" in loss_dict
        assert "mmd_loss" in loss_dict
        assert "coral_loss" in loss_dict
        assert "adaptation_loss" in loss_dict
        assert loss_dict["lambda"] == 0.5


class TestFactoryFunction:
    """Test factory function."""

    def test_create_label_aware_dann(self):
        """Test create_label_aware_dann factory."""
        model = create_label_aware_dann(
            input_dim=41,
            num_classes=5,
            lambda_max=1.0,
        )

        assert isinstance(model, LabelAwareDANN)
        assert model.config.input_dim == 41
        assert model.config.num_classes == 5
        assert model.config.lambda_max == 1.0

        # Test with custom configuration
        model2 = create_label_aware_dann(
            use_conditional=True,
            use_partial_transfer=True,
            use_class_conditional_mmd=True,
        )

        assert model2.config.use_conditional_dann is True
        assert model2.config.use_partial_transfer is True
        assert model2.config.use_class_conditional_mmd is True


class TestIntegration:
    """Integration tests for label-aware DA training."""

    def test_full_training_step(self):
        """Test a complete training step."""
        model = create_label_aware_dann()
        loss_fn = LabelAwareDALoss(adversarial_weight=1.0, mmd_weight=0.5)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Simulate training data
        x_source = torch.randn(32, 41)
        x_target = torch.randn(32, 41)
        y_source = torch.randint(0, 5, (32,))
        y_target = torch.randint(0, 5, (32,))
        y_labels = torch.randint(0, 5, (32,))

        # Forward pass
        outputs = model.forward_label_aware_da(
            x_source,
            x_target,
            y_source,
            y_target,
        )

        # Compute loss
        adaptation_loss, loss_dict = loss_fn(outputs)

        # Task loss (separate)
        task_loss = F.cross_entropy(outputs["class_logits"], y_labels)

        total_loss = task_loss + adaptation_loss

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Check parameters were updated
        assert list(model.parameters())[0].grad is not None

    def test_cross_dataset_transfer_scenario(self):
        """Test realistic cross-dataset transfer scenario.

        Simulates transfer from source (NSL-KDD with all classes)
        to target (UNSW-NB15 missing U2R) with label shift.
        """
        num_classes = 5
        create_label_aware_dann(num_classes=num_classes)
        reweighter = PartialTransferReweighter(num_classes=num_classes)

        # Source: balanced all classes
        source_labels = torch.cat([torch.full((20,), i) for i in range(num_classes)])

        # Target: missing class 4 (U2R-like)
        target_labels = torch.cat([torch.full((20,), i) for i in range(num_classes - 1)])

        # Update reweighter with target distribution
        reweighter.update_target_distribution(target_labels)

        # Compute weights for source samples
        weights = reweighter.compute_sample_weights(source_labels)

        # Class 4 samples should be down-weighted
        class_4_indices = (source_labels == 4).nonzero(as_tuple=True)[0]
        other_indices = (source_labels < 4).nonzero(as_tuple=True)[0]

        if len(class_4_indices) > 0 and len(other_indices) > 0:
            assert weights[class_4_indices].mean() < weights[other_indices].mean()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
