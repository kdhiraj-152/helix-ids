"""
Tests for HELIX-IDS model architectures.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from helix_ids.models.helix_ids import HELIX_VARIANTS, HELIXIDS, FeatureBackbone, HELIXConfig


class TestHELIXVariants:
    """Test HELIX model variants."""

    def test_nano_forward_shape(self, sample_batch):
        """Test HELIX-Nano forward pass produces correct output shape."""
        model = HELIXIDS("nano")
        model.eval()

        with torch.no_grad():
            output = model(sample_batch)

        batch_size = sample_batch.shape[0]
        assert output["binary"].shape == (batch_size, 2)
        assert output["family"].shape == (batch_size, 4)
        assert "features" in output

    def test_lite_forward_shape(self, sample_batch):
        """Test HELIX-Lite forward pass produces correct output shape."""
        model = HELIXIDS("lite")
        model.eval()

        with torch.no_grad():
            output = model(sample_batch)

        batch_size = sample_batch.shape[0]
        assert output["binary"].shape == (batch_size, 2)
        assert output["family"].shape == (batch_size, 4)

    def test_full_forward_shape(self, sample_batch):
        """Test HELIX-Full forward pass produces correct output shape."""
        model = HELIXIDS("full")
        model.eval()

        with torch.no_grad():
            output = model(sample_batch)

        batch_size = sample_batch.shape[0]
        assert output["binary"].shape == (batch_size, 2)
        assert output["family"].shape == (batch_size, 4)

    def test_nano_parameter_count(self):
        """Test HELIX-Nano has fewer parameters than Lite."""
        nano = HELIXIDS("nano")
        lite = HELIXIDS("lite")

        nano_params = sum(p.numel() for p in nano.parameters())
        lite_params = sum(p.numel() for p in lite.parameters())

        assert nano_params < lite_params, (
            f"Nano ({nano_params}) should have fewer params than Lite ({lite_params})"
        )

    def test_lite_parameter_count(self):
        """Test HELIX-Lite has fewer parameters than Full."""
        lite = HELIXIDS("lite")
        full = HELIXIDS("full")

        lite_params = sum(p.numel() for p in lite.parameters())
        full_params = sum(p.numel() for p in full.parameters())

        assert lite_params < full_params, (
            f"Lite ({lite_params}) should have fewer params than Full ({full_params})"
        )

    def test_full_parameter_count(self):
        """Test HELIX-Full parameter count is reasonable."""
        full = HELIXIDS("full")
        full_params = sum(p.numel() for p in full.parameters())

        # Should be less than 500K parameters
        assert full_params < 500_000, f"Full model has {full_params} params, expected < 500K"

    def test_predict_method(self, sample_batch):
        """Test predict method returns valid class indices."""
        model = HELIXIDS("lite")
        model.eval()

        predictions = model.predict(sample_batch)

        assert predictions.shape == (sample_batch.shape[0],)
        assert predictions.min() >= 0
        assert predictions.max() <= 4  # 5 classes: 0-4

    def test_predict_proba(self, sample_batch):
        """Test predict_proba returns valid probabilities."""
        model = HELIXIDS("lite")
        model.eval()

        proba = model.predict_proba(sample_batch)

        assert proba.shape == (sample_batch.shape[0], 5)
        assert torch.allclose(proba.sum(dim=1), torch.ones(proba.shape[0]), atol=1e-5)
        assert (proba >= 0).all()
        assert (proba <= 1).all()


class TestHELIXConfig:
    """Test HELIX configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = HELIXConfig()

        assert config.variant == "lite"
        assert config.input_dim == 41
        assert config.num_classes == 5
        assert config.dropout > 0

    def test_variant_configs_exist(self):
        """Test all variant configs are defined."""
        assert "nano" in HELIX_VARIANTS
        assert "lite" in HELIX_VARIANTS
        assert "full" in HELIX_VARIANTS

    def test_invalid_variant_raises(self):
        """Test invalid variant raises ValueError."""
        with pytest.raises(ValueError):
            HELIXIDS("invalid_variant")

    def test_custom_config(self):
        """Test model with custom configuration."""
        config = HELIXConfig(
            variant="lite",
            input_dim=100,
            hidden_dim=64,
            num_classes=5,
        )
        model = HELIXIDS(config)

        x = torch.randn(8, 100)
        output = model(x)

        assert output["binary"].shape == (8, 2)


class TestFeatureBackbone:
    """Test feature backbone network."""

    def test_backbone_forward(self):
        """Test backbone forward pass."""
        backbone = FeatureBackbone(input_dim=41, hidden_dims=(64, 32), output_dim=16, dropout=0.1)

        x = torch.randn(8, 41)
        output = backbone(x)

        assert output.shape == (8, 16)

    def test_backbone_dropout(self):
        """Test backbone applies dropout in training mode."""
        backbone = FeatureBackbone(
            input_dim=41,
            hidden_dims=(64, 32),
            output_dim=16,
            dropout=0.5,  # High dropout for testing
        )

        x = torch.randn(8, 41)

        # Training mode - outputs should vary
        backbone.train()
        out1 = backbone(x).clone()
        out2 = backbone(x).clone()

        # Outputs differ in training due to dropout
        assert not torch.allclose(out1, out2)

        # Eval mode - outputs should be deterministic
        backbone.eval()
        out3 = backbone(x).clone()
        out4 = backbone(x).clone()

        assert torch.allclose(out3, out4)
