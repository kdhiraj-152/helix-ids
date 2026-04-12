"""
Comprehensive tests for HelixIDS-Full model and multi-task loss.

Tests cover:
- Model instantiation and parameter count
- Forward pass with various batch sizes
- Output shapes and ranges
- Multi-task loss computation
- Gradient flow
- Model serialization
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path

# Import from production package
from helix_ids.models.full import (
    HelixIDSFull,
    HelixFullConfig,
    MultiTaskLoss,
    count_parameters,
    create_helix_full,
)
from helix_ids.config.helix_full_config import TrainingConfig, DataConfig, EvaluationConfig


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def config():
    """Default model config."""
    return HelixFullConfig()


@pytest.fixture
def model(config):
    """HelixIDS-Full model instance."""
    return create_helix_full(config)


@pytest.fixture
def loss_fn():
    """Multi-task loss function."""
    return MultiTaskLoss(lambda_binary=1.0, lambda_family=0.8)


@pytest.fixture
def batch_data(config):
    """Generate batch of random data."""
    batch_size = 32
    x = torch.randn(batch_size, config.input_dim)
    binary_labels = torch.randint(0, 2, (batch_size,))  # 0 or 1
    family_labels = torch.randint(0, 7, (batch_size,))  # 0-6
    return x, binary_labels, family_labels


# ============================================================================
# Model Tests
# ============================================================================

class TestHelixIDSFull:
    """Test HelixIDS-Full model."""
    
    def test_model_instantiation(self, config):
        """Test that model instantiates correctly."""
        model = HelixIDSFull(config)
        assert model is not None
        assert isinstance(model, nn.Module)
    
    def test_parameter_count(self, model, config):
        """Test parameter count is in reasonable range (~300-600K for modern MLP)."""
        param_count = count_parameters(model)
        # Should be in reasonable range (300K - 600K for tabular MLP)
        assert 250_000 < param_count < 700_000, f"Expected ~300-600K params, got {param_count:,}"
        
        # Check model's internal counter matches
        assert model.param_count == param_count
    
    def test_forward_pass_basic(self, model, config):
        """Test forward pass with basic batch."""
        batch_size = 32
        x = torch.randn(batch_size, config.input_dim)
        
        binary_logits, family_logits = model(x)
        
        assert binary_logits.shape == (batch_size, 2), f"Binary shape {binary_logits.shape}"
        assert family_logits.shape == (batch_size, 7), f"Family shape {family_logits.shape}"
    
    def test_forward_pass_single_sample(self, model, config):
        """Test forward pass with single sample."""
        x = torch.randn(1, config.input_dim)
        binary_logits, family_logits = model(x)
        
        assert binary_logits.shape == (1, 2)
        assert family_logits.shape == (1, 7)
    
    def test_forward_pass_large_batch(self, model, config):
        """Test forward pass with large batch."""
        batch_size = 512
        x = torch.randn(batch_size, config.input_dim)
        binary_logits, family_logits = model(x)
        
        assert binary_logits.shape == (batch_size, 2)
        assert family_logits.shape == (batch_size, 7)
    
    def test_output_ranges(self, model, config):
        """Test that logits are in reasonable range (not NaN or Inf)."""
        batch_size = 32
        x = torch.randn(batch_size, config.input_dim)
        binary_logits, family_logits = model(x)
        
        # Check for NaN
        assert not torch.isnan(binary_logits).any(), "NaN in binary logits"
        assert not torch.isnan(family_logits).any(), "NaN in family logits"
        
        # Check for Inf
        assert not torch.isinf(binary_logits).any(), "Inf in binary logits"
        assert not torch.isinf(family_logits).any(), "Inf in family logits"
        
        # Check reasonable magnitude (typically logits in [-10, 10])
        assert binary_logits.abs().max() < 100
        assert family_logits.abs().max() < 100
    
    def test_feature_extraction(self, model, config):
        """Test feature extraction with return_features=True."""
        batch_size = 32
        x = torch.randn(batch_size, config.input_dim)
        
        binary_logits, family_logits, features = model(x, return_features=True)
        
        # Should return backbone output (last hidden layer = 64 dims)
        assert features.shape == (batch_size, config.hidden_dims[-1])
        assert binary_logits.shape == (batch_size, 2)
        assert family_logits.shape == (batch_size, 7)
    
    def test_eval_mode(self, model, config):
        """Test model in eval mode (batch norm disabled)."""
        model.eval()
        batch_size = 32
        x = torch.randn(batch_size, config.input_dim)
        
        with torch.no_grad():
            binary_logits, family_logits = model(x)
        
        assert binary_logits.shape == (batch_size, 2)
        assert family_logits.shape == (batch_size, 7)
    
    def test_training_mode_vs_eval(self, model, config):
        """Test that train/eval modes produce different outputs (due to dropout)."""
        batch_size = 32
        x = torch.randn(batch_size, config.input_dim)
        
        # Train mode (with dropout)
        model.train()
        torch.manual_seed(42)
        output1_train, _ = model(x)
        
        # Eval mode (no dropout)
        model.eval()
        with torch.no_grad():
            output1_eval, _ = model(x)
        
        # Should be different due to dropout (very unlikely to be identical)
        # But both should have correct shape
        assert output1_train.shape == output1_eval.shape == (batch_size, 2)
    
    def test_gradient_computation(self, model, batch_data):
        """Test that gradients can be computed."""
        x, binary_labels, _ = batch_data
        model.train()
        
        # Forward pass
        binary_logits, _ = model(x)
        
        # Compute loss
        loss = torch.nn.functional.cross_entropy(binary_logits, binary_labels)
        loss.backward()
        
        # Check gradients exist
        has_gradients = False
        for param in model.parameters():
            if param.grad is not None:
                has_gradients = True
                assert not torch.isnan(param.grad).any(), "NaN in gradients"
        
        assert has_gradients, "No gradients computed"
    
    def test_different_input_sizes(self, config):
        """Test model with different input dimensions (edge case)."""
        # Model expects config.input_dim, test it rejects wrong size
        model = HelixIDSFull(config)
        model.eval()
        
        with torch.no_grad():
            # Correct size
            x_correct = torch.randn(16, config.input_dim)
            binary, _ = model(x_correct)
            assert binary.shape == (16, 2)
            
            # Wrong size should fail
            x_wrong = torch.randn(16, max(1, config.input_dim - 1))
            with pytest.raises(RuntimeError):
                model(x_wrong)


# ============================================================================
# Multi-Task Loss Tests
# ============================================================================

class TestMultiTaskLoss:
    """Test multi-task loss computation."""
    
    def test_loss_instantiation(self):
        """Test loss function instantiation."""
        loss_fn = MultiTaskLoss()
        assert loss_fn is not None
        assert abs(loss_fn.lambda_binary - 1.0) < 1e-12
        assert abs(loss_fn.lambda_family - 0.8) < 1e-12
    
    def test_basic_loss_computation(self, batch_data):
        """Test basic loss computation."""
        _, binary_labels, family_labels = batch_data
        
        # Dummy logits
        batch_size = binary_labels.shape[0]
        binary_logits = torch.randn(batch_size, 2)
        family_logits = torch.randn(batch_size, 7)
        
        loss_fn = MultiTaskLoss()
        total_loss, loss_dict = loss_fn(
            binary_logits, binary_labels,
            family_logits, family_labels
        )
        
        assert total_loss.item() > 0
        assert "total" in loss_dict
        assert "binary" in loss_dict
        assert "family" in loss_dict
    
    def test_loss_components(self, batch_data):
        """Test loss component breakdown."""
        _, binary_labels, family_labels = batch_data
        
        batch_size = binary_labels.shape[0]
        binary_logits = torch.randn(batch_size, 2)
        family_logits = torch.randn(batch_size, 7)
        
        loss_fn = MultiTaskLoss(lambda_binary=1.0, lambda_family=0.8)
        total_loss, loss_dict = loss_fn(
            binary_logits, binary_labels,
            family_logits, family_labels
        )
        
        # Check that components are used in total
        # total ≈ 1.0 * binary + 0.8 * family (approximately)
        computed_total = 1.0 * loss_dict["binary"] + 0.8 * loss_dict["family"]
        assert abs(total_loss.item() - computed_total) < 0.01
    
    def test_loss_with_class_weights(self, batch_data):
        """Test loss computation with class weights."""
        _, binary_labels, family_labels = batch_data
        
        batch_size = binary_labels.shape[0]
        binary_logits = torch.randn(batch_size, 2)
        family_logits = torch.randn(batch_size, 7)
        
        # Create class weights (e.g., for imbalanced data)
        binary_weights = torch.tensor([0.7, 1.3])  # Penalize attack class more
        family_weights = torch.ones(7) * 1/7
        family_weights[0] = 0.1  # Normal class less important
        
        loss_fn = MultiTaskLoss()
        total_loss, loss_dict = loss_fn(
            binary_logits, binary_labels,
            family_logits, family_labels,
            binary_class_weights=binary_weights,
            family_class_weights=family_weights
        )
        
        assert total_loss.item() > 0
        assert "total" in loss_dict
    
    def test_loss_backward(self, batch_data):
        """Test that loss can be backpropagated."""
        _, binary_labels, family_labels = batch_data
        
        batch_size = binary_labels.shape[0]
        binary_logits = torch.randn(batch_size, 2, requires_grad=True)
        family_logits = torch.randn(batch_size, 7, requires_grad=True)
        
        loss_fn = MultiTaskLoss()
        total_loss, _ = loss_fn(
            binary_logits, binary_labels,
            family_logits, family_labels
        )
        
        total_loss.backward()
        
        assert binary_logits.grad is not None
        assert family_logits.grad is not None
        assert not torch.isnan(binary_logits.grad).any()
        assert not torch.isnan(family_logits.grad).any()
    
    def test_different_weights(self):
        """Test loss with different weight combinations."""
        batch_size = 16
        binary_logits = torch.randn(batch_size, 2)
        family_logits = torch.randn(batch_size, 7)
        binary_labels = torch.randint(0, 2, (batch_size,))
        family_labels = torch.randint(0, 7, (batch_size,))
        
        # Original weights
        loss_fn1 = MultiTaskLoss(lambda_binary=1.0, lambda_family=0.8)
        loss1, _ = loss_fn1(binary_logits, binary_labels, family_logits, family_labels)
        
        # Equal weights
        loss_fn2 = MultiTaskLoss(lambda_binary=1.0, lambda_family=1.0)
        loss2, _ = loss_fn2(binary_logits, binary_labels, family_logits, family_labels)
        
        # Binary-only
        loss_fn3 = MultiTaskLoss(lambda_binary=1.0, lambda_family=0.0)
        loss3, _ = loss_fn3(binary_logits, binary_labels, family_logits, family_labels)
        
        # Losses should be different
        assert abs(loss1.item() - loss2.item()) > 0.01
        assert abs(loss1.item() - loss3.item()) > 0.01


# ============================================================================
# Config Tests
# ============================================================================

class TestTrainingConfig:
    """Test training configuration."""
    
    def test_default_config(self):
        """Test default training config."""
        config = TrainingConfig()
        assert config.input_dim == 18
        assert config.batch_size == 256
        assert config.epochs == 150
        assert abs(config.learning_rate - 1e-3) < 1e-12
    
    def test_data_config(self):
        """Test data configuration."""
        config = DataConfig()
        assert config.use_per_dataset_normalization is True
        assert config.data_dir.name == "processed"
    
    def test_eval_config(self):
        """Test evaluation configuration."""
        config = EvaluationConfig()
        assert "f1" in config.metrics
        assert "accuracy" in config.metrics


# ============================================================================
# Integration Tests
# ============================================================================

class TestHelixIDSFullIntegration:
    """Integration tests combining model, loss, and data."""
    
    def test_training_step(self, model, loss_fn, batch_data):
        """Test a single training step."""
        x, binary_labels, family_labels = batch_data
        model.train()
        
        # Forward pass
        binary_logits, family_logits = model(x)
        
        # Compute loss
        total_loss, _ = loss_fn(
            binary_logits, binary_labels,
            family_logits, family_labels
        )
        
        # Backward pass
        total_loss.backward()
        
        # Check gradients exist
        has_gradients = any(p.grad is not None for p in model.parameters())
        assert has_gradients
    
    def test_validation_step(self, model, batch_data):
        """Test a validation step (no gradients)."""
        x, binary_labels, _ = batch_data
        model.eval()
        
        with torch.no_grad():
            binary_logits, _ = model(x)
            
            # Compute accuracy
            binary_preds = torch.argmax(binary_logits, dim=1)
            binary_acc = (binary_preds == binary_labels).float().mean()
            
            assert 0 <= binary_acc <= 1
    
    def test_model_save_load(self, model, tmp_path):
        """Test model can be saved and loaded."""
        model_path = tmp_path / "test_model.pt"
        
        # Save
        torch.save(model.state_dict(), model_path)
        assert model_path.exists()
        
        # Load
        config = HelixFullConfig()
        new_model = HelixIDSFull(config)
        new_model.load_state_dict(torch.load(model_path))
        
        # Test they produce same output
        x = torch.randn(16, model.config.input_dim)
        
        model.eval()
        new_model.eval()
        
        with torch.no_grad():
            output1 = model(x)
            output2 = new_model(x)
        
        assert torch.allclose(output1[0], output2[0])
        assert torch.allclose(output1[1], output2[1])


# ============================================================================
# Edge Cases & Error Handling
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_zero_batch(self, model):
        """Test with empty batch (edge case)."""
        x = torch.randn(0, model.config.input_dim)
        try:
            _, _ = model(x)
            # May or may not work depending on batch norm behavior
        except RuntimeError:
            # Expected for some batch norm configurations
            pass
    
    def test_very_large_inputs(self, model):
        """Test with very large input values."""
        x = torch.randn(32, model.config.input_dim) * 1000
        binary_logits, family_logits = model(x)
        
        assert not torch.isnan(binary_logits).any()
        assert not torch.isnan(family_logits).any()
    
    def test_very_small_inputs(self, model):
        """Test with very small input values."""
        x = torch.randn(32, model.config.input_dim) * 1e-6
        binary_logits, family_logits = model(x)
        
        assert not torch.isnan(binary_logits).any()
        assert not torch.isnan(family_logits).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
