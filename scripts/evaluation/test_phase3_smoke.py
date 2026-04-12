"""
Smoke test for Phase 3 training setup.

Validates that:
1. HelixIDSFull model instantiation
2. Multi-task loss computation
3. Data loading with proper label conversion
4. Trainer initialization
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Add project to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from helix_ids.config.helix_full_config import TrainingConfig
from helix_ids.data.feature_harmonization import labels_to_multi_task
from helix_ids.models.full import MultiTaskLoss, create_helix_full


def test_model_creation():
    """Test model instantiation."""
    print("Testing model creation...")
    model = create_helix_full()
    print(f"✅ Model created with {model.param_count:,} parameters")
    return model


def test_loss_computation(model):
    """Test multi-task loss."""
    print("\nTesting loss computation...")
    loss_fn = MultiTaskLoss(lambda_binary=1.0, lambda_family=0.8)

    # Dummy batch
    batch_size = 16
    x = torch.randn(batch_size, 31)
    binary_labels = torch.randint(0, 2, (batch_size,))
    family_labels = torch.randint(0, 7, (batch_size,))

    # Forward
    binary_logits, family_logits = model(x)
    loss, loss_dict = loss_fn(binary_logits, binary_labels, family_logits, family_labels)

    print(f"✅ Loss computed: {loss.item():.4f}")
    print(f"   Binary loss: {loss_dict['binary']:.4f}")
    print(f"   Family loss: {loss_dict['family']:.4f}")
    return loss_fn


def test_label_conversion():
    """Test label conversion utilities."""
    print("\nTesting label conversion...")

    family_labels = np.array([0, 1, 2, 0, 3, 1])
    binary_labels, family_labels_out = labels_to_multi_task(family_labels)

    print("✅ Converted labels:")
    print(f"   Original (family): {family_labels}")
    print(f"   Binary: {binary_labels}")
    print(f"   Family: {family_labels_out}")


def test_dataloader():
    """Test DataLoader with multi-task labels."""
    print("\nTesting DataLoader...")

    # Synthetic data
    X = torch.randn(100, 31)
    y_family = torch.randint(0, 7, (100,))

    # Create dataset and loader
    dataset = TensorDataset(X, y_family)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

    # Convert first batch to binary + family
    for batch_x, batch_y_family in loader:
        batch_y_binary = torch.where(batch_y_family == 0, 0, 1)
        print(
            f"✅ Batch loaded: X shape {batch_x.shape}, y_binary unique {batch_y_binary.unique().tolist()}, y_family unique {batch_y_family.unique().tolist()}"
        )
        break


def test_training_config():
    """Test training configuration."""
    print("\nTesting training configuration...")

    config = TrainingConfig(
        batch_size=256,
        epochs=150,
        learning_rate=1e-3,
    )

    print("✅ Training config:")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Epochs: {config.epochs}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Device: {config.device}")


def main():
    """Run all smoke tests."""
    print("=" * 80)
    print("Phase 3: Training Setup Smoke Test")
    print("=" * 80)

    try:
        model = test_model_creation()
        test_loss_computation(model)
        test_label_conversion()
        test_dataloader()
        test_training_config()

        print("\n" + "=" * 80)
        print("✅ All smoke tests passed!")
        print("=" * 80)
        return 0

    except Exception as e:
        print(f"\n❌ Smoke test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
