"""
Training configuration for HelixIDS-Full.

Specifies:
- Model hyperparameters
- Training loop parameters (batch size, epochs, learning rate, warmup)
- Loss weights for multi-task learning
- Class balancing strategy
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TrainingConfig:
    """Training hyperparameters for HelixIDS-Full."""

    # ===== Model Architecture =====
    input_dim: int = 18  # invariant cross-dataset behavior features
    hidden_dims: tuple = field(default_factory=lambda: (256, 192, 128, 64))
    dropout_rates: tuple = field(default_factory=lambda: (0.3, 0.3, 0.25, 0.2))
    use_batch_norm: bool = True
    activation: str = "relu"

    # ===== Multi-Task Loss Weights =====
    lambda_binary: float = 1.0  # Weight for binary head (Normal vs Attack)
    lambda_family: float = 0.8  # Weight for family head (7-class)

    # ===== Training Loop =====
    batch_size: int = 256  # Default batch size for M4 MPS
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 150

    # ===== Warmup & Schedule =====
    warmup_epochs: int = 5  # Linear warmup
    warmup_init_lr: float = 1e-5
    lr_decay_factor: float = 0.1
    lr_decay_steps: tuple = field(default_factory=lambda: (50, 100, 140))  # Decay at these epochs

    # ===== Gradient Clipping =====
    max_grad_norm: float = 1.0

    # ===== Early Stopping =====
    early_stopping_patience: int = 15  # Stop if val loss doesn't improve for N epochs
    early_stopping_threshold: float = 1e-4  # Min improvement to count as progress
    min_family_minority_recall_for_best: float = 0.70

    # ===== Class Weighting =====
    use_class_weights: bool = True  # Use inverse frequency weighting

    # ===== Device =====
    device: str = "mps"  # M4 MacBook MPS acceleration

    # ===== Logging =====
    log_interval: int = 50  # Log metrics every N batches
    val_interval: int = 1  # Validate every N epochs

    # ===== Checkpoints =====
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints/helix_full"))
    save_best_only: bool = True
    save_interval: int = 5  # Save every N epochs

    # ===== Data =====
    num_workers: int = 2  # DataLoader workers
    pin_memory: bool = True


@dataclass
class DataConfig:
    """Data loading configuration."""

    # ===== Dataset Paths =====
    data_dir: Path = field(default_factory=lambda: Path("data/processed"))
    split_file: Path = field(default_factory=lambda: Path("data/processed/splits.pkl"))

    # ===== Train/Val/Test Split Ratios =====
    # Already done in Phase 1: 70% train, 15% val, 15% test (per-dataset then combine)

    # ===== Feature Harmonization =====
    feature_mappings_file: Path = field(
        default_factory=lambda: Path("data/processed/feature_mappings.json")
    )

    # ===== Normalization =====
    # Split-first pipeline: fit imputer/scaler on train only, apply to val/test
    use_per_dataset_normalization: bool = True


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""

    # ===== Metrics =====
    metrics: list = field(
        default_factory=lambda: [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
            "cm",  # confusion matrix
        ]
    )

    # ===== Per-Dataset Evaluation =====
    # After training on combined data, evaluate on held-out per-dataset test sets
    per_dataset_eval: bool = True

    # ===== Quantization Targets (Phase 4) =====
    quantization_target_drop: float = 0.02  # Accept <2% accuracy drop for quantization

    # ===== Artifacts =====
    results_dir: Path = field(default_factory=lambda: Path("results/helix_full"))
    save_predictions: bool = True
    save_model_card: bool = True


def load_training_config(config_path: Optional[str] = None) -> TrainingConfig:
    """
    Load training config from YAML or use defaults.

    Args:
        config_path: Optional path to config YAML

    Returns:
        TrainingConfig instance
    """
    if config_path is None:
        return TrainingConfig()

    # Would implement YAML loading if needed
    # For now, return defaults
    return TrainingConfig()


def save_training_config(config: TrainingConfig, save_path: Path) -> None:
    """Save training config to JSON for reproducibility."""
    config_dict = {
        "model": {
            "input_dim": config.input_dim,
            "hidden_dims": config.hidden_dims,
            "dropout_rates": config.dropout_rates,
            "use_batch_norm": config.use_batch_norm,
            "activation": config.activation,
        },
        "loss": {
            "lambda_binary": config.lambda_binary,
            "lambda_family": config.lambda_family,
        },
        "training": {
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "epochs": config.epochs,
            "warmup_epochs": config.warmup_epochs,
            "warmup_init_lr": config.warmup_init_lr,
            "lr_decay_factor": config.lr_decay_factor,
            "lr_decay_steps": config.lr_decay_steps,
        },
        "device": config.device,
    }

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(config_dict, f, indent=2, default=str)


# ============================================================================
# Preset Configurations
# ============================================================================

# Default configuration (M4 MacBook MPS)
DEFAULT_CONFIG = TrainingConfig()

# Larger model for higher accuracy (if hardware allows)
LARGE_CONFIG = TrainingConfig(
    hidden_dims=(512, 384, 256, 128),
    batch_size=128,
    epochs=200,
)

# Smaller model for faster iteration
SMALL_CONFIG = TrainingConfig(
    hidden_dims=(128, 96, 64, 32),
    batch_size=512,
    epochs=100,
)
