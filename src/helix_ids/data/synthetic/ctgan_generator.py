"""CTGAN-based synthetic data generator for minority class augmentation.

CTGAN (Conditional Tabular GAN) generates realistic synthetic tabular data
by modeling continuous columns with GMMs and categorical columns with one-hot.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


@dataclass
class CTGANConfig:
    """Configuration for CTGAN generator."""

    embedding_dim: int = 128
    generator_dim: tuple[int, ...] = (256, 256)
    discriminator_dim: tuple[int, ...] = (256, 256)
    generator_lr: float = 2e-4
    generator_decay: float = 1e-6
    discriminator_lr: float = 2e-4
    discriminator_decay: float = 1e-6
    batch_size: int = 500
    discriminator_steps: int = 1
    log_frequency: bool = True
    verbose: bool = False
    epochs: int = 300
    pac: int = 10
    cuda: bool = True

    # Class-specific settings
    minority_classes: list[str] = field(default_factory=lambda: ["R2L", "U2R"])
    samples_per_class: dict[str, int] = field(
        default_factory=lambda: {
            "R2L": 10000,
            "U2R": 5000,
        }
    )


class CTGANGenerator:
    """CTGAN-based generator for minority class synthetic data.

    Uses Conditional Tabular GAN to generate high-quality synthetic samples
    for underrepresented attack classes (R2L, U2R).
    """

    def __init__(self, config: CTGANConfig | None = None):
        """Initialize generator with configuration.

        Args:
            config: CTGAN configuration. Uses defaults if None.
        """
        self.config = config or CTGANConfig()
        self._models: dict[str, Any] = {}
        self._metadata: dict[str, Any] = {}
        self._trained = False

        # Check CUDA availability
        if self.config.cuda and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.config.cuda = False

    def fit(
        self,
        data: pd.DataFrame,
        label_column: str = "label",
        discrete_columns: list[str] | None = None,
    ) -> CTGANGenerator:
        """Train CTGAN models for each minority class.

        Args:
            data: Training DataFrame with features and labels
            label_column: Name of label column
            discrete_columns: List of categorical column names

        Returns:
            self for method chaining
        """
        try:
            from ctgan import CTGAN
        except ImportError as err:
            raise ImportError("CTGAN not installed. Run: pip install ctgan") from err

        if discrete_columns is None:
            discrete_columns = self._detect_discrete_columns(data, label_column)

        for class_name in self.config.minority_classes:
            logger.info(f"Training CTGAN for class: {class_name}")

            # Filter data for this class
            class_data = data[data[label_column] == class_name].drop(columns=[label_column])

            if len(class_data) < 10:
                logger.warning(
                    f"Class {class_name} has only {len(class_data)} samples. "
                    "Consider using more training data."
                )

            # Train CTGAN
            model = CTGAN(
                embedding_dim=self.config.embedding_dim,
                generator_dim=self.config.generator_dim,
                discriminator_dim=self.config.discriminator_dim,
                generator_lr=self.config.generator_lr,
                generator_decay=self.config.generator_decay,
                discriminator_lr=self.config.discriminator_lr,
                discriminator_decay=self.config.discriminator_decay,
                batch_size=min(self.config.batch_size, len(class_data)),
                discriminator_steps=self.config.discriminator_steps,
                log_frequency=self.config.log_frequency,
                verbose=self.config.verbose,
                epochs=self.config.epochs,
                pac=self.config.pac,
                cuda=self.config.cuda,
            )

            # Remove label from discrete columns for training
            train_discrete = [c for c in discrete_columns if c != label_column]

            model.fit(class_data, discrete_columns=train_discrete)

            self._models[class_name] = model
            self._metadata[class_name] = {
                "original_count": len(class_data),
                "columns": list(class_data.columns),
                "discrete_columns": train_discrete,
            }

            logger.info(f"Trained CTGAN for {class_name} on {len(class_data)} samples")

        self._trained = True
        return self

    def generate(
        self,
        n_samples: dict[str, int] | None = None,
        add_labels: bool = True,
    ) -> pd.DataFrame:
        """Generate synthetic samples for each minority class.

        Args:
            n_samples: Dict mapping class name to sample count.
                      Uses config.samples_per_class if None.
            add_labels: Whether to add class labels to generated data.

        Returns:
            DataFrame with synthetic samples
        """
        if not self._trained:
            raise RuntimeError("Generator must be fit before generating")

        if n_samples is None:
            n_samples = self.config.samples_per_class

        synthetic_dfs = []

        for class_name, count in n_samples.items():
            if class_name not in self._models:
                logger.warning(f"No model for class {class_name}, skipping")
                continue

            model = self._models[class_name]
            logger.info(f"Generating {count} samples for class {class_name}")

            synthetic = model.sample(count)

            if add_labels:
                synthetic["label"] = class_name

            synthetic_dfs.append(synthetic)
            logger.info(f"Generated {len(synthetic)} samples for {class_name}")

        if not synthetic_dfs:
            return pd.DataFrame()

        return pd.concat(synthetic_dfs, ignore_index=True)

    def fit_generate(
        self,
        data: pd.DataFrame,
        label_column: str = "label",
        discrete_columns: list[str] | None = None,
        n_samples: dict[str, int] | None = None,
    ) -> pd.DataFrame:
        """Convenience method to fit and generate in one call.

        Args:
            data: Training DataFrame
            label_column: Name of label column
            discrete_columns: List of categorical column names
            n_samples: Dict mapping class name to sample count

        Returns:
            DataFrame with synthetic samples
        """
        self.fit(data, label_column, discrete_columns)
        return self.generate(n_samples)

    def save(self, path: str | Path) -> None:
        """Save trained models to disk.

        Args:
            path: Directory path to save models
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        for class_name, model in self._models.items():
            model_path = path / f"ctgan_{class_name}.pkl"
            model.save(str(model_path))
            logger.info(f"Saved model for {class_name} to {model_path}")

        # Save metadata
        import json

        meta_path = path / "ctgan_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(self._metadata, f, indent=2)

    def load(self, path: str | Path) -> CTGANGenerator:
        """Load trained models from disk.

        Args:
            path: Directory path containing saved models

        Returns:
            self for method chaining
        """
        try:
            from ctgan import CTGAN
        except ImportError as err:
            raise ImportError("CTGAN not installed. Run: pip install ctgan") from err

        path = Path(path)

        # Load metadata
        import json

        meta_path = path / "ctgan_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self._metadata = json.load(f)

        # Load models
        for class_name in self.config.minority_classes:
            model_path = path / f"ctgan_{class_name}.pkl"
            if model_path.exists():
                self._models[class_name] = CTGAN.load(str(model_path))
                logger.info(f"Loaded model for {class_name}")

        self._trained = len(self._models) > 0
        return self

    def _detect_discrete_columns(
        self,
        data: pd.DataFrame,
        label_column: str,
    ) -> list[str]:
        """Detect discrete/categorical columns automatically.

        Args:
            data: DataFrame to analyze
            label_column: Label column name (always discrete)

        Returns:
            List of discrete column names
        """
        discrete = [label_column]

        for col in data.columns:
            if col == label_column:
                continue

            # Check if column looks categorical
            if data[col].dtype in ("object", "category") or (
                data[col].nunique() < 20 and data[col].dtype in ["int64", "int32"]
            ):
                discrete.append(col)

        return discrete

    def get_quality_metrics(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
    ) -> dict[str, float]:
        """Compute quality metrics comparing real and synthetic data.

        Args:
            real_data: Original real data
            synthetic_data: Generated synthetic data

        Returns:
            Dict with quality metrics
        """
        metrics: dict[str, float] = {}

        # Numeric columns only
        real_numeric = real_data.select_dtypes(include=[np.number])
        synth_numeric = synthetic_data.select_dtypes(include=[np.number])

        common_cols = list(set(real_numeric.columns) & set(synth_numeric.columns))

        if not common_cols:
            return metrics

        real_numeric = real_numeric[common_cols]
        synth_numeric = synth_numeric[common_cols]

        # Mean absolute difference in column means
        mean_diff = np.abs(real_numeric.mean() - synth_numeric.mean()).mean()
        metrics["mean_diff"] = float(mean_diff)

        # Mean absolute difference in column stds
        std_diff = np.abs(real_numeric.std() - synth_numeric.std()).mean()
        metrics["std_diff"] = float(std_diff)

        # Correlation matrix difference (Frobenius norm)
        real_corr = real_numeric.corr().fillna(0)
        synth_corr = synth_numeric.corr().fillna(0)
        corr_diff = np.linalg.norm(real_corr.values - synth_corr.values, "fro")
        metrics["correlation_diff"] = float(corr_diff)

        return metrics


def augment_minority_classes(
    train_data: pd.DataFrame,
    label_column: str = "label",
    target_samples: dict[str, int] | None = None,
    epochs: int = 300,
) -> pd.DataFrame:
    """Convenience function to augment minority classes with CTGAN.

    Args:
        train_data: Training DataFrame with features and labels
        label_column: Name of label column
        target_samples: Target sample counts per class
        epochs: Training epochs for CTGAN

    Returns:
        Augmented DataFrame with original + synthetic samples
    """
    if target_samples is None:
        target_samples = {"R2L": 10000, "U2R": 5000}

    config = CTGANConfig(
        epochs=epochs,
        minority_classes=list(target_samples.keys()),
        samples_per_class=target_samples,
    )

    generator = CTGANGenerator(config)
    synthetic = generator.fit_generate(train_data, label_column)

    # Combine with original data
    augmented = pd.concat([train_data, synthetic], ignore_index=True)

    logger.info(f"Augmented data: {len(train_data)} -> {len(augmented)} samples")

    return augmented
