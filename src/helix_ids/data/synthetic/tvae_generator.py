"""TVAE-based synthetic data generator for minority class augmentation.

TVAE (Tabular Variational Autoencoder) provides an alternative to CTGAN
with potentially faster training and better mode coverage.
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
class TVAEConfig:
    """Configuration for TVAE generator."""

    embedding_dim: int = 128
    compress_dims: tuple[int, ...] = (128, 128)
    decompress_dims: tuple[int, ...] = (128, 128)
    l2scale: float = 1e-5
    batch_size: int = 500
    epochs: int = 300
    loss_factor: float = 2.0
    cuda: bool = True

    # Class-specific settings
    minority_classes: list[str] = field(default_factory=lambda: ["R2L", "U2R"])
    samples_per_class: dict[str, int] = field(
        default_factory=lambda: {
            "R2L": 10000,
            "U2R": 5000,
        }
    )


class TVAEGenerator:
    """TVAE-based generator for minority class synthetic data.

    Uses Tabular Variational Autoencoder to generate synthetic samples.
    Often trains faster than CTGAN and may provide better coverage.
    """

    def __init__(self, config: TVAEConfig | None = None):
        """Initialize generator with configuration.

        Args:
            config: TVAE configuration. Uses defaults if None.
        """
        self.config = config or TVAEConfig()
        self._models: dict[str, Any] = {}
        self._metadata: dict[str, Any] = {}
        self._trained = False

        if self.config.cuda and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.config.cuda = False

    def fit(
        self,
        data: pd.DataFrame,
        label_column: str = "label",
        discrete_columns: list[str] | None = None,
    ) -> TVAEGenerator:
        """Train TVAE models for each minority class.

        Args:
            data: Training DataFrame with features and labels
            label_column: Name of label column
            discrete_columns: List of categorical column names

        Returns:
            self for method chaining
        """
        try:
            from ctgan import TVAE
        except ImportError as err:
            raise ImportError("TVAE not installed. Run: pip install ctgan") from err

        if discrete_columns is None:
            discrete_columns = self._detect_discrete_columns(data, label_column)

        for class_name in self.config.minority_classes:
            logger.info(f"Training TVAE for class: {class_name}")

            class_data = data[data[label_column] == class_name].drop(columns=[label_column])

            if len(class_data) < 10:
                logger.warning(f"Class {class_name} has only {len(class_data)} samples")

            model = TVAE(
                embedding_dim=self.config.embedding_dim,
                compress_dims=self.config.compress_dims,
                decompress_dims=self.config.decompress_dims,
                l2scale=self.config.l2scale,
                batch_size=min(self.config.batch_size, len(class_data)),
                epochs=self.config.epochs,
                loss_factor=self.config.loss_factor,
                cuda=self.config.cuda,
            )

            train_discrete = [c for c in discrete_columns if c != label_column]
            model.fit(class_data, discrete_columns=train_discrete)

            self._models[class_name] = model
            self._metadata[class_name] = {
                "original_count": len(class_data),
                "columns": list(class_data.columns),
                "discrete_columns": train_discrete,
            }

            logger.info(f"Trained TVAE for {class_name} on {len(class_data)} samples")

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
            add_labels: Whether to add class labels.

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
                logger.warning(f"No model for class {class_name}")
                continue

            model = self._models[class_name]
            logger.info(f"Generating {count} samples for class {class_name}")

            synthetic = model.sample(count)

            if add_labels:
                synthetic["label"] = class_name

            synthetic_dfs.append(synthetic)

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
        """Convenience method to fit and generate in one call."""
        self.fit(data, label_column, discrete_columns)
        return self.generate(n_samples)

    def save(self, path: str | Path) -> None:
        """Save trained models to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        for class_name, model in self._models.items():
            model_path = path / f"tvae_{class_name}.pkl"
            model.save(str(model_path))

        import json

        meta_path = path / "tvae_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(self._metadata, f, indent=2)

    def load(self, path: str | Path) -> TVAEGenerator:
        """Load trained models from disk."""
        try:
            from ctgan import TVAE
        except ImportError as err:
            raise ImportError("TVAE not installed. Run: pip install ctgan") from err

        path = Path(path)

        import json

        meta_path = path / "tvae_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self._metadata = json.load(f)

        for class_name in self.config.minority_classes:
            model_path = path / f"tvae_{class_name}.pkl"
            if model_path.exists():
                self._models[class_name] = TVAE.load(str(model_path))

        self._trained = len(self._models) > 0
        return self

    def _detect_discrete_columns(
        self,
        data: pd.DataFrame,
        label_column: str,
    ) -> list[str]:
        """Detect discrete/categorical columns automatically."""
        discrete = [label_column]

        for col in data.columns:
            if col == label_column:
                continue
            if data[col].dtype in ("object", "category") or (
                data[col].nunique() < 20 and data[col].dtype in ("int64", "int32")
            ):
                discrete.append(col)

        return discrete

    def get_quality_metrics(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
    ) -> dict[str, float]:
        """Compute quality metrics comparing real and synthetic data."""
        metrics: dict[str, float] = {}

        real_numeric = real_data.select_dtypes(include=[np.number])
        synth_numeric = synthetic_data.select_dtypes(include=[np.number])

        common_cols = list(set(real_numeric.columns) & set(synth_numeric.columns))

        if not common_cols:
            return metrics

        real_numeric = real_numeric[common_cols]
        synth_numeric = synth_numeric[common_cols]

        mean_diff = np.abs(real_numeric.mean() - synth_numeric.mean()).mean()
        metrics["mean_diff"] = float(mean_diff)

        std_diff = np.abs(real_numeric.std() - synth_numeric.std()).mean()
        metrics["std_diff"] = float(std_diff)

        real_corr = real_numeric.corr().fillna(0)
        synth_corr = synth_numeric.corr().fillna(0)
        corr_diff = np.linalg.norm(real_corr.values - synth_corr.values, "fro")
        metrics["correlation_diff"] = float(corr_diff)

        return metrics
