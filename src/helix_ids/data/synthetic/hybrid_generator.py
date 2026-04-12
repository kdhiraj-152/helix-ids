"""Hybrid synthetic data generator combining CTGAN and TVAE.

Uses ensemble of generators with quality-based selection to produce
the best synthetic samples.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .ctgan_generator import CTGANConfig, CTGANGenerator
from .quality_validator import SyntheticValidator
from .tvae_generator import TVAEConfig, TVAEGenerator

logger = logging.getLogger(__name__)


@dataclass
class HybridConfig:
    """Configuration for hybrid generator."""

    use_ctgan: bool = True
    use_tvae: bool = True
    ctgan_config: CTGANConfig | None = None
    tvae_config: TVAEConfig | None = None
    selection_strategy: str = "best"  # "best", "ensemble", "weighted"
    ensemble_ratio: float = 0.5  # For "ensemble" strategy

    minority_classes: list[str] = field(default_factory=lambda: ["R2L", "U2R"])
    samples_per_class: dict[str, int] = field(
        default_factory=lambda: {
            "R2L": 10000,
            "U2R": 5000,
        }
    )


class HybridGenerator:
    """Hybrid generator combining multiple synthetic data methods.

    Trains both CTGAN and TVAE, validates quality, and selects or
    combines the best outputs.
    """

    def __init__(self, config: HybridConfig | None = None):
        """Initialize hybrid generator.

        Args:
            config: Hybrid configuration. Uses defaults if None.
        """
        self.config = config or HybridConfig()
        self.generators: dict[str, Any] = {}
        self.quality_scores: dict[str, dict[str, float]] = {}
        self.validator = SyntheticValidator()
        self._trained = False
        self._train_data: pd.DataFrame | None = None

    def fit(
        self,
        data: pd.DataFrame,
        label_column: str = "label",
        discrete_columns: list[str] | None = None,
    ) -> HybridGenerator:
        """Train all configured generators.

        Args:
            data: Training DataFrame with features and labels
            label_column: Name of label column
            discrete_columns: List of categorical column names

        Returns:
            self for method chaining
        """
        self._train_data = data

        # Initialize and train CTGAN
        if self.config.use_ctgan:
            logger.info("Training CTGAN generator...")
            ctgan_config = self.config.ctgan_config or CTGANConfig(
                minority_classes=self.config.minority_classes,
                samples_per_class=self.config.samples_per_class,
            )
            ctgan = CTGANGenerator(ctgan_config)
            ctgan.fit(data, label_column, discrete_columns)
            self.generators["ctgan"] = ctgan

        # Initialize and train TVAE
        if self.config.use_tvae:
            logger.info("Training TVAE generator...")
            tvae_config = self.config.tvae_config or TVAEConfig(
                minority_classes=self.config.minority_classes,
                samples_per_class=self.config.samples_per_class,
            )
            tvae = TVAEGenerator(tvae_config)
            tvae.fit(data, label_column, discrete_columns)
            self.generators["tvae"] = tvae

        self._trained = True
        return self

    def generate(
        self,
        n_samples: dict[str, int] | None = None,
        validate: bool = True,
        label_column: str = "label",
    ) -> pd.DataFrame:
        """Generate synthetic samples using configured strategy.

        Args:
            n_samples: Dict mapping class name to sample count
            validate: Whether to validate and select best output
            label_column: Name of label column

        Returns:
            DataFrame with synthetic samples
        """
        if not self._trained:
            raise RuntimeError("Generator must be fit before generating")

        if n_samples is None:
            n_samples = self.config.samples_per_class

        # Generate from all models
        outputs: dict[str, pd.DataFrame] = {}

        for name, generator in self.generators.items():
            logger.info(f"Generating with {name}...")
            synthetic = generator.generate(n_samples, add_labels=True)
            outputs[name] = synthetic

        if not outputs:
            return pd.DataFrame()

        # Validate and score each output
        if validate and self._train_data is not None:
            for name, synthetic in outputs.items():
                result = self.validator.validate(self._train_data, synthetic, label_column)
                self.quality_scores[name] = {
                    "overall": result.score,
                    "passed": result.passed,
                }
                logger.info(f"{name} quality: {result}")

        # Select or combine based on strategy
        if self.config.selection_strategy == "best":
            return self._select_best(outputs)
        elif self.config.selection_strategy == "ensemble":
            return self._ensemble(outputs)
        elif self.config.selection_strategy == "weighted":
            return self._weighted_ensemble(outputs)
        else:
            # Default: return first output
            return next(iter(outputs.values()))

    def _select_best(self, outputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Select output from best-performing generator."""
        if not self.quality_scores:
            return next(iter(outputs.values()))

        best_name = max(self.quality_scores.keys(), key=lambda k: self.quality_scores[k]["overall"])

        logger.info(f"Selected best generator: {best_name}")
        return outputs[best_name]

    def _ensemble(self, outputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine outputs from all generators."""
        dfs = list(outputs.values())

        if len(dfs) == 1:
            return dfs[0]

        # Take ratio from each generator
        n_total = len(dfs[0])
        n_each = int(n_total * self.config.ensemble_ratio)

        combined = []
        for df in dfs:
            sampled = df.sample(n=min(n_each, len(df)), random_state=42)
            combined.append(sampled)

        return pd.concat(combined, ignore_index=True)

    def _weighted_ensemble(
        self,
        outputs: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Combine outputs weighted by quality score."""
        if not self.quality_scores:
            return self._ensemble(outputs)

        total_score = sum(s["overall"] for s in self.quality_scores.values())

        if total_score == 0:
            return self._ensemble(outputs)

        combined = []
        n_total = len(next(iter(outputs.values())))

        for name, df in outputs.items():
            weight = self.quality_scores.get(name, {}).get("overall", 0.5)
            n_samples = int(n_total * weight / total_score)

            sampled = df.sample(n=min(n_samples, len(df)), random_state=42)
            combined.append(sampled)

        return pd.concat(combined, ignore_index=True)

    def fit_generate(
        self,
        data: pd.DataFrame,
        label_column: str = "label",
        discrete_columns: list[str] | None = None,
        n_samples: dict[str, int] | None = None,
    ) -> pd.DataFrame:
        """Convenience method to fit and generate in one call."""
        self.fit(data, label_column, discrete_columns)
        return self.generate(n_samples, label_column=label_column)

    def get_quality_report(self) -> dict[str, Any]:
        """Get quality report for all generators."""
        return {
            "generators": list(self.generators.keys()),
            "quality_scores": self.quality_scores,
            "best_generator": (
                max(self.quality_scores.keys(), key=lambda k: self.quality_scores[k]["overall"])
                if self.quality_scores
                else None
            ),
        }

    def save(self, path: str | Path) -> None:
        """Save all trained generators to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        for name, generator in self.generators.items():
            gen_path = path / name
            generator.save(gen_path)

        import json

        report_path = path / "hybrid_report.json"
        with open(report_path, "w") as f:
            json.dump(self.get_quality_report(), f, indent=2)

    def load(self, path: str | Path) -> HybridGenerator:
        """Load trained generators from disk."""
        path = Path(path)

        if self.config.use_ctgan:
            ctgan_path = path / "ctgan"
            if ctgan_path.exists():
                ctgan = CTGANGenerator()
                ctgan.load(ctgan_path)
                self.generators["ctgan"] = ctgan

        if self.config.use_tvae:
            tvae_path = path / "tvae"
            if tvae_path.exists():
                tvae = TVAEGenerator()
                tvae.load(tvae_path)
                self.generators["tvae"] = tvae

        self._trained = len(self.generators) > 0
        return self


def augment_with_hybrid(
    train_data: pd.DataFrame,
    label_column: str = "label",
    target_samples: dict[str, int] | None = None,
    strategy: str = "best",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Augment minority classes using hybrid generator.

    Args:
        train_data: Training DataFrame with features and labels
        label_column: Name of label column
        target_samples: Target sample counts per class
        strategy: Selection strategy ("best", "ensemble", "weighted")

    Returns:
        Tuple of (augmented_data, quality_report)
    """
    if target_samples is None:
        target_samples = {"R2L": 10000, "U2R": 5000}

    config = HybridConfig(
        selection_strategy=strategy,
        minority_classes=list(target_samples.keys()),
        samples_per_class=target_samples,
    )

    generator = HybridGenerator(config)
    synthetic = generator.fit_generate(train_data, label_column)

    augmented = pd.concat([train_data, synthetic], ignore_index=True)

    logger.info(f"Augmented data: {len(train_data)} -> {len(augmented)} samples")

    return augmented, generator.get_quality_report()
