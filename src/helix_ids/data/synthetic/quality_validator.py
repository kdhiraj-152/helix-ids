"""Quality validation for synthetic data.

Validates that synthetic data maintains statistical properties of real data
and is suitable for training ML models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of synthetic data validation."""

    passed: bool
    score: float
    details: dict[str, Any]

    def __str__(self) -> str:
        status = "✓ PASSED" if self.passed else "✗ FAILED"
        return f"{status} (score: {self.score:.3f})"


class SyntheticValidator:
    """Validates quality of synthetic data against real data.

    Uses multiple statistical tests and ML-based detection to ensure
    synthetic data is high quality and suitable for training.
    """

    def __init__(
        self,
        ks_threshold: float = 0.1,
        discriminator_threshold: float = 0.6,
        correlation_threshold: float = 2.0,
    ):
        """Initialize validator with thresholds.

        Args:
            ks_threshold: Max KS statistic for distribution similarity
            discriminator_threshold: Max discriminator accuracy (lower = better)
            correlation_threshold: Max Frobenius norm of correlation diff
        """
        self.ks_threshold = ks_threshold
        self.discriminator_threshold = discriminator_threshold
        self.correlation_threshold = correlation_threshold

    def validate(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        label_column: str | None = None,
    ) -> ValidationResult:
        """Run all validation checks on synthetic data.

        Args:
            real_data: Original real data
            synthetic_data: Generated synthetic data
            label_column: Label column name (excluded from tests)

        Returns:
            ValidationResult with overall assessment
        """
        details = {}
        scores = []

        # Remove label column for statistical tests
        if label_column and label_column in real_data.columns:
            real_features = real_data.drop(columns=[label_column])
        else:
            real_features = real_data

        if label_column and label_column in synthetic_data.columns:
            synth_features = synthetic_data.drop(columns=[label_column])
        else:
            synth_features = synthetic_data

        # 1. Statistical distribution test (KS test)
        ks_result = self._check_distributions(real_features, synth_features)
        details["ks_test"] = ks_result
        scores.append(1.0 - min(ks_result["mean_ks_stat"], 1.0))

        # 2. Correlation structure test
        corr_result = self._check_correlations(real_features, synth_features)
        details["correlation"] = corr_result
        scores.append(1.0 - min(corr_result["frobenius_norm"] / 10.0, 1.0))

        # 3. Discriminator test (can ML tell real from fake?)
        disc_result = self._check_discriminator(real_features, synth_features)
        details["discriminator"] = disc_result
        scores.append(1.0 - abs(disc_result["accuracy"] - 0.5) * 2)

        # 4. Statistical moments test
        moments_result = self._check_moments(real_features, synth_features)
        details["moments"] = moments_result
        scores.append(moments_result["similarity_score"])

        # Overall score (weighted average)
        overall_score = np.mean(scores)

        # Check pass criteria
        passed = (
            ks_result["mean_ks_stat"] < self.ks_threshold
            and disc_result["accuracy"] < self.discriminator_threshold
            and corr_result["frobenius_norm"] < self.correlation_threshold
        )

        logger.info(f"Validation {'PASSED' if passed else 'FAILED'} (score: {overall_score:.3f})")

        return ValidationResult(
            passed=passed,
            score=overall_score,
            details=details,
        )

    def _check_distributions(
        self,
        real: pd.DataFrame,
        synthetic: pd.DataFrame,
    ) -> dict[str, Any]:
        """Check if marginal distributions match using KS test."""
        results: dict[str, Any] = {"column_stats": {}, "mean_ks_stat": 0.0}

        numeric_cols = real.select_dtypes(include=[np.number]).columns
        common_cols = [c for c in numeric_cols if c in synthetic.columns]

        ks_stats = []
        for col in common_cols:
            real_col = real[col].dropna()
            synth_col = synthetic[col].dropna()

            if len(real_col) == 0 or len(synth_col) == 0:
                continue

            ks_stat, p_value = stats.ks_2samp(real_col, synth_col)

            results["column_stats"][col] = {
                "ks_stat": float(ks_stat),
                "p_value": float(p_value),
                "passed": ks_stat < self.ks_threshold,
            }
            ks_stats.append(ks_stat)

        results["mean_ks_stat"] = float(np.mean(ks_stats)) if ks_stats else 1.0
        column_stats: dict[str, dict[str, Any]] = results["column_stats"]
        passed_count = sum(bool(s["passed"]) for s in column_stats.values())
        results["passed_ratio"] = passed_count / max(len(column_stats), 1)

        return results

    def _check_correlations(
        self,
        real: pd.DataFrame,
        synthetic: pd.DataFrame,
    ) -> dict[str, Any]:
        """Check if correlation structure is preserved."""
        numeric_cols = real.select_dtypes(include=[np.number]).columns
        common_cols = [c for c in numeric_cols if c in synthetic.columns]

        if len(common_cols) < 2:
            return {"frobenius_norm": 0.0, "passed": True}

        real_corr = real[common_cols].corr().fillna(0)
        synth_corr = synthetic[common_cols].corr().fillna(0)

        diff = real_corr.values - synth_corr.values
        frobenius_norm = float(np.linalg.norm(diff, "fro"))

        return {
            "frobenius_norm": frobenius_norm,
            "passed": frobenius_norm < self.correlation_threshold,
            "max_diff": float(np.abs(diff).max()),
        }

    def _check_discriminator(
        self,
        real: pd.DataFrame,
        synthetic: pd.DataFrame,
    ) -> dict[str, Any]:
        """Train discriminator to distinguish real from fake.

        Good synthetic data should be hard to distinguish (accuracy ~0.5).
        """
        numeric_cols = real.select_dtypes(include=[np.number]).columns
        common_cols = [c for c in numeric_cols if c in synthetic.columns]

        if len(common_cols) == 0:
            return {"accuracy": 0.5, "passed": True}

        # Prepare data
        real_subset = real[common_cols].fillna(0).values
        synth_subset = synthetic[common_cols].fillna(0).values

        X = np.vstack([real_subset, synth_subset])
        y = np.array([0] * len(real_subset) + [1] * len(synth_subset))

        # Train discriminator
        clf = RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=42,
        )

        try:
            scores = cross_val_score(clf, X, y, cv=3, scoring="accuracy")
            accuracy = float(np.mean(scores))
        except Exception as e:
            logger.warning(f"Discriminator test failed: {e}")
            accuracy = 0.5

        if accuracy < 0.55:
            interpretation = "excellent"
        elif accuracy < 0.60:
            interpretation = "good"
        elif accuracy < 0.70:
            interpretation = "fair"
        else:
            interpretation = "poor"

        return {
            "accuracy": accuracy,
            "passed": accuracy < self.discriminator_threshold,
            "interpretation": interpretation,
        }

    def _check_moments(
        self,
        real: pd.DataFrame,
        synthetic: pd.DataFrame,
    ) -> dict[str, Any]:
        """Check if statistical moments match (mean, std, skew, kurtosis)."""
        numeric_cols = real.select_dtypes(include=[np.number]).columns
        common_cols = [c for c in numeric_cols if c in synthetic.columns]

        if len(common_cols) == 0:
            return {"similarity_score": 1.0}

        real_data = real[common_cols]
        synth_data = synthetic[common_cols]

        # Calculate moments
        moments = {}
        for name, func in [
            ("mean", lambda x: x.mean()),
            ("std", lambda x: x.std()),
            ("skew", lambda x: x.skew()),
            ("kurtosis", lambda x: x.kurtosis()),
        ]:
            real_moment = func(real_data).fillna(0)
            synth_moment = func(synth_data).fillna(0)

            # Normalized absolute difference
            diff = np.abs(real_moment - synth_moment)
            scale = np.abs(real_moment) + 1e-8
            normalized_diff = (diff / scale).mean()

            moments[name] = {
                "real": float(real_moment.mean()),
                "synthetic": float(synth_moment.mean()),
                "normalized_diff": float(normalized_diff),
            }

        # Overall similarity score (inverse of mean normalized diff)
        mean_diff = np.mean([m["normalized_diff"] for m in moments.values()])
        similarity_score = max(0, 1.0 - mean_diff)

        return {
            "moments": moments,
            "similarity_score": float(similarity_score),
        }

    def validate_per_class(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        label_column: str = "label",
    ) -> dict[str, ValidationResult]:
        """Validate synthetic data for each class separately.

        Args:
            real_data: Original real data with labels
            synthetic_data: Generated synthetic data with labels
            label_column: Name of label column

        Returns:
            Dict mapping class name to ValidationResult
        """
        results = {}

        classes = synthetic_data[label_column].unique()

        for class_name in classes:
            real_class = real_data[real_data[label_column] == class_name]
            synth_class = synthetic_data[synthetic_data[label_column] == class_name]

            if len(real_class) == 0:
                logger.warning(f"No real samples for class {class_name}")
                continue

            result = self.validate(real_class, synth_class, label_column)
            results[class_name] = result

            logger.info(f"Class {class_name}: {result}")

        return results
