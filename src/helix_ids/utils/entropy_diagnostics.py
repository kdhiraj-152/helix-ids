"""
Entropy Diagnostics Utility

Provides shared entropy calculation and analysis for integrity guards across
training validation, per-dataset evaluation, and deployment scenarios.

Uses a single entropy definition: normalized mean entropy over samples.
"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class EntropySummary:
    """Entropy statistics for a batch of probabilities."""

    mean: float
    min_val: float
    max_val: float
    num_samples: int
    num_classes: int
    collapsed_samples: int  # Count with entropy < 0.1

    def __str__(self) -> str:
        """Human-readable summary."""
        return (
            f"Entropy(n={self.num_samples}, classes={self.num_classes}): "
            f"mean={self.mean:.4f}, "
            f"range=[{self.min_val:.4f}, {self.max_val:.4f}], "
            f"collapsed_samples={self.collapsed_samples}"
        )


def calculate_entropy_stable(
    probs: np.ndarray,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Calculate normalized entropy per sample with numerical stability.
    
    Args:
        probs: (batch_size, num_classes) probability matrix
        eps: Epsilon for numerical stability
    
    Returns:
        (batch_size,) normalized entropy per sample [0, 1]
    """
    if probs.size == 0:
        return np.array([])
    
    # Clip for numerical stability
    safe_probs = np.clip(probs, eps, 1.0 - eps)

    # Compute per-sample entropy: -sum(p * log(p))
    log_probs = np.log(safe_probs)
    per_sample_entropy = -np.sum(probs * log_probs, axis=1)

    # Normalize by maximum possible entropy: log(num_classes)
    num_classes = probs.shape[1]
    max_entropy = np.log(num_classes)
    normalized_entropy = per_sample_entropy / max_entropy

    return np.asarray(normalized_entropy, dtype=np.float64)


def summarize_entropy(
    probs: np.ndarray,
    eps: float = 1e-10,
) -> EntropySummary:
    """
    Compute comprehensive entropy statistics for a batch.
    
    Args:
        probs: (batch_size, num_classes) probability matrix
        eps: Epsilon for numerical stability
    
    Returns:
        EntropySummary with distributions and policy signals
    """
    if probs.size == 0:
        return EntropySummary(
            mean=0.0,
            min_val=0.0,
            max_val=0.0,
            num_samples=0,
            num_classes=0,
            collapsed_samples=0,
        )
    
    # Calculate normalized entropy
    entropy = calculate_entropy_stable(probs, eps=eps)

    # Compute distribution stats
    mean_val = float(np.mean(entropy))
    min_val = float(np.min(entropy))
    max_val = float(np.max(entropy))

    # Count collapsed samples (entropy < 0.1 is very peaked)
    collapsed_count = int(np.sum(entropy < 0.1))

    return EntropySummary(
        mean=mean_val,
        min_val=min_val,
        max_val=max_val,
        num_samples=probs.shape[0],
        num_classes=probs.shape[1],
        collapsed_samples=collapsed_count,
    )


def should_trigger_entropy_guard(
    entropy_summary: EntropySummary,
    has_missing_classes: bool = False,
    streak_count: int = 1,
    temperature: float = 1.0,
) -> tuple[bool, Optional[str]]:
    """
    Policy decision: should entropy guard trigger hard-stop?
    
    Args:
        entropy_summary: Computed entropy statistics
        has_missing_classes: Whether actual classes are absent from predictions
        streak_count: How many consecutive epochs with low entropy
        temperature: Temperature used in softmax (affects baseline thresholds)
    
    Returns:
        (should_trigger, reason_if_true)
    """
    # Policy 1: Mean entropy collapse + missing classes (confirmed mode collapse)
    if entropy_summary.mean < 0.12 and has_missing_classes:
        return True, "entropy_collapse_with_missing_classes"

    # Policy 2: Extreme entropy sustained for 3+ epochs (critical collapse)
    if entropy_summary.mean < 0.08 and streak_count >= 3:
        return True, "critical_entropy_collapse"

    # Policy 3: All samples near-degenerate (entire batch collapsed)
    if (
        entropy_summary.mean < 0.06
        and entropy_summary.collapsed_samples > entropy_summary.num_samples * 0.5
    ):
        return True, "batch_wide_entropy_collapse"

    # Policy 4: Max entropy still very low (numeric instability)
    if entropy_summary.max_val < 0.05:
        return True, "numerically_unstable_entropy"

    return False, None


def detect_batch_composition_risk(
    entropy_summary: EntropySummary,
    predicted_classes: np.ndarray,
    num_expected_classes: int,
) -> dict[str, Any]:
    """
    Identify mode-collapse and class-coverage risks.
    
    Args:
        entropy_summary: Computed entropy statistics
        predicted_classes: (batch_size,) predicted class indices
        num_expected_classes: Expected number of classes (e.g., 7 for family)
    
    Returns:
        Risk diagnostics dict
    """
    unique_predicted = len(np.unique(predicted_classes))
    missing_classes = num_expected_classes - unique_predicted
    
    return {
        "unique_classes_predicted": unique_predicted,
        "missing_classes": missing_classes,
        "missing_class_ratio": missing_classes / num_expected_classes,
        "collapsed_sample_count": entropy_summary.collapsed_samples,
        "collapsed_sample_ratio": entropy_summary.collapsed_samples / max(1, entropy_summary.num_samples),
        "entropy_range": entropy_summary.max_val - entropy_summary.min_val,
        "entropy_is_peaked": entropy_summary.mean < 0.25,
    }
