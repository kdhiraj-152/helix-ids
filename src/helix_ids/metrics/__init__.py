"""
HELIX-IDS Metrics Module

Provides evaluation and tracking utilities for model performance:
- Per-class metrics computation and reporting
- False negative rate tracking per attack class
- Adversarial robustness testing
"""

from .adversarial_test import (
    AdversarialMetrics,
    AdversarialTester,
    run_adversarial_evaluation,
)
from .fn_tracker import FalseNegativeTracker
from .per_class_metrics import (
    DEFAULT_THRESHOLDS,
    ClassMetrics,
    PerClassMetrics,
    PerClassMetricsResult,
)

__all__ = [
    "ClassMetrics",
    "PerClassMetrics",
    "PerClassMetricsResult",
    "DEFAULT_THRESHOLDS",
    "FalseNegativeTracker",
    "AdversarialMetrics",
    "AdversarialTester",
    "run_adversarial_evaluation",
]
