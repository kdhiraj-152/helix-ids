"""Governance summary and reporting functions.

Extracted from train_helix_ids_full.py:
  - _summarize_governance

Provides typed input/output wrappers and the pure summarisation logic
that turns per-seed calibration metrics into a structured governance
verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GovernanceSummaryInput:
    """Input to the governance summariser: per-seed metric records.

    Each element mirrors the dict shape accumulated by
    _run_multiseed_calibrated_governance.
    """

    strict_seed_runs: list[dict[str, Any]]


@dataclass
class GovernanceSummary:
    """Structured governance summary with failure reasons and actions.

    Mirrors the original returned (governance_dict, failures, actions)
    tuple in a single typed object.
    """

    mean_macro_f1: float
    std_macro_f1: float
    mean_class4_precision: float
    mean_class4_recall: float
    min_class4_recall: float
    mean_entropy: float
    max_zero_prediction_classes: int
    status: str = "PASS"
    failure_reasons: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)

    def to_governance_dict(self) -> dict[str, Any]:
        """Return the governance portion of the final payload dict."""
        return {
            "mean_macro_f1": self.mean_macro_f1,
            "std_macro_f1": self.std_macro_f1,
            "mean_class4_precision": self.mean_class4_precision,
            "mean_class4_recall": self.mean_class4_recall,
            "min_class4_recall": self.min_class4_recall,
            "mean_entropy": self.mean_entropy,
            "max_zero_prediction_classes": self.max_zero_prediction_classes,
            "status": self.status,
            "failure_reasons": list(self.failure_reasons),
            "actions": list(self.actions),
        }

    def to_failure_tuple(self) -> tuple[dict[str, Any], list[str], list[str]]:
        """Return the original (governance_dict, failure_reasons, actions) tuple.

        Maintains backward compatibility with code that expects the
        original _summarize_governance return shape.
        """
        return self.to_governance_dict(), list(self.failure_reasons), list(self.actions)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def summarize_governance(
    strict_seed_runs: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Aggregate multi-seed runs and emit strict governance payload.

    This is the exact extracted body of the original
    _summarize_governance function.

    Args:
        strict_seed_runs: List of seed-run dicts, each containing at least
            macro_f1, class4_precision, class4_recall, entropy,
            zero_prediction_classes.

    Returns:
        (governance_dict, failure_reasons, actions) matching the
        original function signature.
    """
    macro_vals = [float(r["macro_f1"]) for r in strict_seed_runs]
    p4_prec_vals = [float(r["class4_precision"]) for r in strict_seed_runs]
    p4_rec_vals = [float(r["class4_recall"]) for r in strict_seed_runs]
    zero_vals = [int(r["zero_prediction_classes"]) for r in strict_seed_runs]
    entropy_vals = [float(r["entropy"]) for r in strict_seed_runs]

    governance: dict[str, Any] = {
        "mean_macro_f1": float(np.mean(macro_vals)),
        "std_macro_f1": float(np.std(macro_vals)),
        "mean_class4_precision": float(np.mean(p4_prec_vals)),
        "mean_class4_recall": float(np.mean(p4_rec_vals)),
        "min_class4_recall": float(np.min(p4_rec_vals)),
        "mean_entropy": float(np.mean(entropy_vals)),
        "max_zero_prediction_classes": int(max(zero_vals)),
    }

    failure_reasons: list[str] = []
    if governance["std_macro_f1"] > 0.03:
        failure_reasons.append("std_macro_f1_gt_0_03")
    if governance["min_class4_recall"] < 0.80:
        failure_reasons.append("min_class4_recall_lt_0_80")
    if governance["mean_class4_precision"] < 0.25:
        failure_reasons.append("mean_class4_precision_lt_0_25")
    if governance["max_zero_prediction_classes"] != 0:
        failure_reasons.append("max_zero_prediction_classes_ne_0")
    if governance["mean_entropy"] <= 0.2:
        failure_reasons.append("mean_entropy_le_0_2")

    actions: list[str] = []
    if governance["mean_class4_precision"] < 0.25:
        actions.append("increase_tau_4")
    if governance["min_class4_recall"] < 0.80:
        actions.append("increase_focal_gamma_up_to_1_5")
    if governance["mean_entropy"] <= 0.2:
        actions.append("increase_temperature_max_to_5_0")

    return governance, failure_reasons, actions
