"""Promotion input/output dataclasses and result builder.

Extracted from train_helix_ids_full.py _summarize_governance helper:

Provides typed interfaces for the governance promotion consensus
pipeline — the summarisation that turns per-seed calibration metrics
into a structured PASS/FAIL governance verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PromotionInput:
    """Per-seed metrics consumed by the promotion summariser.

    Only fields that _summarize_governance actually reads from
    each seed-run dict are included here.
    """

    macro_f1: float
    class4_precision: float
    class4_recall: float
    entropy: float
    zero_prediction_classes: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PromotionInput:
        """Construct from a seed-run dict (as accumulated by the orchestrator)."""
        return cls(
            macro_f1=float(data["macro_f1"]),
            class4_precision=float(data["class4_precision"]),
            class4_recall=float(data["class4_recall"]),
            entropy=float(data["entropy"]),
            zero_prediction_classes=int(data["zero_prediction_classes"]),
        )


@dataclass
class PromotionResult:
    """Aggregated governance verdict across seed runs.

    Carries summary statistics, a PASS/FAIL status, and actionable
    failure reasons / mitigation actions.
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

    def to_dict(self) -> dict[str, Any]:
        """Serialise to the dict shape expected by the governance report."""
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


def build_promotion_result(strict_seed_runs: list[dict[str, Any]]) -> PromotionResult:
    """Aggregate per-seed calibration metrics into a governance verdict.

    Mirrors the original _summarize_governance logic:

    - Computes cross-seed summary statistics (mean, std, min, max).
    - Validates against hard-coded quality gates:
        * std_macro_f1 <= 0.03
        * min_class4_recall >= 0.80
        * mean_class4_precision >= 0.25
        * zero_prediction_classes == 0 for all seeds
        * mean_entropy > 0.2
    - Emits actionable failure reasons and suggested mitigations.
    """
    macro_vals = [float(r["macro_f1"]) for r in strict_seed_runs]
    p4_prec_vals = [float(r["class4_precision"]) for r in strict_seed_runs]
    p4_rec_vals = [float(r["class4_recall"]) for r in strict_seed_runs]
    zero_vals = [int(r["zero_prediction_classes"]) for r in strict_seed_runs]
    entropy_vals = [float(r["entropy"]) for r in strict_seed_runs]

    mean_macro_f1 = float(np.mean(macro_vals))
    std_macro_f1 = float(np.std(macro_vals))
    mean_class4_precision = float(np.mean(p4_prec_vals))
    mean_class4_recall = float(np.mean(p4_rec_vals))
    min_class4_recall = float(np.min(p4_rec_vals))
    mean_entropy = float(np.mean(entropy_vals))
    max_zero_prediction_classes = int(max(zero_vals))

    failure_reasons: list[str] = []
    if std_macro_f1 > 0.03:
        failure_reasons.append("std_macro_f1_gt_0_03")
    if min_class4_recall < 0.80:
        failure_reasons.append("min_class4_recall_lt_0_80")
    if mean_class4_precision < 0.25:
        failure_reasons.append("mean_class4_precision_lt_0_25")
    if max_zero_prediction_classes != 0:
        failure_reasons.append("max_zero_prediction_classes_ne_0")
    if mean_entropy <= 0.2:
        failure_reasons.append("mean_entropy_le_0_2")

    actions: list[str] = []
    if mean_class4_precision < 0.25:
        actions.append("increase_tau_4")
    if min_class4_recall < 0.80:
        actions.append("increase_focal_gamma_up_to_1_5")
    if mean_entropy <= 0.2:
        actions.append("increase_temperature_max_to_5_0")

    status = "PASS" if not failure_reasons else "FAIL"

    return PromotionResult(
        mean_macro_f1=mean_macro_f1,
        std_macro_f1=std_macro_f1,
        mean_class4_precision=mean_class4_precision,
        mean_class4_recall=mean_class4_recall,
        min_class4_recall=min_class4_recall,
        mean_entropy=mean_entropy,
        max_zero_prediction_classes=max_zero_prediction_classes,
        status=status,
        failure_reasons=failure_reasons,
        actions=actions,
    )
