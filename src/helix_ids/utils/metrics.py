"""
HELIX-IDS Metrics Module

Provides evaluation metrics for intrusion detection, including:
- Per-class F1 scores
- Threat-weighted F1
- Production Readiness Index (PRI)
- Edge deployment metrics
"""

import time
from dataclasses import dataclass, field
from typing import Optional, cast

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from helix_ids.contracts.attack_taxonomy import THREAT_WEIGHTS
from helix_ids.governance.parameters import DEFAULT_GOVERNANCE_POLICY


@dataclass
class ModelMetrics:
    """Container for comprehensive model evaluation metrics."""

    # Classification metrics
    accuracy: float = 0.0
    macro_f1: float = 0.0
    weighted_f1: float = 0.0
    threat_weighted_f1: float = 0.0

    # Per-class metrics
    per_class_f1: dict[str, float] = field(default_factory=dict)
    per_class_precision: dict[str, float] = field(default_factory=dict)
    per_class_recall: dict[str, float] = field(default_factory=dict)

    # Minority class specific
    r2l_f1: float = 0.0
    u2r_f1: float = 0.0
    r2l_recall: float = 0.0
    u2r_recall: float = 0.0

    # Edge deployment metrics
    model_size_kb: float = 0.0
    inference_latency_ms: float = 0.0
    memory_footprint_kb: float = 0.0

    # PRI score
    pri_score: float = 0.0

    # Confusion matrix
    confusion_matrix: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "threat_weighted_f1": self.threat_weighted_f1,
            "per_class_f1": self.per_class_f1,
            "per_class_precision": self.per_class_precision,
            "per_class_recall": self.per_class_recall,
            "r2l_f1": self.r2l_f1,
            "u2r_f1": self.u2r_f1,
            "r2l_recall": self.r2l_recall,
            "u2r_recall": self.u2r_recall,
            "model_size_kb": self.model_size_kb,
            "inference_latency_ms": self.inference_latency_ms,
            "memory_footprint_kb": self.memory_footprint_kb,
            "pri_score": self.pri_score,
        }


@dataclass
class MetricsObject:
    """Strict metrics contract object returned by evaluate()."""

    dataset_id: str
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class_f1: dict[str, float]
    confusion_matrix: list[list[int]]
    ci95_lower: float
    ci95_upper: float
    ci95_width: float

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "per_class_f1": self.per_class_f1,
            "confusion_matrix": self.confusion_matrix,
            "ci95_lower": self.ci95_lower,
            "ci95_upper": self.ci95_upper,
            "ci95_width": self.ci95_width,
        }


def calculate_per_class_f1(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: Optional[list[str]] = None
) -> dict[str, float]:
    """
    Calculate F1 score for each class.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Optional class name mapping

    Returns:
        Dictionary of class -> F1 score
    """
    f1_scores = f1_score(y_true, y_pred, average=None, zero_division=0)

    if class_names is None:
        class_names = [str(i) for i in range(len(f1_scores))]

    return {name: float(score) for name, score in zip(class_names, f1_scores)}


def calculate_threat_weighted_f1(
    per_class_f1: dict[str, float], weights: Optional[dict[str, float]] = None
) -> float:
    """
    Calculate threat-severity-weighted F1 score.

    Critical attacks (R2L, U2R) have higher weights to ensure
    the model doesn't ignore them in favor of majority classes.

    Args:
        per_class_f1: Dictionary of class -> F1 score
        weights: Custom weights (default: THREAT_WEIGHTS)

    Returns:
        Weighted F1 score
    """
    if weights is None:
        weights = THREAT_WEIGHTS

    weighted_sum = 0.0
    weight_total = 0.0

    for cls, f1 in per_class_f1.items():
        w = weights.get(cls, 1.0)
        weighted_sum += w * f1
        weight_total += w

    return weighted_sum / weight_total if weight_total > 0 else 0.0


def calculate_pri_score(
    has_hardware_spec: bool = False,
    has_cross_dataset: bool = False,
    cross_dataset_partial: bool = False,
    has_power_measurement: bool = False,
    has_per_class_metrics: bool = False,
    per_class_partial: bool = False,
    has_drift_protocol: bool = False,
    has_xai_quantified: bool = False,
) -> tuple[dict[str, float], float]:
    """
    Calculate Production Readiness Index (PRI) score.

    PRI Framework (from paper):
    - C1 (0.20): Named hardware specification
    - C2 (0.25): Cross-dataset evaluation
    - C3 (0.15): Power measurement (mW)
    - C4 (0.20): Per-class F1 metrics
    - C5 (0.10): Drift/retraining protocol
    - C6 (0.10): XAI tradeoff quantified

    Production threshold: PRI >= 0.70

    Returns:
        Tuple of (criterion scores, total PRI)
    """
    if has_cross_dataset:
        c2_cross_dataset = 1.0
    elif cross_dataset_partial:
        c2_cross_dataset = 0.5
    else:
        c2_cross_dataset = 0.0

    if has_per_class_metrics:
        c4_per_class = 1.0
    elif per_class_partial:
        c4_per_class = 0.5
    else:
        c4_per_class = 0.0

    scores = {
        "C1_hardware": 1.0 if has_hardware_spec else 0.0,
        "C2_cross_dataset": c2_cross_dataset,
        "C3_power": 1.0 if has_power_measurement else 0.0,
        "C4_per_class": c4_per_class,
        "C5_drift": 1.0 if has_drift_protocol else 0.0,
        "C6_xai": 1.0 if has_xai_quantified else 0.0,
    }

    weights = {
        "C1_hardware": 0.20,
        "C2_cross_dataset": 0.25,
        "C3_power": 0.15,
        "C4_per_class": 0.20,
        "C5_drift": 0.10,
        "C6_xai": 0.10,
    }

    pri = sum(scores[k] * weights[k] for k in scores)

    return scores, pri


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[list[str]] = None,
    model_size_kb: float = 0.0,
    inference_latency_ms: float = 0.0,
) -> ModelMetrics:
    """
    Comprehensive model evaluation.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Class name mapping
        model_size_kb: Model size in KB
        inference_latency_ms: Inference latency in ms

    Returns:
        ModelMetrics object with all metrics
    """
    if class_names is None:
        class_names = ["Normal", "DoS", "Probe", "R2L", "U2R"]

    metrics = ModelMetrics()

    # Basic metrics
    metrics.accuracy = accuracy_score(y_true, y_pred)
    metrics.macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    metrics.weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Per-class metrics
    metrics.per_class_f1 = calculate_per_class_f1(y_true, y_pred, class_names)

    precision_scores = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_scores = recall_score(y_true, y_pred, average=None, zero_division=0)

    metrics.per_class_precision = {
        name: float(score) for name, score in zip(class_names, precision_scores)
    }
    metrics.per_class_recall = {
        name: float(score) for name, score in zip(class_names, recall_scores)
    }

    # Threat-weighted F1
    metrics.threat_weighted_f1 = calculate_threat_weighted_f1(metrics.per_class_f1)

    # Minority class specific
    metrics.r2l_f1 = metrics.per_class_f1.get("R2L", 0.0)
    metrics.u2r_f1 = metrics.per_class_f1.get("U2R", 0.0)
    metrics.r2l_recall = metrics.per_class_recall.get("R2L", 0.0)
    metrics.u2r_recall = metrics.per_class_recall.get("U2R", 0.0)

    # Edge metrics
    metrics.model_size_kb = model_size_kb
    metrics.inference_latency_ms = inference_latency_ms

    # Confusion matrix
    metrics.confusion_matrix = confusion_matrix(y_true, y_pred)

    return metrics


def print_evaluation_report(metrics: ModelMetrics, class_names: Optional[list[str]] = None):
    """Print formatted evaluation report."""
    if class_names is None:
        class_names = ["Normal", "DoS", "Probe", "R2L", "U2R"]

    print("\n" + "=" * 70)
    print("HELIX-IDS EVALUATION REPORT")
    print("=" * 70)

    print("\nOverall Metrics:")
    print(f"  Accuracy:           {metrics.accuracy:.4f} ({metrics.accuracy * 100:.2f}%)")
    print(f"  Macro-F1:           {metrics.macro_f1:.4f}")
    print(f"  Weighted-F1:        {metrics.weighted_f1:.4f}")
    print(f"  Threat-Weighted-F1: {metrics.threat_weighted_f1:.4f}")

    print("\nPer-Class F1 Scores:")
    print("-" * 50)
    print(f"  {'Class':<15} {'F1':>10} {'Precision':>12} {'Recall':>10}")
    print("-" * 50)

    for cls in class_names:
        f1 = metrics.per_class_f1.get(cls, 0)
        prec = metrics.per_class_precision.get(cls, 0)
        rec = metrics.per_class_recall.get(cls, 0)
        marker = " ← CRITICAL" if cls in ["R2L", "U2R"] else ""
        print(f"  {cls:<15} {f1:>10.4f} {prec:>12.4f} {rec:>10.4f}{marker}")

    print("\nMinority Class Analysis:")
    print(f"  R2L F1: {metrics.r2l_f1:.4f}, Recall: {metrics.r2l_recall:.4f}")
    print(f"  U2R F1: {metrics.u2r_f1:.4f}, Recall: {metrics.u2r_recall:.4f}")

    if metrics.r2l_f1 == 0:
        print("  ⚠️  WARNING: R2L detection FAILED (F1=0)")
    if metrics.u2r_f1 == 0:
        print("  ⚠️  WARNING: U2R detection FAILED (F1=0)")

    if metrics.model_size_kb > 0:
        print("\nEdge Deployment Metrics:")
        print(f"  Model Size:     {metrics.model_size_kb:.2f} KB")
        print(f"  Latency:        {metrics.inference_latency_ms:.4f} ms/sample")

    print("=" * 70)


def measure_inference_latency(
    model, x_sample: np.ndarray, n_runs: int = 100, warmup_runs: int = 10
) -> float:
    """
    Measure average inference latency.

    Args:
        model: Model with predict() method
        x_sample: Sample input data
        n_runs: Number of timing runs
        warmup_runs: Number of warmup runs (not counted)

    Returns:
        Average latency in milliseconds per sample
    """
    # Warmup
    for _ in range(warmup_runs):
        _ = model.predict(x_sample)

    # Timed runs
    start = time.perf_counter()
    for _ in range(n_runs):
        _ = model.predict(x_sample)
    elapsed = time.perf_counter() - start

    # Per-sample latency in ms
    return (elapsed / n_runs / len(x_sample)) * 1000


def estimate_model_size(model) -> float:
    """
    Estimate model size in KB.

    Args:
        model: Model object

    Returns:
        Estimated size in KB
    """
    import io
    import pickle

    buffer = io.BytesIO()
    pickle.dump(model, buffer)
    return buffer.tell() / 1024


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Contract-safe accuracy helper."""
    return float(accuracy_score(y_true, y_pred))


def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Contract-safe macro-F1 helper."""
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def compute_weighted_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Contract-safe weighted-F1 helper."""
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def compute_per_class_f1_array(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Contract-safe per-class F1 helper."""
    return np.asarray(f1_score(y_true, y_pred, average=None, zero_division=0), dtype=float)


def compute_binary_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Contract-safe binary F1 helper for one-vs-rest threshold tuning."""
    return float(f1_score(y_true, y_pred, zero_division=0))


def compute_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    target_names: list[str],
) -> dict[str, object]:
    """Contract-safe classification report helper."""
    from sklearn.metrics import classification_report

    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        zero_division=0,
        output_dict=True,
    )
    return cast(dict[str, object], report)


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> list[list[int]]:
    """Contract-safe confusion matrix helper."""
    matrix = np.asarray(confusion_matrix(y_true, y_pred), dtype=int)
    return [[int(value) for value in row] for row in matrix]


def bootstrap_macro_f1_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    seed: int,
) -> tuple[float, float, float]:
    """Deterministic stratified bootstrap 95% CI for macro-F1."""
    policy = DEFAULT_GOVERNANCE_POLICY.bootstrap
    rng = np.random.default_rng(seed + policy.seed_offset)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(y_true)

    class_indices = {cls: np.flatnonzero(y_true == cls) for cls in classes}
    class_probs = {cls: len(idx) / len(y_true) for cls, idx in class_indices.items()}

    estimates = []
    for _ in range(policy.n_replicates):
        sampled_indices: list[int] = []
        for cls in classes:
            n_cls = max(1, int(round(class_probs[cls] * len(y_true))))
            sampled_indices.extend(
                rng.choice(class_indices[cls], size=n_cls, replace=True).tolist()
            )

        sampled_indices_arr = np.array(sampled_indices, dtype=int)
        estimates.append(compute_macro_f1(y_true[sampled_indices_arr], y_pred[sampled_indices_arr]))

    lower = float(np.percentile(estimates, policy.lower_percentile))
    upper = float(np.percentile(estimates, policy.upper_percentile))
    return lower, upper, upper - lower


def evaluate(
    preds: np.ndarray,
    targets: np.ndarray,
    dataset_id: str,
    *,
    class_names: Optional[list[str]] = None,
    seed: int = 42,
) -> MetricsObject:
    """Immutable metrics contract entrypoint.

    Args:
        preds: Predicted class ids.
        targets: True class ids.
        dataset_id: Dataset identifier for lineage and reporting.
        class_names: Optional class names for per-class mapping.
        seed: Deterministic seed for bootstrap CI.
    """
    if class_names is None:
        class_names = [str(i) for i in range(int(np.max(targets)) + 1)]

    macro_f1 = compute_macro_f1(targets, preds)
    weighted_f1 = compute_weighted_f1(targets, preds)
    accuracy = compute_accuracy(targets, preds)
    per_class = compute_per_class_f1_array(targets, preds)
    ci_low, ci_high, ci_width = bootstrap_macro_f1_ci(targets, preds, seed=seed)

    per_class_f1 = {
        class_names[i]: float(per_class[i]) for i in range(min(len(class_names), len(per_class)))
    }

    return MetricsObject(
        dataset_id=dataset_id,
        accuracy=accuracy,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        per_class_f1=per_class_f1,
        confusion_matrix=compute_confusion_matrix(targets, preds),
        ci95_lower=ci_low,
        ci95_upper=ci_high,
        ci95_width=ci_width,
    )
