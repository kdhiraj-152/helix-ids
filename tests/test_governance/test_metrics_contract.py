"""Tests for deterministic metrics contract behavior."""

import numpy as np

from helix_ids.utils.metrics import evaluate


def test_evaluate_returns_bootstrap_ci_and_confusion_matrix():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 2, 2])

    metrics = evaluate(y_pred, y_true, "unit", class_names=["Normal", "DoS", "Probe"], seed=13)

    assert metrics.dataset_id == "unit"
    assert "Normal" in metrics.per_class_f1
    assert len(metrics.confusion_matrix) == 3
    assert metrics.ci95_lower <= metrics.macro_f1 <= metrics.ci95_upper


def test_evaluate_bootstrap_is_deterministic_for_same_seed():
    y_true = np.array([0, 0, 1, 1, 2, 2, 2, 1, 0])
    y_pred = np.array([0, 0, 1, 2, 2, 2, 1, 1, 0])

    a = evaluate(y_pred, y_true, "unit", class_names=["Normal", "DoS", "Probe"], seed=42)
    b = evaluate(y_pred, y_true, "unit", class_names=["Normal", "DoS", "Probe"], seed=42)

    assert a.ci95_lower == b.ci95_lower
    assert a.ci95_upper == b.ci95_upper
    assert a.ci95_width == b.ci95_width
