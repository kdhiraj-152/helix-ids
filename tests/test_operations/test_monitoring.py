from __future__ import annotations

import numpy as np

from helix_ids.operations.monitoring import LiveMonitor, MonitorConfig, compute_zero_prediction_classes


def test_zero_prediction_classes_counts_missing() -> None:
    preds = np.array([0, 0, 1, 1, 1])
    missing = compute_zero_prediction_classes(preds, [0, 1, 2, 3])
    assert missing == 2


def test_live_monitor_triggers_alert_on_large_drift() -> None:
    monitor = LiveMonitor(
        baseline_class_distribution=np.array([0.5, 0.5]),
        baseline_entropy=0.693147,
        baseline_macro_f1=0.8,
        config=MonitorConfig(class_distribution_tolerance=0.2, entropy_tolerance=0.2, macro_f1_tolerance=0.05),
    )

    preds = np.array([0] * 95 + [1] * 5)
    labels = np.array([0] * 50 + [1] * 50)
    result = monitor.evaluate(preds, labels)
    assert result["alert"] is True
    assert len(result["alerts"]) >= 1


def test_live_monitor_alerts_on_high_coverage_override_rate() -> None:
    monitor = LiveMonitor(
        baseline_class_distribution=np.array([0.5, 0.5]),
        baseline_entropy=0.693147,
        config=MonitorConfig(
            class_distribution_tolerance=1.0,
            entropy_tolerance=1.0,
            macro_f1_tolerance=1.0,
            coverage_override_rate_tolerance=0.02,
        ),
    )

    preds = np.array([0, 1] * 50)
    overrides = np.array([True] * 3 + [False] * 97)  # 3% > 2%
    result = monitor.evaluate(preds, coverage_override_applied=overrides)
    assert result["coverage_override_rate"] == 0.03
    assert result["alert"] is True
    assert "coverage_override_rate_deviation" in result["alerts"]
