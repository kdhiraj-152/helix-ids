from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score


class SchemaDriftError(AssertionError):
    """Raised when the observed prediction/schema shape differs from baseline."""


class ContractViolationError(RuntimeError):
    """Raised when monitoring detects a contract violation. Payload stored on the exception as `.payload`."""

    def __init__(self, message: str, payload: dict | None = None):
        super().__init__(message)
        self.payload = payload


@dataclass(frozen=True)
class MonitorConfig:
    class_distribution_tolerance: float = 0.15
    entropy_tolerance: float = 0.20
    macro_f1_tolerance: float = 0.05
    coverage_override_rate_tolerance: float = 0.02


def _safe_entropy(p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0)
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))


def compute_zero_prediction_classes(preds: np.ndarray, active_classes: list[int]) -> int:
    preds = np.asarray(preds, dtype=np.int64)
    missing = 0
    for c in active_classes:
        if not np.any(preds == int(c)):
            missing += 1
    return int(missing)


class LiveMonitor:
    def __init__(
        self,
        *,
        baseline_class_distribution: np.ndarray,
        baseline_entropy: float,
        baseline_macro_f1: float | None = None,
        config: MonitorConfig = MonitorConfig(),
    ) -> None:
        p = np.asarray(baseline_class_distribution, dtype=np.float64)
        p = p / max(1e-12, float(p.sum()))
        self.baseline_class_distribution = p
        self.baseline_entropy = float(baseline_entropy)
        self.baseline_macro_f1 = None if baseline_macro_f1 is None else float(baseline_macro_f1)
        self.config = config

    # NOTE: alignment/repair behavior intentionally removed. Any mismatch
    # between observed and baseline class cardinality is considered a
    # contract violation and will be emitted and raised.

    def _compute_macro_f1_delta(
        self,
        preds: np.ndarray,
        labels: np.ndarray | None,
    ) -> tuple[float | None, float | None]:
        if labels is None:
            return None, None
        y_true = np.asarray(labels, dtype=np.int64)
        macro_f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))
        if self.baseline_macro_f1 is None:
            return macro_f1, None
        return macro_f1, abs(macro_f1 - self.baseline_macro_f1)

    @staticmethod
    def _compute_override_rate(
        coverage_override_applied: np.ndarray | list[bool] | None,
    ) -> float | None:
        if coverage_override_applied is None:
            return None
        override_arr = np.asarray(coverage_override_applied, dtype=bool)
        if override_arr.ndim == 0:
            override_arr = override_arr.reshape(1)
        return float(np.mean(override_arr.astype(np.float64)))

    def _collect_alerts(
        self,
        *,
        l1: float,
        entropy_delta: float,
        macro_f1_delta: float | None,
        coverage_override_rate: float | None,
    ) -> list[str]:
        alerts: list[str] = []
        if l1 > self.config.class_distribution_tolerance:
            alerts.append("class_distribution_deviation")
        if entropy_delta > self.config.entropy_tolerance:
            alerts.append("entropy_deviation")
        if macro_f1_delta is not None and macro_f1_delta > self.config.macro_f1_tolerance:
            alerts.append("macro_f1_deviation")
        if (
            coverage_override_rate is not None
            and coverage_override_rate > self.config.coverage_override_rate_tolerance
        ):
            alerts.append("coverage_override_rate_deviation")
        return alerts

    def evaluate(
        self,
        preds: np.ndarray,
        labels: np.ndarray | None = None,
        *,
        coverage_override_applied: np.ndarray | list[bool] | None = None,
        # Optional telemetry/context for strict contract violations
        producer: str | None = None,
        artifact_path: str | None = None,
        schema_hash_expected: str | None = None,
        schema_hash_actual: str | None = None,
        feature_names_expected: list[str] | None = None,
        feature_names_actual: list[str] | None = None,
        telemetry_dir: Path | None = None,
    ) -> dict[str, Any]:
        preds = np.asarray(preds, dtype=np.int64)
        max_class = int(max(preds.max(initial=0), len(self.baseline_class_distribution) - 1))
        counts = np.bincount(preds, minlength=max_class + 1).astype(np.float64)
        if counts.sum() <= 0:
            counts[0] = 1.0
        dist = counts / counts.sum()

        # Strict: do not align/repair shapes. Cardinality mismatch -> drift.
        baseline = self.baseline_class_distribution
        if dist.shape[0] != baseline.shape[0]:
            # Build structured payload
            missing_features: list[str] = []
            extra_features: list[str] = []
            expected_features = list(feature_names_expected or [])
            actual_features = list(feature_names_actual or [])
            if expected_features and actual_features:
                missing_features = [f for f in expected_features if f not in actual_features]
                extra_features = [f for f in actual_features if f not in expected_features]

            payload = {
                "event": "schema_drift_detected",
                "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
                    "+00:00", "Z"
                ),
                "schema_hash_expected": schema_hash_expected,
                "schema_hash_actual": schema_hash_actual,
                "feature_names_expected": expected_features,
                "feature_names_actual": actual_features,
                "missing_features": missing_features,
                "extra_features": extra_features,
                "producer": producer,
                "artifact_path": artifact_path,
                "observed_class_count": int(dist.shape[0]),
                "expected_class_count": int(baseline.shape[0]),
            }

            # Persist telemetry if requested
            try:
                if telemetry_dir is not None:
                    telemetry_dir = Path(telemetry_dir)
                    telemetry_dir.mkdir(parents=True, exist_ok=True)
                    filename = telemetry_dir / f"schema_drift_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
                    filename.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
            except Exception:
                # Best-effort only; do not mask original violation
                pass

            raise ContractViolationError("Schema/cardinality drift detected", payload=payload)

        l1 = float(np.abs(dist - baseline).sum())
        entropy = _safe_entropy(dist)
        entropy_delta = abs(entropy - self.baseline_entropy)

        macro_f1, macro_f1_delta = self._compute_macro_f1_delta(preds, labels)
        coverage_override_rate = self._compute_override_rate(coverage_override_applied)
        alerts = self._collect_alerts(
            l1=l1,
            entropy_delta=entropy_delta,
            macro_f1_delta=macro_f1_delta,
            coverage_override_rate=coverage_override_rate,
        )

        return {
            "class_distribution": dist.tolist(),
            "entropy": entropy,
            "macro_f1": macro_f1,
            "deviation": {
                "class_distribution_l1": l1,
                "entropy_abs": entropy_delta,
                "macro_f1_abs": macro_f1_delta,
            },
            "coverage_override_rate": coverage_override_rate,
            "coverage_override_threshold": float(self.config.coverage_override_rate_tolerance),
            "alert": len(alerts) > 0,
            "alerts": alerts,
        }
