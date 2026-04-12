"""Failure memory persistence and regression/anomaly baselines."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FailureEvent:
    """Structured failure event persisted for governance memory."""

    run_id: str
    stage: str
    gate: str
    failure_type: str
    fingerprint: str | None
    dataset: str | None
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "stage": self.stage,
            "gate": self.gate,
            "failure_type": self.failure_type,
            "fingerprint": self.fingerprint,
            "dataset": self.dataset,
            "timestamp": self.timestamp,
        }


class FailureMemory:
    """Append-only failure memory store used for anomaly and regression signals."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def record(
        self,
        *,
        run_id: str,
        stage: str,
        gate: str,
        failure_type: str,
        fingerprint: str | None,
        dataset: str | None,
    ) -> None:
        event = FailureEvent(
            run_id=run_id,
            stage=stage,
            gate=gate,
            failure_type=failure_type,
            fingerprint=fingerprint,
            dataset=dataset,
            timestamp=time.time(),
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")

    def _iter(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = line.strip()
                if not payload:
                    continue
                rows.append(json.loads(payload))
        return rows

    def failure_rate_baseline(self, *, dataset: str | None, window: int = 50) -> float:
        rows = [row for row in self._iter() if dataset is None or row.get("dataset") == dataset]
        if not rows:
            return 0.0
        return min(1.0, len(rows[-window:]) / float(window))

    def regression_spike(
        self,
        *,
        dataset: str | None,
        current_window: int = 10,
        baseline_window: int = 50,
        multiplier: float = 1.5,
    ) -> bool:
        rows = [row for row in self._iter() if dataset is None or row.get("dataset") == dataset]
        if len(rows) < current_window:
            return False

        baseline_rate = self.failure_rate_baseline(dataset=dataset, window=baseline_window)
        current_rate = min(1.0, len(rows[-current_window:]) / float(current_window))

        if baseline_rate == 0.0:
            return current_rate > 0.0
        return current_rate >= baseline_rate * multiplier
