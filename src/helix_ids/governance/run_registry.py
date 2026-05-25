"""Run registry for lineage, fingerprint consistency, and drift checks."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from .failure_memory import FailureMemory


@dataclass(frozen=True)
class RunRegistryDecision:
    """Validation decision returned by registry operations."""

    accepted: bool
    reason_code: str
    reproducibility_delta: float | None = None
    reproducibility_threshold: float | None = None


class RunRegistry:
    """Append-only run registry with minimal governance validation."""

    REQUIRED_LINEAGE_KEYS = {
        "dataset_hashes",
        "schema_hash",
        "mapping_version",
        "model_artifact",
        "metrics_artifact",
    }

    def __init__(self, path: Path) -> None:
        self.path = path
        self.failure_memory = FailureMemory(path.parent / "failure_memory.jsonl")

    def _iter_records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []

        records: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = line.strip()
                if not payload:
                    continue
                records.append(json.loads(payload))
        return records

    def _append_record(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def _find_parent_record(
        self,
        parent_run_id: str | None,
        records: list[dict[str, Any]],
    ) -> tuple[RunRegistryDecision, dict[str, Any] | None]:
        if not parent_run_id:
            return RunRegistryDecision(True, "OK"), None

        for record in records:
            if record.get("run_id") == parent_run_id:
                return RunRegistryDecision(True, "OK"), record

        return RunRegistryDecision(False, "E-LINEAGE-MISSING-PARENT-INVALID"), None

    def _validate_parent_lineage_match(
        self,
        parent_record: dict[str, Any] | None,
        lineage: dict[str, Any] | None,
    ) -> RunRegistryDecision:
        if parent_record is None or lineage is None:
            return RunRegistryDecision(True, "OK")

        parent_lineage = parent_record.get("lineage")
        if not isinstance(parent_lineage, dict):
            return RunRegistryDecision(True, "OK")

        for key in ("dataset_hashes", "schema_hash", "mapping_version"):
            if key not in parent_lineage or key not in lineage:
                continue
            if parent_lineage.get(key) != lineage.get(key):
                return RunRegistryDecision(False, "E-LINEAGE-HASH-MISMATCH-INVALID")

        return RunRegistryDecision(True, "OK")

    def _validate_lineage_chain(
        self,
        lineage: dict[str, Any] | None,
        *,
        strict: bool,
    ) -> RunRegistryDecision:
        if lineage is None:
            return RunRegistryDecision(
                not strict, "E-LINEAGE-MISSING-CHAIN-INVALID" if strict else "SKIP-LINEAGE"
            )

        missing = sorted(self.REQUIRED_LINEAGE_KEYS - set(lineage.keys()))
        if missing:
            return RunRegistryDecision(False, f"E-LINEAGE-MISSING-LINK-INVALID:{','.join(missing)}")

        return RunRegistryDecision(True, "OK")

    def _validate_orphan_artifacts(
        self,
        lineage: dict[str, Any] | None,
        *,
        strict: bool,
    ) -> RunRegistryDecision:
        if lineage is None:
            return RunRegistryDecision(
                not strict, "E-LINEAGE-MISSING-CHAIN-INVALID" if strict else "SKIP-LINEAGE"
            )

        artifact_paths = [lineage.get("model_artifact"), lineage.get("metrics_artifact")]
        for artifact in artifact_paths:
            if not artifact:
                return RunRegistryDecision(False, "E-LINEAGE-MISSING-ARTIFACT-INVALID")
            if strict and not Path(str(artifact)).exists():
                return RunRegistryDecision(False, f"E-LINEAGE-ORPHAN-ARTIFACT-INVALID:{artifact}")

        return RunRegistryDecision(True, "OK")

    def _check_matching_fingerprint_record(
        self, record: dict[str, Any], fingerprint: str, dataset_id: str,
        macro_f1: float, seed: int | None, tolerance: float
    ) -> tuple[bool, str]:
        """Check if a record matches fingerprint criteria and validate metric consistency."""
        if record.get("fingerprint") != fingerprint:
            return False, ""
        if record.get("dataset_id") != dataset_id:
            return False, ""
        if seed is not None and record.get("seed") == seed:
            return False, ""

        prior_f1 = record.get("macro_f1")
        if prior_f1 is None:
            return False, ""
        if abs(float(prior_f1) - float(macro_f1)) > tolerance:
            return True, "E-FINGERPRINT-METRIC-MISMATCH-INVALID"
        return False, ""

    def _validate_fingerprint_consistency(
        self,
        fingerprint: str | None,
        dataset_id: str,
        macro_f1: float | None,
        records: list[dict[str, Any]],
        seed: int | None,
        *,
        tolerance: float,
        strict: bool,
    ) -> RunRegistryDecision:
        if not fingerprint:
            return RunRegistryDecision(
                not strict, "E-FINGERPRINT-MISSING-INVALID" if strict else "SKIP-FINGERPRINT"
            )
        if macro_f1 is None:
            return RunRegistryDecision(False, "E-FINGERPRINT-MISSING-METRIC-INVALID")

        for record in records:
            is_mismatch, error_code = self._check_matching_fingerprint_record(
                record, fingerprint, dataset_id, macro_f1, seed, tolerance
            )
            if is_mismatch:
                return RunRegistryDecision(False, error_code)

        return RunRegistryDecision(True, "OK")

    def _validate_same_seed_reproducibility(
        self,
        *,
        records: list[dict[str, Any]],
        dataset_id: str,
        seed: int | None,
        fingerprint: str | None,
        current_macro_f1: float | None,
        tolerance: float,
    ) -> RunRegistryDecision:
        if seed is None or fingerprint is None or current_macro_f1 is None:
            return RunRegistryDecision(
                True, "OK", reproducibility_delta=None, reproducibility_threshold=tolerance
            )

        previous = [
            record
            for record in records
            if record.get("dataset_id") == dataset_id
            and record.get("fingerprint") == fingerprint
            and record.get("seed") == seed
            and record.get("state") == "accepted"
            and record.get("macro_f1") is not None
        ]

        if not previous:
            return RunRegistryDecision(
                True, "OK", reproducibility_delta=0.0, reproducibility_threshold=tolerance
            )

        baseline = float(previous[-1]["macro_f1"])
        delta = abs(float(current_macro_f1) - baseline)
        if delta > tolerance:
            return RunRegistryDecision(
                False,
                "E-REPRODUCIBILITY-DELTA-INVALID",
                reproducibility_delta=delta,
                reproducibility_threshold=tolerance,
            )

        return RunRegistryDecision(
            True,
            "OK",
            reproducibility_delta=delta,
            reproducibility_threshold=tolerance,
        )

    def _reject(
        self,
        *,
        run_id: str,
        dataset_id: str,
        macro_f1: float | None,
        fingerprint: str | None,
        parent_run_id: str | None,
        lineage: dict[str, Any] | None,
        reason_code: str,
        seed: int | None,
        reproducibility_delta: float | None,
        reproducibility_threshold: float | None,
    ) -> RunRegistryDecision:
        self._append_record(
            {
                "run_id": run_id,
                "dataset_id": dataset_id,
                "macro_f1": macro_f1,
                "fingerprint": fingerprint,
                "parent_run_id": parent_run_id,
                "seed": seed,
                "lineage": lineage or {},
                "created_at": time.time(),
                "state": "rejected",
                "reason_code": reason_code,
            }
        )
        self.failure_memory.record(
            run_id=run_id,
            stage="registry",
            gate="validate_and_register",
            failure_type=reason_code,
            fingerprint=fingerprint,
            dataset=dataset_id,
        )
        return RunRegistryDecision(
            False,
            reason_code,
            reproducibility_delta=reproducibility_delta,
            reproducibility_threshold=reproducibility_threshold,
        )

    def validate_and_register(
        self,
        *,
        run_id: str,
        dataset_id: str,
        macro_f1: float | None,
        fingerprint: str | None,
        parent_run_id: str | None,
        seed: int | None,
        lineage: dict[str, Any] | None,
        tolerance: float,
        strict_lineage: bool,
        strict_orphan_artifacts: bool,
    ) -> RunRegistryDecision:
        """Validate record constraints then persist it if accepted."""
        records = self._iter_records()

        if macro_f1 is not None and not (0.0 <= float(macro_f1) <= 1.0):
            return self._reject(
                run_id=run_id,
                dataset_id=dataset_id,
                macro_f1=macro_f1,
                fingerprint=fingerprint,
                parent_run_id=parent_run_id,
                lineage=lineage,
                reason_code="E-METRIC-TAMPERING-INVALID",
                seed=seed,
                reproducibility_delta=None,
                reproducibility_threshold=tolerance,
            )

        parent, parent_record = self._find_parent_record(parent_run_id, records)
        if not parent.accepted:
            return self._reject(
                run_id=run_id,
                dataset_id=dataset_id,
                macro_f1=macro_f1,
                fingerprint=fingerprint,
                parent_run_id=parent_run_id,
                lineage=lineage,
                reason_code=parent.reason_code,
                seed=seed,
                reproducibility_delta=parent.reproducibility_delta,
                reproducibility_threshold=tolerance,
            )

        parent_lineage = self._validate_parent_lineage_match(parent_record, lineage)
        if not parent_lineage.accepted:
            return self._reject(
                run_id=run_id,
                dataset_id=dataset_id,
                macro_f1=macro_f1,
                fingerprint=fingerprint,
                parent_run_id=parent_run_id,
                lineage=lineage,
                reason_code=parent_lineage.reason_code,
                seed=seed,
                reproducibility_delta=parent_lineage.reproducibility_delta,
                reproducibility_threshold=tolerance,
            )

        chain = self._validate_lineage_chain(lineage, strict=strict_lineage)
        if not chain.accepted:
            return self._reject(
                run_id=run_id,
                dataset_id=dataset_id,
                macro_f1=macro_f1,
                fingerprint=fingerprint,
                parent_run_id=parent_run_id,
                lineage=lineage,
                reason_code=chain.reason_code,
                seed=seed,
                reproducibility_delta=chain.reproducibility_delta,
                reproducibility_threshold=tolerance,
            )

        orphan = self._validate_orphan_artifacts(
            lineage,
            strict=strict_orphan_artifacts,
        )
        if not orphan.accepted:
            return self._reject(
                run_id=run_id,
                dataset_id=dataset_id,
                macro_f1=macro_f1,
                fingerprint=fingerprint,
                parent_run_id=parent_run_id,
                lineage=lineage,
                reason_code=orphan.reason_code,
                seed=seed,
                reproducibility_delta=orphan.reproducibility_delta,
                reproducibility_threshold=tolerance,
            )

        reproducibility = self._validate_same_seed_reproducibility(
            records=records,
            dataset_id=dataset_id,
            seed=seed,
            fingerprint=fingerprint,
            current_macro_f1=macro_f1,
            tolerance=tolerance,
        )
        if not reproducibility.accepted:
            return self._reject(
                run_id=run_id,
                dataset_id=dataset_id,
                macro_f1=macro_f1,
                fingerprint=fingerprint,
                parent_run_id=parent_run_id,
                lineage=lineage,
                reason_code=reproducibility.reason_code,
                seed=seed,
                reproducibility_delta=reproducibility.reproducibility_delta,
                reproducibility_threshold=reproducibility.reproducibility_threshold,
            )

        consistency = self._validate_fingerprint_consistency(
            fingerprint,
            dataset_id,
            macro_f1,
            records,
            seed,
            tolerance=tolerance,
            strict=strict_lineage,
        )
        if not consistency.accepted:
            return self._reject(
                run_id=run_id,
                dataset_id=dataset_id,
                macro_f1=macro_f1,
                fingerprint=fingerprint,
                parent_run_id=parent_run_id,
                lineage=lineage,
                reason_code=consistency.reason_code,
                seed=seed,
                reproducibility_delta=consistency.reproducibility_delta,
                reproducibility_threshold=tolerance,
            )

        record = {
            "run_id": run_id,
            "dataset_id": dataset_id,
            "macro_f1": macro_f1,
            "fingerprint": fingerprint,
            "parent_run_id": parent_run_id,
            "seed": seed,
            "lineage": lineage or {},
            "created_at": time.time(),
            "state": "accepted",
        }
        self._append_record(record)
        return RunRegistryDecision(
            True,
            "OK",
            reproducibility_delta=reproducibility.reproducibility_delta,
            reproducibility_threshold=reproducibility.reproducibility_threshold,
        )

    def compute_drift(
        self,
        *,
        dataset_id: str,
        current_macro_f1: float,
        baseline_window_runs: int,
        phase_regime: str | None = None,
    ) -> tuple[float, float]:
        """Return absolute drift and z-score using accepted runs as baseline."""
        records = [
            record
            for record in self._iter_records()
            if record.get("dataset_id") == dataset_id and record.get("state") == "accepted"
        ]
        if phase_regime:
            records = [
                record
                for record in records
                if isinstance(record.get("lineage"), dict)
                and str(record["lineage"].get("phase_regime", "")).strip() == str(phase_regime)
            ]
        if not records:
            return 0.0, 0.0

        baseline = records[-baseline_window_runs:]
        values = [
            float(record["macro_f1"]) for record in baseline if record.get("macro_f1") is not None
        ]
        if not values:
            return 0.0, 0.0

        baseline_mean = mean(values)
        drift = abs(float(current_macro_f1) - baseline_mean)

        sigma = pstdev(values)
        if abs(sigma) < 1e-9:  # Check if approximately zero
            return drift, 0.0

        z_score = abs((float(current_macro_f1) - baseline_mean) / sigma)
        return drift, z_score

    def failure_anomaly_baseline(self, *, dataset_id: str, window: int = 50) -> float:
        return self.failure_memory.failure_rate_baseline(dataset=dataset_id, window=window)

    def detect_failure_regression(
        self,
        *,
        dataset_id: str,
        current_window: int = 10,
        baseline_window: int = 50,
    ) -> bool:
        return self.failure_memory.regression_spike(
            dataset=dataset_id,
            current_window=current_window,
            baseline_window=baseline_window,
        )
