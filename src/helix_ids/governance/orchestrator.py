"""Stage-based governance orchestrator and structured gate event logging."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

from .failure_memory import FailureMemory
from .parameters import DEFAULT_GOVERNANCE_POLICY, GovernancePolicy

GateFn = Callable[[dict[str, Any]], Tuple[bool, Optional[float], Optional[float], Optional[str]]]

DEFAULT_STAGE_SEQUENCE = (
    "preload",
    "presplit",
    "pretrain",
    "intrain",
    "posteval",
    "prepromote",
)


@dataclass(frozen=True)
class GateDecision:
    """Structured gate output event consumed by CI parsers."""

    run_id: str
    stage: str
    gate: str
    status: str
    reason_code: str
    metric: float | None
    threshold: float | None
    dataset: str | None
    seed: int | None
    timestamp: float
    fingerprint: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "stage": self.stage,
            "gate": self.gate,
            "status": self.status,
            "reason_code": self.reason_code,
            "metric": self.metric,
            "threshold": self.threshold,
            "dataset": self.dataset,
            "seed": self.seed,
            "timestamp": self.timestamp,
            "fingerprint": self.fingerprint,
        }


class GateOrchestrator:
    """Single callable surface for all stage gates."""

    def __init__(
        self,
        *,
        policy: GovernancePolicy = DEFAULT_GOVERNANCE_POLICY,
        event_log_path: Path | None = None,
        failure_log_path: Path | None = None,
        auto_register_default_gates: bool = True,
        strict_missing_metrics: bool = False,
    ) -> None:
        self.policy = policy
        self._gates: dict[str, list[tuple[str, GateFn]]] = {}
        self.event_log_path = event_log_path
        self.failure_memory = (
            FailureMemory(failure_log_path) if failure_log_path is not None else None
        )
        self.strict_missing_metrics = strict_missing_metrics
        if auto_register_default_gates:
            self._register_default_gates()

    def register_gate(self, stage: str, gate_name: str, gate_fn: GateFn) -> None:
        self._gates.setdefault(stage, []).append((gate_name, gate_fn))

    def run(self, stage: str, context: dict[str, Any]) -> list[GateDecision]:
        """Run all gates registered for a stage and raise on first failure."""
        self._validate_stage_metric_schema(stage, context)

        decisions: list[GateDecision] = []
        for gate_name, gate_fn in self._gates.get(stage, []):
            passed, metric, threshold, reason_code = gate_fn(context)
            status = self._status_from_outcome(passed, reason_code)
            decision = GateDecision(
                run_id=str(context.get("run_id", "unknown")),
                stage=stage,
                gate=gate_name,
                status=status,
                reason_code=reason_code or ("OK" if status == "PASS" else "E-GATE-UNKNOWN"),
                metric=metric,
                threshold=threshold,
                dataset=context.get("dataset"),
                seed=context.get("seed"),
                timestamp=time.time(),
                fingerprint=context.get("fingerprint"),
            )
            self._emit(decision)
            decisions.append(decision)
            if status != "PASS":
                raise RuntimeError(
                    f"Gate failure at stage '{stage}' gate '{gate_name}': {decision.reason_code}"
                )

        return decisions

    def _expected_stage_metric_keys(self, stage: str) -> set[str]:
        keys: set[str] = set()
        for _, gate_fn in self._gates.get(stage, []):
            metric_key = getattr(gate_fn, "_metric_key", None)
            if isinstance(metric_key, str) and metric_key:
                keys.add(metric_key)
        return keys

    def _validate_stage_metric_schema(self, stage: str, context: dict[str, Any]) -> None:
        expected = self._expected_stage_metric_keys(stage)
        if not expected:
            return

        for metric_key in expected:
            if metric_key not in context:
                if self.strict_missing_metrics:
                    self.record_decision(
                        stage=stage,
                        gate="stage_schema",
                        context=context,
                        status="INVALID",
                        reason_code=f"E-GATE-SCHEMA-MISSING-METRIC-INVALID:{metric_key}",
                    )
                    raise RuntimeError(f"E-GATE-SCHEMA-MISSING-METRIC:{metric_key}")
                continue

            try:
                metric = float(context[metric_key])
            except (TypeError, ValueError):
                self.record_decision(
                    stage=stage,
                    gate="stage_schema",
                    context=context,
                    status="INVALID",
                    reason_code=f"E-GATE-SCHEMA-BAD-METRIC-TYPE-INVALID:{metric_key}",
                )
                raise RuntimeError(f"E-GATE-SCHEMA-BAD-METRIC-TYPE:{metric_key}") from None

            if not math.isfinite(metric):
                self.record_decision(
                    stage=stage,
                    gate="stage_schema",
                    context=context,
                    status="INVALID",
                    reason_code=f"E-GATE-SCHEMA-NONFINITE-INVALID:{metric_key}",
                )
                raise RuntimeError(f"E-GATE-SCHEMA-NONFINITE:{metric_key}")

    def run_stage_sequence(
        self,
        context: dict[str, Any],
        *,
        stages: tuple[str, ...] = DEFAULT_STAGE_SEQUENCE,
    ) -> list[GateDecision]:
        """Execute a full ordered stage sequence against a shared context."""
        decisions: list[GateDecision] = []
        for stage in stages:
            decisions.extend(self.run(stage, context))
        return decisions

    def _register_default_gates(self) -> None:
        self.register_gate("preload", "run_identity_present", self._gate_run_identity_present)
        self.register_gate("preload", "entrypoint_present", self._gate_entrypoint_present)
        self.register_gate(
            "preload",
            "preload_timeout",
            self._make_lte_gate(
                metric_key="preload_elapsed_seconds",
                threshold=self.policy.stage_timeouts.preload_seconds,
                fail_reason_code="E-TIMEOUT-PRELOAD",
            ),
        )

        self.register_gate(
            "presplit",
            "presplit_timeout",
            self._make_lte_gate(
                metric_key="presplit_elapsed_seconds",
                threshold=self.policy.stage_timeouts.presplit_seconds,
                fail_reason_code="E-TIMEOUT-PRESPLIT",
            ),
        )
        self.register_gate(
            "presplit",
            "dataset_identity_leakage",
            self._make_lte_gate(
                metric_key="dataset_identity_balanced_accuracy",
                threshold=self.policy.dataset_identity.max_balanced_accuracy,
                fail_reason_code="E-T0-DATASET-IDENTITY-LEAKAGE",
            ),
        )
        self.register_gate(
            "presplit",
            "split_train_rows_positive",
            self._make_gte_gate(
                metric_key="split_train_rows",
                threshold=1,
                fail_reason_code="E-T0-EMPTY-TRAIN-SPLIT",
            ),
        )
        self.register_gate(
            "presplit",
            "split_binary_class_count",
            self._make_gte_gate(
                metric_key="split_binary_class_count",
                threshold=2,
                fail_reason_code="E-T0-SPLIT-CLASS-COLLAPSE",
            ),
        )

        self.register_gate(
            "pretrain",
            "pretrain_timeout",
            self._make_lte_gate(
                metric_key="pretrain_elapsed_seconds",
                threshold=self.policy.stage_timeouts.pretrain_seconds,
                fail_reason_code="E-TIMEOUT-PRETRAIN",
            ),
        )
        self.register_gate(
            "pretrain",
            "family_class_weight_min",
            self._make_gt_gate(
                metric_key="family_class_weight_min",
                threshold=0.0,
                fail_reason_code="E-T1-FAMILY-WEIGHT-INVALID",
            ),
        )
        self.register_gate(
            "pretrain",
            "binary_class_weight_min",
            self._make_gt_gate(
                metric_key="binary_class_weight_min",
                threshold=0.0,
                fail_reason_code="E-T1-BINARY-WEIGHT-INVALID",
            ),
        )

        self.register_gate(
            "intrain",
            "intrain_timeout",
            self._make_lte_gate(
                metric_key="intrain_elapsed_seconds",
                threshold=self.policy.stage_timeouts.intrain_seconds,
                fail_reason_code="E-TIMEOUT-INTRAIN",
            ),
        )
        self.register_gate(
            "intrain",
            "low_entropy_consecutive_batches",
            self._make_lte_gate(
                metric_key="low_entropy_consecutive_batches",
                threshold=self.policy.training_abort.low_entropy_consecutive_batches,
                fail_reason_code="E-T2-LOW-ENTROPY",
            ),
        )
        self.register_gate(
            "intrain",
            "gradient_dominance",
            self._make_lte_gate(
                metric_key="gradient_dominance",
                threshold=self.policy.training_abort.gradient_dominance_threshold,
                fail_reason_code="E-T2-GRADIENT-DOMINANCE",
            ),
        )
        self.register_gate(
            "intrain",
            "epochs_without_improvement",
            self._make_lte_gate(
                metric_key="epochs_without_improvement",
                threshold=self.policy.training_abort.epochs_without_improvement,
                fail_reason_code="E-T2-NO-IMPROVEMENT",
            ),
        )

        self.register_gate(
            "posteval",
            "posteval_timeout",
            self._make_lte_gate(
                metric_key="posteval_elapsed_seconds",
                threshold=self.policy.stage_timeouts.posteval_seconds,
                fail_reason_code="E-TIMEOUT-POSTEVAL",
            ),
        )
        self.register_gate(
            "posteval",
            "macro_f1_ci_width",
            self._make_lte_gate(
                metric_key="macro_f1_ci_width",
                threshold=self.policy.bootstrap.max_ci_width,
                fail_reason_code="E-T2-CI-WIDTH",
            ),
        )
        self.register_gate(
            "posteval",
            "macro_f1_ci_lower_bound",
            self._make_gte_gate(
                metric_key="macro_f1_ci_lower",
                threshold=self.policy.bootstrap.min_ci95_lower_bound,
                fail_reason_code="E-T2-CI-LOWER-BOUND",
            ),
        )
        self.register_gate(
            "posteval",
            "abs_macro_f1_drift",
            self._make_lte_gate(
                metric_key="abs_macro_f1_drift",
                threshold=self.policy.drift.max_abs_macro_f1_drift,
                fail_reason_code="E-T2-DRIFT",
            ),
        )
        self.register_gate(
            "posteval",
            "abs_macro_f1_zscore",
            self._make_lte_gate(
                metric_key="abs_macro_f1_zscore",
                threshold=self.policy.drift.max_abs_z_score,
                fail_reason_code="E-T2-ZSCORE",
            ),
        )

        self.register_gate(
            "prepromote",
            "prepromote_timeout",
            self._make_lte_gate(
                metric_key="prepromote_elapsed_seconds",
                threshold=self.policy.stage_timeouts.prepromote_seconds,
                fail_reason_code="E-TIMEOUT-PREPROMOTE",
            ),
        )
        self.register_gate(
            "prepromote",
            "promotion_contract",
            self._gate_promotion_contract,
        )
        self.register_gate(
            "prepromote",
            "inter_seed_macro_f1_variance",
            self._make_lte_gate(
                metric_key="inter_seed_macro_f1_variance",
                threshold=self.policy.promotion.max_inter_seed_macro_f1_variance,
                fail_reason_code="E-T3-SEED-VARIANCE",
            ),
        )
        self.register_gate(
            "prepromote",
            "reproducibility_delta",
            self._make_lte_gate(
                metric_key="reproducibility_delta",
                threshold=self.policy.promotion.reproducibility_tolerance,
                fail_reason_code="E-T3-REPRODUCIBILITY",
            ),
        )

    def _maybe_float(
        self,
        context: dict[str, Any],
        metric_key: str,
        *,
        missing_reason_code: str,
    ) -> tuple[float | None, str | None, bool]:
        if metric_key not in context:
            if self.strict_missing_metrics:
                return None, missing_reason_code, False
            return None, "SKIP-MISSING-METRIC", True
        try:
            return float(context[metric_key]), None, True
        except (TypeError, ValueError):
            return None, f"E-GATE-BAD-METRIC-TYPE:{metric_key}", False

    def _make_lte_gate(
        self,
        *,
        metric_key: str,
        threshold: float,
        fail_reason_code: str,
    ) -> GateFn:
        def gate(context: dict[str, Any]) -> tuple[bool, float | None, float | None, str | None]:
            metric, reason_code, allowed = self._maybe_float(
                context,
                metric_key,
                missing_reason_code=f"E-GATE-MISSING-METRIC:{metric_key}",
            )
            if metric is None:
                return allowed, None, threshold, reason_code
            passed = metric <= threshold
            return passed, metric, threshold, None if passed else fail_reason_code

        setattr(gate, "_metric_key", metric_key)
        return gate

    def _make_gte_gate(
        self,
        *,
        metric_key: str,
        threshold: float,
        fail_reason_code: str,
    ) -> GateFn:
        def gate(context: dict[str, Any]) -> tuple[bool, float | None, float | None, str | None]:
            metric, reason_code, allowed = self._maybe_float(
                context,
                metric_key,
                missing_reason_code=f"E-GATE-MISSING-METRIC:{metric_key}",
            )
            if metric is None:
                return allowed, None, threshold, reason_code
            passed = metric >= threshold
            return passed, metric, threshold, None if passed else fail_reason_code

        setattr(gate, "_metric_key", metric_key)
        return gate

    def _make_gt_gate(
        self,
        *,
        metric_key: str,
        threshold: float,
        fail_reason_code: str,
    ) -> GateFn:
        def gate(context: dict[str, Any]) -> tuple[bool, float | None, float | None, str | None]:
            metric, reason_code, allowed = self._maybe_float(
                context,
                metric_key,
                missing_reason_code=f"E-GATE-MISSING-METRIC:{metric_key}",
            )
            if metric is None:
                return allowed, None, threshold, reason_code
            passed = metric > threshold
            return passed, metric, threshold, None if passed else fail_reason_code

        setattr(gate, "_metric_key", metric_key)
        return gate

    def _gate_run_identity_present(
        self,
        context: dict[str, Any],
    ) -> tuple[bool, float | None, float | None, str | None]:
        has_identity = bool(str(context.get("run_id", "")).strip())
        return has_identity, None, None, None if has_identity else "E-GATE-MISSING-RUN-ID"

    def _gate_entrypoint_present(
        self,
        context: dict[str, Any],
    ) -> tuple[bool, float | None, float | None, str | None]:
        has_entrypoint = bool(str(context.get("entrypoint", "")).strip())
        return has_entrypoint, None, None, None if has_entrypoint else "E-GATE-MISSING-ENTRYPOINT"

    def _gate_promotion_contract(
        self,
        context: dict[str, Any],
    ) -> tuple[bool, float | None, float | None, str | None]:
        seed_count = context.get("seed_run_count")
        if seed_count is None:
            return False, None, None, "E-T3-MISSING-SEED-RUN-COUNT-INVALID"

        try:
            seed_run_count = int(seed_count)
        except (TypeError, ValueError):
            return False, None, None, "E-T3-BAD-SEED-RUN-COUNT-INVALID"

        if seed_run_count < self.policy.promotion.min_seed_runs:
            if seed_run_count == 1:
                return (
                    False,
                    float(seed_run_count),
                    float(self.policy.promotion.min_seed_runs),
                    "E-T3-SINGLE-SEED-INVALID",
                )
            return (
                False,
                float(seed_run_count),
                float(self.policy.promotion.min_seed_runs),
                "E-T3-SEED-COUNT",
            )

        if "consensus_pass" not in context:
            return False, None, None, "E-T3-MISSING-CONSENSUS-PASS-INVALID"

        consensus_pass = context.get("consensus_pass")
        if not isinstance(consensus_pass, bool):
            return False, None, None, "E-T3-BAD-CONSENSUS-TYPE-INVALID"
        if not consensus_pass:
            return False, 0.0, 1.0, "E-T3-CONSENSUS"

        try:
            ci_lower = float(context["macro_f1_ci_lower"])
            ci_width = float(context["macro_f1_ci_width"])
        except KeyError:
            return False, None, None, "E-T3-MISSING-CI-METRICS-INVALID"
        except (TypeError, ValueError):
            return False, None, None, "E-T3-BAD-CI-METRICS-INVALID"

        if ci_lower < self.policy.bootstrap.min_ci95_lower_bound:
            return (
                False,
                ci_lower,
                self.policy.bootstrap.min_ci95_lower_bound,
                "E-T3-CI-LOWER-BOUND",
            )
        if ci_width > self.policy.bootstrap.max_ci_width:
            return False, ci_width, self.policy.bootstrap.max_ci_width, "E-T3-CI-WIDTH"

        return True, 1.0, 1.0, None

    def _status_from_outcome(self, passed: bool, reason_code: str | None) -> str:
        if passed:
            return "PASS"

        reason = reason_code or ""
        if "-INVALID" in reason or reason.startswith("E-INVALID"):
            return "INVALID"
        return "FAIL"

    def record_decision(
        self,
        *,
        stage: str,
        gate: str,
        context: dict[str, Any],
        status: str,
        reason_code: str,
        metric: float | None = None,
        threshold: float | None = None,
    ) -> GateDecision:
        decision = GateDecision(
            run_id=str(context.get("run_id", "unknown")),
            stage=stage,
            gate=gate,
            status=status,
            reason_code=reason_code,
            metric=metric,
            threshold=threshold,
            dataset=context.get("dataset"),
            seed=context.get("seed"),
            timestamp=time.time(),
            fingerprint=context.get("fingerprint"),
        )
        self._emit(decision)
        return decision

    def _emit(self, decision: GateDecision) -> None:
        self._validate_decision_schema(decision)
        if self.failure_memory is not None and decision.status != "PASS":
            self.failure_memory.record(
                run_id=decision.run_id,
                stage=decision.stage,
                gate=decision.gate,
                failure_type=decision.reason_code,
                fingerprint=decision.fingerprint,
                dataset=decision.dataset,
            )

        if self.event_log_path is None:
            return

        self.event_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.event_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(decision.to_dict(), sort_keys=True) + "\n")

    def _validate_decision_schema(self, decision: GateDecision) -> None:
        if decision.status not in {"PASS", "FAIL", "INVALID"}:
            raise RuntimeError("E-GATE-EVENT-SCHEMA:status")
        if not decision.run_id or not decision.stage or not decision.gate:
            raise RuntimeError("E-GATE-EVENT-SCHEMA:identity")
        if not decision.reason_code:
            raise RuntimeError("E-GATE-EVENT-SCHEMA:reason_code")
        if not math.isfinite(float(decision.timestamp)):
            raise RuntimeError("E-GATE-EVENT-SCHEMA:timestamp")

        for name, value in (("metric", decision.metric), ("threshold", decision.threshold)):
            if value is None:
                continue
            metric = float(value)
            if not math.isfinite(metric):
                raise RuntimeError(f"E-GATE-EVENT-SCHEMA:{name}")
