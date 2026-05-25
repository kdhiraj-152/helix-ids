"""Entrypoint wrapper utilities enforcing import-boundary governance."""

from __future__ import annotations

import functools
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from .orchestrator import DEFAULT_STAGE_SEQUENCE, GateOrchestrator
from .parameters import DEFAULT_GOVERNANCE_POLICY
from .run_registry import RunRegistry


def _resolve_policy_from_env():
    """Resolve governance policy profile from environment.

    `HELIX_GOV_POLICY_PROFILE=smoke` relaxes CI lower-bound for short smoke
    pipeline runs while leaving strict defaults unchanged.
    """
    profile = os.environ.get("HELIX_GOV_POLICY_PROFILE", "").strip().lower()
    if profile == "smoke":
        return replace(
            DEFAULT_GOVERNANCE_POLICY,
            bootstrap=replace(
                DEFAULT_GOVERNANCE_POLICY.bootstrap,
                min_ci95_lower_bound=0.0,
            ),
            drift=replace(
                DEFAULT_GOVERNANCE_POLICY.drift,
                max_abs_z_score=1_000_000.0,
            ),
            promotion=replace(
                DEFAULT_GOVERNANCE_POLICY.promotion,
                min_seed_runs=1,
            ),
        )
    return DEFAULT_GOVERNANCE_POLICY


def _is_smoke_profile() -> bool:
    return os.environ.get("HELIX_GOV_POLICY_PROFILE", "").strip().lower() == "smoke"


def _parse_env_context() -> dict[str, Any]:
    payload = os.environ.get("HELIX_GOV_CONTEXT", "").strip()
    if not payload:
        return {}
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return {}
    if isinstance(parsed, dict):
        return parsed
    return {}


def _parse_stage_sequence() -> tuple[str, ...]:
    configured = os.environ.get("HELIX_GOV_STAGE_SEQUENCE", "").strip()
    if not configured:
        return DEFAULT_STAGE_SEQUENCE

    parsed = tuple(stage.strip() for stage in configured.split(",") if stage.strip())
    if not parsed:
        return DEFAULT_STAGE_SEQUENCE
    return parsed


def _extract_stage_payloads(result: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(result, dict):
        return {}

    payload = result.get("governance_stages")
    if not isinstance(payload, dict):
        return {}

    stage_payloads: dict[str, dict[str, Any]] = {}
    for stage, stage_context in payload.items():
        if isinstance(stage, str) and isinstance(stage_context, dict):
            stage_payloads[stage] = stage_context
    return stage_payloads


def _extract_shared_context(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {}

    for key in ("governance_context", "_governance_context"):
        value = result.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _extract_run_record(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {}

    payload = result.get("governance_run_record")
    if isinstance(payload, dict):
        return payload
    return {}


def _extract_dataset_id(result: Any, base_context: dict[str, Any]) -> str:
    if isinstance(result, dict):
        for key in ("dataset_id", "governance_dataset_id"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value

    dataset = base_context.get("dataset")
    if isinstance(dataset, str) and dataset.strip():
        return dataset
    return "global"


def _coerce_optional_int(value: object) -> int | None:
    """Safely coerce optional seed-like values to int."""
    if value is None:
        return None
    try:
        stripped = str(value).strip()
        if not stripped:
            return None
        return int(stripped)
    except (TypeError, ValueError):
        return None


def _resolve_lineage_payload(run_record: dict[str, Any], smoke_profile: bool) -> dict[str, Any] | None:
    lineage_payload = run_record.get("lineage") if isinstance(run_record.get("lineage"), dict) else None
    if not (smoke_profile and isinstance(lineage_payload, dict)):
        return lineage_payload

    required = set(getattr(RunRegistry, "REQUIRED_LINEAGE_KEYS", set()))
    if required and (required - set(lineage_payload.keys())):
        return None
    return lineage_payload


def _build_run_registry_kwargs(
    run_record: dict[str, Any],
    *,
    run_id: str,
    dataset_id: str,
    shared_context: dict[str, Any],
    base_context: dict[str, Any],
    lineage_payload: dict[str, Any] | None,
    tolerance: float,
    smoke_profile: bool,
) -> dict[str, Any]:
    macro_f1_value = run_record.get("macro_f1")
    fingerprint_value = run_record.get("fingerprint")
    parent_run_id = run_record.get("parent_run_id")
    seed_value = run_record.get("seed")

    return {
        "run_id": str(run_record.get("run_id") or run_id),
        "dataset_id": str(run_record.get("dataset_id") or dataset_id),
        "macro_f1": float(macro_f1_value) if macro_f1_value is not None else None,
        "fingerprint": str(fingerprint_value) if fingerprint_value else None,
        "parent_run_id": str(parent_run_id) if parent_run_id else os.environ.get("HELIX_PARENT_RUN_ID"),
        "seed": _coerce_optional_int(
            seed_value
            if seed_value is not None
            else shared_context.get("seed", base_context.get("seed"))
        ),
        "lineage": lineage_payload,
        "tolerance": tolerance,
        "strict_lineage": not smoke_profile,
        "strict_orphan_artifacts": not smoke_profile,
    }


def _run_governance_stages(
    orchestrator: GateOrchestrator,
    preload_stage: str,
    base_context: dict[str, Any],
    result: Any,
) -> None:
    stage_sequence = _parse_stage_sequence()
    stage_payloads = _extract_stage_payloads(result)

    if stage_payloads:
        for stage in stage_sequence:
            if stage == preload_stage or stage not in stage_payloads:
                continue
            stage_context = dict(base_context)
            stage_context.update(_extract_shared_context(result))
            stage_context.update(stage_payloads[stage])
            orchestrator.run(stage, stage_context)
        return

    context = dict(base_context)
    context.update(_extract_shared_context(result))
    for stage in stage_sequence:
        if stage != preload_stage:
            orchestrator.run(stage, context)


def _register_run_record(
    orchestrator: GateOrchestrator,
    run_id: str,
    base_context: dict[str, Any],
    result: Any,
) -> None:
    run_record = _extract_run_record(result)
    if not run_record:
        return

    shared_context = _extract_shared_context(result)
    registry_path = Path(
        os.environ.get(
            "HELIX_RUN_REGISTRY",
            "results/gates/run_registry.jsonl",
        )
    )
    dataset_id = _extract_dataset_id(result, base_context)

    policy = DEFAULT_GOVERNANCE_POLICY
    smoke_profile = _is_smoke_profile()
    lineage_payload = _resolve_lineage_payload(run_record, smoke_profile)
    registry_kwargs = _build_run_registry_kwargs(
        run_record,
        run_id=run_id,
        dataset_id=dataset_id,
        shared_context=shared_context,
        base_context=base_context,
        lineage_payload=lineage_payload,
        tolerance=policy.promotion.reproducibility_tolerance,
        smoke_profile=smoke_profile,
    )
    decision = RunRegistry(registry_path).validate_and_register(**registry_kwargs)

    reproducibility_status = "PASS" if decision.accepted else "INVALID"
    reproducibility_reason = "OK" if decision.accepted else decision.reason_code
    orchestrator.record_decision(
        stage="prepromote",
        gate="reproducibility_check",
        context=base_context,
        status=reproducibility_status,
        reason_code=reproducibility_reason,
        metric=decision.reproducibility_delta,
        threshold=decision.reproducibility_threshold,
    )

    if not decision.accepted:
        raise RuntimeError(str(decision.reason_code))


def governed_entrypoint(
    *,
    entrypoint_id: str,
    preload_stage: str = "preload",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that forces entrypoint execution through GateOrchestrator at import boundary."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            wrapper_start = os.times().elapsed
            run_id = os.environ.get("HELIX_RUN_ID", f"{entrypoint_id}-local")
            event_path = os.environ.get("HELIX_GATE_EVENTS", "results/gates/gate_events.jsonl")
            failure_path = os.environ.get(
                "HELIX_FAILURE_MEMORY", "results/gates/failure_memory.jsonl"
            )
            strict_missing_metrics = True
            orchestrator = GateOrchestrator(
                policy=_resolve_policy_from_env(),
                event_log_path=Path(event_path),
                failure_log_path=Path(failure_path),
                strict_missing_metrics=strict_missing_metrics,
            )
            base_context = {
                "run_id": run_id,
                "entrypoint": entrypoint_id,
                "dataset": kwargs.get("dataset"),
                "seed": kwargs.get("seed"),
                "fingerprint": kwargs.get("fingerprint"),
                "preload_elapsed_seconds": max(0.001, os.times().elapsed - wrapper_start),
            }
            base_context.update(_parse_env_context())

            # Import-boundary execution for non-bypassable entrypoint registration.
            orchestrator.run(preload_stage, base_context)

            result = func(*args, **kwargs)
            _run_governance_stages(orchestrator, preload_stage, base_context, result)
            _register_run_record(orchestrator, run_id, base_context, result)

            return result

        setattr(wrapped, "__governed_entrypoint__", True)
        setattr(wrapped, "__governed_entrypoint_id__", entrypoint_id)
        return wrapped

    return decorator
