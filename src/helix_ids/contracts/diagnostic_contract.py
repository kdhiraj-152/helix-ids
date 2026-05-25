"""Global diagnostic contract schema, migration, and runtime guards."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Optional, TypedDict

CONTRACT_VERSION = "2.1"
DECISION_MODES = {"probe", "action", "non_identifiable"}
DECISION_TRANSITIONS = {
    "probe": {"probe", "action", "non_identifiable"},
    "action": {"probe"},
    "non_identifiable": set(),
}


class DiagnosticContract(TypedDict):
    mode: str
    confidence: float
    probe_plan: list[dict[str, Any]]
    diagnostic_cycle: dict[str, Any]
    terminal_reason: Optional[str]


def migrate_contract_payload(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    version = str(out.get("version", "2.0"))
    if version == "2.0":
        out["protocol"] = out.get("protocol", "v1")
        out["version"] = CONTRACT_VERSION
    else:
        out.setdefault("protocol", "v1")
        out.setdefault("version", CONTRACT_VERSION)
    return out


def validate_diagnostic_contract(contract: DiagnosticContract) -> None:
    confidence = float(contract["confidence"])
    mode = str(contract["mode"])
    if not (0.0 <= confidence <= 1.0):
        raise AssertionError("confidence must be within [0, 1]")
    if mode not in DECISION_MODES:
        raise AssertionError(f"mode must be one of {sorted(DECISION_MODES)}")
    if mode == "action" and confidence < 0.6:
        raise AssertionError("action mode requires confidence >= 0.6")


def enforce_decision_transition(current_mode: str, next_mode: str) -> None:
    if current_mode not in DECISION_TRANSITIONS:
        raise AssertionError(f"unknown current_mode: {current_mode}")
    if next_mode not in DECISION_TRANSITIONS[current_mode]:
        raise AssertionError(f"illegal transition: {current_mode} -> {next_mode}")


def replay_and_compare(
    input_payload: dict[str, Any],
    expected_output: dict[str, Any],
    evaluator: Callable[[dict[str, Any]], dict[str, Any]],
) -> None:
    replay_output = evaluator(deepcopy(input_payload))
    if replay_output != expected_output:
        raise AssertionError("replay_mismatch_detected")
