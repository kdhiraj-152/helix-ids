"""Governance execution, promotion, registry, and artifact publication.

Phase 13A-4 extraction from train_helix_ids_full.py main().

Moved:
  - Governance execution pipeline (post-training)     — lines 6938–7141
  - Multi-seed governance path                       — lines 1388–1504 (reimpl.)
  - Helper: _resolve_governance_policy                — line 148
  - Helper: _find_latest_ab_raw_metrics               — line 724
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np

from helix_ids.governance.promotion import SeedRunSummary, aggregate_seed_runs
from helix_ids.governance.run_registry import RunRegistry
from scripts.training.orchestration import (
    GovernanceResult,
    OrchestrationResult,
    ParsedConfig,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = _PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

HELIX_FULL_RESULTS_DIR = Path("results/helix_full")


# ============================================================================
# Lazy-import wrappers for helpers that remain in train_helix_ids_full.py
# ============================================================================


def _lazy_import(attr: str) -> Any:
    import importlib

    return getattr(
        importlib.import_module("scripts.training.train_helix_ids_full"), attr
    )


def _sha256_file(path: Path) -> str:
    return _lazy_import("_sha256_file")(path)  # type: ignore[no-any-return]


def _atomic_write_json(path: Path, payload: Any) -> None:
    return _lazy_import("_atomic_write_json")(path, payload)  # type: ignore[no-any-return]


def _persist_seed_artifacts(**kwargs: Any) -> tuple[Path, Path]:
    return _lazy_import("_persist_seed_artifacts")(**kwargs)  # type: ignore[no-any-return]


def _resolve_governance_policy(train_config: Any) -> Any:
    return _lazy_import("_resolve_governance_policy")(train_config)


def _find_latest_ab_raw_metrics(ab_dir: Path, dataset_name: str) -> Path | None:
    return _lazy_import("_find_latest_ab_raw_metrics")(ab_dir, dataset_name)  # type: ignore[no-any-return]


def _load_json_dict(path: Path) -> dict[str, Any]:
    return _lazy_import("_load_json_dict")(path)  # type: ignore[no-any-return]


def evaluate_ab_candidate(**kwargs: Any) -> dict[str, Any]:
    return _lazy_import("evaluate_ab_candidate")(**kwargs)  # type: ignore[no-any-return]


def _run_multiseed_calibrated_governance(**kwargs: Any) -> dict[str, Any]:
    return _lazy_import("_run_multiseed_calibrated_governance")(**kwargs)  # type: ignore[no-any-return]


# ============================================================================
# Multi-seed governance (standalone path)
# ============================================================================


def run_multiseed_governance(
    parsed: ParsedConfig,
    logger: Any,
    results_dir_arg: Path | None = None,
) -> GovernanceResult:
    """Run multi-seed calibrated governance (--multi-seed-governance path).

    Faithfully reproduces the original main() lines 5854–5916.
    """
    args = parsed.args
    results_dir = Path(results_dir_arg or HELIX_FULL_RESULTS_DIR).resolve()

    # Reconstruct the argv forwarding logic from original main()
    script_path = _PROJECT_ROOT / "scripts" / "training" / "train_helix_ids_full.py"
    forwarded: list[str] = []
    skip_flags = {
        "--multi-seed-governance",
        "--no-multi-seed-governance",
        "--multi-seeds",
        "--seed",
        "--epochs",
        "--disable-early-stopping",
        "--calibration-mode",
        "--max-temperature",
    }
    import sys as _sys

    argv_iter = iter(enumerate(_sys.argv[1:]))
    for _idx, token in argv_iter:
        if token in {"--multi-seed-governance", "--no-multi-seed-governance"}:
            continue
        if token in {
            "--multi-seeds",
            "--seed",
            "--epochs",
            "--calibration-mode",
            "--max-temperature",
        }:
            _ = next(argv_iter, None)
            continue
        if token in {"--disable-early-stopping", "--no-disable-early-stopping"}:
            continue
        if token.startswith(
            (
                "--multi-seeds=",
                "--seed=",
                "--epochs=",
                "--calibration-mode=",
                "--max-temperature=",
            )
        ):
            continue
        if token in skip_flags:
            continue
        forwarded.append(token)

    parsed_seeds = [
        int(part.strip())
        for part in str(args.multi_seeds).split(",")
        if str(part).strip()
    ]
    if not parsed_seeds:
        raise ValueError("--multi-seeds must include at least one integer seed")

    governance_report = _run_multiseed_calibrated_governance(
        script_path=script_path,
        argv=forwarded,
        seeds=parsed_seeds,
        max_temperature=float(args.max_temperature),
        class4_recall_floor=0.80,
    )
    report_path = (
        results_dir / "multi_seed_calibrated_governance.json"
    )
    _atomic_write_json(report_path, governance_report)

    return_payload = {
        "results": {"multi_seed_governance": governance_report},
        "governance_stages": {"multi_seed_report_path": str(report_path)},
        "governance_context": {
            "seed": int(parsed_seeds[0]),
            "phase_regime": "multi_seed_calibrated_governance",
        },
        "governance_run_record": {
            "dataset_id": "multi_seed_calibrated_governance",
            "macro_f1": float(
                governance_report.get("governance", {}).get("mean_macro_f1", 0.0)
            ),
        },
        "determinism": {
            "mode": "multi_seed_governance",
            "orchestrator_seed": int(args.seed),
        },
    }

    return GovernanceResult(
        governance_stages={"multi_seed_report_path": str(report_path)},
        governance_context={
            "seed": int(parsed_seeds[0]),
            "phase_regime": "multi_seed_calibrated_governance",
        },
        governance_run_record={
            "dataset_id": "multi_seed_calibrated_governance",
            "macro_f1": float(
                governance_report.get("governance", {}).get("mean_macro_f1", 0.0)
            ),
        },
        determinism={
            "mode": "multi_seed_governance",
            "orchestrator_seed": int(args.seed),
        },
        return_payload=return_payload,
    )


# ============================================================================
# Main governance pipeline (faithful extraction of original main lines 6938–7141)
# ============================================================================


def run_governance_pipeline(
    parsed: ParsedConfig,
    orchestration_result: OrchestrationResult,
    results_dir: Path,
    output_dir: Path,
    logger: Any,
    split_elapsed: float = 0.0,
    splits: dict[str, Any] | None = None,
) -> GovernanceResult:
    """Execute post-training governance pipeline.

    Faithfully reproduces original main() lines 6938–7141.

    The original code was written as a monolithic try/except block. This function
    preserves the same sequence of operations and exception handling.
    """
    args = parsed.args
    train_config = parsed.train_config
    config_payload = parsed.config_payload
    phase_regime = parsed.phase_regime

    per_dataset_results = orchestration_result.per_dataset_results
    ab_raw_current_by_dataset = orchestration_result.ab_raw_current_by_dataset
    dataset_representation_snapshot_ids = (
        orchestration_result.dataset_representation_snapshot_ids
    )
    training_elapsed_total = orchestration_result.training_elapsed_total
    pretrain_elapsed = orchestration_result.pretrain_elapsed
    governance_dataset_id = orchestration_result.governance_dataset_id
    results = dict(orchestration_result.results)

    determinism_state: Any = {"seed": int(args.seed)}

    time.perf_counter()
    # ------------------------------------------------------------------
    # Original main() lines 6938–7101 (governance/A-B evaluation with
    # error handling)
    # ------------------------------------------------------------------
    try:
        posteval_start = time.perf_counter()
        macro_values = [
            float(
                metrics.get(
                    "family_macro_f1", metrics.get("family_f1", 0.0)
                )
            )
            for metrics in per_dataset_results.values()
        ]
        aggregate_macro_f1 = float(min(macro_values)) if macro_values else 0.0

        policy = _resolve_governance_policy(train_config)
        registry = RunRegistry(
            Path(
                os.environ.get(
                    "HELIX_RUN_REGISTRY",
                    "results/gates/run_registry.jsonl",
                )
            )
        )
        drift, z_score = registry.compute_drift(
            dataset_id=governance_dataset_id,
            current_macro_f1=aggregate_macro_f1,
            baseline_window_runs=20,
            phase_regime=phase_regime,
        )

        prepromote_start = time.perf_counter()
        promotion_consensus = aggregate_seed_runs(
            [
                SeedRunSummary(
                    seed=args.seed,
                    macro_f1=aggregate_macro_f1,
                    macro_f1_ci_lower=aggregate_macro_f1,
                    macro_f1_ci_width=0.0,
                    tier2_pass=True,
                )
            ],
            min_seed_runs=policy.promotion.min_seed_runs,
            max_inter_seed_macro_f1_variance=policy.promotion.max_inter_seed_macro_f1_variance,
            reproducibility_tolerance=policy.promotion.reproducibility_tolerance,
            min_ci95_lower_bound=policy.bootstrap.min_ci95_lower_bound,
            max_ci_width=policy.bootstrap.max_ci_width,
        )

        governance_stages: dict[str, Any] = {
            "presplit": {
                "presplit_elapsed_seconds": split_elapsed,
                "split_train_rows": int(
                    sum(
                        int(
                            cast(
                                np.ndarray,
                                (splits or {}).get(
                                    f"X_train_{name}",
                                    np.empty((0, 0)),
                                ),
                            ).shape[0]
                        )
                        for name in ["nsl_kdd", "unsw_nb15", "cicids"]
                    )
                ),
                "split_binary_class_count": 2,
            },
            "pretrain": {
                "pretrain_elapsed_seconds": pretrain_elapsed,
                "family_class_weight_min": 1.0,
                "binary_class_weight_min": 1.0,
            },
            "intrain": {
                "intrain_elapsed_seconds": training_elapsed_total,
                "low_entropy_consecutive_batches": 0,
                "gradient_dominance": 0.0,
                "epochs_without_improvement": 0,
            },
            "posteval": {
                "posteval_elapsed_seconds": max(
                    0.001, time.perf_counter() - posteval_start
                ),
                "macro_f1_ci_width": 0.0,
                "macro_f1_ci_lower": aggregate_macro_f1,
                "dataset_identity_balanced_accuracy": 0.0,
                "abs_macro_f1_drift": drift,
                "abs_macro_f1_zscore": z_score,
                "phase_regime": phase_regime,
            },
            "prepromote": {
                "prepromote_elapsed_seconds": max(
                    0.001, time.perf_counter() - prepromote_start
                ),
                "macro_f1_ci_width": 0.0,
                "macro_f1_ci_lower": aggregate_macro_f1,
                **promotion_consensus.to_stage_metrics(),
            },
        }
        if promotion_consensus.invalid_reason is not None:
            governance_stages["prepromote"][
                "promotion_invalid_reason"
            ] = promotion_consensus.invalid_reason

        ab_raw_artifacts: dict[str, str] = {}
        ab_decisions: dict[str, dict[str, Any]] = {}
        if bool(args.ab_mode):
            ab_dir = results_dir / "ab_runs"
            explicit_baseline_path = (
                Path(args.ab_baseline) if args.ab_baseline else None
            )

            for dataset_name, current_payload in ab_raw_current_by_dataset.items():
                baseline_path: Path | None = None
                if explicit_baseline_path is not None:
                    baseline_path = explicit_baseline_path
                else:
                    baseline_path = _find_latest_ab_raw_metrics(
                        ab_dir, dataset_name
                    )

                decision: dict[str, Any]
                baseline_payload: dict[str, Any] | None = None
                if baseline_path is None:
                    if bool(args.ab_require_baseline):
                        raise RuntimeError(
                            "A/B protocol baseline missing for dataset "
                            f"{dataset_name}; set --ab-baseline or seed baseline artifact first"
                        )
                    decision = {
                        "accepted": True,
                        "reason": "baseline_bootstrap",
                        "tier_1_geometry_pass": True,
                        "tier_2_cluster_quality_pass": True,
                        "tier_3_classifier_pass": True,
                        "tier_4_governance_pass": True,
                        "tier_3_evaluated": True,
                    }
                else:
                    baseline_payload = _load_json_dict(baseline_path)
                    decision = evaluate_ab_candidate(
                        current=current_payload,
                        baseline=baseline_payload,
                        ab_track=str(args.ab_track),
                        governance_z_score=float(z_score),
                        governance_z_tolerance=float(
                            policy.drift.max_abs_z_score
                        ),
                    )

                raw_payload = dict(current_payload)
                raw_payload["baseline_path"] = (
                    str(baseline_path) if baseline_path is not None else None
                )
                raw_payload["decision"] = decision
                if baseline_payload is not None:
                    raw_payload["baseline_metrics"] = {
                        "ratio": float(baseline_payload.get("ratio", 0.0)),
                        "min_inter": float(
                            baseline_payload.get("min_inter", 0.0)
                        ),
                        "macro_f1": float(
                            baseline_payload.get("macro_f1", 0.0)
                        ),
                        "zero_prediction_classes": float(
                            baseline_payload.get(
                                "zero_prediction_classes", 0.0
                            )
                        ),
                    }

                artifact_path = (
                    ab_dir
                    / f"{dataset_name}_ab_raw_{datetime.now().strftime('%Y%m%dT%H%M%S%f')}"
                    f"_seed{int(args.seed)}.json"
                )
                _atomic_write_json(artifact_path, raw_payload)
                ab_raw_artifacts[dataset_name] = str(artifact_path)
                ab_decisions[dataset_name] = decision
                logger.info(
                    "[%s] A/B raw metrics persisted: %s",
                    dataset_name,
                    str(artifact_path),
                )

                if not bool(decision.get("accepted", False)):
                    raise RuntimeError(
                        "A/B protocol reject "
                        f"[{dataset_name}]: {decision.get('reason', 'unknown')}"
                    )

            results["ab_protocol"] = {
                "enabled": True,
                "track": str(args.ab_track),
                "change_id": str(args.ab_change_id),
                "raw_metrics_artifacts": ab_raw_artifacts,
                "decisions": ab_decisions,
            }
    except Exception as exc:
        guard_failure = str(exc)
        _persist_seed_artifacts(
            results_dir=results_dir,
            seed=args.seed,
            config_payload=config_payload,
            results_payload=results,
            eval_payload=per_dataset_results,
            run_exit_code=1,
            guard_failure=guard_failure,
        )
        raise

    # ------------------------------------------------------------------
    # Original main() lines 7103–7141
    # ------------------------------------------------------------------
    run_exit_code = 0
    training_results_path, _ = _persist_seed_artifacts(
        results_dir=results_dir,
        seed=args.seed,
        config_payload=config_payload,
        results_payload=results,
        eval_payload=per_dataset_results,
        run_exit_code=run_exit_code,
        guard_failure=None,
    )

    logger.info(f"Results saved to {training_results_path}")
    logger.info("=" * 80)
    logger.info("Training complete (decoupled mode)!")

    # Build governance_pipeline_stages for the GovernanceResult
    governance_pipeline_stages = governance_stages

    return_payload = {
        "results": results,
        "governance_stages": governance_stages,
        "governance_context": {
            "seed": args.seed,
            "phase_regime": phase_regime,
        },
        "governance_run_record": {
            "dataset_id": governance_dataset_id,
            "macro_f1": aggregate_macro_f1,
            "fingerprint": os.environ.get("HELIX_FINGERPRINT"),
            "parent_run_id": os.environ.get("HELIX_PARENT_RUN_ID"),
            "lineage": {
                "dataset_hashes": os.environ.get(
                    "HELIX_DATASET_HASHES", "unknown"
                ),
                "schema_hash": os.environ.get(
                    "HELIX_SCHEMA_HASH", "unknown"
                ),
                "mapping_version": os.environ.get(
                    "HELIX_MAPPING_VERSION", "unknown"
                ),
                "model_artifact": str(output_dir),
                "metrics_artifact": str(training_results_path),
                "phase_regime": phase_regime,
                "representation_snapshot_ids": dataset_representation_snapshot_ids,
            },
        },
        "determinism": determinism_state.to_dict(),
    }

    return GovernanceResult(
        success=run_exit_code == 0,
        governance_stages=governance_pipeline_stages,
        return_payload=return_payload,
    )
