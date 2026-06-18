"""
Phase 3: Training HelixIDS-Full on multi-dataset unified features.

Trains single unified model on NSL-KDD + UNSW-NB15 + CICIDS combined data.
Uses multi-task learning: binary (Normal vs Attack) + family (7-class).
No QAT—just FP32 training on M4 MPS.

Usage:
    python scripts/train_helix_ids_full.py --config config/helix_config.yaml --output models/helix_full

Output artifacts:
    - models/helix_full_best.pt: Best model checkpoint
    - models/helix_full_final.pt: Final trained model
    - results/helix_full/training_results_seed{seed}.json: Training metrics
    - results/helix_full/eval_results_seed{seed}.json: Per-dataset test evaluation
"""

# ruff: noqa: E402

import hashlib
import json
import logging
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

HELIX_FULL_RESULTS_DIR = Path("results/helix_full")

from helix_ids.config.helix_full_config import TrainingConfig  # noqa: E402
from helix_ids.contracts import (
    runtime_contract_payload,
)
from helix_ids.data.learnability_contract import (
    PREPROCESS_THRESHOLDS,
    assert_contract,
    compute_schema_hash,
    freeze_snapshot_if_valid,
)
from helix_ids.governance import (
    ARTIFACT_MANIFEST_KEY,
    checkpoint_manifest_payload,
    write_contract_sidecars,
)

# Import from helix_ids package
from helix_ids.governance.determinism import (
    reseed_dataloader_generator,
    set_global_determinism,
)
from helix_ids.governance.entrypoint import governed_entrypoint  # noqa: E402
from helix_ids.governance.parameters import DEFAULT_GOVERNANCE_POLICY
from helix_ids.models.full import (  # noqa: F401
    HelixFullConfig,
    HelixIDSFull,
    MultiTaskLoss,
    create_helix_full,
)
from helix_ids.utils.export import (
    build_export_manifest,
    finalize_export_artifact,
    verify_export_artifact,
)
from helix_ids.utils.metrics import (  # noqa: E402
    compute_macro_f1,
)

REQUIRED_GEOMETRY_FEATURE_DIM = 17
MIN_FEATURE_STD = 1e-6
from helix_ids.data.feature_harmonization import ENGINEERED_FEATURE_NAMES  # noqa: E402

# Import core package (Phase 18)
from scripts.training.core import RecoveryManager  # noqa: E402

# Import extracted dataset/sampler components (Phase 12B-3)
from scripts.training.data.dataset_builder import MultiTaskNumpyDataset  # noqa: E402

# ============================================================================
# Import extracted evaluation components (Phase 16)
# ============================================================================
from scripts.training.evaluation import (  # noqa: I001
    EvaluationOrchestrator,  # type: ignore[import-unvalidated]  # noqa: E402
    HelixFullEvaluator,  # type: ignore[import-unvalidated]  # noqa: E402
)

# Import extracted training execution components (Phase 17)
from scripts.training.execution import (  # noqa: I001
    BatchProcessor,  # type: ignore[import-unvalidated]  # noqa: E402
    EpochRunner,  # type: ignore[import-unvalidated]  # noqa: E402
    TrainingOrchestrator,  # type: ignore[import-unvalidated]  # noqa: E402
    WarmupManager,  # type: ignore[import-unvalidated]  # noqa: E402
)

# Import extracted governance components (Phase 12B-7)
# ============================================================================
from scripts.training.governance import (  # noqa: I001
    ab_rejection,
    build_ab_raw_metrics,
    detect_cluster_mode_collapse,
    detect_feature_and_objective_changes,
    load_json_dict,
    load_seed_run_artifacts,
    materialize_phase8_artifacts,
    normalize_calibration_block,
    normalize_metrics_payload,
    normalized_entropy_from_counts,
    summarize_governance,
    validate_ab_contract,
    validate_track,
)
from scripts.training.governance import (  # noqa: I001
    evaluate_ab_candidate as _gov_evaluate_ab_candidate,
)

# Import extracted loss registry and representation components (Phase 13A-3)
from scripts.training.losses import (  # noqa: I001
    LossRegistry,
)
from scripts.training.representation import (  # noqa: I001
    CentroidManager,
    RepresentationCoordinator,
)

# Import extracted scheduler components (Phase 13A-2)
from scripts.training.scheduler import (  # noqa: I001
    EarlyStoppingManager,
    FreezeManager,
    LRScheduler,
    PhaseManager,
    PhaseOrchestrator,
)
from scripts.training.validation import (  # noqa: I001
    ValidationOrchestrator,  # type: ignore[import-unvalidated]  # noqa: E402
)


def _resolve_governance_policy(train_config: TrainingConfig):
    """Return governance policy tuned to training budget.

    Smoke runs (<=10 epochs) validate pipeline integrity, so they use a
    relaxed CI lower-bound gate while preserving strict policy for full runs.
    """
    policy = DEFAULT_GOVERNANCE_POLICY
    if int(getattr(train_config, "epochs", 0)) <= 2:
        return replace(
            policy,
            bootstrap=replace(policy.bootstrap, min_ci95_lower_bound=0.10),
            drift=replace(policy.drift, max_abs_z_score=50.0),
            promotion=replace(policy.promotion, min_seed_runs=1),
        )
    if int(getattr(train_config, "epochs", 0)) <= 10:
        return replace(
            policy,
            bootstrap=replace(policy.bootstrap, min_ci95_lower_bound=0.15),
            drift=replace(policy.drift, max_abs_z_score=50.0),
            promotion=replace(policy.promotion, min_seed_runs=1),
        )
    return policy


def _resolve_class_balance_strategy(balance_strategy_arg: str) -> tuple[str, bool]:
    """Resolve CLI balance strategy aliases into model strategy + class-weight usage.

    Supports explicit learnability aliases:
    - none -> weighted_ce without class weighting
    - sqrt_weighted_ce -> weighted CE with sqrt-inverse class weights
    """
    strategy_raw = str(balance_strategy_arg).strip().lower()
    if strategy_raw == "none":
        return "weighted_ce", False
    if strategy_raw in {"weighted_ce", "focal"}:
        return strategy_raw, True
    if strategy_raw == "sqrt_weighted_ce":
        return "weighted_ce", True
    raise ValueError(
        "--class-balance-strategy must be one of {'none', 'weighted_ce', 'sqrt_weighted_ce', 'focal'}"
    )


def _apply_disable_early_stopping(train_config: TrainingConfig, *, disable_early_stopping: bool) -> None:
    """Disable early stopping by extending patience beyond the full epoch budget."""
    if not disable_early_stopping:
        return
    target_patience = max(int(getattr(train_config, "epochs", 0)) + 1, 10_000)
    train_config.early_stopping_patience = int(target_patience)


def _scan_feature_leakage(
    x_train: np.ndarray,
    y_binary_train: np.ndarray,
    *,
    feature_names: list[str],
    seed: int,
    max_samples: int = 200000,
) -> dict[str, Any]:
    """Detect suspicious single-feature separability against binary target."""
    if x_train.shape[0] == 0:
        return {"max_single_feature_auroc": 0.0, "suspicious_features": []}

    rng = np.random.default_rng(seed)
    sample_n = int(min(max_samples, x_train.shape[0]))
    sample_idx = rng.choice(x_train.shape[0], size=sample_n, replace=False)
    x_sample = x_train[sample_idx]
    y_sample = y_binary_train[sample_idx]

    suspicious_features: list[dict[str, Any]] = []
    max_auc = 0.0

    for feature_idx, feature_name in enumerate(feature_names):
        values = x_sample[:, feature_idx]
        if np.nanstd(values) <= 1e-12:
            continue

        try:
            auc = roc_auc_score(y_sample, values)
        except ValueError:
            continue
        auc = float(max(auc, 1.0 - auc))
        max_auc = max(max_auc, auc)

        if auc >= 0.995:
            suspicious_features.append(
                {
                    "feature": feature_name,
                    "single_feature_auroc": auc,
                }
            )

    suspicious_features.sort(key=lambda item: float(item["single_feature_auroc"]), reverse=True)
    return {
        "max_single_feature_auroc": float(max_auc),
        "suspicious_features": suspicious_features[:10],
    }


def _shuffled_label_sanity_check(
    x_train: np.ndarray,
    y_binary_train: np.ndarray,
    x_val: np.ndarray,
    y_binary_val: np.ndarray,
    *,
    seed: int,
    device: str,
    max_per_class: int = 25000,
    steps: int = 250,
    lr: float = 0.05,
) -> float:
    """Train a tiny probe on shuffled binary labels; high balanced-val accuracy indicates leakage."""
    rng = np.random.default_rng(seed)

    train_pos_idx = np.nonzero(y_binary_train == 1)[0]
    train_neg_idx = np.nonzero(y_binary_train == 0)[0]
    val_pos_idx = np.nonzero(y_binary_val == 1)[0]
    val_neg_idx = np.nonzero(y_binary_val == 0)[0]

    train_take = int(min(max_per_class, train_pos_idx.size, train_neg_idx.size))
    val_take = int(min(max_per_class // 2, val_pos_idx.size, val_neg_idx.size))
    if train_take < 100 or val_take < 50:
        return 0.0

    train_idx = np.concatenate(
        [
            rng.choice(train_pos_idx, size=train_take, replace=False),
            rng.choice(train_neg_idx, size=train_take, replace=False),
        ]
    )
    val_idx = np.concatenate(
        [
            rng.choice(val_pos_idx, size=val_take, replace=False),
            rng.choice(val_neg_idx, size=val_take, replace=False),
        ]
    )
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    x_train_bal = torch.from_numpy(x_train[train_idx]).float().to(device)
    y_train_bal_np = y_binary_train[train_idx].astype(np.int64, copy=False)
    x_val_bal = torch.from_numpy(x_val[val_idx]).float().to(device)
    y_val_bal = torch.from_numpy(y_binary_val[val_idx].astype(np.int64)).long().to(device)

    shuffled_labels_np = y_train_bal_np[rng.permutation(y_train_bal_np.shape[0])]
    shuffled_labels = torch.from_numpy(shuffled_labels_np).long().to(device)

    probe = nn.Linear(x_train_bal.shape[1], 2).to(device)
    optimizer = optim.SGD(probe.parameters(), lr=lr, momentum=0.0, weight_decay=0.0)
    loss_fn = nn.CrossEntropyLoss()

    probe.train()
    for _ in range(steps):
        logits = probe(x_train_bal)
        loss = loss_fn(logits, shuffled_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    probe.eval()
    with torch.no_grad():
        preds = torch.argmax(probe(x_val_bal), dim=1)
        balanced_val_acc = float((preds == y_val_bal).float().mean().item())

    return balanced_val_acc


def _feature_ablation_sanity_check(
    x_train: np.ndarray,
    y_binary_train: np.ndarray,
    x_val: np.ndarray,
    y_binary_val: np.ndarray,
    *,
    seed: int,
    device: str,
    steps: int = 200,
    lr: float = 0.05,
) -> dict[str, float]:
    """Train linear probes with/without top proxy feature; report expected accuracy drop."""
    if x_train.shape[0] == 0 or x_val.shape[0] == 0:
        return {
            "baseline_balanced_val_acc": 0.0,
            "ablated_balanced_val_acc": 0.0,
            "accuracy_drop": 0.0,
            "ablated_feature_idx": -1.0,
        }

    rng = np.random.default_rng(seed)

    def _balanced_indices(y: np.ndarray, max_per_class: int) -> np.ndarray:
        idx_pos = np.nonzero(y == 1)[0]
        idx_neg = np.nonzero(y == 0)[0]
        take = int(min(max_per_class, idx_pos.size, idx_neg.size))
        if take < 50:
            return np.array([], dtype=np.int64)
        idx = np.concatenate(
            [
                rng.choice(idx_pos, size=take, replace=False),
                rng.choice(idx_neg, size=take, replace=False),
            ]
        )
        rng.shuffle(idx)
        return idx

    train_idx = _balanced_indices(y_binary_train, max_per_class=20000)
    val_idx = _balanced_indices(y_binary_val, max_per_class=10000)
    if train_idx.size == 0 or val_idx.size == 0:
        return {
            "baseline_balanced_val_acc": 0.0,
            "ablated_balanced_val_acc": 0.0,
            "accuracy_drop": 0.0,
            "ablated_feature_idx": -1.0,
        }

    x_train_bal = x_train[train_idx]
    y_train_bal = y_binary_train[train_idx].astype(np.int64, copy=False)
    x_val_bal = x_val[val_idx]
    y_val_bal = y_binary_val[val_idx].astype(np.int64, copy=False)

    # Identify top suspicious feature via absolute linear correlation to binary target.
    y_centered = y_train_bal.astype(np.float32) - float(np.mean(y_train_bal))
    correlations = []
    for feature_idx in range(x_train_bal.shape[1]):
        feature = x_train_bal[:, feature_idx]
        denom = float(np.std(feature) * np.std(y_centered))
        if denom <= 1e-12:
            correlations.append(0.0)
            continue
        corr = float(np.mean((feature - np.mean(feature)) * y_centered) / denom)
        correlations.append(abs(corr))

    top_feature_idx = int(np.argmax(np.asarray(correlations)))
    feature_fill = float(np.nanmean(x_train_bal[:, top_feature_idx]))

    def _probe_accuracy(x_train_probe: np.ndarray, x_val_probe: np.ndarray) -> float:
        x_train_t = torch.from_numpy(x_train_probe).float().to(device)
        y_train_t = torch.from_numpy(y_train_bal).long().to(device)
        x_val_t = torch.from_numpy(x_val_probe).float().to(device)
        y_val_t = torch.from_numpy(y_val_bal).long().to(device)

        probe = nn.Linear(x_train_probe.shape[1], 2).to(device)
        optimizer = optim.SGD(probe.parameters(), lr=lr, momentum=0.0, weight_decay=0.0)
        loss_fn = nn.CrossEntropyLoss()

        probe.train()
        for _ in range(steps):
            logits = probe(x_train_t)
            loss = loss_fn(logits, y_train_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        probe.eval()
        with torch.no_grad():
            preds = torch.argmax(probe(x_val_t), dim=1)
            return float((preds == y_val_t).float().mean().item())

    baseline_acc = _probe_accuracy(x_train_bal, x_val_bal)

    x_train_ablated = x_train_bal.copy()
    x_val_ablated = x_val_bal.copy()
    x_train_ablated[:, top_feature_idx] = feature_fill
    x_val_ablated[:, top_feature_idx] = feature_fill
    ablated_acc = _probe_accuracy(x_train_ablated, x_val_ablated)

    return {
        "baseline_balanced_val_acc": baseline_acc,
        "ablated_balanced_val_acc": ablated_acc,
        "accuracy_drop": baseline_acc - ablated_acc,
        "ablated_feature_idx": float(top_feature_idx),
    }


def _assert_validated_unsw_artifact(
    *,
    splits_dir: Path,
    logger: logging.Logger,
    require_frozen: bool,
) -> dict[str, Any]:
    """Assert UNSW learnability contract is present, validated, and schema-consistent."""
    feature_columns_path = splits_dir / "feature_columns.npy"
    if not feature_columns_path.exists():
        raise RuntimeError(
            "Missing feature_columns.npy in processed artifact; validation contract cannot be checked"
        )

    feature_columns = np.load(feature_columns_path, allow_pickle=True).astype(str).tolist()
    expected_schema_hash = compute_schema_hash(
        feature_columns=feature_columns,
        transformations=["split_then_nan_to_num"],
    )
    meta = assert_contract(
        artifact_dir=splits_dir,
        expected_schema_hash=expected_schema_hash,
        require_frozen=bool(require_frozen),
        thresholds=PREPROCESS_THRESHOLDS,
    )
    logger.info(
        "Validated UNSW artifact contract: macro_f1=%.4f unique_coverage=%.4f frozen=%s",
        float(meta.get("linear_probe_macro_f1", 0.0)),
        float(meta.get("unique_pred_coverage", 0.0)),
        bool(meta.get("frozen", False)),
    )
    return cast(dict[str, Any], meta)


def _assert_real_dataset_required(*, project_root: Path, dataset_name: str) -> None:
    """Hard-stop when requested geometry validation dataset is unavailable.

    Geometry diagnostics must never proceed from mock/synthetic-only artifacts.
    """
    dataset_key = str(dataset_name).strip().lower()
    data_dir = project_root / "data"
    train_csv = "train.csv"

    if dataset_key == "nsl_kdd":
        candidates = [
            data_dir / "nsl_kdd" / "raw" / "KDDTrain+.txt",
            data_dir / "nsl_kdd" / train_csv,
            data_dir / "nsl_kdd_5class" / train_csv,
        ]
        dataset_exists = any(path.exists() for path in candidates)
    elif dataset_key == "unsw_nb15":
        candidates = [
            data_dir / "unsw_nb15" / "raw" / "UNSW_NB15_training-set.csv",
            data_dir / "unsw_nb15" / train_csv,
        ]
        dataset_exists = any(path.exists() for path in candidates)
    elif dataset_key == "cicids":
        raw_dir = data_dir / "cicids2018" / "raw"
        dataset_exists = raw_dir.exists() and any(raw_dir.glob("*.csv"))
    else:
        raise ValueError(f"Unsupported dataset for geometry validation: {dataset_name}")

    if not dataset_exists:
        raise RuntimeError(
            "Real dataset required for geometry validation"
        )


def _build_model_contract_artifact(
    *,
    model_state: dict[str, Any],
    feature_order: list[str],
    schema_hash: str,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    contract = runtime_contract_payload()
    contract["feature_order"] = list(feature_order)
    contract["schema_hash"] = schema_hash
    artifact = {
        "model": {
            key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
            for key, value in model_state.items()
        },
        "model_state_dict": {
            key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
            for key, value in model_state.items()
        },
    }
    artifact.update(contract)
    if extra:
        artifact.update(extra)
    return artifact


def _write_model_contract_sidecars(path: Path, artifact: dict[str, Any]) -> None:
    contract = {
        "schema_version": artifact["schema_version"],
        "schema_hash": artifact["schema_hash"],
        "feature_order": artifact["feature_order"],
        "input_dim": artifact["input_dim"],
        "binary_output_dim": artifact["binary_output_dim"],
        "family_output_dim": artifact["family_output_dim"],
    }
    _atomic_write_json(path.with_suffix(path.suffix + ".contract.json"), contract)
    _atomic_write_json(path.with_suffix(path.suffix + ".feature_order.json"), list(artifact["feature_order"]))
    path.with_suffix(path.suffix + ".schema_hash.txt").write_text(str(artifact["schema_hash"]) + "\n", encoding="utf-8")


def _write_checkpoint_artifact(
    path: Path,
    artifact: dict[str, Any],
    *,
    model_architecture: str,
    origin: str,
) -> None:
    contract = {
        "schema_version": artifact["schema_version"],
        "schema_hash": artifact["schema_hash"],
        "feature_order": artifact["feature_order"],
        "input_dim": artifact["input_dim"],
        "binary_output_dim": artifact["binary_output_dim"],
        "family_output_dim": artifact["family_output_dim"],
        "contract_version": artifact.get("contract_version"),
        "feature_order_hash": artifact.get("feature_order_hash"),
    }
    manifest_base = build_export_manifest(
        contract=contract,
        model_architecture=model_architecture,
        export_config={"format": "checkpoint", "origin": origin},
    )
    payload = dict(artifact)
    payload[ARTIFACT_MANIFEST_KEY] = checkpoint_manifest_payload(manifest_base)
    torch.save(payload, path)
    sidecars = write_contract_sidecars(path, contract)
    finalize_export_artifact(path, manifest_base, sidecars=sidecars)
    verify_export_artifact(
        path,
        kind="checkpoint",
        contract=contract,
        embedded_manifest=checkpoint_manifest_payload(manifest_base),
    )


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON atomically via temporary file and replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    with open(tmp_path, "w") as handle:
        json.dump(payload, handle, indent=2, default=str)
    os.replace(tmp_path, path)


def _sha256_file(path: Path) -> str:
    """Compute streaming SHA-256 digest for a file."""
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()


def _write_isolation_snapshot_descriptor(
    *,
    dataset_name: str,
    splits_dir: Path,
    seed: int,
    batch_size: int,
    class_counts: dict[int, int],
    class_multipliers: dict[int, float],
    sampler_indices: np.ndarray,
    val_subset_indices: np.ndarray,
    results_dir: Path,
    snapshot_mode: str,
) -> dict[str, Any]:
    """Persist immutable isolation snapshot descriptor and return metadata."""
    required_files = [
        splits_dir / f"X_train_{dataset_name}.npy",
        splits_dir / f"y_train_{dataset_name}.npy",
        splits_dir / f"X_val_{dataset_name}.npy",
        splits_dir / f"y_val_{dataset_name}.npy",
    ]
    file_hashes = {
        file_path.name: _sha256_file(file_path)
        for file_path in required_files
        if file_path.exists()
    }
    if len(file_hashes) != len(required_files):
        missing = [str(path) for path in required_files if not path.exists()]
        raise RuntimeError(
            "Cannot build isolation snapshot descriptor; missing files: "
            f"{missing}"
        )

    sampler_signature = {
        "batch_size": int(batch_size),
        "class_counts": {int(k): int(v) for k, v in class_counts.items()},
        "class_multipliers": {int(k): float(v) for k, v in class_multipliers.items()},
        "index_count": int(sampler_indices.shape[0]),
        "index_hash": hashlib.sha256(np.asarray(sampler_indices, dtype=np.int64).tobytes()).hexdigest(),
    }
    validation_signature = {
        "target_per_class": 50,
        "index_count": int(val_subset_indices.shape[0]),
        "index_hash": hashlib.sha256(np.asarray(val_subset_indices, dtype=np.int64).tobytes()).hexdigest(),
    }

    descriptor_payload = {
        "dataset": dataset_name,
        "seed": int(seed),
        "snapshot_mode": str(snapshot_mode),
        "file_hashes": file_hashes,
        "sampler_signature": sampler_signature,
        "validation_signature": validation_signature,
    }
    snapshot_hash = hashlib.sha256(
        json.dumps(descriptor_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    snapshot_id = f"{dataset_name}_isolation_v1_{snapshot_hash[:16]}"
    descriptor_payload["snapshot_id"] = snapshot_id

    snapshot_dir = results_dir / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_dir / f"{dataset_name}_isolation_snapshot_seed{seed}.json"
    _atomic_write_json(snapshot_path, descriptor_payload)
    return {
        "snapshot_id": snapshot_id,
        "snapshot_path": str(snapshot_path),
    }


def _persist_seed_artifacts(
    *,
    results_dir: Path,
    seed: int,
    config_payload: dict[str, Any],
    results_payload: dict[str, Any],
    eval_payload: dict[str, Any],
    run_exit_code: int,
    guard_failure: Optional[str],
) -> tuple[Path, Path]:
    """Persist seed-scoped training/eval artifacts atomically."""
    timestamp = datetime.now().isoformat()
    training_path = results_dir / f"training_results_seed{seed}.json"
    eval_path = results_dir / f"eval_results_seed{seed}.json"

    _atomic_write_json(
        training_path,
        {
            "timestamp": timestamp,
            "run_exit_code": int(run_exit_code),
            "guard_failure": guard_failure,
            "config": config_payload,
            "results": results_payload,
        },
    )
    _atomic_write_json(
        eval_path,
        {
            "timestamp": timestamp,
            "run_exit_code": int(run_exit_code),
            "guard_failure": guard_failure,
            "results": eval_payload,
        },
    )
    return training_path, eval_path


def _stable_feature_signature(*, feature_order: list[str], schema_hash: str) -> str:
    """Return deterministic signature for feature-space contract."""
    payload = {
        "feature_order": [str(name) for name in feature_order],
        "schema_hash": str(schema_hash),
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalized_entropy_from_counts(counts: list[int]) -> float:
    """Compute normalized entropy in [0, 1] for cluster-size distribution."""
    return normalized_entropy_from_counts(counts)


def _detect_cluster_mode_collapse(
    cluster_sizes: list[int],
    *,
    min_entropy: float = 0.30,
    max_dominance: float = 0.85,
) -> tuple[bool, dict[str, float]]:
    """Detect cluster mode collapse using entropy and dominant-cluster share."""
    return detect_cluster_mode_collapse(
        cluster_sizes,
        min_entropy=min_entropy,
        max_dominance=max_dominance,
    )


def _load_json_dict(path: Path) -> dict[str, Any]:
    """Load JSON object from path and validate dictionary payload."""
    return load_json_dict(path)


def _find_latest_ab_raw_metrics(ab_dir: Path, dataset_name: str) -> Optional[Path]:
    """Return the latest raw A/B metrics artifact for a dataset, if present."""
    candidates = sorted(
        ab_dir.glob(f"{dataset_name}_ab_raw_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _collect_eval_family_outputs(
    *,
    model: HelixIDSFull,
    loader: DataLoader,
    device: str,
    active_class_ids: Optional[set[int]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect family-label evaluation outputs (labels, logits, probs) for calibration."""
    model.eval()
    labels_chunks: list[np.ndarray] = []
    logits_chunks: list[np.ndarray] = []

    with torch.no_grad():
        for x, _y_binary, y_family in loader:
            x_dev = x.to(device, non_blocking=True)
            _binary_logits, family_logits_dev = model(x_dev)
            family_logits = family_logits_dev.detach().to(device="cpu")

            if active_class_ids:
                allowed = [
                    int(cls)
                    for cls in sorted(active_class_ids)
                    if 0 <= int(cls) < int(family_logits.shape[1])
                ]
                if allowed:
                    mask = torch.full_like(family_logits, float("-inf"))
                    mask[:, allowed] = family_logits[:, allowed]
                    family_logits = mask

            logits_chunks.append(family_logits.numpy().astype(np.float64, copy=False))
            labels_chunks.append(
                y_family.to(device="cpu", dtype=torch.long, non_blocking=True)
                .numpy()
                .astype(np.int64, copy=False)
            )

    if not logits_chunks:
        return (
            np.array([], dtype=np.int64),
            np.empty((0, 0), dtype=np.float64),
            np.empty((0, 0), dtype=np.float64),
        )

    labels = np.concatenate(labels_chunks, axis=0).astype(np.int64, copy=False)
    logits = np.concatenate(logits_chunks, axis=0).astype(np.float64, copy=False)
    logits = np.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)

    logits_shift = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits_shift)
    probs = exp_logits / np.clip(np.sum(exp_logits, axis=1, keepdims=True), 1e-12, None)
    return labels, logits, probs


def _normalized_entropy_from_probs(probs: np.ndarray) -> float:
    """Compute mean normalized entropy in [0, 1] from class probabilities."""
    if probs.ndim != 2 or probs.shape[0] == 0:
        return 0.0
    safe = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    ent = -np.sum(safe * np.log(safe), axis=1)
    class_count = int(safe.shape[1])
    if class_count <= 1:
        return 0.0
    return float(np.mean(ent / math.log(float(class_count))))


def _fit_temperature_nll(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    max_temperature: float,
) -> tuple[float, float]:
    """Fit global temperature by minimizing NLL over a deterministic grid."""
    if logits.ndim != 2 or logits.shape[0] == 0:
        return 1.0, 0.0

    y = np.asarray(labels, dtype=np.int64)
    y = np.clip(y, 0, int(logits.shape[1]) - 1)

    min_t = 1.0
    max_t = max(min_t, float(max_temperature))
    grid = np.unique(
        np.concatenate(
            [
                np.linspace(min_t, min(2.0, max_t), 151),
                np.linspace(min_t, max_t, 301),
            ]
        )
    )

    best_t = 1.0
    best_nll = float("inf")
    for t in grid:
        scaled = np.asarray(logits, dtype=np.float64) / max(1e-6, float(t))
        shifted = scaled - np.max(scaled, axis=1, keepdims=True)
        log_probs = shifted - np.log(np.clip(np.sum(np.exp(shifted), axis=1, keepdims=True), 1e-12, None))
        nll = float(-np.mean(log_probs[np.arange(log_probs.shape[0]), y]))
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)

    return float(best_t), float(best_nll)


def _apply_class4_logit_shift(
    logits: np.ndarray,
    *,
    class4_id: int,
    delta: float,
) -> np.ndarray:
    """Subtract a fixed delta from class-4 logits (inference/eval only)."""
    arr = np.asarray(logits, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return cast(np.ndarray, arr.copy())
    class_idx = int(class4_id)
    if class_idx < 0 or class_idx >= int(arr.shape[1]):
        return cast(np.ndarray, arr.copy())
    shift = float(delta)
    if abs(shift) <= 0.0:
        return cast(np.ndarray, arr.copy())
    out = arr.copy()
    out[:, class_idx] = out[:, class_idx] - shift
    return cast(np.ndarray, out)


def _predict_with_class4_threshold(
    probs: np.ndarray,
    *,
    class4_id: int,
    threshold: float,
) -> np.ndarray:
    """Apply class-4 gating: predict 4 when P4>=tau else argmax over other classes."""
    if probs.ndim != 2 or probs.shape[0] == 0:
        return np.array([], dtype=np.int64)

    pred = np.argmax(probs, axis=1).astype(np.int64, copy=False)
    if class4_id < 0 or class4_id >= int(probs.shape[1]):
        return cast(np.ndarray, pred)

    p4 = probs[:, class4_id]
    choose4 = p4 >= float(threshold)

    others = np.asarray(probs, dtype=np.float64).copy()
    others[:, class4_id] = -np.inf
    fallback = np.argmax(others, axis=1).astype(np.int64, copy=False)

    out = fallback.copy()
    out[choose4] = int(class4_id)
    return cast(np.ndarray, out)


def _compute_multiclass_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    class_count: int,
) -> np.ndarray:
    """Compute multiclass confusion matrix with fixed class space."""
    conf = np.zeros((int(class_count), int(class_count)), dtype=np.int64)
    if y_true.size == 0:
        return cast(np.ndarray, conf)
    y_t = np.clip(np.asarray(y_true, dtype=np.int64), 0, int(class_count) - 1)
    y_p = np.clip(np.asarray(y_pred, dtype=np.int64), 0, int(class_count) - 1)
    np.add.at(conf, (y_t, y_p), 1)
    return cast(np.ndarray, conf)


def _compute_class4_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    class4_id: int,
) -> dict[str, float]:
    """Compute class-4 precision/recall from predicted labels."""
    if y_true.size == 0:
        return {"class4_precision": 0.0, "class4_recall": 0.0}
    pos_true = np.asarray(y_true, dtype=np.int64) == int(class4_id)
    pos_pred = np.asarray(y_pred, dtype=np.int64) == int(class4_id)
    tp = float(np.sum(pos_true & pos_pred))
    fp = float(np.sum((~pos_true) & pos_pred))
    fn = float(np.sum(pos_true & (~pos_pred)))
    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    return {"class4_precision": float(precision), "class4_recall": float(recall)}


def _summarize_prediction_coverage(y_true: np.ndarray, y_pred: np.ndarray) -> int:
    """Count classes present in truth but never predicted."""
    if y_true.size == 0:
        return 0
    present = {int(v) for v in np.unique(y_true).tolist()}
    predicted = {int(v) for v in np.unique(y_pred).tolist()}
    return int(len(sorted(present - predicted)))


def _calibrate_family_predictions(
    *,
    model: HelixIDSFull,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
    class4_id: int = 4,
    max_temperature: float = 5.0,
    threshold_grid: Optional[np.ndarray] = None,
    min_class4_recall: float = 0.80,
    class4_logit_shift: float = 0.0,
) -> dict[str, Any]:
    """Temperature-scale logits and tune class-4 threshold on validation set."""
    if threshold_grid is None:
        threshold_grid = np.linspace(0.3, 0.95, 66)

    y_val, val_logits, val_probs_uncal = _collect_eval_family_outputs(
        model=model,
        loader=val_loader,
        device=device,
    )
    y_test, test_logits, test_probs_uncal = _collect_eval_family_outputs(
        model=model,
        loader=test_loader,
        device=device,
    )

    if y_val.size == 0 or y_test.size == 0:
        return {
            "class4_logit_shift": float(class4_logit_shift),
            "temperature": 1.0,
            "tau_4": 0.5,
            "uncalibrated": {
                "val_argmax": {
                    "class4_precision": 0.0,
                    "class4_recall": 0.0,
                    "macro_f1": 0.0,
                    "zero_prediction_classes": 0,
                    "mean_entropy": 0.0,
                    "confusion_matrix": [],
                },
                "test_argmax": {
                    "class4_precision": 0.0,
                    "class4_recall": 0.0,
                    "macro_f1": 0.0,
                    "zero_prediction_classes": 0,
                    "mean_entropy": 0.0,
                    "confusion_matrix": [],
                },
            },
            "val": {
                "class4_precision": 0.0,
                "class4_recall": 0.0,
                "macro_f1": 0.0,
                "zero_prediction_classes": 0,
                "mean_entropy": 0.0,
            },
            "test": {
                "class4_precision": 0.0,
                "class4_recall": 0.0,
                "macro_f1": 0.0,
                "zero_prediction_classes": 0,
                "mean_entropy": 0.0,
                "confusion_matrix": [],
            },
            "ablation": {
                "without_thresholding": {},
                "without_temperature_scaling": {},
            },
            "pr_curve_class4": {
                "precision": [],
                "recall": [],
                "thresholds": [],
            },
            "threshold_sweep": {
                "tau_min": float(np.min(threshold_grid)),
                "tau_max": float(np.max(threshold_grid)),
                "num_points": int(np.asarray(threshold_grid).size),
                "points": [],
            },
        }

    val_logits = _apply_class4_logit_shift(
        val_logits,
        class4_id=int(class4_id),
        delta=float(class4_logit_shift),
    )
    test_logits = _apply_class4_logit_shift(
        test_logits,
        class4_id=int(class4_id),
        delta=float(class4_logit_shift),
    )

    class_count = int(test_logits.shape[1])

    best_t, _best_nll = _fit_temperature_nll(
        logits=val_logits,
        labels=y_val,
        max_temperature=float(max_temperature),
    )

    def _softmax_with_temperature(logits: np.ndarray, t: float) -> np.ndarray:
        scaled = np.asarray(logits, dtype=np.float64) / max(1e-6, float(t))
        shifted = scaled - np.max(scaled, axis=1, keepdims=True)
        exp_vals = np.exp(shifted)
        return cast(np.ndarray, exp_vals / np.clip(np.sum(exp_vals, axis=1, keepdims=True), 1e-12, None))

    val_probs_uncal = _softmax_with_temperature(val_logits, 1.0)
    test_probs_uncal = _softmax_with_temperature(test_logits, 1.0)
    val_probs_cal = _softmax_with_temperature(val_logits, best_t)
    test_probs_cal = _softmax_with_temperature(test_logits, best_t)

    val_pred_uncal = np.argmax(val_probs_uncal, axis=1).astype(np.int64, copy=False)
    test_pred_uncal = np.argmax(test_probs_uncal, axis=1).astype(np.int64, copy=False)
    val_uncal_c4 = _compute_class4_metrics(y_val, val_pred_uncal, class4_id=int(class4_id))
    test_uncal_c4 = _compute_class4_metrics(y_test, test_pred_uncal, class4_id=int(class4_id))
    val_uncal_macro = float(compute_macro_f1(y_val, val_pred_uncal))
    test_uncal_macro = float(compute_macro_f1(y_test, test_pred_uncal))
    val_uncal_zero = _summarize_prediction_coverage(y_val, val_pred_uncal)
    test_uncal_zero = _summarize_prediction_coverage(y_test, test_pred_uncal)
    val_uncal_entropy = _normalized_entropy_from_probs(val_probs_uncal)
    test_uncal_entropy = _normalized_entropy_from_probs(test_probs_uncal)
    val_uncal_conf = _compute_multiclass_confusion(y_val, val_pred_uncal, class_count=class_count)
    test_uncal_conf = _compute_multiclass_confusion(y_test, test_pred_uncal, class_count=class_count)

    # Sweep tau_4 on validation: maximize class4 precision while preserving recall.
    best_tau = 0.5
    best_precision = -1.0
    best_macro = -1.0
    best_recall = -1.0
    feasible_tau_found = False
    sweep_points: list[dict[str, float]] = []
    for tau in np.asarray(threshold_grid, dtype=np.float64):
        val_pred_tau = _predict_with_class4_threshold(
            val_probs_cal,
            class4_id=int(class4_id),
            threshold=float(tau),
        )
        test_pred_tau = _predict_with_class4_threshold(
            test_probs_cal,
            class4_id=int(class4_id),
            threshold=float(tau),
        )
        val_c4 = _compute_class4_metrics(y_val, val_pred_tau, class4_id=int(class4_id))
        test_c4_tau = _compute_class4_metrics(y_test, test_pred_tau, class4_id=int(class4_id))
        val_macro = float(compute_macro_f1(y_val, val_pred_tau))
        test_macro_tau = float(compute_macro_f1(y_test, test_pred_tau))
        precision = float(val_c4["class4_precision"])
        recall = float(val_c4["class4_recall"])
        test_precision_tau = float(test_c4_tau["class4_precision"])
        test_recall_tau = float(test_c4_tau["class4_recall"])

        sweep_points.append(
            {
                "tau_4": float(tau),
                "val_class4_precision": float(precision),
                "val_class4_recall": float(recall),
                "val_macro_f1": float(val_macro),
                "test_class4_precision": float(test_precision_tau),
                "test_class4_recall": float(test_recall_tau),
                "test_macro_f1": float(test_macro_tau),
            }
        )

        recall_ok = recall >= float(min_class4_recall)
        if not recall_ok:
            continue
        feasible_tau_found = True
        candidate = (
            (precision, val_macro, recall, -float(tau)),
            float(tau),
        )
        incumbent = (
            (best_precision, best_macro, best_recall, -best_tau),
            best_tau,
        )
        if candidate[0] > incumbent[0]:
            best_tau = float(tau)
            best_precision = precision
            best_macro = val_macro
            best_recall = recall

    # Evaluate selected setting on val + test.
    if not feasible_tau_found:
        best_tau = float(np.min(np.asarray(threshold_grid, dtype=np.float64)))
    val_pred = _predict_with_class4_threshold(val_probs_cal, class4_id=int(class4_id), threshold=best_tau)
    test_pred = _predict_with_class4_threshold(test_probs_cal, class4_id=int(class4_id), threshold=best_tau)

    val_c4 = _compute_class4_metrics(y_val, val_pred, class4_id=int(class4_id))
    test_c4 = _compute_class4_metrics(y_test, test_pred, class4_id=int(class4_id))

    val_macro = float(compute_macro_f1(y_val, val_pred))
    test_macro = float(compute_macro_f1(y_test, test_pred))
    val_zero = _summarize_prediction_coverage(y_val, val_pred)
    test_zero = _summarize_prediction_coverage(y_test, test_pred)
    val_entropy = _normalized_entropy_from_probs(val_probs_cal)
    test_entropy = _normalized_entropy_from_probs(test_probs_cal)

    conf_test = _compute_multiclass_confusion(y_test, test_pred, class_count=class_count)

    # Ablations.
    test_pred_temp_only = np.argmax(test_probs_cal, axis=1).astype(np.int64, copy=False)
    test_pred_thresh_only = _predict_with_class4_threshold(
        test_probs_uncal,
        class4_id=int(class4_id),
        threshold=best_tau,
    )

    temp_only_c4 = _compute_class4_metrics(y_test, test_pred_temp_only, class4_id=int(class4_id))
    thresh_only_c4 = _compute_class4_metrics(y_test, test_pred_thresh_only, class4_id=int(class4_id))
    conf_temp_only = _compute_multiclass_confusion(y_test, test_pred_temp_only, class_count=class_count)
    conf_thresh_only = _compute_multiclass_confusion(y_test, test_pred_thresh_only, class_count=class_count)

    # PR curve on class-4 from calibrated test probs.
    y_true_bin = (y_test == int(class4_id)).astype(np.int64)
    if int(np.unique(y_true_bin).size) > 1:
        pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_true_bin, test_probs_cal[:, int(class4_id)])
        pr_precision_list = [float(v) for v in pr_precision.tolist()]
        pr_recall_list = [float(v) for v in pr_recall.tolist()]
        pr_threshold_list = [float(v) for v in pr_thresholds.tolist()]
    else:
        pr_precision_list = []
        pr_recall_list = []
        pr_threshold_list = []

    return {
        "class4_logit_shift": float(class4_logit_shift),
        "temperature": float(best_t),
        "tau_4": float(best_tau),
        "uncalibrated": {
            "val_argmax": {
                "class4_precision": float(val_uncal_c4["class4_precision"]),
                "class4_recall": float(val_uncal_c4["class4_recall"]),
                "macro_f1": float(val_uncal_macro),
                "zero_prediction_classes": int(val_uncal_zero),
                "mean_entropy": float(val_uncal_entropy),
                "confusion_matrix": val_uncal_conf.tolist(),
            },
            "test_argmax": {
                "class4_precision": float(test_uncal_c4["class4_precision"]),
                "class4_recall": float(test_uncal_c4["class4_recall"]),
                "macro_f1": float(test_uncal_macro),
                "zero_prediction_classes": int(test_uncal_zero),
                "mean_entropy": float(test_uncal_entropy),
                "confusion_matrix": test_uncal_conf.tolist(),
            },
        },
        "val": {
            "class4_precision": float(val_c4["class4_precision"]),
            "class4_recall": float(val_c4["class4_recall"]),
            "macro_f1": float(val_macro),
            "zero_prediction_classes": int(val_zero),
            "mean_entropy": float(val_entropy),
        },
        "test": {
            "class4_precision": float(test_c4["class4_precision"]),
            "class4_recall": float(test_c4["class4_recall"]),
            "macro_f1": float(test_macro),
            "zero_prediction_classes": int(test_zero),
            "mean_entropy": float(test_entropy),
            "confusion_matrix": conf_test.tolist(),
        },
        "ablation": {
            "without_thresholding": {
                "class4_precision": float(temp_only_c4["class4_precision"]),
                "class4_recall": float(temp_only_c4["class4_recall"]),
                "macro_f1": float(compute_macro_f1(y_test, test_pred_temp_only)),
                "mean_entropy": float(_normalized_entropy_from_probs(test_probs_cal)),
                "confusion_matrix": conf_temp_only.tolist(),
            },
            "without_temperature_scaling": {
                "class4_precision": float(thresh_only_c4["class4_precision"]),
                "class4_recall": float(thresh_only_c4["class4_recall"]),
                "macro_f1": float(compute_macro_f1(y_test, test_pred_thresh_only)),
                "mean_entropy": float(_normalized_entropy_from_probs(test_probs_uncal)),
                "confusion_matrix": conf_thresh_only.tolist(),
            },
        },
        "pr_curve_class4": {
            "precision": pr_precision_list,
            "recall": pr_recall_list,
            "thresholds": pr_threshold_list,
        },
        "threshold_sweep": {
            "tau_min": float(np.min(threshold_grid)),
            "tau_max": float(np.max(threshold_grid)),
            "num_points": int(np.asarray(threshold_grid).size),
            "points": sweep_points,
        },
    }


def _emit_calibration_artifacts(
    *,
    results_dir: Path,
    dataset_name: str,
    seed: int,
    calibration_payload: dict[str, Any],
    artifact_tag: Optional[str] = None,
) -> dict[str, str]:
    """Persist paper-oriented calibration artifacts for one dataset/seed."""
    calibration_dir = results_dir / "calibration"
    calibration_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_seed{int(seed)}"
    if artifact_tag:
        suffix = f"_{str(artifact_tag)}{suffix}"

    calibration_json_path = calibration_dir / f"{dataset_name}_calibration{suffix}.json"
    _atomic_write_json(calibration_json_path, calibration_payload)

    uncal_test = cast(dict[str, Any], calibration_payload.get("uncalibrated", {}).get("test_argmax", {}))
    threshold_only = cast(
        dict[str, Any],
        calibration_payload.get("ablation", {}).get("without_temperature_scaling", {}),
    )
    calibrated = cast(dict[str, Any], calibration_payload.get("test", {}))

    before_after_rows = [
        {
            "phase": "baseline_collapse",
            "macro_f1": float(uncal_test.get("macro_f1", 0.0)),
            "class4_precision": float(uncal_test.get("class4_precision", 0.0)),
            "class4_recall": float(uncal_test.get("class4_recall", 0.0)),
            "zero_prediction_classes": int(uncal_test.get("zero_prediction_classes", 0)),
            "mean_entropy": float(uncal_test.get("mean_entropy", 0.0)),
        },
        {
            "phase": "enforcement_high_recall_low_precision",
            "macro_f1": float(threshold_only.get("macro_f1", 0.0)),
            "class4_precision": float(threshold_only.get("class4_precision", 0.0)),
            "class4_recall": float(threshold_only.get("class4_recall", 0.0)),
            "zero_prediction_classes": int(uncal_test.get("zero_prediction_classes", 0)),
            "mean_entropy": float(threshold_only.get("mean_entropy", 0.0)),
        },
        {
            "phase": "calibrated_balanced",
            "macro_f1": float(calibrated.get("macro_f1", 0.0)),
            "class4_precision": float(calibrated.get("class4_precision", 0.0)),
            "class4_recall": float(calibrated.get("class4_recall", 0.0)),
            "zero_prediction_classes": int(calibrated.get("zero_prediction_classes", 0)),
            "mean_entropy": float(calibrated.get("mean_entropy", 0.0)),
        },
    ]

    before_after_json_path = calibration_dir / f"{dataset_name}_before_after{suffix}.json"
    _atomic_write_json(
        before_after_json_path,
        {
            "dataset": dataset_name,
            "seed": int(seed),
            "temperature": float(calibration_payload.get("temperature", 1.0)),
            "tau_4": float(calibration_payload.get("tau_4", 0.5)),
            "rows": before_after_rows,
        },
    )
    before_after_csv_path = calibration_dir / f"{dataset_name}_before_after{suffix}.csv"
    pd.DataFrame(before_after_rows).to_csv(before_after_csv_path, index=False)

    pr_payload = cast(dict[str, Any], calibration_payload.get("pr_curve_class4", {}))
    pr_precision = [float(v) for v in cast(list[Any], pr_payload.get("precision", []))]
    pr_recall = [float(v) for v in cast(list[Any], pr_payload.get("recall", []))]
    pr_thresholds = [float(v) for v in cast(list[Any], pr_payload.get("thresholds", []))]
    max_rows = max(len(pr_precision), len(pr_recall), len(pr_thresholds))
    pr_rows: list[dict[str, Any]] = []
    for idx in range(max_rows):
        pr_rows.append(
            {
                "point_index": int(idx),
                "precision": pr_precision[idx] if idx < len(pr_precision) else None,
                "recall": pr_recall[idx] if idx < len(pr_recall) else None,
                "threshold": pr_thresholds[idx] if idx < len(pr_thresholds) else None,
            }
        )
    pr_csv_path = calibration_dir / f"{dataset_name}_pr_curve_class4{suffix}.csv"
    pd.DataFrame(pr_rows).to_csv(pr_csv_path, index=False)

    confusion_payload = {
        "dataset": dataset_name,
        "seed": int(seed),
        "uncalibrated_test_argmax": cast(dict[str, Any], calibration_payload.get("uncalibrated", {}).get("test_argmax", {})).get(
            "confusion_matrix", []
        ),
        "ablation_without_thresholding": cast(dict[str, Any], calibration_payload.get("ablation", {}).get("without_thresholding", {})).get(
            "confusion_matrix", []
        ),
        "ablation_without_temperature_scaling": cast(
            dict[str, Any],
            calibration_payload.get("ablation", {}).get("without_temperature_scaling", {}),
        ).get("confusion_matrix", []),
        "calibrated": cast(dict[str, Any], calibration_payload.get("test", {})).get("confusion_matrix", []),
    }
    confusion_json_path = calibration_dir / f"{dataset_name}_confusion_matrices{suffix}.json"
    _atomic_write_json(confusion_json_path, confusion_payload)

    ablation_json_path = calibration_dir / f"{dataset_name}_ablation{suffix}.json"
    _atomic_write_json(
        ablation_json_path,
        {
            "dataset": dataset_name,
            "seed": int(seed),
            "ablation": cast(dict[str, Any], calibration_payload.get("ablation", {})),
        },
    )

    return {
        "calibration_json": str(calibration_json_path),
        "before_after_json": str(before_after_json_path),
        "before_after_csv": str(before_after_csv_path),
        "pr_curve_csv": str(pr_csv_path),
        "confusion_matrices_json": str(confusion_json_path),
        "ablation_json": str(ablation_json_path),
    }


def _normalize_metrics_payload(metrics: dict[str, Any]) -> dict[str, Any]:
    """Normalize metric aliases into strict external contract keys."""
    return normalize_metrics_payload(metrics)


def _materialize_phase8_artifacts(calibration_artifacts: dict[str, str]) -> dict[str, str]:
    """Create canonical artifact filenames required by strict completion contract."""
    return materialize_phase8_artifacts(calibration_artifacts)


def _normalize_calibration_block(
    *,
    calibration_payload: dict[str, Any],
    calibration_artifacts: dict[str, str],
) -> dict[str, Any]:
    """Normalize calibration outputs into strict contract schema with required paths."""
    return normalize_calibration_block(
        calibration_payload=calibration_payload,
        calibration_artifacts=calibration_artifacts,
    )


def _load_seed_run_artifacts(
    *,
    seed: int,
    proc: subprocess.CompletedProcess[str],
) -> tuple[str, dict[str, Any], dict[str, Any], HelixIDSFull]:
    """Load evaluation and training artifacts for a completed seed run.

    Returns (dataset_name, train_payload, eval_results, model).

    Raises RuntimeError when expected artifacts are missing or the seed
    run exited with a non-zero code.
    """
    return load_seed_run_artifacts(seed=seed, proc=proc)


def _summarize_governance(strict_seed_runs: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str], list[str]]:
    """Aggregate multi-seed runs and emit strict governance payload.

    Returns (governance, failure_reasons, actions).
    """
    return summarize_governance(strict_seed_runs)


def _run_multiseed_calibrated_governance(
    *,
    script_path: Path,
    argv: list[str],
    seeds: list[int],
    max_temperature: float = 5.0,
    class4_recall_floor: float = 0.80,
) -> dict[str, Any]:
    """Run fixed-config 50-epoch training across seeds and emit strict governance payload."""
    strict_seed_runs: list[dict[str, Any]] = []

    for seed in seeds:
        cmd = [
            sys.executable,
            str(script_path),
            *argv,
            "--seed",
            str(int(seed)),
            "--epochs",
            "50",
            "--disable-early-stopping",
            "--calibration-mode",
            "internal_off",
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        dataset_name, _train_payload, _, model = _load_seed_run_artifacts(seed=seed, proc=proc)
        model_config = cast(HelixFullConfig, model.config)

        # Rebuild loaders from persisted split for deterministic calibration pass.
        splits_dir = Path("data/processed/multi_dataset_v1")
        x_val = np.load(splits_dir / f"X_val_{dataset_name}.npy", mmap_mode="r")
        y_val = np.load(splits_dir / f"y_val_{dataset_name}.npy").astype(np.int64, copy=False)
        y_val = np.where(y_val >= int(model_config.family_output_dim), int(model_config.family_output_dim) - 1, y_val)
        x_test = np.load(splits_dir / f"X_test_{dataset_name}.npy", mmap_mode="r")
        y_test = np.load(splits_dir / f"y_test_{dataset_name}.npy").astype(np.int64, copy=False)
        y_test = np.where(y_test >= int(model_config.family_output_dim), int(model_config.family_output_dim) - 1, y_test)

        val_loader = DataLoader(
            MultiTaskNumpyDataset(x_val, y_val),
            batch_size=512,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        test_loader = DataLoader(
            MultiTaskNumpyDataset(x_test, y_test),
            batch_size=512,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        calib = _calibrate_family_predictions(
            model=model,
            val_loader=val_loader,
            test_loader=test_loader,
            device="cpu",
            class4_id=4,
            max_temperature=float(max_temperature),
            min_class4_recall=float(class4_recall_floor),
        )
        raw_artifacts = _emit_calibration_artifacts(
            results_dir=HELIX_FULL_RESULTS_DIR,
            dataset_name=dataset_name,
            seed=int(seed),
            calibration_payload=calib,
            artifact_tag="governance",
        )
        normalized_metrics = _normalize_metrics_payload(cast(dict[str, Any], calib.get("test", {})))

        strict_seed_runs.append(
            {
                "seed": int(seed),
                "dataset": dataset_name,
                "return_code": int(proc.returncode),
                "macro_f1": float(normalized_metrics["macro_f1"]),
                "class4_precision": float(normalized_metrics["class4_precision"]),
                "class4_recall": float(normalized_metrics["class4_recall"]),
                "entropy": float(normalized_metrics["entropy"]),
                "zero_prediction_classes": int(normalized_metrics["zero_prediction_classes"]),
                "raw_calibration_payload": calib,
                "raw_calibration_artifacts": raw_artifacts,
            }
        )

    if not strict_seed_runs:
        raise RuntimeError("No seed runs available for governance aggregation")

    governance, failure_reasons, actions = _summarize_governance(strict_seed_runs)

    governance["status"] = "PASS" if not failure_reasons else "FAIL"
    governance["failure_reasons"] = failure_reasons
    governance["actions"] = actions

    anchor_run = next((run for run in strict_seed_runs if int(run["seed"]) == 42), strict_seed_runs[0])
    canonical_artifacts = _materialize_phase8_artifacts(
        cast(dict[str, str], anchor_run["raw_calibration_artifacts"])
    )
    normalized_calibration = _normalize_calibration_block(
        calibration_payload=cast(dict[str, Any], anchor_run["raw_calibration_payload"]),
        calibration_artifacts=canonical_artifacts,
    )
    final_payload = {
        "macro_f1": float(anchor_run["macro_f1"]),
        "class4_precision": float(anchor_run["class4_precision"]),
        "class4_recall": float(anchor_run["class4_recall"]),
        "entropy": float(anchor_run["entropy"]),
        "zero_prediction_classes": int(anchor_run["zero_prediction_classes"]),
        "calibration": normalized_calibration,
        "governance": governance,
    }
    return final_payload


def _build_ab_raw_metrics(
    *,
    dataset_name: str,
    dataset_id: str,
    split_snapshot_id: str,
    ab_track: str,
    ab_change_id: str,
    k: int,
    seed: int,
    batch_size: int,
    feature_signature: str,
    cluster_objective: str,
    cluster_spectral_affinity: str,
    representation_diagnostics: dict[str, Any],
    dataset_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Build raw per-run A/B metrics payload for a single dataset."""
    return build_ab_raw_metrics(
        dataset_name=dataset_name,
        dataset_id=dataset_id,
        split_snapshot_id=split_snapshot_id,
        ab_track=ab_track,
        ab_change_id=ab_change_id,
        k=k,
        seed=seed,
        batch_size=batch_size,
        feature_signature=feature_signature,
        cluster_objective=cluster_objective,
        cluster_spectral_affinity=cluster_spectral_affinity,
        representation_diagnostics=representation_diagnostics,
        dataset_metrics=dataset_metrics,
    )



def _ab_rejection(reason: str) -> dict[str, Any]:
    """Return standard A/B rejection response."""
    return ab_rejection(reason)


def _validate_ab_contract(
    current: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any] | None:
    """Validate A/B contract fields. Returns error dict if invalid, None if valid."""
    return validate_ab_contract(current, baseline)


def _detect_feature_and_objective_changes(
    current: dict[str, Any],
    baseline: dict[str, Any],
) -> tuple[bool, bool]:
    """Detect if feature or objective changed."""
    return detect_feature_and_objective_changes(current, baseline)


def _validate_track(track: str, feature_changed: bool, objective_changed: bool) -> dict[str, Any] | None:
    """Validate track against changes. Returns error dict if invalid, None if valid."""
    return validate_track(track, feature_changed, objective_changed)


def evaluate_ab_candidate(
    *,
    current: dict[str, Any],
    baseline: dict[str, Any],
    ab_track: str,
    governance_z_score: float,
    governance_z_tolerance: float,
) -> dict[str, Any]:
    """Evaluate strict tiered A/B acceptance gates and promotion rule."""
    return _gov_evaluate_ab_candidate(
        current=current,
        baseline=baseline,
        ab_track=ab_track,
        governance_z_score=governance_z_score,
        governance_z_tolerance=governance_z_tolerance,
    )


# ============================================================================
# Setup Logging
# ============================================================================


def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging to file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("HelixFullTraining")
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_dir / "training.log")
    fh.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def _chunk_finite_check(x: np.ndarray, chunk_rows: int = 250000) -> bool:
    """Check finite values in chunks to avoid large temporary allocations."""
    for start_idx in range(0, int(x.shape[0]), chunk_rows):
        chunk = np.asarray(x[start_idx : start_idx + chunk_rows], dtype=np.float32)
        if not np.isfinite(chunk).all():
            return False
    return True


def _sample_rows(x: np.ndarray, *, seed: int, max_rows: int = 50000) -> np.ndarray:
    """Sample rows for distribution checks without loading full arrays into memory."""
    n_rows = int(x.shape[0])
    if n_rows <= max_rows:
        return np.asarray(x, dtype=np.float32)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_rows, size=max_rows, replace=False)
    return np.asarray(x[idx], dtype=np.float32)


def _normalize_engineered_feature_block(
    *,
    dataset_name: str,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, dict[str, float]]]:
    """Apply train-fit z-score normalization to engineered geometry features only."""
    engineered_indices = [
        idx
        for idx, name in enumerate(feature_names)
        if str(name).strip().lower() in ENGINEERED_FEATURE_NAMES
    ]
    if not engineered_indices:
        return (
            np.asarray(x_train, dtype=np.float32),
            np.asarray(x_val, dtype=np.float32),
            np.asarray(x_test, dtype=np.float32),
            {},
        )

    train = np.asarray(x_train, dtype=np.float32).copy()
    val = np.asarray(x_val, dtype=np.float32).copy()
    test = np.asarray(x_test, dtype=np.float32).copy()

    stats: dict[str, dict[str, float]] = {}
    for idx in engineered_indices:
        feature_name = str(feature_names[idx])
        train_col = np.asarray(train[:, idx], dtype=np.float64)
        if not np.isfinite(train_col).all():
            raise RuntimeError(
                "Hard-stop integrity guard triggered: engineered_feature_non_finite_train_"
                f"{dataset_name}:{feature_name}"
            )

        mean = float(np.mean(train_col))
        std = float(np.std(train_col))
        scale = max(MIN_FEATURE_STD, std)

        train[:, idx] = ((train[:, idx] - mean) / scale).astype(np.float32, copy=False)
        val[:, idx] = ((val[:, idx] - mean) / scale).astype(np.float32, copy=False)
        test[:, idx] = ((test[:, idx] - mean) / scale).astype(np.float32, copy=False)

        stats[feature_name] = {
            "train_mean": float(np.mean(train[:, idx])),
            "train_std": float(np.std(train[:, idx])),
            "train_p99_abs": float(np.percentile(np.abs(train[:, idx]), 99.0)),
        }

    return train, val, test, stats


def _inverse_frequency_weights(y: np.ndarray, *, minlength: int) -> np.ndarray:
    """Return mean-normalized inverse-frequency weights with zeros for missing classes."""
    counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=minlength).astype(np.float64)
    weights = np.zeros_like(counts, dtype=np.float64)
    present_mask = counts > 0
    if bool(np.any(present_mask)):
        present = counts[present_mask]
        weights[present_mask] = present.sum() / (present_mask.sum() * present)
        weights[present_mask] /= max(1e-8, float(np.mean(weights[present_mask])))
    return weights.astype(np.float32)


def _sqrt_inverse_frequency_weights(y: np.ndarray, *, minlength: int) -> np.ndarray:
    """Return mean-normalized sqrt-inverse-frequency weights with zeros for missing classes."""
    counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=minlength).astype(np.float64)
    weights = np.zeros_like(counts, dtype=np.float64)
    present_mask = counts > 0
    if bool(np.any(present_mask)):
        present = counts[present_mask]
        # Equivalent to 1 / sqrt(freq_i) up to a shared constant scaling.
        weights[present_mask] = np.sqrt(present.sum() / (present_mask.sum() * present))
        weights[present_mask] /= max(1e-8, float(np.mean(weights[present_mask])))
    return weights.astype(np.float32)


def _apply_label_merges(
    y: np.ndarray,
    *,
    merges: list[tuple[int, int]],
) -> np.ndarray:
    """Apply deterministic label merges (src -> dst) to family labels."""
    y_int: np.ndarray = np.asarray(y, dtype=np.int64).copy()
    if not merges:
        return y_int

    merge_map = {int(src): int(dst) for src, dst in merges}

    def _resolve(label: int) -> int:
        seen: set[int] = set()
        cur = int(label)
        while cur in merge_map and cur not in seen:
            seen.add(cur)
            cur = int(merge_map[cur])
        return cur

    if merge_map:
        remap = {src: _resolve(src) for src in merge_map}
        for src, dst in remap.items():
            y_int[y_int == int(src)] = int(dst)

    return y_int


def _assert_categorical_encoding_sanity(
    *,
    dataset_name: str,
    feature_names: list[str],
    train_sample: np.ndarray,
    val_sample: np.ndarray,
) -> None:
    categorical_feature_names = {
        "protocol",
        "protocol_type",
        "service",
        "flag",
        "state",
    }
    categorical_indices = [
        idx
        for idx, name in enumerate(feature_names)
        if str(name).strip().lower() in categorical_feature_names
    ]
    for idx in categorical_indices:
        feature_name = str(feature_names[idx])
        train_col = np.asarray(train_sample[:, idx], dtype=np.float64)
        val_col = np.asarray(val_sample[:, idx], dtype=np.float64)

        if not np.isfinite(train_col).all() or not np.isfinite(val_col).all():
            raise RuntimeError(
                "Hard-stop integrity guard triggered: categorical_non_finite_values_"
                f"{dataset_name}:{feature_name}"
            )

        train_integer_like = float(np.mean(np.abs(train_col - np.rint(train_col)) < 1e-6))
        val_integer_like = float(np.mean(np.abs(val_col - np.rint(val_col)) < 1e-6))
        if train_integer_like < 0.999 or val_integer_like < 0.999:
            raise RuntimeError(
                "Hard-stop integrity guard triggered: categorical_not_integer_encoded_"
                f"{dataset_name}:{feature_name}"
            )

        train_codes = np.rint(train_col).astype(np.int64, copy=False)
        if train_codes.size == 0:
            continue
        unique_codes = np.unique(train_codes)
        if unique_codes.size > 4096:
            raise RuntimeError(
                "Hard-stop integrity guard triggered: categorical_code_cardinality_too_high_"
                f"{dataset_name}:{feature_name}:{unique_codes.size}"
            )


def _assert_feature_dimensions(
    *,
    dataset_name: str,
    x_train: np.ndarray,
    x_val: np.ndarray,
    feature_names: list[str],
    expected_feature_dim: Optional[int],
) -> None:
    """Validate rank and feature-dimension alignment for train/val matrices."""
    if x_train.ndim != 2:
        raise RuntimeError(
            f"Hard-stop integrity guard triggered: invalid_train_feature_rank_{dataset_name}"
        )
    if x_val.ndim != 2:
        raise RuntimeError(
            f"Hard-stop integrity guard triggered: invalid_val_feature_rank_{dataset_name}"
        )
    if expected_feature_dim is not None:
        if int(x_train.shape[1]) != int(expected_feature_dim):
            raise RuntimeError(
                "Hard-stop integrity guard triggered: feature_dim_not_expected_"
                f"{dataset_name}:expected_{int(expected_feature_dim)}_got_{int(x_train.shape[1])}"
            )
        if int(x_val.shape[1]) != int(expected_feature_dim):
            raise RuntimeError(
                "Hard-stop integrity guard triggered: val_feature_dim_not_expected_"
                f"{dataset_name}:expected_{int(expected_feature_dim)}_got_{int(x_val.shape[1])}"
            )
    if int(x_train.shape[1]) != len(feature_names):
        raise RuntimeError(
            "Hard-stop integrity guard triggered: feature_name_count_mismatch_"
            f"{dataset_name}"
        )
    if not np.issubdtype(np.asarray(x_train).dtype, np.number):
        raise RuntimeError(
            f"Hard-stop integrity guard triggered: non_numeric_train_features_{dataset_name}"
        )


def _assert_numeric_finite_and_variance(
    *,
    dataset_name: str,
    x_train: np.ndarray,
    x_val: np.ndarray,
    feature_names: list[str],
    min_feature_std: float,
) -> None:
    """Validate full-array finiteness and variance floor before sampling checks."""

    if not _chunk_finite_check(np.asarray(x_train, dtype=np.float32)) or not _chunk_finite_check(
        np.asarray(x_val, dtype=np.float32)
    ):
        raise RuntimeError(
            f"Hard-stop integrity guard triggered: non_finite_feature_values_{dataset_name}"
        )

    full_feature_std = np.nanstd(np.asarray(x_train, dtype=np.float32), axis=0)
    low_std_idx = np.nonzero(full_feature_std < float(min_feature_std))[0]
    if low_std_idx.size > 0:
        low_std_features = [feature_names[int(idx)] for idx in low_std_idx[:10].tolist()]
        raise RuntimeError(
            "Hard-stop integrity guard triggered: low_variance_features_present_"
            f"{dataset_name}: threshold={float(min_feature_std):.2e} features={low_std_features}"
        )


def _assert_feature_sanity_for_dataset(
    *,
    dataset_name: str,
    x_train: np.ndarray,
    x_val: np.ndarray,
    feature_names: list[str],
    expected_feature_dim: Optional[int],
    min_feature_std: float,
    seed: int,
    logger: logging.Logger,
) -> None:
    """Validate feature integrity: no constants, sane scaling, and encoded categoricals."""
    _assert_feature_dimensions(
        dataset_name=dataset_name,
        x_train=x_train,
        x_val=x_val,
        feature_names=feature_names,
        expected_feature_dim=expected_feature_dim,
    )
    _assert_numeric_finite_and_variance(
        dataset_name=dataset_name,
        x_train=x_train,
        x_val=x_val,
        feature_names=feature_names,
        min_feature_std=min_feature_std,
    )

    train_sample = _sample_rows(x_train, seed=seed)
    val_sample = _sample_rows(x_val, seed=seed + 1)

    if not np.isfinite(train_sample).all() or not np.isfinite(val_sample).all():
        raise RuntimeError(
            f"Hard-stop integrity guard triggered: non_finite_feature_values_{dataset_name}"
        )

    feature_std = np.nanstd(train_sample, axis=0)
    constant_idx = np.nonzero(feature_std <= 1e-12)[0]
    if constant_idx.size > 0:
        constant_features = [feature_names[int(idx)] for idx in constant_idx[:10].tolist()]
        raise RuntimeError(
            "Hard-stop integrity guard triggered: constant_features_present_"
            f"{dataset_name}: {constant_features}"
        )

    _assert_categorical_encoding_sanity(
        dataset_name=dataset_name,
        feature_names=feature_names,
        train_sample=train_sample,
        val_sample=val_sample,
    )

    train_scale = float(np.nanpercentile(np.abs(train_sample), 99))
    val_scale = float(np.nanpercentile(np.abs(val_sample), 99))
    if not np.isfinite(train_scale) or not np.isfinite(val_scale):
        raise RuntimeError(
            f"Hard-stop integrity guard triggered: invalid_scale_statistics_{dataset_name}"
        )
    if train_scale > 1e4 or val_scale > 1e4:
        raise RuntimeError(
            f"Hard-stop integrity guard triggered: scale_normalization_not_applied_{dataset_name}"
        )

    min_scale = max(1e-8, float(min(train_scale, val_scale)))
    scale_ratio = max(train_scale, val_scale) / min_scale
    if scale_ratio > 20.0:
        raise RuntimeError(
            "Hard-stop integrity guard triggered: cross_split_scale_mismatch_"
            f"{dataset_name}"
        )

    integer_like_fraction = np.mean(np.abs(train_sample - np.rint(train_sample)) < 1e-6, axis=0)
    integer_like_count = int(np.sum(integer_like_fraction >= 0.999))

    logger.info(
        "FeatureSanity[%s] constant_features=0 scale_p99(train)=%.4f scale_p99(val)=%.4f "
        "scale_ratio=%.3f integer_like_features=%d",
        dataset_name,
        train_scale,
        val_scale,
        scale_ratio,
        integer_like_count,
    )


def build_class_index(y: np.ndarray) -> dict[int, np.ndarray]:
    """Build per-class index lists for balanced batch sampling."""
    class_index: defaultdict[int, list[int]] = defaultdict(list)
    y_int = np.asarray(y, dtype=np.int64)
    for idx, label in enumerate(y_int.tolist()):
        class_index[int(label)].append(idx)
    return {label: np.asarray(idxs, dtype=np.int64) for label, idxs in class_index.items()}


def _default_tail_multiplier(count: int) -> float:
    """Return aggressive oversampling multipliers for extreme tail classes."""
    if count <= 64:
        return 80.0
    if count <= 1000:
        return 15.0
    return 1.0


def _build_stratified_val_subset(
    x: np.ndarray,
    y: np.ndarray,
    *,
    target_per_class: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a stratified validation subset with at least target_per_class per active class."""
    y_int = np.asarray(y, dtype=np.int64)
    selected_idx = _build_stratified_subset_indices(
        y_int,
        target_per_class=target_per_class,
        seed=seed,
    )
    return (
        np.asarray(x[selected_idx], dtype=np.float32),
        np.asarray(y_int[selected_idx], dtype=np.int64),
    )


def _build_stratified_subset_indices(
    y: np.ndarray,
    *,
    target_per_class: int,
    seed: int,
) -> np.ndarray:
    """Return deterministic stratified indices for reproducible evaluation subsets."""
    y_int = np.asarray(y, dtype=np.int64)
    class_index = build_class_index(y_int)
    if not class_index:
        return np.arange(y_int.shape[0], dtype=np.int64)

    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for class_id in sorted(class_index):
        idxs = class_index[class_id]
        sampled = rng.choice(
            idxs,
            size=int(target_per_class),
            replace=bool(idxs.size < int(target_per_class)),
        )
        selected.extend(int(i) for i in sampled.tolist())

    rng.shuffle(selected)
    return np.asarray(selected, dtype=np.int64)


def _build_interleaved_round_robin_indices(  # NOSONAR
    y: np.ndarray,
    *,
    batch_size: int,
    seed: int,
    min_unique_classes_per_batch: int = 2,
    class4_min_per_batch: int = 0,
) -> np.ndarray:
    """Build deterministic interleaved indices with batch-level class diversity.

    Construction logic:
    1) bucket samples by class
    2) shuffle within each class bucket
    3) round-robin merge classes into each batch
    4) hard-validate per-batch unique-class floor
    """
    y_int = np.asarray(y, dtype=np.int64)
    class_index = build_class_index(y_int)
    classes = sorted(class_index.keys())
    batch_size_int = int(batch_size)
    min_unique = max(1, int(min_unique_classes_per_batch))
    class4_quota = max(0, int(class4_min_per_batch))
    _validate_interleaved_round_robin_inputs(
        classes=classes,
        class_index=class_index,
        batch_size_int=batch_size_int,
        min_unique=min_unique,
        class4_quota=class4_quota,
    )

    rng = np.random.default_rng(int(seed))
    steps_per_epoch = max(1, int(math.ceil(y_int.shape[0] / max(1, batch_size_int))))
    buckets: dict[int, np.ndarray] = {
        int(class_id): rng.permutation(np.asarray(indices, dtype=np.int64))
        for class_id, indices in class_index.items()
    }
    pointers: dict[int, int] = {int(class_id): 0 for class_id in classes}
    class_cycle = np.asarray(classes, dtype=np.int64)
    rng.shuffle(class_cycle)
    class_cycle_list = [int(v) for v in class_cycle.tolist()]
    non4_cycle_list = [int(c) for c in class_cycle_list if int(c) != 4] or class_cycle_list

    flat_indices: list[int] = []
    cycle_state = {"cycle_pos": 0, "non4_cycle_pos": 0}
    for _ in range(steps_per_epoch):
        batch_indices = _build_interleaved_round_robin_batch(
            y_int=y_int,
            batch_size_int=batch_size_int,
            min_unique=min_unique,
            class4_quota=class4_quota,
            class_index=class_index,
            buckets=buckets,
            pointers=pointers,
            class_cycle_list=class_cycle_list,
            non4_cycle_list=non4_cycle_list,
            cycle_state=cycle_state,
            rng=rng,
        )
        flat_indices.extend(batch_indices)

    return np.asarray(flat_indices, dtype=np.int64)


def _validate_interleaved_round_robin_inputs(
    *,
    classes: list[int],
    class_index: dict[int, np.ndarray],
    batch_size_int: int,
    min_unique: int,
    class4_quota: int,
) -> None:
    if not classes:
        raise ValueError("No classes available for interleaved sampler")
    if class4_quota > batch_size_int:
        raise ValueError(
            "class4_min_per_batch cannot exceed batch_size: "
            f"quota={class4_quota} batch_size={batch_size_int}"
        )
    if class4_quota > 0 and 4 not in class_index:
        raise ValueError("class4_min_per_batch requested but class 4 is absent from labels")
    if len(classes) < min_unique:
        raise ValueError(
            "Insufficient active classes for interleaved sampler: "
            f"active={len(classes)} required={min_unique}"
        )


def _build_interleaved_round_robin_batch(
    *,
    y_int: np.ndarray,
    batch_size_int: int,
    min_unique: int,
    class4_quota: int,
    class_index: dict[int, np.ndarray],
    buckets: dict[int, np.ndarray],
    pointers: dict[int, int],
    class_cycle_list: list[int],
    non4_cycle_list: list[int],
    cycle_state: dict[str, int],
    rng: np.random.Generator,
) -> list[int]:
    def draw_from_class(class_id: int) -> int:
        cls_id = int(class_id)
        bucket = buckets[cls_id]
        ptr = int(pointers[cls_id])
        if ptr >= int(bucket.shape[0]):
            bucket = rng.permutation(np.asarray(class_index[cls_id], dtype=np.int64))
            buckets[cls_id] = bucket
            ptr = 0
        sample_idx = int(bucket[ptr])
        pointers[cls_id] = ptr + 1
        return sample_idx

    batch_indices: list[int] = []
    if class4_quota > 0:
        for _ in range(class4_quota):
            batch_indices.append(draw_from_class(4))

    seeded_class_ids = [4] if class4_quota > 0 else []
    for _ in range(max(0, min_unique - len(set(seeded_class_ids)))):
        class_id = non4_cycle_list[cycle_state["non4_cycle_pos"] % len(non4_cycle_list)]
        cycle_state["non4_cycle_pos"] += 1
        seeded_class_ids.append(int(class_id))
        batch_indices.append(draw_from_class(int(class_id)))

    while len(batch_indices) < batch_size_int:
        class_id = class_cycle_list[cycle_state["cycle_pos"] % len(class_cycle_list)]
        cycle_state["cycle_pos"] += 1
        batch_indices.append(draw_from_class(int(class_id)))

    rng.shuffle(batch_indices)
    batch_unique = int(np.unique(y_int[np.asarray(batch_indices, dtype=np.int64)]).shape[0])
    if batch_unique < min_unique:
        raise RuntimeError(
            "Hard-stop integrity guard triggered: interleaved_sampler_batch_diversity_violation"
        )
    if class4_quota > 0:
        class4_count = int(np.sum(y_int[np.asarray(batch_indices, dtype=np.int64)] == 4))
        if class4_count < class4_quota:
            raise RuntimeError(
                "Hard-stop integrity guard triggered: interleaved_sampler_class4_quota_violation"
            )
    return batch_indices


def _build_frozen_class_balanced_indices(
    y: np.ndarray,
    *,
    batch_size: int,
    seed: int,
    min_per_class: int = 1,
    class_multipliers: Optional[dict[int, float]] = None,
) -> np.ndarray:
    """Build a fixed class-balanced index schedule once per run."""
    y_int = np.asarray(y, dtype=np.int64)
    class_index = build_class_index(y_int)
    classes = sorted(class_index.keys())
    if not classes:
        raise ValueError("No classes available for frozen sampler")

    min_per_class = max(1, int(min_per_class))
    required_slots = len(classes) * min_per_class
    if required_slots > int(batch_size):
        raise ValueError(
            "batch_size too small for per-class presence constraint: "
            f"required={required_slots}, batch_size={int(batch_size)}"
        )

    class_counts = {class_id: int(class_index[class_id].shape[0]) for class_id in classes}
    if class_multipliers is None:
        multipliers = {
            class_id: _default_tail_multiplier(class_counts[class_id])
            for class_id in classes
        }
    else:
        multipliers = {
            class_id: float(class_multipliers.get(class_id, 1.0))
            for class_id in classes
        }

    steps_per_epoch = max(1, int(math.ceil(y_int.shape[0] / max(1, int(batch_size)))))
    remainder = int(batch_size) - required_slots
    rng = np.random.default_rng(int(seed))
    flat_indices: list[int] = []

    sampling_weights = np.asarray(
        [float(class_counts[class_id]) * float(multipliers[class_id]) for class_id in classes],
        dtype=np.float64,
    )
    if float(sampling_weights.sum()) <= 0.0:
        sampling_weights = np.ones_like(sampling_weights)
    sampling_weights = sampling_weights / float(sampling_weights.sum())

    for _ in range(steps_per_epoch):
        batch_indices: list[int] = []
        for class_id in classes:
            cls_indices = class_index[class_id]
            sampled = rng.choice(cls_indices, size=min_per_class, replace=True)
            batch_indices.extend(int(x) for x in sampled.tolist())

        if remainder > 0:
            extra_classes = rng.choice(classes, size=remainder, replace=True, p=sampling_weights)
            for class_id in extra_classes.tolist():
                cls_indices = class_index[int(class_id)]
                sampled = rng.choice(cls_indices, size=1, replace=True)
                batch_indices.append(int(sampled[0]))

        rng.shuffle(batch_indices)
        flat_indices.extend(batch_indices)

    return np.asarray(flat_indices, dtype=np.int64)


def _build_frozen_tempered_indices(
    y: np.ndarray,
    *,
    batch_size: int,
    seed: int,
    temperature_power: float = 0.5,
    class_multipliers: Optional[dict[int, float]] = None,
) -> tuple[np.ndarray, dict[int, float]]:
    """Build fixed sampler indices with tempered class-frequency sampling.

    This relaxes strict class-balanced batches while preserving deterministic
    per-epoch exposure for all active classes.
    """
    y_int = np.asarray(y, dtype=np.int64)
    class_index = build_class_index(y_int)
    classes = sorted(class_index.keys())
    if not classes:
        raise ValueError("No classes available for tempered sampler")

    class_counts = {class_id: int(class_index[class_id].shape[0]) for class_id in classes}
    if class_multipliers is None:
        multipliers = dict.fromkeys(classes, 1.0)
    else:
        multipliers = {class_id: float(class_multipliers.get(class_id, 1.0)) for class_id in classes}

    power = float(max(0.0, temperature_power))
    raw_weights = np.asarray(
        [
            max(1.0, float(class_counts[class_id])) ** power
            * max(1e-6, float(multipliers[class_id]))
            for class_id in classes
        ],
        dtype=np.float64,
    )
    if float(raw_weights.sum()) <= 0.0:
        raw_weights = np.ones_like(raw_weights)
    class_probs = raw_weights / float(raw_weights.sum())

    steps_per_epoch = max(1, int(math.ceil(y_int.shape[0] / max(1, int(batch_size)))))
    total_draws = steps_per_epoch * int(batch_size)
    rng = np.random.default_rng(int(seed))

    # Guarantee at least one sample per active class each epoch.
    seeded_classes = np.asarray(classes, dtype=np.int64)
    rng.shuffle(seeded_classes)
    seeded_draws = min(int(seeded_classes.shape[0]), total_draws)

    sampled_class_ids = np.empty((total_draws,), dtype=np.int64)
    if seeded_draws > 0:
        sampled_class_ids[:seeded_draws] = seeded_classes[:seeded_draws]
    remaining = total_draws - seeded_draws
    if remaining > 0:
        sampled_class_ids[seeded_draws:] = rng.choice(
            np.asarray(classes, dtype=np.int64),
            size=remaining,
            replace=True,
            p=class_probs,
        )

    flat_indices: list[int] = []
    for class_id in sampled_class_ids.tolist():
        cls_indices = class_index[int(class_id)]
        sample_idx = int(rng.choice(cls_indices, size=1, replace=True)[0])
        flat_indices.append(sample_idx)

    return np.asarray(flat_indices, dtype=np.int64), {
        int(class_id): float(prob) for class_id, prob in zip(classes, class_probs.tolist())
    }


def _validate_per_dataset_splits(  # NOSONAR
    splits: dict[str, np.ndarray],
    *,
    logger: logging.Logger,
    seed: int,
    enforce_cross_dataset_scale: bool = False,
) -> None:
    """Validate class presence, finite features, and cross-dataset scaling consistency."""
    datasets = ["nsl_kdd", "unsw_nb15", "cicids"]
    split_order = ["train", "val", "test"]
    reference_feature_dim: Optional[int] = None
    dataset_scale_stats: dict[str, dict[str, float]] = {}

    for dataset_idx, dataset_name in enumerate(datasets):
        seen_non_empty = False
        for split_name in split_order:
            x_key = f"X_{split_name}_{dataset_name}"
            y_key = f"y_{split_name}_{dataset_name}"
            if x_key not in splits or y_key not in splits:
                continue

            x_arr = splits[x_key]
            y_family = np.asarray(splits[y_key], dtype=np.int64)

            if int(x_arr.shape[0]) == 0:
                continue

            seen_non_empty = True
            if int(x_arr.shape[0]) != int(y_family.shape[0]):
                raise RuntimeError(
                    "Hard-stop integrity guard triggered: split_length_mismatch_"
                    f"{dataset_name}_{split_name}"
                )

            feature_dim = int(x_arr.shape[1])
            if reference_feature_dim is None:
                reference_feature_dim = feature_dim
            elif feature_dim != reference_feature_dim:
                raise RuntimeError(
                    "Hard-stop integrity guard triggered: feature_dim_mismatch_"
                    f"{dataset_name}_{split_name}"
                )

            if not _chunk_finite_check(x_arr):
                raise RuntimeError(
                    "Hard-stop integrity guard triggered: non_finite_features_"
                    f"{dataset_name}_{split_name}"
                )

            family_classes = np.unique(y_family)
            if family_classes.size < 2:
                raise RuntimeError(
                    "Hard-stop integrity guard triggered: single_family_class_"
                    f"{dataset_name}_{split_name}"
                )

            y_binary = (y_family != 0).astype(np.int64, copy=False)
            if np.unique(y_binary).size < 2:
                raise RuntimeError(
                    "Hard-stop integrity guard triggered: single_binary_class_"
                    f"{dataset_name}_{split_name}"
                )

            logger.info(
                f"Integrity[{dataset_name}/{split_name}] rows={x_arr.shape[0]:,}, "
                f"feature_dim={feature_dim}, family_classes={family_classes.size}, "
                "binary_classes=2"
            )

            # Capture one representative split per dataset for scale checks.
            if dataset_name not in dataset_scale_stats:
                sample = _sample_rows(x_arr, seed=seed + dataset_idx)
                p05 = np.percentile(sample, 5, axis=0)
                p95 = np.percentile(sample, 95, axis=0)
                feature_widths = np.clip(p95 - p05, 0.0, None)
                active_widths = feature_widths[feature_widths > 1e-6]
                scale_width = (
                    float(np.median(active_widths)) if active_widths.size > 0 else 0.0
                )
                dataset_scale_stats[dataset_name] = {
                    "width": scale_width,
                    "p01": float(np.percentile(sample, 1)),
                    "p99": float(np.percentile(sample, 99)),
                    "abs_max": float(np.max(np.abs(sample))),
                }

        if not seen_non_empty:
            logger.warning(f"Integrity[{dataset_name}] has no non-empty splits; skipping checks")

    if enforce_cross_dataset_scale and len(dataset_scale_stats) >= 2:
        width_values = np.asarray(
            [stats["width"] for stats in dataset_scale_stats.values()],
            dtype=np.float64,
        )
        median_width = float(max(1e-8, np.median(width_values)))
        for dataset_name, stats in dataset_scale_stats.items():
            width = float(stats["width"])
            p01 = float(stats["p01"])
            p99 = float(stats["p99"])
            abs_max = float(stats.get("abs_max", 0.0))
            ratio = float(width / median_width)
            logger.info(
                f"Scale[{dataset_name}] p01={p01:.6f}, p99={p99:.6f}, abs_max={abs_max:.6f}, "
                f"width={width:.6f}, ratio_to_median={ratio:.3f}"
            )

            # Hard requirement under strict intersection schema: no NaN/inf, bounded ranges.
            if not np.isfinite(abs_max) or abs_max >= 1e6:
                raise RuntimeError(
                    "Hard-stop integrity guard triggered: cross_dataset_scale_explosion_"
                    f"{dataset_name}"
                )

            # Width differences can naturally occur due dataset-specific feature variance.
            if ratio < 0.05 or ratio > 20.0:
                logger.warning(
                    f"Scale[{dataset_name}] width ratio is highly imbalanced ({ratio:.3f}); "
                    "metrics must remain per-dataset and never averaged by sample count."
                )


def _load_eval_array(
    *,
    splits: dict[str, np.ndarray],
    dataset_name: str,
    split_name: str,
    prefix: str,
    logger: logging.Logger,
    expected_feature_dim: Optional[int] = None,
) -> np.ndarray:
    """Load validation/test arrays from the active precomputed split set."""
    key = f"{prefix}_{split_name}_{dataset_name}"
    if key not in splits:
        raise KeyError(f"Missing split key: {key}")

    arr = cast(np.ndarray, splits[key])
    if (
        prefix == "X"
        and expected_feature_dim is not None
        and arr.ndim == 2
        and int(arr.shape[1]) != int(expected_feature_dim)
    ):
        raise RuntimeError(
            "Hard-stop integrity guard triggered: eval_feature_dim_mismatch_"
            f"{dataset_name}_{split_name}"
        )

    if split_name in {"val", "test"}:
        logger.info("Using active precomputed split array: %s", key)
    return arr


def _load_precomputed_splits(
    *,
    splits_dir: Path,
    logger: logging.Logger,
    expected_feature_dim: Optional[int] = None,
) -> Optional[dict[str, np.ndarray]]:
    """Load precomputed split tensors when available to bypass raw CSV harmonization."""
    required = [
        "X_train.npy",
        "y_train.npy",
        "X_val.npy",
        "y_val.npy",
        "X_test_nsl_kdd.npy",
        "y_test_nsl_kdd.npy",
        "X_test_unsw_nb15.npy",
        "y_test_unsw_nb15.npy",
        "X_test_cicids.npy",
        "y_test_cicids.npy",
    ]
    if not splits_dir.exists():
        return None
    if not all((splits_dir / fname).exists() for fname in required):
        return None

    logger.info(f"Loading precomputed splits from {splits_dir}")
    splits: dict[str, np.ndarray] = {}
    for npy_path in sorted(splits_dir.glob("*.npy")):
        key = npy_path.stem
        if not (key.startswith(("X_", "y_")) or key in {"feature_columns", "train_class_weights"}):
            continue
        # Training arrays are loaded eagerly; validation/test arrays can be mem-mapped.
        if key.startswith(("X_test_", "X_val_")):
            splits[key] = cast(np.ndarray, np.load(npy_path, mmap_mode="r"))
        else:
            splits[key] = cast(
                np.ndarray,
                np.load(npy_path, allow_pickle=(key == "feature_columns")),
            )

    if expected_feature_dim is not None and "X_train" in splits:
        x_train = cast(np.ndarray, splits["X_train"])
        if x_train.ndim == 2 and int(x_train.shape[1]) != int(expected_feature_dim):
            logger.warning(
                "Ignoring precomputed splits due to feature_dim mismatch: "
                f"{splits_dir} (cached={int(x_train.shape[1])}, expected={int(expected_feature_dim)})"
            )
            return None

    return splits


# ============================================================================
# Training Loop
# ============================================================================


class HelixFullTrainer:
    """Trainer for HelixIDS-Full model."""

    def __init__(
        self,
        model: HelixIDSFull,
        train_loader: DataLoader,
        val_loaders: dict[str, DataLoader],
        test_loaders: dict[str, DataLoader],
        optimizer: optim.Optimizer,
        loss_fn: MultiTaskLoss,
        config: TrainingConfig,
        binary_class_weights: Optional[torch.Tensor] = None,
        family_class_weights: Optional[torch.Tensor] = None,
        train_family_class_count: Optional[int] = None,
        run_seed: int = 42,
        device: str = "mps",
        logger: Optional[logging.Logger] = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loaders = val_loaders
        self.test_loaders = test_loaders
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.config = config
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        self.binary_class_weights = (
            binary_class_weights.to(device) if binary_class_weights is not None else None
        )
        self.family_class_weights = (
            family_class_weights.to(device) if family_class_weights is not None else None
        )
        self.train_family_class_count = int(train_family_class_count or 0)
        self.base_balance_strategy = "weighted_ce"
        self.focal_warmup_epochs = 0
        self.family_log_prior: Optional[torch.Tensor] = None
        self.tail_class_mask: Optional[torch.Tensor] = None
        self.run_seed = int(run_seed)
        self.train_temperature = 1.0
        self.warmup_kl_uniform_weight = 0.0
        self.kl_uniform_weight = 0.0
        self.logit_floor = -2.0
        self.logit_floor_weight = 0.0
        self.tail_ce_weight = 0.0
        self.warmup_ratio = 0.0
        self.total_train_steps = max(1, int(len(self.train_loader)) * max(1, int(self.config.epochs)))
        self.warmup_steps = max(1, int(math.ceil(self.total_train_steps * self.warmup_ratio)))
        self.global_step = 0
        self.freeze_backbone_epochs = max(
            0,
            int(getattr(self.config, "freeze_backbone_epochs", 0)),
        )
        self.unfreeze_backbone_step = max(
            0,
            int(getattr(self.config, "unfreeze_backbone_step", 0)),
        )
        self.entropy_warmup_steps = max(
            0,
            int(getattr(self.config, "entropy_warmup_steps", 0)),
        )
        self.entropy_warmup_weight = max(
            0.0,
            float(getattr(self.config, "entropy_warmup_weight", 0.0)),
        )
        self.backbone_frozen = False
        self.step10_symmetry_logged = False
        self.backbone_params = [
            param for param in self.model.backbone.parameters() if param.requires_grad
        ]
        self.feature_order: list[str] = []
        self.schema_hash = "unknown"
        self.supcon_weight = 0.0
        self.supcon_temperature = 0.2
        self.step_coverage_check_step = 50
        self.step_coverage_checked = False
        self.active_family_class_ids: set[int] = set()
        self.class4_logit_shift = 0.0
        self.class4_logit_shift_class_id = 4
        self.representation_diagnostic_mode = False
        self.representation_only_steps = 0
        self.head_only_steps = 0
        self.representation_phase_active = False
        self.representation_curriculum_complete = False
        self.in_representation_window = False
        self.representation_window_pattern: list[tuple[bool, int]] = []
        self.head_phase_start_step = -1
        self.joint_finetune_start_step = -1
        self.joint_finetune_active = False
        self.joint_finetune_backbone_lr_multiplier = 0.25
        self.joint_finetune_head_lr_multiplier = 0.15
        self.coverage_check_after_head_steps = 50
        self.representation_diagnostics: dict[str, Any] = {}
        self.rep_phase_feature_chunks: list[torch.Tensor] = []
        self.rep_phase_label_chunks: list[torch.Tensor] = []
        self.representation_snapshot_id: Optional[str] = None
        self.cluster_relabeling_enabled = False
        self.cluster_relabel_k: Optional[int] = None
        self.cluster_relabel_seed = self.run_seed
        self.cluster_relabel_objective = "kmeans"
        self.cluster_relabel_spectral_affinity = "nearest_neighbors"
        self.cluster_centers: Optional[torch.Tensor] = None
        self.phase1_class_centroids: Optional[torch.Tensor] = None
        self.phase1_centroid_class_ids: list[int] = []
        self.geometry_min_inter_threshold = 0.2
        self.geometry_max_intra_inter_ratio_warmup = 2.5
        self.geometry_max_intra_inter_ratio_post_phase = 1.2
        self.geometry_max_intra_inter_ratio = self.geometry_max_intra_inter_ratio_post_phase
        self.geometry_min_cluster_size = 100
        self.geometry_min_nearest_center_acc = 0.6
        self.rep_supcon_weight = 0.2
        self.rep_supcon_temperature = 0.03
        self.rep_supcon_negative_weight = 1.5
        self.rep_supcon_min_negatives = 10
        self.rep_var_lower_bound = 0.08
        self.rep_var_upper_bound = 0.12
        self.rep_var_clamp_weight = 0.05
        self.rep_pair_margin_distance = 1.2
        self.rep_pair_margin_weight = 0.15
        self.rep_hard_negative_weight = 3.0
        self.rep_adaptive_exit_ratio_threshold = 1.6
        self.rep_adaptive_exit_min_inter_threshold = 0.30
        self.rep_centroid_barrier_min_distance = 0.4
        self.rep_centroid_barrier_weight = 0.5
        self.rep_centroid_repulsion_margin = 0.6
        self.rep_centroid_repulsion_weight = 0.6
        self.rep_critical_pair_weight = 0.0
        self.rep_barrier_activation_fraction = 0.30
        self.rep_expansion_target_min_inter = 0.45
        self.rep_compression_supcon_scale = 0.3
        self.rep_topk_nearest_negatives = 3
        self.rep_min_displacement_eps = 0.05
        self.use_energy_based_family_objective = True
        self.energy_gap_margin = 1.0
        self.energy_gap_weight = 1.0
        self.energy_multi_negative_alpha = 1.0
        self.energy_logit_temperature = 2.0
        self.energy_balance_weight = 0.1
        self.energy_winner_weight = 0.5
        self.energy_winner_min_count = 1
        self.energy_emergence_bias_beta = 0.5
        self.energy_emergence_bias_eps = 1e-3
        self.energy_win_rate_ema_momentum = 0.9
        self.energy_emergence_bias_ratio_min = 0.10
        self.energy_emergence_bias_ratio_max = 0.30
        self.energy_emergence_bias_target_ratio = 0.20
        self.energy_isolate_short_horizon = True
        self._logit_temp = 1.0
        self._temperature_calibration = 1.0
        self._temperature_calibration_lr = 1e-3
        self.rep_epoch_feature_chunks: list[torch.Tensor] = []
        self.rep_epoch_label_chunks: list[torch.Tensor] = []
        self.rep_backbone_grad_scale = 2.0
        self.centroid_ema_momentum = 0.9
        self.class_starvation_streak = 0
        self.critical_collision_pairs: set[tuple[int, int]] = {(0, 3), (0, 4), (3, 4)}
        self.emergency_label_merge_map: dict[int, int] = {3: 0, 4: 0}
        self.representation_balance_target_per_class = 64
        self.enforce_all_classes_per_batch = False
        self.sampler_mode = "interleaved_rr"
        self._base_lr_scales: dict[str, float] = {
            str(param_group.get("group_name", f"group_{idx}")): float(
                param_group.get("lr_scale", 1.0)
            )
            for idx, param_group in enumerate(self.optimizer.param_groups)
        }

        # Initialize scheduler delegates (Phase 13A-2)
        self._phase_manager = PhaseManager(
            representation_only_steps=int(self.representation_only_steps),
            head_only_steps=int(self.head_only_steps),
            representation_diagnostic_mode=bool(self.representation_diagnostic_mode),
            use_energy_based_family_objective=bool(self.use_energy_based_family_objective),
            rep_adaptive_exit_ratio_threshold=float(self.rep_adaptive_exit_ratio_threshold),
            rep_adaptive_exit_min_inter_threshold=float(self.rep_adaptive_exit_min_inter_threshold),
            representation_window_pattern=list(self.representation_window_pattern)
            if self.representation_window_pattern
            else [],
            joint_finetune_backbone_lr_multiplier=float(self.joint_finetune_backbone_lr_multiplier),
            joint_finetune_head_lr_multiplier=float(self.joint_finetune_head_lr_multiplier),
        )
        self._early_stopping_manager = EarlyStoppingManager(
            early_stopping_patience=int(getattr(self.config, "early_stopping_patience", 5)),
            early_stopping_threshold=float(getattr(self.config, "early_stopping_threshold", 0.01)),
            min_family_minority_recall_for_best=float(
                getattr(self.config, "min_family_minority_recall_for_best", 0.3)
            ),
            disable_integrity_hard_stops=bool(getattr(self, "disable_integrity_hard_stops", False)),
        )
        self._freeze_manager = FreezeManager()
        self._freeze_manager.backbone_frozen = bool(self.backbone_frozen)
        self._lr_scheduler = LRScheduler(
            learning_rate=float(self.config.learning_rate),
            warmup_epochs=int(getattr(self.config, "warmup_epochs", 0)),
            warmup_init_lr=float(getattr(self.config, "warmup_init_lr", 1e-6)),
            epochs=int(self.config.epochs),
        )

        # Training state
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.best_model_state: Optional[dict[str, torch.Tensor]] = None
        self.patience_counter = 0
        self.val_gap_collapse_streak = 0
        self.entropy_missing_class_streak = 0
        self.entropy_collapse_streak = 0
        self.high_accuracy_high_loss_streak = 0
        self.training_history: dict[str, list[float]] = {
            "train_loss": [],
            "train_binary_acc": [],
            "train_family_acc": [],
            "train_family_logit_max": [],
            "train_family_logit_min": [],
            "val_loss": [],
            "val_binary_acc": [],
            "val_family_acc": [],
            "val_binary_auroc": [],
            "val_binary_auprc": [],
            "val_family_macro_f1": [],
            "val_family_minority_recall_min": [],
            "val_family_entropy": [],
        }

        # Create evaluation orchestrator (Phase 16)
        self._evaluation_orchestrator = EvaluationOrchestrator(
            model=self.model,
            device=self.device,
            loss_fn=self.loss_fn,
            binary_class_weights=self.binary_class_weights,
            family_class_weights=self.family_class_weights,
            logger=self.logger,
            family_log_prior=self.family_log_prior,
            use_energy_based_family_objective=self.use_energy_based_family_objective,
            energy_logit_temperature=self.energy_logit_temperature,
            active_family_class_ids=self.active_family_class_ids,
            class4_logit_shift=self.class4_logit_shift,
            class4_logit_shift_class_id=self.class4_logit_shift_class_id,
            disable_integrity_hard_stops=bool(
                getattr(self, "disable_integrity_hard_stops", False)
            ),
        )

        # Create validation orchestrator (Phase 16)
        self._validation_orchestrator = ValidationOrchestrator()

        # Phase 13A-1: diagnostics delegates
        from scripts.training.diagnostics import (  # noqa: F811
            ClusterAnalyzer,
            GeometryAnalyzer,
            RepresentationDiagnostics,
        )

        self._geometry_analyzer = GeometryAnalyzer(
            use_energy_based_family_objective=self.use_energy_based_family_objective,
            geometry_min_cluster_size=self.geometry_min_cluster_size,
            critical_collision_pairs=self.critical_collision_pairs,
            geometry_min_inter_threshold=self.geometry_min_inter_threshold,
            geometry_max_intra_inter_ratio_warmup=self.geometry_max_intra_inter_ratio_warmup,
            geometry_max_intra_inter_ratio_post_phase=self.geometry_max_intra_inter_ratio_post_phase,
            logger=self.logger,
        )

        self._cluster_analyzer = ClusterAnalyzer(
            model=self.model,
            device=self.device,
            logger=self.logger,
            cluster_relabel_objective=self.cluster_relabel_objective,
            cluster_relabel_seed=self.cluster_relabel_seed,
            cluster_relabel_spectral_affinity=self.cluster_relabel_spectral_affinity,
        )

        self._rep_diagnostics = RepresentationDiagnostics(
            model=self.model,
            device=self.device,
            logger=self.logger,
            representation_only_steps=self.representation_only_steps,
            head_only_steps=self.head_only_steps,
            sampler_mode=self.sampler_mode,
        )

        # Phase 13A-3: representation delegates
        self._centroid_manager = CentroidManager(
            centroid_ema_momentum=float(self.centroid_ema_momentum),
        )

        # Phase 15: orchestration layer
        self._phase_orchestrator = PhaseOrchestrator(
            model=self.model,
            optimizer=self.optimizer,
            logger=self.logger,
            base_lr_scales=self._base_lr_scales,
            phase_manager=self._phase_manager,
            early_stopping_manager=self._early_stopping_manager,
            geometry_analyzer=self._geometry_analyzer,
            cluster_analyzer=self._cluster_analyzer,
            centroid_manager=self._centroid_manager,
            rep_diagnostics=self._rep_diagnostics,
            representation_only_steps=int(self.representation_only_steps),
            head_only_steps=int(self.head_only_steps),
            representation_diagnostic_mode=bool(self.representation_diagnostic_mode),
            use_energy_based_family_objective=bool(self.use_energy_based_family_objective),
            rep_adaptive_exit_ratio_threshold=float(self.rep_adaptive_exit_ratio_threshold),
            rep_adaptive_exit_min_inter_threshold=float(self.rep_adaptive_exit_min_inter_threshold),
            joint_finetune_backbone_lr_multiplier=float(self.joint_finetune_backbone_lr_multiplier),
            joint_finetune_head_lr_multiplier=float(self.joint_finetune_head_lr_multiplier),
            cluster_relabeling_enabled=bool(self.cluster_relabeling_enabled),
            cluster_relabel_k=self.cluster_relabel_k,
            cluster_relabel_seed=int(self.cluster_relabel_seed),
            cluster_relabel_objective=str(self.cluster_relabel_objective),
            cluster_relabel_spectral_affinity=str(self.cluster_relabel_spectral_affinity),
            critical_collision_pairs=self.critical_collision_pairs,
            emergency_label_merge_map=self.emergency_label_merge_map,
            disable_integrity_hard_stops=bool(getattr(self, "disable_integrity_hard_stops", False)),
            min_family_minority_recall_for_best=float(
                getattr(self.config, "min_family_minority_recall_for_best", 0.3)
            ),
            quality_gate_entropy=0.3,
        )
        self._representation_coordinator = RepresentationCoordinator()
        self._loss_registry = LossRegistry(
            entropy_warmup_steps=self.entropy_warmup_steps,
            entropy_warmup_weight=self.entropy_warmup_weight,
            kl_uniform_weight=self.kl_uniform_weight,
            warmup_kl_uniform_weight=self.warmup_kl_uniform_weight,
            logit_floor=self.logit_floor,
            logit_floor_weight=self.logit_floor_weight,
            tail_ce_weight=self.tail_ce_weight,
            tail_class_mask=self.tail_class_mask,
            loss_fn=self.loss_fn,
            energy_gap_weight=self.energy_gap_weight,
            energy_multi_negative_alpha=self.energy_multi_negative_alpha,
            energy_balance_weight=self.energy_balance_weight,
            energy_winner_weight=self.energy_winner_weight,
            energy_winner_min_count=self.energy_winner_min_count,
            energy_logit_temperature=self.energy_logit_temperature,
            energy_win_rate_ema_momentum=self.energy_win_rate_ema_momentum,
            energy_emergence_bias_eps=self.energy_emergence_bias_eps,
            energy_emergence_bias_beta=self.energy_emergence_bias_beta,
            energy_emergence_bias_ratio_max=self.energy_emergence_bias_ratio_max,
        )

        # Create training execution components (Phase 17)
        self._batch_processor = BatchProcessor(
            model=self.model,
            loss_fn=self.loss_fn,
            loss_registry=self._loss_registry,
            config=self.config,
        )
        self._warmup_manager = WarmupManager(
            model=self.model,
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
            device=self.device,
            config=self.config,
        )
        self._epoch_runner = EpochRunner(
            model=self.model,
            train_loader=self.train_loader,
            config=self.config,
            device=self.device,
            logger=self.logger,
            batch_processor=self._batch_processor,
            warmup_manager=self._warmup_manager,
        )
        self._training_orchestrator = TrainingOrchestrator(
            config=self.config,
            logger=self.logger,
            epoch_runner=self._epoch_runner,
        )

        # Phase 18: Recovery manager for structure recovery configuration
        from scripts.training.core import TrainerState

        self._trainer_state = TrainerState(
            model=self.model,
            train_loader=self.train_loader,
            val_loaders=self.val_loaders,
            test_loaders=self.test_loaders,
            optimizer=self.optimizer,
            loss_fn=self.loss_fn,
            config=self.config,
            binary_class_weights=self.binary_class_weights,
            family_class_weights=self.family_class_weights,
            train_family_class_count=self.train_family_class_count,
            run_seed=self.run_seed,
            device=self.device,
            logger=self.logger,
        )
        self._recovery_manager = RecoveryManager(
            state=self._trainer_state,
            loss_registry=self._loss_registry,
            centroid_manager=self._centroid_manager,
            logger=self.logger,
        )

    def _should_exit_representation_curriculum(self) -> bool:
        """Return True once adaptive geometry targets are met for phase-1 termination.

        Delegated to PhaseOrchestrator (Phase 15).
        """
        result = self._phase_orchestrator.should_exit_representation_curriculum(
            rep_phase_feature_chunks=self.rep_phase_feature_chunks,
            rep_phase_label_chunks=self.rep_phase_label_chunks,
            global_step=self.global_step,
            val_loaders=self.val_loaders,
            active_family_class_ids=self.active_family_class_ids,
            run_seed=self.run_seed,
        )
        # Sync diagnostic update back to trainer state
        diag_update = result.get("representation_diagnostics_update", {})
        if diag_update:
            self.representation_diagnostics.update(diag_update)
        return bool(result.get("should_exit", False))

    def _stabilize_centroids(
        self,
        batch_centroids: torch.Tensor,
        class_ids: list[int],
    ) -> torch.Tensor:
        """Apply centroid EMA smoothing via centroid_manager (Phase 13A-3)."""
        return self._centroid_manager.stabilize_centroids(batch_centroids, class_ids)

    def configure_family_controls(
        self,
        *,
        family_class_priors: Optional[torch.Tensor],
        tail_class_mask: Optional[torch.Tensor],
        balance_strategy: str = "focal",
        focal_warmup_epochs: int = 0,
        warmup_ratio: float = 0.12,
        train_temperature: float = 1.5,
    ) -> None:
        """Configure class-balance strategy and optional log-prior correction controls."""
        strategy = str(balance_strategy).strip().lower()
        if strategy not in {"weighted_ce", "focal"}:
            raise ValueError(f"Unsupported balance strategy: {balance_strategy!r}")

        self.base_balance_strategy = strategy
        self.focal_warmup_epochs = max(0, int(focal_warmup_epochs))
        self.warmup_ratio = min(0.5, max(0.0, float(warmup_ratio)))
        self.warmup_steps = max(1, int(math.ceil(self.total_train_steps * self.warmup_ratio)))
        self.train_temperature = max(1.0, float(train_temperature))
        self.family_log_prior = None
        self.tail_class_mask = None

        if tail_class_mask is not None:
            self.tail_class_mask = tail_class_mask.to(device=self.device, dtype=torch.bool)

        if family_class_priors is not None:
            priors = family_class_priors.to(device=self.device, dtype=torch.float32)
            prior_sum = float(priors.sum().item())
            if prior_sum <= 0.0:
                raise ValueError("family_class_priors must sum to > 0")
            normalized = torch.clamp(priors / prior_sum, min=1e-12, max=1.0)
            self.family_log_prior = torch.log(normalized).unsqueeze(0)

    def configure_structure_recovery(  # NOSONAR
        self,
        *,
        active_family_classes: set[int],
        supcon_weight: float,
        supcon_temperature: float,
        step_coverage_check_step: int,
        representation_diagnostic_mode: bool,
        phase_settings: dict[str, Any],
        cluster_relabeling_enabled: bool,
        cluster_relabel_k: Optional[int],
        cluster_relabel_seed: int,
        cluster_relabel_objective: str,
        cluster_relabel_spectral_affinity: str,
    ) -> None:
        """Configure structural anti-collapse constraints for family prediction coverage.

        Delegated to RecoveryManager (Phase 18).
        """
        self._recovery_manager.configure_structure_recovery(
            active_family_classes=active_family_classes,
            supcon_weight=supcon_weight,
            supcon_temperature=supcon_temperature,
            step_coverage_check_step=step_coverage_check_step,
            representation_diagnostic_mode=representation_diagnostic_mode,
            phase_settings=phase_settings,
            cluster_relabeling_enabled=cluster_relabeling_enabled,
            cluster_relabel_k=cluster_relabel_k,
            cluster_relabel_seed=cluster_relabel_seed,
            cluster_relabel_objective=cluster_relabel_objective,
            cluster_relabel_spectral_affinity=cluster_relabel_spectral_affinity,
        )

    def _set_phase_trainability(
        self,
        *,
        train_backbone: bool,
        train_family_head: bool,
        train_family_projection: Optional[bool] = None,
    ) -> None:
        """Toggle trainability for backbone/family head during diagnostic two-phase training.

        Delegates trainability targets to PhaseManager (Phase 13A-2).
        """
        targets = PhaseManager.compute_trainability_targets(
            train_backbone, train_family_head, train_family_projection=train_family_projection
        )
        for param in self.model.backbone.parameters():
            param.requires_grad = targets["train_backbone"]
        if hasattr(self.model, "family_projection"):
            for param in self.model.family_projection.parameters():
                param.requires_grad = targets["train_family_projection"]
        for param in self.model.family_head.parameters():
            param.requires_grad = targets["train_family_head"]

        self.backbone_params = [
            param for param in self.model.backbone.parameters() if param.requires_grad
        ]
        self.logger.info(
            "PhaseTrainability backbone=%s family_projection=%s family_head=%s",
            "train" if targets["train_backbone"] else "frozen",
            "train" if targets["train_family_projection"] else "frozen",
            "train" if train_family_head else "frozen",
        )

    def _set_phase_lr_scales(self, *, backbone_multiplier: float, head_multiplier: float) -> None:
        """Apply phase-specific LR multipliers on top of base group scales.

        Delegates scale targets to LRScheduler (Phase 13A-2).
        """
        scales = LRScheduler.apply_lr_scales(
            self.optimizer.param_groups,
            self._base_lr_scales,
            backbone_multiplier,
            head_multiplier,
        )
        for idx, param_group in enumerate(self.optimizer.param_groups):
            group_name = str(param_group.get("group_name", f"group_{idx}"))
            if group_name in scales:
                param_group["lr_scale"] = scales[group_name]
        self._set_learning_rate()

    def _is_representation_window_step(self, step: int) -> bool:
        """Return whether *step* falls in a representation micro-window.

        Delegates to PhaseOrchestrator (Phase 15).
        """
        return self._phase_orchestrator.is_representation_window_step(step)

    def _prepare_representation_features(self, features: torch.Tensor) -> torch.Tensor:
        """Return L2-normalized embeddings for geometry stages."""
        return self._cluster_analyzer.prepare_representation_features(features)

    @staticmethod
    def _compute_grad_l2_norm(parameters: Any) -> float:
        """Compute total L2 norm for gradients across a parameter iterable."""
        grad_sq_sum = 0.0
        for param in parameters:
            if param.grad is None:
                continue
            grad = param.grad.detach()
            grad_sq_sum += float(torch.sum(grad * grad).item())
        return math.sqrt(max(0.0, grad_sq_sum))

    def _scale_backbone_gradients(self, scale: float) -> None:
        """Scale backbone gradients in-place during representation-only phase."""
        scale_value = float(scale)
        if math.isclose(scale_value, 1.0, rel_tol=0.0, abs_tol=1e-12):
            return
        for param in self.model.backbone.parameters():
            if param.grad is not None:
                param.grad.mul_(scale_value)

    def _intra_class_variance_clamp_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Penalize exploding and collapsing per-class embedding variance (delegated to LossRegistry Phase 13A-3)."""
        return LossRegistry.intra_class_variance_clamp_loss(
            features, labels,
            var_lower_bound=self.rep_var_lower_bound,
            var_upper_bound=self.rep_var_upper_bound,
        )

    def _compute_class_centroids(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, list[int]]:
        """Compute detached class centroids from normalized embeddings."""
        return self._cluster_analyzer.compute_class_centroids(features, labels)

    @staticmethod
    def _compute_batch_class_centroids_for_loss(
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, list[int]]:
        """Compute differentiable per-class centroids (delegated to LossRegistry Phase 13A-3)."""
        return LossRegistry.compute_batch_class_centroids_for_loss(features, labels)

    def _update_running_rep_centroids(
        self,
        batch_centroids: torch.Tensor,
        class_ids: list[int],
    ) -> None:
        """Update running per-class centroids via centroid_manager (Phase 13A-3)."""
        self._centroid_manager.update_running_rep_centroids(batch_centroids, class_ids)

    def _freeze_epoch_centroid_snapshot(self) -> None:
        """Freeze centroid reference frame via centroid_manager (Phase 13A-3)."""
        self._centroid_manager.freeze_epoch_centroid_snapshot()

    def _update_centroids_from_epoch_buffer(self) -> None:
        """Update running centroid EMA from epoch buffers via centroid_manager (Phase 13A-3)."""
        self._centroid_manager.update_centroids_from_epoch_buffer(
            self.rep_epoch_feature_chunks,
            self.rep_epoch_label_chunks,
            self._cluster_analyzer,
        )

    def _global_centroid_guided_losses(
        self,
        batch_centroids: torch.Tensor,
        class_ids: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Compute centroid forces against running global centroids (delegated to LossRegistry Phase 13A-3)."""
        return LossRegistry.global_centroid_guided_losses(
            batch_centroids, class_ids,
            self._centroid_manager.epoch_frozen_centroids,
            rep_centroid_repulsion_margin=self.rep_centroid_repulsion_margin,
            rep_centroid_barrier_min_distance=self.rep_centroid_barrier_min_distance,
        )

    def _critical_pair_centroid_push_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        *,
        min_distance: float,
    ) -> torch.Tensor:
        """Apply direct centroid push for known critically colliding class pairs (delegated to LossRegistry Phase 13A-3)."""
        return LossRegistry.critical_pair_centroid_push_loss(
            features, labels,
            min_distance=min_distance,
            critical_collision_pairs=self.critical_collision_pairs,
        )

    def _current_geometry_ratio_threshold(self) -> float:
        """Return stage-aware geometry ratio threshold."""
        return self._geometry_analyzer.current_geometry_ratio_threshold(
            representation_phase_active=self.representation_phase_active,
            head_phase_start_step=self.head_phase_start_step,
        )

    def _build_representation_snapshot_id(
        self,
        diagnostics: dict[str, Any],
        *,
        label_space: str,
    ) -> str:
        """Build a stable snapshot ID for post-representation geometry state."""
        return self._rep_diagnostics.build_representation_snapshot_id(
            diagnostics,
            label_space=label_space,
            representation_only_steps=self.representation_only_steps,
            head_only_steps=self.head_only_steps,
            sampler_mode=self.sampler_mode,
        )

    def _maybe_activate_joint_finetune_phase(self) -> None:
        """Enable low-LR joint tuning after the head-only stage completes.

        Full delegation to PhaseOrchestrator (Phase 15).
        """
        if self._phase_orchestrator.maybe_activate_joint_finetune_phase(
            global_step=self.global_step,
            representation_diagnostics=self.representation_diagnostics,
        ):
            self.joint_finetune_active = True
            self.logger.info(
                "PhaseTrainability backbone=train family_head=train (joint_low_lr step=%d)",
                int(self.global_step),
            )

    def _rebalance_representation_batch(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        *,
        target_per_class: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rebalance representation-phase batches (delegated to RepresentationCoordinator Phase 13A-3)."""
        return self._representation_coordinator.rebalance_representation_batch(
            features, labels,
            target_per_class=target_per_class,
            run_seed=self.run_seed,
            global_step=self.global_step,
        )

    def _critical_pair_key(self, class_i: int, class_j: int) -> tuple[int, int]:
        """Return normalized class-pair key."""
        return self._geometry_analyzer.critical_pair_key(class_i, class_j)

    def _has_critical_collision_pairs(self, diagnostics: dict[str, Any]) -> bool:
        """Check whether critical collision pairs remain unresolved."""
        return self._geometry_analyzer.has_critical_collision_pairs(diagnostics)

    def _apply_emergency_label_merge(
        self,
        labels: torch.Tensor,
        *,
        merge_map: dict[int, int],
    ) -> torch.Tensor:
        """Merge critically colliding classes before classifier-head phase."""
        if not merge_map:
            return labels

        out = labels.detach().clone().to(dtype=torch.int64)
        for src_class, dst_class in merge_map.items():
            out[out == int(src_class)] = int(dst_class)
        return out

    def _enforce_geometry_integrity(
        self,
        diagnostics: dict[str, Any],
        *,
        label_space: str,
    ) -> None:
        """Fail fast when embedding geometry is not classifier-ready."""
        self._geometry_analyzer.enforce_geometry_integrity(
            diagnostics,
            label_space=label_space,
        )

    def _collect_normalized_embeddings(
        self,
        loader: DataLoader,
        *,
        max_batches: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Collect L2-normalized backbone embeddings and family labels from a loader."""
        return self._rep_diagnostics.collect_normalized_embeddings(loader, max_batches=max_batches)

    def _embed_feature_matrix(
        self,
        features: np.ndarray,
        *,
        batch_size: int = 4096,
    ) -> torch.Tensor:
        """Project feature matrix through backbone and return normalized embeddings."""
        return self._cluster_analyzer.embed_feature_matrix(features, batch_size=batch_size)

    def _assign_labels_from_centers(
        self,
        embeddings: torch.Tensor,
        centers: torch.Tensor,
    ) -> torch.Tensor:
        """Assign nearest-center cluster labels for embeddings."""
        return self._cluster_analyzer.assign_labels_from_centers(embeddings, centers)

    def _fit_embedding_clusters(
        self,
        embeddings: torch.Tensor,
        *,
        n_clusters: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fit KMeans clusters on normalized embeddings and return labels/centers."""
        return self._cluster_analyzer.fit_embedding_clusters(embeddings, n_clusters=n_clusters)

    def _build_cluster_label_bridge(
        self,
        old_labels: torch.Tensor,
        cluster_labels: torch.Tensor,
        *,
        n_clusters: int,
    ) -> dict[str, Any]:
        """Build stable bridge metadata from legacy labels to cluster labels."""
        return self._cluster_analyzer.build_cluster_label_bridge(
            old_labels, cluster_labels, n_clusters=n_clusters
        )

    def _apply_cluster_relabels_to_datasets(
        self,
        centers: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Relabel train/val/test datasets by nearest embedding cluster centers."""
        return self._cluster_analyzer.apply_cluster_relabels_to_datasets(
            centers,
            train_loader=self.train_loader,
            val_loaders=self.val_loaders,
            test_loaders=self.test_loaders,
        )

    def _nearest_center_accuracy(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        centers: dict[int, torch.Tensor],
        class_ids: list[int],
    ) -> float:
        """Compute nearest-center classification accuracy."""
        return self._rep_diagnostics.nearest_center_accuracy(features, labels, centers, class_ids)

    def _build_class_centers(
        self,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        class_ids: list[int],
    ) -> tuple[dict[int, torch.Tensor], list[int]]:
        """Build centers for available classes."""
        return self._cluster_analyzer.build_class_centers(train_features, train_labels, class_ids)

    def _compute_inter_and_intra_distances(
        self,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        centers: dict[int, torch.Tensor],
        dist_mat: torch.Tensor,
        available_class_ids: list[int],
        collision_threshold: float,
    ) -> tuple[list[float], list[dict[str, Any]], list[float]]:
        """Compute inter-class and intra-class distances."""
        return self._geometry_analyzer.compute_inter_and_intra_distances(
            train_features,
            train_labels,
            centers,
            dist_mat,
            available_class_ids,
            collision_threshold,
        )

    def _estimate_local_density_diagnostics(
        self,
        train_features: torch.Tensor,
        *,
        k: int = 20,
        max_samples: int = 4096,
    ) -> dict[str, Any]:
        """Estimate k-NN density stability on embedding space."""
        return self._geometry_analyzer.estimate_local_density_diagnostics(
            train_features,
            k=k,
            max_samples=max_samples,
            run_seed=self.run_seed,
        )

    def _compute_center_pair_diagnostics(
        self,
        dist_mat: torch.Tensor,
        available_class_ids: list[int],
    ) -> tuple[list[dict[str, Any]], float, list[dict[str, Any]]]:
        """Compute pairwise center diagnostics and percentile-based threshold."""
        return self._rep_diagnostics.compute_center_pair_diagnostics(dist_mat, available_class_ids)

    def _compute_representation_diagnostics(
        self,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        val_features: torch.Tensor,
        val_labels: torch.Tensor,
        *,
        class_ids: list[int],
    ) -> dict[str, Any]:
        """Compute center distances, nearest-center accuracy, and intra/inter separability.

        Includes geometric representation Fixes 4-6:
        - Fix 4: Secondary collision detection
        - Fix 5: Nearest-center confusion matrix
        - Fix 6: Embedding capacity assessment
        """
        return self._rep_diagnostics.compute_representation_diagnostics(
            train_features,
            train_labels,
            val_features,
            val_labels,
            class_ids=class_ids,
            geometry_analyzer=self._geometry_analyzer,
            run_seed=self.run_seed,
        )

    def _run_representation_diagnostics(
        self,
        *,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        label_space: str,
    ) -> dict[str, Any]:
        """Run mandatory embedding diagnostics before classifier-head phase."""
        diagnostics = self._rep_diagnostics.run_representation_diagnostics(
            train_features=train_features,
            train_labels=train_labels,
            label_space=label_space,
            active_family_class_ids=self.active_family_class_ids,
            val_loaders=self.val_loaders,
            geometry_analyzer=self._geometry_analyzer,
            run_seed=self.run_seed,
        )
        if label_space and diagnostics:
            self.representation_diagnostics[str(label_space)] = diagnostics
        return diagnostics

    def _active_balance_strategy(self) -> str:
        """Return the loss strategy for the current epoch."""
        if self.base_balance_strategy == "focal" and self.epoch < self.focal_warmup_epochs:
            return "weighted_ce"
        return self.base_balance_strategy

    def _set_epoch_loss_strategy(self) -> str:
        """Synchronize loss strategy with two-stage focal warmup schedule."""
        strategy = self._active_balance_strategy()
        if hasattr(self.loss_fn, "balance_strategy"):
            self.loss_fn.balance_strategy = strategy
        return strategy

    def _apply_family_logit_controls(
        self,
        family_logits: torch.Tensor,
        *,
        apply_prior: bool = True,
        apply_emergence_bias: bool = True,
        active_class_ids: Optional[list[int]] = None,
    ) -> torch.Tensor:
        """Apply temperature and optional log-prior correction before objectives/metrics."""
        controlled_logits = family_logits

        # Pre-argmax emergence bias must be injected in raw-logit space.
        if self.use_energy_based_family_objective and apply_emergence_bias:
            controlled_logits = controlled_logits + self._compute_energy_emergence_bias(
                controlled_logits,
                active_class_ids=active_class_ids,
            )

        if self.use_energy_based_family_objective:
            controlled_logits = controlled_logits / max(1e-6, float(self.energy_logit_temperature))

        if apply_prior and self.family_log_prior is not None:
            if int(self.family_log_prior.shape[-1]) != int(family_logits.shape[-1]):
                raise RuntimeError(
                    "family prior dimension mismatch: "
                    f"priors={int(self.family_log_prior.shape[-1])} logits={int(family_logits.shape[-1])}"
                )
            controlled_logits = controlled_logits - self.family_log_prior

        return controlled_logits

    def _ensure_energy_win_rate_ema(
        self,
        class_count: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Initialize or resize the EMA tracker for per-class argmax win rates (delegated to LossRegistry Phase 14)."""
        return self._loss_registry.ensure_energy_win_rate_ema(
            class_count, device=device, dtype=dtype,
        )

    def _update_energy_win_rate_ema(
        self,
        logits: torch.Tensor,
        *,
        active_class_ids: Optional[list[int]] = None,
    ) -> None:
        """Update EMA win-rate estimate from hard argmax class wins (delegated to LossRegistry Phase 14)."""
        # Guard: validate input (delegation pre-check)
        if int(logits.ndim) != 2 or int(logits.shape[0]) <= 0:
            return
        self._loss_registry.update_energy_win_rate_ema(
            logits,
            active_class_ids=active_class_ids,
        )

    def _compute_energy_emergence_bias(
        self,
        logits: torch.Tensor,
        *,
        active_class_ids: Optional[list[int]] = None,
    ) -> torch.Tensor:
        """Compute per-class pre-argmax bias from inverse EMA win rates (delegated to LossRegistry Phase 14)."""
        return self._loss_registry.compute_energy_emergence_bias(
            logits,
            active_class_ids=active_class_ids,
            active_family_class_ids=self.active_family_class_ids,
        )

    def _reseed_epoch_generators(self) -> None:
        """Reseed DataLoader generators deterministically per epoch."""
        reseed_dataloader_generator(
            cast(Optional[torch.Generator], getattr(self.train_loader, "generator", None)),
            seed=self.run_seed,
            epoch=self.epoch,
        )
        for offset, loader in enumerate(self.val_loaders.values(), start=1):
            reseed_dataloader_generator(
                cast(Optional[torch.Generator], getattr(loader, "generator", None)),
                seed=self.run_seed + offset,
                epoch=self.epoch,
            )
        for offset, loader in enumerate(self.test_loaders.values(), start=101):
            reseed_dataloader_generator(
                cast(Optional[torch.Generator], getattr(loader, "generator", None)),
                seed=self.run_seed + offset,
                epoch=self.epoch,
            )

    @staticmethod
    def _compute_f1_stats_from_confusion(confusion: torch.Tensor) -> dict[str, Any]:
        """Compute F1-related statistics from confusion matrix counts.

        Delegates to HelixFullEvaluator (Phase 12B-6).
        """
        return HelixFullEvaluator._compute_f1_stats_from_confusion(confusion)

    def _get_learning_rate(self) -> float:
        """Compute learning rate with linear warmup and cosine decay.

        Delegates to LRScheduler (Phase 13A-2).
        """
        return self._lr_scheduler.get_learning_rate(self.epoch)

    def _set_learning_rate(self) -> None:
        """Update learning rate in optimizer.

        Computes LR via LRScheduler (Phase 13A-2) and applies to param groups.
        """
        lr = self._lr_scheduler.get_learning_rate(self.epoch)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _set_backbone_freeze_state(self, freeze_backbone: bool) -> None:
        """Freeze/unfreeze backbone while leaving classifier heads trainable.

        Delegates state tracking to FreezeManager (Phase 13A-2).
        """
        freeze_backbone = bool(freeze_backbone)
        if freeze_backbone == self._freeze_manager.backbone_frozen:
            return

        for param in self.model.backbone.parameters():
            param.requires_grad = not freeze_backbone

        self.backbone_params = [
            param for param in self.model.backbone.parameters() if param.requires_grad
        ]
        self.backbone_frozen = freeze_backbone
        self._freeze_manager.backbone_frozen = freeze_backbone
        self.logger.info(
            "Backbone state updated: %s",
            "frozen" if freeze_backbone else "trainable",
        )

    def _current_learning_rate(self) -> float:
        """Return current optimizer learning rate."""
        for param_group in self.optimizer.param_groups:
            if str(param_group.get("group_name", "")).startswith("backbone"):
                return float(param_group.get("lr", 0.0))
        return float(self.optimizer.param_groups[0].get("lr", 0.0))

    def _apply_loss_regularizations(
        self,
        loss: torch.Tensor,
        family_logits_train: torch.Tensor,
        raw_family_logits: torch.Tensor,
        y_family: torch.Tensor,
        in_step_warmup: bool,
    ) -> torch.Tensor:
        """Apply all loss regularization terms (delegated to LossRegistry Phase 14)."""
        return self._loss_registry.compute_total_loss(
            loss,
            family_logits_train,
            raw_family_logits,
            y_family,
            epoch=self.epoch,
            global_step=self.global_step,
            in_step_warmup=in_step_warmup,
        )

    def _update_train_batch_stats(
        self,
        family_pred: torch.Tensor,
        raw_family_logits: torch.Tensor,
        family_pred_counts: Optional[torch.Tensor],
        family_logit_sums: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Update prediction counts and logit sums for diagnostics."""
        class_count = int(raw_family_logits.shape[1])
        if family_pred_counts is None:
            family_pred_counts = torch.zeros(class_count, dtype=torch.int64)
            family_logit_sums = torch.zeros(class_count, dtype=torch.float32)

        family_pred_counts += torch.bincount(
            family_pred.detach().to(device="cpu", dtype=torch.int64),
            minlength=class_count,
        )

        if family_logit_sums is not None:
            family_logit_sums += raw_family_logits.detach().to(
                device="cpu", dtype=torch.float32
            ).sum(dim=0)

        return family_pred_counts, family_logit_sums

    def _log_step10_diagnostics(self, raw_family_logits: torch.Tensor) -> None:
        """Log per-class logit statistics at step 10."""
        if (not self.step10_symmetry_logged) and self.global_step >= 10:
            mean_logits = raw_family_logits.detach().to(
                device="cpu", dtype=torch.float32
            ).mean(dim=0)
            std_logits = raw_family_logits.detach().to(
                device="cpu", dtype=torch.float32
            ).std(dim=0)
            mean_payload = {int(i): float(v) for i, v in enumerate(mean_logits.tolist())}
            std_payload = {int(i): float(v) for i, v in enumerate(std_logits.tolist())}
            self.logger.info(
                "Step10Diag: per_class_mean_logit=%s per_class_std_logit=%s",
                mean_payload,
                std_payload,
            )
            self.step10_symmetry_logged = True

    def _log_batch_progress(
        self,
        batch_idx: int,
        total_steps: int,
        step_log_interval: int,
        total_loss: float,
        total_binary_correct: int,
        total_family_correct: int,
        total_samples: int,
    ) -> None:
        """Log batch training progress."""
        if batch_idx % step_log_interval == 0:
            avg_loss = total_loss / max(1, total_samples)
            binary_acc = total_binary_correct / max(1, total_samples)
            family_acc = total_family_correct / max(1, total_samples)
            lr = self._current_learning_rate()
            print(
                f"step {batch_idx}/{total_steps} loss {avg_loss:.4f} "
                f"binary_acc {binary_acc:.4f} family_acc {family_acc:.4f}",
                flush=True,
            )
            self.logger.info(
                f"Epoch {self.epoch} [{batch_idx}/{total_steps}] "
                f"Loss: {avg_loss:.4f} | "
                f"Binary Acc: {binary_acc:.4f} | "
                f"Family Acc: {family_acc:.4f} | "
                f"LR: {lr:.2e}"
            )

    def _apply_cluster_relabeling(self, rep_features: torch.Tensor, rep_labels: torch.Tensor) -> None:
        """Apply cluster relabeling to representation phase features.

        Delegated to PhaseOrchestrator (Phase 15).
        """

    @staticmethod
    def _as_python_int(value: Any) -> int:
        """Convert tensor/scalar values to Python int."""
        if isinstance(value, torch.Tensor):
            return int(value.item())
        return int(value)

    def _collect_class_to_indices(self, train_dataset: Dataset[Any]) -> dict[int, list[int]]:
        """Build class -> dataset-index mapping for warmup class coverage."""
        class_to_indices: dict[int, list[int]] = defaultdict(list)
        if hasattr(train_dataset, "family_labels"):
            labels_np = np.asarray(train_dataset.family_labels, dtype=np.int64)
            for idx, class_id in enumerate(labels_np.tolist()):
                class_to_indices[int(class_id)].append(int(idx))
            return class_to_indices

        dataset_size = int(len(cast(Any, train_dataset)))
        for idx in range(dataset_size):
            sample = train_dataset[idx]
            y_family_item = self._as_python_int(sample[2])
            class_to_indices[y_family_item].append(int(idx))
        return class_to_indices

    def _resolve_warmup_active_class_ids(
        self,
        class_to_indices: dict[int, list[int]],
        class_count: int,
    ) -> list[int]:
        """Resolve valid active class ids for warmup coverage."""
        if self.active_family_class_ids:
            active_class_ids = sorted(
                int(cls) for cls in self.active_family_class_ids if 0 <= int(cls) < class_count
            )
        else:
            active_class_ids = sorted(
                int(cls) for cls in class_to_indices.keys() if 0 <= int(cls) < class_count
            )
        if not active_class_ids:
            raise RuntimeError("No active classes available for epoch-0 coverage warmup")
        return active_class_ids

    def _build_warmup_batch_tensors(
        self,
        train_dataset: Dataset[Any],
        forced_indices: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
        """Materialize forced warmup batch tensors from selected dataset indices."""
        x_rows: list[torch.Tensor] = []
        y_binary_rows: list[int] = []
        y_family_rows: list[int] = []

        for idx in forced_indices:
            x_item, y_binary_item, y_family_item = train_dataset[idx]
            x_rows.append(x_item if isinstance(x_item, torch.Tensor) else torch.tensor(x_item))
            y_binary_rows.append(self._as_python_int(y_binary_item))
            y_family_rows.append(self._as_python_int(y_family_item))

        x_forced = torch.stack(x_rows, dim=0).to(self.device, non_blocking=True)
        y_binary_forced = torch.tensor(y_binary_rows, dtype=torch.long, device=self.device)
        y_family_forced = torch.tensor(y_family_rows, dtype=torch.long, device=self.device)
        return x_forced, y_binary_forced, y_family_forced, y_family_rows

    def _run_epoch0_forced_coverage_warmup(self) -> None:
        """Run one epoch-0 synthetic warmup step.

        FULL delegation to WarmupManager (Phase 17).
        """
        result = self._warmup_manager.run_warmup(
            self.train_loader.dataset,
            epoch=self.epoch,
            model_training=self.model.training,
            global_step=self.global_step,
            warmup_steps=self.warmup_steps,
            active_family_class_ids=(
                list(self.active_family_class_ids)
                if self.active_family_class_ids
                else None
            ),
            use_energy_based_family_objective=self.use_energy_based_family_objective,
            binary_class_weights=self.binary_class_weights,
            family_class_weights=self.family_class_weights,
            backbone_params=self.backbone_params,
            logger=self.logger,
        )
        if result["warmup_executed"]:
            self.global_step += result["global_step_increment"]

    def _resolve_batch_active_family_class_ids(
        self,
        raw_family_logits: torch.Tensor,
        y_family: torch.Tensor,
    ) -> list[int]:
        """Resolve active class ids for a train batch."""
        class_count = int(raw_family_logits.shape[1])
        if self.active_family_class_ids:
            return sorted(
                int(cls)
                for cls in self.active_family_class_ids
                if 0 <= int(cls) < class_count
            )
        return sorted(int(cls) for cls in torch.unique(y_family.detach(), dim=0).tolist())

    def _stabilize_batch_family_logits(
        self,
        raw_family_logits: torch.Tensor,
        *,
        active_family_class_ids: list[int],
    ) -> torch.Tensor:
        """Apply logits controls and lightweight temperature stabilization."""
        family_logits = self._apply_family_logit_controls(
            raw_family_logits,
            apply_prior=False,
            apply_emergence_bias=bool(self.use_energy_based_family_objective),
            active_class_ids=active_family_class_ids,
        )

        current_std = family_logits.detach().std().clamp(min=1e-6)
        self._logit_temp = (0.9 * float(self._logit_temp)) + (0.1 * float(current_std.item()))
        family_logits = family_logits / max(1e-6, float(self._logit_temp))

        calib_temp = max(1e-6, float(self._temperature_calibration))
        family_logits = family_logits / calib_temp
        with torch.no_grad():
            probs_calib = torch.softmax(family_logits, dim=1)
            safe_probs_calib = torch.clamp(probs_calib, min=1e-12, max=1.0)
            class_count = max(2, int(family_logits.shape[1]))
            uniform_logp = -math.log(float(class_count))
            temp_kl = torch.sum(
                probs_calib * (torch.log(safe_probs_calib) - uniform_logp),
                dim=1,
            ).mean()
            self._temperature_calibration = max(
                0.5,
                min(
                    5.0,
                    float(self._temperature_calibration)
                    + (float(self._temperature_calibration_lr) * float(temp_kl.item())),
                ),
            )

        return torch.clamp(family_logits, -10.0, 10.0)

    @staticmethod
    def _compute_tail_focal_loss(
        family_logits_train: torch.Tensor,
        y_family: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal tail stabilization term for classes 3/4 (delegated to LossRegistry Phase 14)."""
        return LossRegistry.compute_tail_focal_loss(family_logits_train, y_family)

    def _compute_representation_energy_objective(
        self,
        *,
        family_logits_train: torch.Tensor,
        y_family: torch.Tensor,
        y_binary: torch.Tensor,
        binary_logits: torch.Tensor,
        active_family_class_ids: list[int],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute representation-phase energy objective (delegated to LossRegistry Phase 13A-3)."""
        return LossRegistry.compute_representation_energy_objective(
            family_logits_train=family_logits_train,
            y_family=y_family,
            y_binary=y_binary,
            binary_logits=binary_logits,
            active_family_class_ids=active_family_class_ids,
            loss_fn=self.loss_fn,
            energy_gap_weight=self.energy_gap_weight,
            energy_multi_negative_alpha=self.energy_multi_negative_alpha,
            energy_balance_weight=self.energy_balance_weight,
            energy_winner_weight=self.energy_winner_weight,
            energy_winner_min_count=self.energy_winner_min_count,
            epoch=self.epoch,
        )

    def _log_energy_gap_diag_if_needed(self, diagnostics: dict[str, float]) -> None:
        """Emit periodic energy diagnostics for representation phase (delegated to LossRegistry Phase 14)."""
        if int(self.global_step) % 20 != 0:
            return
        self.logger.info(
            self._loss_registry.log_energy_gap_diag_message(
                global_step=self.global_step,
                diagnostics=diagnostics,
            )
        )

    @staticmethod
    def _apply_entropy_floor_regularizer(
        loss: torch.Tensor,
        *,
        family_logits_train: torch.Tensor,
        active_class_count: int,
    ) -> torch.Tensor:
        """Apply entropy floor regularization on family logits (delegated to LossRegistry Phase 14)."""
        return LossRegistry.apply_entropy_floor_regularizer_to_loss(
            loss,
            family_logits_train=family_logits_train,
            active_class_count=active_class_count,
        )

    def _backpropagate_train_batch_loss(
        self,
        loss: torch.Tensor,
        *,
        in_representation_phase: bool,
    ) -> None:
        """Run backward pass, optional diagnostics, and optimizer step."""
        self.optimizer.zero_grad()
        loss.backward()

        if self.config.max_grad_norm > 0 and self.backbone_params:
            nn.utils.clip_grad_norm_(self.backbone_params, self.config.max_grad_norm)

        if in_representation_phase:
            backbone_grad_norm = self._compute_grad_l2_norm(self.model.backbone.parameters())
            projection_grad_norm = 0.0
            if hasattr(self.model, "family_projection"):
                projection_grad_norm = self._compute_grad_l2_norm(
                    self.model.family_projection.parameters()
                )
            self.logger.info(
                "Phase1GradDiag step=%d backbone_grad_norm=%.6f projection_grad_norm=%.6f",
                int(self.global_step),
                backbone_grad_norm,
                projection_grad_norm,
            )

        self.optimizer.step()
        self.global_step += 1

    def _compute_loss_with_optional_energy(
        self,
        *,
        classification_loss: torch.Tensor,
        family_logits_train: torch.Tensor,
        y_family: torch.Tensor,
        y_binary: torch.Tensor,
        binary_logits: torch.Tensor,
        in_representation_phase: bool,
        active_family_class_ids: list[int],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute base loss and optionally replace it with representation energy objective (delegated to LossRegistry Phase 14)."""
        return self._loss_registry.compute_loss_with_optional_energy(
            classification_loss=classification_loss,
            family_logits_train=family_logits_train,
            y_family=y_family,
            y_binary=y_binary,
            binary_logits=binary_logits,
            in_representation_phase=in_representation_phase,
            active_family_class_ids=active_family_class_ids,
            use_energy_based_family_objective=self.use_energy_based_family_objective,
            disable_tail_focal_regularizer=bool(getattr(self, "disable_tail_focal_regularizer", False)),
        )

    def _apply_optional_non_representation_regularizations(
        self,
        loss: torch.Tensor,
        *,
        in_representation_phase: bool,
        family_logits_train: torch.Tensor,
        raw_family_logits: torch.Tensor,
        y_family: torch.Tensor,
        in_step_warmup: bool,
    ) -> torch.Tensor:
        """Apply classifier-side regularizations outside representation phase."""
        if in_representation_phase:
            return loss
        return self._apply_loss_regularizations(
            loss,
            family_logits_train,
            raw_family_logits,
            y_family,
            in_step_warmup,
        )

    def _maybe_store_representation_chunks(
        self,
        *,
        in_representation_phase: bool,
        backbone_features: torch.Tensor,
        y_family: torch.Tensor,
    ) -> None:
        """Store detached representation features for representation-phase diagnostics."""
        if not in_representation_phase:
            return
        rep_features = self._prepare_representation_features(backbone_features)
        self.rep_phase_feature_chunks.append(rep_features.detach().to(device="cpu"))
        self.rep_phase_label_chunks.append(
            y_family.detach().to(device="cpu", dtype=torch.int64)
        )

    def _process_train_batch(
        self,
        x: torch.Tensor,
        y_binary: torch.Tensor,
        y_family: torch.Tensor,
        in_step_warmup: bool,
        in_representation_phase: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
        """Process one training batch.

        FULL delegation to BatchProcessor (Phase 17).
        """
        result = self._batch_processor.process_batch(
            x,
            y_binary,
            y_family,
            in_step_warmup=in_step_warmup,
            in_representation_phase=in_representation_phase,
            optimizer=self.optimizer,
            backbone_params=self.backbone_params,
            global_step=self.global_step,
            warmup_steps=self.warmup_steps,
            binary_class_weights=self.binary_class_weights,
            family_class_weights=self.family_class_weights,
            active_family_class_ids=(
                list(self.active_family_class_ids)
                if self.active_family_class_ids
                else []
            ),
            use_energy_based_family_objective=self.use_energy_based_family_objective,
            disable_tail_focal_regularizer=bool(getattr(
                self, "disable_tail_focal_regularizer", False,
            )),
            energy_logit_temperature=self.energy_logit_temperature,
            family_log_prior=self.family_log_prior,
        )
        self.global_step += result["global_step_increment"]

        if in_representation_phase and self.use_energy_based_family_objective:
            self._log_energy_gap_diag_if_needed(result["energy_diag"])

        return (
            result["loss"],
            result["raw_family_logits"],
            result["family_pred"],
            result["binary_correct"],
            result["family_correct"],
            result["batch_size"],
        )

    def _check_backbone_freeze_state(self) -> None:
        """Check and unfreeze backbone if needed.

        Delegates unfreeze decision to FreezeManager (Phase 13A-2).
        """
        if self.representation_diagnostic_mode:
            return
        if self._freeze_manager.should_unfreeze(self.global_step, self.unfreeze_backbone_step):
            self._set_backbone_freeze_state(False)

    def _check_family_class_coverage(self, y_family: torch.Tensor) -> None:
        """Validate that batch contains all expected family classes.

        FULL delegation to ValidationOrchestrator (Phase 16).
        """
        self._validation_orchestrator.check_family_class_coverage(
            y_family,
            active_family_class_ids=self.active_family_class_ids,
            enforce_all_classes_per_batch=self.enforce_all_classes_per_batch,
        )

    def _check_step_coverage(
        self,
        batch_idx: int,
        family_pred_counts: Optional[torch.Tensor],
    ) -> None:
        """Check step coverage for family class predictions.

        FULL delegation to ValidationOrchestrator (Phase 16).
        """
        self.step_coverage_checked, _ = (
            self._validation_orchestrator.check_step_coverage(
                batch_idx,
                family_pred_counts,
                step_coverage_checked=self.step_coverage_checked,
                representation_diagnostic_mode=self.representation_diagnostic_mode,
                head_phase_start_step=self.head_phase_start_step,
                representation_phase_active=self.representation_phase_active,
                global_step=self.global_step,
                coverage_check_after_head_steps=self.coverage_check_after_head_steps,
                step_coverage_check_step=self.step_coverage_check_step,
                active_family_class_ids=self.active_family_class_ids,
                disable_integrity_hard_stops=bool(
                    getattr(self, "disable_integrity_hard_stops", False)
                ),
                use_energy_based_family_objective=self.use_energy_based_family_objective,
                logger=self.logger,
            )
        )

    def _handle_representation_phase_logic(
        self,
        in_representation_phase: bool,
    ) -> None:
        """Handle representation phase initiation and transition logic.

        Delegated to PhaseOrchestrator (Phase 15).
        """
        result = self._phase_orchestrator.handle_representation_phase_logic(
            in_representation_phase=in_representation_phase,
            global_step=self.global_step,
            rep_phase_feature_chunks=self.rep_phase_feature_chunks,
            rep_phase_label_chunks=self.rep_phase_label_chunks,
            val_loaders=self.val_loaders,
            active_family_class_ids=self.active_family_class_ids,
            run_seed=self.run_seed,
        )
        # Sync trainer state from orchestrator result
        self.rep_phase_feature_chunks = result.get("rep_phase_feature_chunks", [])
        self.rep_phase_label_chunks = result.get("rep_phase_label_chunks", [])
        if "representation_phase_active" in result:
            self.representation_phase_active = result["representation_phase_active"]
        if "representation_curriculum_complete" in result:
            self.representation_curriculum_complete = result["representation_curriculum_complete"]
        if "in_representation_window" in result:
            self.in_representation_window = result["in_representation_window"]
        if result.get("transition_executed"):
            if result.get("phase1_class_centroids") is not None:
                self.phase1_class_centroids = result["phase1_class_centroids"]
                self.phase1_centroid_class_ids = result["phase1_centroid_class_ids"]
            if result.get("representation_snapshot_id"):
                self.representation_snapshot_id = result["representation_snapshot_id"]
            if result.get("representation_diagnostics"):
                self.representation_diagnostics.update(result["representation_diagnostics"])
            if result.get("head_phase_start_step", -1) >= 0:
                self.head_phase_start_step = result["head_phase_start_step"]
                self.joint_finetune_start_step = self.head_phase_start_step + self.head_only_steps
            self.step_coverage_checked = False


    def train_epoch(self) -> dict[str, float]:
        """Train for one epoch.

        FULL delegation to EpochRunner (Phase 17).
        """
        # Pre-epoch setup
        self._set_epoch_loss_strategy()
        self._run_epoch0_forced_coverage_warmup()
        if not self.use_energy_based_family_objective:
            self._freeze_epoch_centroid_snapshot()

        result = self._epoch_runner.run_epoch(
            epoch=self.epoch,
            global_step=self.global_step,
            warmup_steps=self.warmup_steps,
            step_log_interval=max(1, int(self.config.log_interval)),
            representation_diagnostic_mode=self.representation_diagnostic_mode,
            use_energy_based_family_objective=self.use_energy_based_family_objective,
            active_family_class_ids=(
                list(self.active_family_class_ids)
                if self.active_family_class_ids
                else []
            ),
            enforce_all_classes_per_batch=self.enforce_all_classes_per_batch,
            step_coverage_checked=self.step_coverage_checked,
            head_phase_start_step=self.head_phase_start_step,
            representation_phase_active=self.representation_phase_active,
            binary_class_weights=self.binary_class_weights,
            family_class_weights=self.family_class_weights,
            optimizer=self.optimizer,
            backbone_params=self.backbone_params,
            binary_class_weights_warmup=None,
            family_class_weights_warmup=None,
            energy_logit_temperature=self.energy_logit_temperature,
            family_log_prior=self.family_log_prior,
            disable_tail_focal_regularizer=bool(getattr(
                self, "disable_tail_focal_regularizer", False,
            )),
            # Trainer-side hooks for stateful operations
            check_backbone_freeze_state=self._check_backbone_freeze_state,
            check_family_class_coverage=self._check_family_class_coverage,
            handle_representation_phase_logic=self._handle_representation_phase_logic,
            maybe_activate_joint_finetune_phase=self._maybe_activate_joint_finetune_phase,
            check_step_coverage=self._check_step_coverage,
            log_step10_diagnostics=self._log_step10_diagnostics,
            log_batch_progress=self._log_batch_progress,
            log_epoch_completion=self._log_epoch_completion,
            update_centroids=self._update_centroids_from_epoch_buffer,
            _process_train_batch_impl=self._process_train_batch,
            _resolve_batch_active_family_class_ids=self._resolve_batch_active_family_class_ids,
            _update_train_batch_stats_impl=self._update_train_batch_stats,
            logger=self.logger,
        )
        self.global_step = result["global_step"]
        if "class_starvation_streak" in result:
            self.class_starvation_streak = result["class_starvation_streak"]
        if "representation_phase_active" in result:
            self.representation_phase_active = result["representation_phase_active"]
        if result.get("step_coverage_checked"):
            self.step_coverage_checked = True

        return {
            "train_loss": result["train_loss"],
            "train_calibrated_loss": result["train_calibrated_loss"],
            "train_binary_acc": result["train_binary_acc"],
            "train_family_acc": result["train_family_acc"],
            "train_family_logit_max": result["train_family_logit_max"],
            "train_family_logit_min": result["train_family_logit_min"],
        }

    def _log_epoch_completion(
        self,
        epoch_start: float,
        train_logit_min: float,
        train_logit_max: float,
        family_pred_counts: Optional[torch.Tensor],
        family_logit_sums: Optional[torch.Tensor],
        total_samples: int,
        top2_logit_gap_sum: float,
        top2_logit_gap_count: int,
    ) -> None:
        """Log metrics at end of training epoch."""
        self.logger.info(
            "Epoch %d complete | elapsed=%.2fs",
            self.epoch,
            time.perf_counter() - epoch_start,
        )
        self.logger.info(
            "Epoch %d logit_range raw_family[min=%.4f max=%.4f]",
            self.epoch,
            train_logit_min,
            train_logit_max,
        )
        avg_top2_logit_gap = top2_logit_gap_sum / max(1, top2_logit_gap_count)
        if family_pred_counts is not None and family_logit_sums is not None and total_samples > 0:
            pred_count_payload = {int(idx): int(count) for idx, count in enumerate(family_pred_counts.tolist())}
            avg_logit_payload = {
                int(idx): float(total / max(1, total_samples))
                for idx, total in enumerate(family_logit_sums.tolist())
            }
            self.logger.info(
                "Epoch %d diagnostics: per_class_prediction_count=%s per_class_avg_logit=%s top2_logit_gap=%.4f",
                self.epoch,
                pred_count_payload,
                avg_logit_payload,
                avg_top2_logit_gap,
            )
        if train_logit_max > 10.0 or train_logit_min < -10.0:
            self.logger.warning(
                "Epoch %d logit saturation risk detected: raw_family[min=%.4f max=%.4f]",
                self.epoch,
                train_logit_min,
                train_logit_max,
            )

    # ================================================================== #
    # Evaluation methods — FULLY delegated to EvaluationOrchestrator      #
    # ================================================================== #

    def _apply_eval_class4_logit_shift(self, family_logits: torch.Tensor) -> torch.Tensor:
        """Apply inference-time class-4 logit shift.

        FULL delegation to EvaluationOrchestrator (Phase 16).
        """
        return self._evaluation_orchestrator.apply_eval_class4_logit_shift(
            family_logits,
            shift=self.class4_logit_shift,
            class_id=self.class4_logit_shift_class_id,
        )

    def _apply_inference_prediction_floor(
        self,
        family_logits: torch.Tensor,
        family_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Inference-only prediction floor to guarantee per-batch class presence.

        FULL delegation to EvaluationOrchestrator (Phase 16).
        """
        return self._evaluation_orchestrator.apply_inference_prediction_floor(
            family_logits, family_pred,
            active_class_ids=self.active_family_class_ids,
        )

    @torch.no_grad()
    def _evaluate_loader(self, loader: DataLoader, dataset_name: str = "unknown") -> dict[str, Any]:  # NOSONAR
        """Evaluate metrics on a single dataset loader.

        FULL delegation to EvaluationOrchestrator (Phase 16).
        """
        return self._evaluation_orchestrator.evaluate_loader(
            loader,
            dataset_name=dataset_name,
            active_family_class_ids=self.active_family_class_ids,
            class4_logit_shift=self.class4_logit_shift,
            class4_logit_shift_class_id=self.class4_logit_shift_class_id,
        )

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Validate per dataset with strict isolation (worst-case aggregation).

        FULL delegation to EvaluationOrchestrator (Phase 16).
        """
        return self._evaluation_orchestrator.validate(
            self.val_loaders,
            active_family_class_ids=self.active_family_class_ids,
            class4_logit_shift=self.class4_logit_shift,
            class4_logit_shift_class_id=self.class4_logit_shift_class_id,
        )

    def _process_test_batch(
        self,
        x: torch.Tensor,
        y_binary: torch.Tensor,
        y_family: torch.Tensor,
        binary_confusion: torch.Tensor,
        family_confusion: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], int, float]:
        """Process one test batch and accumulate metrics.

        FULL delegation to EvaluationOrchestrator (Phase 16).
        """
        return self._evaluation_orchestrator.process_test_batch(
            x, y_binary, y_family, binary_confusion, family_confusion,
            active_family_class_ids=self.active_family_class_ids,
            class4_logit_shift=self.class4_logit_shift,
            class4_logit_shift_class_id=self.class4_logit_shift_class_id,
        )

    @torch.no_grad()
    def _evaluate_test_loader(self, test_loader: DataLoader) -> dict[str, float]:
        """Evaluate one test loader with tensor-first aggregation.

        FULL delegation to EvaluationOrchestrator (Phase 16).
        """
        return self._evaluation_orchestrator.evaluate_test_loader(
            test_loader,
            active_family_class_ids=self.active_family_class_ids,
            class4_logit_shift=self.class4_logit_shift,
            class4_logit_shift_class_id=self.class4_logit_shift_class_id,
        )

    @torch.no_grad()
    def evaluate_per_dataset(self) -> dict[str, dict[str, float]]:
        """Evaluate on per-dataset test sets.

        FULL delegation to EvaluationOrchestrator (Phase 16).
        """
        return self._evaluation_orchestrator.evaluate_per_dataset(
            self.test_loaders,
            active_family_class_ids=self.active_family_class_ids,
            class4_logit_shift=self.class4_logit_shift,
            class4_logit_shift_class_id=self.class4_logit_shift_class_id,
        )

    def fit(self) -> dict[str, Any]:  # NOSONAR
        """Train for specified epochs.

        FULL delegation to TrainingOrchestrator (Phase 17).
        """
        return self._training_orchestrator.fit(
            # Config
            epochs=self.config.epochs,
            val_interval=self.config.val_interval,
            freeze_backbone_epochs=self.freeze_backbone_epochs,
            # Model / device
            model=self.model,
            device=self.device,
            # Stateful harness callbacks
            reseed_generators=self._reseed_epoch_generators,
            set_backbone_freeze_state=lambda frozen: (
                self._set_backbone_freeze_state(frozen)
            ),
            set_learning_rate=self._set_learning_rate,
            train_epoch=self.train_epoch,
            validate=self.validate,
            log_per_dataset_results=self._log_per_dataset_results,
            post_training_macro_floor=self._post_training_macro_floor,
            evaluate_per_dataset=self.evaluate_per_dataset,
            hard_stop_reason=self._hard_stop_reason,
            update_early_stopping=self._update_early_stopping,
            save_checkpoint=self._save_checkpoint_if_needed,
            # Validation callbacks
            detect_coverage_collapse=(
                self._validation_orchestrator.detect_coverage_collapse
            ),
            check_zero_prediction_classes=(
                self._validation_orchestrator.check_zero_prediction_classes
            ),
            check_per_dataset_macro_floor=(
                self._validation_orchestrator.check_per_dataset_macro_floor
            ),
            # Trainer state (mutated during fit)
            epoch=self.epoch,
            training_history=self.training_history,
            best_model_state=self.best_model_state,
            best_val_loss=self.best_val_loss,
            representation_diagnostics=self.representation_diagnostics,
            train_family_class_count=self.train_family_class_count,
            use_energy_based_family_objective=self.use_energy_based_family_objective,
            disable_integrity_hard_stops=bool(
                getattr(self, "disable_integrity_hard_stops", False)
            ),
            logger=self.logger,
        )

    def _post_training_macro_floor(self) -> float:
        """Return macro-F1 floor calibrated to training budget.

        FULL delegation to ValidationOrchestrator (Phase 16).
        """
        return self._validation_orchestrator.post_training_macro_floor(
            epochs=int(getattr(self.config, "epochs", 0)),
        )

    def _is_smoke_mode(self) -> bool:
        """Return True when trainer is running smoke-governance profile.

        Delegated to PhaseOrchestrator (Phase 15).
        """
        return self._phase_orchestrator.is_smoke_mode(
            epochs=getattr(getattr(self, "config", None), "epochs", None),
            gov_profile=os.getenv("HELIX_GOV_POLICY_PROFILE", ""),
        )

    def _hard_stop_reason(
        self,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
    ) -> Optional[str]:
        """Return hard-stop reason when integrity constraints are violated.

        Delegated to PhaseOrchestrator (Phase 15).
        """
        return self._phase_orchestrator.hard_stop_reason(
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            is_smoke=self._is_smoke_mode(),
            epoch=self.epoch,
        )

    def _update_early_stopping(self, _train_metrics: dict[str, float], val_metrics: dict[str, float]) -> bool:
        """Update early stopping state; return True when training should stop.

        Delegated to PhaseOrchestrator (Phase 15).
        """
        result = self._phase_orchestrator.update_early_stopping(
            val_metrics=val_metrics,
        )
        if result["is_best"]:
            self.best_val_loss = result["best_val_loss"]
            self.best_model_state = result["best_model_state"]
        return result["should_stop"]



    def _save_checkpoint_if_needed(self) -> None:
        """Persist intermediate checkpoint on configured interval."""
        if self.epoch <= 0 or self.epoch % self.config.save_interval != 0:
            return
        checkpoint_path = self.config.checkpoint_dir / f"checkpoint_epoch_{self.epoch}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        artifact = _build_model_contract_artifact(
            model_state=self.model.state_dict(),
            feature_order=self.feature_order,
            schema_hash=self.schema_hash,
            extra={"epoch": self.epoch + 1},
        )
        _write_checkpoint_artifact(
            checkpoint_path,
            artifact,
            model_architecture=self.model.__class__.__name__,
            origin="train_helix_ids_full:interval",
        )
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

    def _log_per_dataset_results(self, per_dataset_results: dict[str, dict[str, float]]) -> None:
        """Log formatted per-dataset metrics.

        FULL delegation to ValidationOrchestrator (Phase 16).
        """
        self._validation_orchestrator.log_per_dataset_results(
            self.logger, per_dataset_results,
        )


# ============================================================================
# Main Training Script
# ============================================================================


@governed_entrypoint(entrypoint_id="scripts.train_helix_ids_full")
def main():  # NOSONAR   # Phase 13A-4 orchestration extraction
    """Main training entry point. parse -> orchestrate -> exit pattern."""
    from scripts.training.orchestration import parse_config

    parsed = parse_config()
    args = parsed.args

    # Early exit for multi-seed governance mode
    if parsed.governance_only_mode:
        import json

        from scripts.training.orchestration.governance_pipeline import run_multiseed_governance
        gov_result = run_multiseed_governance(parsed)
        print(json.dumps(gov_result.return_payload, indent=2, default=str))
        return gov_result.return_payload

    import os
    import time
    from pathlib import Path

    from scripts.training.orchestration.run_orchestrator import (
        _assert_real_dataset_required,
        _assert_validated_unsw_artifact,
        _load_precomputed_splits,
    )

    os.environ["HELIX_STRICT_MISSING"] = "1"
    os.environ["STRICT_MISSING"] = "1"
    os.environ["HELIX_SEED"] = str(args.seed)
    set_global_determinism(args.seed)

    split_start = time.perf_counter()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = HELIX_FULL_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(results_dir)

    if args.dataset is not None:
        _assert_real_dataset_required(
            project_root=PROJECT_ROOT,
            dataset_name=args.dataset,
        )

    logger.info("Decoupled training mode enabled.")
    precomputed_splits_dir = Path(args.precomputed_splits_dir)
    precomputed_splits = None
    if not args.force_recompute_splits:
        precomputed_splits = _load_precomputed_splits(
            splits_dir=precomputed_splits_dir,
            logger=logger,
            expected_feature_dim=None,
        )

    if precomputed_splits is not None:
        logger.info(
            f"Using precomputed per-dataset splits from {precomputed_splits_dir} for isolated training."
        )
        require_frozen_snapshot = str(args.snapshot_mode).strip().lower() == "strict"
        if require_frozen_snapshot:
            freeze_meta = freeze_snapshot_if_valid(artifact_dir=precomputed_splits_dir)
            logger.info(
                "Strict snapshot mode: freeze attempted for learnability contract snapshot_id=%s frozen=%s",
                str(freeze_meta.get("snapshot_id", "")),
                bool(freeze_meta.get("frozen", False)),
            )
        else:
            logger.warning(
                "Running with research_override snapshot mode (allow_unfrozen_snapshot=%s); "
                "reproducibility promotion gates remain disabled for this run.",
                bool(args.allow_unfrozen_snapshot),
            )
        _assert_validated_unsw_artifact(
            splits_dir=precomputed_splits_dir,
            logger=logger,
            require_frozen=require_frozen_snapshot,
        )
        splits = precomputed_splits
    else:
        raise RuntimeError(
            "Training requires validated processed artifacts. "
            "Run preprocessing and scripts/validation/validate_unsw_learnability.py first."
        )

    _validate_per_dataset_splits(
        splits,
        logger=logger,
        seed=args.seed,
        enforce_cross_dataset_scale=False,
    )

    split_end = time.perf_counter()
    split_elapsed = split_end - split_start

    # ----------------------------------------------------------------
    # Orchestration
    # ----------------------------------------------------------------
    from scripts.training.orchestration import run_orchestration
    from scripts.training.orchestration.governance_pipeline import run_governance_pipeline

    orchestration_result = run_orchestration(
        parsed=parsed,
        splits=splits,
        results_dir=results_dir,
        output_dir=output_dir,
        logger=logger,
    )

    gov_result = run_governance_pipeline(
        parsed=parsed,
        orchestration_result=orchestration_result,
        results_dir=results_dir,
        output_dir=output_dir,
        logger=logger,
        split_elapsed=split_elapsed,
        splits=splits,
    )

    return gov_result.return_payload

if __name__ == "__main__":
    # Ensure governance profile is resolved before decorator executes main().
    # For smoke diagnostics and A/B baseline bootstrap runs, default to smoke
    # policy unless the operator explicitly sets a governance profile.
    if not os.environ.get("HELIX_GOV_POLICY_PROFILE"):
        argv = sys.argv[1:]
        epochs_override: Optional[int] = None
        ab_mode_enabled = True
        ab_require_baseline = True

        for idx, token in enumerate(argv):
            if token == "--epochs" and idx + 1 < len(argv):
                try:
                    epochs_override = int(argv[idx + 1])
                except ValueError:
                    epochs_override = None
                break
            if token.startswith(("--epochs=",)):
                try:
                    epochs_override = int(token.split("=", 1)[1])
                except ValueError:
                    epochs_override = None
                break

            if token == "--ab-mode":
                ab_mode_enabled = True
                continue
            if token == "--no-ab-mode":
                ab_mode_enabled = False
                continue

            if token == "--ab-require-baseline":
                ab_require_baseline = True
                continue
            if token == "--no-ab-require-baseline":
                ab_require_baseline = False

        bootstrap_ab_mode = bool(ab_mode_enabled and (not ab_require_baseline))
        short_smoke_run = bool(epochs_override is not None and epochs_override <= 10)
        if bootstrap_ab_mode or short_smoke_run:
            os.environ["HELIX_GOV_POLICY_PROFILE"] = "smoke"
    main()
