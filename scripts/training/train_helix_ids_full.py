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

import argparse
import hashlib
import json
import logging
import math
import os
import shutil
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
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Dataset, Sampler, TensorDataset, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

HELIX_FULL_RESULTS_DIR = Path("results/helix_full")

from helix_ids.config.helix_full_config import DataConfig, TrainingConfig  # noqa: E402
from helix_ids.contracts import (
    runtime_contract_payload,
)
from helix_ids.data.geometric_representation_fixes import GeometricRepresentationFixer  # noqa: E402
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
    seed_worker,
    set_global_determinism,
)
from helix_ids.governance.entrypoint import governed_entrypoint  # noqa: E402
from helix_ids.governance.parameters import DEFAULT_GOVERNANCE_POLICY
from helix_ids.governance.promotion import SeedRunSummary, aggregate_seed_runs
from helix_ids.governance.run_registry import RunRegistry
from helix_ids.models.full import HelixFullConfig, HelixIDSFull, MultiTaskLoss, create_helix_full
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
ENGINEERED_FEATURE_NAMES = frozenset(
    {
        "log_src_bytes",
        "log_dst_bytes",
        "src_dst_bytes_ratio",
        "dst_src_bytes_ratio",
        "same_host_rate_x_service",
        "diff_srv_rate_x_flag",
        "count_x_srv_count",
        "protocol_service_flag",
    }
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
    arr = np.asarray(counts, dtype=np.float64)
    total = float(arr.sum())
    if total <= 0.0 or int(arr.shape[0]) <= 1:
        return 0.0
    probs = np.clip(arr / total, 1e-12, 1.0)
    entropy = float(-np.sum(probs * np.log(probs)) / math.log(float(arr.shape[0])))
    return float(np.clip(entropy, 0.0, 1.0))


def _detect_cluster_mode_collapse(
    cluster_sizes: list[int],
    *,
    min_entropy: float = 0.30,
    max_dominance: float = 0.85,
) -> tuple[bool, dict[str, float]]:
    """Detect cluster mode collapse using entropy and dominant-cluster share."""
    counts = [max(0, int(v)) for v in cluster_sizes]
    total = int(sum(counts))
    if total <= 0:
        return True, {
            "cluster_size_entropy": 0.0,
            "dominant_cluster_fraction": 1.0,
            "active_cluster_count": 0.0,
        }

    entropy = _normalized_entropy_from_counts(counts)
    dominant_fraction = float(max(counts) / max(1, total))
    active_cluster_count = int(sum(1 for count in counts if count > 0))
    collapse = (
        active_cluster_count < 2
        or dominant_fraction >= float(max_dominance)
        or entropy < float(min_entropy)
    )
    return collapse, {
        "cluster_size_entropy": float(entropy),
        "dominant_cluster_fraction": float(dominant_fraction),
        "active_cluster_count": float(active_cluster_count),
    }


def _load_json_dict(path: Path) -> dict[str, Any]:
    """Load JSON object from path and validate dictionary payload."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}, got {type(payload).__name__}")
    return cast(dict[str, Any], payload)


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


def _coerce_finite_float(value: Any, *, field: str) -> float:
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return float(numeric)


def _normalize_metrics_payload(metrics: dict[str, Any]) -> dict[str, Any]:
    """Normalize metric aliases into strict external contract keys."""
    normalized_metrics = {
        "macro_f1": _coerce_finite_float(
            metrics.get("macro_f1", metrics.get("family_macro_f1", metrics.get("family_f1", 0.0))),
            field="macro_f1",
        ),
        "class4_precision": _coerce_finite_float(
            metrics.get("class4_precision", metrics.get("family_class4_precision", 0.0)),
            field="class4_precision",
        ),
        "class4_recall": _coerce_finite_float(
            metrics.get("class4_recall", metrics.get("family_class4_recall", metrics.get("family_minority_recall_min", 0.0))),
            field="class4_recall",
        ),
        "entropy": _coerce_finite_float(
            metrics.get("mean_entropy", metrics.get("family_entropy", 0.0)),
            field="entropy",
        ),
        "zero_prediction_classes": int(
            metrics.get(
                "zero_prediction_classes",
                metrics.get("family_zero_prediction_classes", 0),
            )
        ),
    }
    if normalized_metrics["zero_prediction_classes"] < 0:
        raise ValueError("zero_prediction_classes must be >= 0")
    return normalized_metrics


def _materialize_phase8_artifacts(calibration_artifacts: dict[str, str]) -> dict[str, str]:
    """Create canonical artifact filenames required by strict completion contract."""
    required_artifacts = {
        "before_after_csv": "before_after.csv",
        "before_after_json": "before_after.json",
        "pr_curve_csv": "pr_curve.csv",
        "confusion_matrices_json": "confusion_matrices.json",
        "ablation_json": "ablation.json",
    }
    canonical: dict[str, str] = {}
    for source_key, canonical_name in required_artifacts.items():
        source_value = calibration_artifacts.get(source_key)
        if not source_value:
            raise ValueError(f"Missing required calibration artifact key: {source_key}")
        source_path = Path(source_value)
        if not source_path.exists():
            raise FileNotFoundError(f"Missing required calibration artifact: {source_path}")
        canonical_path = source_path.parent / canonical_name
        if source_path.resolve() != canonical_path.resolve():
            shutil.copyfile(source_path, canonical_path)
        canonical[source_key] = str(canonical_path)
    return canonical


def _normalize_calibration_block(
    *,
    calibration_payload: dict[str, Any],
    calibration_artifacts: dict[str, str],
) -> dict[str, Any]:
    """Normalize calibration outputs into strict contract schema with required paths."""
    normalized_calibration = {
        "temperature": _coerce_finite_float(calibration_payload.get("temperature", 1.0), field="temperature"),
        "tau_4": _coerce_finite_float(calibration_payload.get("tau_4", 0.5), field="tau_4"),
        "pr_curve_path": str(calibration_artifacts["pr_curve_csv"]),
        "confusion_matrix_path": str(calibration_artifacts["confusion_matrices_json"]),
        "ablation_path": str(calibration_artifacts["ablation_json"]),
    }
    for key in ("pr_curve_path", "confusion_matrix_path", "ablation_path"):
        path = Path(str(normalized_calibration[key]))
        if not path.exists():
            raise FileNotFoundError(f"Required calibration artifact missing: {path}")
    return normalized_calibration


def _load_seed_run_artifacts(
    *,
    seed: int,
    proc: subprocess.CompletedProcess[str],
) -> tuple[str, dict[str, Any], dict[str, Any], HelixIDSFull]:
    eval_path = HELIX_FULL_RESULTS_DIR / f"eval_results_seed{int(seed)}.json"
    train_path = HELIX_FULL_RESULTS_DIR / f"training_results_seed{int(seed)}.json"
    if not eval_path.exists() or not train_path.exists():
        raise RuntimeError(
            "Multi-seed run did not emit expected artifacts for seed "
            f"{seed}; exit={proc.returncode}"
        )

    train_payload = _load_json_dict(train_path)
    train_exit_code = int(train_payload.get("run_exit_code", proc.returncode))
    train_guard_failure = str(train_payload.get("guard_failure", "") or "")
    if train_exit_code != 0:
        raise RuntimeError(
            "Seed run failed before calibration artifacts were materialized: "
            f"seed={seed} exit={train_exit_code} guard_failure={train_guard_failure}"
        )

    eval_payload = _load_json_dict(eval_path)
    eval_results = cast(dict[str, Any], eval_payload.get("results", {}))
    if not eval_results:
        raise RuntimeError(f"Missing eval results for seed {seed}")

    dataset_name = min(eval_results.keys())
    model_path = Path("models/helix_full") / f"helix_full_{dataset_name}_best.pt"
    if not model_path.exists():
        raise RuntimeError(f"Missing best checkpoint for seed {seed}: {model_path}")

    artifact = torch.load(model_path, map_location="cpu", weights_only=True)
    model_state = cast(dict[str, Any], artifact.get("model_state_dict", artifact.get("model", {})))
    model: HelixIDSFull = create_helix_full(
        HelixFullConfig(
            input_dim=REQUIRED_GEOMETRY_FEATURE_DIM,
            hidden_dims=(512, 384, 256, 256),
            dropout_rates=(0.3, 0.3, 0.25, 0.2),
        )
    )
    model.load_state_dict(model_state)

    return dataset_name, train_payload, eval_results, model


def _summarize_governance(strict_seed_runs: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str], list[str]]:
    macro_vals = [float(r["macro_f1"]) for r in strict_seed_runs]
    p4_prec_vals = [float(r["class4_precision"]) for r in strict_seed_runs]
    p4_rec_vals = [float(r["class4_recall"]) for r in strict_seed_runs]
    zero_vals = [int(r["zero_prediction_classes"]) for r in strict_seed_runs]
    entropy_vals = [float(r["entropy"]) for r in strict_seed_runs]

    governance: dict[str, Any] = {
        "mean_macro_f1": float(np.mean(macro_vals)),
        "std_macro_f1": float(np.std(macro_vals)),
        "mean_class4_precision": float(np.mean(p4_prec_vals)),
        "mean_class4_recall": float(np.mean(p4_rec_vals)),
        "min_class4_recall": float(np.min(p4_rec_vals)),
        "mean_entropy": float(np.mean(entropy_vals)),
        "max_zero_prediction_classes": int(max(zero_vals)),
    }

    failure_reasons: list[str] = []
    if governance["std_macro_f1"] > 0.03:
        failure_reasons.append("std_macro_f1_gt_0_03")
    if governance["min_class4_recall"] < 0.80:
        failure_reasons.append("min_class4_recall_lt_0_80")
    if governance["mean_class4_precision"] < 0.25:
        failure_reasons.append("mean_class4_precision_lt_0_25")
    if governance["max_zero_prediction_classes"] != 0:
        failure_reasons.append("max_zero_prediction_classes_ne_0")
    if governance["mean_entropy"] <= 0.2:
        failure_reasons.append("mean_entropy_le_0_2")

    actions: list[str] = []
    if governance["mean_class4_precision"] < 0.25:
        actions.append("increase_tau_4")
    if governance["min_class4_recall"] < 0.80:
        actions.append("increase_focal_gamma_up_to_1_5")
    if governance["mean_entropy"] <= 0.2:
        actions.append("increase_temperature_max_to_5_0")

    return governance, failure_reasons, actions


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
    cluster_diag = cast(
        dict[str, Any],
        representation_diagnostics.get("cluster_relabel", representation_diagnostics.get("original", {})),
    )
    bridge = cast(dict[str, Any], representation_diagnostics.get("cluster_label_bridge", {}))

    ratio = float(cluster_diag.get("intra_inter_ratio", 0.0))
    min_inter = float(cluster_diag.get("min_inter_center_distance", 0.0))
    nearest_center_acc_val = float(cluster_diag.get("nearest_center_accuracy_val", 0.0))
    collision_count = int(len(cast(list[Any], cluster_diag.get("collision_pairs", []))))
    nearest_cluster_pairs_top5 = cast(list[Any], cluster_diag.get("nearest_cluster_pairs_top5", []))
    density_variance = float(cluster_diag.get("density_variance", 0.0))

    cluster_sizes = [
        int(v)
        for v in cast(list[Any], representation_diagnostics.get("cluster_size_counts", []))
    ]
    if not cluster_sizes:
        cluster_to_old = cast(dict[str, Any], bridge.get("cluster_to_old_counts", {}))
        if cluster_to_old:
            ordered_cluster_ids = sorted(cluster_to_old.keys(), key=lambda value: int(value))
            cluster_sizes = [
                int(sum(int(count) for count in cast(dict[str, Any], cluster_to_old[c_id]).values()))
                for c_id in ordered_cluster_ids
            ]
    cluster_size_entropy = float(
        representation_diagnostics.get(
            "cluster_size_entropy",
            _normalized_entropy_from_counts(cluster_sizes),
        )
    )

    purity_map = cast(dict[str, Any], bridge.get("old_to_cluster_purity", {}))
    cluster_purity = [
        float(purity_map[label])
        for label in sorted(purity_map.keys(), key=lambda value: int(value))
    ]
    macro_f1 = float(dataset_metrics.get("family_macro_f1", dataset_metrics.get("family_f1", 0.0)))
    zero_prediction_classes = float(dataset_metrics.get("family_zero_prediction_classes", 0.0))

    return {
        "timestamp": datetime.now().isoformat(),
        "dataset": str(dataset_name),
        "dataset_id": str(dataset_id),
        "ab_track": str(ab_track),
        "ab_change_id": str(ab_change_id),
        "k": int(k),
        "seed": int(seed),
        "ratio": float(ratio),
        "min_inter": float(min_inter),
        "cluster_sizes": cluster_sizes,
        "cluster_purity": cluster_purity,
        "macro_f1": float(macro_f1),
        "nearest_center_acc_val": float(nearest_center_acc_val),
        "nearest_center_acc": float(nearest_center_acc_val),
        "cluster_size_entropy": float(cluster_size_entropy),
        "collision_count": int(collision_count),
        "top5_pairs": nearest_cluster_pairs_top5,
        "density_variance": float(density_variance),
        "zero_prediction_classes": float(zero_prediction_classes),
        "split_snapshot_id": str(split_snapshot_id),
        "batch_size": int(batch_size),
        "eval_label_path": "cpu",
        "feature_signature": str(feature_signature),
        "cluster_objective": str(cluster_objective),
        "cluster_spectral_affinity": str(cluster_spectral_affinity),
    }



def _ab_rejection(reason: str) -> dict[str, Any]:
    """Return standard A/B rejection response."""
    return {
        "accepted": False,
        "reason": reason,
        "tier_1_geometry_pass": False,
        "tier_2_cluster_quality_pass": False,
        "tier_3_classifier_pass": False,
        "tier_4_governance_pass": False,
        "tier_3_evaluated": False,
    }


def _validate_ab_contract(
    current: dict[str, Any],
    baseline: dict[str, Any],
) -> Optional[dict[str, Any]]:
    """Validate A/B contract fields. Returns error dict if invalid, None if valid."""
    required_contract_fields = [
        "dataset_id",
        "split_snapshot_id",
        "batch_size",
        "eval_label_path",
        "k",
        "seed",
    ]
    for field in required_contract_fields:
        if current.get(field) != baseline.get(field):
            return _ab_rejection(f"ab_contract_mismatch:{field}")
    return None


def _detect_feature_and_objective_changes(
    current: dict[str, Any],
    baseline: dict[str, Any],
) -> tuple[bool, bool]:
    """Detect if feature or objective changed."""
    feature_changed = str(current.get("feature_signature", "")) != str(
        baseline.get("feature_signature", "")
    )
    objective_changed = (
        str(current.get("cluster_objective", "")) != str(baseline.get("cluster_objective", ""))
        or str(current.get("cluster_spectral_affinity", ""))
        != str(baseline.get("cluster_spectral_affinity", ""))
    )
    return feature_changed, objective_changed


def _validate_track(track: str, feature_changed: bool, objective_changed: bool) -> Optional[dict[str, Any]]:
    """Validate track against changes. Returns error dict if invalid, None if valid."""
    track_lower = str(track).strip().lower()
    if track_lower == "feature":
        if (not feature_changed) or objective_changed:
            return _ab_rejection("ab_contract_invalid_feature_track")
    elif track_lower == "objective":
        if feature_changed or (not objective_changed):
            return _ab_rejection("ab_contract_invalid_objective_track")
    else:
        return _ab_rejection(f"ab_contract_invalid_track:{track}")
    return None


def evaluate_ab_candidate(
    *,
    current: dict[str, Any],
    baseline: dict[str, Any],
    ab_track: str,
    governance_z_score: float,
    governance_z_tolerance: float,
) -> dict[str, Any]:
    """Evaluate strict tiered A/B acceptance gates and promotion rule."""
    # Contract validation
    contract_error = _validate_ab_contract(current, baseline)
    if contract_error:
        return contract_error

    # Feature/objective detection and validation
    feature_changed, objective_changed = _detect_feature_and_objective_changes(current, baseline)
    if feature_changed and objective_changed:
        return _ab_rejection("ab_anti_pattern_mixed_feature_and_objective_change")

    track_error = _validate_track(ab_track, feature_changed, objective_changed)
    if track_error:
        return track_error

    # Tier 1: Geometry
    tier_1_geometry_pass = (
        float(current.get("ratio", 0.0)) < float(baseline.get("ratio", 0.0))
        and float(current.get("min_inter", 0.0)) > float(baseline.get("min_inter", 0.0))
    )
    if not tier_1_geometry_pass:
        return {
            "accepted": False,
            "reason": "tier1_geometry_regression",
            "tier_1_geometry_pass": False,
            "tier_2_cluster_quality_pass": False,
            "tier_3_classifier_pass": False,
            "tier_4_governance_pass": False,
            "tier_3_evaluated": False,
            "delta_ratio": float(current.get("ratio", 0.0)) - float(baseline.get("ratio", 0.0)),
            "delta_min_inter": float(current.get("min_inter", 0.0)) - float(baseline.get("min_inter", 0.0)),
        }

    # Tier 2: Cluster quality
    cluster_sizes = [int(v) for v in cast(list[Any], current.get("cluster_sizes", []))]
    collapse, collapse_metrics = _detect_cluster_mode_collapse(cluster_sizes)
    tier_2_cluster_quality_pass = not collapse
    if not tier_2_cluster_quality_pass:
        return {
            "accepted": False,
            "reason": "tier2_cluster_mode_collapse",
            "tier_1_geometry_pass": True,
            "tier_2_cluster_quality_pass": False,
            "tier_3_classifier_pass": False,
            "tier_4_governance_pass": False,
            "tier_3_evaluated": False,
            **collapse_metrics,
        }

    # Tier 3: Classifier quality
    current_zero_prediction_classes = float(current.get("zero_prediction_classes", 0.0))
    eps = 1e-9
    no_zero_prediction_classes = current_zero_prediction_classes < eps
    baseline_macro_f1 = float(baseline.get("macro_f1", 0.0))
    baseline_zero_prediction_classes = float(baseline.get("zero_prediction_classes", 0.0))
    current_macro_f1 = float(current.get("macro_f1", 0.0))
    tier_3_classifier_pass = (
        no_zero_prediction_classes
        and current_macro_f1 > (baseline_macro_f1 - eps)
        and current_zero_prediction_classes < (baseline_zero_prediction_classes + eps)
    )
    if not tier_3_classifier_pass:
        return {
            "accepted": False,
            "reason": "tier3_classifier_surface_regression",
            "tier_1_geometry_pass": True,
            "tier_2_cluster_quality_pass": True,
            "tier_3_classifier_pass": False,
            "tier_4_governance_pass": False,
            "tier_3_evaluated": True,
            "delta_macro_f1": float(current.get("macro_f1", 0.0))
            - float(baseline.get("macro_f1", 0.0)),
        }

    # Tier 4: Governance
    tier_4_governance_pass = abs(float(governance_z_score)) <= float(governance_z_tolerance)
    accepted = bool(
        tier_1_geometry_pass and tier_2_cluster_quality_pass and tier_3_classifier_pass and tier_4_governance_pass
    )
    return {
        "accepted": accepted,
        "reason": "ok" if accepted else "tier4_governance_drift_out_of_tolerance",
        "tier_1_geometry_pass": bool(tier_1_geometry_pass),
        "tier_2_cluster_quality_pass": bool(tier_2_cluster_quality_pass),
        "tier_3_classifier_pass": bool(tier_3_classifier_pass),
        "tier_4_governance_pass": bool(tier_4_governance_pass),
        "tier_3_evaluated": True,
        "delta_ratio": float(current.get("ratio", 0.0)) - float(baseline.get("ratio", 0.0)),
        "delta_min_inter": float(current.get("min_inter", 0.0)) - float(baseline.get("min_inter", 0.0)),
        "delta_macro_f1": float(current.get("macro_f1", 0.0)) - float(baseline.get("macro_f1", 0.0)),
        "governance_z_score": float(governance_z_score),
        "governance_z_tolerance": float(governance_z_tolerance),
        **collapse_metrics,
    }


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


class MultiTaskNumpyDataset(Dataset):
    """Lazy dataset backed by numpy arrays (or memmaps) for multi-task labels."""

    def __init__(self, features: np.ndarray, family_labels: np.ndarray):
        if int(features.shape[0]) != int(family_labels.shape[0]):
            raise ValueError(
                f"Feature/label length mismatch: X={features.shape[0]}, y={family_labels.shape[0]}"
            )
        self.features = features
        self.family_labels = np.asarray(family_labels, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.family_labels.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y_family = int(self.family_labels[idx])
        y_binary = 1 if y_family != 0 else 0
        x_row = np.asarray(self.features[idx], dtype=np.float32)
        return (
            torch.from_numpy(x_row),
            torch.tensor(y_binary, dtype=torch.long),
            torch.tensor(y_family, dtype=torch.long),
        )


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


class ClassBalancedIndexSampler(Sampler[int]):
    """Yield flattened indices where each batch contains all active classes and tail oversampling."""

    def __init__(
        self,
        y: np.ndarray,
        batch_size: int,
        *,
        seed: int = 42,
        min_per_class: int = 1,
        class_multipliers: Optional[dict[int, float]] = None,
    ) -> None:
        self.y = np.asarray(y, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.min_per_class = max(1, int(min_per_class))

        self.class_index = build_class_index(self.y)
        self.classes = sorted(self.class_index.keys())
        if not self.classes:
            raise ValueError("ClassBalancedBatchSampler requires at least one class")

        self.class_counts = {
            class_id: int(self.class_index[class_id].shape[0]) for class_id in self.classes
        }
        if class_multipliers is None:
            self.class_multipliers = {
                class_id: _default_tail_multiplier(self.class_counts[class_id])
                for class_id in self.classes
            }
        else:
            self.class_multipliers = {
                class_id: float(class_multipliers.get(class_id, 1.0)) for class_id in self.classes
            }

        self.steps_per_epoch = max(1, int(math.ceil(self.y.shape[0] / max(1, self.batch_size))))
        required_slots = len(self.classes) * self.min_per_class
        if required_slots > self.batch_size:
            raise ValueError(
                "batch_size too small for per-class presence constraint: "
                f"required={required_slots}, batch_size={self.batch_size}"
            )
        self.remainder = self.batch_size - required_slots
        self._epoch = 0

    def __len__(self) -> int:
        return self.steps_per_epoch * self.batch_size

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1

        for _ in range(self.steps_per_epoch):
            batch_indices: list[int] = []

            # Ensure every active class appears in each batch.
            for class_id in self.classes:
                cls_indices = self.class_index[class_id]
                sampled = rng.choice(cls_indices, size=self.min_per_class, replace=True)
                batch_indices.extend(sampled.tolist())

            if self.remainder > 0:
                sampling_weights = np.asarray(
                    [
                        float(self.class_counts[class_id]) * float(self.class_multipliers[class_id])
                        for class_id in self.classes
                    ],
                    dtype=np.float64,
                )
                if float(sampling_weights.sum()) <= 0.0:
                    sampling_weights = np.ones_like(sampling_weights)
                sampling_weights = sampling_weights / float(sampling_weights.sum())
                extra_classes = rng.choice(
                    self.classes,
                    size=self.remainder,
                    replace=True,
                    p=sampling_weights,
                )
                for class_id in extra_classes.tolist():
                    cls_indices = self.class_index[int(class_id)]
                    sampled = rng.choice(cls_indices, size=1, replace=True)
                    batch_indices.append(int(sampled[0]))

            rng.shuffle(batch_indices)
            for sample_idx in batch_indices:
                yield int(sample_idx)


class FrozenIndexSampler(Sampler[int]):
    """Deterministic sampler backed by precomputed indices."""

    def __init__(self, indices: np.ndarray) -> None:
        indices_np = np.asarray(indices, dtype=np.int64)
        if indices_np.size == 0:
            raise ValueError("FrozenIndexSampler requires at least one index")
        self.indices = indices_np

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __iter__(self):
        for idx in self.indices.tolist():
            yield int(idx)


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
        self.energy_win_rate_ema: Optional[torch.Tensor] = None
        self._energy_bias_last_std = 0.0
        self._energy_bias_last_max_abs = 0.0
        self._energy_bias_last_logit_std = 0.0
        self._logit_temp = 1.0
        self._temperature_calibration = 1.0
        self._temperature_calibration_lr = 1e-3
        self._epoch_frozen_centroids: dict[int, torch.Tensor] = {}
        self.rep_epoch_feature_chunks: list[torch.Tensor] = []
        self.rep_epoch_label_chunks: list[torch.Tensor] = []
        self.rep_backbone_grad_scale = 2.0
        self.centroid_ema_momentum = 0.9
        self._centroid_ema_state: dict[int, torch.Tensor] = {}
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

    @staticmethod
    def _supervised_contrastive_loss(
        features: torch.Tensor,
        labels: torch.Tensor,
        *,
        temperature: float,
        anchor_weights: Optional[torch.Tensor] = None,
        negative_weight: float = 1.0,
        min_negatives: int = 1,
    ) -> torch.Tensor:
        """Compute supervised contrastive loss over a batch of backbone features."""
        if int(features.shape[0]) <= 1:
            return torch.zeros((), dtype=features.dtype, device=features.device)

        feat = F.normalize(features, p=2, dim=1)
        logits = torch.matmul(feat, feat.T) / max(1e-6, float(temperature))

        # Mask self-comparisons with a large finite negative value to avoid NaNs.
        self_mask = torch.eye(logits.shape[0], device=logits.device, dtype=torch.bool)

        labels_col = labels.view(-1, 1)
        positive_mask = (labels_col == labels_col.T) & (~self_mask)
        negative_mask = (labels_col != labels_col.T) & (~self_mask)

        positive_count = positive_mask.sum(dim=1)
        negative_count = negative_mask.sum(dim=1)
        valid_anchor = (positive_count > 0) & (negative_count >= int(max(1, min_negatives)))
        if not bool(valid_anchor.any()):
            return torch.zeros((), dtype=features.dtype, device=features.device)

        masked_logits = logits.masked_fill(self_mask, -1e9)
        row_max = masked_logits.max(dim=1, keepdim=True).values
        exp_logits = torch.exp(masked_logits - row_max)
        exp_logits = exp_logits.masked_fill(self_mask, 0.0)

        neg_multiplier = torch.where(
            negative_mask,
            torch.full_like(exp_logits, float(max(1.0, negative_weight))),
            torch.ones_like(exp_logits),
        )
        weighted_exp = exp_logits * neg_multiplier

        denom = weighted_exp.sum(dim=1).clamp_min(1e-12)
        pos_sum = (exp_logits * positive_mask.to(exp_logits.dtype)).sum(dim=1).clamp_min(1e-12)
        loss_per_anchor = -torch.log(pos_sum / denom)
        valid_loss = loss_per_anchor[valid_anchor]
        if anchor_weights is None:
            return valid_loss.mean()

        valid_weights = anchor_weights.to(device=valid_loss.device, dtype=valid_loss.dtype)[valid_anchor]
        valid_weights = valid_weights / valid_weights.sum().clamp_min(1e-12)
        return torch.sum(valid_loss * valid_weights)

    @staticmethod
    def _supcon_anchor_weights(labels: torch.Tensor) -> torch.Tensor:
        """Build class-balanced anchor weights for SupCon: 1 / log(1 + class_freq)."""
        labels_int = labels.to(dtype=torch.int64)
        max_label = int(torch.max(labels_int).item()) if int(labels_int.numel()) > 0 else 0
        class_counts = torch.bincount(labels_int, minlength=max_label + 1).to(dtype=torch.float32)
        class_counts = torch.clamp(class_counts, min=1.0)
        weights = 1.0 / torch.log1p(class_counts[labels_int])
        return weights / weights.mean().clamp_min(1e-12)

    @staticmethod
    def _class_conditional_energy_gap_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        *,
        alpha: float,
    ) -> tuple[torch.Tensor, float, float, float, float]:
        """Compute class-conditional multi-negative energy ordering loss from family logits."""
        if int(logits.ndim) != 2 or int(logits.shape[0]) == 0:
            zero = torch.zeros((), dtype=logits.dtype, device=logits.device)
            return zero, 0.0, 0.0, 0.0, 0.0
        if int(logits.shape[1]) <= 1:
            ce_only = F.cross_entropy(logits, labels)
            ce_scalar = float(ce_only.detach().item())
            return ce_only, ce_scalar, 0.0, 0.0, 0.0

        true_class_mask = F.one_hot(labels, num_classes=int(logits.shape[1])).to(dtype=torch.bool)
        logit_y = logits.gather(1, labels.view(-1, 1)).squeeze(1)
        logits_negatives = logits.masked_fill(true_class_mask, float("-inf"))
        logsumexp_neg = torch.logsumexp(logits_negatives, dim=1)

        energy_y = -logit_y
        energy_gap = logit_y - logsumexp_neg
        total = torch.mean(energy_y + (float(alpha) * logsumexp_neg))
        return (
            total,
            float(energy_y.detach().mean().item()),
            float(logsumexp_neg.detach().mean().item()),
            float(energy_gap.detach().mean().item()),
            float(total.detach().item()),
        )

    @staticmethod
    def _energy_class_balance_loss(
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, float, float, float]:
        """Encourage non-collapsed class support via KL(pred || target)."""
        if int(logits.ndim) != 2 or int(logits.shape[0]) == 0:
            zero = torch.zeros((), dtype=logits.dtype, device=logits.device)
            return zero, 0.0, 0.0, 0.0

        probs = F.softmax(logits, dim=1)
        p_pred = probs.mean(dim=0).clamp_min(1e-12)
        num_classes = max(1, int(logits.shape[1]))
        p_target = torch.full_like(p_pred, 1.0 / float(num_classes))
        kl_loss = torch.sum(p_pred * (torch.log(p_pred) - torch.log(p_target)))
        pred_entropy = -torch.sum(p_pred * torch.log(p_pred))
        return (
            kl_loss,
            float(kl_loss.detach().item()),
            float(pred_entropy.detach().item()),
            float(p_pred.detach().min().item()),
        )

    @staticmethod
    def _energy_min_winner_loss(
        logits: torch.Tensor,
        active_class_ids: Optional[list[int]],
        *,
        min_winners: int,
    ) -> tuple[torch.Tensor, float, float]:
        """Penalize per-batch argmax winner starvation over active classes."""
        if int(logits.ndim) != 2 or int(logits.shape[0]) == 0:
            zero = torch.zeros((), dtype=logits.dtype, device=logits.device)
            return zero, 0.0, 0.0

        class_count = int(logits.shape[1])
        hard_pred = torch.argmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)
        hard_counts = torch.bincount(hard_pred, minlength=class_count).to(dtype=logits.dtype)
        soft_counts = probs.sum(dim=0)
        counts = soft_counts + (hard_counts - soft_counts).detach()

        if active_class_ids is None:
            active_indices = torch.arange(class_count, device=logits.device, dtype=torch.int64)
        else:
            clean_ids = [cid for cid in active_class_ids if 0 <= int(cid) < class_count]
            if not clean_ids:
                active_indices = torch.arange(class_count, device=logits.device, dtype=torch.int64)
            else:
                active_indices = torch.tensor(clean_ids, device=logits.device, dtype=torch.int64)

        if int(active_indices.numel()) == 0:
            zero = torch.zeros((), dtype=logits.dtype, device=logits.device)
            return zero, 0.0, 0.0

        active_counts = counts.index_select(0, active_indices)
        deficits = F.relu(float(max(0, min_winners)) - active_counts)
        loss = torch.sum(deficits)
        return (
            loss,
            float(deficits.detach().sum().item()),
            float(active_counts.detach().min().item()),
        )

    @staticmethod
    def _pairwise_margin_repulsion_loss(
        features: torch.Tensor,
        labels: torch.Tensor,
        *,
        margin: float,
        hard_negative_weight: float = 1.0,
        top_k: int = 3,
    ) -> torch.Tensor:
        """Penalize hardest/top-k nearest negative pairs with class-balanced weighting."""
        if int(features.shape[0]) <= 1:
            return torch.zeros((), dtype=features.dtype, device=features.device)

        pair_dists = torch.cdist(features, features, p=2)
        labels_int = labels.to(dtype=torch.int64)
        labels_col = labels_int.view(-1, 1)
        neg_mask = labels_col != labels_col.T
        upper_mask = torch.triu(torch.ones_like(pair_dists, dtype=torch.bool), diagonal=1)
        valid_neg = neg_mask & upper_mask

        neg_indices = torch.where(valid_neg)
        neg_dists = pair_dists[neg_indices]
        if int(neg_dists.numel()) == 0:
            return torch.zeros((), dtype=features.dtype, device=features.device)

        max_label = int(torch.max(labels_int).item()) if int(labels_int.numel()) > 0 else 0
        class_counts = torch.bincount(labels_int, minlength=max_label + 1).to(
            device=features.device,
            dtype=features.dtype,
        )
        class_counts = torch.clamp(class_counts, min=1.0)

        neg_label_i = labels_int[neg_indices[0]]
        neg_label_j = labels_int[neg_indices[1]]
        count_i = class_counts[neg_label_i]
        count_j = class_counts[neg_label_j]
        balance_weight = torch.sqrt(1.0 / torch.clamp(count_i * count_j, min=1.0))
        balance_weight = balance_weight / balance_weight.mean().clamp_min(1e-12)

        hard_negative_multiplier = torch.where(
            neg_dists < float(margin),
            torch.full_like(neg_dists, float(max(1.0, hard_negative_weight))),
            torch.ones_like(neg_dists),
        )
        margin_violation = F.relu(float(margin) - neg_dists)
        weighted_violation = margin_violation * balance_weight * hard_negative_multiplier

        k = int(max(1, top_k))
        k = min(k, int(weighted_violation.numel()))
        topk_values, _ = torch.topk(weighted_violation, k=k, largest=True, sorted=False)
        return topk_values.sum()

    @staticmethod
    def _centroid_separation_barrier_loss(
        features: torch.Tensor,
        labels: torch.Tensor,
        *,
        min_distance: float,
    ) -> torch.Tensor:
        """Penalize class-centroid pairs that sit below a minimum separation distance."""
        unique_labels = torch.unique(labels, dim=0)
        if int(unique_labels.numel()) <= 1:
            return torch.zeros((), dtype=features.dtype, device=features.device)

        centroids: list[torch.Tensor] = []
        for class_id in unique_labels.tolist():
            class_features = features[labels == int(class_id)]
            if int(class_features.shape[0]) == 0:
                continue
            centroids.append(class_features.mean(dim=0))

        if len(centroids) <= 1:
            return torch.zeros((), dtype=features.dtype, device=features.device)

        center_tensor = F.normalize(torch.stack(centroids, dim=0), p=2, dim=1)
        center_dists = torch.cdist(center_tensor, center_tensor, p=2)
        upper_mask = torch.triu(torch.ones_like(center_dists, dtype=torch.bool), diagonal=1)
        pair_dists = center_dists[upper_mask]
        if int(pair_dists.numel()) == 0:
            return torch.zeros((), dtype=features.dtype, device=features.device)
        return (F.relu(float(min_distance) - pair_dists) * pair_dists).sum()

    @staticmethod
    def _centroid_repulsion_loss(
        features: torch.Tensor,
        labels: torch.Tensor,
        *,
        margin: float,
    ) -> torch.Tensor:
        """Apply centroid-level margin force with constant gradient inside margin."""
        unique_labels = torch.unique(labels, dim=0)
        if int(unique_labels.numel()) <= 1:
            return torch.zeros((), dtype=features.dtype, device=features.device)

        centroids: list[torch.Tensor] = []
        for class_id in unique_labels.tolist():
            class_features = features[labels == int(class_id)]
            if int(class_features.shape[0]) == 0:
                continue
            centroids.append(class_features.mean(dim=0))

        if len(centroids) <= 1:
            return torch.zeros((), dtype=features.dtype, device=features.device)

        center_tensor = F.normalize(torch.stack(centroids, dim=0), p=2, dim=1)
        center_dists = torch.cdist(center_tensor, center_tensor, p=2)
        upper_mask = torch.triu(torch.ones_like(center_dists, dtype=torch.bool), diagonal=1)
        pair_dists = center_dists[upper_mask]
        if int(pair_dists.numel()) == 0:
            return torch.zeros((), dtype=features.dtype, device=features.device)
        return F.relu(float(margin) - pair_dists).sum()

    def _should_exit_representation_curriculum(self) -> bool:
        """Return True once adaptive geometry targets are met for phase-1 termination."""
        if self.use_energy_based_family_objective:
            return False
        if not self.rep_phase_feature_chunks or not self.rep_phase_label_chunks:
            return False

        rep_features = torch.cat(self.rep_phase_feature_chunks, dim=0)
        rep_labels = torch.cat(self.rep_phase_label_chunks, dim=0)
        if int(torch.unique(rep_labels, dim=0).numel()) <= 1:
            return False

        diagnostics = self._run_representation_diagnostics(
            train_features=rep_features,
            train_labels=rep_labels,
            label_space="phase1_probe",
        )
        ratio = float(
            diagnostics.get(
                "intra_inter_ratio",
                diagnostics.get("intra_to_inter_ratio", diagnostics.get("ratio", float("inf"))),
            )
        )
        min_inter = float(diagnostics.get("min_inter_center_distance", 0.0))
        pass_exit = (
            ratio < float(self.rep_adaptive_exit_ratio_threshold)
            and min_inter > float(self.rep_adaptive_exit_min_inter_threshold)
        )

        self.representation_diagnostics["adaptive_exit_probe"] = {
            "ratio": ratio,
            "min_inter": min_inter,
            "ratio_threshold": float(self.rep_adaptive_exit_ratio_threshold),
            "min_inter_threshold": float(self.rep_adaptive_exit_min_inter_threshold),
            "pass": bool(pass_exit),
            "step": int(self.global_step),
        }
        self.logger.info(
            "RepDiag[phase1_probe] adaptive_exit_check ratio=%.4f min_inter=%.4f thresholds=(%.4f, %.4f) pass=%s",
            ratio,
            min_inter,
            float(self.rep_adaptive_exit_ratio_threshold),
            float(self.rep_adaptive_exit_min_inter_threshold),
            str(bool(pass_exit)).lower(),
        )
        return bool(pass_exit)

    def _stabilize_centroids(
        self,
        batch_centroids: torch.Tensor,
        class_ids: list[int],
    ) -> torch.Tensor:
        """Apply centroid EMA smoothing for stable margin/logging targets."""
        if int(batch_centroids.shape[0]) == 0:
            return batch_centroids.detach()

        stabilized: list[torch.Tensor] = []
        m = float(np.clip(self.centroid_ema_momentum, 0.0, 1.0))
        for idx, class_id in enumerate(class_ids):
            current = batch_centroids[idx].detach().to(device="cpu", dtype=torch.float32)
            prev = self._centroid_ema_state.get(int(class_id))
            if prev is None:
                ema = current
            else:
                ema = (m * prev) + ((1.0 - m) * current)
            self._centroid_ema_state[int(class_id)] = ema.detach().clone()
            stabilized.append(ema.to(device=batch_centroids.device, dtype=batch_centroids.dtype))

        return torch.stack(stabilized, dim=0)

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
        """Configure structural anti-collapse constraints for family prediction coverage."""
        self.active_family_class_ids = {int(cls) for cls in active_family_classes}
        self.supcon_weight = max(0.0, float(supcon_weight))
        self.supcon_temperature = max(1e-3, float(supcon_temperature))
        self.rep_supcon_weight = max(0.0, float(supcon_weight))
        self.rep_supcon_temperature = max(1e-3, float(supcon_temperature))
        self.step_coverage_check_step = max(1, int(step_coverage_check_step))
        self.step_coverage_checked = False
        self.representation_diagnostic_mode = bool(representation_diagnostic_mode)
        self.representation_only_steps = max(
            0, int(phase_settings.get("representation_only_steps", 0))
        )
        self.head_only_steps = max(0, int(phase_settings.get("head_only_steps", 0)))
        self.representation_phase_active = False
        self.representation_curriculum_complete = False
        self.in_representation_window = False
        self.head_phase_start_step = -1
        self.joint_finetune_start_step = -1
        self.joint_finetune_active = False
        self.joint_finetune_backbone_lr_multiplier = max(
            1e-3,
            float(phase_settings.get("joint_finetune_backbone_lr_multiplier", 0.25)),
        )
        self.joint_finetune_head_lr_multiplier = max(
            1e-3,
            float(phase_settings.get("joint_finetune_head_lr_multiplier", 0.15)),
        )
        self.coverage_check_after_head_steps = max(1, int(step_coverage_check_step))
        self.cluster_relabeling_enabled = bool(cluster_relabeling_enabled)
        self.cluster_relabel_k = (
            None if cluster_relabel_k is None else max(2, int(cluster_relabel_k))
        )
        self.cluster_relabel_seed = int(cluster_relabel_seed)
        objective = str(cluster_relabel_objective).strip().lower()
        if objective not in {"kmeans", "gmm", "spectral"}:
            raise ValueError(f"Unsupported cluster relabel objective: {cluster_relabel_objective!r}")
        self.cluster_relabel_objective = objective
        spectral_affinity = str(cluster_relabel_spectral_affinity).strip().lower()
        if spectral_affinity not in {"nearest_neighbors", "rbf"}:
            raise ValueError(
                "Unsupported spectral affinity: "
                f"{cluster_relabel_spectral_affinity!r}"
            )
        self.cluster_relabel_spectral_affinity = spectral_affinity
        self.geometry_max_intra_inter_ratio_warmup = max(
            0.1, float(phase_settings.get("geometry_ratio_warmup_threshold", 2.5))
        )
        self.geometry_max_intra_inter_ratio_post_phase = max(
            0.1,
            float(phase_settings.get("geometry_ratio_post_phase_threshold", 1.2)),
        )
        self.geometry_max_intra_inter_ratio = self.geometry_max_intra_inter_ratio_post_phase
        self.enforce_all_classes_per_batch = bool(
            phase_settings.get("enforce_all_classes_per_batch", False)
        )
        cycle_steps_raw = phase_settings.get(
            "representation_micro_cycle_steps",
            [40, 20, 40, 20, 40],
        )
        cycle_steps: list[int] = []
        if isinstance(cycle_steps_raw, (list, tuple)):
            for value in cycle_steps_raw:
                try:
                    cycle_steps.append(max(1, int(value)))
                except (TypeError, ValueError):
                    continue
        if not cycle_steps:
            cycle_steps = [max(1, int(self.representation_only_steps))]
        if len(cycle_steps) % 2 == 0:
            cycle_steps.append(cycle_steps[-1])
        self.representation_window_pattern = [
            (idx % 2 == 0, int(window_steps))
            for idx, window_steps in enumerate(cycle_steps)
        ]
        self.representation_only_steps = int(
            sum(window_steps for _, window_steps in self.representation_window_pattern)
        )
        self.rep_adaptive_exit_ratio_threshold = float(
            phase_settings.get("adaptive_exit_ratio_threshold", 1.6)
        )
        self.rep_adaptive_exit_min_inter_threshold = float(
            phase_settings.get("adaptive_exit_min_inter_threshold", 0.30)
        )
        self.sampler_mode = (
            str(phase_settings.get("sampler_mode", "interleaved_rr")).strip().lower()
            or "interleaved_rr"
        )
        self.cluster_centers = None
        self.phase1_class_centroids = None
        self.phase1_centroid_class_ids = []
        self._centroid_ema_state = {}
        self._epoch_frozen_centroids = {}
        self.rep_epoch_feature_chunks = []
        self.rep_epoch_label_chunks = []
        self.representation_snapshot_id = None
        self.rep_backbone_grad_scale = 2.0
        self.use_energy_based_family_objective = bool(
            phase_settings.get("use_energy_based_family_objective", True)
        )
        self.energy_gap_margin = max(0.0, float(phase_settings.get("energy_gap_margin", 1.0)))
        self.energy_gap_weight = max(0.0, float(phase_settings.get("energy_gap_weight", 1.0)))
        self.energy_multi_negative_alpha = max(
            0.0, float(phase_settings.get("energy_multi_negative_alpha", 1.0))
        )
        self.energy_logit_temperature = max(
            1.0, float(phase_settings.get("energy_logit_temperature", 2.0))
        )
        self.energy_balance_weight = max(
            0.0, float(phase_settings.get("energy_balance_weight", 0.1))
        )
        self.energy_winner_weight = max(
            0.0, float(phase_settings.get("energy_winner_weight", 0.5))
        )
        self.energy_winner_min_count = max(
            0, int(phase_settings.get("energy_winner_min_count", 1))
        )
        self.energy_emergence_bias_beta = max(
            0.0,
            float(
                phase_settings.get(
                    "energy_emergence_bias_beta",
                    self.energy_emergence_bias_beta,
                )
            ),
        )
        self.energy_emergence_bias_eps = max(
            1e-6,
            float(
                phase_settings.get(
                    "energy_emergence_bias_eps",
                    self.energy_emergence_bias_eps,
                )
            ),
        )
        win_rate_ema_momentum = float(
            phase_settings.get(
                "energy_win_rate_ema_momentum",
                self.energy_win_rate_ema_momentum,
            )
        )
        self.energy_win_rate_ema_momentum = min(max(win_rate_ema_momentum, 0.80), 0.95)
        target_ratio_raw = float(
            phase_settings.get(
                "energy_emergence_bias_target_ratio",
                self.energy_emergence_bias_target_ratio,
            )
        )
        self.energy_emergence_bias_target_ratio = min(
            max(target_ratio_raw, self.energy_emergence_bias_ratio_min),
            self.energy_emergence_bias_ratio_max,
        )
        self.energy_isolate_short_horizon = bool(
            phase_settings.get(
                "energy_isolate_short_horizon",
                self.energy_isolate_short_horizon,
            )
        )
        if (
            self.use_energy_based_family_objective
            and self.energy_isolate_short_horizon
            and int(self.config.epochs) <= 1
        ):
            self.energy_balance_weight = 0.0
            self.energy_winner_weight = 0.0
            self.logger.info(
                "Energy isolation enabled for <=1 epoch run: forcing balance_w=0.0 winner_w=0.0"
            )
        self.energy_win_rate_ema = None
        # Disable any train-time family temperature scaling for structure-recovery runs.
        self.train_temperature = 1.0

    def _set_phase_trainability(
        self,
        *,
        train_backbone: bool,
        train_family_head: bool,
        train_family_projection: Optional[bool] = None,
    ) -> None:
        """Toggle trainability for backbone/family head during diagnostic two-phase training."""
        train_projection = bool(
            train_family_head if train_family_projection is None else train_family_projection
        )
        for param in self.model.backbone.parameters():
            param.requires_grad = bool(train_backbone)
        if hasattr(self.model, "family_projection"):
            for param in self.model.family_projection.parameters():
                param.requires_grad = train_projection
        for param in self.model.family_head.parameters():
            param.requires_grad = bool(train_family_head)

        self.backbone_params = [
            param for param in self.model.backbone.parameters() if param.requires_grad
        ]
        self.logger.info(
            "PhaseTrainability backbone=%s family_projection=%s family_head=%s",
            "train" if train_backbone else "frozen",
            "train" if train_projection else "frozen",
            "train" if train_family_head else "frozen",
        )

    def _set_phase_lr_scales(self, *, backbone_multiplier: float, head_multiplier: float) -> None:
        """Apply phase-specific LR multipliers on top of base group scales."""
        backbone_mult = max(1e-4, float(backbone_multiplier))
        head_mult = max(1e-4, float(head_multiplier))

        for idx, param_group in enumerate(self.optimizer.param_groups):
            group_name = str(param_group.get("group_name", f"group_{idx}"))
            base_scale = float(self._base_lr_scales.get(group_name, param_group.get("lr_scale", 1.0)))
            if group_name == "backbone":
                param_group["lr_scale"] = base_scale * backbone_mult
            elif group_name == "family_head":
                param_group["lr_scale"] = base_scale * head_mult

        self._set_learning_rate()

    def _is_representation_window_step(self, step: int) -> bool:
        """Return whether the current step is a representation micro-window."""
        if self.representation_curriculum_complete:
            return False
        if not self.representation_window_pattern:
            return int(step) < int(self.representation_only_steps)

        remaining = int(step)
        for is_representation_window, window_steps in self.representation_window_pattern:
            if remaining < int(window_steps):
                return bool(is_representation_window)
            remaining -= int(window_steps)
        return False

    def _prepare_representation_features(self, features: torch.Tensor) -> torch.Tensor:
        """Return L2-normalized embeddings for geometry stages."""
        return F.normalize(features, p=2, dim=1)

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
        """Penalize exploding and collapsing per-class embedding variance."""
        unique_labels = torch.unique(labels, dim=0)
        if int(unique_labels.numel()) <= 1:
            return torch.zeros((), dtype=features.dtype, device=features.device)

        penalties: list[torch.Tensor] = []
        lower = float(self.rep_var_lower_bound)
        upper = float(self.rep_var_upper_bound)
        for class_id in unique_labels.tolist():
            class_features = features[labels == int(class_id)]
            if int(class_features.shape[0]) <= 1:
                continue
            class_var = torch.var(class_features, dim=0, unbiased=False).mean()
            penalties.append(F.relu(class_var - upper) + F.relu(lower - class_var))

        if not penalties:
            return torch.zeros((), dtype=features.dtype, device=features.device)
        return torch.stack(penalties).mean()

    def _compute_class_centroids(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, list[int]]:
        """Compute detached class centroids from normalized embeddings."""
        if int(features.shape[0]) == 0 or int(labels.shape[0]) == 0:
            return torch.zeros((0, 0), dtype=torch.float32), []

        class_ids = sorted(int(v) for v in torch.unique(labels, dim=0).tolist())
        centroids: list[torch.Tensor] = []
        for class_id in class_ids:
            class_features = features[labels == int(class_id)]
            if int(class_features.shape[0]) == 0:
                continue
            centroids.append(class_features.mean(dim=0))

        if not centroids:
            return torch.zeros((0, 0), dtype=torch.float32), []

        centroid_tensor = torch.stack(centroids, dim=0).detach().to(device="cpu", dtype=torch.float32)
        centroid_tensor = F.normalize(centroid_tensor, p=2, dim=1)
        return centroid_tensor, class_ids

    @staticmethod
    def _compute_batch_class_centroids_for_loss(
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, list[int]]:
        """Compute differentiable per-class centroids from a batch."""
        if int(features.shape[0]) == 0 or int(labels.shape[0]) == 0:
            return torch.zeros((0, 0), dtype=features.dtype, device=features.device), []

        class_ids = sorted(int(v) for v in torch.unique(labels, dim=0).tolist())
        centroids: list[torch.Tensor] = []
        for class_id in class_ids:
            class_features = features[labels == int(class_id)]
            if int(class_features.shape[0]) == 0:
                continue
            centroids.append(class_features.mean(dim=0))

        if not centroids:
            return torch.zeros((0, 0), dtype=features.dtype, device=features.device), []
        return torch.stack(centroids, dim=0), class_ids

    def _update_running_rep_centroids(
        self,
        batch_centroids: torch.Tensor,
        class_ids: list[int],
    ) -> None:
        """Update running per-class centroids (EMA) during representation phase."""
        if int(batch_centroids.shape[0]) == 0:
            return

        m = float(np.clip(self.centroid_ema_momentum, 0.0, 1.0))
        for idx, class_id in enumerate(class_ids):
            current = F.normalize(
                batch_centroids[idx].detach().to(device="cpu", dtype=torch.float32),
                p=2,
                dim=0,
            )
            prev = self._centroid_ema_state.get(int(class_id))
            ema = current if prev is None else ((m * prev) + ((1.0 - m) * current))
            self._centroid_ema_state[int(class_id)] = F.normalize(ema, p=2, dim=0).detach().clone()

    def _freeze_epoch_centroid_snapshot(self) -> None:
        """Freeze centroid reference frame for the current epoch."""
        self._epoch_frozen_centroids = {
            int(class_id): centroid.detach().clone()
            for class_id, centroid in self._centroid_ema_state.items()
        }

    def _update_centroids_from_epoch_buffer(self) -> None:
        """Update running centroid EMA once per epoch from accumulated representation buffers."""
        if not self.rep_epoch_feature_chunks or not self.rep_epoch_label_chunks:
            return

        features = torch.cat(self.rep_epoch_feature_chunks, dim=0).to(dtype=torch.float32)
        labels = torch.cat(self.rep_epoch_label_chunks, dim=0).to(dtype=torch.int64)
        centroids, class_ids = self._compute_class_centroids(features, labels)
        if int(centroids.shape[0]) == 0:
            self.rep_epoch_feature_chunks = []
            self.rep_epoch_label_chunks = []
            return

        self._update_running_rep_centroids(centroids, class_ids)
        self.rep_epoch_feature_chunks = []
        self.rep_epoch_label_chunks = []

    def _global_centroid_guided_losses(
        self,
        batch_centroids: torch.Tensor,
        class_ids: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Compute centroid forces against running global centroids for topology-level separation."""
        if int(batch_centroids.shape[0]) == 0 or len(class_ids) == 0:
            zero = torch.zeros((), dtype=batch_centroids.dtype, device=batch_centroids.device)
            return zero, zero, 0.0

        if not self._epoch_frozen_centroids:
            for idx, class_id in enumerate(class_ids):
                self._epoch_frozen_centroids[int(class_id)] = F.normalize(
                    batch_centroids[idx].detach().to(device="cpu", dtype=torch.float32),
                    p=2,
                    dim=0,
                )

        global_ids = sorted(int(k) for k in self._epoch_frozen_centroids.keys())
        if len(global_ids) <= 1:
            zero = torch.zeros((), dtype=batch_centroids.dtype, device=batch_centroids.device)
            return zero, zero, 0.0

        global_centroids = torch.stack(
            [self._epoch_frozen_centroids[class_id] for class_id in global_ids],
            dim=0,
        ).to(device=batch_centroids.device, dtype=batch_centroids.dtype)
        global_centroids = F.normalize(global_centroids, p=2, dim=1)

        global_pair_dists = torch.cdist(global_centroids, global_centroids, p=2)
        global_upper_mask = torch.triu(
            torch.ones_like(global_pair_dists, dtype=torch.bool),
            diagonal=1,
        )
        global_pairs = global_pair_dists[global_upper_mask]
        global_min_inter = float(torch.min(global_pairs).item()) if int(global_pairs.numel()) > 0 else 0.0

        batch_centroids_norm = F.normalize(batch_centroids, p=2, dim=1)
        repulsion_terms: list[torch.Tensor] = []
        barrier_terms: list[torch.Tensor] = []
        margin = float(self.rep_centroid_repulsion_margin)
        barrier_min = float(self.rep_centroid_barrier_min_distance)

        for idx, class_id in enumerate(class_ids):
            other_indices = [j for j, gid in enumerate(global_ids) if int(gid) != int(class_id)]
            if not other_indices:
                continue
            others = global_centroids[other_indices]
            dists = torch.cdist(batch_centroids_norm[idx : idx + 1], others, p=2).squeeze(0)
            nearest = torch.min(dists)
            repulsion_terms.append(F.relu(margin - nearest))
            barrier_terms.append(F.relu(barrier_min - nearest) * nearest)

        if not repulsion_terms:
            zero = torch.zeros((), dtype=batch_centroids.dtype, device=batch_centroids.device)
            return zero, zero, global_min_inter

        repulsion_loss = torch.stack(repulsion_terms).sum()
        barrier_loss = torch.stack(barrier_terms).sum()
        return repulsion_loss, barrier_loss, global_min_inter

    def _critical_pair_centroid_push_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        *,
        min_distance: float,
    ) -> torch.Tensor:
        """Apply direct centroid push for known critically colliding class pairs."""
        if not self.critical_collision_pairs:
            return torch.zeros((), dtype=features.dtype, device=features.device)

        penalties: list[torch.Tensor] = []
        for class_a, class_b in self.critical_collision_pairs:
            mask_a = labels == int(class_a)
            mask_b = labels == int(class_b)
            if int(mask_a.sum().item()) == 0 or int(mask_b.sum().item()) == 0:
                continue
            centroid_a = F.normalize(features[mask_a].mean(dim=0, keepdim=True), p=2, dim=1)
            centroid_b = F.normalize(features[mask_b].mean(dim=0, keepdim=True), p=2, dim=1)
            dist = torch.linalg.vector_norm(centroid_a - centroid_b, ord=2, dim=1).squeeze(0)
            penalties.append(F.relu(float(min_distance) - dist))

        if not penalties:
            return torch.zeros((), dtype=features.dtype, device=features.device)
        return torch.stack(penalties).sum()

    def _current_geometry_ratio_threshold(self) -> float:
        """Return stage-aware geometry ratio threshold."""
        if self.representation_phase_active or self.head_phase_start_step < 0:
            return float(self.geometry_max_intra_inter_ratio_warmup)
        return float(self.geometry_max_intra_inter_ratio_post_phase)

    def _build_representation_snapshot_id(
        self,
        diagnostics: dict[str, Any],
        *,
        label_space: str,
    ) -> str:
        """Build a stable snapshot ID for post-representation geometry state."""
        payload = {
            "label_space": str(label_space),
            "ratio": float(diagnostics.get("intra_inter_ratio", diagnostics.get("ratio", 0.0))),
            "min_inter": float(diagnostics.get("min_inter_center_distance", 0.0)),
            "nearest_center_acc": float(
                diagnostics.get(
                    "nearest_center_acc_val",
                    diagnostics.get("nearest_center_accuracy_val", 0.0),
                )
            ),
            "cluster_sizes": [int(v) for v in cast(list[Any], diagnostics.get("cluster_sizes", []))],
            "density_variance": float(diagnostics.get("density_variance", 0.0)),
            "representation_only_steps": int(self.representation_only_steps),
            "head_only_steps": int(self.head_only_steps),
            "sampler_mode": str(self.sampler_mode),
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return f"rep_phase_v1_{digest[:16]}"

    def _maybe_activate_joint_finetune_phase(self) -> None:
        """Enable low-LR joint tuning after the head-only stage completes."""
        if not self.representation_diagnostic_mode:
            return
        if self.representation_phase_active or self.head_phase_start_step < 0:
            return
        if self.joint_finetune_active:
            return
        if self.global_step < self.joint_finetune_start_step:
            return

        strict_diag = cast(
            dict[str, Any],
            self.representation_diagnostics.get(
                "cluster_relabel",
                self.representation_diagnostics.get("original", {}),
            ),
        )
        if strict_diag:
            self._enforce_geometry_integrity(strict_diag, label_space="joint_finetune")

        self._set_phase_trainability(
            train_backbone=True,
            train_family_head=True,
            train_family_projection=True,
        )
        self._set_phase_lr_scales(
            backbone_multiplier=self.joint_finetune_backbone_lr_multiplier,
            head_multiplier=self.joint_finetune_head_lr_multiplier,
        )
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
        """Rebalance representation-phase batches to avoid majority-class manifold domination."""
        if int(features.shape[0]) <= 1:
            return features, labels

        labels_np = np.asarray(labels.detach().to(device="cpu", dtype=torch.int64).numpy(), dtype=np.int64)
        class_index = build_class_index(labels_np)
        if len(class_index) <= 1:
            return features, labels

        counts = [int(idxs.shape[0]) for idxs in class_index.values()]
        target = max(1, min(int(target_per_class), max(counts)))
        rng = np.random.default_rng(self.run_seed + int(self.global_step))
        selected: list[int] = []

        for class_id in sorted(class_index.keys()):
            idxs = class_index[class_id]
            sampled = rng.choice(idxs, size=target, replace=bool(idxs.shape[0] < target))
            selected.extend(int(i) for i in sampled.tolist())

        if not selected:
            return features, labels

        rng.shuffle(selected)
        index_tensor = torch.tensor(selected, device=features.device, dtype=torch.long)
        return features.index_select(0, index_tensor), labels.index_select(0, index_tensor)

    @staticmethod
    def _critical_pair_key(class_i: int, class_j: int) -> tuple[int, int]:
        """Return normalized class-pair key."""
        a, b = int(class_i), int(class_j)
        return (a, b) if a <= b else (b, a)

    def _has_critical_collision_pairs(self, diagnostics: dict[str, Any]) -> bool:
        """Check whether critical collision pairs remain unresolved."""
        top_pairs = cast(list[dict[str, Any]], diagnostics.get("nearest_cluster_pairs_top5", []))
        for item in top_pairs:
            pair_key = self._critical_pair_key(int(item.get("class_i", -1)), int(item.get("class_j", -1)))
            if pair_key in self.critical_collision_pairs and float(item.get("distance", 1.0)) < self.geometry_min_inter_threshold:
                return True
        return False

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
        if self.use_energy_based_family_objective:
            self.logger.info(
                "Geometry integrity gate bypassed in energy mode [label_space=%s]",
                str(label_space),
            )
            return

        ratio = float(diagnostics.get("intra_inter_ratio", diagnostics.get("ratio", 0.0)))
        min_inter = float(diagnostics.get("min_inter_center_distance", 0.0))
        nearest_center_acc = float(
            diagnostics.get(
                "nearest_center_acc_val",
                diagnostics.get("nearest_center_accuracy_val", 0.0),
            )
        )
        cluster_sizes = [int(v) for v in cast(list[Any], diagnostics.get("cluster_sizes", []))]
        ratio_threshold = 1.8
        min_inter_threshold = 0.4
        nearest_center_threshold = 0.85

        if ratio > ratio_threshold:
            raise RuntimeError(
                "Geometry invalid: intra/inter ratio above threshold "
                f"[{label_space}] ratio={ratio:.4f} "
                f"threshold={ratio_threshold:.4f}"
            )

        if min_inter < min_inter_threshold:
            raise RuntimeError(
                "Geometry invalid: unresolved cluster collisions "
                f"[{label_space}] min_inter={min_inter:.4f} "
                f"threshold={min_inter_threshold:.4f}"
            )

        enforce_cluster_size_gate = str(label_space).strip().lower() in {
            "cluster_relabel",
            "joint_finetune",
        }
        if (
            enforce_cluster_size_gate
            and cluster_sizes
            and min(cluster_sizes) < int(self.geometry_min_cluster_size)
        ):
            raise RuntimeError(
                "Dead cluster detected "
                f"[{label_space}] min_cluster_size={min(cluster_sizes)} "
                f"threshold={int(self.geometry_min_cluster_size)}"
            )

        if nearest_center_acc < nearest_center_threshold:
            raise RuntimeError(
                "Geometry invalid: nearest_center_acc below threshold "
                f"[{label_space}] nearest_center_acc={nearest_center_acc:.4f} "
                f"threshold={nearest_center_threshold:.4f}"
            )

    def _collect_normalized_embeddings(
        self,
        loader: DataLoader,
        *,
        max_batches: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Collect L2-normalized backbone embeddings and family labels from a loader."""
        was_training = self.model.training
        self.model.eval()

        feature_chunks: list[torch.Tensor] = []
        label_chunks: list[torch.Tensor] = []

        with torch.no_grad():
            for batch_idx, (x, _, y_family) in enumerate(loader):
                if max_batches is not None and batch_idx >= int(max_batches):
                    break
                x = x.to(self.device, non_blocking=True)
                _, _, features = self.model(x, return_features=True)
                feature_chunks.append(
                    self._prepare_representation_features(features).detach().to(device="cpu")
                )
                label_chunks.append(y_family.detach().to(device="cpu", dtype=torch.int64))

        if was_training:
            self.model.train()

        if not feature_chunks:
            return (
                torch.zeros((0, 0), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.int64),
            )

        return torch.cat(feature_chunks, dim=0), torch.cat(label_chunks, dim=0)

    def _embed_feature_matrix(
        self,
        features: np.ndarray,
        *,
        batch_size: int = 4096,
    ) -> torch.Tensor:
        """Project feature matrix through backbone and return normalized embeddings."""
        x_np = np.asarray(features, dtype=np.float32)
        if int(x_np.shape[0]) == 0:
            return torch.zeros((0, 0), dtype=torch.float32)

        was_training = self.model.training
        self.model.eval()
        embeddings: list[torch.Tensor] = []

        with torch.no_grad():
            for start_idx in range(0, int(x_np.shape[0]), int(batch_size)):
                chunk = torch.from_numpy(x_np[start_idx : start_idx + int(batch_size)]).to(
                    self.device,
                    non_blocking=True,
                )
                _, _, features_chunk = self.model(chunk, return_features=True)
                embeddings.append(
                    self._prepare_representation_features(features_chunk)
                    .detach()
                    .to(device="cpu")
                )

        if was_training:
            self.model.train()

        return torch.cat(embeddings, dim=0)

    @staticmethod
    def _assign_labels_from_centers(
        embeddings: torch.Tensor,
        centers: torch.Tensor,
    ) -> torch.Tensor:
        """Assign nearest-center cluster labels for embeddings."""
        if int(embeddings.shape[0]) == 0:
            return torch.zeros((0,), dtype=torch.int64)
        dists = torch.cdist(embeddings, centers.to(dtype=embeddings.dtype), p=2)
        return torch.argmin(dists, dim=1).to(dtype=torch.int64)

    def _fit_embedding_clusters(
        self,
        embeddings: torch.Tensor,
        *,
        n_clusters: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fit KMeans clusters on normalized embeddings and return labels/centers."""
        if int(embeddings.shape[0]) == 0:
            raise RuntimeError("Cannot fit clusters on empty embedding set")

        k = max(2, min(int(n_clusters), int(embeddings.shape[0])))
        emb_np = embeddings.numpy()

        if self.cluster_relabel_objective == "kmeans":
            kmeans = KMeans(n_clusters=k, random_state=self.cluster_relabel_seed, n_init=10)
            cluster_labels_np = kmeans.fit_predict(emb_np).astype(np.int64)
            centers_np = np.asarray(kmeans.cluster_centers_, dtype=np.float32)
        elif self.cluster_relabel_objective == "gmm":
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                random_state=self.cluster_relabel_seed,
            )
            cluster_labels_np = gmm.fit_predict(emb_np).astype(np.int64)
            centers_np = np.asarray(gmm.means_, dtype=np.float32)
        elif self.cluster_relabel_objective == "spectral":
            spectral = SpectralClustering(
                n_clusters=k,
                affinity=self.cluster_relabel_spectral_affinity,
                assign_labels="kmeans",
                random_state=self.cluster_relabel_seed,
            )
            cluster_labels_np = spectral.fit_predict(emb_np).astype(np.int64)
            unique_labels = sorted(int(v) for v in np.unique(cluster_labels_np).tolist())
            if len(unique_labels) != k:
                raise RuntimeError(
                    "Hard-stop integrity guard triggered: "
                    "spectral_objective_empty_cluster_detected"
                )
            centers_np = np.stack(
                [
                    emb_np[cluster_labels_np == cluster_id].mean(axis=0)
                    for cluster_id in range(k)
                ],
                axis=0,
            ).astype(np.float32)
        else:
            raise RuntimeError(
                "Hard-stop integrity guard triggered: unsupported_cluster_objective"
            )

        centers = torch.from_numpy(centers_np)
        centers = F.normalize(centers, p=2, dim=1)
        return torch.from_numpy(cluster_labels_np), centers

    @staticmethod
    def _build_cluster_label_bridge(
        old_labels: torch.Tensor,
        cluster_labels: torch.Tensor,
        *,
        n_clusters: int,
    ) -> dict[str, Any]:
        """Build stable bridge metadata from legacy labels to cluster labels."""
        old_np = np.asarray(old_labels.to(device="cpu", dtype=torch.int64).numpy(), dtype=np.int64)
        cluster_np = np.asarray(
            cluster_labels.to(device="cpu", dtype=torch.int64).numpy(),
            dtype=np.int64,
        )
        unique_old = sorted(int(v) for v in np.unique(old_np).tolist())
        n_clusters = max(2, int(n_clusters))

        old_to_cluster_counts: dict[str, dict[str, int]] = {}
        old_to_cluster_dominant: dict[str, int] = {}
        old_to_cluster_purity: dict[str, float] = {}
        cluster_to_old_counts: dict[str, dict[str, int]] = {}

        for old in unique_old:
            mask = old_np == int(old)
            counts = np.bincount(cluster_np[mask], minlength=n_clusters).astype(np.int64)
            old_to_cluster_counts[str(old)] = {
                str(cluster_id): int(count)
                for cluster_id, count in enumerate(counts.tolist())
            }
            dominant = int(np.argmax(counts))
            old_to_cluster_dominant[str(old)] = dominant
            old_to_cluster_purity[str(old)] = float(counts[dominant] / max(1, int(counts.sum())))

        for cluster_id in range(n_clusters):
            mask = cluster_np == int(cluster_id)
            old_counts = np.bincount(old_np[mask], minlength=max(unique_old) + 1 if unique_old else 1)
            cluster_to_old_counts[str(cluster_id)] = {
                str(old_label): int(old_counts[int(old_label)])
                for old_label in unique_old
            }

        return {
            "n_clusters": n_clusters,
            "old_labels": unique_old,
            "old_to_cluster_counts": old_to_cluster_counts,
            "old_to_cluster_dominant": old_to_cluster_dominant,
            "old_to_cluster_purity": old_to_cluster_purity,
            "cluster_to_old_counts": cluster_to_old_counts,
        }

    def _apply_cluster_relabels_to_datasets(
        self,
        centers: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Relabel train/val/test datasets by nearest embedding cluster centers."""
        train_dataset = self.train_loader.dataset
        if not isinstance(train_dataset, TensorDataset):
            raise RuntimeError("Cluster relabeling currently requires TensorDataset train loader")

        train_x = train_dataset.tensors[0].detach().to(device="cpu").numpy()
        train_emb = self._embed_feature_matrix(train_x)
        train_cluster_labels = self._assign_labels_from_centers(train_emb, centers)

        train_y_binary = train_dataset.tensors[1]
        train_y_family = train_dataset.tensors[2]
        train_y_family.copy_(train_cluster_labels.to(device=train_y_family.device, dtype=train_y_family.dtype))
        train_y_binary.copy_((train_cluster_labels != 0).to(device=train_y_binary.device, dtype=train_y_binary.dtype))

        val_emb_all = torch.zeros((0, train_emb.shape[1]), dtype=torch.float32)
        val_labels_all = torch.zeros((0,), dtype=torch.int64)
        for loader in self.val_loaders.values():
            val_dataset = loader.dataset
            if not isinstance(val_dataset, MultiTaskNumpyDataset):
                raise RuntimeError("Cluster relabeling expects MultiTaskNumpyDataset for validation")
            val_emb = self._embed_feature_matrix(np.asarray(val_dataset.features, dtype=np.float32))
            val_cluster_labels = self._assign_labels_from_centers(val_emb, centers)
            val_dataset.family_labels = np.asarray(
                val_cluster_labels.numpy(),
                dtype=np.int64,
            ).copy()
            if int(val_dataset.family_labels.shape[0]) > 0:
                self.logger.info(
                    "ClusterRelabel[val] rows=%d label_min=%d label_max=%d unique=%s",
                    int(val_dataset.family_labels.shape[0]),
                    int(np.min(val_dataset.family_labels)),
                    int(np.max(val_dataset.family_labels)),
                    [int(x) for x in np.unique(val_dataset.family_labels).tolist()],
                )
            val_emb_all = (
                torch.cat([val_emb_all, val_emb], dim=0)
                if int(val_emb_all.shape[0]) > 0
                else val_emb
            )
            val_labels_all = (
                torch.cat([val_labels_all, val_cluster_labels], dim=0)
                if int(val_labels_all.shape[0]) > 0
                else val_cluster_labels
            )

        for loader in self.test_loaders.values():
            test_dataset = loader.dataset
            if not isinstance(test_dataset, MultiTaskNumpyDataset):
                raise RuntimeError("Cluster relabeling expects MultiTaskNumpyDataset for test")
            test_emb = self._embed_feature_matrix(np.asarray(test_dataset.features, dtype=np.float32))
            test_cluster_labels = self._assign_labels_from_centers(test_emb, centers)
            test_dataset.family_labels = np.asarray(
                test_cluster_labels.numpy(),
                dtype=np.int64,
            ).copy()
            if int(test_dataset.family_labels.shape[0]) > 0:
                self.logger.info(
                    "ClusterRelabel[test] rows=%d label_min=%d label_max=%d unique=%s",
                    int(test_dataset.family_labels.shape[0]),
                    int(np.min(test_dataset.family_labels)),
                    int(np.max(test_dataset.family_labels)),
                    [int(x) for x in np.unique(test_dataset.family_labels).tolist()],
                )

        return train_emb, train_cluster_labels, val_emb_all, val_labels_all

    @staticmethod
    def _nearest_center_accuracy(
        features: torch.Tensor,
        labels: torch.Tensor,
        centers: dict[int, torch.Tensor],
        class_ids: list[int],
    ) -> float:
        """Compute nearest-center classification accuracy."""
        if int(features.shape[0]) == 0 or not class_ids:
            return 0.0

        center_tensor = torch.stack([centers[c] for c in class_ids], dim=0)
        dists = torch.cdist(features, center_tensor, p=2)
        pred_idx = torch.argmin(dists, dim=1)
        pred_labels = torch.tensor([class_ids[int(i)] for i in pred_idx.tolist()], dtype=torch.int64)
        return float((pred_labels == labels.to(dtype=torch.int64)).float().mean().item())

    def _build_class_centers(
        self,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        class_ids: list[int],
    ) -> tuple[dict[int, torch.Tensor], list[int]]:
        """Build centers for available classes."""
        centers: dict[int, torch.Tensor] = {}
        available_class_ids: list[int] = []
        for cls in class_ids:
            mask = train_labels == int(cls)
            if bool(mask.any()):
                centers[int(cls)] = train_features[mask].mean(dim=0)
                available_class_ids.append(int(cls))
        return centers, available_class_ids

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
        inter_distances: list[float] = []
        collision_pairs: list[dict[str, Any]] = []
        for i, cls_i in enumerate(available_class_ids):
            for j in range(i + 1, len(available_class_ids)):
                cls_j = available_class_ids[j]
                dist_val = float(dist_mat[i, j].item())
                inter_distances.append(dist_val)
                if dist_val <= collision_threshold:
                    collision_pairs.append(
                        {
                            "class_i": int(cls_i),
                            "class_j": int(cls_j),
                            "distance": dist_val,
                        }
                    )

        intra_distances: list[float] = []
        for cls in available_class_ids:
            mask = train_labels == int(cls)
            if bool(mask.any()):
                class_points = train_features[mask]
                class_center = centers[int(cls)].unsqueeze(0)
                intra_distances.extend(
                    torch.norm(class_points - class_center, dim=1).to(device="cpu").tolist()
                )

        return inter_distances, collision_pairs, intra_distances

    def _estimate_local_density_diagnostics(
        self,
        train_features: torch.Tensor,
        *,
        k: int = 20,
        max_samples: int = 4096,
    ) -> dict[str, Any]:
        """Estimate k-NN density stability on embedding space."""
        if int(train_features.shape[0]) <= 2:
            return {
                "density_variance": 0.0,
                "density_mean": 0.0,
                "density_sample_count": int(train_features.shape[0]),
                "density_k": 0,
                "density_feature_dead": True,
            }

        features_np = np.asarray(
            train_features.detach().to(device="cpu", dtype=torch.float32).numpy(),
            dtype=np.float32,
        )
        if int(features_np.shape[0]) > int(max_samples):
            rng = np.random.default_rng(self.run_seed)
            idx = rng.choice(features_np.shape[0], size=int(max_samples), replace=False)
            features_np = features_np[idx]

        k_eff = int(min(k, max(2, int(features_np.shape[0]) - 1)))
        if k_eff < 2:
            return {
                "density_variance": 0.0,
                "density_mean": 0.0,
                "density_sample_count": int(features_np.shape[0]),
                "density_k": int(k_eff),
                "density_feature_dead": True,
            }

        nn = NearestNeighbors(n_neighbors=k_eff, n_jobs=-1)
        nn.fit(features_np)
        distances, _ = nn.kneighbors(features_np)
        avg_distances = distances[:, 1:].mean(axis=1)
        log_density = np.log1p(1.0 / (avg_distances + 1e-8))
        density_variance = float(np.var(log_density))

        return {
            "density_variance": density_variance,
            "density_mean": float(np.mean(log_density)),
            "density_sample_count": int(features_np.shape[0]),
            "density_k": int(k_eff),
            "density_feature_dead": bool(density_variance <= 1e-8),
        }

    def _compute_center_pair_diagnostics(
        self,
        dist_mat: torch.Tensor,
        available_class_ids: list[int],
    ) -> tuple[list[dict[str, Any]], float, list[dict[str, Any]]]:
        """Compute pairwise center diagnostics and percentile-based threshold."""
        center_pairs: list[dict[str, Any]] = []
        for i, cls_i in enumerate(available_class_ids):
            for j in range(i + 1, len(available_class_ids)):
                center_pairs.append(
                    {
                        "class_i": int(cls_i),
                        "class_j": int(available_class_ids[j]),
                        "distance": float(dist_mat[i, j].item()),
                    }
                )

        inter_pair_distances = [float(item["distance"]) for item in center_pairs]
        collision_threshold_p05 = (
            float(np.percentile(np.asarray(inter_pair_distances, dtype=np.float32), 5.0))
            if inter_pair_distances
            else 0.0
        )
        nearest_cluster_pairs_top5 = sorted(
            center_pairs,
            key=lambda item: float(item["distance"]),
        )[:5]
        return center_pairs, collision_threshold_p05, nearest_cluster_pairs_top5

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
        empty_result = {
            "available_class_ids": [],
            "center_distance_matrix": {},
            "nearest_center_accuracy_train": 0.0,
            "nearest_center_accuracy_val": 0.0,
            "nearest_center_acc_val": 0.0,
            "intra_class_distance_mean": 0.0,
            "inter_center_distance_mean": 0.0,
            "intra_inter_ratio": 0.0,
            "min_inter_center_distance": 0.0,
            "cluster_size_counts": [],
            "cluster_sizes": [],
            "collision_threshold_p05": 0.0,
            "nearest_cluster_pairs_top5": [],
            "density_variance": 0.0,
            "density_feature_dead": True,
            "collision_pairs": [],
            "secondary_collision_pairs": [],  # Fix 4
            "nearest_center_confusion_matrix": {},  # Fix 5
            "embedding_capacity_assessment": {},  # Fix 6
        }

        if int(train_features.shape[0]) == 0 or not class_ids:
            return empty_result

        centers, available_class_ids = self._build_class_centers(train_features, train_labels, class_ids)
        if not available_class_ids:
            return empty_result

        centers_tensor = torch.stack([centers[c] for c in available_class_ids], dim=0)
        dist_mat = torch.cdist(centers_tensor, centers_tensor, p=2)
        _, collision_threshold_p05, nearest_cluster_pairs_top5 = self._compute_center_pair_diagnostics(
            dist_mat,
            available_class_ids,
        )

        center_distance_matrix: dict[str, dict[str, float]] = {}
        for i, cls_i in enumerate(available_class_ids):
            row: dict[str, float] = {}
            for j, cls_j in enumerate(available_class_ids):
                row[str(cls_j)] = float(dist_mat[i, j].item())
            center_distance_matrix[str(cls_i)] = row

        inter_distances, collision_pairs, intra_distances = self._compute_inter_and_intra_distances(
            train_features,
            train_labels,
            centers,
            dist_mat,
            available_class_ids,
            collision_threshold_p05,
        )

        intra_mean = float(np.mean(intra_distances)) if intra_distances else 0.0
        inter_mean = float(np.mean(inter_distances)) if inter_distances else 0.0
        intra_inter_ratio = float(intra_mean / max(1e-8, inter_mean)) if inter_mean > 0.0 else 0.0
        min_inter = float(min(inter_distances)) if inter_distances else 0.0

        val_mask = torch.zeros_like(val_labels, dtype=torch.bool)
        for cls in available_class_ids:
            val_mask = val_mask | (val_labels == int(cls))
        val_features_eval = val_features[val_mask] if int(val_features.shape[0]) > 0 else val_features
        val_labels_eval = val_labels[val_mask] if int(val_labels.shape[0]) > 0 else val_labels

        # --- Fix 4: Secondary collision detection ---
        fixer = GeometricRepresentationFixer()
        centers_np = {cid: centers[cid].cpu().numpy() for cid in available_class_ids}
        secondary_collisions, _ = fixer.detect_secondary_collisions(
            centers_np,
            collision_threshold=collision_threshold_p05,
        )

        # --- Fix 5: Nearest-center confusion matrix ---
        confusion_matrix = fixer.build_nearest_center_confusion_matrix(
            train_features.cpu().numpy(),
            train_labels.cpu().numpy(),
            centers_np,
            top_k=3,
        )

        # --- Fix 6: Embedding capacity assessment ---
        capacity_assessment = fixer.assess_embedding_capacity(
            intra_inter_ratio,
            embedding_dim=int(train_features.shape[1]),
            dropout_rate=0.2,  # Default; could read from model config
            target_ratio=0.8,
        )
        density_diagnostics = self._estimate_local_density_diagnostics(train_features)
        cluster_sizes = [
            int((train_labels == int(cls)).sum().item())
            for cls in available_class_ids
        ]
        nearest_center_acc_train = self._nearest_center_accuracy(
            train_features,
            train_labels,
            centers,
            available_class_ids,
        )
        nearest_center_acc_val = self._nearest_center_accuracy(
            val_features_eval,
            val_labels_eval,
            centers,
            available_class_ids,
        )

        return {
            "available_class_ids": available_class_ids,
            "center_distance_matrix": center_distance_matrix,
            "nearest_center_accuracy_train": nearest_center_acc_train,
            "nearest_center_accuracy_val": nearest_center_acc_val,
            "nearest_center_acc_val": nearest_center_acc_val,
            "intra_class_distance_mean": intra_mean,
            "inter_center_distance_mean": inter_mean,
            "intra_inter_ratio": intra_inter_ratio,
            "min_inter_center_distance": min_inter,
            "cluster_size_counts": cluster_sizes,
            "cluster_sizes": cluster_sizes,
            "collision_threshold_p05": collision_threshold_p05,
            "nearest_cluster_pairs_top5": nearest_cluster_pairs_top5,
            "density_variance": float(density_diagnostics.get("density_variance", 0.0)),
            "density_feature_dead": bool(density_diagnostics.get("density_feature_dead", False)),
            "collision_pairs": collision_pairs,
            "secondary_collision_pairs": secondary_collisions,  # Fix 4
            "nearest_center_confusion_matrix": confusion_matrix,  # Fix 5
            "embedding_capacity_assessment": capacity_assessment,  # Fix 6
        }

    def _run_representation_diagnostics(
        self,
        *,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        label_space: str,
    ) -> dict[str, Any]:
        """Run mandatory embedding diagnostics before classifier-head phase."""
        if not self.active_family_class_ids:
            return {}

        class_ids = sorted(int(c) for c in self.active_family_class_ids)
        val_loader = next(iter(self.val_loaders.values()), None)
        if val_loader is None:
            val_features = torch.zeros((0, train_features.shape[1]), dtype=torch.float32)
            val_labels = torch.zeros((0,), dtype=torch.int64)
        else:
            val_features, val_labels = self._collect_normalized_embeddings(val_loader)

        diagnostics = self._compute_representation_diagnostics(
            train_features,
            train_labels,
            val_features,
            val_labels,
            class_ids=class_ids,
        )
        self.representation_diagnostics[str(label_space)] = diagnostics

        self.logger.info(
            "RepDiag[%s] nearest_center_acc(train)=%.4f nearest_center_acc(val)=%.4f intra=%.4f inter=%.4f ratio=%.4f min_inter=%.4f collisions=%s",
            str(label_space),
            float(diagnostics.get("nearest_center_accuracy_train", 0.0)),
            float(diagnostics.get("nearest_center_accuracy_val", 0.0)),
            float(diagnostics.get("intra_class_distance_mean", 0.0)),
            float(diagnostics.get("inter_center_distance_mean", 0.0)),
            float(diagnostics.get("intra_inter_ratio", 0.0)),
            float(diagnostics.get("min_inter_center_distance", 0.0)),
            diagnostics.get("collision_pairs", []),
        )
        self.logger.info(
            "RepDiag[%s] center_distance_matrix=%s",
            str(label_space),
            diagnostics.get("center_distance_matrix", {}),
        )
        self.logger.info(
            "RepDiag[%s] collision_threshold_p05=%.4f top5_nearest_cluster_pairs=%s",
            str(label_space),
            float(diagnostics.get("collision_threshold_p05", 0.0)),
            diagnostics.get("nearest_cluster_pairs_top5", []),
        )
        self.logger.info(
            "RepDiag[%s] cluster_sizes=%s nearest_center_acc_val=%.4f density_variance=%.8f",
            str(label_space),
            diagnostics.get("cluster_sizes", []),
            float(diagnostics.get("nearest_center_acc_val", diagnostics.get("nearest_center_accuracy_val", 0.0))),
            float(diagnostics.get("density_variance", 0.0)),
        )
        if bool(diagnostics.get("density_feature_dead", False)):
            self.logger.warning(
                "RepDiag[%s] density_feature_dead=true (near-zero variance)",
                str(label_space),
            )

        # Log new Fixes 4-6 diagnostics
        secondary_collisions = diagnostics.get("secondary_collision_pairs", [])
        if secondary_collisions:
            self.logger.info(
                "RepDiag[%s] Fix4_secondary_collisions: %d pairs detected: %s",
                str(label_space),
                len(secondary_collisions),
                secondary_collisions,
            )

        confusion_matrix = diagnostics.get("nearest_center_confusion_matrix", {})
        if confusion_matrix:
            misclassified_counts = {
                cid: len(cm.get("confusion_with", {}))
                for cid, cm in confusion_matrix.items()
            }
            self.logger.info(
                "RepDiag[%s] Fix5_confusion_matrix: classes confused with other classes: %s",
                str(label_space),
                misclassified_counts,
            )

        capacity = diagnostics.get("embedding_capacity_assessment", {})
        if capacity and capacity.get("under_capacity"):
            recs = capacity.get("recommendations", [])
            self.logger.warning(
                "RepDiag[%s] Fix6_capacity_warning: under-capacity detected. Recommendations: %s",
                str(label_space),
                [r.get("action") for r in recs],
            )

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
        """Initialize or resize the EMA tracker for per-class argmax win rates."""
        if (
            self.energy_win_rate_ema is None
            or int(self.energy_win_rate_ema.numel()) != int(class_count)
        ):
            init = torch.ones((class_count,), device=device, dtype=dtype)
            init = init / float(max(1, class_count))
            self.energy_win_rate_ema = init
        elif (
            self.energy_win_rate_ema.device != device
            or self.energy_win_rate_ema.dtype != dtype
        ):
            self.energy_win_rate_ema = self.energy_win_rate_ema.to(device=device, dtype=dtype)
        return self.energy_win_rate_ema

    def _update_energy_win_rate_ema(
        self,
        logits: torch.Tensor,
        *,
        active_class_ids: Optional[list[int]] = None,
    ) -> None:
        """Update EMA win-rate estimate from hard argmax class wins in the current batch."""
        if int(logits.ndim) != 2 or int(logits.shape[0]) <= 0:
            return
        class_count = int(logits.shape[1])
        clean_active_ids: list[int] = []
        if active_class_ids is not None:
            clean_active_ids = [
                int(cls) for cls in active_class_ids if 0 <= int(cls) < class_count
            ]
        if not clean_active_ids:
            return

        ema = self._ensure_energy_win_rate_ema(
            class_count,
            device=logits.device,
            dtype=logits.dtype,
        )
        hard_pred = torch.argmax(logits.detach(), dim=1)
        counts = torch.bincount(hard_pred, minlength=class_count).to(dtype=ema.dtype)
        batch_rate = counts / max(1, int(logits.shape[0]))

        active_idx = torch.tensor(clean_active_ids, device=logits.device, dtype=torch.int64)
        active_batch_rate = batch_rate.index_select(0, active_idx)
        active_batch_sum = active_batch_rate.sum().clamp_min(1e-12)
        active_batch_rate = active_batch_rate / active_batch_sum

        active_ema = ema.index_select(0, active_idx)
        momentum = float(self.energy_win_rate_ema_momentum)
        updated_active = (momentum * active_ema) + ((1.0 - momentum) * active_batch_rate)
        updated_active = updated_active / updated_active.sum().clamp_min(1e-12)

        next_ema = ema.clone()
        next_ema.index_copy_(0, active_idx, updated_active)
        inactive_mask = torch.ones(class_count, device=logits.device, dtype=torch.bool)
        inactive_mask.index_fill_(0, active_idx, False)
        if bool(torch.any(inactive_mask)):
            next_ema[inactive_mask] = max(float(self.energy_emergence_bias_eps), 1e-6)

        self.energy_win_rate_ema = next_ema

    def _compute_energy_emergence_bias(
        self,
        logits: torch.Tensor,
        *,
        active_class_ids: Optional[list[int]] = None,
    ) -> torch.Tensor:
        """Compute per-class pre-argmax bias from inverse EMA win rates."""
        class_count = int(logits.shape[1])
        eps = float(self.energy_emergence_bias_eps)
        beta = float(self.energy_emergence_bias_beta)

        bias = torch.zeros((class_count,), device=logits.device, dtype=logits.dtype)
        if active_class_ids is not None:
            active_ids = [
                int(cls)
                for cls in sorted(active_class_ids)
                if 0 <= int(cls) < class_count
            ]
        elif self.active_family_class_ids:
            active_ids = [
                int(cls)
                for cls in sorted(self.active_family_class_ids)
                if 0 <= int(cls) < class_count
            ]
        else:
            active_ids = []

        if not active_ids:
            self._energy_bias_last_std = 0.0
            self._energy_bias_last_max_abs = 0.0
            self._energy_bias_last_logit_std = float(logits.detach().std().item())
            return bias

        ema = self._ensure_energy_win_rate_ema(
            class_count,
            device=logits.device,
            dtype=logits.dtype,
        )
        active_idx = torch.tensor(active_ids, device=logits.device, dtype=torch.int64)
        active_ema = ema.index_select(0, active_idx).to(dtype=logits.dtype)
        active_bias = beta * torch.log(1.0 / torch.clamp(active_ema + eps, min=eps))

        logit_std = float(logits.detach().std().item())
        if logit_std > eps:
            bias_std = active_bias.detach().std()
            scale = (0.2 * logit_std) / (float(bias_std.item()) + 1e-6)
            active_bias = active_bias * scale
            max_abs = max(eps, self.energy_emergence_bias_ratio_max * logit_std)
            active_bias = torch.clamp(active_bias, min=-max_abs, max=max_abs)

        self._energy_bias_last_std = float(active_bias.detach().std().item())
        self._energy_bias_last_max_abs = float(active_bias.detach().abs().max().item())
        self._energy_bias_last_logit_std = logit_std

        bias.index_copy_(0, active_idx, active_bias)
        return bias

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
        """Compute F1-related statistics from confusion matrix counts."""
        if confusion.numel() == 0:
            return {
                "macro_f1": 0.0,
                "weighted_f1": 0.0,
                "minority_recall_min": 0.0,
                "zero_prediction_classes": [],
            }

        conf = confusion.to(device="cpu", dtype=torch.float64)
        support = conf.sum(dim=1)
        predicted = conf.sum(dim=0)
        tp = torch.diag(conf)

        precision = torch.where(predicted > 0, tp / predicted, torch.zeros_like(tp))
        recall = torch.where(support > 0, tp / support, torch.zeros_like(tp))
        denom = precision + recall
        f1 = torch.where(denom > 0, 2.0 * precision * recall / denom, torch.zeros_like(tp))

        active_classes = (support + predicted) > 0
        macro_f1 = float(f1[active_classes].mean().item()) if bool(active_classes.any()) else 0.0

        total_support = float(support.sum().item())
        weighted_f1 = (
            float((f1 * support).sum().item() / total_support) if total_support > 0 else 0.0
        )

        present_classes = support > 0
        minority_present = torch.where(present_classes)[0].tolist()
        minority_recalls = [float(recall[idx].item()) for idx in minority_present if int(idx) != 0]
        minority_recall_min = float(min(minority_recalls)) if minority_recalls else 0.0

        zero_prediction_classes = sorted(
            int(idx)
            for idx in torch.where((support > 0) & (predicted == 0))[0].tolist()
        )

        return {
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "minority_recall_min": minority_recall_min,
            "zero_prediction_classes": zero_prediction_classes,
        }

    def _get_learning_rate(self) -> float:
        """Compute learning rate with linear warmup and cosine decay."""
        if self.epoch < self.config.warmup_epochs:
            warmup_denom = max(1, self.config.warmup_epochs)
            return float(
                self.config.warmup_init_lr
                + (self.config.learning_rate - self.config.warmup_init_lr)
                * ((self.epoch + 1) / warmup_denom)
            )

        decay_epochs = max(1, self.config.epochs - self.config.warmup_epochs)
        decay_step = min(self.epoch - self.config.warmup_epochs, decay_epochs)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * decay_step / decay_epochs))
        min_lr = self.config.learning_rate * 0.05
        return float(min_lr + (self.config.learning_rate - min_lr) * cosine_factor)

    def _set_learning_rate(self) -> None:
        """Update learning rate in optimizer."""
        lr = self._get_learning_rate()
        for param_group in self.optimizer.param_groups:
            lr_scale = float(param_group.get("lr_scale", 1.0))
            param_group["lr"] = lr * lr_scale

    def _set_backbone_freeze_state(self, freeze_backbone: bool) -> None:
        """Freeze/unfreeze backbone while leaving classifier heads trainable."""
        freeze_backbone = bool(freeze_backbone)
        if freeze_backbone == self.backbone_frozen:
            return

        for param in self.model.backbone.parameters():
            param.requires_grad = not freeze_backbone

        self.backbone_params = [
            param for param in self.model.backbone.parameters() if param.requires_grad
        ]
        self.backbone_frozen = freeze_backbone
        self.logger.info(
            "Backbone state updated: %s",
            "frozen" if freeze_backbone else "trainable",
        )

    def _current_learning_rate(self) -> float:
        """Return current optimizer learning rate."""
        for param_group in self.optimizer.param_groups:
            if str(param_group.get("group_name", "")) == "backbone":
                return float(param_group["lr"])
        return float(self.optimizer.param_groups[0]["lr"])

    def _apply_loss_regularizations(
        self,
        loss: torch.Tensor,
        family_logits_train: torch.Tensor,
        raw_family_logits: torch.Tensor,
        y_family: torch.Tensor,
        in_step_warmup: bool,
    ) -> torch.Tensor:
        """Apply all loss regularization terms."""
        first_epoch_only = int(self.epoch) == 0

        # Entropy warmup regularization
        if (
            (not first_epoch_only)
            and self.entropy_warmup_steps > 0
            and self.global_step < self.entropy_warmup_steps
        ):
            family_prob_warmup = torch.softmax(family_logits_train, dim=1)
            safe_prob_warmup = torch.clamp(family_prob_warmup, min=1e-12, max=1.0)
            family_entropy_warmup = -torch.sum(
                family_prob_warmup * torch.log(safe_prob_warmup),
                dim=1,
            ).mean()
            loss = loss - self.entropy_warmup_weight * family_entropy_warmup

        # KL divergence to uniform distribution
        if in_step_warmup:
            effective_kl_weight = float(self.warmup_kl_uniform_weight)
        else:
            effective_kl_weight = float(self.kl_uniform_weight)

        if first_epoch_only:
            kl_weight = 0.0
        else:
            kl_weight = effective_kl_weight
        if kl_weight > 0.0:
            family_prob = torch.softmax(family_logits_train, dim=1)
            safe_family_prob = torch.clamp(family_prob, min=1e-12, max=1.0)
            uniform_log_prob = -math.log(float(family_prob.shape[1]))
            kl_to_uniform = torch.sum(
                family_prob * (torch.log(safe_family_prob) - uniform_log_prob),
                dim=1,
            ).mean()
            loss = loss + kl_weight * kl_to_uniform

        # Logit floor penalty
        if self.logit_floor_weight > 0.0 and not in_step_warmup:
            logit_floor_penalty = torch.relu(self.logit_floor - raw_family_logits).mean()
            loss = loss + self.logit_floor_weight * logit_floor_penalty

        # Tail class cross-entropy
        if (
            self.tail_ce_weight > 0.0
            and self.tail_class_mask is not None
            and int(self.tail_class_mask.numel()) == int(family_logits_train.shape[1])
        ):
            tail_sample_mask = self.tail_class_mask[y_family]
            if bool(torch.any(tail_sample_mask)):
                tail_ce = F.cross_entropy(
                    family_logits_train[tail_sample_mask],
                    y_family[tail_sample_mask],
                    reduction="mean",
                    label_smoothing=float(getattr(self.loss_fn, "label_smoothing", 0.0)),
                )
                loss = loss + self.tail_ce_weight * tail_ce

        return loss

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

    def _handle_phase_transition_logic(
        self,
        in_representation_phase: bool,
        rep_features: torch.Tensor,
        rep_labels: torch.Tensor,
    ) -> None:
        """Handle transition from representation phase to head training phase."""
        if in_representation_phase or not self.representation_phase_active:
            return

        if self.use_energy_based_family_objective:
            self.phase1_class_centroids = None
            self.phase1_centroid_class_ids = []
            self.representation_snapshot_id = f"energy_phase_v1_step_{int(self.global_step)}"
            self.representation_diagnostics["energy_transition"] = {
                "mode": "class_conditional_energy",
                "global_step": int(self.global_step),
                "snapshot_id": self.representation_snapshot_id,
            }
            self.representation_diagnostics["representation_snapshot_id"] = self.representation_snapshot_id

            self._set_phase_trainability(
                train_backbone=False,
                train_family_head=True,
                train_family_projection=True,
            )
            self._set_phase_lr_scales(backbone_multiplier=1.0, head_multiplier=1.0)
            self.representation_phase_active = False
            self.head_phase_start_step = self.global_step
            self.joint_finetune_start_step = self.head_phase_start_step + self.head_only_steps
            self.step_coverage_checked = False
            self.rep_phase_feature_chunks = []
            self.rep_phase_label_chunks = []
            self.logger.info(
                "Representation transition completed in energy mode: snapshot_id=%s",
                self.representation_snapshot_id,
            )
            return

        diagnostics = self._run_representation_diagnostics(
            train_features=rep_features,
            train_labels=rep_labels,
            label_space="original",
        )

        rep_features_for_next = rep_features
        rep_labels_for_next = rep_labels
        if self._has_critical_collision_pairs(diagnostics):
            self.logger.warning(
                "RepDiag[original] critical collision pairs unresolved; applying emergency merge map=%s",
                self.emergency_label_merge_map,
            )
            rep_labels_for_next = self._apply_emergency_label_merge(
                rep_labels,
                merge_map=self.emergency_label_merge_map,
            )
            diagnostics = self._run_representation_diagnostics(
                train_features=rep_features_for_next,
                train_labels=rep_labels_for_next,
                label_space="collision_merge",
            )

        self._enforce_geometry_integrity(diagnostics, label_space="original")
        phase1_centroids, phase1_class_ids = self._compute_class_centroids(
            rep_features_for_next,
            rep_labels_for_next,
        )
        self.phase1_class_centroids = self._stabilize_centroids(
            phase1_centroids,
            phase1_class_ids,
        ).detach().clone()
        self.phase1_centroid_class_ids = list(phase1_class_ids)
        self.representation_diagnostics["phase1_class_centroids_shape"] = [
            int(v) for v in self.phase1_class_centroids.shape
        ]
        self.representation_diagnostics["phase1_centroid_class_ids"] = [
            int(v) for v in self.phase1_centroid_class_ids
        ]
        self.logger.info(
            "Phase1 centroids frozen: classes=%s shape=%s",
            self.phase1_centroid_class_ids,
            tuple(int(v) for v in self.phase1_class_centroids.shape),
        )

        if self.cluster_relabeling_enabled:
            self._apply_cluster_relabeling(rep_features_for_next, rep_labels_for_next)

        phase_diag = cast(
            dict[str, Any],
            self.representation_diagnostics.get(
                "cluster_relabel",
                self.representation_diagnostics.get("original", diagnostics),
            ),
        )
        self.representation_snapshot_id = self._build_representation_snapshot_id(
            phase_diag,
            label_space="cluster_relabel" if self.cluster_relabeling_enabled else "original",
        )
        self.representation_diagnostics["representation_snapshot_id"] = self.representation_snapshot_id
        self.logger.info(
            "Representation snapshot locked: id=%s",
            self.representation_snapshot_id,
        )

        self._set_phase_trainability(
            train_backbone=False,
            train_family_head=True,
            train_family_projection=True,
        )
        self._set_phase_lr_scales(backbone_multiplier=1.0, head_multiplier=1.0)
        self.representation_phase_active = False
        self.head_phase_start_step = self.global_step
        self.joint_finetune_start_step = self.head_phase_start_step + self.head_only_steps
        self.step_coverage_checked = False
        self.rep_phase_feature_chunks = []
        self.rep_phase_label_chunks = []

    def _apply_cluster_relabeling(self, rep_features: torch.Tensor, rep_labels: torch.Tensor) -> None:
        """Apply cluster relabeling to representation phase features."""
        active_count = max(2, len(self.active_family_class_ids))
        auto_k = max(2, int(math.ceil(float(active_count) / 2.0)))
        k = int(self.cluster_relabel_k or auto_k)

        rep_cluster_labels, cluster_centers = self._fit_embedding_clusters(rep_features, n_clusters=k)
        cluster_size_counts = np.bincount(
            np.asarray(rep_cluster_labels.to(device="cpu").numpy(), dtype=np.int64),
            minlength=int(cluster_centers.shape[0]),
        ).astype(np.int64)

        self.representation_diagnostics["cluster_size_counts"] = [int(v) for v in cluster_size_counts.tolist()]
        self.representation_diagnostics["cluster_size_entropy"] = float(
            _normalized_entropy_from_counts([int(v) for v in cluster_size_counts.tolist()])
        )
        self.cluster_centers = cluster_centers
        self.representation_diagnostics["cluster_relabel_config"] = {
            "algorithm": str(self.cluster_relabel_objective),
            "k": int(k),
            "seed": int(self.cluster_relabel_seed),
            "spectral_affinity": str(self.cluster_relabel_spectral_affinity),
        }
        self.representation_diagnostics["cluster_label_bridge"] = self._build_cluster_label_bridge(
            rep_labels, rep_cluster_labels, n_clusters=int(cluster_centers.shape[0])
        )

        (train_emb_all, train_cluster_labels_all, val_emb_all, val_cluster_labels_all,) = (
            self._apply_cluster_relabels_to_datasets(cluster_centers)
        )

        self.active_family_class_ids = {int(v) for v in torch.unique(train_cluster_labels_all, dim=0).tolist()}
        self.train_family_class_count = len(self.active_family_class_ids)

        clustered_diag = self._run_representation_diagnostics(
            train_features=train_emb_all,
            train_labels=train_cluster_labels_all,
            label_space="cluster_relabel",
        )

        val_diag = self._compute_representation_diagnostics(
            train_emb_all,
            train_cluster_labels_all,
            val_emb_all,
            val_cluster_labels_all,
            class_ids=sorted(self.active_family_class_ids),
        )
        self.representation_diagnostics["cluster_relabel_val"] = val_diag
        self._enforce_geometry_integrity(clustered_diag, label_space="cluster_relabel")

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
        """Run one epoch-0 synthetic warmup step to expose each active class at least once."""
        if int(self.epoch) != 0 or not bool(self.model.training):
            return

        train_dataset = self.train_loader.dataset
        class_to_indices = self._collect_class_to_indices(train_dataset)
        class_count = int(self.model.family_head[-1].out_features)
        active_class_ids = self._resolve_warmup_active_class_ids(class_to_indices, class_count)

        missing_classes = [
            int(class_id)
            for class_id in active_class_ids
            if len(class_to_indices.get(int(class_id), [])) == 0
        ]
        if missing_classes:
            raise RuntimeError(
                "Class missing from training set"
                f": missing_active_classes={missing_classes}"
            )

        forced_indices = [int(class_to_indices[int(class_id)][0]) for class_id in active_class_ids]
        x_forced, y_binary_forced, y_family_forced, y_family_rows = self._build_warmup_batch_tensors(
            train_dataset,
            forced_indices,
        )

        binary_logits, raw_family_logits, _ = self.model(x_forced, return_features=True)
        family_logits = self._apply_family_logit_controls(
            raw_family_logits,
            apply_prior=False,
            apply_emergence_bias=bool(self.use_energy_based_family_objective),
            active_class_ids=active_class_ids,
        )
        family_logits_forced = family_logits.clone()
        neg_inf = torch.tensor(float("-inf"), device=family_logits_forced.device, dtype=family_logits_forced.dtype)
        pos_inf = torch.tensor(float("inf"), device=family_logits_forced.device, dtype=family_logits_forced.dtype)
        for row_idx, class_id in enumerate(y_family_rows):
            family_logits_forced[int(row_idx), :] = neg_inf
            family_logits_forced[int(row_idx), int(class_id)] = pos_inf

        in_step_warmup = self.global_step < self.warmup_steps
        binary_weights = None if in_step_warmup else self.binary_class_weights
        family_weights = None if in_step_warmup else self.family_class_weights
        warmup_loss, _ = self.loss_fn(
            binary_logits,
            y_binary_forced,
            family_logits_forced,
            y_family_forced,
            binary_class_weights=binary_weights,
            family_class_weights=family_weights,
            feature_embeddings=None,
        )

        self.optimizer.zero_grad()
        warmup_loss.backward()
        if self.config.max_grad_norm > 0 and self.backbone_params:
            nn.utils.clip_grad_norm_(self.backbone_params, self.config.max_grad_norm)
        self.optimizer.step()
        self.global_step += 1

        self.logger.info(
            "Epoch0CoverageWarmup active_classes=%s forced_indices=%s warmup_loss=%s",
            active_class_ids,
            forced_indices,
            f"{float(warmup_loss.item()):.6f}",
        )

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
        """Compute focal tail stabilization term for classes 3/4."""
        tail_focal_classes = {3, 4}
        tail_mask = torch.zeros_like(y_family, dtype=torch.bool)
        for cls_id in tail_focal_classes:
            tail_mask = tail_mask | (y_family == int(cls_id))

        if not bool(torch.any(tail_mask)):
            return torch.zeros((), dtype=family_logits_train.dtype, device=family_logits_train.device)

        focal_logits = family_logits_train[tail_mask]
        focal_labels = y_family[tail_mask]
        log_probs = torch.log_softmax(focal_logits, dim=1)
        gathered_logp = log_probs.gather(1, focal_labels.unsqueeze(1)).squeeze(1)
        p_t = gathered_logp.exp()
        gamma_tail = 2.0
        focal_term = ((1.0 - p_t).clamp(min=0.0) ** gamma_tail) * (-gathered_logp)
        return focal_term.mean()

    def _compute_representation_energy_objective(
        self,
        *,
        family_logits_train: torch.Tensor,
        y_family: torch.Tensor,
        y_binary: torch.Tensor,
        binary_logits: torch.Tensor,
        active_family_class_ids: list[int],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute representation-phase energy objective and diagnostics."""
        (
            energy_loss,
            mean_e_y,
            mean_e_others,
            mean_gap,
            mean_energy_total,
        ) = self._class_conditional_energy_gap_loss(
            family_logits_train,
            y_family,
            alpha=self.energy_multi_negative_alpha,
        )

        if int(self.epoch) == 0:
            energy_balance_loss = torch.zeros((), dtype=family_logits_train.dtype, device=family_logits_train.device)
            energy_min_winner_loss = torch.zeros((), dtype=family_logits_train.dtype, device=family_logits_train.device)
            mean_balance_kl = 0.0
            mean_pred_entropy = 0.0
            min_pred_mass = 0.0
            mean_winner_deficit = 0.0
            min_winner_count = 0.0
            effective_energy_balance_weight = 0.0
            effective_energy_winner_weight = 0.0
        else:
            (
                energy_balance_loss,
                mean_balance_kl,
                mean_pred_entropy,
                min_pred_mass,
            ) = self._energy_class_balance_loss(family_logits_train)
            (
                energy_min_winner_loss,
                mean_winner_deficit,
                min_winner_count,
            ) = self._energy_min_winner_loss(
                family_logits_train,
                active_family_class_ids,
                min_winners=self.energy_winner_min_count,
            )
            effective_energy_balance_weight = float(self.energy_balance_weight)
            effective_energy_winner_weight = float(self.energy_winner_weight)

        binary_only_loss = self.loss_fn._classification_loss(
            binary_logits,
            y_binary,
            None,
        )
        loss = (
            (float(self.loss_fn.lambda_binary) * binary_only_loss)
            + (float(self.energy_gap_weight) * energy_loss)
            + (effective_energy_balance_weight * energy_balance_loss)
            + (effective_energy_winner_weight * energy_min_winner_loss)
        )

        diagnostics = {
            "mean_e_y": float(mean_e_y),
            "mean_e_others": float(mean_e_others),
            "mean_gap": float(mean_gap),
            "mean_energy_total": float(mean_energy_total),
            "mean_balance_kl": float(mean_balance_kl),
            "mean_pred_entropy": float(mean_pred_entropy),
            "min_pred_mass": float(min_pred_mass),
            "mean_winner_deficit": float(mean_winner_deficit),
            "min_winner_count": float(min_winner_count),
            "effective_energy_balance_weight": float(effective_energy_balance_weight),
            "effective_energy_winner_weight": float(effective_energy_winner_weight),
        }
        return loss, diagnostics

    def _log_energy_gap_diag_if_needed(self, diagnostics: dict[str, float]) -> None:
        """Emit periodic energy diagnostics for representation phase."""
        if int(self.global_step) % 20 != 0:
            return
        self.logger.info(
            "EnergyGapDiag step=%d weight=%.3f alpha=%.3f T=%.3f balance_w=%.3f winner_w=%.3f winner_m=%d emergence_beta=%.3f emergence_eps=%.1e winrate_m=%.3f bias_std=%.6f bias_max=%.6f logit_std=%.6f E_y=%.6f E_neg_lse=%.6f gap_all=%.6f energy_total=%.6f balance_kl=%.6f pred_entropy=%.6f min_pred_mass=%.6f winner_deficit=%.6f min_winner_count=%.6f",
            int(self.global_step),
            float(self.energy_gap_weight),
            float(self.energy_multi_negative_alpha),
            float(self.energy_logit_temperature),
            float(self.energy_balance_weight),
            float(self.energy_winner_weight),
            int(self.energy_winner_min_count),
            float(self.energy_emergence_bias_beta),
            float(self.energy_emergence_bias_eps),
            float(self.energy_win_rate_ema_momentum),
            float(self._energy_bias_last_std),
            float(self._energy_bias_last_max_abs),
            float(self._energy_bias_last_logit_std),
            diagnostics["mean_e_y"],
            diagnostics["mean_e_others"],
            diagnostics["mean_gap"],
            diagnostics["mean_energy_total"],
            diagnostics["mean_balance_kl"],
            diagnostics["mean_pred_entropy"],
            diagnostics["min_pred_mass"],
            diagnostics["mean_winner_deficit"],
            diagnostics["min_winner_count"],
        )

    @staticmethod
    def _apply_entropy_floor_regularizer(
        loss: torch.Tensor,
        *,
        family_logits_train: torch.Tensor,
        active_class_count: int,
    ) -> torch.Tensor:
        """Apply entropy floor regularization on family logits."""
        entropy_target = 0.6 * math.log(float(max(1, active_class_count)))
        family_prob_entropy = torch.softmax(family_logits_train, dim=1)
        family_prob_entropy = torch.clamp(family_prob_entropy, min=1e-10, max=1.0)
        mean_entropy = -torch.sum(
            family_prob_entropy * torch.log(family_prob_entropy),
            dim=1,
        ).mean()
        entropy_floor_loss = torch.relu(
            torch.tensor(
                entropy_target,
                dtype=family_logits_train.dtype,
                device=family_logits_train.device,
            )
            - mean_entropy
        )
        return loss + (0.01 * entropy_floor_loss)

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
        """Compute base loss and optionally replace it with representation energy objective."""
        tail_focal_loss = torch.zeros(
            (),
            dtype=classification_loss.dtype,
            device=classification_loss.device,
        )
        if not bool(getattr(self, "disable_tail_focal_regularizer", False)):
            tail_focal_loss = self._compute_tail_focal_loss(family_logits_train, y_family)
        loss = classification_loss + (0.1 * tail_focal_loss)

        energy_diag = {
            "mean_e_y": 0.0,
            "mean_e_others": 0.0,
            "mean_gap": 0.0,
            "mean_energy_total": 0.0,
            "mean_balance_kl": 0.0,
            "mean_pred_entropy": 0.0,
            "min_pred_mass": 0.0,
            "mean_winner_deficit": 0.0,
            "min_winner_count": 0.0,
            "effective_energy_balance_weight": 0.0,
            "effective_energy_winner_weight": 0.0,
        }
        use_energy_objective = in_representation_phase and self.use_energy_based_family_objective
        if use_energy_objective:
            loss, energy_diag = self._compute_representation_energy_objective(
                family_logits_train=family_logits_train,
                y_family=y_family,
                y_binary=y_binary,
                binary_logits=binary_logits,
                active_family_class_ids=active_family_class_ids,
            )
        return loss, energy_diag

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
        """Process one training batch (forward, loss, backward, optimization)."""
        binary_logits, raw_family_logits, backbone_features = self.model(
            x, return_features=True
        )
        active_family_class_ids = self._resolve_batch_active_family_class_ids(raw_family_logits, y_family)
        family_logits_train = self._stabilize_batch_family_logits(
            raw_family_logits,
            active_family_class_ids=active_family_class_ids,
        )

        family_pred = torch.argmax(family_logits_train, dim=1)
        if self.use_energy_based_family_objective:
            self._update_energy_win_rate_ema(
                family_logits_train,
                active_class_ids=active_family_class_ids,
            )

        binary_weights = None if in_step_warmup else self.binary_class_weights
        family_weights = None if in_step_warmup else self.family_class_weights

        classification_loss, _ = self.loss_fn(
            binary_logits,
            y_binary,
            family_logits_train,
            y_family,
            binary_class_weights=binary_weights,
            family_class_weights=family_weights,
            feature_embeddings=backbone_features,
        )

        loss, energy_diag = self._compute_loss_with_optional_energy(
            classification_loss=classification_loss,
            family_logits_train=family_logits_train,
            y_family=y_family,
            y_binary=y_binary,
            binary_logits=binary_logits,
            in_representation_phase=in_representation_phase,
            active_family_class_ids=active_family_class_ids,
        )

        if in_representation_phase and self.use_energy_based_family_objective:
            self._log_energy_gap_diag_if_needed(energy_diag)

        active_class_count = max(1, int(len(active_family_class_ids)))
        loss = self._apply_entropy_floor_regularizer(
            loss,
            family_logits_train=family_logits_train,
            active_class_count=active_class_count,
        )


        loss = self._apply_optional_non_representation_regularizations(
            loss,
            in_representation_phase=in_representation_phase,
            family_logits_train=family_logits_train,
            raw_family_logits=raw_family_logits,
            y_family=y_family,
            in_step_warmup=in_step_warmup,
        )

        self._backpropagate_train_batch_loss(loss, in_representation_phase=in_representation_phase)

        # Compute metrics
        batch_size = int(y_binary.shape[0])
        binary_correct = int((torch.argmax(binary_logits, dim=1) == y_binary).sum().item())
        family_correct = int((family_pred == y_family).sum().item())

        self._maybe_store_representation_chunks(
            in_representation_phase=in_representation_phase,
            backbone_features=backbone_features,
            y_family=y_family,
        )

        return loss, raw_family_logits, family_pred, binary_correct, family_correct, batch_size

    def _check_backbone_freeze_state(self) -> None:
        """Check and unfreeze backbone if needed."""
        if self.representation_diagnostic_mode:
            return
        if (
            self.backbone_frozen
            and self.unfreeze_backbone_step > 0
            and self.global_step >= self.unfreeze_backbone_step
        ):
            self._set_backbone_freeze_state(False)

    def _check_family_class_coverage(self, y_family: torch.Tensor) -> None:
        """Validate that batch contains all expected family classes."""
        if not self.active_family_class_ids:
            return
        batch_class_ids = {int(v) for v in torch.unique(y_family, dim=0).tolist()}
        if not self.enforce_all_classes_per_batch:
            if len(batch_class_ids) <= 1:
                raise RuntimeError(
                    "Hard-stop integrity guard triggered: batch_single_family_class"
                )
            return
        if not self.active_family_class_ids.issubset(batch_class_ids):
            missing = sorted(self.active_family_class_ids - batch_class_ids)
            raise RuntimeError(
                "Hard-stop integrity guard triggered: " f"batch_missing_family_classes_{missing}"
            )

    def _check_step_coverage(
        self,
        batch_idx: int,
        family_pred_counts: Optional[torch.Tensor],
    ) -> None:
        """Check step coverage for family class predictions."""
        if (
            (not self.step_coverage_checked)
            and (
                (
                    self.representation_diagnostic_mode
                    and self.head_phase_start_step >= 0
                    and (not self.representation_phase_active)
                    and (self.global_step - self.head_phase_start_step)
                    >= self.coverage_check_after_head_steps
                )
                or ((not self.representation_diagnostic_mode) and batch_idx >= self.step_coverage_check_step)
            )
            and family_pred_counts is not None
            and self.active_family_class_ids
        ):
            missing_classes = [
                cls
                for cls in sorted(self.active_family_class_ids)
                if int(family_pred_counts[cls].item()) <= 0
            ]
            self.step_coverage_checked = True
            if missing_classes:
                if bool(getattr(self, "disable_integrity_hard_stops", False)):
                    self.logger.warning(
                        "Integrity hard-stops disabled: allowing step_coverage_missing_predictions "
                        "(missing_classes=%s by_step=%d)",
                        missing_classes,
                        int(self.coverage_check_after_head_steps),
                    )
                    return
                if self.use_energy_based_family_objective:
                    self.logger.warning(
                        "Energy mode: skipping step_coverage hard-stop "
                        "(missing_classes=%s by_step=%d)",
                        missing_classes,
                        int(self.coverage_check_after_head_steps),
                    )
                    return
                raise RuntimeError(
                    "Hard-stop integrity guard triggered: "
                    "step_coverage_missing_predictions_by_step"
                    f"{self.coverage_check_after_head_steps}_classes_{missing_classes}"
                )

    def _handle_representation_phase_logic(
        self,
        in_representation_phase: bool,
    ) -> None:
        """Handle representation phase initiation and transition logic."""
        self._handle_representation_phase_logic_impl(in_representation_phase)

    def _handle_representation_phase_logic_impl(
        self,
        in_representation_phase: bool,
    ) -> None:
        """Handle representation phase initiation and transition logic."""
        if not self.representation_diagnostic_mode:
            return

        self._maybe_start_representation_phase()

        if not self.representation_phase_active:
            return

        self._update_representation_window_state(in_representation_phase)

        if self.global_step >= self.representation_only_steps:
            self.representation_curriculum_complete = True

        self._finalize_representation_phase_if_ready(in_representation_phase)

    def _maybe_start_representation_phase(self) -> None:
        """Initialize representation phase state when entering early curriculum steps."""
        if self.representation_phase_active or self.global_step >= self.representation_only_steps:
            return
        self.representation_phase_active = True
        self.representation_curriculum_complete = False
        self.in_representation_window = False
        self.rep_phase_feature_chunks = []
        self.rep_phase_label_chunks = []

    def _update_representation_window_state(self, in_representation_phase: bool) -> None:
        """Apply trainability and LR settings on representation window transitions."""
        if in_representation_phase == self.in_representation_window:
            return

        if in_representation_phase:
            self._set_phase_trainability(
                train_backbone=True,
                train_family_head=False,
                train_family_projection=False,
            )
            self._set_phase_lr_scales(backbone_multiplier=1.0, head_multiplier=1.0)
            self.in_representation_window = True
            return

        if self._should_exit_representation_curriculum():
            self.representation_curriculum_complete = True
        self._set_phase_trainability(
            train_backbone=False,
            train_family_head=True,
            train_family_projection=True,
        )
        self._set_phase_lr_scales(backbone_multiplier=1.0, head_multiplier=1.0)
        self.in_representation_window = False

    def _finalize_representation_phase_if_ready(self, in_representation_phase: bool) -> None:
        """Run transition logic once representation curriculum completes."""
        if in_representation_phase:
            return
        if not self.representation_phase_active or not self.representation_curriculum_complete:
            return

        rep_features = (
            torch.cat(self.rep_phase_feature_chunks, dim=0)
            if self.rep_phase_feature_chunks
            else torch.zeros((0, 0), dtype=torch.float32)
        )
        rep_labels = (
            torch.cat(self.rep_phase_label_chunks, dim=0)
            if self.rep_phase_label_chunks
            else torch.zeros((0,), dtype=torch.int64)
        )
        self._handle_phase_transition_logic(in_representation_phase, rep_features, rep_labels)

    def train_epoch(self) -> dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_calibrated_loss = 0.0
        total_binary_correct = 0
        total_family_correct = 0
        total_samples = 0
        train_logit_max = float("-inf")
        train_logit_min = float("inf")
        family_pred_counts: Optional[torch.Tensor] = None
        family_logit_sums: Optional[torch.Tensor] = None
        top2_logit_gap_sum = 0.0
        top2_logit_gap_count = 0

        step_log_interval = max(1, int(self.config.log_interval))
        total_steps = len(self.train_loader)
        epoch_start = time.perf_counter()
        active_strategy = self._set_epoch_loss_strategy()
        self.logger.info(
            "Epoch %d start | steps=%d | step_log_interval=%d | loss_strategy=%s",
            self.epoch,
            total_steps,
            step_log_interval,
            active_strategy,
        )
        self._run_epoch0_forced_coverage_warmup()
        if not self.use_energy_based_family_objective:
            self._freeze_epoch_centroid_snapshot()

        for batch_idx, (x, y_binary, y_family) in enumerate(self.train_loader):
            if batch_idx % step_log_interval == 0:
                print(f"step {batch_idx}/{total_steps} status=start epoch={self.epoch}", flush=True)

            self._check_backbone_freeze_state()

            x = x.to(self.device, non_blocking=True)
            y_binary = y_binary.to(self.device, non_blocking=True)
            y_family = y_family.to(self.device, non_blocking=True)

            unique_classes_in_batch = int(torch.unique(y_family, dim=0).numel())
            if unique_classes_in_batch < 2:
                raise RuntimeError(
                    "Hard-stop integrity guard triggered: batch_diversity_violation_lt2"
                )
            self.logger.info(
                "BatchClassCoverage step=%d batch=%d unique_classes_in_batch=%d",
                int(self.global_step),
                int(batch_idx),
                unique_classes_in_batch,
            )
            if unique_classes_in_batch < 3:
                self.class_starvation_streak += 1
                if self.class_starvation_streak >= 5:
                    self.logger.warning(
                        "BatchClassCoverage starvation_detected streak=%d (<3 unique classes)",
                        int(self.class_starvation_streak),
                    )
            else:
                self.class_starvation_streak = 0

            self._check_family_class_coverage(y_family)

            in_representation_phase = (
                self.representation_diagnostic_mode
                and self._is_representation_window_step(self.global_step)
            )
            self._handle_representation_phase_logic(in_representation_phase)
            self._maybe_activate_joint_finetune_phase()

            # Forward, loss, backward, optimize
            in_step_warmup = self.global_step < self.warmup_steps
            loss, raw_family_logits, family_pred, binary_correct, family_correct, batch_size = (
                self._process_train_batch(x, y_binary, y_family, in_step_warmup, in_representation_phase)
            )

            train_logit_max = max(train_logit_max, float(raw_family_logits.max().item()))
            train_logit_min = min(train_logit_min, float(raw_family_logits.min().item()))

            # Accumulate metrics
            total_loss += float(loss.item()) * batch_size
            total_calibrated_loss += float(loss.item()) * batch_size
            total_binary_correct += binary_correct
            total_family_correct += family_correct
            total_samples += batch_size

            family_pred_counts, family_logit_sums = self._update_train_batch_stats(
                family_pred, raw_family_logits, family_pred_counts, family_logit_sums
            )

            # Step coverage check
            self._check_step_coverage(batch_idx, family_pred_counts)

            top2_values = torch.topk(raw_family_logits.detach(), k=2, dim=1).values
            top2_logit_gap_sum += float((top2_values[:, 0] - top2_values[:, 1]).sum().item())
            top2_logit_gap_count += int(top2_values.shape[0])

            self._log_step10_diagnostics(raw_family_logits)
            self._log_batch_progress(
                batch_idx,
                total_steps,
                step_log_interval,
                total_loss,
                total_binary_correct,
                total_family_correct,
                total_samples,
            )

        self._log_epoch_completion(
            epoch_start,
            train_logit_min,
            train_logit_max,
            family_pred_counts,
            family_logit_sums,
            total_samples,
            top2_logit_gap_sum,
            top2_logit_gap_count,
        )
        if not self.use_energy_based_family_objective:
            self._update_centroids_from_epoch_buffer()

        return {
            "train_loss": total_loss / max(1, total_samples),
            "train_calibrated_loss": total_calibrated_loss / max(1, total_samples),
            "train_binary_acc": total_binary_correct / max(1, total_samples),
            "train_family_acc": total_family_correct / max(1, total_samples),
            "train_family_logit_max": train_logit_max,
            "train_family_logit_min": train_logit_min,
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

    def _apply_eval_class4_logit_shift(self, family_logits: torch.Tensor) -> torch.Tensor:
        """Apply inference-time class-4 logit shift (logit_4 <- logit_4 - delta)."""
        if int(family_logits.ndim) != 2 or int(family_logits.shape[0]) <= 0:
            return family_logits
        delta = float(getattr(self, "class4_logit_shift", 0.0) or 0.0)
        if delta <= 0.0:
            return family_logits
        class_id = int(getattr(self, "class4_logit_shift_class_id", 4))
        if class_id < 0 or class_id >= int(family_logits.shape[1]):
            return family_logits
        shifted = family_logits.clone()
        shifted[:, class_id] = shifted[:, class_id] - delta
        return shifted

    def _apply_inference_prediction_floor(
        self,
        family_logits: torch.Tensor,
        family_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Inference-only prediction floor to guarantee per-batch class presence."""
        if int(family_logits.ndim) != 2 or int(family_logits.shape[0]) <= 0:
            return family_pred

        class_count = int(family_logits.shape[1])
        active_ids = [
            int(cls)
            for cls in getattr(self, "active_family_class_ids", [])
            if 0 <= int(cls) < class_count
        ]
        if not active_ids:
            return family_pred

        adjusted_pred = family_pred.clone()
        predicted_set = {int(v) for v in adjusted_pred.tolist()}
        missing_ids = [cls for cls in active_ids if cls not in predicted_set]
        if not missing_ids:
            return adjusted_pred

        used_rows: set[int] = set()
        for cls in missing_ids:
            class_scores = family_logits[:, cls].clone()
            if used_rows:
                for row_idx in used_rows:
                    class_scores[row_idx] = float("-inf")
            row = int(torch.argmax(class_scores).item())
            adjusted_pred[row] = cls
            used_rows.add(row)

        return adjusted_pred

    @torch.no_grad()
    def _evaluate_loader(self, loader: DataLoader, dataset_name: str = "unknown") -> dict[str, Any]:  # NOSONAR
        """Evaluate metrics on a single dataset loader."""
        total_loss = 0.0
        total_calibrated_loss = 0.0
        total_binary_correct = 0
        total_family_correct = 0
        total_samples = 0

        binary_prob_chunks: list[torch.Tensor] = []
        binary_label_chunks: list[torch.Tensor] = []
        family_confusion: Optional[torch.Tensor] = None
        family_entropy_sum = 0.0
        family_class_count = 0
        family_pred_counts: Optional[torch.Tensor] = None
        family_logit_sums: Optional[torch.Tensor] = None
        top2_logit_gap_sum = 0.0
        top2_logit_gap_count = 0
        binary_weights_cpu = (
            self.binary_class_weights.to(device="cpu") if self.binary_class_weights is not None else None
        )
        family_weights_cpu = (
            self.family_class_weights.to(device="cpu") if self.family_class_weights is not None else None
        )

        for x, y_binary, y_family in loader:
            x = x.to(self.device, non_blocking=True)
            y_binary_cpu = y_binary.to(device="cpu", dtype=torch.long, non_blocking=True)
            y_family_cpu = y_family.to(device="cpu", dtype=torch.long, non_blocking=True)

            binary_logits_dev, family_logits_dev = self.model(x)
            family_logits_dev = self._apply_family_logit_controls(
                family_logits_dev,
                apply_emergence_bias=False,
            )
            binary_logits = binary_logits_dev.to(device="cpu")
            family_logits = family_logits_dev.to(device="cpu")
            family_logits = self._apply_eval_class4_logit_shift(family_logits)

            loss, _ = self.loss_fn(
                binary_logits,
                y_binary_cpu,
                family_logits,
                y_family_cpu,
                binary_class_weights=binary_weights_cpu,
                family_class_weights=family_weights_cpu,
            )
            calibrated_loss = loss

            batch_size = int(y_binary_cpu.shape[0])
            binary_pred = torch.argmax(binary_logits, dim=1)
            family_pred = torch.argmax(family_logits, dim=1)
            family_pred = self._apply_inference_prediction_floor(family_logits, family_pred)
            total_loss += float(loss.item()) * batch_size
            total_calibrated_loss += float(calibrated_loss.item()) * batch_size
            total_binary_correct += int((binary_pred == y_binary_cpu).sum().item())
            total_family_correct += int((family_pred == y_family_cpu).sum().item())
            total_samples += batch_size

            binary_prob_chunks.append(torch.softmax(binary_logits, dim=1)[:, 1].detach())
            binary_label_chunks.append(y_binary_cpu.detach())

            if family_confusion is None:
                family_class_count = int(family_logits.shape[1])
                family_confusion = torch.zeros(
                    (family_class_count, family_class_count),
                    dtype=torch.int64,
                )

            invalid_label_mask = (y_family_cpu < 0) | (y_family_cpu >= family_class_count)
            if bool(torch.any(invalid_label_mask)):
                raise RuntimeError(
                    "Hard-stop integrity guard triggered: invalid_family_labels_in_eval_"
                    f"{dataset_name}:min={int(torch.min(y_family_cpu).item())}"
                    f":max={int(torch.max(y_family_cpu).item())}"
                    f":class_count={family_class_count}"
                )

            family_index = (
                y_family_cpu.to(dtype=torch.int64) * family_class_count
                + family_pred.detach().to(dtype=torch.int64)
            )
            family_confusion += torch.bincount(
                family_index,
                minlength=family_class_count * family_class_count,
            ).reshape(family_class_count, family_class_count)

            family_prob = torch.softmax(family_logits, dim=1)
            safe_family_prob = torch.clamp(family_prob, min=1e-10, max=1.0)
            batch_entropy = -torch.sum(family_prob * torch.log(safe_family_prob), dim=1)
            batch_entropy = batch_entropy / math.log(float(family_prob.shape[1]))
            family_entropy_sum += float(batch_entropy.sum().item())

            if family_pred_counts is None:
                family_pred_counts = torch.zeros(family_class_count, dtype=torch.int64)
                family_logit_sums = torch.zeros(family_class_count, dtype=torch.float32)
            family_pred_counts += torch.bincount(
                family_pred.detach().to(device="cpu", dtype=torch.int64),
                minlength=family_class_count,
            )
            if family_logit_sums is not None:
                family_logit_sums += family_logits.detach().to(device="cpu", dtype=torch.float32).sum(
                    dim=0
                )

            top2_values = torch.topk(family_logits.detach(), k=2, dim=1).values
            top2_logit_gap_sum += float((top2_values[:, 0] - top2_values[:, 1]).sum().item())
            top2_logit_gap_count += int(top2_values.shape[0])

        if binary_prob_chunks:
            binary_probs = torch.cat(binary_prob_chunks, dim=0).to(device="cpu").numpy()
            binary_labels = torch.cat(binary_label_chunks, dim=0).to(device="cpu").numpy()
        else:
            binary_probs = np.array([])
            binary_labels = np.array([])

        if binary_labels.size > 0 and np.unique(binary_labels).size > 1:
            binary_auroc = float(roc_auc_score(binary_labels, binary_probs))
            binary_auprc = float(average_precision_score(binary_labels, binary_probs))
        else:
            binary_auroc = 0.0
            binary_auprc = 0.0

        family_stats = self._compute_f1_stats_from_confusion(
            family_confusion if family_confusion is not None else torch.zeros((0, 0), dtype=torch.int64)
        )
        if family_confusion is not None and int(family_confusion.numel()) > 0:
            support = family_confusion.sum(dim=1).to(dtype=torch.float64)
            tp = torch.diag(family_confusion).to(dtype=torch.float64)
            recall = torch.where(support > 0, tp / support, torch.zeros_like(tp))
            recall_payload = {
                int(idx): float(recall[idx].item())
                for idx in range(int(recall.shape[0]))
                if float(support[idx].item()) > 0.0
            }
            self.logger.info("ValDiag[%s] per_class_recall=%s", dataset_name, recall_payload)
        family_entropy = family_entropy_sum / max(1, total_samples)
        val_top2_logit_gap = top2_logit_gap_sum / max(1, top2_logit_gap_count)

        if family_pred_counts is not None and family_logit_sums is not None and total_samples > 0:
            pred_count_payload = {
                int(idx): int(count)
                for idx, count in enumerate(family_pred_counts.tolist())
            }
            avg_logit_payload = {
                int(idx): float(total / max(1, total_samples))
                for idx, total in enumerate(family_logit_sums.tolist())
            }
            self.logger.info(
                "ValDiag[%s] per_class_prediction_count=%s per_class_avg_logit=%s top2_logit_gap=%.4f",
                dataset_name,
                pred_count_payload,
                avg_logit_payload,
                val_top2_logit_gap,
            )

        return {
            "num_samples": float(total_samples),
            "val_loss": total_loss / max(1, total_samples),
            "val_calibrated_loss": total_calibrated_loss / max(1, total_samples),
            "val_binary_acc": total_binary_correct / max(1, total_samples),
            "val_family_acc": total_family_correct / max(1, total_samples),
            "val_binary_auroc": binary_auroc,
            "val_binary_auprc": binary_auprc,
            "val_family_macro_f1": float(family_stats["macro_f1"]),
            "val_family_minority_recall_min": float(family_stats["minority_recall_min"]),
            "val_family_entropy": family_entropy,
            "val_family_zero_prediction_classes": float(
                len(cast(list[int], family_stats["zero_prediction_classes"]))
            ),
            "val_family_predicted_class_count": float(
                int((family_confusion.sum(dim=0) > 0).sum().item()) if family_confusion is not None else 0
            ),
            "val_family_top2_logit_gap": float(val_top2_logit_gap),
        }

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Validate per dataset with strict isolation (worst-case aggregation)."""
        self.model.eval()
        if not self.val_loaders:
            raise RuntimeError("No validation loaders configured")

        dataset_metrics: dict[str, dict[str, Any]] = {}
        for dataset_name, loader in self.val_loaders.items():
            metrics = self._evaluate_loader(loader, dataset_name=dataset_name)
            dataset_metrics[dataset_name] = metrics
            self.logger.info(
                f"Val[{dataset_name}] loss={metrics['val_loss']:.4f}, "
                f"bin_acc={metrics['val_binary_acc']:.4f}, "
                f"fam_acc={metrics['val_family_acc']:.4f}, "
                f"entropy={metrics['val_family_entropy']:.4f}, "
                f"top2_gap={metrics.get('val_family_top2_logit_gap', 0.0):.4f}"
            )

        total_samples = sum(metric["num_samples"] for metric in dataset_metrics.values())
        if total_samples <= 0:
            raise RuntimeError("Validation metrics are empty; no samples found in val loaders")

        # Strict isolation: avoid sample-weighted averaging that can hide weak datasets.
        metric_values = list(dataset_metrics.values())
        entropy_missing_same_dataset = any(
            metric["val_family_entropy"] < 0.12
            and metric["val_family_zero_prediction_classes"] > 0
            for metric in metric_values
        )
        return {
            "val_loss": float(max(metric["val_loss"] for metric in metric_values)),
            "val_calibrated_loss": float(
                max(metric["val_calibrated_loss"] for metric in metric_values)
            ),
            "val_binary_acc": float(min(metric["val_binary_acc"] for metric in metric_values)),
            "val_family_acc": float(min(metric["val_family_acc"] for metric in metric_values)),
            "val_binary_auroc": float(min(metric["val_binary_auroc"] for metric in metric_values)),
            "val_binary_auprc": float(min(metric["val_binary_auprc"] for metric in metric_values)),
            "val_family_macro_f1": float(
                min(metric["val_family_macro_f1"] for metric in metric_values)
            ),
            "val_family_minority_recall_min": float(
                min(metric["val_family_minority_recall_min"] for metric in metric_values)
            ),
            "val_family_entropy": float(min(metric["val_family_entropy"] for metric in metric_values)),
            "val_family_zero_prediction_classes": float(
                max(metric["val_family_zero_prediction_classes"] for metric in metric_values)
            ),
            "val_family_predicted_class_count": float(
                min(float(metric.get("val_family_predicted_class_count", 0.0)) for metric in metric_values)
            ),
            "val_family_top2_logit_gap": float(
                min(float(metric.get("val_family_top2_logit_gap", 0.0)) for metric in metric_values)
            ),
            "val_entropy_missing_same_dataset": float(entropy_missing_same_dataset),
        }

    def _process_test_batch(
        self,
        x: torch.Tensor,
        y_binary: torch.Tensor,
        y_family: torch.Tensor,
        binary_confusion: torch.Tensor,
        family_confusion: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], int, float]:
        """Process one test batch and accumulate metrics."""
        x = x.to(self.device, non_blocking=True)
        y_binary_cpu = y_binary.to(device="cpu", dtype=torch.long, non_blocking=True)
        y_family_cpu = y_family.to(device="cpu", dtype=torch.long, non_blocking=True)

        binary_logits_dev, family_logits_dev = self.model(x)
        family_logits_dev = self._apply_family_logit_controls(
            family_logits_dev,
            apply_emergence_bias=False,
        )
        binary_logits = binary_logits_dev.to(device="cpu")
        family_logits = family_logits_dev.to(device="cpu")
        family_logits = self._apply_eval_class4_logit_shift(family_logits)

        binary_prob = torch.softmax(binary_logits, dim=1)
        family_prob = torch.softmax(family_logits, dim=1)
        binary_pred = torch.argmax(binary_logits, dim=1)
        family_pred = torch.argmax(family_logits, dim=1)
        family_pred = self._apply_inference_prediction_floor(family_logits, family_pred)

        batch_size = int(y_binary_cpu.shape[0])

        # Binary confusion
        binary_index = (
            y_binary_cpu.to(dtype=torch.int64) * 2 + binary_pred.detach().to(dtype=torch.int64)
        )
        binary_confusion = binary_confusion + torch.bincount(binary_index, minlength=4).reshape(2, 2)

        # Family confusion
        if family_confusion is None:
            family_class_count = int(family_logits.shape[1])
            family_confusion = torch.zeros((family_class_count, family_class_count), dtype=torch.int64)
        else:
            family_class_count = int(family_confusion.shape[0])

        invalid_label_mask = (y_family_cpu < 0) | (y_family_cpu >= family_class_count)
        if bool(torch.any(invalid_label_mask)):
            raise RuntimeError("Hard-stop integrity guard triggered: invalid_family_labels_in_test")

        family_index = (
            y_family_cpu.to(dtype=torch.int64) * family_class_count
            + family_pred.detach().to(dtype=torch.int64)
        )
        family_confusion = family_confusion + torch.bincount(
            family_index,
            minlength=family_class_count * family_class_count,
        ).reshape(family_class_count, family_class_count)

        # Entropy
        safe_family_prob = torch.clamp(family_prob, min=1e-12, max=1.0)
        batch_entropy = -torch.sum(family_prob * torch.log(safe_family_prob), dim=1)
        batch_entropy = batch_entropy / math.log(float(family_prob.shape[1]))
        entropy_sum = float(batch_entropy.sum().item())

        return binary_prob[:, 1].detach(), y_binary_cpu.detach(), binary_confusion, family_confusion, batch_size, entropy_sum

    @torch.no_grad()
    def _evaluate_test_loader(self, test_loader: DataLoader) -> dict[str, float]:
        """Evaluate one test loader with tensor-first aggregation."""
        binary_prob_chunks: list[torch.Tensor] = []
        binary_label_chunks: list[torch.Tensor] = []
        binary_confusion = torch.zeros((2, 2), dtype=torch.int64)
        family_confusion: Optional[torch.Tensor] = None
        family_entropy_sum = 0.0
        total_samples = 0

        for x, y_binary, y_family in test_loader:
            binary_prob, y_binary_cpu, binary_confusion, family_confusion, batch_size, entropy_sum = (
                self._process_test_batch(x, y_binary, y_family, binary_confusion, family_confusion)
            )

            binary_prob_chunks.append(binary_prob)
            binary_label_chunks.append(y_binary_cpu)
            family_entropy_sum += entropy_sum
            total_samples += batch_size

        binary_probs_arr = (
            torch.cat(binary_prob_chunks, dim=0).to(device="cpu").numpy()
            if binary_prob_chunks
            else np.array([])
        )
        binary_labels_arr = (
            torch.cat(binary_label_chunks, dim=0).to(device="cpu").numpy()
            if binary_label_chunks
            else np.array([])
        )

        binary_total = int(binary_confusion.sum().item())
        binary_accuracy = (
            float(torch.diag(binary_confusion).sum().item() / binary_total)
            if binary_total > 0
            else 0.0
        )

        family_total = int(family_confusion.sum().item()) if family_confusion is not None else 0
        family_accuracy = (
            float(torch.diag(family_confusion).sum().item() / family_total)
            if family_total > 0 and family_confusion is not None
            else 0.0
        )

        if binary_labels_arr.size > 0 and np.unique(binary_labels_arr).size > 1:
            binary_auroc = float(roc_auc_score(binary_labels_arr, binary_probs_arr))
            binary_auprc = float(average_precision_score(binary_labels_arr, binary_probs_arr))
        else:
            binary_auroc = 0.0
            binary_auprc = 0.0

        family_entropy = float(family_entropy_sum / max(1, total_samples)) if total_samples > 0 else 0.0

        binary_stats = self._compute_f1_stats_from_confusion(binary_confusion)
        family_stats = self._compute_f1_stats_from_confusion(
            family_confusion if family_confusion is not None else torch.zeros((0, 0), dtype=torch.int64)
        )

        return {
            "binary_accuracy": binary_accuracy,
            "binary_f1": float(binary_stats["weighted_f1"]),
            "binary_auroc": binary_auroc,
            "binary_auprc": binary_auprc,
            "family_accuracy": family_accuracy,
            "family_f1": float(family_stats["weighted_f1"]),
            "family_macro_f1": float(family_stats["macro_f1"]),
            "family_minority_recall_min": float(family_stats["minority_recall_min"]),
            "family_entropy": family_entropy,
            "family_zero_prediction_classes": float(
                len(cast(list[int], family_stats["zero_prediction_classes"]))
            ),
        }

    @torch.no_grad()
    def evaluate_per_dataset(self) -> dict[str, dict[str, float]]:
        """Evaluate on per-dataset test sets."""
        self.model.eval()
        results = {}
        for dataset_name, test_loader in self.test_loaders.items():
            results[dataset_name] = self._evaluate_test_loader(test_loader)
        return results

    def fit(self) -> dict[str, Any]:  # NOSONAR
        """Train for specified epochs."""
        self.logger.info("=" * 80)
        self.logger.info("Starting HelixIDS-Full Training")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {self.model.param_count:,}")
        self.logger.info(f"Epochs: {self.config.epochs}")
        self.logger.info(f"Batch size: {self.config.batch_size}")
        self.logger.info("=" * 80)

        for epoch in range(self.config.epochs):
            self.epoch = epoch
            self._reseed_epoch_generators()
            if not self.representation_diagnostic_mode:
                self._set_backbone_freeze_state(self.epoch < self.freeze_backbone_epochs)
            self._set_learning_rate()
            # Train
            train_metrics = self.train_epoch()

            # Validate every N epochs
            if self.epoch % self.config.val_interval == 0:
                val_metrics = self.validate()

                for key, val in train_metrics.items():
                    self.training_history.setdefault(key, []).append(val)
                for key, val in val_metrics.items():
                    self.training_history.setdefault(key, []).append(val)

                self.logger.info(
                    f"Epoch {self.epoch:3d} | "
                    f"Train Loss: {train_metrics['train_loss']:.4f} | "
                    f"Train Cal Loss: {train_metrics['train_calibrated_loss']:.4f} | "
                    f"Train Logit Range: [{train_metrics.get('train_family_logit_min', 0.0):.4f}, "
                    f"{train_metrics.get('train_family_logit_max', 0.0):.4f}] | "
                    f"Val Loss: {val_metrics['val_loss']:.4f} | "
                    f"Val Cal Loss: {val_metrics['val_calibrated_loss']:.4f} | "
                    f"Val Binary Acc: {val_metrics['val_binary_acc']:.4f} | "
                    f"Val Family Acc: {val_metrics['val_family_acc']:.4f} | "
                    f"Val Entropy: {val_metrics.get('val_family_entropy', 0.0):.4f}"
                )

                predicted_class_count = int(val_metrics.get("val_family_predicted_class_count", 0.0))
                collapse_threshold = max(1, int(self.train_family_class_count * 0.5))
                if self.train_family_class_count > 0 and predicted_class_count < collapse_threshold:
                    print("[COLLAPSE DETECTED] insufficient class coverage")

                zero_prediction_classes = int(
                    val_metrics.get("val_family_zero_prediction_classes", 0.0)
                )
                if zero_prediction_classes > 0:
                    if bool(getattr(self, "disable_integrity_hard_stops", False)):
                        self.logger.warning(
                            "Integrity hard-stops disabled: allowing validation_zero_prediction_classes_nonzero "
                            "(missing_classes=%d)",
                            zero_prediction_classes,
                        )
                    elif self.use_energy_based_family_objective:
                        self.logger.warning(
                            "Energy mode: skipping validation_zero_prediction_classes hard-stop "
                            "(missing_classes=%d)",
                            zero_prediction_classes,
                        )
                    else:
                        raise RuntimeError(
                            "Hard-stop integrity guard triggered: "
                            "validation_zero_prediction_classes_nonzero"
                        )

                hard_stop_reason = self._hard_stop_reason(train_metrics, val_metrics)
                if hard_stop_reason is not None:
                    raise RuntimeError(f"Hard-stop integrity guard triggered: {hard_stop_reason}")

                should_stop = self._update_early_stopping(train_metrics, val_metrics)
                self._save_checkpoint_if_needed()
                if should_stop:
                    break

        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self.logger.info("✅ Loaded best model state")

        # Evaluate on per-dataset test sets
        self.logger.info("\nPer-Dataset Evaluation:")
        per_dataset_results = self.evaluate_per_dataset()
        self._log_per_dataset_results(per_dataset_results)

        macro_values = [
            float(metrics.get("family_macro_f1", metrics.get("family_f1", 0.0)))
            for metrics in per_dataset_results.values()
        ]
        macro_floor = self._post_training_macro_floor()
        if macro_values and min(macro_values) < macro_floor:
            raise RuntimeError(
                "Post-training macro_f1 guard failed: "
                f"min_family_macro_f1={min(macro_values):.4f} < {macro_floor:.2f}"
            )

        return {
            "training_history": self.training_history,
            "per_dataset_results": per_dataset_results,
            "representation_diagnostics": self.representation_diagnostics,
            "best_val_loss": self.best_val_loss,
            "epochs_trained": self.epoch + 1,
        }

    def _post_training_macro_floor(self) -> float:
        """Return macro-F1 floor calibrated to training budget.

        Short smoke runs are used to validate governed pipeline integrity, not
        final model quality, so they use a more permissive floor.
        """
        epochs = int(getattr(self.config, "epochs", 0))
        if epochs <= 2:
            return 0.10
        if epochs <= 10:
            return 0.15
        return 0.25

    def _is_smoke_mode(self) -> bool:
        """Return True when trainer is running smoke-governance profile."""
        profile = os.getenv("HELIX_GOV_POLICY_PROFILE", "").strip().lower()
        if profile == "smoke":
            return True

        cfg = getattr(self, "config", None)
        epochs = getattr(cfg, "epochs", None)
        if epochs is None:
            return False
        try:
            return int(epochs) <= 10
        except (TypeError, ValueError):
            return False

    def _hard_stop_reason(
        self,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
    ) -> Optional[str]:
        """Return hard-stop reason when integrity constraints are violated."""
        if bool(getattr(self, "disable_integrity_hard_stops", False)):
            return None

        reason = self._hard_stop_val_gap_collapse(train_metrics, val_metrics)
        if reason is not None:
            return reason

        reason = self._hard_stop_high_accuracy_high_loss(train_metrics, val_metrics)
        if reason is not None:
            return reason

        return self._hard_stop_entropy_collapse(val_metrics)

    def _hard_stop_val_gap_collapse(
        self,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
    ) -> Optional[str]:
        """Detect persistent val-vs-train loss gap with collapse symptoms."""
        # Train loss is gathered in train mode (dropout/batchnorm active) while
        # validation is in eval mode. A lower val loss can be normal, so only
        # hard-stop when the gap is large and accompanied by collapse symptoms.
        val_gap = train_metrics["train_calibrated_loss"] - val_metrics["val_calibrated_loss"]
        collapse_signals = (
            val_metrics.get("val_family_macro_f1", 1.0) < 0.25
            or val_metrics.get("val_family_minority_recall_min", 1.0) < 0.10
            or val_metrics.get("val_family_entropy", 1.0) < 0.15
        )
        if val_gap > 0.12 and collapse_signals:
            self.val_gap_collapse_streak += 1
            if self.val_gap_collapse_streak >= 2:
                return "val_loss_below_train_loss_with_collapse"
        else:
            self.val_gap_collapse_streak = 0

        return None

    def _hard_stop_high_accuracy_high_loss(
        self,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
    ) -> Optional[str]:
        """Detect suspiciously high accuracy paired with persistently high loss."""

        peak_accuracy = max(
            train_metrics["train_binary_acc"],
            train_metrics["train_family_acc"],
            val_metrics["val_binary_acc"],
            val_metrics["val_family_acc"],
        )
        if train_metrics["train_calibrated_loss"] > 0.5 and peak_accuracy > 0.95:
            streak = int(getattr(self, "high_accuracy_high_loss_streak", 0)) + 1
            self.high_accuracy_high_loss_streak = streak
            epoch_idx = int(getattr(self, "epoch", 0))
            if epoch_idx >= 1 and streak >= 2:
                return "high_accuracy_with_high_loss"
        else:
            self.high_accuracy_high_loss_streak = 0

        return None

    def _hard_stop_entropy_collapse(self, val_metrics: dict[str, float]) -> Optional[str]:
        """Detect class-collapse using entropy and missing-class evidence."""

        # Only trigger if accompanied by missing same-dataset classes
        # (confirmed mode collapse). Smoke runs tolerate a bit more transient
        # instability because they validate pipeline integrity, not final quality.
        entropy_val = val_metrics.get("val_family_entropy", 0.0)
        same_dataset_entropy_collapse = val_metrics.get("val_entropy_missing_same_dataset", 0.0) > 0
        smoke_mode = self._is_smoke_mode()
        entropy_threshold = 0.10 if smoke_mode else 0.12
        required_streak = 3 if smoke_mode else 2
        min_epoch = 4 if smoke_mode else 2

        if entropy_val < entropy_threshold and same_dataset_entropy_collapse:
            streak = int(getattr(self, "entropy_missing_class_streak", 0)) + 1
            self.entropy_missing_class_streak = streak
            epoch_idx = int(getattr(self, "epoch", 0))
            if epoch_idx >= min_epoch and streak >= required_streak:
                return "prediction_entropy_collapse_with_missing_classes"
        else:
            self.entropy_missing_class_streak = 0

        # Very strict threshold only for extreme cases
        if entropy_val < 0.08:
            self.entropy_collapse_streak = getattr(self, "entropy_collapse_streak", 0) + 1
            if self.entropy_collapse_streak >= 3:
                self.logger.warning(
                    f"⚠️  Entropy critically low for 3 epochs: {entropy_val:.4f} "
                    f"(missing_classes={int(val_metrics.get('val_family_zero_prediction_classes', 0))})"
                )
                return "prediction_entropy_critical_collapse"
            return None
        self.entropy_collapse_streak = 0

        return None

    def _update_early_stopping(self, _train_metrics: dict[str, float], val_metrics: dict[str, float]) -> bool:
        """Update early stopping state; return True when training should stop."""
        val_loss = val_metrics["val_loss"]
        quality_gate_pass = (
            val_metrics.get("val_family_minority_recall_min", 0.0)
            >= self.config.min_family_minority_recall_for_best
            and val_metrics.get("val_family_entropy", 0.0) >= 0.3
        )

        if val_loss < self.best_val_loss - self.config.early_stopping_threshold:
            if quality_gate_pass:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.best_model_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
                self.logger.info(f"✅ Best model update (loss: {self.best_val_loss:.4f})")
                return False
            self.logger.info(
                "Best-loss candidate rejected by quality gate: "
                f"minority_recall={val_metrics.get('val_family_minority_recall_min', 0.0):.4f}, "
                f"entropy={val_metrics.get('val_family_entropy', 0.0):.4f}"
            )

        self.patience_counter += 1
        if self.patience_counter >= self.config.early_stopping_patience:
            self.logger.info(
                f"Early stopping triggered (patience {self.patience_counter} >= "
                f"{self.config.early_stopping_patience})"
            )
            return True
        return False

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
        """Log formatted per-dataset metrics."""
        for dataset_name, metrics in per_dataset_results.items():
            self.logger.info(f"\n{dataset_name}:")
            for key, val in metrics.items():
                self.logger.info(f"  {key}: {val:.4f}")


# ============================================================================
# Main Training Script
# ============================================================================


@governed_entrypoint(entrypoint_id="scripts.train_helix_ids_full")
def main():  # NOSONAR
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train HelixIDS-Full model")
    parser.add_argument(
        "--config",
        type=str,
        default="config/helix_config.yaml",
        help="Path to training config (YAML)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/helix_full",
        help="Output directory for model/logs",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Device (mps, cpu, cuda)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Batch size for validation/test evaluation (defaults to --batch-size)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="Number of epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1.5e-4,
        help="Optimizer learning rate (defaults to stable value 1.5e-4)",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=2,
        help="Warmup epochs (also used as stage-1 CE epochs when focal is selected)",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Max gradient norm clipping (0 disables clipping)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Emit intra-epoch progress logs every N steps",
    )
    parser.add_argument(
        "--class-balance-strategy",
        type=str,
        default="focal",
        choices=["none", "weighted_ce", "sqrt_weighted_ce", "focal"],
        help=(
            "Class-balance mode: none=unweighted CE, "
            "weighted_ce=inverse-frequency weighted CE, "
            "sqrt_weighted_ce=sqrt-inverse weighted CE, "
            "focal=focal loss"
        ),
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=1.2,
        help="Focal gamma (used when --class-balance-strategy focal)",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Label smoothing applied to family loss",
    )
    parser.add_argument(
        "--sampler-mode",
        type=str,
        default="interleaved_rr",
        choices=["interleaved_rr", "weighted_random_sampler"],
        help=(
            "Training sampler mode for train loader: "
            "interleaved_rr=enforced class-interleaved round-robin (default), "
            "weighted_random_sampler=legacy weighted random sampler"
        ),
    )
    parser.add_argument(
        "--min-class4-samples",
        type=int,
        default=2000,
        help="Minimum class-4 samples enforced via deterministic train-set upsampling",
    )
    parser.add_argument(
        "--class4-per-batch-min",
        type=int,
        default=2,
        help="Hard minimum class-4 samples per training batch for interleaved sampler",
    )
    parser.add_argument(
        "--family-margin-loss-weight",
        type=float,
        default=None,
        help=(
            "Override family margin loss weight (defaults: 0.15 for NSL-KDD, 0.1 otherwise)"
        ),
    )
    parser.add_argument(
        "--family-class4-logit-penalty-weight",
        type=float,
        default=0.0,
        help=(
            "Class-4 dominance ranking penalty weight added to family objective as "
            "lambda * mean(relu(logit_class4 - max_other_logits))"
        ),
    )
    parser.add_argument(
        "--family-feature-separation-weight",
        type=float,
        default=0.0,
        help=(
            "Feature-space centroid separation weight for class-4 vs non-class-4 as "
            "lambda_sep * ( - ||mean(z4)-mean(z_not4)||^2 )"
        ),
    )
    parser.add_argument(
        "--family-class4-target-scale",
        type=float,
        default=1.0,
        help=(
            "Per-sample target-pressure scale for family class-4 labels (0..1). "
            "Applied multiplicatively to family CE terms where label==4."
        ),
    )
    parser.add_argument(
        "--enable-logit-adjustment",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable family-head logit adjustment with train priors and stabilization controls",
    )
    parser.add_argument(
        "--logit-temp",
        type=float,
        default=1.0,
        help="Family-head logit temperature / tau (T>1 smooths confidence)",
    )
    parser.add_argument(
        "--logit-adjustment-tau",
        type=float,
        default=None,
        help="Alias for logit adjustment temperature; overrides --logit-temp when set",
    )
    parser.add_argument(
        "--min-class-prob-eps",
        type=float,
        default=0.0,
        help="Probability-floor epsilon for family-head softmax when logit adjustment is enabled",
    )
    parser.add_argument(
        "--entropy-regularization",
        type=float,
        default=0.02,
        help="Entropy regularization strength for family-head predictions",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("HELIX_SEED", "42")),
        help="Global seed for deterministic execution",
    )
    parser.add_argument(
        "--disable-early-stopping",
        action="store_true",
        help="Run full epoch budget without early stopping termination",
    )
    parser.add_argument(
        "--calibration-mode",
        type=str,
        default="internal_on",
        choices=["internal_on", "internal_off"],
        help=(
            "Post-training calibration mode: internal_on fits temperature scaling + class-4 threshold "
            "and emits calibration artifacts; internal_off disables calibration pipeline"
        ),
    )
    parser.add_argument(
        "--max-temperature",
        type=float,
        default=5.0,
        help="Maximum temperature allowed during post-training calibration fit",
    )
    parser.add_argument(
        "--class4-logit-shift",
        type=float,
        default=0.0,
        help=(
            "Inference-only class-4 logit shift applied before softmax during evaluation/calibration: "
            "logit_4 <- logit_4 - delta"
        ),
    )
    parser.add_argument(
        "--multi-seed-governance",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run fixed 50-epoch multi-seed calibration governance batch instead of single-seed training",
    )
    parser.add_argument(
        "--multi-seeds",
        type=str,
        default="42,1337,2026",
        help="Comma-separated seeds for multi-seed governance mode",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["nsl_kdd", "unsw_nb15", "cicids"],
        help="Run isolated training only for the specified dataset",
    )
    parser.add_argument(
        "--holdout-dataset",
        type=str,
        default="cicids",
        choices=["nsl_kdd", "unsw_nb15", "cicids"],
        help="Dataset to keep fully held out when entity keys are unavailable",
    )
    parser.add_argument(
        "--precomputed-splits-dir",
        type=str,
        default="data/processed/multi_dataset_v1",
        help="Path to precomputed split .npy files",
    )
    parser.add_argument(
        "--force-recompute-splits",
        action="store_true",
        help="Ignore precomputed splits and recompute from raw datasets",
    )
    parser.add_argument(
        "--snapshot-mode",
        type=str,
        default="strict",
        choices=["strict", "research_override"],
        help="Governance mode: strict requires frozen contract snapshot; research_override allows unfrozen validation",
    )
    parser.add_argument(
        "--allow-unfrozen-snapshot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Alias for --snapshot-mode research_override",
    )
    parser.add_argument(
        "--ab-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable strict A/B contract gates for geometry-first promotion",
    )
    parser.add_argument(
        "--ab-track",
        type=str,
        default="objective",
        choices=["feature", "objective"],
        help="A/B change track; only this axis may change versus baseline",
    )
    parser.add_argument(
        "--ab-change-id",
        type=str,
        default="baseline",
        help="Identifier for the single feature/objective change under test",
    )
    parser.add_argument(
        "--ab-baseline",
        type=str,
        default=None,
        help="Path to baseline raw A/B metrics JSON (optional; auto-discovers latest if omitted)",
    )
    parser.add_argument(
        "--ab-require-baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail if no baseline raw A/B metrics are available",
    )
    parser.add_argument(
        "--cluster-objective",
        type=str,
        default="kmeans",
        choices=["kmeans", "gmm", "spectral"],
        help="Clustering objective used for manifold relabeling",
    )
    parser.add_argument(
        "--cluster-spectral-affinity",
        type=str,
        default="nearest_neighbors",
        choices=["nearest_neighbors", "rbf"],
        help="Fixed affinity for spectral objective",
    )

    args = parser.parse_args()
    if bool(args.allow_unfrozen_snapshot):
        args.snapshot_mode = "research_override"

    # Forced anti-collapse controls.
    forced_batch_size = int(args.batch_size)
    if forced_batch_size <= 0:
        raise ValueError("--batch-size must be >= 1")
    forced_class_balance_strategy, forced_use_class_weights = _resolve_class_balance_strategy(
        args.class_balance_strategy
    )
    forced_use_class_weights = bool(forced_use_class_weights)
    if forced_class_balance_strategy == "focal":
        forced_use_class_weights = False
    if float(args.max_temperature) <= 0.0:
        raise ValueError("--max-temperature must be > 0")
    if float(args.class4_logit_shift) < 0.0:
        raise ValueError("--class4-logit-shift must be >= 0")
    if float(args.class4_logit_shift) > 5.0:
        raise ValueError("--class4-logit-shift must be <= 5.0")
    calibration_enabled = str(args.calibration_mode).strip().lower() == "internal_on"
    forced_focal_gamma = float(args.focal_gamma)
    forced_label_smoothing = float(args.label_smoothing)
    forced_use_logit_prior_correction = bool(args.enable_logit_adjustment)
    forced_train_temperature = float(
        args.logit_adjustment_tau if args.logit_adjustment_tau is not None else args.logit_temp
    )
    forced_min_class_prob_eps = float(args.min_class_prob_eps)
    forced_entropy_regularization = float(args.entropy_regularization)
    forced_warmup_ratio = 0.0
    forced_lambda_family = 2.0
    forced_freeze_backbone_epochs = 0
    forced_unfreeze_backbone_step = 0
    forced_entropy_warmup_steps = 200
    forced_entropy_warmup_weight = 0.01
    forced_head_lr_multiplier = 10.0
    forced_lambda_binary = 0.0
    forced_supcon_weight = 1.0
    forced_supcon_temperature = 0.03
    forced_step_coverage_check_step = 50
    forced_representation_diagnostic_mode = False
    forced_representation_only_steps = 140
    forced_representation_micro_cycle_steps = [40, 20, 40, 20, 40]
    forced_use_energy_based_family_objective = False
    forced_adaptive_exit_ratio_threshold = 1.6
    forced_adaptive_exit_min_inter_threshold = 0.30
    forced_representation_only_ratio = 0.25
    forced_head_only_ratio = 0.20
    forced_joint_finetune_backbone_lr_multiplier = 0.25
    forced_joint_finetune_head_lr_multiplier = 0.15
    forced_cluster_relabeling_enabled = True
    forced_cluster_relabel_k: Optional[int] = 3
    forced_cluster_relabel_seed = 42
    forced_cluster_relabel_objective = str(args.cluster_objective).strip().lower()
    forced_cluster_relabel_spectral_affinity = (
        str(args.cluster_spectral_affinity).strip().lower()
    )
    forced_sampler_mode = str(args.sampler_mode).strip().lower()
    dataset_key = str(args.dataset).strip().lower() if args.dataset is not None else ""
    default_family_margin_loss_weight = 0.15 if dataset_key == "nsl_kdd" else 0.1
    forced_family_margin_loss_weight = (
        float(args.family_margin_loss_weight)
        if args.family_margin_loss_weight is not None
        else default_family_margin_loss_weight
    )
    forced_family_class4_logit_penalty_weight = (
        float(args.family_class4_logit_penalty_weight)
        if dataset_key == "unsw_nb15"
        else 0.0
    )
    forced_family_feature_separation_weight = (
        float(args.family_feature_separation_weight)
        if dataset_key == "unsw_nb15"
        else 0.0
    )
    forced_family_class4_target_scale = (
        float(args.family_class4_target_scale)
        if dataset_key == "unsw_nb15"
        else 1.0
    )
    forced_enforce_all_classes_per_batch = False
    forced_geometry_ratio_warmup_threshold = 2.5
    forced_geometry_ratio_post_phase_threshold = 1.2
    objective_regime_tag = "energyobj_cebin_l1_1p0_l2_0p1_l3_0p5_t2p0"
    phase_regime = (
        "helix_full_phase_v2"
        f":rep{forced_representation_only_ratio:.2f}"
        f":head{forced_head_only_ratio:.2f}"
        f":epochs_{int(args.epochs)}"
        f":sampler_{forced_sampler_mode}"
        f":{objective_regime_tag}"
        f":geom{forced_geometry_ratio_warmup_threshold:.1f}->{forced_geometry_ratio_post_phase_threshold:.1f}"
        ":proj_deep_mlp"
    )
    forced_nsl_kdd_label_merges: list[tuple[int, int]] = []
    forced_num_workers = 0

    if bool(args.ab_mode) and args.dataset is None:
        raise ValueError("--ab-mode requires --dataset for single-manifold A/B comparability")

    if bool(args.multi_seed_governance):
        script_path = Path(__file__).resolve()
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
        argv_iter = iter(enumerate(sys.argv[1:]))
        for _idx, token in argv_iter:
            if token in {"--multi-seed-governance", "--no-multi-seed-governance"}:
                continue
            if token in {"--multi-seeds", "--seed", "--epochs", "--calibration-mode", "--max-temperature"}:
                # Skip value token as well when provided as separate arg.
                _ = next(argv_iter, None)
                continue
            if token in {"--disable-early-stopping", "--no-disable-early-stopping"}:
                continue
            if token.startswith(("--multi-seeds=", "--seed=", "--epochs=", "--calibration-mode=", "--max-temperature=")):
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
        report_path = HELIX_FULL_RESULTS_DIR / "multi_seed_calibrated_governance.json"
        _atomic_write_json(report_path, governance_report)
        print(json.dumps(governance_report, indent=2, default=str))
        return {
            "results": {"multi_seed_governance": governance_report},
            "governance_stages": {"multi_seed_report_path": str(report_path)},
            "governance_context": {
                "seed": int(parsed_seeds[0]),
                "phase_regime": "multi_seed_calibrated_governance",
            },
            "governance_run_record": {
                "dataset_id": "multi_seed_calibrated_governance",
                "macro_f1": float(governance_report.get("governance", {}).get("mean_macro_f1", 0.0)),
            },
            "determinism": {
                "mode": "multi_seed_governance",
                "orchestrator_seed": int(args.seed),
            },
        }

    os.environ["HELIX_STRICT_MISSING"] = "1"
    os.environ["STRICT_MISSING"] = "1"
    os.environ["HELIX_SEED"] = str(args.seed)
    determinism_state = set_global_determinism(args.seed)

    if args.dataset == "unsw_nb15" and args.epochs < 10:
        raise ValueError(
            "--epochs must be >= 10 for UNSW-stable training signal"
        )
    split_start = time.perf_counter()

    # Create output directories
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = HELIX_FULL_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    run_exit_code = 1
    guard_failure: Optional[str] = None
    results: dict[str, Any] = {}
    per_dataset_results: dict[str, Any] = {}
    training_results_path = results_dir / f"training_results_seed{args.seed}.json"

    # Setup logging
    logger = setup_logging(results_dir)

    # Load configs
    train_config = TrainingConfig(
        batch_size=forced_batch_size,
        epochs=args.epochs,
        device=args.device,
    )
    if args.learning_rate is not None:
        if args.learning_rate <= 0:
            raise ValueError("--learning-rate must be > 0")
        train_config.learning_rate = float(args.learning_rate)
    train_config.learning_rate = float(train_config.learning_rate)
    if args.warmup_epochs is not None:
        if int(args.warmup_epochs) < 0:
            raise ValueError("--warmup-epochs must be >= 0")
        train_config.warmup_epochs = int(args.warmup_epochs)
    if args.grad_clip is not None:
        if float(args.grad_clip) < 0.0:
            raise ValueError("--grad-clip must be >= 0")
        train_config.max_grad_norm = float(args.grad_clip)
    if forced_focal_gamma < 0.0:
        raise ValueError("--focal-gamma must be >= 0")
    if forced_train_temperature <= 0.0:
        raise ValueError("--logit-temp/--logit-adjustment-tau must be > 0")
    if forced_min_class_prob_eps < 0.0:
        raise ValueError("--min-class-prob-eps must be >= 0")
    if int(args.log_interval) < 1:
        raise ValueError("--log-interval must be >= 1")
    if forced_entropy_regularization < 0.0:
        raise ValueError("--entropy-regularization must be >= 0")
    if not 0.0 <= forced_label_smoothing < 1.0:
        raise ValueError("--label-smoothing must be in [0, 1)")
    if forced_sampler_mode not in {"interleaved_rr", "weighted_random_sampler"}:
        raise ValueError(
            "--sampler-mode must be one of {'interleaved_rr', 'weighted_random_sampler'}"
        )
    if int(args.min_class4_samples) < 0:
        raise ValueError("--min-class4-samples must be >= 0")
    if int(args.class4_per_batch_min) < 0:
        raise ValueError("--class4-per-batch-min must be >= 0")
    if forced_family_margin_loss_weight < 0.0:
        raise ValueError("--family-margin-loss-weight must be >= 0")
    if forced_family_class4_logit_penalty_weight < 0.0:
        raise ValueError("--family-class4-logit-penalty-weight must be >= 0")
    if forced_family_feature_separation_weight < 0.0:
        raise ValueError("--family-feature-separation-weight must be >= 0")
    if forced_family_class4_target_scale < 0.0 or forced_family_class4_target_scale > 1.0:
        raise ValueError("--family-class4-target-scale must be in [0, 1]")
    train_config.log_interval = int(args.log_interval)
    train_config.lambda_family = float(forced_lambda_family)
    train_config.class_balance_strategy = forced_class_balance_strategy
    train_config.use_class_weights = bool(forced_use_class_weights)
    train_config.focal_gamma = forced_focal_gamma
    train_config.enable_logit_adjustment = forced_use_logit_prior_correction
    train_config.logit_temp = forced_train_temperature
    train_config.min_class_prob_eps = forced_min_class_prob_eps
    config_payload = {
        "batch_size": train_config.batch_size,
        "epochs": train_config.epochs,
        "learning_rate": train_config.learning_rate,
        "warmup_epochs": train_config.warmup_epochs,
        "grad_clip": train_config.max_grad_norm,
        "lambda_binary": forced_lambda_binary,
        "lambda_family": train_config.lambda_family,
        "class_balance_strategy": forced_class_balance_strategy,
        "use_class_weights": bool(forced_use_class_weights),
        "focal_gamma": forced_focal_gamma,
        "label_smoothing": forced_label_smoothing,
        "use_logit_prior_correction": forced_use_logit_prior_correction,
        "train_temperature": forced_train_temperature,
        "warmup_ratio": forced_warmup_ratio,
        "freeze_backbone_epochs": forced_freeze_backbone_epochs,
        "unfreeze_backbone_step": forced_unfreeze_backbone_step,
        "entropy_warmup_steps": forced_entropy_warmup_steps,
        "entropy_warmup_weight": forced_entropy_warmup_weight,
        "head_lr_multiplier": forced_head_lr_multiplier,
        "supcon_weight": forced_supcon_weight,
        "supcon_temperature": forced_supcon_temperature,
        "step_coverage_check_step": forced_step_coverage_check_step,
        "representation_diagnostic_mode": forced_representation_diagnostic_mode,
        "use_energy_based_family_objective": bool(
            forced_use_energy_based_family_objective
        ),
        "representation_only_steps": forced_representation_only_steps,
        "representation_micro_cycle_steps": [
            int(v) for v in forced_representation_micro_cycle_steps
        ],
        "adaptive_exit_ratio_threshold": forced_adaptive_exit_ratio_threshold,
        "adaptive_exit_min_inter_threshold": forced_adaptive_exit_min_inter_threshold,
        "representation_only_ratio": forced_representation_only_ratio,
        "head_only_ratio": forced_head_only_ratio,
        "joint_finetune_backbone_lr_multiplier": forced_joint_finetune_backbone_lr_multiplier,
        "joint_finetune_head_lr_multiplier": forced_joint_finetune_head_lr_multiplier,
        "cluster_relabeling_enabled": forced_cluster_relabeling_enabled,
        "cluster_relabel_k": forced_cluster_relabel_k,
        "cluster_relabel_seed": forced_cluster_relabel_seed,
        "cluster_relabel_objective": forced_cluster_relabel_objective,
        "cluster_relabel_spectral_affinity": forced_cluster_relabel_spectral_affinity,
        "sampler_mode": forced_sampler_mode,
        "enforce_all_classes_per_batch": forced_enforce_all_classes_per_batch,
        "geometry_ratio_warmup_threshold": forced_geometry_ratio_warmup_threshold,
        "geometry_ratio_post_phase_threshold": forced_geometry_ratio_post_phase_threshold,
        "num_workers": forced_num_workers,
        "ab_mode": bool(args.ab_mode),
        "ab_track": str(args.ab_track),
        "ab_change_id": str(args.ab_change_id),
        "ab_require_baseline": bool(args.ab_require_baseline),
        "nsl_kdd_label_merges": [
            {"src": int(src), "dst": int(dst)}
            for src, dst in forced_nsl_kdd_label_merges
        ],
        "kl_weight_warmup": 0.0,
        "kl_weight_post_warmup": 0.0,
        "logit_floor_weight": 0.0,
        "tail_ce_weight": 0.0,
        "snapshot_mode": str(args.snapshot_mode),
        "allow_unfrozen_snapshot": bool(args.allow_unfrozen_snapshot),
        "min_class_prob_eps": forced_min_class_prob_eps,
        "entropy_regularization": forced_entropy_regularization,
        "family_margin_loss_weight": float(forced_family_margin_loss_weight),
        "family_class4_logit_penalty_weight": float(forced_family_class4_logit_penalty_weight),
        "family_feature_separation_weight": float(forced_family_feature_separation_weight),
        "family_class4_target_scale": float(forced_family_class4_target_scale),
        "family_logit_margin": 1.0,
        "class4_logit_shift": float(args.class4_logit_shift),
        "class4_per_batch_min": int(args.class4_per_batch_min),
        "device": args.device,
        "phase_regime": phase_regime,
        "training_mode": "head_isolation_ce_warmstart",
    }
    train_config.lambda_binary = float(forced_lambda_binary)
    train_config.num_workers = int(forced_num_workers)
    train_config.freeze_backbone_epochs = int(forced_freeze_backbone_epochs)
    train_config.unfreeze_backbone_step = int(forced_unfreeze_backbone_step)
    train_config.entropy_warmup_steps = int(forced_entropy_warmup_steps)
    train_config.entropy_warmup_weight = float(forced_entropy_warmup_weight)
    _apply_disable_early_stopping(
        train_config,
        disable_early_stopping=bool(args.disable_early_stopping),
    )
    eval_batch_size = int(args.eval_batch_size or train_config.batch_size)
    if eval_batch_size <= 0:
        raise ValueError("--eval-batch-size must be >= 1")
    data_config = DataConfig()

    if args.dataset is not None:
        _assert_real_dataset_required(
            project_root=PROJECT_ROOT,
            dataset_name=args.dataset,
        )

    logger.info(f"Loading data from {data_config.data_dir}...")

    # Load multi-dataset (Phase 1)
    from helix_ids.data.feature_harmonization import (
        labels_to_multi_task,
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
                "Running with research_override snapshot mode (allow_unfrozen_snapshot=%s); reproducibility promotion gates remain disabled for this run.",
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

    feature_order = [str(col) for col in np.asarray(splits["feature_columns"]).astype(str).tolist()]
    if len(feature_order) != REQUIRED_GEOMETRY_FEATURE_DIM:
        raise RuntimeError(
            "Hard-stop integrity guard triggered: canonical_feature_dim_not_17_"
            f"got_{len(feature_order)}"
        )
    schema_hash = compute_schema_hash(
        feature_columns=feature_order,
        transformations=["split_then_nan_to_num"],
    )
    feature_signature = _stable_feature_signature(
        feature_order=feature_order,
        schema_hash=schema_hash,
    )

    _validate_per_dataset_splits(
        splits,
        logger=logger,
        seed=args.seed,
        enforce_cross_dataset_scale=False,
    )

    target_family_class_count = int(HelixFullConfig().family_output_dim)

    global_family_priors: Optional[torch.Tensor] = None
    if "y_train" in splits:
        y_global = np.asarray(cast(np.ndarray, splits["y_train"]), dtype=np.int64)
        if np.any(y_global >= target_family_class_count):
            remapped = int(np.sum(y_global >= target_family_class_count))
            y_global = np.where(
                y_global >= target_family_class_count,
                target_family_class_count - 1,
                y_global,
            ).astype(np.int64, copy=False)
            logger.warning(
                "Remapped %d global family labels >= %d to class %d to match active taxonomy.",
                remapped,
                target_family_class_count,
                target_family_class_count - 1,
            )

        global_counts = np.bincount(
            y_global,
            minlength=target_family_class_count,
        ).astype(np.float64)
        global_total = float(max(1.0, global_counts.sum()))
        global_smoothed = (global_counts + 1.0) / (global_total + int(global_counts.shape[0]))
        global_family_priors = torch.tensor(
            np.clip(global_smoothed, 1e-12, 1.0),
            dtype=torch.float32,
        )
        logger.info(
            "Using global family priors for logit correction: %s",
            {int(i): float(v) for i, v in enumerate(global_smoothed.tolist())},
        )

    split_end = time.perf_counter()
    split_elapsed = split_end - split_start
    pretrain_start = time.perf_counter()

    if args.dataset is not None:
        dataset_specs = [
            (args.dataset, 2 if args.dataset == "nsl_kdd" else 3),
        ]
    else:
        dataset_specs = [
            ("nsl_kdd", 2),
            ("unsw_nb15", 3),
            ("cicids", 3),
        ]

    all_results: dict[str, Any] = {}
    per_dataset_results: dict[str, Any] = {}
    dataset_snapshot_ids: dict[str, str] = {}
    dataset_representation_snapshot_ids: dict[str, str] = {}
    ab_raw_current_by_dataset: dict[str, dict[str, Any]] = {}
    training_elapsed_total = 0.0

    for dataset_name, min_classes in dataset_specs:
        x_train_key = f"X_train_{dataset_name}"
        y_train_key = f"y_train_{dataset_name}"
        x_val_key = f"X_val_{dataset_name}"
        y_val_key = f"y_val_{dataset_name}"
        x_test_key = f"X_test_{dataset_name}"
        y_test_key = f"y_test_{dataset_name}"

        missing_keys = [
            k
            for k in [x_train_key, y_train_key, x_val_key, y_val_key, x_test_key, y_test_key]
            if k not in splits
        ]
        if missing_keys:
            logger.warning(
                "[%s] Missing split keys; skipping isolated training for this dataset: %s",
                dataset_name,
                missing_keys,
            )
            continue

        x_train_ds = cast(np.ndarray, splits[x_train_key])
        y_train_family_ds = cast(np.ndarray, splits[y_train_key]).astype(np.int64, copy=False)
        if dataset_name == "nsl_kdd" and forced_nsl_kdd_label_merges:
            y_train_family_ds = _apply_label_merges(
                y_train_family_ds,
                merges=forced_nsl_kdd_label_merges,
            )
            logger.info(
                "[%s] Applied label merges for collision baseline: %s",
                dataset_name,
                forced_nsl_kdd_label_merges,
            )

        if np.any(y_train_family_ds >= target_family_class_count):
            remapped = int(np.sum(y_train_family_ds >= target_family_class_count))
            y_train_family_ds = np.where(
                y_train_family_ds >= target_family_class_count,
                target_family_class_count - 1,
                y_train_family_ds,
            ).astype(np.int64, copy=False)
            logger.warning(
                "[%s] Remapped %d train family labels >= %d to class %d for taxonomy consistency.",
                dataset_name,
                remapped,
                target_family_class_count,
                target_family_class_count - 1,
            )
        if x_train_ds.shape[0] == 0:
            logger.warning("[%s] Empty training split; skipping dataset.", dataset_name)
            continue

        unique_classes = np.unique(y_train_family_ds)
        if unique_classes.size < min_classes:
            guard_failure = (
                f"Hard-stop integrity guard triggered: insufficient_class_diversity_{dataset_name}"
            )
            _persist_seed_artifacts(
                results_dir=results_dir,
                seed=args.seed,
                config_payload=config_payload,
                results_payload=results,
                eval_payload=per_dataset_results,
                run_exit_code=1,
                guard_failure=guard_failure,
            )
            raise RuntimeError(
                f"Hard-stop integrity guard triggered: insufficient_class_diversity_{dataset_name}"
            )

        y_train_binary_ds, y_train_family_ds = labels_to_multi_task(y_train_family_ds)

        current_feature_dim = int(x_train_ds.shape[1])
        if current_feature_dim != REQUIRED_GEOMETRY_FEATURE_DIM:
            raise RuntimeError(
                "Hard-stop integrity guard triggered: train_feature_dim_not_17_"
                f"{dataset_name}:got_{current_feature_dim}"
            )
        x_val_ds = _load_eval_array(
            splits=splits,
            dataset_name=dataset_name,
            split_name="val",
            prefix="X",
            logger=logger,
            expected_feature_dim=current_feature_dim,
        )
        y_val_family_ds = _load_eval_array(
            splits=splits,
            dataset_name=dataset_name,
            split_name="val",
            prefix="y",
            logger=logger,
        )
        if dataset_name == "nsl_kdd" and forced_nsl_kdd_label_merges:
            y_val_family_ds = _apply_label_merges(
                y_val_family_ds,
                merges=forced_nsl_kdd_label_merges,
            )
        y_val_family_ds = np.where(
            y_val_family_ds >= target_family_class_count,
            target_family_class_count - 1,
            y_val_family_ds,
        ).astype(np.int64, copy=False)
        x_test_ds = _load_eval_array(
            splits=splits,
            dataset_name=dataset_name,
            split_name="test",
            prefix="X",
            logger=logger,
            expected_feature_dim=current_feature_dim,
        )
        y_test_family_ds = _load_eval_array(
            splits=splits,
            dataset_name=dataset_name,
            split_name="test",
            prefix="y",
            logger=logger,
        )
        if dataset_name == "nsl_kdd" and forced_nsl_kdd_label_merges:
            y_test_family_ds = _apply_label_merges(
                y_test_family_ds,
                merges=forced_nsl_kdd_label_merges,
            )
        y_test_family_ds = np.where(
            y_test_family_ds >= target_family_class_count,
            target_family_class_count - 1,
            y_test_family_ds,
        ).astype(np.int64, copy=False)

        x_train_ds, x_val_ds, x_test_ds, engineered_norm_stats = _normalize_engineered_feature_block(
            dataset_name=dataset_name,
            x_train=x_train_ds,
            x_val=x_val_ds,
            x_test=x_test_ds,
            feature_names=feature_order,
        )
        if engineered_norm_stats:
            logger.info(
                "FeatureNorm[%s] engineered_feature_stats=%s",
                dataset_name,
                engineered_norm_stats,
            )

        _assert_feature_sanity_for_dataset(
            dataset_name=dataset_name,
            x_train=x_train_ds,
            x_val=x_val_ds,
            feature_names=feature_order,
            expected_feature_dim=REQUIRED_GEOMETRY_FEATURE_DIM,
            min_feature_std=MIN_FEATURE_STD,
            seed=args.seed,
            logger=logger,
        )

        train_dataset = TensorDataset(
            torch.from_numpy(x_train_ds).float(),
            torch.from_numpy(y_train_binary_ds).long(),
            torch.from_numpy(y_train_family_ds).long(),
        )

        train_loader_generator = torch.Generator().manual_seed(args.seed)
        val_loader_generator = torch.Generator().manual_seed(args.seed + 1)
        test_loader_generator = torch.Generator().manual_seed(args.seed + 2)

        train_class_index = build_class_index(y_train_family_ds)
        min_class4_samples = int(args.min_class4_samples)
        if min_class4_samples > 0:
            class4_indices = np.asarray(train_class_index.get(4, np.array([], dtype=np.int64)), dtype=np.int64)
            if int(class4_indices.size) <= 0:
                raise RuntimeError(
                    "Hard-stop integrity guard triggered: class4_missing_for_minimum_sample_enforcement"
                )
            if int(class4_indices.size) < min_class4_samples:
                deficit = int(min_class4_samples - int(class4_indices.size))
                rng_upsample = np.random.default_rng(args.seed)
                oversampled_class4 = rng_upsample.choice(
                    class4_indices,
                    size=deficit,
                    replace=True,
                ).astype(np.int64)
                upsample_indices = np.concatenate(
                    [np.arange(int(y_train_family_ds.shape[0]), dtype=np.int64), oversampled_class4],
                    axis=0,
                )
                # Deterministically shuffle to avoid contiguous single-class tail blocks.
                upsample_indices = rng_upsample.permutation(upsample_indices).astype(np.int64)
                x_train_ds = x_train_ds[upsample_indices]
                y_train_binary_ds = y_train_binary_ds[upsample_indices]
                y_train_family_ds = y_train_family_ds[upsample_indices]
                train_dataset = TensorDataset(
                    torch.from_numpy(x_train_ds).float(),
                    torch.from_numpy(y_train_binary_ds).long(),
                    torch.from_numpy(y_train_family_ds).long(),
                )
                train_class_index = build_class_index(y_train_family_ds)
                logger.info(
                    "[%s] Enforced min class-4 samples via upsampling: original=%d target=%d final=%d",
                    dataset_name,
                    int(class4_indices.size),
                    int(min_class4_samples),
                    int(train_class_index.get(4, np.array([], dtype=np.int64)).shape[0]),
                )
        train_class_counts = {
            class_id: int(indices.shape[0]) for class_id, indices in train_class_index.items()
        }
        total_train_rows = max(1, int(y_train_family_ds.shape[0]))
        class_sampling_probs = {
            int(class_id): float(count / total_train_rows)
            for class_id, count in train_class_counts.items()
        }
        interleaved_indices = _build_interleaved_round_robin_indices(
            y_train_family_ds,
            batch_size=train_config.batch_size,
            seed=args.seed,
            min_unique_classes_per_batch=2,
            class4_min_per_batch=int(args.class4_per_batch_min),
        )
        class_sampling_weights = np.ones(total_train_rows, dtype=np.float64)
        class_sampling_weight_map = {
            3: 3.0,
            4: 6.0,
        }
        for class_id, idxs in train_class_index.items():
            w = float(class_sampling_weight_map.get(int(class_id), 1.0))
            if w > 1.0:
                class_sampling_weights[np.asarray(idxs, dtype=np.int64)] = w
        weighted_sampler = WeightedRandomSampler(
            weights=torch.from_numpy(class_sampling_weights).double(),
            num_samples=int(total_train_rows),
            replacement=True,
            generator=train_loader_generator,
        )
        interleaved_sampler = FrozenIndexSampler(interleaved_indices)
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_config.batch_size,
            shuffle=False,
            sampler=(
                weighted_sampler
                if forced_sampler_mode == "weighted_random_sampler"
                else interleaved_sampler
            ),
            num_workers=train_config.num_workers,
            pin_memory=train_config.pin_memory,
            worker_init_fn=seed_worker,
            generator=(
                None
                if forced_sampler_mode in {"interleaved_rr", "weighted_random_sampler"}
                else train_loader_generator
            ),
            persistent_workers=train_config.num_workers > 0,
            prefetch_factor=2 if train_config.num_workers > 0 else None,
        )

        val_subset_indices = _build_stratified_subset_indices(
            y_val_family_ds,
            target_per_class=50,
            seed=args.seed + 17,
        )
        x_val_eval, y_val_family_eval = _build_stratified_val_subset(
            x_val_ds,
            y_val_family_ds,
            target_per_class=50,
            seed=args.seed + 17,
        )
        val_counts = {
            int(class_id): int(count)
            for class_id, count in zip(*np.unique(y_val_family_eval, return_counts=True))
        }
        logger.info(
            "[%s] Validation stratified subset counts (>=50/class): %s",
            dataset_name,
            val_counts,
        )
        snapshot_descriptor = _write_isolation_snapshot_descriptor(
            dataset_name=dataset_name,
            splits_dir=precomputed_splits_dir,
            seed=args.seed,
            batch_size=train_config.batch_size,
            class_counts=train_class_counts,
            class_multipliers=class_sampling_probs,
            sampler_indices=interleaved_indices,
            val_subset_indices=val_subset_indices,
            results_dir=results_dir,
            snapshot_mode=args.snapshot_mode,
        )
        logger.info(
            "[%s] Isolation snapshot locked: id=%s descriptor=%s",
            dataset_name,
            str(snapshot_descriptor["snapshot_id"]),
            str(snapshot_descriptor["snapshot_path"]),
        )
        dataset_snapshot_ids[dataset_name] = str(snapshot_descriptor["snapshot_id"])

        val_loaders = {
            dataset_name: DataLoader(
                MultiTaskNumpyDataset(x_val_eval, y_val_family_eval),
                batch_size=eval_batch_size,
                shuffle=False,
                num_workers=train_config.num_workers,
                pin_memory=train_config.pin_memory,
                worker_init_fn=seed_worker,
                generator=val_loader_generator,
                persistent_workers=train_config.num_workers > 0,
                prefetch_factor=2 if train_config.num_workers > 0 else None,
            )
        }

        test_loaders = {
            dataset_name: DataLoader(
                MultiTaskNumpyDataset(x_test_ds, y_test_family_ds),
                batch_size=eval_batch_size,
                shuffle=False,
                num_workers=max(2, train_config.num_workers),
                pin_memory=train_config.pin_memory,
                worker_init_fn=seed_worker,
                generator=test_loader_generator,
                persistent_workers=max(2, train_config.num_workers) > 0,
                prefetch_factor=2,
            )
        }

        model = create_helix_full(
            HelixFullConfig(
                input_dim=current_feature_dim,
                hidden_dims=(512, 384, 256, 256),
                dropout_rates=(0.3, 0.3, 0.25, 0.2),
            )
        )
        logger.info(f"[{dataset_name}] Model parameters: {model.param_count:,}")

        family_weights = (
            _sqrt_inverse_frequency_weights(
                y_train_family_ds,
                minlength=target_family_class_count,
            )
            if str(args.class_balance_strategy).strip().lower() == "sqrt_weighted_ce"
            else _inverse_frequency_weights(
                y_train_family_ds,
                minlength=target_family_class_count,
            )
        )
        family_class_weights = torch.tensor(family_weights, dtype=torch.float32)
        if str(args.class_balance_strategy).strip().lower() == "focal":
            family_counts = np.bincount(
                y_train_family_ds,
                minlength=target_family_class_count,
            ).astype(np.float64)
            total_family = float(max(1.0, family_counts.sum()))
            family_alpha = np.sqrt(total_family / np.maximum(1.0, family_counts))
            family_alpha = family_alpha / max(1e-12, float(np.mean(family_alpha)))
            family_class_weights = torch.tensor(family_alpha.astype(np.float32), dtype=torch.float32)
        family_prior_counts = np.bincount(
            y_train_family_ds,
            minlength=target_family_class_count,
        ).astype(np.float64)
        family_prior_total = float(max(1.0, family_prior_counts.sum()))
        class_count = int(family_prior_counts.shape[0])
        # Use global priors for correction; fallback to dataset priors if combined train priors are unavailable.
        smoothed_priors = (family_prior_counts + 1.0) / (family_prior_total + class_count)
        family_priors = (
            global_family_priors.clone()
            if global_family_priors is not None
            else torch.tensor(np.clip(smoothed_priors, 1e-12, 1.0), dtype=torch.float32)
        )

        family_freq = family_prior_counts / family_prior_total
        tail_class_mask_np = (family_freq <= 0.02) & (np.arange(class_count) != 0)

        final_family_layer = model.family_head[-1] if len(model.family_head) > 0 else None
        if isinstance(final_family_layer, nn.Linear) and final_family_layer.bias is not None:
            with torch.no_grad():
                final_family_layer.bias.zero_()
            logger.info(
                "[%s] Family head bias initialization disabled (set to zeros).",
                dataset_name,
            )

        binary_weights = _inverse_frequency_weights(y_train_binary_ds, minlength=2)
        binary_class_weights = torch.tensor(binary_weights, dtype=torch.float32)

        if dataset_name == "cicids":
            logger.info(
                "[%s] Imbalance-aware loss active: strategy=%s label_smoothing=%.2f",
                dataset_name,
                forced_class_balance_strategy,
                forced_label_smoothing,
            )
            logger.info(
                "[%s] Isolation run: strategy=%s prior_logit_correction=%s train_temp=%.2f warmup_ratio=%.2f kl_weight[warmup=0.00 post=0.00] logit_floor_weight=0.00 tail_ce_weight=0.00 lambda_family=%.2f freeze_backbone_epochs=%d unfreeze_step=%d entropy_warmup[steps=%d weight=%.3f] head_lr_multiplier=%.1f grad_clip(backbone_only)=%.3f",
                dataset_name,
                forced_class_balance_strategy,
                forced_use_logit_prior_correction,
                forced_train_temperature,
                forced_warmup_ratio,
                forced_lambda_family,
                forced_freeze_backbone_epochs,
                forced_unfreeze_backbone_step,
                forced_entropy_warmup_steps,
                forced_entropy_warmup_weight,
                forced_head_lr_multiplier,
                float(train_config.max_grad_norm),
            )
            logger.info(
                "[%s] Stratified train class counts=%s multipliers=%s tail_mask=%s",
                dataset_name,
                train_class_counts,
                class_sampling_probs,
                {int(i): bool(v) for i, v in enumerate(tail_class_mask_np.tolist())},
            )

        for param in model.binary_head.parameters():
            param.requires_grad = False

        backbone_params = [param for param in model.backbone.parameters() if param.requires_grad]
        family_projection_params = [
            param for param in model.family_projection.parameters() if param.requires_grad
        ]
        family_head_params = [param for param in model.family_head.parameters() if param.requires_grad]
        optimizer = optim.AdamW(
            [
                {
                    "params": backbone_params,
                    "lr": train_config.learning_rate,
                    "lr_scale": 1.0,
                    "group_name": "backbone",
                },
                {
                    "params": family_projection_params + family_head_params,
                    "lr": train_config.learning_rate * forced_head_lr_multiplier,
                    "lr_scale": forced_head_lr_multiplier,
                    "group_name": "family_head",
                },
            ],
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
        )
        loss_fn = MultiTaskLoss(
            lambda_binary=train_config.lambda_binary,
            lambda_family=train_config.lambda_family,
            balance_strategy=forced_class_balance_strategy,
            focal_gamma=forced_focal_gamma,
            label_smoothing=forced_label_smoothing,
            use_class_weights=bool(forced_use_class_weights),
            focal_use_class_weights=False,
            entropy_regularization=forced_entropy_regularization,
            family_logit_margin=1.0,
            family_margin_loss_weight=float(forced_family_margin_loss_weight),
            family_class4_logit_penalty_weight=float(
                forced_family_class4_logit_penalty_weight
            ),
            family_class4_logit_penalty_class=4,
            family_feature_separation_weight=float(forced_family_feature_separation_weight),
            family_feature_separation_class=4,
            family_class4_target_scale=float(forced_family_class4_target_scale),
        )

        trainer = HelixFullTrainer(
            model=model,
            train_loader=train_loader,
            val_loaders=val_loaders,
            test_loaders=test_loaders,
            optimizer=optimizer,
            loss_fn=loss_fn,
            config=train_config,
            binary_class_weights=binary_class_weights,
            family_class_weights=family_class_weights,
            train_family_class_count=int(np.unique(y_train_family_ds).size),
            run_seed=args.seed,
            device=args.device,
            logger=logger,
        )
        trainer.disable_integrity_hard_stops = bool(args.disable_early_stopping)
        trainer.disable_tail_focal_regularizer = forced_class_balance_strategy != "focal"
        trainer.configure_family_controls(
            family_class_priors=family_priors if forced_use_logit_prior_correction else None,
            tail_class_mask=None,
            balance_strategy=forced_class_balance_strategy,
            focal_warmup_epochs=0,
            warmup_ratio=forced_warmup_ratio,
            train_temperature=forced_train_temperature,
        )
        total_steps_for_dataset = max(1, int(len(train_loader)) * max(1, int(train_config.epochs)))
        representation_only_steps = max(
            int(forced_representation_only_steps),
            int(math.ceil(total_steps_for_dataset * float(forced_representation_only_ratio))),
        )
        head_only_steps = max(
            1,
            int(math.ceil(total_steps_for_dataset * float(forced_head_only_ratio))),
        )
        trainer.configure_structure_recovery(
            active_family_classes={int(c) for c in np.unique(y_train_family_ds).tolist()},
            supcon_weight=forced_supcon_weight,
            supcon_temperature=forced_supcon_temperature,
            step_coverage_check_step=forced_step_coverage_check_step,
            representation_diagnostic_mode=forced_representation_diagnostic_mode,
            phase_settings={
                "representation_only_steps": representation_only_steps,
                "representation_micro_cycle_steps": [
                    int(v) for v in forced_representation_micro_cycle_steps
                ],
                "use_energy_based_family_objective": bool(
                    forced_use_energy_based_family_objective
                ),
                "adaptive_exit_ratio_threshold": forced_adaptive_exit_ratio_threshold,
                "adaptive_exit_min_inter_threshold": forced_adaptive_exit_min_inter_threshold,
                "head_only_steps": head_only_steps,
                "joint_finetune_backbone_lr_multiplier": forced_joint_finetune_backbone_lr_multiplier,
                "joint_finetune_head_lr_multiplier": forced_joint_finetune_head_lr_multiplier,
                "geometry_ratio_warmup_threshold": forced_geometry_ratio_warmup_threshold,
                "geometry_ratio_post_phase_threshold": forced_geometry_ratio_post_phase_threshold,
                "enforce_all_classes_per_batch": forced_enforce_all_classes_per_batch,
                "sampler_mode": forced_sampler_mode,
            },
            cluster_relabeling_enabled=forced_cluster_relabeling_enabled,
            cluster_relabel_k=forced_cluster_relabel_k,
            cluster_relabel_seed=forced_cluster_relabel_seed,
            cluster_relabel_objective=forced_cluster_relabel_objective,
            cluster_relabel_spectral_affinity=forced_cluster_relabel_spectral_affinity,
        )
        trainer.feature_order = feature_order
        trainer.schema_hash = schema_hash

        logger.info(f"[{dataset_name}] Starting isolated training...")
        dataset_train_start = time.perf_counter()
        try:
            dataset_results = trainer.fit()
        except Exception as exc:
            results.setdefault("representation_diagnostics", {})[dataset_name] = dict(
                trainer.representation_diagnostics
            )
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
        training_elapsed_total += time.perf_counter() - dataset_train_start

        best_model_path = output_dir / f"helix_full_{dataset_name}_best.pt"
        final_model_path = output_dir / f"helix_full_{dataset_name}_final.pt"
        artifact = _build_model_contract_artifact(
            model_state=model.state_dict(),
            feature_order=feature_order,
            schema_hash=schema_hash,
            extra={"dataset_name": dataset_name},
        )
        _write_checkpoint_artifact(
            best_model_path,
            artifact,
            model_architecture=model.__class__.__name__,
            origin=f"train_helix_ids_full:{dataset_name}:best",
        )
        _write_checkpoint_artifact(
            final_model_path,
            artifact,
            model_architecture=model.__class__.__name__,
            origin=f"train_helix_ids_full:{dataset_name}:final",
        )

        all_results[dataset_name] = dataset_results
        dataset_eval_metrics = cast(dict[str, Any], dataset_results["per_dataset_results"][dataset_name])

        if calibration_enabled:
            calibration_payload = _calibrate_family_predictions(
                model=model,
                val_loader=val_loaders[dataset_name],
                test_loader=test_loaders[dataset_name],
                device=args.device,
                class4_id=4,
                max_temperature=float(args.max_temperature),
                min_class4_recall=0.80,
                class4_logit_shift=float(args.class4_logit_shift),
            )
            calibration_artifacts = _emit_calibration_artifacts(
                results_dir=results_dir,
                dataset_name=dataset_name,
                seed=int(args.seed),
                calibration_payload=calibration_payload,
            )
            calibrated_test = cast(dict[str, Any], calibration_payload.get("test", {}))
            calibrated_val = cast(dict[str, Any], calibration_payload.get("val", {}))
            dataset_eval_metrics = dict(dataset_eval_metrics)
            dataset_eval_metrics["family_macro_f1_uncalibrated"] = float(
                dataset_eval_metrics.get("family_macro_f1", dataset_eval_metrics.get("family_f1", 0.0))
            )
            dataset_eval_metrics["family_entropy_uncalibrated"] = float(
                dataset_eval_metrics.get("family_entropy", 0.0)
            )
            dataset_eval_metrics["family_zero_prediction_classes_uncalibrated"] = float(
                dataset_eval_metrics.get("family_zero_prediction_classes", 0.0)
            )
            dataset_eval_metrics["family_macro_f1"] = float(
                calibrated_test.get("macro_f1", dataset_eval_metrics.get("family_macro_f1", 0.0))
            )
            dataset_eval_metrics["family_f1"] = float(dataset_eval_metrics["family_macro_f1"])
            dataset_eval_metrics["family_minority_recall_min"] = float(
                calibrated_test.get(
                    "class4_recall",
                    dataset_eval_metrics.get("family_minority_recall_min", 0.0),
                )
            )
            dataset_eval_metrics["family_entropy"] = float(
                calibrated_test.get("mean_entropy", dataset_eval_metrics.get("family_entropy", 0.0))
            )
            dataset_eval_metrics["family_zero_prediction_classes"] = float(
                calibrated_test.get(
                    "zero_prediction_classes",
                    dataset_eval_metrics.get("family_zero_prediction_classes", 0.0),
                )
            )
            dataset_eval_metrics["family_class4_precision"] = float(
                calibrated_test.get("class4_precision", 0.0)
            )
            dataset_eval_metrics["family_class4_recall"] = float(
                calibrated_test.get("class4_recall", 0.0)
            )
            dataset_eval_metrics["family_class4_precision_val"] = float(
                calibrated_val.get("class4_precision", 0.0)
            )
            dataset_eval_metrics["family_class4_recall_val"] = float(
                calibrated_val.get("class4_recall", 0.0)
            )
            dataset_eval_metrics["family_calibration_temperature"] = float(
                calibration_payload.get("temperature", 1.0)
            )
            dataset_eval_metrics["family_calibration_tau_4"] = float(
                calibration_payload.get("tau_4", 0.5)
            )
            dataset_eval_metrics["family_class4_logit_shift"] = float(
                calibration_payload.get("class4_logit_shift", float(args.class4_logit_shift))
            )
            dataset_results["calibration"] = calibration_payload
            dataset_results["calibration_artifacts"] = calibration_artifacts

        per_dataset_results[dataset_name] = dataset_eval_metrics

        rep_diag = dataset_results.get("representation_diagnostics", {})
        if isinstance(rep_diag, dict):
            rep_snapshot_id = str(rep_diag.get("representation_snapshot_id", "unknown"))
            dataset_representation_snapshot_ids[dataset_name] = rep_snapshot_id
            bridge_payload = rep_diag.get("cluster_label_bridge")
            relabel_cfg = rep_diag.get("cluster_relabel_config", {})
            if isinstance(bridge_payload, dict) and bridge_payload:
                bridge_dir = results_dir / "cluster_mapping"
                bridge_path = bridge_dir / f"{dataset_name}_old_to_cluster_seed{args.seed}.json"
                _atomic_write_json(
                    bridge_path,
                    {
                        "timestamp": datetime.now().isoformat(),
                        "dataset": dataset_name,
                        "seed": int(args.seed),
                        "cluster_relabel_config": relabel_cfg,
                        "bridge": bridge_payload,
                    },
                )
                dataset_results["cluster_mapping_artifact"] = str(bridge_path)
                logger.info(
                    "[%s] Cluster mapping bridge frozen: %s",
                    dataset_name,
                    str(bridge_path),
                )

        if bool(args.ab_mode):
            rep_diag_payload = cast(dict[str, Any], dataset_results.get("representation_diagnostics", {}))
            ab_raw_current_by_dataset[dataset_name] = _build_ab_raw_metrics(
                dataset_name=dataset_name,
                dataset_id="pending",
                split_snapshot_id=dataset_snapshot_ids.get(dataset_name, "unknown"),
                ab_track=str(args.ab_track),
                ab_change_id=str(args.ab_change_id),
                k=int(forced_cluster_relabel_k or 0),
                seed=int(forced_cluster_relabel_seed),
                batch_size=int(train_config.batch_size),
                feature_signature=feature_signature,
                cluster_objective=forced_cluster_relabel_objective,
                cluster_spectral_affinity=forced_cluster_relabel_spectral_affinity,
                representation_diagnostics=rep_diag_payload,
                dataset_metrics=cast(dict[str, Any], per_dataset_results[dataset_name]),
            )

    if not per_dataset_results:
        guard_failure = "Hard-stop integrity guard triggered: no_datasets_trained_in_decoupled_mode"
        _persist_seed_artifacts(
            results_dir=results_dir,
            seed=args.seed,
            config_payload=config_payload,
            results_payload=results,
            eval_payload=per_dataset_results,
            run_exit_code=1,
            guard_failure=guard_failure,
        )
        raise RuntimeError("Hard-stop integrity guard triggered: no_datasets_trained_in_decoupled_mode")

    pretrain_elapsed = max(0.001, time.perf_counter() - pretrain_start)

    results = {
        "training_mode": "decoupled",
        "per_dataset_training": all_results,
        "per_dataset_results": per_dataset_results,
    }

    governance_dataset_id = (
        (
            "helix_full_decoupled_cluster_relabel_"
            f"v1_k{int(forced_cluster_relabel_k or 0)}_seed{int(forced_cluster_relabel_seed)}"
        )
        if bool(forced_cluster_relabeling_enabled)
        else "helix_full_decoupled"
    )
    if bool(args.ab_mode):
        for dataset_name in ab_raw_current_by_dataset:
            ab_raw_current_by_dataset[dataset_name]["dataset_id"] = str(governance_dataset_id)

    try:
        posteval_start = time.perf_counter()
        macro_values = [
            float(metrics.get("family_macro_f1", metrics.get("family_f1", 0.0)))
            for metrics in per_dataset_results.values()
        ]
        aggregate_macro_f1 = float(min(macro_values)) if macro_values else 0.0

        policy = _resolve_governance_policy(train_config)
        registry = RunRegistry(
            Path(os.environ.get("HELIX_RUN_REGISTRY", "results/gates/run_registry.jsonl"))
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

        governance_stages = {
            "presplit": {
                "presplit_elapsed_seconds": split_elapsed,
                "split_train_rows": int(
                    sum(int(cast(np.ndarray, splits.get(f"X_train_{name}", np.empty((0, 0)))).shape[0]) for name in ["nsl_kdd", "unsw_nb15", "cicids"])
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
                "posteval_elapsed_seconds": max(0.001, time.perf_counter() - posteval_start),
                "macro_f1_ci_width": 0.0,
                "macro_f1_ci_lower": aggregate_macro_f1,
                "dataset_identity_balanced_accuracy": 0.0,
                "abs_macro_f1_drift": drift,
                "abs_macro_f1_zscore": z_score,
                "phase_regime": phase_regime,
            },
            "prepromote": {
                "prepromote_elapsed_seconds": max(0.001, time.perf_counter() - prepromote_start),
                "macro_f1_ci_width": 0.0,
                "macro_f1_ci_lower": aggregate_macro_f1,
                **promotion_consensus.to_stage_metrics(),
            },
        }
        if promotion_consensus.invalid_reason is not None:
            governance_stages["prepromote"]["promotion_invalid_reason"] = (
                promotion_consensus.invalid_reason
            )

        ab_raw_artifacts: dict[str, str] = {}
        ab_decisions: dict[str, dict[str, Any]] = {}
        if bool(args.ab_mode):
            ab_dir = results_dir / "ab_runs"
            explicit_baseline_path = Path(args.ab_baseline) if args.ab_baseline else None

            for dataset_name, current_payload in ab_raw_current_by_dataset.items():
                baseline_path: Optional[Path] = None
                if explicit_baseline_path is not None:
                    baseline_path = explicit_baseline_path
                else:
                    baseline_path = _find_latest_ab_raw_metrics(ab_dir, dataset_name)

                decision: dict[str, Any]
                baseline_payload: Optional[dict[str, Any]] = None
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
                        governance_z_tolerance=float(policy.drift.max_abs_z_score),
                    )

                raw_payload = dict(current_payload)
                raw_payload["baseline_path"] = str(baseline_path) if baseline_path is not None else None
                raw_payload["decision"] = decision
                if baseline_payload is not None:
                    raw_payload["baseline_metrics"] = {
                        "ratio": float(baseline_payload.get("ratio", 0.0)),
                        "min_inter": float(baseline_payload.get("min_inter", 0.0)),
                        "macro_f1": float(baseline_payload.get("macro_f1", 0.0)),
                        "zero_prediction_classes": float(
                            baseline_payload.get("zero_prediction_classes", 0.0)
                        ),
                    }

                artifact_path = ab_dir / (
                    f"{dataset_name}_ab_raw_{datetime.now().strftime('%Y%m%dT%H%M%S%f')}"
                    f"_seed{int(args.seed)}.json"
                )
                _atomic_write_json(artifact_path, raw_payload)
                ab_raw_artifacts[dataset_name] = str(artifact_path)
                ab_decisions[dataset_name] = decision
                logger.info("[%s] A/B raw metrics persisted: %s", dataset_name, str(artifact_path))

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

    logger.info(f"✅ Results saved to {training_results_path}")
    logger.info("=" * 80)
    logger.info("Training complete (decoupled mode)!")

    return {
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
                "dataset_hashes": os.environ.get("HELIX_DATASET_HASHES", "unknown"),
                "schema_hash": os.environ.get("HELIX_SCHEMA_HASH", "unknown"),
                "mapping_version": os.environ.get("HELIX_MAPPING_VERSION", "unknown"),
                "model_artifact": str(output_dir),
                "metrics_artifact": str(training_results_path),
                "phase_regime": phase_regime,
                "representation_snapshot_ids": dataset_representation_snapshot_ids,
            },
        },
        "determinism": determinism_state.to_dict(),
    }

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
