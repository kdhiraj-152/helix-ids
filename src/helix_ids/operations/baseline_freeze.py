from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from typing import cast

import numpy as np
import torch

from helix_ids.contracts.schema_contract import (
    CANONICAL_BINARY_CLASSES,
    CANONICAL_FAMILY_CLASSES,
    CANONICAL_INPUT_DIM,
    SCHEMA_VERSION,
    assert_runtime_contract,
    runtime_contract_payload,
)


IMMUTABLE_DIR_MODE = 0o500
IMMUTABLE_FILE_MODE = 0o400


@dataclass(frozen=True)
class FreezeInputs:
    release_id: str
    model_checkpoint: Path
    artifact_dir: Path
    training_report: Path
    eval_report: Path
    output_root: Path


def _sha256_file(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _dataset_manifest(artifact_dir: Path) -> dict[str, Any]:
    npy_files = sorted(artifact_dir.glob("*.npy"))
    entries = []
    for f in npy_files:
        entries.append({
            "file": f.name,
            "bytes": int(f.stat().st_size),
            "sha256": _sha256_file(f),
        })
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "artifact_dir": str(artifact_dir),
        "file_count": len(entries),
        "files": entries,
        "dataset_hash": hashlib.sha256(canonical).hexdigest(),
    }


def _create_split_indices_snapshot(artifact_dir: Path, split_dir: Path) -> dict[str, Any]:
    split_dir.mkdir(parents=True, exist_ok=True)
    split_entries: list[dict[str, Any]] = []

    for y_file in sorted(artifact_dir.glob("y_*.npy")):
        labels = np.load(y_file, allow_pickle=False)
        indices = np.arange(labels.shape[0], dtype=np.int64)
        out_name = y_file.name.replace("y_", "idx_")
        out_path = split_dir / out_name
        np.save(out_path, indices)
        split_entries.append(
            {
                "source_label_file": y_file.name,
                "index_file": out_name,
                "sample_count": int(indices.shape[0]),
                "sha256": _sha256_file(out_path),
            }
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "entries": split_entries,
    }


def _apply_immutable_permissions(release_dir: Path) -> None:
    for p in sorted(release_dir.rglob("*")):
        try:
            if p.is_dir():
                os.chmod(p, IMMUTABLE_DIR_MODE)
            else:
                os.chmod(p, IMMUTABLE_FILE_MODE)
        except PermissionError:
            # Best-effort on filesystems that do not support POSIX perms.
            continue
    try:
        os.chmod(release_dir, IMMUTABLE_DIR_MODE)
    except PermissionError:
        pass


def seal_baseline(
    *,
    release_id: str,
    model_checkpoint: Path,
    artifact_dir: Path,
    training_report: Path,
    eval_report: Path,
    output_root: Path = Path("artifacts/releases"),
) -> Path:
    """Seal model + data + metrics as immutable release artifact."""
    inputs = FreezeInputs(
        release_id=release_id,
        model_checkpoint=model_checkpoint,
        artifact_dir=artifact_dir,
        training_report=training_report,
        eval_report=eval_report,
        output_root=output_root,
    )

    for p in [inputs.model_checkpoint, inputs.artifact_dir, inputs.training_report, inputs.eval_report]:
        if not Path(p).exists():
            raise FileNotFoundError(f"Missing required input: {p}")

    release_dir = inputs.output_root / inputs.release_id
    if release_dir.exists():
        raise FileExistsError(
            f"Immutable release already exists: {release_dir}. Use a new version id."
        )

    checkpoint_dir = release_dir / "checkpoint"
    config_dir = release_dir / "config"
    dataset_dir = release_dir / "dataset"
    split_dir = release_dir / "splits"
    metrics_dir = release_dir / "metrics"
    governance_dir = release_dir / "governance"

    for d in [checkpoint_dir, config_dir, dataset_dir, split_dir, metrics_dir, governance_dir]:
        d.mkdir(parents=True, exist_ok=True)

    frozen_checkpoint = checkpoint_dir / inputs.model_checkpoint.name
    shutil.copy2(inputs.model_checkpoint, frozen_checkpoint)

    checkpoint_contract_path = inputs.model_checkpoint.with_suffix(inputs.model_checkpoint.suffix + ".contract.json")
    checkpoint_feature_order_path = inputs.model_checkpoint.with_suffix(inputs.model_checkpoint.suffix + ".feature_order.json")
    checkpoint_schema_hash_path = inputs.model_checkpoint.with_suffix(inputs.model_checkpoint.suffix + ".schema_hash.txt")
    for sidecar_path in [checkpoint_contract_path, checkpoint_feature_order_path, checkpoint_schema_hash_path]:
        if not sidecar_path.exists():
            raise RuntimeError(f"Checkpoint sidecar missing during freeze: {sidecar_path}")
        shutil.copy2(sidecar_path, checkpoint_dir / sidecar_path.name)

    checkpoint_obj = torch.load(inputs.model_checkpoint, map_location="cpu", weights_only=True)
    required_metadata = [
        "schema_version",
        "schema_hash",
        "feature_order",
        "input_dim",
        "binary_output_dim",
        "family_output_dim",
    ]
    missing_metadata = [key for key in required_metadata if key not in checkpoint_obj]
    if missing_metadata:
        raise RuntimeError(f"Checkpoint metadata incomplete; missing={missing_metadata}")
    assert_runtime_contract(
        schema_version=str(checkpoint_obj["schema_version"]),
        schema_hash=str(checkpoint_obj["schema_hash"]),
        feature_order=[str(feature) for feature in checkpoint_obj["feature_order"]],
        input_dim=int(checkpoint_obj["input_dim"]),
        binary_output_dim=int(checkpoint_obj["binary_output_dim"]),
        family_output_dim=int(checkpoint_obj["family_output_dim"]),
        context="freeze checkpoint",
    )
    sidecar_payload = runtime_contract_payload()
    loaded_contract = json.loads(checkpoint_contract_path.read_text(encoding="utf-8"))
    assert_runtime_contract(
        schema_version=str(loaded_contract["schema_version"]),
        schema_hash=str(loaded_contract["schema_hash"]),
        feature_order=[str(feature) for feature in loaded_contract["feature_order"]],
        input_dim=int(loaded_contract["input_dim"]),
        binary_output_dim=int(loaded_contract["binary_output_dim"]),
        family_output_dim=int(loaded_contract["family_output_dim"]),
        context="freeze checkpoint sidecar",
    )
    if loaded_contract != sidecar_payload:
        raise RuntimeError("Checkpoint contract sidecar does not match the immutable runtime contract")
    training_payload = _load_json(inputs.training_report)
    eval_payload = _load_json(inputs.eval_report)

    config_snapshot = {
        "release_id": inputs.release_id,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "training_config": training_payload.get("config", {}),
        "schema_version": checkpoint_obj.get("schema_version"),
        "schema_hash": checkpoint_obj.get("schema_hash"),
        "feature_order": checkpoint_obj.get("feature_order", []),
        "input_dim": checkpoint_obj.get("input_dim"),
        "binary_output_dim": checkpoint_obj.get("binary_output_dim"),
        "family_output_dim": checkpoint_obj.get("family_output_dim"),
    }
    _write_json(config_dir / "config_snapshot.json", config_snapshot)

    dataset_manifest = _dataset_manifest(inputs.artifact_dir)
    _write_json(dataset_dir / "dataset_hash_manifest.json", dataset_manifest)

    split_manifest = _create_split_indices_snapshot(inputs.artifact_dir, split_dir)
    _write_json(split_dir / "split_indices_manifest.json", split_manifest)

    metric_report = {
        "release_id": inputs.release_id,
        "training": training_payload,
        "evaluation": eval_payload,
    }
    _write_json(metrics_dir / "metric_report.json", metric_report)

    train_cfg = training_payload.get("config", {}) if isinstance(training_payload, dict) else {}
    determinism_lock = {
        "seed": int(train_cfg.get("seed", 42)),
        "dataloader_order": str(train_cfg.get("sampler_mode", "natural_order")),
        "torch_backend_flags": {
            "torch_use_deterministic_algorithms": True,
            "torch_cudnn_deterministic": True,
            "torch_cudnn_benchmark": False,
        },
        "python_hash_seed": str(train_cfg.get("seed", 42)),
        "guarantee": "bitwise_reproducibility",
    }
    _write_json(governance_dir / "determinism_lock.json", determinism_lock)

    governance_policy = {
        "release_id": inputs.release_id,
        "determinism": {
            "required": True,
            "guarantee": "bitwise_reproducibility",
            "lock_file": "governance/determinism_lock.json",
        },
        "invariants": {
            "zero_prediction_classes": 0,
            "no_crash": True,
            "no_drift": True,
        },
        "allowed_future_axes": [
            "feature_expansion",
            "clustering_objective_upgrade",
        ],
        "constraint": "baseline_must_never_regress",
    }
    _write_json(governance_dir / "baseline_policy.json", governance_policy)

    all_files = [p for p in release_dir.rglob("*") if p.is_file()]
    manifest_files = []
    for p in sorted(all_files):
        manifest_files.append(
            {
                "path": str(p.relative_to(release_dir)),
                "sha256": _sha256_file(p),
                "bytes": int(p.stat().st_size),
            }
        )

    release_manifest = {
        "release_id": inputs.release_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "model_checkpoint": str(inputs.model_checkpoint),
            "artifact_dir": str(inputs.artifact_dir),
            "training_report": str(inputs.training_report),
            "eval_report": str(inputs.eval_report),
        },
        "checkpoint_sha256": _sha256_file(frozen_checkpoint),
        "dataset_hash": dataset_manifest["dataset_hash"],
        "files": manifest_files,
        "immutable": True,
    }
    _write_json(release_dir / "manifest.json", release_manifest)
    (release_dir / "RELEASE_ID").write_text(inputs.release_id + "\n", encoding="utf-8")

    _apply_immutable_permissions(release_dir)
    return release_dir


def seal_baseline_from_cli(
    *,
    release_id: str,
    model_checkpoint: str,
    artifact_dir: str,
    training_report: str,
    eval_report: str,
    output_root: str,
) -> str:
    release_dir = seal_baseline(
        release_id=release_id,
        model_checkpoint=Path(model_checkpoint),
        artifact_dir=Path(artifact_dir),
        training_report=Path(training_report),
        eval_report=Path(eval_report),
        output_root=Path(output_root),
    )
    return str(release_dir)
