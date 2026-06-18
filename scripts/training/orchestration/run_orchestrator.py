"""Per-dataset training orchestration for HelixIDS-Full.

Phase 13A-4 extraction from train_helix_ids_full.py main().

Moved:
  - Dataset iteration loop (per-dataset training)
  - Model creation and trainer construction
  - Trainer.fit() invocation
  - Calibration invocation and result merging
  - Checkpoint artifact writing
  - A/B raw metrics construction

Dependency rules:
  - Uses lazy imports from train_helix_ids_full.py for helper functions
    (avoiding circular imports at module level)
"""

from __future__ import annotations

import datetime
import math
import sys
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from helix_ids.data.learnability_contract import compute_schema_hash
from helix_ids.governance.determinism import seed_worker
from helix_ids.models.full import HelixFullConfig, MultiTaskLoss, create_helix_full
from scripts.training._constants import ENGINEERED_FEATURE_NAMES
from scripts.training.data import (
    FrozenIndexSampler,
    MultiTaskNumpyDataset,
    _apply_label_merges,
    _assert_feature_sanity_for_dataset,
    _build_stratified_subset_indices,
    _build_stratified_val_subset,
    _inverse_frequency_weights,
    _normalize_engineered_feature_block,
    _sqrt_inverse_frequency_weights,
    build_class_index,
)
from scripts.training.orchestration import OrchestrationResult, ParsedConfig

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = _PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================================
# Internal helpers (lazy-imported from train_helix_ids_full when used)
# ============================================================================


def _lazy_import(mod_name: str, attr: str) -> Any:
    """Lazy-import an attribute from train_helix_ids_full to avoid circular imports."""
    import importlib

    m = importlib.import_module("scripts.training.train_helix_ids_full")
    return getattr(m, attr)


def _sha256_file(path: Path) -> str:
    return _lazy_import(__name__, "_sha256_file")(path)  # type: ignore[no-any-return]


def _stable_feature_signature(
    *, feature_order: list[str], schema_hash: str
) -> str:
    return _lazy_import(__name__, "_stable_feature_signature")(  # type: ignore[no-any-return]
        feature_order=feature_order, schema_hash=schema_hash
    )


def _write_isolation_snapshot_descriptor(**kwargs: Any) -> dict[str, Any]:
    return _lazy_import(__name__, "_write_isolation_snapshot_descriptor")(**kwargs)  # type: ignore[no-any-return]


def _build_model_contract_artifact(**kwargs: Any) -> dict[str, Any]:
    return _lazy_import(__name__, "_build_model_contract_artifact")(**kwargs)  # type: ignore[no-any-return]


def _write_checkpoint_artifact(
    path: Path, artifact: dict[str, Any], **kwargs: Any
) -> None:
    return _lazy_import(__name__, "_write_checkpoint_artifact")(path, artifact, **kwargs)  # type: ignore[no-any-return]


def _atomic_write_json(path: Path, payload: Any) -> None:
    return _lazy_import(__name__, "_atomic_write_json")(path, payload)  # type: ignore[no-any-return]


def _persist_seed_artifacts(**kwargs: Any) -> tuple[Path, Path]:
    return _lazy_import(__name__, "_persist_seed_artifacts")(**kwargs)  # type: ignore[no-any-return]


def _calibrate_family_predictions(**kwargs: Any) -> dict[str, Any]:
    return _lazy_import(__name__, "_calibrate_family_predictions")(**kwargs)  # type: ignore[no-any-return]


def _emit_calibration_artifacts(**kwargs: Any) -> dict[str, str]:
    return _lazy_import(__name__, "_emit_calibration_artifacts")(**kwargs)  # type: ignore[no-any-return]


def _load_eval_array(**kwargs: Any) -> np.ndarray:
    return _lazy_import(__name__, "_load_eval_array")(**kwargs)  # type: ignore[no-any-return]


def _build_ab_raw_metrics(**kwargs: Any) -> dict[str, Any]:
    fn = _lazy_import(__name__, "_build_ab_raw_metrics")
    return fn(**kwargs)  # type: ignore[no-any-return]


def _build_interleaved_round_robin_indices(
    y: np.ndarray, **kwargs: Any
) -> np.ndarray:
    return _lazy_import(__name__, "_build_interleaved_round_robin_indices")(  # type: ignore[no-any-return]
        y, **kwargs
    )


def _load_precomputed_splits(**kwargs: Any) -> Any:
    return _lazy_import(__name__, "_load_precomputed_splits")(**kwargs)


def _assert_validated_unsw_artifact(**kwargs: Any) -> None:
    return _lazy_import(__name__, "_assert_validated_unsw_artifact")(**kwargs)  # type: ignore[no-any-return]


def _assert_real_dataset_required(**kwargs: Any) -> None:
    return _lazy_import(__name__, "_assert_real_dataset_required")(**kwargs)  # type: ignore[no-any-return]


def _freeze_snapshot_if_valid(**kwargs: Any) -> dict[str, Any]:
    from helix_ids.data.learnability_contract import freeze_snapshot_if_valid as f
    return f(**kwargs)


# ============================================================================
# Constants (mirrored from train_helix_ids_full.py)
# ============================================================================

REQUIRED_GEOMETRY_FEATURE_DIM = 17


# ============================================================================
# Public API
# ============================================================================


def run_orchestration(
    parsed: ParsedConfig,
    splits: dict[str, Any],
    results_dir: Path,
    output_dir: Path,
    logger: Any,
) -> OrchestrationResult:
    """Execute per-dataset training loop.

    For each dataset in the spec:
      1. Load and validate train/val/test splits
      2. Build data loaders
      3. Create model, optimizer, loss function
      4. Construct and configure trainer
      5. Call trainer.fit()
      6. Save checkpoint artifacts
      7. Run calibration (if enabled)
      8. Build A/B raw metrics (if enabled)

    Parameters
    ----------
    parsed : ParsedConfig
        Consolidated parsed configuration.
    splits : dict[str, Any]
        Precomputed train/val/test split arrays.
    results_dir : Path
        Directory for writing training/eval/calibration artifacts.
    output_dir : Path
        Directory for writing model checkpoint artifacts.
    logger : logging.Logger
        Application logger.

    Returns
    -------
    OrchestrationResult
        Aggregated results, per-dataset metrics, A/B payloads, metadata.
    """
    # Track pretrain timing
    pretrain_start = time.perf_counter()

    # Lazy imports from within function body to avoid circular import
    from helix_ids.data.feature_harmonization import labels_to_multi_task

    # Re-import lazy helpers in function scope
    args = parsed.args
    train_config = parsed.train_config
    config_payload = parsed.config_payload
    calibration_enabled = parsed.calibration_enabled
    precomputed_splits_dir = Path(args.precomputed_splits_dir)
    eval_batch_size = int(args.eval_batch_size or train_config.batch_size)

    feature_order = [
        str(col)
        for col in np.asarray(splits["feature_columns"]).astype(str).tolist()
    ]
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

    target_family_class_count = int(HelixFullConfig().family_output_dim)

    # ------------------------------------------------------------------
    # Global family priors
    # ------------------------------------------------------------------
    global_family_priors: torch.Tensor | None = None
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
        global_smoothed = (global_counts + 1.0) / (
            global_total + int(global_counts.shape[0])
        )
        global_family_priors = torch.tensor(
            np.clip(global_smoothed, 1e-12, 1.0),
            dtype=torch.float32,
        )
        logger.info(
            "Using global family priors for logit correction: %s",
            {int(i): float(v) for i, v in enumerate(global_smoothed.tolist())},
        )

    # ------------------------------------------------------------------
    # Dataset specs
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Per-dataset training loop
    # ------------------------------------------------------------------
    for dataset_name, min_classes in dataset_specs:
        x_train_key = f"X_train_{dataset_name}"
        y_train_key = f"y_train_{dataset_name}"
        x_val_key = f"X_val_{dataset_name}"
        y_val_key = f"y_val_{dataset_name}"
        x_test_key = f"X_test_{dataset_name}"
        y_test_key = f"y_test_{dataset_name}"

        missing_keys = [
            k
            for k in [
                x_train_key,
                y_train_key,
                x_val_key,
                y_val_key,
                x_test_key,
                y_test_key,
            ]
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
        y_train_family_ds = (
            cast(np.ndarray, splits[y_train_key]).astype(np.int64, copy=False)
        )
        if dataset_name == "nsl_kdd" and parsed.forced_nsl_kdd_label_merges:
            y_train_family_ds = _apply_label_merges(
                y_train_family_ds,
                merges=parsed.forced_nsl_kdd_label_merges,
            )
            logger.info(
                "[%s] Applied label merges for collision baseline: %s",
                dataset_name,
                parsed.forced_nsl_kdd_label_merges,
            )

        if np.any(y_train_family_ds >= target_family_class_count):
            remapped = int(
                np.sum(y_train_family_ds >= target_family_class_count)
            )
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
            logger.warning(
                "[%s] Empty training split; skipping dataset.", dataset_name
            )
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
                results_payload=all_results,
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

        # Load eval arrays
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
        if dataset_name == "nsl_kdd" and parsed.forced_nsl_kdd_label_merges:
            y_val_family_ds = _apply_label_merges(
                y_val_family_ds,
                merges=parsed.forced_nsl_kdd_label_merges,
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
        if dataset_name == "nsl_kdd" and parsed.forced_nsl_kdd_label_merges:
            y_test_family_ds = _apply_label_merges(
                y_test_family_ds,
                merges=parsed.forced_nsl_kdd_label_merges,
            )
        y_test_family_ds = np.where(
            y_test_family_ds >= target_family_class_count,
            target_family_class_count - 1,
            y_test_family_ds,
        ).astype(np.int64, copy=False)

        # Normalize engineered features
        (
            x_train_ds,
            x_val_ds,
            x_test_ds,
            engineered_norm_stats,
        ) = _normalize_engineered_feature_block(
            dataset_name=dataset_name,
            x_train=x_train_ds,
            x_val=x_val_ds,
            x_test=x_test_ds,
            feature_names=feature_order,
            engineered_feature_names=set(ENGINEERED_FEATURE_NAMES),
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
            min_feature_std=1e-6,
            seed=args.seed,
            logger=logger,
        )

        # Build train dataset
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

        # Min class-4 sample enforcement
        if min_class4_samples > 0:
            class4_indices = np.asarray(
                train_class_index.get(4, np.array([], dtype=np.int64)),
                dtype=np.int64,
            )
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
            weights=class_sampling_weights.tolist(),
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
                if parsed.forced_sampler_mode == "weighted_random_sampler"
                else interleaved_sampler
            ),
            num_workers=train_config.num_workers,
            pin_memory=train_config.pin_memory,
            worker_init_fn=seed_worker,
            generator=(
                None
                if parsed.forced_sampler_mode in {"interleaved_rr", "weighted_random_sampler"}
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
                parsed.forced_class_balance_strategy,
                parsed.forced_label_smoothing,
            )
            logger.info(
                "[%s] Isolation run: strategy=%s prior_logit_correction=%s train_temp=%.2f warmup_ratio=%.2f kl_weight[warmup=0.00 post=0.00] logit_floor_weight=0.00 tail_ce_weight=0.00 lambda_family=%.2f freeze_backbone_epochs=%d unfreeze_step=%d entropy_warmup[steps=%d weight=%.3f] head_lr_multiplier=%.1f grad_clip(backbone_only)=%.3f",
                dataset_name,
                parsed.forced_class_balance_strategy,
                parsed.forced_use_logit_prior_correction,
                parsed.forced_train_temperature,
                parsed.forced_warmup_ratio,
                parsed.forced_lambda_family,
                parsed.forced_freeze_backbone_epochs,
                parsed.forced_unfreeze_backbone_step,
                parsed.forced_entropy_warmup_steps,
                parsed.forced_entropy_warmup_weight,
                parsed.forced_head_lr_multiplier,
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
                    "lr": train_config.learning_rate * parsed.forced_head_lr_multiplier,
                    "lr_scale": parsed.forced_head_lr_multiplier,
                    "group_name": "family_head",
                },
            ],
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
        )
        loss_fn = MultiTaskLoss(
            lambda_binary=train_config.lambda_binary,
            lambda_family=train_config.lambda_family,
            balance_strategy=parsed.forced_class_balance_strategy,
            focal_gamma=parsed.forced_focal_gamma,
            label_smoothing=parsed.forced_label_smoothing,
            use_class_weights=bool(parsed.forced_use_class_weights),
            focal_use_class_weights=False,
            entropy_regularization=parsed.forced_entropy_regularization,
            family_logit_margin=1.0,
            family_margin_loss_weight=float(parsed.forced_family_margin_loss_weight),
            family_class4_logit_penalty_weight=float(
                parsed.forced_family_class4_logit_penalty_weight
            ),
            family_class4_logit_penalty_class=4,
            family_feature_separation_weight=float(parsed.forced_family_feature_separation_weight),
            family_feature_separation_class=4,
            family_class4_target_scale=float(parsed.forced_family_class4_target_scale),
        )

        _HelixFullTrainer = _lazy_import(__name__, "HelixFullTrainer")
        trainer = _HelixFullTrainer(
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
        trainer.disable_tail_focal_regularizer = parsed.forced_class_balance_strategy != "focal"
        trainer.configure_family_controls(
            family_class_priors=family_priors if parsed.forced_use_logit_prior_correction else None,
            tail_class_mask=None,
            balance_strategy=parsed.forced_class_balance_strategy,
            focal_warmup_epochs=0,
            warmup_ratio=parsed.forced_warmup_ratio,
            train_temperature=parsed.forced_train_temperature,
        )
        total_steps_for_dataset = max(1, int(len(train_loader)) * max(1, int(train_config.epochs)))
        representation_only_steps = max(
            int(parsed.forced_representation_only_steps),
            int(math.ceil(total_steps_for_dataset * float(parsed.forced_representation_only_ratio))),
        )
        head_only_steps = max(
            1,
            int(math.ceil(total_steps_for_dataset * float(parsed.forced_head_only_ratio))),
        )
        trainer.configure_structure_recovery(
            active_family_classes={int(c) for c in np.unique(y_train_family_ds).tolist()},
            supcon_weight=parsed.forced_supcon_weight,
            supcon_temperature=parsed.forced_supcon_temperature,
            step_coverage_check_step=parsed.forced_step_coverage_check_step,
            representation_diagnostic_mode=parsed.forced_representation_diagnostic_mode,
            phase_settings={
                "representation_only_steps": representation_only_steps,
                "representation_micro_cycle_steps": [
                    int(v) for v in parsed.forced_representation_micro_cycle_steps
                ],
                "use_energy_based_family_objective": bool(
                    parsed.forced_use_energy_based_family_objective
                ),
                "adaptive_exit_ratio_threshold": parsed.forced_adaptive_exit_ratio_threshold,
                "adaptive_exit_min_inter_threshold": parsed.forced_adaptive_exit_min_inter_threshold,
                "head_only_steps": head_only_steps,
                "joint_finetune_backbone_lr_multiplier": parsed.forced_joint_finetune_backbone_lr_multiplier,
                "joint_finetune_head_lr_multiplier": parsed.forced_joint_finetune_head_lr_multiplier,
                "geometry_ratio_warmup_threshold": parsed.forced_geometry_ratio_warmup_threshold,
                "geometry_ratio_post_phase_threshold": parsed.forced_geometry_ratio_post_phase_threshold,
                "enforce_all_classes_per_batch": parsed.forced_enforce_all_classes_per_batch,
                "sampler_mode": parsed.forced_sampler_mode,
            },
            cluster_relabeling_enabled=parsed.forced_cluster_relabeling_enabled,
            cluster_relabel_k=parsed.forced_cluster_relabel_k,
            cluster_relabel_seed=parsed.forced_cluster_relabel_seed,
            cluster_relabel_objective=parsed.forced_cluster_relabel_objective,
            cluster_relabel_spectral_affinity=parsed.forced_cluster_relabel_spectral_affinity,
        )
        trainer.feature_order = feature_order
        trainer.schema_hash = schema_hash

        logger.info(f"[{dataset_name}] Starting isolated training...")
        dataset_train_start = time.perf_counter()
        try:
            dataset_results = trainer.fit()
        except Exception as exc:
            all_results.setdefault("representation_diagnostics", {})[dataset_name] = dict(
                trainer.representation_diagnostics
            )
            guard_failure = str(exc)
            _persist_seed_artifacts(
                results_dir=results_dir,
                seed=args.seed,
                config_payload=config_payload,
                results_payload=all_results,
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
                        "timestamp": datetime.datetime.now().isoformat(),
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
                k=int(parsed.forced_cluster_relabel_k or 0),
                seed=int(parsed.forced_cluster_relabel_seed),
                batch_size=int(train_config.batch_size),
                feature_signature=feature_signature,
                cluster_objective=parsed.forced_cluster_relabel_objective,
                cluster_spectral_affinity=parsed.forced_cluster_relabel_spectral_affinity,
                representation_diagnostics=rep_diag_payload,
                dataset_metrics=cast(dict[str, Any], per_dataset_results[dataset_name]),
            )

    if not per_dataset_results:
        guard_failure = "Hard-stop integrity guard triggered: no_datasets_trained_in_decoupled_mode"
        _persist_seed_artifacts(
            results_dir=results_dir,
            seed=args.seed,
            config_payload=config_payload,
            results_payload=all_results,
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
            f"v1_k{int(parsed.forced_cluster_relabel_k or 0)}_seed{int(parsed.forced_cluster_relabel_seed)}"
        )
        if bool(parsed.forced_cluster_relabeling_enabled)
        else "helix_full_decoupled"
    )
    if bool(args.ab_mode):
        for dataset_name in ab_raw_current_by_dataset:
            ab_raw_current_by_dataset[dataset_name]["dataset_id"] = str(governance_dataset_id)


    return OrchestrationResult(
        per_dataset_results=per_dataset_results,
        all_results=all_results,
        ab_raw_current_by_dataset=ab_raw_current_by_dataset,
        dataset_snapshot_ids=dataset_snapshot_ids,
        dataset_representation_snapshot_ids=dataset_representation_snapshot_ids,
        training_elapsed_total=training_elapsed_total,
        feature_order=feature_order,
        schema_hash=schema_hash,
        feature_signature=feature_signature,
        pretrain_elapsed=pretrain_elapsed,
        governance_dataset_id=governance_dataset_id,
        results=results,
    )
