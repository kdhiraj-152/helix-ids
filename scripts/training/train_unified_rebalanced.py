#!/usr/bin/env python3
"""Direct domain-adaptation training on raw NSL-KDD, UNSW-NB15, and CICIDS.

This training path intentionally bypasses the harmonization/probe gate stack:
feature_harmonization -> multi_dataset_loader -> _select_signal_features -> UNSW probe checks.

Instead it routes through transfer-learning adaptation modules:
raw loaders -> dataset-specific preprocessing -> FeatureAligner -> DANN/MMD/CORAL -> classifier.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

# Add project root and src to path for direct script execution.
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.helix_ids.data.loader_core import UnifiedDataLoader
from src.helix_ids.models.adaptation.transfer_learning import (
    MultiDatasetPretrainer,
    TransferLearningConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("DirectAdaptationTraining")


def _predict_in_batches(
    pretrainer: MultiDatasetPretrainer,
    x: np.ndarray,
    *,
    dataset_name: str,
    batch_rows: int,
) -> np.ndarray:
    """Run pretrainer predictions in bounded chunks to avoid large tensor allocations."""
    if x.shape[0] <= batch_rows:
        return pretrainer.predict(x, dataset_name=dataset_name)

    preds: list[np.ndarray] = []
    for start_idx in range(0, int(x.shape[0]), int(batch_rows)):
        end_idx = min(int(x.shape[0]), start_idx + int(batch_rows))
        chunk = np.asarray(x[start_idx:end_idx], dtype=np.float32)
        preds.append(pretrainer.predict(chunk, dataset_name=dataset_name))
    return np.concatenate(preds, axis=0)


def _dataset_eval(
    pretrainer: MultiDatasetPretrainer,
    dataset_name: str,
    *,
    predict_batch_rows: int = 200000,
) -> dict[str, object]:
    """Evaluate trained pretrainer on a single raw dataset with strict schema."""
    loader = UnifiedDataLoader(
        scale_features=True,
        handle_missing=True,
        handle_outliers=True,
        label_mode="unified_5class",
        verbose=False,
    )
    x, y, class_names = cast(
        tuple[np.ndarray, np.ndarray, list[str]],
        loader.load(dataset_name, return_class_names=True),
    )
    preds = _predict_in_batches(
        pretrainer,
        x,
        dataset_name=dataset_name,
        batch_rows=predict_batch_rows,
    )

    labels = np.arange(len(class_names), dtype=np.int64)
    precision, recall, per_class_f1, support = precision_recall_fscore_support(
        y,
        preds,
        labels=labels,
        zero_division=0,
    )
    cm = confusion_matrix(y, preds, labels=labels)
    class_name_map = [str(name) for name in class_names]

    return {
        "samples": int(y.shape[0]),
        "accuracy": float(accuracy_score(y, preds)),
        "macro_f1": float(f1_score(y, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, preds, average="weighted", zero_division=0)),
        "predicted_class_count": int(np.unique(preds).size),
        "class_names": class_name_map,
        "per_class_precision": {
            class_name_map[idx]: float(precision[idx]) for idx in range(len(class_name_map))
        },
        "per_class_recall": {
            class_name_map[idx]: float(recall[idx]) for idx in range(len(class_name_map))
        },
        "per_class_f1": {
            class_name_map[idx]: float(per_class_f1[idx]) for idx in range(len(class_name_map))
        },
        "per_class_support": {
            class_name_map[idx]: int(support[idx]) for idx in range(len(class_name_map))
        },
        "confusion_matrix": cm.astype(int).tolist(),
    }


def main() -> None:
    print(f"T0: start {time.time()}", flush=True)
    parser = argparse.ArgumentParser(description="Direct adaptation training for HelixIDS")
    parser.add_argument("--device", default="mps", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--pretrain-epochs", type=int, default=50)
    parser.add_argument("--finetune-epochs", type=int, default=100)
    parser.add_argument("--freeze-encoder-epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pretrain-datasets",
        nargs="+",
        default=["nsl-kdd", "unsw-nb15", "cicids-2018"],
        help="Datasets sampled equally per pretrain step (default: nsl-kdd unsw-nb15 cicids-2018)",
    )
    parser.add_argument(
        "--eval-datasets",
        nargs="+",
        default=["nsl-kdd", "unsw-nb15", "cicids-2018"],
        help="Datasets to evaluate after training",
    )
    parser.add_argument("--target-dataset", default="nsl-kdd")
    parser.add_argument(
        "--eval-predict-batch-rows",
        type=int,
        default=200000,
        help="Rows per evaluation prediction chunk to prevent large buffer allocations",
    )
    parser.add_argument("--macro-f1-floor", type=float, default=0.0)
    parser.add_argument("--disable-class-weights", action="store_true")
    parser.add_argument(
        "--class-weight-power",
        type=float,
        default=1.0,
        help="Exponent to amplify inverse-frequency class weights (1.0 = standard inverse-frequency)",
    )
    parser.add_argument(
        "--max-class-weight",
        type=float,
        default=10.0,
        help="Upper bound for per-class weights before mean normalization",
    )
    parser.add_argument(
        "--use-focal-loss",
        action="store_true",
        help="Use weighted focal loss instead of weighted cross-entropy",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Focal loss gamma (only used when --use-focal-loss is enabled)",
    )
    parser.add_argument(
        "--use-balanced-sampler",
        action="store_true",
        help="Use class-balanced replacement sampling during pretraining and fine-tuning",
    )
    parser.add_argument("--disable-domain-monitor", action="store_true")
    parser.add_argument("--disable-dann", action="store_true")
    parser.add_argument("--disable-mmd", action="store_true")
    parser.add_argument("--disable-coral", action="store_true")
    parser.add_argument(
        "--no-da",
        action="store_true",
        help="Disable all domain adaptation losses (DANN/MMD/CORAL)",
    )
    args = parser.parse_args()

    use_dann = not args.disable_dann
    use_mmd = not args.disable_mmd
    use_coral = not args.disable_coral
    if args.no_da:
        use_dann = False
        use_mmd = False
        use_coral = False

    logger.info("=" * 80)
    logger.info("Starting direct adaptation training path (harmonization/probe bypass)")
    logger.info("Pretrain datasets: %s", args.pretrain_datasets)
    logger.info("Target dataset: %s", args.target_dataset)
    logger.info("Eval datasets: %s", args.eval_datasets)
    logger.info("Label mode: unified_5class")
    logger.info(
        "Imbalance config: class_weights=%s weight_power=%.2f max_weight=%.2f focal=%s gamma=%.2f sampler=%s",
        not args.disable_class_weights,
        args.class_weight_power,
        args.max_class_weight,
        args.use_focal_loss,
        args.focal_gamma,
        args.use_balanced_sampler,
    )
    logger.info(
        "DA losses: DANN=%s MMD=%s CORAL=%s",
        use_dann,
        use_mmd,
        use_coral,
    )
    logger.info("=" * 80)

    model_dir = Path("models/helix_full_unified")
    results_dir = Path("results/unified_training")
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    config = TransferLearningConfig(
        source_datasets=list(args.pretrain_datasets),
        target_dataset=args.target_dataset,
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        checkpoint_dir=model_dir,
        device=args.device,
        seed=args.seed,
        num_classes=5,
        use_dann=use_dann,
        use_mmd=use_mmd,
        use_coral=use_coral,
        use_class_weights=not args.disable_class_weights,
        class_weight_power=args.class_weight_power,
        max_class_weight=args.max_class_weight,
        use_focal_loss=args.use_focal_loss,
        focal_gamma=args.focal_gamma,
        use_balanced_sampler=args.use_balanced_sampler,
        monitor_domain_collapse=not args.disable_domain_monitor,
    )

    pretrainer = MultiDatasetPretrainer(config)

    # Hard startup trace markers to diagnose stalls before first training batch.
    trace_dataset = str(args.pretrain_datasets[0])
    print(f"T1: before dataset load ({trace_dataset})", flush=True)
    trace_x, trace_y, trace_domain_ids, _trace_n_features = pretrainer._load_dataset(trace_dataset)
    print(
        f"T2: after dataset load ({trace_dataset}) rows={trace_x.shape[0]} cols={trace_x.shape[1]}",
        flush=True,
    )

    print("T3: before dataloader", flush=True)
    trace_loader = pretrainer._create_dataloader(
        trace_x,
        trace_y,
        domain_ids=trace_domain_ids,
        batch_size=args.batch_size,
        shuffle=False,
    )
    print("T4: after dataloader", flush=True)

    print("T5: before first batch", flush=True)
    _trace_batch = next(iter(trace_loader))
    print("T6: after first batch", flush=True)

    logger.info("Pretraining with domain adaptation losses...")
    pretrain_history = pretrainer.pretrain()

    logger.info("Finetuning on NSL-KDD target domain...")
    finetune_history = pretrainer.finetune(
        target_dataset=args.target_dataset,
        epochs=args.finetune_epochs,
        freeze_encoder_epochs=args.freeze_encoder_epochs,
    )

    logger.info("Running per-dataset evaluation on raw loaders...")
    eval_results = {
        dataset_name: _dataset_eval(
            pretrainer,
            dataset_name,
            predict_batch_rows=max(1, int(args.eval_predict_batch_rows)),
        )
        for dataset_name in args.eval_datasets
    }

    macro_summary: dict[str, float] = {}
    for name, metrics in eval_results.items():
        macro_val = metrics.get("macro_f1")
        if not isinstance(macro_val, (int, float)):
            raise TypeError(f"Invalid macro_f1 value for {name}: {macro_val!r}")
        macro_summary[name] = float(macro_val)
    logger.info("Macro-F1 summary: %s", macro_summary)

    macro_values = list(macro_summary.values())
    evaluation_summary = {
        "primary_metric": "macro_f1",
        "macro_f1_min": float(min(macro_values)),
        "macro_f1_mean": float(np.mean(macro_values)),
        "macro_f1_max": float(max(macro_values)),
    }

    if args.macro_f1_floor > 0.0 and evaluation_summary["macro_f1_min"] < args.macro_f1_floor:
        raise RuntimeError(
            "Macro-F1 floor check failed: "
            f"min={evaluation_summary['macro_f1_min']:.4f} < floor={args.macro_f1_floor:.4f}"
        )

    checkpoint_path = model_dir / "helix_transfer_direct_adaptation.pt"
    pretrainer.save_checkpoint(checkpoint_path)

    results = {
        "timestamp": datetime.now().isoformat(),
        "training_mode": "direct_adaptation",
        "routing": {
            "bypassed": [
                "feature_harmonization",
                "multi_dataset_loader",
                "_select_signal_features",
                "unsw_probe_gate",
            ],
            "used": [
                "raw_dataset_loaders",
                "dataset_specific_preprocessing",
                "FeatureAligner",
                "DANN_MMD_CORAL" if (use_dann or use_mmd or use_coral) else "supervised_only",
                "classifier",
            ],
            "label_mapping": "unified_5class",
        },
        "config": {
            "device": args.device,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "pretrain_epochs": args.pretrain_epochs,
            "finetune_epochs": args.finetune_epochs,
            "freeze_encoder_epochs": args.freeze_encoder_epochs,
            "seed": args.seed,
            "source_datasets": config.source_datasets,
            "target_dataset": config.target_dataset,
            "use_dann": use_dann,
            "use_mmd": use_mmd,
            "use_coral": use_coral,
            "use_class_weights": config.use_class_weights,
            "class_weight_power": config.class_weight_power,
            "max_class_weight": config.max_class_weight,
            "use_focal_loss": config.use_focal_loss,
            "focal_gamma": config.focal_gamma,
            "use_balanced_sampler": config.use_balanced_sampler,
        },
        "pretrain_history": pretrain_history,
        "finetune_history": finetune_history,
        "evaluation": eval_results,
        "evaluation_summary": evaluation_summary,
    }

    output_path = results_dir / "training_results.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Results written to %s", output_path)
    logger.info("Checkpoint written to %s", checkpoint_path)


if __name__ == "__main__":
    main()
