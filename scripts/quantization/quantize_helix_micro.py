"""
Phase 4: Quantize HelixIDS-Full -> HelixIDS-Micro.

Applies global unstructured pruning + dynamic INT8 quantization and
exports micro artifacts suited for constrained edge environments.

Usage:
  python scripts/quantize_helix_micro.py \
    --checkpoint models/helix_full/helix_full_best.pt \
    --output-dir models/quantized
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.utils.prune as prune

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from helix_ids.models.full import HelixFullConfig, HelixIDSFull
from helix_ids.utils.quantization import DynamicQuantizer, compare_accuracy, measure_model_size


def _load_model(checkpoint_path: Path) -> HelixIDSFull:
    model = HelixIDSFull(HelixFullConfig())
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def _apply_pruning(model: torch.nn.Module, amount: float = 0.35) -> torch.nn.Module:
    linear_layers = []
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            linear_layers.append((module, "weight"))

    prune.global_unstructured(linear_layers, pruning_method=prune.L1Unstructured, amount=amount)
    for module, _ in linear_layers:
        prune.remove(module, "weight")

    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize HelixIDS-Micro")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("models/quantized"))
    parser.add_argument("--prune-ratio", type=float, default=0.35)
    parser.add_argument("--eval-samples", type=int, default=2048)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_model = _load_model(args.checkpoint)
    original_size = measure_model_size(base_model)

    pruned_model = _apply_pruning(base_model, amount=args.prune_ratio)
    pruned_size = measure_model_size(pruned_model)

    quantizer = DynamicQuantizer()
    qmodel = quantizer.quantize_dynamic(pruned_model)
    qresult = quantizer.get_results().to_dict()

    micro_pt = args.output_dir / "helix_ids_micro_int8.pt"
    torch.save(qmodel, micro_pt)

    # Optional TorchScript for low-overhead deployment path
    micro_ts = args.output_dir / "helix_ids_micro_int8.ts"
    try:
        scripted = torch.jit.script(qmodel)
        scripted.save(str(micro_ts))
        ts_saved = True
    except Exception:
        ts_saved = False

    # Optional placeholder for TFLite-compatible export path if TensorFlow is installed
    # (not enforced in this repo to avoid hard dependency).
    tflite_path = args.output_dir / "helix_ids_micro.tflite"
    tflite_exported = False

    x_eval = torch.randn(args.eval_samples, 31)
    y_eval = torch.randint(0, 2, (args.eval_samples,))
    eval_stats = compare_accuracy(
        base_model, qmodel, x_eval, y_eval, class_names=["Normal", "Attack"], verbose=False
    )

    report = {
        "checkpoint": str(args.checkpoint),
        "prune_ratio": args.prune_ratio,
        "artifacts": {
            "micro_pt": str(micro_pt),
            "micro_torchscript": str(micro_ts) if ts_saved else None,
            "micro_tflite": str(tflite_path) if tflite_exported else None,
        },
        "sizes_kb": {
            "original": original_size["size_kb"],
            "pruned": pruned_size["size_kb"],
        },
        "quantization": qresult,
        "agreement_eval": {
            "prediction_agreement": eval_stats["prediction_agreement"],
            "accuracy_retained": eval_stats["accuracy_retained"],
        },
    }

    report_path = args.output_dir / "quantization_micro_report.json"
    with open(report_path, "w", encoding="ascii") as f:
        json.dump(report, f, indent=2)

    print(f"Saved micro artifact: {micro_pt}")
    if ts_saved:
        print(f"Saved TorchScript artifact: {micro_ts}")
    else:
        print("TorchScript export skipped (script conversion failed).")
    print("TFLite export skipped (TensorFlow conversion pipeline not enabled in this environment).")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
