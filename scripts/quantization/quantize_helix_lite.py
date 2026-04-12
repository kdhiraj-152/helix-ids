"""
Phase 4: Quantize HelixIDS-Full -> HelixIDS-Lite.

Creates a lightweight INT8-quantized model artifact for edge/server-lite deployment
and exports optional ONNX when available.

Usage:
  python scripts/quantize_helix_lite.py \
    --checkpoint models/helix_full/helix_full_best.pt \
    --output-dir models/quantized
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

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


def _export_onnx(model: torch.nn.Module, output_path: Path) -> bool:
    try:
        import onnx  # noqa: F401

        dummy = torch.randn(1, 31)
        torch.onnx.export(
            model,
            (dummy,),
            output_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["features"],
            output_names=["binary_logits", "family_logits"],
            dynamic_axes={"features": {0: "batch"}},
        )
        return True
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize HelixIDS-Lite")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("models/quantized"))
    parser.add_argument("--eval-samples", type=int, default=2048)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = _load_model(args.checkpoint)
    original_size = measure_model_size(model)

    quantizer = DynamicQuantizer()
    qmodel = quantizer.quantize_dynamic(model)
    qresult = quantizer.get_results().to_dict()

    # Save quantized torch artifact
    q_pt = args.output_dir / "helix_ids_lite_int8.pt"
    torch.save(qmodel, q_pt)

    # Optional ONNX export for lite inference runtimes
    onnx_path = args.output_dir / "helix_ids_lite.onnx"
    onnx_exported = _export_onnx(model, onnx_path)

    # Lightweight sanity eval on synthetic data for agreement tracking
    x_eval = torch.randn(args.eval_samples, 31)
    y_eval = torch.randint(0, 2, (args.eval_samples,))
    eval_stats = compare_accuracy(
        model, qmodel, x_eval, y_eval, class_names=["Normal", "Attack"], verbose=False
    )

    report = {
        "checkpoint": str(args.checkpoint),
        "artifacts": {
            "lite_pt": str(q_pt),
            "lite_onnx": str(onnx_path) if onnx_exported else None,
        },
        "original_size_kb": original_size["size_kb"],
        "quantization": qresult,
        "agreement_eval": {
            "prediction_agreement": eval_stats["prediction_agreement"],
            "accuracy_retained": eval_stats["accuracy_retained"],
        },
    }

    report_path = args.output_dir / "quantization_lite_report.json"
    with open(report_path, "w", encoding="ascii") as f:
        json.dump(report, f, indent=2)

    print(f"Saved lite artifact: {q_pt}")
    if onnx_exported:
        print(f"Saved ONNX artifact: {onnx_path}")
    else:
        print("ONNX export skipped (onnx dependency unavailable or export failed).")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
