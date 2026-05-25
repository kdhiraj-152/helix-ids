#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from helix_ids.operations.inference_runtime import HelixInferenceRuntime, InferenceConfig

NULL_TYPE = "float|null"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export frozen HELIX baseline to TorchScript/ONNX")
    p.add_argument("--checkpoint", default="models/helix_full/helix_full_nsl_kdd_best.pt")
    p.add_argument("--output-dir", default="artifacts/releases/helix_ids_v1.0/packaging")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--prediction-floor", type=float, default=1e-6)
    p.add_argument("--class-margin-override-enabled", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--class-margin-override-class-id", type=int, default=4)
    p.add_argument("--class-margin-override-tau", type=float, default=70_000.0)
    p.add_argument("--class-margin-override-use-percentile", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--class-margin-override-percentile-k", type=float, default=75.0)
    p.add_argument("--class-margin-override-hybrid-and", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--class-margin-override-buffer-size", type=int, default=1000)
    p.add_argument("--class-margin-override-warmup-min-samples", type=int, default=200)
    p.add_argument("--class-margin-override-rate-guard-threshold", type=float, default=0.5)
    p.add_argument("--class-margin-override-rate-guard-window", type=int, default=1000)
    p.add_argument("--class-margin-override-baseline-precision", type=float, default=None)
    p.add_argument("--class-margin-override-precision-disable-ratio", type=float, default=0.8)
    p.add_argument("--device", default="cpu")
    p.add_argument("--enable-global-coverage-floor", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--global-coverage-quantile", type=float, default=0.95)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runtime = HelixInferenceRuntime(
        Path(args.checkpoint),
        device=args.device,
        config=InferenceConfig(
            fixed_temperature=args.temperature,
            prediction_floor=args.prediction_floor,
            global_coverage_floor=bool(args.enable_global_coverage_floor),
            global_coverage_quantile=float(args.global_coverage_quantile),
            class_margin_override_enabled=bool(args.class_margin_override_enabled),
            class_margin_override_class_id=int(args.class_margin_override_class_id),
            class_margin_override_tau=float(args.class_margin_override_tau),
            class_margin_override_use_percentile=bool(args.class_margin_override_use_percentile),
            class_margin_override_percentile_k=float(args.class_margin_override_percentile_k),
            class_margin_override_hybrid_and=bool(args.class_margin_override_hybrid_and),
            class_margin_override_buffer_size=int(args.class_margin_override_buffer_size),
            class_margin_override_warmup_min_samples=int(args.class_margin_override_warmup_min_samples),
            class_margin_override_rate_guard_threshold=float(args.class_margin_override_rate_guard_threshold),
            class_margin_override_rate_guard_window=int(args.class_margin_override_rate_guard_window),
            class_margin_override_baseline_precision=(
                None
                if args.class_margin_override_baseline_precision is None
                else float(args.class_margin_override_baseline_precision)
            ),
            class_margin_override_precision_disable_ratio=float(args.class_margin_override_precision_disable_ratio),
        ),
    )

    ts_path = runtime.export_torchscript(out_dir / "helix_ids_v1_0.torchscript.pt")
    onnx_path = runtime.export_onnx(out_dir / "helix_ids_v1_0.onnx")

    contract = {
        "input": {
            "name": "feature_vector",
            "type": "float32",
            "shape": ["batch", runtime.model.input_dim],
            "feature_order": runtime.feature_order,
            "canonical_input_dim": runtime.model.input_dim,
            "canonical_binary_classes": 2,
            "canonical_family_classes": 7,
            "schema_hash": runtime.schema_hash,
            "contract_version": runtime.contract_version,
        },
        "output": {
            "family_class": "int",
            "confidence": "float",
            "coverage_override_applied": "bool",
            "coverage_override_class": "int|null",
            "coverage_override_logit": NULL_TYPE,
            "coverage_override_threshold": NULL_TYPE,
            "class_margin_override_applied": "bool",
            "class_margin_override_second_class": "int|null",
            "class_margin_override_margin": NULL_TYPE,
            "class_margin_tau_adaptive": NULL_TYPE,
            "class_margin_adaptive_frozen": "bool",
            "class_margin_buffer_size": "int",
            "class_margin_enabled": "bool",
            "class_margin_collapse_alert": "bool",
        },
        "canonical_contract": runtime._contract_metadata(),
        "inference_controls": {
            "logits_normalization": True,
            "fixed_temperature": args.temperature,
            "prediction_floor": args.prediction_floor,
            "global_coverage_floor": bool(runtime.config.global_coverage_floor),
            "global_coverage_quantile": float(runtime.config.global_coverage_quantile),
            "class_margin_override_enabled": bool(runtime.config.class_margin_override_enabled),
            "class_margin_override_class_id": int(runtime.config.class_margin_override_class_id),
            "class_margin_override_tau": float(runtime.config.class_margin_override_tau),
            "class_margin_override_use_percentile": bool(runtime.config.class_margin_override_use_percentile),
            "class_margin_override_percentile_k": float(runtime.config.class_margin_override_percentile_k),
            "class_margin_override_hybrid_and": bool(runtime.config.class_margin_override_hybrid_and),
            "class_margin_override_buffer_size": int(runtime.config.class_margin_override_buffer_size),
            "class_margin_override_warmup_min_samples": int(runtime.config.class_margin_override_warmup_min_samples),
            "class_margin_override_rate_guard_threshold": float(runtime.config.class_margin_override_rate_guard_threshold),
            "class_margin_override_rate_guard_window": int(runtime.config.class_margin_override_rate_guard_window),
            "class_margin_override_baseline_precision": runtime.config.class_margin_override_baseline_precision,
            "class_margin_override_precision_disable_ratio": float(
                runtime.config.class_margin_override_precision_disable_ratio
            ),
            "class_margin_buffer_variance_epsilon": float(runtime.config.class_margin_buffer_variance_epsilon),
            "coverage_override_rate_alert_threshold": 0.02,
        },
        "artifacts": {
            "torchscript": str(ts_path),
            "onnx": str(onnx_path),
            "torchscript_contract": str(ts_path) + ".contract.json",
            "onnx_contract": str(onnx_path) + ".contract.json",
        },
    }
    (out_dir / "service_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    print(f"Exported TorchScript: {ts_path}")
    print(f"Exported ONNX: {onnx_path}")
    print(f"Contract: {out_dir / 'service_contract.json'}")


if __name__ == "__main__":
    main()
