#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from helix_ids.operations.inference_runtime import HelixInferenceRuntime, InferenceConfig

try:
    import uvicorn
    from fastapi import FastAPI, Response
    from pydantic import BaseModel, Field
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "FastAPI/uvicorn required for REST service. Install with: pip install fastapi uvicorn"
    ) from exc


class PredictRequest(BaseModel):
    features: list[float] | list[list[float]] = Field(..., description="Input feature vector or batch")


class PredictResponse(BaseModel):
    family_class: int | list[int]
    confidence: float | list[float]
    coverage_override_applied: bool
    coverage_override_class: int | None
    coverage_override_logit: float | None
    coverage_override_threshold: float | None
    class_margin_override_applied: bool
    class_margin_override_second_class: int | None
    class_margin_override_margin: float | None
    class_margin_tau_adaptive: float | None
    class_margin_adaptive_frozen: bool
    class_margin_buffer_size: int
    class_margin_enabled: bool
    class_margin_collapse_alert: bool


def _format_prometheus_metrics(
    requests_total: int,
    override_total: int,
    *,
    degraded: bool,
    class_distribution: dict[int, int] | None = None,
    entropy: float | None = None,
) -> str:
    rate = float(override_total) / float(requests_total) if requests_total > 0 else 0.0
    lines = [
        "# HELP helix_requests_total Total inference requests",
        "# TYPE helix_requests_total counter",
        f"helix_requests_total {requests_total}",
        "# HELP helix_coverage_override_total Total override activations",
        "# TYPE helix_coverage_override_total counter",
        f"helix_coverage_override_total {override_total}",
        "# HELP helix_coverage_override_rate Override rate",
        "# TYPE helix_coverage_override_rate gauge",
        f"helix_coverage_override_rate {rate:.10f}",
        "# HELP helix_degraded_state 1 when degraded (override_rate > 0.02)",
        "# TYPE helix_degraded_state gauge",
        f"helix_degraded_state {1 if degraded else 0}",
    ]

    if class_distribution:
        lines.append("# HELP helix_class_predictions_total Predictions count by class")
        lines.append("# TYPE helix_class_predictions_total counter")
        for cls in sorted(class_distribution.keys()):
            lines.append(f'helix_class_predictions_total{{class="{int(cls)}"}} {int(class_distribution[cls])}')

    if entropy is not None:
        ent = 0.0 if abs(float(entropy)) < 5e-13 else float(entropy)
        lines.append("# HELP helix_class_entropy Class distribution entropy")
        lines.append("# TYPE helix_class_entropy gauge")
        lines.append(f"helix_class_entropy {ent:.10f}")

    return "\n".join(lines) + "\n"


def create_app(runtime: HelixInferenceRuntime) -> FastAPI:
    app = FastAPI(title="HELIX IDS Baseline API", version="1.0")

    metrics = {
        "helix_requests_total": 0,
        "helix_coverage_override_total": 0,
    }
    class_counts: dict[int, int] = {}
    state_lock = threading.Lock()

    capture_path = Path("artifacts/operations/live_events.jsonl")
    capture_path.parent.mkdir(parents=True, exist_ok=True)

    def _rate() -> float:
        req = int(metrics["helix_requests_total"])
        ov = int(metrics["helix_coverage_override_total"])
        return float(ov) / float(req) if req > 0 else 0.0

    def _degraded() -> bool:
        return _rate() > 0.02

    def _entropy_from_counts() -> float | None:
        total = int(metrics["helix_requests_total"])
        if total <= 0 or not class_counts:
            return None
        probs = np.asarray([v for _, v in sorted(class_counts.items())], dtype=np.float64)
        probs = probs / max(1e-12, float(probs.sum()))
        p = np.clip(probs, 1e-12, 1.0)
        return float(-np.sum(p * np.log(p)))

    def _append_event(*, features: list[float] | list[list[float]], result: dict[str, Any]) -> None:
        event = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "inputs": features,
            "prediction": {
                "family_class": result.get("family_class"),
                "confidence": result.get("confidence"),
            },
            "override": {
                "applied": bool(result.get("coverage_override_applied", False)),
                "class": result.get("coverage_override_class"),
                "logit": result.get("coverage_override_logit"),
                "threshold": result.get("coverage_override_threshold"),
            },
            "class_margin_override": {
                "applied": bool(result.get("class_margin_override_applied", False)),
                "second_class": result.get("class_margin_override_second_class"),
                "margin": result.get("class_margin_override_margin"),
                "tau_adaptive": result.get("class_margin_tau_adaptive"),
                "adaptive_frozen": bool(result.get("class_margin_adaptive_frozen", False)),
                "buffer_size": int(result.get("class_margin_buffer_size", 0)),
                "enabled": bool(result.get("class_margin_enabled", True)),
                "collapse_alert": bool(result.get("class_margin_collapse_alert", False)),
            },
        }
        with capture_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "schema_hash": runtime.schema_hash, "contract_version": runtime.contract_version}

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> dict[str, Any]:
        feats = np.asarray(req.features, dtype=np.float32)
        result = runtime.predict(feats)

        with state_lock:
            metrics["helix_requests_total"] += 1
            if bool(result.get("coverage_override_applied", False)):
                metrics["helix_coverage_override_total"] += 1

            family_class = result.get("family_class")
            if isinstance(family_class, list):
                for c in family_class:
                    ci = int(c)
                    class_counts[ci] = int(class_counts.get(ci, 0)) + 1
            else:
                ci = int(family_class)
                class_counts[ci] = int(class_counts.get(ci, 0)) + 1

            result["degraded"] = _degraded()
            result["override_rate"] = _rate()

        _append_event(features=req.features, result=result)

        return {
            "family_class": result["family_class"],
            "confidence": result["confidence"],
            "coverage_override_applied": bool(result.get("coverage_override_applied", False)),
            "coverage_override_class": result.get("coverage_override_class"),
            "coverage_override_logit": result.get("coverage_override_logit"),
            "coverage_override_threshold": result.get("coverage_override_threshold"),
            "class_margin_override_applied": bool(result.get("class_margin_override_applied", False)),
            "class_margin_override_second_class": result.get("class_margin_override_second_class"),
            "class_margin_override_margin": result.get("class_margin_override_margin"),
            "class_margin_tau_adaptive": result.get("class_margin_tau_adaptive"),
            "class_margin_adaptive_frozen": bool(result.get("class_margin_adaptive_frozen", False)),
            "class_margin_buffer_size": int(result.get("class_margin_buffer_size", 0)),
            "class_margin_enabled": bool(result.get("class_margin_enabled", True)),
            "class_margin_collapse_alert": bool(result.get("class_margin_collapse_alert", False)),
        }

    @app.get("/metrics")
    def metrics_endpoint() -> Response:
        with state_lock:
            payload = _format_prometheus_metrics(
                requests_total=int(metrics["helix_requests_total"]),
                override_total=int(metrics["helix_coverage_override_total"]),
                degraded=_degraded(),
                class_distribution=dict(class_counts),
                entropy=_entropy_from_counts(),
            )
        return Response(content=payload, media_type="text/plain")

    return app


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serve HELIX IDS baseline via REST")
    p.add_argument("--checkpoint", default="models/helix_full/helix_full_nsl_kdd_best.pt")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
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
    p.add_argument(
        "--class-margin-override-freeze-adaptive-tau",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument("--class-margin-override-frozen-tau-adaptive", type=float, default=None)
    p.add_argument(
        "--class-margin-override-use-margin-zscore",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument("--class-margin-override-margin-zscore-epsilon", type=float, default=1e-9)
    p.add_argument("--device", default="cpu")
    p.add_argument("--enable-global-coverage-floor", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--global-coverage-quantile", type=float, default=0.95)
    return p.parse_args()


def main() -> None:
    args = parse_args()
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
            class_margin_override_freeze_adaptive_tau=bool(args.class_margin_override_freeze_adaptive_tau),
            class_margin_override_frozen_tau_adaptive=(
                None
                if args.class_margin_override_frozen_tau_adaptive is None
                else float(args.class_margin_override_frozen_tau_adaptive)
            ),
            class_margin_override_use_margin_zscore=bool(args.class_margin_override_use_margin_zscore),
            class_margin_override_margin_zscore_epsilon=float(args.class_margin_override_margin_zscore_epsilon),
        ),
    )
    app = create_app(runtime)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
