#!/usr/bin/env python3
"""Fixed E2E Benchmark for HELIX-IDS v2."""

import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

from helix_ids.governance.determinism import set_global_determinism
from helix_ids.governance.entrypoint import governed_entrypoint
from helix_ids.governance.parameters import DEFAULT_GOVERNANCE_POLICY
from helix_ids.governance.promotion import SeedRunSummary, aggregate_seed_runs
from helix_ids.governance.run_registry import RunRegistry
from helix_ids.utils.metrics import evaluate as evaluate_contract

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

RESULTS_DIR = PROJECT_ROOT / "results" / "v2_fixed"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
if torch.cuda.is_available():
    device_str = "cuda"
elif torch.backends.mps.is_available():
    device_str = "mps"
else:
    device_str = "cpu"
DEVICE = torch.device(device_str)
CLASS_NAMES = ["Normal", "DoS", "Probe", "R2L", "U2R"]


def load_model_and_data(platform="production"):
    from scripts.training.train_multidataset import HELIXMLP5Class, SafeDataLoader

    model_dir = PROJECT_ROOT / "models" / "v2_fixed" / platform
    with open(model_dir / "model_card_v2.json") as f:
        card = json.load(f)
    for sidecar_path in [
        model_dir / "model_v2.pt.contract.json",
        model_dir / "model_v2.pt.feature_order.json",
        model_dir / "model_v2.pt.schema_hash.txt",
    ]:
        if not sidecar_path.exists():
            raise RuntimeError(f"Missing benchmark provenance sidecar: {sidecar_path}")
    hidden_dims = eval(card["architecture"])
    n_features = card["n_features"]
    model = HELIXMLP5Class(
        input_dim=n_features, hidden_dims=hidden_dims, num_classes=5, dropout=0.0
    )
    model.load_state_dict(torch.load(model_dir / "model_v2.pt", map_location=DEVICE, weights_only=True))
    model.eval()
    with open(model_dir / "nsl_scaler.pkl", "rb") as f:
        nsl_scaler = pickle.load(f)
    with open(model_dir / "unsw_scaler.pkl", "rb") as f:
        unsw_scaler = pickle.load(f)
    loader = SafeDataLoader()
    x_nsl, y_nsl = loader.load_nsl_kdd(PROJECT_ROOT / "data" / "nsl_kdd" / "test.csv")
    x_unsw, y_unsw = loader.load_unsw_nb15(PROJECT_ROOT / "data" / "unsw_nb15" / "test.csv")
    x_nsl = nsl_scaler.transform(np.nan_to_num(x_nsl[:, :n_features], 0))
    x_unsw = unsw_scaler.transform(np.nan_to_num(x_unsw[:, :n_features], 0))
    return model, x_nsl, y_nsl, x_unsw, y_unsw


def benchmark_latency(model, X, n_runs=50):
    x_t = torch.FloatTensor(X[:500]).to(DEVICE)
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            model(x_t)
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            s = time.perf_counter()
            model(x_t)
            times.append((time.perf_counter() - s) * 1000)
    return {
        "mean_ms": float(np.mean(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "throughput": float(1000 / np.mean(times) * len(x_t)),
    }


def evaluate(model, X, y, name):
    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(X).to(DEVICE))
        preds = logits.argmax(dim=1).cpu().numpy()
    metrics = evaluate_contract(
        preds=preds, targets=np.asarray(y), dataset_id=name, class_names=CLASS_NAMES
    )
    logger.info(f"\n{name}: Acc={metrics.accuracy:.4f} F1-macro={metrics.macro_f1:.4f}")
    for class_name, class_f1 in metrics.per_class_f1.items():
        logger.info(f"  {class_name}: {class_f1:.4f}")
    return metrics.to_dict()


def _benchmark_platform(platform: str):
    try:
        model, xn, yn, xu, yu = load_model_and_data(platform)
        return {
            "nsl_kdd": evaluate(model, xn, yn, f"{platform}/NSL"),
            "unsw_nb15": evaluate(model, xu, yu, f"{platform}/UNSW"),
            "latency": benchmark_latency(model, xn),
        }
    except Exception as exc:
        logger.warning(f"Skip {platform}: {exc}")
        return None


def _run_platform_benchmarks(platforms: list[str]) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for platform in platforms:
        platform_results = _benchmark_platform(platform)
        if platform_results is not None:
            results[platform] = platform_results
    return results


def _collect_governance_metrics(results: dict[str, dict]) -> tuple[list[float], list[float], list[float], list[float]]:
    ci_widths = []
    ci_lowers = []
    cross_dataset_drifts = []
    macro_values = []
    for platform_results in results.values():
        nsl = platform_results.get("nsl_kdd", {})
        unsw = platform_results.get("unsw_nb15", {})
        if isinstance(nsl, dict) and "ci95_width" in nsl:
            ci_widths.append(float(nsl["ci95_width"]))
            ci_lowers.append(float(nsl.get("ci95_lower", nsl.get("macro_f1", 0.0))))
            macro_values.append(float(nsl.get("macro_f1", 0.0)))
        if isinstance(unsw, dict) and "ci95_width" in unsw:
            ci_widths.append(float(unsw["ci95_width"]))
            ci_lowers.append(float(unsw.get("ci95_lower", unsw.get("macro_f1", 0.0))))
            macro_values.append(float(unsw.get("macro_f1", 0.0)))
        if isinstance(nsl, dict) and isinstance(unsw, dict):
            if "macro_f1" in nsl and "macro_f1" in unsw:
                cross_dataset_drifts.append(abs(float(nsl["macro_f1"]) - float(unsw["macro_f1"])))

    return ci_widths, ci_lowers, cross_dataset_drifts, macro_values


def _build_governance_summary(results: dict[str, dict], seed: int, posteval_start: float) -> dict:
    ci_widths, ci_lowers, cross_dataset_drifts, macro_values = _collect_governance_metrics(results)

    aggregate_macro_f1 = float(np.mean(macro_values)) if macro_values else 0.0
    policy = DEFAULT_GOVERNANCE_POLICY
    registry = RunRegistry(
        Path(os.environ.get("HELIX_RUN_REGISTRY", "results/gates/run_registry.jsonl"))
    )
    drift, z_score = registry.compute_drift(
        dataset_id="benchmark_e2e",
        current_macro_f1=aggregate_macro_f1,
        baseline_window_runs=20,
    )
    min_ci_lower = min(ci_lowers) if ci_lowers else 0.0
    max_ci_width = max(ci_widths) if ci_widths else 0.0
    tier2_pass = (
        min_ci_lower >= policy.bootstrap.min_ci95_lower_bound
        and max_ci_width <= policy.bootstrap.max_ci_width
        and drift <= policy.drift.max_abs_macro_f1_drift
        and z_score <= policy.drift.max_abs_z_score
    )
    promotion_consensus = aggregate_seed_runs(
        [
            SeedRunSummary(
                seed=seed,
                macro_f1=aggregate_macro_f1,
                macro_f1_ci_lower=min_ci_lower,
                macro_f1_ci_width=max_ci_width,
                tier2_pass=tier2_pass,
            )
        ],
        min_seed_runs=policy.promotion.min_seed_runs,
        max_inter_seed_macro_f1_variance=policy.promotion.max_inter_seed_macro_f1_variance,
        reproducibility_tolerance=policy.promotion.reproducibility_tolerance,
        min_ci95_lower_bound=policy.bootstrap.min_ci95_lower_bound,
        max_ci_width=policy.bootstrap.max_ci_width,
    )
    prepromote_elapsed = max(0.001, time.perf_counter() - posteval_start)

    governance_stages = {
        "posteval": {
            "posteval_elapsed_seconds": max(0.001, time.perf_counter() - posteval_start),
            "macro_f1_ci_width": max_ci_width,
            "macro_f1_ci_lower": min_ci_lower,
            "abs_macro_f1_drift": max(max(cross_dataset_drifts) if cross_dataset_drifts else 0.0, drift),
            "abs_macro_f1_zscore": z_score,
        },
        "prepromote": {
            "prepromote_elapsed_seconds": prepromote_elapsed,
            "macro_f1_ci_width": max_ci_width,
            "macro_f1_ci_lower": min_ci_lower,
            **promotion_consensus.to_stage_metrics(),
        },
    }
    if promotion_consensus.invalid_reason is not None:
        governance_stages["prepromote"]["promotion_invalid_reason"] = promotion_consensus.invalid_reason

    return {
        "governance_stages": governance_stages,
        "governance_run_record": {
            "dataset_id": "benchmark_e2e",
            "macro_f1": aggregate_macro_f1,
            "fingerprint": os.environ.get("HELIX_FINGERPRINT"),
            "parent_run_id": os.environ.get("HELIX_PARENT_RUN_ID"),
            "lineage": {
                "dataset_hashes": os.environ.get("HELIX_DATASET_HASHES", "unknown"),
                "schema_hash": os.environ.get("HELIX_SCHEMA_HASH", "unknown"),
                "mapping_version": os.environ.get("HELIX_MAPPING_VERSION", "unknown"),
                "model_artifact": str(PROJECT_ROOT / "models" / "v2_fixed"),
                "metrics_artifact": str(RESULTS_DIR / "e2e_benchmark_v2.json"),
            },
        },
    }


@governed_entrypoint(entrypoint_id="scripts.benchmark_e2e")
def main():
    seed = int(os.environ.get("HELIX_SEED", "42"))
    os.environ["HELIX_SEED"] = str(seed)
    determinism_state = set_global_determinism(seed)

    logger.info("HELIX-IDS E2E Benchmark v2")
    results = _run_platform_benchmarks(["production", "rpi4", "rpi_zero", "esp32"])
    posteval_start = time.perf_counter()
    with open(RESULTS_DIR / "e2e_benchmark_v2.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Done — saved to results/v2_fixed/e2e_benchmark_v2.json")
    governance_summary = _build_governance_summary(results, seed, posteval_start)
    return {
        "results": results,
        **governance_summary,
        "governance_context": {
            "seed": seed,
        },
        "determinism": determinism_state.to_dict(),
    }


if __name__ == "__main__":
    main()
