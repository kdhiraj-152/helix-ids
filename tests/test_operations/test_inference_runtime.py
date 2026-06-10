from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from helix_ids.contracts.schema_contract import runtime_contract_payload
from helix_ids.models.full import HelixFullConfig, create_helix_full
from helix_ids.operations.inference_runtime import HelixInferenceRuntime, InferenceConfig


def _make_checkpoint(path: Path) -> None:
    model = create_helix_full(HelixFullConfig(input_dim=17, family_output_dim=7))
    payload = {
        "model_state_dict": model.state_dict(),
        **runtime_contract_payload(),
    }
    from helix_ids.governance import (
        ARTIFACT_MANIFEST_KEY,
        checkpoint_manifest_payload,
        write_contract_sidecars,
    )
    from helix_ids.utils.export import build_export_manifest, finalize_export_artifact
    manifest_base = build_export_manifest(
        contract=runtime_contract_payload(),
        model_architecture=model.__class__.__name__,
        export_config={"format": "checkpoint", "origin": "test_inference_runtime"},
    )
    payload[ARTIFACT_MANIFEST_KEY] = checkpoint_manifest_payload(manifest_base)
    torch.save(payload, path)
    # Write sidecar files expected by HelixInferenceRuntime
    sidecars = write_contract_sidecars(path, runtime_contract_payload())
    finalize_export_artifact(path, manifest_base, sidecars=sidecars)


def test_predict_outputs_contract(tmp_path: Path) -> None:
    ckpt = tmp_path / "model.pt"
    _make_checkpoint(ckpt)

    runtime = HelixInferenceRuntime(
        ckpt,
        config=InferenceConfig(fixed_temperature=1.0, prediction_floor=1e-6),
    )
    x = np.random.default_rng(0).normal(size=(8, 17)).astype(np.float32)

    out = runtime.predict(x)
    assert "family_class" in out and "confidence" in out
    assert "coverage_override_applied" in out
    assert "coverage_override_class" in out
    assert "coverage_override_logit" in out
    assert "coverage_override_threshold" in out
    assert "class_margin_override_applied" in out
    assert "class_margin_override_second_class" in out
    assert "class_margin_override_margin" in out
    assert "class_margin_tau_adaptive" in out
    assert "class_margin_adaptive_frozen" in out
    assert "class_margin_buffer_size" in out
    assert "class_margin_enabled" in out
    assert "class_margin_collapse_alert" in out
    assert isinstance(out["family_class"], list)
    assert isinstance(out["confidence"], list)
    assert len(out["family_class"]) == 8
    assert len(out["confidence"]) == 8
    assert all(0.0 <= c <= 1.0 for c in out["confidence"])


def test_predict_is_deterministic(tmp_path: Path) -> None:
    ckpt = tmp_path / "model.pt"
    _make_checkpoint(ckpt)
    runtime = HelixInferenceRuntime(ckpt)
    x = np.random.default_rng(1).normal(size=(4, 17)).astype(np.float32)

    out1 = runtime.predict(x)
    out2 = runtime.predict(x)
    assert out1["family_class"] == out2["family_class"]
    assert np.allclose(np.asarray(out1["confidence"]), np.asarray(out2["confidence"]))


class _DummyModel(torch.nn.Module):
    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self._logits = logits
        self.input_dim = logits.shape[1]

    def forward(self, x):
        n = x.shape[0]
        fam = self._logits[:n].to(x.device)
        bin_logits = torch.zeros((n, 2), dtype=fam.dtype, device=x.device)
        return bin_logits, fam


def test_global_coverage_floor_has_no_batch_coupling(tmp_path: Path) -> None:
    ckpt = tmp_path / "model.pt"
    _make_checkpoint(ckpt)
    runtime = HelixInferenceRuntime(
        ckpt,
        config=InferenceConfig(
            fixed_temperature=1.0,
            prediction_floor=1e-6,
            global_coverage_floor=True,
            global_coverage_quantile=0.95,
        ),
    )

    # Crafted logits where prior batch-coupled implementation would override in batch mode
    # but not in single-sample mode. New implementation must be per-sample consistent.
    logits = torch.tensor(
        [
            [4.0, 1.0, 0.5, 0.2, 3.9, 0.0, 0.0],
            [4.2, 1.0, 0.4, 0.1, 3.8, 0.0, 0.0],
            [4.3, 1.0, 0.3, 0.1, 3.7, 0.0, 0.0],
            [4.5, 1.0, 0.3, 0.1, 4.49, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    runtime.model = _DummyModel(logits)

    x = np.zeros((4, 7), dtype=np.float32)

    batch_out = runtime.predict(x, active_classes=[0, 4], enforce_global_coverage=True)
    single_outs = [
        runtime.predict(x[i], active_classes=[0, 4], enforce_global_coverage=True)
        for i in range(x.shape[0])
    ]

    single_classes = [int(o["family_class"]) for o in single_outs]
    single_any_override = any(bool(o["coverage_override_applied"]) for o in single_outs)

    assert batch_out["family_class"] == single_classes
    assert bool(batch_out["coverage_override_applied"]) is bool(single_any_override)


def test_class_margin_override_switches_target_class_when_margin_below_tau(tmp_path: Path) -> None:
    ckpt = tmp_path / "model.pt"
    _make_checkpoint(ckpt)
    runtime = HelixInferenceRuntime(
        ckpt,
        config=InferenceConfig(
            fixed_temperature=1.0,
            prediction_floor=1e-6,
            class_margin_override_enabled=True,
            class_margin_override_class_id=4,
            class_margin_override_tau=50_000.0,
            class_margin_override_use_percentile=False,
        ),
    )

    # Row 0: class 4 wins by margin 40_000 (< tau) => switch to class 2
    # Row 1: class 4 wins by margin 60_000 (>= tau) => keep class 4
    logits = torch.tensor(
        [
            [0.0, -1.0, 20_000.0, 0.0, 60_000.0, -2.0, -3.0],
            [0.0, -1.0, 20_000.0, 0.0, 80_000.0, -2.0, -3.0],
        ],
        dtype=torch.float32,
    )
    runtime.model = _DummyModel(logits)

    x = np.zeros((2, 7), dtype=np.float32)
    out = runtime.predict(x, enforce_global_coverage=False)

    assert out["family_class"] == [2, 4]
    assert bool(out["class_margin_override_applied"]) is True
    assert int(out["class_margin_override_second_class"]) == 2


def test_class_margin_override_disabled_keeps_argmax(tmp_path: Path) -> None:
    ckpt = tmp_path / "model.pt"
    _make_checkpoint(ckpt)
    runtime = HelixInferenceRuntime(
        ckpt,
        config=InferenceConfig(
            fixed_temperature=1.0,
            prediction_floor=1e-6,
            class_margin_override_enabled=False,
            class_margin_override_class_id=4,
            class_margin_override_tau=50_000.0,
        ),
    )

    logits = torch.tensor(
        [[0.0, -1.0, 20_000.0, 0.0, 60_000.0, -2.0, -3.0]],
        dtype=torch.float32,
    )
    runtime.model = _DummyModel(logits)

    x = np.zeros((1, 7), dtype=np.float32)
    out = runtime.predict(x, enforce_global_coverage=False)
    assert int(out["family_class"]) == 4
    assert bool(out["class_margin_override_applied"]) is False


def test_class_margin_override_applies_after_global_coverage(tmp_path: Path) -> None:
    ckpt = tmp_path / "model.pt"
    _make_checkpoint(ckpt)
    runtime = HelixInferenceRuntime(
        ckpt,
        config=InferenceConfig(
            fixed_temperature=1.0,
            prediction_floor=1e-6,
            global_coverage_floor=True,
            global_coverage_quantile=0.95,
            class_margin_override_enabled=True,
            class_margin_override_class_id=4,
            class_margin_override_tau=50_000.0,
            class_margin_override_use_percentile=False,
        ),
    )

    # Coverage can force class 4 in some regimes, but class-margin override must be final.
    # Here class 4 top-1 margin is below tau, so final prediction must switch to class 2.
    logits = torch.tensor(
        [[0.0, -1.0, 20_000.0, 0.0, 60_000.0, -2.0, -3.0]],
        dtype=torch.float32,
    )
    runtime.model = _DummyModel(logits)

    x = np.zeros((1, 7), dtype=np.float32)
    out = runtime.predict(x, active_classes=[0, 2, 4], enforce_global_coverage=True)
    assert int(out["family_class"]) == 2
    assert bool(out["class_margin_override_applied"]) is True


def test_class_margin_override_percentile_mode_matches_fixed_tau_at_equivalent_cutoff(tmp_path: Path) -> None:
    ckpt = tmp_path / "model.pt"
    _make_checkpoint(ckpt)

    # Build a batch where argmax is class 4 with margins [10, 20, 30, 40, 50].
    # Warmup is pre-seeded so percentile k=60 -> tau at 34, and hybrid AND with tau_fixed=34 switches {10,20,30}.
    margins = [10.0, 20.0, 30.0, 40.0, 50.0]
    logits_rows = []
    for m in margins:
        logits_rows.append([0.0, -1.0, 100.0 - m, 0.0, 100.0, -2.0, -3.0])
    logits = torch.tensor(logits_rows, dtype=torch.float32)
    x = np.zeros((len(margins), 7), dtype=np.float32)

    runtime = HelixInferenceRuntime(
        ckpt,
        config=InferenceConfig(
            fixed_temperature=1.0,
            prediction_floor=1e-6,
            class_margin_override_enabled=True,
            class_margin_override_class_id=4,
            class_margin_override_tau=34.0,
            class_margin_override_use_percentile=True,
            class_margin_override_percentile_k=60.0,
            class_margin_override_hybrid_and=True,
            class_margin_override_buffer_size=1000,
            class_margin_override_warmup_min_samples=5,
        ),
    )
    runtime.model = _DummyModel(logits)
    for m in margins:
        runtime._class_margin_buffer.append(float(m))

    out = runtime.predict(x, enforce_global_coverage=False)
    assert out["family_class"] == [2, 2, 2, 4, 4]


def test_class_margin_override_warmup_blocks_until_min_samples(tmp_path: Path) -> None:
    ckpt = tmp_path / "model.pt"
    _make_checkpoint(ckpt)

    logits = torch.tensor(
        [[0.0, -1.0, 90.0, 0.0, 100.0, -2.0, -3.0]],
        dtype=torch.float32,
    )
    x = np.zeros((1, 7), dtype=np.float32)

    runtime = HelixInferenceRuntime(
        ckpt,
        config=InferenceConfig(
            fixed_temperature=1.0,
            prediction_floor=1e-6,
            class_margin_override_enabled=True,
            class_margin_override_class_id=4,
            class_margin_override_tau=70_000.0,
            class_margin_override_use_percentile=True,
            class_margin_override_percentile_k=75.0,
            class_margin_override_hybrid_and=True,
            class_margin_override_warmup_min_samples=200,
        ),
    )
    runtime.model = _DummyModel(logits)

    out = runtime.predict(x, enforce_global_coverage=False)
    # Warmup not satisfied -> no override despite fixed tau condition.
    assert int(out["family_class"]) == 4
    assert bool(out["class_margin_override_applied"]) is False
    assert out["class_margin_tau_adaptive"] is None


def test_class_margin_override_percentile_mode_singleton_target_applies_at_boundary(tmp_path: Path) -> None:
    ckpt = tmp_path / "model.pt"
    _make_checkpoint(ckpt)

    # Single sample where class 4 is top-1 with margin 10.
    # Seed warmup buffer so percentile tau is available and boundary condition applies.
    logits = torch.tensor(
        [[0.0, -1.0, 90.0, 0.0, 100.0, -2.0, -3.0]],
        dtype=torch.float32,
    )
    x = np.zeros((1, 7), dtype=np.float32)

    runtime_pct = HelixInferenceRuntime(
        ckpt,
        config=InferenceConfig(
            fixed_temperature=1.0,
            prediction_floor=1e-6,
            class_margin_override_enabled=True,
            class_margin_override_class_id=4,
            class_margin_override_tau=11.0,
            class_margin_override_use_percentile=True,
            class_margin_override_percentile_k=60.0,
            class_margin_override_hybrid_and=True,
            class_margin_override_warmup_min_samples=1,
        ),
    )
    runtime_pct.model = _DummyModel(logits)
    runtime_pct._class_margin_buffer.append(11.0)

    out_pct = runtime_pct.predict(x, enforce_global_coverage=False)
    assert int(out_pct["family_class"]) == 2
    assert bool(out_pct["class_margin_override_applied"]) is True


def test_class_margin_override_rate_guard_freezes_adaptive_and_falls_back_to_fixed(tmp_path: Path) -> None:
    ckpt = tmp_path / "model.pt"
    _make_checkpoint(ckpt)

    logits = torch.tensor(
        [[0.0, -1.0, 90.0, 0.0, 100.0, -2.0, -3.0]],
        dtype=torch.float32,
    )
    x = np.zeros((1, 7), dtype=np.float32)

    runtime = HelixInferenceRuntime(
        ckpt,
        config=InferenceConfig(
            fixed_temperature=1.0,
            prediction_floor=1e-6,
            class_margin_override_enabled=True,
            class_margin_override_class_id=4,
            class_margin_override_tau=21.0,
            class_margin_override_use_percentile=True,
            class_margin_override_percentile_k=75.0,
            class_margin_override_hybrid_and=True,
            class_margin_override_warmup_min_samples=1,
            class_margin_override_rate_guard_threshold=0.5,
            class_margin_override_rate_guard_window=2,
        ),
    )
    runtime.model = _DummyModel(logits)
    runtime._class_margin_buffer.append(11.0)

    out1 = runtime.predict(x, enforce_global_coverage=False)
    assert bool(out1["class_margin_override_applied"]) is True
    assert bool(out1["class_margin_adaptive_frozen"]) is True

    # Clear adaptive evidence; frozen adaptive must still allow fixed-tau fallback.
    runtime._class_margin_buffer.clear()
    out2 = runtime.predict(x, enforce_global_coverage=False)
    assert bool(out2["class_margin_adaptive_frozen"]) is True
    assert bool(out2["class_margin_override_applied"]) is True


def test_class_margin_override_collapse_alert_when_buffer_variance_zero(tmp_path: Path) -> None:
    ckpt = tmp_path / "model.pt"
    _make_checkpoint(ckpt)

    logits = torch.tensor(
        [[0.0, -1.0, 90.0, 0.0, 100.0, -2.0, -3.0]],
        dtype=torch.float32,
    )
    x = np.zeros((1, 7), dtype=np.float32)

    runtime = HelixInferenceRuntime(
        ckpt,
        config=InferenceConfig(
            fixed_temperature=1.0,
            prediction_floor=1e-6,
            class_margin_override_enabled=True,
            class_margin_override_class_id=4,
            class_margin_override_tau=0.0,
            class_margin_override_use_percentile=True,
            class_margin_override_percentile_k=75.0,
            class_margin_override_hybrid_and=True,
            class_margin_override_warmup_min_samples=1,
            class_margin_buffer_variance_epsilon=1e-12,
        ),
    )
    runtime.model = _DummyModel(logits)
    runtime._class_margin_buffer.append(10.0)

    out = runtime.predict(x, enforce_global_coverage=False)
    assert bool(out["class_margin_collapse_alert"]) is True


def test_precision_guard_uses_only_labeled_pred4_counts(tmp_path: Path) -> None:
    ckpt = tmp_path / "model.pt"
    _make_checkpoint(ckpt)

    logits = torch.tensor(
        [[0.0, -1.0, 90.0, 0.0, 100.0, -2.0, -3.0]],
        dtype=torch.float32,
    )
    x = np.zeros((1, 7), dtype=np.float32)

    runtime = HelixInferenceRuntime(
        ckpt,
        config=InferenceConfig(
            fixed_temperature=1.0,
            prediction_floor=1e-6,
            class_margin_override_enabled=True,
            class_margin_override_class_id=4,
            class_margin_override_tau=70_000.0,
            class_margin_override_use_percentile=False,
            class_margin_override_baseline_precision=0.109,
            class_margin_override_precision_disable_ratio=0.8,
            class_margin_override_rate_guard_threshold=1.1,
        ),
    )
    runtime.model = _DummyModel(logits)

    # Unlabeled traffic: should not influence precision denominator used by disable guard.
    for _ in range(20):
        runtime.predict(x, enforce_global_coverage=False)

    # One labeled true-positive for class 4 keeps precision guard healthy.
    out = runtime.predict(x, enforce_global_coverage=False, labels=np.asarray([4], dtype=np.int64))
    assert bool(out["class_margin_enabled"]) is True


def test_precision_guard_does_not_disable_when_labeled_pred4_is_zero(tmp_path: Path) -> None:
    ckpt = tmp_path / "model.pt"
    _make_checkpoint(ckpt)

    # Model predicts class 2, so labeled_pred4==0 in guard window.
    logits = torch.tensor(
        [[0.0, -1.0, 100.0, 0.0, 90.0, -2.0, -3.0]],
        dtype=torch.float32,
    )
    x = np.zeros((1, 7), dtype=np.float32)

    runtime = HelixInferenceRuntime(
        ckpt,
        config=InferenceConfig(
            fixed_temperature=1.0,
            prediction_floor=1e-6,
            class_margin_override_enabled=True,
            class_margin_override_class_id=4,
            class_margin_override_tau=70_000.0,
            class_margin_override_use_percentile=False,
            class_margin_override_baseline_precision=0.109,
            class_margin_override_precision_disable_ratio=0.8,
            class_margin_override_rate_guard_threshold=1.1,
        ),
    )
    runtime.model = _DummyModel(logits)

    out = runtime.predict(x, enforce_global_coverage=False, labels=np.asarray([4], dtype=np.int64))
    assert bool(out["class_margin_enabled"]) is True
