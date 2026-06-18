"""Regression tests for Phase 13A-4 extracted orchestration components.

Validates that the orchestration package functions behave identically to
the original code extracted from train_helix_ids_full.py.

The extraction is a pure move — no behavioral changes are expected.
These tests verify that:
1. Dataclass constructors produce expected defaults.
2. Parse_config returns a valid ParsedConfig.
3. Run_orchestration and run_governance_pipeline are importable.
4. Key helpers are present and have the correct signature.
5. Error paths (missing keys, empty data) are handled.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from torch.utils.data import WeightedRandomSampler

from helix_ids.config.helix_full_config import TrainingConfig
from helix_ids.data.learnability_contract import compute_schema_hash
from scripts.training.orchestration import (
    GovernanceResult,
    OrchestrationResult,
    ParsedConfig,
    parse_config,
    run_governance_pipeline,
    run_orchestration,
)

# ======================================================================
# OrchestrationResult dataclass
# ======================================================================


class TestOrchestrationResult:
    def test_default_construction(self):
        """Default OrchestrationResult should have empty/null values."""
        result = OrchestrationResult()
        assert result.per_dataset_results == {}
        assert result.all_results == {}
        assert result.ab_raw_current_by_dataset == {}
        assert result.dataset_snapshot_ids == {}
        assert result.dataset_representation_snapshot_ids == {}
        assert result.training_elapsed_total == 0.0
        assert result.feature_order == []
        assert result.schema_hash == ""
        assert result.feature_signature == ""
        assert result.pretrain_elapsed == 0.0
        assert result.guard_failure is None
        assert result.governance_dataset_id == "helix_full_decoupled"
        assert result.results == {}
        assert result.determinism_state is None

    def test_partial_construction(self):
        """Construction with some fields should preserve defaults for rest."""
        result = OrchestrationResult(
            per_dataset_results={"cicids": {"macro_f1": 0.85}},
            feature_order=["feat1", "feat2"],
            pretrain_elapsed=42.0,
        )
        assert result.per_dataset_results == {"cicids": {"macro_f1": 0.85}}
        assert result.feature_order == ["feat1", "feat2"]
        assert result.pretrain_elapsed == 42.0
        assert result.schema_hash == ""

    def test_full_construction(self):
        """Construction with every field should round-trip correctly."""
        result = OrchestrationResult(
            per_dataset_results={"a": {"f1": 0.9}},
            all_results={"training_mode": "decoupled"},
            ab_raw_current_by_dataset={"b": {"metric": 1.0}},
            dataset_snapshot_ids={"c": "snap1"},
            dataset_representation_snapshot_ids={"d": "rep1"},
            training_elapsed_total=100.0,
            feature_order=["x", "y"],
            schema_hash="abc123",
            feature_signature="def456",
            pretrain_elapsed=50.0,
            guard_failure=None,
            governance_dataset_id="helix_full_decoupled_cluster_relabel_v1_k3_seed42",
            results={"n_datasets": 2},
            determinism_state={"seed": 42},
        )
        assert result.per_dataset_results == {"a": {"f1": 0.9}}
        assert result.all_results == {"training_mode": "decoupled"}
        assert result.governance_dataset_id == "helix_full_decoupled_cluster_relabel_v1_k3_seed42"
        assert result.determinism_state == {"seed": 42}


# ======================================================================
# GovernanceResult dataclass
# ======================================================================


class TestGovernanceResult:
    def test_default_construction(self):
        """Default GovernanceResult should have empty/null values."""
        result = GovernanceResult()
        assert result.governance_stages == {}
        assert result.governance_context == {}
        assert result.governance_run_record == {}
        assert result.determinism == {}
        assert result.return_payload == {}
        assert result.success is False

    def test_full_construction(self):
        """Construction with every field should round-trip correctly."""
        result = GovernanceResult(
            governance_stages={"promotion": {"passed": True}},
            governance_context={"mode": "ab"},
            governance_run_record={"id": "run001"},
            determinism={"seed": 42, "rank": 0},
            return_payload={"phase_regime": "ab_phase"},
            success=True,
        )
        assert result.governance_stages == {"promotion": {"passed": True}}
        assert result.success is True
        assert result.return_payload == {"phase_regime": "ab_phase"}


# ======================================================================
# ParsedConfig dataclass
# ======================================================================


class TestParsedConfig:
    def test_default_construction(self):
        """Default ParsedConfig with omitted optional fields."""
        import argparse

        args = argparse.Namespace()
        config = ParsedConfig(
            args=args,
            train_config=TrainingConfig(),
            config_payload={},
            phase_regime="head_isolation_ce_warmstart",
            calibration_enabled=False,
            governance_only_mode=False,
        )
        assert isinstance(config.args, argparse.Namespace)
        assert isinstance(config.train_config, TrainingConfig)
        assert config.phase_regime == "head_isolation_ce_warmstart"
        assert config.calibration_enabled is False
        assert config.governance_only_mode is False
        assert config.forced_class_balance_strategy == "focal"
        assert config.forced_cluster_relabel_k == 3
        assert config.forced_cluster_relabel_seed == 42
        assert config.forced_sampler_mode == "interleaved_rr"
        assert config.forced_num_workers == 0
        assert config.multi_seed_result is None

    def test_forced_overrides(self):
        """Forced overrides set in construction should be reflected."""
        import argparse

        args = argparse.Namespace()
        config = ParsedConfig(
            args=args,
            train_config=TrainingConfig(),
            config_payload={},
            phase_regime="ab_phase",
            calibration_enabled=True,
            governance_only_mode=False,
            forced_class_balance_strategy="sqrt_weighted_ce",
            forced_cluster_relabel_k=5,
            forced_family_margin_loss_weight=0.25,
        )
        assert config.forced_class_balance_strategy == "sqrt_weighted_ce"
        assert config.forced_cluster_relabel_k == 5
        assert config.forced_family_margin_loss_weight == 0.25

    def test_nsl_kdd_label_merges_default(self):
        """Default merges should be an empty list."""
        import argparse

        args = argparse.Namespace()
        config = ParsedConfig(
            args=args,
            train_config=TrainingConfig(),
            config_payload={},
            phase_regime="test",
            calibration_enabled=False,
            governance_only_mode=False,
        )
        assert config.forced_nsl_kdd_label_merges == []


# ======================================================================
# parse_config
# ======================================================================


class TestParseConfig:
    def test_parse_config_is_callable(self):
        """parse_config should be a callable function."""
        assert callable(parse_config)

    def test_parse_config_returns_parsed_config(self):
        """parse_config returns a ParsedConfig instance with minimal CLI args."""
        # Use only args the parser actually defines
        result = parse_config(
            [
                "--precomputed-splits-dir",
                "/tmp/splits",
                "--seed",
                "42",
                "--output",
                "/tmp/output",
                "--config",
                "config/helix_config.yaml",
                "--batch-size",
                "256",
                "--epochs",
                "5",
                "--no-ab-mode",
            ]
        )
        assert isinstance(result, ParsedConfig)
        # Check that config was populated from args
        assert result.args.seed == 42
        assert result.args.batch_size == 256
        assert result.args.epochs == 5
        assert result.train_config is not None


# ======================================================================
# Module-level constants and helpers
# ======================================================================


class TestModuleConstants:
    def test_required_geometry_feature_dim(self):
        """The required feature dim constant should be imported correctly."""
        from scripts.training.orchestration.run_orchestrator import (
            REQUIRED_GEOMETRY_FEATURE_DIM,
        )

        assert REQUIRED_GEOMETRY_FEATURE_DIM == 17

    def test_schema_hash_compute(self):
        """compute_schema_hash should produce deterministic output for same input."""
        features = ["col1", "col2", "col3"]
        hash1 = compute_schema_hash(
            feature_columns=features, transformations=["split_then_nan_to_num"]
        )
        hash2 = compute_schema_hash(
            feature_columns=features, transformations=["split_then_nan_to_num"]
        )
        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) > 0


# ======================================================================
# Lazy import helpers
# ======================================================================


class TestLazyImports:
    def test_lazy_import_helper_works(self):
        """_lazy_import should be able to import from train_helix_ids_full."""
        from scripts.training.orchestration.run_orchestrator import (
            _lazy_import,
        )

        result = _lazy_import(
            "scripts.training.train_helix_ids_full", "REQUIRED_GEOMETRY_FEATURE_DIM"
        )
        assert result == 17

    def test_atomic_write_json_is_callable(self):
        """_atomic_write_json should be a callable function."""
        from scripts.training.orchestration.run_orchestrator import (
            _atomic_write_json,
        )

        assert callable(_atomic_write_json)

    def test_persist_seed_artifacts_is_callable(self):
        """_persist_seed_artifacts should be a callable function."""
        from scripts.training.orchestration.run_orchestrator import (
            _persist_seed_artifacts,
        )

        assert callable(_persist_seed_artifacts)

    def test_calibrate_family_predictions_is_callable(self):
        """_calibrate_family_predictions should be a callable function."""
        from scripts.training.orchestration.run_orchestrator import (
            _calibrate_family_predictions,
        )

        assert callable(_calibrate_family_predictions)

    def test_emit_calibration_artifacts_is_callable(self):
        """_emit_calibration_artifacts should be a callable function."""
        from scripts.training.orchestration.run_orchestrator import (
            _emit_calibration_artifacts,
        )

        assert callable(_emit_calibration_artifacts)


# ======================================================================
# Governance pipeline module integrity
# ======================================================================


class TestGovernancePipeline:
    def test_governance_pipeline_is_callable(self):
        """run_governance_pipeline should be a callable function."""
        assert callable(run_governance_pipeline)

    def test_pipeline_rejects_empty_orchestration_result(self):
        """Empty result should raise on required-field validation."""
        from dataclasses import dataclass

        @dataclass
        class MinimalOrchResult:
            per_dataset_results: dict
            all_results: dict

        with pytest.raises((AttributeError, TypeError)):
            run_governance_pipeline(
                orchestration_result=MinimalOrchResult({}, {}),
                parsed=ParsedConfig(
                    args=type("Args", (), {})(),
                    train_config=TrainingConfig(),
                    config_payload={},
                    phase_regime="test",
                    calibration_enabled=False,
                    governance_only_mode=False,
                ),
                results_dir=Path("/tmp"),
                logger=print,
            )


# ======================================================================
# run_orchestration function interface
# ======================================================================


class TestRunOrchestration:
    def test_run_orchestration_is_callable(self):
        """run_orchestration should be a callable function."""
        assert callable(run_orchestration)


# ======================================================================
# Edge cases and error paths
# ======================================================================


class TestErrorPaths:
    def test_run_orchestration_raises_on_empty_splits(self):
        """Missing or wrong-dim feature_columns should raise RuntimeError."""
        with pytest.raises(RuntimeError, match="canonical_feature_dim_not_17"):
            run_orchestration(
                parsed=ParsedConfig(
                    args=type(
                        "Args",
                        (),
                        {
                            "seed": 42,
                            "precomputed_splits_dir": "/tmp/splits",
                            "eval_batch_size": 256,
                        },
                    )(),
                    train_config=TrainingConfig(batch_size=256),
                    config_payload={},
                    phase_regime="test",
                    calibration_enabled=False,
                    governance_only_mode=False,
                ),
                splits={"feature_columns": ["a", "b"]},  # wrong dim (2 not 17)
                results_dir=Path("/tmp"),
                output_dir=Path("/tmp"),
                logger=print,
            )


# ======================================================================
# DataLoader and sampler construction validation
# ======================================================================


class TestSamplerConstruction:
    def test_weighted_random_sampler_weights(self):
        """WeightedRandomSampler can be constructed with numpy weights list."""
        weights_np = np.ones(100, dtype=np.float64)
        sampler = WeightedRandomSampler(
            weights=weights_np.tolist(),
            num_samples=50,
            replacement=True,
        )
        indices = list(sampler)
        assert len(indices) == 50
        assert all(0 <= i < 100 for i in indices)


# ======================================================================
# Orchestration module __all__ exports
# ======================================================================


class TestModuleExports:
    def test_all_exports_present(self):
        """The public API exports should match __all__."""
        from scripts.training.orchestration import __all__ as all_exports

        expected = {"parse_config", "run_orchestration", "run_governance_pipeline"}
        assert expected.issubset(set(all_exports))
