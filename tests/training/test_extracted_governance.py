"""Regression tests for Phase 12B-7 extracted governance components.

Validates that the governance package functions behave identically to
the original code extracted from train_helix_ids_full.py.

The extraction is a pure move — no behavioral changes are expected.
These tests verify that:
1. Pure functions produce identical outputs for identical inputs.
2. Validation/rejection logic matches original behavior.
3. Error paths are handled identically.
4. Edge cases (empty data, boundary values) are preserved.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scripts.training.governance import (
    ABEvaluationInput,
    ABEvaluationResult,
    CoerceFloatError,
    PromotionInput,
    PromotionResult,
    ab_rejection,
    build_ab_raw_metrics,
    build_promotion_result,
    detect_cluster_mode_collapse,
    detect_feature_and_objective_changes,
    evaluate_ab_candidate,
    load_json_dict,
    materialize_phase8_artifacts,
    normalize_calibration_block,
    normalize_metrics_payload,
    normalized_entropy_from_counts,
    summarize_governance,
    validate_ab_contract,
    validate_track,
)

# ======================================================================
# normalized_entropy_from_counts
# ======================================================================


class TestNormalizedEntropyFromCounts:
    def test_uniform_distribution(self):
        """Uniform cluster sizes should give entropy ~= 1.0."""
        entropy = normalized_entropy_from_counts([10, 10, 10, 10, 10])
        assert entropy == pytest.approx(1.0, abs=0.01)

    def test_concentrated_distribution(self):
        """Single large cluster should give entropy < 0.5."""
        entropy = normalized_entropy_from_counts([100, 1, 1, 1])
        assert entropy < 0.5

    def test_single_cluster(self):
        """Single cluster returns 0.0 (no entropy)."""
        assert normalized_entropy_from_counts([100]) == 0.0

    def test_all_empty(self):
        """All zero counts returns 0.0."""
        assert normalized_entropy_from_counts([0, 0, 0]) == 0.0

    def test_empty_list(self):
        """Empty list returns 0.0."""
        assert normalized_entropy_from_counts([]) == 0.0

    def test_negative_counts(self):
        """Negative counts are clamped to zero."""
        entropy = normalized_entropy_from_counts([-5, 10, 10])
        assert 0 <= entropy <= 1.0

    def test_two_clusters_equal(self):
        """Two equal clusters give perfect balanced entropy."""
        entropy = normalized_entropy_from_counts([50, 50])
        assert entropy == pytest.approx(1.0, abs=0.01)

    def test_clip_does_not_exceed_one(self):
        """Entropy is always in [0, 1]."""
        for _ in range(10):
            sizes = np.random.randint(1, 100, size=np.random.randint(2, 10)).tolist()
            e = normalized_entropy_from_counts(sizes)
            assert 0.0 <= e <= 1.0 + 1e-9


# ======================================================================
# detect_cluster_mode_collapse
# ======================================================================


class TestDetectClusterModeCollapse:
    def test_no_collapse(self):
        """Well-balanced clusters should not trigger collapse."""
        collapsed, metrics = detect_cluster_mode_collapse([30, 25, 20, 25])
        assert not collapsed
        assert metrics["active_cluster_count"] >= 4
        assert metrics["dominant_cluster_fraction"] <= 0.85
        assert metrics["cluster_size_entropy"] >= 0.30

    def test_dominant_cluster(self):
        """One cluster >85% of total triggers collapse."""
        collapsed, metrics = detect_cluster_mode_collapse([90, 5, 5])
        assert collapsed
        assert metrics["dominant_cluster_fraction"] >= 0.85

    def test_single_active_cluster(self):
        """Only one active cluster triggers collapse."""
        collapsed, _ = detect_cluster_mode_collapse([100, 0, 0])
        assert collapsed

    def test_all_empty_clusters(self):
        """All zeros triggers collapse."""
        collapsed, metrics = detect_cluster_mode_collapse([0, 0, 0])
        assert collapsed
        assert metrics["cluster_size_entropy"] == 0.0
        assert metrics["active_cluster_count"] == 0.0

    def test_empty_list(self):
        """Empty list triggers collapse."""
        collapsed, _ = detect_cluster_mode_collapse([])
        assert collapsed

    def test_custom_thresholds(self):
        """Custom min_entropy and max_dominance are respected."""
        collapsed, _ = detect_cluster_mode_collapse(
            [60, 20, 20],
            min_entropy=0.10,
            max_dominance=0.90,
        )
        # 60% dominant, 0.9 max_dominance, entropy ~0.95 — should be fine
        assert not collapsed

    def test_collapse_metrics_shape(self):
        """Collapse metrics contain the expected keys."""
        _, metrics = detect_cluster_mode_collapse([30, 30, 40])
        assert "cluster_size_entropy" in metrics
        assert "dominant_cluster_fraction" in metrics
        assert "active_cluster_count" in metrics

    def test_low_entropy_triggers_collapse(self):
        """Very low entropy triggers collapse even with balanced counts."""
        collapsed, _ = detect_cluster_mode_collapse(
            [95, 3, 2],
            min_entropy=0.50,
        )
        assert collapsed


# ======================================================================
# ab_rejection
# ======================================================================


class TestAbRejection:
    def test_standard_rejection_shape(self):
        """Rejection payload has the expected structure."""
        result = ab_rejection("test_reason")
        assert result == {
            "accepted": False,
            "reason": "test_reason",
            "tier_1_geometry_pass": False,
            "tier_2_cluster_quality_pass": False,
            "tier_3_classifier_pass": False,
            "tier_4_governance_pass": False,
            "tier_3_evaluated": False,
        }

    def test_rejection_carries_reason(self):
        """Reason string is conveyed faithfully."""
        result = ab_rejection("custom_failure_x")
        assert result["reason"] == "custom_failure_x"
        assert not result["accepted"]


# ======================================================================
# validate_ab_contract
# ======================================================================


class TestValidateAbContract:
    @pytest.fixture
    def base_payload(self) -> dict[str, Any]:
        return {
            "dataset_id": "test_ds_v1",
            "split_snapshot_id": "snap_001",
            "batch_size": 256,
            "eval_label_path": "cpu",
            "k": 5,
            "seed": 42,
        }

    def test_matching_contracts(self, base_payload):
        """Identical contract fields pass validation."""
        result = validate_ab_contract(base_payload, base_payload)
        assert result is None

    def test_mismatched_field(self, base_payload):
        """A single mismatched field returns a rejection."""
        current = dict(base_payload)
        current["seed"] = 99
        result = validate_ab_contract(current, base_payload)
        assert result is not None
        assert not result["accepted"]
        assert "ab_contract_mismatch:seed" in result["reason"]

    def test_missing_field(self, base_payload):
        """Missing field in current returns rejection."""
        current = dict(base_payload)
        del current["k"]
        result = validate_ab_contract(current, base_payload)
        assert result is not None
        assert "ab_contract_mismatch:k" in result["reason"]

    def test_extra_field(self, base_payload):
        """Extra field in current does not affect validation."""
        current = dict(base_payload)
        current["extra_field"] = "value"
        assert validate_ab_contract(current, base_payload) is None

    def test_empty_dicts(self):
        """Both empty dicts should still pass (no fields to mismatch)."""
        assert validate_ab_contract({}, {}) is None


# ======================================================================
# detect_feature_and_objective_changes
# ======================================================================


class TestDetectFeatureAndObjectiveChanges:
    def test_no_changes(self):
        """Identical payloads -> no changes."""
        f_changed, o_changed = detect_feature_and_objective_changes(
            {"feature_signature": "abc", "cluster_objective": "sil", "cluster_spectral_affinity": "rbf"},
            {"feature_signature": "abc", "cluster_objective": "sil", "cluster_spectral_affinity": "rbf"},
        )
        assert not f_changed
        assert not o_changed

    def test_feature_changed(self):
        """Different feature signature detected."""
        f_changed, o_changed = detect_feature_and_objective_changes(
            {"feature_signature": "abc"},
            {"feature_signature": "xyz"},
        )
        assert f_changed
        assert not o_changed

    def test_objective_changed(self):
        """Different cluster objective detected."""
        f_changed, o_changed = detect_feature_and_objective_changes(
            {"cluster_objective": "kmeans"},
            {"cluster_objective": "spectral"},
        )
        assert not f_changed
        assert o_changed

    def test_spectral_affinity_changed(self):
        """Different spectral affinity detected as objective change."""
        f_changed, o_changed = detect_feature_and_objective_changes(
            {"cluster_spectral_affinity": "rbf"},
            {"cluster_spectral_affinity": "nearest_neighbors"},
        )
        assert o_changed

    def test_missing_keys(self):
        """Missing keys default to empty strings."""
        f_changed, o_changed = detect_feature_and_objective_changes({}, {})
        assert not f_changed
        assert not o_changed


# ======================================================================
# validate_track
# ======================================================================


class TestValidateTrack:
    def test_feature_track_valid(self):
        """Feature track with only feature change -> valid."""
        result = validate_track("feature", feature_changed=True, objective_changed=False)
        assert result is None

    def test_feature_track_invalid(self):
        """Feature track with no feature change -> invalid."""
        result = validate_track("feature", feature_changed=False, objective_changed=False)
        assert result is not None
        assert "ab_contract_invalid_feature_track" in result["reason"]

    def test_objective_track_valid(self):
        """Objective track with only objective change -> valid."""
        result = validate_track("objective", feature_changed=False, objective_changed=True)
        assert result is None

    def test_objective_track_invalid(self):
        """Objective track with no objective change -> invalid."""
        result = validate_track("objective", feature_changed=False, objective_changed=False)
        assert result is not None
        assert "ab_contract_invalid_objective_track" in result["reason"]

    def test_unknown_track(self):
        """Unknown track value -> invalid."""
        result = validate_track("unknown_track", feature_changed=True, objective_changed=False)
        assert result is not None
        assert "ab_contract_invalid_track:unknown_track" in result["reason"]

    def test_track_case_insensitive(self):
        """Track value is case-insensitive."""
        assert validate_track("FEATURE", feature_changed=True, objective_changed=False) is None
        assert validate_track("Objective", feature_changed=False, objective_changed=True) is None


# ======================================================================
# build_promotion_result / summarize_governance
# ======================================================================


class TestBuildPromotionResult:
    def test_happy_path(self):
        """Well-performing seeds -> PASS status."""
        seed_runs = [
            {"macro_f1": 0.92, "class4_precision": 0.85, "class4_recall": 0.90, "entropy": 0.65, "zero_prediction_classes": 0},
            {"macro_f1": 0.91, "class4_precision": 0.84, "class4_recall": 0.89, "entropy": 0.67, "zero_prediction_classes": 0},
            {"macro_f1": 0.93, "class4_precision": 0.86, "class4_recall": 0.91, "entropy": 0.66, "zero_prediction_classes": 0},
        ]
        result = build_promotion_result(seed_runs)
        assert result.status == "PASS"
        assert result.mean_macro_f1 == pytest.approx(0.92, abs=0.01)
        assert result.min_class4_recall >= 0.80

    def test_fails_on_low_recall(self):
        """Low class4_recall triggers FAIL."""
        seed_runs = [
            {"macro_f1": 0.70, "class4_precision": 0.30, "class4_recall": 0.50, "entropy": 0.50, "zero_prediction_classes": 0},
        ]
        result = build_promotion_result(seed_runs)
        assert result.status == "FAIL"
        assert "min_class4_recall_lt_0_80" in result.failure_reasons

    def test_fails_on_high_variance(self):
        """High std_macro_f1 triggers FAIL."""
        seed_runs = [
            {"macro_f1": 0.99, "class4_precision": 0.85, "class4_recall": 0.90, "entropy": 0.65, "zero_prediction_classes": 0},
            {"macro_f1": 0.50, "class4_precision": 0.30, "class4_recall": 0.85, "entropy": 0.60, "zero_prediction_classes": 0},
        ]
        result = build_promotion_result(seed_runs)
        assert result.status == "FAIL"
        assert "std_macro_f1_gt_0_03" in result.failure_reasons

    def test_fails_on_zero_prediction_classes(self):
        """Non-zero zero_prediction_classes triggers FAIL."""
        seed_runs = [
            {"macro_f1": 0.91, "class4_precision": 0.85, "class4_recall": 0.90, "entropy": 0.65, "zero_prediction_classes": 1},
        ]
        result = build_promotion_result(seed_runs)
        assert result.status == "FAIL"
        assert "max_zero_prediction_classes_ne_0" in result.failure_reasons

    def test_to_dict_roundtrip(self):
        """PromotionResult.to_dict() produces expected keys."""
        result = PromotionResult(
            mean_macro_f1=0.92, std_macro_f1=0.01,
            mean_class4_precision=0.85, mean_class4_recall=0.90,
            min_class4_recall=0.88, mean_entropy=0.65,
            max_zero_prediction_classes=0,
            status="PASS",
        )
        d = result.to_dict()
        assert d["mean_macro_f1"] == 0.92
        assert d["status"] == "PASS"
        assert isinstance(d["failure_reasons"], list)
        assert isinstance(d["actions"], list)


class TestSummarizeGovernance:
    def test_summarize_shape(self):
        """summarize_governance returns the expected triple."""
        seed_runs = [
            {"macro_f1": 0.92, "class4_precision": 0.85, "class4_recall": 0.90, "entropy": 0.65, "zero_prediction_classes": 0},
            {"macro_f1": 0.94, "class4_precision": 0.86, "class4_recall": 0.91, "entropy": 0.67, "zero_prediction_classes": 0},
        ]
        governance, failures, actions = summarize_governance(seed_runs)
        assert isinstance(governance, dict)
        assert isinstance(failures, list)
        assert isinstance(actions, list)
        assert "mean_macro_f1" in governance
        assert "std_macro_f1" in governance
        assert "mean_class4_precision" in governance
        assert "min_class4_recall" in governance
        assert "mean_entropy" in governance
        assert "max_zero_prediction_classes" in governance

    def test_single_seed(self):
        """Single seed run should work (variance will be 0)."""
        seed_runs = [
            {"macro_f1": 0.85, "class4_precision": 0.70, "class4_recall": 0.82, "entropy": 0.55, "zero_prediction_classes": 0},
        ]
        governance, failures, actions = summarize_governance(seed_runs)
        assert governance["std_macro_f1"] == 0.0
        assert governance["mean_macro_f1"] == 0.85

    def test_promotion_input_from_dict(self):
        """PromotionInput.from_dict() parses seed run dict correctly."""
        inp = PromotionInput.from_dict({
            "macro_f1": 0.92, "class4_precision": 0.85,
            "class4_recall": 0.90, "entropy": 0.65,
            "zero_prediction_classes": 0,
        })
        assert inp.macro_f1 == 0.92
        assert inp.class4_recall == 0.90
        assert inp.zero_prediction_classes == 0


# ======================================================================
# evaluate_ab_candidate
# ======================================================================


class TestEvaluateABCandidate:
    @pytest.fixture
    def base_current(self) -> dict[str, Any]:
        return {
            "dataset_id": "test_ds_v1",
            "split_snapshot_id": "snap_001",
            "batch_size": 256,
            "eval_label_path": "cpu",
            "k": 5,
            "seed": 42,
            "feature_signature": "xyz789",
            "cluster_objective": "sil",
            "cluster_spectral_affinity": "rbf",
            "ratio": 0.5,
            "min_inter": 1.2,
            "cluster_sizes": [30, 25, 20, 25],
            "zero_prediction_classes": 0.0,
            "macro_f1": 0.92,
        }

    @pytest.fixture
    def base_baseline(self) -> dict[str, Any]:
        return {
            "dataset_id": "test_ds_v1",
            "split_snapshot_id": "snap_001",
            "batch_size": 256,
            "eval_label_path": "cpu",
            "k": 5,
            "seed": 42,
            "feature_signature": "abc123",
            "cluster_objective": "sil",
            "cluster_spectral_affinity": "rbf",
            "ratio": 0.7,
            "min_inter": 0.8,
            "zero_prediction_classes": 0.0,
            "macro_f1": 0.90,
        }

    def test_full_accept(self, base_current, base_baseline):
        """Candidate that passes all tiers -> accepted."""
        decision = evaluate_ab_candidate(
            current=base_current,
            baseline=base_baseline,
            ab_track="feature",
            governance_z_score=0.5,
            governance_z_tolerance=2.0,
        )
        assert decision["accepted"]
        assert decision["reason"] == "ok"

    def test_tier1_geometry_regression(self, base_current, base_baseline):
        """Worse geometry (higher ratio, lower min_inter) -> tier 1 rejection."""
        current = dict(base_current)
        current["ratio"] = 0.9  # worse than baseline 0.7
        current["min_inter"] = 0.5  # worse than baseline 0.8
        decision = evaluate_ab_candidate(
            current=current,
            baseline=base_baseline,
            ab_track="feature",
            governance_z_score=0.5,
            governance_z_tolerance=2.0,
        )
        assert not decision["accepted"]
        assert decision["reason"] == "tier1_geometry_regression"

    def test_tier2_cluster_mode_collapse(self, base_current, base_baseline):
        """Cluster mode collapse -> tier 2 rejection."""
        current = dict(base_current)
        current["cluster_sizes"] = [95, 2, 3]
        decision = evaluate_ab_candidate(
            current=current,
            baseline=base_baseline,
            ab_track="feature",
            governance_z_score=0.5,
            governance_z_tolerance=2.0,
        )
        assert not decision["accepted"]
        assert decision["reason"] == "tier2_cluster_mode_collapse"

    def test_tier3_classifier_regression(self, base_current, base_baseline):
        """Worse macro_f1 than baseline -> tier 3 rejection."""
        current = dict(base_current)
        current["macro_f1"] = 0.80  # worse than baseline 0.90
        decision = evaluate_ab_candidate(
            current=current,
            baseline=base_baseline,
            ab_track="feature",
            governance_z_score=0.5,
            governance_z_tolerance=2.0,
        )
        assert not decision["accepted"]
        assert decision["reason"] == "tier3_classifier_surface_regression"

    def test_tier4_z_score_out_of_tolerance(self, base_current, base_baseline):
        """Z-score exceeding tolerance -> tier 4 rejection."""
        decision = evaluate_ab_candidate(
            current=base_current,
            baseline=base_baseline,
            ab_track="feature",
            governance_z_score=5.0,
            governance_z_tolerance=2.0,
        )
        assert not decision["accepted"]
        assert decision["reason"] == "tier4_governance_drift_out_of_tolerance"

    def test_contract_mismatch(self, base_current, base_baseline):
        """Contract mismatch -> early rejection."""
        current = dict(base_current)
        current["seed"] = 99
        decision = evaluate_ab_candidate(
            current=current,
            baseline=base_baseline,
            ab_track="feature",
            governance_z_score=0.5,
            governance_z_tolerance=2.0,
        )
        assert not decision["accepted"]
        assert "ab_contract_mismatch" in decision["reason"]

    def test_mixed_change_anti_pattern(self, base_current, base_baseline):
        """Both feature and objective changed -> anti-pattern rejection."""
        current = dict(base_current)
        current["feature_signature"] = "different"
        current["cluster_objective"] = "spectral"
        decision = evaluate_ab_candidate(
            current=current,
            baseline=base_baseline,
            ab_track="feature",
            governance_z_score=0.5,
            governance_z_tolerance=2.0,
        )
        assert not decision["accepted"]
        assert "ab_anti_pattern" in decision["reason"]

    def test_zero_prediction_classes_rejection(self, base_current, base_baseline):
        """Non-zero zero_prediction_classes -> tier 3 rejection."""
        current = dict(base_current)
        current["zero_prediction_classes"] = 3.0
        decision = evaluate_ab_candidate(
            current=current,
            baseline=base_baseline,
            ab_track="feature",
            governance_z_score=0.5,
            governance_z_tolerance=2.0,
        )
        assert not decision["accepted"]
        assert "tier3" in decision["reason"]


# ======================================================================
# normalize_metrics_payload
# ======================================================================


class TestNormalizeMetricsPayload:
    def test_direct_keys(self):
        """Standard keys are passed through correctly."""
        metrics = {"macro_f1": 0.85, "class4_precision": 0.75, "class4_recall": 0.90, "mean_entropy": 0.60, "zero_prediction_classes": 0}
        normalized = normalize_metrics_payload(metrics)
        assert normalized["macro_f1"] == 0.85
        assert normalized["class4_precision"] == 0.75
        assert normalized["class4_recall"] == 0.90
        assert normalized["entropy"] == 0.60
        assert normalized["zero_prediction_classes"] == 0

    def test_aliased_family_keys(self):
        """Family-namespaced aliases are resolved."""
        metrics = {"family_macro_f1": 0.82, "family_class4_precision": 0.70, "family_class4_recall": 0.88, "family_entropy": 0.50, "family_zero_prediction_classes": 0}
        normalized = normalize_metrics_payload(metrics)
        assert normalized["macro_f1"] == 0.82
        assert normalized["class4_precision"] == 0.70
        assert normalized["class4_recall"] == 0.88

    def test_prefers_direct_keys(self):
        """Direct keys take precedence over family aliases."""
        metrics = {"macro_f1": 0.92, "family_macro_f1": 0.82}
        normalized = normalize_metrics_payload(metrics)
        assert normalized["macro_f1"] == 0.92

    def test_raises_on_non_finite(self):
        """Non-finite values raise CoerceFloatError."""
        with pytest.raises(CoerceFloatError):
            normalize_metrics_payload({"macro_f1": float("nan"), "class4_precision": 0.5, "class4_recall": 0.5, "mean_entropy": 0.5, "zero_prediction_classes": 0})

    def test_raises_on_negative_zero_classes(self):
        """Negative zero_prediction_classes raises ValueError."""
        with pytest.raises(ValueError, match="zero_prediction_classes must be >= 0"):
            normalize_metrics_payload({"macro_f1": 0.5, "class4_precision": 0.5, "class4_recall": 0.5, "mean_entropy": 0.5, "zero_prediction_classes": -1})


# ======================================================================
# build_ab_raw_metrics
# ======================================================================


class TestBuildAbRawMetrics:
    def test_basic_payload_shape(self):
        """build_ab_raw_metrics returns expected keys."""
        result = build_ab_raw_metrics(
            dataset_name="test_ds",
            dataset_id="test_ds_v1",
            split_snapshot_id="snap_001",
            ab_track="feature",
            ab_change_id="change_001",
            k=5,
            seed=42,
            batch_size=256,
            feature_signature="abc",
            cluster_objective="sil",
            cluster_spectral_affinity="rbf",
            representation_diagnostics={
                "cluster_relabel": {
                    "intra_inter_ratio": 0.5,
                    "min_inter_center_distance": 1.2,
                    "nearest_center_accuracy_val": 0.85,
                    "collision_pairs": [],
                    "nearest_cluster_pairs_top5": [],
                    "density_variance": 0.1,
                },
                "cluster_size_counts": [30, 25, 20, 25],
                "cluster_label_bridge": {
                    "old_to_cluster_purity": {"0": 0.9, "1": 0.8},
                },
            },
            dataset_metrics={"family_macro_f1": 0.92, "family_zero_prediction_classes": 0},
        )
        assert result["dataset"] == "test_ds"
        assert result["k"] == 5
        assert result["seed"] == 42
        assert result["ratio"] == 0.5
        assert result["macro_f1"] == 0.92
        assert result["zero_prediction_classes"] == 0.0
        assert "cluster_sizes" in result
        assert "cluster_purity" in result

    def test_fallback_from_cluster_to_old_counts(self):
        """When cluster_size_counts is empty, fallback to bridge counts."""
        result = build_ab_raw_metrics(
            dataset_name="ds",
            dataset_id="id",
            split_snapshot_id="s",
            ab_track="feature",
            ab_change_id="c",
            k=3,
            seed=1,
            batch_size=64,
            feature_signature="sig",
            cluster_objective="sil",
            cluster_spectral_affinity="rbf",
            representation_diagnostics={
                "cluster_label_bridge": {
                    "cluster_to_old_counts": {
                        "0": {"old_0": 10, "old_1": 5},
                        "1": {"old_2": 8},
                    },
                    "old_to_cluster_purity": {"0": 0.9, "1": 0.8},
                },
            },
            dataset_metrics={"family_macro_f1": 0.85, "family_zero_prediction_classes": 0},
        )
        # Should have 2 clusters with sizes 15 and 8
        assert "cluster_sizes" in result

    def test_timestamp_in_payload(self):
        """Timestamp field is populated."""
        result = build_ab_raw_metrics(
            dataset_name="ds", dataset_id="id", split_snapshot_id="s",
            ab_track="feature", ab_change_id="c", k=3, seed=1, batch_size=64,
            feature_signature="sig", cluster_objective="sil",
            cluster_spectral_affinity="rbf",
            representation_diagnostics={},
            dataset_metrics={},
        )
        assert "timestamp" in result


# ======================================================================
# load_json_dict
# ======================================================================


class TestLoadJsonDict:
    def test_load_valid_json(self, tmp_path):
        """Valid JSON object is loaded correctly."""
        p = tmp_path / "test.json"
        p.write_text('{"key": "value", "num": 42}')
        result = load_json_dict(p)
        assert result == {"key": "value", "num": 42}

    def test_raises_on_non_dict(self, tmp_path):
        """JSON array raises ValueError."""
        p = tmp_path / "list.json"
        p.write_text('[1, 2, 3]')
        with pytest.raises(ValueError, match="Expected JSON object"):
            load_json_dict(p)

    def test_raises_on_missing_file(self, tmp_path):
        """Missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_json_dict(tmp_path / "nonexistent.json")


# ======================================================================
# materialize_phase8_artifacts
# ======================================================================


class TestMaterializePhase8Artifacts:
    def test_all_artifacts_present(self, tmp_path):
        """All required artifacts get canonical names."""
        artifacts = {}
        expected_files = {
            "before_after_csv": "before_after.csv",
            "before_after_json": "before_after.json",
            "pr_curve_csv": "pr_curve.csv",
            "confusion_matrices_json": "confusion_matrices.json",
            "ablation_json": "ablation.json",
        }
        for src_key, canonical_name in expected_files.items():
            p = tmp_path / f"source_{canonical_name}"
            p.write_text("content")
            artifacts[src_key] = str(p)

        result = materialize_phase8_artifacts(artifacts)
        for src_key in artifacts:
            assert src_key in result

    def test_missing_artifact_key(self, tmp_path):
        """Missing required key raises ValueError."""
        with pytest.raises(ValueError, match="Missing required calibration artifact"):
            materialize_phase8_artifacts({"pr_curve_csv": "/tmp/foo.csv"})

    def test_missing_file(self, tmp_path):
        """Non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            materialize_phase8_artifacts({
                "before_after_csv": "/nonexistent/before_after.csv",
                "before_after_json": "/nonexistent/before_after.json",
                "pr_curve_csv": "/nonexistent/pr_curve.csv",
                "confusion_matrices_json": "/nonexistent/confusion_matrices.json",
                "ablation_json": "/nonexistent/ablation.json",
            })


# ======================================================================
# normalize_calibration_block
# ======================================================================


class TestNormalizeCalibrationBlock:
    def test_basic_normalization(self, tmp_path):
        """Calibration block is normalized correctly."""
        p = tmp_path / "pr_curve.csv"
        p.write_text("x,y")
        c = tmp_path / "confusion.json"
        c.write_text("{}")
        a = tmp_path / "ablation.json"
        a.write_text("{}")

        result = normalize_calibration_block(
            calibration_payload={"temperature": 1.2, "tau_4": 0.7},
            calibration_artifacts={
                "pr_curve_csv": str(p),
                "confusion_matrices_json": str(c),
                "ablation_json": str(a),
            },
        )
        assert result["temperature"] == 1.2
        assert result["tau_4"] == 0.7
        assert str(p) in result["pr_curve_path"]
        assert str(a) in result["ablation_path"]

    def test_defaults(self, tmp_path):
        """Missing calibration fields use defaults."""
        p = tmp_path / "pr_curve.csv"
        p.write_text("")
        c = tmp_path / "confusion.json"
        c.write_text("")
        a = tmp_path / "ablation.json"
        a.write_text("")

        result = normalize_calibration_block(
            calibration_payload={},
            calibration_artifacts={
                "pr_curve_csv": str(p),
                "confusion_matrices_json": str(c),
                "ablation_json": str(a),
            },
        )
        assert result["temperature"] == 1.0
        assert result["tau_4"] == 0.5

    def test_missing_artifact_file(self, tmp_path):
        """Non-existent artifact file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Required calibration artifact missing"):
            normalize_calibration_block(
                calibration_payload={"temperature": 1.0, "tau_4": 0.5},
                calibration_artifacts={
                    "pr_curve_csv": str(tmp_path / "nonexistent.csv"),
                    "confusion_matrices_json": str(tmp_path / "nonexistent.json"),
                    "ablation_json": str(tmp_path / "nonexistent.json"),
                },
            )

    def test_non_finite_values(self, tmp_path):
        """Non-finite temperature raises CoerceFloatError."""
        p = tmp_path / "pr_curve.csv"
        p.write_text("")
        c = tmp_path / "confusion.json"
        c.write_text("")
        a = tmp_path / "ablation.json"
        a.write_text("")

        with pytest.raises(CoerceFloatError):
            normalize_calibration_block(
                calibration_payload={"temperature": float("nan"), "tau_4": 0.5},
                calibration_artifacts={
                    "pr_curve_csv": str(p),
                    "confusion_matrices_json": str(c),
                    "ablation_json": str(a),
                },
            )


# ======================================================================
# GovernanceSummary dataclass
# ======================================================================


class TestGovernanceSummary:
    def test_to_governance_dict(self):
        """GovernanceSummary.to_governance_dict() returns expected keys."""
        from scripts.training.governance.reporting import GovernanceSummary
        gs = GovernanceSummary(
            mean_macro_f1=0.92, std_macro_f1=0.01,
            mean_class4_precision=0.85, mean_class4_recall=0.90,
            min_class4_recall=0.88, mean_entropy=0.65,
            max_zero_prediction_classes=0, status="PASS",
        )
        d = gs.to_governance_dict()
        assert d["mean_macro_f1"] == 0.92
        assert d["status"] == "PASS"

    def test_to_failure_tuple(self):
        """GovernanceSummary.to_failure_tuple() returns original triple."""
        from scripts.training.governance.reporting import GovernanceSummary
        gs = GovernanceSummary(
            mean_macro_f1=0.92, std_macro_f1=0.01,
            mean_class4_precision=0.85, mean_class4_recall=0.90,
            min_class4_recall=0.88, mean_entropy=0.65,
            max_zero_prediction_classes=0,
            status="FAIL", failure_reasons=["test_reason"],
        )
        gov_dict, failures, actions = gs.to_failure_tuple()
        assert gov_dict["mean_macro_f1"] == 0.92
        assert failures == ["test_reason"]

    def test_empty_failure_tuple(self):
        """Empty GovernanceSummary produces empty lists."""
        from scripts.training.governance.reporting import GovernanceSummary
        gs = GovernanceSummary(
            mean_macro_f1=0.0, std_macro_f1=0.0,
            mean_class4_precision=0.0, mean_class4_recall=0.0,
            min_class4_recall=0.0, mean_entropy=0.0,
            max_zero_prediction_classes=0,
        )
        _, failures, actions = gs.to_failure_tuple()
        assert failures == []
        assert actions == []


# ======================================================================
# ABEvaluationResult / ABEvaluationInput dataclasses
# ======================================================================


class TestABEvaluationDataclasses:
    def test_ab_evaluation_input(self):
        """ABEvaluationInput stores and retrieves fields."""
        inp = ABEvaluationInput(
            current={"key": "val"},
            baseline={"key": "baseline"},
            ab_track="feature",
            governance_z_score=1.0,
            governance_z_tolerance=2.0,
        )
        assert inp.current["key"] == "val"
        assert inp.ab_track == "feature"

    def test_ab_evaluation_result_to_dict(self):
        """ABEvaluationResult.to_dict() returns the right structure."""
        result = ABEvaluationResult(
            accepted=True, reason="ok",
            tier_1_geometry_pass=True, tier_2_cluster_quality_pass=True,
            tier_3_classifier_pass=True, tier_4_governance_pass=True,
            tier_3_evaluated=True,
            extra={"delta_macro_f1": 0.02},
        )
        d = result.to_dict()
        assert d["accepted"]
        assert d["delta_macro_f1"] == 0.02
        assert d["tier_1_geometry_pass"]

    def test_ab_evaluation_result_rejection(self):
        """ABEvaluationResult with rejection has correct defaults."""
        result = ABEvaluationResult(
            accepted=False, reason="test",
            tier_1_geometry_pass=False, tier_2_cluster_quality_pass=False,
            tier_3_classifier_pass=False, tier_4_governance_pass=False,
            tier_3_evaluated=False,
        )
        d = result.to_dict()
        assert not d["accepted"]
        assert d["reason"] == "test"
        assert not d["tier_3_evaluated"]


# ======================================================================
# CoerceFloatError
# ======================================================================


class TestCoerceFloatError:
    def test_coerce_float_error_is_value_error(self):
        """CoerceFloatError is a subclass of ValueError."""
        assert issubclass(CoerceFloatError, ValueError)

    def test_coerce_float_error_carries_message(self):
        """CoerceFloatError carries the field name in message."""
        try:
            raise CoerceFloatError("macro_f1 must be finite, got nan")
        except CoerceFloatError as e:
            assert "macro_f1" in str(e)

    def test_direct_usage_in_normalize_metrics(self):
        """Non-finite values through normalize_metrics_payload raise CoerceFloatError."""
        with pytest.raises(CoerceFloatError):
            normalize_metrics_payload({
                "macro_f1": float("inf"),
                "class4_precision": 0.5,
                "class4_recall": 0.5,
                "mean_entropy": 0.5,
                "zero_prediction_classes": 0,
            })
