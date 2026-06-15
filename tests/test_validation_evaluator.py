"""Comprehensive regression tests for the extracted validation evaluator module.

Phase 12B-4: covers all functions exported from
scripts/training/validation/evaluator.py.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scripts.training.validation.evaluator import (
    _ab_rejection,
    _build_ab_raw_metrics,
    _detect_feature_and_objective_changes,
    _normalized_entropy_from_probs,
    _validate_ab_contract,
    _validate_track,
    evaluate_ab_candidate,
)

# ============================================================================
# _normalized_entropy_from_probs
# ============================================================================


class TestNormalizedEntropyFromProbs:
    def test_uniform_probs_max_entropy(self) -> None:
        """Uniform distribution should yield entropy of 1.0."""
        probs = np.array(
            [[0.25, 0.25, 0.25, 0.25]], dtype=np.float64
        )
        entropy = _normalized_entropy_from_probs(probs)
        assert entropy == pytest.approx(1.0, abs=1e-6)

    def test_onehot_min_entropy(self) -> None:
        """One-hot (deterministic) distribution should yield entropy of 0.0."""
        probs = np.array(
            [[1.0, 0.0, 0.0, 0.0]], dtype=np.float64
        )
        entropy = _normalized_entropy_from_probs(probs)
        assert entropy == pytest.approx(0.0, abs=1e-6)

    def test_two_class_binary(self) -> None:
        """Two classes: 0.5/0.5 yields 1.0, 0.9/0.1 yields intermediate."""
        probs_05 = np.array([[0.5, 0.5]], dtype=np.float64)
        probs_09 = np.array([[0.9, 0.1]], dtype=np.float64)
        e_05 = _normalized_entropy_from_probs(probs_05)
        e_09 = _normalized_entropy_from_probs(probs_09)
        assert e_05 == pytest.approx(1.0, abs=1e-6)
        assert 0.0 < e_09 < 1.0

    def test_multi_sample_averaging(self) -> None:
        """Entropy should be averaged across multiple samples."""
        probs = np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        entropy = _normalized_entropy_from_probs(probs)
        assert entropy == pytest.approx(0.0, abs=1e-6)

    def test_single_class_returns_zero(self) -> None:
        """Single-class probabilities yield entropy of 0.0."""
        probs = np.array([[1.0], [1.0]], dtype=np.float64)
        entropy = _normalized_entropy_from_probs(probs)
        assert entropy == pytest.approx(0.0)

    def test_empty_input_returns_zero(self) -> None:
        """Empty probabilities should yield entropy of 0.0."""
        probs = np.empty((0, 5), dtype=np.float64)
        entropy = _normalized_entropy_from_probs(probs)
        assert entropy == pytest.approx(0.0)

    def test_entropy_range(self) -> None:
        """Entropy must always be in [0, 1]."""
        rng = np.random.RandomState(1234)
        probs = rng.dirichlet(np.ones(8), size=100).astype(np.float64)
        entropy = _normalized_entropy_from_probs(probs)
        assert 0.0 <= entropy <= 1.0

    def test_entropy_invariant_symmetric(self) -> None:
        """Entropy is invariant under label permutation (depends on distribution shape)."""
        probs_a = np.array([[0.8, 0.2]], dtype=np.float64)
        probs_b = np.array([[0.2, 0.8]], dtype=np.float64)
        assert _normalized_entropy_from_probs(probs_a) == pytest.approx(
            _normalized_entropy_from_probs(probs_b)
        )

    def test_deterministic(self) -> None:
        """Identical inputs should produce identical outputs."""
        probs = np.array([[0.5, 0.3, 0.2]], dtype=np.float64)
        e1 = _normalized_entropy_from_probs(probs)
        e2 = _normalized_entropy_from_probs(probs)
        assert e1 == e2

    def test_clipped_probs_never_nan(self) -> None:
        """Zero probabilities are clipped to avoid log(0)."""
        probs = np.array([[0.0, 1.0]], dtype=np.float64)
        entropy = _normalized_entropy_from_probs(probs)
        assert np.isfinite(entropy)


# ============================================================================
# _ab_rejection
# ============================================================================


class TestAbRejection:
    def test_returns_standard_rejection_payload(self) -> None:
        """_ab_rejection should return the standard rejection dict."""
        result = _ab_rejection("some_reason")
        assert result == {
            "accepted": False,
            "reason": "some_reason",
            "tier_1_geometry_pass": False,
            "tier_2_cluster_quality_pass": False,
            "tier_3_classifier_pass": False,
            "tier_4_governance_pass": False,
            "tier_3_evaluated": False,
        }

    def test_reason_is_propagated(self) -> None:
        """Reason string should be faithfully propagated."""
        result = _ab_rejection("ab_contract_mismatch:seed")
        assert result["reason"] == "ab_contract_mismatch:seed"

    def test_all_tiers_false(self) -> None:
        """All tier gates must be explicitly False."""
        result = _ab_rejection("test")
        assert result["tier_1_geometry_pass"] is False
        assert result["tier_2_cluster_quality_pass"] is False
        assert result["tier_3_classifier_pass"] is False
        assert result["tier_4_governance_pass"] is False


# ============================================================================
# _validate_ab_contract
# ============================================================================


class TestValidateAbContract:
    _BASELINE = {
        "dataset_id": "nsl_kdd",
        "split_snapshot_id": "v2_2024",
        "batch_size": 128,
        "eval_label_path": "cpu",
        "k": 5,
        "seed": 42,
    }

    def test_matching_contract_returns_none(self) -> None:
        """Matching current and baseline should return None."""
        current = dict(self._BASELINE)
        assert _validate_ab_contract(current, self._BASELINE) is None

    def test_mismatched_dataset_id_rejected(self) -> None:
        """Dataset_id mismatch should be rejected."""
        current = dict(self._BASELINE, dataset_id="unsw_nb15")
        result = _validate_ab_contract(current, self._BASELINE)
        assert result is not None
        assert result["accepted"] is False
        assert "dataset_id" in result["reason"]

    def test_mismatched_seed_rejected(self) -> None:
        """Seed mismatch should be rejected."""
        current = dict(self._BASELINE, seed=99)
        result = _validate_ab_contract(current, self._BASELINE)
        assert result is not None
        assert "seed" in result["reason"]

    def test_mismatched_batch_size_rejected(self) -> None:
        """Batch size mismatch should be rejected."""
        current = dict(self._BASELINE, batch_size=64)
        result = _validate_ab_contract(current, self._BASELINE)
        assert result is not None
        assert "batch_size" in result["reason"]

    def test_missing_field_triggers_default_comparison(self) -> None:
        """Missing field in current should compare None vs baseline value."""
        current = dict(self._BASELINE)
        del current["k"]
        result = _validate_ab_contract(current, self._BASELINE)
        assert result is not None


# ============================================================================
# _detect_feature_and_objective_changes
# ============================================================================


class TestDetectFeatureAndObjectiveChanges:
    def test_no_changes(self) -> None:
        """Identical feature/objective should detect no changes."""
        current = {"feature_signature": "abc", "cluster_objective": "x", "cluster_spectral_affinity": "y"}
        baseline = {"feature_signature": "abc", "cluster_objective": "x", "cluster_spectral_affinity": "y"}
        feature_changed, objective_changed = _detect_feature_and_objective_changes(current, baseline)
        assert feature_changed is False
        assert objective_changed is False

    def test_feature_changed_only(self) -> None:
        """Only feature signature change."""
        current = {"feature_signature": "xyz", "cluster_objective": "x", "cluster_spectral_affinity": "y"}
        baseline = {"feature_signature": "abc", "cluster_objective": "x", "cluster_spectral_affinity": "y"}
        feature_changed, objective_changed = _detect_feature_and_objective_changes(current, baseline)
        assert feature_changed is True
        assert objective_changed is False

    def test_objective_changed_only(self) -> None:
        """Only objective or affinity change."""
        current = {"feature_signature": "abc", "cluster_objective": "z", "cluster_spectral_affinity": "y"}
        baseline = {"feature_signature": "abc", "cluster_objective": "x", "cluster_spectral_affinity": "y"}
        feature_changed, objective_changed = _detect_feature_and_objective_changes(current, baseline)
        assert feature_changed is False
        assert objective_changed is True

    def test_both_changed(self) -> None:
        """Both feature and objective change."""
        current = {"feature_signature": "xyz", "cluster_objective": "z", "cluster_spectral_affinity": "w"}
        baseline = {"feature_signature": "abc", "cluster_objective": "x", "cluster_spectral_affinity": "y"}
        feature_changed, objective_changed = _detect_feature_and_objective_changes(current, baseline)
        assert feature_changed is True
        assert objective_changed is True

    def test_affinity_change_triggers_objective(self) -> None:
        """Affinity change alone should be detected as objective change."""
        current = {"feature_signature": "abc", "cluster_objective": "x", "cluster_spectral_affinity": "new_affinity"}
        baseline = {"feature_signature": "abc", "cluster_objective": "x", "cluster_spectral_affinity": "old_affinity"}
        _, objective_changed = _detect_feature_and_objective_changes(current, baseline)
        assert objective_changed is True

    def test_missing_signature_treated_as_empty(self) -> None:
        """Missing keys should be treated as empty strings."""
        current: dict[str, Any] = {}
        baseline: dict[str, Any] = {}
        feature_changed, objective_changed = _detect_feature_and_objective_changes(current, baseline)
        assert feature_changed is False
        assert objective_changed is False


# ============================================================================
# _validate_track
# ============================================================================


class TestValidateTrack:
    def test_feature_track_feature_changed_only(self) -> None:
        """Feature track should pass when only feature changed."""
        assert _validate_track("feature", feature_changed=True, objective_changed=False) is None

    def test_feature_track_rejected_when_objective_also_changed(self) -> None:
        """Feature track should be rejected if objective also changed."""
        result = _validate_track("feature", feature_changed=True, objective_changed=True)
        assert result is not None
        assert "invalid_feature_track" in result["reason"]

    def test_feature_track_rejected_when_nothing_changed(self) -> None:
        """Feature track should be rejected if neither changed."""
        result = _validate_track("feature", feature_changed=False, objective_changed=False)
        assert result is not None

    def test_objective_track_objective_changed_only(self) -> None:
        """Objective track should pass when only objective changed."""
        assert _validate_track("objective", feature_changed=False, objective_changed=True) is None

    def test_objective_track_rejected_when_feature_also_changed(self) -> None:
        """Objective track should be rejected if feature also changed."""
        result = _validate_track("objective", feature_changed=True, objective_changed=True)
        assert result is not None
        assert "invalid_objective_track" in result["reason"]

    def test_objective_track_rejected_when_nothing_changed(self) -> None:
        """Objective track should be rejected if neither changed."""
        result = _validate_track("objective", feature_changed=False, objective_changed=False)
        assert result is not None

    def test_unknown_track_rejected(self) -> None:
        """Unrecognized track should be rejected."""
        result = _validate_track("invalid_track", feature_changed=True, objective_changed=False)
        assert result is not None
        assert "invalid_track" in result["reason"]

    def test_track_case_insensitive(self) -> None:
        """Track matching should be case-insensitive."""
        assert _validate_track("FEATURE", feature_changed=True, objective_changed=False) is None
        assert _validate_track("OBJECTIVE", feature_changed=False, objective_changed=True) is None


# ============================================================================
# evaluate_ab_candidate (full tiered evaluation)
# ============================================================================


class TestEvaluateAbCandidate:
    _BASELINE = {
        "dataset_id": "nsl_kdd",
        "split_snapshot_id": "v2_2024",
        "batch_size": 128,
        "eval_label_path": "cpu",
        "k": 5,
        "seed": 42,
        "ratio": 0.5,
        "min_inter": 1.0,
        "macro_f1": 0.85,
        "zero_prediction_classes": 0.0,
        "cluster_sizes": [30, 25, 20, 15, 10],
        "feature_signature": "abc",
        "cluster_objective": "x",
        "cluster_spectral_affinity": "y",
    }
    _CURRENT = dict(_BASELINE, feature_signature="def", ratio=0.4, min_inter=2.0)

    def test_full_acceptance(self) -> None:
        """All tiers pass should yield accepted=True."""
        result = evaluate_ab_candidate(
            current=self._CURRENT,
            baseline=self._BASELINE,
            ab_track="feature",
            governance_z_score=0.5,
            governance_z_tolerance=2.0,
        )
        assert result["accepted"] is True
        assert result["reason"] == "ok"
        assert result["tier_3_evaluated"] is True

    def test_contract_mismatch_rejected(self) -> None:
        """Contract mismatch should fail fast before tier 1."""
        current = dict(self._CURRENT, dataset_id="unsw_nb15")
        result = evaluate_ab_candidate(
            current=current,
            baseline=self._BASELINE,
            ab_track="feature",
            governance_z_score=0.5,
            governance_z_tolerance=2.0,
        )
        assert result["accepted"] is False
        assert result["tier_1_geometry_pass"] is False
        assert result["tier_3_evaluated"] is False

    def test_tier1_geometry_failure(self) -> None:
        """Worse ratio/min_inter should fail tier 1."""
        current = dict(self._CURRENT, ratio=0.6, min_inter=0.5)
        result = evaluate_ab_candidate(
            current=current,
            baseline=self._BASELINE,
            ab_track="feature",
            governance_z_score=0.5,
            governance_z_tolerance=2.0,
        )
        assert result["accepted"] is False
        assert result["tier_1_geometry_pass"] is False
        assert result["reason"] == "tier1_geometry_regression"
        assert "delta_ratio" in result
        assert "delta_min_inter" in result

    def test_tier1_ratio_worse_mininter_better_still_fails(self) -> None:
        """Both conditions must pass (ratio < baseline AND min_inter > baseline)."""
        current = dict(self._CURRENT, ratio=0.6, min_inter=2.0)
        result = evaluate_ab_candidate(
            current=current,
            baseline=self._BASELINE,
            ab_track="feature",
            governance_z_score=0.5,
            governance_z_tolerance=2.0,
        )
        assert result["accepted"] is False
        assert result["tier_1_geometry_pass"] is False

    def test_tier2_cluster_collapse_failure(self) -> None:
        """Cluster mode collapse should fail tier 2."""
        current = dict(self._CURRENT, cluster_sizes=[95, 1, 1, 1, 2])
        result = evaluate_ab_candidate(
            current=current,
            baseline=self._BASELINE,
            ab_track="feature",
            governance_z_score=0.5,
            governance_z_tolerance=2.0,
        )
        assert result["accepted"] is False
        assert result["tier_1_geometry_pass"] is True
        assert result["tier_2_cluster_quality_pass"] is False
        assert result["reason"] == "tier2_cluster_mode_collapse"

    def test_tier3_classifier_failure(self) -> None:
        """Worse macro_f1 should fail tier 3."""
        current = dict(self._CURRENT, macro_f1=0.75, zero_prediction_classes=0.0)
        result = evaluate_ab_candidate(
            current=current,
            baseline=self._BASELINE,
            ab_track="feature",
            governance_z_score=0.5,
            governance_z_tolerance=2.0,
        )
        assert result["accepted"] is False
        assert result["tier_1_geometry_pass"] is True
        assert result["tier_2_cluster_quality_pass"] is True
        assert result["tier_3_classifier_pass"] is False
        assert result["tier_3_evaluated"] is True
        assert "delta_macro_f1" in result

    def test_tier4_governance_failure(self) -> None:
        """Z-score out of tolerance should fail tier 4."""
        result = evaluate_ab_candidate(
            current=self._CURRENT,
            baseline=self._BASELINE,
            ab_track="feature",
            governance_z_score=5.0,
            governance_z_tolerance=2.0,
        )
        assert result["accepted"] is False
        assert result["reason"] == "tier4_governance_drift_out_of_tolerance"
        assert result["tier_4_governance_pass"] is False

    def test_both_feature_and_objective_changed_rejected(self) -> None:
        """Mixed feature+objective change is an anti-pattern rejection."""
        current = dict(
            self._CURRENT,
            feature_signature="new_feat",
            cluster_objective="new_obj",
        )
        result = evaluate_ab_candidate(
            current=current,
            baseline=self._BASELINE,
            ab_track="feature",
            governance_z_score=0.5,
            governance_z_tolerance=2.0,
        )
        assert result["accepted"] is False
        assert result["reason"] == "ab_anti_pattern_mixed_feature_and_objective_change"

    def test_invalid_track_rejected(self) -> None:
        """Invalid track should be rejected."""
        result = evaluate_ab_candidate(
            current=self._CURRENT,
            baseline=self._BASELINE,
            ab_track="nonsense",
            governance_z_score=0.5,
            governance_z_tolerance=2.0,
        )
        assert result["accepted"] is False
        assert "invalid_track" in result["reason"]

    def test_all_tier_flags_in_accepted_response(self) -> None:
        """Accepted response should have all tier flags True."""
        result = evaluate_ab_candidate(
            current=self._CURRENT,
            baseline=self._BASELINE,
            ab_track="feature",
            governance_z_score=0.5,
            governance_z_tolerance=2.0,
        )
        assert result["tier_1_geometry_pass"] is True
        assert result["tier_2_cluster_quality_pass"] is True
        assert result["tier_3_classifier_pass"] is True
        assert result["tier_4_governance_pass"] is True


# ============================================================================
# _build_ab_raw_metrics
# ============================================================================


class TestBuildAbRawMetrics:
    def test_basic_payload_structure(self) -> None:
        """Verify full payload structure with expected keys."""
        result = _build_ab_raw_metrics(
            dataset_name="nsl_kdd",
            dataset_id="nsl_kdd",
            split_snapshot_id="v2_2024",
            ab_track="feature",
            ab_change_id="change_001",
            k=5,
            seed=42,
            batch_size=128,
            feature_signature="abc123",
            cluster_objective="x",
            cluster_spectral_affinity="y",
            representation_diagnostics={
                "cluster_relabel": {
                    "intra_inter_ratio": 0.75,
                    "min_inter_center_distance": 1.5,
                    "nearest_center_accuracy_val": 0.92,
                    "collision_pairs": [],
                    "nearest_cluster_pairs_top5": [],
                    "density_variance": 0.1,
                },
                "cluster_size_counts": [30, 25, 20, 15, 10],
                "cluster_size_entropy": 0.85,
                "cluster_label_bridge": {
                    "old_to_cluster_purity": {"0": 0.9, "1": 0.85, "2": 0.8, "3": 0.75, "4": 0.7},
                    "cluster_to_old_counts": {},
                },
            },
            dataset_metrics={
                "family_macro_f1": 0.88,
                "family_zero_prediction_classes": 0.0,
            },
        )
        assert result["dataset"] == "nsl_kdd"
        assert result["ratio"] == 0.75
        assert result["min_inter"] == 1.5
        assert result["macro_f1"] == 0.88
        assert result["zero_prediction_classes"] == 0.0
        assert result["k"] == 5
        assert result["seed"] == 42
        assert result["cluster_purity"] == [0.9, 0.85, 0.8, 0.75, 0.7]
        assert "timestamp" in result

    def test_fallback_to_bridge_cluster_sizes(self) -> None:
        """When cluster_size_counts is empty, fall back to bridge cluster_to_old_counts."""
        result = _build_ab_raw_metrics(
            dataset_name="test",
            dataset_id="test",
            split_snapshot_id="v1",
            ab_track="feature",
            ab_change_id="ch1",
            k=3,
            seed=1,
            batch_size=64,
            feature_signature="sig",
            cluster_objective="obj",
            cluster_spectral_affinity="aff",
            representation_diagnostics={
                "cluster_label_bridge": {
                    "cluster_to_old_counts": {
                        "0": {"0": 10, "1": 5},
                        "1": {"2": 8, "3": 7},
                    },
                    "old_to_cluster_purity": {"0": 0.9, "1": 0.8},
                },
            },
            dataset_metrics={"family_macro_f1": 0.8, "family_zero_prediction_classes": 0.0},
        )
        assert len(result["cluster_sizes"]) >= 2
        assert result["cluster_purity"] == [0.9, 0.8]

    def test_family_f1_fallback(self) -> None:
        """When family_macro_f1 absent, fall back to family_f1."""
        result = _build_ab_raw_metrics(
            dataset_name="test",
            dataset_id="test",
            split_snapshot_id="v1",
            ab_track="feature",
            ab_change_id="ch1",
            k=3,
            seed=1,
            batch_size=64,
            feature_signature="sig",
            cluster_objective="obj",
            cluster_spectral_affinity="aff",
            representation_diagnostics={
                "cluster_relabel": {},
                "cluster_label_bridge": {},
            },
            dataset_metrics={"family_f1": 0.75},
        )
        assert result["macro_f1"] == 0.75

    def test_missing_cluster_diag_fallback(self) -> None:
        """Missing cluster_relabel should fall back to original."""
        result = _build_ab_raw_metrics(
            dataset_name="test",
            dataset_id="test",
            split_snapshot_id="v1",
            ab_track="feature",
            ab_change_id="ch1",
            k=3,
            seed=1,
            batch_size=64,
            feature_signature="sig",
            cluster_objective="obj",
            cluster_spectral_affinity="aff",
            representation_diagnostics={
                "original": {"intra_inter_ratio": 0.6, "min_inter_center_distance": 1.2},
                "cluster_label_bridge": {},
            },
            dataset_metrics={"family_macro_f1": 0.8},
        )
        assert result["ratio"] == 0.6
        assert result["min_inter"] == 1.2
