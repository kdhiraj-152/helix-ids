"""CI guardrails for UNSW processed artifact learnability contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from helix_ids.data.learnability_contract import assert_contract, compute_schema_hash, load_meta


def test_unsw_processed_artifact_contract_structure_exists() -> None:
    """Verify that the learnability contract meta.json exists and has required structure."""
    artifact_dir = Path("data/processed/multi_dataset_v1")
    
    # Verify files exist
    feature_columns_path = artifact_dir / "feature_columns.npy"
    assert feature_columns_path.exists(), "Missing feature_columns.npy for processed artifact"
    
    meta = load_meta(artifact_dir=artifact_dir)
    
    # Verify complete structure
    assert "dataset" in meta
    assert "validated" in meta
    assert "violations" in meta
    assert "diagnosis" in meta, "Missing diagnosis (new feature)"
    assert "action" in meta, "Missing action (new feature)"
    assert "summary" in meta, "Missing summary (new feature)"
    
    # Verify diagnosis structure
    diagnosis = meta["diagnosis"]
    assert "primary" in diagnosis
    assert "secondary" in diagnosis
    assert "confidence" in diagnosis
    assert "scores" in diagnosis
    assert "mode" in diagnosis
    assert 0 <= diagnosis["confidence"] <= 1, "Confidence must be 0-1"
    
    # Verify summary structure
    summary = meta["summary"]
    assert "status" in summary
    assert summary["status"] in ["PASS", "FAIL"]
    assert "primary_issue" in summary
    assert "action" in summary
    assert "confidence" in summary
    assert "mode" in summary


def test_unsw_processed_artifact_contract_has_enhanced_error() -> None:
    """Verify that validation failure produces enhanced error message with root cause."""
    artifact_dir = Path("data/processed/multi_dataset_v1")
    feature_columns_path = artifact_dir / "feature_columns.npy"

    assert feature_columns_path.exists(), "Missing feature_columns.npy for processed artifact"

    feature_columns = np.load(feature_columns_path, allow_pickle=True).astype(str).tolist()
    expected_schema_hash = compute_schema_hash(
        feature_columns=feature_columns,
        transformations=["split_then_nan_to_num"],
    )

    # The artifact is expected to fail validation (feature_space_collapse)
    # Verify that the error includes diagnosis information
    with pytest.raises(RuntimeError) as exc_info:
        assert_contract(
            artifact_dir=artifact_dir,
            expected_schema_hash=expected_schema_hash,
        )
    
    error_msg = str(exc_info.value)
    
    # Verify enhanced error message format (new deterministic format)
    assert "UNSW CONTRACT FAILURE" in error_msg, "Missing enhanced error header"
    assert "Primary:" in error_msg, "Missing Primary cause in error"
    assert "Action:" in error_msg, "Missing Action in error"
    assert "Confidence:" in error_msg, "Missing Confidence in error"
    
    # Verify it contains a reasonable root cause
    assert any(cause in error_msg for cause in [
        "feature_space_collapse",
        "class_prediction_collapse",
        "feature_degeneracy",
        "scaling_destruction",
        "label_distribution_issue",
        "weak_signal",
        "distribution_shift",
        "composite_failure",
        "metric_inconsistency",
    ]), "Error should contain a recognized root cause"


def test_unsw_processed_artifact_contract_summary_available() -> None:
    """Verify that summary is available for CI integration."""
    artifact_dir = Path("data/processed/multi_dataset_v1")
    
    meta = load_meta(artifact_dir=artifact_dir)
    summary = meta["summary"]
    
    # Verify CI-relevant fields
    assert summary["status"] in ["PASS", "FAIL"]
    assert len(summary["primary_issue"]) > 0
    assert len(summary["action"]) > 0
    assert summary["confidence"] > 0

