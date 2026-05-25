import json
import numpy as np
import pandas as pd
import pytest

from src.helix_ids.operations.monitoring import LiveMonitor, ContractViolationError
from src.helix_ids.data.feature_harmonization import (
    FEATURE_ORDER,
    SchemaDriftError,
    create_cicids_mapping,
    enforce_feature_order,
    harmonize_features,
    load_artifact,
)
from src.helix_ids.data.multi_dataset_loader import MultiDatasetLoader
from src.helix_ids.contracts import CONTRACT_VERSION
from src.helix_ids.contracts import SCHEMA_VERSION


def test_monitor_raises_on_cardinality_mismatch(tmp_path):
    baseline = np.array([0.5, 0.5])
    monitor = LiveMonitor(baseline_class_distribution=baseline, baseline_entropy=0.69)

    # preds include classes up to 3 -> observed cardinality 4 != baseline 2
    preds = np.array([0, 1, 2, 3, 2, 1])

    with pytest.raises(ContractViolationError) as exc:
        monitor.evaluate(
            preds,
            producer="unit-test",
            artifact_path="/tmp/fake.pt",
            schema_hash_expected="abc",
            schema_hash_actual="def",
            feature_names_expected=["a", "b"],
            feature_names_actual=["a", "b", "c"],
            telemetry_dir=tmp_path,
        )

    payload = exc.value.payload
    assert payload is not None
    assert payload["event"] == "schema_drift_detected"
    assert payload["timestamp_utc"].endswith("Z")
    assert payload["producer"] == "unit-test"
    assert payload["feature_names_expected"] == ["a", "b"]
    assert payload["feature_names_actual"] == ["a", "b", "c"]
    # telemetry file written
    files = list(tmp_path.iterdir())
    assert any(p.name.startswith("schema_drift_") and p.suffix == ".json" for p in files)
    telemetry = json.loads(next(p for p in files if p.suffix == ".json").read_text())
    assert telemetry["event"] == "schema_drift_detected"
    assert telemetry["timestamp_utc"].endswith("Z")


def test_feature_harmonization_rejects_column_reorder(tmp_path):
    df = pd.DataFrame(
        {
            "Flow Duration": [1.0],
            "Protocol": [6],
            "TotLen Fwd Pkts": [10.0],
            "TotLen Bwd Pkts": [5.0],
            "SYN Flag Cnt": [1.0],
            "RST Flag Cnt": [0.0],
            "ACK Flag Cnt": [1.0],
            "FIN Flag Cnt": [0.0],
            "Tot Fwd Pkts": [2.0],
            "Tot Bwd Pkts": [3.0],
            "Label": ["BENIGN"],
        }
    )
    harmonized = harmonize_features(df, create_cicids_mapping(), label_col="label")
    permuted = harmonized[FEATURE_ORDER].sample(frac=1, axis=1, random_state=7)

    model_path = tmp_path / "artifact_reorder.pt"
    import torch

    torch.save(
        {
            "model": {"w": torch.zeros((1, 1))},
            "schema_version": SCHEMA_VERSION,
            "schema_hash": harmonized.attrs["schema_hash"],
            "feature_order": FEATURE_ORDER,
            "input_dim": len(FEATURE_ORDER),
            "binary_output_dim": 2,
            "family_output_dim": 7,
            "contract_version": CONTRACT_VERSION,
        },
        model_path,
    )

    with pytest.raises(SchemaDriftError):
        load_artifact(
            model_path,
            permuted,
        )


def test_feature_harmonization_rejects_missing_and_extra_features(tmp_path):
    base = pd.DataFrame(
        {
            "Flow Duration": [1.0],
            "Protocol": [6],
            "TotLen Fwd Pkts": [10.0],
            "TotLen Bwd Pkts": [5.0],
            "SYN Flag Cnt": [1.0],
            "RST Flag Cnt": [0.0],
            "ACK Flag Cnt": [1.0],
            "FIN Flag Cnt": [0.0],
            "Tot Fwd Pkts": [2.0],
            "Tot Bwd Pkts": [3.0],
            "Label": ["BENIGN"],
        }
    )
    harmonized = harmonize_features(base, create_cicids_mapping(), label_col="label")
    features = harmonized[FEATURE_ORDER].astype(np.float32)
    model_path = tmp_path / "artifact.pt"
    import torch

    torch.save(
        {
            "model": {"w": torch.zeros((1, 1))},
            "schema_version": SCHEMA_VERSION,
            "schema_hash": harmonized.attrs["schema_hash"],
            "feature_order": FEATURE_ORDER,
            "input_dim": len(FEATURE_ORDER),
            "binary_output_dim": 2,
            "family_output_dim": 7,
            "contract_version": CONTRACT_VERSION,
        },
        model_path,
    )

    with pytest.raises(SchemaDriftError):
        load_artifact(model_path, features.drop(columns=[FEATURE_ORDER[0]]))

    with pytest.raises(SchemaDriftError):
        load_artifact(model_path, features.assign(extra_feature=1.0))


def test_feature_harmonization_rejects_schema_hash_mismatch(tmp_path):
    base = pd.DataFrame(
        {
            "Flow Duration": [1.0],
            "Protocol": [6],
            "TotLen Fwd Pkts": [10.0],
            "TotLen Bwd Pkts": [5.0],
            "SYN Flag Cnt": [1.0],
            "RST Flag Cnt": [0.0],
            "ACK Flag Cnt": [1.0],
            "FIN Flag Cnt": [0.0],
            "Tot Fwd Pkts": [2.0],
            "Tot Bwd Pkts": [3.0],
            "Label": ["BENIGN"],
        }
    )
    harmonized = harmonize_features(base, create_cicids_mapping(), label_col="label")
    features = harmonized[FEATURE_ORDER].astype(np.float32)
    model_path = tmp_path / "artifact_hash.pt"
    import torch

    torch.save(
        {
            "model": {"w": torch.zeros((1, 1))},
            "schema_version": SCHEMA_VERSION,
            "schema_hash": "deadbeef",
            "feature_order": FEATURE_ORDER,
            "input_dim": len(FEATURE_ORDER),
            "binary_output_dim": 2,
            "family_output_dim": 7,
            "contract_version": CONTRACT_VERSION,
        },
        model_path,
    )

    with pytest.raises(AssertionError, match="schema_hash mismatch"):
        load_artifact(model_path, features)


def test_feature_harmonization_rejects_label_space_mismatch():
    loader = MultiDatasetLoader()
    df = pd.DataFrame(
        {
            "Flow Duration": [1.0],
            "Protocol": [6],
            "TotLen Fwd Pkts": [10.0],
            "TotLen Bwd Pkts": [5.0],
            "SYN Flag Cnt": [1.0],
            "RST Flag Cnt": [0.0],
            "ACK Flag Cnt": [1.0],
            "FIN Flag Cnt": [0.0],
            "Tot Fwd Pkts": [2.0],
            "Tot Bwd Pkts": [3.0],
            "Label": ["UNKNOWN_ATTACK"],
        }
    )

    with pytest.raises(ValueError, match="label-space mismatch"):
        loader.harmonize_cicids(df)


def test_enforce_feature_order_rejects_mismatch():
    # Create DataFrame with wrong column order
    cols = list(FEATURE_ORDER)
    df = pd.DataFrame([[0]*len(cols)], columns=cols[::-1])
    with pytest.raises(SchemaDriftError):
        enforce_feature_order(df, FEATURE_ORDER)
