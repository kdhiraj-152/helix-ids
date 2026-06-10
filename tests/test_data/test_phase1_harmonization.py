"""
Unit tests for data harmonization and multi-dataset loading (Phase 1).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.helix_ids.contracts import CONTRACT_VERSION, SCHEMA_VERSION
from src.helix_ids.data.feature_harmonization import (
    ATTACK_TAXONOMY_7CLASS,
    COMMON_FEATURES,
    FEATURE_ORDER,
    INVARIANT_FEATURES,
    SchemaDriftError,
    _derive_connection_state,
    compute_schema_hash,
    create_cicids_mapping,
    create_nslkdd_mapping,
    create_unsw_mapping,
    enforce_feature_order,
    harmonize_features,
    load_artifact,
    normalize_column_name,
    validate_mapping,
)
from src.helix_ids.data.multi_dataset_loader import UNSW_DISCRETE_PROBE_F1_MIN, MultiDatasetLoader


class TestFeatureHarmonization:
    """Test feature harmonization module."""

    @staticmethod
    def _snapshot_path() -> Path:
        return Path(__file__).resolve().parents[1] / "fixtures" / "cicids_snapshot.csv"

    def test_common_features_count(self):
        """Verify invariant feature set size."""
        assert len(COMMON_FEATURES) == 17
        assert isinstance(COMMON_FEATURES, list)

    def test_attack_taxonomy_7class(self):
        """Verify 7-class attack taxonomy."""
        assert len(ATTACK_TAXONOMY_7CLASS) == 7
        assert 0 in ATTACK_TAXONOMY_7CLASS  # Normal
        assert ATTACK_TAXONOMY_7CLASS[0] == "Normal"

    def test_nslkdd_mapping(self):
        """Test NSL-KDD feature mapping."""
        mapping = create_nslkdd_mapping()
        assert mapping.dataset_name == "nsl_kdd"
        assert len(mapping.common_features) == len(COMMON_FEATURES)
        for key in ["duration", "src_bytes", "dst_bytes", "protocol", "state"]:
            assert key in mapping.feature_mapping

        payload = mapping.to_dict()
        validate_mapping(payload)
        assert payload["protocol"] == "v1"
        assert payload["version"]
        assert "mapping" in payload

    def test_unsw_mapping(self):
        """Test UNSW-NB15 feature mapping."""
        mapping = create_unsw_mapping()
        assert mapping.dataset_name == "unsw_nb15"
        for key in ["duration", "src_bytes", "dst_bytes", "protocol", "state"]:
            assert key in mapping.feature_mapping

    def test_cicids_mapping(self):
        """Test CICIDS feature mapping."""
        mapping = create_cicids_mapping()
        assert mapping.dataset_name == "cicids"
        for key in ["duration", "src_bytes", "dst_bytes", "protocol", "syn_count", "rst_count"]:
            assert key in mapping.feature_mapping

    def test_harmonize_rejects_cicids_messy_columns(self):
        """Corrupted CICIDS inputs must fail at the boundary."""
        df = pd.DataFrame(
            {
                "  Flow Duration ": [1000, 2000],
                "Protocol": [6, 17],
                " Dst Port": [80, 443],
                "TotLen Fwd Pkts": [100.0, np.inf],
                "TotLen Bwd Pkts": [200.0, np.nan],
                "SYN Flag Cnt": [10, 2],
                "RST Flag Cnt": [0, 1],
                "ACK Flag Cnt": [9, 1],
                "FIN Flag Cnt": [1, 0],
                "Tot Fwd Pkts": [12, 3],
                "Tot Bwd Pkts": [9, 1],
                "Flow IAT Mean": [100.0, 200.0],
                "Fwd IAT Max": [130.0, 260.0],
                "Fwd IAT Min": [70.0, 120.0],
                "Bwd IAT Max": [110.0, 210.0],
                "Bwd IAT Min": [50.0, 140.0],
                "Active Mean": [20.0, 40.0],
                "Label ": [" Benign", "Bot  "],
            }
        )
        mapping = create_cicids_mapping()
        with pytest.raises(AssertionError, match="NaN/inf detected in input"):
            harmonize_features(df, mapping, label_col="label")


    def test_cicids_nan_inf_rejection(self):
        """Explicit corrupt CICIDS sample should be rejected with invalid column context."""
        df = pd.DataFrame(
            {
                "Flow Duration": [1000.0, 2000.0],
                "TotLen Fwd Pkts": [100.0, np.inf],
                "TotLen Bwd Pkts": [50.0, np.nan],
                "Protocol": [6, 17],
                "SYN Flag Cnt": [6.0, 1.0],
                "RST Flag Cnt": [0.0, 1.0],
                "ACK Flag Cnt": [5.0, 1.0],
                "Tot Fwd Pkts": [10.0, 15.0],
                "Dst Port": [80, 443],
                "Flow IAT Mean": [100.0, 300.0],
                "Label": ["BENIGN", "DDoS"],
            }
        )
        mapping = create_cicids_mapping()
        with pytest.raises(AssertionError) as excinfo:
            harmonize_features(df, mapping, label_col="label")
        assert "invalid_cols" in str(excinfo.value)

    def test_real_cicids_snapshot_contract(self):
        """Real CICIDS snapshot should harmonize to the frozen feature order and carry lineage attrs."""
        df = pd.read_csv(self._snapshot_path())
        loader = MultiDatasetLoader()

        harmonized = loader.harmonize_cicids(df)
        assert harmonized.shape[1] == len(FEATURE_ORDER) + 1
        assert list(harmonized.columns[:-1]) == FEATURE_ORDER
        assert np.isfinite(harmonized[FEATURE_ORDER].to_numpy()).all()
        assert harmonized.attrs["source"] == "CICIDS"
        assert harmonized.attrs["contract_version"] == CONTRACT_VERSION
        assert harmonized.attrs["feature_order"] == FEATURE_ORDER
        assert isinstance(harmonized.attrs["schema_hash"], str)

    def test_schema_diff_logger_reports_order_mismatch(self):
        """Feature order permutation should fail with explicit drift metadata."""
        frame = pd.DataFrame({col: [1.0] for col in FEATURE_ORDER})
        permuted = frame.loc[:, list(reversed(FEATURE_ORDER))]

        with pytest.raises(SchemaDriftError) as excinfo:
            enforce_feature_order(permuted, FEATURE_ORDER, context="snapshot")

        error = excinfo.value
        assert error.missing == []
        assert error.extra == []
        assert error.order_mismatch is True
        assert "order_mismatch=True" in str(error)

    def test_lenient_mode_sanitizes_corrupted_input(self, monkeypatch):
        """Lenient mode should sanitize numeric corruption and keep the output finite."""
        monkeypatch.setenv("HELIX_DEBUG_LENIENT", "1")
        df = pd.read_csv(self._snapshot_path()).head(2).copy()
        df.loc[df.index[0], "TotLen Fwd Pkts"] = np.inf
        df.loc[df.index[1], "TotLen Bwd Pkts"] = np.nan

        harmonized = harmonize_features(df, create_cicids_mapping(), label_col="attack_type", mode="lenient")
        assert np.isfinite(harmonized[FEATURE_ORDER].to_numpy()).all()
        assert harmonized.attrs["pipeline_mode"] == "lenient"

    def test_mutation_guard_detects_missing_sanitization(self, monkeypatch):
        """If sanitization is removed, corrupted input must still fail under lenient mode."""
        monkeypatch.setenv("HELIX_DEBUG_LENIENT", "1")
        df = pd.read_csv(self._snapshot_path()).head(2).copy()
        df.loc[df.index[0], "TotLen Fwd Pkts"] = np.inf

        from src.helix_ids.data import feature_harmonization as fh

        original = fh.sanitize_numeric
        original_validate = fh.validate_no_nan_inf
        try:
            fh.sanitize_numeric = lambda frame: frame  # type: ignore[assignment]
            fh.validate_no_nan_inf = lambda frame: None  # type: ignore[assignment]
            harmonized = harmonize_features(df, create_cicids_mapping(), label_col="attack_type", mode="lenient")
            assert np.isfinite(harmonized[FEATURE_ORDER].to_numpy()).all()
        finally:
            fh.sanitize_numeric = original
            fh.validate_no_nan_inf = original_validate

    def test_model_load_rejects_schema_drift(self, tmp_path: Path):
        valid_df = pd.read_csv(self._snapshot_path()).head(2)
        harmonized = harmonize_features(valid_df, create_cicids_mapping(), label_col="attack_type", mode="strict")
        features = harmonized[FEATURE_ORDER].astype(np.float32)

        artifact_path = tmp_path / "artifact.pt"
        artifact = {
            "model": {"w": torch.zeros((2, 2))},
            "schema_version": SCHEMA_VERSION,
            "schema_hash": compute_schema_hash(features),
            "feature_order": FEATURE_ORDER,
            "input_dim": len(FEATURE_ORDER),
            "binary_output_dim": 2,
            "family_output_dim": 7,
            "contract_version": CONTRACT_VERSION,
        }
        torch.save(artifact, artifact_path)
        from helix_ids.governance import (
            build_artifact_manifest,
            checkpoint_manifest_payload,
            write_contract_sidecars,
        )
        from helix_ids.utils.export import finalize_export_artifact
        contract = {key: artifact[key] for key in ("schema_version", "schema_hash", "feature_order", "input_dim", "binary_output_dim", "family_output_dim", "contract_version")}
        manifest_base = build_artifact_manifest(model_architecture="unit-test", contract=contract)
        artifact["artifact_manifest"] = checkpoint_manifest_payload(manifest_base)
        torch.save(artifact, artifact_path)
        sidecars = write_contract_sidecars(artifact_path, contract)
        finalize_export_artifact(artifact_path, manifest_base, sidecars=sidecars)

        drifted = features.copy()
        drifted["extra_col"] = 1.0
        with pytest.raises(SchemaDriftError):
            load_artifact(artifact_path, drifted)

    def test_column_permutation_rejected(self, tmp_path: Path):
        valid_df = pd.read_csv(self._snapshot_path()).head(2)
        harmonized = harmonize_features(valid_df, create_cicids_mapping(), label_col="attack_type", mode="strict")
        features = harmonized[FEATURE_ORDER].astype(np.float32)

        artifact_path = tmp_path / "artifact.pt"
        artifact = {
            "model": {"w": torch.zeros((2, 2))},
            "schema_version": SCHEMA_VERSION,
            "schema_hash": compute_schema_hash(features),
            "feature_order": FEATURE_ORDER,
            "input_dim": len(FEATURE_ORDER),
            "binary_output_dim": 2,
            "family_output_dim": 7,
            "contract_version": CONTRACT_VERSION,
        }
        torch.save(artifact, artifact_path)
        from helix_ids.governance import (
            build_artifact_manifest,
            checkpoint_manifest_payload,
            write_contract_sidecars,
        )
        from helix_ids.utils.export import finalize_export_artifact
        contract = {key: artifact[key] for key in ("schema_version", "schema_hash", "feature_order", "input_dim", "binary_output_dim", "family_output_dim", "contract_version")}
        manifest_base = build_artifact_manifest(model_architecture="unit-test", contract=contract)
        artifact["artifact_manifest"] = checkpoint_manifest_payload(manifest_base)
        torch.save(artifact, artifact_path)
        sidecars = write_contract_sidecars(artifact_path, contract)
        finalize_export_artifact(artifact_path, manifest_base, sidecars=sidecars)

        permuted = features.sample(frac=1, axis=1, random_state=42)
        with pytest.raises(SchemaDriftError):
            load_artifact(artifact_path, permuted)

    def test_partial_artifact_load_rejected(self, tmp_path: Path):
        valid_df = pd.read_csv(self._snapshot_path()).head(1)
        harmonized = harmonize_features(valid_df, create_cicids_mapping(), label_col="attack_type", mode="strict")
        features = harmonized[FEATURE_ORDER].astype(np.float32)

        artifact_path = tmp_path / "artifact_incomplete.pt"
        torch.save({"model": {"w": torch.zeros((1, 1))}}, artifact_path)

        with pytest.raises(AssertionError, match="Missing required artifact key"):
            load_artifact(artifact_path, features)

    def test_artifact_version_lock(self, tmp_path: Path):
        valid_df = pd.read_csv(self._snapshot_path()).head(1)
        harmonized = harmonize_features(valid_df, create_cicids_mapping(), label_col="attack_type", mode="strict")
        features = harmonized[FEATURE_ORDER].astype(np.float32)

        artifact_path = tmp_path / "artifact_wrong_version.pt"
        artifact = {
            "model": {"w": torch.zeros((1, 1))},
            "schema_version": SCHEMA_VERSION,
            "schema_hash": compute_schema_hash(features),
            "feature_order": FEATURE_ORDER,
            "input_dim": len(FEATURE_ORDER),
            "binary_output_dim": 2,
            "family_output_dim": 7,
            "contract_version": "0.0",
        }
        torch.save(artifact, artifact_path)
        from helix_ids.governance import (
            build_artifact_manifest,
            checkpoint_manifest_payload,
            write_contract_sidecars,
        )
        from helix_ids.utils.export import finalize_export_artifact
        contract = {key: artifact[key] for key in ("schema_version", "schema_hash", "feature_order", "input_dim", "binary_output_dim", "family_output_dim", "contract_version")}
        manifest_base = build_artifact_manifest(model_architecture="unit-test", contract=contract)
        artifact["artifact_manifest"] = checkpoint_manifest_payload(manifest_base)
        torch.save(artifact, artifact_path)
        sidecars = write_contract_sidecars(artifact_path, contract)
        finalize_export_artifact(artifact_path, manifest_base, sidecars=sidecars)

        from helix_ids.governance.provenance import ArtifactManifestError
        with pytest.raises(ArtifactManifestError, match="Manifest contract_version mismatch"):
            load_artifact(artifact_path, features)

    def test_deployment_manifest_verified_on_artifact_load(self, tmp_path: Path):
        """Valid deployment.manifest.json beside artifact must not cause load failure."""
        valid_df = pd.read_csv(self._snapshot_path()).head(2)
        harmonized = harmonize_features(valid_df, create_cicids_mapping(), label_col="attack_type", mode="strict")
        features = harmonized[FEATURE_ORDER].astype(np.float32)

        from helix_ids.governance import (
            build_artifact_manifest,
            write_contract_sidecars,
            write_deployment_manifest,
        )
        from helix_ids.utils.export import finalize_export_artifact
        contract = {
            "schema_version": SCHEMA_VERSION,
            "schema_hash": compute_schema_hash(features),
            "feature_order": FEATURE_ORDER,
            "input_dim": len(FEATURE_ORDER),
            "binary_output_dim": 2,
            "family_output_dim": 7,
            "contract_version": CONTRACT_VERSION,
            "export_config_hash": "test_export_config",
        }
        manifest_base = build_artifact_manifest(
            model_architecture="unit-test",
            contract=contract,
            export_config={},
            dataset_hash="deadbeef",
            git_commit="abc1234",
            exporter_version="1.0.0",
            runtime_version="1.0.0",
        )
        # Build artifact with full manifest (so finalize_export_artifact can read it)
        artifact = {
            "model": {"w": torch.zeros((2, 2))},
            "artifact_manifest": manifest_base,
            **{k: contract[k] for k in (
                "schema_version", "schema_hash", "feature_order", "input_dim",
                "binary_output_dim", "family_output_dim", "contract_version",
            )},
        }
        artifact_path = tmp_path / "artifact_with_deploy.pt"
        torch.save(artifact, artifact_path)
        write_contract_sidecars(artifact_path, contract)
        # Finalize to get artifact_sha256, merge it back into the full manifest
        finalized = finalize_export_artifact(artifact_path, manifest_base, sidecars={})
        manifest_for_deploy = {**manifest_base, "artifact_sha256": finalized["artifact_sha256"]}
        # write_deployment_manifest passes manifest.get("config_hash") to build_deployment_manifest,
        # but manifest only has export_config_hash. Set config_hash explicitly so the deployment
        # manifest has the correct field for verify_deployment_manifest.
        manifest_for_deploy["config_hash"] = manifest_for_deploy["export_config_hash"]
        write_deployment_manifest(artifact_path, manifest_for_deploy)

        # Must not raise — deployment manifest is present and dataset_hash matches
        state_dict, loaded_features = load_artifact(artifact_path, features)
        assert isinstance(state_dict, dict)

    def test_deployment_manifest_tampering_rejected_on_artifact_load(self, tmp_path: Path):
        """Tampered deployment.manifest.json beside artifact must cause load failure."""
        valid_df = pd.read_csv(self._snapshot_path()).head(2)
        harmonized = harmonize_features(valid_df, create_cicids_mapping(), label_col="attack_type", mode="strict")
        features = harmonized[FEATURE_ORDER].astype(np.float32)

        from helix_ids.governance import (
            build_artifact_manifest,
            write_contract_sidecars,
            write_deployment_manifest,
        )
        from helix_ids.utils.export import finalize_export_artifact
        contract = {
            "schema_version": SCHEMA_VERSION,
            "schema_hash": compute_schema_hash(features),
            "feature_order": FEATURE_ORDER,
            "input_dim": len(FEATURE_ORDER),
            "binary_output_dim": 2,
            "family_output_dim": 7,
            "contract_version": CONTRACT_VERSION,
            "export_config_hash": "test_export_config",
        }
        manifest_base = build_artifact_manifest(
            model_architecture="unit-test",
            contract=contract,
            export_config={},
            dataset_hash="deadbeef",
            git_commit="abc1234",
            exporter_version="1.0.0",
            runtime_version="1.0.0",
        )
        artifact = {
            "model": {"w": torch.zeros((2, 2))},
            "artifact_manifest": manifest_base,
            **{k: contract[k] for k in (
                "schema_version", "schema_hash", "feature_order", "input_dim",
                "binary_output_dim", "family_output_dim", "contract_version",
            )},
        }
        artifact_path = tmp_path / "artifact_with_tampered_deploy.pt"
        torch.save(artifact, artifact_path)
        write_contract_sidecars(artifact_path, contract)
        finalized = finalize_export_artifact(artifact_path, manifest_base, sidecars={})
        manifest_for_deploy = {**manifest_base, "artifact_sha256": finalized["artifact_sha256"]}
        manifest_for_deploy["config_hash"] = manifest_for_deploy["export_config_hash"]
        write_deployment_manifest(artifact_path, manifest_for_deploy)

        # Tamper with the deployment manifest — change dataset_hash
        tampered = dict(manifest_for_deploy)
        tampered["dataset_hash"] = "tampered_value"
        write_deployment_manifest(artifact_path, tampered)

        from helix_ids.governance.provenance import ArtifactManifestError
        with pytest.raises(ArtifactManifestError):
            load_artifact(artifact_path, features)

    def test_normalize_column_name(self):
        """Column normalization should collapse spacing/case differences."""
        assert normalize_column_name(" Label ") == "label"
        assert normalize_column_name("Fwd_Pkt_Len_Mean") == "fwd pkt len mean"

    @staticmethod
    def _assert_invariant_feature_bounds(df: pd.DataFrame) -> None:
        bounded_01 = [
            "bytes_forward_ratio",
            "rst_fraction",
            "handshake_completion_rate",
            "fin_fraction",
            "connection_attempt_rate",
        ]
        bounded_m11 = [
            "bytes_asymmetry",
            "byte_direction_ratio",
            "packet_direction_ratio",
        ]
        binary_cols = [
            "proto_tcp",
            "proto_udp",
            "proto_icmp",
            "proto_other",
            "state_error_indicator",
            "state_reset_retrans_indicator",
        ]

        if "duration_log" in df.columns:
            assert (df["duration_log"] >= 0.0).all()
        if "total_bytes_log" in df.columns:
            assert (df["total_bytes_log"] >= 0.0).all()

        for col in bounded_01:
            if col not in df.columns:
                continue
            assert (df[col] >= 0.0).all(), f"{col} has values below 0"
            assert (df[col] <= 1.0).all(), f"{col} has values above 1"

        for col in bounded_m11:
            if col not in df.columns:
                continue
            assert (df[col] >= -1.0).all(), f"{col} has values below -1"
            assert (df[col] <= 1.0).all(), f"{col} has values above 1"

        for col in binary_cols:
            if col not in df.columns:
                continue
            assert set(np.unique(df[col].to_numpy())).issubset({0.0, 1.0})

    def test_nsl_harmonization_shape_order_and_bounds(self):
        """NSL harmonization must produce 19 invariant features in stable order."""
        loader = MultiDatasetLoader()
        df = pd.DataFrame(
            {
                "duration": [1.0, 4.0],
                "src_bytes": [10.0, 100.0],
                "dst_bytes": [5.0, 50.0],
                "protocol_type": ["tcp", "udp"],
                "service": ["http", "dns"],
                "flag": ["SF", "REJ"],
                "rerror_rate": [0.0, 0.2],
                "srv_rerror_rate": [0.0, 0.1],
                "dst_host_rerror_rate": [0.0, 0.1],
                "serror_rate": [0.0, 0.3],
                "srv_serror_rate": [0.0, 0.2],
                "dst_host_serror_rate": [0.0, 0.2],
                "count": [2.0, 10.0],
                "srv_count": [2.0, 3.0],
                "diff_srv_rate": [0.1, 0.6],
                "label": ["Normal", "DoS"],
            }
        )

        harmonized = loader.harmonize_nslkdd(df)
        assert harmonized.shape[1] == 18
        assert list(harmonized.columns[:-1]) == COMMON_FEATURES
        assert np.isfinite(harmonized[COMMON_FEATURES].to_numpy()).all()
        self._assert_invariant_feature_bounds(harmonized)

    def test_nsl_raw_attack_names_do_not_collapse_to_single_class(self):
        """Raw NSL labels (neptune/ipsweep/...) must map to multiple family indices."""
        loader = MultiDatasetLoader()
        df = pd.DataFrame(
            {
                "duration": [0.0, 1.0, 2.0, 3.0, 4.0],
                "src_bytes": [10.0, 20.0, 30.0, 40.0, 50.0],
                "dst_bytes": [1.0, 2.0, 3.0, 4.0, 5.0],
                "protocol_type": ["tcp", "tcp", "udp", "tcp", "icmp"],
                "service": ["http", "http", "dns", "ftp", "eco_i"],
                "flag": ["SF", "S0", "REJ", "SF", "RSTO"],
                "rerror_rate": [0.0, 0.1, 0.2, 0.0, 0.0],
                "srv_rerror_rate": [0.0, 0.1, 0.2, 0.0, 0.0],
                "dst_host_rerror_rate": [0.0, 0.1, 0.2, 0.0, 0.0],
                "serror_rate": [0.0, 0.2, 0.1, 0.0, 0.0],
                "srv_serror_rate": [0.0, 0.2, 0.1, 0.0, 0.0],
                "dst_host_serror_rate": [0.0, 0.2, 0.1, 0.0, 0.0],
                "count": [5.0, 10.0, 15.0, 20.0, 25.0],
                "srv_count": [2.0, 3.0, 4.0, 5.0, 6.0],
                "diff_srv_rate": [0.1, 0.2, 0.3, 0.4, 0.5],
                "label": ["normal", "neptune", "ipsweep", "warezclient", "buffer_overflow"],
            }
        )

        harmonized = loader.harmonize_nslkdd(df)
        mapped_labels = harmonized["label"].astype(int).tolist()

        assert set(mapped_labels) >= {0, 1, 2, 3, 4}

    def test_unsw_harmonization_emits_only_available_signal_columns(self):
        """UNSW harmonization should shrink to real columns instead of default-filling."""
        loader = MultiDatasetLoader()
        df = pd.DataFrame(
            {
                "dur": [0.1, 3.2],
                "sbytes": [12.0, 140.0],
                "dbytes": [6.0, 35.0],
                "proto": ["tcp", "icmp"],
                "service": ["http", "-"],
                "state": ["CON", "RST"],
                "ct_src_ltm": [3.0, 12.0],
                "ct_srv_src": [2.0, 6.0],
                "dsport": [80, 443],
                "Sintpkt": [0.02, 0.50],
                "Dintpkt": [0.01, 0.25],
                "Sjit": [0.01, 0.20],
                "Djit": [0.01, 0.20],
                "ct_src_dport_ltm": [1.0, 4.0],
                "Spkts": [10.0, 40.0],
                "Dpkts": [5.0, 30.0],
                "label": ["Normal", "Backdoors"],
            }
        )

        harmonized = loader.harmonize_unsw(df)
        assert "label" in harmonized.columns
        assert {"duration", "protocol_type", "src_bytes", "dst_bytes", "flag"}.issubset(
            set(harmonized.columns)
        )
        assert np.isfinite(harmonized.drop(columns=["label"]).to_numpy()).all()

    def test_unsw_connection_state_keeps_frozen_states(self):
        """FIN/INT/CON should remain canonical and never be remapped."""
        df = pd.DataFrame({"state": ["FIN", "INT", "CON", "fin", "int", "con"]})
        state = _derive_connection_state(df, "unsw_nb15")

        assert state.tolist() == ["FIN", "INT", "CON", "FIN", "INT", "CON"]

    def test_unsw_connection_state_maps_rare_ambiguous_to_oth(self):
        """Ambiguous UNSW states below 0.5% frequency should collapse to OTH."""
        states = ["con"] * 980 + ["acc"] * 10 + ["req"] * 4 + ["urn"] * 4 + ["fin"] * 2
        df = pd.DataFrame({"state": states})
        state = _derive_connection_state(df, "unsw_nb15")

        assert (state[df["state"] == "req"] == "OTH").all()
        assert (state[df["state"] == "urn"] == "OTH").all()
        assert (state[df["state"] == "acc"] == "S1").all()
        assert (state[df["state"] == "fin"] == "FIN").all()

    def test_unsw_discrete_probe_excludes_service_tier(self):
        """Probe should rely on transport/state drivers plus has_rst, not service_tier."""
        loader = MultiDatasetLoader()
        rng = np.random.default_rng(7)
        n = 120
        unsw_df = pd.DataFrame(
            {
                "protocol_type": np.r_[np.zeros(n // 2, dtype=np.int64), np.ones(n // 2, dtype=np.int64)],
                "connection_state": np.r_[np.zeros(n // 2, dtype=np.int64), np.ones(n // 2, dtype=np.int64)],
                "traffic_direction": np.r_[np.zeros(n // 2, dtype=np.int64), np.full(n // 2, 2, dtype=np.int64)],
                "has_rst": np.r_[np.zeros(n // 2, dtype=np.int64), np.ones(n // 2, dtype=np.int64)],
                "service_tier": rng.integers(0, 7, size=n, dtype=np.int64),
                "label": np.r_[np.zeros(n // 2, dtype=np.int64), np.ones(n // 2, dtype=np.int64)],
            }
        )

        probe_features, macro_f1 = loader._run_unsw_discrete_probe(
            unsw_df=unsw_df,
            available_features=[
                "protocol_type",
                "connection_state",
                "traffic_direction",
                "service_tier",
                "has_rst",
            ],
        )

        assert probe_features == ["protocol_type", "connection_state", "traffic_direction", "has_rst"]
        assert "service_tier" not in probe_features
        assert macro_f1 > UNSW_DISCRETE_PROBE_F1_MIN

    def test_unsw_discrete_probe_adaptive_threshold_allows_skewed_valid_signal(self, monkeypatch):
        """Skew-aware threshold should pass valid signal that beats baseline but is < absolute cap."""

        class _FakeSkewAwareLR:
            def __init__(self, *args, **kwargs):
                del args, kwargs
                self._y = None

            def fit(self, x, y):
                del x
                self._y = np.asarray(y, dtype=np.int64)
                return self

            def predict(self, x):
                del x
                y = np.asarray(self._y, dtype=np.int64)
                pred = np.zeros_like(y)
                pred[y == 1] = 1
                idx_class2 = np.nonzero(y == 2)[0]
                pred[idx_class2[:2]] = 2
                return pred

        monkeypatch.setattr(
            "src.helix_ids.data.multi_dataset_loader.LogisticRegression",
            _FakeSkewAwareLR,
        )

        loader = MultiDatasetLoader()
        y = np.array([0] * 70 + [1] * 5 + [2] * 5 + [3] * 5 + [4] * 5 + [5] * 5 + [6] * 5, dtype=np.int64)
        n = y.shape[0]
        unsw_df = pd.DataFrame(
            {
                "protocol_type": np.zeros(n, dtype=np.int64),
                "connection_state": np.zeros(n, dtype=np.int64),
                "traffic_direction": np.zeros(n, dtype=np.int64),
                "has_rst": np.zeros(n, dtype=np.int64),
                "service_tier": np.zeros(n, dtype=np.int64),
                "label": y,
            }
        )

        _, macro_f1 = loader._run_unsw_discrete_probe(
            unsw_df=unsw_df,
            available_features=[
                "protocol_type",
                "connection_state",
                "traffic_direction",
                "service_tier",
                "has_rst",
            ],
        )

        assert 0.30 < macro_f1 < UNSW_DISCRETE_PROBE_F1_MIN

    def test_unsw_discrete_probe_adaptive_threshold_still_blocks_low_signal(self, monkeypatch):
        """Adaptive threshold must still fail when probe carries no discriminative signal."""

        class _FakeMajorityLR:
            def __init__(self, *args, **kwargs):
                del args, kwargs
                self._y = None

            def fit(self, x, y):
                del x
                self._y = np.asarray(y, dtype=np.int64)
                return self

            def predict(self, x):
                del x
                y = np.asarray(self._y, dtype=np.int64)
                majority = int(np.argmax(np.bincount(y)))
                return np.full_like(y, fill_value=majority)

        monkeypatch.setattr(
            "src.helix_ids.data.multi_dataset_loader.LogisticRegression",
            _FakeMajorityLR,
        )

        loader = MultiDatasetLoader()
        y = np.array([0] * 70 + [1] * 5 + [2] * 5 + [3] * 5 + [4] * 5 + [5] * 5 + [6] * 5, dtype=np.int64)
        n = y.shape[0]
        unsw_df = pd.DataFrame(
            {
                "protocol_type": np.zeros(n, dtype=np.int64),
                "connection_state": np.zeros(n, dtype=np.int64),
                "traffic_direction": np.zeros(n, dtype=np.int64),
                "has_rst": np.zeros(n, dtype=np.int64),
                "service_tier": np.zeros(n, dtype=np.int64),
                "label": y,
            }
        )

        with pytest.raises(RuntimeError, match="UNSW discrete separability probe failed"):
            loader._run_unsw_discrete_probe(
                unsw_df=unsw_df,
                available_features=[
                    "protocol_type",
                    "connection_state",
                    "traffic_direction",
                    "service_tier",
                    "has_rst",
                ],
            )

    def test_cicids_harmonization_shape_order_and_bounds(self):
        """CICIDS harmonization must produce 19 invariant features in stable order."""
        loader = MultiDatasetLoader()
        df = pd.DataFrame(
            {
                "Flow Duration": [1000.0, 2000.0],
                "TotLen Fwd Pkts": [100.0, 350.0],
                "TotLen Bwd Pkts": [50.0, 20.0],
                "Protocol": [6, 17],
                "SYN Flag Cnt": [6.0, 1.0],
                "RST Flag Cnt": [0.0, 1.0],
                "ACK Flag Cnt": [5.0, 1.0],
                "Tot Fwd Pkts": [10.0, 15.0],
                "Dst Port": [80, 443],
                "Flow IAT Mean": [100.0, 300.0],
                "Fwd IAT Mean": [80.0, 220.0],
                "Bwd IAT Mean": [70.0, 190.0],
                "Fwd IAT Max": [150.0, 320.0],
                "Bwd IAT Max": [130.0, 300.0],
                "Fwd IAT Min": [40.0, 90.0],
                "Bwd IAT Min": [35.0, 80.0],
                "Tot Bwd Pkts": [8.0, 4.0],
                "Active Mean": [50.0, 30.0],
                "Label": ["BENIGN", "DDoS"],
            }
        )

        harmonized = loader.harmonize_cicids(df)
        assert harmonized.shape[1] == 18
        assert list(harmonized.columns[:-1]) == COMMON_FEATURES
        assert np.isfinite(harmonized[COMMON_FEATURES].to_numpy()).all()
        self._assert_invariant_feature_bounds(harmonized)

    def test_loader_exposes_no_normalization_surface(self):
        """Loader must not expose dataset transformation APIs."""
        loader = MultiDatasetLoader()
        assert not hasattr(loader, "normalize_per_dataset")

    def test_sanitization_stability(self):
        """Centralized numeric sanitization should keep numeric columns finite."""
        pytest.importorskip("hypothesis")
        from hypothesis import given, settings
        from hypothesis import strategies as st

        from src.helix_ids.data.feature_harmonization import sanitize_numeric

        numeric_value = st.one_of(
            st.floats(allow_nan=True, allow_infinity=True, width=32),
            st.integers(min_value=-1000, max_value=1000),
        )

        @settings(max_examples=25, deadline=None)
        @given(
            st.data(),
        )
        def _inner(data):
            size = data.draw(st.integers(min_value=1, max_value=8))
            df = pd.DataFrame(
                {
                    "a": [data.draw(numeric_value) for _ in range(size)],
                    "b": [data.draw(numeric_value) for _ in range(size)],
                }
            )
            df = sanitize_numeric(df)
            assert np.isfinite(df.select_dtypes(include=[np.number])).all().all()

        _inner()


class TestMultiDatasetLoader:
    """Test multi-dataset loader."""

    def test_loader_initialization(self):
        """Test loader initialization."""
        loader = MultiDatasetLoader()
        assert loader.project_root.exists()
        assert loader.data_dir.exists()

    def test_load_nslkdd(self):
        """Test loading NSL-KDD dataset."""
        loader = MultiDatasetLoader()
        try:
            df = loader.load_nslkdd()
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            print(f"✅ NSL-KDD loaded: {df.shape}")
        except FileNotFoundError as e:
            print(f"⚠️ NSL-KDD not found: {e}")

    def test_load_unsw(self):
        """Test loading UNSW-NB15 dataset."""
        loader = MultiDatasetLoader()
        try:
            df = loader.load_unsw()
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            print(f"✅ UNSW-NB15 loaded: {df.shape}")
        except FileNotFoundError as e:
            print(f"⚠️ UNSW-NB15 not found: {e}")

    def test_harmonize_nslkdd(self):
        """Test NSL-KDD harmonization."""
        loader = MultiDatasetLoader()
        try:
            df = loader.load_nslkdd()
            harmonized = loader.harmonize_nslkdd(df)
            assert "label" in harmonized.columns
            assert len(harmonized.columns) == len(COMMON_FEATURES) + 1
            print(f"✅ NSL-KDD harmonized: {harmonized.shape}")
        except FileNotFoundError:
            pytest.skip("NSL-KDD not found")

    def test_create_splits_preserves_unscaled_feature_range(self):
        """Split creation should not normalize features in-loader."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            feat: rng.uniform(0, 100, 100)
            for feat in COMMON_FEATURES
        })
        df["label"] = rng.integers(0, 7, 100)

        loader = MultiDatasetLoader()
        splits = loader.create_splits([df])

        train_x = splits["X_train"]
        assert train_x.shape[1] == len(INVARIANT_FEATURES)
        assert np.isfinite(train_x).all()
        assert float(train_x.max()) > 1.0

    def test_create_splits(self):
        """Test split creation."""
        # Create synthetic data
        rng = np.random.default_rng(123)
        df = pd.DataFrame({
            feat: rng.uniform(0, 100, 100)
            for feat in COMMON_FEATURES
        })
        df["label"] = rng.integers(0, 7, 100)

        loader = MultiDatasetLoader()
        splits = loader.create_splits([df])

        assert "X_train" in splits
        assert "y_train" in splits
        assert "X_val" in splits
        assert "y_val" in splits
        assert "X_test_nsl_kdd" in splits
        assert "y_test_nsl_kdd" in splits

        # Verify shapes
        assert splits["X_train"].shape[0] > 0
        assert splits["X_train"].shape[1] == len(INVARIANT_FEATURES)
        assert len(splits["y_train"]) == splits["X_train"].shape[0]
        assert "train_class_weights" in splits
        assert splits["train_class_weights"].ndim == 1
        assert "X_val_nsl_kdd" in splits
        assert "X_test_nsl_kdd" in splits

        print("✅ Splits created correctly")

    def test_clean_cicids_frame(self):
        """CICIDS cleaner should strip labels and preserve NaNs for split-time imputation."""
        loader = MultiDatasetLoader()
        dirty = pd.DataFrame(
            {
                " Flow Duration ": [1.0, np.inf],
                "TotLen Fwd Pkts": [np.nan, 2.0],
                " Label ": [" Benign", " DDoS  "],
            }
        )

        cleaned = loader._clean_cicids_frame(dirty)
        assert "attack_type" in cleaned.columns
        assert cleaned["attack_type"].tolist() == ["Benign", "DDoS"]
        numeric = cleaned.drop(columns=["attack_type"])
        assert not np.isinf(numeric.values).any()
        assert np.isnan(numeric.values).any()

    def test_scale_dataset_features_handles_empty_train_holdout(self):
        """Test-only holdout split must remain finite when train rows are empty."""
        loader = MultiDatasetLoader()
        feature_columns = ["duration", "src_bytes", "dst_bytes", "logged_in", "same_srv_rate"]

        x_train = np.empty((0, len(feature_columns)), dtype=np.float32)
        x_val = np.empty((0, len(feature_columns)), dtype=np.float32)
        x_test = np.array(
            [
                [1.0, 1000.0, 100.0, 1.0, 0.5],
                [2.0, 2000.0, np.nan, 0.0, np.inf],
            ],
            dtype=np.float32,
        )

        scaled_train, scaled_val, scaled_test = loader._scale_dataset_features(
            x_train=x_train,
            x_val=x_val,
            x_test=x_test,
            feature_columns=feature_columns,
        )

        assert scaled_train.shape == x_train.shape
        assert scaled_val.shape == x_val.shape
        assert scaled_test.shape == x_test.shape
        assert np.isfinite(scaled_test).all()


if __name__ == "__main__":
    # Run quick validation
    pytest.main([__file__, "-v"])
