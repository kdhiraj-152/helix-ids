"""
Regression tests for the canonical attack taxonomy contract.

These tests enforce that:
1. No module defines duplicate taxonomy definitions (must import from canonical).
2. All mapping sources resolve identically.
3. The canonical module is self-consistent.
4. All existing dataset labels can be mapped through the canonical contract.
"""

import ast
import os
from pathlib import Path

import pytest

from helix_ids.contracts.attack_taxonomy import (
    ATTACK_FAMILIES,
    ATTACK_TAXONOMY_7CLASS,
    CICIDS_TO_7CLASS,
    CICIDS_TO_UNIFIED_5CLASS,
    FAMILY_INDEX_TO_NAME,
    FAMILY_TO_INDEX,
    HELIX_CLASSES,
    NSL_KDD_ATTACK_MAPPING,
    NSLKDD_TO_7CLASS,
    SEVEN_CLASS_THREAT_WEIGHTS,
    THREAT_WEIGHTS,
    UNSW_TO_UNIFIED_5CLASS,
    threat_weight_tensor,
)

# ============================================================================
# Test 1: No duplicate taxonomy definitions
# ============================================================================

# Literal mapping symbols whose redefinition as literals is forbidden
# outside the canonical module.
_FORBIDDEN_LITERALS = {
    "NSL_KDD_ATTACK_MAPPING",
    "UNSW_TO_UNIFIED_5CLASS",
    "CICIDS_TO_UNIFIED_5CLASS",
    "ATTACK_TAXONOMY_7CLASS",
    "NSLKDD_TO_7CLASS",
    "UNSW_TO_7CLASS",
    "CICIDS_TO_7CLASS",
    "CICIDS2018_TO_7CLASS",
    "THREAT_WEIGHTS",
}

_CANONICAL_MODULE = "helix_ids.contracts.attack_taxonomy"

# Modules that are allowed to define these symbols (e.g. dataset_config for
# backward-compatible re-exports). The canonical module itself is always
# allowed. Re-export aliases (name=imported_name) are NOT considered literals.
_ALLOWED_DEFINITIONS: set[tuple[str, str]] = {
    (_CANONICAL_MODULE, s) for s in _FORBIDDEN_LITERALS
}
# dataset_config.py re-exports these as aliases — not literal redefinitions
_ALLOWED_DEFINITIONS.update({
    ("helix_ids.data.dataset_config", "NSL_KDD_ATTACK_MAPPING"),
    ("helix_ids.data.dataset_config", "UNSW_TO_UNIFIED_5CLASS"),
    ("helix_ids.data.dataset_config", "CICIDS_TO_UNIFIED_5CLASS"),
    ("helix_ids.data.dataset_config", "UNIFIED_5CLASS"),
})
# export.py re-exports HELIX_CLASSES and DEFAULT_THREAT_WEIGHTS as aliases
_ALLOWED_DEFINITIONS.update({
    ("helix_ids.utils.export", "HELIX_CLASSES"),
})
# loss.py is allowed DEFAULT_THREAT_WEIGHTS (tensor, different type)
_ALLOWED_DEFINITIONS.add(("helix_ids.models.loss", "DEFAULT_THREAT_WEIGHTS"))
# utils/export.py is allowed DEFAULT_THREAT_WEIGHTS (7-class dict for export)
_ALLOWED_DEFINITIONS.add(("helix_ids.utils.export", "DEFAULT_THREAT_WEIGHTS"))


def _is_literal_definition(node) -> bool:
    """Return True if *node* is a literal dict definition (not an import
    alias, function call, or attribute reference)."""
    if isinstance(node, ast.Dict):
        return True
    # torch.tensor(...) is also a literal for threat_weight purposes
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr == "tensor":
            return True
    return False


def test_no_duplicate_taxonomy_definitions():
    """
    Fail if any module redefines a taxonomy mapping as a dict literal
    instead of importing from the canonical source.
    """
    src_root = Path(__file__).parent.parent / "src"
    offenders = []

    for root, dirs, files in os.walk(src_root / "helix_ids"):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in files:
            if not f.endswith(".py"):
                continue
            path = os.path.join(root, f)
            rel = os.path.relpath(path, src_root).replace(os.sep, "/").replace(".py", "")
            if rel.endswith("/__init__"):
                rel = rel[: -len("/__init__")] or "helix_ids"
            mod = rel.replace("/", ".")

            with open(path) as fh:
                source = fh.read()
            tree = ast.parse(source)

            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id in _FORBIDDEN_LITERALS:
                            # Name aliases (X = Y) don't create duplicates
                            if isinstance(node.value, ast.Name) or isinstance(node.value, ast.Attribute):
                                continue
                            if _is_literal_definition(node.value):
                                offenders.append((mod, target.id, node.lineno))

    unexpected = [
        (m, s, ln)
        for (m, s, ln) in offenders
        if (m, s) not in _ALLOWED_DEFINITIONS
    ]

    if unexpected:
        lines = [
            f"Found {len(unexpected)} non-canonical taxonomy definition(s):",
        ]
        for mod, sym, ln in unexpected:
            lines.append(f"  {mod}:{ln} — defines {sym} as a literal")
        lines.append(
            f"\nCanonical source is {_CANONICAL_MODULE}. "
            "Import instead of redefining."
        )
        pytest.fail("\n".join(lines))


# ============================================================================
# Test 2: Canonical module self-consistency
# ============================================================================


class TestCanonicalConsistency:
    """Verify all constants in the canonical module are self-consistent."""

    def test_attack_families_tuple_and_list(self):
        assert len(ATTACK_FAMILIES) == 5
        # UNIFIED_5CLASS compatibility alias
        from helix_ids.contracts.attack_taxonomy import UNIFIED_5CLASS
        assert list(ATTACK_FAMILIES) == UNIFIED_5CLASS

    def test_family_to_index_consistency(self):
        assert len(FAMILY_TO_INDEX) == 5
        assert FAMILY_TO_INDEX["Normal"] == 0
        assert FAMILY_TO_INDEX["DoS"] == 1
        assert FAMILY_TO_INDEX["Probe"] == 2
        assert FAMILY_TO_INDEX["R2L"] == 3
        assert FAMILY_TO_INDEX["U2R"] == 4

    def test_family_index_to_name_consistency(self):
        assert all(FAMILY_INDEX_TO_NAME[i] == name for name, i in FAMILY_TO_INDEX.items())
        assert len(FAMILY_INDEX_TO_NAME) == 5

    def test_threat_weights_matches_families(self):
        assert set(THREAT_WEIGHTS.keys()) == set(ATTACK_FAMILIES)
        assert THREAT_WEIGHTS["Normal"] < THREAT_WEIGHTS["DoS"]
        assert THREAT_WEIGHTS["DoS"] < THREAT_WEIGHTS["Probe"]
        assert THREAT_WEIGHTS["Probe"] < THREAT_WEIGHTS["R2L"]
        assert THREAT_WEIGHTS["R2L"] < THREAT_WEIGHTS["U2R"]

    def test_threat_weight_tensor_consistency(self):
        t = threat_weight_tensor()
        assert t.shape == (5,)
        for i, fam in enumerate(ATTACK_FAMILIES):
            assert abs(t[i].item() - THREAT_WEIGHTS[fam]) < 1e-5

    def test_helix_classes_7class(self):
        assert len(HELIX_CLASSES) == 7
        assert HELIX_CLASSES[:5] == list(ATTACK_FAMILIES)
        assert HELIX_CLASSES[5] == "Generic"
        assert HELIX_CLASSES[6] == "Backdoor"

    def test_attack_taxonomy_7class(self):
        assert len(ATTACK_TAXONOMY_7CLASS) == 7
        for idx, name in ATTACK_TAXONOMY_7CLASS.items():
            assert isinstance(idx, int)
            assert 0 <= idx <= 6
            assert HELIX_CLASSES[idx] == name


# ============================================================================
# Test 3: All mappings resolve to valid families
# ============================================================================


class TestMappingValues:
    """Verify all mapping targets are valid family classes."""

    def test_nsl_kdd_mapping_values(self):
        valid_families = set(ATTACK_FAMILIES)
        for raw, family in NSL_KDD_ATTACK_MAPPING.items():
            assert family in valid_families, (
                f"NSL-KDD mapping {raw!r} -> {family!r} is not a valid family"
            )

    def test_unsw_mapping_values(self):
        valid_families = set(ATTACK_FAMILIES)
        for raw, family in UNSW_TO_UNIFIED_5CLASS.items():
            assert family in valid_families, (
                f"UNSW mapping {raw!r} -> {family!r} is not a valid family"
            )

    def test_cicids_mapping_values(self):
        valid_families = set(ATTACK_FAMILIES)
        for raw, family in CICIDS_TO_UNIFIED_5CLASS.items():
            assert family in valid_families, (
                f"CICIDS mapping {raw!r} -> {family!r} is not a valid family"
            )

    def test_seven_class_threat_weights_keys(self):
        assert set(SEVEN_CLASS_THREAT_WEIGHTS.keys()) == set(HELIX_CLASSES)
        for cls_name in HELIX_CLASSES:
            assert cls_name in SEVEN_CLASS_THREAT_WEIGHTS


# ============================================================================
# Test 4: All mappings resolve identically across sources
# ============================================================================


class test_ExportDatasetMaps:
    """Verify 7-class per-dataset maps agree with 5-class maps."""

    def test_nslkdd_to_7class_has_all_5class_keys(self):
        for fam in ATTACK_FAMILIES:
            if fam == "Normal":
                assert NSLKDD_TO_7CLASS["Normal"] == 0
            elif fam == "DoS":
                assert NSLKDD_TO_7CLASS["DoS"] == 1
            elif fam == "Probe":
                assert NSLKDD_TO_7CLASS["Probe"] == 2
            elif fam == "R2L":
                assert NSLKDD_TO_7CLASS["R2L"] == 3
            elif fam == "U2R":
                assert NSLKDD_TO_7CLASS["U2R"] == 4

    def test_cicids_to_7class_heartbleed_maps_to_u2r(self):
        # CICIDS maps "Heartbleed" to class 4 (U2R) in 7-class
        assert CICIDS_TO_7CLASS.get("Heartbleed") == 4


# ============================================================================
# Test 5: Existing datasets still map correctly
# ============================================================================


class test_DatasetLabelCoverage:
    """Prove that all existing dataset labels survive the mapping."""

    def test_nsl_kdd_representative_labels(self):
        """Canonical NSL-KDD map must include all known attack labels."""
        known = {
            "normal", "back", "neptune", "smurf", "teardrop",
            "ipsweep", "nmap", "portsweep", "satan",
            "ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy",
            "buffer_overflow", "loadmodule", "perl", "rootkit",
        }
        for label in known:
            assert label in NSL_KDD_ATTACK_MAPPING, (
                f"NSL-KDD label {label!r} missing from canonical mapping"
            )

    def test_unsw_representative_labels(self):
        known = {
            "normal", "analysis", "backdoor", "dos", "exploits",
            "fuzzers", "generic", "reconnaissance", "shellcode", "worms",
        }
        for label in known:
            assert label in UNSW_TO_UNIFIED_5CLASS, (
                f"UNSW label {label!r} missing from canonical mapping"
            )

    def test_cicids_representative_labels(self):
        known = {
            "benign", "ddos", "dos goldeneye", "dos hulk",
            "dos slowhttptest", "dos slowloris", "portscan",
            "bot", "ftp-patator", "ssh-patator", "infiltration",
            "heartbleed",
        }
        for label in known:
            assert label in CICIDS_TO_UNIFIED_5CLASS, (
                f"CICIDS label {label!r} missing from canonical mapping"
            )
