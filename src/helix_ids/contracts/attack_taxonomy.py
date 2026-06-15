"""
Canonical attack taxonomy for HELIX-IDS.

This is the SINGLE authoritative source for all attack family definitions,
label mappings, threat weights, and lookup helpers. Every other module in the
codebase MUST import from here rather than redefining its own mapping tables.

Maintainers: If you add a new dataset or classification scheme, add the
mapping here and re-export from the appropriate callers. Do NOT define
parallel mappings in consumer modules.

Constants
---------
ATTACK_FAMILIES : tuple[str, ...]
    Ordered 5-class family names (Normal, DoS, Probe, R2L, U2R).
THREAT_WEIGHTS : dict[str, float]
    Family-name → weight for threat-aware loss functions.
FAMILY_TO_INDEX : dict[str, int]
    5-class family name → integer index (Normal=0 … U2R=4).
FAMILY_INDEX_TO_NAME : dict[int, str]
    Reverse of FAMILY_TO_INDEX.
NSL_KDD_ATTACK_MAPPING : dict[str, str]
    Raw NSL-KDD attack name → family name.
UNSW_TO_UNIFIED_5CLASS : dict[str, str]
    Raw UNSW-NB15 attack name → family name.
CICIDS_TO_UNIFIED_5CLASS : dict[str, str]
    Raw CICIDS attack name → family name.

Functions
---------
threat_weight_tensor(order=None)
    Return threat weights as a float Tensor in family order.
resolve_family(dataset_name, attack_type)
    Map a raw attack label to its canonical family name.
validate_family(name)
    Raise ValueError if *name* is not a recognised family.
get_all_family_names()
    Return the full list of canonical 5-class family names.
"""

from __future__ import annotations

from typing import Any

# ============================================================================
# Canonical 5-class family definitions
# ============================================================================

ATTACK_FAMILIES: tuple[str, ...] = ("Normal", "DoS", "Probe", "R2L", "U2R")

# Canonical 5-class ordered list (backward-compat with UNIFIED_5CLASS name)
UNIFIED_5CLASS: list[str] = list(ATTACK_FAMILIES)

FAMILY_TO_INDEX: dict[str, int] = {
    "Normal": 0,
    "DoS": 1,
    "Probe": 2,
    "R2L": 3,
    "U2R": 4,
}

FAMILY_INDEX_TO_NAME: dict[int, str] = {v: k for k, v in FAMILY_TO_INDEX.items()}

# ============================================================================
# Threat weights (5-class, conservative — prevents gradient collapse)
# ============================================================================

THREAT_WEIGHTS: dict[str, float] = {
    "Normal": 1.0,
    "DoS": 1.2,
    "Probe": 1.5,
    "R2L": 3.0,
    "U2R": 4.0,
}


def threat_weight_tensor() -> Any:
    """
    Return threat weights as a PyTorch float tensor in family order.

    Order: Normal=0, DoS=1, Probe=2, R2L=3, U2R=4.
    """
    import torch

    values = [THREAT_WEIGHTS[f] for f in ATTACK_FAMILIES]
    return torch.tensor(values, dtype=torch.float32)


# ============================================================================
# Per-dataset raw-attack → family mappings
# ============================================================================

# ── NSL-KDD ────────────────────────────────────────────────────────────────

NSL_KDD_ATTACK_MAPPING: dict[str, str] = {
    "normal": "Normal",
    # DoS
    "back": "DoS",
    "land": "DoS",
    "neptune": "DoS",
    "pod": "DoS",
    "smurf": "DoS",
    "teardrop": "DoS",
    "mailbomb": "DoS",
    "apache2": "DoS",
    "processtable": "DoS",
    "udpstorm": "DoS",
    # Probe
    "ipsweep": "Probe",
    "nmap": "Probe",
    "portsweep": "Probe",
    "satan": "Probe",
    "mscan": "Probe",
    "saint": "Probe",
    # R2L
    "ftp_write": "R2L",
    "guess_passwd": "R2L",
    "imap": "R2L",
    "multihop": "R2L",
    "phf": "R2L",
    "spy": "R2L",
    "warezclient": "R2L",
    "warezmaster": "R2L",
    "sendmail": "R2L",
    "named": "R2L",
    "snmpgetattack": "R2L",
    "snmpguess": "R2L",
    "xlock": "R2L",
    "xsnoop": "R2L",
    "worm": "R2L",
    # U2R
    "buffer_overflow": "U2R",
    "loadmodule": "U2R",
    "perl": "U2R",
    "rootkit": "U2R",
    "httptunnel": "U2R",
    "ps": "U2R",
    "sqlattack": "U2R",
    "xterm": "U2R",
}

# ── UNSW-NB15 → 5-class ───────────────────────────────────────────────────

UNSW_TO_UNIFIED_5CLASS: dict[str, str] = {
    "normal": "Normal",
    "analysis": "Probe",
    "backdoor": "R2L",
    "dos": "DoS",
    "exploits": "R2L",
    "fuzzers": "Probe",
    "generic": "DoS",
    "reconnaissance": "Probe",
    "shellcode": "U2R",
    "worms": "R2L",
}

# ── CICIDS → 5-class ──────────────────────────────────────────────────────

CICIDS_TO_UNIFIED_5CLASS: dict[str, str] = {
    "benign": "Normal",
    "ddos": "DoS",
    "dos goldeneye": "DoS",
    "dos hulk": "DoS",
    "dos slowhttptest": "DoS",
    "dos slowloris": "DoS",
    "portscan": "Probe",
    "bot": "R2L",
    "ftp-patator": "R2L",
    "ssh-patator": "R2L",
    "infiltration": "R2L",
    "heartbleed": "R2L",
    "web attack - brute force": "R2L",
    "web attack - sql injection": "R2L",
    "web attack - xss": "R2L",
}

# ============================================================================
# Per-dataset raw-name → canonical-name (for 7-class export schema)
# ============================================================================

# The 7-class taxonomy used by the unified model / ONNX export.
HELIX_CLASSES: list[str] = [
    "Normal",
    "DoS",
    "Probe",
    "R2L",
    "U2R",
    "Generic",
    "Backdoor",
]

HELIX_CLASS_TO_INDEX: dict[str, int] = {
    name: idx for idx, name in enumerate(HELIX_CLASSES)
}

# 7-class index → name mapping (backward compat with ATTACK_TAXONOMY_7CLASS name)
ATTACK_TAXONOMY_7CLASS: dict[int, str] = dict(enumerate(HELIX_CLASSES))

# 7-class threat weights (separate — used only for 7-class export metadata)
# These intentionally differ from the 5-class training weights because the
# combined model has a different classification schema.
SEVEN_CLASS_THREAT_WEIGHTS: dict[str, float] = {
    "Normal": 1.0,
    "DoS": 2.0,
    "Probe": 2.5,
    "R2L": 4.0,
    "U2R": 5.0,
    "Generic": 3.5,
    "Backdoor": 5.5,
}

# 5-class → 7-class index shift (7-class: Generic=5, Backdoor=6)
_7CLASS_FAMILY_PREFIX_LEN = 5  # Normal, DoS, Probe, R2L, U2R overlap

# ── 7-class per-dataset mappings ──────────────────────────────────────────

NSLKDD_TO_7CLASS: dict[str, int] = {
    "Normal": 0,
    "DoS": 1,
    "Probe": 2,
    "R2L": 3,
    "U2R": 4,
}

UNSW_TO_7CLASS: dict[str, int] = {
    "Normal": 0,
    "Analysis": 2,
    "Backdoors": 6,
    "DoS": 1,
    "Exploits": 3,
    "Fuzzers": 2,
    "Generic": 5,
    "Reconnaissance": 2,
    "Shellcode": 4,
    "Worms": 6,
}

CICIDS_TO_7CLASS: dict[str, int] = {
    "BENIGN": 0,
    "DDoS": 1,
    "DoS GoldenEye": 1,
    "DoS Hulk": 1,
    "DoS slowloris": 1,
    "DoS Slowhttptest": 1,
    "PortScan": 2,
    "Bot": 5,
    "Infiltration": 5,
    "Web Attack - Brute Force": 3,
    "Web Attack - XSS": 3,
    "Web Attack - Sql Injection": 3,
    "FTP-Patator": 3,
    "SSH-Patator": 3,
    "Heartbleed": 4,
}

CICIDS2018_TO_7CLASS: dict[str, int] = {
    "BENIGN": 0,
    "DDoS": 1,
    "DoS": 1,
    "PortScan": 2,
    "Bot": 5,
    "Infiltration": 5,
    "Brute Force": 3,
    "SQL Injection": 3,
    "SSH-Patator": 3,
    "FTP-Patator": 3,
}

# ============================================================================
# Helper functions
# ============================================================================


def get_all_family_names() -> list[str]:
    """Return the full list of canonical 5-class family names."""
    return list(ATTACK_FAMILIES)


def validate_family(name: str) -> None:
    """Raise ValueError if *name* is not a recognised attack family."""
    if name not in FAMILY_TO_INDEX:
        valid = ", ".join(ATTACK_FAMILIES)
        raise ValueError(f"Unknown attack family '{name}'. Valid: {valid}")


def resolve_family(dataset_name: str, attack_type: str) -> str | None:
    """
    Map a raw attack label to its canonical family name.

    Parameters
    ----------
    dataset_name : str
        One of ``"nsl-kdd"``, ``"unsw-nb15"``, ``"cicids"`` (or related
        variants such as ``"nsl_kdd"``, ``"cicids-2018"``).
    attack_type : str
        Raw attack label as it appears in the dataset.

    Returns
    -------
    str or None
        Canonical family name, or *None* if no mapping exists.
    """
    key = attack_type.lower().strip()
    ds = dataset_name.lower().replace("_", "-").split("-")[0]

    if ds in ("nsl", "nsl-kdd", "nsl_kdd"):
        return NSL_KDD_ATTACK_MAPPING.get(key)
    elif ds in ("unsw", "unsw-nb15"):
        return UNSW_TO_UNIFIED_5CLASS.get(key)
    elif ds in ("cicids", "cicids-2018", "cicids-2017"):
        return CICIDS_TO_UNIFIED_5CLASS.get(key)

    return None


# ============================================================================
# Validation helpers
# ============================================================================


def assert_mapping_covers(
    mapping: dict[str, str | int],
    expected_families: set[str],
    label: str = "mapping",
) -> None:
    """
    Assert that *mapping* values are all valid family classes.

    Raises
    ------
    AssertionError
        If any value is not in *expected_families*.
    """
    for raw, family in mapping.items():
        if isinstance(family, int):
            if family >= len(ATTACK_FAMILIES):
                pass  # 7-class indices are handled separately
        elif isinstance(family, str):
            assert family in expected_families, (
                f"{label}[{raw!r}] = {family!r} is not a valid family "
                f"(expected one of {expected_families})"
            )


__all__ = [
    # 5-class
    "ATTACK_FAMILIES",
    "UNIFIED_5CLASS",
    "FAMILY_TO_INDEX",
    "FAMILY_INDEX_TO_NAME",
    "THREAT_WEIGHTS",
    "threat_weight_tensor",
    # Per-dataset raw → 5-class
    "NSL_KDD_ATTACK_MAPPING",
    "UNSW_TO_UNIFIED_5CLASS",
    "CICIDS_TO_UNIFIED_5CLASS",
    # 7-class (export schema)
    "HELIX_CLASSES",
    "HELIX_CLASS_TO_INDEX",
    "ATTACK_TAXONOMY_7CLASS",
    "SEVEN_CLASS_THREAT_WEIGHTS",
    "NSLKDD_TO_7CLASS",
    "UNSW_TO_7CLASS",
    "CICIDS_TO_7CLASS",
    "CICIDS2018_TO_7CLASS",
    # Helpers
    "get_all_family_names",
    "validate_family",
    "resolve_family",
    "assert_mapping_covers",
]
