"""
Dataset configurations and attack-type label mappings for HELIX-IDS.

Extracted from unified_loader.py (was >1230 lines).
This module owns all dataset metadata — paths, class names, column names.

Canonical attack taxonomy mappings are imported from
:mod:`helix_ids.contracts.attack_taxonomy` and re-exported for backward
compatibility.
"""

from dataclasses import dataclass, field
from pathlib import Path

from helix_ids.contracts.attack_taxonomy import (
    CICIDS_TO_UNIFIED_5CLASS,
    NSL_KDD_ATTACK_MAPPING,
    UNIFIED_5CLASS,
    UNSW_TO_UNIFIED_5CLASS,
)

# Backward-compatible re-exports (originally defined here, now canonical)
UNIFIED_5CLASS: list[str] = UNIFIED_5CLASS
NSL_KDD_ATTACK_MAPPING: dict[str, str] = NSL_KDD_ATTACK_MAPPING
UNSW_TO_UNIFIED_5CLASS: dict[str, str] = UNSW_TO_UNIFIED_5CLASS
CICIDS_TO_UNIFIED_5CLASS: dict[str, str] = CICIDS_TO_UNIFIED_5CLASS

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent  # src/helix_ids/data -> RP-2
DATA_DIR = PROJECT_ROOT / "data"
ARCHIVE_DIR = PROJECT_ROOT / "archive"
ARCHIVE2_DIR = PROJECT_ROOT / "archive-2"
ARCHIVE3_DIR = PROJECT_ROOT / "archive-3"

# ============================================================================
# Column exclusions
# ============================================================================

# CICFlowMeter drops include identifier columns not used as model features
CICIDS_EXCLUDE_COLUMNS = {"Flow ID", "Src IP", "Dst IP", "Src Port"}

CSV_PATTERN = "*.csv"

# ============================================================================
# Attack-type label mappings (imported from canonical)
# ============================================================================

# ── CICIDS constants used by the mappings below ───────────────────────────

WEBATTACK_BRUTEFORCE = "Web Attack - Brute Force"
WEBATTACK_XSS = "Web Attack - XSS"
WEBATTACK_SQLINJECTION = "Web Attack - Sql Injection"
DOS_GOLDENEYE = "DoS GoldenEye"
DOS_HULK = "DoS Hulk"
DOS_SLOWLORIS = "DoS slowloris"
DOS_SLOWHTTPTEST = "DoS Slowhttptest"

CICIDS_LABEL_MAPPING: dict[str, str] = {
    "benign": "BENIGN",
    "ddos": "DDoS",
    "portscan": "PortScan",
    "bot": "Bot",
    "infiltration": "Infiltration",
    "web attack \u2013 brute force": WEBATTACK_BRUTEFORCE,
    "web attack \u2013 xss": WEBATTACK_XSS,
    "web attack \u2013 sql injection": WEBATTACK_SQLINJECTION,
    "web attack  brute force": WEBATTACK_BRUTEFORCE,
    "web attack  xss": WEBATTACK_XSS,
    "web attack  sql injection": WEBATTACK_SQLINJECTION,
    "ftp-patator": "FTP-Patator",
    "ssh-patator": "SSH-Patator",
    "dos hulk": DOS_HULK,
    "dos goldeneye": DOS_GOLDENEYE,
    "dos slowloris": DOS_SLOWLORIS,
    "dos slowhttptest": DOS_SLOWHTTPTEST,
    "heartbleed": "Heartbleed",
}

CICIDS_2018_LABEL_MAPPING: dict[str, str] = {
    "benign": "BENIGN",
    "ddos attack-hoic": "DDoS",
    "ddos attack-loic-udp": "DDoS",
    "ddos attacks-loic-http": "DDoS",
    "dos attacks-goldeneye": DOS_GOLDENEYE,
    "dos attacks-hulk": DOS_HULK,
    "dos attacks-slowhttptest": DOS_SLOWHTTPTEST,
    "dos attacks-slowloris": DOS_SLOWLORIS,
    "ftp-bruteforce": "FTP-Patator",
    "ssh-bruteforce": "SSH-Patator",
    "brute force -web": WEBATTACK_BRUTEFORCE,
    "brute force -xss": WEBATTACK_XSS,
    "sql injection": WEBATTACK_SQLINJECTION,
    "bot": "Bot",
    "infilteration": "Infiltration",
    "infiltration": "Infiltration",
}

# ============================================================================
# NSL-KDD column names (for headerless TXT/ARFF files)
# ============================================================================

NSL_KDD_FEATURE_NAMES = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

# ============================================================================
# Dataset configurations
# ============================================================================


@dataclass
class DatasetConfig:
    """Configuration for a specific dataset."""

    name: str
    class_names: list[str]
    label_column: str
    feature_count: int
    categorical_columns: list[str] = field(default_factory=list)
    drop_columns: list[str] = field(default_factory=list)
    paths: list[Path] = field(default_factory=list)
    file_pattern: str = CSV_PATTERN
    reference: str = ""


NSL_KDD_CONFIG = DatasetConfig(
    name="NSL-KDD",
    class_names=["Normal", "DoS", "Probe", "R2L", "U2R"],
    label_column="class",
    feature_count=41,
    categorical_columns=["protocol_type", "service", "flag"],
    drop_columns=["difficulty"],
    paths=[
        DATA_DIR / "nsl_kdd" / "raw",
        DATA_DIR / "nsl_kdd_5class",
        DATA_DIR / "nsl_kdd",
    ],
    file_pattern=CSV_PATTERN,
    reference="Tavallaee et al. (2009)",
)

UNSW_NB15_CONFIG = DatasetConfig(
    name="UNSW-NB15",
    class_names=[
        "Normal",
        "Analysis",
        "Backdoor",
        "DoS",
        "Exploits",
        "Fuzzers",
        "Generic",
        "Reconnaissance",
        "Shellcode",
        "Worms",
    ],
    label_column="attack_cat",
    feature_count=47,
    categorical_columns=["proto", "service", "state"],
    drop_columns=["id", "label"],
    paths=[DATA_DIR / "unsw_nb15" / "raw", DATA_DIR / "unsw_nb15", ARCHIVE_DIR],
    file_pattern=CSV_PATTERN,
    reference="Moustafa & Slay (2015)",
)

CICIDS_2017_CONFIG = DatasetConfig(
    name="CICIDS-2018",
    class_names=[
        "BENIGN",
        "Bot",
        "DDoS",
        "DoS GoldenEye",
        "DoS Hulk",
        "DoS Slowhttptest",
        "DoS slowloris",
        "FTP-Patator",
        "Heartbleed",
        "Infiltration",
        "PortScan",
        "SSH-Patator",
        WEBATTACK_BRUTEFORCE,
        WEBATTACK_SQLINJECTION,
        WEBATTACK_XSS,
    ],
    label_column="Label",
    feature_count=78,
    paths=[DATA_DIR / "cicids2018" / "raw", ARCHIVE2_DIR],
    file_pattern=CSV_PATTERN,
    reference="Sharafaldin et al. (2018)",
)

TON_IOT_CONFIG = DatasetConfig(
    name="TON-IoT",
    class_names=[
        "Normal",
        "Backdoor",
        "DDoS",
        "DoS",
        "Injection",
        "Password",
        "Ransomware",
        "Scanning",
        "XSS",
        "MITM",
    ],
    label_column="type",
    feature_count=44,
    categorical_columns=["proto", "service", "conn_state"],
    drop_columns=["src_ip", "dst_ip", "src_port", "dst_port"],
    paths=[DATA_DIR / "ton_iot" / "raw"],
    file_pattern=CSV_PATTERN,
    reference="Moustafa (2021)",
)

CICIDS_2018_CONFIG = DatasetConfig(
    name="CICIDS-2018",
    class_names=[
        "BENIGN",
        "Bot",
        "DDoS",
        DOS_GOLDENEYE,
        DOS_HULK,
        DOS_SLOWHTTPTEST,
        DOS_SLOWLORIS,
        "FTP-Patator",
        "Infiltration",
        "SSH-Patator",
        WEBATTACK_BRUTEFORCE,
        WEBATTACK_SQLINJECTION,
        WEBATTACK_XSS,
    ],
    label_column="Label",
    feature_count=79,
    paths=[DATA_DIR / "cicids2018" / "raw", ARCHIVE2_DIR],
    file_pattern=CSV_PATTERN,
    reference="CSE-CIC-IDS2018",
)

DATASET_CONFIGS: dict[str, DatasetConfig] = {
    "nsl-kdd": NSL_KDD_CONFIG,
    "nsl_kdd": NSL_KDD_CONFIG,
    "unsw-nb15": UNSW_NB15_CONFIG,
    "unsw_nb15": UNSW_NB15_CONFIG,
    "cicids-2018": CICIDS_2018_CONFIG,
    "cicids_2017": CICIDS_2018_CONFIG,
    "cicids2017": CICIDS_2018_CONFIG,
    "cicids_2018": CICIDS_2018_CONFIG,
    "cicids2018": CICIDS_2018_CONFIG,
    "ton-iot": TON_IOT_CONFIG,
    "ton_iot": TON_IOT_CONFIG,
    "toniot": TON_IOT_CONFIG,
}
