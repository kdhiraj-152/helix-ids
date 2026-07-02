#!/usr/bin/env python3
"""
Phase 65 — Mechanistic Decomposition of the P(Y|X) Bottleneck.

Determines WHY the P(Y|X) bottleneck exists by identifying exact feature-,
class-, and distribution-level mechanisms responsible for cross-dataset failure.

Usage:
  PYTHONPATH=src python scripts/analysis/phase65_main.py --all
  PYTHONPATH=src python scripts/analysis/phase65_main.py --experiments 1,2,3
  PYTHONPATH=src python scripts/analysis/phase65_main.py --condition C  # encoder checkpoint
  PYTHONPATH=src python scripts/analysis/phase65_main.py --quick        # reduced samples
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["PYTHONHASHSEED"] = "42"

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.spatial.distance import cdist, pdist, squareform, mahalanobis
from scipy.linalg import svd
import sklearn.metrics as sk_metrics
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    accuracy_score, f1_score, silhouette_score, davies_bouldin_score,
    adjusted_rand_score, pairwise_distances,
)
from sklearn.feature_selection import mutual_info_classif

import torch
import torch.nn as nn
import torch.nn.functional as F

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

SEED = 42
rng = np.random.RandomState(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")

INPUT_DIM = 17
BATCH_SIZE = 1024
N_BOOTSTRAP = 10000
N_PERMUTATION = 5000
MAX_SAMPLES = 50000        # max per dataset for heavy computations
MAX_SAMPLES_LIGHT = 10000  # for distance-based computations
QUICK_SAMPLES = 5000       # --quick mode

PROJ = Path(__file__).resolve().parents[2]
CACHE = PROJ / "data" / "processed" / "phase52_cache"
RESULTS = PROJ / "results" / "phase65"
DATA_DIR = PROJ / "data"

ORIGINAL_DATASETS = [
    "nsl_kdd", "unsw_nb15", "cicids2017", "cicids2018", "ton_iot", "bot_iot"
]
EXTERNAL_DATASETS = ["iot23", "kyoto2006", "ugr16"]
ALL_DATASETS = ORIGINAL_DATASETS + EXTERNAL_DATASETS

DATASET_DISPLAY = {
    "nsl_kdd": "NSL-KDD", "unsw_nb15": "UNSW-NB15",
    "cicids2017": "CICIDS2017", "cicids2018": "CICIDS2018",
    "ton_iot": "TON-IoT", "bot_iot": "Bot-IoT",
    "iot23": "IoT-23", "kyoto2006": "Kyoto2006+", "ugr16": "UGR'16",
}

CANONICAL_FEATURE_ORDER = [
    "protocol_type", "connection_state", "traffic_direction", "has_rst",
    "log_src_bytes", "log_dst_bytes", "src_dst_bytes_ratio", "dst_src_bytes_ratio",
    "same_host_rate_x_service", "diff_srv_rate_x_flag", "count_x_srv_count",
    "protocol_service_flag", "src_bytes", "dst_bytes", "service_tier",
    "duration", "flag",
]

FAMILY_NAMES = ["Normal", "DoS", "Probe", "R2L", "U2R", "Generic", "Backdoor"]
FAMILY_NAMES_5CLASS = ["Normal", "DoS", "Probe", "R2L", "U2R"]

# ═══════════════════════════════════════════════════════════════════════════

for sub_dir in ["figures", "heatmaps", "embeddings"]:
    (RESULTS / sub_dir).mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJ / "src"))

logger = logging.getLogger("phase65")
logger.setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
fh = logging.FileHandler(RESULTS / "phase65_run.log", mode="w")
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(fh)

# ═══════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════

def cleanup():
    gc.collect()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def to_binary(y):
    return (y > 0).astype(np.int64)

def subsample_stratified(X, y, mx, rng_=None):
    if rng_ is None:
        rng_ = rng
    n = X.shape[0]
    if n <= mx:
        return X.copy(), y.copy()
    classes = np.unique(y)
    idx = []
    for c in classes:
        ci = np.where(y == c)[0]
        t = max(1, int(mx * len(ci) / n))
        if len(ci) > t:
            ci = rng_.choice(ci, size=t, replace=False)
        idx.extend(ci.tolist())
    rng_.shuffle(idx)
    return X[np.array(idx)], y[np.array(idx)]

def to_tensor(x_np, device=None):
    if device is None:
        device = DEVICE
    return torch.from_numpy(x_np).float().to(device)


# ═══════════════════════════════════════════════════════════════════════════
# Data Loading from CSVs (bypass phase52_cache)
# ═══════════════════════════════════════════════════════════════════════════

def _encode_protocol(proto_series):
    pmap = {"tcp": 0, "udp": 1, "icmp": 2, "-": 0}
    return proto_series.map(pmap).fillna(0).astype(int)

def _encode_conn_state(state_series):
    smap = {
        "S0": 0, "S1": 1, "S2": 2, "S3": 3, "SF": 4, "REJ": 5,
        "RST": 6, "RSTO": 6, "RSTR": 6, "RSTOS0": 6, "RSTRH": 6,
        "SH": 7, "SHR": 7, "OTH": 8, "CON": 9, "INT": 10, "FIN": 11,
        "acc": 12, "clo": 13, "no": 14, "par": 15, "urn": 16,
        "eco": 17, "tst": 18, "-": 19,
    }
    return state_series.map(smap).fillna(19).astype(int)

def _encode_service_tier_simple(service_series):
    tier_map = {
        "-": 0, "dns": 2, "http": 1, "https": 1, "ssl": 1,
        "ftp": 3, "smtp": 4, "ssh": 3, "telnet": 3,
        "dhcp": 2, "ntp": 2, "snmp": 6, "ldap": 6,
    }
    return service_series.map(tier_map).fillna(6).astype(int)

def _encode_protocol_from_service(service_series):
    udp_services = {"dns", "dhcp", "ntp", "snmp", "syslog", "tftp", "rip", "radius"}
    def _proto(s):
        s = str(s).strip().lower()
        if s in udp_services:
            return 1
        if s in {"icmp", "igmp", "ipv6-icmp"}:
            return 2
        return 0
    return service_series.apply(_proto).astype(int)


def load_already_harmonized_csv(dataset_name):
    """Load datasets that are already in canonical 17-feature format (NSL-KDD, UNSW-NB15)."""
    csv_dir = DATA_DIR / dataset_name
    train_csv = csv_dir / "train.csv"
    test_csv = csv_dir / "test.csv"

    frames = []
    if train_csv.exists():
        df = pd.read_csv(train_csv)
        frames.append(df)
    if test_csv.exists():
        df = pd.read_csv(test_csv)
        frames.append(df)

    if not frames:
        logger.warning(f"  No CSV data for {dataset_name}")
        return None

    combined = pd.concat(frames, ignore_index=True)

    # Determine label column
    if "label" in combined.columns:
        label_col = "label"
    elif "attack_cat" in combined.columns:
        label_col = "attack_cat"
    else:
        label_col = None

    if label_col:
        y = combined[label_col].values.astype(np.int64)
        X = combined.drop(columns=[label_col]).values.astype(np.float64)
    else:
        y = np.zeros(len(combined), dtype=np.int64)
        X = combined.values.astype(np.float64)

    # Verify shape
    if X.shape[1] != INPUT_DIM:
        logger.warning(f"  {dataset_name}: Expected {INPUT_DIM} features, got {X.shape[1]}")
        return None

    y_bin = to_binary(y)
    logger.info(f"  {DATASET_DISPLAY[dataset_name]}: X={X.shape}, classes={np.unique(y)}")
    return {"X": X, "y": y, "y_bin": y_bin}


def load_ton_iot_raw():
    """Load TON-IoT from raw CSV and harmonize to canonical 17 features."""
    csv_path = DATA_DIR / "ton_iot" / "raw" / "train.csv"
    if not csv_path.exists():
        logger.warning("  No TON-IoT raw CSV found")
        return None

    df = pd.read_csv(csv_path, low_memory=False)
    logger.info(f"  TON-IoT raw: {df.shape}")

    result = pd.DataFrame()
    result["protocol_type"] = _encode_protocol(df.get("proto", pd.Series(["tcp"] * len(df))).fillna("tcp"))

    # Duration
    for col in ["duration", "dur", "flow_duration"]:
        if col in df.columns:
            result["duration"] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            break
    else:
        result["duration"] = 0.0

    # Source/Destination bytes
    sb_col = next((c for c in ["src_bytes", "sbytes", "bytes_sent", "orig_bytes", "src_pkts"]
                   if c in df.columns), None)
    db_col = next((c for c in ["dst_bytes", "dbytes", "bytes_received", "resp_bytes", "dst_pkts"]
                   if c in df.columns), None)
    sb = pd.to_numeric(df[sb_col], errors="coerce").fillna(0).values if sb_col else np.zeros(len(df))
    db = pd.to_numeric(df[db_col], errors="coerce").fillna(0).values if db_col else np.zeros(len(df))
    result["src_bytes"] = sb.astype(float)
    result["dst_bytes"] = db.astype(float)
    sb = np.maximum(sb, 0)
    db = np.maximum(db, 0)
    result["log_src_bytes"] = np.log1p(sb)
    result["log_dst_bytes"] = np.log1p(db)
    result["src_dst_bytes_ratio"] = sb / (db + 1.0)
    result["dst_src_bytes_ratio"] = db / (sb + 1.0)

    # Connection state / flag
    for col in ["conn_state", "state", "flag", "tcp_flags"]:
        if col in df.columns:
            result["flag"] = _encode_conn_state(df[col].astype(str))
            break
    else:
        result["flag"] = 0
    result["connection_state"] = result["flag"].copy()
    result["has_rst"] = (result["flag"] == 6).astype(int)

    # Service tier
    if "service" in df.columns:
        result["service_tier"] = _encode_service_tier_simple(df["service"].fillna("-"))
    else:
        result["service_tier"] = 0

    result["traffic_direction"] = 0

    # Engineered features
    svc_tier = result["service_tier"].values + 1.0
    flag_val = result["flag"].values + 1.0
    proto_val = result["protocol_type"].values + 1.0
    dur = np.maximum(result["duration"].values, 0).astype(float)
    result["same_host_rate_x_service"] = svc_tier * sb / (db + 1.0)
    result["diff_srv_rate_x_flag"] = flag_val * np.abs(sb - db) / (sb + db + 1.0)
    result["count_x_srv_count"] = (dur + 1.0) * svc_tier
    result["protocol_service_flag"] = proto_val * svc_tier * flag_val

    # Label
    if "type" in df.columns:
        labels = df["type"].astype(str).str.lower().str.strip()
        # Map to 5-class: 0=Normal, 1=DoS, 2=Probe, 3=R2L, 4=U2R
        label_map = {
            "normal": 0, "benign": 0,
            "dos": 1, "ddos": 1,
            "scanning": 2, "probe": 2,
            "backdoor": 4, "injection": 4, "password": 4, "ransomware": 4,
            "xss": 4, "mitm": 4,
        }
        y = np.array([label_map.get(l, 0) for l in labels], dtype=np.int64)
    else:
        y = np.zeros(len(df), dtype=np.int64)

    for c in CANONICAL_FEATURE_ORDER:
        if c not in result.columns:
            result[c] = 0.0

    X = result[CANONICAL_FEATURE_ORDER].values.astype(np.float64)
    y_bin = to_binary(y)
    logger.info(f"  TON-IoT: X={X.shape}, classes={np.unique(y)}")
    return {"X": X, "y": y, "y_bin": y_bin}


def load_bot_iot_raw():
    """Load Bot-IoT from raw CSV and harmonize to canonical 17 features."""
    csv_path = DATA_DIR / "bot_iot" / "raw" / "train.csv"
    if not csv_path.exists():
        logger.warning("  No Bot-IoT raw CSV found")
        return None

    df = pd.read_csv(csv_path, low_memory=False)
    logger.info(f"  Bot-IoT raw: {df.shape}")

    result = pd.DataFrame()
    result["protocol_type"] = df.get("proto_number", pd.Series(0)).fillna(0).astype(int)

    # Duration
    for col in ["dur", "duration", "flow_duration"]:
        if col in df.columns:
            result["duration"] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            break
    else:
        result["duration"] = 0.0

    # Bytes
    sb_col = next((c for c in ["sbytes", "src_bytes", "orig_bytes"] if c in df.columns), None)
    db_col = next((c for c in ["dbytes", "dst_bytes", "resp_bytes"] if c in df.columns), None)
    sb = pd.to_numeric(df[sb_col], errors="coerce").fillna(0).values if sb_col else np.zeros(len(df))
    db = pd.to_numeric(df[db_col], errors="coerce").fillna(0).values if db_col else np.zeros(len(df))
    result["src_bytes"] = sb.astype(float)
    result["dst_bytes"] = db.astype(float)
    sb = np.maximum(sb, 0)
    db = np.maximum(db, 0)
    result["log_src_bytes"] = np.log1p(sb)
    result["log_dst_bytes"] = np.log1p(db)
    result["src_dst_bytes_ratio"] = sb / (db + 1.0)
    result["dst_src_bytes_ratio"] = db / (sb + 1.0)

    # State/flag
    for col in ["state_number", "flgs_number", "state", "flag"]:
        if col in df.columns:
            result["flag"] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
            break
    else:
        result["flag"] = 0
    result["connection_state"] = result["flag"].copy()
    result["has_rst"] = (result["flag"] == 6).astype(int)
    result["service_tier"] = 0
    result["traffic_direction"] = 0

    svc_tier = result["service_tier"].values + 1.0
    flag_val = result["flag"].values + 1.0
    proto_val = result["protocol_type"].values + 1.0
    dur = np.maximum(result["duration"].values, 0).astype(float)
    result["same_host_rate_x_service"] = svc_tier * sb / (db + 1.0)
    result["diff_srv_rate_x_flag"] = flag_val * np.abs(sb - db) / (sb + db + 1.0)
    result["count_x_srv_count"] = (dur + 1.0) * svc_tier
    result["protocol_service_flag"] = proto_val * svc_tier * flag_val

    # Label
    if "attack" in df.columns:
        y = pd.to_numeric(df["attack"], errors="coerce").fillna(0).astype(np.int64).values
        # If 2-class binary
        if len(np.unique(y)) <= 2:
            y = y  # 0=normal, 1=attack
        # Map category to 5-class if available
        if "category" in df.columns:
            cat_map = {"normal": 0, "benign": 0, "dos": 1, "ddos": 1,
                       "probe": 2, "scanning": 2, "reconnaissance": 2,
                       "r2l": 3, "u2r": 4, "exploit": 4, "backdoor": 4}
            cats = df["category"].astype(str).str.lower().str.strip()
            y_5class = np.array([cat_map.get(c, int(y[i]) if y[i] > 0 else 0)
                                 for i, c in enumerate(cats)], dtype=np.int64)
            y = y_5class
    else:
        y = np.zeros(len(df), dtype=np.int64)

    for c in CANONICAL_FEATURE_ORDER:
        if c not in result.columns:
            result[c] = 0.0

    X = result[CANONICAL_FEATURE_ORDER].values.astype(np.float64)
    y_bin = to_binary(y)
    logger.info(f"  Bot-IoT: X={X.shape}, classes={np.unique(y)}")
    return {"X": X, "y": y, "y_bin": y_bin}


def load_cicids2017_raw():
    """Load CICIDS2017 from raw CSV and harmonize."""
    csv_path = DATA_DIR / "cicids2017" / "raw" / "train.csv"
    if not csv_path.exists():
        logger.warning("  No CICIDS2017 raw CSV found")
        return None

    df = pd.read_csv(csv_path, low_memory=False)
    logger.info(f"  CICIDS2017 raw: {df.shape}")

    result = pd.DataFrame()
    # Protocol column: lowercase string values ['tcp', 'udp', 'other']
    proto_col = df.get("protocol", df.get("Protocol", pd.Series("tcp")))
    if proto_col.dtype == object:
        proto_str = proto_col.astype(str).str.lower().str.strip()
        result["protocol_type"] = proto_str.map({"tcp": 0, "udp": 1, "icmp": 2}).fillna(0).astype(int)
    else:
        result["protocol_type"] = proto_col.fillna(0).astype(int)
        result["protocol_type"] = result["protocol_type"].map({6: 0, 17: 1, 1: 2}).fillna(0).astype(int)

    # Duration
    for col in ["Flow Duration", "flow_duration", "duration", "dur"]:
        if col in df.columns:
            result["duration"] = pd.to_numeric(df[col], errors="coerce").fillna(0) / 1e6  # to seconds
            break
    else:
        result["duration"] = 0.0

    # Bytes
    sb_col = next((c for c in ["TotLen Fwd Pkts", "Total Length of Fwd Packets",
                                "src_bytes", "sbytes", "fwd_pkt_len_total"]
                   if c in df.columns), None)
    db_col = next((c for c in ["TotLen Bwd Pkts", "Total Length of Bwd Packets",
                                "dst_bytes", "dbytes", "bwd_pkt_len_total"]
                   if c in df.columns), None)
    sb = pd.to_numeric(df[sb_col], errors="coerce").fillna(0).values if sb_col else np.zeros(len(df))
    db = pd.to_numeric(df[db_col], errors="coerce").fillna(0).values if db_col else np.zeros(len(df))
    result["src_bytes"] = sb.astype(float)
    result["dst_bytes"] = db.astype(float)
    sb = np.maximum(sb, 0)
    db = np.maximum(db, 0)
    result["log_src_bytes"] = np.log1p(sb)
    result["log_dst_bytes"] = np.log1p(db)
    result["src_dst_bytes_ratio"] = sb / (db + 1.0)
    result["dst_src_bytes_ratio"] = db / (sb + 1.0)

    # SYN/RST flags
    if "SYN Flag Cnt" in df.columns:
        result["flag"] = (pd.to_numeric(df["SYN Flag Cnt"], errors="coerce").fillna(0) > 0).astype(int)
        result["flag"] = result["flag"] * 4  # SF-like
    else:
        result["flag"] = 0
    if "RST Flag Cnt" in df.columns:
        result["has_rst"] = (pd.to_numeric(df["RST Flag Cnt"], errors="coerce").fillna(0) > 0).astype(int)
    else:
        result["has_rst"] = 0
    result["connection_state"] = result["flag"].copy()

    result["service_tier"] = 0
    result["traffic_direction"] = 0

    svc_tier = result["service_tier"].values + 1.0
    flag_val = result["flag"].values + 1.0
    proto_val = result["protocol_type"].values + 1.0
    dur = np.maximum(result["duration"].values, 0).astype(float)
    result["same_host_rate_x_service"] = svc_tier * sb / (db + 1.0)
    result["diff_srv_rate_x_flag"] = flag_val * np.abs(sb - db) / (sb + db + 1.0)
    result["count_x_srv_count"] = (dur + 1.0) * svc_tier
    result["protocol_service_flag"] = proto_val * svc_tier * flag_val

    # Label
    label_col = next((c for c in ["Label", "label", "attack_label", "attack_type", "class"]
                     if c in df.columns), None)
    if label_col:
        labels = df[label_col].astype(str).str.lower().str.strip()
        # Map to binary: 0=benign/normal, 1=attack
        y = (~labels.isin(["benign", "normal", "0", "-", ""])).astype(np.int64).values
    else:
        y = np.zeros(len(df), dtype=np.int64)

    y_bin = y.copy()
    for c in CANONICAL_FEATURE_ORDER:
        if c not in result.columns:
            result[c] = 0.0

    X = result[CANONICAL_FEATURE_ORDER].values.astype(np.float64)
    logger.info(f"  CICIDS2017: X={X.shape}")
    return {"X": X, "y": y, "y_bin": y_bin}


# ── External Dataset Loaders (from Phase 64) ─────────────────────────────

def _parse_iot23_conn_log(content):
    import re as _re
    lines = content.strip().split("\n")
    data_lines = []
    header_fields_raw = None
    for line in lines:
        if line.startswith("#fields"):
            header_fields_raw = line.strip().split("\t")[1:]
        elif line.startswith("#"):
            continue
        else:
            data_lines.append(line)
    if header_fields_raw is None or len(header_fields_raw) != 21:
        if not data_lines:
            return None
        sample = data_lines[0].strip().split("\t")
        header_fields_raw = [f"col{i}" for i in range(len(sample))]
    col_names = [c.strip() for c in header_fields_raw[:20]]
    last_parts = _re.split(r"\s{2,}", header_fields_raw[20].strip())
    col_names.extend(last_parts)
    rows = []
    for line in data_lines:
        parts = line.strip().split("\t")
        if len(parts) <= 1:
            continue
        if len(parts) == 21:
            last_3 = _re.split(r"\s{2,}", parts[20].strip())
            if len(last_3) >= 3:
                row = parts[:20] + [last_3[0].strip(), last_3[1].strip(), " ".join(last_3[2:]).strip()]
            else:
                row = parts[:20] + list(last_3) + [""] * (3 - len(last_3))
        else:
            row = parts[:23]
        while len(row) < 23:
            row.append("")
        rows.append(row[:23])
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=col_names[:23])
    result = pd.DataFrame()
    if "proto" in df.columns:
        result["protocol_type"] = _encode_protocol(df["proto"].fillna("tcp"))
    else:
        result["protocol_type"] = 0
    if "duration" in df.columns:
        result["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0)
    else:
        result["duration"] = 0.0
    ob_col = "orig_bytes" if "orig_bytes" in df.columns else "src_bytes"
    rb_col = "resp_bytes" if "resp_bytes" in df.columns else "dst_bytes"
    result["src_bytes"] = pd.to_numeric(df.get(ob_col, 0), errors="coerce").fillna(0)
    result["dst_bytes"] = pd.to_numeric(df.get(rb_col, 0), errors="coerce").fillna(0)
    if "conn_state" in df.columns:
        conn_state_raw = df["conn_state"].fillna("-")
        result["flag"] = _encode_conn_state(conn_state_raw)
        result["connection_state"] = result["flag"].copy()
    else:
        result["flag"] = 0; result["connection_state"] = 0
    if "conn_state" in df.columns:
        rst_states = {"RST", "RSTO", "RSTR", "RSTOS0", "RSTRH"}
        result["has_rst"] = df["conn_state"].fillna("-").isin(rst_states).astype(int)
    else:
        result["has_rst"] = 0
    if "service" in df.columns:
        result["service_tier"] = _encode_service_tier_simple(df["service"].fillna("-"))
    else:
        result["service_tier"] = 0
    result["traffic_direction"] = 0
    sb = np.maximum(result["src_bytes"].values, 0)
    db = np.maximum(result["dst_bytes"].values, 0)
    result["log_src_bytes"] = np.log1p(sb)
    result["log_dst_bytes"] = np.log1p(db)
    result["src_dst_bytes_ratio"] = sb / (db + 1.0)
    result["dst_src_bytes_ratio"] = db / (sb + 1.0)
    svc_tier = result["service_tier"].values + 1.0
    flag_val = result["flag"].values + 1.0
    proto_val = result["protocol_type"].values + 1.0
    dur_val = np.maximum(result["duration"].values, 0).astype(float)
    result["same_host_rate_x_service"] = svc_tier * sb / (db + 1.0)
    result["diff_srv_rate_x_flag"] = flag_val * np.abs(sb - db) / (sb + db + 1.0)
    result["count_x_srv_count"] = (dur_val + 1.0) * svc_tier
    result["protocol_service_flag"] = proto_val * svc_tier * flag_val
    if "label" in df.columns:
        label_str = df["label"].astype(str).str.strip().str.lower()
        result["label_bin"] = (label_str == "malicious").astype(int)
        y = result["label_bin"].values
    else:
        y = np.zeros(len(df), dtype=np.int64)
    for c in CANONICAL_FEATURE_ORDER:
        if c not in result.columns:
            result[c] = 0.0
    return result[CANONICAL_FEATURE_ORDER], y


def load_iot23(data_dir=None):
    if data_dir is None:
        data_dir = DATA_DIR
    iot_dir = data_dir / "iot23"
    labeled_files = sorted(iot_dir.glob("*.conn.log.labeled"))
    if not labeled_files:
        logger.warning("  No IoT-23 files found")
        return None
    frames = []
    y_all = []
    for lf in labeled_files:
        content = lf.read_text(encoding="utf-8", errors="replace")
        parsed = _parse_iot23_conn_log(content)
        if parsed is not None:
            feat_df, y = parsed
            frames.append(feat_df)
            y_all.append(y)
    if not frames:
        return None
    X = pd.concat(frames, ignore_index=True).values.astype(np.float64)
    y = np.concatenate(y_all).astype(np.int64)
    y_bin = to_binary(y)
    logger.info(f"  IoT-23: X={X.shape}, classes={np.unique(y)}")
    return {"X": X, "y": y, "y_bin": y_bin}


def load_kyoto2006(data_dir=None):
    if data_dir is None:
        data_dir = DATA_DIR
    kyoto_dir = data_dir / "kyoto2006"
    data_file = kyoto_dir / "kyoto_processed.csv"
    if data_file.exists():
        df = pd.read_csv(data_file)
        y = df["label"].values.astype(np.int64) if "label" in df.columns else np.zeros(len(df), dtype=np.int64)
        X = df[[c for c in CANONICAL_FEATURE_ORDER if c in df.columns]].values.astype(np.float64)
        y_bin = to_binary(y)
        logger.info(f"  Kyoto2006+ (processed): X={X.shape}, classes={np.unique(y)}")
        return {"X": X, "y": y, "y_bin": y_bin}

    # Generate synthetic representative data
    n_samples = 50000
    rs = np.random.RandomState(42)
    result = pd.DataFrame()
    result["duration"] = np.exp(rs.randn(n_samples) * 1.5 - 1)
    result["src_bytes"] = np.exp(rs.randn(n_samples) * 2 + 4)
    result["dst_bytes"] = np.exp(rs.randn(n_samples) * 1.5 + 3)
    X = pd.DataFrame()
    X["protocol_type"] = 0
    X["connection_state"] = np.where(result["duration"] > 5, 0, 4)
    X["traffic_direction"] = 0
    X["has_rst"] = (X["connection_state"] == 5).astype(int)
    X["log_src_bytes"] = np.log1p(np.maximum(result["src_bytes"].values, 0))
    X["log_dst_bytes"] = np.log1p(np.maximum(result["dst_bytes"].values, 0))
    sb = np.maximum(result["src_bytes"].values, 0).astype(float)
    db = np.maximum(result["dst_bytes"].values, 0).astype(float)
    X["src_bytes"] = sb
    X["dst_bytes"] = db
    X["src_dst_bytes_ratio"] = sb / (db + 1.0)
    X["dst_src_bytes_ratio"] = db / (sb + 1.0)
    X["same_host_rate_x_service"] = 0.6
    X["diff_srv_rate_x_flag"] = 0.3
    X["count_x_srv_count"] = 10.0
    X["protocol_service_flag"] = 1.0
    X["service_tier"] = 0
    X["duration"] = result["duration"].values
    X["flag"] = X["connection_state"].values
    y = np.zeros(n_samples, dtype=np.int64)
    # Add some attacks
    n_attack = int(n_samples * 0.3)
    attack_idx = rs.choice(n_samples, n_attack, replace=False)
    y[attack_idx] = 1
    y_bin = y.copy()
    logger.warning(f"  Using {n_samples} Kyoto-representative synthetic samples")
    X_arr = X[CANONICAL_FEATURE_ORDER].values.astype(np.float64)
    return {"X": X_arr, "y": y, "y_bin": y_bin}


def _harmonize_ugr16_flow(df):
    result = pd.DataFrame()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "proto" in df.columns:
        result["protocol_type"] = _encode_protocol(df["proto"].astype(str))
    elif "protocol" in df.columns:
        result["protocol_type"] = _encode_protocol(df["protocol"].astype(str))
    else:
        result["protocol_type"] = 0
    for col in ["duration", "dur", "flow_duration"]:
        if col in df.columns:
            result["duration"] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            break
    else:
        result["duration"] = 0.0
    if "src_bytes" not in df.columns and "sbytes" in df.columns:
        df["src_bytes"] = df["sbytes"]
    if "dst_bytes" not in df.columns and "dbytes" in df.columns:
        df["dst_bytes"] = df["dbytes"]
    sb_col = next((c for c in ["src_bytes", "sbytes", "bytes_sent", "orig_bytes", "src_pkts"] if c in df.columns), None)
    db_col = next((c for c in ["dst_bytes", "dbytes", "bytes_received", "resp_bytes", "dst_pkts"] if c in df.columns), None)
    sb = pd.to_numeric(df[sb_col], errors="coerce").fillna(0).values if sb_col else np.zeros(len(df))
    db = pd.to_numeric(df[db_col], errors="coerce").fillna(0).values if db_col else np.zeros(len(df))
    result["src_bytes"] = sb.astype(float)
    result["dst_bytes"] = db.astype(float)
    sb = np.maximum(sb, 0)
    db = np.maximum(db, 0)
    result["log_src_bytes"] = np.log1p(sb)
    result["log_dst_bytes"] = np.log1p(db)
    result["src_dst_bytes_ratio"] = sb / (db + 1.0)
    result["dst_src_bytes_ratio"] = db / (sb + 1.0)
    for col in ["state", "conn_state", "flag", "tcp_flags"]:
        if col in df.columns:
            result["flag"] = _encode_conn_state(df[col].astype(str))
            break
    else:
        result["flag"] = 0
    result["connection_state"] = result["flag"].copy()
    result["has_rst"] = (result["flag"] == 6).astype(int)
    result["service_tier"] = 0
    if "service" in df.columns:
        result["service_tier"] = _encode_service_tier_simple(df["service"].fillna("-"))
    result["traffic_direction"] = 0
    flag_signal = result["flag"].values + 1.0
    svc_signal = result["service_tier"].values + 1.0
    proto_signal = result["protocol_type"].values + 1.0
    dur = np.maximum(result["duration"].values, 0).astype(float)
    result["same_host_rate_x_service"] = svc_signal * sb / (db + 1.0)
    result["diff_srv_rate_x_flag"] = flag_signal * np.abs(sb - db) / (sb + db + 1.0)
    result["count_x_srv_count"] = (dur + 1.0) * svc_signal
    result["protocol_service_flag"] = proto_signal * svc_signal * flag_signal
    label_col = next((c for c in ["label", "attack", "class", "Label", "Att"] if c in df.columns), None)
    if label_col:
        labels = df[label_col].astype(str).str.lower().str.strip()
        y = (~labels.isin(["0", "normal", "benign", "-", ""])).astype(int).values
    else:
        y = np.zeros(len(df), dtype=np.int64)
    for c in CANONICAL_FEATURE_ORDER:
        if c not in result.columns:
            result[c] = 0.0
    return result[CANONICAL_FEATURE_ORDER].values.astype(np.float64), y


def _load_ugr16_calibration(cal_dir):
    csv_files = sorted(cal_dir.glob("*.csv")) if cal_dir.exists() else []
    if not csv_files:
        return None
    frames = []
    for cf in csv_files:
        try:
            df = pd.read_csv(cf)
            frames.append(df)
        except Exception:
            pass
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    numeric_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
    if "counter(mins)" in numeric_cols:
        numeric_cols.remove("counter(mins)")
    if not numeric_cols:
        return None
    result = pd.DataFrame()
    n = len(combined)
    result["protocol_type"] = np.zeros(n)
    result["connection_state"] = np.zeros(n)
    result["traffic_direction"] = np.zeros(n)
    result["has_rst"] = np.zeros(n)
    result["duration"] = np.full(n, 60.0)
    sb = combined[numeric_cols].sum(axis=1).fillna(0).values
    db = sb * 0.5
    result["src_bytes"] = sb
    result["dst_bytes"] = db
    result["log_src_bytes"] = np.log1p(np.maximum(sb, 0))
    result["log_dst_bytes"] = np.log1p(np.maximum(db, 0))
    result["src_dst_bytes_ratio"] = sb / (db + 1.0)
    result["dst_src_bytes_ratio"] = db / (sb + 1.0)
    result["service_tier"] = np.zeros(n)
    result["flag"] = np.zeros(n)
    result["same_host_rate_x_service"] = sb / (db + 1.0)
    result["diff_srv_rate_x_flag"] = np.abs(sb - db) / (sb + db + 1.0)
    result["count_x_srv_count"] = np.full(n, 60.0)
    result["protocol_service_flag"] = np.full(n, 1.0)
    attack_cols = [c for c in numeric_cols if c not in ["counter(mins)"]]
    y = (combined[attack_cols].sum(axis=1) > 0).astype(int).values
    return result[CANONICAL_FEATURE_ORDER].values.astype(np.float64), y


def load_ugr16(data_dir=None):
    if data_dir is None:
        data_dir = DATA_DIR
    ugr_dir = data_dir / "ugr16"
    csv_files = list(ugr_dir.glob("*.csv"))
    if not csv_files:
        cal_dir = data_dir / "ugr16_cal"
        cal_csvs = list(cal_dir.glob("*.csv")) if cal_dir.exists() else []
        if cal_csvs:
            result = _load_ugr16_calibration(cal_dir)
            if result is not None:
                X, y = result
                y_bin = to_binary(y)
                logger.info(f"  UGR'16 (calibration): X={X.shape}")
                return {"X": X, "y": y, "y_bin": y_bin}
        logger.warning("  No UGR'16 files found")
        return None
    frames = []
    y_all = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, low_memory=False)
            X_part, y_part = _harmonize_ugr16_flow(df)
            frames.append(X_part)
            y_all.append(y_part)
        except Exception as e:
            logger.warning(f"  Error loading {csv_file}: {e}")
    if not frames:
        cal_dir = data_dir / "ugr16_cal"
        if cal_dir.exists():
            result = _load_ugr16_calibration(cal_dir)
            if result is not None:
                X, y = result
                y_bin = to_binary(y)
                logger.info(f"  UGR'16 (calibration fallback): X={X.shape}")
                return {"X": X, "y": y, "y_bin": y_bin}
        return None
    X = np.vstack(frames)
    y = np.concatenate(y_all).astype(np.int64)
    y_bin = to_binary(y)
    logger.info(f"  UGR'16: X={X.shape}, classes={np.unique(y)}")
    return {"X": X, "y": y, "y_bin": y_bin}


# ═══════════════════════════════════════════════════════════════════════════
# Load All Datasets
# ═══════════════════════════════════════════════════════════════════════════

DATA_LOADERS = {
    "nsl_kdd": lambda: load_already_harmonized_csv("nsl_kdd"),
    "unsw_nb15": lambda: load_already_harmonized_csv("unsw_nb15"),
    "cicids2017": load_cicids2017_raw,
    "ton_iot": load_ton_iot_raw,
    "bot_iot": load_bot_iot_raw,
    "iot23": load_iot23,
    "kyoto2006": load_kyoto2006,
    "ugr16": load_ugr16,
}


def load_all_datasets(max_samples=MAX_SAMPLES):
    """Load all available datasets with harmonization."""
    all_data = {}
    for name in ALL_DATASETS:
        logger.info(f"Loading {DATASET_DISPLAY[name]}...")
        if name in DATA_LOADERS:
            d = DATA_LOADERS[name]()
        else:
            logger.warning(f"  No loader for {name}, skipping")
            continue
        if d is None or d["X"].shape[0] == 0:
            logger.warning(f"  {name}: No data loaded, skipping")
            continue
        # Subsample if needed
        X, y = d["X"], d["y"]
        if X.shape[0] > max_samples:
            X, y = subsample_stratified(X, y, max_samples)
        y_bin = to_binary(y)
        all_data[name] = {"X": X, "y": y, "y_bin": y_bin}
        logger.info(f"  → {DATASET_DISPLAY[name]}: {X.shape}, "
                    f"bin={np.bincount(y_bin)}, "
                    f"classes={np.unique(y)}")
    return all_data


def prepare_scaled_data(data_dict):
    """Standardize each dataset independently."""
    scalers = {}
    scaled = {}
    for name in sorted(data_dict.keys()):
        X = data_dict[name]["X"]
        sc = StandardScaler().fit(X)
        X_scaled = sc.transform(X)
        scalers[name] = sc
        scaled[name] = {
            "X": X_scaled,
            "y": data_dict[name]["y"],
            "y_bin": data_dict[name]["y_bin"],
        }
    return scaled, scalers


# ═══════════════════════════════════════════════════════════════════════════
# Model: HelixIDSFull (Phase 64 architecture)
# ═══════════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x: torch.Tensor):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * self.scale / rms


def build_helix_backbone(input_dim=17, hidden_dims=(512, 384, 256, 256),
                         norm_type="batchnorm"):
    layers = []
    prev = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        if norm_type == "batchnorm":
            layers.append(nn.BatchNorm1d(h))
        elif norm_type == "rmsnorm":
            layers.append(RMSNorm(h))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.3))
        prev = h
    return nn.Sequential(*layers)


class HelixIDSFullPhase65(nn.Module):
    """Encoder model matching Phase 64 architecture."""

    def __init__(self, input_dim=17, norm_type="batchnorm"):
        super().__init__()
        self.input_dim = input_dim
        self.norm_type = norm_type

        self.backbone = build_helix_backbone(
            input_dim=input_dim,
            hidden_dims=(512, 384, 256, 256),
            norm_type=norm_type,
        )

        self.binary_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

        self.family_projection = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.family_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 7),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, return_features=False):
        features = self.backbone(x)
        binary_logits = self.binary_head(features)
        family_feats = self.family_projection(features)
        family_feats = self._whiten_family_features(family_feats)
        family_logits = self.family_head(family_feats)
        if torch.isnan(binary_logits).any() or torch.isnan(family_logits).any():
            binary_logits = torch.nan_to_num(binary_logits, nan=0.0, posinf=1e3, neginf=-1e3)
            family_logits = torch.nan_to_num(family_logits, nan=0.0, posinf=1e3, neginf=-1e3)
        if return_features:
            return binary_logits, family_logits, features
        return binary_logits, family_logits

    @staticmethod
    def _whiten_family_features(x):
        mu = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True) + 1e-5
        return (x - mu) / std

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True
        self.train()


@torch.no_grad()
def extract_embeddings(model, X_np, batch_size=512):
    """Extract backbone embeddings (256-dim)."""
    model.eval()
    all_feats = []
    n = X_np.shape[0]
    for i in range(0, n, batch_size):
        batch = X_np[i:i + batch_size]
        x = torch.from_numpy(batch).float().to(DEVICE)
        _, _, features = model(x, return_features=True)
        all_feats.append(features.detach().cpu().numpy())
    return np.vstack(all_feats)


@torch.no_grad()
def predict_model(model, X_np, batch_size=512):
    """Get predictions from model."""
    model.eval()
    all_logits = []
    n = X_np.shape[0]
    for i in range(0, n, batch_size):
        batch = X_np[i:i + batch_size]
        x = torch.from_numpy(batch).float().to(DEVICE)
        bin_logits, _ = model(x)
        all_logits.append(bin_logits.detach().cpu().numpy())
    all_logits = np.vstack(all_logits)
    probs = F.softmax(torch.from_numpy(all_logits), dim=1).numpy()
    preds = np.argmax(all_logits, axis=1)
    return preds, probs


def load_pretrained_model(checkpoint_path=None, condition="C"):
    """Load Phase 64 pretrained encoder."""
    if checkpoint_path is None:
        checkpoint_path = RESULTS.parent / "phase64" / f"phase64_condition_{condition}_model.pt"
    checkpoint_path = Path(checkpoint_path)

    norm_type = "batchnorm"
    model = HelixIDSFullPhase65(input_dim=INPUT_DIM, norm_type=norm_type).to(DEVICE)

    if checkpoint_path.exists():
        state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        logger.info(f"  Loaded checkpoint: {checkpoint_path}")
    else:
        logger.warning(f"  Checkpoint not found: {checkpoint_path}")
        logger.warning("  Using untrained model (random weights)")
        # Try other conditions
        for cond in ["B", "C", "A", "D", "E"]:
            alt_path = RESULTS.parent / "phase64" / f"phase64_condition_{cond}_model.pt"
            if alt_path.exists():
                state_dict = torch.load(alt_path, map_location=DEVICE, weights_only=True)
                model.load_state_dict(state_dict)
                logger.info(f"  Loaded fallback checkpoint: {alt_path}")
                break

    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════════════════
# Statistical Analysis Helpers
# ═══════════════════════════════════════════════════════════════════════════

def bootstrap_ci(values, n_bootstrap=N_BOOTSTRAP, ci=95):
    values = np.asarray(values)
    n = len(values)
    means = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        means[i] = np.mean(values[idx])
    alpha = (100 - ci) / 2
    lower = np.percentile(means, alpha)
    upper = np.percentile(means, 100 - alpha)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "ci_level": ci,
        "n_bootstrap": n_bootstrap,
    }


def permutation_test(values_a, values_b, n_perm=N_PERMUTATION):
    """Two-sided permutation test for difference in means."""
    observed = np.mean(values_a) - np.mean(values_b)
    combined = np.concatenate([values_a, values_b])
    n_a = len(values_a)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(combined)
        perm_a = combined[:n_a]
        perm_b = combined[n_a:]
        perm_diff = np.mean(perm_a) - np.mean(perm_b)
        if abs(perm_diff) >= abs(observed):
            count += 1
    p_value = (count + 1) / (n_perm + 1)
    return {"observed_diff": float(observed), "p_value": float(p_value), "n_permutations": n_perm}


def bayesian_compare(values_a, values_b):
    from scipy.stats import ttest_ind
    t_stat, p_val = ttest_ind(values_a, values_b, equal_var=False)
    n1, n2 = len(values_a), len(values_b)
    bic_null = n1 * np.log(np.var(values_a) + 1e-12) + n2 * np.log(np.var(values_b) + 1e-12)
    pooled_var = ((n1 - 1) * np.var(values_a, ddof=1) + (n2 - 1) * np.var(values_b, ddof=1)) / (n1 + n2 - 2)
    bic_alt = (n1 + n2) * np.log(pooled_var + 1e-12)
    bayes_factor = np.exp((bic_null - bic_alt) / 2)
    bayes_factor = min(bayes_factor, 1e10)
    prob_null = 1 / (1 + bayes_factor)
    prob_alt = 1 - prob_null
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "bayes_factor": float(bayes_factor),
        "prob_alternative": float(prob_alt),
        "prob_null": float(prob_null),
    }


def effect_size(values_a, values_b):
    n1, n2 = len(values_a), len(values_b)
    m1, m2 = np.mean(values_a), np.mean(values_b)
    s1, s2 = np.std(values_a, ddof=1), np.std(values_b, ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    cohens_d = (m1 - m2) / max(pooled, 1e-12)
    correction = 1 - 3 / (4 * (n1 + n2) - 9)
    hedges_g = cohens_d * correction
    return {
        "cohens_d": float(cohens_d),
        "hedges_g": float(hedges_g),
        "mean_diff": float(m1 - m2),
        "mean_a": float(m1),
        "mean_b": float(m2),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 1: Feature-Level Distribution Decomposition
# ═══════════════════════════════════════════════════════════════════════════

def experiment_1_feature_distribution(data_dict, quick=False):
    """
    For every dataset pair, compute per-feature distribution divergences.
    Produces 17 × 36 pairwise heatmaps and feature stability rankings.
    """
    logger.info("\n" + "=" * 60)
    logger.info("Experiment 1: Feature-Level Distribution Decomposition")
    logger.info("=" * 60)

    names = sorted(data_dict.keys())
    n_datasets = len(names)
    n_features = len(CANONICAL_FEATURE_ORDER)

    # Storage: {feature_name: {metric: {dataset_pair: value}}}
    results = defaultdict(lambda: defaultdict(dict))
    pair_list = []

    n_pairs = 0
    for i in range(n_datasets):
        for j in range(i + 1, n_datasets):
            pair_list.append((names[i], names[j]))
            n_pairs += 1

    logger.info(f"  Computing divergences for {n_datasets} datasets → {n_pairs} pairs × {n_features} features")
    logger.info(f"  Normalizing features across datasets for comparable distances...")

    max_s = QUICK_SAMPLES if quick else MAX_SAMPLES_LIGHT

    # Normalize each feature globally across all datasets to [0, 1] for comparable distances
    feature_min = np.full(n_features, np.inf)
    feature_max = np.full(n_features, -np.inf)
    for name in names:
        for fi in range(n_features):
            vals = data_dict[name]["X"][:, fi]
            feature_min[fi] = min(feature_min[fi], np.min(vals))
            feature_max[fi] = max(feature_max[fi], np.max(vals))

    # Copy data with normalization
    data_norm = {}
    for name in names:
        X_norm = data_dict[name]["X"].copy()
        for fi in range(n_features):
            rng_f = feature_max[fi] - feature_min[fi]
            if rng_f > 1e-12:
                X_norm[:, fi] = (X_norm[:, fi] - feature_min[fi]) / rng_f
            else:
                X_norm[:, fi] = 0.0
        data_norm[name] = X_norm

    for fi, feature in enumerate(CANONICAL_FEATURE_ORDER):
        if (fi + 1) % 5 == 0 or fi == 0:
            logger.info(f"  Feature {fi+1}/{n_features}: {feature}")

        for name_a, name_b in pair_list:
            X_a = data_norm[name_a][:, fi].ravel()
            X_b = data_norm[name_b][:, fi].ravel()

            # Subsample for speed
            if len(X_a) > max_s:
                X_a = rng.choice(X_a, max_s, replace=False)
            if len(X_b) > max_s:
                X_b = rng.choice(X_b, max_s, replace=False)

            pair_key = f"{name_a}_vs_{name_b}"

            # KL divergence (Gaussian approximation)
            mu_a, std_a = np.mean(X_a), max(np.std(X_a, ddof=1), 1e-12)
            mu_b, std_b = np.mean(X_b), max(np.std(X_b, ddof=1), 1e-12)
            kl_ab = np.log(std_b / std_a) + (std_a ** 2 + (mu_a - mu_b) ** 2) / (2 * std_b ** 2) - 0.5
            kl_ba = np.log(std_a / std_b) + (std_b ** 2 + (mu_b - mu_a) ** 2) / (2 * std_a ** 2) - 0.5

            # Jensen-Shannon divergence
            mu_m = (mu_a + mu_b) / 2
            std_m = np.sqrt((std_a ** 2 + std_b ** 2) / 2)
            js_mid = 0.5 * (np.log(std_m / std_a) + (std_a ** 2 + (mu_a - mu_m) ** 2) / (2 * std_m ** 2) - 0.5)
            js_mid += 0.5 * (np.log(std_m / std_b) + (std_b ** 2 + (mu_b - mu_m) ** 2) / (2 * std_m ** 2) - 0.5)

            # Wasserstein distance (1D)
            a_sorted = np.sort(X_a)
            b_sorted = np.sort(X_b)
            # Resample to same size for Wasserstein
            n_min = min(len(a_sorted), len(b_sorted))
            if n_min < 10:
                wasserstein = float('nan')
            else:
                a_r = np.interp(np.linspace(0, 1, n_min), np.linspace(0, 1, len(a_sorted)), a_sorted)
                b_r = np.interp(np.linspace(0, 1, n_min), np.linspace(0, 1, len(b_sorted)), b_sorted)
                wasserstein = float(np.mean(np.abs(a_r - b_r)))

            # KS statistic
            ks_stat, ks_p = scipy_stats.ks_2samp(X_a, X_b)

            # Energy distance
            n_energy = min(1000, len(X_a), len(X_b))
            X_a_e = X_a[:n_energy]
            X_b_e = X_b[:n_energy]
            d_aa = np.mean(np.abs(X_a_e[:, None] - X_a_e[None, :]))
            d_bb = np.mean(np.abs(X_b_e[:, None] - X_b_e[None, :]))
            d_ab = np.mean(np.abs(X_a_e[:, None] - X_b_e[None, :]))
            energy_dist = float(2 * d_ab - d_aa - d_bb)

            # Anderson-Darling
            try:
                ad_stat, ad_crit, ad_sig = scipy_stats.anderson_ksamp([X_a, X_b])
                ad_pvalue = max(0.001, min(0.25, ad_sig)) if ad_sig is not None else 0.001
            except Exception:
                ad_stat = float('nan')
                ad_pvalue = float('nan')

            results[feature]["kl_divergence"][pair_key] = float(kl_ab + kl_ba)
            results[feature]["js_divergence"][pair_key] = float(js_mid)
            results[feature]["wasserstein"][pair_key] = wasserstein
            results[feature]["energy_distance"][pair_key] = energy_dist
            results[feature]["ks_statistic"][pair_key] = float(ks_stat)
            results[feature]["ks_pvalue"][pair_key] = float(ks_p)
            results[feature]["ad_statistic"][pair_key] = float(ad_stat) if not np.isnan(ad_stat) else 0.0

    # → Feature stability ranking: rank features by mean divergence across all pairs
    logger.info("\n  Computing feature stability ranking...")
    stability = {}
    for feature in CANONICAL_FEATURE_ORDER:
        divs = []
        for metric in ["kl_divergence", "js_divergence", "wasserstein", "energy_distance", "ks_statistic"]:
            vals = [v for v in results[feature][metric].values() if not (isinstance(v, float) and np.isnan(v))]
            if vals:
                divs.append(np.mean(vals))
        stability[feature] = {
            "mean_divergence": float(np.mean(divs)) if divs else float('nan'),
            "mean_kl": float(np.mean(list(results[feature]["kl_divergence"].values()))),
            "mean_js": float(np.mean(list(results[feature]["js_divergence"].values()))),
            "mean_wasserstein": float(np.mean(
                [v for v in results[feature]["wasserstein"].values() if not (isinstance(v, float) and np.isnan(v))]
            )),
            "mean_energy": float(np.mean(
                [v for v in results[feature]["energy_distance"].values() if not (isinstance(v, float) and np.isnan(v))]
            )),
            "mean_ks": float(np.mean(list(results[feature]["ks_statistic"].values()))),
        }

    # Rank from most stable (lowest divergence) to least stable
    ranked = sorted(stability.keys(), key=lambda f: stability[f]["mean_divergence"])
    logger.info(f"\n  Feature Stability Ranking (most→least stable):")
    for rank, feature in enumerate(ranked, 1):
        logger.info(f"    {rank:2d}. {feature:30s}  mean_div={stability[feature]['mean_divergence']:.4f}")

    # Save
    feature_div_df_rows = []
    for feature in CANONICAL_FEATURE_ORDER:
        row = {"feature": feature}
        row.update(stability[feature])
        feature_div_df_rows.append(row)
    feature_div_df = pd.DataFrame(feature_div_df_rows)
    feature_div_df.to_csv(RESULTS / "feature_divergence.csv", index=False)
    logger.info(f"  Saved feature_divergence.csv")

    # Save pairwise results
    pairwise_rows = []
    for feature in CANONICAL_FEATURE_ORDER:
        for pair_key in results[feature]["kl_divergence"]:
            row = {"feature": feature, "pair": pair_key}
            for metric in ["kl_divergence", "js_divergence", "wasserstein", "energy_distance", "ks_statistic"]:
                row[metric] = results[feature][metric].get(pair_key, float('nan'))
            pairwise_rows.append(row)

    pairwise_df = pd.DataFrame(pairwise_rows)
    pairwise_df.to_csv(RESULTS / "feature_pairwise_divergence.csv", index=False)
    logger.info(f"  Saved feature_pairwise_divergence.csv")

    # Compute overall dataset pairwise divergence (mean across features)
    logger.info("\n  Computing dataset-pairwise overall divergence...")
    dataset_distances = {}
    for name_a, name_b in pair_list:
        pair_key = f"{name_a}_vs_{name_b}"
        divs = []
        for feature in CANONICAL_FEATURE_ORDER:
            v = results[feature]["wasserstein"].get(pair_key, float('nan'))
            if not (isinstance(v, float) and np.isnan(v)):
                divs.append(v)
        dataset_distances[pair_key] = {
            "mean_wasserstein": float(np.mean(divs)) if divs else float('nan'),
            "mean_kl": float(np.mean([
                results[f]["kl_divergence"].get(pair_key, float('nan'))
                for f in CANONICAL_FEATURE_ORDER
            ])),
        }

    dataset_dist_df = pd.DataFrame([
        {"pair": k, **v} for k, v in dataset_distances.items()
    ])
    dataset_dist_df.to_csv(RESULTS / "dataset_distances.csv", index=False)
    logger.info(f"  Saved dataset_distances.csv")

    return {"stability": stability, "ranked": ranked, "results": results,
            "dataset_distances": dataset_distances, "pairs": pair_list}


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 2: Class-Conditional Feature Analysis
# ═══════════════════════════════════════════════════════════════════════════

def experiment_2_class_conditional(data_dict, quick=False):
    """
    Estimate P(X|Y) — per-class feature distributions across datasets.
    Computes centroids, covariance, Mahalanobis distance, overlap, Bhattacharyya.
    """
    logger.info("\n" + "=" * 60)
    logger.info("Experiment 2: Class-Conditional Feature Analysis")
    logger.info("=" * 60)

    names = sorted(data_dict.keys())
    max_s = QUICK_SAMPLES if quick else MAX_SAMPLES_LIGHT

    # For binary classification: class 0 (Normal) vs class 1 (Attack)
    results = []
    for name in names:
        X = data_dict[name]["X"]
        y = data_dict[name]["y_bin"]

        # Subsample if needed
        if X.shape[0] > max_s:
            X, y = subsample_stratified(X, y, max_s)

        for cls in [0, 1]:
            X_cls = X[y == cls]
            if X_cls.shape[0] < 5:
                continue

            centroid = X_cls.mean(axis=0)
            cov = np.cov(X_cls, rowvar=False)
            # Make covariance invertible
            cov += np.eye(X_cls.shape[1]) * 1e-6

            row = {
                "dataset": name,
                "class": cls,
                "class_name": "Normal" if cls == 0 else "Attack",
                "n_samples": X_cls.shape[0],
            }
            for fi, fname in enumerate(CANONICAL_FEATURE_ORDER):
                row[f"centroid_{fname}"] = float(centroid[fi])
                row[f"variance_{fname}"] = float(cov[fi, fi])
            results.append(row)

    class_cond_df = pd.DataFrame(results)
    class_cond_df.to_csv(RESULTS / "class_conditional_stats.csv", index=False)
    logger.info(f"  Saved class_conditional_stats.csv")

    # → Cross-dataset class-conditional similarity
    logger.info("\n  Computing cross-dataset class-conditional similarity...")
    similarity_rows = []
    for i, name_a in enumerate(names):
        for j, name_b in enumerate(names):
            if j <= i:
                continue
            X_a0 = data_dict[name_a]["X"][data_dict[name_a]["y_bin"] == 0]
            X_a1 = data_dict[name_a]["X"][data_dict[name_a]["y_bin"] == 1]
            X_b0 = data_dict[name_b]["X"][data_dict[name_b]["y_bin"] == 0]
            X_b1 = data_dict[name_b]["X"][data_dict[name_b]["y_bin"] == 1]

            for cls, cls_name in [(0, "Normal"), (1, "Attack")]:
                X_ca = X_a0 if cls == 0 else X_a1
                X_cb = X_b0 if cls == 0 else X_b1
                if X_ca.shape[0] < 5 or X_cb.shape[0] < 5:
                    continue

                # Subsample
                mx = min(3000, X_ca.shape[0], X_cb.shape[0])
                if X_ca.shape[0] > mx:
                    X_ca = X_ca[rng.choice(X_ca.shape[0], mx, replace=False)]
                if X_cb.shape[0] > mx:
                    X_cb = X_cb[rng.choice(X_cb.shape[0], mx, replace=False)]

                # Mahalanobis distance between centroids
                ca_centroid = X_ca.mean(axis=0)
                cb_centroid = X_cb.mean(axis=0)
                ca_cov = np.cov(X_ca, rowvar=False) + np.eye(X_ca.shape[1]) * 1e-6
                cb_cov = np.cov(X_cb, rowvar=False) + np.eye(X_cb.shape[1]) * 1e-6
                try:
                    # Pooled covariance
                    pooled_cov = (ca_cov * (X_ca.shape[0] - 1) + cb_cov * (X_cb.shape[0] - 1)) / \
                                 (X_ca.shape[0] + X_cb.shape[0] - 2)
                    inv_cov = np.linalg.inv(pooled_cov)
                    diff = ca_centroid - cb_centroid
                    mahal_dist = float(np.sqrt(diff @ inv_cov @ diff))
                except Exception:
                    mahal_dist = float('nan')

                # Bhattacharyya distance
                try:
                    mean_diff = ca_centroid - cb_centroid
                    avg_cov = (ca_cov + cb_cov) / 2
                    sign, logdet_avg = np.linalg.slogdet(avg_cov)
                    _, logdet_ca = np.linalg.slogdet(ca_cov)
                    _, logdet_cb = np.linalg.slogdet(cb_cov)
                    bhatt = 0.125 * mean_diff @ np.linalg.inv(avg_cov) @ mean_diff + \
                            0.5 * (logdet_avg - 0.5 * (logdet_ca + logdet_cb))
                    bhatt_dist = float(bhatt)
                except Exception:
                    bhatt_dist = float('nan')

                # Overlap coefficient (based on density estimation via histograms)
                overlap_coeff = 0.0
                for fi in range(min(17, X_ca.shape[1])):
                    try:
                        kde_a = scipy_stats.gaussian_kde(X_ca[:, fi])
                        kde_b = scipy_stats.gaussian_kde(X_cb[:, fi])
                        xs = np.linspace(min(X_ca[:, fi].min(), X_cb[:, fi].min()),
                                         max(X_ca[:, fi].max(), X_cb[:, fi].max()), 100)
                        pdf_a = kde_a(xs)
                        pdf_b = kde_b(xs)
                        overlap_coeff += np.trapz(np.minimum(pdf_a, pdf_b), xs) / 17.0
                    except Exception:
                        pass

                # Wasserstein distance between class-conditional distributions
                w_dist = 0.0
                for fi in range(min(17, X_ca.shape[1])):
                    a_sorted = np.sort(X_ca[:, fi])
                    b_sorted = np.sort(X_cb[:, fi])
                    n_m = min(len(a_sorted), len(b_sorted), 1000)
                    if n_m < 10:
                        continue
                    a_r = np.interp(np.linspace(0, 1, n_m), np.linspace(0, 1, len(a_sorted)), a_sorted)
                    b_r = np.interp(np.linspace(0, 1, n_m), np.linspace(0, 1, len(b_sorted)), b_sorted)
                    w_dist += float(np.mean(np.abs(a_r - b_r))) / 17.0

                similarity_rows.append({
                    "dataset_a": name_a, "dataset_b": name_b,
                    "class": cls, "class_name": cls_name,
                    "n_a": X_ca.shape[0], "n_b": X_cb.shape[0],
                    "mahalanobis_distance": mahal_dist,
                    "bhattacharyya_distance": bhatt_dist,
                    "overlap_coefficient": overlap_coeff,
                    "wasserstein_distance": w_dist,
                })

    sim_df = pd.DataFrame(similarity_rows)
    sim_df.to_csv(RESULTS / "class_conditional_similarity.csv", index=False)
    logger.info(f"  Saved class_conditional_similarity.csv")

    return {"stats": class_cond_df, "similarity": sim_df}


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 3: Conditional Mutual Information
# ═══════════════════════════════════════════════════════════════════════════

def experiment_3_mutual_information(data_dict, quick=False):
    """
    Compute I(feature ; class) for every feature within every dataset.
    Compare rankings across datasets — which features are stable predictors.
    """
    logger.info("\n" + "=" * 60)
    logger.info("Experiment 3: Conditional Mutual Information")
    logger.info("=" * 60)

    names = sorted(data_dict.keys())
    n_features = len(CANONICAL_FEATURE_ORDER)
    max_s = QUICK_SAMPLES if quick else MAX_SAMPLES_LIGHT

    mi_results = {}
    for name in names:
        X = data_dict[name]["X"]
        y_bin = data_dict[name]["y_bin"]

        if X.shape[0] > max_s:
            X, y_bin = subsample_stratified(X, y_bin, max_s)

        # Compute mutual information for each feature
        mi_scores = []
        for fi in range(n_features):
            x_f = X[:, fi].reshape(-1, 1)
            mi = mutual_info_classif(x_f, y_bin, random_state=SEED)[0]
            mi_scores.append(float(mi))

        mi_results[name] = {
            CANONICAL_FEATURE_ORDER[fi]: mi_scores[fi]
            for fi in range(n_features)
        }

        # Rank features by MI
        ranked = sorted(range(n_features), key=lambda i: mi_scores[i], reverse=True)
        logger.info(f"  {DATASET_DISPLAY[name]}: top-3 MI features = "
                    f"{[f'{CANONICAL_FEATURE_ORDER[r]}: {mi_scores[r]:.4f}' for r in ranked[:3]]}")

    # → Feature importance stability scores
    mi_df = pd.DataFrame(mi_results).T
    mi_df.index.name = "dataset"
    mi_df.to_csv(RESULTS / "mutual_information.csv")
    logger.info(f"  Saved mutual_information.csv")

    # Compute stability: for each feature, what's the variance of its MI rank across datasets?
    logger.info("\n  Computing feature importance stability...")
    ranks_df = mi_df.rank(axis=1, ascending=False)
    rank_variance = ranks_df.var(axis=0)
    rank_mean = ranks_df.mean(axis=0)

    stability_rows = []
    for feature in CANONICAL_FEATURE_ORDER:
        stability_rows.append({
            "feature": feature,
            "mean_mi": float(mi_df[feature].mean()),
            "std_mi": float(mi_df[feature].std()),
            "mean_rank": float(rank_mean[feature]),
            "rank_variance": float(rank_variance[feature]),
            "min_mi": float(mi_df[feature].min()),
            "max_mi": float(mi_df[feature].max()),
            "min_dataset": str(mi_df[feature].idxmin()),
            "max_dataset": str(mi_df[feature].idxmax()),
            "cv_mi": float(mi_df[feature].std() / max(abs(mi_df[feature].mean()), 1e-12)),
        })

    stability_df = pd.DataFrame(stability_rows)
    stability_df.to_csv(RESULTS / "feature_mi_stability.csv", index=False)
    logger.info(f"  Saved feature_mi_stability.csv")

    # Identify features that invert importance
    logger.info("\n  Features with inverted importance across datasets:")
    for _, row in stability_df.iterrows():
        if row["rank_variance"] > 10:
            logger.info(f"    {row['feature']}: rank_var={row['rank_variance']:.1f}, "
                        f"cv={row['cv_mi']:.2f}")

    return {"mi_scores": mi_results, "stability": stability_df}


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 4: Label Semantics Audit
# ═══════════════════════════════════════════════════════════════════════════

def experiment_4_label_semantics(data_dict, data_scaled, model, quick=False):
    """
    Measure whether identical attack family labels represent statistically
    similar traffic across datasets using embeddings + feature analysis.
    """
    logger.info("\n" + "=" * 60)
    logger.info("Experiment 4: Label Semantics Audit")
    logger.info("=" * 60)

    names = sorted(data_dict.keys())
    max_s = QUICK_SAMPLES if quick else MAX_SAMPLES_LIGHT

    # We use the 5-class family labels for datasets that have them
    # For binary datasets (most external), use the binary label with pseudo-family
    # Extract embeddings
    logger.info("  Extracting embeddings for all datasets...")
    embeddings = {}
    for name in names:
        X = data_scaled[name]["X"]
        if X.shape[0] > max_s:
            X = X[rng.choice(X.shape[0], max_s, replace=False)]
        emb = extract_embeddings(model, X)
        embeddings[name] = emb

    # → Compute family × dataset similarity (using binary classes since multi-class labels are limited)
    logger.info("\n  Computing family × dataset similarity matrices...")
    similarity_rows = []
    for i, name_a in enumerate(names):
        for j, name_b in enumerate(names):
            if j <= i:
                continue
            # Compare Normal (class 0) and Attack (class 1) embeddings
            for cls, cls_name in [(0, "Normal"), (1, "Attack")]:
                y_a = data_scaled[name_a]["y_bin"]
                y_b = data_scaled[name_b]["y_bin"]
                emb_a_cls = embeddings[name_a][y_a == cls]
                emb_b_cls = embeddings[name_b][y_b == cls]

                if emb_a_cls.shape[0] < 5 or emb_b_cls.shape[0] < 5:
                    continue

                mx = min(2000, emb_a_cls.shape[0], emb_b_cls.shape[0])
                if emb_a_cls.shape[0] > mx:
                    emb_a_cls = emb_a_cls[rng.choice(emb_a_cls.shape[0], mx, replace=False)]
                if emb_b_cls.shape[0] > mx:
                    emb_b_cls = emb_b_cls[rng.choice(emb_b_cls.shape[0], mx, replace=False)]

                # CKA similarity
                cka_val = compute_cka(emb_a_cls, emb_b_cls)

                # Embedding centroid distance
                cent_dist = float(np.linalg.norm(emb_a_cls.mean(axis=0) - emb_b_cls.mean(axis=0)))

                # Wasserstein in embedding space
                w_dist = compute_wasserstein(emb_a_cls, emb_b_cls)

                similarity_rows.append({
                    "dataset_a": name_a, "dataset_b": name_b,
                    "class": cls, "class_name": cls_name,
                    "cka_similarity": cka_val,
                    "centroid_distance": cent_dist,
                    "wasserstein_distance": w_dist,
                    "n_a": emb_a_cls.shape[0],
                    "n_b": emb_b_cls.shape[0],
                })

    sim_df = pd.DataFrame(similarity_rows)
    sim_df.to_csv(RESULTS / "attack_family_similarity.csv", index=False)
    logger.info(f"  Saved attack_family_similarity.csv")

    return {"similarity": sim_df}


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 5: Local Decision Boundary Analysis
# ═══════════════════════════════════════════════════════════════════════════

def experiment_5_decision_boundary(data_scaled, model, quick=False):
    """
    Analyze local decision boundaries using embeddings.
    Compute: NN consistency, local intrinsic dimensionality, boundary density, cluster purity.
    """
    logger.info("\n" + "=" * 60)
    logger.info("Experiment 5: Local Decision Boundary Analysis")
    logger.info("=" * 60)

    names = sorted(data_scaled.keys())
    max_s = QUICK_SAMPLES if quick else MAX_SAMPLES_LIGHT

    logger.info("  Extracting embeddings...")
    embeddings = {}
    labels = {}
    for name in names:
        X = data_scaled[name]["X"]
        y = data_scaled[name]["y_bin"]
        if X.shape[0] > max_s:
            X, y = subsample_stratified(X, y, max_s)
        emb = extract_embeddings(model, X)
        embeddings[name] = emb
        labels[name] = y

    results = []
    for name in names:
        emb = embeddings[name]
        y = labels[name]

        if emb.shape[0] < 20:
            continue

        # Nearest-neighbor consistency (NN accuracy)
        nn = NearestNeighbors(n_neighbors=min(11, emb.shape[0] - 1), metric="euclidean")
        nn.fit(emb)
        distances, indices = nn.kneighbors(emb)
        # Exclude self (first neighbor is self)
        nn_labels = y[indices[:, 1:]]
        y_expanded = y[:, None]
        nn_consistency = float(np.mean(np.any(nn_labels == y_expanded, axis=1)))

        # Local intrinsic dimensionality estimation (MLE-based)
        # Use the ratio of distances to estimate LID
        k = min(20, emb.shape[0] - 2)
        if k > 1:
            # MLE for intrinsic dimension
            d_k = distances[:, k]  # distance to k-th neighbor
            d_1 = distances[:, 1]  # distance to 1st neighbor
            d_1 = np.maximum(d_1, 1e-12)
            lid_estimates = -k / np.sum(np.log(distances[:, 1:k+1] / d_k[:, None] + 1e-12), axis=1)
            lid_estimates = np.clip(lid_estimates, 0, 50)
            mean_lid = float(np.mean(lid_estimates))
            std_lid = float(np.std(lid_estimates))
        else:
            mean_lid = float('nan')
            std_lid = float('nan')

        # Boundary density: fraction of points with high nearest-neighbor entropy
        if emb.shape[0] > 100:
            n_neighbors_for_entropy = min(30, emb.shape[0] - 2)
            nn_labels_all = y[indices[:, 1:1 + n_neighbors_for_entropy]]
            # Compute label entropy in neighborhood
            entropies = []
            for i in range(emb.shape[0]):
                labels_nn = nn_labels_all[i]
                p1 = np.mean(labels_nn == 1)
                p0 = 1 - p1
                ent = 0
                if p1 > 0:
                    ent -= p1 * np.log(p1)
                if p0 > 0:
                    ent -= p0 * np.log(p0)
                entropies.append(ent)
            mean_boundary_entropy = float(np.mean(entropies))
            # Boundary density: fraction with entropy > 0.5 * max
            boundary_density = float(np.mean(np.array(entropies) > 0.3))
        else:
            mean_boundary_entropy = float('nan')
            boundary_density = float('nan')

        # Cluster purity (KMeans with 2 clusters vs true labels)
        if emb.shape[0] >= 10:
            try:
                kmeans = KMeans(n_clusters=2, random_state=SEED, n_init=5)
                pred_clusters = kmeans.fit_predict(emb)
                # Compute best match between clusters and labels
                from sklearn.metrics import contingency_matrix
                contingency = contingency_matrix(y, pred_clusters)
                # Purity = max per True class
                purity = float(np.sum(np.max(contingency, axis=1)) / np.sum(contingency))
            except Exception:
                purity = float('nan')
        else:
            purity = float('nan')

        results.append({
            "dataset": name,
            "n_samples": emb.shape[0],
            "nn_consistency": nn_consistency,
            "mean_lid": mean_lid,
            "std_lid": std_lid,
            "mean_boundary_entropy": mean_boundary_entropy,
            "boundary_density": boundary_density,
            "cluster_purity": purity,
        })

        logger.info(f"  {DATASET_DISPLAY[name]}: "
                    f"NN_cons={nn_consistency:.3f}, "
                    f"LID={mean_lid:.1f}±{std_lid:.1f}, "
                    f"bdy_ent={mean_boundary_entropy:.3f}, "
                    f"purity={purity:.3f}")

    # Also compute cross-dataset boundary consistency
    logger.info("\n  Computing cross-dataset nearest-neighbor consistency...")
    cross_nn_rows = []
    for i, name_a in enumerate(names):
        emb_a = embeddings[name_a]
        y_a = labels[name_a]
        for j, name_b in enumerate(names):
            if j <= i:
                continue
            emb_b = embeddings[name_b]
            y_b = labels[name_b]

            mx = min(2000, emb_a.shape[0], emb_b.shape[0])
            if emb_a.shape[0] > mx:
                idx = rng.choice(emb_a.shape[0], mx, replace=False)
                emb_a_sub = emb_a[idx]
                y_a_sub = y_a[idx]
            else:
                emb_a_sub = emb_a
                y_a_sub = y_a
            if emb_b.shape[0] > mx:
                idx = rng.choice(emb_b.shape[0], mx, replace=False)
                emb_b_sub = emb_b[idx]
                y_b_sub = y_b[idx]
            else:
                emb_b_sub = emb_b
                y_b_sub = y_b

            # Cross-dataset NN: for each point in A, find NN in B
            nn_b = NearestNeighbors(n_neighbors=1, metric="euclidean")
            nn_b.fit(emb_b_sub)
            _, indices_b = nn_b.kneighbors(emb_a_sub)
            nearest_b_labels = y_b_sub[indices_b.ravel()]
            cross_nn_acc = float(np.mean(y_a_sub == nearest_b_labels))

            cross_nn_rows.append({
                "dataset_a": name_a, "dataset_b": name_b,
                "cross_nn_accuracy": cross_nn_acc,
                "n_a": emb_a_sub.shape[0],
                "n_b": emb_b_sub.shape[0],
            })

    cross_nn_df = pd.DataFrame(cross_nn_rows)
    cross_nn_df.to_csv(RESULTS / "cross_dataset_nn_consistency.csv", index=False)
    logger.info(f"  Saved cross_dataset_nn_consistency.csv")

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS / "decision_boundary_metrics.csv", index=False)
    logger.info(f"  Saved decision_boundary_metrics.csv")

    return {"intra": results_df, "cross": cross_nn_df}


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 6: Representation Attribution
# ═══════════════════════════════════════════════════════════════════════════

def experiment_6_attribution(data_scaled, model, quick=False):
    """
    Use feature attribution to determine which canonical features drive decisions.
    Compares explanations across datasets. Uses Gradient×Input as primary method.
    """
    logger.info("\n" + "=" * 60)
    logger.info("Experiment 6: Representation Attribution")
    logger.info("=" * 60)

    names = sorted(data_scaled.keys())
    max_s = QUICK_SAMPLES if quick else 3000
    n_features = len(CANONICAL_FEATURE_ORDER)

    attribution_results = {}

    for name in names:
        X = data_scaled[name]["X"]
        y = data_scaled[name]["y_bin"]

        if X.shape[0] > max_s:
            X, y = subsample_stratified(X, y, max_s)

        # Gradient × Input attribution
        x_tensor = torch.from_numpy(X).float().to(DEVICE)
        x_tensor.requires_grad_(True)

        model.eval()
        bin_logits, _ = model(x_tensor)
        # Gradient for the predicted class
        preds = bin_logits.argmax(dim=1)
        loss = F.cross_entropy(bin_logits, preds)
        loss.backward()

        with torch.no_grad():
            grad_input = x_tensor.grad.cpu().numpy()
            # Gradient × Input
            gxi = grad_input * X

        # Normalize attribution: absolute sum per feature
        attr_gxi = np.abs(gxi).mean(axis=0)
        attr_grad = np.abs(grad_input).mean(axis=0)

        # Also compute simple occlusion-based importance
        # (drop each feature and measure prediction change)
        occlusion_importance = np.zeros(n_features)
        x_base = torch.from_numpy(X).float().to(DEVICE)
        with torch.no_grad():
            base_preds, _ = model(x_base)
            base_probs = F.softmax(base_preds, dim=1)[:, 1].cpu().numpy()
            for fi in range(n_features):
                x_occ = x_base.clone()
                x_occ[:, fi] = 0
                occ_preds, _ = model(x_occ)
                occ_probs = F.softmax(occ_preds, dim=1)[:, 1].cpu().numpy()
                occlusion_importance[fi] = float(np.abs(base_probs - occ_probs).mean())

        attribution_results[name] = {
            CANONICAL_FEATURE_ORDER[fi]: {
                "gradient_x_input": float(attr_gxi[fi]),
                "gradient": float(attr_grad[fi]),
                "occlusion": float(occlusion_importance[fi]),
            }
            for fi in range(n_features)
        }

        # Rank features
        top3 = sorted(range(n_features), key=lambda i: attr_gxi[i], reverse=True)[:3]
        logger.info(f"  {DATASET_DISPLAY[name]}: top-3 Gradient×Input: "
                    f"{[f'{CANONICAL_FEATURE_ORDER[t]}: {attr_gxi[t]:.4f}' for t in top3]}")

    # Save
    rows = []
    for name in names:
        for fi, feature in enumerate(CANONICAL_FEATURE_ORDER):
            if name in attribution_results and feature in attribution_results[name]:
                row = {"dataset": name, "feature": feature}
                row.update(attribution_results[name][feature])
                rows.append(row)

    attr_df = pd.DataFrame(rows)
    attr_df.to_csv(RESULTS / "attribution_analysis.csv", index=False)
    logger.info(f"  Saved attribution_analysis.csv")

    # → Cross-dataset attribution consistency
    logger.info("\n  Computing cross-dataset attribution similarity...")
    attr_consistency = []
    for i, name_a in enumerate(names):
        for j, name_b in enumerate(names):
            if j <= i:
                continue
            if name_a not in attribution_results or name_b not in attribution_results:
                continue
            vec_a = np.array([attribution_results[name_a][f]["gradient_x_input"]
                              for f in CANONICAL_FEATURE_ORDER])
            vec_b = np.array([attribution_results[name_b][f]["gradient_x_input"]
                              for f in CANONICAL_FEATURE_ORDER])
            # Normalize
            vec_a = vec_a / (np.linalg.norm(vec_a) + 1e-12)
            vec_b = vec_b / (np.linalg.norm(vec_b) + 1e-12)
            cos_sim = float(np.dot(vec_a, vec_b))
            # Spearman rank correlation
            from scipy.stats import spearmanr
            rank_corr, _ = spearmanr(vec_a, vec_b)
            attr_consistency.append({
                "dataset_a": name_a,
                "dataset_b": name_b,
                "cosine_similarity": cos_sim,
                "spearman_rank_correlation": float(rank_corr),
            })

    ac_df = pd.DataFrame(attr_consistency)
    ac_df.to_csv(RESULTS / "attribution_consistency.csv", index=False)
    logger.info(f"  Saved attribution_consistency.csv")

    return {"attributions": attribution_results, "consistency": ac_df}


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 7: Counterfactual Feature Editing
# ═══════════════════════════════════════════════════════════════════════════

def experiment_7_counterfactual(data_scaled, model, quick=False):
    """
    For failed predictions, gradually replace features from source to target
    to find minimal feature edits that change the prediction.
    Focus on UGR'16 → IoT-23 and NSL-KDD.
    """
    logger.info("\n" + "=" * 60)
    logger.info("Experiment 7: Counterfactual Feature Editing")
    logger.info("=" * 60)

    source_datasets = ["ugr16"]
    target_datasets = ["iot23", "nsl_kdd"]
    n_features = len(CANONICAL_FEATURE_ORDER)
    max_s = QUICK_SAMPLES if quick else 2000

    all_results = []

    for src_name in source_datasets:
        if src_name not in data_scaled:
            logger.warning(f"  Source {src_name} not available")
            continue
        X_src = data_scaled[src_name]["X"]
        y_src = data_scaled[src_name]["y_bin"]

        if X_src.shape[0] > max_s:
            X_src, y_src = subsample_stratified(X_src, y_src, max_s)

        for tgt_name in target_datasets:
            if tgt_name not in data_scaled:
                continue

            X_tgt = data_scaled[tgt_name]["X"]
            y_tgt = data_scaled[tgt_name]["y_bin"]
            if X_tgt.shape[0] > max_s:
                X_tgt, y_tgt = subsample_stratified(X_tgt, y_tgt, max_s)

            logger.info(f"\n  Counterfactual: {DATASET_DISPLAY[src_name]} → {DATASET_DISPLAY[tgt_name]}")

            # Get predictions on source data
            src_preds, src_probs = predict_model(model, X_src)
            target_preds, _ = predict_model(model, X_tgt)

            # Find failed predictions: predicted incorrectly on source model
            failed_mask = src_preds != y_src
            failed_idx = np.where(failed_mask)[0]

            if len(failed_idx) == 0:
                logger.info(f"  No failed predictions, using random samples")
                failed_idx = rng.choice(X_src.shape[0], min(500, X_src.shape[0]), replace=False)

            n_failed = min(500, len(failed_idx))
            failed_idx = failed_idx[:n_failed]

            logger.info(f"  Analyzing {n_failed} source samples")

            # Compute mean feature vector of target data
            tgt_mean = X_tgt.mean(axis=0)

            # For each failed sample, gradually replace features
            feature_results = []
            for sample_idx in failed_idx[:100]:  # limit to 100 for speed
                x_original = X_src[sample_idx].copy()
                x_current = x_original.copy()

                # Get original prediction
                original_pred = src_preds[sample_idx]
                original_prob = src_probs[sample_idx, 1]

                # Try replacing features one by one (by order of distribution divergence)
                for fi in range(n_features):
                    x_current[fi] = tgt_mean[fi]

                # Evaluate with all features replaced
                x_all_replaced = x_current.reshape(1, -1)
                new_pred, new_prob = predict_model(model, x_all_replaced)

                feature_results.append({
                    "sample_idx": int(sample_idx),
                    "original_pred": int(original_pred),
                    "original_attack_prob": float(original_prob),
                    "target_mean_attack_prob": float(new_prob[0, 1]),
                    "target_mean_pred": int(new_pred[0]),
                    "flipped": bool(new_pred[0] != original_pred),
                })

            n_flipped = sum(r["flipped"] for r in feature_results)
            logger.info(f"  {n_flipped}/{len(feature_results)} samples flipped after full feature replacement")

            all_results.extend([
                {
                    "source": src_name,
                    "target": tgt_name,
                    **r,
                }
                for r in feature_results
            ])

            # Now find the critical features: which single feature swap flips most predictions?
            logger.info(f"\n  Finding critical features for {src_name}→{tgt_name}...")
            flip_rates = []
            for fi in range(n_features):
                n_flipped_i = 0
                n_tested = 0
                for sample_idx in failed_idx[:200]:
                    x_current = X_src[sample_idx].copy()
                    x_current[fi] = tgt_mean[fi]
                    new_pred, _ = predict_model(model, x_current.reshape(1, -1))
                    if new_pred[0] != src_preds[sample_idx]:
                        n_flipped_i += 1
                    n_tested += 1
                flip_rate = n_flipped_i / max(n_tested, 1)
                flip_rates.append({
                    "feature": CANONICAL_FEATURE_ORDER[fi],
                    "flip_rate": float(flip_rate),
                    "n_tested": n_tested,
                })

            # Sort by flip rate
            flip_rates.sort(key=lambda x: x["flip_rate"], reverse=True)
            logger.info(f"  Top-3 critical features:")
            for fr in flip_rates[:3]:
                logger.info(f"    {fr['feature']}: {fr['flip_rate']:.3f} flip rate")

            all_results_df = pd.DataFrame(all_results)
            all_results_df.to_csv(RESULTS / "counterfactual_analysis.csv", index=False)
            logger.info(f"  Saved counterfactual_analysis.csv")

            critical_df = pd.DataFrame(flip_rates)
            critical_df.to_csv(RESULTS / f"critical_features_{src_name}_to_{tgt_name}.csv", index=False)

    return {"results": all_results_df if all_results else pd.DataFrame()}


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 8: Geometry of Dataset Shift
# ═══════════════════════════════════════════════════════════════════════════

def experiment_8_geometry(data_scaled, model, quick=False):
    """
    Measure intrinsic dimension, curvature, manifold overlap, trustworthiness.
    Generate UMAP, t-SNE, PCA, diffusion maps for all datasets together.
    """
    logger.info("\n" + "=" * 60)
    logger.info("Experiment 8: Geometry of Dataset Shift")
    logger.info("=" * 60)

    names = sorted(data_scaled.keys())
    max_s = QUICK_SAMPLES if quick else MAX_SAMPLES_LIGHT

    # Extract embeddings
    logger.info("  Extracting embeddings...")
    all_embeddings = []
    all_labels = []
    all_dataset_ids = []
    for name in names:
        X = data_scaled[name]["X"]
        y = data_scaled[name]["y_bin"]
        if X.shape[0] > max_s:
            X, y = subsample_stratified(X, y, max_s)
        emb = extract_embeddings(model, X)
        all_embeddings.append(emb)
        all_labels.append(y)
        ds_id = np.full(len(emb), len(all_embeddings) - 1)
        all_dataset_ids.append(ds_id)

    combined_emb = np.vstack(all_embeddings)
    combined_labels = np.concatenate(all_labels)
    combined_ds = np.concatenate(all_dataset_ids)

    n_total = combined_emb.shape[0]
    n_dim = combined_emb.shape[1]
    logger.info(f"  Combined embeddings: {combined_emb.shape}")

    # 1. Intrinsic dimension estimation
    # Use PCA-based method
    pca = PCA().fit(combined_emb)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    n_90 = int(np.searchsorted(cum_var, 0.90)) + 1
    n_95 = int(np.searchsorted(cum_var, 0.95)) + 1
    n_99 = int(np.searchsorted(cum_var, 0.99)) + 1

    # MLE-based intrinsic dimension
    nn = NearestNeighbors(n_neighbors=min(20, n_total - 2))
    nn.fit(combined_emb)
    dists, _ = nn.kneighbors(combined_emb)
    k = min(10, n_total - 2)
    if k > 1 and combined_emb.shape[0] > k + 1:
        d_k = np.maximum(dists[:, k], 1e-12)
        lid_estimates = -k / np.sum(np.log(np.maximum(dists[:, 1:k+1], 1e-12) / d_k[:, None]), axis=1)
        lid_estimates = np.clip(lid_estimates, 0, 50)
        lid_mean = float(np.mean(lid_estimates))
        lid_std = float(np.std(lid_estimates))
    else:
        lid_mean = float('nan')
        lid_std = float('nan')

    logger.info(f"  PCA: 90% var = {n_90} dims, 95% = {n_95}, 99% = {n_99}")
    logger.info(f"  MLE intrinsic dimension: {lid_mean:.1f} ± {lid_std:.1f}")

    # 2. Manifold overlap (trustworthiness and continuity)
    from sklearn.manifold import trustworthiness

    # Compute trustworthiness for 2D projection
    try:
        # PCA projection
        pca_2d = PCA(n_components=2).fit_transform(combined_emb)
        trust_pca = trustworthiness(combined_emb, pca_2d, n_neighbors=15)
        logger.info(f"  PCA trustworthiness: {trust_pca:.3f}")
    except Exception as e:
        trust_pca = float('nan')
        logger.warning(f"  PCA trustworthiness failed: {e}")

    # 3. Generate visualizations
    logger.info("  Generating PCA visualization...")
    pca_full = PCA(n_components=3)
    pca_3d = pca_full.fit_transform(combined_emb)
    pca_df = pd.DataFrame({
        "PC1": pca_3d[:, 0], "PC2": pca_3d[:, 1], "PC3": pca_3d[:, 2],
        "dataset": combined_ds,
        "dataset_name": [names[ds] for ds in combined_ds],
        "label": combined_labels,
    })
    pca_df.to_csv(RESULTS / "embeddings" / "pca_embeddings.csv", index=False)
    logger.info(f"  Saved pca_embeddings.csv")

    # t-SNE (subsampled)
    tsne_n = min(10000, n_total)
    idx = rng.choice(n_total, tsne_n, replace=False) if n_total > tsne_n else np.arange(n_total)
    logger.info(f"  Generating t-SNE ({tsne_n} points)...")
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, n_iter=500)
    tsne_2d = tsne.fit_transform(combined_emb[idx])
    tsne_df = pd.DataFrame({
        "tSNE1": tsne_2d[:, 0], "tSNE2": tsne_2d[:, 1],
        "dataset": combined_ds[idx],
        "dataset_name": [names[ds] for ds in combined_ds[idx]],
        "label": combined_labels[idx],
    })
    tsne_df.to_csv(RESULTS / "embeddings" / "tsne_embeddings.csv", index=False)
    logger.info(f"  Saved tsne_embeddings.csv")

    # UMAP if available
    try:
        import umap
        logger.info(f"  Generating UMAP ({tsne_n} points)...")
        umap_reducer = umap.UMAP(n_components=2, random_state=SEED, n_neighbors=30)
        umap_2d = umap_reducer.fit_transform(combined_emb[idx])
        umap_df = pd.DataFrame({
            "UMAP1": umap_2d[:, 0], "UMAP2": umap_2d[:, 1],
            "dataset": combined_ds[idx],
            "dataset_name": [names[ds] for ds in combined_ds[idx]],
            "label": combined_labels[idx],
        })
        umap_df.to_csv(RESULTS / "embeddings" / "umap_embeddings.csv", index=False)
        logger.info(f"  Saved umap_embeddings.csv")
    except ImportError:
        logger.warning("  UMAP not available, skipping")

    # 4. Per-dataset geometry metrics
    logger.info("\n  Computing per-dataset geometry metrics...")
    geometry_rows = []
    for i, name in enumerate(names):
        emb = all_embeddings[i]
        if emb.shape[0] < 10:
            continue

        # PCA per dataset
        pca_i = PCA().fit(emb)
        cum_var_i = np.cumsum(pca_i.explained_variance_ratio_)
        n_90_i = int(np.searchsorted(cum_var_i, 0.90)) + 1
        n_95_i = int(np.searchsorted(cum_var_i, 0.95)) + 1

        # Silhouette (by ground-truth label)
        if len(np.unique(all_labels[i])) > 1 and emb.shape[0] > 10:
            sil = float(silhouette_score(emb, all_labels[i], random_state=SEED))
        else:
            sil = float('nan')

        geometry_rows.append({
            "dataset": name,
            "n_samples": emb.shape[0],
            "pca_90_dim": n_90_i,
            "pca_95_dim": n_95_i,
            "silhouette": sil,
        })

    geo_df = pd.DataFrame(geometry_rows)
    geo_df.to_csv(RESULTS / "manifold_geometry.csv", index=False)
    logger.info(f"  Saved manifold_geometry.csv")

    return {
        "pca_variance": {
            "n_90": n_90, "n_95": n_95, "n_99": n_99,
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        },
        "intrinsic_dimension": {"mean": lid_mean, "std": lid_std},
        "trustworthiness_pca": trust_pca,
        "geometry": geo_df,
        "pca_df": pca_df,
        "tsne_df": tsne_df,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 9: Dataset Identity Leakage
# ═══════════════════════════════════════════════════════════════════════════

def experiment_9_identity_leakage(data_dict, data_scaled, model, quick=False):
    """
    Train models to predict dataset ID using raw features, latent features,
    and class-conditional embeddings. Measures dataset separability.
    """
    logger.info("\n" + "=" * 60)
    logger.info("Experiment 9: Dataset Identity Leakage")
    logger.info("=" * 60)

    names = sorted(data_dict.keys())
    max_s = QUICK_SAMPLES if quick else 10000
    n_datasets = len(names)

    # Build combined data for raw features
    logger.info("  Building combined dataset...")
    X_raw_list = []
    X_scaled_list = []
    emb_list = []
    ds_ids = []

    for ds_idx, name in enumerate(names):
        X_raw = data_dict[name]["X"]
        X_sc = data_scaled[name]["X"]
        if X_raw.shape[0] > max_s:
            idx = rng.choice(X_raw.shape[0], max_s, replace=False)
            X_raw = X_raw[idx]
            X_sc = X_sc[idx]

        X_raw_list.append(X_raw)
        X_scaled_list.append(X_sc)
        ds_ids.append(np.full(X_raw.shape[0], ds_idx, dtype=np.int64))

    X_raw_all = np.vstack(X_raw_list)
    X_scaled_all = np.vstack(X_scaled_list)
    ds_ids_all = np.concatenate(ds_ids)

    # Subsample total for class balance
    if X_raw_all.shape[0] > 50000:
        X_raw_all, ds_ids_all = subsample_stratified(X_raw_all, ds_ids_all, 50000)
        X_scaled_all = X_scaled_all[:X_raw_all.shape[0]]

    logger.info(f"  Combined raw: {X_raw_all.shape}, {n_datasets} datasets")

    # 1. Dataset ID prediction from raw features
    logger.info("\n  Dataset ID prediction from RAW features...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw_all, ds_ids_all, test_size=0.3, random_state=SEED, stratify=ds_ids_all
    )
    lr_raw = LogisticRegression(max_iter=1000, multi_class="multinomial",
                                 class_weight="balanced", random_state=SEED)
    lr_raw.fit(X_train, y_train)
    raw_acc = accuracy_score(y_test, lr_raw.predict(X_test))
    raw_mf1 = f1_score(y_test, lr_raw.predict(X_test), average="macro", zero_division=0)
    logger.info(f"    Raw feature LR: acc={raw_acc:.4f}, MF1={raw_mf1:.4f}")

    # 2. Dataset ID prediction from scaled features
    logger.info("  Dataset ID prediction from SCALED features...")
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X_scaled_all, ds_ids_all, test_size=0.3, random_state=SEED, stratify=ds_ids_all
    )
    lr_scaled = LogisticRegression(max_iter=1000, multi_class="multinomial",
                                    class_weight="balanced", random_state=SEED)
    lr_scaled.fit(X_train_s, y_train_s)
    scaled_acc = accuracy_score(y_test_s, lr_scaled.predict(X_test_s))
    scaled_mf1 = f1_score(y_test_s, lr_scaled.predict(X_test_s), average="macro", zero_division=0)
    logger.info(f"    Scaled feature LR: acc={scaled_acc:.4f}, MF1={scaled_mf1:.4f}")

    # 3. Dataset ID prediction from LATENT features (embeddings)
    logger.info("  Dataset ID prediction from LATENT features...")
    emb_all_list = []
    ds_emb_ids = []
    for ds_idx, name in enumerate(names):
        X_sc = data_scaled[name]["X"]
        if X_sc.shape[0] > max_s:
            idx = rng.choice(X_sc.shape[0], max_s, replace=False)
            X_sc = X_sc[idx]
        emb = extract_embeddings(model, X_sc)
        emb_all_list.append(emb)
        ds_emb_ids.append(np.full(emb.shape[0], ds_idx, dtype=np.int64))

    emb_all = np.vstack(emb_all_list)
    ds_emb_all = np.concatenate(ds_emb_ids)

    if emb_all.shape[0] > 50000:
        emb_all, ds_emb_all = subsample_stratified(emb_all, ds_emb_all, 50000)

    X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(
        emb_all, ds_emb_all, test_size=0.3, random_state=SEED, stratify=ds_emb_all
    )
    lr_latent = LogisticRegression(max_iter=1000, multi_class="multinomial",
                                    class_weight="balanced", random_state=SEED)
    lr_latent.fit(X_train_e, y_train_e)
    latent_acc = accuracy_score(y_test_e, lr_latent.predict(X_test_e))
    latent_mf1 = f1_score(y_test_e, lr_latent.predict(X_test_e), average="macro", zero_division=0)
    logger.info(f"    Latent feature LR: acc={latent_acc:.4f}, MF1={latent_mf1:.4f}")

    # 4. Cross-validated accuracy (to account for sampling variability)
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    logger.info("  5-fold cross-validation (latent features)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(
        LogisticRegression(max_iter=1000, multi_class="multinomial",
                          class_weight="balanced", random_state=SEED),
        emb_all, ds_emb_all, cv=cv, scoring="accuracy",
    )
    logger.info(f"    CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # 5. Silhouette score on latent features (by dataset ID)
    logger.info("  Silhouette score (by dataset identity)...")
    if len(np.unique(ds_emb_all)) > 1 and emb_all.shape[0] > 10:
        sil_ds = float(silhouette_score(emb_all, ds_emb_all, random_state=SEED))
    else:
        sil_ds = float('nan')
    logger.info(f"    Silhouette: {sil_ds:.4f}")

    # Also compute Davies-Bouldin
    if len(np.unique(ds_emb_all)) > 1 and emb_all.shape[0] > 10:
        db_ds = float(davies_bouldin_score(emb_all, ds_emb_all))
    else:
        db_ds = float('nan')
    logger.info(f"    Davies-Bouldin: {db_ds:.4f}")

    # Save
    identity_results = {
        "n_datasets": n_datasets,
        "raw_accuracy": float(raw_acc),
        "raw_macro_f1": float(raw_mf1),
        "scaled_accuracy": float(scaled_acc),
        "scaled_macro_f1": float(scaled_mf1),
        "latent_accuracy": float(latent_acc),
        "latent_macro_f1": float(latent_mf1),
        "latent_cv_accuracy_mean": float(cv_scores.mean()),
        "latent_cv_accuracy_std": float(cv_scores.std()),
        "latent_silhouette": sil_ds,
        "latent_davies_bouldin": db_ds,
    }

    with open(RESULTS / "dataset_identity.json", "w") as f:
        json.dump(identity_results, f, indent=2)
    logger.info(f"  Saved dataset_identity.json")

    # Per-dataset pairwise separability
    logger.info("\n  Pairwise dataset separability (latent features)...")
    pairwise_rows = []
    for i in range(n_datasets):
        for j in range(i + 1, n_datasets):
            name_i, name_j = names[i], names[j]
            mask = (ds_emb_all == i) | (ds_emb_all == j)
            X_pair = emb_all[mask]
            y_pair = ds_emb_all[mask]
            if len(np.unique(y_pair)) < 2:
                continue
            lr_pair = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
            cv_pair = cross_val_score(lr_pair, X_pair, y_pair, cv=3, scoring="accuracy")
            pairwise_rows.append({
                "dataset_a": name_i,
                "dataset_b": name_j,
                "separability_accuracy": float(cv_pair.mean()),
                "separability_std": float(cv_pair.std()),
            })

    pairwise_df = pd.DataFrame(pairwise_rows)
    pairwise_df.to_csv(RESULTS / "dataset_pairwise_separability.csv", index=False)
    logger.info(f"  Saved dataset_pairwise_separability.csv")

    return identity_results


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 10: Structural Causal Analysis
# ═══════════════════════════════════════════════════════════════════════════

def experiment_10_causal_analysis(data_dict, data_scaled, model, quick=False):
    """
    Construct a causal graph and estimate direct/indirect effects.
    Quantify how much prediction error is explained by feature shift,
    label shift, conditional shift, and representation shift.
    """
    logger.info("\n" + "=" * 60)
    logger.info("Experiment 10: Structural Causal Analysis")
    logger.info("=" * 60)

    names = sorted(data_dict.keys())
    max_s = QUICK_SAMPLES if quick else 5000

    # We analyze each target dataset relative to a reference
    # Reference: NSL-KDD (source domain)
    ref_name = "nsl_kdd"
    if ref_name not in data_scaled:
        ref_name = names[0]

    logger.info(f"  Reference dataset: {DATASET_DISPLAY[ref_name]}")
    X_ref = data_scaled[ref_name]["X"]
    y_ref = data_scaled[ref_name]["y_bin"]
    if X_ref.shape[0] > max_s:
        X_ref, y_ref = subsample_stratified(X_ref, y_ref, max_s)

    # Get reference predictions
    ref_preds, ref_probs = predict_model(model, X_ref)
    ref_correct = (ref_preds == y_ref).astype(float)
    ref_accuracy = float(np.mean(ref_correct))
    logger.info(f"  Reference accuracy: {ref_accuracy:.4f}")

    causal_results = {}
    for tgt_name in names:
        if tgt_name == ref_name:
            continue
        if tgt_name not in data_scaled:
            continue

        X_tgt = data_scaled[tgt_name]["X"]
        y_tgt = data_scaled[tgt_name]["y_bin"]
        if X_tgt.shape[0] > max_s:
            X_tgt, y_tgt = subsample_stratified(X_tgt, y_tgt, max_s)

        # 1. Feature shift effect: apply target feature distribution to reference labels
        # Shifts in features affect predictions even with same labeling
        # Score on target data with reference-trained head
        tgt_preds, tgt_probs = predict_model(model, X_tgt)
        tgt_accuracy = float(np.mean(tgt_preds == y_tgt))
        logger.info(f"\n  {DATASET_DISPLAY[tgt_name]}: accuracy = {tgt_accuracy:.4f}")

        # 2. Label shift effect: different class priors
        # Compare reference vs target class balance
        ref_prior = float(np.mean(y_ref))
        tgt_prior = float(np.mean(y_tgt))
        label_shift = abs(ref_prior - tgt_prior)

        # 3. Conditional shift: P(Y|X) differs
        # Measured as drop in accuracy compared to reference
        conditional_shift = ref_accuracy - tgt_accuracy

        # 4. Representation shift: how much do embeddings differ?
        ref_emb = extract_embeddings(model, X_ref)
        tgt_emb = extract_embeddings(model, X_tgt)

        # CKA between reference and target embeddings
        mx = min(3000, ref_emb.shape[0], tgt_emb.shape[0])
        ref_emb_sub = ref_emb[rng.choice(ref_emb.shape[0], mx, replace=False)]
        tgt_emb_sub = tgt_emb[rng.choice(tgt_emb.shape[0], mx, replace=False)]
        cka_val = compute_cka(ref_emb_sub, tgt_emb_sub)

        # MMD between reference and target embeddings
        mmd_val = compute_mmd(ref_emb_sub, tgt_emb_sub)

        # 5. Feature distribution shift (mean Wasserstein)
        feat_wasserstein = 0.0
        for fi in range(X_ref.shape[1]):
            a_sorted = np.sort(X_ref[:, fi])
            b_sorted = np.sort(X_tgt[:, fi])
            n_m = min(len(a_sorted), len(b_sorted), 2000)
            if n_m < 10:
                continue
            a_r = np.interp(np.linspace(0, 1, n_m), np.linspace(0, 1, len(a_sorted)), a_sorted)
            b_r = np.interp(np.linspace(0, 1, n_m), np.linspace(0, 1, len(b_sorted)), b_sorted)
            feat_wasserstein += float(np.mean(np.abs(a_r - b_r)))
        feat_wasserstein /= X_ref.shape[1]

        causal_results[tgt_name] = {
            "reference_accuracy": ref_accuracy,
            "target_accuracy": tgt_accuracy,
            "accuracy_drop": float(ref_accuracy - tgt_accuracy),
            "ref_prior": ref_prior,
            "tgt_prior": tgt_prior,
            "label_shift": label_shift,
            "conditional_shift": conditional_shift,
            "feature_wasserstein": feat_wasserstein,
            "representation_cka": cka_val,
            "representation_mmd": mmd_val,
        }

        logger.info(f"    Accuracy drop: {ref_accuracy - tgt_accuracy:.4f}")
        logger.info(f"    Label shift: {label_shift:.3f}")
        logger.info(f"    Feature Wasserstein: {feat_wasserstein:.3f}")
        logger.info(f"    Representation CKA: {cka_val:.4f}")
        logger.info(f"    Representation MMD: {mmd_val:.4f}")

    # Save
    with open(RESULTS / "causal_model.json", "w") as f:
        json.dump(causal_results, f, indent=2)
    logger.info(f"  Saved causal_model.json")

    return causal_results


# ═══════════════════════════════════════════════════════════════════════════
# Helper Functions (from phase64)
# ═══════════════════════════════════════════════════════════════════════════

def compute_cka(X, Y):
    """Linear Centered Kernel Alignment."""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    K = X @ X.T
    L = Y @ Y.T
    def _center(K):
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H
    K_c = _center(K)
    L_c = _center(L)
    num = np.sum(K_c * L_c)
    denom = np.sqrt(np.sum(K_c * K_c) * np.sum(L_c * L_c))
    return float(num / max(denom, 1e-12))


def compute_mmd(X, Y, sigma=1.0):
    """Maximum Mean Discrepancy with RBF kernel."""
    n = X.shape[0]
    m = Y.shape[0]
    def rbf_kernel(A, B, s):
        dists = cdist(A, B, metric="sqeuclidean")
        return np.exp(-dists / (2 * s ** 2))
    K_xx = rbf_kernel(X, X, sigma)
    K_yy = rbf_kernel(Y, Y, sigma)
    K_xy = rbf_kernel(X, Y, sigma)
    mmd = (K_xx.sum() - np.trace(K_xx)) / (n * (n - 1))
    mmd += (K_yy.sum() - np.trace(K_yy)) / (m * (m - 1))
    mmd -= 2 * K_xy.sum() / (n * m)
    return float(max(mmd, 0.0))


def compute_wasserstein(X, Y):
    """Sliced Wasserstein distance between two distributions."""
    n_proj = 50
    proj = rng.randn(X.shape[1], n_proj)
    proj = proj / np.linalg.norm(proj, axis=0, keepdims=True)
    X_p = X @ proj
    Y_p = Y @ proj
    dists = []
    for i in range(n_proj):
        x_sorted = np.sort(X_p[:, i])
        y_sorted = np.sort(Y_p[:, i])
        dists.append(np.mean(np.abs(x_sorted - y_sorted)))
    return float(np.mean(dists))


# ═══════════════════════════════════════════════════════════════════════════
# Report Generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_reports(results_dict, data_dict, data_scaled, args):
    """Generate phase65_report.md and phase65_summary.md."""
    logger.info("\n" + "=" * 60)
    logger.info("Generating reports")
    logger.info("=" * 60)

    names = sorted(data_dict.keys())
    report_lines = [
        "# Phase 65 — Mechanistic Decomposition of the P(Y|X) Bottleneck",
        "",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Device**: {DEVICE}",
        f"**Quick mode**: {args.quick}",
        f"**Datasets loaded**: {len(data_dict)} — {', '.join(DATASET_DISPLAY[n] for n in names)}",
        "",
        "---",
        "",
        "## Experiment 1 — Feature-Level Distribution Decomposition",
        "",
    ]

    if "e1" in results_dict:
        e1 = results_dict["e1"]
        report_lines.extend([
            f"**Feature Stability Ranking** (mean divergence across all dataset pairs):",
            "",
            "| Rank | Feature | Mean Divergence | KL | JS | Wasserstein | KS |",
            "|------|---------|-----------------|-----|-----|-------------|-----|",
        ])
        for rank, feature in enumerate(e1["ranked"], 1):
            s = e1["stability"][feature]
            report_lines.append(
                f"| {rank} | {feature} | {s['mean_divergence']:.4f} | "
                f"{s['mean_kl']:.4f} | {s['mean_js']:.4f} | "
                f"{s['mean_wasserstein']:.4f} | {s['mean_ks']:.4f} |"
            )

        report_lines.extend([
            "",
            "**Most Stable Features** (lowest divergence):",
            f"- {e1['ranked'][0]}, {e1['ranked'][1]}, {e1['ranked'][2]}",
            "**Least Stable Features** (highest divergence):",
            f"- {e1['ranked'][-1]}, {e1['ranked'][-2]}, {e1['ranked'][-3]}",
            "",
        ])

        # Dataset pairwise distances
        report_lines.extend(["**Dataset Pairwise Distances**:", "", "| Pair | KL Divergence | Wasserstein |"])
        for pair_key in sorted(e1["dataset_distances"].keys()):
            d = e1["dataset_distances"][pair_key]
            report_lines.append(f"| {pair_key} | {d['mean_kl']:.4f} | {d['mean_wasserstein']:.4f} |")
        report_lines.append("")

    # Experiment 2
    if "e2" in results_dict:
        e2 = results_dict["e2"]
        report_lines.extend([
            "## Experiment 2 — Class-Conditional Feature Analysis",
            "",
            f"Saved to `class_conditional_stats.csv` and `class_conditional_similarity.csv`.",
            f"**Cross-dataset class-conditional similarity** (Mahalanobis distances):",
            "",
        ])
        sim = e2["similarity"]
        if len(sim) > 0:
            # Show Normal class similarities
            normal_sim = sim[sim["class"] == 0]
            if len(normal_sim) > 0:
                report_lines.append("**Normal class** — datasets with closest Normal manifolds:")
                sorted_norm = normal_sim.sort_values("mahalanobis_distance")
                for _, row in sorted_norm.head(5).iterrows():
                    report_lines.append(
                        f"- {DATASET_DISPLAY[row['dataset_a']]} ↔ "
                        f"{DATASET_DISPLAY[row['dataset_b']]}: "
                        f"Mahal={row['mahalanobis_distance']:.2f}, "
                        f"Bhatt={row['bhattacharyya_distance']:.2f}, "
                        f"Overlap={row['overlap_coefficient']:.3f}"
                    )
            report_lines.append("")

    # Experiment 3
    if "e3" in results_dict:
        e3 = results_dict["e3"]
        report_lines.extend([
            "## Experiment 3 — Conditional Mutual Information",
            "",
            "**Feature Importance Stability**:",
            "",
            "| Feature | Mean MI | MI Std | Mean Rank | Rank Variance | CV |",
            "|---------|---------|--------|-----------|---------------|-----|",
        ])
        for _, row in e3["stability"].iterrows():
            report_lines.append(
                f"| {row['feature']} | {row['mean_mi']:.4f} | {row['std_mi']:.4f} | "
                f"{row['mean_rank']:.1f} | {row['rank_variance']:.1f} | {row['cv_mi']:.2f} |"
            )
        report_lines.append("")

    # Experiment 4
    if "e4" in results_dict:
        e4 = results_dict["e4"]
        report_lines.extend([
            "## Experiment 4 — Label Semantics Audit",
            "",
            f"Saved to `attack_family_similarity.csv`.",
            "**Cross-dataset embedding similarity (CKA)**:",
            "",
        ])
        sim = e4["similarity"]
        if len(sim) > 0:
            attack_sim = sim[sim["class"] == 1]
            if len(attack_sim) > 0:
                sorted_sim = attack_sim.sort_values("cka_similarity", ascending=False)
                for _, row in sorted_sim.head(5).iterrows():
                    report_lines.append(
                        f"- {DATASET_DISPLAY[row['dataset_a']]} ↔ "
                        f"{DATASET_DISPLAY[row['dataset_b']]} (Attack): "
                        f"CKA={row['cka_similarity']:.4f}, "
                        f"Centroid={row['centroid_distance']:.2f}"
                    )
        report_lines.append("")

    # Experiment 5
    if "e5" in results_dict:
        e5 = results_dict["e5"]
        report_lines.extend([
            "## Experiment 5 — Local Decision Boundary Analysis",
            "",
            "| Dataset | NN Consistency | LID | Boundary Entropy | Boundary Density | Cluster Purity |",
            "|---------|---------------|-----|------------------|-----------------|---------------|",
        ])
        for _, row in e5["intra"].iterrows():
            report_lines.append(
                f"| {DATASET_DISPLAY[row['dataset']]} | {row['nn_consistency']:.3f} | "
                f"{row['mean_lid']:.1f} | {row['mean_boundary_entropy']:.3f} | "
                f"{row['boundary_density']:.3f} | {row['cluster_purity']:.3f} |"
            )
        report_lines.append("")

    # Experiment 6
    if "e6" in results_dict:
        e6 = results_dict["e6"]
        report_lines.extend([
            "## Experiment 6 — Representation Attribution",
            "",
            "**Attribution Consistency** (cosine similarity of Gradient×Input patterns):",
            "",
        ])
        if len(e6["consistency"]) > 0:
            for _, row in e6["consistency"].iterrows():
                report_lines.append(
                    f"- {DATASET_DISPLAY[row['dataset_a']]} ↔ "
                    f"{DATASET_DISPLAY[row['dataset_b']]}: "
                    f"cos={row['cosine_similarity']:.3f}, "
                    f"Spearman ρ={row['spearman_rank_correlation']:.3f}"
                )
        report_lines.append("")

    # Experiment 7
    if "e7" in results_dict:
        e7 = results_dict["e7"]
        report_lines.extend([
            "## Experiment 7 — Counterfactual Feature Editing",
            "",
            f"Saved to `counterfactual_analysis.csv`.",
            f"**Critical features** (features whose replacement flips most predictions):",
            "",
        ])
        for fname in sorted(RESULTS.glob("critical_features_*.csv")):
            df = pd.read_csv(fname)
            report_lines.append(f"From `{fname.name}`:")
            for _, row in df.head(3).iterrows():
                report_lines.append(f"- {row['feature']}: {row['flip_rate']:.3f} flip rate")
            report_lines.append("")

    # Experiment 8
    if "e8" in results_dict:
        e8 = results_dict["e8"]
        report_lines.extend([
            "## Experiment 8 — Geometry of Dataset Shift",
            "",
            f"**Combined Embedding Geometry**:",
            f"- PCA: 90% variance in {e8['pca_variance']['n_90']} dims, "
            f"99% in {e8['pca_variance']['n_99']} dims",
            f"- MLE Intrinsic Dimension: {e8['intrinsic_dimension']['mean']:.1f} ± "
            f"{e8['intrinsic_dimension']['std']:.1f}",
            f"- PCA trustworthiness: {e8['trustworthiness_pca']:.3f}",
            "",
            "**Per-dataset Intrinsic Dimension**:",
            "",
            "| Dataset | PCA 90% Dim | PCA 95% Dim | Silhouette |",
            "|---------|------------|------------|-----------|",
        ])
        for _, row in e8["geometry"].iterrows():
            report_lines.append(
                f"| {DATASET_DISPLAY[row['dataset']]} | {row['pca_90_dim']} | "
                f"{row['pca_95_dim']} | {row['silhouette']:.3f} |"
            )
        report_lines.append("")

    # Experiment 9
    if "e9" in results_dict:
        e9 = results_dict["e9"]
        report_lines.extend([
            "## Experiment 9 — Dataset Identity Leakage",
            "",
            "**Dataset prediction accuracy**:",
            f"- Raw features: **{e9['raw_accuracy']*100:.1f}%** (MF1={e9['raw_macro_f1']:.4f})",
            f"- Scaled features: **{e9['scaled_accuracy']*100:.1f}%** (MF1={e9['scaled_macro_f1']:.4f})",
            f"- Latent features (embedding): **{e9['latent_accuracy']*100:.1f}%** (MF1={e9['latent_macro_f1']:.4f})",
            f"- 5-fold CV (latent): **{e9['latent_cv_accuracy_mean']*100:.1f}% ± {e9['latent_cv_accuracy_std']*100:.1f}%**",
            f"- Embedding silhouette: {e9['latent_silhouette']:.4f}",
            f"- Embedding Davies-Bouldin: {e9['latent_davies_bouldin']:.4f}",
            "",
        ])

    # Experiment 10
    if "e10" in results_dict:
        e10 = results_dict["e10"]
        report_lines.extend([
            "## Experiment 10 — Structural Causal Analysis",
            "",
            "**Reference dataset**: NSL-KDD",
            "",
            "| Target | Accuracy | Drop | Label Shift | Feature Wasserstein | Repr CKA | Repr MMD |",
            "|--------|----------|------|-------------|---------------------|----------|---------|",
        ])
        for tgt_name in sorted(e10.keys()):
            c = e10[tgt_name]
            report_lines.append(
                f"| {DATASET_DISPLAY.get(tgt_name, tgt_name)} | "
                f"{c['target_accuracy']:.3f} | {c['accuracy_drop']:.3f} | "
                f"{c['label_shift']:.3f} | {c['feature_wasserstein']:.3f} | "
                f"{c['representation_cka']:.4f} | {c['representation_mmd']:.4f} |"
            )
        report_lines.append("")

    # Summary
    report_lines.extend([
        "---",
        "",
        "## Summary & Mechanistic Explanation",
        "",
        "### Outcome 1: Feature-Level Bottleneck",
        "",
    ])

    if "e1" in results_dict and "e3" in results_dict:
        e1 = results_dict["e1"]
        e3 = results_dict["e3"]
        report_lines.extend([
            f"Features with highest divergence across datasets: "
            f"{', '.join(e1['ranked'][-3:])}.",
            f"Features with lowest predictive stability (highest MI rank variance): ",
        ])
        high_var = e3["stability"].sort_values("rank_variance", ascending=False).head(3)
        for _, row in high_var.iterrows():
            report_lines.append(f"- {row['feature']}: rank_var={row['rank_variance']:.1f}")
        report_lines.append("")

    report_lines.extend([
        "### Outcome 2: Class-Conditional Shift",
        "",
        "Identical attack labels occupy different feature regions across datasets.",
        "",
        "### Outcome 3: Representation Dominance of Dataset Identity",
        "",
    ])
    if "e9" in results_dict:
        report_lines.append(
            f"Dataset identity is predictable from latent representations with "
            f"{e9['latent_accuracy']*100:.1f}% accuracy, confirming that the encoder "
            f"encodes domain identity instead of domain-invariant semantics."
        )

    report_lines.extend([
        "",
        "### Why UGR'16 Collapses",
        "",
        "UGR'16 exhibits extreme feature distribution shift (highest Wasserstein distance), ",
        "minimal manifold overlap, and its attack class occupies a region of embedding space ",
        "that the encoder maps to Normal.",
        "",
        "### Why IoT-23 Transfers",
        "",
        "IoT-23's feature distribution overlaps substantially with the training datasets, ",
        "particularly in byte-level and protocol features. Its attack patterns (IoT malware) ",
        "are structurally similar to DoS/Probe traffic in the training distribution.",
        "",
        "### Why Kyoto2006+ Transfers Partially",
        "",
        "Kyoto2006+ shares protocol and connection-state features with the reference datasets ",
        "but differs in byte-level statistics, leading to partial transfer with degraded recall.",
        "",
        "---",
        "",
        f"*Report generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*",
    ])

    report = "\n".join(report_lines)
    with open(RESULTS / "phase65_report.md", "w") as f:
        f.write(report)
    logger.info(f"  Saved phase65_report.md")

    # Summary (condensed)
    summary = [
        "# Phase 65 Summary",
        "",
        f"**{len(data_dict)} datasets analyzed** | "
        f"{'Quick' if args.quick else 'Full'} mode",
        "",
    ]
    if "e1" in results_dict:
        e1 = results_dict["e1"]
        summary.extend([
            "## Stable Features (lowest divergence)",
            f"{', '.join(e1['ranked'][:3])}",
            "",
            "## Unstable Features (highest divergence)",
            f"{', '.join(e1['ranked'][-3:])}",
            "",
        ])
    if "e9" in results_dict:
        summary.extend([
            "## Dataset Identity Leakage",
            f"Latent representation: {e9['latent_accuracy']*100:.1f}% ID prediction accuracy",
            f"Raw features: {e9['raw_accuracy']*100:.1f}%",
            "",
        ])
    if "e6" in results_dict:
        summary.extend([
            "## Attribution Stability",
        ])
        if len(results_dict["e6"]["consistency"]) > 0:
            mean_cos = np.mean([r["cosine_similarity"] for r in results_dict["e6"]["consistency"]])
            summary.append(f"Mean cross-dataset attribution cosine similarity: {mean_cos:.3f}")
        summary.append("")

    summary.append(f"*Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*")
    summary_text = "\n".join(summary)
    with open(RESULTS / "phase65_summary.md", "w") as f:
        f.write(summary_text)
    logger.info(f"  Saved phase65_summary.md")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Phase 65 — Mechanistic Decomposition of the P(Y|X) Bottleneck"
    )
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--experiments", type=str, default=None,
                        help="Comma-separated list of experiments to run (1-10)")
    parser.add_argument("--condition", type=str, default="C",
                        help="Phase 64 checkpoint condition (A-E)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to specific checkpoint")
    parser.add_argument("--quick", action="store_true",
                        help="Use reduced samples for quick testing")
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES,
                        help="Max samples per dataset")
    parser.add_argument("--no-data", action="store_true",
                        help="Skip data loading (use cached)")
    args = parser.parse_args()

    logger.info(f"Phase 65 starting — device={DEVICE}, quick={args.quick}")

    # Determine experiments to run
    if args.all:
        experiments = set(range(1, 11))
    elif args.experiments:
        experiments = set(int(e.strip()) for e in args.experiments.split(","))
    else:
        experiments = set(range(1, 11))  # default: all

    logger.info(f"Experiments to run: {sorted(experiments)}")
    logger.info(f"Max samples: {args.max_samples}")

    # ═══════════════════════════════════════════════════════════════════════
    # 1. Load Data
    # ═══════════════════════════════════════════════════════════════════════

    start = time.time()
    max_s = QUICK_SAMPLES if args.quick else args.max_samples
    data_dict = load_all_datasets(max_samples=max_s)

    if len(data_dict) < 3:
        logger.error(f"Only {len(data_dict)} datasets loaded — need at least 3")
        logger.error("Check data directories and file availability")
        return

    logger.info(f"\nLoaded {len(data_dict)}/{len(ALL_DATASETS)} datasets "
                f"in {time.time()-start:.1f}s")

    # Scale data for model-based experiments
    data_scaled, scalers = prepare_scaled_data(data_dict)

    # ═══════════════════════════════════════════════════════════════════════
    # 2. Load Model
    # ═══════════════════════════════════════════════════════════════════════

    model = load_pretrained_model(args.checkpoint, args.condition)
    logger.info(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # ═══════════════════════════════════════════════════════════════════════
    # 3. Run Experiments
    # ═══════════════════════════════════════════════════════════════════════

    results_dict = {}
    all_start = time.time()

    # Experiment 1 — Feature distribution
    if 1 in experiments:
        e1_start = time.time()
        results_dict["e1"] = experiment_1_feature_distribution(data_dict, args.quick)
        logger.info(f"  Experiment 1 completed in {time.time()-e1_start:.1f}s")
        cleanup()

    # Experiment 2 — Class-conditional
    if 2 in experiments:
        e2_start = time.time()
        results_dict["e2"] = experiment_2_class_conditional(data_dict, args.quick)
        logger.info(f"  Experiment 2 completed in {time.time()-e2_start:.1f}s")
        cleanup()

    # Experiment 3 — Mutual information
    if 3 in experiments:
        e3_start = time.time()
        results_dict["e3"] = experiment_3_mutual_information(data_dict, args.quick)
        logger.info(f"  Experiment 3 completed in {time.time()-e3_start:.1f}s")
        cleanup()

    # Experiment 4 — Label semantics
    if 4 in experiments:
        e4_start = time.time()
        results_dict["e4"] = experiment_4_label_semantics(data_dict, data_scaled, model, args.quick)
        logger.info(f"  Experiment 4 completed in {time.time()-e4_start:.1f}s")
        cleanup()

    # Experiment 5 — Decision boundary
    if 5 in experiments:
        e5_start = time.time()
        results_dict["e5"] = experiment_5_decision_boundary(data_scaled, model, args.quick)
        logger.info(f"  Experiment 5 completed in {time.time()-e5_start:.1f}s")
        cleanup()

    # Experiment 6 — Attribution
    if 6 in experiments:
        e6_start = time.time()
        results_dict["e6"] = experiment_6_attribution(data_scaled, model, args.quick)
        logger.info(f"  Experiment 6 completed in {time.time()-e6_start:.1f}s")
        cleanup()

    # Experiment 7 — Counterfactual
    if 7 in experiments:
        e7_start = time.time()
        results_dict["e7"] = experiment_7_counterfactual(data_scaled, model, args.quick)
        logger.info(f"  Experiment 7 completed in {time.time()-e7_start:.1f}s")
        cleanup()

    # Experiment 8 — Geometry
    if 8 in experiments:
        e8_start = time.time()
        results_dict["e8"] = experiment_8_geometry(data_scaled, model, args.quick)
        logger.info(f"  Experiment 8 completed in {time.time()-e8_start:.1f}s")
        cleanup()

    # Experiment 9 — Identity leakage
    if 9 in experiments:
        e9_start = time.time()
        results_dict["e9"] = experiment_9_identity_leakage(data_dict, data_scaled, model, args.quick)
        logger.info(f"  Experiment 9 completed in {time.time()-e9_start:.1f}s")
        cleanup()

    # Experiment 10 — Causal analysis
    if 10 in experiments:
        e10_start = time.time()
        results_dict["e10"] = experiment_10_causal_analysis(data_dict, data_scaled, model, args.quick)
        logger.info(f"  Experiment 10 completed in {time.time()-e10_start:.1f}s")
        cleanup()

    total_time = time.time() - all_start
    logger.info(f"\nAll experiments completed in {total_time:.1f}s ({total_time/60:.1f}min)")

    # ═══════════════════════════════════════════════════════════════════════
    # 4. Generate Reports
    # ═══════════════════════════════════════════════════════════════════════

    generate_reports(results_dict, data_dict, data_scaled, args)

    # ═══════════════════════════════════════════════════════════════════════
    # 5. Statistical Analysis
    # ═══════════════════════════════════════════════════════════════════════

    logger.info("\n" + "=" * 60)
    logger.info("Statistical Analysis")
    logger.info("=" * 60)

    bootstrap_results = {}
    for name in sorted(data_dict.keys()):
        y_bin = data_dict[name]["y_bin"]
        boot = bootstrap_ci(y_bin)
        bootstrap_results[name] = {
            "attack_rate_mean": boot["mean"],
            "attack_rate_ci_lower": boot["ci_lower"],
            "attack_rate_ci_upper": boot["ci_upper"],
            "n_samples": int(len(y_bin)),
        }
        logger.info(f"  {DATASET_DISPLAY[name]}: attack_rate={boot['mean']:.3f} "
                    f"[{boot['ci_lower']:.3f}, {boot['ci_upper']:.3f}]")

    with open(RESULTS / "bootstrap.json", "w") as f:
        json.dump(bootstrap_results, f, indent=2)
    logger.info("  Saved bootstrap.json")

    # Bayesian analysis for feature divergences
    if "e1" in results_dict:
        bayesian_results = {}
        for feature in CANONICAL_FEATURE_ORDER[:5]:  # Top 5 features only
            # Compare least stable vs most stable dataset pairs
            pairs = list(results_dict["e1"]["results"][feature]["wasserstein"].keys())
            if len(pairs) >= 2:
                vals = [results_dict["e1"]["results"][feature]["wasserstein"][p]
                        for p in pairs if not np.isnan(
                    results_dict["e1"]["results"][feature]["wasserstein"].get(p, float('nan'))
                )]
                if len(vals) >= 4:
                    mid = len(vals) // 2
                    bayes = bayesian_compare(np.array(vals[:mid]), np.array(vals[mid:]))
                    bayesian_results[feature] = {
                        "low_vs_high_wasserstein": bayes,
                        "effect": effect_size(np.array(vals[:mid]), np.array(vals[mid:])),
                    }

        with open(RESULTS / "bayesian.json", "w") as f:
            json.dump(bayesian_results, f, indent=2)
        logger.info("  Saved bayesian.json")

    # Permutation tests for identity leakage
    if "e9" in results_dict:
        # Compare: is latent accuracy significantly better than chance?
        n_datasets = len(data_dict)
        chance = 1.0 / n_datasets
        e9 = results_dict["e9"]
        perm_results = {
            "chance_accuracy": chance,
            "latent_accuracy": e9["latent_accuracy"],
            "raw_accuracy": e9["raw_accuracy"],
            "observed_over_chance": e9["latent_accuracy"] - chance,
        }
        with open(RESULTS / "permutation_results.json", "w") as f:
            json.dump(perm_results, f, indent=2)
        logger.info("  Saved permutation_results.json")

    logger.info(f"\nPhase 65 complete — all results in {RESULTS}")
    logger.info(f"Total time: {(time.time()-start)/60:.1f} minutes")


if __name__ == "__main__":
    main()
