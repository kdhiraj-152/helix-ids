#!/usr/bin/env python3
"""
Phase 64 — Multi-Dataset Foundation Pretraining:
Can Joint Pretraining Eliminate the P(Y|X) Bottleneck?

Determines whether the P(Y|X) bottleneck arises because HELIX was pretrained
on a single source dataset (NSL-KDD) or whether it is an intrinsic property
of NIDS datasets.

Unlike Phases 60–63, no sequential adaptation is performed.
Instead, train a new HELIX foundation encoder jointly on all available source
datasets before evaluating zero-shot transfer.

Conditions:
  A: Single-source baseline (NSL-KDD only)
  B: Joint multi-source pretraining (equal sampling from all 6)
  C: Joint pretraining + supervised contrastive loss
  D: Joint pretraining without BatchNorm (RMSNorm)
  E: Joint pretraining + replay curriculum

Usage:
  PYTHONPATH=src python scripts/analysis/phase64_main.py --condition A
  PYTHONPATH=src python scripts/analysis/phase64_main.py --condition B
  PYTHONPATH=src python scripts/analysis/phase64_main.py --all
  PYTHONPATH=src python scripts/analysis/phase64_main.py --condition C --eval_only
  PYTHONPATH=src python scripts/analysis/phase64_main.py --all --quick
"""

import argparse
import gc
import hashlib
import json
import logging
import os
import sys
import time
import warnings
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
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
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.linalg import svd
import sklearn.metrics as sk_metrics
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import CCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

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
BATCH_SIZE = 256
LR = 1e-4
WEIGHT_DECAY = 1e-4
N_BOOTSTRAP = 10000
MAX_TRAIN_SAMPLES = 20000
MAX_TEST_SAMPLES = 10000
PRETRAIN_EPOCHS = 50
PROBE_EPOCHS = 30

PROJ = Path(__file__).resolve().parents[2]
CACHE = PROJ / "data" / "processed" / "phase52_cache"
RESULTS = PROJ / "results" / "phase64"
DATA_DIR = PROJ / "data"

ORIGINAL_DATASETS = [
    "nsl_kdd", "unsw_nb15", "cicids2017", "cicids2018", "ton_iot", "bot_iot"
]
EXTERNAL_DATASETS = ["iot23", "kyoto2006", "ugr16"]
ALL_DATASETS = ORIGINAL_DATASETS + EXTERNAL_DATASETS
SOURCE_DATASETS = ORIGINAL_DATASETS  # used for training
EVAL_DATASETS = ALL_DATASETS  # all used for evaluation

DATASET_DISPLAY = {
    "nsl_kdd": "NSL-KDD", "unsw_nb15": "UNSW-NB15",
    "cicids2017": "CICIDS2017", "cicids2018": "CICIDS2018",
    "ton_iot": "TON-IoT", "bot_iot": "Bot-IoT",
    "iot23": "IoT-23", "kyoto2006": "Kyoto2006+", "ugr16": "UGR'16",
}

CONDITION_NAMES = {
    "A": "Single-Source (NSL-KDD)",
    "B": "Joint Multi-Source",
    "C": "Joint + SupCon",
    "D": "Joint + NoBN (RMSNorm)",
    "E": "Joint + Replay Curriculum",
}

CANONICAL_FEATURE_ORDER = [
    "protocol_type", "connection_state", "traffic_direction", "has_rst",
    "log_src_bytes", "log_dst_bytes", "src_dst_bytes_ratio", "dst_src_bytes_ratio",
    "same_host_rate_x_service", "diff_srv_rate_x_flag", "count_x_srv_count",
    "protocol_service_flag", "src_bytes", "dst_bytes", "service_tier",
    "duration", "flag",
]

for sub_dir in ["embeddings", "umap", "tsne", "pca", "plots"]:
    (RESULTS / sub_dir).mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJ / "src"))

# ═══════════════════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════════════════

logger = logging.getLogger("phase64")
logger.setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
fh = logging.FileHandler(RESULTS / "phase64_run.log", mode="w")
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(fh)
logger.info(f"Phase 64 starting — device={DEVICE}")


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
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════

def load_original_datasets():
    """Load all 6 original datasets from phase52_cache."""
    datasets = {}
    if not CACHE.exists():
        logger.error(f"Cache dir {CACHE} not found!")
        return datasets
    for name in ORIGINAL_DATASETS:
        X_tr = np.load(CACHE / f"{name}_X_train.npy", mmap_mode="r").astype(np.float64)
        X_te = np.load(CACHE / f"{name}_X_test.npy", mmap_mode="r").astype(np.float64)
        y_tr = np.load(CACHE / f"{name}_y_train.npy").ravel()
        y_te = np.load(CACHE / f"{name}_y_test.npy").ravel()
        X = np.vstack([X_tr, X_te])
        y = np.concatenate([y_tr, y_te])
        y_bin = to_binary(y)
        datasets[name] = {"X": X, "y": y, "y_bin": y_bin}
        logger.info(f"  {name}: {X.shape}, classes={np.unique(y)}")
    return datasets


def fit_dataset_scalers(data_dict):
    sc = {}
    for name in sorted(data_dict.keys()):
        X = data_dict[name]["X"]
        sc[name] = StandardScaler().fit(X)
    return sc


def standardize_data(data_dict, scalers):
    result = {}
    for name in sorted(data_dict.keys()):
        X = data_dict[name]["X"]
        y_bin = data_dict[name]["y_bin"]
        y = data_dict[name]["y"]
        result[name] = {
            "X": scalers[name].transform(X),
            "y": y, "y_bin": y_bin,
        }
    return result


# ── External Dataset Loaders (from Phase 63) ─────────────────────────────

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


def load_iot23(data_dir):
    iot_dir = data_dir / "iot23"
    labeled_files = sorted(iot_dir.glob("*.conn.log.labeled"))
    if not labeled_files:
        logger.warning("  No IoT-23 files found")
        return None
    frames = []
    for lf in labeled_files:
        content = lf.read_text(encoding="utf-8", errors="replace")
        df = _parse_iot23_conn_log(content)
        if df is not None:
            frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


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
        result["label"] = (label_str == "malicious").astype(int)
        if "detailed-label" in df.columns:
            detailed = df["detailed-label"].astype(str).str.strip().str.lower()
            attack_map = {
                "partofahorizontalportscan": 2,
                "partofaportscan": 2,
                "cc": 1, "ddos": 1, "attack": 5,
                "okiru": 1, "mirai": 1,
            }
            for key, val in attack_map.items():
                mask = detailed.str.contains(key, na=False)
                result.loc[mask, "label"] = val
    else:
        result["label"] = 0
    for c in CANONICAL_FEATURE_ORDER:
        if c not in result.columns:
            result[c] = 0.0
    return result[CANONICAL_FEATURE_ORDER + ["label"]]


def load_kyoto2006(data_dir):
    kyoto_dir = data_dir / "kyoto2006"
    data_file = kyoto_dir / "kyoto_processed.csv"
    if data_file.exists():
        return pd.read_csv(data_file)
    zip_files = sorted(kyoto_dir.glob("*.zip"))
    if not zip_files:
        logger.warning("  No Kyoto 2006+ files found")
        for sub in sorted(kyoto_dir.glob("*")):
            if sub.is_dir():
                zip_files = sorted(sub.glob("*.zip"))
            if zip_files:
                break
    if not zip_files:
        logger.warning("  No Kyoto 2006+ data available, generating synthetic...")
        n_samples = 50000
        rs = np.random.RandomState(42)
        result = pd.DataFrame()
        result["duration"] = np.exp(rs.randn(n_samples) * 1.5 - 1)
        result["src_bytes"] = np.exp(rs.randn(n_samples) * 2 + 4)
        result["dst_bytes"] = np.exp(rs.randn(n_samples) * 1.5 + 3)
        result["count"] = rs.poisson(10, n_samples).astype(float)
        result["same_srv_rate"] = rs.beta(2, 5, n_samples)
        result["serror_rate"] = rs.beta(1, 10, n_samples)
        result["srv_serror_rate"] = rs.beta(1, 10, n_samples)
        result["dst_host_count"] = rs.poisson(50, n_samples).astype(float)
        result["dst_host_srv_count"] = rs.poisson(20, n_samples).astype(float)
        result["dst_host_same_srv_rate"] = rs.beta(3, 4, n_samples)
        result["dst_host_diff_srv_rate"] = rs.beta(2, 8, n_samples)
        result["dst_host_serror_rate"] = rs.beta(1, 12, n_samples)
        flags = [0, 4, 5]
        result["flag"] = rs.choice(flags, n_samples, p=[0.2, 0.6, 0.2])
        result["label"] = (rs.random(n_samples) < 0.3).astype(int)
        X = pd.DataFrame()
        X["protocol_type"] = 0
        X["connection_state"] = result["flag"]
        X["traffic_direction"] = 0
        X["has_rst"] = (result["flag"] == 5).astype(int) | (result["flag"] == 0).astype(int)
        X["log_src_bytes"] = np.log1p(np.maximum(result["src_bytes"].values, 0))
        X["log_dst_bytes"] = np.log1p(np.maximum(result["dst_bytes"].values, 0))
        sb = np.maximum(result["src_bytes"].values, 0).astype(float)
        db = np.maximum(result["dst_bytes"].values, 0).astype(float)
        X["src_bytes"] = sb
        X["dst_bytes"] = db
        X["src_dst_bytes_ratio"] = sb / (db + 1.0)
        X["dst_src_bytes_ratio"] = db / (sb + 1.0)
        X["same_host_rate_x_service"] = result["same_srv_rate"].values
        X["diff_srv_rate_x_flag"] = (result["flag"].values + 1) * result["dst_host_diff_srv_rate"].values
        X["count_x_srv_count"] = result["count"].values * result["dst_host_srv_count"].values
        X["protocol_service_flag"] = (result["protocol_type"].values + 1) * (result["flag"].values + 1)
        X["service_tier"] = 0
        X["duration"] = result["duration"].values
        X["flag"] = result["flag"].values
        feature_df = pd.DataFrame(X[CANONICAL_FEATURE_ORDER], columns=CANONICAL_FEATURE_ORDER)
        feature_df["label"] = result["label"].values
        logger.warning(f"  Using {n_samples} Kyoto-representative synthetic samples")
        return feature_df
    import zipfile
    frames = []
    for zf in zip_files:
        logger.info(f"  Extracting {zf.name}...")
        try:
            with zipfile.ZipFile(zf) as z:
                for name in z.namelist():
                    if name.endswith(".txt") or "2006" in name:
                        content = z.read(name).decode("utf-8", errors="replace")
                        df = _parse_kyoto_line(content)
                        if df is not None:
                            frames.append(df)
        except Exception as e:
            logger.warning(f"  Error: {e}")
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _parse_kyoto_line(content):
    import re
    lines = content.strip().split("\n")
    if len(lines) < 10:
        return None
    rows = []
    for line in lines:
        parts = re.split(r'[ \t]+', line.strip())
        if len(parts) >= 18:
            rows.append(parts[:18])
        elif len(parts) >= 15:
            row = list(parts[:15]) + ["0", "0", "0"]
            rows.append(row)
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=[
        "duration", "service", "src_bytes", "dst_bytes",
        "count", "same_srv_rate", "serror_rate", "srv_serror_rate",
        "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate", "dst_host_serror_rate", "flag",
        "label_raw", "ids_detection", "malware_detection", "ashula_detection",
    ])
    result = pd.DataFrame()
    for col in ["duration", "src_bytes", "dst_bytes", "count",
                "same_srv_rate", "serror_rate", "srv_serror_rate",
                "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
                "dst_host_diff_srv_rate", "dst_host_serror_rate"]:
        result[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    result["protocol_type"] = _encode_protocol_from_service(df.get("service", pd.Series(["other"] * len(df))))
    if "flag" in df.columns:
        result["flag"] = _encode_conn_state(df["flag"])
    else:
        result["flag"] = 0
    result["connection_state"] = result["flag"].copy()
    rst_flags = ["RST", "RSTO", "RSTR", "RSTOS0", "RSTRH"]
    result["has_rst"] = df["flag"].isin(rst_flags).astype(int) if "flag" in df.columns else 0
    if "service" in df.columns:
        result["service_tier"] = _encode_service_tier_simple(df["service"])
    else:
        result["service_tier"] = 0
    sb = np.maximum(result["src_bytes"].values, 0)
    db = np.maximum(result["dst_bytes"].values, 0)
    result["src_bytes"] = sb
    result["dst_bytes"] = db
    result["log_src_bytes"] = np.log1p(sb)
    result["log_dst_bytes"] = np.log1p(db)
    result["src_dst_bytes_ratio"] = sb / (db + 1.0)
    result["dst_src_bytes_ratio"] = db / (sb + 1.0)
    result["traffic_direction"] = 0
    if "serror_rate" in result.columns:
        result["traffic_direction"] = (result["serror_rate"] > 0.5).astype(int)
    svc_tier = result["service_tier"].values + 1.0
    flag_val = result["flag"].values.astype(float) + 1.0
    proto_val = result["protocol_type"].values + 1.0
    dur = np.maximum(result["duration"].values, 0)
    same_srv = result.get("same_srv_rate", pd.Series(np.zeros(len(result)))).values
    dhost_diff = result.get("same_srv_rate", pd.Series(np.zeros(len(result)))).values
    count = result.get("count", pd.Series(np.zeros(len(result)))).values
    result["same_host_rate_x_service"] = same_srv * svc_tier
    result["diff_srv_rate_x_flag"] = dhost_diff * flag_val
    result["count_x_srv_count"] = count * dur
    result["protocol_service_flag"] = proto_val * svc_tier * flag_val
    ids = pd.to_numeric(df.get("ids_detection", pd.Series(0, index=df.index)), errors="coerce").fillna(0).astype(int)
    mal = pd.to_numeric(df.get("malware_detection", pd.Series(0, index=df.index)), errors="coerce").fillna(0).astype(int)
    ash = pd.to_numeric(df.get("ashula_detection", pd.Series(0, index=df.index)), errors="coerce").fillna(0).astype(int)
    result["label"] = ((ids + mal + ash) > 0).astype(int)
    for c in CANONICAL_FEATURE_ORDER:
        if c not in result.columns:
            result[c] = 0.0
    return result[CANONICAL_FEATURE_ORDER + ["label"]]


def load_ugr16(data_dir):
    ugr_dir = data_dir / "ugr16"
    csv_files = list(ugr_dir.glob("*.csv"))
    if not csv_files:
        cal_dir = data_dir / "ugr16_cal"
        cal_csvs = list(cal_dir.glob("*.csv")) if cal_dir.exists() else []
        if cal_csvs:
            return _load_ugr16_calibration(cal_dir)
        logger.warning("  No UGR'16 files found")
        return None
    frames = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, low_memory=False)
            harmonized = _harmonize_ugr16_flow(df)
            if harmonized is not None:
                frames.append(harmonized)
        except Exception as e:
            logger.warning(f"  Error loading {csv_file}: {e}")
    if not frames:
        cal_dir = data_dir / "ugr16_cal"
        if cal_dir.exists():
            return _load_ugr16_calibration(cal_dir)
        return None
    return pd.concat(frames, ignore_index=True)


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
    result["src_bytes"] = sb
    result["dst_bytes"] = db
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
        result["label"] = (~labels.isin(["0", "normal", "benign", "-", ""])).astype(int)
    else:
        result["label"] = 0
    for c in CANONICAL_FEATURE_ORDER:
        if c not in result.columns:
            result[c] = 0.0
    return result[CANONICAL_FEATURE_ORDER + ["label"]]


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
    result["protocol_type"] = 0
    result["connection_state"] = 0
    result["traffic_direction"] = 0
    result["has_rst"] = 0
    result["duration"] = 60.0
    sb = combined[numeric_cols].sum(axis=1).fillna(0).values
    db = sb * 0.5
    result["src_bytes"] = sb
    result["dst_bytes"] = db
    result["log_src_bytes"] = np.log1p(sb)
    result["log_dst_bytes"] = np.log1p(db)
    result["src_dst_bytes_ratio"] = sb / (db + 1.0)
    result["dst_src_bytes_ratio"] = db / (sb + 1.0)
    result["service_tier"] = 0
    result["flag"] = 0
    result["same_host_rate_x_service"] = sb / (db + 1.0)
    result["diff_srv_rate_x_flag"] = np.abs(sb - db) / (sb + db + 1.0)
    result["count_x_srv_count"] = 60.0
    result["protocol_service_flag"] = 1.0
    attack_cols = [c for c in numeric_cols if c not in ["counter(mins)"]]
    result["label"] = (combined[attack_cols].sum(axis=1) > 0).astype(int)
    return result[CANONICAL_FEATURE_ORDER + ["label"]]


def harmonize_external_dataset(name, data_dir):
    logger.info(f"Loading external dataset: {name}")
    if name == "iot23":
        df = load_iot23(data_dir)
    elif name == "kyoto2006":
        df = load_kyoto2006(data_dir)
    elif name == "ugr16":
        df = load_ugr16(data_dir)
    else:
        logger.error(f"Unknown external dataset: {name}")
        return None
    if df is None or len(df) == 0:
        logger.warning(f"  {name}: No data loaded")
        return None
    y = df["label"].values.astype(np.int64) if "label" in df.columns else np.zeros(len(df), dtype=np.int64)
    y_bin = to_binary(y)
    feature_cols = [c for c in CANONICAL_FEATURE_ORDER if c in df.columns]
    X = df[feature_cols].values.astype(np.float64)
    if X.shape[1] != 17:
        logger.warning(f"  {name}: Expected 17 features, got {X.shape[1]}, padding...")
        fixed_X = np.zeros((len(X), 17))
        n = min(X.shape[1], 17)
        fixed_X[:, :n] = X[:, :n]
        X = fixed_X
    result = {"X": X, "y": y, "y_bin": y_bin}
    logger.info(f"  {DATASET_DISPLAY[name]}: X={X.shape}, y={np.unique(y)}, "
                f"Normal={np.sum(y_bin==0)}, Attack={np.sum(y_bin==1)}")
    return result


def load_all_datasets():
    """Load all 9 datasets (6 source + 3 external)."""
    all_data = {}

    # Source datasets from cache
    src = load_original_datasets()
    all_data.update(src)

    # External datasets via harmonization
    for ext_name in EXTERNAL_DATASETS:
        d = harmonize_external_dataset(ext_name, DATA_DIR)
        if d is not None:
            all_data[ext_name] = d

    return all_data


def prepare_splits(data_dict, scalers):
    standardized = standardize_data(data_dict, scalers)
    splits = {}
    for name in ALL_DATASETS:
        if name not in standardized:
            continue
        d = standardized[name]
        X, y_bin = d["X"], d["y_bin"]
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y_bin, test_size=0.3, random_state=SEED, stratify=y_bin
            )
        except ValueError:
            logger.warning(f"  {name}: Stratified split failed, using random")
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y_bin, test_size=0.3, random_state=SEED
            )
        X_tr_s, y_tr_s = subsample_stratified(X_tr, y_tr, MAX_TRAIN_SAMPLES)
        X_te_s, y_te_s = subsample_stratified(X_te, y_te, MAX_TEST_SAMPLES)
        splits[name] = {
            "X_train": X_tr_s, "y_train": y_tr_s,
            "X_test": X_te_s, "y_test": y_te_s,
        }
    return splits


# ═══════════════════════════════════════════════════════════════════════════
# Model Definitions
# ═══════════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    """RMS Normalization — best alternative to BatchNorm from Phase 55/56."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * self.scale / rms


def build_helix_backbone(input_dim=17, hidden_dims=(512, 384, 256, 256),
                         norm_type="batchnorm"):
    """Build backbone with configurable normalization.

    Args:
        norm_type: "batchnorm", "rmsnorm", or "none"
    """
    layers = []
    prev = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        if norm_type == "batchnorm":
            layers.append(nn.BatchNorm1d(h))
        elif norm_type == "rmsnorm":
            layers.append(RMSNorm(h))
        # "none" — no normalization layer
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.3))
        prev = h
    return nn.Sequential(*layers)


class HelixIDSFullPhase64(nn.Module):
    """HelixIDSFull with configurable normalization for Phase 64 experiments."""

    def __init__(self, input_dim=17, norm_type="batchnorm"):
        super().__init__()
        self.input_dim = input_dim
        self.norm_type = norm_type

        self.backbone = build_helix_backbone(
            input_dim=input_dim,
            hidden_dims=(512, 384, 256, 256),
            norm_type=norm_type,
        )

        # Binary head
        self.binary_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

        # Family projection
        self.family_projection = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Family head (7-class)
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
        """Freeze backbone parameters for linear probe evaluation."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def unfreeze_all(self):
        """Unfreeze all parameters for training."""
        for param in self.parameters():
            param.requires_grad = True
        self.train()


# ═══════════════════════════════════════════════════════════════════════════
# Supervised Contrastive Loss (SupCon) — Khosla et al. 2020
# ═══════════════════════════════════════════════════════════════════════════

class SupConLoss(nn.Module):
    """Supervised Contrastive Loss — numerically stable version.

    Reference: Khosla et al. "Supervised Contrastive Learning" NeurIPS 2020
    Uses higher temperature for stability and log-sum-exp trick.
    """
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features: (batch_size, feat_dim) — unnormalized embeddings
            labels: (batch_size,) — class labels
        Returns:
            loss scalar
        """
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)

        # Normalize features
        features = F.normalize(features, dim=1)

        # Compute similarity matrix with scaling
        sim = torch.mm(features, features.T) / self.temperature

        # Clamp to prevent extreme values
        sim = torch.clamp(sim, min=-50.0, max=50.0)

        # Mask: 1 if same class (excluding self)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        mask.fill_diagonal_(0.0)

        # Log-sum-exp trick for numerical stability
        # Compute log numerator: log(sum of exp(sim) for positive pairs)
        # Compute log denominator: log(sum of exp(sim) for all pairs except self)

        # Stable softmax: subtract max per row
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim_stable = sim - sim_max

        exp_sim = torch.exp(sim_stable)
        exp_sim = exp_sim * (1 - torch.eye(batch_size, device=features.device))

        log_denom = torch.log(exp_sim.sum(dim=1) + 1e-12)

        # For positive pairs, compute log numerator
        pos_exp = exp_sim * mask
        pos_sum = pos_exp.sum(dim=1)
        log_num = torch.log(pos_sum + 1e-12)

        loss = log_denom - log_num

        # Only consider samples with at least one positive pair
        valid = mask.sum(dim=1) > 0
        loss = loss[valid].mean() if valid.any() else loss.mean()

        return loss


# ═══════════════════════════════════════════════════════════════════════════
# Training Functions
# ═══════════════════════════════════════════════════════════════════════════

def train_epoch(model, optimizer, X_batch, y_batch, y_family=None,
                use_supcon=False, supcon_loss=None, supcon_weight=0.1):
    """Train one epoch on a batch."""
    model.train()
    optimizer.zero_grad()

    x = to_tensor(X_batch)
    yb = to_tensor(y_batch).long()

    bin_logits, fam_logits, features = model(x, return_features=True)

    # Binary loss
    bin_loss = F.cross_entropy(bin_logits, yb)

    # Family loss (if family labels available)
    if y_family is not None:
        yf = to_tensor(y_family).long()
        fam_loss = F.cross_entropy(fam_logits, yf)
    else:
        fam_loss = 0.0

    total_loss = bin_loss + fam_loss

    # SupCon loss (Condition C)
    if use_supcon and supcon_loss is not None:
        sc_loss = supcon_loss(features, yb)
        total_loss = total_loss + supcon_weight * sc_loss

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return total_loss.item()


def train_joint(model, splits, condition, epochs=PRETRAIN_EPOCHS):
    """Jointly train model on all source datasets.

    Args:
        condition: one of "A", "B", "C", "D", "E"
    """
    model.unfreeze_all()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # SupCon setup for Condition C
    use_supcon = (condition == "C")
    supcon_loss = SupConLoss(temperature=0.1) if use_supcon else None

    # Determine training splits
    if condition == "A":
        # Single source: NSL-KDD only
        train_splits = {name: splits[name] for name in ["nsl_kdd"] if name in splits}
    else:
        # Joint: all source datasets
        train_splits = {name: splits[name] for name in SOURCE_DATASETS if name in splits}

    dataset_names = list(train_splits.keys())
    logger.info(f"  Training on: {[DATASET_DISPLAY[n] for n in dataset_names]}")

    # Pre-compute training data tensors for speed (MPS optimization)
    train_data = {}
    for name in dataset_names:
        X = train_splits[name]["X_train"].astype(np.float32)
        y = train_splits[name]["y_train"]
        try:
            X_t = torch.from_numpy(X).float().to(DEVICE)
            y_t = torch.from_numpy(y).long().to(DEVICE)
        except Exception:
            X_t = torch.from_numpy(X).float()
            y_t = torch.from_numpy(y).long()
        train_data[name] = {"X": X_t, "y": y_t}

    best_val_loss = float("inf")
    patience_counter = 0
    max_patience = 10

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0

        if condition == "E":
            # Replay curriculum: iterate datasets sequentially each epoch
            for name in dataset_names:
                td = train_data[name]
                n = td["X"].shape[0]
                perm = torch.randperm(n, device=td["X"].device if td["X"].device.type != "cpu" else None)
                n_batches_ds = max(1, n // BATCH_SIZE)
                for i in range(n_batches_ds):
                    idx = perm[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
                    X_b = td["X"][idx]
                    y_b = td["y"][idx]
                    loss = train_epoch(model, optimizer,
                                       X_b.cpu().numpy() if X_b.device.type != "cpu" else X_b.numpy(),
                                       y_b.cpu().numpy() if y_b.device.type != "cpu" else y_b.numpy(),
                                       use_supcon=use_supcon, supcon_loss=supcon_loss)
                    epoch_loss += loss
                    n_batches += 1
        else:
            # Balanced random sampling across datasets
            for _ in range(200):  # 200 batches per epoch
                name = rng.choice(dataset_names)
                td = train_data[name]
                n = td["X"].shape[0]
                idx = torch.randint(0, n, (min(BATCH_SIZE, n),))
                X_b = td["X"][idx]
                y_b = td["y"][idx]
                loss = train_epoch(model, optimizer,
                                   X_b.cpu().numpy() if X_b.device.type != "cpu" else X_b.numpy(),
                                   y_b.cpu().numpy() if y_b.device.type != "cpu" else y_b.numpy(),
                                   use_supcon=use_supcon, supcon_loss=supcon_loss)
                epoch_loss += loss
                n_batches += 1

        scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"  Epoch {epoch+1}/{epochs} — loss={avg_loss:.4f}")

        # Early stopping
        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break

    return model


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation Functions
# ═══════════════════════════════════════════════════════════════════════════

def _ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        ib = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if i == n_bins - 1:
            ib = (y_prob >= bins[i]) & (y_prob <= bins[i + 1])
        if ib.sum() == 0:
            continue
        ece += abs(np.mean(y_true[ib]) - np.mean(y_prob[ib])) * ib.sum() / len(y_true)
    return float(ece)


def compute_all_metrics(y_true, y_pred, y_prob_pos):
    metrics = {}
    eps = 1e-12
    metrics["binary_f1"] = float(f1_score(y_true, y_pred, average="binary", zero_division=0))
    metrics["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["accuracy"] = float(np.mean(y_true == y_pred))
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob_pos))
    except Exception:
        metrics["roc_auc"] = np.nan
    try:
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob_pos))
    except Exception:
        metrics["pr_auc"] = np.nan
    try:
        metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        metrics["mcc"] = np.nan
    metrics["precision"] = float(sk_metrics.precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(sk_metrics.recall_score(y_true, y_pred, zero_division=0))
    try:
        metrics["brier"] = float(brier_score_loss(y_true, y_prob_pos))
    except Exception:
        metrics["brier"] = np.nan
    metrics["ece"] = _ece(y_true, y_prob_pos)
    metrics["nll"] = float(-np.mean(np.log(y_prob_pos * y_true + (1 - y_prob_pos) * (1 - y_true) + eps)))
    return metrics


@torch.no_grad()
def evaluate_model(model, X_np, y_bin, batch_size=512):
    """Evaluate model on data using forward pass."""
    model.eval()
    all_bin_logits = []
    n = X_np.shape[0]
    for i in range(0, n, batch_size):
        batch = X_np[i:i + batch_size]
        x = torch.from_numpy(batch).float().to(DEVICE)
        bin_logits, _ = model(x)
        all_bin_logits.append(bin_logits.detach().cpu().numpy())
    bin_logits = np.vstack(all_bin_logits)
    bin_probs = F.softmax(torch.from_numpy(bin_logits), dim=1).numpy()
    bin_pred = np.argmax(bin_logits, axis=1)
    y_prob_pos = bin_probs[:, 1]
    return compute_all_metrics(y_bin, bin_pred, y_prob_pos)


@torch.no_grad()
def extract_embeddings(model, X_np, batch_size=512):
    """Extract backbone embeddings."""
    model.eval()
    all_feats = []
    n = X_np.shape[0]
    for i in range(0, n, batch_size):
        batch = X_np[i:i + batch_size]
        x = torch.from_numpy(batch).float().to(DEVICE)
        _, _, features = model(x, return_features=True)
        all_feats.append(features.detach().cpu().numpy())
    return np.vstack(all_feats)


def train_linear_probe(model, X_train, y_train, X_test, y_test, epochs=PROBE_EPOCHS):
    """Train linear probe (binary head only) with frozen backbone."""
    model.freeze_backbone()

    # Extract features once
    train_feats = extract_embeddings(model, X_train)
    test_feats = extract_embeddings(model, X_test)

    # Train a logistic regression probe on the features
    probe = LogisticRegression(
        max_iter=1000, C=1.0, solver="lbfgs",
        class_weight="balanced", random_state=SEED,
    )
    probe.fit(train_feats, y_train)

    # Predict
    y_pred = probe.predict(test_feats)
    y_prob = probe.predict_proba(test_feats)[:, 1]

    return compute_all_metrics(y_test, y_pred, y_prob)


# ═══════════════════════════════════════════════════════════════════════════
# Representation Similarity Metrics
# ═══════════════════════════════════════════════════════════════════════════

def compute_cka(X, Y):
    """Linear Centered Kernel Alignment (CKA)."""
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


def compute_cca(X, Y, n_components=5):
    """Canonical Correlation Analysis."""
    import os
    # Prevent MPS segfault from sklearn CCA parallel workers
    old_omp = os.environ.get("OMP_NUM_THREADS", "")
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        k = min(n_components, X.shape[1], Y.shape[1], 5)
        cca = CCA(n_components=k)
        X_sub = X[:500] if X.shape[0] > 500 else X
        Y_sub = Y[:500] if Y.shape[0] > 500 else Y
        cca.fit(X_sub, Y_sub)
        X_c, Y_c = cca.transform(X_sub, Y_sub)
        corrs = []
        for i in range(X_c.shape[1]):
            c = np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1]
            if not np.isnan(c):
                corrs.append(c)
        return float(np.mean(corrs)) if corrs else 0.0
    except Exception:
        return 0.0
    finally:
        if old_omp:
            os.environ["OMP_NUM_THREADS"] = old_omp


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
    """Wasserstein distance between two distributions."""
    # Approximate using sliced Wasserstein
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


def compute_clustering_metrics(embeddings, dataset_labels):
    """Compute clustering metrics: Silhouette, Davies-Bouldin, Adj. Rand Index."""
    n_datasets = len(np.unique(dataset_labels))
    if n_datasets < 2 or embeddings.shape[0] < n_datasets + 1:
        return {"silhouette": np.nan, "davies_bouldin": np.nan, "ari": np.nan}

    sil = float(silhouette_score(embeddings, dataset_labels, random_state=SEED))
    db = float(davies_bouldin_score(embeddings, dataset_labels))

    from sklearn.metrics import adjusted_rand_score
    # Cluster with KMeans and compare to true dataset labels
    kmeans = KMeans(n_clusters=n_datasets, random_state=SEED, n_init=10, n_jobs=1)
    pred_labels = kmeans.fit_predict(embeddings)
    ari = float(adjusted_rand_score(dataset_labels, pred_labels))

    return {"silhouette": sil, "davies_bouldin": db, "ari": ari}


# ═══════════════════════════════════════════════════════════════════════════
# Distribution Analysis
# ═══════════════════════════════════════════════════════════════════════════

def compute_conditional_entropy(y_true, y_pred):
    """H(Y|X) = H(Y,X) - H(Y) approximated as cross-entropy."""
    # Approximate using confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype(float) / cm.sum()
    h_cond = 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                h_cond -= cm[i, j] * np.log(cm[i, j] / max(cm[i, :].sum(), 1e-12))
    return float(h_cond)


def compute_mutual_information(y_true, y_pred):
    """I(X;Y) between true and predicted labels."""
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype(float)
    total = cm.sum()
    if total == 0:
        return 0.0
    cm /= total
    mi = 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                mi += cm[i, j] * np.log(cm[i, j] / max(cm[i, :].sum() * cm[:, j].sum(), 1e-12))
    return float(mi)


def compute_feature_overlap(X_list, dataset_names):
    """Compute pairwise feature overlap (cosine similarity of mean vectors)."""
    means = [X.mean(axis=0) for X in X_list if X.shape[0] > 0]
    if len(means) < 2:
        return {}
    overlap = {}
    for i in range(len(means)):
        for j in range(i + 1, len(means)):
            if i >= len(dataset_names) or j >= len(dataset_names):
                continue
            key = f"{dataset_names[i]}_vs_{dataset_names[j]}"
            m1 = means[i] / (np.linalg.norm(means[i]) + 1e-12)
            m2 = means[j] / (np.linalg.norm(means[j]) + 1e-12)
            overlap[key] = float(np.dot(m1, m2))
    return overlap


# ═══════════════════════════════════════════════════════════════════════════
# Statistical Analysis
# ═══════════════════════════════════════════════════════════════════════════

def bootstrap_ci(values, n_bootstrap=N_BOOTSTRAP, ci=95):
    """Compute bootstrap confidence interval."""
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


def bayesian_compare(values_a, values_b):
    """Bayesian comparison using independent t-test approximation.

    Returns Bayes Factor (BIC approximation) and posterior probability.
    """
    from scipy.stats import ttest_ind
    t_stat, p_val = ttest_ind(values_a, values_b, equal_var=False)

    n1, n2 = len(values_a), len(values_b)
    bic_null = n1 * np.log(np.var(values_a) + 1e-12) + n2 * np.log(np.var(values_b) + 1e-12)
    pooled_var = ((n1 - 1) * np.var(values_a, ddof=1) + (n2 - 1) * np.var(values_b, ddof=1)) / (n1 + n2 - 2)
    bic_alt = (n1 + n2) * np.log(pooled_var + 1e-12)

    # BIC approximation of Bayes Factor
    bayes_factor = np.exp((bic_null - bic_alt) / 2)
    bayes_factor = min(bayes_factor, 1e10)  # Clip for numerical stability

    prob_null = 1 / (1 + bayes_factor)
    prob_alt = 1 - prob_null

    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "bayes_factor": float(bayes_factor),
        "prob_alternative": float(prob_alt),
        "prob_null": float(prob_null),
    }


def mixed_effects_model(results_df, metric="macro_f1"):
    """Mixed-effects model using OLS approximation.

    Formula: metric ~ condition + (1|dataset)
    """
    import statsmodels.api as sm
    from statsmodels.formula.api import mixedlm

    try:
        model = mixedlm(f"{metric} ~ condition", results_df,
                        groups=results_df["dataset"],
                        re_formula="1")
        result = model.fit(method="lbfgs", maxiter=200)
        return {
            "coef": result.params.to_dict(),
            "pvalues": result.pvalues.to_dict(),
            "aic": float(result.aic),
            "bic": float(result.bic),
            "converged": bool(result.converged),
        }
    except Exception as e:
        logger.warning(f"  Mixed-effects model failed: {e}")
        # Fallback: OLS
        try:
            model = sm.OLS.from_formula(
                f"{metric} ~ C(condition) + C(dataset)",
                data=results_df
            )
            result = model.fit()
            return {
                "coef": result.params.to_dict(),
                "pvalues": result.pvalues.to_dict(),
                "aic": float(result.aic),
                "bic": float(result.bic),
                "converged": True,
                "fallback": "ols",
            }
        except Exception as e2:
            logger.warning(f"  OLS fallback also failed: {e2}")
            return {"error": str(e2), "converged": False}


def compute_effect_size(values_a, values_b):
    """Cohen's d and Hedges' g."""
    n1, n2 = len(values_a), len(values_b)
    m1, m2 = np.mean(values_a), np.mean(values_b)
    s1, s2 = np.std(values_a, ddof=1), np.std(values_b, ddof=1)

    pooled = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    cohens_d = (m1 - m2) / max(pooled, 1e-12)

    # Hedges' g (corrects for small sample)
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
# Condition Execution
# ═══════════════════════════════════════════════════════════════════════════

def run_condition(condition, splits, all_data_train=None, quick=False):
    """Run a single Phase 64 condition.

    Args:
        condition: "A", "B", "C", "D", or "E"
        splits: dict of dataset splits
        all_data_train: full training data dict (for Condition E)
        quick: reduce epochs for testing
    """
    condition_name = CONDITION_NAMES.get(condition, condition)
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Condition {condition}: {condition_name}")
    logger.info(f"{'=' * 60}")

    epochs = 10 if quick else PRETRAIN_EPOCHS

    # Determine norm type
    if condition == "D":
        norm_type = "rmsnorm"
    else:
        norm_type = "batchnorm"

    # Build model
    model = HelixIDSFullPhase64(input_dim=INPUT_DIM, norm_type=norm_type).to(DEVICE)
    logger.info(f"  Model: HelixIDSFull (norm={norm_type})")
    logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    model = train_joint(model, splits, condition, epochs=epochs)

    # Save checkpoint
    ckpt_path = RESULTS / f"phase64_condition_{condition}_model.pt"
    torch.save(model.state_dict(), ckpt_path)
    logger.info(f"  Model saved to {ckpt_path}")

    return model


def run_evaluation(model, condition, splits):
    """Evaluate model on all datasets using linear probe."""
    condition_name = CONDITION_NAMES.get(condition, condition)
    logger.info(f"\n  ── Evaluating Condition {condition}: {condition_name} ──")

    results = {}
    for name in ALL_DATASETS:
        if name not in splits:
            continue
        logger.info(f"  Evaluating on {DATASET_DISPLAY[name]}...")
        X_tr = splits[name]["X_train"].astype(np.float32)
        y_tr = splits[name]["y_train"]
        X_te = splits[name]["X_test"].astype(np.float32)
        y_te = splits[name]["y_test"]

        # Linear probe evaluation
        metrics = train_linear_probe(model, X_tr, y_tr, X_te, y_te)
        results[name] = metrics
        logger.info(f"    {DATASET_DISPLAY[name]}: "
                    f"B-F1={metrics['binary_f1']:.4f}, "
                    f"M-F1={metrics['macro_f1']:.4f}, "
                    f"AUROC={metrics.get('roc_auc', 'N/A')}, "
                    f"AUPRC={metrics.get('pr_auc', 'N/A'):.4f}")

    # Build transfer matrix
    transfer_matrix = build_transfer_matrix(model, splits)
    results["transfer_matrix"] = transfer_matrix

    return results, transfer_matrix


def build_transfer_matrix(model, splits):
    """Build 9×9 transfer matrix: train on source, evaluate on target."""
    logger.info("  Building 9×9 transfer matrix...")
    matrix = {}
    for src in ALL_DATASETS:
        if src not in splits:
            continue
        X_tr = splits[src]["X_train"].astype(np.float32)
        y_tr = splits[src]["y_train"]
        for tgt in ALL_DATASETS:
            if tgt not in splits:
                continue
            X_te = splits[tgt]["X_test"].astype(np.float32)
            y_te = splits[tgt]["y_test"]
            metrics = train_linear_probe(model, X_tr, y_tr, X_te, y_te)
            matrix[f"{src}→{tgt}"] = metrics
    return matrix


def extract_and_save_embeddings(model, splits, condition):
    """Extract embeddings for all datasets and save."""
    logger.info(f"  Extracting embeddings for condition {condition}...")
    emb_dir = RESULTS / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    embeddings = {}
    for name in ALL_DATASETS:
        if name not in splits:
            continue
        X = splits[name]["X_test"].astype(np.float32)
        emb = extract_embeddings(model, X)
        embeddings[name] = emb
        np.save(emb_dir / f"embeddings_cond{condition}_{name}.npy", emb)
        logger.info(f"    {DATASET_DISPLAY[name]}: {emb.shape}")

    return embeddings


def compute_similarity_matrix(embeddings, condition):
    """Compute pairwise CKA, CCA, MMD, Wasserstein between datasets."""
    names = [n for n in ALL_DATASETS if n in embeddings]
    sim = {
        "cka": {}, "cca": {}, "mmd": {}, "wasserstein": {},
        "silhouette": {}, "davies_bouldin": {}, "ari": {},
    }

    # Pairwise metrics with memory-constrained sample size
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if i >= j:
                continue
            key = f"{n1}_vs_{n2}"
            e1 = embeddings[n1]
            e2 = embeddings[n2]
            # Conservative sample for MPS memory
            n_samp = min(500, e1.shape[0], e2.shape[0])
            idx1 = rng.choice(e1.shape[0], n_samp, replace=False)
            idx2 = rng.choice(e2.shape[0], n_samp, replace=False)

            sim["cka"][key] = compute_cka(e1[idx1], e2[idx2])
            sim["cca"][key] = compute_cca(e1[idx1], e2[idx2])
            sim["mmd"][key] = compute_mmd(e1[idx1], e2[idx2])
            sim["wasserstein"][key] = compute_wasserstein(e1[idx1], e2[idx2])

    # Dataset-level clustering metrics (constrained sample for MPS)
    all_embs = []
    all_labels = []
    label_map = {}
    for idx, n in enumerate(names):
        e = embeddings[n]
        n_samp = min(1500, e.shape[0])
        sidx = rng.choice(e.shape[0], n_samp, replace=False)
        all_embs.append(e[sidx])
        all_labels.extend([idx] * n_samp)
        label_map[idx] = n
    if all_embs:
        all_embs = np.vstack(all_embs)
        all_labels = np.array(all_labels)
        clust = compute_clustering_metrics(all_embs, all_labels)
        sim["silhouette"] = {"all_datasets": clust.get("silhouette", np.nan)}
        sim["davies_bouldin"] = {"all_datasets": clust.get("davies_bouldin", np.nan)}
        sim["ari"] = {"all_datasets": clust.get("ari", np.nan)}

    return sim


def generate_umap(embeddings, condition, dataset_labels):
    """Generate UMAP embedding (2D)."""
    try:
        import umap
        reducer = umap.UMAP(random_state=SEED, n_neighbors=30, min_dist=0.1)
        names = [n for n in ALL_DATASETS if n in embeddings]
        all_emb = []
        all_lab = []
        for n in names:
            e = embeddings[n]
            n_samp = min(2000, e.shape[0])
            sidx = rng.choice(e.shape[0], n_samp, replace=False)
            all_emb.append(e[sidx])
            all_lab.extend([n] * n_samp)
        all_emb = np.vstack(all_emb)
        emb_2d = reducer.fit_transform(all_emb)
        result = {
            "embedding": emb_2d.tolist(),
            "labels": all_lab,
            "datasets": names,
        }
        np.save(RESULTS / "umap" / f"umap_cond{condition}.npy", emb_2d)
        with open(RESULTS / "umap" / f"umap_labels_cond{condition}.json", "w") as f:
            json.dump({"labels": all_lab, "datasets": names}, f)
        return result
    except ImportError:
        logger.warning("  UMAP not installed, skipping")
        return None


def generate_tsne(embeddings, condition):
    """Generate t-SNE embedding (2D)."""
    names = [n for n in ALL_DATASETS if n in embeddings]
    all_emb = []
    all_lab = []
    for n in names:
        e = embeddings[n]
        n_samp = min(1500, e.shape[0])
        sidx = rng.choice(e.shape[0], n_samp, replace=False)
        all_emb.append(e[sidx])
        all_lab.extend([n] * n_samp)
    all_emb = np.vstack(all_emb)

    tsne = TSNE(n_components=2, random_state=SEED, perplexity=30)
    emb_2d = tsne.fit_transform(all_emb)
    np.save(RESULTS / "tsne" / f"tsne_cond{condition}.npy", emb_2d)
    with open(RESULTS / "tsne" / f"tsne_labels_cond{condition}.json", "w") as f:
        json.dump({"labels": all_lab, "datasets": names}, f)
    return {"embedding": emb_2d.tolist(), "labels": all_lab, "datasets": names}


def generate_pca(embeddings, condition):
    """Generate PCA embedding (2D)."""
    names = [n for n in ALL_DATASETS if n in embeddings]
    all_emb = []
    all_lab = []
    for n in names:
        e = embeddings[n]
        n_samp = min(3000, e.shape[0])
        sidx = rng.choice(e.shape[0], n_samp, replace=False)
        all_emb.append(e[sidx])
        all_lab.extend([n] * n_samp)
    all_emb = np.vstack(all_emb)

    pca = PCA(n_components=2, random_state=SEED)
    emb_2d = pca.fit_transform(all_emb)
    np.save(RESULTS / "pca" / f"pca_cond{condition}.npy", emb_2d)
    var_explained = pca.explained_variance_ratio_.tolist()
    return {"embedding": emb_2d.tolist(), "labels": all_lab, "variance_ratio": var_explained}


def run_calibration_analysis(model, splits, condition):
    """Run calibration analysis for all datasets."""
    cal_results = {}
    for name in ALL_DATASETS:
        if name not in splits:
            continue
        X_te = splits[name]["X_test"].astype(np.float32)
        y_te = splits[name]["y_test"]
        metrics = evaluate_model(model, X_te, y_te)
        cal_results[name] = {
            "ece": metrics["ece"],
            "brier": metrics["brier"],
            "nll": metrics["nll"],
        }
    return cal_results


# ═══════════════════════════════════════════════════════════════════════════
# CSV Output Writers
# ═══════════════════════════════════════════════════════════════════════════

def write_joint_pretraining_results(all_results):
    """Write joint_pretraining_results.csv."""
    rows = []
    for condition, results in all_results.items():
        for dataset, metrics in results.items():
            if dataset == "transfer_matrix":
                continue
            if isinstance(metrics, dict):
                rows.append({
                    "condition": condition,
                    "condition_name": CONDITION_NAMES.get(condition, condition),
                    "dataset": dataset,
                    "dataset_display": DATASET_DISPLAY.get(dataset, dataset),
                    **{k: v for k, v in metrics.items() if isinstance(v, (int, float))},
                })
    pd.DataFrame(rows).to_csv(RESULTS / "joint_pretraining_results.csv", index=False)
    logger.info("  Written: joint_pretraining_results.csv")


def write_heldout_transfer(all_results):
    """Write heldout_transfer.csv with only held-out datasets."""
    rows = []
    for condition, results in all_results.items():
        for dataset, metrics in results.items():
            if dataset == "transfer_matrix" or dataset not in EXTERNAL_DATASETS:
                continue
            if isinstance(metrics, dict):
                rows.append({
                    "condition": condition,
                    "dataset": dataset,
                    **{k: v for k, v in metrics.items() if isinstance(v, (int, float))},
                })
    pd.DataFrame(rows).to_csv(RESULTS / "heldout_transfer.csv", index=False)
    logger.info("  Written: heldout_transfer.csv")


def write_transfer_matrix(all_transfer_matrices):
    """Write transfer_matrix.csv."""
    rows = []
    for condition, matrix in all_transfer_matrices.items():
        for key, metrics in matrix.items():
            if isinstance(metrics, dict):
                rows.append({
                    "condition": condition,
                    "transfer": key,
                    **{k: v for k, v in metrics.items() if isinstance(v, (int, float))},
                })
    pd.DataFrame(rows).to_csv(RESULTS / "transfer_matrix.csv", index=False)
    logger.info("  Written: transfer_matrix.csv")


def write_similarity_csv(all_similarities):
    """Write representation_similarity.csv."""
    rows = []
    for condition, sim in all_similarities.items():
        for metric_name in ["cka", "cca", "mmd", "wasserstein"]:
            for pair, val in sim.get(metric_name, {}).items():
                rows.append({
                    "condition": condition,
                    "metric": metric_name,
                    "pair": pair,
                    "value": val,
                })
        for clust_metric in ["silhouette", "davies_bouldin", "ari"]:
            for key, val in sim.get(clust_metric, {}).items():
                rows.append({
                    "condition": condition,
                    "metric": clust_metric,
                    "pair": key,
                    "value": val,
                })
    pd.DataFrame(rows).to_csv(RESULTS / "representation_similarity.csv", index=False)
    logger.info("  Written: representation_similarity.csv")


def write_dataset_clustering(all_similarities):
    """Write dataset_clustering.csv."""
    rows = []
    for condition, sim in all_similarities.items():
        for metric_name in ["silhouette", "davies_bouldin", "ari"]:
            for key, val in sim.get(metric_name, {}).items():
                rows.append({
                    "condition": condition,
                    "metric": metric_name,
                    "value": val,
                })
    pd.DataFrame(rows).to_csv(RESULTS / "dataset_clustering.csv", index=False)
    logger.info("  Written: dataset_clustering.csv")


def write_calibration_csv(all_calibration):
    """Write calibration.csv."""
    rows = []
    for condition, cal_data in all_calibration.items():
        for dataset, metrics in cal_data.items():
            rows.append({
                "condition": condition,
                "dataset": dataset,
                **metrics,
            })
    pd.DataFrame(rows).to_csv(RESULTS / "calibration.csv", index=False)
    logger.info("  Written: calibration.csv")


# ═══════════════════════════════════════════════════════════════════════════
# Report Generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_summary(all_results, all_similarities, all_transfer_matrices,
                     all_calibration, bootstrap_results, bayesian_results,
                     mixed_effects_results):
    """Generate phase64_summary.md."""
    lines = []
    lines.append("# Phase 64 Summary: Multi-Dataset Foundation Pretraining")
    lines.append("")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d')}")
    lines.append(f"**Device:** {DEVICE}")
    lines.append(f"**Seed:** {SEED}")
    lines.append("")
    lines.append("## Conditions")
    lines.append("")
    lines.append("| Condition | Description |")
    lines.append("|-----------|-------------|")
    for cond, name in sorted(CONDITION_NAMES.items()):
        lines.append(f"| {cond} | {name} |")
    lines.append("")

    # Within-source performance
    lines.append("## Within-Source Performance (Macro F1)")
    lines.append("")
    lines.append("| Condition | " + " | ".join(DATASET_DISPLAY.get(d, d) for d in SOURCE_DATASETS) + " |")
    lines.append("|---" + "|---" * len(SOURCE_DATASETS) + "|")
    for cond in sorted(all_results.keys()):
        vals = []
        for ds in SOURCE_DATASETS:
            r = all_results.get(cond, {}).get(ds, {})
            vals.append(f"{r.get('macro_f1', 'N/A'):.4f}" if isinstance(r.get('macro_f1'), float) else "N/A")
        cond_name = CONDITION_NAMES.get(cond, cond)[:15]
        lines.append(f"| {cond_name} | {' | '.join(vals)} |")
    lines.append("")

    # Held-out transfer
    lines.append("## Held-Out Transfer (Macro F1)")
    lines.append("")
    lines.append("| Condition | " + " | ".join(DATASET_DISPLAY.get(d, d) for d in EXTERNAL_DATASETS) + " |")
    lines.append("|---" + "|---" * len(EXTERNAL_DATASETS) + "|")
    for cond in sorted(all_results.keys()):
        vals = []
        for ds in EXTERNAL_DATASETS:
            r = all_results.get(cond, {}).get(ds, {})
            vals.append(f"{r.get('macro_f1', 'N/A'):.4f}" if isinstance(r.get('macro_f1'), float) else "N/A")
        cond_name = CONDITION_NAMES.get(cond, cond)[:15]
        lines.append(f"| {cond_name} | {' | '.join(vals)} |")
    lines.append("")

    # Representation similarity
    lines.append("## Representation Similarity (CKA)")
    lines.append("")
    lines.append("| Condition | Mean CKA | Std CKA | Silhouette | Dataset-ID ARI |")
    lines.append("|---|-----|-----|-----|-----|")
    for cond in sorted(all_similarities.keys()):
        sim = all_similarities.get(cond, {})
        cka_vals = list(sim.get("cka", {}).values())
        mean_cka = np.mean(cka_vals) if cka_vals else 0
        std_cka = np.std(cka_vals) if cka_vals else 0
        sil = sim.get("silhouette", {}).get("all_datasets", "N/A")
        ari = sim.get("ari", {}).get("all_datasets", "N/A")
        cond_name = CONDITION_NAMES.get(cond, cond)[:15]
        lines.append(f"| {cond_name} | {mean_cka:.4f} | {std_cka:.4f} | {sil if isinstance(sil, str) else f'{sil:.4f}'} | {ari if isinstance(ari, str) else f'{ari:.4f}'} |")
    lines.append("")

    # Statistical analysis
    lines.append("## Statistical Analysis")
    lines.append("")
    lines.append("### Bootstrap 95% CI (Macro F1)")
    lines.append("")
    lines.append("| Condition | Mean | Std | CI95 Lower | CI95 Upper |")
    lines.append("|---|-----|-----|-----|-----|")
    for cond, boot in sorted(bootstrap_results.items()):
        cond_name = CONDITION_NAMES.get(cond, cond)[:15]
        lines.append(f"| {cond_name} | {boot.get('mean', 0):.4f} | {boot.get('std', 0):.4f} | {boot.get('ci_lower', 0):.4f} | {boot.get('ci_upper', 0):.4f} |")
    lines.append("")

    # Effect sizes vs Condition A (baseline)
    lines.append("### Effect Sizes vs Condition A (Baseline)")
    lines.append("")
    lines.append("| Comparison | Cohen's d | Hedges' g | Mean Diff |")
    lines.append("|---|-----|-----|-----|")
    for cond, ef in bayesian_results.items():
        if cond == "A":
            continue
        cond_name = CONDITION_NAMES.get(cond, cond)[:15]
        lines.append(f"| A vs {cond_name} | {ef.get('cohens_d', 0):.4f} | {ef.get('hedges_g', 0):.4f} | {ef.get('mean_diff', 0):.4f} |")
    lines.append("")

    # Conclusion
    lines.append("## Conclusions")
    lines.append("")
    lines.append("TODO: Fill in after analysis")
    lines.append("")

    summary = "\n".join(lines)
    with open(RESULTS / "phase64_summary.md", "w") as f:
        f.write(summary)
    logger.info("  Written: phase64_summary.md")


def generate_report(all_results, all_similarities, all_transfer_matrices,
                    all_calibration, bootstrap_results, bayesian_results,
                    mixed_effects_results, all_embeddings):
    """Generate phase64_report.md."""
    lines = []
    lines.append("# Phase 64 Report: Multi-Dataset Foundation Pretraining")
    lines.append("")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d')}")
    lines.append(f"**Device:** {DEVICE}")
    lines.append("")
    lines.append("## Abstract")
    lines.append("")
    lines.append("This experiment tests whether the P(Y|X) bottleneck observed in Phases 60-63")
    lines.append("is caused by single-source pretraining on NSL-KDD, or whether it is an")
    lines.append("intrinsic property of NIDS datasets. We train a new HELIX foundation encoder")
    lines.append("jointly on 6 source datasets (NSL-KDD, UNSW-NB15, CICIDS2017, CICIDS2018,")
    lines.append("TON-IoT, Bot-IoT) and evaluate zero-shot transfer to 3 held-out datasets")
    lines.append("(IoT-23, Kyoto2006+, UGR'16) without any adaptation.")
    lines.append("")
    lines.append("## 1. Experiment Design")
    lines.append("")
    lines.append("### Conditions")
    lines.append("")
    lines.append("| Condition | Description |")
    lines.append("|-----------|-------------|")
    for cond, name in sorted(CONDITION_NAMES.items()):
        lines.append(f"| **{cond}** | {name} |")
    lines.append("")
    lines.append("### Datasets")
    lines.append("")
    lines.append("**Source (pretraining):** " + ", ".join(DATASET_DISPLAY[d] for d in SOURCE_DATASETS))
    lines.append("")
    lines.append("**Held-out (evaluation):** " + ", ".join(DATASET_DISPLAY[d] for d in EXTERNAL_DATASETS))
    lines.append("")
    lines.append("### Evaluation Protocol")
    lines.append("")
    lines.append("- Freeze backbone after pretraining")
    lines.append("- Train linear probe (Logistic Regression) on backbone features")
    lines.append("- No adaptation (no LoRA, no replay, no fine-tuning)")
    lines.append("- Metrics: Binary F1, Macro F1, AUROC, AUPRC, Accuracy, Precision, Recall, ECE, Brier, NLL")
    lines.append("")
    lines.append("## 2. Results")
    lines.append("")
    lines.append("### 2.1 Within-Source Performance")
    lines.append("")
    lines.append("TODO: Fill in after running")
    lines.append("")
    lines.append("### 2.2 Held-Out Transfer")
    lines.append("")
    lines.append("TODO: Fill in after running")
    lines.append("")
    lines.append("### 2.3 9×9 Transfer Matrix")
    lines.append("")
    lines.append("TODO: Fill in after running")
    lines.append("")
    lines.append("## 3. Representation Analysis")
    lines.append("")
    lines.append("### 3.1 CKA (Centered Kernel Alignment)")
    lines.append("")
    lines.append("TODO: Fill in after running")
    lines.append("")
    lines.append("### 3.2 Dimensionality Reduction")
    lines.append("")
    lines.append("See `umap/`, `tsne/`, `pca/` directories for 2D visualizations.")
    lines.append("")
    lines.append("### 3.3 Dataset Clustering")
    lines.append("")
    lines.append("TODO: Fill in after running")
    lines.append("")
    lines.append("## 4. Distribution Analysis")
    lines.append("")
    lines.append("TODO: Fill in after running")
    lines.append("")
    lines.append("## 5. Statistical Analysis")
    lines.append("")
    lines.append("### 5.1 Bootstrap Confidence Intervals")
    lines.append("")
    lines.append("TODO: Fill in after running")
    lines.append("")
    lines.append("### 5.2 Bayesian Comparison")
    lines.append("")
    lines.append("TODO: Fill in after running")
    lines.append("")
    lines.append("### 5.3 Mixed-Effects Model")
    lines.append("")
    lines.append("TODO: Fill in after running")
    lines.append("")
    lines.append("### 5.4 Effect Sizes")
    lines.append("")
    lines.append("TODO: Fill in after running")
    lines.append("")
    lines.append("## 6. Conclusions")
    lines.append("")
    lines.append("TODO: Fill in after analysis")
    lines.append("")
    lines.append("## File Manifest")
    lines.append("")
    lines.append("```")
    import glob as _g
    for f in sorted(_g.glob(str(RESULTS / "*"))):
        lines.append(f"  {Path(f).relative_to(RESULTS)}")
    lines.append("```")

    report = "\n".join(lines)
    with open(RESULTS / "phase64_report.md", "w") as f:
        f.write(report)
    logger.info("  Written: phase64_report.md")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Phase 64: Multi-Dataset Foundation Pretraining")
    parser.add_argument("--condition", choices=["A", "B", "C", "D", "E"],
                        help="Run a single condition")
    parser.add_argument("--all", action="store_true",
                        help="Run all conditions")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test (fewer epochs + subsample)")
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training, only evaluate existing models")
    args = parser.parse_args()

    logger.info(f"Phase 64 starting on {DEVICE}")
    logger.info(f"Quick mode: {args.quick}")
    logger.info(f"Eval only: {args.eval_only}")

    if args.quick:
        global PRETRAIN_EPOCHS, MAX_TRAIN_SAMPLES, MAX_TEST_SAMPLES
        PRETRAIN_EPOCHS = 5
        MAX_TRAIN_SAMPLES = 5000
        MAX_TEST_SAMPLES = 2000
        global N_BOOTSTRAP
        N_BOOTSTRAP = 100

    if not args.condition and not args.all:
        parser.print_help()
        return

    conditions_to_run = []
    if args.all:
        conditions_to_run = ["A", "B", "C", "D", "E"]
    else:
        conditions_to_run = [args.condition]

    # ═══════════════════════════════════════════════════════════════════════
    # Step 1: Data Loading
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("Loading all datasets...")
    logger.info("=" * 60)

    all_data = load_all_datasets()
    logger.info(f"Loaded {len(all_data)} datasets")

    # Fit scalers on source datasets
    source_data = {k: v for k, v in all_data.items() if k in SOURCE_DATASETS}
    scalers = fit_dataset_scalers(source_data)

    # Include external datasets in scaler fitting (using source scaler for each)
    for ext_name in EXTERNAL_DATASETS:
        if ext_name in all_data:
            scalers[ext_name] = StandardScaler().fit(all_data[ext_name]["X"])

    # Prepare splits
    splits = prepare_splits(all_data, scalers)
    logger.info(f"Splits ready for {len(splits)} datasets")

    # ═══════════════════════════════════════════════════════════════════════
    # Step 2: Training + Evaluation
    # ═══════════════════════════════════════════════════════════════════════
    all_results = {}
    all_transfer_matrices = {}
    all_similarities = {}
    all_embeddings = {}
    all_calibration = {}
    all_bootstrap = {}
    all_bayesian = {}
    all_mixed_effects = {}

    for condition in conditions_to_run:
        cleanup()

        if not args.eval_only:
            model = run_condition(condition, splits, quick=args.quick)
        else:
            # Load existing model
            ckpt_path = RESULTS / f"phase64_condition_{condition}_model.pt"
            if not ckpt_path.exists():
                logger.error(f"  No checkpoint found for condition {condition} at {ckpt_path}")
                continue
            model = HelixIDSFullPhase64(
                input_dim=INPUT_DIM,
                norm_type="rmsnorm" if condition == "D" else "batchnorm"
            ).to(DEVICE)
            state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
            model.load_state_dict(state)
            logger.info(f"  Loaded model for condition {condition}")

        # Evaluation
        results, transfer_matrix = run_evaluation(model, condition, splits)
        all_results[condition] = results
        all_transfer_matrices[condition] = transfer_matrix

        # Extract embeddings
        embeddings = extract_and_save_embeddings(model, splits, condition)
        all_embeddings[condition] = embeddings

        # Skip heavy representation analysis (CKA, UMAP, t-SNE) during main run
        # to avoid MPS segfaults from OpenBLAS multiprocessing.
        # These will be computed in a separate post-processing step.
        sim = {"cka": {}, "cca": {}, "mmd": {}, "wasserstein": {},
               "silhouette": {}, "davies_bouldin": {}, "ari": {}}
        all_similarities[condition] = sim

        # Calibration
        cal = run_calibration_analysis(model, splits, condition)
        all_calibration[condition] = cal

        # Cleanup
        del model
        cleanup()

    # ═══════════════════════════════════════════════════════════════════════
    # Step 3: Statistical Analysis
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("Statistical Analysis")
    logger.info("=" * 60)

    # Flag to skip heavy similarity if crashed previously
    rep_analysis_done = False

    # Collect macro_f1 values per condition
    condition_mf1 = defaultdict(list)
    for cond, results in all_results.items():
        for ds, metrics in results.items():
            if ds == "transfer_matrix":
                continue
            if isinstance(metrics, dict) and "macro_f1" in metrics:
                condition_mf1[cond].append(metrics["macro_f1"])

    # Bootstrap
    for cond in sorted(condition_mf1.keys()):
        vals = condition_mf1[cond]
        if len(vals) > 1:
            boot = bootstrap_ci(vals)
            all_bootstrap[cond] = boot
            logger.info(f"  Condition {cond}: mean={boot['mean']:.4f} "
                        f"[{boot['ci_lower']:.4f}, {boot['ci_upper']:.4f}]")
        else:
            all_bootstrap[cond] = {"mean": float(np.mean(vals)) if vals else 0, "error": "insufficient_data"}

    # Bayesian comparisons vs Condition A
    base_vals = condition_mf1.get("A", [])
    for cond in sorted(condition_mf1.keys()):
        if cond == "A":
            continue
        vals = condition_mf1[cond]
        if len(base_vals) > 0 and len(vals) > 0:
            bayes = bayesian_compare(base_vals, vals)
            ef = compute_effect_size(base_vals, vals)
            bayes.update(ef)
            all_bayesian[cond] = bayes
            logger.info(f"  A vs {cond}: t={bayes['t_statistic']:.4f}, "
                        f"p={bayes['p_value']:.4f}, BF={bayes['bayes_factor']:.2f}, "
                        f"d={bayes['cohens_d']:.4f}")

    # Mixed-effects model
    mf1_rows = []
    for cond, results in all_results.items():
        for ds, metrics in results.items():
            if ds == "transfer_matrix":
                continue
            if isinstance(metrics, dict) and "macro_f1" in metrics:
                mf1_rows.append({
                    "condition": cond,
                    "dataset": ds,
                    "macro_f1": metrics["macro_f1"],
                })
    if mf1_rows:
        mf1_df = pd.DataFrame(mf1_rows)
        mixed = mixed_effects_model(mf1_df)
        all_mixed_effects = mixed
        logger.info(f"  Mixed-effects model: AIC={mixed.get('aic', 'N/A')}")

    # ═══════════════════════════════════════════════════════════════════════
    # Step 4: Write CSV Outputs
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("Writing CSV outputs...")
    logger.info("=" * 60)

    write_joint_pretraining_results(all_results)
    write_heldout_transfer(all_results)
    write_transfer_matrix(all_transfer_matrices)
    write_similarity_csv(all_similarities)
    write_dataset_clustering(all_similarities)
    write_calibration_csv(all_calibration)

    # Write JSON statistical results
    with open(RESULTS / "bootstrap.json", "w") as f:
        json.dump(all_bootstrap, f, indent=2, default=str)
    with open(RESULTS / "bayesian.json", "w") as f:
        json.dump(all_bayesian, f, indent=2, default=str)
    with open(RESULTS / "mixed_effects.json", "w") as f:
        json.dump(all_mixed_effects, f, indent=2, default=str)

    logger.info("  Written: bootstrap.json, bayesian.json, mixed_effects.json")

    # ═══════════════════════════════════════════════════════════════════════
    # Step 5: Generate Reports
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("Generating reports...")
    logger.info("=" * 60)

    generate_summary(all_results, all_similarities, all_transfer_matrices,
                     all_calibration, all_bootstrap, all_bayesian,
                     all_mixed_effects)
    generate_report(all_results, all_similarities, all_transfer_matrices,
                    all_calibration, all_bootstrap, all_bayesian,
                    all_mixed_effects, all_embeddings)

    logger.info("\nPhase 64 complete!")
    logger.info(f"Results in: {RESULTS}")


if __name__ == "__main__":
    main()
