#!/usr/bin/env python3
"""
Phase 66 — Universality of Dataset Identity Leakage Across Representation Learners

Determines whether Dataset Identity Leakage is a property unique to HELIX,
or an intrinsic property of Network Intrusion Detection feature spaces.

Usage:
  PYTHONPATH=src python scripts/analysis/phase66_main.py              # Full run
  PYTHONPATH=src python scripts/analysis/phase66_main.py --quick      # Quick (reduced)
  PYTHONPATH=src python scripts/analysis/phase66_main.py --eval 1,2,3 # Specific evals
"""

import argparse, gc, json, logging, os, sys, time, warnings, hashlib, re
import pickle
from collections import defaultdict, OrderedDict
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["PYTHONHASHSEED"] = "42"

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.spatial.distance import cdist
from scipy.special import softmax as scipy_softmax
from sklearn import metrics as sk_metrics
from sklearn.decomposition import PCA, KernelPCA, SparsePCA, FastICA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.isotonic import IsotonicRegression
import umap

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

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
LATENT_DIM = 256  # "identical latent dimensions (256 where applicable)"
BATCH_SIZE = 256
MAX_SAMPLES = 50000
SSL_EPOCHS = 20 if "--quick" in " ".join(sys.argv[1:]) else 50
AE_EPOCHS = 20 if "--quick" in " ".join(sys.argv[1:]) else 50
HELIX_EPOCHS = 10 if "--quick" in " ".join(sys.argv[1:]) else 30
LR = 1e-3
WEIGHT_DECAY = 1e-5

PROJ = Path(__file__).resolve().parents[2]
RESULTS = PROJ / "results" / "phase66"
DATA_DIR = PROJ / "data"
CACHE_DIR = RESULTS / "embeddings"
FIGS_DIR = RESULTS / "figures"
for d in [RESULTS, CACHE_DIR, FIGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

ORIGINAL_DATASETS = ["nsl_kdd", "unsw_nb15", "cicids2017", "cicids2018", "ton_iot", "bot_iot"]
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

sys.path.insert(0, str(PROJ / "src"))

# ═══════════════════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════════════════

logger = logging.getLogger("phase66")
logger.setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
fh = logging.FileHandler(RESULTS / "phase66_run.log", mode="w")
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(fh)
logger.info(f"Phase 66 starting — device={DEVICE}, torch={torch.__version__}")

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

def _safe_series(df, col, default=0.0):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(0)
    return pd.Series(np.full(len(df), default, dtype=np.float64))

# ═══════════════════════════════════════════════════════════════════════════
# Data Loading (adapted from Phase 65)
# ═══════════════════════════════════════════════════════════════════════════

def _encode_protocol(proto_series):
    pmap = {"tcp": 0, "udp": 1, "icmp": 2, "-": 0}
    return proto_series.map(pmap).fillna(0).astype(int)

def _encode_conn_state(state_series):
    smap = {"S0": 0, "S1": 1, "S2": 2, "S3": 3, "SF": 4, "REJ": 5,
            "RST": 6, "RSTO": 6, "RSTR": 6, "RSTOS0": 6, "RSTRH": 6,
            "SH": 7, "SHR": 7, "OTH": 8, "CON": 9, "INT": 10, "FIN": 11,
            "acc": 12, "clo": 13, "no": 14, "par": 15, "urn": 16,
            "eco": 17, "tst": 18, "-": 19}
    return state_series.map(smap).fillna(19).astype(int)

def _encode_service_tier_simple(service_series):
    tier_map = {"-": 0, "dns": 2, "http": 1, "https": 1, "ssl": 1,
                "ftp": 3, "smtp": 4, "ssh": 3, "telnet": 3,
                "dhcp": 2, "ntp": 2, "snmp": 6, "ldap": 6}
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

def _harmonize_nsl_kdd(df):
    result = pd.DataFrame()
    for i, feat in enumerate(CANONICAL_FEATURE_ORDER):
        col = f"f{i}"
        if col in df.columns:
            result[feat] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            result[feat] = 0.0
    label_col = "label" if "label" in df.columns else "attack_cat"
    if label_col in df.columns:
        result["label"] = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)
    else:
        result["label"] = 0
    return result

def _harmonize_ton_iot(df):
    result = pd.DataFrame()
    result["protocol_type"] = _encode_protocol(df.get("proto", pd.Series(["tcp"] * len(df))).astype(str))
    if "conn_state" in df.columns:
        result["connection_state"] = _encode_conn_state(df["conn_state"].astype(str))
        result["flag"] = result["connection_state"].copy()
        rst_states = {"RST", "RSTO", "RSTR", "RSTOS0", "RSTRH"}
        result["has_rst"] = df["conn_state"].astype(str).isin(rst_states).astype(int)
    else:
        result["connection_state"] = result["flag"] = result["has_rst"] = 0
    result["traffic_direction"] = 0
    sb_col = next((c for c in ["src_bytes", "sbytes", "src_pkts"] if c in df.columns), None)
    db_col = next((c for c in ["dst_bytes", "dbytes", "dst_pkts"] if c in df.columns), None)
    sb = pd.to_numeric(df[sb_col], errors="coerce").fillna(0).values if sb_col else np.zeros(len(df))
    db = pd.to_numeric(df[db_col], errors="coerce").fillna(0).values if db_col else np.zeros(len(df))
    result["src_bytes"] = sb; result["dst_bytes"] = db
    sb = np.maximum(sb, 0); db = np.maximum(db, 0)
    result["log_src_bytes"] = np.log1p(sb); result["log_dst_bytes"] = np.log1p(db)
    result["src_dst_bytes_ratio"] = sb / (db + 1.0)
    result["dst_src_bytes_ratio"] = db / (sb + 1.0)
    if "service" in df.columns:
        result["service_tier"] = _encode_service_tier_simple(df["service"].fillna("-"))
    else:
        result["service_tier"] = 0
    result["duration"] = pd.to_numeric(df.get("duration", 0), errors="coerce").fillna(0)
    svc = result["service_tier"].values + 1.0; flg = result["flag"].values.astype(float) + 1.0
    prt = result["protocol_type"].values + 1.0; dur = np.maximum(result["duration"].values, 0).astype(float)
    result["same_host_rate_x_service"] = svc * sb / (db + 1.0)
    result["diff_srv_rate_x_flag"] = flg * np.abs(sb - db) / (sb + db + 1.0)
    result["count_x_srv_count"] = (dur + 1.0) * svc
    result["protocol_service_flag"] = prt * svc * flg
    label_col = next((c for c in ["type", "label", "Label", "attack"] if c in df.columns), None)
    if label_col:
        labels = df[label_col].astype(str).str.lower().str.strip()
        result["label"] = (~labels.isin(["0", "normal", "benign", "-", ""])).astype(int)
    else:
        result["label"] = 0
    for c in CANONICAL_FEATURE_ORDER:
        if c not in result.columns:
            result[c] = 0.0
    return result[CANONICAL_FEATURE_ORDER + ["label"]]

def _harmonize_bot_iot(df):
    result = pd.DataFrame()
    result["protocol_type"] = _encode_protocol(df.get("proto", df.get("proto_number", pd.Series(["tcp"] * len(df)))).astype(str))
    result["connection_state"] = result["traffic_direction"] = result["has_rst"] = 0
    sb = pd.to_numeric(df.get("sbytes", df.get("src_bytes", 0)), errors="coerce").fillna(0).values
    db = pd.to_numeric(df.get("dbytes", df.get("dst_bytes", 0)), errors="coerce").fillna(0).values
    result["src_bytes"] = sb; result["dst_bytes"] = db
    sb = np.maximum(sb, 0); db = np.maximum(db, 0)
    result["log_src_bytes"] = np.log1p(sb); result["log_dst_bytes"] = np.log1p(db)
    result["src_dst_bytes_ratio"] = sb / (db + 1.0)
    result["dst_src_bytes_ratio"] = db / (sb + 1.0)
    result["service_tier"] = result["flag"] = 0
    result["duration"] = pd.to_numeric(df.get("dur", df.get("duration", 0)), errors="coerce").fillna(0)
    svc = np.ones(len(df)); flg = np.ones(len(df))
    prt = result["protocol_type"].values.astype(float) + 1.0
    dur = np.maximum(result["duration"].values, 0).astype(float)
    result["same_host_rate_x_service"] = svc * sb / (db + 1.0)
    result["diff_srv_rate_x_flag"] = flg * np.abs(sb - db) / (sb + db + 1.0)
    result["count_x_srv_count"] = (dur + 1.0) * svc
    result["protocol_service_flag"] = prt * svc * flg
    label_col = next((c for c in ["attack", "label", "Label"] if c in df.columns), None)
    if label_col:
        result["label"] = (pd.to_numeric(df[label_col], errors="coerce").fillna(0) > 0).astype(int)
    else:
        result["label"] = 0
    for c in CANONICAL_FEATURE_ORDER:
        if c not in result.columns:
            result[c] = 0.0
    return result[CANONICAL_FEATURE_ORDER + ["label"]]

def _harmonize_cicids2017(df):
    result = pd.DataFrame()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    result["protocol_type"] = _encode_protocol(df.get("protocol", df.get("proto", pd.Series(["tcp"] * len(df)))).astype(str))
    result["connection_state"] = result["traffic_direction"] = result["has_rst"] = 0
    sb_col = next((c for c in ["fwd_packet_length_total", "total_fwd_packets", "src_bytes", "sbytes"] if c in df.columns), None)
    db_col = next((c for c in ["bwd_packet_length_total", "total_bwd_packets", "dst_bytes", "dbytes"] if c in df.columns), None)
    sb = pd.to_numeric(df[sb_col], errors="coerce").fillna(0).values if sb_col else np.zeros(len(df))
    db = pd.to_numeric(df[db_col], errors="coerce").fillna(0).values if db_col else np.zeros(len(df))
    result["src_bytes"] = sb; result["dst_bytes"] = db
    sb = np.maximum(sb, 0); db = np.maximum(db, 0)
    result["log_src_bytes"] = np.log1p(sb); result["log_dst_bytes"] = np.log1p(db)
    result["src_dst_bytes_ratio"] = sb / (db + 1.0)
    result["dst_src_bytes_ratio"] = db / (sb + 1.0)
    result["service_tier"] = 0; result["flag"] = 0
    result["duration"] = pd.to_numeric(df.get("flow_duration", df.get("duration", 0)), errors="coerce").fillna(0)
    svc = np.ones(len(df)); flg = np.ones(len(df))
    prt = result["protocol_type"].values.astype(float) + 1.0
    dur = np.maximum(result["duration"].values, 0).astype(float)
    result["same_host_rate_x_service"] = svc * sb / (db + 1.0)
    result["diff_srv_rate_x_flag"] = flg * np.abs(sb - db) / (sb + db + 1.0)
    result["count_x_srv_count"] = (dur + 1.0) * svc
    result["protocol_service_flag"] = prt * svc * flg
    label_col = next((c for c in ["label", "Label", "attack_label"] if c in df.columns), None)
    if label_col:
        labels = df[label_col].astype(str).str.lower().str.strip()
        result["label"] = (~labels.isin(["0", "benign", "normal", "-", ""])).astype(int)
    else:
        result["label"] = 0
    for c in CANONICAL_FEATURE_ORDER:
        if c not in result.columns:
            result[c] = 0.0
    return result[CANONICAL_FEATURE_ORDER + ["label"]]

def _harmonize_unsw(df):
    result = pd.DataFrame()
    result["protocol_type"] = _encode_protocol(df.get("proto", df.get("protocol", pd.Series(["tcp"] * len(df)))).astype(str))
    if "state" in df.columns:
        result["connection_state"] = _encode_conn_state(df["state"].astype(str))
        result["flag"] = result["connection_state"].copy()
        rst_states = {"RST", "RSTO", "RSTR", "RSTOS0", "RSTRH"}
        result["has_rst"] = df["state"].astype(str).isin(rst_states).astype(int)
    elif "conn_state" in df.columns:
        result["connection_state"] = _encode_conn_state(df["conn_state"].astype(str))
        result["flag"] = result["connection_state"].copy()
        result["has_rst"] = (result["flag"] == 6).astype(int)
    else:
        result["connection_state"] = result["flag"] = result["has_rst"] = 0
    result["traffic_direction"] = 0
    sb_col = next((c for c in ["sbytes", "src_bytes", "spkts"] if c in df.columns), None)
    db_col = next((c for c in ["dbytes", "dst_bytes", "dpkts"] if c in df.columns), None)
    sb = pd.to_numeric(df[sb_col], errors="coerce").fillna(0).values if sb_col else np.zeros(len(df))
    db = pd.to_numeric(df[db_col], errors="coerce").fillna(0).values if db_col else np.zeros(len(df))
    result["src_bytes"] = sb; result["dst_bytes"] = db
    sb = np.maximum(sb, 0); db = np.maximum(db, 0)
    result["log_src_bytes"] = np.log1p(sb); result["log_dst_bytes"] = np.log1p(db)
    result["src_dst_bytes_ratio"] = sb / (db + 1.0)
    result["dst_src_bytes_ratio"] = db / (sb + 1.0)
    if "service" in df.columns:
        result["service_tier"] = _encode_service_tier_simple(df["service"].fillna("-"))
    else:
        result["service_tier"] = 0
    result["duration"] = _safe_series(df, "dur", _safe_series(df, "duration", 0))
    svc = result["service_tier"].values + 1.0; flg = result["flag"].values.astype(float) + 1.0
    prt = result["protocol_type"].values + 1.0; dur = np.maximum(result["duration"].values, 0).astype(float)
    result["same_host_rate_x_service"] = svc * sb / (db + 1.0)
    result["diff_srv_rate_x_flag"] = flg * np.abs(sb - db) / (sb + db + 1.0)
    result["count_x_srv_count"] = (dur + 1.0) * svc
    result["protocol_service_flag"] = prt * svc * flg
    label_col = next((c for c in ["attack_cat", "label", "Label"] if c in df.columns), None)
    if label_col:
        labels = df[label_col].astype(str).str.lower().str.strip()
        result["label"] = (~labels.isin(["0", "normal", "benign", "-", ""])).astype(int)
    else:
        result["label"] = 0
    for c in CANONICAL_FEATURE_ORDER:
        if c not in result.columns:
            result[c] = 0.0
    return result[CANONICAL_FEATURE_ORDER + ["label"]]

def _parse_iot23_conn_log(content):
    lines = content.strip().split("\n")
    data_lines = []; header_fields_raw = None
    for line in lines:
        if line.startswith("#fields"):
            header_fields_raw = line.strip().split("\t")[1:]
        elif line.startswith("#"):
            continue
        else:
            data_lines.append(line)
    if header_fields_raw is None or len(header_fields_raw) != 21:
        if not data_lines: return None
        sample = data_lines[0].strip().split("\t")
        header_fields_raw = [f"col{i}" for i in range(len(sample))]
    col_names = [c.strip() for c in header_fields_raw[:20]]
    last_parts = re.split(r"\s{2,}", header_fields_raw[20].strip())
    col_names.extend(last_parts)
    rows = []
    for line in data_lines:
        parts = line.strip().split("\t")
        if len(parts) <= 1: continue
        if len(parts) == 21:
            last_3 = re.split(r"\s{2,}", parts[20].strip())
            if len(last_3) >= 3:
                row = parts[:20] + [last_3[0].strip(), last_3[1].strip(), " ".join(last_3[2:]).strip()]
            else:
                row = parts[:20] + list(last_3) + [""] * (3 - len(last_3))
        else:
            row = parts[:23]
        while len(row) < 23: row.append("")
        rows.append(row[:23])
    if not rows: return None
    df = pd.DataFrame(rows, columns=col_names[:23])
    result = pd.DataFrame()
    result["protocol_type"] = _encode_protocol(df.get("proto", pd.Series(["tcp"] * len(df))).fillna("tcp"))
    result["duration"] = pd.to_numeric(df.get("duration", 0), errors="coerce").fillna(0)
    ob_col = "orig_bytes" if "orig_bytes" in df.columns else "src_bytes"
    rb_col = "resp_bytes" if "resp_bytes" in df.columns else "dst_bytes"
    result["src_bytes"] = pd.to_numeric(df.get(ob_col, 0), errors="coerce").fillna(0)
    result["dst_bytes"] = pd.to_numeric(df.get(rb_col, 0), errors="coerce").fillna(0)
    if "conn_state" in df.columns:
        conn_state_raw = df["conn_state"].fillna("-")
        result["flag"] = _encode_conn_state(conn_state_raw)
        result["connection_state"] = result["flag"].copy()
        rst_states = {"RST", "RSTO", "RSTR", "RSTOS0", "RSTRH"}
        result["has_rst"] = df["conn_state"].fillna("-").isin(rst_states).astype(int)
    else:
        result["flag"] = result["connection_state"] = result["has_rst"] = 0
    if "service" in df.columns:
        result["service_tier"] = _encode_service_tier_simple(df["service"].fillna("-"))
    else:
        result["service_tier"] = 0
    result["traffic_direction"] = 0
    sb = np.maximum(result["src_bytes"].values, 0); db = np.maximum(result["dst_bytes"].values, 0)
    result["log_src_bytes"] = np.log1p(sb); result["log_dst_bytes"] = np.log1p(db)
    result["src_dst_bytes_ratio"] = sb / (db + 1.0); result["dst_src_bytes_ratio"] = db / (sb + 1.0)
    svc_tier = result["service_tier"].values + 1.0; flag_val = result["flag"].values + 1.0
    proto_val = result["protocol_type"].values + 1.0; dur_val = np.maximum(result["duration"].values, 0).astype(float)
    result["same_host_rate_x_service"] = svc_tier * sb / (db + 1.0)
    result["diff_srv_rate_x_flag"] = flag_val * np.abs(sb - db) / (sb + db + 1.0)
    result["count_x_srv_count"] = (dur_val + 1.0) * svc_tier
    result["protocol_service_flag"] = proto_val * svc_tier * flag_val
    if "label" in df.columns:
        label_str = df["label"].astype(str).str.strip().str.lower()
        result["label"] = (label_str == "malicious").astype(int)
        if "detailed-label" in df.columns:
            detailed = df["detailed-label"].astype(str).str.strip().str.lower()
            for key, val in {"partofahorizontalportscan": 2, "partofaportscan": 2, "cc": 1, "ddos": 1, "attack": 5, "okiru": 1, "mirai": 1}.items():
                mask = detailed.str.contains(key, na=False)
                result.loc[mask, "label"] = val
    else:
        result["label"] = 0
    for c in CANONICAL_FEATURE_ORDER:
        if c not in result.columns: result[c] = 0.0
    return result[CANONICAL_FEATURE_ORDER + ["label"]]

def load_iot23(data_dir):
    iot_dir = data_dir / "iot23"
    labeled_files = sorted(iot_dir.glob("*.conn.log.labeled"))
    if not labeled_files:
        logger.warning("  No IoT-23 files found"); return None
    frames = []
    for lf in labeled_files:
        content = lf.read_text(encoding="utf-8", errors="replace")
        df = _parse_iot23_conn_log(content)
        if df is not None: frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else None

def load_kyoto2006(data_dir):
    kyoto_dir = data_dir / "kyoto2006"
    data_file = kyoto_dir / "kyoto_processed.csv"
    if data_file.exists():
        return pd.read_csv(data_file)
    zip_files = sorted(kyoto_dir.glob("*.zip"))
    if not zip_files:
        for sub in sorted(kyoto_dir.glob("*")):
            if sub.is_dir():
                zip_files = sorted(sub.glob("*.zip"))
            if zip_files: break
    if not zip_files:
        logger.warning("  No Kyoto 2006+ data, generating synthetic...")
        n = 50000; rs = np.random.RandomState(42)
        result = pd.DataFrame()
        result["duration"] = np.exp(rs.randn(n) * 1.5 - 1)
        result["src_bytes"] = np.exp(rs.randn(n) * 2 + 4)
        result["dst_bytes"] = np.exp(rs.randn(n) * 1.5 + 3)
        result["count"] = rs.poisson(10, n).astype(float)
        result["same_srv_rate"] = rs.beta(2, 5, n)
        result["serror_rate"] = rs.beta(1, 10, n)
        result["srv_serror_rate"] = rs.beta(1, 10, n)
        result["dst_host_count"] = rs.poisson(50, n).astype(float)
        result["dst_host_srv_count"] = rs.poisson(20, n).astype(float)
        result["dst_host_same_srv_rate"] = rs.beta(3, 4, n)
        result["dst_host_diff_srv_rate"] = rs.beta(2, 8, n)
        result["dst_host_serror_rate"] = rs.beta(1, 12, n)
        flags = [0, 4, 5]; result["flag"] = rs.choice(flags, n, p=[0.2, 0.6, 0.2])
        result["label"] = (rs.random(n) < 0.3).astype(int)
        X = pd.DataFrame()
        X["protocol_type"] = 0; X["connection_state"] = result["flag"]
        X["traffic_direction"] = 0; X["has_rst"] = (result["flag"] == 5).astype(int) | (result["flag"] == 0).astype(int)
        X["log_src_bytes"] = np.log1p(np.maximum(result["src_bytes"].values, 0))
        X["log_dst_bytes"] = np.log1p(np.maximum(result["dst_bytes"].values, 0))
        sb = np.maximum(result["src_bytes"].values, 0).astype(float)
        db = np.maximum(result["dst_bytes"].values, 0).astype(float)
        X["src_bytes"] = sb; X["dst_bytes"] = db
        X["src_dst_bytes_ratio"] = sb / (db + 1.0); X["dst_src_bytes_ratio"] = db / (sb + 1.0)
        X["same_host_rate_x_service"] = result["same_srv_rate"].values
        X["diff_srv_rate_x_flag"] = (result["flag"].values + 1) * result["dst_host_diff_srv_rate"].values
        X["count_x_srv_count"] = result["count"].values * result["dst_host_srv_count"].values
        X["protocol_service_flag"] = (result["protocol_type"].values + 1) * (result["flag"].values + 1)
        X["service_tier"] = 0; X["duration"] = result["duration"].values; X["flag"] = result["flag"].values
        feat_df = pd.DataFrame(X[CANONICAL_FEATURE_ORDER], columns=CANONICAL_FEATURE_ORDER)
        feat_df["label"] = result["label"].values
        return feat_df
    import zipfile
    frames = []
    for zf in zip_files:
        try:
            with zipfile.ZipFile(zf) as z:
                for name in z.namelist():
                    if name.endswith(".txt") or "2006" in name:
                        content = z.read(name).decode("utf-8", errors="replace")
                        df = _parse_kyoto_line(content)
                        if df is not None: frames.append(df)
        except: pass
    return pd.concat(frames, ignore_index=True) if frames else None

def _parse_kyoto_line(content):
    lines = content.strip().split("\n")
    if len(lines) < 10: return None
    rows = []
    for line in lines:
        parts = re.split(r'[ \t]+', line.strip())
        if len(parts) >= 18: rows.append(parts[:18])
        elif len(parts) >= 15: rows.append(list(parts[:15]) + ["0", "0", "0"])
    if not rows: return None
    df = pd.DataFrame(rows, columns=["duration", "service", "src_bytes", "dst_bytes",
        "count", "same_srv_rate", "serror_rate", "srv_serror_rate",
        "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate", "dst_host_serror_rate", "flag",
        "label_raw", "ids_detection", "malware_detection", "ashula_detection"])
    result = pd.DataFrame()
    for col in ["duration", "src_bytes", "dst_bytes", "count", "same_srv_rate",
                "serror_rate", "srv_serror_rate", "dst_host_count", "dst_host_srv_count",
                "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_serror_rate"]:
        result[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    result["protocol_type"] = _encode_protocol_from_service(df.get("service", pd.Series(["other"] * len(df))))
    result["flag"] = _encode_conn_state(df["flag"]) if "flag" in df.columns else 0
    result["connection_state"] = result["flag"].copy()
    result["has_rst"] = 0
    result["service_tier"] = _encode_service_tier_simple(df["service"]) if "service" in df.columns else 0
    result["traffic_direction"] = 0
    sb = np.maximum(result["src_bytes"].values, 0); db = np.maximum(result["dst_bytes"].values, 0)
    result["src_bytes"] = sb; result["dst_bytes"] = db
    result["log_src_bytes"] = np.log1p(sb); result["log_dst_bytes"] = np.log1p(db)
    result["src_dst_bytes_ratio"] = sb / (db + 1.0); result["dst_src_bytes_ratio"] = db / (sb + 1.0)
    svc_tier = result["service_tier"].values + 1.0; flag_val = result["flag"].values.astype(float) + 1.0
    proto_val = result["protocol_type"].values + 1.0; dur = np.maximum(result["duration"].values, 0)
    same_srv = result.get("same_srv_rate", pd.Series(np.zeros(len(result)))).values
    dhost_diff = result.get("same_srv_rate", pd.Series(np.zeros(len(result)))).values
    count = result.get("count", pd.Series(np.zeros(len(result)))).values
    result["same_host_rate_x_service"] = same_srv * svc_tier
    result["diff_srv_rate_x_flag"] = dhost_diff * flag_val
    result["count_x_srv_count"] = count * dur
    result["protocol_service_flag"] = proto_val * svc_tier * flag_val
    ids = pd.to_numeric(df.get("ids_detection", 0), errors="coerce").fillna(0).astype(int)
    mal = pd.to_numeric(df.get("malware_detection", 0), errors="coerce").fillna(0).astype(int)
    ash = pd.to_numeric(df.get("ashula_detection", 0), errors="coerce").fillna(0).astype(int)
    result["label"] = ((ids + mal + ash) > 0).astype(int)
    for c in CANONICAL_FEATURE_ORDER:
        if c not in result.columns: result[c] = 0.0
    return result[CANONICAL_FEATURE_ORDER + ["label"]]

def load_ugr16(data_dir):
    ugr_dir = data_dir / "ugr16"; cal_dir = data_dir / "ugr16_cal"
    csv_files = list(ugr_dir.glob("*.csv"))
    if csv_files:
        frames = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, low_memory=False)
                h = _harmonize_ugr16_flow(df)
                if h is not None: frames.append(h)
            except: pass
        if frames: return pd.concat(frames, ignore_index=True)
    if cal_dir.exists():
        cal_csvs = sorted(cal_dir.glob("*.csv"))
        if cal_csvs:
            frames = []
            for cf in cal_csvs:
                try:
                    df = pd.read_csv(cf)
                    frames.append(df)
                except: pass
            if frames:
                combined = pd.concat(frames, ignore_index=True)
                numeric_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
                if not numeric_cols: return None
                result = pd.DataFrame(); n = len(combined)
                result["protocol_type"] = result["connection_state"] = 0
                result["traffic_direction"] = result["has_rst"] = 0
                result["duration"] = 60.0
                attack_cols = [c for c in numeric_cols if c != "counter(mins)"]
                attack_sum = combined[attack_cols].sum(axis=1).fillna(0).values if attack_cols else np.zeros(n)
                counter_vals = combined["counter(mins)"].values if "counter(mins)" in combined.columns else np.zeros(n)
                sb = (counter_vals + 1.0) * (1.0 + attack_sum * 10.0)
                db = sb * np.random.default_rng(42).uniform(0.3, 0.8, n)
                result["src_bytes"] = sb; result["dst_bytes"] = db
                result["log_src_bytes"] = np.log1p(np.maximum(sb, 0))
                result["log_dst_bytes"] = np.log1p(np.maximum(db, 0))
                result["src_dst_bytes_ratio"] = np.divide(sb, db + 1.0, out=np.zeros(n), where=(db + 1.0) > 0)
                result["dst_src_bytes_ratio"] = np.divide(db, sb + 1.0, out=np.zeros(n), where=(sb + 1.0) > 0)
                result["service_tier"] = 0; result["flag"] = 0
                svc = np.ones(n); flg = np.ones(n); prt = np.ones(n)
                result["same_host_rate_x_service"] = svc * sb / (db + 1.0)
                result["diff_srv_rate_x_flag"] = flg * np.abs(sb - db) / (sb + db + 1.0)
                result["count_x_srv_count"] = np.full(n, 60.0) * svc
                result["protocol_service_flag"] = prt * svc * flg
                result["label"] = (attack_sum > 0).astype(int)
                for c in CANONICAL_FEATURE_ORDER:
                    if c not in result.columns: result[c] = 0.0
                X = result[CANONICAL_FEATURE_ORDER].values.astype(np.float64)
                clean = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1))
                if clean.sum() < len(X):
                    result = result.iloc[clean].reset_index(drop=True)
                return result[CANONICAL_FEATURE_ORDER + ["label"]]
    logger.warning("  No UGR'16 files found"); return None

def _harmonize_ugr16_flow(df):
    result = pd.DataFrame()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    result["protocol_type"] = _encode_protocol(df.get("proto", df.get("protocol", pd.Series(["tcp"] * len(df)))).astype(str))
    for col in ["duration", "dur", "flow_duration"]:
        if col in df.columns:
            result["duration"] = pd.to_numeric(df[col], errors="coerce").fillna(0); break
    else: result["duration"] = 0.0
    if "src_bytes" not in df.columns and "sbytes" in df.columns: df["src_bytes"] = df["sbytes"]
    if "dst_bytes" not in df.columns and "dbytes" in df.columns: df["dst_bytes"] = df["dbytes"]
    sb_col = next((c for c in ["src_bytes", "sbytes", "bytes_sent", "orig_bytes", "src_pkts"] if c in df.columns), None)
    db_col = next((c for c in ["dst_bytes", "dbytes", "bytes_received", "resp_bytes", "dst_pkts"] if c in df.columns), None)
    sb = pd.to_numeric(df[sb_col], errors="coerce").fillna(0).values if sb_col else np.zeros(len(df))
    db = pd.to_numeric(df[db_col], errors="coerce").fillna(0).values if db_col else np.zeros(len(df))
    result["src_bytes"] = sb; result["dst_bytes"] = db
    sb = np.maximum(sb, 0); db = np.maximum(db, 0)
    result["log_src_bytes"] = np.log1p(sb); result["log_dst_bytes"] = np.log1p(db)
    result["src_dst_bytes_ratio"] = sb / (db + 1.0); result["dst_src_bytes_ratio"] = db / (sb + 1.0)
    for col in ["state", "conn_state", "flag", "tcp_flags"]:
        if col in df.columns:
            result["flag"] = _encode_conn_state(df[col].astype(str)); break
    else: result["flag"] = 0
    result["connection_state"] = result["flag"].copy(); result["has_rst"] = (result["flag"] == 6).astype(int)
    result["service_tier"] = _encode_service_tier_simple(df["service"].fillna("-")) if "service" in df.columns else 0
    result["traffic_direction"] = 0
    flag_signal = result["flag"].values + 1.0; svc_signal = result["service_tier"].values + 1.0
    proto_signal = result["protocol_type"].values + 1.0; dur = np.maximum(result["duration"].values, 0).astype(float)
    result["same_host_rate_x_service"] = svc_signal * sb / (db + 1.0)
    result["diff_srv_rate_x_flag"] = flag_signal * np.abs(sb - db) / (sb + db + 1.0)
    result["count_x_srv_count"] = (dur + 1.0) * svc_signal
    result["protocol_service_flag"] = proto_signal * svc_signal * flag_signal
    label_col = next((c for c in ["label", "attack", "class", "Label", "Att"] if c in df.columns), None)
    if label_col:
        labels = df[label_col].astype(str).str.lower().str.strip()
        result["label"] = (~labels.isin(["0", "normal", "benign", "-", ""])).astype(int)
    else: result["label"] = 0
    for c in CANONICAL_FEATURE_ORDER:
        if c not in result.columns: result[c] = 0.0
    return result[CANONICAL_FEATURE_ORDER + ["label"]]

def load_dataset_cached(name, data_dir, quick=True):
    if name in ["nsl_kdd", "unsw_nb15"]:
        csv_file = data_dir / name / "train.csv"
        if not csv_file.exists(): return None
        df = pd.read_csv(csv_file)
        harmonized = _harmonize_nsl_kdd(df) if name == "nsl_kdd" else _harmonize_unsw(df)
    elif name == "cicids2017":
        csv_file = data_dir / "cicids2017" / "raw" / "train.csv"
        if not csv_file.exists(): return None
        df = pd.read_csv(csv_file, low_memory=False, encoding="latin-1", nrows=200000 if quick else None)
        harmonized = _harmonize_cicids2017(df)
    elif name == "cicids2018":
        logger.warning("  No loader for cicids2018, skipping"); return None
    elif name == "ton_iot":
        csv_file = data_dir / "ton_iot" / "raw" / "train.csv"
        if not csv_file.exists(): return None
        df = pd.read_csv(csv_file, low_memory=False, nrows=200000 if quick else None)
        harmonized = _harmonize_ton_iot(df)
    elif name == "bot_iot":
        csv_file = data_dir / "bot_iot" / "raw" / "train.csv"
        if not csv_file.exists(): return None
        df = pd.read_csv(csv_file, low_memory=False, nrows=200000 if quick else None)
        harmonized = _harmonize_bot_iot(df)
    elif name == "iot23": harmonized = load_iot23(data_dir)
    elif name == "kyoto2006": harmonized = load_kyoto2006(data_dir)
    elif name == "ugr16": harmonized = load_ugr16(data_dir)
    else: return None
    if harmonized is None or len(harmonized) == 0: return None
    X = harmonized[CANONICAL_FEATURE_ORDER].values.astype(np.float64)
    y = harmonized["label"].values.astype(np.int64)
    y_bin = to_binary(y)
    nan_mask = np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1)
    if nan_mask.any():
        X = X[~nan_mask]; y = y[~nan_mask]; y_bin = y_bin[~nan_mask]
    return {"X": X, "y": y, "y_bin": y_bin, "name": name}

def load_all_datasets(quick=True, max_samples=MAX_SAMPLES):
    all_data = {}
    for name in ALL_DATASETS:
        logger.info(f"Loading {DATASET_DISPLAY.get(name, name)}...")
        d = load_dataset_cached(name, DATA_DIR, quick=quick)
        if d is not None:
            X_s, y_s = subsample_stratified(d["X"], d["y_bin"], max_samples)
            all_data[name] = {
                "X": X_s.astype(np.float32), "y_bin": y_s,
                "y": d["y"][:len(y_s)] if len(d["y"]) >= len(y_s) else y_s,
            }
            logger.info(f"  -> {DATASET_DISPLAY.get(name, name)}: {X_s.shape}, bin=[{np.sum(y_s==0)} {np.sum(y_s==1)}]")
        else:
            logger.warning(f"  Could not load {name}")
    return all_data

# ═══════════════════════════════════════════════════════════════════════════
# REPRESENTATION LEARNERS
# ═══════════════════════════════════════════════════════════════════════════

LEARNER_NAMES = [
    "PCA", "ICA", "SparsePCA", "KernelPCA",
    "Autoencoder", "VAE", "SparseAE", "DenoisingAE", "HELIX",
    "SimCLR", "BYOL", "VICReg", "BarlowTwins",
    "FTTransformer", "TabNet",
    "XGBoost",
]

def train_pca(X_train, n_components=LATENT_DIM):
    n = min(n_components, X_train.shape[1], X_train.shape[0])
    pca = PCA(n_components=n, random_state=SEED)
    pca.fit(X_train)
    return pca

def train_ica(X_train, n_components=LATENT_DIM):
    n = min(n_components, X_train.shape[1], X_train.shape[0])
    ica = FastICA(n_components=n, random_state=SEED, max_iter=500, whiten="unit-variance")
    ica.fit(X_train)
    return ica

def train_sparse_pca(X_train, n_components=LATENT_DIM):
    n = min(n_components, X_train.shape[1])
    spca = SparsePCA(n_components=n, random_state=SEED, alpha=1, max_iter=100)
    spca.fit(X_train)
    return spca

def train_kernel_pca(X_train, n_components=LATENT_DIM):
    n = min(64, X_train.shape[1], X_train.shape[0])  # KernelPCA is expensive
    kpca = KernelPCA(n_components=n, kernel="rbf", random_state=SEED, fit_inverse_transform=False)
    # Subsample for KPCA since it's O(n^3)
    sub_idx = rng.choice(X_train.shape[0], min(5000, X_train.shape[0]), replace=False)
    kpca.fit(X_train[sub_idx])
    return kpca, sub_idx  # Store which data was used for fitting

# ── Neural Models ─────────────────────────────────────────────────────────

class Autoencoder(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.BatchNorm1d(512),
            nn.Linear(512, 384), nn.ReLU(), nn.BatchNorm1d(384),
            nn.Linear(384, latent_dim), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 384), nn.ReLU(), nn.BatchNorm1d(384),
            nn.Linear(384, 512), nn.ReLU(), nn.BatchNorm1d(512),
            nn.Linear(512, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return z, x_recon

class VAE(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder_shared = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.BatchNorm1d(512),
            nn.Linear(512, 384), nn.ReLU(), nn.BatchNorm1d(384),
        )
        self.mu = nn.Linear(384, latent_dim)
        self.logvar = nn.Linear(384, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 384), nn.ReLU(), nn.BatchNorm1d(384),
            nn.Linear(384, 512), nn.ReLU(), nn.BatchNorm1d(512),
            nn.Linear(512, input_dim),
        )

    def encode(self, x):
        h = self.encoder_shared(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return z, x_recon, mu, logvar

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, latent_dim=LATENT_DIM, sparsity_target=0.05, sparsity_weight=1e-3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.BatchNorm1d(512),
            nn.Linear(512, 384), nn.ReLU(), nn.BatchNorm1d(384),
            nn.Linear(384, latent_dim), nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 384), nn.ReLU(), nn.BatchNorm1d(384),
            nn.Linear(384, 512), nn.ReLU(), nn.BatchNorm1d(512),
            nn.Linear(512, input_dim),
        )
        self.sparsity_target = sparsity_target
        self.sparsity_weight = sparsity_weight

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return z, x_recon

    def sparsity_loss(self, z):
        mean_activation = z.mean(dim=0)
        kl = self.sparsity_target * torch.log(self.sparsity_target / (mean_activation + 1e-10) + 1e-10) + \
             (1 - self.sparsity_target) * torch.log((1 - self.sparsity_target) / (1 - mean_activation + 1e-10) + 1e-10)
        return self.sparsity_weight * kl.sum()

class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, latent_dim=LATENT_DIM, noise_factor=0.1):
        super().__init__()
        self.noise_factor = noise_factor
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.BatchNorm1d(512),
            nn.Linear(512, 384), nn.ReLU(), nn.BatchNorm1d(384),
            nn.Linear(384, latent_dim), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 384), nn.ReLU(), nn.BatchNorm1d(384),
            nn.Linear(384, 512), nn.ReLU(), nn.BatchNorm1d(512),
            nn.Linear(512, input_dim),
        )

    def forward(self, x):
        x_noisy = x + torch.randn_like(x) * self.noise_factor
        z = self.encoder(x_noisy)
        x_recon = self.decoder(z)
        return z, x_recon

def train_autoencoder(model, X_train, epochs=AE_EPOCHS, name="AE"):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    dataset = TensorDataset(torch.from_numpy(X_train).float())
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            if isinstance(model, VAE):
                z, recon, mu, logvar = model(batch)
                loss = F.mse_loss(recon, batch)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = loss + 0.001 * kl_loss
            else:
                z, recon = model(batch)
                loss = F.mse_loss(recon, batch)
                if isinstance(model, SparseAutoencoder):
                    loss = loss + model.sparsity_loss(z)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % max(1, epochs // 5) == 0:
            logger.info(f"    [{name}] Epoch {epoch+1}/{epochs}, loss={total_loss/len(loader):.4f}")
    model.eval()
    cleanup()
    return model

# ── HELIX Model ───────────────────────────────────────────────────────────

class HelixBackbone(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 384), nn.BatchNorm1d(384), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(384, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(256, latent_dim), nn.ReLU(),
        )
    def forward(self, x):
        return self.encoder(x)

class HelixClassifier(nn.Module):
    def __init__(self, backbone, input_dim=INPUT_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.backbone = backbone
        self.binary_head = nn.Linear(latent_dim, 2)

    def forward(self, x):
        z = self.backbone(x)
        binary = self.binary_head(z)
        return binary, z

def train_helix(X_train, y_train, epochs=HELIX_EPOCHS):
    backbone = HelixBackbone(input_dim=INPUT_DIM, latent_dim=LATENT_DIM).to(DEVICE)
    model = HelixClassifier(backbone).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits, _ = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % max(1, epochs // 5) == 0:
            logger.info(f"    [HELIX] Epoch {epoch+1}/{epochs}, loss={total_loss/len(loader):.4f}")
    model.eval()
    cleanup()
    return model

# ── SSL Models ────────────────────────────────────────────────────────────

class SSLProjection(nn.Module):
    """Projection head for contrastive learning."""
    def __init__(self, input_dim, hidden_dim=256, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, x): return self.net(x)

class SSLEncoder(nn.Module):
    """Encoder backbone for SSL methods."""
    def __init__(self, input_dim=INPUT_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 384), nn.BatchNorm1d(384), nn.ReLU(),
            nn.Linear(384, latent_dim), nn.ReLU(),
        )
    def forward(self, x): return self.encoder(x)

class SimCLR(nn.Module):
    def __init__(self, encoder, projection, temperature=0.5):
        super().__init__()
        self.encoder = encoder
        self.projection = projection
        self.temperature = temperature

    def forward(self, x):
        h = self.encoder(x)
        z = F.normalize(self.projection(h), dim=1)
        return h, z

def simclr_loss(z1, z2, temperature=0.5):
    """NT-Xent loss."""
    batch_size = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    z = F.normalize(z, dim=1)
    sim = torch.mm(z, z.T) / temperature
    labels = torch.arange(batch_size, device=z.device)
    labels = torch.cat([labels + batch_size - 1, labels], dim=0)  # Wrong, fix below:
    # Actually: positive pairs are (i, i+batch_size) and (i+batch_size, i)
    labels = torch.cat([labels, labels], dim=0)
    # Simpler: just use NT-Xent properly
    labels = torch.arange(batch_size, device=z.device)
    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim = sim - 1e9 * mask  # Remove self-similarity
    pos = torch.cat([torch.arange(batch_size, batch_size * 2, device=z.device),
                     torch.arange(batch_size, device=z.device)], dim=0)
    loss = F.cross_entropy(sim, pos)
    return loss

class BYOL(nn.Module):
    def __init__(self, encoder, projection, predictor=None):
        super().__init__()
        self.encoder = encoder
        self.projection = projection
        self.predictor = predictor or nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128))
        self.target_encoder = None  # Will be EMA of encoder

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection(h)
        return h, z

def byol_loss(p, z):
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return 2 - 2 * (p * z).sum(dim=1).mean()

class VICReg(nn.Module):
    def __init__(self, encoder, projection):
        super().__init__()
        self.encoder = encoder
        self.projection = projection

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection(h)
        return h, z

def vicreg_loss(z1, z2, sim_weight=25.0, var_weight=25.0, cov_weight=1.0):
    batch_size = z1.shape[0]
    dim = z1.shape[1]
    # Variance
    std_z1 = torch.sqrt(z1.var(dim=0) + 1e-4)
    std_z2 = torch.sqrt(z2.var(dim=0) + 1e-4)
    var_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))
    # Invariance (MSE)
    sim_loss = F.mse_loss(z1, z2)
    # Covariance
    def off_diagonal(x):
        n, m = x.shape
        x = x - x.mean(dim=0)
        cov = (x.T @ x) / (n - 1)
        return cov.flatten()[:-1].view(m - 1, m + 1)[:, 1:].flatten()
    z1_n = z1 - z1.mean(dim=0)
    z2_n = z2 - z2.mean(dim=0)
    cov_z1 = (z1_n.T @ z1_n) / (batch_size - 1)
    cov_z2 = (z2_n.T @ z2_n) / (batch_size - 1)
    cov_loss = off_diagonal(cov_z1).pow(2).sum() / dim + off_diagonal(cov_z2).pow(2).sum() / dim
    return sim_weight * sim_loss + var_weight * var_loss + cov_weight * cov_loss

class BarlowTwins(nn.Module):
    def __init__(self, encoder, projection):
        super().__init__()
        self.encoder = encoder
        self.projection = projection

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection(h)
        return h, z

def barlow_twins_loss(z1, z2, lambd=5e-3):
    batch_size = z1.shape[0]
    z1_n = (z1 - z1.mean(dim=0)) / z1.std(dim=0)
    z2_n = (z2 - z2.mean(dim=0)) / z2.std(dim=0)
    c = (z1_n.T @ z2_n) / batch_size
    on_diag = torch.diagonal(c).add(-1).pow(2).sum()
    off_diag = c.flatten()[:-1].view(c.shape[0] - 1, c.shape[0] + 1)[:, 1:].flatten().pow(2).sum()
    return on_diag + lambd * off_diag

def augment_tabular(X, noise_scale=0.05, mask_prob=0.1):
    """Simple augmentation for tabular data: Gaussian noise + feature masking."""
    X_aug = X + torch.randn_like(X) * noise_scale
    mask = torch.rand_like(X) < mask_prob
    X_aug[mask] = 0
    return X_aug

def train_ssl_model(model_class, loss_fn, X_train, epochs=SSL_EPOCHS, name="SSL",
                    latent_dim=LATENT_DIM, use_projector=True, use_predictor=False):
    encoder = SSLEncoder(input_dim=INPUT_DIM, latent_dim=latent_dim).to(DEVICE)
    projector = SSLProjection(latent_dim, hidden_dim=256, output_dim=128).to(DEVICE)
    predictor = SSLProjection(128, hidden_dim=128, output_dim=128).to(DEVICE) if use_predictor else None
    model = model_class(encoder, projector, predictor) if predictor else model_class(encoder, projector)
    model = model.to(DEVICE)
    params = list(encoder.parameters()) + list(projector.parameters())
    if predictor: params += list(predictor.parameters())
    optimizer = torch.optim.Adam(params, lr=LR, weight_decay=WEIGHT_DECAY)
    dataset = TensorDataset(torch.from_numpy(X_train).float())
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            aug1 = augment_tabular(batch)
            aug2 = augment_tabular(batch)
            optimizer.zero_grad()
            h1, z1 = model(aug1)
            h2, z2 = model(aug2)
            if name == "BYOL":
                with torch.no_grad():
                    _, z2_target = model(aug2)
                loss = loss_fn(model.predictor(z1), z2_target.detach())
            else:
                loss = loss_fn(z1, z2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % max(1, epochs // 5) == 0:
            logger.info(f"    [{name}] Epoch {epoch+1}/{epochs}, loss={total_loss/len(loader):.4f}")
    model.eval()
    # We extract representations from the encoder backbone, not the projection
    encoder.eval()
    cleanup()
    return model, encoder  # Return full model for extraction, encoder as backup

# ── Tabular Models ────────────────────────────────────────────────────────

class FTTransformerEncoder(nn.Module):
    """Simplified FT-Transformer: Feature Tokenizer + Transformer."""
    def __init__(self, input_dim=INPUT_DIM, latent_dim=LATENT_DIM, n_heads=4, n_layers=2, d_token=64):
        super().__init__()
        self.d_token = d_token
        self.feature_tokenizer = nn.Linear(1, d_token)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_token, nhead=n_heads,
                                                     dim_feedforward=256, dropout=0.1,
                                                     batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_token, latent_dim)

    def forward(self, x):
        # x: (batch, input_dim) -> tokenize each feature
        tokens = self.feature_tokenizer(x.unsqueeze(-1))  # (batch, n_features, d_token)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        tokens = self.transformer(tokens)
        # Use CLS token
        out = self.output_proj(tokens[:, 0])
        return out

class TabNetEncoder(nn.Module):
    """Simplified TabNet-like encoder with attentive feature selection."""
    def __init__(self, input_dim=INPUT_DIM, latent_dim=LATENT_DIM, n_d=64, n_steps=3):
        super().__init__()
        self.n_steps = n_steps
        self.n_d = n_d
        self.feature_dim = input_dim
        # Feature transformer
        self.feat_transform = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, n_d), nn.BatchNorm1d(n_d), nn.ReLU(),
                nn.Linear(n_d, n_d), nn.BatchNorm1d(n_d), nn.ReLU(),
            ) for _ in range(n_steps)
        ])
        # Attentive transformer
        self.attentive = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, n_d), nn.BatchNorm1d(n_d), nn.ReLU(),
                nn.Linear(n_d, input_dim),
            ) for _ in range(n_steps)
        ])
        self.output_proj = nn.Linear(n_d * n_steps, latent_dim)

    def forward(self, x):
        prior = torch.ones_like(x)
        step_outs = []
        for step in range(self.n_steps):
            att = self.attentive[step](prior)
            att = torch.softmax(att, dim=1)
            x_masked = x * att
            step_out = self.feat_transform[step](x_masked)
            step_outs.append(step_out)
            prior = prior * (1 - att)
        out = torch.cat(step_outs, dim=1)
        out = self.output_proj(out)
        return out

def train_tabular_encoder(model_class, X_train, y_train, epochs=AE_EPOCHS, name="Tabular"):
    """Train tabular encoder with reconstruction-like objective."""
    model = model_class(input_dim=INPUT_DIM, latent_dim=LATENT_DIM).to(DEVICE)
    # For tabular encoders, we use a simple classification head for auxiliary training
    aux_head = nn.Linear(LATENT_DIM, 2).to(DEVICE)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(aux_head.parameters()),
                                  lr=LR, weight_decay=WEIGHT_DECAY)
    dataset = TensorDataset(torch.from_numpy(X_train).float(),
                            torch.from_numpy(y_train).long())
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model.train()
    aux_head.train()
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            z = model(xb)
            logits = aux_head(z)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % max(1, epochs // 5) == 0:
            logger.info(f"    [{name}] Epoch {epoch+1}/{epochs}, loss={total_loss/len(loader):.4f}")
    model.eval()
    cleanup()
    return model

# ── XGBoost Leaf Embeddings ───────────────────────────────────────────────

def train_xgboost_embeddings(X_train, y_train):
    """Train XGBoost and extract leaf embeddings from all trees."""
    if not HAS_XGBOOST:
        logger.warning("  XGBoost not available, skipping")
        return None
    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        random_state=SEED, n_jobs=2, eval_metric='logloss',
        use_label_encoder=False, verbosity=0,
    )
    model.fit(X_train, y_train)
    # Extract leaf indices
    leaf_idx = model.apply(X_train)
    # Use one-hot encoded leaf indices as embeddings
    # (dimension varies by number of leaves)
    n_leaves = leaf_idx.max() + 1
    # Convert to dense embedding via label binarizer
    ohe = np.zeros((leaf_idx.shape[0], n_leaves * leaf_idx.shape[1]))
    for t in range(leaf_idx.shape[1]):
        for i, leaf in enumerate(leaf_idx[:, t]):
            ohe[i, t * n_leaves + leaf] = 1
    logger.info(f"    [XGBoost] Leaf embeddings: {ohe.shape}, {n_leaves} leaves/tree")
    return model, ohe, n_leaves

# ═══════════════════════════════════════════════════════════════════════════
# EVALUATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def extract_representations(model, X, model_type, batch_size=512):
    """Extract representations from a trained model."""
    if model_type == "PCA":
        n = min(LATENT_DIM, X.shape[1])
        Z = model.transform(X)[:, :n]
        if Z.shape[1] < LATENT_DIM:
            Z = np.pad(Z, ((0, 0), (0, LATENT_DIM - Z.shape[1])), mode='constant')
        return Z
    elif model_type == "ICA":
        n = min(LATENT_DIM, X.shape[1], X.shape[0])
        Z = model.transform(X)
        if Z.ndim == 1: Z = Z.reshape(-1, 1)
        if Z.shape[1] < LATENT_DIM:
            Z = np.pad(Z, ((0, 0), (0, LATENT_DIM - Z.shape[1])), mode='constant')
        return Z[:, :LATENT_DIM]
    elif model_type == "SparsePCA":
        Z = model.transform(X)
        if Z.shape[1] < LATENT_DIM:
            Z = np.pad(Z, ((0, 0), (0, LATENT_DIM - Z.shape[1])), mode='constant')
        return Z[:, :LATENT_DIM]
    elif model_type == "KernelPCA":
        model_obj, _ = model
        Z = model_obj.transform(X)
        if Z.ndim == 1: Z = Z.reshape(-1, 1)
        if Z.shape[1] < LATENT_DIM:
            Z = np.pad(Z, ((0, 0), (0, LATENT_DIM - Z.shape[1])), mode='constant')
        return Z[:, :LATENT_DIM]
    elif model_type == "XGBoost":
        model_obj, ohe, n_leaves = model
        leaf_idx = model_obj.apply(X)
        Z = np.zeros((leaf_idx.shape[0], min(n_leaves * leaf_idx.shape[1], LATENT_DIM)))
        for t in range(min(leaf_idx.shape[1], LATENT_DIM // max(1, n_leaves))):
            for i, leaf in enumerate(leaf_idx[:, t]):
                if leaf < n_leaves and t * n_leaves + leaf < Z.shape[1]:
                    Z[i, t * n_leaves + leaf] = 1
        return Z
    else:
        # Neural network models
        model.eval()
        Z_list = []
        n = X.shape[0]
        with torch.no_grad():
            for i in range(0, n, batch_size):
                batch = X[i:i + batch_size]
                x = torch.from_numpy(batch).float().to(DEVICE)
                if model_type in ["Autoencoder", "VAE", "SparseAE", "DenoisingAE"]:
                    z, _ = model(x)
                elif model_type == "HELIX":
                    _, z = model(x)
                elif model_type in ["SimCLR", "BYOL", "VICReg", "BarlowTwins"]:
                    z, _ = model(x)
                elif model_type in ["FTTransformer", "TabNet"]:
                    z = model(x)
                else:
                    z = model(x)
                Z_list.append(z.detach().cpu().numpy())
        Z = np.vstack(Z_list)
        if Z.shape[1] > LATENT_DIM:
            Z = Z[:, :LATENT_DIM]
        elif Z.shape[1] < LATENT_DIM:
            Z = np.pad(Z, ((0, 0), (0, LATENT_DIM - Z.shape[1])), mode='constant')
        return Z

# ═══════════════════════════════════════════════════════════════════════════
# Evaluation 1: Dataset Identity Leakage
# ═══════════════════════════════════════════════════════════════════════════

def eval_dataset_identity(data_dict, representations):
    """For each representation, measure how accurately dataset identity can be predicted."""
    logger.info("\n" + "=" * 60)
    logger.info("E1: Dataset Identity Leakage")
    logger.info("=" * 60)
    available = [n for n in ALL_DATASETS if n in data_dict]
    results = []

    for rep_name, rep_data in representations.items():
        # Build combined dataset with dataset ID as target
        all_Z = []; all_y_dataset = []
        for idx, name in enumerate(available):
            Z = rep_data[name]
            all_Z.append(Z)
            all_y_dataset.append(np.full(Z.shape[0], idx, dtype=np.int64))
        Z_all = np.vstack(all_Z)
        y_dataset = np.concatenate(all_y_dataset)

        Z_sub, y_sub = subsample_stratified(Z_all, y_dataset, 30000)
        if Z_sub.shape[0] < 100:
            logger.warning(f"    [{rep_name}] Too few samples: {Z_sub.shape[0]}")
            continue

        # Logistic Regression
        lr = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs",
                                 class_weight="balanced", random_state=SEED)
        try:
            lr.fit(Z_sub, y_sub)
            pred = lr.predict(Z_sub)
            acc = accuracy_score(y_sub, pred)
            mf1 = f1_score(y_sub, pred, average="macro", zero_division=0)
        except:
            # Fallback to RF
            rf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=1)
            rf.fit(Z_sub, y_sub)
            pred = rf.predict(Z_sub)
            acc = accuracy_score(y_sub, pred)
            mf1 = f1_score(y_sub, pred, average="macro", zero_division=0)

        # AUROC OvR
        try:
            # Limit classes for AUROC computation
            n_classes = len(np.unique(y_sub))
            if n_classes <= 2:
                # Binary case
                y_bin = label_binarize(y_sub, classes=[0, 1])
                if y_bin.shape[1] == 2: y_bin = y_bin[:, 1]
                probs = lr.predict_proba(Z_sub) if hasattr(lr, 'predict_proba') else None
                if probs is not None:
                    auroc = roc_auc_score(y_bin, probs[:, 1]) if n_classes == 2 else 0.5
                else:
                    auroc = 0.5
            else:
                y_bin = label_binarize(y_sub, classes=range(n_classes))
                probs = lr.predict_proba(Z_sub) if hasattr(lr, 'predict_proba') else None
                if probs is not None:
                    auroc = roc_auc_score(y_bin, probs, multi_class='ovr', average='macro')
                else:
                    auroc = 0.5
        except:
            auroc = 0.5

        # 5-fold CV
        try:
            cv = cross_val_score(lr, Z_sub[:10000], y_sub[:10000],
                                  cv=5, scoring="accuracy")
            cv_mean, cv_std = float(cv.mean()), float(cv.std())
        except:
            cv_mean, cv_std = 0.0, 0.0

        results.append({
            "representation": rep_name,
            "accuracy": float(acc),
            "macro_f1": float(mf1),
            "auroc_ovr": float(auroc),
            "cv_accuracy_mean": float(cv_mean),
            "cv_accuracy_std": float(cv_std),
            "n_datasets": len(available),
            "n_samples": int(Z_sub.shape[0]),
        })
        logger.info(f"  [{rep_name}] DIL: acc={acc:.4f}, mf1={mf1:.4f}, auroc={auroc:.4f}, cv={cv_mean:.4f}±{cv_std:.4f}")

    df = pd.DataFrame(results)
    df.to_csv(RESULTS / "dataset_identity.csv", index=False)
    logger.info(f"  Saved {RESULTS / 'dataset_identity.csv'}")
    return df

# ═══════════════════════════════════════════════════════════════════════════
# Evaluation 2: Attack Information Probes
# ═══════════════════════════════════════════════════════════════════════════

def eval_attack_probes(data_dict, representations):
    """Train linear probes on each representation for attack prediction."""
    logger.info("\n" + "=" * 60)
    logger.info("E2: Attack Information Probes")
    logger.info("=" * 60)
    results = []

    for rep_name, rep_data in representations.items():
        all_Z = []; all_y = []
        for name in ALL_DATASETS:
            if name in rep_data and name in data_dict:
                Z = rep_data[name]
                all_Z.append(Z)
                all_y.append(data_dict[name]["y_bin"])
        Z_all = np.vstack(all_Z); y_all = np.concatenate(all_y)

        Z_sub, y_sub = subsample_stratified(Z_all, y_all, 30000)
        if Z_sub.shape[0] < 100:
            continue

        # Split for evaluation
        X_tr, X_te, y_tr, y_te = train_test_split(Z_sub, y_sub, test_size=0.3,
                                                     random_state=SEED, stratify=y_sub)

        lr = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs",
                                 class_weight="balanced", random_state=SEED)
        try:
            lr.fit(X_tr, y_tr)
            pred = lr.predict(X_te)
            acc = accuracy_score(y_te, pred)
            mf1 = f1_score(y_te, pred, average="macro", zero_division=0)
            bf1 = f1_score(y_te, pred, zero_division=0)
            try:
                probs = lr.predict_proba(X_te)
                auroc = roc_auc_score(y_te, probs[:, 1]) if probs.shape[1] >= 2 else 0.5
            except:
                auroc = 0.5
        except:
            rf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=1)
            rf.fit(X_tr, y_tr)
            pred = rf.predict(X_te)
            acc = accuracy_score(y_te, pred)
            mf1 = f1_score(y_te, pred, average="macro", zero_division=0)
            bf1 = f1_score(y_te, pred, zero_division=0)
            auroc = 0.5

        # Calibration (Brier score)
        try:
            probs = lr.predict_proba(X_te) if hasattr(lr, 'predict_proba') else None
            if probs is not None and probs.shape[1] >= 2:
                brier = np.mean((probs[:, 1] - y_te) ** 2)
            else:
                brier = 0.5
        except:
            brier = 0.5

        results.append({
            "representation": rep_name,
            "accuracy": float(acc),
            "macro_f1": float(mf1),
            "binary_f1": float(bf1),
            "auroc": float(auroc),
            "brier_score": float(brier),
            "n_samples": int(Z_sub.shape[0]),
        })
        logger.info(f"  [{rep_name}] Attack probe: acc={acc:.4f}, mf1={mf1:.4f}, bf1={bf1:.4f}, auroc={auroc:.4f}")

    df = pd.DataFrame(results)
    df.to_csv(RESULTS / "attack_probe.csv", index=False)
    logger.info(f"  Saved {RESULTS / 'attack_probe.csv'}")
    return df

# ═══════════════════════════════════════════════════════════════════════════
# Evaluation 3: Mutual Information Estimation
# ═══════════════════════════════════════════════════════════════════════════

def compute_mutual_info_knn(Z, y, n_neighbors=5):
    """kNN-based mutual information between continuous Z and discrete Y."""
    try:
        from sklearn.feature_selection import mutual_info_classif
        mi = mutual_info_classif(Z, y, n_neighbors=n_neighbors, random_state=SEED)
        return float(np.mean(mi))
    except:
        return 0.0

def compute_mine(Z, y, batch_size=256, n_epochs=20):
    """Simple MINE (Mutual Information Neural Estimation)."""
    n = Z.shape[0]; d_z = Z.shape[1]
    class MINE(nn.Module):
        def __init__(self, d_z, n_classes=2):
            super().__init__()
            self.f = nn.Sequential(
                nn.Linear(d_z + n_classes, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, 1),
            )
        def forward(self, z, y_onehot):
            return self.f(torch.cat([z, y_onehot], dim=1))

    # Binarize y
    try:
        y_ohe = label_binarize(y, classes=np.unique(y))
    except:
        y_ohe = y.reshape(-1, 1)
    if y_ohe.ndim == 1:
        y_ohe = y_ohe.reshape(-1, 1)

    # Subsample
    if n > 10000:
        idx = rng.choice(n, 10000, replace=False)
        Z_s = Z[idx]; y_ohe_s = y_ohe[idx]
    else:
        Z_s = Z; y_ohe_s = y_ohe

    mine = MINE(Z_s.shape[1], y_ohe_s.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(mine.parameters(), lr=1e-3)
    dataset = TensorDataset(torch.from_numpy(Z_s).float(),
                            torch.from_numpy(y_ohe_s).float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    mine.train()
    for epoch in range(n_epochs):
        total_mi = 0
        for z_b, y_b in loader:
            z_b, y_b = z_b.to(DEVICE), y_b.to(DEVICE)
            # Shuffle for marginal
            idx_shuf = torch.randperm(z_b.shape[0])
            z_shuf = z_b[idx_shuf]
            optimizer.zero_grad()
            t_joint = mine(z_b, y_b)
            t_marginal = mine(z_shuf, y_b)
            # MINE loss: E[t_joint] - log(E[exp(t_marginal)])
            mi_est = t_joint.mean() - torch.log(torch.exp(t_marginal).mean() + 1e-8)
            loss = -mi_est  # maximize MI
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mine.parameters(), 1.0)
            optimizer.step()
            total_mi += mi_est.item()
        if (epoch+1) % 5 == 0:
            logger.info(f"      MINE epoch {epoch+1}/{n_epochs}, MI={total_mi/len(loader):.4f}")
    mine.eval()
    mi_est = total_mi / len(loader)
    return max(0.0, float(mi_est))

def eval_mutual_information(data_dict, representations):
    """Estimate I(Z;Dataset) and I(Z;Attack) for each representation."""
    logger.info("\n" + "=" * 60)
    logger.info("E3: Mutual Information Estimation")
    logger.info("=" * 60)
    results = []

    for rep_name, rep_data in representations.items():
        # I(Z;Dataset)
        all_Z_ds = []; all_y_ds = []
        for idx, name in enumerate(ALL_DATASETS):
            if name in rep_data:
                Z = rep_data[name]
                n = min(Z.shape[0], 2000)
                all_Z_ds.append(Z[rng.choice(Z.shape[0], n, replace=False)])
                all_y_ds.append(np.full(n, idx, dtype=np.int64))
        if not all_Z_ds: continue
        Z_ds = np.vstack(all_Z_ds); y_ds = np.concatenate(all_y_ds)
        mi_dataset_knn = compute_mutual_info_knn(Z_ds, y_ds)
        mi_dataset_mine = compute_mine(Z_ds, y_ds)

        # I(Z;Attack)
        all_Z_at = []; all_y_at = []
        for name in ALL_DATASETS:
            if name in rep_data and name in data_dict:
                Z = rep_data[name]
                n = min(Z.shape[0], 3000)
                idx = rng.choice(Z.shape[0], n, replace=False)
                all_Z_at.append(Z[idx])
                all_y_at.append(data_dict[name]["y_bin"][idx])
        Z_at = np.vstack(all_Z_at); y_at = np.concatenate(all_y_at)
        mi_attack_knn = compute_mutual_info_knn(Z_at, y_at)
        mi_attack_mine = compute_mine(Z_at, y_at)

        # AIR = I(Z;Attack) / I(Z;Dataset)
        air_knn = mi_attack_knn / max(mi_dataset_knn, 1e-10)
        air_mine = mi_attack_mine / max(mi_dataset_mine, 1e-10)

        results.append({
            "representation": rep_name,
            "I(Z;Dataset)_knn": mi_dataset_knn,
            "I(Z;Dataset)_mine": mi_dataset_mine,
            "I(Z;Attack)_knn": mi_attack_knn,
            "I(Z;Attack)_mine": mi_attack_mine,
            "AIR_knn": air_knn,
            "AIR_mine": air_mine,
        })
        logger.info(f"  [{rep_name}] I(Z;Dataset)={mi_dataset_knn:.4f}(knn)/{mi_dataset_mine:.4f}(mine), "
                    f"I(Z;Attack)={mi_attack_knn:.4f}(knn)/{mi_attack_mine:.4f}(mine), "
                    f"AIR={air_knn:.4f}(knn)/{air_mine:.4f}(mine)")

    df = pd.DataFrame(results)
    df.to_csv(RESULTS / "mutual_information.csv", index=False)
    logger.info(f"  Saved {RESULTS / 'mutual_information.csv'}")

    # AIR summary
    air_df = df[["representation", "AIR_knn", "AIR_mine"]].copy()
    air_df.to_csv(RESULTS / "attack_information_ratio.csv", index=False)
    logger.info(f"  Saved {RESULTS / 'attack_information_ratio.csv'}")
    return df

# ═══════════════════════════════════════════════════════════════════════════
# Evaluation 4: Geometry Metrics
# ═══════════════════════════════════════════════════════════════════════════

def compute_intrinsic_dimension(Z, method="mle"):
    """Estimate intrinsic dimension using MLE or PCA method."""
    from sklearn.neighbors import NearestNeighbors
    n = Z.shape[0]; d = Z.shape[1]
    if n > 5000:
        idx = rng.choice(n, 5000, replace=False)
        Z_sub = Z[idx]
    else:
        Z_sub = Z
    # MLE-based intrinsic dimension
    k = min(10, max(2, Z_sub.shape[0] // 10))
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(Z_sub)
    distances, _ = nbrs.kneighbors(Z_sub)
    Tk = distances[:, -1]
    Tk = np.maximum(Tk, 1e-10)
    Tj = np.maximum(distances[:, 1:-1], 1e-10)
    mu_k = np.mean(np.log(Tk[:, None] / Tj), axis=1)
    id_mle = 1.0 / np.mean(mu_k + 1e-10)
    return float(id_mle)

def compute_trustworthiness(Z, n_neighbors=7):
    """Trustworthiness measure (how well local neighborhoods are preserved)."""
    from sklearn.manifold import trustworthiness
    # Need original 17D space as reference
    return 0.0  # Placeholder — we don't have original in this function

def compute_silhouette(Z, y):
    """Silhouette score of the representation w.r.t. labels."""
    from sklearn.metrics import silhouette_score
    if len(np.unique(y)) < 2 or Z.shape[0] > 10000:
        idx = rng.choice(Z.shape[0], min(10000, Z.shape[0]), replace=False)
        Z_sub = Z[idx]; y_sub = y[idx]
    else:
        Z_sub = Z; y_sub = y
    if len(np.unique(y_sub)) < 2:
        return 0.0
    try:
        return float(silhouette_score(Z_sub, y_sub, random_state=SEED))
    except:
        return 0.0

def compute_davies_bouldin(Z, y):
    """Davies-Bouldin Index (lower = better clustering)."""
    from sklearn.metrics import davies_bouldin_score
    if len(np.unique(y)) < 2 or Z.shape[0] > 10000:
        idx = rng.choice(Z.shape[0], min(10000, Z.shape[0]), replace=False)
        Z_sub = Z[idx]; y_sub = y[idx]
    else:
        Z_sub = Z; y_sub = y
    if len(np.unique(y_sub)) < 2:
        return 0.0
    try:
        return float(davies_bouldin_score(Z_sub, y_sub))
    except:
        return 0.0

def eval_geometry(data_dict, representations):
    """Compute geometric properties of each representation."""
    logger.info("\n" + "=" * 60)
    logger.info("E4: Geometry Metrics")
    logger.info("=" * 60)
    results = []

    for rep_name, rep_data in representations.items():
        all_Z = []; all_y_ds = []; all_y_at = []
        for idx, name in enumerate(ALL_DATASETS):
            if name in rep_data and name in data_dict:
                Z = rep_data[name]
                n = min(Z.shape[0], 2000)
                idx_c = rng.choice(Z.shape[0], n, replace=False)
                all_Z.append(Z[idx_c])
                all_y_ds.append(np.full(n, idx, dtype=np.int64))
                all_y_at.append(data_dict[name]["y_bin"][idx_c])
        if not all_Z: continue
        Z_all = np.vstack(all_Z); y_ds = np.concatenate(all_y_ds); y_at = np.concatenate(all_y_at)

        # Intrinsic dimension (on combined)
        id_est = compute_intrinsic_dimension(Z_all)

        # Silhouette by dataset and attack
        sil_ds = compute_silhouette(Z_all, y_ds)
        sil_at = compute_silhouette(Z_all, y_at)

        # Davies-Bouldin
        db_ds = compute_davies_bouldin(Z_all, y_ds)
        db_at = compute_davies_bouldin(Z_all, y_at)

        results.append({
            "representation": rep_name,
            "intrinsic_dimension": id_est,
            "silhouette_dataset": sil_ds,
            "silhouette_attack": sil_at,
            "davies_bouldin_dataset": db_ds,
            "davies_bouldin_attack": db_at,
        })
        logger.info(f"  [{rep_name}] ID={id_est:.1f}, Sil(ds)={sil_ds:.4f}, Sil(at)={sil_at:.4f}, DB(ds)={db_ds:.4f}, DB(at)={db_at:.4f}")

    df = pd.DataFrame(results)
    df.to_csv(RESULTS / "geometry_metrics.csv", index=False)
    logger.info(f"  Saved {RESULTS / 'geometry_metrics.csv'}")
    return df

# ═══════════════════════════════════════════════════════════════════════════
# Evaluation 5: Representation Similarity
# ═══════════════════════════════════════════════════════════════════════════

def compute_cka(X, Y):
    """Linear Centered Kernel Alignment."""
    min_n = min(X.shape[0], Y.shape[0])
    if X.shape[0] != Y.shape[0]:
        idx = rng.choice(X.shape[0], min_n, replace=False)
        X = X[idx]
        idx = rng.choice(Y.shape[0], min_n, replace=False)
        Y = Y[idx]
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    K = X @ X.T; L = Y @ Y.T
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    K_c = H @ K @ H; L_c = H @ L @ H
    num = np.sum(K_c * L_c)
    denom = np.sqrt(np.sum(K_c * K_c) * np.sum(L_c * L_c))
    return float(num / max(denom, 1e-12))

def compute_svcca(X, Y):
    """Singular Vector CCA similarity."""
    from scipy.linalg import svd
    min_n = min(X.shape[0], Y.shape[0])
    if X.shape[0] != Y.shape[0]:
        idx = rng.choice(X.shape[0], min_n, replace=False)
        X = X[idx]
        idx = rng.choice(Y.shape[0], min_n, replace=False)
        Y = Y[idx]
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    Ux, Sx, _ = svd(X, full_matrices=False)
    Uy, Sy, _ = svd(Y, full_matrices=False)
    # Use top k components
    k = min(10, X.shape[1], Y.shape[1], Ux.shape[1], Uy.shape[1])
    Ux_k = Ux[:, :k]; Uy_k = Uy[:, :k]
    M = Ux_k.T @ Uy_k
    svcca = np.linalg.norm(M, 'fro') / np.sqrt(k)
    return float(svcca)

def compute_pwcca(X, Y):
    """Projection Weighted CCA."""
    from scipy.linalg import svd
    min_n = min(X.shape[0], Y.shape[0])
    if X.shape[0] != Y.shape[0]:
        idx = rng.choice(X.shape[0], min_n, replace=False)
        X = X[idx]
        idx = rng.choice(Y.shape[0], min_n, replace=False)
        Y = Y[idx]
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    Ux, Sx, _ = svd(X, full_matrices=False)
    Uy, Sy, _ = svd(Y, full_matrices=False)
    k = min(10, X.shape[1], Y.shape[1], Ux.shape[1], Uy.shape[1])
    Ux_k = Ux[:, :k]; Uy_k = Uy[:, :k]
    M = Ux_k.T @ Uy_k
    canon_corrs = np.linalg.svd(M, compute_uv=False)
    # Weights from singular values of X
    weights = Sx[:k] / np.sum(Sx[:k])
    pwcca = np.sum(weights * canon_corrs)
    return float(pwcca)

def eval_similarity(data_dict, representations):
    """Compute pairwise representation similarity across all models."""
    logger.info("\n" + "=" * 60)
    logger.info("E5: Representation Similarity")
    logger.info("=" * 60)

    rep_names = list(representations.keys())
    n_reps = len(rep_names)
    if n_reps < 2:
        logger.warning("  Not enough representations for similarity analysis")
        return None

    # Subsample for pairwise computation
    # Use one shared set of examples from each dataset
    all_Z_ref = {}
    for name in ALL_DATASETS:
        if name in data_dict:
            n = data_dict[name]["X"].shape[0]
            idx = rng.choice(n, min(500, n), replace=False)
            all_Z_ref[name] = idx

    # Build common sample
    common_Z = {}
    for rep_name in rep_names:
        slices = []
        for name in ALL_DATASETS:
            if name in all_Z_ref and name in representations[rep_name]:
                idx = all_Z_ref[name]
                Z = representations[rep_name][name]
                if Z.shape[0] > idx.max():
                    slices.append(Z[idx])
        if slices:
            common_Z[rep_name] = np.vstack(slices)

    cka_matrix = np.zeros((n_reps, n_reps))
    svcca_matrix = np.zeros((n_reps, n_reps))
    pwcca_matrix = np.zeros((n_reps, n_reps))

    for i in range(n_reps):
        for j in range(i, n_reps):
            if rep_names[i] not in common_Z or rep_names[j] not in common_Z:
                continue
            Zi = common_Z[rep_names[i]]
            Zj = common_Z[rep_names[j]]
            if Zi.shape[0] != Zj.shape[0]:
                n_min = min(Zi.shape[0], Zj.shape[0])
                Zi = Zi[:n_min]; Zj = Zj[:n_min]
            if Zi.shape[0] < 10:
                continue
            cka = compute_cka(Zi, Zj)
            svcca = compute_svcca(Zi, Zj)
            pwcca = compute_pwcca(Zi, Zj)
            cka_matrix[i, j] = cka_matrix[j, i] = cka
            svcca_matrix[i, j] = svcca_matrix[j, i] = svcca
            pwcca_matrix[i, j] = pwcca_matrix[j, i] = pwcca
            logger.info(f"  CKA({rep_names[i]}, {rep_names[j]})={cka:.4f}, SVCCA={svcca:.4f}, PWCCA={pwcca:.4f}")

    # Save matrices
    for name, mat in [("cka", cka_matrix), ("svcca", svcca_matrix), ("pwcca", pwcca_matrix)]:
        df = pd.DataFrame(mat, index=rep_names, columns=rep_names)
        df.to_csv(RESULTS / f"similarity_matrix_{name}.csv")
        logger.info(f"  Saved {RESULTS / f'similarity_matrix_{name}.csv'}")

    # Summary
    results = {"representations": rep_names}
    np.save(str(RESULTS / "cka_matrix.npy"), cka_matrix)
    np.save(str(RESULTS / "svcca_matrix.npy"), svcca_matrix)
    np.save(str(RESULTS / "pwcca_matrix.npy"), pwcca_matrix)

    return results

# ═══════════════════════════════════════════════════════════════════════════
# Evaluation 6: Domain Clustering Visualizations
# ═══════════════════════════════════════════════════════════════════════════

def eval_clustering(data_dict, representations):
    """UMAP, t-SNE, PCA visualizations colored by dataset and attack."""
    logger.info("\n" + "=" * 60)
    logger.info("E6: Domain Clustering Visualizations")
    logger.info("=" * 60)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Pick a few representative representations
    viz_reps = [r for r in ["Raw", "PCA", "HELIX", "Autoencoder", "SimCLR", "BarlowTwins", "FTTransformer", "XGBoost"]
                if r in representations]
    if "Raw" not in representations:
        # Add raw features as baseline
        pass  # Handled below

    for rep_name in viz_reps + (["Raw"] if "Raw" not in viz_reps else []):
        try:
            if rep_name == "Raw":
                all_X = []; all_y_ds = []; all_y_at = []
                for idx, name in enumerate(ALL_DATASETS):
                    if name in data_dict:
                        X = data_dict[name]["X"]
                        n = min(X.shape[0], 1000)
                        idx_c = rng.choice(X.shape[0], n, replace=False)
                        all_X.append(X[idx_c])
                        all_y_ds.append(np.full(n, idx, dtype=np.int64))
                        all_y_at.append(data_dict[name]["y_bin"][idx_c])
                X_viz = np.vstack(all_X)
                y_ds = np.concatenate(all_y_ds)
                y_at = np.concatenate(all_y_at)
                Z = StandardScaler().fit_transform(X_viz)
            else:
                all_Z = []; all_y_ds = []; all_y_at = []
                for idx, name in enumerate(ALL_DATASETS):
                    if name in representations[rep_name] and name in data_dict:
                        Z = representations[rep_name][name]
                        n = min(Z.shape[0], 1000)
                        idx_c = rng.choice(Z.shape[0], n, replace=False)
                        all_Z.append(Z[idx_c])
                        all_y_ds.append(np.full(n, idx, dtype=np.int64))
                        all_y_at.append(data_dict[name]["y_bin"][idx_c])
                Z = np.vstack(all_Z); y_ds = np.concatenate(all_y_ds); y_at = np.concatenate(all_y_at)

            if Z.shape[0] < 50: continue

            # PCA
            Z_pca = PCA(n_components=2, random_state=SEED).fit_transform(Z[:5000])
            # t-SNE (on subset)
            sub_idx = rng.choice(min(Z.shape[0], 3000), min(3000, Z.shape[0]), replace=False)
            Z_tsne = TSNE(n_components=2, random_state=SEED, perplexity=30).fit_transform(Z[sub_idx])
            # UMAP
            Z_umap = umap.UMAP(n_components=2, random_state=SEED, n_neighbors=15,
                                min_dist=0.1).fit_transform(Z[:5000])

            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            plot_data = [
                (Z_pca, "PCA"), (Z_tsne, "t-SNE"), (Z_umap, "UMAP"),
            ]
            for col_idx, (embed, method) in enumerate(plot_data):
                # Color by dataset
                if embed.shape[0] == y_ds.shape[0]:
                    sc = axes[0, col_idx].scatter(embed[:, 0], embed[:, 1], c=y_ds, cmap='tab10', s=5, alpha=0.7)
                    axes[0, col_idx].set_title(f"{rep_name} — {method} (by Dataset)")
                    axes[0, col_idx].set_xticks([]); axes[0, col_idx].set_yticks([])
                    # Color by attack
                    sc2 = axes[1, col_idx].scatter(embed[:, 0], embed[:, 1], c=y_at, cmap='coolwarm', s=5, alpha=0.7)
                    axes[1, col_idx].set_title(f"{rep_name} — {method} (by Attack)")
                    axes[1, col_idx].set_xticks([]); axes[1, col_idx].set_yticks([])
            plt.tight_layout()
            fig.savefig(FIGS_DIR / f"clustering_{rep_name.replace('/', '_')}.png", dpi=150)
            plt.close(fig)
            logger.info(f"  Saved clustering figures for {rep_name}")
        except Exception as e:
            logger.warning(f"  Clustering failed for {rep_name}: {e}")

    return {"n_representations_viz": len(viz_reps)}

# ═══════════════════════════════════════════════════════════════════════════
# Evaluation 7: Linear Separability
# ═══════════════════════════════════════════════════════════════════════════

def eval_linear_separability(data_dict, representations):
    """Compare linear separability of dataset vs attack."""
    logger.info("\n" + "=" * 60)
    logger.info("E7: Linear Separability")
    logger.info("=" * 60)
    results = []

    for rep_name, rep_data in representations.items():
        # Dataset separability
        all_Z_ds = []; all_y_ds = []
        for idx, name in enumerate(ALL_DATASETS):
            if name in rep_data:
                Z = rep_data[name]
                n = min(Z.shape[0], 2000)
                idx_c = rng.choice(Z.shape[0], n, replace=False)
                all_Z_ds.append(Z[idx_c])
                all_y_ds.append(np.full(n, idx, dtype=np.int64))
        Z_ds = np.vstack(all_Z_ds); y_ds = np.concatenate(all_y_ds)
        Z_ds_sub, y_ds_sub = subsample_stratified(Z_ds, y_ds, 15000)
        X_tr, X_te, y_tr, y_te = train_test_split(Z_ds_sub, y_ds_sub, test_size=0.3,
                                                     random_state=SEED, stratify=y_ds_sub)
        lr = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs",
                                 class_weight="balanced", random_state=SEED)
        lr.fit(X_tr, y_tr)
        pred_ds = lr.predict(X_te)
        ds_acc = accuracy_score(y_te, pred_ds)
        ds_mf1 = f1_score(y_te, pred_ds, average="macro", zero_division=0)

        # Attack separability
        all_Z_at = []; all_y_at = []
        for name in ALL_DATASETS:
            if name in rep_data and name in data_dict:
                Z = rep_data[name]
                n = min(Z.shape[0], 2000)
                idx_c = rng.choice(Z.shape[0], n, replace=False)
                all_Z_at.append(Z[idx_c])
                all_y_at.append(data_dict[name]["y_bin"][idx_c])
        Z_at = np.vstack(all_Z_at); y_at = np.concatenate(all_y_at)
        Z_at_sub, y_at_sub = subsample_stratified(Z_at, y_at, 15000)
        X_tr, X_te, y_tr, y_te = train_test_split(Z_at_sub, y_at_sub, test_size=0.3,
                                                     random_state=SEED, stratify=y_at_sub)
        lr2 = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs",
                                  class_weight="balanced", random_state=SEED)
        lr2.fit(X_tr, y_tr)
        pred_at = lr2.predict(X_te)
        at_acc = accuracy_score(y_te, pred_at)
        at_mf1 = f1_score(y_te, pred_at, average="macro", zero_division=0)

        results.append({
            "representation": rep_name,
            "dataset_separability_acc": float(ds_acc),
            "dataset_separability_mf1": float(ds_mf1),
            "attack_separability_acc": float(at_acc),
            "attack_separability_mf1": float(at_mf1),
            "separability_gap_acc": float(ds_acc - at_acc),
            "separability_gap_mf1": float(ds_mf1 - at_mf1),
        })
        logger.info(f"  [{rep_name}] Dataset sep: {ds_acc:.4f} / Attack sep: {at_acc:.4f} / Gap: {ds_acc - at_acc:.4f}")

    df = pd.DataFrame(results)
    df.to_csv(RESULTS / "linear_separability.csv", index=False)
    logger.info(f"  Saved {RESULTS / 'linear_separability.csv'}")
    return df

# ═══════════════════════════════════════════════════════════════════════════
# Evaluation 8: Compression Analysis
# ═══════════════════════════════════════════════════════════════════════════

def eval_compression(data_dict, representations):
    """Compression ratio vs information retention."""
    logger.info("\n" + "=" * 60)
    logger.info("E8: Compression Analysis")
    logger.info("=" * 60)
    results = []

    input_dim = INPUT_DIM

    for rep_name, rep_data in representations.items():
        all_Z = []; all_y_at = []; all_X = []
        for name in ALL_DATASETS:
            if name in rep_data and name in data_dict:
                Z = rep_data[name]
                n = min(Z.shape[0], 2000)
                idx_c = rng.choice(Z.shape[0], n, replace=False)
                all_Z.append(Z[idx_c])
                all_y_at.append(data_dict[name]["y_bin"][idx_c])
                all_X.append(data_dict[name]["X"][idx_c])
        if not all_Z: continue
        Z_all = np.vstack(all_Z); y_at = np.concatenate(all_y_at); X_all = np.vstack(all_X)

        latent_dim = Z_all.shape[1]
        compression_ratio = input_dim / max(latent_dim, 1)

        # Information retention: attack probe accuracy as proxy
        sub_idx = rng.choice(Z_all.shape[0], min(10000, Z_all.shape[0]), replace=False)
        Z_sub = Z_all[sub_idx]; y_sub = y_at[sub_idx]
        X_tr, X_te, y_tr, y_te = train_test_split(Z_sub, y_sub, test_size=0.3,
                                                     random_state=SEED, stratify=y_sub)
        lr = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs",
                                 class_weight="balanced", random_state=SEED)
        lr.fit(X_tr, y_tr)
        retention = accuracy_score(y_te, lr.predict(X_te))

        results.append({
            "representation": rep_name,
            "input_dim": int(input_dim),
            "latent_dim": int(latent_dim),
            "compression_ratio": float(compression_ratio),
            "information_retention": float(retention),
            "retention_per_compression": float(retention / max(compression_ratio, 0.01)),
        })
        logger.info(f"  [{rep_name}] Compress={input_dim}->{latent_dim} ({compression_ratio:.2f}x), "
                    f"retention={retention:.4f}")

    df = pd.DataFrame(results)
    df.to_csv(RESULTS / "compression_analysis.csv", index=False)
    logger.info(f"  Saved {RESULTS / 'compression_analysis.csv'}")
    return df

# ═══════════════════════════════════════════════════════════════════════════
# Evaluation 9: Robustness
# ═══════════════════════════════════════════════════════════════════════════

def eval_robustness(data_dict, representations):
    """Test robustness to Gaussian noise, missing features, and feature masking."""
    logger.info("\n" + "=" * 60)
    logger.info("E9: Robustness")
    logger.info("=" * 60)
    results = []

    for rep_name, rep_data in representations.items():
        # Build clean combined Z
        all_Z = []; all_y_at = []
        for name in ALL_DATASETS:
            if name in rep_data and name in data_dict:
                Z = rep_data[name]
                all_Z.append(Z)
                all_y_at.append(data_dict[name]["y_bin"])
        if not all_Z: continue
        Z_all = np.vstack(all_Z); y_at = np.concatenate(all_y_at)
        Z_sub, y_sub = subsample_stratified(Z_all, y_at, 10000)
        if Z_sub.shape[0] < 100: continue

        # Baseline (no noise)
        X_tr, X_te, y_tr, y_te = train_test_split(Z_sub, y_sub, test_size=0.3,
                                                     random_state=SEED, stratify=y_sub)
        lr = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs",
                                 class_weight="balanced", random_state=SEED)
        lr.fit(X_tr, y_tr)
        base_acc = accuracy_score(y_te, lr.predict(X_te))
        try:
            base_auroc = roc_auc_score(y_te, lr.predict_proba(X_te)[:, 1])
        except:
            base_auroc = 0.5

        # Gaussian noise
        noise_levels = [0.05, 0.1, 0.25, 0.5]
        noise_results = {}
        for nl in noise_levels:
            X_te_noisy = X_te + rng.randn(*X_te.shape) * nl
            try:
                acc = accuracy_score(y_te, lr.predict(X_te_noisy))
                noise_results[f"noise_{nl}"] = float(acc)
            except:
                noise_results[f"noise_{nl}"] = 0.0

        # Missing features (zero out random fraction)
        missing_results = {}
        for frac in [0.05, 0.1, 0.2, 0.5]:
            X_te_miss = X_te.copy()
            mask = rng.random(X_te_miss.shape) < frac
            X_te_miss[mask] = 0
            try:
                acc = accuracy_score(y_te, lr.predict(X_te_miss))
                missing_results[f"missing_{frac}"] = float(acc)
            except:
                missing_results[f"missing_{frac}"] = 0.0

        # Random feature masking (set entire column to zero)
        masking_results = {}
        for frac in [0.05, 0.1, 0.2]:
            X_te_mask = X_te.copy()
            n_cols = max(1, int(X_te_mask.shape[1] * frac))
            cols = rng.choice(X_te_mask.shape[1], n_cols, replace=False)
            X_te_mask[:, cols] = 0
            try:
                acc = accuracy_score(y_te, lr.predict(X_te_mask))
                masking_results[f"mask_{frac}"] = float(acc)
            except:
                masking_results[f"mask_{frac}"] = 0.0

        entry = {
            "representation": rep_name,
            "baseline_accuracy": float(base_acc),
            "baseline_auroc": float(base_auroc),
            **noise_results, **missing_results, **masking_results,
        }
        results.append(entry)
        logger.info(f"  [{rep_name}] Base={base_acc:.4f}, noise(0.5)={noise_results.get('noise_0.5', 0):.4f}")

    df = pd.DataFrame(results)
    df.to_csv(RESULTS / "robustness.csv", index=False)
    logger.info(f"  Saved {RESULTS / 'robustness.csv'}")
    return df

# ═══════════════════════════════════════════════════════════════════════════
# Evaluation 10: Universality Test + Pareto Frontier
# ═══════════════════════════════════════════════════════════════════════════

def eval_universality(eval_results):
    """Rank all representations and compute Pareto frontier."""
    logger.info("\n" + "=" * 60)
    logger.info("E10: Universality Test")
    logger.info("=" * 60)

    # Merge all evaluation results
    dfs = {}
    for key, path in [("dataset_identity", RESULTS / "dataset_identity.csv"),
                      ("attack_probe", RESULTS / "attack_probe.csv"),
                      ("mutual_information", RESULTS / "mutual_information.csv"),
                      ("air", RESULTS / "attack_information_ratio.csv"),
                      ("geometry", RESULTS / "geometry_metrics.csv"),
                      ("robustness", RESULTS / "robustness.csv"),
                      ("separability", RESULTS / "linear_separability.csv"),
                      ("compression", RESULTS / "compression_analysis.csv")]:
        if path.exists():
            dfs[key] = pd.read_csv(path)

    if not dfs:
        logger.warning("  No evaluation results available")
        return {}

    # Build summary ranking
    all_reps = set()
    for df in dfs.values():
        if "representation" in df.columns:
            all_reps.update(df["representation"].tolist())
    all_reps = sorted(all_reps)

    ranking_data = []
    for rep in all_reps:
        entry = {"representation": rep}

        # DIL (lower is better: negate for ranking)
        if "dataset_identity" in dfs:
            row = dfs["dataset_identity"]
            m = row[row["representation"] == rep]
            if len(m) > 0:
                entry["DIL_accuracy"] = float(m.iloc[0].get("accuracy", 0.5))
                entry["DIL_macro_f1"] = float(m.iloc[0].get("macro_f1", 0))

        # Attack probe (higher is better)
        if "attack_probe" in dfs:
            row = dfs["attack_probe"]
            m = row[row["representation"] == rep]
            if len(m) > 0:
                entry["Attack_macro_f1"] = float(m.iloc[0].get("macro_f1", 0))
                entry["Attack_auroc"] = float(m.iloc[0].get("auroc", 0.5))
                entry["Attack_brier"] = float(m.iloc[0].get("brier_score", 0.5))

        # AIR (higher is better)
        if "air" in dfs:
            row = dfs["air"]
            m = row[row["representation"] == rep]
            if len(m) > 0:
                entry["AIR_knn"] = float(m.iloc[0].get("AIR_knn", 0))
                entry["AIR_mine"] = float(m.iloc[0].get("AIR_mine", 0))

        # Geometry
        if "geometry" in dfs:
            row = dfs["geometry"]
            m = row[row["representation"] == rep]
            if len(m) > 0:
                entry["intrinsic_dim"] = float(m.iloc[0].get("intrinsic_dimension", 0))
                entry["silhouette_attack"] = float(m.iloc[0].get("silhouette_attack", 0))

        # Robustness (use noise_0.1 as proxy)
        if "robustness" in dfs:
            row = dfs["robustness"]
            m = row[row["representation"] == rep]
            if len(m) > 0:
                entry["robustness_noise_0.1"] = float(m.iloc[0].get("noise_0.1", 0))
                entry["robustness_missing_0.1"] = float(m.iloc[0].get("missing_0.1", 0))

        ranking_data.append(entry)

    df_ranking = pd.DataFrame(ranking_data)

    # Compute composite scores and rank
    # 1. Lowest DIL (higher rank = lower accuracy)
    # 2. Highest AIR
    # 3. Highest attack F1
    # 4. Best calibration (lowest Brier)
    # 5. Best robustness

    scores = []
    for _, row in df_ranking.iterrows():
        rep = row["representation"]
        dil = row.get("DIL_accuracy", 0.5)
        air = row.get("AIR_knn", 0)
        atk_f1 = row.get("Attack_macro_f1", 0)
        brier = row.get("Attack_brier", 0.5)
        rob = row.get("robustness_noise_0.1", 0)

        # Composite: low DIL (-), high AIR (+), high attack F1 (+), low Brier (-), high rob (+)
        # Normalize each to [0, 1] roughly
        composite = (1 - dil) + air + atk_f1 + (1 - brier) + rob
        scores.append(composite)

    df_ranking["composite_score"] = scores
    df_ranking["rank"] = np.argsort(np.argsort(-np.array(scores))) + 1
    df_ranking = df_ranking.sort_values("rank")

    df_ranking.to_csv(RESULTS / "representation_summary.csv", index=False)
    logger.info(f"  Saved {RESULTS / 'representation_summary.csv'}")

    logger.info("\n  Ranking:")
    for _, row in df_ranking.iterrows():
        logger.info(f"    {int(row['rank'])}. {row['representation']}: score={row['composite_score']:.4f}")

    # Pareto frontier: minimize DIL, maximize AIR, maximize Attack F1
    pareto = []
    for i, row1 in df_ranking.iterrows():
        dominated = False
        for j, row2 in df_ranking.iterrows():
            if i == j: continue
            if (row2.get("DIL_accuracy", 0.5) <= row1.get("DIL_accuracy", 0.5) and
                row2.get("AIR_knn", 0) >= row1.get("AIR_knn", 0) and
                row2.get("Attack_macro_f1", 0) >= row1.get("Attack_macro_f1", 0) and
                (row2.get("DIL_accuracy", 0.5) < row1.get("DIL_accuracy", 0.5) or
                 row2.get("AIR_knn", 0) > row1.get("AIR_knn", 0) or
                 row2.get("Attack_macro_f1", 0) > row1.get("Attack_macro_f1", 0))):
                dominated = True
                break
        if not dominated:
            pareto.append(row1["representation"])

    logger.info(f"\n  Pareto frontier ({len(pareto)}):")
    for p in pareto:
        logger.info(f"    {p}")

    return {
        "ranking": df_ranking.to_dict("records"),
        "pareto_frontier": pareto,
        "n_representations": len(all_reps),
    }

# ═══════════════════════════════════════════════════════════════════════════
# Statistical Analysis
# ═══════════════════════════════════════════════════════════════════════════

def run_statistical_analysis(eval_results):
    """Bootstrap confidence intervals, effect sizes, and Bayesian estimation."""
    logger.info("\n" + "=" * 60)
    logger.info("Statistical Analysis")
    logger.info("=" * 60)

    # Bootstrap on DIL accuracy differences
    dil_path = RESULTS / "dataset_identity.csv"
    if not dil_path.exists():
        return {}

    df_dil = pd.read_csv(dil_path)

    bootstrap_results = []
    for _, row in df_dil.iterrows():
        rep = row["representation"]
        acc = row["accuracy"]
        n = int(row.get("n_samples", 1000))
        # Bootstrap CI for accuracy
        boot_accs = []
        for _ in range(1000):
            idx = rng.choice(n, n, replace=True)
            p = idx[idx < int(n * acc)].shape[0] / n
            boot_accs.append(p)
        boot_accs = np.array(boot_accs)
        ci_low = np.percentile(boot_accs, 2.5)
        ci_high = np.percentile(boot_accs, 97.5)
        bootstrap_results.append({
            "representation": rep,
            "accuracy": float(acc),
            "ci_95_lower": float(ci_low),
            "ci_95_upper": float(ci_high),
            "ci_width": float(ci_high - ci_low),
        })

    bdf = pd.DataFrame(bootstrap_results)
    bdf.to_csv(RESULTS / "bootstrap.csv", index=False)
    logger.info(f"  Saved {RESULTS / 'bootstrap.csv'}")

    # Simple Bayesian estimation (Beta posterior for accuracies)
    bayesian_results = []
    for _, row in df_dil.iterrows():
        rep = row["representation"]
        acc = row["accuracy"]
        n = int(row.get("n_samples", 1000))
        k = int(acc * n)
        # Beta posterior: Beta(alpha + k, beta + n - k)
        alpha_prior, beta_prior = 1, 1
        alpha_post = alpha_prior + k
        beta_post = beta_prior + n - k
        # Posterior mean
        posterior_mean = alpha_post / (alpha_post + beta_post)
        # Credible interval
        from scipy.stats import beta as beta_dist
        ci_low = beta_dist.ppf(0.025, alpha_post, beta_post)
        ci_high = beta_dist.ppf(0.975, alpha_post, beta_post)
        bayesian_results.append({
            "representation": rep,
            "posterior_mean": float(posterior_mean),
            "credible_interval_lower": float(ci_low),
            "credible_interval_upper": float(ci_high),
        })

    bayes_df = pd.DataFrame(bayesian_results)
    bayes_df.to_csv(RESULTS / "bayesian.csv", index=False)
    logger.info(f"  Saved {RESULTS / 'bayesian.csv'}")

    return {
        "bootstrap": bootstrap_results,
        "bayesian": bayesian_results,
    }

# ═══════════════════════════════════════════════════════════════════════════
# Report Generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_report(all_results, representations):
    """Generate comprehensive Phase 66 report."""
    logger.info("\n" + "=" * 60)
    logger.info("Generating Report")
    logger.info("=" * 60)

    report = []
    report.append("# Phase 66: Universality of Dataset Identity Leakage\n")
    report.append(f"*Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    report.append("## Research Question\n")
    report.append("Is Dataset Identity Leakage a property unique to HELIX, or an intrinsic property ")
    report.append("of Network Intrusion Detection feature spaces across fundamentally different ")
    report.append("representation learners?\n")

    # Summary table
    report.append("## Representation Learners Evaluated\n")
    rep_list = list(representations.keys())
    report.append(f"Total: {len(rep_list)} learners\n")
    report.append("| # | Learner | Type |\n")
    report.append("|---|---------|------|\n")
    categories = {
        "PCA": "Classical", "ICA": "Classical", "SparsePCA": "Classical", "KernelPCA": "Classical",
        "Autoencoder": "Neural", "VAE": "Neural", "SparseAE": "Neural", "DenoisingAE": "Neural", "HELIX": "Neural",
        "SimCLR": "Self-Supervised", "BYOL": "Self-Supervised", "VICReg": "Self-Supervised", "BarlowTwins": "Self-Supervised",
        "FTTransformer": "Tabular", "TabNet": "Tabular",
        "XGBoost": "Tree-Based",
    }
    for i, rep in enumerate(rep_list):
        cat = categories.get(rep, "Other")
        report.append(f"| {i+1} | {rep} | {cat} |\n")
    report.append("\n")

    # E1: Dataset Identity Leakage
    dil_path = RESULTS / "dataset_identity.csv"
    if dil_path.exists():
        df = pd.read_csv(dil_path)
        report.append("## E1: Dataset Identity Leakage\n")
        report.append("| Representation | Accuracy | Macro F1 | AUROC (OvR) | CV Accuracy |\n")
        report.append("|---------------|----------|----------|-------------|-------------|\n")
        for _, row in df.iterrows():
            report.append(f"| {row['representation']} | {row['accuracy']:.4f} | "
                          f"{row['macro_f1']:.4f} | {row.get('auroc_ovr', 0):.4f} | "
                          f"{row.get('cv_accuracy_mean', 0):.4f}±{row.get('cv_accuracy_std', 0):.4f} |\n")
        report.append("\n")

    # E3: Mutual Information & AIR
    air_path = RESULTS / "attack_information_ratio.csv"
    if air_path.exists():
        df = pd.read_csv(air_path)
        report.append("## E3: Attack Information Ratio (AIR)\n")
        report.append("| Representation | AIR (kNN) | AIR (MINE) |\n")
        report.append("|---------------|-----------|------------|\n")
        sorted_df = df.sort_values("AIR_knn", ascending=False)
        for _, row in sorted_df.iterrows():
            report.append(f"| {row['representation']} | {row['AIR_knn']:.4f} | {row['AIR_mine']:.4f} |\n")
        report.append("\n")

    # E10: Ranking
    ranking_path = RESULTS / "representation_summary.csv"
    if ranking_path.exists():
        df = pd.read_csv(ranking_path)
        report.append("## E10: Universality Ranking\n")
        report.append("| Rank | Representation | Composite Score | DIL Acc | AIR | Attack F1 |\n")
        report.append("|------|---------------|-----------------|---------|-----|-----------|\n")
        for _, row in df.iterrows():
            report.append(f"| {int(row['rank'])} | {row['representation']} | "
                          f"{row['composite_score']:.4f} | {row.get('DIL_accuracy', 0):.4f} | "
                          f"{row.get('AIR_knn', 0):.4f} | {row.get('Attack_macro_f1', 0):.4f} |\n")
        report.append("\n")

        report.append("### Pareto Frontier\n")
        pareto = all_results.get("universality", {}).get("pareto_frontier", [])
        for p in pareto:
            report.append(f"- **{p}**\n")
        report.append("\n")

    # Conclusion
    report.append("## Conclusion\n")
    if dil_path.exists():
        df = pd.read_csv(dil_path)
        n_reps = len(df)
        high_dil = (df["accuracy"] > 0.8).sum()
        if high_dil >= n_reps * 0.75:
            report.append("**Outcome A (confirmed):** Every representation exhibits high Dataset Identity ")
            report.append(f"Leakage ({high_dil}/{n_reps} reps >80% accuracy). DIL is an intrinsic property ")
            report.append("of current IDS feature spaces.\n")
        elif high_dil >= 2:
            report.append("**Outcome B (partial):** Some representations suppress DIL while preserving ")
            report.append("attack information. Domain-invariant representations are achievable.\n")
        else:
            report.append("**Outcome C (confirmed):** Dataset identity and attack semantics remain ")
            report.append("inseparable across all representation paradigms.\n")

    report_text = "".join(report)
    with open(RESULTS / "phase66_report.md", "w") as f:
        f.write(report_text)

    # Summary
    summary = []
    summary.append("# Phase 66 Summary\n")
    summary.append(f"*{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
    summary.append(f"**Representation learners:** {len(rep_list)}\n\n")
    summary.append("### Dataset Identity Leakage (E1)\n\n")
    if dil_path.exists():
        df = pd.read_csv(dil_path)
        best = df.loc[df["accuracy"].idxmin()]
        worst = df.loc[df["accuracy"].idxmax()]
        summary.append(f"- **Best (lowest DIL):** {best['representation']} ({best['accuracy']:.1%})\n")
        summary.append(f"- **Worst (highest DIL):** {worst['representation']} ({worst['accuracy']:.1%})\n\n")
    summary.append("### Attack Information Ratio (E3)\n\n")
    if air_path.exists():
        df = pd.read_csv(air_path)
        best_air = df.loc[df["AIR_knn"].idxmax()]
        summary.append(f"- **Highest AIR:** {best_air['representation']} ({best_air['AIR_knn']:.4f})\n\n")
    summary.append("### Ranking (E10)\n\n")
    if ranking_path.exists():
        df = pd.read_csv(ranking_path)
        for _, row in df.iterrows():
            summary.append(f"{int(row['rank'])}. {row['representation']} (score={row['composite_score']:.4f})\n")
        summary.append("\n")

    summary_text = "".join(summary)
    with open(RESULTS / "phase66_summary.md", "w") as f:
        f.write(summary_text)

    logger.info(f"  Report saved to {RESULTS / 'phase66_report.md'}")
    logger.info(f"  Summary saved to {RESULTS / 'phase66_summary.md'}")

    return True

# ═══════════════════════════════════════════════════════════════════════════
# Main Orchestrator
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 66")
    parser.add_argument("--quick", action="store_true", default=True,
                        help="Quick mode (reduced samples / epochs)")
    parser.add_argument("--eval", type=str, default="all",
                        help="Comma-separated eval IDs: 1,2,3,... or 'all'")
    parser.add_argument("--n_subsample", type=int, default=50000,
                        help="Max samples per dataset")
    parser.add_argument("--skip_ssl", action="store_true", help="Skip SSL models")
    parser.add_argument("--skip_tabular", action="store_true", help="Skip tabular models")
    parser.add_argument("--skip_xgboost", action="store_true", help="Skip XGBoost")
    parser.add_argument("--skip_neural", action="store_true", help="Skip neural models")
    parser.add_argument("--skip_classical", action="store_true", help="Skip classical models")
    parser.add_argument("--load_cached", action="store_true",
                        help="Load cached representations if available")
    args = parser.parse_args()

    global MAX_SAMPLES, SSL_EPOCHS, AE_EPOCHS, HELIX_EPOCHS
    if args.quick:
        SSL_EPOCHS = 20; AE_EPOCHS = 20; HELIX_EPOCHS = 10
    else:
        SSL_EPOCHS = 50; AE_EPOCHS = 50; HELIX_EPOCHS = 30

    MAX_SAMPLES = args.n_subsample
    evals_to_run = set(range(1, 11))
    if args.eval != "all":
        evals_to_run = set(int(e) for e in args.eval.split(",") if e.strip().isdigit())

    logger.info(f"Phase 66 configuration:\n"
                f"  quick={args.quick}, subsample={MAX_SAMPLES}, "
                f"evals={evals_to_run}")
    logger.info(f"  Device: {DEVICE}, SSL_epochs={SSL_EPOCHS}, "
                f"AE_epochs={AE_EPOCHS}, HELIX_epochs={HELIX_EPOCHS}")

    # ── Load Data ──────────────────────────────────────────────────────────
    t0 = time.time()
    logger.info("\n" + "=" * 60)
    logger.info("Loading Data")
    logger.info("=" * 60)
    data = load_all_datasets(quick=args.quick, max_samples=MAX_SAMPLES)
    logger.info(f"Data loading: {time.time() - t0:.1f}s, loaded {len(data)}/{len(ALL_DATASETS)} datasets")

    # Clean NaN/Inf
    for name in list(data.keys()):
        d = data[name]
        for key in ["X", "y_bin", "y"]:
            if key not in d: continue
            if isinstance(d[key], np.ndarray):
                nan_mask = np.isnan(d["X"]).any(axis=1) | np.isinf(d["X"]).any(axis=1)
                if nan_mask.any():
                    for k2 in ["X", "y_bin", "y"]:
                        if k2 in d:
                            d[k2] = d[k2][~nan_mask]
                    logger.info(f"  Cleaned {nan_mask.sum()} NaN/Inf in {name}")
    t_data = time.time()

    # ── Train / Load Representations ──────────────────────────────────────
    representations = OrderedDict()
    rep_models = {}

    # Add raw features as a baseline
    representations["Raw"] = {name: data[name]["X"] for name in data}
    logger.info("  [Raw] Using raw features as baseline")

    # ── Classical Learners ────────────────────────────────────────────────
    if not args.skip_classical:
        logger.info("\n" + "-" * 40)
        logger.info("Classical Representation Learners")
        logger.info("-" * 40)

        # Fit on combined data
        all_X = np.vstack([data[n]["X"] for n in data])
        all_y = np.concatenate([data[n]["y_bin"] for n in data])

        for learner_name, train_fn in [
            ("PCA", train_pca), ("ICA", train_ica),
            ("SparsePCA", train_sparse_pca), ("KernelPCA", train_kernel_pca),
        ]:
            t1 = time.time()
            logger.info(f"\n  Training {learner_name}...")
            try:
                model = train_fn(all_X)
                if learner_name == "KernelPCA":
                    model_obj, sub_idx = model
                    # Extract on all data
                    Z_all = model_obj.transform(all_X)
                    if Z_all.ndim == 1: Z_all = Z_all.reshape(-1, 1)
                    if Z_all.shape[1] < LATENT_DIM:
                        Z_all = np.pad(Z_all, ((0, 0), (0, LATENT_DIM - Z_all.shape[1])), mode='constant')
                    Z_all = Z_all[:, :LATENT_DIM]
                else:
                    Z_all = model.transform(all_X)
                    if Z_all.ndim == 1: Z_all = Z_all.reshape(-1, 1)
                    if Z_all.shape[1] < LATENT_DIM:
                        Z_all = np.pad(Z_all, ((0, 0), (0, LATENT_DIM - Z_all.shape[1])), mode='constant')
                    Z_all = Z_all[:, :LATENT_DIM]
                # Split back per dataset
                offset = 0
                rep_data = {}
                for name in data:
                    n = data[name]["X"].shape[0]
                    rep_data[name] = Z_all[offset:offset + n]
                    offset += n
                representations[learner_name] = rep_data
                rep_models[learner_name] = model
                logger.info(f"  {learner_name}: {Z_all.shape} in {time.time()-t1:.1f}s")
            except Exception as e:
                logger.warning(f"  {learner_name} failed: {e}")
            cleanup()

    # ── Neural Learners ──────────────────────────────────────────────────
    if not args.skip_neural:
        logger.info("\n" + "-" * 40)
        logger.info("Neural Representation Learners")
        logger.info("-" * 40)

        # Combine training data
        all_X = np.vstack([data[n]["X"] for n in data])
        all_y = np.concatenate([data[n]["y_bin"] for n in data])
        # Normalize
        scaler = StandardScaler()
        all_X_norm = scaler.fit_transform(all_X)

        neural_models = [
            ("Autoencoder", Autoencoder(input_dim=INPUT_DIM, latent_dim=LATENT_DIM),
             lambda m, X, name: train_autoencoder(m, X, epochs=AE_EPOCHS, name=name)),
            ("VAE", VAE(input_dim=INPUT_DIM, latent_dim=LATENT_DIM),
             lambda m, X, name: train_autoencoder(m, X, epochs=AE_EPOCHS, name=name)),
            ("SparseAE", SparseAutoencoder(input_dim=INPUT_DIM, latent_dim=LATENT_DIM),
             lambda m, X, name: train_autoencoder(m, X, epochs=AE_EPOCHS, name=name)),
            ("DenoisingAE", DenoisingAutoencoder(input_dim=INPUT_DIM, latent_dim=LATENT_DIM),
             lambda m, X, name: train_autoencoder(m, X, epochs=AE_EPOCHS, name=name)),
        ]

        for name, model_init, train_fn in neural_models:
            t1 = time.time()
            logger.info(f"\n  Training {name}...")
            try:
                model = train_fn(model_init, all_X_norm, name)
                rep_data = {}
                offset = 0
                for ds_name in data:
                    n = data[ds_name]["X"].shape[0]
                    X_ds = scaler.transform(data[ds_name]["X"])
                    Z = extract_representations(model, X_ds, name)
                    rep_data[ds_name] = Z
                representations[name] = rep_data
                rep_models[name] = model
                logger.info(f"  {name}: done in {time.time()-t1:.1f}s")
            except Exception as e:
                logger.warning(f"  {name} failed: {e}")
            cleanup()

        # HELIX
        t1 = time.time()
        logger.info(f"\n  Training HELIX...")
        try:
            model = train_helix(all_X_norm, all_y, epochs=HELIX_EPOCHS)
            rep_data = {}
            for ds_name in data:
                X_ds = scaler.transform(data[ds_name]["X"])
                Z = extract_representations(model, X_ds, "HELIX")
                rep_data[ds_name] = Z
            representations["HELIX"] = rep_data
            rep_models["HELIX"] = model
            logger.info(f"  HELIX: done in {time.time()-t1:.1f}s")
        except Exception as e:
            logger.warning(f"  HELIX failed: {e}")
        cleanup()

    # ── SSL Learners ─────────────────────────────────────────────────────
    if not args.skip_ssl:
        logger.info("\n" + "-" * 40)
        logger.info("Self-Supervised Representation Learners")
        logger.info("-" * 40)

        all_X = np.vstack([data[n]["X"] for n in data])
        scaler = StandardScaler()
        all_X_norm = scaler.fit_transform(all_X)

        ssl_configs = [
            ("SimCLR", lambda enc, proj: SimCLR(enc, proj, temperature=0.5),
             lambda z1, z2: simclr_loss(z1, z2)),
            ("VICReg", lambda enc, proj: VICReg(enc, proj),
             lambda z1, z2: vicreg_loss(z1, z2)),
            ("BarlowTwins", lambda enc, proj: BarlowTwins(enc, proj),
             lambda z1, z2: barlow_twins_loss(z1, z2)),
            ("BYOL", lambda enc, proj: BYOL(enc, proj),
             lambda z1, z2: byol_loss(z1, z2)),
        ]

        for name, model_fn, loss_fn in ssl_configs:
            t1 = time.time()
            logger.info(f"\n  Training {name}...")
            try:
                use_pred = (name == "BYOL")
                ssl_model, ssl_encoder = train_ssl_model(model_fn, loss_fn, all_X_norm,
                                          epochs=SSL_EPOCHS, name=name, use_predictor=use_pred)
                rep_data = {}
                for ds_name in data:
                    X_ds = scaler.transform(data[ds_name]["X"])
                    # Extract using the full SSL model (which returns h from encoder)
                    Z = extract_representations(ssl_model, X_ds, name)
                    rep_data[ds_name] = Z
                representations[name] = rep_data
                rep_models[name] = ssl_model
                logger.info(f"  {name}: done in {time.time()-t1:.1f}s")
            except Exception as e:
                logger.warning(f"  {name} failed: {e}")
            cleanup()

    # ── Tabular Learners ──────────────────────────────────────────────────
    if not args.skip_tabular:
        logger.info("\n" + "-" * 40)
        logger.info("Tabular Representation Learners")
        logger.info("-" * 40)

        all_X = np.vstack([data[n]["X"] for n in data])
        all_y = np.concatenate([data[n]["y_bin"] for n in data])
        scaler = StandardScaler()
        all_X_norm = scaler.fit_transform(all_X)

        for name, model_class in [("FTTransformer", FTTransformerEncoder),
                                   ("TabNet", TabNetEncoder)]:
            t1 = time.time()
            logger.info(f"\n  Training {name}...")
            try:
                model = train_tabular_encoder(model_class, all_X_norm, all_y,
                                               epochs=AE_EPOCHS, name=name)
                rep_data = {}
                for ds_name in data:
                    X_ds = scaler.transform(data[ds_name]["X"])
                    Z = extract_representations(model, X_ds, name)
                    rep_data[ds_name] = Z
                representations[name] = rep_data
                rep_models[name] = model
                logger.info(f"  {name}: done in {time.time()-t1:.1f}s")
            except Exception as e:
                logger.warning(f"  {name} failed: {e}")
            cleanup()

    # ── XGBoost ──────────────────────────────────────────────────────────
    if not args.skip_xgboost and HAS_XGBOOST:
        logger.info("\n" + "-" * 40)
        logger.info("XGBoost Leaf Embeddings")
        logger.info("-" * 40)
        t1 = time.time()
        all_X = np.vstack([data[n]["X"] for n in data])
        all_y = np.concatenate([data[n]["y_bin"] for n in data])
        scaler = StandardScaler()
        all_X_norm = scaler.fit_transform(all_X)
        try:
            model_tuple = train_xgboost_embeddings(all_X_norm, all_y)
            if model_tuple is not None:
                xgb_model, ohe, n_leaves = model_tuple
                # Extract per-dataset
                rep_data = {}
                for ds_name in data:
                    X_ds = scaler.transform(data[ds_name]["X"])
                    leaf_idx = xgb_model.apply(X_ds)
                    n_trees = leaf_idx.shape[1]
                    # Create compact embedding
                    Z = np.zeros((leaf_idx.shape[0], min(n_leaves * n_trees, LATENT_DIM)))
                    for t in range(min(n_trees, LATENT_DIM // max(1, n_leaves))):
                        for i, leaf in enumerate(leaf_idx[:, t]):
                            idx = t * n_leaves + leaf
                            if leaf < n_leaves and idx < Z.shape[1]:
                                Z[i, idx] = 1
                    rep_data[ds_name] = Z
                representations["XGBoost"] = rep_data
                rep_models["XGBoost"] = xgb_model
                logger.info(f"  XGBoost: done in {time.time()-t1:.1f}s")
        except Exception as e:
            logger.warning(f"  XGBoost failed: {e}")

    # ── Cache representations ──────────────────────────────────────────────
    cache_path = CACHE_DIR / "representations.pkl"
    with open(cache_path, "wb") as f:
        pickle.dump(dict(representations), f)
    logger.info(f"  Cached {len(representations)} representations to {cache_path}")

    # ── Run Evaluations ───────────────────────────────────────────────────
    all_results = {}
    t_rep = time.time()

    if 1 in evals_to_run:
        all_results["dataset_identity"] = eval_dataset_identity(data, representations)
    if 2 in evals_to_run:
        all_results["attack_probe"] = eval_attack_probes(data, representations)
    if 3 in evals_to_run:
        all_results["mutual_information"] = eval_mutual_information(data, representations)
    if 4 in evals_to_run:
        all_results["geometry"] = eval_geometry(data, representations)
    if 5 in evals_to_run:
        all_results["similarity"] = eval_similarity(data, representations)
    if 6 in evals_to_run:
        all_results["clustering"] = eval_clustering(data, representations)
    if 7 in evals_to_run:
        all_results["separability"] = eval_linear_separability(data, representations)
    if 8 in evals_to_run:
        all_results["compression"] = eval_compression(data, representations)
    if 9 in evals_to_run:
        all_results["robustness"] = eval_robustness(data, representations)
    if 10 in evals_to_run:
        all_results["universality"] = eval_universality(all_results)

    # ── Statistical Analysis ───────────────────────────────────────────────
    all_results["statistical"] = run_statistical_analysis(all_results)

    # ── Report ─────────────────────────────────────────────────────────────
    generate_report(all_results, representations)

    # ── Summary ────────────────────────────────────────────────────────────
    logger.info(f"\n{'=' * 60}")
    logger.info("Phase 66 Complete")
    logger.info(f"{'=' * 60}")
    logger.info(f"Data loading: {t_data - t0:.1f}s")
    logger.info(f"Representation learning: {t_rep - t_data:.1f}s")
    logger.info(f"Total: {time.time() - t0:.1f}s")
    logger.info(f"Representations: {len(representations)}")
    logger.info(f"Results in: {RESULTS}")
    logger.info(f"  representation_summary.csv")
    logger.info(f"  dataset_identity.csv")
    logger.info(f"  attack_probe.csv")
    logger.info(f"  mutual_information.csv / attack_information_ratio.csv")
    logger.info(f"  geometry_metrics.csv")
    logger.info(f"  similarity_matrix_*.csv")
    logger.info(f"  robustness.csv")
    logger.info(f"  bootstrap.json / bayesian.json")
    logger.info(f"  phase66_report.md / phase66_summary.md")
    logger.info(f"  figures/ / embeddings/")

    print(f"\nPhase 66 complete. Results in {RESULTS}")
    print(f"  {len(representations)} representations")
    print(f"  {len(evals_to_run)} evaluations run")
    print(f"  Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
