"""TON-IoT Learnability Contract Validation — proper data handling."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from helix_ids.contracts.schema_contract import (
    CANONICAL_FEATURE_ORDER,
    CANONICAL_INPUT_DIM,
    SCHEMA_HASH,
    SCHEMA_VERSION,
    compute_schema_hash,
)
from helix_ids.data.feature_harmonization import (
    FEATURE_ORDER,
    create_ton_iot_mapping,
    harmonize_features,
)
from helix_ids.contracts.attack_taxonomy import TONIOT_TO_7CLASS

results = {
    "test": "TON-IoT Learnability Contract Validation",
    "schema_version": SCHEMA_VERSION,
    "canonical_schema_hash": SCHEMA_HASH,
    "expected_feature_order": list(FEATURE_ORDER),
    "expected_feature_count": CANONICAL_INPUT_DIM,
}

# ============================================================================
# Step 1: Load raw TON-IoT exactly as pipeline does
# ============================================================================

raw_path = PROJECT_ROOT / "data" / "ton_iot" / "raw" / "train.csv"
print(f"Loading TON-IoT from {raw_path}")
raw = pd.read_csv(raw_path, low_memory=False)

# Only rename type→label if type exists (TON-IoT has both binary 'label' 
# and multi-class 'type' columns; we use 'type' as the multi-class label).
# Mirrors _clean_ton_iot_frame logic.
df = raw.copy()
if "type" in df.columns:
    if "label" in df.columns:
        df = df.drop(columns=["label"])
    df.rename(columns={"type": "label"}, inplace=True)
elif "label" not in df.columns and "attack_type" in df.columns:
    df.rename(columns={"attack_type": "label"}, inplace=True)

results["raw"] = {
    "n_rows": len(raw),
    "n_columns": len(raw.columns),
    "columns": list(raw.columns),
    "has_label_column": "label" in df.columns,
    "has_type_column": "type" in df.columns,
}
print(f"  Raw shape: {raw.shape}")
print(f"  Has 'label': {'label' in df.columns}, Has 'type': {'type' in df.columns}")

# ============================================================================
# Step 2: Dedup exactly as pipeline does
# ============================================================================

before_dedup = len(df)
df = df.drop_duplicates()
after_dedup = len(df)
results["cleaning"] = {
    "rows_before_dedup": before_dedup,
    "rows_after_dedup": after_dedup,
    "rows_removed": before_dedup - after_dedup,
    "removal_pct": round((before_dedup - after_dedup) / before_dedup * 100, 2),
}
print(f"  After dedup: {df.shape} (removed {before_dedup - after_dedup})")

# ============================================================================
# Step 3: Harmonize
# ============================================================================

mapping = create_ton_iot_mapping()
print(f"  Calling harmonize_features with {mapping.dataset_name} mapping...")
harmonized = harmonize_features(df, mapping)
print(f"  After harmonization: {harmonized.shape}")
print(f"  Harmonized columns: {list(harmonized.columns)}")

# Map labels using TONIOT_TO_7CLASS (mirrors harmonize_ton_iot logic)
normalized_labels = harmonized["label"].astype(str).str.strip().str.lower()
mapped_labels = normalized_labels.map({k.lower(): v for k, v in TONIOT_TO_7CLASS.items()})
unresolved = normalized_labels[mapped_labels.isna()].unique().tolist()
results["label_mapping"] = {
    "unresolved_labels": unresolved,
    "n_unresolved": len(unresolved),
}
if unresolved:
    print(f"  WARNING: Unresolved labels: {unresolved}")
harmonized["label"] = mapped_labels.astype(int)

# ============================================================================
# Step 4: Feature validation
# ============================================================================

feature_cols = [c for c in harmonized.columns if c != "label"]
n_features = len(feature_cols)
feature_order_ok = feature_cols == FEATURE_ORDER
computed_hash = compute_schema_hash(feature_order=feature_cols, input_dim=n_features)
schema_hash_ok = computed_hash == SCHEMA_HASH

results["features"] = {
    "n_features": n_features,
    "expected_n_features": CANONICAL_INPUT_DIM,
    "count_ok": n_features == CANONICAL_INPUT_DIM,
    "feature_order_ok": feature_order_ok,
    "feature_order": feature_cols,
    "expected_feature_order": FEATURE_ORDER,
    "computed_schema_hash": computed_hash,
    "expected_schema_hash": SCHEMA_HASH,
    "schema_hash_ok": schema_hash_ok,
}
print(f"  Features: {n_features} (expected {CANONICAL_INPUT_DIM}), order ok: {feature_order_ok}, hash ok: {schema_hash_ok}")

# ============================================================================
# Step 5: Label integrity
# ============================================================================

labels = harmonized["label"]
unique_labels = sorted(labels.unique())
label_range_ok = bool(labels.between(0, 6).all())

results["labels"] = {
    "dtype": str(labels.dtype),
    "n_unique": int(labels.nunique()),
    "values": [int(v) for v in unique_labels],
    "range_ok": label_range_ok,
    "min": int(labels.min()),
    "max": int(labels.max()),
    "distribution": {int(k): int(v) for k, v in labels.value_counts().sort_index().items()},
}
print(f"  Labels: {unique_labels}, range ok: {label_range_ok}")

# ============================================================================
# Step 6: NaN / Inf / Constant / Impossible checks
# ============================================================================

num_cols = harmonized.select_dtypes(include=[np.number]).columns

# NaN
nan_cols = [str(c) for c in num_cols if harmonized[c].isna().any()]
has_nan = len(nan_cols) > 0

# Inf (check on feature columns only)
X = harmonized[num_cols].to_numpy(dtype=np.float64, copy=False)
inf_mask = np.isinf(X)
inf_col_indices = np.where(inf_mask.any(axis=0))[0]
inf_features = [str(num_cols[i]) for i in inf_col_indices]
has_inf = len(inf_features) > 0

# Constant
constant_features = []
for c in harmonized.columns:
    if c != "label":
        if harmonized[c].nunique() <= 1:
            constant_features.append(str(c))
has_constant = len(constant_features) > 0

# Impossible values
impossible = {}
if pd.api.types.is_numeric_dtype(harmonized["src_bytes"]):
    neg_src = int((harmonized["src_bytes"] < 0).sum())
    if neg_src > 0:
        impossible["negative_src_bytes"] = neg_src
if pd.api.types.is_numeric_dtype(harmonized["dst_bytes"]):
    neg_dst = int((harmonized["dst_bytes"] < 0).sum())
    if neg_dst > 0:
        impossible["negative_dst_bytes"] = neg_dst
if pd.api.types.is_numeric_dtype(harmonized["duration"]):
    neg_dur = int((harmonized["duration"] < 0).sum())
    if neg_dur > 0:
        impossible["negative_duration"] = neg_dur
has_impossible = len(impossible) > 0

results["integrity"] = {
    "nan_features": nan_cols,
    "has_nan": has_nan,
    "inf_features": inf_features,
    "has_inf": has_inf,
    "constant_features": constant_features,
    "has_constant": has_constant,
    "impossible_values": impossible,
    "has_impossible_values": has_impossible,
}
print(f"  NaN: {has_nan}, Inf: {has_inf}, Constant: {has_constant}, Impossible: {has_impossible}")
if nan_cols:
    print(f"    NaN in: {nan_cols}")
if inf_features:
    print(f"    Inf in: {inf_features}")
if constant_features:
    print(f"    Constant: {constant_features}")
if impossible:
    print(f"    Impossible: {impossible}")

# ============================================================================
# Step 7: Distribution diagnostics
# ============================================================================

distributions = {}
for c in feature_cols:
    vals = harmonized[c]
    if pd.api.types.is_numeric_dtype(vals):
        series = vals.dropna()
        if len(series) > 0:
            distributions[c] = {
                "dtype": str(vals.dtype),
                "min": float(series.min()),
                "max": float(series.max()),
                "mean": float(series.mean()),
                "std": float(series.std()),
                "n_unique": int(vals.nunique()),
                "n_zero": int((vals == 0).sum()),
            }
results["distributions"] = distributions

# ============================================================================
# Verdict
# ============================================================================

failures = []
if not schema_hash_ok:
    failures.append(f"Schema hash mismatch: {computed_hash} != {SCHEMA_HASH}")
if n_features != CANONICAL_INPUT_DIM:
    failures.append(f"Feature count: {n_features} != {CANONICAL_INPUT_DIM}")
if not feature_order_ok:
    failures.append("Feature order mismatch")
if has_nan:
    failures.append(f"NaN in features: {nan_cols}")
if has_inf:
    failures.append(f"Inf in features: {inf_features}")
if has_constant:
    failures.append(f"Constant features: {constant_features}")
if has_impossible:
    failures.append(f"Impossible values: {impossible}")
if not label_range_ok:
    failures.append(f"Labels out of range [0,6]: {unique_labels}")

results["verdict"] = "FAIL" if failures else "PASS"
results["failures"] = failures
print(f"\n=== VERDICT: {results['verdict']} ===")
if failures:
    for f in failures:
        print(f"  FAIL: {f}")

# ============================================================================
# Generate report
# ============================================================================

output_path = PROJECT_ROOT / "docs" / "phase25c" / "TON_IOT_CONTRACT_REPORT.md"
output_content = f"""# TON-IoT Learnability Contract Report

**Generated:** 2026-06-21 IST
**Pipeline:** `load_ton_iot()` → `harmonize_ton_iot()` → contract validation

## Verdict

**{results['verdict']}**

"""

if failures:
    output_content += "### Failures\n\n"
    for f in failures:
        output_content += f"- {f}\n"
    output_content += "\n"

output_content += f"""## Summary

| Check | Status |
|-------|--------|
| Feature count = {CANONICAL_INPUT_DIM} | {'PASS' if n_features == CANONICAL_INPUT_DIM else 'FAIL'} |
| Feature order matches canonical | {'PASS' if feature_order_ok else 'FAIL'} |
| Schema hash matches canonical | {'PASS' if schema_hash_ok else 'FAIL'} |
| No NaN in features | {'PASS' if not has_nan else 'FAIL'} |
| No Inf in features | {'PASS' if not has_inf else 'FAIL'} |
| No constant features | {'PASS' if not has_constant else 'FAIL'} |
| No impossible values | {'PASS' if not has_impossible else 'FAIL'} |
| Label range valid [0–6] | {'PASS' if label_range_ok else 'FAIL'} |
| All labels resolved | {'PASS' if len(unresolved) == 0 else 'FAIL'} |

## Data Details

| Metric | Value |
|--------|-------|
| Raw rows loaded | {results['raw']['n_rows']:,} |
| Raw columns | {results['raw']['n_columns']} |
| After dedup | {after_dedup:,} |
| After harmonization | {len(harmonized):,} |
| Feature count | {n_features} |
| Label classes | {unique_labels} |
| Class distribution | {json.dumps({str(k): v for k, v in results['labels']['distribution'].items()})} |

## Schema

| Property | Value |
|----------|-------|
| Schema version | {SCHEMA_VERSION} |
| Canonical hash | {SCHEMA_HASH} |
| Computed hash | {computed_hash} |
| Hash match | {'PASS' if schema_hash_ok else 'FAIL'} |

## Feature List

```text
{json.dumps(feature_cols, indent=2)}
```

## Integrity Details

| Check | Result |
|-------|--------|
| NaN features | {nan_cols if nan_cols else 'None'} |
| Inf features | {inf_features if inf_features else 'None'} |
| Constant features | {constant_features if constant_features else 'None'} |
| Impossible values | {json.dumps(impossible) if impossible else 'None'} |

## Raw Label Distribution (Before Mapping)

The TON-IoT raw CSV contains both a `label` and a `type` column. The pipeline uses the `label` column directly (no rename needed).

## Feature Distributions

```text
{json.dumps({k: {sk: (round(sv, 4) if isinstance(sv, float) else sv) for sk, sv in v.items()} for k, v in distributions.items()}, indent=2)[:2000]}
```
"""

with open(output_path, "w") as f:
    f.write(output_content)

print(f"\nReport written to {output_path}")
