"""Check TON-IoT raw label values."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

raw_path = PROJECT_ROOT / "data" / "ton_iot" / "raw" / "train.csv"
raw = pd.read_csv(raw_path, low_memory=False)

print("=== LABEL COLUMN VALUES ===")
print(f"Unique 'label' values: {raw['label'].unique()}")
print(f"Value counts (label):")
print(raw['label'].value_counts())

print(f"\n=== TYPE COLUMN VALUES ===")
print(f"Unique 'type' values: {raw['type'].unique()}")
print(f"Value counts (type):")
print(raw['type'].value_counts())

print(f"\n=== CROSS-TAB: label vs type ===")
ct = pd.crosstab(raw['label'].astype(str).str.strip().str.lower(), 
                 raw['type'].astype(str).str.strip().str.lower(), 
                 margins=True)
for idx in ct.index:
    row = ct.loc[idx]
    non_zero = {str(k): int(v) for k, v in row.items() if v > 0}
    print(f"  label='{idx}': type_values={non_zero}")
