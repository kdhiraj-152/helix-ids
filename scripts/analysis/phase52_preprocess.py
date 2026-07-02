#!/usr/bin/env python3
"""Pre-process the remaining IDS datasets for Phase 52 experiments."""
import sys, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "phase52_cache"
DATA_DIR.mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_all():
    """Load and cache all 6 datasets as numpy arrays."""
    from helix_ids.data.feature_harmonization import FEATURE_ORDER
    from helix_ids.data.multi_dataset_loader import MultiDatasetLoader
    from sklearn.model_selection import train_test_split

    loader = MultiDatasetLoader(project_root=PROJECT_ROOT)

    # Already processed datasets
    existing = {
        "nsl_kdd": ("X_train_nsl_kdd", "y_train_nsl_kdd", "X_test_nsl_kdd", "y_test_nsl_kdd"),
        "unsw_nb15": ("X_train_unsw_nb15", "y_train_unsw_nb15", "X_test_unsw_nb15", "y_test_unsw_nb15"),
    }
    base = PROJECT_ROOT / "data" / "processed" / "multi_dataset_v1"

    for ds_name, (x_tr_f, y_tr_f, x_te_f, y_te_f) in existing.items():
        X_train = np.load(base / f"{x_tr_f}.npy")
        X_test = np.load(base / f"{x_te_f}.npy")
        y_train = np.load(base / f"{y_tr_f}.npy", allow_pickle=True)
        y_test = np.load(base / f"{y_te_f}.npy", allow_pickle=True)

        for arr_name, arr in [("X_train", X_train), ("X_test", X_test),
                               ("y_train", y_train), ("y_test", y_test)]:
            finite = np.isfinite(arr).all(axis=1) if arr.ndim > 1 else np.isfinite(arr)
            if arr_name.startswith("y"):
                globals()[arr_name] = arr[finite]
            else:
                globals()[arr_name] = arr[finite]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_test = scaler.transform(X_test).astype(np.float32)

        np.save(DATA_DIR / f"{ds_name}_X_train.npy", X_train)
        np.save(DATA_DIR / f"{ds_name}_y_train.npy", y_train)
        np.save(DATA_DIR / f"{ds_name}_X_test.npy", X_test)
        np.save(DATA_DIR / f"{ds_name}_y_test.npy", y_test)
        print(f"{ds_name}: train={X_train.shape[0]}, test={X_test.shape[0]}, classes={np.unique(y_train)}")

    # CICIDS2018 - use test data (subsampled)
    print("\nProcessing CICIDS2018...")
    X_all = np.load(base / "X_test_cicids.npy", mmap_mode='r')
    y_path = base / "y_test_cicids.npy"
    if y_path.exists():
        y_all = np.load(y_path, allow_pickle=True)
    else:
        # Need to load through pipeline for labels
        df = loader.load_cicids(year=2018)
        if df is not None:
            harm = loader.harmonize_cicids(df)
            if "label" in harm.columns and all(c in harm.columns for c in FEATURE_ORDER):
                y_all = harm["label"].to_numpy(dtype=np.int64)
            else:
                print("  SKIP: CICIDS2018 label extraction failed")
                return
        else:
            print("  SKIP: CICIDS2018 not available")
            return

    # Subsample
    rng = np.random.default_rng(42)
    idx = rng.choice(len(y_all), min(30000, len(y_all)), replace=False)
    X_all_sub = X_all[idx].copy()
    y_all_sub = y_all[idx]
    scaler = StandardScaler()
    X_all_sub = scaler.fit_transform(X_all_sub).astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(
        X_all_sub, y_all_sub, test_size=0.3, random_state=42, stratify=y_all_sub)
    np.save(DATA_DIR / "cicids2018_X_train.npy", X_train)
    np.save(DATA_DIR / "cicids2018_y_train.npy", y_train)
    np.save(DATA_DIR / "cicids2018_X_test.npy", X_test)
    np.save(DATA_DIR / "cicids2018_y_test.npy", y_test)
    print(f"cicids2018: train={len(X_train)}, test={len(X_test)}, classes={np.unique(y_train)}")

    # Remaining datasets through pipeline
    remaining = ["cicids2017", "bot_iot", "ton_iot"]
    for ds_name in remaining:
        print(f"\nProcessing {ds_name}...")
        t0 = time.time()
        try:
            load_fn = getattr(loader, f"load_{ds_name}")
            harmonize_fn = getattr(loader, f"harmonize_{ds_name}")
        except AttributeError:
            print(f"  SKIP: no loader for {ds_name}")
            continue

        df = load_fn()
        if df is None or len(df) == 0:
            print(f"  SKIP: empty dataset")
            continue

        print(f"  Raw: {df.shape} ({time.time()-t0:.1f}s)")

        harm = harmonize_fn(df)
        del df  # free memory
        print(f"  Harmonized: {harm.shape} ({time.time()-t0:.1f}s)")

        if "label" not in harm.columns:
            print(f"  SKIP: no label column")
            continue

        missing = [c for c in FEATURE_ORDER if c not in harm.columns]
        if missing:
            print(f"  SKIP: missing features: {missing}")
            continue

        X = harm[FEATURE_ORDER].to_numpy(dtype=np.float32)
        y = harm["label"].to_numpy(dtype=np.int64)
        del harm

        finite = np.isfinite(X).all(axis=1) & (y >= 0)
        X = X[finite]
        y = y[finite]

        if len(X) == 0:
            print(f"  SKIP: empty after NaN removal")
            continue

        # Subsample if needed
        max_samples = 30000
        if len(X) > max_samples:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X), max_samples, replace=False)
            X = X[idx]
            y = y[idx]

        scaler = StandardScaler()
        X = scaler.fit_transform(X).astype(np.float32)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y)

        np.save(DATA_DIR / f"{ds_name}_X_train.npy", X_train)
        np.save(DATA_DIR / f"{ds_name}_y_train.npy", y_train)
        np.save(DATA_DIR / f"{ds_name}_X_test.npy", X_test)
        np.save(DATA_DIR / f"{ds_name}_y_test.npy", y_test)
        print(f"  Saved: train={len(X_train)}, test={len(X_test)}, classes={np.unique(y_train)}")
        print(f"  Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    preprocess_all()
