#!/usr/bin/env python3
"""Strict NSL-KDD processing entrypoint for unified multi-dataset contract runs."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from helix_ids.data.feature_harmonization import FEATURE_ORDER  # noqa: E402
from helix_ids.data.learnability_contract import compute_schema_hash  # noqa: E402
from helix_ids.data.multi_dataset_loader import MultiDatasetLoader  # noqa: E402


def main() -> None:
    loader = MultiDatasetLoader(project_root=PROJECT_ROOT)
    raw = loader.load_nslkdd()
    harmonized = loader.harmonize_nslkdd(raw)

    out_dir = PROJECT_ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "nsl-kdd_cleaned.csv"
    harmonized.to_csv(out_path, index=False)

    schema_hash = compute_schema_hash(
        feature_columns=list(FEATURE_ORDER),
        transformations=["split_then_nan_to_num"],
    )

    print("NSL_KDD_PROCESS_DONE")
    print(f"rows={len(harmonized)}")
    print(f"output={out_path}")
    print(f"schema_hash={schema_hash}")
    print(f"feature_order={','.join(FEATURE_ORDER)}")


if __name__ == "__main__":
    main()
