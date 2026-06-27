#!/usr/bin/env python3
"""
Phase 25: Download new IDS datasets that HelixIDS-Full has NEVER seen.

Downloads TON-IoT (network), CIC-DDoS2019, and BoT-IoT from Hugging Face
and saves them as CSVs under data/<dataset_name>/ for harmonization.

Usage:
    python scripts/data/download_new_datasets.py [--datasets ton_iot,cic_ddos2019,bot_iot]
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


def download_ton_iot() -> Path:
    """Download TON_IoT_network dataset from Hugging Face (Zeek-style flow features)."""
    out_dir = DATA_DIR / "ton_iot" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.csv"
    test_path = out_dir / "test.csv"

    if train_path.exists() and test_path.exists():
        logger.info("TON_IoT already downloaded, skipping")
        return out_dir

    from datasets import load_dataset

    logger.info("Downloading TON_IoT_network (train split)...")
    ds = load_dataset("codymlewis/TON_IoT_network", split="train", streaming=True)

    rows = []
    for i, row in enumerate(ds):
        rows.append(dict(row))
        if (i + 1) % 50000 == 0:
            logger.info(f"  ... {i+1} rows collected")

    df = pd.DataFrame(rows)
    df.to_csv(train_path, index=False)
    logger.info(f"TON_IoT train saved: {len(df)} rows -> {train_path}")

    logger.info("Downloading TON_IoT_network (test split)...")
    ds_test = load_dataset("codymlewis/TON_IoT_network", split="test", streaming=True)
    rows_test = []
    for i, row in enumerate(ds_test):
        rows_test.append(dict(row))
        if (i + 1) % 10000 == 0:
            logger.info(f"  ... {i+1} rows collected")

    df_test = pd.DataFrame(rows_test)
    df_test.to_csv(test_path, index=False)
    logger.info(f"TON_IoT test saved: {len(df_test)} rows -> {test_path}")

    return out_dir


def download_cic_ddos2019() -> Path:
    """Download CIC-DDoS2019 dataset from Hugging Face (CICFlowMeter-style features)."""
    out_dir = DATA_DIR / "cic_ddos2019" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.csv"

    if train_path.exists():
        logger.info("CIC-DDoS2019 already downloaded, skipping")
        return out_dir

    from datasets import load_dataset

    logger.info("Downloading CIC-DDoS2019 (large dataset, may take a while)...")
    ds = load_dataset("baalajimaestro/CICDDoS2019", split="train", streaming=True)

    rows = []
    for i, row in enumerate(ds):
        rows.append(dict(row))
        if (i + 1) % 50000 == 0:
            logger.info(f"  ... {i+1} rows collected")

    df = pd.DataFrame(rows)
    df.to_csv(train_path, index=False)
    logger.info(f"CIC-DDoS2019 saved: {len(df)} rows -> {train_path}")

    return out_dir


def download_bot_iot() -> Path:
    """Download BoT-IoT dataset.

    Tries multiple sources:
    1. masoltani/bot-iot (may have statistical features rather than raw flow)
    2. Falls back to UNSW CloudStor if available
    """
    out_dir = DATA_DIR / "bot_iot" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.csv"

    if train_path.exists() and train_path.stat().st_size > 1000:
        logger.info("BoT-IoT already downloaded, skipping")
        return out_dir

    from datasets import load_dataset

    logger.info("Downloading BoT-IoT...")
    try:
        ds = load_dataset("masoltani/bot-iot", split="train", streaming=True)
        rows = []
        for i, row in enumerate(ds):
            rows.append(dict(row))
            if (i + 1) % 50000 == 0:
                logger.info(f"  ... {i+1} rows collected")

        df = pd.DataFrame(rows)
        df.to_csv(train_path, index=False)
        logger.info(f"BoT-IoT saved: {len(df)} rows -> {train_path}")
    except Exception as e:
        logger.warning(f"BoT-IoT download from HuggingFace failed: {e}")
        logger.warning("BoT-IoT will need a manual download from UNSW research page")

    return out_dir


def main():
    parser = argparse.ArgumentParser(description="Download new IDS datasets")
    parser.add_argument(
        "--datasets",
        default="ton_iot,cic_ddos2019,bot_iot",
        help="Comma-separated datasets to download",
    )
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",")]
    downloaders = {
        "ton_iot": download_ton_iot,
        "cic_ddos2019": download_cic_ddos2019,
        "bot_iot": download_bot_iot,
    }

    for name in datasets:
        if name in downloaders:
            logger.info(f"\n{'='*60}")
            logger.info(f"Downloading {name}...")
            logger.info(f"{'='*60}")
            try:
                out = downloaders[name]()
                logger.info(f"✓ {name} saved to {out}")
            except Exception as e:
                logger.error(f"✗ {name} failed: {e}")
        else:
            logger.warning(f"Unknown dataset: {name}")


if __name__ == "__main__":
    main()
