#!/usr/bin/env python3
"""
Download Bot-IoT and CICIDS2017 with reservoir sampling to cache.

Bot-IoT has ~7.5M rows; we take a 200K stratified-ish random sample
to avoid downloading the entire dataset.
"""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_subsample")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

MAX_SAMPLES = 200_000
RANDOM_SEED = 42


def reservoir_sample(iterable, k: int, rng: random.Random) -> list:
    """Reservoir sampling (algorithm R) over any iterable, returning exactly k items."""
    k = min(k, 10_000_000)  # upper bound safety
    reservoir = []
    for i, item in enumerate(iterable):
        if i < k:
            reservoir.append(item)
        else:
            j = rng.randint(0, i)
            if j < k:
                reservoir[j] = item
        if (i + 1) % 200_000 == 0:
            logger.info(f"  ... scanned {i+1} rows, reservoir holds {len(reservoir)}")
    logger.info(f"  Scanned {i+1} total rows, final sample size {len(reservoir)}")
    return reservoir


def download_bot_iot():
    """Download Bot-IoT subsample via reservoir sampling."""
    out_dir = DATA_DIR / "bot_iot" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "train.csv"

    if csv_path.exists() and csv_path.stat().st_size > 1000:
        logger.info(f"Bot-IoT already cached at {csv_path} ({csv_path.stat().st_size} bytes)")
        return

    from datasets import load_dataset

    logger.info("Downloading Bot-IoT (reservoir sampling to 200K)...")
    ds = load_dataset("masoltani/bot-iot", split="train", streaming=True)

    rng = random.Random(RANDOM_SEED)
    sampled = reservoir_sample(iter(ds), MAX_SAMPLES, rng)

    df = pd.DataFrame(sampled)
    df.to_csv(csv_path, index=False)
    logger.info(f"Bot-IoT cached: {len(df)} rows -> {csv_path}")


def download_cicids2017():
    """Download CICIDS2017 subsample."""
    out_dir = DATA_DIR / "cicids2017" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "train.csv"

    if csv_path.exists() and csv_path.stat().st_size > 1000:
        logger.info(f"CICIDS2017 already cached at {csv_path} ({csv_path.stat().st_size} bytes)")
        return

    from datasets import load_dataset

    logger.info("Downloading CICIDS2017 (reservoir sampling to 200K)...")
    ds = load_dataset("rdpahalavan/CIC-IDS2017", split="train", streaming=True)

    rng = random.Random(RANDOM_SEED)
    sampled = reservoir_sample(iter(ds), MAX_SAMPLES, rng)

    df = pd.DataFrame(sampled)
    df.to_csv(csv_path, index=False)
    logger.info(f"CICIDS2017 cached: {len(df)} rows -> {csv_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1]
        if target == "bot_iot":
            download_bot_iot()
        elif target == "cicids2017":
            download_cicids2017()
        else:
            print(f"Unknown: {target}")
    else:
        download_bot_iot()
        download_cicids2017()
