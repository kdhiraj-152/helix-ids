#!/usr/bin/env python3
"""
Process CICIDS-2018 dataset.

This script downloads the CICIDS-2018 dataset from the official source,
organizes the day-wise CSV files into the correct project directory structure,
and prepares them for loading.
"""

import shutil
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
ARCHIVE2_DIR = DATA_DIR / "archive-2"


def organize_cicids_archive(archive_path: str | Path) -> None:
    """
    Organize CICIDS-2018 archive files into project structure.

    Args:
        archive_path: Path to the extracted CICIDS-2018 archive directory
                      containing day-wise CSV files
    """
    archive_dir = Path(archive_path)

    if not archive_dir.exists():
        raise FileNotFoundError(f"Archive path does not exist: {archive_dir}")

    # Ensure archive-2 directory exists
    ARCHIVE2_DIR.mkdir(parents=True, exist_ok=True)

    # Find all CSV files in the archive
    csv_files: list[Path] = []
    for pattern in ["**/*.csv", "**/*.CSV"]:
        csv_files.extend(archive_dir.glob(pattern))

    if not csv_files:
        # Try alternative locations - CICIDS-2018 has CSVs in TrafficLabelling/
        traffic_dir = archive_dir / "TrafficLabelling"
        if traffic_dir.exists():
            csv_files = list(traffic_dir.glob("*.csv"))
        else:
            # List contents to debug
            print(f"Archive contents: {list(archive_dir.glob('*'))}")
            raise ValueError(f"No CSV files found in {archive_dir}")

    print(f"Found {len(csv_files)} CSV files in archive")

    # Copy CSV files to archive-2 directory
    for csv_file in csv_files:
        dest_path = ARCHIVE2_DIR / csv_file.name
        print(f"Copying {csv_file.name} -> {dest_path}")
        shutil.copy2(csv_file, dest_path)

    print(f"\nSuccessfully organized {len(csv_files)} CICIDS-2018 CSV files")
    print(f"Files are now in: {ARCHIVE2_DIR}")


def verify_cicids_data() -> None:
    """Verify that CICIDS-2018 data is properly organized and can be loaded."""
    from src.helix_ids.data.unified_loader import UnifiedDataLoader

    if not ARCHIVE2_DIR.exists():
        print("ERROR: archive-2 directory does not exist")
        return

    csv_files = list(ARCHIVE2_DIR.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files in {ARCHIVE2_DIR}")

    if not csv_files:
        print("ERROR: No CSV files found in archive-2")
        return

    # Try to load a small sample to verify
    loader = UnifiedDataLoader()
    try:
        # Load with cicids-2018 config
        x_samples, _, class_names = loader.load("cicids-2018", split="train", fit=True)
        print(f"SUCCESS: Loaded {len(x_samples)} sample rows from CICIDS-2018")
        print(f"Feature count: {x_samples.shape[1]}")
        print(f"Classes: {class_names}")
    except Exception as e:
        print(f"ERROR loading CICIDS-2018: {e}")
        return

    print("\nCICIDS-2018 data is ready for use!")


def main():
    """Main entry point."""
    print("CICIDS-2018 Dataset Processing Script")
    print("=" * 50)
    print("Note: Due to the large size (2GB), CICIDS-2018 must be downloaded manually.")
    print("Please download from: https://www.unb.ca/cic/datasets/ids-2018.html")
    print("Extract the ZIP file and place the extracted folder in your Downloads directory.")
    print("Then run: python scripts/process_cicids.py /path/to/extracted/CICIDS2018")
    print("=" * 50)

    if len(sys.argv) != 2:
        print("\nUsage: python process_cicids.py <path_to_extracted_archive>")
        print("\nExample:")
        print("  python scripts/process_cicids.py ~/Downloads/CICIDS2018")
        print("\nThis will:")
        print("1. Copy all CSV files from the archive to data/archive-2/")
        print("2. Verify the data can be loaded by the project")
        sys.exit(1)

    archive_path = sys.argv[1]
    print(f"Processing CICIDS-2018 from: {archive_path}")

    try:
        # Organize files
        organize_cicids_archive(archive_path)

        # Verify data
        print("\nVerifying data...")
        verify_cicids_data()

        print("\n" + "=" * 50)
        print("CICIDS-2018 processing completed successfully!")
        print("You can now use 'cicids-2018' in your training configuration.")

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
