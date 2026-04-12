"""HELIX-IDS source package."""

import sys
from pathlib import Path

# Register archive/legacy_code as a module namespace
archive_path = Path(__file__).parent.parent / "archive" / "legacy_code"
if str(archive_path) not in sys.path:
    sys.path.insert(0, str(archive_path))
