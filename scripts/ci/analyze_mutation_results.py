#!/usr/bin/env python3
"""Analyze cosmic-ray session SQLite and print mutation score summary.

Usage:
    python3 analyze_mutation_results.py <session.sqlite>
"""

import sqlite3
import sys
from pathlib import Path


def analyze(session_path: str) -> None:
    """Read cosmic-ray SQLite session and compute mutation score."""
    if not Path(session_path).exists():
        print(f"ERROR: Session file not found: {session_path}")
        print("Total: 0, Killed: 0, Survived: 0")
        print("Score: N/A (session file missing)")
        return

    conn = sqlite3.connect(session_path)
    cursor = conn.cursor()

    # Check if work_items table exists
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='work_items'"
    )
    if not cursor.fetchone():
        print("ERROR: No 'work_items' table in session (session may be empty)")
        print("Total: 0, Killed: 0, Survived: 0")
        print("Score: N/A")
        conn.close()
        return

    cursor.execute("SELECT count(*) FROM work_items")
    total = cursor.fetchone()[0]

    # Check if work_results table exists and has data
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='work_results'"
    )
    has_results = bool(cursor.fetchone())

    killed = 0
    if has_results:
        cursor.execute("SELECT test_outcome FROM work_results")
        outcomes = cursor.fetchall()
        killed = sum(1 for row in outcomes if row[0] == 'KILLED')

    conn.close()

    survived = total - killed
    score = (killed / max(total, 1)) * 100

    print(f"Total: {total}, Killed: {killed}, Survived: {survived}")
    print(f"Score: {score:.1f}%")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: analyze_mutation_results.py <session.sqlite>")
        sys.exit(1)
    analyze(sys.argv[1])
