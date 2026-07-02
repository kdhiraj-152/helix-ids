#!/usr/bin/env python3
"""Check alternative sources for external IDS datasets."""
import urllib.request
import json

# Try Kaggle mirrors and alternative sources
sources = [
    # Edge-IIoTset from alternative source
    ("Edge-IIoTset (alternative)", "https://raw.githubusercontent.com/ThijmenL94/Edge-IIoTset-Dataset/main/README.md"),
    # IoT-23 from CTU University
    ("IoT-23 CTU", "https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/"),
    # CIC-IDS2017 from alternative
    ("CIC-IDS2017 (alternative)", "https://raw.githubusercontent.com/rahilq/CIC-IDS-2017-Analysis/main/README.md"),
]

for name, url in sources:
    try:
        req = urllib.request.Request(url, method="GET")
        resp = urllib.request.urlopen(req, timeout=10)
        content = resp.read().decode("utf-8")[:200]
        print(f"✓ {name}: {resp.status} - {content[:100]}")
    except Exception as e:
        print(f"✗ {name}: {type(e).__name__}: {e}")

# Try to find direct CSV links for IDS datasets on various mirrors
print("\n--- Checking specific data file availability ---")

csv_checks = [
    ("IoT-23 conn.log", "https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/CTU-IoT-Malware-Capture-1-1conn.log.labeled"),
    ("UGR'16 sample", "https://nesg.ugr.es/nesg-ugr16/download/README"),
    ("Kyoto 2006+", "https://www.takakura.com/Kyoto_data/"),
]

for name, url in csv_checks:
    try:
        req = urllib.request.Request(url, method="GET")
        resp = urllib.request.urlopen(req, timeout=10)
        print(f"✓ {name}: {resp.status}")
    except Exception as e:
        print(f"✗ {name}: {type(e).__name__}")

print("\nDone.")
