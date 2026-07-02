#!/usr/bin/env python3
"""
Check what external IDS datasets are available for download.
Tests URL accessibility for several public IDS datasets.
"""
import urllib.request
import json
import sys

datasets = {
    "UNB CIC-IDS2017 (sample)": "https://www.unb.ca/cic/datasets/ids-2017.html",
    "CSE-CIC-IDS2018": "https://www.unb.ca/cic/datasets/ids-2018.html",
    "Edge-IIoTset": "https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot",
    "IoT-23": "https://www.stratosphereips.org/datasets-iot23",
    "UGR'16": "https://nesg.ugr.es/nesg-ugr16/",
}

for name, url in datasets.items():
    try:
        req = urllib.request.Request(url, method="HEAD")
        resp = urllib.request.urlopen(req, timeout=10)
        print(f"✓ {name}: {resp.status}")
    except Exception as e:
        print(f"✗ {name}: {type(e).__name__}")

# Check for direct downloadable CSVs
csv_urls = {
    "CIC-IDS2017 (CSV)": "https://github.com/Soheil-ab/CIC-IDS2017/raw/master/MachineLearningCSV/CIC-IDS2017/TrafficLabelling/Thursday-WorkingHours.pcap_ISCX.csv",
    "IoT-23 sample": "https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/CTU-IoT-Malware-Capture-1-1conn.log.labeled",
}

for name, url in csv_urls.items():
    try:
        req = urllib.request.Request(url, method="HEAD")
        resp = urllib.request.urlopen(req, timeout=10)
        print(f"✓ {name}: {resp.status}")
    except Exception as e:
        print(f"✗ {name}: {type(e).__name__}")

print("\nDone checking.")
