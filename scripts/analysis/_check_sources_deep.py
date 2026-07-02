#!/usr/bin/env python3
"""Check Kyoto 2006+ data files and IoT-23 bro logs."""
import urllib.request
from html.parser import HTMLParser

class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for name, value in attrs:
                if name == "href":
                    self.links.append(value)

# Check Kyoto data directory
kyoto_data = "https://www.takakura.com/Kyoto_data/data/"
print(f"Checking {kyoto_data}...")
try:
    req = urllib.request.Request(kyoto_data)
    resp = urllib.request.urlopen(req, timeout=15)
    html = resp.read().decode("utf-8")
    parser = LinkParser()
    parser.feed(html)
    items = [l for l in parser.links if l != "../"]
    print(f"  Contents ({len(items)}): {items[:30]}")
except Exception as e:
    print(f"  Error: {type(e).__name__}: {e}")

# Check Kyoto data with IP directory
kyoto_ip = "https://www.takakura.com/Kyoto_data/data_with_IP/"
print(f"\nChecking {kyoto_ip}...")
try:
    req = urllib.request.Request(kyoto_ip)
    resp = urllib.request.urlopen(req, timeout=15)
    html = resp.read().decode("utf-8")
    parser = LinkParser()
    parser.feed(html)
    items = [l for l in parser.links if l != "../"]
    print(f"  Contents ({len(items)}): {items[:20]}")
except Exception as e:
    print(f"  Error: {type(e).__name__}: {e}")

# Check IoT-23 bro logs directory
bro_dir = "https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/CTU-IoT-Malware-Capture-1-1/bro/"
print(f"\nChecking {bro_dir}...")
try:
    req = urllib.request.Request(bro_dir)
    resp = urllib.request.urlopen(req, timeout=15)
    html = resp.read().decode("utf-8")
    parser = LinkParser()
    parser.feed(html)
    items = [l for l in parser.links if l != "../"]
    print(f"  Contents ({len(items)}): {items[:20]}")
except Exception as e:
    print(f"  Error: {type(e).__name__}: {e}")

# Try a direct IoT-23 scenario file
conn_log = "https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/CTU-IoT-Malware-Capture-1-1/2018-05-09-192.168.100.103.pcap"
print(f"\nChecking PCAP file...")
try:
    req = urllib.request.Request(conn_log, method="HEAD")
    resp = urllib.request.urlopen(req, timeout=10)
    print(f"  ✓ Available (status {resp.status})")
except Exception as e:
    print(f"  Error: {type(e).__name__}: {e}")

# Check for labeled conn.log in the bro directory
bro_conn = "https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/CTU-IoT-Malware-Capture-1-1/bro/conn.log.labeled"
print(f"\nChecking {bro_conn}...")
try:
    req = urllib.request.Request(bro_conn, method="HEAD")
    resp = urllib.request.urlopen(req, timeout=10)
    print(f"  ✓ Available ({resp.status})")
except Exception as e:
    print(f"  Error: {type(e).__name__}: {e}")
