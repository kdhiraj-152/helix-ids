#!/usr/bin/env python3
"""Check IoT-23 scenario files and Kyoto 2006+ data."""
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

# Check individual IoT-23 scenarios for actual data files
scenario = "https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/CTU-IoT-Malware-Capture-1-1/"
print(f"Checking {scenario}...")
try:
    req = urllib.request.Request(scenario)
    resp = urllib.request.urlopen(req, timeout=10)
    html = resp.read().decode("utf-8")
    parser = LinkParser()
    parser.feed(html)
    items = [l for l in parser.links if l != "../"]
    print(f"  Files ({len(items)}): {items}")
except Exception as e:
    print(f"  Error: {type(e).__name__}: {e}")

# Check IoT-23 small tarball size
print("\nChecking IoT-23 small tarball...")
try:
    req = urllib.request.Request("https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/iot_23_datasets_small.tar.gz", method="HEAD")
    resp = urllib.request.urlopen(req, timeout=10)
    size_mb = int(resp.headers.get("Content-Length", 0)) / 1024 / 1024
    print(f"  Size: {size_mb:.1f} MB")
except Exception as e:
    print(f"  Error: {type(e).__name__}: {e}")

# Check Kyoto 2006+ data availability
kyoto_base = "https://www.takakura.com/Kyoto_data/"
print(f"\nChecking {kyoto_base}...")
try:
    req = urllib.request.Request(kyoto_base)
    resp = urllib.request.urlopen(req, timeout=10)
    html = resp.read().decode("utf-8")
    # Find links to data files
    import re
    data_links = re.findall(r'href="([^"]*\.(?:gz|zip|tar|csv|data|txt))"', html, re.IGNORECASE)
    print(f"  Data files: {data_links[:20]}")
    # Also find directory links
    dir_links = re.findall(r'href="([^"]*/)"', html)
    print(f"  Subdirectories: {[l for l in dir_links if l != '../' and not l.startswith('?')][:15]}")
except Exception as e:
    print(f"  Error: {type(e).__name__}: {e}")
