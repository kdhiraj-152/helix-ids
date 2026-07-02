#!/usr/bin/env python3
"""Try to find accessible files on IoT-23 and Kyoto 2006+ repositories."""
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

# Check IoT-23 data directory
base = "https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/"
print(f"Checking {base}...")
try:
    req = urllib.request.Request(base)
    resp = urllib.request.urlopen(req, timeout=10)
    html = resp.read().decode("utf-8")
    parser = LinkParser()
    parser.feed(html)
    print(f"  Directories: {[l for l in parser.links if not l.startswith('?') and l != '../'][:20]}")
except Exception as e:
    print(f"  Error: {type(e).__name__}: {e}")

# Check if there's data available under IndividualScenarios
subdir = "https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/"
print(f"\nChecking {subdir}...")
try:
    req = urllib.request.Request(subdir)
    resp = urllib.request.urlopen(req, timeout=10)
    html = resp.read().decode("utf-8")
    parser = LinkParser()
    parser.feed(html)
    items = [l for l in parser.links if l != "../"]
    print(f"  Files: {items[:15]}")
    # Check if .labeld files exist
    labeled = [l for l in items if "labeled" in l or "label" in l]
    print(f"  Labeled files: {labeled[:10]}")
except Exception as e:
    print(f"  Error: {type(e).__name__}: {e}")

# Check Kyoto 2006+
kyoto = "https://www.takakura.com/Kyoto_data/"
print(f"\nChecking {kyoto}...")
try:
    req = urllib.request.Request(kyoto)
    resp = urllib.request.urlopen(req, timeout=10)
    html = resp.read().decode("utf-8")
    print(f"  Response: {html[:500]}")
except Exception as e:
    print(f"  Error: {type(e).__name__}: {e}")

# Try alternative: KDD Cup 1999 (can use as additional external)
kdd = "https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html"
print(f"\nChecking KDD99 ({kdd})...")
try:
    req = urllib.request.Request(kdd)
    resp = urllib.request.urlopen(req, timeout=10)
    print(f"  ✓ Available ({resp.status})")
except Exception as e:
    print(f"  Error: {type(e).__name__}: {e}")

print("\nDone.")
