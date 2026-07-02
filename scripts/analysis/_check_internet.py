import urllib.request
try:
    resp = urllib.request.urlopen("https://httpbin.org/get", timeout=5)
    print(f"Internet OK: {resp.status}")
except Exception as e:
    print(f"No internet: {e}")
