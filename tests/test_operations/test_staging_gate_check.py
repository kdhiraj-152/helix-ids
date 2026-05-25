from __future__ import annotations

import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class _MetricsHandler(BaseHTTPRequestHandler):
    payload = ""

    def do_GET(self):  # noqa: N802
        if self.path != "/metrics":
            self.send_response(404)
            self.end_headers()
            return
        body = self.payload.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # noqa: A003
        return


def _run_gate(metrics_text: str) -> subprocess.CompletedProcess[str]:
    _MetricsHandler.payload = metrics_text
    port = _free_port()
    server = HTTPServer(("127.0.0.1", port), _MetricsHandler)
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    try:
        cmd = [
            sys.executable,
            "scripts/operations/staging_gate_check.py",
            "--metrics-endpoint",
            f"http://127.0.0.1:{port}/metrics",
        ]
        return subprocess.run(cmd, check=False, text=True, capture_output=True)
    finally:
        server.shutdown()
        th.join(timeout=5)
        server.server_close()


def test_staging_gate_passes_when_below_threshold_and_not_degraded() -> None:
    result = _run_gate(
        "\n".join(
            [
                "# HELP helix_coverage_override_rate Override rate",
                "helix_coverage_override_rate 0.009",
                "helix_degraded_state 0",
                "",
            ]
        )
    )
    assert result.returncode == 0
    assert result.stdout == "[HELIX GATE] OK\noverride_rate=0.009\n"


def test_staging_gate_fails_when_override_rate_exceeds_threshold() -> None:
    result = _run_gate(
        "\n".join(
            [
                "helix_coverage_override_rate 0.034",
                "helix_degraded_state 0",
                "",
            ]
        )
    )
    assert result.returncode == 1
    assert result.stdout == "[HELIX GATE] BLOCKED\noverride_rate=0.034\ndegraded_state=0\n"


def test_staging_gate_fails_when_degraded_state_is_one() -> None:
    result = _run_gate(
        "\n".join(
            [
                "helix_coverage_override_rate 0.009",
                "helix_degraded_state 1",
                "",
            ]
        )
    )
    assert result.returncode == 1
    assert result.stdout == "[HELIX GATE] BLOCKED\noverride_rate=0.009\ndegraded_state=1\n"
