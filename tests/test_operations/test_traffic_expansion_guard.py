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


class _DynamicMetricsHandler(BaseHTTPRequestHandler):
    states: list[int] = [0]
    idx: int = 0

    def do_GET(self):  # noqa: N802
        if self.path != "/metrics":
            self.send_response(404)
            self.end_headers()
            return

        i = min(self.__class__.idx, len(self.__class__.states) - 1)
        degraded = int(self.__class__.states[i])
        self.__class__.idx += 1

        payload = "\n".join(
            [
                "# HELP helix_degraded_state 1 when degraded",
                "# TYPE helix_degraded_state gauge",
                f"helix_degraded_state {degraded}",
                "",
            ]
        ).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):  # noqa: A003
        return


def _run_guard(*, states: list[int], interval: float = 0.1, timeout: float = 3.0):
    _DynamicMetricsHandler.states = states
    _DynamicMetricsHandler.idx = 0

    port = _free_port()
    server = HTTPServer(("127.0.0.1", port), _DynamicMetricsHandler)
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    try:
        cmd = [
            sys.executable,
            "scripts/operations/traffic_expansion_guard.py",
            "--metrics-endpoint",
            f"http://127.0.0.1:{port}/metrics",
            "--interval",
            str(interval),
        ]
        try:
            p = subprocess.run(cmd, check=False, text=True, capture_output=True, timeout=timeout)
            return {"timed_out": False, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
        except subprocess.TimeoutExpired as exc:
            out = exc.stdout.decode("utf-8") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            err = exc.stderr.decode("utf-8") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            return {"timed_out": True, "returncode": None, "stdout": out, "stderr": err}
    finally:
        server.shutdown()
        th.join(timeout=5)
        server.server_close()


def test_guard_halts_immediately_when_degraded() -> None:
    result = _run_guard(states=[1], interval=0.1, timeout=2.0)
    assert result["timed_out"] is False
    assert result["returncode"] == 1
    assert result["stdout"] == "[HELIX GUARD] HALT\ndegraded_state=1\n"


def test_guard_runs_and_emits_ok_when_not_degraded() -> None:
    result = _run_guard(states=[0], interval=0.1, timeout=1.2)
    assert result["timed_out"] is True
    assert "[HELIX GUARD] OK\n" in result["stdout"]


def test_guard_halts_after_transition_to_degraded() -> None:
    result = _run_guard(states=[0, 0, 1], interval=0.1, timeout=3.0)
    assert result["timed_out"] is False
    assert result["returncode"] == 1
    assert result["stdout"].startswith("[HELIX GUARD] OK\n")
    assert result["stdout"].endswith("[HELIX GUARD] HALT\ndegraded_state=1\n")
