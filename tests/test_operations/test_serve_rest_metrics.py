from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from scripts.operations.serve_rest import create_app


class _FakeRuntime:
    def __init__(self) -> None:
        self.schema_hash = "abc"
        self.contract_version = "1.0"
        self._override_next = False

    def set_override(self, value: bool) -> None:
        self._override_next = bool(value)

    def predict(self, _features: Any) -> dict[str, Any]:
        return {
            "family_class": 1,
            "confidence": 0.9,
            "coverage_override_applied": self._override_next,
            "coverage_override_class": 4 if self._override_next else None,
            "coverage_override_logit": -5.0 if self._override_next else None,
            "coverage_override_threshold": -7.0 if self._override_next else None,
            "class_margin_override_applied": False,
            "class_margin_override_second_class": None,
            "class_margin_override_margin": None,
            "class_margin_tau_adaptive": None,
            "class_margin_adaptive_frozen": False,
            "class_margin_buffer_size": 0,
            "class_margin_enabled": True,
            "class_margin_collapse_alert": False,
        }


def test_metrics_endpoint_prometheus_format_and_counters() -> None:
    capture = Path('artifacts/operations/live_events.jsonl')
    if capture.exists():
        capture.unlink()

    runtime = _FakeRuntime()
    app = create_app(runtime)
    client = TestClient(app)

    r0 = client.get('/metrics')
    assert r0.status_code == 200
    assert r0.headers['content-type'].startswith('text/plain')
    assert 'helix_requests_total 0' in r0.text
    assert 'helix_coverage_override_total 0' in r0.text
    assert 'helix_coverage_override_rate 0.0000000000' in r0.text
    assert 'helix_degraded_state 0' in r0.text

    runtime.set_override(False)
    p1 = client.post('/predict', json={'features': [0.1] * 17})
    assert p1.status_code == 200
    body1 = p1.json()
    assert body1["class_margin_override_applied"] is False
    assert body1["class_margin_adaptive_frozen"] is False
    assert body1["class_margin_enabled"] is True

    r1 = client.get('/metrics')
    assert 'helix_requests_total 1' in r1.text
    assert 'helix_coverage_override_total 0' in r1.text
    assert 'helix_coverage_override_rate 0.0000000000' in r1.text
    assert 'helix_degraded_state 0' in r1.text

    runtime.set_override(True)
    p2 = client.post('/predict', json={'features': [0.2] * 17})
    assert p2.status_code == 200
    body2 = p2.json()
    assert body2["class_margin_override_second_class"] is None
    assert body2["class_margin_buffer_size"] == 0
    assert body2["class_margin_collapse_alert"] is False

    r2 = client.get('/metrics')
    assert 'helix_requests_total 2' in r2.text
    assert 'helix_coverage_override_total 1' in r2.text
    assert 'helix_coverage_override_rate 0.5000000000' in r2.text
    assert 'helix_degraded_state 1' in r2.text
    assert 'helix_class_predictions_total{class="1"} 2' in r2.text
    assert 'helix_class_entropy 0.0000000000' in r2.text

    assert capture.exists()
    lines = capture.read_text(encoding='utf-8').strip().splitlines()
    assert len(lines) == 2
    events = [json.loads(line) for line in lines]
    assert events[0]['prediction']['family_class'] == 1
    assert events[0]['override']['applied'] is False
    assert events[0]['class_margin_override']['applied'] is False
    assert events[1]['override']['applied'] is True
    assert events[1]['class_margin_override']['second_class'] is None
