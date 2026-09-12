"""Integration: end-to-end runner harness (offline, mock mode)."""

import json
from pathlib import Path

from cortex.e2e import EndToEndRunner, FakeUltraSmall, MockTransport
from backend.lmstudio import LMStudioBackend


def test_e2e_mock_run_produces_report():
    conv = json.loads(
        Path("hivebench/tests/fixtures/generated/short_001.json").read_text(encoding="utf-8")
    )
    backend = LMStudioBackend(
        base_url="localhost:1234", model="qwen", transport=MockTransport()
    )
    runner = EndToEndRunner(backend, ultra_small=FakeUltraSmall())
    report = runner.run(conv, max_turns=5)

    assert report["conversation_id"] == "short_001"
    assert report["aggregate"]["user_turns"] == 5
    assert len(report["turns"]) == 5

    for t in report["turns"]:
        assert t["reply"]
        assert t["generation_ms"] > 0        # mock simulates generation latency
        assert t["routed_to"] in ("ultra_small", "medium", "escalation")
        assert 0.0 <= t["utilization"] <= 1.0

    assert report["aggregate"]["pes"]["min"] is not None
    assert report["aggregate"]["pes"]["min"] >= 0


def test_e2e_prefix_caching_enabled():
    conv = json.loads(
        Path("hivebench/tests/fixtures/generated/short_001.json").read_text(encoding="utf-8")
    )
    backend = LMStudioBackend(
        base_url="localhost:1234", model="qwen", transport=MockTransport()
    )
    runner = EndToEndRunner(backend, ultra_small=FakeUltraSmall())
    report = runner.run(conv, max_turns=3)

    assert report["backend"] == "LMStudioBackend"
    assert all(t["cache_mode"] == "prefix_caching" for t in report["turns"])
    assert backend.pinned_prefix  # set by the cache manager
