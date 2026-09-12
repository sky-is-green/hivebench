"""Unit tests for testing.mcp_battery (S3).

Drives the whole recall + latency battery against the in-memory FakeSidecar:
probe planning from fixtures, deterministic recall scoring, the A/B fold, and
the CLI's empty-corpus guard. No network, no encoder.
"""

import pytest

from testing.mcp_battery import (
    build_probes,
    content_terms,
    format_report,
    main,
    recall_score,
    run_mcp_battery,
)
from tests.unit._fake_sidecar import FakeSidecar

BASE = "http://sidecar.test"

CONV = {
    "conversation_id": "test_001",
    "turns": [
        {"role": "user", "content": "Let's settle throttling for the gateway."},
        {"role": "assistant",
         "content": "Key decision: throttling = bucket throttling quotas."},
        {"role": "user", "content": "Let's settle routing for the gateway."},
        {"role": "assistant",
         "content": "Key decision: routing = circuit breaker routing."},
        {"role": "user", "content": "Remind me what we decided for throttling."},
        {"role": "assistant",
         "content": "The throttling decision is bucket throttling quotas."},
        {"role": "user", "content": "What was the final decision on routing?"},
        {"role": "assistant",
         "content": "The routing decision is circuit breaker routing."},
    ],
}


# ---------------------------------------------------------------------------
# probe planning
# ---------------------------------------------------------------------------
def test_build_probes_finds_long_horizon_recall_turns():
    memory, probes = build_probes(CONV)
    assert [m["turn"] for m in memory] == [1, 3, 5, 7]
    # only the recap turns that reference an *earlier* decision are probes
    assert [p["turn"] for p in probes] == [4, 6]
    assert probes[0]["facet"] == "throttling"
    assert "bucket throttling quotas" in probes[0]["expected"]
    assert probes[1]["facet"] == "routing"


def test_build_probes_falls_back_to_recent_memory():
    conv = {
        "conversation_id": "plain",
        "turns": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "the sky is blue today"},
            {"role": "user", "content": "what colour?"},
            {"role": "assistant", "content": "blue"},
        ],
    }
    _, probes = build_probes(conv)
    assert [p["turn"] for p in probes] == [2]
    assert probes[0]["expected"] == "the sky is blue today"


# ---------------------------------------------------------------------------
# recall scoring
# ---------------------------------------------------------------------------
def test_content_terms_strips_stopwords_and_short_tokens():
    assert content_terms("The key decision: to use a big cache!") == {"big", "cache"}


def test_recall_score_full_partial_and_none():
    assert recall_score("bucket throttling quotas", "the bucket throttling quotas apply") == 1.0
    assert recall_score("bucket throttling quotas", "only throttling here") == pytest.approx(1 / 3, abs=1e-4)
    assert recall_score("", "anything") is None


# ---------------------------------------------------------------------------
# battery over the fake sidecar
# ---------------------------------------------------------------------------
def test_battery_measures_both_paths_and_folds_ab():
    fake = FakeSidecar()
    report = run_mcp_battery(
        base_url=BASE, conversations=[CONV], http=fake,
        conversation_id="bench", max_probes=10,
    )
    assert report.handshake_ok
    assert report.probes == 2
    assert report.mcp["probes"] == 2
    assert report.mcp["errors"] == 0
    assert report.mcp["recall"] == 1.0
    assert report.raw["recall"] == 1.0
    assert report.mcp["avg_query_latency_ms"] is not None
    assert report.mcp["avg_ingest_latency_ms"] is not None
    assert report.ab["winner"] in {"A", "B", "tie"}
    assert report.ab["config_a"]["pes"] == 100.0
    # isolation: each conversation+path gets its own store scope
    assert "bench-test_001-mcp" in fake.store
    assert "bench-test_001-raw" in fake.store


def test_battery_raw_turn_mode_uses_the_turn_endpoint():
    fake = FakeSidecar()
    report = run_mcp_battery(
        base_url=BASE, conversations=[CONV], http=fake,
        conversation_id="bench", raw_mode="turn",
    )
    assert report.raw_mode == "turn"
    assert report.raw["recall"] == 1.0
    assert any(url.endswith("/v1/strata/turn") for url, _b, _h in fake.requests)


def test_battery_counts_errors_when_sidecar_is_down():
    fake = FakeSidecar()
    fake.http_status = 500
    report = run_mcp_battery(
        base_url=BASE, conversations=[CONV], http=fake, conversation_id="down",
    )
    assert not report.handshake_ok
    assert report.mcp["errors"] > 0
    # the A/B fold still runs, but every metric is absent
    assert report.ab["config_a"]["pes"] is None
    assert report.ab["config_b"]["pes"] is None


def test_battery_skips_conversations_without_probes():
    empty = {"conversation_id": "empty", "turns": []}
    report = run_mcp_battery(
        base_url=BASE, conversations=[empty], http=FakeSidecar(),
        conversation_id="bench",
    )
    assert report.probes == 0


# ---------------------------------------------------------------------------
# reporting / CLI
# ---------------------------------------------------------------------------
def test_format_report_includes_both_paths():
    fake = FakeSidecar()
    report = run_mcp_battery(
        base_url=BASE, conversations=[CONV], http=fake, conversation_id="bench",
    )
    text = format_report(report)
    assert "MCP path" in text and "raw path" in text
    assert "A/B winner" in text


def test_cli_rejects_empty_corpus(tmp_path, capsys):
    code = main(["--conversations", str(tmp_path / "nope")])
    assert code == 2
    assert "no conversations" in capsys.readouterr().out
