"""Unit tests for testing.mcp_client (S3).

All offline: an in-memory FakeSidecar stands in for the live sidecar so the
JSON-RPC / REST drivers are exercised end to end without a network or encoder.
"""

import pytest

from testing.mcp_client import (
    MCP_PATH,
    SidecarHttpError,
    McpClient,
    RawTurnClient,
)
from tests.unit._fake_sidecar import FakeSidecar

BASE = "http://sidecar.test"


def _mcp() -> tuple[McpClient, FakeSidecar]:
    fake = FakeSidecar()
    return McpClient(BASE, "conv-1", http=fake), fake


# ---------------------------------------------------------------------------
# MCP handshake / tools
# ---------------------------------------------------------------------------
def test_initialize_negotiates_protocol_and_server_info():
    client, fake = _mcp()
    result = client.initialize()
    assert result.ok
    assert result.payload["protocolVersion"] == "2025-06-18"
    assert result.payload["serverInfo"]["name"] == "strata-memory"
    assert fake.requests[0][0] == f"{BASE}{MCP_PATH}"


def test_list_tools_exposes_both_tools():
    client, _ = _mcp()
    result = client.list_tools()
    assert result.ok
    assert {t["name"] for t in result.payload["tools"]} == {
        "strata_search", "strata_remember"}


def test_remember_then_search_roundtrip():
    client, _ = _mcp()
    stored = client.remember("Deploy tokens rotate every 90 days.")
    assert stored.ok and stored.payload["stored"] is True

    found = client.search("What is the deploy token rotation period?")
    assert found.ok
    assert "90 days" in found.payload["assembled_content"]
    assert found.payload["budget"] == 4096


def test_search_carries_top_k_and_default_conversation_id():
    client, fake = _mcp()
    client.search("throttling", top_k=3)
    assert fake.last_search["conversation_id"] == "conv-1"
    assert fake.last_search["top_k"] == 3


def test_explicit_conversation_id_switches_scope_and_isolates():
    client, fake = _mcp()
    client.remember("fact for a", conversation_id="conv-a")
    assert "fact for a" in client.search("fact", conversation_id="conv-a").payload[
        "assembled_content"]
    other = client.search("fact", conversation_id="conv-b")
    assert other.payload["assembled_content"] == ""


def test_missing_conversation_id_is_a_tool_error_from_sidecar():
    client, fake = _mcp()
    result = client.call_tool("strata_search", {"query": "q"})
    # FakeSidecar mirrors the server's required-argument validation.
    assert result.ok
    # call_tool injects the client default, so no error here; the sidecar still
    # rejects a blank one.
    blank = client.call_tool(
        "strata_search", {"query": "q", "conversation_id": "  "})
    assert not blank.ok
    assert "conversation_id" in blank.error


def test_jsonrpc_error_is_reported_not_raised():
    client, fake = _mcp()
    fake.rpc_error = {"code": -32601, "message": "method not found: bogus"}
    result = client.list_tools()
    assert not result.ok
    assert result.error == "method not found: bogus"
    assert result.payload == {}


def test_http_error_is_reported_not_raised():
    client, fake = _mcp()
    fake.http_status = 401
    result = client.remember("secret", conversation_id="c")
    assert not result.ok
    assert "401" in result.error
    assert result.latency_ms == 0.0


def test_token_header_sent_when_configured():
    fake = FakeSidecar()
    client = McpClient(BASE, "c", http=fake, token="tok-123")
    client.initialize()
    assert fake.requests[0][2]["x-strata-token"] == "tok-123"


def test_no_token_header_by_default():
    client, fake = _mcp()
    client.initialize()
    assert "x-strata-token" not in fake.requests[0][2]


# ---------------------------------------------------------------------------
# raw REST driver
# ---------------------------------------------------------------------------
def test_raw_observe_then_curate_roundtrip():
    fake = FakeSidecar()
    raw = RawTurnClient(BASE, "raw-1", http=fake)
    assert raw.observe("routing = circuit breaker", "raw-1").ok
    result = raw.curate("what routing?", "raw-1")
    assert result.ok
    assert "circuit breaker" in result.payload["assembled_content"]


def test_raw_turn_returns_reply_and_assembled():
    fake = FakeSidecar()
    raw = RawTurnClient(BASE, "raw-1", http=fake)
    result = raw.turn("hello", "raw-1")
    assert result.ok
    assert result.payload["reply"] == "fake reply"


def test_raw_reset_clears_the_store():
    fake = FakeSidecar()
    raw = RawTurnClient(BASE, "raw-1", http=fake)
    raw.observe("forget me", "raw-1")
    raw.reset("raw-1")
    assert raw.curate("anything", "raw-1").payload["assembled_content"] == ""


def test_raw_http_error_is_reported():
    fake = FakeSidecar()
    fake.http_status = 500
    raw = RawTurnClient(BASE, "raw-1", http=fake)
    result = raw.curate("q", "raw-1")
    assert not result.ok and "500" in result.error


def test_sidecar_http_error_is_a_runtime_error():
    err = SidecarHttpError(503, "down")
    assert err.status == 503
    with pytest.raises(RuntimeError):
        raise err
