"""Sidecar MCP tests (S2): POST /v1/mcp wiring on the running app.

All offline: fake drone + mock transport backend, tmp cwd, mirroring the
harness-service fixture. Covers the ticket's acceptance shape end to end:
initialize -> tools/list -> splinter_remember -> splinter_search in a fresh
conversation_id, plus cross-conversation isolation.
"""

import json

import pytest
from fastapi.testclient import TestClient

from backend.openai_compat import OpenAICompatBackend
from cortex.e2e import FakeUltraSmall, MockTransport
from harness.app import create_app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # event logs land in tmp

    def backend_factory(model=None, provider=None):
        return OpenAICompatBackend(
            base_url="http://mock", model=model or "mock-model",
            transport=MockTransport(latency_ms=0),
        )

    app = create_app(
        ultra_factory=FakeUltraSmall,
        backend_factory=backend_factory,
        runs_root=tmp_path / "runs",
        providers_file=tmp_path / "providers.local.json",
        log_dir=str(tmp_path / "logs"),
    )
    with TestClient(app) as c:
        yield c


def _rpc(c, body):
    return c.post("/v1/mcp", json=body)


def test_mcp_handshake_and_tools_list(client):
    init = _rpc(client, {
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                   "clientInfo": {"name": "s2-test", "version": "0"}},
    })
    assert init.status_code == 200
    result = init.json()["result"]
    assert result["protocolVersion"] == "2025-06-18"
    assert result["serverInfo"]["name"] == "splinter-memory"

    # notification-only body: accepted, nothing to answer
    assert _rpc(client, {
        "jsonrpc": "2.0", "method": "notifications/initialized",
    }).status_code == 202

    tools = _rpc(client, {
        "jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {},
    }).json()["result"]["tools"]
    assert {t["name"] for t in tools} == {"splinter_search", "splinter_remember"}


def test_mcp_remember_then_search_roundtrip_and_isolation(client):
    conv = "mcp-accept-fresh"
    text = "Deploy tokens rotate every 90 days per policy."
    stored = _rpc(client, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "splinter_remember", "arguments": {
            "conversation_id": conv, "text": text}},
    }).json()
    assert "error" not in stored
    assert json.loads(stored["result"]["content"][0]["text"])["stored"] is True

    found = _rpc(client, {
        "jsonrpc": "2.0", "id": 2, "method": "tools/call",
        "params": {"name": "splinter_search", "arguments": {
            "conversation_id": conv,
            "query": "What is the deploy token rotation period?"}},
    }).json()
    assert "error" not in found
    payload = json.loads(found["result"]["content"][0]["text"])
    assert "90 days" in payload["assembled_content"]

    # isolation: a different conversation must not see it
    other = _rpc(client, {
        "jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": {"name": "splinter_search", "arguments": {
            "conversation_id": "mcp-accept-other",
            "query": "What is the deploy token rotation period?"}},
    }).json()
    assert "error" not in other
    assert "90 days" not in json.loads(
        other["result"]["content"][0]["text"])["assembled_content"]


def test_mcp_call_without_conversation_id_is_invalid_params(client):
    body = _rpc(client, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "splinter_search",
                   "arguments": {"query": "hello"}},
    }).json()
    assert body["error"]["code"] == -32602


def test_mcp_unknown_tool_and_bad_json(client):
    body = _rpc(client, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "nope", "arguments": {"conversation_id": "c"}},
    }).json()
    assert body["error"]["code"] == -32601

    raw = client.post("/v1/mcp", content=b"not json",
                      headers={"Content-Type": "application/json"})
    assert raw.status_code in (400, 422)


def test_mcp_get_is_method_not_allowed(client):
    assert client.get("/v1/mcp").status_code == 405
