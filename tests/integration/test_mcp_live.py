"""Live MCP acceptance against a running sidecar (S3).

Skipped unless ``STRATA_SIDECAR_URL`` is set, so the default offline suite
stays green. Guard runs this against the sidecar on its chosen port::

    STRATA_SIDECAR_URL=http://127.0.0.1:8790 \
        venv/bin/python -m pytest hivebench/tests/integration/test_mcp_live.py -q

(`venv/bin/python -m testing.mcp_battery --base-url ...` is the standalone
entrypoint with the same measurements.)
"""

import os
import uuid

import pytest

from testing.mcp_battery import run_mcp_battery
from testing.mcp_client import McpClient

BASE_URL = os.environ.get("STRATA_SIDECAR_URL", "").strip()

pytestmark = pytest.mark.skipif(
    not BASE_URL, reason="set STRATA_SIDECAR_URL to run the live MCP acceptance"
)


@pytest.fixture()
def conversation_id():
    return f"hivebench-live-{uuid.uuid4().hex[:12]}"


def test_live_handshake_and_tool_list():
    client = McpClient(BASE_URL, "hivebench-live-handshake", timeout=30)
    init = client.initialize()
    assert init.ok, init.error
    assert init.payload["serverInfo"]["name"] == "strata-memory"

    tools = client.list_tools()
    assert tools.ok, tools.error
    assert {t["name"] for t in tools.payload["tools"]} == {
        "strata_search", "strata_remember"}


def test_live_remember_then_search_roundtrip_and_isolation(conversation_id):
    client = McpClient(BASE_URL, conversation_id, timeout=60)
    assert client.initialize().ok

    text = "Deploy tokens rotate every 90 days per policy."
    stored = client.remember(text)
    assert stored.ok and stored.payload["stored"] is True, stored.error

    found = client.search("What is the deploy token rotation period?")
    assert found.ok, found.error
    assert "90 days" in found.payload["assembled_content"]

    other = client.search("What is the deploy token rotation period?",
                          conversation_id=f"{conversation_id}-other")
    assert other.ok
    assert "90 days" not in other.payload["assembled_content"]


def test_live_battery_produces_recall_and_latency():
    report = run_mcp_battery(
        base_url=BASE_URL,
        conversation_id=f"hivebench-live-battery-{uuid.uuid4().hex[:8]}",
        max_probes=6,
        timeout=60,
    )
    assert report.handshake_ok, report.handshake_error
    assert report.probes > 0
    assert report.mcp["errors"] == 0
    assert report.mcp["recall"] is not None
    assert report.mcp["avg_query_latency_ms"] is not None
