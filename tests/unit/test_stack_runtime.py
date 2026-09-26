"""Unit tests for the LSC runtime-config generator (``harness/stack_runtime.py``).

Fully offline: the stack status payload is a plain dict (the
``StackManager.status()`` shape), the base profile is a small text fixture
with the ``!!js`` tags the bundled default carries, and the generated rows
are checked by parsing each as JSON (they are rendered as JSON flow
mappings, which the dsh YAML loader reads as ordinary mappings).
"""

from __future__ import annotations

import json

from harness.stack_runtime import (
    OPENAI_COMPLETIONS_API,
    PI_AI_PLUGIN,
    PI_AI_ROW_ID,
    SPAWN_PROVIDER,
    SUBAGENT_TOOL_PLUGIN,
    config_digest,
    generate_runtime_config,
    render_entries,
    routing_entries,
    tier_routes,
)

BASE_TEXT = """\
# fixture base profile with loader tags a round-trip would lose
- id: sdk-jsonrpc-server
  name: '@deepseek-ai/dsh-sdk-jsonrpc-server'
- id: sessions
  name: '@deepseek-ai/dsh-session-persistence-jsonl'
  config:
    root: !!js process.env.DSH_SESSION_ROOT ?? './.sessions'
"""


def _status(*, with_worker: bool = True, with_agency: bool = True) -> dict:
    tiers = [{
        "role": "face", "key": "s-face", "port": 1234, "ctx": 262144,
        "model": "Qwen3.8-27B-UD-Q5_K_S", "resident": True,
    }]
    if with_worker:
        tiers.append({
            "role": "worker", "key": "s-worker", "port": 1235, "ctx": 131072,
            "model": "Qwen3.5-4B-UD-Q4_K_XL", "resident": True,
        })
    if with_agency:
        tiers.append({
            "role": "agency", "key": "s-agency", "port": 1236, "ctx": 32768,
            "model": "Ornith-1.5-9B-AD-Q5_K-Q4_K", "resident": True,
        })
    return {"ok": True, "stack": "peer-3tier", "tiers": tiers, "warnings": []}


def _row_config(entries: list[dict], row_id: str) -> dict:
    for entry in entries:
        if entry.get("id") == row_id:
            return entry.get("config") or {}
    raise AssertionError(f"row {row_id!r} missing from {[e.get('id') for e in entries]}")


# ---------------------------------------------------------------------------
# routing
# ---------------------------------------------------------------------------


def test_routes_skip_the_face_and_compose_the_port_url():
    routes = tier_routes(_status())

    assert [r.role for r in routes] == ["worker", "agency"]
    assert routes[0].route == "tier-worker"
    assert routes[0].base_url == "http://127.0.0.1:1235/v1"
    assert routes[0].tool == "delegate_worker"
    assert routes[0].model == "Qwen3.5-4B-UD-Q4_K_XL"
    assert routes[0].ctx == 131072


def test_routes_skip_non_resident_tiers_without_a_port():
    status = _status()
    status["tiers"][1]["port"] = 0  # worker not resident yet

    assert [r.role for r in tier_routes(status)] == ["agency"]


def test_explicit_base_url_wins_over_the_port():
    status = _status(with_agency=False)
    status["tiers"][1]["base_url"] = "http://127.0.0.1:9999/v1/"

    routes = tier_routes(status)
    assert routes[0].base_url == "http://127.0.0.1:9999/v1"


# ---------------------------------------------------------------------------
# generated rows
# ---------------------------------------------------------------------------


def test_entries_mount_routes_and_one_tool_per_tier():
    entries = routing_entries(tier_routes(_status()))
    ids = [e["id"] for e in entries]

    assert ids == [PI_AI_ROW_ID, "lsc-delegate-worker", "lsc-delegate-agency"]
    assert entries[0]["name"] == PI_AI_PLUGIN

    providers = entries[0]["config"]["providers"]
    assert sorted(providers) == ["tier-agency", "tier-worker"]
    worker = providers["tier-worker"]
    assert worker["api"] == OPENAI_COMPLETIONS_API
    assert worker["baseURL"] == "http://127.0.0.1:1235/v1"
    assert worker["models"] == [
        {"id": "Qwen3.5-4B-UD-Q4_K_XL", "contextWindow": 131072}
    ]

    tool = _row_config(entries, "lsc-delegate-worker")
    assert entries[1]["name"] == SUBAGENT_TOOL_PLUGIN
    assert tool["provider"] == SPAWN_PROVIDER
    assert tool["toolName"] == "delegate_worker"
    assert tool["agentOptions"] == {
        "provider": "tier-worker", "model": "Qwen3.5-4B-UD-Q4_K_XL",
    }
    assert tool["maxDepth"] == 0  # workers cannot spawn children
    assert tool["enableRunInBackground"] is False


def test_no_non_face_tiers_generates_nothing():
    assert routing_entries(tier_routes({})) == []
    assert routing_entries(tier_routes(_status(with_worker=False, with_agency=False))) == []


# ---------------------------------------------------------------------------
# rendering and file generation
# ---------------------------------------------------------------------------


def test_rendered_rows_are_json_flow_mappings():
    entries = routing_entries(tier_routes(_status()))
    block = render_entries(entries)

    lines = [line for line in block.splitlines() if line.strip()]
    assert len(lines) == len(entries)
    for line in lines:
        assert line.startswith("- ")
        json.loads(line[2:])  # JSON is valid YAML; the loader reads it as a mapping


def test_generate_appends_to_the_base_and_is_deterministic(tmp_path):
    base = tmp_path / "base.yml"
    base.write_text(BASE_TEXT, encoding="utf-8")
    out = tmp_path / "out"

    first = generate_runtime_config(_status(), base_path=base, out_dir=out)
    second = generate_runtime_config(_status(), base_path=base, out_dir=out)

    assert first.path is not None and first.path.is_file()
    assert first.path == second.path  # same stack -> same digest -> same file
    assert first.digest == second.digest

    text = first.path.read_text(encoding="utf-8")
    assert text.startswith(BASE_TEXT)            # base preserved verbatim
    assert "!!js" in text                        # loader tags untouched
    assert "- " + json.dumps(
        routing_entries(tier_routes(_status()))[0], sort_keys=True,
        separators=(", ", ": ")) in text

    assert first.route_for("worker").tool == "delegate_worker"
    assert first.route_for("face") is None


def test_different_endpoints_get_different_configs(tmp_path):
    base = tmp_path / "base.yml"
    base.write_text(BASE_TEXT, encoding="utf-8")
    out = tmp_path / "out"

    one = _status()
    two = _status()
    two["tiers"][1]["port"] = 4321

    first = generate_runtime_config(one, base_path=base, out_dir=out)
    second = generate_runtime_config(two, base_path=base, out_dir=out)

    assert first.digest != second.digest
    assert first.path != second.path


def test_generate_without_base_or_out_returns_no_path(tmp_path):
    result = generate_runtime_config(_status(), base_path=None, out_dir=tmp_path)

    assert result.path is None
    assert result.entries  # routing facts still available to the caller
    assert result.routes


def test_digest_ignores_row_order_within_the_same_stack():
    entries = routing_entries(tier_routes(_status()))

    assert config_digest(entries) == config_digest(list(reversed(entries)))
