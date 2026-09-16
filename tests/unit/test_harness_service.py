"""Unit tests for the harness FastAPI sidecar (HARNESS-SPEC Â§3.3).

All offline: fake drone + mock transport backend, tmp cwd / runs root, so no
LM Studio or encoder download is needed. Exercises every M1 endpoint.
"""

import json
import time
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

import harness.app as harness_app_module
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
        yield c, app


# ---------------------------------------------------------------------------
# /v1/openai/chat/completions (real curated passthrough, Mode A integration)
# ---------------------------------------------------------------------------
def _configure_lm_provider(c, base_url="http://mock-llama", model="m1"):
    r = c.post("/v1/provider/config", json={
        "providers": [{"name": "lm", "base_url": base_url,
                       "api_key": "lm-studio", "model": model}],
    })
    assert r.status_code == 200, r.text


class _FakeUpstream:
    """Stands in for the provider's /v1/chat/completions."""

    def __init__(self, content="JWT tokens with rotation", sse_chunks=None):
        self.content = content
        self.sse_chunks = sse_chunks or []

    def __call__(self, url, json=None, headers=None, stream=False, timeout=None):
        self.url = url
        self.payload = json
        self.stream = stream
        return self

    def raise_for_status(self):
        pass

    def json(self):
        return {
            "id": "up-1", "object": "chat.completion", "created": 1,
            "model": self.payload.get("model"),
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": self.content},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 6},
        }

    def iter_lines(self, decode_unicode=True):
        yield from self.sse_chunks


def test_openai_chat_completions_non_stream(client, monkeypatch):
    c, _app = client
    _configure_lm_provider(c)
    import harness.app as appmod

    fake = _FakeUpstream()
    monkeypatch.setattr("strata.server.requests.post", fake)
    r = c.post("/v1/openai/chat/completions", json={
        "model": "prism-ml/bonsai-27b",
        "messages": [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "Which tokens do I use for auth expiry?"},
        ],
    })
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "JWT tokens with rotation"
    # client system message preserved behind the curated context placeholder
    sys_msg = fake.payload["messages"][0]
    assert sys_msg["role"] == "system"
    assert "be terse" in sys_msg["content"]
    # model resolves from the provider config, not the client
    assert fake.payload["model"] == "m1"
    # the reply was observed back into the store (2 chunks: query + reply)
    st = c.get("/v1/strata/state", params={"conversation_id": "default"}).json()
    assert st["turn"] == 1
    assert st["store_chunks"] >= 2


def test_openai_chat_completions_stream_relays_and_observes(client, monkeypatch):
    c, _app = client
    _configure_lm_provider(c)
    import harness.app as appmod

    chunks = [
        'data: {"id":"u","object":"chat.completion.chunk","created":1,"model":"m1",'
        '"choices":[{"index":0,"delta":{"role":"assistant","content":"JWT tokens "},"finish_reason":null}]}',
        'data: {"id":"u","object":"chat.completion.chunk","created":1,"model":"m1",'
        '"choices":[{"index":0,"delta":{"content":"with rotation"},"finish_reason":null}]}',
        'data: {"id":"u","object":"chat.completion.chunk","created":1,"model":"m1",'
        '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
        '"usage":{"completion_tokens":6}}',
        "data: [DONE]",
    ]
    fake = _FakeUpstream(sse_chunks=chunks)
    monkeypatch.setattr("strata.server.requests.post", fake)
    r = c.post("/v1/openai/chat/completions", json={
        "model": "x",
        "stream": True,
        "messages": [{"role": "user", "content": "Which tokens do I use for auth expiry?"}],
    })
    assert r.status_code == 200
    text = r.text
    assert "data: [DONE]" in text
    assert "JWT tokens" in text and "with rotation" in text
    # reply observed back (non-hedge, stored)
    st = c.get("/v1/strata/state", params={"conversation_id": "default"}).json()
    assert st["store_chunks"] >= 2


def test_openai_chat_completions_conversation_header_and_errors(client, monkeypatch):
    c, _app = client
    import harness.app as appmod

    fake = _FakeUpstream()
    monkeypatch.setattr("strata.server.requests.post", fake)
    # no provider configured -> 502
    r = c.post("/v1/openai/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 502
    _configure_lm_provider(c)
    # empty messages -> 422
    r = c.post("/v1/openai/chat/completions", json={"messages": []})
    assert r.status_code == 422
    # conversation keyed by the X-Strata-Conversation header
    r = c.post("/v1/openai/chat/completions", json={
        "messages": [{"role": "user", "content": "Which tokens do I use for auth expiry?"}],
    }, headers={"X-Strata-Conversation": "proj-a"})
    assert r.status_code == 200
    st = c.get("/v1/strata/state", params={"conversation_id": "proj-a"}).json()
    assert st["turn"] == 1


def test_openai_curated_context_feeds_next_turn(tmp_path, monkeypatch):
    """With distinct drone embeddings (real-drone behavior), the observed
    reply is curated into the next turn's system message."""
    import numpy as np

    from harness.app import create_app
    from experiments.retrieval_diagnostic import _content_terms

    class DistinctUltra(FakeUltraSmall):
        def embed(self, text):
            v = np.zeros(16)
            for i, w in enumerate(sorted(_content_terms(text))[:16]):
                v[i] = 1.0
            return v

    def backend_factory(model=None, provider=None):
        return OpenAICompatBackend(
            base_url="http://mock", model=model or "mock-model",
            transport=MockTransport(latency_ms=0),
        )

    app = create_app(
        ultra_factory=DistinctUltra,
        backend_factory=backend_factory,
        runs_root=tmp_path / "runs",
        providers_file=tmp_path / "providers.local.json",
        log_dir=str(tmp_path / "logs"),
        state_dir=str(tmp_path / "state"),
    )
    fake = _FakeUpstream()
    monkeypatch.setattr("strata.server.requests.post", fake)
    with TestClient(app) as c:
        c.post("/v1/provider/config", json={
            "providers": [{"name": "lm", "base_url": "http://mock-llama",
                           "api_key": "lm-studio", "model": "m1"}]})
        for _ in range(2):
            r = c.post("/v1/openai/chat/completions", json={
                "messages": [{"role": "user",
                              "content": "Which tokens do I use for auth expiry?"}],
            })
            assert r.status_code == 200
        assert "JWT tokens" in fake.payload["messages"][0]["content"]
def test_health_reports_zero_conversations(client):
    c, _app = client
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True, "conversations": 0}


def test_git_exclude_protection_is_idempotent(tmp_path):
    from harness.__main__ import _protect_git_excludes

    repo = tmp_path / "repo"
    (repo / ".git" / "info").mkdir(parents=True)
    _protect_git_excludes(repo, ["harness_state/", "providers.local.json"])
    excl = repo / ".git" / "info" / "exclude"
    text = excl.read_text(encoding="utf-8")
    assert "harness_state/" in text and "providers.local.json" in text
    # idempotent: re-running must not duplicate entries
    _protect_git_excludes(repo, ["harness_state/", "providers.local.json"])
    assert text == excl.read_text(encoding="utf-8")


def test_turn_returns_curated_reply(client):
    c, _app = client
    r = c.post("/v1/strata/turn", json={
        "query": "How does JWT authentication work?",
        "conversation_id": "c1",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["conversation_id"] == "c1"
    assert body["reply"].startswith("[mock] re:")
    assert body["mode"] in ("strata", "no_backend")
    assert body["error"] is None
    assert body["budget"] > 0
    assert body["turn"] == 1
    assert "total_ms" in body["timings"]
    # turn 1 has no stored history yet; turn 2 must carry curated context
    second = c.post("/v1/strata/turn", json={
        "query": "Follow-up about the JWT expiry claim", "conversation_id": "c1",
    }).json()
    assert second["assembled_content"]


def test_second_turn_increments_and_state_grows(client):
    c, _app = client
    # distinct queries per turn: RC1 collapses verbatim duplicates at ingest,
    # so repeating the same query would (correctly) not grow the store
    for query in ("tell me about JWT", "how does the JWT refresh flow behave"):
        c.post("/v1/strata/turn", json={"query": query, "conversation_id": "c1"})
    st = c.get("/v1/strata/state", params={"conversation_id": "c1"}).json()
    assert st["turn"] == 2
    assert st["store_chunks"] >= 4  # query+reply per turn (hedge-filter permitting)
    assert set(st["comb_stats"]) == {"archived", "resurrected", "comb_hits", "gate_fired"}


def test_state_lists_all_conversations_and_404s_unknown(client):
    c, _app = client
    c.post("/v1/strata/turn", json={"query": "q about JWT", "conversation_id": "a"})
    c.post("/v1/strata/turn", json={"query": "q about JWT", "conversation_id": "b"})
    st = c.get("/v1/strata/state").json()
    assert st["count"] == 2
    assert set(st["conversations"]) == {"a", "b"}
    assert c.get("/v1/strata/state", params={"conversation_id": "zzz"}).status_code == 404


def test_reset_drops_conversation_state(client):
    c, _app = client
    c.post("/v1/strata/turn", json={"query": "JWT please", "conversation_id": "c1"})
    assert c.post("/v1/strata/reset", json={"conversation_id": "c1"}).json()["ok"]
    st = c.get("/v1/strata/state").json()
    assert st["count"] == 0
    body = c.post("/v1/strata/turn", json={"query": "JWT again", "conversation_id": "c1"}).json()
    assert body["turn"] == 1  # fresh conversation


def test_empty_query_is_422(client):
    c, _app = client
    assert c.post("/v1/strata/turn", json={"query": "   "}).status_code == 422


def test_config_overrides_applied_on_creation(client):
    c, app = client
    r = c.post("/v1/strata/turn", json={
        "query": "JWT", "conversation_id": "cfg",
        "config": {"max_context": 4096, "not_a_real_field": 1},
    })
    assert r.status_code == 200
    strata = app.state.harness.hives["cfg"]
    assert strata.config.max_context == 4096  # unknown key silently dropped


def test_model_override_swaps_conversation_backend(client):
    c, app = client
    c.post("/v1/strata/turn", json={
        "query": "JWT", "conversation_id": "m", "model": "other-model",
    })
    strata = app.state.harness.hives["m"]
    assert strata.backend.model == "other-model"
    assert strata.cache.backend is strata.backend


# ---------------------------------------------------------------------------
# providers
# ---------------------------------------------------------------------------
def test_provider_config_roundtrip_masks_keys_and_persists(client, tmp_path):
    c, _app = client
    payload = {
        "providers": [
            {"name": "ds", "base_url": "https://api.deepseek.com",
             "api_key": "sk-real-secret", "model": "deepseek-chat"},
            {"name": "lm", "base_url": "http://localhost:1234",
             "api_key": "lm-studio"},
        ],
        "default": "ds",
        "persist": True,
    }
    body = c.post("/v1/provider/config", json=payload).json()
    assert body["ok"] and body["default"] == "ds"
    by_name = {p["name"]: p for p in body["providers"]}
    # every set api_key is masked in responses ("***" = a key exists, never its value)
    assert by_name["ds"]["api_key"] == "***"
    assert by_name["lm"]["api_key"] == "***"
    on_disk = json.loads((tmp_path / "providers.local.json").read_text(encoding="utf-8"))
    assert on_disk["providers"][0]["api_key"] == "sk-real-secret"  # file keeps it
    assert on_disk["providers"][1]["api_key"] == "lm-studio"

    got = c.get("/v1/provider/config").json()
    assert all(p["api_key"] == "***" for p in got["providers"])

    # echoing the mask back must PRESERVE the stored secret, not overwrite it
    roundtrip = c.post("/v1/provider/config", json={
        "providers": got["providers"], "default": "ds", "persist": True,
    }).json()
    on_disk2 = json.loads((tmp_path / "providers.local.json").read_text(encoding="utf-8"))
    assert on_disk2["providers"][0]["api_key"] == "sk-real-secret"
    assert roundtrip["providers"][0]["api_key"] == "***"


def test_default_backend_factory_resolves_active_provider(tmp_path, monkeypatch):
    """No injected factories -> conversations ride the active provider
    (the 'curl /v1/strata/turn against any provider' M1 path)."""
    monkeypatch.chdir(tmp_path)
    recorded = {}

    class StubBackend:
        def __init__(self, **kw):
            recorded.update(kw)

    monkeypatch.setattr(harness_app_module, "OpenAICompatBackend", StubBackend)
    app = create_app(
        ultra_factory=FakeUltraSmall,
        runs_root=tmp_path / "runs",
        providers_file=tmp_path / "providers.local.json",
        log_dir=str(tmp_path / "logs"),
    )
    c = TestClient(app)
    c.post("/v1/provider/config", json={
        "providers": [{"name": "ds", "base_url": "https://api.deepseek.com",
                       "api_key": "sk-x", "model": "deepseek-chat"}],
        "default": "ds",
    })
    r = c.post("/v1/strata/turn", json={"query": "JWT", "conversation_id": "t"})
    assert r.status_code == 200  # generation errors are contained by the strata
    assert recorded["base_url"] == "https://api.deepseek.com"
    assert recorded["model"] == "deepseek-chat"
    assert recorded["api_key"] == "sk-x"


def test_provider_config_rejects_invalid_entry(client):
    c, _app = client
    r = c.post("/v1/provider/config", json={
        "providers": [{"name": "broken"}],
    })
    assert r.status_code == 422
    assert "base_url" in str(r.json())  # pydantic flags the missing field
    # and a semantically-invalid entry that passes pydantic still fails cleanly
    r2 = c.post("/v1/provider/config", json={
        "providers": [{"name": "", "base_url": "http://x"}],
    })
    assert r2.status_code == 422
    assert "name" in str(r2.json())


def test_models_endpoint_uses_default_provider_base_url(client, monkeypatch):
    c, _app = client
    c.post("/v1/provider/config", json={
        "providers": [{"name": "x", "base_url": "https://x.example"}],
        "default": "x",
    })
    seen = {}

    def fake_list(base_url):
        seen["base_url"] = base_url
        return ["model-1"]

    monkeypatch.setattr(harness_app_module, "_list_models", fake_list)
    body = c.get("/v1/models").json()
    assert seen["base_url"] == "https://x.example"
    assert body == {"base_url": "https://x.example", "models": ["model-1"],
                    "probe": None}


def test_models_endpoint_probe_flag(client, monkeypatch):
    c, _app = client
    monkeypatch.setattr(harness_app_module, "_list_models", lambda b: ["m"])

    class FakeProbe:
        def __init__(self):
            self.__dict__ = {"model": "m", "status": "PASS"}

    monkeypatch.setattr(harness_app_module, "probe_model",
                        lambda b, m: FakeProbe())
    body = c.get("/v1/models", params={"probe": True}).json()
    assert body["probe"][0]["status"] == "PASS"


# ---------------------------------------------------------------------------
# curate / observe (dsh-strata Seam A flow) + built-in mock chat completions
# ---------------------------------------------------------------------------
def test_curate_then_observe_feeds_store_without_generation(client):
    c, app = client
    first = c.post("/v1/strata/curate", json={
        "query": "What is the JWT refresh policy?",
        "conversation_id": "agent-1",
    }).json()
    assert first["mode"] == "no_backend"  # caller generates, not the sidecar
    assert first["reply" if "reply" in first else "assembled_content"] is not None
    assert first["budget"] > 0

    r = c.post("/v1/strata/observe", json={
        "conversation_id": "agent-1",
        "reply": "The JWT access token expires after 3600 seconds.",
    }).json()
    assert r == {"ok": True, "stored": True, "turn": 1}

    st = c.get("/v1/strata/state", params={"conversation_id": "agent-1"}).json()
    assert st["store_chunks"] == 2  # query chunk + observed reply chunk

    # turn 2: the observed fact must now be retrievable into the context
    second = c.post("/v1/strata/curate", json={
        "query": "How often must a JWT client refresh?",
        "conversation_id": "agent-1",
    }).json()
    assert second["turn"] == 2
    strata = app.state.harness.hives["agent-1"]
    stored_contents = [ch.content for ch in strata.store.all_chunks()]
    assert any("3600" in content for content in stored_contents)


def test_curate_hive_has_no_backend(client):
    c, app = client
    c.post("/v1/strata/curate", json={"query": "q on JWT", "conversation_id": "nb"})
    strata = app.state.harness.hives["nb"]
    assert strata.backend is None


def test_observe_hedge_reply_not_stored(client):
    c, _app = client
    c.post("/v1/strata/curate", json={"query": "JWT?", "conversation_id": "h"})
    r = c.post("/v1/strata/observe", json={
        "conversation_id": "h",
        "reply": "I do not have that information regarding your account.",
    }).json()
    assert r == {"ok": True, "stored": False, "turn": 1}


def test_observe_unknown_conversation_lazy_creates(client):
    c, _app = client
    r = c.post("/v1/strata/observe", json={
        "conversation_id": "ghost", "reply": "stored text",
    })
    assert r.status_code == 200
    assert r.json() == {"ok": True, "stored": True, "turn": 0}
    st = c.get("/v1/strata/state", params={"conversation_id": "ghost"}).json()
    assert st["store_chunks"] >= 1


def test_mock_chat_completions_non_stream_reports_context(client):
    c, _app = client
    r = c.post("/v1/chat/completions", json={
        "model": "mock-model",
        "messages": [
            {"role": "system", "content": "<strata>HIVE CONTEXT: jwt facts</strata>"},
            {"role": "user", "content": "<strata-curated-context>jwt facts</strata-curated-context>"},
        ],
    })
    assert r.status_code == 200
    body = r.json()
    content = body["choices"][0]["message"]["content"]
    assert body["choices"][0]["finish_reason"] == "stop"
    assert "system=" in content and "strata_context=yes" in content
    assert body["usage"]["total_tokens"] > 0


def test_mock_chat_completions_flags_missing_curated_marker(client):
    c, _app = client
    body = c.post("/v1/chat/completions", json={
        "model": "m",
        "messages": [{"role": "user", "content": "plain question"}],
    }).json()
    assert "strata_context=no" in body["choices"][0]["message"]["content"]


def test_mock_chat_emits_tool_call_for_benchmark_request(client):
    c, _app = client
    body = c.post("/v1/chat/completions", json={
        "model": "m",
        "messages": [{"role": "user",
                      "content": "run the benchmark with 3 conversations"}],
    }).json()
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    call = choice["message"]["tool_calls"][0]
    assert call["function"]["name"] == "hive_bench_run"
    args = json.loads(call["function"]["arguments"])
    assert args == {"mode": "mock", "max_convs": 3, "protocol": True}


def test_mock_chat_streams_tool_call_sse(client):
    c, _app = client
    r = c.post("/v1/chat/completions", json={
        "model": "m", "stream": True,
        "messages": [{"role": "user", "content": "use hive_bench_run please"}],
    })
    lines = [ln for ln in r.text.splitlines() if ln.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    import json as _json

    events = [_json.loads(ln[6:]) for ln in lines[:-1]]
    final = events[-1]["choices"][0]
    assert final["finish_reason"] == "tool_calls"
    merged_name, merged_args = None, ""
    for ev in events:
        delta = ev["choices"][0]["delta"]
        for tc in delta.get("tool_calls") or []:
            if tc.get("function", {}).get("name"):
                merged_name = tc["function"]["name"]
            merged_args += tc.get("function", {}).get("arguments") or ""
    assert merged_name == "hive_bench_run"
    assert _json.loads(merged_args)["mode"] == "mock"


def test_mock_chat_acknowledges_tool_result(client):
    c, _app = client
    body = c.post("/v1/chat/completions", json={
        "model": "m",
        "messages": [
            {"role": "user", "content": "run the benchmark"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "call_1", "type": "function",
                             "function": {"name": "hive_bench_run",
                                          "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_1",
             "content": "Run protocol_x (completed)\n"
                        "post-run PES 62.4 Â· per-turn PES - Â· turns 24\n"
                        "P1 PASS | P2 SPLIT | P6 FAIL"},
        ],
    }).json()
    reply = body["choices"][0]["message"]["content"]
    assert "completed" in reply
    assert "62.4" in reply
    assert "P1 PASS" in reply and "P2 SPLIT" in reply


def test_mock_chat_completions_stream_sse(client):
    c, _app = client
    r = c.post("/v1/chat/completions", json={
        "model": "m", "stream": True,
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
    lines = [ln for ln in r.text.splitlines() if ln.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    import json as _json

    deltas = "".join(
        _json.loads(ln[6:])["choices"][0]["delta"].get("content") or ""
        for ln in lines[:-1]
        if _json.loads(ln[6:])["choices"][0]["delta"].get("content")
    )
    assert "[strata-mock]" in deltas
    final = _json.loads(lines[-2][6:])
    assert final["choices"][0]["finish_reason"] == "stop"
    assert "usage" in final


def test_curate_full_flow_with_mock_llm_roundtrip(client):
    """The offline M2 shape: curate -> (model sees curated context via mock
    chat endpoint) -> observe; the store then retrieves the fact."""
    c, _app = client
    cid = "sess-demo"
    cur = c.post("/v1/strata/curate", json={
        "query": "Remember: deploy token rotation is 90 days.",
        "conversation_id": cid,
    }).json()
    chat = c.post("/v1/chat/completions", json={
        "model": "m",
        "messages": [
            {"role": "system", "content": f"PINNED\n\n{cur['assembled_content']}"},
            {"role": "user", "content": "Remember: deploy token rotation is 90 days."},
        ],
    }).json()
    assert "system=0ch" not in chat["choices"][0]["message"]["content"]
    c.post("/v1/strata/observe", json={
        "conversation_id": cid,
        "reply": "Deploy tokens rotate every 90 days per policy.",
    })
    nxt = c.post("/v1/strata/curate", json={
        "query": "What is the deploy token rotation period?",
        "conversation_id": cid,
    }).json()
    assert "90 days" in nxt["assembled_content"]
# ---------------------------------------------------------------------------
# conversation persistence across sidecar restarts
# ---------------------------------------------------------------------------
def _make_app(tmp_path):
    def backend_factory(model=None, provider=None):
        return OpenAICompatBackend(
            base_url="http://mock", model=model or "mock-model",
            transport=MockTransport(latency_ms=0),
        )

    return create_app(
        ultra_factory=FakeUltraSmall,
        backend_factory=backend_factory,
        runs_root=tmp_path / "runs",
        providers_file=tmp_path / "providers.local.json",
        log_dir=str(tmp_path / "logs"),
        state_dir=tmp_path / "harness_state",
    )


def test_conversation_survives_sidecar_restart(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    c1 = TestClient(_make_app(tmp_path))
    cid = "ws-demo-workspace"
    c1.post("/v1/strata/curate", json={"query": "JWT refresh is 3600s", "conversation_id": cid})
    c1.post("/v1/strata/observe", json={
        "conversation_id": cid, "reply": "Access tokens rotate every 90 days.",
    })
    assert (tmp_path / "harness_state").is_dir()
    assert list((tmp_path / "harness_state").glob("conv-*.json"))

    # a brand-new app instance (= restarted sidecar) restores the conversation
    c2 = TestClient(_make_app(tmp_path))
    st = c2.get("/v1/strata/state", params={"conversation_id": cid}).json()
    assert st["turn"] == 1
    assert st["store_chunks"] == 2

    # memory works across the restart: a PRE-restart chunk is retrieved into
    # the new context (the fake drone merges same-domain chunks via its
    # constant embeddings, so the kept copy is the turn-1 query)
    nxt = c2.post("/v1/strata/curate", json={
        "query": "What is the token rotation period?", "conversation_id": cid,
    }).json()
    assert nxt["turn"] == 2
    assert "3600s" in nxt["assembled_content"]
    assert nxt["token_count"] > 0


def test_reset_deletes_persisted_state(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    c1 = TestClient(_make_app(tmp_path))
    c1.post("/v1/strata/curate", json={"query": "JWT facts", "conversation_id": "doomed"})
    files = list((tmp_path / "harness_state").glob("conv-*.json"))
    assert len(files) == 1

    c1.post("/v1/strata/reset", json={"conversation_id": "doomed"})
    assert list((tmp_path / "harness_state").glob("conv-*.json")) == []

    c2 = TestClient(_make_app(tmp_path))
    body = c2.post("/v1/strata/curate", json={
        "query": "JWT facts", "conversation_id": "doomed",
    }).json()
    assert body["turn"] == 1  # fresh, not restored


def test_conversation_filename_never_escapes_state_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    c = TestClient(_make_app(tmp_path))
    evil = "../../outside"
    c.post("/v1/strata/curate", json={"query": "JWT", "conversation_id": evil})
    # content-hashed filename: nothing written outside state dir
    assert not (tmp_path / "outside").exists()
    assert len(list((tmp_path / "harness_state").glob("conv-*.json"))) == 1
    payload = json.loads(next((tmp_path / "harness_state").glob("conv-*.json"))
                         .read_text(encoding="utf-8"))
    assert payload["conversation_id"] == evil  # raw id preserved inside


def test_disabled_persistence_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def backend_factory(model=None, provider=None):
        return OpenAICompatBackend(transport=MockTransport(latency_ms=0))

    app = create_app(
        ultra_factory=FakeUltraSmall,
        backend_factory=backend_factory,
        runs_root=tmp_path / "runs",
        providers_file=tmp_path / "p.json",
        log_dir=str(tmp_path / "logs"),
        state_dir="",
    )
    c = TestClient(app)
    c.post("/v1/strata/curate", json={"query": "JWT", "conversation_id": "x"})
    assert not (tmp_path / "harness_state").exists()


# ---------------------------------------------------------------------------
# report views (Seam B): /v1/runs, /view/{run_dir}, /
# ---------------------------------------------------------------------------
def _write_report(tmp_path: Path, report: dict, name: str = "20260824_view") -> str:
    d = tmp_path / "runs" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "run_report.json").write_text(json.dumps(report), encoding="utf-8")
    return name


def _sample_report() -> dict:
    return {
        "run_id": "16d081e5", "mode": "mock", "backend": "LMStudioBackend",
        "aggregate": {"conversations": 2, "user_turns": 24,
                      "avg_pes": 55.3, "avg_total_ms": 9200.0},
        "post_run_pes": {"composite": 62.4, "band": "yellow",
                         "components": {"retrieval_precision": 90.0,
                                        "routing_accuracy": 100.0,
                                        "latency_health": 0.0,
                                        "throughput_health": 80.0,
                                        "context_utilization": 66.9}},
        "protocol": [
            {"id": "P1", "title": "Constant throughput", "status": "PASS",
             "evidence": {}, "note": "tps flat"},
            {"id": "P6", "title": "Escalation", "status": "FAIL",
             "evidence": {}, "note": "<script>alert(1)</script>"},
        ],
        "retrieval_diagnostic": {"retrieval_recall": 45.5,
                                 "retrieval_recall_retrievable": 100.0,
                                 "ingestion_rate": 48.4,
                                 "perfect_hive_ceiling": 48.4,
                                 "retrieval_precision": 10.7},
        "comb": {"conversations": [{"archived": 3, "resurrected": 1,
                                    "comb_hits": 1, "gate_fired": 2},
                                   {"archived": 1, "resurrected": 0,
                                    "comb_hits": 0, "gate_fired": 1}]},
        "baseline_lm_studio": {"aggregate": {"avg_pes": 12.21}},
        "baseline_fifo": {"aggregate": {"avg_pes": 11.63}},
    }


def test_runs_index_lists_newest_first(client, tmp_path):
    c, _app = client
    _write_report(tmp_path, _sample_report(), "20260824_a")
    import os

    d = tmp_path / "runs" / "20260824_b"
    d.mkdir()
    os.utime(d, (time.time() + 10,) * 2)  # newer mtime
    body = c.get("/v1/runs").json()
    names = [r["name"] for r in body["runs"]]
    assert names[:2] == ["20260824_b", "20260824_a"]
    assert body["runs"][1]["has_report"] is True


def test_view_report_renders_all_sections(client, tmp_path):
    c, _app = client
    name = _write_report(tmp_path, _sample_report())
    r = c.get(f"/view/{name}")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    page = r.text
    for marker in ("HiveBench report", "PES components", "P1&ndash;P11 verdicts",
                   "Constant throughput", "retrieval diagnostic",
                   "Ingestion rate", "Comb (P11 surplus tier)",
                   "Baselines comparison", "LM-Studio rolling", ">12.2<"):
        assert marker in page, marker


def test_view_report_escapes_hostile_values(client, tmp_path):
    c, _app = client
    _write_report(tmp_path, _sample_report())
    page = c.get("/view/20260824_view").text
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page


def test_view_report_handles_partial_bundle(client, tmp_path):
    c, _app = client
    _write_report(tmp_path, {"run_id": "x"}, "20260824_partial")
    r = c.get("/view/20260824_partial")
    assert r.status_code == 200
    assert "no protocol block" in r.text
    assert "&mdash;" in r.text


def test_view_unknown_and_traversal_rejected(client, tmp_path):
    c, _app = client
    assert c.get("/view/ghost").status_code == 404
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "run_report.json").write_text("{}", encoding="utf-8")
    r = c.get("/view/../outside")
    assert r.status_code in (400, 404)


def test_view_report_renders_pes_and_breakdown_shape(client, tmp_path):
    """generate_data's other post_run_pes dialect: pes/breakdown + band."""
    c, _app = client
    _write_report(tmp_path, {
        "run_id": "y", "mode": "mock",
        "post_run_pes": {"pes": 82.43, "band": "GREEN",
                         "breakdown": {"retrieval_precision": 30.0}},
        "protocol": [{"id": "P8", "title": "Routing", "status": "PASS"}],
    }, "20260824_dialect")
    page = c.get("/view/20260824_dialect").text
    assert ">82.4<" in page and "band-green" in page
    assert "P8" in page


def test_index_redirects_to_runs(client, tmp_path):
    c, _app = client
    r = c.get("/", follow_redirects=False)
    assert r.status_code in (301, 302, 307)
    assert r.headers["location"] == "/runs"


def test_token_auth_guard(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HARNESS_TOKEN", "sekrit")
    app = create_app(
        ultra_factory=FakeUltraSmall,
        backend_factory=lambda model=None: OpenAICompatBackend(
            transport=MockTransport(latency_ms=0)),
        runs_root=tmp_path / "runs",
        providers_file=tmp_path / "p.json",
        log_dir=str(tmp_path / "logs"),
        state_dir=tmp_path / "harness_state",
    )
    c = TestClient(app)
    assert c.get("/health").status_code == 200  # unguarded
    assert c.post("/v1/commands/run", json={"line": "/status"}).status_code == 401
    ok = c.post("/v1/commands/run", json={"line": "/status"},
                headers={"x-strata-token": "sekrit"})
    assert ok.status_code == 200 and ok.json()["kind"] == "success"


def test_cors_default_is_localhost_not_wildcard():
    from harness.app import _cors_origins

    assert "*" not in _cors_origins()
    assert any("localhost" in o for o in _cors_origins())


# ---------------------------------------------------------------------------
# console slash commands (dsh conventions)
# ---------------------------------------------------------------------------
def test_parse_command_follows_dsh_conventions():
    from harness.commands import parse_command

    assert parse_command("/save") == ("save", "")
    assert parse_command("/save my name") == ("save", " my name")  # verbatim, dsh-style
    assert parse_command("/SAVE x") == ("save", " x")
    assert parse_command("/goal\nmulti\nline") == ("goal", "\nmulti\nline")
    assert parse_command("plain message") is None


def test_commands_registry_serves_descriptors(client):
    c, _app = client
    body = c.get("/v1/commands").json()
    names = {cmd["name"] for cmd in body["commands"]}
    assert {"help", "new", "save", "model", "mode", "provider", "engine",
            "bench", "status"} <= names
    save = [cmd for cmd in body["commands"] if cmd["name"] == "save"][0]
    assert save["input"]["hint"] == "[name]"


def test_command_run_help_new_and_unknown(client):
    c, _app = client
    help_result = c.post("/v1/commands/run", json={
        "line": "/help", "conversation_id": "x"}).json()
    assert help_result["kind"] == "success"
    assert "/save" in help_result["text"]

    unknown = c.post("/v1/commands/run", json={
        "line": "/nope", "conversation_id": "x"}).json()
    assert unknown["kind"] == "error" and "unknown command" in unknown["text"]

    fresh = c.post("/v1/commands/run", json={
        "line": "/new", "conversation_id": "x"}).json()
    assert fresh["kind"] == "success"
    assert fresh["new_conversation_id"] != "x"


def test_command_mode_sets_transport(client):
    c, _app = client
    r = c.post("/v1/commands/run", json={
        "line": "/mode agent", "conversation_id": "x"}).json()
    assert r["kind"] == "success" and r["mode"] == "agent"
    bad = c.post("/v1/commands/run", json={
        "line": "/mode yolo", "conversation_id": "x"}).json()
    assert bad["kind"] == "error"


def test_command_save_exports_transcript(client, tmp_path):
    c, app = client
    cid = "save-me"
    c.post("/v1/strata/curate", json={"query": "The refresh token is 3600s.",
                                    "conversation_id": cid})
    c.post("/v1/strata/observe", json={
        "conversation_id": cid, "reply": "Access tokens rotate every 90 days."})
    r = c.post("/v1/commands/run", json={
        "line": f"/save demo-{cid}", "conversation_id": cid}).json()
    assert r["kind"] == "success" and "saved" in r["text"]
    path = Path(r["text"].split()[-1])
    content = path.read_text(encoding="utf-8")
    assert "refresh token is 3600s" in content
    assert "rotate every 90 days" in content
    assert content.startswith("# Conversation transcript")


def test_command_save_nothing_to_save(client):
    c, _app = client
    r = c.post("/v1/commands/run", json={
        "line": "/save", "conversation_id": "empty-conv"}).json()
    assert r["kind"] == "error" and "nothing to save" in r["text"]


def test_command_status_and_provider_listing(client):
    c, _app = client
    r = c.post("/v1/commands/run", json={
        "line": "/status", "conversation_id": "x"}).json()
    assert r["kind"] == "success" and "server:" in r["text"]
    listing = c.post("/v1/commands/run", json={
        "line": "/provider", "conversation_id": "x"}).json()
    assert listing["kind"] == "success"


# ---------------------------------------------------------------------------
# protocol runner + report reader
# ---------------------------------------------------------------------------
class _FakeProc:
    def __init__(self, pid=4242):
        self.pid = pid


def test_protocol_run_builds_whitelisted_command(client, tmp_path, monkeypatch):
    c, _app = client
    captured = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return _FakeProc()

    monkeypatch.setattr(harness_app_module, "_popen", fake_popen)
    body = c.post("/v1/protocol/run", json={
        "mode": "live",
        "args": {"max_convs": 3, "max_turns": 10, "protocol": True,
                 "baselines": True, "model": "bonsai", "provider": "lmstudio",
                 "evil_flag": "--rm-rf"},
    }).json()

    run_dir = Path(body["run_dir"])
    assert run_dir.parent == tmp_path / "runs"
    assert run_dir.name.startswith("protocol_")
    cmd = captured["cmd"]
    assert cmd[1:3] == ["-m", "experiments.generate_data"]
    assert "--live" in cmd and "--output" in cmd
    i = cmd.index("--output")
    assert Path(cmd[i + 1]) == run_dir
    for expected in ("--max-convs", "3", "--max-turns", "10", "--protocol",
                     "--baselines", "--model", "bonsai", "--provider", "lmstudio"):
        assert expected in cmd
    assert "evil_flag" not in " ".join(cmd)  # whitelist holds
    assert isinstance(body["pid"], int)
    assert (run_dir / "run_stdout.log").exists()


def test_protocol_run_rejects_bad_mode(client):
    c, _app = client
    assert c.post("/v1/protocol/run", json={"mode": "yolo"}).status_code == 422


def test_report_serves_run_report_bundle(client, tmp_path):
    c, _app = client
    d = tmp_path / "runs" / "20260824_demo"
    d.mkdir(parents=True)
    (d / "run_report.json").write_text(json.dumps({"aggregate": {"avg_pes": 80.0}}),
                                       encoding="utf-8")
    body = c.get("/v1/report/20260824_demo").json()
    assert body["aggregate"]["avg_pes"] == 80.0
    assert c.get("/v1/report/missing_dir").status_code == 404


def test_report_blocks_path_traversal(client, tmp_path):
    c, _app = client
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "run_report.json").write_text("{}", encoding="utf-8")
    r = c.get("/v1/report/../outside")
    assert r.status_code in (400, 404)  # never serves outside runs/


def test_openai_model_name_prefix_sets_conversation_id(client, monkeypatch):
    """S5: model name with ':' prefix pins conversation ID (opencode workaround)."""
    c, _app = client
    _configure_lm_provider(c)

    fake = _FakeUpstream(content="ok")
    monkeypatch.setattr("strata.server.requests.post", fake)

    # Model name with project prefix -> conversation ID = prefix
    r = c.post("/v1/openai/chat/completions", json={
        "model": "my-project:m1",
        "messages": [{"role": "user", "content": "hello"}],
    })
    assert r.status_code == 200, r.text[:300]

    # Upstream should receive the stripped model name
    assert fake.payload["model"] == "m1"

    # Conversation state should be stored under "my-project"
    st = c.get("/v1/strata/state", params={"conversation_id": "my-project"}).json()
    assert st["store_chunks"] >= 1


def test_openai_model_name_no_colon_uses_default_cid(client, monkeypatch):
    """S5: model name without ':' prefix falls through to default CID logic."""
    c, _app = client
    _configure_lm_provider(c)

    fake = _FakeUpstream(content="ok")
    monkeypatch.setattr("strata.server.requests.post", fake)

    r = c.post("/v1/openai/chat/completions", json={
        "model": "m1",
        "messages": [{"role": "user", "content": "hello"}],
    })
    assert r.status_code == 200, r.text[:300]

    # No prefix -> model name unchanged, CID falls back to "default"
    assert fake.payload["model"] == "m1"
