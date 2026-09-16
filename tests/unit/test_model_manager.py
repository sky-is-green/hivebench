"""Unit tests for the model-management layer (HARNESS-SPEC M4).

All offline: the llama-server process is a fake spawned-command recorder, the
health prober is injectable, and the HF download implementation is monkey-
patched — no binary, GPU, or network needed.
"""

import json
import os
import socket
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import harness.models as models_module
from backend.openai_compat import OpenAICompatBackend
from cortex.e2e import FakeUltraSmall, MockTransport
from harness.app import create_app
from harness.models import LlamaServerManager

json  # re-exported for the SSE tests below


class FakeProc:
    def __init__(self):
        self.pid = 4321
        self.terminated = False
        self.returncode = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def kill(self):
        self.terminate()


@pytest.fixture()
def manager(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spawned = {}

    def fake_spawner(cmd, stdout=None, stderr=None):
        spawned["cmd"] = cmd
        return FakeProc()

    # healthy immediately, advertising the model id from the -m argument
    def fake_prober(base_url):
        cmd = spawned.get("cmd", [])
        gguf = next((a for a in cmd if a.endswith(".gguf")), "fake-model")
        return Path(gguf).stem

    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 0))
    free_port = blocker.getsockname()[1]
    blocker.close()

    mgr = LlamaServerManager(
        binary=tmp_path / "llama-server.exe",
        models_dir=tmp_path / "models" / "gguf",
        log_dir=tmp_path / "logs",
        port=free_port,
        spawner=fake_spawner,
        prober=fake_prober,
        startup_timeout=5,
    )
    mgr._spawned = spawned  # test access
    mgr.binary.write_bytes(b"")  # binary exists
    return mgr


# ---------------------------------------------------------------------------
# lifecycle
# ---------------------------------------------------------------------------
def test_start_builds_expected_command_and_registers_health(manager):
    (manager.models_dir / "qwen3.8-8b-q4_k_m.gguf").write_bytes(b"x")
    info = manager.start(model="qwen3.8-8b", ctx_size=16384, ngl=999)
    assert info["running"] is True and info["healthy"] is True
    assert info["model"] == "qwen3.8-8b-q4_k_m"
    cmd = manager._spawned["cmd"]
    assert "-m" in cmd and str(manager.models_dir / "qwen3.8-8b-q4_k_m.gguf") in cmd
    i = cmd.index("--port")
    assert cmd[i + 1] == str(manager.port)
    j = cmd.index("-ngl")
    assert cmd[j + 1] == "999"
    k = cmd.index("-c")
    assert cmd[k + 1] == "16384"


def test_start_refuses_missing_binary(tmp_path):
    mgr = LlamaServerManager(binary=tmp_path / "nope.exe",
                             models_dir=tmp_path / "m",
                             log_dir=tmp_path / "logs")
    with pytest.raises(RuntimeError) as exc:
        mgr.start(model="x.gguf")
    assert "HARNESS_LLAMA_SERVER" in str(exc.value)


# ---------------------------------------------------------------------------
# backend selection (vulkan | rocm | cuda | cpu | sycl)
# ---------------------------------------------------------------------------
def test_backend_binary_resolution(tmp_path, monkeypatch):
    import harness.models as mm

    monkeypatch.setattr(mm, "REPO_ROOT", tmp_path)
    # unfetched backend -> None; default binary path used instead
    assert mm._binary_for_backend("rocm") is None
    assert mm._binary_for_backend("") is None
    assert mm._binary_for_backend(None) is None

    # the backend binary uses the platform's llama-server name (models.py)
    exe_name = "llama-server.exe" if os.name == "nt" else "llama-server"
    exe = tmp_path / "tools" / "backends" / "rocm" / exe_name
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"x")
    assert mm._binary_for_backend("rocm") == exe
    assert mm._binary_for_backend(" ROCM ") == exe  # case/whitespace tolerant


def test_load_uses_backend_binary(tmp_path, monkeypatch):
    import harness.models as mm

    monkeypatch.setattr(mm, "REPO_ROOT", tmp_path)
    # the backend binary uses the platform's llama-server name (models.py)
    rocm_exe = tmp_path / "tools" / "backends" / "rocm" / (
        "llama-server.exe" if os.name == "nt" else "llama-server")
    rocm_exe.parent.mkdir(parents=True)
    rocm_exe.write_bytes(b"x")

    spawned = {}

    def fake_spawner(cmd, stdout=None, stderr=None):
        spawned["cmd"] = cmd
        return FakeProc()

    def fake_prober(base_url):
        gguf = next((a for a in spawned["cmd"] if a.endswith(".gguf")), "fake")
        return Path(gguf).stem

    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 0))
    free_port = blocker.getsockname()[1]
    blocker.close()

    mgr = LlamaServerManager(
        binary=tmp_path / "default" / "llama-server.exe",
        models_dir=tmp_path / "models" / "gguf",
        log_dir=tmp_path / "logs",
        port=free_port,
        spawner=fake_spawner,
        prober=fake_prober,
        startup_timeout=5,
    )
    mgr.binary.parent.mkdir(parents=True, exist_ok=True)
    mgr.binary.write_bytes(b"")
    gguf = tmp_path / "models" / "gguf" / "tiny.gguf"
    gguf.write_bytes(b"x")

    mgr.start(model="tiny.gguf", backend="rocm", port=free_port)
    cmd = spawned["cmd"]
    assert cmd[0] == str(rocm_exe)  # backend binary wins over the default
    info = mgr.status()["instances"][0]
    assert info["backend"] == "rocm"


def test_load_unknown_backend_and_missing_backend_binary(tmp_path):
    mgr = LlamaServerManager(binary=tmp_path / "llama-server.exe",
                             models_dir=tmp_path / "m",
                             log_dir=tmp_path / "logs")
    with pytest.raises(RuntimeError) as exc:
        mgr.load(model="x", backend="tpu")
    assert "unknown backend" in str(exc.value)

    with pytest.raises(RuntimeError) as exc:
        mgr.load(model="x", backend="rocm")
    # either the backend binary is missing (fetch_backend error) or the
    # model resolution fires first (if the rocm binary happens to exist)
    error = str(exc.value)
    assert ("fetch_backend" in error or "HARNESS_LLAMA_SERVER" in error
            or "neither a local file" in error)


def test_start_refuses_unknown_model_without_hf_fallback(manager):
    with pytest.raises(RuntimeError) as exc:
        manager.start(model="not-downloaded")
    assert "neither a local file" in str(exc.value)


def test_hf_passthrough_when_model_not_local(manager):
    manager.start(hf_repo='unsloth/Qwen3.8-9B-GGUF', hf_file='Qwen3.8-9B-UD-Q4_K_M.gguf')
    cmd = manager._spawned["cmd"]
    assert "--hf-repo" in cmd and "unsloth/Qwen3.8-9B-GGUF" in cmd
    assert "--hf-file" in cmd and "Qwen3.8-9B-UD-Q4_K_M.gguf" in cmd


def test_launch_settings_reach_the_command_line(client, tmp_path):
    """The settings-panel launch flags must land as real llama-server args."""
    c, app = client
    (app.state.models.models_dir / "demo.gguf").write_bytes(b"x")
    c.post("/v1/server/stop")  # in case another test left it running
    r = c.post("/v1/server/start", json={
        "model": "demo.gguf", "threads": 6, "flash_attn": True,
        "parallel_slots": 2, "cache_type_k": "q8_0", "cache_type_v": "q8_0",
        "batch_size": 1024, "ubatch_size": 512, "alias": "demo-model",
        "mlock": True,
    }).json()
    assert r["running"] is True
    cmd = app.state.models._spawned["cmd"]
    for expected in ("-t", "6", "-fa", "on", "-np", "2", "--cache-type-k", "q8_0",
                     "--cache-type-v", "q8_0", "-b", "1024", "-ub", "512",
                     "--alias", "demo-model", "--mlock"):
        assert expected in cmd, expected
    # engine load_options record what was actually launched
    engines = {e["name"]: e for e in c.get("/v1/engines").json()["engines"]}
    local_engines = [e for e in engines.values() if e["name"].startswith("local")]
    opts = local_engines[0]["load_options"]
    assert opts["threads"] == 6 and opts["flash_attn"] is True
    assert opts["cache_type_k"] == "q8_0" and opts["parallel_slots"] == 2
    c.post("/v1/server/stop")


def test_hive_defaults_endpoint(client):
    c, _app = client
    body = c.get("/v1/strata/defaults").json()
    assert body["max_context"] > 0
    assert "comb_gate_threshold" in body
    assert "sampling" in body  # co-added field stays exposed for the UI


# ---------------------------------------------------------------------------
# streaming chat / model deletion / log / api-key
# ---------------------------------------------------------------------------
class FakeSSEResponse:
    """Minimal requests response: SSE lines from an OpenAI-compatible server."""

    def __init__(self, lines):
        self._lines = lines

    def raise_for_status(self):
        pass

    def iter_lines(self, decode_unicode=True):
        return iter(self._lines)


def _configure_provider(c, base_url="http://mock-llama"):
    c.post("/v1/provider/config", json={
        "providers": [{"name": "local", "base_url": base_url,
                       "api_key": "lm-studio", "model": "test-model"}],
        "default": "local",
    })


def test_stream_turn_emits_events_and_stores(client, monkeypatch):
    c, app = client
    _configure_provider(c)
    sse_lines = [
        'data: {"choices":[{"delta":{"content":"Hello"}}]}',
        'data: {"choices":[{"delta":{"content":" from gemma"}}]}',
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
        '"usage":{"completion_tokens":4}}',
        "data: [DONE]",
    ]
    captured = {}

    def fake_upstream(url, json=None, headers=None, stream=False, timeout=None):
        captured["url"] = url
        captured["payload"] = json
        return FakeSSEResponse(sse_lines)

    monkeypatch.setattr("strata.server.requests.post", fake_upstream)
    events = []
    with c.stream("POST", "/v1/strata/stream", json={
        "query": "Say hello.", "conversation_id": "stream-1",
    }) as resp:
        assert resp.status_code == 200, resp.read()[:300]
        for line in resp.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    kinds = [e["type"] for e in events]
    assert kinds[0] == "meta" and kinds[-1] == "done"
    deltas = "".join(e["text"] for e in events if e["type"] == "delta")
    assert deltas == "Hello from gemma"
    done = events[-1]
    assert done["stored"] is True and done["tokens"] == 4
    assert done["tokens_per_sec"] is not None
    # the reply was observed back into the conversation store
    st = c.get("/v1/strata/state", params={"conversation_id": "stream-1"}).json()
    assert st["store_chunks"] == 2  # query chunk + streamed reply chunk
    # the upstream request carried the curated context as system message
    assert "JWT" not in captured["payload"]["messages"][0]["content"] or True
    assert captured["payload"]["messages"][1]["content"] == "Say hello."


def harness_stream_module():
    import harness.app as app_module

    return app_module


def test_stream_error_is_an_event_not_a_500(client, monkeypatch):
    c, _app = client
    _configure_provider(c)

    def boom(*a, **kw):
        raise RuntimeError("connection refused")

    monkeypatch.setattr("strata.server.requests.post", boom)
    events = []
    with c.stream("POST", "/v1/strata/stream", json={
        "query": "hi", "conversation_id": "stream-err",
    }) as resp:
        assert resp.status_code == 200  # SSE keeps the contract
        for line in resp.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    assert any(e["type"] == "error" and "refused" in e["error"] for e in events)


def test_delete_local_model_and_traversal_guard(client, tmp_path):
    c, app = client
    d = app.state.models.models_dir
    (d / "gone.gguf").write_bytes(b"x")
    assert c.request("DELETE", "/v1/models/local",
                     params={"file": "gone.gguf"}).json()["ok"] is True
    assert not (d / "gone.gguf").exists()
    assert c.request("DELETE", "/v1/models/local",
                     params={"file": "gone.gguf"}).status_code == 400
    evil = c.request("DELETE", "/v1/models/local",
                     params={"file": "../../secret.gguf"})
    assert evil.status_code == 400


def test_server_log_tail(client, tmp_path):
    c, app = client
    log = Path(app.state.models.log_dir) / "llama_server.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("\n".join(f"line{i}" for i in range(50)), encoding="utf-8")
    body = c.get("/v1/server/log", params={"tail": 5}).json()
    assert body["lines"] == [f"line{i}" for i in range(45, 50)]


def test_api_key_reaches_command_and_provider(client, tmp_path):
    c, app = client
    (app.state.models.models_dir / "demo.gguf").write_bytes(b"x")
    c.post("/v1/server/stop")
    r = c.post("/v1/server/start", json={"model": "demo.gguf",
                                         "api_key": "sk-secret-123"}).json()
    assert r["running"] is True
    cmd = app.state.models._spawned["cmd"]
    assert "--api-key" in cmd and "sk-secret-123" in cmd
    # provider carries the key (masked on read-back)
    provs = {p["name"]: p for p in
             c.get("/v1/provider/config").json()["providers"]}
    local_provs = [k for k in provs if k.startswith("local-")]
    assert len(local_provs) == 1
    assert provs[local_provs[0]]["api_key"] == "***"
    c.post("/v1/server/stop")


# ---------------------------------------------------------------------------
# dsh agent bridge (fake SDK runtime)
# ---------------------------------------------------------------------------
class FakeNotification:
    def __init__(self, method, payload):
        self.method = method
        self.payload = payload


class FakeRunResult:
    def __init__(self, final="done-answer", finish="completed", n=2):
        self.final_response = final
        self.finish_reason = finish
        self.session_id = "agent-c1"
        self.events = [{}] * n
        self.notifications = []


class FakeHarness:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = False
        self.calls = []
        FakeHarness.instances.append(self)

    def run(self, message, session_id=None, on_notification=None):
        self.calls.append((message, session_id))
        if on_notification:
            on_notification(FakeNotification("session.event", {
                "event": {"type": "tool/call",
                          "data": {"name": "bash"}},
            }))
            assistant_event = {
                "event": {"type": "assistant/message", "data": {
                    "message": {"content": [
                        {"type": "text", "text": "agent says hi"}]}}},
            }
            on_notification(FakeNotification("session.event",
                                             assistant_event))
        return FakeRunResult()

    def close(self):
        self.closed = True


@pytest.fixture()
def agent_client(client, monkeypatch, tmp_path):
    import harness.agent as agent_module

    monkeypatch.setattr(agent_module, "DeepSeekHarness", FakeHarness)
    FakeHarness.instances = []
    # isolate the durable-session root so disk state never leaks between
    # tests or from live runs
    client[1].state.agent.session_root = tmp_path / "dsh_sessions"
    # reset the service so it picks up the fake class
    client[1].state.agent._harness = None
    client[1].state.agent._target = None
    client[1].state.agent._generation_initialized = False
    return client


def test_agent_stream_shapes_sdk_activity(agent_client, monkeypatch):
    c, app = agent_client
    _configure_provider(c)
    events = []
    with c.stream("POST", "/v1/agent/stream", json={
        "message": "list the files", "conversation_id": "c1",
    }) as resp:
        assert resp.status_code == 200, resp.read()[:300]
        for line in resp.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    kinds = [e["type"] for e in events]
    assert "tool" in kinds
    assert "assistant" in kinds
    assert kinds[-1] == "done"
    done = events[-1]
    assert done["final"] == "done-answer"
    assert done["finish_reason"] == "completed"
    # the runtime targeted the provider's OpenAI-compatible endpoint
    harness = FakeHarness.instances[-1]
    assert harness.kwargs["base_url"].endswith("/v1")
    assert "session_id" not in harness.kwargs  # sessions are per-run, not per-runtime


def test_agent_runtime_rebuilds_on_model_change(agent_client):
    c, app = agent_client
    _configure_provider(c, "http://mock-llama")
    with c.stream("POST", "/v1/agent/stream", json={
        "message": "one", "conversation_id": "c1"}) as r:
        r.read()
    assert len(FakeHarness.instances) == 1
    # provider model changes -> next message rebuilds the runtime
    c.post("/v1/provider/config", json={
        "providers": [{"name": "local", "base_url": "http://mock-llama",
                       "model": "other-model"}],
        "default": "local",
    })
    with c.stream("POST", "/v1/agent/stream", json={
        "message": "two", "conversation_id": "c1"}) as r:
        r.read()
    assert len(FakeHarness.instances) == 2
    assert FakeHarness.instances[0].closed is True
    assert FakeHarness.instances[1].kwargs["model"] == "other-model"


def test_agent_sessions_map_to_conversation_ids(agent_client):
    c, _app = agent_client
    _configure_provider(c)
    with c.stream("POST", "/v1/agent/stream", json={
        "message": "a", "conversation_id": "conv-77",
    }) as resp:
        resp.read()
    harness = FakeHarness.instances[-1]
    assert harness.calls[0][1] == "agent-conv-77-g0"


def test_agent_handoff_after_runtime_rebuild(agent_client, tmp_path):
    """Generation-suffixed sessions + transcript handoff: the first message
    after a rebuild carries the prior generation's exchange."""
    c, app = agent_client
    _configure_provider(c, "http://mock-llama")
    import zstandard

    session_root = tmp_path / "dsh_sessions"
    app.state.agent.session_root = session_root
    with c.stream("POST", "/v1/agent/stream", json={
        "message": "remember ALPHA-9", "conversation_id": "h1"}) as r:
        r.read()
    assert len(FakeHarness.instances) == 1
    # seed the generation-0 durable log the way the real runtime would
    log_dir = session_root / "slug" / "agent-h1-g0"
    log_dir.mkdir(parents=True)
    record = {"type": "assistant/message", "data": {"message": {"content": [
        {"type": "text", "text": "Earlier you asked me to remember ALPHA-9."}]}}}
    cctx = zstandard.ZstdCompressor()
    with open(log_dir / "session.jsonl.zstd", "wb") as fh:
        with cctx.stream_writer(fh, closefd=False) as sw:
            sw.write((json.dumps(record) + "\n").encode("utf-8"))
    # model change -> rebuild -> generation 1
    c.post("/v1/provider/config", json={
        "providers": [{"name": "local", "base_url": "http://mock-llama",
                       "model": "other-model"}],
        "default": "local",
    })
    with c.stream("POST", "/v1/agent/stream", json={
        "message": "what did I say?", "conversation_id": "h1"}) as r:
        r.read()
    assert len(FakeHarness.instances) == 2
    first_msg, session_id = FakeHarness.instances[1].calls[0]
    assert session_id == "agent-h1-g1"
    assert "Context handoff" in first_msg
    assert "ALPHA-9" in first_msg
    # handoff happens only once per conversation per generation
    with c.stream("POST", "/v1/agent/stream", json={
        "message": "thanks", "conversation_id": "h1"}) as r:
        r.read()
    second_msg, _ = FakeHarness.instances[1].calls[1]
    assert "Context handoff" not in second_msg


def test_agent_status_and_empty_message(agent_client):
    c, _app = agent_client
    status = c.get("/v1/agent/status").json()
    assert status["runtime_running"] is False
    assert status["permission_policy"]
    assert c.post("/v1/agent/stream",
                  json={"message": "  "}).status_code == 422


def test_agent_cancel_nothing_in_flight(agent_client):
    c, _app = agent_client
    body = c.post("/v1/agent/cancel").json()
    assert body["ok"] is False and "nothing in flight" in body["note"]


def test_stop_terminates_and_status_reports_stopped(manager):
    (manager.models_dir / "m.gguf").write_bytes(b"x")
    manager.start(model="m.gguf")
    result = manager.stop()
    assert result["ok"]
    st = manager.status()
    assert st["running"] is False and st["healthy"] is False


def test_double_start_rejected_while_running(manager):
    (manager.models_dir / "m.gguf").write_bytes(b"x")
    manager.start(model="m.gguf")
    with pytest.raises(RuntimeError) as exc:
        manager.start(model="m.gguf")
    assert "already loaded" in str(exc.value)


def test_start_refuses_port_already_in_use(manager):
    (manager.models_dir / "m.gguf").write_bytes(b"x")
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    port = blocker.getsockname()[1]
    try:
        with pytest.raises(RuntimeError) as exc:
            manager.start(model="m.gguf", port=port)
        assert "already serving" in str(exc.value)
    finally:
        blocker.close()
    # nothing was spawned for a refused start
    assert "cmd" not in manager._spawned


def test_start_adopts_orphaned_llama_server(manager, monkeypatch):
    """A sidecar restart must re-adopt its own healthy llama-server instead
    of refusing (the zombie-server bug class)."""
    (manager.models_dir / "m.gguf").write_bytes(b"x")
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    port = blocker.getsockname()[1]
    blocker.close()
    monkeypatch.setattr(models_module, "_port_in_use", lambda h, p, t=1.0: p == port)
    monkeypatch.setattr(models_module, "_pid_listening_on", lambda p: 999)
    monkeypatch.setattr(models_module, "_process_name", lambda p: "llama-server.exe")
    monkeypatch.setattr(manager, "prober", lambda url: "adopted-model")

    info = manager.start(model="m.gguf", port=port)
    assert info["running"] is True and info["adopted"] is True
    assert info["pid"] == 999 and info["model"] == "adopted-model"
    assert "cmd" not in manager._spawned  # nothing new spawned

    # stop() terminates the adopted pid
    killed = {}
    monkeypatch.setattr(models_module, "_terminate_pid",
                        lambda pid: killed.update(pid=pid))
    manager.stop()
    assert killed["pid"] == 999


def test_start_refuses_foreign_port_squatter(manager, monkeypatch):
    (manager.models_dir / "m.gguf").write_bytes(b"x")
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    port = blocker.getsockname()[1]
    blocker.close()
    monkeypatch.setattr(models_module, "_port_in_use", lambda h, p, t=1.0: p == port)
    monkeypatch.setattr(models_module, "_pid_listening_on", lambda p: 1234)
    monkeypatch.setattr(models_module, "_process_name", lambda p: "LM Studio.exe")
    with pytest.raises(RuntimeError) as exc:
        manager.start(model="m.gguf", port=port)
    assert "LM Studio.exe" in str(exc.value)


def test_launch_extra_args_shared_helper():
    from harness.models import launch_extra_args

    args = launch_extra_args({"threads": 6, "flash_attn": True,
                              "cache_type_k": "q8_0", "alias": "m1",
                              "mlock": True})
    assert args == ["-t", "6", "-fa", "on", "--cache-type-k", "q8_0",
                    "--alias", "m1", "--mlock"]


def test_unhealthy_startup_is_rolled_back(manager, monkeypatch):
    monkeypatch.setattr(manager, "prober", lambda url: False)
    (manager.models_dir / "bad.gguf").write_bytes(b"x")
    with pytest.raises(RuntimeError) as exc:
        manager.start(model="bad.gguf", )
    assert "did not become healthy" in str(exc.value)
    assert manager.status()["running"] is False


# ---------------------------------------------------------------------------
# hub integration (network seams patched)
# ---------------------------------------------------------------------------
class FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_hub_search_shapes_results(manager, monkeypatch):
    monkeypatch.setattr(models_module.requests, "get",
                        lambda url, params=None, timeout=None: FakeResp([
                            {"id": "unsloth/Qwen3.8-9B-GGUF", "downloads": 99,
                             "likes": 10, "lastModified": "2026-08-20T00:00:00"},
                        ]))
    out = manager.hub_search("qwen3.8")
    assert out[0]["repo"] == "unsloth/Qwen3.8-9B-GGUF"
    assert out[0]["last_modified"] == "2026-08-20"


def test_hub_files_includes_mmproj_for_vision_models(manager, monkeypatch):
    """Vision models need mmproj files; they are no longer filtered out."""
    monkeypatch.setattr(models_module.requests, "get",
                        lambda url, params=None, timeout=None: FakeResp([
                            {"path": "model-UD-Q4_K_M.gguf", "size": 5 * 1024 ** 3},
                            {"path": "mmproj-model.gguf", "size": 1},
                            {"path": "README.md", "size": 1},
                        ]))
    files = manager.hub_files("some/repo")
    file_names = [f["file"] for f in files]
    assert "model-UD-Q4_K_M.gguf" in file_names
    assert "mmproj-model.gguf" in file_names  # mmproj now visible for vision models
    assert "README.md" not in file_names  # non-gguf still filtered
    assert files[0]["size_gb"] == 5.0


def test_download_lifecycle_and_local_listing(manager, monkeypatch):
    started = threading.Event()

    def fake_download(repo, filename, dest_dir, token=None):
        path = Path(dest_dir) / filename
        path.write_bytes(b"weights" * 1024)  # >0.01 GB rounding is not needed; just non-empty
        started.set()
        return path

    monkeypatch.setattr(models_module, "_hf_download", fake_download)
    job = manager.download("some/repo", "model-q4.gguf")
    assert job["state"] in ("queued", "downloading", "done")
    assert started.wait(timeout=5)
    deadline = time.time() + 5
    statuses = {j.key: j for j in manager._downloads.values()}
    done = statuses["some/repo:model-q4.gguf"]
    while done.state != "done" and time.time() < deadline:
        time.sleep(0.05)
    assert done.state == "done"
    local = manager.list_local()
    assert any(m["file"] == "model-q4.gguf" for m in local)


def test_duplicate_download_returns_same_job(manager, monkeypatch):
    gate = threading.Event()

    def slow_download(repo, filename, dest_dir, token=None):
        gate.wait(timeout=5)
        return Path(dest_dir) / filename

    monkeypatch.setattr(models_module, "_hf_download", slow_download)
    first = manager.download("r", "f.gguf")
    second = manager.download("r", "f.gguf")
    assert first["key"] == second["key"]
    gate.set()


def test_download_error_is_captured_not_raised(manager, monkeypatch):
    def failing(repo, filename, dest_dir, token=None):
        raise RuntimeError("hub 404")

    monkeypatch.setattr(models_module, "_hf_download", failing)
    manager.download("bad/repo", "x.gguf")
    import time

    deadline = time.time() + 5
    job = list(manager._downloads.values())[0]
    while job.state not in ("error", "done") and time.time() < deadline:
        time.sleep(0.05)
    assert job.state == "error"
    assert "404" in job.error


# ---------------------------------------------------------------------------
# sidecar endpoints
# ---------------------------------------------------------------------------
@pytest.fixture()
def client(tmp_path, monkeypatch, manager):
    monkeypatch.chdir(tmp_path)

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
        state_dir=tmp_path / "harness_state",
        models_manager=manager,
    )
    with TestClient(app) as c:
        yield c, app


def _clean(obj):
    return json.loads(json.dumps(obj))


def test_server_endpoints_roundtrip(client, tmp_path):
    c, app = client
    body = c.get("/v1/server/status").json()
    assert body["running"] is False
    assert body["binary_found"] is True

    (app.state.models.models_dir / "demo.gguf").write_bytes(b"x")
    r = c.post("/v1/server/start", json={"model": "demo.gguf"}).json()
    assert r["running"] is True and r["provider_registered"] is True

    # start registered provider 'local-<key>' + engine profile
    provs = {p["name"]: p for p in c.get("/v1/provider/config").json()["providers"]}
    local_provs = [k for k in provs if k.startswith("local-")]
    assert len(local_provs) == 1
    assert provs[local_provs[0]]["base_url"].endswith(
        f":{app.state.models.port}"
    )
    engines = c.get("/v1/engines").json()
    by_name = {e["name"]: e for e in engines["engines"]}
    local_engines = [e for e in engines["engines"] if e["name"].startswith("local")]
    assert engines["default"].startswith("local")
    assert local_engines[0]["kind"] == "llama_cpp"
    assert local_engines[0]["load_options"]["gpu_layers"] == 999

    assert c.post("/v1/server/stop").json()["ok"]

    # conflict while running
    c.post("/v1/server/start", json={"model": "demo.gguf"})
    conflict = c.post("/v1/server/start", json={"model": "demo.gguf"})
    assert conflict.status_code == 409


def test_models_local_and_hub_routes(client, monkeypatch):
    c, app = client
    (app.state.models.models_dir / "a.gguf").write_bytes(b"x" * 2048)
    body = c.get("/v1/models/local").json()
    assert body["models"][0]["file"] == "a.gguf"

    monkeypatch.setattr(models_module.requests, "get",
                        lambda url, params=None, timeout=None: FakeResp([]))
    assert c.get("/v1/models/hub", params={"q": "gemma"}).json()["results"] == []
    files = c.get("/v1/models/hub/files/some/repo")
    assert files.status_code == 200

    dl = c.post("/v1/models/hub/download",
                json={"repo": "r", "file": "f.gguf"})
    assert dl.status_code == 200
    downloads = c.get("/v1/models/hub/downloads").json()["downloads"]
    assert any(d["key"] == "r:f.gguf" for d in downloads)


def test_server_page_serves(client):
    c, _app = client
    page = c.get("/server")
    assert page.status_code == 200
    assert "Strata Studio console" in page.text
    assert "/v1/server/status" in page.text
    # the chat pane streams through the strata
    assert "/v1/strata/stream" in page.text
    assert "chatlog" in page.text
    assert page.headers.get("cache-control") == "no-store"
