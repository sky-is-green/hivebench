"""Memory-lifecycle tests: conversations are bounded, loggers close, and
nothing accumulates when browser sessions come and go (HARNESS-SPEC M4)."""

import json

import pytest
from fastapi.testclient import TestClient

import harness.app as harness_app_module
from backend.openai_compat import OpenAICompatBackend
from cortex.e2e import FakeUltraSmall, MockTransport
from harness.app import create_app


class SpyLogger:
    """Wraps the real EventLogger API the harness uses; records close()."""

    instances = []

    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.closed = False
        SpyLogger.instances.append(self)

    def close(self):
        self.closed = True

    def log(self, *a, **kw):
        pass


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(harness_app_module, "EventLogger", SpyLogger)
    SpyLogger.instances = []
    monkeypatch.setenv("HARNESS_MAX_CONVERSATIONS", "5")

    def backend_factory(model=None):
        return OpenAICompatBackend(transport=MockTransport(latency_ms=0))

    from harness.models import LlamaServerManager

    manager = LlamaServerManager(
        binary=tmp_path / "llama-server.exe",
        models_dir=tmp_path / "models" / "gguf",
        log_dir=tmp_path / "logs",
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


def test_reset_and_drop_close_the_event_logger(env):
    c, app = env
    c.post("/v1/strata/curate", json={"query": "JWT", "conversation_id": "a"})
    assert any(not lg.closed for lg in SpyLogger.instances)
    c.post("/v1/strata/reset", json={"conversation_id": "a"})
    assert all(lg.closed for lg in SpyLogger.instances)


def test_conversations_bounded_and_evicted_restore_from_disk(env):
    c, app = env
    for i in range(7):  # cap is 5 via env
        c.post("/v1/strata/curate", json={
            "query": f"fact {i}: the number is {i}00",
            "conversation_id": f"conv-{i}",
        })
    st = c.get("/v1/strata/state").json()
    assert st["count"] <= 5  # bounded

    # evicted conversations persist: conv-0 (oldest) restores on next touch
    body = c.post("/v1/strata/curate", json={
        "query": "what is fact 0?", "conversation_id": "conv-0",
    }).json()
    assert body["turn"] == 2  # restored, not fresh
    strata = app.state.harness.hives["conv-0"]
    contents = " ".join(ch.content for ch in strata.store.all_chunks())
    assert "the number is 000" in contents  # the original fact came back from disk


def test_inflight_conversations_are_never_evicted(env):
    c, app = env
    st = app.state.harness
    for i in range(5):
        c.post("/v1/strata/curate", json={"query": f"q{i}", "conversation_id": f"k{i}"})
    # simulate conv-0 mid-turn: it must survive the next creation even though
    # it is the least recently touched
    st.begin("k0")
    c.post("/v1/strata/curate", json={"query": "new", "conversation_id": "k-new"})
    assert "k0" in st.hives
    st.end("k0")


def test_download_registry_prunes_finished_jobs(env, monkeypatch):
    c, app = env
    manager = app.state.models

    def fake_download(repo, filename, dest_dir, token=None):
        path = manager.models_dir / filename
        path.write_bytes(b"x" * 1024)
        return path

    monkeypatch.setattr(harness_app_module.__dict__.get(
        "models_module", __import__("harness.models", fromlist=["_hf_download"])),
        "_hf_download", fake_download)
    for i in range(25):
        c.post("/v1/models/hub/download",
               json={"repo": f"r{i}", "file": f"f{i}.gguf"})
        manager.downloads_status()
    statuses = manager.downloads_status()
    finished = [s for s in statuses if s["state"] in ("done", "error")]
    assert len(finished) <= 20  # finished jobs prune; active ones stay


def test_memory_probe_endpoint(env):
    c, _app = env
    body = c.get("/v1/server/memory").json()
    assert body["rss_mb"] > 0
    assert body["max_conversations"] == 5
    assert body["threads"] > 0
