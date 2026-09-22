"""Engine profiles — the "LM Studio-like" engine-management surface.

Covers the EngineProfile validation (kind/capabilities/sampling), registry
load/save round-trips, sampling-default merging, and the sidecar /v1/engines
endpoints (with engine-driven sampling defaults flowing into splinter turns).
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.engines import (
    ENGINE_CAPABILITIES,
    ENGINE_KINDS,
    EngineProfile,
    EngineRegistry,
    load_engines,
    save_engines,
)
from backend.openai_compat import OpenAICompatBackend
from cortex.e2e import FakeUltraSmall, MockTransport
from harness.app import create_app


# ---------------------------------------------------------------------------
# EngineProfile


def test_profile_round_trip_and_defaults():
    p = EngineProfile(
        name="bonsai", kind="lmstudio", base_url="http://localhost:1234",
        load_options={"context": 32768, "gpu_layers": 99},
        capabilities=["prefix_caching", "streaming"],
        sampling={"temperature": 0.7},
    )
    restored = EngineProfile.from_dict(p.to_dict())
    assert restored == p


def test_profile_validates_kind():
    with pytest.raises(ValueError, match="unknown kind"):
        EngineProfile.from_dict({"name": "x", "kind": "quantum"})
    for kind in ENGINE_KINDS:
        assert EngineProfile.from_dict({"name": "x", "kind": kind}).kind == kind


def test_profile_validates_capabilities():
    with pytest.raises(ValueError, match="unknown capabilities"):
        EngineProfile.from_dict({"name": "x", "capabilities": ["teleport"]})
    good = EngineProfile.from_dict(
        {"name": "x", "capabilities": list(ENGINE_CAPABILITIES)}
    )
    assert len(good.capabilities) == len(ENGINE_CAPABILITIES)


def test_profile_requires_name():
    with pytest.raises(ValueError, match="missing 'name'"):
        EngineProfile.from_dict({})


def test_merged_sampling_defaults_and_overrides():
    p = EngineProfile(name="x", sampling={"temperature": 0.7, "top_p": 0.9})
    assert p.merged_sampling({}) == {"temperature": 0.7, "top_p": 0.9}
    assert p.merged_sampling({"temperature": 0.2}) == {
        "temperature": 0.2, "top_p": 0.9,
    }


# ---------------------------------------------------------------------------
# Registry


def test_registry_resolve_default_and_missing():
    reg = EngineRegistry(
        engines=[EngineProfile(name="a"), EngineProfile(name="b")], default="b"
    )
    assert reg.resolve().name == "b"
    assert reg.resolve("a").name == "a"
    with pytest.raises(LookupError):
        reg.resolve("nope")


def test_registry_save_load_round_trip(tmp_path):
    path = tmp_path / "engines.local.json"
    reg = EngineRegistry(
        engines=[EngineProfile(name="a", kind="vllm",
                               sampling={"temperature": 0.3})],
        default="a",
    )
    save_engines(reg, path)
    loaded = load_engines(path)
    assert loaded.default == "a"
    assert loaded.engines[0].kind == "vllm"
    assert loaded.engines[0].sampling == {"temperature": 0.3}
    assert load_engines(tmp_path / "missing.json").engines == []


# ---------------------------------------------------------------------------
# Sidecar endpoints


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def backend_factory(model=None):
        return OpenAICompatBackend(
            base_url="http://mock", model=model or "mock-model",
            transport=MockTransport(latency_ms=0),
        )

    app = create_app(
        ultra_factory=FakeUltraSmall,
        backend_factory=backend_factory,
        runs_root=tmp_path / "runs",
        providers_file=tmp_path / "providers.local.json",
        engines_file=tmp_path / "engines.local.json",
        log_dir=str(tmp_path / "logs"),
    )
    with TestClient(app) as c:
        yield c, app


def test_engines_endpoints_round_trip(client):
    c, _app = client
    r = c.post("/v1/engines", json={
        "engines": [{
            "name": "bonsai", "kind": "lmstudio",
            "load_options": {"context": 32768},
            "capabilities": ["prefix_caching", "streaming"],
            "sampling": {"temperature": 0.7, "top_p": 0.9},
        }],
        "default": "bonsai",
        "persist": True,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["default"] == "bonsai"
    assert body["engines"][0]["sampling"] == {"temperature": 0.7, "top_p": 0.9}
    assert body["persisted_to"]

    r = c.get("/v1/engines")
    assert r.status_code == 200
    assert r.json()["engines"][0]["name"] == "bonsai"


def test_engines_endpoint_rejects_bad_kind(client):
    c, _app = client
    r = c.post("/v1/engines", json={
        "engines": [{"name": "x", "kind": "quantum"}], "default": "x",
    })
    assert r.status_code == 422


def test_engine_sampling_defaults_flow_into_turn(client):
    c, _app = client
    c.post("/v1/engines", json={
        "engines": [{
            "name": "warm", "kind": "lmstudio",
            "sampling": {"temperature": 0.7, "top_p": 0.9},
        }],
        "default": "warm",
    })
    r = c.post("/v1/splinter/turn", json={
        "query": "How does JWT authentication work?",
        "conversation_id": "c1",
    })
    assert r.status_code == 200
    # The splinter built for c1 should carry the engine's sampling defaults.
    app = _app
    splinter = app.state.harness.hives["c1"]
    assert splinter.config.sampling == {"temperature": 0.7, "top_p": 0.9}


def test_engine_sampling_defaults_do_not_clobber_explicit_config(client):
    c, _app = client
    c.post("/v1/engines", json={
        "engines": [{"name": "warm", "kind": "lmstudio",
                     "sampling": {"temperature": 0.7}}],
        "default": "warm",
    })
    r = c.post("/v1/splinter/turn", json={
        "query": "q", "conversation_id": "c2",
        "config": {"sampling": {"temperature": 0.2, "top_k": 40}},
    })
    assert r.status_code == 200
    splinter = _app.state.harness.hives["c2"]
    assert splinter.config.sampling == {"temperature": 0.2, "top_k": 40}