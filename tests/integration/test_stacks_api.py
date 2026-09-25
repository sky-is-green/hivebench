"""``/v1/stacks/*`` router (T38, HIVE-PLAN §B3) — round-trips and 4xx mapping.

T38 runs against a **stub manager** and in-memory ``schema``/``residency``
fakes: T35/T36/T37 land in parallel (ADR-L8/ADR-L9), so this module pins only
the router's own contract — route order, the §B3 payloads, and "invalid input
is 4xx, never 500".  No process, no GPU, no real ``stacks/<name>.json``.
"""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import harness.stack.api as api

# --------------------------------------------------------------------------- #
# Fakes for the frozen schema/residency surfaces (skeletons at branch time)
# --------------------------------------------------------------------------- #


class _FakeStack:
    """Minimal stand-in for ``schema.Stack`` (only what the router calls)."""

    def __init__(self, doc: dict):
        self._doc = doc

    @property
    def name(self) -> str:
        return self._doc.get("name", "")

    @property
    def roles(self) -> list[str]:
        return [t.get("role") for t in self._doc.get("tiers", [])]

    def to_dict(self) -> dict:
        return copy.deepcopy(self._doc)

    @classmethod
    def from_dict(cls, data) -> "_FakeStack":
        if not isinstance(data, dict):
            raise ValueError("stack document must be an object")
        if not isinstance(data.get("tiers", []), list):
            raise ValueError("tiers must be a list")
        return cls(dict(data))


class _FakePlan:
    def __init__(self, doc: dict):
        self._doc = doc

    def to_dict(self) -> dict:
        return copy.deepcopy(self._doc)


class _Store:
    """In-memory ``stacks/<name>.json`` store; records the ``root`` it is given."""

    def __init__(self) -> None:
        self.docs: dict[str, dict] = {}
        self.roots: list = []

    def load(self, name: str, root=None) -> _FakeStack:
        self.roots.append(root)
        if name not in self.docs:
            raise FileNotFoundError(name)
        doc = self.docs[name]
        if doc.get("__load_error__"):
            raise ValueError("malformed on disk")
        return _FakeStack(doc)

    def save(self, stack: _FakeStack, root=None):
        self.roots.append(root)
        if stack._doc.get("__save_error__"):
            raise ValueError("cannot save")
        self.docs[stack.name] = stack.to_dict()
        return Path(str(root)) / f"{stack.name}.json"

    def listing(self, root=None) -> list[dict]:
        self.roots.append(root)
        return [
            {"name": n, "tiers": len(d.get("tiers", [])), "roles": [t.get("role") for t in d.get("tiers", [])]}
            for n, d in sorted(self.docs.items())
        ]

    def delete(self, name: str, root=None) -> bool:
        self.roots.append(root)
        return self.docs.pop(name, None) is not None

    def validate_shape(self, stack: _FakeStack) -> list[str]:
        errors: list[str] = []
        if not stack.name:
            errors.append("stack name is required")
        roles = [t.get("role") for t in stack._doc.get("tiers", [])]
        if len(roles) != len(set(roles)):
            errors.append("duplicate tier role")
        for tier in stack._doc.get("tiers", []):
            if not tier.get("repo") or not tier.get("file"):
                errors.append("tier missing repo/file")
        return errors

    def plan(self, stack: _FakeStack) -> _FakePlan:
        if stack._doc.get("__plan_error__"):
            raise ValueError("tier cannot be resolved")
        return _FakePlan({
            "ok": True,
            "per_card": [{"card": 0, "weights": 1.0, "kv": 0.5, "total": 1.5, "budget": 24.0}],
            "warnings": [],
        })


class _StubManager:
    """Stub ``StackManager``: records calls, returns §B3 shapes."""

    def __init__(self) -> None:
        self.applied: list = []
        self.unloaded: list = []
        self.apply_error: Exception | None = None

    def apply(self, stack):
        if self.apply_error is not None:
            raise self.apply_error
        self.applied.append(stack)
        return {"ok": True, "tiers": [{"role": "face", "port": 8080, "key": "face"}]}

    def status(self):
        return {
            "ok": True,
            "stack": "peer-2tier",
            "tiers": [{"role": "face", "key": "face", "port": 8080, "ctx": 8192,
                       "model": "m", "backend": "vulkan", "resident": True,
                       "vram_gb": 20.0, "tok_s": 31.5, "per_card": []}],
            "warnings": [],
        }

    def unload(self, stack=None):
        self.unloaded.append(stack)
        return {"ok": True, "unloaded": ["face"]}

    def instance_for(self, role):
        return None


def _valid_doc(name: str = "peer-2tier") -> dict:
    return {
        "name": name,
        "version": 1,
        "tiers": [
            {"role": "face", "repo": "unsloth/Qwen3.8-27B-GGUF",
             "file": "Qwen3.8-27B-UD-Q6_K.gguf", "ctx": 262144, "ngl": 99},
            {"role": "worker", "repo": "empero-ai/Qwen3.8-4B-Distill",
             "file": "Qwen3.8-4B-Q6_K.gguf", "ctx": 131072, "ngl": 99},
        ],
        "routing": {"workers_as": "subagent"},
    }


@pytest.fixture
def env(tmp_path, monkeypatch):
    store = _Store()
    fake_schema = SimpleNamespace(
        Stack=_FakeStack,
        load_stack=store.load,
        save_stack=store.save,
        list_stacks=store.listing,
        delete_stack=store.delete,
        validate_shape=store.validate_shape,
    )
    fake_residency = SimpleNamespace(plan_residency=store.plan)
    monkeypatch.setattr(api, "_schema", fake_schema)
    monkeypatch.setattr(api, "_residency", fake_residency)

    manager = _StubManager()
    stacks_root = tmp_path / "stacks"
    app = FastAPI()
    app.include_router(api.create_router(manager, stacks_root=stacks_root))
    return SimpleNamespace(
        client=TestClient(app),
        store=store,
        manager=manager,
        stacks_root=stacks_root,
        repo_root=tmp_path,
    )


# --------------------------------------------------------------------------- #
# list / read / write / delete round-trips
# --------------------------------------------------------------------------- #


def test_list_empty(env):
    r = env.client.get("/v1/stacks")
    assert r.status_code == 200, r.text
    assert r.json() == []


def test_put_get_list_delete_roundtrip(env):
    doc = _valid_doc()
    r = env.client.put("/v1/stacks/peer-2tier", json=doc)
    assert r.status_code == 200, r.text
    assert r.json() == doc

    got = env.client.get("/v1/stacks/peer-2tier")
    assert got.status_code == 200
    assert got.json() == doc

    listing = env.client.get("/v1/stacks").json()
    assert [row["name"] for row in listing] == ["peer-2tier"]
    assert listing[0]["roles"] == ["face", "worker"]

    assert env.client.delete("/v1/stacks/peer-2tier").status_code == 200
    assert env.client.get("/v1/stacks/peer-2tier").status_code == 404
    assert env.client.delete("/v1/stacks/peer-2tier").status_code == 404


def test_put_without_name_uses_path(env):
    doc = _valid_doc()
    del doc["name"]
    r = env.client.put("/v1/stacks/from-path", json=doc)
    assert r.status_code == 200, r.text
    assert r.json()["name"] == "from-path"
    assert env.store.docs["from-path"]["name"] == "from-path"


def test_put_name_mismatch_is_400(env):
    doc = _valid_doc("other-name")
    r = env.client.put("/v1/stacks/peer-2tier", json=doc)
    assert r.status_code == 400
    assert "does not match" in r.json()["detail"]
    assert "peer-2tier" not in env.store.docs


def test_get_missing_is_404_not_500(env):
    r = env.client.get("/v1/stacks/nope")
    assert r.status_code == 404


def test_unreadable_stack_is_400_not_500(env):
    env.store.docs["broken"] = {"name": "broken", "__load_error__": True}
    r = env.client.get("/v1/stacks/broken")
    assert r.status_code == 400


# --------------------------------------------------------------------------- #
# shape validation -> 4xx
# --------------------------------------------------------------------------- #


def test_put_duplicate_role_is_400_and_not_saved(env):
    doc = _valid_doc()
    doc["tiers"][1]["role"] = "face"  # duplicate
    r = env.client.put("/v1/stacks/peer-2tier", json=doc)
    assert r.status_code == 400
    assert "duplicate" in r.json()["detail"]
    assert "peer-2tier" not in env.store.docs


def test_put_missing_repo_is_400(env):
    doc = _valid_doc()
    del doc["tiers"][0]["repo"]
    r = env.client.put("/v1/stacks/peer-2tier", json=doc)
    assert r.status_code == 400


def test_put_malformed_document_is_400_not_500(env):
    r = env.client.put("/v1/stacks/peer-2tier", json={"name": "peer-2tier", "tiers": "nope"})
    assert r.status_code == 400


def test_put_non_object_body_is_4xx(env):
    r = env.client.put("/v1/stacks/peer-2tier", json=[1, 2, 3])
    assert 400 <= r.status_code < 500


# --------------------------------------------------------------------------- #
# validate / apply / status / unload
# --------------------------------------------------------------------------- #


def test_validate_roundtrip(env):
    env.client.put("/v1/stacks/peer-2tier", json=_valid_doc())
    r = env.client.post("/v1/stacks/peer-2tier/validate")
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"ok", "per_card", "warnings"}
    assert body["ok"] is True
    assert body["per_card"][0]["card"] == 0


def test_validate_missing_is_404(env):
    assert env.client.post("/v1/stacks/nope/validate").status_code == 404


def test_validate_unresolvable_is_400_not_500(env):
    doc = _valid_doc()
    doc["__plan_error__"] = True
    env.store.docs["peer-2tier"] = doc
    r = env.client.post("/v1/stacks/peer-2tier/validate")
    assert r.status_code == 400


def test_apply_roundtrip(env):
    env.client.put("/v1/stacks/peer-2tier", json=_valid_doc())
    r = env.client.post("/v1/stacks/peer-2tier/apply")
    assert r.status_code == 200, r.text
    assert r.json() == {"ok": True, "tiers": [{"role": "face", "port": 8080, "key": "face"}]}
    assert [s.name for s in env.manager.applied] == ["peer-2tier"]


def test_apply_missing_is_404(env):
    assert env.client.post("/v1/stacks/nope/apply").status_code == 404


def test_apply_over_budget_is_409_not_500(env):
    env.client.put("/v1/stacks/peer-2tier", json=_valid_doc())
    env.manager.apply_error = RuntimeError("face over budget: 41.0 > 24.0 GiB")
    r = env.client.post("/v1/stacks/peer-2tier/apply")
    assert r.status_code == 409
    assert "over budget" in r.json()["detail"]


def test_status_roundtrip(env):
    r = env.client.get("/v1/stacks/status")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["tiers"][0]["role"] == "face"


def test_status_is_not_captured_as_a_stack_name(env):
    """/status is registered before /{name} (frozen route-order caveat)."""
    env.store.docs["status"] = _valid_doc("status")
    r = env.client.get("/v1/stacks/status")
    assert r.status_code == 200
    assert "tiers" in r.json() and r.json()["tiers"][0]["key"] == "face"
    assert env.client.get("/v1/stacks/status").json().get("name") != "status"


def test_unload_roundtrip(env):
    env.client.put("/v1/stacks/peer-2tier", json=_valid_doc())
    r = env.client.post("/v1/stacks/peer-2tier/unload")
    assert r.status_code == 200
    assert r.json() == {"ok": True, "unloaded": ["face"]}
    assert env.manager.unloaded == ["peer-2tier"]


# --------------------------------------------------------------------------- #
# deployment seam: PREFIX + stacks_root translation
# --------------------------------------------------------------------------- #


def test_router_is_mounted_at_prefix():
    from fastapi.routing import APIRoute

    router = api.create_router(_StubManager(), stacks_root="/tmp/stacks")
    paths = {r.path for r in router.routes if isinstance(r, APIRoute)}
    assert paths == {
        "/v1/stacks",
        "/v1/stacks/status",
        "/v1/stacks/{name}",
        "/v1/stacks/{name}/validate",
        "/v1/stacks/{name}/apply",
        "/v1/stacks/{name}/unload",
    }


def test_stacks_root_is_a_directory_not_a_repo_root(env):
    """``stacks_root`` names the stack dir; schema gets the repo root above it."""
    env.client.put("/v1/stacks/peer-2tier", json=_valid_doc())
    env.client.get("/v1/stacks/peer-2tier")
    env.client.get("/v1/stacks")
    assert env.store.roots, "schema was never called"
    assert set(env.store.roots) == {env.repo_root}
