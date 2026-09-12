"""Deep-research queue endpoints: the console collects questions; execution
is QUEEN-only (primary session picks entries up on wake)."""

import pytest
from fastapi.testclient import TestClient

import harness.app as app_module
from harness.app import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(
        app_module, "RESEARCH_QUEUE", tmp_path / "RESEARCH-QUEUE.md"
    )
    return TestClient(create_app())


def test_queue_empty_by_default(client):
    assert client.get("/v1/research/queue").json() == {"items": []}


def test_enqueue_and_read_back(client):
    q = "does overnight model quantization shift paired_ab recall?"
    r = client.post("/v1/research/queue", json={"question": q})
    assert r.json() == {"queued": True, "question": q}
    items = client.get("/v1/research/queue").json()["items"]
    assert items == [q]


def test_enqueue_requires_question(client):
    r = client.post("/v1/research/queue", json={})
    assert r.status_code == 422
    assert client.get("/v1/research/queue").json() == {"items": []}


def test_multiple_entries_preserve_order(client):
    for i in range(3):
        client.post("/v1/research/queue", json={"question": f"q{i}"})
    items = client.get("/v1/research/queue").json()["items"]
    assert items == ["q0", "q1", "q2"]
