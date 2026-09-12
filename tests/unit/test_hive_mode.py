"""AFK mode endpoint (strata mode toggle): canonical workspace-level
HIVE-MODE.json, isolated per-test via MODE_FILE monkeypatch."""

import json

import pytest
from fastapi.testclient import TestClient

import harness.app as app_module
from harness.app import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "MODE_FILE", tmp_path / "HIVE-MODE.json")
    return TestClient(create_app())


def test_afk_defaults_off(client):
    body = client.get("/v1/strata/mode").json()
    assert body["afk"] is False


def test_afk_toggle_roundtrip_writes_canonical_file(client, tmp_path):
    r = client.post("/v1/strata/mode", json={"afk": True, "note": "going away"})
    assert r.json()["afk"] is True

    data = client.get("/v1/strata/mode").json()
    assert data["afk"] is True
    assert data["note"] == "going away"
    assert "GREEN/YELLOW fixes" in data["preapproved"]

    on_disk = json.loads((tmp_path / "HIVE-MODE.json").read_text(encoding="utf-8-sig"))
    assert on_disk["mode"] == "AFK"
    assert on_disk["queue_for_return"][0].startswith("pushes to public masters")

    client.post("/v1/strata/mode", json={"afk": False})
    assert client.get("/v1/strata/mode").json()["afk"] is False
    assert not (tmp_path / "HIVE-MODE.json").exists()


def test_afk_note_truncated_to_200(client):
    client.post("/v1/strata/mode", json={"afk": True, "note": "x" * 500})
    assert len(client.get("/v1/strata/mode").json()["note"]) == 200
