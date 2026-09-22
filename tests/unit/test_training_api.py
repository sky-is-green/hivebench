"""Training-tab endpoints: catalogue, launch, status, report bridge.

Spawns no process (``training._popen`` is intercepted) and touches no GPU.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from harness import training
from harness.app import create_app


class _FakeProc:
    pid = 5150


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "ternary_runner", lambda: ["python", "runner.py"])
    monkeypatch.setattr(training, "_popen", lambda *a, **k: _FakeProc())
    return TestClient(create_app(runs_root=tmp_path))


def _launch(client, method="ste-rotate", **fields):
    base = {"model_dir": "m", "corpus": "c"}
    base.update(fields)
    return client.post("/v1/training/launch", json={
        "engine": "hive-ternary", "method": method, "name": "20k", "fields": base,
    })


def test_engines_catalogue(client):
    body = client.get("/v1/training/engines").json()
    ids = {e["id"] for e in body["engines"]}
    assert ids == {"unsloth-core", "hive-ternary"}
    # the resolved runner is reported for the UI; shape only (its value is env-dependent)
    assert isinstance(body["runner"], list)
    assert isinstance(body["python"], str)


def test_launch_creates_run_bundle(client, tmp_path):
    r = _launch(client, steps=20000)
    assert r.status_code == 200, r.text
    payload = r.json()
    run_dir = tmp_path / payload["run_dir"]
    assert payload["pid"] == 5150
    assert (run_dir / "training.json").is_file()
    assert (run_dir / "pid").read_text() == "5150"


def test_launch_rejects_unimplemented_engine(client):
    r = client.post("/v1/training/launch", json={
        "engine": "unsloth-core", "method": "lora", "fields": {}})
    assert r.status_code == 400
    assert "not implemented" in r.json()["detail"]


def test_launch_reports_missing_fields(client):
    r = _launch(client, model_dir="")
    assert r.status_code == 400
    assert "missing required" in r.json()["detail"]


def test_status_bridges_report_and_runs_index(client, tmp_path):
    run_dir = _launch(client).json()["run_dir"]
    (tmp_path / run_dir / "eval-regions.json").write_text(json.dumps({
        "regions": [{"name": "r0", "ppl": 12.0, "teacher_ppl": 10.0, "ratio": 1.2}],
        "aggregate": {"n_regions": 1, "mean_ratio": 1.2152,
                      "min_ratio": 1.2, "max_ratio": 1.2, "std_ratio": 0.0},
    }))

    status = client.get(f"/v1/training/status/{run_dir}").json()
    assert status["state"] == "finished"

    report = client.post(f"/v1/training/report/{run_dir}").json()
    assert report["kind"] == "training"
    assert report["retention"] == pytest.approx(82.3, abs=0.1)

    page = client.get(f"/view/{run_dir}")
    assert page.status_code == 200
    assert "Hive-Ternary" in page.text

    entry = next(e for e in client.get("/v1/runs").json()["runs"]
                 if e["name"] == run_dir)
    assert entry["kind"] == "training"


def test_report_404_for_non_training_run(client, tmp_path):
    (tmp_path / "protocol_x").mkdir()
    r = client.post("/v1/training/report/protocol_x")
    assert r.status_code == 404
