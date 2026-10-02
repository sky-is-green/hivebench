"""Unit tests for the ``/v1/cascade`` router (Track C console contract)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from harness.cascade import api as cascade_api
from harness.cascade.registry import default_registry

REPO_ROOT = Path(__file__).resolve().parents[2]
CASCADE = REPO_ROOT / "experiments" / "cascade"


@pytest.fixture()
def client(tmp_path):
    """Router-only app over a tmp runs root, with a recording fake launcher."""
    tasks_dir = tmp_path / "experiments"
    tasks_dir.mkdir()
    (tasks_dir / "tasks-hard.json").write_text(
        json.dumps({"name": "t", "tasks": [{"id": "x", "bucket": "qa", "prompt": "?",
                                            "checker": {"type": "contains", "expect": ["a"]}}]}),
        encoding="utf-8",
    )
    runner = tasks_dir / "run_cascade.py"
    runner.write_text("# fake runner\n", encoding="utf-8")
    spawned: list[list[str]] = []

    def fake_spawn(cmd, cwd, log_path):
        spawned.append(list(cmd))
        Path(log_path).write_text("", encoding="utf-8")
        return 4242

    router = cascade_api.create_router(
        runs_root=tmp_path / "runs",
        tasks_root=tasks_dir,
        runner=runner,
        spawn=fake_spawn,
        registry=default_registry(),
    )
    app = FastAPI()
    app.include_router(router)
    test_client = TestClient(app)
    test_client.spawned = spawned  # type: ignore[attr-defined]
    test_client.runs_root = tmp_path / "runs"  # type: ignore[attr-defined]
    return test_client


def test_roles_lists_the_taxonomy(client):
    data = client.get("/v1/cascade/roles").json()
    assert data["count"] == 22
    ids = {role["id"] for role in data["roles"]}
    assert {"A1", "C1", "D3", "E4", "H1"} <= ids
    c1 = next(role for role in data["roles"] if role["id"] == "C1")
    assert c1["kind"] == "route"
    assert "igpu" in c1["devices"]


def test_paths_and_route_table(client):
    data = client.get("/v1/cascade/paths").json()
    assert [p["id"] for p in data["paths"]] == ["P0", "P1", "P2", "P3", "P4", "P5", "P6"]
    assert data["route_table"]["qa"] == "P2"
    assert data["route_table"]["reasoning"] == "P5"


def test_registry_covers_every_role(client):
    data = client.get("/v1/cascade/registry").json()
    assert data["count"] == 21
    # G1 is the deterministic harness policy (no model candidate by design).
    assert data["missing_roles"] == ["G1"]
    ids = {c["id"] for c in data["candidates"]}
    assert {"qwen3.5-2b", "ornith-1.5-9b", "scion-35b-a3b"} <= ids


def test_plan_prices_the_route(client):
    data = client.get("/v1/cascade/plan", params={"bucket": "qa", "confidence": 0.9, "out_len": 10}).json()
    plan = data["plan"]
    assert plan["path"] == "P2"
    assert plan["blocking_ms"] == pytest.approx(260.0)
    assert plan["escalation_path"] == "P3"
    assert plan["total_expected_ms"] == pytest.approx(301.0)
    assert data["latency_model"]["costs"] == []


def test_plan_rejects_out_of_range_confidence(client):
    assert client.get("/v1/cascade/plan", params={"confidence": 1.5}).status_code == 422
    assert client.get("/v1/cascade/plan", params={"out_len": -1}).status_code == 422


def test_plan_unknown_bucket_falls_back_to_standard_qa(client):
    plan = client.get("/v1/cascade/plan", params={"bucket": "nonsense"}).json()["plan"]
    assert plan["path"] == "P2"


def test_frontier_empty_and_measured(client):
    empty = client.get("/v1/cascade/frontier").json()
    assert empty["measurements"] == []
    assert empty["frontiers"] == {}
    oracle = client.runs_root.parent / "experiments" / "oracle.json"  # type: ignore[attr-defined]
    oracle.write_text(
        json.dumps(
            [
                {"candidate": "a", "role": "C1", "device": "igpu", "quality": 0.9, "latency_ms": 10},
                {"candidate": "b", "role": "C1", "device": "igpu", "quality": 0.8, "latency_ms": 12},
            ]
        ),
        encoding="utf-8",
    )
    data = client.get("/v1/cascade/frontier", params={"file": "oracle.json"}).json()
    assert len(data["measurements"]) == 2
    assert [r["candidate"] for r in data["frontiers"]["C1"]] == ["a"]


def test_frontier_rejects_non_json_and_malformed(client):
    assert client.get("/v1/cascade/frontier", params={"file": "x.txt"}).status_code == 422
    bad = client.runs_root.parent / "experiments" / "bad.json"  # type: ignore[attr-defined]
    bad.write_text(json.dumps([{"candidate": "a"}]), encoding="utf-8")
    assert client.get("/v1/cascade/frontier", params={"file": "bad.json"}).status_code == 400


def test_run_launches_the_runner(client):
    resp = client.post("/v1/cascade/run", json={"run_name": "cascade-test", "limit": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] and body["pid"] == 4242 and body["run"] == "cascade-test"
    cmd = client.spawned[-1]  # type: ignore[attr-defined]
    assert cmd[0] == sys.executable
    assert "--tasks" in cmd and cmd[cmd.index("--tasks") + 1].endswith("tasks-hard.json")
    assert "--run-name" in cmd and "cascade-test" in cmd
    assert "--limit" in cmd
    assert (client.runs_root / "cascade-test").is_dir()  # type: ignore[attr-defined]


def test_run_defaults_name_and_refuses_duplicates(client):
    first = client.post("/v1/cascade/run", json={}).json()
    assert first["run"].startswith("cascade-")
    dup = client.post("/v1/cascade/run", json={"run_name": first["run"]})
    assert dup.status_code == 409


def test_run_input_validation(client):
    assert client.post("/v1/cascade/run", json={"tasks": "../secret.json"}).status_code == 422
    assert client.post("/v1/cascade/run", json={"tasks": "missing.json"}).status_code == 404
    assert client.post("/v1/cascade/run", json={"run_name": "..bad"}).status_code == 422
    assert client.post("/v1/cascade/run", json={"limit": -2}).status_code == 422


def test_runs_index_and_report(client):
    root = client.runs_root  # type: ignore[attr-defined]
    done = root / "cascade-done"
    done.mkdir(parents=True)
    (done / "report.json").write_text(
        json.dumps(
            {
                "spec": "cascade-pilot-hard",
                "summary": {
                    "tasks": 14, "scion_accuracy": 0.64, "api_accuracy": 1.0,
                    "cascade_accuracy": 0.93, "escalation_rate": 0.29,
                    "usage": {"total_cost_usd": 0.0071},
                },
                "records": [],
            }
        ),
        encoding="utf-8",
    )
    (root / "cascade-running").mkdir()
    index = client.get("/v1/cascade/runs").json()["runs"]
    assert {r["run"] for r in index} == {"cascade-done", "cascade-running"}
    row = next(r for r in index if r["run"] == "cascade-done")
    assert row["status"] == "done"
    assert row["cascade_accuracy"] == 0.93
    assert row["cost_usd"] == 0.0071
    running = next(r for r in index if r["run"] == "cascade-running")
    assert running["status"] == "running"
    report = client.get("/v1/cascade/report/cascade-done").json()
    assert report["summary"]["tasks"] == 14


def test_report_validation(client):
    assert client.get("/v1/cascade/report/nope").status_code == 404
    assert client.get("/v1/cascade/report/..bad").status_code == 422
