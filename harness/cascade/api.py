"""``/v1/cascade/*`` router factory — the console's contract.

Thin HTTP handlers over :mod:`harness.cascade` and the pilot/oracle run
artifacts.  This is the seam that lets any console (the DSH panel, the Studio
page, a bare ``curl``) drive the cascade engine without re-implementing it:
the router wraps ``plan_request``, the registry, the oracle fronts and the
runner under ``experiments/cascade/``.

Route table:

============================  ====  ========================================
Path                          Verb  Contract
============================  ====  ========================================
``/v1/cascade/roles``         GET   role taxonomy A1–H1 (data)
``/v1/cascade/paths``         GET   paths P0–P6 + the route table
``/v1/cascade/registry``      GET   candidate catalog + role coverage
``/v1/cascade/plan``          GET   ``plan_request`` for bucket/confidence/out_len
``/v1/cascade/frontier``      GET   per-role Pareto fronts from an oracle file
``/v1/cascade/run``           POST  launch a pilot/oracle run (background)
``/v1/cascade/runs``          GET   run index, newest first
``/v1/cascade/report/{run}``  GET   one run's ``report.json``
============================  ====  ========================================

Malformed input is 4xx, never 500.  ``run`` launches a subprocess, so its
``tasks``/``run_name`` inputs are strictly sanitised and confined to the
cascade experiment directory.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

from fastapi import APIRouter, Body, HTTPException, Query

from harness.cascade.oracle import CandidateMeasurement, frontier_by_role
from harness.cascade.paths import PATHS, ROUTE_TABLE
from harness.cascade.registry import CandidateRegistry, default_registry
from harness.cascade.roles import ROLES, Device
from harness.cascade.scheduler import LatencyModel, RouteDecision, plan_request

#: Router prefix (the app mounts the router without adding another prefix).
PREFIX = "/v1/cascade"

REPO_ROOT = Path(__file__).resolve().parents[2]
CASCADE_DIR = REPO_ROOT / "experiments" / "cascade"
DEFAULT_RUNS_ROOT = CASCADE_DIR / "runs"
DEFAULT_RUNNER = CASCADE_DIR / "run_cascade.py"

#: Run directory names the router will create or read.
_RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

Spawner = Callable[[list[str], Path, Path], int]


def _default_spawn(cmd: list[str], cwd: Path, log_path: Path) -> int:
    """Start ``cmd`` detached with stdout+stderr in ``log_path``; return its pid."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab") as fh:
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=fh, stderr=subprocess.STDOUT)
    return proc.pid


def _check_run_name(name: str) -> str:
    if not _RUN_NAME_RE.match(name):
        raise HTTPException(
            422, "run name must match [A-Za-z0-9][A-Za-z0-9._-]{0,63}"
        )
    return name


def _resolve_run(runs_root: Path, name: str) -> Path:
    _check_run_name(name)
    target = (runs_root / name).resolve()
    root = runs_root.resolve()
    if target != root and root not in target.parents:
        raise HTTPException(422, "run name escapes the runs root")
    return target


def _run_summary(run_dir: Path) -> dict[str, Any]:
    """One run-index row; ``status`` is ``done`` when the report exists."""
    report_path = run_dir / "report.json"
    row: dict[str, Any] = {
        "run": run_dir.name,
        "path": str(run_dir),
        "created": time.strftime(
            "%Y-%m-%dT%H:%M:%S", time.localtime(run_dir.stat().st_mtime)
        ),
        "status": "done" if report_path.is_file() else "running",
    }
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            summary = report.get("summary") or {}
            row.update(
                {
                    "spec": report.get("spec"),
                    "tasks": summary.get("tasks"),
                    "scion_accuracy": summary.get("scion_accuracy"),
                    "api_accuracy": summary.get("api_accuracy"),
                    "cascade_accuracy": summary.get("cascade_accuracy"),
                    "escalation_rate": summary.get("escalation_rate"),
                    "cost_usd": (summary.get("usage") or {}).get("total_cost_usd"),
                }
            )
        except (OSError, ValueError):
            row["status"] = "unreadable"
    return row


def create_router(
    *,
    runs_root: Any = None,
    runner: Any = None,
    tasks_root: Any = None,
    spawn: Optional[Spawner] = None,
    registry: Optional[CandidateRegistry] = None,
) -> APIRouter:
    """Return the ``/v1/cascade`` router.

    ``runs_root``/``tasks_root``/``runner`` are injectable so tests (and the
    app) can point the router at a tmp tree; ``spawn`` replaces the subprocess
    launcher in tests.
    """
    router = APIRouter(prefix=PREFIX, tags=["cascade"])
    runs = Path(runs_root) if runs_root is not None else DEFAULT_RUNS_ROOT
    tasks_dir = Path(tasks_root) if tasks_root is not None else CASCADE_DIR
    runner_path = Path(runner) if runner is not None else DEFAULT_RUNNER
    launcher: Spawner = spawn or _default_spawn
    catalog = registry or default_registry()

    @router.get("/roles")
    def roles() -> dict[str, Any]:
        return {"count": len(ROLES), "roles": [role.to_dict() for role in ROLES]}

    @router.get("/paths")
    def paths() -> dict[str, Any]:
        return {
            "paths": [path.to_dict() for path in PATHS],
            "route_table": dict(ROUTE_TABLE),
        }

    @router.get("/registry")
    def registry_view() -> dict[str, Any]:
        candidates = [c.to_dict() for c in catalog]
        return {
            "count": len(candidates),
            "candidates": candidates,
            "missing_roles": list(catalog.missing_roles([r.id for r in ROLES])),
        }

    @router.get("/plan")
    def plan(
        bucket: str = Query("qa"),
        confidence: float = Query(1.0, ge=0.0, le=1.0),
        out_len: int = Query(0, ge=0, le=1_000_000),
    ) -> dict[str, Any]:
        decision = RouteDecision(bucket=bucket, confidence=confidence)
        result = plan_request(decision, out_len=out_len)
        return {
            "request": {
                "bucket": bucket,
                "confidence": confidence,
                "out_len": out_len,
            },
            "plan": result.to_dict(),
            "latency_model": LatencyModel.from_brief().to_dict(),
        }

    @router.get("/frontier")
    def frontier(file: str = Query("oracle.json")) -> dict[str, Any]:
        # A basename only: the query cannot escape the cascade directory.
        name = Path(file).name
        if not name.endswith(".json"):
            raise HTTPException(422, "oracle file must be a .json in the cascade dir")
        path = tasks_dir / name
        if not path.is_file():
            return {
                "file": name,
                "measurements": [],
                "frontiers": {},
                "note": "no oracle measurements yet",
            }
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise HTTPException(400, f"unreadable oracle file {name!r}: {exc}") from exc
        rows_raw = raw.get("measurements", raw) if isinstance(raw, dict) else raw
        if not isinstance(rows_raw, list):
            raise HTTPException(400, "oracle file must be a list of measurements")
        try:
            rows = [
                CandidateMeasurement(
                    candidate=str(r["candidate"]),
                    role=str(r["role"]),
                    device=Device(str(r["device"])),
                    quality=float(r["quality"]),
                    latency_ms=float(r["latency_ms"]),
                    unit=str(r.get("unit", "call")),
                    ctx=int(r.get("ctx", 0)),
                    precision=str(r.get("precision", "bf16")),
                    bytes_gb=(None if r.get("bytes_gb") is None else float(r["bytes_gb"])),
                    notes=str(r.get("notes", "")),
                )
                for r in rows_raw
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(400, f"malformed measurement: {exc}") from exc
        fronts = frontier_by_role(rows)
        return {
            "file": name,
            "measurements": [r.to_dict() for r in rows],
            "frontiers": {role: [r.to_dict() for r in front] for role, front in fronts.items()},
        }

    @router.post("/run")
    def launch(body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        tasks_name = str(body.get("tasks") or "tasks-hard.json")
        if Path(tasks_name).name != tasks_name or not tasks_name.endswith(".json"):
            raise HTTPException(422, "tasks must be a .json file name in the cascade dir")
        tasks_path = tasks_dir / tasks_name
        if not tasks_path.is_file():
            raise HTTPException(404, f"no such task file: {tasks_name}")
        if not runner_path.is_file():
            raise HTTPException(409, f"runner not found at {runner_path}")
        limit = int(body.get("limit") or 0)
        if limit < 0:
            raise HTTPException(422, "limit must be >= 0")
        run_name = str(
            body.get("run_name") or time.strftime("cascade-%Y%m%d-%H%M%S")
        )
        _check_run_name(run_name)
        run_dir = runs / run_name
        if run_dir.exists():
            raise HTTPException(409, f"run {run_name!r} already exists")
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, str(runner_path),
            "--tasks", str(tasks_path),
            "--out", str(runs),
            "--run-name", run_name,
        ]
        if limit:
            cmd += ["--limit", str(limit)]
        pid = launcher(cmd, REPO_ROOT, run_dir / "run_stdout.log")
        return {
            "ok": True,
            "run": run_name,
            "run_dir": str(run_dir),
            "log": str(run_dir / "run_stdout.log"),
            "pid": pid,
        }

    @router.get("/runs")
    def runs_index() -> dict[str, Any]:
        if not runs.is_dir():
            return {"runs": []}
        rows = [
            _run_summary(child)
            for child in runs.iterdir()
            if child.is_dir() and _RUN_NAME_RE.match(child.name)
        ]
        rows.sort(key=lambda r: r["created"], reverse=True)
        return {"runs": rows}

    @router.get("/report/{run}")
    def report(run: str) -> dict[str, Any]:
        target = _resolve_run(runs, run)
        path = target / "report.json"
        if not path.is_file():
            raise HTTPException(404, f"no report for run {run!r}")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise HTTPException(400, f"unreadable report for {run!r}: {exc}") from exc

    return router
