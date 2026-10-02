"""Run-history loading for the cascade experiments (no ``harness`` imports)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[2]
CASCADE = REPO / "experiments" / "cascade"
TASK_FILES = (CASCADE / "tasks.json", CASCADE / "tasks-hard.json")
RUNS = CASCADE / "runs"


def load_tasks() -> dict[str, dict]:
    """All task definitions by id (easy + hard sets)."""
    tasks: dict[str, dict] = {}
    for path in TASK_FILES:
        if path.is_file():
            for task in json.loads(path.read_text(encoding="utf-8"))["tasks"]:
                tasks[task["id"]] = task
    return tasks


def load_history(path: Optional[str] = None) -> dict[str, dict]:
    """Latest record per task id across run reports (newest wins).

    With ``path``, one report only; otherwise every run under ``runs/``.  The
    merged set is the labelled data the decision models are calibrated and
    evaluated on: task id -> record + ``_source`` (run name).
    """
    if path:
        reports = [Path(path)]
    else:
        reports = [
            run_dir / "report.json"
            for run_dir in sorted(RUNS.iterdir(), key=lambda p: p.stat().st_mtime)
            if (run_dir / "report.json").is_file()
        ]
    merged: dict[str, dict] = {}
    for report_path in reports:
        data = json.loads(report_path.read_text(encoding="utf-8"))
        for record in data.get("records", []):
            merged[record["id"]] = {**record, "_source": report_path.parent.name}
    return merged
