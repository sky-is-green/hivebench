"""CI test-scope resolver (bisect aid for the hosted windows runner).

Scopes resolve to explicit paths here rather than in the workflow, so the
workflow stays platform-neutral (pwsh on windows, bash on linux) and the
dispatch input is never interpreted by a shell:

    all                    tests/unit + tests/integration
    unit                   tests/unit
    integration            tests/integration
    unit:<start>-<end>     the ``[start, end)`` slice of the sorted unit test
                           files (``unit:0-41``, ``unit:41-``, ``unit:-20``)

``CI_SCOPE`` selects the scope (default ``all``); ``CI_JOBS`` selects the
pytest-xdist worker count (``<=1`` or unset runs serial).  Extra CLI arguments
are passed through to pytest, e.g.::

    CI_SCOPE="unit:0-20" CI_JOBS=4 python scripts/ci_scope.py -q
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def unit_files() -> list[str]:
    """Sorted, repo-relative paths of every unit test file."""
    files = sorted((ROOT / "tests" / "unit").rglob("test_*.py"))
    return [p.relative_to(ROOT).as_posix() for p in files]


def resolve(scope: str) -> list[str]:
    """The pytest paths for ``scope``; raises ``SystemExit`` when unknown."""
    scope = (scope or "all").strip()
    if scope == "all":
        return ["tests/unit", "tests/integration"]
    if scope == "unit":
        return ["tests/unit"]
    if scope == "integration":
        return ["tests/integration"]
    if scope.startswith("unit:"):
        files = unit_files()
        start_s, _, end_s = scope[len("unit:"):].partition("-")
        start = int(start_s) if start_s.strip() else 0
        end = int(end_s) if end_s.strip() else len(files)
        start = max(0, min(start, len(files)))
        end = max(start, min(end, len(files)))
        return files[start:end] or ["tests/unit"]
    raise SystemExit(f"unknown CI_SCOPE {scope!r}")


def pytest_args() -> list[str]:
    args = list(sys.argv[1:])
    jobs = os.environ.get("CI_JOBS", "").strip()
    if jobs.isdigit() and int(jobs) > 1:
        # One worker per test file: module-scoped fixtures (e.g. the 24 GiB
        # library in test_stack_residency) are built once per worker, not once
        # per test.
        args += ["-n", jobs, "--dist=loadfile"]
    return args + resolve(os.environ.get("CI_SCOPE", "all"))


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main(pytest_args()))
