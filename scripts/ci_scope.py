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

Windows routing
---------------
Four heavy files exercise no platform API at all (verified: no ``os.name``,
``sys.platform``, ``subprocess`` or ``winreg`` use) but pay a large NTFS
penalty for their sized-file setup.  A ``--durations`` run put 512s in two
``test_stack_residency`` tests alone, with ``test_protocol``,
``test_generate_data`` and ``test_fit_gpu_layers`` adding several more
minutes; on ext4 the same fixtures are metadata-only truncates.  Those files
therefore run on linux and are skipped on windows: they still run in every
full CI run, just once, and the platform-sensitive suites (paths, ``.exe``
names, subprocess launches, sparse-file handling) keep both legs.  Set
``CI_PLATFORM_FILES=all`` to run them on windows too (for bisecting a
windows-only failure).
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

# Platform-independent files that run on linux only (see module docstring).
PLATFORM_INDEPENDENT = (
    "tests/integration/test_generate_data.py",
    "tests/integration/test_protocol.py",
    "tests/unit/test_fit_gpu_layers.py",
    "tests/unit/test_stack_residency.py",
)


def unit_files() -> list[str]:
    """Sorted, repo-relative paths of every unit test file."""
    files = sorted((ROOT / "tests" / "unit").rglob("test_*.py"))
    return [p.relative_to(ROOT).as_posix() for p in files]


def _nt_skip_enabled(nt: bool | None = None) -> bool:
    """True when platform-independent files should be skipped here."""
    if nt is None:
        nt = os.name == "nt"
    return nt and os.environ.get("CI_PLATFORM_FILES", "").strip().lower() != "all"


def _filter_slice(files: list[str], nt: bool | None = None) -> list[str]:
    """Drop linux-only files from an explicit unit slice.

    A slice that is *entirely* linux-only runs anyway: silently executing
    nothing would make a bisect run look green for the wrong reason.
    """
    if not _nt_skip_enabled(nt):
        return files
    kept = [p for p in files if p not in PLATFORM_INDEPENDENT]
    if not kept:
        print("[ci_scope] whole slice is platform-independent; running it here")
        return files
    skipped = [p for p in files if p in PLATFORM_INDEPENDENT]
    for p in skipped:
        print(f"[ci_scope] windows: skipping platform-independent {p}")
    return kept


def platform_ignores(nt: bool | None = None) -> list[str]:
    """pytest ``--ignore`` arguments for the directory scopes on windows."""
    if not _nt_skip_enabled(nt):
        return []
    print(
        f"[ci_scope] windows: ignoring {len(PLATFORM_INDEPENDENT)} "
        "platform-independent file(s) (they run on linux in this workflow)"
    )
    return [f"--ignore={p}" for p in PLATFORM_INDEPENDENT]


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
        return _filter_slice(files[start:end] or ["tests/unit"])
    raise SystemExit(f"unknown CI_SCOPE {scope!r}")


def pytest_args() -> list[str]:
    args = list(sys.argv[1:])
    jobs = os.environ.get("CI_JOBS", "").strip()
    if jobs.isdigit() and int(jobs) > 1:
        # One worker per test file: module-scoped fixtures (e.g. the sized
        # library in test_stack_residency) are built once per worker, not once
        # per test.
        args += ["-n", jobs, "--dist=loadfile"]
    return args + resolve(os.environ.get("CI_SCOPE", "all")) + platform_ignores()


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main(pytest_args()))
