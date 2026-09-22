"""Splinter test runner, organized by what each group measures.

Groups
------
- **speed**        : performance benchmarks (latency, throughput, memory, drones).
- **intelligence** : accuracy / reasoning / scientific-quality tests (retrieval
                     precision, routing/classifier accuracy, auditor, P1-P10,
                     P5 training, A/B statistics).
- **skills**       : component functionality & integration correctness (logger,
                     drones, splinter context, backends, security, resilience, E2E).
- **maximum**      : everything (used for full coverage / hardware min-maxing).

Each group reports PASS/FAIL plus the measured duration and an estimated time.
Running one group avoids the full multi-minute suite when only a subset is
needed.

Usage::

    python tests/run_hive_tests.py --group speed
    python tests/run_hive_tests.py --group intelligence
    python tests/run_hive_tests.py --group skills
    python tests/run_hive_tests.py --group maximum     # default
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # this repo (hivebench)
_env_home = os.environ.get("SPLINTER_HOME") or os.environ.get("STRATA_HOME")
if _env_home:
    HIVE_HOME = Path(_env_home)
else:
    HIVE_HOME = next((ROOT.parent / _n for _n in ("splinter-memory", "strata-memory")
                      if (ROOT.parent / _n).is_dir()), ROOT.parent / "splinter-memory")
HIVE = HIVE_HOME / "splinter"  # system package root (sibling checkout)
HIVEBENCH = ROOT  # this repo holds the suite
TESTS = ROOT / "tests"
PY = [sys.executable]
ENV = {
    **os.environ,
    "PYTHONPATH": os.pathsep.join(
        [str(HIVE), str(HIVEBENCH), str(ROOT / "vendor"),
         os.environ.get("PYTHONPATH", "")]
    ),
}

# Rough estimated wall-clock time per group (seconds). Recalibrate as you go.
ESTIMATES = {
    "speed": 60,
    "intelligence": 120,
    "skills": 175,
    "maximum": 360,
}

# Tests that measure accuracy / reasoning / scientific quality (P2/P6/P8/P1-P10,
# classifier, auditor, ground truth, statistical A/B).
INTELLIGENCE = [
    "tests/unit/test_classifier.py",
    "tests/unit/test_routing.py",
    "tests/unit/test_ground_truth.py",
    "tests/unit/test_auditor.py",
    "tests/unit/test_optimization.py",
    "tests/unit/test_ab_test.py",
    "tests/unit/test_mcp_battery.py",
    "tests/unit/test_ablation.py",
    "tests/unit/test_labeling.py",
    "tests/unit/test_false_eviction.py",
    "tests/unit/test_retrieval_diagnostic.py",
    "tests/integration/test_comb_topic_return.py",
    "tests/integration/test_protocol.py",
    "tests/integration/test_return_corpus.py",
    "tests/integration/test_p5_smoke.py",
    "tests/integration/test_auditor_pipeline.py",
]

# Performance benchmark scripts (speed group).
SPEED_BENCHMARKS = sorted(
    [p for p in (TESTS / "benchmarks").glob("bench_*.py")]
    + [TESTS / "benchmarks" / "full_benchmark.py"]
)


def _all_pytest_files() -> list[str]:
    files = []
    for d in ("unit", "integration"):
        files += [str(p.relative_to(ROOT)) for p in (TESTS / d).glob("test_*.py")]
    return sorted(files)


def _run_pytest(files: list[str]):
    if not files:
        return "skip", 0.0
    t0 = time.time()
    # Fresh basetemp per invocation: the default pytest-of-<user> base holds a
    # broken 'pytest-current' symlink on this Windows box whose cleanup raises
    # PermissionError at sessionfinish (making every group report FAIL).
    basetemp = Path(tempfile.mkdtemp(prefix="hive_pytest_"))
    result = subprocess.run(
        [*PY, "-m", "pytest", *files, "-q", "--basetemp", str(basetemp)],
        cwd=ROOT,
        env=ENV,
    )
    return ("pass" if result.returncode == 0 else "fail"), time.time() - t0


def _run_benchmarks():
    statuses = []
    t0 = time.time()
    for script in SPEED_BENCHMARKS:
        module = f"tests.benchmarks.{script.stem}"
        result = subprocess.run([*PY, "-m", module], cwd=ROOT, env=ENV)
        statuses.append(result.returncode == 0)
    return ("pass" if all(statuses) else "fail"), time.time() - t0


def run_group(group: str):
    if group == "speed":
        return _run_benchmarks()
    if group == "intelligence":
        return _run_pytest(INTELLIGENCE)
    if group == "skills":
        all_files = _all_pytest_files()
        skills = [f for f in all_files if f not in INTELLIGENCE]
        return _run_pytest(skills)
    raise ValueError(f"unknown group: {group}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Splinter test runner (grouped)")
    parser.add_argument(
        "--group",
        choices=["speed", "intelligence", "skills", "maximum"],
        default="maximum",
        help="which group to run (default: maximum = everything)",
    )
    args = parser.parse_args(argv)

    groups = ["speed", "intelligence", "skills"] if args.group == "maximum" else [args.group]

    print("Splinter test groups (estimated durations):")
    for g in ["speed", "intelligence", "skills", "maximum"]:
        marker = " <- running" if g in groups else ""
        print(f"  {g:<14} ~{ESTIMATES[g]}s{marker}")
    print()

    results = {}
    total = 0.0
    for g in groups:
        t0 = time.time()
        status, dt = run_group(g)
        total += dt
        results[g] = (status, dt)

    print("\n=== Splinter Test Suite Summary ===")
    ok = True
    for g in ["speed", "intelligence", "skills"]:
        if g not in results:
            continue
        status, dt = results[g]
        flag = {"pass": "PASS", "fail": "FAIL", "skip": "SKIP"}[status]
        print(f"  {g:<14} {flag}   {dt:6.1f}s  (est ~{ESTIMATES[g]}s)")
        if status == "fail":
            ok = False
    print(f"  {'total':<14}        {total:6.1f}s")
    print("=" * 36)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
