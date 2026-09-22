"""S0 throughput benchmark for the foundation infrastructure.

Measures sustained event-logger throughput (events/sec) and PES computations per
second. Writes results JSON and prints a summary. Exit code 0 on success.

Usage::

    python tests/benchmarks/bench_throughput.py [--n 5000]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_HIVEBENCH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_HIVE = os.path.join(os.path.dirname(_HIVEBENCH), "splinter")
sys.path.insert(0, _HIVEBENCH)
sys.path.insert(0, _HIVE)

from cortex.efficiency import EfficiencyScorer
from logs.event_logger import EventLogger

RESULTS_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "results"


def bench_logger_throughput(n: int, tmp: Path) -> float:
    logger = EventLogger(log_dir=tmp)
    start = time.perf_counter()
    for _ in range(n):
        logger.log("bench", "throughput", {"i": _})
    logger.flush()
    logger.close()
    elapsed = time.perf_counter() - start
    return n / elapsed if elapsed > 0 else 0.0


def bench_pes_throughput(n: int) -> float:
    scorer = EfficiencyScorer()
    start = time.perf_counter()
    for _ in range(n):
        scorer.compute(avg_latency_ms=50, actual_tps=30, baseline_tps=30)
    elapsed = time.perf_counter() - start
    return n / elapsed if elapsed > 0 else 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000)
    args = parser.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = RESULTS_DIR / "_bench_tmp"

    results = {
        "stage": "S0",
        "logger_events_per_sec": round(bench_logger_throughput(args.n, tmp), 1),
        "pes_computations_per_sec": round(bench_pes_throughput(args.n), 1),
    }
    out = RESULTS_DIR / "bench_throughput_s0.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("S0 throughput benchmark")
    print(f"  logger events/sec          : {results['logger_events_per_sec']}")
    print(f"  PES computations/sec       : {results['pes_computations_per_sec']}")
    print(f"Wrote {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())