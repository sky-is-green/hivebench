"""S0 latency benchmark for the foundation infrastructure.

Measures p50/p95/p99 latency of the event logger enqueue path and of the PES
computation, over N iterations. Real drone/assembly latency benchmarks arrive
with S1/S2. Writes results JSON and prints a summary. Exit code 0 on success.

Usage::

    python tests/benchmarks/bench_latency.py [--n 1000]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

_HIVEBENCH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_HIVE = os.path.join(os.path.dirname(_HIVEBENCH), "strata")
sys.path.insert(0, _HIVEBENCH)
sys.path.insert(0, _HIVE)

import numpy as np

from cortex.efficiency import EfficiencyScorer
from logs.event_logger import EventLogger

RESULTS_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "results"


def _percentiles(values: list[float]) -> dict:
    if not values:
        return {"p50": None, "p95": None, "p99": None}
    return {
        "p50": round(float(np.percentile(values, 50)), 3),
        "p95": round(float(np.percentile(values, 95)), 3),
        "p99": round(float(np.percentile(values, 99)), 3),
    }


def bench_logger_latency(n: int, tmp: Path) -> dict:
    logger = EventLogger(log_dir=tmp)
    lat = []
    import time

    for _ in range(n):
        start = time.perf_counter()
        logger.log("bench", "latency", {"n": n})
        lat.append((time.perf_counter() - start) * 1000.0)
    logger.flush()
    logger.close()
    return {
        "iterations": n,
        "enqueue_latency_ms": _percentiles(lat),
        "mean_enqueue_ms": round(statistics.mean(lat), 4),
    }


def bench_pes_latency(n: int) -> dict:
    scorer = EfficiencyScorer()
    import time

    lat = []
    for _ in range(n):
        start = time.perf_counter()
        scorer.compute(
            retrieval_precision=80, routing_accuracy=90, avg_latency_ms=50,
            actual_tps=30, baseline_tps=30, budget_used=700, budget_total=1000,
        )
        lat.append((time.perf_counter() - start) * 1000.0)
    return {"iterations": n, "pes_compute_latency_ms": _percentiles(lat)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    args = parser.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = RESULTS_DIR / "_bench_tmp"

    results = {
        "stage": "S0",
        "logger": bench_logger_latency(args.n, tmp),
        "pes": bench_pes_latency(args.n),
    }
    out = RESULTS_DIR / "bench_latency_s0.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("S0 latency benchmark")
    print("  logger enqueue p50/p95/p99 (ms):",
          results["logger"]["enqueue_latency_ms"])
    print("  PES compute p50/p95/p99 (ms):",
          results["pes"]["pes_compute_latency_ms"])
    print(f"Wrote {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())