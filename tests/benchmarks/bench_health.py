"""S3 health-monitor latency benchmark.

Measures p50/p95/p99 latency of the congestion check + degradation update path.
Writes results JSON and exits 0.

Usage::

    python -m tests.benchmarks.bench_health [--n 2000]
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

import numpy as np

from cortex.degradation import GracefulDegradation
from cortex.health import PipelineHealthMonitor

RESULTS_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "results"


def _pct(values, p):
    return round(float(np.percentile(values, p)), 3)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000)
    args = parser.parse_args(argv)

    monitor = PipelineHealthMonitor(logger=None)
    degradation = GracefulDegradation()

    latencies = []
    for i in range(args.n):
        if i % 10 == 0:
            monitor.metrics.queue_depth = 6
            for _ in range(10):
                monitor.metrics.record_drone_latency(30)
        start = time.perf_counter()
        report = monitor.check_congestion()
        degradation.update(report)
        latencies.append((time.perf_counter() - start) * 1000.0)

    results = {
        "stage": "S3",
        "iterations": args.n,
        "health_check_latency_ms": {
            "p50": _pct(latencies, 50),
            "p95": _pct(latencies, 95),
            "p99": _pct(latencies, 99),
        },
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "bench_health_s3.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("S3 pipeline health monitor latency benchmark")
    print(f"  p50/p95/p99 : {results['health_check_latency_ms']} ms")
    print(f"Wrote {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())