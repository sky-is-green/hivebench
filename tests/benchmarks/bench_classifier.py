"""S4 routing-classifier inference benchmark.

Measures inference latency p50/p95/p99 on a trained classifier. Writes results
JSON and exits 0.

Usage::

    python -m tests.benchmarks.bench_classifier [--n 500]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

_HIVEBENCH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_HIVE = os.path.join(os.path.dirname(_HIVEBENCH), "splinter")
sys.path.insert(0, _HIVEBENCH)
sys.path.insert(0, _HIVE)

import numpy as np

from cortex.classifier import RoutingClassifier, RoutingRecord

RESULTS_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "results"


def _pct(values, p):
    return round(float(np.percentile(values, p)), 4)


def _label(f):
    _l, kd, code, _d, _a, _dr = f
    if kd > 0.3 and code > 0:
        return "escalation"
    if kd > 0.1:
        return "medium"
    return "ultra_small"


def _record(rng):
    f = [rng.randint(5, 2000), rng.choice([0.0, 0.05, 0.2, 0.5]),
         rng.choice([0, 0, 0, 1, 2, 3]), rng.randint(0, 60), 0.0, 0.0]
    return RoutingRecord(query="x", optimal_route=_label(f), features=f)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    args = parser.parse_args(argv)

    rng = random.Random(3)
    clf = RoutingClassifier()
    clf.train([_record(rng) for _ in range(1000)])
    feats = [r.features for r in [_record(rng) for _ in range(args.n)]]

    latencies = []
    for f in feats:
        start = time.perf_counter()
        clf.predict_features(f)
        latencies.append((time.perf_counter() - start) * 1000.0)

    results = {
        "stage": "S4",
        "iterations": args.n,
        "inference_latency_ms": {
            "p50": _pct(latencies, 50),
            "p95": _pct(latencies, 95),
            "p99": _pct(latencies, 99),
        },
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "bench_classifier_s4.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("S4 routing-classifier inference benchmark")
    print(f"  p50/p95/p99 : {results['inference_latency_ms']} ms")
    print(f"Wrote {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())