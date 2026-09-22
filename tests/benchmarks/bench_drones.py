"""S1 drone latency benchmark.

Measures p50/p95/p99 latency of the ultra-small drone over N query/chunk pairs.
Uses the real all-MiniLM-L6-v2 model when available; otherwise falls back to a
lightweight deterministic encoder so the suite still passes on machines without
the model. Writes results JSON and exits 0.

Usage::

    python -m tests.benchmarks.bench_drones [--n 200] [--chunks 20]
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

from sieve.ultra_small import UltraSmallDrone

RESULTS_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "results"


def _pct(values, p):
    return round(float(np.percentile(values, p)), 3)


def _make_real_drone():
    try:
        drone = UltraSmallDrone()
        drone._ensure_loaded()
        return drone
    except Exception:  # noqa: BLE001
        return None


def _make_fake_drone():
    def enc(texts):
        return np.random.default_rng(0).normal(size=(len(texts), 384)).astype(np.float32)

    return UltraSmallDrone(encode_fn=enc, vocab=None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200, help="query/chunk-pair batches")
    parser.add_argument("--chunks", type=int, default=20)
    args = parser.parse_args(argv)

    real = _make_real_drone()
    drone = real if real is not None else _make_fake_drone()
    model_kind = "real:paraphrase-MiniLM-L3-v2" if real else "fake:deterministic"

    queries = [f"query number {i} about authentication and sessions" for i in range(args.n)]
    chunks = [f"chunk {i} discussing the schema index and JWT expiry policy" for i in range(args.chunks)]

    # Batch latency: score the whole chunk set per query.
    batch_latencies = []
    for q in queries:
        start = time.perf_counter()
        drone.score(q, chunks)
        batch_latencies.append((time.perf_counter() - start) * 1000.0)

    # Per-pair latency: score a single chunk per query (matches the 5ms/query budget).
    pair_latencies = []
    for i, q in enumerate(queries):
        start = time.perf_counter()
        drone.score(q, [chunks[i % len(chunks)]])
        pair_latencies.append((time.perf_counter() - start) * 1000.0)

    results = {
        "stage": "S1",
        "model": model_kind,
        "batches": args.n,
        "chunks_per_batch": args.chunks,
        "batch_latency_ms": {
            "p50": _pct(batch_latencies, 50),
            "p95": _pct(batch_latencies, 95),
            "p99": _pct(batch_latencies, 99),
        },
        "per_pair_latency_ms": {
            "p50": _pct(pair_latencies, 50),
            "p95": _pct(pair_latencies, 95),
            "p99": _pct(pair_latencies, 99),
        },
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "bench_drones_s1.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("S1 ultra-small drone latency benchmark")
    print(f"  model        : {model_kind}")
    print(f"  per-pair     : {results['per_pair_latency_ms']} ms")
    print(f"  batch ({args.chunks} chunks): {results['batch_latency_ms']} ms")
    print(f"Wrote {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())