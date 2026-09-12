"""P11 comb benchmark: CombStore retrieve cost at max_records scale.

Measures the per-turn comb overhead a gated Strata pays: put (append) latency,
lexical retrieval over a near-full comb (2000 records), and the flush after
touch. Retrieval is lexical-only since comb_probe (2026-08-24) measured that
lexical ranking beats the drone on return turns AND costs ~1 ms instead of
~820 ms. Writes results JSON and prints a summary. Exit code 0 on success.

Usage::

    python -m tests.benchmarks.bench_comb [--n 2000]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_HIVEBENCH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_HIVE = os.path.join(os.path.dirname(_HIVEBENCH), "strata")
sys.path.insert(0, _HIVEBENCH)
sys.path.insert(0, _HIVE)

import numpy as np

from retention.comb import CombStore

RESULTS_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "results"

TOPICS = [
    "authentication JWT token expiry and refresh flow",
    "database schema migration for the billing service",
    "structured logging with json correlation ids",
    "blue green deployment rollback and health checks",
    "rate limiter sliding window per api key",
    "cursor based pagination for the rest api",
    "idempotency keys with redis caching layer",
    "secrets management in the deploy pipeline",
]


def _records(n: int, base: CombStore):
    for i in range(n):
        topic = TOPICS[i % len(TOPICS)]
        chunk = type("C", (), {
            "id": f"c{i:05d}", "content": f"{topic} detail number {i}",
            "turn": i, "fingerprint": f"fp{i}", "timestamp": "",
            "decay_multiplier": 1.0, "times_saved": 1,
            "last_referenced_turn": i,
            "relevance_history": [(i, 0.7)],
        })()
        base.put(chunk)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=2000)
    args = p.parse_args(argv)

    print("P11 comb benchmark (lexical retrieval)")
    t0 = time.perf_counter()
    comb = CombStore(Path(os.environ.get("TEMP", ".")) / "bench_comb.jsonl",
                     max_records=args.n)
    _records(args.n, comb)
    fill_s = time.perf_counter() - t0
    print(f"  fill ({args.n} puts) : {fill_s:.2f}s  ({fill_s / args.n * 1000:.2f} ms/put)")

    queries = ["how does authentication work", "what is the rate limit",
               "how do we roll back a deployment", "how is logging structured"]

    latencies = []
    for i in range(200):
        q = queries[i % len(queries)]
        t0 = time.perf_counter()
        hits = comb.retrieve(q, k=3)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95) - 1]

    t0 = time.perf_counter()
    comb.touch([h.id for h in hits], turn=999_999)
    flush_ms = (time.perf_counter() - t0) * 1000.0

    summary = {
        "model": "lexical (no drone pass)",
        "records": args.n,
        "fill_ms_per_put": round(fill_s / args.n * 1000.0, 3),
        "retrieve_p50_ms": round(p50, 3),
        "retrieve_p95_ms": round(p95, 3),
        "touch_flush_ms": round(flush_ms, 3),
        "candidates_found": len(comb),
    }
    print(f"  retrieve p50/p95  : {p50:.2f} / {p95:.2f} ms (per gated turn)")
    print(f"  touch+flush       : {flush_ms:.2f} ms")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "bench_comb_p11.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())