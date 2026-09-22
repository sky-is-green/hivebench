"""S2 assembly pipeline latency benchmark (deterministic, fake drones).

Measures p50/p95/p99 latency of a full assemble() over a 50-chunk store. Uses
injected drones so it is fast and offline; real-drone assembly timing is covered
by integration tests. Writes results JSON and exits 0.

Usage::

    python -m tests.benchmarks.bench_assembly [--n 50]
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

from cortex.routing import DroneRouter, EscalationHandler
from focal.assembly import ContextAssembler
from focal.budget import AdaptiveBudget
from membrane.dedup import ContextDeduplicator
from membrane.drift import TopicDriftDetector
from retention.store import ContextStore
from sieve.scores import ChunkScore

RESULTS_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "results"


class FakeUltraSmall:
    def score(self, query, chunks):
        return [
            ChunkScore(i, 0.9 if "JWT" in c else 0.2, 1.0)
            for i, c in enumerate(chunks)
        ]

    def embed(self, text):
        return np.array([1.0, 0.0, 0.0])


class FakeMedium:
    def score(self, query, chunks):
        return [ChunkScore(i, 0.5, 0.85, source="medium") for i in range(len(chunks))]


def _pct(values, p):
    return round(float(np.percentile(values, p)), 3)


def _build_store():
    store = ContextStore(embed_fn=lambda c: np.array([1.0, 0.0, 0.0]))
    for t in range(1, 51):
        if t % 2:
            store.add_chunk(t, f"authentication JWT schema index {t}")
        else:
            store.add_chunk(t, f"gardening watering plants turn {t}")
    return store


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args(argv)

    store = _build_store()
    assembler = ContextAssembler()
    ultra = FakeUltraSmall()
    medium = FakeMedium()

    latencies = []
    for _ in range(args.n):
        start = time.perf_counter()
        assembler.assemble(
            query="how does authentication work", current_turn=50, store=store,
            router=DroneRouter(), ultra_small=ultra, medium=medium,
            escalation=EscalationHandler(), dedup=ContextDeduplicator(),
            drift_detector=TopicDriftDetector(embed_fn=lambda t: np.array([1.0, 0.0, 0.0])),
            budget=AdaptiveBudget(), max_context=8192,
        )
        latencies.append((time.perf_counter() - start) * 1000.0)

    results = {
        "stage": "S2",
        "iterations": args.n,
        "assembly_latency_ms": {
            "p50": _pct(latencies, 50),
            "p95": _pct(latencies, 95),
            "p99": _pct(latencies, 99),
        },
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "bench_assembly_s2.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("S2 assembly pipeline latency benchmark (fake drones, 50-chunk store)")
    print(f"  p50/p95/p99 : {results['assembly_latency_ms']} ms")
    print(f"Wrote {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())