"""Integration tests for the S1 drone pipeline (query -> router -> drones)."""

import time

import numpy as np
import pytest

from cortex.routing import DroneRouter, EscalationHandler
from sieve.medium import MediumDrone
from sieve.scores import ChunkScore
from sieve.ultra_small import UltraSmallDrone


def _real_ultra_small():
    """Load the real default ultra drone (paraphrase-MiniLM-L3-v2) or skip if
    unavailable."""
    try:
        drone = UltraSmallDrone()
        drone._ensure_loaded()
        return drone
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"default ultra drone unavailable: {exc}")


CHUNKS = [
    "The authentication service uses JWT with a 15 minute expiry.",
    "We decided to use Redis with TTL for the session store.",
    "def login(): return authenticate(request)",
    "Let's talk about the weather forecast for the weekend.",
    "The database schema is normalized to 3NF with a composite index.",
]


def test_mock_pipeline_output_shape():
    """Full pipeline with injected drones: correct shape, no errors."""
    vec = {"q": np.array([1.0, 0.0])}
    for c in CHUNKS:
        vec[c] = np.array([1.0, 0.0])

    def enc(texts):
        return np.array([vec.get(t, np.zeros(2)) for t in texts])

    ultra = UltraSmallDrone(encode_fn=enc, vocab=None)
    medium = MediumDrone(score_pair_fn=lambda q, c: 0.8)
    router = DroneRouter()
    escalation = EscalationHandler()

    query = "How does authentication work?"
    decision = router.route(query)
    assert decision.route_to in ("ultra_small", "medium", "escalation")

    scores = escalation.process(query, CHUNKS, ultra, medium)
    assert len(scores) == len(CHUNKS)
    assert all(isinstance(s, ChunkScore) for s in scores)
    assert all(-1.0 <= s.relevance_score <= 2.0 for s in scores)


def test_mock_pipeline_latency_under_budget():
    """Simple (ultra-small only) path stays well under 100ms."""

    def enc(texts):
        return np.ones((len(texts), 8))

    ultra = UltraSmallDrone(encode_fn=enc, vocab=None)
    start = time.perf_counter()
    for _ in range(10):
        ultra.score("query", CHUNKS)
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / 10.0
    assert elapsed_ms < 100.0


def test_real_ultra_small_scores_in_range():
    drone = _real_ultra_small()
    start = time.perf_counter()
    scores = drone.score("How does authentication work?", CHUNKS)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    assert len(scores) == len(CHUNKS)
    # cosine range [-1, 1] + vocab boost headroom; L3-v2 legitimately returns
    # negative cosines for dissimilar pairs (regression-fixed 2026-08-23 after
    # the L3-v2 default swap: the old [0.0, 1.15] bound assumed all-MiniLM
    # scores were non-negative)
    assert all(-1.0 <= s.relevance_score <= 1.15 for s in scores)  # cosine + boost
    # Gross-sanity latency bound only; precise p50/p95/p99 is measured by the
    # bench_drones benchmark (per-pair CPU budget is ~13ms).
    assert elapsed_ms < 1000.0


def test_real_ultra_small_distinguishes_relevant_from_irrelevant():
    drone = _real_ultra_small()
    scores = drone.score("authentication and sessions", CHUNKS)
    code_index = CHUNKS.index("def login(): return authenticate(request)")
    weather_index = CHUNKS.index("Let's talk about the weather forecast for the weekend.")
    # relevant (auth/session/code) should score above the irrelevant weather chunk
    assert scores[0].relevance_score > scores[weather_index].relevance_score
    assert scores[code_index].relevance_score > scores[weather_index].relevance_score