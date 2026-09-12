"""Determinism / golden regression tests (E1)."""

import numpy as np

from cortex.routing import DroneRouter, EscalationHandler
from focal.assembly import ContextAssembler
from focal.budget import AdaptiveBudget
from membrane.dedup import ContextDeduplicator
from membrane.drift import TopicDriftDetector
from retention.store import ContextStore
from sieve.scores import ChunkScore


class DetUS:
    def score(self, query, chunks):
        return [ChunkScore(i, 0.9 if "JWT" in c else 0.2, 1.0) for i, c in enumerate(chunks)]

    def embed(self, text):
        return np.array([1.0, 0.0, 0.0])


class DetMed:
    def score(self, query, chunks):
        return [ChunkScore(i, 0.5, 0.85, source="medium") for i in range(len(chunks))]


def _run():
    store = ContextStore(embed_fn=lambda c: np.array([1.0, 0.0, 0.0]))
    store.add_chunk(1, "authentication JWT schema index")
    store.add_chunk(2, "gardening watering plants")
    store.add_chunk(3, "the JWT refresh token policy for the auth service")
    return ContextAssembler().assemble(
        query="auth", current_turn=3, store=store, router=DroneRouter(),
        ultra_small=DetUS(), medium=DetMed(), escalation=EscalationHandler(),
        dedup=ContextDeduplicator(),
        drift_detector=TopicDriftDetector(embed_fn=lambda t: np.array([1.0, 0.0, 0.0])),
        budget=AdaptiveBudget(), max_context=8192,
    )


def test_assembler_output_is_deterministic():
    r1 = _run()
    r2 = _run()
    assert r1.content == r2.content
    assert r1.token_count == r2.token_count
    assert r1.budget == r2.budget
    assert r1.selected_chunk_ids == r2.selected_chunk_ids


def test_assembler_golden_shape():
    r = _run()
    # regression guard on stable properties
    assert r.chunks_used >= 1
    assert r.token_count <= r.budget
    assert "JWT" in r.content
    assert isinstance(r.content, str)