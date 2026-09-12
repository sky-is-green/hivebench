"""Unit test for focal.assembly (S2.7) — end-to-end assembly on a 50-turn store."""

import numpy as np

from cortex.routing import DroneRouter, EscalationHandler
from focal.assembly import ContextAssembler
from focal.budget import AdaptiveBudget
from membrane.dedup import ContextDeduplicator
from membrane.drift import TopicDriftDetector
from retention.store import ContextStore
from sieve.scores import ChunkScore


class FakeUltraSmall:
    def __init__(self, relevance_fn, embed_fn):
        self.relevance_fn = relevance_fn
        self.embed_fn = embed_fn

    def score(self, query, chunks):
        return [ChunkScore(i, self.relevance_fn(query, c), 1.0) for i, c in enumerate(chunks)]

    def embed(self, text):
        return self.embed_fn(text)


class FakeMedium:
    def score(self, query, chunks):
        return [ChunkScore(i, 0.5, 0.85, source="medium") for i, _ in enumerate(chunks)]


def _build_store():
    store = ContextStore(
        embed_fn=lambda c: np.array([float(hash(c) % 1000) / 1000.0, 1.0])
    )
    for t in range(1, 26):
        store.add_chunk(t, f"authentication JWT token expiry {t} minutes for the schema")
    for t in range(26, 51):
        store.add_chunk(t, f"gardening tips for watering plants in turn {t}")
    return store


def test_assembly_produces_bounded_relevant_context():
    store = _build_store()
    relevance = lambda q, c: 0.9 if "JWT" in c else 0.3
    ultra = FakeUltraSmall(relevance, lambda t: np.array([1.0, 0.0, 0.0]))
    medium = FakeMedium()

    assembler = ContextAssembler()
    result = assembler.assemble(
        query="how does authentication work",
        current_turn=50,
        store=store,
        router=DroneRouter(),
        ultra_small=ultra,
        medium=medium,
        escalation=EscalationHandler(),
        dedup=ContextDeduplicator(),
        drift_detector=TopicDriftDetector(embed_fn=lambda t: np.array([1.0, 0.0, 0.0])),
        budget=AdaptiveBudget(),
        max_context=8192,
    )

    assert result.chunks_used > 0
    assert result.token_count <= result.budget          # within budget
    assert "JWT" in result.content                       # relevant chunks present
    assert result.drift_detected is False
    assert result.routing_decision.route_to in ("ultra_small", "medium", "escalation")
    assert len(result.selected_chunk_ids) == result.chunks_used


def test_assembly_does_not_duplicate_concepts():
    store = _build_store()
    relevance = lambda q, c: 0.9 if "JWT" in c else 0.3
    ultra = FakeUltraSmall(relevance, lambda t: np.array([1.0, 0.0, 0.0]))

    assembler = ContextAssembler()
    result = assembler.assemble(
        query="how does authentication work", current_turn=50, store=store,
        router=DroneRouter(), ultra_small=ultra, medium=FakeMedium(),
        escalation=EscalationHandler(), dedup=ContextDeduplicator(),
        drift_detector=TopicDriftDetector(embed_fn=lambda t: np.array([1.0, 0.0, 0.0])),
        budget=AdaptiveBudget(), max_context=8192,
    )
    # unique selected ids
    assert len(set(result.selected_chunk_ids)) == len(result.selected_chunk_ids)