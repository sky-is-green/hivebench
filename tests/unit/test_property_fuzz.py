"""Property / fuzz tests (E1)."""

import random

import numpy as np

from cortex.routing import DroneRouter, EscalationHandler
from focal.assembly import ContextAssembler
from focal.budget import AdaptiveBudget
from membrane.dedup import ContextDeduplicator
from membrane.drift import TopicDriftDetector
from retention.decay import DecayMatrix
from retention.store import ContextStore
from sieve.scores import ChunkScore


class FakeUS:
    def score(self, query, chunks):
        return [ChunkScore(i, 0.9 if "JWT" in c else 0.2, 1.0) for i, c in enumerate(chunks)]

    def embed(self, text):
        return np.array([1.0, 0.0, 0.0])


class FakeMed:
    def score(self, query, chunks):
        return [ChunkScore(i, 0.5, 0.85, source="medium") for i in range(len(chunks))]


def _drift():
    return TopicDriftDetector(embed_fn=lambda t: np.array([1.0, 0.0, 0.0]))


def test_decay_never_negative():
    matrix = DecayMatrix()
    rng = random.Random(0)
    for _ in range(50):
        store = ContextStore()
        cid = store.add_chunk(1, "x")
        c = store.chunks[cid]
        c.last_referenced_turn = rng.randint(0, 100)
        eff = matrix.apply([c], 150, {cid: rng.uniform(0, 1)})
        assert eff[cid] >= 0.0


def test_decay_monotonic_in_age():
    matrix = DecayMatrix()
    store = ContextStore()
    a = store.add_chunk(1, "a")
    b = store.add_chunk(1, "b")
    ca, cb = store.chunks[a], store.chunks[b]
    ca.last_referenced_turn = 5   # older
    cb.last_referenced_turn = 20  # fresher
    eff = matrix.apply([ca, cb], 30, {ca.id: 1.0, cb.id: 1.0})
    assert eff[ca.id] <= eff[cb.id]


def test_assembler_always_within_budget():
    assembler = ContextAssembler()
    rng = random.Random(1)
    for _ in range(10):
        store = ContextStore(embed_fn=lambda c: np.array([1.0, 0.0, 0.0]))
        for t in range(1, 31):
            store.add_chunk(t, f"chunk content number {rng.randint(0, 1000)} with words")
        r = assembler.assemble(
            query="q", current_turn=30, store=store, router=DroneRouter(),
            ultra_small=FakeUS(), medium=FakeMed(), escalation=EscalationHandler(),
            dedup=ContextDeduplicator(), drift_detector=_drift(),
            budget=AdaptiveBudget(), max_context=8192,
        )
        assert r.token_count <= r.budget
        assert r.budget > 0


def test_router_always_returns_valid_route():
    router = DroneRouter()
    rng = random.Random(0)
    words = ["refactor", "debug", "hello", "analyze", "design", "foo", "bar"]
    for _ in range(200):
        q = " ".join(rng.choice(words) for _ in range(rng.randint(1, 20)))
        assert router.route(q).route_to in ("ultra_small", "medium", "escalation")