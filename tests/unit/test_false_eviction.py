"""Deterministic pipeline-level false-eviction measurement (whitepaper §8 row).

Ground truth is the chunks' own fact terms — no LLM auditor, no confound. The
store cap (`max_chunks`) forces LRU eviction; a "false" eviction is one where a
later query's ground-truth answer lived in the evicted chunk. The mechanism
comparison (refresh protection) shows the retention layer's lever, mirroring
the P4 sweep's fixed-budget confound isolation.
"""

import numpy as np

from cortex.routing import DroneRouter, EscalationHandler
from focal.assembly import ContextAssembler
from focal.budget import AdaptiveBudget
from membrane.dedup import ContextDeduplicator
from membrane.drift import TopicDriftDetector
from retention.store import ContextStore
from sieve.scores import ChunkScore


class FactDrone:
    """Scores chunks by whether they contain the query's fact term."""

    def score(self, query, chunks):
        return [
            ChunkScore(i, 0.9 if query in c else 0.1, 1.0) for i, c in enumerate(chunks)
        ]

    def embed(self, text):
        return np.array([1.0, 0.0, 0.0])


def _fact_chunks(count):
    return [f"the api endpoint returns FACT_{i} in its json payload" for i in range(count)]


def _distinct_embed(chunks):
    """One-hot per fact: distinct chunks are orthogonal, so dedup never merges."""

    def embed(content):
        return np.array([1.0 if f"FACT_{i}" in content else 0.0 for i in range(len(chunks))])

    return embed


def _assemble(store, query):
    result = ContextAssembler().assemble(
        query=query,
        current_turn=7,
        store=store,
        router=DroneRouter(),
        ultra_small=FactDrone(),
        medium=FactDrone(),
        escalation=EscalationHandler(),
        dedup=ContextDeduplicator(),
        drift_detector=TopicDriftDetector(
            embed_fn=lambda t: np.array([1.0, 0.0, 0.0])
        ),
        budget=AdaptiveBudget(),
        max_context=8192,
    )
    return result.content


def test_baseline_false_eviction_rate_is_deterministic():
    store = ContextStore(max_chunks=3, embed_fn=_distinct_embed(_fact_chunks(6)))
    chunks = _fact_chunks(6)
    for i, content in enumerate(chunks):
        store.add_chunk(i + 1, content)

    # cap 3 with 6 adds => the three oldest chunks (0..2) are LRU-evicted on add
    assert set(store.turn_index) == {4, 5, 6}
    lost = []
    for i in range(3):
        if f"FACT_{i}" not in _assemble(store, f"FACT_{i}"):
            lost.append(i)
    # every evicted chunk was later needed and is gone: deterministic 100% under
    # blind LRU with no refresh protection
    assert lost == [0, 1, 2]

    for i in range(3, 6):
        assert f"FACT_{i}" in _assemble(store, f"FACT_{i}")


def test_refresh_protects_needed_chunks_from_false_eviction():
    store = ContextStore(max_chunks=3, embed_fn=_distinct_embed(_fact_chunks(6)))
    chunks = _fact_chunks(6)
    cids = [store.add_chunk(i + 1, chunks[i]) for i in range(3)]
    # chunks 0..2 were referenced by earlier retrieval: refreshed to turn 10
    store.apply_refresh({cid: 10 for cid in cids})
    for i in range(3, 6):
        store.add_chunk(i + 1, chunks[i])

    # the refreshed facts outrank the never-referenced newer chunks for LRU
    assert set(store.turn_index) == {1, 2, 3}
    for i in range(3):
        assert f"FACT_{i}" in _assemble(store, f"FACT_{i}")


def test_false_eviction_requires_ground_truth_need():
    store = ContextStore(max_chunks=3, embed_fn=_distinct_embed(_fact_chunks(6)))
    for i, content in enumerate(_fact_chunks(6)):
        store.add_chunk(i + 1, content)

    # chunks 0..2 are evicted but never asked about: zero false evictions
    assert set(store.turn_index) == {4, 5, 6}
    for i in range(3):
        assert f"FACT_{i}" not in store.all_contents()
    for i in range(3, 6):
        assert f"FACT_{i}" in _assemble(store, f"FACT_{i}")