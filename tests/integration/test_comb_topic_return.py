"""P11 (proposed) — comb resurrection: deterministic topic-return measurement.

The scenario the active store measurably fails: a topic leaves the budget
(evicted), a second topic fills the store past the stale wall (age > 20), and
the first topic *returns* with a new question about an old fact. Without the
comb, the fact is unretrievable (P4's stale factor walls off every old fact).
With the comb, the evicted fact is frozen to disk and resurrected as a
budget-competitive candidate (raw relevance, exempt from stale/drift).

Ground truth is the chunks' own fact terms — no auditor, no confound. Mirrors
the P2 diagnostic's fact-term math and the P4 sweep's fixed-budget isolation.
"""

import numpy as np

from cortex.routing import DroneRouter, EscalationHandler
from focal.assembly import ContextAssembler
from focal.budget import AdaptiveBudget
from membrane.dedup import ContextDeduplicator
from membrane.drift import TopicDriftDetector
from retention.comb import CombStore
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


def _topic_a_facts(n=2):
    return [f"the rate limiter returns LIMIT_{i} requests per minute" for i in range(n)]


def _topic_b_fact(i):
    return f"the deployment uses BLUE_GREEN_{i} traffic shifting"


def _distinct_embed(all_chunks):
    def embed(content):
        return np.array(
            [
                1.0 if f"LIMIT_{i}" in content else 0.0 for i in range(10)
            ]
            + [1.0 if f"BLUE_GREEN_{i}" in content else 0.0 for i in range(10)]
        )

    return embed


def _build_comb_store(tmp_path, max_chunks=4):
    comb = CombStore(tmp_path / "comb.jsonl", max_records=500, embed_fn=_distinct_embed([]))
    store = ContextStore(
        max_chunks=max_chunks,
        embed_fn=_distinct_embed([]),
        comb=comb,
        comb_relevant_only=True,
    )
    return store, comb


def _curate(store, cids):
    """Simulate the remembrance pass: saved chunks carry an escalated decay
    multiplier — the production 'once curated' signal that gates archiving."""
    for cid in cids:
        store.chunks[cid].decay_multiplier = 1.8
        store.chunks[cid].times_saved = 1


def _seed_store(store):
    cids = [store.add_chunk(i + 1, c) for i, c in enumerate(_topic_a_facts())]
    _curate(store, cids)
    for i in range(30):
        store.add_chunk(20 + i, _topic_b_fact(i))
    return cids


def _assemble(store, query, turn, comb_candidates=None):
    result = ContextAssembler().assemble(
        query=query,
        current_turn=turn,
        store=store,
        router=DroneRouter(),
        ultra_small=FactDrone(),
        medium=FactDrone(),
        escalation=EscalationHandler(),
        dedup=ContextDeduplicator(),
        drift_detector=TopicDriftDetector(embed_fn=lambda t: np.array([1.0, 0.0, 0.0])),
        budget=AdaptiveBudget(),
        max_context=8192,
        comb_candidates=comb_candidates,
    )
    return result.content


def test_topic_return_fact_unretrievable_without_comb(tmp_path):
    store, _ = _build_comb_store(tmp_path)
    _seed_store(store)
    # topic A returns with a NEW question about an OLD fact
    content = _assemble(store, f"LIMIT_0", turn=60)
    assert f"LIMIT_0" not in content


def test_comb_resurrects_topic_return_fact(tmp_path):
    store, comb = _build_comb_store(tmp_path)
    _seed_store(store)
    # only the two curated topic-A facts were archived; topic-B chunks dropped
    assert len(comb) == 2
    candidates = comb.retrieve("LIMIT_0", k=3)
    # lexical ranking: the fact must be in the candidate SET (the assembly's
    # greedy budget fill includes every candidate, so set membership is what
    # determines resurrection, not rank-0)
    assert candidates and any(f"LIMIT_0" in c.content for c in candidates)
    content = _assemble(store, f"LIMIT_0", turn=60, comb_candidates=candidates)
    assert f"LIMIT_0" in content


def test_comb_candidates_compete_within_budget(tmp_path):
    store, comb = _build_comb_store(tmp_path)
    _seed_store(store)
    candidates = comb.retrieve("LIMIT_0", k=3)
    content = _assemble(store, f"LIMIT_0", turn=60, comb_candidates=candidates)
    # resurrection must not enlarge the context: the assembled window is bounded
    # by the same budget as a no-comb turn
    assert len(content.split("\n\n")) <= 8


def test_comb_persists_across_reload(tmp_path):
    store, comb = _build_comb_store(tmp_path)
    _seed_store(store)
    comb.close()
    reloaded = CombStore(tmp_path / "comb.jsonl", max_records=500)
    assert len(reloaded) == 2
    hits = reloaded.retrieve("LIMIT_0", k=3)
    assert any("LIMIT_0" in h.content for h in hits)


def test_comb_touch_refreshes_selected_records(tmp_path):
    store, comb = _build_comb_store(tmp_path)
    _seed_store(store)
    candidates = comb.retrieve("LIMIT_0", k=3)
    target = next(c for c in candidates if "LIMIT_0" in c.content)
    touched = comb.touch([target.id], turn=61)
    assert touched == 1
    reloaded = CombStore(tmp_path / "comb.jsonl", max_records=500)
    hit = next(h for h in reloaded.retrieve("LIMIT_0", k=3)
               if "LIMIT_0" in h.content)
    assert hit.last_referenced_turn == 61