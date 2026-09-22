"""Splinter-level comb behavior: stale-out archiving, retrieve gate, stats, decay.

Builds on test_comb_topic_return (assembly-level resurrection); these tests
drive Splinter.process_turn itself, so the gate, the stale-out trigger, and the
per-conversation stats are exercised through the real orchestrator path.
"""

import hashlib

import numpy as np

from cortex.config import SplinterConfig
from cortex.splinter import Splinter
from sieve.scores import ChunkScore


class GateDrone:
    """Query-keyed scoring: strong only for the term the query asks about."""

    def __init__(self, hot: str = "HOTTERM"):
        self.hot = hot

    def score(self, query, chunks):
        want = self.hot if self.hot in query else "OTHERTHING"
        return [
            ChunkScore(i, 0.9 if want in c else 0.1, 1.0)
            for i, c in enumerate(chunks)
        ]

    def embed(self, text):
        # hashed bag-of-words (4096 dims): distinct chunks land ~0.8 cosine,
        # below the 0.92 dedup threshold — like real embeddings, unlike a
        # term-based vector that would make all filler chunks identical
        v = np.zeros(4096)
        for w in text.lower().split():
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 4096] = 1.0
        return v


def _hive(tmp_path, hot="HOTTERM", **config_kwargs):
    cfg = SplinterConfig(
        comb_enabled=True,
        comb_dir=str(tmp_path / "comb"),
        comb_relevant_only=True,
        **config_kwargs,
    )
    splinter = Splinter(config=cfg, ultra=GateDrone(hot))
    return splinter


def _curate(splinter, cid):
    splinter.store.chunks[cid].decay_multiplier = 1.8
    splinter.store.chunks[cid].times_saved = 1


def _fill_store(splinter, n=450):
    """Fill the store past the 3000-token ultra-small budget so low-scoring
    chunks are genuinely excluded from selection (the honest budget-pressure
    regime where stale-out archiving matters)."""
    for i in range(n):
        splinter.store.add_chunk(
            2 + i, f"the OTHERTHING config file number {i} settings for the deployment pipeline"
        )


def test_stale_out_archives_unselected_old_chunks(tmp_path):
    splinter = _hive(tmp_path, stale_threshold=20)
    old = splinter.store.add_chunk(1, "the HOTTERM spec says refresh every hour")
    _curate(splinter, old)
    _fill_store(splinter)
    # 25 turns of OTHERTHING queries: the HOTTERM chunk is outranked out of
    # the budget and ages past the stale wall (age 25-1 = 24 > 20)
    for _ in range(25):
        splinter.process_turn("what is the OTHERTHING config", record_exchange=False)
    # the curated old chunk left the active store for the comb
    assert old not in splinter.store.chunks
    assert old in splinter.comb
    assert splinter.comb_stats["archived"] >= 1


def test_gate_consults_comb_only_when_store_is_weak(tmp_path):
    splinter = _hive(tmp_path, stale_threshold=20)
    # strong store match: gate must NOT fire, comb never consulted
    splinter.store.add_chunk(2, "the HOTTERM details are in the api docs")
    splinter.process_turn("what is the HOTTERM", record_exchange=False)
    assert splinter.comb_stats["gate_fired"] == 0
    assert splinter.comb_stats["resurrected"] == 0

    # weak store match in a fresh conversation: gate fires, the archived fact
    # is resurrected (seed the new conversation's comb directly)
    splinter.reset_conversation()
    chunk = type("C", (), {
        "id": "comb_old", "content": "the HOTTERM spec says refresh every hour",
        "turn": 1, "fingerprint": "fp", "timestamp": "",
        "decay_multiplier": 1.8, "times_saved": 1, "last_referenced_turn": 1,
        "relevance_history": [(1, 0.7)],
    })()
    splinter.comb.put(chunk)
    splinter.store.add_chunk(1, "unrelated database migration notes")
    splinter.process_turn("what is the HOTTERM refresh policy", record_exchange=False)
    assert splinter.comb_stats["gate_fired"] == 1
    assert splinter.comb_stats["comb_hits"] >= 1


def test_comb_stats_finalized_per_conversation(tmp_path):
    splinter = _hive(tmp_path, stale_threshold=20)
    splinter.store.add_chunk(1, "unrelated database migration notes")
    splinter.process_turn("what is the HOTTERM refresh policy", record_exchange=False)
    splinter.reset_conversation()
    assert len(splinter.comb_stats_history) == 1
    assert splinter.comb_stats == {"archived": 0, "resurrected": 0, "comb_hits": 0, "gate_fired": 0}


def test_comb_prunes_unreferenced_records_by_age(tmp_path):
    splinter = _hive(tmp_path, comb_max_age_turns=100)
    record = type("C", (), {
        "id": "comb_old", "content": "the HOTTERM spec says refresh every hour",
        "turn": 1, "fingerprint": "fp", "timestamp": "",
        "decay_multiplier": 1.8, "times_saved": 1, "last_referenced_turn": 1,
        "relevance_history": [(1, 0.7)],
    })()
    splinter.comb.put(record)
    assert record.id in splinter.comb
    # 150 turns: the record (referenced at turn 1) ages past the 100-turn
    # archive horizon and is pruned by the per-turn maintenance call
    for i in range(150):
        splinter.process_turn(f"turn {i} unrelated churn", record_exchange=False)
    assert record.id not in splinter.comb


def test_stale_out_skips_never_curated_chunks(tmp_path):
    splinter = _hive(tmp_path, stale_threshold=20)
    # a chunk too large to ever fit the budget is never selected - and
    # selection-as-curation means it never carries relevance history, so it is
    # never archived even when it ages past the stale wall. (The greedy budget
    # fill selects everything on a small store, so a short chunk WOULD be
    # curated on early turns - that is the intended surplus-tier semantics.)
    # Prose filler, not a character run: store-time hygiene (U2) collapses
    # long base64-shaped runs, which would shrink this below the budget.
    raw = splinter.store.add_chunk(1, "the HOTTERM spec says refresh every hour. " + "lorem ipsum dolor sit amet " * 300)
    _fill_store(splinter)
    for _ in range(25):
        splinter.process_turn("what is the OTHERTHING config", record_exchange=False)
    assert raw in splinter.store.chunks
    assert raw not in splinter.comb