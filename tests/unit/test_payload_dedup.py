"""Unit tests for payload dedup + configurable ultra_small budget.

Recency echo: stored chunks that are verbatim copies of content already in
the incoming request messages must not consume curation budget.
"""

import numpy as np

from cortex.config import SplinterConfig
from cortex.routing import DroneRouter, EscalationHandler
from focal.assembly import ContextAssembler
from focal.budget import AdaptiveBudget
from membrane.dedup import ContextDeduplicator
from membrane.drift import TopicDriftDetector
from retention.store import ContextStore
from retention.hygiene import content_fingerprint
from sieve.scores import ChunkScore


ECHO = "the assistant previously said: deployment uses blue-green slots"
NOVEL = "runbook fact: rollback requires the on-call paging token"


class FakeUltraSmall:
    def score(self, query, chunks):
        return [ChunkScore(i, 0.9, 1.0) for i, _ in enumerate(chunks)]

    def embed(self, text):
        # distinct orthogonal-ish embeddings so membrane dedup keeps both
        if text == ECHO:
            return np.array([1.0, 0.0, 0.0])
        return np.array([0.0, 1.0, 0.0])


def _deps():
    return dict(
        router=DroneRouter(),
        ultra_small=FakeUltraSmall(),
        medium=None,
        escalation=EscalationHandler(),
        dedup=ContextDeduplicator(),
        drift_detector=TopicDriftDetector(embed_fn=lambda t: np.array([1.0, 0.0, 0.0])),
        budget=AdaptiveBudget(),
        max_context=8192,
    )


def _store():
    store = ContextStore(embed_fn=FakeUltraSmall().embed)
    store.add_chunk(1, ECHO)
    store.add_chunk(1, NOVEL)
    return store


def _assemble(store, **kw):
    args = _deps()
    args.update(kw)
    return ContextAssembler().assemble(
        query="short status query", current_turn=2, store=store, **args
    )


def test_content_fingerprint_matches_store():
    store = ContextStore()
    cid = store.add_chunk(1, ECHO)
    assert store.chunks[cid].fingerprint == content_fingerprint(ECHO)
    assert len(content_fingerprint(ECHO)) == 12


def test_echo_chunk_skipped_novel_selected():
    store = _store()
    result = _assemble(store, payload_fingerprints={content_fingerprint(ECHO)})
    assert result.payload_dedup_skipped == 1
    assert NOVEL in result.content
    assert ECHO not in result.content


def test_flag_off_both_eligible():
    store = _store()
    result = _assemble(
        store,
        payload_fingerprints={content_fingerprint(ECHO)},
        dedup_against_payload=False,
    )
    assert result.payload_dedup_skipped == 0
    assert ECHO in result.content
    assert NOVEL in result.content


def test_no_fingerprints_nothing_skipped():
    store = _store()
    result = _assemble(store)
    assert result.payload_dedup_skipped == 0
    assert result.chunks_used == 2


def test_ultra_small_budget_default_and_configured():
    assert AdaptiveBudget().compute("ultra_small", 0) == 1000
    assert AdaptiveBudget(ultra_small_budget_tokens=2000).compute("ultra_small", 0) == 2000
    # other routes untouched by the new knob
    assert AdaptiveBudget(ultra_small_budget_tokens=2000).compute("medium", 0) == 3000
    assert AdaptiveBudget(ultra_small_budget_tokens=2000).compute("escalation", 0) == 4000


def test_config_defaults():
    cfg = SplinterConfig()
    assert cfg.dedup_against_payload is True
    assert cfg.ultra_small_budget_tokens == 1000


def test_splinter_reports_skip_count_via_inspect():
    from cortex.splinter import Splinter

    cfg = SplinterConfig()
    splinter = Splinter(config=cfg, ultra=FakeUltraSmall(), backend=None)
    splinter.store.add_chunk(1, ECHO)
    splinter.store.add_chunk(1, NOVEL)
    result = splinter.process_turn(
        "short status query",
        conversation_id="t",
        record_exchange=False,
        payload_fingerprints={content_fingerprint(ECHO)},
    )
    assert result.assembled.payload_dedup_skipped == 1
    assert splinter.inspect_turn(result)["payload_dedup_skipped"] == 1
