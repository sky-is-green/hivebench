"""Unit tests for retention.remembrance (S2.2)."""

import pytest

from retention.remembrance import RemembrancePass
from retention.store import ContextStore
from sieve.scores import ChunkScore


class FakeDrone:
    """score(query, chunks) -> known relevance per chunk content."""

    def __init__(self, relevance_fn):
        self.relevance_fn = relevance_fn

    def score(self, query, chunks):
        return [
            ChunkScore(i, self.relevance_fn(c), 1.0) for i, c in enumerate(chunks)
        ]


def _chunk(content, turn=1):
    store = ContextStore()
    cid = store.add_chunk(turn, content)
    return store.chunks[cid]


def test_save_relevant_discard_irrelevant():
    pass_ = RemembrancePass()
    relevant = _chunk("The JWT auth service uses a schema with foreign keys.")
    irrelevant = _chunk("what is the weather like outside today my friend")
    drone = FakeDrone(lambda c: 0.9 if "JWT" in c else 0.2)

    results = pass_.process([relevant, irrelevant], "auth", drone)
    assert results[0].saved is True
    assert results[1].saved is False
    # only saved chunks get new_decay
    assert results[0].new_decay is not None
    assert results[1].new_decay is None


def test_decay_multiplier_increases_per_save():
    pass_ = RemembrancePass()
    chunk = _chunk("The JWT schema index endpoint must be secure.")
    drone = FakeDrone(lambda c: 0.9)

    sequence = []
    for _ in range(3):
        result = pass_.process([chunk], "auth", drone)[0]
        sequence.append(result.new_decay)

    # 1.0 * 1.8, then * 2.1, then * 2.4
    assert sequence[0] == pytest.approx(1.8)
    assert sequence[1] == pytest.approx(1.8 * 2.1)
    assert sequence[2] == pytest.approx(1.8 * 2.1 * 2.4)
    assert chunk.times_saved == 3
    assert chunk.decay_multiplier == pytest.approx(sequence[2])


def test_compress_strips_filler_keeps_domain():
    pass_ = RemembrancePass()
    raw = "basically I think we should use JWT for the auth service."
    compressed = pass_._compress(raw)
    assert "basically" not in compressed.lower()
    assert "I think" not in compressed.lower()
    assert "JWT" in compressed


def test_compress_returns_original_if_no_domain_terms():
    pass_ = RemembrancePass()
    raw = "basically I think we should just go with the usual plan."
    compressed = pass_._compress(raw)
    # filler stripped, but no domain terms -> original cleaned text returned
    assert "basically" not in compressed.lower()
    assert compressed  # non-empty