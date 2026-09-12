"""Unit tests for membrane.dedup (S2.4)."""

import numpy as np

from membrane.dedup import ContextDeduplicator
from retention.store import ContextChunk, ContextStore


def _chunk(cid, content, turn):
    return ContextChunk(id=cid, content=content, turn=turn,
                        fingerprint=cid, last_referenced_turn=turn)


def test_keeps_denser_of_duplicate_pair():
    dedup = ContextDeduplicator()
    lo = _chunk("lo", "JWT", turn=1)                    # density 1.0
    hi = _chunk("hi", "JWT token schema index endpoint", turn=2)  # denser
    other = _chunk("other", "gardening tips for the weekend", turn=3)
    embeddings = {
        "lo": np.array([1.0, 0.0, 0.0]),
        "hi": np.array([1.0, 0.0, 0.0]),   # duplicate of lo (cos 1.0)
        "other": np.array([0.0, 1.0, 0.0]),
    }
    survivors, refresh_map = dedup.deduplicate([lo, hi, other], embeddings)

    ids = {c.id for c in survivors}
    assert "other" in ids
    # exactly one of the duplicate pair survives, and it is the denser one
    assert ("lo" in ids) != ("hi" in ids)
    assert "hi" in ids
    # refresh map points the kept chunk at the freshest turn
    assert refresh_map.get("hi") == 2


def test_keeps_all_non_duplicates():
    dedup = ContextDeduplicator()
    chunks = [_chunk(f"c{i}", f"unique content {i}", i) for i in range(4)]
    embeddings = {c.id: np.eye(4)[i] for i, c in enumerate(chunks)}  # orthogonal
    survivors, refresh_map = dedup.deduplicate(chunks, embeddings)
    assert len(survivors) == 4
    assert refresh_map == {}


def test_single_chunk_passthrough():
    dedup = ContextDeduplicator()
    c = _chunk("c", "only one", 1)
    survivors, refresh_map = dedup.deduplicate([c], {c.id: np.array([1.0])})
    assert survivors == [c]
    assert refresh_map == {}


def test_density_formula_prefers_information():
    dedup = ContextDeduplicator()
    lo = _chunk("lo", "JWT", turn=1)
    hi = _chunk("hi", "JWT token schema index endpoint", turn=2)
    assert dedup._info_density(hi.content) > dedup._info_density(lo.content)