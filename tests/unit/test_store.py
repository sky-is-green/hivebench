"""Unit tests for retention.store (S2.1)."""

import numpy as np

from retention.store import ContextStore


def test_add_and_retrieve():
    store = ContextStore()
    cid = store.add_chunk(1, "first chunk content")
    assert cid in store.chunks
    chunk = store.chunks[cid]
    assert chunk.turn == 1
    assert chunk.last_referenced_turn == 1
    assert chunk.decay_multiplier == 1.0
    assert chunk.times_saved == 0


def test_turn_index():
    store = ContextStore()
    a = store.add_chunk(1, "hello")
    b = store.add_chunk(1, "world")
    c = store.add_chunk(2, "again")
    assert store.turn_index[1] == [a, b]
    assert store.turn_index[2] == [c]
    assert store.get_turns() == [1, 2]


def test_fingerprint_dedup_by_content():
    store = ContextStore()
    a = store.add_chunk(1, "same content")
    b = store.add_chunk(2, "same content")
    # RC1 fingerprint guard: same fingerprint collapses onto the earliest
    # copy — one chunk, recency refreshed, remembrance counter untouched
    assert b == a
    assert store.count() == 1
    assert store.chunks[a].fingerprint == store.chunks[b].fingerprint
    assert store.chunks[a].times_saved == 0
    assert store.chunks[a].last_referenced_turn == 2
    assert store.chunks[a].turn == 1


def test_apply_refresh():
    store = ContextStore()
    cid = store.add_chunk(1, "content")
    assert store.chunks[cid].last_referenced_turn == 1
    store.apply_refresh({cid: 7})
    assert store.chunks[cid].last_referenced_turn == 7
    # max semantics: never decreases
    store.apply_refresh({cid: 3})
    assert store.chunks[cid].last_referenced_turn == 7


def test_all_chunks_consistent_order():
    store = ContextStore()
    a = store.add_chunk(1, "one")
    b = store.add_chunk(2, "two")
    c = store.add_chunk(3, "three")
    ids = [chunk.id for chunk in store.all_chunks()]
    assert ids == [a, b, c]
    assert store.all_contents() == ["one", "two", "three"]


def test_deletion_candidates_overflow():
    store = ContextStore(max_chunks=2)
    store.add_chunk(1, "oldest")
    store.add_chunk(2, "middle")
    store.add_chunk(3, "newest")
    # the cap is enforced on add: oldest (least-recently-referenced) is evicted
    assert store.count() == 2
    assert store.chunks[store.turn_index[3][0]].content == "newest"
    assert store.turn_index.get(1) is None


def test_overflow_evicts_lru_with_metadata():
    store = ContextStore(max_chunks=2, embed_fn=lambda c: [1.0])
    a = store.add_chunk(1, "oldest")
    store.all_embeddings()  # warm the embedding cache for a
    store.add_chunk(2, "middle")
    store.add_chunk(3, "newest")
    assert store.count() == 2
    assert a not in store.chunks
    assert a not in store._embeddings  # evicted embedding is dropped too
    assert store.turn_index.get(1) is None  # turn index entry cleaned up


def test_get_recent_chunks():
    store = ContextStore()
    store.add_chunk(1, "t1")
    store.add_chunk(2, "t2a")
    store.add_chunk(2, "t2b")
    store.add_chunk(3, "t3")
    recent = store.get_recent_chunks(2)
    assert [c.content for c in recent] == ["t2a", "t2b", "t3"]


def test_all_embeddings_lazy():
    store = ContextStore(embed_fn=lambda content: np.array([float(len(content))]))
    store.add_chunk(1, "abc")
    store.add_chunk(2, "abcdef")
    embs = store.all_embeddings()
    assert len(embs) == 2
    assert all(isinstance(v, np.ndarray) for v in embs.values())


def test_to_dict_roundtrip():
    store = ContextStore(embed_fn=lambda c: c)
    a = store.add_chunk(1, "auth JWT token schema")
    store.chunks[a].decay_multiplier = 1.8
    store.chunks[a].times_saved = 2
    store.chunks[a].last_referenced_turn = 5
    store.chunks[a].relevance_history = [(1, 0.9), (3, 0.7)]
    b = store.add_chunk(2, "gardening watering")
    store.apply_refresh({b: 9})

    restored = ContextStore.from_dict(store.to_dict(), embed_fn=lambda c: c)
    assert restored.count() == store.count()
    assert restored.chunks[a].content == "auth JWT token schema"
    assert restored.chunks[a].decay_multiplier == 1.8
    assert restored.chunks[a].times_saved == 2
    assert restored.chunks[a].last_referenced_turn == 5
    assert restored.chunks[a].relevance_history == [(1, 0.9), (3, 0.7)]
    assert restored.chunks[b].last_referenced_turn == 9
    assert restored.turn_index == {1: [a], 2: [b]}