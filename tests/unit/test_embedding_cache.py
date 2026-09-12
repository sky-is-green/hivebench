"""Unit tests for sieve.embedding_cache."""

import numpy as np

from sieve.embedding_cache import EmbeddingCache


def test_cache_hit_returns_same_object():
    cache = EmbeddingCache(max_size=10)
    computed = []

    def compute(text):
        computed.append(text)
        return np.array([1.0, 2.0])

    a = cache.get_or_compute("hello", compute)
    b = cache.get_or_compute("hello", compute)
    assert a is b            # same object on hit
    assert len(computed) == 1  # compute called once
    assert cache.size == 1
    assert cache.hit_rate == 0.5


def test_cache_miss_computes():
    cache = EmbeddingCache()
    calls = []
    cache.get_or_compute("a", lambda t: calls.append(t) or np.array([1.0]))
    cache.get_or_compute("b", lambda t: calls.append(t) or np.array([2.0]))
    assert calls == ["a", "b"]
    assert cache.size == 2


def test_hash_includes_content():
    cache = EmbeddingCache()
    cache.get_or_compute("same", lambda t: np.array([1.0]))
    cache.get_or_compute("samf", lambda t: np.array([2.0]))  # edit -> new hash
    assert cache.size == 2


def test_lru_eviction_at_capacity():
    cache = EmbeddingCache(max_size=2)
    cache.get_or_compute("x1", lambda t: np.array([1.0]))
    cache.get_or_compute("x2", lambda t: np.array([2.0]))
    cache.get_or_compute("x3", lambda t: np.array([3.0]))  # evicts x1 (oldest)
    assert cache.size == 2
    # x1 was evicted -> recompute; x2/x3 still cached
    computed = []

    def compute(t):
        computed.append(t)
        return np.array([0.0])

    cache.get_or_compute("x1", compute)
    assert computed == ["x1"]


def test_lru_reorders_on_hit():
    cache = EmbeddingCache(max_size=2)
    cache.get_or_compute("a", lambda t: np.array([1.0]))
    cache.get_or_compute("b", lambda t: np.array([2.0]))
    cache.get_or_compute("a", lambda t: np.array([1.0]))  # touch a
    cache.get_or_compute("c", lambda t: np.array([3.0]))  # evicts b, not a
    assert cache.get_or_compute("a", lambda t: np.array([9.0]))[0] == 1.0  # a cached


def test_persist_load_roundtrip(tmp_path):
    cache = EmbeddingCache()
    cache.get_or_compute("a", lambda t: np.array([1.0, 2.0]))
    cache.get_or_compute("b", lambda t: np.array([3.0, 4.0]))
    path = tmp_path / "emb.npz"
    cache.persist(path)
    assert path.exists()

    restored = EmbeddingCache.load(path)
    assert restored.size == 2
    assert np.allclose(restored.get_or_compute("a", lambda t: np.zeros(2)), [1.0, 2.0])
    assert np.allclose(restored.get_or_compute("b", lambda t: np.zeros(2)), [3.0, 4.0])


def test_persist_empty_cache(tmp_path):
    cache = EmbeddingCache()
    path = tmp_path / "empty.npz"
    cache.persist(path)
    restored = EmbeddingCache.load(path)
    assert restored.size == 0


def test_persist_tags_model_and_load_matches(tmp_path):
    cache = EmbeddingCache(model="model-a")
    cache.get_or_compute("x", lambda t: np.array([1.0, 2.0]))
    path = tmp_path / "c.npz"
    cache.persist(path)
    restored = EmbeddingCache.load(path, model="model-a")
    assert restored.size == 1
    assert restored.model == "model-a"


def test_load_rejects_model_mismatch(tmp_path):
    cache = EmbeddingCache(model="model-a")
    cache.get_or_compute("x", lambda t: np.array([1.0, 2.0]))
    path = tmp_path / "c.npz"
    cache.persist(path)
    # different model may use a different dimension -> must not reuse embeddings
    mismatched = EmbeddingCache.load(path, model="model-b")
    assert mismatched.size == 0


def test_namespace_per_model():
    p1 = EmbeddingCache.namespace("BAAI/bge-small-en-v1.5", "cache")
    p2 = EmbeddingCache.namespace("sentence-transformers/all-MiniLM-L6-v2", "cache")
    assert p1 != p2
    assert p1.name.startswith("emb_cache_")
    assert p1.parent.name == "cache"