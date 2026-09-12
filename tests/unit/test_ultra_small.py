"""Offline unit tests for sieve.ultra_small using an injected encoder."""

import numpy as np
import pytest

from sieve.scores import ChunkScore
from sieve.ultra_small import UltraSmallDrone
from sieve.vocabulary import Vocabulary


def fixed_encoder(vec_map):
    """Injected encode_fn mapping known texts -> fixed numpy vectors."""
    dim = next(iter(vec_map.values())).shape[0]

    def encode(texts):
        out = []
        for t in texts:
            out.append(vec_map.get(t, np.zeros(dim)))
        return np.array(out)

    return encode


def test_cosine_similarity_within_tolerance():
    vec_map = {
        "q": np.array([1.0, 0.0]),
        "same": np.array([1.0, 0.0]),
        "opposite": np.array([0.0, 1.0]),
        "diag": np.array([np.sqrt(0.5), np.sqrt(0.5)]),
    }
    drone = UltraSmallDrone(encode_fn=fixed_encoder(vec_map), vocab=None)
    scores = drone.score("q", ["same", "opposite", "diag"])

    expected = [1.0, 0.0, np.sqrt(0.5)]
    for score, exp in zip(scores, expected):
        assert score.relevance_score == pytest.approx(exp, abs=0.05)
        assert score.chunk_id == scores.index(score)
        assert score.source == "ultra_small"


def test_confidence_deterministic_encoder_is_one():
    vec_map = {"q": np.array([1.0, 0.0]), "c": np.array([1.0, 0.0])}
    drone = UltraSmallDrone(encode_fn=fixed_encoder(vec_map), vocab=None)
    scores = drone.score("q", ["c"])
    assert scores[0].confidence == pytest.approx(1.0)


class NoisyEncoder:
    def __init__(self, base, noise_scale, seed=0):
        self.base = base
        self.noise = noise_scale
        self.rng = np.random.default_rng(seed)

    def __call__(self, texts):
        out = []
        for t in texts:
            v = self.base[t].copy()
            v = v + self.rng.normal(0.0, self.noise, size=v.shape)
            out.append(v)
        return np.array(out)


def test_confidence_low_with_noisy_encoder():
    vec_map = {"q": np.array([1.0, 0.0]), "c": np.array([1.0, 0.0])}
    drone = UltraSmallDrone(
        encode_fn=NoisyEncoder(vec_map, noise_scale=0.5), vocab=None
    )
    scores = drone.score("q", ["c"])
    # noise scale 0.5 relative to the 0.1 normalizer -> confidence near 0
    assert 0.0 <= scores[0].confidence < 0.5


def test_vocab_boost_applied():
    vec_map = {
        "q": np.array([1.0, 0.0]),
        "implement JWT auth": np.array([1.0, 0.0]),  # contains vocab term "JWT"
        "implement plain login": np.array([1.0, 0.0]),
    }
    drone = UltraSmallDrone(
        encode_fn=fixed_encoder(vec_map),
        vocab=Vocabulary(["JWT"]),
        vocab_boost=0.15,
    )
    scores = drone.score("q", ["implement JWT auth", "implement plain login"])
    # same raw cosine; boosted chunk gets +0.15
    assert scores[0].relevance_score == pytest.approx(1.0 + 0.15)
    assert scores[1].relevance_score == pytest.approx(1.0)
    assert scores[0].relevance_score - scores[1].relevance_score == pytest.approx(0.15)


def test_empty_chunks_returns_empty():
    drone = UltraSmallDrone(encode_fn=lambda texts: np.zeros((len(texts), 8)), vocab=None)
    assert drone.score("q", []) == []


def test_confidence_mode_off_skips_passes():
    vec_map = {"q": np.array([1.0, 0.0]), "c": np.array([1.0, 0.0])}
    drone = UltraSmallDrone(
        encode_fn=NoisyEncoder(vec_map, 0.5), vocab=None, confidence_mode="off"
    )
    scores = drone.score("q", ["c"])
    assert scores[0].confidence == 1.0


def test_invalid_confidence_mode_raises():
    with pytest.raises(ValueError):
        UltraSmallDrone(confidence_mode="bogus")


def test_embedding_cache_hit_rate_on_repeated_conversation():
    vec_map = {"q": np.array([1.0, 0.0])}
    chunks = ["chunk number %d" % i for i in range(20)]
    for c in chunks:
        vec_map[c] = np.array([1.0, 0.0])
    drone = UltraSmallDrone(encode_fn=fixed_encoder(vec_map), vocab=None)

    for _ in range(5):
        drone.score("q", chunks)

    # Repeating the same chunks across turns reuses cached embeddings (>50%).
    assert drone.cache.hit_rate > 0.5