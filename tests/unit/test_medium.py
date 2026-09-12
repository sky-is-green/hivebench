"""Unit tests for sieve.medium (offline, via injected scoring/embedding)."""

import numpy as np
import pytest

from sieve.medium import MediumDrone


def test_medium_interface_and_scores():
    drone = MediumDrone(score_pair_fn=lambda q, c: 0.42)
    scores = drone.score("query", ["a", "b", "c"])
    assert len(scores) == 3
    for i, s in enumerate(scores):
        assert s.chunk_id == i
        assert s.relevance_score == 0.42
        assert s.confidence == 0.85
        assert s.source == "medium"


def test_medium_differentiates_code_from_natural_language():
    def code_aware(q, chunk):
        return 1.0 if ("def " in chunk or "import " in chunk) else 0.0

    drone = MediumDrone(score_pair_fn=code_aware)
    scores = drone.score("implement auth", [
        "def login(): pass",              # code -> high
        "let's discuss the weather today",  # NL -> low
    ])
    assert scores[0].relevance_score > scores[1].relevance_score
    assert scores[0].relevance_score == 1.0
    assert scores[1].relevance_score == 0.0


def test_empty_chunks_returns_empty():
    drone = MediumDrone(score_pair_fn=lambda q, c: 0.0)
    assert drone.score("q", []) == []


def test_medium_bi_mode_cosine():
    vec = {"q": np.array([1.0, 0.0]), "a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])}
    drone = MediumDrone(embed_fn=lambda t: vec.get(t, np.zeros(2)), mode="bi")
    scores = drone.score("q", ["a", "b"])
    assert scores[0].relevance_score == pytest.approx(1.0)
    assert scores[1].relevance_score == pytest.approx(0.0)
    assert all(s.source == "medium" for s in scores)
    assert all(s.confidence == 0.85 for s in scores)


def test_medium_bi_mode_normalizes():
    vec = {"q": np.array([3.0, 0.0]), "a": np.array([1.0, 0.0])}
    drone = MediumDrone(embed_fn=lambda t: vec[t], mode="bi")
    scores = drone.score("q", ["a"])
    # cosine is magnitude-invariant -> 1.0
    assert scores[0].relevance_score == pytest.approx(1.0)


def test_medium_invalid_mode_raises():
    with pytest.raises(ValueError):
        MediumDrone(mode="bogus")
    with pytest.raises(ValueError):
        MediumDrone(pooling="bogus")