"""Unit tests for membrane.drift (S2.6)."""

import numpy as np

from membrane.drift import TopicDriftDetector
from retention.store import ContextStore


def _chunks(contents):
    store = ContextStore()
    for i, c in enumerate(contents, start=1):
        store.add_chunk(i, c)
    return store.all_chunks()


def test_similar_topics_no_reset():
    embed_fn = lambda text: np.array([1.0, 0.0, 0.0])  # identical embeddings
    detector = TopicDriftDetector(embed_fn=embed_fn)
    all_chunks = _chunks(["topic a chunk %d" % i for i in range(6)])
    result = detector.check(all_chunks[-1:], all_chunks)
    assert result.drift_score == 0.0
    assert result.should_reset is False


def test_dissimilar_topics_triggers_reset():
    embed_fn = lambda text: np.array([1.0, 0.0]) if "gardening" in text else np.array([0.0, 1.0])
    detector = TopicDriftDetector(embed_fn=embed_fn)
    all_chunks = _chunks(
        ["gardening tips %d" % i for i in range(3)]
        + ["quantum physics %d" % i for i in range(3)]
    )  # distinct contents: RC1 collapses verbatim duplicates at ingest
    # recent chunks are the last ones (quantum physics), historical are first half (gardening)
    recent = all_chunks[-2:]
    result = detector.check(recent, all_chunks)
    assert result.drift_score > 0.6
    assert result.should_reset is True


def test_too_few_chunks_no_reset():
    detector = TopicDriftDetector(embed_fn=lambda t: np.array([1.0]))
    all_chunks = _chunks(["a", "b", "c"])  # fewer than 5
    result = detector.check(all_chunks[-1:], all_chunks)
    assert result.should_reset is False


def test_no_recent_chunks_no_reset():
    detector = TopicDriftDetector(embed_fn=lambda t: np.array([1.0]))
    all_chunks = _chunks(["a"] * 6)
    result = detector.check([], all_chunks)
    assert result.should_reset is False