"""Unit tests for cortex.classifier (S4.6)."""

import random
import time

from cortex.classifier import RoutingClassifier, RoutingRecord


def label_for(f):
    """Deterministic rule the tree must learn from the feature vector."""
    _length, kd, code, _depth, _age, _drift = f
    if kd > 0.3 and code > 0:
        return "escalation"
    if kd > 0.1:
        return "medium"
    return "ultra_small"


def _rand_record(rng):
    length = rng.randint(5, 2000)
    kd = rng.choice([0.0, 0.05, 0.2, 0.5])
    code = rng.choice([0, 0, 0, 1, 2, 3, 4])
    depth = rng.randint(0, 60)
    f = [length, kd, code, depth, 0.0, 0.0]
    return RoutingRecord(query="x", optimal_route=label_for(f), features=f)


def test_classifier_learns_routing_rule():
    rng = random.Random(7)
    train = [_rand_record(rng) for _ in range(1000)]
    test = [_rand_record(rng) for _ in range(200)]

    clf = RoutingClassifier()
    clf.train(train)

    correct = sum(1 for r in test if clf.predict_features(r.features) == r.optimal_route)
    agreement = correct / len(test)
    assert agreement >= 0.90


def test_inference_latency_under_20ms():
    rng = random.Random(1)
    train = [_rand_record(rng) for _ in range(500)]
    clf = RoutingClassifier()
    clf.train(train)

    start = time.perf_counter()
    for _ in range(200):
        clf.predict("refactor and analyze the module")
    avg_ms = (time.perf_counter() - start) * 1000.0 / 200.0
    assert avg_ms < 20.0


def test_extract_features_shape():
    clf = RoutingClassifier()
    feats = clf._extract_features("debug the slow query", history=[1, 2, 3])
    assert len(feats) == 6
    assert feats[0] == len("debug the slow query")  # message_length
    assert feats[3] == 3                             # conversation_depth


def test_extract_features_context_stats_populated():
    clf = RoutingClassifier()
    feats = clf._extract_features(
        "debug the slow query", history=[],
        context_stats={"avg_chunk_age": 3.0, "topic_drift_score": 0.7},
    )
    assert feats[4] == 3.0  # avg_chunk_age from live pipeline
    assert feats[5] == 0.7  # topic_drift_score from live pipeline
    # defaults when stats are absent
    feats_default = clf._extract_features("q", history=[])
    assert feats_default[4] == 0.0
    assert feats_default[5] == 0.0