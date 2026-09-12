"""Unit tests for auditor.labeling (ground-truth labeling workflow)."""

from cortex.baselines.runner import load_conversations
from auditor.labeling import (
    generate_eviction_labels,
    generate_query_chunk_pairs,
    generate_routing_decision_labels,
    topic_of,
)

CONV = load_conversations("tests/fixtures/generated")


def test_topic_of_detects_domain():
    assert topic_of("how should the auth service handle JWT expiry") == "authentication"
    assert topic_of("unrelated small talk about the weather") is None


def test_query_chunk_pairs_count_and_shape():
    pairs = generate_query_chunk_pairs(CONV, n=200)
    assert len(pairs) == 200
    assert all({"query", "chunk", "relevant"} <= set(p) for p in pairs)
    assert any(p["relevant"] for p in pairs)
    assert any(not p["relevant"] for p in pairs)


def test_routing_decisions_count_and_shape():
    decisions = generate_routing_decision_labels(CONV, n=200)
    assert len(decisions) == 200
    assert all("query" in d and "optimal_route_auto" in d for d in decisions)
    routes = {d["optimal_route_auto"] for d in decisions}
    assert routes <= {"ultra_small", "medium", "escalation"}


def test_eviction_labels_count_and_shape():
    labels = generate_eviction_labels(CONV, n=100)
    assert len(labels) == 100
    assert all("chunk" in x and "needed_later_auto" in x for x in labels)


def test_labels_deterministic():
    a = generate_query_chunk_pairs(CONV, n=50, seed=1)
    b = generate_query_chunk_pairs(CONV, n=50, seed=1)
    assert a == b