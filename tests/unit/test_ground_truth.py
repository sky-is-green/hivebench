"""Unit tests for auditor.ground_truth (S4.2)."""

import pytest

from auditor.ground_truth import GroundTruthDB


@pytest.fixture
def db():
    d = GroundTruthDB()
    yield d
    d.close()


def test_precision_recall_false_eviction_hand_computed(db):
    # A: TP, B: FP, C: FN(evicted-need), D: TN
    db.record_auditor_label(1, "A", True, True)
    db.record_auditor_label(1, "B", True, False)
    db.record_auditor_label(1, "C", False, True)
    db.record_auditor_label(1, "D", False, False)

    assert db.retrieval_precision() == 50.0   # 1/2
    assert db.retrieval_recall() == 50.0      # 1/2
    assert db.false_eviction_rate() == 50.0   # 1/2 of evicted were needed


def test_empty_db_returns_zero(db):
    assert db.retrieval_precision() == 0.0
    assert db.retrieval_recall() == 0.0
    assert db.false_eviction_rate() == 0.0
    assert db.routing_accuracy() == 0.0


def test_window_limits_queries(db):
    for _ in range(50):
        db.record_auditor_label(1, "x", True, True)   # all correct
    for _ in range(50):
        db.record_auditor_label(1, "x", False, False)  # all evicted-correctly
    assert db.retrieval_precision() == 100.0
    # within a window of 50 of the most recent (the evicted ones) -> 0 precision
    assert db.retrieval_precision(window=50) == 0.0


def test_routing_accuracy(db):
    db.record_hive_decision(1, "route", {}, "correct")
    db.record_hive_decision(2, "route", {}, "wrong")
    db.record_hive_decision(3, "route", {}, "correct")
    assert db.routing_accuracy() == pytest.approx(100.0 * 2 / 3)


def test_parameter_version(db):
    db.record_parameter_version("v1", {"decay_multiplier": 1.8})
    n = db._conn.execute("SELECT COUNT(*) AS n FROM parameter_versions").fetchone()["n"]
    assert n == 1


def test_10000_labels_performant(db):
    import time

    start = time.perf_counter()
    for i in range(10000):
        db.record_auditor_label(i % 100, "c", bool(i % 2), bool(i % 3))
    elapsed = time.perf_counter() - start
    assert db.label_count() == 10000
    assert elapsed < 5.0  # plenty of headroom for 10k inserts
    assert 0.0 <= db.retrieval_precision() <= 100.0