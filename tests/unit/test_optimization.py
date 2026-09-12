"""Unit tests for testing.optimization (S4.4)."""

import pytest

from auditor.ground_truth import GroundTruthDB
from testing.optimization import (
    optimize_budget_ranges,
    optimize_decay,
    optimize_routing_threshold,
    replay_decay,
    replay_routing,
    sweep,
)


def test_sweep_finds_minimum():
    best, best_val, results = sweep([1.0, 1.8, 2.5], lambda c: (c - 1.8) ** 2)
    assert best == 1.8
    assert best_val == 0.0
    assert len(results) == 3


def test_sweep_maximizes_when_requested():
    best, best_val, _ = sweep([1, 2, 3], lambda c: c, minimize=False)
    assert best == 3 and best_val == 3


def test_optimize_decay_with_objective():
    best, _, _ = optimize_decay(
        candidates=[1.2, 1.8, 2.5], objective=lambda m: abs(m - 1.8)
    )
    assert best == 1.8


def test_optimize_routing_threshold_with_objective():
    best, _, _ = optimize_routing_threshold(
        candidates=[1, 2, 3, 4], objective=lambda t: -abs(t - 3)
    )
    assert best == 3


def test_optimize_budget_ranges():
    best, _, _ = optimize_budget_ranges(
        budget_candidates=[(1000, 3000), (2000, 4000), (3000, 5000)],
        objective=lambda r: r[1],  # largest upper bound is "best"
    )
    assert best == (3000, 5000)


def test_optimize_decay_requires_objective_or_db():
    with pytest.raises(ValueError):
        optimize_decay(db=None)


# ---------------------------------------------------------------------------
# Replay-based default objectives (candidate-dependent, not constant)
# ---------------------------------------------------------------------------
@pytest.fixture
def decay_db():
    db = GroundTruthDB()
    # old relevant (decays out at high multiplier), old irrelevant, recent relevant
    for _ in range(4):
        db.record_auditor_label(turn=1, chunk_id="old_rel", predicted_relevant=True,
                               actually_relevant=True, score=0.9)
        db.record_auditor_label(turn=1, chunk_id="old_irr", predicted_relevant=False,
                               actually_relevant=False, score=0.2)
        db.record_auditor_label(turn=10, chunk_id="recent", predicted_relevant=True,
                               actually_relevant=True, score=0.9)
    yield db
    db.close()


def test_replay_decay_varies_with_multiplier(decay_db):
    fe_low, keep_low = replay_decay(decay_db, 1.2)
    fe_high, keep_high = replay_decay(decay_db, 2.5)
    # high multiplier evicts the old relevant chunks -> higher false-eviction
    assert fe_high > fe_low
    assert keep_high < keep_low


def test_optimize_decay_replay_is_candidate_dependent(decay_db):
    best, best_val, results = optimize_decay(decay_db)
    values = [v for _, v in results]
    assert len(set(round(v, 6) for v in values)) > 1  # not a constant -> no tie
    assert best_val == min(values)


@pytest.fixture
def routing_db():
    db = GroundTruthDB()
    db.record_routing_decision(1, score=0, optimal_route="ultra_small")
    db.record_routing_decision(2, score=1, optimal_route="ultra_small")
    db.record_routing_decision(3, score=2, optimal_route="medium")
    db.record_routing_decision(4, score=3, optimal_route="escalation")
    yield db
    db.close()


def test_optimize_routing_replay_finds_interior_optimum(routing_db):
    best, _, results = optimize_routing_threshold(routing_db)
    values = [v for _, v in results]
    assert len(set(round(v, 6) for v in values)) > 1
    # threshold 2 matches the auditor's own default and minimizes cost -> best
    assert best == 2
    assert max(values) == max(v for _, v in results)


def test_replay_routing_matches_hand_computation(routing_db):
    acc, medium_rate = replay_routing(routing_db, threshold=2)
    assert acc == 1.0        # all four match at t=2
    assert medium_rate == 0.25