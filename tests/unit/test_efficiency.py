"""Unit tests for cortex.efficiency (S0.2 PES)."""

import pytest

from cortex.efficiency import (
    EfficiencyScorer,
    context_utilization,
    interpret,
    latency_health,
    throughput_health,
)


def test_latency_health_formula():
    assert latency_health(50) == 100.0
    assert latency_health(200) == 0.0
    assert latency_health(100) == pytest.approx(66.5)
    # below 50ms the raw formula exceeds 100; must be capped.
    assert latency_health(20) == 100.0
    assert latency_health(500) == 0.0  # floor at 0


def test_throughput_health():
    assert throughput_health(30, 30) == 100.0
    assert throughput_health(15, 30) == 50.0
    assert throughput_health(0, 30) == 0.0
    assert throughput_health(40, 30) == 100.0  # capped


def test_context_utilization_sweet_spot_and_penalties():
    assert context_utilization(500, 1000) == 50.0      # 0.5 utilization -> 50
    assert context_utilization(700, 1000) == 100.0      # 0.7 sweet spot
    assert context_utilization(900, 1000) == 100.0      # 0.9 sweet spot
    assert context_utilization(970, 1000) == pytest.approx(80.0)  # 0.97 -> 80
    assert context_utilization(990, 1000) == pytest.approx(60.0)  # 0.99 -> 60
    assert context_utilization(1000, 1000) == pytest.approx(50.0)  # 1.0 -> 50


def test_composite_with_all_components():
    scorer = EfficiencyScorer()
    result = scorer.compute(
        retrieval_precision=80,
        routing_accuracy=90,
        avg_latency_ms=50,        # latency_health 100
        actual_tps=30,
        baseline_tps=30,          # throughput_health 100
        budget_used=700,
        budget_total=1000,        # context_utilization 100
    )
    assert result.composite == pytest.approx(92.0)
    assert 0 <= result.composite <= 100
    assert set(result.active_components) == {
        "retrieval_precision", "routing_accuracy", "latency_health",
        "throughput_health", "context_utilization",
    }


def test_missing_components_renormalize():
    scorer = EfficiencyScorer()
    # Only latency + throughput present -> weights 0.2 and 0.15 renormalized.
    result = scorer.compute(
        avg_latency_ms=50, actual_tps=30, baseline_tps=30
    )
    assert result.composite == pytest.approx(100.0)
    assert set(result.active_components) == {"latency_health", "throughput_health"}


def test_only_latency_present():
    scorer = EfficiencyScorer()
    result = scorer.compute(avg_latency_ms=200)
    assert result.composite == pytest.approx(0.0)


def test_no_components_returns_zero():
    result = EfficiencyScorer().compute()
    assert result.composite == 0.0
    assert result.active_components == []


def test_interpretation_bands():
    assert interpret(85)[0] == "GREEN"
    assert interpret(70)[0] == "YELLOW"
    assert interpret(50)[0] == "RED"
    assert interpret(30)[0] == "CRITICAL"
    assert interpret(100)[0] == "GREEN"
    assert interpret(0)[0] == "CRITICAL"
    # every band maps to a non-empty action string
    for score in (100, 70, 50, 10):
        _, action = interpret(score)
        assert isinstance(action, str) and action


def test_weights_sum_to_one():
    assert sum(EfficiencyScorer.WEIGHTS.values()) == pytest.approx(1.0)