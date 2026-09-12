"""Unit tests for testing.ab_test (S4.3)."""

import random

from testing.ab_test import ABTestRunner

CONV = [
    {"turns": [{"content": f"query {i}"} for i in range(60)]},
    {"turns": [{"content": f"q2-{i}"} for i in range(40)]},
]


def _metrics(pes):
    return {"pes": pes, "retrieval_precision": 80, "routing_accuracy": 90,
            "latency_ms": 30, "context_utilization": 0.8}


def test_identical_configs_tie_within_2_percent():
    runner = ABTestRunner(turns=50)
    result = runner.run(
        CONV,
        process_a=lambda q, t: _metrics(80.0),
        process_b=lambda q, t: _metrics(80.5),  # within 2% noise
    )
    assert result.winner == "tie"
    assert result.config_a_metrics["pes"] == 80.0


def test_bad_config_b_loses():
    runner = ABTestRunner(turns=50)
    result = runner.run(
        CONV,
        process_a=lambda q, t: _metrics(80.0),
        process_b=lambda q, t: _metrics(40.0),  # bad decay=5
    )
    assert result.winner == "A"


def test_bad_config_b_can_win_if_really_better():
    runner = ABTestRunner(turns=50)
    result = runner.run(
        CONV,
        process_a=lambda q, t: _metrics(60.0),
        process_b=lambda q, t: _metrics(95.0),
    )
    assert result.winner == "B"


def test_aggregates_missing_metrics_as_none():
    runner = ABTestRunner(turns=10)
    result = runner.run(
        CONV,
        process_a=lambda q, t: {"pes": 70},
        process_b=lambda q, t: {"pes": 70},
    )
    assert result.config_a_metrics["retrieval_precision"] is None


def test_statistical_tie_for_overlapping_distributions():
    rng = random.Random(1)
    runner = ABTestRunner(turns=50)

    def noisy(mu, sigma):
        return lambda q, t: {"pes": mu + rng.gauss(0, sigma)}

    # large overlap relative to the mean gap -> not statistically significant
    result = runner.run(CONV, process_a=noisy(80.0, 8.0), process_b=noisy(81.0, 8.0))
    assert result.winner == "tie"


def test_statistical_winner_for_separated_distributions():
    rng = random.Random(2)
    runner = ABTestRunner(turns=50)

    def noisy(mu):
        return lambda q, t: {"pes": mu + rng.gauss(0, 1.0)}

    result = runner.run(CONV, process_a=noisy(60.0), process_b=noisy(95.0))
    assert result.winner == "B"