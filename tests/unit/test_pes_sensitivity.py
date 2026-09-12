"""PES sensitivity tests (E2)."""

from cortex.efficiency import EfficiencyScorer, interpret


def test_composite_monotonic_in_each_component():
    s = EfficiencyScorer()
    base = dict(retrieval_precision=50, routing_accuracy=50, avg_latency_ms=50,
                actual_tps=30, baseline_tps=30, budget_used=700, budget_total=1000)
    for key, hi in (
        ("retrieval_precision", 90), ("routing_accuracy", 90),
    ):
        low = s.compute(**base).composite
        high = s.compute(**{**base, key: hi}).composite
        assert high > low
    # latency lower = better
    assert s.compute(**{**base, "avg_latency_ms": 30}).composite >= s.compute(**base).composite


def test_composite_always_in_range():
    s = EfficiencyScorer()
    for p in range(0, 101, 10):
        r = s.compute(retrieval_precision=p, routing_accuracy=p, avg_latency_ms=100,
                      actual_tps=30, baseline_tps=30, budget_used=700, budget_total=1000)
        assert 0.0 <= r.composite <= 100.0


def test_band_boundaries_exact():
    # GREEN at/above 80, YELLOW 60-79, RED 40-59, CRITICAL <40
    assert interpret(80.0)[0] == "GREEN"
    assert interpret(79.9)[0] == "YELLOW"
    assert interpret(60.0)[0] == "YELLOW"
    assert interpret(59.9)[0] == "RED"
    assert interpret(40.0)[0] == "RED"
    assert interpret(39.9)[0] == "CRITICAL"


def test_renormalization_keeps_0_100():
    s = EfficiencyScorer()
    # only one component present -> composite equals that component (capped)
    assert s.compute(retrieval_precision=100).composite == 100.0
    assert s.compute(avg_latency_ms=200).composite == 0.0