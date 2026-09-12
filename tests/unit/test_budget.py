"""Unit tests for focal.budget (S2.5)."""

from focal.budget import AdaptiveBudget


def test_budget_ranges_per_route():
    budget = AdaptiveBudget()
    assert budget.compute("ultra_small", 0) == 1000
    assert budget.compute("medium", 0) == 3000
    assert budget.compute("escalation", 0) == 4000


def test_budget_scales_with_high_relevance():
    budget = AdaptiveBudget()
    assert budget.compute("ultra_small", 10) == 3000   # fill factor 1.0
    assert budget.compute("ultra_small", 5) == 2000    # fill factor 0.5
    assert budget.compute("ultra_small", 20) == 3000   # capped at 1.0


def test_budget_capped_by_max_context_minus_headroom():
    budget = AdaptiveBudget()
    # escalation range would be 6000 but max_context - headroom caps it
    assert budget.compute("escalation", 10, max_context=5000) == 5000 - 2048
    assert budget.compute("escalation", 10, max_context=8192) == 6000


def test_unknown_route_uses_default_range():
    budget = AdaptiveBudget()
    assert budget.compute("unknown", 0) == 2000
    assert budget.compute("unknown", 10) == 4000