"""Unit tests for cortex.rollback (S5.3)."""

from cortex.rollback import AutomatedRollback


def test_too_few_samples_no_rollback():
    rb = AutomatedRollback()
    assert rb.check([80, 80]).should_rollback is False


def test_immediate_rollback_pes_below_50():
    rb = AutomatedRollback()
    pes = [80] * 5 + [40] * 10
    decision = rb.check(pes)
    assert decision.should_rollback is True
    assert "50" in decision.reason


def test_warning_rollback_pes_below_60_for_25():
    rb = AutomatedRollback()
    pes = [55] * 25
    decision = rb.check(pes)
    assert decision.should_rollback is True
    assert "60" in decision.reason


def test_trend_rollback_declining():
    rb = AutomatedRollback()
    pes = [90 - i * 1.0 for i in range(50)]  # slope -1.0
    assert rb.check(pes).should_rollback is True


def test_healthy_no_rollback():
    rb = AutomatedRollback()
    assert rb.check([80] * 30).should_rollback is False


def test_borderline_healthy_no_rollback():
    rb = AutomatedRollback()
    # not all < 50 in last 10, not all < 60 in last 25, no trend
    pes = [70] * 30
    assert rb.check(pes).should_rollback is False