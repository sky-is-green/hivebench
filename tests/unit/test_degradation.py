"""Unit tests for cortex.degradation (S3.4)."""

from cortex.congestion import CongestionReport
from cortex.degradation import GracefulDegradation


def _report(severity):
    return CongestionReport(
        severity=severity, queue_depth=0, avg_drone_latency_ms=0.0,
        pending_assemblies=0, recommended_actions=[], signal_breakdown={},
    )


def test_level_escalation():
    g = GracefulDegradation()
    assert g.current_level == 0
    g.update(_report("warning"))
    assert g.current_level == 1
    g.update(_report("critical"))
    assert g.current_level == 2
    g.update(_report("emergency"))
    assert g.current_level == 3


def test_recovery_one_level_at_a_time():
    g = GracefulDegradation()
    g.current_level = 3
    g.update(_report("normal"))
    assert g.current_level == 2
    g.update(_report("normal"))
    assert g.current_level == 1
    g.update(_report("normal"))
    assert g.current_level == 0


def test_no_downgrade_on_continued_stress():
    g = GracefulDegradation()
    g.update(_report("critical"))
    g.update(_report("critical"))
    assert g.current_level == 2


def test_skip_flags_per_level():
    g = GracefulDegradation()
    g.current_level = 1
    assert g.should_skip_medium() and g.should_skip_dedup()
    assert not g.should_skip_remembrance()

    g.current_level = 2
    assert g.should_skip_remembrance() and g.should_use_cached_only()

    g.current_level = 3
    assert g.should_fallback_fifo()


def test_reset():
    g = GracefulDegradation()
    g.current_level = 3
    g.reset()
    assert g.current_level == 0