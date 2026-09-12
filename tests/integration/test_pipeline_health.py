"""Integration: congestion simulation -> degradation activate -> recover (S3.4)."""

from cortex.degradation import GracefulDegradation
from cortex.health import PipelineHealthMonitor


def test_congestion_escalates_and_degrades_then_recovers():
    monitor = PipelineHealthMonitor(logger=None)
    degradation = GracefulDegradation()

    # 1. Normal.
    assert monitor.check_congestion().severity == "normal"
    degradation.update(monitor.check_congestion())
    assert degradation.current_level == 0

    # 2. Inject load: deep queue + high drone latency -> critical/emergency.
    monitor.metrics.queue_depth = 16
    for _ in range(10):
        monitor.metrics.record_drone_latency(150)
    report = monitor.check_congestion()
    assert report.severity in ("critical", "emergency")
    degradation.update(report)
    assert degradation.current_level >= 2
    assert degradation.should_skip_medium()
    assert degradation.should_skip_remembrance()

    # 3. Clear the load.
    monitor.metrics.queue_depth = 0
    for _ in range(10):
        monitor.metrics.record_drone_latency(5)

    # 4. Recovery is gradual (one level per update, per Appendix C.3).
    levels = []
    for _ in range(5):
        report = monitor.check_congestion()
        levels.append(degradation.update(report))
    assert levels[0] <= 2          # recovered at most one level on first step
    assert levels[-1] == 0         # fully recovered
    assert not degradation.should_fallback_fifo()


def test_emergency_triggers_fifo_fallback():
    monitor = PipelineHealthMonitor(logger=None)
    degradation = GracefulDegradation()

    monitor.metrics.queue_depth = 20
    for _ in range(10):
        monitor.metrics.record_drone_latency(200)
    monitor.metrics.pending_assemblies = 3

    report = monitor.check_congestion()
    degradation.update(report)
    assert degradation.current_level == 3
    assert degradation.should_fallback_fifo()