"""Unit tests for cortex.health (S3.3)."""

from cortex.health import PipelineHealthMonitor, RollingMetrics


class FakeLogger:
    def __init__(self):
        self.events = []

    def log(self, component, event_type, payload, latency_ms=0.0):
        self.events.append((component, event_type, payload))


def test_rolling_average_of_last_10():
    m = RollingMetrics(latency_window=10)
    for i in range(1, 11):
        m.record_drone_latency(i)
    assert m.avg_drone_latency_ms == 5.5
    m.record_drone_latency(100)  # slides window
    assert m.avg_drone_latency_ms == (2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 100) / 10


def test_rolling_average_empty_is_none():
    assert RollingMetrics().avg_drone_latency_ms is None


def test_health_normal_no_log():
    logger = FakeLogger()
    monitor = PipelineHealthMonitor(logger=logger)
    report = monitor.check_congestion()
    assert report.severity == "normal"
    assert logger.events == []


def test_health_warning_logs_congestion():
    logger = FakeLogger()
    monitor = PipelineHealthMonitor(logger=logger)
    monitor.metrics.queue_depth = 6  # warning
    report = monitor.check_congestion()
    assert report.severity == "warning"
    assert ("congestion", "congestion_detected") in [
        (e[0], e[1]) for e in logger.events
    ]


def test_health_critical_from_latency():
    logger = FakeLogger()
    monitor = PipelineHealthMonitor(logger=logger)
    for _ in range(10):
        monitor.metrics.record_drone_latency(150)
    report = monitor.check_congestion()
    assert report.severity == "critical"


def test_rolling_window_record():
    m = RollingMetrics(window_size=3)
    m.record("a", 1)
    m.record("b", 2)
    m.record("c", 3)
    m.record("d", 4)  # evicts a
    assert list(m.window) == [("b", 2), ("c", 3), ("d", 4)]