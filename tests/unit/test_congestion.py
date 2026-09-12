"""Unit tests for cortex.congestion (S0.3)."""

from cortex.congestion import CongestionDetector, RollingAverage


def test_normal_all_signals():
    report = CongestionDetector().evaluate(queue_depth=0, avg_drone_latency_ms=10, pending_assemblies=0)
    assert report.severity == "normal"
    assert report.recommended_actions == []


def test_queue_warning_batches():
    report = CongestionDetector().evaluate(queue_depth=6)
    assert report.severity == "warning"
    assert "batch_similar_chunks" in report.recommended_actions


def test_queue_critical_skips_medium():
    report = CongestionDetector().evaluate(queue_depth=16)
    assert report.severity == "critical"
    assert "skip_medium_drone" in report.recommended_actions
    assert "batch_similar_chunks" in report.recommended_actions


def test_latency_warning():
    report = CongestionDetector().evaluate(avg_drone_latency_ms=30)
    assert report.severity == "warning"
    assert "investigate_gpu_contention" in report.recommended_actions


def test_latency_critical_uses_cached_embeddings():
    report = CongestionDetector().evaluate(avg_drone_latency_ms=120)
    assert report.severity == "critical"
    assert "use_cached_embeddings" in report.recommended_actions


def test_backlog_warning_queues_messages():
    report = CongestionDetector().evaluate(pending_assemblies=1)
    assert report.severity == "warning"
    assert "queue_messages" in report.recommended_actions


def test_backlog_critical_aggressive_compression():
    report = CongestionDetector().evaluate(pending_assemblies=2)
    assert report.severity == "critical"
    assert "aggressive_compression" in report.recommended_actions


def test_multiple_critical_signals_emergency():
    report = CongestionDetector().evaluate(queue_depth=16, avg_drone_latency_ms=120)
    assert report.severity == "emergency"
    assert "fallback_to_truncation" in report.recommended_actions


def test_signal_breakdown():
    report = CongestionDetector().evaluate(queue_depth=16, avg_drone_latency_ms=10, pending_assemblies=0)
    assert report.signal_breakdown == {"queue": 2, "latency": 0, "backlog": 0}


def test_rolling_average():
    ra = RollingAverage(window_size=3)
    assert ra.value is None
    ra.push(1)
    assert ra.value == 1.0
    ra.push(2)
    assert ra.value == 1.5
    ra.push(3)
    assert ra.value == 2.0
    ra.push(10)  # window slides: (2+3+10)/3
    assert ra.value == 5.0
    assert len(ra) == 3