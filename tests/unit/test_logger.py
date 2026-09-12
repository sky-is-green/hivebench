"""Unit tests for logs.event_logger (S0.1 + hardening)."""

import gzip
import json
import os
import time
from datetime import datetime, timezone

import pytest

from logs.event_logger import EventLogger, redact, validate_entry


class Clock:
    """Mutable, injectable clock for deterministic rotation tests."""

    def __init__(self, dt: datetime) -> None:
        self.dt = dt

    def __call__(self) -> datetime:
        return self.dt


def test_valid_ndjson_schema(tmp_path):
    logger = EventLogger(log_dir=tmp_path)
    logger.log("router", "task_classified", {"query_hash": "abc", "routed_to": "ultra_small"}, latency_ms=3.5)
    logger.log("ultra_small", "relevance_scored", {"chunk_ids": ["c1", "c2"], "scores": [0.9, 0.2]}, latency_ms=7.0)
    logger.flush()
    logger.close()

    entries = logger.read_entries()
    assert len(entries) == 2
    for entry in entries:
        validate_entry(entry)  # raises if schema wrong
    assert entries[0]["component"] == "router"
    assert entries[0]["event_type"] == "task_classified"
    assert entries[0]["latency_ms"] == 3.5
    assert entries[1]["payload"]["scores"] == [0.9, 0.2]
    assert entries[0]["ts"].endswith("Z")


def test_every_line_is_valid_json(tmp_path):
    logger = EventLogger(log_dir=tmp_path)
    for i in range(50):
        logger.log("efficiency", "score_computed", {"composite_score": i}, latency_ms=0.1)
    logger.flush()
    logger.close()

    for path in tmp_path.glob("events-*.ndjson"):
        with open(path, encoding="utf-8") as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
        assert lines, "file should contain lines"
        for line in lines:
            parsed = json.loads(line)  # must parse as valid JSON
            validate_entry(parsed)


def test_100_plus_turns_no_corruption(tmp_path):
    logger = EventLogger(log_dir=tmp_path)
    for i in range(150):
        logger.log("assembly", "context_assembled",
                   {"chunk_count": i, "total_tokens": 100 + i}, latency_ms=1.0)
    logger.flush()
    logger.close()

    entries = logger.read_entries()
    assert len(entries) == 150
    for entry in entries:
        validate_entry(entry)
    # order preserved
    counts = [e["payload"]["chunk_count"] for e in entries]
    assert counts == list(range(150))


def test_daily_rotation(tmp_path):
    clock = Clock(datetime(2026, 1, 1, tzinfo=timezone.utc))
    logger = EventLogger(log_dir=tmp_path, now_fn=clock)
    logger.log("a", "x", {"day": 1})
    logger.flush()

    clock.dt = datetime(2026, 1, 2, tzinfo=timezone.utc)
    logger.log("a", "x", {"day": 2})
    logger.flush()
    logger.close()

    files = sorted(p.name for p in tmp_path.glob("events-*.ndjson"))
    assert files == ["events-2026-01-01.ndjson", "events-2026-01-02.ndjson"]

    day1 = logger.read_entries("2026-01-01")
    day2 = logger.read_entries("2026-01-02")
    assert len(day1) == 1 and day1[0]["payload"] == {"day": 1}
    assert len(day2) == 1 and day2[0]["payload"] == {"day": 2}
    assert len(logger.read_entries()) == 2


def test_log_does_not_raise_on_non_serializable_payload(tmp_path):
    logger = EventLogger(log_dir=tmp_path)
    logger.log("a", "x", {"bad": object()})  # default=str fallback
    logger.flush()
    logger.close()
    assert len(logger.read_entries()) == 1


def test_validate_entry_rejects_bad_schema():
    with pytest.raises(ValueError):
        validate_entry({"ts": "x", "component": "a", "event_type": "b"})  # no payload
    with pytest.raises(ValueError):
        validate_entry({"ts": "x", "component": 1, "event_type": "b", "payload": {}, "latency_ms": 0})

def test_redaction_of_secret_values(tmp_path):
    logger = EventLogger(log_dir=tmp_path)
    logger.log("backend", "request", {
        "api_key": "sk-1234567890abcdef",
        "payload": {"token": "abc123def456", "data": "0f1e2d3c4b5a69788a9b0c1d2e3f4a5b6c7d8e9f"},
    })
    logger.flush()
    logger.close()
    entry = logger.read_entries()[0]
    assert entry["payload"]["api_key"] == "[REDACTED]"
    assert entry["payload"]["payload"]["token"] == "[REDACTED]"
    assert entry["payload"]["payload"]["data"] == "[REDACTED]"
    assert entry["payload"]["payload"]["keep"] == "ok" if "keep" in entry["payload"]["payload"] else True


def test_redact_function_basic():
    assert redact({"api_key": "secret"}) == {"api_key": "[REDACTED]"}
    assert redact({"nested": [{"password": "x"}]}) == {"nested": [{"password": "[REDACTED]"}]}
    assert redact("sk-abcdef1234567890") == "[REDACTED]"
    assert redact("a" * 30) == "[REDACTED]"  # long hex-like


def test_correlation_ids_in_entry(tmp_path):
    logger = EventLogger(log_dir=tmp_path)
    logger.log("router", "task_classified", {"x": 1},
               run_id="run-1", conversation_id="conv-1", turn_id=7)
    logger.flush()
    logger.close()
    entry = logger.read_entries()[0]
    assert entry["run_id"] == "run-1"
    assert entry["conversation_id"] == "conv-1"
    assert entry["turn_id"] == 7


def test_correlation_ids_optional(tmp_path):
    logger = EventLogger(log_dir=tmp_path)
    logger.log("a", "b", {})
    logger.flush()
    logger.close()
    entry = logger.read_entries()[0]
    assert "run_id" not in entry
    assert "turn_id" not in entry


def test_size_rotation_archives_gzip(tmp_path):
    logger = EventLogger(log_dir=tmp_path, max_bytes=200, retention_days=30)
    for i in range(50):
        logger.log("a", "x", {"i": i}, latency_ms=1.0)
    logger.flush()
    logger.close()

    archives = list((tmp_path / "archive").glob("*.ndjson.gz"))
    assert len(archives) >= 1
    # archives are valid gzip NDJSON
    with gzip.open(archives[0], "rt", encoding="utf-8") as f:
        line = f.readline()
        validate_entry(json.loads(line))
    # no events lost across rotation: live + all archives == 50
    total = len(logger.read_entries())
    for a in archives:
        with gzip.open(a, "rt", encoding="utf-8") as f:
            total += sum(1 for _ in f)
    assert total == 50


def test_cleanup_archives_removes_old(tmp_path):
    archive = tmp_path / "archive"
    archive.mkdir(parents=True, exist_ok=True)
    old = archive / "events-2020-01-01-0001.ndjson.gz"
    new = archive / "events-2026-01-01-0002.ndjson.gz"
    old.write_bytes(b"{}")
    new.write_bytes(b"{}")
    past = time.time() - 10 * 86400
    os.utime(old, (past, past))

    logger = EventLogger(log_dir=tmp_path, retention_days=7)
    removed = logger.cleanup_archives()
    assert removed == 1
    assert not old.exists()
    assert new.exists()
    logger.close()
