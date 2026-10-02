"""Unit tests for the judgement broker and the telemetry log."""

from __future__ import annotations

import pytest

from harness.cascade.judgement import (
    Judgement,
    JudgementBroker,
    JudgementResult,
    JudgementState,
    result_for,
)
from harness.cascade.telemetry import RoleRecord, TelemetryLog


class Clock:
    """A hand-wound monotonic clock (seconds), so latency is deterministic."""

    def __init__(self, value: float = 100.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def tick(self, seconds: float) -> None:
        self.value += seconds


def _j(jid: str, role: str = "C1", request: str = "r1", generation: int = 0) -> Judgement:
    return Judgement(
        id=jid, request_id=request, generation_id=generation, role=role,
        kind="route", payload={"prompt": jid},
    )


def test_submit_stamps_time_and_queues():
    clock = Clock(12.5)
    broker = JudgementBroker(clock=clock)
    stamped = broker.submit(_j("a"))
    assert stamped.submitted_ms == 12.5
    assert broker.pending_count() == 1
    assert broker.pending()[0].id == "a"


def test_batching_groups_by_role_in_one_handler_call():
    broker = JudgementBroker()
    batches: list[list[str]] = []

    def handler(batch):
        batches.append([j.id for j in batch])
        return [result_for(j, "qa", 0.9, latency_ms=7.0) for j in batch]

    broker.register_handler("C1", handler)
    broker.submit(_j("a", request="r1"))
    broker.submit(_j("b", request="r2"))
    broker.submit(_j("c", request="r1"))
    produced = broker.run_pending()
    assert batches == [["a", "b", "c"]]
    assert len(produced) == 3
    assert all(r.latency_ms == 7.0 for r in produced)
    consumed = broker.consume()
    assert {r.judgement_id for r in consumed} == {"a", "b", "c"}
    assert broker.inbox_count() == 0


def test_handler_response_shape_is_per_judgement():
    broker = JudgementBroker()
    broker.register_handler(
        "C1", lambda batch: [result_for(batch[0], "qa", 0.7)]
    )
    broker.submit(_j("a"))
    results = broker.run_pending()
    assert results[0].state is JudgementState.DONE
    # The handler answered one of one, so no error row is synthesized.
    assert broker.consume()[0].decision == "qa"


def test_cancel_drops_pending_and_late_results():
    broker = JudgementBroker()
    broker.register_handler("C1", lambda batch: [result_for(j, "qa", 0.9) for j in batch])
    broker.submit(_j("a", request="r1", generation=0))
    broker.submit(_j("b", request="r1", generation=1))
    assert broker.cancel("r1", 0) == 1
    assert broker.pending_count() == 1
    broker.run_pending()
    consumed = broker.consume()
    assert [r.judgement_id for r in consumed] == ["b"]
    assert broker.drain_dropped() == 1


def test_cancel_after_run_discards_the_result():
    broker = JudgementBroker()
    broker.register_handler("C1", lambda batch: [result_for(j, "qa", 0.9) for j in batch])
    broker.submit(_j("a", request="r1", generation=0))
    broker.run_pending()
    assert broker.inbox_count() == 1
    broker.cancel("r1", 0)
    assert broker.consume() == ()
    assert broker.drain_dropped() == 1


def test_cancelled_generation_is_dropped_on_arrival():
    broker = JudgementBroker()
    broker.cancel("r1", 3)
    broker.submit(_j("late", request="r1", generation=3))
    assert broker.pending_count() == 0
    assert broker.drain_dropped() == 1
    assert broker.consume() == ()


def test_handler_error_degrades_not_blocks():
    broker = JudgementBroker()

    def boom(batch):
        raise RuntimeError("model exploded")

    broker.register_handler("C1", boom)
    broker.submit(_j("a"))
    results = broker.run_pending()
    assert results[0].state is JudgementState.ERROR
    assert "model exploded" in (results[0].error or "")
    assert broker.consume()[0].state is JudgementState.ERROR


def test_unhandled_role_stays_pending_until_a_handler_arrives():
    broker = JudgementBroker()
    broker.submit(_j("a", role="D2"))
    assert broker.run_pending() == ()
    assert broker.pending_count() == 1
    broker.register_handler("D2", lambda batch: [result_for(j, "accept", 0.9) for j in batch])
    broker.run_pending()
    assert broker.consume()[0].decision == "accept"


def test_handler_that_returns_nothing_is_an_error_row():
    broker = JudgementBroker()
    broker.register_handler("C1", lambda batch: [])
    broker.submit(_j("a"))
    results = broker.run_pending()
    assert results[0].state is JudgementState.ERROR
    assert "no result" in (results[0].error or "")


def test_telemetry_log_summary_and_grouping():
    log = TelemetryLog()
    log.log("r1", "P2", "E2", "dgpu0", 100.0, tokens=50)
    log.log("r1", "P2", "E2", "dgpu0", 200.0, tokens=60)
    log.log("r1", "P2", "D2", "igpu", 30.0, outcome="reject", async_=True)
    log.log("r2", "P4", "E3", "dgpu1", 500.0, tokens=200, escalated=True)
    assert len(log) == 4
    assert set(log.by_role()) == {"E2", "D2", "E3"}
    assert set(log.by_request()) == {"r1", "r2"}
    summary = log.summary()
    assert summary["E2"]["count"] == 2
    assert summary["E2"]["tokens"] == 110
    assert summary["E2"]["latency_ms"]["p50"] == 100.0
    assert summary["E2"]["latency_ms"]["max"] == 200.0
    assert summary["D2"]["non_ok_rate"] == 1.0
    assert summary["E3"]["escalated"] == 1
    assert summary["D2"]["async"] == 1
    log.clear()
    assert len(log) == 0
