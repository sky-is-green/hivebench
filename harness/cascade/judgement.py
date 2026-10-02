"""Async judgement plumbing — fire-and-consume with generation cancellation.

Routing, gating, scoring and verification are *judgements* (brief §"Async
judgements"): the pipeline fires them asynchronously and consumes their
results when ready, so a slow decision model never blocks the first token.

Two harness responsibilities live here, and nothing else can own them
(brief §"What must live in the harness"):

- **Batching** — judgements queued across concurrent requests run in one
  handler call per role, amortizing a decision model's cost.
- **Cancellation** — a judgement carries a request id and a generation id;
  when the generation is cancelled (rejection, escalation, client gone) its
  pending judgement is dropped and any late result is discarded.

The broker is a single-consumer, synchronous queue: tests and offline runs
drive it directly, and the serving layer owns the threading around it.
Handlers are injected (models stubbed), so the whole flow is unit-testable
with no model in sight.

Deliberately minimal: no asyncio dependency, no wall-clock scheduling, no
retries.  A handler that raises marks its batch's results as errors — the
pipeline treats an errored judgement exactly like a timeout (degrade, don't
block).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence


class JudgementState(str, Enum):
    PENDING = "pending"
    DONE = "done"
    DROPPED = "dropped"
    ERROR = "error"


@dataclass(frozen=True)
class Judgement:
    """A request for one learned decision.

    ``generation_id`` increments on every escalation/regeneration of the same
    request, which is what makes cancellation precise: cancelling generation
    *n* must not drop generation *n+1*.
    """

    id: str
    request_id: str
    generation_id: int
    role: str
    kind: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    submitted_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "request_id": self.request_id,
            "generation_id": self.generation_id,
            "role": self.role,
            "kind": self.kind,
            "payload": dict(self.payload),
        }


@dataclass(frozen=True)
class JudgementResult:
    """The answer to one :class:`Judgement`.

    ``decision`` is the model's verdict in role-specific vocabulary
    (``"accept"``/``"reject"``, a bucket label, a score); ``confidence`` is the
    calibrated probability the policy thresholds read.
    """

    judgement_id: str
    request_id: str
    generation_id: int
    role: str
    decision: str
    confidence: float
    payload: Mapping[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    state: JudgementState = JudgementState.DONE
    error: Optional[str] = None

    @property
    def dropped(self) -> bool:
        return self.state == JudgementState.DROPPED

    @property
    def ok(self) -> bool:
        return self.state == JudgementState.DONE

    def to_dict(self) -> dict[str, Any]:
        return {
            "judgement_id": self.judgement_id,
            "request_id": self.request_id,
            "generation_id": self.generation_id,
            "role": self.role,
            "decision": self.decision,
            "confidence": self.confidence,
            "payload": dict(self.payload),
            "latency_ms": self.latency_ms,
            "state": self.state.value,
            "error": self.error,
        }


def result_for(
    judgement: Judgement,
    decision: str,
    confidence: float,
    *,
    latency_ms: float = 0.0,
    payload: Optional[Mapping[str, Any]] = None,
) -> JudgementResult:
    """Build a done-result for ``judgement`` (handlers' usual return shape)."""
    return JudgementResult(
        judgement_id=judgement.id,
        request_id=judgement.request_id,
        generation_id=judgement.generation_id,
        role=judgement.role,
        decision=decision,
        confidence=float(confidence),
        payload=dict(payload or {}),
        latency_ms=float(latency_ms),
    )


#: A handler answers every judgement in its batch.  It may return fewer
#: results than inputs (the broker then treats the missing ones as errors).
JudgementHandler = Callable[[Sequence[Judgement]], Iterable[JudgementResult]]


class JudgementBroker:
    """Queue + batch + cancellation for asynchronous judgements."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._handlers: dict[str, JudgementHandler] = {}
        self._pending: list[Judgement] = []
        self._inbox: list[JudgementResult] = []
        self._cancelled: set[tuple[str, int]] = set()
        self._dropped_count = 0

    # -- registration -------------------------------------------------

    def register_handler(self, role: str, handler: JudgementHandler) -> None:
        self._handlers[role] = handler

    def handler_roles(self) -> tuple[str, ...]:
        return tuple(self._handlers)

    # -- submission ---------------------------------------------------

    def submit(self, judgement: Judgement) -> Judgement:
        """Queue a judgement; cancelled generations are dropped on arrival."""
        if judgement.submitted_ms == 0.0:
            judgement = Judgement(
                id=judgement.id,
                request_id=judgement.request_id,
                generation_id=judgement.generation_id,
                role=judgement.role,
                kind=judgement.kind,
                payload=judgement.payload,
                submitted_ms=self._clock(),
            )
        if not self.is_generation_live(judgement.request_id, judgement.generation_id):
            self._dropped_count += 1
            self._inbox.append(self._mk_state(judgement, JudgementState.DROPPED, "cancelled"))
            return judgement
        self._pending.append(judgement)
        return judgement

    def pending(self) -> tuple[Judgement, ...]:
        return tuple(self._pending)

    def pending_by_role(self) -> dict[str, tuple[Judgement, ...]]:
        grouped: dict[str, list[Judgement]] = {}
        for judgement in self._pending:
            grouped.setdefault(judgement.role, []).append(judgement)
        return {role: tuple(items) for role, items in grouped.items()}

    # -- cancellation -------------------------------------------------

    def cancel(self, request_id: str, generation_id: int) -> int:
        """Invalidate one generation; returns judgments dropped immediately.

        Late results for the same generation are discarded by :meth:`consume`.
        """
        self._cancelled.add((request_id, generation_id))
        keep: list[Judgement] = []
        dropped = 0
        for judgement in self._pending:
            if judgement.request_id == request_id and judgement.generation_id == generation_id:
                dropped += 1
                self._dropped_count += 1
                self._inbox.append(
                    self._mk_state(judgement, JudgementState.DROPPED, "cancelled")
                )
            else:
                keep.append(judgement)
        self._pending = keep
        return dropped

    def is_generation_live(self, request_id: str, generation_id: int) -> bool:
        return (request_id, generation_id) not in self._cancelled

    # -- execution ----------------------------------------------------

    def run_pending(self) -> tuple[JudgementResult, ...]:
        """Run queued judgements, one batch per role.

        Only roles with a registered handler run; the rest stay pending until
        a handler appears (models are pluggable).  A handler that raises marks
        its whole batch as errored and the queue drains.
        """
        grouped = self.pending_by_role()
        produced: list[JudgementResult] = []
        for role, batch in grouped.items():
            handler = self._handlers.get(role)
            if handler is None:
                continue
            self._pending = [j for j in self._pending if j.role != role]
            started = self._clock()
            try:
                results = list(handler(batch))
            except Exception as exc:  # degrade, never block the pipeline
                elapsed = (self._clock() - started) * 1000.0
                produced.extend(
                    self._mk_state(j, JudgementState.ERROR, str(exc), latency_ms=elapsed)
                    for j in batch
                )
                self._inbox.extend(produced[-len(batch):])
                continue
            elapsed = (self._clock() - started) * 1000.0
            by_id = {r.judgement_id: r for r in results if r.judgement_id}
            for judgement in batch:
                result = by_id.get(judgement.id)
                if result is None:
                    result = self._mk_state(
                        judgement, JudgementState.ERROR, "handler returned no result",
                        latency_ms=elapsed,
                    )
                elif result.state == JudgementState.DONE and result.latency_ms == 0.0:
                    result = JudgementResult(
                        judgement_id=result.judgement_id,
                        request_id=result.request_id,
                        generation_id=result.generation_id,
                        role=result.role,
                        decision=result.decision,
                        confidence=result.confidence,
                        payload=result.payload,
                        latency_ms=elapsed,
                        state=result.state,
                        error=result.error,
                    )
                produced.append(result)
                self._inbox.append(result)
        return tuple(produced)

    def consume(self) -> tuple[JudgementResult, ...]:
        """Drain live results; drops stale-generation results on the way out."""
        live: list[JudgementResult] = []
        for result in self._inbox:
            if result.state != JudgementState.DROPPED and not self.is_generation_live(
                result.request_id, result.generation_id
            ):
                # A handler answered after the generation was cancelled: the
                # result is discarded exactly like a pending one.
                self._dropped_count += 1
                continue
            if result.state == JudgementState.DROPPED:
                continue  # already counted at cancel/drop time
            live.append(result)
        self._inbox = []
        return tuple(live)

    def drain_dropped(self) -> int:
        """Number of judgements dropped so far (cancellation telemetry)."""
        return self._dropped_count

    def pending_count(self) -> int:
        return len(self._pending)

    def inbox_count(self) -> int:
        return len(self._inbox)

    @staticmethod
    def _mk_state(
        judgement: Judgement,
        state: JudgementState,
        error: Optional[str] = None,
        *,
        latency_ms: float = 0.0,
    ) -> JudgementResult:
        return JudgementResult(
            judgement_id=judgement.id,
            request_id=judgement.request_id,
            generation_id=judgement.generation_id,
            role=judgement.role,
            decision="",
            confidence=0.0,
            payload=dict(judgement.payload),
            latency_ms=latency_ms,
            state=state,
            error=error,
        )
