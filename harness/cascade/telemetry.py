"""Per-role telemetry — the harness feeds the offline tuner.

The brief keeps telemetry in the harness because only the harness sees all
concurrent work ("the offline tuner needs per-role numbers from real
traffic").  This is a deliberately dumb append-only log plus a summary: no
sampling, no background threads, no clocks the caller cannot control.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RoleRecord:
    """One role invocation's outcome, as the tuner wants to read it."""

    request_id: str
    path: str
    role: str
    device: str
    latency_ms: float
    tokens: int = 0
    outcome: str = "ok"
    async_: bool = False
    escalated: bool = False
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "path": self.path,
            "role": self.role,
            "device": self.device,
            "latency_ms": self.latency_ms,
            "tokens": self.tokens,
            "outcome": self.outcome,
            "async": self.async_,
            "escalated": self.escalated,
            "detail": self.detail,
        }


def _percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile (q in [0, 1]); deterministic for small n."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index]


class TelemetryLog:
    """Append-only role records + a per-role summary."""

    def __init__(self) -> None:
        self._records: list[RoleRecord] = []

    def record(self, record: RoleRecord) -> RoleRecord:
        self._records.append(record)
        return record

    def log(
        self,
        request_id: str,
        path: str,
        role: str,
        device: str,
        latency_ms: float,
        *,
        tokens: int = 0,
        outcome: str = "ok",
        async_: bool = False,
        escalated: bool = False,
        detail: str = "",
    ) -> RoleRecord:
        return self.record(
            RoleRecord(
                request_id=request_id,
                path=path,
                role=role,
                device=device,
                latency_ms=float(latency_ms),
                tokens=int(tokens),
                outcome=outcome,
                async_=async_,
                escalated=escalated,
                detail=detail,
            )
        )

    def records(self) -> tuple[RoleRecord, ...]:
        return tuple(self._records)

    def by_role(self) -> dict[str, tuple[RoleRecord, ...]]:
        grouped: dict[str, list[RoleRecord]] = {}
        for record in self._records:
            grouped.setdefault(record.role, []).append(record)
        return {role: tuple(items) for role, items in grouped.items()}

    def by_request(self) -> dict[str, tuple[RoleRecord, ...]]:
        grouped: dict[str, list[RoleRecord]] = {}
        for record in self._records:
            grouped.setdefault(record.request_id, []).append(record)
        return {rid: tuple(items) for rid, items in grouped.items()}

    def summary(self) -> dict[str, dict[str, Any]]:
        """Per-role count, latency percentiles, tokens, outcome rates."""
        out: dict[str, dict[str, Any]] = {}
        for role, records in self.by_role().items():
            latencies = [r.latency_ms for r in records]
            count = len(records)
            bad = sum(1 for r in records if r.outcome not in ("ok",))
            out[role] = {
                "count": count,
                "latency_ms": {
                    "mean": round(sum(latencies) / count, 6) if count else 0.0,
                    "p50": round(_percentile(latencies, 0.50), 6),
                    "p95": round(_percentile(latencies, 0.95), 6),
                    "max": round(max(latencies), 6) if count else 0.0,
                },
                "tokens": sum(r.tokens for r in records),
                "non_ok": bad,
                "non_ok_rate": round(bad / count, 6) if count else 0.0,
                "escalated": sum(1 for r in records if r.escalated),
                "async": sum(1 for r in records if r.async_),
            }
        return out

    def clear(self) -> None:
        self._records = []

    def __len__(self) -> int:
        return len(self._records)

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": [r.to_dict() for r in self._records],
            "summary": self.summary(),
        }
