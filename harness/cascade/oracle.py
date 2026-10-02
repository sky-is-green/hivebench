"""Per-role oracle records and Pareto frontiers (pilot stage 0).

Pilot stage 0 (brief §"Pilot stages"): *for each role, measure candidates
(quality and latency per device) and build the per-role frontier.  No router
yet.*  The measurement itself runs through :mod:`harness.stack` (load a
candidate, run the role's eval set); this module owns the record schema and
the frontier arithmetic so the results are comparable and auditable.

Quality is "higher is better" by construction (an accuracy, a score, or a
sign-flipped error); latency is milliseconds per call or per token, matching
the role's ``latency_unit``.  The front is computed within one
``(role, unit)`` group — mixing units is a caller bug.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

from harness.cascade.roles import Device


@dataclass(frozen=True)
class CandidateMeasurement:
    """One candidate measured on one role (one device, one precision)."""

    candidate: str
    role: str
    device: Device
    quality: float
    latency_ms: float
    unit: str = "call"
    ctx: int = 0
    precision: str = "bf16"
    bytes_gb: Optional[float] = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "role": self.role,
            "device": self.device.value,
            "quality": self.quality,
            "latency_ms": self.latency_ms,
            "unit": self.unit,
            "ctx": self.ctx,
            "precision": self.precision,
            "bytes_gb": self.bytes_gb,
            "notes": self.notes,
        }


def dominates(a: CandidateMeasurement, b: CandidateMeasurement) -> bool:
    """True when ``a`` is at least as good as ``b`` on both axes, strictly
    better on one (quality up, latency down)."""
    return (
        a.quality >= b.quality
        and a.latency_ms <= b.latency_ms
        and (a.quality > b.quality or a.latency_ms < b.latency_ms)
    )


def pareto_frontier(
    rows: Sequence[CandidateMeasurement],
) -> tuple[CandidateMeasurement, ...]:
    """Non-dominated rows, in input order.

    Ties (identical quality and latency) do not dominate each other, so both
    survive — a tie is a real measurement outcome (e.g. two precisions of the
    same model) and the caller decides by bytes.
    """
    return tuple(
        row for row in rows if not any(dominates(other, row) for other in rows if other is not row)
    )


def frontier_by_role(
    rows: Iterable[CandidateMeasurement],
) -> dict[str, tuple[CandidateMeasurement, ...]]:
    """Per-role,(device,precision) frontiers from a flat measurement list.

    Grouping is ``(role, unit)``: a frontier only compares like with like.
    Candidates of the same id measured on two devices both stay in their
    group's frontier (they are different placement options).
    """
    groups: dict[tuple[str, str], list[CandidateMeasurement]] = {}
    for row in rows:
        groups.setdefault((row.role, row.unit), []).append(row)
    return {role: pareto_frontier(group) for (role, _unit), group in groups.items()}


def best_per_candidate(
    rows: Iterable[CandidateMeasurement],
) -> dict[str, CandidateMeasurement]:
    """The best measurement per candidate id — the form the registry keeps.

    Best = highest quality; ties broken by lower latency.
    """
    best: dict[str, CandidateMeasurement] = {}
    for row in rows:
        current = best.get(row.candidate)
        if current is None or (row.quality, -row.latency_ms) > (
            current.quality, -current.latency_ms
        ):
            best[row.candidate] = row
    return best
