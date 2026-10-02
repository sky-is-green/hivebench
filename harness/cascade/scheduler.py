"""Scheduling — latency model, path planning and expected-cost choice.

The scheduler's job (brief §"Optimization"): for each request, choose a path —
a subset of roles — that minimizes expected cost subject to a quality target.
Cost is dominated by generation; the expected cost of an escalation is
``P(escalate) × cost(escalation)``.

The skeleton is honest about what is not measured yet: the latency model
starts from the brief's planning table and is replaced by per-role oracle
measurements, and the quality target is applied only when the oracle provides
per-path quality (``quality=None`` keeps the route table's first path).  The
*arithmetic* — step expansion, async exclusion, escalation expectation — is
deterministic and pinned by tests, so filling in measurements cannot silently
change the policy shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Optional

from harness.cascade.paths import (
    Path,
    ROUTE_TABLE,
    get_path,
    paths_for_bucket,
    resolve_steps,
)
from harness.cascade.policy import Policy, escalation_probability
from harness.cascade.roles import Device, get_role


@dataclass(frozen=True)
class RoleCost:
    """One measured (or planned) cost point for a role on a device.

    ``per`` is ``"call"`` (fixed per invocation) or ``"token"`` (multiplied by
    the request's output length).
    """

    role: str
    device: Device
    ms: float
    per: str = "call"

    def estimate(self, *, out_len: int = 0) -> float:
        if self.per == "token":
            return float(self.ms) * max(0, int(out_len))
        return float(self.ms)


class LatencyModel:
    """Per-(role, device) costs with a brief-budget fallback.

    Unknown points fall back to the role's planning budget midpoint so the
    scheduler always produces a complete estimate; ``known(role)`` tells a
    caller whether the number is measured.
    """

    def __init__(self, costs: Iterable[RoleCost] = ()) -> None:
        self._costs: dict[tuple[str, str], RoleCost] = {}
        for cost in costs:
            self.add(cost)

    def add(self, cost: RoleCost) -> RoleCost:
        self._costs[(cost.role, cost.device.value)] = cost
        return cost

    def get(self, role: str, device: Device | str) -> Optional[RoleCost]:
        value = device.value if isinstance(device, Device) else str(device)
        return self._costs.get((role, value))

    def known(self, role: str) -> tuple[RoleCost, ...]:
        return tuple(c for c in self._costs.values() if c.role == role)

    def pick_device(self, role_id: str, preferred: Optional[Device] = None) -> Device:
        """Explicit device, else the role's first tier with a cost, else its
        first declared tier."""
        role = get_role(role_id)
        if preferred is not None:
            return preferred
        for device in role.devices:
            if self.get(role_id, device) is not None:
                return device
        return role.devices[0]

    def cost_ms(
        self,
        role_id: str,
        *,
        device: Optional[Device] = None,
        out_len: int = 0,
    ) -> float:
        role = get_role(role_id)
        picked = self.pick_device(role_id, device)
        cost = self.get(role_id, picked)
        if cost is not None:
            return cost.estimate(out_len=out_len)
        low, high = role.latency_ms
        mid = (float(low) + float(high)) / 2.0
        if role.latency_unit == "token":
            return mid * max(0, int(out_len))
        return mid

    def is_measured(self, role_id: str) -> bool:
        return bool(self.known(role_id))

    @classmethod
    def from_brief(cls) -> "LatencyModel":
        """A model whose every point is the brief's budget (all fallback)."""
        return cls()

    def to_dict(self) -> dict[str, Any]:
        return {
            "costs": [
                {"role": c.role, "device": c.device.value, "ms": c.ms, "per": c.per}
                for c in self._costs.values()
            ],
            "roles_measured": sorted({c.role for c in self._costs.values()}),
        }


@dataclass(frozen=True)
class StageEstimate:
    """One scheduled role invocation inside a plan."""

    role: str
    device: Device
    ms: float
    per: str = "call"
    async_: bool = False
    optional: bool = False
    gate: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "device": self.device.value,
            "ms": self.ms,
            "per": self.per,
            "async": self.async_,
            "optional": self.optional,
            "gate": self.gate,
        }


@dataclass(frozen=True)
class RouteDecision:
    """The C1 judgement the scheduler plans from."""

    bucket: str
    confidence: float = 1.0
    request_id: str = ""


@dataclass(frozen=True)
class Plan:
    """The scheduler's answer: a path, its stages, and the escalation math.

    ``expected_ms`` counts blocking stages only (async judgements hide behind
    generation); ``total_expected_ms`` adds the probability-weighted cost of
    the escalation path.
    """

    request_id: str
    path: str
    steps: tuple[StageEstimate, ...]
    blocking_ms: float
    async_ms: float
    escalation_path: Optional[str] = None
    escalation_probability: float = 0.0
    escalation_ms: float = 0.0
    quality: Optional[float] = None

    @property
    def expected_ms(self) -> float:
        return self.blocking_ms

    @property
    def total_expected_ms(self) -> float:
        return self.blocking_ms + self.escalation_probability * self.escalation_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "path": self.path,
            "steps": [s.to_dict() for s in self.steps],
            "blocking_ms": self.blocking_ms,
            "async_ms": self.async_ms,
            "escalation_path": self.escalation_path,
            "escalation_probability": self.escalation_probability,
            "escalation_ms": self.escalation_ms,
            "total_expected_ms": self.total_expected_ms,
            "quality": self.quality,
        }


def estimate_path(
    path: Path,
    model: LatencyModel,
    *,
    out_len: int = 0,
    device_overrides: Optional[Mapping[str, Device]] = None,
) -> tuple[StageEstimate, ...]:
    """Expand a path into per-stage estimates (``extends`` included)."""
    overrides = device_overrides or {}
    steps: list[StageEstimate] = []
    for step in resolve_steps(path):
        device = overrides.get(step.role, step.device)
        picked = model.pick_device(step.role, device)
        ms = model.cost_ms(step.role, device=picked, out_len=out_len)
        role = get_role(step.role)
        steps.append(
            StageEstimate(
                role=step.role,
                device=picked,
                ms=ms,
                per=role.latency_unit,
                async_=step.async_,
                optional=step.optional,
                gate=step.gate,
            )
        )
    return tuple(steps)


def _sum(steps: Iterable[StageEstimate], *, include_async: bool) -> float:
    return sum(
        step.ms for step in steps if include_async or not step.async_
    )


def plan_request(
    route: RouteDecision,
    *,
    model: Optional[LatencyModel] = None,
    policy: Optional[Policy] = None,
    out_len: int = 0,
    quality: Optional[Callable[[str], Optional[float]]] = None,
    device_overrides: Optional[Mapping[str, Device]] = None,
) -> Plan:
    """Choose a path for ``route`` and return the expected-cost plan.

    When ``quality`` is provided (the oracle's per-path metric), paths below
    ``Policy.quality_target`` are dropped before choosing the cheapest
    expected cost; otherwise the route table's first path is the choice and
    the scheduler only prices the escalation.
    """
    active_model = model or LatencyModel.from_brief()
    active_policy = policy or Policy()

    candidates = list(paths_for_bucket(route.bucket))
    if not candidates:
        candidates = [get_path(ROUTE_TABLE.get(route.bucket, "P2"))]
    if quality is not None:
        measured = [(p, quality(p.id)) for p in candidates]
        passing = [(p, q) for p, q in measured if q is not None and q >= active_policy.quality_target]
        if passing:
            candidates = [p for p, _ in passing]

    best: Optional[Plan] = None
    for path in candidates:
        steps = estimate_path(
            path, active_model, out_len=out_len, device_overrides=device_overrides
        )
        blocking = _sum(steps, include_async=False)
        async_ms = _sum(steps, include_async=True) - blocking
        probability = escalation_probability(route.confidence)
        escalation_path: Optional[str] = path.escalates_to[0] if path.escalates_to else None
        escalation_ms = 0.0
        if escalation_path is not None and probability > 0.0:
            esc_steps = estimate_path(
                get_path(escalation_path), active_model, out_len=out_len,
                device_overrides=device_overrides,
            )
            escalation_ms = _sum(esc_steps, include_async=False)
        plan = Plan(
            request_id=route.request_id,
            path=path.id,
            steps=steps,
            blocking_ms=blocking,
            async_ms=async_ms,
            escalation_path=escalation_path,
            escalation_probability=probability,
            escalation_ms=escalation_ms,
            quality=(quality(path.id) if quality is not None else None),
        )
        if best is None or plan.total_expected_ms < best.total_expected_ms:
            best = plan
    assert best is not None  # candidates is never empty
    return best
