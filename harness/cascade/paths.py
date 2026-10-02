"""Cascade path definitions P0–P6 and escalation triggers.

A *path* is the brief's request lifecycle (brief §"Path map"): an ordered set
of roles for one class of request.  Paths are data, not code — the scheduler
reads them, the harness executes them, and adding a path is a registry edit.

``P3``/``P4`` extend earlier paths (``extends``) so shared stages are declared
once; :func:`resolve_steps` expands the chain.  Steps flagged ``async_`` are
judgements: they are fired and consumed when ready and never block the first
token (brief §"Async judgements").
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from harness.cascade.roles import Device, get_role


class EscalationTrigger(str, Enum):
    """Why a request moves up a tier (brief §"Escalation triggers")."""

    ROUTE_LOW_CONFIDENCE = "route-low-confidence"
    VERIFIER_REJECT = "verifier-reject"
    BUDGET_EXHAUSTED = "budget-exhausted"
    TOOL_FAILED = "tool-failed"


ESCALATION_TRIGGERS: tuple[EscalationTrigger, ...] = tuple(EscalationTrigger)


@dataclass(frozen=True)
class PathStep:
    """One role invocation inside a path.

    ``device`` pins the step to a tier when the role has several; ``None``
    means "the role's own preference order decides".  ``gate`` names the role
    whose verdict gates this step (``None`` = ungated).
    """

    role: str
    optional: bool = False
    async_: bool = False
    device: Optional[Device] = None
    gate: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "optional": self.optional,
            "async": self.async_,
            "device": self.device.value if self.device else None,
            "gate": self.gate,
        }


@dataclass(frozen=True)
class Path:
    """One request class.

    ``buckets`` are the C1 labels that select this path; ``devices`` is the
    union of the tiers its steps run on (the residency constraint);
    ``latency_class`` is the brief's qualitative class.
    """

    id: str
    name: str
    trigger: str
    buckets: tuple[str, ...]
    devices: tuple[Device, ...]
    latency_class: str
    steps: tuple[PathStep, ...]
    extends: Optional[str] = None
    loop: bool = False
    escalates_to: tuple[str, ...] = ()
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "trigger": self.trigger,
            "buckets": list(self.buckets),
            "devices": [d.value for d in self.devices],
            "latency_class": self.latency_class,
            "steps": [s.to_dict() for s in self.steps],
            "extends": self.extends,
            "loop": self.loop,
            "escalates_to": list(self.escalates_to),
        }


#: The brief's named paths (§"Named paths").  ``P3`` extends ``P2``: the full
#: escalated path is P2's stages followed by the 9B generation and its check.
PATHS: tuple[Path, ...] = (
    Path(
        id="P0", name="cache hit", trigger="near-duplicate prompt",
        buckets=("cache",),
        devices=(Device.CPU,),
        latency_class="~1-10 ms",
        steps=(PathStep("A1"),),
        notes="prefix/semantic cache lookup only; no generation",
    ),
    Path(
        id="P1", name="fast", trigger="easy prompt, high route confidence",
        buckets=("easy",),
        devices=(Device.CPU, Device.DGPU0),
        latency_class="generation-bound, 2B",
        steps=(
            PathStep("C1", async_=True),
            PathStep("E1"),
        ),
        escalates_to=("P3", "P5"),
    ),
    Path(
        id="P2", name="standard QA", trigger="normal question",
        buckets=("qa",),
        devices=(Device.CPU, Device.DGPU0),
        latency_class="+ retrieval 10-60 ms",
        steps=(
            PathStep("C1", async_=True),
            PathStep("C2"),
            PathStep("B1", gate="C2"),
            PathStep("B2", gate="C2"),
            PathStep("E2"),
            PathStep("D1", async_=True),
            PathStep("D2"),
        ),
        escalates_to=("P3",),
    ),
    Path(
        id="P3", name="escalated", trigger="low confidence, or D rejects",
        buckets=(),
        devices=(Device.DGPU1, Device.DGPU0),
        latency_class="+ 9B generation",
        steps=(
            PathStep("E3", gate="D2"),
            PathStep("D3", async_=True),
        ),
        extends="P2",
        escalates_to=("P5",),
        notes="P2 stages then the larger generator; verify runs pipelined",
    ),
    Path(
        id="P4", name="agentic/coding", trigger="code or tool request",
        buckets=("code", "tool"),
        devices=(Device.DGPU1, Device.DGPU0),
        latency_class="tool-loop bound",
        steps=(
            PathStep("C1", async_=True),
            PathStep("C3"),
            PathStep("E3"),
            PathStep("D3", async_=True),
        ),
        loop=True,
        escalates_to=("P5",),
    ),
    Path(
        id="P5", name="deep reasoning", trigger="hard reasoning",
        buckets=("reasoning",),
        devices=(Device.DGPU1,),
        latency_class="thinking-bound, H1-capped",
        steps=(
            PathStep("C1", async_=True),
            PathStep("E4"),
            PathStep("H1", async_=True),
            PathStep("D3"),
        ),
    ),
    Path(
        id="P6", name="vision", trigger="image input",
        buckets=("vision",),
        devices=(Device.IGPU, Device.DGPU0),
        latency_class="+ vision encode",
        steps=(
            PathStep("A2"),
            PathStep("E2"),
            PathStep("D3", async_=True),
        ),
    ),
)

_PATH_BY_ID: dict[str, Path] = {path.id: path for path in PATHS}

#: Paths that do not extend another path.
ROOT_PATHS: tuple[Path, ...] = tuple(p for p in PATHS if p.extends is None)

#: First path per bucket — the skeleton's routing table.  Once the oracle has
#: per-role quality, the scheduler may pick among all ``paths_for_bucket``.
ROUTE_TABLE: dict[str, str] = {
    bucket: path.id for path in PATHS for bucket in path.buckets
}


def get_path(path_id: str) -> Path:
    try:
        return _PATH_BY_ID[path_id]
    except KeyError:
        known = ", ".join(_PATH_BY_ID)
        raise KeyError(f"unknown cascade path {path_id!r} (known: {known})") from None


def paths_for_bucket(bucket: str) -> tuple[Path, ...]:
    """Every path that can serve a C1 bucket, in declaration order."""
    return tuple(path for path in PATHS if bucket in path.buckets)


def resolve_steps(path: Path) -> tuple[PathStep, ...]:
    """Expand ``extends`` into the full step sequence, parent first.

    Detects cycles (a malformed registry) with a ``ValueError`` rather than
    recursing forever.
    """
    chain: list[Path] = []
    seen: set[str] = set()
    current: Optional[Path] = path
    while current is not None:
        if current.id in seen:
            raise ValueError(f"cyclic cascade path extends at {current.id!r}")
        seen.add(current.id)
        chain.append(current)
        current = get_path(current.extends) if current.extends else None
    steps: list[PathStep] = []
    for entry in reversed(chain):
        steps.extend(entry.steps)
    return tuple(steps)


def blocking_steps(path: Path) -> tuple[PathStep, ...]:
    """Steps that gate the first token (i.e. not async judgements)."""
    return tuple(step for step in resolve_steps(path) if not step.async_)


def async_steps(path: Path) -> tuple[PathStep, ...]:
    """Steps that are fired and consumed later, never blocking the pipeline."""
    return tuple(step for step in resolve_steps(path) if step.async_)


def paths_for_device(device: Device | str) -> tuple[Path, ...]:
    value = device.value if isinstance(device, Device) else str(device)
    return tuple(p for p in PATHS if any(d.value == value for d in p.devices))


def validate_paths(paths: tuple[Path, ...] = PATHS) -> list[str]:
    """Structural checks over the path registry; empty list means valid."""
    problems: list[str] = []
    seen: set[str] = set()
    for path in paths:
        if path.id in seen:
            problems.append(f"duplicate path id {path.id!r}")
        seen.add(path.id)
        if not path.steps and not path.extends:
            problems.append(f"path {path.id!r} has no steps")
        if path.extends and path.extends not in {p.id for p in paths}:
            problems.append(f"path {path.id!r} extends unknown {path.extends!r}")
        for step in path.steps:
            try:
                get_role(step.role)
            except KeyError:
                problems.append(f"path {path.id!r} names unknown role {step.role!r}")
        for target in path.escalates_to:
            if target not in {p.id for p in paths}:
                problems.append(f"path {path.id!r} escalates to unknown {target!r}")
    for path in paths:
        if path.extends:
            try:
                resolve_steps(path)
            except ValueError as exc:
                problems.append(str(exc))
    return problems
