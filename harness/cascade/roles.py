"""Cascade role taxonomy — the functional roles of the pilot brief.

The cascade splits the system into *functional roles* (brief §"Roles, not
models"): each role declares its input/output contract, device tier, latency
budget, context cap, KV policy and candidate ids.  Roles carry **no model
choice** — :mod:`harness.cascade.registry` maps candidates onto roles — so
adding a role or a candidate is a local data change and optimization is
per-role selection plus scheduling.

The ids here are the brief's taxonomy verbatim: A1–A2 encoders, B1–B4
retrieval, C1–C4 routing/decisions, D1–D3 verification, E1–E5 generation,
F1–F2 memory, G1 scheduler, H1 reasoning-budget control.

Latency budgets are the brief's planning table (per call unless
``latency_unit == "token"``); they are planning inputs, not measurements — the
per-role oracle (:mod:`harness.cascade.oracle`) replaces them with measured
numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class Device(str, Enum):
    """Compute tiers of the box (brief §"Compute inventory").

    The iGPU row is a real tier now (gfx1036, ROCm-visible); it shares system
    RAM, so it is a separate resource from the two dGPUs' VRAM.
    """

    CPU = "cpu"
    IGPU = "igpu"
    DGPU0 = "dgpu0"
    DGPU1 = "dgpu1"


class Kind(str, Enum):
    """What a role does — coarse enough to group candidates, fine enough that
    a candidate can declare which roles it covers (brief §"Role
    consolidation")."""

    EMBED = "embed"
    VISION = "vision"
    RETRIEVE = "retrieve"
    RERANK = "rerank"
    LATE_INTERACTION = "late_interaction"
    CURATE = "curate"
    ROUTE = "route"
    GATE = "gate"
    NEXT_STEP = "next_step"
    SCORE = "score"
    CONSISTENCY = "consistency"
    ACCEPT = "accept"
    HALLUCINATION_CHECK = "hallucination_check"
    GENERATE = "generate"
    TRANSFORM = "transform"
    MEMORY_STORE = "memory_store"
    MEMORY_COMPRESS = "memory_compress"
    SCHEDULE = "schedule"
    BUDGET = "budget"


@dataclass(frozen=True)
class Role:
    """One capability slot of the cascade.

    ``devices`` is the placement set in preference order (the first is the
    brief's tier).  ``context_cap`` of 0 means the role has no growing cache;
    ``kv`` is the KV policy for the roles that do (``"none"`` otherwise).
    ``candidates`` are registry ids, not repositories.
    """

    id: str
    name: str
    kind: Kind
    input_kind: str
    output_kind: str
    devices: tuple[Device, ...]
    latency_ms: tuple[float, float]
    latency_unit: str = "call"
    context_cap: int = 0
    kv: str = "none"
    candidates: tuple[str, ...] = ()
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind.value,
            "input": self.input_kind,
            "output": self.output_kind,
            "devices": [d.value for d in self.devices],
            "latency_ms": list(self.latency_ms),
            "latency_unit": self.latency_unit,
            "context_cap": self.context_cap,
            "kv": self.kv,
            "candidates": list(self.candidates),
            "notes": self.notes,
        }


#: KV policy for generator roles — the brief's floor (asymmetric K/V).
GEN_KV = "k-q5_0/v-q4_1"


def _role(
    rid: str,
    name: str,
    kind: Kind,
    input_kind: str,
    output_kind: str,
    devices: tuple[Device, ...],
    latency_ms: tuple[float, float],
    *,
    latency_unit: str = "call",
    context_cap: int = 0,
    kv: str = "none",
    candidates: tuple[str, ...] = (),
    notes: str = "",
) -> Role:
    return Role(
        id=rid,
        name=name,
        kind=kind,
        input_kind=input_kind,
        output_kind=output_kind,
        devices=devices,
        latency_ms=latency_ms,
        latency_unit=latency_unit,
        context_cap=context_cap,
        kv=kv,
        candidates=candidates,
        notes=notes,
    )


#: The pilot taxonomy (brief §"Roles, not models").  Order is presentation
#: order; lookups go through :func:`get_role`.
ROLES: tuple[Role, ...] = (
    _role(
        "A1", "text embedding", Kind.EMBED, "text", "vector",
        (Device.CPU, Device.IGPU), (1.0, 10.0),
        context_cap=512,
        candidates=("minilm-l6", "bge-small-en", "bge-m3"),
        notes="bi-encoder; also serves B1/F1 with an index",
    ),
    _role(
        "A2", "vision encoding", Kind.VISION, "image", "vector/region",
        (Device.IGPU, Device.DGPU0), (20.0, 120.0),
        candidates=("florence-2-base", "smolvlm2-256m", "altclip"),
    ),
    _role(
        "B1", "retrieval", Kind.RETRIEVE, "query", "chunks",
        (Device.CPU,), (5.0, 40.0),
        context_cap=512,
        candidates=("minilm-l6", "bge-small-en", "bge-m3"),
        notes="A1 + vector index",
    ),
    _role(
        "B2", "reranking", Kind.RERANK, "query+chunk", "score",
        (Device.CPU, Device.IGPU), (5.0, 20.0),
        context_cap=512,
        candidates=("bge-reranker-v2-m3", "jina-reranker-v3", "mxbai-rerank"),
    ),
    _role(
        "B3", "late interaction", Kind.LATE_INTERACTION, "query+chunk", "score",
        (Device.CPU,), (5.0, 20.0),
        context_cap=512,
        candidates=("colbert",),
    ),
    _role(
        "B4", "curation / dedup", Kind.CURATE, "chunks", "chunks",
        (Device.CPU,), (2.0, 10.0),
        context_cap=512,
        candidates=("minilm-l6", "bge-small-en"),
        notes="heuristics + A1",
    ),
    _role(
        "C1", "intent / task routing", Kind.ROUTE, "prompt", "bucket",
        (Device.IGPU, Device.DGPU0), (20.0, 200.0),
        context_cap=4096,
        candidates=("qwen3.5-0.8b", "qwen3.5-2b", "intern-decision-4b"),
    ),
    _role(
        "C2", "gating (answer/escalate/refuse)", Kind.GATE, "prompt+state", "decision",
        (Device.IGPU,), (10.0, 100.0),
        context_cap=4096,
        candidates=("qwen3.5-0.8b", "tiny-jev-1.7b", "intern-decision-4b"),
        notes="calibrated probabilities",
    ),
    _role(
        "C3", "next-step / tool choice", Kind.NEXT_STEP, "state", "action",
        (Device.DGPU0,), (20.0, 50.0),
        context_cap=8192,
        candidates=("intern-decision-4b", "qwen3.5-4b"),
    ),
    _role(
        "C4", "scoring / ranking", Kind.SCORE, "pair", "score",
        (Device.CPU, Device.IGPU), (10.0, 50.0),
        context_cap=512,
        candidates=("intern-decision-4b", "bge-reranker-v2-m3", "jina-reranker-v3"),
    ),
    _role(
        "D1", "consistency / truthfulness", Kind.CONSISTENCY, "answer+evidence", "decision",
        (Device.IGPU,), (10.0, 50.0),
        context_cap=4096,
        candidates=("bge-reranker-v2-m3", "intern-decision-4b"),
    ),
    _role(
        "D2", "accept / reject", Kind.ACCEPT, "answer", "decision",
        (Device.IGPU,), (10.0, 50.0),
        context_cap=4096,
        candidates=("tiny-jev-1.7b", "intern-decision-4b"),
    ),
    _role(
        "D3", "hallucination check", Kind.HALLUCINATION_CHECK, "answer", "decision",
        (Device.DGPU0,), (20.0, 100.0),
        context_cap=4096,
        candidates=("intern-decision-4b", "qwen3.5-2b"),
    ),
    _role(
        "E1", "fast chat", Kind.GENERATE, "prompt", "text",
        (Device.DGPU0,), (3.0, 8.0),
        latency_unit="token", context_cap=32768, kv=GEN_KV,
        candidates=("qwen3.5-2b",),
    ),
    _role(
        "E2", "general generation", Kind.GENERATE, "prompt", "text",
        (Device.DGPU0,), (8.0, 20.0),
        latency_unit="token", context_cap=32768, kv=GEN_KV,
        candidates=("qwen3.5-4b",),
    ),
    _role(
        "E3", "coding", Kind.GENERATE, "prompt", "text",
        (Device.DGPU1,), (10.0, 20.0),
        latency_unit="token", context_cap=131072, kv=GEN_KV,
        candidates=("ornith-1.5-9b", "qwen3.5-4b"),
    ),
    _role(
        "E4", "agentic / reasoning", Kind.GENERATE, "prompt", "text",
        (Device.DGPU1,), (15.0, 30.0),
        latency_unit="token", context_cap=131072, kv=GEN_KV,
        candidates=("qwen3.5-27b", "flash-next", "scion-35b-a3b"),
    ),
    _role(
        "E5", "transform (summarize/rewrite)", Kind.TRANSFORM, "text", "text",
        (Device.DGPU0,), (3.0, 8.0),
        latency_unit="token", context_cap=32768, kv=GEN_KV,
        candidates=("qwen3.5-2b",),
    ),
    _role(
        "F1", "memory store / dedup", Kind.MEMORY_STORE, "turns", "store",
        (Device.CPU,), (2.0, 20.0),
        context_cap=512,
        candidates=("minilm-l6", "bge-small-en"),
        notes="A1 + index",
    ),
    _role(
        "F2", "memory compression", Kind.MEMORY_COMPRESS, "turns", "summary",
        (Device.DGPU0,), (3.0, 8.0),
        latency_unit="token", context_cap=32768, kv=GEN_KV,
        candidates=("qwen3.5-2b", "qwen3.5-4b"),
    ),
    _role(
        "G1", "scheduler / router", Kind.SCHEDULE, "request", "path",
        (Device.CPU,), (1.0, 5.0),
        candidates=(),
        notes="deterministic harness policy fed by C1/C2 signals",
    ),
    _role(
        "H1", "reasoning budget control", Kind.BUDGET, "partial CoT + question", "continue/stop",
        (Device.IGPU, Device.CPU), (10.0, 50.0),
        context_cap=4096,
        candidates=("tiny-jev-1.7b", "intern-decision-4b", "qwen3.5-0.8b"),
    ),
)

_ROLE_BY_ID: dict[str, Role] = {role.id: role for role in ROLES}


def get_role(role_id: str) -> Role:
    """Return the role with ``role_id`` or raise ``KeyError``."""
    try:
        return _ROLE_BY_ID[role_id]
    except KeyError:
        known = ", ".join(_ROLE_BY_ID)
        raise KeyError(f"unknown cascade role {role_id!r} (known: {known})") from None


def role_ids() -> tuple[str, ...]:
    """All role ids in taxonomy order."""
    return tuple(_ROLE_BY_ID)


def roles_for_kind(kind: Kind | str) -> tuple[Role, ...]:
    """Roles of one kind (accepts a :class:`Kind` or its string value)."""
    value = kind.value if isinstance(kind, Kind) else str(kind)
    return tuple(role for role in ROLES if role.kind.value == value)


def roles_on(device: Device | str) -> tuple[Role, ...]:
    """Roles that can be placed on ``device``."""
    value = device.value if isinstance(device, Device) else str(device)
    return tuple(role for role in ROLES if any(d.value == value for d in role.devices))


def validate_roles(roles: tuple[Role, ...] = ROLES) -> list[str]:
    """Structural checks over a taxonomy; returns a list of problems.

    Empty means valid.  Used by the tests and by any future loader so a
    taxonomy edit cannot silently ship duplicate ids or empty contracts.
    """
    problems: list[str] = []
    seen: set[str] = set()
    for role in roles:
        if not role.id:
            problems.append("role with empty id")
        if role.id in seen:
            problems.append(f"duplicate role id {role.id!r}")
        seen.add(role.id)
        if not role.devices:
            problems.append(f"role {role.id!r} has no device tier")
        if len(role.latency_ms) != 2 or role.latency_ms[0] > role.latency_ms[1]:
            problems.append(f"role {role.id!r} has a malformed latency budget")
        if role.latency_unit not in ("call", "token"):
            problems.append(f"role {role.id!r} has latency_unit {role.latency_unit!r}")
    return problems
