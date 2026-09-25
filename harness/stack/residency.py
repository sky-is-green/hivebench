"""Per-card residency plan for a stack (HIVE-PLAN §B3 ``validate``).

Turns a :class:`~harness.stack.schema.Stack` into the §B3 payload::

    {"ok": bool,
     "per_card": [{"card", "weights", "kv", "total", "budget"}, ...],
     "warnings": [...]}

Weights and KV come from the GGUF + the attention-aware estimator landed by
T33 (``harness.models.attention_kv_estimate`` and ``preflight_memory``) — never
from a hardcoded layer count (ADR-L5).  This module plans only; it spawns
nothing, so an over-budget stack is refused *before* apply.

Frozen at T34 (ADR-L8).  T36 implements it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from harness.stack.schema import Stack


@dataclass
class CardPlan:
    """Residency for one GPU card, in GiB.

    ``total = weights + kv``; ``budget`` is the card's usable VRAM after the
    desktop cost and compute buffers (LOCAL-STACKS.md §3).  A card is
    over-budget when ``total > budget``.
    """

    card: int
    weights: float
    kv: float
    total: float
    budget: float

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError


@dataclass
class ResidencyPlan:
    """The §B3 ``validate`` result.

    ``ok`` is False when any card is over budget or a tier cannot be resolved.
    ``warnings`` carries non-fatal notes (offload, thin GGUF metadata,
    unknown display card, ...); ``per_card`` is always populated when the
    model metadata resolved.
    """

    ok: bool
    per_card: list[CardPlan] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError


def plan_residency(
    stack: Stack,
    *,
    models_manager: Any = None,
    hardware: Optional[Callable[[], dict[str, Any]]] = None,
) -> ResidencyPlan:
    """Build the per-card plan for ``stack``.

    ``models_manager`` is a ``LlamaServerManager``-like object used to resolve
    each ``tier.file`` to a path and read its GGUF metadata (``resolve_model``,
    ``models_dir``, ``list_local``); ``hardware`` is a zero-arg callable
    returning the ``_hardware_summary()`` shape.  Both default to the real
    implementations when ``None``.
    """
    raise NotImplementedError
