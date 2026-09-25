"""Stack lifecycle manager — apply / status / unload (HIVE-PLAN §B3).

Each tier is spawned as one ``ServerInstance`` through
``harness.models.LlamaServerManager.load``.  The launch config is expressed as
an ``EngineProfile.load_options`` dict (ADR-L4) so there is no parallel config
format; the few llama-server flags that ``load_options`` cannot carry
(``--split-mode layer --ts``, ``--spec-type draft-mtp``) travel as extra args /
env.  ``apply`` runs the T36 residency plan first and refuses an over-budget
stack before spawning anything (ADR-L5).

Frozen at T34 (ADR-L8).  T37 implements it; T38 (api) and T44 (wiring) consume
it.  ``status`` is the shape the Stack-tab residency strip renders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from harness.stack.schema import Stack, Tier


@dataclass
class TierRuntime:
    """One live tier after ``apply``: the role → ``ServerInstance`` mapping.

    ``to_dict`` is a row of ``GET /v1/stacks/status``::

        {"role", "key", "port", "ctx", "model", "backend", "resident",
         "vram_gb", "tok_s", "per_card": [...]}

    ``vram_gb`` / ``tok_s`` are live measurements (``None`` when unavailable);
    ``per_card`` is the planned residency from ``residency.plan_residency``.
    """

    role: str
    key: str
    port: int
    ctx: int
    model: str = ""
    backend: str = "vulkan"
    resident: bool = True
    vram_gb: Optional[float] = None
    tok_s: Optional[float] = None
    per_card: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError


def tier_load_options(tier: Tier) -> dict[str, Any]:
    """The tier's launch config as an ``EngineProfile.load_options`` dict.

    Maps ``ctx``→``context``, ``ngl``→``gpu_layers``, ``cache_k``/``cache_v``
    →``cache_type_k``/``cache_type_v``, ``mmproj``→``mmproj`` (ADR-L4).
    """
    raise NotImplementedError


def tier_env(tier: Tier) -> dict[str, str]:
    """Process env for the tier's ``load()`` (``pin`` → ``HIP_VISIBLE_DEVICES``)."""
    raise NotImplementedError


def tier_extra_args(tier: Tier) -> list[str]:
    """Flags the ``load_options`` seam cannot carry.

    Emits ``--split-mode layer --ts <ts>`` when ``tier.ts`` is set, and the
    speculative-decoding flags for ``tier.spec`` (``--spec-type draft-mtp``,
    ``--draft-max <n_max>``) when present.
    """
    raise NotImplementedError


class StackManager:
    """Owns the currently applied stack: one ``ServerInstance`` per tier.

    ``models_manager`` is the shared ``LlamaServerManager``.  ``apply``,
    ``status`` and ``unload`` are the frozen §B3 surface; the manager holds at
    most one applied stack (spawn discipline: unload before switching).
    """

    def __init__(
        self,
        models_manager: Any,
        *,
        stacks_root: Any = None,
        hardware: Any = None,
    ) -> None:
        raise NotImplementedError

    def apply(self, stack: Stack) -> dict[str, Any]:
        """Load every tier, face first.

        Returns ``{"ok": bool, "tiers": [{"role", "port", "key"}, ...]}``.
        On any failure the tiers already spawned are rolled back and the error
        is raised as ``RuntimeError`` with the offending role attached.
        """
        raise NotImplementedError

    def status(self) -> dict[str, Any]:
        """Per-tier residency, port, ctx, VRAM and tok/s.

        Returns ``{"ok", "stack", "tiers": [TierRuntime.to_dict(), ...],
        "warnings": [...]}``.  ``{}`` / empty ``tiers`` when nothing is applied
        (never raises for an unapplied stack).
        """
        raise NotImplementedError

    def unload(self, stack: Optional[Stack | str] = None) -> dict[str, Any]:
        """Stop every tier of the applied (or named) stack.

        Returns ``{"ok": bool, "unloaded": [role, ...]}``; unknown/unapplied
        names are a no-op with ``ok`` True.
        """
        raise NotImplementedError

    def instance_for(self, role: str) -> Any:
        """The ``ServerInstance`` for ``role``, or ``None``."""
        raise NotImplementedError
