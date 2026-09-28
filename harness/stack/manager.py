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

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from harness.models import launch_extra_args
from harness.stack.residency import plan_residency
from harness.stack.schema import ROLE_FACE, Stack, Tier


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
        return {
            "role": self.role,
            "key": self.key,
            "port": self.port,
            "ctx": self.ctx,
            "model": self.model,
            "backend": self.backend,
            "resident": self.resident,
            "vram_gb": self.vram_gb,
            "tok_s": self.tok_s,
            "per_card": list(self.per_card),
        }


def _tier_field(tier: Any, name: str, default: Any = None) -> Any:
    """Read one tier field, accepting a ``Tier`` or its §B3 mapping form.

    ``T43``'s ``stack_summary`` passes the face tier as a plain dict (its
    ``face_tier`` returns ``dict(tier)``), so the load-options helper must not
    assume the dataclass.  A field explicitly set to ``None`` falls back to
    ``default``.
    """
    if isinstance(tier, Mapping):
        value = tier.get(name, default)
    else:
        value = getattr(tier, name, default)
    return default if value is None else value


def tier_load_options(tier: Tier) -> dict[str, Any]:
    """The tier's launch config as an ``EngineProfile.load_options`` dict.

    Maps ``ctx``→``context``, ``ngl``→``gpu_layers``, ``cache_k``/``cache_v``
    →``cache_type_k``/``cache_type_v``, ``mmproj``→``mmproj`` (ADR-L4).
    """
    options: dict[str, Any] = {
        "context": int(_tier_field(tier, "ctx", 8192)),
        "gpu_layers": int(_tier_field(tier, "ngl", 99)),
    }
    for source, target in (("cache_k", "cache_type_k"),
                           ("cache_v", "cache_type_v")):
        value = _tier_field(tier, source, None)
        if value:
            options[target] = str(value)
    mmproj = _tier_field(tier, "mmproj", None)
    if mmproj:
        options["mmproj"] = str(mmproj)
    return options


def tier_env(tier: Tier) -> dict[str, str]:
    """Process env for the tier's ``load()`` (``pin`` → ``HIP_VISIBLE_DEVICES``)."""
    pin = _tier_field(tier, "pin", None)
    if not pin:
        return {}
    # ``pin`` is a full assignment in the stack file ("HIP_VISIBLE_DEVICES=0,1");
    # a bare card list ("0,1") is also accepted and expands to the variable name.
    text = str(pin).strip()
    if "=" in text:
        name, _, value = text.partition("=")
        if name.strip():
            return {name.strip(): value.strip()}
    return {"HIP_VISIBLE_DEVICES": text}


def tier_extra_args(tier: Tier) -> list[str]:
    """Flags the ``load_options`` seam cannot carry.

    Emits ``--split-mode layer --ts <ts>`` when ``tier.ts`` is set, and the
    speculative-decoding flags for ``tier.spec`` (``--spec-type draft-mtp``,
    ``--draft-max <n_max>``) when present.
    """
    args: list[str] = []
    tensor_split = _tier_field(tier, "ts", None)
    if tensor_split:
        args += ["--split-mode", "layer", "--ts", str(tensor_split)]
    spec = _tier_field(tier, "spec", None)
    if isinstance(spec, Mapping):
        spec_type = spec.get("type")
        if spec_type:
            args += ["--spec-type", str(spec_type)]
        n_max = spec.get("n_max")
        if n_max is not None:
            args += ["--draft-max", str(n_max)]
    elif spec:
        args += ["--spec-type", str(spec)]
    return args


def _card_dict(card: Any) -> dict[str, Any]:
    """A residency card as a plain dict (``CardPlan`` or an already-dict row)."""
    if isinstance(card, Mapping):
        return dict(card)
    to_dict = getattr(card, "to_dict", None)
    if callable(to_dict):
        return dict(to_dict())
    return dict(vars(card))


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
        self.models_manager = models_manager
        self.stacks_root = stacks_root
        self.hardware = hardware
        self._applied: Optional[Stack] = None
        self._plan: Any = None
        self._runtimes: dict[str, TierRuntime] = {}
        self._keys: dict[str, str] = {}

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    @staticmethod
    def _ordered_tiers(stack: Stack) -> list[Tier]:
        """Stack tiers, face first, others in declared order."""
        tiers = list(stack.tiers)
        face = [t for t in tiers if getattr(t, "role", None) == ROLE_FACE]
        rest = [t for t in tiers if getattr(t, "role", None) != ROLE_FACE]
        return face + rest

    def _ordered_roles(self) -> list[str]:
        order = [ROLE_FACE] + [role for role in self._runtimes
                               if role != ROLE_FACE]
        return [role for role in order if role in self._runtimes]

    def _base_port(self) -> int:
        try:
            return int(getattr(self.models_manager, "port", 1234))
        except (TypeError, ValueError):
            return 1234

    def _busy_ports(self) -> set[int]:
        """Ports already held by the shared manager (cheap, no probing)."""
        busy: set[int] = set()
        instances = getattr(self.models_manager, "_instances", None)
        if isinstance(instances, dict):
            for instance in instances.values():
                port = getattr(instance, "port", None)
                if port is None and isinstance(instance, Mapping):
                    port = instance.get("port")
                if port is not None:
                    busy.add(int(port))
        return busy

    def _safe_unload(self, key: str) -> None:
        try:
            self.models_manager.unload(key)
        except Exception:  # noqa: BLE001 - already gone is success for unload
            pass

    # ------------------------------------------------------------------
    # §B3 surface
    # ------------------------------------------------------------------
    def apply(self, stack: Stack) -> dict[str, Any]:
        """Load every tier, face first.

        Returns ``{"ok": bool, "tiers": [{"role", "port", "key"}, ...]}``.
        On any failure the tiers already spawned are rolled back and the error
        is raised as ``RuntimeError`` with the offending role attached.
        """
        plan = plan_residency(stack, models_manager=self.models_manager,
                              hardware=self.hardware)
        if not plan.ok:
            detail = "; ".join(plan.warnings) or "over budget"
            raise RuntimeError(f"stack '{stack.name}' refused: {detail}")

        # Spawn discipline: the manager holds at most one applied stack.
        if self._applied is not None:
            self.unload()

        # ``per_card`` has no per-role breakdown in the §B3 plan; every tier
        # row carries the whole-stack card residency (the only planned figure
        # the estimator produces).
        per_card = [_card_dict(card) for card in plan.per_card]
        runtimes: dict[str, TierRuntime] = {}
        keys: dict[str, str] = {}
        busy = self._busy_ports()
        base = self._base_port()
        offset = 0
        tier: Any = None
        try:
            for tier in self._ordered_tiers(stack):
                while base + offset in busy:
                    offset += 1
                port = base + offset
                offset += 1
                options = tier_load_options(tier)
                info = self.models_manager.load(
                    model=tier.file,
                    hf_repo=tier.repo,
                    hf_file=tier.file,
                    key=f"{stack.name}-{tier.role}",
                    port=port,
                    ctx_size=int(options.get("context", tier.ctx)),
                    ngl=options.get("gpu_layers", tier.ngl),
                    extra_args=launch_extra_args(options)
                    + tier_extra_args(tier),
                    env=tier_env(tier) or None,
                    backend=tier.backend or None,
                )
                key = str(info.get("key") or f"{stack.name}-{tier.role}")
                runtimes[tier.role] = TierRuntime(
                    role=tier.role,
                    key=key,
                    port=int(info.get("port") or port),
                    ctx=int(tier.ctx),
                    model=str(info.get("model") or Path(tier.file).stem),
                    backend=str(tier.backend or ""),
                    resident=True,
                    per_card=[dict(card) for card in per_card],
                )
                keys[tier.role] = key
                busy.add(port)
        except Exception as exc:  # noqa: BLE001 - roll back then re-raise
            for key in keys.values():
                self._safe_unload(key)
            role = getattr(tier, "role", "?")
            raise RuntimeError(
                f"stack '{stack.name}' failed on tier '{role}': {exc}"
            ) from exc

        self._applied = stack
        self._plan = plan
        self._runtimes = runtimes
        self._keys = keys
        ordered = [ROLE_FACE] + [role for role in runtimes if role != ROLE_FACE]
        return {
            "ok": True,
            "tiers": [
                {"role": runtimes[role].role, "port": runtimes[role].port,
                 "key": runtimes[role].key}
                for role in ordered if role in runtimes
            ],
        }

    def status(self) -> dict[str, Any]:
        """Per-tier residency, port, ctx, VRAM and tok/s.

        Returns ``{"ok", "stack", "tiers": [TierRuntime.to_dict(), ...],
        "warnings": [...]}``.  ``{}`` / empty ``tiers`` when nothing is applied
        (never raises for an unapplied stack).
        """
        if self._applied is None or not self._runtimes:
            return {"ok": True, "stack": None, "tiers": [], "warnings": []}
        warnings = list(getattr(self._plan, "warnings", []) or [])
        return {
            "ok": True,
            "stack": self._applied.name,
            "tiers": [self._runtimes[role].to_dict()
                      for role in self._ordered_roles()],
            "warnings": warnings,
        }

    def unload(self, stack: Optional[Stack | str] = None) -> dict[str, Any]:
        """Stop every tier of the applied (or named) stack.

        Returns ``{"ok": bool, "unloaded": [role, ...]}``; unknown/unapplied
        names are a no-op with ``ok`` True.
        """
        if self._applied is None:
            return {"ok": True, "unloaded": []}
        if isinstance(stack, str):
            target: Optional[str] = stack
        else:
            target = getattr(stack, "name", None)
        if target is not None and target != self._applied.name:
            return {"ok": True, "unloaded": []}

        unloaded: list[str] = []
        for role in self._ordered_roles():
            self._safe_unload(self._keys[role])
            unloaded.append(role)
        self._applied = None
        self._plan = None
        self._runtimes = {}
        self._keys = {}
        return {"ok": True, "unloaded": unloaded}

    def instance_for(self, role: str) -> Any:
        """The ``ServerInstance`` for ``role``, or ``None``."""
        key = self._keys.get(role)
        if key is None:
            return None
        instances = getattr(self.models_manager, "_instances", None)
        if isinstance(instances, dict):
            return instances.get(key)
        return None
