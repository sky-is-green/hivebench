"""Stack lifecycle manager — apply / status / unload (HIVE-PLAN §B3).

Each locally launched tier is spawned as one ``ServerInstance`` through
``harness.models.LlamaServerManager.load``.  The launch config is expressed as
an ``EngineProfile.load_options`` dict (ADR-L4) so there is no parallel config
format; the few llama-server flags that ``load_options`` cannot carry
(``--split-mode layer --tensor-split``, ``--spec-type draft-mtp``) travel as extra args /
env.  ``apply`` runs the T36 residency plan first and refuses an over-budget
stack before spawning anything (ADR-L5).

A tier with an ``endpoint`` (Strata's ``serve/server.py``, or any
OpenAI-compatible server launched outside the harness) is **not spawned as a
llama-server**: ``apply`` health-checks it through
``harness.stack.endpoint.probe_endpoint`` and registers it.  With a ``launch``
block the manager also owns the lifecycle when needed — an endpoint that
answers is adopted, one that does not is started through
``harness.stack.strata.StrataLauncher`` (generated config, own port/log) and
stopped again on unload.  An adopted process is never killed, so ownership
stays with the engine's own run script whenever the human started it; the
cascade and the DSH runtime see the same generator contract (endpoint + caps +
residency) either way.

Frozen at T34 (ADR-L8).  T37 implements it; T38 (api) and T44 (wiring) consume
it.  ``status`` is the shape the Stack-tab residency strip renders.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from harness.models import launch_extra_args
from harness.stack.deepinfer import DeepInferLauncher
from harness.stack.endpoint import endpoint_base, endpoint_port, probe_endpoint
from harness.stack.residency import plan_residency
from harness.stack.schema import ROLE_FACE, Stack, Tier
from harness.stack.strata import StrataLauncher


@dataclass
class TierRuntime:
    """One live tier after ``apply``: the role → ``ServerInstance`` mapping.

    ``to_dict`` is a row of ``GET /v1/stacks/status``::

        {"role", "key", "port", "ctx", "model", "backend", "resident",
         "vram_gb", "tok_s", "base_url", "caps", "per_card": [...]}

    ``vram_gb`` / ``tok_s`` are live measurements (``None`` when unavailable);
    ``base_url`` is set for endpoint-backed tiers (the OpenAI base the stack
    file declares) and consumed by ``harness.stack_runtime``; ``caps`` carries
    the tier's declared measured serving facts; ``managed``/``pid`` say whether
    the harness started the endpoint process (and stops it on unload) or
    adopted one it does not own; ``per_card`` is the planned residency from
    ``residency.plan_residency``.
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
    base_url: Optional[str] = None
    caps: Optional[dict[str, Any]] = None
    managed: bool = False
    pid: Optional[int] = None
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
            "base_url": self.base_url,
            "caps": dict(self.caps) if self.caps else None,
            "managed": self.managed,
            "pid": self.pid,
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

    Always emits ``--jinja``: llama.cpp only applies the model's tool template
    under it, so without this a tier cannot call tools (LOCAL-STACKS §6 makes
    it mandatory for tool-driving tiers).  Emits ``--split-mode layer
    --tensor-split <ts>`` when ``tier.ts`` is set, and the speculative-decoding
    flags for ``tier.spec`` (``--spec-type draft-mtp``, ``--draft-max <n_max>``)
    when present.  ``tier.engine_args`` (schema-validated: only the curated
    engine knobs) is appended verbatim, last, so a stack can pin e.g.
    ``--device``, ``-np`` or ``--lazy-mode`` for a tier.
    """
    args: list[str] = ["--jinja"]
    tensor_split = _tier_field(tier, "ts", None)
    if tensor_split:
        args += ["--split-mode", "layer", "--tensor-split", str(tensor_split)]
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
    engine_args = _tier_field(tier, "engine_args", None)
    if engine_args:
        args += [str(arg) for arg in engine_args]
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

    A tier carrying ``endpoint`` is never spawned as a llama-server: apply
    health-checks it and registers it.  With a ``launch`` block the manager
    also owns the lifecycle when needed: an endpoint that answers is
    **adopted** (``unload`` leaves it running); one that does not is
    **started** through :class:`~harness.stack.strata.StrataLauncher` and
    stopped again on unload.
    """

    def __init__(
        self,
        models_manager: Any,
        *,
        stacks_root: Any = None,
        hardware: Any = None,
        endpoint_prober: Optional[Callable[[str], dict[str, Any]]] = None,
        strata_launcher: Optional[StrataLauncher] = None,
        deepinfer_launcher: Optional[DeepInferLauncher] = None,
    ) -> None:
        self.models_manager = models_manager
        self.stacks_root = stacks_root
        self.hardware = hardware
        self.endpoint_prober = endpoint_prober or probe_endpoint
        if strata_launcher is None:
            strata_launcher = StrataLauncher(
                log_dir=getattr(models_manager, "log_dir", None),
                prober=self.endpoint_prober,
            )
        if deepinfer_launcher is None:
            deepinfer_launcher = DeepInferLauncher(
                log_dir=getattr(models_manager, "log_dir", None),
                prober=self.endpoint_prober,
            )
        self.strata_launcher = strata_launcher
        self.deepinfer_launcher = deepinfer_launcher
        self._applied: Optional[Stack] = None
        self._plan: Any = None
        self._runtimes: dict[str, TierRuntime] = {}
        self._keys: dict[str, str] = {}
        self._launched: set[str] = set()
        #: Which launcher owns each launched role (the launch kind decides).
        self._launcher_of: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _launcher_for(self, launch: Any) -> Any:
        """The launcher that owns a ``launch`` block, by its ``kind``.

        ``strata`` and ``deepinfer`` are the two kinds this build knows; an
        unknown kind falls back to the Strata launcher, whose behaviour is
        the closest match (a generated config plus a managed process).
        """
        kind = ""
        if isinstance(launch, Mapping):
            kind = str(launch.get("kind") or "").strip().lower()
        if kind == "deepinfer":
            return self.deepinfer_launcher
        return self.strata_launcher

    @staticmethod
    def _ordered_tiers(stack: Stack) -> list[Tier]:
        """Stack tiers, face first, others in declared order."""
        tiers = list(stack.tiers)
        face = [t for t in tiers if getattr(t, "role", None) == ROLE_FACE]
        rest = [t for t in tiers if getattr(t, "role", None) != ROLE_FACE]
        return face + rest

    @staticmethod
    def _endpoint(tier: Any) -> str:
        """The tier's endpoint URL, or ``""`` when it is locally launched."""
        value = _tier_field(tier, "endpoint", None)
        return str(value).strip() if value else ""

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
        A tier with ``endpoint`` is health-checked and registered rather than
        spawned; on any failure the tiers already spawned are rolled back and
        the error is raised as ``RuntimeError`` with the offending role
        attached.
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
        launched: set[str] = set()
        # Managed launches must not be handed a port an endpoint tier already
        # owns in this stack: endpoint ports live outside the model manager.
        busy = self._busy_ports()
        busy.update(
            port for port in (endpoint_port(self._endpoint(t))
                              for t in stack.tiers) if port > 0
        )
        base = self._base_port()
        offset = 0
        tier: Any = None
        try:
            for tier in self._ordered_tiers(stack):
                endpoint = self._endpoint(tier)
                if endpoint:
                    launch = _tier_field(tier, "launch", None)
                    info: dict[str, Any] = {}
                    managed = False
                    try:
                        health = self.endpoint_prober(endpoint)
                    except RuntimeError:
                        if not isinstance(launch, Mapping):
                            raise
                        # The contract is down and the tier knows how to bring
                        # it up: start it (and own the process for unload).
                        launcher = self._launcher_for(launch)
                        info = launcher.start(
                            stack_name=stack.name, role=tier.role,
                            tier=tier, endpoint=endpoint,
                        )
                        managed = True
                        launched.add(tier.role)
                        self._launcher_of[tier.role] = launcher
                        health = info.get("health") or self.endpoint_prober(endpoint)
                    if not isinstance(health, Mapping):
                        health = {}
                    expect = _tier_field(tier, "expect_model", None)
                    reported_model = health.get("model")
                    if expect and reported_model and str(expect) != str(reported_model):
                        raise RuntimeError(
                            f"endpoint {endpoint} reports model {reported_model!r}, "
                            f"expected {expect!r} (stale route or wrong server)")
                    reported_ctx = health.get("max_context")
                    runtimes[tier.role] = TierRuntime(
                        role=tier.role,
                        key=f"{stack.name}-{tier.role}",
                        port=endpoint_port(endpoint),
                        ctx=int(reported_ctx or _tier_field(tier, "ctx", 8192)),
                        model=str(health.get("model")
                                  or Path(_tier_field(tier, "file", "")).stem),
                        backend=str(_tier_field(tier, "backend", "") or "endpoint"),
                        resident=bool(health.get("loaded", True)),
                        base_url=str(health.get("base_url")
                                     or endpoint.rstrip("/")),
                        caps=dict(_tier_field(tier, "caps", {}) or {}),
                        managed=managed,
                        pid=info.get("pid"),
                        per_card=[dict(card) for card in per_card],
                    )
                    continue
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
            for role in launched:
                launcher = self._launcher_of.get(role, self.strata_launcher)
                launcher.stop(role)
            role = getattr(tier, "role", "?")
            raise RuntimeError(
                f"stack '{stack.name}' failed on tier '{role}': {exc}"
            ) from exc

        self._applied = stack
        self._plan = plan
        self._runtimes = runtimes
        self._keys = keys
        self._launched = launched
        self._launcher_of = dict(self._launcher_of)
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
        """Stop every launched tier of the applied (or named) stack.

        Endpoint-backed tiers are always deregistered; one whose endpoint this
        manager started (its ``launch`` block) is stopped, while an adopted
        endpoint is left running for its owner.  Returns
        ``{"ok": bool, "unloaded": [role, ...]}``; unknown/unapplied names are
        a no-op with ``ok`` True.
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
            key = self._keys.get(role)
            if key is not None:
                self._safe_unload(key)
            elif role in self._launched:
                launcher = self._launcher_of.get(role, self.strata_launcher)
                launcher.stop(role)
            unloaded.append(role)
        self._applied = None
        self._plan = None
        self._runtimes = {}
        self._keys = {}
        self._launched = set()
        self._launcher_of = {}
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
