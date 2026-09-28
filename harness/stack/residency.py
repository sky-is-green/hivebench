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

Where the numbers come from
---------------------------

Per card (``LOCAL-STACKS.md`` §3)::

    budget(card) = vram_total - desktop_used - sum(compute_buffer(tier))
    total(card)  = weights(card) + kv(card)
    over budget  when total > budget

* ``desktop_used`` is charged only to the card that drives a display — the
  compositor's framebuffers are real VRAM, worth ~1.6 GiB on the reference box.
  ``harness.hardware.linux_amd_devices()`` flags the card through sysfs
  (``display``), never through GLX, and reports the live ``used_gb``; that is
  the figure charged.  A display-flagged card with no ``used_gb`` falls back to
  :data:`DESKTOP_COST_FALLBACK_GIB` (the measured ~1.6 GiB).
* ``compute_buffer`` is per resident tier and sized by that tier's weights
  (§3: ~1.5 GiB for a 27B, ~0.5 GiB for a small model).
* ``weights`` is the GGUF file size on disk, plus the tier's ``mmproj`` and,
  when a draft head is configured, the draft-head reservation (§6).
* ``kv`` is ``KVEstimate.bytes_at(ctx, cache_type)`` from the T33 estimator,
  priced with the *wider* of ``cache_k``/``cache_v`` so a mixed pair cannot
  under-reserve.
* A tier's weights and KV are split across its pinned cards in the ``ts``
  ratio (llama.cpp splits the cache with the layers); KV follows the weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from harness.models import attention_kv_estimate
from harness.stack.schema import Stack

#: Bytes per GiB — every figure in the plan is reported in GiB.
GIB = 1024.0 ** 3

#: Desktop cost charged to a display card that reports no ``used_gb``
#: (LOCAL-STACKS.md §3: "the desktop card loses ~1.6 GiB").
DESKTOP_COST_FALLBACK_GIB = 1.6

#: Weights at or above which a tier gets the 27B-class compute buffer (§3).
BIG_TIER_GIB = 10.0
COMPUTE_BUFFER_BIG_GIB = 1.5
COMPUTE_BUFFER_SMALL_GIB = 0.5

#: Draft head reservation when ``tier.spec`` asks for speculative decoding
#: (§6: "~1.2 GiB for ~1.3–1.6× TG").  The stack file names no draft file, so
#: this is a documented reservation, not a stat of a resolved GGUF.
DRAFT_HEAD_GIB = 1.2

#: llama.cpp's default window, used when a tier declares a non-positive ctx.
DEFAULT_CTX = 8192

#: ``spec.type`` values that mean "a draft head is loaded alongside the tier".
_DRAFT_SPEC_TYPES = ("draft", "mtp")


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
        return {
            "card": self.card,
            "weights": round(float(self.weights), 3),
            "kv": round(float(self.kv), 3),
            "total": round(float(self.total), 3),
            "budget": round(float(self.budget), 3),
        }


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
        return {
            "ok": bool(self.ok),
            "per_card": [card.to_dict() for card in self.per_card],
            "warnings": list(self.warnings),
        }


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
    warnings: list[str] = []
    devices = _visible_devices(_hardware_summary(hardware), warnings)
    if not devices:
        return ResidencyPlan(ok=False, per_card=[], warnings=warnings + [
            "no visible GPU devices: cannot plan residency"])

    by_index = {card["index"]: card for card in devices}
    weights: dict[int, float] = {card["index"]: 0.0 for card in devices}
    kv: dict[int, float] = {card["index"]: 0.0 for card in devices}
    resident: dict[int, list[float]] = {card["index"]: [] for card in devices}
    manager = _model_manager(models_manager, warnings)
    resolved = True

    for tier in stack.tiers:
        resolved = _charge_tier(
            tier,
            manager=manager,
            devices=devices,
            by_index=by_index,
            weights=weights,
            kv=kv,
            resident=resident,
            warnings=warnings,
        ) and resolved

    per_card: list[CardPlan] = []
    for card in devices:
        index = card["index"]
        budget = _budget(card, resident[index], warnings)
        per_card.append(CardPlan(
            card=index,
            weights=weights[index],
            kv=kv[index],
            total=weights[index] + kv[index],
            budget=budget,
        ))
    per_card.sort(key=lambda plan: plan.card)
    over = [plan for plan in per_card if plan.total > plan.budget]
    for plan in over:
        warnings.append(
            f"card {plan.card} is over budget: {plan.total:.1f} GiB needed, "
            f"{plan.budget:.1f} GiB usable")

    return ResidencyPlan(ok=resolved and not over, per_card=per_card,
                         warnings=warnings)


# ---------------------------------------------------------------------------
# hardware
# ---------------------------------------------------------------------------
def _hardware_summary(
    hardware: Optional[Callable[[], dict[str, Any]]],
) -> dict[str, Any]:
    """The ``_hardware_summary()`` dict, from the caller or the real probe."""
    if hardware is not None:
        def probe() -> dict[str, Any]:
            return hardware()

    else:
        def probe() -> dict[str, Any]:
            from harness.app import _hardware_summary as real_summary

            return real_summary()

    try:
        result = probe()
    except Exception as exc:  # noqa: BLE001 - a plan must never 500 on probing
        return {"__error__": str(exc)}
    return result if isinstance(result, dict) else {}


def _model_manager(manager: Any, warnings: list[str]) -> Any:
    """The caller's model manager, or the real ``LlamaServerManager``.

    A manager that cannot be constructed is not fatal: every tier then fails to
    resolve, which the plan already reports as ``ok: False`` with a warning.
    """
    if manager is not None:
        return manager
    try:
        from harness.models import LlamaServerManager

        return LlamaServerManager()
    except Exception as exc:  # noqa: BLE001 - degrade to 'unresolvable'
        warnings.append(f"model library unavailable: {exc}")
        return None


def _visible_devices(hw: dict[str, Any],
                     warnings: list[str]) -> list[dict[str, Any]]:
    """Cards the plan may place a tier on, in device order.

    Honours the ``visible`` flag ``_hardware_summary()`` already computes from
    ``HIP_VISIBLE_DEVICES``/``CUDA_VISIBLE_DEVICES``; a card with no usable
    ``memory_gb`` is skipped rather than planned at zero.
    """
    if hw.get("__error__"):
        warnings.append(f"hardware probe failed: {hw['__error__']}")
    out: list[dict[str, Any]] = []
    for position, device in enumerate(hw.get("devices") or []):
        if not isinstance(device, dict):
            continue
        if not device.get("visible", True):
            continue
        total = _device_total_gb(device)
        if total <= 0:
            continue
        card = dict(device)
        card["index"] = int(device.get("index", position))
        card["_total_gb"] = total
        out.append(card)
    return out


def _device_total_gb(device: dict[str, Any]) -> float:
    """Total VRAM of one device, derived from free+used when total is absent."""
    for key in ("memory_gb", "total_gb", "vram_gb"):
        value = device.get(key)
        if value:
            return float(value)
    free, used = device.get("free_gb"), device.get("used_gb")
    if free is not None or used is not None:
        return float(free or 0.0) + float(used or 0.0)
    return 0.0


def _desktop_cost_gb(device: dict[str, Any], warnings: list[str]) -> float:
    """VRAM the display server already holds on ``device`` (0 when headless).

    The card that drives a display has already lost its framebuffers, so that
    memory is unavailable to a tier no matter how much of ``memory_gb`` the
    summary still lists as free (LOCAL-STACKS.md §3).
    """
    if not device.get("display"):
        return 0.0
    used = device.get("used_gb")
    if used:
        cost = float(used)
    else:
        cost = DESKTOP_COST_FALLBACK_GIB
        warnings.append(
            f"card {device['index']} drives a display but reports no used_gb; "
            f"charging the measured {DESKTOP_COST_FALLBACK_GIB} GiB desktop cost")
    warnings.append(
        f"card {device['index']} drives a display: {cost:.1f} GiB desktop cost "
        f"excluded from its budget")
    return cost


def _compute_buffer_gb(tier_weights: float) -> float:
    """Per-process compute buffer for a resident tier (§3)."""
    return (COMPUTE_BUFFER_BIG_GIB if tier_weights >= BIG_TIER_GIB
            else COMPUTE_BUFFER_SMALL_GIB)


def _budget(card: dict[str, Any], tier_weights: list[float],
            warnings: list[str]) -> float:
    """Usable GiB on one card: total − desktop cost − compute buffers."""
    buffers = sum(_compute_buffer_gb(w) for w in tier_weights)
    return max(0.0, card["_total_gb"] - _desktop_cost_gb(card, warnings) - buffers)


# ---------------------------------------------------------------------------
# tier -> cards
# ---------------------------------------------------------------------------
def _charge_tier(tier, *, manager, devices, by_index, weights, kv, resident,
                 warnings) -> bool:
    """Charge one tier's weights and KV onto its cards; False when unresolvable."""
    path = _resolve(tier, manager, warnings)
    if path is None:
        return False
    size_gib = _file_size_gib(path)
    if size_gib is None:
        warnings.append(f"tier {tier.role}: cannot size {path}")
        return False

    estimate = _kv_estimate(tier, path, manager, warnings)
    ctx = _ctx(tier, warnings)
    if estimate is not None:
        kv_gib = _kv_gib(tier, estimate, ctx)
    else:
        kv_gib = 0.0
        warnings.append(
            f"tier {tier.role}: {Path(path).name} has no readable KV metadata "
            f"(no block_count / kv head count); KV is charged as 0 and only "
            f"the weights are budgeted")

    tier_weights = size_gib + _sidecar_weights(tier, manager, warnings)
    if _uses_draft_head(tier):
        tier_weights += DRAFT_HEAD_GIB
        warnings.append(
            f"tier {tier.role}: speculative decoding reserves "
            f"{DRAFT_HEAD_GIB} GiB of draft head (LOCAL-STACKS.md §6)")

    for card, share in _split(tier, devices, by_index, warnings).items():
        weights[card] = weights.get(card, 0.0) + tier_weights * share
        kv[card] = kv.get(card, 0.0) + kv_gib * share
        resident.setdefault(card, []).append(tier_weights)
    return True


def _split(tier, devices, by_index, warnings) -> dict[int, float]:
    """Card -> share of this tier's weights (and KV), per pin + tensor split.

    One pinned card takes everything.  Several cards split in the ``ts`` ratio
    llama.cpp was launched with; KV follows the weights, because a tensor split
    also splits the cache across the layers.  Several cards without a ``ts``
    that names each of them is unplannable, so the whole tier is charged to the
    first pinned card with a warning rather than spread on a guess.
    """
    cards = _pinned_cards(tier, devices, by_index, warnings)
    if not cards:
        cards = [device["index"] for device in devices]
    if len(cards) == 1:
        return {cards[0]: 1.0}

    shares = _tensor_split(tier, len(cards))
    if shares is None:
        warnings.append(
            f"tier {tier.role}: spans cards {cards} without a -ts that names "
            f"each of them ({tier.ts!r}); charging the whole tier to card "
            f"{cards[0]}")
        return {cards[0]: 1.0}
    return dict(zip(cards, shares))


def _pinned_cards(tier, devices, by_index, warnings) -> list[int]:
    """Card indices named by ``tier.pin`` (``HIP_VISIBLE_DEVICES=0,1``)."""
    raw = (tier.pin or "").strip()
    if not raw:
        return []
    value = raw.split("=", 1)[1] if "=" in raw else raw
    cards: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            card = int(part)
        except ValueError:
            warnings.append(
                f"tier {tier.role}: pin {raw!r} is not a card list; "
                f"planning on every visible card")
            return []
        if card not in by_index:
            warnings.append(
                f"tier {tier.role}: pin {raw!r} names card {card}, which is not "
                f"a visible device; planning on the visible cards")
            return []
        if card not in cards:
            cards.append(card)
    return cards


def _tensor_split(tier, count: int) -> Optional[list[float]]:
    """Normalised ``-ts`` fractions, or None when there is nothing to use."""
    raw = (tier.ts or "").strip()
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != count:
        return None
    try:
        values = [float(p) for p in parts]
    except ValueError:
        return None
    if any(v < 0 for v in values) or sum(values) <= 0:
        return None
    total = sum(values)
    return [v / total for v in values]


# ---------------------------------------------------------------------------
# model resolution, metadata, weights, KV
# ---------------------------------------------------------------------------
def _resolve(tier, manager, warnings):
    """Absolute path of ``tier.file``, or None when the library has no such GGUF."""
    name = (getattr(tier, "file", "") or "").strip()
    if not name:
        warnings.append(f"tier {tier.role}: no GGUF file named")
        return None
    resolver = getattr(manager, "resolve_model", None) if manager else None
    try:
        path = resolver(name) if resolver is not None else None
    except Exception as exc:  # noqa: BLE001 - a bad library must not 500
        warnings.append(f"tier {tier.role}: resolving {name} failed: {exc}")
        return None
    if path is None:
        probe = Path(name)
        if probe.is_file():
            path = probe
    if path is None:
        warnings.append(
            f"tier {tier.role}: {name} is not in the local model library; "
            f"download it before applying the stack")
        return None
    return Path(path)


def _sidecar_weights(tier, manager, warnings) -> float:
    """The tier's ``mmproj`` projector, charged next to the weights it serves."""
    name = (getattr(tier, "mmproj", "") or "").strip()
    if not name:
        return 0.0
    resolver = getattr(manager, "resolve_model", None) if manager else None
    try:
        path = resolver(name) if resolver is not None else None
    except Exception:  # noqa: BLE001 - a missing projector is a warning, not a crash
        path = None
    if path is None:
        probe = Path(name)
        path = probe if probe.is_file() else None
    if path is None:
        warnings.append(
            f"tier {tier.role}: vision projector {name} is not in the local "
            f"model library; its weights are not budgeted")
        return 0.0
    size = _file_size_gib(path)
    if size is None:
        warnings.append(f"tier {tier.role}: cannot size {path}")
        return 0.0
    return size


def _uses_draft_head(tier) -> bool:
    """True when ``tier.spec`` asks llama.cpp to load a draft head."""
    spec = getattr(tier, "spec", None)
    if not isinstance(spec, dict):
        return False
    kind = str(spec.get("type") or "").strip().lower()
    return any(marker in kind for marker in _DRAFT_SPEC_TYPES)


def _file_size_gib(path) -> Optional[float]:
    try:
        return Path(path).stat().st_size / GIB
    except OSError:
        return None


def _ctx(tier, warnings) -> int:
    """The tier's context window, falling back to llama.cpp's default."""
    try:
        ctx = int(getattr(tier, "ctx", 0) or 0)
    except (TypeError, ValueError):
        ctx = 0
    if ctx <= 0:
        warnings.append(
            f"tier {tier.role}: ctx is {tier.ctx!r}; pricing KV at the "
            f"llama.cpp default of {DEFAULT_CTX}")
        return DEFAULT_CTX
    return ctx


def _kv_estimate(tier, path, manager, warnings):
    """T33's :class:`~harness.models.KVEstimate` for ``tier``, or None.

    The header is read from the GGUF itself; when that comes up thin (an
    unreadable or older file) the library's own parsed metadata from
    ``list_local()`` is used instead, so a model already in the library is
    priced from data the manager has already read.
    """
    cache_type = str(getattr(tier, "cache_k", "") or "f16")
    estimate = attention_kv_estimate(path, kv_cache_type=cache_type)
    if estimate is not None:
        return estimate
    meta = _library_metadata(path, manager)
    if meta:
        return attention_kv_estimate(path, gguf_meta=meta,
                                     kv_cache_type=cache_type)
    return None


def _library_metadata(path, manager) -> Optional[dict]:
    """``gguf_metadata`` of ``path`` from the manager's ``list_local()`` rows."""
    lister = getattr(manager, "list_local", None) if manager else None
    if lister is None:
        return None
    try:
        rows = lister() or []
    except Exception:  # noqa: BLE001 - the fallback is best-effort
        return None
    name = Path(path).name
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_file = str(row.get("file") or row.get("path") or "")
        if row_file and Path(row_file).name != name:
            continue
        meta = row.get("gguf_metadata") or row.get("ggufMetadata")
        if isinstance(meta, dict) and meta:
            return meta
    return None


def _kv_gib(tier, estimate, ctx: int) -> float:
    """KV cache GiB for ``tier``, priced at the wider of cache_k / cache_v.

    K and V carry separate ``-ctk``/``-ctv`` types, and a mixed pair must not
    be priced at the cheaper one: the cache is stored once per tensor, so the
    wider type sets the real footprint.
    """
    cache_k = str(getattr(tier, "cache_k", "") or "f16")
    cache_v = str(getattr(tier, "cache_v", "") or cache_k)
    bytes_k = estimate.bytes_at(ctx, cache_k)
    if cache_v.strip().lower() == cache_k.strip().lower():
        return bytes_k / GIB
    return max(bytes_k, estimate.bytes_at(ctx, cache_v)) / GIB
