"""Stack schema — the ``stacks/<name>.json`` data model (HIVE-PLAN §B3).

A *stack* is an ordered list of **tiers**, each tier one local GGUF plus its
launch configuration.  This module owns the pure data model, the on-disk JSON
format, and **shape** validation (unknown / missing / duplicate roles,
malformed tiers).  It does no hardware probing and spawns no processes — that
is ``residency``/``manager``.

Frozen at T34 (ADR-L8).  The skeletons are the wire contract; T35 implements
this module and nobody re-negotiates the signatures.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

# Tier roles (LOCAL-STACKS.md §2) — ADR-L1: roles, not sizes.  ``face`` is the
# model the human talks to; every other role is a subordinate tier.
ROLE_FACE = "face"
ROLE_WORKER = "worker"
ROLE_AGENCY = "agency"
ROLE_MECHANICS = "mechanics"
ROLES: tuple[str, ...] = (ROLE_FACE, ROLE_WORKER, ROLE_AGENCY, ROLE_MECHANICS)

#: On-disk schema version (§B3 ``version``).
STACK_VERSION = 1

#: Default ``routing`` block (§B3): non-face tiers are dispatched as subagents
#: (ADR-L2), never through a router engine.
DEFAULT_ROUTING: dict[str, Any] = {"workers_as": "subagent"}

#: Default directory under the repo root that holds ``<name>.json``.
STACKS_DIRNAME = "stacks"

#: Fields the ``Tier`` dataclass round-trips, in §B3 document order.  The
#: optional tail is written only when set — a 4B worker tier in §B3 carries
#: no ``spec``/``mmproj``/``ts`` keys at all.
_TIER_REQUIRED: tuple[str, ...] = ("role", "repo", "file")
_TIER_FIELDS: tuple[str, ...] = (
    "role", "repo", "file", "ctx", "ngl", "backend", "cache_k", "cache_v",
)
_TIER_OPTIONAL: tuple[str, ...] = ("spec", "mmproj", "pin", "ts")
_STACK_FIELDS: tuple[str, ...] = ("name", "version", "tiers", "routing")


def _as_text(value: Any, field_name: str) -> str:
    """A required, non-empty, non-blank string."""
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string, got {type(value).__name__}")
    text = value.strip()
    if not text:
        raise ValueError(f"{field_name} must not be empty")
    return text


def _as_optional_text(value: Any, field_name: str) -> Optional[str]:
    """``None``/``""`` stay absent; anything present must be a non-blank string."""
    if value is None or value == "":
        return None
    return _as_text(value, field_name)


def _as_int(value: Any, field_name: str, minimum: int) -> int:
    """An ``int`` (a digit string is coerced, matching ``load``'s ``ngl``)."""
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer, got bool")
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value.strip())
    if not isinstance(value, int):
        raise ValueError(
            f"{field_name} must be an integer, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}, got {value}")
    return int(value)


def _reject_unknown(data: Mapping[str, Any], known: tuple[str, ...], where: str) -> None:
    """A typo'd key is a malformed file, not a silently dropped field."""
    unknown = [str(k) for k in data if k not in known]
    if unknown:
        raise ValueError(
            f"{where} has unknown field(s): {', '.join(sorted(unknown))}; "
            f"known: {', '.join(known)}")


@dataclass
class Tier:
    """One stack tier: a downloaded GGUF and the config to launch it.

    Field names mirror the §B3 stack file exactly.  ``pin`` is the tier's
    ``HIP_VISIBLE_DEVICES`` string; ``ts`` is the llama.cpp tensor split
    (e.g. ``"1,1"``); ``spec`` is the speculative-decoding block
    (``{"type": "draft-mtp", "n_max": 3}``).
    """

    role: str
    repo: str
    file: str
    ctx: int = 8192
    ngl: int = 99
    backend: str = "vulkan"
    cache_k: str = "q8_0"
    cache_v: str = "q8_0"
    spec: Optional[dict[str, Any]] = None
    mmproj: Optional[str] = None
    pin: Optional[str] = None
    ts: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """The tier as it appears in a ``stacks/<name>.json`` file."""
        out: dict[str, Any] = {
            "role": self.role,
            "repo": self.repo,
            "file": self.file,
            "ctx": int(self.ctx),
            "ngl": int(self.ngl),
            "backend": self.backend,
            "cache_k": self.cache_k,
            "cache_v": self.cache_v,
        }
        if self.spec is not None:
            out["spec"] = dict(self.spec)
        if self.mmproj is not None:
            out["mmproj"] = self.mmproj
        if self.pin is not None:
            out["pin"] = self.pin
        if self.ts is not None:
            out["ts"] = self.ts
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Tier":
        """Parse one tier object; raise ``ValueError`` on malformed fields."""
        if not isinstance(data, Mapping):
            raise ValueError(f"tier must be an object, got {type(data).__name__}")
        _reject_unknown(data, _TIER_FIELDS + _TIER_OPTIONAL, "tier")

        missing = [key for key in _TIER_REQUIRED if data.get(key) is None]
        if missing:
            raise ValueError(f"tier is missing required field(s): {', '.join(missing)}")

        spec = data.get("spec")
        if spec is not None and not isinstance(spec, Mapping):
            raise ValueError(
                f"tier spec must be an object, got {type(spec).__name__}")

        return cls(
            role=_as_text(data["role"], "tier role"),
            repo=_as_text(data["repo"], "tier repo"),
            file=_as_text(data["file"], "tier file"),
            ctx=_as_int(data.get("ctx", 8192), "tier ctx", minimum=1),
            ngl=_as_int(data.get("ngl", 99), "tier ngl", minimum=0),
            backend=_as_text(data.get("backend", "vulkan"), "tier backend"),
            cache_k=_as_text(data.get("cache_k", "q8_0"), "tier cache_k"),
            cache_v=_as_text(data.get("cache_v", "q8_0"), "tier cache_v"),
            spec=dict(spec) if spec is not None else None,
            mmproj=_as_optional_text(data.get("mmproj"), "tier mmproj"),
            pin=_as_optional_text(data.get("pin"), "tier pin"),
            ts=_as_optional_text(data.get("ts"), "tier ts"),
        )


@dataclass
class Stack:
    """A saved stack: ordered tiers plus the routing block (§B3)."""

    name: str
    version: int = STACK_VERSION
    tiers: list[Tier] = field(default_factory=list)
    routing: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_ROUTING))

    def to_dict(self) -> dict[str, Any]:
        """The full ``{"name", "version", "tiers", "routing"}`` document."""
        return {
            "name": self.name,
            "version": int(self.version),
            "tiers": [tier.to_dict() for tier in self.tiers],
            "routing": dict(self.routing),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Stack":
        """Parse a stack document; raise ``ValueError`` on malformed input."""
        if not isinstance(data, Mapping):
            raise ValueError(f"stack must be an object, got {type(data).__name__}")
        _reject_unknown(data, _STACK_FIELDS, "stack")

        if data.get("name") is None:
            raise ValueError("stack is missing required field: name")

        version = _as_int(data.get("version", STACK_VERSION), "stack version", minimum=1)
        if version != STACK_VERSION:
            raise ValueError(
                f"unsupported stack version {version}; this build reads "
                f"version {STACK_VERSION}")

        raw_tiers = data.get("tiers", [])
        if raw_tiers is None:
            raw_tiers = []
        if not isinstance(raw_tiers, (list, tuple)):
            raise ValueError(
                f"stack tiers must be a list, got {type(raw_tiers).__name__}")

        routing = data.get("routing")
        if routing is None:
            routing = dict(DEFAULT_ROUTING)
        elif not isinstance(routing, Mapping):
            raise ValueError(
                f"stack routing must be an object, got {type(routing).__name__}")

        return cls(
            name=_as_text(data["name"], "stack name"),
            version=version,
            tiers=[Tier.from_dict(tier) for tier in raw_tiers],
            routing=dict(routing),
        )

    @property
    def roles(self) -> list[str]:
        """Tier roles in stack order (used by the list route)."""
        return [tier.role for tier in self.tiers]


def _check_name(name: str) -> str:
    """A stack name is a filename stem, never a path."""
    if not isinstance(name, str):
        raise ValueError(f"stack name must be a string, got {type(name).__name__}")
    stem = name.strip()
    if not stem:
        raise ValueError("stack name must not be empty")
    if stem.endswith(".json"):
        stem = stem[: -len(".json")].strip()
    if not stem:
        raise ValueError("stack name must not be empty")
    if "\x00" in stem:
        raise ValueError("stack name must not contain a NUL byte")
    if os.path.isabs(stem) or Path(stem).name != stem or "/" in stem or "\\" in stem:
        raise ValueError(f"stack name must be a bare filename, got {name!r}")
    if stem in (".", "..") or stem.startswith("."):
        raise ValueError(f"stack name must be a bare filename, got {name!r}")
    return stem


def stacks_dir(root: str | Path | None = None) -> Path:
    """Directory holding ``<name>.json``.

    ``root`` overrides the repo root (tests); default is ``<repo>/stacks``.
    """
    base = Path(__file__).resolve().parents[2] if root is None else Path(root)
    return base / STACKS_DIRNAME


def stack_path(name: str, *, root: str | Path | None = None) -> Path:
    """Path of ``stacks/<name>.json`` (rejects path traversal in ``name``)."""
    return stacks_dir(root) / f"{_check_name(name)}.json"


def load_stack(name: str, *, root: str | Path | None = None) -> Stack:
    """Read and shape-validate one saved stack.

    Raises ``FileNotFoundError`` when absent and ``ValueError`` when the file
    is malformed; it does not run residency validation (that is
    ``residency.plan_residency``).
    """
    path = stack_path(name, root=root)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"no saved stack at {path}") from None
    except OSError as exc:
        raise ValueError(f"cannot read stack at {path}: {exc}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"stack file {path.name} is not valid JSON: {exc}") from exc

    stack = Stack.from_dict(data)
    problems = validate_shape(stack)
    if problems:
        raise ValueError(
            f"stack {stack.name!r} has an invalid shape: " + "; ".join(problems))
    return stack


def list_stacks(*, root: str | Path | None = None) -> list[dict[str, Any]]:
    """Saved-stack summaries for ``GET /v1/stacks``.

    One row per file: ``{"name", "tiers", "roles"}``, sorted by name.
    """
    directory = stacks_dir(root)
    if not directory.is_dir():
        return []

    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json"), key=lambda p: p.name):
        try:
            stack = Stack.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, json.JSONDecodeError):
            # A hand-mangled file must not take the whole list route down; the
            # per-stack GET is where the error surfaces.
            continue
        rows.append({
            "name": stack.name or path.stem,
            "tiers": len(stack.tiers),
            "roles": stack.roles,
        })
    rows.sort(key=lambda row: str(row["name"]))
    return rows


def save_stack(stack: Stack, *, root: str | Path | None = None) -> Path:
    """Write ``stack`` to ``stacks/<stack.name>.json`` atomically; return the path."""
    if not isinstance(stack, Stack):
        raise ValueError(f"expected a Stack, got {type(stack).__name__}")
    path = stack_path(stack.name, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(stack.to_dict(), indent=2, ensure_ascii=False) + "\n"

    handle, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.stem}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def delete_stack(name: str, *, root: str | Path | None = None) -> bool:
    """Delete ``stacks/<name>.json``; return whether a file was removed."""
    path = stack_path(name, root=root)
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    except IsADirectoryError:
        return False
    except OSError as exc:
        raise ValueError(f"cannot delete stack at {path}: {exc}") from exc
    return True


def validate_shape(stack: Stack) -> list[str]:
    """Structural errors only: ``[]`` means the shape is valid.

    Checks at minimum: stack name present, every role in :data:`ROLES`, no
    duplicate roles, and every tier has ``repo``/``file``.  No VRAM/KV budget
    checks — those are ``residency`` (ADR-L5).
    """
    if not isinstance(stack, Stack):
        return [f"expected a Stack, got {type(stack).__name__}"]

    errors: list[str] = []
    try:
        _check_name(stack.name)
    except ValueError as exc:
        errors.append(str(exc))

    if not isinstance(stack.routing, Mapping):
        errors.append(
            f"routing must be an object, got {type(stack.routing).__name__}")

    if not stack.tiers:
        errors.append("stack has no tiers (every stack needs a face tier)")
        return errors

    seen: dict[str, int] = {}
    for position, tier in enumerate(stack.tiers, start=1):
        label = f"tier {position}"
        role = getattr(tier, "role", None)
        if not isinstance(role, str) or not role.strip():
            errors.append(f"{label}: missing tier role")
        elif role not in ROLES:
            errors.append(
                f"{label}: role {role!r} is not one of {', '.join(ROLES)}")
        elif role in seen:
            errors.append(
                f"{label}: duplicate tier role {role!r} (already used by tier "
                f"{seen[role]})")
        else:
            seen[role] = position

        for required in ("repo", "file"):
            value = getattr(tier, required, None)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{label}: missing {required}")

        ctx = getattr(tier, "ctx", None)
        if isinstance(ctx, bool) or not isinstance(ctx, int) or ctx <= 0:
            errors.append(f"{label}: ctx must be a positive integer, got {ctx!r}")

    return errors
