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
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Tier":
        """Parse one tier object; raise ``ValueError`` on malformed fields."""
        raise NotImplementedError


@dataclass
class Stack:
    """A saved stack: ordered tiers plus the routing block (§B3)."""

    name: str
    version: int = STACK_VERSION
    tiers: list[Tier] = field(default_factory=list)
    routing: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_ROUTING))

    def to_dict(self) -> dict[str, Any]:
        """The full ``{"name", "version", "tiers", "routing"}`` document."""
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Stack":
        """Parse a stack document; raise ``ValueError`` on malformed input."""
        raise NotImplementedError

    @property
    def roles(self) -> list[str]:
        """Tier roles in stack order (used by the list route)."""
        raise NotImplementedError


def stacks_dir(root: str | Path | None = None) -> Path:
    """Directory holding ``<name>.json``.

    ``root`` overrides the repo root (tests); default is ``<repo>/stacks``.
    """
    raise NotImplementedError


def stack_path(name: str, *, root: str | Path | None = None) -> Path:
    """Path of ``stacks/<name>.json`` (rejects path traversal in ``name``)."""
    raise NotImplementedError


def load_stack(name: str, *, root: str | Path | None = None) -> Stack:
    """Read and shape-validate one saved stack.

    Raises ``FileNotFoundError`` when absent and ``ValueError`` when the file
    is malformed; it does not run residency validation (that is
    ``residency.plan_residency``).
    """
    raise NotImplementedError


def list_stacks(*, root: str | Path | None = None) -> list[dict[str, Any]]:
    """Saved-stack summaries for ``GET /v1/stacks``.

    One row per file: ``{"name", "tiers", "roles"}``, sorted by name.
    """
    raise NotImplementedError


def save_stack(stack: Stack, *, root: str | Path | None = None) -> Path:
    """Write ``stack`` to ``stacks/<stack.name>.json`` atomically; return the path."""
    raise NotImplementedError


def delete_stack(name: str, *, root: str | Path | None = None) -> bool:
    """Delete ``stacks/<name>.json``; return whether a file was removed."""
    raise NotImplementedError


def validate_shape(stack: Stack) -> list[str]:
    """Structural errors only: ``[]`` means the shape is valid.

    Checks at minimum: stack name present, every role in :data:`ROLES`, no
    duplicate roles, and every tier has ``repo``/``file``.  No VRAM/KV budget
    checks — those are ``residency`` (ADR-L5).
    """
    raise NotImplementedError
