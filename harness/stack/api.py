"""``/v1/stacks/*`` router factory (HIVE-PLAN §B3).

Builds the FastAPI router for the six stack paths.  It is a *factory* so T44
can mount it into the app in the single integration commit; nothing here is
registered at import time.  Handlers stay thin: ``schema`` load/save,
``residency.plan_residency``, ``manager`` apply/status/unload.

Frozen at T34 (ADR-L8).  T38 implements it, T44 mounts it.

Route table (§B3) — register ``/status`` *before* ``/{name}`` or ``status``
would be captured as a stack name:

===============================  ====  ======================================
Path                             Verb  Contract
===============================  ====  ======================================
``/v1/stacks``                   GET   list ``{name, tiers, roles}``
``/v1/stacks/status``            GET   per-tier residency/port/ctx/VRAM/tok/s
``/v1/stacks/{name}``            GET   read one stack
``/v1/stacks/{name}``            PUT   write one stack (body = §B3 document)
``/v1/stacks/{name}``            DELETE delete one stack
``/v1/stacks/{name}/validate``   POST  ``{ok, per_card, warnings}``
``/v1/stacks/{name}/apply``      POST  ``{ok, tiers:[{role, port, key}]}``
``/v1/stacks/{name}/unload``     POST  stop all tiers
===============================  ====  ======================================

Malformed input / missing stacks must surface as 4xx (``HTTPException``), never
500 (T38).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from harness.stack.manager import StackManager

#: Router prefix (the app mounts the router without adding another prefix).
PREFIX = "/v1/stacks"


def create_router(manager: StackManager, *, stacks_root: Any = None) -> APIRouter:
    """Return the ``/v1/stacks`` router bound to ``manager``.

    ``stacks_root`` overrides the stack directory (default ``<repo>/stacks``);
    it is a ``Path``-like, not a full filesystem path to one file.
    """
    raise NotImplementedError
