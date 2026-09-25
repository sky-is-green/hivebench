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

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, HTTPException

from harness.stack import residency as _residency
from harness.stack import schema as _schema
from harness.stack.manager import StackManager

#: Router prefix (the app mounts the router without adding another prefix).
PREFIX = "/v1/stacks"


def _resolve_schema_root(stacks_root: Any) -> Any:
    """Translate the router's stack *directory* into ``schema``'s repo root.

    ``create_router``'s ``stacks_root`` names the directory that holds
    ``<name>.json`` (default ``<repo>/stacks``); every ``schema`` entry point
    takes the *repo root* and appends ``stacks/`` itself.  Passing ``None``
    keeps ``schema``'s own default resolution.
    """
    if stacks_root is None:
        return None
    return Path(stacks_root).parent


def create_router(manager: StackManager, *, stacks_root: Any = None) -> APIRouter:
    """Return the ``/v1/stacks`` router bound to ``manager``.

    ``stacks_root`` overrides the stack directory (default ``<repo>/stacks``);
    it is a ``Path``-like, not a full filesystem path to one file.
    """
    router = APIRouter(prefix=PREFIX, tags=["stacks"])
    schema_root = _resolve_schema_root(stacks_root)

    def _load(name: str):
        """Load one stack, mapping absent/malformed files to 4xx."""
        try:
            return _schema.load_stack(name, root=schema_root)
        except FileNotFoundError as exc:
            raise HTTPException(404, f"no such stack: {name}") from exc
        except ValueError as exc:
            raise HTTPException(400, f"invalid stack {name!r}: {exc}") from exc

    # NOTE: ``/status`` must precede ``/{name}`` — route order is matching order.
    @router.get("")
    def list_saved() -> list[dict[str, Any]]:
        return _schema.list_stacks(root=schema_root)

    @router.get("/status")
    def stack_status() -> dict[str, Any]:
        return manager.status()

    @router.get("/{name}")
    def get_stack(name: str) -> dict[str, Any]:
        return _load(name).to_dict()

    @router.put("/{name}")
    def put_stack(name: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        payload = dict(body)
        payload.setdefault("name", name)
        if payload.get("name") != name:
            raise HTTPException(
                400,
                f"body stack name {payload.get('name')!r} does not match path {name!r}",
            )
        try:
            stack = _schema.Stack.from_dict(payload)
        except ValueError as exc:
            raise HTTPException(400, f"invalid stack {name!r}: {exc}") from exc
        errors = _schema.validate_shape(stack)
        if errors:
            raise HTTPException(400, "invalid stack: " + "; ".join(errors))
        try:
            _schema.save_stack(stack, root=schema_root)
        except ValueError as exc:
            raise HTTPException(400, f"invalid stack {name!r}: {exc}") from exc
        return stack.to_dict()

    @router.delete("/{name}")
    def delete_saved(name: str) -> dict[str, Any]:
        try:
            removed = _schema.delete_stack(name, root=schema_root)
        except ValueError as exc:
            raise HTTPException(400, f"invalid stack {name!r}: {exc}") from exc
        if not removed:
            raise HTTPException(404, f"no such stack: {name}")
        return {"ok": True, "deleted": name}

    @router.post("/{name}/validate")
    def validate_stack(name: str) -> dict[str, Any]:
        stack = _load(name)
        try:
            plan = _residency.plan_residency(stack)
        except ValueError as exc:
            raise HTTPException(400, f"invalid stack {name!r}: {exc}") from exc
        return plan.to_dict()

    @router.post("/{name}/apply")
    def apply_stack(name: str) -> dict[str, Any]:
        stack = _load(name)
        try:
            return manager.apply(stack)
        except ValueError as exc:
            raise HTTPException(400, f"invalid stack {name!r}: {exc}") from exc
        except RuntimeError as exc:
            raise HTTPException(409, f"cannot apply {name!r}: {exc}") from exc

    @router.post("/{name}/unload")
    def unload_stack(name: str) -> dict[str, Any]:
        return manager.unload(name)

    return router
