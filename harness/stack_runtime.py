"""Generate the dsh runtime profile for the applied stack (LSC wiring, T48).

The dsh base bundle already mounts everything the delegation needs — the
``@deepseek-ai/dsh-subagent-spawn-in-process`` provider (name ``spawn``), the
``dsh-tool-subagent`` tool plugin, and a dormant ``llm-pi-ai`` adapter row —
but it mounts **no tier routes and no per-tier tools**.  This module fills
exactly that gap from ``StackManager.status()``:

* one ``llm-pi-ai`` route per non-face tier (``tier-worker``, ``tier-agency``,
  ``tier-mechanics``) pointing at that tier's llama-server endpoint, with the
  tier's model and context window;
* one ``dsh-tool-subagent`` instance per tier (``delegate_worker``, ...) whose
  ``agentOptions`` route children to that tier, capped at delegation depth 0
  so a worker cannot spawn further children.

The output is the runtime *profile* config: the caller's base profile text
(the SDK's bundled default, or an explicit ``DSH_CORDIS_CONFIG``) with the
generated rows appended.  Each appended row is a JSON flow mapping — JSON is
valid YAML — so the module needs no YAML dependency and never re-serialises
the base profile (which carries ``!!js`` tags a round-trip would lose).

``dsh`` config is list-of-plugin rows: a row whose ``id`` matches one the
base bundle mounted replaces that row's config (last write wins).  The
generated ``llm-pi-ai`` row therefore flips the dormant adapter live with our
routes; the tool rows are new ids.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

#: dsh plugin module names (the base bundle ships them; we mount/configure).
PI_AI_PLUGIN = "@deepseek-ai/dsh-llm-pi-ai"
SUBAGENT_TOOL_PLUGIN = "@deepseek-ai/dsh-tool-subagent"

#: Row id of the dormant pi-ai adapter in the dsh base bundle.  Reusing the id
#: replaces that row's (empty) config instead of mounting a second adapter.
PI_AI_ROW_ID = "llm-pi-ai"

#: pi-ai protocol name for OpenAI chat-completions endpoints (llama-server).
OPENAI_COMPLETIONS_API = "openai-completions"

#: Provider route registered by ``dsh-subagent-spawn-in-process`` in the base
#: bundle; the tool rows delegate through it.
SPAWN_PROVIDER = "spawn"

#: System-side subagent tool names per role (face is never a subagent).
TOOL_NAMES: dict[str, str] = {
    "worker": "delegate_worker",
    "agency": "delegate_agency",
    "mechanics": "delegate_mechanics",
}


def _tool_name(role: str) -> str:
    """Model-facing tool name for a role: ``delegate_<role>`` (slugged)."""
    name = TOOL_NAMES.get(role)
    if name:
        return name
    slug = "".join(ch if ch.isalnum() else "_" for ch in role.lower()).strip("_")
    return f"delegate_{slug or 'tier'}"


def _endpoint_base_url(tier: Mapping[str, Any]) -> str:
    """Tier endpoint URL: an explicit ``base_url`` else composed from ``port``."""
    url = str(tier.get("base_url") or "").strip()
    if url:
        return url.rstrip("/")
    port = tier.get("port")
    try:
        port = int(port) if port is not None else 0
    except (TypeError, ValueError):
        port = 0
    return f"http://127.0.0.1:{port}/v1" if port > 0 else ""


@dataclass
class TierRoute:
    """One non-face tier's routing facts (the generated config carries them)."""

    role: str
    route: str
    model: str
    tool: str
    base_url: str
    ctx: int = 0


@dataclass
class GeneratedRuntime:
    """Result of :func:`generate_runtime_config`."""

    path: Optional[Path]
    digest: str
    routes: list[TierRoute] = field(default_factory=list)
    entries: list[dict[str, Any]] = field(default_factory=list)

    def route_for(self, role: str) -> Optional[TierRoute]:
        for route in self.routes:
            if route.role == role:
                return route
        return None


def tier_routes(status: Optional[Mapping[str, Any]]) -> list[TierRoute]:
    """The non-face tiers of a ``StackManager.status()`` payload, face first order."""
    tiers = (status or {}).get("tiers") or []
    routes: list[TierRoute] = []
    for tier in tiers:
        if not isinstance(tier, Mapping):
            continue
        role = str(tier.get("role") or "").strip()
        if not role or role == "face":
            continue  # the face is the caller, never a subagent (ADR-L1)
        base_url = _endpoint_base_url(tier)
        if not base_url:
            continue  # not resident / no port yet — skip, never raise
        model = str(tier.get("model") or "").strip() or role
        try:
            ctx = int(tier.get("ctx") or 0)
        except (TypeError, ValueError):
            ctx = 0
        routes.append(TierRoute(
            role=role,
            route=f"tier-{role}",
            model=model,
            tool=_tool_name(role),
            base_url=base_url,
            ctx=max(0, ctx),
        ))
    return routes


def routing_entries(routes: list[TierRoute]) -> list[dict[str, Any]]:
    """The dsh plugin rows for ``routes`` (empty when there are none)."""
    if not routes:
        return []
    providers: dict[str, Any] = {}
    tools: list[dict[str, Any]] = []
    for route in routes:
        # ``api`` is required for a route the pi-ai catalog does not ship;
        # llama-server speaks OpenAI chat completions.
        model: dict[str, Any] = {"id": route.model}
        if route.ctx > 0:
            # Size the request context to the tier's served window; a worker
            # that overruns its llama-server ctx fails its whole dispatch.
            model["contextWindow"] = route.ctx
        providers[route.route] = {
            "api": OPENAI_COMPLETIONS_API,
            "baseURL": route.base_url,
            "models": [model],
        }
        tools.append({
            "id": f"lsc-delegate-{route.role}",
            "name": SUBAGENT_TOOL_PLUGIN,
            "config": {
                "provider": SPAWN_PROVIDER,
                "toolName": route.tool,
                "agentOptions": {
                    "provider": route.route,
                    "model": route.model,
                },
                # A worker runs its own tools; it must not spawn children.
                "maxDepth": 0,
                # Delegation is a foreground call the face waits on (v1).
                "enableRunInBackground": False,
            },
        })
    return [
        {"id": PI_AI_ROW_ID, "name": PI_AI_PLUGIN, "config": {"providers": providers}},
        *tools,
    ]


def render_entries(entries: list[dict[str, Any]]) -> str:
    """The generated rows as a YAML block (one JSON flow mapping per row).

    JSON is a YAML subset, so the runtime's YAML loader reads each row as a
    mapping; rendering with ``json`` keeps this module free of a YAML
    dependency and byte-deterministic for a given stack.
    """
    return "".join(
        "- " + json.dumps(entry, sort_keys=True, separators=(", ", ": ")) + "\n"
        for entry in entries
    )


def config_digest(entries: list[dict[str, Any]]) -> str:
    """Stable short digest of the routing rows (drives the config filename).

    Row order carries no load semantics in dsh, so the digest sorts by row id
    and lets ``json`` sort nested keys: the same stack always maps to the same
    file, regardless of how the status rows were ordered.
    """
    canonical = sorted(entries, key=lambda entry: str(entry.get("id") or ""))
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def default_base_config() -> Optional[Path]:
    """The caller's base profile: ``$DSH_CORDIS_CONFIG`` else the bundled default."""
    override = os.environ.get("DSH_CORDIS_CONFIG")
    if override:
        path = Path(override)
        return path if path.is_file() else None
    try:
        from deepseek_harness_runtime import bundled_default_config_path

        path = bundled_default_config_path()
    except Exception:  # noqa: BLE001 - runtime package absent in dev/CI checkouts
        return None
    return path if path.is_file() else None


def generate_runtime_config(
    status: Optional[Mapping[str, Any]],
    *,
    base_path: Optional[Path] = None,
    out_dir: Optional[Path] = None,
) -> GeneratedRuntime:
    """Write the profile for ``status`` and return its path + routing facts.

    Returns a result with ``path=None`` and no entries when the stack has no
    resident non-face tier (or no base profile / output directory): the
    service then launches with the unmodified base config.
    """
    routes = tier_routes(status)
    entries = routing_entries(routes)
    digest = config_digest(entries)
    if not entries or base_path is None or out_dir is None:
        return GeneratedRuntime(path=None, digest=digest, routes=routes, entries=entries)
    base = Path(base_path)
    out_dir = Path(out_dir)
    try:
        base_text = base.read_text(encoding="utf-8")
    except OSError:
        return GeneratedRuntime(path=None, digest=digest, routes=routes, entries=entries)
    if not base_text.endswith("\n"):
        base_text += "\n"
    profile = base_text + render_entries(entries)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"lsc-stack-{digest}.yml"
    path.write_text(profile, encoding="utf-8")
    return GeneratedRuntime(path=path, digest=digest, routes=routes, entries=entries)
