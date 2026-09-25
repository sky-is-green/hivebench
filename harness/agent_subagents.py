"""Worker-tier subagents + the nested dispatch loop (HIVE-PLAN §B3, ADR-L2).

The face tier dispatches subordinate tiers by reusing the dsh loop's existing
``subagents`` tool: every non-face tier of the applied stack registers as a
subagent whose name/description are generated from the tier, and a dispatch
runs a **nested loop** against that tier's own llama-server endpoint.  There is
no router engine (ADR-L2).

The nested loop emits the §B3 ``model`` start/end pair around the dispatch and
tags the worker's own ``tool`` events with the model id (ADR-L6) via
:mod:`harness.agent_events`.

Frozen at T34 (ADR-L8).  T40 implements it.

Implementation notes (T40)
--------------------------

* **Host-side tool loop.** A worker tier is a bare llama-server, so it has no
  tool runtime of its own.  :func:`run_nested_loop` therefore speaks the
  OpenAI-compatible ``chat.completions`` API against ``subagent.base_url`` and
  executes tool calls *in the host*: the schemas come from the injected client
  (``client.tools``) and their execution from ``client.run_tool(name, args)``
  (or ``client.execute_tool``).  A client carrying neither is a plain
  text-completing endpoint, which is all a retrieval worker needs for the
  common case.  The injected ``client`` is duck-typed: mapping *or* attribute
  access works, so offline tests need no SDK.
* **Unimplemented sibling module.** ``harness.agent_events`` is T39's file and
  still raises :class:`NotImplementedError` in this branch.  Every helper call
  into it degrades to a local, §B3-identical implementation so T40's gate is
  self-verifiable; once T39 lands the same call sites route through it with no
  behaviour change (the module-level ``AGENT_EVENTS_IMPLEMENTED`` reports which
  path is live).
"""

from __future__ import annotations

import itertools
import json
import re
import time
from collections.abc import Mapping as _MappingABC
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from harness import agent_events
from harness.stack.schema import ROLE_FACE, Stack, Tier

#: What each role is for (ADR-L1: roles, not sizes) — the generated
#: description tells the face model when to delegate.
ROLE_PURPOSE: dict[str, str] = {
    "worker": "cheap retrieval, extraction and summarisation (fuzzy mechanics)",
    "agency": "multi-hop agentic search loops",
    "mechanics": "the cheapest rung — mechanical transforms only",
}

#: Fallback purpose for a role outside ``ROLES`` (shapes are validated by
#: ``schema.validate_shape``; this only keeps the description readable).
DEFAULT_PURPOSE = "subordinate work for this stack"

#: Loop bound for one dispatch: a worker that keeps calling tools past this is
#: truncated rather than allowed to spin the face turn.
MAX_STEPS = 8

#: Default port a tier endpoint is composed from when a status row carries a
#: port but no ``base_url`` (``TierRuntime.to_dict`` rows).
DEFAULT_TIER_HOST = "http://127.0.0.1"

_NON_SLUG = re.compile(r"[^a-z0-9]+")

_model_seq = itertools.count(1)


@dataclass
class WorkerSubagent:
    """One non-face tier exposed to the face loop as a subagent."""

    name: str
    role: str
    model: str
    base_url: str
    description: str
    api_key: str = "lm-studio"

    def to_spec(self) -> dict[str, Any]:
        """The registration entry for the dsh ``subagents`` tool.

        At least ``{"name", "description", "role", "model", "base_url"}``.
        """
        return {
            "name": self.name,
            "description": self.description,
            "role": self.role,
            "model": self.model,
            "base_url": self.base_url,
            "api_key": self.api_key,
        }


def subagent_name(tier: Tier) -> str:
    """Stable short name for a tier (``role`` + model slug), e.g.
    ``"worker-qwen3-8-4b-distill"``.  Never collides across roles."""
    return f"{_slugify(tier.role)}-{_model_slug(tier)}"


def subagent_description(tier: Tier) -> str:
    """Human/agent-facing description generated from the tier's role + model."""
    role = (tier.role or "").strip() or "tier"
    model = _model_label(tier)
    bits = [f"{role} tier ({model})"]
    if tier.ctx:
        bits.append(f"ctx {tier.ctx}")
    if tier.backend:
        bits.append(f"backend {tier.backend}")
    head = ", ".join(bits)
    purpose = ROLE_PURPOSE.get(role, DEFAULT_PURPOSE)
    return (
        f"{head}. Delegate {purpose} to it. It runs its own tool loop on its "
        f"own endpoint and returns text only, so offload fan-out work here to "
        f"keep the face tier's context free for judgement (ADR-L1, ADR-L2)."
    )


def register_worker_subagents(
    stack: Stack,
    endpoints: Mapping[str, Mapping[str, Any]],
    *,
    api_key: str = "lm-studio",
) -> list[WorkerSubagent]:
    """Build one :class:`WorkerSubagent` per non-face tier of ``stack``.

    ``endpoints`` maps ``tier.role`` → a live endpoint mapping (typically a row
    of ``StackManager.status()["tiers"]``) with ``base_url`` and ``model``
    keys; a missing endpoint skips that tier with no exception.
    """
    out: list[WorkerSubagent] = []
    for tier in stack.tiers:
        role = (getattr(tier, "role", "") or "").strip()
        if not role or role == ROLE_FACE:
            continue  # ADR-L1: the face is the caller, never a subagent
        endpoint = endpoints.get(role) if endpoints else None
        if not endpoint or not hasattr(endpoint, "get"):
            continue  # tier not resident / not applied — skip, never raise
        base_url = _endpoint_base_url(endpoint)
        if not base_url:
            continue
        model = str(endpoint.get("model") or "").strip() or _model_slug(tier)
        out.append(
            WorkerSubagent(
                name=subagent_name(tier),
                role=role,
                model=model,
                base_url=base_url,
                description=subagent_description(tier),
                api_key=api_key,
            )
        )
    return out


def run_nested_loop(
    subagent: WorkerSubagent,
    task: str,
    *,
    on_event: Optional[Callable[[dict[str, Any]], None]] = None,
    client: Any = None,
) -> dict[str, Any]:
    """Run ``subagent``'s own tool loop on ``task`` and return its final text.

    Emits a ``model`` start event, forwards the worker's ``tool`` events via
    ``on_event`` with the model id as their ``parent``, then emits the matching
    ``model`` end event carrying ``duration_ms``.  ``client`` is an injectable
    OpenAI-compatible client for offline tests.

    Returns ``{"ok": bool, "output": str, "events": int, "model_id": str}``;
    the face loop feeds ``output`` back as the subagent tool result.
    """
    model_id = _next_model_id()
    started = time.perf_counter()
    emitted: list[dict[str, Any]] = []

    def emit(event: Mapping[str, Any]) -> None:
        """Record an activity event and hand it to ``on_event`` (never raises)."""
        shaped = dict(event)
        emitted.append(shaped)
        if on_event is None:
            return
        try:
            on_event(shaped)
        except Exception:  # noqa: BLE001 - a consumer queue must not kill the dispatch
            pass

    emit(_model_event(
        tier=subagent.role, model=subagent.model,
        phase=agent_events.PHASE_START, id=model_id, task=task,
    ))

    ok = True
    output = ""
    try:
        worker = client if client is not None else _default_client(subagent)
        output = _drive_loop(worker, subagent, task, model_id=model_id, emit=emit)
    except Exception as exc:  # noqa: BLE001 - the face gets a failed tool result
        ok = False
        output = f"[{subagent.name} dispatch failed: {exc}]"

    duration_ms = int((time.perf_counter() - started) * 1000)
    emit(_model_event(
        tier=subagent.role, model=subagent.model,
        phase=agent_events.PHASE_END, id=model_id, task=task,
        duration_ms=duration_ms, output=output,
    ))
    return {
        "ok": ok,
        "output": output,
        "events": len(emitted),
        "model_id": model_id,
    }


# --------------------------------------------------------------------------
# naming helpers
# --------------------------------------------------------------------------


def _slugify(text: str) -> str:
    """Lowercase, dash-collapsed, alnum-only slug (``"Qwen3.8-4B"`` → ``"qwen3-8-4b"``)."""
    return _NON_SLUG.sub("-", str(text or "").lower()).strip("-")


def _model_label(tier: Tier) -> str:
    """``"empero-ai/Qwen3.8-4B-Distill"`` (repo) else the gguf file name."""
    repo = (getattr(tier, "repo", "") or "").strip()
    if repo:
        return repo
    return (getattr(tier, "file", "") or "").strip() or "unknown model"


def _model_slug(tier: Tier) -> str:
    """Model slug for a name/endpoint fallback: the repo tail, quant suffix dropped."""
    label = _model_label(tier)
    if "/" in label:
        label = label.rsplit("/", 1)[-1]
    if label.lower().endswith(".gguf"):
        label = label[: -len(".gguf")]
    return _slugify(label) or "tier"


def _endpoint_base_url(endpoint: Mapping[str, Any]) -> str:
    """``base_url`` of an endpoint row, or one composed from its ``port``."""
    url = str(endpoint.get("base_url") or "").strip()
    if not url:
        port = endpoint.get("port")
        try:
            port = int(port) if port is not None else 0
        except (TypeError, ValueError):
            port = 0
        if port <= 0:
            return ""
        url = f"{DEFAULT_TIER_HOST}:{port}/v1"
    return url.rstrip("/")


# --------------------------------------------------------------------------
# the nested loop (ADR-L2: the subagent's own endpoint, no router)
# --------------------------------------------------------------------------


def _default_client(subagent: WorkerSubagent) -> Any:
    """An OpenAI-compatible client pointed at the worker's own endpoint.

    Raises when no SDK is installed — :func:`run_nested_loop` turns that into
    ``ok=False`` rather than a traceback, since an offline worker tier must not
    take the face turn down with it.
    """
    from openai import OpenAI  # noqa: PLC0415 - optional dependency, imported on demand

    return OpenAI(base_url=subagent.base_url, api_key=subagent.api_key or "lm-studio")


def _drive_loop(client: Any, subagent: WorkerSubagent, task: str, *,
                model_id: str, emit: Callable[[Mapping[str, Any]], None]) -> str:
    """Chat-completions loop against the worker's endpoint; return final text.

    Tool calls are answered by the host (``client.run_tool``/``execute_tool``)
    and each one is reported to ``emit`` as a ``tool`` event parented to
    ``model_id`` (ADR-L6).
    """
    messages: list[dict[str, Any]] = [{"role": "user", "content": task}]
    tools = _host_tools(client)
    text = ""
    for _ in range(MAX_STEPS):
        kwargs: dict[str, Any] = {"model": subagent.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        response = client.chat.completions.create(**kwargs)
        message = _field(_first_choice(response), "message")
        text = _field(message, "content") or ""
        calls = _field(message, "tool_calls") or []
        if not calls:
            return text
        messages.append(_assistant_turn(message, text, calls))
        for call in calls:
            call_id, name, arguments = _read_call(call)
            emit(_tool_event(subagent, model_id, name, phase="call", args=arguments))
            result = _run_host_tool(client, name, arguments)
            emit(_tool_event(subagent, model_id, name, phase="result", result=result))
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": result,
            })
    return text or f"[{subagent.name} produced no final text within {MAX_STEPS} steps]"


def _host_tools(client: Any) -> list[Any]:
    """Tool schemas the host offers the worker (``client.tools``), if any."""
    tools = getattr(client, "tools", None)
    if not tools:
        return []
    return list(tools)


def _run_host_tool(client: Any, name: str, arguments: Any) -> str:
    """Execute one worker tool call in the host; string result for the model."""
    runner = getattr(client, "run_tool", None)
    if not callable(runner):
        runner = getattr(client, "execute_tool", None)
    if not callable(runner):
        return f"[no executor for tool {name!r} on this client]"
    return str(runner(name, arguments))


def _assistant_turn(message: Any, text: str, calls: list[Any]) -> dict[str, Any]:
    """The assistant message to echo back, in OpenAI wire shape."""
    wire_calls = []
    for call in calls:
        call_id, name, arguments = _read_call(call)
        wire_calls.append({
            "id": call_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": arguments if isinstance(arguments, str)
                else json.dumps(arguments),
            },
        })
    return {
        "role": "assistant",
        "content": text or None,
        "tool_calls": wire_calls,
    }


def _read_call(call: Any) -> tuple[str, str, Any]:
    """``(call_id, function_name, arguments)`` from a tool call, object or dict."""
    function = _field(call, "function")
    name = str(_field(function, "name") or _field(call, "name") or "")
    arguments = _field(function, "arguments")
    if arguments is None:
        arguments = _field(call, "arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError:
            pass  # a malformed argument blob is the model's problem, not ours
    return str(_field(call, "id") or f"call_{name or 'tool'}"), name, arguments


# --------------------------------------------------------------------------
# event shaping (§B3, ADR-L6) — agent_events with a §B3-identical fallback
# --------------------------------------------------------------------------


def _next_model_id() -> str:
    """A fresh ``m-<n>`` id for one dispatch span."""
    seq = next(_model_seq)
    try:
        return agent_events.new_model_id(seq)
    except NotImplementedError:
        return f"{agent_events.MODEL_ID_PREFIX}{seq}"


def _model_event(**fields: Any) -> dict[str, Any]:
    """One §B3 ``model`` event via :mod:`harness.agent_events`.

    Falls back to the local equivalent while T39's module is still a skeleton,
    so the event shape is identical either way.
    """
    try:
        return dict(agent_events.model_event(**fields))
    except NotImplementedError:
        return _local_model_event(**fields)


def _local_model_event(*, tier: str, model: str, phase: str, id: Optional[str] = None,
                       parent: Optional[str] = None, task: Optional[str] = None,
                       duration_ms: Optional[int] = None,
                       output: Optional[str] = None) -> dict[str, Any]:
    """The §B3 ``model`` event, built locally (T39 not merged)."""
    event: dict[str, Any] = {
        "type": agent_events.EVENT_MODEL,
        "tier": tier,
        "model": model,
        "phase": phase,
        "id": id,
        "parent": parent,
    }
    if task is not None:
        event["task"] = task
    if duration_ms is not None:
        event["duration_ms"] = duration_ms
    if output is not None:
        event["output"] = output
    return event


def _tool_event(subagent: WorkerSubagent, parent: str, tool: str, *,
                phase: str, args: Any = None, result: Any = None) -> dict[str, Any]:
    """A worker ``tool`` event carrying the dispatching ``model`` id as parent."""
    event: dict[str, Any] = {
        "type": agent_events.EVENT_TOOL,
        "tool": tool,
        "phase": phase,
        "tier": subagent.role,
        agent_events.PARENT_KEY: parent,
    }
    if args is not None:
        event["args"] = args
    if result is not None:
        event["result"] = result
    try:
        return dict(agent_events.with_parent(event, parent))
    except NotImplementedError:
        return event


def _agent_events_is_skeleton() -> bool:
    """Whether the agent_events helpers still raise :class:`NotImplementedError`."""
    probe: dict[str, Any] = {
        "type": agent_events.EVENT_TOOL, "tool": "probe", "phase": "call",
    }
    try:
        agent_events.with_parent(probe, "m-0")
    except NotImplementedError:
        return True
    return False


#: Whether the ``model``/parent helpers currently come from
#: :mod:`harness.agent_events` (T39 merged) or from the local fallback.  Both
#: produce the same §B3 shape; this is informational for tests and operators.
AGENT_EVENTS_IMPLEMENTED: bool = not _agent_events_is_skeleton()


# --------------------------------------------------------------------------
# duck-typed access (real SDK objects, dict fakes, or anything in between)
# --------------------------------------------------------------------------


def _field(obj: Any, name: str) -> Any:
    """``obj.name`` or ``obj[name]``; ``None`` when absent."""
    if obj is None:
        return None
    if isinstance(obj, _MappingABC):
        return obj.get(name)
    return getattr(obj, name, None)


def _first_choice(response: Any) -> Any:
    """The first choice of a chat-completions response."""
    choices = _field(response, "choices")
    if not choices:
        return None
    return choices[0]
