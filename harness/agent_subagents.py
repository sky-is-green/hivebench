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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from harness.stack.schema import Stack, Tier


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
        raise NotImplementedError


def subagent_name(tier: Tier) -> str:
    """Stable short name for a tier (``role`` + model slug), e.g.
    ``"worker-qwen3-8-4b-distill"``.  Never collides across roles."""
    raise NotImplementedError


def subagent_description(tier: Tier) -> str:
    """Human/agent-facing description generated from the tier's role + model."""
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
