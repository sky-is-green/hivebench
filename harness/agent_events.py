"""Activity-event helpers — the ``model`` event + parent ids (HIVE-PLAN §B3).

``harness.agent._shape_notification`` already emits ``assistant`` / ``tool`` /
``lifecycle`` activity events.  This module adds the fourth type, ``model``,
which wraps a tier dispatch, and the parent-id helpers that nest a worker's own
``tool`` events under it::

    {"type": "model", "tier": "worker", "model": "Qwen3.8-4B-Distill",
     "parent": null, "id": "m-7", "phase": "start|end",
     "task": "...", "duration_ms": 1812, "output": "..."}

Attribution flows from the nested loop (ADR-L6): the dispatcher emits a
``model`` start/end pair and calls :func:`with_parent` on the worker's tool
events.  Nothing here infers a parent from timing.

Frozen at T34 (ADR-L8).  T39 implements it; T40 (nested loop) and T42 (cards)
consume it.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

#: Activity-event ``type`` values (the three existing ones + ``model``).
EVENT_ASSISTANT = "assistant"
EVENT_TOOL = "tool"
EVENT_LIFECYCLE = "lifecycle"
EVENT_MODEL = "model"

#: ``phase`` values for a ``model`` event.
PHASE_START = "start"
PHASE_END = "end"
MODEL_PHASES: tuple[str, ...] = (PHASE_START, PHASE_END)

#: Key carrying a child event's parent ``model`` id.
PARENT_KEY = "parent"

#: Prefix for generated ``model`` ids (§B3 example: ``"m-7"``).
MODEL_ID_PREFIX = "m-"


def new_model_id(seq: int) -> str:
    """Deterministic ``model`` id for sequence number ``seq`` (``7`` → ``"m-7"``)."""
    raise NotImplementedError


def model_event(
    *,
    tier: str,
    model: str,
    phase: str,
    id: Optional[str] = None,
    parent: Optional[str] = None,
    task: Optional[str] = None,
    duration_ms: Optional[int] = None,
    output: Optional[str] = None,
) -> dict[str, Any]:
    """Build one §B3 ``model`` event.

    ``phase`` is ``"start"`` or ``"end"``.  ``id`` defaults to a fresh unique
    id (``m-<n>``); the ``end`` event of a span must reuse the ``start`` id.
    ``tier`` is the tier role (``"face"``, ``"worker"``, ...); ``parent`` is
    ``None`` for a face-tier dispatch.
    """
    raise NotImplementedError


def parent_id(event: Mapping[str, Any]) -> Optional[str]:
    """The parent id an event carries, or ``None``."""
    raise NotImplementedError


def with_parent(event: Mapping[str, Any], parent: Optional[str]) -> dict[str, Any]:
    """A copy of ``event`` with ``parent`` set (does not mutate ``event``).

    ``parent=None`` returns a copy with no parent key; a worker tool event
    carries the id of the ``model`` start event that dispatched it.
    """
    raise NotImplementedError


def is_model_event(event: Mapping[str, Any]) -> bool:
    """Whether ``event`` is a ``model`` event."""
    raise NotImplementedError


def is_tool_event(event: Mapping[str, Any]) -> bool:
    """Whether ``event`` is a ``tool`` event (a potential model child)."""
    raise NotImplementedError
