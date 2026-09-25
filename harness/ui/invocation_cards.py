"""Invocation cards — collapsed ``model`` cards with nested tool children (§B3).

Hermes / Antigravity pattern applied locally: every tier dispatch rendered as a
collapsed header (``tier · model · duration · status``) that expands to
``task → tool calls → output``.  Cards come from the shaped activity events in
:mod:`harness.agent_events`: a ``model`` event is a card, and the ``tool``
events whose ``parent`` equals its ``id`` nest underneath.  Events with no
model parent (face ``assistant`` / ``lifecycle`` / bare ``tool``) stay
top-level groups.

Pure renderers: take fixture events, return an HTML fragment string; dynamic
values are HTML-escaped.  ``css()`` returns the sibling ``invocation_cards.css``.

Frozen at T34 (ADR-L8).  T42 implements it (plus ``invocation_cards.css``);
T44 mounts the fragment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

#: Sibling stylesheet read by :func:`css`.
CSS_NAME = "invocation_cards.css"


@dataclass
class CardGroup:
    """One rendered card: a top-level event plus its nested children.

    ``event`` is the ``model`` / ``assistant`` / ``lifecycle`` event; for a
    ``model`` event ``children`` are the ``tool`` events parented to it (and
    may themselves nest, recursion is the renderer's choice).
    """

    event: dict[str, Any]
    children: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError


def css() -> str:
    """Contents of the sibling ``invocation_cards.css`` (re-read per call)."""
    raise NotImplementedError


def group_events(events: Sequence[Mapping[str, Any]]) -> list[CardGroup]:
    """Nest child events under their ``model`` parent by id, preserving order.

    A child is any event whose ``parent`` matches an earlier ``model`` event's
    ``id``; unmatched children stay top-level.  Reads ``id``/``parent``
    directly so it does not depend on the T39 implementation.
    """
    raise NotImplementedError


def render_invocation_card(group: CardGroup) -> str:
    """One collapsed card, expandable to task → tool children → output."""
    raise NotImplementedError


def render_invocation_cards(events: Sequence[Mapping[str, Any]] = ()) -> str:
    """The invocation-card list for a conversation's activity events."""
    raise NotImplementedError
