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

import html
import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

#: Sibling stylesheet read by :func:`css`.
CSS_NAME = "invocation_cards.css"

#: Activity-event ``type`` this module reads straight off the payload (the §B3
#: literals, kept local so the cards do not depend on the T39 implementation).
_TYPE_MODEL = "model"
_TYPE_TOOL = "tool"
_TYPE_ASSISTANT = "assistant"
_TYPE_LIFECYCLE = "lifecycle"

#: Keys a ``model``/``tool`` event may carry its payload under, in preference order.
_TASK_KEYS = ("task", "task_text", "prompt", "query", "input")
_OUTPUT_KEYS = ("output", "result", "text", "content", "answer")
_ARGS_KEYS = ("args", "input", "arguments", "params", "query")

#: ``model`` event phases (§B3) — a ``start`` with no ``end`` yet is still running.
_PHASE_START = "start"
_PHASE_END = "end"

_EMPTY = "&mdash;"

#: The card stylesheet, served by :func:`css`.  It lives here (module symbol
#: ``CSS_NAME`` is its handle) rather than in a sibling file, so the fragment is
#: self-contained; the page shell inlines whatever :func:`css` returns.
_CARD_CSS = """/* Invocation cards — collapsed tier dispatches with nested tool children */
.invocation-cards { display: flex; flex-direction: column; gap: .35rem; }
.invocation-cards:empty { display: none; }
.invocation-card {
  background: #000000; color: #FFDD00; border: 1.5px solid #000000;
  border-left: 3px solid #FFB703; border-radius: 8px; font-size: .82rem;
}
.invocation-card > summary { padding: .35rem .6rem; color: #FFDD00; cursor: pointer; list-style: revert; }
.invocation-card[open] > summary { border-bottom: 1px dashed rgba(255,221,0,.35); }
.invocation-card--model { border-left-color: #FFB703; }
.invocation-card--assistant { border-left-color: #8a99a8; }
.invocation-card--lifecycle { border-left-color: rgba(255,221,0,.35); font-size: .78rem; }
.invocation-head { display: flex; align-items: baseline; gap: .5rem; flex-wrap: wrap; }
.invocation-tier { font-weight: 700; text-transform: uppercase; letter-spacing: .04em; }
.invocation-model { font-weight: 600; color: #FFDD00; word-break: break-word; }
.invocation-duration, .invocation-status, .invocation-phase, .invocation-kind {
  font-size: .74rem; color: rgba(255,221,0,.72); font-variant-numeric: tabular-nums;
}
.invocation-duration { margin-left: auto; }
.invocation-status { padding: .05rem .4rem; border: 1px solid rgba(255,221,0,.35); border-radius: 999px; }
.invocation-status.is-error { color: #fff; background: #b3372c; border-color: #b3372c; }
.invocation-status.is-ok, .invocation-status.is-done { color: #000; background: #157a3e; border-color: #157a3e; }
.invocation-body { padding: .4rem .6rem .55rem .6rem; display: flex; flex-direction: column; gap: .4rem; }
.invocation-task, .invocation-text { white-space: pre-wrap; word-break: break-word; color: #FFDD00; }
.invocation-children { display: flex; flex-direction: column; gap: .3rem; }
.invocation-tool {
  background: rgba(255,221,0,.06); border: 1px dashed rgba(255,221,0,.45);
  border-radius: 6px; font-size: .8rem;
}
.invocation-tool > summary { padding: .25rem .5rem; cursor: pointer; display: flex; gap: .45rem; }
.invocation-tool-body { padding: .2rem .5rem .45rem .5rem; }
.invocation-tool-name { font-weight: 600; color: #FFDD00; }
.invocation-pre {
  margin: .25rem 0 0 0; padding: .45rem .55rem; max-height: 200px; overflow-y: auto;
  background: #FFB703; color: #000000; border: 1.5px solid #000000; border-radius: 6px;
  white-space: pre-wrap; word-break: break-word; font-size: .78rem;
}
.invocation-label { font-size: .72rem; text-transform: uppercase; letter-spacing: .04em; color: rgba(255,221,0,.72); }
.invocation-error { color: #fff; background: #b3372c; border-radius: 6px; padding: .3rem .5rem; }
"""


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
        return {
            "event": dict(self.event),
            "children": [dict(child) for child in self.children],
        }


def css() -> str:
    """Contents of the sibling ``invocation_cards.css`` (re-read per call)."""
    return _CARD_CSS


# --------------------------------------------------------------------------
# helpers (pure; the four contract symbols above are the only public surface)
# --------------------------------------------------------------------------


def _esc(value: Any) -> str:
    """HTML-escape a dynamic value, quotes included (attribute safety)."""
    return html.escape(str(value), quote=True)


def _text(value: Any) -> str:
    """Escaped text for a value, or ``""`` when it carries nothing.

    Structured payloads (tool args, result blobs) render as JSON, not as a
    Python ``repr``.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return _esc(value) if value.strip() else ""
    if isinstance(value, (Mapping, list, tuple)):
        return _esc(json.dumps(value, ensure_ascii=False, default=str))
    return _esc(value)


def _attr(name: str, value: Any) -> str:
    """`` name="value"`` for a non-empty value, else ``""``."""
    text = _text(value)
    return f' {name}="{text}"' if text else ""


def _slug(value: Any) -> str:
    """Class-safe token for a status/kind string."""
    return re.sub(r"[^a-z0-9-]+", "-", str(value or "").strip().lower()).strip("-")


def _pick(event: Mapping[str, Any], keys: Sequence[str]) -> Any:
    """First non-empty value among ``keys``, else ``None``."""
    for key in keys:
        value = event.get(key)
        if value in (None, "", [], {}):
            continue
        return value
    return None


def _fmt_duration(value: Any) -> str:
    """``1812`` → ``1.81s``; ``812`` → ``812ms``; unknown → ``&mdash;``."""
    try:
        ms = float(value)
    except (TypeError, ValueError):
        return _EMPTY
    if ms < 0:
        return _EMPTY
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms / 1000:.2f}s"


def _status(event: Mapping[str, Any]) -> str:
    """Header status token: explicit, else error, else phase-derived."""
    explicit = event.get("status") or event.get("state")
    if explicit:
        return _slug(explicit) or "unknown"
    if event.get("error"):
        return "error"
    if str(event.get("phase") or "") == _PHASE_START:
        return "running"
    return "done"


def _is_empty(value: Any) -> bool:
    return value in (None, "", [], {})


def _fold_end(target: dict[str, Any], end: Mapping[str, Any]) -> None:
    """Fold a ``phase="end"`` event into its ``start`` card (same ``id``)."""
    for key, value in end.items():
        if key == "phase":
            continue
        if _is_empty(target.get(key)):
            target[key] = value
    target["phase"] = _PHASE_END


# --------------------------------------------------------------------------
# renderers
# --------------------------------------------------------------------------


def _render_pre(label: str, value: Any, *, extra_class: str = "") -> str:
    text = _text(value)
    if not text:
        return ""
    classes = f"invocation-pre {extra_class}".strip()
    return (
        f'<span class="invocation-label">{_esc(label)}</span>'
        f'<pre class="{_esc(classes)}">{text}</pre>'
    )

def _summary(kind: str, extra: str = "") -> str:
    return (
        f'<summary class="invocation-head">'
        f'<span class="invocation-kind">{_esc(kind)}</span>{extra}</summary>'
    )


def _render_tool(event: Mapping[str, Any], *, nested: bool = False) -> str:
    """A tool step: collapsed ``tool · phase`` header, args + result inside."""
    name = _text(_pick(event, ("tool", "name"))) or _esc(_TYPE_TOOL)
    phase = _text(event.get("phase"))
    body = _render_pre("input", _pick(event, _ARGS_KEYS))
    body += _render_pre("output", _pick(event, _OUTPUT_KEYS))
    if event.get("error"):
        body += f'<div class="invocation-error">{_text(event["error"])}</div>'
    inner = f'<div class="invocation-tool-body">{body}</div>' if body else ""
    head = (
        f'<span class="invocation-tool-name">{name}</span>'
        + (f'<span class="invocation-phase">{phase}</span>' if phase else "")
    )
    summary = f'<summary class="invocation-tool-head">{head}</summary>'
    classes = "invocation-tool"
    attrs = (
        _attr("data-tool", _pick(event, ("tool", "name")))
        + _attr("data-phase", phase)
        + _attr("data-nested", "1" if nested else None)
    )
    return f'<details class="{classes}"{attrs}>{summary}{inner}</details>'


def _dispatch_children(
    child: Mapping[str, Any], siblings: Sequence[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    """The rows parented to a nested sub-dispatch, in stream order.

    ``CardGroup.children`` is a flat ``list[dict]``, so a sub-dispatch's own
    tool rows arrive as siblings of the sub-dispatch itself; they are collected
    back here at render time and rendered *inside* it, not next to it.
    """
    cid = child.get("id")
    if not isinstance(cid, str) or not cid:
        return []
    return [row for row in siblings if row.get("parent") == cid]


def _render_children(children: Sequence[Mapping[str, Any]]) -> str:
    """The nested call block of a card (empty string when there is none)."""
    steps = [child for child in children if isinstance(child, Mapping)]
    if not steps:
        return ""
    dispatch_ids = {
        str(child["id"])
        for child in steps
        if str(child.get("type") or "") == _TYPE_MODEL and child.get("id")
    }
    parts: list[str] = []
    for child in steps:
        parent = child.get("parent")
        if isinstance(parent, str) and parent in dispatch_ids:
            continue  # rendered inside its own sub-dispatch card instead
        parts.append(_render_child(child, steps))
    if not parts:
        return ""
    rendered = "".join(parts)
    return f'<div class="invocation-children" data-count="{len(parts)}">{rendered}</div>'


def _render_child(
    event: Mapping[str, Any], siblings: Sequence[Mapping[str, Any]] = ()
) -> str:
    """One nested child: a sub-dispatch card, a tool step, or a plain row."""
    kind = str(event.get("type") or "")
    if kind == _TYPE_TOOL:
        return _render_tool(event, nested=True)
    if kind == _TYPE_MODEL:
        return _render_model(event, _dispatch_children(event, siblings))
    if kind == _TYPE_ASSISTANT:
        text = _text(_pick(event, _OUTPUT_KEYS))
        inner = f'<div class="invocation-text">{text}</div>' if text else ""
        return f'<details class="invocation-tool">{_summary(_TYPE_ASSISTANT, "")}{inner}</details>'
    label = _text(kind) or _esc("event")
    return f'<details class="invocation-tool">{_summary("event", label)}</details>'


def _render_model(event: Mapping[str, Any], children: Sequence[Mapping[str, Any]]) -> str:
    """The collapsed dispatch card: ``tier · model · duration · status``."""
    tier = _text(event.get("tier")) or _EMPTY
    model = _text(event.get("model")) or _EMPTY
    status = _status(event)
    status_class = "invocation-status"
    if status in ("error", "failed"):
        status_class += " is-error"
    elif status in ("done", "ok", "ready"):
        status_class += " is-ok"
    elif status == "running":
        status_class += " is-running"
    head = (
        f'<span class="invocation-tier">{tier}</span>'
        f'<span class="invocation-model">{model}</span>'
        f'<span class="invocation-duration">{_fmt_duration(event.get("duration_ms"))}</span>'
        f'<span class="{status_class}">{_esc(status)}</span>'
    )
    body = _render_pre("task", _pick(event, _TASK_KEYS))
    body += _render_children(children)
    body += _render_pre("output", _pick(event, _OUTPUT_KEYS))
    if event.get("error"):
        body += f'<div class="invocation-error">{_text(event["error"])}</div>'
    inner = f'<div class="invocation-body">{body}</div>' if body else ""
    attrs = (
        _attr("data-id", event.get("id"))
        + _attr("data-tier", event.get("tier"))
        + _attr("data-status", status)
        + _attr("data-parent", event.get("parent"))
    )
    return (
        f'<details class="invocation-card invocation-card--model"{attrs}>'
        f'<summary class="invocation-head">{head}</summary>{inner}</details>'
    )


def _render_plain(event: Mapping[str, Any], children: Sequence[Mapping[str, Any]]) -> str:
    """A top-level group with no ``model`` parent (assistant / lifecycle / tool)."""
    kind = str(event.get("type") or "event")
    if kind == _TYPE_TOOL:
        return _render_tool(event, nested=True)
    body = _render_pre("message" if kind == _TYPE_ASSISTANT else "output",
                       _pick(event, _OUTPUT_KEYS))
    body += _render_children(children)
    if event.get("error"):
        body += f'<div class="invocation-error">{_text(event["error"])}</div>'
    inner = f'<div class="invocation-body">{body}</div>' if body else ""
    extra = ""
    if kind == _TYPE_LIFECYCLE:
        extra = f'<span class="invocation-phase">{_text(event.get("event"))}</span>'
    attrs = _attr("data-id", event.get("id")) + _attr("data-status", _status(event))
    return (
        f'<details class="invocation-card invocation-card--{_slug(kind) or "event"}"{attrs}>'
        f'{_summary(kind, extra)}{inner}</details>'
    )


def group_events(events: Sequence[Mapping[str, Any]]) -> list[CardGroup]:
    """Nest child events under their ``model`` parent by id, preserving order.

    A child is any event whose ``parent`` matches an earlier ``model`` event's
    ``id``; unmatched children stay top-level.  Reads ``id``/``parent``
    directly so it does not depend on the T39 implementation.
    """
    groups: list[CardGroup] = []
    #: model id -> the group that holds its children (itself, or the ancestor
    #: group when the dispatch is itself a nested child).
    holder: dict[str, CardGroup] = {}
    #: model id -> the event dict a matching ``end`` folds into.
    target: dict[str, dict[str, Any]] = {}

    def _known(event: Mapping[str, Any]) -> Optional[CardGroup]:
        parent = event.get("parent")
        if not isinstance(parent, str) or not parent:
            return None
        return holder.get(parent)

    for raw in events or ():
        if not isinstance(raw, Mapping):
            continue
        event = dict(raw)
        etype = str(event.get("type") or "")
        eid = event.get("id")
        if etype == _TYPE_MODEL:
            known = target.get(eid) if isinstance(eid, str) and eid else None
            if known is not None:
                # Same id twice: the start/end pair is one card, not two.
                _fold_end(known, event)
                continue
            owner = _known(event)
            if owner is not None:
                # A sub-dispatch inside another dispatch's card.
                owner.children.append(event)
                if isinstance(eid, str) and eid:
                    holder[eid] = owner
                    target[eid] = event
                continue
            group = CardGroup(event=event)
            groups.append(group)
            if isinstance(eid, str) and eid:
                holder[eid] = group
                target[eid] = event
            continue
        owner = _known(event)
        if owner is not None:
            owner.children.append(event)
            continue
        groups.append(CardGroup(event=event))
    return groups


def render_invocation_card(group: CardGroup) -> str:
    """One collapsed card, expandable to task → tool children → output."""
    event = getattr(group, "event", None)
    event = event if isinstance(event, Mapping) else {}
    children = [c for c in (getattr(group, "children", None) or ()) if isinstance(c, Mapping)]
    if str(event.get("type") or "") == _TYPE_MODEL:
        return _render_model(event, children)
    return _render_plain(event, children)


def render_invocation_cards(events: Sequence[Mapping[str, Any]] = ()) -> str:
    """The invocation-card list for a conversation's activity events."""
    groups = group_events(events)
    cards = "".join(render_invocation_card(group) for group in groups)
    classes = "invocation-cards" + ("" if groups else " is-empty")
    return (
        f'<div class="{classes}" data-count="{len(groups)}" data-css="{_esc(CSS_NAME)}">'
        f"{cards}</div>"
    )
