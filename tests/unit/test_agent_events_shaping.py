"""Unit tests for the §B3 activity shaping over dsh notifications (T48).

Offline: notifications are tiny objects with the SDK's ``method``/``payload``
shape, built from the fork's protocol types (``session.event`` carrying a
session-log event, plus ``subagent.started``/``finished``).  The shaper is the
piece that turns dsh-native delegation into the ``model`` spans and parented
``tool`` events the invocation cards render.
"""

from __future__ import annotations

from harness.agent import ActivityShaper


class _N:
    """A dsh notification, duck-typed like the SDK's model."""

    def __init__(self, method: str, payload: dict) -> None:
        self.method = method
        self.payload = payload


def _event(session_id: str, kind: str, data: dict | None = None) -> _N:
    return _N("session.event", {
        "sessionId": session_id,
        "event": {"type": kind, "data": data or {}},
    })


def _started(parent: str, child: str) -> _N:
    return _N("subagent.started", {"parentSessionId": parent, "childSessionId": child})


def _finished(child: str, *, output: str = "") -> _N:
    payload = {"parentSessionId": "root", "childSessionId": child}
    if output:
        payload["lastAssistantMessage"] = [{"type": "text", "text": output}]
    return _N("subagent.finished", payload)


def _collect() -> tuple[ActivityShaper, list[dict]]:
    events: list[dict] = []
    return ActivityShaper("root", events.append), events


def _tool_call(session_id: str, name: str, arguments: str = "{}") -> _N:
    return _event(session_id, "tool/call", {
        "callId": "c1", "name": name, "arguments": arguments,
    })


# ---------------------------------------------------------------------------
# face events are unchanged
# ---------------------------------------------------------------------------


def test_face_events_keep_the_existing_shape():
    shaper, events = _collect()

    shaper(_event("root", "assistant/message", {
        "message": {"content": [{"type": "text", "text": "hello"}]},
    }))
    shaper(_event("root", "tool/call", {"name": "bash"}))
    shaper(_event("root", "turn/start"))

    assert events[0] == {"type": "assistant", "text": "hello"}
    assert events[1] == {"type": "tool", "tool": "bash", "phase": "call"}
    assert events[2] == {"type": "lifecycle", "event": "turn/start"}


# ---------------------------------------------------------------------------
# a delegated dispatch becomes one model span with nested tools
# ---------------------------------------------------------------------------


def test_dispatch_becomes_model_span_with_parented_tools():
    shaper, events = _collect()

    shaper(_started("root", "child-1"))
    assert events == []  # the tier is not known until the child's request

    shaper(_event("child-1", "user/message", {
        "message": {"content": [{"type": "text", "text": "summarise the thread"}]},
    }))
    shaper(_event("child-1", "request/context", {
        "provider": "tier-worker", "model": "Qwen3.5-4B-UD-Q4_K_XL",
    }))
    shaper(_tool_call("child-1", "read_file", '{"path": "a.md"}'))
    shaper(_event("child-1", "tool/result", {
        "name": "read_file",
        "message": {"content": [{"type": "text", "text": "contents"}]},
    }))
    shaper(_finished("child-1", output="final answer"))

    start = events[0]
    assert start["type"] == "model" and start["phase"] == "start"
    assert start["tier"] == "worker"
    assert start["model"] == "Qwen3.5-4B-UD-Q4_K_XL"
    assert start["task"] == "summarise the thread"
    model_id = start["id"]
    assert model_id.startswith("m-")

    call, result = events[1], events[2]
    assert call["type"] == "tool" and call["phase"] == "call"
    assert call["tool"] == "read_file" and call["parent"] == model_id
    assert call["args"] == {"path": "a.md"}
    assert result["phase"] == "result" and result["parent"] == model_id
    assert result["result"] == "contents"

    end = events[3]
    assert end["type"] == "model" and end["phase"] == "end"
    assert end["id"] == model_id and end["output"] == "final answer"
    assert isinstance(end["duration_ms"], int)


def test_events_before_the_tier_is_known_are_buffered_then_parented():
    shaper, events = _collect()

    shaper(_started("root", "child-2"))
    shaper(_tool_call("child-2", "search", '{"q": "x"}'))
    assert events == []  # buffered until the parent id exists

    shaper(_event("child-2", "request/header", {
        "header": {"config": {"provider": "tier-agency", "model": "Ornith-1.5-9B"}},
    }))

    start, tool = events[0], events[1]
    assert start["phase"] == "start" and start["tier"] == "agency"
    assert tool["parent"] == start["id"] and tool["tool"] == "search"


def test_request_header_config_is_a_valid_resolution():
    shaper, events = _collect()
    shaper(_started("root", "child-3"))

    shaper(_event("child-3", "request/header", {
        "header": {"config": {"provider": "tier-mechanics", "model": "Qwen3.8-2B"}},
    }))
    shaper(_finished("child-3"))

    assert [e["phase"] for e in events] == ["start", "end"]
    assert events[0]["tier"] == "mechanics"


def test_child_assistant_text_is_not_flooded_but_becomes_the_output():
    shaper, events = _collect()
    shaper(_started("root", "child-4"))
    shaper(_event("child-4", "request/context", {"provider": "tier-worker", "model": "m"}))
    before = len(events)

    shaper(_event("child-4", "assistant/message", {
        "message": {"content": [{"type": "text", "text": "work in progress"}]},
    }))
    assert len(events) == before  # no assistant event for child turns

    shaper(_finished("child-4"))
    assert events[-1]["output"] == "work in progress"


# ---------------------------------------------------------------------------
# edges
# ---------------------------------------------------------------------------


def test_unresolved_dispatch_still_emits_a_pair():
    shaper, events = _collect()

    shaper(_started("root", "child-5"))
    shaper(_finished("child-5"))

    assert [e["phase"] for e in events] == ["start", "end"]
    assert events[0]["tier"] == "unknown"


def test_foreign_and_unknown_sessions_are_ignored():
    shaper, events = _collect()

    shaper(_started("someone-else", "child-6"))
    shaper(_event("child-6", "request/context", {"provider": "tier-worker", "model": "m"}))
    shaper(_finished("child-6"))
    shaper(_event("unknown-session", "tool/call", {"name": "bash"}))

    assert events == []


def test_each_dispatch_gets_its_own_model_id():
    shaper, events = _collect()

    for child in ("c-a", "c-b"):
        shaper(_started("root", child))
        shaper(_event(child, "request/context", {"provider": "tier-worker", "model": "m"}))
        shaper(_finished(child))

    starts = [e["id"] for e in events if e["phase"] == "start"]
    assert len(starts) == 2 and starts[0] != starts[1]
