"""Agent activity events — the §B3 ``model`` type + parent-id helpers (T39).

Covers the frozen T34 contract in ``harness/agent_events.py``: building a
``model`` event with all nine §B3 fields, deterministic/unique ids, parent-id
reads, a non-mutating ``with_parent`` copy, and the type predicates.  The
dispatch acceptance is a nested-loop simulation (the T40 shape): exactly one
``model`` start/end pair whose id parents every worker ``tool`` event.
"""

from types import MappingProxyType

import pytest

from harness.agent_events import (
    EVENT_ASSISTANT,
    EVENT_LIFECYCLE,
    EVENT_MODEL,
    EVENT_TOOL,
    MODEL_ID_PREFIX,
    PARENT_KEY,
    PHASE_END,
    PHASE_START,
    is_model_event,
    is_tool_event,
    model_event,
    new_model_id,
    parent_id,
    with_parent,
)


# ---------------------------------------------------------------------------
# new_model_id


def test_new_model_id_is_deterministic():
    assert new_model_id(7) == "m-7"
    assert new_model_id(0) == "m-0"
    assert new_model_id(42) == f"{MODEL_ID_PREFIX}42"


# ---------------------------------------------------------------------------
# model_event


def test_model_event_carries_all_nine_b3_fields():
    event = model_event(
        tier="worker",
        model="Qwen3.8-4B-Distill",
        phase=PHASE_START,
        task="summarize the page",
        duration_ms=1812,
        output="done",
    )

    assert set(event) == {
        "type", "tier", "model", "parent", "id", "phase",
        "task", "duration_ms", "output",
    }
    assert event["type"] == EVENT_MODEL
    assert event["tier"] == "worker"
    assert event["model"] == "Qwen3.8-4B-Distill"
    assert event["phase"] == PHASE_START
    assert event["task"] == "summarize the page"
    assert event["duration_ms"] == 1812
    assert event["output"] == "done"
    assert isinstance(event["id"], str) and event["id"].startswith(MODEL_ID_PREFIX)


def test_model_event_optional_fields_default_to_none():
    event = model_event(tier="face", model="Qwen3.8-27B", phase=PHASE_START)

    assert event["parent"] is None
    assert event["task"] is None
    assert event["duration_ms"] is None
    assert event["output"] is None


def test_model_event_end_reuses_start_id():
    start = model_event(tier="worker", model="m", phase=PHASE_START)
    end = model_event(
        tier="worker", model="m", phase=PHASE_END, id=start["id"],
        duration_ms=5,
    )

    assert start["id"] == end["id"]


def test_model_event_uses_explicit_id():
    event = model_event(tier="face", model="m", phase=PHASE_END, id="m-7")
    assert event["id"] == "m-7"


def test_model_event_auto_ids_are_fresh_and_unique():
    ids = [model_event(tier="face", model="m", phase=PHASE_START)["id"]
           for _ in range(50)]

    assert len(set(ids)) == 50
    assert all(i.startswith(MODEL_ID_PREFIX) for i in ids)


def test_model_event_rejects_unknown_phase():
    with pytest.raises(ValueError, match="phase"):
        model_event(tier="face", model="m", phase="middle")


# ---------------------------------------------------------------------------
# parent_id


def test_parent_id_reads_explicit_parent():
    event = model_event(
        tier="worker", model="m", phase=PHASE_START, parent="m-1",
    )
    assert parent_id(event) == "m-1"


def test_parent_id_defaults_to_none():
    assert parent_id({}) is None
    assert parent_id({PARENT_KEY: None}) is None
    assert parent_id(model_event(tier="face", model="m", phase=PHASE_START)) is None
    # non-string parents are not ids
    assert parent_id({PARENT_KEY: 7}) is None


# ---------------------------------------------------------------------------
# with_parent


def test_with_parent_sets_a_copy_without_mutating_input():
    tool = {"type": EVENT_TOOL, "tool": "search", "phase": "start"}
    original = dict(tool)

    child = with_parent(tool, "m-7")

    assert tool == original
    assert PARENT_KEY not in tool
    assert child is not tool
    assert child[PARENT_KEY] == "m-7"
    assert child["type"] == EVENT_TOOL
    assert parent_id(child) == "m-7"


def test_with_parent_none_removes_parent_key_without_mutating():
    tool = {"type": EVENT_TOOL, "tool": "search", PARENT_KEY: "m-1"}
    original = dict(tool)

    out = with_parent(tool, None)

    assert PARENT_KEY not in out
    assert tool == original  # input keeps its parent
    assert out is not tool


def test_with_parent_accepts_any_mapping_and_returns_dict():
    proxy = MappingProxyType({"type": EVENT_TOOL, "tool": "bash"})
    out = with_parent(proxy, "m-3")
    assert isinstance(out, dict)
    assert out == {"type": EVENT_TOOL, "tool": "bash", PARENT_KEY: "m-3"}


# ---------------------------------------------------------------------------
# predicates


def test_type_predicates():
    model = model_event(tier="face", model="m", phase=PHASE_START)

    assert is_model_event(model) is True
    assert is_tool_event(model) is False
    assert is_model_event({"type": EVENT_TOOL}) is False
    assert is_tool_event({"type": EVENT_TOOL}) is True
    assert is_model_event({"type": EVENT_ASSISTANT}) is False
    assert is_tool_event({"type": EVENT_ASSISTANT}) is False
    assert is_model_event({"type": EVENT_LIFECYCLE}) is False
    assert is_tool_event({"type": EVENT_LIFECYCLE}) is False
    assert is_model_event({}) is False
    assert is_tool_event({}) is False


# ---------------------------------------------------------------------------
# dispatch acceptance — one start/end pair; child tools carry the parent id


def test_dispatch_emits_one_model_pair_and_parents_child_tools():
    emitted = []
    start = model_event(
        tier="worker", model="Qwen3.8-4B-Distill",
        phase=PHASE_START, task="find the answer",
    )
    emitted.append(start)
    model_id = start["id"]

    # the worker's own tool events are tagged with the dispatch id (ADR-L6)
    for phase in ("start", "end"):
        emitted.append(with_parent(
            {"type": EVENT_TOOL, "tool": "search", "phase": phase}, model_id,
        ))

    emitted.append(model_event(
        tier="worker", model="Qwen3.8-4B-Distill",
        phase=PHASE_END, id=model_id, duration_ms=1812, output="the answer",
    ))

    models = [e for e in emitted if is_model_event(e)]
    starts = [e for e in models if e["phase"] == PHASE_START]
    ends = [e for e in models if e["phase"] == PHASE_END]

    assert len(models) == 2
    assert len(starts) == 1
    assert len(ends) == 1
    assert starts[0]["id"] == ends[0]["id"] == model_id

    tools = [e for e in emitted if is_tool_event(e)]
    assert len(tools) == 2
    assert all(parent_id(e) == model_id for e in tools)

    assert ends[0]["duration_ms"] == 1812
    assert ends[0]["output"] == "the answer"


def test_two_dispatches_have_distinct_model_ids():
    first = model_event(tier="worker", model="m", phase=PHASE_START)
    second = model_event(tier="worker", model="m", phase=PHASE_START)
    assert first["id"] != second["id"]
