"""Integration: face tier → worker subagent dispatch (HIVE-PLAN §B4 T40, ADR-L2/L6).

The gate is offline and hermetic: no GGUF, no llama-server, no dsh runtime.
A fake OpenAI-compatible client stands in for each tier's endpoint, so the
assertions are about the wiring the module owns — the registration entry the
dsh ``subagents`` tool consumes, the nested loop, and the §B3 ``model`` event
pair with its parented child ``tool`` events (ADR-L6).

Acceptance (HIVE-PLAN §B4 T40) is
``test_face_calls_worker_subagent_and_receives_its_text``: the face model, given
a retrieval task, calls the worker subagent and the worker's returned text
reaches the face's context.
"""

from __future__ import annotations

import json

from harness import agent_subagents
from harness.agent_subagents import (
    MAX_STEPS,
    WorkerSubagent,
    register_worker_subagents,
    run_nested_loop,
    subagent_description,
    subagent_name,
)
from harness.stack.schema import ROLE_FACE, ROLE_MECHANICS, ROLE_WORKER, Stack, Tier

# The §B3 ``stacks/peer-2tier.json`` card, verbatim.
FACE_TIER = Tier(
    role=ROLE_FACE,
    repo="unsloth/Qwen3.8-27B-GGUF",
    file="Qwen3.8-27B-UD-Q6_K.gguf",
    ctx=262144,
)
WORKER_TIER = Tier(
    role=ROLE_WORKER,
    repo="empero-ai/Qwen3.8-4B-Distill",
    file="Qwen3.8-4B-Q6_K.gguf",
    ctx=131072,
)

# What a live ``StackManager.status()["tiers"]`` row looks like: no base_url,
# a port and a model name.
WORKER_ENDPOINT = {"role": ROLE_WORKER, "key": "worker", "port": 8082,
                   "ctx": 131072, "model": "Qwen3.8-4B-Distill", "resident": True}
FACE_ENDPOINT = {"role": ROLE_FACE, "key": "face", "port": 8081,
                 "ctx": 262144, "model": "Qwen3.8-27B", "resident": True}


# --------------------------------------------------------------------------
# fakes: minimal OpenAI-compatible endpoints, object *or* dict style
# --------------------------------------------------------------------------


class FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class FakeChoice:
    def __init__(self, message):
        self.message = message


class FakeResponse:
    def __init__(self, message):
        self.choices = [FakeChoice(message)]


class FakeCompletions:
    """Returns scripted turns in order; records every ``create`` call."""

    def __init__(self, turns):
        self._turns = list(turns)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._turns:
            raise AssertionError("worker asked for more turns than scripted")
        return FakeResponse(self._turns.pop(0))


class FakeChat:
    def __init__(self, turns):
        self.completions = FakeCompletions(turns)


class FakeClient:
    """An OpenAI-compatible client: ``.chat.completions`` + host tool loop."""

    def __init__(self, turns, tools=None, run_tool=None):
        self.chat = FakeChat(turns)
        if tools:
            self.tools = tools
        if run_tool is not None:
            self.run_tool = run_tool


def tool_call(call_id, name, arguments):
    class _Fn:
        pass

    fn = _Fn()
    fn.name = name
    fn.arguments = json.dumps(arguments)
    call = _Fn()
    call.id = call_id
    call.type = "function"
    call.function = fn
    return call


# --------------------------------------------------------------------------
# naming + registration
# --------------------------------------------------------------------------


def test_subagent_name_is_role_plus_model_slug():
    assert subagent_name(WORKER_TIER) == "worker-qwen3-8-4b-distill"
    assert subagent_name(FACE_TIER) == "face-qwen3-8-27b-gguf"
    # never collides across roles
    assert subagent_name(FACE_TIER) != subagent_name(WORKER_TIER)
    assert " " not in subagent_name(WORKER_TIER)


def test_subagent_description_is_generated_from_role_and_model():
    desc = subagent_description(WORKER_TIER)
    assert ROLE_WORKER in desc
    assert "empero-ai/Qwen3.8-4B-Distill" in desc
    assert "131072" in desc
    assert "ADR-L2" in desc  # points the face model at the subagent mechanism
    assert subagent_description(FACE_TIER) != desc


def test_register_builds_one_subagent_per_live_non_face_tier():
    stack = Stack(name="peer-2tier", tiers=[FACE_TIER, WORKER_TIER])
    agents = register_worker_subagents(stack, {ROLE_FACE: FACE_ENDPOINT,
                                              ROLE_WORKER: WORKER_ENDPOINT})
    assert [a.name for a in agents] == ["worker-qwen3-8-4b-distill"]
    agent = agents[0]
    assert agent.role == ROLE_WORKER
    assert agent.model == "Qwen3.8-4B-Distill"        # from the live row
    assert agent.base_url == "http://127.0.0.1:8082/v1"  # composed from the port
    assert agent.api_key == "lm-studio"
    spec = agent.to_spec()
    assert spec["name"] == agent.name
    assert spec["description"] == agent.description
    assert spec["role"] == ROLE_WORKER
    assert spec["model"] == "Qwen3.8-4B-Distill"
    assert spec["base_url"] == "http://127.0.0.1:8082/v1"
    assert set(spec) >= {"name", "description", "role", "model", "base_url"}


def test_register_skips_unresident_and_malformed_tiers_without_raising():
    mechanics = Tier(role=ROLE_MECHANICS, repo="local/tiny", file="tiny-Q8_0.gguf")
    stack = Stack(name="mixed", tiers=[FACE_TIER, WORKER_TIER, mechanics])
    # worker resident, mechanics unapplied, face never registers
    agents = register_worker_subagents(stack, {ROLE_WORKER: WORKER_ENDPOINT})
    assert [a.role for a in agents] == [ROLE_WORKER]
    # a row without base_url or port is skipped, not a 500
    assert register_worker_subagents(stack, {ROLE_WORKER: {"model": "x"},
                                             ROLE_MECHANICS: {"resident": True}}) == []
    # explicit base_url wins over the port
    explicit = register_worker_subagents(
        stack, {ROLE_WORKER: {**WORKER_ENDPOINT, "base_url": "http://127.0.0.1:9/v1/"}})
    assert explicit[0].base_url == "http://127.0.0.1:9/v1"
    # no endpoints at all — every non-face tier skipped, face excluded
    assert register_worker_subagents(stack, {}) == []


# --------------------------------------------------------------------------
# the nested loop (ADR-L6: attribution from the loop)
# --------------------------------------------------------------------------


def _worker_agent():
    return WorkerSubagent(
        name=subagent_name(WORKER_TIER),
        role=ROLE_WORKER,
        model="Qwen3.8-4B-Distill",
        base_url="http://127.0.0.1:8082/v1",
        description=subagent_description(WORKER_TIER),
    )


def test_nested_loop_returns_final_text_and_events_one_model_pair():
    client = FakeClient([FakeMessage(content="three call sites in agent.py")])
    events = []
    result = run_nested_loop(_worker_agent(), "where is the subagents tool defined?",
                             on_event=events.append, client=client)

    assert result["ok"] is True
    assert result["output"] == "three call sites in agent.py"
    assert result["model_id"].startswith("m-")
    assert result["events"] == len(events) == 2

    start, end = events
    assert (start["type"], start["phase"]) == ("model", "start")
    assert (end["type"], end["phase"]) == ("model", "end")
    assert start["id"] == end["id"] == result["model_id"]
    assert start.get("parent") is None  # a face-tier dispatch has no parent
    for event in events:
        assert event["tier"] == ROLE_WORKER
        assert event["model"] == "Qwen3.8-4B-Distill"
        assert event["task"] == "where is the subagents tool defined?"
    assert end["output"] == result["output"]
    assert isinstance(end["duration_ms"], int) and end["duration_ms"] >= 0

    # the task actually reached the worker endpoint
    sent = client.chat.completions.calls[0]
    assert sent["model"] == "Qwen3.8-4B-Distill"
    assert sent["messages"][0] == {
        "role": "user", "content": "where is the subagents tool defined?"}


def test_nested_loop_parents_worker_tool_events_to_the_model_id():
    calls = []

    def run_tool(name, arguments):
        calls.append((name, arguments))
        return "harness/agent.py:6"

    tools = [{"type": "function", "function": {"name": "grep", "parameters": {}}}]
    client = FakeClient(
        [FakeMessage(tool_calls=[tool_call("c1", "grep", {"q": "subagents"})]),
         FakeMessage(content="harness/agent.py:6 mentions the subagents tool")],
        tools=tools, run_tool=run_tool,
    )
    events = []
    result = run_nested_loop(_worker_agent(), "find it", on_event=events.append,
                             client=client)

    assert result["ok"] is True
    assert result["output"] == "harness/agent.py:6 mentions the subagents tool"
    assert calls == [("grep", {"q": "subagents"})]
    assert result["events"] == 4  # model start, tool call, tool result, model end

    tool_events = [e for e in events if e["type"] == "tool"]
    assert [e["phase"] for e in tool_events] == ["call", "result"]
    for event in tool_events:
        assert event["parent"] == result["model_id"]  # ADR-L6
        assert event["tier"] == ROLE_WORKER
    assert tool_events[0]["args"] == {"q": "subagents"}
    assert tool_events[1]["result"] == "harness/agent.py:6"

    # the tool result is fed back so the worker can answer from it
    second = client.chat.completions.calls[1]
    assert second["tools"] == tools
    assert second["messages"][-1] == {"role": "tool", "tool_call_id": "c1",
                                      "content": "harness/agent.py:6"}


def test_nested_loop_reports_failure_without_leaking_an_exception():
    class Broken:
        @property
        def chat(self):
            raise RuntimeError("connection refused")

    events = []
    result = run_nested_loop(_worker_agent(), "anything", on_event=events.append,
                             client=Broken())
    assert result["ok"] is False
    assert "connection refused" in result["output"]
    # the span is still closed: attribution survives a failed dispatch
    assert [e["phase"] for e in events] == ["start", "end"]
    assert events[0]["id"] == events[1]["id"] == result["model_id"]


def test_nested_loop_is_bounded_and_survives_a_silent_on_event_callback():
    class Swallowing:
        def __call__(self, _event):
            raise RuntimeError("consumer queue full")

    # an endless tool-calling worker is truncated, not left spinning
    spinning = FakeClient(
        [FakeMessage(tool_calls=[tool_call(f"c{i}", "grep", {"q": i})])
         for i in range(MAX_STEPS + 4)]
    )
    result = run_nested_loop(_worker_agent(), "loop", on_event=Swallowing(),
                             client=spinning)
    assert result["ok"] is True
    assert result["events"] == 2 * MAX_STEPS + 2  # span start/end + 2 per step
    assert "produced no final text" in result["output"]


def test_default_client_path_reports_a_dead_endpoint_instead_of_raising():
    """No injected client: the module builds its own and an unreachable worker
    tier degrades to ``ok=False`` (never takes the face turn down)."""
    unreachable = WorkerSubagent(
        name="worker-probe", role=ROLE_WORKER, model="Qwen3.8-4B-Distill",
        base_url="http://127.0.0.1:9/v1",  # discard port, nothing listens
        description=subagent_description(WORKER_TIER),
    )
    events = []
    result = run_nested_loop(unreachable, "retrieve", on_event=events.append)
    assert result["ok"] is False
    assert result["output"].startswith("[worker-probe dispatch failed:")
    assert [e["phase"] for e in events] == ["start", "end"]


def test_events_route_through_agent_events_once_t39_lands(monkeypatch):
    """Merge-day guard: with ``harness.agent_events`` implemented, T40's call
    sites must use it (no duplicate local shaping) and produce the same span."""
    calls = {"model_event": [], "with_parent": [], "new_model_id": []}

    def model_event(**fields):
        calls["model_event"].append(fields)
        return dict(fields, type="model")

    def with_parent(event, parent):
        calls["with_parent"].append((dict(event), parent))
        out = dict(event)
        out["parent"] = parent
        return out

    def new_model_id(seq):
        calls["new_model_id"].append(seq)
        return f"m-{seq}"

    monkeypatch.setattr(agent_subagents.agent_events, "model_event", model_event)
    monkeypatch.setattr(agent_subagents.agent_events, "with_parent", with_parent)
    monkeypatch.setattr(agent_subagents.agent_events, "new_model_id", new_model_id)

    client = FakeClient([FakeMessage(content="ok")])
    result = run_nested_loop(_worker_agent(), "task", client=client)

    assert result["ok"] is True and result["output"] == "ok"
    assert len(calls["model_event"]) == 2  # the start/end pair, no local shaping
    assert calls["model_event"][0]["phase"] == "start"
    assert calls["model_event"][1]["phase"] == "end"
    assert calls["model_event"][0]["id"] == calls["model_event"][1]["id"] \
        == result["model_id"]
    assert calls["new_model_id"] == [int(result["model_id"].removeprefix("m-"))]


# --------------------------------------------------------------------------
# acceptance: the face calls the worker, and the worker's text reaches it
# --------------------------------------------------------------------------


def test_face_calls_worker_subagent_and_receives_its_text():
    """§B4 T40: face model + retrieval task → worker subagent → text in face context."""
    retrieved = "ADR-L2 registers each worker tier as a dsh subagent."

    stack = Stack(name="peer-2tier", tiers=[FACE_TIER, WORKER_TIER])
    registered = register_worker_subagents(
        stack, {ROLE_FACE: FACE_ENDPOINT, ROLE_WORKER: WORKER_ENDPOINT},
        api_key="lm-studio")
    assert len(registered) == 1
    spec = registered[0].to_spec()

    # The worker tier's own endpoint: a retrieval turn, answered in one step.
    worker_client = FakeClient([FakeMessage(content=retrieved)])
    # The face tier: turn 1 delegates via the registered subagent tool, turn 2
    # answers from the tool result it was given.
    face_turns = [
        FakeMessage(tool_calls=[tool_call("s1", spec["name"],
                                          {"task": "what does ADR-L2 say?"})]),
        FakeMessage(content=f"ADR-L2: {retrieved}"),
    ]
    face_client = FakeClient(face_turns)

    def host_dispatch(name, arguments, call):
        """What the dsh ``subagents`` tool does with a registered entry."""
        assert name == spec["name"]  # the face called the subagent by its name
        assert arguments["task"]
        return run_nested_loop(registered[0], arguments["task"],
                               on_event=call, client=worker_client)

    events = []
    face_messages = [{"role": "user", "content": "what does ADR-L2 say?"}]
    face_tools = [{"type": "function", "function": {"name": spec["name"],
                                                    "description": spec["description"]}}]
    answer = ""
    for _ in range(MAX_STEPS):
        sent = face_client.chat.completions.create(model="Qwen3.8-27B",
                                                   messages=face_messages,
                                                   tools=face_tools)
        message = sent.choices[0].message
        if not message.tool_calls:
            answer = message.content
            break
        face_messages.append({"role": "assistant", "content": message.content,
                              "tool_calls": message.tool_calls})
        for call in message.tool_calls:
            result = host_dispatch(call.function.name,
                                   json.loads(call.function.arguments), events.append)
            face_messages.append({"role": "tool", "tool_call_id": call.id,
                                  "content": result["output"]})

    # 1. the face model called the worker subagent
    first_call = face_client.chat.completions.calls[0]
    assert [t["function"]["name"] for t in first_call["tools"]] == [spec["name"]]
    assert face_client.chat.completions.calls[1]["messages"][-1]["role"] == "tool"
    # 2. the worker's returned text reached the face's context and was used
    tool_result = face_messages[-1]
    assert tool_result["content"] == retrieved
    assert retrieved in answer
    # 3. the dispatch was attributed: one model span, parented tool events
    span = [e for e in events if e["type"] == "model"]
    assert [e["phase"] for e in span] == ["start", "end"]
    assert span[0]["id"] == span[1]["id"]
    assert span[1]["output"] == retrieved
    assert span[1]["tier"] == ROLE_WORKER
    # the worker's request carried the retrieval task, not the face's context
    assert worker_client.chat.completions.calls[0]["messages"] == [
        {"role": "user", "content": "what does ADR-L2 say?"}]
