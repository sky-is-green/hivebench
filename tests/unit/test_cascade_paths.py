"""Unit tests for harness.cascade paths and escalation policy."""

from __future__ import annotations

import pytest

from harness.cascade.judgement import JudgementResult, JudgementState
from harness.cascade.paths import (
    ESCALATION_TRIGGERS,
    PATHS,
    EscalationTrigger,
    async_steps,
    blocking_steps,
    get_path,
    paths_for_bucket,
    resolve_steps,
    validate_paths,
)
from harness.cascade.policy import (
    Policy,
    budget_says_halt,
    escalation_for,
    escalation_probability,
    route_needs_escalation,
    should_escalate,
    streamable_tokens,
    verifier_rejects,
)


def test_path_registry_is_structurally_valid():
    assert validate_paths() == []


def test_route_table_covers_every_bucket():
    from harness.cascade.paths import ROUTE_TABLE

    assert ROUTE_TABLE["qa"] == "P2"
    assert ROUTE_TABLE["easy"] == "P1"
    assert ROUTE_TABLE["code"] == "P4"
    assert ROUTE_TABLE["tool"] == "P4"
    assert ROUTE_TABLE["reasoning"] == "P5"
    assert ROUTE_TABLE["vision"] == "P6"
    assert ROUTE_TABLE["cache"] == "P0"


def test_paths_for_bucket_and_get_path():
    assert [p.id for p in paths_for_bucket("qa")] == ["P2"]
    assert paths_for_bucket("nope") == ()
    with pytest.raises(KeyError):
        get_path("P9")


def test_p3_extends_p2():
    p2 = get_path("P2")
    p3 = get_path("P3")
    assert resolve_steps(p3) == resolve_steps(p2) + p3.steps
    assert p3.extends == "P2"


def test_async_and_blocking_partition():
    p2 = get_path("P2")
    assert {s.role for s in async_steps(p2)} == {"C1", "D1"}
    assert {s.role for s in blocking_steps(p2)} == {"C2", "B1", "B2", "E2", "D2"}
    assert [s.role for s in blocking_steps(get_path("P1"))] == ["E1"]
    assert get_path("P4").loop is True


def test_every_path_escalation_target_exists():
    known = {p.id for p in PATHS}
    for path in PATHS:
        assert set(path.escalates_to) <= known


def test_escalation_trigger_set():
    assert len(ESCALATION_TRIGGERS) == 4
    assert EscalationTrigger.VERIFIER_REJECT in ESCALATION_TRIGGERS


def _result(decision: str, confidence: float, *, state=JudgementState.DONE):
    return JudgementResult(
        judgement_id="j1", request_id="r1", generation_id=0, role="D2",
        decision=decision, confidence=confidence, state=state,
        error=None if state is JudgementState.DONE else "boom",
    )


def test_policy_thresholds():
    policy = Policy()
    assert route_needs_escalation(0.4, policy)
    assert not route_needs_escalation(0.5, policy)
    assert verifier_rejects(_result("reject", 0.99), policy)
    assert verifier_rejects(_result("accept", 0.1), policy)
    assert not verifier_rejects(_result("accept", 0.6), policy)
    assert not verifier_rejects(_result("accept", 0.0, state=JudgementState.ERROR), policy)
    assert budget_says_halt(_result("stop", 0.8), policy)
    assert not budget_says_halt(_result("continue", 0.8), policy)


def test_escalation_choice_priority_and_budget():
    triggers = [EscalationTrigger.ROUTE_LOW_CONFIDENCE, EscalationTrigger.VERIFIER_REJECT]
    assert should_escalate(triggers, Policy()) is EscalationTrigger.VERIFIER_REJECT
    assert should_escalate(triggers, Policy(), escalations_used=2) is None
    assert should_escalate([], Policy()) is None


def test_escalation_target_and_probability():
    assert escalation_for(EscalationTrigger.VERIFIER_REJECT, get_path("P2")) == "P3"
    assert escalation_for(EscalationTrigger.ROUTE_LOW_CONFIDENCE, get_path("P1")) == "P3"
    assert escalation_probability(1.0) == 0.0
    assert escalation_probability(0.0) == 1.0
    assert escalation_probability(0.8) == pytest.approx(0.2)


def test_streamable_tokens_window():
    policy = Policy(verify_window=24)
    assert streamable_tokens(100, 90, policy) == 90
    assert streamable_tokens(100, 0, policy) == 76
    assert streamable_tokens(10, 0, policy) == 0
    assert streamable_tokens(10, 0, Policy(verify_window=0)) == 10
    assert streamable_tokens(0, 0, policy) == 0
    assert streamable_tokens(5, 99, policy) == 5  # verified clamps to generated


def test_policy_round_trips():
    data = Policy().to_dict()
    assert data["verify_window"] == 24
    assert data["max_escalations"] == 2
