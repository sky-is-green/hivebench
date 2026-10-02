"""Escalation policy and the verify window.

Tunable policy, not model logic (brief §"Harness vs model vs training"): the
thresholds, the window that is held under verification, the reasoning-budget
cadence and the escalation budget are all harness configuration.  Calibration
and threshold fitting live offline and write these values back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from harness.cascade.judgement import JudgementResult
from harness.cascade.paths import EscalationTrigger, Path


@dataclass(frozen=True)
class Policy:
    """Cascade tunables (defaults are the brief's planning values).

    ``quality_target`` is the per-request task-metric floor used when the
    oracle provides per-path quality; ``route_confidence`` and
    ``accept_confidence`` are the calibrated-probability thresholds for
    escalation and acceptance.  ``verify_window`` is the maximum number of
    generated-but-unverified tokens held in memory; ``budget_check_every`` is
    H1's cadence in thinking tokens.
    """

    quality_target: float = 0.90
    route_confidence: float = 0.50
    accept_confidence: float = 0.40
    verify_window: int = 24
    max_escalations: int = 2
    budget_check_every: int = 256
    draft_accept_payback: float = 0.40

    def to_dict(self) -> dict[str, Any]:
        return {
            "quality_target": self.quality_target,
            "route_confidence": self.route_confidence,
            "accept_confidence": self.accept_confidence,
            "verify_window": self.verify_window,
            "max_escalations": self.max_escalations,
            "budget_check_every": self.budget_check_every,
            "draft_accept_payback": self.draft_accept_payback,
        }


def route_needs_escalation(route_confidence: float, policy: Policy) -> bool:
    """True when C1's confidence is below the escalation threshold."""
    return route_confidence < policy.route_confidence


def verifier_rejects(result: JudgementResult, policy: Policy) -> bool:
    """True when an accept/reject judgement falls below the accept threshold.

    Errored/dropped judgements never reject: the pipeline degrades open and
    lets the next verifier or the escalation policy decide.
    """
    if not result.ok:
        return False
    if result.decision in ("reject", "refuse"):
        return True
    return result.confidence < policy.accept_confidence


def budget_says_halt(result: JudgementResult, policy: Policy) -> bool:
    """H1's continue/stop verdict (``"stop"`` halts the thinking trace)."""
    if not result.ok:
        return False
    return result.decision == "stop"


def escalation_for(trigger: EscalationTrigger, path: Path) -> Optional[str]:
    """First declared escalation target for ``trigger``, if any.

    The path table is ordered by preference; P3 before P5 for a normal QA
    escalation, for example.
    """
    if trigger not in EscalationTrigger:
        raise ValueError(f"unknown escalation trigger {trigger!r}")
    for target in path.escalates_to:
        return target
    return None


def should_escalate(
    triggers: Sequence[EscalationTrigger],
    policy: Policy,
    *,
    escalations_used: int = 0,
) -> Optional[EscalationTrigger]:
    """Pick the escalation trigger to act on, or ``None``.

    Deterministic priority — a verifier rejection outranks a low-confidence
    route because it is observed later and carries more information.  The
    budget is ``Policy.max_escalations`` per request.
    """
    if escalations_used >= policy.max_escalations:
        return None
    priority = (
        EscalationTrigger.VERIFIER_REJECT,
        EscalationTrigger.TOOL_FAILED,
        EscalationTrigger.BUDGET_EXHAUSTED,
        EscalationTrigger.ROUTE_LOW_CONFIDENCE,
    )
    for candidate in priority:
        if candidate in triggers:
            return candidate
    return None


def escalation_probability(confidence: float) -> float:
    """A first-order estimate of P(escalate | route confidence).

    The skeleton's calibration stand-in: confidence 0 escalates with
    probability 1, confidence 1 never escalates.  Offline threshold fitting
    replaces this with the measured curve.
    """
    return max(0.0, min(1.0, 1.0 - float(confidence)))


def streamable_tokens(tokens_generated: int, tokens_verified: int, policy: Policy) -> int:
    """How many generated tokens are safe to emit.

    At most ``verify_window`` tokens are held under check; everything older
    has been through the verifier (or outlived the window).  The invariant is
    monotone: more generation or more verification never shrinks the stream.
    """
    generated = max(0, int(tokens_generated))
    verified = max(0, min(int(tokens_verified), generated))
    unverified = generated - verified
    held = min(unverified, max(0, int(policy.verify_window)))
    return generated - held
