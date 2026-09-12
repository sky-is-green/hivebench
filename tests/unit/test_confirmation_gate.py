"""S6 — Confirmation Gate & Imprint Grading tests.

Covers the grading math, the accept/reject/flag bands, the imprint providers
(fixture + digest), rule-parity (the gate must catch everything the rule-based
hedge filter catches), and the Strata wiring with the mechanism-attribution
condition (gate disabled == rule-based behavior).
"""

import pytest

from cortex.config import StrataConfig
from cortex.confirmation_gate import (
    ConfirmationGate,
    DigestImprint,
    FixtureImprint,
    content_terms,
    fact_terms,
)
from cortex.strata import Strata
from cortex.e2e import FakeUltraSmall

FIXTURE_ANSWER_MAP = {
    "conv_a": {
        "What is the rate limit for the API?": (
            "The API allows 100 requests per minute with a token bucket; "
            "bursts up to 250 are tolerated for 10 seconds."
        ),
        "How are errors reported?": (
            "Errors are reported as JSON bodies with an error code field and "
            "a human-readable message plus a request id."
        ),
    },
}


class _StubBackend:
    """Returns scripted replies; nothing else (no KV cache expectations)."""

    def __init__(self, replies: dict):
        self._replies = replies
        self.last_usage = {}

    def generate(self, content, query, sampling=None):
        return self._replies.get(query, "")


# ---------------------------------------------------------------------------
# Grading math


def test_content_terms_basic():
    assert "rate" in content_terms("The API rate limit is 100/min")
    assert "the" not in content_terms("the api")
    assert content_terms("") == set()


def test_fact_terms_answer_minus_query():
    facts = fact_terms(
        "What is the rate limit for the API?",
        "The API allows 100 requests per minute with a token bucket.",
    )
    assert "100" in facts and "rate" not in facts and "api" not in facts
    assert facts  # the answer adds distinctive facts


# ---------------------------------------------------------------------------
# Imprints


def test_fixture_imprint_facts_for_known_and_unknown():
    imp = FixtureImprint(FIXTURE_ANSWER_MAP)
    facts = imp.facts_for("conv_a", "What is the rate limit for the API?")
    assert "100" in facts
    assert imp.facts_for("conv_a", "totally new query") == set()
    assert imp.facts_for("other_conv", "What is the rate limit for the API?") == set()


def test_digest_imprint_accumulates_confirmed_facts():
    imp = DigestImprint()
    assert imp.facts_for("c1", "q") == set()
    imp.confirm("c1", {"rate", "100"})
    imp.confirm("c1", {"bucket"})
    assert imp.facts_for("c1", "q") == {"rate", "100", "bucket"}
    assert imp.facts_for("c2", "q") == set()  # per-conversation isolation


# ---------------------------------------------------------------------------
# Gate decisions


def test_accept_when_close_to_copy():
    gate = ConfirmationGate()
    imp = FixtureImprint(FIXTURE_ANSWER_MAP)
    d = gate.decide(
        "conv_a", "What is the rate limit for the API?",
        "The API allows 100 requests per minute with a token bucket; bursts "
        "up to 250 are tolerated.",
        imp,
    )
    assert d.decision == "accept"
    assert d.ingestion_ratio >= 0.4
    assert d.rule_hedge is False


def test_reject_when_rule_hedge_even_if_copy_like():
    gate = ConfirmationGate()
    imp = FixtureImprint(FIXTURE_ANSWER_MAP)
    d = gate.decide(
        "conv_a", "What is the rate limit for the API?",
        "I do not have access to the API documentation.",
        imp,
    )
    assert d.decision == "reject"
    assert d.reason == "rule_hedge"
    assert gate.stats["rule_hedge_rejects"] == 1


def test_reject_when_not_a_copy():
    gate = ConfirmationGate()
    imp = FixtureImprint(FIXTURE_ANSWER_MAP)
    d = gate.decide(
        "conv_a", "What is the rate limit for the API?",
        "The weather in Berlin is mild today with some clouds.",
        imp,
    )
    assert d.decision == "reject"
    assert d.reason == "not_copy"
    assert d.ingestion_ratio == 0.0


def test_flag_on_borderline():
    gate = ConfirmationGate()
    imp = FixtureImprint(FIXTURE_ANSWER_MAP)
    d = gate.decide(
        "conv_a", "What is the rate limit for the API?",
        "There is a rate limit involving a bucket that allows requests per "
        "minute; details vary by environment and configuration.",
        imp,
    )
    assert d.decision == "flag"
    assert d.reason == "borderline"


def test_first_mention_substantive_accept_and_thin_reject():
    gate = ConfirmationGate()
    imp = FixtureImprint(FIXTURE_ANSWER_MAP)  # unknown query -> no imprint facts
    d = gate.decide(
        "conv_a", "Tell me about the new feature",
        "The new feature adds a background sync worker that retries failed "
        "jobs with exponential backoff and a circuit breaker.",
        imp,
    )
    assert d.decision == "accept"
    assert d.reason == "substantive_first_mention"
    assert d.ingestion_ratio is None

    d2 = gate.decide("conv_a", "Tell me about the new feature", "Okay.", imp)
    assert d2.decision == "reject"
    assert d2.reason == "thin"


def test_gate_parity_with_rule_hedge_filter():
    """The gate must reject everything the rule-based filter catches,
    including the contraction variants that once slipped through."""
    gate = ConfirmationGate()
    imp = FixtureImprint(FIXTURE_ANSWER_MAP)
    hedges = [
        "Based on the provided context, there is no information regarding that.",
        "I don't have access to the API docs.",
        "I can't show you the internal configuration.",
        "There is no specific information available about the endpoint.",
    ]
    for h in hedges:
        d = gate.decide("conv_a", "What is the rate limit for the API?", h, imp)
        assert d.decision == "reject", f"should reject hedge: {h!r}"
        assert d.rule_hedge is True
    # A factual answer with a mid-reply caveat must NOT be rejected — but the
    # caveat must sit beyond the 90-char lead window (a caveat inside the
    # opening is indistinguishable from a refusal and IS rejected by design).
    factual = (
        "The API allows 100 requests per minute with a token bucket, and "
        "bursts up to 250 are tolerated for ten seconds in normal operation. "
        "I don't have specific details about your setup, but the general "
        "model is a token bucket."
    )
    d = gate.decide("conv_a", "What is the rate limit for the API?", factual, imp)
    assert d.decision == "accept"
    assert d.rule_hedge is False


def test_gate_summary_counts():
    gate = ConfirmationGate()
    imp = FixtureImprint(FIXTURE_ANSWER_MAP)
    gate.decide("conv_a", "What is the rate limit for the API?",
                "The API allows 100 requests per minute with a token bucket.",
                imp)  # accept (ratio ~0.58 >= 0.4)
    gate.decide("conv_a", "What is the rate limit for the API?",
                "I do not have access.", imp)  # rule_hedge reject
    gate.decide("conv_a", "What is the rate limit for the API?",
                "Irrelevant chatter about the weather.", imp)  # not_copy reject
    s = gate.summary()
    assert s["accepted"] == 1
    assert s["rejected"] == 2
    assert s["rule_hedge_rejects"] == 1
    assert s["mean_ingestion_ratio"] is not None


# ---------------------------------------------------------------------------
# Strata wiring


def test_gate_disabled_by_default_uses_rule_filter():
    strata = Strata(StrataConfig(), ultra=FakeUltraSmall())
    assert strata.gate is None
    assert strata.config.gate_enabled is False


def test_gate_wired_when_enabled_and_hedges_not_stored():
    config = StrataConfig(gate_enabled=True)
    imp = FixtureImprint(FIXTURE_ANSWER_MAP)
    strata = Strata(
        config,
        ultra=FakeUltraSmall(),
        backend=_StubBackend({
            "What is the rate limit for the API?": (
                "The API allows 100 requests per minute with a token bucket."
            ),
        }),
        confirmation_imprint=imp,
    )
    assert strata.gate is not None

    # Turn 1: factual reply -> accept -> stored.
    r1 = strata.process_turn(
        "What is the rate limit for the API?", conversation_id="conv_a"
    )
    assert r1.reply
    stored_contents = [c.content for c in strata.store.all_chunks()]
    assert any("100 requests per minute" in c for c in stored_contents)
    assert strata.gate_stats["decisions"][-1]["decision"] == "accept"

    # Turn 2: a refusal reply -> reject -> NOT stored (query chunk still is).
    hive2 = Strata(
        config, ultra=FakeUltraSmall(),
        backend=_StubBackend({
            "What is the rate limit for the API?": (
                "I do not have access to that information."
            ),
        }),
        confirmation_imprint=FixtureImprint(FIXTURE_ANSWER_MAP),
    )
    hive2.process_turn("What is the rate limit for the API?", conversation_id="conv_a")
    stored2 = [c.content for c in hive2.store.all_chunks()]
    assert len(stored2) == 1  # only the query chunk
    assert "I do not have access" not in stored2[0]
    assert hive2.gate_stats["decisions"][-1]["decision"] == "reject"


def test_mechanism_attribution_gate_disabled_matches_rule():
    """With the gate disabled, the rule-based hedge filter governs exactly
    as before (the mechanism-attribution condition)."""
    config = StrataConfig(filter_hedge_replies=True)
    strata = Strata(
        config, ultra=FakeUltraSmall(),
        backend=_StubBackend({
            "q": "I don't have access to that information.",
        }),
    )
    strata.process_turn("q", conversation_id="conv_a")
    stored = [c.content for c in strata.store.all_chunks()]
    assert len(stored) == 1  # query only; hedge filtered by the rule


def test_digest_imprint_wiring_accumulates_accepted_facts():
    """Live mode: the digest imprint grows from accepted replies, so a later
    ask about an established fact is graded against it."""
    config = StrataConfig(gate_enabled=True)
    strata = Strata(
        config, ultra=FakeUltraSmall(),
        backend=_StubBackend({
            "Tell me about the new feature": (
                "The new feature adds a background sync worker with "
                "exponential backoff and a circuit breaker."
            ),
            "What does the sync worker do?": (
                "The sync worker retries failed jobs with exponential backoff "
                "and a circuit breaker."
            ),
        }),
    )  # no imprint injected -> DigestImprint
    assert isinstance(strata.gate_imprint, DigestImprint)

    strata.process_turn("Tell me about the new feature", conversation_id="c1")
    facts = strata.gate_imprint.facts_for("c1", "What does the sync worker do?")
    assert "backoff" in facts  # established by the first accepted reply

    d = strata.gate.decide(
        "c1", "What does the sync worker do?",
        "The sync worker retries failed jobs with exponential backoff and a "
        "circuit breaker.",
        strata.gate_imprint,
    )
    assert d.decision == "accept"
    assert d.ingestion_ratio is not None and d.ingestion_ratio > 0