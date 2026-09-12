"""Regression-locked invariants for the P11 return corpus (``generate --return``).

The comb probe (2026-08-24) measured that only 45% of the ORIGINAL fixture's
return turns are lexically retrievable (composition queries never lexically
name the old facts) and that lexical retrieval reaches 76-87% recall on the
retrievable ones. This corpus builds SHADOW-style pure-fact return questions;
these tests lock the invariants the probe's design lesson demands:

  - A facts age past the stale wall (>= 21 turns) before the first return query
  - A-decision terms leak nowhere (not into the filler phase, not into return
    queries) — only A answers and return answers restate them
  - every return query is lexically retrievable: it shares a content word with
    its fact chunk (the probe's retrievability condition)
  - the abstain query has NO fact terms in any A chunk (genuinely absent)
"""

from pathlib import Path

from experiments.retrieval_diagnostic import (
    _answer_fact_terms,
    _content_terms,
    _fixture_answer_map,
)

from tests.fixtures.synthetic_conversations.generate import (
    RETURN_ESTABLISH_ASPECTS,
    RETURN_FILLER_TURNS,
    RETURN_SEED,
    generate_return,
)

FIXTURES = Path("hivebench/tests/fixtures/generated_return")


def _load() -> list[dict]:
    return generate_return(FIXTURES, seed=RETURN_SEED)


def test_return_facts_age_past_stale_wall():
    assert RETURN_FILLER_TURNS >= 21
    for conv in _load():
        turns = conv["turns"]
        n_a = conv["establish_turns"]
        # age of the FIRST A fact at the FIRST return query: the return phase
        # starts after phase A (2*n_a turns) + the filler phase (2*filler turns)
        first_return = 2 * n_a + 2 * conv["filler_turns"]
        first_a_answer = 1  # turn index of the first A answer
        assert first_return - first_a_answer >= 21
        # and the LAST A answer is also past the wall
        last_a_answer = 2 * n_a - 1
        assert first_return - last_a_answer >= 21


def test_return_decision_terms_are_distinctive():
    for conv in _load():
        turns = conv["turns"]
        n_a = conv["establish_turns"]
        c_start = 2 * n_a + 2 * conv["filler_turns"]
        a_decisions = set()
        for i in range(0, 2 * n_a, 2):
            # answer terms BEYOND the query (the diagnostic's notion: the
            # aspect name is in the query, so it is not a decision term)
            a_decisions |= _answer_fact_terms(turns[i]["content"], turns[i + 1]["content"])
        assert a_decisions, "no decision terms found in phase A"
        for i, t in enumerate(turns):
            if i < 2 * n_a:
                continue  # A answers carry the facts by design
            if t["role"] == "assistant" and i >= c_start:
                continue  # return answers restate the facts by design
            leak = a_decisions & _content_terms(t["content"])
            assert not leak, (
                f"decision-term leak in {conv['conversation_id']} turn {i}: {leak}"
            )


def test_return_queries_lexically_retrievable():
    """The probe's retrievability condition: each return query shares a content
    word with at least one prior fact chunk (the chunk whose facts it asks)."""
    for conv in _load():
        answers = _fixture_answer_map([conv])
        cid = conv["conversation_id"]
        turns = conv["turns"]
        for q in conv["return_queries"]:
            facts = _answer_fact_terms(q, answers[cid][q])
            assert facts, f"empty answer facts for return query: {q}"
            q_terms = _content_terms(q)
            hit = any(
                t["role"] == "assistant"
                and (_content_terms(t["content"]) & q_terms)
                and (facts & _content_terms(t["content"]))
                for t in turns
            )
            assert hit, f"return query not lexically retrievable: {q}"


def test_return_abstain_query_has_no_fact():
    for conv in _load():
        answers = _fixture_answer_map([conv])
        cid = conv["conversation_id"]
        q = conv["abstain_query"]
        facts = _answer_fact_terms(q, answers[cid][q])
        n_a = conv["establish_turns"]
        a_chunks = [
            t["content"]
            for i, t in enumerate(conv["turns"])
            if t["role"] == "assistant" and i < 2 * n_a
        ]
        assert not any(facts & _content_terms(c) for c in a_chunks), (
            f"abstain fact terms leaked into phase A in {cid}"
        )


def test_return_corpus_metadata_consistent():
    for conv in _load():
        assert conv["profile"] == "return"
        assert conv["establish_turns"] == RETURN_ESTABLISH_ASPECTS
        assert conv["filler_turns"] == RETURN_FILLER_TURNS
        assert len(conv["return_queries"]) == RETURN_ESTABLISH_ASPECTS + 1  # + multi-key
        assert conv["abstain_query"] not in conv["return_queries"]
        # every turn is well-formed user/assistant alternation
        for i, t in enumerate(conv["turns"]):
            assert t["role"] == ("user" if i % 2 == 0 else "assistant")
            assert t["content"].strip()