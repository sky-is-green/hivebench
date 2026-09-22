"""Unit tests for the deterministic P2 retrieval diagnostic (model-fidelity
reframe: recall is measured on facts the model actually stated in stored
replies; ingestion_rate and perfect_hive_ceiling bound it)."""

from experiments.retrieval_diagnostic import (
    _answer_fact_terms,
    _is_retrievable,
    _turn_precision,
    compute_retrieval_vs_fixture,
)

# A minimal fixture with a known query -> answer mapping and prior context.
FIXTURE = [
    {
        "conversation_id": "test_conv",
        "turns": [
            {"role": "user", "content": "How should we handle session store?"},
            {"role": "assistant", "content": "Use Redis with TTL for session store."},
            {"role": "user", "content": "How should we handle pagination?"},
            {"role": "assistant", "content": "Use cursor-based for pagination in the REST API."},
            # Repeated ask: the answer now exists in prior history.
            {"role": "user", "content": "What would you recommend for pagination in our REST API?"},
            {"role": "assistant", "content": "Use cursor-based for pagination in the REST API."},
        ],
    }
]

# Turn-1 query without a fixture answer: not sampled, but its reply is stored
# (it is where the fact first enters the conversation).
FILLER = "How should we design the API?"


def _record(turns, cid="test_conv"):
    return {"conversation_id": cid, "turns": turns}


# --- The reframe's key scenario: the model DID state the fact, splinter retrieved ---
HIT_RECORD = _record([
    {"turn": 1, "query": FILLER,
     "reply": "Use cursor-based for pagination.",
     "assembled_content": FILLER},
    {"turn": 2, "query": "What would you recommend for pagination in our REST API?",
     "reply": "Use cursor-based for pagination.",
     "assembled_content": "Use cursor-based for pagination. Use Redis with TTL."},
])

# --- Model stated the fact, splinter FAILED to retrieve it (genuine retrieval miss) ---
MISS_RECORD = _record([
    {"turn": 1, "query": FILLER,
     "reply": "Use cursor-based for pagination.",
     "assembled_content": FILLER},
    {"turn": 2, "query": "What would you recommend for pagination in our REST API?",
     "reply": "Use cursor-based for pagination.",
     "assembled_content": "Only unrelated content about the order schema."},
])

# --- The live3 edge case: model NEVER stated the expected fact -> not a splinter
#     failure, ingestion_rate < 100% and recall denominator excludes it. ---
NOT_STATED_RECORD = _record([
    {"turn": 1, "query": FILLER,
     "reply": "We should use offset paging for best simplicity.",
     "assembled_content": FILLER},
    {"turn": 2, "query": "What would you recommend for pagination in our REST API?",
     "reply": "We should use offset paging.",
     "assembled_content": "Use offset paging for the REST API."},
])

# --- Hedge reply must not count as stating a fact ---
HEDGE_RECORD = _record([
    {"turn": 1, "query": FILLER,
     "reply": "Based on the provided context, there is no information regarding pagination.",
     "assembled_content": FILLER},
    {"turn": 2, "query": "What would you recommend for pagination in our REST API?",
     "reply": "Use cursor-based for pagination.",
     "assembled_content": "Use cursor-based for pagination."},
])

FIRST_MENTION_RECORD = _record([
    {"turn": 1, "query": "How should we handle session store?",
     "reply": "Use Redis with TTL for session store.",
     "assembled_content": "Use Redis with TTL for session store."},
])


def test_answer_fact_terms_excludes_query_words():
    facts = _answer_fact_terms(
        "How should we handle session store?",
        "Use Redis with TTL for session store.",
    )
    # "session" and "store" appear in the query, so only the answer's additions
    # are facts; "ttl" is below the >=4-char word filter.
    assert "redis" in facts
    assert "session" not in facts
    assert "ttl" not in facts


def test_is_retrievable_distinguishes_first_mention():
    assert not _is_retrievable(
        "How should we handle session store?", ""  # no prior context
    )
    assert _is_retrievable(
        "What would you recommend for pagination in our REST API?",
        "Use cursor-based for pagination. The REST API returns JSON responses.",
    )
    # A single shared generic term is not enough to mark a turn retrievable.
    assert not _is_retrievable(
        "How should we handle auth service sessions?",
        "We should use Redis for the order schema cache.",
    )


def test_turn_precision_proxy():
    # All sentences share the topic term "pagination".
    assert _turn_precision(
        "How should we handle pagination?",
        "Use cursor-based for pagination. Pagination is important.",
    ) == 1.0
    # None share a topic term.
    assert _turn_precision(
        "How should we handle pagination?",
        "Only unrelated content about the order schema.",
    ) == 0.0
    # Empty assembled content -> None (not measurable).
    assert _turn_precision("How should we handle pagination?", "") is None


def test_recall_counts_only_stated_facts():
    """The honest metric: recall is over facts the model actually stated."""
    result = compute_retrieval_vs_fixture([HIT_RECORD], FIXTURE)
    assert result["retrieval_recall"] == 100.0
    assert result["retrieval_recall_retrievable"] == 100.0
    assert result["ingestion_rate"] == 100.0
    assert result["perfect_hive_ceiling"] == 100.0
    assert result["retrievable_turns"] == 1
    turn = result["turns"][-1]
    assert "cursor-based" in turn["stated_facts"]
    assert "cursor-based" in turn["facts_found"]


def test_recall_miss_when_stated_but_not_retrieved():
    """Model stated the fact but the splinter dropped it -> genuine splinter miss."""
    result = compute_retrieval_vs_fixture([MISS_RECORD], FIXTURE)
    assert result["retrieval_recall"] == 0.0
    assert result["ingestion_rate"] == 100.0  # fact WAS stated; splinter's fault
    assert result["perfect_hive_ceiling"] == 100.0


def test_never_stated_fact_not_a_hive_failure():
    """The live3 edge case: model said 'offset paging' not 'cursor-based'.
    The expected fact never entered the store -> excluded from recall, but the
    fidelity bounds report it."""
    result = compute_retrieval_vs_fixture([NOT_STATED_RECORD], FIXTURE)
    # no stated fact -> recall None (no measurable turns)
    assert result["retrieval_recall"] is None
    assert result["measurable_turns"] == 0
    # ingestion (fidelity) bound: model stated 0 of the expected facts
    assert result["ingestion_rate"] == 0.0
    assert result["perfect_hive_ceiling"] == 0.0
    # the turn is still reported with the diagnosis
    turn = result["turns"][-1]
    assert turn["stated_facts"] == []
    assert "cursor-based" in turn["answer_facts"]


def test_hedge_reply_does_not_count_as_stating():
    """A hedge reply ('no information regarding X') is filtered at store time,
    so its content must not make an expected fact 'stated'."""
    result = compute_retrieval_vs_fixture([HEDGE_RECORD], FIXTURE)
    # turn 1's reply is a hedge -> not stored -> cursor-based never enters the
    # store, so the turn-2 ask has no stated fact to retrieve.
    turn = result["turns"][-1]
    assert turn["stated_facts"] == []
    assert result["ingestion_rate"] == 0.0


def test_first_mention_not_counted_as_retrievable():
    result = compute_retrieval_vs_fixture([FIRST_MENTION_RECORD, HIT_RECORD], FIXTURE)
    assert result["sampled_turns"] == 2
    assert result["retrievable_turns"] == 1
    assert result["first_mention_turns"] == 1
    assert result["retrieval_recall"] == 100.0


def test_compute_retrieval_no_answer_skipped():
    # A turn whose query has no fixture answer is not sampled at all.
    result = compute_retrieval_vs_fixture(
        [_record([
            {"turn": 9, "query": "Something with no fixture answer.",
             "reply": "whatever", "assembled_content": "whatever"}])],
        FIXTURE,
    )
    assert result["sampled_turns"] == 0