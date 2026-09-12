"""Unit tests for auditor.auditor (S4.1)."""

import json

from auditor.auditor import Auditor, TurnRecord


def _record(turn, ctx="some context", q="q", resp="a"):
    return TurnRecord(
        turn=turn, assembled_context=ctx, user_query=q, llm_response=resp,
        chunk_ids=["chunk_a", "chunk_b"],
    )


def test_evaluate_turn_parses_sufficient():
    auditor = Auditor(
        generate_fn=lambda p: json.dumps(
            {"sufficient": True, "used_pieces": ["chunk_a"], "missing": [], "score": 5}
        )
    )
    label = auditor.evaluate_turn(_record(1))
    assert label.turn == 1
    assert label.context_sufficient is True
    assert label.sufficiency_score == 5
    assert label.context_used == ["chunk_a"]
    assert label.chunk_labels["chunk_a"] is True


def test_evaluate_turn_parses_insufficient():
    auditor = Auditor(
        generate_fn=lambda p: json.dumps(
            {"sufficient": False, "used_pieces": [], "missing": ["the schema"], "score": 2}
        )
    )
    label = auditor.evaluate_turn(_record(2))
    assert label.context_sufficient is False
    assert label.sufficiency_score == 2
    assert label.missing_context == ["the schema"]


def test_run_batch_respects_sampling_rate():
    auditor = Auditor(
        generate_fn=lambda p: json.dumps(
            {"sufficient": True, "used_pieces": [], "missing": [], "score": 4}
        )
    )
    log = [_record(i) for i in range(30)]
    labels = auditor.run_batch(log, sample_rate=0.1)  # every 10th turn
    assert [l.turn for l in labels] == [0, 10, 20]
    assert len(labels) == 3


def test_run_batch_sample_rate_50():
    auditor = Auditor(
        generate_fn=lambda p: json.dumps(
            {"sufficient": True, "used_pieces": [], "missing": [], "score": 4}
        )
    )
    labels = auditor.run_batch([_record(i) for i in range(6)], sample_rate=0.5)
    assert [l.turn for l in labels] == [0, 2, 4]


def test_malformed_json_raises():
    auditor = Auditor(generate_fn=lambda p: "not json")
    import pytest

    with pytest.raises(Exception):
        auditor.evaluate_turn(_record(1))


def test_empty_response_raises():
    auditor = Auditor(generate_fn=lambda p: "")
    import pytest

    with pytest.raises(ValueError):
        auditor.evaluate_turn(_record(1))


def test_markdown_fenced_json_parses():
    auditor = Auditor(
        generate_fn=lambda p: '```json\n{"sufficient": true, "used_pieces": [], "missing": [], "score": 4}\n```'
    )
    label = auditor.evaluate_turn(_record(1))
    assert label.context_sufficient is True
    assert label.sufficiency_score == 4


def test_prose_wrapped_json_parses():
    auditor = Auditor(
        generate_fn=lambda p: 'Here is my judgment: {"sufficient": false, "used_pieces": [], "missing": ["x"], "score": 2} Hope that helps.'
    )
    label = auditor.evaluate_turn(_record(1))
    assert label.context_sufficient is False
    assert label.sufficiency_score == 2
    assert label.missing_context == ["x"]


def test_trailing_garbage_json_parses():
    auditor = Auditor(
        generate_fn=lambda p: '{"sufficient": true, "used_pieces": [], "missing": [], "score": 5} (reasoning follows)'
    )
    label = auditor.evaluate_turn(_record(1))
    assert label.context_sufficient is True
    assert label.sufficiency_score == 5