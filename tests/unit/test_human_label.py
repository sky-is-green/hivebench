"""Unit tests for the P7 human-labeling tool (experiments.human_label)."""

import json

import pytest

from experiments.human_label import (HumanRater, build_items, compute_agreement,
                                     run_auditor)


def test_build_items_balanced_and_seeded():
    a = build_items(50, seed=7)
    b = build_items(50, seed=7)
    c = build_items(50, seed=8)
    content = lambda items: [(it["query"], it["chunk"]) for it in items]
    assert content(a) == content(b)
    assert content(a) != content(c)
    assert all({"query", "chunk", "source", "relevant_gold", "item_id"} <= set(it)
               for it in a)
    assert any(it["relevant_gold"] for it in a)
    assert any(not it["relevant_gold"] for it in a)


def test_group_of_returns_position_for_every_item():
    """Every item index must resolve to a (group, position, size) triple —
    regression for the ValueError in _group_of (list.index on a non-member)."""
    from experiments.human_label import _build_groups, group_of

    items = build_items(50, seed=7)
    groups = _build_groups(items)
    assert sum(len(g["indices"]) for g in groups) == len(items)
    # simulate the GUI path: group_of over the flat index range
    for idx in range(len(items)):
        gi, pos, gsize = group_of(groups, idx)
        assert gi >= 0 and 0 <= pos < gsize
        assert groups[gi]["indices"][pos] == idx


def test_human_rater_resumes(tmp_path):
    items = build_items(10, seed=7)
    out = tmp_path / "human.ndjson"
    rater = HumanRater(items, out)
    rater.answer(items[0], 1)
    rater.answer(items[1], 2)
    resumed = HumanRater(items, out)
    assert resumed.answered_count() == 2
    assert resumed.answers[items[0]["item_id"]] == 1
    # answered items are skipped by next_unanswered
    idx, item = resumed.next_unanswered(0)
    assert idx == 2


def test_agreement_pass_and_fail(tmp_path):
    items = build_items(20, seed=7)
    human = tmp_path / "h.ndjson"
    auditor = tmp_path / "o.ndjson"
    r = HumanRater(items, human)
    for it in items:
        r.answer(it, 1 if it["relevant_gold"] else 2)
    # auditor agrees perfectly -> PASS
    with auditor.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(
                {"item_id": it["item_id"], "label": 1 if it["relevant_gold"] else 2}) + "\n")
    res = compute_agreement(items, human, auditor)
    assert res["verdict"] == "PASS"
    assert res["auditor_human_agreement"] == 1.0
    # auditor disagrees on everything -> FAIL below 90%
    with auditor.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(
                {"item_id": it["item_id"], "label": 2 if it["relevant_gold"] else 1}) + "\n")
    res = compute_agreement(items, human, auditor)
    assert res["verdict"] == "FAIL"
    assert "90%" in res["reasons"][0]


def test_human_human_below_auditor_human_fails(tmp_path):
    items = build_items(20, seed=7)
    human = tmp_path / "h.ndjson"
    human2 = tmp_path / "h2.ndjson"
    auditor = tmp_path / "o.ndjson"
    r = HumanRater(items, human)
    for it in items:
        r.answer(it, 1 if it["relevant_gold"] else 2)
    r2 = HumanRater(items, human2)
    for it in items:
        r2.answer(it, 2 if it["relevant_gold"] else 1)  # human2 disagrees with human
    with auditor.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(
                {"item_id": it["item_id"], "label": 1 if it["relevant_gold"] else 2}) + "\n")
    res = compute_agreement(items, human, auditor, human2)
    assert res["verdict"] == "FAIL"
    assert "human-human" in res["reasons"][0]


def test_auditor_runs_with_injected_generate_fn(tmp_path):
    items = build_items(5, seed=7)
    out = tmp_path / "auditor.ndjson"

    def fake_gen(prompt: str) -> str:
        return json.dumps({"relevant": True, "reason": "used"})

    run_auditor(items, out, "http://x", "m", generate_fn=fake_gen)
    labels = {}
    for line in out.read_text(encoding="utf-8").strip().splitlines():
        rec = json.loads(line)
        labels[rec["item_id"]] = rec["label"]
    assert len(labels) == len(items)
    assert all(v == 1 for v in labels.values())


def test_degenerate_query_detection():
    """Fixture 'X fit with X' artifacts are excluded from P7 agreement —
    regression for the 2026-08-23 contamination fix (100/500 items)."""
    from experiments.human_label import compute_agreement, is_degenerate_query

    assert is_degenerate_query("How does log levels fit with log levels?")
    assert is_degenerate_query("how does idempotency fit with idempotency?")
    assert not is_degenerate_query("Show me the code for migrations in the order schema.")
    assert not is_degenerate_query("How does log levels fit with log levels? (really)")
    assert not is_degenerate_query("How does structured logs fit with error envelope?")

    items = build_items(50, seed=7)
    # ensure the exclusion path is exercised end-to-end: build a human/auditor
    # pair that would PASS on clean items, and confirm degenerate ids appear
    # in the excluded block
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        human = d / "h.ndjson"
        auditor = d / "o.ndjson"
        r = HumanRater(items, human)
        for it in items:
            r.answer(it, 1 if it["relevant_gold"] else 2)
        with auditor.open("w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(
                    {"item_id": it["item_id"],
                     "label": 1 if it["relevant_gold"] else 2}) + "\n")
        res = compute_agreement(items, human, auditor)
        assert res["verdict"] == "PASS"
        assert res["excluded_degenerate"]["n"] > 0
        assert len(res["excluded_degenerate"]["item_ids"]) == res["excluded_degenerate"]["n"]