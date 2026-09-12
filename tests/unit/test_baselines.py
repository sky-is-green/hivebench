"""Unit tests for the S0.5 baseline harness (cortex.baselines)."""

import json

from cortex.baselines import metrics as m
from cortex.baselines import runner
from cortex.baselines.runner import (
    MockClient,
    build_fifo_messages,
    build_lmstudio_messages,
    load_conversations,
    run_baseline,
)

GENERATED_DIR = "tests/fixtures/generated"


def test_load_conversations_generated_corpus():
    convs = load_conversations(GENERATED_DIR)
    assert len(convs) == 50
    assert all("turns" in c for c in convs)


def test_estimate_tokens():
    assert m.estimate_tokens("") >= 1
    assert m.estimate_tokens("a" * 40) == 10
    assert m.estimate_tokens("hello world") >= 1


def test_lmstudio_messages_full_history():
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "question"},
    ]
    assert build_lmstudio_messages(history) == history


def test_fifo_messages_truncate_to_window():
    history = [
        {"role": "user", "content": "old" * 2000},   # ~1500 tokens
        {"role": "assistant", "content": "old" * 2000},
        {"role": "user", "content": "newest question"},
    ]
    msgs = build_fifo_messages(history)
    # newest message always retained
    assert msgs[-1] == history[-1]
    total = sum(m.estimate_tokens(x["content"]) for x in msgs)
    assert total <= runner.FIFO_WINDOW_TOKENS


def test_mock_run_baseline_and_record(tmp_path):
    convs = load_conversations(GENERATED_DIR)[:3]
    client = MockClient()
    results = run_baseline(
        convs, client, mode="lm_studio",
        build_messages=build_lmstudio_messages,
        baseline_tps=30.0, max_context=32768,
    )
    assert len(results) == 3
    for r in results:
        assert r.mode == "lm_studio"
        assert r.avg_latency_ms > 0
        assert 0 <= r.pes <= 100
        assert r.context_utilization is not None and 0 <= r.context_utilization <= 1

    out = tmp_path / "baseline_lmstudio.json"
    doc = m.record_baseline(results, out, max_context=32768)
    assert out.exists()
    assert doc["conversation_count"] == 3
    assert "aggregate" in doc and "conversations" in doc
    assert "avg_tokens_per_sec" in doc["aggregate"]
    # file is valid JSON
    json.loads(out.read_text(encoding="utf-8"))


def test_fifo_baseline_runs_with_mock(tmp_path):
    convs = load_conversations(GENERATED_DIR)[:2]
    results = run_baseline(
        convs, MockClient(), mode="fifo",
        build_messages=build_fifo_messages, baseline_tps=30.0, max_context=32768,
    )
    assert all(r.mode == "fifo" for r in results)
    assert all(r.max_prompt_tokens <= runner.FIFO_WINDOW_TOKENS for r in results)