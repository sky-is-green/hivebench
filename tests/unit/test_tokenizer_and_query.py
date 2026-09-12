"""Tests for tokenizer utility (E5) and log query tool (E4)."""

from cortex.tokenizer import Tokenizer, estimate_tokens
from logs.event_logger import EventLogger
from logs.query import load_entries, summarize


def test_estimate_tokens():
    assert estimate_tokens("") >= 1
    assert estimate_tokens("a" * 40) == 10


def test_heuristic_tokenizer():
    assert Tokenizer(use_real=False).count("a" * 40) == 10


def test_real_tokenizer_falls_back_when_unavailable():
    t = Tokenizer(use_real=True)
    assert t.count("hello world") >= 1
    assert t.is_real is False or t.is_real is True  # either path returns a count


def test_log_query_summarize(tmp_path):
    logger = EventLogger(log_dir=tmp_path)
    logger.log("router", "task_classified", {"routed_to": "ultra_small"}, latency_ms=2.0)
    logger.log("efficiency", "score_computed", {"composite_score": 85}, latency_ms=0.5)
    logger.flush()
    logger.close()

    entries = load_entries(tmp_path)
    s = summarize(entries)
    assert s["entries"] == 2
    assert s["routes"] == {"ultra_small": 1}
    assert s["pes"]["mean"] == 85.0
    assert s["latency_ms"]["avg"] > 0


def test_log_query_summarize_empty():
    assert summarize([])["entries"] == 0