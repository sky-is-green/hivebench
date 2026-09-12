"""cortex.tokenizer — exact token counting via the active real tokenizer.

The heuristic (~4 chars/token) is the default; ``set_active_tokenizer``
makes budgets exact for a run without touching call sites. Tests build a tiny
real tokenizer.json in tmp and verify: (1) the active tokenizer changes
counts, (2) the heuristic stays the default, (3) a bad path falls back.
"""

import json
import re
from pathlib import Path

from cortex.tokenizer import (
    Tokenizer,
    active_tokenizer,
    estimate_tokens,
    set_active_tokenizer,
)


def _write_wordpiece_tokenizer(path: Path) -> Path:
    """A minimal real tokenizer.json (word-level whitespace split) the
    ``tokenizers`` lib can load — no downloads, deterministic counts."""
    vocab = {}
    words = ["the", "api", "allows", "requests", "per", "minute", "token",
             "bucket", "100", "with", "a", "rate", "limit", "is", "error",
             "code", "field"]
    # sentencepiece-style vocab: id 0 = <unk>, rest sorted by frequency count
    for i, w in enumerate(sorted(words)):
        vocab[w] = i
    data = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "model": {
            "type": "WordPiece",
            "unk_token": "<unk>",
            "continuing_subword_prefix": "##",
            "max_input_chars_per_word": 100,
            "vocab": vocab,
        },
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_heuristic_is_default():
    assert active_tokenizer() is None
    assert estimate_tokens("a" * 40) == 10


def test_real_tokenizer_counts_differ_and_are_exact(tmp_path):
    tok_path = _write_wordpiece_tokenizer(tmp_path / "tokenizer.json")
    tok = Tokenizer(use_real=True, tokenizer_path=str(tok_path))
    assert tok.is_real
    # "api allows requests per minute" -> 5 whitespace tokens (not 24/4=6)
    assert tok.count("api allows requests per minute") == 5
    assert tok.count("") == 1


def test_set_active_tokenizer_changes_estimate(tmp_path):
    tok_path = _write_wordpiece_tokenizer(tmp_path / "tokenizer.json")
    tok = Tokenizer(use_real=True, tokenizer_path=str(tok_path))
    set_active_tokenizer(tok)
    try:
        assert estimate_tokens("api allows requests per minute") == 5
        assert estimate_tokens("") == 1
    finally:
        set_active_tokenizer(None)
    assert active_tokenizer() is None
    assert estimate_tokens("a" * 40) == 10  # heuristic restored


def test_bad_tokenizer_path_falls_back():
    tok = Tokenizer(use_real=True, tokenizer_path="no/such/file.json")
    assert not tok.is_real
    assert tok.count("whatever text") == max(1, len("whatever text") // 4)


def test_tokenizer_from_model_json_helper(tmp_path):
    from cortex.tokenizer import tokenizer_from_model_json

    assert tokenizer_from_model_json(tmp_path / "missing.json") is None
    tok_path = _write_wordpiece_tokenizer(tmp_path / "tokenizer.json")
    tok = tokenizer_from_model_json(tok_path)
    assert tok is not None and tok.is_real


def test_unknown_words_do_not_crash(tmp_path):
    tok_path = _write_wordpiece_tokenizer(tmp_path / "tokenizer.json")
    tok = Tokenizer(use_real=True, tokenizer_path=str(tok_path))
    # unknown words map to unk / split — must not raise and must be >= 1
    assert tok.count("zzzzzzz qqqqqqq") >= 1