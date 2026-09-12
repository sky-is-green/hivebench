"""Unit tests for experiments.model_probe (all-models speed probe)."""

import json

import pytest

from experiments.model_probe import (
    PROBE_PROMPT,
    ProbeResult,
    _list_models,
    probe_model,
)


class FakeModelsResponse:
    def __init__(self, models):
        self._models = models

    def raise_for_status(self):
        pass

    def json(self):
        return {"data": [{"id": m} for m in self._models]}


class FakeStreamResponse:
    def __init__(self, chunks, ok=True):
        self._chunks = chunks
        self.ok = ok
        self.status_code = 200 if ok else 500

    def raise_for_status(self):
        if not self.ok:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_lines(self, decode_unicode=True):
        for c in self._chunks:
            if c == "[DONE]":
                yield b"data: [DONE]"
            else:
                yield b"data: " + json.dumps(c).encode("utf-8")


def _chunk(content, usage=None):
    d = {"choices": [{"delta": {"content": content}}]}
    if usage:
        d["usage"] = usage
    return d


class FakeHttp:
    def __init__(self, models=("m1", "m2"), chunks=None, ok=True):
        self._models = models
        self._chunks = chunks
        self._ok = ok
        self.posts = []
        self.gets = []

    def get(self, url, timeout=None):
        self.gets.append(url)
        return FakeModelsResponse(self._models)

    def post(self, url, json=None, stream=None, timeout=None):
        self.posts.append((url, json))
        chunks = self._chunks if self._chunks is not None else [
            _chunk("ok", usage={"completion_tokens": 2}),
            "[DONE]",
        ]
        return FakeStreamResponse(chunks, self._ok)


def test_list_models_sorted_and_filtered():
    http = FakeHttp(models=("beta", "alpha"))
    assert _list_models("http://x", http=http) == ["alpha", "beta"]


def test_probe_pass_measures_ttft_tps():
    http = FakeHttp()
    r = probe_model("http://x", "m1", http=http)
    assert r.status == "PASS"
    assert r.reply_len == 2
    assert r.ttft_ms is not None
    assert r.decode_tps is not None
    assert r.completion_tokens == 2
    # streaming + enable_thinking=false were requested
    _url, payload = http.posts[0]
    assert payload["stream"] is True
    assert payload["enable_thinking"] is False
    assert payload["max_tokens"] == 64


def test_probe_sends_probe_prompt():
    http = FakeHttp()
    probe_model("http://x", "m1", http=http)
    _url, payload = http.posts[0]
    assert payload["messages"][1]["content"] == PROBE_PROMPT


def test_probe_empty_reply_flags_reasoning_burn():
    # completion_tokens hit the cap with no visible content -> reasoning burn
    http = FakeHttp(chunks=[
        _chunk("", usage={"completion_tokens": 64}),
        "[DONE]",
    ])
    r = probe_model("http://x", "m1", http=http)
    assert r.status == "EMPTY"
    assert "reasoning" in r.note


def test_probe_empty_reply_no_cap_note():
    http = FakeHttp(chunks=[_chunk("", usage={"completion_tokens": 3}), "[DONE]"])
    r = probe_model("http://x", "m1", http=http)
    assert r.status == "EMPTY"
    assert "no visible content" in r.note


def test_probe_http_error_returns_fail():
    http = FakeHttp(chunks=[], ok=False)
    r = probe_model("http://x", "m1", http=http)
    assert r.status == "FAIL"
    assert r.error


def test_probe_disable_thinking_off_omits_flag():
    http = FakeHttp()
    probe_model("http://x", "m1", disable_thinking=False, http=http)
    _url, payload = http.posts[0]
    assert "enable_thinking" not in payload