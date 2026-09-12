"""Unit tests for the backend package (S3.1)."""

import pytest

from backend.lmstudio import LMStudioBackend
from backend.openai_compat import OpenAICompatBackend, resolve_endpoint
from backend.vllm import VLLMBackend


class FakeResponse:
    def __init__(self, payload, ok=True):
        self._payload = payload
        self.ok = ok
        self.status_code = 200 if ok else 500

    def json(self):
        return self._payload

    def raise_for_status(self):
        if not self.ok:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeTransport:
    def __init__(self, chat_payload=None, ok=True):
        self.chat_payload = chat_payload or {
            "choices": [{"message": {"content": "hello from backend"}}]
        }
        self.ok = ok
        self.posts = []
        self.gets = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.posts.append((url, json))
        return FakeResponse(self.chat_payload, self.ok)

    def get(self, url, headers=None, timeout=None):
        self.gets.append(url)
        return FakeResponse({"data": [{"id": "model-1"}]}, self.ok)


def test_resolve_endpoint():
    assert resolve_endpoint("") == "http://localhost:1234"
    assert resolve_endpoint("localhost") == "http://localhost"
    assert resolve_endpoint("localhost:8000") == "http://localhost:8000"
    assert resolve_endpoint("http://10.0.0.1:8000") == "http://10.0.0.1:8000"
    assert resolve_endpoint("https://example.com") == "https://example.com"


def test_openai_compat_generate_formats_messages():
    transport = FakeTransport()
    backend = OpenAICompatBackend(
        base_url="localhost:1234", model="m", transport=transport
    )
    result = backend.generate("SYSTEM CONTEXT", "USER QUERY")

    assert result == "hello from backend"
    url, payload = transport.posts[0]
    assert url == "http://localhost:1234/v1/chat/completions"
    assert payload["model"] == "m"
    assert payload["messages"] == [
        {"role": "system", "content": "SYSTEM CONTEXT"},
        {"role": "user", "content": "USER QUERY"},
    ]
    assert payload["temperature"] == 0.2  # default sampling


def test_openai_compat_health():
    transport = FakeTransport()
    backend = OpenAICompatBackend(base_url="localhost", transport=transport)
    assert backend.health() is True
    assert transport.gets  # hit /v1/models


def test_openai_compat_error_propagates():
    class RaisingTransport:
        def post(self, url, json=None, headers=None, timeout=None):
            raise RuntimeError("connection refused")

        def get(self, url, headers=None, timeout=None):
            return FakeResponse({}, ok=False)

    backend = OpenAICompatBackend(base_url="localhost", transport=RaisingTransport())
    with pytest.raises(RuntimeError):
        backend.generate("ctx", "query")


def test_vllm_defaults_and_surgical():
    backend = VLLMBackend(transport=FakeTransport())
    assert backend.base_url == "http://localhost:8000"
    assert backend.model == "qwen3.8-27b"
    assert backend.supports_surgical_edits is True


def test_lmstudio_defaults_and_noop():
    backend = LMStudioBackend(transport=FakeTransport())
    assert backend.base_url == "http://localhost:1234"
    assert backend.api_key == "lm-studio"
    assert backend.supports_surgical_edits is False


def test_pinned_prefix_sent_first_as_system_message():
    transport = FakeTransport()
    backend = OpenAICompatBackend(
        base_url="localhost:1234", model="m", transport=transport,
        pinned_prefix="STABLE RULES",
    )
    backend.generate("DYNAMIC CONTEXT", "USER")
    _url, payload = transport.posts[0]
    # merged into a single leading system message (strict templates require this)
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][0]["content"] == "STABLE RULES\n\nDYNAMIC CONTEXT"
    assert payload["messages"][1] == {"role": "user", "content": "USER"}


def test_no_pinned_prefix_omits_extra_message():
    transport = FakeTransport()
    backend = OpenAICompatBackend(base_url="localhost", model="m", transport=transport)
    backend.generate("CTX", "Q")
    _url, payload = transport.posts[0]
    assert payload["messages"] == [
        {"role": "system", "content": "CTX"},
        {"role": "user", "content": "Q"},
    ]


def test_last_usage_tracked_from_response():
    transport = FakeTransport(chat_payload={
        "choices": [{"message": {"content": "ok"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    })
    backend = OpenAICompatBackend(base_url="localhost", model="m", transport=transport)
    assert backend.last_usage == {}
    backend.generate("ctx", "query")
    assert backend.last_usage == {
        "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
    }


def test_pinned_prefix_byte_stable_across_turns():
    from backend.cache_manager import KVCacheManager

    transport = FakeTransport()
    backend = OpenAICompatBackend(base_url="localhost", model="m", transport=transport)
    mgr = KVCacheManager(backend)
    prefix = "STABLE RULES ARE HERE"
    first = mgr.update_cache("context version A", persistent_prefix=prefix)
    backend.generate("context version A", "Q1")
    second = mgr.update_cache("context version B", persistent_prefix=prefix)
    backend.generate("context version B", "Q2")

    sys_a = transport.posts[0][1]["messages"][0]["content"]
    sys_b = transport.posts[1][1]["messages"][0]["content"]
    # both system messages lead with the identical pinned-prefix bytes
    assert sys_a.startswith(prefix + "\n\n")
    assert sys_b.startswith(prefix + "\n\n")
    head = len(prefix + "\n\n")
    assert sys_a[:head] == sys_b[:head]
    assert sys_a != sys_b  # the dynamic context portion differs
    # the cache manager reports the prefix as stable across turns
    assert second["mode"] == "prefix_caching"
    assert second["prefix_stable"] is True
    assert first["cache_invalidated"] is False