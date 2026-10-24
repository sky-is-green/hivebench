"""Unit tests for the DeepInfer facade (``tools/deepinfer/serve.py``).

DeepInfer ships no HTTP server, so the facade is what makes it reachable
as a stack tier.  These tests use a **stub engine**: no DeepInfer, no GPU
and no checkpoint -- just the request/response contract the harness and the
dsh routes depend on (``/health``, ``/v1/models``, ``/v1/chat/completions``,
``/v1/completions``, streaming and not).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FACADE = REPO_ROOT / "tools" / "deepinfer" / "serve.py"


def _load_facade():
    """Import the facade by path (it is a script, not a package module)."""
    spec = importlib.util.spec_from_file_location("deepinfer_facade", FACADE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["deepinfer_facade"] = module
    spec.loader.exec_module(module)
    return module


facade = _load_facade()


class _StubTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "\n".join(
            f"{message['role']}: {message['content']}" for message in messages
        ) + "\nassistant:"


class _StubLLM:
    """A stand-in for ``deepinfer.LLM``: echoes the prompt back."""

    def __init__(self) -> None:
        self.closed = False

    def get_tokenizer(self):
        return _StubTokenizer()

    def generate(self, prompts, sampling_params=None, use_tqdm=False):
        return [f"echo: {prompt.strip()}" for prompt in prompts]

    def close(self) -> None:
        self.closed = True


class _StubSamplingParams:
    def __init__(self, **kwargs) -> None:
        self.kwargs = dict(kwargs)


class _StubEngine(facade.Engine):
    """The facade's engine with the DeepInfer load replaced.

    The batching worker is the real one, so the request path (queue, batch,
    deliver) is exercised end to end -- only the model is a stub.
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._llm = _StubLLM()
        self._tokenizer = self._llm.get_tokenizer()
        self._sampling_cls = _StubSamplingParams
        self._start_worker()


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    engine = _StubEngine({"model": "/models/stub", "served_model_name": "stub",
                          "ctx": 2048})
    app = facade._create_app(engine)
    with TestClient(app) as test_client:
        test_client.engine = engine  # type: ignore[attr-defined]
        yield test_client


def test_the_health_route_reports_the_service(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["loaded"] is True
    assert payload["model"] == "stub"
    assert payload["max_context"] == 2048
    assert payload["service"] == "deepinfer"


def test_the_models_route_is_the_openai_shape(client) -> None:
    response = client.get("/v1/models")
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert payload["data"][0]["id"] == "stub"


def test_a_chat_completion_returns_the_message(client) -> None:
    response = client.post("/v1/chat/completions", json={
        "model": "stub",
        "messages": [{"role": "user", "content": "hello"}],
    })
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "chat.completion"
    choice = payload["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert "hello" in choice["message"]["content"]
    assert choice["finish_reason"] == "stop"


def test_a_chat_completion_streams_the_events(client) -> None:
    response = client.post("/v1/chat/completions", json={
        "model": "stub",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
    })
    assert response.status_code == 200
    body = response.text
    assert body.count("data: ") >= 3
    assert body.rstrip().endswith("[DONE]")
    assert '"chat.completion.chunk"' in body


def test_a_chat_completion_without_messages_is_a_400(client) -> None:
    response = client.post("/v1/chat/completions", json={"model": "stub", "messages": []})
    assert response.status_code == 400


def test_a_plain_completion_returns_the_text(client) -> None:
    response = client.post("/v1/completions", json={
        "model": "stub", "prompt": "hello", "max_tokens": 8,
    })
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "text_completion"
    assert "hello" in payload["choices"][0]["text"]


def test_the_sampling_fields_are_mapped_onto_the_engine() -> None:
    mapped = facade._sampling_kwargs({
        "temperature": 0.2, "top_p": 0.9, "max_tokens": 32,
        "model": "stub", "messages": [], "stream": False,
    })
    assert mapped == {"temperature": 0.2, "top_p": 0.9, "max_tokens": 32}
