#!/usr/bin/env python3
"""An OpenAI-compatible HTTP facade over DeepInfer's offline engine.

DeepInfer ships **no server**.  This release exposes an offline
``LLM.generate()`` API plus a distributed launcher, and the README is
explicit that a KV-cache system, prefill/decode disaggregation and the
load balancing across replicas are *not* included.

The harness, however, speaks HTTP: every tier in a stack is an
OpenAI-compatible endpoint (llama-server, Strata's ``serve/server.py``),
and the dsh routes are OpenAI chat completions.  This module is the
missing piece -- it wraps DeepInfer's offline API in the HTTP surface the
rest of the harness expects, so a DeepInfer tier is a tier like any other.

Endpoints:

* ``GET  /health``            -- the harness's health probe
* ``GET  /v1/models``         -- the OpenAI model list
* ``POST /v1/chat/completions`` -- chat completions (streaming and not)
* ``POST /v1/completions``    -- plain completions (streaming and not)

Usage::

    python tools/deepinfer/serve.py --config <config.json> --port 8090

The config is a JSON object::

    {
      "model": "/path/to/DeepSeek-V4.1-Flash-1017",
      "served_model_name": "deepseek-v4.1-flash",
      "ctx": 4096,
      "expert_parallel_size": 1,
      "gpu_memory_utilization": 0.9,
      "seed": 0,
      "engine_args": ["--max-num-seqs", "8"]
    }
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

#: The repo root (the facade lives in ``tools/deepinfer/``).
REPO_ROOT = Path(__file__).resolve().parents[2]

#: How many pending requests one ``generate`` call may batch.
MAX_BATCH = 16

#: How long the worker waits for more requests before running a short batch.
BATCH_WINDOW_S = 0.02

#: The optional engine knobs the facade forwards to DeepInfer's EngineArgs.
#: They are named exactly as upstream names them, so the config reads like
#: the engine's own argument list.
_ENGINE_KWARGS = (
    "max_num_seqs",
    "max_num_batched_tokens",
    "data_parallel_size",
    "context_parallel_size",
    "chunked_prefill",
    "load_format",
    "trust_remote_code",
    "ignore_len",
    "tokens_per_block",
    "lookahead_tokens",
)


@dataclass
class PendingRequest:
    """One chat/completion request waiting for the engine."""

    prompt: str
    sampling: dict[str, Any]
    created: int = field(default_factory=lambda: int(time.time()))
    request_id: str = field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    result: Optional[list[str]] = None
    error: Optional[str] = None
    event: threading.Event = field(default_factory=threading.Event)


class Engine:
    """The DeepInfer engine, plus the batching loop that feeds it.

    DeepInfer's ``LLM.generate`` is blocking and batch-oriented, so the
    facade keeps a queue of pending requests and runs them in batches: a
    worker thread drains the queue (waiting a short window so a burst
    becomes one batch) and hands the prompts to the engine.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = dict(config)
        self.model_path = str(self.config.get("model") or "")
        if not self.model_path:
            raise SystemExit("the config must name a model directory")
        self.served_model_name = str(
            self.config.get("served_model_name") or Path(self.model_path).name)
        self.ctx = int(self.config.get("ctx") or 4096)
        self._lock = threading.Lock()
        self._queue: list[PendingRequest] = []
        self._wake = threading.Condition(self._lock)
        self._llm: Any = None
        self._tokenizer: Any = None
        self._sampling_cls: Any = None
        self._worker: Optional[threading.Thread] = None
        self._stop = False

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def load(self) -> None:
        """Build the engine and start the batching worker."""
        from deepinfer.entrypoints.llm import LLM

        kwargs: dict[str, Any] = {
            "model": self.model_path,
            "served_model_name": self.served_model_name,
            "gpu_memory_utilization": float(
                self.config.get("gpu_memory_utilization", 0.9)),
            "seed": int(self.config.get("seed", 0)),
            "max_model_len": self.ctx,
            "expert_parallel_size": int(self.config.get("expert_parallel_size", 1)),
        }
        # The optional engine knobs, mapped onto DeepInfer's EngineArgs.
        for key in _ENGINE_KWARGS:
            if self.config.get(key) is not None:
                kwargs[key] = self.config[key]
        self._llm = LLM(**kwargs)
        self._tokenizer = self._llm.get_tokenizer()
        from deepinfer.sampling_params import SamplingParams

        self._sampling_cls = SamplingParams
        self._start_worker()

    def _start_worker(self) -> None:
        """Start the batching worker that feeds the engine."""
        self._worker = threading.Thread(target=self._run, name="deepinfer-batch", daemon=True)
        self._worker.start()

    def close(self) -> None:
        self._stop = True
        with self._wake:
            self._wake.notify_all()
        if self._worker is not None:
            self._worker.join(timeout=30)
        if self._llm is not None:
            try:
                self._llm.close()
            except Exception:  # noqa: BLE001 - closing must never raise
                pass

    # ------------------------------------------------------------------
    # the batching loop
    # ------------------------------------------------------------------
    def _run(self) -> None:
        while not self._stop:
            with self._wake:
                while not self._queue and not self._stop:
                    self._wake.wait(timeout=0.5)
                if self._stop:
                    return
                deadline = time.time() + BATCH_WINDOW_S
                while len(self._queue) < MAX_BATCH and time.time() < deadline:
                    self._wake.wait(timeout=max(0.0, deadline - time.time()))
                batch = self._queue[:MAX_BATCH]
                del self._queue[:len(batch)]
            self._serve(batch)

    def _serve(self, batch: list[PendingRequest]) -> None:
        """Run one batch through the engine."""
        try:
            prompts = [request.prompt for request in batch]
            params = [self._make_sampling(request.sampling) for request in batch]
            outputs = self._llm.generate(prompts, params, use_tqdm=False)
        except Exception as exc:  # noqa: BLE001 - the error goes to the client
            for request in batch:
                request.error = f"{type(exc).__name__}: {exc}"
                request.event.set()
            return
        for request, output in zip(batch, outputs, strict=False):
            request.result = [output] if isinstance(output, str) else list(output)
            request.event.set()

    def _make_sampling(self, sampling: dict[str, Any]) -> Any:
        """Build the engine's sampling parameters for one request."""
        if self._sampling_cls is None:
            raise RuntimeError("the DeepInfer engine is not loaded")
        return self._sampling_cls(**_sampling_kwargs(sampling))

    def submit(self, request: PendingRequest) -> None:
        """Queue a request and wait for its result."""
        with self._wake:
            self._queue.append(request)
            self._wake.notify_all()
        request.event.wait()
        if request.error:
            raise RuntimeError(request.error)
        return request.result

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def render_chat(self, messages: list[dict[str, Any]]) -> str:
        """Apply the model's chat template to a message list."""
        if self._tokenizer is None:
            raise RuntimeError("the engine is not loaded")
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


def _sampling_kwargs(sampling: dict[str, Any]) -> dict[str, Any]:
    """Map the OpenAI sampling fields onto DeepInfer's SamplingParams."""
    allowed = (
        "temperature", "top_p", "top_k", "max_tokens", "presence_penalty",
        "frequency_penalty", "repetition_penalty", "seed", "stop", "min_p",
        "logprobs", "skip_special_tokens", "ignore_eos", "include_stop_str_in_output",
    )
    out: dict[str, Any] = {}
    for key, value in (sampling or {}).items():
        if key in allowed and value is not None:
            out[key] = value
    return out


def _create_app(engine: Engine) -> Any:
    """The FastAPI application."""
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse, StreamingResponse

    app = FastAPI(title="DeepInfer facade", version="1.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "loaded": engine._llm is not None,
            "model": engine.served_model_name,
            "max_context": engine.ctx,
            "service": "deepinfer",
            "accelerator": os.environ.get("DEEPINFER_ACCELERATOR", ""),
        }

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [{
                "id": engine.served_model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "deepseek",
            }],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(body: dict[str, Any]) -> Any:
        messages = body.get("messages") or []
        if not messages:
            raise HTTPException(400, "messages must not be empty")
        try:
            prompt = engine.render_chat(messages)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, f"cannot render the chat template: {exc}") from exc

        request = PendingRequest(prompt=prompt, sampling=dict(body))
        stream = bool(body.get("stream"))

        if not stream:
            texts = engine.submit(request)
            text = texts[0] if texts else ""
            return {
                "id": request.request_id,
                "object": "chat.completion",
                "created": request.created,
                "model": engine.served_model_name,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }

        def event_stream() -> Any:
            texts = engine.submit(request)
            text = texts[0] if texts else ""
            first = {
                "id": request.request_id,
                "object": "chat.completion.chunk",
                "created": request.created,
                "model": engine.served_model_name,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(first)}\n\n"
            body_chunk = {
                "id": request.request_id,
                "object": "chat.completion.chunk",
                "created": request.created,
                "model": engine.served_model_name,
                "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(body_chunk)}\n\n"
            last = {
                "id": request.request_id,
                "object": "chat.completion.chunk",
                "created": request.created,
                "model": engine.served_model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(last)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.post("/v1/completions")
    def completions(body: dict[str, Any]) -> Any:
        prompt = body.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise HTTPException(400, "prompt must be a non-empty string")
        request = PendingRequest(prompt=prompt, sampling=dict(body))
        texts = engine.submit(request)
        text = texts[0] if texts else ""
        return {
            "id": request.request_id,
            "object": "text_completion",
            "created": request.created,
            "model": engine.served_model_name,
            "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    @app.exception_handler(RuntimeError)
    def runtime_error_handler(_request: Any, exc: RuntimeError) -> Any:
        return JSONResponse(status_code=500, content={"error": {"message": str(exc)}})

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--config", required=True, help="the facade config (JSON)")
    ap.add_argument("--port", type=int, default=0, help="the listen port")
    args = ap.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.port:
        config["port"] = args.port
    port = int(config.get("port") or 8080)

    # The DeepInfer checkout and the AMD overlay must be importable.
    for extra in (
        str(REPO_ROOT / "vendor"),
        str(REPO_ROOT / "vendor" / "deepinfer"),
        str(REPO_ROOT),
    ):
        if extra not in sys.path:
            sys.path.insert(0, extra)

    engine = Engine(config)
    print(f"deepinfer facade: loading {engine.model_path} ...", flush=True)
    engine.load()
    print(f"deepinfer facade: listening on 127.0.0.1:{port}", flush=True)

    import uvicorn

    try:
        uvicorn.run(_create_app(engine), host="127.0.0.1", port=port, log_level="warning")
    finally:
        engine.close()


if __name__ == "__main__":
    main()
