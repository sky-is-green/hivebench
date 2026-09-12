"""HiveBench model speed probe.

Cycles through every model loaded by LM Studio (or any OpenAI-compatible server),
probes each with a tiny streaming completion, and reports per model:

- reachable / loadable
- time-to-first-token (TTFT, ms) — how quickly the model *starts* answering
- decode speed (tokens/sec) from ``usage.completion_tokens`` when the server
  returns usage (LM Studio does on the final stream chunk)
- visible reply length, and **empty-reply / reasoning-burn detection** (a model
  that burns its whole output budget on chain-of-thought despite
  ``enable_thinking=false`` produces an empty visible reply with many completion
  tokens — the qwen MoE family in LM Studio does exactly this)
- stall detection (no token within the stream timeout)

Exit code: ``0`` if every probed model answered with visible content, ``1`` if
any model failed or returned an empty reply, ``2`` if the server is unreachable.

Usage::

    python -m experiments.model_probe                 # probe all loaded models
    python -m experiments.model_probe --model qwen3.6  # substring filter
    python -m experiments.model_probe --max-tokens 32 --no-thinking
    python -m experiments.model_probe --json runs/probe.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Iterable, Optional

import requests

from backend.openai_compat import DEFAULT_BASE_URL, resolve_endpoint
from backend.providers import apply_provider_overrides, load_registry
from backend.sampling import parse_sampling
from experiments.dashboard import KeepAwake

PROBE_PROMPT = "Reply with the single word: ok"
DEFAULT_MAX_TOKENS = 64
REQUEST_TIMEOUT = 180  # s: first load of a large model + generation can be slow
STALL_TIMEOUT = 45     # s without any token before the probe is declared stalled
MODELS_TIMEOUT = 10    # s for the /v1/models listing


@dataclass
class ProbeResult:
    """Outcome of a single model probe."""

    model: str
    status: str = "PASS"  # PASS | EMPTY | FAIL
    ttft_ms: Optional[float] = None
    decode_tps: Optional[float] = None
    effective_tps: Optional[float] = None
    completion_tokens: int = 0
    reply_len: int = 0
    error: str = ""
    note: str = ""
    duration_ms: float = 0.0
    meta: dict = field(default_factory=dict)


def _list_models(base_url: str, http=None) -> list[str]:
    """Return the sorted model ids advertised by the server's /v1/models."""
    http = http or requests
    resp = http.get(f"{base_url}/v1/models", timeout=MODELS_TIMEOUT)
    resp.raise_for_status()
    ids = [m.get("id", "") for m in resp.json().get("data", [])]
    return sorted(i for i in ids if i)


def _post_stream(base_url: str, payload: dict, timeout: int, http=None):
    """POST chat/completions with streaming; yield parsed JSON chunks.

    Follows the SSE framing LM Studio / llama.cpp emit: lines ``data: {...}``
    with a terminating ``data: [DONE]``. Kept generator-based so tests can feed
    synthetic lines.
    """
    http = http or requests
    resp = http.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        stream=True,
        timeout=timeout,
    )
    resp.raise_for_status()
    for raw in resp.iter_lines(decode_unicode=True):
        if raw is None:
            continue
        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if not data or data == "[DONE]":
            return
        try:
            yield json.loads(data)
        except json.JSONDecodeError:
            continue


def probe_model(
    base_url: str,
    model: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    disable_thinking: bool = True,
    sampling: dict | None = None,
    timeout: int = REQUEST_TIMEOUT,
    stall_timeout: float = STALL_TIMEOUT,
    http=None,
) -> ProbeResult:
    """Probe one model with a tiny streaming completion and measure speed.

    ``sampling`` carries experimenter sampling overrides (backend.sampling
    fields; temperature 0.0 is the probe default, overridable). ``http`` is
    injectable for tests and must expose ``get(url, timeout=)`` and
    ``post(url, json=, stream=, timeout=)``.
    """
    started = time.monotonic()
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a speed probe. Be concise."},
            {"role": "user", "content": PROBE_PROMPT},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if sampling:
        payload.update(sampling)
    if disable_thinking:
        payload["enable_thinking"] = False

    ttft = None
    ttft_time = None  # monotonic() timestamp when the first content token arrived
    content_parts: list[str] = []
    completion_tokens = 0
    first = time.monotonic()
    try:
        for chunk in _post_stream(base_url, payload, timeout, http=http):
            choices = chunk.get("choices") or []
            if choices and choices[0].get("delta", {}).get("content"):
                if ttft is None:
                    ttft = (time.monotonic() - first) * 1000.0
                    ttft_time = time.monotonic()
                content_parts.append(choices[0]["delta"]["content"])
            usage = chunk.get("usage")
            if usage:
                completion_tokens = usage.get("completion_tokens", 0) or 0
            # stall guard: no content within stall_timeout of the request start
            if ttft is None and (time.monotonic() - started) > stall_timeout:
                break
    except requests.RequestException as exc:
        dur = (time.monotonic() - started) * 1000.0
        detail = ""
        resp = getattr(exc, "response", None)
        if resp is not None:
            try:
                detail = resp.json().get("error", {}).get("message", "")
            except Exception:  # noqa: BLE001 - body may not be JSON
                detail = resp.text[:300]
        return ProbeResult(
            model=model, status="FAIL",
            error=str(exc) + (f": {detail}" if detail else ""),
            duration_ms=dur,
        )
    except Exception as exc:  # noqa: BLE001 - any probe failure is reported, not raised
        dur = (time.monotonic() - started) * 1000.0
        return ProbeResult(
            model=model, status="FAIL", error=str(exc), duration_ms=dur
        )

    dur = (time.monotonic() - started) * 1000.0
    reply = "".join(content_parts)
    # time.monotonic() is in seconds.
    # effective_tps = tokens / whole request (load + prefill + decode) — the real
    # "how usable is this model" number.
    effective_s = max(time.monotonic() - first, 1e-6)
    effective_tps = completion_tokens / effective_s if completion_tokens else None
    # decode_tps = tokens / first-token->end span — pure generation speed,
    # excluding the one-time model load/prefill cost.
    if completion_tokens and ttft_time is not None:
        decode_s = max(time.monotonic() - ttft_time, 1e-6)
        decode_tps = completion_tokens / decode_s
    else:
        decode_tps = None

    result = ProbeResult(
        model=model,
        ttft_ms=round(ttft, 1) if ttft is not None else None,
        decode_tps=round(decode_tps, 1) if decode_tps is not None else None,
        effective_tps=round(effective_tps, 1) if effective_tps is not None else None,
        completion_tokens=completion_tokens,
        reply_len=len(reply.strip()),
        duration_ms=round(dur, 1),
        meta={"used_stream_usage": completion_tokens > 0},
    )

    if not reply.strip():
        result.status = "EMPTY"
        if completion_tokens >= max_tokens:
            result.note = (
                "empty visible reply but completion_tokens hit the cap: "
                "model likely burned its budget on reasoning "
                "(enable_thinking=false ignored)"
            )
        else:
            result.note = "server returned no visible content"
    return result


def _print_table(results: list[ProbeResult]) -> None:
    header = (f"{'MODEL':<42} {'STATUS':<6} {'TTFT ms':>8} "
              f"{'eff tps':>8} {'dec tps':>8} {'tok':>5} {'len':>4}  NOTES")
    print(header)
    print("-" * len(header))
    for r in results:
        notes = r.error if r.error else r.note
        print(
            f"{r.model:<42} {r.status:<6} "
            f"{str(r.ttft_ms or '-'):>8} {str(r.effective_tps or '-'):>8} "
            f"{str(r.decode_tps or '-'):>8} {str(r.completion_tokens):>5} "
            f"{str(r.reply_len):>4}  {notes}"
        )


def run_probe(
    base_url: str,
    model_filter: str = "",
    max_tokens: int = DEFAULT_MAX_TOKENS,
    disable_thinking: bool = True,
    keep_awake: bool = True,
    sampling: dict | None = None,
    http=None,
) -> tuple[list[ProbeResult], int]:
    """Probe all (filtered) models; returns (results, exit_code)."""
    base_url = resolve_endpoint(base_url)
    try:
        models = _list_models(base_url, http=http)
    except requests.RequestException as exc:
        print(f"error: cannot list models from {base_url}: {exc}")
        return [], 2
    if model_filter:
        models = [m for m in models if model_filter.lower() in m.lower()]
    if not models:
        print("error: no models match the filter")
        return [], 2

    ka = KeepAwake() if (keep_awake and sys.platform == "win32") else None
    try:
        for model in models:
            r = probe_model(
                base_url, model, max_tokens=max_tokens,
                disable_thinking=disable_thinking, sampling=sampling,
                http=http,
            )
            results.append(r)
    finally:
        if ka is not None:
            ka.close()

    _print_table(results)
    n_pass = sum(1 for r in results if r.status == "PASS")
    n_empty = sum(1 for r in results if r.status == "EMPTY")
    n_fail = sum(1 for r in results if r.status == "FAIL")
    print(
        f"\n{len(results)} models: {n_pass} PASS, {n_empty} EMPTY, "
        f"{n_fail} FAIL"
    )
    exit_code = 0 if n_fail == 0 and n_empty == 0 else 1
    return results, exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cycle every loaded model with a tiny streaming probe and "
                    "report TTFT, decode tps, and failures.",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                        help="OpenAI-compatible server base URL")
    parser.add_argument("--model", default="",
                        help="only probe models whose id contains this substring")
    parser.add_argument("--provider", default="",
                        help="provider name from the providers config: fills "
                             "base_url (explicit --base-url wins)")
    parser.add_argument("--providers-file", default="",
                        help="path to the providers JSON config")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help="reply ceiling for the probe")
    parser.add_argument(
        "--sampling", default="",
        help="sampling overrides as JSON, e.g. --sampling "
             "'{\"temperature\":0.7,\"repeat_penalty\":1.1}' (backend.sampling "
             "fields; the probe default is temperature 0.0)",
    )
    parser.add_argument("--no-thinking", action="store_true", default=True,
                        help="send enable_thinking=false (default on)")
    parser.add_argument("--keep-awake", action="store_true", default=True,
                        help="prevent OS sleep during the sweep (default on)")
    parser.add_argument("--json", default="",
                        help="write the full results as JSON to this path")
    args = parser.parse_args(argv)

    if args.provider:
        try:
            prov = load_registry(args.providers_file or None).resolve(args.provider)
        except LookupError as exc:
            print(f"error: {exc}")
            return 2
        apply_provider_overrides(vars(parser.parse_args([])), args, prov)
        print(f"provider: {prov.name} -> {args.base_url}")

    results, exit_code = run_probe(
        args.base_url,
        model_filter=args.model,
        max_tokens=args.max_tokens,
        disable_thinking=args.no_thinking,
        sampling=parse_sampling(args.sampling) if args.sampling else None,
    )
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(
                {"base_url": resolve_endpoint(args.base_url),
                 "results": [asdict(r) for r in results]},
                fh, indent=2, default=str,
            )
        print(f"wrote {args.json}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())