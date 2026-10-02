#!/usr/bin/env python
"""Live cancel-mechanics probe — what can the harness do to a live generation?

The judgement broker can *decide* to cancel; this measures what the serving
layer can *do* with that decision, against the fork llama-server that serves
Scion.  The ground truth is the server's own ``tokens_predicted_total`` counter
(``--metrics``), not client-side timing, plus a follow-up request on a
single-slot server (``-np 1``) whose latency is the time the slot stayed busy.

Three phases:

1. ``control``    — a stream is left to finish; the follow-up fired right after
   the first delta must wait for it.  Proves occupancy is observable.
2. ``disconnect`` — a plain streaming request is closed mid-generation (the
   presumed abort path: httplib ``is_connection_closed`` -> ``should_stop``).
   Token growth after close = generation that survived the abort.
3. ``resumable``  — a stream opened with ``X-Conversation-Id`` is stopped with
   ``DELETE /v1/stream?conv_id=...``.  The fork's own comment says this cancels
   the *pipe* but not the generation, which keeps running to EOS; token growth
   after DELETE measures the waste.

Writes ``experiments/cascade/stream-probe.json``.

    .venv/bin/python experiments/cascade/stream_probe.py --base http://127.0.0.1:1235/v1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import requests

REPO = Path(__file__).resolve().parents[2]
CASCADE = REPO / "experiments" / "cascade"

PROMPT = (
    "A train leaves Station A at 60 km/h. A second train leaves Station B, "
    "180 km away, at 80 km/h toward Station A at the same time. How long "
    "until they meet? Think step by step, then give the final answer."
)

KEYS = (
    "llamacpp:tokens_predicted_total",
    "llamacpp:requests_processing",
    "llamacpp:requests_deferred",
)


def root_of(base: str) -> str:
    """The server root (metrics lives at /metrics, not under /v1)."""
    return base[:-3] if base.endswith("/v1") else base


def metrics(base: str) -> Optional[dict[str, float]]:
    try:
        text = requests.get(f"{root_of(base)}/metrics", timeout=5).text
    except Exception:  # noqa: BLE001
        return None
    if "llamacpp:" not in text:
        return None
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#") or " " not in line:
            continue
        name, _, value = line.rpartition(" ")
        try:
            out[name] = float(value)
        except ValueError:
            continue
    return out


def sample(base: str, seconds: float, interval: float = 0.2) -> list[dict[str, Any]]:
    """Poll the counters for ``seconds``; returns time-ordered samples."""
    started = time.time()
    trace: list[dict[str, Any]] = []
    while time.time() - started <= seconds:
        snap = metrics(base)
        if snap is not None:
            trace.append(
                {
                    "t_ms": round((time.time() - started) * 1000.0, 1),
                    **{key.split(":")[-1]: snap.get(key) for key in KEYS},
                }
            )
        time.sleep(interval)
    return trace


def start_stream(
    base: str, *, max_tokens: int, conv_id: Optional[str] = None
) -> requests.Response:
    headers = {"content-type": "application/json"}
    if conv_id:
        headers["X-Conversation-Id"] = conv_id
    payload = {
        "model": "scion",
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
    }
    resp = requests.post(
        f"{base}/chat/completions",
        json=payload,
        headers=headers,
        stream=True,
        timeout=(15, 300),
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
    return resp


def followup_ms(base: str) -> float:
    """A 1-token request; on a single-slot server it waits for the slot."""
    started = time.time()
    resp = requests.post(
        f"{base}/chat/completions",
        json={
            "model": "scion",
            "messages": [{"role": "user", "content": "Reply with the single word ok."}],
            "temperature": 0,
            "max_tokens": 1,
        },
        timeout=300,
    )
    ms = (time.time() - started) * 1000.0
    if resp.status_code >= 400:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
    return round(ms, 1)


def spawn_followup(base: str) -> tuple[threading.Thread, dict]:
    box: dict[str, Any] = {}
    thread = threading.Thread(target=lambda: box.update(ms=followup_ms(base)))
    thread.start()
    return thread, box


def read_stream(
    resp: requests.Response,
    *,
    stop_after: int,
    max_wait_s: float,
    on_first: Optional[Any] = None,
) -> dict:
    """One-pass SSE read; ``on_first`` fires once, after the first delta."""
    started = time.time()
    deltas = 0
    first_ms: Optional[float] = None
    fired = False
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        if line[5:].strip() == "[DONE]":
            break
        try:
            event = json.loads(line[5:].strip())
        except ValueError:
            continue
        choices = event.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        if not (delta.get("content") or delta.get("reasoning_content")):
            continue
        deltas += 1
        if first_ms is None:
            first_ms = (time.time() - started) * 1000.0
        if on_first is not None and not fired:
            fired = True
            on_first()
        if deltas >= stop_after or time.time() - started >= max_wait_s:
            break
    return {
        "deltas": deltas,
        "first_delta_ms": round(first_ms, 1) if first_ms is not None else None,
        "read_ms": round((time.time() - started) * 1000.0, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:1235/v1")
    parser.add_argument("--abort-deltas", type=int, default=32)
    parser.add_argument("--sample-s", type=float, default=2.5)
    parser.add_argument("--out", default=str(CASCADE / "stream-probe.json"))
    args = parser.parse_args()

    base = args.base.rstrip("/")
    if requests.get(f"{base}/models", timeout=5).status_code != 200:
        print(f"server not reachable at {base}", file=sys.stderr)
        return 2
    if metrics(base) is None:
        print("server has no /metrics; run it with --metrics", file=sys.stderr)
        return 2
    result: dict[str, Any] = {"base": base, "abort_deltas": args.abort_deltas, "phases": {}}

    # 1) control — the follow-up must wait for a stream left to finish.
    resp = start_stream(base, max_tokens=96)
    thread2: Optional[threading.Thread] = None
    box2: dict[str, Any] = {}

    def on_first() -> None:
        nonlocal thread2, box2
        thread2, box2 = spawn_followup(base)

    read = read_stream(resp, stop_after=10_000, max_wait_s=120.0, on_first=on_first)
    resp.close()
    if thread2 is not None:
        thread2.join(timeout=300)
    result["phases"]["control"] = {
        "deltas": read["deltas"],
        "first_delta_ms": read["first_delta_ms"],
        "stream_read_ms": read["read_ms"],
        "followup_ms": box2.get("ms"),
    }
    print(
        f"control:    {read['deltas']} deltas in {read['read_ms']} ms, "
        f"follow-up waited {box2.get('ms')} ms"
    )

    # 2) disconnect — close a plain stream mid-generation; watch the counter.
    resp = start_stream(base, max_tokens=512)
    read = read_stream(resp, stop_after=args.abort_deltas, max_wait_s=30.0)
    before = metrics(base)
    resp.close()
    thread, box = spawn_followup(base)
    trace = sample(base, args.sample_s)
    thread.join(timeout=300)
    after = metrics(base)
    result["phases"]["disconnect"] = {
        "deltas_read": read["deltas"],
        "read_ms": read["read_ms"],
        "followup_ms": box.get("ms"),
        "tokens_before": (before or {}).get(KEYS[0]),
        "tokens_after": (after or {}).get(KEYS[0]),
        "tokens_generated_after_close": (
            ((after or {}).get(KEYS[0], 0) - (before or {}).get(KEYS[0], 0))
        ),
        "metrics_trace": trace,
    }
    print(
        f"disconnect: read {read['deltas']} deltas, follow-up {box.get('ms')} ms, "
        f"tokens after close +{result['phases']['disconnect']['tokens_generated_after_close']}"
    )

    # 3) resumable — DELETE /v1/stream; the fork says generation continues.
    conv = f"probe-{uuid.uuid4().hex[:10]}"
    resp = start_stream(base, max_tokens=256, conv_id=conv)
    read = read_stream(resp, stop_after=args.abort_deltas, max_wait_s=30.0)
    before = metrics(base)
    dele = requests.delete(f"{base}/stream", params={"conv_id": conv}, timeout=10)
    thread, box = spawn_followup(base)
    trace = sample(base, args.sample_s)
    thread.join(timeout=300)
    after = metrics(base)
    gone = requests.get(f"{base}/stream", params={"conv_id": conv}, timeout=5)
    result["phases"]["resumable"] = {
        "deltas_read": read["deltas"],
        "read_ms": read["read_ms"],
        "delete_status": dele.status_code,
        "replay_status": gone.status_code,
        "followup_ms": box.get("ms"),
        "tokens_before": (before or {}).get(KEYS[0]),
        "tokens_after": (after or {}).get(KEYS[0]),
        "tokens_generated_after_delete": (
            ((after or {}).get(KEYS[0], 0) - (before or {}).get(KEYS[0], 0))
        ),
        "metrics_trace": trace,
    }
    resp.close()
    print(
        f"resumable:  read {read['deltas']} deltas, DELETE {dele.status_code}, replay "
        f"{gone.status_code}, follow-up {box.get('ms')} ms, tokens after delete "
        f"+{result['phases']['resumable']['tokens_generated_after_delete']}"
    )

    Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("artifact:", args.out)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
