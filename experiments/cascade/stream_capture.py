#!/usr/bin/env python
"""Capture Scion's full generation stream (reasoning + answer, with timing).

The pilot reports store only the final `content`; Scion's reasoning is the bulk
of the stream, and the reasoning is exactly what a mid-generation judge would
consume.  This records the real stream — every SSE delta with its arrival time
— so the prefix/steering evals run on what the generator actually emits, not on
the stripped final answer.

Writes ``experiments/cascade/streams/<name>/capture.json``:

    {task: {prompt, bucket, system, chunks: [{t_ms, kind, text}], content,
            reasoning, usage, finish_reason, ms, tokens}}

Run (Scion stack applied through hivebench):

    .venv/bin/python experiments/cascade/stream_capture.py --limit 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import requests

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.cascade.history import load_tasks  # noqa: E402
from experiments.cascade.run_cascade import SCION_SYSTEM  # noqa: E402

CASCADE = REPO / "experiments" / "cascade"
SCION_BASE = os.environ.get(
    "CASCADE_SCION_BASE",
    f"http://127.0.0.1:{os.environ.get('CASCADE_SCION_PORT', '1234')}/v1",
)


def stream_once(base: str, prompt: str, *, max_tokens: int) -> dict[str, Any]:
    """One streaming completion; returns the delta trace with arrival times."""
    payload = {
        "model": "scion",
        "messages": [
            {"role": "system", "content": SCION_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.time()
    chunks: list[dict[str, Any]] = []
    content: list[str] = []
    reasoning: list[str] = []
    usage: Optional[dict] = None
    finish: Optional[str] = None
    with requests.post(
        f"{base}/chat/completions", json=payload, stream=True, timeout=(15, 900)
    ) as resp:
        if resp.status_code >= 400:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except ValueError:
                continue
            if event.get("usage"):
                usage = event["usage"]
            choices = event.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
            delta = choice.get("delta") or {}
            for kind, key in (("reasoning", "reasoning_content"), ("content", "content")):
                text = delta.get(key) or ""
                if not text:
                    continue
                chunks.append(
                    {"t_ms": round((time.time() - started) * 1000.0, 1), "kind": kind, "text": text}
                )
                (reasoning if kind == "reasoning" else content).append(text)
    return {
        "chunks": chunks,
        "content": "".join(content),
        "reasoning": "".join(reasoning),
        "usage": usage,
        "finish_reason": finish,
        "ms": round((time.time() - started) * 1000.0, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=SCION_BASE)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=1536)
    parser.add_argument("--name", default=time.strftime("%Y%m%d-%H%M%S"))
    parser.add_argument(
        "--out", default=str(CASCADE / "streams"), help="streams root"
    )
    args = parser.parse_args()

    try:
        models = requests.get(f"{args.base}/models", timeout=5)
        if models.status_code != 200:
            raise RuntimeError(models.text[:200])
    except Exception as exc:  # noqa: BLE001
        print(f"Scion not reachable at {args.base}: {exc}", file=sys.stderr)
        return 2

    tasks = list(load_tasks().values())
    if args.limit:
        tasks = tasks[: args.limit]

    records: dict[str, dict] = {}
    for index, task in enumerate(tasks, 1):
        out = stream_once(args.base, task["prompt"], max_tokens=args.max_tokens)
        records[task["id"]] = {"prompt": task["prompt"], "bucket": task["bucket"], **out}
        n_reason = sum(1 for c in out["chunks"] if c["kind"] == "reasoning")
        n_content = len(out["chunks"]) - n_reason
        print(
            f"[{index}/{len(tasks)}] {task['id']:26} {out['ms']:>7.0f} ms  "
            f"reason_deltas={n_reason:>4} content_deltas={n_content:>4} "
            f"finish={out['finish_reason']}",
            file=sys.stderr,
            flush=True,
        )

    out_dir = Path(args.out) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "base": args.base,
        "max_tokens": args.max_tokens,
        "tasks": len(records),
        "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "records": records,
    }
    (out_dir / "capture.json").write_text(json.dumps(artifact, indent=1), encoding="utf-8")
    print("artifact:", out_dir / "capture.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
