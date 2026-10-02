#!/usr/bin/env python
"""Capture generator streams over a task set, with checkers and provenance.

Serves two purposes at once:

1. **Dataset asset** — full streams (reasoning + answer + timing), K samples
   per task, checker outcomes, for router/steerer training and calibration.
2. **Publishable results** — every run writes a manifest with the model sha,
   stack, git commit, task-set hash, decode settings, per-task/per-benchmark
   results and caveats, plus a `MODEL-CARD.md` that can be posted on the model
   page (`bench_report.py` renders it from the manifest).

Artifacts under ``experiments/cascade/results/<name>/``:

    capture.json   first sample per task, legacy shape (prefix probes)
    streams.json   every sample, full chunk trace
    manifest.json  provenance + per-task results + aggregates
    MODEL-CARD.md  generated results section for the model page

Run (Scion stack applied through hivebench):

    .venv/bin/python experiments/cascade/stream_capture.py \
        --tasks experiments/cascade/tasks-bench.json --split test \
        --samples 3 --temperature 0.7 --name scion-bench-test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import requests

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.cascade.checkers import check  # noqa: E402
from experiments.cascade.run_cascade import SCION_SYSTEM  # noqa: E402

CASCADE = REPO / "experiments" / "cascade"
SCION_BASE = os.environ.get(
    "CASCADE_SCION_BASE",
    f"http://127.0.0.1:{os.environ.get('CASCADE_SCION_PORT', '1234')}/v1",
)

CHECKER_NOTES = (
    "gsm8k: numeric match on the final number; "
    "math: normalized LaTeX/\\boxed{} match with numeric+sympy fallback; "
    "humaneval: sandboxed execution of the canonical test harness; "
    "truthfulqa: multiple-choice letter (MC1)."
)


def sha256_file(path: Path, cache_path: Path) -> str:
    """File sha256 with a small path/mtime/size cache (GGUFs are GBs)."""
    stat = path.stat()
    key = f"{path}|{stat.st_size}|{int(stat.st_mtime)}"
    cache: dict[str, str] = {}
    if cache_path.is_file():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except ValueError:
            cache = {}
    if key in cache:
        return cache[key]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    cache[key] = digest.hexdigest()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    return cache[key]


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=REPO, capture_output=True, text=True, timeout=10
            ).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""

    dirty = [line for line in run("status", "--porcelain").splitlines() if line]
    return {"commit": run("rev-parse", "HEAD"), "dirty_files": len(dirty)}


def stream_once(base: str, prompt: str, *, temperature: float, max_tokens: int) -> dict[str, Any]:
    """One streaming completion; returns the delta trace with arrival times."""
    payload = {
        "model": "scion",
        "messages": [
            {"role": "system", "content": SCION_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
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
        f"{base}/chat/completions", json=payload, stream=True, timeout=(15, 1200)
    ) as resp:
        if resp.status_code >= 400:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        resp.encoding = "utf-8"  # llama-server sends utf-8 without a charset
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
                    {
                        "t_ms": round((time.time() - started) * 1000.0, 1),
                        "kind": kind,
                        "text": text,
                    }
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


def task_accuracy(rows: list[dict]) -> dict[str, float]:
    sample_ok = [1.0 if s["ok"] else 0.0 for r in rows for s in r["samples"]]
    per_task = [sum(1.0 for s in r["samples"] if s["ok"]) / len(r["samples"]) for r in rows]
    return {
        "tasks": len(rows),
        "samples": len(sample_ok),
        "sample_accuracy": round(sum(sample_ok) / len(sample_ok), 4) if sample_ok else 0.0,
        "task_macro": round(sum(per_task) / len(per_task), 4) if per_task else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=SCION_BASE)
    parser.add_argument("--tasks", default=str(CASCADE / "tasks-bench.json"))
    parser.add_argument("--split", choices=("train", "test", "all"), default="test")
    parser.add_argument("--benchmark", default="", help="limit to one benchmark")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=1536)
    parser.add_argument("--stack", default="scion-35b-cascade")
    parser.add_argument("--name", default=time.strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--out", default=str(CASCADE / "results"))
    args = parser.parse_args()

    try:
        props = requests.get(
            args.base.replace("/v1", "") + "/props", timeout=10
        ).json()
    except Exception:  # noqa: BLE001
        props = {}

    spec = json.loads(Path(args.tasks).read_text(encoding="utf-8"))
    tasks = [
        t
        for t in spec["tasks"]
        if (args.split == "all" or t.get("split") == args.split)
        and (not args.benchmark or t.get("benchmark") == args.benchmark)
    ]
    if args.limit:
        tasks = tasks[: args.limit]
    if not tasks:
        print("no tasks selected", file=sys.stderr)
        return 2

    model_path = Path(str(props.get("model_path") or ""))
    model_sha = (
        sha256_file(model_path, CASCADE / "results" / ".model-sha-cache.json")
        if model_path.is_file()
        else ""
    )
    tasks_bytes = Path(args.tasks).read_bytes()
    provenance = {
        "run": args.name,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "host": os.uname().nodename,
        "git": git_state(),
        "generator": {
            "stack": args.stack,
            "base": args.base,
            "model_path": str(model_path),
            "model_sha256": model_sha,
            "total_slots": props.get("total_slots"),
            "n_ctx": props.get("default_generation_settings", {}).get("n_ctx"),
            "build": props.get("build_info"),
        },
        "tasks": {
            "file": str(Path(args.tasks).resolve()),
            "name": spec.get("name"),
            "sha256": hashlib.sha256(tasks_bytes).hexdigest(),
            "split": args.split,
            "selected": len(tasks),
            "by_benchmark": {
                b: sum(1 for t in tasks if t.get("benchmark") == b)
                for b in sorted({t.get("benchmark", "?") for t in tasks})
            },
        },
        "decode": {
            "samples": args.samples,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "system": SCION_SYSTEM,
        },
        "checker_notes": CHECKER_NOTES,
    }

    results: list[dict[str, Any]] = []
    started = time.time()
    for index, task in enumerate(tasks, 1):
        samples = []
        for _ in range(args.samples):
            out = stream_once(
                args.base,
                task["prompt"],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            samples.append(
                {
                    "ok": bool(check(task, out["content"])),
                    "content": out["content"],
                    "reasoning": out["reasoning"],
                    "chunks": out["chunks"],
                    "usage": out["usage"],
                    "finish_reason": out["finish_reason"],
                    "ms": out["ms"],
                }
            )
        results.append(
            {
                "id": task["id"],
                "benchmark": task.get("benchmark"),
                "bucket": task.get("bucket"),
                "split": task.get("split"),
                "prompt": task["prompt"],
                "checker": task["checker"],
                "samples": samples,
            }
        )
        passed = sum(1 for s in samples if s["ok"])
        print(
            f"[{index}/{len(tasks)}] {task['id']:16} {passed}/{args.samples}  "
            f"({time.time() - started:.0f}s)",
            file=sys.stderr,
            flush=True,
        )

    aggregates: dict[str, Any] = {"overall": task_accuracy(results), "by_benchmark": {}}
    for benchmark in sorted({r["benchmark"] for r in results}):
        rows = [r for r in results if r["benchmark"] == benchmark]
        entry = task_accuracy(rows)
        entry["mean_ms"] = round(
            sum(s["ms"] for r in rows for s in r["samples"])
            / sum(len(r["samples"]) for r in rows),
            1,
        )
        entry["mean_tokens"] = round(
            sum(
                float((s["usage"] or {}).get("completion_tokens") or 0)
                for r in rows
                for s in r["samples"]
            )
            / sum(len(r["samples"]) for r in rows),
            1,
        )
        aggregates["by_benchmark"][benchmark] = {**entry, "split": rows[0]["split"]}

    out_dir = Path(args.out) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        **provenance,
        "aggregates": aggregates,
        "results": [
            {
                **row,
                "samples": [{k: v for k, v in s.items() if k != "chunks"} for s in row["samples"]],
            }
            for row in results
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (out_dir / "streams.json").write_text(
        json.dumps({"tasks": {r["id"]: r for r in results}}, indent=1), encoding="utf-8"
    )
    # legacy shape (first sample per task) for the prefix/steerer probes
    (out_dir / "capture.json").write_text(
        json.dumps(
            {
                "base": args.base,
                "max_tokens": args.max_tokens,
                "tasks": len(results),
                "records": {
                    r["id"]: {
                        "prompt": r["prompt"],
                        "bucket": r["bucket"],
                        "content": r["samples"][0]["content"],
                        "reasoning": r["samples"][0]["reasoning"],
                        "chunks": r["samples"][0]["chunks"],
                        "usage": r["samples"][0]["usage"],
                        "finish_reason": r["samples"][0]["finish_reason"],
                        "ms": r["samples"][0]["ms"],
                    }
                    for r in results
                },
            },
            indent=1,
        ),
        encoding="utf-8",
    )

    index_path = Path(args.out) / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else []
    index = [row for row in index if row.get("run") != args.name] + [
        {
            "run": args.name,
            "created": provenance["created"],
            "model_sha256": model_sha,
            "tasks_sha256": provenance["tasks"]["sha256"],
            "split": args.split,
            "aggregates": aggregates,
        }
    ]
    index_path.write_text(json.dumps(index, indent=1), encoding="utf-8")

    print(f"\n=== {args.name} ===")
    for benchmark, entry in aggregates["by_benchmark"].items():
        print(
            f"{benchmark:12} {entry['tasks']:>3} tasks  sample_acc {entry['sample_accuracy']:.3f}  "
            f"task_macro {entry['task_macro']:.3f}  {entry['mean_ms']:.0f} ms"
        )
    print("overall:", json.dumps(aggregates["overall"]))
    print("artifact:", out_dir / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
