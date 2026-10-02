#!/usr/bin/env python
"""Render a model-page results section from a capture manifest.

    ~/Desktop/work/.venv-rocm/bin/python experiments/cascade/bench_report.py \
        --run scion-bench-test

Reads ``results/<run>/manifest.json`` and writes ``MODEL-CARD.md`` next to it
(provenance, per-benchmark table, methodology, caveats).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CASCADE = Path(__file__).resolve().parents[2] / "experiments" / "cascade"


def render(manifest: dict) -> str:
    gen = manifest["generator"]
    tasks = manifest["tasks"]
    decode = manifest["decode"]
    aggregates = manifest["aggregates"]
    by_bench = aggregates["by_benchmark"]
    lines = [
        "# Benchmark results — Scion-35B-A3B v2 (ternary PQ2_0)",
        "",
        "Run with the local hivebench cascade harness; checked programmatically.",
        "",
        "## Provenance",
        "",
        "| field | value |",
        "|---|---|",
        f"| run | `{manifest['run']}` ({manifest['created']}) |",
        f"| model | `{gen['model_path']}` |",
        f"| model sha256 | `{gen['model_sha256'][:16]}…` |",
        f"| stack / serving | `{gen['stack']}`, {gen.get('total_slots')} slots, "
        f"ctx {gen.get('n_ctx')} |",
        f"| git | `{manifest['git']['commit'][:12]}` "
        f"({manifest['git']['dirty_files']} dirty files) |",
        f"| task set | `{manifest['tasks']['name']}` "
        f"`{tasks['sha256'][:16]}…`, split `{tasks['split']}` |",
        f"| decode | T={decode['temperature']}, max_tokens={decode['max_tokens']}, "
        f"{decode['samples']} samples/task |",
        "",
        "## Results",
        "",
        "| benchmark | tasks | samples | sample acc | task macro | mean ms | mean tokens |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for benchmark, entry in by_bench.items():
        lines.append(
            f"| {benchmark} | {entry['tasks']} | {entry['samples']} | "
            f"{entry['sample_accuracy']:.3f} | {entry['task_macro']:.3f} | "
            f"{entry['mean_ms']:.0f} | {entry['mean_tokens']:.0f} |"
        )
    overall = aggregates["overall"]
    lines += [
        f"| **overall** | {overall['tasks']} | {overall['samples']} | "
        f"**{overall['sample_accuracy']:.3f}** | **{overall['task_macro']:.3f}** | — | — |",
        "",
        "`sample acc` pools all samples; `task macro` averages per-task pass rates "
        "(equal weight per task).",
        "",
        "## Protocol",
        "",
        f"- System prompt: `{decode['system']}`",
        "- Temperature > 0 means samples are stochastic; report both pooled and "
        "macro accuracy, not a single number.",
        "- Grading: " + manifest.get("checker_notes", ""),
        "",
        "## Caveats",
        "",
        "- Checker-based grading undercounts correct answers in unusual formats "
        "(especially LaTeX in MATH).",
        "- HumanEval solutions execute the canonical tests in a local sandbox "
        "(15 s timeout); no hidden tests.",
        "- TruthfulQA is MC1 only.",
        "- Results are the ternary-local generator; see the cascade report for "
        "escalation economics.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="", help="run name under results/ (default: newest)")
    parser.add_argument("--results", default=str(CASCADE / "results"))
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    root = Path(args.results)
    if args.run:
        run_dir = root / args.run
    else:
        candidates = sorted(
            (p for p in root.iterdir() if (p / "manifest.json").is_file()),
            key=lambda p: p.stat().st_mtime,
        )
        if not candidates:
            raise SystemExit(f"no manifests under {root}")
        run_dir = candidates[-1]
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    text = render(manifest)
    out = Path(args.out) if args.out else run_dir / "MODEL-CARD.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print("\nartifact:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
