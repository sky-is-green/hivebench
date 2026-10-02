#!/usr/bin/env python
"""Build the publishable benchmark task set: GSM8K, MATH, HumanEval, TruthfulQA.

The hand-written 27-task sets proved the plumbing; they are too small and too
self-authored for claims.  This builder samples with a fixed seed, stratifies
by benchmark, assigns train/test splits, and emits prompts + checkers in the
schema the cascade runner already uses — plus the provenance a model-page
result needs (dataset ids, sampling seed, per-benchmark counts, task hashes).

    ~/Desktop/work/.venv-rocm/bin/python experiments/cascade/bench_tasks.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterator

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.cascade.checkers import _boxed  # noqa: E402

CASCADE = REPO / "experiments" / "cascade"

SAMPLE_COUNTS = {"gsm8k": 40, "math": 25, "humaneval": 20, "truthfulqa": 15}
SPLIT_TEST = 0.30
SEED = 0
MATH_SUBJECTS = (
    "algebra", "counting_and_probability", "geometry", "intermediate_algebra",
    "number_theory", "prealgebra", "precalculus",
)

CODE_PRELUDE = (
    "from typing import *\n"
    "import math\nimport collections\nimport heapq\nimport bisect\n"
    "import functools\nimport itertools\nimport re\nimport sys\n"
)


def _gsm8k(count: int, rng: random.Random) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset("openai/gsm8k", "main", split="test")
    picks = rng.sample(range(len(dataset)), count)
    for index in picks:
        row = dataset[index]
        expected = row["answer"].split("####")[-1].strip().replace(",", "")
        number = float(expected)
        yield {
            "source": "openai/gsm8k",
            "bucket": "reasoning",
            "prompt": (
                f"{row['question']}\n\nSolve the problem, showing your work. "
                "End your answer with 'Final answer: <number>'."
            ),
            "checker": {"type": "number", "expect": number},
            "meta": {"source_split": "test"},
        }


def _math(count: int, rng: random.Random) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset

    pools: list[list[dict[str, Any]]] = []
    for subject in MATH_SUBJECTS:
        dataset = load_dataset("EleutherAI/hendrycks_math", subject, split="test")
        pools.append([dict(row, subject=subject) for row in dataset])
    for pool in pools:
        rng.shuffle(pool)
    taken = 0
    for round_index in range(max(len(p) for p in pools)):
        for pool in pools:
            if taken >= count:
                return
            if round_index >= len(pool):
                continue
            row = pool[round_index]
            expected = _boxed(row["solution"])
            if not expected:
                continue
            taken += 1
            yield {
                "source": f"EleutherAI/hendrycks_math:{row['subject']}",
                "bucket": "reasoning",
                "prompt": (
                    f"{row['problem']}\n\nSolve the problem, showing your work. "
                    "Put the final answer in \\boxed{}."
                ),
                "checker": {"type": "boxed", "expect": expected},
                "meta": {
                    "source_split": "test",
                    "subject": row["subject"],
                    "level": row.get("level"),
                },
            }


def _humaneval(count: int, rng: random.Random) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset("openai/openai_humaneval", split="test")
    picks = rng.sample(range(len(dataset)), count)
    for index in picks:
        row = dataset[index]
        yield {
            "source": "openai/openai_humaneval",
            "bucket": "code",
            "prompt": (
                "Complete the following Python function. Return only the full "
                "implementation in a single ```python code block.\n\n"
                f"```python\n{row['prompt']}```"
            ),
            "checker": {
                "type": "code",
                "prelude": CODE_PRELUDE,
                "tests": f"{row['test']}\ncheck({row['entry_point']})",
                "footer": "print('PASS')",
            },
            "meta": {"source_split": "test", "entry_point": row["entry_point"]},
        }


def _truthfulqa(count: int, rng: random.Random) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    picks = rng.sample(range(len(dataset)), count)
    for index in picks:
        row = dataset[index]
        choices = list(row["mc1_targets"]["choices"])
        labels = list(row["mc1_targets"]["labels"])
        correct = labels.index(1)
        letters = [chr(ord("A") + i) for i in range(len(choices))]
        listing = "\n".join(f"{letter}. {choice}" for letter, choice in zip(letters, choices))
        yield {
            "source": "truthfulqa/truthful_qa:mc1",
            "bucket": "qa",
            "prompt": (
                f"{row['question']}\n\nChoices:\n{listing}\n\n"
                f"Answer with the letter of the correct choice (A-{letters[-1]})."
            ),
            "checker": {"type": "choice", "expect": letters[correct]},
            "meta": {"source_split": "validation"},
        }


BUILDERS = {
    "gsm8k": _gsm8k,
    "math": _math,
    "humaneval": _humaneval,
    "truthfulqa": _truthfulqa,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--out", default=str(CASCADE / "tasks-bench.json")
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    tasks: list[dict[str, Any]] = []
    for benchmark, count in SAMPLE_COUNTS.items():
        got = 0
        for item in BUILDERS[benchmark](count, rng):
            got += 1
            tasks.append({"id": f"{benchmark}-{got:04d}", "benchmark": benchmark, **item})
        print(f"{benchmark:12} {got} tasks", file=sys.stderr)

    # stratified train/test split per benchmark
    by_bench: dict[str, list[dict]] = {}
    for task in tasks:
        by_bench.setdefault(task["benchmark"], []).append(task)
    for benchmark, rows in by_bench.items():
        rng.shuffle(rows)
        n_test = max(1, round(len(rows) * SPLIT_TEST))
        for index, row in enumerate(rows):
            row["split"] = "test" if index < n_test else "train"

    payload = {
        "name": "cascade-bench-v1",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": args.seed,
        "samples": SAMPLE_COUNTS,
        "splits": {
            split: sum(1 for t in tasks if t["split"] == split)
            for split in ("train", "test")
        },
        "tasks": tasks,
    }
    body = json.dumps(payload, indent=1)
    out = Path(args.out)
    out.write_text(body, encoding="utf-8")
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
    print(
        f"wrote {out}  ({len(tasks)} tasks; "
        f"{payload['splits']['train']} train / {payload['splits']['test']} test; "
        f"sha256:{digest})",
        file=sys.stderr,
    )
    for task in tasks[:3]:
        print(f"  {task['id']} [{task['split']}] {task['prompt'][:70]!r} -> {task['checker']['type']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
