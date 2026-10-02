#!/usr/bin/env python
"""Build per-task token windows of the captured streams for test-mtp-probe.

Tokenization is the server's own (authoritative for the GGUF); the capture
texts become one window per task, padded to the longest, plus a meta file with
lengths and checker labels.  Run with the Scion stack applied, then:

    test-mtp-probe <model.gguf> <out>/tokens.bin <n> <seq> <out>/dump 999
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import requests

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.cascade.checkers import check  # noqa: E402
from experiments.cascade.history import load_tasks  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:1234")
    parser.add_argument(
        "--capture",
        default=str(REPO / "experiments/cascade/streams/scion-v2-streams/capture.json"),
    )
    parser.add_argument(
        "--out",
        default=str(REPO / "experiments/cascade/streams/scion-v2-streams/hidden"),
    )
    args = parser.parse_args()

    data = json.loads(Path(args.capture).read_text(encoding="utf-8"))
    tasks = load_tasks()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    order: list[str] = []
    lengths: list[int] = []
    labels: list[bool] = []
    ids_by_task: list[list[int]] = []
    for tid, record in data["records"].items():
        text = (
            (record.get("reasoning") or "") + "\n" + (record.get("content") or "")
        ).strip()
        resp = requests.post(
            f"{args.base}/tokenize",
            json={"content": text, "add_special": True},
            timeout=180,
        )
        resp.raise_for_status()
        ids = [int(t) for t in resp.json()["tokens"]]
        order.append(tid)
        lengths.append(len(ids))
        labels.append(bool(check(tasks[tid], record["content"])))
        ids_by_task.append(ids)
        print(f"{tid:26} {len(ids):>5} tok  ok={labels[-1]}", file=sys.stderr, flush=True)

    seq = max(lengths)
    arr = np.zeros((len(order), seq), dtype=np.int32)
    for row, ids in enumerate(ids_by_task):
        arr[row, : len(ids)] = ids
    arr.tofile(out / "tokens.bin")
    (out / "meta.json").write_text(
        json.dumps(
            {
                "capture": args.capture,
                "order": order,
                "lengths": lengths,
                "labels": labels,
                "seq": seq,
                "n_embd": None,  # filled from the probe dump
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(
        f"wrote {out/'tokens.bin'} ({arr.shape[0]} x {seq}) and meta.json",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
