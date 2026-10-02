#!/usr/bin/env python
"""Offline router evaluation — score decision models against known outcomes.

The cascade's C1 router decides whether a task goes to the local generator
(cheap) or straight to the API (expensive).  Because a pilot run records the
local *and* API answer for every task, any router candidate can be scored
offline: no generation reruns, no API spend.

Candidates:

- the frontier API prompt (the routes recorded in the run's ``router_route``);
- **Tiny-Jev-1.7B** (``noul``: P(the local model can answer this correctly)),
  the brief's lite decision model.

Reported per threshold: routing accuracy (route local iff the local model was
actually right), policy quality (local answer when routed local, API answer
when routed API), API-route share, and the always-local / always-API / oracle
bounds.  Writes ``experiments/cascade/router-eval.json``.

Usage::

    .venv/bin/python experiments/cascade/router_eval.py \
        --report experiments/cascade/runs/20261001-221942/report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.cascade.checkers import check  # noqa: E402
from experiments.cascade.history import load_history, load_tasks  # noqa: E402

CASCADE = REPO / "experiments" / "cascade"
DEFAULT_MODEL = "lostargon/Tiny-Jev-1.7B"
QUESTION = "Which model should handle this task?"
CRITERIA = {
    "local": "the local model answers correctly",
    "api": "the API model is needed",
}


def _probability(result: Any) -> float:
    """Read a probability out of Tiny-Jev's return shape."""
    if isinstance(result, (int, float)):
        return float(result)
    if hasattr(result, "item"):
        return float(result.item())
    if isinstance(result, dict):
        for key in ("probability", "prob", "p", "confidence", "score"):
            value = result.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        for value in result.values():
            if isinstance(value, (int, float)):
                return float(value)
    raise ValueError(f"cannot read a probability from {result!r}")


class TinyJev:
    """The lite decision model: structured state + question -> probability."""

    def __init__(self, model_id: str = DEFAULT_MODEL, device: str = "auto") -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        self.model.eval()
        self.device = device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def p_local_correct(self, task: dict) -> tuple[float, float]:
        state = {"task": task["prompt"], "bucket": task["bucket"]}
        t0 = time.time()
        with self.torch.no_grad():
            result = self.model.choice(self.tok, state, QUESTION, CRITERIA)
        ms = (time.time() - t0) * 1000.0
        probabilities = result.get("probabilities") or {} if isinstance(result, dict) else {}
        p_local = float(probabilities.get("local", 0.0))
        return p_local, ms


def policy_quality(route_local: bool, scion_ok: bool, api_ok: bool) -> bool:
    return scion_ok if route_local else api_ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="", help="run report JSON (default: newest)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-model", action="store_true", help="baselines only")
    parser.add_argument("--out", default=str(CASCADE / "router-eval.json"))
    args = parser.parse_args()

    tasks = load_tasks()
    history = load_history(args.report or None)
    rows: list[dict[str, Any]] = []
    for tid, record in history.items():
        task = tasks.get(tid)
        if task is None or not record.get("scion_answer"):
            continue  # no measured local answer (e.g. a gated task): not calibration data
        rows.append(
            {
                "id": tid,
                "bucket": record.get("bucket"),
                "scion_ok": check(task, record["scion_answer"]),
                "api_ok": (check(task, record["api_answer"]) if record.get("api_answer") else None),
                "api_router": (record.get("router_route") or "").lower() or None,
                "source": record.get("_source"),
            }
        )
    n = len(rows)
    if not n:
        raise SystemExit("no scorable records in the run history")
    api_rows = [r for r in rows if r["api_ok"] is not None]

    summary: dict[str, Any] = {
        "tasks": n,
        "sources": sorted({r["source"] for r in rows}),
        "api_coverage": len(api_rows) / n,
    }
    summary["always_local"] = {
        "quality": sum(r["scion_ok"] for r in rows) / n,
        "api_share": 0.0,
    }
    summary["always_api"] = {
        "quality": (sum(r["api_ok"] for r in api_rows) / len(api_rows)) if api_rows else None,
        "api_share": 1.0,
    }
    summary["oracle"] = {
        "quality": 1.0,
        "api_share": sum(not r["scion_ok"] for r in rows) / n,
    }

    # the frontier API router, as recorded during the pilot (measure-mode runs)
    routed = [r for r in rows if r["api_router"] in ("local", "api")]
    if routed:
        with_api = [r for r in routed if r["api_ok"] is not None]
        summary["api_router"] = {
            "tasks": len(routed),
            "routing_accuracy": sum(
                (r["api_router"] == "local") == r["scion_ok"] for r in routed
            ) / len(routed),
            "quality": (sum(
                policy_quality(r["api_router"] == "local", r["scion_ok"], r["api_ok"])
                for r in with_api
            ) / len(with_api)) if with_api else None,
            "api_share": sum(r["api_router"] == "api" for r in routed) / len(routed),
        }

    if not args.no_model:
        print(f"loading {args.model} ...", file=sys.stderr)
        jev = TinyJev(args.model, args.device)
        for row in rows:
            task = tasks[row["id"]]
            p, ms = jev.p_local_correct(task)
            row["p_local_correct"] = round(p, 4)
            row["router_ms"] = round(ms, 1)
        sweep = []
        for step in range(5, 100, 5):
            threshold = step / 100.0
            routed_local = [r["p_local_correct"] >= threshold for r in rows]
            quality_rows = [(rl, r) for rl, r in zip(routed_local, rows) if r["api_ok"] is not None]
            sweep.append(
                {
                    "threshold": threshold,
                    "routing_accuracy": sum(
                        rl == r["scion_ok"] for rl, r in zip(routed_local, rows)
                    ) / n,
                    "quality": (sum(
                        policy_quality(rl, r["scion_ok"], r["api_ok"]) for rl, r in quality_rows
                    ) / len(quality_rows)) if quality_rows else None,
                    "api_share": sum(not rl for rl in routed_local) / n,
                }
            )
        summary["tiny_jev"] = {
            "model": args.model,
            "device": jev.device,
            "mean_ms": sum(r["router_ms"] for r in rows) / n,
            "sweep": sweep,
        }
        rows = rows  # keep per-task probabilities in the artifact

    out = Path(args.out)
    out.write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=1), encoding="utf-8"
    )

    print(f"\n=== router eval ({n} tasks, sources: {', '.join(summary['sources'])}) ===")
    print(f"{'router':22} {'quality':>8} {'api share':>10} {'accuracy':>9}")
    for name in ("always_local", "always_api", "oracle"):
        s = summary[name]
        quality = "—" if s["quality"] is None else f"{s['quality']:.3f}"
        acc = "—" if name == "oracle" else ""
        print(f"{name:22} {quality:>8} {s['api_share']:>10.3f} {acc:>9}")
    if "api_router" in summary:
        s = summary["api_router"]
        quality = "—" if s["quality"] is None else f"{s['quality']:.3f}"
        print(f"{'api router (pilot)':22} {quality:>8} {s['api_share']:>10.3f} {s['routing_accuracy']:>9.3f}")
    if "tiny_jev" in summary:
        s = summary["tiny_jev"]
        best = max(
            s["sweep"],
            key=lambda r: ((r["quality"] or 0.0), r["routing_accuracy"]),
        )
        print(f"{'tiny-jev (best thr)':22} {best['quality']:>8.3f} {best['api_share']:>10.3f} {best['routing_accuracy']:>9.3f}"
              f"   thr={best['threshold']:.2f}  {s['mean_ms']:.0f} ms/decision")
    print("\nthreshold sweep (tiny-jev):")
    for row in summary.get("tiny_jev", {}).get("sweep", []):
        quality = "  —  " if row["quality"] is None else f"{row['quality']:.3f}"
        print(f"  thr {row['threshold']:.2f}  quality {quality}  api {row['api_share']:.3f}  acc {row['routing_accuracy']:.3f}")
    print("artifact:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
