#!/usr/bin/env python
"""Offline judge evaluation — score the D3 verifier against known outcomes.

Every pilot run records the local answer and whether it was correct, so a
judge candidate (the frontier API prompt, Tiny-Jev, Intern-Decision-4B) can be
scored offline on the same labelled set: does it accept correct answers and
reject wrong ones?

Metrics per threshold: true accepts, false accepts (accepted a wrong answer —
the quality leak), false rejects (rejected a correct one — wasted escalation),
and the judge's own accept rate.  Writes ``experiments/cascade/judge-eval.json``.

Run with the ROCm venv for the local judge (4B on GPU):

    ~/Desktop/work/.venv-rocm/bin/python experiments/cascade/judge_eval.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.cascade.checkers import check  # noqa: E402
from experiments.cascade.history import RUNS, load_tasks  # noqa: E402

CASCADE = REPO / "experiments" / "cascade"


def load_judgeable() -> dict[str, dict]:
    """Latest record per task that has both a candidate answer and a verdict.

    ``load_history`` keeps the newest record per task, and gate-mode records
    have no candidate — this keeps the records a judge can actually be scored
    on (i.e. the measure-mode runs), newest wins.
    """
    merged: dict[str, dict] = {}
    for report_path in sorted(RUNS.glob("*/report.json"), key=lambda p: p.stat().st_mtime):
        data = json.loads(report_path.read_text(encoding="utf-8"))
        for record in data.get("records", []):
            if record.get("scion_answer") and record.get("verdict") in ("accept", "reject"):
                merged[record["id"]] = {**record, "_source": report_path.parent.name}
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="internlm/Intern-Decision-4B")
    parser.add_argument("--family", choices=("auto", "intern", "jev"), default="auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-model", action="store_true", help="recorded API verdicts only")
    parser.add_argument("--out", default=str(CASCADE / "judge-eval.json"))
    args = parser.parse_args()

    tasks = load_tasks()
    history = load_judgeable()
    rows: list[dict[str, Any]] = []
    for tid, record in history.items():
        task = tasks.get(tid)
        if task is None or not record.get("scion_answer"):
            continue
        rows.append(
            {
                "id": tid,
                "bucket": record.get("bucket"),
                "candidate": record["scion_answer"],
                "scion_ok": check(task, record["scion_answer"]),
                # the recorded frontier verdict, when the run measured one
                "api_verdict": record.get("verdict"),
                "api_confidence": record.get("verdict_confidence"),
            }
        )
    n = len(rows)
    if not n:
        raise SystemExit("no measured records in the history")

    summary: dict[str, Any] = {"tasks": n, "sources": sorted({r.get('bucket') for r in rows})}
    api_rows = [r for r in rows if r["api_verdict"] in ("accept", "reject")]
    if api_rows:
        accepted = [r for r in api_rows if r["api_verdict"] == "accept"]
        summary["api_judge"] = {
            "tasks": len(api_rows),
            "accept_rate": len(accepted) / len(api_rows),
            "false_accepts": sum(1 for r in accepted if not r["scion_ok"]),
            "false_rejects": sum(
                1 for r in api_rows if r["api_verdict"] == "reject" and r["scion_ok"]
            ),
            "correct_verdicts": sum(
                1
                for r in api_rows
                if (r["api_verdict"] == "accept") == r["scion_ok"]
            ),
        }

    out = Path(args.out)
    if not args.no_model:
        from experiments.cascade.decision import InternDecisionJudge, Jev9BJudge

        family = args.family
        if family == "auto":
            family = "jev" if "jev" in args.model.lower() else "intern"
        print(f"loading {args.model} ({family}) ...", file=sys.stderr)
        judge = (Jev9BJudge if family == "jev" else InternDecisionJudge)(
            args.model, device=args.device
        )
        engine = getattr(judge, "engine", None)
        for row in rows:
            task = tasks[row["id"]]
            p, ms = judge.verdict(task, row["candidate"])
            row["p_correct"] = round(p, 4)
            row["judge_ms"] = round(ms, 1)
        sweep = []
        for step in range(5, 100, 5):
            threshold = step / 100.0
            accepted = [r["p_correct"] >= threshold for r in rows]
            sweep.append(
                {
                    "threshold": threshold,
                    "accept_rate": sum(accepted) / n,
                    "false_accepts": sum(
                        1 for a, r in zip(accepted, rows) if a and not r["scion_ok"]
                    ),
                    "false_rejects": sum(
                        1 for a, r in zip(accepted, rows) if not a and r["scion_ok"]
                    ),
                    "correct_verdicts": sum(
                        1 for a, r in zip(accepted, rows) if a == r["scion_ok"]
                    ),
                }
            )
        no_leak = next((s for s in sweep if s["false_accepts"] == 0), None)
        entry = {
            "model": args.model,
            "family": family,
            "device": str(engine.backend.device.type) if engine is not None else judge.device,
            "temperature": engine.temperature if engine is not None else judge.temperatures,
            "mean_ms": round(sum(r["judge_ms"] for r in rows) / n, 1),
            "no_leak": no_leak,
            "sweep": sweep,
            "p_correct": {r["id"]: r["p_correct"] for r in rows},
        }
        # Merge into one artifact so the scaling curve accumulates across runs.
        data = {"labels": {r["id"]: {"bucket": r["bucket"], "scion_ok": r["scion_ok"]} for r in rows}, "models": {}}
        if out.is_file():
            try:
                data = json.loads(out.read_text(encoding="utf-8"))
            except ValueError:
                pass
        data.setdefault("models", {})[args.model.split("/")[-1]] = entry
        out.write_text(json.dumps(data, indent=1), encoding="utf-8")
        summary["intern_decision"] = entry

    if args.no_model:
        out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1), encoding="utf-8")

    print(f"\n=== judge eval ({n} tasks) ===")
    print(f"{'judge':22} {'accept rate':>11} {'false acc':>9} {'false rej':>9} {'correct':>8}")
    if "api_judge" in summary:
        s = summary["api_judge"]
        print(f"{'api judge (recorded)':22} {s['accept_rate']:>11.3f} {s['false_accepts']:>9} {s['false_rejects']:>9} {s['correct_verdicts']}/{s['tasks']}")
    if "intern_decision" in summary:
        s = summary["intern_decision"]
        label = (s.get("family", "intern") + " " + s["model"].split("/")[-1])[:21]
        best = max(s["sweep"], key=lambda r: (r["correct_verdicts"], -r["false_accepts"]))
        print(f"{label:22} {best['accept_rate']:>11.3f} {best['false_accepts']:>9} {best['false_rejects']:>9} {best['correct_verdicts']}/{n}   thr={best['threshold']:.2f} {s['mean_ms']:.0f} ms")
        nl = s.get("no_leak")
        if nl:
            print(f"{'  no-leak point':22} {nl['accept_rate']:>11.3f} {nl['false_accepts']:>9} {nl['false_rejects']:>9} {nl['correct_verdicts']}/{n}   thr={nl['threshold']:.2f}")
    print("\nthreshold sweep (intern-decision):")
    for row in summary.get("intern_decision", {}).get("sweep", []):
        print(f"  thr {row['threshold']:.2f}  accept {row['accept_rate']:.3f}  false_acc {row['false_accepts']}  false_rej {row['false_rejects']}  correct {row['correct_verdicts']}/{n}")
    print("artifact:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
