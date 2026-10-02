#!/usr/bin/env python
"""Prefix-judge probe — can a verdict on a partial generation steer it?

The async-steering hypothesis: a judgement consumed while the generator is
still running can cancel or redirect it (fire-and-consume + cancel-on-reject,
``harness/cascade/judgement.py``).  That only pays if the judge has signal on
a *prefix*, and if answers that end correct do not dip below the threshold on
the way — a false cancel costs a paid escalation and latency.

Two data sources:

- default: the recorded ``scion_answer`` from the run reports (final content
  only; most answers are short, so only a few tasks reach a prefix);
- ``--streams capture.json``: full captured streams (reasoning + answer, from
  ``stream_capture.py``) — the real object a mid-generation judge consumes.

Framings asked through JEV-9B's ``noul`` head (polarity +1 = reject on high p):

- ``complete`` — the trained D2 question ("correct and complete"): the baseline,
  ill-posed on a prefix (nothing is complete mid-stream);
- ``ontrack``  — "the partial answer is on track to be correct and complete";
- ``willbe``   — "the final answer will be correct and complete" (prediction);
- ``fatal``    — "the partial answer already shows a fatal error continuing
  will not fix" (the cancel question).

Reported per framing and prefix: separation (AUC of the reject score against
ground truth), confusion at the no-leak threshold, and a cancel simulation
(cancel at the first prefix that trips the threshold; tokens saved, false
cancels, leaks).  Writes ``experiments/cascade/prefix-eval.json``.

Run with the ROCm venv (JEV-9B bf16, both cards):

    HIP_VISIBLE_DEVICES=0,1 ~/Desktop/work/.venv-rocm/bin/python \
        experiments/cascade/prefix_eval.py --model autotrust/JEV-9B \
        --streams experiments/cascade/streams/<run>/capture.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.cascade.checkers import check  # noqa: E402
from experiments.cascade.history import load_tasks  # noqa: E402
from experiments.cascade.judge_eval import load_judgeable  # noqa: E402

CASCADE = REPO / "experiments" / "cascade"

#: Token cut points; the full text is always added per task.
PREFIXES = (16, 32, 64, 128, 256, 512)

#: Framings asked through the noul head.  ``polarity`` is the direction that
#: means "reject": -1 = reject when p is low, +1 = reject when p is high.
FRAMINGS: dict[str, dict[str, Any]] = {
    "complete": {
        "question": "The candidate answer is correct and complete for the task.",
        "polarity": -1,
    },
    "ontrack": {
        "question": (
            "The partial candidate answer is on track to be correct and "
            "complete for the task."
        ),
        "polarity": -1,
    },
    "willbe": {
        "question": (
            "The final answer will be correct and complete for the task."
        ),
        "polarity": -1,
    },
    "fatal": {
        "question": (
            "The partial candidate answer already shows a fatal error that "
            "continuing will not fix."
        ),
        "polarity": +1,
    },
}

#: Thresholds for the cancel simulation (reject score >= theta cancels).
SIM_THETAS = (0.50, 0.70, 0.85)


def auc(scores: list[float], labels: list[bool]) -> Optional[float]:
    """Rank AUC of ``scores`` against boolean ``labels`` (ties count 0.5)."""
    pos = [s for s, label in zip(scores, labels) if label]
    neg = [s for s, label in zip(scores, labels) if not label]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 4)


def reject_score(p: float, polarity: int) -> float:
    return p if polarity > 0 else 1.0 - p


def load_stream_records(path: Path) -> dict[str, dict]:
    """Captured streams -> {task id: stream text + final content + prompt}."""
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for tid, record in data.get("records", {}).items():
        reasoning = record.get("reasoning") or ""
        content = record.get("content") or ""
        stream = (reasoning + "\n" + content).strip() if reasoning else content
        out[tid] = {
            "prompt": record.get("prompt", ""),
            "stream": stream,
            "content": content,
            "chunks": len(record.get("chunks") or []),
        }
    return out


def cancel_sim(rows: list[dict], frame: str, theta: float) -> dict[str, Any]:
    """Cancel at the first prefix whose reject score trips ``theta``.

    A cancel that lands on a wrong answer is a *catch* (same correction as the
    full gate, fewer local tokens); on a correct answer it is a false cancel
    (unnecessary escalation).  Wrong answers that never trip are leaks.
    """
    caught = false_cancels = leaks = 0
    saved = 0
    for row in rows:
        cuts = sorted(int(t) for t in row["reject"][frame])
        hit = next((t for t in cuts if row["reject"][frame][str(t)] >= theta), None)
        if hit is None:
            if not row["scion_ok"]:
                leaks += 1
        elif row["scion_ok"]:
            false_cancels += 1
        else:
            caught += 1
            saved += max(0, row["n_tokens"] - hit)
    return {
        "theta": theta,
        "caught": caught,
        "false_cancels": false_cancels,
        "leaks": leaks,
        "tokens_saved": saved,
        "mean_saved_per_catch": round(saved / caught, 1) if caught else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="autotrust/JEV-9B")
    parser.add_argument("--family", choices=("auto", "jev", "intern"), default="auto")
    parser.add_argument(
        "--framings",
        default="",
        help="comma-separated subset of the framings (default: all)",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--streams",
        default="",
        help="capture.json from stream_capture.py (default: recorded answers)",
    )
    parser.add_argument(
        "--prefixes",
        default="",
        help="comma-separated token cut points (default: 16,32,64,128,256,512)",
    )
    parser.add_argument("--out", default=str(CASCADE / "prefix-eval.json"))
    args = parser.parse_args()

    prefixes = tuple(int(x) for x in args.prefixes.split(",")) if args.prefixes else PREFIXES
    frames = (
        {k: FRAMINGS[k] for k in args.framings.split(",")}
        if args.framings
        else dict(FRAMINGS)
    )
    tasks = load_tasks()

    if args.streams:
        streams = load_stream_records(Path(args.streams))
        records = [
            (tid, rec) for tid, rec in streams.items() if tasks.get(tid) and rec["stream"]
        ]
        mode = "stream"
    else:
        history = load_judgeable()
        records = [
            (tid, rec)
            for tid, rec in history.items()
            if tasks.get(tid) and rec.get("scion_answer")
        ]
        mode = "answer"
    if args.limit:
        records = records[: args.limit]
    if not records:
        raise SystemExit("no records to judge")

    from experiments.cascade import decision as decision_models

    family = args.family
    if family == "auto":
        family = "jev" if "jev" in args.model.lower() else "intern"
    print(f"loading {args.model} ({family}) ...", file=sys.stderr, flush=True)
    if family == "jev":
        judge = decision_models.Jev9BJudge(args.model, device=args.device)
    else:
        judge = decision_models.InternDecisionJudge(args.model, device=args.device)
    tokenizer = getattr(judge, "tokenizer", None) or judge.engine.tokenizer

    rows: list[dict[str, Any]] = []
    started = time.time()
    for index, (tid, record) in enumerate(records, 1):
        task = tasks[tid]
        if mode == "stream":
            text = record["stream"]
            ok = check(task, record["content"])
        else:
            text = record["scion_answer"]
            ok = check(task, text)
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        n_tokens = len(ids)
        cuts = sorted({t for t in prefixes if t < n_tokens} | {n_tokens})
        row: dict[str, Any] = {
            "id": tid,
            "bucket": record.get("bucket"),
            "scion_ok": ok,
            "n_tokens": n_tokens,
            "p": {},
            "reject": {},
        }
        for t in cuts:
            prefix = tokenizer.decode(ids[:t], skip_special_tokens=True)
            for frame, spec in frames.items():
                if family == "jev":
                    if mode == "stream":
                        state = f"Task: {task['prompt']}\nPartial response so far:\n{prefix}"
                    else:
                        state = f"Task: {task['prompt']}\nCandidate answer: {prefix}"
                    p, _ms = judge.noul(state, spec["question"])
                else:
                    p, _ms = judge.noul(
                        {"task": task["prompt"], "candidate_answer": prefix},
                        spec["question"],
                    )
                row["p"].setdefault(frame, {})[str(t)] = round(p, 4)
                row["reject"].setdefault(frame, {})[str(t)] = round(
                    reject_score(p, spec["polarity"]), 4
                )
        rows.append(row)
        print(
            f"[{index}/{len(records)}] {tid:26} {n_tokens:>4} tok  ok={ok}  "
            f"({time.time() - started:.0f}s)",
            file=sys.stderr,
            flush=True,
        )

    labels = [not row["scion_ok"] for row in rows]  # "wrong" is the positive class
    summary: dict[str, Any] = {
        "mode": mode,
        "family": family,
        "model": args.model,
        "tasks": len(rows),
        "prefixes": list(prefixes),
        "ok": sum(1 for r in rows if r["scion_ok"]),
    }
    for frame, spec in frames.items():
        per_prefix: dict[str, Any] = {}
        for t in prefixes:
            subset = [r for r in rows if str(t) in r["reject"][frame]]
            if not subset:
                continue
            scores = [r["reject"][frame][str(t)] for r in subset]
            subset_labels = [not r["scion_ok"] for r in subset]
            at85 = [s >= 0.85 for s in scores]
            per_prefix[str(t)] = {
                "n": len(subset),
                "auc": auc(scores, subset_labels),
                "at_085": {
                    "cancelled": sum(at85),
                    "false_cancels": sum(
                        1 for c, r in zip(at85, subset) if c and r["scion_ok"]
                    ),
                    "caught": sum(
                        1 for c, r in zip(at85, subset) if c and not r["scion_ok"]
                    ),
                    "leaks": sum(
                        1 for c, r in zip(at85, subset) if not c and not r["scion_ok"]
                    ),
                },
                "mean_p_correct": round(
                    sum(r["p"][frame][str(t)] for r in subset if r["scion_ok"])
                    / max(1, sum(1 for r in subset if r["scion_ok"])),
                    4,
                ),
                "mean_p_wrong": round(
                    sum(r["p"][frame][str(t)] for r in subset if not r["scion_ok"])
                    / max(1, sum(1 for r in subset if not r["scion_ok"])),
                    4,
                ),
            }
        full_scores = [r["reject"][frame][str(r["n_tokens"])] for r in rows]
        summary[frame] = {
            "question": spec["question"],
            "polarity": spec["polarity"],
            "auc_full": auc(full_scores, labels),
            "prefix": per_prefix,
            "cancel_sim": [cancel_sim(rows, frame, theta) for theta in SIM_THETAS],
        }

    out = Path(args.out)
    artifact = {"model": args.model, "summary": summary, "rows": rows}
    out.write_text(json.dumps(artifact, indent=1), encoding="utf-8")

    print(f"\n=== prefix judge probe ({len(rows)} tasks, mode={mode}) ===")
    print(
        f"{'framing':10} {'t':>5} {'n':>3} {'AUC':>6} "
        f"{'f_cxl':>6} {'catch':>6} {'leak':>5} {'p_ok':>6} {'p_bad':>6}"
    )
    for frame in frames:
        for t in prefixes:
            entry = summary[frame]["prefix"].get(str(t))
            if not entry:
                continue
            conf = entry["at_085"]
            print(
                f"{frame:10} {t:>5} {entry['n']:>3} "
                f"{entry['auc'] if entry['auc'] is not None else float('nan'):>6} "
                f"{conf['false_cancels']:>6} {conf['caught']:>6} {conf['leaks']:>5} "
                f"{entry['mean_p_correct']:>6} {entry['mean_p_wrong']:>6}"
            )
    print("\nfull-text gate (same framing):")
    for frame in frames:
        print(f"  {frame:10} AUC {summary[frame]['auc_full']}")
    print("\ncancel simulation (cancel at first prefix with reject >= theta):")
    print(f"{'framing':10} {'theta':>5} {'caught':>6} {'false_cxl':>9} {'leaks':>5} {'saved':>6}")
    for frame in frames:
        for sim in summary[frame]["cancel_sim"]:
            print(
                f"{frame:10} {sim['theta']:>5.2f} {sim['caught']:>6} "
                f"{sim['false_cancels']:>9} {sim['leaks']:>5} {sim['tokens_saved']:>6}"
            )
    print("\nartifact:", out)
    sys.stdout.flush()
    # ROCm teardown segfaults after the artifact is written; skip destructors.
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
