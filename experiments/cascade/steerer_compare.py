#!/usr/bin/env python
"""Unified steerer comparison — every sensor scored as P(final answer correct).

Collects the probes run against the same captured streams (decision heads,
ordinary LLMs, prompts) and recomputes the cancel trade with one polarity:
cancel when the sensor's P(correct) is below theta.  Outputs a matrix plus
``experiments/cascade/steerer-compare.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

CASCADE = Path(__file__).resolve().parents[2] / "experiments" / "cascade"
CUTS = (16, 32, 64, 128, 256)

#: label -> (artifact, kind, key).  kind "frame" reads rows[].p[frame][t];
#: kind "llm" reads rows[].p[t].
SENSORS = {
    "jev9b-complete": ("prefix-stream-eval.json", "frame", "complete"),
    "jev9b-ontrack": ("prefix-stream-eval.json", "frame", "ontrack"),
    "intern-0.8b": ("prefix-steerer-08b.json", "frame", "complete"),
    "intern-2b": ("prefix-steerer-2b.json", "frame", "complete"),
    "qwen3-1.7b-prob": ("steerer-qwen17.json", "llm", None),
    "qwen3-1.7b-binary": ("steerer-qwen17-binary.json", "llm", None),
    "qwen3-1.7b-thinking": ("steerer-qwen17-thinking.json", "llm", None),
    "olmoe-1b7b-base-binary": ("steerer-olmoe-binary.json", "llm", None),
    "qwen3.5-4b-binary": ("steerer-qwen35-4b-binary.json", "llm", None),
    "qwen3-1.7b-binlogit": ("steerer-qwen17-binlogit.json", "llm", None),
    "qwen3.5-4b-binlogit": ("steerer-qwen35-4b-binlogit.json", "llm", None),
    "qwen3.5-4b-prob": ("steerer-qwen35-4b-prob.json", "llm", None),
}


def rows_of(path: Path, kind: str, key: str | None) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for row in data.get("rows", []):
        if kind == "frame":
            p = (row.get("p") or {}).get(key) or {}
        else:
            p = row.get("p") or {}
        out.append({"id": row["id"], "ok": bool(row["scion_ok"]), "p": p})
    return out


def auc(scores: list[float], labels: list[bool]) -> float | None:
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 4)


def main() -> None:
    comparison: dict = {"cuts": CUTS, "sensors": {}}
    header = f"{'sensor':26}" + "".join(f"{('t=' + str(t)):>18}" for t in CUTS)
    print(header)
    print(f"{'':26}" + "".join(f"{'AUC  c/F':>18}" for _ in CUTS))
    for name, (file, kind, key) in SENSORS.items():
        path = CASCADE / file
        if not path.is_file():
            print(f"{name:26}  (missing {file})")
            continue
        rows = rows_of(path, kind, key)
        entry: dict = {}
        line = f"{name:26}"
        for t in CUTS:
            sub = [r for r in rows if str(t) in r["p"] and r["p"][str(t)] is not None]
            if len(sub) < 4:
                line += f"{'-':>18}"
                continue
            scores = [r["p"][str(t)] for r in sub]
            labels = [r["ok"] for r in sub]
            caught = sum(1 for s, l in zip(scores, labels) if not l and s < 0.5)
            false_cxl = sum(1 for s, l in zip(scores, labels) if l and s < 0.5)
            leaks = sum(1 for s, l in zip(scores, labels) if not l and s >= 0.5)
            entry[str(t)] = {
                "n": len(sub),
                "auc": auc(scores, labels),
                "caught_at_0.5": caught,
                "false_cancels_at_0.5": false_cxl,
                "leaks_at_0.5": leaks,
                "wrong_scores": {
                    r["id"]: round(r["p"][str(t)], 3) for r in sub if not r["ok"]
                },
            }
            a = entry[str(t)]["auc"]
            line += f"{f'{a:.2f}  {caught}/{false_cxl}':>18}"
        comparison["sensors"][name] = entry
        print(line)
    out = CASCADE / "steerer-compare.json"
    out.write_text(json.dumps(comparison, indent=1), encoding="utf-8")
    print("\ncancel = P(correct) < 0.5; task counts: wrong=4, correct=23")
    print("artifact:", out)


if __name__ == "__main__":
    main()
