"""Compare two (or more) run bundles side by side.

Reads ``runs/<ts>/run_report.json`` bundles and prints a compact comparison:
PES + components, deterministic retrieval diagnostic (recall / ingestion /
ceiling / precision), per-turn stats, protocol verdicts, and a regression
flag when any headline metric moved against the paper's targets.

Usage::

    python -m experiments.run_compare runs/20260822_211131 runs/20260823_014521 [more...]

Exit code: 0 = no regressions vs the first run, 1 = a regression was flagged
(any headline metric moved >1pt against its target direction), 2 = nothing
comparable found.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FIELD_BY_PATH = {
    "post_run_pes.pes": "pes",
    "retrieval_diagnostic.retrieval_recall": "rd_recall",
    "retrieval_diagnostic.ingestion_rate": "rd_ingestion",
    "retrieval_diagnostic.perfect_hive_ceiling": "rd_ceiling",
    "retrieval_diagnostic.retrieval_precision": "rd_precision",
}

HEADLINES = (
    ("post_run_pes.pes", "PES (post-run)", "up"),
    ("retrieval_diagnostic.retrieval_recall", "P2 recall (honest %)", "up"),
    ("retrieval_diagnostic.ingestion_rate", "ingestion_rate %", "up"),
    ("retrieval_diagnostic.perfect_hive_ceiling", "perfect-strata ceiling %", "up"),
    ("retrieval_diagnostic.retrieval_precision", "precision (sentence proxy %)", "up"),
)


def _get(obj: dict, path: str):
    cur = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def load_run(run_dir: Path) -> dict:
    report = json.loads((run_dir / "run_report.json").read_text(encoding="utf-8"))
    verdicts = {}
    for p in report.get("protocol", []) or []:
        verdicts[p.get("id", "?")] = p.get("status", "?")
    return {
        "run_dir": str(run_dir),
        "name": run_dir.name,
        "mode": report.get("mode", "?"),
        "turns": (report.get("aggregate", {}) or {}).get("user_turns"),
        "conversations": (report.get("aggregate", {}) or {}).get("conversations"),
        "pes": _get(report, "post_run_pes.pes"),
        "pes_band": _get(report, "post_run_pes.band"),
        "pes_components": _get(report, "post_run_pes.components"),
        "rd_recall": _get(report, "retrieval_diagnostic.retrieval_recall"),
        "rd_ingestion": _get(report, "retrieval_diagnostic.ingestion_rate"),
        "rd_ceiling": _get(report, "retrieval_diagnostic.perfect_hive_ceiling"),
        "rd_precision": _get(report, "retrieval_diagnostic.retrieval_precision"),
        "comb": _get(report, "comb.total"),
        "verdicts": verdicts,
    }


def _fmt(v, nd: int = 1) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def compare(runs: list[Path]) -> int:
    loaded = []
    for r in runs:
        try:
            loaded.append(load_run(r))
        except (OSError, ValueError):
            print(f"error: unreadable run bundle at {r}")
            return 2
    if not loaded:
        print("error: no comparable run bundles")
        return 2

    # Headline table (direction-aware; later runs are deltas vs the first).
    print(f"{'metric':<32}" + "".join(f"{r['name'][:16]:>18}" for r in loaded))
    rows = []
    for key, label, direction in HEADLINES:
        values = [r[FIELD_BY_PATH[key]] for r in loaded]
        rows.append((key, label, direction, values))
        print(f"{label:<32}" + "".join(f"{_fmt(v):>18}" for v in values))
    if all(r["pes_band"] for r in loaded):
        print(f"{'PES band':<32}" + "".join(f"{r['pes_band']:>18}" for r in loaded))
    print(f"{'turns / convs':<32}" + "".join(
        f"{f'{r['turns'] or 0}/{r['conversations'] or 0}':>18}" for r in loaded
    ))
    print()

    # Protocol verdicts.
    ids = sorted({pid for r in loaded for pid in r["verdicts"]})
    if ids:
        print(f"{'prediction':<12}" + "".join(f"{r['name'][:16]:>18}" for r in loaded))
        for pid in ids:
            print(f"{pid:<12}" + "".join(f"{r['verdicts'].get(pid, '-'):>18}" for r in loaded))
        print()

    # Comb totals.
    if any(r["comb"] for r in loaded):
        print(f"{'comb archived':<32}" + "".join(
            f"{_fmt(r['comb'].get('archived')):>18}" for r in loaded
        ))
        print(f"{'comb resurrected':<32}" + "".join(
            f"{_fmt(r['comb'].get('resurrected')):>18}" for r in loaded
        ))
        print()

    # Regression scan: later runs vs the first, direction-aware (1pt margin).
    regressions = []
    for key, label, direction, values in rows:
        base = values[0]
        if base is None:
            continue
        for r in loaded[1:]:
            v = r[FIELD_BY_PATH[key]]
            if v is None:
                continue
            delta = v - base
            if direction == "up" and delta < -1.0:
                regressions.append(f"{r['name']}: {label} {base} -> {v} ({delta:+.1f})")
            elif direction == "down" and delta > 1.0:
                regressions.append(f"{r['name']}: {label} {base} -> {v} ({delta:+.1f})")

    if regressions:
        print("REGRESSIONS (vs the first run):")
        for line in regressions:
            print(f"  - {line}")
        return 1
    print("No regressions vs the first run.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare HiveBench run bundles")
    parser.add_argument("run_dirs", nargs="+", help="run directories (run_report.json inside)")
    args = parser.parse_args(argv)
    runs = []
    for raw in args.run_dirs:
        path = Path(raw)
        if not (path / "run_report.json").is_file():
            print(f"error: no run_report.json in {path}")
            return 2
        runs.append(path)
    return compare(runs)


if __name__ == "__main__":
    sys.exit(main())