"""Bench regression gate (X9): run the protocol, compare PES against a
committed baseline, exit nonzero on regression.

Usage:
    python -m harness.bench_gate --baseline <baseline.json> [--threshold 5.0]
    python -m harness.bench_gate --update <baseline.json>  # save new baseline

The baseline file stores the post-run PES from a known-good protocol run.
CI: run the protocol, compare, fail the pipeline if PES drops by more than
the threshold (default 5.0 points — noise tolerance for a non-deterministic
LLM backend).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def run_protocol(output_dir: Path, max_convs: int = 2) -> dict:
    """Run the mock protocol and return the report dict."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "experiments.generate_data",
        "--mock", "--protocol", "--max-convs", str(max_convs),
        "--output", str(output_dir),
    ]
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    report_path = output_dir / "run_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"protocol run produced no report: {report_path}")
    return json.loads(report_path.read_text(encoding="utf-8"))


def extract_pes(report: dict) -> dict:
    """Extract the PES metrics we gate on."""
    post = report.get("post_run_pes") or {}
    return {
        "pes": post.get("composite") or post.get("pes") or 0,
        "band": post.get("band", ""),
        "retrieval_recall": (report.get("retrieval_diagnostic") or {})
        .get("retrieval_recall_retrievable"),
        "protocol_passes": sum(
            1 for p in (report.get("protocol") or [])
            if p.get("status") == "PASS"),
        "protocol_total": len(report.get("protocol") or []),
    }


def extract_tok_per_sec(report: dict) -> float:
    """Extract tok/s from a report, accepting several legacy keys."""
    perf = report.get("performance") or report.get("metrics") or {}
    raw = perf.get("tokPerSec") or perf.get("tok_per_sec") or perf.get("throughput")
    if isinstance(raw, (int, float)):
        return float(raw)
    # Fallback: derive from tok/s embedded in post_run_pes detail if present.
    return 0.0


def append_history(history_path: Path, report: dict, run_name: str) -> dict:
    """Append one protocol run to the bench history file (S18).

    The history is a JSON array of {pes, tokPerSec, runName, timestamp}.
    Oldest entries are dropped when the file grows past 30 points.
    """
    import time

    entry = {
        "pes": extract_pes(report)["pes"],
        "tokPerSec": extract_tok_per_sec(report),
        "runName": run_name,
        "timestamp": int(time.time() * 1000),
    }
    history: list[dict] = []
    if history_path.is_file():
        try:
            loaded = json.loads(history_path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                history = loaded
        except Exception:
            history = []
    history.append(entry)
    # Cap at 30 points, oldest dropped.
    if len(history) > 30:
        history = history[-30:]
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return entry


def load_history(history_path: Path) -> list[dict]:
    """Load the bench history array, returning [] when missing or malformed."""
    if not history_path.is_file():
        return []
    try:
        data = json.loads(history_path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def main(argv=None):
    parser = argparse.ArgumentParser(description="Bench regression gate (X9)")
    parser.add_argument("--baseline", required=True,
                        help="path to the baseline JSON file")
    parser.add_argument("--threshold", type=float, default=5.0,
                        help="max allowed PES drop (default 5.0)")
    parser.add_argument("--update", action="store_true",
                        help="run protocol and save as new baseline")
    parser.add_argument("--max-convs", type=int, default=2)
    parser.add_argument("--output", default="",
                        help="run output dir (default: auto)")
    args = parser.parse_args(argv)

    baseline_path = Path(args.baseline)
    output_dir = Path(args.output) if args.output else (
        REPO_ROOT / "runs_mock" / f"bench_gate_{baseline_path.stem}"
    )

    print("bench-gate: running protocol (mock)...")
    report = run_protocol(output_dir, max_convs=args.max_convs)
    current = extract_pes(report)
    print(f"bench-gate: current PES {current['pes']} ({current['band']})")

    if args.update:
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(current, indent=2), encoding="utf-8")
        print(f"bench-gate: baseline saved to {baseline_path}")
        return 0

    if not baseline_path.is_file():
        print(f"bench-gate: no baseline at {baseline_path}; "
              "run with --update to create one")
        return 1

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    print(f"bench-gate: baseline PES {baseline['pes']} ({baseline['band']})")

    delta = current["pes"] - baseline["pes"]
    if delta < -args.threshold:
        print(f"bench-gate: REGRESSION — PES dropped {abs(delta):.1f} points "
              f"(threshold {args.threshold})")
        return 1
    print(f"bench-gate: PASS — delta {delta:+.1f} points")
    return 0


if __name__ == "__main__":
    sys.exit(main())
