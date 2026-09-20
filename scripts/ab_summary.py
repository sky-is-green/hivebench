#!/usr/bin/env python3
"""Summarise TBR head-to-head eval reports (PQ2_0 vs Q8_0 vs Q4_K_M).

Reads the ``ab-*.json`` reports written by ``scripts/run_ab.sh`` and prints a
comparison table (answer recall / fact hit / context fidelity, hive vs FIFO),
plus per-model provenance from the run manifest. Also emits a JSON roll-up.

Usage:
  python scripts/ab_summary.py                      # artifacts/ternary/eval/ab-*.json
  python scripts/ab_summary.py --glob 'artifacts/ternary/eval/ab-*.json'
  python scripts/ab_summary.py --json-out artifacts/ternary/eval/ab-summary.json
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
from pathlib import Path

HB = Path(__file__).resolve().parents[1]

METRICS = (
    ("turns_compared", "turns"),
    ("hive_answer_recall", "recall_hive"),
    ("fifo_answer_recall", "recall_fifo"),
    ("hive_avg_fact_hit_ratio", "fact_hive"),
    ("fifo_avg_fact_hit_ratio", "fact_fifo"),
    ("hive_avg_context_fidelity", "ctx_fid_hive"),
    ("fifo_avg_context_fidelity", "ctx_fid_fifo"),
    ("fidelity_hive_gt_fifo_ratio", "fid_hive>fifo%"),
    ("hive_ge_fifo_ratio", "hive>=fifo%"),
)


def load_report(path: Path) -> dict:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"label": path.stem, "error": f"unreadable: {exc}"}
    paired = report.get("paired_ab") or {}
    metrics = paired.get("metrics") or {}
    manifest = report.get("manifest") or {}
    return {
        "label": path.stem.replace("ab-", ""),
        "path": str(path),
        "ok": report.get("ok"),
        "seconds": report.get("seconds"),
        "smoke": (report.get("smoke") or {}).get("passed"),
        "smoke_prompts": (report.get("smoke") or {}).get("prompts"),
        "error": report.get("error") or (paired.get("error") if isinstance(paired, dict) else None),
        "metrics": metrics,
        "identity": manifest.get("identity") or {},
        "ngl": (report.get("instance") or {}).get("ngl"),
        "hardware": manifest.get("hardware") or {},
    }


def fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=str(HB / "artifacts/ternary/eval/ab-*.json"))
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    paths = [Path(p) for p in sorted(globmod.glob(args.glob))]
    rows = [load_report(p) for p in paths]
    if not rows:
        print(f"no reports matched {args.glob}")
        return 1

    header = ["model", "ok", "smoke", "ngl", "quant"] + [label for _, label in METRICS]
    print(" | ".join(header))
    print(" | ".join("---" for _ in header))
    for row in rows:
        if row.get("error"):
            print(f"{row['label']} | FAIL | {row['error']}")
            continue
        ident = row["identity"]
        cells = [
            row["label"], fmt(row["ok"]),
            f"{row.get('smoke')}/{row.get('smoke_prompts')}",
            fmt(row.get("ngl")), fmt(ident.get("quantization")),
        ]
        cells += [fmt(row["metrics"].get(key)) for key, _ in METRICS]
        print(" | ".join(cells))

    print("\nprovenance:")
    for row in rows:
        if row.get("error"):
            continue
        ident = row["identity"]
        hw = row.get("hardware") or {}
        print(f"  {row['label']}: name={ident.get('name')} arch={ident.get('architecture')} "
              f"size={ident.get('size_gb')}GB vram_free={hw.get('vram_free_gb')}GB "
              f"seconds={row.get('seconds')}")

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
