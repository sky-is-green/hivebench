"""Token-growth analysis: is the per-turn context actually flat?

Replays one or more run bundles (``run_report.json`` from
``experiments.generate_data``) and computes, for every user turn, three
context sizes:

  - ``strata`` — what the strata actually delivered to the model (the turn's
    assembled ``token_count``)
  - ``raw``  — the unbounded history size (every prior exchange plus the
    current query), which grows linearly with session length
  - ``fifo`` — the naive FIFO window over that same history (default 4k cap)

Prints bucketed medians across all conversations, optionally writes JSON and
a dependency-free SVG chart.

Usage::

    python -m experiments.token_growth runs/<ts> [--fifo-budget 4000] [--json out.json] [--plot figures/token_growth.svg]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from cortex.baselines.metrics import estimate_tokens
from experiments.paired_ab import _fifo_context

BUCKET_EDGES = [1, 5, 10, 20, 40, 10**9]  # inclusive upper bounds per bucket


def _bucket_label(turn: int) -> str:
    for i, edge in enumerate(BUCKET_EDGES):
        if turn <= edge:
            if i == 0:
                return "1"
            prev = BUCKET_EDGES[i - 1]
            return f"{prev + 1}-{edge}" if edge != BUCKET_EDGES[-1] else f"{prev + 1}+"
    return "41+"


def _bucket_order(label: str) -> int:
    if label == "1":
        return 0
    if label.endswith("+"):
        return 10**9
    return int(label.split("-")[-1])


def conversation_series(conv_turns: list[dict], fifo_budget: int) -> list[dict]:
    """Per-turn {turn, strata, raw, fifo} for one conversation record."""
    history: list[dict] = []
    points = []
    for t in conv_turns:
        q = t.get("query", "")
        hist_text = "\n\n".join([m["content"] for m in history] + [q])
        raw = estimate_tokens(hist_text)
        fifo_ctx = _fifo_context(history + [{"role": "user", "content": q}],
                                 fifo_budget)
        fifo = estimate_tokens(fifo_ctx)
        points.append({
            "turn": t["turn"],
            "strata": t.get("token_count", 0),
            "raw": raw,
            "fifo": fifo,
        })
        history.append({"role": "user", "content": q})
        reply = t.get("reply", "")
        if reply:
            history.append({"role": "assistant", "content": reply})
    return points


def analyze(run_dirs: list[str | Path], fifo_budget: int) -> dict:
    runs = []
    buckets: dict[str, dict[str, list[int]]] = {}
    turns_seen = 0
    for rd in run_dirs:
        path = Path(rd) / "run_report.json"
        doc = json.loads(path.read_text(encoding="utf-8"))
        series = []
        for conv in doc.get("conversations", []):
            pts = conversation_series(conv.get("turns", []), fifo_budget)
            series.extend(pts)
            for p in pts:
                label = _bucket_label(p["turn"])
                slot = buckets.setdefault(label, {"raw": [], "fifo": [], "strata": []})
                for k in ("raw", "fifo", "strata"):
                    slot[k].append(p[k])
                turns_seen += 1
        runs.append({
            "run_dir": str(rd),
            "conversations": len(doc.get("conversations", [])),
            "series": sorted(series, key=lambda p: (p["turn"],)),
        })

    table = []
    for label in sorted(buckets, key=_bucket_order):
        slot = buckets[label]
        table.append({
            "turns": label,
            "n": len(slot["strata"]),
            "hive_median": round(statistics.median(slot["strata"]), 1),
            "fifo_median": round(statistics.median(slot["fifo"]), 1),
            "raw_median": round(statistics.median(slot["raw"]), 1),
        })
    return {
        "fifo_budget_tokens": fifo_budget,
        "turns_analyzed": turns_seen,
        "buckets": table,
        "runs": runs,
    }


def _svg(chart_path: Path, analysis: dict) -> None:
    """Dependency-free line chart: median tokens vs session length."""
    W, H = 920, 480
    ML, MB, MT, MR = 70, 46, 30, 16
    pw, ph = W - ML - MR, H - MB - MT

    def med(key):
        return [b[f"{key}_median"] for b in analysis["buckets"]]

    labels = [b["turns"] for b in analysis["buckets"]]
    n = len(labels)
    xs = [ML + (i * pw / max(n - 1, 1)) for i in range(n)]

    def y(v, vmax):
        return MT + ph - (v / vmax) * ph

    vmax = max(max(med("raw")), max(med("fifo")), max(med("strata")), 1) * 1.08

    def polyline(key, color):
        pts = " ".join(f"{x:.1f},{y(v, vmax):.1f}"
                       for x, v in zip(xs, med(key)))
        return (f'<polyline fill="none" stroke="{color}" stroke-width="2.5" '
                f'points="{pts}"/>')

    grid, yticks = [], []
    for i in range(5):
        v = vmax * i / 4
        yy = y(v, vmax)
        grid.append(f'<line x1="{ML}" y1="{yy:.1f}" x2="{W-MR}" y2="{yy:.1f}" '
                    f'stroke="#ddd" stroke-width="1"/>')
        yticks.append(f'<text x="{ML-8}" y="{yy+4:.1f}" text-anchor="end" '
                      f'font-size="12" fill="#555">{int(v)}</text>')
    xticks = [f'<text x="{x:.1f}" y="{H-MB+18}" text-anchor="middle" '
              f'font-size="12" fill="#555">{lbl}</text>'
              for x, lbl in zip(xs, labels)]

    legend = [
        ("#b22222", "unbounded history (grows linearly)"),
        ("#e28c1e", "FIFO window"),
        ("#2e7d32", "strata curated context"),
    ]
    litems = "".join(
        f'<rect x="{W-MR-268}" y="{MT+8+i*20}" width="14" height="4" fill="{c}"/>'
        f'<text x="{W-MR-248}" y="{MT+13+i*20}" font-size="12" fill="#333">{t}</text>'
        for i, (c, t) in enumerate(legend))

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Helvetica,Arial,sans-serif">
<rect width="{W}" height="{H}" fill="white"/>
<text x="{W/2}" y="20" text-anchor="middle" font-size="15" fill="#222">Context tokens delivered per user turn (medians across {analysis['runs'][0]['conversations']}+ conversations)</text>
{''.join(grid)}{''.join(yticks)}
<line x1="{ML}" y1="{MT+ph}" x2="{W-MR}" y2="{MT+ph}" stroke="#999"/>
<line x1="{ML}" y1="{MT}" x2="{ML}" y2="{MT+ph}" stroke="#999"/>
{polyline("raw", "#b22222")}
{polyline("fifo", "#e28c1e")}
{polyline("strata", "#2e7d32")}
{''.join(xticks)}
<text x="{(ML+W-MR)/2:.0f}" y="{H-8}" text-anchor="middle" font-size="12" fill="#555">session length (user turns)</text>
{litems}
</svg>'''
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    chart_path.write_text(svg, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dirs", nargs="+", help="run directories to analyze")
    parser.add_argument("--fifo-budget", type=int, default=4000)
    parser.add_argument("--json", default="", help="write full per-turn series JSON here")
    parser.add_argument("--plot", default="", help="write an SVG chart here")
    args = parser.parse_args(argv)

    analysis = analyze(args.run_dirs, args.fifo_budget)

    print(f"Turns analyzed: {analysis['turns_analyzed']} "
          f"(FIFO window {args.fifo_budget} tokens)")
    print(f"{'turn bucket':<12}{'n':>6}{'strata':>10}{'fifo':>10}{'raw':>10}")
    print("-" * 48)
    for b in analysis["buckets"]:
        print(f"{b['turns']:<12}{b['n']:>6}{b['hive_median']:>10}"
              f"{b['fifo_median']:>10}{b['raw_median']:>10}")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
        print(f"wrote {out.resolve()}")
    if args.plot:
        _svg(Path(args.plot), analysis)
        print(f"wrote {Path(args.plot).resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())