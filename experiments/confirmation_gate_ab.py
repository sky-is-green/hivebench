"""P12 — Confirmation-Gate Hypothesis: deterministic A/B replay.

Grades every recorded (query, reply) exchange of a run against the fixture
imprint and compares the two ingestion policies on the SAME conversations:

- **Config A** — the rule-based hedge filter (``cortex.hedges.is_hedge_reply``,
  the current production behavior): reply stored unless it is a hedge.
- **Config B** — the S6 Confirmation Gate (fixture imprint): reply stored only
  when the gate decides accept/flag; hedges, non-copies, and thin replies are
  rejected and never enter memory.

No LLM calls: the run's recorded replies are replayed deterministically (the
same fact math the deterministic P2 diagnostic uses).

Hypothesis (HIVE-HANDOFF.md §4.8): confirming generations against the imprint
before storage raises ingestion of genuine facts and suppresses hedge/refusal
pollution — (a) ``ingestion_rate`` (share of imprint facts present in *stored*
replies) and (b) honest retrieval recall improve relative to the rule alone,
with (c) fewer stored refusals, at equal run cost.

Usage::

    python -m experiments.confirmation_gate_ab runs/<ts>

Reads ``runs/<ts>/run_report.json`` (``conversations`` records with per-turn
``query``/``reply`` + ``conversation_id``) and the fixture conversations
(default ``hivebench/tests/fixtures/generated``).

Exit code: 0 = hypothesis supported (B ingestion >= A AND B stored-refusals
<= A), 1 = not supported, 2 = nothing to grade (no replies / no imprint).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from cortex.confirmation_gate import ConfirmationGate, FixtureImprint
from cortex.hedges import is_hedge_reply
from experiments.retrieval_diagnostic import _fixture_answer_map

_WORD_RE = re.compile(r"[a-z0-9_+#./-]+")


def _load_records(run_dir: Path) -> list[dict]:
    report = json.loads((run_dir / "run_report.json").read_text(encoding="utf-8"))
    return report.get("conversations", [])


def _exchange_map(records: list[dict]) -> dict[str, list[tuple[str, str]]]:
    """conversation_id -> [(query, reply), ...] in turn order."""
    out: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for conv in records:
        cid = conv.get("conversation_id", "unknown")
        for turn in conv.get("turns", []):
            q = turn.get("query", "")
            r = turn.get("reply", "")
            if q:
                out[cid].append((q, r))
    return out


def _facts_expected(imprint, conv_id: str, queries: list[str]) -> set[str]:
    facts: set[str] = set()
    for q in queries:
        facts |= imprint.facts_for(conv_id, q)
    return facts


def _facts_in(replies: list[str], expected: set[str]) -> set[str]:
    found: set[str] = set()
    for r in replies:
        words = set(_WORD_RE.findall((r or "").lower()))
        found |= words & expected
    return found


def _ingestion_rate(replies: list[str], expected: set[str]) -> float:
    return len(_facts_in(replies, expected)) / len(expected) if expected else 0.0


def _stored_refusals(replies: list[str]) -> int:
    return sum(1 for r in replies if is_hedge_reply(r))


def run_ab(run_dir: Path, conversations_path: str) -> dict:
    from cortex.baselines.runner import load_conversations

    conversations = load_conversations(conversations_path)
    records = _load_records(run_dir)
    exchanges = _exchange_map(records)
    imprint = FixtureImprint(_fixture_answer_map(conversations))

    gate = ConfirmationGate()
    a_store: dict[str, list[str]] = defaultdict(list)
    b_store: dict[str, list[str]] = defaultdict(list)
    b_decisions = defaultdict(lambda: defaultdict(int))
    graded = 0

    for conv_id, pairs in exchanges.items():
        for query, reply in pairs:
            if not reply:
                continue
            graded += 1
            # Config A: rule-based hedge filter (current behavior).
            if not is_hedge_reply(reply):
                a_store[conv_id].append(reply)
            # Config B: confirmation gate.
            d = gate.decide(conv_id, query, reply, imprint)
            b_decisions[conv_id][d.decision] += 1
            if d.decision in ("accept", "flag"):
                b_store[conv_id].append(reply)

    per_conv = {}
    total_expected: set[str] = set()
    a_all: list[str] = []
    b_all: list[str] = []
    for conv_id, pairs in exchanges.items():
        queries = [q for q, _ in pairs]
        expected = _facts_expected(imprint, conv_id, queries)
        if not expected:
            continue
        total_expected |= expected
        a_all.extend(a_store[conv_id])
        b_all.extend(b_store[conv_id])
        per_conv[conv_id] = {
            "expected_facts": len(expected),
            "stored_a": len(a_store[conv_id]),
            "stored_b": len(b_store[conv_id]),
            "ingestion_a": round(_ingestion_rate(a_store[conv_id], expected), 3),
            "ingestion_b": round(_ingestion_rate(b_store[conv_id], expected), 3),
            "refusals_a": _stored_refusals(a_store[conv_id]),
            "refusals_b": _stored_refusals(b_store[conv_id]),
            "decisions_b": dict(b_decisions[conv_id]),
        }

    ingestion_a = _ingestion_rate(a_all, total_expected)
    ingestion_b = _ingestion_rate(b_all, total_expected)
    refusals_a = _stored_refusals(a_all)
    refusals_b = _stored_refusals(b_all)
    supported = (
        len(total_expected) > 0
        and ingestion_b >= ingestion_a
        and refusals_b <= refusals_a
    )

    return {
        "run_dir": str(run_dir),
        "exchanges_graded": graded,
        "conversations": len(exchanges),
        "imprint_facts_total": len(total_expected),
        "gate": gate.summary(),
        "a_rule": {
            "stored_replies": len(a_all),
            "ingestion_rate": round(ingestion_a, 3),
            "stored_refusals": refusals_a,
        },
        "b_gate": {
            "stored_replies": len(b_all),
            "ingestion_rate": round(ingestion_b, 3),
            "stored_refusals": refusals_b,
        },
        "hypothesis_supported": supported,
        "per_conversation": per_conv,
    }
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P12 confirmation-gate A/B replay")
    parser.add_argument("run_dir", help="a run directory containing run_report.json")
    parser.add_argument("--conversations", default="hivebench/tests/fixtures/generated")
    parser.add_argument("--json", default="", help="write the report to this path")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    if not (run_dir / "run_report.json").is_file():
        print(f"error: no run_report.json in {run_dir}")
        return 2
    report = run_ab(run_dir, args.conversations)
    if report["exchanges_graded"] == 0 or report["imprint_facts_total"] == 0:
        print(
            "SKIP: no graded exchanges or no imprint facts "
            f"(graded={report['exchanges_graded']}, "
            f"imprint_facts={report['imprint_facts_total']})"
        )
        return 2

    print(f"P12 confirmation-gate A/B replay — {run_dir}")
    print(f"  exchanges graded : {report['exchanges_graded']}  "
          f"conversations: {report['conversations']}  "
          f"imprint facts: {report['imprint_facts_total']}")
    a, b = report["a_rule"], report["b_gate"]
    print(f"  A rule  : stored {a['stored_replies']:>4}  "
          f"ingestion {a['ingestion_rate']:.3f}  refusals {a['stored_refusals']}")
    print(f"  B gate  : stored {b['stored_replies']:>4}  "
          f"ingestion {b['ingestion_rate']:.3f}  refusals {b['stored_refusals']}")
    print(f"  gate    : {report['gate']}")
    print(f"  verdict : {'SUPPORTED' if report['hypothesis_supported'] else 'NOT SUPPORTED'}")

    if args.json:
        out = Path(args.json)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"  wrote   : {out.resolve()}")
    return 0 if report["hypothesis_supported"] else 1


if __name__ == "__main__":
    sys.exit(main())