"""MCP-path recall + latency battery (S3).

Makes the sidecar's MCP tools a measured feature: it drives
``strata_remember`` + ``strata_search`` over live JSON-RPC (``POST /v1/mcp``)
and measures recall and latency on the same long-horizon conversations the
raw REST path is measured on, then folds the per-turn numbers through the
existing :class:`~testing.ab_test.ABTestRunner` so an MCP run yields the same
comparative shape as every other A/B in hivebench.

Method (deterministic, no LLM required):

1. From each fixture conversation, parse the ``Key decision: <facet> = <value>``
   facts the assistant stated and the later user turns that ask for them again
   (``Remind me ... throttling``). Fall back to recent-memory probes when a
   conversation carries no decision markers.
2. Ingest every earlier fact through the path under test (MCP ``strata_remember``
   vs raw ``/v1/strata/observe``), then query it (MCP ``strata_search`` vs raw
   ``/v1/strata/curate``), so both paths see byte-identical memory and only the
   transport differs.
3. Recall for a query is the token overlap between the assembled context and the
   fact it should surface; latency is the query round trip. ``--raw-mode turn``
   swaps the raw baseline to the full generating ``/v1/strata/turn``.

Usage::

    venv/bin/python -m testing.mcp_battery --base-url http://127.0.0.1:8790
    venv/bin/python -m testing.mcp_battery --raw-mode turn --max-probes 10 --json out.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from collections import deque
from typing import Callable, Optional

from cortex.baselines.runner import load_conversations
from testing.ab_test import ABTestRunner
from testing.mcp_client import (
    DEFAULT_BASE_URL,
    DEFAULT_CONVERSATION_ID,
    DEFAULT_TIMEOUT,
    McpClient,
    RawTurnClient,
)

DEFAULT_CONVERSATIONS = "tests/fixtures/generated_horizon"
DEFAULT_TOP_K = 5
DEFAULT_MAX_PROBES = 20

_KEY_DECISION = re.compile(r"key decision:\s*(.+?)\s*=\s*(.+)", re.IGNORECASE)
_STOPWORDS = frozenset(
    "the a an and or of to in for is are was were be been with on at by this "
    "that it its as from key decision should use what did we agree final "
    "remind me again restate need can you".split()
)


# ---------------------------------------------------------------------------
# Deterministic recall scoring
# ---------------------------------------------------------------------------
def content_terms(text: str) -> set[str]:
    """Lowercased content tokens (length > 2, stopwords removed)."""
    return {
        word
        for word in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(word) > 2 and word not in _STOPWORDS
    }


def recall_score(expected: str, assembled: str) -> Optional[float]:
    """Fraction of expected-fact tokens that appear in the assembled context."""
    exp = content_terms(expected)
    if not exp:
        return None
    got = content_terms(assembled)
    return round(len(exp & got) / len(exp), 4)


def _utilization(payload: dict) -> Optional[float]:
    budget = payload.get("budget") or 0
    if not budget:
        return None
    return round((payload.get("token_count") or 0) / budget, 4)


def _metrics(recall: Optional[float], latency_ms: float, util: Optional[float]) -> dict:
    pes = round(recall * 100.0, 3) if recall is not None else None
    return {
        "pes": pes,
        "retrieval_precision": pes,
        "latency_ms": round(latency_ms, 3),
        "context_utilization": util,
    }


# ---------------------------------------------------------------------------
# Fixture -> probe plan
# ---------------------------------------------------------------------------
def build_probes(conversation: dict) -> tuple[list[dict], list[dict]]:
    """Return ``(memory, probes)`` for one conversation fixture.

    ``memory`` is the ordered list of assistant facts to ingest;
    ``probes`` are the user turns that should recall an *earlier* fact, each
    carrying the expected-fact text. Falls back to recent-memory probes.
    """
    turns = conversation.get("turns", [])
    memory: list[dict] = []
    decisions: list[dict] = []
    for i, turn in enumerate(turns):
        if turn.get("role") != "assistant":
            continue
        content = turn.get("content") or ""
        memory.append({"turn": i, "content": content})
        match = _KEY_DECISION.search(content)
        if match:
            decisions.append(
                {
                    "facet": match.group(1).strip().lower(),
                    "value": match.group(2).strip(),
                    "turn": i,
                    "expected": content,
                }
            )

    probes: list[dict] = []
    for i, turn in enumerate(turns):
        if turn.get("role") != "user":
            continue
        query = (turn.get("content") or "").strip()
        if not query:
            continue
        ql = query.lower()
        matches = [
            d for d in decisions if d["turn"] < i and d["facet"] and d["facet"] in ql
        ]
        if matches:
            best = max(matches, key=lambda d: len(d["facet"]))
            probes.append(
                {"turn": i, "query": query, "facet": best["facet"],
                 "expected": best["expected"]}
            )

    if not probes:  # no decision markers: probe the immediately-preceding fact
        for i, turn in enumerate(turns):
            if turn.get("role") == "user" and i > 0 \
                    and turns[i - 1].get("role") == "assistant":
                probes.append(
                    {"turn": i, "query": (turn.get("content") or "").strip(),
                     "facet": "", "expected": turns[i - 1].get("content", "")}
                )
    return memory, probes


# ---------------------------------------------------------------------------
# One path over one conversation
# ---------------------------------------------------------------------------
@dataclass
class PathRun:
    metrics: list[dict] = field(default_factory=list)
    query_latencies: list[float] = field(default_factory=list)
    ingest_latencies: list[float] = field(default_factory=list)
    recalls: list[float] = field(default_factory=list)
    errors: int = 0
    ingests: int = 0

    def extend(self, other: "PathRun") -> None:
        self.metrics.extend(other.metrics)
        self.query_latencies.extend(other.query_latencies)
        self.ingest_latencies.extend(other.ingest_latencies)
        self.recalls.extend(other.recalls)
        self.errors += other.errors
        self.ingests += other.ingests

    def summary(self, label: str) -> dict:
        return {
            "label": label,
            "probes": len(self.metrics),
            "ingests": self.ingests,
            "errors": self.errors,
            "recall": _avg(self.recalls),
            "avg_query_latency_ms": _avg(self.query_latencies),
            "avg_ingest_latency_ms": _avg(self.ingest_latencies),
        }


def _avg(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def _run_path(
    memory: list[dict],
    probes: list[dict],
    ingest: Callable[[str], object],
    query: Callable[[dict], object],
) -> PathRun:
    """Ingest each earlier fact, then query it; collect recall + latencies."""
    run = PathRun()
    ingested = 0
    for probe in probes:
        while ingested < len(memory) and memory[ingested]["turn"] < probe["turn"]:
            run.ingests += 1
            res = ingest(memory[ingested]["content"])
            if getattr(res, "ok", False):
                run.ingest_latencies.append(getattr(res, "latency_ms", 0.0))
            else:
                run.errors += 1
            ingested += 1

        res = query(probe)
        if not getattr(res, "ok", False):
            run.errors += 1
            run.metrics.append(_metrics(None, 0.0, None))
            continue
        run.query_latencies.append(res.latency_ms)
        payload = getattr(res, "payload", {}) or {}
        recall = recall_score(probe["expected"], payload.get("assembled_content", ""))
        if recall is not None:
            run.recalls.append(recall)
        run.metrics.append(_metrics(recall, res.latency_ms, _utilization(payload)))

    while ingested < len(memory):
        run.ingests += 1
        res = ingest(memory[ingested]["content"])
        if getattr(res, "ok", False):
            run.ingest_latencies.append(getattr(res, "latency_ms", 0.0))
        else:
            run.errors += 1
        ingested += 1
    return run


def _ab_compare(raw_metrics: list[dict], mcp_metrics: list[dict]) -> dict:
    """Fold both paths' per-turn metrics through the existing ABTestRunner."""
    n = min(len(raw_metrics), len(mcp_metrics))
    if n == 0:
        return {}
    raw_q = deque(raw_metrics[:n])
    mcp_q = deque(mcp_metrics[:n])

    def process_raw(query: str, turn: int) -> dict:
        return raw_q.popleft() if raw_q else {}

    def process_mcp(query: str, turn: int) -> dict:
        return mcp_q.popleft() if mcp_q else {}

    synthetic = [{"turns": [{"content": str(i)} for i in range(n)]}]
    result = ABTestRunner(turns=n).run(
        synthetic, process_raw, process_mcp, key="pes"
    )
    return {
        "winner": result.winner,
        "key": result.detail.get("key"),
        "config_a": result.config_a_metrics,
        "config_b": result.config_b_metrics,
    }


# ---------------------------------------------------------------------------
# Battery
# ---------------------------------------------------------------------------
@dataclass
class BatteryReport:
    base_url: str
    conversation_id: str
    raw_mode: str
    handshake_ok: bool
    handshake_error: str
    probes: int
    ab: dict
    mcp: dict
    raw: dict

    def to_dict(self) -> dict:
        return asdict(self)


def run_mcp_battery(
    base_url: str = DEFAULT_BASE_URL,
    conversations: Optional[list] = None,
    conversations_dir: str = DEFAULT_CONVERSATIONS,
    max_probes: Optional[int] = DEFAULT_MAX_PROBES,
    top_k: int = DEFAULT_TOP_K,
    timeout: float = DEFAULT_TIMEOUT,
    http=None,
    token: str = "",
    conversation_id: str = "",
    raw_mode: str = "curate",
) -> BatteryReport:
    """Drive the MCP path and the raw path over *conversations*; return metrics.

    No host/port is hardcoded: ``base_url`` is required by the caller (CLI
    default aside). Each fixture conversation gets its own sidecar
    conversation_id per path so stores stay isolated.
    """
    if conversations is None:
        conversations = load_conversations(conversations_dir)
    run_id = conversation_id or f"{DEFAULT_CONVERSATION_ID}-{int(time.time())}"

    mcp = McpClient(base_url, run_id, timeout=timeout, http=http, token=token)
    raw = RawTurnClient(base_url, run_id, timeout=timeout, http=http, token=token)

    handshake = mcp.initialize()
    mcp_run = PathRun()
    raw_run = PathRun()
    probe_count = 0

    for conv in conversations:
        conv_slug = str(conv.get("conversation_id", "conv"))
        memory, probes = build_probes(conv)
        if max_probes:
            probes = probes[:max_probes]
        if not probes:
            continue
        probe_count += len(probes)
        mcp_cid = f"{run_id}-{conv_slug}-mcp"
        raw_cid = f"{run_id}-{conv_slug}-raw"
        raw.reset(raw_cid)

        mcp_run.extend(
            _run_path(
                memory,
                probes,
                ingest=lambda text, cid=mcp_cid: mcp.remember(text, cid),
                query=lambda probe, cid=mcp_cid: mcp.search(
                    probe["query"], top_k, cid
                ),
            )
        )

        def _raw_query(probe, cid=raw_cid):
            if raw_mode == "turn":
                return raw.turn(probe["query"], cid)
            return raw.curate(probe["query"], cid)

        raw_run.extend(
            _run_path(
                memory,
                probes,
                ingest=lambda text, cid=raw_cid: raw.observe(text, cid),
                query=_raw_query,
            )
        )

    return BatteryReport(
        base_url=base_url.rstrip("/"),
        conversation_id=run_id,
        raw_mode=raw_mode,
        handshake_ok=handshake.ok,
        handshake_error=handshake.error,
        probes=probe_count,
        ab=_ab_compare(raw_run.metrics, mcp_run.metrics),
        mcp=mcp_run.summary("mcp"),
        raw=raw_run.summary("raw"),
    )


# ---------------------------------------------------------------------------
# Reporting + CLI
# ---------------------------------------------------------------------------
def format_report(report: BatteryReport) -> str:
    lines = [
        f"MCP-path battery  [{'OK' if report.handshake_ok else 'HANDSHAKE FAILED'}]",
        f"  base_url        : {report.base_url}",
        f"  conversation_id : {report.conversation_id}",
        f"  raw baseline    : /v1/strata/{report.raw_mode}",
        f"  probes          : {report.probes}",
    ]
    if not report.handshake_ok:
        lines.append(f"  handshake error : {report.handshake_error}")
        return "\n".join(lines)

    def row(label: str, path: dict) -> str:
        recall = path["recall"]
        return (
            f"  {label:<13}: recall {recall if recall is not None else '-'}"
            f"  query {path['avg_query_latency_ms']}ms"
            f"  ingest {path['avg_ingest_latency_ms']}ms"
            f"  errors {path['errors']}"
        )

    lines.append(row("MCP path", report.mcp))
    lines.append(row("raw path", report.raw))
    if report.ab:
        lines.append(
            f"  A/B winner      : {report.ab['winner']} "
            f"(A=raw, B=mcp; key={report.ab['key']})"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Drive the sidecar MCP tools live and measure recall + "
                    "latency against the raw REST path (S3).",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                        help="sidecar base URL, e.g. http://127.0.0.1:8790")
    parser.add_argument("--conversations", default=DEFAULT_CONVERSATIONS,
                        help="directory of conversation JSON fixtures")
    parser.add_argument("--max-probes", type=int, default=DEFAULT_MAX_PROBES,
                        help="cap the recall probes per conversation (0 = all)")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help="strata_search top_k")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                        help="per-request timeout in seconds")
    parser.add_argument("--token", default="",
                        help="sidecar token (sent as x-strata-token)")
    parser.add_argument("--conversation-id", default="",
                        help="base conversation id (default: timestamped)")
    parser.add_argument("--raw-mode", choices=("curate", "turn"), default="curate",
                        help="raw baseline endpoint: curate (no generation) or "
                             "the full /v1/strata/turn")
    parser.add_argument("--json", default="",
                        help="write the full report as JSON to this path")
    args = parser.parse_args(argv)

    conversations = load_conversations(args.conversations)
    if not conversations:
        print(f"error: no conversations under {args.conversations}")
        return 2

    report = run_mcp_battery(
        base_url=args.base_url,
        conversations=conversations,
        max_probes=args.max_probes or None,
        top_k=args.top_k,
        timeout=args.timeout,
        token=args.token,
        conversation_id=args.conversation_id,
        raw_mode=args.raw_mode,
    )
    print(format_report(report))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report.to_dict(), fh, indent=2, default=str)
        print(f"wrote {args.json}")

    if not report.handshake_ok:
        return 2
    if report.mcp["errors"] and report.mcp["probes"] == 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
