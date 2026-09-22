"""Unified data-generation workflow.

Drives the full Splinter pipeline over a set of conversations and writes a
self-contained run directory with:

  - NDJSON event logs (correlation-tagged, redacted, rotated)
  - a ground-truth SQLite DB (routing decisions + auditor labels)
  - a per-conversation E2E report
  - optional P1-P10 protocol report (--protocol)
  - optional LM Studio / FIFO baselines (--baselines)

Run directory layout::

    runs/<timestamp>/
        run_report.json
        ground_truth.sqlite
        logs/events-*.ndjson
        labels/...

Usage::

    # Live against LM Studio (recommended for real data):
    python -m experiments.generate_data --live --max-convs 20

    # Offline to validate the workflow / generate synthetic data:
    python -m experiments.generate_data --mock --max-convs 5

    # Full: live + protocol + baselines:
    python -m experiments.generate_data --live --protocol --baselines
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import uuid
from collections import deque
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

from backend.cache_manager import KVCacheManager
from backend.lmstudio import LMStudioBackend
from backend.providers import (
    apply_provider_overrides,
    backend_kwargs,
    load_registry,
)
from cortex.baselines.runner import load_conversations
from cortex.config import SplinterConfig
from cortex.e2e import FakeUltraSmall, MockTransport
from cortex.splinter import Splinter
from cortex.routing import DroneRouter
from experiments.dashboard import KeepAwake, TermDashboard
from logs.event_logger import EventLogger
from auditor.auditor import Auditor, TurnRecord
from auditor.ground_truth import GroundTruthDB
from auditor.labeling import generate_all
from retention.store import ContextStore
from sieve.medium import MediumDrone
from sieve.ultra_small import UltraSmallDrone

DEFAULT_PINNED_PREFIX = (
    "You are an assistant operating in the Splinter Memory system. "
    "Answer using the provided context and conversation history whenever they "
    "contain the needed information. If the context is insufficient, you may "
    "draw on your general knowledge, but clearly mark any such part."
)


# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------
def _fmt_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _total_user_turns(conversations, max_turns) -> int:
    total = 0
    for conv in conversations:
        n = sum(1 for t in conv.get("turns", []) if t.get("role") == "user")
        total += min(n, max_turns) if max_turns else n
    return total


def _print_progress(done: int, total: int, start: float, **ctx) -> None:
    """Persistent, newline-terminated per-turn status line.

    Used when the terminal dashboard is off. A plain ``\\n`` line (not a
    carriage-return overwrite) so it scrolls as a readable log and is visible
    even when stdout is redirected/captured (launcher app, pipes, GUI) rather
    than a TTY — this keeps the console updating and errors greppable.
    """
    if total <= 0:
        return
    elapsed = time.time() - start
    pct = done / total * 100.0
    eta = ctx.get("eta") or ((elapsed / done) * (total - done) if done else 0.0)
    sys.stdout.write(
        f"  turn {ctx.get('turn', '?')}/{ctx.get('cap', '?')}  "
        f"conv {ctx.get('conv', '?')}/{ctx.get('conv_total', '?')}  "
        f"{done}/{total} ({pct:5.1f}%)  "
        f"elapsed {_fmt_seconds(elapsed)}  ETA {_fmt_seconds(eta)}\n"
    )
    sys.stdout.flush()


def _phase(msg: str) -> None:
    if _QUIET_STDOUT:
        return
    print(f"\n--- {msg} ---", flush=True)


def _ttft_probe_ms(backend, pinned_prefix: str, assembled_content: str,
                   query: str, timeout: int = 60) -> Optional[float]:
    """Stream one request with the exact leading system message + context and
    measure time-to-first-content-token — the prefix-cache-hit proxy.

    llama.cpp's automatic prefix caching reuses the byte-stable pinned prefix;
    a cache hit makes prefill (and thus TTFT) ~constant regardless of context
    size, a miss makes TTFT scale with prompt length. LM Studio hides
    ``prompt_eval_count``, so this is the measurable attribution signal
    (white paper Threat 7). Mirrors the backend's message construction
    (single leading system message).
    """
    base = getattr(backend, "base_url", "")
    model = getattr(backend, "model", "")
    if not base or not model:
        return None
    system = f"{pinned_prefix}\n\n{assembled_content}" if pinned_prefix \
        else assembled_content
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ],
        "max_tokens": 16,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    headers = dict(getattr(backend, "_headers", lambda: {})() or {})
    headers["Authorization"] = f"Bearer {getattr(backend, 'api_key', 'lm-studio')}"
    started = time.monotonic()
    try:
        with requests.post(
            f"{base}/v1/chat/completions", json=payload, headers=headers,
            stream=True, timeout=timeout,
        ) as resp:
            if resp.status_code != 200:
                return None
            for line in resp.iter_lines():
                if not line:
                    continue
                text = line.decode("utf-8", errors="ignore")
                if not text.startswith("data:"):
                    continue
                data = text[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except ValueError:
                    continue
                choices = chunk.get("choices") or []
                if choices and choices[0].get("delta", {}).get("content"):
                    return (time.monotonic() - started) * 1000.0
    except requests.RequestException:
        return None
    return None


# Module-level flag: when a terminal dashboard owns the screen, suppress the
# CLI banner/progress writes so the two never fight over stdout.
_QUIET_STDOUT = False



def _resolve_backend(args):
    """Return (backend, ultra, live) or raise a clear error."""
    prov = getattr(args, "provider_resolved", None)
    pkw = backend_kwargs(prov) if prov else {}
    if args.mock:
        return (
            LMStudioBackend(base_url=args.base_url, model=args.model,
                            api_key=pkw.get("api_key", "lm-studio"),
                            extra_headers=pkw.get("extra_headers"),
                            transport=MockTransport(),
                            disable_thinking=args.no_thinking),
            FakeUltraSmall(),
            False,
        )
    ultra = UltraSmallDrone(confidence_mode=args.confidence)
    ultra._ensure_loaded()
    backend = LMStudioBackend(base_url=args.base_url, model=args.model,
                              api_key=pkw.get("api_key", "lm-studio"),
                              extra_headers=pkw.get("extra_headers"),
                              disable_thinking=args.no_thinking)
    if not backend.health():
        raise RuntimeError(
            f"LM Studio not reachable at {args.base_url}. Start it with a model "
            "loaded, or pass --mock to run offline."
        )
    return backend, ultra, True


def _pid_alive(pid: int) -> bool:
    try:
        import psutil

        return psutil.pid_exists(pid)
    except Exception:  # pragma: no cover
        return True  # conservative: assume alive if we cannot tell


def _acquire_run_lock(run_dir) -> bool:
    """Refuse to start if another live process already owns this run dir.

    Self-healing: a lock left by a crashed/killed process (dead PID) is
    overwritten. A lock held by this same PID (resume in-process) is allowed.
    """
    lock_path = run_dir / "run.lock"
    if lock_path.exists():
        try:
            old = int(lock_path.read_text(encoding="utf-8").strip())
        except (ValueError, OSError):
            old = None
        if old is not None and old != os.getpid() and _pid_alive(old):
            return False
    lock_path.write_text(str(os.getpid()), encoding="utf-8")
    return True


def _run_conversations(splinter, conversations, max_turns, conversation_id=None,
                       show_progress=True, resume=None, checkpoint_path=None,
                       checkpoint_every=10, run_args=None, dashboards=None,
                       ttft_probe_every: int = 0):
    """Run conversations through the splinter, optionally resuming a prior run.

    ``resume`` (dict) carries ``conv_index`` (0-based index of the conversation
    being processed), ``turn_index`` (user turns already done inside it),
    ``records`` (completed conversation records), ``current_record`` (the partial
    record of the conversation in progress, if any), and ``done`` (total turns).
    A checkpoint is written every ``checkpoint_every`` turns and at each
    conversation boundary when ``checkpoint_path`` is set, so a crashed run can
    be restarted in place without losing the partial conversation.
    """
    start_conv = resume.get("conv_index", 0) if resume else 0
    start_turn = resume.get("turn_index", 0) if resume else 0
    records = list(resume.get("records", [])) if resume else []
    current_record = resume.get("current_record") if resume else None
    done = resume.get("done", 0) if resume else 0

    total = _total_user_turns(conversations, max_turns)
    start = time.time()
    conv_total = len(conversations)
    turn_times = deque(maxlen=20)  # rolling window of recent turn durations (ms)

    def save_checkpoint(conv_index, turn_index, rec):
        if checkpoint_path is None:
            return
        checkpoint_path.write_text(json.dumps({
            "version": 1,
            "run_id": splinter.run_id,
            "config": splinter.config.to_dict(),
            "store": splinter.store.to_dict(),
            "hive_turn": splinter.turn,
            "comb_stats_history": list(getattr(splinter, "comb_stats_history", []) or []),
            "comb_stats": dict(getattr(splinter, "comb_stats", {}) or {}),
            "progress": {
                "conv_index": conv_index,
                "turn_index": turn_index,
                "records": records,
                "current_record": rec,
                "done": done,
            },
            "run_args": run_args or {},
        }, indent=2, default=str), encoding="utf-8")

    for ci in range(start_conv, conv_total):
        conv = conversations[ci]
        # Per-conversation store isolation: reset before every conversation
        # EXCEPT a mid-conversation resume, whose store was restored from the
        # checkpoint and must keep the partial conversation's chunks. Without
        # this reset, one Splinter/store across all conversations lets chunks from
        # earlier conversations crowd out the current one's relevant context.
        if not (resume is not None and ci == start_conv and current_record is not None):
            splinter.reset_conversation()
        if ci == start_conv and current_record is not None:
            conv_record = current_record
        else:
            conv_record = {
                "conversation_id": conv.get("conversation_id", "unknown"),
                "profile": conv.get("profile", "unknown"),
                "turns": [],
            }
        n_user = sum(1 for t in conv.get("turns", []) if t.get("role") == "user")
        cap = min(n_user, max_turns) if max_turns else n_user
        turn_count = 0
        for td in conv.get("turns", []):
            if td.get("role") != "user":
                continue
            turn_count += 1
            if max_turns and turn_count > max_turns:
                break
            if ci == start_conv and turn_count <= start_turn:
                continue  # already processed before the checkpoint
            res = splinter.process_turn(td["content"], conversation_id=conversation_id)
            ttft_ms = None
            if ttft_probe_every and turn_count % ttft_probe_every == 0 \
                    and splinter.backend is not None and res.assembled:
                ttft_ms = _ttft_probe_ms(
                    splinter.backend, splinter.pinned_prefix, res.assembled.content,
                    td["content"],
                )
            conv_record["turns"].append({
                "turn": res.turn,
                "query": res.query,
                "reply": res.reply,
                "assembled_content": res.assembled.content if res.assembled else "",
                "mode": res.mode,
                "pes": res.pes,
                "degradation_level": res.degradation_level,
                "timings": res.timings,
                "token_count": res.assembled.token_count if res.assembled else 0,
                "budget": res.assembled.budget if res.assembled else 0,
                "completion_tokens": (
                    (getattr(splinter.backend, "last_usage", {}) or {}).get("completion_tokens")
                ),
                "prompt_tokens": (
                    (getattr(splinter.backend, "last_usage", {}) or {}).get("prompt_tokens")
                ),
                "ttft_probe_ms": ttft_ms,
            })
            done += 1
            # ETA from the median of recent turn durations (robust to outlier
            # slow/sleep-contaminated turns and responsive to current speed),
            # rather than the run-wide average which stays pinned high by the
            # early turns and barely moves.
            turn_times.append(res.timings.get("total_ms", 0.0))
            eta_s = statistics.median(turn_times) / 1000.0 * (total - done)
            for d in (dashboards or []):
                d.update_progress(
                    done, total, time.time() - start, eta_s,
                    ci + 1, conv_total, turn_count, cap,
                )
                d.add_turn(
                    td["content"], res.reply, res.pes,
                    res.timings.get("generation_ms", 0.0),
                    res.timings.get("total_ms", 0.0),
                )
            if checkpoint_path is not None and done % checkpoint_every == 0:
                save_checkpoint(ci, turn_count, conv_record)
            if show_progress:
                _print_progress(done, total, start, eta=eta_s,
                                conv=ci + 1, conv_total=conv_total, turn=turn_count, cap=cap)
                if res.error:
                    print(f"  ! turn {res.turn} ERROR: {res.error}", file=sys.stderr)
                elif not (res.reply or "").strip():
                    print(f"  ! turn {res.turn} WARNING: empty reply", file=sys.stderr)
                elif res.mode == "fifo_fallback":
                    print(
                        f"  ! turn {res.turn} WARNING: fifo fallback "
                        f"(degradation level {res.degradation_level})",
                        file=sys.stderr,
                    )
        records.append(conv_record)
        # boundary checkpoint: this conversation is now complete in the record
        save_checkpoint(ci + 1, 0, None)
        current_record = None
    if show_progress:
        print()
    return records


def _aggregate(records):
    turns = [t for c in records for t in c["turns"]]
    if not turns:
        return {}
    peps = [t["pes"] for t in turns]
    lat = [t["timings"].get("total_ms", 0.0) for t in turns]
    return {
        "user_turns": len(turns),
        "conversations": len(records),
        "avg_pes": round(sum(peps) / len(peps), 2),
        "min_pes": round(min(peps), 2),
        "avg_total_ms": round(sum(lat) / len(lat), 2),
        "fifo_fallbacks": sum(1 for t in turns if t["mode"] == "fifo_fallback"),
        "drift_events": sum(1 for t in turns if t["degradation_level"] >= 1),
    }


def _populate_ground_truth(db, records, auditor, logger=None, push=None, workers=1):
    """Record routing decisions and auditor labels from a run.

    The LLM-as-judge label calls are independent and dominate this phase's wall
    time, so they run concurrently on a thread pool (``workers`` > 1). Results
    are collected back on the calling thread and written to the DB there, since
    the SQLite connection is single-threaded. Concurrency only helps if the
    backend serves parallel slots (llama.cpp ``-np`` / LM Studio parallel
    requests); with a single slot the requests queue (harmless).
    """
    for conv in records:
        for t in conv["turns"]:
            query = t["query"]
            route = DroneRouter().route(query)
            db.record_routing_decision(t["turn"], _router_score(route), route.route_to)
    # auditor labels on a sample of assembled turns
    sampled = [t for c in records for t in c["turns"]][::10]
    if push is not None:
        push("set_phase", f"ground truth labeling ({len(sampled)} sampled)")
    if not sampled:
        return

    def eval_one(t):
        try:
            label = auditor.evaluate_turn(
                TurnRecord(
                    turn=t["turn"],
                    assembled_context=t.get("assembled_content") or "",
                    user_query=t["query"], llm_response=t["reply"], chunk_ids=[],
                )
            )
            return label, None
        except Exception as exc:  # noqa: BLE001 — one bad auditor call must not kill the run
            return None, str(exc)

    if workers > 1 and len(sampled) > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(eval_one, sampled))
    else:
        results = [eval_one(t) for t in sampled]

    for i, (t, (label, err)) in enumerate(zip(sampled, results), 1):
        if err is not None:
            if logger is not None:
                logger.log("auditor", "label_failed",
                           {"turn": t["turn"], "error": err[:200]})
            if push is not None:
                push("add_line", f"auditor {i}/{len(sampled)} turn {t['turn']}: FAILED ({err})")
            # Always surface on stderr so a label failure is visible even with no
            # terminal dashboard / captured stdout.
            print(f"  auditor {i}/{len(sampled)} turn {t['turn']}: label FAILED ({err})",
                  file=sys.stderr)
            continue
        # if no chunk ids, record one aggregate label per sampled turn
        db.record_auditor_label(t["turn"], "aggregate", True,
                               label.context_sufficient, label.sufficiency_score)
        if push is not None:
            push("add_line", f"auditor {i}/{len(sampled)} turn {t['turn']}: "
                             f"sufficient={label.context_sufficient} score={label.sufficiency_score}")
        print(f"  auditor {i}/{len(sampled)} turn {t['turn']}: "
              f"sufficient={label.context_sufficient} score={label.sufficiency_score}",
              file=sys.stderr)


def _router_score(decision) -> float:
    try:
        return len(decision.reason.split(",")) if decision.reason else 0
    except Exception:  # noqa: BLE001
        return 0


def _read_baseline_tps(run_dir: Path) -> float | None:
    """Measured LM-Studio rolling tps from a ``--baselines`` run, if present."""
    path = run_dir / "baseline_lm_studio.json"
    if not path.exists():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        tps = doc.get("aggregate", {}).get("avg_tokens_per_sec")
        return float(tps) if tps else None
    except (OSError, ValueError, TypeError):
        return None


def _compute_post_run_pes(records, db, baseline_tps: float | None = None):
    """Post-run Pipeline Efficiency Score from ground-truth + measured metrics.

    The per-turn in-process PES (``Splinter.process_turn``) only sees latency and
    context utilization, so in live runs it floors near zero (the paper's
    LatencyHealth is ms-calibrated and live generation is seconds). This computes
    the paper's real PES after the run, when auditor retrieval/routing metrics
    exist. Components that are genuinely unavailable are dropped (the scorer
    renormalizes). Returns a dict, or None with no turns.
    """
    from cortex.efficiency import (
        EfficiencyScorer,
        context_utilization,
        latency_health,
        throughput_health,
    )

    turns = [t for c in records for t in c["turns"]]
    if not turns:
        return None
    n = len(turns)
    avg_total_ms = sum(t["timings"].get("total_ms", 0.0) for t in turns) / n
    avg_util = sum(t["token_count"] / max(t["budget"], 1) for t in turns) / n
    completions = [t.get("completion_tokens") for t in turns if t.get("completion_tokens")]
    total_gen_s = sum(t["timings"].get("generation_ms", 0.0) for t in turns) / 1000.0
    actual_tps = sum(completions) / total_gen_s if completions and total_gen_s > 0 else None
    baseline_tps = baseline_tps or 30.0

    has_labels = db.label_count() > 0
    precision = db.retrieval_precision() if has_labels else None
    routing = db.routing_accuracy()

    result = EfficiencyScorer().compute(
        retrieval_precision=precision,
        routing_accuracy=routing,
        avg_latency_ms=avg_total_ms,
        actual_tps=actual_tps,
        baseline_tps=baseline_tps,
        budget_used=avg_util,
        budget_total=1.0,
    )
    notes = []
    if latency_health(avg_total_ms) == 0 and avg_total_ms > 200:
        notes.append(
            f"LatencyHealth floors at 0 because the paper's formula is ms-calibrated "
            f"(50ms=100, 200ms=0) and live turns average ~{avg_total_ms / 1000:.0f}s; "
            "the PES below is therefore driven by retrieval/routing/throughput/utilization."
        )
    return {
        "pes": result.composite,
        "band": result.band,
        "breakdown": result.breakdown,
        "active_components": result.active_components,
        "components": {
            "retrieval_precision": round(precision, 1) if precision is not None else None,
            "routing_accuracy": round(routing, 1),
            "latency_health": round(latency_health(avg_total_ms), 1),
            "throughput_health": round(throughput_health(actual_tps, baseline_tps), 1) if actual_tps else None,
            "context_utilization": round(context_utilization(avg_util, 1.0), 1),
        },
        "measurements": {
            "turns": n,
            "avg_total_ms": round(avg_total_ms, 1),
            "avg_context_utilization_pct": round(avg_util * 100.0, 1),
            "actual_tps": round(actual_tps, 1) if actual_tps else None,
            "baseline_tps": baseline_tps,
            "auditor_labels": db.label_count(),
        },
        "notes": notes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Splinter data-generation workflow")
    parser.add_argument("--live", action="store_true", help="use real LM Studio + all-MiniLM")
    parser.add_argument("--mock", action="store_true", help="offline (fake drone + mock backend)")
    parser.add_argument("--conversations", default="tests/fixtures/generated")
    parser.add_argument("--max-convs", type=int, default=50)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--base-url", default="http://localhost:1234")
    parser.add_argument("--model", default="")
    parser.add_argument(
        "--provider", default="",
        help="provider name from the providers config: fills base_url/api_key/"
             "model/headers (explicit --base-url/--model win)",
    )
    parser.add_argument(
        "--providers-file", default="",
        help="path to the providers JSON config (default providers.local.json)",
    )
    parser.add_argument("--pinned-prefix", default=DEFAULT_PINNED_PREFIX)
    parser.add_argument("--output", default="")
    parser.add_argument("--protocol", action="store_true", help="also run P1-P10")
    parser.add_argument("--baselines", action="store_true", help="also run baseline harnesses")
    parser.add_argument(
        "--max-tokens", type=int, default=None,
        help="cap E2E reply length (default: uncapped, 4096 ceiling). On reasoning "
             "models a small cap yields empty replies unless --no-thinking is used",
    )
    parser.add_argument(
        "--sampling", default="",
        help="sampling overrides as JSON, e.g. --sampling "
             "'{\"temperature\":0.7,\"top_p\":0.9,\"repeat_penalty\":1.1}'. "
             "Known fields: temperature, top_p, top_k, min_p, repeat_penalty, "
             "presence_penalty, frequency_penalty, stop, seed, mirostat, "
             "mirostat_tau, mirostat_eta. Recorded in the run report.",
    )
    parser.add_argument(
        "--baseline-max-tokens", type=int, default=None,
        help="cap baseline reply length (baseline tps is a decode measurement)",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=10,
        help="write a resume checkpoint every N turns",
    )
    parser.add_argument(
        "--resume", default="",
        help="resume a run directory (reads its checkpoint.json)",
    )
    parser.add_argument(
        "--term", action="store_true",
        help="render a live dashboard inside the terminal (ANSI, no deps)",
    )
    parser.add_argument(
        "--confidence", choices=("mcdropout", "single", "off"), default="mcdropout",
        help="drone confidence mode; 'off' skips the MC-dropout passes (the stock "
             "embedding model yields confidence ~1.0 regardless)",
    )
    parser.add_argument(
        "--auditor-workers", type=int, default=4,
        help="parallel workers for the ground-truth auditor-labeling phase. "
             "Only speeds up when the backend serves parallel slots (llama.cpp "
             "-np / LM Studio parallel requests); single-slot servers queue "
             "the requests (harmless, no speedup)",
    )
    parser.add_argument(
        "--no-thinking", action="store_true",
        help="send enable_thinking=false with every request, so reasoning models "
             "skip chain-of-thought (faster, and reply caps stop yielding empty "
             "output). Combine with LM Studio's 'thinking' toggle for models that "
             "only honor the GUI setting.",
    )
    parser.add_argument(
        "--ttft-probe-every", type=int, default=0,
        help="every N turns, stream one probe request with the exact assembled "
             "context and record time-to-first-token (the prefix-cache-hit "
             "proxy; 0 = off). Cost: one short request per probe",
    )
    parser.add_argument(
        "--tokenizer", default="",
        help="path to a HF tokenizer.json for exact token budgets (assembly, "
             "utilization). Default: heuristic ~4 chars/token",
    )
    parser.add_argument(
        "--baseline-tps", type=float, default=None,
        help="baseline tokens/sec used for the ThroughputHealth component of "
             "post_run_pes (paper weight 0.15). Default: if --baselines ran, the "
             "measured LM-Studio rolling tps from baseline_lm_studio.json; else "
             "30.0 (the paper's placeholder, known to be above real hardware "
             "~14-21 tps).",
    )
    parser.add_argument(
        "--comb-dir", default="",
        help="enable the comb (P11 surplus SSD tier) and write per-conversation "
             "archive files here, e.g. runs/<ts>/comb. Chunks the splinter once "
             "curated that leave the active store (LRU eviction or stale-out) "
             "are frozen to disk instead of dropped, and resurrected when a "
             "returned topic's query is weak in the store (comb_gate_threshold).",
    )
    parser.add_argument(
        "--comb-top-k", type=int, default=3,
        help="comb candidates competing for the token budget per gated turn",
    )
    parser.add_argument(
        "--comb-max-records", type=int, default=2000,
        help="comb records kept per conversation (least-referenced pruned)",
    )
    args = parser.parse_args(argv)

    # Which options did the user pass explicitly? Used by --resume so it only
    # restores checkpoint parameters the user did NOT override on the command
    # line (e.g. a "--resume X --max-convs 3" run must stay at 3, not silently
    # resume the original run's max_convs).
    _defaults = vars(parser.parse_args([]))
    explicit = {k for k, v in vars(args).items() if v != _defaults.get(k)}

    if not args.live and not args.mock:
        args.mock = True  # default to offline for safety
    if args.live and args.mock:
        print("pass either --live or --mock, not both")
        return 2
    if args.resume and args.output:
        print("--output is ignored when --resume is given")
        args.output = ""

    # --- resume: restore run parameters from the checkpoint ---
    resume_ckpt = None
    if args.resume:
        rdir = Path(args.resume)
        ckpt_path = rdir / "checkpoint.json"
        if not ckpt_path.exists():
            print(f"no checkpoint found at {ckpt_path}")
            return 3
        resume_ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
        ra = resume_ckpt.get("run_args", {})
        for key in ("conversations", "max_convs", "max_turns", "base_url",
                    "model", "pinned_prefix", "confidence", "no_thinking",
                    "comb_dir", "comb_top_k", "comb_max_records"):
            if ra.get(key) is not None and key not in explicit:
                setattr(args, key, ra[key])
        if "mode" in ra and "live" not in explicit and "mock" not in explicit:
            if ra["mode"] == "mock":
                args.mock, args.live = True, False
            elif ra["mode"] == "live":
                args.mock, args.live = False, True
        print(f"resuming run {resume_ckpt.get('run_id')} from {args.resume}")
        print(f"  conversation {resume_ckpt['progress'].get('conv_index', 0) + 1}, "
              f"turn {resume_ckpt['progress'].get('turn_index', 0)} "
              f"({resume_ckpt['progress'].get('done', 0)} turns done)")
        overridden = [k for k in ("conversations", "max_convs", "max_turns",
                                  "base_url", "model", "confidence", "no_thinking",
                                  "comb_dir", "comb_top_k", "comb_max_records")
                      if k not in explicit and ra.get(k) is not None]
        if overridden:
            print("  restored from checkpoint:", ", ".join(sorted(overridden)))

    # --- provider config: fill base_url/api_key/model/headers from a named
    # provider (explicit --base-url/--model flags win over the provider) ---
    if args.provider:
        try:
            prov = load_registry(args.providers_file or None).resolve(args.provider)
        except LookupError as exc:
            print(str(exc))
            return 2
        apply_provider_overrides(vars(parser.parse_args([])), args, prov)
        args.provider_resolved = prov  # consumed by _resolve_backend (api_key)
        print(f"provider: {prov.name} -> {args.base_url} (model: {args.model or 'server default'})")

    if args.protocol and args.max_tokens:
        print("note: --max-tokens caps only the E2E phase; P1-P10 generations are uncapped")

    # --- run directory ---
    run_dir = (Path(args.resume) if args.resume else
               (Path(args.output) if args.output else
                Path("runs") / datetime.now().strftime("%Y%m%d_%H%M%S")))
    log_dir = run_dir / "logs"
    (run_dir / "labels").mkdir(parents=True, exist_ok=True)
    logger = EventLogger(log_dir=log_dir)

    # --- backend + splinter ---
    try:
        backend, ultra, live = _resolve_backend(args)
    except RuntimeError as exc:
        logger.close()
        print(str(exc))
        return 3

    if not _acquire_run_lock(run_dir):
        logger.close()
        print(f"run directory {run_dir} is already in use by a live process (run.lock)")
        return 3

    config = (SplinterConfig.from_dict(resume_ckpt["config"]) if resume_ckpt
              else SplinterConfig(max_tokens=args.max_tokens))
    if args.sampling:
        from backend.sampling import parse_sampling

        config.sampling = parse_sampling(args.sampling)
    tokenizer_label = "heuristic"
    if args.tokenizer:
        from cortex.tokenizer import Tokenizer, set_active_tokenizer

        tok = Tokenizer(use_real=True, tokenizer_path=args.tokenizer)
        if tok.is_real:
            set_active_tokenizer(tok)
            tokenizer_label = tok.label
        else:
            print(f"warning: tokenizer '{args.tokenizer}' failed to load; "
                  "using the heuristic (~4 chars/token)")
    if args.comb_dir:
        config.comb_enabled = True
        config.comb_dir = args.comb_dir
        config.comb_top_k = args.comb_top_k
        config.comb_max_records = args.comb_max_records
        Path(config.comb_dir).mkdir(parents=True, exist_ok=True)
    splinter = Splinter(
        config=config,
        ultra=ultra,
        medium=MediumDrone(score_pair_fn=lambda q, c: 0.5),
        backend=backend,
        logger=logger,
        pinned_prefix=args.pinned_prefix,
    )
    if resume_ckpt:
        splinter.run_id = resume_ckpt["run_id"]
        splinter.store = ContextStore.from_dict(
            resume_ckpt["store"], embed_fn=splinter.ultra.embed
        )
        splinter.turn = resume_ckpt["hive_turn"]
        if resume_ckpt.get("comb_stats_history") is not None:
            splinter.comb_stats_history = list(resume_ckpt["comb_stats_history"])
        if resume_ckpt.get("comb_stats") is not None:
            splinter.comb_stats = dict(resume_ckpt["comb_stats"])

    # Pinned prefix must reach the backend as a byte-stable leading system
    # message for llama.cpp automatic prefix caching (see KVCacheManager).
    if args.pinned_prefix and getattr(backend, "pinned_prefix", None) is None:
        print("warning: backend does not expose pinned_prefix; prefix caching disabled")

    # --- live terminal dashboard + keep-awake (optional) ---
    global _QUIET_STDOUT
    term = TermDashboard(enabled=args.term)
    if term.enabled:
        print("dashboard: terminal dashboard enabled (ANSI)")
    _QUIET_STDOUT = term.enabled  # terminal dashboard owns stdout banners
    dashboards = [d for d in (term,) if d.enabled]

    def push(method: str, *a, **kw) -> None:
        for d in dashboards:
            getattr(d, method)(*a, **kw)

    keep_awake = KeepAwake() if live else None
    if keep_awake is not None and keep_awake.active:
        print("keep-awake: system sleep disabled for this run (ES_SYSTEM_REQUIRED)")

    # --- run ---
    # A checkpoint may carry a stale/relative conversations path (e.g. from
    # before a repo restructure); a silently-empty corpus would make the whole
    # run hollow (the FIFO baseline exit-2 bug found 2026-08-24). Fall back
    # loudly instead of running nothing.
    if not load_conversations(args.conversations):
        fallback = "tests/fixtures/generated"
        print(f"warning: conversations path '{args.conversations}' resolves to "
              f"no conversations; using '{fallback}'")
        args.conversations = fallback
    conversations = load_conversations(args.conversations)[: args.max_convs]
    run_args = {
        "mode": "live" if live else "mock",
        "conversations": args.conversations,
        "max_convs": args.max_convs,
        "max_turns": args.max_turns,
        "base_url": args.base_url,
        "model": args.model,
        "pinned_prefix": args.pinned_prefix,
        "comb_dir": args.comb_dir,
        "comb_top_k": args.comb_top_k,
        "comb_max_records": args.comb_max_records,
        "confidence": args.confidence,
        "no_thinking": args.no_thinking,
        "conversation_id": run_dir.name,
    }
    _phase(f"1/3 E2E conversations ({len(conversations)} convs, ~{_total_user_turns(conversations, args.max_turns)} turns)")
    push("set_phase", f"1/3 E2E ({len(conversations)} convs)")
    records = _run_conversations(
        splinter, conversations, args.max_turns, conversation_id=run_dir.name,
        resume=resume_ckpt.get("progress") if resume_ckpt else None,
        checkpoint_path=run_dir / "checkpoint.json",
        checkpoint_every=args.checkpoint_every,
        run_args=run_args,
        dashboards=dashboards,
        show_progress=not term.enabled,
        ttft_probe_every=args.ttft_probe_every,
    )

    # --- baselines (optional; run before PES so a measured tps can calibrate
    # the ThroughputHealth component) ---
    baseline_report = None
    if args.baselines:
        _phase("2/3 Baselines (LM Studio rolling + FIFO)")
        push("set_phase", "2/3 Baselines")
        baseline_report = _run_baselines(args, run_dir)
        for name, res in (baseline_report or {}).items():
            push("add_line", f"baseline {name}: exit={res.get('exit')} error={res.get('error', '')}")

    baseline_tps = args.baseline_tps
    if baseline_tps is None and args.baselines:
        baseline_tps = _read_baseline_tps(run_dir)
        if baseline_tps is not None:
            print(f"  baseline tps : {baseline_tps:.1f} (measured, LM Studio rolling)")

    # --- ground truth ---
    db_path = run_dir / "ground_truth.sqlite"
    db = GroundTruthDB(db_path)
    auditor = Auditor(generate_fn=_mock_auditor if args.mock else _live_auditor(backend))
    _populate_ground_truth(db, records, auditor, logger=logger, push=push,
                           workers=args.auditor_workers)
    post_run_pes = _compute_post_run_pes(records, db, baseline_tps=baseline_tps)
    has_labels = db.label_count() > 0
    ground_truth_metrics = {
        "auditor_labels": db.label_count(),
        "retrieval_precision": round(db.retrieval_precision(), 1) if has_labels else None,
        "retrieval_recall": round(db.retrieval_recall(), 1) if has_labels else None,
        "false_eviction_rate": round(db.false_eviction_rate(), 1) if has_labels else None,
        "routing_accuracy": round(db.routing_accuracy(), 1),
    }
    db.close()

    # Deterministic P2 (no auditor): measure retrieval against the fixture's own
    # ground-truth answers. The auditor-based precision above is really a per-turn
    # sufficiency rate (predicted_relevant is hardcoded True), so recall/eviction
    # are trivial there; this is the white paper's actual labeled-chunk metric.
    from experiments.retrieval_diagnostic import compute_retrieval_vs_fixture

    retrieval_diagnostic = compute_retrieval_vs_fixture(records, conversations)

    # --- protocol (optional) ---
    protocol_report = None
    if args.protocol:
        _phase("3/3 P1-P10 protocol")
        push("set_phase", "3/3 P1-P10 protocol")
        from experiments.run_p1_p10 import PredictionSuite, _load_labels

        labels = _load_labels(conversations)
        suite = PredictionSuite(backend, ultra, MediumDrone(score_pair_fn=lambda q, c: 0.5),
                                conversations, labels, auditor, live=live)
        protocol_report = [r.__dict__ for r in suite.run()]
        for r in protocol_report:
            push("add_line", f"P{r['id']} {r['status']}: {r['title']}")

    logger.flush()
    logger.close()
    for d in dashboards:
        d.close()
    if keep_awake is not None:
        keep_awake.close()

    # --- report ---
    ttft_samples = [
        t.get("ttft_probe_ms") for c in records
        for t in c["turns"] if t.get("ttft_probe_ms") is not None
    ]
    ttft_block = None
    if ttft_samples:
        ttft_block = {
            "enabled": bool(args.ttft_probe_every),
            "probes": len(ttft_samples),
            "median_ttft_ms": round(statistics.median(ttft_samples), 1),
            "min_ttft_ms": round(min(ttft_samples), 1),
            "max_ttft_ms": round(max(ttft_samples), 1),
            "note": "TTFT is the prefix-cache-hit proxy: a cache hit keeps "
                    "prefill (and TTFT) ~constant as context grows; a rising "
                    "TTFT trend means the pinned prefix stopped being reused",
        }
    comb_report = None
    if getattr(splinter, "comb_stats_history", None):
        comb_report = {
            "enabled": bool(args.comb_dir),
            "dir": args.comb_dir,
            "per_conversation": splinter.comb_stats_history,
            "total": {
                k: sum(c[k] for c in splinter.comb_stats_history) for k in ("archived", "resurrected", "comb_hits", "gate_fired")
            },
        }
    report = {
        "run_id": splinter.run_id,
        "mode": "live" if live else "mock",
        "backend": type(backend).__name__,
        "engine": {
            "backend": type(backend).__name__,
            "model": getattr(backend, "model", "") or args.model,
            "base_url": getattr(backend, "base_url", "") or args.base_url,
            "sampling": config.sampling or None,
            "max_tokens": config.max_tokens,
            "no_thinking": bool(args.no_thinking),
            "tokenizer": tokenizer_label,
            "drone": config.ultra_model,
            "confidence_mode": config.confidence_mode,
            "pinned_prefix": args.pinned_prefix,
        },
        "run_dir": str(run_dir.resolve()),
        "generated_at": datetime.now().astimezone().isoformat(),
        "ground_truth_db": str(db_path.resolve()),
        "event_logs": str(log_dir.resolve()),
        "aggregate": _aggregate(records),
        "ground_truth": ground_truth_metrics,
        "post_run_pes": post_run_pes,
        "retrieval_diagnostic": retrieval_diagnostic,
        "comb": comb_report,
        "ttft_probe": ttft_block,
        "conversations": records,
        "protocol": protocol_report,
        "baselines": baseline_report,
    }
    (run_dir / "run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # --- summary ---
    agg = report["aggregate"]
    print(f"Run {splinter.run_id} ({report['mode']}) -> {run_dir.resolve()}")
    print(f"  conversations : {agg.get('conversations', 0)}  turns: {agg.get('user_turns', 0)}")
    print(f"  PES min/avg   : {agg.get('min_pes')} / {agg.get('avg_pes')}")
    if post_run_pes:
        print(f"  post-run PES : {post_run_pes['pes']} ({post_run_pes['band']})")
    if retrieval_diagnostic.get("retrieval_recall") is not None:
        print(f"  P2 recall    : {retrieval_diagnostic['retrieval_recall']}% "
              f"(honest, stated-facts only; ingestion {retrieval_diagnostic.get('ingestion_rate')}%, "
              f"ceiling {retrieval_diagnostic.get('perfect_hive_ceiling')}%)")
    print(f"  avg total ms  : {agg.get('avg_total_ms')}")
    print(f"  fifo fallbacks: {agg.get('fifo_fallbacks')}")
    if protocol_report:
        print(f"  protocol      : {sum(1 for r in protocol_report if r['status'] == 'PASS')}/{len(protocol_report)} PASS")
    print(f"  report        : {(run_dir / 'run_report.json').resolve()}")
    (run_dir / "run.lock").unlink(missing_ok=True)
    return 0


def _mock_auditor(prompt: str) -> str:
    return json.dumps({"sufficient": True, "used_pieces": [], "missing": [], "score": 4})


def _live_auditor(backend):
    def fn(prompt: str) -> str:
        # The auditor is a JSON-evaluation task, not a context-answer task: drop
        # the E2E pinned prefix, frame JSON explicitly, and leave headroom for
        # reasoning tokens (reasoning models spend their budget on CoT first).
        backend.pinned_prefix = ""
        return backend.generate(
            "You are an evaluation engine. Respond with ONLY a valid JSON object.",
            prompt, {"temperature": 0.0, "max_tokens": 2048},
        )
    return fn


def _baseline_done(path: Path) -> bool:
    """True if a baseline output file exists and is a completed (parseable) report."""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return isinstance(doc, dict) and "aggregate" in doc
    except (OSError, ValueError):
        return False


def _run_baselines(args, run_dir):
    from cortex.baselines import lm_studio_baseline, fifo_baseline

    out = {}
    for module, name in ((lm_studio_baseline, "lm_studio"), (fifo_baseline, "fifo")):
        out_path = run_dir / f"baseline_{name}.json"
        if _baseline_done(out_path):
            out[name] = {"exit": 0, "output": str(out_path), "skipped": "existing output"}
            continue
        base = [
            "--conversations", args.conversations,
            "--base-url", args.base_url, "--model", args.model,
            "--output", str(out_path),
        ]
        if args.baseline_max_tokens:
            base += ["--max-tokens", str(args.baseline_max_tokens)]
        if args.mock:
            base += ["--mock"]
        try:
            code = module.main(base)
            out[name] = {"exit": code, "output": str(out_path)}
        except Exception as exc:  # noqa: BLE001
            out[name] = {"error": str(exc)}
    return out


if __name__ == "__main__":
    sys.exit(main())
