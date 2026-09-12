"""Paired live A/B: strata-curated context vs the naive FIFO window on the same
turns — the "improvement over the current standard on LLM performance"
measurement the paper's PES cannot make alone.

For every retrievable turn with a fixture ground-truth answer, the same model
generates two replies:

  - arm H: context assembled by the strata (the store replays the live pipeline:
    query chunk + non-hedge reply chunk per turn)
  - arm F: the last-``fifo_budget``-token window of the same history

Each reply is then scored *deterministically* on answer-fact presence — did the
answer contain the fixture ground-truth facts (``_answer_fact_terms``)? Optionally
(``--auditor``) each arm is also scored for context sufficiency by the LLM auditor.

Metrics (over retrievable turns with measurable facts):

  - ``hive_answer_recall`` / ``fifo_answer_recall`` — share of turns whose
    *answer* contained the ground-truth facts (binary at hit-ratio >= 0.5,
    matching the retrieval diagnostic's convention)
  - ``hive_ge_fifo_ratio`` — turns where the strata answer carried the facts while
    the FIFO answer did not (strict win) OR both did; the answer-level analogue
    of P3's context-level A/B
  - ``strict_hive_only_ratio`` — strict wins only
  - context-level sufficiency columns (P3-style) are recorded for comparison

Usage::

    python -m experiments.paired_ab --mock        # offline harness verification
    python -m experiments.paired_ab --live --max-convs 10 --max-turns 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from backend.cache_manager import KVCacheManager
from backend.lmstudio import LMStudioBackend
from backend.sampling import parse_sampling
from cortex.baselines.runner import FIFO_WINDOW_TOKENS, load_conversations
from cortex.baselines.metrics import estimate_tokens
from cortex.e2e import FakeUltraSmall, MockTransport
from cortex.strata import Strata
from cortex.routing import DroneRouter, EscalationHandler
from focal.assembly import ContextAssembler
from focal.budget import AdaptiveBudget
from membrane.dedup import ContextDeduplicator
from membrane.drift import TopicDriftDetector
from auditor.auditor import Auditor, TurnRecord
from retention.store import ContextStore
from sieve.medium import MediumDrone
from sieve.ultra_small import UltraSmallDrone


def _fifo_context(history: list[dict], budget_tokens: int = FIFO_WINDOW_TOKENS) -> str:
    """Naive FIFO truncation: keep the most recent messages that fit the window
    (the current user turn is newest, so always retained). Mirrors
    ``cortex.baselines.runner.build_fifo_messages``."""
    total = 0
    kept: list[dict] = []
    for msg in reversed(history):
        cost = estimate_tokens(msg.get("content", ""))
        if kept and total + cost > budget_tokens:
            continue
        kept.append(msg)
        total += cost
    return "\n\n".join(m.get("content", "") for m in reversed(kept))


def _answer_fact_scoring(facts, *texts):
    """(hit_ratio, suff) for each text: share of facts present, and binary
    presence. Mirrors the retrieval diagnostic's hit convention."""
    out = []
    for text in texts:
        from experiments.retrieval_diagnostic import _content_terms

        terms = _content_terms(text or "")
        present = facts & terms if facts else set()
        hit = len(present) / len(facts) if facts else 0.0
        out.append((hit, bool(facts) and present == facts))
    return out


def run_paired(conversations, backend, ultra, medium, sampling=None,
               max_turns=None, fifo_budget: int = FIFO_WINDOW_TOKENS,
               auditor=None, verbose: bool = False,
               checkpoint_path: str | Path | None = None,
               checkpoint_every: int = 5,
               resume: dict | None = None,
               live_store: bool = False) -> dict:
    """Run the paired A/B and return the report dict (metrics + per-turn rows).

    Long live runs can be resumed: ``checkpoint_path`` writes a JSON checkpoint
    every ``checkpoint_every`` user turns and at each conversation boundary;
    ``resume`` (a loaded checkpoint dict) skips completed conversations/turns
    and restores the current conversation's store and prior history.

    ``live_store=False`` (default) replays the strata store from the fixture
    answers, so both arms see the identical history and the comparison
    isolates *selection*: does curated context beat the naive window? The
    earlier live-replay mode (``live_store=True``, where the store grew from
    the strata arm's live replies) was asymmetric — FIFO's context is the
    canonical fixture history while the strata's store held whatever the model
    actually said, which under-credits the strata (the paper's ingestion
    problem). Both modes remain available; the fair selection test is the
    default.
    """
    from experiments.retrieval_diagnostic import (
        _answer_fact_terms,
        _fixture_answer_map,
        _is_retrievable,
    )

    ans = _fixture_answer_map(conversations)

    def save_checkpoint(ci, turn_index, store, prior):
        if checkpoint_path is None:
            return
        ckpt_path = Path(checkpoint_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        ckpt_path.write_text(json.dumps({
            "version": 2,
            "conv_index": ci,
            "turn_index": turn_index,
            "prior": prior,
            "store": store.to_dict(),
            "rows": rows,
            "first_mention": first_mention,
            "no_facts": no_facts,
            "done_turns": done_turns,
            "live_store": live_store,
        }, indent=2, default=str), encoding="utf-8")
        # A killed run should still leave readable partial results: mirror the
        # final report shape (minus fidelity, computed only at the end) beside
        # the checkpoint.
        partial = {
            "partial": True,
            "turns_compared": len(rows),
            "first_mention_excluded": first_mention,
            "no_facts_excluded": no_facts,
            "turns": rows,
        }
        if ckpt_path.name.endswith(".ckpt.json"):
            report_path = ckpt_path.with_name(ckpt_path.name[: -len(".ckpt.json")] + ".json")
            report_path.write_text(json.dumps(partial, indent=2), encoding="utf-8")

    assembler = ContextAssembler()
    rows = list(resume.get("rows", [])) if resume else []
    first_mention = resume.get("first_mention", 0) if resume else 0
    no_facts = resume.get("no_facts", 0) if resume else 0
    done_turns = resume.get("done_turns", 0) if resume else 0

    for ci, conv in enumerate(conversations):
        if resume is not None and ci < resume.get("conv_index", 0):
            continue
        cid = conv.get("conversation_id", "unknown")
        conv_answers = ans.get(cid, {})
        store = ContextStore(embed_fn=ultra.embed)
        if resume is not None and ci == resume.get("conv_index", 0):
            store = ContextStore.from_dict(resume["store"], embed_fn=ultra.embed)
        turn = 0
        prior_parts: list[str] = []
        if resume is not None and ci == resume.get("conv_index", 0):
            prior_parts = list(resume.get("prior", []))
        for td in conv.get("turns", []):
            if td.get("role") == "user":
                turn += 1
                if resume is not None and ci == resume.get("conv_index", 0) \
                        and turn <= resume.get("turn_index", 0):
                    continue
                if max_turns and turn > max_turns:
                    break
                q = td["content"]
                answer = conv_answers.get(q, "")
                facts = _answer_fact_terms(q, answer) if answer else set()
                stored_reply = ""
                if not facts:
                    no_facts += 1
                elif not _is_retrievable(q, " ".join(prior_parts)):
                    first_mention += 1
                    # Skipped turns still grow the store like the live pipeline;
                    # the fixture answer stands in for the reply (P3's replay
                    # convention — no LLM call is spent on unscorable turns).
                    stored_reply = answer
                else:
                    if verbose:
                        print(f"  turn {turn} [{cid}] strata vs fifo ...", flush=True)
                    hive_ctx = assembler.assemble(
                        query=q, current_turn=turn, store=store,
                        router=DroneRouter(), ultra_small=ultra, medium=medium,
                        escalation=EscalationHandler(), dedup=ContextDeduplicator(),
                        drift_detector=TopicDriftDetector(embed_fn=ultra.embed),
                        budget=AdaptiveBudget(), max_context=8192,
                    ).content
                    history = [{"role": t["role"], "content": t["content"]}
                               for t in conv["turns"][: 2 * turn - 1]]
                    fifo_ctx = _fifo_context(history, fifo_budget)

                    reply_h = backend.generate(hive_ctx, q, sampling or None) or ""
                    reply_f = backend.generate(fifo_ctx, q, sampling or None) or ""
                    # Fair-selection default: both arms' stores replay the same
                    # canonical history (fixture answers). live_store=True
                    # keeps the asymmetric live-replay mode for the full-system
                    # measurement (ingestion + selection together).
                    stored_reply = reply_h if live_store else answer

                    ctx_h_hit, ctx_h_suff = _answer_fact_scoring(facts, hive_ctx)[0]
                    ctx_f_hit, ctx_f_suff = _answer_fact_scoring(facts, fifo_ctx)[0]
                    ans_h_hit, ans_h_suff = _answer_fact_scoring(facts, reply_h)[0]
                    ans_f_hit, ans_f_suff = _answer_fact_scoring(facts, reply_f)[0]

                    row = {
                        "turn": turn,
                        "conversation_id": cid,
                        "query": q,
                        "hive_ctx_tokens": estimate_tokens(hive_ctx),
                        "fifo_ctx_tokens": estimate_tokens(fifo_ctx),
                        "hive_ctx": hive_ctx[:2000],
                        "fifo_ctx": fifo_ctx[:2000],
                        "ctx_hive_suff": ctx_h_suff,
                        "ctx_fifo_suff": ctx_f_suff,
                        "answer_hive_hit_ratio": round(ans_h_hit, 3),
                        "answer_fifo_hit_ratio": round(ans_f_hit, 3),
                        "answer_hive_suff": ans_h_suff,
                        "answer_fifo_suff": ans_f_suff,
                        "reply_hive": reply_h[:500],
                        "reply_fifo": reply_f[:500],
                    }
                    if auditor is not None:
                        row["auditor"] = {}
                        for arm, ctx, reply in (("strata", hive_ctx, reply_h),
                                                ("fifo", fifo_ctx, reply_f)):
                            try:
                                label = auditor.evaluate_turn(TurnRecord(
                                    turn=turn, assembled_context=ctx, user_query=q,
                                    llm_response=reply, chunk_ids=[],
                                ))
                                row["auditor"][arm] = {
                                    "sufficient": label.context_sufficient,
                                    "score": label.sufficiency_score,
                                }
                            except Exception as exc:  # noqa: BLE001
                                row["auditor"][arm] = {"error": str(exc)[:200]}
                    rows.append(row)

                # Grow the strata store like the live pipeline on every user turn:
                # query chunk always, the reply chunk (strata arm's live reply, or
                # the fixture answer for skipped turns) unless it is a hedge.
                store.add_chunk(turn, q)
                if stored_reply and not Strata._is_hedge_reply(stored_reply):
                    store.add_chunk(turn, stored_reply)
                prior_parts.append(td.get("content", "") or "")
                done_turns += 1
                if checkpoint_path is not None and done_turns % checkpoint_every == 0:
                    save_checkpoint(ci, turn, store, prior_parts)
            else:
                prior_parts.append(td.get("content", "") or "")
        # conversation boundary checkpoint: next conv starts fresh
        if checkpoint_path is not None:
            save_checkpoint(ci + 1, 0, store, [])

    n = len(rows)
    if n == 0:
        return {
            "turns_compared": 0,
            "first_mention_excluded": first_mention,
            "no_facts_excluded": no_facts,
            "metrics": {},
            "turns": [],
        }

    from experiments.retrieval_diagnostic import _content_terms

    def _fidelity(reply: str, ctx: str) -> float:
        """Share of the answer's distinctive terms that came from its own
        context — isolates the curation's contribution to the answer."""
        r_terms = _content_terms(reply or "")
        if not r_terms:
            return 0.0
        c_terms = _content_terms(ctx or "")
        return len(r_terms & c_terms) / len(r_terms)

    for r in rows:
        r["fidelity_hive"] = round(_fidelity(r["reply_hive"], r["hive_ctx"]), 3)
        r["fidelity_fifo"] = round(_fidelity(r["reply_fifo"], r["fifo_ctx"]), 3)

    hive_only = fifo_only = both = neither = 0
    ctx_hive_only = ctx_fifo_only = ctx_both = ctx_neither = 0
    for r in rows:
        if r["answer_hive_suff"] and not r["answer_fifo_suff"]:
            hive_only += 1
        elif r["answer_fifo_suff"] and not r["answer_hive_suff"]:
            fifo_only += 1
        elif r["answer_hive_suff"] and r["answer_fifo_suff"]:
            both += 1
        else:
            neither += 1
        if r["ctx_hive_suff"] and not r["ctx_fifo_suff"]:
            ctx_hive_only += 1
        elif r["ctx_fifo_suff"] and not r["ctx_hive_suff"]:
            ctx_fifo_only += 1
        elif r["ctx_hive_suff"] and r["ctx_fifo_suff"]:
            ctx_both += 1
        else:
            ctx_neither += 1

    fact_retrievable = hive_only + fifo_only + both
    ctx_fact_retrievable = ctx_hive_only + ctx_fifo_only + ctx_both
    ans_recall_h = sum(1 for r in rows if r["answer_hive_hit_ratio"] >= 0.5) / n
    ans_recall_f = sum(1 for r in rows if r["answer_fifo_hit_ratio"] >= 0.5) / n
    avg_hit_h = sum(r["answer_hive_hit_ratio"] for r in rows) / n
    avg_hit_f = sum(r["answer_fifo_hit_ratio"] for r in rows) / n

    metrics = {
        "turns_compared": n,
        "first_mention_excluded": first_mention,
        "no_facts_excluded": no_facts,
        "hive_answer_recall": round(ans_recall_h * 100.0, 1),
        "fifo_answer_recall": round(ans_recall_f * 100.0, 1),
        "hive_avg_fact_hit_ratio": round(avg_hit_h, 3),
        "fifo_avg_fact_hit_ratio": round(avg_hit_f, 3),
        "hive_avg_context_fidelity": round(
            sum(r["fidelity_hive"] for r in rows) / n, 3),
        "fifo_avg_context_fidelity": round(
            sum(r["fidelity_fifo"] for r in rows) / n, 3),
        "fidelity_hive_gt_fifo_ratio": round(
            sum(1 for r in rows if r["fidelity_hive"] > r["fidelity_fifo"]) / n * 100.0, 1),
        "fidelity_hive_ge_fifo_ratio": round(
            sum(1 for r in rows if r["fidelity_hive"] >= r["fidelity_fifo"]) / n * 100.0, 1),
        "hive_only": hive_only,
        "fifo_only": fifo_only,
        "both_sufficient": both,
        "neither_sufficient": neither,
        "hive_ge_fifo_ratio": round(
            (hive_only + both) / fact_retrievable * 100.0, 1) if fact_retrievable else 0.0,
        "strict_hive_only_ratio": round(
            hive_only / fact_retrievable * 100.0, 1) if fact_retrievable else 0.0,
        "ctx_hive_only": ctx_hive_only,
        "ctx_fifo_only": ctx_fifo_only,
        "ctx_both_sufficient": ctx_both,
        "ctx_neither_sufficient": ctx_neither,
        "ctx_hive_ge_fifo_ratio": round(
            (ctx_hive_only + ctx_both) / ctx_fact_retrievable * 100.0, 1)
            if ctx_fact_retrievable else 0.0,
        "fifo_budget_tokens": fifo_budget,
    }
    if auditor is not None:
        qh = [r["auditor"]["strata"] for r in rows if "strata" in r.get("auditor", {})
              and "sufficient" in r["auditor"]["strata"]]
        qf = [r["auditor"]["fifo"] for r in rows if "fifo" in r.get("auditor", {})
              and "sufficient" in r["auditor"]["fifo"]]
        metrics["auditor_hive_sufficient_rate"] = round(
            sum(1 for x in qh if x["sufficient"]) / len(qh) * 100.0, 1) if qh else None
        metrics["auditor_fifo_sufficient_rate"] = round(
            sum(1 for x in qf if x["sufficient"]) / len(qf) * 100.0, 1) if qf else None

    return {"metrics": metrics, "turns": rows}


def _live_auditor(backend):
    def fn(prompt: str) -> str:
        backend.pinned_prefix = ""
        return backend.generate(
            "You are an evaluation engine. Respond with ONLY a valid JSON object.",
            prompt, {"temperature": 0.0, "max_tokens": 2048},
        )
    return fn


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Paired live A/B: strata context vs FIFO window on the same turns")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--conversations", default="hivebench/tests/fixtures/generated")
    parser.add_argument("--max-convs", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--base-url", default="http://localhost:1234")
    parser.add_argument("--model", default="")
    parser.add_argument("--provider", default="",
                        help="provider name from the providers config: fills "
                             "base_url/api_key/model (explicit flags win)")
    parser.add_argument("--providers-file", default="",
                        help="path to the providers JSON config")
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--sampling", default="",
        help="sampling overrides as JSON, e.g. --sampling "
             "'{\"temperature\":0.7}' (backend.sampling fields)",
    )
    parser.add_argument(
        "--auditor", action="store_true",
        help="also score each arm's context sufficiency with the LLM auditor "
             "(doubles label-phase LLM cost; mock mode uses the mock auditor)",
    )
    parser.add_argument(
        "--fifo-budget", type=int, default=FIFO_WINDOW_TOKENS,
        help="FIFO window size in tokens (default %(default)s)",
    )
    parser.add_argument(
        "--confidence", choices=("mcdropout", "single", "off"), default="mcdropout",
        help="drone confidence mode; 'off' skips the MC-dropout passes (the stock "
             "embedding model yields confidence ~1.0 regardless)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=None,
        help="cap both arms' reply length (default: uncapped). On reasoning "
             "models a small cap yields empty replies unless --no-thinking is "
             "used; the fixture-fact scoring only needs the facts named",
    )
    parser.add_argument(
        "--no-thinking", action="store_true",
        help="send enable_thinking=false with every request, so reasoning models "
             "skip chain-of-thought (faster, and reply caps stop yielding empty "
             "output). Combine with LM Studio's 'thinking' toggle for models "
             "that only honor the GUI setting.",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=5,
        help="write a resume checkpoint every N user turns",
    )
    parser.add_argument(
        "--checkpoint", default="",
        help="resume-checkpoint output path (default: <output>.ckpt.json)",
    )
    parser.add_argument(
        "--resume", default="",
        help="resume a checkpoint file written by a previous (killed) run",
    )
    parser.add_argument(
        "--live-store", action="store_true",
        help="grow the strata store from the strata arm's live replies instead of "
             "the fixture answers. Default (fixture replay) is the fair "
             "selection test — both arms see identical history; this flag "
             "restores the asymmetric live-replay measurement (ingestion + "
             "selection together, at the cost of comparing against a store "
             "that lacks facts the model never stated)",
    )
    args = parser.parse_args(argv)

    if args.provider:
        from backend.providers import apply_provider_overrides, backend_kwargs, load_registry

        try:
            prov = load_registry(args.providers_file or None).resolve(args.provider)
        except LookupError as exc:
            print(f"error: {exc}")
            return 2
        apply_provider_overrides(vars(parser.parse_args([])), args, prov)
        pkw = backend_kwargs(prov)
    else:
        pkw = {}

    conversations = load_conversations(args.conversations)
    if args.max_convs:
        conversations = conversations[: args.max_convs]
    if not conversations:
        print(f"No conversation files found in {args.conversations}")
        return 2

    if args.mock or not args.live:
        ultra = FakeUltraSmall()
        backend = LMStudioBackend(base_url=args.base_url, model=args.model,
                                  api_key=pkw.get("api_key", "lm-studio"),
                                  extra_headers=pkw.get("extra_headers"),
                                  transport=MockTransport())
        live = False
    else:
        ultra = UltraSmallDrone(confidence_mode=args.confidence)
        ultra._ensure_loaded()
        backend = LMStudioBackend(base_url=args.base_url, model=args.model,
                                  api_key=pkw.get("api_key", "lm-studio"),
                                  extra_headers=pkw.get("extra_headers"),
                                  disable_thinking=args.no_thinking)
        if not backend.health():
            print(f"LM Studio not reachable at {args.base_url}")
            return 3
        live = True

    medium = MediumDrone(score_pair_fn=lambda q, c: 0.5)
    auditor = None
    if args.auditor:
        auditor = Auditor(generate_fn=_live_auditor(backend) if live else
                      (lambda p: json.dumps(
                          {"sufficient": True, "used_pieces": [], "missing": [],
                           "score": 4})))
    sampling = parse_sampling(args.sampling) if args.sampling else None
    if args.max_tokens:
        sampling = {**(sampling or {}), "max_tokens": args.max_tokens}

    from experiments.dashboard import KeepAwake

    keep_awake = KeepAwake() if live else None
    if keep_awake is not None and keep_awake.active:
        print("keep-awake: system sleep disabled for this run (ES_SYSTEM_REQUIRED)")

    resume = None
    if args.resume:
        ckpt_path = Path(args.resume)
        if not ckpt_path.exists():
            print(f"no checkpoint found at {args.resume}")
            return 3
        resume = json.loads(ckpt_path.read_text(encoding="utf-8"))
        if not args.live_store and resume.get("live_store"):
            args.live_store = True
            print("restored --live-store from checkpoint (store was built in live-replay mode)")
        print(f"resuming from {args.resume}: conv "
              f"{resume.get('conv_index', 0) + 1}, turn {resume.get('turn_index', 0)}, "
              f"{len(resume.get('rows', []))} rows done")

    report = run_paired(conversations, backend, ultra, medium, sampling=sampling,
                        max_turns=args.max_turns, fifo_budget=args.fifo_budget,
                        auditor=auditor, verbose=True,
                        checkpoint_path=(
                            Path(args.checkpoint) if args.checkpoint else
                            (Path(args.output) if args.output else
                             Path("logs") / "paired_ab.json").with_suffix(".ckpt.json")
                        ),
                        checkpoint_every=args.checkpoint_every,
                        resume=resume, live_store=args.live_store)

    if keep_awake is not None:
        keep_awake.close()

    out = Path(args.output) if args.output else Path("logs") / "paired_ab.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    m = report["metrics"]
    if not m:
        print("Paired A/B: no measurable turns "
              f"({report['first_mention_excluded']} first-mention excluded, "
              f"{report['no_facts_excluded']} no-facts excluded)")
        print(f"Wrote {out.resolve()}")
        return 0
    print("Paired A/B — strata context vs FIFO window, same model, same turns")
    print(f"  turns compared      : {m['turns_compared']} "
          f"(first-mention excluded {m['first_mention_excluded']})")
    print(f"  answer recall       : strata {m['hive_answer_recall']}% "
          f"vs FIFO {m['fifo_answer_recall']}% "
          f"(fixture-fact presence in replies)")
    print(f"  avg fact hit ratio  : strata {m['hive_avg_fact_hit_ratio']} "
          f"vs FIFO {m['fifo_avg_fact_hit_ratio']}")
    print(f"  context fidelity    : strata {m['hive_avg_context_fidelity']} "
          f"vs FIFO {m['fifo_avg_context_fidelity']} "
          f"(answer terms sourced from the arm's own context)")
    print(f"  fidelity strata > FIFO: {m['fidelity_hive_gt_fifo_ratio']}% of turns "
          f"(>= {m['fidelity_hive_ge_fifo_ratio']}%)")
    print(f"  strata >= FIFO answers: {m['hive_ge_fifo_ratio']}% "
          f"(strict strata-only {m['strict_hive_only_ratio']}%)")
    print(f"  answer buckets      : hive_only {m['hive_only']} / "
          f"fifo_only {m['fifo_only']} / both {m['both_sufficient']} / "
          f"neither {m['neither_sufficient']}")
    print(f"  context sufficiency : strata {m['ctx_hive_ge_fifo_ratio']}% "
          f">= FIFO (P3-style)")
    if m.get("auditor_hive_sufficient_rate") is not None:
        print(f"  auditor sufficiency   : strata {m['auditor_hive_sufficient_rate']}% "
              f"vs FIFO {m['auditor_fifo_sufficient_rate']}%")
    print(f"Wrote {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())