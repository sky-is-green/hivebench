"""P9 — Densest-Duplicate Hypothesis (deterministic A/B, no LLM auditor).

The prediction: when semantically duplicate chunks merge (cosine > 0.92),
keeping the information-densest version beats keeping the most recent one —
same fact, fewer tokens, so the fixed budget holds more signal. Measured as
sufficiency-per-1k-tokens (fact presence weighted by the kept copy's token
cost), following the project's deterministic fact-presence convention (P2/P3).

Corpus: ``tests/fixtures/generated_p9`` (``generate --p9``) — conversations
where each aspect is stated once DENSE (~33 tokens) and once VERBOSE (~57
tokens, pair cosine > 0.92 — engineered to merge), in two orders:

  - "recency_favors_verbose": dense first, verbose restatement later — a
    recency-keeping policy would keep the verbose copy,
  - "control": verbose first, dense later — both policies keep the dense.

A/B: the SAME conversations run through assembly twice — with the real
densest-keeping dedup vs a recency-keeping variant (identical threshold and
refresh semantics, only the keep decision differs). Per recap turn, both
policies are scored on sufficiency-per-1k-tokens.

Falsification (paper P9): recency retention wins on >=55% of turns, or no
measurable difference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HIVEBENCH_ROOT = Path(__file__).resolve().parents[1]
P9_DIR = HIVEBENCH_ROOT / "tests" / "fixtures" / "generated_p9"
DUPLICATE_THRESHOLD = 0.92


def _cosine_matrix(rows: list) -> np.ndarray:
    m = np.asarray(rows, dtype=float)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    sim = m @ m.T
    denom = norms @ norms.T
    denom = np.where(denom == 0, 1e-9, denom)
    return sim / denom


def deduplicate(chunks: list, embeddings: dict, keep: str) -> tuple[list, dict]:
    """Dedup with a selectable keep policy: "densest" (real ContextDeduplicator
    semantics) or "recency" (keep the later chunk). Identical threshold and
    refresh_map semantics for both."""
    if len(chunks) <= 1:
        return chunks, {}
    sim = _cosine_matrix([embeddings[c.id] for c in chunks])
    keep_set = set(range(len(chunks)))
    refresh_map: dict[str, int] = {}
    for i in range(len(chunks)):
        if i not in keep_set:
            continue
        for j in range(i + 1, len(chunks)):
            if j not in keep_set:
                continue
            if sim[i][j] > DUPLICATE_THRESHOLD:
                if keep == "recency":
                    keep_idx, discard = (i, j) if chunks[i].turn >= chunks[j].turn else (j, i)
                else:
                    di = _info_density(chunks[i].content)
                    dj = _info_density(chunks[j].content)
                    keep_idx, discard = (i, j) if di >= dj else (j, i)
                keep_set.discard(discard)
                freshest = max(chunks[i].turn, chunks[j].turn)
                refresh_map[chunks[keep_idx].id] = max(
                    refresh_map.get(chunks[keep_idx].id, 0), freshest)
    return [chunks[i] for i in sorted(keep_set)], refresh_map


def _info_density(content: str) -> float:
    import re

    from sieve.vocabulary import Vocabulary

    words = re.findall(r"\w+", content)
    total_words = len(words) or 1
    term_count = len(Vocabulary.load("code", "general").matched_terms(content))
    sentences = re.split(r"(?<=[.!?])\s+", content.strip())
    avg_sent_len = sum(len(re.findall(r"\w+", s)) for s in sentences) / (
        len(sentences) or 1
    )
    return (term_count / total_words) * avg_sent_len


def _fact_cost(selected_chunks: list, facts: set, store) -> tuple[bool, int]:
    """(full_fact_present, token cost of the chunk(s) carrying the facts).

    Full presence = the facts appear in the assembled context (strict
    ``facts <= ctx terms``, the P3 convention — partial overlaps are not
    sufficiency). Cost = the tokens of the selected chunk(s) that carry the
    facts: a single covering chunk when there is one, else the summed tokens
    of the chunks that contain any fact term (split retrieval). 0 when the
    facts are absent."""
    from cortex.baselines.metrics import estimate_tokens
    from experiments.retrieval_diagnostic import _content_terms

    covering = []
    partial = []
    for c in selected_chunks:
        terms = _content_terms(c.content)
        if facts <= terms:
            covering.append(c)
        elif facts & terms:
            partial.append(c)
    if covering:
        return True, min(estimate_tokens(c.content) for c in covering)
    if partial:
        union = set()
        for c in partial:
            union |= _content_terms(c.content)
        if facts <= union:
            return True, sum(estimate_tokens(c.content) for c in partial)
    return False, 0


def run_policy(convs: list[dict], ultra, keep: str) -> tuple[list[dict], int]:
    """Run assembly over the P9 corpus with the given dedup keep policy;
    return (per-RECAP-TURN measurements, total dedup merges observed).

    Only recap-phase turns are scored (turn > 2 * aspects_per_conv): the
    verbose turns' answers embed filler words ("We noted this in the notes...")
    that become fact terms and are only carried by the verbose copies —
    measuring them would conflate filler presence with fact retrieval. The
    recap answers are canonical ("The {aspect} for the {feature} is
    {decision}."), so recap facts are pure decision terms."""
    from cortex.baselines.metrics import estimate_tokens
    from cortex.splinter import Splinter
    from cortex.routing import DroneRouter, EscalationHandler
    from experiments.retrieval_diagnostic import (
        _answer_fact_terms, _content_terms, _fixture_answer_map,
    )
    from focal.assembly import ContextAssembler
    from focal.budget import AdaptiveBudget
    from membrane.drift import TopicDriftDetector
    from retention.store import ContextStore
    from sieve.medium import MediumDrone
    from tests.fixtures.synthetic_conversations.generate import P9_ASPECTS_PER_CONV

    ans = _fixture_answer_map(convs)
    rows: list[dict] = []
    merges = 0
    for conv in convs:
        conv_answers = ans.get(conv.get("conversation_id", "unknown"), {})
        store = ContextStore(embed_fn=ultra.embed, max_chunks=1000)
        prior_fixture = ""
        turn = 0
        recap_start = 2 * conv.get("aspects_per_conv", P9_ASPECTS_PER_CONV)
        for td in conv["turns"]:
            if td["role"] != "user":
                prior_fixture += " " + (td.get("content") or "")
                continue
            turn += 1
            q = td["content"]
            answer = conv_answers.get(q, "")
            facts = _answer_fact_terms(q, answer) if answer else set()
            prior_terms = _content_terms(prior_fixture)
            dedup = _PolicyDedup(keep)
            ctx = ContextAssembler().assemble(
                query=q, current_turn=turn, store=store,
                router=DroneRouter(), ultra_small=ultra,
                medium=MediumDrone(score_pair_fn=lambda qq, cc: 0.5),
                escalation=EscalationHandler(),
                dedup=dedup,
                drift_detector=TopicDriftDetector(embed_fn=ultra.embed),
                budget=AdaptiveBudget(), max_context=8192,
            )
            merges += dedup.merge_count
            if turn > recap_start and facts and facts <= prior_terms:
                selected = [store.chunks[cid] for cid in ctx.selected_chunk_ids]
                hit, cost = _fact_cost(selected, facts, store)
                rows.append({
                    "conversation_id": conv.get("conversation_id"),
                    "turn": turn,
                    "order": conv.get("order"),
                    "hit": hit,
                    "fact_tokens": cost,
                    "per_1k": (1000.0 / cost) if hit and cost else 0.0,
                })
            store.add_chunk(turn, q)
            reply = (conv_answers.get(q) or "").strip() or ""
            if reply and not Splinter._is_hedge_reply(reply):
                store.add_chunk(turn, reply)
    return rows, merges


class _PolicyDedup:
    """Adapter exposing the keep-policy dedup through the assembler's interface."""

    def __init__(self, keep: str) -> None:
        self.keep = keep
        self.merge_count = 0

    def deduplicate(self, chunks, embeddings):
        surviving, refresh_map = deduplicate(chunks, embeddings, self.keep)
        self.merge_count += len(chunks) - len(surviving)
        return surviving, refresh_map


def run_ab(convs: list[dict], ultra) -> dict:
    """Deterministic A/B: densest-kept vs recency-kept on the same corpus.

    The verdict uses the *informative* turns — the recency_favors_verbose
    conversations, where the policies genuinely disagree about which copy to
    keep. The control conversations (both policies keep the dense copy) are
    reported separately as the no-effect control.
    """
    densest, d_merges = run_policy(convs, ultra, "densest")
    recency, r_merges = run_policy(convs, ultra, "recency")
    assert len(densest) == len(recency)

    def tally(rows_d, rows_r):
        d_wins = r_wins = ties = 0
        d_hits = r_hits = 0
        d_tokens = r_tokens = 0
        for d, r in zip(rows_d, rows_r):
            if d["per_1k"] > r["per_1k"]:
                d_wins += 1
            elif r["per_1k"] > d["per_1k"]:
                r_wins += 1
            else:
                ties += 1
            d_hits += int(d["hit"])
            r_hits += int(r["hit"])
            d_tokens += d["fact_tokens"]
            r_tokens += r["fact_tokens"]
        return d_wins, r_wins, ties, d_hits, r_hits, d_tokens, r_tokens

    info = [(d, r) for d, r in zip(densest, recency) if d["order"] != "control"]
    ctrl = [(d, r) for d, r in zip(densest, recency) if d["order"] == "control"]
    dw, rw, ties, dh, rh, dt, rt = tally(*zip(*info)) if info else (0, 0, 0, 0, 0, 0, 0)
    cdw, crw, cties, cdh, crh, cdt, crt = tally(*zip(*ctrl)) if ctrl else (0, 0, 0, 0, 0, 0, 0)
    n = dw + rw + ties
    return {
        "informative_turns": n,
        "control_turns": cdw + crw + cties,
        "merges": d_merges + r_merges,
        "densest_wins": dw,
        "recency_wins": rw,
        "ties": ties,
        "densest_share": round(dw / n, 3) if n else None,
        "recency_share": round(rw / n, 3) if n else None,
        "densest_hits": dh,
        "recency_hits": rh,
        "densest_fact_tokens": dt,
        "recency_fact_tokens": rt,
        "densest_per_1k": round(1000.0 * dh / dt, 1) if dt else None,
        "recency_per_1k": round(1000.0 * rh / rt, 1) if rt else None,
        "control_densest_per_1k": round(1000.0 * cdh / cdt, 1) if cdt else None,
        "control_recency_per_1k": round(1000.0 * crh / crt, 1) if crt else None,
    }


def verdict(ab: dict) -> tuple[str, str]:
    """PAPER P9 falsification: recency wins on >=55% of turns, or no
    measurable difference (<5 points). Otherwise the densest policy's
    advantage is confirmed. The verdict uses the informative (non-control)
    turns only, and requires at least one observed dedup merge (no merges =
    the corpus's duplicates didn't register — unmeasurable, e.g. a fake
    drone)."""
    ds, rs = ab["densest_share"], ab["recency_share"]
    if ab.get("merges", 0) == 0:
        return "SKIP", "no duplicate pairs merged (corpus or drone issue)"
    if ds is None or rs is None or ab["informative_turns"] == 0:
        return "SKIP", "no measured turns"
    if rs >= 0.55:
        return "FAIL", "recency wins on >=55% of turns"
    if abs(ds - rs) <= 0.05:
        return "FAIL", "no measurable difference (<=5 points)"
    return "PASS", f"densest wins on {ds:.1%} of turns vs recency {rs:.1%}"


def main() -> int:
    parser = argparse.ArgumentParser(description="P9 densest-duplicate A/B")
    parser.add_argument("--conversations", default=str(P9_DIR))
    args = parser.parse_args()
    from cortex.baselines.runner import load_conversations
    from sieve.ultra_small import UltraSmallDrone

    convs = load_conversations(args.conversations)
    if not convs:
        print("P9 corpus missing — run `python -m "
              "tests.fixtures.synthetic_conversations.generate --p9`")
        return 2
    ultra = UltraSmallDrone(confidence_mode="off")
    ab = run_ab(convs, ultra)
    status, note = verdict(ab)
    print(f"P9 densest-duplicate A/B ({ab['informative_turns']} informative turns, "
          f"{ab['control_turns']} control turns):")
    print(f"  densest wins: {ab['densest_wins']} ({ab['densest_share']:.1%})  "
          f"recency wins: {ab['recency_wins']} ({ab['recency_share']:.1%})  "
          f"ties: {ab['ties']}")
    print(f"  sufficiency per 1k tokens: densest {ab['densest_per_1k']} vs "
          f"recency {ab['recency_per_1k']} "
          f"(fact tokens: {ab['densest_fact_tokens']} vs {ab['recency_fact_tokens']})")
    print(f"  control (both keep dense): {ab['control_densest_per_1k']} vs "
          f"{ab['control_recency_per_1k']} per 1k — no effect expected")
    print(f"  verdict: {status} — {note}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())