"""Comb probe (P11 make-or-break): real-drone measurements for the return regime.

Answers the three questions that decide whether the comb can work, with the
real default drone (paraphrase-MiniLM-L3-v2) and deterministic fixture ground
truth (no auditor):

  Q1 retrieval : on topic-return turns, does comb-style retrieval (lexical
                 pre-filter + drone ranking, as CombStore.retrieve does) put
                 the relevant archived records in top-k?
  Q2 gate      : on return turns, what is the active store's best raw drone
                 score? If it rarely falls below comb_gate_threshold (0.5),
                 the gate never fires and the comb is never consulted — the
                 calibration question.
  Q3 crowding  : on non-return turns, do archived records outrank relevant
                 active-store chunks? (P11 falsification clause 2)

Ground truth: a chunk is relevant to a query iff it contains a fact term of
the fixture's ground-truth answer (retrieval_diagnostic's math). A *return*
turn is a query whose relevant fact was stated >= RETURN_AGE turns earlier —
the regime the stale wall walls off. The long fixture conversations contain
real return turns (e.g. "How does rollbacks fit with order schema:
normalization=3NF...", asked ~40 turns after the order-schema facts).

Usage::

    python -m experiments.comb_probe --json models/b/comb_probe_l3.json
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
import sys
from pathlib import Path

import numpy as np

from experiments.retrieval_diagnostic import (
    _answer_fact_terms,
    _content_terms,
    _fixture_answer_map,
)

HIVEBENCH_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = HIVEBENCH_ROOT / "tests" / "fixtures" / "generated"

RETURN_AGE = 15       # a fact is "returned" when last stated this many turns ago
STORE_WINDOW = 10     # recent turns count as the active store
MAX_SCORE = 100       # comb.retrieve's drone-pass cap (lexical-rank preselected)
WORD_RE = re.compile(r"[a-z0-9]{4,}")


# ---------------------------------------------------------------------------
# Case construction (deterministic, fixture ground truth)
# ---------------------------------------------------------------------------
def _load_long_conversations() -> list[dict]:
    files = sorted(glob.glob(str(FIXTURE_DIR / "long_*.json")))
    if not files:
        raise SystemExit(f"no long fixture conversations in {FIXTURE_DIR}")
    return [json.loads(Path(f).read_text(encoding="utf-8")) for f in files]


def build_cases(return_age: int = RETURN_AGE, store_window: int = STORE_WINDOW):
    """Yield (query, store_chunks, archive_chunks, relevant_archive) per turn."""
    convs = _load_long_conversations()
    answers = _fixture_answer_map(convs)
    cases = []
    for conv in convs:
        cid = conv.get("conversation_id", "unknown")
        turns = conv.get("turns", [])
        stored: list[dict] = []  # {"content": ..., "turn": index}
        for i, t in enumerate(turns):
            if t.get("role") != "user":
                continue
            query = t.get("content", "")
            ans = answers.get(cid, {}).get(query, "")
            facts = _answer_fact_terms(query, ans) if ans else set()
            if facts and stored:
                prior = [s for s in stored if s["content"].strip()]
                relevant = [s for s in prior if facts & _content_terms(s["content"])]
                if relevant:
                    ages = [i - s["turn"] for s in relevant]
                    archive = [s for s in prior if i - s["turn"] > return_age]
                    relevant_archive = [s for s in relevant if i - s["turn"] > return_age]
                    store = [s for s in prior if i - s["turn"] <= store_window]
                    cases.append({
                        "conversation_id": cid,
                        "turn": i,
                        "query": query,
                        "facts": sorted(facts),
                        "store": [s["content"] for s in store],
                        "archive": [s["content"] for s in archive],
                        "relevant_archive": [s["content"] for s in relevant_archive],
                        "is_return": bool(relevant_archive),
                        "ages": sorted(ages),
                    })
            nxt = turns[i + 1] if i + 1 < len(turns) else None
            if nxt and nxt.get("role") == "assistant":
                stored.append({"content": nxt.get("content", ""), "turn": i + 1})
    return cases


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------
def _make_drone():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")
    model.eval()

    def score(query: str, chunks: list[str]) -> list[float]:
        if not chunks:
            return []
        q_vec = model.encode([query], normalize_embeddings=True)[0]
        vecs = model.encode(chunks, normalize_embeddings=True, batch_size=32)
        dots = np.asarray(vecs) @ q_vec
        return [float(d) for d in dots]

    return score


def _lexical_overlap(query: str, chunk: str) -> int:
    qw = set(WORD_RE.findall(query.lower()))
    cw = set(WORD_RE.findall(chunk.lower()))
    return len(qw & cw)


def _comb_retrieve(query: str, archive: list[str], score_fn=None,
                   max_score: int = MAX_SCORE):
    """Mirror CombStore.retrieve (2026-08-24): lexical overlap ranking, no
    drone pass (measured 2-3x better than the drone on return turns and
    ~100x cheaper). score_fn kept for the drone-comparison variant."""
    qwords = {w for w in WORD_RE.findall(query.lower())}
    if not qwords:
        return []
    ranked = sorted(
        (c for c in archive if _lexical_overlap(query, c) > 0),
        key=lambda c: _lexical_overlap(query, c),
        reverse=True,
    )
    return ranked[:max_score]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _gate_fraction(store_scores: list[float], threshold: float) -> float:
    if not store_scores:
        return 0.0
    return sum(1 for s in store_scores if s < threshold) / len(store_scores)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--return-age", type=int, default=RETURN_AGE)
    ap.add_argument("--max-score", type=int, default=MAX_SCORE)
    ap.add_argument("--json", default=None, help="write report to this path")
    args = ap.parse_args()

    cases = build_cases(return_age=args.return_age)
    score_fn = _make_drone()
    return_cases = [c for c in cases if c["is_return"]]
    non_return = [c for c in cases if not c["is_return"] and c["store"]]

    print(f"cases        : {len(cases)} turns  (return={len(return_cases)}, "
          f"non-return-with-store={len(non_return)})")

    # ---- Q1: retrieval on return turns ------------------------------------
    # Split into *lexically retrievable* turns (the relevant archived fact
    # shares >=1 content word with the query — the regime lexical retrieval
    # can work) and *artifact-labeled* turns (zero overlap; 55% of the current
    # fixture's return turns are artifact cases where the answer-map's
    # first-occurrence rule labels old-topic chunks as "relevant" to a query
    # that never lexically names them — those are corpus-design misses, not
    # retrieval failures; the --return corpus must build SHADOW-style
    # pure-fact queries that lexically name the old decision).
    ks = (1, 3, 5, 8)
    rec = {k: 0 for k in ks}
    retrievable = artifact = 0
    gate_top_scores = []
    for c in return_cases:
        rel = set(c["relevant_archive"])
        store_scores = score_fn(c["query"], c["store"]) if c["store"] else []
        gate_top_scores.append(max(store_scores) if store_scores else 1.0)
        if not any(_lexical_overlap(c["query"], ch) > 0 for ch in rel):
            artifact += 1
            continue
        retrievable += 1
        ranked = _comb_retrieve(c["query"], c["archive"], max_score=args.max_score)
        if not ranked:
            continue
        for k in ks:
            if set(ranked[:k]) & rel:
                rec[k] += 1
    n = len(return_cases)
    q1 = {
        "n_return_turns": n,
        "n_lexically_retrievable": retrievable,
        "n_artifact_labeled": artifact,
        "recall_on_retrievable": {
            str(k): round(rec[k] / retrievable, 4) if retrievable else None for k in ks
        },
        "returned_fact_ages": sorted(c["ages"][0] for c in return_cases),
    }
    print(f"Q1 retrieval : retrievable={retrievable}/{n} ({retrievable / n:.0%})  "
          f"artifact={artifact} ({artifact / n:.0%})")
    for k in ks:
        print(f"  recall@{k:<2} (retrievable) = {q1['recall_on_retrievable'][str(k)]}")

    # ---- Q2: gate calibration on return turns ------------------------------
    q2 = {
        "store_top_score_mean": round(statistics.mean(gate_top_scores), 4) if gate_top_scores else None,
        "store_top_score_p50": round(statistics.median(gate_top_scores), 4) if gate_top_scores else None,
        "gate_fires_at_0_4": round(_gate_fraction(gate_top_scores, 0.4), 4),
        "gate_fires_at_0_5": round(_gate_fraction(gate_top_scores, 0.5), 4),
        "gate_fires_at_0_6": round(_gate_fraction(gate_top_scores, 0.6), 4),
        "gate_fires_at_0_7": round(_gate_fraction(gate_top_scores, 0.7), 4),
    }
    print(f"Q2 gate      : store top-score p50={q2['store_top_score_p50']}  "
          f"fires@0.5={q2['gate_fires_at_0_5']}  fires@0.6={q2['gate_fires_at_0_6']}  "
          f"fires@0.7={q2['gate_fires_at_0_7']}")

    # ---- Q3: crowding on non-return turns ----------------------------------
    crowded = 0
    relevant_store_scores, archive_top_scores = [], []
    for c in non_return:
        if not c["store"] or not c["facts"]:
            continue
        # relevant store chunks: recent chunks carrying the answer's facts
        rel_recent = [s for s in c["store"] if set(c["facts"]) & _content_terms(s)]
        if not rel_recent:
            continue
        store_scores = score_fn(c["query"], c["store"])
        best_rel = max(s for s, chunk in zip(store_scores, c["store"]) if chunk in rel_recent)
        relevant_store_scores.append(best_rel)
        if c["archive"]:
            arch_scores = score_fn(c["query"], c["archive"])
            archive_top_scores.append(max(arch_scores))
            if max(arch_scores) > best_rel:
                crowded += 1
    q3 = {
        "n_crowding_turns": len(relevant_store_scores),
        "crowded_fraction": round(crowded / len(relevant_store_scores), 4) if relevant_store_scores else None,
        "relevant_store_p50": round(statistics.median(relevant_store_scores), 4) if relevant_store_scores else None,
        "archive_top_p50": round(statistics.median(archive_top_scores), 4) if archive_top_scores else None,
    }
    print(f"Q3 crowding  : crowded={q3['crowded_fraction']}  "
          f"rel-store p50={q3['relevant_store_p50']}  archive-top p50={q3['archive_top_p50']}")

    if args.json:
        out = {
            "drone": "sentence-transformers/paraphrase-MiniLM-L3-v2",
            "return_age": args.return_age,
            "max_score": args.max_score,
            "q1_retrieval": q1,
            "q2_gate": q2,
            "q3_crowding": q3,
        }
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"report       : {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())