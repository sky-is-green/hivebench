"""Encoder probe: reproducible retrieval-precision measurement for the B avenue.

Measures *any* encoder's retrieval quality on labeled query-chunk pairs,
producing the same metrics the precision investigation used:
  - top-K precision/recall curve (K in 1/3/5/8/10), averaged per query
  - threshold sweep: best precision achievable at recall >= 0.90
    (the hand-off claim: "no all-MiniLM threshold reaches >=85% precision
    at >=90% recall")
  - score distribution of relevant vs irrelevant chunks

Encoders (--encoder):
  ultra                      all-MiniLM-L6-v2 via UltraSmallDrone (vocab boost OFF,
                             so the probe measures the encoder, not the pipeline)
  l3                         paraphrase-MiniLM-L3-v2 (3-layer sibling, smallest MiniLM)
  medium                     graphcodebert-base bi-encoder (MediumDrone mode=bi)
  bge-m3                     BAAI/bge-m3 (large pretrained retrieval encoder, B1)
  bge-m3-mrl:DIM             BAAI/bge-m3 with Matryoshka truncation to DIM dims
                             (e.g. 256) — the ST org's variable-size-embedding
                             efficiency technique; bge-m3 was MRL-trained
  checkpoint:PATH            a trained SentenceTransformer checkpoint (B2/B3)

Pair sources (--pairs):
  fixture                    tests/fixtures/generated + auditor.labeling (standard test set)
  live:RUN_DIR               reconstruct conversations from a run_report.json and label
                             with the same topic machinery (temporal-split source, B3)
  json:FILE                  a saved pairs file

Whitelist compliance: pairs are never part of any training set for the
encoder being measured (B2/B3 train on disjoint fresh-seed or earlier-run
pairs; the measured set is held out).

Usage::

    python -m experiments.encoder_probe --encoder ultra --pairs fixture --n 500
    python -m experiments.encoder_probe --encoder bge-m3 --json models/b/b1_bgem3.json
    python -m experiments.encoder_probe --encoder checkpoint:models/b/b2_fresh --pairs live:runs/20260822_211131
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

from auditor.labeling import generate_query_chunk_pairs
from experiments.retrieval_diagnostic import (_answer_fact_terms,
                                              _content_terms,
                                              _fixture_answer_map)

HIVEBENCH_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = HIVEBENCH_ROOT / "tests" / "fixtures" / "generated"
DEFAULT_K = [1, 3, 5, 8, 10]
RECALL_FLOOR = 0.90


# ---------------------------------------------------------------------------
# Pair loading
# ---------------------------------------------------------------------------
def _load_fixture_conversations() -> list[dict]:
    import glob

    files = sorted(glob.glob(str(FIXTURE_DIR / "*.json")))
    if not files:
        raise SystemExit(f"no fixture conversations in {FIXTURE_DIR}")
    return [json.loads(Path(f).read_text(encoding="utf-8")) for f in files]


def _reconstruct_conversations_from_run(run_dir: Path) -> list[dict]:
    """Rebuild conversation dicts (turns with role/content) from a run report."""
    report_path = run_dir / "run_report.json"
    if not report_path.exists():
        raise SystemExit(f"no run_report.json in {run_dir}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    convs = []
    for c in report.get("conversations", []):
        turns = []
        for t in c.get("turns", []):
            q = (t.get("query") or "").strip()
            r = (t.get("reply") or "").strip()
            if q:
                turns.append({"role": "user", "content": q})
            if r:
                turns.append({"role": "assistant", "content": r})
        convs.append({"conversation_id": c.get("conversation_id", "unknown"),
                      "turns": turns})
    if not convs:
        raise SystemExit(f"no conversations in {report_path}")
    return convs


def _load_live_pairs(run_dir: Path, max_pairs: int) -> list[dict]:
    """Query-chunk pairs from a live run, relevance = fixture answer facts.

    Rebuilds the measurement that produced the hand-off's precision ceiling:
    for each user query in the run, the candidate chunks are the *prior stored
    reply chunks* (non-hedge replies, mirroring what the splinter stores), and a
    chunk is relevant iff it contains a fact term of the fixture's
    ground-truth answer for that query (the deterministic P2 diagnostic's own
    notion of relevance — see ``retrieval_diagnostic._answer_fact_terms``).

    Pairs are causal: only chunks stored *before* the query are candidates,
    exactly what the splinter could have retrieved.
    """
    from experiments.retrieval_diagnostic import _answer_fact_terms as _aft

    report = json.loads((run_dir / "run_report.json").read_text(encoding="utf-8"))
    convs = report.get("conversations", [])
    # fixture answers by conversation_id
    fixture_convs = _load_fixture_conversations()
    answers = _fixture_answer_map(fixture_convs)

    from cortex.splinter import Splinter

    pairs: list[dict] = []
    for conv in convs:
        cid = conv.get("conversation_id", "unknown")
        stored: list[str] = []  # prior stored reply chunks
        for t in conv.get("turns", []):
            q = (t.get("query") or "").strip()
            r = (t.get("reply") or "").strip()
            if not q:
                continue
            # answer fact terms for this query (only if the fixture knows it)
            ans = answers.get(cid, {}).get(q, "")
            facts = _aft(q, ans) if ans else set()
            if facts and stored:
                for chunk in stored:
                    if not chunk.strip():
                        continue
                    relevant = bool(facts & _content_terms(chunk))
                    pairs.append({"query": q, "chunk": chunk, "relevant": relevant})
            if r and not Splinter._is_hedge_reply(r):
                stored.append(r)
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


def load_pairs(source: str, n: int, seed: int, subchunk: bool = False) -> list[dict]:
    if source == "fixture":
        convs = _load_fixture_conversations()
        pairs = generate_query_chunk_pairs(convs, n=n, seed=seed)
    elif source.startswith("live:"):
        pairs = _load_live_pairs(Path(source.split(":", 1)[1]), n)
    elif source.startswith("json:"):
        pairs = json.loads(Path(source.split(":", 1)[1]).read_text(encoding="utf-8"))
        pairs = pairs[:n]
    else:
        raise SystemExit(f"unknown --pairs source: {source!r}")
    if not pairs:
        raise SystemExit(f"no pairs generated from {source}")
    if subchunk:
        pairs = _subchunk_pairs(pairs)
    return pairs


def _subchunk_pairs(pairs: list[dict]) -> list[dict]:
    """Granularity experiment (2026-08-23): split whole-reply chunks into
    sentence/paragraph units, re-label each unit by the deterministic answer-
    fact test, and measure retrieval on the *units* — the question being
    whether finer units are more separable than whole replies (the B avenue
    measured whole replies only)."""
    from experiments.human_label import _global_answer_facts, _subchunk

    answer_facts = _global_answer_facts()
    out: list[dict] = []
    for p in pairs:
        for unit in _subchunk(p["chunk"]):
            # unit relevant iff it contains a fact term of the fixture
            # answer for this query (NOT the chunk's own terms — a unit is a
            # subset of its parent, so chunk∩unit is trivially non-empty)
            from experiments.retrieval_diagnostic import _content_terms as _ct

            facts = answer_facts.get(p["query"], set())
            relevant = bool(facts & _ct(unit))
            out.append({"query": p["query"], "chunk": unit,
                        "relevant": relevant, "parent": p["chunk"]})
    return out


# ---------------------------------------------------------------------------
# Scorers (higher = more relevant); all return list[float] aligned to chunks
# ---------------------------------------------------------------------------
def _cosine_scores(q_vec: np.ndarray, chunk_vecs: np.ndarray) -> list[float]:
    dots = chunk_vecs @ q_vec
    qn = float(np.linalg.norm(q_vec))
    cn = np.linalg.norm(chunk_vecs, axis=1)
    denom = qn * cn + 1e-9
    return [float(d) for d in dots / denom]


def make_scorer(encoder: str):
    if encoder == "ultra":
        from sieve.ultra_small import UltraSmallDrone

        drone = UltraSmallDrone(confidence_mode="off", vocab=None, vocab_boost=0.0)

        def score(query: str, chunks: list[str]) -> list[float]:
            drone._ensure_loaded()
            q = drone.embed(query)
            embs = np.stack([drone.embed(c) for c in chunks])
            return _cosine_scores(q, embs)

        return score, "sentence-transformers/all-MiniLM-L6-v2"

    if encoder == "l3":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")
        model.eval()

        def score(query: str, chunks: list[str]) -> list[float]:
            q_vec = model.encode([query], normalize_embeddings=True)[0]
            if not chunks:
                return []
            vecs = model.encode(chunks, normalize_embeddings=True, batch_size=32)
            return _cosine_scores(q_vec, np.asarray(vecs))

        return score, "sentence-transformers/paraphrase-MiniLM-L3-v2"

    if encoder.startswith("bge-m3-mrl:"):
        from sentence_transformers import SentenceTransformer

        dim = int(encoder.split(":", 1)[1])
        model = SentenceTransformer("BAAI/bge-m3")
        model.eval()

        def score(query: str, chunks: list[str]) -> list[float]:
            q_vec = model.encode([query], normalize_embeddings=True)[0][:dim]
            if not chunks:
                return []
            vecs = model.encode(chunks, normalize_embeddings=True, batch_size=32)[:, :dim]
            return _cosine_scores(q_vec, np.asarray(vecs))

        return score, f"BAAI/bge-m3 (MRL dim={dim})"
        from sieve.medium import MediumDrone

        drone = MediumDrone(mode="bi")

        def score(query: str, chunks: list[str]) -> list[float]:
            drone._ensure_loaded()
            q = drone._embed(query)
            return [drone._cosine(q, drone._embed(c)) for c in chunks]

        return score, "microsoft/graphcodebert-base (bi)"

    if encoder == "bge-m3" or encoder.startswith("bge-m3-mrl:"):
        raise SystemExit(f"bge-m3 handlers moved above; encoder={encoder!r}")

    if encoder.startswith("checkpoint:"):
        from sentence_transformers import SentenceTransformer

        path = encoder.split(":", 1)[1]
        model = SentenceTransformer(path)
        model.eval()

        def score(query: str, chunks: list[str]) -> list[float]:
            q_vec = model.encode([query], normalize_embeddings=True)[0]
            if not chunks:
                return []
            vecs = model.encode(chunks, normalize_embeddings=True, batch_size=32)
            return _cosine_scores(q_vec, np.asarray(vecs))

        return score, f"checkpoint:{path}"

    raise SystemExit(f"unknown --encoder: {encoder!r}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _group_by_query(pairs: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for p in pairs:
        groups.setdefault(p["query"], []).append(p)
    return groups


def compute_topk(pairs: list[dict], score_fn, ks: list[int]) -> dict:
    """Per-query ranking -> precision@k / recall@k, averaged over queries."""
    groups = _group_by_query(pairs)
    per_k: dict[int, list[dict]] = {k: [] for k in ks}
    for query, group in groups.items():
        if len(group) < 2:
            continue
        chunks = [g["chunk"] for g in group]
        scores = score_fn(query, chunks)
        ranked = sorted(zip(scores, group), key=lambda x: x[0], reverse=True)
        n_rel = sum(1 for g in group if g["relevant"])
        if n_rel == 0:
            continue
        for k in ks:
            if len(ranked) < k:
                continue
            top = ranked[:k]
            rel_in_top = sum(1 for _, g in top if g["relevant"])
            per_k[k].append({
                "precision": rel_in_top / k,
                "recall": rel_in_top / n_rel,
                "score_top": top[0][0],
            })
    out = {}
    for k, rows in per_k.items():
        if rows:
            out[str(k)] = {
                "precision": round(statistics.mean(r["precision"] for r in rows), 4),
                "recall": round(statistics.mean(r["recall"] for r in rows), 4),
                "queries": len(rows),
            }
    return out


def compute_threshold_curve(pairs: list[dict], score_fn) -> dict:
    """Sweep score thresholds; report best precision at recall >= floor."""
    if not pairs:
        return {}
    rows = []
    for p in pairs:
        s = score_fn(p["query"], [p["chunk"]])[0]
        rows.append((s, bool(p["relevant"])))
    rows.sort(key=lambda x: x[0])
    n_rel = sum(1 for _, rel in rows if rel)
    n_irr = len(rows) - n_rel
    if n_rel == 0 or n_irr == 0:
        return {"note": "unbalanced pairs; threshold curve unavailable"}
    best = None
    curve = []
    # sweep each score as a candidate threshold (predict relevant above it)
    for i in range(len(rows) + 1):
        tp = sum(1 for s, rel in rows[i:] if rel)
        fp = sum(1 for s, rel in rows[i:] if not rel)
        fn = n_rel - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        thresh = rows[i][0] if i < len(rows) else -1.0
        curve.append({"threshold": round(thresh, 4), "precision": round(precision, 4),
                      "recall": round(recall, 4)})
        if recall >= RECALL_FLOOR and (best is None or precision > best["precision"]):
            best = {"threshold": thresh, "precision": precision, "recall": recall}
    return {
        "best_precision_at_recall_ge_90": round(best["precision"], 4) if best else None,
        "best_threshold": round(best["threshold"], 4) if best else None,
        "best_recall": round(best["recall"], 4) if best else None,
        "recall_floor": RECALL_FLOOR,
        "n_pairs": len(rows),
        "n_relevant": n_rel,
        "n_irrelevant": n_irr,
        "curve": curve,
    }


def compute_score_distribution(pairs: list[dict], score_fn) -> dict:
    rel, irr = [], []
    for p in pairs:
        s = score_fn(p["query"], [p["chunk"]])[0]
        (rel if p["relevant"] else irr).append(s)
    out = {}
    if rel:
        out["relevant"] = {"mean": round(statistics.mean(rel), 4),
                           "p50": round(statistics.median(rel), 4),
                           "n": len(rel)}
    if irr:
        out["irrelevant"] = {"mean": round(statistics.mean(irr), 4),
                             "p50": round(statistics.median(irr), 4),
                             "n": len(irr)}
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", default="ultra",
                    help="ultra|medium|bge-m3|checkpoint:PATH")
    ap.add_argument("--pairs", default="fixture",
                    help="fixture|live:RUN_DIR|json:FILE")
    ap.add_argument("--n", type=int, default=528, help="max pairs to load")
    ap.add_argument("--seed", type=int, default=0, help="pair-generation seed")
    ap.add_argument("--subchunk", action="store_true",
                    help="granularity experiment: split chunks into sentence units")
    ap.add_argument("--json", default=None, help="write report to this path")
    ap.add_argument("--topk", default=",".join(map(str, DEFAULT_K)))
    args = ap.parse_args()

    ks = [int(x) for x in args.topk.split(",")]
    pairs = load_pairs(args.pairs, args.n, args.seed, subchunk=args.subchunk)
    score_fn, model_name = make_scorer(args.encoder)

    topk = compute_topk(pairs, score_fn, ks)
    threshold = compute_threshold_curve(pairs, score_fn)
    dist = compute_score_distribution(pairs, score_fn)

    print(f"encoder      : {args.encoder}  ({model_name})")
    print(f"pairs source : {args.pairs}  (n={len(pairs)}, "
          f"relevant={sum(1 for p in pairs if p['relevant'])}"
          f"{', subchunked' if args.subchunk else ''})")
    print(f"top-K curve  :")
    for k in ks:
        row = topk.get(str(k))
        if row:
            print(f"  top-{k:<3} precision={row['precision']:.1%}  "
                  f"recall={row['recall']:.1%}  (queries={row['queries']})")
    print(f"threshold    : best precision at recall>={RECALL_FLOOR:.0%} = "
          f"{threshold.get('best_precision_at_recall_ge_90')} "
          f"(thresh={threshold.get('best_threshold')}, "
          f"recall={threshold.get('best_recall')})")
    print(f"distribution : relevant p50={dist.get('relevant', {}).get('p50')}  "
          f"irrelevant p50={dist.get('irrelevant', {}).get('p50')}")

    if args.json:
        out = {
            "encoder": args.encoder,
            "model_name": model_name,
            "pairs_source": args.pairs,
            "n_pairs": len(pairs),
            "n_relevant": sum(1 for p in pairs if p["relevant"]),
            "seed": args.seed,
            "topk": topk,
            "threshold": threshold,
            "score_distribution": dist,
        }
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"report       : {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())