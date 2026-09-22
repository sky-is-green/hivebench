"""P7 human-labeling tool (whitepaper Auditor-Agreement Hypothesis).

P7's measurement: LLM-as-auditor relevance labels (framed as context-utilization
questions) must agree with human labels on >=90% of a sample, and human-human
agreement must be >= auditor-human agreement. This tool provides the human side
(and the auditor side) of that protocol.

Workflow (all subcommands against the same item set):

    1. sample   — build the item set (query, chunk, source, gold) from live-run
                  pairs + fixture pairs, balanced and seeded (deterministic).
    2. rate     — the human labels: a Tkinter app, keyboard-driven
                  (1=relevant 2=not 3=uncertain), auto-saves every answer to
                  NDJSON (crash-safe, resumable). A --cli fallback exists.
    3. auditor   — label the same items with the LLM auditor (LM Studio backend)
                  using the utilization framing.
    4. agree    — compute auditor-human and human-human agreement (P7 verdict).

Item semantics: given the user query, is this chunk of the conversation
relevant to answering it? (The auditor version asks "was this context used /
would it help" — the utilization framing that avoids the parametric-knowledge
confound.)

Usage::

    python -m experiments.human_label sample --n 500 --out models/p7/items.json
    python -m experiments.human_label rate  --items models/p7/items.json --out models/p7/human_raterA.ndjson
    python -m experiments.human_label auditor --items models/p7/items.json --out models/p7/auditor.ndjson
    python -m experiments.human_label agree  --items models/p7/items.json --human models/p7/human_raterA.ndjson --auditor models/p7/auditor.ndjson
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

from auditor.labeling import generate_query_chunk_pairs

HIVEBENCH_ROOT = Path(__file__).resolve().parents[1]
LIVE_RUNS = ["runs/20260822_211131", "runs/20260822_live2", "runs/20260822_live3"]
DEFAULT_N = 500


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def _live_pairs(run_dir: Path, max_pairs: int) -> list[dict]:
    from experiments.encoder_probe import _load_live_pairs

    return _load_live_pairs(run_dir, max_pairs)


def _fixture_pairs(n: int, seed: int) -> list[dict]:
    import glob

    files = sorted(glob.glob(str(HIVEBENCH_ROOT / "tests" / "fixtures" / "generated" / "*.json")))
    convs = [json.loads(Path(f).read_text(encoding="utf-8")) for f in files]
    return generate_query_chunk_pairs(convs, n=n, seed=seed)


def _subchunk(text: str, max_len: int = 600) -> list[str]:
    """Split a reply into focused units: sentence-split first, then merge short
    sentences up to ~max_len chars (paragraph-ish units).

    This is the granularity experiment: instead of whole replies as chunks
    (where one relevant sentence costs the whole chunk's budget), the store
    keeps only the relevant unit. Returns list of non-empty units.
    """
    import re

    parts = re.split(r"(?<=[.!?])\s+|\n+", text or "")
    units: list[str] = []
    buf = ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if buf and len(buf) + len(p) > max_len:
            units.append(buf)
            buf = ""
        buf = (buf + " " + p).strip() if buf else p
    if buf:
        units.append(buf)
    return units


def build_items(n: int, seed: int, live_share: float = 0.6,
                subchunk: bool = False) -> list[dict]:
    """Deterministic, balanced item set: (query, chunk, source, relevant_gold).

    ``relevant_gold`` is the deterministic label (fact-term/topic overlap) —
    shown to the human AFTER they answer, as calibration feedback, and used by
    ``agree`` only for the human-vs-deterministic diagnostic (P7's verdict is
    auditor-vs-human, which is blind to gold).

    With ``subchunk=True`` the chunks are sentence/paragraph units of the
    stored replies instead of whole replies — the granularity experiment
    (finer units are more separable, and the store keeps only relevant units).
    """
    rng = random.Random(seed)
    items: list[dict] = []
    n_live = int(n * live_share)
    answer_facts = _global_answer_facts() if subchunk else {}
    for run in LIVE_RUNS:
        if len(items) >= n_live:
            break
        rd = Path(run)
        if not (rd / "run_report.json").exists():
            continue
        pairs = _live_pairs(rd, 10_000)
        for p in pairs:
            chunks = _subchunk(p["chunk"]) if subchunk else [p["chunk"]]
            for c in chunks:
                gold = bool(p["relevant"]) if not subchunk else _subchunk_gold(
                    p["query"], c, answer_facts)
                items.append({
                    "query": p["query"], "chunk": c,
                    "source": f"live:{run}", "relevant_gold": gold,
                })
    # top up with fixture pairs (easy cross-domain contrast keeps the set honest)
    if len(items) < n:
        for p in _fixture_pairs(n * 2, seed):
            chunks = _subchunk(p["chunk"]) if subchunk else [p["chunk"]]
            for c in chunks:
                gold = bool(p["relevant"]) if not subchunk else _subchunk_gold(
                    p["query"], c, answer_facts)
                items.append({
                    "query": p["query"], "chunk": c, "source": "fixture",
                    "relevant_gold": gold,
                })
            if len(items) >= n:
                break
    items = items[:n]
    rng.shuffle(items)
    for i, it in enumerate(items):
        it["item_id"] = f"item_{i:04d}"
    return items


def _global_answer_facts() -> dict[str, set]:
    """query -> set of fixture answer fact terms (deduped across conversations).

    The deterministic relevance signal for a query is the fixture
    ground-truth answer's distinctive terms (``_answer_fact_terms``), NOT the
    reply's own content terms — the reply may state different facts. Queries
    repeat across fixture conversations with the same canonical answer, so a
    global map is stable.
    """
    import glob

    from experiments.retrieval_diagnostic import (_answer_fact_terms,
                                                  _fixture_answer_map)

    facts: dict[str, set] = {}
    files = sorted(glob.glob(str(HIVEBENCH_ROOT / "tests" / "fixtures" / "generated" / "*.json")))
    convs = [json.loads(Path(f).read_text(encoding="utf-8")) for f in files]
    for cid, per_conv in _fixture_answer_map(convs).items():
        for q, ans in per_conv.items():
            terms = _answer_fact_terms(q, ans)
            if terms:
                facts.setdefault(q, set()).update(terms)
    return facts


def _subchunk_gold(query: str, unit: str, answer_facts: dict[str, set]) -> bool:
    """Gold for a sub-chunked unit: relevant iff the unit contains any fact
    term of the fixture's ground-truth answer for this query (the same
    deterministic signal used at whole-chunk level, applied at unit
    granularity)."""
    from experiments.retrieval_diagnostic import _content_terms

    facts = answer_facts.get(query)
    if not facts:
        return False
    return bool(facts & _content_terms(unit))


# ---------------------------------------------------------------------------
# Human rating: Tkinter app (keyboard-driven) + CLI fallback
# ---------------------------------------------------------------------------
class HumanRater:
    """Core rating logic shared by GUI and CLI: load, answer, autosave, resume."""

    def __init__(self, items: list[dict], out: Path) -> None:
        self.items = items
        self.out = out
        self.answers: dict[str, int] = {}
        self._load_existing()

    def _load_existing(self) -> None:
        if self.out.exists():
            for line in self.out.read_text(encoding="utf-8").strip().splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                self.answers[rec["item_id"]] = rec["label"]

    def answered_count(self) -> int:
        return len(self.answers)

    def answer(self, item: dict, label: int) -> None:
        self.answers[item["item_id"]] = label
        rec = {"item_id": item["item_id"], "label": label,
               "query": item["query"], "chunk": item["chunk"],
               "source": item["source"], "relevant_gold": item["relevant_gold"]}
        with self.out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        self.out.parent.mkdir(parents=True, exist_ok=True)

    def next_unanswered(self, start: int = 0) -> tuple[int, dict | None]:
        for i in range(start, len(self.items)):
            if self.items[i]["item_id"] not in self.answers:
                return i, self.items[i]
        return -1, None


def _label_text(label: int) -> str:
    return {1: "relevant", 2: "not relevant", 3: "uncertain"}.get(label, "?")


DEGENERATE_QUERY_RE = None  # built lazily (avoid import cost at module load)


def is_degenerate_query(query: str) -> bool:
    """A fixture-artifact query: self-referential 'X fit with X' templates
    (e.g. 'How does log levels fit with log levels?'). These are not real
    questions — relevance to them is interpretation-dependent, so they are
    excluded from the P7 agreement and reported separately (2026-08-23)."""
    import re

    global DEGENERATE_QUERY_RE
    if DEGENERATE_QUERY_RE is None:
        DEGENERATE_QUERY_RE = re.compile(
            r"how does (.+?) fit with \1\??$", re.IGNORECASE
        )
    return bool(DEGENERATE_QUERY_RE.search(query))


def _build_groups(items: list[dict]) -> list[dict]:
    """Group consecutive items by query: [{query, indices: [...]}, ...].

    Presentation only — item ids and answers are unchanged, so the P7
    protocol (per-item absolute labels) is untouched. Grouping lets the
    rater judge N chunks against one query without re-reading it, and makes
    same-domain discrimination (the hard cases) comparable within a query.
    """
    groups: list[dict] = []
    order: list[str] = []
    by_query: dict[str, list[int]] = {}
    for i, it in enumerate(items):
        if it["query"] not in by_query:
            by_query[it["query"]] = []
            order.append(it["query"])
        by_query[it["query"]].append(i)
    for q in order:
        groups.append({"query": q, "indices": by_query[q]})
    return groups


def group_of(groups: list[dict], idx: int) -> tuple[int, int, int]:
    """(group_index, position_within_group, group_size) for flat item idx."""
    for gi, g in enumerate(groups):
        if idx in g["indices"]:
            return gi, g["indices"].index(idx), len(g["indices"])
    return -1, -1, -1


RUBRIC = (
    "Judge the chunk as a retrieval unit: if the splinter put this chunk into the "
    "context for this query, would it have helped answer?\n"
    "  1 = RELEVANT — the chunk contains anything that bears on answering this "
    "query (a fact, decision, code, partial answer). Partial overlap counts.\n"
    "  2 = NOT RELEVANT — nothing in the chunk helps answer this query "
    "(different topic, boilerplate, unrelated decisions).\n"
    "  3 = UNCERTAIN — genuinely ambiguous.\n"
)


class RatingApp:
    """Tkinter keyboard-driven rater: 1/2/3 = relevant/not/uncertain.

    Keys: 1,2,3 answer and advance; Left/Right navigate; u undo last; s save;
    q quit. Progress + live counts in the header; deterministic gold is shown
    only after answering (feedback, keeps the human judgment blind upfront).
    """

    def __init__(self, rater: HumanRater, items: list[dict]) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.rater = rater
        self.items = items
        self.root = tk.Tk()
        self.root.title("P7 relevance labeling")
        self.root.geometry("980x620")
        self.idx = 0
        self.last_undo: list[str] = []
        self.groups = _build_groups(items)

        self.header = ttk.Label(self.root, font=("Segoe UI", 10), anchor="w")
        self.header.pack(fill="x", padx=10, pady=6)

        self.progress = ttk.Progressbar(self.root, maximum=max(len(items), 1))
        self.progress.pack(fill="x", padx=10)

        self.rubric = ttk.Label(self.root, font=("Segoe UI", 9), foreground="#444",
                                justify="left", wraplength=940)
        self.rubric.config(text=RUBRIC.strip())
        self.rubric.pack(fill="x", padx=10, pady=(10, 0))

        self.q_label = ttk.Label(self.root, font=("Segoe UI", 12, "bold"),
                                 wraplength=940, justify="left")
        self.q_label.pack(fill="x", padx=10, pady=(12, 4))

        self.caption = ttk.Label(self.root, font=("Segoe UI", 9, "bold"),
                                 foreground="#2255aa", anchor="w")
        self.caption.pack(fill="x", padx=10, pady=(4, 0))

        self.c_text = tk.Text(self.root, wrap="word", font=("Consolas", 10),
                              height=13, relief="solid", borderwidth=1)
        self.c_text.pack(fill="both", expand=True, padx=10, pady=6)
        self.c_text.configure(state="disabled")

        self.gold_label = ttk.Label(self.root, font=("Segoe UI", 9), foreground="#666")
        self.gold_label.pack(fill="x", padx=10)
        self.hint = ttk.Label(self.root, font=("Segoe UI", 9), foreground="#888",
                              text="1 = relevant   2 = not relevant   3 = uncertain   "
                                   "<- -> navigate   u = undo   q = quit")
        self.hint.pack(fill="x", padx=10, pady=(2, 6))

        self.root.bind("1", lambda e: self._answer(1))
        self.root.bind("2", lambda e: self._answer(2))
        self.root.bind("3", lambda e: self._answer(3))
        self.root.bind("<Left>", lambda e: self._goto(self.idx - 1))
        self.root.bind("<Right>", lambda e: self._goto(self.idx + 1))
        self.root.bind("u", lambda e: self._undo())
        self.root.bind("q", lambda e: self.root.destroy())
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)

        self._goto(0)

    def _goto(self, idx: int) -> None:
        self.idx = max(0, min(idx, len(self.items) - 1))
        item = self.items[self.idx]
        answered = self.rater.answers.get(item["item_id"])
        self.progress["value"] = self.rater.answered_count()
        gi, pos, gsize = group_of(self.groups, self.idx)
        self.header.config(
            text=f"item {self.idx + 1}/{len(self.items)}   "
                 f"answered {self.rater.answered_count()}   "
                 f"source: {item['source']}" + (f"   [answered: {_label_text(answered)}]" if answered else ""))
        self.q_label.config(text=f"QUERY ({gi + 1}/{len(self.groups)}): {item['query']}")
        self.caption.config(
            text=f"CANDIDATE CHUNK — stored history (chunk {pos + 1} of {gsize} "
                 f"for this query):")
        self.c_text.configure(state="normal")
        self.c_text.delete("1.0", "end")
        self.c_text.insert("1.0", item["chunk"])
        self.c_text.configure(state="disabled")
        if answered:
            self.gold_label.config(
                text=f"answered {_label_text(answered)}  |  deterministic gold: "
                     f"{'relevant' if item['relevant_gold'] else 'not relevant'}")
        else:
            self.gold_label.config(text="")
        self.root.title(f"P7 relevance labeling  ({self.rater.answered_count()}/{len(self.items)})")

    def _answer(self, label: int) -> None:
        item = self.items[self.idx]
        self.rater.answer(item, label)
        self.last_undo.append(item["item_id"])
        self._goto(min(self.idx + 1, len(self.items) - 1))

    def _undo(self) -> None:
        if not self.last_undo:
            return
        item_id = self.last_undo.pop()
        if item_id in self.rater.answers:
            del self.rater.answers[item_id]
            # rewrite the NDJSON without this record
            lines = [json.loads(l) for l in self.out_lines() if l.strip()]
            kept = [r for r in lines if r["item_id"] != item_id]
            self.rater.out.write_text("\n".join(json.dumps(r) for r in kept) + "\n", encoding="utf-8")
        self._goto(self.idx)

    def out_lines(self) -> list[str]:
        return self.rater.out.read_text(encoding="utf-8").splitlines() if self.rater.out.exists() else []

    def run(self) -> None:
        self.root.mainloop()


def rate_cli(rater: HumanRater, items: list[dict]) -> int:
    """Terminal fallback (non-Tkinter). Same keys as the GUI."""
    idx = 0
    while True:
        _, item = rater.next_unanswered(idx)
        if item is None:
            print(f"\nall {len(items)} items answered. saved to {rater.out}")
            return 0
        idx = next(i for i, it in enumerate(items) if it["item_id"] == item["item_id"])
        print(f"\n[{idx + 1}/{len(items)}] answered {rater.answered_count()}  (source: {item['source']})")
        print(f"QUERY: {item['query']}")
        print("-" * 80)
        print(item["chunk"])
        print("-" * 80)
        key = input("1=relevant 2=not 3=uncertain [n]ext [q]uit: ").strip().lower()
        if key == "q":
            print(f"quit — {rater.answered_count()} answered, saved to {rater.out}")
            return 0
        if key == "1" or key == "2" or key == "3":
            rater.answer(item, int(key))
        idx += 1


# ---------------------------------------------------------------------------
# Auditor side: label the same items with the utilization framing
# ---------------------------------------------------------------------------
QUEEN_PAIR_PROMPT = """You are evaluating whether a context chunk was relevant to answering a user's question.

User question: {query}

Candidate context chunk:
---
{chunk}
---

Would this chunk have helped answer the question, or was it actually used?
Answer ONLY in this JSON shape:
{{"relevant": true, "reason": "..."}}"""


def run_auditor(items: list[dict], out: Path, base_url: str, model: str,
               generate_fn=None) -> int:
    """Label items with the LLM auditor. ``generate_fn`` injectable for tests;
    default uses LM Studio (OpenAI-compatible) via requests."""
    import json as _json
    import requests

    results = {}
    if out.exists():
        for line in out.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                rec = _json.loads(line)
                results[rec["item_id"]] = rec

    def gen(prompt: str) -> str:
        if generate_fn is not None:
            return generate_fn(prompt)
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.0, "max_tokens": 128},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    from auditor.auditor import Auditor

    auditor = Auditor(gen)
    remaining = [it for it in items if it["item_id"] not in results]
    print(f"auditor: {len(results)} already labeled, {len(remaining)} to go")
    for i, item in enumerate(remaining, 1):
        prompt = QUEEN_PAIR_PROMPT.format(query=item["query"], chunk=item["chunk"])
        try:
            parsed = auditor._extract_json(gen(prompt))
            if parsed is None:
                label = 3  # unparseable -> uncertain
            else:
                label = 1 if bool(parsed.get("relevant")) else 2
        except Exception as exc:  # network/parse failure -> uncertain, keep going
            print(f"  item {i}/{len(remaining)} auditor error: {exc}")
            label = 3
        rec = {"item_id": item["item_id"], "label": label, "query": item["query"],
               "chunk": item["chunk"]}
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        results[item["item_id"]] = rec
        if i % 10 == 0:
            print(f"  auditor progress: {i}/{len(remaining)}")
    print(f"auditor done: {len(results)} labels -> {out}")
    return 0


# ---------------------------------------------------------------------------
# Agreement (P7 verdict)
# ---------------------------------------------------------------------------
def _load_labels(path: Path) -> dict[str, int]:
    out = {}
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        if line.strip():
            rec = json.loads(line)
            out[rec["item_id"]] = rec["label"]
    return out


def compute_agreement(items: list[dict], human: Path, auditor: Path,
                      human2: Path | None = None) -> dict:
    h1 = _load_labels(human)
    o = _load_labels(auditor)
    h2 = _load_labels(human2) if human2 is not None else {}
    ids = [it["item_id"] for it in items]
    # Degenerate fixture queries ("X fit with X") are not real questions:
    # relevance to them is interpretation-dependent, so they are excluded from
    # the P7 verdict and reported separately (2026-08-23).
    degenerate_ids = [it["item_id"] for it in items
                      if is_degenerate_query(it.get("query", ""))]
    clean_ids = [i for i in ids if i not in degenerate_ids]

    def agreed(a: dict, b: dict, ids_: list) -> tuple[int, int]:
        n = same = 0
        for i in ids_:
            if i in a and i in b:
                n += 1
                if a[i] == b[i]:
                    same += 1
        return same, n

    def agreement(a: dict, b: dict, ids_: list) -> float | None:
        same, n = agreed(a, b, ids_)
        return same / n if n else None

    oh_same, oh_n = agreed(h1, o, clean_ids)
    oa = agreement(h1, o, clean_ids)
    hh = agreement(h1, h2, clean_ids) if h2 else None
    oh_discord = [i for i in clean_ids if i in h1 and i in o and h1[i] != o[i]]

    verdict = "FAIL"
    reasons = []
    if oa is None or oa < 0.90:
        reasons.append(f"auditor-human agreement {oa if oa is not None else 'n/a'} < 90%")
    if hh is not None and oa is not None and hh < oa:
        reasons.append(f"human-human {hh:.1%} < auditor-human {oa:.1%}")
    if not reasons:
        verdict = "PASS"

    return {
        "auditor_human_agreement": round(oa, 4) if oa is not None else None,
        "auditor_human_n": oh_n,
        "human_human_agreement": round(hh, 4) if hh is not None else None,
        "verdict": verdict,
        "reasons": reasons,
        "discordant_item_ids": oh_discord[:20],
        "n_discordant": len(oh_discord),
        "excluded_degenerate": {
            "n": len(degenerate_ids),
            "item_ids": degenerate_ids,
            "note": "fixture 'X fit with X' artifacts excluded from the "
                    "agreement (relevance is interpretation-dependent)",
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("sample", help="build the item set")
    p.add_argument("--n", type=int, default=DEFAULT_N)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--subchunk", action="store_true",
                   help="split replies into sentence/paragraph units (granularity experiment)")
    p.add_argument("--out", default="models/p7/items.json")

    p = sub.add_parser("rate", help="human labeling (GUI; --cli for terminal)")
    p.add_argument("--items", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cli", action="store_true", help="terminal fallback")

    p = sub.add_parser("auditor", help="label items with the LLM auditor")
    p.add_argument("--items", required=True)
    p.add_argument("--out", default="models/p7/auditor.ndjson")
    p.add_argument("--base-url", default="http://localhost:1234")
    p.add_argument("--model", default="prism-ml/bonsai-27b")

    p = sub.add_parser("agree", help="P7 agreement verdict")
    p.add_argument("--items", required=True)
    p.add_argument("--human", required=True)
    p.add_argument("--auditor", required=True)
    p.add_argument("--human2", default=None)

    args = ap.parse_args(argv)

    if args.cmd == "sample":
        items = build_items(args.n, args.seed, subchunk=args.subchunk)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(items, indent=2), encoding="utf-8")
        n_rel = sum(1 for it in items if it["relevant_gold"])
        print(f"{len(items)} items -> {out}  (gold relevant: {n_rel}, "
              f"not: {len(items) - n_rel})")
        return 0

    if args.cmd == "rate":
        items = json.loads(Path(args.items).read_text(encoding="utf-8"))
        rater = HumanRater(items, Path(args.out))
        print(f"resuming: {rater.answered_count()}/{len(items)} already answered")
        if args.cli:
            return rate_cli(rater, items)
        try:
            app = RatingApp(rater, items)
            app.run()
        except Exception as exc:
            print(f"Tkinter unavailable ({exc}); falling back to --cli")
            return rate_cli(rater, items)
        print(f"saved to {args.out}")
        return 0

    if args.cmd == "auditor":
        items = json.loads(Path(args.items).read_text(encoding="utf-8"))
        return run_auditor(items, Path(args.out), args.base_url, args.model)

    if args.cmd == "agree":
        items = json.loads(Path(args.items).read_text(encoding="utf-8"))
        res = compute_agreement(items, Path(args.human), Path(args.auditor),
                                Path(args.human2) if args.human2 else None)
        print(json.dumps(res, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())