"""P5 — Targeted-masking MLM experiment.

Falsifiable prediction: a drone fine-tuned with domain-targeted masked-language
modeling (masking only domain vocabulary) achieves higher relevance-scoring
accuracy on in-domain text than the same model fine-tuned with uniform random
masking, at equal training compute.

Two variants are trained for the SAME number of steps / batch / lr:
  - random:    mask tokens uniformly (15%)
  - targeted:  mask only tokens belonging to the domain vocabulary (code/general)

Downstream evaluation: retrieval precision/recall over held-out query-chunk pairs
(conversations not seen in training).

Usage::

    python -m experiments.p5_targeted_masking --quick      # smoke (few steps)
    python -m experiments.p5_targeted_masking --steps 300  # full run (CPU, bert-tiny)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from transformers import BertForMaskedLM, BertTokenizer

from cortex.baselines.runner import load_conversations
from auditor.labeling import generate_query_chunk_pairs
from sieve.vocabulary import Vocabulary

DEFAULT_MODEL = "prajjwal1/bert-tiny"
SEQ_LEN = 128
MLM_PROBABILITY = 0.15
RUNS_DIR = Path(__file__).resolve().parents[1] / "models" / "p5"


# ---------------------------------------------------------------------------
# Corpus -> fixed-length token sequences
# ---------------------------------------------------------------------------
def build_corpus_sequences(conversations, tokenizer, max_seqs: int | None = None):
    seqs = []
    for conv in conversations:
        for turn in conv.get("turns", []):
            text = turn["content"]
            if not text.strip():
                continue
            ids = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=SEQ_LEN)
            if len(ids) < SEQ_LEN:
                ids = ids + [tokenizer.pad_token_id] * (SEQ_LEN - len(ids))
            seqs.append(ids[:SEQ_LEN])
            if max_seqs and len(seqs) >= max_seqs:
                return seqs
    return seqs


# ---------------------------------------------------------------------------
# Data collators (80/10/10 masking)
# ---------------------------------------------------------------------------
class RandomDataCollator:
    def __init__(self, tokenizer, mlm_probability=MLM_PROBABILITY):
        self.tokenizer = tokenizer
        self.p = mlm_probability

    def __call__(self, batch):
        input_ids = torch.tensor(batch, dtype=torch.long)
        labels = input_ids.clone()
        masked = torch.bernoulli(torch.full(input_ids.shape, self.p)).bool()
        labels[~masked] = -100

        replaced = masked & (torch.rand(masked.shape) < 0.8)
        input_ids[replaced] = self.tokenizer.mask_token_id
        random_idx = masked & ~replaced & (torch.rand(masked.shape) < 0.5)
        random_words = torch.randint(len(self.tokenizer), masked.shape, dtype=torch.long)
        input_ids[random_idx] = random_words[random_idx]
        return {"input_ids": input_ids, "labels": labels}


class TargetedDataCollator:
    def __init__(self, tokenizer, vocab_terms, mlm_probability=MLM_PROBABILITY, seed=0):
        self.tokenizer = tokenizer
        self.vocab_terms = {str(t).lower() for t in vocab_terms}
        self.p = mlm_probability
        self.rng = random.Random(seed)

    def _is_domain(self, token_str: str) -> bool:
        token_str = token_str.lstrip("##").strip().lower()
        if not token_str:
            return False
        if token_str in self.vocab_terms:
            return True
        return any(token_str in term or term in token_str for term in self.vocab_terms)

    def __call__(self, batch):
        input_ids = torch.tensor(batch, dtype=torch.long)
        labels = input_ids.clone()
        special = {
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.pad_token_id,
        }
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for b in range(input_ids.size(0)):
            for t in range(input_ids.size(1)):
                tok = int(input_ids[b, t])
                if tok in special:
                    continue
                if self._is_domain(self.tokenizer.decode([tok])) and self.rng.random() < self.p:
                    mask[b, t] = True
        labels[~mask] = -100

        replaced = mask & (torch.rand(mask.shape) < 0.8)
        input_ids[replaced] = self.tokenizer.mask_token_id
        random_idx = mask & ~replaced & (torch.rand(mask.shape) < 0.5)
        random_words = torch.randint(len(self.tokenizer), mask.shape, dtype=torch.long)
        input_ids[random_idx] = random_words[random_idx]
        return {"input_ids": input_ids, "labels": labels}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_mlm(tokenizer, corpus_seqs, collator, steps, lr, batch_size, seed=0):
    model = BertForMaskedLM.from_pretrained(DEFAULT_MODEL)
    model.train()
    opt = AdamW(model.parameters(), lr=lr)
    rng = random.Random(seed)
    losses = []
    step = 0
    while step < steps:
        batch = rng.sample(corpus_seqs, min(batch_size, len(corpus_seqs)))
        inputs = collator(batch)
        outputs = model(**inputs)
        loss = outputs.loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        step += 1
    return model, losses


# ---------------------------------------------------------------------------
# Downstream retrieval evaluation
# ---------------------------------------------------------------------------
def _cosine(a, b):
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denom) if denom else 0.0


def embed(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=SEQ_LEN)
    model.eval()
    with torch.no_grad():
        out = model.bert(**inputs).last_hidden_state[:, 0, :]
    return out.numpy()[0]


def evaluate_retrieval(model, tokenizer, pairs, threshold=0.5):
    tp = fp = fn = 0
    for p in pairs:
        sim = _cosine(embed(model, tokenizer, p["query"]), embed(model, tokenizer, p["chunk"]))
        pred = sim > threshold
        rel = bool(p["relevant"])
        if pred and rel:
            tp += 1
        elif pred and not rel:
            fp += 1
        elif not pred and rel:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return precision, recall, {"tp": tp, "fp": fp, "fn": fn}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run_experiment(
    conversations_dir: str,
    steps: int,
    lr: float,
    batch_size: int,
    eval_size: int,
    seed: int,
    max_seqs: int | None,
) -> dict:
    conversations = load_conversations(conversations_dir)
    rng = random.Random(seed)
    rng.shuffle(conversations)
    split = max(1, int(len(conversations) * 0.8))
    train_conv, eval_conv = conversations[:split], conversations[split:]

    tokenizer = BertTokenizer.from_pretrained(DEFAULT_MODEL)
    corpus = build_corpus_sequences(train_conv, tokenizer, max_seqs=max_seqs)
    eval_pairs = generate_query_chunk_pairs(eval_conv, n=eval_size, seed=seed)

    vocab = Vocabulary.load("code", "general")

    results = {}
    for variant, collator_factory in (
        ("random", lambda: RandomDataCollator(tokenizer)),
        ("targeted", lambda: TargetedDataCollator(tokenizer, vocab.terms, seed=seed)),
    ):
        model, losses = train_mlm(
            tokenizer, corpus, collator_factory(), steps, lr, batch_size, seed=seed
        )
        precision, recall, counts = evaluate_retrieval(model, tokenizer, eval_pairs)
        results[variant] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "counts": counts,
            "final_loss": round(losses[-1], 4) if losses else None,
        }
        model.save_pretrained(RUNS_DIR / variant)

    return {
        "model": DEFAULT_MODEL,
        "steps": steps,
        "lr": lr,
        "batch_size": batch_size,
        "eval_pairs": len(eval_pairs),
        "corpus_seqs": len(corpus),
        "results": results,
        "targeted_beats_random": results["targeted"]["precision"] > results["random"]["precision"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P5 targeted-masking experiment")
    parser.add_argument("--conversations", default="hivebench/tests/fixtures/generated")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-seqs", type=int, default=None)
    parser.add_argument("--quick", action="store_true", help="tiny smoke run")
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)

    if args.quick:
        args.steps = min(args.steps, 2)
        args.eval_size = min(args.eval_size, 10)
        args.max_seqs = args.max_seqs or 40

    report = run_experiment(
        args.conversations, args.steps, args.lr, args.batch_size,
        args.eval_size, args.seed, args.max_seqs,
    )
    output = Path(args.output) if args.output else RUNS_DIR / "report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("P5 targeted-masking experiment")
    for variant, r in report["results"].items():
        print(f"  {variant:9s} precision={r['precision']} recall={r['recall']} final_loss={r['final_loss']}")
    print(f"  targeted_beats_random: {report['targeted_beats_random']}")
    print(f"Wrote {output.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
