"""Contrastive fine-tuning of a drone encoder on query-chunk pairs (B2/B3).

The B avenue's "tunable hypothesis": a small encoder trained directly on the
retrieval task (query -> relevant chunk) may beat the untuned stock encoders
that all hit the same ~10.7% precision ceiling. Training data is *disjoint*
from the measured test set, per the whitelist constraint:

  - ``--source fresh:SEED``  fresh-seed synthetic conversations (same generator,
    different seed -> different conversations, untouched test set)
  - ``--source live:RUN_DIR``  query-chunk pairs reconstructed from an earlier
    live run (temporal split, B3)

Positives are (query, chunk) pairs where the topic-based labeler says
"relevant" (``auditor.labeling.generate_query_chunk_pairs``). The model is
trained with MultipleNegativesRankingLoss (in-batch negatives) — the standard
contrastive retrieval objective.

Output: a SentenceTransformer checkpoint directory that
``experiments.encoder_probe --encoder checkpoint:PATH`` can measure on the
held-out live-run pairs.

Usage::

    python -m experiments.contrastive_finetune --source fresh:4242 --out models/b/b2_fresh --epochs 3
    python -m experiments.contrastive_finetune --source live:runs/20260822_live2 --out models/b/b3_live2 --epochs 3
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from auditor.labeling import generate_query_chunk_pairs

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_fresh_conversations(seed: int) -> list[dict]:
    from tests.fixtures.synthetic_conversations.generate import generate

    out_dir = REPO_ROOT / "models" / "b" / f"fresh_seed_{seed}"
    convs = generate(out_dir, seed=seed)
    print(f"generated {len(convs)} fresh conversations (seed {seed})")
    return convs


def _load_live_pairs(run_dir: Path, max_pairs: int) -> list[dict]:
    """(query, chunk, relevant) pairs from a live run's stored replies."""
    from experiments.encoder_probe import _load_live_pairs

    pairs = _load_live_pairs(run_dir, max_pairs)
    print(f"loaded {len(pairs)} pairs from {run_dir}")
    return pairs


def build_training_pairs(source: str, n: int, seed: int) -> list[dict]:
    if source.startswith("fresh:"):
        convs = _load_fresh_conversations(int(source.split(":", 1)[1]))
        pairs = generate_query_chunk_pairs(convs, n=n, seed=seed)
    elif source.startswith("live:"):
        pairs = _load_live_pairs(Path(source.split(":", 1)[1]), n)
    else:
        raise SystemExit(f"unknown --source: {source!r}")
    positives = [p for p in pairs if p["relevant"]]
    if not positives:
        raise SystemExit("no relevant (positive) pairs — cannot contrastively train")
    print(f"training positives: {len(positives)} / {len(pairs)} pairs")
    return positives


def train(source: str, out: str, n: int, seed: int, epochs: int, steps: int) -> str:
    from datasets import Dataset
    from sentence_transformers import (SentenceTransformer,
                                       SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments,
                                       losses)

    positives = build_training_pairs(source, n, seed)
    train_ds = Dataset.from_dict({
        "anchor": [p["query"] for p in positives],
        "positive": [p["chunk"] for p in positives],
    })

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    train_loss = losses.MultipleNegativesRankingLoss(model)

    args = SentenceTransformerTrainingArguments(
        output_dir=out,
        num_train_epochs=epochs,
        max_steps=steps if steps is not None else -1,
        per_device_train_batch_size=16,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        save_strategy="no",
        logging_steps=10,
        report_to=[],
        disable_tqdm=False,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        loss=train_loss,
    )
    trainer.train()

    out_path = Path(out)
    if out_path.exists():
        shutil.rmtree(out_path)
    model.save(out_path)
    meta = {
        "source": source,
        "n_positive_pairs": len(positives),
        "seed": seed,
        "epochs": epochs,
        "max_steps": steps,
        "base_model": "sentence-transformers/all-MiniLM-L6-v2",
        "loss": "MultipleNegativesRankingLoss",
        "test_policy": "measured on held-out live-run pairs; training data "
                       "never overlaps the measured set",
    }
    (out_path / "b_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"checkpoint -> {out_path}")
    return str(out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True,
                    help="fresh:SEED | live:RUN_DIR")
    ap.add_argument("--out", required=True, help="checkpoint output dir")
    ap.add_argument("--n", type=int, default=528, help="max pairs to build")
    ap.add_argument("--seed", type=int, default=0, help="pair-generation seed")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--steps", type=int, default=-1,
                    help="cap training steps (-1 = epochs-driven)")
    args = ap.parse_args()
    steps = args.steps if args.steps > 0 else None
    train(args.source, args.out, args.n, args.seed, args.epochs, steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())