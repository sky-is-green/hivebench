#!/usr/bin/env python
"""Hidden-state steerer probe — is final correctness linearly visible in Scion's
own post-norm hidden states before the answer exists?

The steerer role (separate from the epilogue judge) would read the generator's
trajectory and decide continue / stop / force-answer.  Text decision models
zero-shot cannot do it (JEV-9B anti-correlated on prefixes; 0.8B/2B worse).
This probes the strongest alternative: Scion's per-position hidden state, as
dumped by ``test-mtp-probe`` (the same tensor the fork's MTP sidecar already
consumes at ~1 ms/position).

For each token cut t, features are the hidden state after t tokens (raw, and
mean-pooled over the last window) and the label is the checker's verdict on the
final answer.  Leave-one-task-out logistic regression and a centroid margin
give an honest AUROC at each horizon, plus the cancel trade-off (catch wrong
answers early at the price of false cancels).

    ~/Desktop/work/.venv-rocm/bin/python experiments/cascade/hidden_probe.py \
        --dir experiments/cascade/streams/scion-v2-streams/hidden
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
CUTS = (16, 32, 64, 128, 256)


def rank_auc(scores: np.ndarray, labels: np.ndarray) -> Optional[float]:
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 4)


def fit_lr(X: np.ndarray, y: np.ndarray, *, l2: float = 1e-2, iters: int = 60):
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y.astype(np.float32))
    mu = Xt.mean(0, keepdim=True)
    sd = Xt.std(0, keepdim=True) + 1e-6
    Xs = (Xt - mu) / sd
    w = torch.zeros(Xt.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logits = Xs @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yt)
        loss = loss + l2 * (w ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach(), b.detach(), mu, sd


def score_lr(params, X: np.ndarray) -> np.ndarray:
    w, b, mu, sd = params
    Xt = torch.tensor(X, dtype=torch.float32)
    return ((Xt - mu) / sd @ w + b).detach().numpy()


def centroid_params(X: np.ndarray, y: np.ndarray):
    mu_pos = X[y].mean(0)
    mu_neg = X[~y].mean(0)
    direction = mu_pos - mu_neg
    w = direction / (np.linalg.norm(direction) + 1e-9)
    b = float(-((mu_pos + mu_neg) / 2.0) @ w)
    return w, b


def loo_scores(X: np.ndarray, y: np.ndarray, kind: str) -> np.ndarray:
    scores = np.zeros(len(y))
    for i in range(len(y)):
        train = np.arange(len(y)) != i
        if kind == "lr":
            scores[i] = score_lr(fit_lr(X[train], y[train]), X[i : i + 1])[0]
        else:
            w, b = centroid_params(X[train], y[train])
            scores[i] = float(X[i] @ w + b)
    return scores


def cancel_table(scores: np.ndarray, y: np.ndarray) -> list[dict]:
    """Walk from most-doomed to least: catches vs false cancels."""
    order = np.argsort(scores)  # lowest = most doomed
    caught = false_cancels = 0
    rows = []
    for rank, idx in enumerate(order, 1):
        if y[idx]:
            false_cancels += 1
        else:
            caught += 1
        if caught and (rank == len(order) or caught == np.sum(~y) or false_cancels <= 3):
            rows.append(
                {
                    "rank": rank,
                    "caught": int(caught),
                    "false_cancels": int(false_cancels),
                    "task": None,
                }
            )
        if caught == int(np.sum(~y)):
            break
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        default=str(REPO / "experiments/cascade/streams/scion-v2-streams/hidden"),
    )
    parser.add_argument("--cuts", default=",".join(str(c) for c in CUTS))
    parser.add_argument("--pool", type=int, default=8, help="pool window before t")
    parser.add_argument("--out", default=str(REPO / "experiments/cascade/hidden-probe.json"))
    args = parser.parse_args()

    root = Path(args.dir)
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    lengths = np.array(meta["lengths"])
    y = np.array(meta["labels"], dtype=bool)
    order = meta["order"]
    seq = meta["seq"]

    dump = root / "dump"
    files = sorted(dump.glob("win_*_h.bin"))
    if len(files) != len(order):
        raise SystemExit(f"expected {len(order)} hidden windows, found {len(files)}")
    n_embd = files[0].stat().st_size // (seq * 4)
    H = np.empty((len(order), seq, n_embd), dtype=np.float32)
    for i, path in enumerate(files):
        H[i] = np.fromfile(path, dtype=np.float32).reshape(seq, n_embd)

    cuts = [int(c) for c in args.cuts.split(",")]
    summary: dict = {
        "tasks": len(order),
        "wrong": int((~y).sum()),
        "n_embd": int(n_embd),
        "pool": args.pool,
        "cuts": {},
    }
    print(f"{'cut':>4} {'n':>3} {'AUC_lr':>7} {'AUC_ctr':>7}  wrong-task LOO LR ranks (of n)")
    for t in cuts:
        valid = np.where(lengths >= t)[0]
        if len(valid) < 4 or y[valid].sum() in (0, len(valid)):
            continue
        raw = np.stack([H[i, t - 1] for i in valid])
        start = max(0, t - args.pool)
        pooled = np.stack([H[i, start:t].mean(0) for i in valid])
        yv = y[valid]
        funcs = {}
        for name, X in (("raw", raw), ("pooled", pooled)):
            lr = loo_scores(X, yv, "lr")
            ctr = loo_scores(X, yv, "centroid")
            funcs[name] = {"lr": lr, "centroid": ctr}
        best = funcs["pooled"]
        auc_lr = rank_auc(best["lr"], yv)
        auc_ctr = rank_auc(best["centroid"], yv)
        ranks = {
            order[valid[i]]: int((best["lr"] < best["lr"][i]).sum() + 1)
            for i in range(len(valid))
            if not yv[i]
        }
        summary["cuts"][str(t)] = {
            "n": int(len(valid)),
            "auc_pooled_lr": auc_lr,
            "auc_pooled_centroid": auc_ctr,
            "auc_raw_lr": rank_auc(funcs["raw"]["lr"], yv),
            "auc_raw_centroid": rank_auc(funcs["raw"]["centroid"], yv),
            "wrong_ranks_pooled_lr": ranks,
            "cancel_pooled_lr": cancel_table(best["lr"], yv),
        }
        print(
            f"{t:>4} {len(valid):>3} {auc_lr if auc_lr is not None else float('nan'):>7} "
            f"{auc_ctr if auc_ctr is not None else float('nan'):>7}  {ranks}"
        )

    Path(args.out).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("\nartifact:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
