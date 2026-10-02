#!/usr/bin/env python
"""Ordinary-LLM steerer probe — can a general instruct model read a partial
trace better than the specialized decision heads?

JEV-9B and the Intern-Decision family are locked to trained question shapes and
were never trained on partials; a general instruct model can follow an
arbitrary "estimate the probability this ends correct" instruction.  Same
captured streams, same cuts, same metrics as ``prefix_eval.py`` — this swaps
only the sensor for an ordinary chat model and verbalized probabilities.

    ~/Desktop/work/.venv-rocm/bin/python experiments/cascade/steerer_llm_eval.py \
        --model Qwen/Qwen3-1.7B --out experiments/cascade/steerer-qwen17.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.cascade.checkers import check  # noqa: E402
from experiments.cascade.history import load_tasks  # noqa: E402
from experiments.cascade.prefix_eval import auc  # noqa: E402

CUTS = (16, 32, 64, 128, 256)

SYSTEM = (
    "You monitor another AI's work in progress. You see a TASK and the PARTIAL "
    "RESPONSE it has produced so far (it may stop mid-sentence). Estimate the "
    "probability that the AI's final answer will be correct and complete for "
    "the task. Reply with a single integer from 0 to 100, nothing else."
)

SYSTEM_BINARY = (
    "You monitor another AI's work in progress. You see a TASK and the PARTIAL "
    "RESPONSE it has produced so far (it may stop mid-sentence). Decide whether "
    "the AI's final answer will be correct and complete for the task. Reply with "
    "exactly YES or NO, nothing else."
)

QUESTION = {
    "prob": "Probability (0-100):",
    "binary": "Answer YES or NO:",
    "binary_logit": "Answer YES or NO:",
}


def parse_prob(text: str) -> Optional[float]:
    matches = re.findall(r"\d{1,3}", text)
    if not matches:
        return None
    return min(100.0, max(0.0, float(matches[-1]))) / 100.0


def parse_answer(text: str, mode: str) -> Optional[float]:
    # a thinking model: read the post-trace answer only
    for marker in (" response", "<|end_of_thinking|>", "``", "``"):
        if marker in text:
            text = text.rsplit(marker, 1)[-1]
    if mode == "binary":
        match = re.search(r"\b(yes|no)\b", text.lower())
        if not match:
            return None
        return 1.0 if match.group(1) == "yes" else 0.0
    return parse_prob(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--mode", choices=("prob", "binary", "binary_logit"), default="prob")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="pass enable_thinking=False to the chat template",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--capture",
        default=str(REPO / "experiments/cascade/streams/scion-v2-streams/capture.json"),
    )
    parser.add_argument("--cuts", default="".join(f"{c}," for c in CUTS).rstrip(","))
    parser.add_argument("--out", default=str(REPO / "experiments/cascade/steerer-llm.json"))
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tasks = load_tasks()
    capture = json.loads(Path(args.capture).read_text(encoding="utf-8"))
    records = [
        (tid, rec)
        for tid, rec in capture["records"].items()
        if tasks.get(tid) and (rec.get("reasoning") or rec.get("content"))
    ]
    if args.limit:
        records = records[: args.limit]

    print(f"loading {args.model} ...", file=sys.stderr, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=getattr(torch, args.dtype),
        device_map=args.device,
        trust_remote_code=args.trust_remote_code,
    ).eval()

    def ask(task: dict, prefix: str) -> tuple[Optional[float], str]:
        system = SYSTEM_BINARY if args.mode.startswith("binary") else SYSTEM
        user = (
            f"TASK:\n{task['prompt']}\n\nPARTIAL RESPONSE:\n{prefix}\n\n"
            f"{QUESTION[args.mode]}"
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if getattr(tokenizer, "chat_template", None):
            kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
            if args.no_thinking:
                kwargs["enable_thinking"] = False
            try:
                prompt = tokenizer.apply_chat_template(messages, **kwargs)
            except Exception:  # noqa: BLE001 - template kwargs differ across models
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
        else:  # base checkpoints without a chat template
            prompt = f"{system}\n\n{user}\n"
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        if args.mode == "binary_logit":
            with torch.no_grad():
                logits = model(**ids).logits[0, -1]
            def token_id(*variants: str) -> list[int]:
                out = []
                for variant in variants:
                    encoded = tokenizer.encode(variant, add_special_tokens=False)
                    if len(encoded) == 1:
                        out.append(encoded[0])
                return out
            yes = token_id(" YES", "YES", "Yes")
            no = token_id(" NO", "NO", "No")
            yv = max(float(logits[i]) for i in yes)
            nv = max(float(logits[i]) for i in no)
            import math
            p = math.exp(yv) / (math.exp(yv) + math.exp(nv))
            return p, f"logits YES={yv:.1f} NO={nv:.1f}"
        with torch.no_grad():
            out = model.generate(
                **ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                temperature=None, top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0][ids["input_ids"].shape[1] :], skip_special_tokens=True)
        return parse_answer(text, args.mode), text.strip()

    cuts = [int(c) for c in args.cuts.split(",")]
    rows: list[dict[str, Any]] = []
    started = time.time()
    for index, (tid, rec) in enumerate(records, 1):
        task = tasks[tid]
        stream = ((rec.get("reasoning") or "") + "\n" + (rec.get("content") or "")).strip()
        ids = tokenizer(stream, add_special_tokens=False)["input_ids"]
        n = len(ids)
        ok = check(task, rec["content"])
        row: dict[str, Any] = {"id": tid, "scion_ok": ok, "n_tokens": n, "p": {}, "raw": {}}
        for t in cuts:
            if t > n:
                continue
            prefix = tokenizer.decode(ids[:t], skip_special_tokens=True)
            p, raw = ask(task, prefix)
            row["p"][str(t)] = p
            row["raw"][str(t)] = raw[:40]
        rows.append(row)
        print(
            f"[{index}/{len(records)}] {tid:26} {n:>4} tok ok={ok} "
            f"({time.time() - started:.0f}s)",
            file=sys.stderr,
            flush=True,
        )

    summary: dict[str, Any] = {"model": args.model, "mode": args.mode, "tasks": len(rows), "cuts": {}}
    for t in cuts:
        sub = [r for r in rows if str(t) in r["p"] and r["p"][str(t)] is not None]
        if len(sub) < 4:
            continue
        ok = [r["scion_ok"] for r in sub]
        scores = [r["p"][str(t)] for r in sub]
        wrong = [1 for s, o in zip(scores, ok) if not o]
        cancelled = [s >= 0.85 for s in scores]
        summary["cuts"][str(t)] = {
            "n": len(sub),
            "auc": auc(scores, ok),
            "mean_p_correct": round(sum(s for s, o in zip(scores, ok) if o) / max(1, sum(ok)), 4),
            "mean_p_wrong": round(sum(s for s, o in zip(scores, ok) if not o) / max(1, len(wrong)), 4),
            "at_085": {
                "cancelled": sum(cancelled),
                "false_cancels": sum(1 for c, o in zip(cancelled, ok) if c and o),
                "caught": sum(1 for c, o in zip(cancelled, ok) if c and not o),
                "leaks": sum(1 for c, o in zip(cancelled, ok) if not c and not o),
            },
        }
        for theta in (0.5, 0.7, 0.85):
            caught = false_cxl = 0
            for r in sub:
                s = r["p"][str(t)]
                if s < theta:
                    if r["scion_ok"]:
                        false_cxl += 1
                    else:
                        caught += 1
            summary["cuts"][str(t)].setdefault("cancel", {})[str(theta)] = {
                "caught": caught,
                "false_cancels": false_cxl,
            }

    Path(args.out).write_text(
        json.dumps({"model": args.model, "summary": summary, "rows": rows}, indent=1),
        encoding="utf-8",
    )
    print(f"\n=== ordinary-LLM steerer probe: {args.model} ===")
    print(f"{'t':>4} {'n':>3} {'AUC':>6} {'p_ok':>6} {'p_bad':>6} {'cxl@.5':>7} {'caught@.5':>9} {'cxl@.7':>7} {'caught@.7':>9}")
    for t in cuts:
        c = summary["cuts"].get(str(t))
        if not c:
            continue
        print(
            f"{t:>4} {c['n']:>3} {c['auc'] if c['auc'] is not None else float('nan'):>6} "
            f"{c['mean_p_correct']:>6} {c['mean_p_wrong']:>6} "
            f"{c['cancel']['0.5']['false_cancels']:>7} {c['cancel']['0.5']['caught']:>9} "
            f"{c['cancel']['0.7']['false_cancels']:>7} {c['cancel']['0.7']['caught']:>9}"
        )
    print("\nartifact:", args.out)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
