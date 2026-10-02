#!/usr/bin/env python
"""Intern-Decision judge — the local D3 (accept/reject) for the cascade.

Wraps the model repo's own inference path (`inference.py` from
``internlm/Intern-Decision-4B``): one masked-next-token forward pass over a
complete assistant skeleton, logits read immediately before each ``<decision>``
placeholder, softmax over the field's candidate symbols, then the checkpoint's
fitted temperature (T≈1.99).  No autoregressive generation.

For the cascade's judge role we ask a single ``noul`` field: *"the candidate
answer is correct and complete"* -> P(yes), which is the calibrated confidence
the verifier policy thresholds on.

Run it with the ROCm venv (GPU) — the model is 4B bf16 (~8 GB):

    ~/Desktop/work/.venv-rocm/bin/python experiments/cascade/run_cascade.py \
        --router jev --judge local ...
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "internlm/Intern-Decision-4B"

#: Per-checkpoint evaluation temperatures from the project's
#: ``benchmarks/temperature-presets.json`` (fitted on their calibration split;
#: do not refit on test data).
TEMPERATURES = {
    "internlm/Intern-Decision-0.8B": 2.747760550702957,
    "internlm/Intern-Decision-2B": 2.100509348277736,
    "internlm/Intern-Decision-4B": 1.99241824,
}


class InternDecisionJudge:
    """One loaded engine; ``verdict(task, candidate)`` per judgement."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        *,
        device: str = "cuda",
        dtype: str = "bfloat16",
        temperature: float | None = None,
    ) -> None:
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download(model_id)
        if snapshot not in sys.path:
            sys.path.insert(0, snapshot)
        import inference  # type: ignore

        # Text-only judge: the repo's HFBackend always builds the multimodal
        # AutoProcessor (which pulls torchvision/video deps we neither need nor
        # have).  The text path never touches the processor, so stub it.
        class _TextOnlyProcessor:
            tokenizer = None

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

        inference.AutoProcessor = _TextOnlyProcessor

        self.model_id = model_id
        if temperature is None:
            temperature = TEMPERATURES.get(model_id, inference.DEFAULT_TEMPERATURE)
        self.engine = inference.DecisionEngine(
            checkpoint=snapshot,
            temperature=temperature,
            device=device,
            dtype=dtype,
        )

    def verdict(self, task: dict, candidate: str) -> tuple[float, float]:
        """P(candidate is correct) and the forward-pass latency in ms."""
        request: dict[str, Any] = {
            "state": {"task": task["prompt"], "candidate_answer": candidate},
            "questions": {
                "verdict": {
                    "type": "noul",
                    "instructions": "The candidate answer is correct and complete for the task.",
                    "criteria": {
                        "yes": "correct and complete",
                        "no": "incorrect, incomplete, or contradictory",
                    },
                }
            },
        }
        result = self.engine.predict(request)
        answer = result["answers"]["verdict"]
        return float(answer["noul"]), float(result["timing"]["inference_ms"])


class Jev9BJudge:
    """autotrust/JEV-9B — a Jev-1.13 student, transformers readout.

    The repo ships the pristine Qwen3.5-9B backbone, a PEFT LoRA (``adapter/``),
    a 24-slot decision head (``head.safetensors``) and the calibration/config
    JSONs.  One forward pass over the ``bare-v1`` prompt; the head projects the
    final normed hidden state onto the 24 verbalizer slots (``lm_head_included``
    is true, so no lm_head is added), then the per-kind temperature from
    ``calibration.json`` is applied.
    """

    def __init__(
        self,
        model_id: str = "autotrust/JEV-9B",
        *,
        device: str = "auto",
        dtype: str = "bfloat16",
    ) -> None:
        import torch
        from huggingface_hub import snapshot_download
        from peft import PeftModel
        from safetensors.torch import load_file
        from transformers import AutoModelForCausalLM, AutoTokenizer

        snapshot = snapshot_download(model_id)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(snapshot)
        base = AutoModelForCausalLM.from_pretrained(
            snapshot,
            dtype=getattr(torch, dtype),
            device_map=device if device != "auto" else "auto",
            # The iGPU is enumerated as a third CUDA device; keep the model on
            # the two dGPUs and split if it does not fit one card.
            max_memory={0: "18GiB", 1: "18GiB"} if device == "auto" else None,
        )
        # Unmerged LoRA (the vLLM reference path): no merge copy, same math.
        self.model = PeftModel.from_pretrained(base, str(Path(snapshot) / "adapter")).eval()
        head = load_file(str(Path(snapshot) / "head.safetensors"))
        self.head_w = head["proj.weight"].float()
        self.head_b = head["proj.bias"].float()
        cfg = json.loads((Path(snapshot) / "judge_config.json").read_text(encoding="utf-8"))
        self.ranges = cfg["slots"]["ranges"]
        calib = json.loads((Path(snapshot) / "calibration.json").read_text(encoding="utf-8"))
        self.temperatures = calib["per_kind"]
        self.device = str(next(self.model.parameters()).device)
        placement = getattr(self.model, "hf_device_map", None)
        print(f"[jev9b] loaded on {placement or self.device}", file=sys.stderr)

    def _decide(self, kind: str, state: str, question: str, options: list[str]) -> dict:
        prompt = (
            f"[kind] {kind}\n[state] {state}\n[question] {question}\n[options]\n"
            + "\n".join(options)
            + "\n[decision]:"
        )
        ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ].to(self.model.device)
        with self.torch.no_grad():
            transformer = self.model.get_base_model().model
            hidden = transformer(input_ids=ids, use_cache=False).last_hidden_state[:, -1]
        start, stop = self.ranges[kind]
        logits = hidden.float().cpu() @ self.head_w[start:stop].T + self.head_b[start:stop]
        probs = self.torch.softmax(logits / self.temperatures[kind], dim=-1)[0].tolist()
        return dict(zip(options, probs))

    def verdict(self, task: dict, candidate: str) -> tuple[float, float]:
        """P(candidate is correct) and the forward-pass latency in ms."""
        state = f"Task: {task['prompt']}\nCandidate answer: {candidate}"
        started = time.time()
        probs = self._decide(
            "noul",
            state,
            "The candidate answer is correct and complete for the task.",
            ["false", "true"],
        )
        return float(probs["true"]), (time.time() - started) * 1000.0


if __name__ == "__main__":  # tiny manual probe
    import json
    import time

    judge = InternDecisionJudge(device="cuda")
    task = {
        "prompt": "A rectangle has length 12 cm and width 5 cm. What is the length of its diagonal in cm? End with the final number.",
    }
    for candidate in ("13", "17", "13 cm. The diagonal is 13."):
        p, ms = judge.verdict(task, candidate)
        print(f"candidate={candidate!r:40} p_correct={p:.3f}  ({ms:.0f} ms)")
