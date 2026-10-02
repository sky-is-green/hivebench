#!/usr/bin/env python
"""Ternary decision-model pipeline: JEV-9B bf16 -> PQ2_0.

The research question: does ternary quantization preserve a decision model's
calibrated probabilities, and can our KD machinery distill the bf16 teacher's
decision distribution into the ternary student?

Steps:

- ``merge``     base + PEFT LoRA, then bake the 24-slot decision head into the
                ``lm_head`` rows so a first-position logprob readout over the
                verbalizer tokens gives the decision distribution.
- ``convert``   merged HF -> f16 GGUF with the fork's ``convert_hf_to_gguf.py``
                (``Qwen3_5ForCausalLM`` is registered in ``conversion/qwen.py``).
- ``quantize``  f16 GGUF -> PQ2_0 (2.13 bpw, Prism ternary) or TQ2_0.
- ``all``       merge -> convert -> quantize.

The head bias cannot ride in a GGUF lm_head, so ``head.safetensors`` and the
calibration/config JSONs are copied next to the merged dir; the eval applies
bias + per-kind temperature client-side (exactly the model card's math).

Run with the ROCm venv and the iGPU hidden:

    HIP_VISIBLE_DEVICES=0,1 ~/Desktop/work/.venv-rocm/bin/python \
        experiments/cascade/ternarize.py all \
        --out-dir ~/.local/share/hivebench-litellm/jev9b-ternary --type PQ2_0
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

FORK = Path("/home/penis/llama.cpp")
CONVERT = FORK / "convert_hf_to_gguf.py"
GGUF_PY = FORK / "gguf-py"
QUANTIZE = FORK / "build" / "bin" / "llama-quantize"
DEFAULT_MODEL = "autotrust/JEV-9B"


def step_merge(model_id: str, out_dir: Path, dtype: str = "bfloat16") -> Path:
    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    snapshot = Path(snapshot_download(model_id))
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    base = AutoModelForCausalLM.from_pretrained(
        snapshot,
        dtype=getattr(torch, dtype),
        device_map="auto",
        max_memory={0: "18GiB", 1: "18GiB"},
    )
    model = PeftModel.from_pretrained(base, str(snapshot / "adapter")).merge_and_unload()

    # Bake the decision head into the lm_head rows for the verbalizer tokens.
    config = json.loads((snapshot / "judge_config.json").read_text(encoding="utf-8"))
    head = load_file(str(snapshot / "head.safetensors"))
    lm_head = model.get_output_embeddings()
    weight = lm_head.weight.data
    head_w = head["proj.weight"].to(device=weight.device, dtype=weight.dtype)
    with torch.no_grad():
        for slot, token_id in enumerate(config["verbalizer_ids"]):
            weight[token_id] = head_w[slot]
    print(f"[merge] baked {len(config['verbalizer_ids'])} decision rows into lm_head")

    AutoTokenizer.from_pretrained(snapshot).save_pretrained(out_dir)
    model.save_pretrained(out_dir, safe_serialization=True, max_shard_size="4GB")
    for name in ("head.safetensors", "judge_config.json", "calibration.json"):
        shutil.copy2(snapshot / name, out_dir / name)
    print(f"[merge] done in {(time.time() - started) / 60:.1f} min -> {out_dir}")
    return out_dir


def step_convert(hf_dir: Path, out_file: Path) -> Path:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(GGUF_PY))
    cmd = [
        sys.executable, str(CONVERT), str(hf_dir),
        "--outfile", str(out_file), "--outtype", "f16",
    ]
    print("[convert]", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)
    print(f"[convert] done -> {out_file} ({out_file.stat().st_size / 2**30:.1f} GiB)")
    return out_file


def step_quantize(f16: Path, out_file: Path, qtype: str) -> Path:
    cmd = [str(QUANTIZE), str(f16), str(out_file), qtype]
    print("[quantize]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"[quantize] done -> {out_file} ({out_file.stat().st_size / 2**30:.2f} GiB)")
    return out_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("merge", "convert", "quantize", "all"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--out-dir",
        default=str(Path.home() / ".local/share/hivebench-litellm/jev9b-ternary"),
    )
    parser.add_argument("--type", default="PQ2_0", help="llama-quantize type")
    parser.add_argument("--hf-dir", default="", help="merged dir (convert step)")
    parser.add_argument("--f16", default="", help="f16 GGUF (quantize step)")
    args = parser.parse_args()

    out = Path(args.out_dir)
    hf_dir = Path(args.hf_dir) if args.hf_dir else out / "hf"
    f16 = Path(args.f16) if args.f16 else out / "jev9b-f16.gguf"
    quant = out / f"jev9b-{args.type.lower()}.gguf"

    if args.step in ("merge", "all"):
        step_merge(args.model, hf_dir)
    if args.step in ("convert", "all"):
        step_convert(hf_dir, f16)
    if args.step in ("quantize", "all"):
        step_quantize(f16, quant, args.type)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
