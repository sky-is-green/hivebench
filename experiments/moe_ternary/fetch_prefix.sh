#!/usr/bin/env bash
# Partial download for the MoE router probe (E1): embedding + layers 0-3 only.
set -euo pipefail
DEST=/home/penis/Desktop/work/hivebench/artifacts/ternary/moe/empero-hf
mkdir -p "$DEST"
BASE=https://huggingface.co/empero-ai/Qwen3.8-35B-A3B-Distill/resolve/main
for f in config.json tokenizer.json tokenizer_config.json generation_config.json model.safetensors.index.json; do
  curl -sL --fail "$BASE/$f" -o "$DEST/$f" || echo "warn: $f failed"
done
for s in model-00001-of-00021.safetensors model-00002-of-00021.safetensors model-00003-of-00021.safetensors; do
  echo "downloading $s"
  curl -L --fail -C - "$BASE/$s" -o "$DEST/$s"
  echo "done $s"
done
echo ALLDONE
