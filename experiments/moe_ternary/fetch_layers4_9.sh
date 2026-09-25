#!/usr/bin/env bash
# Deeper-prefix shards (layers 4-9) for the MoE router probe.
set -euo pipefail
DEST=/home/penis/Desktop/work/hivebench/artifacts/ternary/moe/empero-hf
BASE=https://huggingface.co/empero-ai/Qwen3.8-35B-A3B-Distill/resolve/main
for s in model-00004-of-00021.safetensors model-00005-of-00021.safetensors; do
  echo "downloading $s"
  curl -L --fail -C - "$BASE/$s" -o "$DEST/$s"
  echo "done $s"
done
echo ALLDONE
