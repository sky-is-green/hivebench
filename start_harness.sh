#!/usr/bin/env bash
# Start the strata harness sidecar from a REAL terminal (not the AI sandbox).
#
# Why this exists: the AI sandbox caps every process it spawns at 8 GiB of
# virtual address space (RLIMIT_AS, hard limit - cannot be raised from inside).
# torch + MiniLM exceeds that at first encode and OpenBLAS kills the worker.
# A normal shell has no such cap, which is why llama-server never had problems.
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1            # encoder is 12M params; 1 thread ~5ms
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
if venv/bin/python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/health', timeout=2)" 2>/dev/null; then
    echo "Something already answers on :8765 - stop it first."
    exit 1
fi
exec venv/bin/python -m harness --no-open
