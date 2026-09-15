#!/usr/bin/env bash
# Start the strata harness sidecar from a REAL terminal (not the AI sandbox).
#
# Why this exists: the AI sandbox caps every process it spawns at 8 GiB of
# virtual address space (RLIMIT_AS, hard limit - cannot be raised from inside).
# torch + MiniLM exceeds that at first encode and OpenBLAS kills the worker.
# A normal shell has no such cap, which is why llama-server never had problems.
#
# Post-split layout (2026-09-12): the sidecar code lives here in hivebench, but
# the Python env (torch/fastapi/...) and the live conversation store live in the
# sibling strata-memory checkout. We run strata's venv python from THIS CWD so
# `harness`/`experiments` resolve locally and `strata` resolves via STRATA_HOME.
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1            # encoder is 12M params; 1 thread ~5ms
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

WORK_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STRATA_HOME="${STRATA_HOME:-$WORK_DIR/strata-memory}"
export STRATA_HOME
PY="$STRATA_HOME/venv/bin/python"
STATE_DIR="${STRATA_STATE_DIR:-$STRATA_HOME/harness_state}"

if [ ! -x "$PY" ]; then
    echo "strata venv python not found at: $PY" >&2
    exit 1
fi
if "$PY" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/health', timeout=2)" 2>/dev/null; then
    echo "Something already answers on :8765 - stop it first."
    exit 1
fi
exec "$PY" -m harness --no-open --state-dir "$STATE_DIR"
