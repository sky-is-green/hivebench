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
#
# Settings: reads strata_port and strata_state_dir from Unsloth Studio's
# app_settings table (if available), falling back to env vars or defaults.
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1            # encoder is 12M params; 1 thread ~5ms
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

WORK_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STRATA_HOME="${STRATA_HOME:-$WORK_DIR/strata-memory}"
export STRATA_HOME
PY="$STRATA_HOME/venv/bin/python"

# --- Read settings from Unsloth Studio app_settings (if available) ---
STUDIO_DB="${HOME}/.unsloth/studio/studio.db"
studio_setting() {
    local key="$1" default="$2"
    if command -v sqlite3 >/dev/null 2>&1 && [ -f "$STUDIO_DB" ]; then
        local val
        val=$(sqlite3 "$STUDIO_DB" "SELECT value_json FROM app_settings WHERE key='$key';" 2>/dev/null | tr -d '"')
        if [ -n "$val" ]; then
            echo "$val"
            return
        fi
    fi
    echo "$default"
}

PORT="$(studio_setting strata_port "${STRATA_PORT:-8765}")"
STATE_DIR="$(studio_setting strata_state_dir "${STRATA_STATE_DIR:-$STRATA_HOME/harness_state}")"

if [ ! -x "$PY" ]; then
    echo "strata venv python not found at: $PY" >&2
    exit 1
fi

# Check if port is already in use
if "$PY" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:$PORT/health', timeout=2)" 2>/dev/null; then
    echo "Something already answers on :$PORT - stop it first."
    exit 1
fi

echo "Starting strata sidecar on port $PORT (state: $STATE_DIR)"
exec "$PY" -m harness --no-open --port "$PORT" --state-dir "$STATE_DIR"
