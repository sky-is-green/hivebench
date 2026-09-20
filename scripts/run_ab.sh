#!/usr/bin/env bash
# TBR head-to-head A/B runner: serve each model through the Prism fork with the
# eval driver's hardening (OOM preflight, configurable timeouts, warmup,
# liveness, deadline, run manifest) and write one report per model.
#
# Preconditions: refs fetched (see fetch_refs) and the fork binary present.
# Env overrides: HIP_VISIBLE_DEVICES, HARNESS_VRAM_GB, MAX_CONVS, DEADLINE,
#                FORCE=1 (skip the GPU-contention refusal), FETCH=1 (download refs).
set -uo pipefail

cd "$(dirname "$0")/.." || exit 2
ROOT=$PWD
FORK=${FORK:-artifacts/ternary/oracle/prism-fork/bin/llama-prism-b10709-9a9394a}
PIN=${PIN:-/home/penis/Desktop/work/worktrees/strata-memory/hivebench-STRATA-PIN}
PY=${PY:-/home/penis/Desktop/work/strata-memory/venv/bin/python}
PORT=${PORT:-8090}
MAX_CONVS=${MAX_CONVS:-10}
DEADLINE=${DEADLINE:-7200}
STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-1800}
REQ_TIMEOUT=${REQ_TIMEOUT:-900}
OUTDIR=${OUTDIR:-artifacts/ternary/eval}

export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-1}
export HARNESS_VRAM_GB=${HARNESS_VRAM_GB:-20}
export LD_LIBRARY_PATH="$ROOT/$FORK"
export STRATA_HOME=$PIN
export PYTHONPATH=$PIN/strata
export GGML_CUDA_NO_PINNED=1

PQ2=${PQ2:-artifacts/ternary/oracle/bonsai27/Ternary-Bonsai-2-27B-PQ2_0.gguf}
Q8=${Q8:-artifacts/ternary/refs/Qwen3.8-27B-Q8_0.gguf}
Q4=${Q4:-artifacts/ternary/refs/Qwen3.8-27B-UD-Q4_K_M.gguf}
FORBID=${FORBID:-dflash,uncensored,turbo,fable,mtp,nuslerp,dau}

log() { printf '[ab] %s\n' "$*"; }
die() { printf '[ab] ERROR: %s\n' "$*" >&2; exit 2; }

sweep_port() {
  "$PY" - "$PORT" <<'PY' || true
import sys
from harness.models import stop_stale_server
print("[ab] sweep:", stop_stale_server("127.0.0.1", int(sys.argv[1])))
PY
}

gpu_guard() {
  local busy
  busy=$("$PY" - <<'PY'
import json
from harness.models import competing_gpu_processes
print(json.dumps(competing_gpu_processes()))
PY
)
  if [ "$busy" != "[]" ]; then
    if [ "${FORCE:-0}" = "1" ]; then
      log "WARNING: competing llama processes present, FORCE=1: $busy"
    else
      die "another llama.cpp process is running (one heavy ROCm process at a time): $busy  [FORCE=1 to override]"
    fi
  fi
}

disk_guard() {
  local need_gb=${NEED_GB:-0}
  [ "$need_gb" = "0" ] && return 0
  local free_gb
  free_gb=$(df -Pk "$ROOT" | awk 'NR==2 {printf "%d", $4/1024/1024}')
  if [ "$free_gb" -lt "$need_gb" ]; then
    die "need ${need_gb}GB free, have ${free_gb}GB"
  fi
  log "disk free ${free_gb}GB (need ${need_gb}GB)"
}

fetch_refs() {
  [ "${FETCH:-0}" = "1" ] || return 0
  local base="https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main"
  mkdir -p "$(dirname "$Q8")"
  fetch_one "$base/Qwen3.8-27B-Q8_0.gguf" "$Q8" 29050000000
  fetch_one "$base/Qwen3.8-27B-UD-Q4_K_M.gguf" "$Q4" 16460000000
}

fetch_one() {
  local url=$1 dest=$2 expect=$3
  if [ -f "$dest" ]; then
    local have
    have=$(stat -c %s "$dest")
    if [ "$have" -ge "$expect" ]; then
      log "have $(basename "$dest") (${have} bytes)"
      return 0
    fi
    log "resuming $(basename "$dest") (${have}/${expect})"
  fi
  curl -fL -C - --retry 3 --retry-delay 2 -o "$dest" "$url" || die "download failed: $url"
  local have
  have=$(stat -c %s "$dest")
  [ "$have" -ge "$expect" ] || die "short download $(basename "$dest"): $have < $expect"
  log "fetched $(basename "$dest") (${have} bytes)"
}

run_one() {
  local label=$1 gguf=$2 expect_name=$3
  if [ ! -f "$gguf" ]; then
    log "SKIP $label: missing $gguf"
    return 1
  fi
  log "=== $label: $gguf ==="
  "$PY" -m experiments.ternary_eval \
    --gguf "$gguf" \
    --fork-bin "$FORK/llama-server" \
    --host 127.0.0.1 --port "$PORT" \
    --no-thinking \
    --warmup \
    --startup-timeout "$STARTUP_TIMEOUT" \
    --timeout "$REQ_TIMEOUT" \
    --deadline "$DEADLINE" \
    --max-convs "$MAX_CONVS" \
    --expect-name "$expect_name" \
    --forbid "$FORBID" \
    --output "$OUTDIR/ab-${label}.json"
  local rc=$?
  log "$label exit=$rc"
  return $rc
}

sweep_port
gpu_guard
disk_guard
fetch_refs

rc=0
run_one pq2_0 "$PQ2" "Ternary-Bonsai-2-27B" || rc=1
run_one q8_0 "$Q8" "Qwen3.8-27B" || rc=1
run_one q4_k_m "$Q4" "Qwen3.8-27B" || rc=1

sweep_port
log "reports in $OUTDIR/ab-*.json"
[ "$rc" = "0" ] && log "A/B DONE" || log "A/B DONE WITH FAILURES"
exit $rc
