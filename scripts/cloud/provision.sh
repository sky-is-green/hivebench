#!/usr/bin/env bash
# T12 — provision a rented CUDA GPU box for the TBR runs.
#
# HOTSPOT (`HIVE-PLAN.md` §6): secrets + cost logic; commit alone. Provider-
# agnostic: the human creates the instance, exports HOST/SSH_KEY/HF_TOKEN, and
# this script ships the repo, installs the remote venv, downloads the pinned
# teacher checkpoint, stores the token in a 600 env file, and arms the
# auto-shutdown timer. The timer is the cost cap of last resort; PRICE_PER_HOUR
# plus MAX_COST_USD (optional) shrink MAX_HOURS before anything is scheduled.
set -eu
. "$(dirname "$0")/lib.sh"

require_env HOST HF_TOKEN
require_cmd rsync
require_cmd ssh

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
REMOTE_DIR="$(remote_dir)"
MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3.8-27B}"
MODEL_REVISION="${MODEL_REVISION:-1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
SKIP_DEPS="${SKIP_DEPS:-0}"
SKIP_MODEL="${SKIP_MODEL:-0}"
MAX_HOURS="$(cost_cap_hours)"

[ -f "$REPO_ROOT/experiments/ternary/run_quant.py" ] \
  || die "REPO_ROOT is not the hivebench checkout: $REPO_ROOT"

if is_dry; then
  log "DRY-RUN provision plan for $HOST:$REMOTE_DIR (auto-shutdown in ${MAX_HOURS}h)"
  echo "  1. mkdir -p $REMOTE_DIR/{models,logs,runs,artifacts}"
  echo "  2. rsync $REPO_ROOT/ -> $HOST:$REMOTE_DIR/ (excl. venv, artifacts, .git)"
  echo "  3. remote venv + torch ($TORCH_INDEX_URL) + transformers/safetensors/datasets/pyyaml/huggingface_hub"
  echo "  4. write ~/.tbr_env (mode 600) with HF_TOKEN"
  echo "  5. hf download $MODEL_REPO@${MODEL_REVISION:0:12} -> $REMOTE_DIR/models/Qwen3.8-27B"
  echo "  6. arm auto-shutdown: sudo shutdown -h +$((MAX_HOURS * 60))"
  exit 0
fi

log "provisioning $HOST:$REMOTE_DIR (auto-shutdown in ${MAX_HOURS}h)"
ssh_do "mkdir -p '$REMOTE_DIR/models' '$REMOTE_DIR/logs' '$REMOTE_DIR/runs' '$REMOTE_DIR/artifacts'"

log "shipping repo"
rsync -az --delete-excluded \
  --exclude venv --exclude artifacts --exclude .git --exclude __pycache__ --exclude '*.pyc' \
  -e "$(rsync_ssh)" "$REPO_ROOT/" "$HOST:$REMOTE_DIR/"

if [ "$SKIP_DEPS" != "1" ]; then
  log "installing remote python deps"
  ssh_do "cd '$REMOTE_DIR' && python3 -m venv .venv && . .venv/bin/activate && \
    python -m pip install -q --upgrade pip && \
    python -m pip install -q torch --index-url '$TORCH_INDEX_URL' && \
    python -m pip install -q transformers safetensors datasets pyyaml 'huggingface_hub[cli]' numpy"
fi

if [ "$SKIP_MODEL" != "1" ]; then
  log "downloading $MODEL_REPO@${MODEL_REVISION:0:12} (pinned)"
  ssh_do "cd '$REMOTE_DIR' && . .venv/bin/activate && \
    (hf download '$MODEL_REPO' --revision '$MODEL_REVISION' --local-dir models/Qwen3.8-27B || \
     huggingface-cli download '$MODEL_REPO' --revision '$MODEL_REVISION' --local-dir models/Qwen3.8-27B)"
fi

# Secret handling: the token only ever lands in a mode-600 env file the stages
# source; teardown deletes it. Never written to the repo or logs.
printf 'export HF_TOKEN=%q\n' "$HF_TOKEN" \
  | ssh_do "umask 077 && cat > ~/.tbr_env && chmod 600 ~/.tbr_env"
ssh_do "date -u +%s > '$REMOTE_DIR/runs/.started_at'"
arm_shutdown "$MAX_HOURS"

log "provisioned. stages source ~/.tbr_env; auto-shutdown armed at ${MAX_HOURS}h"
log "next: scripts/cloud/run_remote.sh <name> -- <command...>; status.sh; teardown.sh"
