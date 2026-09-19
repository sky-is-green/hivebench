#!/usr/bin/env bash
# T12 — run one remote stage, detached by default, with logging + a run record.
#
# Usage:
#   scripts/cloud/run_remote.sh ptq -- python -m experiments.ternary.run_quant \
#       --config configs/ternary/27b.yaml --source safetensors \
#       --model-dir models/Qwen3.8-27B
#   scripts/cloud/run_remote.sh kd --foreground -- python -m experiments.ternary.kd_data ...
#
# The command runs from $REMOTE_DIR with the remote venv and ~/.tbr_env (mode
# 600) sourced; stdout/stderr land in logs/<name>.log, the PID in runs/<name>.pid.
set -eu
. "$(dirname "$0")/lib.sh"
require_env HOST

name="${1:-}"
[ -n "$name" ] || die "usage: run_remote.sh <name> [--foreground] -- <cmd...>"
shift
mode="detach"
if [ "${1:-}" = "--foreground" ]; then
  mode="foreground"
  shift
fi
[ "${1:-}" = "--" ] && shift || true
[ "$#" -ge 1 ] || die "missing remote command after --"

remote="$(remote_dir)"
qcmd="$(printf '%q' "$*")"
if [ "$mode" = "detach" ]; then
  ssh_do "cd '$remote' && . ~/.tbr_env && . .venv/bin/activate && \
    nohup bash -c $qcmd > 'logs/$name.log' 2>&1 < /dev/null & \
    echo \$! > 'runs/$name.pid'; cat 'runs/$name.pid'"
  log "stage '$name' started; logs/$name.log (check: status.sh $name)"
else
  ssh_do "cd '$remote' && . ~/.tbr_env && . .venv/bin/activate && bash -c $qcmd"
fi
