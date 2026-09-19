#!/usr/bin/env bash
# T12 — shared helpers for the cloud runbook scripts.
#
# Sourced by the other scripts; defines functions only. Every secret comes from
# the environment (HF_TOKEN), never from a committed file or argv.
#
# Env contract (all optional unless marked):
#   HOST            (required) user@host of the rented instance
#   SSH_KEY         path to the private key
#   SSH_PORT        non-default SSH port
#   REMOTE_DIR      remote checkout dir (default: tbr)
#   DRY_RUN=1       print the plan, execute nothing
#   MAX_HOURS       hard auto-shutdown cap (default 6)
#   MAX_COST_USD    optional dollar cap; PRICE_PER_HOUR converts it to hours
#   PRICE_PER_HOUR  optional instance price
set -eu

DRY_RUN="${DRY_RUN:-0}"

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }
is_dry() { [ "$DRY_RUN" = "1" ]; }

require_env() {
  for name in "$@"; do
    if eval "[ -z \"\${$name:-}\" ]"; then
      die "$name is required (export it; never commit secrets)"
    fi
  done
}

require_cmd() { command -v "$1" >/dev/null 2>&1 || die "missing local command: $1"; }

# run CMD... — execute, or print under DRY_RUN.
run() {
  if is_dry; then
    printf '  DRY-RUN: %s\n' "$*"
  else
    log "+ $*"
    "$@"
  fi
}

SSH_COMMON_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15)

# ssh_do CMD — run a single command string on the remote box.
ssh_do() {
  [ -n "${HOST:-}" ] || die "HOST is required (user@host)"
  local cmd="$1"
  if is_dry; then
    printf '  DRY-RUN: ssh %s -- %s\n' "$HOST" "$cmd"
    return 0
  fi
  local opts=("${SSH_COMMON_OPTS[@]}")
  [ -n "${SSH_KEY:-}" ] && opts+=(-i "$SSH_KEY")
  [ -n "${SSH_PORT:-}" ] && opts+=(-p "$SSH_PORT")
  ssh "${opts[@]}" "$HOST" "$cmd"
}

rsync_ssh() {
  local opts="ssh"
  [ -n "${SSH_KEY:-}" ] && opts="$opts -i $SSH_KEY"
  [ -n "${SSH_PORT:-}" ] && opts="$opts -p $SSH_PORT"
  printf '%s -o BatchMode=yes -o StrictHostKeyChecking=accept-new' "$opts"
}

# arm_shutdown HOURS — schedule the remote auto-shutdown (cancellable).
arm_shutdown() {
  local hours="$1"
  ssh_do "sudo shutdown -h +$((hours * 60)) 'TBR auto-shutdown: ${hours}h cost cap' >/dev/null 2>&1 || true"
}

# cost_cap_hours — reconcile MAX_HOURS with MAX_COST_USD/PRICE_PER_HOUR.
cost_cap_hours() {
  local hours="${MAX_HOURS:-6}"
  if [ -n "${MAX_COST_USD:-}" ] && [ -n "${PRICE_PER_HOUR:-}" ]; then
    local allowed
    allowed="$(awk -v c="$MAX_COST_USD" -v p="$PRICE_PER_HOUR" 'BEGIN { printf "%d", c / p }')"
    if [ "$allowed" -lt "$hours" ]; then
      log "cost cap \$${MAX_COST_USD} at \$${PRICE_PER_HOUR}/h allows ${allowed}h (< MAX_HOURS=${hours})"
      hours="$allowed"
    fi
  fi
  [ "$hours" -ge 1 ] || die "cost cap leaves no runnable hours"
  printf '%s' "$hours"
}

remote_dir() { printf '%s' "${REMOTE_DIR:-tbr}"; }
