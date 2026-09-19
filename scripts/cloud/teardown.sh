#!/usr/bin/env bash
# T12 — end a rental: pull artifacts, delete the token file, cancel the timer,
# report elapsed/cost, and (STOP_NOW=1) power the instance off.
#
# Usage: scripts/cloud/teardown.sh [destination]
set -eu
. "$(dirname "$0")/lib.sh"
require_env HOST
require_cmd ssh

dest="${1:-artifacts/cloud}"
remote="$(remote_dir)"

"$(dirname "$0")/pull.sh" "$dest"

if is_dry; then
  printf '  DRY-RUN: rm -f ~/.tbr_env; sudo shutdown -c; STOP_NOW=%s\n' "${STOP_NOW:-0}"
  exit 0
fi

ssh_do "rm -f ~/.tbr_env"
started="$(ssh_do "cat '$remote/runs/.started_at' 2>/dev/null" || true)"
if [ -n "$started" ]; then
  elapsed_min="$(awk -v s="$started" -v n="$(date -u +%s)" 'BEGIN { printf "%d", (n - s) / 60 }')"
  log "instance ran ~${elapsed_min} min"
  if [ -n "${PRICE_PER_HOUR:-}" ]; then
    awk -v m="$elapsed_min" -v p="$PRICE_PER_HOUR" \
      'BEGIN { printf "estimated cost: $%.2f (%.1f h x $%s)\n", (m/60.0)*p, m/60.0, p }'
  fi
fi
ssh_do "sudo shutdown -c >/dev/null 2>&1 || true"
if [ "${STOP_NOW:-0}" = "1" ]; then
  ssh_do "sudo shutdown -h now"
  log "shutdown requested"
else
  log "auto-shutdown timer cancelled; STOP_NOW=1 to power off now"
fi
