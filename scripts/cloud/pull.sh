#!/usr/bin/env bash
# T12 — rsync artifacts back from the instance (default: artifacts/cloud/).
#
# Usage: scripts/cloud/pull.sh [destination]
set -eu
. "$(dirname "$0")/lib.sh"
require_env HOST
require_cmd rsync

dest="${1:-artifacts/cloud}"
remote="$(remote_dir)"
if is_dry; then
  printf '  DRY-RUN: rsync %s:%s/artifacts/ -> %s/\n' "$HOST" "$remote" "$dest"
  exit 0
fi
mkdir -p "$dest"
rsync -az -e "$(rsync_ssh)" "$HOST:$remote/artifacts/" "$dest/"
log "pulled $HOST:$remote/artifacts/ -> $dest/"
