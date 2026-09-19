#!/usr/bin/env bash
# T12 — remote run status: pid liveness and log tail (TAIL=8 by default).
#
# Usage: scripts/cloud/status.sh [name]
set -eu
. "$(dirname "$0")/lib.sh"
require_env HOST

name="${1:-}"
remote="$(remote_dir)"
if [ -n "$name" ]; then
  ssh_do "cd '$remote' && pid=\$(cat 'runs/$name.pid' 2>/dev/null || echo none); \
    if [ \"\$pid\" != none ] && kill -0 \"\$pid\" 2>/dev/null; then \
      echo '$name RUNNING pid='\$pid; else echo '$name stopped (pid='\$pid')'; fi; \
    echo '--- logs/$name.log ---'; tail -n ${TAIL:-8} 'logs/$name.log' 2>/dev/null || true"
else
  ssh_do "cd '$remote' && echo '--- runs ---' && ls -l runs 2>/dev/null || true; \
    echo '--- logs ---' && ls -l logs 2>/dev/null || true"
fi
