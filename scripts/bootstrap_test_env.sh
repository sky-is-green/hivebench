#!/usr/bin/env bash
# Bootstrap the hivebench test environment from a bare worktree.
#
# Creates .venv at the repo root and installs requirements-dev.txt (pinned,
# CPU-only torch). Idempotent: re-running installs nothing new unless the
# requirements change. Prefers `uv` (fast, and it can fetch the Python version
# torch ships wheels for) and falls back to stdlib venv + pip.
#
# Usage:
#     scripts/bootstrap_test_env.sh
#     .venv/bin/python -m pytest tests/unit -q
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
VENV="$ROOT/.venv"
REQ="$ROOT/requirements-dev.txt"

if [[ ! -f "$REQ" ]]; then
  echo "bootstrap_test_env: missing $REQ" >&2
  exit 1
fi

if command -v uv >/dev/null 2>&1; then
  if [[ ! -x "$VENV/bin/python" ]]; then
    echo "bootstrap_test_env: creating $VENV (python $PYTHON_VERSION) with uv"
    uv venv --python "$PYTHON_VERSION" "$VENV"
  fi
  echo "bootstrap_test_env: installing $REQ with uv"
  # torch's CPU wheel lives on PyTorch's index while the rest live on PyPI;
  # uv's default "first index that has the package" rule would strand
  # setuptools et al. on the PyTorch index, so let it pick the best match.
  uv pip install --python "$VENV/bin/python" --index-strategy unsafe-best-match -r "$REQ"
else
  # No uv: use a Python the pinned CPU torch wheel supports (3.10-3.13).
  PY="${PYTHON_BIN:-$(command -v python3.13 || command -v python3.12 \
      || command -v python3.11 || command -v python3.10 || command -v python3 || true)}"
  if [[ -z "$PY" ]]; then
    echo "bootstrap_test_env: no python3 found; install uv or Python 3.12" >&2
    exit 1
  fi
  if ! "$PY" -c 'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] <= (3, 13) else 1)'; then
    echo "bootstrap_test_env: $PY is unsupported by the pinned CPU torch wheel " \
         "(need Python 3.10-3.13). Install uv or set PYTHON_BIN." >&2
    exit 1
  fi
  if [[ ! -x "$VENV/bin/python" ]]; then
    echo "bootstrap_test_env: creating $VENV with $PY"
    "$PY" -m venv "$VENV"
  fi
  echo "bootstrap_test_env: installing $REQ with pip"
  "$VENV/bin/python" -m pip install --upgrade pip >/dev/null
  "$VENV/bin/python" -m pip install -r "$REQ"
fi

echo "bootstrap_test_env: ready — run: .venv/bin/python -m pytest tests/unit -q"
