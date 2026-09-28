#!/usr/bin/env bash
# Install the prism AMD engine as Hivebench's "rocm" llama-server backend.
#
# The engine is the ROCm llama.cpp fork (sky-is-green/prism-ml-llama.cpp,
# branch engine/amd-rig): upstream master + the HSA upload fix + the engine
# rig.  Hivebench launches per-backend binaries from
# tools/backends/<backend>/llama-server (harness/models.py), so this script
# installs the built engine there and smoke-tests it.
#
# Usage:
#   scripts/install_amd_engine.sh            # symlink the built binary
#   scripts/install_amd_engine.sh --copy     # copy the whole bin/ directory
#
# Env:
#   PRISM_DIR     engine checkout (default ~/Desktop/work/prism-ml-llama.cpp)
#   PRISM_BRANCH  branch to require   (default engine/amd-rig)
#   SKIP_BUILD=1  do not run cmake; fail if build-hip/bin is missing
#
# The installed binary keeps llama.cpp's absolute RUNPATH into
# $PRISM_DIR/build-hip/bin, so the symlink stays valid while that build lives;
# use --copy (and keep the build) if you later move things around.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO_ROOT/tools/backends/rocm"

PRISM_DIR="${PRISM_DIR:-$HOME/Desktop/work/prism-ml-llama.cpp}"
PRISM_BRANCH="${PRISM_BRANCH:-engine/amd-rig}"
MODE="symlink"
if [[ "${1:-}" == "--copy" ]]; then
    MODE="copy"
fi

echo "[amd-engine] repo:  $REPO_ROOT"
echo "[amd-engine] engine: $PRISM_DIR (branch $PRISM_BRANCH)"

if [[ ! -d "$PRISM_DIR" ]]; then
    cat >&2 <<EOF
[amd-engine] engine checkout not found: $PRISM_DIR
  git clone --single-branch --branch $PRISM_BRANCH \\
      https://github.com/sky-is-green/prism-ml-llama.cpp "$PRISM_DIR"
EOF
    exit 1
fi

if [[ -d "$PRISM_DIR/.git" ]]; then
    current="$(git -C "$PRISM_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
    if [[ "$current" != "$PRISM_BRANCH" ]]; then
        echo "[amd-engine] warning: engine is on '$current', expected '$PRISM_BRANCH'" >&2
    fi
fi

BIN="$PRISM_DIR/build-hip/bin"
if [[ "${SKIP_BUILD:-0}" != "1" && ! -x "$BIN/llama-server" ]]; then
    echo "[amd-engine] building llama-server / llama-perplexity / llama-cli ..."
    cmake -S "$PRISM_DIR" -B "$PRISM_DIR/build-hip" \
        -DGGML_HIP=ON -DGGML_HIP_GRAPHS=ON -DGGML_NATIVE=ON \
        -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF \
        -DCMAKE_HIP_ARCHITECTURES=gfx1100
    cmake --build "$PRISM_DIR/build-hip" -j 12 \
        --target llama-server llama-perplexity llama-cli
fi
if [[ ! -x "$BIN/llama-server" ]]; then
    echo "[amd-engine] no llama-server under $BIN (build it, or unset SKIP_BUILD)" >&2
    exit 1
fi

mkdir -p "$DEST"
if [[ "$MODE" == "copy" ]]; then
    cp -a "$BIN"/. "$DEST"/
    echo "[amd-engine] copied $BIN -> $DEST"
else
    ln -sfn "$BIN/llama-server" "$DEST/llama-server"
    echo "[amd-engine] linked $DEST/llama-server -> $BIN/llama-server"
fi

"$DEST/llama-server" --version >/dev/null
echo "[amd-engine] ok: $("$DEST/llama-server" --version 2>/dev/null | head -1)"

cat <<EOF

[amd-engine] next:
  - apply a stack with "backend": "rocm"  (see stacks/ember.json)
  - or point the harness at it directly:
      scripts/install_amd_engine.sh && \\
      HARNESS_LLAMA_SERVER=$DEST/llama-server python -m harness
  - engine docs: docs/EMBER.md
EOF
