# AMD FreeToken engine — Hivebench on the prism ROCm build

Hivebench can serve stacks with the AMD engine instead of a stock llama.cpp
binary. The goal is the FreeToken/Strata shape: a model whose expert working set
does not fit VRAM still runs at useful speed (dispatch-budget-aware kernels,
hot-expert residency, curated context), on consumer AMD (2× RX 7900 XT 20 GB,
ROCm).

## The engine

Repo: `sky-is-green/prism-ml-llama.cpp`, branch **`engine/amd-rig`**
(local checkout: `~/Desktop/work/prism-ml-llama.cpp`).

- upstream llama.cpp master (has `qwen4exp` / Qwen3.8-Flash-Next and
  `--lazy-mode` PLE streaming) + the HSA upload fix (`set_tensor` staging;
  prevents the ROCm dual-GPU load hang) + `LLAMA_GRAPH_DUMP`.
- The engine test rig (budget microbenchmarks, correctness harness, 125B
  scripts, measured notes) lives in that repo under `tools/engine-rig/`.
  Key numbers (dual 7900 XT, 125B MoE): ~44 s load, 460–497 t/s prefill,
  24–28 t/s decode, and a measured step budget dominated by per-kernel
  dispatch (~3.5 µs × ~5,000 kernels at this model size).

Hivebench integrates it through the existing per-backend binary convention
(`harness/models.py::_binary_for_backend`):

```
tools/backends/rocm/llama-server      # the prism build
stacks/*.json             → "backend": "rocm"
```

## Install

```sh
# build/symlink the engine into tools/backends/rocm/
scripts/install_amd_engine.sh            # symlink (keeps the build in place)
scripts/install_amd_engine.sh --copy     # copy the whole bin/ directory

# optional knobs
PRISM_DIR=~/Desktop/work/prism-ml-llama.cpp scripts/install_amd_engine.sh
SKIP_BUILD=1 scripts/install_amd_engine.sh
```

The script checks the engine checkout/branch, builds
`llama-server llama-perplexity llama-cli` when needed, installs
`tools/backends/rocm/llama-server`, and smoke-tests `--version`.

## Use

- **Stack**: `stacks/amd-freetoken.json` — Bonsai-2 27B `PQ2_0` face plus the
  1.7B worker, both on the headless card (`HIP_VISIBLE_DEVICES=1`). Apply it
  through the stack API/UI as usual (POST `/v1/stacks/amd-freetoken/apply`);
  every tier launches the prism binary with `backend: rocm`.
  The models are read from the local model library (this box:
  `~/.lmstudio/models`, which already has both Bonsai GGUFs from the T29
  runtime work).
- **One-off**: `HARNESS_LLAMA_SERVER=$PWD/tools/backends/rocm/llama-server
  python -m harness` forces the engine for every managed server.

### MoE recipe (the FreeToken target: 125B MoE on 2×20 GB)

```sh
MODELS_DIR=~/.lmstudio/models     # or whatever --models-dir points at
ln -sfn /home/penis/Desktop/work/models/qwen38-q2_0/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf "$MODELS_DIR/"
ln -sfn /home/penis/Desktop/work/models/qwen38-q2_0/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf "$MODELS_DIR/"
```

Tier shape (both shards are discovered from the first file):

```json
{
  "role": "face",
  "repo": "local/amd-freetoken",
  "file": "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf",
  "ctx": 4096,
  "ngl": 99,
  "backend": "rocm",
  "cache_k": "q8_0",
  "cache_v": "q8_0",
  "pin": "HIP_VISIBLE_DEVICES=0,1",
  "ts": "0.53,0.47"
}
```

`--lazy-mode auto` (upstream default) streams the 15 GiB PLE table on demand
and the tensor split keeps all 48 expert layers resident across both cards.
Measured on the engine rig: 44 s load, 460–497 t/s prefill, 24–28 t/s decode.

## Engine flags the stack schema cannot carry yet

`Tier` fields today: `role`, `repo`, `file`, `ctx`, `ngl`, `backend`,
`cache_k`, `cache_v`, `spec`, `mmproj`, `pin`, `ts`. That covers the MoE recipe
via defaults (mmap load, lazy-auto, flash-attn auto) but not, for example,
`--device` ordering, `-t/--threads`, `--ncmoe`, `-np`, or an explicit
`--lazy-mode`. Adding a tier field (e.g. `engine_args` or the individual keys)
touches `harness/stack/schema.py` + `harness/stack/manager.py`; that seam is
planned as a follow-up so the AMD-specific knobs become first-class instead of
env-only.

## Roadmap toward FreeToken parity (engine side)

1. **Kernel count is the budget lever** (measured): fold repeated elementwise
   chains; upstream already fuses the MoE expert reduction
   (`ggml_cuda_op_moe_weighted_reduction`).
2. **Sparse per-token expert dispatch** for the hot/cold split: the archived
   sidecar (static hot set) was performance-neutral because the cold pass still
   computes every expert; the real feature is variable-k dispatch with the hot
   set resident. Hivebench will need per-tier engine-policy fields for it
   (cache size, q* split, telemetry).
3. **Context curation (Strata side)** stays on the harness layer; stack
   profiles already shape per-tier context, and `/v1/stacks/status` reports
   live `tok_s` / VRAM per tier for the A/B batteries.

## Verification

```sh
pytest tests/unit/test_stack_schema.py -q     # amd-freetoken shape + round-trip
scripts/install_amd_engine.sh && tools/backends/rocm/llama-server --version
```
