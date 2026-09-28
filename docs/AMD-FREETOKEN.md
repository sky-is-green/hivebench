# AMD FreeToken engine — Hivebench on the prism ROCm build

Hivebench can serve stacks with the AMD engine instead of a stock llama.cpp
binary. The goal is the FreeToken/Strata shape: a model whose expert working set
does not fit VRAM still runs at useful speed (dispatch-budget-aware kernels,
hot-expert residency, curated context), on consumer AMD (2× RX 7900 XT 20 GB,
ROCm).

## Status (2026-09-28 evening)

- **Verified end-to-end** (T1/T3): the 125B MoE applies through the stack API
  on the prism ROCm binary, loads in ~42 s and serves at 446–534 t/s prefill /
  26–28 t/s decode; status reports `backend: "rocm"` per tier.
- **Engine A/B** (T2): prism vs a stock `origin/master` build on the same
  stack — +33–37% prefill, +11% decode, byte-identical greedy output.
- **Engine-flags seam** (T4): validated `engine_args` on a tier; the 125B
  stack pins `--lazy-mode on -np 1` (see below).
- **FreeToken expert cache (T5, first cut)**: the engine branch
  `engine/moe-slot-cache` adds `--moe-slot-cache N` (or `auto`): host-placed
  expert banks (via `-ncmoe`/`-cmoe`/`-ot`) keep hot experts per layer in VRAM,
  decode fetches misses over PCIe and computes every routed expert on the GPU.
  Measured warm on the 125B (185-token prompt, 64-token greedy): all-resident
  28.0 t/s; `-ncmoe 12` (8.1 GiB off) 19.1 → **24.1** with `-msc 256` (62% hit
  rate); `-ncmoe 24` (16.2 GiB off) 12.9 → **20.8** (+61%); `-ncmoe 36`
  (24.3 GiB off) 4.4 → **8.2** (+86%), outputs token-identical to the
  all-resident run. When the slots cover a layer (`-msc auto`, or a size >= the
  expert count) the bank is promoted once and runs at all-GPU speed
  (`-ncmoe 12 -msc 512`: 27.4 t/s). On the 35B-A3B (one card) 64 slots
  (2.5 GiB) buy 60.0 t/s vs 37.8 t/s CPU-only, and full coverage equals the
  all-GPU 108.9 t/s.
  `stacks/amd-freetoken-125b-lowvram.json` is this config through the harness.
  (Upstream FreeToken's RDNA3 foundation and q*/cache-budget policies are
  reviewed in the engine rig's `SLOT-CACHE-20260928.md`.)
- **Not done** (T5 remainder): prefill streaming for cached layers (prefill
  still falls back to CPU compute), the q* CPU overflow split for capped
  fetches, runtime cache resizing, and telemetry endpoints. The legacy Bonsai
  `PQ2_0` stack still cannot load on `engine/amd-rig` (fork-private tensor
  type 142 lives only on `moe-corr-runtime`).

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

- **Stack**: `stacks/amd-freetoken-125b.json` — the actual FreeToken shape: the
  125B Qwen3.8-Flash-Next MoE (Q2_0 experts, `--lazy-mode auto` streams the
  15.3 GiB PLE table) split headless-first across both cards
  (`pin HIP_VISIBLE_DEVICES=1,0`, `ts 0.522,0.478`; see the note below on why
  the env order, not `--device`, carries the ordering). Apply it through the
  stack API (`POST /v1/stacks/amd-freetoken-125b/apply`); the tier launches the
  prism binary with `backend: rocm`.
- **Legacy Bonsai stack**: `stacks/amd-freetoken.json` (Bonsai-2 27B/1.7B) is
  **not loadable on `engine/amd-rig`**: the released PQ2_0 GGUFs use the
  fork-private tensor type 142 (`GGML_TYPE_PQ2_0`, `general.file_type 141`),
  which only the legacy `moe-corr-runtime` line carries. Keep the stack file
  for that forward-port; smoke the engine with the 125B stack instead.
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

`stacks/amd-freetoken-125b.json` is this recipe as an applied stack. Two
deliberate differences from the bare rig command:

- `pin` is `HIP_VISIBLE_DEVICES=1,0` because the rig's `--device ROCm1,ROCm0`
  cannot be expressed by the stack schema yet (the `engine_args` seam below);
  the env order gives llama.cpp the same device order (headless first).
- `ts` is `0.522,0.478` (not `0.53,0.47`): the T36 residency planner charges
  the whole shard-1 file plus a 1.5 GiB compute buffer per tier and refuses
  0.53 on the headless card by ~0.2 GiB. The nudge keeps 53/47 in spirit; the
  measured residency (19.5 / 18.4 GiB used) shows the planner's estimate is
  conservative, not wrong (see the verified-end-to-end section).

## Engine flags: the `engine_args` seam

`Tier` fields today: `role`, `repo`, `file`, `ctx`, `ngl`, `backend`,
`cache_k`, `cache_v`, `spec`, `mmproj`, `pin`, `ts`, `engine_args`. `ts` is
emitted as `--split-mode layer --tensor-split <ts>` and `pin` is the process
env (`HIP_VISIBLE_DEVICES=...`); everything else the MoE recipe needs runs on
defaults (mmap load, flash-attn auto, threads auto — the engine picked 8
threads on this box by itself).

`schema.py` adds a **validated `engine_args`** list to a tier: tokens are
appended to the launch verbatim, last, after the managed flags. Only the
curated engine knobs are accepted (see `ENGINE_ARGS_ALLOWED`): `--device`,
`--n-cpu-moe`, `--lazy-mode`, `--flash-attn`, `--threads`,
`--threads-batch`, `--parallel`, `--load-mode`, `--batch-size`,
`--ubatch-size`, `--split-mode`, `--main-gpu`, `--n-gpu-layers`, `--no-mmap`,
`--mlock`, `--no-op-offload`, `--cpu-moe`/`-cmoe`, `--n-cpu-moe`/`-ncmoe`,
`--moe-slot-cache`/`-msc` (long and short forms; values pass through).
Model path, host, port and sampling flags stay owned by the managed launch, so
an authored stack cannot hijack them. `amd-freetoken-125b` uses it for the
rig's `--lazy-mode on -np 1`; `amd-freetoken-125b-lowvram` adds
`-ncmoe 12 -msc 128` (12 layers' experts host-side, 128 GPU slots each).

Two harness bugs blocked the first end-to-end run and are fixed on this branch:

- `harness/models.py` resolved the repo root one level too high (left over
  from the flat-package move), so `tools/backends/rocm/llama-server` was never
  found; backend selection always refused.
- `harness/stack/manager.py` emitted `--ts`, which no llama.cpp understands
  (`-ts` / `--tensor-split` is the real flag); every stack with a `ts` died at
  startup with `invalid argument: --ts`.

## Roadmap toward FreeToken parity (engine side)

1. **Kernel count is the budget lever** (measured): fold repeated elementwise
   chains; upstream already fuses the MoE expert reduction
   (`ggml_cuda_op_moe_weighted_reduction`).
2. **Sparse per-token expert dispatch** — first cut landed on the engine
   branch `engine/moe-slot-cache`: per-step LRU ensure + slot mapping + H2D
   fetch (`GGML_OP_MOE_CACHE_MAP`), all routed experts computed on the GPU,
   `-msc auto` sizing from free VRAM, and full-coverage banks promoted once at
   load (identity, no per-step overhead, graph capture stays on). Measured warm
   (see the engine rig note `SLOT-CACHE-20260928.md`): 125B `-ncmoe 24 -msc 256`
   12.9 → 20.8 t/s (+61%); `-ncmoe 36 -msc 336` 4.4 → 8.2 t/s (+86%, 64.8% hit
   rate); 35B-A3B `-cmoe -msc 64` (2.5 GiB) 37.8 → 60.0 t/s, full coverage =
   all-GPU 108.9 t/s.
   Remaining for parity: a device-side ensure/gather to drop the per-layer
   host sync (~150 µs/layer, currently the biggest tok/s cost on cached arms),
   prefill double-buffered streaming (cached layers still fall back to CPU
   compute in prefill), the q* CPU/GPU split (only worth it once the CPU MoE
   path beats ~2× PCIe, per FreeToken's own criterion), and runtime resizing.
3. **Context curation (Strata side)** stays on the harness layer; stack
   profiles already shape per-tier context. `/v1/stacks/status` has schema
   fields for live `vram_gb` / `tok_s` per tier, but **nothing samples them
   yet** (they are always `null`) — filling them is part of the telemetry seam
   below, which should also surface the slot-cache hit rate.

### Measured end-to-end (2026-09-28 evening, this box)

First run of the engine through Hivebench (`amd-freetoken-125b` applied via
`POST /v1/stacks/amd-freetoken-125b/apply`):

| measurement | harness stack | engine rig (reference) |
|---|---|---|
| load-to-healthy | 41.9–42.9 s | 44 s |
| prefill (185-token prompt) | 446–527 t/s | 460–497 t/s |
| decode | 26.0–28.2 t/s | 24–28 t/s |
| VRAM used, headless / display | 19.49 / 18.39 GiB | 19.4 / 18.4 GiB |

- Process: `tools/backends/rocm/llama-server` with `HIP_VISIBLE_DEVICES=1,0`,
  argv `-ngl 99 -c 4096 --cache-type-k/v q8_0 --jinja --split-mode layer
  --tensor-split 0.522,0.478` (defaults: mmap, `--lazy-mode auto`, `-fa auto`,
  8 threads, 4 unified-KV slots).
- With `engine_args: ["--lazy-mode", "on", "-np", "1"]` the argv carries the
  rig's streaming/slot settings and the same benchmark reads 498–534 t/s
  prefill / 26.4–28.3 t/s decode (load 41.9 s) — i.e. the seam reproduces the
  rig config, defaults were already within a few percent.
- `/v1/stacks/status` reports the tier's `backend: "rocm"`, port, ctx, model,
  and the per-card plan; a chat completion round-trips
  (`"The capital of France is"` → reasoning content + completion).
- The Bonsai `amd-freetoken` stack cannot load on this engine (type 142, see
  above) — that needs the `moe-corr-runtime` PQ2_0 forward-port, not a
  harness change.

### Engine A/B vs the stock arm (2026-09-28)

`amd-freetoken-35b` (Qwen3.8-35B-A3B-IQ2_M, single card, MoE path) applied
through the harness twice, swapping only `tools/backends/rocm/llama-server`:

| arm | build | prefill (185 tok) | decode (128 tok) | greedy 64 tok |
|---|---|---:|---:|---|
| prism `engine/amd-rig` | 0.5.0-dev b11240 @2d1f9b104 | 1400–1445 t/s | 101.7–102.1 t/s | identical |
| stock `origin/master` | 0.3.0-dev b10686 @dc178a7cf | 1056 t/s | 91.4–91.6 t/s | identical |

- The prism build is ~+33–37% prefill / ~+11% decode on this model; both arms
  emit byte-identical greedy output (temperature 0, seed 42, 64 tokens), so the
  delta is kernel/engine work, not a sampling difference. The builds also
  differ by upstream drift (the engine branch merged current upstream), so the
  split between fork commits and upstream is not attributed here.
- The stock binary was built from `origin/master` in a worktree
  (`git -C ~/Desktop/work/prism-ml-llama.cpp worktree add ../prism-stock origin/master`,
  cmake flags as in `scripts/install_amd_engine.sh`).
- `experiments/stack_ab.py --live` cannot A/B these two arms: the harness holds
  at most one applied stack, and two 125B-class servers do not co-reside in
  40 GiB. The A/B above is therefore sequential (apply → bench → unload →
  swap binary → apply), same stack document and flags on both arms.
- The 125B dual-resident load has **no stock arm**: without the HSA bounce the
  dual-GPU load hangs (documented in the engine rig's `LOAD-HANG-RESULTS.md`);
  reproducing it risks driver state. Stock cannot serve the 125B on this box;
  the prism build does (T3 above).

## Verification

```sh
pytest tests/unit/test_stack_schema.py tests/integration/test_stack_manager.py -q
bash scripts/install_amd_engine.sh && tools/backends/rocm/llama-server --version

# the FreeToken smoke: apply the 125B stack and read the status back
.venv/bin/python -m harness --no-open --no-auto-start &
curl -sX POST localhost:8765/v1/stacks/amd-freetoken-125b/validate
curl -sX POST localhost:8765/v1/stacks/amd-freetoken-125b/apply
curl -s localhost:8765/v1/stacks/status
```
