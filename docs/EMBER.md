# Ember — AMD MoE expert residency for Hivebench

**Project codename: Ember.** Our own work on serving MoE models whose expert
working set does not fit VRAM on consumer AMD (2× RX 7900 XT 20 GB, ROCm), by
keeping a hot-expert slot band in VRAM and fetching the routed experts on
demand. Ember is an **independent implementation** in the prism ROCm engine and
the Hivebench harness; it is *inspired by* [FlashML
FreeToken](https://github.com/FlashML-org/FreeToken)'s published design (LRU
expert caching, bandwidth-adaptive co-execution) but shares no code with it and
targets a different runtime (llama.cpp/ggml, not their Python/Triton engine).
Formerly tracked as "AMD FreeToken".

> **Consolidated 2026-10-02:** Ember is a lane of the **hive_cascade** program.
> The standalone `ember` branch is retired (tip `4d07d81`, fully contained in
> `hive_cascade`); the engine branch **`engine/ember`** in the prism repo keeps
> its name. Anything Ember-marked lives on `hive_cascade` now.

## Status (2026-09-28, shelved)

- **Engine end-to-end through Hivebench** — the 125B Qwen3.8-Flash-Next MoE
  (Q2_0, 512 experts top-10, 48 layers) applies via the stack API on the prism
  ROCm binary; `backend: "rocm"` per tier; chat completions round-trip.
- **Engine A/B vs a stock build**: prism `engine/amd-rig` vs `origin/master` on
  the 35B MoE — +33–37% prefill, +11% decode, byte-identical greedy output.
- **Ember slot cache** (`--moe-slot-cache N` / `-msc auto`): host-resident
  expert banks keep a hot-expert slot band in VRAM; decode fetches misses over
  PCIe and runs every routed expert on the GPU; fully-covered banks are
  promoted once at load (identity slots, no per-step overhead, graph capture
  stays on). Steady-state 125B measurements (256-token decode, 185-token
  prompt, ctx 4096, headless-first split):

  | config | experts off-VRAM | slot cache | decode | vs no-cache | hit rate |
  |---|---:|---:|---:|---:|---:|
  | all-resident | 0 | – | 26.9 t/s | – | – |
  | `-ncmoe 12` | 8.1 GiB | – | 19.3 | – | – |
  | `-ncmoe 12 -msc 256` | 8.1 GiB | 4.1 GiB | **24.5** | **+27%** | 87.9% |
  | `-ncmoe 24` | 16.2 GiB | – | 14.5 | – | – |
  | `-ncmoe 24 -msc 256` | 16.2 GiB | 8.1 GiB | **23.0** | **+59%** | 88.6% |
  | `-ncmoe 36` | 24.3 GiB | – | 3.9 | – | – |
  | `-ncmoe 36 -msc 336/448` | 24.3 GiB | 15.9–21.2 GiB | **11.5–15.4** | **+200–300%** | 89.7% |

  35B-A3B one card: 37.8 → **60.0 t/s** from 2.5 GiB of slots (75.9% hits);
  full coverage = all-GPU **108.9 t/s** (identity promotion, token-identical).
  Harness stack `stacks/ember-125b-lowvram.json` (`-ncmoe 12 -msc auto`) picks
  its own size (449 slots / 7.1 GiB at apply time) and serves warm
  268–307 t/s prefill / 20.6–27.4 t/s decode.
- **Shelved with these open items** (see "Next steps" below): kernel-count
  reduction, telemetry, prefill streaming for cached layers, device-side
  ensure.

## Why it works (measured diagnosis)

- Steady-state hit rate is **~88–90%** (cold-start runs understate it); at
  ≤16 GiB of offload the misses are page-cache served (~50–70 MiB disk per
  256-token generation) and decode is within **9–15%** of all-resident.
- That residual gap is the **per-layer host round trip + lost CUDA-graph
  replay** (~4–6 ms/step at 24 cached layers): the map op syncs the stream, so
  the graph containing it cannot be captured.
- At 24 GiB of offload the **page cache thrashes** (1.6–7.4 GiB disk reads per
  generation) and decode becomes disk-sensitive (11.5–15.4 t/s), still 3–4×
  the no-cache arm.
- A GPU kernel **cannot read unpinned mmap memory** on this box ("Page not
  present"), so FreeToken's device-side gather needs pinned banks:
  `hipHostMalloc(8 GiB)` works and reads at ~12.5 GB/s (half the ~25 GB/s DMA
  copy rate). 30 GB RAM caps how much of the expert set can ever be pinned.
- All-resident is **dispatch-bound** (~5,000 kernels/step × ~3.5 µs per
  `tools/engine-rig/notes/BUDGET-20260928.md`) — the biggest untouched tok/s
  lever for every configuration.

## The engine

Repo: `sky-is-green/prism-ml-llama.cpp`, **branch `engine/ember`** (based on
`engine/amd-rig`; local checkout `~/Desktop/work/prism-ml-llama.cpp`, built in
`build-hip`).

- upstream llama.cpp master (has `qwen4exp` / Qwen3.8-Flash-Next and
  `--lazy-mode` PLE streaming) + the HSA upload fix (`set_tensor` staging;
  prevents the ROCm dual-GPU load hang) + `LLAMA_GRAPH_DUMP` + Ember.
- Ember pieces: `GGML_OP_MOE_CACHE_MAP` (+ `ggml/include/ggml-moe-cache.h`),
  `src/llama-moe-slot-cache.{h,cpp}`, the `build_moe_ffn` hook,
  `--moe-slot-cache N|auto`, `LLAMA_MOE_SLOT_STATS=1` hit/fetch counters.
- Research record: `tools/engine-rig/notes/EMBER-20260928.md` (design,
  measurements, FreeToken alignment); budget notes and rig under
  `tools/engine-rig/`.

Hivebench integrates it through the per-backend binary convention
(`harness/models.py::_binary_for_backend`):

```
tools/backends/rocm/llama-server      # the prism build (scripts/install_amd_engine.sh)
stacks/ember*.json                    # "backend": "rocm" tiers
```

## Install / use

```sh
# build/symlink the engine into tools/backends/rocm/  (~2 s if the engine is built)
scripts/install_amd_engine.sh            # or: --copy, SKIP_BUILD=1, PRISM_DIR=...
tools/backends/rocm/llama-server --version

# serve the 125B FreeToken-shape stack
.venv/bin/python -m harness --no-open --no-auto-start &      # sidecar on :8765
curl -sX POST localhost:8765/v1/stacks/ember-125b/validate
curl -sX POST localhost:8765/v1/stacks/ember-125b/apply
curl -s localhost:8765/v1/stacks/status

# the offload+cache variant (12 layers' experts host-side, cache auto-sized)
curl -sX POST localhost:8765/v1/stacks/ember-125b-lowvram/apply
```

Stacks in this branch:

| stack | what | status |
|---|---|---|
| `ember-125b` | 125B MoE all-resident (headless-first split) | works |
| `ember-125b-lowvram` | same + `-ncmoe 12 -msc auto` | works |
| `ember-35b` | 35B-A3B single-card smoke / A/B model | works |
| `ember` | Bonsai-2 27B/1.7B (PQ2_0) | **cannot load** on `engine/amd-rig`: fork-private tensor type 142 lives only on the legacy `moe-corr-runtime` line; keep for a future forward-port |

Model library: the GGUFs are symlinked into `~/.lmstudio/models` (the harness's
default `models_dir` on this box): the 125B in
`~/Desktop/work/models/qwen38-q2_0/Q2_0/`, the 35B in
`~/Desktop/work/hivebench/artifacts/ternary/refs/qwen35/`.

## Toolchain fixes that made this possible (hivebench)

- `harness/models.py` resolved the repo root one level too high (left over from
  the flat-package move): `tools/backends/rocm/llama-server` was never found and
  backend selection always refused.
- `harness/stack/manager.py` emitted `--ts`, which no llama.cpp understands
  (`-ts` / `--tensor-split` is the real flag); every stack with a `ts` died at
  startup.
- `engine_args` seam: a validated whitelist (`ENGINE_ARGS_ALLOWED`) so a stack
  can carry `-ncmoe`, `-msc`, `--lazy-mode`, `-np`, etc. without owning the
  model/host/port flags.

## Strata — external reference point (2026-10-02)

[Strata](https://github.com/Niko1221/Strata) (MIT, ~5.7k stars, trending
September 2026) runs the same Qwen3.8-Flash-Next Q2_0 pack on consumer GPUs
with an overlapping design: an adaptive hot-expert cache in VRAM and all
experts in RAM, but **off-VRAM experts are computed by the CPU in place**
(AVX-512, concurrent with the GPU) instead of fetched over PCIe — plus the
model's own MTP speculation (2.4-3.2 accepted tokens/pass), 8K prefill chunks
with layer-ahead expert prefetch, KV streaming and k8v4, mapped/resident
low-RAM modes, and a full OpenAI + Anthropic serving surface with per-tier
`/metrics` and per-box calibration.

Published Q2_0 numbers on a 12 GB RTX 5070 / 64 GB PC: 87.3 t/s at 1K context,
73.7 at 128K, 60.3 at 262K. Our all-resident 125B measurement is 26.9 t/s; the
gap is speculation (1.6-1.8x alone), CPU co-execution, prefill chunking, and
the dispatch floor noted below. Their design validates Ember's thesis; the
unshelve decision is open. Full comparison and borrow list:
`docs/CASCADE-PILOT.md` ("Strata — the local Flash-Next engine").

### MTP speculation: what it actually takes (2026-10-02)

The GSQ-RCO GGUF **ships no MTP head**. The 31 `mtp.*` tensors (fc_embedding,
hyper-connection mixers, one MTP layer with its own attention/indexer/experts,
pre-fc norms) live only in the BF16 checkpoint `Qwen/Qwen3.8-Flash-Next`
(~360 GB, 131 shards), spread over 28 shards. Strata's `tools/mtp_fetch.py`
reads the safetensors headers and range-fetches exactly those tensors
(SHA256-pinned revision `de4b8e4d…`), ~6 GB. So this gap-closer is not a flag:
it needs (a) an equivalent range-fetch/pack step for the MTP block and (b) an
MTP draft path in the engine — our fork's `llama-mtp-sidecar` is a different
(Scion) head architecture. Strata's MIT implementation is the reference;
porting it with attribution, or adopting Strata as the Flash-Next serving
engine, are the two paths. Decide before the baseline run.

Model count: HF GGUF metadata reports **~176.9B tensor elements** (so the
handoff's ~180B is right; Strata's "125B" is not the count to use). Corrected
in the cascade registry (`flash-next`: 177B, Q2_0, 66 GB).

## E4 baseline protocol (ready to run — box free 2026-10-02)

Goal: Flash-Next on AMD at Strata-level performance, measured before any
cascade changes. Ask the human first (one heavy track at a time; 30 GB RAM).

Preconditions: no other heavy job; ≥26 GB RAM free; driver state clean.
Apply through the stack API (`ember-125b-lowvram`) or directly:

```sh
LLAMA_MOE_SLOT_STATS=1 tools/backends/rocm/llama-server \
  -m ~/Desktop/work/models/qwen38-q2_0/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
  -ngl 99 -ncmoe 12 -msc auto -c 4096 -np 1 -t 8 \
  -ts 0.522,0.478 --split-mode layer -lm mmap --lazy-mode on -fa auto
```

Runs, in order (record every one under `experiments/cascade/results/` with
model sha + engine commit + flags):

1. all-resident baseline (`-ncmoe 0`), then `-ncmoe 12 -msc auto`, then
   `-ncmoe 24 -msc 256`;
2. per run: cold prefill (1K/4K prompt) and warm prefill, 256-token decode,
   `expert tiers` hit rates, host-RAM peak, VRAM peak, disk MB per generation;
3. optional reference: Strata's AMD/HIP build on the same box, same prompts —
   this is the "as well as Strata" comparison;
4. abort criteria: free RAM < 4 GB during load, OOM, driver reset.

Then the gap-closing ladder, in value order: MTP layer (fetch/pack, then
`--model-draft`/`--spec-draft-max` through the engine-args seam), CPU
co-execution of misses, prefill chunking + layer-ahead prefetch, KV
streaming/k8v4, dispatch-floor reduction, telemetry/calibration.

## Next steps (in value order, when unshelved)

1. **Kernel-count reduction / fusion** (all configs, biggest headroom). The
   budget note's candidates are still unfused; targets the ~5,000
   kernels/step dispatch floor. Expected to lift the 27 t/s all-resident and
   the 23–24 t/s cached arms together.
2. **Telemetry + tuning seam**: expose Ember hit/fetch rate and per-tier
   `tok_s`/`vram_gb` through `/metrics` and Hivebench `/v1/stacks/status`
   (currently `null`), so `-msc auto` and offload choices are validated by the
   harness instead of by hand.
3. **Prefill for cached layers**: pp 125–186 first-request vs 224 all-resident
   (warm harness 268–307); needed for long agent contexts, and it is the same
   bytes-over-disk problem.
4. **Device-side ensure + pinned gather** (recovers the 9–15% gap and
   re-enables graph capture): needs pinned banks (8 GiB works; 16 GiB risky on
   30 GB RAM) and the gather reads at ~12.5 GB/s. Only worth it if cached must
   equal all-resident exactly.

## Verification

```sh
pytest tests/unit/test_stack_schema.py tests/integration/test_stack_manager.py -q
tools/backends/rocm/llama-server --version

# 125B steady-state smoke (256-token decode; expect ~19–25 t/s depending on offload)
LLAMA_MOE_SLOT_STATS=1 tools/backends/rocm/llama-server \
  -m ~/Desktop/work/models/qwen38-q2_0/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
  -ngl 99 -ncmoe 24 -msc 256 -c 4096 -np 1 -t 8 \
  -ts 0.522,0.478 --split-mode layer -lm mmap --lazy-mode on -fa auto
```

- Engine A/B setup: stock binary built from `origin/master` in
  `~/Desktop/work/prism-stock` (worktree), same cmake flags as the installer.
- `experiments/stack_ab.py --live` cannot A/B 125B-class arms (one applied
  stack at a time; two servers do not co-reside in 40 GiB) — the A/B is
  sequential: apply → bench → unload → swap binary → apply.
- The 125B has no stock arm: without the HSA bounce the dual-GPU load hangs
  (`LOAD-HANG-RESULTS.md`); not reproduced to protect driver state.
