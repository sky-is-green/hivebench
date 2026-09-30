# moe_tier — portable MoE placement planner for llama.cpp

Decides **how to place a MoE model** across the hardware you actually have:
which device(s) run the layers, how many expert layers must stay on the CPU,
how many threads the tail gets — and prints ready-to-run `llama-server` /
`llama-bench` commands. No fork, no vendor SDK, no OS-specific probing.

```
python moe_tier.py probe --engine-dir /path/to/llama.cpp/bin
python moe_tier.py plan  --model model.gguf --engine-dir /path/to/bin
python moe_tier.py plan  --model model.gguf --devices-json devices.json \
                         --gpus 0 --vram-budget-mib 2048  # simulate a small card
python moe_tier.py serve --model model.gguf --engine-dir /path/to/bin --run
```

## How it works (and why it is portable)

Two inputs, both engine-owned rather than OS-owned:

1. **Devices** come from `llama-bench --list-devices` — the same parseable
   output for ROCm, CUDA, Metal, Vulkan and CPU-only builds:
   `ROCm0: AMD Radeon RX 7900 XT (20464 MiB, 20404 MiB free)`.
2. **Model layout** comes from the GGUF header (metadata + tensor table, no
   weights read): per-layer expert-bank bytes, non-expert bytes, block count,
   KV geometry.

The planner then computes: `budget = VRAM - reserve - non-expert - KV`, picks
as many expert layers as fit (`-ncmoe` = the rest), and decides whether the
result fits one card (`--split-mode none --main-gpu N`) or needs a layer split.

Requirements: Python 3.9+ (stdlib only) and a llama.cpp recent enough to have
`--list-devices`, `--n-cpu-moe`/`-ncmoe` and `--override-tensor` (both are in
upstream llama.cpp). If a build is older, `plan --json` still gives you the
arithmetic; swap `-ncmoe N` for the equivalent `-ot` regex list.

## Decisions baked in (measured 2026-09-27, see `ternary-serve/placement-sweep-20260927`)

| Knob | Default | Evidence |
|---|---|---|
| Threads | physical cores | `-t 16` (SMT) collapsed MoE generation 95 → 35 t/s on a 7800X3D; `-t 8` = physical cores |
| Split | single GPU when weights+KV fit one card | 330 vs 188 t/s (2 GPU layer split) on a model that fits one card |
| Expert offload | minimum layers that make VRAM fit | full CPU offload costs −34% prefill / −50% generation (2.125 bpw ternary) |
| KV | f16, counted against VRAM | placement changed nothing in PPL: 533.0294 at `-ncmoe 0` and `16`; type quality (2026-09-29, release GGUF, 50×512, `-fa on`): **f16 7.4851 / q8_0 7.4824 / q4_0 7.5031 — q8_0 is quality-free, q4_0 costs +0.24% PPL for 4× less KV** |

## Notes / limits

- `-ncmoe N` keeps the **first N layers'** expert banks on CPU. Layers are
  usually similar in size; if they are not, use `plan --json` output and
  `-ot "blk\.<layer>\.ffn_.*_exps\.weight=CPU"` for an exact set.
- Sizes come from the GGUF tensor-offset table (≤ `general.alignment` padding
  per tensor — irrelevant for placement).
- KV estimate: `2 × kv_layers × kv_heads × head_dim × bytes × context`
  (`--kv-type f16|bf16|f32|q8_0|q4_0`; q8_0/q4_0 are approximate bpw).
  **Hybrid geometry (2026-09-30):** `kv_layers` comes from the GGUF's
  `attention.recurrent_layers` array — only non-recurrent layers carry
  context-growing KV (Qwen3.5/3.8: 10 of 40).  The 30 GDN layers carry a
  *fixed* f32 recurrent state, priced separately from the `ssm.*` metadata
  (`(key_dim + value_dim)·(d_conv−1) + d_state²·n_v_heads` per layer; 62 MiB
  for the released 35B).  Assuming all 40 layers carry KV overestimate the KV
  term ~4×.
- KV quality is priced from the measurement: `KV_QUALITY_DELTA_PCT` carries the
  release's 50×512 deltas (+0.000 / −0.036 / +0.240 % PPL for f16 / q8_0 /
  q4_0) into the plan notes.
- CPU expert compute cost grows with **bytes**, so the same placement is much
  cheaper on a 2 bpw container than on f16 (measured: −50% vs −75% generation
  under full offload).
- The planner targets a single context size; agent workloads that edit context
  (see FreeToken's semantic anchors) recompute differently — that is a serving
  layer question, not a placement one.
