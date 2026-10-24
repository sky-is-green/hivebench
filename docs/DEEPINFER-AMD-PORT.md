# DeepInfer on AMD (ROCm) — the port

DeepInfer is DeepSeek's own inference framework ([deepseek-ai/DeepInfer],
Apache-2.0). It is vendored at `vendor/deepinfer/` (upstream commit
`774ce244b87c`, re-vendored by `scripts/revendor_deepinfer.py`) and the AMD
lane is the overlay at `vendor/deepinfer_amd/`.

This page is the port's record: what runs on AMD today, what does not, and
why — with the numbers and the hardware they were measured on.

> **The box these numbers come from:** 2 × AMD Radeon RX 7900 XT
> (Navi 31, **gfx1100**, 42 CUs, 20 GB each), 30 GB system RAM, ROCm
> **7.2.4** at `/opt/rocm`, Linux, no NVIDIA device present. The ROCm
> PyTorch is **2.11.0+rocm7.2** (HIP 7.2.26015). DeepInfer's C++ extension
> is built with `hipcc` from the same ROCm.

## The short version

DeepInfer as released targets **NVIDIA SM100** (CUDA 13) and **Huawei
Ascend 950** (CANN 9.2). Its heavy kernels come from five vendored
repositories — DeepGEMM, FlashMLA, DeepSelect, DeepEP, DeepGEMM-Ascend —
and there is no ROCm build of any of them.

So the port has two halves, and they are not equal:

* **The engine is portable.** The C++ core (the scheduler, the sequence and
  output plumbing, the block-load planner, the sampler driver, the graph
  timer) is not CUDA code. It builds with `hipcc` unchanged apart from two
  device-runtime files. **This half is done.**
* **The kernels are not.** DeepInfer's kernels are TileLang/DeepGEMM
  kernels for Blackwell, with FP8 weights, FP4 experts, MLA over an FP8 KV
  cache, and the DeepSeek-V4.1-specific mHC head and Engram n-gram tables.

And the model decides the rest:

> **DeepSeek-V4.1-Flash cannot run on gfx1100, at any effort level.** Its
> checkpoint is FP8 weights with FP4 experts (`quantization_config:
> quant_method "fp8", expert_dtype "fp4"`), and gfx1100 (RDNA3) has **no
> FP8 hardware at all** — not in the matrix units, not in the conversion
> path. RDNA4 (gfx1200) has FP8; CDNA3 (gfx942) has FP8 and FP4.

So the AMD lane's target is **a bf16 model without the mHC head and
without Engram** — which is exactly what our own ternary models are, and
what the harness needs.

### Why this is worth doing at all

Because DeepSeek-V4.1 turns out to be **the closest published reference
implementation of our own model family**. Feature for feature, our
Qwen4Exp model and DeepSeek-V4.1 have the same ideas under different
names:

| DeepSeek-V4.1 | ours (Qwen4Exp) |
|---|---|
| the mHC head, `hc_mult` **4** | `hc_count` **4** — hyper-connections |
| the Engram n-gram tables | `ngram_size` **3** — the n-gram cache |
| the index attention (`index_topk`, candidate blocks) | the attention indexer, budget 2048 |
| DFlash / `num_nextn_predict_layers` | the MTP layer |
| MXFP8 weights, 128×128 blocks | MXFP8 weights, 128×128 blocks |
| an MoE with a shared expert and a `sqrtsoftplus` router | 512 experts, top-10, shared expert |

So the port is not "run someone else's engine on our hardware" -- it is
**"port the reference implementation of our own architecture to our
hardware"**, and DeepInfer is the reference. The kernels we have to write
for gfx1100 are the kernels *our model needs*, and the shape of the model
is already known.


## What runs today

| Piece | State | Notes |
|---|---|---|
| `deepinfer._C` (the C++ engine) | **builds and loads** | `vendor/deepinfer_amd/build/build_amd.py`; two device-runtime files replaced with HIP versions (`vendor/deepinfer_amd/csrc/`) |
| the platform layer | **works** | `deepinfer.PLATFORM` is `PlatformAMD`: `is_gpu` False, `is_npu` False, `is_amd` True |
| the device discovery | **works** | 2 GPUs, `gfx1100`, 42 CUs, no FP8 — read from `/sys/class/kfd` |
| the V4.1 model executor | **imports** | `deepinfer.model_executor.models.deepseek_vl` imports clean |
| `LLMEngine`, `CacheEngine` | **import** | the whole engine initialises on AMD |
| the TileLang-op dispatcher | **works** | the `amd` backend is selected; `norm_tl` and `gather_tl` are implemented in torch |
| `deep_gemm`, `tile_kernels`, `flash_mla`, `deep_ep`, `deep_select`, `flash_attn`, `tilelang` | **replaced** | torch-native versions under `vendor/deepinfer_amd/ops/` |
| the OpenAI-compatible facade | **works** | `tools/deepinfer/serve.py`; `/health`, `/v1/models`, `/v1/chat/completions`, `/v1/completions` |
| the harness integration | **works** | `launch.kind: "deepinfer"`, `harness/stack/deepinfer.py`, `stacks/deepinfer-v41.json` |

### The verified import

```console
$ PYTHONPATH=vendor:vendor/deepinfer python -c "import deepinfer; print(deepinfer.PLATFORM)"
<deepinfer_amd.platform.PlatformAMD object at 0x...>
```

### The build

```console
$ ROCM_HOME=/opt/rocm python vendor/deepinfer_amd/build/build_amd.py
deepinfer._C built for amd-x86_64 in vendor/deepinfer/deepinfer/_so/amd-x86_64
```

ROCm's PyTorch hipifies the sources for us (its `cpp_extension` runs
`hipify` over every `.cpp`/`.cu` when `torch.version.hip` is set), and the
hipify pass reports **0 unsupported CUDA function calls** over the
upstream tree. Two files are compiled from the overlay instead, because
hipify's map misses a few event flags: `csrc/amd_utils.cpp` and
`csrc/graph_timer_amd.cpp`.

## The platform decision

Upstream has two platforms and the code branches on `PLATFORM.is_gpu` and
`PLATFORM.is_npu`. AMD is **neither**, and the port models it that way:

```python
class PlatformAMD:
    enum = AmdPlatformEnum.AMD
    tilelang_ops_backend = "amd"
    distributed_backend = "nccl"      # ROCm's PyTorch exposes RCCL as "nccl"
    is_gpu = False                     # <- the important one
    is_npu = False
    is_amd = True
```

`is_gpu = False` is the load-bearing choice. Upstream uses `is_gpu` to
mean *"NVIDIA, with all the Blackwell-only fast paths"*: the FP8 einsum
output, the mega gate, the fused shared experts, the NVLink userbuffers,
the CUDA-graph probes, the eager `set_pdl`. None of those exist on RDNA3,
so they must all be **off**. Leaving `is_gpu` True would silently switch
on FP8 and mega-kernel paths that cannot work.

What `is_gpu = False` buys, concretely: the MoE takes the
`tile_kernels.moe.moe_topk_gate` branch instead of the SM100 mega gate;
the attention skips the fused norm+rope+attention+FP8-cast kernel; the
expert-parallel path skips the NVLink symmetric buffer.

What had to be patched in the vendored tree: **one file, two hunks**
(`vendor/deepinfer_amd/patches/0001-amd-platform-hook.patch`):

* `deepinfer/__init__.py` calls `deepinfer_amd.install()` before anything
  reads the platform, and accepts `"amd"` as a platform tag.

The rest of the port lives entirely in the overlay, registered at install
time:

| Upstream seam | What the overlay does |
|---|---|
| `deep_cluster.hardware.accelerator_kind()` | patched to return `"amd"` (upstream would return `"cpu"`, which makes `get_bf16_tflops_limit()` raise) |
| `deep_cluster.hardware.visible_device_env_name()` | patched to `HIP_VISIBLE_DEVICES` (upstream returns `None` on a ROCm host) |
| `deepinfer.platform.get_platform()` | patched to return `PlatformAMD` |
| `import deep_gemm` / `tile_kernels` / `flash_mla` / `flash_ep` / `deep_select` / `flash_attn` / `tilelang` | registered in `sys.modules` as the torch-native replacements |
| `deepinfer...tilelang_ops.<backend>` | registered as the `amd` op backend |

## What is refused, and why

Every refusal carries a reason, so a mis-configured run fails at load time
with a message instead of producing wrong numbers.

| Refused | Why |
|---|---|
| every `deep_gemm` FP8/FP4 entry point | gfx1100 has no FP8; RDNA4 has FP8 but no FP4 |
| every `tile_kernels.quant.per_token_cast*` | same |
| `flash_mla.flash_mla_sparse_fwd` / `flash_mla_with_kvcache` | they read DeepSeek's FP8-quantised KV cache, which gfx1100 cannot produce |
| `flash_mla.fused_norm_rope_attn_rope_cast.prefill/.decode` | they fuse the FP8 activation cast into the attention |
| the mHC group-RMSNorm epilogue and the fused mHC inference path | their exact normalisation is defined by DeepSeek's TileLang kernel, which is not part of this release — and the model that uses them needs FP8 anyway |
| `tile_kernels.engram.*` | the Engram n-gram tables are not ported (a DeepSeek-V4.1 feature) |
| `deep_ep.EPBuffer.dispatch/.combine` | single-rank expert parallelism: the dispatch is a local copy, not a comms round trip |
| `deep_gemm.SymmBuffer` | NVLink symmetric buffers |

The mHC refusal is the one worth stating plainly: **the V4.1 model is out
of scope for gfx1100 for hardware reasons**, and the refusal says so
rather than hiding it.

## The remaining work, in order

### 1. The bf16 TileLang-op backend (the critical path)

Upstream's `deepinfer/model_executor/layers/tilelang_ops/cuda/` holds 30
modules and about 50 kernel factories. Each is a TileLang kernel; the
AMD backend replaces them with torch.

**Done:** `norm_tl` (RMSNorm with the fused residual add), `gather_tl`
(the row gather).

**To do, in the order a bf16 text model needs them:**

| Kernel | What it computes | Difficulty |
|---|---|---|
| `store_kv_cache_tl` / `load_paged_kv_cache_tl` | the paged KV cache | medium — upstream ships `*_ref` reference implementations to copy the semantics from |
| `fused_sample_tl` / `random_sampling_tl` | the sampler | medium — upstream ships `fused_sample_ref` |
| `sum_smaller_probs_tl` | the top-p mass | easy |
| `map_lookup_tl`, `expand_sampling_params_tl`, `remap_draft_topk_tl` | the sampler plumbing | easy |
| `get_accept_result_tl`, `finalize_verification_tl`, `prepare_verification_metadata_tl` | the MTP verification | medium |
| `get_kv_index_tl`, `prepare_v4_decode_metadata_tl`, `copy_block_tables_tl` | the decode metadata | medium |
| `state_cache_tl` | the SWA state | medium |
| `swa_index_tl` | the sliding-window index | easy |
| `structure_output` | the xgrammar bitmask | medium |
| `compute_seqlens_q_by_confidence_rates_tl` | the DSpark confidence schedule | medium |
| the DFlash / candidate-block / compress set | DeepSeek-V4.1-specific | only if the V4.1 model is in scope |

The good news: upstream ships a **pure-torch reference implementation**
(`*_ref`) for nearly every one of these, in the dispatcher module next to
the kernel. The AMD backend can be built by wrapping those references in
the `*_tl` factory protocol, which is exactly what upstream's own tests
compare the CUDA kernels against.

### 2. Our model

DeepInfer's model executor is `DeepseekV41ForCausalLM`, and its config
loader is written around the V4.1 HF schema (`model_type:
"deepseek_v41"`, a `vl_aligner` block, `compress_ratios`, `hc_*`,
`dspark_*`, `engram_*`).

**Our model is `Qwen4ExpForConditionalGeneration`** (`model_type:
"qwen4_exp"`), from the HF checkpoint at
`models/qwen38-flashnext-fp8/`. Its `text_config`, read off the
checkpoint:

| | |
|---|---|
| layers | 48: **36 linear attention** (gated delta net) + **12 full attention** (`full_attention_interval: 4`) |
| hidden | 2560, 24 attention heads, 2 KV heads, head_dim 256, partial rotary 0.25 |
| MoE | **512 routed experts, top-10**, shared expert intermediate 640, routed intermediate 640 |
| linear attention | 16 key heads × 128, 48 value heads × 128, conv kernel 4 |
| hyper-connections | `hc_count` **4**, `hc_lowrank` 320 (`attn_hyper_connection`, `mlp_hyper_connection`) |
| n-gram | `ngram_size` **3** |
| attention indexer | budget **2048**, 4 heads, 1 KV head, compress ratio 4 |
| MTP | 1 layer |
| vision | 27 blocks, hidden 1152, 16 heads |
| quantisation | **MXFP8**, 128×128 weight blocks |
| on disk | **64 GB** (131 shards) |

#### The features line up

This is the finding that makes the port worth doing: **DeepSeek-V4.1 is
the closest published reference implementation of our own model family.**
Feature for feature:

| DeepSeek-V4.1 | ours | the same idea |
|---|---|---|
| the mHC head (`hc_mult` **4**) | `hc_count` **4** | hyper-connections: a trunk of residual paths with learned mixes |
| the Engram n-gram tables | `ngram_size` **3** | a learned n-gram cache beside the token embedding |
| the index attention (`index_topk`, `candidate_blocks`) | the attention indexer (budget 2048) | sparse attention over a selected subset of the KV |
| DFlash / `num_nextn_predict_layers` | the MTP layer | multi-token prediction as a speculative drafter |
| the MoE with a shared expert, `sqrtsoftplus` scoring | 512 experts, top-10, shared expert | the same router shape |
| FP8 weights with block scale factors | MXFP8, 128×128 blocks | the same weight format |

So "our models on DeepInfer" is not a foreign-architecture port: it is
**porting the reference implementation of our own architecture to AMD**, and
the AMD lane's job is to make that implementation run on our hardware.

#### The weight format is the blocker

DeepInfer reads **HF safetensors**. Ours are **GGUF** — Q2_0, Q6_K, and
our ternary PTQ1_0 at 1.75 bpw (bit-identical to `llama-tq`'s reference
quantiser, per the Strata fork's own parity test).

The arithmetic decides it:

| checkpoint | size | fits this box? |
|---|---|---|
| the HF MXFP8 checkpoint | **64 GB** | no (30 GB RAM, 40 GB VRAM) |
| the same weights dequantised to bf16 | ~128 GB | no |
| the ternary GGUF (`Scion-35B-A3B-PQ2_0-corr.gguf`) | **10.6 GB** | yes |

So the AMD lane must load **GGUF**, and the MXFP8 path is not an option on
gfx1100 anyway (no FP8 hardware).

Three pieces:

1. a **model executor** for the Qwen4Exp architecture, registered through
   `ModelRegistry` (which is a plain dict, so the overlay can add to it);
2. a **config loader** for the `qwen4_exp` HF schema (upstream's
   `load_hf_model_config` rejects anything that is not `deepseek_v41`);
3. a **GGUF weight loader** that hands the dequantised weights to torch.
   The ternary path is the interesting one: PTQ1_0 is 1.75 bpw with a
   group-128 scale, and the dequantisation must be **bit-exact with what
   Strata loads today** — Strata's own `ptq1_0_parity.cpp` is the
   reference, and our loader can reuse its layout (group-128, the same
   scale convention) so the two engines agree token for token.

The good news: Strata's fork already has the ternary kernels
(`src/kernels/cpu/ptq1_0.cpp`, the CUDA `iq_kernels.cu` path, and the
half-block PTQ1_0 dot behind `-DSTRATA_PTQ1_QUARTER`). The AMD lane needs
the same numerics in torch, not a new format.


### 3. The measurement protocol

DeepInfer ships golden PPL baselines
(`tests/models/offline_decode_ppl.json`) for the NVIDIA and Ascend
platforms. The AMD lane should add its own, so that a change to a kernel
shows up as a PPL delta rather than as a silent drift.

For our own models the protocol is the one hivebench already runs: the
paired A/B (`experiments/paired_ab.py`), which compares two engines
token-for-token on the same prompts.

## The harness side

DeepInfer ships **no HTTP server**. It is an offline `LLM.generate()` API
plus a distributed launcher, and its README is explicit that the KV-cache
system, the prefill/decode disaggregation and the load balancing are not
part of the release.

The harness speaks HTTP — every tier in a stack is an OpenAI-compatible
endpoint, and the dsh routes are OpenAI chat completions — so the fold-in
needed one more piece: **`tools/deepinfer/serve.py`**, a facade that wraps
the offline API in `/health`, `/v1/models`, `/v1/chat/completions` and
`/v1/completions`, with a batching worker (the engine is blocking and
batch-oriented).

With that in place, DeepInfer is a tier like any other:

```json
"launch": {
  "kind": "deepinfer",
  "repo": "~/Desktop/work/hivebench",
  "config": "configs/deepinfer-v41.json",
  "python": ".venv-amd/bin/python",
  "engine_args": [],
  "timeout_s": 1800
}
```

and `harness/stack/deepinfer.py` owns the lifecycle the same way
`harness/stack/strata.py` does: an endpoint that answers is **adopted**
(never killed), one that does not is **started** from a generated copy of
the config and stopped again on unload.

`DeepInferLauncher.capabilities()` is the honest gate: it reports the
accelerator, the gfx architecture, the device count, whether the hardware
has FP8, and whether the C++ extension is built — so a host that cannot
run the requested model gets a refusal with a reason.

## What is *not* in this port

* **Multi-GPU expert parallelism.** DeepEP is NVLink/RDMA; consumer AMD
  has neither. The AMD lane is single-rank, which is also what our
  harness's stacks use.
* **The FP8 and FP4 paths.** Not on RDNA3.
* **DeepSeek-V4.1-Flash itself.** It needs FP8 weights, FP4 experts, the
  mHC head, Engram and MLA over an FP8 KV cache.
* **The distributed launcher** (`deep_cluster`, `examples/_launch.py`). It
  is built around `torchrun` and NCCL/RCCL groups; single-node only for
  now.

[deepseek-ai/DeepInfer]: https://github.com/deepseek-ai/DeepInfer
