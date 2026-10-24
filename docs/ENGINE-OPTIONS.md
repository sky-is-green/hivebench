# Engine options for our models on the AMD box

Written 2026-10-21, after `github.com/mw00/project-maya` turned up. This is the
re-scope question: which engine should carry "our models on our AMD hardware",
and what is DeepInfer for if not that.

Everything below is read off the code in the trees, not from the READMEs'
claims. Where something is unverified it says so.

## The decision

**Maya is not a dependency.** What we needed from it - the ability to run
frontier models on consumer hardware - is already in *our* engine, and was
already measured on this hardware. The rest of this document is the evidence
for that, and the record of what the measurements changed.

| | what carries it | state |
|---|---|---|
| our models (Qwen3.8-Flash-Next, Scion-35B) | **Strata** (`work/strata`) | running: 20.52 tok/s on 2 x 7900 XT, identical tokens to Maya |
| frontier models (DeepSeek-V4.1-Flash) | **Strata** (`work/strata-ds41`, branch `ds41-merge`) | engine merged and built for gfx1100 on this box; the weights are being fetched |
| a second opinion on V4.1 | **DeepInfer** (the vendored upstream engine) | AMD port in progress; it is the only copy of DeepSeek's own implementation, so it is the check on the other two |
| project-maya | a **reference**, not a dependency | read for the tiering and the AMD expert kernels; its pack was deleted, the checkout kept |

The one thing Maya gave us that we keep: the *measurement* that our own
storage layout, not our engines, was the dominant cost - see "Why it was
slow" below. That finding changed every number in this document, and it
would have been found the hard way anyway.

## The frontier lane: DeepSeek-V4.1 on Strata

This is what Maya was worth looking at for, and the answer was already ours.

`work/strata-ds41` (branch `ds41-merge`, a worktree of `work/strata`) carries
a **DeepSeek-V4.1-Flash engine**: 40 layers x 384 routed experts, top-6,
Engram, CSA/indexer attention, mHC, DSpark speculative decoding, and the
expert tier (VRAM cache -> pinned RAM arena -> file tier).

- The AMD port is **already merged**: the `dsv41/amd-gfx1100` branch is
  *fully contained* in `ds41-merge` (0 commits ahead, 1001 behind), and
  `docs/ds4/AMD_HIP_NOTES.md` records the bring-up.
- It is **built and green on this box**: `build-hip/ds41_generate` and
  `strata 0.1.44`, configured with
  `-DSTRATA_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100
  -DSTRATA_PREFILL_MMQ=ON -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_NATIVE_EXPERTS=ON`.
- It has **already been measured on a 7900 XTX** (the AMD notes): 12.6 tok/s
  decode, 14.2 tok/s prompt, with the expert tier at ~78% of a decode step.
  A two-card peer tier was built and measured and **lost ~5%** - one card is
  the configuration.
- The branch is parked on purpose, waiting for the weights:
  *"the V4.1 552B GSQ fetch is deferred until a job actually needs it"*. The
  job is here now.

**The weights.** The model is our own GSQ mixed-tier quant (2.25 bpw),
`IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored` on Hugging Face: the
`V41-GSQ.gguf` arrives as 16 x 10 GiB parts (163 GB) and the two Engram
tables come from the FP8 release's shards 47 and 48 (94.56 GiB each). Both
repos are public. `local/fetch-model.sh` does the whole thing - parts, the
sha256-verified join, the Engram extraction, the merged index.

**The disk, which is the real constraint.** The run-time payload is 365 GB
(163 GB of GSQ + 101 GB per Engram table) and this box has 422 GB free across
four SSDs - but not in one place, and the join needs the parts beside the file
it writes. So:

| | where | why |
|---|---|---|
| the GSQ parts + the joined GGUF | the FUSE HDD (800 GB free) | the join is in-directory, and the file is read sequentially at load; the file tier pays for it (below) |
| the FP8 shards (transient) | the FUSE HDD | read once, sequentially, to extract the tables |
| Engram layer 1 (101 GB) | `/var/tmp` (nvme1n1p2, SSD) | 48 random 90-byte rows per token |
| Engram layer 14 (101 GB) | the work drive (nvme0n1p5, SSD) | same |

The Engram tables must be on an SSD (48 random reads a token; 7.5 ms a seek
on the HDD would be 360 ms a token). The GSQ is the one that could move: its
file tier reads 1.38 MB blobs, so the HDD costs a seek plus the transfer
(~17 ms a read against ~1.5 ms on an SSD). If the decode number comes out
much worse than the AMD notes' 12.6 tok/s, that is the first thing to fix -
and the fix is disk, not code.

**The lane in the harness.** `stacks/ds41-strata.json` plus
`strata-ds41/local/ds41-hivebench.json`, driven by the existing `strata`
launcher: Maya's `serve/server.py` is Strata's, so the launcher is the same
one the qwen4exp lane uses. The one difference is the context flag - the
V4.1 engine takes `--ctx` and refuses any flag it does not know, so the
launch block names it (`"ctx_flag": "--ctx"`), which is why
`harness/stack/strata.py` now takes `ctx_flag` and the schema validates it.

The box's numbers differ from the reference box (2 x 7900 **XT** 20 GB + 30 GB
RAM against 2 x 7900 **XTX** 24 GB + 128 GB), so the lane is configured for
ours: `--arena-gib 12` (the RAM tier; the recipe says 12-16 GiB for this
box), `--prompt-batch 2048` (against their 12288), `--threads 12`, `--ctx
32768`, one card.

## Two cards: the weights in two places

The V4.1 model does not fit on one card, and the reference box measured the
expert tier at **78% of a decode step** (`dense 99.9 ms, tier 365.1 ms` per
pass). So where the weights live is the difference between a frontier model
that runs and one that does not. Three designs are on the table, and two of
them are already ours.

**1. The layer split - our own qwen4exp path, measured on this box.**
`--layer-split auto` puts contiguous layer ranges on each card; each card
keeps an expert cache for **its own layers only**, so two cards hold about
twice the experts one card holds. A token crosses from one card to the next
**once per verify window** (a few hundred KB through pinned RAM), not twice
per layer.

The measurement in `docs/MULTI_GPU.md` is on *this machine* - 2x RX 7900 XT
20 GB, Ryzen 7 7800X3D, 30 GB RAM, ROCm 7.2:

> decode **41.4 / 41.8 / 41.0 / 41.4 -> 61.7 / 68.4 / 70.5** ... steady
> **~70-71 tok/s**, prompt reads 61.5 -> 98.4 tok/s. `--layer-split auto` =
> 26/22, `--expert-cache auto` resident 12007 + 10690 slots, **92% of the
> routed pairs** resident across the two cards.

**+71% on the same hardware, with the greedy text byte-identical from the
third request on.** The knobs and the pitfalls are all in that document
(`STRATA_STAGE_TRIM=1` so each card loads only its own layers' dense
weights, `STRATA_SPLIT_RING`, the prompt path's loan from the card's own
cache, the drafter belongs on the *last* card on this box).

**2. The peer tier - the V4.1 engine's own attempt, and it lost.**
`local/peer-tier-attempt.patch` (901 lines, three files) gives a second card
a disjoint slice of the ranked experts and has it compute the rows the
primary misses, with the same kernels and the same q8_1 activations so the
rows are bit-identical. Measured: **12.6 -> 11.8 tok/s**, a 5% loss.

The reason is in `docs/ds4/V41_AMD_PEER_TIER.md`: the peer removed 35% of
the routed rows from card 0 - and the tier was only 23% of the step, so
that is ~3% of the step, against a **per-layer hand-off** (activation
upload, metadata, an event and a wait, per-row copies) on ~40 layers a
token. The dense half (63% of the step) stayed on one card.

That is the difference between the two designs: a peer *serves misses* and
pays a hand-off per layer; a layer split *owns layers* and pays one hand-off
per window. The peer keeps the dense half whole; the split divides it.

**3. Expert parallelism - what DeepInfer does.** DeepInfer partitions the
experts across the EP group: each rank owns a contiguous slice of the
physical experts (`eplb/__init__.py`: `ep_group.rank() * num_local_physical_experts
: (rank + 1) * num_local_physical_experts`) and the routed outputs are
combined through symmetric memory (DeepEP-style). So "the weights in two
places, both cards streaming" is exactly DeepInfer's design, and our AMD
overlay stubs it out - the `deep_ep` shim refuses the EP dispatch and
combine.

**What this means for the lane**

- The V4.1 engine gets **design 1**: the layer split (and, if it pays here
  as it does on the qwen4exp path, the pipeline windows that overlap the two
  stages across verify windows - `--pipeline-windows`, `STRATA_PIPELINE_*`).
  The qwen4exp path is the template and it is in the same repo, measured on
  this box.
- DeepInfer's EP (design 3) is the *reference* for the placement rule, and
  it is what the AMD port of DeepInfer would implement - but DeepInfer
  cannot load the 748 B FP8 checkpoint on this box (748 GB of weights
  against 40 GB of VRAM), so the runnable V4.1 here is the GSQ in Strata.
- The peer tier is the wrong shape for this engine and its own measurement
  says so.

**The port (in progress).** `work/strata-ds41`, `docs/ds4/V41_AMD_SPLIT.md`:
`Ds41Dense` gained a layer range (`layer_lo`/`layer_hi`/`own_head`/
`own_embed`) so a stage loads only its own layers' weights and keeps only
its own layers' KV / indexer state; `Ds41MoeTier` gained a device so each
card has its own tier; the engine gained `--layer-split K --split-device D`,
the hand-off at the boundary (the residual and the pre-mix), and the head on
the last stage. The boundary is **forced to 20** and is provably clean: the
ratio-1 kv source at layer 20 is read by every later layer, so a source and
its readers cannot be on different cards. Not wired yet: `--serve` and the
layer-major prompt batch (they refuse a split for now), and the stage overlap
across windows.

## The question

We started by folding DeepInfer (DeepSeek's own V4.1 engine) into hivebench and
porting it to AMD. The goal was stated as: *make the engine work with my models
so we can test it and compare it with what we have already implemented in
hivebench.*

DeepInfer cannot run our models today, and making it do so is a much bigger job
than the port: DeepInfer is V4.1-shaped (MLA, sparse index attention, Engram,
mHC, FP8/FP4 experts). Our models are a different family:

- `qwen4exp` - Qwen3.8-Flash-Next: hybrid GatedDeltaNet + gated attention,
  hyper-connections, a PLE n-gram embedding, 512 routed experts, MXFP8.
- `qwen35moe` - Qwen3.6-35B-A3B (Scion): the same GDN + attention pattern
  without the hyper-connections, 256 experts, ternary experts.

Supporting either in DeepInfer means writing a new model family inside an
engine written for a different one. Maya already ships one of them.

## What project-maya is

`src/` in project-maya **is Strata** - the same `include/strata/`,
`src/core`, `src/kernels`, `src/artifact` layout as our `work/strata` tree
(`include/strata/artifact/gguf_reader.hpp`, `src/core/layer.cpp`,
`src/program/generate.cpp` are Strata's files). On top of that it adds:

- `src/ds4/` - a DeepSeek-V4.1 engine of its own (22 files), with AMD
  wave32 WMMA paths written for `gfx1100/1101/1102/1103/115x/120x`
  (`src/ds4/ds4_gguf.cu:1005`).
- a GLM-5.3-Flash runner (321 B, KDA linear attention, MTP drafter).
- a HIP build for `gfx1100` / `gfx1201` / `gfx1151` / `gfx1030`, with the RDNA
  decode expert kernels **optimised and measured for gfx1100**
  (`docs/AMD_EXPERT_KERNELS.md`: IQ2_XXS gate/up, IQ2_S and IQ3_XXS down).
- tiered expert storage: VRAM -> pinned RAM -> NVMe, with the engine sizing
  the tiers from what it measures.
- its own quantisation pipeline (`tools/maya_quant/`: GPTQ, calibration, KL
  evaluation, recipes) and published Maya quants of GLM-5.3 and V4.1.

The binary that serves our family is `maya` itself: `src/program/generate.cpp`
is the **qwen4exp** driver, and `maya.py`'s `--gguf-dir` points the engine at
a GGUF we already have. The ggml type table targets **Q2_0 (type 42)**, the
ternary container (`include/strata/artifact/gguf_reader.hpp:50`: "42 = Q2_0,
the PrismML ternary 2-bit encoding this engine targets").

## The engines side by side

| | DeepInfer (vendored) | project-maya | DS41Engine | Strata (ours) |
|---|---|---|---|---|
| what it is | DeepSeek's own V4.1 engine (vLLM-derived) | Strata core + GLM-5.3 runner + DS4 runner | Strata fork + its own V4.1 engine | our engine |
| models it runs | V4.1 only | GLM-5.3, V4.1, **qwen4exp** | V4.1 | **qwen4exp**, **qwen35moe** |
| our ternary containers | needs its own loader | Q2_0 (42) | Q2_K / IQ2_XXS | Q2_0 + PTQ1_0/PQ2_0 (repack tool) |
| AMD | our port: engine core runs, ops shimmed in torch | built for gfx1100, expert kernels tuned for it | 7900 XTX, two cards, measured | HIP build |
| expert tiering | none | VRAM -> RAM -> NVMe | VRAM -> RAM -> NVMe | VRAM -> RAM -> NVMe (EP work) |
| frontier models | V4.1, and only V4.1 | GLM-5.3 321 B and V4.1 748 B, with quants | V4.1 | - |

## What this means

**Maya is the better vehicle for "our models on our AMD box", and by a wide
margin.** It is our own engine, extended, with the AMD work already done and
measured - not something we would write from scratch inside DeepInfer. The
hivebench side of that is small: a `maya` launcher next to
`harness/stack/strata.py` and `harness/stack/deepinfer.py`, so the harness can
drive it and compare it against the stacks we already have.

**DeepInfer keeps a different and still unique job: it is the only copy of
DeepSeek's own V4.1 implementation.** Maya's V4.1 and DS41Engine's V4.1 are
both reimplementations, and both measure themselves against the original
("Maya-S picks the same next token as the original 92.6% of the time"). The
original is the ground truth, and DeepInfer is the only accessible copy of it.
That makes the DeepInfer port a *cross-check* for the V4.1 engines, not a
second way to serve our models.

**The honest caveat about V4.1 on this box.** 748 B parameters: even at 2 bpw
that is ~200 GB of files against 31 GB of RAM and 40 GB of VRAM. Both Maya and
DS41Engine handle it by streaming experts from NVMe, and they are honest that
it is slow there. DeepInfer has no expert tiering at all, so a V4.1 run in
DeepInfer on this box is not a near-term thing; it is a numbers-check for a
machine with the memory, or a small-model check here.

**What DeepInfer would still need for our models**, if we ever want it: a
`qwen4exp` executor (hybrid GDN + gated attention + hyper-connections + PLE +
512-expert MoE), a GGUF loader with the ternary codecs, and a GDN recurrent
state in the cache. DeepInfer's `CacheEngine` already has a recurrent
`StateBlock`, so the state part fits; the model and the loader do not exist.

## Recommendation

Revised after the A/B and the storage finding.

1. **Our models run on our own Strata.** 20.52 tok/s on 2 x 7900 XT, and the
   storage layout is what made the difference. The `scion-35b-*` and
   `prism-*` lanes are unchanged; their models now all resolve to the NVMe.
2. **Frontier models run on our own Strata too** - `work/strata-ds41`, whose
   V4.1 engine is merged, built and measured on a 7900 XTX. The weights are
   being fetched; the lane is `stacks/ds41-strata.json`.
3. **Keep the DeepInfer port, re-targeted.** It is the only copy of
   DeepSeek's own V4.1 implementation, which is what makes it a check on
   `strata-ds41` and on DS41Engine's engine. It does not need to run our
   models: that is a new model family in a foreign engine, and our Strata
   already does it better.
4. **Maya is a reference, not a dependency.** Nothing in the harness points
   at it. Its checkout is kept for the tiering design and its AMD expert
   kernels; its 38 GB pack was deleted.
5. **The storage layout is a first-class concern.** `scripts/check_model_storage.py`
   and its test guard it, because a model on the slow volume costs 6x the
   decode rate and nothing else says so.

## What we measured on this box (2 x RX 7900 XT, gfx1100, ROCm 7.2.4)

**It builds.** `maya.sh --backend hip` detects both 7900 XTs, compiles
`build-hip/maya` for gfx1100 and runs. One thing is missing from this box's
ROCm: `rocwmma/rocwmma.hpp` (ROCm 7.2 dropped it from the base install), so
`src/kernels/cuda/glm_batch.cu` fails to compile. It is header-only in 7.2 and
comes from the `rocwmma-dev` package in ROCm's apt repo; it is now unpacked
into `work/.rocm-extra/prefix/include` next to the other ROCm extras this box
was missing (MIOpen, hipBLASLt, rocFFT, RCCL, ...).

**Maya's engine is Strata's engine.** `src/` in project-maya is Strata's own
`src/` (`include/strata/...`, `src/core`, `src/kernels`, `src/program`), and
`serve/server.py` is the same server, whose `--engine` still takes
`mock|strata`. The engine binary is whatever the config's `exe` names, and the
server runs `[exe, "--serve", *args]`. So hivebench's existing
`harness/stack/strata.py` launcher drives Maya unchanged: the stack's config
points `exe` at `build-hip/maya` and the same `serve/server.py` starts it.
No new launcher is needed - `stacks/maya-qwen4exp-q2_0.json` and
`project-maya/maya-qwen4exp-q2_0.json` are all it takes.

**The pack is the trap.** The pack we already had
(`~/Desktop/work/strata-data/packs/q2_0`) is Strata's *canonical* pack. Handed
to Maya it loads, then **spins forever**: one core at 100%, VRAM flat, no I/O,
no output, for 13+ minutes with the GPU at 0%. The engine is inside
`session_capture` (`src/program/generate.cpp:5277`), which only runs when the
pack is *not* native - and on HIP the per-layer graph capture never comes back.

Maya's own pack format is the fix: `tools/iq_pack.py` builds a **native** pack
(`native_experts.txt`, one line per layer with the expert blobs' absolute
offsets) from the same GGUF, and it needs no 34 GB copy - the experts stay in
the GGUF and the engine reads them from there. A native pack skips the capture
path entirely. It also needs three flags the canonical path does not:

```sh
maya --pack pack/native_q2_0 \
     --native <shard1.gguf> --ple-gguf <shard2.gguf> \
     --spec 4 --prefill auto --tokens "9419,11,821,803,369"
```

With those, it loads the model (1416 MiB of weights from the pack, 301 native
projection matrices, the 320,001,536-row PLE table from shard 2) and starts
the expert arena. The canonical-pack spin is an AMD-specific trap worth
reporting upstream: it looks like a slow start, not a hang.

## Why it was slow: the model was on a spinning disk

The single biggest factor in every measurement above was **not the engine and
not the expert cache - it was where the model files live.**

| | device | random 4 KiB read |
|---|---|---|
| `~/Desktop/work/models/qwen38-q2_0/` (both GGUF shards) | `/dev/sda2` - a **FUSE mount of a rotational 2 TB Seagate ST2000DM008** | **7.56 ms p50, 10.5 ms p90** |
| `work/` (the checkout, the pack, the builds) | `/dev/nvme0n1p5` - a CT1000P310 SSD | **0.19 ms p50, 0.22 ms p90** |

40x on random reads. And the PLE table - 320,001,536 rows of 90 bytes in
shard 2 - is read *randomly*, one row per token per layer, straight off that
disk with unbuffered I/O (`--ple-io direct`, the default). Every row is a
seek.

That is what the engine's `host` term is: `verify.cpp:2393` times
`ss.ple.table->gather_batch(...)`, the PLE gather, and adds it to `ms_host`.
Moving the two shards to the NVMe (`work/models-nvme/`, 66.4 GB) changes it:

| | PLE in prefill | host per round | decode |
|---|---|---|---|
| model on the HDD, 90% cache | 412.5 ms | 249.0 ms | 3.39 tok/s |
| model on the HDD, 80% cache | 426.3 ms | 245.8 ms | 3.40 tok/s |
| model on the HDD, 80% cache, immediate re-run | 425.6 ms | 245.8 ms | 3.40 tok/s |
| model on the HDD, `--ple-io mmap` | 681.8 ms | 266.2 ms | 3.14 tok/s |
| **model on the NVMe, 90% cache** | **2.9 ms** | **1.2 ms** | **20.52 tok/s** |
| **model on the NVMe, 80% cache** | **2.1 ms** | **1.2 ms** | **19.65 tok/s** |

The PLE gather went from 412 ms to 2.9 ms and the host term from 249 ms to
1.2 ms per round - a 140x and a 210x improvement, and the whole of the
6x decode gain. The one 14.80 tok/s run was not a better configuration; it was
the one where the PLE rows happened to still be cached.

`--ple-io mmap` does not help: the page cache cannot hold a 28.8 GB table on a
30 GB box, so the rows still miss.

**Two lessons worth keeping:**

- **The expert cache was never the bottleneck.** At 90% coverage the misses
  are 0.5 per layer and the pool is idle; at 80% they are 1.1 and the
  difference is 4% of the decode rate. The cache matters only in that it
  must be *large enough that the pool is not the bottleneck* - and at 80% it
  already is.
- **The storage layout mattered more than every engine choice combined.**
  It also means the 1.57x and 5.2x "Maya is slower" figures below are
  measured with the disk out of the way, and the remaining gap is real.

## Where the models live (the harness's model library)

`hivebench/models/gguf/` is the library the harness serves from - it is what
`LlamaServerManager.resolve_model` searches, and what an HF download lands
in. Every entry in it is a symlink; **every one now points at the NVMe**:

| entry | size | points at | used by |
|---|---|---|---|
| `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf` + `-00002-...` | 35.0 + 26.8 GiB | `work/models-nvme/` (copied off the FUSE volume) | `prism-125b`, `prism-125b-lowvram`, `strata-125b`, `maya-qwen4exp-q2_0` |
| `Qwen3.8-35B-A3B-IQ2_M.gguf` | 11.7 GiB | `artifacts/ternary/refs/qwen35/` | `prism-35b` |
| `Ternary-Bonsai-2-27B-PQ2_0.gguf` | 6.7 GiB | `artifacts/ternary/oracle/bonsai27/` | `live-proof` (face) |
| `Ternary-Bonsai-1.7B-PQ2_0.gguf` | 0.4 GiB | `artifacts/ternary/oracle/bonsai17/` | `live-proof` (worker) |
| `Scion-35B-A3B-PQ2_0-corr.gguf` | 10.6 GiB | a real file in the library | - |
| `qwen35-release-v2-cur05-soup.gguf` | 10.6 GiB | `artifacts/ternary/moe/qwen35/rental-v2/hf-release/` | `scion-35b-v2`, `-cascade`, `-v2-dspark` |
| `mtp-dspark-v2-cur05.gguf`, `mtp-sidecar-v2-cur05.gguf` | 0.1 GiB | `artifacts/ternary/moe/qwen35/` | the dspark sidecars |

The remaining stack tiers (`peer-2tier`, `peer-3tier`,
`scion-flashnext-strata`) name real Hugging Face repositories and are fetched
into the library - on the NVMe - at load time. `deepinfer-v41` names
DeepSeek-V4.1-Flash-1017, which is not on this machine.

**What is still on the FUSE volume** is the training/eval material, not
served models: the `export-2l/` and `export-4l/` qwen4exp exports
(3.7-31.9 GiB each), `qwen4exp-48l-ptq1_0-corr-step3000.gguf` (34.4 GiB) and
`scion-strata/2l-test-q2_0.gguf` (4.0 GiB). The 48l model and the 2l test
are copied to `work/models-nvme/` as well; the 2l/4l exports stay where they
are (they are read by the scion tooling, not served).

### The guard

`scripts/check_model_storage.py` walks every stack, resolves its model the
way the harness does, and reports the size and the device behind it,
flagging anything on a FUSE mount or a rotational disk:

```sh
python scripts/check_model_storage.py        # the report (exit 1 if any is slow)
python scripts/check_model_storage.py --slow # only the offenders
```

`tests/unit/test_model_storage.py` runs the same check as a test, so a model
that lands back on the slow volume fails CI rather than quietly costing 6x
the decode rate.

## The A/B: our Strata vs project-maya, same model, same box

`Qwen3.8-Flash-Next-GSQ-RCO-Q2_0` (35B, 512 experts, ternary Q2_0), prompt
`9419,11,821,803,369` ("Hello, my name is"), 16 greedy tokens, `--mmap-experts`,
the same expert profile, the same PLE table. **Every configuration produced the
same 16 tokens** - the engines agree.

### Why it is fast or slow: where the experts live

The 24,576 expert slots (512 x 48 layers, 1.38 MB each) are 33.97 GB. The
router takes 10 of 512 per layer, so a token touches ~480 slots and *which*
ones changes every token. The engine's VRAM expert cache is what decides
whether those come off the GPU or off the CPU pool, and the pool costs
70-107 ms per round.

| config | cards | cache slots | policy | MTP head | CPU experts/layer | decode | prefill | TTFT |
|---|---|---|---|---|---|---|---|---|
| Strata | 1 | 2,048 (8%) | shared counter | yes | 9.26 | 8.55 tok/s | 2.64 | 1641 ms |
| Strata | 1 | 8,000 (33%) | shared counter | no | 8.70 | 2.96 tok/s | 3.12 | 1444 ms |
| Strata | 1 | 8,000 (33%) | per-layer | yes | 5.19 | 5.73 tok/s | 4.12 | 1166 ms |
| **Strata** | **2** | **22,000 (90%)** | shared counter | no | **0.49** | **14.80 tok/s** | **7.65** | **576 ms** |
| Strata | 2 | 22,000 (90%) | shared counter | yes | 0.49 | 0.27 tok/s | 0.08 | 54,398 ms |
| Maya | 2 | 22,000 (90%) | shared counter | no | 0.52 | 2.83 tok/s | 5.00 | 1070 ms |
| Maya | 2 | 22,000 (90%) | shared counter | yes | 0.47 | 2.77 tok/s | 3.46 | 2708 ms |

Three things fall out of this:

1. **The experts do fit, and when they are resident the CPU pool goes idle.**
   At 90% coverage the misses drop from 8.7 per layer to 0.49, the pool from
   ~100 ms/round to 20 ms, and the decode rate goes from 8.55 to **14.80
   tok/s** - 34 GB of experts in the 40 GB of VRAM these two cards have.
2. **The cache's default policy wastes most of the budget.** The engine fills
   slots "in arrival order from one shared counter" (its own words), so 8,000
   slots bought only a 27% hit rate; the same 8,000 with
   `--expert-cache-per-layer` halved the misses (8.70 -> 5.19).
3. **The MTP drafter does not survive the two-card split**, in either engine:
   the draft head lives on one device and the "wait for rings" term goes to
   192 ms/round in Maya and 1,942 ms/round in ours - 0.27 tok/s. So it is
   either two cards without the drafter (14.80 tok/s) or one card with it.

### The coverage curve, with the model on the NVMe

Two cards, 22,000 slots, no MTP head, same prompt, same profile, **model on
the NVMe**:

| | Strata (0.1.37) | Maya (0.1.27) |
|---|---|---|
| output | `8254 13 353 1044 264 220 17 20 4514 6086 8330 13 353 599 978 23330` | **identical** |
| decode | **20.52 tok/s** | 6.18 tok/s |
| prefill | 8.50 tok/s | **10.92 tok/s** |
| time to first token | 530 ms | 531 ms |
| PLE gather | 2.9 ms | 2.2 ms |
| CPU experts per round | 0.53 | 0.51 |
| host thread per round | 1.2 ms | 1.7 ms |
| pool per round | 21.6 ms | 101.3 ms |
| wait for rings | 19.9 ms | 48.3 ms |

With the disk out of the way the two agree token for token, Maya is *faster*
at prefill, and the remaining 3.3x at decode is the **CPU expert pool**
(101.3 ms/round against our 21.6) plus the ring wait (48.3 vs 19.9) - not the
host thread any more. Our engine is at revision 0.1.37 against Maya's 0.1.27,
so that is where the newer work is.

### What coverage buys, and where the ceiling is

All on the NVMe, Strata, the same 16 tokens:

| config | cards | slots | coverage | policy | MTP | CPU misses/layer | decode | prefill | TTFT |
|---|---|---|---|---|---|---|---|---|---|
| **A** | 2 | 22,000 | 90% | shared | no | 0.53 | **20.52 tok/s** | 8.50 | 530 ms |
| **B** | 2 | 19,660 | **80%** | shared | no | 1.13 | **19.65 tok/s** | 11.63 | 393 ms |
| C | 1 | 8,000 | 33% | per-layer | yes | 5.56 | 16.90 tok/s | 6.02 | 728 ms |
| D | 2 | 23,000 | 93% | shared | no | - | **does not fit** - card 1 at 21.2 GB of 20 GB | | |
| E | 1 | 11,000 | 45% | shared | yes | - | **does not fit** - VRAM exhausted with the MTP head | | |
| - | 2 | 22,000 | 90% | shared | no | 0.51 | *3.39 tok/s on the HDD* | 5.27 | 931 ms |

**80% is fine**: 19.65 tok/s, 4% under the 90% configuration, and it leaves
0.6 GB more VRAM per card. The coverage curve is nearly flat from 80% up; the
cache only has to be *big enough that the CPU pool is not the bottleneck*,
and 80% already is.

The ceiling is VRAM, not coverage: 93% (23,000 slots) exhausts the card
before it starts. And the MTP head is a one-card feature - with the model
split across two cards the draft head's ring wait goes to 1,942 ms/round
(0.27 tok/s), so the choice is two cards without the drafter (20.52) or one
card with it (16.90, and the second card is free).

Before the model was moved, the same comparison read 3.39 tok/s against
2.83 tok/s, and the "Maya is 1.57x slower" figure from an earlier revision of
this document was an artefact of the under-provisioned cache. Both were
measuring the disk.

So for **our own model, our own Strata is the better engine today** - 3.3x at
decode. Maya's value is elsewhere: the frontier models (GLM-5.3 321 B, V4.1
748 B) with their quants and tiering, the Windows/AMD packaging, and the
public project around it.

## Memory budget on this box (read this before running anything)

30 GB RAM, 2 x 20 GB VRAM, and a 62 GB model. The numbers that matter:

| what | size |
|---|---|
| the model's experts (512 x 48 x 1.38 MB) | 34 GB, read from the GGUF |
| the PLE table (320,001,536 rows) | 28.8 GB, read from shard 2 |
| the dense weights | 1.9 GB |
| a VRAM expert cache (`--expert-cache auto`) | up to 14 GiB of the card |
| the MTP draft head | 799 MiB of the card |

The engine's **resident expert arena** (`ArenaExpertSource`) copies the whole
34 GB into host RAM and pins it. That does not fit here, and it is what
OOM-killed this box on 2026-10-21 when a run also asked for
`--expert-cache auto` (14 GiB of VRAM) and the MTP head on top.

Rules for a run on this box:

- `--mmap-experts` (the experts come from the file through the page cache) or
  a native pack with `--experts-bin`; never the resident arena.
- `--expert-cache 11000 --expert-cache-device1 11000` - the two cards' VRAM
  caches, 14.16 GiB each, 22,000 of the 24,576 slots (90%). That is the
  configuration that reaches 14.80 tok/s. `--expert-cache auto` picks 15 GiB
  off *one* card and leaves the rest to the CPU pool.
- **No MTP draft head (`--mtp`) when the model is split across two cards**:
  the draft head is on one device and the ring wait goes to 192 ms/round
  (Maya) or 1,942 ms/round (ours). It is a one-card feature.
- No `--expert-cache-per-layer` on a full cache: it is a *small*-cache policy
  (it matters at 8,000 slots, where it halves the misses, and is pointless at
  22,000, where there is nothing to choose).
- One model process at a time, and watch `VmRSS` while it loads. The
  two-card configuration still reads ~30 GB of experts from the GGUF at
  startup; that is page cache, but it is worth knowing a start can take
  several minutes cold.

## Unverified

- The decode speed on this box: the run gets through a ~7 minute startup
  (34 GB of experts read from NVMe into a 31.64 GiB arena, the PLE table read
  alongside it) and the decode measurement was still running when this was
  written.
- Whether the 2 x 20 GB split (`gpu: [0, 1]`, `layer_split: auto`) beats the
  single-GPU run we measured.
- Whether the 62 GB model leaves enough for the file tier to be useful here.
