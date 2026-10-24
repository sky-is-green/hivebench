# The slot layer: what the prior art already solved

Hive's tiers name a role and carry the engine as a field (`backend`, `launch.kind`, `pin`, args); the harness owns
each tier's lifecycle. That capability is not new, and the README says so. This note records what the closest projects
actually do, and what is worth taking rather than re-inventing.

## The four references

| project | what it is | how it slots engines |
|---|---|---|
| **LocalAI** (Go) | local-first, 60+ backends, one OpenAI-compatible API | every backend is its own **gRPC process** (`backend/backend.proto`: `LoadModel`, `Predict`, `PredictStream`, `Health`, `Free`, `Status`, `GetMetrics`, `ModelMetadata`, `SlotCache`); per-model `backend:` field; `tensor_split`/`main_gpu`; hardware-aware defaults; on-demand backend gallery; external backends registered at runtime |
| **NVIDIA Triton** (C++) | data-center inference server, any framework | backend API; model repository + per-model config; `instance_group` places instances on named GPUs; ensembles/BLS compose models into DAGs |
| **Ray Serve LLM** (Python) | distributed serving framework | engine-agnostic `LLMEngine` protocol (vLLM, SGLang); placement groups; TP/PP/PD topologies |
| **ROCm Infera** (AMD) | OpenAI endpoint + router for AMD GPUs | workers are vLLM / SGLang / ATOM; KV-aware routing, PD disaggregation, tiered KV cache (GPU→RAM→NVMe→network) |

Control planes (Modelplane, Kthena, Moreh) go further up: topology as data, model placement as a policy. Hive's
stack JSON is already in that family.

## What to borrow

1. **A backend manifest** — LocalAI's `backend/index.yaml` declares, per backend, its acceleration variants
   (CPU/CUDA/ROCm/Vulkan/Metal) and capabilities, and the gallery installs the right one on demand. Hive today
   assumes `tools/backends/<backend>/llama-server` and knows nothing about variants or capabilities.
   *Landing spot:* a manifest per engine + `hive serve` choosing the variant from detected hardware.
2. **Hardware-aware defaults** — LocalAI's `hardware_defaults.go` maps VRAM to physical batch, context guard and
   parallel slots. We hand-write `ngl`/`ts`/`arena` in every stack. *Landing spot:* `hive serve --cards` computes
   `pin`, `ts`, `ngl`, ctx and KV type from `rocm-smi`/`amd-smi`.
3. **Model identity on every request** — LocalAI added `ModelIdentity` after a recycled gRPC port let a stale route
   answer from the wrong model (#10952). Hive's endpoint tiers have the same failure mode: a port can come back
   with a different model behind it. *Landing spot:* each endpoint tier declares its model name, and the client
   refuses an answer whose model does not match.
4. **Status and metrics as a contract** — LocalAI's `Status` reports state + memory usage; `GetMetrics` reports
   tokens/s, prompt/generated tokens per slot. Hive's `caps` is a placeholder today. *Landing spot:* `/health`
   carries model, state, VRAM and measured tok/s; the scheduler's cost table can read it instead of the planning
   table.
5. **PD disaggregation as the *correct* two-card split** — Infera runs prefill and decode as separate pools with a
   KV hand-off. That is the two-card parallelism that respects the dependency chain: different *phases*, not
   different layers. *Landing spot:* future; prefill on one card while decode runs on the other is the one
   two-card arrangement that overlaps real work instead of idling a card.
6. **Slot cache** — LocalAI's `SlotCache` saves/restores per-slot KV without exposing file paths. For the cascade
   this would make an escalation reuse the first tier's prefix instead of re-prefilling it. *Landing spot:* future,
   after the two-card serving path.

## What not to borrow

* **The gRPC transport.** Hive's engines already speak OpenAI over HTTP; adding a proto layer buys nothing for
  three adapters and costs a schema, a codegen step and a second wire format.
* **Triton's C++ backend ABI.** Wrong scale for a local harness.
* **Kubernetes operators (Kthena, Infera operator).** Hive is deliberately local-first: no cluster, no CRDs.
* **Infera as the base engine.** It orchestrates vLLM/SGLang/ATOM workers, i.e. HF-format models. Hive's lane is
  GGUF through llama.cpp and the Strata fork; Infera could be an *engine slot* later, not the foundation.

## What none of them have

The loop: slots chosen as **roles** in solve → detect → repair → escalate, a scheduler that picks a path per
request from measured cost and a quality target, and a verifier that never sees the answer key. Triton's ensembles
are static DAGs; LocalAI's backends are a deployment convenience; neither has a notion of "this answer failed,
escalate". That is the part to keep building.

## Consequences for `hive serve`

1. Engine-blind CLI: apply a stack, wait for health, print the endpoint, unload on exit.
2. A manifest + hardware detection so `--cards 2` writes the right `pin`/`ts`/`ngl` instead of asking the user.
3. Model identity on endpoint tiers (borrow #3) — cheap, and it prevents a class of silent wrong answers.
4. Default two-card GGUF stack through llama.cpp `--split-mode layer`; the strata slot stays the advanced path for
   models bigger than the cards.
