# Hive

**A local-first stack for memory, verification, and routing in language models.**

Hive began as a benchmark for a long-context memory system. It now contains
two papers, the systems they describe, and the harness that measures them.

| the paper | what it is |
|---|---|
| **[Splinter Memory](splinter/SPLINTER-WHITE-PAPER.md)** | managed-decay context curation for long-horizon conversations: a bounded context, a membrane for dedup and drift, and a small encoder fleet. The [HF card](https://huggingface.co/SkyIsNotGreen/splinter-memory) is the Hub edition; the [project](splinter/README.md) has the code. |
| **[The Hive Cascade](cascade/CASCADE-PAPER.md)** | a gold-free verification-and-repair loop: solve, detect, repair, escalate. It carries the role taxonomy, the controller contract, the measured tier results, and the negative results I kept. The [visual companion](https://huggingface.co/spaces/SkyIsNotGreen/hive-cascade-paper) walks through it. |

**Two papers, one stack.** The memory decides what the model sees; the cascade
decides whether the answer is trusted; HiveBench measures both without an
answer key in the serving path.

## What this repo does

HiveBench is the evaluation suite and Studio sidecar. It answers two questions
that ordinary benchmarks do not:

1. **Was the answer right?** Deterministic verifiers (the task's own tests,
   arithmetic consistency, canonical forms), not an LLM judge in the evidence
   path.
2. **Was the context the reason?** The same turns and the same model, one arm
   curated and one naive, scored the same way.

The harness is local-first: models, verification and memory stay on the box;
the API is opt-in. It runs the cascade, the memory arms and the A/B batteries
from the same result schema, and every run leaves a report, a manifest and a
harvest. The detailed guide is in [`hivebench/README.md`](hivebench/README.md).

## One harness, any engine, placed per role

A stack tier does not hard-code an engine. It names a **role** — in the cascade,
a functional slot such as `face`, `worker` or the escalation tier — and carries
the engine as a field: `backend`, `launch.kind`, `pin` (which card), `ctx`, KV
cache types, args. The harness owns each tier's lifecycle (start, health-check,
route, stop), and the endpoint contract is the same whether the engine behind it
is llama.cpp, the Strata fork (which serves a 163 GB DeepSeek-V4.1 through its
own server) or DeepInfer. A new engine is one adapter module plus a
`launch.kind` entry; the stack schema, routing, API and app do not change.

**What this is not.** Multi-engine serving is not new and we do not claim it.
NVIDIA Triton runs several backends with per-model GPU placement; LocalAI
coordinates 60+ backend processes behind one API with per-model GPU settings and
runtime registration; Ray Serve LLM defines an engine-agnostic protocol; control
planes such as Modelplane advertise "any model on any engine". If the claim were
"slots for engines", those would be counterexamples.

What we have not found elsewhere is the combination: the slots are **functional
roles in a verification loop** (solve → detect → repair → escalate), the engine
and the card are chosen from that role's measured cost, and the verifier never
sees an answer key. In Triton or LocalAI the multiple engines are a deployment
convenience; here they are the mechanism by which a wrong answer is caught. That
combination — plus local-first operation on two consumer cards, with no
Kubernetes, no container runtime and no API key — is the claim.

## Layout

| path | what lives there |
|---|---|
| [`splinter/`](splinter/README.md) | the memory system, its white paper, the encoder notes and figures |
| [`cascade/`](cascade/README.md) | the cascade paper, the pilot, the controller contract and the prior-art survey |
| [`hivebench/`](hivebench/README.md) | the evaluation suite and Studio sidecar docs |
| [`docs/`](docs/) | the shared record: [status](docs/STATUS.md), [A/B results](docs/AB-RESULTS.md), failures, engine notes |
| `harness/`, `experiments/`, `tests/`, `testing/` | the implementation that runs the suite |

The paper and the implementation live side by side; the papers are written to
be re-run, not just read.

## Start here

| | |
|---|---|
| the two papers | [Splinter Memory](splinter/SPLINTER-WHITE-PAPER.md) · [The Hive Cascade](cascade/CASCADE-PAPER.md) |
| the current state | [measured / invalid / unmeasured / blocked](docs/STATUS.md) |
| the visual companion | [hive-cascade-paper](https://huggingface.co/spaces/SkyIsNotGreen/hive-cascade-paper) |
| the task sets and run reports | [SkyIsNotGreen/hive-cascade](https://huggingface.co/datasets/SkyIsNotGreen/hive-cascade) |
| the memory cards | [splinter-memory](https://huggingface.co/SkyIsNotGreen/splinter-memory) · [splinter-encoder](https://huggingface.co/SkyIsNotGreen/splinter-encoder) |

## Install

```bash
git clone https://github.com/sky-is-green/hive.git
cd hive
python3 -m venv .venv
.venv/bin/python -m pip install -e splinter   # the memory system
.venv/bin/python -m pip install -e .          # the suite and the Studio
```

The suite's grouped runs and the Studio are in
[`hivebench/README.md`](hivebench/README.md).

## The state

The short version, from [`docs/STATUS.md`](docs/STATUS.md): the 35B and 27B
tiers are measured; the 176B's generation is net zero; the memory's dedup
decision is measured, and the drift benchmark is invalid and being rebuilt. The
API tier is measured (28/28 on the 28 tasks, 2.6 s median, free OpenCode), so
is the execution checker (6/7 errors, 4/4 at p >= 0.97, 0/21 false rejects,
0.23 ms median), and so is the scripted controller (28/28, 7 escalations,
0/21 correct sent up, 9.24 s mean / 17.22 s conditional), and the
API-in-the-27B-slot path (DeepSeek V4.1 Flash Free: 28/28, 7 API escalations,
0/21, 8.57 s / 14.54 s; GLM 5.3 Flash Free: 27/28, 7 API, 0/21, 7.80 s /
11.45 s). The API-driven controller is unmeasured; grounding is not pursued.
This is a research harness, not a product.

## Engine notes

The serving engine, the hardware constraints and the engine options are in
[`docs/ENGINE.md`](docs/ENGINE.md).

## License

MIT. The paper citations are in each paper, and the software citation is in
[`CITATION.cff`](CITATION.cff).
