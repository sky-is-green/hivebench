# HiveBench

> Part of [Hive](../README.md). This is the evaluation suite and Studio sidecar.

[![CI](https://github.com/sky-is-green/hive/actions/workflows/ci.yml/badge.svg)](https://github.com/sky-is-green/hive/actions/workflows/ci.yml)

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](https://github.com/sky-is-green/hive)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)](https://github.com/sky-is-green/hive)
[![pytest](https://img.shields.io/badge/pytest-0A9EDC?style=flat&logo=pytest&logoColor=white)](https://github.com/sky-is-green/hive)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://github.com/sky-is-green/hive)
[![Last commit](https://img.shields.io/github/last-commit/sky-is-green/hive/main?style=flat&label=Last%20commit&logo=git)](https://github.com/sky-is-green/hive/commits/main)
[![Repo size](https://img.shields.io/github/repo-size/sky-is-green/hive?style=flat&label=Repo%20size)](https://github.com/sky-is-green/hive)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../LICENSE)

**A local-first AI harness: a console that runs, measures and self-corrects models
on your own hardware — plus the deterministic suite that proves the measurements.**

The evaluation suite + the Studio sidecar for the
[Splinter Memory](../splinter/README.md) system. The
Studio serves **106 REST operations** across 97 paths (97 operations under `/v1/`).
The suite is green (**1183 unit + 126 integration** tests, verified Oct 22); the
live batteries and the Studio are under active development.

---

## What this suite is for

Most harnesses tell you *how a model performed*. Two questions are more
useful, and almost nothing answers them:

1. **Was the answer right?** Not "it sounded right" — right, checked
   deterministically, without a judge model in the loop.
2. **Was the context you fed it the reason it worked?** Not "the prompt
   looked reasonable" — the same turns, the same model, one arm curated
   and one naive, scored the same way.

HiveBench is built to answer both, on your own machine, with evidence you
can re-run:

- **The answers are verified, not judged.** A verifier reads a gold-stripped
  spec: the task's own tests, arithmetic consistency, canonical-form checks.
  No LLM-as-judge in the evidence path.
- **The memory is measured, not asserted.** The white paper's P1–P11
  predictions ship as executable tests with measured PASS/FAIL verdicts.
- **The failures become data.** Every ladder-resolved item is harvested to
  `escalation.jsonl` — task, draft, findings, repair, resolution — a corpus
  built from your own failures, for training and distillation.
- **Nothing leaves the box.** Local models, local verification, local memory.
  The API is opt-in, and it is asked first.

The goal, in one line: **make a local model as trustworthy as a hosted one,
and prove it without a phone-home.**

---

## The capabilities

### The quality loop — the Hive Cascade

**generate → detect → repair → escalate**, entirely local. The runtime never
sees an answer key; the verifier reads a gold-stripped `runtime_spec`.

- **Detect** — deterministic verifiers (`code_exec`, `constraint_check`,
  `schema_ast`, `consistency`) report findings.
- **Repair** — one greedy re-attempt with the finding quoted; truncated or
  byte-identical attempts are rejected; a cleared repair is re-framed
  canonically.
- **Escalate** — a **human gate** (the first trigger verbose, later compact)
  and an optional API fallback. The gate is a *label*: "solve" means local
  was recoverable, "escalate" means the API was genuinely needed.
- **A learned gate** (opt-in) — a small head as a third trigger, blind-spot
  only. **Off by default**, by a rule written down before the measurement:
  three runs agreed no 0.8B head separates on the silent distribution.
- Measured: **58/66 → 62/66**, and 65/66 in the greedy regime with **zero
  false rejects**.

### The memory — curated context, and the tools around it

- The **curated context**: a store with decay, dedup, a topic-drift
  detector, a comb (the SSD archive that can resurrect what fell out of the
  budget), and a medium drone for the hard candidates.
- **`/import <path>`** — shows the LLM a file: the text rides along with
  your next message *and* lands in the conversation's store, so it is there
  for later turns. Secrets are redacted on the way in; binary files are
  refused.
- **MCP** — the memory over `POST /v1/mcp`, with the tools an external
  client may call, and a read-only mode.
- **Global harness controls** — one place for everything the harness does:
  **the encoder** (the small model that does the retrieval and the memory
  management — you choose it), the curation and retrieval defaults, the
  cascade's defaults, the MCP surface, and the runtime a new engine is
  launched with. There is no second menu for any of it.

### The console — HiveBench Studio

- **A launch console with one tab per engine** — the shared dials (the
  context window, the port) above, then **llama.cpp** (the model, the GPU
  layers, the auto-fit, the devices, the threads, the slots, the KV cache,
  the batch sizes, the alias) or **Strata** (the config profile, the
  checkout, the VRAM reserve, the prefill, the spec depth, the extra engine
  flags). Start launches the engine the tab names.
- **The chat** — Splinter mode (one curated reply) or Agent mode (the dsh
  loop, with tools). Every message says who produced it: the model, the
  studio encoder, the **Command Console** (cyan), or you (black on yellow).
- **Stop is a real stop** — the Studio owns the provider connection, so
  stopping closes it and the *engine* stops generating (llama.cpp and Strata
  both stop on the disconnect). A closed tab stops the turn too.
- **Unload really unloads** — the local instances and the Strata engine give
  their VRAM back; the provider, the engine profile and the default move
  with them, so nothing is left pointing at a dead port.
- **The cascade console** — every runner knob in one form, from the UI or
  the CLI.
- **Compaction as a free switch point** — a conversation that outgrows its
  window becomes a checkpoint, and a checkpoint is where you may safely
  change which engine serves the session, behind a canary gate.
- **The app assistant** — the dsh agent with the Studio's own API map in its
  context: it reports your hardware, models, engines and runs, and acts on
  them.
- **The library and the hub** — search Hugging Face for GGUF repos, list
  their files, download with live progress, manage the local folder.
- **The command system** — 15 console commands, a palette (Ctrl+K), and a row
  above the chat bar holding the ones you picked (each with a small red `x`).
- **Developer mode** (`/dev on`) — the deep-research queue, the Docker setup
  tab, the raw server status.

### The evidence — the offline suite

- **1183 unit + 126 integration tests**, ~2 minutes, **no LLM calls and no
  API keys**. CI runs exactly this.
- **Paired A/B** — the same turns, the same model, curated context vs the
  naive FIFO window, both scored deterministically, with both arms' stores
  replaying identical history so the comparison isolates *selection*.
- **Checkpointed, resumable** live runs that survive a crash or a reboot.
- **Probes** — the fit estimator (the tiers, the disk guard, the context
  that fits), the leak gate (does an unload really give the memory back?),
  the launch verification (are the weights there, does it fit, is the binary
  there, is the port free — each with the fix), and the model discovery.
- **The encoder suite** — the three jobs the harness gives its encoder
  (rank, dedup, drift), measured on the harness's own traffic with lexical
  labels, plus the cost of each candidate. See `splinter/docs/ENCODER-NOTES.md`.

---

## How we made the core

**The Studio is the host; the Splinter is the memory.** The Studio owns the
process, the transport and the console; the memory system owns the
curation, the store and the drones. The two share one conversation
registry, so there is no second copy of the truth. The Studio's routes are
matched before the mount, so the host is the transport where it needs to
be — that is how one button can close the provider connection and stop the
engine.

**The evidence path has no model in it.** Everything that decides whether
an answer is right is a deterministic check: the task's own tests, the
arithmetic, the canonical form. A judge model appears only where a judge
model is the subject (the learned gate), never where a verdict is claimed.
The auditor layer corroborates asynchronously and is explicitly not
constitutive: it shares the served model's biases.

**The runtime is gold-free by construction.** The runtime receives a
`runtime_spec` with the answer stripped. The checker sees the answer. A
correct answer and a lucky guess are indistinguishable to the runtime, on
purpose.

**A decision to not build is a result.** The learned gate is off because
three measurements agreed it does not separate, and the rule was written
down first. The AFK mode and the M5 embedding path were shelved the same
way, and the shelves are written down too.

**The UI is the product.** One theme, one place per setting, the truth
where the action is: the model list in the library, the fit bar in the
engine profile, the error on the bar that was blocking the screen. The
console is where the harness is honest with the person using it.

---

## Install

The memory system and the suite live in one checkout:

```
hive/
  splinter/   # the system: drones, cortex, retention, backends
  harness/    # the suite + Studio sidecar
```

Windows (PowerShell):

```powershell
git clone https://github.com/sky-is-green/hive.git
cd hive
python -m venv .venv
.\.venv\Scripts\python -m pip install -e splinter   # the system
.\.venv\Scripts\python -m pip install -e .          # suite + studio (console scripts)
```

Linux/macOS:

```bash
git clone https://github.com/sky-is-green/hive.git
cd hive
python3 -m venv .venv
.venv/bin/python -m pip install -e splinter   # the system
.venv/bin/python -m pip install -e .          # suite + studio (console scripts)
```

Flat import names are preserved (`experiments.*`, `testing.*`, `tests.*`,
`harness.*`, `deepseek_harness`); the system under test resolves as `splinter` /
`cortex` / ... from `splinter/`. Override its location with `$SPLINTER_HOME`.

## Run the studio (HiveBench Studio)

Two commands are the whole story, run from this checkout: `--setup` copies
`providers.example.json` → `providers.local.json` if missing, probes for a reachable
backend (LM Studio on `:1234`, or auto-starts the local `llama-server` from
`models/gguf`), and prints the next step. The studio serves the splinter over a
FastAPI API; the endpoint contract lives in `harness/app.py`, and the
[`splinter/docs/INTEGRATE.md`](../splinter/docs/INTEGRATE.md) shows how to point external clients at it.

```powershell
.\.venv\Scripts\python -m harness --setup   # copies providers config, probes backend, warms the drone
.\.venv\Scripts\python -m harness           # studio UI on http://127.0.0.1:8765
```

## Sandbox (optional, Docker-backed)

The Studio ships a `/v1/sandbox/exec` endpoint: run a command in an isolated
container when the Docker daemon is up (network disabled, 512 MiB / 1 CPU,
hard timeout), and fall back to a local subprocess otherwise. It backs the
cascade's `code_exec` detector and gives the agent a safe execution surface.
Check availability with `GET /v1/sandbox/status`.

```powershell
curl -X POST http://127.0.0.1:8765/v1/sandbox/exec -H "Content-Type: application/json" -d '{"command":"python -c \"print(42)\"", "timeout":10}'
```

## Hive Cascade (core feature)

Hive Cascade is the local-first answer-quality loop built into HiveBench: **generate → detect → repair → escalate**. It runs entirely on your machine — no answer key is ever exposed to the runtime (the verifier reads a gold-stripped `runtime_spec`).

- **Detect** — deterministic verifier (`consistency` + the task's own tests via `code_exec`), reported as findings.
- **Repair** — one greedy local re-attempt with the finding quoted; truncated/byte-identical attempts are rejected; cleared repairs are re-framed by `canonical()`.
- **Escalate** — human gate (first trigger verbose, later compact) and an optional API fallback.
- **Learned gate** (opt-in) — `intern-0.8b` readout as a third trigger, blind-spot only; off by default by its own pre-registered rule.

Control it from the **Studio UI** at `/cascade` (every runner knob exposed) or the **CLI**:

```powershell
.venv\Scripts\python -m experiments.cascade.run_cascade `
  --tasks experiments/cascade/tasks-bench.json `
  --judge deterministic --repair auto `
  --gate intern-0.8b          # optional third trigger; off by default
```

Every ladder-resolved item is harvested to `escalation.jsonl` next to the report (training-side only), feeding repair SFT and local-tier distillation. See `../cascade/CASCADE-PILOT.md` and `experiments/cascade/LEARNED-GATE-PLAN.md`.

## Console & UI (HiveBench Studio)

The Studio (`python -m harness`) is the control surface for everything. It serves
a single hive theme (yellow/black) across every page.

### Pages

| Page | What it does |
|---|---|
| `/` → `/runs` | Run bundle index |
| `/server` | The console: **Home**, **Settings**, **Global**, **Cascade** + a "More tools" group (Agent, Engines, Local library, Providers, Hub, Inspector, Training). The chat pane is the middle column; the launch panel is on the right |
| `/cascade` | Hive Cascade launch console — every runner knob, advanced collapsed |
| `/runs`, `/view/{run}` | Browse and inspect run bundles (rendered reports) |
| `/stacks` | Stack management (multi-tier model stacks) |
| `/docs` | Live OpenAPI docs for all 106 operations |

### Papers

| Paper | What it is |
|---|---|
| [the Hub card](https://huggingface.co/SkyIsNotGreen/splinter-memory) | **Long-Context Memory Curation Management** — the Splinter Memory white paper in the form the Hub reads it; the long-form paper is [`splinter/SPLINTER-WHITE-PAPER.md`](../splinter/SPLINTER-WHITE-PAPER.md) |
| [`cascade/CASCADE-PAPER.md`](../cascade/CASCADE-PAPER.md) | **The Hive Cascade** — detect → repair → escalate: the gold-free runtime contract, the role taxonomy, the measured results and the pre-registered negative result |

### The A/B campaign

| Document | What it is |
|---|---|
| [`docs/AB-PLAN.md`](../docs/AB-PLAN.md) | The scope: the comparisons, their metrics, their falsification conditions and their costs, ordered by information per hour |
| [`docs/AB-RESULTS.md`](../docs/AB-RESULTS.md) | The results so far: the two Scions head to head, the cascade's arms in the multi-sample regime, the curation against the FIFO window, and the encoder's three jobs |
| [`splinter/docs/ENCODER-NOTES.md`](../splinter/docs/ENCODER-NOTES.md) | Which encoder the harness should use: the sweep, the held-out test, and the falsified hypothesis about the membrane's thresholds |

### Console features

- **Home tab** — orientation, quick links, and the **backend selector**
  (Strata / Llama.cpp / API — one source of truth for the Studio and Cascade).
- **Settings tab** — a searchable settings index: 33 settings across 8
  categories, each linking to where it lives.
- **Global tab** — the harness-wide controls, in one place: **the encoder**
  (the small model that does the retrieval and the memory management — the
  one you pick is the one that runs), the memory and RAG defaults, the
  cascade's defaults, the MCP surface, the runtime a new engine is launched
  with. The separate Splinter menu is gone; there is one home for all of it.
- **Command palette** (Ctrl+K) — every console command, searchable. Fifteen
  commands ship today: `help`, `new`, `save`, `model`, `mode`, `provider`,
  `engine`, `bench`, `train`, `status`, `cascade`, `sandbox`, `compact`, `dev`,
  `import`.
- **Your commands, at hand** — the row above the chat bar holds the commands
  you pick (the picker sits at its far right); every one carries a small red
  `x` to take it off the row. Tabs work the same way: pin the ones you use.
- **`/import <path>`** — shows the LLM a file: the text rides along with your
  next message and lands in the conversation's store, so it is there for later
  turns too. Text files only; secrets in the file are redacted on the way in.
- **Dev mode** — `/dev on` reveals the developer surface (the deep-research
  queue, the Docker/VHDX setup tab, the raw server status). Off by default.
- **Chat voices** — every message says who produced it: the model, the studio
  encoder, the **Command Console** (cyan), and you (black on yellow, no label
  — you know you typed it).
- **The launch console** — the dials a run actually needs, with one tab per
  engine: the shared ones (the context window, the port) above, then
  **llama.cpp** (the model, the GPU layers, the auto-fit, the devices, the
  threads, the slots, the KV cache, the batch sizes, the alias) or
  **Strata** (the config profile, the checkout, the VRAM reserve, the
  prefill, the spec depth, the extra engine flags). Start launches the
  engine the tab names.
- **Stop and Unload, two different things** — **Stop** (in the chat) closes the
  provider connection, so the engine stops generating at once instead of
  finishing a reply nobody is waiting for; both llama.cpp and strata stop on
  the disconnect, and a closed tab stops the turn too. **Unload** (in the
  Launch panel) releases the model itself: the local llama.cpp instances and
  the Strata engine give their VRAM back, and the profiles stay.
- **The load bar** — while a model loads or unloads, a bar across the bottom
  of the screen says what is happening (and, when a load fails, why it
  failed). A successful unload says so in a toast.
- **Model library** — search Hugging Face for GGUF repos, list files, download
  with progress, manage the local folder.
- **Engines** — reusable profiles (model, context, GPU layers, sampling, provider
  reference) instead of re-typing launch settings each time.
- **Provider references** — engine profiles point at a provider by name; raw
  credentials live in exactly one place.
- **Sandbox** (`/v1/sandbox/exec`) — Docker-backed code execution (network off,
  512 MiB / 1 CPU, hard timeout) with a local-subprocess fallback.
- **Escalation lane** — every ladder-resolved item is harvested to
  `escalation.jsonl` next to the report: task, draft, findings, repair, and the
  resolution. Training-side only; the runtime never reads it.
- **Dev server** — run it on a port you and your agent can both open, and
  iterate on the UI together.

### API surface (97 operations across 88 paths under `/v1/`)

| Area | Endpoints |
|---|---|
| Cascade | `/v1/cascade/run`, `/runs`, `/report/{run}`, `/registry`, `/roles`, `/paths`, `/plan`, `/frontier`, `/validate`, `/validator` |
| Models | `/v1/models/local`, `/v1/models/local/path`, `/v1/models/linux`, `/v1/models/hub`, `/v1/models/hub/files/{repo}`, `/v1/models/hub/download`, `/v1/models/hub/downloads` |
| Servers | `/v1/server/start`, `/stop`, `/unload`, `/unload-loaded`, `/status`, `/log`, `/memory`, `/metrics`, `/fit` |
| Engines | `/v1/engines`, `/v1/engines/preset`, `/v1/engines/bench`, `/v1/engines/ab/bench` |
| Providers | `/v1/provider/config` |
| Agent | `/v1/agent/stream`, `/status`, `/cancel`, `/v1/agent-presets/*` |
| Commands | `/v1/commands`, `/v1/commands/run` |
| Studio | `/v1/studio/settings`, `/v1/studio/strata` |
| Compaction | `/v1/compaction/run`, `/surgery`, `/status` |
| Sandbox | `/v1/sandbox/exec`, `/v1/sandbox/status` |
| Chat | `/v1/splinter/stream`, `/v1/chat/stop` |
| Strata | `/v1/strata/configs`, `/v1/strata/launch` |
| Stacks | `/v1/stacks`, `/v1/stacks/{name}`, `/apply`, `/unload`, `/validate`, `/v1/stacks/status` |
| Runs | `/v1/runs`, `/v1/report/{run}`, `/v1/protocol/run` |
| Training | `/v1/training/engines`, `/launch`, `/status/{run}`, `/report/{run}`, `/v1/trainer/*` |
| Conversation | `/v1/splinter/*` (turn, stream, reset, mode), `/v1/openai/*` |

Live, always-current OpenAPI docs: `http://127.0.0.1:8765/docs`.

**Hand-maintained endpoint map:** [`API.md`](API.md) — every endpoint
grouped by what it is for, with the conventions (auth, revisions, secrets).
A test compares it with the running app, so the map cannot drift.

## Running the suite

Offline, no LLM required (CI runs exactly this):

```
scripts/bootstrap_test_env.sh          # venv + requirements-dev.txt (once)
pytest tests/unit tests/integration
# or by group:
python tests/run_hive_tests.py --group speed|intelligence|skills|maximum
```

The integration tests need the **`splinter/` project** (the system under
test) and the **real encoder**; `scripts/bootstrap_test_env.sh` plus the
project's own install covers both, and the fixture corpora are generated by
the tests themselves if they are missing.

| Group | Measures |
|---|---|
| `speed` | latency / PES thresholds |
| `intelligence` | retrieval / assembly quality |
| `skills` | pipeline / backends |
| `maximum` | full suite (default) |

## Try it live

The live benchmark talks to an OpenAI-compatible backend (e.g. LM Studio on
`localhost:1234`). A quick resumable iteration run, from this checkout:

```powershell
.\.venv\Scripts\python -m experiments.generate_data --live --no-thinking --confidence off --max-convs 3 --max-turns 10
```

## Console scripts

| Script | Entry point |
|---|---|
| `hivebench` | `experiments.generate_data:main` |
| `hivebench-protocol` | `experiments.run_p1_p10:main` |
| `hivebench-probe` | `experiments.model_probe:main` |
| `hivebench-diagnostic` | `experiments.retrieval_diagnostic:main` |
| `hivebench-gate-ab` | `experiments.confirmation_gate_ab:main` |
| `hivebench-compare` | `experiments.run_compare:main` |
| `hivebench-ab` | `experiments.paired_ab:main` |
| `hivebench-harness` | `harness.__main__:main` |

## Tree

| Path | What it is |
|---|---|
| `experiments/` | data generation, probes, A/B batteries, finetune runs (console scripts above) |
| `testing/` | ablation / AB test tooling |
| `tests/` | unit + integration suite for the splinter system — run from here |
| `harness/` | HiveBench Studio sidecar (`python -m harness`) |
| `vendor/deepseek_harness/` | vendored dsh Python SDK (the agent bridge; re-based with `scripts/revendor_dsh.py`) |
| `vendor/deepinfer/` | vendored DeepInfer engine (deepseek-ai/DeepInfer; re-vendored with `scripts/revendor_deepinfer.py`) |
| `vendor/deepinfer_amd/` | our AMD (ROCm) lane for it — the platform, the torch-native kernels, the HIP build |
| `tools/deepinfer/serve.py` | the OpenAI-compatible facade over DeepInfer's offline engine |

## The engine lane: DeepInfer on AMD

DeepInfer is DeepSeek's own inference framework. It ships for NVIDIA
SM100 and Huawei Ascend 950, it has no HTTP server, and its kernels are
Blackwell TileLang/DeepGEMM kernels with FP8 weights and FP4 experts.

This repo carries an **AMD (ROCm) lane** for it:

- `vendor/deepinfer_amd/` replaces the platform, the device discovery and
  the CUDA-only packages (`deep_gemm`, `tile_kernels`, `flash_mla`,
  `deep_ep`, `deep_select`, `flash_attn`, `tilelang`) with torch-native
  versions, and builds DeepInfer's C++ core with `hipcc`;
- `tools/deepinfer/serve.py` wraps DeepInfer's offline API in the
  OpenAI-compatible surface the harness (and the dsh routes) speak, so a
  DeepInfer tier is a tier like any other;
- `harness/stack/deepinfer.py` owns its lifecycle the way the Strata
  launcher does, and `stacks/deepinfer-v41.json` is the stack.

Measured on 2× RX 7900 XT (gfx1100), ROCm 7.2.4: the C++ core builds and
loads, the platform reports `amd`, the V4.1 model executor imports, and
the whole engine initialises.

The honest limit: **DeepSeek-V4.1-Flash cannot run on gfx1100** — its
checkpoint is FP8 with FP4 experts and RDNA3 has no FP8. The lane's target
is a bf16 model without the mHC head, which is what our own ternary
models are. The measurements, the refusals and the remaining work are in
[`../docs/DEEPINFER-AMD-PORT.md`](../docs/DEEPINFER-AMD-PORT.md).

## Where we're going

- **The cascade in the chat path** — the cascade is a batch evaluation tool
  today. Making it the always-on chat backend needs one thing first: a
  measurement on *conversational* tasks that says it actually helps.
- **A trained encoder** — the app assistant is the dsh agent plus the
  Studio's own API map (prompt-context, v1). A dedicated small model tuned
  for app navigation needs a usage-harvest → dataset → train pipeline.
- **A file explorer and diff view** — OpenCode-style side panels; the model
  can already read a file (`/import`), you cannot yet browse one.
- Live batteries stay live-gated in CI (skipped without a backend); threshold
  benchmarks stay out of CI on purpose — shared runners flake them.

The system under test, its white paper, and the measured-outcome table behind
every claim: the [Splinter Memory](../splinter/README.md) project.

## License

MIT. See [`LICENSE`](../LICENSE). Copyright (c) 2026 HiveBench Studio contributors.
