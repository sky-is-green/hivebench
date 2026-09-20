# HiveBench

[![CI](https://github.com/sky-is-green/hivebench/actions/workflows/ci.yml/badge.svg)](https://github.com/sky-is-green/hivebench/actions/workflows/ci.yml)

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](https://github.com/sky-is-green/hivebench)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)](https://github.com/sky-is-green/hivebench)
[![pytest](https://img.shields.io/badge/pytest-0A9EDC?style=flat&logo=pytest&logoColor=white)](https://github.com/sky-is-green/hivebench)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://github.com/sky-is-green/hivebench)
[![Top language](https://img.shields.io/github/languages/top/sky-is-green/hivebench?style=flat&logo=python&logoColor=white)](https://github.com/sky-is-green/hivebench)
[![Last commit](https://img.shields.io/github/last-commit/sky-is-green/hivebench/main?style=flat&label=Last%20commit&logo=git)](https://github.com/sky-is-green/hivebench/commits/main)
[![Repo size](https://img.shields.io/github/repo-size/sky-is-green/hivebench?style=flat&label=Repo%20size)](https://github.com/sky-is-green/hivebench)

Evaluation suite + Studio sidecar harness for the [Strata Memory](https://github.com/sky-is-green/strata-memory) system.

**Status: work in progress.** Carved out of strata-memory on 2026-09-12 as its
own repository. The offline suite is green (546 unit + 53 integration, verified
Sep 12); the live batteries and Studio are under active development — expect
moving parts here.

## What it aims to achieve

Most evaluation harnesses tell you how a model performs in a sandbox. HiveBench
tells you *whether the context you feed the model is the reason it works*, and
it does it deterministically, offline, and replayably:

- **Falsifiable, not vibes.** The white paper's P1-P11 predictions ship as
  executable tests with measured PASS/FAIL verdicts (strata repo,
  `STRATA-WHITE-PAPER.md` section 8). Every number in the strata README is
  reproduced by a command in this repo.
- **No LLM-as-judge circularity in the evidence path.** The deterministic
  diagnostics score fact presence against fixture ground truth, stated-facts
  recall, first-mention exclusion, hedge filtering. **The Strata auditor**, an
  asynchronous ground-truth layer that labels, after each turn, whether the
  assembled context was actually sufficient for the query, corroborates that
  evidence; because it shares the served model's biases, it never constitutes
  it (white paper section 9, Threat 1).
- **The full test suite runs offline in ~30 seconds**: no LLM calls and no API
  keys; CI-friendly via `--mock`. (Running the system *live* does require a
  local model backend, LM Studio / llama.cpp, which on most rigs means a GPU;
  the drones themselves stay on CPU.)
- **Paired head-to-head A/B** (`hivebench-ab`): the same turns, the same model,
  strata-curated context vs the naive FIFO window, both answers scored
  deterministically (fixture-fact presence + context fidelity), with both
  arms' stores replaying identical history so the comparison isolates
  selection. The scoring path is unit-tested; interim live results are
  recorded per run under `runs/`.
- **Built for long evidence runs.** Checkpointed, resumable live runs survive
  crashes and reboots:

  ```powershell
  .\.venv\Scripts\python -m experiments.paired_ab --live --model prism-ml/bonsai-27b --max-turns 45 --fifo-budget 1500 --checkpoint-every 2 --output runs/paired_ab.json
  # killed mid-run? relaunch with --resume runs/paired_ab_trunc-style checkpoint,
  # or let tools/resume_evidence.ps1 loop until the final report exists.
  ```

- **Honest by design.** The suite surfaced its own failures first; the
  measurement fixes that made PES trustworthy (latency floor, stated-facts
  reframe, hedge poisoning) are documented in the paper's threats section
  (section 9), not hidden.

## Install

The checkouts live side by side:

```
~/Desktop/work/strata-memory   # the system
~/Desktop/work/hivebench       # this repo
```

Windows (PowerShell):

```powershell
git clone https://github.com/sky-is-green/strata-memory.git
git clone https://github.com/sky-is-green/hivebench.git
cd strata-memory
python -m venv .venv
.\.venv\Scripts\python -m pip install -e .   # the system: drones, cortex, retention, backends
cd ..\hivebench
python -m venv .venv
.\.venv\Scripts\python -m pip install -e .   # suite + studio (console scripts)
```

Linux/macOS: same in each checkout — `python3 -m venv .venv && .venv/bin/python -m pip install -e .`

Flat import names are preserved (`experiments.*`, `testing.*`, `tests.*`,
`harness.*`, `deepseek_harness`); the system under test resolves as `strata` /
`cortex` / ... from the sibling checkout. Override its location with
`$STRATA_HOME`.

## Run the studio (HiveBench Studio)

Two commands are the whole story, run from this checkout: `--setup` copies
`providers.example.json` → `providers.local.json` if missing, probes for a reachable
backend (LM Studio on `:1234`, or auto-starts the local `llama-server` from
`models/gguf`), and prints the next step. The studio serves the strata over a
FastAPI API; the endpoint contract lives in `harness/app.py`, and the
strata repo's `docs/INTEGRATE.md` shows how to point external clients at it.

```powershell
.\.venv\Scripts\python -m harness --setup   # copies providers config, probes backend, warms the drone
.\.venv\Scripts\python -m harness           # studio UI on http://127.0.0.1:8765
```

## Running the suite

Offline, no LLM required (CI runs exactly this):

```
pytest tests/unit tests/integration
# or by group:
python tests/run_hive_tests.py --group speed|intelligence|skills|maximum
```

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
| `tests/` | unit + integration suite for the strata system — run from here |
| `harness/` | HiveBench Studio sidecar (`python -m harness`) |
| `vendor/deepseek_harness/` | vendored dsh Python SDK (agent bridge; not an installed package) |

## Where we're going

- **S4/S5** — Studio provider row → sidecar (recall battery done; provider row
  pending UI confirm); opencode provider config → sidecar with conversation id
  = project name.
- Live batteries stay live-gated in CI (skipped without a backend); threshold
  benchmarks stay out of CI on purpose — shared runners flake them.

The system under test, its white paper, and the measured-outcome table behind
every claim: the [Strata Memory](https://github.com/sky-is-green/strata-memory) repo.
