# Training tab — implementation plan (slice 1)

Status: slice 1 in progress. Scope agreed with the project lead: **engine switch +
Hive-Ternary launch/monitor**, Unsloth Core declared but not yet launchable.

## Names

| Name | What it is |
|---|---|
| **TBR** | the wire contract (`bonsai_forensics`-style spec, `tbr-1.2`, `SPEC_SHA256`) — unchanged |
| **Hive-Ternary** | the engine that implements TBR: rotation, STE, KD, GGUF pack, eval |
| **Training** | the Studio tab that exposes both engines |

## Engines

```
Training
 ├─ Unsloth Core   (external, Apache-2.0 `unsloth` package; depend on it, never vendor it)
 └─ Hive-Ternary   (native; implements the TBR contract)
```

- **Unsloth Core** is declared in the catalogue with `implemented=False` for slice 1.
  It is consumed as a pinned dependency (`unsloth==2026.9.7`) in the Unsloth ROCm
  env. The AGPL-3.0 Studio UI is never vendored or copied.
- **Hive-Ternary** is launchable. Its runner is resolved, in order, from:
  1. `$HIVE_TERNARY_RUNNER` (a `.py` path or a `module.path`),
  2. `<repo>/hive_ternary/train.py` (the future in-repo home),
  3. the current forensics harness (temporary bridge, documented).

  The interpreter is `$HIVE_TERNARY_PYTHON`, else
  `~/.unsloth/studio/unsloth_studio/bin/python`, else `sys.executable`.

## Recipes (Hive-Ternary, slice 1)

| id | label | flags |
|---|---|---|
| `ste` | Ternary STE (plain) | `--ste` |
| `ste-rotate` | Rotation + STE | `--ste --rotate` |
| `ste-rotate-lsq` | Rotation + STE + LSQ | `--ste --rotate --learn-scale` |
| `ste-rotate-kd` | Rotation + STE + KD | `--ste --rotate --lam 0.2` |

`pack` and `eval` are declared with `implemented=False`.

## Run bundles

Mirrors `POST /v1/protocol/run`: a directory under `runs_root` with

- `training.json` — engine, method, argv, config, spec pin,
- `run_stdout.log` — streamed stdout (progress lines are `[rmd] step …`),
- `run_report.json` — written by `harness.training.write_run_report`, `kind: "training"`,
  consumed by the existing `/runs` registry.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/v1/training/engines` | engine + recipe catalogue, resolved runner/python |
| POST | `/v1/training/launch` | build argv, create run dir, spawn detached |
| GET | `/v1/training/status/{run}` | parsed step / loss / ratio / elapsed |
| POST | `/v1/training/report/{run}` | build + persist the bridged `run_report.json` |

## UI

- A `Training` button joins the existing `.tabs` bar; its pane is rendered by
  `harness.reports.render_training_pane()` (self-contained, wires its own click
  listener so the big tab script is untouched).
- Results appear in `/runs` and `/view/{run}` via `render_report_page`, which
  dispatches on `report["kind"] == "training"`.

## Deliberate non-goals (slice 1)

- No Batch control: the ternary harness trains one 512-token window per step.
- No Unsloth launch path yet (declared only).
- No GPU work: launching is opt-in and never triggered by the tests.
