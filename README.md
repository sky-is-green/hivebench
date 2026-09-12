# HiveBench

Evaluation suite + Studio sidecar harness for the
[Strata Memory](https://github.com/sky-is-green/strata-memory) system.

| Tree | What it is |
|---|---|
| `experiments/` | data generation, probes, A/B batteries, finetune runs (console scripts below) |
| `testing/` | ablation / AB test tooling |
| `tests/` | unit + integration suite for the strata system — run from here |
| `harness/` | HiveBench Studio sidecar (`python -m harness`) |
| `vendor/deepseek_harness/` | vendored dsh Python SDK (agent bridge; not an installed package) |

## Install

The checkouts live side by side:

```
~/Desktop/work/strata-memory   # the system
~/Desktop/work/hivebench       # this repo
```

```bash
pip install -e ../strata-memory   # from here, first
pip install -e .
```

Flat import names are preserved (`experiments.*`, `testing.*`, `tests.*`,
`harness.*`, `deepseek_harness`); the system under test resolves as `strata` /
`cortex` / ... from the sibling checkout. Override its location with
`$STRATA_HOME`.

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

## Running the suite

```bash
pytest tests/unit tests/integration
# or by group:
python tests/run_hive_tests.py --group speed|intelligence|skills|maximum
```
