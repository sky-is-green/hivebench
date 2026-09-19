# TBR Cloud Runbook (T12)

Rental procedure for the 27B runs: PTQ base (T13), KD/logit cache (T17), recovery
(T18/T19), evaluation (T20). Scripts live in `scripts/cloud/`; every step is
provider-agnostic (any Ubuntu + CUDA box reachable over SSH).

## 1. Instance requirements

| Item | Value |
|---|---|
| GPU | 1x 80 GB (H100 preferred, A100 fallback). Spot is fine: per-layer checkpoints survive preemption |
| Disk | >= 200 GB ephemeral (55.6 GB BF16 checkpoint + captures + artifacts) |
| Image | Ubuntu 22.04+, CUDA driver >= 12.4, passwordless `sudo` (auto-shutdown) |
| Network | HF access for the pinned teacher download; SSH from the workstation |

## 2. Environment contract (workstation)

```sh
export HOST=user@instance-ip          # required
export SSH_KEY=~/.ssh/id_ed25519      # optional
export SSH_PORT=2222                  # optional
export HF_TOKEN=hf_...                # required; never committed/logged
export MAX_HOURS=6                    # hard auto-shutdown cap
export PRICE_PER_HOUR=2.20            # optional, enables cost math
export MAX_COST_USD=60                # optional; effective hours = min(MAX_HOURS, cap/price)
```

Secrets policy: `HF_TOKEN` is passed over stdin to `~/.tbr_env` (mode 600) on the
instance, sourced by each stage, and deleted by `teardown.sh`. It is never in the
repo, argv, or logs. Rotate the token after any rental.

## 3. Procedure

```sh
# 1. provision: ship repo, venv, pinned model, token file, arm auto-shutdown
scripts/cloud/provision.sh

# 2. PTQ base + calibration arms (T13-T15); start detached, watch the log
scripts/cloud/run_remote.sh ptq -- python -m experiments.ternary.run_quant \
    --config configs/ternary/27b-A.yaml --source safetensors \
    --model-dir models/Qwen3.8-27B
scripts/cloud/status.sh ptq

# 3. KD corpus + teacher top-k logits cache (T17), once
scripts/cloud/run_remote.sh kd -- python -m experiments.ternary.kd_data \
    --config configs/ternary/kd.yaml --model-dir models/Qwen3.8-27B
scripts/cloud/status.sh kd

# 4. recovery iterations (T18/T19): merge -> re-ternarize -> next iteration
scripts/cloud/run_remote.sh recover -- python -m experiments.ternary.recover \
    --model-dir models/Qwen3.8-27B --corpus <kd-corpus> \
    --out artifacts/ternary/recovered/run1 --resume

# 5. same-binary KLD/PPL against the Bonsai PQ2_0 reference (T20)
scripts/cloud/run_remote.sh eval -- python -m experiments.ternary.eval_kld ...

# 6. collect + stop
scripts/cloud/pull.sh                       # rsync artifacts/ -> artifacts/cloud/
scripts/cloud/teardown.sh                   # deletes token, cancels timer, cost report
STOP_NOW=1 scripts/cloud/teardown.sh        # power off when finished
```

`DRY_RUN=1` prints the full plan/commands without touching the instance:
`DRY_RUN=1 HOST=... HF_TOKEN=... scripts/cloud/provision.sh`.

## 4. Cost policy

- `MAX_HOURS` is armed on the instance with `sudo shutdown -h +N` at provision
  time — the machine dies at the cap even if every script fails to run.
- `MAX_COST_USD` / `PRICE_PER_HOUR` shrink the cap before scheduling.
- Budgets (`HIVE-PLAN.md` §11): PTQ <= 6 h / $60, recovery <= 12 h / $150.
- `teardown.sh` reports elapsed time and the estimated cost; verify against the
  provider's billing page before destroying nothing.

## 5. Failure handling

- PTQ and recovery checkpoint per tensor / per `save_every` steps; re-run the
  same stage with `--resume` after a preemption.
- Model/logit caches live under `artifacts/` and are pulled by `pull.sh`, so a
  destroyed instance never forces a re-download of teacher logits.
- If the auto-shutdown fired mid-stage, `provision.sh` re-arms it on the next
  instance; check `runs/<name>.pid` staleness with `status.sh`.

## 6. Acceptance checklist (T12)

- [ ] `DRY_RUN=1` plan for every script, no instance touched
- [ ] secrets: only `~/.tbr_env` (600), deleted by teardown
- [ ] `MAX_HOURS`/`MAX_COST_USD` reconciled before arming shutdown
- [ ] shell syntax clean (`bash -n` on every script; run shellcheck once the
      tool is installed on the workstation)
- [ ] QUEEN sign-off
