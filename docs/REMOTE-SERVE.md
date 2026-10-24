# Serving a model from a rented GPU (the "remote lane")

What this is: a second lane where the model runs on a GPU rented by the second, reached through the same
OpenAI-compatible endpoint contract the harness already uses for local tiers.  It exists to serve models that do
not fit here, and to give a **hardware reference** for the ones that do.

What it is not: a privacy feature.  See §3 before sending anything.

## 1. What already exists

Two halves are already in the tree; neither needs a new provider abstraction.

* **Client side — endpoint tiers.**  A stack tier with an `endpoint` is not spawned; `apply` health-checks it
  (`GET <root>/health`, then `GET <root>/v1/models`) and registers it (`harness/stack/manager.py`,
  `harness/stack/endpoint.py`).  So "point the harness at a model someone else runs" works today, for any
  OpenAI-compatible server.
* **Server side — the cloud scripts.**  `scripts/cloud/` provisions any Ubuntu + CUDA box over SSH, ships the
  repo, runs a detached stage with logging and a PID file (`run_remote.sh <name> -- <cmd>`), arms a hard
  `sudo shutdown -h +N`, enforces `MAX_COST_USD`, keeps secrets in `~/.tbr_env` (600) and has `DRY_RUN=1` for
  every script.  See `docs/CLOUD-RUNBOOK.md`.  That runbook is written for training (PTQ, distillation); the
  serving lane is the same mechanism with a different stage command.

## 2. What is missing

1. **Auth on endpoint tiers.**  `probe_endpoint` sends no headers, and the cascade client has no per-tier key, so
   a hosted endpoint that wants `Authorization: Bearer ...` fails the probe.  A tier-level `api_key` (or
   `headers`) threaded through probe and client is a small, testable change.
2. **Model staging.**  Our V4.1 weights are 163 GB.  At 1 Gbps a fresh download is ~22 minutes, at 10 Gbps ~2.2
   minutes, before the first token.  A per-run download makes short tests pointless, so the host needs either a
   persistent network volume or a pre-baked image with the weights.  Both are per-provider work; neither is
   `rsync`.
3. **A serving stage.**  `provision.sh` pins `MODEL_REPO` to the 27B teacher.  A serving profile has to start
   Strata's server (or `tools/v41/engine`'s serve mode) with an API key, a context that fits the rented card, and
   a health route the probe understands.
4. **A payload guard.**  The remote lane should refuse requests whose context is not from `data/benches/` or a
   named fixture.  This is the only part that makes the privacy stance in §3 enforceable rather than aspirational.

## 3. Privacy, plainly

The host can see every prompt, every output, and every timing.  "Obscurity" is not a control: a rented box is
someone else's computer with root on it.  The rule that actually holds is the boring one:

* send only what you would publish — benchmark corpora, public documents, synthetic prompts;
* never send private notes, credentials, or anything you could not put in a paper;
* keep private work on the home box (21.8 tok/s is slow, but it is ours).

Under that rule the lane is useful and safe.  Without it, no amount of "they probably don't log" helps.

## 4. What it buys

* **Capacity.**  Models that do not fit in 14.7 GiB of usable VRAM plus 31.7 GiB of RAM.  This is the only way to
  test a model that is bigger than the box, short of buying hardware.
* **A hardware reference.**  Running *the same engine and the same weights* on a known-good card separates "our
  box is slow" from "our engine is slow".  Today we cannot tell those apart: 24 ms of a 45 ms token is kernel
  time at ~572 GB/s, which looks near the card's practical limit, but we have no second data point.
* **Parallel sweeps.**  N configs at once instead of one at a time on the home box.

Two properties are worth keeping even if the economics never work out: the endpoint contract (already used by
local stacks) and the reference-lane use, which needs no model staging at all when the model is a small one.

## 5. Cost shape, and why "cheapest GPU" is not always cheapest

Billing is per second or per minute on most hosts, but a short run is dominated by things that are not the GPU:

* the weight download (see §2.2), unless the volume is persistent;
* the model load: 38.7 s locally for V4.1 from NVMe, and a network volume is slower than local NVMe;
* for Strata specifically, a host with **less VRAM than the model** is a legitimate configuration — that is what
  the engine does here.  A 1x80 GB card with 256 GB of RAM can run V4.1 at home-box-like speeds, and costs a
  fraction of the 4x80 GB a pure-VRAM deployment needs.  The trade is RAM bandwidth and PCIe, which is exactly what
  the home box already measures.

So the right first rental is probably not the biggest GPU; it is a cheap high-RAM box to see whether the reference
lane answers the "engine vs hardware" question at all.

## 6. First steps, in order

1. `api_key` on endpoint tiers + probe/client auth.  Can be tested against any hosted endpoint today.
2. A counting guard: the remote lane only accepts bench/fixture payloads (§2.4).
3. One manual rental, driven by hand through `scripts/cloud/`: provision, start the server, point a stack tier at
   its endpoint, run one bench, tear down.  No new automation until this works once.
4. Only then: a `serve` profile in `provision.sh` and, if it is still worth it, a provider lifecycle kind that
   creates and destroys the box.

## 7. Sequencing

The home box stays the primary lane.  It is at 21.8 tok/s with 24 of 45 ms in kernels; the remaining host time
(§ the profile in `strata-ds41/docs/ds4/V41_AMD_SPLIT.md`) is the current work.  The remote lane is for what the
box cannot do at all, and for the reference measurement above.
