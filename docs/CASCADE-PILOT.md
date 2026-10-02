# Cascade pilot — harness layer

The cascade is a **mixture of models by role**, not a fused model: each
request is assigned to the cheapest model that is good at it, and work is
spread over CPU, iGPU and both dGPUs.  It has two goals — cost routing (run
the smallest sufficient model) and a hallucination lever (verification that
can reject and regenerate).

The design brief lives in the research workspace (`steer-exp/CASCADE-PILOT.md`);
this repo owns the **deterministic harness layer** of it, `harness/cascade/`.

## Division of labour

| layer | lives in | examples |
|---|---|---|
| deterministic policy | `harness/cascade/` | role taxonomy, registry, paths, scheduling, judgement batching/cancellation, thresholds, telemetry |
| learned judgment | models | routers (C1), decision models (C2–C4, D1–D3, H1), generators, drafters, encoders |
| measurement / training | offline | per-role oracle, calibration, threshold fitting |
| launches and residency | `harness/stack/` | spawning llama-server tiers, validating capacity, per-tier status |

## What is implemented

- **`roles.py`** — the role taxonomy A1–H1 as data: input/output contract,
  device tier, latency budget, context cap, KV policy and candidate ids.  No
  model choice is baked in.
- **`registry.py`** — candidate catalog (repo, precision, size, roles covered)
  and `select_resident_set()`: the brief's minimal-cover consolidation within
  per-device capacity.
- **`paths.py`** — path definitions P0–P6 (cache hit, fast, standard QA,
  escalated, agentic/coding, deep reasoning, vision), with `extends` expansion
  and async/blocking step flags.
- **`judgement.py`** — the async judgement broker: batch per role, consume
  when ready, and drop pending *or late* results when a generation is
  cancelled (fire-and-consume with cancellation).
- **`policy.py`** — escalation thresholds, verify window, reasoning-budget
  cadence, escalation budget.
- **`scheduler.py`** — per-(role, device) latency model with a brief-budget
  fallback, path estimation, and expected-cost plan selection including the
  probability-weighted escalation cost.  ``quality=`` accepts the oracle's
  per-path metric once measured.
- **`telemetry.py`** — append-only per-role records + summary (the offline
  tuner's input).
- **`oracle.py`** — the per-role measurement record and Pareto-frontier
  arithmetic for pilot stage 0.

Models are **stubbed** at this layer: judgement handlers are injected, the
latency model starts from the brief's planning table, and nothing loads a
model or spawns a server.

## Use

```python
from harness.cascade import (
    default_registry, select_resident_set,
    RouteDecision, plan_request, JudgementBroker,
)

registry = default_registry()
resident = select_resident_set(
    registry, ["A1", "B2", "C1", "D2", "E2"],
    capacity_gb={"cpu": 8, "igpu": 14, "dgpu0": 20, "dgpu1": 20},
)

broker = JudgementBroker()
broker.register_handler("C1", classify_prompt)     # a model call in production
plan = plan_request(RouteDecision("qa", confidence=0.62), out_len=256)
```

## Next steps

1. **Stage 0 — per-role oracle.**  Run candidates per role through
   `harness/stack` and record `CandidateMeasurement` rows; build the per-role
   frontiers; write the measured cells back into the registry.
2. **Endpoint wiring.**  A `/v1/cascade/*` router (roles, registry, frontier,
   plan) mounted like the stack router, so the Studio and the DSH sidecar can
   drive it.
3. **Resident-set → stack.**  Turn a chosen resident set into a stack document
   and launch it through `harness/stack/manager.py`.
4. **Verifier loop (D1–D3).**  The first end-to-end path with the hallucination
   lever: generation → async verify → reject/regenerate or escalate.

## First live pilot (2026-10-01)

The first end-to-end run: **Scion summoned through the stack system** and a
frontier API tier playing the learned-judgment roles. 14 tasks with
programmatic checkers (`experiments/cascade/tasks-hard.json`), runner
`experiments/cascade/run_cascade.py` (built on this package: `plan_request`,
the judgement broker, `Policy`, `TelemetryLog`).

Setup:

```sh
# the fork binary (ternary PQ2_0 + drafter) as a named backend
ln -s /home/penis/llama.cpp/build/bin/llama-server tools/backends/scion/llama-server
./start_harness.sh                                   # sidecar on :8765
curl -X POST localhost:8765/v1/stacks/scion-35b-cascade/apply
# Scion GGUF must be in the model library (~/.lmstudio/models or models/gguf)
.venv/bin/python experiments/cascade/run_cascade.py \
    --tasks experiments/cascade/tasks-hard.json
```

The frontier tier is `opencode-go` / `deepseek-v4.1-flash` (OpenAI-compatible,
needs the `x-opencode-session` header; key read from the OpenCode auth store).

| arm | accuracy | notes |
|---|---|---|
| Scion alone | **9/14 (64%)** | 4/4 coding, 4/5 QA, 1/5 reasoning |
| API alone | **14/14 (100%)** | ceiling, $0.0015 |
| **cascade** | **13/14 (93%)** | 4 escalations, all corrected; 1 verifier false-accept |
| router agreement | 50% | it cannot see the local model's actual competence |

Escalation rate **29%**; the verifier caught **4/5** local errors with **0 false
rejects**; total API spend **$0.0071**. Scion mean 3.9 s/answer (261 completion
tokens, reasoning); API mean 2.2 s/call.

Read: local-first + verify/escalate recovers most of the gap to the ceiling for
a fraction of a cent, and the failure mode to watch is the **verifier accepting
a self-contradicting answer** (the one miss: Scion wrote "the sum is 420. The
final number is 40." and the verifier accepted 420 without flagging 40). The
router cannot predict local competence from the task alone — routing needs the
local model's measured per-bucket accuracy (the oracle), not a zero-shot guess.
The easy set (13 tasks) had Scion at 13/13 and showed only that the plumbing
works. Artifacts: `experiments/cascade/runs/<timestamp>/report.json`.

## Console contract (`/v1/cascade/*`)

The sidecar now exposes the engine over HTTP (`harness/cascade/api.py`, mounted
in `harness/app.py`).  Any console — the DSH panel, the Studio page, `curl` —
drives the same Python:

| endpoint | verb | wraps |
|---|---|---|
| `/v1/cascade/roles` | GET | the A1–H1 taxonomy |
| `/v1/cascade/paths` | GET | P0–P6 + the route table |
| `/v1/cascade/registry` | GET | candidate catalog + role coverage |
| `/v1/cascade/plan` | GET | `plan_request(bucket, confidence, out_len)` |
| `/v1/cascade/frontier` | GET | per-role Pareto fronts from an oracle JSON |
| `/v1/cascade/run` | POST | launch a pilot/oracle run (background, pid) |
| `/v1/cascade/runs` | GET | run index, newest first, with summary |
| `/v1/cascade/report/{run}` | GET | one run's `report.json` |

```sh
curl -X POST localhost:8765/v1/cascade/run \
     -H 'content-type: application/json' \
     -d '{"tasks":"tasks-hard.json","limit":3,"run_name":"my-run"}'
curl localhost:8765/v1/cascade/runs
```

## Local gateway (LiteLLM)

`tools/gateway/` runs **LiteLLM** locally on :4000 as the provider-grade
control plane: model groups `scion` (the hivebench stack tier) and `flash`
(opencode-go), retries and a `scion → flash` **error** fallback. Postgres
(userspace, via the `pgserver` binaries on 127.0.0.1:5433) backs the control
surface: **per-role virtual keys** (router / verifier / escalation) with daily
budgets, a spend ledger, request logs, and the dashboard at
<http://127.0.0.1:4000/ui/>. The runner uses the per-role keys
(`CASCADE_ROUTER_KEY` / `CASCADE_VERIFIER_KEY` / `CASCADE_ESCALATION_KEY`,
stored outside the repo at `~/.local/share/hivebench-litellm/keys.env`).
Verified: both routes; fallback with the local tier unloaded returns a flash
answer; per-key spend after a pilot (router $0.000523/0.25, verifier
$0.000358/0.50, escalation $0.000266/2.00). Quality-gated escalation stays in
`harness/cascade`; the gateway routes on availability, not correctness. See
`tools/gateway/README.md`.

## Router (C1) — first local decision model

`Tiny-Jev-1.7B` (the brief's lite Jev) now plays C1 in the pilot. It is a
non-autoregressive decision model (Qwen3-1.7B base, `trust_remote_code`): one
forward pass returns a probability distribution over typed options, no text
generation. Offline scoring (`experiments/cascade/router_eval.py`) uses the
recorded Scion/API outcomes, so router candidates are compared with no
generation reruns:

| router (14 hard tasks) | policy quality | API share | routing accuracy |
|---|---|---|---|
| always local | 0.643 | 0.00 | — |
| always API | 1.000 | 1.00 | — |
| oracle | 1.000 | 0.357 | — |
| frontier API prompt | 0.714 | 0.214 | 0.571 |
| **Tiny-Jev `choice`** | **1.000** | 0.643 | 0.714 |

Reads: the `choice` framing ("which model should handle this?" → local/api)
**separates all five Scion failures** — wrong tasks land at 0.56–0.88, the
confident-correct code tasks at 0.97+ — giving perfect policy quality zero-shot
at threshold 0.90, where the frontier router managed 0.714. It over-escalates
correct math (0.74–0.81), so the next lever is calibration on measured outcomes.
The `noul` framings ("can the local model answer this?") are miscalibrated:
probabilities cluster high and route everything local, or (rephrased) everything
API. ~180 ms/decision on CPU; 10–40 ms expected on GPU/iGPU.

**End-to-end** (`cascade-jev-router`, measure mode): cascade quality 13/14,
router agreement **0.50 → 0.714**, router API spend **$0.0026 → $0**, run cost
**$0.0071 → $0.0045 (−37%)**.

**Gate mode** (`cascade-gate-090`, threshold 0.90 — the calibrated
perfect-quality point): an API route now skips local generation *and* the
verifier, so 9/14 tasks went straight to the API and only the 5 local-routed
tasks were generated and verified (0 verifier escalations). Result: **14/14
quality** — the verifier's one false-accept (even-sum, p=0.56) was gated away —
at **$0.000743**, a **90% cut** vs the original API-router run and −83% vs
measure mode; router spend stayed $0 (verifier 5 calls, API 9). The threshold
trade-off from the 27-task calibration: 0.60 → 85% routing accuracy at 3.7%
API share; 0.75–0.80 → 0.857 quality at 22–30% share; 0.90 → 1.000 quality at
59% share. 0.90 is the zero-shot separation point; lower thresholds lean on
the verifier to catch what the router misses. Next: the 4B Jev judge so D3
goes local too.

## Judge (D3) — local verifier

`Intern-Decision-4B` (the brief's 4B Jev) now plays D3. It is a structured
decision model (Qwen3.5-4B base, masked-next-token scoring over `<decision>`
placeholders, fitted temperature T≈1.99): we ask one `noul` field — *"the
candidate answer is correct and complete"* — and threshold P(yes). The
checkpoint's own `inference.py` is used as-is (text-only; the multimodal
processor is stubbed), ~280 ms/judgement on the 7900 XT.

Offline eval on the 27-task labelled history
(`experiments/cascade/judge_eval.py`):

| judge | accept rate | false accepts | false rejects | correct verdicts |
|---|---|---|---|---|
| frontier API prompt (recorded) | 0.852 | 1 | 0 | 26/27 |
| **Intern-Decision-4B @ 0.85** | 0.704 | **0** | 3 | 24/27 |
| Intern-Decision-4B @ 0.50 | 0.889 | 4 | 2 | 21/27 |

At 0.85 the local judge is **quality-first**: no wrong answer is accepted; the
price is 3 unnecessary escalations. The API judge leaks once (its one false
accept) but wastes no escalations. Either is defensible; 0.85 is the
no-leak point.

**Scaling — does the judge want more parameters?** Same 27-task set, each
checkpoint at its own no-leak operating point (the smallest threshold with zero
false accepts):

| model | thr | accept rate | false rejects | correct verdicts | ms |
|---|---|---|---|---|---|
| Intern-Decision-0.8B | 0.75 | 0.111 | 19 | 8/27 | 122 |
| Intern-Decision-2B | 0.70 | 0.519 | 8 | 19/27 | 170 |
| **Intern-Decision-4B** | **0.85** | **0.704** | **3** | **24/27** | 310 |
| **JEV-9B (bf16)** | 0.85 | 0.852 | **0** | **26/27** | 292 |
| frontier API (generative) | — | 0.852 | 0 | 26/27 (1 leak) | — |

Monotone with diminishing returns: 0.8B → 2B buys +11 correct verdicts,
2B → 4B +5. The family's published benchmarks show the same shape
(79.4 → 84.7 → 90.0 average). The 4B is the top of the family — beyond it the
options are a *generative* local verifier (the API judge's architecture, which
can actually recompute the answer) or training our own larger judge, not
another off-the-shelf sibling.

**JEV-9B (`autotrust/JEV-9B`, bf16) matches the frontier API judge** on this
set: 26/27 correct verdicts, 1 false accept, 0 false rejects at threshold 0.85
— a direct student of TypeSafe Jev 1.13 (Qwen3.5-9B + LoRA + a 24-slot
decision head, `bare-v1` prompt, per-kind temperature). It runs locally
(18 GB bf16, split across both cards, ~292 ms/judgement) with the repo's own
transformers readout — no vLLM needed. At 0.95 it is also the no-leak point
(24/27, 3 false rejects). The judge role can now be fully local at
API-judge quality.

**Full local cascade** (`cascade-full-local`: Tiny-Jev router + Intern-Decision
judge + Scion generator, API only for the 9 router-gated answers): **14/14
quality at $0.000481**, with *no* router and *no* verifier API calls. The
progression on the same 14 tasks:

| configuration | quality | paid roles | cost |
|---|---|---|---|
| API router + API judge | 0.929 | answer + router + verifier | $0.007136 |
| local router, API judge (gate) | 1.000 | answer + verifier | $0.000743 |
| **local router + local judge (gate)** | **1.000** | **answer only** | **$0.000481** |

That is a **93% cost cut** with quality *up* (the router gated away the one
case the API verifier used to miss). Caveats: 14 tasks, single sample, and the
judge runs in-process (the ROCm venv) rather than as a served endpoint.

## Judge vs steerer — async-steering experiments (2026-10-02)

The judgement broker can fire mid-generation, so the question was what a
verdict on a *partial* generation is worth. It splits the verification roles:

- **Judge (D2/D3)** — predicate on a finished artifact: P(answer correct) at
  EOS, answer-only. JEV-9B scores 26/27, and is measurably worse when shown the
  reasoning (false accepts 1 → 3 on the full stream); the contract is now
  explicitly **answer-only**.
- **Steerer (new role D0)** — forecast on a trajectory: P(final outcome |
  prefix) at checkpoints, whose only actions are continue / cancel-to-escalate /
  force-answer. No trustworthy implementation yet, so **cancel-on-reject is not
  wired**.

Mechanics (dedicated `-np 1 --metrics` server): both cancel paths work —
closing a plain stream and `DELETE /v1/stream` stop compute in ≤200 ms
(`stop: cancel task` in the log, `requests_processing → 0`, ~35 tokens of
decode-ahead overrun, slot free in ~150 ms). The pilot stack runs 4 slots, so
per-slot cancel is fine but occupancy timing needs `-np 1`.

Streams and sensors (27 captured Scion streams, reasoning + answer; the answer
starts at a **median 92%** of the stream, so steering can only act on the
reasoning):

| sensor | AUC @16/64/128 | cancel @0.5: caught wrong / false cancels |
|---|---|---|
| JEV-9B `complete` | 0.82 / 0.81 / 0.75 | 3 / 7 |
| Intern-Decision 0.8B · 2B | 0.67 / 0.62 / 0.88 · 0.61 / 0.70 / 0.70 | 4 / 16 · 3 / 20 |
| Qwen3-1.7B, YES/NO | 0.78 / 0.60 / 0.75 | **3 / 4** |
| Qwen3.5-4B, YES/NO | 0.53 / 0.56 / 0.65 | 3 / 14 |
| Qwen3-1.7B, verbalized P | 0.28 / 0.11 / 0.64 | — |
| linear probe on Scion hidden states | 0.19 / 0.08 / 0.20 | underpowered |

Read: every sensor catches the same long, visibly confused traces and misses
the same short wrong answer. The best zero-shot sensor (1.7B, binary decision)
still false-cancels four correct answers per three caught — more escalation
cost than the ~1k local tokens it saves. **Do not wire cancel-on-reject yet.**
The binding asset is a steering dataset (~100 hard tasks × 3–5 samples with
streams, hidden dumps and checker labels — the capture pipeline exists) and a
steerer trained with an asymmetric false-cancel penalty; H1's "answer
determined" signal needs the same treatment.

Scripts: `stream_capture.py`, `stream_probe.py`, `prefix_eval.py`,
`steerer_llm_eval.py`, `steerer_compare.py`, `build_hidden_windows.py`,
`hidden_probe.py`. Artifacts: `prefix-stream-eval.json`,
`prefix-steerer-{08b,2b}.json`, `steerer-*.json`, `steerer-compare.json`,
`hidden-probe.json`, `stream-probe.json`, `streams/scion-v2-streams/`.

## Related work and prior art (2026-10-02)

Routing and cascading between independently trained LLMs is an established
field. The 2026 survey *Dynamic Model Routing and Cascading for Efficient LLM
Inference* (Moslem & Kelleher, arXiv:2603.04445) maps it along three axes:
when the decision is made (pre-generation / post-generation / multi-stage),
what signals it uses (query / model metadata / response / feedback), and how
it is computed (heuristic / supervised / bandit / RL). This harness is the
multi-stage, response+feedback, supervised+heuristic corner of that space.

Closest methods: **FrugalGPT** (router + quality estimator + cost-aware stop
judge), **AutoMix** (self-verification + POMDP stop), **Cascade Routing**
(dynamic model reordering per query), **RouteLLM** (binary win-prediction
trained on preference/outcome labels), Arch-Router / UniRoute / bandit routers
(query-side), **CALM** (per-timestep confidence early exit with calibrated
thresholds), **speculative cascades** (token-level deferral rule), **PRMs**
(step-level supervision). Provider practice: OpenAI GPT-5's continuously
trained real-time router; Azure Foundry model router (trained model with
Balanced/Cost/Quality modes and failover); Bedrock intelligent prompt routing
(quality prediction within a family, fallback model + quality-difference
criterion); OpenRouter/gateway auto-routing. Mid-stream generation control
exists in production only for safety filters (`finish_reason:
content_filter`), never for quality; thinking budgets / effort controls are
the shipped form of H1.

Historical lineage, all pre-2000: **Chow's reject option** (1970) is our judge
threshold; **Wald's SPRT** (1945) is calibrated early stopping; **boosted
cascades** (Viola–Jones 2001, on 1990s boosting) are our P-paths;
**algorithm portfolios and restarts under heavy-tailed runtimes** (Rice 1976;
Luby 1993; Gomes & Selman 2001) are our candidate pools; **mixture of experts**
(Jacobs, Jordan, Nowlan & Hinton 1991) is our role mixture; **anytime
algorithms and value of computation** (Russell & Wefald 1991; Zilberstein;
Horvitz 1988) plus **Simon's satisficing** are H1 and the quality target;
**Hearsay-II blackboard control** (1980) is our scheduler and broker;
**Minton's utility problem** (1988) says verification must pay rent; **MYCIN
certainty factors** are the historical twin of the verbalized-confidence
failure we measured. The old era stalled because there was no calibrated
per-step evidence, no cheap multiple models, no metered cost; all three now
exist.

What we borrow, concretely:

1. **SPRT/CALM boundaries** with an explicit false-cancel budget once the
   steerer has a score — replace the hand-fitted single threshold.
2. **Instrument P0–P6 as a designed cascade** (reach/reject/cost per stage,
   Viola–Jones style) and allow dynamic ordering (cascade routing).
3. **Train deferral jointly with the real cost matrix** (learning-to-defer),
   not separately thresholded models.
4. **Measure the generator runtime distribution**; test restart/interleave
   policies against single long paths (heavy tails).
5. **VOC gate**: skip the judge/steerer when expected gain < its cost
   (Minton's utility problem); no shipped cascade does this explicitly.
6. **Value-aware opportunistic broker pump** (Hearsay), not submission order.

### Strata — the local Flash-Next engine (2026)

[Strata](https://github.com/Niko1221/Strata) (MIT, ~5.7k stars, trending
September 2026) runs **Qwen3.8-Flash-Next** — the registry's `flash-next` E4
candidate — on consumer GPUs by tiering its 24,576 experts across
VRAM/RAM/CPU/SSD: attention, mixers, routers and an **adaptive expert cache**
on the GPU (it re-learns the hot experts per conversation), all experts pinned
in RAM, the rest computed in place on the CPU, and a 28.8 GB n-gram lookup
table read from the SSD. Speed comes from the model's own MTP layer (up to 3
drafted tokens, 2.4-3.2 accepted per pass) plus prompt-lookup drafts, 8K-token
prefill chunks with next-layer expert prefetch over PCIe, and optional KV
streaming / quantization (`--kv-resident`, `k8v4`). Low-RAM modes map experts
from the pack (`--mmap-experts`), keep a resident budget
(`--resident-budget-gib`), or prefetch predicted pages (`STRATA_LOOKAHEAD`).

The serving layer is as instructive as the engine: an OpenAI **and** Anthropic
compatible API, reasoning-effort levels plus a hard `reasoning_budget_tokens`
(H1's shipped form), `/metrics` with per-tier expert counts and draft
acceptance, **idle unload + min-free-VRAM guard + before-load hooks**, prompt
and conversation checkpoints, MCP in both directions, and per-box calibration
(`--calibrate`) of PCIe fraction, draft threshold and CPU pool.

What to borrow for hivebench:

1. The **idle-unload / min-free-VRAM / before-load triad** is exactly the
   arbitration our stack manager needs for two research tracks on one box
   (today one track OOMs the other).
2. **Per-tier expert telemetry** belongs in the oracle and residency math —
   our candidates are monolithic `bytes_gb` today.
3. **Per-box calibration** is an oracle primitive we already model, but
   Strata actually measures and keeps only >3% wins.
4. Their **bench/results layout** (dated run dirs, engine version, settings,
   per-length matrices, explicit caveats) matches our manifest/MODEL-CARD
   tracking; adopt the dated-dir convention.
5. **Determinism caveats**: adaptive expert tiers and speculation make greedy
   runs non-byte-identical; our eval manifests should record this and the
   reproducibility switches.

Strata is a candidate backend for the `flash-next` E4 tier — our ISTA-DASLab
Q2_0 pack is already local (62 GB on disk) — but this box has 30 GB RAM
against their 64 GB recommendation; the mapped low-RAM mode with both 20 GB
cards is the untested path, and it is the same model the flash-next research
track is exercising.

This is the same problem as **project Ember** (`docs/EMBER.md`): our own
HIP/ROCm expert-residency engine (`--moe-slot-cache`,
`GGML_OP_MOE_CACHE_MAP`) already runs the same 125B Q2_0 pack through the
stack API and measures the same regime on this box (all-resident 26.9 t/s;
8.1 GiB off-VRAM + 4.1 GiB slots → 24.5 t/s, +27%, ~88% hits; 24.3 GiB
off-VRAM + slots → 11.5-15.4 t/s). Strata adds what Ember's open list is
missing: CPU co-execution of misses instead of a PCIe gather, MTP and
prompt-lookup speculation, 8K prefill chunks with layer-ahead prefetch, KV
streaming and k8v4, adaptive per-conversation caches, and the serving/product
surface. Ember's advantage: it is ours, HIP-native, measured on this exact
box, and already wired through `stacks/ember-*.json`.

## Priority plan after the literature pass

1. **Evaluation floor first**: held-out split plus a real benchmark slice
   (GSM8K / MATH / HumanEval / TruthfulQA, local) at 3–5 samples per task;
   thresholds fitted on train, reported on test. Nothing above is credible
   without this.
2. **Dataset asset**: capture ~100 hard tasks × 3–5 samples with streams,
   hidden dumps and checker labels (the `stream_capture` +
   `build_hidden_windows` pipeline); it feeds router training, steerer
   training, calibration and the benchmarks.
3. **Router (C1)**: train/calibrate on accumulated outcome labels
   (RouteLLM-style binary win prediction) instead of the zero-shot Tiny-Jev
   threshold.
4. **Judge (D2/D3)**: cost-sensitive operating point from the actual
   escalation-vs-wrong cost ratio (Chow), fitted on held-out data; keep
   answer-only.
5. **Steerer (D0)**: trained probe (text or hidden state) with an asymmetric
   false-cancel loss and SPRT/CALM boundaries; only then wire
   cancel-on-reject. H1's "answer determined" signal rides the same probe.
6. **Policy upgrades**: cascade instrumentation and dynamic ordering, the VOC
   gate, value-aware pump; then serve the decision models as endpoints and
   build the DSH console.

## Results tracking

Benchmark runs land under `experiments/cascade/results/<run>/`: `manifest.json`
(provenance — model sha256, stack, git commit, task-set sha, decode settings —
plus per-task and per-benchmark aggregates with sample-accuracy and
task-macro), `streams.json` (full traces, regenerable), and a generated
`MODEL-CARD.md` for the model page. `results/index.json` indexes runs by model
sha and task-set sha. Manifests and cards are tracked in git; the large stream
traces are ignored.

## Related

- `HIVE-PLAN.md` (local-only) — track plan and task rows.
- `LOCAL-STACKS.md` (local-only) — model-stack mechanics, KV math.
- DeepSeek Harness (DSH) side — the agent bridge in `harness/agent.py`; the
  fork's sidecar/engine packages host the models the cascade picks from.
