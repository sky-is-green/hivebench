# Local LLM gateway (LiteLLM)

One OpenAI-compatible control plane in front of the hive's model fleet,
operated locally on **:4000**, with a Postgres-backed control surface
(virtual keys, budgets, spend, dashboard). This is the provider-grade plumbing
layer: provider fan-out, retries, error fallbacks, governance. Quality-gated
escalation stays in `harness/cascade` — a gateway routes on availability, not
correctness.

## Start

```sh
./start.sh                      # foreground on :4000
GATEWAY_PORT=4001 ./start.sh    # another port
```

`start.sh` ensures Postgres is running, exports `DATABASE_URL` and the
OpenCode key for the proxy process only; secrets never enter the config.

## PostgreSQL (userspace, no root)

Postgres comes from the `pgserver` package's bundled binaries and runs on
**127.0.0.1:5433** with its cluster under
`~/.local/share/hivebench-litellm/pgdata`:

```sh
.venv/bin/python pg.py                     # ensure running; print URI
.venv/bin/python pg.py --sql "CREATE DATABASE litellm"
.venv/bin/python pg.py --uri-db litellm    # postgresql://postgres@127.0.0.1:5433/litellm
.venv/bin/python pg.py --stop
```

One-time after installing/upgrading LiteLLM (the proxy needs the generated
client; `PATH` must include the venv so the Python generator is found):

```sh
PATH="$PWD/.venv/bin:$PATH" .venv/bin/prisma generate \
  --schema=.venv/lib/python3.12/site-packages/litellm/proxy/schema.prisma
```

## Model groups (`config.yaml`)

| group | upstream | notes |
|---|---|---|
| `scion` | `http://127.0.0.1:1234/v1` | the hivebench stack tier (`scion-35b-cascade`) |
| `flash` | `https://opencode.ai/zen/go/v1` | opencode-go `deepseek-v4.1-flash`; sends `x-opencode-session`; costs declared per deployment |

Router settings: `simple-shuffle`, `num_retries: 2`, `fallbacks: [{scion: [flash]}]`.

## Per-role virtual keys

One key per cascade role, each with a daily budget and a model allow-list.
Mint them with the master key (`sk-hive-local`):

```sh
curl -X POST localhost:4000/key/generate -H 'authorization: Bearer sk-hive-local' \
  -H 'content-type: application/json' \
  -d '{"key_alias":"cascade-router","models":["flash"],"max_budget":0.25,"budget_duration":"1d"}'
```

The live keys live outside the repo at
`~/.local/share/hivebench-litellm/keys.env` (chmod 600) as
`CASCADE_ROUTER_KEY` / `CASCADE_VERIFIER_KEY` / `CASCADE_ESCALATION_KEY`; the
pilot runner reads them per stage.

## Control plane

- **Dashboard**: <http://127.0.0.1:4000/ui/> (sign in with the master key) —
  keys, budgets, spend, request logs.
- **Per-key spend** (verified after a 3-task pilot): router $0.000523/0.25,
  verifier $0.000358/0.50, escalation $0.000266/2.00.
- `GET /key/info?key=<key>` and `GET /spend/logs` are open; the aggregate
  `/global/spend/report` is an enterprise feature.

## Use

```sh
curl localhost:4000/v1/chat/completions \
  -H 'authorization: Bearer sk-hive-local' -H 'content-type: application/json' \
  -d '{"model":"scion","messages":[{"role":"user","content":"hi"}],"max_tokens":64}'
```

Pilot runner through the gateway:

```sh
set -a; . ~/.local/share/hivebench-litellm/keys.env; set +a
CASCADE_API_URL=http://127.0.0.1:4000/v1/chat/completions CASCADE_API_MODEL=flash \
  .venv/bin/python experiments/cascade/run_cascade.py --limit 3
```

## Verified (2026-10-01)

- `scion` route → llama-server; `flash` route → the API provider.
- **Fallback**: with the stack unloaded, a `scion` request returned a
  `deepseek-v4.1-flash` answer (~6.5 s incl. retries).
- DB-backed proxy boots (Prisma migrations applied) and prices the custom
  deployment, so per-role budgets and spend are real.
- Pilot runs: `cascade-via-gateway`, `cascade-gateway-keys`, `cascade-priced-keys`.

## Known limits

- The fallback is error-based, not quality-based; the verifier gate is ours.
- Postgres runs in userspace (pgserver) — it is not a system service; start it
  with `pg.py`/`start.sh` after a reboot.
- One stack tier today; add model groups as the cascade's resident set grows.
