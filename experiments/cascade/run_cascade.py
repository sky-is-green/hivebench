#!/usr/bin/env python
"""Cascade pilot v1 — Scion (hivebench stack tier) + a frontier API tier.

The first real exercise of ``harness/cascade``: Scion is *summoned through
hivebench* (the ``scion-35b-cascade`` stack, spawned by the stack manager),
and the frontier API model plays the cascade's learned-judgment roles —
C1 router, D3 verifier, E4 escalation.  The deterministic layer under test is
the package built this session: path planning, the async judgement broker
(batched per role, generation-cancellable), the escalation policy and the
telemetry log.

Per task, four measurements:

1. ``local``      — Scion answers alone.
2. ``api``        — the frontier model answers alone (the ceiling).
3. ``router``     — the frontier model predicts whether Scion can answer.
4. ``cascade``    — Scion answers, the frontier verifier accepts or rejects,
                    a rejection escalates to the frontier answer.

Artifacts: ``experiments/cascade/runs/<timestamp>/report.json``.

Usage::

    .venv/bin/python experiments/cascade/run_cascade.py --limit 3
    .venv/bin/python experiments/cascade/run_cascade.py            # full set
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import requests

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from harness.cascade import (  # noqa: E402
    Judgement,
    JudgementBroker,
    Policy,
    RouteDecision,
    TelemetryLog,
    default_registry,
    plan_request,
    result_for,
)
from harness.cascade.registry import CandidateRegistry  # noqa: E402
from harness.cascade.policy import verifier_rejects  # noqa: E402

from experiments.cascade.checkers import check, extract_json  # noqa: E402

# ---------------------------------------------------------------- providers

SIDECAR = os.environ.get("CASCADE_SIDECAR", "http://127.0.0.1:8765")
SCION_PORT = int(os.environ.get("CASCADE_SCION_PORT", "1234"))
SCION_BASE = os.environ.get("CASCADE_SCION_BASE", f"http://127.0.0.1:{SCION_PORT}/v1")
API_URL = os.environ.get(
    "CASCADE_API_URL", "https://opencode.ai/zen/go/v1/chat/completions"
)
API_MODEL = os.environ.get("CASCADE_API_MODEL", "deepseek-v4.1-flash")
AUTH_PATH = Path.home() / ".local" / "share" / "opencode" / "auth.json"
PRICING = {"input": 0.15 / 1e6, "output": 0.60 / 1e6}  # USD per token

SCION_SYSTEM = (
    "You are a precise assistant. Think if you need to, then give the final "
    "answer. For code questions return only the requested Python function in a "
    "single code block."
)
ANSWER_SYSTEM = (
    "You are a precise assistant. Answer the user's task. For code questions "
    "return only the requested Python function in a single code block."
)
ROUTER_SYSTEM = (
    "You route requests in a local-first cascade. LOCAL is a 35B-parameter "
    "ternary-quantized mixture-of-experts model running on consumer hardware: "
    "fast and cheap, but weak on multi-step reasoning and precise coding. API "
    "is a frontier model: slower and paid, much stronger on those tasks. "
    'Decide whether LOCAL can plausibly answer the task correctly without '
    'tools. Reply with JSON only: {"route":"local"|"api","confidence":0-1,'
    '"reason":"<short>"}'
)
VERIFIER_SYSTEM = (
    "You are a strict answer verifier. You receive a TASK and a CANDIDATE "
    "answer. Decide whether the candidate is correct and complete; check the "
    "work, do not just trust its confidence. Reply with JSON only: "
    '{"verdict":"accept"|"reject","confidence":0-1,"reason":"<short>"}'
)


class Usage:
    """Token + cost accounting, split by stage tag."""

    def __init__(self) -> None:
        self.by_tag: dict[str, dict[str, float]] = {}

    def add(self, tag: str, usage: Optional[dict]) -> None:
        if not usage:
            return
        row = self.by_tag.setdefault(tag, {"calls": 0, "input": 0.0, "output": 0.0})
        row["calls"] += 1
        row["input"] += float(usage.get("prompt_tokens") or 0)
        row["output"] += float(usage.get("completion_tokens") or 0)

    def cost(self, tag: Optional[str] = None) -> float:
        rows = [self.by_tag[tag]] if tag else list(self.by_tag.values())
        return sum(
            r["input"] * PRICING["input"] + r["output"] * PRICING["output"] for r in rows
        )

    def summary(self) -> dict[str, Any]:
        return {
            tag: {**row, "cost_usd": round(self.cost(tag), 6)}
            for tag, row in sorted(self.by_tag.items())
        } | {"total_cost_usd": round(self.cost(), 6)}


def _post(url: str, payload: dict, headers: dict, timeout: int) -> tuple[dict, float]:
    t0 = time.time()
    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    elapsed = (time.time() - t0) * 1000.0
    if resp.status_code >= 400:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
    return resp.json(), elapsed


def _api_key() -> str:
    data = json.loads(AUTH_PATH.read_text(encoding="utf-8"))
    return data["opencode-go"]["key"]


class Frontier:
    """The paid tier: OpenAI-compatible endpoint, retries, usage accounting."""

    def __init__(self, usage: Usage, session: Optional[str] = None) -> None:
        self.usage = usage
        self.session = session or f"cascade-pilot-{uuid.uuid4().hex[:12]}"
        # A gateway (LiteLLM) in front of the provider takes its own bearer
        # token; fall back to the OpenCode auth store when calling direct.
        self.key = os.environ.get("CASCADE_API_KEY") or _api_key()
        # Per-role virtual keys: route / verify / escalate each get their own
        # spend line at the gateway.  The call tag names the role.
        self.role_keys = {
            "router": os.environ.get("CASCADE_ROUTER_KEY", ""),
            "verifier": os.environ.get("CASCADE_VERIFIER_KEY", ""),
            "answer": os.environ.get("CASCADE_ESCALATION_KEY", ""),
        }

    def chat(self, messages: list[dict], *, tag: str, max_tokens: int) -> dict:
        payload = {
            "model": API_MODEL,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        headers = {
            "authorization": f"Bearer {self.role_keys.get(tag) or self.key}",
            "x-opencode-session": self.session,
            "content-type": "application/json",
            "user-agent": "curl/8.9.1",
        }
        last: Optional[Exception] = None
        budget = max_tokens
        for attempt in range(3):
            try:
                data, ms = _post(API_URL, payload | {"max_tokens": budget}, headers, timeout=180)
                self.usage.add(tag, data.get("usage"))
                choice = data["choices"][0]
                msg = choice["message"]
                content = msg.get("content") or ""
                finish = choice.get("finish_reason")
                time.sleep(0.2)
                if content.strip() or finish != "length":
                    return {
                        "content": content,
                        "reasoning": msg.get("reasoning_content") or "",
                        "latency_ms": ms,
                        "finish_reason": finish,
                        "tag": tag,
                    }
                budget *= 2  # reasoning consumed the whole budget: retry bigger
            except Exception as exc:  # noqa: BLE001 - retry then surface
                last = exc
                time.sleep(1.5 * (attempt + 1))
        if last is not None:
            raise RuntimeError(f"frontier call failed ({tag}): {last}")
        return {"content": "", "reasoning": "", "latency_ms": 0.0,
                "finish_reason": "length", "tag": tag}


class ScionTier:
    """The local tier: the llama-server hivebench spawned for the stack."""

    def __init__(self, base: str = SCION_BASE) -> None:
        self.base = base.rstrip("/")

    def alive(self) -> bool:
        try:
            r = requests.get(f"{self.base}/models", timeout=5)
            return r.status_code == 200
        except Exception:  # noqa: BLE001
            return False

    def chat(self, messages: list[dict], *, max_tokens: int = 768) -> dict:
        payload = {
            "model": "scion",
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        data, ms = _post(
            f"{self.base}/chat/completions", payload, {"content-type": "application/json"},
            timeout=600,
        )
        msg = data["choices"][0]["message"]
        content = msg.get("content") or ""
        usage = data.get("usage") or {}
        # A reasoning model that ran out of budget before answering: retry once.
        if not content.strip() and int(usage.get("completion_tokens") or 0) >= max_tokens:
            payload["max_tokens"] = 2048
            data, ms = _post(
                f"{self.base}/chat/completions", payload,
                {"content-type": "application/json"}, timeout=900,
            )
            msg = data["choices"][0]["message"]
            content = msg.get("content") or ""
            usage = data.get("usage") or {}
        return {
            "content": content,
            "reasoning": msg.get("reasoning_content") or "",
            "latency_ms": ms,
            "usage": usage,
        }


# ---------------------------------------------------------------- pipeline


def answer_prompt(task: dict) -> list[dict]:
    return [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": task["prompt"]},
    ]


def route_prompt(task: dict) -> list[dict]:
    return [
        {"role": "system", "content": ROUTER_SYSTEM},
        {"role": "user", "content": task["prompt"]},
    ]


def verify_prompt(task: dict, candidate: str) -> list[dict]:
    return [
        {"role": "system", "content": VERIFIER_SYSTEM},
        {"role": "user", "content": f"TASK:\n{task['prompt']}\n\nCANDIDATE:\n{candidate}"},
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="first N tasks only")
    parser.add_argument("--tasks", default=str(Path(__file__).with_name("tasks.json")))
    parser.add_argument(
        "--out", default=str(REPO / "experiments" / "cascade" / "runs"),
        help="artifact root",
    )
    parser.add_argument(
        "--run-name", default="",
        help="explicit run directory name under --out (the API pre-allocates it)",
    )
    parser.add_argument("--scion-base", default=SCION_BASE)
    parser.add_argument(
        "--router", choices=("api", "jev"), default="api",
        help="C1 router: the frontier API prompt (default) or local Tiny-Jev",
    )
    parser.add_argument(
        "--router-threshold", type=float, default=0.90,
        help="Tiny-Jev: route local when P(local answers correctly) >= this",
    )
    parser.add_argument("--router-model", default="lostargon/Tiny-Jev-1.7B")
    parser.add_argument(
        "--judge", choices=("api", "local"), default="api",
        help="D3 judge: frontier API prompt (default) or local Intern-Decision-4B",
    )
    parser.add_argument("--judge-model", default="internlm/Intern-Decision-4B")
    parser.add_argument("--judge-device", default="cuda")
    parser.add_argument(
        "--judge-threshold", type=float, default=0.5,
        help="accept the candidate when P(correct) >= this",
    )
    parser.add_argument(
        "--router-mode", choices=("measure", "gate"), default="measure",
        help="measure: record the route but always generate locally; "
             "gate: an API route skips local generation and the verifier",
    )
    args = parser.parse_args()

    spec = json.loads(Path(args.tasks).read_text(encoding="utf-8"))
    tasks = spec["tasks"][: args.limit] if args.limit else spec["tasks"]

    scion = ScionTier(args.scion_base)
    if not scion.alive():
        print(
            f"Scion tier not reachable at {scion.base} — apply the "
            "'scion-35b-cascade' stack through hivebench first "
            "(POST /v1/stacks/scion-35b-cascade/apply).",
            file=sys.stderr,
        )
        return 2

    usage = Usage()
    frontier = Frontier(usage)
    telemetry = TelemetryLog()
    policy = Policy(accept_confidence=args.judge_threshold)
    broker = JudgementBroker()

    def router_handler(batch):
        results = []
        for j in batch:
            task = j.payload["task"]
            try:
                out = frontier.chat(route_prompt(task), tag="router", max_tokens=120)
                parsed = extract_json(out["content"]) or extract_json(out["reasoning"]) or {}
                route = str(parsed.get("route", "")).lower()
                if route not in ("local", "api"):
                    route = "api"
                results.append(
                    result_for(
                        j, route, float(parsed.get("confidence") or 0.0),
                        latency_ms=out["latency_ms"],
                        payload={"reason": parsed.get("reason", "")},
                    )
                )
            except Exception as exc:  # noqa: BLE001
                results.append(
                    result_for(j, "api", 0.0, payload={"error": str(exc)[:200]})
                )
        return results

    def verify_handler(batch):
        results = []
        for j in batch:
            task = j.payload["task"]
            try:
                out = frontier.chat(
                    verify_prompt(task, j.payload["candidate"]),
                    tag="verifier", max_tokens=400,
                )
                parsed = extract_json(out["content"]) or extract_json(out["reasoning"]) or {}
                verdict = str(parsed.get("verdict", "")).lower()
                if verdict not in ("accept", "reject"):
                    verdict = "reject"  # unparseable = do not trust
                results.append(
                    result_for(
                        j, verdict, float(parsed.get("confidence") or 0.0),
                        latency_ms=out["latency_ms"],
                        payload={"reason": parsed.get("reason", "")},
                    )
                )
            except Exception as exc:  # noqa: BLE001
                results.append(result_for(j, "reject", 0.0, payload={"error": str(exc)[:200]}))
        return results

    broker.register_handler("C1", router_handler)
    broker.register_handler("D3", verify_handler)

    if args.router == "jev":
        # Local C1: Tiny-Jev's `choice` primitive, one forward pass per task.
        from experiments.cascade.router_eval import TinyJev

        print(f"[cascade] loading local router {args.router_model} ...")
        jev = TinyJev(args.router_model)

        def jev_router_handler(batch):
            results = []
            for j in batch:
                task = j.payload["task"]
                state = {"task": task["prompt"], "bucket": task["bucket"]}
                t0 = time.time()
                with jev.torch.no_grad():
                    res = jev.model.choice(
                        jev.tok, state, "Which model should handle this task?",
                        {"local": "the local model answers correctly",
                         "api": "the API model is needed"},
                    )
                p_local = float((res.get("probabilities") or {}).get("local", 0.0))
                ms = (time.time() - t0) * 1000.0
                decision = "local" if p_local >= args.router_threshold else "api"
                results.append(
                    result_for(j, decision, p_local, latency_ms=ms,
                               payload={"p_local": p_local, "router": "tiny-jev"})
                )
            return results

        broker.register_handler("C1", jev_router_handler)

    if args.judge == "local":
        # Local D3: Intern-Decision-4B, one masked-next-token forward pass per
        # judgement (calibrated P(candidate is correct)).
        from experiments.cascade.decision import InternDecisionJudge

        print(f"[cascade] loading local judge {args.judge_model} ...")
        judge = InternDecisionJudge(args.judge_model, device=args.judge_device)

        def local_verify_handler(batch):
            results = []
            for j in batch:
                try:
                    p, ms = judge.verdict(j.payload["task"], j.payload["candidate"])
                    decision = "accept" if p >= policy.accept_confidence else "reject"
                    results.append(
                        result_for(j, decision, p, latency_ms=ms,
                                   payload={"judge": "intern-decision-4b"})
                    )
                except Exception as exc:  # noqa: BLE001
                    results.append(
                        result_for(j, "reject", 0.0, payload={"error": str(exc)[:200]})
                    )
            return results

        broker.register_handler("D3", local_verify_handler)

    # -- phase A: router batch -----------------------------------------
    print(f"[cascade] router phase over {len(tasks)} tasks ...")
    for task in tasks:
        broker.submit(
            Judgement(
                id=f"route-{task['id']}", request_id=task["id"], generation_id=0,
                role="C1", kind="route", payload={"task": task},
            )
        )
    broker.run_pending()
    routes = {r.request_id: r for r in broker.consume()}

    registry = default_registry()
    records: list[dict[str, Any]] = []
    for index, task in enumerate(tasks, 1):
        route_result = routes.get(task["id"])
        planned = plan_request(
            RouteDecision(task["bucket"], confidence=(route_result.confidence if route_result else 1.0)),
            out_len=256,
        )
        gated = (
            args.router_mode == "gate"
            and route_result is not None
            and route_result.decision == "api"
        )
        suffix = "  [gate->api]" if gated else ""
        print(f"[{index}/{len(tasks)}] {task['id']}  path={planned.path}{suffix}")

        if gated:
            # The router sent this to the API: skip local generation and the
            # verifier entirely; the API answer is the cascade answer.
            api_direct = frontier.chat(answer_prompt(task), tag="answer", max_tokens=700)
            api_ok = check(task, api_direct["content"])
            telemetry.log(
                task["id"], planned.path, "E4-api", "api", api_direct["latency_ms"],
                outcome="ok" if api_ok else "wrong", escalated=True,
            )
            records.append(
                {
                    "id": task["id"], "bucket": task["bucket"],
                    "planned_path": planned.path,
                    "scion_answer": None, "scion_ok": None, "scion_ms": None,
                    "scion_tokens": 0,
                    "router_route": route_result.decision,
                    "router_confidence": route_result.confidence,
                    "router_agrees": None,
                    "verdict": "gated", "verdict_confidence": None,
                    "verdict_reason": "router sent this task to the API",
                    "escalated": True, "gated": True,
                    "api_ok": api_ok, "api_answer": api_direct["content"][:4000],
                    "api_finish": api_direct.get("finish_reason"),
                    "api_ms": round(api_direct["latency_ms"], 1),
                    "api_direct_correct": api_ok,
                    "cascade_ok": api_ok,
                    "false_accept": False, "false_reject": False, "corrected": False,
                }
            )
            continue

        local = scion.chat(answer_prompt(task))
        local_ok = check(task, local["content"])
        telemetry.log(
            task["id"], planned.path, "E4-local", "dgpu1", local["latency_ms"],
            tokens=int((local["usage"] or {}).get("completion_tokens") or 0),
            outcome="ok" if local_ok else "wrong",
        )

        broker.submit(
            Judgement(
                id=f"verify-{task['id']}", request_id=task["id"], generation_id=0,
                role="D3", kind="verify",
                payload={"task": task, "candidate": local["content"]},
            )
        )
        broker.run_pending()
        verdicts = {r.request_id: r for r in broker.consume()}
        verdict = verdicts.get(task["id"])
        if verdict is None:
            verdict = result_for(
                Judgement(id=f"verify-{task['id']}", request_id=task["id"],
                          generation_id=0, role="D3", kind="verify"),
                "reject", 0.0, payload={"error": "no verdict"},
            )
        escalate = verifier_rejects(verdict, policy)
        telemetry.log(
            task["id"], planned.path, "D3-verify", "api", verdict.latency_ms,
            outcome="reject" if escalate else "accept", async_=True,
        )

        api_direct = None
        if escalate or args.router_mode == "measure":
            api_direct = frontier.chat(answer_prompt(task), tag="answer", max_tokens=700)
        api_ok = check(task, api_direct["content"]) if api_direct is not None else None

        cascade_answer = api_direct["content"] if escalate else local["content"]
        cascade_ok = api_ok if escalate else local_ok

        if escalate:
            telemetry.log(
                task["id"], planned.path, "E4-api", "api", api_direct["latency_ms"],
                outcome="ok" if api_ok else "wrong", escalated=True,
            )

        records.append(
            {
                "id": task["id"],
                "bucket": task["bucket"],
                "planned_path": planned.path,
                "scion_answer": local["content"][:4000],
                "scion_ok": local_ok,
                "scion_ms": round(local["latency_ms"], 1),
                "scion_tokens": int((local["usage"] or {}).get("completion_tokens") or 0),
                "router_route": (route_result.decision if route_result else None),
                "router_confidence": (route_result.confidence if route_result else None),
                "router_agrees": ((route_result.decision == "local") == local_ok)
                if route_result
                else None,
                "verdict": verdict.decision,
                "verdict_confidence": verdict.confidence,
                "verdict_reason": verdict.payload.get("reason", ""),
                "escalated": escalate,
                "gated": False,
                "api_ok": api_ok,
                "api_answer": (api_direct["content"][:4000] if api_direct else None),
                "api_finish": (api_direct.get("finish_reason") if api_direct else None),
                "api_ms": (round(api_direct["latency_ms"], 1) if api_direct else None),
                "api_direct_correct": api_ok,
                "cascade_ok": cascade_ok,
                "false_accept": (not escalate) and (not local_ok),
                "false_reject": escalate and local_ok,
                "corrected": escalate and local_ok is False and api_ok,
            }
        )

    # -- aggregate -----------------------------------------------------
    n = len(records)
    measured = [r for r in records if r["scion_ok"] is not None]
    gated = [r for r in records if r.get("gated")]
    api_rows = [r for r in records if r["api_ok"] is not None]
    accepted = [r for r in records if not r["escalated"]]

    def mean(rows, key):
        values = [r[key] for r in rows if r.get(key) is not None]
        return round(sum(values) / len(values), 4) if values else None

    summary = {
        "tasks": n,
        "gated": len(gated),
        "gate_rate": round(len(gated) / n, 4),
        "scion_accuracy": mean(measured, "scion_ok"),
        "api_accuracy": mean(api_rows, "api_ok"),
        "cascade_accuracy": round(sum(r["cascade_ok"] for r in records) / n, 4),
        "escalation_rate": round(sum(r["escalated"] for r in records) / n, 4),
        "accept_rate": round(len(accepted) / n, 4),
        "false_accept_rate": mean(measured, "false_accept"),
        "accepted_and_wrong": sum(r["false_accept"] for r in measured),
        "false_rejects": sum(r["false_reject"] for r in measured),
        "corrections": sum(r["corrected"] for r in measured),
        "router_agreement": mean(measured, "router_agrees"),
        "scion_mean_ms": mean(measured, "scion_ms"),
        "scion_mean_tokens": mean(measured, "scion_tokens"),
        "api_mean_ms": mean(api_rows, "api_ms"),
        "usage": usage.summary(),
        "telemetry": telemetry.summary(),
    }

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out) / (args.run_name.strip() or stamp)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "spec": spec["name"],
        "scion_base": scion.base,
        "api_model": API_MODEL,
        "summary": summary,
        "records": records,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")

    print("\n=== cascade pilot ===")
    header = f"{'task':26} {'bucket':10} {'scion':>5} {'api':>4} {'route':>5} {'verdict':>8} {'esc':>4} {'cascade':>7}"
    print(header)
    for r in records:
        print(
            f"{r['id']:26} {r['bucket']:10} {str(r['scion_ok']):>5} {str(r['api_ok']):>4} "
            f"{str(r['router_route']):>5} {r['verdict']:>8} {str(r['escalated']):>4} "
            f"{str(r['cascade_ok']):>7}"
        )
    print("\nsummary:", json.dumps({k: v for k, v in summary.items() if k not in ('usage', 'telemetry')}, indent=1))
    print("usage:", json.dumps(summary["usage"], indent=1))
    print("report:", out_dir / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
