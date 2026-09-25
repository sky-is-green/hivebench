"""Stack A/B — two saved stacks, the **same turns**, one report.

Gate item 6 (HIVE-PLAN §B0): "Two stacks can be A/B'd on the same turns via the
existing paired_ab / engines compare machinery."  This module is the experiment
that answers that item.  It deliberately builds **no new A/B engine** — it
composes the two that already exist:

* **quality** — :func:`experiments.paired_ab.run_paired` is run once per stack
  over the *same* conversation list with the *same* ``max_turns`` /
  ``fifo_budget`` / auditor.  Turn selection inside ``run_paired`` is a pure
  function of the fixtures (``_is_retrievable`` over the query and the prior
  history), never of the backend, so both arms necessarily score the identical
  ``(conversation_id, turn)`` set; :func:`turn_alignment` re-derives that set
  from the two reports and the report records it — "same turns" is an assertion
  in the artifact, not a comment in the code.
* **throughput** — the harness engines A/B (``POST /v1/engines/ab/bench`` and
  its two aliases, the ``harness/app.py:2046`` region) is asked to bench the two
  stacks' face tiers on one prompt list.  Its response — including the winner
  and its ≤2% noise band — is embedded **verbatim**.  The winner is never
  recomputed here; when the harness is unreachable the block is
  ``{"ok": false, ...}`` and the quality verdict stands on its own.

Everything the console prints is read back out of the report dict that is
written to ``results.json``, with its dotted path printed beside it, so every
number is traceable to a JSON location (HIVE-PLAN §7 report rule).

The two arms differ only in the **backend** they are handed: :func:`run_stack_ab`
takes a ``backend_for`` seam, ``stack -> backend``, which in ``--live`` mode is
the applied stack's face-tier endpoint (read from ``GET /v1/stacks/status``, since
a stack file carries no port — ADR-L3) and in ``--mock`` mode an offline stub.
Stack *documents* are read through the frozen ``Stack.to_dict()`` seam, so T43
does not need T35 merged: a real ``harness.stack.schema.Stack``, any duck-typed
``to_dict()`` object, or the raw §B3 mapping all work.

The engines A/B resolves profile names through the harness registry, so a live
run needs both face tiers registered as engines there — which is what the studio
does when a stack is applied (``register_local``, ``harness/app.py:1995``).

Usage::

    python -m experiments.stack_ab --mock                       # offline
    python -m experiments.stack_ab --live --stack-a peer-2tier \\
        --stack-b peer-3tier --max-convs 10 --max-turns 20 \\
        --harness-url http://127.0.0.1:8760 --output logs/stack_ab/results.json

Exit codes: ``0`` ran and both arms scored the same turns, ``2`` bad arguments or
empty corpus, ``3`` a live endpoint could not be resolved, ``4`` the two arms
scored different turn sets (the report is still written — the divergence is in
``turns.only_a`` / ``turns.only_b``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Optional

from cortex.baselines.runner import FIFO_WINDOW_TOKENS, load_conversations
from experiments.paired_ab import run_paired

#: Report schema version (``results.json``).
STACK_AB_VERSION = 1

#: Default artifact path. The name is fixed by the plan's report rule
#: ("numbers traceable to ``results.json``", HIVE-PLAN §7).
DEFAULT_OUTPUT = Path("logs") / "stack_ab" / "results.json"

#: Harness route for the engines A/B (harness/app.py:2230 and its two aliases).
ENGINES_AB_ROUTE = "/v1/engines/ab/bench"

#: Harness route carrying the applied stack's per-tier residency/port (§B3).
STACKS_STATUS_ROUTE = "/v1/stacks/status"

#: Default harness base URL (the local sidecar's port, per HIVE-HANDOFF).
DEFAULT_HARNESS_URL = "http://127.0.0.1:8760"

#: The face tier is the model the human talks to (ADR-L1).
FACE_ROLE = "face"

#: Quality metrics compared between the two stacks and their direction.
#: Keys and semantics are ``run_paired``'s own; a ``None`` direction means the
#: key is a bookkeeping count, not a comparable score.
QUALITY_METRICS: tuple[tuple[str, Optional[str]], ...] = (
    ("hive_answer_recall", "up"),
    ("fifo_answer_recall", "up"),
    ("hive_avg_fact_hit_ratio", "up"),
    ("fifo_avg_fact_hit_ratio", "up"),
    ("hive_avg_context_fidelity", "up"),
    ("fifo_avg_context_fidelity", "up"),
    ("fidelity_hive_ge_fifo_ratio", "up"),
    ("hive_ge_fifo_ratio", "up"),
    ("strict_hive_only_ratio", "up"),
    ("ctx_hive_ge_fifo_ratio", "up"),
    ("turns_compared", None),
)


# ---------------------------------------------------------------------------
# stack documents
# ---------------------------------------------------------------------------

def stack_doc(stack: Any) -> dict[str, Any]:
    """Normalise a stack to its §B3 document.

    Accepts a ``harness.stack.schema.Stack`` (through the frozen ``to_dict()``
    seam), any duck-typed ``to_dict()`` object, or the document itself.  Raises
    ``ValueError`` when the object is neither a mapping nor has ``to_dict``.
    """
    if isinstance(stack, dict):
        return dict(stack)
    to_dict = getattr(stack, "to_dict", None)
    if callable(to_dict):
        doc = to_dict()
        if not isinstance(doc, dict):
            raise ValueError(f"stack.to_dict() returned {type(doc).__name__}, want dict")
        return dict(doc)
    raise ValueError(
        f"stack must be a §B3 mapping or expose to_dict(); got {type(stack).__name__}"
    )


def face_tier(doc: dict[str, Any]) -> dict[str, Any]:
    """The stack's face tier, falling back to the first tier.

    ADR-L1: ``face`` is the tier the human converses with, so it is the tier an
    A/B arm talks to.  A stack with no face tier is still benchable — the first
    tier answers and the report records which role it was.
    """
    tiers = [t for t in (doc.get("tiers") or [])]
    if not tiers:
        raise ValueError(f"stack {doc.get('name')!r} has no tiers")
    for tier in tiers:
        if str(tier.get("role") or "") == FACE_ROLE:
            return dict(tier)
    return dict(tiers[0])


def stack_summary(stack: Any) -> dict[str, Any]:
    """Report block for one arm: identity, roles, face tier, launch config.

    ``launch`` comes from ``harness.stack.manager.tier_load_options`` once T37
    has landed, and is marked ``unavailable`` otherwise — this module carries no
    second copy of the tier→``load_options`` mapping (ADR-L7).
    """
    doc = stack_doc(stack)
    face = face_tier(doc)
    summary: dict[str, Any] = {
        "name": str(doc.get("name") or ""),
        "version": doc.get("version"),
        "tier_count": len(doc.get("tiers") or []),
        "roles": [str(t.get("role") or "") for t in (doc.get("tiers") or [])],
        "routing": doc.get("routing") or {},
        "face": {
            "role": str(face.get("role") or ""),
            "repo": str(face.get("repo") or ""),
            "file": str(face.get("file") or ""),
            "ctx": face.get("ctx"),
            "ngl": face.get("ngl"),
            "cache_k": face.get("cache_k"),
            "cache_v": face.get("cache_v"),
        },
    }
    try:
        from harness.stack.manager import tier_load_options
    except Exception:
        summary["launch"] = {"source": "unavailable (harness.stack.manager not importable)"}
    else:
        summary["launch"] = {
            "source": "harness.stack.manager.tier_load_options",
            "load_options": tier_load_options(face),
        }
    return summary


def engine_profile(summary: dict[str, Any], base_url: str) -> dict[str, Any]:
    """The arm's face tier as an engines-A/B profile record.

    The engines A/B resolves names through the harness registry, so the name
    has to be a real engine there: the face tier's GGUF stem, which is what the
    studio registers on apply.  ``base_url`` is carried for the record; the
    bench itself picks the port (basePort for A, basePort+1 for B).
    """
    stem = Path(str(summary.get("face", {}).get("file") or "")).stem
    return {
        "name": stem or str(summary.get("name") or "stack"),
        "kind": "llama_cpp",
        "base_url": base_url,
        "load_options": dict(summary.get("launch", {}).get("load_options") or {}),
        "capabilities": ["streaming", "prefix_caching"],
    }


# ---------------------------------------------------------------------------
# turn alignment — the "same turns" evidence
# ---------------------------------------------------------------------------

def turn_keys(report: dict[str, Any]) -> list[tuple[str, int]]:
    """``(conversation_id, turn)`` for every scored turn of a ``run_paired``
    report, in run order."""
    return [(str(r.get("conversation_id") or ""), int(r.get("turn") or 0))
            for r in (report.get("turns") or [])]


def turn_alignment(report_a: dict[str, Any], report_b: dict[str, Any]) -> dict[str, Any]:
    """Prove — or refute — that both arms scored the same turns.

    ``identical`` is the load-bearing claim; ``only_a`` / ``only_b`` name any
    divergence so a mispaired report cannot quietly look paired.
    """
    keys_a, keys_b = turn_keys(report_a), turn_keys(report_b)
    set_a, set_b = set(keys_a), set(keys_b)
    return {
        "identical": keys_a == keys_b,
        "count": len(keys_a),
        "keys": [list(k) for k in keys_a],
        "only_a": [list(k) for k in sorted(set_a - set_b)],
        "only_b": [list(k) for k in sorted(set_b - set_a)],
    }


def per_turn_rows(report_a: dict[str, Any], report_b: dict[str, Any]) -> list[dict[str, Any]]:
    """Turn-for-turn rows: each stack's fact-hit ratios beside the other's.

    ``fidelity_*`` is deliberately absent — ``run_paired`` computes it after the
    loop, so a resumed run's rows would otherwise differ from a fresh one's.
    """
    fields = ("answer_hive_hit_ratio", "answer_fifo_hit_ratio",
              "hive_ctx_tokens", "fifo_ctx_tokens")
    rows_b = {(str(r.get("conversation_id") or ""), int(r.get("turn") or 0)): r
              for r in (report_b.get("turns") or [])}
    rows_a = {(str(r.get("conversation_id") or ""), int(r.get("turn") or 0)): r
              for r in (report_a.get("turns") or [])}
    out: list[dict[str, Any]] = []
    for key in turn_keys(report_a):
        if key not in rows_b:
            continue
        out.append({
            "conversation_id": key[0],
            "turn": key[1],
            "a": {f: rows_a[key].get(f) for f in fields},
            "b": {f: rows_b[key].get(f) for f in fields},
        })
    return out


# ---------------------------------------------------------------------------
# the between-stack verdict
# ---------------------------------------------------------------------------

def quality_verdict(report_a: dict[str, Any], report_b: dict[str, Any]) -> dict[str, Any]:
    """Align the two arms' ``run_paired`` metrics and name a winner.

    A metric counts toward the verdict only when both arms produced a number and
    its direction is known; counts (``turns_compared``) are reported and
    excluded.  The winner is the side winning the most metrics, an exact tie
    reported as ``"tie"`` — deliberately *not* the engines A/B's rule: a quality
    arm is scored, not timed, so there is no tok/s noise band to apply.
    """
    ma = report_a.get("metrics") or {}
    mb = report_b.get("metrics") or {}
    rows: list[dict[str, Any]] = []
    a_wins = b_wins = 0
    for name, direction in QUALITY_METRICS:
        va, vb = ma.get(name), mb.get(name)
        row: dict[str, Any] = {"name": name, "direction": direction, "a": va, "b": vb}
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)) \
                and not isinstance(va, bool) and not isinstance(vb, bool):
            delta = va - vb
            row["delta"] = round(delta, 3)
            if direction is None:
                row["better"] = None
            elif delta > 0:
                row["better"] = "a"
                a_wins += 1
            elif delta < 0:
                row["better"] = "b"
                b_wins += 1
            else:
                row["better"] = "tie"
        else:
            row["better"] = None
            row["delta"] = None
        rows.append(row)
    return {"winner": "a" if a_wins > b_wins else ("b" if b_wins > a_wins else "tie"),
            "a_wins": a_wins, "b_wins": b_wins, "metrics": rows}


# ---------------------------------------------------------------------------
# the engines A/B seam (harness/app.py:2046 region, reused as a service)
# ---------------------------------------------------------------------------

def _post_json(url: str, payload: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
    import requests

    resp = requests.post(url, json=payload, timeout=timeout)
    raise_for_status = getattr(resp, "raise_for_status", None)
    if callable(raise_for_status):
        raise_for_status()
    if hasattr(resp, "json"):
        return resp.json()
    return json.loads(resp.text)


def _get_json(url: str, timeout: float = 10.0) -> dict[str, Any]:
    import requests

    resp = requests.get(url, timeout=timeout)
    raise_for_status = getattr(resp, "raise_for_status", None)
    if callable(raise_for_status):
        raise_for_status()
    if hasattr(resp, "json"):
        return resp.json()
    return json.loads(resp.text)


def bench_stacks(profile_a: dict[str, Any], profile_b: dict[str, Any], *,
                 harness_url: str = DEFAULT_HARNESS_URL,
                 base_port: Optional[int] = None,
                 prompts: Optional[list[str]] = None,
                 post: Optional[Callable[..., Any]] = None) -> dict[str, Any]:
    """Bench the two face tiers with the harness engines A/B.

    The request carries only the keys ``_ab_handle`` reads (``profile_a``,
    ``profile_b``, ``basePort``, ``prompts``) — names, not profile blobs, because
    the endpoint resolves them through its own registry.  The response is
    returned **verbatim** under ``"response"``: the winner and the ≤2% noise band
    belong to ``harness/app.py`` and are not re-derived here.  Any failure
    (unreachable harness, unknown engine) becomes ``{"ok": False, ...}`` so the
    quality half of the report still stands.
    """
    url = f"{harness_url.rstrip('/')}{ENGINES_AB_ROUTE}"
    body: dict[str, Any] = {"profile_a": profile_a.get("name"),
                            "profile_b": profile_b.get("name")}
    if base_port is not None:
        body["basePort"] = int(base_port)
    if prompts:
        body["prompts"] = [str(p) for p in prompts]
    call = post or _post_json
    try:
        resp = call(url, body)
    except Exception as exc:  # noqa: BLE001 — a bench failure is data, not a crash
        return {"ok": False, "route": ENGINES_AB_ROUTE, "url": url, "request": body,
                "error": f"{type(exc).__name__}: {exc}"[:300]}
    if not isinstance(resp, dict):
        return {"ok": False, "route": ENGINES_AB_ROUTE, "url": url, "request": body,
                "error": f"non-object response {type(resp).__name__}"}
    return {
        "ok": "winner" in resp,
        "route": ENGINES_AB_ROUTE,
        "url": url,
        "request": body,
        # Which arm the harness actually benched, by the profile name it echoed
        # back — the check that we asked for the stacks we meant to ask for.
        "engines_known": {
            side: str((resp.get(side) or {}).get("profile") or "") == str(profile.get("name") or "")
            for side, profile in (("a", profile_a), ("b", profile_b))
        },
        "response": resp,
    }


def applied_face_url(harness_url: str, stack_name: str, *,
                     get: Optional[Callable[..., Any]] = None) -> str:
    """The face tier's base URL of an *applied* stack, from the harness status.

    Raises ``LookupError`` when the harness has not applied that stack, or has
    applied a different one — the caller should apply the stack first
    (``POST /v1/stacks/{name}/apply``) rather than bench a stale endpoint.
    """
    call = get or _get_json
    status = call(f"{harness_url.rstrip('/')}{STACKS_STATUS_ROUTE}")
    if not isinstance(status, dict):
        raise LookupError(f"{STACKS_STATUS_ROUTE} returned {type(status).__name__}")
    applied = str(status.get("stack") or "")
    if stack_name and applied and applied != stack_name:
        raise LookupError(
            f"harness has {applied!r} applied, not {stack_name!r}; apply it first "
            f"(POST /v1/stacks/{stack_name}/apply) or pass --live-base-url")
    for row in status.get("tiers") or []:
        if isinstance(row, dict) and str(row.get("role") or "") == FACE_ROLE:
            port = row.get("port")
            if port is None:
                raise LookupError(f"applied {applied or stack_name!r} face tier has no port")
            return f"http://{row.get('host') or '127.0.0.1'}:{port}"
    raise LookupError(
        f"no face tier in {STACKS_STATUS_ROUTE} for {applied or stack_name!r}")


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

def synthetic_stack(name: str, roles: list[str]) -> dict[str, Any]:
    """A stand-in §B3 stack for offline/mock runs.

    T35 authors the real ``stacks/peer-2tier.json`` / ``peer-3tier.json``; the A/B
    harness only needs the *shape* to report on, and T43 must not depend on a
    sibling task's files (ADR-L7).
    """
    tiers = [{
        "role": role,
        "repo": f"mock/{role}",
        "file": f"{name}-{role}-{i}.gguf",
        "ctx": 8192 * (2 ** i),
        "ngl": 99,
        "backend": "vulkan",
        "cache_k": "q8_0",
        "cache_v": "q8_0",
    } for i, role in enumerate(roles)]
    return {"name": name, "version": 1, "tiers": tiers,
            "routing": {"workers_as": "subagent"}}


def load_stack_document(name: str, root: str | Path = "stacks") -> dict[str, Any]:
    """Read a saved stack document as a §B3 mapping.

    Uses ``harness.stack.schema.load_stack`` when T35 has landed (so its shape
    validation runs) and falls back to the raw document otherwise, so T43 does
    not depend on a sibling task's merge.
    """
    try:
        from harness.stack.schema import load_stack
    except Exception:
        load_stack = None
    if load_stack is not None:
        return stack_doc(load_stack(name))
    path = Path(root) / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"no stack {name!r} at {path} and harness.stack.schema is unavailable")
    return json.loads(path.read_text(encoding="utf-8"))


def run_stack_ab(stack_a: Any, stack_b: Any, conversations: list[dict[str, Any]],
                 backend_for: Callable[[Any], Any], *,
                 ultra: Any, medium: Any,
                 bench: Optional[dict[str, Any]] = None,
                 base_url_a: str = "", base_url_b: str = "",
                 max_turns: Optional[int] = None,
                 fifo_budget: int = FIFO_WINDOW_TOKENS,
                 sampling: Optional[dict[str, Any]] = None,
                 auditor: Any = None,
                 verbose: bool = False) -> dict[str, Any]:
    """Run both stacks over ``conversations`` and return one report dict.

    ``backend_for(stack)`` is the only thing that differs between the arms, so
    the run isolates the stack.  Everything else — the corpus, ``max_turns``,
    ``fifo_budget``, ``sampling``, the auditor, the drones — is shared.  ``bench``
    is a pre-computed :func:`bench_stacks` block; ``None`` records the throughput
    half as skipped.  The returned dict is exactly what is written to
    ``results.json``.
    """
    summary_a, summary_b = stack_summary(stack_a), stack_summary(stack_b)
    doc_a, doc_b = stack_doc(stack_a), stack_doc(stack_b)

    shared = dict(max_turns=max_turns, fifo_budget=fifo_budget, auditor=auditor,
                  verbose=verbose)
    report_a = run_paired(conversations, backend_for(doc_a), ultra, medium,
                          sampling=sampling, **shared)
    report_b = run_paired(conversations, backend_for(doc_b), ultra, medium,
                          sampling=sampling, **shared)

    alignment = turn_alignment(report_a, report_b)
    quality = {
        "a": report_a.get("metrics") or {},
        "b": report_b.get("metrics") or {},
        "first_mention_excluded": report_a.get("first_mention_excluded", 0),
        "no_facts_excluded": report_a.get("no_facts_excluded", 0),
    }
    profile_a = engine_profile(summary_a, base_url_a)
    profile_b = engine_profile(summary_b, base_url_b)
    bench_block = bench or {"ok": False, "skipped": True,
                            "reason": "not requested (--no-bench)"}
    return {
        "version": STACK_AB_VERSION,
        "kind": "stack_ab",
        "arms": {
            "a": {**summary_a, "base_url": base_url_a, "engine_profile": profile_a},
            "b": {**summary_b, "base_url": base_url_b, "engine_profile": profile_b},
        },
        "settings": {
            "conversations": len(conversations),
            "max_turns": max_turns,
            "fifo_budget_tokens": fifo_budget,
            "auditor": auditor is not None,
            "sampling": dict(sampling or {}),
        },
        "turns": alignment,
        "quality": quality,
        "quality_verdict": quality_verdict(report_a, report_b),
        "per_turn": per_turn_rows(report_a, report_b),
        "throughput": bench_block,
        "caveats": caveats(summary_a, summary_b, profile_a, profile_b, bench_block),
    }


def caveats(summary_a: dict[str, Any], summary_b: dict[str, Any],
            profile_a: dict[str, Any], profile_b: dict[str, Any],
            bench: dict[str, Any]) -> list[str]:
    """Things a verifier must know before reading the numbers off this report.

    Each caveat is a limitation of the run, not a defect: an offline bench, a
    launch config T37 has not produced yet, or two stacks whose face tiers
    resolve to the same engines-A/B profile.
    """
    out: list[str] = []
    if not bench.get("ok"):
        out.append(f"throughput not measured: "
                   f"{bench.get('error') or bench.get('reason') or 'unavailable'}")
    for side, summary in (("a", summary_a), ("b", summary_b)):
        source = str((summary.get("launch") or {}).get("source") or "")
        if source.startswith("unavailable"):
            out.append(f"arm {side} launch config unavailable ({source})")
    if profile_a.get("name") == profile_b.get("name"):
        out.append(
            f"both arms' face tiers resolve to the same engines-A/B profile "
            f"{profile_a.get('name')!r}; the throughput half would bench one "
            f"engine against itself")
    return out


# ---------------------------------------------------------------------------
# reporting — every printed number is read back out of the report dict
# ---------------------------------------------------------------------------

def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def format_report(report: dict[str, Any]) -> str:
    """The console report, rendered from ``report`` alone.

    Every figure is printed next to the dotted path (or the ``key=value`` trail
    within the row) that locates it in ``results.json`` — the traceability
    requirement of HIVE-PLAN §7.
    """
    arms = report.get("arms") or {}
    a, b = arms.get("a") or {}, arms.get("b") or {}
    turns = report.get("turns") or {}
    settings = report.get("settings") or {}
    quality = report.get("quality") or {}
    verdict = report.get("quality_verdict") or {}

    lines = [
        "Stack A/B — two stacks, same turns, one report",
        f"  arm A            : {_fmt(a.get('name'))} [arms.a.name] "
        f"tiers={_fmt(a.get('tier_count'))} [arms.a.tier_count] "
        f"roles={_fmt(a.get('roles'))} [arms.a.roles] "
        f"face={_fmt((a.get('face') or {}).get('file'))} [arms.a.face.file]",
        f"  arm B            : {_fmt(b.get('name'))} [arms.b.name] "
        f"tiers={_fmt(b.get('tier_count'))} [arms.b.tier_count] "
        f"roles={_fmt(b.get('roles'))} [arms.b.roles] "
        f"face={_fmt((b.get('face') or {}).get('file'))} [arms.b.face.file]",
        f"  launch config    : A={_fmt((a.get('launch') or {}).get('source'))} "
        f"[arms.a.launch.source] | B={_fmt((b.get('launch') or {}).get('source'))} "
        f"[arms.b.launch.source]",
        f"  turns compared   : {_fmt(turns.get('count'))} [turns.count] "
        f"identical={_fmt(turns.get('identical'))} [turns.identical] "
        f"keys={_fmt(turns.get('keys'))} [turns.keys]",
        f"  corpus           : {_fmt(settings.get('conversations'))} "
        f"[settings.conversations] conversations, max_turns="
        f"{_fmt(settings.get('max_turns'))} [settings.max_turns], fifo_budget="
        f"{_fmt(settings.get('fifo_budget_tokens'))} [settings.fifo_budget_tokens]",
        f"  excluded turns   : first-mention "
        f"{_fmt(quality.get('first_mention_excluded'))} "
        f"[quality.first_mention_excluded], no-facts "
        f"{_fmt(quality.get('no_facts_excluded'))} [quality.no_facts_excluded]",
    ]
    if not turns.get("identical"):
        lines.append(
            f"  ! turn sets differ: only_a={_fmt(turns.get('only_a'))} [turns.only_a] "
            f"only_b={_fmt(turns.get('only_b'))} [turns.only_b]")

    lines.append("  quality (run_paired metrics, same fixtures, same turns):")
    for row in verdict.get("metrics") or []:
        name = row.get("name")
        lines.append(
            f"    {str(name):<28} A={_fmt(row.get('a'))} B={_fmt(row.get('b'))}  "
            f"[quality_verdict.metrics.{name}: a={_fmt(row.get('a'))} "
            f"b={_fmt(row.get('b'))} delta={_fmt(row.get('delta'))} "
            f"better={_fmt(row.get('better'))} "
            f"direction={_fmt(row.get('direction'))}]")

    lines.append(
        f"  quality winner   : {_fmt(verdict.get('winner'))} [quality_verdict.winner] "
        f"— A {_fmt(verdict.get('a_wins'))} [quality_verdict.a_wins] / B "
        f"{_fmt(verdict.get('b_wins'))} [quality_verdict.b_wins] metrics")

    bench = report.get("throughput") or {}
    if bench.get("ok"):
        resp = bench.get("response") or {}
        lines.append(
            f"  throughput winner: {_fmt(resp.get('winner'))} "
            f"[throughput.response.winner] — A {_fmt(resp.get('a_tok_per_sec'))} "
            f"[throughput.response.a_tok_per_sec] tok/s vs B "
            f"{_fmt(resp.get('b_tok_per_sec'))} [throughput.response.b_tok_per_sec] "
            f"tok/s (engines A/B, verbatim)")
    else:
        reason = bench.get("error") or bench.get("reason") or "unavailable"
        lines.append(
            f"  throughput       : not measured — {reason} "
            f"[throughput.error|throughput.reason] (the harness engines A/B at "
            f"{ENGINES_AB_ROUTE} is reused verbatim when it is up)")
    for i, caveat in enumerate(report.get("caveats") or []):
        lines.append(f"  ! caveat[{i}]      : {caveat}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _mock_backend(tag: str) -> Any:
    """An offline arm backend: deterministic, network-free, tagged per stack."""
    class _MockArm:
        name = tag

        def generate(self, context, query, sampling=None):
            return f"[mock:{self.name}] re: {str(query)[:40]}"

    return _MockArm()


def _mock_auditor() -> Any:
    """Deterministic offline auditor (mirrors ``paired_ab``'s mock auditor)."""
    return lambda prompt: json.dumps(
        {"sufficient": True, "used_pieces": [], "missing": [], "score": 4})


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stack A/B: two saved stacks on the same turns, one report")
    parser.add_argument("--mock", action="store_true",
                        help="offline: stub backends, synthetic 2-tier/3-tier stacks")
    parser.add_argument("--live", action="store_true",
                        help="A/B stacks applied in a running harness (implies --mock "
                             "off: no stub backend, no synthetic stack)")
    parser.add_argument("--stack-a", default="", help="stack name (a file under stacks/ "
                                                      "in --live, else the report label)")
    parser.add_argument("--stack-b", default="", help="stack name (see --stack-a)")
    parser.add_argument("--conversations", default="tests/fixtures/generated")
    parser.add_argument("--max-convs", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--fifo-budget", type=int, default=FIFO_WINDOW_TOKENS,
                        help="FIFO window in tokens, shared by both arms "
                             "(default %(default)s)")
    parser.add_argument("--harness-url", default=DEFAULT_HARNESS_URL,
                        help="harness base URL for the engines A/B and "
                             "/v1/stacks/status")
    parser.add_argument("--base-port", type=int, default=None,
                        help="basePort for the engines A/B (arm A on it, arm B on +1)")
    parser.add_argument("--live-base-url-a", default="",
                        help="override arm A's face endpoint instead of reading status")
    parser.add_argument("--live-base-url-b", default="",
                        help="override arm B's face endpoint instead of reading status")
    parser.add_argument("--no-bench", action="store_true",
                        help="skip the engines A/B; the quality verdict still runs")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help="results.json path (default %(default)s)")
    parser.add_argument("--sampling", default="", help="sampling overrides as JSON")
    parser.add_argument("--auditor", action="store_true",
                        help="also score context sufficiency with the auditor, both arms")
    return parser.parse_args(argv)


def _live_backend_for(doc: dict[str, Any], base_url: str) -> Any:
    """An ``LMStudioBackend`` on a stack's applied face tier."""
    from backend.lmstudio import LMStudioBackend

    return LMStudioBackend(base_url=base_url, model=face_tier(doc).get("file") or "")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    live = bool(args.live) and not args.mock

    conversations = load_conversations(args.conversations)
    if args.max_convs:
        conversations = conversations[: args.max_convs]
    if not conversations:
        print(f"No conversation files found in {args.conversations}")
        return 2
    if live and not (args.stack_a and args.stack_b):
        print("--live needs both --stack-a and --stack-b")
        return 2

    try:
        if live:
            doc_a = load_stack_document(args.stack_a)
            doc_b = load_stack_document(args.stack_b)
        else:
            doc_a = synthetic_stack(args.stack_a or "mock-2tier", ["face", "worker"])
            doc_b = synthetic_stack(args.stack_b or "mock-3tier",
                                    ["face", "worker", "mechanics"])
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}")
        return 2

    summary_a, summary_b = stack_summary(doc_a), stack_summary(doc_b)
    base_url_a, base_url_b = args.live_base_url_a, args.live_base_url_b
    bench: dict[str, Any] = {"ok": False, "skipped": True, "reason": "not requested"}

    if live:
        for label, name, override in (("A", doc_a.get("name"), base_url_a),
                                      ("B", doc_b.get("name"), base_url_b)):
            if override:
                continue
            try:
                resolved = applied_face_url(args.harness_url, name or "")
            except Exception as exc:  # noqa: BLE001
                print(f"error: arm {label} ({name!r}) face endpoint: {exc}")
                return 3
            if label == "A":
                base_url_a = resolved
            else:
                base_url_b = resolved
        if not args.no_bench:
            bench = bench_stacks(
                engine_profile(summary_a, base_url_a),
                engine_profile(summary_b, base_url_b),
                harness_url=args.harness_url, base_port=args.base_port)
        backend_for = lambda doc: _live_backend_for(  # noqa: E731
            doc, base_url_a if doc.get("name") == doc_a.get("name") else base_url_b)
    else:
        if not args.no_bench:
            bench = {"ok": False, "skipped": True,
                     "reason": "offline (--mock): the engines A/B needs a live harness"}
        backend_for = lambda doc: _mock_backend(str(doc.get("name") or "stack"))  # noqa: E731

    from sieve.medium import MediumDrone

    if live:
        from sieve.ultra_small import UltraSmallDrone

        ultra = UltraSmallDrone()
        ultra._ensure_loaded()
    else:
        from cortex.e2e import FakeUltraSmall

        ultra = FakeUltraSmall()
    medium = MediumDrone(score_pair_fn=lambda q, c: 0.5)
    auditor = None
    if args.auditor:
        if live:
            print("error: --auditor is offline-only; score sufficiency with "
                  "experiments.paired_ab --auditor instead")
            return 2
        auditor = _mock_auditor()

    sampling = None
    if args.sampling:
        from backend.sampling import parse_sampling

        sampling = parse_sampling(args.sampling)

    report = run_stack_ab(
        doc_a, doc_b, conversations, backend_for,
        ultra=ultra, medium=medium, bench=bench,
        base_url_a=base_url_a, base_url_b=base_url_b,
        max_turns=args.max_turns, fifo_budget=args.fifo_budget,
        sampling=sampling, auditor=auditor, verbose=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(format_report(report))
    print(f"Wrote {out.resolve()}")
    return 0 if report["turns"]["identical"] else 4


if __name__ == "__main__":
    sys.exit(main())
