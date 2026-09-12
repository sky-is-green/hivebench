"""Parameter optimization via sweeps over logged data (S4.4).

For each candidate value we evaluate an objective (default: a replay over the
GroundTruthDB that re-simulates the component under the candidate) and return
the candidate minimizing (or maximizing) it. An explicit ``objective`` can be
injected for offline/unit tests.

- ``optimize_decay``: replays the decay pass over stored auditor labels — each
  candidate multiplier changes which chunks are predicted-kept, trading
  false-evictions against context bloat.
- ``optimize_routing_threshold``: replays the medium-tier assignment over stored
  routing decisions, trading routing accuracy against compute cost.
"""

from __future__ import annotations

import json
from typing import Callable, Iterable, Optional

DECAY_CANDIDATES = [1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5]
ROUTING_THRESHOLD_CANDIDATES = [1, 2, 3, 4, 5]

DECAY_KEEP_THRESHOLD = 0.6
BLOAT_PENALTY_WEIGHT = 0.3
ROUTING_COST_WEIGHT = 0.1


def sweep(
    candidates: Iterable,
    objective: Callable,
    minimize: bool = True,
) -> tuple:
    """Evaluate *objective(candidate)* for each candidate.

    Returns ``(best_candidate, best_value, results)`` where ``results`` is a list
    of ``(candidate, value)`` tuples.
    """
    results = []
    for c in candidates:
        value = objective(c)
        results.append((c, value))
    results.sort(key=lambda kv: kv[1], reverse=not minimize)
    best_candidate, best_value = results[0]
    return best_candidate, best_value, results


# ---------------------------------------------------------------------------
# Replay helpers
# ---------------------------------------------------------------------------
def replay_decay(db, multiplier: float) -> tuple[float, float]:
    """Re-simulate the decay pass under *multiplier* over stored auditor labels.

    Returns ``(false_eviction_rate, keep_rate)``. Age is derived from each
    label's turn (oldest chunks decay most); a chunk is predicted-kept when its
    decayed score exceeds DECAY_KEEP_THRESHOLD.
    """
    rows = db._conn.execute(
        "SELECT turn, actually_relevant, score FROM auditor_labels"
    ).fetchall()
    if not rows:
        return 0.0, 0.0
    max_turn = max(r["turn"] for r in rows)

    kept = 0
    evicted = 0
    false_evictions = 0
    for r in rows:
        age = max_turn - r["turn"]
        age_factor = min(age / 10.0, 3.0)
        effective = (r["score"] or 0.0) / (multiplier ** age_factor)
        if effective > DECAY_KEEP_THRESHOLD:
            kept += 1
        else:
            evicted += 1
            if r["actually_relevant"]:
                false_evictions += 1

    false_eviction_rate = false_evictions / evicted if evicted else 0.0
    keep_rate = kept / len(rows)
    return false_eviction_rate, keep_rate


def replay_routing(db, threshold: int) -> tuple[float, float]:
    """Re-simulate medium-tier assignment under *threshold* over stored decisions.

    Returns ``(accuracy, medium_invocation_rate)``. Each routing decision must be
    recorded with ``params={"score": ..., "optimal": ...}`` (see
    GroundTruthDB.record_routing_decision).
    """
    rows = db._conn.execute(
        "SELECT params FROM strata_decisions WHERE decision_type='route'"
    ).fetchall()
    total = len(rows)
    if not total:
        return 0.0, 0.0

    correct = 0
    medium_invoked = 0
    for r in rows:
        p = json.loads(r["params"]) if r["params"] else {}
        score = p.get("score", 0)
        optimal = p.get("optimal")
        if optimal is None:
            continue
        if score >= 3:
            predicted = "escalation"
        elif score >= threshold:
            predicted = "medium"
        else:
            predicted = "ultra_small"
        if predicted == optimal:
            correct += 1
        if predicted == "medium":
            medium_invoked += 1

    return correct / total, medium_invoked / total


# ---------------------------------------------------------------------------
# Optimization entry points
# ---------------------------------------------------------------------------
def optimize_decay(
    db=None,
    candidates: Iterable[float] = DECAY_CANDIDATES,
    objective: Optional[Callable[[float], float]] = None,
) -> tuple:
    """Find the decay multiplier minimizing false evictions + context bloat.

    Default objective replays the decay pass over the DB (higher multiplier ->
    more eviction -> less bloat but more false-eviction risk; the weights create
    a genuine trade-off rather than a trivial edge).
    """
    if objective is None:
        if db is None:
            raise ValueError("provide an objective or a GroundTruthDB")

        def objective(m):
            fe, keep = replay_decay(db, m)
            return fe + BLOAT_PENALTY_WEIGHT * keep

    return sweep(candidates, objective, minimize=True)


def optimize_routing_threshold(
    db=None,
    candidates: Iterable[int] = ROUTING_THRESHOLD_CANDIDATES,
    objective: Optional[Callable[[int], float]] = None,
) -> tuple:
    """Find the routing threshold maximizing accuracy while minimizing cost.

    Default objective replays the medium-tier assignment over the DB, penalizing
    unnecessary medium-drone invocations.
    """
    if objective is None:
        if db is None:
            raise ValueError("provide an objective or a GroundTruthDB")

        def objective(t):
            acc, medium_rate = replay_routing(db, t)
            return acc - ROUTING_COST_WEIGHT * medium_rate

    return sweep(candidates, objective, minimize=False)


def optimize_budget_ranges(
    budget_candidates: Iterable,
    objective: Callable,
) -> tuple:
    """Find the budget ranges maximizing sufficiency while minimizing waste."""
    return sweep(budget_candidates, objective, minimize=False)
