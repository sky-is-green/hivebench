"""A/B testing framework.

Runs two configurations of the splinter on the same conversations and compares their
aggregate metrics. Config A is the current production config; Config B is the
experiment. Both process the same input; only A's output goes to the user.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy import stats

# Metrics the runner aggregates from each process_turn result.
AGGREGATE_KEYS = (
    "retrieval_precision",
    "routing_accuracy",
    "latency_ms",
    "context_utilization",
    "pes",
)


@dataclass
class ABTestResult:
    config_a_metrics: dict
    config_b_metrics: dict
    winner: str  # "A" | "B" | "tie"
    detail: dict = field(default_factory=dict)


class ABTestRunner:
    def __init__(self, turns: int = 50) -> None:
        self.turns = turns

    def run(
        self,
        conversations: list,
        process_a: Callable[[str, int], dict],
        process_b: Callable[[str, int], dict],
        key: str = "pes",
    ) -> ABTestResult:
        """Run both configs over the same turns; return comparative metrics.

        ``process_x(query, turn) -> dict`` of metrics (e.g. {"pes": 80, ...}).
        """
        results_a = []
        results_b = []
        for conv in conversations:
            turns = conv.get("turns", [])
            for turn_i, turn in enumerate(turns[: self.turns]):
                query = turn.get("content", "")
                results_a.append(process_a(query, turn_i))
                results_b.append(process_b(query, turn_i))

        metrics_a = self._compute_metrics(results_a)
        metrics_b = self._compute_metrics(results_b)
        winner = self._statistical_winner(results_a, results_b, key)
        if winner is None:
            winner = self._determine_winner(metrics_a, metrics_b, key)
        return ABTestResult(
            config_a_metrics=metrics_a,
            config_b_metrics=metrics_b,
            winner=winner,
            detail={"key": key, "turns": self.turns},
        )

    @staticmethod
    def _compute_metrics(results: list[dict]) -> dict:
        aggregates = {}
        for key in AGGREGATE_KEYS:
            values = [r.get(key) for r in results if r.get(key) is not None]
            if values:
                aggregates[key] = round(sum(values) / len(values), 3)
            else:
                aggregates[key] = None
        return aggregates

    @staticmethod
    def _determine_winner(metrics_a: dict, metrics_b: dict, key: str) -> str:
        a = metrics_a.get(key)
        b = metrics_b.get(key)
        if a is None or b is None:
            return "tie"
        tol = max(0.02 * max(a, b), 1e-6)  # 2% statistical-noise margin
        if abs(a - b) <= tol:
            return "tie"
        return "A" if a > b else "B"

    @staticmethod
    def _statistical_winner(
        results_a: list[dict], results_b: list[dict], key: str, alpha: float = 0.05
    ):
        """Mann-Whitney U test on per-turn metric samples.

        Returns "A", "B", or "tie" when there are enough varied samples; returns
        None when the test cannot run (e.g. constant samples), so the caller
        falls back to the tolerance-based comparison.
        """
        va = [r[key] for r in results_a if r.get(key) is not None]
        vb = [r[key] for r in results_b if r.get(key) is not None]
        if len(va) < 3 or len(vb) < 3:
            return None
        if len(set(va)) < 2 or len(set(vb)) < 2:
            return None  # constant samples -> no variance
        try:
            _stat, p = stats.mannwhitneyu(va, vb, alternative="two-sided")
        except Exception:  # noqa: BLE001
            return None
        if p > alpha:
            return "tie"
        return "A" if np.mean(va) > np.mean(vb) else "B"
