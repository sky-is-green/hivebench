"""Ablation studies (S4.5).

Runs the pipeline with components disabled to measure each component's
contribution. The runner is component-agnostic: a ``make_hive(config)`` callable
builds a per-config process_turn function, and the runner aggregates metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

# The 8 ablation configs (plan S4.5): which components are active.
ABLATION_CONFIGS = [
    {"name": "full", "decay": True, "drones": True, "remembrance": True, "dedup": True, "adaptive": True, "drift": True},
    {"name": "no_decay", "decay": False, "drones": True, "remembrance": True, "dedup": True, "adaptive": True, "drift": True},
    {"name": "no_drones", "decay": True, "drones": False, "remembrance": True, "dedup": True, "adaptive": True, "drift": True},
    {"name": "no_remembrance", "decay": True, "drones": True, "remembrance": False, "dedup": True, "adaptive": True, "drift": True},
    {"name": "no_dedup", "decay": True, "drones": True, "remembrance": True, "dedup": False, "adaptive": True, "drift": True},
    {"name": "no_adaptive", "decay": True, "drones": True, "remembrance": True, "dedup": True, "adaptive": False, "drift": True},
    {"name": "no_drift", "decay": True, "drones": True, "remembrance": True, "dedup": True, "adaptive": True, "drift": False},
    {"name": "baseline", "decay": False, "drones": False, "remembrance": False, "dedup": False, "adaptive": False, "drift": False},
]


@dataclass
class AblationResult:
    results: dict = field(default_factory=dict)  # config_name -> aggregate metrics
    contributions: dict = field(default_factory=dict)  # component -> PES delta vs full


class AblationRunner:
    def __init__(self, configs: list = ABLATION_CONFIGS) -> None:
        self.configs = configs

    def run(
        self,
        conversations: list,
        make_hive: Callable[[dict], Callable[[str, int], dict]],
        key: str = "pes",
    ) -> AblationResult:
        """For each config, build a process_turn fn and aggregate its metrics."""
        results = {}
        for config in self.configs:
            process_turn = make_hive(config)
            agg = self._aggregate(conversations, process_turn, key)
            results[config["name"]] = agg
        contributions = self._contributions(results, key)
        return AblationResult(results=results, contributions=contributions)

    def _aggregate(self, conversations, process_turn, key) -> dict:
        values = []
        for conv in conversations:
            for turn_i, turn in enumerate(conv.get("turns", [])):
                metrics = process_turn(turn.get("content", ""), turn_i)
                v = metrics.get(key)
                if v is not None:
                    values.append(v)
        if not values:
            return {key: None}
        return {key: round(sum(values) / len(values), 3)}

    @staticmethod
    def _contributions(results: dict, key: str) -> dict:
        full = results.get("full", {}).get(key)
        contributions = {}
        for name, agg in results.items():
            if name == "full":
                continue
            val = agg.get(key)
            contributions[name] = (full - val) if (full is not None and val is not None) else None
        return contributions
