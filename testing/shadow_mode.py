"""Shadow-mode testing.

Runs an experimental configuration alongside production without affecting the
user: production output goes to the user, shadow output is logged only. After N
turns, compare accumulated metrics and recommend promote / discard / continue.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ShadowEvaluation:
    recommend: str  # "promote" | "discard" | "continue"
    production_avg: float
    shadow_avg: float
    improvement: float
    turns: int


class ShadowMode:
    def __init__(
        self,
        run_production,
        run_shadow,
        key: str = "pes",
        n_turns: int = 100,
        margin: float = 0.05,
    ) -> None:
        self.run_production = run_production  # (query, turn) -> metrics dict
        self.run_shadow = run_shadow
        self.key = key
        self.n_turns = n_turns
        self.margin = margin  # relative improvement needed to promote
        self.shadow_log: list = []

    def process_turn(self, query: str, turn: int) -> dict:
        """Production output returns to the caller; shadow output is logged."""
        production = self.run_production(query, turn)
        shadow = self.run_shadow(query, turn)
        self.shadow_log.append({"turn": turn, "production": production, "shadow": shadow})
        return production

    def evaluate_after(self, n_turns: int | None = None) -> ShadowEvaluation:
        n = n_turns or self.n_turns
        window = self.shadow_log[-n:]
        prod = [w["production"].get(self.key) for w in window]
        shadow = [w["shadow"].get(self.key) for w in window]
        prod = [v for v in prod if v is not None]
        shadow = [v for v in shadow if v is not None]

        if not prod or not shadow:
            return ShadowEvaluation("continue", 0.0, 0.0, 0.0, 0)

        pa = sum(prod) / len(prod)
        sa = sum(shadow) / len(shadow)
        improvement = sa - pa
        base = max(pa, 1e-6)
        if sa > pa + self.margin * base:
            recommend = "promote"
        elif sa < pa - self.margin * base:
            recommend = "discard"
        else:
            recommend = "continue"
        return ShadowEvaluation(recommend, pa, sa, improvement, len(window))
