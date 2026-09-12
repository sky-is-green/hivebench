"""Integration: S5 shadow mode, rollback+checkpoint, and 500-turn stability."""

import numpy as np

from cortex.checkpoint import StrataCheckpoint
from cortex.efficiency import EfficiencyScorer
from cortex.rollback import AutomatedRollback
from cortex.routing import DroneRouter, EscalationHandler
from focal.assembly import ContextAssembler
from focal.budget import AdaptiveBudget
from membrane.dedup import ContextDeduplicator
from membrane.drift import TopicDriftDetector
from retention.store import ContextStore
from sieve.scores import ChunkScore
from testing.shadow_mode import ShadowMode


class FakeUltraSmall:
    def score(self, query, chunks):
        return [ChunkScore(i, 0.9 if "JWT" in c else 0.2, 1.0) for i, c in enumerate(chunks)]

    def embed(self, text):
        return np.array([1.0, 0.0, 0.0])


class FakeMedium:
    def score(self, query, chunks):
        return [ChunkScore(i, 0.5, 0.85, source="medium") for i in range(len(chunks))]


def _store():
    s = ContextStore(embed_fn=lambda c: np.array([1.0, 0.0, 0.0]))
    for t in range(1, 26):
        s.add_chunk(t, f"authentication JWT token schema index turn {t}")
    for t in range(26, 51):
        s.add_chunk(t, f"gardening watering plants turn {t}")
    return s


def _process(assembler, pes_offset=0.0):
    def process(query, turn):
        assembled = assembler.assemble(
            query=query, current_turn=50, store=_store(),
            router=DroneRouter(), ultra_small=FakeUltraSmall(), medium=FakeMedium(),
            escalation=EscalationHandler(), dedup=ContextDeduplicator(),
            drift_detector=TopicDriftDetector(embed_fn=lambda t: np.array([1.0, 0.0, 0.0])),
            budget=AdaptiveBudget(), max_context=8192,
        )
        util = assembled.token_count / max(assembled.budget, 1)
        pes = 50 + (30 if "JWT" in assembled.content else 0) + min(20.0, util * 20.0)
        return {"pes": pes + pes_offset}

    return process


def test_shadow_mode_identical_configs_continue():
    assembler = ContextAssembler()
    shadow = ShadowMode(
        _process(assembler, 0), _process(assembler, 0), n_turns=100, margin=0.05
    )
    for turn in range(100):
        shadow.process_turn("how does authentication work", turn)
    ev = shadow.evaluate_after()
    assert ev.recommend == "continue"
    assert abs(ev.production_avg - ev.shadow_avg) < 1e-6


def test_shadow_mode_better_config_promotes():
    assembler = ContextAssembler()
    shadow = ShadowMode(
        _process(assembler, 0), _process(assembler, 15), n_turns=100, margin=0.05
    )
    for turn in range(100):
        shadow.process_turn("how does authentication work", turn)
    assert shadow.evaluate_after().recommend == "promote"


def test_rollback_fires_and_restores_checkpoint(tmp_path):
    ck = StrataCheckpoint(tmp_path)
    known_good = {"decay_multiplier": 1.8, "threshold": 2}
    saved = ck.auto_checkpoint(known_good, pes=90)
    assert saved is not None

    rb = AutomatedRollback()
    pes_history = [90] * 5 + [40] * 12  # 10+ consecutive turns below 50
    decision = rb.check(pes_history)
    assert decision.should_rollback is True

    # restore the last-known-good state saved earlier
    tag = saved.stem.replace("checkpoint_", "")
    assert ck.restore(tag) == known_good


def test_500_turn_stability_pes_above_60():
    store = _store()
    assembler = ContextAssembler()
    scorer = EfficiencyScorer()
    pes_history = []
    for turn in range(1, 501):
        assembled = assembler.assemble(
            query="how does authentication work", current_turn=turn, store=store,
            router=DroneRouter(), ultra_small=FakeUltraSmall(), medium=FakeMedium(),
            escalation=EscalationHandler(), dedup=ContextDeduplicator(),
            drift_detector=TopicDriftDetector(embed_fn=lambda t: np.array([1.0, 0.0, 0.0])),
            budget=AdaptiveBudget(), max_context=8192,
        )
        if turn % 5 == 0:
            pes = scorer.compute(
                retrieval_precision=85, routing_accuracy=90, avg_latency_ms=30,
                actual_tps=35, baseline_tps=30,
                budget_used=assembled.token_count, budget_total=assembled.budget,
            ).composite
            pes_history.append(pes)
    assert len(pes_history) == 100
    assert min(pes_history) > 60