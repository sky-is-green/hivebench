"""Integration: S4 auditor roundtrip and A/B testing with the real assembler."""

import json

import numpy as np

from cortex.routing import DroneRouter, EscalationHandler
from focal.assembly import ContextAssembler
from focal.budget import AdaptiveBudget
from membrane.dedup import ContextDeduplicator
from membrane.drift import TopicDriftDetector
from auditor.auditor import Auditor, TurnRecord
from auditor.ground_truth import GroundTruthDB
from retention.store import ContextStore
from sieve.scores import ChunkScore
from testing.ab_test import ABTestRunner


class FakeUltraSmall:
    def score(self, query, chunks):
        return [ChunkScore(i, 0.9 if "JWT" in c else 0.2, 1.0) for i, c in enumerate(chunks)]

    def embed(self, text):
        return np.array([1.0, 0.0, 0.0])


class FakeMedium:
    def score(self, query, chunks):
        return [ChunkScore(i, 0.5, 0.85, source="medium") for i in range(len(chunks))]


class TinyBudget:
    def compute(self, route, count, max_context=8192):
        return 0  # no chunk fits -> empty assembled context


def _store():
    s = ContextStore(embed_fn=lambda c: np.array([1.0, 0.0, 0.0]))
    for t in range(1, 26):
        s.add_chunk(t, f"authentication JWT token schema index turn {t}")
    for t in range(26, 51):
        s.add_chunk(t, f"gardening watering plants turn {t}")
    return s


def _make_process(assembler, budget):
    def process(query, turn):
        store = _store()
        assembled = assembler.assemble(
            query=query, current_turn=50, store=store,
            router=DroneRouter(), ultra_small=FakeUltraSmall(), medium=FakeMedium(),
            escalation=EscalationHandler(), dedup=ContextDeduplicator(),
            drift_detector=TopicDriftDetector(embed_fn=lambda t: np.array([1.0, 0.0, 0.0])),
            budget=budget, max_context=8192,
        )
        relevant = "JWT" in assembled.content
        util = assembled.token_count / max(assembled.budget, 1)
        pes = 50 + (30 if relevant else 0) + min(20.0, util * 20.0)
        return {"pes": pes, "context_utilization": util}

    return process


def test_ab_test_full_vs_tiny_budget_real_assembler():
    conv = [{"turns": [{"content": "how does authentication work"}]}]
    assembler = ContextAssembler()
    result = ABTestRunner(turns=10).run(
        conv,
        process_a=_make_process(assembler, AdaptiveBudget()),
        process_b=_make_process(assembler, TinyBudget()),
    )
    # Config A (adaptive budget) retrieves relevant context -> better PES.
    assert result.winner == "A"
    assert result.config_a_metrics["pes"] > result.config_b_metrics["pes"]


def test_auditor_to_ground_truth_roundtrip():
    assembler = ContextAssembler()
    store = _store()
    assembled = assembler.assemble(
        query="how does authentication work", current_turn=50, store=store,
        router=DroneRouter(), ultra_small=FakeUltraSmall(), medium=FakeMedium(),
        escalation=EscalationHandler(), dedup=ContextDeduplicator(),
        drift_detector=TopicDriftDetector(embed_fn=lambda t: np.array([1.0, 0.0, 0.0])),
        budget=AdaptiveBudget(), max_context=8192,
    )

    auditor = Auditor(
        generate_fn=lambda p: json.dumps(
            {"sufficient": True, "used_pieces": [], "missing": [], "score": 4}
        )
    )
    label = auditor.evaluate_turn(
        TurnRecord(
            turn=1, assembled_context=assembled.content, user_query="q",
            llm_response="r", chunk_ids=assembled.selected_chunk_ids,
        )
    )
    assert label.context_sufficient is True

    db = GroundTruthDB()
    for cid in assembled.selected_chunk_ids:
        # auditor says every selected chunk was relevant
        db.record_auditor_label(1, cid, True, bool(label.chunk_labels.get(cid, True)))
    assert 0.0 <= db.retrieval_precision() <= 100.0
    assert db.label_count() == len(assembled.selected_chunk_ids)
    db.close()