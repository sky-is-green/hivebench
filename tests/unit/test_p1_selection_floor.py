"""P1-FLOOR — relevance floor + per-chunk window-share cap in focal assembly.

Spec: SPLINTER-PLAN.md "P1-FLOOR". Selection must skip chunks whose RAW drone
score is below ``relevance_floor`` (pre-decay, pre-drift, uniform incl. comb
candidates) and truncate any chunk costing more than ``max_chunk_share`` of
the budget to the cap at selection time only.
"""

import numpy as np

from cortex.baselines.metrics import estimate_tokens
from cortex.routing import DroneRouter, EscalationHandler
from focal.assembly import ContextAssembler
from focal.budget import AdaptiveBudget
from membrane.dedup import ContextDeduplicator
from membrane.drift import TopicDriftDetector
from retention.store import ContextStore
from sieve.scores import ChunkScore


class FakeChunk:
    def __init__(self, cid, content):
        self.id = cid
        self.content = content
        self.relevance_history = []


def _pool(*chunks):
    return {c.id: c for c in chunks}


class FloorDrone:
    def score(self, query, chunks):
        return [ChunkScore(i, 0.05, 1.0) for i, _ in enumerate(chunks)]

    def embed(self, text):
        return np.array([1.0, 0.0, 0.0])


class FloorMedium:
    def score(self, query, chunks):
        return [ChunkScore(i, 0.5, 0.85, source="medium") for i, _ in enumerate(chunks)]


class ZeroDrone:
    def score(self, query, chunks):
        return [ChunkScore(i, 0.05, 1.0) for i, _ in enumerate(chunks)]

    def embed(self, text):
        return np.array([1.0, 0.0, 0.0])


def test_floor_selects_only_chunks_at_or_above_raw_floor():
    a = FakeChunk("a", "x" * 400)
    b = FakeChunk("b", "y" * 400)
    c = FakeChunk("c", "z" * 400)
    d = FakeChunk("d", "w" * 400)
    raw = {"a": 0.9, "b": 0.30, "c": 0.24, "d": 0.05}
    scored = [("a", 0.9), ("b", 0.30), ("c", 0.24), ("d", 0.05)]

    selected = ContextAssembler()._select_within_budget(
        scored, _pool(a, b, c, d), 100_000, raw_scores=raw
    )

    assert [c.id for c in selected] == ["a", "b"]


def test_floor_consults_raw_not_effective_score():
    stale = FakeChunk("stale", "s" * 400)
    loud = FakeChunk("loud", "l" * 400)
    raw = {"stale": 0.8, "loud": 0.1}
    scored = [("loud", 0.95), ("stale", 0.05)]

    selected = ContextAssembler()._select_within_budget(
        scored, _pool(stale, loud), 100_000, raw_scores=raw
    )

    assert [c.id for c in selected] == ["stale"]


def test_cap_truncates_oversized_chunk_and_keeps_budget_intact():
    big = FakeChunk("big", "B" * 3000)
    small = FakeChunk("small", "s" * 400)
    raw = {"big": 0.9, "small": 0.8}
    scored = [("big", 0.9), ("small", 0.8)]

    selected = ContextAssembler()._select_within_budget(
        scored, _pool(big, small), 1000, raw_scores=raw, max_chunk_share=0.5
    )

    assert [c.id for c in selected] == ["big", "small"]
    big_selected = selected[0]
    assert big_selected is not big
    assert estimate_tokens(big_selected.content) == 500
    assert len(big_selected.content) < len(big.content)
    assert big.content == "B" * 3000
    assert sum(estimate_tokens(c.content) for c in selected) <= 1000


def test_cap_truncation_below_min_chars_is_skipped():
    chunk = FakeChunk("big", "B" * 100)

    selected = ContextAssembler()._select_within_budget(
        [("big", 0.9)], _pool(chunk), 12, raw_scores={"big": 0.9},
        max_chunk_share=0.5,
    )

    assert selected == []


def test_all_below_floor_assembles_empty_content():
    store = ContextStore(embed_fn=lambda c: np.array([1.0, 0.0, 0.0]))
    store.add_chunk(1, "gardening tips for watering plants")
    store.add_chunk(2, "quarterly tax filing checklist")

    result = ContextAssembler().assemble(
        query="how does authentication work",
        current_turn=3,
        store=store,
        router=DroneRouter(),
        ultra_small=FloorDrone(),
        medium=FloorMedium(),
        escalation=EscalationHandler(),
        dedup=ContextDeduplicator(),
        drift_detector=TopicDriftDetector(embed_fn=lambda t: np.array([1.0, 0.0, 0.0])),
        budget=AdaptiveBudget(),
        max_context=8192,
    )

    assert result.content == ""
    assert result.chunks_used == 0
    assert result.selected_chunk_ids == []
    assert result.token_count == 0


def test_regression_matches_old_greedy_when_all_above_floor():
    chunks = [FakeChunk(str(i), chr(65 + i) * 100) for i in range(5)]
    scored = [(c.id, 0.9 - i * 0.05) for i, c in enumerate(chunks)]
    raw = {c.id: 0.9 for c in chunks}
    pool = _pool(*chunks)

    selected = ContextAssembler()._select_within_budget(
        scored, pool, 120, raw_scores=raw
    )

    expected = []
    used = 0
    for cid, _score in scored:
        cost = estimate_tokens(pool[cid].content)
        if used + cost <= 120:
            expected.append(cid)
            used += cost

    assert [c.id for c in selected] == expected
    assert [c.id for c in selected] == ["0", "1", "2", "3"]


def test_no_arg_selection_call_is_back_compatible():
    big = FakeChunk("big", "B" * 3000)
    small = FakeChunk("small", "s" * 400)
    scored = [("big", 0.9), ("small", 0.8)]

    selected = ContextAssembler()._select_within_budget(
        scored, _pool(big, small), 1000
    )

    assert [c.id for c in selected] == ["big", "small"]
    assert selected[0] is big
    assert selected[0].content == "B" * 3000
    assert sum(estimate_tokens(c.content) for c in selected) == 850


def test_app_tolerates_empty_assembly(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from harness.app import create_app

    monkeypatch.chdir(tmp_path)
    app = create_app(
        ultra_factory=ZeroDrone,
        runs_root=tmp_path / "runs",
        providers_file=tmp_path / "providers.local.json",
        log_dir=str(tmp_path / "logs"),
        state_dir=tmp_path / "harness_state",
    )
    with TestClient(app) as client:
        response = client.post("/v1/splinter/curate", json={
            "query": "how does authentication work",
            "conversation_id": "floor-empty",
        })

    assert response.status_code == 200
    body = response.json()
    assert body["assembled_content"] == ""
    assert body["token_count"] == 0
    assert body["error"] is None
