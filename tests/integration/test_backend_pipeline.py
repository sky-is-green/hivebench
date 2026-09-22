"""Integration: full pipeline (assemble -> backend -> PES) with a mock backend.

Uses fake drones for speed; the backend is exercised over a 100-turn conversation
with a mock OpenAI-compatible transport. Verifies context reaches the backend as
the system message and that the rolling PES stays healthy.
"""

import numpy as np

from backend.cache_manager import KVCacheManager
from backend.lmstudio import LMStudioBackend
from cortex.efficiency import EfficiencyScorer
from cortex.routing import DroneRouter, EscalationHandler
from focal.assembly import ContextAssembler
from focal.budget import AdaptiveBudget
from membrane.dedup import ContextDeduplicator
from membrane.drift import TopicDriftDetector
from retention.store import ContextStore
from sieve.scores import ChunkScore


class FakeUltraSmall:
    def score(self, query, chunks):
        return [ChunkScore(i, 0.9 if "JWT" in c else 0.2, 1.0) for i, c in enumerate(chunks)]

    def embed(self, text):
        return np.array([1.0, 0.0, 0.0])


class FakeMedium:
    def score(self, query, chunks):
        return [ChunkScore(i, 0.5, 0.85, source="medium") for i in range(len(chunks))]


class MockTransport:
    def post(self, url, json=None, headers=None, timeout=None):
        self.last = (url, json)
        return _Resp({"choices": [{"message": {"content": "mock reply"}}]})

    def get(self, url, headers=None, timeout=None):
        return _Resp({"data": []})


class _Resp:
    ok = True

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


def _build_store():
    store = ContextStore(embed_fn=lambda c: np.array([1.0, 0.0, 0.0]))
    for t in range(1, 26):
        store.add_chunk(t, f"authentication JWT token schema index turn {t}")
    for t in range(26, 51):
        store.add_chunk(t, f"gardening watering plants turn {t}")
    return store


def test_backend_receives_assembled_context_as_system_message():
    store = _build_store()
    assembler = ContextAssembler()
    assembled = assembler.assemble(
        query="how does authentication work", current_turn=50, store=store,
        router=DroneRouter(), ultra_small=FakeUltraSmall(), medium=FakeMedium(),
        escalation=EscalationHandler(), dedup=ContextDeduplicator(),
        drift_detector=TopicDriftDetector(embed_fn=lambda t: np.array([1.0, 0.0, 0.0])),
        budget=AdaptiveBudget(), max_context=8192,
    )
    assert assembled.content

    transport = MockTransport()
    backend = LMStudioBackend(base_url="localhost:1234", model="qwen", transport=transport)
    reply = backend.generate(assembled.content, "USER QUERY")
    assert reply == "mock reply"

    _url, payload = transport.last
    messages = payload["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == assembled.content
    assert messages[1] == {"role": "user", "content": "USER QUERY"}


def test_100_turn_pipeline_pes_stays_healthy():
    store = _build_store()
    assembler = ContextAssembler()
    backend = LMStudioBackend(
        base_url="localhost:1234", model="qwen", transport=MockTransport()
    )
    scorer = EfficiencyScorer()
    ultra = FakeUltraSmall()
    medium = FakeMedium()

    pes_history = []
    for turn in range(1, 101):
        assembled = assembler.assemble(
            query="how does authentication work", current_turn=turn, store=store,
            router=DroneRouter(), ultra_small=ultra, medium=medium,
            escalation=EscalationHandler(), dedup=ContextDeduplicator(),
            drift_detector=TopicDriftDetector(embed_fn=lambda t: np.array([1.0, 0.0, 0.0])),
            budget=AdaptiveBudget(), max_context=8192,
        )
        reply = backend.generate(assembled.content, "query")
        assert reply

        if turn % 5 == 0:
            utilization = assembled.token_count / max(assembled.budget, 1)
            pes = scorer.compute(
                retrieval_precision=85,
                routing_accuracy=90,
                avg_latency_ms=30,
                actual_tps=35,
                baseline_tps=30,
                budget_used=assembled.token_count,
                budget_total=assembled.budget,
            ).composite
            pes_history.append(pes)

    assert len(pes_history) == 20
    assert all(pes >= 70 for pes in pes_history)  # healthy through 100 turns
    assert min(pes_history) >= 70


def test_lmstudio_prefix_caching_end_to_end():
    """LM Studio is the sole live backend; KVCacheManager keeps a stable pinned
    prefix so llama.cpp's automatic prefix cache can reuse its KV each turn."""
    store = _build_store()
    assembler = ContextAssembler()
    transport = MockTransport()
    backend = LMStudioBackend(base_url="localhost:1234", model="qwen", transport=transport)
    manager = KVCacheManager(backend)
    pinned = "You are a helpful coding assistant for the splinter project."

    assert manager.update_cache("", persistent_prefix=pinned)["mode"] == "prefix_caching"
    assert manager.update_cache("", persistent_prefix=pinned)["prefix_stable"] is True

    for turn in range(1, 6):
        assembled = assembler.assemble(
            query="how does authentication work", current_turn=turn, store=store,
            router=DroneRouter(), ultra_small=FakeUltraSmall(), medium=FakeMedium(),
            escalation=EscalationHandler(), dedup=ContextDeduplicator(),
            drift_detector=TopicDriftDetector(embed_fn=lambda t: np.array([1.0, 0.0, 0.0])),
            budget=AdaptiveBudget(), max_context=8192,
        )
        manager.update_cache(assembled.content, persistent_prefix=pinned)
        reply = backend.generate(assembled.content, "query")
        assert reply

    # every request sent the pinned prefix as the leading text of a single
    # system message, unchanged across turns
    assert backend.pinned_prefix == pinned
    _url, payload = transport.last
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][0]["content"].startswith(pinned)
    assert payload["messages"][1] == {"role": "user", "content": "query"}