"""Unit tests for cortex.routing (DroneRouter + EscalationHandler)."""

from cortex.routing import DroneRouter, EscalationHandler
from sieve.scores import ChunkScore

SIMPLE_QUERIES = [
    "hello", "what time is it", "good morning", "thanks", "what is 2+2",
    "tell me a joke", "how are you", "bye", "ok", "who are you",
    "what day is it today", "please repeat that", "no thanks", "sounds good",
    "let's continue", "can you help", "fine", "sure", "maybe later", "not sure",
    "that makes sense", "got it", "interesting", "hmm", "great",
]

COMPLEX_QUERIES = [
    "refactor and analyze the authentication module",
    "design and review the database schema",
    "debug and optimize the slow query",
    "explain and audit the security posture",
    "compare and analyze the two architectures",
    "refactor and optimize the legacy codebase",
    "design and debug the new service",
    "review and audit the payment flow",
    "analyze and compare the caching strategies",
    "explain and optimize the retry logic",
    "refactor the router and analyze the routing decisions",
    "debug the endpoint and explain why latency is high",
    "design the architecture, review the plan, and audit the risks",
    "analyze, compare, and optimize all three approaches",
    "explain the schema and audit the migrations",
    "refactor the strata, analyze the decay, and optimize the budget",
    "debug the router and review the escalation handler",
    "design and audit the checkpoint system",
    "review and optimize the embedding cache",
    "compare and design the drone fleet",
    "analyze the auditor and explain the ground truth labels",
    "refactor and debug the context assembler",
    "optimize and review the congestion detector",
    "design, debug, and analyze the full pipeline",
    "explain how the pagedattention kv-cache works and audit the backend",
]


def test_router_85_percent_correct_on_50_labeled():
    router = DroneRouter()
    correct = 0
    total = 0
    for q in SIMPLE_QUERIES:
        decision = router.route(q)
        total += 1
        if decision.route_to == "ultra_small":
            correct += 1
    for q in COMPLEX_QUERIES:
        decision = router.route(q)
        total += 1
        if decision.route_to in ("medium", "escalation"):
            correct += 1
    assert total == 50
    assert correct / total >= 0.85


def test_router_reason_and_confidence():
    router = DroneRouter()
    simple = router.route("hello")
    assert simple.route_to == "ultra_small"
    assert simple.confidence == 0.9
    assert simple.reason  # non-empty

    complex_q = router.route("refactor and analyze the authentication module")
    assert complex_q.route_to in ("medium", "escalation")
    assert "keyword:" in complex_q.reason


def test_router_code_density_and_length():
    router = DroneRouter()
    query = "here is the code:\n```\ndef a():\n```\n```\ndef b():\n```\n```\ndef c():\n```\nfix it"
    decision = router.route(query)
    # 3 code blocks -> +2, plus "fix"? fix not a keyword. So score=2 -> medium.
    assert decision.route_to in ("medium", "escalation")

    long_query = "x" * 501
    decision2 = router.route(long_query)
    # length +1 only -> still ultra_small (score 1)
    assert decision2.route_to == "ultra_small"


def test_router_depth_boost():
    router = DroneRouter()
    history = list(range(40))  # depth > 30
    # one keyword + depth = 2 -> medium
    decision = router.route("debug the problem", history)
    assert decision.route_to in ("medium", "escalation")


# ---------------------------------------------------------------------------
# EscalationHandler
# ---------------------------------------------------------------------------
class FakeUltraSmall:
    def __init__(self, pairs):
        self.pairs = pairs  # list[(score, confidence)]
        self.last_chunks = None

    def score(self, query, chunks):
        self.last_chunks = list(chunks)
        return [ChunkScore(i, s, c) for i, (s, c) in enumerate(self.pairs)]


class FakeMedium:
    def __init__(self, score_fn):
        self.score_fn = score_fn
        self.seen = []

    def score(self, query, chunks):
        self.seen.append(list(chunks))
        return [
            ChunkScore(i, self.score_fn(c), 0.9, source="medium")
            for i, c in enumerate(chunks)
        ]


def test_escalation_invokes_medium_only_for_uncertain_chunks():
    chunks = ["c0", "c1", "c2", "c3"]
    us = FakeUltraSmall([(0.8, 0.4), (0.9, 0.9), (0.5, 0.3), (0.75, 0.5)])
    medium = FakeMedium(lambda c: 0.99)
    handler = EscalationHandler()

    result = handler.process("query", chunks, us, medium)

    # medium saw only chunks 0 and 3 (high score + low confidence)
    assert medium.seen == [["c0", "c3"]]

    # uncertain chunks merged with medium scores
    assert result[0].source == "medium_validated"
    assert result[0].relevance_score == 0.99
    assert result[0].confidence == 0.9
    assert result[3].source == "medium_validated"
    assert result[3].relevance_score == 0.99

    # confident / low-score chunks untouched
    assert result[1].source == "ultra_small" and result[1].confidence == 0.9
    assert result[2].source == "ultra_small" and result[2].confidence == 0.3


def test_escalation_skips_medium_when_all_confident():
    chunks = ["a", "b"]
    us = FakeUltraSmall([(0.9, 0.9), (0.8, 0.95)])
    medium = FakeMedium(lambda c: 1.0)
    result = EscalationHandler().process("q", chunks, us, medium)
    assert medium.seen == []
    assert all(s.source == "ultra_small" for s in result)