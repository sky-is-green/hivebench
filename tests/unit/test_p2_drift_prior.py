"""P2-DILUTION — budget-floor-aware dilution + drift-reset prior.

Spec: STRATA-PLAN.md "P2-DILUTION". The drift reset currently boosts recency
(recent chunks x1.0, everything else x0.1), which can over-promote a *query
echo* — a chunk that merely restates the user's own words (the live 1.15
self-echo case) — above a genuine fact. The opt-in ``drift_prior="downweight"``
flag softly down-weights the echo's drift contribution instead of boosting
recency, and does not touch the P1 floor contract. With the flag off, behavior
is byte-identical to today.
"""

import numpy as np

from cortex.config import SplinterConfig
from cortex.routing import DroneRouter, EscalationHandler
from cortex.splinter import Splinter
from focal.assembly import AssembledContext, ContextAssembler
from focal.budget import AdaptiveBudget
from membrane.dedup import ContextDeduplicator
from membrane.drift import DriftResult
from retention.store import ContextStore
from sieve.scores import ChunkScore

ECHO = "what gpu do i have"
FACT = "my gpu is an rx 7900 xtx"
FILLER_A = "gardening tips for watering tomatoes"
FILLER_B = "the build finished after lunch"

_EMB = {FILLER_A: 0, FACT: 1, FILLER_B: 2, ECHO: 3}


def _embed(text):
    vec = np.zeros(8)
    vec[_EMB.get(text, 7)] = 1.0
    return vec


class FakeChunk:
    def __init__(self, cid, content):
        self.id = cid
        self.content = content
        self.relevance_history = []


class ScriptedDrone:
    """Echo text scores 1.15 (the live self-echo case), the genuine fact 0.9."""

    def score(self, query, chunks):
        out = []
        for i, text in enumerate(chunks):
            if "7900" in text:
                out.append(ChunkScore(i, 0.9, 1.0))
            elif "gpu do i have" in text:
                out.append(ChunkScore(i, 1.15, 1.0))
            else:
                out.append(ChunkScore(i, 0.2, 1.0))
        return out

    def embed(self, text):
        return _embed(text)


class AlwaysDrift:
    """Force the drift reset so the prior is actually exercised."""

    def check(self, recent_chunks, all_chunks, drone=None):
        return DriftResult(drift_score=1.0, should_reset=True)


def _store():
    store = ContextStore(embed_fn=_embed)
    store.add_chunk(1, FILLER_A)
    store.add_chunk(2, FACT)
    store.add_chunk(3, FILLER_B)
    store.add_chunk(4, ECHO)
    return store


def _assemble(**kwargs):
    return ContextAssembler().assemble(
        query=ECHO,
        current_turn=4,
        store=_store(),
        router=DroneRouter(),
        ultra_small=ScriptedDrone(),
        medium=ScriptedDrone(),
        escalation=EscalationHandler(),
        dedup=ContextDeduplicator(),
        drift_detector=AlwaysDrift(),
        budget=AdaptiveBudget(),
        max_context=8192,
        skip_remembrance=True,
        **kwargs,
    )


def test_drift_prior_off_is_current_behavior():
    echo = FakeChunk("echo", ECHO)
    fact = FakeChunk("fact", FACT)

    default = ContextAssembler._apply_drift_reset([echo, fact], [echo])
    explicit = ContextAssembler._apply_drift_reset(
        [echo, fact], [echo], query=ECHO, drift_prior="off", drift_prior_factor=0.5
    )

    assert default == {"echo": 1.0, "fact": 0.1}
    assert explicit == default


def test_downweight_suppresses_recent_echo_boost():
    echo = FakeChunk("echo", ECHO)
    fact = FakeChunk("fact", FACT)

    pen = ContextAssembler._apply_drift_reset(
        [echo, fact], [echo, fact], query=ECHO,
        drift_prior="downweight", drift_prior_factor=0.5,
    )

    assert pen["echo"] == 0.5
    assert pen["fact"] == 1.0


def test_downweight_factor_is_tunable():
    echo = FakeChunk("echo", ECHO)

    pen = ContextAssembler._apply_drift_reset(
        [echo], [echo], query=ECHO,
        drift_prior="downweight", drift_prior_factor=0.25,
    )

    assert pen["echo"] == 0.25


def test_query_echo_wins_without_flag():
    result = _assemble()

    assert result.top_raw_score == 1.15
    assert result.content.index(ECHO) < result.content.index(FACT)


def test_query_echo_ranks_below_genuine_fact_when_flag_on():
    result = _assemble(drift_prior="downweight", drift_prior_factor=0.5)

    assert len(result.selected_chunk_ids) == 2
    assert result.content.index(FACT) < result.content.index(ECHO)


def test_assemble_default_equals_explicit_off():
    default = _assemble()
    explicit = _assemble(drift_prior="off")

    assert default.content == explicit.content
    assert default.selected_chunk_ids == explicit.selected_chunk_ids
    assert default.token_count == explicit.token_count


def test_process_turn_forwards_drift_prior(monkeypatch):
    captured = {}

    def fake_assemble(self, **kwargs):
        captured.update(kwargs)
        return AssembledContext(
            content="", token_count=0, budget=0, chunks_used=0,
            routing_decision=None, drift_detected=False,
        )

    monkeypatch.setattr(ContextAssembler, "assemble", fake_assemble)
    splinter = Splinter(config=SplinterConfig(), ultra=ScriptedDrone(), store=_store())

    splinter.process_turn(ECHO, drift_prior="downweight", drift_prior_factor=0.25)

    assert captured.get("drift_prior") == "downweight"
    assert captured.get("drift_prior_factor") == 0.25


def test_process_turn_defaults_drift_prior_off(monkeypatch):
    captured = {}

    def fake_assemble(self, **kwargs):
        captured.update(kwargs)
        return AssembledContext(
            content="", token_count=0, budget=0, chunks_used=0,
            routing_decision=None, drift_detected=False,
        )

    monkeypatch.setattr(ContextAssembler, "assemble", fake_assemble)
    splinter = Splinter(config=SplinterConfig(), ultra=ScriptedDrone(), store=_store())

    splinter.process_turn(ECHO)

    assert captured.get("drift_prior") == "off"
    assert captured.get("drift_prior_factor") == 0.5
