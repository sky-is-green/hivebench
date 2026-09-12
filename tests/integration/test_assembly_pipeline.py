"""Integration test: full assembly pipeline with the real ultra-small drone."""

import json
from pathlib import Path

import numpy as np
import pytest

from cortex.routing import DroneRouter, EscalationHandler
from focal.assembly import ContextAssembler
from focal.budget import AdaptiveBudget
from membrane.dedup import ContextDeduplicator
from membrane.drift import TopicDriftDetector
from retention.store import ContextStore
from sieve.medium import MediumDrone
from sieve.ultra_small import UltraSmallDrone

GENERATED = Path("hivebench/tests/fixtures/generated")


def _real_drone():
    try:
        drone = UltraSmallDrone()
        drone._ensure_loaded()
        return drone
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"all-MiniLM-L6-v2 unavailable: {exc}")


def _build_store_from_conversation(drone, conv, limit=30):
    store = ContextStore(embed_fn=drone.embed)
    for i, turn in enumerate(conv["turns"][:limit], start=1):
        store.add_chunk(i, turn["content"])
    return store


def test_assembly_pipeline_e2e_real_drone():
    drone = _real_drone()
    conv = json.loads((GENERATED / "medium_001.json").read_text(encoding="utf-8"))
    store = _build_store_from_conversation(drone, conv)

    assembler = ContextAssembler()
    result = assembler.assemble(
        query="How does authentication work in the auth service?",
        current_turn=30,
        store=store,
        router=DroneRouter(),
        ultra_small=drone,
        medium=MediumDrone(score_pair_fn=lambda q, c: 0.5),
        escalation=EscalationHandler(),
        dedup=ContextDeduplicator(),
        drift_detector=TopicDriftDetector(embed_fn=drone.embed),
        budget=AdaptiveBudget(),
        max_context=8192,
    )

    assert result.chunks_used > 0
    assert result.token_count <= result.budget
    assert result.budget > 0
    utilization = result.token_count / result.budget
    assert 0.0 < utilization <= 1.0
    assert result.content.strip()


def test_assembly_pipeline_remembers_across_turns():
    """Running assembly over successive turns yields valid context each time."""
    drone = _real_drone()
    conv = json.loads((GENERATED / "short_001.json").read_text(encoding="utf-8"))
    store = _build_store_from_conversation(drone, conv, limit=12)

    assembler = ContextAssembler()
    medium = MediumDrone(score_pair_fn=lambda q, c: 0.5)
    results = []
    for turn in range(1, 11):
        query = "continue working on the auth service"
        results.append(
            assembler.assemble(
                query=query, current_turn=turn, store=store,
                router=DroneRouter(), ultra_small=drone, medium=medium,
                escalation=EscalationHandler(), dedup=ContextDeduplicator(),
                drift_detector=TopicDriftDetector(embed_fn=drone.embed),
                budget=AdaptiveBudget(), max_context=8192,
            )
        )

    assert all(r.token_count <= r.budget for r in results)
    assert all(r.content for r in results)