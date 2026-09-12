"""Unit tests for retention.decay (S2.3)."""

import pytest

from retention.decay import DecayMatrix
from retention.store import ContextStore


def _chunk(content, turn, last_ref=None):
    store = ContextStore()
    cid = store.add_chunk(turn, content)
    c = store.chunks[cid]
    c.last_referenced_turn = last_ref if last_ref is not None else turn
    return c


def test_effective_decreases_with_age():
    matrix = DecayMatrix()
    fresh = _chunk("fresh content", turn=50, last_ref=50)   # age 0
    old = _chunk("old content", turn=50, last_ref=20)       # age 30
    raw = {fresh.id: 1.0, old.id: 1.0}
    eff = matrix.apply([fresh, old], 50, raw)
    assert eff[fresh.id] == pytest.approx(1.0)
    assert eff[old.id] < eff[fresh.id]


def test_stale_detection_at_20_turns():
    matrix = DecayMatrix()
    stale = _chunk("c", turn=50, last_ref=29)   # age 21 > 20 -> stale
    raw = {stale.id: 1.0}
    eff = matrix.apply([stale], 50, raw)
    # age_factor = 2.1, decay 1 -> 1 / 1 = 1, then stale *0.5
    assert eff[stale.id] == pytest.approx(0.5)


def test_decay_multiplier_penalizes_resaved_chunks():
    matrix = DecayMatrix()
    a = _chunk("alpha chunk", turn=50, last_ref=40)  # age 10, decay 1
    b = _chunk("beta chunk", turn=50, last_ref=40)   # age 10, decay 3
    b.decay_multiplier = 3.0
    raw = {a.id: 1.0, b.id: 1.0}
    eff = matrix.apply([a, b], 50, raw)
    # a: 1 / 1^1 = 1 ; b: 1 / 3^1 = 0.333
    assert eff[a.id] == pytest.approx(1.0)
    assert eff[b.id] == pytest.approx(1.0 / 3.0)


def test_drift_penalty_multiplies():
    matrix = DecayMatrix()
    c = _chunk("c", turn=50, last_ref=50)
    raw = {c.id: 1.0}
    eff = matrix.apply([c], 50, raw, drift_penalties={c.id: 0.1})
    assert eff[c.id] == pytest.approx(0.1)


def test_missing_raw_score_is_zero():
    matrix = DecayMatrix()
    c = _chunk("c", turn=50, last_ref=50)
    eff = matrix.apply([c], 50, {})
    assert eff[c.id] == 0.0