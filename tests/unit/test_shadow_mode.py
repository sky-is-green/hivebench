"""Unit tests for testing.shadow_mode (S5.1)."""

from testing.shadow_mode import ShadowMode


def _metrics(pes):
    return {"pes": pes}


def test_identical_configs_within_noise_continue():
    shadow = ShadowMode(
        lambda q, t: _metrics(80.0),
        lambda q, t: _metrics(80.5),  # within 5% noise margin
        n_turns=10,
    )
    for i in range(10):
        shadow.process_turn("q", i)
    ev = shadow.evaluate_after()
    assert ev.recommend == "continue"
    assert ev.production_avg == 80.0


def test_better_shadow_promotes():
    shadow = ShadowMode(
        lambda q, t: _metrics(80.0),
        lambda q, t: _metrics(95.0),
        n_turns=10,
    )
    for i in range(10):
        shadow.process_turn("q", i)
    ev = shadow.evaluate_after()
    assert ev.recommend == "promote"
    assert ev.shadow_avg == 95.0
    assert ev.improvement == 15.0


def test_worse_shadow_discards():
    shadow = ShadowMode(
        lambda q, t: _metrics(80.0),
        lambda q, t: _metrics(60.0),
        n_turns=10,
    )
    for i in range(10):
        shadow.process_turn("q", i)
    assert shadow.evaluate_after().recommend == "discard"


def test_process_turn_returns_production_and_logs_shadow():
    shadow = ShadowMode(
        lambda q, t: {"pes": 70},
        lambda q, t: {"pes": 71},
    )
    result = shadow.process_turn("q", 3)
    assert result == {"pes": 70}  # only production reaches the caller
    assert len(shadow.shadow_log) == 1
    assert shadow.shadow_log[0]["shadow"] == {"pes": 71}


def test_no_turns_continue():
    ev = ShadowMode(lambda q, t: {}, lambda q, t: {}).evaluate_after()
    assert ev.recommend == "continue"