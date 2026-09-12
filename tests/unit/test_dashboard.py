"""Unit tests for experiments.dashboard (terminal dashboard + keep-awake)."""

from experiments.dashboard import KeepAwake, TermDashboard


def test_term_dashboard_noop_when_not_tty():
    # pytest captures stdout (not a TTY), so an enabled request degrades to no-op
    t = TermDashboard(enabled=True)
    assert t.enabled is False
    t.set_phase("x")
    t.update_progress(1, 10, 2.0, 3.0, 1, 2, 5, 10)
    t.add_turn("q", "r", 61.0, 100.0, 200.0)
    t.add_line("line")
    t.close()


def test_term_dashboard_render_block():
    t = TermDashboard(enabled=False)
    t.phase = "1/3 E2E (2 convs)"
    t.update_progress(5, 10, 60.0, 120.0, 1, 2, 5, 10)
    t.add_turn("query text here", "reply text here", 60.0, 100.0, 200.0)
    block = t._render()
    assert any("HiveBench" in line for line in block)
    assert any("ETA" in line for line in block)
    assert any("50.0%" in line for line in block)
    assert any("recent turns" in line for line in block)
    assert any("query text here" in line for line in block)


def test_keep_awake_lifecycle():
    k = KeepAwake()
    # active may be False on non-Windows, but close() must always be safe
    assert isinstance(k.active, bool)
    k.close()


def test_keep_awake_close_twice():
    k = KeepAwake()
    k.close()
    k.close()  # idempotent