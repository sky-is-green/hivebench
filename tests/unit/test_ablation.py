"""Unit tests for testing.ablation (S4.5)."""

from testing.ablation import ABLATION_CONFIGS, AblationRunner


def _make_hive(config):
    """Fake splinter: each disabled component costs 5 PES points off a base of 80."""

    def process_turn(query, turn):
        base = 80.0
        for comp in ("decay", "drones", "remembrance", "dedup", "adaptive", "drift"):
            if not config.get(comp, True):
                base -= 5.0
        return {"pes": base}

    return process_turn


def test_all_8_configs_run():
    conv = [{"turns": [{"content": f"q{i}"} for i in range(20)]}]
    runner = AblationRunner()
    result = runner.run(conv, _make_hive, key="pes")

    assert set(result.results.keys()) == {c["name"] for c in ABLATION_CONFIGS}
    assert len(result.results) == 8
    for name, agg in result.results.items():
        assert agg["pes"] is not None


def test_full_is_baseline_and_contributions_negative():
    conv = [{"turns": [{"content": f"q{i}"} for i in range(20)]}]
    result = AblationRunner().run(conv, _make_hive, key="pes")

    full = result.results["full"]["pes"]
    assert full == 80.0
    # contribution = full - ablated, so a disabled component yields a positive
    # delta (full outperforms the ablated config).
    for name, delta in result.contributions.items():
        if name == "baseline":
            continue
        assert delta > 0
    assert result.contributions["baseline"] == 80.0 - 50.0  # all 6 off


def test_baseline_lower_bound():
    conv = [{"turns": [{"content": f"q{i}"} for i in range(20)]}]
    result = AblationRunner().run(conv, _make_hive, key="pes")
    assert result.results["baseline"]["pes"] < result.results["full"]["pes"]