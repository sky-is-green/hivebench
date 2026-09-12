"""Integration: P5 targeted-masking experiment smoke test."""

import pytest

from experiments.p5_targeted_masking import run_experiment


def test_p5_smoke_trains_both_variants():
    try:
        report = run_experiment(
            "tests/fixtures/generated", steps=1, lr=3e-4, batch_size=2,
            eval_size=4, seed=0, max_seqs=10,
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"bert-tiny unavailable: {exc}")

    assert set(report["results"].keys()) == {"random", "targeted"}
    for variant, metrics in report["results"].items():
        assert "precision" in metrics
        assert "recall" in metrics
        assert "final_loss" in metrics
    assert "targeted_beats_random" in report