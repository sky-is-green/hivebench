"""run_compare — side-by-side run bundle comparison (offline)."""

import json

from experiments.run_compare import compare, load_run


def _write_run(tmp_path, name: str, pes: float, recall: float, ingestion: float,
               protocol: list[dict], turns: int = 100, convs: int = 5) -> None:
    run_dir = tmp_path / name
    run_dir.mkdir()
    report = {
        "mode": "live",
        "aggregate": {"user_turns": turns, "conversations": convs},
        "post_run_pes": {"pes": pes, "band": "GREEN" if pes >= 80 else "YELLOW",
                         "components": {}},
        "retrieval_diagnostic": {
            "retrieval_recall": recall, "ingestion_rate": ingestion,
            "perfect_hive_ceiling": min(ingestion, 100.0),
            "retrieval_precision": 11.7,
        },
        "protocol": protocol,
        "comb": {"total": {"archived": 3, "resurrected": 2}},
    }
    (run_dir / "run_report.json").write_text(json.dumps(report), encoding="utf-8")


def test_compare_reports_no_regression(tmp_path, capsys):
    _write_run(tmp_path, "base", 80.0, 90.3, 48.4, [
        {"id": "P1", "status": "PASS"}, {"id": "P2", "status": "FAIL"},
    ])
    _write_run(tmp_path, "later", 82.0, 91.0, 50.0, [
        {"id": "P1", "status": "PASS"}, {"id": "P2", "status": "FAIL"},
    ])
    code = compare([tmp_path / "base", tmp_path / "later"])
    out = capsys.readouterr().out
    assert code == 0
    assert "No regressions" in out
    assert "90.3" in out and "91.0" in out


def test_compare_flags_regression(tmp_path, capsys):
    _write_run(tmp_path, "base", 80.0, 90.3, 48.4, [])
    _write_run(tmp_path, "worse", 71.0, 75.7, 54.3, [])
    code = compare([tmp_path / "base", tmp_path / "worse"])
    out = capsys.readouterr().out
    assert code == 1
    assert "REGRESSIONS" in out
    assert "P2 recall" in out


def test_compare_missing_report(tmp_path):
    (tmp_path / "empty").mkdir()
    assert compare([tmp_path / "empty"]) == 2


def test_load_run_missing_fields(tmp_path):
    (tmp_path / "bare").mkdir()
    (tmp_path / "bare" / "run_report.json").write_text('{"mode": "mock"}', encoding="utf-8")
    loaded = load_run(tmp_path / "bare")
    assert loaded["pes"] is None
    assert loaded["verdicts"] == {}