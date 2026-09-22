"""Training engines: catalogue, command construction, launch bundle, bridge.

Pure-stdlib behaviour of ``harness.training`` — no GPU, no process spawn, no
network. The launch test intercepts ``training._popen``.
"""

from __future__ import annotations

import json

import pytest

from harness import training


class _FakeProc:
    pid = 424242


def _hive_fields(tmp_path):
    return {
        "model_dir": "/models/canary/hf",
        "corpus": "/corpus/tinyshakespeare.txt",
        "out": str(tmp_path / "out"),
        "steps": 20000,
        "seq": 512,
        "lr": 5e-5,
        "device": "cuda:1",
        "eval_regions": 8,
        "eval_windows": 8,
    }


def test_catalog_has_both_engines():
    ids = {e["id"] for e in training.engine_catalog()}
    assert ids == {training.ENGINE_UNSLOTH, training.ENGINE_HIVE_TERNARY}


def test_hive_ternary_recipes_and_unimplemented_flags():
    engine = training.get_engine(training.ENGINE_HIVE_TERNARY)
    methods = {m.id: m for m in engine.methods}
    assert methods["ste-rotate"].flags == ("--ste", "--rotate")
    assert methods["ste-rotate"].implemented is True
    assert methods["pack"].implemented is False


def test_unsloth_core_is_catalogue_only():
    engine = training.get_engine(training.ENGINE_UNSLOTH)
    assert engine.external is True
    assert all(m.implemented is False for m in engine.methods)


def test_build_command_rotation_ste(monkeypatch, tmp_path):
    monkeypatch.setattr(training, "ternary_runner", lambda: ["python", "runner.py"])
    argv = training.build_command(training.ENGINE_HIVE_TERNARY, "ste-rotate", _hive_fields(tmp_path))
    assert argv[:2] == ["python", "runner.py"]
    assert "--ste" in argv and "--rotate" in argv
    assert argv[argv.index("--steps") + 1] == "20000"
    assert argv[argv.index("--model-dir") + 1] == "/models/canary/hf"
    assert argv[argv.index("--corpus") + 1] == "/corpus/tinyshakespeare.txt"
    assert argv[argv.index("--eval-regions") + 1] == "8"


def test_build_command_optional_flags_drop_when_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(training, "ternary_runner", lambda: ["python", "runner.py"])
    fields = {"model_dir": "m", "corpus": "c", "out": "o"}
    argv = training.build_command(training.ENGINE_HIVE_TERNARY, "ste", fields)
    assert argv == ["python", "runner.py", "--ste", "--model-dir", "m", "--corpus", "c", "--out", "o"]


def test_build_command_missing_required(monkeypatch):
    monkeypatch.setattr(training, "ternary_runner", lambda: ["python", "runner.py"])
    with pytest.raises(ValueError, match="missing required"):
        training.build_command(training.ENGINE_HIVE_TERNARY, "ste", {"model_dir": "m"})


def test_build_command_rejects_unknown_and_unimplemented(monkeypatch, tmp_path):
    monkeypatch.setattr(training, "ternary_runner", lambda: ["python", "runner.py"])
    with pytest.raises(ValueError, match="unknown method"):
        training.build_command(training.ENGINE_HIVE_TERNARY, "nope", _hive_fields(tmp_path))
    with pytest.raises(ValueError, match="not implemented"):
        training.build_command(training.ENGINE_HIVE_TERNARY, "pack", _hive_fields(tmp_path))
    with pytest.raises(ValueError, match="not implemented"):
        training.build_command(training.ENGINE_UNSLOTH, "lora", _hive_fields(tmp_path))


def test_build_command_without_runner_is_an_error(monkeypatch, tmp_path):
    monkeypatch.setattr(training, "ternary_runner", lambda: [])
    with pytest.raises(RuntimeError, match="HIVE_TERNARY_RUNNER"):
        training.build_command(training.ENGINE_HIVE_TERNARY, "ste", _hive_fields(tmp_path))


def test_launch_creates_bundle(monkeypatch, tmp_path):
    monkeypatch.setattr(training, "ternary_runner", lambda: ["python", "runner.py"])
    monkeypatch.setattr(training, "_popen", lambda *a, **k: _FakeProc())
    result = training.launch(tmp_path, training.ENGINE_HIVE_TERNARY, "ste-rotate",
                             _hive_fields(tmp_path), name="ste-rotate 20k")
    run_dir = tmp_path / result["run_dir"]
    assert run_dir.is_dir()
    assert result["pid"] == 424242
    manifest = json.loads((run_dir / "training.json").read_text())
    assert manifest["engine"] == training.ENGINE_HIVE_TERNARY
    assert manifest["method"] == "ste-rotate"
    # out is forced to the run dir, not the caller's value
    assert manifest["config"]["out"] == str(run_dir)
    assert (run_dir / "run_stdout.log").exists()
    assert (run_dir / "pid").read_text() == "424242"


def test_launch_rejects_unsloth(tmp_path):
    with pytest.raises(ValueError, match="not implemented"):
        training.launch(tmp_path, training.ENGINE_UNSLOTH, "lora", {})


def test_parse_progress_reads_step_loss_ratio():
    log = (
        "[rmd] init step 0: deployed_ratio 5827.3511 (min 3368.666, max 10146.651)\n"
        "[rmd] step 500/20000 loss 1.41 306s\n"
        "[rmd] train step 500: deployed_ratio 2.2083 (min 1.740, max 2.665)\n"
        "[rmd] step 1000/20000 loss 1.64 625s\n"
    )
    progress = training.parse_progress(log)
    assert progress["step"] == 1000
    assert progress["total_steps"] == 20000
    assert progress["loss"] == 1.64
    assert progress["elapsed_s"] == 625
    assert progress["percent"] == 5.0
    assert progress["deployed_ratio"] == 2.2083
    assert progress["init_ratio"] == 5827.3511


def test_build_run_report_bridges_eval_regions(tmp_path):
    (tmp_path / "training.json").write_text(json.dumps({
        "engine": training.ENGINE_HIVE_TERNARY,
        "method": "ste-rotate",
        "config": {"steps": 20000},
        "argv": ["python", "runner.py", "--ste", "--rotate"],
    }))
    (tmp_path / "eval-regions.json").write_text(json.dumps({
        "regions": [
            {"name": "r0", "ppl": 12.0, "teacher_ppl": 10.0, "ratio": 1.2},
            {"name": "r1", "ppl": 11.0, "teacher_ppl": 10.0, "ratio": 1.1},
        ],
        "aggregate": {"n_regions": 2, "mean_ratio": 1.2152,
                      "min_ratio": 1.1, "max_ratio": 1.2},
    }))
    report = training.build_run_report(tmp_path)
    assert report["kind"] == "training"
    assert report["status"] == "finished"
    assert report["retention"] == pytest.approx(82.3, abs=0.1)
    assert len(report["regions"]) == 2


def test_run_status_persists_run_report(monkeypatch, tmp_path):
    (tmp_path / "training.json").write_text(json.dumps({
        "engine": training.ENGINE_HIVE_TERNARY, "method": "ste",
        "argv": ["python", "runner.py", "--ste"],
    }))
    (tmp_path / "run_stdout.log").write_text("[rmd] step 500/20000 loss 1.41 306s\n")
    monkeypatch.setattr(training, "_pid_alive", lambda pid: False)
    status = training.run_status(tmp_path)
    assert status["state"] == "stopped"
    assert (tmp_path / "run_report.json").is_file()
