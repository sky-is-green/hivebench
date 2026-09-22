"""Integration: unified data-generation workflow (mock mode)."""

import json

from experiments.generate_data import _acquire_run_lock, main


def test_generate_data_mock_run(tmp_path):
    out = tmp_path / "run"
    code = main([
        "--mock", "--conversations", "tests/fixtures/generated",
        "--max-convs", "2", "--max-turns", "3", "--output", str(out),
    ])
    assert code == 0

    report = json.loads((out / "run_report.json").read_text(encoding="utf-8"))
    assert report["mode"] == "mock"
    assert report["aggregate"]["conversations"] == 2
    assert report["aggregate"]["user_turns"] >= 1
    assert (out / "ground_truth.sqlite").exists()
    assert list((out / "logs").glob("events-*.ndjson"))
    # post-run PES + ground-truth metrics are reported
    assert report["post_run_pes"]["pes"] >= 0.0
    assert report["post_run_pes"]["band"] in ("GREEN", "YELLOW", "RED", "CRITICAL")
    assert report["ground_truth"]["auditor_labels"] >= 1
    assert report["ground_truth"]["routing_accuracy"] is not None


def test_generate_data_mock_with_protocol(tmp_path):
    out = tmp_path / "run2"
    code = main([
        "--mock", "--protocol", "--conversations", "tests/fixtures/generated",
        "--max-convs", "1", "--max-turns", "3", "--output", str(out),
    ])
    assert code == 0
    report = json.loads((out / "run_report.json").read_text(encoding="utf-8"))
    assert report["protocol"] is not None
    assert len(report["protocol"]) == 11
    assert all(r["status"] in ("PASS", "FAIL", "SKIP", "REPORT") for r in report["protocol"])


def test_generate_data_resume_roundtrip(tmp_path):
    out = tmp_path / "run3"
    code = main([
        "--mock", "--conversations", "tests/fixtures/generated",
        "--max-convs", "2", "--max-turns", "3", "--output", str(out),
        "--checkpoint-every", "1",
    ])
    assert code == 0
    ckpt = json.loads((out / "checkpoint.json").read_text(encoding="utf-8"))
    assert ckpt["progress"]["done"] > 0
    assert ckpt["run_id"]
    first = json.loads((out / "run_report.json").read_text(encoding="utf-8"))
    first_turns = first["aggregate"]["user_turns"]

    # resume from the checkpoint; the run must complete with the same results
    code2 = main(["--resume", str(out), "--mock"])
    assert code2 == 0
    report = json.loads((out / "run_report.json").read_text(encoding="utf-8"))
    assert report["mode"] == "mock"
    assert report["aggregate"]["conversations"] == 2
    assert report["aggregate"]["user_turns"] == first_turns  # no turns lost on resume
    assert (out / "checkpoint.json").exists()


def test_generate_data_resume_stale_conversations_path_falls_back(tmp_path):
    out = tmp_path / "run5"
    code = main([
        "--mock", "--conversations", "tests/fixtures/generated",
        "--max-convs", "1", "--max-turns", "2", "--output", str(out),
        "--checkpoint-every", "1",
    ])
    assert code == 0
    ckpt_path = out / "checkpoint.json"
    ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
    # simulate a checkpoint written before a repo restructure: the stored
    # conversations path no longer resolves (found live 2026-08-24: a stale
    # path silently produced 0 conversations and the FIFO baseline exited 2)
    ckpt["run_args"]["conversations"] = "tests/fixtures/generated"
    ckpt_path.write_text(json.dumps(ckpt), encoding="utf-8")

    code2 = main(["--resume", str(out), "--mock", "--baselines"])
    assert code2 == 0
    report = json.loads((out / "run_report.json").read_text(encoding="utf-8"))
    assert report["baselines"]["lm_studio"]["exit"] == 0
    assert report["baselines"]["fifo"]["exit"] == 0
    assert (out / "baseline_fifo.json").exists()


def test_generate_data_max_tokens_cap(tmp_path):
    out = tmp_path / "run4"
    code = main([
        "--mock", "--conversations", "tests/fixtures/generated",
        "--max-convs", "1", "--max-turns", "2", "--output", str(out),
        "--max-tokens", "32",
    ])
    assert code == 0
    report = json.loads((out / "run_report.json").read_text(encoding="utf-8"))
    assert report["aggregate"]["user_turns"] >= 1


def test_generate_data_confidence_flag(tmp_path):
    out = tmp_path / "run5"
    code = main([
        "--mock", "--conversations", "tests/fixtures/generated",
        "--max-convs", "1", "--max-turns", "2", "--output", str(out),
        "--confidence", "off",
    ])
    assert code == 0
    assert (out / "run_report.json").exists()
    assert not (out / "run.lock").exists()  # lock released on completion


def test_run_lock_same_process_allowed(tmp_path):
    assert _acquire_run_lock(tmp_path) is True
    assert _acquire_run_lock(tmp_path) is True  # same PID (in-process) is allowed


def test_run_lock_dead_pid_overwritten(tmp_path):
    (tmp_path / "run.lock").write_text("999999999")  # PID that cannot exist
    assert _acquire_run_lock(tmp_path) is True


def test_run_lock_live_foreign_pid_refused(tmp_path, monkeypatch):
    from experiments import generate_data

    (tmp_path / "run.lock").write_text("12345")
    monkeypatch.setattr(generate_data, "_pid_alive", lambda pid: pid == 12345)
    assert _acquire_run_lock(tmp_path) is False


def test_per_conversation_store_isolation(tmp_path):
    """Chunks from one conversation must not leak into another.

    A benchmark run over many conversations shares one Splinter; without a
    per-conversation reset, later conversations retrieve mostly *other*
    conversations' chunks, collapsing P2 precision. This regression test runs
    two conversations and asserts the second one's context store contains no
    first-conversation content.
    """
    import os

    from cortex.config import SplinterConfig
    from cortex.e2e import FakeUltraSmall, MockTransport
    from cortex.splinter import Splinter
    from backend.lmstudio import LMStudioBackend
    from cortex.baselines.runner import load_conversations

    convs = load_conversations("tests/fixtures/generated")
    # edge_001 is about the order schema; edge_002 about the deploy pipeline.
    first = next(c for c in convs if c["conversation_id"] == "edge_001")
    second = next(c for c in convs if c["conversation_id"] == "edge_002")

    config = SplinterConfig(confidence_mode="off", sanitize_context=False)
    splinter = Splinter(
        config=config,
        ultra=FakeUltraSmall(),
        medium=__import__("sieve.medium", fromlist=["MediumDrone"]).MediumDrone(
            score_pair_fn=lambda q, c: 0.5
        ),
        backend=LMStudioBackend(base_url="http://localhost:1234", transport=MockTransport()),
    )

    # Conversation 1 fills the store with order-schema chunks.
    for td in first["turns"]:
        if td.get("role") != "user":
            continue
        splinter.process_turn(td["content"])
    first_chunks = [c.content for c in splinter.store.all_chunks()]
    assert first_chunks and "order schema" in " ".join(first_chunks).lower()

    # Conversation 2 starts from a fresh store: no order-schema content.
    splinter.reset_conversation()
    for td in second["turns"]:
        if td.get("role") != "user":
            continue
        splinter.process_turn(td["content"])
    second_chunks = [c.content for c in splinter.store.all_chunks()]
    assert second_chunks
    joined = " ".join(second_chunks).lower()
    assert "order schema" not in joined
    assert "blue-green" in joined