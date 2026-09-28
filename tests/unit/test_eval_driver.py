"""T11 — offline tests for the ternary eval driver.

No GPU, network, or sibling Splinter checkout: mock mode uses the in-module stub
manager, chat and paired_ab calls are injected. The real
``harness.models.LlamaServerManager`` integration test skips unless the F6
``SPLINTER_HOME`` pin makes the harness package importable.
"""

from __future__ import annotations

import json
import socket
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments import ternary_eval as te

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT_PATH = ROOT / "pyproject.toml"
SPEC_SHA256 = "0d2c008b4aee726351f9b90e44ec003c18b579d8690db24c77a089d9e1fc652b"


def _unused_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_spec_hash_is_pinned() -> None:
    # The spec file now lives with the ternary package in the forensics repo
    # (`bonsai_forensics/spec.md`), where tests/ternary/test_spec.py pins its
    # hash. Here we only pin the driver's copy of the constant.
    assert te.SPEC_SHA256 == SPEC_SHA256


def test_cli_entry_point_registered() -> None:
    cfg = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
    assert cfg["project"]["scripts"]["hivebench-ternary"] == "experiments.ternary_eval:main"
    assert callable(te.main)


def test_serve_gguf_delegates_to_manager(tmp_path: Path) -> None:
    manager, gguf = te.build_mock_manager(tmp_path)
    instance = te.serve_gguf(gguf, manager=manager, ngl=42, ctx_size=2048)
    assert instance["key"] == gguf.stem
    assert instance["base_url"] == f"http://127.0.0.1:{instance['port']}"
    assert instance["running"] is True
    assert manager.status()["running"] is True
    assert manager.unload(instance["key"]) == {"ok": True, "unloaded": gguf.stem}


def test_serve_gguf_rejects_missing_file(tmp_path: Path) -> None:
    manager, _ = te.build_mock_manager(tmp_path)
    with pytest.raises(FileNotFoundError):
        te.serve_gguf(tmp_path / "nope.gguf", manager=manager)


def test_smoke_runs_five_prompts_through_injected_chat() -> None:
    manager, gguf = te.build_mock_manager()
    instance = te.serve_gguf(gguf, manager=manager)
    seen: list[str] = []

    def chat(base_url, prompt, **kwargs):
        seen.append(prompt)
        return {"response": f"ok via {kwargs.get('model')}"}

    report = te.run_smoke(instance, chat_fn=chat)
    assert report["prompts"] == 5 == len(seen)
    assert report["passed"] == 5 and report["failed"] == 0
    assert seen == list(te.SMOKE_PROMPTS)
    assert all(row["status"] == "PASS" and row["response"] for row in report["rows"])


def test_smoke_records_failed_prompts_without_raising() -> None:
    manager, gguf = te.build_mock_manager()
    instance = te.serve_gguf(gguf, manager=manager)

    def chat(base_url, prompt, **kwargs):
        if "France" in prompt:
            raise RuntimeError("boom")
        if "sea" in prompt:
            return {"response": "   "}
        return {"response": "ok"}

    report = te.run_smoke(instance, chat_fn=chat)
    assert report["passed"] == 3 and report["failed"] == 2
    assert {row["status"] for row in report["rows"]} == {"PASS", "FAIL", "EMPTY"}


def test_evaluate_mock_pipeline_writes_serializable_report(tmp_path: Path) -> None:
    manager, gguf = te.build_mock_manager(tmp_path)
    paired_keys: list[str] = []

    def paired(instance):
        paired_keys.append(instance["key"])
        return {"mock": True, "turns_compared": 1}

    report = te.evaluate(gguf=gguf, manager=manager, chat_fn=te._mock_chat,
                         paired_fn=paired)
    assert report["ok"] is True
    assert report["task"] == "T11" and report["spec_sha256"] == SPEC_SHA256
    assert report["smoke"]["passed"] == 5
    assert report["paired_ab"] == {"mock": True, "turns_compared": 1}
    assert paired_keys == [gguf.stem]
    assert report["unloaded"] == {"ok": True, "unloaded": gguf.stem}
    json.dumps(report)


def test_evaluate_keeps_model_loaded_when_asked(tmp_path: Path) -> None:
    manager, gguf = te.build_mock_manager(tmp_path)
    report = te.evaluate(gguf=gguf, manager=manager, chat_fn=te._mock_chat,
                         paired_fn=None, keep_loaded=True)
    assert "unloaded" not in report
    assert report["paired_ab"] is None
    assert manager.status()["running"] is True


def test_evaluate_records_paired_failure(tmp_path: Path) -> None:
    manager, gguf = te.build_mock_manager(tmp_path)

    def paired(instance):
        raise RuntimeError("no fixtures")

    report = te.evaluate(gguf=gguf, manager=manager, chat_fn=te._mock_chat,
                         paired_fn=paired)
    assert report["paired_ab"] == {"error": "no fixtures"}
    assert report["ok"] is False


def test_paired_subset_delegates_to_cli_with_one_conversation(tmp_path: Path) -> None:
    output = tmp_path / "paired.json"
    stub = {"metrics": {"turns_compared": 1}, "turns": []}
    seen: list[list[str]] = []

    def run(argv):
        seen.append(list(argv))
        output.write_text(json.dumps(stub), encoding="utf-8")
        return SimpleNamespace(returncode=0, stderr="")

    report = te.paired_ab_subset(
        base_url="http://127.0.0.1:8090", model="mock-ternary",
        conversations="tests/fixtures/generated", max_convs=1, max_turns=3,
        output=output, run=run,
    )
    assert report == stub
    argv = seen[0]
    assert argv[1:3] == ["-m", "experiments.paired_ab"]
    assert "--live" in argv
    assert argv[argv.index("--conversations") + 1] == "tests/fixtures/generated"
    assert argv[argv.index("--max-convs") + 1] == "1"
    assert argv[argv.index("--max-turns") + 1] == "3"
    assert argv[argv.index("--base-url") + 1] == "http://127.0.0.1:8090"
    assert argv[argv.index("--output") + 1] == str(output)


def test_paired_subset_raises_on_failure(tmp_path: Path) -> None:
    def run(argv):
        return SimpleNamespace(returncode=2, stderr="boom: no conversation files")

    with pytest.raises(RuntimeError, match="paired_ab subset failed"):
        te.paired_ab_subset(base_url="http://127.0.0.1:8090",
                            output=tmp_path / "x.json", run=run)


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeHTTP:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    def post(self, url, json=None, timeout=None):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        return _FakeResponse(self.payload)


def test_chat_completion_parses_openai_response() -> None:
    http = _FakeHTTP({
        "model": "mock-ternary",
        "choices": [{"message": {"content": "ok"}}],
        "usage": {"completion_tokens": 2},
    })
    row = te.chat_completion("http://127.0.0.1:8090", "say ok",
                             model="mock-ternary", http=http)
    assert row["response"] == "ok" and row["completion_tokens"] == 2
    call = http.calls[0]
    assert call["url"] == "http://127.0.0.1:8090/v1/chat/completions"
    assert call["json"]["messages"][-1] == {"role": "user", "content": "say ok"}
    assert call["json"]["model"] == "mock-ternary"


def test_no_thinking_args_are_llama_server_flags() -> None:
    assert te.NO_THINKING_ARGS[0] == "--chat-template-kwargs"
    assert json.loads(te.NO_THINKING_ARGS[1]) == {"enable_thinking": False}


def test_cli_mock_mode_writes_report(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    assert te.main(["--mock", "--no-thinking", "--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["task"] == "T11"
    assert report["smoke"]["passed"] == 5
    assert report["paired_ab"]["mock"] is True


def test_cli_live_mode_requires_gguf() -> None:
    assert te.main([]) == 2


def test_serve_gguf_through_real_model_manager(tmp_path: Path) -> None:
    try:
        from harness import models as hm
    except ImportError:
        pytest.skip("harness package needs the F6 SPLINTER_HOME pin")
    assert hm._FILE_TYPE_NAMES.get(141) == "PQ2_0"
    spawned: list[list[str]] = []

    class _FakeProc:
        pid = 4321

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

        def terminate(self):
            return None

        def kill(self):
            return None

    def fake_spawner(cmd, **kwargs):
        spawned.append(list(cmd))
        return _FakeProc()

    def fake_prober(base_url):
        return "mock-ternary"

    gguf = tmp_path / "mock-ternary.gguf"
    gguf.write_bytes(b"GGUF" + b"\x00" * 32)
    manager = te.build_manager(
        binary=tmp_path / "llama-server", models_dir=tmp_path / "models",
        log_dir=tmp_path / "logs", port=_unused_port(),
        spawner=fake_spawner, prober=fake_prober, startup_timeout=5,
    )
    manager.binary.write_bytes(b"")
    instance = te.serve_gguf(gguf, manager=manager, ctx_size=4096, ngl=99,
                             extra_args=list(te.NO_THINKING_ARGS))

    cmd = spawned[0]
    assert cmd[cmd.index("-m") + 1] == str(gguf)
    assert cmd[cmd.index("--port") + 1] == str(manager.port)
    assert cmd[cmd.index("-ngl") + 1] == "99"
    assert cmd[-2:] == list(te.NO_THINKING_ARGS)
    assert instance["key"] == "mock-ternary"
    assert instance["base_url"] == f"http://127.0.0.1:{manager.port}"
    assert manager.unload(instance["key"])["unloaded"] == "mock-ternary"


# ---------------------------------------------------------------------------
# A/B hardening: manifest, warmup, liveness, deadline, timeout
# ---------------------------------------------------------------------------
def test_evaluate_records_manifest_and_warmup(tmp_path: Path) -> None:
    manager, gguf = te.build_mock_manager(tmp_path)
    calls: list[str] = []

    def chat(base_url, prompt, **kwargs):
        calls.append(prompt)
        return {"response": "ok"}

    report = te.evaluate(gguf=gguf, manager=manager, chat_fn=chat, paired_fn=None,
                         warmup=True, manifest={"mode": "live", "x": 1})
    assert report["manifest"] == {"mode": "live", "x": 1}
    assert len(calls) == 6  # 1 warmup + 5 smoke
    assert calls[0] == "Reply with the single word: ok"
    json.dumps(report)


def test_evaluate_aborts_when_server_unreachable(tmp_path: Path) -> None:
    manager, gguf = te.build_mock_manager(tmp_path)
    report = te.evaluate(gguf=gguf, manager=manager, chat_fn=te._mock_chat,
                         paired_fn=None, health_fn=lambda url: False)
    assert report["ok"] is False
    assert "not reachable" in report["smoke"]["error"]


def test_run_smoke_deadline_marks_failures() -> None:
    manager, gguf = te.build_mock_manager()
    instance = te.serve_gguf(gguf, manager=manager)
    report = te.run_smoke(instance, chat_fn=te._mock_chat,
                          deadline=te.time.monotonic() - 1)
    assert report["passed"] == 0 and report["failed"] == 5
    assert all(row["error"] == "deadline exceeded" for row in report["rows"])


def test_run_smoke_passes_timeout() -> None:
    manager, gguf = te.build_mock_manager()
    instance = te.serve_gguf(gguf, manager=manager)
    seen: dict = {}

    def chat(base_url, prompt, **kwargs):
        seen.update(kwargs)
        return {"response": "ok"}

    te.run_smoke(instance, chat_fn=chat, timeout=42)
    assert seen.get("timeout") == 42
