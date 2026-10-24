"""Unit tests for ``harness.stack.deepinfer`` — the managed DeepInfer launcher.

No network, no GPU and no DeepInfer: a temp "repo" with a config and a
venv python, a recording fake spawner, and a scripted prober.  The
launcher's contract under test is the generated config, the command line,
the health wait, and the owned-process bookkeeping that ``StackManager``
relies on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.stack.deepinfer import DeepInferLauncher
from harness.stack.schema import Tier


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------
class _Proc:
    """Minimal Popen surface; ``dies`` controls terminate/wait behaviour."""

    def __init__(self, pid: int = 7777, *, alive: bool = True, dies: bool = True) -> None:
        self.pid = pid
        self._alive = alive
        self._dies = dies
        self.terminated = False
        self.killed = False

    def poll(self):
        return None if self._alive else 3

    def terminate(self) -> None:
        self.terminated = True
        if self._dies:
            self._alive = False

    def wait(self, timeout=None):  # noqa: ANN001 - Popen-compatible
        if not self._alive:
            return 0
        if self._dies:
            self._alive = False
            return 0
        raise TimeoutError("still running")

    def kill(self) -> None:
        self.killed = True
        self._alive = False


class _Spawner:
    def __init__(self, proc: _Proc | None = None) -> None:
        self.cmds: list[list[str]] = []
        self.kwargs: list[dict] = []
        self.proc = proc or _Proc()

    def __call__(self, cmd, **kwargs):  # noqa: ANN001 - Popen-compatible
        self.cmds.append([str(a) for a in cmd])
        self.kwargs.append(dict(kwargs))
        return self.proc


def _health(endpoint: str) -> dict:
    return {
        "endpoint": endpoint, "base_url": endpoint.rstrip("/"),
        "health_url": f"{endpoint.rstrip('/')}/health", "model": "deepseek-v4.1-flash",
        "loaded": True, "max_context": 4096, "service": "deepinfer",
    }


@pytest.fixture
def repo(tmp_path) -> Path:
    """A temp repo with a facade config and a venv python."""
    root = tmp_path / "hivebench"
    (root / ".venv-amd" / "bin").mkdir(parents=True)
    (root / ".venv-amd" / "bin" / "python").write_text("", encoding="utf-8")
    (root / "configs").mkdir()
    (root / "configs" / "deepinfer-v41.json").write_text(json.dumps({
        "model": "/models/DeepSeek-V4.1-Flash-1017",
        "served_model_name": "deepseek-v4.1-flash",
        "ctx": 4096,
        "expert_parallel_size": 1,
    }), encoding="utf-8")
    return root


def _launch_block(repo: Path, **extra) -> dict:
    block = {
        "kind": "deepinfer",
        "repo": str(repo),
        "config": "configs/deepinfer-v41.json",
        "python": ".venv-amd/bin/python",
    }
    block.update(extra)
    return block


def _tier(repo: Path | None = None, **extra) -> Tier:
    data = {
        "role": "face", "repo": "local/deepinfer", "file": "DeepSeek-V4.1-Flash-1017",
        "ctx": 4096,
    }
    if repo is not None:
        data["launch"] = _launch_block(repo)
    data.update(extra)
    return Tier.from_dict(data)


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------
def test_resolve_reads_repo_config_python_and_port(repo) -> None:
    launcher = DeepInferLauncher(state_dir=repo / "state", log_dir=repo / "logs")
    resolved = launcher.resolve(_launch_block(repo), "http://127.0.0.1:8090/v1")
    assert resolved["repo"] == repo.resolve()
    assert resolved["config"] == (repo / "configs" / "deepinfer-v41.json").resolve()
    assert resolved["python"] == (repo / ".venv-amd" / "bin" / "python").absolute()
    assert resolved["port"] == 8090


def test_resolve_rejects_a_missing_repo(tmp_path) -> None:
    launcher = DeepInferLauncher(state_dir=tmp_path, log_dir=tmp_path)
    with pytest.raises(RuntimeError, match="deepinfer repo not found"):
        launcher.resolve(_launch_block(tmp_path / "nope"), "http://127.0.0.1:8090/v1")


def test_resolve_rejects_a_config_outside_the_repo(repo) -> None:
    launcher = DeepInferLauncher(state_dir=repo / "state", log_dir=repo / "logs")
    with pytest.raises(RuntimeError, match="must stay inside the repo"):
        launcher.resolve(_launch_block(repo, config="/etc/passwd"),
                         "http://127.0.0.1:8090/v1")


def test_resolve_rejects_a_missing_python(repo) -> None:
    launcher = DeepInferLauncher(state_dir=repo / "state", log_dir=repo / "logs")
    with pytest.raises(RuntimeError, match="deepinfer python not found"):
        launcher.resolve(_launch_block(repo, python=".venv-nope/bin/python"),
                         "http://127.0.0.1:8090/v1")


# ---------------------------------------------------------------------------
# the generated config
# ---------------------------------------------------------------------------
def test_build_config_syncs_the_port_and_the_context(repo) -> None:
    launcher = DeepInferLauncher(state_dir=repo / "state", log_dir=repo / "logs")
    base = {"model": "/models/x", "ctx": 4096, "port": 1}
    config = launcher.build_config(base, _launch_block(repo), _tier(repo, ctx=2048),
                                   "http://127.0.0.1:8099/v1")
    assert config["port"] == 8099
    assert config["ctx"] == 2048
    assert config["model"] == "/models/x"          # the base survives
    assert base["port"] == 1                        # the base is not mutated


def test_build_config_merges_the_options_and_the_engine_args(repo) -> None:
    launcher = DeepInferLauncher(state_dir=repo / "state", log_dir=repo / "logs")
    block = _launch_block(repo, engine_args=["--max-num-seqs", "8"],
                          options={"seed": 7})
    config = launcher.build_config({"model": "/models/x"}, block, _tier(repo),
                                   "http://127.0.0.1:8090/v1")
    assert config["engine_args"] == ["--max-num-seqs", "8"]
    assert config["seed"] == 7


# ---------------------------------------------------------------------------
# the lifecycle
# ---------------------------------------------------------------------------
def test_start_spawns_the_facade_and_waits_for_health(repo) -> None:
    spawner = _Spawner()
    launcher = DeepInferLauncher(
        state_dir=repo / "state", log_dir=repo / "logs", spawner=spawner,
        prober=lambda endpoint: _health(endpoint), timeout_s=5,
    )
    tier = _tier(repo)
    info = launcher.start(stack_name="deepinfer-v41", role="face", tier=tier,
                          endpoint="http://127.0.0.1:8090/v1")

    assert launcher.managed("face") is True
    assert launcher.alive("face") is True
    assert info["pid"] == 7777
    assert info["health"]["model"] == "deepseek-v4.1-flash"

    cmd = spawner.cmds[0]
    assert cmd[0].endswith("python")
    assert str(cmd[1]).replace("\\", "/").endswith("tools/deepinfer/serve.py")
    assert "--config" in cmd and "--port" in cmd
    assert cmd[cmd.index("--port") + 1] == "8090"

    # The generated config landed in the state directory.
    generated = Path(info["config"])
    assert generated.is_file()
    payload = json.loads(generated.read_text(encoding="utf-8"))
    assert payload["port"] == 8090

    # The DeepInfer checkout and the AMD overlay are importable.
    python_path = spawner.kwargs[0]["env"]["PYTHONPATH"]
    assert "vendor" in python_path
    assert (str(Path("vendor") / "deepinfer").replace("\\", "/")
            in python_path.replace("\\", "/"))


def test_stop_terminates_the_owned_process(repo) -> None:
    spawner = _Spawner()
    launcher = DeepInferLauncher(
        state_dir=repo / "state", log_dir=repo / "logs", spawner=spawner,
        prober=lambda endpoint: _health(endpoint), timeout_s=5,
    )
    tier = _tier(repo)
    launcher.start(stack_name="s", role="face", tier=tier,
                  endpoint="http://127.0.0.1:8090/v1")
    assert launcher.stop("face") is True
    assert spawner.proc.terminated is True
    assert launcher.managed("face") is False
    # A second stop is a no-op: the launcher only owns what it started.
    assert launcher.stop("face") is False


def test_start_raises_when_the_process_dies_and_keeps_nothing_registered(repo) -> None:
    spawner = _Spawner(_Proc(alive=False))
    launcher = DeepInferLauncher(
        state_dir=repo / "state", log_dir=repo / "logs", spawner=spawner,
        prober=lambda endpoint: _health(endpoint), timeout_s=5,
    )
    tier = _tier(repo)
    with pytest.raises(RuntimeError, match="exited during startup"):
        launcher.start(stack_name="s", role="face", tier=tier,
                      endpoint="http://127.0.0.1:8090/v1")
    assert launcher.managed("face") is False


def test_start_raises_when_the_endpoint_never_answers(repo) -> None:
    def _refuse(_endpoint: str) -> dict:
        raise RuntimeError("connection refused")

    launcher = DeepInferLauncher(
        state_dir=repo / "state", log_dir=repo / "logs", spawner=_Spawner(),
        prober=_refuse, timeout_s=0.2,
    )
    tier = _tier(repo)
    with pytest.raises(RuntimeError, match="did not become healthy"):
        launcher.start(stack_name="s", role="face", tier=tier,
                      endpoint="http://127.0.0.1:8090/v1")
    assert launcher.managed("face") is False


def test_stop_all_stops_every_role(repo) -> None:
    launcher = DeepInferLauncher(
        state_dir=repo / "state", log_dir=repo / "logs", spawner=_Spawner(),
        prober=lambda endpoint: _health(endpoint), timeout_s=5,
    )
    tier = _tier(repo)
    launcher.start(stack_name="s", role="face", tier=tier,
                  endpoint="http://127.0.0.1:8090/v1")
    assert launcher.stop_all() == ["face"]
    assert launcher.managed("face") is False
