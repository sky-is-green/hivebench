"""Unit tests for ``harness.stack.strata`` — the managed Strata launcher.

No network and no real Strata: a temp "repo" with the server entry point and
its venv python, a recording fake spawner, and a scripted prober.  The
launcher's contract under test is the generated config, the command line, the
health wait, and the owned-process bookkeeping that ``StackManager`` relies on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.stack.schema import Tier
from harness.stack.strata import StrataLauncher, merge_engine_args

# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------

class _Proc:
    """Minimal Popen surface; ``dies`` controls terminate/wait behaviour."""

    def __init__(self, pid: int = 4242, *, alive: bool = True, dies: bool = True) -> None:
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
        "health_url": f"{endpoint.rstrip('/')}/health", "model": "strata",
        "loaded": True, "max_context": 4096, "service": "strata",
    }


@pytest.fixture
def strata_tree(tmp_path) -> Path:
    """A temp Strata checkout: serve/server.py, venv python, base config."""
    repo = tmp_path / "strata"
    (repo / "serve").mkdir(parents=True)
    (repo / ".venv" / "bin").mkdir(parents=True)
    (repo / "serve" / "server.py").write_text("", encoding="utf-8")
    (repo / ".venv" / "bin" / "python").write_text("", encoding="utf-8")
    (repo / "strata-q2_0-coexist.json").write_text(json.dumps({
        "exe": str(repo / "engine" / "strata"),
        "args": ["--pack", "packs/q2_0", "--max-context", "4096",
                 "--vram-reserve-mib", "1024", "--prefill", "512"],
        "cwd": str(repo),
        "port": 8080,
        "log": str(repo / "strata.log"),
        "backend": "hip",
        "gpu": [0, 1],
    }), encoding="utf-8")
    return repo


def _tier(repo: Path, *, ctx: int = 4096, **launch_overrides) -> Tier:
    launch = {
        "kind": "strata",
        "repo": str(repo),
        "config": "strata-q2_0-coexist.json",
        "engine_args": ["--vram-reserve-mib", "3072"],
    }
    launch.update(launch_overrides)
    return Tier(
        role="face", repo="Qwen/Qwen3.8-Flash-Next", file="m.gguf",
        ctx=ctx, backend="strata",
        endpoint="http://127.0.0.1:18080/v1", launch=launch,
    )


def _launcher(tmp_path, spawner, prober, *, timeout_s: float = 2.0) -> StrataLauncher:
    return StrataLauncher(
        state_dir=tmp_path / "state",
        log_dir=tmp_path / "logs",
        spawner=spawner,
        prober=prober,
        timeout_s=timeout_s,
    )


# ---------------------------------------------------------------------------
# argv merge / config generation
# ---------------------------------------------------------------------------

def test_merge_engine_args_replaces_in_place_and_appends():
    base = ["--a", "1", "--b", "2", "3", "--c"]
    assert merge_engine_args(base, ["--b", "9", "--d", "4"]) == [
        "--a", "1", "--b", "9", "--c", "--d", "4"]


def test_merge_engine_args_keeps_negative_numbers_as_values():
    assert merge_engine_args(["--spec-min-p", "0.5"], ["--spec-min-p", "-0.5"]) == [
        "--spec-min-p", "-0.5"]


def test_build_config_syncs_port_ctx_args_log_and_options(tmp_path):
    launcher = StrataLauncher()
    base = {
        "args": ["--prefill", "512", "--vram-reserve-mib", "1024", "--pack", "x"],
        "port": 1,
        "log": "/old/log",
    }
    tier = _tier(tmp_path, ctx=8192)
    config = launcher.build_config(
        base, tier.launch, tier, "http://127.0.0.1:18080/v1",
        engine_log=tmp_path / "engine.log",
    )

    args = config["args"]
    assert config["port"] == 18080
    assert config["log"] == str(tmp_path / "engine.log")
    # an existing flag is replaced, never duplicated
    assert args.count("--vram-reserve-mib") == 1
    assert args[args.index("--vram-reserve-mib") + 1] == "3072"
    # ctx is appended when the base has no --max-context yet
    assert args[args.index("--max-context") + 1] == "8192"
    assert args[args.index("--prefill") + 1] == "512"


def test_build_config_merges_options_on_top(tmp_path):
    launcher = StrataLauncher()
    tier = _tier(tmp_path, options={"idle_unload_s": 600})
    config = launcher.build_config(
        {"args": [], "port": 1}, tier.launch, tier,
        "http://127.0.0.1:18080/v1", tmp_path / "e.log")
    assert config["idle_unload_s"] == 600


def test_build_config_uses_the_lanes_own_context_flag(tmp_path):
    """The DeepSeek-V4.1 engine takes ``--ctx`` and refuses unknown flags.

    The launcher forces the context onto the engine's argv, so a lane whose
    engine spells it differently names it (``launch.ctx_flag``) and the
    default spelling is left out entirely - passing both would abort the
    engine on the flag it does not know.
    """
    launcher = StrataLauncher()
    tier = _tier(tmp_path, ctx=32768, ctx_flag="--ctx")
    config = launcher.build_config(
        {"args": ["--ctx", "262144", "--experts", "gpu"], "port": 1},
        tier.launch, tier, "http://127.0.0.1:18080/v1", tmp_path / "e.log")

    args = config["args"]
    assert args.count("--ctx") == 1
    assert args[args.index("--ctx") + 1] == "32768"
    assert "--max-context" not in args
    assert args[args.index("--experts") + 1] == "gpu"


def test_build_config_defaults_to_max_context(tmp_path):
    launcher = StrataLauncher()
    tier = _tier(tmp_path, ctx=8192)
    config = launcher.build_config(
        {"args": [], "port": 1}, tier.launch, tier,
        "http://127.0.0.1:18080/v1", tmp_path / "e.log")
    args = config["args"]
    assert args[args.index("--max-context") + 1] == "8192"
    assert "--ctx" not in args


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------

def test_resolve_expands_paths_and_port(strata_tree):
    resolved = StrataLauncher().resolve(
        _tier(strata_tree).launch, "http://127.0.0.1:18080/v1")
    assert resolved["repo"] == strata_tree.resolve()
    assert resolved["config"] == (strata_tree / "strata-q2_0-coexist.json").resolve()
    assert resolved["python"] == (strata_tree / ".venv/bin/python").resolve()
    assert resolved["port"] == 18080


def test_resolve_keeps_the_venv_python_symlink(strata_tree, tmp_path):
    """A venv python is a symlink to the system interpreter; do not follow it."""
    target = tmp_path / "system-python"
    target.write_text("", encoding="utf-8")
    python = strata_tree / ".venv/bin/python"
    python.unlink()
    python.symlink_to(target)

    resolved = StrataLauncher().resolve(
        _tier(strata_tree).launch, "http://127.0.0.1:18080/v1")

    assert resolved["python"] == python
    assert resolved["python"] != target.resolve()


def test_launcher_directories_are_absolute(tmp_path, monkeypatch):
    """A relative log_dir must not leak into the generated config: the Strata
    server resolves the engine log against its own cwd."""
    monkeypatch.chdir(tmp_path)
    launcher = StrataLauncher(log_dir="logs", state_dir="state")
    assert launcher.log_dir == tmp_path / "logs"
    assert launcher.state_dir == tmp_path / "state"


def test_resolve_refuses_paths_outside_the_repo(strata_tree, tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="inside the repo"):
        StrataLauncher().resolve(
            {"kind": "strata", "repo": str(strata_tree), "config": str(outside)},
            "http://127.0.0.1:18080/v1")


def test_resolve_refuses_missing_repo_config_and_python(tmp_path):
    with pytest.raises(RuntimeError, match="repo not found"):
        StrataLauncher().resolve(
            {"kind": "strata", "repo": str(tmp_path / "nope"), "config": "c.json"},
            "http://127.0.0.1:18080/v1")

    repo = tmp_path / "repo"
    repo.mkdir()
    with pytest.raises(RuntimeError, match="config not found"):
        StrataLauncher().resolve(
            {"kind": "strata", "repo": str(repo), "config": "c.json"},
            "http://127.0.0.1:18080/v1")

    (repo / "c.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="python not found"):
        StrataLauncher().resolve(
            {"kind": "strata", "repo": str(repo), "config": "c.json"},
            "http://127.0.0.1:18080/v1")


def test_resolve_refuses_an_endpoint_without_a_port(strata_tree):
    with pytest.raises(RuntimeError, match="no usable port"):
        StrataLauncher().resolve(
            _tier(strata_tree).launch, "http://127.0.0.1:not-a-port/v1")


# ---------------------------------------------------------------------------
# start / stop
# ---------------------------------------------------------------------------

def test_start_writes_the_generated_config_and_waits_for_health(strata_tree, tmp_path):
    spawner = _Spawner()
    polls = {"n": 0}

    def prober(endpoint: str) -> dict:
        polls["n"] += 1
        if polls["n"] < 2:
            raise RuntimeError("still starting")
        return _health(endpoint)

    launcher = _launcher(tmp_path, spawner, prober)
    tier = _tier(strata_tree, ctx=8192)
    info = launcher.start(stack_name="strata-125b", role="face",
                          tier=tier, endpoint=tier.endpoint)

    assert info["pid"] == 4242
    assert launcher.managed("face") and launcher.alive("face")
    assert launcher.info("face")["health"]["model"] == "strata"
    assert polls["n"] == 2

    cmd = spawner.cmds[0]
    assert cmd[0] == str((strata_tree / ".venv/bin/python").resolve())
    assert cmd[1:] == ["serve/server.py", "--engine", "strata",
                       "--config", info["config"], "--port", "18080"]
    assert spawner.kwargs[0]["cwd"] == str(strata_tree.resolve())

    generated = json.loads(Path(info["config"]).read_text(encoding="utf-8"))
    assert generated["port"] == 18080
    assert generated["log"] == info["engine_log"]
    args = generated["args"]
    assert args[args.index("--max-context") + 1] == "8192"
    assert args[args.index("--vram-reserve-mib") + 1] == "3072"
    assert Path(info["log"]).parent == tmp_path / "logs"


def test_start_raises_when_the_process_exits_during_startup(strata_tree, tmp_path):
    dead = _Proc(alive=False)
    launcher = _launcher(tmp_path, _Spawner(dead),
                         lambda endpoint: _health(endpoint))
    tier = _tier(strata_tree)
    with pytest.raises(RuntimeError, match="exited during startup"):
        launcher.start(stack_name="s", role="face", tier=tier,
                       endpoint=tier.endpoint)
    assert not launcher.managed("face")


def test_start_times_out_and_stops_the_process(strata_tree, tmp_path):
    def prober(endpoint: str) -> dict:
        raise RuntimeError("no answer")

    launcher = _launcher(tmp_path, _Spawner(), prober, timeout_s=0.2)
    tier = _tier(strata_tree)
    with pytest.raises(RuntimeError, match="did not become healthy"):
        launcher.start(stack_name="s", role="face", tier=tier,
                       endpoint=tier.endpoint)
    assert not launcher.managed("face")


def test_start_uses_the_blocks_timeout(strata_tree, tmp_path):
    calls = {"n": 0}

    def prober(endpoint: str) -> dict:
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("still starting")
        return _health(endpoint)

    # The launcher default is too short to survive the 1 s poll backoff; the
    # block's timeout_s is what the start path uses.
    launcher = _launcher(tmp_path, _Spawner(), prober, timeout_s=0.2)
    tier = _tier(strata_tree, timeout_s=30)
    info = launcher.start(stack_name="s", role="face", tier=tier,
                          endpoint=tier.endpoint)
    assert info["pid"] == 4242 and calls["n"] == 2


def test_stop_terminates_the_owned_process_and_is_a_noop_otherwise(tmp_path):
    proc = _Proc()
    launcher = _launcher(tmp_path, _Spawner(proc), lambda e: _health(e))
    launcher._procs["face"] = proc
    launcher._meta["face"] = {"pid": 1}

    assert launcher.stop("missing") is False
    assert launcher.stop("face") is True
    assert proc.terminated and not launcher.managed("face")


def test_stop_kills_a_process_that_ignores_terminate(tmp_path):
    proc = _Proc(dies=False)
    launcher = _launcher(tmp_path, _Spawner(proc), lambda e: _health(e))
    launcher._procs["face"] = proc

    assert launcher.stop("face") is True
    assert proc.terminated and proc.killed
    assert launcher.stop_all() == []
