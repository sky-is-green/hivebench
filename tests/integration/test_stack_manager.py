"""Integration: StackManager apply / status / unload (HIVE-PLAN §B4 T37).

No GGUFs exist on this box and no llama-server binary is shipped, so the
manager is exercised against the *real* ``harness.models.LlamaServerManager``
with an injected fake spawner (a ``subprocess.Popen`` stand-in) and a fake
prober: ``load()`` runs end to end, registers a ``ServerInstance`` per tier,
and the manager is the only thing under test.  The real apply gate is T45.

Residency planning is T36's module (still a skeleton in this worktree), so
``plan_residency`` is monkeypatched here — the manager's contract with it is
the fake plan's ``ok`` / ``per_card`` / ``warnings`` shape.
"""

from __future__ import annotations

import pytest

import harness.models as models_mod
import harness.stack.manager as manager_mod
from harness.models import LlamaServerManager
from harness.stack.manager import (
    StackManager,
    TierRuntime,
    tier_env,
    tier_extra_args,
    tier_load_options,
)
from harness.stack.schema import Stack, Tier


# --------------------------------------------------------------------------
# fakes: the injected spawner + a stub residency plan
# --------------------------------------------------------------------------

class _FakeProc:
    """The minimum ``subprocess.Popen`` surface ``LlamaServerManager`` uses."""

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self._alive = True
        self.terminated = False
        self.killed = False

    def poll(self):
        return None if self._alive else 0

    def wait(self, timeout=None):  # noqa: ANN001 - Popen-compatible
        self._alive = False
        return 0

    def terminate(self) -> None:
        self.terminated = True
        self._alive = False

    def kill(self) -> None:
        self.killed = True
        self._alive = False


class _Spawner:
    """Records every spawn (argv + kwargs) and returns live fake processes."""

    def __init__(self) -> None:
        self.cmds: list[list[str]] = []
        self.kwargs: list[dict] = []
        self.procs: list[_FakeProc] = []

    def __call__(self, cmd, **kwargs):  # noqa: ANN001 - Popen-compatible
        self.cmds.append([str(a) for a in cmd])
        self.kwargs.append(dict(kwargs))
        proc = _FakeProc(1000 + len(self.procs))
        self.procs.append(proc)
        return proc


class _Plan:
    """Stand-in for ``residency.ResidencyPlan`` (T36)."""

    def __init__(self, *, ok: bool = True, per_card=None, warnings=None) -> None:
        self.ok = ok
        self.per_card = list(per_card or [])
        self.warnings = list(warnings or [])


_CARD = {"card": 0, "weights": 8.0, "kv": 2.0, "total": 10.0, "budget": 15.0}


def _make_manager(tmp_path, monkeypatch, port: int = 18137):
    binary = tmp_path / "llama-server"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    spawner = _Spawner()
    manager = LlamaServerManager(
        binary=binary,
        models_dir=tmp_path / "models",
        port=port,
        log_dir=tmp_path / "logs",
        spawner=spawner,
        prober=lambda base_url: "fake-model",
        startup_timeout=5.0,
        memory_guard=False,
    )
    # Ports are allocated by the StackManager; nothing actually binds, so make
    # the liveness probe deterministic instead of depending on the host.
    monkeypatch.setattr(models_mod, "_port_in_use", lambda *a, **k: False)
    return manager, spawner


def _install_plan(monkeypatch, plan: _Plan) -> None:
    monkeypatch.setattr(manager_mod, "plan_residency", lambda stack, **kw: plan)


def _tier(role: str, **overrides) -> Tier:
    data = {
        "role": role,
        "repo": f"mock/{role}",
        "file": f"{role}.gguf",
        "ctx": 8192,
        "ngl": 99,
        "backend": "vulkan",
        "cache_k": "q8_0",
        "cache_v": "q8_0",
    }
    data.update(overrides)
    return Tier(**data)


def _two_tier_stack() -> Stack:
    """The §B3 ``peer-2tier`` shape: a face with MTP + split + pin, a worker."""
    return Stack(
        name="peer-2tier",
        tiers=[
            _tier("face", ctx=262144, mmproj="mmproj-F16.gguf",
                  pin="HIP_VISIBLE_DEVICES=0,1", ts="1,1",
                  spec={"type": "draft-mtp", "n_max": 3}),
            _tier("worker", ctx=131072, pin="HIP_VISIBLE_DEVICES=1"),
        ],
    )


# --------------------------------------------------------------------------
# apply
# --------------------------------------------------------------------------

def test_apply_registers_two_instances_on_distinct_ports(tmp_path, monkeypatch):
    manager, spawner = _make_manager(tmp_path, monkeypatch)
    _install_plan(monkeypatch, _Plan(per_card=[_CARD]))
    stack = _two_tier_stack()

    manager_api = StackManager(manager, stacks_root=tmp_path / "stacks")
    result = manager_api.apply(stack)

    assert result["ok"] is True
    assert [t["role"] for t in result["tiers"]] == ["face", "worker"]
    ports = [t["port"] for t in result["tiers"]]
    assert ports[0] != ports[1]
    assert len({t["key"] for t in result["tiers"]}) == 2
    # two live ServerInstances registered by load()
    assert len(manager._instances) == 2
    assert sorted(i.port for i in manager._instances.values()) == sorted(ports)


def test_status_reports_the_plans_per_tier_figures(tmp_path, monkeypatch):
    manager, spawner = _make_manager(tmp_path, monkeypatch)
    _install_plan(monkeypatch, _Plan(per_card=[_CARD],
                                     warnings=["display budget estimated"]))
    manager_api = StackManager(manager)
    manager_api.apply(_two_tier_stack())

    status = manager_api.status()
    assert status["ok"] is True
    assert status["stack"] == "peer-2tier"
    assert status["warnings"] == ["display budget estimated"]
    rows = {row["role"]: row for row in status["tiers"]}
    assert set(rows) == {"face", "worker"}
    face, worker = rows["face"], rows["worker"]
    assert (face["ctx"], worker["ctx"]) == (262144, 131072)
    assert face["model"] == "fake-model" == worker["model"]
    assert face["backend"] == "vulkan" and face["resident"] is True
    assert face["per_card"] == [_CARD]
    assert face["port"] != worker["port"]
    assert face["vram_gb"] is None and face["tok_s"] is None
    assert face["key"].endswith("peer-2tier-face")


def test_apply_passes_the_tier_launch_config_to_load(tmp_path, monkeypatch):
    manager, spawner = _make_manager(tmp_path, monkeypatch)
    _install_plan(monkeypatch, _Plan(per_card=[_CARD]))
    StackManager(manager).apply(_two_tier_stack())

    face_cmd, worker_cmd = spawner.cmds
    assert face_cmd[face_cmd.index("-c") + 1] == "262144"
    assert face_cmd[face_cmd.index("-ngl") + 1] == "99"
    assert "--cache-type-k" in face_cmd and "q8_0" in face_cmd
    assert "--mmproj" in face_cmd and "mmproj-F16.gguf" in face_cmd
    assert ["--split-mode", "layer", "--ts", "1,1"] == \
        face_cmd[face_cmd.index("--split-mode"):face_cmd.index("--ts") + 2]
    assert ["--spec-type", "draft-mtp", "--draft-max", "3"] == \
        face_cmd[face_cmd.index("--spec-type"):face_cmd.index("--draft-max") + 2]
    # per-tier env: face pinned to both cards, worker to one
    assert spawner.kwargs[0]["env"]["HIP_VISIBLE_DEVICES"] == "0,1"
    assert spawner.kwargs[1]["env"]["HIP_VISIBLE_DEVICES"] == "1"
    # the worker keeps its own ctx and carries none of the face's MTP flags
    assert worker_cmd[worker_cmd.index("-c") + 1] == "131072"
    assert "--split-mode" not in worker_cmd and "--spec-type" not in worker_cmd


def test_apply_is_face_first_even_when_the_stack_leads_with_a_worker(tmp_path, monkeypatch):
    manager, spawner = _make_manager(tmp_path, monkeypatch)
    _install_plan(monkeypatch, _Plan())
    stack = Stack(name="rev", tiers=[_tier("worker", ctx=4096),
                                    _tier("face", ctx=8192)])
    result = StackManager(manager).apply(stack)
    assert [t["role"] for t in result["tiers"]] == ["face", "worker"]
    assert spawner.cmds[0][spawner.cmds[0].index("-c") + 1] == "8192"
    assert spawner.cmds[1][spawner.cmds[1].index("-c") + 1] == "4096"


def test_apply_refuses_over_budget_before_spawning(tmp_path, monkeypatch):
    manager, spawner = _make_manager(tmp_path, monkeypatch)
    _install_plan(monkeypatch, _Plan(ok=False, warnings=["face: 20.00 > 16.00"]))
    with pytest.raises(RuntimeError, match="refused"):
        StackManager(manager).apply(_two_tier_stack())
    assert spawner.cmds == []
    assert manager._instances == {}


def test_apply_rolls_back_when_a_later_tier_fails(tmp_path, monkeypatch):
    manager, spawner = _make_manager(tmp_path, monkeypatch)
    _install_plan(monkeypatch, _Plan())
    original = manager.load

    def flaky(*args, **kwargs):
        if str(kwargs.get("key", "")).endswith("-worker"):
            raise RuntimeError("no VRAM for the worker")
        return original(*args, **kwargs)

    monkeypatch.setattr(manager, "load", flaky)
    manager_api = StackManager(manager)
    with pytest.raises(RuntimeError, match="tier 'worker'"):
        manager_api.apply(_two_tier_stack())

    assert manager._instances == {}
    assert spawner.procs[0].terminated is True
    assert manager_api.status()["tiers"] == []


# --------------------------------------------------------------------------
# unload
# --------------------------------------------------------------------------

def test_unload_stops_both_tiers(tmp_path, monkeypatch):
    manager, spawner = _make_manager(tmp_path, monkeypatch)
    _install_plan(monkeypatch, _Plan())
    manager_api = StackManager(manager)
    manager_api.apply(_two_tier_stack())

    # a name that is not the applied stack is a no-op
    assert manager_api.unload("someone-else") == {"ok": True, "unloaded": []}
    assert len(manager._instances) == 2

    result = manager_api.unload()
    assert result == {"ok": True, "unloaded": ["face", "worker"]}
    assert manager._instances == {}
    assert all(proc.terminated for proc in spawner.procs)
    assert manager_api.status() == {"ok": True, "stack": None,
                                    "tiers": [], "warnings": []}
    # a second unload is a clean no-op
    assert manager_api.unload() == {"ok": True, "unloaded": []}


def test_status_before_apply_is_empty(tmp_path, monkeypatch):
    manager, _ = _make_manager(tmp_path, monkeypatch)
    status = StackManager(manager).status()
    assert status["stack"] is None
    assert status["tiers"] == []


def test_instance_for_returns_the_server_instance(tmp_path, monkeypatch):
    manager, _ = _make_manager(tmp_path, monkeypatch)
    _install_plan(monkeypatch, _Plan())
    manager_api = StackManager(manager)
    manager_api.apply(_two_tier_stack())

    instance = manager_api.instance_for("face")
    assert instance is not None
    assert instance.key.endswith("peer-2tier-face")
    assert instance.port == manager_api.status()["tiers"][0]["port"]
    assert manager_api.instance_for("missing") is None


# --------------------------------------------------------------------------
# tier config helpers (the EngineProfile.load_options seam, ADR-L4)
# --------------------------------------------------------------------------

def test_tier_load_options_maps_the_b3_fields():
    tier = _tier("face", ctx=262144, ngl=99, mmproj="mmproj-F16.gguf")
    assert tier_load_options(tier) == {
        "context": 262144,
        "gpu_layers": 99,
        "cache_type_k": "q8_0",
        "cache_type_v": "q8_0",
        "mmproj": "mmproj-F16.gguf",
    }


def test_tier_load_options_accepts_the_mapping_form():
    # T43's stack_summary passes face_tier()'s plain dict, not the dataclass.
    face = {"role": "face", "ctx": 8192, "ngl": 99,
            "cache_k": "q8_0", "cache_v": "q8_0"}
    assert tier_load_options(face)["context"] == 8192
    assert tier_load_options(face)["gpu_layers"] == 99


def test_tier_env_parses_the_pin_assignment():
    assert tier_env(_tier("face", pin="HIP_VISIBLE_DEVICES=0,1")) == {
        "HIP_VISIBLE_DEVICES": "0,1"}
    assert tier_env(_tier("face", pin="1")) == {"HIP_VISIBLE_DEVICES": "1"}
    assert tier_env(_tier("face")) == {}


def test_tier_extra_args_emits_split_and_spec_flags():
    tier = _tier("face", ts="1,1", spec={"type": "draft-mtp", "n_max": 3})
    assert tier_extra_args(tier) == [
        "--split-mode", "layer", "--ts", "1,1",
        "--spec-type", "draft-mtp", "--draft-max", "3",
    ]
    assert tier_extra_args(_tier("worker")) == []


def test_tier_runtime_to_dict_is_the_status_row():
    runtime = TierRuntime(role="face", key="k", port=1234, ctx=8192,
                          model="m", per_card=[_CARD])
    assert runtime.to_dict() == {
        "role": "face", "key": "k", "port": 1234, "ctx": 8192, "model": "m",
        "backend": "vulkan", "resident": True, "vram_gb": None, "tok_s": None,
        "per_card": [_CARD],
    }
