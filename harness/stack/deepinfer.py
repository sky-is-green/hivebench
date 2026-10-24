"""Managed DeepInfer lifecycle for endpoint-backed tiers (adopt-or-start).

DeepInfer is DeepSeek's own inference engine (Apache-2.0).  This release
ships for two accelerators -- NVIDIA SM100 (CUDA 13) and Huawei Ascend 950
(CANN 9.2) -- and **no HTTP server at all**: it exposes an offline
``LLM.generate()`` API plus a distributed launcher.

So there are two pieces to running it from the harness:

* **this module** -- the managed lifecycle, mirroring
  :class:`harness.stack.strata.StrataLauncher`: adopt a running endpoint,
  or start one and own the process;
* **the facade** (``tools/deepinfer/serve.py``) -- the OpenAI-compatible
  HTTP surface the harness (and the dsh routes) speak.  It wraps DeepInfer's
  offline API and is what makes DeepInfer reachable as a tier at all.

Generated config: the base config (the tier's ``launch.config``) is copied,
the listen ``port`` is synced to the endpoint URL, the engine ``args`` get
the block's ``engine_args``, and ``options`` are merged on top.  The command
is the facade, run with the DeepInfer checkout on ``PYTHONPATH``.

The AMD lane
------------
The vendored tree carries the AMD overlay (``vendor/deepinfer_amd``), which
registers an AMD platform and torch-native replacements for the CUDA-only
packages.  :meth:`DeepInferLauncher.capabilities` reports, honestly, what
this host can actually run: the accelerator, the gfx architecture, whether
the hardware has FP8, whether the C++ extension is built, and which models
the lane can serve.  A host that cannot run the requested model gets a
refusal with a reason, not a stack trace.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from harness.stack.endpoint import endpoint_port, probe_endpoint
from harness.stack.schema import Tier

#: Repo root (``harness/stack/deepinfer.py`` → repo).
REPO_ROOT = Path(__file__).resolve().parents[2]

#: The OpenAI-compatible facade over DeepInfer's offline API.
FACADE = REPO_ROOT / "tools" / "deepinfer" / "serve.py"

#: Startup wait for the harness-managed DeepInfer server (a first load pulls
#: the checkpoint, so the block's ``timeout_s`` overrides it).
DEFAULT_TIMEOUT_S = 900.0

#: The accelerator values DeepInfer's build understands, and the lane each
#: one implies.
_ACCELERATORS = ("nvidia", "ascend", "amd")


def _read_tail(path: Path, limit: int = 2000) -> str:
    try:
        with Path(path).open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - limit))
            return handle.read().decode("utf-8", "replace")
    except OSError:
        return ""


class DeepInferLauncher:
    """Start/stop the DeepInfer facade for one applied stack's endpoint tiers.

    The launcher tracks only the processes it started, per role.  A failed
    start leaves nothing registered; :meth:`stop` is a no-op for a role it
    does not own.
    """

    def __init__(
        self,
        *,
        state_dir: Any = None,
        log_dir: Any = None,
        spawner: Callable[..., object] = subprocess.Popen,
        prober: Optional[Callable[[str], dict[str, Any]]] = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self.state_dir = (Path(state_dir).absolute() if state_dir
                          else REPO_ROOT / "harness_state" / "deepinfer")
        self.log_dir = (Path(log_dir).absolute() if log_dir
                        else REPO_ROOT / "logs")
        self.spawner = spawner
        self.prober = prober or probe_endpoint
        self.timeout_s = float(timeout_s)
        self._procs: dict[str, Any] = {}
        self._meta: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # capabilities (the honest gate)
    # ------------------------------------------------------------------
    def capabilities(self, launch: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
        """What this host can run of DeepInfer.

        Reports the accelerator the vendored lane would pick, the graphics
        architecture, the device count, whether the hardware has FP8, whether
        the C++ extension is built, and whether the requested model is
        servable here.  Never raises: an unknown host reports ``ok`` False
        with a reason.
        """
        report: dict[str, Any] = {
            "ok": False,
            "accelerator": "",
            "gfx": "",
            "devices": 0,
            "fp8": False,
            "extension_built": False,
            "model": "",
            "model_ok": False,
            "reason": "",
        }
        try:
            from vendor.deepinfer_amd import hardware as amd_hardware
        except Exception:  # noqa: BLE001 - the overlay may not be importable
            try:
                sys_path = str(REPO_ROOT / "vendor")
                if sys_path not in sys.path:
                    import sys

                    sys.path.insert(0, sys_path)
                from deepinfer_amd import hardware as amd_hardware  # type: ignore
            except Exception as exc:  # noqa: BLE001
                report["reason"] = f"the AMD overlay is not importable: {exc}"
                return report

        try:
            gpus = amd_hardware.gpu_nodes()
        except Exception as exc:  # noqa: BLE001
            report["reason"] = f"cannot read the AMD topology: {exc}"
            return report

        report["devices"] = len(gpus)
        if gpus:
            report["accelerator"] = "amd"
            report["gfx"] = str(gpus[0].get("gfx") or "")
            report["fp8"] = amd_hardware.supports_fp8()

        if launch:
            repo = str(launch.get("repo") or "")
            report["extension_built"] = self._extension_built(repo)

        return report

    @staticmethod
    def _extension_built(repo: str) -> bool:
        """Whether ``deepinfer._C`` is built for the AMD platform tag."""
        if not repo:
            return False
        root = Path(repo).expanduser()
        so_dir = root / "deepinfer" / "_so"
        if not so_dir.is_dir():
            return False
        return any(
            (path / "deepinfer" in ()) or (path.is_dir() and any(path.glob("*.so")))
            for path in so_dir.iterdir()
        )

    # ------------------------------------------------------------------
    # resolution / generation (pure enough to test without processes)
    # ------------------------------------------------------------------
    def resolve(self, launch: Mapping[str, Any], endpoint: str) -> dict[str, Any]:
        """Resolve and sanity-check the block's repo/config/python/port."""
        repo = Path(str(launch.get("repo") or "")).expanduser()
        if not repo.is_dir():
            raise RuntimeError(f"deepinfer repo not found: {repo}")
        repo = repo.resolve()

        raw_config = Path(str(launch.get("config") or ""))
        config = raw_config if raw_config.is_absolute() else repo / raw_config
        config = config.resolve()
        if config != repo and repo not in config.parents:
            raise RuntimeError(
                f"deepinfer config {config} must stay inside the repo {repo}")
        if not config.is_file():
            raise RuntimeError(f"deepinfer config not found: {config}")

        raw_python = Path(str(launch.get("python") or ".venv/bin/python")).expanduser()
        python = raw_python if raw_python.is_absolute() else repo / raw_python
        # Keep the symlink path: a venv's python is a symlink to the system
        # interpreter, and resolving it would escape the venv's site-packages.
        python = python.absolute()
        if not python.is_file():
            raise RuntimeError(
                f"deepinfer python not found: {python} (set launch.python)")

        if not FACADE.is_file():
            raise RuntimeError(f"the DeepInfer facade is missing: {FACADE}")

        port = endpoint_port(endpoint)
        if port <= 0:
            raise RuntimeError(f"endpoint {endpoint!r} carries no usable port")
        return {"repo": repo, "config": config, "python": python, "port": port}

    def build_config(
        self,
        base: Mapping[str, Any],
        launch: Mapping[str, Any],
        tier: Tier,
        endpoint: str,
    ) -> dict[str, Any]:
        """The generated facade config: base + port + ctx + args + options."""
        config = json.loads(json.dumps(dict(base)))  # deep copy
        config["port"] = endpoint_port(endpoint)
        config["ctx"] = int(getattr(tier, "ctx", 4096))
        engine_args = [str(token) for token in (launch.get("engine_args") or [])]
        if engine_args:
            config["engine_args"] = engine_args
        options = launch.get("options")
        if isinstance(options, Mapping):
            config.update(dict(options))
        return config

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def alive(self, role: str) -> bool:
        proc = self._procs.get(role)
        return proc is not None and proc.poll() is None

    def managed(self, role: str) -> bool:
        """True when this launcher started (and owns) the role's process."""
        return role in self._procs

    def info(self, role: str) -> dict[str, Any]:
        return dict(self._meta.get(role) or {})

    def start(
        self,
        *,
        stack_name: str,
        role: str,
        tier: Tier,
        endpoint: str,
    ) -> dict[str, Any]:
        """Start the DeepInfer facade for ``tier`` and wait for its endpoint.

        Raises ``RuntimeError`` when paths resolve badly, the process exits
        during startup, or the endpoint does not answer within the timeout;
        nothing stays registered on failure.
        """
        launch = getattr(tier, "launch", None)
        if not isinstance(launch, Mapping):
            raise RuntimeError(f"tier {role!r} has no launch block")
        resolved = self.resolve(launch, endpoint)
        base = json.loads(resolved["config"].read_text(encoding="utf-8-sig"))

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", f"deepinfer-{stack_name}-{role}").strip("-")
        gen_path = self.state_dir / f"{slug}.json"
        log_path = self.log_dir / f"deepinfer_{resolved['port']}.log"

        config = self.build_config(base, launch, tier, endpoint)
        payload = json.dumps(config, indent=2, ensure_ascii=False) + "\n"
        tmp_path = gen_path.with_suffix(gen_path.suffix + ".tmp")
        tmp_path.write_text(payload, encoding="utf-8")
        os.replace(tmp_path, gen_path)

        cmd = [
            str(resolved["python"]),
            str(FACADE),
            "--config", str(gen_path),
            "--port", str(resolved["port"]),
        ]
        # The DeepInfer checkout and the AMD overlay must be importable.
        env = dict(os.environ)
        python_path = [
            str(resolved["repo"]),
            str(REPO_ROOT / "vendor"),
            str(REPO_ROOT / "vendor" / "deepinfer"),
            str(REPO_ROOT),
        ]
        existing = env.get("PYTHONPATH", "")
        if existing:
            python_path.append(existing)
        env["PYTHONPATH"] = os.pathsep.join(python_path)

        log_handle = open(log_path, "ab")
        try:
            proc = self.spawner(
                cmd, cwd=str(resolved["repo"]), env=env,
                stdout=log_handle, stderr=subprocess.STDOUT,
            )
        finally:
            log_handle.close()

        meta: dict[str, Any] = {
            "pid": getattr(proc, "pid", None),
            "cmd": cmd,
            "config": str(gen_path),
            "log": str(log_path),
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._procs[role] = proc
        self._meta[role] = meta

        timeout = float(launch.get("timeout_s") or self.timeout_s)
        deadline = time.time() + timeout
        last_error = ""
        while time.time() < deadline:
            if proc.poll() is not None:
                code = proc.poll()
                self.stop(role)
                raise RuntimeError(
                    f"deepinfer server for tier '{role}' exited during startup "
                    f"(code {code}); see {log_path}: {_read_tail(log_path)[-400:]}")
            try:
                health = self.prober(endpoint)
            except Exception as exc:  # noqa: BLE001 - keep polling until deadline
                last_error = str(exc)
                time.sleep(1.0)
                continue
            meta["health"] = health
            return dict(meta)

        self.stop(role)
        raise RuntimeError(
            f"deepinfer endpoint {endpoint} did not become healthy within "
            f"{timeout:.0f}s ({last_error or 'no health response'}); "
            f"see {log_path}"
        )

    def stop(self, role: str) -> bool:
        """Stop the process this launcher started for ``role`` (no-op if none)."""
        proc = self._procs.pop(role, None)
        self._meta.pop(role, None)
        if proc is None:
            return False
        try:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except Exception:  # noqa: BLE001 - escalate to kill
                proc.kill()
        except Exception:  # noqa: BLE001 - already gone is success
            pass
        return True

    def stop_all(self) -> list[str]:
        roles = list(self._procs)
        for role in roles:
            self.stop(role)
        return roles
