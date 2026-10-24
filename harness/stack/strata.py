"""Managed Strata lifecycle for endpoint-backed tiers (adopt-or-start).

A tier with a ``launch`` block (``kind: "strata"``) gets the hybrid contract:
on ``apply``, an already-running server at the tier's endpoint is **adopted**
(never killed); an endpoint that does not answer is **started** from a
generated copy of the block's config, and ``unload`` stops exactly that
process.

Generated config: the base config (Strata's ``strata-<model>.json``) is
copied, the listen ``port`` is synced to the endpoint URL, the engine ``args``
get the context flag (``--max-context`` by default, or the lane's
``ctx_flag`` - the DeepSeek-V4.1 engine takes ``--ctx``) with ``<tier.ctx>``
plus the block's ``engine_args`` (an existing flag is replaced in place, a new
one appended), the engine ``log`` is pointed at the harness log directory, and
``options`` are merged on top.  The command is Strata's own server:
``<repo>/.venv/bin/python serve/server.py --engine strata --config <generated>
--port <port>`` from ``<repo>``.

Files land under ``harness_state/strata/`` (generated config) and the shared
``logs/`` directory (server stdout+stderr, engine log), matching the
llama-server conventions.
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

#: Repo root (``harness/stack/strata.py`` → repo).
REPO_ROOT = Path(__file__).resolve().parents[2]

#: A flag token: ``--like-this`` / ``-x`` (a negative number is a value).
_FLAG = re.compile(r"^-{1,2}[A-Za-z]")

#: Startup wait for the harness-managed Strata server (first loads take longer
#: than later ones; the block's ``timeout_s`` overrides it).
DEFAULT_TIMEOUT_S = 600.0


def _is_flag(token: str) -> bool:
    return bool(_FLAG.match(token))


def merge_engine_args(base: list[str], overrides: list[str]) -> list[str]:
    """Merge a flag/value override list into a base argv in place.

    ``--flag value...`` spans replace the base's span for that flag (or append
    when absent); value-only tokens append.  This is how a stack overrides
    e.g. ``--vram-reserve-mib`` without duplicating it and how
    ``--max-context`` is forced to the tier's ctx.
    """
    args = [str(token) for token in base]
    pending = [str(token) for token in overrides]
    while pending:
        token = pending.pop(0)
        if not _is_flag(token):
            args.append(token)
            continue
        values: list[str] = []
        while pending and not _is_flag(pending[0]):
            values.append(pending.pop(0))
        replaced = False
        index = 0
        while index < len(args):
            if args[index] == token:
                end = index + 1
                while end < len(args) and not _is_flag(args[end]):
                    end += 1
                args[index:end] = [token, *values]
                replaced = True
                break
            index += 1
        if not replaced:
            args.extend([token, *values])
    return args


def _read_tail(path: Path, limit: int = 2000) -> str:
    try:
        with Path(path).open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - limit))
            return handle.read().decode("utf-8", "replace")
    except OSError:
        return ""


class StrataLauncher:
    """Start/stop ``serve/server.py`` for one applied stack's endpoint tiers.

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
                          else REPO_ROOT / "harness_state" / "strata")
        self.log_dir = (Path(log_dir).absolute() if log_dir
                        else REPO_ROOT / "logs")
        self.spawner = spawner
        self.prober = prober or probe_endpoint
        self.timeout_s = float(timeout_s)
        self._procs: dict[str, Any] = {}
        self._meta: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # resolution / generation (pure enough to test without processes)
    # ------------------------------------------------------------------
    def resolve(self, launch: Mapping[str, Any], endpoint: str) -> dict[str, Any]:
        """Resolve and sanity-check the block's repo/config/python/port."""
        repo = Path(str(launch.get("repo") or "")).expanduser()
        if not repo.is_dir():
            raise RuntimeError(f"strata repo not found: {repo}")
        repo = repo.resolve()

        raw_config = Path(str(launch.get("config") or ""))
        config = raw_config if raw_config.is_absolute() else repo / raw_config
        config = config.resolve()
        if config != repo and repo not in config.parents:
            raise RuntimeError(
                f"strata config {config} must stay inside the repo {repo}")
        if not config.is_file():
            raise RuntimeError(f"strata config not found: {config}")

        raw_python = Path(str(launch.get("python") or ".venv/bin/python")).expanduser()
        python = raw_python if raw_python.is_absolute() else repo / raw_python
        # Keep the symlink path: a venv's python is a symlink to the system
        # interpreter, and resolving it would escape the venv's site-packages
        # (the smoke run hit exactly that: `regex` went missing).
        python = python.absolute()
        if not python.is_file():
            raise RuntimeError(
                f"strata python not found: {python} (set launch.python)")

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
        engine_log: Path,
    ) -> dict[str, Any]:
        """The generated server config: base + port + ctx/args + log + options."""
        config = json.loads(json.dumps(dict(base)))  # deep copy
        config["port"] = endpoint_port(endpoint)
        # The context flag is the engine's own: the DeepSeek-V4.1 engine takes
        # `--ctx` and refuses anything it does not know, so the lane names it
        # (launch.ctx_flag).  The default is the Strata/llama.cpp spelling.
        ctx_flag = str(launch.get("ctx_flag") or "--max-context")
        overrides = [ctx_flag, str(int(getattr(tier, "ctx", 8192)))]
        overrides += [str(token) for token in (launch.get("engine_args") or [])]
        config["args"] = merge_engine_args(list(config.get("args") or []), overrides)
        config["log"] = str(engine_log)
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
        """Start Strata for ``tier`` and wait for its endpoint to be healthy.

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
        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", f"strata-{stack_name}-{role}").strip("-")
        gen_path = self.state_dir / f"{slug}.json"
        log_path = self.log_dir / f"strata_{resolved['port']}.log"
        engine_log = self.log_dir / f"strata_{resolved['port']}.engine.log"

        config = self.build_config(base, launch, tier, endpoint, engine_log)
        payload = json.dumps(config, indent=2, ensure_ascii=False) + "\n"
        tmp_path = gen_path.with_suffix(gen_path.suffix + ".tmp")
        tmp_path.write_text(payload, encoding="utf-8")
        os.replace(tmp_path, gen_path)

        cmd = [
            str(resolved["python"]),
            "serve/server.py",
            "--engine", str(launch.get("kind") or "strata"),
            "--config", str(gen_path),
            "--port", str(resolved["port"]),
        ]
        log_handle = open(log_path, "ab")
        try:
            proc = self.spawner(
                cmd, cwd=str(resolved["repo"]),
                stdout=log_handle, stderr=subprocess.STDOUT,
            )
        finally:
            log_handle.close()

        meta: dict[str, Any] = {
            "pid": getattr(proc, "pid", None),
            "cmd": cmd,
            "config": str(gen_path),
            "log": str(log_path),
            "engine_log": str(engine_log),
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
                    f"strata server for tier '{role}' exited during startup "
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
            f"strata endpoint {endpoint} did not become healthy within "
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
