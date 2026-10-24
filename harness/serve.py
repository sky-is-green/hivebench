"""``hive serve`` — apply one stack, keep it up, unload it on exit.

The command is deliberately engine-blind: a stack tier either names an engine
through its ``launch`` block (``strata``, ``deepinfer``, or whatever adapter is
registered) or is a llama-server tier with a ``backend``; an endpoint tier is
adopted after a health check.  This module only owns the lifecycle — it does not
know how any engine works, and adding one does not change this file.

For the common case it can also write the stack for you: ``--model`` turns a
HuggingFace GGUF and the cards it finds into a stack, so a two-card box does not
need a hand-edited JSON.

Usage::

    hive serve peer-2tier                  # a stack under stacks/
    hive serve stacks/my-stack.json        # or a path
    hive serve --model org/repo:model.gguf --cards 2
    hive serve --model org/repo --file model.gguf --cards 2 --save my-stack
    hive serve peer-2tier --once           # apply, print, unload (smoke test)
    hive serve --list                      # what is available

Exit codes: 0 on a clean stop, 1 when the stack cannot be applied, 2 on a usage
error.  SIGINT/SIGTERM unload the stack before exiting.
"""

from __future__ import annotations

import argparse
import json
import re
import signal
import sys
import time
from pathlib import Path
from typing import Any

from harness.hardware import linux_amd_devices
from harness.models import LlamaServerManager, backend_binary
from harness.stack.manager import StackManager
from harness.stack.schema import (
    Stack, list_stacks, load_stack, stack_path, validate_shape)


def _load_stack_arg(value: str, stacks_dir: str | None) -> Stack:
    """A saved stack name (``stacks/<name>.json``) or an explicit path."""
    path = Path(value)
    if path.suffix == ".json" or path.is_file() or "/" in value:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise FileNotFoundError(f"no stack file at {path}") from None
        except json.JSONDecodeError as exc:
            raise ValueError(f"stack file {path} is not valid JSON: {exc}") from exc
        stack = Stack.from_dict(data)
        problems = validate_shape(stack)
        if problems:
            raise ValueError(
                f"stack {stack.name!r} has an invalid shape: " + "; ".join(problems))
        return stack
    return load_stack(value, root=stacks_dir)


def _detect_devices() -> list[dict]:
    """The cards this machine can serve on, in HIP index order."""
    try:
        return linux_amd_devices()
    except OSError:
        return []


def _select_devices(devices: list[dict], count: int | None,
                    requested: str | None) -> list[dict]:
    if requested:
        try:
            wanted = [int(x) for x in requested.replace(" ", "").split(",") if x]
        except ValueError:
            raise ValueError("--devices must be comma-separated HIP indices, e.g. 0,1") from None
        by_index = {int(d["index"]): d for d in devices}
        missing = [i for i in wanted if i not in by_index]
        if missing:
            raise ValueError(
                f"--devices {missing} not among the detected cards {sorted(by_index)}")
        return [by_index[i] for i in wanted]
    if count is not None and count > len(devices):
        raise ValueError(f"--cards {count} but only {len(devices)} GPU(s) detected")
    n = count if count is not None else len(devices)
    return devices[:n] if n else []


def _pick_backend(devices: list[dict], requested: str | None) -> str | None:
    """The backend whose binary is actually installed, not just available."""
    if requested:
        return requested
    if devices:
        for name in ("rocm", "vulkan"):
            if backend_binary(name):
                return name
        return None
    return "cpu" if backend_binary("cpu") else None


def _slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-.")
    return slug[:64] or "stack"


def _stack_from_model(*, model: str, file: str, devices: list[dict], ctx: int,
                      backend: str | None, role: str,
                      name: str | None) -> tuple[dict, list[str]]:
    tier: dict[str, Any] = {
        "role": role, "repo": model, "file": file, "ctx": ctx, "ngl": 99,
        "cache_k": "q8_0", "cache_v": "q8_0",
    }
    if backend:
        tier["backend"] = backend
    warnings: list[str] = []
    if devices:
        tier["pin"] = ",".join(str(d["index"]) for d in devices)
        warnings += [
            f"card {d['index']} drives a display; expect desktop stutter"
            for d in devices if d.get("display")
        ]
        if len(devices) > 1:
            gb = [float(d.get("memory_gb") or 1.0) for d in devices]
            if max(gb) - min(gb) < 0.5:
                tier["ts"] = ",".join("1" for _ in gb)
            else:
                tier["ts"] = ",".join(f"{x:.0f}" for x in gb)
    stem = _slug(Path(file).stem)
    cards = f"{len(devices)}card" if devices else "cpu"
    data = {"name": name or _slug(f"{stem}-{cards}"), "version": 1, "tiers": [tier]}
    return data, warnings


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hive serve",
        description="Apply a stack (one process per tier, any engine) and keep it serving.")
    parser.add_argument("stack", nargs="?",
                        help="stack name under stacks/, or a path to a .json stack")
    parser.add_argument("--list", action="store_true",
                        help="list the saved stacks and exit")
    parser.add_argument("--model", default=None,
                        help="HuggingFace repo (or repo:file) to serve without a stack file")
    parser.add_argument("--file", default=None,
                        help="GGUF file inside the repo (with --model)")
    parser.add_argument("--cards", type=int, default=None,
                        help="how many detected cards to use (default: all)")
    parser.add_argument("--devices", default=None,
                        help="explicit HIP indices, e.g. 0,1 (overrides --cards)")
    parser.add_argument("--ctx", type=int, default=8192,
                        help="context size for a generated stack (default 8192)")
    parser.add_argument("--backend", default=None,
                        help="llama-server backend (default: the first installed of rocm, vulkan)")
    parser.add_argument("--role", default="face",
                        help="role to give a generated tier (default: face)")
    parser.add_argument("--save", default=None,
                        help="write a generated stack to stacks/<name>.json")
    parser.add_argument("--stacks-dir", default=None,
                        help="directory of stack files (default: <repo>/stacks)")
    parser.add_argument("--models-dir", default=None, help="GGUF model directory")
    parser.add_argument("--llama-server", default=None, help="llama-server binary")
    parser.add_argument("--log-dir", default=None, help="engine log directory")
    parser.add_argument("--host", default="127.0.0.1",
                        help="host the tiers bind/publish on (default 127.0.0.1)")
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="per-tier startup timeout in seconds (default 300)")
    parser.add_argument("--once", action="store_true",
                        help="apply, print the endpoints, unload, exit (smoke test)")
    parser.add_argument("--json", action="store_true", dest="as_json",
                        help="print the applied stack as JSON instead of a table")
    return parser


def _endpoint_of(row: dict[str, Any], host: str) -> str:
    base = row.get("base_url")
    if base:
        return str(base)
    port = row.get("port")
    return f"http://{host}:{port}/v1" if port else ""


def _print_table(rows: list[dict[str, Any]], host: str) -> None:
    width = max((len(str(r.get("role", ""))) for r in rows), default=4)
    for row in rows:
        role = str(row.get("role", "?"))
        endpoint = _endpoint_of(row, host)
        ctx = row.get("ctx")
        backend = row.get("backend") or ""
        extra = " ".join(str(x) for x in (backend, f"ctx {ctx}" if ctx else "") if x)
        print(f"  {role:<{width}}  {endpoint}")
        if extra:
            print(f"  {'':<{width}}  {extra}")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list:
        rows = list_stacks(root=args.stacks_dir)
        if not rows:
            print("no saved stacks")
            return 0
        for row in rows:
            roles = ",".join(row.get("roles", []))
            print(f"  {row.get('name', '?'):<24} {roles}")
        return 0

    generated: dict | None = None
    warnings: list[str] = []
    if args.model:
        if args.stack:
            parser.error("give either a stack name or --model, not both")
        model, file = args.model, args.file
        if file is None and ":" in model:
            model, _, file = model.partition(":")
        if not file:
            print("hive serve: --model needs the GGUF file: --file NAME or --model repo:file",
                  file=sys.stderr)
            return 2
        if model.endswith(".gguf") or Path(model).suffix == ".gguf":
            print("hive serve: local GGUF paths are not a stack field yet; use a repo:file",
                  file=sys.stderr)
            return 2
        if args.ctx <= 0:
            print("hive serve: --ctx must be positive", file=sys.stderr)
            return 2
        try:
            devices = _select_devices(_detect_devices(), args.cards, args.devices)
        except ValueError as exc:
            print(f"hive serve: {exc}", file=sys.stderr)
            return 2
        backend = _pick_backend(devices, args.backend)
        generated, warnings = _stack_from_model(
            model=model, file=file, devices=devices, ctx=args.ctx,
            backend=backend, role=args.role, name=args.save)
    elif not args.stack:
        parser.error("a stack name, a path, or --model is required (or --list)")

    try:
        if generated is not None:
            stack = Stack.from_dict(generated)
            problems = validate_shape(stack)
            if problems:
                raise ValueError(
                    f"generated stack {stack.name!r} is invalid: " + "; ".join(problems))
            if args.save:
                path = stack_path(args.save, root=args.stacks_dir)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(generated, indent=2) + "\n", encoding="utf-8")
                print(f"wrote {path}")
        else:
            stack = _load_stack_arg(args.stack, args.stacks_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"hive serve: {exc}", file=sys.stderr)
        return 2

    models = LlamaServerManager(
        binary=Path(args.llama_server) if args.llama_server else None,
        models_dir=Path(args.models_dir) if args.models_dir else None,
        host=args.host,
        log_dir=Path(args.log_dir) if args.log_dir else None,
        startup_timeout=args.timeout,
    )
    manager = StackManager(models, stacks_root=args.stacks_dir)

    try:
        manager.apply(stack)
    except RuntimeError as exc:
        print(f"hive serve: {exc}", file=sys.stderr)
        return 1

    status = manager.status()
    rows = list(status.get("tiers", []))
    if args.as_json:
        print(json.dumps(status, indent=2, sort_keys=True))
    else:
        print(f"stack {stack.name!r} applied")
        _print_table(rows, args.host)
        for warning in warnings + list(status.get("warnings", []) or []):
            print(f"  ! {warning}")

    if args.once:
        manager.unload()
        return 0

    print("serving; Ctrl-C to stop", flush=True)

    def _stop(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        manager.unload()
    print("stopped", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
