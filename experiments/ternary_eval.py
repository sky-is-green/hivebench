"""T11 — hivebench evaluation driver for ternary GGUF artifacts.

The evaluation path shared by T20/T29:

1. serve one GGUF through the harness model manager (``harness.models``),
2. run a 5-prompt smoke against the served llama-server,
3. optionally run one ``experiments.paired_ab`` subset against the same
   backend,
4. write a stable JSON report (``--output``).

Live mode needs a llama-server binary (``--fork-bin`` or
``$HARNESS_LLAMA_SERVER``); the released Bonsai ``PQ2_0`` artifacts use the
Prism ROCm fork. The paired subset delegates to
``python -m experiments.paired_ab --live``, so it inherits the F6
``STRATA_HOME`` pin. ``--mock`` exercises the whole wiring offline with a stub
manager, chat, and paired step — no GPU, network, or sibling checkout needed.

``SPEC_SHA256`` pins the frozen wire contract consumed from
``experiments/ternary/spec.md`` (``tbr-1.2``).
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

SPEC_SHA256 = "0d2c008b4aee726351f9b90e44ec003c18b579d8690db24c77a089d9e1fc652b"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8090
DEFAULT_NGL = 99
DEFAULT_CTX = 8192
DEFAULT_MAX_TOKENS = 64
CHAT_TIMEOUT = 180.0

# Reasoning models (Bonsai/Qwen templates) otherwise burn the reply budget on
# hidden thoughts and return empty visible content; the Prism fork takes this
# as a spawn-time llama-server flag.
NO_THINKING_ARGS: tuple[str, ...] = (
    "--chat-template-kwargs",
    '{"enable_thinking": false}',
)

SMOKE_PROMPTS: tuple[str, ...] = (
    "Reply with the single word: ok",
    "Name the capital of France.",
    "What is 17 times 3? Answer with the number only.",
    "Write one short sentence about the sea.",
    "Repeat exactly: ternary.",
)


def _free_port(host: str = DEFAULT_HOST) -> int:
    with socket.socket() as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _server_alive(base_url: str, timeout: float = 2.0) -> bool:
    """TCP reachability probe used as a mid-run liveness check."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(str(base_url))
        with socket.create_connection((parsed.hostname or DEFAULT_HOST,
                                       parsed.port or 80), timeout=timeout):
            return True
    except OSError:
        return False


def _git_rev() -> str:
    try:
        proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
                              capture_output=True, text=True, timeout=5)
        return proc.stdout.strip() if proc.returncode == 0 else ""
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return ""


def _hardware() -> dict:
    try:
        from harness.app import _hardware_summary
        hw = _hardware_summary()
        return {k: hw.get(k) for k in ("vram_gb", "combined_vram_gb",
                                       "available_ram_gb", "vram_source")}
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return {}


def build_manager(*, binary=None, models_dir=None, log_dir=None,
                  host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                  spawner=None, prober=None, startup_timeout: float = 300.0):
    """Construct the real ``harness.models.LlamaServerManager``.

    Imported lazily: the harness package pulls sibling Strata modules, so
    offline callers inject the stub manager instead."""
    from harness.models import LlamaServerManager

    kwargs: dict = {}
    if spawner is not None:
        kwargs["spawner"] = spawner
    if prober is not None:
        kwargs["prober"] = prober
    return LlamaServerManager(
        binary=binary, models_dir=models_dir, log_dir=log_dir, host=host,
        port=port, startup_timeout=startup_timeout, **kwargs,
    )


class _StubManager:
    """Offline stand-in for ``LlamaServerManager`` (``--mock`` and tests).

    Mirrors the ``load``/``unload``/``status`` surface the driver uses."""

    def __init__(self, *, host: str = DEFAULT_HOST, port: int | None = None,
                 binary=None, models_dir=None, log_dir=None) -> None:
        self.host = host
        self.port = int(port) if port else _free_port(host)
        self.binary = Path(binary) if binary else None
        self.models_dir = Path(models_dir) if models_dir else None
        self.log_dir = Path(log_dir) if log_dir else None
        self._instances: dict[str, dict] = {}

    def load(self, model=None, key: str | None = None, ctx_size: int = DEFAULT_CTX,
             ngl: int = DEFAULT_NGL, backend: str | None = None,
             extra_args: Sequence[str] | None = None, embedding: bool = False) -> dict:
        path = Path(str(model))
        key = key or path.stem
        port = self.port + len(self._instances)
        instance = {
            "key": key,
            "running": True,
            "healthy": True,
            "port": port,
            "base_url": f"http://{self.host}:{port}",
            "model": key,
            "pid": 0,
            "adopted": False,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "backend": backend or "",
            "binary": str(self.binary or ""),
            "embedding": embedding,
        }
        self._instances[key] = instance
        return dict(instance)

    def unload(self, key: str) -> dict:
        if key not in self._instances:
            raise RuntimeError(f"no such loaded instance: {key}")
        del self._instances[key]
        return {"ok": True, "unloaded": key}

    def status(self) -> dict:
        return {"running": bool(self._instances),
                "instances": [dict(i) for i in self._instances.values()]}


def build_mock_manager(tmp_dir=None) -> tuple[_StubManager, Path]:
    """Offline manager + fake GGUF for ``--mock`` and unit tests."""
    root = Path(tmp_dir) if tmp_dir else Path(tempfile.mkdtemp(prefix="ternary-eval-"))
    root.mkdir(parents=True, exist_ok=True)
    gguf = root / "mock-ternary.gguf"
    gguf.write_bytes(b"GGUF" + b"\x00" * 32)
    manager = _StubManager(binary=root / "llama-server",
                           models_dir=root / "models", log_dir=root / "logs")
    return manager, gguf


def serve_gguf(gguf, *, manager, key: str | None = None,
               ctx_size: int = DEFAULT_CTX, ngl: int = DEFAULT_NGL,
               backend: str | None = None, extra_args: Sequence[str] | None = None,
               embedding: bool = False) -> dict:
    """Serve one GGUF through the model manager and return its instance dict."""
    path = Path(gguf)
    if not path.is_file():
        raise FileNotFoundError(f"GGUF not found: {path}")
    return manager.load(model=str(path), key=key or path.stem, ctx_size=ctx_size,
                        ngl=ngl, backend=backend, extra_args=extra_args,
                        embedding=embedding)


def chat_completion(base_url: str, prompt: str, *, model: str = "",
                    system: str = "You are a concise assistant.",
                    max_tokens: int = DEFAULT_MAX_TOKENS, temperature: float = 0.0,
                    timeout: float = CHAT_TIMEOUT, http=None) -> dict:
    """One non-streaming chat completion against an OpenAI-compatible server.

    ``http`` is injectable for tests (needs ``post(url, json=, timeout=)``)."""
    import requests

    http = http or requests
    payload = {
        "model": model or "local",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    resp = http.post(f"{str(base_url).rstrip('/')}/v1/chat/completions",
                     json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices") or [{}]
    message = choices[0].get("message") or {}
    usage = data.get("usage") or {}
    return {
        "prompt": prompt,
        "response": message.get("content") or "",
        "completion_tokens": usage.get("completion_tokens"),
        "model": data.get("model") or model,
    }


def run_smoke(instance: dict, *, prompts: Sequence[str] | None = None,
              chat_fn=None, max_tokens: int = DEFAULT_MAX_TOKENS,
              timeout: float = CHAT_TIMEOUT, deadline: float | None = None,
              health_fn=None) -> dict:
    """Run the 5-prompt smoke; per-prompt failures are recorded, not raised."""
    chat_fn = chat_fn or chat_completion
    prompts = list(prompts or SMOKE_PROMPTS)
    base_url = instance["base_url"]
    rows: list[dict] = []
    for prompt in prompts:
        started = time.monotonic()
        if deadline is not None and started > deadline:
            rows.append({"prompt": prompt, "response": "", "status": "FAIL",
                         "error": "deadline exceeded", "seconds": 0.0})
            continue
        if health_fn is not None and not health_fn(base_url):
            rows.append({"prompt": prompt, "response": "", "status": "FAIL",
                         "error": "server not reachable", "seconds": 0.0})
            continue
        try:
            out = chat_fn(base_url, prompt,
                          model=instance.get("model") or "", max_tokens=max_tokens,
                          timeout=timeout)
            response = str(out.get("response") or "")
            row = {
                "prompt": prompt,
                "response": response,
                "status": "PASS" if response.strip() else "EMPTY",
                "completion_tokens": out.get("completion_tokens"),
            }
        except Exception as exc:  # noqa: BLE001 - every failure lands in the report
            row = {"prompt": prompt, "response": "", "status": "FAIL",
                   "error": str(exc)}
        row["seconds"] = round(time.monotonic() - started, 3)
        rows.append(row)
    passed = sum(1 for row in rows if row["status"] == "PASS")
    return {"prompts": len(rows), "passed": passed,
            "failed": len(rows) - passed, "rows": rows}


def _run_command(argv: Sequence[str]):
    return subprocess.run(list(argv), cwd=str(REPO_ROOT),
                          capture_output=True, text=True)


def paired_ab_subset(*, base_url: str, model: str = "",
                     conversations: str = "tests/fixtures/generated",
                     max_convs: int = 1, max_turns: int | None = None,
                     output=None, run=None) -> dict:
    """Run one ``paired_ab`` subset against the served backend.

    Delegates to the paired_ab CLI (``--live``), which owns the sibling
    imports and the F6 ``STRATA_HOME`` env; ``run`` is injectable for tests."""
    out_path = Path(output) if output else REPO_ROOT / "logs" / "ternary_paired_ab.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    argv = [sys.executable, "-m", "experiments.paired_ab", "--live",
            "--conversations", str(conversations),
            "--max-convs", str(int(max_convs)),
            "--base-url", str(base_url)]
    if model:
        argv += ["--model", str(model)]
    if max_turns:
        argv += ["--max-turns", str(int(max_turns))]
    argv += ["--output", str(out_path)]
    runner = run or _run_command
    proc = runner(argv)
    returncode = int(getattr(proc, "returncode", 0) or 0)
    if returncode != 0:
        stderr = getattr(proc, "stderr", "") or ""
        raise RuntimeError(f"paired_ab subset failed ({returncode}): {stderr[-400:]}")
    if out_path.is_file():
        return json.loads(out_path.read_text(encoding="utf-8"))
    return {"skipped": "paired_ab wrote no report", "command": argv}


def evaluate(*, gguf, manager, chat_fn=None, paired_fn=None,
             prompts: Sequence[str] | None = None,
             max_tokens: int = DEFAULT_MAX_TOKENS, ctx_size: int = DEFAULT_CTX,
             ngl: int = DEFAULT_NGL, backend: str | None = None,
             extra_args: Sequence[str] | None = None, key: str | None = None,
             keep_loaded: bool = False, timeout: float = CHAT_TIMEOUT,
             deadline: float | None = None, warmup: bool = False,
             health_fn=None, manifest: dict | None = None) -> dict:
    """Serve, smoke, optional paired subset, unload; return the report dict."""
    started = time.monotonic()
    if deadline:
        deadline = started + float(deadline)
    instance = serve_gguf(gguf, manager=manager, key=key, ctx_size=ctx_size,
                          ngl=ngl, backend=backend, extra_args=extra_args)
    report: dict = {"task": "T11", "spec_sha256": SPEC_SHA256,
                    "gguf": str(gguf), "instance": instance,
                    "manifest": manifest or {}, "smoke": None, "paired_ab": None}
    try:
        if health_fn is not None and not health_fn(instance["base_url"]):
            report["smoke"] = {"prompts": 0, "passed": 0, "failed": 0, "rows": [],
                               "error": "server not reachable after load"}
        else:
            if warmup:
                try:
                    (chat_fn or chat_completion)(
                        instance["base_url"], "Reply with the single word: ok",
                        model=instance.get("model") or "", max_tokens=4,
                        timeout=timeout)
                except Exception:  # noqa: BLE001 - warmup is best-effort
                    pass
            report["smoke"] = run_smoke(instance, prompts=prompts, chat_fn=chat_fn,
                                        max_tokens=max_tokens, timeout=timeout,
                                        deadline=deadline, health_fn=health_fn)
        if paired_fn is not None:
            if deadline is not None and time.monotonic() > deadline:
                report["paired_ab"] = {"error": "deadline exceeded before paired subset"}
            else:
                try:
                    report["paired_ab"] = paired_fn(instance)
                except Exception as exc:  # noqa: BLE001 - reported, not raised
                    report["paired_ab"] = {"error": str(exc)}
    finally:
        if not keep_loaded:
            try:
                report["unloaded"] = manager.unload(instance["key"])
            except Exception as exc:  # noqa: BLE001
                report["unloaded"] = {"error": str(exc)}
    report["seconds"] = round(time.monotonic() - started, 3)
    smoke = report["smoke"] or {}
    paired = report["paired_ab"]
    report["ok"] = (bool(smoke) and smoke.get("failed", 1) == 0
                    and not smoke.get("error")
                    and not (isinstance(paired, dict) and paired.get("error")))
    return report


def write_report(report: dict, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return path


def _mock_chat(base_url: str, prompt: str, *, model: str = "",
               max_tokens: int = DEFAULT_MAX_TOKENS, **kwargs) -> dict:
    return {"prompt": prompt, "response": "ok", "completion_tokens": 1,
            "model": model or "mock"}


def _mock_paired(instance: dict) -> dict:
    return {"mock": True, "turns_compared": 0, "base_url": instance["base_url"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ternary GGUF eval driver: serve through the harness model "
                    "manager, run a 5-prompt smoke and an optional paired_ab subset.",
    )
    parser.add_argument("--gguf", default="", help="GGUF artifact to serve")
    parser.add_argument("--fork-bin", default="",
                        help="llama-server binary (Prism fork for PQ2_0); "
                             "defaults to $HARNESS_LLAMA_SERVER")
    parser.add_argument("--backend", default="",
                        help="tools/backends/<backend> binary selector "
                             "(ignored when --fork-bin is given)")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--ngl", default="auto",
                        help="GPU layers: an integer, or 'auto' to fit to free "
                             "VRAM and offload the rest to CPU/GTT")
    parser.add_argument("--ctx-size", type=int, default=DEFAULT_CTX)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--conversations", default="tests/fixtures/generated",
                        help="conversation fixture dir for the paired subset")
    parser.add_argument("--max-convs", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--no-paired", action="store_true",
                        help="skip the paired_ab subset")
    parser.add_argument("--keep-loaded", action="store_true",
                        help="leave llama-server running after the eval")
    parser.add_argument("--no-thinking", action="store_true",
                        help="disable template thinking (reasoning models return "
                             "empty visible replies otherwise); passed to "
                             "llama-server as --chat-template-kwargs")
    parser.add_argument("--output", default="", help="report JSON path")
    parser.add_argument("--startup-timeout", type=float, default=300.0,
                        help="seconds to wait for llama-server to become healthy "
                             "(raise for large/offloaded models)")
    parser.add_argument("--timeout", type=float, default=CHAT_TIMEOUT,
                        help="per-request chat timeout in seconds")
    parser.add_argument("--deadline", type=float, default=0.0,
                        help="global run budget in seconds (0 = unlimited)")
    parser.add_argument("--no-warmup", dest="warmup", action="store_false",
                        default=True,
                        help="skip the unscored warmup turn before the smoke")
    parser.add_argument("--no-health", dest="health", action="store_false",
                        default=True,
                        help="skip mid-run TCP liveness checks")
    parser.add_argument("--expect-name", default="",
                        help="refuse to score unless the GGUF name contains this "
                             "(reference-identity guard, e.g. Qwen3.8-27B)")
    parser.add_argument("--expect-arch", default="",
                        help="refuse to score unless the GGUF architecture matches")
    parser.add_argument("--forbid", default="",
                        help="comma-separated name markers that fail the reference "
                             "identity guard (e.g. dflash,uncensored,turbo)")
    parser.add_argument("--mock", action="store_true",
                        help="offline wiring check (stub manager/chat/paired)")
    args = parser.parse_args(argv)

    if args.mock:
        manager, gguf = build_mock_manager()
        chat_fn = _mock_chat
        paired_fn = None if args.no_paired else _mock_paired
        health_fn = None
        manifest: dict = {"mode": "mock", "spec_sha256": SPEC_SHA256}
    else:
        if not args.gguf:
            print("error: --gguf is required (or use --mock)", file=sys.stderr)
            return 2
        gguf = Path(args.gguf)
        manager = build_manager(
            binary=Path(args.fork_bin) if args.fork_bin else None,
            log_dir=REPO_ROOT / "logs", host=args.host, port=args.port,
            startup_timeout=args.startup_timeout,
        )
        chat_fn = None
        stem = Path(args.gguf).stem
        try:
            from harness.models import gguf_identity, verify_reference
            identity = gguf_identity(gguf)
        except Exception:  # noqa: BLE001 - provenance is best-effort
            identity = {}
        if args.expect_name or args.expect_arch:
            check = verify_reference(
                gguf, expect_name=args.expect_name or None,
                expect_arch=args.expect_arch or None,
                forbid=[m for m in args.forbid.split(",") if m])
            if not check.get("ok", True):
                print(f"error: reference identity check failed: "
                      f"{check.get('reason')}", file=sys.stderr)
                return 2
        manifest = {
            "mode": "live",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git_rev": _git_rev(),
            "spec_sha256": SPEC_SHA256,
            "binary": str(args.fork_bin or args.backend or ""),
            "identity": identity,
            "ctx_size": args.ctx_size,
            "ngl": args.ngl,
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "timeout": args.timeout,
            "startup_timeout": args.startup_timeout,
            "warmup": args.warmup,
            "hardware": _hardware(),
        }
        health_fn = _server_alive if args.health else None

        if args.no_paired:
            paired_fn = None
        else:
            def paired_fn(instance: dict) -> dict:
                return paired_ab_subset(
                    base_url=instance["base_url"],
                    model=instance.get("model") or "",
                    conversations=args.conversations, max_convs=args.max_convs,
                    max_turns=args.max_turns,
                    output=REPO_ROOT / "logs" / f"ternary_paired_{stem}.json",
                )

    try:
        report = evaluate(
            gguf=gguf, manager=manager, chat_fn=chat_fn, paired_fn=paired_fn,
            max_tokens=args.max_tokens, ctx_size=args.ctx_size, ngl=args.ngl,
            backend=(None if args.fork_bin else (args.backend or None)),
            extra_args=list(NO_THINKING_ARGS) if args.no_thinking else None,
            keep_loaded=args.keep_loaded, timeout=args.timeout,
            deadline=(args.deadline or None), warmup=args.warmup,
            health_fn=health_fn, manifest=manifest,
        )
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        try:
            output = Path(args.output) if args.output else (
                REPO_ROOT / "artifacts" / "ternary" / "eval"
                / f"{Path(str(gguf)).stem}.json")
            write_report({
                "task": "T11", "spec_sha256": SPEC_SHA256, "gguf": str(gguf),
                "error": str(exc), "error_type": type(exc).__name__, "ok": False,
            }, output)
            print(f"  report : {output}")
        except Exception:  # noqa: BLE001 - best-effort failure report
            pass
        return 2

    output = Path(args.output) if args.output else (
        REPO_ROOT / "artifacts" / "ternary" / "eval" / f"{Path(str(gguf)).stem}.json")
    write_report(report, output)

    smoke = report["smoke"] or {}
    paired = report["paired_ab"]
    print(f"eval {'(mock) ' if args.mock else ''}{gguf}")
    print(f"  served : {report['instance'].get('base_url')} "
          f"({report['instance'].get('model')})")
    print(f"  smoke  : {smoke.get('passed', 0)}/{smoke.get('prompts', 0)} PASS")
    if paired is not None:
        if isinstance(paired, dict) and paired.get("error"):
            print(f"  paired : ERROR {paired['error']}")
        else:
            turns = paired.get("metrics", {}).get("turns_compared", "?")
            print(f"  paired : {'mock' if paired.get('mock') else turns} turn(s)")
    print(f"  report : {output}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
