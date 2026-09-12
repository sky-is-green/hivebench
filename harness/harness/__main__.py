"""Run the harness sidecar: ``python -m harness [--host H] [--port P]``.

``--detach`` gives the desktop-app launch feel: the command relaunches itself
windowless (``pythonw``, detached process group), logs to ``logs/studio.log``,
opens the browser once the port answers, and returns immediately;
``--stop`` ends that instance. Foreground launches open the browser too
(``--no-open`` opts out).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

import uvicorn


def _log_file(log_dir: Path) -> Path:
    """Append-mode log capturing a detached instance's output."""
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "studio.log"


def _windowless_interpreter() -> str:
    """The console-less interpreter matching the current one (Windows)."""
    exe = Path(sys.executable)
    if sys.platform == "win32":
        windowless = exe.with_name("pythonw.exe")
        if windowless.exists():
            return str(windowless)
    return str(exe)


def _wait_and_open(url: str, timeout_s: float = 45.0) -> None:
    """Open the browser as soon as the HTTP port answers (dsh-web behavior)."""
    import socket
    from urllib.parse import urlsplit

    parts = urlsplit(url)
    port = parts.port or 80
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((parts.hostname or "127.0.0.1", port),
                                          timeout=0.4):
                webbrowser.open(url)
                return
        except OSError:
            time.sleep(0.25)


def _spawn_detached(child_argv: list[str], log_path: Path) -> int:
    """Relaunch this command headless; the caller exits right after."""
    log_handle = open(log_path, "ab")  # noqa: SIM115 - lifetime is the child's
    extra: dict = {"env": {**os.environ, "STRATA_STUDIO_CHILD": "1"}}
    if sys.platform == "win32":
        # No console attached at all; output lands in the log file.
        extra["creationflags"] = (subprocess.DETACHED_PROCESS
                                  | subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        extra["start_new_session"] = True
    proc = subprocess.Popen(
        [_windowless_interpreter(), *child_argv],
        stdin=subprocess.DEVNULL, stdout=log_handle, stderr=subprocess.STDOUT,
        **extra,
    )
    return proc.pid


def _stop(pidfile: Path) -> int:
    """Kill the detached instance recorded by --detach, plus every managed
    llama-server child (taskkill /T misses them once the parent detaches)."""
    if not pidfile.exists():
        print(f"no running studio recorded ({pidfile})")
        return 0
    raw = pidfile.read_text().strip()
    llama_pids: list[int] = []
    try:
        record = json.loads(raw)  # {"pid": ..., "llama_pids": [...]} (or legacy)
        pid = int(record["pid"])
        legacy = record.get("llama_pid")
        if legacy:
            llama_pids.append(int(legacy))
        llama_pids += [int(p) for p in record.get("llama_pids", []) or []]
    except (ValueError, TypeError):
        pid = int(raw)
    for llama_pid in llama_pids:
        try:
            import psutil

            psutil.Process(llama_pid).terminate()
        except Exception:  # noqa: BLE001 - already gone
            pass
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                       check=False, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    else:
        import signal
        os.kill(pid, signal.SIGTERM)
    pidfile.unlink(missing_ok=True)
    print(f"stopped studio (pid {pid}, {len(llama_pids)} model server(s))")
    return 0

from backend.providers import providers_path
from cortex.e2e import FakeUltraSmall, MockTransport
from harness.app import DEFAULT_PORT, OpenAICompatBackend, create_app
from harness.models import launch_extra_args, LlamaServerManager


def _protect_git_excludes(root: Path, names: list[str]) -> None:
    """Self-protecting runtime state (pattern borrowed from Faber): add the
    studio's runtime directories to ``.git/info/exclude`` — a local-only
    ignore that produces no diff and can never be committed — so state files
    (which may contain conversation content) can't reach GitHub even if the
    user forgets their own .gitignore. Idempotent; never fatal."""
    d = root.resolve()
    while True:
        git_dir = d / ".git"
        if git_dir.is_dir():
            break
        if d == d.parent:
            return  # not a git repository
        d = d.parent
    try:
        info = git_dir / "info"
        info.mkdir(parents=True, exist_ok=True)
        excl = info / "exclude"
        existing = excl.read_text(encoding="utf-8") if excl.exists() else ""
        missing = [n for n in names if n not in existing]
        if missing:
            with excl.open("a", encoding="utf-8") as fh:
                fh.write("\n# strata-memory studio runtime state (auto-added)\n")
                fh.write("\n".join(missing) + "\n")
    except OSError:
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="HiveBench Studio harness sidecar (FastAPI, local-only)",
    )
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind address (default 127.0.0.1; keep it local)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--providers-file", default="",
                        help="path to the providers JSON config "
                             "(default providers.local.json)")
    parser.add_argument("--log-dir", default="logs",
                        help="NDJSON event-log directory for strata turns")
    parser.add_argument(
        "--state-dir", default="harness_state",
        help="directory where conversations persist across restarts "
             "(default ./harness_state; pass an empty string to disable)",
    )
    parser.add_argument("--runs-root", default="",
                        help="root directory holding run bundles (default ./runs)")
    parser.add_argument(
        "--mock", action="store_true",
        help="offline mode: fake drone + mock backend (no LM Studio needed)",
    )
    parser.add_argument("--llama-server", default="",
                        help="path to llama-server binary (default "
                             "HARNESS_LLAMA_SERVER or tools/llama.cpp/)")
    parser.add_argument("--llama-port", type=int, default=1234,
                        help="port for the managed llama-server (default 1234, "
                             "LM Studio-compatible)")
    parser.add_argument("--models-dir", default="",
                        help="local GGUF library directory (default models/gguf)")
    parser.add_argument(
        "--no-auto-start", action="store_true",
        help="do not auto-launch llama-server even when a local model exists",
    )
    parser.add_argument(
        "--setup", action="store_true",
        help="guided first-run check: create providers.local.json if missing, "
             "probe for a reachable backend, and print the next command",
    )
    parser.add_argument(
        "--detach", action="store_true",
        help="launch headless: relaunch windowless, open the browser, and "
             "return immediately (log: <log-dir>/studio.log)",
    )
    parser.add_argument(
        "--stop", action="store_true",
        help="stop the instance previously launched with --detach",
    )
    parser.add_argument(
        "--no-open", action="store_true",
        help="do not open the browser once the server is up",
    )
    parser.add_argument(
        "--reload", action="store_true",
        help="auto-reload on Python file changes (dev: CSS in studio.css is live on refresh without this)",
    )
    args = parser.parse_args(argv)

    _protect_git_excludes(Path.cwd(), [
        "harness_state/", "logs/", "runs/", "runs_mock/", "transcripts/",
        ".sessions/", ".dsh-home/", "providers.local.json",
    ])

    if args.setup:
        return _setup(providers_path(args.providers_file or None))

    log_dir = Path(args.log_dir)
    pidfile = log_dir / "studio.pid"
    url = f"http://{args.host}:{args.port}"

    if args.stop:
        return _stop(pidfile)

    if args.detach:
        raw = sys.argv[1:] if argv is None else list(argv)
        child_argv = [a for a in raw if a not in ("--detach", "-d")]
        log_path = _log_file(log_dir)
        # Always relaunch through ``-m harness``: the current invocation may
        # be a script path (``python path/to/__main__.py``), whose argv[0]
        # must not leak into the child's interpreter arguments.
        pid = _spawn_detached(["-m", "harness", *child_argv], log_path)
        pidfile.write_text(str(pid))
        print(f"HiveBench Studio starting at {url} (pid {pid}; log: {log_path})")
        if not args.no_open:
            _wait_and_open(url)
        return 0

    kwargs = {
        "providers_file": providers_path(args.providers_file or None),
        "log_dir": args.log_dir,
        "state_dir": args.state_dir,  # empty string disables persistence
    }
    if args.runs_root:
        kwargs["runs_root"] = args.runs_root
    if args.mock:
        def mock_backend(model):
            return OpenAICompatBackend(transport=MockTransport(), model=model or "mock")

        kwargs["ultra_factory"] = FakeUltraSmall
        kwargs["backend_factory"] = mock_backend
        kwargs["runs_root"] = args.runs_root or Path("runs_mock")

    kwargs["models_manager"] = LlamaServerManager(
        binary=Path(args.llama_server) if args.llama_server else None,
        models_dir=Path(args.models_dir) if args.models_dir else None,
        log_dir=Path(args.log_dir),
        port=args.llama_port,
    )
    app = create_app(**kwargs)

    # Auto-start removed: studio no longer loads an LLM on boot.
    # Models are started explicitly via POST /v1/server/start or the UI Load button.
    # The --no-auto-start flag is kept for compatibility but is now the default.

    # Detached instances record every managed server so --stop can take the
    # whole stack down (taskkill /T cannot reach the grandchildren).
    if os.environ.get("STRATA_STUDIO_CHILD"):
        try:
            llama_pids = [i["pid"] for i in
                          app.state.models.status()["instances"] if i["pid"]]
            pidfile.write_text(json.dumps({"pid": os.getpid(),
                                           "llama_pids": llama_pids}))
        except Exception:  # noqa: BLE001 - pidfile is best-effort
            pass

    # Browser opens once the port answers — foreground and detached alike
    # (dsh-web behavior; --no-open opts out).
    if not args.no_open:
        threading.Thread(target=_wait_and_open, args=(url,), daemon=True).start()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info", reload=args.reload)
    return 0


def _setup(providers_file: Path) -> int:
    """Guided first run: config, backend probe, next command."""
    import shutil

    from backend.providers import DEFAULT_PROVIDERS_FILE

    print("HiveBench Studio — first-run setup")
    print("-" * 46)

    # 1. providers config
    if not providers_file.exists():
        example = Path("providers.example.json")
        if example.exists():
            providers_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(example, providers_file)
            print(f"  [ok] created {providers_file} from providers.example.json")
            print("       (default provider is 'lmstudio' on localhost:1234)")
        else:
            print(f"  [..] no providers.example.json found; using defaults "
                  f"({DEFAULT_PROVIDERS_FILE} not required)")
    else:
        print(f"  [ok] providers config present: {providers_file}")

    # 2. backend probe: LM Studio on :1234 (or a provider's base_url)
    from backend.openai_compat import OpenAICompatBackend

    found_backend = False
    for label, base in (("LM Studio (localhost:1234)", "http://localhost:1234"),):
        try:
            ok = OpenAICompatBackend(base_url=base).health()
        except Exception:  # noqa: BLE001
            ok = False
        print(f"  [{'ok' if ok else '..'}] {label}: "
              f"{'reachable' if ok else 'not running'}")
        found_backend = found_backend or ok

    # 3. drone model pre-fetch (the default ultra-small downloads ~60 MB from
    # Hugging Face on first use — do it now so the first turn is instant)
    try:
        from sieve.ultra_small import UltraSmallDrone

        UltraSmallDrone(confidence_mode="off")._ensure_loaded()
        print("  [ok] default drone ready (paraphrase-MiniLM-L3-v2, cached)")
    except Exception as exc:  # noqa: BLE001 - offline machines can still run --mock
        print(f"  [..] drone download skipped ({str(exc)[:80]})")

    # 4. local llama-server binary (managed mode)
    llama_bin = Path("tools/llama.cpp/llama-server.exe")
    if llama_bin.exists():
        print(f"  [ok] local llama-server found ({llama_bin}); "
              "auto-start will serve models/gguf on :1234")
    else:
        print("  [..] no local llama-server (auto-start needs models/gguf)")

    # 5. next step
    print("-" * 46)
    if found_backend:
        print("Ready. Start the studio with:")
        print("    python -m harness            # or: hivebench-harness")
        print("    python -m harness --detach   # headless: no console window,"
              " opens the browser")
        print("Then open http://127.0.0.1:8765")
    else:
        print("No reachable backend yet. Either:")
        print("  1. start LM Studio with a model loaded (localhost:1234), or")
        print("  2. drop GGUF files into models/gguf and run:")
        print("     python -m harness")
        print("     (auto-start will launch llama-server for you)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
