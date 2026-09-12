"""Console slash commands — dsh command conventions over the Studio seam.

Mirrors ``@deepseek-ai/dsh-commands``: lowercase names without the slash,
immutable descriptors (``name``/``description``/``input.hint``) served to the
UI for discovery, ``/name raw-input`` parsing, and handlers that settle as
``{kind: 'success'|'error', text}``.

Scope note: dsh's own interactive commands (/plan, /compact, /goal) live in
its Web host; the SDK transport intentionally does not expose them. These are
the Studio's console commands — same shape, wired to the seams we own
(server, providers, engines, strata conversations, benchmark, transcripts).
"""

from __future__ import annotations

import html
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from cortex.config import StrataConfig


@dataclass
class CommandDescriptor:
    name: str
    description: str
    input_hint: Optional[str] = None

    def to_dict(self) -> dict:
        out = {"name": self.name, "description": self.description}
        if self.input_hint:
            out["input"] = {"hint": self.input_hint}
        return out


@dataclass
class CommandResult:
    kind: str  # success | error
    text: str = ""
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "text": self.text, **self.extras}


def parse_command(line: str) -> Optional[tuple[str, str]]:
    """/name raw-input -> (name, raw_input); None when not a command line.

    dsh convention: raw_input is the VERBATIM remainder including separator
    whitespace ('/goal\\ncreate' -> ('goal', '\\ncreate'))."""
    stripped = line.strip()
    if not stripped.startswith("/"):
        return None
    body = stripped[1:]
    if not body:
        return ("", "")
    match = re.match(r"^(\S+)(.*)$", body, flags=re.DOTALL)
    return (match.group(1).lower(), match.group(2)) if match else (body.lower(), "")


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "-", text).strip("-") or "conversation"


# ---------------------------------------------------------------------------
# transcript export (/save)
# ---------------------------------------------------------------------------
def _hive_markdown(st, conversation_id: str) -> list[str]:
    """Strata-side turns from the persisted conversation store: per turn the
    query chunk was stored before the reply chunk."""
    path = st._conv_path(conversation_id)
    lines: list[str] = []
    if path is None or not path.exists():
        return lines
    data = json.loads(path.read_text(encoding="utf-8"))
    store = data.get("store") or {}
    chunks = store.get("chunks") or []
    by_turn: dict[int, dict[str, str]] = {}
    for chunk in chunks:
        turn = int(chunk.get("turn", 0))
        slot = by_turn.setdefault(turn, {})
        if "query" not in slot:
            slot["query"] = chunk.get("content", "")
        else:
            slot["reply"] = chunk.get("content", "")
    for turn in sorted(by_turn):
        entry = by_turn[turn]
        lines.append(f"## Turn {turn}")
        lines.append(f"**User:** {html.escape(entry.get('query', ''))}")
        if entry.get("reply"):
            lines.append(f"\n**Assistant:** {html.escape(entry['reply'])}")
        lines.append("")
    return lines


def _agent_markdown(session_root: Path, conversation_id: str) -> list[str]:
    """Agent-side turns from the durable dsh session log:
    ``<session_root>/<cwd-slug>/agent-<id>/session.jsonl.zstd``."""
    lines: list[str] = []
    root = Path(session_root)
    if not root.is_dir():
        return lines
    session_files = list(root.glob(f"*/agent-{conversation_id}/session.jsonl.zstd"))
    for path in session_files:
        for raw in _read_zstd_lines(path):
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                continue
            kind = record.get("type")
            if kind == "user/message":
                blocks = record.get("data", {}).get("content") or []
                text = "".join(b.get("text", "") for b in blocks
                               if isinstance(b, dict) and b.get("type") == "text")
                if text.strip() and "strata-curated-context" not in text:
                    lines.append(f"**User:** {html.escape(text.strip())}\n")
            elif kind == "assistant/message":
                blocks = record.get("data", {}).get("message", {}).get("content") or []
                text = "".join(b.get("text", "") for b in blocks
                               if isinstance(b, dict) and b.get("type") == "text")
                if text.strip():
                    lines.append(f"**Assistant:** {html.escape(text.strip())}\n")
    return lines


def _read_zstd_lines(path: Path) -> list[str]:
    import zstandard

    dctx = zstandard.ZstdDecompressor()
    with open(path, "rb") as fh:
        reader = dctx.stream_reader(fh, read_across_frames=True)
        text = reader.read().decode("utf-8", errors="replace")
    return [ln for ln in text.splitlines() if ln.strip()]


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
Handler = Callable[[str, str], CommandResult]


@dataclass
class ConsoleCommands:
    """Descriptor registry + handlers; wired to the app's services."""

    st: object
    models: object
    agent: object
    transcripts_dir: Path

    def __post_init__(self) -> None:
        self._handlers: dict[str, Handler] = {
            "help": self._help,
            "new": self._new,
            "save": self._save,
            "model": self._model,
            "mode": self._mode,
            "provider": self._provider,
            "engine": self._engine,
            "bench": self._bench,
            "status": self._status,
        }
        self._descriptors = [
            CommandDescriptor("help", "list available commands"),
            CommandDescriptor("new", "start a fresh conversation (strata + agent)"),
            CommandDescriptor("save", "export the conversation transcript as markdown",
                              input_hint="[name]"),
            CommandDescriptor("model", "list local models or load one",
                              input_hint="[name]"),
            CommandDescriptor("mode", "switch the chat pane transport",
                              input_hint="[strata|agent]"),
            CommandDescriptor("provider", "list providers or set the default",
                              input_hint="[name]"),
            CommandDescriptor("engine", "list engines or set the default",
                              input_hint="[name]"),
            CommandDescriptor("bench", "run the mock HiveBench protocol",
                              input_hint="[max_convs]"),
            CommandDescriptor("status", "server / agent / provider summary"),
        ]

    # ------------------------------------------------------------------
    def descriptors(self) -> list[dict]:
        return [d.to_dict() for d in self._descriptors]

    def run(self, line: str, conversation_id: str) -> CommandResult:
        parsed = parse_command(line)
        if parsed is None:
            return CommandResult("error", "not a command line")
        name, raw_args = parsed
        if not name:
            return self._help("")
        handler = self._handlers.get(name)
        if handler is None:
            return CommandResult(
                "error", f"unknown command '/{name}' — try /help")
        try:
            return handler(raw_args.strip(), conversation_id)
        except Exception as exc:  # noqa: BLE001 - handlers settle as errors
            return CommandResult("error", f"/{name} failed: {exc}")

    # ------------------------------------------------------------------
    def _help(self, _args: str, _cid: str) -> CommandResult:
        lines = [f"/{d.name}{(' — ' + d.description) if d.description else ''}"
                 for d in self._descriptors]
        return CommandResult("success", "\n".join(lines))

    def _new(self, _args: str, _cid: str) -> CommandResult:
        new_id = f"console-{int(time.time() * 1000):x}"
        return CommandResult(
            "success", "Fresh conversation started (strata store + agent session).",
            {"new_conversation_id": new_id})

    def _save(self, args: str, conversation_id: str) -> CommandResult:
        lines = ["# Conversation transcript — "
                 f"{conversation_id}", "",
                 f"_exported {time.strftime('%Y-%m-%d %H:%M:%S')}_", ""]
        strata_lines = _hive_markdown(self.st, conversation_id)
        agent_lines = _agent_markdown(self.agent.session_root, conversation_id)
        if strata_lines:
            lines += ["## Strata conversation", ""] + strata_lines
        if agent_lines:
            lines += ["## Agent (dsh) session", ""] + agent_lines
        if not strata_lines and not agent_lines:
            return CommandResult("error",
                                 "nothing to save for this conversation yet")
        stamp = time.strftime("%Y%m%d_%H%M%S")
        name = _safe_name(args) if args else _safe_name(conversation_id)
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)
        out = self.transcripts_dir / f"{stamp}-{name}.md"
        out.write_text("\n".join(lines), encoding="utf-8")
        return CommandResult("success", f"saved {out}")

    def _model(self, args: str, _cid: str) -> CommandResult:
        local = self.models.list_local()
        if not args:
            current = self.models.status().get("model") or "(none loaded)"
            lines = [f"loaded: {current}"] + [
                f"  {m['file']} — {m['size_gb']} GB" for m in local]
            return CommandResult("success", "\n".join(lines))
        resolved = self.models.resolve_model(args)
        if resolved is None:
            return CommandResult("error",
                                 f"no local model matches '{args}'")
        from harness.models import launch_extra_args

        load_options = self._last_load_options()
        info = self.models.start(model=str(resolved),
                                 ctx_size=load_options.get("context", 8192),
                                 ngl=load_options.get("gpu_layers", 999),
                                 extra_args=launch_extra_args(load_options))
        self.st_app().register_local(info, load_options=load_options)
        return CommandResult("success",
                             f"loaded {resolved.name} on port {info['port']}")

    def _launch_extra_args(self, load_options: dict) -> list[str]:
        from harness.models import launch_extra_args

        return launch_extra_args(load_options)

    def _last_load_options(self) -> dict:
        try:
            engine = self.st.engines.resolve("local")
            return dict(engine.load_options or {})
        except LookupError:
            return {}

    def st_app(self):
        """The FastAPI app (register_local lives there); set at wiring time."""
        return self._app

    def _mode(self, args: str, _cid: str) -> CommandResult:
        mode = args.lower()
        if mode not in ("strata", "agent"):
            return CommandResult("error", "usage: /mode [strata|agent]")
        return CommandResult("success", f"chat mode set to {mode}",
                             {"mode": mode})

    def _provider(self, args: str, _cid: str) -> CommandResult:
        reg = self.st.registry
        if not args:
            lines = [f"default: {reg.default or '(first)'}"] + [
                f"  {p.name} -> {p.base_url}" for p in reg.providers]
            return CommandResult("success", "\n".join(lines))
        for p in reg.providers:
            if p.name.lower() == args.lower():
                reg.default = p.name
                return CommandResult("success", f"default provider: {p.name}")
        return CommandResult("error", f"unknown provider '{args}'")

    def _engine(self, args: str, _cid: str) -> CommandResult:
        reg = self.st.engines
        if not args:
            lines = [f"default: {reg.default or '(first)'}"] + [
                f"  {e.name} ({e.kind})" for e in reg.engines]
            return CommandResult("success", "\n".join(lines))
        for e in reg.engines:
            if e.name.lower() == args.lower():
                reg.default = e.name
                return CommandResult("success", f"default engine: {e.name}")
        return CommandResult("error", f"unknown engine '{args}'")

    def _bench(self, args: str, _cid: str) -> CommandResult:
        max_convs = int(args) if args.isdigit() else 2
        stamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = self.st.runs_root / f"protocol_{stamp}"
        import subprocess
        import sys as _sys
        from harness.app import REPO_ROOT

        run_dir.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(
            [_sys.executable, "-m", "experiments.generate_data",
             "--mock", "--protocol", "--max-convs", str(max_convs),
             "--output", str(run_dir)],
            cwd=str(REPO_ROOT),
            stdout=open(run_dir / "run_stdout.log", "ab"),
            stderr=subprocess.STDOUT,
        )
        return CommandResult(
            "success",
            f"mock benchmark launched ({max_convs} convs) -> {run_dir.name}; "
            f"report at /view/{run_dir.name} when done")

    def _status(self, _args: str, _cid: str) -> CommandResult:
        s = self.models.status()
        agent = "running" if self.agent.runtime_running else "not started"
        provider = self.st.registry.default or "(first)"
        engine = self.st.engines.default or "(first)"
        return CommandResult(
            "success",
            f"server: {'up' if s['running'] else 'down'}"
            f"{' / healthy' if s.get('healthy') else ''} · "
            f"model: {s.get('model') or '(none)'} · "
            f"agent runtime: {agent} · provider: {provider} · engine: {engine}")
