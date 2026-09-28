"""dsh agent bridge — the DeepSeek Harness agent loop behind the console.

One persistent SDK runtime (``deepseek_harness.DeepSeekHarness``, JSON-RPC
stdio subprocess from the pinned fork's node carrier) serves agent sessions
keyed by conversation id. Each message runs the REAL agent loop: tools
(bash, fs, code-runtime, web, subagents), multi-step turns, the durable dsh
session log — with the model pointed at whatever the Studio currently has
loaded (provider ``local`` → managed llama-server).

The runtime is rebuilt automatically when the target (base_url, api_key,
model) changes, so reloading a different model in the Server tab is picked
up on the next agent message. Requires ``DSH_RUNTIME_MODE=node`` (the dev
carrier staged from the fork; Windows has no exe carrier).
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from itertools import count as _count
from pathlib import Path
from typing import Callable, Optional

from harness import agent_events

# The vendored dsh Python SDK ships inside this repo (vendor/deepseek_harness)
# so the sidecar runs from a clean checkout without an editable install of the
# fork's SDK. Inserted before the import below.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor"))

from deepseek_harness import DeepSeekHarness  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


def _shape_notification(notification) -> Optional[dict]:
    """Map an SDK notification to a compact UI activity event."""
    method = getattr(notification, "method", "")
    payload = getattr(notification, "payload", None) or {}
    if method != "session.event":
        return None
    event = payload.get("event") if isinstance(payload, dict) else None
    if not isinstance(event, dict):
        return None
    kind = event.get("type", "")
    data = event.get("data") or {}
    if kind == "assistant/message":
        blocks = data.get("message", {}).get("content") or []
        text = "".join(b.get("text", "") for b in blocks
                       if isinstance(b, dict) and b.get("type") == "text")
        if text.strip():
            return {"type": "assistant", "text": text}
        return None
    if kind.startswith("tool/"):
        name = data.get("name") or data.get("tool") or kind
        return {"type": "tool", "tool": str(name), "phase": kind.split("/", 1)[1]}
    if kind in ("turn/start", "step/start", "turn/end", "step/end"):
        return {"type": "lifecycle", "event": kind}
    return None


def _content_text(blocks) -> str:
    """Text of an OpenAI-style content-block list (dicts or SDK objects)."""
    texts: list[str] = []
    for block in blocks or []:
        text = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
        if isinstance(text, str) and text:
            texts.append(text)
    return "".join(texts)


def _json_or_text(value):
    """A tool-call argument blob as parsed JSON when it parses, else raw text."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value) if value.strip() else {}
    except json.JSONDecodeError:
        return value


class _ModelSpan:
    """One live child-session dispatch awaiting its §B3 ``model`` span."""

    __slots__ = ("model_id", "parent", "role", "model", "task", "started",
                 "output", "pending", "calls")

    def __init__(self, model_id: str, parent: str, started: float) -> None:
        self.model_id = model_id
        self.parent = parent
        self.role: Optional[str] = None
        self.model: Optional[str] = None
        self.task: Optional[str] = None
        self.started = started
        self.output = ""
        self.pending: list[dict] = []
        #: ``callId -> tool name`` for a ``tool/result``, whose own payload
        #: carries only the call id (in the message's source).
        self.calls: dict[str, str] = {}


class ActivityShaper:
    """Per-turn activity shaper: face events + §B3 model spans (T48).

    dsh runs the delegation, so the notifications carry the attribution:
    ``subagent.started``/``finished`` delimit a child session, and the child's
    ``request/context`` (or ``request/header.config``) names its provider route
    (``tier-worker``) and model.  That is the structural parent id for the
    §B3 ``model`` start/end pair — never inferred from timing — and the child's
    own ``tool/*`` events are forwarded parented to that model id so the cards
    nest (ADR-L6).
    """

    def __init__(self, session_id: str, on_event: Callable[[dict], None]) -> None:
        self._root = session_id
        self._on_event = on_event
        self._spans: dict[str, _ModelSpan] = {}
        self._seq = _count(1)

    def __call__(self, notification) -> None:
        method = getattr(notification, "method", "")
        payload = getattr(notification, "payload", None) or {}
        if not isinstance(payload, dict):
            return
        try:
            if method == "subagent.started":
                self._started(payload)
            elif method == "subagent.finished":
                self._finished(payload)
            elif method == "session.event":
                self._session_event(payload, notification)
        except Exception:  # noqa: BLE001 - a consumer queue must not kill the run
            pass

    # -- emission -----------------------------------------------------------
    def _emit(self, event: dict) -> None:
        try:
            self._on_event(event)
        except Exception:  # noqa: BLE001
            pass

    def _emit_child(self, span: _ModelSpan, event: dict) -> None:
        event = dict(event)
        event["tier"] = span.role or ""
        self._emit(agent_events.with_parent(event, span.model_id))

    def _model_start(self, span: _ModelSpan) -> None:
        self._emit(agent_events.model_event(
            tier=span.role or "unknown", model=span.model or "",
            phase=agent_events.PHASE_START, id=span.model_id, task=span.task,
        ))
        for event in span.pending:
            self._emit_child(span, event)
        span.pending = []

    def _child_tool(self, span: _ModelSpan, event: dict) -> None:
        if span.role is None:
            span.pending.append(event)  # the start event carries the parent id
        else:
            self._emit_child(span, event)

    # -- notifications ------------------------------------------------------
    def _started(self, payload: dict) -> None:
        child = payload.get("childSessionId")
        parent = payload.get("parentSessionId")
        if not isinstance(child, str) or not child:
            return
        if parent != self._root and parent not in self._spans:
            return  # another tree's child (or a nested one we do not own)
        self._spans[child] = _ModelSpan(
            agent_events.new_model_id(next(self._seq)),
            str(parent or self._root),
            time.perf_counter(),
        )

    def _finished(self, payload: dict) -> None:
        child = payload.get("childSessionId")
        span = self._spans.pop(child, None) if isinstance(child, str) else None
        if span is None:
            return
        if span.role is None:
            # The child finished without a request event we understood; keep
            # the pair (an unknown-tier card) rather than losing the dispatch.
            span.role = "unknown"
            span.model = span.model or ""
            self._model_start(span)
        output = _content_text(payload.get("lastAssistantMessage")) or span.output
        self._emit(agent_events.model_event(
            tier=span.role, model=span.model or "", phase=agent_events.PHASE_END,
            id=span.model_id, task=span.task,
            duration_ms=int((time.perf_counter() - span.started) * 1000),
            output=output or None,
        ))

    def _session_event(self, payload: dict, notification) -> None:
        session_id = payload.get("sessionId")
        event = payload.get("event")
        if not isinstance(event, dict):
            return
        if session_id is None or session_id == self._root:
            # No sessionId is the legacy/fake shape: shape it as a face event,
            # exactly as before delegation existed.  Real notifications always
            # carry the id.
            shaped = _shape_notification(notification)
            if shaped is not None:
                self._emit(shaped)
            return
        span = self._spans.get(session_id) if isinstance(session_id, str) else None
        if span is None:
            return

        kind = event.get("type", "")
        data = event.get("data") or {}
        if kind in ("request/context", "request/header"):
            if kind == "request/context":
                provider, model = data.get("provider"), data.get("model")
            else:
                config = (data.get("header") or {}).get("config") or {}
                provider, model = config.get("provider"), config.get("model")
            self._resolve(span, provider, model)
        elif kind == "user/message":
            if span.task is None:
                # The event data *is* the user message; tolerate a wrapped one.
                message = data.get("message") or data
                text = _content_text(message.get("content") if isinstance(message, dict) else None)
                span.task = text[:400] or None
        elif kind == "assistant/message":
            text = _content_text((data.get("message") or {}).get("content"))
            if text:
                span.output = text
        elif kind == "tool/call":
            name = str(data.get("name") or "tool")
            call_id = str(data.get("callId") or "")
            if call_id:
                span.calls[call_id] = name
            self._child_tool(span, {
                "type": agent_events.EVENT_TOOL,
                "tool": name,
                "phase": "call",
                "args": _json_or_text(data.get("arguments")),
            })
        elif kind == "tool/result":
            message = data.get("message") or {}
            source = message.get("source") if isinstance(message, dict) else None
            call_id = str((source or {}).get("callId") or "") if isinstance(source, dict) else ""
            name = span.calls.pop(call_id) if call_id in span.calls else None
            result = _content_text(message.get("content")) if isinstance(message, dict) else ""
            self._child_tool(span, {
                "type": agent_events.EVENT_TOOL,
                "tool": name or "tool",
                "phase": "result",
                "result": result,
            })

    def _resolve(self, span: _ModelSpan, provider, model) -> None:
        if span.role is not None:
            return
        route = str(provider or "").strip()
        role = route[len("tier-"):] if route.startswith("tier-") else route
        span.role = role or "unknown"
        span.model = str(model or "").strip()
        self._model_start(span)


def _windows_bash_path() -> Optional[str]:
    """Git for Windows ships bash but rarely puts it on PATH."""
    for candidate in (
        r"C:\Program Files\Git\bin",
        r"C:\Program Files\Git\usr\bin",
        r"C:\Program Files (x86)\Git\bin",
    ):
        if Path(candidate, "bash.exe").is_file():
            return candidate
    return None


def _runtime_env() -> dict:
    """Extra env for the dsh runtime subprocess."""
    env: dict[str, str] = {}
    if os.name == "nt":
        git_bin = _windows_bash_path()
        if git_bin:
            # the bash tool spawns `bash`; Git for Windows' bash is the
            # standard local shell available on this machine
            env["PATH"] = git_bin + os.pathsep + os.environ.get("PATH", "")
    return env


class DshAgentService:
    """Persistent dsh runtime + per-conversation agent sessions.

    The SDK transport has no cancel method (initialize/session/prompt/
    shutdown only), so cancelling a run terminates the runtime subprocess:
    the blocked ``run()`` raises, the SSE reports ``cancelled``, and the
    conversation survives — the durable JSONL session is continued by the
    rebuilt runtime on the next message (same session id + session_root).
    """

    def __init__(self, default_cwd: Path, session_root: Path, *,
                 runtime_config: Optional[Callable[[], Optional[str]]] = None) -> None:
        self.default_cwd = Path(default_cwd)
        self.session_root = Path(session_root)
        #: Called before each runtime build; returns the profile config path for
        #: the currently applied stack (``None`` = the runtime's own default).
        self._runtime_config = runtime_config
        self._cordis_path: Optional[str] = None
        self._lock = threading.Lock()
        self._harness = None
        self._target: Optional[tuple] = None
        self._run_lock = threading.Lock()  # one agent turn at a time (v1)
        self._cancel_requested = threading.Event()
        self._inflight = False
        self._generation = 0  # bumped on every runtime rebuild
        self._generation_initialized = False
        self._handoff_done: set[str] = set()

    def _init_generation(self) -> None:
        """Seed the generation counter from durable session dirs so a fresh
        sidecar never collides with logs written by a previous life."""
        if self._generation_initialized or not self.session_root.is_dir():
            self._generation_initialized = True
            return
        max_gen = -1
        for d in self.session_root.glob("*/agent-*"):
            m = re.match(r"agent-.*-g(\d+)$", d.name)
            gen = int(m.group(1)) if m else 0  # unsuffixed = generation 0
            max_gen = max(max_gen, gen)
        self._generation = max_gen + 1 if max_gen >= 0 else 0
        self._generation_initialized = True

    def _session_id(self, conversation_id: str) -> str:
        """Per-generation session id: the runtime rejects a persisted-id
        collision across restarts, so each runtime generation gets its own
        session; continuity comes from the transcript handoff instead."""
        return f"agent-{conversation_id}-g{self._generation}"

    def _prior_transcript(self, conversation_id: str,
                          max_messages: int = 10) -> list[tuple[str, str]]:
        """User/assistant pairs from the previous generation's durable log."""
        import zstandard

        prior_gen = self._generation - 1
        candidates = [self.session_root.glob(
            f"*/agent-{conversation_id}-g{prior_gen}/session.jsonl.zstd")]
        if prior_gen == 0:
            # legacy unsuffixed sessions from before generations existed
            candidates.append(self.session_root.glob(
                f"*/agent-{conversation_id}/session.jsonl.zstd"))
        pairs: list[tuple[str, str]] = []
        for glob in candidates:
            for path in glob:
                dctx = zstandard.ZstdDecompressor()
                with open(path, "rb") as fh:
                    text = dctx.stream_reader(fh, read_across_frames=True) \
                        .read().decode("utf-8", errors="replace")
                pending_user: Optional[str] = None
                for line in text.splitlines():
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if record.get("type") != "assistant/message":
                        continue
                    data = record.get("data") or {}
                    message = data.get("message") or {}
                    blocks = message.get("content") or []
                    reply = "".join(b.get("text", "") for b in blocks
                                    if isinstance(b, dict)
                                    and b.get("type") == "text").strip()
                    if not reply:
                        continue
                    pairs.append((pending_user or "(continued)", reply[:400]))
                    pending_user = None
        return pairs[-max_messages:]

    def _handoff_prefix(self, conversation_id: str) -> str:
        if self._generation == 0 or conversation_id in self._handoff_done:
            return ""
        pairs = self._prior_transcript(conversation_id)
        self._handoff_done.add(conversation_id)
        if not pairs:
            return ""
        lines = ["[Context handoff — your previous session on this workspace, "
                 "for continuity:]"]
        for user, reply in pairs:
            lines.append(f"User: {user[:400]}")
            lines.append(f"You: {reply[:400]}")
        lines.append("[End handoff — continue naturally.]")
        return "\n".join(lines) + "\n\n"

    @property
    def runtime_running(self) -> bool:
        return self._harness is not None

    @property
    def permission_policy(self) -> str:
        # the bundled jsonrpc-agent composition mounts no approval/permission
        # plugin: tools run with the runtime process's full access
        return "full-access (auto-approve; the runtime's own permissions apply)"

    def _ensure(self, base_url: str, api_key: str, model: str):
        cordis = self._runtime_cordis()
        key = (base_url, api_key, model, cordis)
        with self._lock:
            if self._harness is not None and self._target == key:
                return self._harness
            if self._harness is not None:
                try:
                    self._harness.close()
                except Exception:  # noqa: BLE001 - replacing a dead runtime
                    pass
                # new runtime generation: fresh session ids (the runtime
                # rejects persisted-id collisions across restarts)
                self._generation += 1
                self._handoff_done.clear()
            else:
                self._init_generation()
            # the dev node carrier is opt-in by design; the sidecar always
            # wants it (Windows has no exe carrier)
            os.environ["DSH_RUNTIME_MODE"] = "node"
            os.environ.setdefault("DSH_HOME",
                                  str(REPO_ROOT / ".dsh-home"))
            self.session_root.mkdir(parents=True, exist_ok=True)
            self._harness = DeepSeekHarness(
                provider="deepseek-official",
                base_url=base_url,
                api_key=api_key or "lm-studio",
                model=model,
                cwd=str(self.default_cwd),
                session_root=str(self.session_root),
                cordis=cordis,
                env=_runtime_env(),
            )
            self._target = key
            return self._harness

    def _runtime_cordis(self) -> Optional[str]:
        """The applied-stack profile path, refreshed before each runtime build.

        A broken generator must never block chat: on error the previously
        resolved path (or ``None``) is reused.
        """
        if self._runtime_config is None:
            return None
        try:
            path = self._runtime_config()
        except Exception:  # noqa: BLE001
            return self._cordis_path
        self._cordis_path = str(path) if path else None
        return self._cordis_path

    def cancel(self) -> dict:
        """Request cancellation of the in-flight run.

        With no graceful cancel in the SDK protocol this kills the runtime;
        the blocked run() raises, the worker reports 'cancelled', and the
        durable session continues on the next message.
        """
        if not self._inflight:
            return {"ok": False, "note": "nothing in flight"}
        self._cancel_requested.set()
        with self._lock:
            harness = self._harness
        if harness is None:
            self._cancel_requested.clear()
            return {"ok": False, "note": "no runtime running"}
        # close() terminates the subprocess; run() unblocks with an error
        threading.Thread(target=self.close, daemon=True).start()
        return {"ok": True, "note": "runtime terminated; run will settle as cancelled"}

    def close(self) -> None:
        with self._lock:
            if self._harness is not None:
                try:
                    self._harness.close()
                finally:
                    self._harness = None
                    self._target = None
        self._cancel_requested.clear()

    def run_turn(
        self,
        conversation_id: str,
        message: str,
        base_url: str,
        api_key: str,
        model: str,
        on_event: Optional[Callable[[dict], None]] = None,
    ) -> dict:
        """Run one agent turn (blocking); stream activity via on_event."""
        harness = self._ensure(base_url, api_key, model)
        session_id = self._session_id(conversation_id)
        handoff = self._handoff_prefix(conversation_id)
        message_to_send = handoff + message if handoff else message

        shaper = ActivityShaper(session_id, on_event) if on_event is not None else None

        def on_notification(notification) -> None:
            if shaper is not None:
                shaper(notification)

        with self._run_lock:
            self._inflight = True
            try:
                result = harness.run(
                    message_to_send,
                    session_id=session_id,
                    on_notification=on_notification,
                )
            except Exception as exc:  # noqa: BLE001 - cancel surfaces here
                if self._cancel_requested.is_set():
                    self._cancel_requested.clear()
                    return {
                        "final": "(cancelled by user)",
                        "finish_reason": "cancelled",
                        "session_id": session_id,
                        "events": 0,
                    }
                raise
            finally:
                self._inflight = False
        return {
            "final": result.final_response,
            "finish_reason": result.finish_reason,
            "session_id": result.session_id,
            "events": len(result.events or []),
        }


def summarize_activity(events: list[dict]) -> str:
    """One-line human summary for a batch of shaped activity events."""
    tools = [e.get("tool") for e in events if e.get("type") == "tool"]
    if tools:
        return "tools: " + ", ".join(tools)
    return json.dumps(events)[:120]
