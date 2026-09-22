"""MCP-path client drivers for hivebench (S3).

The sidecar exposes two ways to reach the same curation engine:

- the **MCP path** — stateless JSON-RPC on ``POST /v1/mcp`` with the
  ``splinter_search`` / ``splinter_remember`` tools (S2's fixed contract), and
- the **raw REST path** — the sidecar's own endpoints
  (``/v1/splinter/turn`` for a full turn, ``/v1/splinter/curate`` /
  ``/v1/splinter/observe`` for external-shell integrators).

``McpClient`` speaks the former (``initialize`` → ``tools/list`` →
``tools/call``); ``RawTurnClient`` speaks the latter. Both take ``base_url``
as a parameter and accept an injectable ``http`` object so the benchmark and
its tests never hardcode a host/port and never require a live sidecar unless
one is actually wanted.

``http`` must be ``requests``-compatible: ``post(url, json=, headers=,
timeout=)`` returning an object with ``raise_for_status()``, ``json()`` and
``status_code``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Optional

import requests

DEFAULT_BASE_URL = "http://127.0.0.1:8765"
MCP_PATH = "/v1/mcp"
TURN_PATH = "/v1/splinter/turn"
CURATE_PATH = "/v1/splinter/curate"
OBSERVE_PATH = "/v1/splinter/observe"
RESET_PATH = "/v1/splinter/reset"

DEFAULT_TIMEOUT = 120.0
DEFAULT_CONVERSATION_ID = "hivebench-mcp"
TOKEN_HEADER = "x-splinter-token"


class SidecarHttpError(RuntimeError):
    """Transport-level failure (connection, non-2xx, non-JSON body)."""

    def __init__(self, status: int, detail: str = "") -> None:
        self.status = status
        self.detail = detail
        super().__init__(f"HTTP {status}: {detail}" if detail else f"HTTP {status}")


@dataclass
class CallResult:
    """Outcome of one sidecar call, with the round-trip latency."""

    ok: bool
    payload: dict
    latency_ms: float
    error: str = ""
    method: str = ""


class _BaseClient:
    """Shared HTTP plumbing: token header, latency, error normalization."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        conversation_id: str = DEFAULT_CONVERSATION_ID,
        timeout: float = DEFAULT_TIMEOUT,
        http=None,
        token: str = "",
    ) -> None:
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.conversation_id = conversation_id
        self.timeout = timeout
        self.token = token or ""
        self._http = http or requests

    def _headers(self) -> dict:
        return {TOKEN_HEADER: self.token} if self.token else {}

    def _post(self, path: str, body: dict) -> tuple[dict, float]:
        started = time.perf_counter()
        resp = self._http.post(
            f"{self.base_url}{path}",
            json=body,
            headers=self._headers(),
            timeout=self.timeout,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        try:
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001 - normalized for callers
            detail = ""
            try:
                detail = (resp.text or "")[:300]
            except Exception:  # noqa: BLE001 - body may be unreadable
                detail = ""
            raise SidecarHttpError(
                getattr(resp, "status_code", 0), detail or str(exc)
            ) from None
        try:
            data = resp.json()
        except ValueError as exc:
            raise SidecarHttpError(
                getattr(resp, "status_code", 0), f"invalid JSON body: {exc}"
            ) from None
        if not isinstance(data, dict):
            raise SidecarHttpError(
                getattr(resp, "status_code", 0), "expected a JSON object"
            )
        return data, latency_ms


class McpClient(_BaseClient):
    """JSON-RPC MCP client for the sidecar's ``POST /v1/mcp`` endpoint."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._next_id = 0

    def _rpc(self, method: str, params: Optional[dict] = None) -> CallResult:
        self._next_id += 1
        body: dict = {"jsonrpc": "2.0", "id": self._next_id, "method": method}
        if params is not None:
            body["params"] = params
        try:
            data, latency_ms = self._post(MCP_PATH, body)
        except SidecarHttpError as exc:
            return CallResult(False, {}, 0.0, str(exc), method)
        if "error" in data and data["error"]:
            err = data["error"]
            return CallResult(
                False, {}, latency_ms, err.get("message", str(err)), method
            )
        result = data.get("result")
        if not isinstance(result, dict):
            result = {}
        return CallResult(True, result, latency_ms, "", method)

    def initialize(self, protocol_version: str = "2025-06-18") -> CallResult:
        return self._rpc(
            "initialize",
            {
                "protocolVersion": protocol_version,
                "capabilities": {},
                "clientInfo": {"name": "hivebench", "version": "1"},
            },
        )

    def list_tools(self) -> CallResult:
        return self._rpc("tools/list", {})

    def call_tool(
        self, name: str, arguments: dict, conversation_id: Optional[str] = None
    ) -> CallResult:
        """Invoke a tool; unwraps the MCP ``content`` block into its payload."""
        args = dict(arguments)
        if conversation_id is not None:
            args["conversation_id"] = conversation_id
        elif "conversation_id" not in args:
            args["conversation_id"] = self.conversation_id
        result = self._rpc("tools/call", {"name": name, "arguments": args})
        if not result.ok:
            return result
        content = result.payload.get("content") or []
        text = content[0].get("text", "") if content else ""
        if result.payload.get("isError"):
            return CallResult(False, {}, result.latency_ms, text, name)
        try:
            payload = json.loads(text) if text else {}
        except ValueError:
            payload = {"raw": text}
        if not isinstance(payload, dict):
            payload = {"raw": payload}
        return CallResult(True, payload, result.latency_ms, "", name)

    def remember(self, text: str, conversation_id: Optional[str] = None) -> CallResult:
        return self.call_tool(
            "splinter_remember", {"text": text}, conversation_id=conversation_id
        )

    def search(
        self, query: str, top_k: int = 5, conversation_id: Optional[str] = None
    ) -> CallResult:
        return self.call_tool(
            "splinter_search",
            {"query": query, "top_k": top_k},
            conversation_id=conversation_id,
        )


class RawTurnClient(_BaseClient):
    """Raw REST driver: full turns plus the external-shell curate/observe pair."""

    def turn(self, query: str, conversation_id: Optional[str] = None) -> CallResult:
        cid = conversation_id or self.conversation_id
        try:
            data, latency_ms = self._post(
                TURN_PATH, {"query": query, "conversation_id": cid}
            )
        except SidecarHttpError as exc:
            return CallResult(False, {}, 0.0, str(exc), "splinter_turn")
        return CallResult(True, data, latency_ms, "", "splinter_turn")

    def curate(self, query: str, conversation_id: Optional[str] = None) -> CallResult:
        cid = conversation_id or self.conversation_id
        try:
            data, latency_ms = self._post(
                CURATE_PATH, {"query": query, "conversation_id": cid}
            )
        except SidecarHttpError as exc:
            return CallResult(False, {}, 0.0, str(exc), "splinter_curate")
        return CallResult(True, data, latency_ms, "", "splinter_curate")

    def observe(self, reply: str, conversation_id: Optional[str] = None) -> CallResult:
        cid = conversation_id or self.conversation_id
        try:
            data, latency_ms = self._post(
                OBSERVE_PATH, {"reply": reply, "conversation_id": cid}
            )
        except SidecarHttpError as exc:
            return CallResult(False, {}, 0.0, str(exc), "splinter_observe")
        return CallResult(True, data, latency_ms, "", "splinter_observe")

    def reset(self, conversation_id: Optional[str] = None) -> CallResult:
        cid = conversation_id or self.conversation_id
        try:
            data, latency_ms = self._post(RESET_PATH, {"conversation_id": cid})
        except SidecarHttpError as exc:
            return CallResult(False, {}, 0.0, str(exc), "splinter_reset")
        return CallResult(True, data, latency_ms, "", "splinter_reset")
