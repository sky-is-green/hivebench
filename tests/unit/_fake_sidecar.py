"""In-memory fake sidecar for MCP-path unit tests (S3).

Emulates just enough of the sidecar to exercise ``McpClient`` /
``RawTurnClient`` / ``run_mcp_battery`` without a network or an encoder: the
``/v1/mcp`` JSON-RPC endpoint and the raw ``observe`` / ``curate`` / ``turn`` /
``reset`` REST endpoints, with per-conversation verbatim stores.
"""

from __future__ import annotations

import json
from urllib.parse import urlparse

MCP_PATH = "/v1/mcp"
TURN_PATH = "/v1/strata/turn"
CURATE_PATH = "/v1/strata/curate"
OBSERVE_PATH = "/v1/strata/observe"
RESET_PATH = "/v1/strata/reset"


class FakeResponse:
    def __init__(self, payload, status: int = 200, text: str | None = None) -> None:
        self._payload = payload
        self.status_code = status
        self.text = text if text is not None else json.dumps(payload)

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON body")
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSidecar:
    """A requests-compatible POST target with sidecar semantics."""

    def __init__(self) -> None:
        self.store: dict[str, list[str]] = {}
        self.requests: list[tuple[str, dict, dict]] = []
        self.last_search: dict = {}
        self.rpc_error: dict | None = None
        self.http_status: int | None = None

    # -- transport ---------------------------------------------------------
    def post(self, url, json=None, headers=None, timeout=None):
        self.requests.append((url, json or {}, headers or {}))
        if self.http_status:
            return FakeResponse(None, self.http_status, text="transport error")
        path = urlparse(url).path
        if self.rpc_error is not None and path == MCP_PATH:
            return FakeResponse({"jsonrpc": "2.0", "id": 1, "error": self.rpc_error})
        if path == MCP_PATH:
            return FakeResponse(self._mcp(json or {}))
        if path == OBSERVE_PATH:
            cid = (json or {}).get("conversation_id", "")
            reply = ((json or {}).get("reply") or "").strip()
            if reply:
                self.store.setdefault(cid, []).append(reply)
            return FakeResponse({"ok": True, "stored": bool(reply)})
        if path == CURATE_PATH:
            return FakeResponse(self._assembled(json or {}))
        if path == TURN_PATH:
            return FakeResponse({**self._assembled(json or {}), "reply": "fake reply"})
        if path == RESET_PATH:
            self.store.pop((json or {}).get("conversation_id", ""), None)
            return FakeResponse({"ok": True})
        return FakeResponse(None, 404, text="not found")

    # -- MCP ---------------------------------------------------------------
    def _mcp(self, body: dict) -> dict:
        mid = body.get("id") if isinstance(body, dict) else None
        if not isinstance(body, dict) or not body.get("method"):
            return {"jsonrpc": "2.0", "id": mid,
                    "error": {"code": -32600, "message": "invalid request"}}
        method = body["method"]
        if method == "initialize":
            return {"jsonrpc": "2.0", "id": mid, "result": {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "strata-memory", "version": "test"},
            }}
        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": mid, "result": {"tools": [
                {"name": "strata_search"}, {"name": "strata_remember"},
            ]}}
        if method != "tools/call":
            return {"jsonrpc": "2.0", "id": mid,
                    "error": {"code": -32601, "message": f"method not found: {method}"}}

        params = body.get("params") or {}
        name = params.get("name")
        args = params.get("arguments") or {}
        cid = args.get("conversation_id")
        if not isinstance(cid, str) or not cid.strip():
            return {"jsonrpc": "2.0", "id": mid,
                    "error": {"code": -32602,
                              "message": "conversation_id is required"}}
        if name == "strata_remember":
            text = (args.get("text") or "").strip()
            if not text:
                return {"jsonrpc": "2.0", "id": mid,
                        "error": {"code": -32602, "message": "text required"}}
            self.store.setdefault(cid, []).append(text)
            payload = {"ok": True, "stored": True, "chunk_id": f"c{len(self.store[cid])}",
                       "turn": len(self.store[cid]), "store_chunks": len(self.store[cid])}
        elif name == "strata_search":
            query = (args.get("query") or "").strip()
            if not query:
                return {"jsonrpc": "2.0", "id": mid,
                        "error": {"code": -32602, "message": "query required"}}
            self.last_search = {"conversation_id": cid, "query": query,
                                "top_k": args.get("top_k", 5)}
            payload = self._search_payload(cid, query, args.get("top_k", 5))
        else:
            return {"jsonrpc": "2.0", "id": mid,
                    "error": {"code": -32601, "message": f"unknown tool: {name}"}}
        return {"jsonrpc": "2.0", "id": mid, "result": {
            "content": [{"type": "text", "text": json.dumps(payload)}]}}

    # -- raw ---------------------------------------------------------------
    def _search_payload(self, cid: str, query: str, top_k: int) -> dict:
        assembled = " ".join(self.store.get(cid, []))
        chunks = [{"id": f"c{i}", "score": 1.0, "preview": self.store[cid][i][:200]}
                  for i in range(min(len(self.store.get(cid, [])), int(top_k)))]
        return {"query": query, "turn": 0, "assembled_content": assembled,
                "token_count": len(assembled.split()), "budget": 4096,
                "mode": "no_backend", "chunks": chunks}

    def _assembled(self, body: dict) -> dict:
        cid = body.get("conversation_id", "")
        query = body.get("query", "")
        assembled = " ".join(self.store.get(cid, []))
        return {"conversation_id": cid, "turn": 0, "assembled_content": assembled,
                "token_count": len(assembled.split()), "budget": 4096,
                "mode": "no_backend", "pes": 0.0, "degradation_level": 0}
