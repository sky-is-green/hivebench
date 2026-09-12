"""Resilience tests (E1): malformed responses, unreachable backend, auditor."""

import json

import pytest

from backend.openai_compat import OpenAICompatBackend
from auditor.auditor import Auditor, TurnRecord


class _Resp:
    ok = True

    def __init__(self, payload, ok=True):
        self._payload = payload
        self.ok = ok

    def json(self):
        return self._payload

    def raise_for_status(self):
        if not self.ok:
            raise RuntimeError("HTTP 500")


class MalformedTransport:
    def post(self, url, json=None, headers=None, timeout=None):
        return _Resp({})  # no "choices"

    def get(self, url, headers=None, timeout=None):
        return _Resp({"data": []})


class DownTransport:
    def get(self, url, headers=None, timeout=None):
        return _Resp({"data": []}, ok=False)


def test_backend_malformed_response_raises():
    backend = OpenAICompatBackend(base_url="localhost", transport=MalformedTransport())
    with pytest.raises(Exception):
        backend.generate("ctx", "q")


def test_backend_health_false_when_down():
    backend = OpenAICompatBackend(base_url="localhost", transport=DownTransport())
    assert backend.health() is False


def test_auditor_malformed_json_raises():
    auditor = Auditor(generate_fn=lambda p: "not json")
    with pytest.raises(Exception):
        auditor.evaluate_turn(TurnRecord(1, "c", "q", "r"))


def test_auditor_valid_json_ok():
    auditor = Auditor(
        generate_fn=lambda p: json.dumps({"sufficient": True, "used_pieces": [], "missing": [], "score": 4})
    )
    label = auditor.evaluate_turn(TurnRecord(1, "c", "q", "r"))
    assert label.context_sufficient is True