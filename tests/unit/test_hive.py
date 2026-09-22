"""Unit tests for cortex.splinter (unified orchestrator)."""

from backend.lmstudio import LMStudioBackend
from cortex.config import SplinterConfig
from cortex.e2e import FakeUltraSmall, MockTransport
from cortex.splinter import Splinter
from logs.event_logger import EventLogger
from sieve.medium import MediumDrone


def _hive(backend=None, logger=None, config=None, pinned_prefix=""):
    return Splinter(
        config=config or SplinterConfig(),
        ultra=FakeUltraSmall(),
        medium=MediumDrone(score_pair_fn=lambda q, c: 0.5),
        backend=backend, logger=logger, pinned_prefix=pinned_prefix,
    )


def test_process_turn_basic():
    h = _hive()
    r = h.process_turn("how does authentication work")
    assert r.turn == 1
    assert r.mode in ("splinter", "no_backend")
    assert r.assembled is not None
    assert 0.0 <= r.pes <= 100.0
    assert h.store.count() >= 1
    assert "scoring_ms" in r.timings


def test_enable_medium_wires_real_medium_drone():
    """enable_medium=True must construct the configured MediumDrone, not the
    lightweight placeholder (score_pair_fn constant 0.5)."""
    config = SplinterConfig(enable_medium=True, medium_model="microsoft/graphcodebert-base")
    h = Splinter(config=config, ultra=FakeUltraSmall(), backend=None)
    # not the placeholder: a real MediumDrone with a real model_name and no
    # injected score_pair_fn
    assert not hasattr(h.medium, "_score_pair_fn") or h.medium._score_pair_fn is None
    assert h.medium.model_name == "microsoft/graphcodebert-base"
    assert h.medium.mode == "cross"


def test_enable_medium_false_uses_placeholder():
    config = SplinterConfig(enable_medium=False)
    h = Splinter(config=config, ultra=FakeUltraSmall(), backend=None)
    assert h.medium._score_pair_fn is not None  # placeholder, cheap


def test_store_grows_across_turns():
    backend = LMStudioBackend(base_url="localhost", model="m", transport=MockTransport())
    h = _hive(backend=backend)
    h.process_turn("q1")
    n1 = h.store.count()
    h.process_turn("q2")
    assert h.store.count() == n1 + 2  # query + reply


def test_hedge_replies_not_stored_as_chunks():
    """Refusal/hedge replies ('no information regarding X') must not be stored:
    they would later be retrieved as context, poisoning retrieval. The query
    chunk is still stored so what was asked is recorded."""
    from cortex.e2e import MockTransport

    class HedgeTransport(MockTransport):
        def post(self, url, json=None, headers=None, timeout=None):
            self.last = (url, json)
            return _Resp_hedge({"choices": [{"message": {"content":
                "Based on the provided context, there is no information "
                "regarding rate limits for the REST API."}}]})

    class _Resp_hedge:
        ok = True
        def __init__(self, payload):
            self._payload = payload
        def json(self):
            return self._payload
        def raise_for_status(self):
            pass

    backend = LMStudioBackend(base_url="localhost", model="m", transport=HedgeTransport())
    h = _hive(backend=backend)
    r = h.process_turn("Can we change how rate limits works?")
    assert r.reply  # reply still returned to the user
    contents = [c.content for c in h.store.all_chunks()]
    assert len(contents) == 1  # only the query chunk
    assert contents[0] == "Can we change how rate limits works?"
    assert not any("no information" in c for c in contents)


def test_hedge_filter_can_be_disabled():
    from cortex.e2e import MockTransport

    backend = LMStudioBackend(base_url="localhost", model="m", transport=MockTransport())
    h = _hive(backend=backend, config=SplinterConfig(filter_hedge_replies=False))
    h.process_turn("q1")
    assert h.store.count() == 2  # query + (non-hedge) reply both stored


def test_hedge_contraction_variants_caught():
    """'don't have access' / 'can't provide' refusals must be caught even though
    the markers use 'do not have' / 'cannot' (contraction normalization)."""
    cases = [
        "I don't have access to your specific deployment pipeline or repository.",
        "I don't have access to specific code from a log pipeline.",
        "I can't provide the exact implementation without more context.",
        "I can't show you your actual canary-handling code.",
        "Based on the context provided, I don't have specific information about your pipeline.",
    ]
    for c in cases:
        assert Splinter._is_hedge_reply(c), f"should be hedge: {c!r}"


def test_hedge_lead_anchored_mid_reply_caveat_not_hedge():
    """A factual reply that merely mentions it lacks specifics mid-text must NOT
    be filtered: it carries facts. Refusals announce themselves at the start."""
    factual = (
        "To recommend the best **log levels** for your log pipeline, it's "
        "important to understand what kind of data is being logged. Since I "
        "don't have specific details about your setup, here's a general "
        "framework you can use: DEBUG for development, INFO in production..."
    )
    assert not Splinter._is_hedge_reply(factual)
    # but the same signal at the START is a hedge
    assert Splinter._is_hedge_reply(
        "I don't have specific details about your setup, so I cannot recommend log levels."
    )


def test_hedge_factual_context_openers_not_filtered():
    """'Based on the context provided, here is a recommendation' replies are
    factual (they answer), even though they open with the same boilerplate
    as the live refusals."""
    factual = (
        "Based on the provided context, here is a comprehensive walkthrough "
        "for implementing a consistent error envelope in your REST API."
    )
    assert not Splinter._is_hedge_reply(factual)


def test_no_backend_mode():
    h = _hive()
    r = h.process_turn("q")
    assert r.mode == "no_backend"
    assert r.reply == ""


def test_backend_generates_with_pinned_prefix():
    backend = LMStudioBackend(base_url="localhost", model="m", transport=MockTransport())
    h = _hive(backend=backend, pinned_prefix="PIN")
    r = h.process_turn("how does authentication work", conversation_id="c1")
    assert r.mode == "splinter"
    assert r.reply
    assert r.timings["generation_ms"] > 0
    assert backend.pinned_prefix == "PIN"


def test_fifo_fallback_at_emergency():
    h = _hive()
    h.degradation.current_level = 3
    h.monitor.metrics.queue_depth = 20
    for _ in range(10):
        h.monitor.metrics.record_drone_latency(200)
    h.monitor.metrics.pending_assemblies = 3
    r = h.process_turn("q")
    assert r.mode == "fifo_fallback"
    assert r.assembled is None


def test_logging_correlation_ids(tmp_path):
    logger = EventLogger(log_dir=tmp_path)
    h = _hive(logger=logger)
    h.process_turn("q", conversation_id="conv-1")
    logger.flush()
    logger.close()
    entries = logger.read_entries()
    assert entries
    assert all(e.get("conversation_id") == "conv-1" for e in entries)
    assert all(e.get("run_id") == h.run_id for e in entries)
    assert all(e.get("turn_id") == 1 for e in entries)


def test_latency_breakdown_populated():
    h = _hive()
    r = h.process_turn("q")
    for key in ("remembrance_ms", "scoring_ms", "dedup_ms", "drift_ms", "decay_ms", "select_ms", "assembly_total_ms"):
        assert key in r.timings


class _FailingBackend:
    def generate(self, content, query, sampling_params=None):
        raise RuntimeError("HTTP 400 Bad Request")

    def health(self):
        return True


def test_backend_error_is_captured_not_raised(tmp_path):
    backend = _FailingBackend()
    h = Splinter(
        config=SplinterConfig(), ultra=FakeUltraSmall(),
        medium=MediumDrone(score_pair_fn=lambda q, c: 0.5),
        backend=backend, logger=EventLogger(log_dir=tmp_path), pinned_prefix="PIN",
    )
    r = h.process_turn("q", conversation_id="c1")
    assert r.mode == "error"
    assert r.error is not None
    assert "400" in r.error
    assert r.reply == ""


def test_error_logged_as_event(tmp_path):
    backend = _FailingBackend()
    logger = EventLogger(log_dir=tmp_path)
    h = Splinter(
        config=SplinterConfig(), ultra=FakeUltraSmall(),
        medium=MediumDrone(score_pair_fn=lambda q, c: 0.5),
        backend=backend, logger=logger,
    )
    h.process_turn("q")
    logger.flush()
    logger.close()
    events = [e for e in logger.read_entries() if e["event_type"] == "turn_failed"]
    assert len(events) == 1


class _ReasoningStarvedBackend:
    """Returns empty content because the token budget was eaten by reasoning."""

    def generate(self, content, query, sampling_params=None):
        self.last_usage = {
            "completion_tokens": 128,
            "completion_tokens_details": {"reasoning_tokens": 128},
        }
        return ""

    def health(self):
        return True


def test_empty_reply_reasoning_starved_warns_once(tmp_path, capsys):
    backend = _ReasoningStarvedBackend()
    logger = EventLogger(log_dir=tmp_path)
    h = Splinter(
        config=SplinterConfig(max_tokens=128), ultra=FakeUltraSmall(),
        medium=MediumDrone(score_pair_fn=lambda q, c: 0.5),
        backend=backend, logger=logger,
    )
    r1 = h.process_turn("q1")
    assert r1.reply == ""
    assert r1.mode == "splinter"  # not a crash
    h.process_turn("q2")  # second turn: no duplicate warning
    logger.flush()
    logger.close()
    events = [e for e in logger.read_entries()
              if e["event_type"] == "empty_reply_reasoning_starved"]
    assert len(events) == 1  # warned exactly once
    err = capsys.readouterr().err
    assert "reasoning" in err.lower()
