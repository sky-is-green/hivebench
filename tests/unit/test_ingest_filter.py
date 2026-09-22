"""Ingest-time harness-boilerplate filter: system-reminder control text the
upstream harness injects around tool calls must never become persistent
chunks (it would otherwise score highly at recall and pollute the LLM's
context with stale instructions)."""

from cortex.config import SplinterConfig
from cortex.e2e import FakeUltraSmall
from cortex.splinter import Splinter
from retention.hygiene import (
    DEFAULT_INGEST_BLOCK_PREFIXES,
    PREFIX_WINDOW,
    is_boilerplate_line,
    strip_boilerplate,
)
from retention.store import ContextStore


def _hive(**config_kwargs):
    cfg = SplinterConfig(**config_kwargs)
    return Splinter(config=cfg, ultra=FakeUltraSmall(), backend=None)


# ------------------------------------------------ blocklisted strings alone
def test_each_blocklisted_string_alone_stores_nothing():
    assert len(DEFAULT_INGEST_BLOCK_PREFIXES) == 2
    for blocked in DEFAULT_INGEST_BLOCK_PREFIXES:
        store = ContextStore()
        assert store.add_chunk(1, blocked) is None
        assert store.count() == 0
        assert store.all_chunks() == []


def test_truncated_variant_still_matches_on_prefix_head():
    # the store truncates long chunks, so matching uses only the head
    for blocked in DEFAULT_INGEST_BLOCK_PREFIXES:
        head = blocked[:PREFIX_WINDOW]
        assert len(head) == PREFIX_WINDOW
        store = ContextStore()
        assert store.add_chunk(1, head + " ...extra tail that differs...") is None
        assert store.count() == 0


def test_boilerplate_with_leading_whitespace_still_blocked():
    store = ContextStore()
    assert store.add_chunk(1, "   " + DEFAULT_INGEST_BLOCK_PREFIXES[0]) is None
    assert store.count() == 0


# ------------------------------------------------ normal + mixed content
def test_normal_content_stored_unchanged():
    prose = "We decided on JWT auth with 15-minute refresh tokens."
    store = ContextStore()
    cid = store.add_chunk(1, prose)
    assert cid is not None
    assert store.count() == 1
    assert store.all_chunks()[0].content == prose


def test_mixed_turn_stores_only_non_boilerplate_portion():
    store = ContextStore()
    cid = store.add_chunk(
        1, "Deploy tokens rotate every 90 days.\n"
        + DEFAULT_INGEST_BLOCK_PREFIXES[1]
        + "\nReminder set for the team.",
    )
    assert cid is not None
    assert store.count() == 1
    content = store.all_chunks()[0].content
    assert "90 days" in content and "Reminder set" in content
    assert DEFAULT_INGEST_BLOCK_PREFIXES[1][:PREFIX_WINDOW] not in content


def test_filter_helpers_agree():
    assert is_boilerplate_line(DEFAULT_INGEST_BLOCK_PREFIXES[0])
    assert not is_boilerplate_line("Which tokens do I use for auth expiry?")
    assert strip_boilerplate("plain question") == "plain question"
    assert strip_boilerplate(DEFAULT_INGEST_BLOCK_PREFIXES[0]) == ""


# ------------------------------------------------ config override
def test_config_default_blocklist_and_override():
    cfg = SplinterConfig()
    assert list(cfg.ingest_block_prefixes) == list(DEFAULT_INGEST_BLOCK_PREFIXES)

    custom = ["Custom harness preamble: ignore everything below."]
    cfg2 = SplinterConfig.from_dict({"ingest_block_prefixes": custom})
    assert cfg2.ingest_block_prefixes == custom
    # unknown keys are still dropped (existing Gatekeeper contract)
    cfg3 = SplinterConfig.from_dict(
        {"ingest_block_prefixes": custom, "not_a_real_field": 1}
    )
    assert cfg3.ingest_block_prefixes == custom

    # the store inherits the conversation config's blocklist ...
    assert _hive().store.ingest_block_prefixes == list(
        DEFAULT_INGEST_BLOCK_PREFIXES
    )
    hive = _hive(ingest_block_prefixes=custom)
    assert hive.store.ingest_block_prefixes == custom
    # ... and enforces it: custom prefix blocked, default prefix passes
    assert hive.store.add_chunk(1, custom[0] + " more words") is None
    assert hive.store.add_chunk(1, DEFAULT_INGEST_BLOCK_PREFIXES[0]) is not None

    # empty blocklist disables the filter entirely
    open_store = ContextStore(ingest_block_prefixes=[])
    assert open_store.add_chunk(1, DEFAULT_INGEST_BLOCK_PREFIXES[0]) is not None


def test_process_turn_never_stores_boilerplate_query():
    hive = _hive()
    hive.process_turn(DEFAULT_INGEST_BLOCK_PREFIXES[0])
    for chunk in hive.store.all_chunks():
        assert not is_boilerplate_line(chunk.content)
    assert not any(
        DEFAULT_INGEST_BLOCK_PREFIXES[0][:PREFIX_WINDOW] in c.content
        for c in hive.store.all_chunks()
    )


def test_checkpoint_roundtrip_preserves_blocklist():
    store = ContextStore(ingest_block_prefixes=["Custom prefix here"])
    store.add_chunk(1, "JWT refresh is 3600s")
    restored = ContextStore.from_dict(store.to_dict())
    assert restored.ingest_block_prefixes == ["Custom prefix here"]
    assert restored.add_chunk(2, "Custom prefix here plus tail") is None
    # pre-filter payloads (no key) fall back to the defaults
    legacy = ContextStore.from_dict({"max_chunks": None, "chunks": [],
                                     "turn_index": {}})
    assert legacy.ingest_block_prefixes == list(DEFAULT_INGEST_BLOCK_PREFIXES)
