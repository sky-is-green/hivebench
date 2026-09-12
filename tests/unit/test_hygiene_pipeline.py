"""The write-side hygiene pipeline (retention.hygiene.prepare_for_storage):
boilerplate strip -> secret/blob sanitization -> length cap, as one composite
entry point shared by ContextStore.add_chunk and the host payload-echo path.
The invariant under test: both sides of echo detection fingerprint the SAME
normalized form."""

from retention.store import ContextStore
from retention.hygiene import (
    DEFAULT_INGEST_BLOCK_PREFIXES,
    content_fingerprint,
    prepare_for_storage,
)

BOILER = DEFAULT_INGEST_BLOCK_PREFIXES[0]


def test_blank_and_fully_boilerplate_return_none():
    assert prepare_for_storage("") is None
    assert prepare_for_storage("   \n  ") is None
    assert prepare_for_storage(BOILER) is None
    assert prepare_for_storage(BOILER + "\n" + DEFAULT_INGEST_BLOCK_PREFIXES[1]) is None


def test_mixed_turn_keeps_only_remainder():
    text = BOILER + "\nReal question about the GPU."
    assert prepare_for_storage(text) == "Real question about the GPU."


def test_secrets_redacted_after_strip():
    text = BOILER + "\nkey: sk-abcdef1234567890abcdef ok"
    out = prepare_for_storage(text)
    assert "sk-abcdef1234567890abcdef" not in out
    assert "[redacted-secret]" in out


def test_sanitize_false_strips_boilerplate_but_keeps_secrets():
    text = BOILER + "\nkey: sk-abcdef1234567890abcdef"
    out = prepare_for_storage(text, sanitize=False)
    assert "sk-abcdef1234567890abcdef" in out
    assert BOILER[:40] not in (out or "")


def test_store_and_payload_paths_fingerprint_identically():
    # The whole point of the composite: add_chunk and the payload-echo path
    # must produce the same fingerprint for the same logical content.
    store = ContextStore()
    text = BOILER + "\nWhat is the deal with RX 7900 XT power draw?"
    cid = store.add_chunk(1, text)
    assert cid is not None
    prepared = prepare_for_storage(
        text, store.max_chunk_chars, store.ingest_block_prefixes
    )
    assert content_fingerprint(prepared) == store.chunks[cid].fingerprint


def test_idempotent_reingest_dedups_to_one_chunk():
    store = ContextStore()
    text = BOILER + "\nSame message twice."
    first = store.add_chunk(1, text)
    second = store.add_chunk(2, text)
    assert first == second
    assert store.count() == 1
