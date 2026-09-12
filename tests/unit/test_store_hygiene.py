"""Store-time hygiene (U2): credentials, base64 blobs, and oversized chunks
must not reach the persistent tiers. sanitize_for_storage runs inside
add_chunk(), so the store, checkpoints, and comb archives all inherit clean
content. Normal prose passes through byte-identical."""

import pytest

from cortex.config import StrataConfig
from retention.store import ContextStore
from retention.hygiene import DEFAULT_MAX_CHUNK_CHARS, sanitize_for_storage


# ---------------------------------------------------------------- patterns
def test_openai_style_key_redacted():
    text = "use this key sk-proj-abcdef1234567890abcd in prod"
    out = sanitize_for_storage(text)
    assert "sk-proj-abcdef1234567890abcd" not in out
    assert "[redacted-secret]" in out
    assert "use this key" in out


def test_github_token_redacted():
    out = sanitize_for_storage("token ghp_0123456789abcdefghijklmnopqrstuv ok")
    assert "ghp_0123456789" not in out
    assert "[redacted-secret]" in out


def test_aws_access_key_redacted():
    out = sanitize_for_storage("AKIAIOSFODNN7EXAMPLE is my key")
    assert "AKIAIOSFODNN7EXAMPLE" not in out


def test_key_value_assignment_redacted_but_label_kept():
    out = sanitize_for_storage('api_key = "super-secret-value-123"')
    assert "super-secret-value-123" not in out
    assert "api_key" in out
    assert "[redacted]" in out


def test_bearer_header_value_redacted():
    out = sanitize_for_storage("Authorization: Bearer abc.def.ghi-jkl_123")
    assert "abc.def.ghi" not in out
    assert "authorization" in out.lower()


def test_normal_prose_untouched():
    prose = "We decided on JWT auth with 15-minute refresh tokens."
    assert sanitize_for_storage(prose) == prose


# ---------------------------------------------------------------- base64/len
def test_base64_blob_collapsed():
    blob = "A" * 400
    out = sanitize_for_storage(f"data payload {blob} end")
    assert blob not in out
    assert "[base64 blob stripped]" in out


def test_length_cap_truncates_with_marker():
    text = ("lorem ipsum dolor sit amet " * 200)[: DEFAULT_MAX_CHUNK_CHARS + 500]
    out = sanitize_for_storage(text)
    assert len(out) == DEFAULT_MAX_CHUNK_CHARS + len("\n…[truncated]")
    assert out.endswith("[truncated]")


# ---------------------------------------------------------------- store path
def test_add_chunk_stores_sanitized_content():
    store = ContextStore()
    store.add_chunk(1, "my key is sk-abcdef1234567890abcd keep secret")
    (chunk,) = store.all_chunks()
    assert "sk-abcdef1234567890abcd" not in chunk.content
    assert "[redacted-secret]" in chunk.content


def test_sanitize_off_keeps_raw():
    store = ContextStore(sanitize=False)
    raw = "key sk-abcdef1234567890abcd"
    store.add_chunk(1, raw)
    assert store.all_chunks()[0].content == raw


def test_fingerprint_groups_on_sanitized_form():
    """Two chunks differing only in a secret value dedup as one."""
    store = ContextStore()
    cid_a = store.add_chunk(1, "key sk-aaaaaaaaaaaaaaaaaaaaaaaa")
    cid_b = store.add_chunk(1, "key sk-bbbbbbbbbbbbbbbbbbbbbbbb")
    assert cid_a == cid_b
    assert store.count() == 1
    assert "[redacted-secret]" in store.all_chunks()[0].content


def test_comb_archives_sanitized_content():
    archived = []

    class FakeComb:
        def put(self, chunk, embedding):
            archived.append(chunk)

    store = ContextStore(comb=FakeComb(), comb_relevant_only=False)
    store.add_chunk(1, "password: hunter2-disguised-longvalue")
    chunk = store.all_chunks()[0]
    store._remove_chunk(chunk.id)
    assert len(archived) == 1
    assert "hunter2-disguised-longvalue" not in archived[0].content


# ---------------------------------------------------------------- config wire
def test_hive_config_exposes_hygiene_knobs():
    cfg = StrataConfig()
    assert cfg.strip_secrets is True
    assert cfg.max_chunk_chars == DEFAULT_MAX_CHUNK_CHARS
