"""RC2 — sanitize both sides of the similarity compare.

Meta/tool control text must be stripped (ingest blocklist) and contractions
expanded (shared hedge map) on BOTH the query side and every candidate side
before similarity scoring; candidates with nothing scorable left are dropped
and never retrieved. The sidecar's payload echo-dedup fingerprints the same
normalized form the store persists, so sanitized/truncated chunks still match
their payload echo instead of escaping echo-dedup forever.
"""

import numpy as np

from cortex.routing import DroneRouter, EscalationHandler
from focal.assembly import ContextAssembler, normalize_for_scoring
from focal.budget import AdaptiveBudget
from membrane.dedup import ContextDeduplicator
from membrane.drift import TopicDriftDetector
from retention.hygiene import DEFAULT_INGEST_BLOCK_PREFIXES, strip_boilerplate
from retention.store import ContextChunk, ContextStore
from retention.hygiene import content_fingerprint, sanitize_for_storage
from sieve.scores import ChunkScore


META = DEFAULT_INGEST_BLOCK_PREFIXES[0]
REAL = "JWT auth uses 15-minute refresh tokens with rotation"


class FakeDrone:
    """Scores the meta control line highest on raw text — without the RC2
    strip pass it would win retrieval; with it, it never reaches scoring."""

    def __init__(self):
        self.seen_queries = []
        self.seen_texts = []

    def score(self, query, chunks):
        self.seen_queries.append(query)
        self.seen_texts.append(list(chunks))

        def relevance(c):
            if "enabled tools" in c:
                return 0.99
            if "JWT" in c:
                return 0.9
            return 0.3

        return [ChunkScore(i, relevance(c), 1.0) for i, c in enumerate(chunks)]

    def embed(self, text):
        # deterministic orthogonal-ish vectors so membrane dedup never
        # collapses the pool (str hash randomization makes hash() unusable)
        if "JWT" in text:
            return np.array([1.0, 0.0])
        if "redacted" in text or "api_key" in text:
            return np.array([0.0, 1.0])
        return np.array([0.5, 0.5])


class FakeMedium:
    def score(self, query, chunks):
        return [ChunkScore(i, 0.5, 0.85, source="medium") for i, _ in enumerate(chunks)]


def _deps(drone):
    return dict(
        router=DroneRouter(),
        ultra_small=drone,
        medium=FakeMedium(),
        escalation=EscalationHandler(),
        dedup=ContextDeduplicator(),
        drift_detector=TopicDriftDetector(embed_fn=lambda t: np.array([1.0, 0.0])),
        budget=AdaptiveBudget(),
        max_context=8192,
    )


def _polluted_store():
    """Legacy-style store: meta control text persisted as a chunk, as all
    pre-filter data did (inserted directly, bypassing the ingest filter the
    same way historical writes did)."""
    drone = FakeDrone()
    store = ContextStore(embed_fn=drone.embed)
    real_id = store.add_chunk(1, REAL)
    legacy = ContextChunk(
        id="legacy-meta", content=META, turn=1,
        fingerprint=content_fingerprint(META),
        timestamp="", last_referenced_turn=1,
    )
    store.chunks[legacy.id] = legacy
    store.turn_index.setdefault(1, []).append(legacy.id)
    return store, legacy.id, real_id


# ------------------------------------------------ normalize_for_scoring unit
def test_boilerplate_normalizes_to_empty():
    assert normalize_for_scoring(META) == ""
    assert normalize_for_scoring("   \n  ") == ""
    assert normalize_for_scoring("") == ""


def test_normal_content_passes_through():
    assert normalize_for_scoring(REAL) == REAL


def test_contraction_map_applies_to_both_sides_identically():
    assert normalize_for_scoring("I don't know the answer") == normalize_for_scoring(
        "I do not know the answer"
    )


def test_mixed_query_keeps_only_the_question():
    query = META + "\nHow does JWT auth work?"
    normalized = normalize_for_scoring(query)
    assert META[:80] not in normalized
    assert "How does JWT auth work?" in normalized


# ------------------------------------------------ assembly: meta never retrieved
def test_meta_candidate_dropped_before_scoring():
    store, meta_id, real_id = _polluted_store()
    drone = FakeDrone()
    result = ContextAssembler().assemble(
        query="How does JWT auth work?", current_turn=2, store=store, **_deps(drone)
    )
    # the meta chunk never reached the drone, won nothing, selected nothing
    for texts in drone.seen_texts:
        assert all("enabled tools" not in t for t in texts)
    assert meta_id not in result.raw_scores
    assert meta_id not in result.selected_chunk_ids
    assert real_id in result.selected_chunk_ids
    assert "JWT" in result.content


def test_query_side_boilerplate_stripped_before_scoring():
    store = ContextStore(embed_fn=FakeDrone().embed)
    store.add_chunk(1, REAL)
    drone = FakeDrone()
    ContextAssembler().assemble(
        query=META + "\nHow does JWT auth work?",
        current_turn=2,
        store=store,
        **_deps(drone),
    )
    assert drone.seen_queries, "drone must have scored"
    for q in drone.seen_queries:
        assert META[:80] not in q
        assert "How does JWT auth work?" in q


# ------------------------------------------------ sanitized echo still dedups
SECRET_REPLY = "The deploy token is api_key=supersecretvalue12345 with rotation"


def _payload_fingerprints_like_app(store, messages):
    """The exact recipe harness/harness/app.py uses for payload fingerprints."""
    fps = set()
    for text in messages:
        normalized = strip_boilerplate(text, store.ingest_block_prefixes)
        if not normalized or not normalized.strip():
            continue
        normalized = sanitize_for_storage(normalized, store.max_chunk_chars)
        fps.add(content_fingerprint(normalized))
    return fps


def test_sanitized_chunk_matches_its_raw_payload_echo():
    store = ContextStore(embed_fn=FakeDrone().embed)
    secret_id = store.add_chunk(1, SECRET_REPLY)
    novel_id = store.add_chunk(1, "runbook fact: rollback needs the on-call token")
    # the store persists the sanitized form, so a raw fingerprint can never
    # match it — this is how sanitized chunks escaped echo-dedup forever
    assert content_fingerprint(SECRET_REPLY) != store.chunks[secret_id].fingerprint

    drone = FakeDrone()
    result = ContextAssembler().assemble(
        query="what is the rollback runbook?",
        current_turn=2,
        store=store,
        payload_fingerprints=_payload_fingerprints_like_app(
            store, [SECRET_REPLY, "what is the rollback runbook?"]
        ),
        **_deps(drone),
    )
    assert result.payload_dedup_skipped == 1
    assert secret_id not in result.selected_chunk_ids
    assert novel_id in result.selected_chunk_ids


# ------------------------------------------------ sidecar: app.py echo-dedup live
def _openai_client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from backend.openai_compat import OpenAICompatBackend
    from cortex.e2e import FakeUltraSmall, MockTransport
    from harness.app import create_app

    monkeypatch.chdir(tmp_path)

    def backend_factory(model=None, provider=None):
        return OpenAICompatBackend(
            base_url="http://mock", model=model or "mock-model",
            transport=MockTransport(latency_ms=0),
        )

    app = create_app(
        ultra_factory=FakeUltraSmall,
        backend_factory=backend_factory,
        runs_root=tmp_path / "runs",
        providers_file=tmp_path / "providers.local.json",
        log_dir=str(tmp_path / "logs"),
    )
    return TestClient(app), app


class _FakeUpstream:
    """Non-streaming upstream stand-in returning a fixed secret-bearing reply."""

    def __init__(self, content):
        self.content = content
        self.payload = None

    def __call__(self, url, json=None, headers=None, stream=False, timeout=None):
        self.payload = json
        return self

    def raise_for_status(self):
        pass

    def json(self):
        return {
            "id": "up-1", "object": "chat.completion", "created": 1,
            "model": "m1",
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": self.content},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 6},
        }


def test_openai_passthrough_skips_sanitized_echo(tmp_path, monkeypatch):
    client, _app = _openai_client(tmp_path, monkeypatch)
    r = client.post("/v1/provider/config", json={
        "providers": [{"name": "lm", "base_url": "http://mock-llama",
                       "api_key": "lm-studio", "model": "m1"}],
    })
    assert r.status_code == 200, r.text

    import harness.app as appmod

    fake = _FakeUpstream(SECRET_REPLY)
    monkeypatch.setattr(appmod, "_upstream_stream", fake)

    # turn 1: the query itself is pure harness boilerplate, so the ingest
    # filter stores nothing for it — the ONLY chunk persisted is the observed
    # secret-bearing reply, in sanitized form. A raw fingerprint of the reply
    # can therefore never match the stored chunk (this is how sanitized
    # chunks escaped echo-dedup forever under the old code).
    assert sanitize_for_storage(SECRET_REPLY) != SECRET_REPLY
    r = client.post("/v1/openai/chat/completions", json={
        "model": "m1",
        "messages": [{"role": "user", "content": META}],
    })
    assert r.status_code == 200, r.text
    st = client.get("/v1/strata/state", params={"conversation_id": "default"}).json()
    assert st["store_chunks"] == 1

    # turn 2: the raw secret-bearing reply re-enters as an assistant message
    # (Studio proxies the full thread history every turn). System-first, as
    # real proxied payloads are: echo-dedup fingerprints what is forwarded.
    r = client.post("/v1/openai/chat/completions", json={
        "model": "m1",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": SECRET_REPLY},
            {"role": "user", "content": "and what about rotation?"},
        ],
    })
    assert r.status_code == 200, r.text

    inspect = client.get("/v1/strata/inspect/default").json()
    # the sanitized chunk matches its raw echo via the normalized fingerprint;
    # the old raw-fingerprint code skipped nothing here
    assert inspect["payload_dedup_skipped"] == 1


# ------------------------------------------------ pre-trim payload fingerprints
TRIMMED_FACT = (
    "FACT: the JWT signing key rotates at canary ROTATE-42; "
    "runbook lives at /opt/runbook.md"
)


def _long_thread_padding(n=400):
    return [
        {"role": "assistant", "content": "filler " * 40 + str(i)}
        for i in range(n)
    ]


def _store_only_fact_via_boilerplate_turn(client):
    """Turn 1 with a boilerplate-only query stores nothing for the query, so
    the observed reply is the store's only chunk."""
    r = client.post("/v1/openai/chat/completions", json={
        "model": "m1",
        "messages": [{"role": "user", "content": META}],
    })
    assert r.status_code == 200, r.text
    st = client.get("/v1/strata/state", params={"conversation_id": "default"}).json()
    assert st["store_chunks"] == 1


def test_openai_long_thread_trim_keeps_fact_in_curation(tmp_path, monkeypatch):
    """Oversized-thread bug: payload fingerprints were computed over ALL
    messages, then the forwarded body was trimmed to the last 8 — an early
    fact was skipped from curation AND absent from the forwarded body, so the
    model saw it nowhere. Fingerprints must cover only what is forwarded."""
    client, _app = _openai_client(tmp_path, monkeypatch)
    r = client.post("/v1/provider/config", json={
        "providers": [{"name": "lm", "base_url": "http://mock-llama",
                       "api_key": "lm-studio", "model": "m1"}],
    })
    assert r.status_code == 200, r.text

    import harness.app as appmod

    fake = _FakeUpstream(TRIMMED_FACT)
    monkeypatch.setattr(appmod, "_upstream_stream", fake)
    _store_only_fact_via_boilerplate_turn(client)

    messages = (
        [{"role": "system", "content": "sys"},
         {"role": "assistant", "content": TRIMMED_FACT}]
        + _long_thread_padding()
        + [{"role": "user", "content": "what is the JWT rotation token?"}]
    )
    assert sum(len(m["content"]) for m in messages[1:]) > 60_000
    r = client.post("/v1/openai/chat/completions",
                    json={"model": "m1", "messages": messages})
    assert r.status_code == 200, r.text

    forwarded = fake.payload["messages"]
    assert len(forwarded) - 1 == 8  # system + trimmed body
    assert all(m.get("content") != TRIMMED_FACT for m in forwarded)

    inspect = client.get("/v1/strata/inspect/default").json()
    # the fact was trimmed away, so it must still be curated (not echo-skipped)
    assert inspect["payload_dedup_skipped"] == 0
    assert "ROTATE-42" in inspect["assembled_preview"]


def test_openai_short_thread_echo_skip_unchanged(tmp_path, monkeypatch):
    """Regression: on a short thread the fact is both fingerprinted and
    forwarded, so recency-echo dedup still drops it from curation."""
    client, _app = _openai_client(tmp_path, monkeypatch)
    r = client.post("/v1/provider/config", json={
        "providers": [{"name": "lm", "base_url": "http://mock-llama",
                       "api_key": "lm-studio", "model": "m1"}],
    })
    assert r.status_code == 200, r.text

    import harness.app as appmod

    fake = _FakeUpstream(TRIMMED_FACT)
    monkeypatch.setattr(appmod, "_upstream_stream", fake)
    _store_only_fact_via_boilerplate_turn(client)

    r = client.post("/v1/openai/chat/completions", json={
        "model": "m1",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": TRIMMED_FACT},
            {"role": "user", "content": "what is the JWT rotation token?"},
        ],
    })
    assert r.status_code == 200, r.text

    inspect = client.get("/v1/strata/inspect/default").json()
    assert inspect["payload_dedup_skipped"] == 1
    assert "ROTATE-42" not in inspect["assembled_preview"]
