"""Unit tests for focal.predictive (S5.2)."""

from focal.predictive import PredictivePreloader
from retention.store import ContextStore


def _store():
    s = ContextStore()
    err = s.add_chunk(1, "the traceback shows a runtime error in the login service")
    s.add_chunk(2, "we watered the plants this morning")
    return s, err


def test_debugging_preloads_error_chunks():
    store, err_id = _store()
    pre = PredictivePreloader()
    pred = pre.predict_next_context(["debug the traceback error", "why is it failing"], store)
    assert err_id in pred


def test_empty_queries_no_patterns():
    store, _ = _store()
    assert PredictivePreloader().predict_next_context([], store) == []


def test_alternating_topics_preloads_next_topic_chunks():
    store = ContextStore()
    auth_ids = [store.add_chunk(i, f"JWT token authentication schema {i}") for i in range(1, 6)]
    garden_ids = [store.add_chunk(i + 10, f"plant watering garden tips {i}") for i in range(5)]
    pre = PredictivePreloader()
    queries = [
        "fix the JWT token bug",
        "water the tomato plants",
        "debug the auth schema error",
        "plant the new roses",   # next topic: gardening
    ]
    pred = pre.predict_next_context(queries, store)
    garden_pred = set(garden_ids) & set(pred)
    # at least 60% of the next topic's chunks are pre-loaded
    assert len(garden_pred) / len(garden_ids) >= 0.6
    assert len(pred) == len(set(pred))  # no duplicates


def test_dedup_preserves_order():
    store = ContextStore()
    cid = store.add_chunk(1, "error bug traceback")  # matches debugging twice
    pre = PredictivePreloader()
    pred = pre.predict_next_context(["debug error", "traceback fail"], store)
    assert pred.count(cid) == 1