"""A tier may pin the model it expects behind an endpoint.

Borrowed from LocalAI (#10952): a stopped backend's port can be recycled by a
different model's server, and a liveness-only probe cannot tell that apart from
a valid route.  ``expect_model`` makes the mismatch refuse the apply.
"""

import pytest

from harness.stack.manager import StackManager
from harness.stack.schema import Stack


def _stack(**tier):
    data = {"role": "face", "repo": "org/m", "file": "m.gguf",
            "endpoint": "http://127.0.0.1:8150/v1"}
    data.update(tier)
    return Stack.from_dict({"name": "t", "version": 1, "tiers": [data]})


def _manager(model):
    return StackManager(object(), endpoint_prober=lambda url: {
        "status": "ok", "loaded": True, "model": model})


def test_expected_model_matches():
    manager = _manager("deepseek-v4.1")
    out = manager.apply(_stack(expect_model="deepseek-v4.1"))
    assert out["tiers"][0]["role"] == "face"
    manager.unload()


def test_expected_model_mismatch_is_refused():
    manager = _manager("some-other-model")
    with pytest.raises(RuntimeError, match="reports model"):
        manager.apply(_stack(expect_model="deepseek-v4.1"))


def test_no_expectation_adopts_whatever_answers():
    manager = _manager("whatever")
    out = manager.apply(_stack())
    assert out["tiers"][0]["role"] == "face"
    manager.unload()


def test_expect_model_round_trips_through_the_stack_file():
    stack = _stack(expect_model="m")
    tier = stack.tiers[0]
    assert tier.to_dict()["expect_model"] == "m"
    again = Stack.from_dict({"name": "t", "version": 1, "tiers": [tier.to_dict()]})
    assert again.tiers[0].expect_model == "m"
