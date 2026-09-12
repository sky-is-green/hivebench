"""backend.sampling — the experimenter sampling surface."""

import pytest

from backend.sampling import (
    SAMPLING_FIELDS,
    merge_sampling,
    parse_sampling,
    sampling_fingerprint,
)


def test_parse_sampling_json_string_and_dict():
    assert parse_sampling('{"temperature": 0.7, "top_p": 0.9}') == {
        "temperature": 0.7, "top_p": 0.9,
    }
    assert parse_sampling({"temperature": 0.7}) == {"temperature": 0.7}
    assert parse_sampling(None) == {}
    assert parse_sampling("") == {}


def test_parse_sampling_drops_unknown_keys():
    out = parse_sampling('{"temperature": 0.7, "bogus_knob": 9}')
    assert out == {"temperature": 0.7}


def test_parse_sampling_rejects_non_object():
    with pytest.raises(ValueError):
        parse_sampling("[1, 2, 3]")


def test_known_fields_are_the_openai_compat_surface():
    for k in ("temperature", "top_p", "top_k", "min_p", "repeat_penalty",
              "presence_penalty", "frequency_penalty", "stop", "seed",
              "mirostat", "mirostat_tau", "mirostat_eta"):
        assert k in SAMPLING_FIELDS


def test_merge_sampling_overrides_win():
    assert merge_sampling({"temperature": 0.7, "top_p": 0.9},
                          {"temperature": 0.2}) == {
        "temperature": 0.2, "top_p": 0.9,
    }
    assert merge_sampling({"temperature": 0.7}, None) == {"temperature": 0.7}


def test_fingerprint_is_stable_and_ordered():
    a = sampling_fingerprint({"top_p": 0.9, "temperature": 0.7})
    b = sampling_fingerprint({"temperature": 0.7, "top_p": 0.9})
    assert a == b
    assert sampling_fingerprint({}) == "default"