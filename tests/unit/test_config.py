"""Unit tests for cortex.config (StrataConfig)."""

from cortex.config import StrataConfig


def test_defaults_roundtrip(tmp_path):
    c = StrataConfig()
    path = tmp_path / "strata.json"
    c.save(path)
    loaded = StrataConfig.load(path)
    assert loaded == c


def test_from_dict_ignores_unknown_keys():
    c = StrataConfig.from_dict({"decay_multiplier_init": 2.2, "bogus": 1, "nope": None})
    assert c.decay_multiplier_init == 2.2
    assert not hasattr(c, "bogus")
    assert c.max_context == StrataConfig().max_context  # defaults preserved


def test_apply_gatekeeper_overrides():
    c = StrataConfig().apply_gatekeeper_overrides({"decay_multiplier_init": 2.5})
    assert c.decay_multiplier_init == 2.5
    assert c.max_context == StrataConfig().max_context


def test_apply_gatekeeper_overrides_ignores_none():
    c = StrataConfig().apply_gatekeeper_overrides({"decay_multiplier_init": None, "drift_threshold": 0.7})
    assert c.decay_multiplier_init == StrataConfig().decay_multiplier_init
    assert c.drift_threshold == 0.7


def test_to_dict_has_all_fields():
    d = StrataConfig().to_dict()
    assert d["decay_multiplier_init"] == 1.1
    assert "budget_ranges" in d