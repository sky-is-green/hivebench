"""Unit tests for cortex.interop (S3.5 Gatekeeper seam)."""

from cortex.interop import GatekeeperSeam


def test_normalize_lm_endpoint():
    seam = GatekeeperSeam()
    assert seam.normalize_lm_endpoint("") == "http://localhost:1234"
    assert seam.normalize_lm_endpoint("localhost") == "http://localhost"
    assert seam.normalize_lm_endpoint("http://x:8000") == "http://x:8000"


def test_calibrate_confidence():
    seam = GatekeeperSeam()
    # pulls predicted toward measured pass rate by half the gap
    assert seam.calibrate_confidence(0.9, 0.7) == 0.8
    assert seam.calibrate_confidence(0.5, None) == 0.5


def test_drone_reliability():
    seam = GatekeeperSeam()
    assert seam.drone_reliability(0.9, over_confidence=0.1) == 0.8
    assert seam.drone_reliability(0.2, over_confidence=0.5) == 0.0  # floored


def test_merge_config_precedence():
    seam = GatekeeperSeam()
    defaults = {"decay": 1.8, "threshold": 0.6, "only_default": True}
    overrides = {"decay": 2.2, "new": 1}
    merged = seam.merge_config(defaults, overrides)
    assert merged["decay"] == 2.2       # override wins
    assert merged["threshold"] == 0.6   # default preserved
    assert merged["only_default"] is True
    assert merged["new"] == 1           # new key added


def test_merge_config_ignores_none():
    seam = GatekeeperSeam()
    merged = seam.merge_config({"a": 1}, {"a": None, "b": 2})
    assert merged["a"] == 1  # None override ignored
    assert merged["b"] == 2