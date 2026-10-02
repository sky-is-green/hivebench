"""Unit tests for the cascade latency model and path planning."""

from __future__ import annotations

import pytest

from harness.cascade import scheduler as sched
from harness.cascade.paths import get_path
from harness.cascade.roles import Device
from harness.cascade.scheduler import (
    LatencyModel,
    Plan,
    RoleCost,
    RouteDecision,
    estimate_path,
    plan_request,
)


def test_fallback_uses_the_brief_midpoint():
    model = LatencyModel()
    assert model.cost_ms("A1") == pytest.approx(5.5)          # (1 + 10) / 2
    assert model.cost_ms("E1", out_len=10) == pytest.approx(55.0)  # token role
    assert not model.is_measured("A1")


def test_measured_cost_overrides_and_scales_with_tokens():
    model = LatencyModel()
    model.add(RoleCost("E1", Device.DGPU0, 6.0, per="token"))
    assert model.is_measured("E1")
    assert model.cost_ms("E1", out_len=2) == pytest.approx(12.0)
    assert model.cost_ms("E1") == pytest.approx(0.0)


def test_pick_device_prefers_measured_then_declared_order():
    model = LatencyModel()
    assert model.pick_device("C1") is Device.IGPU
    model.add(RoleCost("C1", Device.DGPU0, 40.0))
    assert model.pick_device("C1") is Device.DGPU0
    assert model.pick_device("C1", Device.IGPU) is Device.IGPU


def test_estimate_path_expands_extends_and_flags_async():
    steps = estimate_path(get_path("P3"), LatencyModel(), out_len=10)
    assert [s.role for s in steps] == ["C1", "C2", "B1", "B2", "E2", "D1", "D2", "E3", "D3"]
    assert [s.role for s in steps if s.async_] == ["C1", "D1", "D3"]
    e2 = next(s for s in steps if s.role == "E2")
    assert e2.ms == pytest.approx(140.0)  # 14 ms/token * 10
    assert e2.device is Device.DGPU0


def test_plan_prices_p2_and_its_escalation():
    plan = plan_request(RouteDecision("qa", 0.9, request_id="r1"), out_len=10)
    assert isinstance(plan, Plan)
    assert plan.path == "P2"
    assert plan.blocking_ms == pytest.approx(260.0)
    assert plan.async_ms == pytest.approx(140.0)
    assert plan.escalation_path == "P3"
    assert plan.escalation_probability == pytest.approx(0.1)
    assert plan.escalation_ms == pytest.approx(410.0)  # P3's D3 runs async
    assert plan.total_expected_ms == pytest.approx(301.0)
    assert plan.request_id == "r1"
    assert plan.expected_ms == plan.blocking_ms


def test_plan_confidence_controls_escalation_weight():
    timid = plan_request(RouteDecision("qa", 0.0), out_len=10)
    sure = plan_request(RouteDecision("qa", 1.0), out_len=10)
    assert timid.escalation_probability == 1.0
    assert sure.escalation_probability == 0.0
    assert timid.total_expected_ms > sure.total_expected_ms


def test_unknown_bucket_falls_back_to_standard_qa():
    plan = plan_request(RouteDecision("unmapped-bucket", 1.0))
    assert plan.path == "P2"


def test_quality_provider_filters_paths(monkeypatch):
    monkeypatch.setattr(
        sched, "paths_for_bucket", lambda bucket: (get_path("P1"), get_path("P2"))
    )
    fast = plan_request(RouteDecision("qa", 1.0), out_len=10)
    assert fast.path == "P1"  # cheapest with no quality signal
    quality = {"P1": 0.5, "P2": 0.95}
    good = plan_request(RouteDecision("qa", 1.0), out_len=10, quality=quality.get)
    assert good.path == "P2"
    assert good.quality == 0.95


def test_quality_provider_that_filters_everything_keeps_the_route_path(monkeypatch):
    monkeypatch.setattr(
        sched, "paths_for_bucket", lambda bucket: (get_path("P1"), get_path("P2"))
    )
    plan = plan_request(RouteDecision("qa", 1.0), out_len=10, quality=lambda pid: 0.1)
    assert plan.path == "P1"


def test_plan_round_trips():
    data = plan_request(RouteDecision("easy", 0.8), out_len=4).to_dict()
    assert data["path"] == "P1"
    assert data["escalation_path"] == "P3"
    assert "total_expected_ms" in data
    assert data["steps"][0]["role"] == "C1"
    assert data["steps"][0]["async"] is True
