"""Unit tests for harness.cascade roles, registry and the oracle fronts."""

from __future__ import annotations

import pytest

from harness.cascade.oracle import (
    CandidateMeasurement,
    best_per_candidate,
    dominates,
    frontier_by_role,
    pareto_frontier,
)
from harness.cascade.registry import (
    DEFAULT_CANDIDATES,
    Candidate,
    CandidateRegistry,
    default_registry,
    select_resident_set,
    with_measurement,
)
from harness.cascade.roles import (
    ROLES,
    Device,
    Kind,
    Role,
    get_role,
    role_ids,
    roles_for_kind,
    roles_on,
    validate_roles,
)


def test_taxonomy_is_structurally_valid():
    assert validate_roles() == []
    assert len(ROLES) == len(role_ids())


def test_role_ids_are_unique_and_lookupable():
    assert len(set(role_ids())) == len(role_ids())
    assert get_role("C1").kind == Kind.ROUTE
    with pytest.raises(KeyError):
        get_role("Z9")


def test_role_budget_shape():
    role = get_role("E1")
    assert role.devices == (Device.DGPU0,)
    assert role.latency_unit == "token"
    assert role.context_cap == 32768
    assert role.kv != "none"


def test_kind_and_device_queries():
    assert {r.id for r in roles_for_kind(Kind.GENERATE)} == {"E1", "E2", "E3", "E4"}
    assert "A1" in {r.id for r in roles_on(Device.CPU)}
    assert "E4" not in {r.id for r in roles_on(Device.IGPU)}
    assert get_role("G1").candidates == ()


def test_validate_roles_flags_duplicates_and_empty_devices():
    bad = (
        get_role("A1"),
        get_role("A1"),
        Role(
            id="X1", name="broken", kind=Kind.EMBED, input_kind="a",
            output_kind="b", devices=(), latency_ms=(1.0, 2.0),
        ),
    )
    problems = validate_roles(bad)
    assert any("duplicate" in p for p in problems)
    assert any("device tier" in p for p in problems)


def test_default_catalog_roles_exist_and_ids_unique():
    known = set(role_ids())
    for candidate in DEFAULT_CANDIDATES:
        assert candidate.roles, candidate.id
        assert set(candidate.roles) <= known, candidate.id
    registry = default_registry()
    assert len(registry) == len(DEFAULT_CANDIDATES)


def test_registry_rejects_duplicate_ids():
    registry = CandidateRegistry()
    registry.register(Candidate(id="m", label="m", repo="r/x"))
    with pytest.raises(ValueError):
        registry.register(Candidate(id="m", label="other", repo="r/y"))


def test_registry_role_and_device_lookups():
    registry = CandidateRegistry(
        [
            Candidate(id="enc", label="enc", repo="r/enc", devices=(Device.CPU,), roles=("A1",)),
            Candidate(id="dec", label="dec", repo="r/dec", devices=(Device.DGPU0,), roles=("E1",)),
        ]
    )
    assert [c.id for c in registry.for_role("A1")] == ["enc"]
    assert [c.id for c in registry.for_device(Device.DGPU0)] == ["dec"]
    assert registry.missing_roles(["A1", "E1", "E4"]) == ("E4",)
    assert registry.coverage(["A1"])["A1"][0].id == "enc"


def test_select_resident_set_covers_roles_within_capacity():
    registry = CandidateRegistry(
        [
            Candidate(id="enc", label="enc", repo="r/enc", bytes_gb=0.1,
                      devices=(Device.CPU,), roles=("A1", "B1")),
            Candidate(id="dec", label="dec", repo="r/dec", bytes_gb=4.0,
                      devices=(Device.DGPU0,), roles=("E1",)),
            Candidate(id="big", label="big", repo="r/big", bytes_gb=20.0,
                      devices=(Device.DGPU0,), roles=("E1", "E4")),
        ]
    )
    picked = select_resident_set(
        registry, ["A1", "B1", "E1"],
        capacity_gb={Device.CPU: 2.0, Device.DGPU0: 10.0},
    )
    assert picked.ok
    assert {c.id for c in picked.candidates} == {"enc", "dec"}
    assert picked.bytes_by_device["dgpu0"] == 4.0
    assert set(picked.roles_covered()) == {"A1", "B1", "E1"}
    assert picked.uncovered == ()


def test_select_resident_set_reports_uncovered_and_unknown_sizes():
    registry = CandidateRegistry(
        [
            Candidate(id="big", label="big", repo="r/big", bytes_gb=20.0,
                      devices=(Device.DGPU0,), roles=("E1",)),
            Candidate(id="unsized", label="unsized", repo="r/u", bytes_gb=None,
                      devices=(Device.CPU,), roles=("A1",)),
        ]
    )
    picked = select_resident_set(
        registry, ["E1", "A1"],
        capacity_gb={Device.DGPU0: 10.0, Device.CPU: 1.0},
    )
    assert picked.uncovered == ("E1",)
    assert picked.unknown_sizes == ("unsized",)
    assert [c.id for c in picked.candidates] == ["unsized"]


def test_select_resident_set_prefers_requested_candidate():
    registry = CandidateRegistry(
        [
            Candidate(id="first", label="first", repo="r/a", bytes_gb=1.0,
                      devices=(Device.CPU,), roles=("A1",)),
            Candidate(id="second", label="second", repo="r/b", bytes_gb=1.5,
                      devices=(Device.CPU,), roles=("A1",)),
        ]
    )
    picked = select_resident_set(
        registry, ["A1"], capacity_gb={Device.CPU: 2.0}, prefer=("second",)
    )
    assert [c.id for c in picked.candidates] == ["second"]


def test_select_resident_set_rejects_unknown_role():
    with pytest.raises(KeyError):
        select_resident_set(default_registry(), ["Z9"], capacity_gb={Device.CPU: 1.0})


def test_with_measurement_refines_size_and_precision():
    candidate = Candidate(id="m", label="m", repo="r/m")
    refined = with_measurement(candidate, bytes_gb=3.5, precision="int4")
    assert refined.bytes_gb == 3.5
    assert refined.precision == "int4"
    assert with_measurement(candidate) is candidate


def _m(candidate: str, quality: float, latency: float, role: str = "C4",
       device: Device = Device.CPU) -> CandidateMeasurement:
    return CandidateMeasurement(
        candidate=candidate, role=role, device=device,
        quality=quality, latency_ms=latency,
    )


def test_dominates_and_frontier():
    a = _m("a", 0.9, 10.0)
    b = _m("b", 0.8, 12.0)
    c = _m("c", 0.95, 15.0)
    assert dominates(a, b)
    assert not dominates(a, c)
    front = pareto_frontier([a, b, c])
    assert {row.candidate for row in front} == {"a", "c"}


def test_frontier_keeps_ties_and_groups_by_role():
    a = _m("a", 0.9, 10.0)
    b = _m("b", 0.9, 10.0)
    assert len(pareto_frontier([a, b])) == 2
    groups = frontier_by_role(
        [
            _m("e1", 0.5, 4.0, role="E1"),
            _m("e2", 0.6, 9.0, role="E2"),
            _m("e1b", 0.55, 5.0, role="E1"),
        ]
    )
    assert set(groups) == {"E1", "E2"}
    assert {row.candidate for row in groups["E1"]} == {"e1", "e1b"}


def test_best_per_candidate_tie_breaks_on_latency():
    best = best_per_candidate([_m("a", 0.9, 12.0), _m("a", 0.9, 10.0), _m("b", 0.5, 1.0)])
    assert best["a"].latency_ms == 10.0
    assert best["b"].quality == 0.5
